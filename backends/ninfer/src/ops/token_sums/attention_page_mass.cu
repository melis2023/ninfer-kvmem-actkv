#include "ninfer/ops/attention_page_mass.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
__device__ std::int64_t address(int page,int head,int offset,int dimension,int heads,int width) {
    return ((std::int64_t(page)*heads+head)*64+offset)*width+dimension;
}
template<bool Int8Keys>
__device__ float key_value(const void* keys,const void* scales,KvCacheStorage storage,
                           int page,int head,int offset,int d,int heads) {
    if constexpr(Int8Keys) {
        return float(static_cast<const std::int8_t*>(keys)[address(page,head,offset,d,heads,256)])*
            __half2float(static_cast<const half*>(scales)[address(page,head,offset,d/64,heads,4)]);
    }
    if(storage==KvCacheStorage::BFloat16)
        return __bfloat162float(static_cast<const __nv_bfloat16*>(keys)[address(page,head,offset,d,heads,256)]);
    if(storage==KvCacheStorage::Nvfp4Group16) {
        const auto packed=static_cast<const std::uint8_t*>(keys)[address(page,head,offset,d/2,heads,128)];
        const auto scale=static_cast<const std::uint8_t*>(scales)[address(page,head,offset,d/16,heads,16)];
        const auto value=detail::decode_nvfp4_e2m1x2(packed);
        return (d&1 ? value.y : value.x)*detail::decode_nvfp4_e4m3(scale);
    }
    if(storage==KvCacheStorage::Fp8E4M3Row256 || storage==KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto code=static_cast<const std::uint8_t*>(keys)[address(page,head,offset,d,heads,256)];
        const auto scale=static_cast<const half*>(scales)[address(page,head,offset,0,heads,1)];
        return detail::decode_nvfp4_e4m3(code)*__half2float(scale);
    }
    int code;
    if(storage==KvCacheStorage::RotatedLloyd4KeyInt4Value ||
       storage==KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
        const auto packed=static_cast<const std::uint8_t*>(keys)[address(page,head,offset,d/2,heads,128)];
        code=(packed>>((d&1)*4))&15;
        code=storage==KvCacheStorage::RotatedLloyd4KeyInt4Value ? kv_cache_lloyd4_code(code) : (code^8)-8;
    } else if(storage==KvCacheStorage::RotatedE8RootKeyInt4Value) {
        const auto index=address(page,head,offset,(d&~15)/4,heads,64);
        const auto packed=*reinterpret_cast<const std::uint32_t*>(static_cast<const std::uint8_t*>(keys)+index);
        const int4 decoded=kv_cache_e8_root_unpack_i8x16(packed);
        code=reinterpret_cast<const signed char*>(&decoded)[d&15];
    } else {
        code=static_cast<const std::int8_t*>(keys)[address(page,head,offset,d,heads,256)];
    }
    return float(code)*__half2float(static_cast<const half*>(scales)[address(page,head,offset,d/64,heads,4)]);
}
template<bool Int8Keys, bool Nvfp4Keys=false>
__global__ void page_lse(const __nv_bfloat16* q,const int* rows,const int* positions,
    const int* table,const void* keys,const void* scales,KvCacheStorage storage,int qheads,
    int kvheads,int pages,int physical_pages,float* output) {
    __shared__ float values[128];
    const int page=int(blockIdx.x),distribution=int(blockIdx.y);
    const int sample=distribution/qheads,head=distribution%qheads;
    const int tid=int(threadIdx.x),lane=tid&31,warp=tid/32;
    const int visible=positions[sample]+1;
    if(page*64>=visible) { if(tid==0) output[size_t(distribution)*pages+page]=-CUDART_INF_F; return; }
    float query[8];
    for(int i=0;i<8;++i) query[i]=__bfloat162float(q[(size_t(rows[sample])*qheads+head)*256+lane+32*i]);
    if constexpr(Int8Keys || Nvfp4Keys) normalized_hadamard_d256_inplace(query,lane);
    else if(storage!=KvCacheStorage::BFloat16) normalized_hadamard_d256_inplace(query,lane);
    if constexpr(Nvfp4Keys) {
        // From d=lane+32*i to pairs d=2*lane+64*i. Do the register transpose
        // once per Q, outside the token loop; keep the represented FP32 rotation.
#pragma unroll
        for(int i=0;i<4;++i) {
            const int source=2*(lane&15);
            const float low_even=__shfl_sync(0xffffffff,query[2*i],source);
            const float high_even=__shfl_sync(0xffffffff,query[2*i+1],source);
            const float low_odd=__shfl_sync(0xffffffff,query[2*i],source+1);
            const float high_odd=__shfl_sync(0xffffffff,query[2*i+1],source+1);
            query[2*i]=lane<16 ? low_even:high_even;
            query[2*i+1]=lane<16 ? low_odd:high_odd;
        }
    }
    const int physical=table[page],kvhead=head/(qheads/kvheads);
    if(physical<0 || physical>=physical_pages) {
        if(tid==0) output[size_t(distribution)*pages+page]=CUDART_NAN_F;
        return;
    }
    for(int offset=warp;offset<64;offset+=4) {
        float dot=0;
        if(page*64+offset<visible && physical>=0) {
            if constexpr(Nvfp4Keys) {
                const auto* codes=static_cast<const std::uint8_t*>(keys)+address(physical,kvhead,offset,0,kvheads,128);
                const auto* groups=static_cast<const std::uint8_t*>(scales)+address(physical,kvhead,offset,0,kvheads,16);
                // Sixteen G16 scales per K row, decoded once each and shared by
                // the eight lanes consuming that group. Each packed byte is
                // decoded once, and both E2M1 values contribute to the FP32 dot.
                const float decoded_scale=lane<16 ? detail::decode_nvfp4_e4m3(groups[lane]):0.f;
#pragma unroll
                for(int i=0;i<4;++i) {
                    const float scale=__shfl_sync(0xffffffff,decoded_scale,4*i+lane/8);
                    const auto pair=detail::decode_nvfp4_e2m1x2(codes[lane+32*i]);
                    dot=fmaf(query[2*i],pair.x*scale,dot);
                    dot=fmaf(query[2*i+1],pair.y*scale,dot);
                }
            } else {
                for(int i=0;i<8;++i) dot=fmaf(query[i],key_value<Int8Keys>(keys,scales,storage,physical,kvhead,offset,lane+32*i,kvheads),dot);
            }
            for(int s=16;s;s/=2) dot+=__shfl_down_sync(0xffffffff,dot,s);
        }
        if(lane==0) values[offset]=page*64+offset<visible && physical>=0 ? dot/16 : -CUDART_INF_F;
    }
    __syncthreads();
    float value=tid<64 ? values[tid] : -CUDART_INF_F;
    values[tid]=value; __syncthreads();
    for(int s=64;s;s/=2) { if(tid<s) values[tid]=fmaxf(values[tid],values[tid+s]); __syncthreads(); }
    const float maximum=values[0];
    __syncthreads(); // Every warp must read the maximum before values[0] becomes a sum.
    value=isfinite(value) ? expf(value-maximum) : 0;
    values[tid]=value; __syncthreads();
    for(int s=64;s;s/=2) { if(tid<s) values[tid]+=values[tid+s]; __syncthreads(); }
    if(tid==0) output[size_t(distribution)*pages+page]=values[0]>0 ? maximum+logf(values[0]) : -CUDART_INF_F;
}
__global__ void normalize_lse(const float* lse,int pages,float* norms) {
    __shared__ float sums[128];
    const int tid=int(threadIdx.x),row=int(blockIdx.x);
    float maximum=-CUDART_INF_F;
    for(int p=tid;p<pages;p+=128) maximum=fmaxf(maximum,lse[size_t(row)*pages+p]);
    sums[tid]=maximum; __syncthreads();
    for(int s=64;s;s/=2) { if(tid<s) sums[tid]=fmaxf(sums[tid],sums[tid+s]); __syncthreads(); }
    maximum=sums[0];
    __syncthreads();
    float total=0;
    for(int p=tid;p<pages;p+=128) total+=expf(lse[size_t(row)*pages+p]-maximum);
    sums[tid]=total; __syncthreads();
    for(int s=64;s;s/=2) { if(tid<s) sums[tid]+=sums[tid+s]; __syncthreads(); }
    if(tid==0) { norms[2*row]=maximum; norms[2*row+1]=1/sums[0]; }
}
__global__ void average_heads(const float* lse,const float* norms,int pages,int heads,float* output) {
    const int page=int(blockIdx.x)*128+int(threadIdx.x),sample=int(blockIdx.y);
    if(page>=pages) return;
    float mass=0;
    for(int h=0;h<heads;++h) {
        const int row=sample*heads+h;
        mass+=expf(lse[size_t(row)*pages+page]-norms[2*row])*norms[2*row+1];
    }
    output[size_t(sample)*pages+page]=mass/heads;
}
} // namespace
void attention_page_mass(const Tensor& q,const Tensor& rows,const Tensor& positions,int table_row,
    const PagedKVBatchLayerView& cache,Tensor& scratch,Tensor& norms,Tensor& output,cudaStream_t stream) {
    const auto samples=rows.numel();
    const auto profile=d256_kv_cache_profile(cache.storage);
    if(!q.data || q.dtype!=DType::BF16 || !q.is_contiguous() || q.ne[0]!=256 || q.ne[3]!=1 ||
       !rows.data || !positions.data || rows.dtype!=DType::I32 || positions.dtype!=DType::I32 ||
       !rows.is_contiguous() || !positions.is_contiguous() || samples!=positions.numel() || samples<=0 || samples>64 ||
       cache.head_dim!=256 || cache.num_kv_heads<=0 || q.ne[1]%cache.num_kv_heads ||
       !cache.block_tables.data || cache.block_tables.dtype!=DType::I32 || !cache.block_tables.is_contiguous() ||
       !cache.k_pages.data || !cache.k_pages.is_contiguous() || cache.k_pages.dtype!=profile.key_code_dtype ||
       cache.k_pages.ne[0]!=profile.key_leading_extent || cache.k_pages.ne[1]!=64 ||
       cache.k_pages.ne[2]!=cache.num_kv_heads || cache.k_pages.ne[3]<=0 ||
       table_row<0 || table_row>=cache.block_tables.ne[1] || output.ne[0]<=0 || output.ne[1]!=samples ||
       output.ne[0]>cache.block_tables.ne[0] || !scratch.data || !norms.data || !output.data ||
       scratch.ne[0]!=output.ne[0] || scratch.ne[1]!=q.ne[1] || scratch.ne[2]!=samples ||
       norms.numel()!=2*samples*q.ne[1] || scratch.dtype!=DType::FP32 || norms.dtype!=DType::FP32 ||
       output.dtype!=DType::FP32 || !scratch.is_contiguous() || !norms.is_contiguous() || !output.is_contiguous())
        throw std::invalid_argument("attention_page_mass: invalid sample geometry");
    if(cache.storage!=KvCacheStorage::BFloat16 &&
       (!cache.k_scale_pages.data || !cache.k_scale_pages.is_contiguous() ||
        cache.k_scale_pages.dtype!=profile.key_scale_dtype || cache.k_scale_pages.ne[0]!=profile.scale_leading_extent ||
        cache.k_scale_pages.ne[1]!=64 || cache.k_scale_pages.ne[2]!=cache.num_kv_heads ||
        cache.k_scale_pages.ne[3]!=cache.k_pages.ne[3]))
        throw std::invalid_argument("attention_page_mass: invalid native K scales");
    const int pages=output.ne[0],heads=q.ne[1];
    const auto blocks=std::uint64_t(pages)*heads*samples;
    if(blocks>INT32_MAX) throw std::length_error("attention_page_mass: sample grid exceeds addressing");
    const auto* tables=static_cast<const int*>(cache.block_tables.data)+size_t(table_row)*cache.block_tables.ne[0];
    // Dispatch formats outside the dot-product loop. INT8/rk8v4 share K encoding;
    // NVFP4 uses paired codes and warp-shared G16 scales. Other formats stay generic.
    if(cache.storage==KvCacheStorage::Int8Group64 ||
       cache.storage==KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        page_lse<true><<<dim3(unsigned(pages),unsigned(samples*heads)),128,0,stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),static_cast<const int*>(rows.data),static_cast<const int*>(positions.data),
            tables,cache.k_pages.data,cache.k_scale_pages.data,cache.storage,heads,cache.num_kv_heads,pages,
            cache.k_pages.ne[3],static_cast<float*>(scratch.data));
    } else if(cache.storage==KvCacheStorage::Nvfp4Group16) {
        page_lse<false,true><<<dim3(unsigned(pages),unsigned(samples*heads)),128,0,stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),static_cast<const int*>(rows.data),static_cast<const int*>(positions.data),
            tables,cache.k_pages.data,cache.k_scale_pages.data,cache.storage,heads,cache.num_kv_heads,pages,
            cache.k_pages.ne[3],static_cast<float*>(scratch.data));
    } else {
        page_lse<false><<<dim3(unsigned(pages),unsigned(samples*heads)),128,0,stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),static_cast<const int*>(rows.data),static_cast<const int*>(positions.data),
        tables,cache.k_pages.data,cache.k_scale_pages.data,cache.storage,heads,cache.num_kv_heads,pages,
        cache.k_pages.ne[3],static_cast<float*>(scratch.data));
    }
    normalize_lse<<<unsigned(samples*heads),128,0,stream>>>(static_cast<const float*>(scratch.data),pages,static_cast<float*>(norms.data));
    average_heads<<<dim3(unsigned((pages+127)/128),unsigned(samples)),128,0,stream>>>(
        static_cast<const float*>(scratch.data),static_cast<const float*>(norms.data),pages,heads,static_cast<float*>(output.data));
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(std::string("attention_page_mass: ")+cudaGetErrorString(error));
}
} // namespace ninfer::ops
