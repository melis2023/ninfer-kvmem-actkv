#include "ninfer/ops/attention_page_mass.h"
#include "ops/op_tester.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace ninfer;
using namespace ninfer::test;

double e4m3(unsigned code) {
    const auto exponent=(code>>3)&15,mantissa=code&7;
    const double value=exponent ? std::ldexp(1.+mantissa/8.,int(exponent)-7) : std::ldexp(double(mantissa),-9);
    return code&128 ? -value:value;
}
int root_code(unsigned root,unsigned radius_axis,int dimension) {
    constexpr float radius[]{0,.0992f,.125f,.1575f,.1984f,.25f,.315f,.3969f,
        .5f,.63f,.7937f,1,1.2599f,1.5874f,2,2.5198f};
    if(root>=240 || !(radius_axis>>4)) return 0;
    int value=0;
    if(root<112) {
        unsigned pair=root/4;
        for(int i=0;i<7;++i) for(int j=i+1;j<8;++j) {
            if(pair--==0) {
                if(dimension==i) value=root&2 ? 4:-4;
                if(dimension==j) value=root&1 ? 4:-4;
                i=7;break;
            }
        }
    } else {
        const unsigned signs=root-112;
        value=dimension<7 ? (signs&(1u<<dimension) ? 2:-2) :
            ((7-std::popcount(signs))%2 ? -2:2);
    }
    if(dimension==int((radius_axis>>1)&7)) value+=radius_axis&1 ? -1:1;
    return int(std::nearbyint(float(value)*radius[radius_axis>>4]));
}

// Independent dense FP64 oracle from public BF16 Q and independently decoded
// persistent K bytes/scales. H256 is a dense sign matrix, not a warp butterfly.
void qualify(KvCacheStorage storage,int heads,int kvheads,int pages,int samples=4,bool scale_edges=false) {
    constexpr int dim=256,tokens=7;
    const bool bf16=storage==KvCacheStorage::BFloat16;
    const bool fp8=storage==KvCacheStorage::Fp8E4M3Row256 || storage==KvCacheStorage::Fp8KeyNvfp4Value;
    const bool fp4=storage==KvCacheStorage::Nvfp4Group16;
    const bool root=storage==KvCacheStorage::RotatedE8RootKeyInt4Value;
    const bool lloyd=storage==KvCacheStorage::RotatedLloyd4KeyInt4Value;
    const bool int4=storage==KvCacheStorage::RotatedInt4KeyInt4ValueE8;
    const int leading=root?64:(fp4||lloyd||int4)?128:256;
    const auto dtype=bf16?DType::BF16:fp8?DType::FP8_E4M3FN:(root||fp4||lloyd||int4)?DType::U8:DType::I8;
    const int groups=fp4?16:fp8?1:4;
    const auto row_count=size_t(kvheads)*64*pages;
    std::mt19937 rng(1234+heads+pages+int(storage));
    std::uniform_real_distribution<float> random(-1.5f,1.5f);
    std::vector<std::uint16_t> q(size_t(dim)*heads*tokens);
    for(auto& x:q) x=f32_to_bf16(random(rng));
    std::vector<std::uint8_t> bytes(row_count*leading*(bf16?2:1)),scales(row_count*groups*(fp4?1:2));
    std::vector<double> represented(row_count*dim);
    constexpr int lloyd_codes[]{6,18,31,44,58,75,96,127};
    constexpr double fp4_codes[]{0,.5,1,1.5,2,3,4,6};
    for(size_t row=0;row<row_count;++row) {
        for(int g=0;g<groups;++g) {
            if(fp4) {
                // Vary every G16 group, head and token so an incorrect warp
                // scale broadcast cannot hide behind a uniform scale.
                constexpr std::uint8_t ordinary[]{0,1,7,8,16,24,32,40,48,56,60,52,44,36,28,20};
                constexpr std::uint8_t edges[]{0,1,7,8,15,16,32,48,64,80,96,112,120,124,125,126};
                scales[row*groups+g]=(scale_edges?edges:ordinary)[(row*3+g)%16];
            }
            else {const std::uint16_t bits=root?0x3400:0x2400;std::memcpy(scales.data()+2*(row*groups+g),&bits,2);}
        }
        for(int d=0;d<leading*(bf16?2:1);++d) bytes[row*leading*(bf16?2:1)+d]=std::uint8_t(rng());
        if(root) for(int d=0;d<64;d+=2) bytes[row*64+d]=std::uint8_t(rng()%240);
        for(int d=0;d<dim;++d) {
            double value=0;
            if(bf16) {
                const auto bits=f32_to_bf16(random(rng));std::memcpy(bytes.data()+2*(row*dim+d),&bits,2);
                value=bf16_to_f32(bits);
            } else if(fp8) {
                const auto code=std::uint8_t(rng()%127 | (rng()%2)*128);
                bytes[row*dim+d]=code;value=e4m3(code)/64;
            } else if(root) {
                const auto* pair=bytes.data()+row*64+(d/8)*2;
                value=root_code(pair[0],pair[1],d%8)/4.;
            } else if(fp4 || lloyd || int4) {
                const int code=(bytes[row*128+d/2]>>((d&1)*4))&15;
                if(fp4) value=(code&8 ? -1:1)*fp4_codes[code&7]*e4m3(scales[row*groups+d/16]);
                else if(lloyd) value=(code&8 ? -1:1)*lloyd_codes[code&7]/64.;
                else value=((code^8)-8)/64.;
            } else value=int(std::bit_cast<std::int8_t>(bytes[row*dim+d]))/64.;
            represented[row*dim+d]=value;
        }
    }
    std::vector<int> table(pages+1,-1);for(int p=0;p<pages;++p) table[p]=pages-1-p;
    std::vector<int> rows(samples),positions(samples);
    const int boundaries[]{0,63,std::min(64,pages*64-1),pages*64-11};
    for(int s=0;s<samples;++s) {rows[s]=(s%4)*2;positions[s]=boundaries[s%4];}
    if(samples==1) {rows[0]=6;positions[0]=pages*64-11;}
    const int output_pages=pages+1;
    auto dq=to_device(q),dk=to_device(bytes),ds=to_device(scales),dt=to_device(table);
    auto dr=to_device(rows),dp=to_device(positions);
    DeviceBuffer db(size_t(output_pages)*heads*samples*4),dn(size_t(2)*heads*samples*4),dy(size_t(output_pages)*samples*4);
    Tensor query(dq.p,DType::BF16,{dim,heads,tokens}),row(dr.p,DType::I32,{samples}),position(dp.p,DType::I32,{samples});
    Tensor scratch(db.p,DType::FP32,{output_pages,heads,samples}),norm(dn.p,DType::FP32,{2,heads,samples}),output(dy.p,DType::FP32,{output_pages,samples});
    PagedKVBatchLayerView cache;
    cache.head_dim=dim;cache.num_kv_heads=kvheads;cache.storage=storage;
    cache.k_pages=Tensor(dk.p,dtype,{leading,64,kvheads,pages});
    cache.k_scale_pages=Tensor(ds.p,fp4?DType::U8:DType::FP16,{groups,64,kvheads,pages});
    cache.block_tables=Tensor(dt.p,DType::I32,{output_pages,1});
    ops::attention_page_mass(query,row,position,0,cache,scratch,norm,output,nullptr);
    cuda_check(cudaDeviceSynchronize(),"sample");
    const auto actual=from_device<float>(dy,output.numel());std::vector<double> expected(actual.size());
    for(int s=0;s<samples;++s) for(int h=0;h<heads;++h) {
        std::vector<double> represented_q(dim);
        for(int d=0;d<dim;++d) {
            if(bf16) represented_q[d]=bf16_to_f32(q[(size_t(rows[s])*heads+h)*dim+d]);
            else {
                double value=0;
                for(int j=0;j<dim;++j) value+=(std::popcount(unsigned(d&j))%2?-1:1)*double(bf16_to_f32(q[(size_t(rows[s])*heads+h)*dim+j]));
                represented_q[d]=value/16;
            }
        }
        std::vector<double> logits(positions[s]+1);
        for(int t=0;t<=positions[s];++t) {
            const int physical=table[t/64],kh=h/(heads/kvheads);
            for(int d=0;d<dim;++d) logits[t]+=represented_q[d]*represented[((size_t(physical)*kvheads+kh)*64+t%64)*dim+d]/16;
        }
        const double maximum=*std::max_element(logits.begin(),logits.end());double sum=0;
        for(auto& v:logits) {v=std::exp(v-maximum);sum+=v;}
        for(int t=0;t<=positions[s];++t) expected[size_t(s)*output_pages+t/64]+=logits[t]/sum/heads;
    }
    double worst=0;
    for(size_t i=0;i<actual.size();++i) {
        const auto error=std::abs(actual[i]-expected[i]);worst=std::max(worst,error);
        if(!std::isfinite(actual[i]) || error>2e-5) {
            std::cerr<<"storage="<<int(storage)<<" i="<<i<<" actual="<<actual[i]<<" expected="<<expected[i]<<'\n';
            throw std::runtime_error("page mass differs from dense FP64 oracle");
        }
    }
    std::cout<<"attention page mass storage="<<int(storage)<<" H="<<heads<<" KVH="<<kvheads<<" pages="<<pages
        <<" samples="<<samples<<" scale_edges="<<scale_edges<<" max_abs="<<worst<<'\n';
}
int main() {
    try {
        if(cuda_unavailable()) return 77;
        for(auto storage:{KvCacheStorage::BFloat16,KvCacheStorage::Int8Group64,
            KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,KvCacheStorage::RotatedLloyd4KeyInt4Value,
            KvCacheStorage::RotatedInt4KeyInt4ValueE8,KvCacheStorage::RotatedE8RootKeyInt4Value,
            KvCacheStorage::Fp8E4M3Row256,KvCacheStorage::Fp8KeyNvfp4Value,KvCacheStorage::Nvfp4Group16})
            qualify(storage,24,4,4);
        qualify(KvCacheStorage::BFloat16,8,2,3);qualify(KvCacheStorage::Int8Group64,8,2,3);
        qualify(KvCacheStorage::Nvfp4Group16,24,4,4,64);
        qualify(KvCacheStorage::Nvfp4Group16,8,2,3,8,true);
        qualify(KvCacheStorage::Nvfp4Group16,24,4,577,1);
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
