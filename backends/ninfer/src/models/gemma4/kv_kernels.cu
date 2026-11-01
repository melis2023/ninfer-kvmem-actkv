// Gemma 4's quantized KV planes: storing F16 rows into them, loading rows back to F16, and turning
// queries for rotated planes (kernels.h, kv_codec.cuh).

#include "models/gemma4/kernels.h"
#include "models/gemma4/kv_codec.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4::kernels {
namespace {

constexpr int kWarps = 4; // warps per block, one (row, head, block) unit each

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ std::uint8_t* row_codes(const KvPlaneView& v, int sequence, int head,
                                                   int slot) {
    return v.codes + sequence * v.code_seq + head * v.code_head + slot * v.code_row;
}
__device__ __forceinline__ std::uint8_t* row_scales(const KvPlaneView& v, int sequence, int head,
                                                    int slot) {
    return v.scales + sequence * v.scale_seq + head * v.scale_head + slot * v.scale_row;
}

template <KvCoding C>
__launch_bounds__(kWarps * 32) __global__ void kv_store_kernel(KvStoreArgs a) {
    constexpr int kCode = kv_code_bytes(C), kScale = kv_scale_bytes(C);
    __shared__ float scratch[kWarps][kKvBlockDims];
    __shared__ __align__(16) std::uint8_t codes_s[kWarps][kCode];
    __shared__ __align__(16) std::uint8_t scales_s[kWarps][kScale];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int unit = blockIdx.x * kWarps + warp;
    const int blocks = a.dst.blocks;
    if (unit >= a.rows * a.heads * blocks) { return; }
    const int block = unit % blocks;
    const int head  = (unit / blocks) % a.heads;
    const int r     = unit / (blocks * a.heads);
    const int position = a.positions != nullptr ? a.positions[r] : a.first + r;
    const int sequence = a.sequences != nullptr ? a.sequences[r] : a.sequence;
    const int src_slot = a.src_ring != 0 ? position % a.src_ring : r;
    const int width    = blocks * kKvBlockDims;
    __half* src = a.src + head * a.src_head_stride + std::int64_t(src_slot) * width +
                  block * kKvBlockDims;
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { x[i] = __half2float(src[lane + 32 * i]); }
    if (a.dst.rotated) { ops::normalized_hadamard_d256_inplace(x, lane); }
    kv::encode_block<C>(x, codes_s[warp], scales_s[warp], scratch[warp], lane);
    __syncwarp();
    if (position >= a.store_from) {
        const int slot = a.dst_ring != 0 ? position % a.dst_ring : position;
        std::uint8_t* codes  = row_codes(a.dst, sequence, head, slot) + block * kCode;
        std::uint8_t* scales = row_scales(a.dst, sequence, head, slot) + block * kScale;
        for (int w = lane; w < kCode / 16; w += 32) {
            reinterpret_cast<uint4*>(codes)[w] = reinterpret_cast<const uint4*>(codes_s[warp])[w];
        }
        if (lane < kScale) { scales[lane] = scales_s[warp][lane]; }
    }
    if (a.write_back) {
        kv::decode_block<C>(codes_s[warp], scales_s[warp], x, scratch[warp], lane);
        if (a.dst.rotated) { ops::normalized_hadamard_d256_inplace(x, lane); }
#pragma unroll
        for (int i = 0; i < 8; ++i) { src[lane + 32 * i] = __float2half(x[i]); }
    }
}

template <KvCoding C>
__launch_bounds__(kWarps * 32) __global__ void kv_load_kernel(KvLoadArgs a) {
    constexpr int kCode = kv_code_bytes(C), kScale = kv_scale_bytes(C);
    __shared__ float scratch[kWarps][kKvBlockDims];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int unit = blockIdx.x * kWarps + warp;
    const int blocks = a.src.blocks;
    if (unit >= (a.end - a.begin) * a.heads * blocks) { return; }
    const int block    = unit % blocks;
    const int head     = (unit / blocks) % a.heads;
    const int position = a.begin + unit / (blocks * a.heads);
    const int slot     = a.src_ring != 0 ? position % a.src_ring : position;
    const std::uint8_t* codes  = row_codes(a.src, a.sequence, head, slot) + block * kCode;
    const std::uint8_t* scales = row_scales(a.src, a.sequence, head, slot) + block * kScale;
    float x[8];
    kv::decode_block<C>(codes, scales, x, scratch[warp], lane);
    if (a.src.rotated) { ops::normalized_hadamard_d256_inplace(x, lane); }
    const int dst_slot = a.dst_ring != 0 ? position % a.dst_ring : position;
    __half* dst = a.dst + head * a.dst_head_stride +
                  std::int64_t(dst_slot) * (blocks * kKvBlockDims) + block * kKvBlockDims;
#pragma unroll
    for (int i = 0; i < 8; ++i) { dst[lane + 32 * i] = __float2half(x[i]); }
}

__launch_bounds__(kWarps * 32) __global__ void rotate_rows_kernel(float* x, int rows) {
    const int row  = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) { return; }
    float* p = x + std::int64_t(row) * kKvBlockDims;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = p[lane + 32 * i]; }
    ops::normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i) { p[lane + 32 * i] = v[i]; }
}

// One row's block of a plane: x from the F16 row, turned for a rotated plane, encoded in place.
template <KvCoding C>
__device__ __forceinline__ void store_row_block(const __half* src, const KvPlaneView& plane,
                                                int sequence, int head, int slot, int block,
                                                float* scratch, int lane) {
    float x[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { x[i] = __half2float(src[lane + 32 * i]); }
    if (plane.rotated) { ops::normalized_hadamard_d256_inplace(x, lane); }
    kv::encode_block<C>(x, row_codes(plane, sequence, head, slot) + block * kv_code_bytes(C),
                        row_scales(plane, sequence, head, slot) + block * kv_scale_bytes(C), scratch,
                        lane);
}

// Units: the K blocks (KC not F16), then the V blocks, then the query rows to turn.
template <KvCoding KC, KvCoding VC>
__launch_bounds__(kWarps * 32) __global__ void kv_store_rows_kernel(KvRowsArgs a) {
    __shared__ float scratch[kWarps][kKvBlockDims];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int unit = blockIdx.x * kWarps + warp;
    const int blocks = a.v_plane.blocks;
    const int width  = blocks * kKvBlockDims;
    const int planed = a.rows * a.heads * blocks;
    const auto place = [&](int u, int& r, int& head, int& block) {
        block = u % blocks;
        head  = (u / blocks) % a.heads;
        r     = u / (blocks * a.heads);
    };
    if constexpr (KC != KvCoding::F16) {
        if (unit < planed) {
            int r, head, block;
            place(unit, r, head, block);
            const int p = a.positions[r];
            store_row_block<KC>(a.k + head * a.head_stride + std::int64_t(r) * width + block * kKvBlockDims,
                                a.k_plane, a.sequences[r], head, a.ring != 0 ? p % a.ring : p, block,
                                scratch[warp], lane);
            return;
        }
        unit -= planed;
    }
    if (unit < planed) {
        int r, head, block;
        place(unit, r, head, block);
        const int p = a.positions[r];
        store_row_block<VC>(a.v + head * a.head_stride + std::int64_t(r) * width + block * kKvBlockDims,
                            a.v_plane, a.sequences[r], head, a.ring != 0 ? p % a.ring : p, block,
                            scratch[warp], lane);
        return;
    }
    unit -= planed;
    if (unit < a.q_rows) {
        float* q = a.q + std::int64_t(unit) * kKvBlockDims;
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { v[i] = q[lane + 32 * i]; }
        ops::normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
        for (int i = 0; i < 8; ++i) { q[lane + 32 * i] = v[i]; }
    }
}

template <template <KvCoding> class Launch, typename... Args>
void dispatch(KvCoding coding, Args&&... args) {
    switch (coding) {
    case KvCoding::Int8G64: Launch<KvCoding::Int8G64>::run(args...); return;
    case KvCoding::Int4G32: Launch<KvCoding::Int4G32>::run(args...); return;
    case KvCoding::Fp8Row: Launch<KvCoding::Fp8Row>::run(args...); return;
    case KvCoding::Nvfp4G16: Launch<KvCoding::Nvfp4G16>::run(args...); return;
    case KvCoding::Lloyd4: Launch<KvCoding::Lloyd4>::run(args...); return;
    case KvCoding::Int4E8: Launch<KvCoding::Int4E8>::run(args...); return;
    case KvCoding::RootE8: Launch<KvCoding::RootE8>::run(args...); return;
    case KvCoding::F16: break;
    }
    throw std::invalid_argument("gemma4 kv: not a quantized coding");
}

template <KvCoding C>
struct StoreLaunch {
    static void run(const KvStoreArgs& a, int units, cudaStream_t stream) {
        kv_store_kernel<C><<<(units + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(a);
    }
};
template <KvCoding C>
struct LoadLaunch {
    static void run(const KvLoadArgs& a, int units, cudaStream_t stream) {
        kv_load_kernel<C><<<(units + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(a);
    }
};

} // namespace

void kv_store(const KvStoreArgs& a, cudaStream_t stream) {
    const int units = a.rows * a.heads * a.dst.blocks;
    if (units <= 0) { return; }
    dispatch<StoreLaunch>(a.dst.coding, a, units, stream);
    check_launch("kv_store");
}

void kv_load(const KvLoadArgs& a, cudaStream_t stream) {
    const int units = (a.end - a.begin) * a.heads * a.src.blocks;
    if (units <= 0) { return; }
    dispatch<LoadLaunch>(a.src.coding, a, units, stream);
    check_launch("kv_load");
}

void kv_store_rows(const KvRowsArgs& a, cudaStream_t stream) {
    const bool keys  = a.k_plane.codes != nullptr;
    const int planed = a.rows * a.heads * a.v_plane.blocks;
    const int units  = planed * (keys ? 2 : 1) + (a.q != nullptr ? a.q_rows : 0);
    if (units <= 0) { return; }
    const dim3 grid((units + kWarps - 1) / kWarps), block(kWarps * 32);
    KvRowsArgs args = a;
    if (args.q == nullptr) { args.q_rows = 0; }
    using C = KvCoding;
    const C k = keys ? a.k_plane.coding : C::F16, v = a.v_plane.coding;
#define GEMMA4_KV_ROWS(KC, VC)                                                                     \
    if (k == C::KC && v == C::VC) {                                                                \
        kv_store_rows_kernel<C::KC, C::VC><<<grid, block, 0, stream>>>(args);                      \
        check_launch("kv_store_rows");                                                             \
        return;                                                                                    \
    }
    // The sliding blocks' (key, value) codings, then the full blocks' value planes.
    GEMMA4_KV_ROWS(Int8G64, Int8G64)
    GEMMA4_KV_ROWS(Fp8Row, Fp8Row)
    GEMMA4_KV_ROWS(Int8G64, Int4G32)
    GEMMA4_KV_ROWS(Nvfp4G16, Nvfp4G16)
    GEMMA4_KV_ROWS(Fp8Row, Nvfp4G16)
    GEMMA4_KV_ROWS(Lloyd4, Int4G32)
    GEMMA4_KV_ROWS(Int4E8, Int4G32)
    GEMMA4_KV_ROWS(RootE8, Int4G32)
    GEMMA4_KV_ROWS(F16, Int8G64)
    GEMMA4_KV_ROWS(F16, Fp8Row)
    GEMMA4_KV_ROWS(F16, Nvfp4G16)
    GEMMA4_KV_ROWS(F16, Int4G32)
#undef GEMMA4_KV_ROWS
    throw std::invalid_argument("gemma4 kv_store_rows: no kernel for this key/value coding");
}

void rotate_rows(float* x, int rows, cudaStream_t stream) {
    if (rows <= 0) { return; }
    rotate_rows_kernel<<<(rows + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(x, rows);
    check_launch("rotate_rows");
}

} // namespace ninfer::models::gemma4::kernels
