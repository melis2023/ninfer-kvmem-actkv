#include "models/gemma4/vision_kernels.h"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4::vision_kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 vision ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float s = lane < int(blockDim.x >> 5) ? partial[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) { total = s; }
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Row kernels: one float4 of a row per thread (width % 4 == 0, width / 4 <= 1024, a multiple of
// 32 threads).
__device__ __forceinline__ float4 load_bf16x4(const __nv_bfloat16* p) {
    const uint2 w   = *reinterpret_cast<const uint2*>(p);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 w;
    w.x = *reinterpret_cast<const unsigned*>(&lo);
    w.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = w;
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ __forceinline__ float4 scale4(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float squares(float4 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
}
__device__ __forceinline__ float4 ld4(const float* p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}

int row_threads(int width) {
    if (width % 4 != 0 || width / 4 > 1024 || (width / 4) % 32 != 0) {
        throw std::invalid_argument("gemma4 vision: row width must be a multiple of 128 up to 4096");
    }
    return width / 4;
}

unsigned blocks_for(std::int64_t n, int threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

__global__ void embed_kernel(EmbedArgs a) {
    const int p = blockIdx.x, i = threadIdx.x;
    const int local = p % a.per_unit;
    const int x = local % a.grid_w, y = local / a.grid_w;
    const std::int64_t row = std::int64_t(p) * a.hidden;
    float4 v = ld4(a.x + row, i);
    v = add4(v, load_bf16x4(a.table + std::int64_t(x) * a.hidden + 4 * i));
    v = add4(v, load_bf16x4(a.table + std::int64_t(a.positions + y) * a.hidden + 4 * i));
    reinterpret_cast<float4*>(a.h + row)[i] = v;
    const float rs = rsqrtf(block_sum(squares(v)) / float(a.hidden) + a.eps);
    store_bf16x4(a.xn + row + 4 * i, mul4(scale4(v, rs), ld4(a.norm, i)));
}

// Block per patch, a warp per head; a head's 72 values in lanes l, l + 32, l + 64 (< D).
template <int D>
__global__ void qkv_kernel(QkvArgs a) {
    static_assert(D % 4 == 0 && D <= 96, "head width");
    constexpr int kQuarter = D / 4, kHalf = D / 2;
    extern __shared__ float rows[]; // [heads][D]
    const int p = blockIdx.x, head = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (head >= a.heads) { return; }
    float* s = rows + head * D;
    const int local = p % a.per_unit;
    const float column = float(local % a.grid_w), row = float(local / a.grid_w);
    const int hidden = a.heads * D;
    const float* in = a.qkv + std::int64_t(p) * 3 * hidden + head * D;
    __nv_bfloat16* outs[3]   = {a.q, a.k, a.v};
    const float* gammas[3]   = {a.q_norm, a.k_norm, nullptr};
    for (int part = 0; part < 3; ++part) {
        const float* src = in + part * hidden;
        float values[3];
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int d = lane + 32 * j;
            values[j]   = d < D ? src[d] : 0.0f;
            ss += values[j] * values[j];
        }
        const float rs = rsqrtf(warp_sum(ss) / float(D) + a.eps);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int d = lane + 32 * j;
            if (d < D) { s[d] = values[j] * rs * (gammas[part] ? gammas[part][d] : 1.0f); }
        }
        __syncwarp();
        __nv_bfloat16* dst = outs[part] + std::int64_t(p) * hidden + head * D;
        const float out_scale = part == 0 ? a.q_scale : 1.0f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int d = lane + 32 * j;
            if (d >= D) { continue; }
            float out = s[d];
            if (part < 2) {
                // Half 0 turns by the column, half 1 by the row; pair (i, i + D/4) in each.
                const int half = d / kHalf, local_d = d % kHalf, i = local_d % kQuarter;
                const float angle = (half == 0 ? column : row) * a.inv_freq[i];
                float sn, cs;
                sincosf(angle, &sn, &cs);
                out = local_d < kQuarter ? s[d] * cs - s[d + kQuarter] * sn
                                         : s[d] * cs + s[d - kQuarter] * sn;
            }
            dst[d] = __float2bfloat16_rn(out * out_scale);
        }
        __syncwarp();
    }
}

__global__ void residual_kernel(float* __restrict__ h, const float* __restrict__ o,
                                int hidden, const float* __restrict__ post,
                                const float* __restrict__ next, float eps,
                                __nv_bfloat16* __restrict__ xn) {
    const int p = blockIdx.x, i = threadIdx.x;
    const std::int64_t row = std::int64_t(p) * hidden;
    const float4 delta = ld4(o + row, i);
    const float rs     = rsqrtf(block_sum(squares(delta)) / float(hidden) + eps);
    float4 v = reinterpret_cast<float4*>(h + row)[i];
    v = add4(v, mul4(scale4(delta, rs), ld4(post, i)));
    reinterpret_cast<float4*>(h + row)[i] = v;
    if (next == nullptr) { return; }
    const float rs2 = rsqrtf(block_sum(squares(v)) / float(hidden) + eps);
    store_bf16x4(xn + row + 4 * i, mul4(scale4(v, rs2), ld4(next, i)));
}

__global__ void geglu_kernel(const float* __restrict__ gu, std::int64_t n, int width,
                             __nv_bfloat16* __restrict__ mid) {
    const std::int64_t j = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; // 4 values each
    if (j * 4 >= n) { return; }
    const std::int64_t p = (j * 4) / width, i = (j * 4) % width;
    const float4 g = *reinterpret_cast<const float4*>(gu + p * 2 * width + i);
    const float4 u = *reinterpret_cast<const float4*>(gu + p * 2 * width + width + i);
    store_bf16x4(mid + p * width + i, make_float4(gelu_tanh(g.x) * u.x, gelu_tanh(g.y) * u.y,
                                                  gelu_tanh(g.z) * u.z, gelu_tanh(g.w) * u.w));
}

__global__ void pool_kernel(PoolArgs a) {
    const int t = blockIdx.x, i = threadIdx.x;
    const int columns = a.grid_w / a.pooling, rows = a.grid_h / a.pooling;
    const int per_unit = columns * rows;
    const int unit = t / per_unit, local = t % per_unit;
    const int ty = local / columns, tx = local % columns;
    const std::int64_t base = std::int64_t(unit) * a.grid_w * a.grid_h;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int dy = 0; dy < a.pooling; ++dy) {
        for (int dx = 0; dx < a.pooling; ++dx) {
            const std::int64_t patch =
                base + std::int64_t(ty * a.pooling + dy) * a.grid_w + tx * a.pooling + dx;
            sum = add4(sum, ld4(a.h + patch * a.hidden, i));
        }
    }
    float4 v = scale4(sum, a.scale / float(a.pooling * a.pooling));
    if (a.std_bias != nullptr) {
        const float4 b = ld4(a.std_bias, i), s = ld4(a.std_scale, i);
        v = make_float4((v.x - b.x) * s.x, (v.y - b.y) * s.y, (v.z - b.z) * s.z,
                        (v.w - b.w) * s.w);
    }
    const float rs = rsqrtf(block_sum(squares(v)) / float(a.hidden) + a.eps);
    store_bf16x4(a.out + std::int64_t(t) * a.hidden + 4 * i, scale4(v, rs));
}

__global__ void scatter_rows_kernel(const __nv_bfloat16* __restrict__ src, int per_unit,
                                    int width, const std::int32_t* __restrict__ rows,
                                    float* __restrict__ dst) {
    const int t = blockIdx.x, i = threadIdx.x;
    const std::int64_t row = rows[t / per_unit] + t % per_unit;
    reinterpret_cast<float4*>(dst + row * width)[i] = load_bf16x4(src + std::int64_t(t) * width + 4 * i);
}

} // namespace

void embed(const EmbedArgs& a, cudaStream_t stream) {
    if (a.patches <= 0) { return; }
    if (a.per_unit <= 0 || a.grid_w <= 0 || a.per_unit % a.grid_w != 0) {
        throw std::invalid_argument("gemma4 vision embed: invalid patch grid");
    }
    embed_kernel<<<a.patches, row_threads(a.hidden), 0, stream>>>(a);
    check_launch("embed");
}

void qkv(const QkvArgs& a, cudaStream_t stream) {
    if (a.patches <= 0) { return; }
    if (a.head_dim != 72 || a.heads <= 0 || a.heads > 32) {
        throw std::invalid_argument("gemma4 vision qkv: the kernel implements heads of 72");
    }
    const std::size_t shared = std::size_t(a.heads) * 72 * sizeof(float);
    qkv_kernel<72><<<a.patches, 32 * a.heads, shared, stream>>>(a);
    check_launch("qkv");
}

void residual(float* h, const float* o, int patches, int hidden, const float* post,
              const float* next, float eps, __nv_bfloat16* xn, cudaStream_t stream) {
    if (patches <= 0) { return; }
    residual_kernel<<<patches, row_threads(hidden), 0, stream>>>(h, o, hidden, post, next, eps, xn);
    check_launch("residual");
}

void geglu(const float* gu, int patches, int width, __nv_bfloat16* mid,
           cudaStream_t stream) {
    if (patches <= 0) { return; }
    if (width % 4 != 0) { throw std::invalid_argument("gemma4 vision geglu: width % 4"); }
    const std::int64_t n = std::int64_t(patches) * width;
    geglu_kernel<<<blocks_for(n / 4, 256), 256, 0, stream>>>(gu, n, width, mid);
    check_launch("geglu");
}

void pool(const PoolArgs& a, cudaStream_t stream) {
    if (a.units <= 0) { return; }
    if (a.pooling <= 0 || a.grid_w % a.pooling != 0 || a.grid_h % a.pooling != 0) {
        throw std::invalid_argument("gemma4 vision pool: the grid must be whole pooling blocks");
    }
    const int tokens = a.units * (a.grid_w / a.pooling) * (a.grid_h / a.pooling);
    pool_kernel<<<tokens, row_threads(a.hidden), 0, stream>>>(a);
    check_launch("pool");
}

void scatter_rows(const __nv_bfloat16* src, int tokens, int per_unit, int width,
                  const std::int32_t* rows, float* dst, cudaStream_t stream) {
    if (tokens <= 0) { return; }
    scatter_rows_kernel<<<tokens, row_threads(width), 0, stream>>>(src, per_unit, width, rows, dst);
    check_launch("scatter_rows");
}

} // namespace ninfer::models::gemma4::vision_kernels
