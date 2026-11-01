// ninfer::ops - hyper-connection read/write of Qwen3.8-Flash-Next (contract in
// include/ninfer/ops/hyper_connection.h). Four launches: per-stream RMSNorm into FP32 workspace,
// the down and inject GEMVs with their activations, the up GEMV with the gated stream mix, and the
// weighted write. Write/read fusion folds that write into the following norm while preserving
// the FP32 stack store. Every product accumulates in FP32. Up to eight tokens, the two GEMVs stream their
// 6.5 MB of weights each in 16-byte vectors, specialized for the token count: every thread issues
// all of its weight loads before its first product, so a whole matrix is in flight at once; the
// down rows split K over a CTA's warps, two rows a CTA sharing each activation load, and the up rows
// of eight hidden indices are one contiguous run per stream, their low-rank input staged in shared
// memory. Q8_0 matrices take kernels of their own: a warp copies runs of 34-byte blocks into
// shared memory in 16-byte vectors, and each lane decodes four weights at a time into FP32 once
// and multiplies them with the FP32 activations of every token, the lanes' activation vectors
// adjacent. Wider calls run the down and up products
// as cuBLAS tensor-core GEMMs over BF16 operands (the normalized stack and the low-rank activation
// rounded to BF16, as the checkpoint's own arithmetic keeps them, Q8_0 matrices rounded to BF16
// first) and keep the four inject rows on the FP32 GEMVs.
#include "ninfer/ops/hyper_connection.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kStreams = 4;
constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kLowrank = 320;
constexpr std::int32_t kWidth   = kStreams * kHidden;
// Tokens one CTA accumulates at once; wider calls cover column chunks with grid.y.
constexpr int kColumnChunk = 8;
// The down GEMV: one CTA per row, its warps splitting the 10240-wide input.
constexpr int kDownWarps = 8;
constexpr int kDownSlice = kWidth / kDownWarps; // 1280 inputs, five 8-wide vectors per lane
static_assert(kDownSlice % 256 == 0);
constexpr int kDownVectors = kDownSlice / 256;
// Rows of [down; inject] one narrow down CTA takes: they share each activation load.
constexpr int kDownRows = 2;
// The up GEMV: per CTA eight hidden indices, one warp per stream, four lanes per 320-wide row.
constexpr int kUpRows    = 8;
constexpr int kUpLanes   = 4;
constexpr int kUpVectors = kLowrank / 8; // 40 per row
static_assert(kUpRows * kUpLanes == 32 && kUpVectors % kUpLanes == 0);
static_assert(kUpRows * kColumnChunk <= kStreams * 32);
// The norm: one thread per four values of a stream.
constexpr int kNormThreads = kHidden / 4;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float to_float(float value) { return value; }
__device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&out)[8]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

// Eight consecutive FP32 values, 32-byte aligned.
__device__ __forceinline__ void load_f32x8(const float* p, float (&out)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w;
    out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
}

// One block per (stream, token), four values per thread: xn = x * rsqrt(mean x^2 + eps) * (1 + g),
// and its BF16 rounding when asked for. The fused form first writes the block output to the
// FP32 stack; the register values have that same rounding and feed the norm without reloading.
template <bool Write, typename Output = float>
__global__ void __launch_bounds__(kNormThreads)
    hc_norm_kernel(float* __restrict__ stack, const Output* __restrict__ y,
                   const float* __restrict__ previous_inject,
                   const __nv_bfloat16* __restrict__ norm,
                   float eps, float* __restrict__ normalized,
                   __nv_bfloat16* __restrict__ normalized_bf16) {
    const int c             = blockIdx.x;
    const int t             = blockIdx.y;
    const std::int64_t base = (static_cast<std::int64_t>(t) * kStreams + c) * kHidden;
    __shared__ float partial[kNormThreads / 32];
    float4 x = reinterpret_cast<const float4*>(stack + base)[threadIdx.x];
    if constexpr (Write) {
        const std::int64_t y_base = static_cast<std::int64_t>(t) * kHidden + 4 * threadIdx.x;
        const float weight = previous_inject[t * kStreams + c];
        x.x = fmaf(to_float(y[y_base]), weight, x.x);
        x.y = fmaf(to_float(y[y_base + 1]), weight, x.y);
        x.z = fmaf(to_float(y[y_base + 2]), weight, x.z);
        x.w = fmaf(to_float(y[y_base + 3]), weight, x.w);
        reinterpret_cast<float4*>(stack + base)[threadIdx.x] = x;
    }
    float sum      = warp_sum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = sum; }
    __syncthreads();
    sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNormThreads / 32; ++w) { sum += partial[w]; }
    const float scale = rsqrtf(sum / kHidden + eps);
    const auto* g   = reinterpret_cast<const __nv_bfloat162*>(norm + c * kHidden) + 2 * threadIdx.x;
    const float2 g0 = __bfloat1622float2(g[0]), g1 = __bfloat1622float2(g[1]);
    const float4 xn = make_float4(x.x * scale * (1.0f + g0.x), x.y * scale * (1.0f + g0.y),
                                  x.z * scale * (1.0f + g1.x), x.w * scale * (1.0f + g1.y));
    reinterpret_cast<float4*>(normalized + base)[threadIdx.x] = xn;
    if (normalized_bf16 != nullptr) {
        auto* out = reinterpret_cast<__nv_bfloat162*>(normalized_bf16 + base) + 2 * threadIdx.x;
        out[0]    = __floats2bfloat162_rn(xn.x, xn.y);
        out[1]    = __floats2bfloat162_rn(xn.z, xn.w);
    }
}

// One block per (row of [down; inject] from first_row on, column chunk): lowrank + streams rows
// over the 10240-wide input, each warp a 1280-wide slice. Rows below lowrank write silu(v / n) to
// `low`; inject rows write 2 sigmoid(v / n).
__global__ void __launch_bounds__(kDownWarps * 32)
    hc_down_kernel(const float* __restrict__ normalized, const __nv_bfloat16* __restrict__ down,
                   const __nv_bfloat16* __restrict__ inject, int tokens, int first_row,
                   float* __restrict__ low, float* __restrict__ inject_weights) {
    __shared__ float partial[kDownWarps][kColumnChunk];
    const int row   = first_row + static_cast<int>(blockIdx.x);
    const int t0    = blockIdx.y * kColumnChunk;
    const int count = min(kColumnChunk, tokens - t0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16* weights = row < kLowrank
                                       ? down + static_cast<std::int64_t>(row) * kWidth
                                       : inject + static_cast<std::int64_t>(row - kLowrank) * kWidth;
    const int first              = warp * kDownSlice;
    float acc[kColumnChunk]      = {};
#pragma unroll
    for (int i = 0; i < kDownSlice / 256; ++i) {
        const int k = first + 8 * (lane + 32 * i);
        float w[8];
        unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(weights + k)), w);
#pragma unroll
        for (int j = 0; j < kColumnChunk; ++j) {
            if (j < count) {
                float x[8];
                load_f32x8(normalized + static_cast<std::int64_t>(t0 + j) * kWidth + k, x);
#pragma unroll
                for (int e = 0; e < 8; ++e) { acc[j] = fmaf(w[e], x[e], acc[j]); }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kColumnChunk; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) { partial[warp][j] = v; }
    }
    __syncthreads();
    if (warp != 0 || lane >= count) { return; }
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kDownWarps; ++w) { v += partial[w][lane]; }
    v /= kStreams;
    const int t = t0 + lane;
    if (row < kLowrank) {
        low[static_cast<std::int64_t>(t) * kLowrank + row] = v * sigmoid_f(v);
    } else {
        inject_weights[static_cast<std::int64_t>(t) * kStreams + row - kLowrank] =
            2.0f * sigmoid_f(v);
    }
}

// The narrow form of hc_down_kernel for T tokens (all of them): one block per kDownRows rows of
// [down; inject] (`rows` of them), each warp a 1280-wide slice of every row. A thread loads its
// rows' weight vectors first and each activation vector once for all of them; every row
// accumulates in the same order as hc_down_kernel, so the two agree bit for bit.
template <int T>
__global__ void __launch_bounds__(kDownWarps * 32)
    hc_down_narrow_kernel(const float* __restrict__ normalized,
                          const __nv_bfloat16* __restrict__ down,
                          const __nv_bfloat16* __restrict__ inject, int rows,
                          float* __restrict__ low, float* __restrict__ inject_weights) {
    __shared__ float partial[kDownWarps][kDownRows][T];
    const int row0 = static_cast<int>(blockIdx.x) * kDownRows;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int first = warp * kDownSlice;
    uint4 w[kDownRows][kDownVectors];
#pragma unroll
    for (int r = 0; r < kDownRows; ++r) {
        const int row = min(row0 + r, rows - 1);
        const __nv_bfloat16* weights =
            row < kLowrank ? down + static_cast<std::int64_t>(row) * kWidth
                           : inject + static_cast<std::int64_t>(row - kLowrank) * kWidth;
#pragma unroll
        for (int i = 0; i < kDownVectors; ++i) {
            w[r][i] = __ldg(reinterpret_cast<const uint4*>(weights + first + 8 * (lane + 32 * i)));
        }
    }
    float acc[kDownRows][T] = {};
#pragma unroll
    for (int i = 0; i < kDownVectors; ++i) {
        const int k = first + 8 * (lane + 32 * i);
#pragma unroll
        for (int j = 0; j < T; ++j) {
            float x[8];
            load_f32x8(normalized + static_cast<std::int64_t>(j) * kWidth + k, x);
#pragma unroll
            for (int r = 0; r < kDownRows; ++r) {
                float v[8];
                unpack_bf16x8(w[r][i], v);
#pragma unroll
                for (int e = 0; e < 8; ++e) { acc[r][j] = fmaf(v[e], x[e], acc[r][j]); }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kDownRows; ++r) {
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float v = warp_sum(acc[r][j]);
            if (lane == 0) { partial[warp][r][j] = v; }
        }
    }
    __syncthreads();
    if (threadIdx.x >= kDownRows * T) { return; }
    const int r = threadIdx.x / T, t = threadIdx.x % T;
    const int row = row0 + r;
    if (row >= rows) { return; }
    float v = 0.0f;
#pragma unroll
    for (int w8 = 0; w8 < kDownWarps; ++w8) { v += partial[w8][r][t]; }
    v /= kStreams;
    if (row < kLowrank) {
        low[static_cast<std::int64_t>(t) * kLowrank + row] = v * sigmoid_f(v);
    } else {
        inject_weights[static_cast<std::int64_t>(t) * kStreams + row - kLowrank] =
            2.0f * sigmoid_f(v);
    }
}

// One block per eight consecutive hidden indices, for T tokens (all of them), one warp per stream
// c: the eight gate rows c * hidden + d of `up` are contiguous (each lowrank wide), four lanes on
// each, their weight vectors loaded before the products and the low-rank input staged in shared
// memory. Then mixed[d] = (1/n) sum_c sigmoid(gate_c) xn[c, d].
template <int T>
__global__ void __launch_bounds__(kStreams * 32)
    hc_up_mix_kernel(const float* __restrict__ normalized, const float* __restrict__ low,
                     const __nv_bfloat16* __restrict__ up, __nv_bfloat16* __restrict__ mixed) {
    constexpr int kVectors = kUpVectors / kUpLanes;
    __shared__ float gated[kStreams][kUpRows][T];
    __shared__ __align__(16) float staged[T][kLowrank];
    const int c    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane / kUpLanes, part = lane % kUpLanes;
    const int d0                 = blockIdx.x * kUpRows;
    const int d                  = d0 + r;
    const __nv_bfloat16* weights = up + static_cast<std::int64_t>(c * kHidden + d) * kLowrank;
    uint4 w[kVectors];
#pragma unroll
    for (int i = 0; i < kVectors; ++i) {
        w[i] = __ldg(reinterpret_cast<const uint4*>(weights + 8 * (part + kUpLanes * i)));
    }
    for (int i = threadIdx.x; i < T * kLowrank / 4; i += kStreams * 32) {
        reinterpret_cast<float4*>(&staged[0][0])[i] = reinterpret_cast<const float4*>(low)[i];
    }
    __syncthreads();
    float acc[T] = {};
#pragma unroll
    for (int i = 0; i < kVectors; ++i) {
        const int v = part + kUpLanes * i;
        float wf[8];
        unpack_bf16x8(w[i], wf);
#pragma unroll
        for (int j = 0; j < T; ++j) {
            float x[8];
            load_f32x8(&staged[j][8 * v], x);
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[j] = fmaf(wf[e], x[e], acc[j]); }
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 1);
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 2);
        if (part == 0) {
            gated[c][r][j] = sigmoid_f(acc[j]) *
                             normalized[static_cast<std::int64_t>(j) * kWidth + c * kHidden + d];
        }
    }
    __syncthreads();
    if (threadIdx.x >= kUpRows * T) { return; }
    const int row = threadIdx.x / T, j = threadIdx.x % T;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) { sum += gated[s][row][j]; }
    mixed[static_cast<std::int64_t>(j) * kHidden + d0 + row] = __float2bfloat16_rn(sum / kStreams);
}

// --- Q8_0 matrices: 32 consecutive inputs of a row per 34-byte block, a binary16 scale d and 32
// signed codes q, value d * q. A row of down or inject is 320 blocks, a row of up 10. Blocks are
// only 2-byte aligned, so a warp copies a run of 32 consecutive blocks (1,088 bytes, 16-byte
// aligned in every use below) into shared memory in 16-byte vectors and decodes the weights from
// there into FP32 values d * q, exact, before its products.
constexpr int kQ8Values   = 32;
constexpr int kQ8Bytes    = 34;
constexpr int kQ8RunBytes = 32 * kQ8Bytes;
constexpr int kQ8RunVectors = kQ8RunBytes / 16; // 68
constexpr std::int64_t kDownQ8RowBytes = std::int64_t(kWidth) / kQ8Values * kQ8Bytes;
constexpr std::int64_t kUpQ8RowBytes   = std::int64_t(kLowrank) / kQ8Values * kQ8Bytes;
// The narrow down product: a warp per run of 32 blocks (1,024 inputs) of each of a CTA's four
// rows. Every CTA reads the activations whole: four rows share them (81 CTAs for 324 rows), which
// measured faster than two at every width up to eight on an RTX 3090.
constexpr int kDownQ8Warps = kWidth / kQ8Values / 32;
constexpr int kDownQ8Rows  = 4;
static_assert(kDownQ8Warps * 32 * kQ8Values == kWidth && kQ8RunBytes % 16 == 0);
// The narrow up product: eight hidden indices per CTA, as hc_up_mix_kernel, a warp per stream
// staging its eight contiguous rows (2,720 bytes), four lanes per row.
constexpr int kUpQ8Rows      = kUpRows;
constexpr int kUpQ8Bytes     = kUpQ8Rows * int(kUpQ8RowBytes);
constexpr int kUpQ8Vectors   = kUpQ8Bytes / 16; // 170
static_assert(kUpQ8Rows * kUpLanes == 32 && kUpQ8Bytes % 16 == 0 && kLowrank % (4 * kUpLanes) == 0);

// Two codes of a 16-bit word as floats: each two's-complement byte, biased by 128, becomes the low
// mantissa byte of 2^23, which leaves 2^23 + 128 + q exactly.
__device__ __forceinline__ float2 q8_pair(unsigned word) {
    const unsigned biased = word ^ 0x8080u;
    return make_float2(__uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7540)) - 8388736.0f,
                       __uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7541)) - 8388736.0f);
}

// A block in shared memory as its 32 values d * q.
__device__ __forceinline__ void decode_q8(const std::uint8_t* block, float (&w)[kQ8Values]) {
    const auto* words = reinterpret_cast<const unsigned short*>(block);
    const float d     = __half2float(__ushort_as_half(words[0]));
#pragma unroll
    for (int i = 0; i < kQ8Values / 2; ++i) {
        const float2 q = q8_pair(words[1 + i]);
        w[2 * i]       = q.x * d;
        w[2 * i + 1]   = q.y * d;
    }
}

// Values 4 quad .. 4 quad + 3 of a block in shared memory, d * q exact.
__device__ __forceinline__ float4 decode_q8x4(const std::uint8_t* block, int quad) {
    const auto* words = reinterpret_cast<const unsigned short*>(block);
    const float d     = __half2float(__ushort_as_half(words[0]));
    const float2 lo = q8_pair(words[1 + 2 * quad]), hi = q8_pair(words[2 + 2 * quad]);
    return make_float4(lo.x * d, lo.y * d, hi.x * d, hi.y * d);
}

__device__ __forceinline__ float dot4(const float4& w, const float4& x, float sum) {
    sum = fmaf(w.x, x.x, sum);
    sum = fmaf(w.y, x.y, sum);
    sum = fmaf(w.z, x.z, sum);
    return fmaf(w.w, x.w, sum);
}

// A run's three 16-byte vectors per lane (the third only on four lanes).
struct Q8Run {
    uint4 v[3];
};

__device__ __forceinline__ Q8Run load_q8_run(const std::uint8_t* run, int lane) {
    const auto* src = reinterpret_cast<const uint4*>(run);
    Q8Run out;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int at = lane + 32 * i;
        out.v[i]     = at < kQ8RunVectors ? __ldg(src + at) : uint4{};
    }
    return out;
}

__device__ __forceinline__ void store_q8_run(const Q8Run& run, std::uint8_t* staged, int lane) {
    auto* dst = reinterpret_cast<uint4*>(staged);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int at = lane + 32 * i;
        if (at < kQ8RunVectors) { dst[at] = run.v[i]; }
    }
}

// hc_down_narrow_kernel over Q8_0 rows, kDownQ8Rows of them a CTA: warp w stages run w (1,024
// inputs) of each, all loads before the first product; lane l then takes the run's 16-byte input
// vectors l, l + 32, ... (four values of block 4 m + l / 8 each), so a warp's activation loads are
// contiguous.
template <int T>
__global__ void __launch_bounds__(kDownQ8Warps * 32)
    hc_down_narrow_q8_kernel(const float* __restrict__ normalized,
                             const std::uint8_t* __restrict__ down,
                             const std::uint8_t* __restrict__ inject, int rows,
                             float* __restrict__ low, float* __restrict__ inject_weights) {
    __shared__ __align__(16) std::uint8_t staged[kDownQ8Warps][kDownQ8Rows][kQ8RunBytes];
    __shared__ float partial[kDownQ8Warps][kDownQ8Rows][T];
    const int row0 = static_cast<int>(blockIdx.x) * kDownQ8Rows;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    Q8Run runs[kDownQ8Rows];
#pragma unroll
    for (int r = 0; r < kDownQ8Rows; ++r) {
        const int row = min(row0 + r, rows - 1);
        const std::uint8_t* base =
            row < kLowrank ? down + std::int64_t(row) * kDownQ8RowBytes
                           : inject + std::int64_t(row - kLowrank) * kDownQ8RowBytes;
        runs[r] = load_q8_run(base + std::int64_t(warp) * kQ8RunBytes, lane);
    }
#pragma unroll
    for (int r = 0; r < kDownQ8Rows; ++r) { store_q8_run(runs[r], staged[warp][r], lane); }
    __syncwarp();
    const float* x0 = normalized + warp * 32 * kQ8Values;
    float acc[kDownQ8Rows][T] = {};
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int vector = lane + 32 * m;
        float4 w[kDownQ8Rows];
#pragma unroll
        for (int r = 0; r < kDownQ8Rows; ++r) {
            w[r] = decode_q8x4(staged[warp][r] + (vector >> 3) * kQ8Bytes, vector & 7);
        }
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float4 x = *reinterpret_cast<const float4*>(x0 + std::int64_t(j) * kWidth + 4 * vector);
#pragma unroll
            for (int r = 0; r < kDownQ8Rows; ++r) { acc[r][j] = dot4(w[r], x, acc[r][j]); }
        }
    }
#pragma unroll
    for (int r = 0; r < kDownQ8Rows; ++r) {
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float v = warp_sum(acc[r][j]);
            if (lane == 0) { partial[warp][r][j] = v; }
        }
    }
    __syncthreads();
    if (threadIdx.x >= kDownQ8Rows * T) { return; }
    const int r = threadIdx.x / T, t = threadIdx.x % T;
    const int row = row0 + r;
    if (row >= rows) { return; }
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < kDownQ8Warps; ++w) { v += partial[w][r][t]; }
    v /= kStreams;
    if (row < kLowrank) {
        low[static_cast<std::int64_t>(t) * kLowrank + row] = v * sigmoid_f(v);
    } else {
        inject_weights[static_cast<std::int64_t>(t) * kStreams + row - kLowrank] =
            2.0f * sigmoid_f(v);
    }
}

// hc_up_mix_kernel over Q8_0 rows: warp c stages the eight rows c * hidden + d0.. of the CTA, the
// low-rank input staged in shared memory; lane part p of row r takes the row's 16-byte input
// vectors p, p + 4, ..., so the four lanes of a row read consecutive low-rank vectors.
template <int T>
__global__ void __launch_bounds__(kStreams * 32)
    hc_up_mix_q8_kernel(const float* __restrict__ normalized, const float* __restrict__ low,
                        const std::uint8_t* __restrict__ up, __nv_bfloat16* __restrict__ mixed) {
    constexpr int kLoads = (kUpQ8Vectors + 31) / 32;
    __shared__ __align__(16) std::uint8_t staged_rows[kStreams][kUpQ8Bytes];
    __shared__ __align__(16) float staged[T][kLowrank];
    __shared__ float gated[kStreams][kUpQ8Rows][T];
    const int c    = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane / kUpLanes, part = lane % kUpLanes;
    const int d0 = blockIdx.x * kUpQ8Rows;
    const auto* src =
        reinterpret_cast<const uint4*>(up + std::int64_t(c * kHidden + d0) * kUpQ8RowBytes);
    uint4 v[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int at = lane + 32 * i;
        v[i]         = at < kUpQ8Vectors ? __ldg(src + at) : uint4{};
    }
    for (int i = threadIdx.x; i < T * kLowrank / 4; i += kStreams * 32) {
        reinterpret_cast<float4*>(&staged[0][0])[i] = reinterpret_cast<const float4*>(low)[i];
    }
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int at = lane + 32 * i;
        if (at < kUpQ8Vectors) { reinterpret_cast<uint4*>(staged_rows[c])[at] = v[i]; }
    }
    __syncthreads();
    const std::uint8_t* row = staged_rows[c] + r * int(kUpQ8RowBytes);
    float acc[T] = {};
#pragma unroll
    for (int m = 0; m < kLowrank / 4 / kUpLanes; ++m) {
        const int vector = part + kUpLanes * m;
        const float4 w   = decode_q8x4(row + (vector >> 3) * kQ8Bytes, vector & 7);
#pragma unroll
        for (int j = 0; j < T; ++j) {
            acc[j] = dot4(w, *reinterpret_cast<const float4*>(&staged[j][4 * vector]), acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 1);
        acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 2);
        if (part == 0) {
            gated[c][r][j] = sigmoid_f(acc[j]) *
                             normalized[static_cast<std::int64_t>(j) * kWidth + c * kHidden + d0 + r];
        }
    }
    __syncthreads();
    if (threadIdx.x >= kUpQ8Rows * T) { return; }
    const int rr = threadIdx.x / T, j = threadIdx.x % T;
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) { sum += gated[s][rr][j]; }
    mixed[static_cast<std::int64_t>(j) * kHidden + d0 + rr] = __float2bfloat16_rn(sum / kStreams);
}

// The wide path's inject rows over Q8_0, kept in FP32 like hc_down_kernel's: CTA (c, chunk) takes
// row c for up to kColumnChunk tokens, warp w its run w, lane l the run's input vectors l + 32 m.
__global__ void __launch_bounds__(kDownQ8Warps * 32)
    hc_inject_q8_kernel(const float* __restrict__ normalized, const std::uint8_t* __restrict__ inject,
                        int tokens, float* __restrict__ inject_weights) {
    __shared__ __align__(16) std::uint8_t staged[kDownQ8Warps][kQ8RunBytes];
    __shared__ float partial[kDownQ8Warps][kColumnChunk];
    const int c     = blockIdx.x;
    const int t0    = blockIdx.y * kColumnChunk;
    const int count = min(kColumnChunk, tokens - t0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    store_q8_run(load_q8_run(inject + std::int64_t(c) * kDownQ8RowBytes +
                                 std::int64_t(warp) * kQ8RunBytes,
                             lane),
                 staged[warp], lane);
    __syncwarp();
    float4 w[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int vector = lane + 32 * m;
        w[m]             = decode_q8x4(staged[warp] + (vector >> 3) * kQ8Bytes, vector & 7);
    }
    const float* x0 = normalized + warp * 32 * kQ8Values;
#pragma unroll
    for (int j = 0; j < kColumnChunk; ++j) {
        float v = 0.0f;
        if (j < count) {
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                v = dot4(w[m],
                         *reinterpret_cast<const float4*>(x0 + std::int64_t(t0 + j) * kWidth +
                                                          4 * (lane + 32 * m)),
                         v);
            }
        }
        v = warp_sum(v);
        if (lane == 0) { partial[warp][j] = v; }
    }
    __syncthreads();
    if (warp != 0 || lane >= count) { return; }
    float v = 0.0f;
#pragma unroll
    for (int w8 = 0; w8 < kDownQ8Warps; ++w8) { v += partial[w8][lane]; }
    inject_weights[static_cast<std::int64_t>(t0 + lane) * kStreams + c] =
        2.0f * sigmoid_f(v / kStreams);
}

// The wide path's BF16 operand of a Q8_0 matrix: a warp per run of 32 blocks, block i's values
// d * q rounded to BF16 at out[32 i + k], each lane writing its block's 64 bytes.
__global__ void __launch_bounds__(256)
    hc_dequantize_q8_kernel(const std::uint8_t* __restrict__ q8, __nv_bfloat16* __restrict__ out,
                            std::int64_t runs) {
    __shared__ __align__(16) std::uint8_t staged[8][kQ8RunBytes];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const std::int64_t run = std::int64_t(blockIdx.x) * 8 + warp;
    if (run >= runs) { return; }
    store_q8_run(load_q8_run(q8 + run * kQ8RunBytes, lane), staged[warp], lane);
    __syncwarp();
    float w[kQ8Values];
    decode_q8(staged[warp] + lane * kQ8Bytes, w);
    auto* o = reinterpret_cast<uint4*>(out + (run * 32 + lane) * kQ8Values);
#pragma unroll
    for (int i = 0; i < kQ8Values / 8; ++i) {
        __nv_bfloat162 pairs[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            pairs[e] = __floats2bfloat162_rn(w[8 * i + 2 * e], w[8 * i + 2 * e + 1]);
        }
        o[i] = *reinterpret_cast<const uint4*>(pairs);
    }
}

// The narrow read's operands: `rows` of [down; inject] (the inject rows only with an inject).
struct NarrowRead {
    const float* normalized;
    const void* down;
    const void* inject;
    int rows;
    const void* up;
    bool down_q8; // down and inject
    bool up_q8;
    float* low;
    float* inject_out;
    __nv_bfloat16* mixed;
};

// The narrow read's two GEMVs for T tokens.
template <int T>
void narrow_products(const NarrowRead& a, cudaStream_t stream) {
    if (a.down_q8) {
        hc_down_narrow_q8_kernel<T>
            <<<div_up(a.rows, kDownQ8Rows), kDownQ8Warps * 32, 0, stream>>>(
            a.normalized, static_cast<const std::uint8_t*>(a.down),
            static_cast<const std::uint8_t*>(a.inject), a.rows, a.low, a.inject_out);
    } else {
        hc_down_narrow_kernel<T><<<div_up(a.rows, kDownRows), kDownWarps * 32, 0, stream>>>(
            a.normalized, static_cast<const __nv_bfloat16*>(a.down),
            static_cast<const __nv_bfloat16*>(a.inject), a.rows, a.low, a.inject_out);
    }
    CUDA_CHECK(cudaGetLastError());
    if (a.up_q8) {
        hc_up_mix_q8_kernel<T><<<kHidden / kUpQ8Rows, kStreams * 32, 0, stream>>>(
            a.normalized, a.low, static_cast<const std::uint8_t*>(a.up), a.mixed);
    } else {
        hc_up_mix_kernel<T><<<kHidden / kUpRows, kStreams * 32, 0, stream>>>(
            a.normalized, a.low, static_cast<const __nv_bfloat16*>(a.up), a.mixed);
    }
    CUDA_CHECK(cudaGetLastError());
}

// The wide path's activation of the down product, v FP32 [tokens][lowrank]: silu(v / n) in BF16,
// the up product's operand.
__global__ void __launch_bounds__(256)
    hc_low_kernel(const float* __restrict__ v, __nv_bfloat16* __restrict__ low,
                  std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float x = v[i] / kStreams;
    low[i]        = __float2bfloat16_rn(x * sigmoid_f(x));
}

// The wide path's mix over the up product's gate logits FP32 [tokens][streams * hidden]:
// mixed[d] = (1/n) sum_c sigmoid(gate_c) xn[c, d].
__global__ void __launch_bounds__(256)
    hc_mix_kernel(const float* __restrict__ normalized, const float* __restrict__ gates,
                  __nv_bfloat16* __restrict__ mixed, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / kHidden, d = i % kHidden;
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < kStreams; ++c) {
        const std::int64_t at = t * kWidth + c * kHidden + d;
        sum += sigmoid_f(gates[at]) * normalized[at];
    }
    mixed[i] = __float2bfloat16_rn(sum / kStreams);
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("hyper_connection: cuBLAS ") + what + " failed (" +
                                 std::to_string(static_cast<int>(status)) + ")");
    }
}

// One handle per device, created on first use and kept: the stream is set per call.
cublasHandle_t blas_handle() {
    static constexpr int kMaxDevices = 16;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("hyper_connection: device index out of range");
    }
    if (handles[device] == nullptr) { check_blas(cublasCreate(&handles[device]), "create"); }
    return handles[device];
}

// C FP32 (m x n, column-major, ldc = m) = A^T B for A BF16 stored k-contiguous per output row
// (k x m column-major) and B BF16 k-contiguous per token (k x n column-major).
void gemm_bf16(cudaStream_t stream, int m, int n, int k, const __nv_bfloat16* a,
               const __nv_bfloat16* b, float* c) {
    const cublasHandle_t handle = blas_handle();
    check_blas(cublasSetStream(handle, stream), "set stream");
    const float one = 1.0f, zero = 0.0f;
    check_blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, a, CUDA_R_16BF, k, b,
                            CUDA_R_16BF, k, &zero, c, CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "GEMM");
}

template <typename Output>
__global__ void __launch_bounds__(256)
    hc_write_kernel(float* __restrict__ stack, const Output* __restrict__ y,
                    const float* __restrict__ inject_weights, std::int64_t elements) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) { return; }
    const std::int64_t d      = i % kHidden;
    const std::int64_t column = i / kHidden; // t * streams + c
    const std::int64_t t      = column / kStreams;
    stack[i] = fmaf(to_float(y[t * kHidden + d]), inject_weights[column], stack[i]);
}

// `streams`, when given, holds each stream's own term in the stack's layout.
__global__ void __launch_bounds__(256)
    hc_expand_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ streams,
                     float* __restrict__ stack, std::int64_t elements) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    // stack element i is stream (i / hidden) % streams of token i / width.
    const std::int64_t token = i / kWidth;
    const std::int64_t d     = i % kHidden;
    const float shared       = __bfloat162float(x[token * kHidden + d]);
    stack[i]                 = streams == nullptr ? shared : __bfloat162float(streams[i]) + shared;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hyper_connection: ") + message); }
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// The kernels read the weights in 16-byte vectors.
void require_bf16(const Tensor* tensor, std::int32_t n0, std::int32_t n1, const char* name) {
    require(tensor != nullptr && tensor->data != nullptr && aligned16(tensor->data), name);
    require(tensor->dtype == DType::BF16 && tensor->is_contiguous() && tensor->ne[0] == n0 &&
                tensor->ne[1] == n1 && tensor->ne[2] == 1 && tensor->ne[3] == 1,
            name);
}

void require_matrix(const HyperConnectionMatrix& matrix, const char* name) {
    require(matrix.data != nullptr && aligned16(matrix.data), name);
}

bool q8(const HyperConnectionMatrix& matrix) {
    return matrix.format == HyperConnectionMatrix::Format::Q8_0;
}

// A Q8_0 matrix of `values` values as BF16 in `out`.
void dequantize(const HyperConnectionMatrix& matrix, std::int64_t values, __nv_bfloat16* out,
                cudaStream_t stream) {
    const std::int64_t runs = values / (32 * kQ8Values);
    hc_dequantize_q8_kernel<<<static_cast<unsigned>(div_up(runs, std::int64_t{8})), 256, 0,
                              stream>>>(static_cast<const std::uint8_t*>(matrix.data), out, runs);
    CUDA_CHECK(cudaGetLastError());
}

void require_geometry(std::int32_t streams, std::int32_t hidden, std::int32_t lowrank) {
    require(streams == kStreams && hidden == kHidden && lowrank == kLowrank,
            "unsupported geometry (streams 4, hidden 2560, lowrank 320 are implemented)");
}

void require_write_inputs(const Tensor& y, const Tensor& inject_weights, std::int32_t tokens) {
    require((y.dtype == DType::BF16 || y.dtype == DType::FP32) && y.is_contiguous() &&
                y.data != nullptr && y.ne[0] == kHidden && y.ne[1] == tokens &&
                y.ne[2] == 1 && y.ne[3] == 1,
            "y must be contiguous BF16 or FP32 [2560, tokens]");
    require(inject_weights.dtype == DType::FP32 && inject_weights.is_contiguous() &&
                inject_weights.data != nullptr && inject_weights.ne[0] == kStreams &&
                inject_weights.ne[1] == tokens && inject_weights.ne[2] == 1 &&
                inject_weights.ne[3] == 1,
            "inject weights must be contiguous FP32 [4, tokens]");
}

void normalize(const Tensor& stack, const Tensor* y, const Tensor* previous_inject,
               const Tensor& norm, float eps, float* normalized,
               __nv_bfloat16* normalized_bf16, cudaStream_t stream) {
    auto* xs = static_cast<float*>(stack.data);
    const auto* norm_p = static_cast<const __nv_bfloat16*>(norm.data);
    const dim3 grid(kStreams, stack.ne[2]);
    if (y == nullptr) {
        hc_norm_kernel<false, float><<<grid, kNormThreads, 0, stream>>>(
            xs, nullptr, nullptr, norm_p, eps, normalized, normalized_bf16);
    } else if (y->dtype == DType::FP32) {
        hc_norm_kernel<true><<<grid, kNormThreads, 0, stream>>>(
            xs, static_cast<const float*>(y->data), static_cast<const float*>(previous_inject->data),
            norm_p, eps, normalized, normalized_bf16);
    } else {
        hc_norm_kernel<true><<<grid, kNormThreads, 0, stream>>>(
            xs, static_cast<const __nv_bfloat16*>(y->data),
            static_cast<const float*>(previous_inject->data), norm_p, eps, normalized,
            normalized_bf16);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t hyper_connection_read_workspace_bytes(std::int32_t streams, std::int32_t hidden,
                                                  std::int32_t lowrank, std::int32_t tokens) {
    require_geometry(streams, hidden, lowrank);
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kWidth, tokens});
    (void)layout.alloc(DType::FP32, {kLowrank, tokens});
    (void)layout.alloc(DType::FP32, {kStreams, tokens});
    if (tokens > kColumnChunk) {
        (void)layout.alloc(DType::BF16, {kWidth, tokens});
        (void)layout.alloc(DType::BF16, {kLowrank, tokens});
        (void)layout.alloc(DType::FP32, {kWidth, tokens});
        // Q8_0 matrices' BF16 operands of the tensor-core products: down and up.
        (void)layout.alloc(DType::BF16, {kWidth, kLowrank});
        (void)layout.alloc(DType::BF16, {kLowrank, kWidth});
    }
    return layout.peak_bytes(1);
}

namespace {

void read(const Tensor& stack, const Tensor* y, const Tensor* previous_inject,
          const HyperConnectionWeights& weights, float eps, WorkspaceArena& workspace,
          Tensor& mixed, Tensor* inject_weights, cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                aligned16(stack.data) && stack.ne[0] == kHidden && stack.ne[1] == kStreams &&
                stack.ne[3] == 1,
            "stack must be contiguous 16-byte aligned FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    if (y != nullptr) { require_write_inputs(*y, *previous_inject, tokens); }
    require(eps > 0.0f, "eps must be positive");
    require_bf16(weights.norm, kWidth, 1, "norm must be BF16 [10240]");
    require_matrix(weights.down, "down must be a 16-byte aligned [10240, 320]");
    require_matrix(weights.up, "up must be a 16-byte aligned [320, 10240]");
    const bool has_inject = weights.inject.data != nullptr;
    require(has_inject == (inject_weights != nullptr),
            "inject weights and their output come together");
    require(!has_inject || q8(weights.inject) == q8(weights.down),
            "inject and down share a format");
    if (has_inject) {
        require_matrix(weights.inject, "inject must be a 16-byte aligned [10240, 4]");
        require(inject_weights->dtype == DType::FP32 && inject_weights->is_contiguous() &&
                    inject_weights->data != nullptr && inject_weights->ne[0] == kStreams &&
                    inject_weights->ne[1] == tokens,
                "inject output must be contiguous FP32 [4, tokens]");
    }
    require(mixed.dtype == DType::BF16 && mixed.is_contiguous() && mixed.data != nullptr &&
                mixed.ne[0] == kHidden && mixed.ne[1] == tokens,
            "mixed must be contiguous BF16 [2560, tokens]");

    auto scope         = workspace.scope();
    Tensor normalized  = workspace.alloc(DType::FP32, {kWidth, tokens});
    Tensor low         = workspace.alloc(DType::FP32, {kLowrank, tokens});
    Tensor scratch     = workspace.alloc(DType::FP32, {kStreams, tokens});
    float* inject_out  = inject_weights != nullptr ? static_cast<float*>(inject_weights->data)
                                                   : static_cast<float*>(scratch.data);

    auto* normalized_p   = static_cast<float*>(normalized.data);
    const int chunks     = div_up(tokens, kColumnChunk);
    if (tokens > kColumnChunk) {
        Tensor normalized_bf16 = workspace.alloc(DType::BF16, {kWidth, tokens});
        Tensor low_bf16        = workspace.alloc(DType::BF16, {kLowrank, tokens});
        Tensor products        = workspace.alloc(DType::FP32, {kWidth, tokens});
        auto* xn16             = static_cast<__nv_bfloat16*>(normalized_bf16.data);
        auto* low16            = static_cast<__nv_bfloat16*>(low_bf16.data);
        auto* products_p       = static_cast<float*>(products.data);
        // The tensor-core products take BF16 operands: Q8_0 matrices are rounded to BF16 first.
        const auto operand = [&](const HyperConnectionMatrix& matrix, std::int32_t n0,
                                 std::int32_t n1) -> const __nv_bfloat16* {
            if (!q8(matrix)) { return static_cast<const __nv_bfloat16*>(matrix.data); }
            auto* out = static_cast<__nv_bfloat16*>(workspace.alloc(DType::BF16, {n0, n1}).data);
            dequantize(matrix, std::int64_t(n0) * n1, out, stream);
            return out;
        };
        const __nv_bfloat16* down_w = operand(weights.down, kWidth, kLowrank);
        const __nv_bfloat16* up_w   = operand(weights.up, kLowrank, kWidth);
        normalize(stack, y, previous_inject, *weights.norm, eps, normalized_p, xn16, stream);
        if (has_inject && q8(weights.inject)) {
            hc_inject_q8_kernel<<<dim3(kStreams, chunks), kDownQ8Warps * 32, 0, stream>>>(
                normalized_p, static_cast<const std::uint8_t*>(weights.inject.data), tokens,
                inject_out);
            CUDA_CHECK(cudaGetLastError());
        } else if (has_inject) {
            hc_down_kernel<<<dim3(kStreams, chunks), kDownWarps * 32, 0, stream>>>(
                normalized_p, down_w, static_cast<const __nv_bfloat16*>(weights.inject.data),
                tokens, kLowrank, nullptr, inject_out);
            CUDA_CHECK(cudaGetLastError());
        }
        // v [tokens][lowrank] = down . xn, then low = silu(v / n); the gate logits
        // [tokens][streams * hidden] = up . low reuse the same plane.
        gemm_bf16(stream, kLowrank, tokens, kWidth, down_w, xn16, products_p);
        const std::int64_t low_count = std::int64_t(kLowrank) * tokens;
        hc_low_kernel<<<static_cast<unsigned>(div_up(low_count, std::int64_t{256})), 256, 0,
                        stream>>>(products_p, low16, low_count);
        CUDA_CHECK(cudaGetLastError());
        gemm_bf16(stream, kWidth, tokens, kLowrank, up_w, low16, products_p);
        const std::int64_t mixed_count = std::int64_t(kHidden) * tokens;
        hc_mix_kernel<<<static_cast<unsigned>(div_up(mixed_count, std::int64_t{256})), 256, 0,
                        stream>>>(normalized_p, products_p, static_cast<__nv_bfloat16*>(mixed.data),
                                  mixed_count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    normalize(stack, y, previous_inject, *weights.norm, eps, normalized_p, nullptr, stream);
    const NarrowRead read{.normalized = normalized_p,
                          .down       = weights.down.data,
                          .inject     = weights.inject.data,
                          .rows       = kLowrank + (has_inject ? kStreams : 0),
                          .up         = weights.up.data,
                          .down_q8    = q8(weights.down),
                          .up_q8      = q8(weights.up),
                          .low        = static_cast<float*>(low.data),
                          .inject_out = inject_out,
                          .mixed      = static_cast<__nv_bfloat16*>(mixed.data)};
    switch (tokens) {
    case 1: return narrow_products<1>(read, stream);
    case 2: return narrow_products<2>(read, stream);
    case 3: return narrow_products<3>(read, stream);
    case 4: return narrow_products<4>(read, stream);
    case 5: return narrow_products<5>(read, stream);
    case 6: return narrow_products<6>(read, stream);
    case 7: return narrow_products<7>(read, stream);
    default: return narrow_products<8>(read, stream);
    }
}

} // namespace

void hyper_connection_read(const Tensor& stack, const HyperConnectionWeights& weights, float eps,
                           WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                           cudaStream_t stream) {
    read(stack, nullptr, nullptr, weights, eps, workspace, mixed, inject_weights, stream);
}

void hyper_connection_write_read(Tensor& stack, const Tensor& y, const Tensor& previous_inject,
                                 const HyperConnectionWeights& weights, float eps,
                                 WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                                 cudaStream_t stream) {
    read(stack, &y, &previous_inject, weights, eps, workspace, mixed, inject_weights, stream);
}

void hyper_connection_write(Tensor& stack, const Tensor& y, const Tensor& inject_weights,
                            cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require_write_inputs(y, inject_weights, tokens);
    const std::int64_t elements = static_cast<std::int64_t>(kWidth) * tokens;
    const auto blocks = static_cast<unsigned>(div_up(elements, std::int64_t{256}));
    if (y.dtype == DType::FP32) {
        hc_write_kernel<<<blocks, 256, 0, stream>>>(static_cast<float*>(stack.data),
                                                    static_cast<const float*>(y.data),
                                                    static_cast<const float*>(inject_weights.data),
                                                    elements);
    } else {
        hc_write_kernel<<<blocks, 256, 0, stream>>>(
            static_cast<float*>(stack.data), static_cast<const __nv_bfloat16*>(y.data),
            static_cast<const float*>(inject_weights.data), elements);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace {

void expand(const Tensor& x, const Tensor* streams, Tensor& stack, cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.data != nullptr &&
                x.ne[0] == kHidden && x.ne[1] == tokens,
            "x must be contiguous BF16 [2560, tokens]");
    if (streams != nullptr) {
        require(streams->dtype == DType::BF16 && streams->is_contiguous() &&
                    streams->data != nullptr && streams->ne[0] == kHidden &&
                    streams->ne[1] == kStreams && streams->ne[2] == tokens && streams->ne[3] == 1,
                "streams must be contiguous BF16 [2560, 4, tokens]");
    }
    const std::int64_t elements = static_cast<std::int64_t>(kWidth) * tokens;
    hc_expand_kernel<<<static_cast<unsigned>(div_up(elements, std::int64_t{256})), 256, 0,
                       stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        streams != nullptr ? static_cast<const __nv_bfloat16*>(streams->data) : nullptr,
        static_cast<float*>(stack.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void hyper_connection_expand(const Tensor& x, Tensor& stack, cudaStream_t stream) {
    expand(x, nullptr, stack, stream);
}

void hyper_connection_expand(const Tensor& x, const Tensor& streams, Tensor& stack,
                             cudaStream_t stream) {
    expand(x, &streams, stack, stream);
}

} // namespace ninfer::ops
