// ninfer::ops - hyper-connection read/write of Qwen3.8-Flash-Next (contract in
// include/ninfer/ops/hyper_connection.h). Four launches: per-stream RMSNorm into FP32 workspace,
// the down and inject GEMVs with their activations, the up GEMV with the gated stream mix, and the
// weighted write. Every product accumulates in FP32. Up to eight tokens, the two GEMVs stream their
// 6.5 MB of weights each in 16-byte vectors, specialized for the token count: every thread issues
// all of its weight loads before its first product, so a whole matrix is in flight at once; the
// down rows split K over a CTA's warps, two rows a CTA sharing each activation load, and the up rows
// of eight hidden indices are one contiguous run per stream, their low-rank input staged in shared
// memory. Wider calls run the down and up products as cuBLAS tensor-core GEMMs over BF16 operands
// (the normalized stack and the low-rank activation rounded to BF16, as the checkpoint's own
// arithmetic keeps them) and keep the four inject rows on the FP32 GEMV.
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
// and its BF16 rounding when asked for.
__global__ void __launch_bounds__(kNormThreads)
    hc_norm_kernel(const float* __restrict__ stack, const __nv_bfloat16* __restrict__ norm,
                   float eps, float* __restrict__ normalized,
                   __nv_bfloat16* __restrict__ normalized_bf16) {
    const int c             = blockIdx.x;
    const int t             = blockIdx.y;
    const std::int64_t base = (static_cast<std::int64_t>(t) * kStreams + c) * kHidden;
    __shared__ float partial[kNormThreads / 32];
    const float4 x = reinterpret_cast<const float4*>(stack + base)[threadIdx.x];
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

// The narrow read's operands: `rows` of [down; inject] (the inject rows only with an inject).
struct NarrowRead {
    const float* normalized;
    const __nv_bfloat16* down;
    const __nv_bfloat16* inject;
    int rows;
    const __nv_bfloat16* up;
    float* low;
    float* inject_out;
    __nv_bfloat16* mixed;
};

// The narrow read's two GEMVs for T tokens.
template <int T>
void narrow_products(const NarrowRead& a, cudaStream_t stream) {
    hc_down_narrow_kernel<T><<<div_up(a.rows, kDownRows), kDownWarps * 32, 0, stream>>>(
        a.normalized, a.down, a.inject, a.rows, a.low, a.inject_out);
    CUDA_CHECK(cudaGetLastError());
    hc_up_mix_kernel<T>
        <<<kHidden / kUpRows, kStreams * 32, 0, stream>>>(a.normalized, a.low, a.up, a.mixed);
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

__device__ __forceinline__ float to_float(float value) { return value; }
__device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

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

__global__ void __launch_bounds__(256)
    hc_expand_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ stack,
                     std::int64_t elements) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    // stack element i is stream (i / hidden) % streams of token i / width.
    const std::int64_t token = i / kWidth;
    const std::int64_t d     = i % kHidden;
    stack[i]                 = __bfloat162float(x[token * kHidden + d]);
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

void require_geometry(std::int32_t streams, std::int32_t hidden, std::int32_t lowrank) {
    require(streams == kStreams && hidden == kHidden && lowrank == kLowrank,
            "unsupported geometry (streams 4, hidden 2560, lowrank 320 are implemented)");
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
    }
    return layout.peak_bytes(1);
}

void hyper_connection_read(const Tensor& stack, const HyperConnectionWeights& weights, float eps,
                           WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                           cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                aligned16(stack.data) && stack.ne[0] == kHidden && stack.ne[1] == kStreams &&
                stack.ne[3] == 1,
            "stack must be contiguous 16-byte aligned FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require(eps > 0.0f, "eps must be positive");
    require_bf16(weights.norm, kWidth, 1, "norm must be BF16 [10240]");
    require_bf16(weights.down, kWidth, kLowrank, "down must be BF16 [10240, 320]");
    require_bf16(weights.up, kLowrank, kWidth, "up must be BF16 [320, 10240]");
    require((weights.inject == nullptr) == (inject_weights == nullptr),
            "inject weights and their output come together");
    if (weights.inject != nullptr) {
        require_bf16(weights.inject, kWidth, kStreams, "inject must be BF16 [10240, 4]");
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

    const auto* down_w   = static_cast<const __nv_bfloat16*>(weights.down->data);
    const auto* inject_w = weights.inject != nullptr
                               ? static_cast<const __nv_bfloat16*>(weights.inject->data)
                               : nullptr;
    auto* normalized_p   = static_cast<float*>(normalized.data);
    const int chunks     = div_up(tokens, kColumnChunk);
    if (tokens > kColumnChunk) {
        Tensor normalized_bf16 = workspace.alloc(DType::BF16, {kWidth, tokens});
        Tensor low_bf16        = workspace.alloc(DType::BF16, {kLowrank, tokens});
        Tensor products        = workspace.alloc(DType::FP32, {kWidth, tokens});
        auto* xn16             = static_cast<__nv_bfloat16*>(normalized_bf16.data);
        auto* low16            = static_cast<__nv_bfloat16*>(low_bf16.data);
        auto* products_p       = static_cast<float*>(products.data);
        hc_norm_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
            static_cast<const float*>(stack.data),
            static_cast<const __nv_bfloat16*>(weights.norm->data), eps, normalized_p, xn16);
        CUDA_CHECK(cudaGetLastError());
        if (inject_w != nullptr) {
            hc_down_kernel<<<dim3(kStreams, chunks), kDownWarps * 32, 0, stream>>>(
                normalized_p, down_w, inject_w, tokens, kLowrank, nullptr, inject_out);
            CUDA_CHECK(cudaGetLastError());
        }
        // v [tokens][lowrank] = down . xn, then low = silu(v / n); the gate logits
        // [tokens][streams * hidden] = up . low reuse the same plane.
        gemm_bf16(stream, kLowrank, tokens, kWidth, down_w, xn16, products_p);
        const std::int64_t low_count = std::int64_t(kLowrank) * tokens;
        hc_low_kernel<<<static_cast<unsigned>(div_up(low_count, std::int64_t{256})), 256, 0,
                        stream>>>(products_p, low16, low_count);
        CUDA_CHECK(cudaGetLastError());
        gemm_bf16(stream, kWidth, tokens, kLowrank,
                  static_cast<const __nv_bfloat16*>(weights.up->data), low16, products_p);
        const std::int64_t mixed_count = std::int64_t(kHidden) * tokens;
        hc_mix_kernel<<<static_cast<unsigned>(div_up(mixed_count, std::int64_t{256})), 256, 0,
                        stream>>>(normalized_p, products_p, static_cast<__nv_bfloat16*>(mixed.data),
                                  mixed_count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    hc_norm_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
        static_cast<const float*>(stack.data),
        static_cast<const __nv_bfloat16*>(weights.norm->data), eps, normalized_p, nullptr);
    CUDA_CHECK(cudaGetLastError());
    const NarrowRead read{.normalized = normalized_p,
                          .down       = down_w,
                          .inject     = inject_w,
                          .rows       = kLowrank + (inject_w != nullptr ? kStreams : 0),
                          .up         = static_cast<const __nv_bfloat16*>(weights.up->data),
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

void hyper_connection_write(Tensor& stack, const Tensor& y, const Tensor& inject_weights,
                            cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require((y.dtype == DType::BF16 || y.dtype == DType::FP32) && y.is_contiguous() &&
                y.data != nullptr && y.ne[0] == kHidden && y.ne[1] == tokens,
            "y must be contiguous BF16 or FP32 [2560, tokens]");
    require(inject_weights.dtype == DType::FP32 && inject_weights.is_contiguous() &&
                inject_weights.data != nullptr && inject_weights.ne[0] == kStreams &&
                inject_weights.ne[1] == tokens,
            "inject weights must be contiguous FP32 [4, tokens]");
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

void hyper_connection_expand(const Tensor& x, Tensor& stack, cudaStream_t stream) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.data != nullptr &&
                x.ne[0] == kHidden && x.ne[1] == tokens,
            "x must be contiguous BF16 [2560, tokens]");
    const std::int64_t elements = static_cast<std::int64_t>(kWidth) * tokens;
    hc_expand_kernel<<<static_cast<unsigned>(div_up(elements, std::int64_t{256})), 256, 0,
                       stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                 static_cast<float*>(stack.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
