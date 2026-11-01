// ninfer::ops - Qwen3.8-Flash-Next MoE router (contract in include/ninfer/ops/moe_route.h).
// Two steps: the E + 1 logits (the experts and the shared gate) as a GEMV with one warp per row
// and the router streamed in 16-byte vectors by every CTA of the device (or, for a BF16 input of
// more than eight tokens, as cuBLAS tensor-core GEMMs with FP32 accumulation), then one warp per
// token taking ten rounds of an arg-max over its logits that prefers the lower index on ties
// (softmax is monotonic).
#include "ninfer/ops/moe_route.h"

#include "core/device.h"
#include "core/layout.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHidden     = 2560;
constexpr int kMaxExperts = 512;
constexpr int kTopK       = 10;
// The logits GEMV: one warp per row, four rows and eight token columns per CTA, so a decode
// step's 513 rows spread over every multiprocessor.
constexpr int kRowWarps    = 4;
constexpr int kColumnChunk = 8;
constexpr int kVectors     = kHidden / 8; // 16-byte vectors per router row
constexpr int kVectorsPerLane = kVectors / 32;
static_assert(kVectors % 32 == 0);
// The selection: one warp per token, sixteen logits per lane.
constexpr int kSelectWarps   = 4;
constexpr int kLogitsPerLane = kMaxExperts / 32;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&out)[8]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

template <typename Input>
__device__ __forceinline__ void load_input(const Input* p, float (&out)[8]) {
    if constexpr (std::is_same_v<Input, float>) {
        const float4 a = *reinterpret_cast<const float4*>(p);
        const float4 b = *reinterpret_cast<const float4*>(p + 4);
        out[0] = a.x, out[1] = a.y, out[2] = a.z, out[3] = a.w;
        out[4] = b.x, out[5] = b.y, out[6] = b.z, out[7] = b.w;
    } else {
        unpack_bf16x8(*reinterpret_cast<const uint4*>(p), out);
    }
}

// logits[t, row] for rows 0..E (row E the shared gate), FP32 [E + 1, tokens].
template <typename Input>
__global__ void __launch_bounds__(kRowWarps * 32)
    moe_route_logits_kernel(const Input* __restrict__ m, const __nv_bfloat16* __restrict__ router,
                            const __nv_bfloat16* __restrict__ shared_gate, int experts, int tokens,
                            float* __restrict__ logits) {
    const int row   = blockIdx.x * kRowWarps + (threadIdx.x >> 5);
    const int lane  = threadIdx.x & 31;
    const int t0    = blockIdx.y * kColumnChunk;
    const int count = min(kColumnChunk, tokens - t0);
    if (row > experts) { return; }
    const auto* w = reinterpret_cast<const uint4*>(
        row < experts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate);
    // Every weight load of the lane is in flight before the first product.
    uint4 wv[kVectorsPerLane];
#pragma unroll
    for (int i = 0; i < kVectorsPerLane; ++i) { wv[i] = __ldg(w + lane + 32 * i); }
    float acc[kColumnChunk] = {};
#pragma unroll
    for (int i = 0; i < kVectorsPerLane; ++i) {
        const int v = lane + 32 * i;
        float wf[8];
        unpack_bf16x8(wv[i], wf);
#pragma unroll
        for (int j = 0; j < kColumnChunk; ++j) {
            if (j < count) {
                float x[8];
                load_input(m + static_cast<std::int64_t>(t0 + j) * kHidden + 8 * v, x);
#pragma unroll
                for (int e = 0; e < 8; ++e) { acc[j] = fmaf(wf[e], x[e], acc[j]); }
            }
        }
    }
    float mine = 0.0f;
#pragma unroll
    for (int j = 0; j < kColumnChunk; ++j) {
        if (j < count) { // uniform across the warp
            const float v = warp_sum(acc[j]);
            if (j == lane) { mine = v; }
        }
    }
    if (lane < count) { logits[static_cast<std::int64_t>(t0 + lane) * (experts + 1) + row] = mine; }
}

// The better of two (logit, expert) candidates: the larger logit, the lower expert on a tie.
__device__ __forceinline__ bool better(float value, int index, float other, int other_index) {
    return value > other || (value == other && index < other_index);
}

__global__ void __launch_bounds__(kSelectWarps * 32)
    moe_route_select_kernel(const float* __restrict__ logits, int experts, int tokens,
                            int* __restrict__ ids, float* __restrict__ weights,
                            float* __restrict__ shared) {
    const int t    = blockIdx.x * kSelectWarps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (t >= tokens) { return; }
    const float* row = logits + static_cast<std::int64_t>(t) * (experts + 1);
    // Lane l holds experts l, l + 32, ...: ascending within a lane.
    float values[kLogitsPerLane];
#pragma unroll
    for (int i = 0; i < kLogitsPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? row[e] : -INFINITY;
    }
    float chosen[kTopK];
    int chosen_index[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        float best = -INFINITY;
        int slot   = 0;
#pragma unroll
        for (int i = 0; i < kLogitsPerLane; ++i) {
            if (values[i] > best) {
                best = values[i];
                slot = i;
            }
        }
        // A lane with nothing left offers an index past every expert.
        int index = best == -INFINITY ? kMaxExperts + lane : lane + 32 * slot;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other     = __shfl_xor_sync(0xffffffffu, best, offset);
            const int other_index = __shfl_xor_sync(0xffffffffu, index, offset);
            if (better(other, other_index, best, index)) {
                best  = other;
                index = other_index;
            }
        }
        chosen[k]       = best;
        chosen_index[k] = index;
        if (index == lane + 32 * slot) {
#pragma unroll
            for (int i = 0; i < kLogitsPerLane; ++i) {
                if (i == slot) { values[i] = -INFINITY; }
            }
        }
    }
    if (lane != 0) { return; }
    shared[t] = 1.0f / (1.0f + __expf(-row[experts]));
    // Renormalised softmax over the chosen ten: the full softmax's denominator cancels.
    float sum = 0.0f;
    float e[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) { sum += (e[k] = __expf(chosen[k] - chosen[0])); }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        ids[t * kTopK + k]     = chosen_index[k];
        weights[t * kTopK + k] = e[k] / sum;
    }
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// One handle per device, created on first use and kept: the stream is set per call.
cublasHandle_t blas_handle() {
    static constexpr int kMaxDevices = 16;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("moe_route: device index out of range");
    }
    if (handles[device] == nullptr && cublasCreate(&handles[device]) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("moe_route: cublasCreate failed");
    }
    return handles[device];
}

// logits rows [first, first + rows) of every token (column-major, ldc = experts + 1) = W . m for
// W BF16 [rows][2560] and m BF16 [tokens][2560].
void logits_gemm(cudaStream_t stream, const __nv_bfloat16* w, int rows, const __nv_bfloat16* m,
                 int tokens, float* logits, int ldc) {
    const cublasHandle_t handle = blas_handle();
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("moe_route: cublasSetStream failed");
    }
    const float one = 1.0f, zero = 0.0f;
    if (cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, kHidden, &one, w, CUDA_R_16BF,
                     kHidden, m, CUDA_R_16BF, kHidden, &zero, logits, CUDA_R_32F, ldc,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("moe_route: logits GEMM failed");
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_route: ") + message); }
}

} // namespace

std::size_t moe_route_workspace_bytes(std::int32_t tokens, std::int32_t experts) {
    require(tokens > 0 && experts >= kTopK && experts <= kMaxExperts,
            "tokens must be positive and experts in [10, 512]");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {experts + 1, tokens});
    return layout.peak_bytes(1);
}

void moe_route(const Tensor& m, const Tensor& router, const Tensor& shared_gate,
               WorkspaceArena& workspace, Tensor& ids, Tensor& weights, Tensor& shared,
               cudaStream_t stream) {
    require((m.dtype == DType::BF16 || m.dtype == DType::FP32) && m.is_contiguous() &&
                m.data != nullptr && aligned16(m.data) && m.ne[0] == kHidden && m.ne[1] > 0 &&
                m.ne[2] == 1,
            "m must be contiguous 16-byte aligned BF16 or FP32 [2560, tokens]");
    const std::int32_t tokens  = m.ne[1];
    const std::int32_t experts = router.ne[1];
    require(router.dtype == DType::BF16 && router.is_contiguous() && router.data != nullptr &&
                aligned16(router.data) && router.ne[0] == kHidden && experts >= kTopK &&
                experts <= kMaxExperts && router.ne[2] == 1,
            "router must be 16-byte aligned BF16 [2560, experts] with 10 to 512 experts");
    require(shared_gate.dtype == DType::BF16 && shared_gate.is_contiguous() &&
                shared_gate.data != nullptr && aligned16(shared_gate.data) &&
                shared_gate.ne[0] == kHidden && shared_gate.ne[1] == 1,
            "shared_gate must be 16-byte aligned BF16 [2560]");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.data != nullptr &&
                ids.ne[0] == kTopK && ids.ne[1] == tokens,
            "ids must be I32 [10, tokens]");
    require(weights.dtype == DType::FP32 && weights.is_contiguous() && weights.data != nullptr &&
                weights.ne[0] == kTopK && weights.ne[1] == tokens,
            "weights must be FP32 [10, tokens]");
    require(shared.dtype == DType::FP32 && shared.is_contiguous() && shared.data != nullptr &&
                shared.ne[0] == tokens,
            "shared must be FP32 [tokens]");
    auto scope     = workspace.scope();
    Tensor logits  = workspace.alloc(DType::FP32, {experts + 1, tokens});
    const auto* r  = static_cast<const __nv_bfloat16*>(router.data);
    const auto* g  = static_cast<const __nv_bfloat16*>(shared_gate.data);
    auto* logits_p = static_cast<float*>(logits.data);
    const dim3 grid(static_cast<unsigned>((experts + 1 + kRowWarps - 1) / kRowWarps),
                    static_cast<unsigned>((tokens + kColumnChunk - 1) / kColumnChunk));
    if (m.dtype == DType::FP32) {
        moe_route_logits_kernel<float><<<grid, kRowWarps * 32, 0, stream>>>(
            static_cast<const float*>(m.data), r, g, experts, tokens, logits_p);
    } else if (tokens > kColumnChunk) {
        const auto* x = static_cast<const __nv_bfloat16*>(m.data);
        logits_gemm(stream, r, experts, x, tokens, logits_p, experts + 1);
        logits_gemm(stream, g, 1, x, tokens, logits_p + experts, experts + 1);
    } else {
        moe_route_logits_kernel<__nv_bfloat16><<<grid, kRowWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(m.data), r, g, experts, tokens, logits_p);
    }
    CUDA_CHECK(cudaGetLastError());
    moe_route_select_kernel<<<static_cast<unsigned>((tokens + kSelectWarps - 1) / kSelectWarps),
                              kSelectWarps * 32, 0, stream>>>(
        logits_p, experts, tokens, static_cast<int*>(ids.data), static_cast<float*>(weights.data),
        static_cast<float*>(shared.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
