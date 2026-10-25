// ninfer::ops - Qwen3.8-Flash-Next MoE experts over BF16 weights (contract in
// include/ninfer/ops/moe_experts.h). Two launches: every (token, expert) pair's SwiGLU middle into
// FP32 workspace, one warp per middle element; then each output row's weighted sum over the ten
// routed experts and the shared one, one warp per row.
#include "ninfer/ops/moe_experts.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHidden  = 2560;
constexpr int kWidth   = 640;
constexpr int kExperts = 512;
constexpr int kTopK    = 10;
constexpr int kSlots   = kTopK + 1; // the shared expert is the last slot
constexpr int kWarps   = 8;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float dot_row(const __nv_bfloat16* __restrict__ row,
                                         const __nv_bfloat16* __restrict__ x, int lane) {
    float acc = 0.0f;
    for (int k = 2 * lane; k < kHidden; k += 64) {
        const float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(row + k));
        const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + k));
        acc = fmaf(w.x, v.x, fmaf(w.y, v.y, acc));
    }
    return warp_sum(acc);
}

// grid (kWidth / kWarps, kSlots, tokens): middle[t][slot][i] = silu(gate_i . m) * (up_i . m).
__global__ void __launch_bounds__(kWarps * 32)
    moe_middle_kernel(const __nv_bfloat16* __restrict__ m, const int* __restrict__ ids,
                      const __nv_bfloat16* __restrict__ gate_up,
                      const __nv_bfloat16* __restrict__ shared_gate_up,
                      float* __restrict__ middle) {
    const int i    = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int slot = blockIdx.y;
    const int t    = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* bank =
        slot < kTopK ? gate_up + static_cast<std::int64_t>(ids[t * kTopK + slot]) * 2 * kWidth * kHidden
                     : shared_gate_up;
    const __nv_bfloat16* x = m + static_cast<std::int64_t>(t) * kHidden;
    const float gate = dot_row(bank + static_cast<std::int64_t>(i) * kHidden, x, lane);
    const float up   = dot_row(bank + static_cast<std::int64_t>(kWidth + i) * kHidden, x, lane);
    if (lane == 0) {
        middle[(static_cast<std::int64_t>(t) * kSlots + slot) * kWidth + i] =
            gate / (1.0f + __expf(-gate)) * up;
    }
}

// grid (kHidden / kWarps, tokens): y[t][d] = sum_slot scale_slot * (down_slot[d] . middle_slot).
__global__ void __launch_bounds__(kWarps * 32)
    moe_down_kernel(const int* __restrict__ ids, const float* __restrict__ weights,
                    const float* __restrict__ shared, const __nv_bfloat16* __restrict__ down,
                    const __nv_bfloat16* __restrict__ shared_down,
                    const float* __restrict__ middle, float* __restrict__ y) {
    const int d    = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const int t    = blockIdx.y;
    const int lane = threadIdx.x & 31;
    float total    = 0.0f;
    for (int slot = 0; slot < kSlots; ++slot) {
        const __nv_bfloat16* row =
            (slot < kTopK ? down + static_cast<std::int64_t>(ids[t * kTopK + slot]) * kHidden * kWidth
                          : shared_down) +
            static_cast<std::int64_t>(d) * kWidth;
        const float* h = middle + (static_cast<std::int64_t>(t) * kSlots + slot) * kWidth;
        float acc      = 0.0f;
        for (int k = 2 * lane; k < kWidth; k += 64) {
            const float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(row + k));
            acc = fmaf(w.x, h[k], fmaf(w.y, h[k + 1], acc));
        }
        const float scale = slot < kTopK ? weights[t * kTopK + slot] : shared[t];
        total             = fmaf(scale, warp_sum(acc), total);
    }
    if (lane == 0) y[static_cast<std::int64_t>(t) * kHidden + d] = total;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_experts_bf16: ") + message); }
}

bool shaped(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2 = 1) {
    return tensor.dtype == dtype && tensor.is_contiguous() && tensor.data != nullptr &&
           tensor.ne[0] == n0 && tensor.ne[1] == n1 && tensor.ne[2] == n2 && tensor.ne[3] == 1;
}

} // namespace

std::size_t moe_experts_bf16_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kWidth, kSlots, tokens});
    return layout.peak_bytes(1);
}

void moe_experts_bf16(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const Tensor& gate_up, const Tensor& down,
                      const Tensor& shared_gate_up, const Tensor& shared_down,
                      WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(shaped(weights, DType::FP32, kTopK, tokens), "weights must be FP32 [10, tokens]");
    require(shaped(shared, DType::FP32, tokens, 1), "shared must be FP32 [tokens]");
    require(shaped(gate_up, DType::BF16, kHidden, 2 * kWidth, kExperts),
            "gate_up must be BF16 [2560, 1280, 512]");
    require(shaped(down, DType::BF16, kWidth, kHidden, kExperts), "down must be BF16 [640, 2560, 512]");
    require(shaped(shared_gate_up, DType::BF16, kHidden, 2 * kWidth),
            "shared_gate_up must be BF16 [2560, 1280]");
    require(shaped(shared_down, DType::BF16, kWidth, kHidden), "shared_down must be BF16 [640, 2560]");
    require(shaped(y, DType::FP32, kHidden, tokens), "y must be FP32 [2560, tokens]");
    auto scope    = workspace.scope();
    Tensor middle = workspace.alloc(DType::FP32, {kWidth, kSlots, tokens});
    moe_middle_kernel<<<dim3(kWidth / kWarps, kSlots, tokens), kWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(m.data), static_cast<const int*>(ids.data),
        static_cast<const __nv_bfloat16*>(gate_up.data),
        static_cast<const __nv_bfloat16*>(shared_gate_up.data), static_cast<float*>(middle.data));
    CUDA_CHECK(cudaGetLastError());
    moe_down_kernel<<<dim3(kHidden / kWarps, tokens), kWarps * 32, 0, stream>>>(
        static_cast<const int*>(ids.data), static_cast<const float*>(weights.data),
        static_cast<const float*>(shared.data), static_cast<const __nv_bfloat16*>(down.data),
        static_cast<const __nv_bfloat16*>(shared_down.data), static_cast<const float*>(middle.data),
        static_cast<float*>(y.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
