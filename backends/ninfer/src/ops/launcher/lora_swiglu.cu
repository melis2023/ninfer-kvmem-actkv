// Implements: include/ninfer/ops/lora_swiglu.h
// Two launches: one rank reduction for both projections, one delta-and-product epilogue whose
// column tile is a compile-time constant.
#include "ops/launcher/lora_swiglu.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/lora_swiglu.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

int grid_ceil(std::int64_t work, std::int64_t per_block) {
    return static_cast<int>(std::max<std::int64_t>(1, (work + per_block - 1) / per_block));
}

template <int COLUMNS>
void launch_epilogue(const float* scratch, const __half* b_gate, const __half* b_up,
                     const __nv_bfloat16* plane, __nv_bfloat16* activation, int rows, int rank,
                     int tokens, int stride, int splits, int plane_stride, float scale_gate,
                     float scale_up, cudaStream_t stream) {
    const std::size_t shared_bytes = static_cast<std::size_t>(2) * rank * COLUMNS * sizeof(float);
    const dim3 grid(grid_ceil(rows, kLoraSwiGLURows), grid_ceil(tokens, COLUMNS));
    lora_swiglu_kernel<COLUMNS><<<grid, kLoraSwiGLURows, shared_bytes, stream>>>(
        scratch, b_gate, b_up, plane, activation, rows, rank, tokens, stride, splits, plane_stride,
        scale_gate, scale_up);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void lora_swiglu_launch(const Tensor& x, const Tensor& a_gate, const Tensor& b_gate,
                        const Tensor& a_up, const Tensor& b_up, const Tensor& plane,
                        Tensor& activation, const Tensor& scratch, float scale_gate,
                        float scale_up, cudaStream_t stream) {
    const int inner       = x.ne[0];
    const int tokens      = x.ne[1];
    const int rank        = a_gate.ne[1];
    const int rows        = activation.ne[0];
    const int stride      = scratch.ne[0];
    const int splits      = std::max(1, stride / (2 * rank));
    const int plane_stride = static_cast<int>(plane.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    auto* scratch_ptr     = static_cast<float*>(scratch.data);

    const dim3 block1(kLoraSwiGLUTileColumns, kLoraSwiGLUTileInner);
    const dim3 grid1(2 * rank, splits, grid_ceil(tokens, kLoraSwiGLUTileColumns));
    lora_swiglu_pass1_kernel<<<grid1, block1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __half*>(a_gate.data),
        static_cast<const __half*>(a_up.data), scratch_ptr, rank, inner, tokens, stride, splits);
    CUDA_CHECK(cudaGetLastError());

    const auto* bg  = static_cast<const __half*>(b_gate.data);
    const auto* bu  = static_cast<const __half*>(b_up.data);
    const auto* p   = static_cast<const __nv_bfloat16*>(plane.data);
    auto* act       = static_cast<__nv_bfloat16*>(activation.data);
    if (tokens <= 1) {
        launch_epilogue<1>(scratch_ptr, bg, bu, p, act, rows, rank, tokens, stride, splits,
                           plane_stride, scale_gate, scale_up, stream);
    } else if (tokens <= 2) {
        launch_epilogue<2>(scratch_ptr, bg, bu, p, act, rows, rank, tokens, stride, splits,
                           plane_stride, scale_gate, scale_up, stream);
    } else if (tokens <= 4) {
        launch_epilogue<4>(scratch_ptr, bg, bu, p, act, rows, rank, tokens, stride, splits,
                           plane_stride, scale_gate, scale_up, stream);
    } else if (tokens <= 8) {
        launch_epilogue<8>(scratch_ptr, bg, bu, p, act, rows, rank, tokens, stride, splits,
                           plane_stride, scale_gate, scale_up, stream);
    } else if (tokens <= 16) {
        launch_epilogue<16>(scratch_ptr, bg, bu, p, act, rows, rank, tokens, stride, splits,
                            plane_stride, scale_gate, scale_up, stream);
    } else {
        launch_epilogue<32>(scratch_ptr, bg, bu, p, act, rows, rank, tokens, stride, splits,
                            plane_stride, scale_gate, scale_up, stream);
    }
}

} // namespace ninfer::ops::detail
