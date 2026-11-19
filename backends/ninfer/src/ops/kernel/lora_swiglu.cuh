#pragma once

// Implements: include/ninfer/ops/lora_swiglu.h
//
// Two kernels, both over the same column tile shape the standalone lora_accum uses:
//
//   pass 1  one launch for BOTH projections: grid.x selects (projection, rank row), so the gate and
//           up rank reductions share a launch and each `x` column is read once per projection.
//           Result: scratch[projection * rank*splits + split*rank + rr, column].
//   epilogue one thread per output row over a COLUMNS-wide tile: stages both projections' folded
//           vectors in shared memory, adds each rank reduction to its pre-activation row, and
//           applies SiLU * up in the same pass. The plane is read once; no delta is ever written
//           back into it, so the fused route's memory traffic is recovered.
//
// Silu is `x / (1 + exp(-x))` taken from ops/common/math.cuh, matching silu_mul exactly.

#include "ops/common/math.cuh"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

inline constexpr int kLoraSwiGLUTileColumns = 4;
inline constexpr int kLoraSwiGLUTileInner   = 64;
inline constexpr int kLoraSwiGLURows        = 128;

// pass 1 for both projections. blockIdx.x in [0, 2*rank): projection 0 = gate, 1 = up.
__global__ void lora_swiglu_pass1_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __half* __restrict__ a_gate,
                                         const __half* __restrict__ a_up, float* __restrict__ temp,
                                         int rank, int inner, int tokens, int stride, int splits) {
    const int projection = blockIdx.x / rank;
    const int rr         = blockIdx.x - projection * rank;
    const int split      = blockIdx.y;
    const int lane       = threadIdx.y;
    const int column     = blockIdx.z * kLoraSwiGLUTileColumns + threadIdx.x;
    if (column >= tokens) { return; }
    const __half* row = (projection == 0 ? a_gate : a_up) + static_cast<std::size_t>(rr) * inner;
    float acc         = 0.0F;
    const int step    = kLoraSwiGLUTileInner * splits;
    for (int kk = lane + split * kLoraSwiGLUTileInner; kk < inner; kk += step) {
        acc += __half2float(row[kk]) *
               __bfloat162float(x[static_cast<std::size_t>(kk) * tokens + column]);
    }
    __shared__ float partial[kLoraSwiGLUTileInner * kLoraSwiGLUTileColumns];
    partial[lane * kLoraSwiGLUTileColumns + threadIdx.x] = acc;
    __syncthreads();
    if (lane == 0) {
        float total = partial[threadIdx.x];
        for (int slice = 1; slice < kLoraSwiGLUTileInner; ++slice) {
            total += partial[slice * kLoraSwiGLUTileColumns + threadIdx.x];
        }
        temp[static_cast<std::size_t>(column) * stride + projection * rank * splits +
             split * rank + rr] = total;
    }
}

// The delta and the product in one pass.
template <int COLUMNS>
__global__ void lora_swiglu_kernel(const float* __restrict__ temp,
                                   const __half* __restrict__ b_gate,
                                   const __half* __restrict__ b_up,
                                   const __nv_bfloat16* __restrict__ plane,
                                   __nv_bfloat16* __restrict__ activation, int rows, int rank,
                                   int tokens, int stride, int splits, int plane_stride,
                                   float scale_gate, float scale_up) {
    // folded[projection][rr][c], row-major in the column so a thread's walk is contiguous.
    extern __shared__ float folded[];
    const int column_base = blockIdx.y * COLUMNS;
    const int columns     = min(COLUMNS, tokens - column_base);
    const int projection_stride = rank * COLUMNS;

    const int staged = 2 * rank * columns;
    for (int index = threadIdx.x; index < staged; index += blockDim.x) {
        const int projection = index / (rank * columns);
        const int rest       = index - projection * rank * columns;
        const int c          = rest / rank;
        const int rr         = rest - c * rank;
        const float* src =
            temp + static_cast<std::size_t>(column_base + c) * stride + projection * rank * splits;
        float sum = 0.0F;
        for (int split = 0; split < splits; ++split) { sum += src[split * rank + rr]; }
        folded[projection * projection_stride + rr * COLUMNS + c] = sum;
    }
    __syncthreads();

    const int row = blockIdx.x * kLoraSwiGLURows + threadIdx.x;
    if (row >= rows) { return; }
    const __half* bg = b_gate + static_cast<std::size_t>(row) * rank;
    const __half* bu = b_up + static_cast<std::size_t>(row) * rank;

    for (int c = 0; c < COLUMNS; ++c) {
        if (c >= columns) { break; }
        const float* fg = folded + c;
        const float* fu = folded + projection_stride + c;
        float delta_gate = 0.0F;
        float delta_up   = 0.0F;
        for (int rr = 0; rr < rank; ++rr) {
            delta_gate += __half2float(bg[rr]) * fg[rr * COLUMNS];
            delta_up += __half2float(bu[rr]) * fu[rr * COLUMNS];
        }
        const int column      = column_base + c;
        const std::size_t at = static_cast<std::size_t>(column) * plane_stride + row;
        const float gate = __bfloat162float(plane[at]) + scale_gate * delta_gate;
        const float up   = __bfloat162float(plane[at + rows]) + scale_up * delta_up;
        activation[static_cast<std::size_t>(column) * rows + row] =
            __float2bfloat16(silu(gate) * up);
    }
}

} // namespace ninfer::ops
