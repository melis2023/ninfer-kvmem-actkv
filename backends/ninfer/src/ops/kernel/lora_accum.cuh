#pragma once

// Implements: include/ninfer/ops/lora_accum.h
//
// A rank-row of A produces a single output, so the inner dimension -- not the row count -- carries
// the parallelism. Two kernels:
//
//   pass 1  folded[column, rr] = sum_kk A[rr, kk] * x[kk, column], split `splits` ways over the
//           inner dimension into FP32 partials. This is the only place the adapter's A is read.
//   pass 2  out[row, column] += scale * sum_rr B[row, rr] * folded[column, rr].
//
// Pass 2 owns a tile of rows and a compile-time number of columns: the caller dispatches on the
// column count (1, 2, 4, 8, 16 or 32), so a decode round at 4 columns issues 4 multiply-adds per
// rank step instead of 32. Staging the folded product tile in shared memory lets every thread of
// the block read its own B row once and reuse it across the whole tile, and the grid is
// `ceil(rows/128) x ceil(columns/COLUMNS)` rather than one block per (row, column) pair. An
// earlier one-warp-per-row version produced 2560 blocks for a single 5120-row projection and a
// fixed 32-column inner loop wasted 8x of its arithmetic at decode width.
//
// Both write through `out_ld`, the destination's column stride, so a row region of a wider plane (a
// fused gate/up parent) is addressed with the parent's stride. Arithmetic is FP32 throughout; `out`
// is rounded once, after the adapter delta is added to the value it already holds.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

// Pass 1: a (columns, inner lanes) tile.
inline constexpr int kLoraTileColumns = 4;
inline constexpr int kLoraTileInner   = 64;
// Pass 2: one thread per output row, a tile of columns per block.
inline constexpr int kLoraPass2Rows   = 128;

__device__ __forceinline__ float lora_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

// temp[column, split * rank + rr] = this split's slice of the inner dimension.
__global__ void lora_pass1_kernel(const __nv_bfloat16* __restrict__ x,
                                  const __half* __restrict__ a, float* __restrict__ temp, int rank,
                                  int inner, int tokens, int temp_stride, int splits) {
    const int column = blockIdx.z * kLoraTileColumns + threadIdx.x;
    const int rr     = blockIdx.x;
    const int split  = blockIdx.y;
    const int lane   = threadIdx.y;
    if (column >= tokens) { return; }
    const __half* row = a + static_cast<std::size_t>(rr) * inner;
    float acc         = 0.0F;
    const int step    = kLoraTileInner * splits;
    for (int kk = lane + split * kLoraTileInner; kk < inner; kk += step) {
        acc += __half2float(row[kk]) *
               __bfloat162float(x[static_cast<std::size_t>(kk) * tokens + column]);
    }
    __shared__ float partial[kLoraTileInner * kLoraTileColumns];
    partial[lane * kLoraTileColumns + threadIdx.x] = acc;
    __syncthreads();
    if (lane == 0) {
        float total = partial[threadIdx.x];
        for (int slice = 1; slice < kLoraTileInner; ++slice) {
            total += partial[slice * kLoraTileColumns + threadIdx.x];
        }
        temp[static_cast<std::size_t>(column) * temp_stride + split * rank + rr] = total;
    }
}

// One thread per output row over a COLUMNS-wide tile; the folded tile sits in shared memory, so the
// thread's B row is read once and reused by every column of the tile.
template <int COLUMNS>
__global__ void lora_pass2_kernel(const float* __restrict__ temp, const __half* __restrict__ b,
                                  __nv_bfloat16* __restrict__ out, int rows, int rank, int tokens,
                                  int temp_stride, int splits, int out_ld, float scale) {
    extern __shared__ float folded[]; // [rank][COLUMNS], row-major so a row walk is contiguous
    const int column_base = blockIdx.y * COLUMNS;
    const int columns     = min(COLUMNS, tokens - column_base);

    // Fold the pass-1 splits while staging. Consecutive threads take consecutive `rr` of one
    // column, so the `temp` reads are coalesced.
    const int staged = rank * columns;
    for (int index = threadIdx.x; index < staged; index += blockDim.x) {
        const int c      = index / rank;
        const int rr     = index - c * rank;
        const float* src = temp + static_cast<std::size_t>(column_base + c) * temp_stride;
        float sum        = 0.0F;
        for (int split = 0; split < splits; ++split) { sum += src[split * rank + rr]; }
        folded[rr * COLUMNS + c] = sum;
    }
    __syncthreads();

    const int row = blockIdx.x * kLoraPass2Rows + threadIdx.x;
    if (row >= rows) { return; }
    const __half* brow = b + static_cast<std::size_t>(row) * rank;

    float acc[COLUMNS];
#pragma unroll
    for (int c = 0; c < COLUMNS; ++c) { acc[c] = 0.0F; }
    for (int rr = 0; rr < rank; ++rr) {
        const float bv      = __half2float(brow[rr]);
        const float* shared = folded + rr * COLUMNS;
#pragma unroll
        for (int c = 0; c < COLUMNS; ++c) { acc[c] += bv * shared[c]; }
    }

#pragma unroll
    for (int c = 0; c < COLUMNS; ++c) {
        if (c >= columns) { break; }
        __nv_bfloat16* destination = out + static_cast<std::size_t>(column_base + c) * out_ld + row;
        *destination = __float2bfloat16(__bfloat162float(*destination) + scale * acc[c]);
    }
}

// Every instantiation the dispatcher may reach, in one place.
inline constexpr int kLoraPass2Widths[] = {1, 2, 4, 8, 16, 32};

} // namespace ninfer::ops
