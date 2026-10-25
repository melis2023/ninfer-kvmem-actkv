#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4LinearAddLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens);

// Upstream's routes over the unified templates (gemv, sliced-K MMA to 32 columns, MMA tiles), taken
// where fused_route_table("unified/q4_linear_add") says so.
Q4LinearAddLaunch select_q4_linear_add_unified(std::int32_t tokens);

// The individual routes, named so a route-boundary sweep can time the ones the table does not
// currently select (bench/ops/dense_linear_add_schedule_bench.cu).
void q4_linear_add_gemv_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
// Small-T MMA with direct loads, one, two or four eight-column tiles (up to 8 / 16 / 32 tokens).
// Taken only where the device profile's "q4_linear_add/5120x<k>" entry routes a width to it.
void q4_linear_add_small_t_c8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_small_t_c16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_small_t_c32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit4_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_ksplit32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r32_c32_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r32_c64_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c48_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c64_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c80_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c96_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c112_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q4_linear_add_mma_r64_c128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
