#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_int8_route.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::unified {

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream);

// The integer-activation route on the s8 tensor cores (sm_86): every positive width, its last
// token tile padded, the projection added into the residual in the epilogue. Requires the exact
// int8-grouped workspace capacity for the call's token count. The table names the wide mainloop:
// unified runs the grouped ladder, own runs MMQ.
void nvfp4_linear_add_int8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                  cudaStream_t stream);

[[nodiscard]] bool nvfp4_linear_add_int8_tokens_supported(std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail::unified
