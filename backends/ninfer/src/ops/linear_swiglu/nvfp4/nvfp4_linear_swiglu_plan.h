#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

// Mirrors the dispatch decision for the int8-grouped route: true exactly when a call with this
// policy and width would take that integer route under the route table the process faces right
// now. The int8 policy falls back to the A16 composition whenever the route table or the
// whole-tile admission rule does not admit it -- silently, by design. This query makes that
// fallback observable, so a caller (the op tests above all) can assert the integer kernel really
// runs instead of measuring the A16 fallback it shadows.
[[nodiscard]] bool nvfp4_linear_swiglu_int8_route_active(LinearPolicy policy,
                                                         std::int32_t tokens);

// The chain predicate behind the int8 fused-edge entry, evaluated at the device profile's
// `nvfp4_int8/linear_swiglu` band. True means the caller may skip its standalone rmsnorm and call
// unified::nvfp4_linear_swiglu_int8_fused_norm_launch; false means the own chain answers the call
// and that entry refuses. Every width the unified route serves qualifies: both chains rescale
// against the per-32 activation plane the edge writes.
[[nodiscard]] bool nvfp4_linear_swiglu_int8_fused_norm_eligible(LinearPolicy policy,
                                                                std::int32_t tokens);

} // namespace ninfer::ops::detail
