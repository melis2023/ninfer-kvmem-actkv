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

[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream);

// The integer-activation route on the s8 tensor cores (sm_86): every positive width, its last
// token tile padded, the per-(token,32) quantised activation planes in the caller workspace. The
// table names the wide mainloop: unified runs the grouped ladder, own runs MMQ.
void nvfp4_attn_input_int8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                  Nvfp4Int8Mainloop table, cudaStream_t stream);

[[nodiscard]] bool nvfp4_attn_input_int8_tokens_supported(std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

// The same route with the RMSNorm folded into the quantise pass: one edge kernel turns the
// un-normed `residual` into the very planes `nvfp4_attn_input_int8_launch` would have built from
// the normed row, and the mainloop half is shared rather than duplicated. Only the unified table
// answers here (the interleaved MMQ plane belongs to the own chain), and the caller must then skip
// its own rmsnorm -- see the eligibility query in the wrapper.
void nvfp4_attn_input_int8_fused_norm_launch(const Tensor& residual, const Tensor& norm_weight,
                                             float eps, const Weight& weight, Tensor& q,
                                             Tensor& gate, Tensor& k, Tensor& v,
                                             WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                             cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail::unified
