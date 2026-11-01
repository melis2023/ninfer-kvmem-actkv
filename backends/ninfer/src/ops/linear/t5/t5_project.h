// Adapted from JGamboa/ninfer-4090-windows@e905787f (Apache-2.0); see NOTICE.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// out_s = W x for a T5_G128_FP16 RowSplit weight W [N,K] and BF16 x [K,T] (K % 1024 == 0).
// The N parent rows are written in order to BF16 outputs [rows_s, T] with contiguous rows (sum rows_s
// == N, at most four), e.g. query/key/gate/value of one fused parent. With accumulate,
// out_s += W x in FP32 before one BF16 rounding (the residual epilogue of linear_add). FP32
// accumulation, one FP16 scale per 128-column group.
//
// A8 only: the policy must allow A8 and a workspace of t5_workspace_capacity_bytes is required.
// x is quantized to int8 with one FP32 scale per token and 128-column group (t5_a8.cuh), then
// multiplied in integer arithmetic: dp4a GEMV through T = 4, a small-T int8 MMA route through
// T = 32 (weights read once per 32 tokens, K split over the warps of a 16-row CTA), the int8 MMA
// prefill GEMM beyond (or 32-token small-T tiles when N % 64 != 0).
// Existing fused producers supply the already rotated x. Through T = 4 the
// quantization kernel also asks L2 for
// the weight's first MiB of code rows, which the GEMV reads first. Graph-capturable: static grid
// per (N, T), no host sync.
void t5_project(const Tensor& x, const Weight& w, std::span<Tensor* const> outputs,
                bool accumulate, LinearPolicy policy, WorkspaceArena* workspace,
                cudaStream_t stream);

// Transient bytes of t5_project for any T <= max_tokens: the int8 activation, its group scales
// and group and slice sums. Requires a policy that allows A8.
std::size_t t5_workspace_capacity_bytes(LinearPolicy policy, std::int32_t input_rows,
                                        std::int32_t max_tokens);

// Embedding gather from a T5_G128_FP16 table [vocab, K]: out[:, t] = row ids[t] of the logical
// table, i.e. the decoded stored row z' or, with signs, S (H_1024 z') / 32 per
// 1024-column block (the W' H S algebra of a rotated projection). ids I32 [T], out BF16 [K, T].
void t5_embedding(const Tensor& ids, const Weight& table, const Tensor& signs, Tensor& out, cudaStream_t stream);

// Throws unless w is a resident T5_G128_FP16 RowSplit view with K % 1024 == 0.
void validate_t5_weight(const Weight& w, const char* op);

} // namespace ninfer::ops::detail
