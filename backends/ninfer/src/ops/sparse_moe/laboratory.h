#pragma once

#include "ninfer/ops/sparse_moe.h"
#include "ops/sparse_moe/candidate.h"

namespace ninfer::ops::detail {

// Exact workspace for one forced candidate at T. No environment or global override is involved.
[[nodiscard]] std::size_t sparse_moe_candidate_workspace_bytes(QType gate, QType down,
                                                               std::int32_t tokens,
                                                               SparseMoeCandidate candidate);

// Uses the public Op's validation and represented operands. Numerical qualification still uses
// the complete Op oracle, including the NVFP4 prefill activation-quantization criterion.
void sparse_moe_candidate(const Tensor& x, const SparseMoeWeights& weights,
                          SparseMoeEpilogue epilogue, Tensor& destination,
                          SparseMoeCandidate candidate, WorkspaceArena& workspace,
                          cudaStream_t stream);

} // namespace ninfer::ops::detail
