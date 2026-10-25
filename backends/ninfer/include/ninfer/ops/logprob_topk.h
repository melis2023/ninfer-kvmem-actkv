#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// The number of most likely tokens logprob_topk() reports per row.
inline constexpr std::int32_t kLogprobTopK = 20;

/**
 * Op: logprob_topk
 *
 * The log probabilities of the distribution that sample() and the speculative acceptance Ops draw
 * a token from, before top_k, top_p and min_p truncate it, with its kLogprobTopK most likely tokens.
 *
 * Math / indexing:
 *   For verification column c and batch row b, let a[v] be sampling.h's adjusted logit of token v
 *   under configs[b]: -inf when column c of configs[b].token_mask excludes v, and otherwise
 *   logits[v,c,b] less the presence and frequency penalties of v's count in
 *   configs[b].token_counts plus its occurrences in drafts[0..c, b), the overlay the speculative
 *   Ops apply to column c. With
 *
 *     T[b]   = configs[b].temperature > 0 ? configs[b].temperature : 1
 *     s[v]   = a[v] / T[b] for every v in [0,token_domain) with a finite a[v]
 *     lse    = log(sum_v exp(s[v]))
 *
 *   top_ids[k,c,b] is the token with the k-th largest s (the lower id first on a tie),
 *   top_values[k,c,b] = s[top_ids[k,c,b]] - lse and lse[c,b] = lse. A token with a non-finite
 *   a[v], a masked one among them, takes no part; when fewer than kLogprobTopK tokens remain, the
 *   remaining slots hold id -1 and value -inf.
 *
 * Logical shapes:
 *   logits is BF16 [physical_rows,columns,batch] with kLogprobTopK<=token_domain<=physical_rows.
 *   configs points to `batch` contiguous device SamplingConfig records. drafts is I32
 *   [columns-1,batch] when columns>1 and is not read otherwise. top_ids is I32 and top_values FP32
 *   [kLogprobTopK,columns,batch]; lse is FP32 [columns,batch]; active is a device I32 [1].
 *
 * Supported domain:
 *   Every tensor is contiguous. Each row has at least one token with a finite adjusted logit, or
 *   that row's outputs are unspecified. Token masks and counts follow sampling.h; they are only
 *   read.
 *
 * Numeric:
 *   FP32 approximation of the ideal values; the reduction order and accumulator precision are the
 *   implementation's. The independent oracle evaluates the formula in FP64 from the BF16 inputs.
 *
 * Effects:
 *   Writes top_ids, top_values and lse when *active is nonzero, and nothing at all when it is
 *   zero, so a captured CUDA graph can carry the Op unconditionally. Inputs are unchanged; outputs
 *   must not overlap the inputs or one another.
 *
 * Workspace:
 *   Caller-owned transient storage reported by logprob_topk_workspace_capacity_bytes() for
 *   rows=columns*batch; no persistent state.
 */
void logprob_topk(const Tensor& logits, const SamplingConfig* configs, const Tensor* drafts,
                  std::int32_t token_domain, Tensor& top_ids, Tensor& top_values, Tensor& lse,
                  const Tensor& active, WorkspaceArena& workspace, cudaStream_t stream);

// Transient bytes logprob_topk() needs for a call over `rows` = columns*batch rows.
[[nodiscard]] std::size_t logprob_topk_workspace_capacity_bytes(std::int32_t token_domain,
                                                                std::int32_t rows);

} // namespace ninfer::ops
