#pragma once

#include "core/tensor.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8.
[[nodiscard]] std::size_t candidate_selector_path_workspace_capacity_bytes(std::int32_t min_steps,
                                                                           std::int32_t max_steps,
                                                                           std::int32_t min_batch,
                                                                           std::int32_t max_batch);

/**
 * Op: candidate_selector_path
 *
 * For K in [1,15] and B in [1,8], the inputs are contiguous candidate_ids I32 [16,K,B],
 * unary_scores FP32 [16,K,B], projected_hidden BF16 [256,K,B], anchors I32 [B],
 * predecessor_codebook and successor_codebook BF16 [256,248320], base_positions I32 [B], and a
 * device-resident SamplingConfig[B]. Candidate rank is the fastest axis. The 16 candidate ids in
 * each row are distinct, and all candidate and anchor token ids lie in [0,248077); the registered
 * vocabulary, artifact binding, and linear_topk producer establish that trusted value contract.
 *
 * Starting with predecessor=anchors[b], each position i in [0,K) computes:
 *
 *   edge[c] = unary_scores[c,i,b]
 *           + sum_r predecessor_codebook[r,predecessor]
 *                   * projected_hidden[r,i,b]
 *                   * successor_codebook[r,candidate_ids[c,i,b]].
 *
 * A row with configs[b].temperature<=0 selects the lowest candidate rank attaining max(edge) and
 * writes its exact one-hot distribution. A positive-temperature row writes the FP32 softmax of
 * edge/temperature, then draws a candidate with counter key
 * (configs[b].seed,base_positions[b]+i,kSamplePurposeDFlash2Proposal). The selected global id is
 * written to drafts[i,b] and becomes the next predecessor. The Op ignores all other
 * SamplingConfig fields and never updates token_counts.
 *
 * drafts is contiguous I32 [K,B] and proposal_q is contiguous FP32 [16,K,B]. Both outputs are
 * completely overwritten. Inputs, outputs, codebooks, and the config array must be pairwise
 * non-overlapping. The Op has no persistent state or internal allocation. Caller workspace is
 * transient and must not overlap any input or output.
 */
void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream);

// Transient bytes candidate_topk() needs for U columns.
[[nodiscard]] std::size_t candidate_topk_workspace_capacity_bytes(std::int32_t columns);

/**
 * Op: candidate_topk
 *
 * The selector's candidates from dense logits (a head that the fused linear_topk does not take):
 * for each column u of logits, contiguous BF16 or FP32 [rows,U], the 16 rows v < valid_rows with
 * the largest logits[v,u], in descending order with the lower id first on a tie, as candidate_ids
 * I32 and their logits as candidate_scores FP32, each any contiguous tensor of 16 x U elements
 * with 16 fastest ([16,K,B] for U = K B). A NaN logit takes no part; a slot that no logit fills
 * holds id 0 and score -inf. valid_rows is in [16,rows].
 */
void candidate_topk(const Tensor& logits, std::int32_t valid_rows, Tensor& candidate_ids,
                    Tensor& candidate_scores, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
