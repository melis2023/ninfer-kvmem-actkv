#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The block indexer of Qwen3.8-Flash-Next's sparse attention (transformers
 * Qwen4ExpTextQSAIndexer), for one sequence. A query at position p sees the complete 4-token
 * blocks b with 4b + 3 <= p; each block has a pooled key
 *
 *   K_b = rope(groupnorm(mean_{i<4} k[4b + i], key_norm), 4b)            FP32 [128]
 *
 * from the raw indexer keys k (projection rows 512..639), and the query selects
 *
 *   score_b = sum_{h<4} relu(q_h . K_b),   q_h = rope(groupnorm(q[h], query_norm), p)
 *
 * the min(512, blocks) best blocks, ties to the lower block index. groupnorm is the zero-centred
 * RMSNorm x / sqrt(mean x^2 + eps) * (1 + w) over 128 values; rope rotates the first 64 dims
 * rotate-half at angle fl32(fl32(position) * f_i), f_i = fl32(1e7^(-2i/64)), i < 32, and passes
 * dims 64..127 through. The query's own incomplete block (positions 4 floor((p+1)/4) .. p) is
 * attended by every query and is not part of the selection.
 *
 * `projection` is BF16 [640, tokens] (4 query heads of 128, then the key) for the consecutive
 * positions `first_position` .. `first_position + tokens - 1`; `first_position` is a device I32
 * word, so a call captured in a graph replays at whatever position the word holds. `pooled` is
 * FP32 [128, capacity], the sequence's pooled keys by block, sized by the caller for every
 * position it will reach (blocks at or past `capacity` are neither pooled nor selected); `tail`
 * FP32 [128, 3] the raw keys of the incomplete block before `first_position` (slot = position
 * mod 4). The oracle evaluates in FP64 from the
 * represented inputs (angles as above); pooled keys are compared as FP32, and a selection must
 * equal the oracle's up to blocks whose FP64 scores tie the 512th within 1e-5 relative.
 */
struct QsaIndexerWeights {
    const Tensor* query_norm = nullptr; // BF16 [128]
    const Tensor* key_norm   = nullptr; // BF16 [128]
};

// Pools every block that completes in [first_position, first_position + tokens) into `pooled`
// and leaves the incomplete block's raw keys in `tail`.
void qsa_indexer_append(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, Tensor& pooled, Tensor& tail,
                        cudaStream_t stream);

[[nodiscard]] std::size_t qsa_indexer_select_workspace_bytes(std::int32_t tokens,
                                                             std::int32_t capacity);

// Writes each query's selected blocks in increasing order into `selected` I32 [512, tokens] and
// their number into `counts` I32 [tokens]. Run after qsa_indexer_append for the same tokens.
void qsa_indexer_select(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const Tensor& pooled,
                        WorkspaceArena& workspace, Tensor& selected, Tensor& counts,
                        cudaStream_t stream);

} // namespace ninfer::ops
