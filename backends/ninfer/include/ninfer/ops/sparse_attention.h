#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen3.8-Flash-Next's sparse attention for one sequence: query p attends only to the 4-token
 * blocks its indexer selected (qsa_indexer_select) and to its own incomplete block, positions
 * 4 floor((p + 1) / 4) .. p:
 *
 *   out[h] = sum_{j in S_p} softmax_j(scale * q[h] . k[kv(h), j]) v[kv(h), j],   kv(h) = h / 12
 *
 * `q` is BF16 [256, 24, tokens] (normalised and rotated) for the consecutive positions
 * `first_position` .., read from a device I32 word so that a call captured in a graph replays at
 * whatever position the word holds; `selected` I32 [512, tokens] and `counts` I32 [tokens] are the
 * indexer's blocks; `cache` is the sequence's paged KV layer (2 heads of 256) holding every
 * selected and tail position, its block table covering every position the caller reaches, in any
 * storage kv_cache_append writes (BF16, the INT8 family, FP8 rows, NVFP4, K8V4); `out` is BF16
 * [256, 24, tokens]. The oracle evaluates the softmax and the weighted sum in FP64 from the
 * represented cache and queries: a quantized cache's decoded keys lie in the rotated basis
 * kv_cache_append stores them in, against the query under the same normalized Hadamard transform,
 * and its decoded values as stored, NVFP4 and K8V4 values rotated as well with the transform
 * applied back to the result. The kernel stages a quantized cache's decoded rows in FP16; the
 * output is compared after its BF16 store. Up to eight queries keep partial softmaxes in FP32
 * workspace; no state.
 */
[[nodiscard]] std::size_t sparse_softmax_attention_workspace_bytes(std::int32_t tokens);

void sparse_softmax_attention(const Tensor& q, const Tensor& first_position, const Tensor& selected,
                              const Tensor& counts, const PagedKVLayerView& cache, float scale,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
