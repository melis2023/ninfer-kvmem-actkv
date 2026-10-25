#pragma once

// Prompt-route causal attention on sm_70 (Tesla V100). The prompt kernels of this family are
// written against Ampere's mma.sync, ldmatrix and cp.async, which a Volta build only emulates; a
// single-sequence prompt over any KV storage takes this route instead (prompt_sm70.cu). A CTA owns
// 64 query rows of one KV head (the group's query heads times consecutive tokens) and walks the
// visible keys in tiles of 32: its threads read each tile from the pages and decode it to FP16 in
// shared memory once, and each warp runs the decode kernel's m8n8k4 products and FP32 online
// softmax over it for eight rows (small_t_sm70.cuh). The next tile's codes are read into registers
// while the current one is multiplied. No workspace: the pages are read where they are.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Whether this build takes the route for a prompt over `storage` with `batch` rows.
[[nodiscard]] bool causal_attention_prompt_sm70_applies(KvCacheStorage storage, std::int32_t batch);

// The cached form: K/V are already in the pages of `cache`.
void causal_attention_prompt_sm70_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache, Tensor& out,
                                         cudaStream_t stream);

// One row of a batched view (after its K/V were appended): `table_rows[0]` selects its table, and
// a non-null `valid_columns` limits the queries to its first valid_columns[0] columns (the rest
// of the output is zero).
void causal_attention_prompt_sm70_launch(const Tensor& q, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& table_rows,
                                         float scale, const PagedKVBatchLayerView& cache,
                                         Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
