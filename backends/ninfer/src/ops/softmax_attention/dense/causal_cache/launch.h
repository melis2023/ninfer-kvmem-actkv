#pragma once

// ninfer::ops::detail - private launch prototypes for causal_softmax_attention policies.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt };

// The widest prompt-route row block (the eight-warp fast INT8 kernel); every prompt kernel's row
// block divides it.
inline constexpr std::int32_t kPromptWaveRows = 128;

// Whether a prompt call on this storage takes the fast kernel (register-resident rows, FP16 PV per
// 64-key tile): when the caller asks for it (--fast-prefill-kernel), else as NINFER_PROMPT_FAST=0|1
// or the device profile's "attn_prompt_fast" says. Of the other storages only NVFP4-G16 has one
// (prompt_nvfp4_fast.cuh, Blackwell builds, over more than 2048 visible keys).
[[nodiscard]] bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested);

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             KvCacheStorage cache_storage,
                                             CausalAttentionExecutionEnvelope envelope,
                                             std::int32_t batch_size = 1);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size, KvCacheStorage storage,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

// A non-null gate is applied by the reduce epilogue at the store. Only the shared BF16/INT8
// reducer accepts one; the caller keeps the standalone multiply for every other storage.
void causal_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& table_rows, float scale,
                                     PagedKVBatchLayerView cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     std::int32_t column_begin, std::int32_t width,
                                     Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                     Tensor& out, cudaStream_t stream, const void* gate = nullptr);

// Parallel query tiles for a single row's chunked small-T width over an INT8-family cache: the
// tile width (0 when off) whose tiles run as the batch rows of one split-KV launch and one reduce,
// after one batched append, instead of the serial fused chunks. Opt-in: NINFER_ATTN_PARALLEL_TILES
// or the device profile's "attn_parallel_tiles".
[[nodiscard]] std::int32_t causal_attention_parallel_tile_width(std::int32_t q_heads,
                                                                std::int32_t width,
                                                                std::int32_t batch_size,
                                                                KvCacheStorage storage);

// Runs those tiles over keys already in the cache. tile_valid (only with valid_columns) and
// tile_rows hold one I32 per tile; the partials are sized for tile_width columns, the split
// capacity of that width at batch = tiles, and tiles batch rows.
void causal_attention_small_t_tiles_launch(const Tensor& q, const Tensor& positions,
                                           const Tensor* valid_columns, const Tensor* table_rows,
                                           float scale, PagedKVBatchLayerView cache,
                                           CausalAttentionExecutionEnvelope envelope,
                                           std::int32_t tile_width, Tensor& tile_valid,
                                           Tensor& tile_rows, Tensor& partial_acc,
                                           Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                           cudaStream_t stream, const void* gate = nullptr);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream);

void causal_attention_small_t_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_nvfp4_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  CausalAttentionExecutionEnvelope envelope,
                                                  Tensor& partial_acc, Tensor& partial_m,
                                                  Tensor& partial_l, Tensor& out,
                                                  cudaStream_t stream);

void causal_attention_small_t_k8v4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_k8v4_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 CausalAttentionExecutionEnvelope envelope,
                                                 Tensor& partial_acc, Tensor& partial_m,
                                                 Tensor& partial_l, Tensor& out,
                                                 cudaStream_t stream);

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, bool fast,
                                    cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out, bool fast,
                                              cudaStream_t stream);

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream);

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream);

// The fast NVFP4 prompt kernel (block-scaled FP4 QK, V decoded in registers; Ian Ranson's
// prompt_nvfp4_fast.cuh) over keys already in the cache. A single-row launch whose row blocks alone
// would leave SMs idle splits its keys across CTAs into `workspace`. Blackwell (120a) builds only;
// causal_attention_prompt_nvfp4_fast_applies() is false everywhere else.
void causal_attention_prompt_nvfp4_fast_launch(const Tensor& q, const Tensor& positions,
                                               const Tensor* valid_columns,
                                               const Tensor* table_rows, float scale,
                                               PagedKVBatchLayerView cache,
                                               std::uint32_t max_visible_keys,
                                               WorkspaceArena& workspace, Tensor& out,
                                               cudaStream_t stream);

void causal_attention_prompt_k8v4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& positions, const Tensor& valid_columns,
                                         const Tensor& table_rows, float scale,
                                         PagedKVBatchLayerView cache, Tensor& out,
                                         cudaStream_t stream);

void causal_attention_prompt_k8v4_attention_launch(const Tensor& q, const Tensor& positions,
                                                   float scale, const PagedKVLayerView& cache,
                                                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
