#pragma once

#include "ops/trellis/trellis_reference.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail::trellis {

// A device-resident transform plan: the stored-index permutation of a kind-2 side plus the Paley
// matrix it crosses with. Owned by the dispatcher's cache; `plan` is null for kinds 0 and 1.
struct DevicePlan {
    const std::int32_t* perm = nullptr;
    const signed char* paley = nullptr;
};

// x [K,T] BF16 -> xa [K,T] FP32 with the input gather and pre scale applied.
void gather_scale(const Tensor& x, const Weight& w, float* out, int tokens, std::int64_t K,
                  cudaStream_t stream);

// Per-column Sylvester RHT of `block` entries, in place, FP32 [n,columns].
void rht_inplace(float* v, std::int64_t n, int block, int columns, cudaStream_t stream);

// Per-column permutation gather (dir=0: dst[d]=src[perm[d]]) or scatter (dir=1).
void permute_side(const float* src, float* dst, const std::int32_t* perm, std::int64_t n,
                  int columns, bool scatter, cudaStream_t stream);

// One pass over the fused shards: gather into the logical order, the per-run butterflies and the
// Paley cross (or the reverse order for the inverse side), then scatter back. `tmp` and `tmp2` are
// [n,columns] FP32 scratch; `dst` may alias `src` exactly because the scatter reads `tmp2` alone.
void hadfull_pass(const float* src, float* dst, float* tmp, float* tmp2, const SidePlan& plan,
                  const DevicePlan& device, bool inverse, int columns, cudaStream_t stream);

// Coded tile product: yr [N,T] FP32 = W_hat * xa.
void tile_product(const float* xa, const Weight& w, float* yr, int tokens, cudaStream_t stream);

// yr -> out: post scale, output gather, one BF16 rounding.
void post_and_gather(const float* yr, const Weight& w, Tensor& out, int tokens,
                     cudaStream_t stream);

// Coded row lookup: g [K,T] FP32 decoded rows (no side transform applied yet).
void decode_row(const Tensor& ids, const Weight& w, float* g, int tokens, cudaStream_t stream);

// g -> rows: pre scale, input gather, BF16 rounding.
void pre_and_gather_bf16(const float* g, const Weight& w, Tensor& rows, int tokens,
                         cudaStream_t stream);

// dst[r][t] = src[index[r]][t] over BF16 rows of `tokens` entries: the banded head's final
// vocab-order read (the artifact's `vocab_gather` is the vocab -> concatenated-band map).
void gather_rows_bf16(const void* src, const std::int32_t* index, std::int64_t rows, int tokens,
                      void* dst, cudaStream_t stream);

} // namespace ninfer::ops::detail::trellis
