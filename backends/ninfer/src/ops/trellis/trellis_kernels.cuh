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

// Coded tile product: yr [N,T] FP32 = W_hat * xa. `xf`/`xs` are the prefill prologue scratch
// (fp16 B-fragment plane and per-token 2^-e scales); unused by the single-token split route.
// `partials` holds kPfMaxSplits [N,T] FP32 planes when the small-T route splits K, else null.
void tile_product(const float* xa, const Weight& w, float* yr, float* partials, int tokens,
                  std::uint16_t* xf, float* xs, cudaStream_t stream);

// Number of K ranges the small-T coded MMA may run in parallel; one plane per range.
inline constexpr int kPfMaxSplits = 6;

// Number of K ranges the single-token route accumulates in parallel. One token only reaches N/16
// warps, so the plain product is latency-bound; splitting K restores the parallelism.
inline constexpr int kDecodeSplits = 16;

// NINFER_TRELLIS_PFDECODE=1: single-token decode takes the staged prefill GEMM path instead of
// the legacy split-K GEMV. Temporary A/B while the new decode kernel is validated.
bool trellis_pf_decode_enabled();

// Single-token coded tile product: `partials` holds kDecodeSplits [N] planes and receives the
// reduced result in `yr`. Requires w.k >= 16 * kDecodeSplits * 2.
void tile_product_split1(const float* xa, const Weight& w, float* yr, float* partials,
                         cudaStream_t stream);

// NINFER_TRELLIS_GEMV=1: single-token decode takes the reference-structured chunked GEMV (weights as
// the A operand, the prologue's fp16 fragment as B, one MMA per k-tile, cp.async chunk ring).
bool trellis_gemv_enabled();

// The same single-token product through the new kernel; `xf`/`xs` are the prologue scratch and must
// be at least trellis_linear_workspace_bytes' xf/xs planes for one token.
void tile_product_split1_gemv(const float* xa, const Weight& w, float* yr, float* partials,
                              std::uint16_t* xf, float* xs, cudaStream_t stream);

// yr -> out: post scale, output gather, one BF16 rounding.
void post_and_gather(const float* yr, const Weight& w, Tensor& out, int tokens,
                     cudaStream_t stream);

// Coded row lookup: g [K,T] FP32 decoded rows (no side transform applied yet).
void decode_row(const Tensor& ids, const Weight& w, float* g, int tokens, cudaStream_t stream);

// g -> rows: pre scale, input gather, BF16 rounding.
void pre_and_gather_bf16(const float* g, const Weight& w, const Tensor& ids, Tensor& rows, int tokens,
                         cudaStream_t stream);

// dst[r][t] = src[global row index[r]][t] over BF16 rows of `tokens` entries: the banded head's final
// vocab-order read. The source plane is one token-major slab per band (`band_rows`/`band_count`
// describe them), so the kernel resolves a global row to its slab, token and local row itself.
void gather_rows_bf16(const void* src, const std::int32_t* index, std::int64_t rows, int tokens,
                      const std::int32_t* band_rows, std::int32_t band_count, void* dst,
                      cudaStream_t stream);

// dst[dst_row + r][t] = src[from(r)][t] with `from(r) = index[r]` when `index` is set, otherwise
// `(r / group) * stride + origin + r % group` (an interleaved row pattern), and the identity
// `dst_row + r` when `group` is zero as well. `src_rows`/`dst_rows` are the two planes' row counts.
void gather_rows_into_bf16(const void* src, const std::int32_t* index, std::int64_t rows,
                           std::int64_t dst_row, std::int32_t group, std::int32_t stride,
                           std::int32_t origin, int tokens, std::int64_t src_rows,
                           std::int64_t dst_rows, void* dst, cudaStream_t stream);

} // namespace ninfer::ops::detail::trellis
