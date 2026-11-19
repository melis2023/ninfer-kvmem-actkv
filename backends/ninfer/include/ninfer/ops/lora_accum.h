#pragma once

// ninfer::ops - low-rank adapter (LoRA) update of a projection result.

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: lora_accum
 *
 *   out[i, t] += scale * sum_{rr < rank} B[i, rr] * ( sum_{kk < in} A[rr, kk] * x[kk, t] )
 *
 * `a` is FP16 [rank, in_features] (ne[0] = in_features), `b` is FP16 [out_features, rank]
 * (ne[0] = rank), `x` is BF16 [in_features, tokens] and `out` is BF16 [out_features, tokens].
 * `x`, `a` and `b` are contiguous; `out` may be a row region of a wider plane (a fused gate/up
 * parent, for instance), so it only has to be contiguous along its rows and to keep a column
 * stride of at least its row count. x and out are column-major like every ninfer activation.
 * The two products accumulate in FP32 and `out` is rounded once, after the adapter delta is
 * added to the value it already holds, so the Op composes with a projection epilogue.
 *
 * `temp` carries the FP32 intermediate and is owned by the caller, not by a workspace arena: a
 * captured decode graph must keep the address it was captured with. A decoded column splits its
 * inner reduction `lora_accum_decode_splits()` ways, so the buffer holds
 * `lora_accum_scratch_rank(rank)` rows; one buffer serves every factor of a block at every column
 * count up to its capacity.
 *
 * The Op never touches the weight: the delta is evaluated against the same `x` the preceding
 * GEMM read, which is the llama.cpp `build_lora_mm` contract (Wx + scale * B(Ax)).
 */
[[nodiscard]] std::size_t lora_accum_scratch_bytes(std::int32_t scratch_rows,
                                                   std::int32_t max_tokens);

[[nodiscard]] constexpr std::int32_t lora_accum_decode_splits() noexcept { return 8; }

// Widest rank the one-launch fused route covers; anything wider takes the split route.
[[nodiscard]] constexpr std::int32_t lora_accum_fused_max_rank() noexcept { return 32; }

// Scratch rows one factor of `rank` needs: a decode call parks `rank` partial sums per split.
[[nodiscard]] constexpr std::int32_t lora_accum_scratch_rank(std::int32_t rank) noexcept {
    return rank > 0 ? rank * lora_accum_decode_splits() : 0;
}

void lora_accum(const Tensor& x, const Tensor& a, const Tensor& b, Tensor& out, const Tensor& temp,
                float scale, cudaStream_t stream);

} // namespace ninfer::ops
