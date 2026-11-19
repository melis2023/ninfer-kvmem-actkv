#pragma once

// ninfer::ops - fused gate/up adapter delta and SwiGLU product.

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

// The two factors of one block's dense FFN, both reading the same block input.
struct LoraSwiGLUFactors {
    Tensor a_gate; // FP16 [rank, in]
    Tensor b_gate; // FP16 [intermediate, rank]
    Tensor a_up;   // FP16 [rank, in]
    Tensor b_up;   // FP16 [intermediate, rank]
    float scale_gate = 0.0F;
    float scale_up   = 0.0F;
};

/**
 * Op: lora_swiglu
 *
 *   activation[i, t] = silu(plane[i, t]            + scale_gate * Bg[i] . (Ag x[:, t]))
 *                    * (   plane[inter + i, t]     + scale_up   * Bu[i] . (Au x[:, t]))
 *
 * `x` is BF16 [in, T], `plane` is BF16 [2*inter, T] holding the pre-activation rows with gate
 * first and up second (the layout of a fused GGUF gate/up parent), and `activation` is BF16
 * [inter, T]. `silu` is `x / (1 + exp(-x))` from ops/common/math.cuh, the same arithmetic as the
 * standalone `silu_mul`, so a zero adapter reproduces the fused kernel's result bit for bit.
 *
 * The Op exists so an attached adapter does not have to give up the fused route entirely: it
 * replaces "two `lora_accum` launches plus one `silu_mul` launch" with a single launch, and folds
 * the two rank reductions into the product instead of writing them into the plane. `scratch` is
 * FP32 [2 * rank_capacity, T_capacity] owned by the caller: pass 1 parks the gate partials in its
 * first half and the up partials in its second.
 */
[[nodiscard]] std::size_t lora_swiglu_scratch_rank(std::int32_t rank) noexcept;

void lora_swiglu(const Tensor& x, const Tensor& plane, const LoraSwiGLUFactors& factors,
                 Tensor& activation, const Tensor& scratch, cudaStream_t stream);

} // namespace ninfer::ops
