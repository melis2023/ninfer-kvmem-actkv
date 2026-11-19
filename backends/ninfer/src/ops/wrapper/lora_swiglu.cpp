// ninfer::ops - lora_swiglu wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/lora_swiglu.h"

#include "ninfer/ops/lora_accum.h"    // lora_accum_decode_splits
#include "ops/launcher/lora_swiglu.h" // detail::lora_swiglu_launch

#include <stdexcept>
#include <string>

namespace ninfer::ops {

std::size_t lora_swiglu_scratch_rank(std::int32_t rank) noexcept {
    return rank > 0 ? static_cast<std::size_t>(2) * rank * lora_accum_decode_splits() : 0;
}

void lora_swiglu(const Tensor& x, const Tensor& plane, const LoraSwiGLUFactors& factors,
                 Tensor& activation, const Tensor& scratch, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || plane.dtype != DType::BF16 ||
        activation.dtype != DType::BF16) {
        throw std::invalid_argument("lora_swiglu: x/plane/activation must be BF16");
    }
    if (scratch.dtype != DType::FP32) {
        throw std::invalid_argument("lora_swiglu: the scratch must be FP32");
    }
    for (const Tensor* factor : {&factors.a_gate, &factors.b_gate, &factors.a_up, &factors.b_up}) {
        if (factor->dtype != DType::FP16) {
            throw std::invalid_argument("lora_swiglu: the adapter factors must be FP16");
        }
        if (!factor->is_contiguous()) {
            throw std::invalid_argument("lora_swiglu: the adapter factors must be contiguous");
        }
    }
    const std::int32_t inner  = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t rows   = activation.ne[0];
    if (!x.is_contiguous() || !plane.is_contiguous() || !activation.is_contiguous() ||
        !scratch.is_contiguous()) {
        throw std::invalid_argument("lora_swiglu: x/plane/activation/scratch must be contiguous");
    }
    if (plane.ne[0] != 2 * rows || plane.ne[1] != tokens) {
        throw std::invalid_argument("lora_swiglu: plane must be [2*inter, T] of the activation");
    }
    if (activation.ne[1] != tokens) {
        throw std::invalid_argument("lora_swiglu: activation must hold one row per token");
    }
    const std::int32_t rank = factors.a_gate.ne[1];
    if (factors.a_gate.ne[0] != inner || factors.a_up.ne[0] != inner ||
        factors.a_up.ne[1] != rank) {
        throw std::invalid_argument("lora_swiglu: A must be [rank, in_features] of x");
    }
    if (factors.b_gate.ne[0] != rank || factors.b_up.ne[0] != rank ||
        factors.b_gate.ne[1] != rows || factors.b_up.ne[1] != rows) {
        throw std::invalid_argument("lora_swiglu: B must be [intermediate, rank]");
    }
    if (scratch.ne[0] < static_cast<std::int32_t>(lora_swiglu_scratch_rank(rank)) ||
        scratch.ne[1] < tokens) {
        throw std::invalid_argument("lora_swiglu: the scratch needs " +
                                    std::to_string(lora_swiglu_scratch_rank(rank)) +
                                    " rows for " + std::to_string(tokens) + " columns");
    }
    if (inner <= 0 || tokens <= 0 || rank <= 0 || rows <= 0) { return; }
    detail::lora_swiglu_launch(x, factors.a_gate, factors.b_gate, factors.a_up, factors.b_up, plane,
                               activation, scratch, factors.scale_gate, factors.scale_up, stream);
}

} // namespace ninfer::ops
