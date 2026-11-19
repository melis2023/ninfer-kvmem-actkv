// ninfer::ops - lora_accum wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/lora_accum.h"

#include "ops/launcher/lora_accum.h" // detail::lora_accum_launch

#include <stdexcept>
#include <string>

namespace ninfer::ops {

std::size_t lora_accum_scratch_bytes(std::int32_t scratch_rows, std::int32_t max_tokens) {
    if (scratch_rows <= 0 || max_tokens <= 0) { return 0; }
    return static_cast<std::size_t>(scratch_rows) * static_cast<std::size_t>(max_tokens) *
           sizeof(float);
}

void lora_accum(const Tensor& x, const Tensor& a, const Tensor& b, Tensor& out, const Tensor& temp,
                float scale, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("lora_accum: x/out must be BF16");
    }
    if (a.dtype != DType::FP16 || b.dtype != DType::FP16) {
        throw std::invalid_argument("lora_accum: the adapter factors must be FP16");
    }
    if (temp.dtype != DType::FP32) {
        throw std::invalid_argument("lora_accum: the scratch must be FP32");
    }
    if (!x.is_contiguous() || !a.is_contiguous() || !b.is_contiguous() ||
        !temp.is_contiguous()) {
        throw std::invalid_argument("lora_accum: x/a/b/temp must be contiguous");
    }
    // `out` may be a row region of a wider plane: contiguous along its rows, any column stride.
    if (out.ne[0] > 1 && out.nb[0] != static_cast<std::int64_t>(sizeof(std::uint16_t))) {
        throw std::invalid_argument("lora_accum: out rows must be contiguous");
    }
    if (out.nb[1] < static_cast<std::int64_t>(out.ne[0]) * static_cast<std::int64_t>(sizeof(std::uint16_t))) {
        throw std::invalid_argument("lora_accum: out column stride is shorter than a row");
    }
    const std::int32_t inner  = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t rank   = a.ne[1];
    if (a.ne[0] != inner) {
        throw std::invalid_argument("lora_accum: A must be [rank, in_features] of x");
    }
    if (b.ne[0] != rank || b.ne[1] != out.ne[0]) {
        throw std::invalid_argument(
            "lora_accum: B is [" + std::to_string(b.ne[0]) + ", " + std::to_string(b.ne[1]) +
            "] but the base needs [" + std::to_string(rank) + ", " + std::to_string(out.ne[0]) +
            "] (x [" + std::to_string(x.ne[0]) + ", " + std::to_string(x.ne[1]) + "], a [" +
            std::to_string(a.ne[0]) + ", " + std::to_string(a.ne[1]) + "], out [" +
            std::to_string(out.ne[0]) + ", " + std::to_string(out.ne[1]) + "])");
    }
    if (out.ne[1] != tokens) {
        throw std::invalid_argument("lora_accum: out must hold one column per column of x");
    }
    // The fused route needs `rank` shared slots; the split route parks `splits * rank` partials.
    const std::int32_t needed = rank <= lora_accum_fused_max_rank()
                                    ? rank
                                    : lora_accum_scratch_rank(rank);
    if (temp.ne[0] < needed || temp.ne[1] < tokens) {
        throw std::invalid_argument("lora_accum: the scratch needs at least " +
                                    std::to_string(needed) + " rows for " +
                                    std::to_string(tokens) + " columns");
    }
    if (inner <= 0 || tokens <= 0 || rank <= 0 || out.ne[0] <= 0) { return; }
    if (x.data == nullptr || a.data == nullptr || b.data == nullptr || out.data == nullptr ||
        temp.data == nullptr) {
        throw std::invalid_argument("lora_accum: every operand needs storage");
    }
    detail::lora_accum_launch(x, a, b, out, temp, scale, stream);
}

} // namespace ninfer::ops
