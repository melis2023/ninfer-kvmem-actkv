// The host-callable half of the fused [RMSNorm -> int8 quantise] edge. It lives in a CUDA
// translation unit so that a route launcher or a test can call the pass without including warp
// intrinsics; the declarations and the contract are in nvfp4_int8_fused_norm_quant.h.

#include "ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::unified {

void launch_rmsnorm_int8_quantize(const Tensor& residual, const Tensor& norm_weight, float eps,
                                  std::int8_t* codes, float* scales, std::int32_t addressed,
                                  cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (residual.ne[0] != fused_detail::kHidden) {
        throw std::invalid_argument("fused rmsnorm+quantise: the row must be 5120 wide");
    }
    if (tokens <= 0 || addressed < tokens) {
        throw std::invalid_argument("fused rmsnorm+quantise: invalid token interval");
    }
    if (codes == nullptr || scales == nullptr || residual.data == nullptr ||
        norm_weight.data == nullptr) {
        throw std::invalid_argument("fused rmsnorm+quantise: null buffer");
    }
    rmsnorm_int8_quantize_kernel<<<static_cast<unsigned>(addressed), fused_detail::kBlock, 0,
                                   stream>>>(
        static_cast<const __nv_bfloat162*>(residual.data),
        static_cast<const __nv_bfloat162*>(norm_weight.data), codes, scales, tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

void launch_gated_rmsnorm_int8_quantize(const Tensor& source, const Tensor& norm_weight,
                                        const Tensor& gate, float eps, std::int8_t* codes,
                                        float* scales, std::int32_t tokens, std::int32_t addressed,
                                        cudaStream_t stream) {
    // The caller names the token count: the model layer hands this a 3-D per-head view
    // {128, 48, T} whose memory is the same {6144, T} row the planes are indexed by, so deriving the
    // width from `ne[1]` would read the head count as a token count.
    const auto numel = [](const Tensor& t) {
        return static_cast<std::int64_t>(t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3];
    };
    if (numel(source) != static_cast<std::int64_t>(fused_detail::kGdnValueWidth) * tokens ||
        numel(gate) != numel(source)) {
        throw std::invalid_argument("fused gated rmsnorm+quantise: unexpected row geometry");
    }
    if (norm_weight.ne[0] != fused_detail::kGdnHeadDim) {
        throw std::invalid_argument(
            "fused gated rmsnorm+quantise: the norm weight must be one head wide");
    }
    if (tokens <= 0 || addressed < tokens) {
        throw std::invalid_argument("fused gated rmsnorm+quantise: invalid token interval");
    }
    if (codes == nullptr || scales == nullptr || source.data == nullptr ||
        norm_weight.data == nullptr || gate.data == nullptr) {
        throw std::invalid_argument("fused gated rmsnorm+quantise: null buffer");
    }
    gated_rmsnorm_int8_quantize_kernel<fused_detail::kGdnValueWidth, fused_detail::kGdnHeadDim,
                                       fused_detail::kGdnBlock>
        <<<static_cast<unsigned>(addressed), fused_detail::kGdnBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(source.data),
            static_cast<const __nv_bfloat162*>(norm_weight.data),
            static_cast<const __nv_bfloat162*>(gate.data), codes, scales, tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

void launch_nvfp4_int8_quantize32(const void* x, std::int32_t tokens, std::int32_t k,
                                  std::int8_t* codes, float* scales, std::int32_t addressed,
                                  cudaStream_t stream) {
    if (x == nullptr || codes == nullptr || scales == nullptr || tokens <= 0 || addressed < tokens ||
        (k % 32) != 0) {
        throw std::invalid_argument("nvfp4 int8 quantise: invalid call");
    }
    quantize_nvfp4_int8_grouped<<<static_cast<unsigned>(addressed), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), tokens, k, codes, scales);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::unified
