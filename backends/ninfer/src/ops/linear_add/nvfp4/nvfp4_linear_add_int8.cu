// The LinearAdd int8-grouped route: identity rows over the down/o_proj geometries, the projection
// added into the residual in the epilogue. The token tile is padded to 128 and its surplus columns
// never stored -- which matters here more than elsewhere, because this store reads the residual
// back before writing it, so an unpredicated padding column would be an out-of-bounds read too. Shares the
// mainloop with the other NVFP4 int8 routes via nvfp4_int8_grouped_mma.cuh.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmq_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmvq_launch.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

// The epilogue adds the scaled projection into the residual tile before the single BF16 rounding,
// which is exactly linear_add's contract (A16 routes fuse the same addition).
struct Nvfp4LinearAddInt8Output {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        __nv_bfloat16* slot = data + static_cast<std::int64_t>(token) * rows + row;
        *slot = __float2bfloat16_rn(__bfloat162float(*slot) + value);
    }
};

} // namespace

bool nvfp4_linear_add_int8_tokens_supported(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) { return false; }
    if (min_tokens == max_tokens) { return nvfp4_int8_route_admits(min_tokens); }
    return nvfp4_int8_route_interval_admits(min_tokens, max_tokens);
}

void nvfp4_linear_add_int8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                  cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // The unified table answers this width with the narrow kernel, whose columns are the call's width,
    // so the planes are the call's rows. Allocation and the quantiser's grid follow this one number.
    const bool narrow_route = narrow::select(tokens, table);
    const std::int32_t padded = narrow::addressed_tokens(tokens, table);
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.group != 16 || weight.n != 5120 || (weight.k != 6144 && weight.k != 17408) ||
        weight.qdata == nullptr || weight.scales == nullptr ||
        !nvfp4_int8_route_admits(tokens)) {
        throw std::invalid_argument("nvfp4 linear_add int8: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 linear_add int8");

    auto scope = workspace.scope();
    // The interleaved plane the MMQ mainloop reads and the two separated planes the grouped
    // mainloop reads cost the same bytes per token; only the quantiser that fills them differs.
    const bool mmq_route  = mmq::select(weight.k, tokens, table);
    // Below the crossover the MMVQ-shaped mainloop reads these same two planes, so it needs no
    // second quantisation pass and none of the interleaved form's routing.
    const bool mmvq_route = mmvq::select(tokens);
    DeviceSpan codes;
    DeviceSpan scales;
    if (mmq_route) {
        codes = workspace.alloc_bytes(mmq::activation_bytes(tokens, weight.k));
    } else {
        codes  = workspace.alloc_bytes(static_cast<std::size_t>(padded) * weight.k);
        scales = workspace.alloc_bytes(static_cast<std::size_t>(padded) * (weight.k / 32) *
                                       sizeof(float));
    }
    if (mmq_route) {
        mmq::quantize_activations(x.data, tokens, weight.k, codes.data, stream);
    } else {
        quantize_nvfp4_int8_grouped<<<padded, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, weight.k,
            reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<float*>(scales.data));
    }

    const Nvfp4LinearAddInt8Output output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto* w_codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* w_scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* x_int8   = reinterpret_cast<const std::int8_t*>(codes.data);
    const auto* x_scale  = reinterpret_cast<const float*>(scales.data);
    // alpha = 1/weight_scale_divisor: the E2M1 1/2 already folded into the E4M3 scale decode, where
    // llama.cpp folds it too, so the half factor appears once and the store rounds once.
    const float alpha = 1.0F / weight.weight_scale_divisor;
    if (mmvq_route) {
        if (narrow_route) {
            narrow::launch<Nvfp4LinearAddInt8Output>(w_codes, w_scales, x_int8, x_scale, alpha,
                                                     output, weight.k, tokens, weight.n, stream);
        } else {
            mmvq::launch<Nvfp4LinearAddInt8Output>(w_codes, w_scales, x_int8, x_scale, alpha, output,
                                                   weight.k, tokens, weight.n, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mmq_route) {
        mmq::launch<Nvfp4LinearAddInt8Output>(w_codes, w_scales, x_int8, alpha, output, weight.k,
                                              tokens, weight.n, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    nvfp4_int8_grouped_launch<Nvfp4LinearAddInt8Output>(w_codes, w_scales, x_int8, x_scale, alpha,
                                                        output, weight.k, tokens, weight.n, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::unified
