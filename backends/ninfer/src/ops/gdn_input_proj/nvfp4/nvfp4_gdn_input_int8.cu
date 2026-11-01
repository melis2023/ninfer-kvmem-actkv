// The GDN input-projection int8-grouped route: identity rows over the fused 16384-row NVFP4
// weight (qkv and z segments), whole 128-token tiles only. Shares the mainloop with the other
// NVFP4 int8 routes via nvfp4_int8_grouped_mma.cuh; the segmented output adapter maps global
// rows to the two output tensors.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmq_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmvq_launch.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

constexpr std::int32_t kRows = 16384; // qkv [0,10240) then z [10240,16384)
constexpr std::int32_t kCols = 5120;

} // namespace

bool nvfp4_gdn_input_int8_tokens_supported(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) { return false; }
    if (min_tokens == max_tokens) { return nvfp4_int8_route_admits(min_tokens); }
    return nvfp4_int8_route_interval_admits(min_tokens, max_tokens);
}

void nvfp4_gdn_input_int8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                 cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // The unified table answers this width with the narrow kernel, whose columns are the call's width,
    // so the planes are the call's rows. Allocation and the quantiser's grid follow this one number.
    const bool narrow_route = narrow::select(tokens, table);
    const std::int32_t padded = narrow::addressed_tokens(tokens, table);
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.group != 16 || weight.n != kRows || weight.k != kCols ||
        weight.qdata == nullptr || weight.scales == nullptr ||
        !nvfp4_int8_route_admits(tokens)) {
        throw std::invalid_argument("nvfp4 gdn_input_proj int8: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj int8");

    auto scope = workspace.scope();
    // The interleaved plane the MMQ mainloop reads and the two separated planes the grouped
    // mainloop reads cost the same bytes per token; only the quantiser that fills them differs.
    const bool mmq_route  = mmq::select(kCols, tokens, table);
    // Below the crossover the MMVQ-shaped mainloop reads these same two planes, so it needs no
    // second quantisation pass and none of the interleaved form's routing.
    const bool mmvq_route = mmvq::select(tokens);
    DeviceSpan codes;
    DeviceSpan scales;
    if (mmq_route) {
        codes = workspace.alloc_bytes(mmq::activation_bytes(tokens, kCols));
    } else {
        codes  = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kCols);
        scales = workspace.alloc_bytes(static_cast<std::size_t>(padded) * (kCols / 32) *
                                       sizeof(float));
    }
    if (mmq_route) {
        mmq::quantize_activations(x.data, tokens, kCols, codes.data, stream);
    } else {
        quantize_nvfp4_int8_grouped<<<padded, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, kCols,
            reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<float*>(scales.data));
    }

    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                     static_cast<__nv_bfloat16*>(z.data)};
    // alpha = 1/weight_scale_divisor: the E2M1 1/2 already folded into the E4M3 scale decode,
    // where llama.cpp folds it too, so the half factor appears once and the store rounds once.
    const auto* w_codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* w_scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* x_int8   = reinterpret_cast<const std::int8_t*>(codes.data);
    const auto* x_scale  = reinterpret_cast<const float*>(scales.data);
    const float alpha    = 1.0F / weight.weight_scale_divisor;
    if (mmvq_route) {
        if (narrow_route) {
            narrow::launch<Nvfp4GdnInputOutput>(w_codes, w_scales, x_int8, x_scale, alpha, output, kCols,
                                               tokens, kRows, stream);
        } else {
            mmvq::launch<Nvfp4GdnInputOutput>(w_codes, w_scales, x_int8, x_scale, alpha, output, kCols,
                                              tokens, kRows, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mmq_route) {
        mmq::launch<Nvfp4GdnInputOutput>(w_codes, w_scales, x_int8, alpha, output, kCols, tokens,
                                         kRows, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    nvfp4_int8_grouped_launch<Nvfp4GdnInputOutput>(w_codes, w_scales, x_int8, x_scale, alpha, output,
                                                   kCols, tokens, kRows, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::unified
