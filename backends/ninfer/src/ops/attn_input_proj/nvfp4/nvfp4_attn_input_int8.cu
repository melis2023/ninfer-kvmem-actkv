// The attention-input int8-grouped route: identity rows over the fused 14336-row NVFP4 weight
// (q, gate, k, v segments), its token tile padded to 128 and the surplus columns discarded.
// Shares the mainloop with the other NVFP4
// int8 routes via nvfp4_int8_grouped_mma.cuh; the segmented output adapter maps global rows to
// the four output tensors.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmq_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_mmvq_launch.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

constexpr std::int32_t kRows = 14336;
constexpr std::int32_t kCols = 5120;

// The mainloop half, identical whichever way the planes were filled: the quantise pass above it is
// the only thing the fused route replaces.
void launch_from_planes(const Weight& weight, const void* codes, const void* scales, Tensor& q,
                        Tensor& gate, Tensor& k, Tensor& v, bool narrow_route, bool mmq_route,
                        bool mmvq_route, std::int32_t tokens, cudaStream_t stream) {
    // The fused weight's rows are [query; key; gate; value], the order every other route's
    // segmented adapter maps (see nvfp4_attn_input_a16.cu); q, k, gate, v here, not the call's
    // argument order.
    const Nvfp4AttentionInputOutput output{static_cast<__nv_bfloat16*>(q.data),
                                           static_cast<__nv_bfloat16*>(k.data),
                                           static_cast<__nv_bfloat16*>(gate.data),
                                           static_cast<__nv_bfloat16*>(v.data)};
    // alpha = 1/weight_scale_divisor: the E2M1 1/2 already folded into the E4M3 scale decode,
    // where llama.cpp folds it too, so the half factor appears once and the store rounds once.
    const auto* w_codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* w_scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* x_int8   = reinterpret_cast<const std::int8_t*>(codes);
    const auto* x_scale  = reinterpret_cast<const float*>(scales);
    const float alpha    = 1.0F / weight.weight_scale_divisor;
    if (mmvq_route) {
        if (narrow_route) {
            narrow::launch<Nvfp4AttentionInputOutput>(w_codes, w_scales, x_int8, x_scale, alpha,
                                                      output, kCols, tokens, kRows, stream);
        } else {
            mmvq::launch<Nvfp4AttentionInputOutput>(w_codes, w_scales, x_int8, x_scale, alpha,
                                                   output, kCols, tokens, kRows, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mmq_route) {
        mmq::launch<Nvfp4AttentionInputOutput>(w_codes, w_scales, x_int8, alpha, output, kCols,
                                               tokens, kRows, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    nvfp4_int8_grouped_launch<Nvfp4AttentionInputOutput>(w_codes, w_scales, x_int8, x_scale, alpha,
                                                         output, kCols, tokens, kRows, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool nvfp4_attn_input_int8_tokens_supported(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) { return false; }
    if (min_tokens == max_tokens) { return nvfp4_int8_route_admits(min_tokens); }
    return nvfp4_int8_route_interval_admits(min_tokens, max_tokens);
}

void nvfp4_attn_input_int8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                  Nvfp4Int8Mainloop table, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    // Below the crossover on the unified table this call goes to the narrow kernel, whose column
    // count is the call's width -- so the two planes are the call's rows, and the surplus rows a
    // padded plane existed to hold would be quantised for nobody to read. Allocation and the
    // quantiser's grid both follow this one number; disagreeing halves leave columns uninitialised.
    const bool narrow_route = narrow::select(tokens, table);
    const std::int32_t padded = narrow::addressed_tokens(tokens, table);
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.group != 16 || weight.n != kRows || weight.k != kCols ||
        weight.qdata == nullptr || weight.scales == nullptr ||
        !nvfp4_int8_route_admits(tokens)) {
        throw std::invalid_argument("nvfp4 attn_input_proj int8: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 attn_input_proj int8");

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

    launch_from_planes(weight, codes.data, scales.data, q, gate, k, v, narrow_route, mmq_route,
                       mmvq_route, tokens, stream);
}

// The fused-edge entry: the same route, the same planes and the same mainloop, but the planes are
// filled by one [RMSNorm -> quantise] launch over `residual` instead of by the model layer's
// standalone rmsnorm followed by this route's own quantise pass. Reachable only when the device
// band answers "unified": the interleaved plane belongs to the own chain's MMQ and is not
// produced here.
void nvfp4_attn_input_int8_fused_norm_launch(const Tensor& residual, const Tensor& norm_weight,
                                             float eps, const Weight& weight, Tensor& q,
                                             Tensor& gate, Tensor& k, Tensor& v,
                                             WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                             cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const bool narrow_route   = narrow::select(tokens, table);
    const std::int32_t padded = narrow::addressed_tokens(tokens, table);
    if (weight.qtype != QType::NVFP4 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.group != 16 || weight.n != kRows || weight.k != kCols ||
        weight.qdata == nullptr || weight.scales == nullptr ||
        !nvfp4_int8_route_admits(tokens)) {
        throw std::invalid_argument("nvfp4 attn_input_proj int8 fused: unsupported profile");
    }
    (void)validate_nvfp4_weight(weight, "nvfp4 attn_input_proj int8 fused");
    if (residual.ne[0] != kCols) {
        throw std::invalid_argument("nvfp4 attn_input_proj int8 fused: residual width must be 5120");
    }

    auto scope = workspace.scope();
    const bool mmq_route  = mmq::select(kCols, tokens, table);
    const bool mmvq_route = mmvq::select(tokens);
    if (mmq_route) {
        throw std::invalid_argument(
            "nvfp4 attn_input_proj int8 fused: the interleaved MMQ plane is the own chain's");
    }
    DeviceSpan codes = workspace.alloc_bytes(static_cast<std::size_t>(padded) * kCols);
    DeviceSpan scales =
        workspace.alloc_bytes(static_cast<std::size_t>(padded) * (kCols / 32) * sizeof(float));
    launch_rmsnorm_int8_quantize(residual, norm_weight, eps,
                                 static_cast<std::int8_t*>(codes.data),
                                 static_cast<float*>(scales.data), padded, stream);
    launch_from_planes(weight, codes.data, scales.data, q, gate, k, v, narrow_route, mmq_route,
                       mmvq_route, tokens, stream);
}

} // namespace ninfer::ops::detail::unified
