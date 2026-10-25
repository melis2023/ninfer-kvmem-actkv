#include "models/qwen3_5/load/bindings.h"

#include "ninfer/ops/rmsnorm_rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {

void bind_dflash2(Bindings& b, DraftWeights& weights, const DraftConfig& config,
                  const TextConfig& target) {
    const auto& extra = config.dflash2.value();
    // The drafter's local attention normalizes and rotates q and k with the fused Op, which
    // compiles in its theta and epsilon; it has no three-call form to fall back to.
    if (!ops::rmsnorm_rope_constants_match(config.rope_theta, config.rms_norm_eps)) {
        throw std::invalid_argument(
            "DFlash2 drafter rope_theta and rms_norm_eps must be 1e7 and 1e-6 (the fused q/k "
            "norm and RoPE constants)");
    }
    const auto h      = target.hidden_size;
    const auto rows =
        artifact::checked_mul(2ULL * extra.conv_kernel_size, h / extra.conv_group_size,
                              "DFlash2 dynamic projection rows");
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p           = "dflash2/layers/" + std::to_string(i) + "/";
        const auto convolution = [&](const std::string& name) {
            return DynamicConvWeights{
                b.direct(p + name + "/base_kernel", {2, extra.conv_kernel_size, h}),
                b.parameter(p + name + "/kernel_projection", {rows, h}, {p + name + "_input"})};
        };
        weights.layers[i].attention_conv = convolution("attention_conv");
        weights.layers[i].mlp_conv       = convolution("mlp_conv");
    }
    weights.selector =
        SelectorWeights{b.parameter("dflash2/candidate_selector/hidden_projection",
                                    {extra.selector_rank, h}, {"dflash2/final_hidden"}),
                        b.direct("dflash2/candidate_selector/predecessor_codebook",
                                 {target.vocab_size, extra.selector_rank}),
                        b.direct("dflash2/candidate_selector/successor_codebook",
                                 {target.vocab_size, extra.selector_rank})};
}

} // namespace ninfer::models::qwen3_5::loading
