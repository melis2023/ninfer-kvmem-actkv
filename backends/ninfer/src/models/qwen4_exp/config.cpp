#include "models/qwen4_exp/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name, bool positive = true) {
    const auto n = artifact::require_u64(config.at(name), name, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(name) + ": exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

// The geometry the engine's Ops implement; any other value is refused by name.
void require_value(std::uint32_t actual, std::uint32_t implemented, const char* name) {
    if (actual != implemented) {
        throw ArtifactError(std::string(name) + " is " + std::to_string(actual) +
                            "; this engine implements " + std::to_string(implemented));
    }
}

} // namespace

TextConfig parse_text_config(const Json& value) {
    artifact::require_members(
        value,
        {"architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
         "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps", "eos_token_id",
         "layer_types", "num_attention_heads", "num_key_value_heads", "head_dim",
         "rope_parameters", "indexer_n_heads", "indexer_head_dim", "indexer_compress_ratio",
         "indexer_budget", "linear_num_key_heads", "linear_key_head_dim",
         "linear_num_value_heads", "linear_value_head_dim", "linear_conv_kernel_dim",
         "num_experts", "num_experts_per_tok", "moe_intermediate_size",
         "shared_expert_intermediate_size", "hc_count", "hc_lowrank", "ple_layers",
         "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
         "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "ngram_seed"},
        {}, "Qwen4Exp text config");
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "Qwen4ExpForCausalLM" || value.at("model_type") != "qwen4_exp_text") {
        throw ArtifactError("Qwen4Exp text config: architecture/model_type mismatch");
    }
    TextConfig out;
    out.hidden_size             = integer(value, "hidden_size");
    out.vocab_size              = integer(value, "vocab_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.max_position_embeddings = integer(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    if (!value.at("tie_word_embeddings").is_boolean()) {
        throw ArtifactError("tie_word_embeddings must be boolean");
    }
    out.tie_word_embeddings = value.at("tie_word_embeddings").get<bool>();
    const auto eos          = integer(value, "eos_token_id", false);
    if (eos >= out.vocab_size) { throw ArtifactError("eos_token_id exceeds the vocabulary"); }
    out.eos_token_id = static_cast<std::int32_t>(eos);

    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must describe every block");
    }
    for (const auto& layer : layers) {
        if (layer == "linear_attention") {
            out.layer_types.push_back(MixerKind::GatedDeltaNet);
        } else if (layer == "full_attention") {
            out.layer_types.push_back(MixerKind::SparseAttention);
        } else {
            throw ArtifactError("unknown layer_type");
        }
    }

    out.num_attention_heads    = integer(value, "num_attention_heads");
    out.num_key_value_heads    = integer(value, "num_key_value_heads");
    out.head_dim               = integer(value, "head_dim");
    out.indexer_n_heads        = integer(value, "indexer_n_heads");
    out.indexer_head_dim       = integer(value, "indexer_head_dim");
    out.indexer_compress_ratio = integer(value, "indexer_compress_ratio");
    out.indexer_budget         = integer(value, "indexer_budget");
    const auto& rope           = value.at("rope_parameters");
    artifact::require_members(rope, {"rope_theta", "partial_rotary_factor", "mrope_section"}, {},
                              "Qwen4Exp RoPE");
    out.rope_theta            = positive_float(rope, "rope_theta");
    out.partial_rotary_factor = positive_float(rope, "partial_rotary_factor");
    const auto& sections      = rope.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("mrope_section must hold three sections");
    }
    std::uint32_t pairs = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] =
            static_cast<std::uint32_t>(artifact::require_u64(sections[i], "MRoPE section", false));
        pairs += out.mrope_section[i];
    }
    if (static_cast<double>(pairs) * 2 != out.head_dim * double(out.partial_rotary_factor)) {
        throw ArtifactError("MRoPE sections differ from the rotary width");
    }

    out.linear_num_key_heads   = integer(value, "linear_num_key_heads");
    out.linear_key_head_dim    = integer(value, "linear_key_head_dim");
    out.linear_num_value_heads = integer(value, "linear_num_value_heads");
    out.linear_value_head_dim  = integer(value, "linear_value_head_dim");
    out.linear_conv_kernel_dim = integer(value, "linear_conv_kernel_dim");
    out.num_experts                     = integer(value, "num_experts");
    out.num_experts_per_tok             = integer(value, "num_experts_per_tok");
    out.moe_intermediate_size           = integer(value, "moe_intermediate_size");
    out.shared_expert_intermediate_size = integer(value, "shared_expert_intermediate_size");
    out.hc_count             = integer(value, "hc_count");
    out.hc_lowrank           = integer(value, "hc_lowrank");
    out.ple_embed_dim        = integer(value, "ple_embed_dim");
    out.ple_conv_kernel_size = integer(value, "ple_conv_kernel_size");
    out.ngram                = NgramHashSpec{
        .vocab_size      = out.vocab_size,
        .ngram_size      = integer(value, "ngram_size"),
        .heads_per_ngram = integer(value, "heads_per_ngram"),
        .ple_layer_index = 0,
        .vocab_base      = artifact::require_u64(value.at("ngram_vocab_size_base"),
                                                 "ngram_vocab_size_base", true),
        .divisible_by    = artifact::require_u64(value.at("make_ngram_vocab_size_divisible_by"),
                                                 "make_ngram_vocab_size_divisible_by", true),
        .seed = artifact::require_u64(value.at("ngram_seed"), "ngram_seed", false),
    };
    const auto& ple = value.at("ple_layers");
    if (!ple.is_array()) { throw ArtifactError("ple_layers must be an array of blocks"); }
    for (const auto& block : ple) {
        const auto index = artifact::require_u64(block, "PLE block", false);
        if (index >= out.num_hidden_layers ||
            out.layer_types[index] != MixerKind::GatedDeltaNet ||
            (!out.ple_layers.empty() && index <= out.ple_layers.back())) {
            throw ArtifactError("ple_layers must be increasing Gated DeltaNet blocks");
        }
        out.ple_layers.push_back(static_cast<std::uint32_t>(index));
    }

    // The implemented geometry (Ops registered for these shapes only).
    require_value(out.hidden_size, 2560, "hidden_size");
    require_value(out.num_attention_heads, 24, "num_attention_heads");
    require_value(out.num_key_value_heads, 2, "num_key_value_heads");
    require_value(out.head_dim, 256, "head_dim");
    require_value(out.indexer_n_heads, 4, "indexer_n_heads");
    require_value(out.indexer_head_dim, 128, "indexer_head_dim");
    require_value(out.indexer_compress_ratio, 4, "indexer_compress_ratio");
    require_value(out.indexer_budget, 2048, "indexer_budget");
    require_value(out.linear_num_key_heads, 16, "linear_num_key_heads");
    require_value(out.linear_key_head_dim, 128, "linear_key_head_dim");
    require_value(out.linear_num_value_heads, 48, "linear_num_value_heads");
    require_value(out.linear_value_head_dim, 128, "linear_value_head_dim");
    require_value(out.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim");
    // 512 experts, or fewer in an expert-pruned release (the router keeps one row per expert).
    if (out.num_experts < out.num_experts_per_tok || out.num_experts > 512) {
        throw ArtifactError("num_experts is " + std::to_string(out.num_experts) +
                            "; this engine implements 10 to 512");
    }
    require_value(out.num_experts_per_tok, 10, "num_experts_per_tok");
    require_value(out.moe_intermediate_size, 640, "moe_intermediate_size");
    require_value(out.shared_expert_intermediate_size, 640, "shared_expert_intermediate_size");
    require_value(out.hc_count, 4, "hc_count");
    require_value(out.hc_lowrank, 320, "hc_lowrank");
    require_value(out.ple_embed_dim, 2560, "ple_embed_dim");
    require_value(out.ple_conv_kernel_size, 4, "ple_conv_kernel_size");
    require_value(out.ngram.ngram_size, 3, "ngram_size");
    require_value(out.ngram.heads_per_ngram, 8, "heads_per_ngram");
    require_value(out.ple_embed_dim / out.ngram_heads(), 160, "n-gram head width");
    return out;
}

MtpConfig parse_mtp_config(const Json& value, const TextConfig& text) {
    artifact::require_members(value, {"architectures", "rope_theta", "num_experts"}, {},
                              "Qwen4Exp MTP config");
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "Qwen4ExpMTP") {
        throw ArtifactError("Qwen4Exp MTP config: architecture mismatch");
    }
    MtpConfig out;
    out.rope_theta  = positive_float(value, "rope_theta");
    out.num_experts = integer(value, "num_experts");
    // The sparse-attention kernels rotate at the text model's theta.
    if (out.rope_theta != text.rope_theta) {
        throw ArtifactError("the MTP block's rope_theta differs from the text model's");
    }
    if (out.num_experts < text.num_experts_per_tok || out.num_experts > 512) {
        throw ArtifactError("the MTP block keeps " + std::to_string(out.num_experts) +
                            " experts; this engine implements 10 to 512");
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
