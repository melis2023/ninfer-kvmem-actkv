#include "models/gemma4/config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::gemma4 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name) {
    if (!config.contains(name)) { throw ArtifactError(std::string("Gemma 4 config: missing ") + name); }
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(name) + ": exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name, bool positive) {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string("Gemma 4 config: ") + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value < 0 || (positive && value == 0)) {
        throw ArtifactError(std::string("Gemma 4 config: ") + name + " is out of range");
    }
    return value;
}

} // namespace

TextConfig parse_text_config(const Json& value) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "Gemma4ForCausalLM" || value.value("model_type", "") != "gemma4_text") {
        throw ArtifactError("Gemma 4 text config: architecture/model_type mismatch");
    }
    TextConfig out;
    out.hidden_size             = integer(value, "hidden_size");
    out.vocab_size              = integer(value, "vocab_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.max_position_embeddings = integer(value, "max_position_embeddings");
    out.rms_norm_eps            = real(value, "rms_norm_eps", true);
    out.final_logit_softcapping = real(value, "final_logit_softcapping", false);
    out.num_attention_heads     = integer(value, "num_attention_heads");
    out.num_key_value_heads     = integer(value, "num_key_value_heads");
    out.head_dim                = integer(value, "head_dim");
    out.sliding_window          = integer(value, "sliding_window");
    out.rope_theta_sliding      = real(value, "rope_theta_sliding", true);
    out.num_global_key_value_heads = integer(value, "num_global_key_value_heads");
    out.global_head_dim            = integer(value, "global_head_dim");
    out.rope_theta_full            = real(value, "rope_theta_full", true);
    const float factor             = real(value, "partial_rotary_factor_full", true);
    out.rotary_pairs_full =
        static_cast<std::uint32_t>(factor * float(out.global_head_dim) / 2.0f + 1e-3f);
    out.intermediate_size = integer(value, "intermediate_size");
    out.moe               = value.value("enable_moe_block", false);
    if (out.moe) {
        out.num_experts           = integer(value, "num_experts");
        out.top_k_experts         = integer(value, "top_k_experts");
        out.moe_intermediate_size = integer(value, "moe_intermediate_size");
        if (out.top_k_experts > 8 || out.top_k_experts > out.num_experts ||
            out.num_experts > 1024 || out.moe_intermediate_size % 32 != 0) {
            throw ArtifactError("Gemma 4 config: unsupported expert geometry");
        }
    }
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("Gemma 4 config: layer_types must name every block");
    }
    for (const auto& kind : layers) {
        if (kind == "full_attention") {
            out.full_attention.push_back(true);
        } else if (kind == "sliding_attention") {
            out.full_attention.push_back(false);
        } else {
            throw ArtifactError("Gemma 4 config: unknown layer type");
        }
    }
    // The flash-attention kernels implement widths 256 (two query heads per KV head) and 512
    // (eight); the decode kernels the same.
    if (out.head_dim != 256 || out.global_head_dim != 512 ||
        out.num_attention_heads != 2 * out.num_key_value_heads ||
        out.num_attention_heads != 8 * out.num_global_key_value_heads ||
        out.rotary_pairs_full == 0 || out.rotary_pairs_full > out.global_head_dim / 2 ||
        out.hidden_size % 256 != 0 || out.intermediate_size % 32 != 0) {
        throw ArtifactError("Gemma 4 config: attention geometry outside the implemented kernels");
    }
    return out;
}

AssistantConfig parse_assistant_config(const Json& value, const TextConfig& target) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "Gemma4AssistantForCausalLM") {
        throw ArtifactError("Gemma 4 assistant config: architecture mismatch");
    }
    AssistantConfig out;
    out.hidden_size          = integer(value, "hidden_size");
    out.backbone_hidden_size = integer(value, "backbone_hidden_size");
    out.intermediate_size    = integer(value, "intermediate_size");
    out.vocab_size           = integer(value, "vocab_size");
    out.num_attention_heads  = integer(value, "num_attention_heads");
    out.rms_norm_eps         = real(value, "rms_norm_eps", true);
    const auto& layers       = value.at("layer_types");
    if (!layers.is_array() || layers.size() != integer(value, "num_hidden_layers")) {
        throw ArtifactError("Gemma 4 assistant config: layer_types must name every block");
    }
    for (const auto& kind : layers) {
        if (kind != "full_attention" && kind != "sliding_attention") {
            throw ArtifactError("Gemma 4 assistant config: unknown layer type");
        }
        out.full_attention.push_back(kind == "full_attention");
    }
    const auto full = std::count(out.full_attention.begin(), out.full_attention.end(), true);
    const auto sliding = std::ptrdiff_t(out.full_attention.size()) - full;
    if (full == 0 || sliding == 0 || std::count(target.full_attention.begin(),
                                                target.full_attention.end(), true) == 0) {
        throw ArtifactError("Gemma 4 assistant config: needs both kinds of attention block");
    }
    // It reads the target's caches with the target's kernels: the same heads and widths.
    if (out.backbone_hidden_size != target.hidden_size || out.vocab_size != target.vocab_size ||
        out.num_attention_heads != target.num_attention_heads ||
        integer(value, "num_key_value_heads") != target.num_key_value_heads ||
        integer(value, "num_global_key_value_heads") != target.num_global_key_value_heads ||
        integer(value, "head_dim") != target.head_dim ||
        integer(value, "global_head_dim") != target.global_head_dim ||
        integer(value, "sliding_window") != target.sliding_window ||
        out.hidden_size % 256 != 0 || out.intermediate_size % 32 != 0) {
        throw ArtifactError("Gemma 4 assistant config: geometry differs from the target's");
    }
    return out;
}

VisionConfig parse_vision_config(const Json& value, const TextConfig& target) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "Gemma4VisionModel") {
        throw ArtifactError("Gemma 4 vision config: architecture mismatch");
    }
    VisionConfig out;
    out.hidden_size             = integer(value, "hidden_size");
    out.intermediate_size       = integer(value, "intermediate_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.num_attention_heads     = integer(value, "num_attention_heads");
    out.head_dim                = integer(value, "head_dim");
    out.patch_size              = integer(value, "patch_size");
    out.pooling_kernel_size     = integer(value, "pooling_kernel_size");
    out.position_embedding_size = integer(value, "position_embedding_size");
    out.projection_dim          = integer(value, "projection_dim");
    out.default_output_length   = integer(value, "default_output_length");
    out.rope_theta              = real(value, "rope_theta", true);
    out.rms_norm_eps            = real(value, "rms_norm_eps", true);
    out.standardize             = value.value("standardize", false);
    out.bidirectional_attention = value.value("bidirectional_attention", false);
    const auto token = [&](const char* name) {
        const std::uint32_t id = integer(value, name);
        if (id >= target.vocab_size) {
            throw ArtifactError(std::string("Gemma 4 vision config: ") + name +
                                " is outside the vocabulary");
        }
        return static_cast<std::int32_t>(id);
    };
    out.image_token_id = token("image_token_id");
    out.video_token_id = token("video_token_id");
    out.boi_token_id   = token("boi_token_id");
    out.eoi_token_id   = token("eoi_token_id");
    // The packed attention kernel's registered profile is 16 heads of 72; the rest follows the
    // released tower.
    if (out.num_attention_heads != 16 || out.head_dim != 72 ||
        out.hidden_size != out.num_attention_heads * out.head_dim ||
        out.projection_dim != target.hidden_size || out.patch_size != 16 ||
        out.pooling_kernel_size == 0 || out.pooling_kernel_size > 4 ||
        out.intermediate_size % 16 != 0 || out.position_embedding_size == 0 ||
        out.num_hidden_layers == 0 || out.default_output_length == 0) {
        throw ArtifactError("Gemma 4 vision config: geometry outside the implemented kernels");
    }
    return out;
}

} // namespace ninfer::models::gemma4
