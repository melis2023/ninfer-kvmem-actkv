#include "models/flint/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::flint {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name) {
    if (!config.contains(name)) { throw ArtifactError(std::string("Flint config: missing ") + name); }
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(name) + ": exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name) {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string("Flint config: ") + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value <= 0) {
        throw ArtifactError(std::string("Flint config: ") + name + " is out of range");
    }
    return value;
}

} // namespace

TextConfig parse_text_config(const Json& value) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "FlintForCausalLM" || value.value("model_type", "") != "flint") {
        throw ArtifactError("Flint text config: architecture/model_type mismatch");
    }
    TextConfig out;
    out.hidden_size              = integer(value, "hidden_size");
    out.vocab_size               = integer(value, "vocab_size");
    out.num_hidden_layers        = integer(value, "num_hidden_layers");
    out.max_position_embeddings  = integer(value, "max_position_embeddings");
    out.rms_norm_eps             = real(value, "rms_norm_eps");
    out.rope_theta               = real(value, "rope_theta");
    out.num_attention_heads      = integer(value, "num_attention_heads");
    out.num_key_value_heads      = integer(value, "num_key_value_heads");
    out.head_dim                 = integer(value, "head_dim");
    out.sliding_window           = integer(value, "sliding_window");
    out.dense_layers             = integer(value, "dense_layers");
    out.intermediate_size        = integer(value, "intermediate_size");
    out.num_experts              = integer(value, "num_experts");
    out.top_k_experts            = integer(value, "num_experts_per_tok");
    out.moe_intermediate_size    = integer(value, "moe_intermediate_size");
    out.shared_intermediate_size = integer(value, "shared_expert_intermediate_size");
    const auto& layers = value.at("layer_types");
    const auto& rope   = value.at("rope_layers");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers || !rope.is_array() ||
        rope.size() != out.num_hidden_layers) {
        throw ArtifactError("Flint config: layer_types and rope_layers must name every block");
    }
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (layers[i] == "full_attention") {
            out.full_attention.push_back(true);
        } else if (layers[i] == "sliding_attention") {
            out.full_attention.push_back(false);
        } else {
            throw ArtifactError("Flint config: unknown layer type");
        }
        if (!rope[i].is_boolean()) { throw ArtifactError("Flint config: rope_layers must be booleans"); }
        out.rope.push_back(rope[i].get<bool>());
    }
    if (out.dense_layers >= out.num_hidden_layers) {
        throw ArtifactError("Flint config: every block is dense");
    }
    // The flash-attention and decode kernels implement width 128 with eight query heads per KV
    // head; the routing kernels at most 8 of 256 experts; the row kernels whole float4s of at most
    // 4096 values and q8_1 blocks of 32.
    if (out.head_dim != 128 || out.num_attention_heads != 8 * out.num_key_value_heads ||
        out.hidden_size % 256 != 0 || out.hidden_size > 4096 || out.num_experts > 256 ||
        out.top_k_experts == 0 || out.top_k_experts > 8 || out.top_k_experts > out.num_experts ||
        out.intermediate_size % 32 != 0 || out.moe_intermediate_size % 32 != 0 ||
        out.shared_intermediate_size % 32 != 0 || out.sliding_window == 0) {
        throw ArtifactError("Flint config: geometry outside the implemented kernels");
    }
    return out;
}

} // namespace ninfer::models::flint
