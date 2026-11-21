#include "models/dspine/config.h"

#include "artifact/reader.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::dspine {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name) {
    if (!config.contains(name)) { throw ArtifactError(std::string("DSpine config: missing ") + name); }
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string("DSpine config: ") + name + " exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name) {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string("DSpine config: ") + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value <= 0) {
        throw ArtifactError(std::string("DSpine config: ") + name + " is out of range");
    }
    return value;
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw ArtifactError("DSpine config: " + message); }
}

// The windows the sliding-window attention Op registers.
constexpr std::uint32_t kWindows[] = {2048, 4096, 8192};

Config parse_config(const Json& value) {
    require(value.is_object(), "expected an object");
    const auto& architectures = value.at("architectures");
    require(architectures.is_array() && architectures.size() == 1 &&
                architectures[0] == "DSpineDraftModel" && value.value("model_type", "") == "qwen3",
            "architecture/model_type mismatch");
    Config out;
    out.hidden_size             = integer(value, "hidden_size");
    out.intermediate_size       = integer(value, "intermediate_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.num_attention_heads     = integer(value, "num_attention_heads");
    out.num_key_value_heads     = integer(value, "num_key_value_heads");
    out.head_dim                = integer(value, "head_dim");
    out.max_position_embeddings = integer(value, "max_position_embeddings");
    out.vocab_size              = integer(value, "vocab_size");
    out.num_target_layers       = integer(value, "num_target_layers");
    out.block_size              = integer(value, "block_size");
    out.sliding_window          = integer(value, "sliding_window");
    out.rms_norm_eps            = real(value, "rms_norm_eps");
    require(value.contains("rope_parameters") && value.at("rope_parameters").is_object(),
            "missing rope_parameters");
    out.rope_theta = real(value.at("rope_parameters"), "rope_theta");
    const auto& layers = value.at("layer_types");
    require(layers.is_array() && layers.size() == out.num_hidden_layers &&
                std::all_of(layers.begin(), layers.end(),
                            [](const Json& kind) { return kind == "sliding_attention"; }),
            "every layer must be sliding_attention");
    require(value.contains("dspine_config") && value.at("dspine_config").is_object(),
            "missing dspine_config");
    const auto& bridge = value.at("dspine_config");
    const auto& taps   = bridge.at("target_layer_ids");
    require(taps.is_array() && !taps.empty(), "target_layer_ids must name the taps");
    for (const auto& tap : taps) {
        const auto layer = artifact::require_u64(tap, "target_layer_ids", false);
        require(layer < out.num_target_layers, "a tap lies outside the target's layers");
        out.target_layer_ids.push_back(static_cast<std::uint32_t>(layer));
    }
    auto sorted = out.target_layer_ids;
    std::sort(sorted.begin(), sorted.end());
    require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
            "target_layer_ids repeat a layer");
    out.mask_token_id = static_cast<std::uint32_t>(
        artifact::require_u64(bridge.at("mask_token_id"), "mask_token_id", false));
    out.code_size  = integer(bridge, "code_size");
    out.rank       = integer(bridge, "rank");
    out.head_top_k = integer(bridge, "head_top_k");
    out.bridge_eps = real(bridge, "bridge_eps");
    require(out.mask_token_id < out.vocab_size, "mask token outside the vocabulary");
    // The geometry the Ops implement: the attention profile (D=128, 32 query and 8 KV heads), the
    // bridge Ops' rank and vector widths, and a block the verification frame takes.
    require(out.head_dim == 128 && out.num_attention_heads == 32 && out.num_key_value_heads == 8,
            "the drafter's attention must be 32 query and 8 KV heads of 128");
    require(out.hidden_size % 8 == 0 && out.code_size % 8 == 0,
            "hidden and code widths must be multiples of 8");
    require(out.rank % 32 == 0 && out.rank <= 512, "the bridge rank must be a multiple of 32 up "
                                                   "to 512");
    require(out.head_top_k == 16, "the refinement chains the head's top 16");
    require(out.block_size >= 2 && out.block_size <= 16, "the block must be 2..16 wide");
    require(std::find(std::begin(kWindows), std::end(kWindows), out.sliding_window) !=
                std::end(kWindows),
            "the sliding window must be 2048, 4096 or 8192");
    require(out.rms_norm_eps == 1e-6F && out.rope_theta == 1e7F,
            "the drafter's norm epsilon must be 1e-6 and its RoPE theta 1e7");
    return out;
}

bool is_dspine_artifact(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    if (components.size() != 1) { return false; }
    const auto found = components.find("dspine");
    if (found == components.end()) { return false; }
    const auto& config = found->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "DSpineDraftModel";
}

} // namespace

Config read_config(const artifact::Reader& reader, const std::filesystem::path& path) {
    if (!is_dspine_artifact(reader)) {
        throw ArtifactError(path.string() + ": not a DSpine drafter artifact (its only component "
                                            "must be dspine; convert one with tools/convert)");
    }
    return parse_config(reader.directory().component("dspine").config);
}

Config read_config(const std::filesystem::path& path) {
    return read_config(artifact::Reader(path), path);
}

void check_target(const Config& config, std::uint64_t hidden_size, std::uint64_t vocab_size,
                  std::uint64_t layers, std::string_view family) {
    const auto fail = [&](const std::string& what) {
        throw std::invalid_argument("--draft-model: the DSpine drafter was trained for another "
                                    "target than this " + std::string(family) + " model (" + what +
                                    ")");
    };
    if (config.hidden_size != hidden_size) {
        fail("hidden width " + std::to_string(config.hidden_size) + ", the model's " +
             std::to_string(hidden_size));
    }
    if (config.vocab_size != vocab_size) {
        fail("vocabulary " + std::to_string(config.vocab_size) + ", the model's " +
             std::to_string(vocab_size));
    }
    if (config.num_target_layers != layers) {
        fail(std::to_string(config.num_target_layers) + " target layers, the model's " +
             std::to_string(layers));
    }
}

std::uint32_t ring_capacity(const Config& config, std::uint32_t max_context) {
    // A block at the last frontier reaches block_size - 1 positions past it.
    const std::uint64_t reach = std::uint64_t(max_context) + config.block_size;
    for (const std::uint32_t window : kWindows) {
        if (window >= config.sliding_window) { break; }
        if (window >= reach) { return window; }
    }
    return config.sliding_window;
}

std::uint64_t ring_bytes(const Config& config, std::uint32_t capacity) {
    return std::uint64_t(config.num_hidden_layers) * 2 * config.key_width() * capacity * 2;
}

std::uint64_t head_write_bytes(const Config& config) {
    return std::uint64_t(config.rank) * config.vocab_size * 2;
}

std::uint64_t weight_bytes(const std::filesystem::path& path) {
    return artifact::Reader(path).directory().payload_bytes;
}

} // namespace ninfer::models::dspine
