#include "models/qwen4_exp/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp {
namespace {

using qwen3_5::loading::Bindings;
using Shape = artifact::Shape;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config,
                                 bool vision) {
    const auto resource = [&](std::string_view role, const char* component = "text") {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    if (vision) {
        out.preprocessor_config_json       = resource("preprocessor_config.json", "vision");
        out.video_preprocessor_config_json = resource("video_preprocessor_config.json", "vision");
    }
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    for (const auto token : out.tokenizer->default_stop_token_ids()) {
        if (!out.tokenizer->is_valid_token(token)) {
            throw artifact::ArtifactError("stop token is outside the public token domain");
        }
    }
    return out;
}

HyperConnectionWeights bind_hc(Bindings& b, const TextConfig& c, const std::string& p,
                               bool inject) {
    const std::uint64_t width = std::uint64_t(c.hc_count) * c.hidden_size;
    HyperConnectionWeights out;
    out.norm = b.direct(p + "norm", {width});
    out.down = b.parameter(p + "down", {c.hc_lowrank, width}, {p + "normalized"});
    out.up   = b.parameter(p + "up", {width, c.hc_lowrank}, {p + "low"});
    if (inject) { out.inject = b.parameter(p + "inject", {c.hc_count, width}, {p + "normalized"}); }
    return out;
}

GdnWeights bind_gdn(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h  = c.hidden_size;
    const std::uint64_t kd = std::uint64_t(c.linear_num_key_heads) * c.linear_key_head_dim;
    const std::uint64_t vd = std::uint64_t(c.linear_num_value_heads) * c.linear_value_head_dim;
    const std::string g = p + "gdn/", input = p + "mixer_input";
    GdnWeights out;
    out.a_log        = b.direct(g + "a_log", {c.linear_num_value_heads}, QType::FP32);
    out.dt_bias      = b.direct(g + "dt_bias", {c.linear_num_value_heads}, QType::FP32);
    out.convolution  = b.direct(g + "convolution", {c.linear_conv_kernel_dim, 2 * kd + vd});
    out.a_projection = b.parameter(g + "a_projection", {c.linear_num_value_heads, h}, {input});
    out.b_projection = b.parameter(g + "b_projection", {c.linear_num_value_heads, h}, {input});
    out.query        = b.parameter(g + "query", {kd, h}, {input});
    out.key          = b.parameter(g + "key", {kd, h}, {input});
    out.value        = b.parameter(g + "value", {vd, h}, {input});
    out.z            = b.parameter(g + "z", {vd, h}, {input});
    out.norm         = b.direct(g + "norm", {c.linear_value_head_dim});
    out.output       = b.parameter(g + "output", {h, vd}, {g + "gated_output"});
    return out;
}

SparseAttentionWeights bind_attention(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t h  = c.hidden_size;
    const std::uint64_t q  = std::uint64_t(c.num_attention_heads) * c.head_dim;
    const std::uint64_t kv = std::uint64_t(c.num_key_value_heads) * c.head_dim;
    const std::string a = p + "attention/", i = p + "indexer/", input = p + "mixer_input";
    SparseAttentionWeights out;
    out.query       = b.parameter(a + "query", {q, h}, {input});
    out.gate        = b.parameter(a + "gate", {q, h}, {input});
    out.key         = b.parameter(a + "key", {kv, h}, {input});
    out.value       = b.parameter(a + "value", {kv, h}, {input});
    out.query_norm  = b.direct(a + "query_norm", {c.head_dim});
    out.key_norm    = b.direct(a + "key_norm", {c.head_dim});
    out.output      = b.parameter(a + "output", {h, q}, {a + "gated_output"});
    out.index_query = b.parameter(
        i + "query", {std::uint64_t(c.indexer_n_heads) * c.indexer_head_dim, h}, {input});
    out.index_key        = b.parameter(i + "key", {c.indexer_head_dim, h}, {input});
    out.index_query_norm = b.direct(i + "query_norm", {c.indexer_head_dim});
    out.index_key_norm   = b.direct(i + "key_norm", {c.indexer_head_dim});
    return out;
}

// Where a parameter's rows sit in the artifact's files, without demanding its object.
ExpertLocation locate(const artifact::Reader& reader, const std::string& name, std::uint64_t rows,
                      std::uint64_t k) {
    const auto& bindings = reader.directory().bindings;
    const auto found     = bindings.find(name);
    if (found == bindings.end() || found->second.parts.size() != 1) {
        throw artifact::ArtifactError(name + ": a disk-resident expert must be one stored region");
    }
    const auto& part     = found->second.parts.front();
    const auto& geometry = reader.geometry(part.object);
    if (!is_gguf(geometry.format) || geometry.shape.size() != 2 || geometry.shape[1] != k ||
        part.end - part.begin != rows * k || part.begin % k != 0) {
        throw artifact::ArtifactError(name + ": disk-resident experts must be GGUF block rows");
    }
    const auto block              = gguf_block_shape(geometry.format);
    const std::uint64_t row_bytes = k / block.elements * block.bytes;
    const std::uint64_t logical =
        reader.directory().tensor(part.object).offset + part.begin / k * row_bytes;
    const auto segments = reader.segments(logical, rows * row_bytes);
    if (segments.empty() || segments.size() > 2) {
        throw artifact::ArtifactError(name + ": a disk-resident expert spans more than two files");
    }
    ExpertLocation out{.runs      = {},
                       .bytes     = rows * row_bytes,
                       .format    = geometry.format,
                       .row_bytes = static_cast<std::int64_t>(row_bytes),
                       .rows      = static_cast<std::int32_t>(rows)};
    for (std::size_t i = 0; i < segments.size(); ++i) {
        out.runs[i] = {segments[i].file_index, segments[i].file_offset, segments[i].bytes};
    }
    return out;
}

// Zero bytes after each down bank; see bind_moe.
constexpr std::uint64_t kExpertTail = 256;

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& p,
                    ExpertResidency residency) {
    const std::uint64_t h = c.hidden_size, w = c.moe_intermediate_size;
    const std::string m = p + "moe/", input = p + "ffn_input";
    MoeWeights out;
    out.router       = b.parameter(m + "router", {c.num_experts, h}, {input});
    out.shared_score = b.parameter(m + "shared_score", {1, h}, {input});
    const artifact::Residency experts = residency == ExpertResidency::Host
                                            ? artifact::Residency::Pinned
                                            : artifact::Residency::Device;
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        const std::string x = m + "experts/" + std::to_string(e) + "/";
        if (residency == ExpertResidency::Disk) {
            const auto& reader = b.binder.reader();
            out.located_gate.push_back(locate(reader, x + "gate", w, h));
            out.located_up.push_back(locate(reader, x + "up", w, h));
            out.located_down.push_back(locate(reader, x + "down", h, w));
            continue;
        }
        out.gate.push_back(b.parameter(x + "gate", {w, h}, {input}, {}, experts));
        out.up.push_back(b.parameter(x + "up", {w, h}, {input}, {}, experts));
        out.down.push_back(b.parameter(x + "down", {h, w}, {x + "product"}, {}, experts));
    }
    const std::uint64_t s = c.shared_expert_intermediate_size;
    out.shared_gate       = b.parameter(m + "shared/gate", {s, h}, {input});
    out.shared_up         = b.parameter(m + "shared/up", {s, h}, {input});
    out.shared_down       = b.parameter(m + "shared/down", {h, s}, {m + "shared/product"});
    // The expert matrix kernel reads a down row's 640 values in three 256-value steps, past the end
    // of each bank's last row: zeros follow every down bank in device memory.
    std::vector<WeightId> downs{out.shared_down};
    if (residency == ExpertResidency::Device) {
        downs.insert(downs.end(), out.down.begin(), out.down.end());
    }
    for (const WeightId id : downs) {
        for (const auto& part : b.at(id).reference.binding.parts) {
            b.binder.device_tail(part.object, kExpertTail);
        }
    }
    return out;
}

PleWeights bind_ple(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t width = std::uint64_t(c.hc_count) * c.hidden_size;
    const std::string q = p + "ple/", input = p + "ple/embedding";
    PleWeights out;
    out.key         = b.parameter(q + "key", {width, c.ple_embed_dim}, {input});
    out.value       = b.parameter(q + "value", {c.hidden_size, c.ple_embed_dim}, {input});
    out.norm_key    = b.direct(q + "norm_key", {width});
    out.norm_query  = b.direct(q + "norm_query", {width});
    out.norm_conv   = b.direct(q + "norm_conv", {width});
    out.convolution = b.direct(q + "convolution", {width, c.ple_conv_kernel_size});
    return out;
}

// Every parameter a layer owns, which moves with it to its stage.
std::vector<WeightId> layer_weights(const LayerWeights& layer, bool with_experts) {
    std::vector<WeightId> out;
    for (const HyperConnectionWeights* hc : {&layer.attn_hc, &layer.mlp_hc}) {
        out.insert(out.end(), {hc->norm, hc->down, hc->up});
        if (hc->inject) { out.push_back(*hc->inject); }
    }
    if (const auto* gdn = std::get_if<GdnWeights>(&layer.mixer)) {
        out.insert(out.end(),
                   {gdn->query, gdn->key, gdn->value, gdn->z, gdn->a_projection, gdn->b_projection,
                    gdn->a_log, gdn->dt_bias, gdn->convolution, gdn->norm, gdn->output});
    } else {
        const auto& a = std::get<SparseAttentionWeights>(layer.mixer);
        out.insert(out.end(), {a.query, a.gate, a.key, a.value, a.output, a.query_norm, a.key_norm,
                               a.index_query, a.index_key, a.index_query_norm, a.index_key_norm});
    }
    const MoeWeights& m = layer.moe;
    out.insert(out.end(), {m.router, m.shared_score, m.shared_gate, m.shared_up, m.shared_down});
    if (with_experts) {
        out.insert(out.end(), m.gate.begin(), m.gate.end());
        out.insert(out.end(), m.up.begin(), m.up.end());
        out.insert(out.end(), m.down.begin(), m.down.end());
    }
    if (layer.ple) {
        const PleWeights& p = *layer.ple;
        out.insert(out.end(),
                   {p.key, p.value, p.norm_key, p.norm_query, p.norm_conv, p.convolution});
    }
    return out;
}

// Stored bytes behind `ids`, each object counted once.
std::uint64_t stored_bytes(const Bindings& b, std::span<const WeightId> ids,
                           std::set<std::size_t>& seen) {
    std::uint64_t bytes = 0;
    for (const WeightId id : ids) {
        for (const auto& part : b.at(id).reference.binding.parts) {
            if (seen.insert(part.object.index).second) {
                bytes += artifact::object_bytes(b.binder.reader().directory().object(part.object));
            }
        }
    }
    return bytes;
}

// Contiguous layer ranges whose stored bytes are as even as the layer granularity allows; the
// first stage also carries the embedding and the last the final mixer and the head.
StagePlan balanced_stages(std::span<const std::uint64_t> layer_bytes, std::uint64_t first_extra,
                          std::uint64_t last_extra, std::size_t ranks) {
    const auto layers = static_cast<std::uint32_t>(layer_bytes.size());
    std::vector<std::uint64_t> prefix(layers + 1, 0);
    for (std::uint32_t i = 0; i < layers; ++i) { prefix[i + 1] = prefix[i] + layer_bytes[i]; }
    const double total  = double(prefix[layers] + first_extra + last_extra);
    const double target = total / double(ranks);
    std::vector<std::uint32_t> boundaries;
    std::uint32_t begin = 0;
    for (std::size_t stage = 0; stage + 1 < ranks; ++stage) {
        // The end whose cumulative bytes come closest to (stage + 1) shares, leaving at least one
        // layer for each later stage.
        const std::uint32_t last_end = layers - static_cast<std::uint32_t>(ranks - stage - 1);
        std::uint32_t best           = begin + 1;
        double best_gap              = 1e300;
        for (std::uint32_t end = begin + 1; end <= last_end; ++end) {
            const double cumulative = double(prefix[end] + first_extra);
            const double gap        = std::abs(cumulative - target * double(stage + 1));
            if (gap < best_gap) {
                best_gap = gap;
                best     = end;
            }
        }
        boundaries.push_back(best);
        begin = best;
    }
    boundaries.push_back(layers);
    return StagePlan(layers, std::move(boundaries));
}

} // namespace

Model::Model(TextConfig config, LoadOptions options, TextWeights weights, StagePlan stages,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(std::move(options)),
      weights_(std::move(weights)), stages_(std::move(stages)), bound_(std::move(bound)),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    const auto& use = parameter.uses.front();
    if (use.hadamard_signs) {
        throw std::invalid_argument(parameter.name +
                                    ": Hadamard-rotated matrices are not supported");
    }
    ops::WeightInput result{parameter.view, use.policy, use.activation_input_divisor};
    if (use.input_columns) {
        result.input_columns = replicas_.on_rank(bound_, *use.input_columns, parameter.rank);
    }
    return result;
}

bool is_qwen4_exp(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "Qwen4ExpForCausalLM";
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_qwen4_exp(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not Qwen4ExpForCausalLM");
    }
    if (options.ranks == 0 || options.ranks != device.size()) {
        throw std::invalid_argument("Qwen3.8-Flash-Next stages must match the device ranks");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    artifact::Binder binder(reader);
    std::optional<qwen3_5::VisionConfig> vision_config;
    if (options.vision) {
        const auto& components = reader.directory().components;
        if (!components.contains("vision")) {
            throw std::invalid_argument("Vision needs an artifact converted with its tower "
                                        "(--components text,vision)");
        }
        vision_config = qwen3_5::parse_vision_config(reader.directory());
    }
    FrontendResources resources = bind_resources(binder, config, options.vision);
    Bindings b(binder);

    TextWeights weights;
    weights.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    weights.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                      {"text/final_hidden"});
    weights.final_mixer = bind_hc(b, config, "text/final_mixer/", false);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        LayerWeights layer{.attn_hc = bind_hc(b, config, p + "attn_hc/", true),
                           .mlp_hc  = bind_hc(b, config, p + "mlp_hc/", true),
                           .mixer   = GdnWeights{},
                           .moe     = {},
                           .ple     = std::nullopt};
        if (config.layer_types[i] == MixerKind::GatedDeltaNet) {
            layer.mixer = bind_gdn(b, config, p);
        } else {
            layer.mixer = bind_attention(b, config, p);
        }
        layer.moe = bind_moe(b, config, p, options.experts);
        if (std::find(config.ple_layers.begin(), config.ple_layers.end(), i) !=
            config.ple_layers.end()) {
            layer.ple = bind_ple(b, config, p);
        }
        weights.layers.push_back(std::move(layer));
    }
    // The tower stays on rank 0, with the token embedding its output joins.
    std::optional<qwen3_5::VisionWeights> vision_weights;
    if (vision_config) {
        vision_weights = qwen3_5::loading::bind_vision(b, *vision_config, config.hidden_size,
                                                       artifact::Residency::Device);
    }

    // The stage split: what was asked, or the even split of the bytes each device would hold.
    const bool device_experts = options.experts == ExpertResidency::Device;
    StagePlan stages(config.num_hidden_layers);
    if (options.ranks > 1) {
        if (!options.stage_layers.empty()) {
            if (options.stage_layers.size() != options.ranks) {
                throw std::invalid_argument("--stage-layers must list one count per device");
            }
            stages = StagePlan::from_layer_counts(config.num_hidden_layers, options.stage_layers);
        } else {
            std::set<std::size_t> seen;
            std::vector<std::uint64_t> layer_bytes;
            for (const auto& layer : weights.layers) {
                const auto ids = layer_weights(layer, device_experts);
                layer_bytes.push_back(stored_bytes(b, ids, seen));
            }
            const std::vector<WeightId> first{weights.token_embedding};
            const std::vector<WeightId> last{weights.output_head, weights.final_mixer.norm,
                                             weights.final_mixer.down, weights.final_mixer.up};
            stages = balanced_stages(layer_bytes, stored_bytes(b, first, seen),
                                     stored_bytes(b, last, seen), options.ranks);
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const std::size_t rank = stages.placement(i).stage;
            for (const WeightId id : layer_weights(weights.layers[i], device_experts)) {
                b.place(id, rank);
            }
        }
        b.place(weights.token_embedding, 0);
        for (const WeightId id : {weights.output_head, weights.final_mixer.norm,
                                  weights.final_mixer.down, weights.final_mixer.up}) {
            b.place(id, stages.stages() - 1);
        }
    } else if (!options.stage_layers.empty()) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }

    InstanceInfo info;
    info.name = reader.directory().metadata.value("name", std::string("Qwen3.8-Flash-Next"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    std::vector<std::filesystem::path> files;
    for (const auto& record : reader.directory().files) {
        files.push_back(record.path ? options.artifact.parent_path() / *record.path
                                    : options.artifact);
    }
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    auto model           = std::unique_ptr<Model>(
        new Model(std::move(config), options, std::move(weights), std::move(stages),
                  std::move(bound), std::move(resources), std::move(info), std::move(backing)));
    model->files_          = std::move(files);
    model->replicas_       = qwen3_5::AuxiliaryReplicas(model->bound_, device);
    model->vision_config_  = std::move(vision_config);
    model->vision_weights_ = std::move(vision_weights);
    return model;
}

} // namespace ninfer::models::qwen4_exp
