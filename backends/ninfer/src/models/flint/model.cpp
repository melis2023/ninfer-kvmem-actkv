#include "models/flint/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::flint {
namespace {

using qwen3_5::loading::Bindings;

// Zero bytes after a matrix whose rows are not whole 256-value steps: the integer matrix kernel
// reads past the last row's end (gguf_project, moe_matrix_product).
constexpr std::uint64_t kRowTail = 512;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.vocab_size || !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    return out;
}

void tail(Bindings& b, WeightId id) {
    for (const auto& part : b.at(id).reference.binding.parts) {
        if (is_gguf(b.binder.reader().geometry(part.object).format)) {
            b.binder.device_tail(part.object, kRowTail);
        }
    }
}

// Resolve stored GGUF rows directly. Disk mode validates without materializing the bank.
// Host mode borrows the materializer's pinned parent. Neither path repacks model weights.
std::vector<ExpertBlob> expert_blobs(const artifact::Reader& reader,
                                    const artifact::ParameterReference& reference,
                                    const artifact::MaterializedArtifact& backing,
                                    std::uint64_t experts, std::uint64_t rows, std::uint64_t k,
                                    QType& format, std::uint64_t& row_bytes) {
    std::vector<ExpertBlob> result;
    const auto& parts = reference.binding.parts;
    if (parts.size() != 1) {
        throw artifact::ArtifactError(reference.name + ": expert bank must be one stored region");
    }
    const auto& part = parts.front();
    const auto& geometry = reader.geometry(part.object);
    if (!is_gguf(geometry.format) || geometry.shape.size() != 2 || geometry.shape[1] != k ||
        part.begin % k || part.end - part.begin != experts * rows * k) {
        throw artifact::ArtifactError(reference.name + ": experts must be stored GGUF block rows");
    }
    format = geometry.format;
    const auto block = gguf_block_shape(format);
    if (k % block.elements) {
        throw artifact::ArtifactError(reference.name + ": partial stored quantization block");
    }
    row_bytes = k / block.elements * block.bytes;
    reader.validate_object(part.object);
    const auto start = part.begin / k * row_bytes;
    for (std::uint64_t e = 0; e < experts; ++e) {
        ExpertBlob blob;
        blob.bytes = rows * row_bytes;
        blob.zero_tail = kRowTail;
        const auto offset = start + e * blob.bytes;
        if (reference.residency == artifact::Residency::Pinned) {
            const auto* parent = backing.pinned_parent(part.object).data;
            blob.pinned = parent + offset;
            blob.pinned_region = reinterpret_cast<std::uintptr_t>(parent);
        } else {
            const auto logical = reader.directory().tensor(part.object).offset + offset;
            for (const auto& segment : reader.segments(logical, blob.bytes)) {
                blob.extents.push_back({segment.file_index, segment.file_offset, segment.bytes});
            }
        }
        result.push_back(std::move(blob));
    }
    return result;
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    const auto& use = parameter.uses.front();
    return ops::WeightInput{parameter.view, use.policy, use.activation_input_divisor};
}

bool is_flint(const artifact::Reader& reader) {
    const auto& components = reader.directory().components;
    const auto text        = components.find("text");
    if (text == components.end()) { return false; }
    const auto& config = text->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0] == "FlintForCausalLM";
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_flint(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not FlintForCausalLM");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    artifact::Binder binder(reader);
    FrontendResources resources = bind_resources(binder, config);
    Bindings b(binder);
    const bool offload = options.experts != ExpertResidency::Device;
    std::vector<std::array<artifact::ParameterReference, 2>> expert_references;
    const std::uint64_t h = config.hidden_size;
    TextWeights weights;
    weights.token_embedding = b.parameter("text/token_embedding", {config.vocab_size, h});
    weights.output_head     = b.parameter("text/output_head", {config.vocab_size, h},
                                          {"text/final_hidden"});
    weights.final_norm      = b.direct("text/final_norm", {h}, QType::FP32);
    const std::uint64_t d  = config.head_dim;
    const std::uint64_t kv = config.num_key_value_heads;
    const std::uint64_t q  = std::uint64_t(config.num_attention_heads) * d;
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/";
        LayerWeights layer;
        layer.input_norm           = b.direct(p + "input_norm", {h}, QType::FP32);
        layer.ffn_norm             = b.direct(p + "ffn_norm", {h}, QType::FP32);
        layer.attention.query      = b.parameter(a + "query", {q, h}, {p + "mixer_input"});
        layer.attention.key        = b.parameter(a + "key", {kv * d, h}, {p + "mixer_input"});
        layer.attention.value      = b.parameter(a + "value", {kv * d, h}, {p + "mixer_input"});
        layer.attention.output     = b.parameter(a + "output", {h, q}, {p + "attention_output"});
        layer.attention.query_norm = b.direct(a + "query_norm", {d}, QType::FP32);
        layer.attention.key_norm   = b.direct(a + "key_norm", {d}, QType::FP32);
        layer.attention.sinks      = b.direct(a + "sinks", {config.num_attention_heads}, QType::FP32);
        tail(b, layer.attention.output);
        if (!config.moe(i)) {
            const std::uint64_t width = config.intermediate_size;
            layer.mlp_gate = b.parameter(p + "mlp/gate", {width, h}, {p + "mlp_input"});
            layer.mlp_up   = b.parameter(p + "mlp/up", {width, h}, {p + "mlp_input"});
            layer.mlp_down = b.parameter(p + "mlp/down", {h, width}, {p + "mlp_product"});
            tail(b, *layer.mlp_down);
            weights.layers.push_back(std::move(layer));
            continue;
        }
        const std::uint64_t e = config.num_experts, w = config.moe_intermediate_size;
        const std::uint64_t s = config.shared_intermediate_size;
        const std::string m   = p + "moe/";
        MoeWeights moe;
        moe.router      = b.direct(m + "router", {e, h}, QType::FP32);
        moe.shared_gate = b.parameter(m + "shared_gate", {s, h}, {m + "input"});
        moe.shared_up   = b.parameter(m + "shared_up", {s, h}, {m + "input"});
        moe.shared_down = b.parameter(m + "shared_down", {h, s}, {m + "shared_product"});
        tail(b, moe.shared_down);
        if (offload) {
            const auto residency = options.experts == ExpertResidency::Host
                                       ? artifact::Residency::Pinned : artifact::Residency::Values;
            expert_references.push_back({
                binder.parameter(m + "gate_up", {e * 2 * w, h}, residency),
                binder.parameter(m + "down", {e * h, w}, residency)});
            // Uses still belong to the family contract, even without a resident operand.
            for (const auto& use : {std::pair{m + "gate_up", m + "input"},
                                    std::pair{m + "down", m + "product"}}) {
                const auto& contract = binder.use(use.first, use.second);
                if (!contract.activation_policy || !contract.auxiliaries.empty()) {
                    throw artifact::ArtifactError(use.first +
                        ": streamed GGUF experts require an activation policy and no auxiliaries");
                }
            }
        } else {
            moe.gate_up = b.parameter(m + "gate_up", {e * 2 * w, h}, {m + "input"});
            moe.down    = b.parameter(m + "down", {e * h, w}, {m + "product"});
            tail(b, *moe.down);
            tail(b, *moe.gate_up);
        }
        layer.moe = moe;
        weights.layers.push_back(std::move(layer));
    }
    InstanceInfo info;
    info.name            = reader.directory().metadata.value("name", std::string("Flint"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer);
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    std::vector<ExpertBanks> expert_banks;
    for (const auto& reference : expert_references) {
        ExpertBanks bank;
        auto gu = expert_blobs(reader, reference[0], backing, config.num_experts,
                               2 * config.moe_intermediate_size, h,
                               bank.gate_up_type, bank.gate_up_row_bytes);
        auto down = expert_blobs(reader, reference[1], backing, config.num_experts,
                                 h, config.moe_intermediate_size,
                                 bank.down_type, bank.down_row_bytes);
        for (std::size_t e = 0; e < gu.size(); ++e) {
            bank.delivery.experts.push_back({std::move(gu[e]), std::move(down[e])});
        }
        // Gate rows, up rows, the [gate; up] parent, down rows: the executor's expert tables.
        bank.delivery.aliases = {{0, 0}, {0, config.moe_intermediate_size * bank.gate_up_row_bytes},
                                 {0, 0}, {1, 0}};
        expert_banks.push_back(std::move(bank));
    }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights),
                                                  std::move(bound), std::move(resources),
                                                  std::move(info), std::move(backing)));
    model->expert_residency_ = options.experts;
    model->expert_banks_     = std::move(expert_banks);
    if (options.experts == ExpertResidency::Disk) {
        for (const auto& file : reader.directory().files) {
            model->expert_files_.push_back(file.path ? options.artifact.parent_path() / *file.path
                                                     : options.artifact);
        }
    }
    return model;
}

} // namespace ninfer::models::flint
