#include "models/dspine/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/weight_view.h"
#include "models/qwen3_5/load/bindings.h"
#include "ops/common/device_tier.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::dspine {
namespace {

using qwen3_5::BoundWeight;
using qwen3_5::WeightId;
using qwen3_5::loading::Bindings;

struct LayerIds {
    WeightId input_norm, post_attention_norm, query_norm, key_norm;
    WeightId query, key, value, output, context_key, context_value, gate, up, down, read, write;
};

struct Ids {
    WeightId feature_projection, context_norm, final_norm, bridge_projection, gate_bias;
    WeightId head_write, sender_table;
    std::vector<LayerIds> layers;
};

Ids bind(Bindings& b, const Config& c) {
    const std::uint64_t h = c.hidden_size, q = c.query_width(), k = c.key_width();
    const std::uint64_t d = c.head_dim, inner = c.intermediate_size;
    const std::uint64_t code = c.code_size, rank = c.rank;
    Ids ids;
    ids.feature_projection =
        b.parameter("dspine/feature_projection", {h, std::uint64_t(c.taps()) * h},
                    {"dspine/target_features"});
    ids.context_norm = b.direct("dspine/context_norm", {h});
    ids.final_norm   = b.direct("dspine/final_norm", {h});
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string p = "dspine/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/";
        const std::string r = "dspine/bridge/layers/" + std::to_string(i) + "/";
        LayerIds layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.query      = b.parameter(a + "query", {q, h}, {p + "query_projection_input"});
        layer.key        = b.parameter(a + "key", {k, h}, {p + "query_projection_input"});
        layer.value      = b.parameter(a + "value", {k, h}, {p + "query_projection_input"});
        layer.context_key = b.parameter(a + "context_key", {k, h}, {"dspine/context_input"});
        layer.context_value =
            b.parameter(a + "context_value", {k, h}, {"dspine/context_input"});
        layer.query_norm = b.direct(a + "query_norm", {d});
        layer.key_norm   = b.direct(a + "key_norm", {d});
        layer.output     = b.parameter(a + "output", {h, q}, {p + "attention_output"});
        layer.gate       = b.parameter(p + "mlp/gate", {inner, h}, {p + "mlp_input"});
        layer.up         = b.parameter(p + "mlp/up", {inner, h}, {p + "mlp_input"});
        layer.down       = b.parameter(p + "mlp/down", {h, inner}, {p + "mlp_product"});
        layer.read       = b.parameter(r + "read", {code, h}, {r + "read_input"});
        layer.write      = b.parameter(r + "write", {h, rank}, {r + "message"});
        ids.layers.push_back(layer);
    }
    ids.bridge_projection =
        b.parameter("dspine/bridge/projection", {3 * rank, code}, {"dspine/bridge/code_input"});
    ids.gate_bias    = b.direct("dspine/bridge/gate_bias", {rank});
    ids.head_write   = b.direct("dspine/bridge/head_write", {rank, h});
    ids.sender_table = b.direct("dspine/bridge/sender_table", {c.vocab_size, 2 * rank});
    return ids;
}

// A direct parameter as a Tensor, the stored shape reversed (the last axis fastest).
Tensor tensor(const BoundWeight& bound) {
    const auto& shape = bound.view.shape;
    for (const auto extent : shape) {
        if (extent == 0 || extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(bound.name + ": direct parameter exceeds Tensor extent");
        }
    }
    const auto at = [&](std::size_t i) { return static_cast<std::int32_t>(shape[i]); };
    if (shape.size() == 1) { return weight_tensor(bound.view, {at(0)}); }
    if (shape.size() == 2) { return weight_tensor(bound.view, {at(1), at(0)}); }
    throw std::invalid_argument(bound.name + ": a DSpine direct parameter is a vector or matrix");
}

ops::WeightInput input(const BoundWeight& bound) {
    if (bound.uses.size() != 1 || bound.uses.front().hadamard_signs ||
        bound.uses.front().input_columns) {
        throw std::invalid_argument(bound.name + ": a DSpine projection needs one plain use");
    }
    const auto& use = bound.uses.front();
    return ops::WeightInput{bound.view, use.policy, use.activation_input_divisor};
}

ops::SingleProjectionWeight projection(const BoundWeight& bound) {
    return ops::prepare_linear_weight(input(bound));
}

} // namespace

Model::~Model() = default;

std::string Model::artifact_hex() const {
    std::string out;
    for (const std::byte b : reader_->artifact_id()) {
        constexpr char kDigits[] = "0123456789abcdef";
        out += kDigits[std::to_integer<unsigned>(b) >> 4U];
        out += kDigits[std::to_integer<unsigned>(b) & 15U];
    }
    return out;
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, DeviceContext& device,
                                  const StartupObserver* observer,
                                  const artifact::MaterializeOptions& materialize) {
    auto reader    = std::make_shared<const artifact::Reader>(path);
    auto model     = std::unique_ptr<Model>(new Model());
    model->config_ = read_config(*reader, path);
    artifact::Binder binder(*reader);
    Bindings b(binder);
    const Ids ids = bind(b, model->config_);
    // Every weight sits on rank 0, the target's head device: the A8/A4 permissions a stored
    // weight carries hold only where that device runs the 120a code.
    const std::array tiers{ops::device_tier(device.rank(0).device)};
    qwen3_5::loading::apply_rank_policies(b, tiers);
    auto pending    = std::move(b.weights);
    auto plan       = std::move(binder).finish();
    model->backing_ = artifact::materialize(
        *reader, std::move(plan), device, observer, nullptr,
        artifact::MaterializeOptions{.suspendable     = materialize.suspendable,
                                     .retained_reader = materialize.suspendable ? reader : nullptr});
    model->bound_      = qwen3_5::loading::resolve_weights(std::move(pending), model->backing_);
    const auto& bound  = model->bound_;
    const auto at      = [&](WeightId id) -> const BoundWeight& { return bound.at(id.index); };
    Operands& out      = model->operands_;
    out.feature_projection = projection(at(ids.feature_projection));
    out.context_norm       = tensor(at(ids.context_norm));
    out.final_norm         = tensor(at(ids.final_norm));
    out.bridge_projection  = projection(at(ids.bridge_projection));
    out.gate_bias          = tensor(at(ids.gate_bias));
    // Stored [R, H] (the write transposed): as a BF16 [H, R] activation, column r is write[:, r].
    out.head_write_columns = tensor(at(ids.head_write));
    out.sender_table       = tensor(at(ids.sender_table));
    for (const auto& layer : ids.layers) {
        LayerOperands l;
        l.input_norm          = tensor(at(layer.input_norm));
        l.post_attention_norm = tensor(at(layer.post_attention_norm));
        l.query_norm          = tensor(at(layer.query_norm));
        l.key_norm            = tensor(at(layer.key_norm));
        l.query               = projection(at(layer.query));
        l.key                 = projection(at(layer.key));
        l.value               = projection(at(layer.value));
        l.output              = projection(at(layer.output));
        l.context_key         = projection(at(layer.context_key));
        l.context_value       = projection(at(layer.context_value));
        l.gate                = projection(at(layer.gate));
        l.up                  = projection(at(layer.up));
        l.down                = projection(at(layer.down));
        // The converter stores each input's projections as rows of one parent, as it does a
        // DFlash drafter's: the fused routes then read q/k/v and gate/up in one pass. The drafter
        // takes them only for the formats and geometries they have (drafter.cpp); any other
        // drafter keeps the joined view and runs the separate projections.
        if (const std::array qkv{input(at(layer.query)), input(at(layer.key)),
                                 input(at(layer.value))};
            ops::joins(qkv)) {
            l.query_key_value = ops::prepare_linear_weight(qkv);
        }
        if (const std::array mlp{input(at(layer.gate)), input(at(layer.up))}; ops::joins(mlp)) {
            l.gate_up = ops::prepare_linear_swiglu_weight(mlp[0], mlp[1]);
        }
        l.read                = projection(at(layer.read));
        l.write               = projection(at(layer.write));
        out.layers.push_back(l);
    }
    for (const Tensor* t : {&out.context_norm, &out.final_norm, &out.gate_bias,
                            &out.head_write_columns, &out.sender_table}) {
        if (t->dtype != DType::BF16) {
            throw artifact::ArtifactError(path.string() +
                                          ": the DSpine norms, bias and tables must be BF16");
        }
    }
    model->name_         = reader->directory().metadata.value("name", std::string("DSpine"));
    model->path_         = path;
    // The weight arena on rank 0, where every drafter weight sits. device_backing_bytes() counts
    // only arenas a suspend can release, which a plain load does not reserve.
    model->device_bytes_ = model->backing_.stats().device_capacity_bytes;
    model->reader_       = std::move(reader);
    return model;
}

} // namespace ninfer::models::dspine
