#include "models/qwen3_5/load/bindings.h"

#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query      = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key        = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    out.gate       = b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

using Range = std::pair<std::uint64_t, std::uint64_t>;

// The `rank`-th of `ranks` equal parts of `count` units, `unit` wide each.
Range share(std::uint64_t count, std::uint64_t unit, std::size_t rank, std::size_t ranks) {
    const std::uint64_t each = count / ranks * unit;
    return {each * rank, each * (rank + 1)};
}

Range offset(Range range, std::uint64_t by) { return {range.first + by, range.second + by}; }

ParameterPiece rows(Range range) { return {ParameterPiece::Cut::Rows, {range}}; }

ParameterPiece columns(std::vector<Range> ranges) {
    return {ParameterPiece::Cut::Columns, std::move(ranges)};
}

// A copy of a whole parameter for a rank: rank 0 binds it as it is, another rank a piece that
// covers it.
WeightId bind_replica(Bindings& b, const std::string& name, const artifact::Shape& shape,
                      const std::vector<std::string>& inputs, std::optional<QType> format,
                      std::size_t rank) {
    if (rank == 0) { return b.parameter(name, shape, inputs, format); }
    const ParameterPiece whole = shape.size() == 2 ? rows({0, shape[0]}) : columns({{0, shape[0]}});
    return b.piece(name, shape, inputs, format, rank, whole);
}

// A rank's share of a GDN mixer: its value heads, the key heads they read, and their convolution
// channels; the output projection takes the matching input columns. The gated norm is per head,
// and the gating projection is small, so every rank holds all of both and keeps its heads' gates.
GdnWeights bind_gdn_share(Bindings& b, const TextConfig& text, const std::string& p,
                          std::size_t rank, std::size_t ranks) {
    const auto& g      = text.gdn.value();
    const auto h       = text.hidden_size;
    const auto k       = g.key_width();
    const auto v       = g.value_width();
    const auto heads   = g.linear_num_value_heads;
    const Range key    = share(g.linear_num_key_heads, g.linear_key_head_dim, rank, ranks);
    const Range value  = share(heads, g.linear_value_head_dim, rank, ranks);
    const auto input   = std::vector<std::string>{p + "mixer_input"};
    GdnWeights out;
    out.query        = b.piece(p + "gdn/query", {k, h}, input, {}, rank, rows(key));
    out.key          = b.piece(p + "gdn/key", {k, h}, input, {}, rank, rows(key));
    out.value        = b.piece(p + "gdn/value", {v, h}, input, {}, rank, rows(value));
    out.z            = b.piece(p + "gdn/z", {v, h}, input, {}, rank, rows(value));
    out.a_projection = bind_replica(b, p + "gdn/a_projection", {heads, h}, input, {}, rank);
    out.b_projection = bind_replica(b, p + "gdn/b_projection", {heads, h}, input, {}, rank);
    out.a_log        = bind_replica(b, p + "gdn/a_log", {heads}, {}, QType::FP32, rank);
    out.dt_bias      = bind_replica(b, p + "gdn/dt_bias", {heads}, {}, QType::FP32, rank);
    out.convolution  = b.piece(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()},
                               {}, QType::BF16, rank,
                               columns({key, offset(key, k), offset(value, 2 * k)}));
    out.norm   = bind_replica(b, p + "gdn/norm", {g.linear_value_head_dim}, {}, QType::BF16, rank);
    out.output = b.piece(p + "gdn/output", {h, v}, {p + "gdn/gated_output"}, {}, rank,
                         columns({value}));
    return out;
}

// A rank's share of a dense FFN: its intermediate rows of gate and up, and those input columns of
// down.
DenseWeights bind_dense_share(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                              const std::string& p, std::size_t rank, std::size_t ranks) {
    const Range part  = share(intermediate, 1, rank, ranks);
    const auto input  = std::vector<std::string>{p + "ffn_input"};
    return {b.piece(p + "mlp/gate", {intermediate, h}, input, {}, rank, rows(part)),
            b.piece(p + "mlp/up", {intermediate, h}, input, {}, rank, rows(part)),
            b.piece(p + "mlp/down", {h, intermediate}, {p + "mlp/product"}, {}, rank,
                    columns({part}))};
}

// A norm every rank reads.
WeightId bind_norm_replica(Bindings& b, const std::string& name, std::uint64_t width,
                           std::size_t rank) {
    return bind_replica(b, name, {width}, {}, QType::BF16, rank);
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    const auto residency = prefix.starts_with("mtp/") && b.resident_mtp_experts
                               ? artifact::Residency::Device
                           : b.expert_residency == ExpertResidency::Host
                               ? artifact::Residency::Pinned
                           : b.expert_residency == ExpertResidency::Disk
                               ? artifact::Residency::Values
                               : artifact::Residency::Device;
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        out.experts.push_back({b.parameter(ep + "gate", {ir, h}, {input}, {}, residency),
                               b.parameter(ep + "up", {ir, h}, {input}, {}, residency),
                               b.parameter(ep + "down", {h, ir}, {ep + "product"}, {}, residency)});
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

// A stage owns its layers whole: norms, the attention or GDN projections, and the FFN or experts all
// live on the stage's device, next to the KV cache and recurrent state that layer reads and writes.
// That is what lets the stage run its layers with the same kernels as a single GPU.
void place_layer(Bindings& b, const BlockWeights& block, std::size_t rank) {
    for (const WeightId id : layer_weights(block)) { b.place(id, rank); }
}

} // namespace

std::vector<WeightId> layer_weights(const BlockWeights& block) {
    std::vector<WeightId> out{block.input_norm, block.post_attention_norm};
    if (const auto* attention = std::get_if<AttentionWeights>(&block.mixer)) {
        out.insert(out.end(), {attention->query, attention->key, attention->gate, attention->value,
                               attention->query_norm, attention->key_norm, attention->output});
    } else {
        const auto& gdn = std::get<GdnWeights>(block.mixer);
        out.insert(out.end(), {gdn.query, gdn.key, gdn.value, gdn.z, gdn.a_projection,
                               gdn.b_projection, gdn.a_log, gdn.dt_bias, gdn.convolution, gdn.norm,
                               gdn.output});
    }
    const auto add_dense = [&](const DenseWeights& dense) {
        out.insert(out.end(), {dense.gate, dense.up, dense.down});
    };
    if (const auto* dense = std::get_if<DenseWeights>(&block.ffn)) {
        add_dense(*dense);
        return out;
    }
    const auto& moe = std::get<MoeWeights>(block.ffn);
    out.push_back(moe.router);
    out.push_back(moe.shared_score);
    for (const auto& expert : moe.experts) { add_dense(expert); }
    add_dense(moe.shared);
    return out;
}

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer) {
    BlockWeights out;
    out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
    out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p);
    }
    return out;
}

StagePlan plan_stage_plan(std::uint32_t layers, const LoadOptions& options) {
    if (options.tensor_parallel) {
        if (!options.stage_layers.empty()) {
            throw std::invalid_argument("--stage-layers and --tensor-parallel cannot be combined");
        }
        return StagePlan(layers);
    }
    if (options.ranks <= 1) {
        if (!options.stage_layers.empty()) {
            throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
        }
        return StagePlan(layers);
    }
    if (options.ranks > layers) {
        throw std::invalid_argument("--devices asks for more devices than the model has layers");
    }
    // The layer counts come from the caller when it has measured device memory (the engine solves
    // them against each card's free bytes), or from `--stage-layers`. Without either, equal counts:
    // the sensible default while layers are near-uniform in size.
    if (!options.stage_layers.empty()) {
        if (options.stage_layers.size() != options.ranks) {
            throw std::invalid_argument("--stage-layers lists " +
                                        std::to_string(options.stage_layers.size()) +
                                        " counts but --devices names " +
                                        std::to_string(options.ranks) + " devices");
        }
        return StagePlan::from_layer_counts(layers, options.stage_layers);
    }
    return StagePlan::even(layers, options.ranks);
}

namespace {

// Tensor parallelism: rank 0's share of a layer beside its whole attention, and every other
// rank's share.
BlockWeights bind_block_shares(Bindings& b, const TextConfig& config, const std::string& p,
                               MixerKind mixer, std::size_t ranks,
                               std::vector<std::vector<RankLayerWeights>>& rank_layers) {
    const auto* dense = std::get_if<DenseConfig>(&config.ffn);
    if (dense == nullptr) {
        throw std::invalid_argument("--tensor-parallel supports models with a dense FFN");
    }
    BlockWeights out;
    out.input_norm          = bind_norm_replica(b, p + "input_norm", config.hidden_size, 0);
    out.post_attention_norm = bind_norm_replica(b, p + "post_attention_norm", config.hidden_size, 0);
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn_share(b, config, p, 0, ranks);
    }
    out.ffn = bind_dense_share(b, config.hidden_size, dense->intermediate_size, p, 0, ranks);
    for (std::size_t rank = 1; rank < ranks; ++rank) {
        RankLayerWeights share;
        share.input_norm = bind_norm_replica(b, p + "input_norm", config.hidden_size, rank);
        share.post_attention_norm =
            bind_norm_replica(b, p + "post_attention_norm", config.hidden_size, rank);
        if (mixer != MixerKind::FullAttention) {
            share.gdn = bind_gdn_share(b, config, p, rank, ranks);
        }
        share.ffn = bind_dense_share(b, config.hidden_size, dense->intermediate_size, p, rank, ranks);
        rank_layers[rank - 1].push_back(std::move(share));
    }
    return out;
}

} // namespace

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options,
                      StagePlan stages) {
    TextWeights out;
    out.stages = std::move(stages);
    if (options.tensor_parallel) {
        // Divisibility of the heads and rows, before any piece is cut.
        (void)tensor_parallel_share(config, options.ranks);
        out.rank_layers.resize(options.ranks - 1);
    }
    out.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    out.final_norm  = b.direct("text/final_norm", {config.hidden_size});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "text/layers/" + std::to_string(i) + "/";
        if (options.tensor_parallel) {
            out.layers.push_back(bind_block_shares(b, config, prefix, config.layer_types[i],
                                                   options.ranks, out.rank_layers));
            continue;
        }
        out.layers.push_back(bind_block(b, config, prefix, config.layer_types[i]));
        // A layer lives on its stage. The embedding, final norm and head stay on rank 0 with the
        // round state, sampling and prefill buffers that reach them directly: the last stage sends
        // the residual back, one extra hop per forward pass.
        // One stage is the single-GPU route and binds exactly what it always bound.
        if (!out.stages.single_stage()) {
            place_layer(b, out.layers.back(), out.stages.placement(i).stage);
        }
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    out.head = b.parameter("proposal/head", {rows, target.hidden_size},
                           {std::string(options.speculative_component()) + "/final_hidden"});
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
