#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "core/dtype.h"
#include "core/evictable_weight_pool.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/stage_plan.h"
#include "models/dspine/config.h"
#include "models/dspine/model.h"
#include "models/draft_reader.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/vision_cpu_cache.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    std::filesystem::path artifact_path;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    // --draft-model on DFlash: the drafter's own artifact, its parameters and its plan. The drafter
    // materializes as a second backing beside the target's; a DSpine drafter keeps its own Model
    // instead (models/dspine), since it runs its own pipeline.
    std::shared_ptr<artifact::Reader> draft_reader;
    std::vector<loading::PendingWeight> draft_pending;
    artifact::MaterializationPlan draft_materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::shared_ptr<CpuVisionCache> cpu_vision;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

std::vector<ops::DeviceTier> rank_tiers(const DeviceContext& device) {
    std::vector<ops::DeviceTier> tiers;
    tiers.reserve(device.size());
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        tiers.push_back(ops::device_tier(device.rank(rank).device));
    }
    return tiers;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options,
                   std::filesystem::path artifact_path, std::vector<ops::DeviceTier> rank_tiers) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->artifact_path = std::move(artifact_path);
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder, options);
    const auto& text = out->config.text;
    if (options.expert_residency == ExpertResidency::Disk && out->artifact_path.empty()) {
        throw std::invalid_argument("disk expert residency requires the artifact entry path");
    }
    if (options.expert_residency != ExpertResidency::Device &&
        !std::holds_alternative<MoeConfig>(text.ffn)) {
        throw std::invalid_argument("--expert-residency host/disk requires a MoE model");
    }
    if (options.ranks > 1 && options.overlay_vision()) {
        // Overlay borrows weight memory from the primary device's evictable tail; a later stage
        // holds whole layers and nothing a Vision window could take. The two residency schemes are
        // answers to the same question -- where the bytes for something else come from -- and no
        // sound combination of them exists today, so say so rather than half-apply one.
        throw std::invalid_argument(
            "--vision-residency overlay and a multi-device --devices split cannot be combined");
    }
    out->weights.text =
        loading::bind_text(bindings, text, options,
                           loading::plan_stage_plan(text.num_hidden_layers, options));
    if (options.overlay_vision() && !out->config.vision) {
        throw std::invalid_argument("--vision-residency overlay requires a Vision artifact");
    }
    if (options.cpu_vision() && !out->config.vision) {
        throw std::invalid_argument("--vision-residency cpu requires a Vision artifact");
    }
    if (options.cpu_vision()) {
        if (out->artifact_path.empty()) {
            throw std::invalid_argument("CPU Vision lazy loading requires the artifact entry path");
        }
        loading::validate_cpu_vision(binder, *out->config.vision, text);
        out->cpu_vision = std::make_shared<CpuVisionCache>(
            [path = std::filesystem::absolute(out->artifact_path), id = reader.artifact_id(),
             vision = *out->config.vision, target = text] {
                artifact::Reader source(path);
                if (source.artifact_id() != id) {
                    throw artifact::ArtifactError("CPU Vision artifact changed since model loading");
                }
                artifact::Binder source_bindings(source);
                return loading::load_cpu_vision(source_bindings, vision, target);
            }, std::chrono::seconds(options.cpu_vision_idle_seconds));
    } else if (out->config.vision) {
        out->weights.vision = loading::bind_vision(
            bindings, *out->config.vision, text.hidden_size,
            options.overlay_vision() ? artifact::Residency::Pinned : artifact::Residency::Device);
    }
    std::pair<std::size_t, std::size_t> mtp_parameters{bindings.weights.size(),
                                                       bindings.weights.size()};
    if (out->config.mtp) {
        out->weights.mtp      = loading::bind_mtp(bindings, text, out->weights.text);
        mtp_parameters.second = bindings.weights.size();
    }
    std::pair<std::size_t, std::size_t> draft_parameters{bindings.weights.size(),
                                                         bindings.weights.size()};
    // A DSpine drafter is its own artifact, which materialize_model loads beside this one, and a
    // DFlash drafter given as --draft-model binds further down, once every target-side pass has
    // run: none of them may read a handle numbered by another artifact.
    if (out->config.draft && !out->config.draft->dspine && options.draft_model.empty()) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
        draft_parameters.second = bindings.weights.size();
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    loading::apply_storage_trades(bindings, out->config, out->weights, options);
    if (rank_tiers.empty()) { rank_tiers.assign(options.ranks, ops::current_device_tier()); }
    loading::apply_rank_policies(bindings, rank_tiers);
    loading::apply_device_layouts(bindings, out->weights, rank_tiers);
    if (options.overlay_vision()) {
        loading::apply_vision_overlay_placement(bindings, out->weights, mtp_parameters,
                                                draft_parameters);
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    // --draft-model on DFlash: the drafter's weights come from its own artifact, so they bind
    // through a second binder and materialize as a second backing. Its embedding and head are the
    // target's, which is why bind_draft takes the text weights either way.
    if (out->config.draft && !out->config.draft->dspine && !options.draft_model.empty()) {
        const std::string component(options.speculative_component());
        out->draft_reader = open_drafter(options.draft_model);
        artifact::Binder draft_binder(*out->draft_reader);
        loading::Bindings draft_bindings(draft_binder, options);
        out->weights.draft = loading::bind_draft(draft_bindings, *out->config.draft, text,
                                                 out->weights.text, component);
        // The drafter sits on rank 0, the head's device, where the round runs it.
        if (rank_tiers.empty()) {
            rank_tiers.assign(options.ranks, ops::current_device_tier());
        }
        const std::array tiers{rank_tiers.front()};
        loading::apply_rank_policies(draft_bindings, tiers);
        if (out->weights.proposal) { out->weights.draft->output_head = out->weights.proposal->head; }
        // A target that never carried a drafter declares no use for the drafter's hidden state, so
        // the head reads it through the text's own use instead.
        const std::string head_input = component + "/final_hidden";
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         bindings.has_use(out->weights.draft->output_head, head_input)
                             ? head_input
                             : std::string("text/final_hidden"));
        out->draft_pending         = std::move(draft_bindings.weights);
        out->draft_materialization = std::move(draft_binder).finish();
    }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish(
        options.overlay_vision() ? EvictableWeightPool::kChunkBytes : 1);
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

namespace {

// What one stage's device needs whatever layers it owns, beyond weights: the CUDA context, the
// workspace and graph memory of a forward pass over a full chunk. An estimate, since the Program
// plans these later; it only decides how the layers are dealt out, and `--stage-layers` overrides it.
constexpr std::uint64_t kStageFixedBytes = 1536ULL << 20;
// What rank 0 needs on top: the round buffers, logits and sampling state that stay with the head.
constexpr std::uint64_t kHeadStageFixedBytes = 1024ULL << 20;

std::uint64_t parameter_bytes(const artifact::Reader& reader, const loading::Bindings& bindings,
                              WeightId id) {
    if (bindings.at(id).reference.residency != artifact::Residency::Device) { return 0; }
    std::uint64_t total = 0;
    for (const auto& part : bindings.at(id).reference.binding.parts) {
        total += reader.geometry(part.object).bytes;
    }
    return total;
}

// An external drafter's attributes number their weights from zero, inside its own artifact; in the
// model's table they follow the target's, so each moves by one offset. Its token embedding and
// output head are the target's and stay put, as does the head's Use. An unset handle keeps its
// sentinel instead of wrapping.
void shift_draft_ids(DraftWeights& draft, std::size_t offset) {
    const auto move = [offset](WeightId& id) {
        if (id.index != WeightId{}.index) { id.index += offset; }
    };
    move(draft.feature_projection);
    move(draft.context_norm);
    move(draft.final_norm);
    for (auto& layer : draft.layers) {
        move(layer.input_norm);
        move(layer.post_attention_norm);
        move(layer.attention.query);
        move(layer.attention.key);
        move(layer.attention.value);
        move(layer.attention.context_key);
        move(layer.attention.context_value);
        move(layer.attention.query_norm);
        move(layer.attention.key_norm);
        move(layer.attention.output);
        move(layer.mlp.gate);
        move(layer.mlp.up);
        move(layer.mlp.down);
        if (layer.attention_conv) {
            move(layer.attention_conv->base_kernel);
            move(layer.attention_conv->kernel_projection);
        }
        if (layer.mlp_conv) {
            move(layer.mlp_conv->base_kernel);
            move(layer.mlp_conv->kernel_projection);
        }
    }
    if (draft.selector) {
        move(draft.selector->hidden_projection);
        move(draft.selector->predecessor_codebook);
        move(draft.selector->successor_codebook);
    }
    if (draft.dspine_bridge) {
        auto& bridge = *draft.dspine_bridge;
        move(bridge.codebook);
        move(bridge.early_read_weights);
        move(bridge.feature_write_weights);
        move(bridge.gate_weight);
        move(bridge.gate_bias);
        move(bridge.out_weight);
        move(bridge.prev_weight);
        move(bridge.predecessor_gate_weight);
        move(bridge.read_weight);
    }
}

// The Uses an external drafter's parameters carry may name auxiliary weights of its own artifact
// (a Hadamard sign vector, an input gather), and those handles move by the same offset.
void shift_draft_uses(std::vector<BoundWeight>& draft, std::size_t offset) {
    const auto move = [offset](std::optional<WeightId>& id) {
        if (id && id->index != WeightId{}.index) { id->index += offset; }
    };
    for (auto& weight : draft) {
        for (auto& use : weight.uses) {
            move(use.hadamard_signs);
            move(use.input_columns);
        }
    }
}

} // namespace

std::vector<std::uint32_t> default_stage_layers(const artifact::Reader& reader,
                                                LoadOptions options, const StageSizing& sizing,
                                                std::span<const std::uint64_t> free_bytes) {
    if (free_bytes.size() < 2) {
        throw std::invalid_argument("default stage layers need at least two devices");
    }
    // Bind the text model on one stage purely to size its layers.
    options.ranks = 1;
    options.stage_layers.clear();
    const Config config = parse_config(reader.directory(), options);
    const auto& text    = config.text;
    artifact::Binder binder(reader);
    loading::Bindings bindings(binder, options);
    const TextWeights weights =
        loading::bind_text(bindings, text, options, StagePlan(text.num_hidden_layers));

    std::vector<LayerCost> layers;
    layers.reserve(weights.layers.size());
    std::uint64_t layer_bytes_total = 0;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        LayerCost cost;
        for (const WeightId id : loading::layer_weights(weights.layers[layer])) {
            cost.weight_bytes += parameter_bytes(reader, bindings, id);
        }
        if (text.layer_types[layer] == MixerKind::FullAttention && text.attention) {
            // One page group of this layer: every plane of a page, for every KV head, as stored.
            const PagedKVStorageLayout storage = paged_kv_storage_layout(
                sizing.kv_storage, static_cast<std::int32_t>(text.attention->head_dim));
            cost.kv_bytes_per_page_group = static_cast<std::uint64_t>(kPagedKVPageSize) *
                                           text.attention->num_key_value_heads *
                                           storage.physical_bytes_per_token_head();
        } else if (text.gdn) {
            // The convolution window and the recurrent matrix of every state slot.
            const auto& gdn = *text.gdn;
            const std::uint64_t conv_bytes =
                gdn.conv_channels() * (gdn.linear_conv_kernel_dim - 1) * dtype_size(DType::BF16);
            const std::uint64_t recurrent_bytes =
                static_cast<std::uint64_t>(gdn.linear_key_head_dim) * gdn.linear_value_head_dim *
                gdn.linear_num_value_heads *
                dtype_size(options.gdn_state_fp16 ? DType::FP16 : DType::FP32);
            cost.state_bytes = sizing.state_slots * (conv_bytes + recurrent_bytes);
        }
        layer_bytes_total += cost.weight_bytes;
        layers.push_back(cost);
    }

    // The head stage keeps the embedding, the head and the final norm; a resident Vision tower, MTP's
    // layer, or a DFlash drafter's layers and feature projection, sit with them.
    std::uint64_t head_bytes = parameter_bytes(reader, bindings, weights.token_embedding) +
                               parameter_bytes(reader, bindings, weights.output_head) +
                               parameter_bytes(reader, bindings, weights.final_norm);
    if (config.mtp && options.speculative == SpeculativeBackend::Mtp) {
        head_bytes += layer_bytes_total / std::max<std::size_t>(layers.size(), 1);
    }
    if (config.vision && options.vision && !options.host_staged_vision()) {
        // A resident Vision tower runs where the embedding is.
        const std::size_t first = bindings.weights.size();
        (void)loading::bind_vision(bindings, *config.vision, text.hidden_size,
                                   artifact::Residency::Device);
        for (std::size_t index = first; index < bindings.weights.size(); ++index) {
            head_bytes += parameter_bytes(reader, bindings, WeightId{index});
        }
    }
    if (config.draft && config.draft->dspine) {
        // A DSpine drafter's own artifact and the table it derives from the head.
        head_bytes += dspine::weight_bytes(options.draft_model) +
                      dspine::head_write_bytes(*config.draft->dspine);
    } else if (config.draft && options.masked_draft()) {
        // The drafter shares the target's embedding and head, so only what it binds anew counts.
        const std::string component(options.speculative_component());
        if (options.draft_model.empty()) {
            const std::size_t first = bindings.weights.size();
            (void)loading::bind_draft(bindings, *config.draft, text, weights, component);
            for (std::size_t index = first; index < bindings.weights.size(); ++index) {
                head_bytes += parameter_bytes(reader, bindings, WeightId{index});
            }
        } else {
            // An external drafter's own objects, sized against the container that stores them.
            const auto drafter_reader = open_drafter(options.draft_model);
            artifact::Binder drafter_binder(*drafter_reader);
            loading::Bindings drafter_bindings(drafter_binder, options);
            const std::size_t first = drafter_bindings.weights.size();
            (void)loading::bind_draft(drafter_bindings, *config.draft, text, weights, component);
            for (std::size_t index = first; index < drafter_bindings.weights.size(); ++index) {
                head_bytes += parameter_bytes(*drafter_reader, drafter_bindings, WeightId{index});
            }
        }
    }
    // A resident Vision tower sits on the primary device too. An overlay tower is pinned on the
    // host and borrows the head's evictable tail, which `head_bytes` already counts.
    if (config.vision && options.vision && !options.host_staged_vision()) {
        const std::size_t first = bindings.weights.size();
        (void)loading::bind_vision(bindings, *config.vision, text.hidden_size, artifact::Residency::Device);
        for (std::size_t index = first; index < bindings.weights.size(); ++index) {
            head_bytes += parameter_bytes(reader, bindings, WeightId{index});
        }
    }

    std::vector<StageBudget> budgets;
    budgets.reserve(free_bytes.size());
    for (std::size_t stage = 0; stage < free_bytes.size(); ++stage) {
        budgets.push_back({.available_bytes = free_bytes[stage],
                           .fixed_bytes     = kStageFixedBytes +
                                          (stage == 0 ? kHeadStageFixedBytes + head_bytes : 0)});
    }
    // The estimates above are rough. When they say nothing fits, deal the layers out evenly and let
    // the Program's exact planning accept the split or report which device runs out.
    const StagePlan plan = [&] {
        try {
            return solve_stage_plan(layers, budgets).plan;
        } catch (const std::runtime_error&) {
            return StagePlan::even(text.num_hidden_layers, free_bytes.size());
        }
    }();
    std::vector<std::uint32_t> counts;
    counts.reserve(plan.stages());
    for (std::size_t stage = 0; stage < plan.stages(); ++stage) {
        counts.push_back(plan.stage_layers(stage));
    }
    return counts;
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer,
                                         const artifact::MaterializeOptions& materialize) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data = std::move(plan.impl_);
    std::unique_ptr<EvictableWeightPool> pool;
    if (data->options.overlay_vision()) {
        // The tail is sized against the encode window once execution planning knows it; the
        // pool's mirror is captured then (see Model::weight_pool).
        const auto& materialization = data->materialization;
        if (materialization.evictable_tail_bytes == 0 || materialization.pinned_objects.empty()) {
            throw std::logic_error("overlay Vision load plan has no evictable tail or pinned tower");
        }
        if (!EvictableWeightPool::supported(device)) {
            throw std::invalid_argument(
                "--vision-residency overlay requires CUDA virtual memory management support");
        }
        pool = std::make_unique<EvictableWeightPool>(
            device, EvictableWeightPool::Config{
                        .arena_bytes =
                            static_cast<std::size_t>(materialization.device_capacity(0)),
                        .evictable_tail_bytes =
                            static_cast<std::size_t>(materialization.evictable_tail_bytes),
                    });
    }
    const auto& reader = *data->materialization.source;
    auto backing = artifact::materialize(reader,
                                         std::move(data->materialization), device, observer,
                                         std::move(pool), materialize);
    auto experts = loading::expert_sources(reader, data->pending, data->weights, backing,
                                            data->options, data->artifact_path);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    // --draft-model on DFlash: the drafter's own arena, materialized beside the target's. Its
    // weights land after the target's in the model's weight table, so every handle the drafter
    // already holds moves by that offset once; its embedding and head keep the target's.
    std::optional<artifact::MaterializedArtifact> draft_backing;
    if (data->draft_reader) {
        draft_backing.emplace(artifact::materialize(
            *data->draft_reader, std::move(data->draft_materialization), device, observer,
            nullptr,
            artifact::MaterializeOptions{
                .suspendable     = materialize.suspendable,
                .retained_reader = materialize.suspendable ? data->draft_reader : nullptr}));
        auto draft               = loading::resolve_weights(std::move(data->draft_pending),
                                                            *draft_backing);
        const std::size_t offset = bound.size();
        shift_draft_ids(*data->weights.draft, offset);
        shift_draft_uses(draft, offset);
        for (auto& weight : draft) { bound.push_back(std::move(weight)); }
    }
    std::optional<VisionOverlayLayout> vision_overlay;
    if (data->options.overlay_vision()) {
        vision_overlay =
            loading::vision_overlay_layout(*data->weights.vision, bound, backing.pinned_block());
    }
    AuxiliaryReplicas replicas(bound, device);
    const bool dspine     = data->options.dspine();
    const auto draft_path = data->options.draft_model;
    auto model            = std::unique_ptr<Model>(
        new Model(std::move(data->config), data->options, std::move(data->weights),
                  std::move(bound), std::move(replicas), std::move(data->resources),
                  std::move(data->info), std::move(backing), std::move(draft_backing),
                  std::move(vision_overlay), std::move(data->cpu_vision), std::move(experts)));
    if (dspine) {
        // On the head's device, where the drafts' embedding, head and round state are.
        model->dspine_ = dspine::load_model(
            draft_path, device, observer,
            artifact::MaterializeOptions{.suspendable = materialize.suspendable});
    }
    return model;
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options, path, rank_tiers(device)), device,
                             observer);
}

} // namespace ninfer::models::qwen3_5
