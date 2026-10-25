#include "models/qwen4_exp/executor.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/weight_view.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/expert_stream.h"
#include "models/qwen4_exp/ngram_table.h"
#include "models/qwen3_5/execution/vision.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ngram_rows.h"
#include "ninfer/ops/ple_inject.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_attention.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::int32_t kAlign = 256;
// Calls wider than this run their experts through the matrix kernel, from device memory.
constexpr std::int32_t kVectorTokens = 8;
// Zero bytes after every device slot's down matrix (see expert_stream.cpp and moe_experts_gguf).
constexpr std::uint64_t kSlotTail = 256;

std::uint64_t round_up(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// A logical projection's output rows, from one or more native operands: contiguous runs of one
// parent are fused into one operand, other parts write their rows separately.
struct Projection {
    struct Piece {
        ops::SingleProjectionWeight weight;
        std::int32_t row  = 0;
        std::int32_t rows = 0;
    };

    std::vector<Piece> pieces;
    std::int32_t rows = 0;
    std::int32_t k    = 0;
};

Projection make_projection(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    Projection out;
    std::size_t begin = 0;
    while (begin < inputs.size()) {
        std::size_t end = begin + 1;
        while (end < inputs.size() && ops::joins(std::span<const ops::WeightInput>(
                                          inputs.data() + begin, end + 1 - begin))) {
            ++end;
        }
        const std::span<const ops::WeightInput> run(inputs.data() + begin, end - begin);
        Projection::Piece piece{ops::prepare_linear_weight(run), out.rows, 0};
        piece.rows = piece.weight.weight.n;
        if (out.k == 0) { out.k = piece.weight.weight.k; }
        if (piece.weight.weight.k != out.k) {
            throw std::invalid_argument("projection parts read different input widths");
        }
        out.rows += piece.rows;
        out.pieces.push_back(std::move(piece));
        begin = end;
    }
    return out;
}

std::size_t projection_workspace(const Projection& p, std::int32_t tokens) {
    std::size_t bytes = 0;
    for (const auto& piece : p.pieces) {
        std::size_t need = ops::linear_workspace_capacity_bytes(
            piece.weight.weight.qtype, piece.rows, p.k, piece.weight.policy, 1, tokens);
        if (p.pieces.size() > 1) { need += round_up(std::uint64_t(piece.rows) * tokens * 2); }
        bytes = std::max(bytes, need);
    }
    return bytes;
}

// out (BF16 [p.rows, T]) <- the projection of x (BF16 [p.k, T]).
void project(const Projection& p, const Tensor& x, Tensor& out, WorkspaceArena& workspace,
             cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (p.pieces.size() == 1) {
        ops::linear(x, p.pieces.front().weight.weight, out, p.pieces.front().weight.policy,
                    workspace, stream);
        return;
    }
    for (const auto& piece : p.pieces) {
        auto scope  = workspace.scope();
        Tensor part = workspace.alloc(DType::BF16, {piece.rows, tokens});
        ops::linear(x, piece.weight.weight, part, piece.weight.policy, workspace, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(out.data) + std::size_t(piece.row) * 2,
                                     std::size_t(p.rows) * 2, part.data,
                                     std::size_t(piece.rows) * 2, std::size_t(piece.rows) * 2,
                                     tokens, cudaMemcpyDeviceToDevice, stream));
    }
}

Tensor direct(const Model& model, WeightId id, std::initializer_list<std::int32_t> shape,
              DType dtype = DType::BF16) {
    const auto& view = model.weight(id).view;
    Tensor tensor    = weight_tensor(view, shape);
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as " +
                                    (dtype == DType::BF16 ? "bf16" : "fp32"));
    }
    return tensor;
}

// The Vision tower's native operands, prepared as the Qwen3.5 Program prepares them.
qwen3_5::execution::VisionParameters vision_parameters_for(const Model& model,
                                                           const qwen3_5::VisionWeights& w) {
    const auto linear = [&](WeightId id) { return ops::prepare_linear_weight(model.input(id)); };
    const auto tensor = [&](WeightId id) {
        const auto& view = model.weight(id).view;
        if (view.shape.size() == 1) {
            return weight_tensor(view, {static_cast<std::int32_t>(view.shape[0])});
        }
        return weight_tensor(view, {static_cast<std::int32_t>(view.shape[1]),
                                    static_cast<std::int32_t>(view.shape[0])});
    };
    const auto norm = [&](const qwen3_5::NormWeights& n) {
        return qwen3_5::execution::NormParameters{tensor(n.weight), tensor(n.bias)};
    };
    // The query, key and value biases form one bank, as the fused QKV projection reads them.
    const auto joined = [&](std::array<WeightId, 3> ids) {
        WeightView view;
        std::uint64_t count = 0;
        for (const WeightId id : ids) {
            const auto& input = model.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    throw std::invalid_argument("Vision QKV biases must be one contiguous bank");
                }
                view.parts.push_back(part);
            }
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    };
    qwen3_5::execution::VisionParameters out;
    out.patch_embedding      = linear(w.patch_embedding);
    out.patch_embedding_bias = tensor(w.patch_embedding_bias);
    out.position_embedding   = tensor(w.position_embedding);
    for (const auto& layer : w.layers) {
        const std::array qkv{model.input(layer.query), model.input(layer.key),
                             model.input(layer.value)};
        out.layers.push_back({norm(layer.norm1), norm(layer.norm2),
                              ops::prepare_linear_weight(qkv),
                              joined({layer.query_bias, layer.key_bias, layer.value_bias}),
                              linear(layer.output), linear(layer.fc1), linear(layer.fc2),
                              tensor(layer.output_bias), tensor(layer.fc1_bias),
                              tensor(layer.fc2_bias)});
    }
    out.merger_norm     = norm(w.merger_norm);
    out.merger_fc1      = linear(w.merger_fc1);
    out.merger_fc2      = linear(w.merger_fc2);
    out.merger_fc1_bias = tensor(w.merger_fc1_bias);
    out.merger_fc2_bias = tensor(w.merger_fc2_bias);
    return out;
}

struct HcPlan {
    Tensor norm, down, up, inject;
    bool has_inject = false;

    [[nodiscard]] ops::HyperConnectionWeights weights() const {
        return {&norm, &down, &up, has_inject ? &inject : nullptr};
    }
};

HcPlan make_hc(const Model& model, const HyperConnectionWeights& w, const TextConfig& c) {
    const auto width = static_cast<std::int32_t>(c.hc_count * c.hidden_size);
    const auto low   = static_cast<std::int32_t>(c.hc_lowrank);
    HcPlan out;
    out.norm = direct(model, w.norm, {width});
    out.down = direct(model, w.down, {width, low});
    out.up   = direct(model, w.up, {low, width});
    if (w.inject) {
        out.inject     = direct(model, *w.inject, {width, static_cast<std::int32_t>(c.hc_count)});
        out.has_inject = true;
    }
    return out;
}

struct GdnPlan {
    Projection qkv, z, a, b, output;
    Tensor convolution, a_log, dt_bias, norm;
};

struct QsaPlan {
    Projection query, gate, key, value, index, output;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
};

struct PlePlan {
    Projection key, value;
    Tensor norm_key, norm_query, norm_conv, convolution;
};

// One projection of every expert of a layer, as a device table of expert base pointers on the
// layer's device.
struct ExpertTable {
    QType format           = QType::GGUF_Q8_0;
    std::int64_t row_bytes = 0;
    std::int32_t rows      = 0;
    std::vector<const void*> pointers; // each expert's rows where the artifact put them
    DeviceBuffer table;

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes};
    }
};

ExpertTable make_table(const Model& model, std::span<const WeightId> ids) {
    ExpertTable out;
    std::vector<const void*> pointers;
    for (const WeightId id : ids) {
        const Weight w = native_weight(model.weight(id).view);
        if (!is_gguf(w.qtype)) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": the expert kernels take GGUF block banks");
        }
        const auto block       = gguf_block_shape(w.qtype);
        const std::int64_t row = std::int64_t(w.k / block.elements) * block.bytes;
        if (pointers.empty()) {
            out.format    = w.qtype;
            out.row_bytes = row;
        } else if (w.qtype != out.format || row != out.row_bytes) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": every expert of a bank must share its block type");
        }
        pointers.push_back(w.qdata);
        out.rows = w.n;
    }
    out.table = DeviceBuffer(pointers.size() * sizeof(void*));
    out.table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    out.pointers = std::move(pointers);
    return out;
}

// A layer's table for experts that stay in the artifact's files: every entry null until the
// expert stream makes the expert resident.
ExpertTable make_located_table(std::span<const ExpertLocation> locations) {
    ExpertTable out;
    for (const auto& location : locations) {
        if (out.row_bytes == 0) {
            out.format    = location.format;
            out.row_bytes = location.row_bytes;
            out.rows      = location.rows;
        } else if (location.format != out.format || location.row_bytes != out.row_bytes ||
                   location.rows != out.rows) {
            throw std::invalid_argument("every expert of a bank must share its block type");
        }
    }
    out.pointers.assign(locations.size(), nullptr);
    out.table = DeviceBuffer(out.pointers.size() * sizeof(void*));
    out.table.copy_from_host(out.pointers.data(), out.pointers.size() * sizeof(void*));
    return out;
}

struct MoePlan {
    Tensor router, shared_gate;
    ExpertTable gate, up, down, shared_gate_table, shared_up_table, shared_down_table;
    // Device banks or disk experts' device slots, with zeros after the down banks; host experts
    // are read across the bus.
    bool device_resident = false;

    [[nodiscard]] ops::GgufMoeWeights banks() const {
        return {gate.view(),
                up.view(),
                down.view(),
                shared_gate_table.view(),
                shared_up_table.view(),
                shared_down_table.view(),
                static_cast<std::int32_t>(gate.pointers.size()),
                device_resident};
    }
};

struct LayerPlan {
    std::size_t rank = 0;
    HcPlan attn_hc, mlp_hc;
    std::optional<GdnPlan> gdn;
    std::optional<QsaPlan> qsa;
    std::optional<PlePlan> ple;
    MoePlan moe;
};

// One sequence's mutable state of one layer, on that layer's device.
struct LayerState {
    DeviceBuffer ssm, conv;                      // Gated DeltaNet
    DeviceBuffer k_pages, v_pages, pooled, tail; // sparse attention
    DeviceBuffer k_scales, v_scales;             // a quantized KV's scale planes
    DeviceBuffer history;                        // PLE
};

struct SequenceState {
    std::uint32_t position = 0;
    NgramContext context;
    std::vector<LayerState> layers;
    // A prompt's media (Vision): for each prompt position the column of its merged embedding, or
    // -1, and the prompt's RoPE positions, axis-major [3, tokens]; positions past them rotate at
    // their index plus rope_delta.
    std::vector<std::int32_t> media_columns, media_rope;
    std::int32_t rope_delta = 0;
    // One decode graph per segment, captured at the sequence's second decode step: the first runs
    // eagerly, so every lazy initialization of the ops it reaches happens outside capture. The
    // graphs read the token, its position and n-gram rows from the ranks' staged planes and stay
    // valid for the sequence's lifetime, across resets.
    std::vector<DecodeGraphExecutable> decode;
    std::uint32_t decode_steps = 0;
};

// Consecutive layers on one rank, in pass order. The first segment also embeds the tokens, the
// segment on the head rank after the last layer also runs the final mixer and the head.
struct Segment {
    std::size_t rank  = 0;
    std::size_t begin = 0, end = 0; // layers
    bool embed = false, head = false;
};

// One sequence's tokens within a pass: columns [column, column + count) of the activation planes.
// A prefill pass has one part; a batched decode pass has one single-token part per sequence. The
// per-token work (embedding, hyper-connections, MoE, head) runs over every column at once; the
// mixers and the PLE injection, which read and write a sequence's state, run part by part.
struct Part {
    SequenceState* sequence = nullptr;
    std::int32_t column     = 0;
    std::int32_t count      = 0;
};

// What one device holds for the layers it runs: activation planes sized for a full chunk and a
// scoped workspace.
struct RankState {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    DeviceBuffer activations;
    std::unique_ptr<DeviceArena> workspace;
    DeviceBuffer block_table;
    std::unique_ptr<PinnedHostBuffer> staging;
    cudaEvent_t staged = nullptr;
    cudaEvent_t done   = nullptr;
    cudaEvent_t routes = nullptr; // the last pass's routes reached the host
    // Activation planes (capacity: prefill_chunk tokens).
    float* stack            = nullptr;
    std::int32_t* ids       = nullptr;
    std::int32_t* positions = nullptr; // the tokens' sequence indices: KV slots, causal order
    std::int32_t* rope      = nullptr; // their 1-D RoPE positions
    std::int32_t* mrope     = nullptr; // a media prompt's [T, 3] RoPE positions
    std::int32_t* scatter   = nullptr; // columns that take a Vision embedding
    void* mixed             = nullptr; // BF16 [H, T]
    float* inject           = nullptr; // [4, T]
    void* y                 = nullptr; // BF16 or FP32 [H, T]
    void* a                 = nullptr; // generic planes
    void* b                 = nullptr;
    void* c                 = nullptr;
    void* d                 = nullptr;
    void* e                 = nullptr;
    void* f                 = nullptr;
    void* g                 = nullptr;
    float* route_weights    = nullptr;
    float* route_shared     = nullptr;
    std::int32_t* selected  = nullptr;
    std::int32_t* counts    = nullptr;
    void* rows              = nullptr; // staged n-gram rows
    void* logits            = nullptr; // head rank only
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    TextConfig config;
    NgramHashConstants ngram;
    std::unique_ptr<NgramTableReader> table;
    Weight embedding_table;
    Projection head;
    HcPlan final_mixer;
    std::vector<LayerPlan> layers;
    std::vector<Segment> segments;
    std::vector<RankState> ranks;
    std::vector<SequenceState> sequences;
    std::uint32_t pages          = 0; // KV pages per sequence and attention layer
    PagedKVStorageLayout kv_layout;   // the planes of every sparse-attention layer's KV
    std::uint32_t pooled_slots   = 0; // indexer blocks per sequence and attention layer
    std::uint32_t max_logit_rows = 0;
    ExecutorMemory memory;
    std::vector<std::uint64_t> row_ids;
    std::vector<std::uint8_t> row_bytes_host;
    // Expert cache (host-resident experts): every layer's routes of the last pass, on its device
    // and in pinned host memory, and how many tokens they cover.
    std::unique_ptr<ExpertCache> cache;
    // Disk-resident experts: made resident before each layer's experts run.
    std::unique_ptr<ExpertStream> stream;
    std::vector<DeviceBuffer> route_records;
    std::unique_ptr<PinnedHostBuffer> route_host;
    std::uint32_t pending_routes = 0;

    // Host-resident experts, wide calls: a layer's routed experts that the cache does not hold are
    // copied into device slots and the matrix kernel reads them there; one pool and one set of
    // tables per rank, whose layers run one at a time.
    struct SlotPool {
        DeviceBuffer storage; // zeroed slots, each down matrix followed by kSlotTail zero bytes
        std::uint64_t slot_bytes = 0;
        std::array<std::uint64_t, 3> offset{};
        std::uint32_t slots = 0;
        DeviceBuffer tables; // gate, up and down tables of every expert
        // Bytes of each slot's down region a down has written: past a smaller down from another
        // layer, the tail is zeroed again.
        std::vector<std::uint64_t> down_written;
    };

    std::vector<SlotPool> slot_pools; // by rank; empty unless the experts are host resident
    std::unique_ptr<PinnedHostBuffer> slot_entries;

    // Vision (a model loaded with its tower), on rank 0 beside the token embedding: the encoder's
    // workspace, whose handoff region receives one item at a time, and the merged embeddings of
    // the media of the prompt a sequence prefills, item after item.
    std::optional<qwen3_5::execution::VisionParameters> vision_parameters;
    std::unique_ptr<qwen3_5::execution::VisionContext> vision_context;
    qwen3_5::execution::VisionWorkspacePlan vision_plan;
    DeviceBuffer vision_backing, vision_embeddings;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(std::move(o)), config(m.config()),
          ngram(derive_ngram_hash_constants(config.ngram)) {
        if (options.prefill_chunk == 0 || options.prefill_chunk > 4096 || options.sequences == 0 ||
            options.max_context == 0) {
            throw std::invalid_argument("qwen4_exp executor: chunk must be in [1, 4096]");
        }
        if (options.ngram) {
            table =
                std::make_unique<NgramTableReader>(options.ngram->layout, options.ngram_residency);
        }
        max_logit_rows = std::min<std::uint32_t>(options.prefill_chunk, 512);
        pages          = (options.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        pooled_slots   = options.max_context / config.indexer_compress_ratio + 1;
        kv_layout =
            paged_kv_storage_layout(options.kv_cache, static_cast<std::int32_t>(config.head_dim));
        memory.ranks.resize(device.size());
        plan_weights();
        plan_segments();
        allocate_ranks();
        allocate_vision();
        allocate_sequences();
        allocate_cache();
        for (const auto& rank : memory.ranks) {
            memory.state_bytes += rank.state_bytes;
            memory.workspace_bytes += rank.workspace_bytes;
            memory.expert_cache_bytes += rank.expert_cache_bytes;
        }
    }

    ~Impl() {
        for (auto& sequence : sequences) {
            for (std::size_t i = 0; i < sequence.decode.size(); ++i) {
                RankBinding bind(device, segments[i].rank);
                sequence.decode[i].reset();
            }
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            if (rank.staged != nullptr) { cudaEventDestroy(rank.staged); }
            if (rank.done != nullptr) { cudaEventDestroy(rank.done); }
            if (rank.routes != nullptr) { cudaEventDestroy(rank.routes); }
        }
    }

    void plan_weights() {
        const auto& w   = model.weights();
        embedding_table = native_weight(model.weight(w.token_embedding).view);
        {
            RankBinding bind(device, model.head_rank());
            head        = make_projection(model, {w.output_head});
            final_mixer = make_hc(model, w.final_mixer, config);
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan plan;
            plan.rank = model.stages().placement(i).stage;
            RankBinding bind(device, plan.rank);
            plan.attn_hc = make_hc(model, lw.attn_hc, config);
            plan.mlp_hc  = make_hc(model, lw.mlp_hc, config);
            if (const auto* g = std::get_if<GdnWeights>(&lw.mixer)) {
                GdnPlan gdn;
                gdn.qkv             = make_projection(model, {g->query, g->key, g->value});
                gdn.z               = make_projection(model, {g->z});
                gdn.a               = make_projection(model, {g->a_projection});
                gdn.b               = make_projection(model, {g->b_projection});
                gdn.output          = make_projection(model, {g->output});
                const auto channels = static_cast<std::int32_t>(gdn.qkv.rows);
                gdn.convolution =
                    direct(model, g->convolution,
                           {channels, static_cast<std::int32_t>(config.linear_conv_kernel_dim)});
                const auto heads = static_cast<std::int32_t>(config.linear_num_value_heads);
                gdn.a_log        = direct(model, g->a_log, {heads}, DType::FP32);
                gdn.dt_bias      = direct(model, g->dt_bias, {heads}, DType::FP32);
                gdn.norm = direct(model, g->norm,
                                  {static_cast<std::int32_t>(config.linear_value_head_dim)});
                plan.gdn = std::move(gdn);
            } else {
                const auto& a = std::get<SparseAttentionWeights>(lw.mixer);
                QsaPlan qsa;
                qsa.query            = make_projection(model, {a.query});
                qsa.gate             = make_projection(model, {a.gate});
                qsa.key              = make_projection(model, {a.key});
                qsa.value            = make_projection(model, {a.value});
                qsa.index            = make_projection(model, {a.index_query, a.index_key});
                qsa.output           = make_projection(model, {a.output});
                const auto d         = static_cast<std::int32_t>(config.head_dim);
                const auto di        = static_cast<std::int32_t>(config.indexer_head_dim);
                qsa.query_norm       = direct(model, a.query_norm, {d});
                qsa.key_norm         = direct(model, a.key_norm, {d});
                qsa.index_query_norm = direct(model, a.index_query_norm, {di});
                qsa.index_key_norm   = direct(model, a.index_key_norm, {di});
                plan.qsa             = std::move(qsa);
            }
            // Without the n-gram table the injection adds nothing (an all-zero embedding projects
            // to zero keys and values), so the layer runs without it.
            if (lw.ple && options.ngram) {
                PlePlan ple;
                ple.key          = make_projection(model, {lw.ple->key});
                ple.value        = make_projection(model, {lw.ple->value});
                const auto width = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
                ple.norm_key     = direct(model, lw.ple->norm_key, {width});
                ple.norm_query   = direct(model, lw.ple->norm_query, {width});
                ple.norm_conv    = direct(model, lw.ple->norm_conv, {width});
                ple.convolution =
                    direct(model, lw.ple->convolution,
                           {static_cast<std::int32_t>(config.ple_conv_kernel_size), width});
                plan.ple = std::move(ple);
            }
            const auto h = static_cast<std::int32_t>(config.hidden_size);
            plan.moe.router =
                direct(model, lw.moe.router, {h, static_cast<std::int32_t>(config.num_experts)});
            plan.moe.shared_gate       = direct(model, lw.moe.shared_score, {h});
            if (lw.moe.gate.empty()) {
                plan.moe.gate = make_located_table(lw.moe.located_gate);
                plan.moe.up   = make_located_table(lw.moe.located_up);
                plan.moe.down = make_located_table(lw.moe.located_down);
            } else {
                plan.moe.gate = make_table(model, lw.moe.gate);
                plan.moe.up   = make_table(model, lw.moe.up);
                plan.moe.down = make_table(model, lw.moe.down);
            }
            plan.moe.device_resident   = model.options().experts != ExpertResidency::Host;
            plan.moe.shared_gate_table = make_table(model, std::span(&lw.moe.shared_gate, 1));
            plan.moe.shared_up_table   = make_table(model, std::span(&lw.moe.shared_up, 1));
            plan.moe.shared_down_table = make_table(model, std::span(&lw.moe.shared_down, 1));
            layers.push_back(std::move(plan));
        }
    }

    void plan_segments() {
        Segment current{.rank = 0, .begin = 0, .end = 0, .embed = true};
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].rank != current.rank) {
                segments.push_back(current);
                current = Segment{.rank = layers[i].rank, .begin = i, .end = i};
            }
            current.end = i + 1;
        }
        if (current.rank != model.head_rank()) {
            segments.push_back(current);
            current =
                Segment{.rank = model.head_rank(), .begin = layers.size(), .end = layers.size()};
        }
        current.head = true;
        segments.push_back(current);
    }

    [[nodiscard]] std::size_t workspace_bytes(std::size_t rank) const {
        const std::int32_t t = static_cast<std::int32_t>(options.prefill_chunk);
        std::size_t bytes    = ops::hyper_connection_read_workspace_bytes(
            static_cast<std::int32_t>(config.hc_count),
            static_cast<std::int32_t>(config.hidden_size),
            static_cast<std::int32_t>(config.hc_lowrank), t);
        bytes = std::max(
            {bytes, ops::moe_experts_gguf_workspace_bytes(t),
             ops::moe_route_workspace_bytes(t, static_cast<std::int32_t>(config.num_experts))});
        for (const auto& plan : layers) {
            if (plan.rank != rank) { continue; }
            if (plan.gdn) {
                for (const Projection* p : {&plan.gdn->qkv, &plan.gdn->z, &plan.gdn->a,
                                            &plan.gdn->b, &plan.gdn->output}) {
                    bytes = std::max(bytes, projection_workspace(*p, t));
                }
                bytes =
                    std::max(bytes, ops::gated_delta_net_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(config.linear_num_key_heads),
                                        static_cast<std::int32_t>(config.linear_num_value_heads),
                                        true, 1, t));
            }
            if (plan.qsa) {
                for (const Projection* p :
                     {&plan.qsa->query, &plan.qsa->gate, &plan.qsa->key, &plan.qsa->value,
                      &plan.qsa->index, &plan.qsa->output}) {
                    bytes = std::max(bytes, projection_workspace(*p, t));
                }
                bytes = std::max({bytes,
                                  ops::qsa_indexer_select_workspace_bytes(
                                      t, static_cast<std::int32_t>(pooled_slots)),
                                  ops::sparse_softmax_attention_workspace_bytes(t)});
            }
            if (plan.ple) {
                bytes = std::max({bytes, projection_workspace(plan.ple->key, t),
                                  projection_workspace(plan.ple->value, t),
                                  ops::ple_inject_workspace_bytes(t)});
            }
        }
        if (rank == model.head_rank()) {
            bytes = std::max(bytes,
                             projection_workspace(head, static_cast<std::int32_t>(max_logit_rows)));
        }
        return bytes + (16U << 20);
    }

    void allocate_ranks() {
        const std::uint64_t t     = options.prefill_chunk;
        const std::uint64_t h     = config.hidden_size;
        const std::uint64_t width = std::uint64_t(config.hc_count) * h;
        const std::uint64_t plane = std::max<std::uint64_t>(width, 6144); // widest BF16 plane rows
        const std::uint64_t row_b = options.ngram ? ops::ngram_row_bytes(options.ngram->format) : 0;
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            RankState rank;
            rank.rank                                            = r;
            rank.stream                                          = device.rank(r).stream;
            std::vector<std::pair<void**, std::uint64_t>> planes = {
                {reinterpret_cast<void**>(&rank.stack), width * t * 4},
                {reinterpret_cast<void**>(&rank.ids), t * 4},
                {reinterpret_cast<void**>(&rank.positions), t * 4},
                {reinterpret_cast<void**>(&rank.rope), t * 4},
                {reinterpret_cast<void**>(&rank.mrope), 3 * t * 4},
                {reinterpret_cast<void**>(&rank.scatter), t * 4},
                {&rank.mixed, h * t * 2},
                {reinterpret_cast<void**>(&rank.inject), std::uint64_t(config.hc_count) * t * 4},
                {&rank.y, h * t * 4},
                {&rank.a, plane * t * 2},
                {&rank.b, plane * t * 2},
                {&rank.c, plane * t * 2},
                {&rank.d, plane * t * 2},
                {&rank.e, plane * t * 2},
                {&rank.f, plane * t * 2},
                {&rank.g, plane * t * 4},
                {reinterpret_cast<void**>(&rank.route_weights),
                 std::uint64_t(config.num_experts_per_tok) * t * 4},
                {reinterpret_cast<void**>(&rank.route_shared), t * 4},
                {reinterpret_cast<void**>(&rank.selected),
                 std::uint64_t(config.indexer_block_budget()) * t * 4},
                {reinterpret_cast<void**>(&rank.counts), t * 4},
                {&rank.rows, std::uint64_t(config.ngram_heads()) * t * row_b},
            };
            if (r == model.head_rank()) {
                planes.push_back(
                    {&rank.logits, std::uint64_t(config.vocab_size) * max_logit_rows * 2});
            }
            std::uint64_t total = 0;
            for (const auto& [slot, bytes] : planes) { total += round_up(bytes); }
            rank.activations     = DeviceBuffer(total);
            std::uint64_t offset = 0;
            for (const auto& [slot, bytes] : planes) {
                *slot = static_cast<std::byte*>(rank.activations.p) + offset;
                offset += round_up(bytes);
            }
            rank.workspace = std::make_unique<DeviceArena>(workspace_bytes(r));
            std::vector<std::int32_t> identity(pages);
            std::iota(identity.begin(), identity.end(), 0);
            rank.block_table = DeviceBuffer(identity.size() * 4);
            rank.block_table.copy_from_host(identity.data(), identity.size() * 4);
            // ids, positions, RoPE positions, scatter columns, three-axis RoPE positions, rows.
            rank.staging = std::make_unique<PinnedHostBuffer>(
                round_up(t * 28) + round_up(std::uint64_t(config.ngram_heads()) * t * row_b));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.routes, cudaEventDisableTiming));
            memory.ranks[r].workspace_bytes += total + rank.workspace->capacity();
            ranks.push_back(std::move(rank));
        }
    }

    void allocate_vision() {
        const auto& vision = model.vision_config();
        if (!vision) { return; }
        if (options.vision_max_merged_tokens == 0) {
            throw std::invalid_argument("qwen4_exp: Vision needs a positive merged-token limit");
        }
        RankBinding bind(device, 0);
        vision_parameters.emplace(vision_parameters_of(*model.vision_weights()));
        vision_plan = qwen3_5::execution::VisionContext::plan_workspace(
            *vision, *vision_parameters, options.vision_max_merged_tokens, 1);
        if (vision_plan.output_hidden != static_cast<std::int32_t>(config.hidden_size)) {
            throw std::invalid_argument("qwen4_exp: the Vision merger's width differs from the "
                                        "text model's");
        }
        vision_backing    = DeviceBuffer(vision_plan.capacity_bytes);
        vision_embeddings = DeviceBuffer(std::size_t(config.hidden_size) *
                                         options.vision_max_merged_tokens * 2);
        vision_context    = std::make_unique<qwen3_5::execution::VisionContext>(
            device, *vision, *vision_parameters, ranks[0].stream);
        memory.ranks[0].workspace_bytes += vision_backing.bytes + vision_embeddings.bytes;
    }

    [[nodiscard]] qwen3_5::execution::VisionParameters
    vision_parameters_of(const qwen3_5::VisionWeights& w) const {
        return vision_parameters_for(model, w);
    }

    void set_media(std::uint32_t s, std::span<const MediaItem> items,
                   std::vector<std::int32_t> rope_positions, std::int32_t rope_delta) {
        if (!vision_context) {
            throw std::invalid_argument("the model was loaded without its Vision tower");
        }
        SequenceState& sequence = sequences.at(s);
        if (sequence.position != 0) {
            throw std::invalid_argument("qwen4_exp: media start a prompt at position zero");
        }
        if (rope_positions.empty() || rope_positions.size() % 3) {
            throw std::invalid_argument("qwen4_exp: RoPE positions must cover three axes");
        }
        const std::size_t tokens = rope_positions.size() / 3;
        std::vector<std::int32_t> columns(tokens, -1);
        RankBinding bind(device, 0);
        const DeviceSpan backing{vision_backing.p, vision_backing.bytes};
        const std::size_t h = config.hidden_size;
        std::size_t column  = 0;
        for (const MediaItem& item : items) {
            const auto& control = *item.control;
            if (column + control.merged_count > options.vision_max_merged_tokens) {
                throw std::invalid_argument("the prompt's media exceed the Vision merged-token "
                                            "limit (--vision-max-merged)");
            }
            Tensor output = qwen3_5::execution::VisionContext::bind_output(backing, vision_plan,
                                                                           control.merged_count);
            vision_context->encode(qwen3_5::execution::VisionItemView{item.patches, &control},
                                   output, backing, vision_plan);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(vision_embeddings.p) +
                                           column * h * 2,
                                       output.data, output.bytes(), cudaMemcpyDeviceToDevice,
                                       ranks[0].stream));
            for (std::size_t j = 0; j < control.scatter_indices.size(); ++j) {
                const std::int32_t at = control.scatter_indices[j];
                if (at < 0 || std::size_t(at) >= tokens) {
                    throw std::invalid_argument("qwen4_exp: a media token lies outside the prompt");
                }
                columns[std::size_t(at)] = static_cast<std::int32_t>(column + j);
            }
            column += control.merged_count;
        }
        sequence.media_columns = std::move(columns);
        sequence.media_rope    = std::move(rope_positions);
        sequence.rope_delta    = rope_delta;
        CUDA_CHECK(cudaStreamSynchronize(ranks[0].stream));
    }

    void allocate_sequences() {
        const std::uint64_t width = std::uint64_t(config.hc_count) * config.hidden_size;
        for (std::uint32_t s = 0; s < options.sequences; ++s) {
            SequenceState sequence;
            for (const auto& plan : layers) {
                RankBinding bind(device, plan.rank);
                LayerState state;
                if (plan.gdn) {
                    const std::uint64_t dk = config.linear_key_head_dim,
                                        dv = config.linear_value_head_dim;
                    state.ssm  = DeviceBuffer(dk * dv * config.linear_num_value_heads * 4);
                    state.conv = DeviceBuffer(std::uint64_t(plan.gdn->qkv.rows) *
                                              (config.linear_conv_kernel_dim - 1) * 2);
                    memory.ranks[plan.rank].state_bytes += state.ssm.bytes + state.conv.bytes;
                }
                if (plan.qsa) {
                    // One plane per K or V vector and per scale, each [leading, page, head, page].
                    const auto plane = [&](std::int32_t leading, DType dtype) {
                        return DeviceBuffer(std::uint64_t(leading) * kPagedKVPageSize *
                                            config.num_key_value_heads * pages * dtype_size(dtype));
                    };
                    state.k_pages =
                        plane(kv_layout.key.data_leading_extent, kv_layout.key.data_dtype);
                    state.v_pages =
                        plane(kv_layout.value.data_leading_extent, kv_layout.value.data_dtype);
                    if (kv_layout.key.has_scale()) {
                        state.k_scales =
                            plane(kv_layout.key.scale_leading_extent, kv_layout.key.scale_dtype);
                    }
                    if (kv_layout.value.has_scale()) {
                        state.v_scales = plane(kv_layout.value.scale_leading_extent,
                                               kv_layout.value.scale_dtype);
                    }
                    state.pooled =
                        DeviceBuffer(std::uint64_t(config.indexer_head_dim) * pooled_slots * 4);
                    state.tail = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                              (config.indexer_compress_ratio - 1) * 4);
                    const std::uint64_t kv = state.k_pages.bytes + state.v_pages.bytes +
                                             state.k_scales.bytes + state.v_scales.bytes;
                    memory.ranks[plan.rank].state_bytes +=
                        kv + state.pooled.bytes + state.tail.bytes;
                    memory.kv_bytes += kv;
                }
                if (plan.ple) {
                    state.history = DeviceBuffer(width * (config.ple_conv_kernel_size - 1) *
                                                 config.ngram.ngram_size * 4);
                    memory.ranks[plan.rank].state_bytes += state.history.bytes;
                }
                sequence.layers.push_back(std::move(state));
            }
            sequences.push_back(std::move(sequence));
            reset(s);
        }
    }

    void allocate_cache() {
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        for (const auto& plan : layers) {
            RankBinding bind(device, plan.rank);
            route_records.emplace_back(pairs * sizeof(std::int32_t));
        }
        route_host =
            std::make_unique<PinnedHostBuffer>(layers.size() * pairs * sizeof(std::int32_t));
        const ExpertResidency residency = model.options().experts;
        if (residency == ExpertResidency::Host) { allocate_slot_pools(); }
        if (residency == ExpertResidency::Device ||
            (residency == ExpertResidency::Host && options.expert_cache_bytes == 0)) {
            return;
        }
        // What each rank lends: the request split evenly, or what is free less a margin for the
        // allocations later startup makes (sampling, CUDA context growth).
        constexpr std::uint64_t kMargin = 1536ULL << 20;
        std::vector<std::uint64_t> bytes(device.size(), 0);
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            const std::uint64_t available = free_bytes > kMargin ? free_bytes - kMargin : 0;
            bytes[r] = options.expert_cache_bytes == ExecutorOptions::kAutoExpertCache
                           ? available
                           : std::min<std::uint64_t>(available,
                                                     options.expert_cache_bytes / device.size());
        }
        if (residency == ExpertResidency::Disk) {
            if (options.expert_cache_bytes == 0) {
                throw std::invalid_argument("disk-resident experts need a device expert cache");
            }
            std::vector<ExpertStreamLayer> stream_layers;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const auto& moe = model.weights().layers[i].moe;
                stream_layers.push_back(ExpertStreamLayer{
                    .rank    = layers[i].rank,
                    .stream  = ranks[layers[i].rank].stream,
                    .experts = {moe.located_gate, moe.located_up, moe.located_down},
                    .tables  = {layers[i].moe.gate.table.p, layers[i].moe.up.table.p,
                                layers[i].moe.down.table.p}});
            }
            stream = std::make_unique<ExpertStream>(device, model.files(), std::move(stream_layers),
                                                    bytes);
            for (std::size_t r = 0; r < bytes.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += bytes[r];
            }
            return;
        }
        std::vector<ExpertCacheLayer> cache_layers;
        for (const auto& plan : layers) {
            const auto bank = [](const ExpertTable& table) {
                return ExpertBank{.expert_bytes = std::int64_t(table.rows) * table.row_bytes,
                                  .host         = table.pointers,
                                  .table        = table.table.p};
            };
            cache_layers.push_back(ExpertCacheLayer{.rank   = plan.rank,
                                                    .stream = ranks[plan.rank].stream,
                                                    .gate   = bank(plan.moe.gate),
                                                    .up     = bank(plan.moe.up),
                                                    .down   = bank(plan.moe.down)});
        }
        cache = std::make_unique<ExpertCache>(device, std::move(cache_layers), bytes);
        for (std::size_t r = 0; r < bytes.size(); ++r) {
            memory.ranks[r].expert_cache_bytes += bytes[r];
        }
    }

    // One slot per expert on every rank, sized for its layers' widest projections, when half the
    // free memory holds them; the expert cache takes what is left.
    void allocate_slot_pools() {
        const std::size_t experts = config.num_experts;
        slot_pools.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            std::array<std::uint64_t, 3> widest{};
            for (const auto& plan : layers) {
                if (plan.rank != r) { continue; }
                const ExpertTable* tables[3] = {&plan.moe.gate, &plan.moe.up, &plan.moe.down};
                for (int k = 0; k < 3; ++k) {
                    widest[k] = std::max<std::uint64_t>(widest[k], std::uint64_t(tables[k]->rows) *
                                                                       tables[k]->row_bytes);
                }
            }
            if (widest[0] == 0) { continue; }
            SlotPool& pool  = slot_pools[r];
            pool.offset     = {0, round_up(widest[0]), round_up(widest[0]) + round_up(widest[1])};
            pool.slot_bytes = pool.offset[2] + round_up(widest[2] + kSlotTail);
            RankBinding bind(device, r);
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            const std::uint64_t bytes = experts * pool.slot_bytes;
            if (bytes > free_bytes / 2) { continue; }
            pool.slots   = static_cast<std::uint32_t>(experts);
            pool.down_written.assign(experts, 0);
            pool.storage = DeviceBuffer(bytes);
            CUDA_CHECK(cudaMemset(pool.storage.p, 0, pool.storage.bytes));
            pool.tables = DeviceBuffer(3 * experts * sizeof(void*));
            memory.ranks[r].workspace_bytes += pool.storage.bytes + pool.tables.bytes;
        }
        slot_entries = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
    }

    // Feeds the last pass's routes to the expert cache and lets it swap experts, before the next
    // pass reads the tables.
    void settle_routes() {
        if (pending_routes == 0) { return; }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaEventSynchronize(rank.routes));
        }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        const auto* host = static_cast<const std::int32_t*>(route_host->data());
        if (cache) {
            const std::size_t used = std::size_t(config.num_experts_per_tok) * pending_routes;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                cache->observe(i, std::span(host + i * pairs, used), pending_routes);
            }
            // A prompt chunk moves the working set; a decode step only nudges it.
            cache->rebalance(pending_routes > 1 ? (2048ULL << 20) : (48ULL << 20));
        }
        pending_routes = 0;
    }

    // Brings the pass's routes to the host for the expert cache's counts; nothing else reads them.
    void record_routes(std::uint32_t tokens) {
        if (!cache) { return; }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        auto* host = static_cast<std::int32_t*>(route_host->data());
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            CUDA_CHECK(cudaMemcpyAsync(host + i * pairs, route_records[i].p,
                                       std::size_t(config.num_experts_per_tok) * tokens * 4,
                                       cudaMemcpyDeviceToHost, ranks[layers[i].rank].stream));
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaEventRecord(rank.routes, rank.stream));
        }
        pending_routes = tokens;
    }

    void reset(std::uint32_t s) {
        auto& sequence    = sequences.at(s);
        sequence.position = 0;
        sequence.context  = NgramContext::sequence_start(ngram, config.eos_token_id);
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            const cudaStream_t stream = ranks[layers[i].rank].stream;
            for (DeviceBuffer* buffer :
                 {&sequence.layers[i].ssm, &sequence.layers[i].conv, &sequence.layers[i].pooled,
                  &sequence.layers[i].tail, &sequence.layers[i].history}) {
                if (buffer->p != nullptr) {
                    CUDA_CHECK(cudaMemsetAsync(buffer->p, 0, buffer->bytes, stream));
                }
            }
        }
    }

    // Moves the stack's first `tokens` columns from one rank to another, ordered after the
    // source's work.
    void cross(std::size_t from, std::size_t to, std::int32_t tokens) {
        RankState& source = ranks[from];
        RankState& target = ranks[to];
        {
            RankBinding bind(device, from);
            CUDA_CHECK(cudaEventRecord(source.done, source.stream));
        }
        RankBinding bind(device, to);
        CUDA_CHECK(cudaStreamWaitEvent(target.stream, source.done, 0));
        const std::size_t bytes =
            std::size_t(config.hc_count) * config.hidden_size * std::size_t(tokens) * 4;
        CUDA_CHECK(cudaMemcpyPeerAsync(target.stack, device.rank(to).device, source.stack,
                                       device.rank(from).device, bytes, target.stream));
    }

    // A pass's Vision work: the runs of consecutive embedding columns its tokens take (each run's
    // destination columns sit in the scatter plane at `offset`), and whether its one part rotates
    // at a media prompt's three-axis positions.
    struct MediaRun {
        std::int32_t source = 0, count = 0, offset = 0;
    };
    std::vector<MediaRun> media_runs;
    bool mrope = false;

    void stage_inputs(std::span<const Part> parts, std::span<const std::int32_t> tokens) {
        const auto t            = static_cast<std::int32_t>(tokens.size());
        const std::size_t c     = options.prefill_chunk;
        // Host side first: positions (sequence indices and RoPE positions), the n-gram rows of the
        // PLE layer's tokens, each part from its own sequence's position and n-gram context, and
        // where a media prompt's tokens take their embeddings and rotate.
        std::vector<std::int32_t> positions(tokens.size()), rope(tokens.size()), scatter;
        std::vector<std::int32_t> mrope_positions;
        std::size_t row_b       = 0;
        const std::size_t heads = config.ngram_heads();
        if (table) {
            row_ids.resize(tokens.size() * heads);
            row_b = ops::ngram_row_bytes(options.ngram->format);
        }
        media_runs.clear();
        mrope = false;
        for (const Part& part : parts) {
            const SequenceState& sequence = *part.sequence;
            const auto first = positions.begin() + part.column;
            std::iota(first, first + part.count, static_cast<std::int32_t>(sequence.position));
            for (std::int32_t i = 0; i < part.count; ++i) {
                rope[std::size_t(part.column + i)] =
                    static_cast<std::int32_t>(sequence.position) + i + sequence.rope_delta;
            }
            if (table) {
                ngram_row_ids(ngram, tokens.subspan(std::size_t(part.column), std::size_t(part.count)),
                              config.eos_token_id, config.vocab_size, part.sequence->context,
                              std::span(row_ids).subspan(std::size_t(part.column) * heads,
                                                         std::size_t(part.count) * heads));
            }
            const std::size_t prompt = sequence.media_rope.size() / 3;
            if (prompt == 0 || sequence.position >= prompt) { continue; }
            if (parts.size() != 1) {
                throw std::logic_error("qwen4_exp: a media prompt prefills in a pass of its own");
            }
            mrope = true;
            mrope_positions.resize(3 * std::size_t(t));
            for (std::int32_t i = 0; i < t; ++i) {
                const std::size_t at = sequence.position + std::size_t(i);
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    mrope_positions[axis * std::size_t(t) + std::size_t(i)] =
                        at < prompt ? sequence.media_rope[axis * prompt + at]
                                    : static_cast<std::int32_t>(at) + sequence.rope_delta;
                }
                const std::int32_t column = at < prompt ? sequence.media_columns[at] : -1;
                if (column < 0) { continue; }
                if (!media_runs.empty() &&
                    media_runs.back().source + media_runs.back().count == column) {
                    ++media_runs.back().count;
                } else {
                    media_runs.push_back({.source = column,
                                          .count  = 1,
                                          .offset = static_cast<std::int32_t>(scatter.size())});
                }
                scatter.push_back(i);
            }
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            // The staging buffer is rewritten only once the previous upload has left it.
            CUDA_CHECK(cudaEventSynchronize(rank.staged));
            auto* base = static_cast<std::byte*>(rank.staging->data());
            const auto upload = [&](void* target, std::size_t at,
                                    const std::vector<std::int32_t>& values) {
                std::memcpy(base + at, values.data(), values.size() * 4);
                CUDA_CHECK(cudaMemcpyAsync(target, base + at, values.size() * 4,
                                           cudaMemcpyHostToDevice, rank.stream));
            };
            std::memcpy(base, tokens.data(), tokens.size() * 4);
            CUDA_CHECK(cudaMemcpyAsync(rank.ids, base, tokens.size() * 4, cudaMemcpyHostToDevice,
                                       rank.stream));
            upload(rank.positions, 4 * c, positions);
            upload(rank.rope, 8 * c, rope);
            if (!scatter.empty() && rank.rank == 0) { upload(rank.scatter, 12 * c, scatter); }
            if (mrope) { upload(rank.mrope, 16 * c, mrope_positions); }
            const bool ple_here =
                std::any_of(layers.begin(), layers.end(),
                            [&](const LayerPlan& p) { return p.ple && p.rank == rank.rank; });
            if (ple_here) {
                auto* rows = base + round_up(28 * std::uint64_t(c));
                table->read_rows(row_ids, std::span(reinterpret_cast<std::uint8_t*>(rows),
                                                    row_ids.size() * row_b));
                CUDA_CHECK(cudaMemcpyAsync(rank.rows, rows, row_ids.size() * row_b,
                                           cudaMemcpyHostToDevice, rank.stream));
            }
            CUDA_CHECK(cudaEventRecord(rank.staged, rank.stream));
        }
    }

    // The part's columns of a BF16 [rows, *] plane.
    static void* columns(void* plane, std::int32_t rows, std::int32_t column, std::size_t bytes) {
        return static_cast<std::byte*>(plane) + std::size_t(rows) * std::size_t(column) * bytes;
    }

    void run_gdn(const LayerPlan& plan, LayerState& state, RankState& rank, const Part& part) {
        const GdnPlan& g     = *plan.gdn;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto t         = part.count;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto kd =
            static_cast<std::int32_t>(config.linear_num_key_heads * config.linear_key_head_dim);
        const auto vd =
            static_cast<std::int32_t>(config.linear_num_value_heads * config.linear_value_head_dim);
        const auto nk = static_cast<std::int32_t>(config.linear_num_key_heads);
        const auto nv = static_cast<std::int32_t>(config.linear_num_value_heads);
        const auto dh = static_cast<std::int32_t>(config.linear_value_head_dim);
        const Tensor mixed(columns(rank.mixed, h, part.column, 2), DType::BF16, {h, t});
        Tensor qkv(rank.a, DType::BF16, {g.qkv.rows, t});
        Tensor z(rank.b, DType::BF16, {vd, t});
        Tensor ga(rank.c, DType::BF16, {nv, t});
        Tensor gb(static_cast<std::byte*>(rank.c) + round_up(std::uint64_t(nv) * t * 2),
                  DType::BF16, {nv, t});
        project(g.qkv, mixed, qkv, ws, s);
        project(g.z, mixed, z, ws, s);
        project(g.a, mixed, ga, ws, s);
        project(g.b, mixed, gb, ws, s);
        // Convolution in place of the projections' q/k/v planes.
        auto* conv_out = static_cast<std::byte*>(rank.d);
        Tensor q(conv_out, DType::BF16, {kd, t});
        Tensor k(conv_out + round_up(std::uint64_t(kd) * t * 2), DType::BF16, {kd, t});
        Tensor v(rank.e, DType::BF16, {vd, t});
        Tensor conv_state(
            state.conv.p, DType::BF16,
            {g.qkv.rows, static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1)});
        ops::causal_conv1d_silu_split(qkv, g.convolution, conv_state, conv_state, q, k, v, s);
        Tensor gate(rank.g, DType::FP32, {nv, t});
        Tensor beta(static_cast<std::byte*>(rank.g) + round_up(std::uint64_t(nv) * t * 4),
                    DType::FP32, {nv, t});
        ops::gdn_gating(ga, gb, g.a_log, g.dt_bias, gate, beta, s);
        Tensor q3(q.data, DType::BF16, {dh, nk, t}), k3(k.data, DType::BF16, {dh, nk, t});
        Tensor v3(v.data, DType::BF16, {dh, nv, t});
        Tensor ssm(state.ssm.p, DType::FP32, {dh, dh, nv});
        Tensor o(rank.f, DType::BF16, {dh, nv, t});
        {
            auto scope = ws.scope();
            ops::gated_delta_net(q3, k3, v3, gate, beta, 1.0F / std::sqrt(float(dh)), true, ws, ssm,
                                 o, s);
        }
        Tensor rows_o(rank.f, DType::BF16, {dh, nv * t});
        Tensor rows_z(rank.b, DType::BF16, {dh, nv * t});
        Tensor gated_rows(rank.a, DType::BF16, {dh, nv * t});
        ops::gated_rmsnorm(rows_o, g.norm, rows_z, ops::GateActivation::Sigmoid,
                           config.rms_norm_eps, gated_rows, s);
        const Tensor gated(rank.a, DType::BF16, {vd, t});
        Tensor y(columns(rank.y, h, part.column, 2), DType::BF16, {h, t});
        project(g.output, gated, y, ws, s);
    }

    void run_qsa(const LayerPlan& plan, LayerState& state, RankState& rank, const Part& part) {
        const QsaPlan& a     = *plan.qsa;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto t         = part.count;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto d         = static_cast<std::int32_t>(config.head_dim);
        const auto nq        = static_cast<std::int32_t>(config.num_attention_heads);
        const auto nk        = static_cast<std::int32_t>(config.num_key_value_heads);
        const Tensor mixed(columns(rank.mixed, h, part.column, 2), DType::BF16, {h, t});
        Tensor q(rank.a, DType::BF16, {d * nq, t});
        Tensor gate(rank.b, DType::BF16, {d * nq, t});
        auto* kv = static_cast<std::byte*>(rank.c);
        Tensor k(kv, DType::BF16, {d * nk, t});
        Tensor v(kv + round_up(std::uint64_t(d) * nk * t * 2), DType::BF16, {d * nk, t});
        Tensor index(kv + 2 * round_up(std::uint64_t(d) * nk * t * 2), DType::BF16,
                     {a.index.rows, t});
        project(a.query, mixed, q, ws, s);
        project(a.gate, mixed, gate, ws, s);
        project(a.key, mixed, k, ws, s);
        project(a.value, mixed, v, ws, s);
        project(a.index, mixed, index, ws, s);
        Tensor q3(rank.a, DType::BF16, {d, nq, t}), k3(k.data, DType::BF16, {d, nk, t});
        Tensor qo(rank.d, DType::BF16, {d, nq, t});
        Tensor ko(rank.e, DType::BF16, {d, nk, t});
        std::int32_t* const part_positions = rank.positions + part.column;
        const Tensor positions(part_positions, DType::I32, {t});
        if (mrope) {
            // A media prompt: the three-axis positions of its images and video, each pair of
            // rotated dimensions on axis i % 3 (the interleaved MRoPE sections).
            const Tensor axes(rank.mrope, DType::I32, {t, 3});
            ops::rmsnorm(q3, a.query_norm, config.rms_norm_eps, true, qo, s);
            ops::rmsnorm(k3, a.key_norm, config.rms_norm_eps, true, ko, s);
            ops::rope(axes, static_cast<int>(float(d) * config.partial_rotary_factor),
                      config.rope_theta, qo, ko, s);
        } else {
            const Tensor rope(rank.rope + part.column, DType::I32, {t});
            ops::rmsnorm_rope(rope, a.query_norm, a.key_norm, q3, k3, qo, ko, s);
        }
        const auto plane = [&](const DeviceBuffer& memory, std::int32_t leading, DType dtype) {
            return Tensor(memory.p, dtype,
                          {leading, static_cast<std::int32_t>(kPagedKVPageSize), nk,
                           static_cast<std::int32_t>(pages)});
        };
        PagedKVLayerView cache{};
        cache.k_pages =
            plane(state.k_pages, kv_layout.key.data_leading_extent, kv_layout.key.data_dtype);
        cache.v_pages =
            plane(state.v_pages, kv_layout.value.data_leading_extent, kv_layout.value.data_dtype);
        if (kv_layout.key.has_scale()) {
            cache.k_scale_pages = plane(state.k_scales, kv_layout.key.scale_leading_extent,
                                        kv_layout.key.scale_dtype);
        }
        if (kv_layout.value.has_scale()) {
            cache.v_scale_pages = plane(state.v_scales, kv_layout.value.scale_leading_extent,
                                        kv_layout.value.scale_dtype);
        }
        cache.block_table =
            Tensor(rank.block_table.p, DType::I32, {static_cast<std::int32_t>(pages)});
        cache.head_dim     = d;
        cache.num_kv_heads = nk;
        cache.storage      = options.kv_cache;
        const Tensor v3(v.data, DType::BF16, {d, nk, t});
        ops::kv_cache_append(ko, v3, positions, cache, s);
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        Tensor pooled(state.pooled.p, DType::FP32, {di, static_cast<std::int32_t>(pooled_slots)});
        Tensor tail(state.tail.p, DType::FP32,
                    {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
        const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
        // The call's first position is the part's first staged one, read on the device.
        const Tensor first(part_positions, DType::I32, {1});
        ops::qsa_indexer_append(index, first, iw, config.rms_norm_eps, pooled, tail, s);
        Tensor selected(rank.selected, DType::I32,
                        {static_cast<std::int32_t>(config.indexer_block_budget()), t});
        Tensor counts(rank.counts, DType::I32, {t});
        {
            auto scope = ws.scope();
            ops::qsa_indexer_select(index, first, iw, config.rms_norm_eps, pooled, ws, selected,
                                    counts, s);
        }
        Tensor attention(rank.f, DType::BF16, {d, nq, t});
        {
            auto scope = ws.scope();
            ops::sparse_softmax_attention(qo, first, selected, counts, cache,
                                          1.0F / std::sqrt(float(d)), ws, attention, s);
        }
        Tensor flat(rank.f, DType::BF16, {d * nq, t});
        ops::sigmoid_mul(gate, flat, s);
        Tensor y(columns(rank.y, h, part.column, 2), DType::BF16, {h, t});
        project(a.output, flat, y, ws, s);
    }

    void run_ple(const LayerPlan& plan, LayerState& state, RankState& rank, const Part& part) {
        const PlePlan& p     = *plan.ple;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto t         = part.count;
        const auto heads     = static_cast<std::int32_t>(config.ngram_heads());
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto width     = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
        const auto row_b = static_cast<std::int32_t>(ops::ngram_row_bytes(options.ngram->format));
        const Tensor rows(columns(rank.rows, row_b * heads, part.column, 1), DType::U8,
                          {row_b, heads * t});
        Tensor stack(rank.stack + std::size_t(width) * std::size_t(part.column), DType::FP32,
                     {h, static_cast<std::int32_t>(config.hc_count), t});
        Tensor embedding(rank.a, DType::BF16, {static_cast<std::int32_t>(config.ple_embed_dim), t});
        ops::ngram_embed_rows(rows, options.ngram->format, heads, embedding, s);
        Tensor key(rank.g, DType::BF16, {width, t});
        Tensor value(rank.b, DType::BF16, {h, t});
        project(p.key, embedding, key, ws, s);
        project(p.value, embedding, value, ws, s);
        Tensor history(state.history.p, DType::FP32,
                       {width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                         config.ngram.ngram_size)});
        auto scope = ws.scope();
        ops::ple_inject(stack, key, value,
                        {&p.norm_key, &p.norm_query, &p.norm_conv, &p.convolution},
                        config.rms_norm_eps, history, ws, s);
    }

    // A wide call over host-resident experts: once its routes reach the host, the routed experts
    // the cache does not hold are copied into the rank's slots, and the experts run from device
    // memory through tables that point at cache slots and pool slots. False, leaving the call to
    // the vector products over the cache's tables, when the pool cannot hold them.
    bool run_through_slots(const LayerPlan& plan, RankState& rank, std::int32_t t,
                           std::size_t index, Tensor& ids, Tensor& weights, Tensor& shared) {
        SlotPool& pool = slot_pools[rank.rank];
        if (pool.slots == 0) { return false; }
        const cudaStream_t s = rank.stream;
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        auto* routes         = static_cast<std::int32_t*>(route_host->data()) +
                               index * std::size_t(top) * options.prefill_chunk;
        CUDA_CHECK(
            cudaMemcpyAsync(routes, ids.data, std::size_t(top) * t * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        std::vector<char> routed(experts, 0);
        for (std::int32_t i = 0; i < top * t; ++i) {
            const std::int32_t e = routes[i];
            if (e >= 0 && std::size_t(e) < experts) { routed[e] = 1; }
        }
        auto* entries                = static_cast<const void**>(slot_entries->data());
        const ExpertTable* tables[3] = {&m.gate, &m.up, &m.down};
        std::uint32_t used           = 0;
        for (std::size_t e = 0; e < experts; ++e) {
            for (int k = 0; k < 3; ++k) { entries[k * experts + e] = nullptr; }
            if (!routed[e]) { continue; }
            if (cache && cache->cached(index, 0, std::int32_t(e)) != nullptr) {
                for (int k = 0; k < 3; ++k) {
                    entries[k * experts + e] = cache->cached(index, k, std::int32_t(e));
                }
                continue;
            }
            if (used == pool.slots) { return false; }
            const std::uint32_t slot = used++;
            auto* base =
                static_cast<std::byte*>(pool.storage.p) + std::size_t(slot) * pool.slot_bytes;
            for (int k = 0; k < 3; ++k) {
                const std::uint64_t bytes = std::uint64_t(tables[k]->rows) * tables[k]->row_bytes;
                std::byte* target         = base + pool.offset[k];
                CUDA_CHECK(cudaMemcpyAsync(target, tables[k]->pointers[e], std::size_t(bytes),
                                           cudaMemcpyHostToDevice, s));
                if (k == 2) {
                    std::uint64_t& written = pool.down_written[slot];
                    if (written > bytes) {
                        CUDA_CHECK(
                            cudaMemsetAsync(target + bytes, 0,
                                            std::size_t(std::min(written - bytes, kSlotTail)), s));
                    }
                    written = std::max(written, bytes);
                }
                entries[k * experts + e] = target;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(pool.tables.p, entries, 3 * experts * sizeof(void*),
                                   cudaMemcpyHostToDevice, s));
        ops::GgufMoeWeights banks = m.banks();
        const auto* table_p       = static_cast<const void* const*>(pool.tables.p);
        banks.gate.experts        = table_p;
        banks.up.experts          = table_p + experts;
        banks.down.experts        = table_p + 2 * experts;
        banks.device_resident     = true;
        const auto h              = static_cast<std::int32_t>(config.hidden_size);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = rank.workspace->scope();
        ops::moe_experts_gguf(mixed, ids, weights, shared, banks, *rank.workspace, y, s);
        return true;
    }

    void run_moe(const LayerPlan& plan, RankState& rank, std::int32_t t, std::size_t index) {
        const MoePlan& m     = plan.moe;
        WorkspaceArena& ws   = *rank.workspace;
        const cudaStream_t s = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        const Tensor mixed(rank.mixed, DType::BF16, {h, t});
        Tensor ids(route_records[index].p, DType::I32, {top, t});
        Tensor weights(rank.route_weights, DType::FP32, {top, t});
        Tensor shared(rank.route_shared, DType::FP32, {t});
        {
            auto scope = ws.scope();
            ops::moe_route(mixed, m.router, m.shared_gate, ws, ids, weights, shared, s);
        }
        if (t > kVectorTokens && !slot_pools.empty() &&
            run_through_slots(plan, rank, t, index, ids, weights, shared)) {
            return;
        }
        if (stream) {
            // The routes reach the host before the layer's experts can be made resident.
            auto* host = static_cast<std::int32_t*>(route_host->data()) +
                         index * std::size_t(top) * options.prefill_chunk;
            CUDA_CHECK(cudaMemcpyAsync(host, ids.data, std::size_t(top) * t * 4,
                                       cudaMemcpyDeviceToHost, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
            stream->prepare(index, std::span<const std::int32_t>(host, std::size_t(top) * t));
        }
        Tensor y(rank.y, DType::FP32, {h, t});
        auto scope = ws.scope();
        ops::moe_experts_gguf(mixed, ids, weights, shared, m.banks(), ws, y, s);
    }

    // Queues one segment of the pass for the parts' `t` tokens on its rank's stream; with `head`,
    // the logits of the last `logit_rows` tokens.
    void run_segment(std::span<const Part> parts, const Segment& segment, std::int32_t t,
                     std::uint32_t logit_rows) {
        RankState& rank = ranks[segment.rank];
        const auto h    = static_cast<std::int32_t>(config.hidden_size);
        const auto hc   = static_cast<std::int32_t>(config.hc_count);
        Tensor stack(rank.stack, DType::FP32, {h, hc, t});
        Tensor mixed(rank.mixed, DType::BF16, {h, t});
        if (segment.embed) {
            const Tensor ids(rank.ids, DType::I32, {t});
            ops::embedding(ids, embedding_table, mixed, rank.stream);
            // A media prompt's image and video tokens take the tower's merged embeddings.
            for (const MediaRun& run : media_runs) {
                const Tensor source(static_cast<std::byte*>(vision_embeddings.p) +
                                        std::size_t(run.source) * std::size_t(h) * 2,
                                    DType::BF16, {h, run.count});
                const Tensor columns(rank.scatter + run.offset, DType::I32, {run.count});
                ops::scatter(source, columns, mixed, rank.stream);
            }
            ops::hyper_connection_expand(mixed, stack, rank.stream);
        }
        for (std::size_t i = segment.begin; i < segment.end; ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.ple) {
                for (const Part& part : parts) {
                    run_ple(plan, part.sequence->layers[i], rank, part);
                }
            }
            Tensor inject(rank.inject, DType::FP32, {hc, t});
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_read(stack, plan.attn_hc.weights(), config.rms_norm_eps,
                                           *rank.workspace, mixed, &inject, rank.stream);
            }
            for (const Part& part : parts) {
                LayerState& state = part.sequence->layers[i];
                if (plan.gdn) {
                    run_gdn(plan, state, rank, part);
                } else {
                    run_qsa(plan, state, rank, part);
                }
            }
            ops::hyper_connection_write(stack, Tensor(rank.y, DType::BF16, {h, t}), inject,
                                        rank.stream);
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_read(stack, plan.mlp_hc.weights(), config.rms_norm_eps,
                                           *rank.workspace, mixed, &inject, rank.stream);
            }
            run_moe(plan, rank, t, i);
            ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}), inject,
                                        rank.stream);
        }
        if (!segment.head) { return; }
        const auto n = static_cast<std::int32_t>(logit_rows);
        const Tensor last(rank.stack + std::size_t(hc) * h * (t - n), DType::FP32, {h, hc, n});
        Tensor head_mixed(rank.mixed, DType::BF16, {h, n});
        {
            auto scope = rank.workspace->scope();
            ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                       *rank.workspace, head_mixed, nullptr, rank.stream);
        }
        Tensor logits(rank.logits, DType::BF16, {static_cast<std::int32_t>(config.vocab_size), n});
        project(head, head_mixed, logits, *rank.workspace, rank.stream);
    }

    void capture_decode(SequenceState& sequence) {
        std::vector<DecodeGraphExecutable> graphs;
        for (const Segment& segment : segments) {
            RankBinding bind(device, segment.rank);
            DecodeGraphDefinition definition;
            const Part part{.sequence = &sequence, .column = 0, .count = 1};
            definition.capture(ranks[segment.rank].stream, [&] {
                run_segment(std::span(&part, 1), segment, 1, 1);
            });
            DecodeGraphExecutable graph;
            graph.instantiate(definition);
            graphs.push_back(std::move(graph));
        }
        sequence.decode = std::move(graphs);
    }

    void check_tokens(std::span<const std::int32_t> tokens, std::uint32_t logit_rows) const {
        if (tokens.empty() || tokens.size() > options.prefill_chunk) {
            throw std::invalid_argument("qwen4_exp forward: 1..prefill_chunk tokens per call");
        }
        if (logit_rows == 0 || logit_rows > tokens.size() || logit_rows > max_logit_rows) {
            throw std::invalid_argument("qwen4_exp forward: invalid logit rows");
        }
        for (const auto token : tokens) {
            if (token < 0 || std::uint32_t(token) >= config.vocab_size) {
                throw std::invalid_argument("qwen4_exp forward: token outside the vocabulary");
            }
        }
    }

    // One pass over `parts`, whose tokens are `tokens` in column order. A pass of one single-token
    // part replays its sequence's decode graphs when it has them.
    void pass(std::span<const Part> parts, std::span<const std::int32_t> tokens,
              std::uint32_t logit_rows) {
        const auto t = static_cast<std::int32_t>(tokens.size());
        for (const Part& part : parts) {
            if (part.sequence->position + std::uint32_t(part.count) > options.max_context) {
                throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
            }
        }
        settle_routes();
        stage_inputs(parts, tokens);
        // Disk-resident experts need the host between a layer's routing and its experts, so their
        // passes stay eager, and so does a batch of several sequences.
        SequenceState& first = *parts.front().sequence;
        const bool graph     = options.cuda_graphs && t == 1 && !stream;
        if (graph && first.decode.empty() && first.decode_steps++ > 0) { capture_decode(first); }
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const Segment& segment = segments[i];
            if (i > 0) { cross(segments[i - 1].rank, segment.rank, t); }
            RankBinding bind(device, segment.rank);
            if (graph && !first.decode.empty()) {
                first.decode[i].launch(ranks[segment.rank].stream);
            } else {
                run_segment(parts, segment, t, logit_rows);
            }
        }
        for (const Part& part : parts) {
            part.sequence->position += static_cast<std::uint32_t>(part.count);
        }
        record_routes(static_cast<std::uint32_t>(t));
    }

    void forward(std::uint32_t s, std::span<const std::int32_t> tokens, std::uint32_t logit_rows) {
        check_tokens(tokens, logit_rows);
        const Part part{.sequence = &sequences.at(s),
                        .column   = 0,
                        .count    = static_cast<std::int32_t>(tokens.size())};
        pass(std::span(&part, 1), tokens, logit_rows);
    }

    void decode(std::span<const std::uint32_t> batch, std::span<const std::int32_t> tokens) {
        if (batch.empty() || batch.size() != tokens.size()) {
            throw std::invalid_argument("qwen4_exp decode: one token per sequence");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(tokens.size()));
        std::vector<Part> parts;
        parts.reserve(batch.size());
        for (std::size_t j = 0; j < batch.size(); ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            for (const Part& other : parts) {
                if (other.sequence == sequence) {
                    throw std::invalid_argument("qwen4_exp decode: a sequence appears twice");
                }
            }
            parts.push_back({.sequence = sequence, .column = std::int32_t(j), .count = 1});
        }
        pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
    }

    void snapshot(std::uint32_t s, SequenceSnapshot& out) {
        const SequenceState& sequence = sequences.at(s);
        if (out.layers.size() != layers.size()) {
            out.layers.clear();
            for (std::size_t i = 0; i < layers.size(); ++i) {
                RankBinding bind(device, layers[i].rank);
                const LayerState& state = sequence.layers[i];
                SequenceSnapshot::Layer layer;
                for (const DeviceBuffer* buffer : {&state.ssm, &state.conv, &state.tail,
                                                   &state.history}) {
                    layer.buffers.push_back(buffer->p != nullptr ? DeviceBuffer(buffer->bytes)
                                                                 : DeviceBuffer());
                }
                out.layers.push_back(std::move(layer));
            }
        }
        copy_state(const_cast<SequenceState&>(sequence), out, true);
        out.position = sequence.position;
        out.context  = sequence.context;
    }

    void restore(std::uint32_t s, const SequenceSnapshot& from) {
        if (from.layers.size() != layers.size()) {
            throw std::invalid_argument("qwen4_exp restore: the snapshot holds no state");
        }
        SequenceState& sequence = sequences.at(s);
        copy_state(sequence, const_cast<SequenceSnapshot&>(from), false);
        sequence.position = from.position;
        sequence.context  = from.context;
        // Snapshots hold text prompts only: a media prompt is not reused.
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
    }

    // Copies the recurrent state between a sequence and a snapshot on the layers' streams, and
    // waits for the copies: the paged KV and the indexer's pooled keys stay where they are, since
    // a sequence only writes positions at or past its own.
    void copy_state(SequenceState& sequence, SequenceSnapshot& snapshot, bool save) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            LayerState& state = sequence.layers[i];
            DeviceBuffer* live[] = {&state.ssm, &state.conv, &state.tail, &state.history};
            for (std::size_t k = 0; k < 4; ++k) {
                DeviceBuffer& copy = snapshot.layers[i].buffers[k];
                if (live[k]->p == nullptr || copy.bytes == 0) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(save ? copy.p : live[k]->p, save ? live[k]->p : copy.p,
                                           copy.bytes, cudaMemcpyDeviceToDevice,
                                           ranks[layers[i].rank].stream));
            }
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, std::move(options))) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept { return impl_->memory; }

void Executor::reset(std::uint32_t sequence) { impl_->reset(sequence); }

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->sequences.at(sequence).position;
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows) {
    impl_->forward(sequence, tokens, logit_rows);
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

void Executor::set_media(std::uint32_t sequence, std::span<const MediaItem> items,
                         std::vector<std::int32_t> rope_positions, std::int32_t rope_delta) {
    impl_->set_media(sequence, items, std::move(rope_positions), rope_delta);
}

bool Executor::vision() const noexcept { return impl_->vision_context != nullptr; }

Tensor Executor::logits(std::uint32_t rows) const {
    return Tensor(
        impl_->ranks[impl_->model.head_rank()].logits, DType::BF16,
        {static_cast<std::int32_t>(impl_->config.vocab_size), static_cast<std::int32_t>(rows)});
}

std::size_t Executor::head_rank() const noexcept { return impl_->model.head_rank(); }

ExpertCacheStats Executor::expert_cache_stats() const noexcept {
    if (impl_->stream) {
        const auto stream = impl_->stream->stats();
        return ExpertCacheStats{.routes       = stream.routes,
                                .hits         = stream.hits,
                                .admitted     = stream.routes - stream.hits,
                                .copied_bytes = stream.read_bytes,
                                .slots        = stream.slots};
    }
    return impl_->cache ? impl_->cache->stats() : ExpertCacheStats{};
}

cudaStream_t Executor::head_stream() const noexcept {
    return impl_->ranks[impl_->model.head_rank()].stream;
}

} // namespace ninfer::models::qwen4_exp
