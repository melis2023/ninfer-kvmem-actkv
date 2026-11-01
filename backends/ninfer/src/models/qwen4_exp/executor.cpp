#include "models/qwen4_exp/executor.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/gdn_replay_records.h"
#include "core/host_worker_pool.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/weight_view.h"
#include "ninfer/ops/target_logprobs.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/ngram_draft_prefetch.h"
#include "models/expert_profile.h"
#include "models/qwen4_exp/expert_misses.h"
#include "models/qwen4_exp/expert_stream.h"
#include "models/qwen4_exp/ngram_table.h"
#include "models/qwen3_5/execution/vision.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/kv_stream.h"
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
#include <future>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::int32_t kAlign = 256;
// Calls wider than this run their experts through the matrix kernel, from device memory.
constexpr std::int32_t kVectorTokens = 8;
// The family name of Flash-Next's built-in routing profiles (models/expert_profile.h).
constexpr std::string_view kExpertProfileFamily = "qwen3.8-flash-next";
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

Projection make_projection(const std::vector<ops::WeightInput>& inputs) {
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

Projection make_projection(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    return make_projection(inputs);
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
    Tensor norm;
    ops::HyperConnectionMatrix down, up, inject;

    [[nodiscard]] ops::HyperConnectionWeights weights() const { return {&norm, down, up, inject}; }
};

// A hyper-connection matrix as stored: ggml Q8_0 blocks, or BF16 words.
ops::HyperConnectionMatrix hc_matrix(const Model& model, WeightId id,
                                     std::initializer_list<std::int32_t> shape) {
    const Weight w = native_weight(model.weight(id).view);
    if (w.qtype == QType::GGUF_Q8_0) {
        return {w.qdata, ops::HyperConnectionMatrix::Format::Q8_0};
    }
    return {direct(model, id, shape).data, ops::HyperConnectionMatrix::Format::BF16};
}

HcPlan make_hc(const Model& model, const HyperConnectionWeights& w, const TextConfig& c) {
    const auto width = static_cast<std::int32_t>(c.hc_count * c.hidden_size);
    const auto low   = static_cast<std::int32_t>(c.hc_lowrank);
    HcPlan out;
    out.norm = direct(model, w.norm, {width});
    out.down = hc_matrix(model, w.down, {width, low});
    out.up   = hc_matrix(model, w.up, {low, width});
    if (w.inject) {
        out.inject = hc_matrix(model, *w.inject, {width, static_cast<std::int32_t>(c.hc_count)});
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

// Native operands retain the artifact's planes. The Program owns only these device tables.
struct NativeExpertTable {
    QType format = QType::BF16;
    bool integer_a8 = false;
    DeviceBuffer table;
    std::vector<Weight> operands;
    [[nodiscard]] ops::NativeExpertTable view() const {
        return {format, static_cast<const Weight*>(table.p), integer_a8};
    }
};

NativeExpertTable make_native_table(const Model& model, std::span<const WeightId> ids,
                                  std::int32_t rows, std::int32_t columns) {
    NativeExpertTable out;
    std::vector<Weight> operands;
    for (const WeightId id : ids) {
        const auto input = model.input(id);
        const auto format = native_weight(input.weight).qtype;
        const bool a8 = format != QType::BF16 && input.policy != ops::LinearPolicy::A16Only;
        Weight operand = ops::prepare_native_expert(input, rows, columns, a8);
        if (operands.empty()) {
            out.format = operand.qtype;
            out.integer_a8 = a8;
        } else if (operand.qtype != out.format || a8 != out.integer_a8) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": native expert bank has mixed formats or policies");
        }
        operands.push_back(operand);
    }
    out.table = DeviceBuffer(operands.size() * sizeof(Weight));
    out.table.copy_from_host(operands.data(), operands.size() * sizeof(Weight));
    out.operands = std::move(operands);
    return out;
}

struct NativeMoePlan {
    NativeExpertTable gate, up, down, shared_gate, shared_up, shared_down;
    std::int32_t experts = 0;
    [[nodiscard]] ops::NativeMoeWeights banks() const {
        return {gate.view(), up.view(), down.view(), shared_gate.view(), shared_up.view(),
                shared_down.view(), experts};
    }
};

struct MoePlan {
    Tensor router, shared_gate;
    ExpertTable gate, up, down, shared_gate_table, shared_up_table, shared_down_table;
    std::optional<NativeMoePlan> native;
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

// One plane of a sparse-attention layer's paged KV, [leading, 64, heads, pages]: device memory,
// or pinned host memory the device addresses directly when the layer streams.
struct KvPlane {
    DeviceBuffer device;
    std::unique_ptr<PinnedHostBuffer> host;
    void* p             = nullptr;
    std::uint64_t bytes = 0;
};

// One sequence's mutable state of one layer, on that layer's device.
struct LayerState {
    Tensor ssm, conv;          // Gated DeltaNet: the sequence's slot of the pool
    std::array<KvPlane, 4> kv; // sparse attention: K, V and a quantized KV's K and V scales
    DeviceBuffer pooled, tail; // the indexer's pooled keys and incomplete block
    // A streamed layer (ExecutorOptions::kv_resident): `kv` lives in host memory and the device
    // keeps a pool of 4-position pages of the same planes, with its residency map.
    std::array<DeviceBuffer, 4> pool;
    DeviceBuffer map_memory;
    ops::KvStreamMap map;
    DeviceBuffer history; // PLE
};

struct SequenceState {
    std::uint32_t slot     = 0; // its index, and its slot of every Gated DeltaNet state pool
    std::uint32_t position = 0;
    NgramContext context;
    std::vector<LayerState> layers;
    // The MTP block (an executor with drafts), on the head rank: its layer's state, the target's
    // stack at the sequence's last position (which the next cell pairs with the next token), and
    // whether the block follows the sequence's tokens, which a media prompt breaks.
    LayerState mtp;
    DeviceBuffer mtp_pending, mtp_scratch_tail;
    bool mtp_follows = true;
    // The last verify(): the n-gram context before it and the tokens it ran.
    NgramContext verify_context;
    std::vector<std::int32_t> verify_tokens;
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
    // The same for a verification of this sequence alone (a graph per segment) and for its draft
    // chain on one device with every expert there (one graph), captured at the second of each.
    struct VerifyGraphs {
        std::vector<DecodeGraphExecutable> segments;
        std::uint32_t runs = 0;
    };
    // The captured record strides and kernels depend on the actual verification width.
    std::array<VerifyGraphs, 17> verify;
    // Draft chains by step count (adaptive MTP drafts fewer than draft_tokens), and the single
    // steps by index of a chain a probability floor can end early.
    struct DraftGraph {
        DecodeGraphExecutable chain;
        std::uint32_t runs = 0;
    };
    std::array<DraftGraph, 16> draft;
    std::array<DraftGraph, 15> step;
};

// Consecutive layers on one rank, in pass order. The first segment also embeds the tokens, the
// segment on the head rank after the last layer also runs the final mixer and the head. A rank's
// first PLE layer starts a segment (`rows`) before which the pass uploads the n-gram rows, so the
// layers before it run while the rows are read. A segment holds at most kSegmentLayers layers: a
// captured segment is one CUDA Graph, and the device starts a graph only once its launch has
// submitted every node (about 1.8 us a kernel on a 4090: 2.9 ms for a 44-layer verification), so
// shorter graphs let the first run while the host launches the rest.
struct Segment {
    std::size_t rank  = 0;
    std::size_t begin = 0, end = 0; // layers
    bool embed = false, head = false, rows = false;
};

// One wait for the n-gram rows on the stream of the rank that reads them: `wanted` completes when
// the layers before the PLE layer are done, `ready` when the rows are uploaded, so the time between
// them is the time the stream stood still.
struct StallEvents {
    cudaEvent_t wanted = nullptr, ready = nullptr;
    bool pending = false;
};

// One sequence's tokens within a pass: columns [column, column + count) of the activation planes.
// A prefill pass has one part; a batched decode pass has one single-token part per sequence. The
// per-token work (embedding, hyper-connections, MoE, head) runs over every column at once; the
// mixers and the PLE injection, which read and write a sequence's state, run part by part.
struct Part {
    SequenceState* sequence = nullptr;
    std::int32_t column     = 0;
    std::int32_t count      = 0;
    // Its first position past the sequence's: a branch after the prompt part of its pass.
    std::uint32_t offset = 0;
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
    cudaEvent_t copied = nullptr; // its stack has been copied in from the previous rank
    cudaEvent_t routes = nullptr; // the last pass's routes reached the host
    ops::GgufMoeSide side;        // the decode MoE's shared expert runs there concurrently
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
    // The Gated DeltaNet state of the rank's layers, one slot per sequence (gdn_local maps a layer
    // to its pool layer).
    DeviceBuffer gdn_backing;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    DeviceBuffer slot_ids; // I32 [sequences] = 0, 1, ...: a sequence's pool slot on the device
    // Speculative verification (an executor with drafts): the replay records of the rank's Gated
    // DeltaNet layers and their fold, each sparse-attention layer's indexer projections (BF16
    // [640, columns] per layer), the PLE layer's normalised convolution input (FP32
    // [10240, columns]), and scratch for the states a verification reads but must not advance.
    DeviceBuffer record_backing;
    std::optional<GdnReplayRecords> records;
    std::optional<ops::GdnReplayFoldPlan> fold;
    DeviceBuffer index_records, ple_records;
    DeviceBuffer conv_scratch, tail_scratch;
    // Streamed sparse-attention layers: a prompt chunk reads one layer's whole KV from this
    // device copy (the four planes, every page), which holds `kv_staged`'s positions before
    // `kv_staged_end`; the layers of a span's attention phase keep it across the span's chunks.
    std::array<DeviceBuffer, 4> kv_staging;
    const LayerState* kv_staged = nullptr;
    std::uint32_t kv_staged_end = 0;
};

} // namespace

struct Executor::Impl {
    // The MTP block's operands on the head rank: its input norms and projections, its layer and
    // final mixer, and the head as its final mixer's output reads it.
    struct MtpPlan {
        Tensor embedding_norm, hidden_norm;
        Projection fc_embedding, fc_hidden, head;
        LayerPlan layer;
        HcPlan final_mixer;
    };

    // One sequence's cells of an MTP pass: `count` consecutive cells from position `first`, at
    // columns [column, column + count) of the pass.
    struct MtpPart {
        SequenceState* sequence = nullptr;
        std::int32_t column     = 0;
        std::int32_t count      = 0;
        std::uint32_t first     = 0;
    };

    // Tokens a target pass committed for one sequence: `count` of them from column `column` of
    // the pass, the first at `position`.
    struct Committed {
        SequenceState* sequence = nullptr;
        std::int32_t column     = 0;
        std::int32_t count      = 0;
        std::uint32_t position  = 0;
        std::span<const std::int32_t> tokens;
    };

    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    TextConfig config;
    NgramHashConstants ngram;
    std::unique_ptr<NgramTableReader> table;
    std::unique_ptr<NgramDraftPrefetch> draft_prefetch;
    // Native host banks use the CPU/GPU scheduler; GGUF banks retain their mapped-host route.
    std::vector<std::size_t> hybrid_layers;
    // The rank of the first PLE layer, whose staging buffer the rows are read into; whether this
    // pass's stage_rows() has waited for them; the waits the stall counters have yet to measure.
    std::size_t rows_rank = 0;
    bool rows_waited      = true;
    bool warming          = false; // warm_up() runs: the expert cache counts no routes
    std::array<StallEvents, 8> stall_events{};
    std::size_t stall_next = 0;
    std::uint64_t stalls   = 0;
    double stall_seconds   = 0.0;
    Weight embedding_table;
    Projection head;
    HcPlan final_mixer;
    std::vector<LayerPlan> layers;
    static constexpr std::size_t kSegmentLayers = 6;
    std::vector<Segment> segments;
    std::vector<RankState> ranks;
    std::vector<SequenceState> sequences;
    // A layer's index in its rank's Gated DeltaNet pool, and in its rank's indexer records.
    std::vector<std::uint32_t> gdn_local, qsa_local;
    // MTP speculative decoding (options.draft_tokens > 0). The MTP layer's index in the per-layer
    // route records, expert cache and expert stream is num_hidden_layers.
    std::optional<MtpPlan> mtp;
    // Maximum and current verification widths (drafts plus anchor), and the public vocabulary.
    std::uint32_t verify_width = 0;
    std::uint32_t active_verify_width = 0;
    std::int32_t domain        = 0;
    bool verifying             = false;
    std::vector<SequenceState*> verified; // the last verify()'s sequences, in order
    // While forward_branches() runs: the state its branches start from. Every part of the pass is
    // a branch of one sequence, and each part after the first takes the snapshot's recurrent state
    // of a layer before it runs that layer.
    const SequenceSnapshot* branch_origin = nullptr;
    // While forward_prompt_branches() runs: the snapshot the pass's first part, the prompt's last
    // tokens, leaves its state in, a layer's as soon as that part has run it; branch_origin points
    // to it, and the branches start from it.
    SequenceSnapshot* branch_capture = nullptr;
    // The MTP chain on the head rank: token ids and positions of a pass's cells (a draft step's at
    // step * sequences), staged from the host.
    DeviceBuffer mtp_ids, mtp_positions, ngram_prefetch_tokens;
    DeviceBuffer mtp_logprobs;
    std::unique_ptr<PinnedHostBuffer> mtp_staging;
    cudaEvent_t mtp_staged       = nullptr;
    std::uint32_t mtp_routes     = 0; // MTP route pairs recorded for the expert cache
    // While a draft chain is captured: its replays count no routes for the expert cache (the
    // catch-up passes after a commit still do).
    bool capturing_draft = false;
    std::uint32_t pages          = 0; // KV pages per sequence and attention layer
    PagedKVStorageLayout kv_layout;   // the planes of every sparse-attention layer's KV
    std::uint32_t pooled_slots   = 0; // indexer blocks per sequence and attention layer
    // Streamed sparse-attention layers (options.kv_resident): each sequence's layer keeps
    // stream_slots pages of 4 positions on the device, from which calls of at most stream_width
    // tokens read; wider calls read the rank's staging copy. Zero: every layer resident.
    std::uint32_t stream_slots = 0, stream_width = 0;
    std::uint32_t max_logit_rows = 0;
    ExecutorMemory memory;
    std::vector<std::uint64_t> row_ids;
    std::vector<std::uint8_t> row_bytes_host;
    // Expert cache (host-resident experts): every layer's routes of the last pass, on its device
    // and in pinned host memory, and how many tokens they cover.
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<HybridExperts> hybrid;
    const std::atomic<bool>* cancelled = nullptr;
    // Disk-resident experts: made resident before each layer's experts run.
    std::unique_ptr<ExpertStream> stream;
    // GGUF host or disk experts: a decode or verification call's missing experts, staged (or
    // computed by the CPU) while its cached ones run.
    std::unique_ptr<ExpertMisses> misses;
    // Each expert layer's routes of the current pass, [layer][10 * prefill_chunk] in one block per
    // rank (a stage's layers are consecutive), so one copy brings a rank's text layers home.
    std::vector<DeviceBuffer> route_blocks; // by rank
    std::vector<void*> route_records;       // by expert layer
    std::unique_ptr<PinnedHostBuffer> route_host;
    std::uint32_t pending_routes = 0;

    // Host-resident experts, wide calls: a layer's routed experts that the cache does not hold are
    // copied into device slots and the matrix kernel reads them there; one pool and one set of
    // tables per rank, whose layers run one at a time.
    // The pool holds a bank per projection laid out as the host's: expert e at e times the
    // layer's matrix bytes, so experts adjacent on the host cross the bus in one copy.
    struct SlotPool {
        struct Buffer {
            DeviceBuffer storage; // gate, up and down banks, zeroed when allocated
            DeviceBuffer tables;  // gate, up and down tables of every expert
        };

        std::array<std::uint64_t, 3> offset{}; // of each bank in a buffer's storage
        std::uint32_t slots = 0;               // experts a buffer holds: all of a layer's, or none
        // One buffer, or two that consecutive layers alternate between, so that a wide call's
        // copies of the next layer run while this layer's experts do.
        std::vector<Buffer> buffers;
    };

    std::vector<SlotPool> slot_pools; // by rank; empty unless the experts are host resident
    // With the automatic expert cache the pool buffers are cache slots between prompts
    // (ExpertCache::lend): a wide pass takes them back, the next decode pass returns them.
    bool pools_lent = false;
    std::unique_ptr<PinnedHostBuffer> slot_entries;
    // Wide host calls of at least kPrefetchTokens touch nearly every expert of a layer: every
    // expert the cache lacks is copied into the pool on the transfer stream while the layers
    // before run, without waiting for the layer's routes. Two pinned entry tables alternate.
    // A layer's copies are hundreds of operations, and an enqueue blocks while the stream's queue
    // is full, so a thread of their own (pool_copier) issues them: the layers between keep being
    // launched.
    static constexpr std::int32_t kPrefetchTokens = 256;
    struct PoolPrefetch {
        // Per pool buffer: the layer whose experts it holds or receives, its copies' completion
        // and the completion of the last call that read it, and the copier's task that issues
        // the copies.
        struct Buffer {
            std::size_t layer = ~std::size_t{0};
            cudaEvent_t ready = nullptr, free = nullptr;
            std::future<void> issued;
        };

        std::array<Buffer, 2> buffers;
        // Pinned table sources: a copy reads its buffer when it runs, so a buffer is rewritten only
        // once the copy that last read it is done (`copied`); the host runs ahead of the device.
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> entries;
        std::array<cudaEvent_t, 2> copied{};
        int next = 0;
    };
    std::vector<PoolPrefetch> prefetches; // by rank
    std::unique_ptr<HostWorkerPool> pool_copier;

    // Layer-major prompt spans (forward_span): the chunks' stacks, staged positions and n-gram
    // rows while every chunk of a span passes a layer before any passes the next. span_tokens is 0
    // when spans are off.
    static constexpr std::uint32_t kSpanChunks = 8;
    std::uint32_t span_tokens = 0;
    DeviceBuffer span_stack, span_positions, span_rope, span_rows;
    // A span's n-gram rows, read in one batch while its layers before the first PLE layer run.
    std::unique_ptr<PinnedHostBuffer> span_host_rows;
    std::vector<std::uint64_t> span_row_ids;
    // Each chunk's MoE input and hyper-connection injection between a layer's two phases.
    DeviceBuffer span_mixed, span_inject;
    std::size_t span_row_bytes = 0; // n-gram rows of one token
    // A span's expert phase runs up to kGroupTokens of consecutive chunks in one wide call, whose
    // matrix products then take wider column tiles: the group's routes and output. span_group is
    // the chunks a group takes, 1 when groups are off.
    static constexpr std::uint32_t kGroupTokens = 4096;
    std::uint32_t span_group = 1;
    DeviceBuffer span_ids, span_weights, span_shared, span_y;
    // A span's chunk before its last runs a layer's experts: the pool keeps that layer's experts.
    bool span_hold = false;

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
            table = std::make_unique<NgramTableReader>(options.ngram->layout, options.ngram_read);
        }
        max_logit_rows = std::min<std::uint32_t>(options.prefill_chunk, 512);
        if (!std::isfinite(options.draft_min_p) || options.draft_min_p < 0 ||
            options.draft_min_p > 1 || (options.draft_min_p != 0 && options.draft_tokens == 0)) {
            throw std::invalid_argument("qwen4_exp: draft_min_p needs MTP and a finite value in [0,1]");
        }
        if (options.draft_tokens > 0) {
            if (!model.mtp_weights()) {
                throw std::invalid_argument("qwen4_exp: drafts need a model loaded with its MTP "
                                            "block");
            }
            if (options.draft_tokens > 15) {
                throw std::invalid_argument("qwen4_exp: an MTP round drafts 1 to 15 tokens");
            }
            verify_width = options.draft_tokens + 1;
            if (verify_width * options.sequences > options.prefill_chunk) {
                throw std::invalid_argument(
                    "qwen4_exp: the prefill chunk must hold a verification of every sequence, (" +
                    std::to_string(verify_width) + " tokens) x " +
                    std::to_string(options.sequences));
            }
            max_logit_rows = std::max(max_logit_rows, verify_width * options.sequences);
            domain         = static_cast<std::int32_t>(model.resources().public_token_count);
        }
        pages          = (options.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        pooled_slots   = options.max_context / config.indexer_compress_ratio + 1;
        kv_layout =
            paged_kv_storage_layout(options.kv_cache, static_cast<std::int32_t>(config.head_dim));
        if (options.kv_resident != 0) {
            // A decode, verification or MTP call reads at most this many queries' blocks at once.
            const std::uint32_t width = std::max<std::uint32_t>(verify_width, 1);
            const std::uint32_t slots = std::max<std::uint32_t>(
                (options.kv_resident + ops::kKvStreamPageTokens - 1) / ops::kKvStreamPageTokens,
                std::uint32_t(ops::kv_stream_minimum_slots(std::int32_t(width))));
            if (std::uint64_t(slots) * ops::kKvStreamPageTokens < options.max_context) {
                stream_slots = slots;
                stream_width = width;
            }
        }
        memory.ranks.resize(device.size());
        plan_weights();
        if (verify_width > 0) { plan_mtp(); }
        const bool native_host = model.options().experts == ExpertResidency::Host &&
            std::any_of(layers.begin(), layers.end(), [](const auto& layer) { return layer.moe.native.has_value(); });
        if (native_host && options.sequences != 1) {
            throw std::invalid_argument("native host experts require concurrency 1");
        }
        if (!native_host && options.hybrid_experts.adaptive_cache) {
            throw std::invalid_argument("--expert-cache-adaptive requires native host experts");
        }
        if (!native_host && model.options().experts != ExpertResidency::Host &&
            (!options.hybrid_experts.routing_profile.empty() ||
             !options.hybrid_experts.record_profile.empty())) {
            throw std::invalid_argument("expert routing profiles require host experts");
        }
        const auto residency = model.options().experts;
        if (residency == ExpertResidency::Device &&
            (options.hybrid_experts.dma_share || options.hybrid_experts.cpu_threads != 0 ||
             options.hybrid_experts.mapped_misses)) {
            throw std::invalid_argument("expert DMA shares, CPU threads and miss paths require "
                                        "host or disk experts");
        }
        if (options.hybrid_experts.mapped_misses &&
            (residency != ExpertResidency::Host || native_host ||
             options.hybrid_experts.dma_share.value_or(1.0F) != 1.0F)) {
            throw std::invalid_argument("mapped misses need GGUF host experts and all misses on the GPU");
        }
        if (options.hybrid_experts.tail_skip_rank != 0 &&
            (residency != ExpertResidency::Host || native_host ||
             options.hybrid_experts.mapped_misses)) {
            throw std::invalid_argument(
                "--expert-tail-skip needs GGUF host experts with staged misses");
        }
        if (options.hybrid_experts.prefill_buffers != 0 &&
            (residency != ExpertResidency::Host || native_host)) {
            throw std::invalid_argument("--expert-prefill-buffers needs GGUF host experts");
        }
        plan_segments();
        allocate_ranks();
        allocate_vision();
        allocate_sequences();
        allocate_staging();
        if (verify_width > 0) { allocate_speculation(); }
        allocate_span();
        allocate_cache();
        memory.streamed_kv = stream_slots != 0;
        for (const auto& rank : memory.ranks) {
            memory.state_bytes += rank.state_bytes;
            memory.workspace_bytes += rank.workspace_bytes;
            memory.expert_cache_bytes += rank.expert_cache_bytes;
        }
    }

    ~Impl() {
        draft_prefetch.reset();
        // The copier's queued copies finish before the events and buffers they use go.
        for (auto& prefetch : prefetches) {
            for (auto& buffer : prefetch.buffers) {
                if (!buffer.issued.valid()) { continue; }
                try {
                    buffer.issued.get();
                } catch (...) {}
            }
        }
        pool_copier.reset();
        // A pass that failed between starting the row reads and waiting for them leaves them
        // writing into a staging buffer.
        if (table) {
            try {
                table->wait();
            } catch (...) {}
        }
        for (auto& sequence : sequences) {
            for (std::size_t i = 0; i < sequence.decode.size(); ++i) {
                RankBinding bind(device, segments[i].rank);
                sequence.decode[i].reset();
            }
            for (auto& width : sequence.verify) {
                for (std::size_t i = 0; i < width.segments.size(); ++i) {
                    RankBinding bind(device, segments[i].rank);
                    width.segments[i].reset();
                }
            }
            const auto reset = [&](auto& graphs) {
                for (auto& graph : graphs) {
                    if (graph.chain.ready()) {
                        RankBinding bind(device, model.head_rank());
                        graph.chain.reset();
                    }
                }
            };
            reset(sequence.draft);
            reset(sequence.step);
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            if (rank.staged != nullptr) { cudaEventDestroy(rank.staged); }
            if (rank.done != nullptr) { cudaEventDestroy(rank.done); }
            if (rank.copied != nullptr) { cudaEventDestroy(rank.copied); }
            if (rank.routes != nullptr) { cudaEventDestroy(rank.routes); }
            if (rank.side.fork != nullptr) { cudaEventDestroy(rank.side.fork); }
            if (rank.side.join != nullptr) { cudaEventDestroy(rank.side.join); }
            if (rank.side.stream != nullptr) { cudaStreamDestroy(rank.side.stream); }
        }
        if (mtp_staged != nullptr) {
            RankBinding bind(device, model.head_rank());
            cudaEventDestroy(mtp_staged);
        }
        for (const StallEvents& events : stall_events) {
            RankBinding bind(device, rows_rank);
            if (events.wanted != nullptr) { cudaEventDestroy(events.wanted); }
            if (events.ready != nullptr) { cudaEventDestroy(events.ready); }
        }
        for (std::size_t r = 0; r < prefetches.size(); ++r) {
            RankBinding bind(device, r);
            for (const auto& buffer : prefetches[r].buffers) {
                if (buffer.ready != nullptr) { cudaEventDestroy(buffer.ready); }
                if (buffer.free != nullptr) { cudaEventDestroy(buffer.free); }
            }
            for (cudaEvent_t copied : prefetches[r].copied) {
                if (copied != nullptr) { cudaEventDestroy(copied); }
            }
        }
    }

    void plan_weights() {
        const auto& w   = model.weights();
        embedding_table = native_weight(model.weight(w.token_embedding).view);
        {
            RankBinding bind(device, model.head_rank());
            head = make_projection(std::vector{model.input(w.output_head, "text/final_hidden")});
            final_mixer = make_hc(model, w.final_mixer, config);
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            layers.push_back(make_layer_plan(w.layers[i], model.stages().placement(i).stage,
                                             config.num_experts));
        }
    }

    void plan_mtp() {
        const MtpWeights& w = *model.mtp_weights();
        RankBinding bind(device, model.head_rank());
        const auto h = static_cast<std::int32_t>(config.hidden_size);
        MtpPlan plan;
        plan.embedding_norm = direct(model, w.embedding_norm, {h});
        plan.hidden_norm =
            direct(model, w.hidden_norm, {static_cast<std::int32_t>(config.hc_count) * h});
        plan.fc_embedding = make_projection(model, {w.fc_embedding});
        plan.fc_hidden    = make_projection(model, {w.fc_hidden});
        plan.head         = make_projection(
            std::vector{model.input(model.weights().output_head, "mtp/final_hidden")});
        plan.layer = make_layer_plan(w.layer, model.head_rank(), model.mtp_config()->num_experts);
        plan.final_mixer = make_hc(model, w.final_mixer, config);
        mtp              = std::move(plan);
    }

    // A layer's operands on its rank; `experts` its routed experts (the MTP block may keep more
    // than an expert-pruned text model).
    LayerPlan make_layer_plan(const LayerWeights& lw, std::size_t rank, std::uint32_t experts) {
        LayerPlan plan;
        plan.rank = rank;
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
            gdn.norm =
                direct(model, g->norm, {static_cast<std::int32_t>(config.linear_value_head_dim)});
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
        const auto h    = static_cast<std::int32_t>(config.hidden_size);
        plan.moe.router = direct(model, lw.moe.router, {h, static_cast<std::int32_t>(experts)});
        plan.moe.shared_gate = direct(model, lw.moe.shared_score, {h});
        if (!lw.moe.gate.empty() &&
            !is_gguf(native_weight(model.weight(lw.moe.gate.front()).view).qtype)) {
            const auto width = static_cast<std::int32_t>(config.moe_intermediate_size);
            const auto shared_width = static_cast<std::int32_t>(config.shared_expert_intermediate_size);
            NativeMoePlan native;
            native.gate = make_native_table(model, lw.moe.gate, width, h);
            native.up = make_native_table(model, lw.moe.up, width, h);
            native.down = make_native_table(model, lw.moe.down, h, width);
            native.shared_gate = make_native_table(model, std::span(&lw.moe.shared_gate, 1), shared_width, h);
            native.shared_up = make_native_table(model, std::span(&lw.moe.shared_up, 1), shared_width, h);
            native.shared_down = make_native_table(model, std::span(&lw.moe.shared_down, 1), h, shared_width);
            native.experts = static_cast<std::int32_t>(lw.moe.gate.size());
            plan.moe.native = std::move(native);
            return plan;
        }
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
        return plan;
    }

    void plan_segments() {
        std::vector<bool> ple_seen(device.size(), false);
        bool first_ple = true;
        Segment current{.rank = 0, .begin = 0, .end = 0, .embed = true};
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const std::size_t rank = layers[i].rank;
            const bool rows        = layers[i].ple.has_value() && !ple_seen[rank];
            if (rows) {
                ple_seen[rank] = true;
                if (std::exchange(first_ple, false)) { rows_rank = rank; }
            }
            if (rank != current.rank || rows) {
                if (current.end > current.begin || current.embed) { segments.push_back(current); }
                current = Segment{.rank = rank, .begin = i, .end = i, .rows = rows};
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
        // Launch-sized pieces; a boundary writes the stack before the next layer reads it, the
        // rounding the fused write-read takes (hyper_connection_write_read).
        std::vector<Segment> pieces;
        for (const Segment& segment : segments) {
            Segment piece = segment;
            while (piece.end - piece.begin > kSegmentLayers) {
                Segment first = piece;
                first.end     = piece.begin + kSegmentLayers;
                first.head    = false;
                pieces.push_back(first);
                piece.begin = first.end;
                piece.embed = false;
                piece.rows  = false;
            }
            pieces.push_back(piece);
        }
        segments = std::move(pieces);
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
        if (const std::uint32_t group = planned_group_chunks(); group > 1 && rank == 0) {
            bytes = std::max(bytes, ops::moe_experts_gguf_workspace_bytes(
                                        static_cast<std::int32_t>(group) * t));
        }
        std::vector<const LayerPlan*> planned;
        for (const auto& plan : layers) {
            if (plan.rank == rank) { planned.push_back(&plan); }
        }
        if (mtp && rank == model.head_rank()) {
            planned.push_back(&mtp->layer);
            bytes = std::max(
                {bytes,
                 ops::moe_route_workspace_bytes(
                     t, static_cast<std::int32_t>(model.mtp_config()->num_experts)),
                 projection_workspace(mtp->fc_embedding, t),
                 projection_workspace(mtp->fc_hidden,
                                      t * static_cast<std::int32_t>(config.hc_count)),
                 projection_workspace(mtp->head, static_cast<std::int32_t>(max_logit_rows)),
                 // The MTP block's calls over cached experts only: their kept ids and weights.
                 ops::moe_experts_gguf_workspace_bytes(t) +
                     2 * ((std::size_t(config.num_experts_per_tok) * t * 4 + 255) / 256 * 256)});
        }
        for (const LayerPlan* layer : planned) {
            const LayerPlan& plan = *layer;
            if (plan.moe.native) {
                bytes = std::max(bytes, ops::moe_experts_native_workspace_bytes(t));
            }
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
            std::vector<std::int32_t> identity(std::max<std::uint32_t>(pages, options.sequences));
            std::iota(identity.begin(), identity.end(), 0);
            rank.block_table = DeviceBuffer(std::size_t(pages) * 4);
            rank.block_table.copy_from_host(identity.data(), std::size_t(pages) * 4);
            rank.slot_ids = DeviceBuffer(std::size_t(options.sequences) * 4);
            rank.slot_ids.copy_from_host(identity.data(), std::size_t(options.sequences) * 4);
            // ids, positions, RoPE positions, scatter columns, three-axis RoPE positions, rows.
            rank.staging = std::make_unique<PinnedHostBuffer>(
                round_up(t * 28) + round_up(std::uint64_t(config.ngram_heads()) * t * row_b));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.copied, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.routes, cudaEventDisableTiming));
            CUDA_CHECK(cudaStreamCreateWithFlags(&rank.side.stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.side.fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.side.join, cudaEventDisableTiming));
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
        // The MTP block drafts from text cells only; the sequence stops drafting until a reset.
        sequence.mtp_follows = false;
        CUDA_CHECK(cudaStreamSynchronize(ranks[0].stream));
    }

    // The planes of the sparse-attention layers' KV storage, in LayerState::kv's order (K, V,
    // K scales, V scales), each a row width and dtype; absent scale planes have width 0.
    [[nodiscard]] std::array<std::pair<std::int32_t, DType>, 4> kv_planes() const {
        return {{{kv_layout.key.data_leading_extent, kv_layout.key.data_dtype},
                 {kv_layout.value.data_leading_extent, kv_layout.value.data_dtype},
                 {kv_layout.key.has_scale() ? kv_layout.key.scale_leading_extent : 0,
                  kv_layout.key.scale_dtype},
                 {kv_layout.value.has_scale() ? kv_layout.value.scale_leading_extent : 0,
                  kv_layout.value.scale_dtype}}};
    }

    // One plane's bytes for `pages` pages of `page_tokens` positions.
    [[nodiscard]] std::uint64_t kv_plane_bytes(std::size_t plane, std::uint64_t page_tokens,
                                               std::uint64_t count) const {
        const auto [leading, dtype] = kv_planes()[plane];
        return std::uint64_t(leading) * page_tokens * config.num_key_value_heads * count *
               dtype_size(dtype);
    }

    // A view of a sparse-attention layer's KV planes `data(i)`: pages of `page_tokens` positions,
    // `count` of them, read through `table` (`entries` I32).
    template <class Data>
    [[nodiscard]] PagedKVLayerView kv_view(Data&& data, std::int32_t page_tokens,
                                           std::int32_t count, void* table,
                                           std::int32_t entries) const {
        PagedKVLayerView cache{};
        Tensor* targets[4] = {&cache.k_pages, &cache.v_pages, &cache.k_scale_pages,
                              &cache.v_scale_pages};
        const auto planes = kv_planes();
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            *targets[i] = Tensor(data(i), planes[i].second,
                                 {planes[i].first, page_tokens,
                                  static_cast<std::int32_t>(config.num_key_value_heads), count});
        }
        cache.block_table  = Tensor(table, DType::I32, {entries});
        cache.head_dim     = static_cast<std::int32_t>(config.head_dim);
        cache.num_kv_heads = static_cast<std::int32_t>(config.num_key_value_heads);
        cache.storage      = options.kv_cache;
        return cache;
    }

    // A sparse-attention layer's per-sequence state: paged KV (one plane per K or V vector and
    // per scale, each [leading, page, head, page]), pooled keys and the indexer's tail. A
    // streamed layer keeps the paged KV in pinned host memory, and its pool and map on the device.
    void allocate_attention_state(LayerState& state, std::size_t rank) {
        std::uint64_t device_kv = 0, bytes = 0;
        const auto planes       = kv_planes();
        for (std::size_t i = 0; i < planes.size(); ++i) {
            if (planes[i].first == 0) { continue; }
            KvPlane& plane = state.kv[i];
            plane.bytes    = kv_plane_bytes(i, kPagedKVPageSize, pages);
            if (stream_slots != 0) {
                plane.host    = std::make_unique<PinnedHostBuffer>(plane.bytes);
                plane.p       = plane.host->data();
                state.pool[i] = DeviceBuffer(kv_plane_bytes(i, ops::kKvStreamPageTokens,
                                                            stream_slots));
                device_kv += state.pool[i].bytes;
                memory.host_kv_bytes += plane.bytes;
            } else {
                plane.device = DeviceBuffer(plane.bytes);
                plane.p      = plane.device.p;
                device_kv += plane.bytes;
            }
        }
        if (stream_slots != 0) {
            const auto blocks = static_cast<std::int32_t>(
                std::uint64_t(pages) * (kPagedKVPageSize / ops::kKvStreamPageTokens));
            const auto slots  = static_cast<std::int32_t>(stream_slots);
            state.map_memory  = DeviceBuffer(ops::kv_stream_map_bytes(blocks, slots));
            state.map         = ops::kv_stream_map(state.map_memory.p, blocks, slots);
            bytes += state.map_memory.bytes;
        }
        state.pooled = DeviceBuffer(std::uint64_t(config.indexer_head_dim) * pooled_slots * 4);
        state.tail   = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                    (config.indexer_compress_ratio - 1) * 4);
        memory.ranks[rank].state_bytes += device_kv + bytes + state.pooled.bytes + state.tail.bytes;
        memory.kv_bytes += device_kv;
    }

    // Streamed layers: each rank that runs one gets the staging copy a prompt chunk reads.
    void allocate_staging() {
        if (stream_slots == 0) { return; }
        for (RankState& rank : ranks) {
            const bool attention =
                std::any_of(layers.begin(), layers.end(), [&](const LayerPlan& plan) {
                    return plan.qsa && plan.rank == rank.rank;
                }) ||
                (mtp && rank.rank == model.head_rank());
            if (!attention) { continue; }
            RankBinding bind(device, rank.rank);
            const auto planes = kv_planes();
            for (std::size_t i = 0; i < planes.size(); ++i) {
                if (planes[i].first == 0) { continue; }
                rank.kv_staging[i] = DeviceBuffer(kv_plane_bytes(i, kPagedKVPageSize, pages));
                memory.ranks[rank.rank].workspace_bytes += rank.kv_staging[i].bytes;
            }
        }
    }

    // Before a prompt chunk of a streamed layer: the rank's staging copy holds the layer's
    // positions before `first`, copied from its host copy unless the copy is still there from the
    // layer's previous chunk.
    void stage(const LayerState& state, RankState& rank, std::uint32_t first) {
        if (rank.kv_staged == &state && rank.kv_staged_end >= first) { return; }
        const std::uint64_t used = (std::uint64_t(first) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        for (std::size_t i = 0; i < state.kv.size(); ++i) {
            const KvPlane& plane = state.kv[i];
            if (plane.p == nullptr || used == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(rank.kv_staging[i].p, plane.p, plane.bytes / pages * used,
                                       cudaMemcpyDefault, rank.stream));
        }
        rank.kv_staged = &state;
    }

    // Each rank's Gated DeltaNet layers keep their state in one pool with a slot per sequence,
    // the layout the replay fold of a speculative commit reads.
    void allocate_gdn_pools() {
        gdn_local.assign(layers.size(), 0);
        std::vector<std::uint32_t> count(ranks.size(), 0);
        std::int32_t channels = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (!layers[i].gdn) { continue; }
            gdn_local[i] = count[layers[i].rank]++;
            channels     = layers[i].gdn->qkv.rows;
        }
        for (RankState& rank : ranks) {
            if (count[rank.rank] == 0) { continue; }
            RankBinding bind(device, rank.rank);
            const LinearAttentionStatePoolSpec spec{
                .layers         = count[rank.rank],
                .conv_channels  = channels,
                .conv_width     = static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1),
                .value_heads    = static_cast<std::int32_t>(config.linear_num_value_heads),
                .value_head_dim = static_cast<std::int32_t>(config.linear_value_head_dim),
                .key_head_dim   = static_cast<std::int32_t>(config.linear_key_head_dim),
                .slot_count     = static_cast<std::int32_t>(options.sequences),
            };
            LayoutBuilder builder;
            const auto layout = plan_linear_attention_state_pool(builder, spec);
            rank.gdn_backing  = DeviceBuffer(builder.finish(256, "Gated DeltaNet state"));
            rank.gdn          = std::make_unique<LinearAttentionStatePool>(
                DeviceSpan{rank.gdn_backing.p, rank.gdn_backing.bytes}, layout);
            memory.ranks[rank.rank].state_bytes += rank.gdn_backing.bytes;
        }
    }

    void allocate_sequences() {
        const std::uint64_t width = std::uint64_t(config.hc_count) * config.hidden_size;
        allocate_gdn_pools();
        for (std::uint32_t s = 0; s < options.sequences; ++s) {
            SequenceState sequence;
            sequence.slot = s;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const LayerPlan& plan = layers[i];
                RankBinding bind(device, plan.rank);
                LayerState state;
                if (plan.gdn) {
                    const auto slot = static_cast<std::int32_t>(s);
                    state.ssm       = ranks[plan.rank].gdn->recurrent_slot(gdn_local[i], slot);
                    state.conv      = ranks[plan.rank].gdn->conv_slot(gdn_local[i], slot);
                }
                if (plan.qsa) { allocate_attention_state(state, plan.rank); }
                if (plan.ple) {
                    state.history = DeviceBuffer(width * (config.ple_conv_kernel_size - 1) *
                                                 config.ngram.ngram_size * 4);
                    memory.ranks[plan.rank].state_bytes += state.history.bytes;
                }
                sequence.layers.push_back(std::move(state));
            }
            if (mtp) {
                const std::size_t head = model.head_rank();
                RankBinding bind(device, head);
                allocate_attention_state(sequence.mtp, head);
                sequence.mtp_pending = DeviceBuffer(width * 4);
                memory.ranks[head].state_bytes += sequence.mtp_pending.bytes;
            }
            sequences.push_back(std::move(sequence));
            reset(s);
        }
    }

    // What a speculative round needs beyond the sequences' state: on every rank the replay records
    // of its Gated DeltaNet layers with their fold, its sparse-attention layers' indexer
    // projections, the PLE layer's normalised convolution input and scratch states; on the head
    // rank the MTP chain's tokens and positions.
    void allocate_speculation() {
        const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
        const std::uint64_t index_rows =
            std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
        const std::uint64_t stack_width = std::uint64_t(config.hc_count) * config.hidden_size;
        qsa_local.assign(layers.size(), 0);
        std::vector<std::uint32_t> attention(ranks.size(), 0);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].qsa) { qsa_local[i] = attention[layers[i].rank]++; }
        }
        for (RankState& rank : ranks) {
            RankBinding bind(device, rank.rank);
            std::uint64_t bytes = 0;
            if (rank.gdn) {
                const auto& pool = rank.gdn->spec();
                const GdnReplayRecordSpec spec{
                    .layers          = static_cast<std::int32_t>(pool.layers),
                    .record_capacity = static_cast<std::int32_t>(options.sequences),
                    .width           = static_cast<std::int32_t>(verify_width),
                    .conv_channels   = pool.conv_channels,
                    .qk_heads        = static_cast<std::int32_t>(config.linear_num_key_heads),
                    .value_heads     = pool.value_heads,
                    .key_dim         = pool.key_head_dim,
                    .value_dim       = pool.value_head_dim,
                };
                LayoutBuilder builder;
                const auto layout   = plan_gdn_replay_records(builder, spec);
                rank.record_backing = DeviceBuffer(builder.finish(256, "GDN replay records"));
                rank.records.emplace(DeviceSpan{rank.record_backing.p, rank.record_backing.bytes},
                                     layout);
                rank.fold.emplace(*rank.records, rank.gdn->all_layers_view());
                rank.conv_scratch = DeviceBuffer(std::uint64_t(pool.conv_channels) *
                                                 (config.linear_conv_kernel_dim - 1) * 2);
                bytes += rank.record_backing.bytes + rank.conv_scratch.bytes;
            }
            if (attention[rank.rank] > 0) {
                rank.index_records = DeviceBuffer(attention[rank.rank] * index_rows * columns * 2);
                rank.tail_scratch  = DeviceBuffer(std::uint64_t(config.indexer_head_dim) *
                                                  (config.indexer_compress_ratio - 1) * 4);
                bytes += rank.index_records.bytes + rank.tail_scratch.bytes;
            }
            const bool ple_here =
                std::any_of(layers.begin(), layers.end(),
                            [&](const LayerPlan& p) { return p.ple && p.rank == rank.rank; });
            if (ple_here) {
                rank.ple_records = DeviceBuffer(stack_width * columns * 4);
                bytes += rank.ple_records.bytes;
            }
            memory.ranks[rank.rank].workspace_bytes += bytes;
        }
        const std::size_t head = model.head_rank();
        RankBinding bind(device, head);
        const std::uint64_t cells = std::max<std::uint64_t>(
            options.prefill_chunk, std::uint64_t(verify_width) * options.sequences);
        mtp_ids       = DeviceBuffer(cells * 4);
        mtp_positions = DeviceBuffer(cells * 4);
        mtp_staging   = std::make_unique<PinnedHostBuffer>(cells * 8);
        if (options.draft_min_p > 0) {
            mtp_logprobs = DeviceBuffer(std::uint64_t(options.draft_tokens) * options.sequences * 4);
        }
        if (table && options.ngram_read.residency != NgramResidency::Ram) {
            draft_prefetch = std::make_unique<NgramDraftPrefetch>(
                device.rank(head), *table, ngram, config.eos_token_id, config.vocab_size,
                options.sequences, options.draft_tokens);
            ngram_prefetch_tokens = DeviceBuffer(2 * std::uint64_t(options.sequences) *
                                                  verify_width * sizeof(std::int32_t));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&mtp_staged, cudaEventDisableTiming));
        // Each sequence's MTP tail as the draft steps advance it (a catch-up redoes those cells).
        for (SequenceState& sequence : sequences) {
            sequence.mtp_scratch_tail = DeviceBuffer(sequence.mtp.tail.bytes);
        }
        memory.ranks[head].workspace_bytes += mtp_ids.bytes + mtp_positions.bytes + mtp_logprobs.bytes + ngram_prefetch_tokens.bytes +
                                              sequences.size() * sequences.front().mtp.tail.bytes;
    }

    // The layers whose experts the route records, the expert cache and the expert stream serve:
    // the text layers, then the MTP block's.
    [[nodiscard]] std::vector<std::pair<const LayerPlan*, const MoeWeights*>>
    expert_layers() const {
        std::vector<std::pair<const LayerPlan*, const MoeWeights*>> out;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            out.emplace_back(&layers[i], &model.weights().layers[i].moe);
        }
        if (mtp) { out.emplace_back(&mtp->layer, &model.mtp_weights()->layer.moe); }
        return out;
    }

    void allocate_cache() {
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        const auto expert = expert_layers();
        std::vector<std::size_t> on_rank(device.size(), 0);
        for (const auto& [plan, moe] : expert) { ++on_rank[plan->rank]; }
        route_blocks.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            if (on_rank[r] == 0) { continue; }
            RankBinding bind(device, r);
            route_blocks[r] = DeviceBuffer(on_rank[r] * pairs * sizeof(std::int32_t));
        }
        std::fill(on_rank.begin(), on_rank.end(), 0);
        for (const auto& [plan, moe] : expert) {
            route_records.push_back(static_cast<std::byte*>(route_blocks[plan->rank].p) +
                                    on_rank[plan->rank]++ * pairs * sizeof(std::int32_t));
        }
        route_host =
            std::make_unique<PinnedHostBuffer>(expert.size() * pairs * sizeof(std::int32_t));
        const ExpertResidency residency = model.options().experts;
        const bool native_host = residency == ExpertResidency::Host &&
            std::any_of(expert.begin(), expert.end(), [](const auto& item) { return item.first->moe.native.has_value(); });
        if (native_host) {
            std::vector<HybridExpertLayer> host_layers;
            hybrid_layers.resize(expert.size());
            for (std::size_t index = 0; index < expert.size(); ++index) {
                const auto& [plan, moe] = expert[index];
                if (!plan->moe.native) { continue; }
                hybrid_layers[index] = host_layers.size();
                const auto& native = *plan->moe.native;
                HybridExpertLayer layer;
                layer.rank = plan->rank;
#if defined(__linux__)
                layer.registered = true; // native Host bindings use Residency::Registered
#endif
                for (std::int32_t e = 0; e < native.experts; ++e) {
                    layer.experts.push_back({{native.gate.operands[e], native.gate.integer_a8},
                                             {native.up.operands[e], native.up.integer_a8},
                                             {native.down.operands[e], native.down.integer_a8}});
                }
                host_layers.push_back(std::move(layer));
            }
            if (!options.hybrid_experts.routing_profile.empty()) {
                std::vector<std::size_t> widths;
                for (const auto& layer : host_layers) { widths.push_back(layer.experts.size()); }
                auto counts = read_expert_profile(options.hybrid_experts.routing_profile,
                                                  model.info().artifact_id, widths);
                for (std::size_t layer = 0; layer < host_layers.size(); ++layer) {
                    host_layers[layer].route_counts = std::move(counts[layer]);
                }
            }
            hybrid = std::make_unique<HybridExperts>(device, std::move(host_layers),
                options.prefill_chunk, options.expert_cache_bytes, options.hybrid_experts);
            for (std::size_t r = 0; r < ranks.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += hybrid->cache_bytes(r);
                memory.ranks[r].workspace_bytes += hybrid->working_bytes(r);
            }
            return;
        }
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
        // Disk experts keep their per-layer residency until staged experts can enter its slots.
        const bool staged = residency == ExpertResidency::Host && !options.hybrid_experts.mapped_misses;
        if (staged) {
            for (auto& b : bytes) { b = b > kStagingBytes ? b - kStagingBytes : 0; }
        }
        if (residency == ExpertResidency::Disk) {
            if (options.expert_cache_bytes == 0) {
                throw std::invalid_argument("disk-resident experts need a device expert cache");
            }
            std::vector<ExpertStreamLayer> stream_layers;
            for (const auto& [plan, moe] : expert) {
                stream_layers.push_back(ExpertStreamLayer{
                    .rank    = plan->rank,
                    .stream  = ranks[plan->rank].stream,
                    .experts = {moe->located_gate, moe->located_up, moe->located_down},
                    .tables  = {plan->moe.gate.table.p, plan->moe.up.table.p,
                                plan->moe.down.table.p}});
            }
            stream = std::make_unique<ExpertStream>(device, model.files(), std::move(stream_layers),
                                                    bytes);
            for (std::size_t r = 0; r < bytes.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += bytes[r];
            }
            return;
        }
        std::vector<ExpertCacheLayer> cache_layers;
        for (const auto& [plan, moe] : expert) {
            const auto bank = [](const ExpertTable& table) {
                return ExpertBank{.expert_bytes = std::int64_t(table.rows) * table.row_bytes,
                                  .host         = table.pointers,
                                  .table        = table.table.p};
            };
            cache_layers.push_back(ExpertCacheLayer{.rank   = plan->rank,
                                                    .stream = ranks[plan->rank].stream,
                                                    .gate   = bank(plan->moe.gate),
                                                    .up     = bank(plan->moe.up),
                                                    .down   = bank(plan->moe.down)});
        }
        // The routing profile: the one given, else the one recorded on this family's release.
        std::vector<std::size_t> widths;
        for (const auto& [plan, moe] : expert) { widths.push_back(plan->moe.gate.pointers.size()); }
        std::optional<ExpertRouteCounts> profile;
        if (!options.hybrid_experts.routing_profile.empty()) {
            profile = read_expert_profile(options.hybrid_experts.routing_profile,
                                          model.info().artifact_id, widths);
        } else {
            profile = builtin_expert_profile(kExpertProfileFamily, widths);
        }
        cache = std::make_unique<ExpertCache>(device, std::move(cache_layers), bytes,
                                              profile ? &*profile : nullptr,
                                              expert.size() > layers.size());
        for (std::size_t r = 0; r < bytes.size(); ++r) {
            memory.ranks[r].expert_cache_bytes += bytes[r];
        }
        if (profile) { cache->seed(*profile); }
        if (staged) { allocate_misses(); }
    }

    // Device staging of the missing experts, taken from the expert cache's share of each device.
    static constexpr std::uint64_t kStagingBytes = 128ULL << 20;

    void allocate_misses() {
        std::vector<MissLayer> out;
        const auto expert = expert_layers();
        for (std::size_t i = 0; i < expert.size(); ++i) {
            const auto& [plan, moe] = expert[i];
            MissLayer layer;
            layer.rank    = plan->rank;
            layer.experts = plan->moe.gate.pointers.size();
            const ExpertTable* tables[3] = {&plan->moe.gate, &plan->moe.up, &plan->moe.down};
            const std::span<const ExpertLocation> located[3] = {moe->located_gate, moe->located_up,
                                                                moe->located_down};
            for (int k = 0; k < 3; ++k) {
                auto& p        = layer.projections[std::size_t(k)];
                p.format       = tables[k]->format;
                p.row_bytes    = tables[k]->row_bytes;
                p.rows         = tables[k]->rows;
                p.k            = std::int32_t(k == 2 ? config.moe_intermediate_size
                                                     : config.hidden_size);
                p.expert_bytes = std::int64_t(p.rows) * p.row_bytes;
                if (stream) {
                    p.located = located[k];
                } else {
                    p.host = tables[k]->pointers;
                }
            }
            layer.gate_table = static_cast<const void* const*>(plan->moe.gate.table.p);
            // The text layers' slots follow their misses; the MTP block's keep the cache's own
            // policy.
            if (cache && !stream && i < layers.size()) {
                cache->hand_over(i);
                layer.admit = true;
            }
            if (cache) {
                layer.resident = cache->storage(i);
            } else if (stream) {
                layer.resident[0] = stream->storage(plan->rank);
            }
            out.push_back(std::move(layer));
        }
        MissOptions miss;
        miss.token_capacity = kVectorTokens;
        miss.cpu_balance    = !options.hybrid_experts.dma_share && !options.hybrid_experts.mapped_misses;
        miss.cpu_share      = miss.cpu_balance ? 0.0 : 1.0 - double(options.hybrid_experts.dma_share.value_or(1.0F));
        miss.cpu_threads    = options.hybrid_experts.cpu_threads;
        miss.staging_bytes  = kStagingBytes;
        miss.tail_skip_rank = options.hybrid_experts.tail_skip_rank;
        misses              = std::make_unique<ExpertMisses>(
            device, std::move(out),
            stream ? model.files() : std::vector<std::filesystem::path>{}, miss, cache.get());
        for (auto& rank : memory.ranks) { rank.workspace_bytes += kStagingBytes; }
    }

    // One slot per expert on every rank, sized for its layers' widest projections, when half the
    // free memory holds them, and a second such buffer when half holds both and two are wanted:
    // --expert-prefill-buffers 2, or by default with the automatic expert cache, which holds both
    // between prompts. The expert cache takes what is left.
    void allocate_slot_pools() {
        const std::size_t experts = config.num_experts;
        const std::uint32_t wanted =
            options.hybrid_experts.prefill_buffers != 0 ? options.hybrid_experts.prefill_buffers
            : options.expert_cache_bytes == ExecutorOptions::kAutoExpertCache ? 2
                                                                               : 1;
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
            SlotPool& pool = slot_pools[r];
            pool.offset    = {0, round_up(experts * widest[0]),
                              round_up(experts * widest[0]) + round_up(experts * widest[1])};
            RankBinding bind(device, r);
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            // Zeros follow the last down in its bank: the matrix kernel reads past a down.
            const std::uint64_t bytes = pool.offset[2] + round_up(experts * widest[2] + kSlotTail);
            if (bytes > free_bytes / 2) { continue; }
            pool.slots = static_cast<std::uint32_t>(experts);
            pool.buffers.resize(wanted == 2 && 2 * bytes <= free_bytes / 2 ? 2 : 1);
            for (auto& buffer : pool.buffers) {
                buffer.storage = DeviceBuffer(bytes);
                CUDA_CHECK(cudaMemset(buffer.storage.p, 0, buffer.storage.bytes));
                buffer.tables = DeviceBuffer(3 * experts * sizeof(void*));
                memory.ranks[r].workspace_bytes += buffer.storage.bytes + buffer.tables.bytes;
            }
        }
        slot_entries = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
        prefetches.resize(device.size());
        for (std::size_t r = 0; r < device.size(); ++r) {
            if (slot_pools[r].slots == 0) { continue; }
            RankBinding bind(device, r);
            auto& prefetch = prefetches[r];
            for (std::size_t b = 0; b < slot_pools[r].buffers.size(); ++b) {
                CUDA_CHECK(
                    cudaEventCreateWithFlags(&prefetch.buffers[b].ready, cudaEventDisableTiming));
                CUDA_CHECK(
                    cudaEventCreateWithFlags(&prefetch.buffers[b].free, cudaEventDisableTiming));
            }
            for (auto& entries : prefetch.entries) {
                entries = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
            }
            for (cudaEvent_t& copied : prefetch.copied) {
                CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(copied, device.rank(r).transfer_stream));
            }
            if (!pool_copier) { pool_copier = std::make_unique<HostWorkerPool>(1, 4); }
        }
    }

    // Copies the experts of text layer `index` that `wanted` selects and the cache lacks into the
    // pool's banks, each at its own index, and points `entries` (the gate, up and down tables) at
    // the cache or the copies; unselected experts get none. A run of such experts goes in one copy
    // per projection wherever the host's are adjacent, and zeros follow the run's last down, which
    // the matrix kernel reads past: inside a run the bytes after a down are the next expert's
    // down, as in a host bank.
    template <class Wanted>
    void stage_pool(const SlotPool& pool, const SlotPool::Buffer& buffer, std::size_t index,
                    Wanted&& wanted, const void** entries, cudaStream_t stream) {
        const MoePlan& m             = layers[index].moe;
        const std::size_t experts    = m.gate.pointers.size();
        const ExpertTable* tables[3] = {&m.gate, &m.up, &m.down};
        std::array<std::uint64_t, 3> bytes{};
        for (int k = 0; k < 3; ++k) {
            bytes[k] = std::uint64_t(tables[k]->rows) * tables[k]->row_bytes;
        }
        auto* storage     = static_cast<std::byte*>(buffer.storage.p);
        const auto cached = [&](std::size_t e) {
            return cache && cache->cached(index, 0, std::int32_t(e)) != nullptr;
        };
        std::size_t e = 0;
        while (e < experts) {
            if (!wanted(e) || cached(e)) {
                for (int k = 0; k < 3; ++k) {
                    entries[k * experts + e] =
                        wanted(e) ? cache->cached(index, k, std::int32_t(e)) : nullptr;
                }
                ++e;
                continue;
            }
            std::size_t end = e + 1;
            while (end < experts && wanted(end) && !cached(end)) { ++end; }
            for (int k = 0; k < 3; ++k) {
                std::byte* bank = storage + pool.offset[k];
                for (std::size_t a = e; a < end;) {
                    std::size_t b = a + 1;
                    while (b < end && static_cast<const std::byte*>(tables[k]->pointers[b - 1]) +
                                              bytes[k] ==
                                          tables[k]->pointers[b]) {
                        ++b;
                    }
                    CUDA_CHECK(cudaMemcpyAsync(bank + a * bytes[k], tables[k]->pointers[a],
                                               std::size_t((b - a) * bytes[k]),
                                               cudaMemcpyHostToDevice, stream));
                    a = b;
                }
                for (std::size_t x = e; x < end; ++x) {
                    entries[k * experts + x] = bank + x * bytes[k];
                }
            }
            CUDA_CHECK(cudaMemsetAsync(storage + pool.offset[2] + end * bytes[2], 0, kSlotTail,
                                       stream));
            e = end;
        }
    }

    // The pool buffer that text layer `index`'s wide calls read: with two, consecutive layers
    // alternate.
    [[nodiscard]] std::size_t pool_buffer(std::size_t index) const {
        return slot_pools[layers[index].rank].buffers.size() == 2 ? index % 2 : 0;
    }

    // Whether text layer `index` exists on `rank` and runs its wide calls through the pool.
    [[nodiscard]] bool pooled_layer(std::size_t index, std::size_t rank) const {
        return index < layers.size() && layers[index].rank == rank && !layers[index].moe.native;
    }

    // Has the copier copy every expert of text layer `index` that the cache lacks into its pool
    // buffer (issue_prefetch), which then receives the layer.
    void prefetch_layer(std::size_t index) {
        auto& target  = prefetches[layers[index].rank].buffers[pool_buffer(index)];
        target.layer  = index;
        target.issued = pool_copier->submit([this, index] { issue_prefetch(index); });
    }

    // Before a call waits for a buffer's ready event: the copier has recorded it. Rethrows the
    // copier's failure.
    static void await_prefetch(PoolPrefetch::Buffer& buffer) {
        if (buffer.issued.valid()) { buffer.issued.get(); }
    }

    // Waits until the copier has issued every queued layer.
    void drain_prefetches() {
        for (auto& prefetch : prefetches) {
            for (auto& buffer : prefetch.buffers) { await_prefetch(buffer); }
        }
    }

    // On the copier thread: once the last call that read layer `index`'s pool buffer is done with
    // it, copies every expert of the layer that the cache lacks into the buffer on the transfer
    // stream and points the buffer's tables at the cache or the copies. Waiting on the host keeps
    // copies that cannot run yet out of the stream's queue.
    void issue_prefetch(std::size_t index) {
        const LayerPlan& plan = layers[index];
        SlotPool& pool        = slot_pools[plan.rank];
        PoolPrefetch& state   = prefetches[plan.rank];
        const std::size_t b   = pool_buffer(index);
        auto& target          = state.buffers[b];
        RankBinding bind(device, plan.rank);
        const cudaStream_t t  = device.rank(plan.rank).transfer_stream;
        CUDA_CHECK(cudaEventSynchronize(target.free));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        const auto source         = std::size_t(state.next);
        state.next ^= 1;
        CUDA_CHECK(cudaEventSynchronize(state.copied[source]));
        auto* entries = static_cast<const void**>(state.entries[source]->data());
        stage_pool(pool, pool.buffers[b], index, [](std::size_t) { return true; }, entries, t);
        CUDA_CHECK(cudaMemcpyAsync(pool.buffers[b].tables.p, entries, 3 * experts * sizeof(void*),
                                   cudaMemcpyHostToDevice, t));
        CUDA_CHECK(cudaEventRecord(state.copied[source], t));
        CUDA_CHECK(cudaEventRecord(target.ready, t));
    }

    // Before a pass: a wide one takes the pool buffers back from the expert cache, a decode or
    // verification pass returns them.
    void place_pools(bool wide) {
        if (!pools_lent) { return; }
        for (std::size_t r = 0; r < slot_pools.size(); ++r) {
            if (slot_pools[r].slots == 0) { continue; }
            const cudaStream_t transfer = device.rank(r).transfer_stream;
            if (wide) {
                cache->reclaim(r, transfer);
            } else {
                cache->give_back(r, transfer);
            }
        }
    }

    // Forgets every prefetched layer, so the next wide call copies its own.
    void forget_prefetches() {
        drain_prefetches();
        for (auto& prefetch : prefetches) {
            for (auto& buffer : prefetch.buffers) { buffer.layer = ~std::size_t{0}; }
        }
    }

    // Feeds the last pass's routes to the expert cache and lets it swap experts, before the next
    // pass reads the tables.
    void settle_routes() {
        if (pending_routes == 0 && mtp_routes == 0) { return; }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaEventSynchronize(rank.routes));
        }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        const auto* host = static_cast<const std::int32_t*>(route_host->data());
        if (cache) {
            const std::size_t used = std::size_t(config.num_experts_per_tok) * pending_routes;
            // The tier learns from decode and verification passes; a prompt chunk, which routes
            // nearly every expert, only counts toward the statistics.
            const bool decode = pending_routes <= std::uint32_t(kVectorTokens);
            if (pending_routes > 0) {
                for (std::size_t i = 0; i < layers.size(); ++i) {
                    cache->observe(i, std::span(host + i * pairs, used), decode);
                }
            }
            if (mtp_routes > 0) {
                cache->observe(layers.size(),
                               std::span(host + layers.size() * pairs,
                                         std::size_t(config.num_experts_per_tok) * mtp_routes),
                               true);
            }
            if (decode) { cache->adapt(); }
        }
        pending_routes = 0;
        mtp_routes     = 0;
    }

    // Brings the pass's routes to the host for the expert cache's counts; nothing else reads them.
    void record_routes(std::uint32_t tokens) {
        if (!cache || warming) { return; }
        const std::uint64_t pairs =
            std::uint64_t(config.num_experts_per_tok) * options.prefill_chunk;
        auto* host = static_cast<std::int32_t*>(route_host->data());
        for (std::size_t i = 0; i < layers.size();) {
            std::size_t end = i;
            while (end < layers.size() && layers[end].rank == layers[i].rank) { ++end; }
            RankBinding bind(device, layers[i].rank);
            CUDA_CHECK(cudaMemcpy2DAsync(host + i * pairs, pairs * 4, route_records[i], pairs * 4,
                                         std::size_t(config.num_experts_per_tok) * tokens * 4,
                                         end - i, cudaMemcpyDeviceToHost,
                                         ranks[layers[i].rank].stream));
            i = end;
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaEventRecord(rank.routes, rank.stream));
        }
        pending_routes = tokens;
    }

    // The state regions a reset zeroes, a snapshot keeps and a restore puts back: the recurrent
    // state, the convolution window, the indexer's tail and the PLE history, where the layer has
    // them; null where it does not.
    static std::array<std::pair<void*, std::size_t>, 4> recurrent_regions(const LayerState& state) {
        return {{{state.ssm.data, state.ssm.data != nullptr ? state.ssm.bytes() : 0},
                 {state.conv.data, state.conv.data != nullptr ? state.conv.bytes() : 0},
                 {state.tail.p, state.tail.bytes},
                 {state.history.p, state.history.bytes}}};
    }

    void reset(std::uint32_t s) {
        auto& sequence    = sequences.at(s);
        if (std::find(verified.begin(), verified.end(), &sequence) != verified.end()) { verified.clear(); }
        sequence.position = 0;
        sequence.context  = NgramContext::sequence_start(ngram, config.eos_token_id);
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta  = 0;
        sequence.mtp_follows = true;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            const cudaStream_t stream = ranks[layers[i].rank].stream;
            LayerState& state         = sequence.layers[i];
            for (const auto& [data, bytes] : recurrent_regions(state)) {
                if (data != nullptr) { CUDA_CHECK(cudaMemsetAsync(data, 0, bytes, stream)); }
            }
            if (state.pooled.p != nullptr) {
                CUDA_CHECK(cudaMemsetAsync(state.pooled.p, 0, state.pooled.bytes, stream));
            }
        }
        if (mtp) {
            RankBinding bind(device, model.head_rank());
            const cudaStream_t stream = ranks[model.head_rank()].stream;
            for (DeviceBuffer* buffer : {&sequence.mtp.pooled, &sequence.mtp.tail}) {
                CUDA_CHECK(cudaMemsetAsync(buffer->p, 0, buffer->bytes, stream));
            }
        }
        forget_streamed_kv(sequence);
    }

    // A sequence's streamed layers: every block leaves their pools, and no rank's staging copy
    // stands for them, after a reset or a loaded image rewrote what their positions hold.
    void forget_streamed_kv(SequenceState& sequence) {
        if (stream_slots == 0) { return; }
        const auto forget = [&](LayerState& state, std::size_t rank) {
            if (state.map.table == nullptr) { return; }
            RankBinding bind(device, rank);
            ops::kv_stream_reset(state.map, ranks[rank].stream);
            if (ranks[rank].kv_staged == &state) { ranks[rank].kv_staged = nullptr; }
        };
        for (std::size_t i = 0; i < layers.size(); ++i) {
            forget(sequence.layers[i], layers[i].rank);
        }
        if (mtp) { forget(sequence.mtp, model.head_rank()); }
    }

    // Moves the stack's first `tokens` columns from one rank to another, ordered after the
    // source's work. The source's next pass (the next prompt chunk, queued before the target gets
    // to this copy) rewrites the source's stack, so it waits until the copy has read it.
    void cross(std::size_t from, std::size_t to, std::int32_t tokens) {
        RankState& source = ranks[from];
        RankState& target = ranks[to];
        {
            RankBinding bind(device, from);
            CUDA_CHECK(cudaEventRecord(source.done, source.stream));
        }
        {
            RankBinding bind(device, to);
            CUDA_CHECK(cudaStreamWaitEvent(target.stream, source.done, 0));
            const std::size_t bytes =
                std::size_t(config.hc_count) * config.hidden_size * std::size_t(tokens) * 4;
            CUDA_CHECK(cudaMemcpyPeerAsync(target.stack, device.rank(to).device, source.stack,
                                           device.rank(from).device, bytes, target.stream));
            CUDA_CHECK(cudaEventRecord(target.copied, target.stream));
        }
        RankBinding bind(device, from);
        CUDA_CHECK(cudaStreamWaitEvent(source.stream, target.copied, 0));
    }

    // A pass's Vision work: the runs of consecutive embedding columns its tokens take (each run's
    // destination columns sit in the scatter plane at `offset`), and whether its one part rotates
    // at a media prompt's three-axis positions.
    struct MediaRun {
        std::int32_t source = 0, count = 0, offset = 0;
    };
    std::vector<MediaRun> media_runs;
    bool mrope = false;

    // Hints the table rows of `next`, the known tokens after the sequence's hashed ones, from a
    // copy of its hash context: forward() calls it once its own rows are read.
    void hint_next(const SequenceState& sequence, std::span<const std::int32_t> next) {
        if (next.empty() || !table || options.ngram_read.residency == NgramResidency::Ram ||
            options.ngram_read.io == NgramIo::Direct) { return; }
        NgramContext context = sequence.context;
        std::vector<std::uint64_t> rows(next.size() * config.ngram_heads());
        ngram_row_ids(ngram, next, config.eos_token_id, config.vocab_size, context, rows);
        table->prefetch(rows);
    }

    // Stages a pass's inputs. With `read_rows` it also starts reading the tokens' n-gram rows into
    // the rows' staging (stage_rows uploads them); without, the rows' ids stay in row_ids.
    void stage_inputs(std::span<const Part> parts, std::span<const std::int32_t> tokens,
                      bool read_rows = true) {
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
            const auto start = static_cast<std::int32_t>(sequence.position + part.offset);
            std::iota(first, first + part.count, start);
            for (std::int32_t i = 0; i < part.count; ++i) {
                rope[std::size_t(part.column + i)] = start + i + sequence.rope_delta;
            }
            if (table) {
                // A verification hashes from a copy, which commit() replaces by the context
                // advanced over the tokens it keeps. Each branch hashes from the origin's context,
                // which the sequence holds: the last branch advances it, the others a copy. A
                // pass's prompt part advances it first, and the origin takes it there.
                NgramContext branch_context;
                const bool opening = branch_capture != nullptr && &part == &parts.front();
                const bool copied  = branch_origin != nullptr && !opening && &part != &parts.back();
                if (copied) { branch_context = branch_origin->context; }
                NgramContext& context = copied      ? branch_context
                                        : verifying ? part.sequence->verify_context
                                                    : part.sequence->context;
                ngram_row_ids(ngram,
                              tokens.subspan(std::size_t(part.column), std::size_t(part.count)),
                              config.eos_token_id, config.vocab_size, context,
                              std::span(row_ids).subspan(std::size_t(part.column) * heads,
                                                         std::size_t(part.count) * heads));
                if (opening) { branch_capture->context = context; }
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
        if (table) {
            // Rows a failed pass left reading go nowhere now.
            try {
                table->wait();
            } catch (...) {}
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
            if (table && read_rows && rank.rank == rows_rank) {
                // The rows are read while the layers before the PLE layer run; stage_rows()
                // uploads them and marks the staging buffer free.
                table->submit(row_ids, staged_rows(rank, row_ids.size() * row_b));
                rows_waited = false;
            }
            const bool ple_here =
                std::any_of(layers.begin(), layers.end(),
                            [&](const LayerPlan& p) { return p.ple && p.rank == rank.rank; });
            if (!ple_here || !read_rows) { CUDA_CHECK(cudaEventRecord(rank.staged, rank.stream)); }
        }
    }

    // The n-gram rows' place in a rank's staging buffer.
    std::span<std::uint8_t> staged_rows(RankState& rank, std::size_t bytes) const {
        auto* base = static_cast<std::byte*>(rank.staging->data());
        return {reinterpret_cast<std::uint8_t*>(
                    base + round_up(28 * std::uint64_t(options.prefill_chunk))),
                bytes};
    }

    // Before a segment that starts with its rank's first PLE layer: the rows stage_inputs() started
    // reading, uploaded to the rank once read. The pass's first upload waits for the read between
    // a pair of stall events.
    void stage_rows(RankState& rank) {
        const std::size_t bytes = row_ids.size() * ops::ngram_row_bytes(options.ngram->format);
        RankState& reader       = ranks[rows_rank];
        await_rows(reader);
        std::span<std::uint8_t> rows = staged_rows(reader, bytes);
        if (&rank != &reader) {
            const auto own = staged_rows(rank, bytes);
            std::memcpy(own.data(), rows.data(), bytes);
            rows = own;
        }
        CUDA_CHECK(
            cudaMemcpyAsync(rank.rows, rows.data(), bytes, cudaMemcpyHostToDevice, rank.stream));
        CUDA_CHECK(cudaEventRecord(rank.staged, rank.stream));
    }

    // Waits for the rows being read, unless already waited for, between a pair of stall events on
    // the reading rank's stream.
    void await_rows(RankState& reader) {
        if (rows_waited) { return; }
        StallEvents& events = stall_events[stall_next];
        stall_next          = (stall_next + 1) % stall_events.size();
        if (events.wanted == nullptr) {
            CUDA_CHECK(cudaEventCreate(&events.wanted));
            CUDA_CHECK(cudaEventCreate(&events.ready));
        }
        measure_stall(events, true);
        CUDA_CHECK(cudaEventRecord(events.wanted, reader.stream));
        table->wait();
        rows_waited = true;
        CUDA_CHECK(cudaEventRecord(events.ready, reader.stream));
        events.pending = true;
    }

    // Adds a finished wait to the stall counters; `block` waits for it to finish.
    void measure_stall(StallEvents& events, bool block) {
        if (!events.pending) { return; }
        if (block) {
            CUDA_CHECK(cudaEventSynchronize(events.ready));
        } else {
            const cudaError_t status = cudaEventQuery(events.ready);
            if (status == cudaErrorNotReady) { return; }
            CUDA_CHECK(status);
        }
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, events.wanted, events.ready));
        events.pending = false;
        // An upload queued behind unfinished layers completes right after them; anything beyond
        // a few microseconds is time the stream waited for the read.
        if (ms > 0.002F) {
            ++stalls;
            stall_seconds += double(ms) * 1e-3;
        }
    }

    NgramTableStats ngram_stats() {
        if (!table) { return {}; }
        RankBinding bind(device, rows_rank);
        for (StallEvents& events : stall_events) { measure_stall(events, false); }
        NgramTableStats out = table->counters();
        out.stalls          = stalls;
        out.stall_seconds   = stall_seconds;
        return out;
    }

    // The part's columns of a BF16 [rows, *] plane.
    static void* columns(void* plane, std::int32_t rows, std::int32_t column, std::size_t bytes) {
        return static_cast<std::byte*>(plane) + std::size_t(rows) * std::size_t(column) * bytes;
    }

    // A verification's record row of the part: its position among the verified sequences.
    [[nodiscard]] std::int32_t record_row(const Part& part) const {
        return part.column / static_cast<std::int32_t>(active_verify_width);
    }

    void run_gdn(const LayerPlan& plan, std::size_t index, LayerState& state, RankState& rank,
                 const Part& part) {
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
        // A verification records its raw projection, the transitions and the gates for the fold
        // that commits a prefix, and leaves the sequence's states as they were.
        std::optional<GdnReplayRecordLayer> record;
        if (verifying) {
            const auto row = record_row(part);
            const auto all = rank.records->layer(static_cast<std::int32_t>(gdn_local[index]),
                                                 static_cast<std::int32_t>(verified.size()),
                                                 static_cast<std::int32_t>(active_verify_width));
            record = GdnReplayRecordLayer{all.conv.slice(2, row, 1), all.key.slice(3, row, 1),
                                          all.value.slice(3, row, 1), all.gate.slice(3, row, 1)};
        }
        Tensor qkv = record ? Tensor(record->conv.data, DType::BF16, {g.qkv.rows, t})
                            : Tensor(rank.a, DType::BF16, {g.qkv.rows, t});
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
        const auto window = static_cast<std::int32_t>(config.linear_conv_kernel_dim - 1);
        Tensor conv_state(state.conv.data, DType::BF16, {g.qkv.rows, window});
        Tensor conv_next =
            record ? Tensor(rank.conv_scratch.p, DType::BF16, {g.qkv.rows, window}) : conv_state;
        ops::causal_conv1d_silu_split(qkv, g.convolution, conv_state, conv_next, q, k, v, s);
        Tensor gate(rank.g, DType::FP32, {nv, t});
        Tensor beta(static_cast<std::byte*>(rank.g) + round_up(std::uint64_t(nv) * t * 4),
                    DType::FP32, {nv, t});
        ops::gdn_gating(ga, gb, g.a_log, g.dt_bias, gate, beta, s);
        Tensor q3(q.data, DType::BF16, {dh, nk, t}), k3(k.data, DType::BF16, {dh, nk, t});
        Tensor v3(v.data, DType::BF16, {dh, nv, t});
        Tensor o(rank.f, DType::BF16, {dh, nv, t});
        const float scale = 1.0F / std::sqrt(float(dh));
        if (record) {
            const Tensor states = rank.gdn->layer_view(gdn_local[index]).recurrent;
            const Tensor slot(static_cast<std::int32_t*>(rank.slot_ids.p) + part.sequence->slot,
                              DType::I32, {1});
            Tensor o4 = o.view({dh, nv, t, 1});
            ops::gated_delta_net_replay_record(q3.view({dh, nk, t, 1}), k3.view({dh, nk, t, 1}),
                                               v3.view({dh, nv, t, 1}), gate.view({nv, t, 1}),
                                               beta.view({nv, t, 1}), scale, states, Tensor{}, slot,
                                               record->key, record->value, record->gate, o4, s);
        } else {
            Tensor ssm(state.ssm.data, DType::FP32, {dh, dh, nv});
            auto scope = ws.scope();
            ops::gated_delta_net(q3, k3, v3, gate, beta, scale, true, ws, ssm, o, s);
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

    // Where a sparse-attention layer finds its positions and keeps its indexer state. By default
    // the staged planes and the sequence's live tail; a verification projects the indexer into its
    // record row and runs a scratch copy of the tail (commit() advances the live one); an MTP pass
    // reads its cells' positions and, drafting, its scratch tail.
    struct AttentionInputs {
        const std::int32_t* positions = nullptr; // sequence indices: KV slots, causal order
        const std::int32_t* rope      = nullptr; // 1-D RoPE positions
        void* index                   = nullptr; // BF16 [640, count]
        void* tail                    = nullptr; // FP32 [128, 3]
        std::uint32_t first           = 0;       // the first position, as the host knows it
    };

    [[nodiscard]] AttentionInputs target_inputs(std::size_t index, LayerState& state,
                                                RankState& rank, const Part& part) {
        AttentionInputs in{.positions = rank.positions + part.column,
                           .rope      = rank.rope + part.column,
                           .index     = nullptr,
                           .tail      = state.tail.p,
                           .first     = part.sequence->position + part.offset};
        if (verifying) {
            const std::uint64_t rows =
                std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
            const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
            in.index = static_cast<std::byte*>(rank.index_records.p) +
                       (qsa_local[index] * columns + std::uint64_t(part.column)) * rows * 2;
            CUDA_CHECK(cudaMemcpyAsync(rank.tail_scratch.p, state.tail.p, state.tail.bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
            in.tail = rank.tail_scratch.p;
        }
        return in;
    }

    void run_qsa(const LayerPlan& plan, LayerState& state, RankState& rank, const Part& part,
                 const AttentionInputs& in) {
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
        Tensor index(in.index != nullptr ? in.index
                                         : kv + 2 * round_up(std::uint64_t(d) * nk * t * 2),
                     DType::BF16, {a.index.rows, t});
        project(a.query, mixed, q, ws, s);
        project(a.gate, mixed, gate, ws, s);
        project(a.key, mixed, k, ws, s);
        project(a.value, mixed, v, ws, s);
        project(a.index, mixed, index, ws, s);
        Tensor q3(rank.a, DType::BF16, {d, nq, t}), k3(k.data, DType::BF16, {d, nk, t});
        Tensor qo(rank.d, DType::BF16, {d, nq, t});
        Tensor ko(rank.e, DType::BF16, {d, nk, t});
        auto* const part_positions = const_cast<std::int32_t*>(in.positions);
        const Tensor positions(part_positions, DType::I32, {t});
        const ops::RopeYarn& yarn = options.rope_yarn;
        if (mrope || yarn.active()) {
            // A media prompt: the three-axis positions of its images and video, each pair of
            // rotated dimensions on axis i % 3 (the interleaved MRoPE sections). Under YaRN, text
            // too: the fused norm and rotation fixes the unscaled one, so the rotation is its own
            // Op, one tensor at a time (its YaRN forms cover 24 query and 2 key heads that way).
            const Tensor axes(rank.mrope, DType::I32, {t, 3});
            const Tensor line(const_cast<std::int32_t*>(in.rope), DType::I32, {t});
            const int rotary = static_cast<int>(float(d) * config.partial_rotary_factor);
            ops::rmsnorm(q3, a.query_norm, config.rms_norm_eps, true, qo, s);
            ops::rmsnorm(k3, a.key_norm, config.rms_norm_eps, true, ko, s);
            if (yarn.active()) {
                ops::rope(mrope ? axes : line, rotary, config.rope_theta, yarn, qo, s);
                ops::rope(mrope ? axes : line, rotary, config.rope_theta, yarn, ko, s);
            } else {
                ops::rope(axes, rotary, config.rope_theta, qo, ko, s);
            }
        } else {
            const Tensor rope(const_cast<std::int32_t*>(in.rope), DType::I32, {t});
            ops::rmsnorm_rope(rope, a.query_norm, a.key_norm, q3, k3, qo, ko, s);
        }
        // The paged KV: device pages, or a streamed layer's host copy. A streamed layer's call of
        // at most stream_width tokens reads its device pool once kv_stream_resolve has made the
        // call's blocks resident there; a wider call (a prompt chunk) reads the rank's staging
        // copy, into which it appends as well, and then drops the blocks it wrote from the pool.
        const auto paged_count = static_cast<std::int32_t>(pages);
        const PagedKVLayerView cache =
            kv_view([&](std::size_t i) { return state.kv[i].p; }, kPagedKVPageSize, paged_count,
                    rank.block_table.p, paged_count);
        const bool streamed = state.map.table != nullptr;
        const bool pooled_read = streamed && t <= static_cast<std::int32_t>(stream_width);
        const Tensor v3(v.data, DType::BF16, {d, nk, t});
        ops::kv_cache_append(ko, v3, positions, cache, s);
        PagedKVLayerView read = cache;
        if (pooled_read) {
            if (rank.kv_staged == &state) { rank.kv_staged = nullptr; }
            read = kv_view([&](std::size_t i) { return state.pool[i].p; }, ops::kKvStreamPageTokens,
                           static_cast<std::int32_t>(stream_slots), state.map.table,
                           state.map.blocks);
        } else if (streamed) {
            stage(state, rank, in.first);
            read = kv_view([&](std::size_t i) { return rank.kv_staging[i].p; }, kPagedKVPageSize,
                           paged_count, rank.block_table.p, paged_count);
            ops::kv_cache_append(ko, v3, positions, read, s);
            rank.kv_staged_end = in.first + std::uint32_t(t);
        }
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        Tensor pooled(state.pooled.p, DType::FP32, {di, static_cast<std::int32_t>(pooled_slots)});
        Tensor tail(in.tail, DType::FP32,
                    {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
        const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
        // The call's first position is the part's first staged one, read on the device.
        const Tensor first(part_positions, DType::I32, {1});
        ops::qsa_indexer_append(index, first, iw, config.rms_norm_eps, yarn, pooled, tail, s);
        Tensor selected(rank.selected, DType::I32,
                        {static_cast<std::int32_t>(config.indexer_block_budget()), t});
        Tensor counts(rank.counts, DType::I32, {t});
        {
            auto scope = ws.scope();
            ops::qsa_indexer_select(index, first, iw, config.rms_norm_eps, yarn, pooled, ws,
                                    selected, counts, s);
        }
        if (pooled_read) {
            ops::kv_stream_resolve(state.map, first, selected, counts, cache, read, s);
        }
        Tensor attention(rank.f, DType::BF16, {d, nq, t});
        {
            auto scope = ws.scope();
            ops::sparse_softmax_attention(qo, first, selected, counts, read,
                                          1.0F / std::sqrt(float(d)), ws, attention, s);
        }
        if (streamed && !pooled_read) {
            const auto block = [](std::uint32_t position) {
                return static_cast<std::int32_t>(position / ops::kKvStreamPageTokens);
            };
            ops::kv_stream_evict(state.map, block(in.first),
                                 block(in.first + std::uint32_t(t) - 1) + 1, s);
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
        const ops::PleInjectWeights weights{&p.norm_key, &p.norm_query, &p.norm_conv,
                                            &p.convolution};
        auto scope = ws.scope();
        if (verifying) {
            // The history stays; commit() advances it over the kept positions' recorded input.
            Tensor normalized(static_cast<float*>(rank.ple_records.p) +
                                  std::size_t(width) * std::size_t(part.column),
                              DType::FP32, {width, t});
            ops::ple_inject_record(stack, key, value, weights, config.rms_norm_eps, history,
                                   normalized, ws, s);
            return;
        }
        ops::ple_inject(stack, key, value, weights, config.rms_norm_eps, history, ws, s);
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
        const std::size_t b        = pool_buffer(index);
        const bool double_buffered = pool.buffers.size() == 2;
        // A span's last chunk may be shorter than kPrefetchTokens, but the pool still holds the
        // layer's experts that its earlier chunks prefetched: it runs from the pool too, rather
        // than copying its routed experts over the pool again, and the next layer's copies start
        // behind it as behind any prefetched call.
        const bool prefetched =
            !prefetches.empty() && prefetches[rank.rank].buffers[b].layer == index;
        if ((t >= kPrefetchTokens || prefetched) && pool.slots >= plan.moe.gate.pointers.size()) {
            PoolPrefetch& state = prefetches[rank.rank];
            auto& buffer        = state.buffers[b];
            if (buffer.layer != index) { prefetch_layer(index); }
            // With two buffers the next layer's copies need only the previous layer's calls done:
            // they run while this layer's experts do.
            if (double_buffered && pooled_layer(index + 1, rank.rank) &&
                state.buffers[1 - b].layer != index + 1) {
                prefetch_layer(index + 1);
            }
            await_prefetch(buffer);
            CUDA_CHECK(cudaStreamWaitEvent(s, buffer.ready, 0));
            ops::GgufMoeWeights banks = plan.moe.banks();
            const auto* table_p       = static_cast<const void* const*>(pool.buffers[b].tables.p);
            const std::size_t experts = plan.moe.gate.pointers.size();
            banks.gate.experts        = table_p;
            banks.up.experts          = table_p + experts;
            banks.down.experts        = table_p + 2 * experts;
            banks.device_resident     = true;
            const auto h              = static_cast<std::int32_t>(config.hidden_size);
            const Tensor mixed(rank.mixed, DType::BF16, {h, t});
            Tensor y(rank.y, DType::FP32, {h, t});
            {
                auto scope = rank.workspace->scope();
                ops::moe_experts_gguf(mixed, ids, weights, shared, banks, *rank.workspace, y, s);
            }
            CUDA_CHECK(cudaEventRecord(buffer.free, s));
            // A span's later chunks run this layer's experts again from the pool; otherwise the
            // buffer takes the copies of the next text layer on this device that reads it.
            if (span_hold) { return true; }
            const std::size_t next = index + (double_buffered ? 2 : 1);
            if (pooled_layer(next, rank.rank)) {
                prefetch_layer(next);
            } else {
                buffer.layer = ~std::size_t{0};
            }
            return true;
        }
        const auto top       = static_cast<std::int32_t>(config.num_experts_per_tok);
        auto* routes         = static_cast<std::int32_t*>(route_host->data()) +
                               index * std::size_t(top) * options.prefill_chunk;
        CUDA_CHECK(
            cudaMemcpyAsync(routes, ids.data, std::size_t(top) * t * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        const MoePlan& m          = plan.moe;
        const std::size_t experts = m.gate.pointers.size();
        // The copies below overwrite the first buffer, so no buffer holds a prefetched layer.
        if (!prefetches.empty()) {
            drain_prefetches();
            for (auto& buffer : prefetches[rank.rank].buffers) { buffer.layer = ~std::size_t{0}; }
        }
        std::vector<char> routed(experts, 0);
        for (std::int32_t i = 0; i < top * t; ++i) {
            const std::int32_t e = routes[i];
            if (e >= 0 && std::size_t(e) < experts) { routed[e] = 1; }
        }
        if (pool.slots < experts) { return false; }
        auto* entries     = static_cast<const void**>(slot_entries->data());
        const auto wanted = [&](std::size_t e) { return routed[e] != 0; };
        stage_pool(pool, pool.buffers[0], index, wanted, entries, s);
        CUDA_CHECK(cudaMemcpyAsync(pool.buffers[0].tables.p, entries, 3 * experts * sizeof(void*),
                                   cudaMemcpyHostToDevice, s));
        ops::GgufMoeWeights banks = m.banks();
        const auto* table_p       = static_cast<const void* const*>(pool.buffers[0].tables.p);
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
        Tensor ids(route_records[index], DType::I32, {top, t});
        Tensor weights(rank.route_weights, DType::FP32, {top, t});
        Tensor shared(rank.route_shared, DType::FP32, {t});
        {
            auto scope = ws.scope();
            ops::moe_route(mixed, m.router, m.shared_gate, ws, ids, weights, shared, s);
        }
        if (m.native) {
            Tensor y(rank.y, DType::FP32, {h, t});
            if (hybrid) {
                hybrid->run(hybrid_layers[index], mixed, ids, weights, shared, m.native->banks(), t == 1 || verifying,
                            ws, y);
                return;
            }
            auto scope = ws.scope();
            ops::moe_experts_native(mixed, ids, weights, shared, m.native->banks(), nullptr, ws, y, s);
            return;
        }
        if (misses && index == layers.size() && t > kVectorTokens) {
            // The MTP block only steers drafts, which verification checks: a prompt chunk's
            // missing experts are dropped rather than fetched. Draft steps and a round's cells
            // serve them as a decode layer does, so the drafts follow the block's own routing.
            Tensor y(rank.y, DType::FP32, {h, t});
            auto scope = ws.scope();
            Tensor kept_ids     = ws.alloc(DType::I32, {top, t});
            Tensor kept_weights = ws.alloc(DType::FP32, {top, t});
            misses->drop_missing(index, ids, weights, kept_ids, kept_weights, s);
            if (ops::moe_experts_gguf_decode_supported(m.banks(), t)) {
                const auto state = ops::moe_experts_gguf_decode_begin(mixed, kept_ids, shared,
                                                                      m.banks(), ws, s, &rank.side);
                ops::moe_experts_gguf_decode_finish(state, kept_ids, nullptr, kept_weights,
                                                    m.banks(), nullptr, nullptr, nullptr, nullptr,
                                                    y, s);
                return;
            }
            const auto stage = ops::moe_experts_gguf_begin(mixed, ws, s);
            ops::moe_experts_gguf_add(stage, mixed, kept_ids, kept_weights, &shared, m.banks(), ws,
                                      s);
            ops::moe_experts_gguf_finish(stage, y, s);
            return;
        }
        if (misses && t <= kVectorTokens) {
            // The cached pairs run at once; the missing experts arrive (or the CPU computes them)
            // meanwhile, and run in a second stage.
            Tensor y(rank.y, DType::FP32, {h, t});
            auto scope      = ws.scope();
            const auto call = misses->begin(index, ids, mixed, t, m.banks(), s);
            if (ops::moe_experts_gguf_decode_supported(m.banks(), t)) {
                const auto state = ops::moe_experts_gguf_decode_begin(
                    mixed, call.cached_ids, shared, m.banks(), ws, s, &rank.side);
                misses->await(index, s);
                ops::moe_experts_gguf_decode_missing(state, mixed, call.missing_ids,
                                                     call.missing_banks, s);
                ops::moe_experts_gguf_decode_finish(state, call.cached_ids, &call.missing_ids,
                                                    weights, m.banks(), &call.missing_banks,
                                                    call.products, call.pairs, call.count, y, s);
                return;
            }
            const auto stage = ops::moe_experts_gguf_begin(mixed, ws, s);
            ops::moe_experts_gguf_add(stage, mixed, call.cached_ids, weights, &shared, m.banks(),
                                      ws, s);
            misses->await(index, s);
            ops::moe_experts_gguf_add(stage, mixed, call.missing_ids, weights, nullptr,
                                      call.missing_banks, ws, s);
            if (call.max_products > 0) {
                ops::moe_experts_gguf_add_products(stage, call.products, call.pairs, call.count,
                                                   call.max_products, weights, s);
            }
            ops::moe_experts_gguf_finish(stage, y, s);
            return;
        }
        // The slot pools are sized for the text layers' experts; the MTP block's wide calls stay
        // on the vector products over its host banks.
        if (t > kVectorTokens && !slot_pools.empty() && index < layers.size() &&
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
        // A layer's MoE output is written into the stack by the next layer's read, which takes
        // the same rounding, unless the PLE comes between.
        Tensor inject(rank.inject, DType::FP32, {hc, t});
        // In a branch pass the first branch continues the state its sequence holds and each later
        // one starts a layer from the origin's; a prompt part ahead of the branches leaves its
        // state there as it goes.
        const std::size_t first_branch = branch_capture != nullptr ? 1 : 0;
        const Tensor moe_out(rank.y, DType::FP32, {h, t});
        bool pending = false;
        for (std::size_t i = segment.begin; i < segment.end; ++i) {
            const LayerPlan& plan = layers[i];
            if (plan.ple) {
                if (pending) { ops::hyper_connection_write(stack, moe_out, inject, rank.stream); }
                pending = false;
                for (std::size_t k = 0; k < parts.size(); ++k) {
                    LayerState& state = parts[k].sequence->layers[i];
                    if (k > first_branch) { restore_branch_layer(state, i, rank, kPleRegions); }
                    run_ple(plan, state, rank, parts[k]);
                    if (k == 0) { capture_branch_layer(state, i, rank, kPleRegions); }
                }
            }
            {
                auto scope = rank.workspace->scope();
                if (pending) {
                    ops::hyper_connection_write_read(stack, moe_out, inject, plan.attn_hc.weights(),
                                                     config.rms_norm_eps, *rank.workspace, mixed,
                                                     &inject, rank.stream);
                } else {
                    ops::hyper_connection_read(stack, plan.attn_hc.weights(), config.rms_norm_eps,
                                               *rank.workspace, mixed, &inject, rank.stream);
                }
                pending = false;
            }
            for (std::size_t k = 0; k < parts.size(); ++k) {
                const Part& part  = parts[k];
                LayerState& state = part.sequence->layers[i];
                if (k > first_branch) { restore_branch_layer(state, i, rank, kMixerRegions); }
                if (plan.gdn) {
                    run_gdn(plan, i, state, rank, part);
                } else {
                    run_qsa(plan, state, rank, part, target_inputs(i, state, rank, part));
                }
                if (k == 0) { capture_branch_layer(state, i, rank, kMixerRegions); }
            }
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_write_read(
                    stack, Tensor(rank.y, DType::BF16, {h, t}), inject, plan.mlp_hc.weights(),
                    config.rms_norm_eps, *rank.workspace, mixed, &inject, rank.stream);
            }
            run_moe(plan, rank, t, i);
            pending = true;
        }
        if (pending) { ops::hyper_connection_write(stack, moe_out, inject, rank.stream); }
        if (!segment.head) { return; }
        const auto n = static_cast<std::int32_t>(logit_rows);
        if (branch_origin != nullptr) {
            // Each branch's last column, to the front of the stack in branch order. A branch's
            // last column lies at or past its index, so no copy overwrites a later one's source.
            const std::size_t bytes = std::size_t(hc) * h * 4;
            for (std::size_t b = 0; b + first_branch < parts.size(); ++b) {
                const Part& part       = parts[b + first_branch];
                const std::size_t last = std::size_t(part.column + part.count - 1);
                if (last == b) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(rank.stack + b * std::size_t(hc) * h,
                                           rank.stack + last * std::size_t(hc) * h, bytes,
                                           cudaMemcpyDeviceToDevice, rank.stream));
            }
        }
        const std::size_t first_row = branch_origin != nullptr ? 0 : std::size_t(t - n);
        const Tensor last(rank.stack + std::size_t(hc) * h * first_row, DType::FP32, {h, hc, n});
        Tensor head_mixed(rank.mixed, DType::BF16, {h, n});
        {
            auto scope = rank.workspace->scope();
            ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                       *rank.workspace, head_mixed, nullptr, rank.stream);
        }
        Tensor logits(rank.logits, DType::BF16, {static_cast<std::int32_t>(config.vocab_size), n});
        project(head, head_mixed, logits, *rank.workspace, rank.stream);
    }

    // A graph per segment of a pass over one part of `t` tokens of `sequence` (a decode step, or a
    // verification while `verifying`).
    std::vector<DecodeGraphExecutable> capture_segments(SequenceState& sequence, std::int32_t t) {
        std::vector<DecodeGraphExecutable> graphs;
        for (const Segment& segment : segments) {
            RankBinding bind(device, segment.rank);
            DecodeGraphDefinition definition;
            const Part part{.sequence = &sequence, .column = 0, .count = t};
            definition.capture(ranks[segment.rank].stream, [&] {
                run_segment(std::span(&part, 1), segment, t, std::uint32_t(t));
            });
            DecodeGraphExecutable graph;
            graph.instantiate(definition);
            graphs.push_back(std::move(graph));
        }
        return graphs;
    }

    void capture_decode(SequenceState& sequence) {
        sequence.decode = capture_segments(sequence, 1);
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
            if (part.sequence->position + part.offset + std::uint32_t(part.count) >
                options.max_context) {
                throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
            }
        }
        settle_routes();
        // A copy queued for an earlier pass may predate the cache's last swaps.
        forget_prefetches();
        place_pools(t > kVectorTokens);
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_inputs(parts, tokens);
        // Disk-resident experts need the host between a layer's routing and its experts, so their
        // passes stay eager, and so does a batch of several sequences. A verification of one
        // sequence replays its graphs while its experts take the vector products. Native host
        // experts coordinate with their supervisor through captured handshakes.
        SequenceState& first = *parts.front().sequence;
        // Staged misses keep disk experts' decode and verification free of host synchronization.
        const bool host_free = !stream || misses;
        const bool graph =
            options.cuda_graphs && t == 1 && host_free && !verifying && branch_origin == nullptr;
        const bool verify_graph = options.cuda_graphs && verifying && parts.size() == 1 &&
                                  host_free && t <= kVectorTokens && branch_origin == nullptr;
        if (misses) { misses->keep_alive(); }
        if (graph && first.decode.empty() && first.decode_steps++ > 0) { capture_decode(first); }
        auto* verification = verify_graph ? &first.verify.at(std::size_t(t)) : nullptr;
        if (verification && verification->segments.empty() && verification->runs++ > 0) {
            verification->segments = capture_segments(first, t);
        }
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const Segment& segment = segments[i];
            if (i > 0 && segments[i - 1].rank != segment.rank) {
                cross(segments[i - 1].rank, segment.rank, t);
            }
            RankBinding bind(device, segment.rank);
            if (segment.rows && table) { stage_rows(ranks[segment.rank]); }
            if (graph && !first.decode.empty()) {
                first.decode[i].launch(ranks[segment.rank].stream);
            } else if (verification && !verification->segments.empty()) {
                verification->segments[i].launch(ranks[segment.rank].stream);
            } else {
                run_segment(parts, segment, t, logit_rows);
            }
        }
        if (hybrid) { hybrid->finish(); }
        // A verification commits its positions later (commit()).
        if (!verifying) {
            for (const Part& part : parts) {
                part.sequence->position += static_cast<std::uint32_t>(part.count);
            }
        }
        record_routes(static_cast<std::uint32_t>(t));
        // While a staged layer waits on the device for its missing experts, no other launch may
        // load a kernel lazily: loading can wait for the device to go idle, the waiting kernel for
        // the misses' service, and the service's own calls for the loader. The caller's next
        // kernels therefore follow a pass that may have waited.
        if (misses && t <= kVectorTokens) { synchronize_ranks(); }
    }

    void synchronize_ranks() {
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
    }

    // The tokens a prompt span takes, 0 without spans. Spans need one device and GGUF host
    // experts, whose wide calls copy every uncached expert of a layer into the pool: per span
    // instead of per chunk.
    [[nodiscard]] std::uint32_t planned_span_tokens() const {
        const bool native = std::any_of(layers.begin(), layers.end(),
                                        [](const LayerPlan& plan) { return plan.moe.native.has_value(); });
        // One device: its segments split only where the PLE layer's rows arrive.
        const bool one_rank = std::all_of(segments.begin(), segments.end(),
                                          [](const Segment& segment) { return segment.rank == 0; });
        if (model.options().experts != ExpertResidency::Host || native || !one_rank) { return 0; }
        const std::uint32_t tokens =
            std::min<std::uint32_t>(options.max_context, kSpanChunks * options.prefill_chunk);
        return tokens > options.prefill_chunk ? tokens : 0;
    }

    // The chunks one expert call of a span's expert phase takes: as many as kGroupTokens holds.
    [[nodiscard]] std::uint32_t planned_group_chunks() const {
        if (planned_span_tokens() == 0) { return 1; }
        return std::clamp<std::uint32_t>(kGroupTokens / options.prefill_chunk, 1, kSpanChunks);
    }

    void allocate_span() {
        const std::uint32_t tokens = planned_span_tokens();
        if (tokens == 0) { return; }
        span_tokens       = tokens;
        const auto width  = std::size_t(config.hc_count) * config.hidden_size;
        RankBinding bind(device, 0);
        span_stack     = DeviceBuffer(std::size_t(tokens) * width * 4);
        span_positions = DeviceBuffer(std::size_t(tokens) * 4);
        span_rope      = DeviceBuffer(std::size_t(tokens) * 4);
        span_mixed     = DeviceBuffer(std::size_t(tokens) * config.hidden_size * 2);
        span_inject    = DeviceBuffer(std::size_t(tokens) * config.hc_count * 4);
        std::uint64_t bytes = span_stack.bytes + span_positions.bytes + span_rope.bytes +
                              span_mixed.bytes + span_inject.bytes;
        if (table) {
            span_row_bytes = std::size_t(config.ngram_heads()) *
                             ops::ngram_row_bytes(options.ngram->format);
            span_rows      = DeviceBuffer(std::size_t(tokens) * span_row_bytes);
            span_host_rows = std::make_unique<PinnedHostBuffer>(span_rows.bytes);
            bytes += span_rows.bytes;
        }
        span_group = planned_group_chunks();
        if (span_group > 1) {
            const std::size_t group = std::size_t(span_group) * options.prefill_chunk;
            const std::size_t pairs = group * config.num_experts_per_tok;
            span_ids     = DeviceBuffer(pairs * 4);
            span_weights = DeviceBuffer(pairs * 4);
            span_shared  = DeviceBuffer(group * 4);
            span_y       = DeviceBuffer(group * config.hidden_size * 4);
            bytes += span_ids.bytes + span_weights.bytes + span_shared.bytes + span_y.bytes;
        }
        memory.ranks[0].workspace_bytes += bytes;
    }

    // The chunk after the expert group of a span's `n` tokens that starts at chunk `first` of
    // `chunks` (c tokens each but the last). A group needs the full pool and kPrefetchTokens or
    // more, so that it runs from the layer's prefetched pool copy; a last chunk of
    // kVectorTokens or fewer runs alone, as forward() runs it.
    [[nodiscard]] std::size_t group_end(std::size_t first, std::size_t chunks, std::size_t n,
                                        std::size_t c) const {
        if (span_group == 1 || slot_pools.empty() || slot_pools[0].slots < config.num_experts) {
            return first + 1;
        }
        std::size_t last = std::min<std::size_t>(chunks, first + span_group);
        if (last > first + 1 && n - (last - 1) * c <= std::size_t(kVectorTokens)) { --last; }
        return std::min(last * c, n) - first * c >= std::size_t(kPrefetchTokens) ? last
                                                                                 : first + 1;
    }

    // The experts of a span's chunks first..last-1 in one wide call through the pool, into
    // span_y. rank.mixed holds the first chunk's MoE input and the others' follow it. Each chunk
    // is routed on its own, as forward() routes it, into the group's route planes.
    void run_group(const LayerPlan& plan, RankState& rank, std::size_t index, std::size_t first,
                   std::size_t last, std::size_t n, std::size_t c) {
        const MoePlan& m      = plan.moe;
        const cudaStream_t st = rank.stream;
        const auto h          = static_cast<std::int32_t>(config.hidden_size);
        const auto top        = static_cast<std::int32_t>(config.num_experts_per_tok);
        auto* const ids_p     = static_cast<std::int32_t*>(span_ids.p);
        auto* const weights_p = static_cast<float*>(span_weights.p);
        auto* const shared_p  = static_cast<float*>(span_shared.p);
        for (std::size_t j = first; j < last; ++j) {
            const auto t         = static_cast<std::int32_t>(std::min(c, n - j * c));
            const std::size_t at = (j - first) * c;
            const Tensor mixed(static_cast<std::byte*>(rank.mixed) + at * std::size_t(h) * 2,
                               DType::BF16, {h, t});
            Tensor ids(ids_p + at * std::size_t(top), DType::I32, {top, t});
            Tensor weights(weights_p + at * std::size_t(top), DType::FP32, {top, t});
            Tensor shared(shared_p + at, DType::FP32, {t});
            auto scope = rank.workspace->scope();
            ops::moe_route(mixed, m.router, m.shared_gate, *rank.workspace, ids, weights, shared,
                           st);
        }
        const auto t = static_cast<std::int32_t>(std::min(last * c, n) - first * c);
        Tensor ids(ids_p, DType::I32, {top, t});
        Tensor weights(weights_p, DType::FP32, {top, t});
        Tensor shared(shared_p, DType::FP32, {t});
        void* const own_y = rank.y;
        rank.y            = span_y.p;
        bool pooled       = false;
        try {
            pooled = run_through_slots(plan, rank, t, index, ids, weights, shared);
        } catch (...) {
            rank.y = own_y;
            throw;
        }
        rank.y = own_y;
        if (!pooled) { throw std::logic_error("qwen4_exp: a span group ran outside the pool"); }
    }

    // A prompt span of several chunks, layer by layer: every chunk passes a layer before any passes
    // the next, so a wide call's pool copy of the layer's uncached experts serves every chunk. The
    // arithmetic of each chunk is that of forward(); the logits are the last chunk's.
    void forward_span(std::uint32_t s, std::span<const std::int32_t> tokens,
                      std::uint32_t logit_rows, std::span<const std::int32_t> next) {
        const std::size_t n = tokens.size();
        const std::size_t c = options.prefill_chunk;
        if (span_tokens == 0 || n > span_tokens) {
            throw std::invalid_argument("qwen4_exp forward: tokens exceed the prompt span");
        }
        SequenceState& sequence = sequences.at(s);
        const std::uint32_t start = sequence.position;
        if (start + n > options.max_context) {
            throw std::invalid_argument("qwen4_exp forward: the sequence exceeds max_context");
        }
        const std::size_t prompt = sequence.media_rope.size() / 3;
        if (prompt != 0 && start < prompt) {
            throw std::invalid_argument("qwen4_exp forward: a media prompt prefills chunk by chunk");
        }
        const std::size_t chunks = (n + c - 1) / c;
        const auto last_count    = static_cast<std::uint32_t>(n - (chunks - 1) * c);
        check_tokens(tokens.subspan((chunks - 1) * c), logit_rows);
        for (std::size_t k = 0; k + 1 < chunks; ++k) { check_tokens(tokens.subspan(k * c, c), 1); }
        RankBinding bind(device, 0);
        RankState& rank      = ranks[0];
        const cudaStream_t st = rank.stream;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto hc        = static_cast<std::int32_t>(config.hc_count);
        const std::size_t width = std::size_t(hc) * std::size_t(h);
        const auto chunk_stack  = [&](std::size_t k) {
            return static_cast<float*>(span_stack.p) + k * c * width;
        };
        const auto count_of = [&](std::size_t k) {
            return static_cast<std::int32_t>(std::min(c, n - k * c));
        };
        settle_routes();
        forget_prefetches();
        place_pools(true);
        float* const own_stack = rank.stack;
        // The chunks' inputs, staged one after another as forward() stages them, kept in the
        // span planes; their embeddings start their stacks. The span's n-gram rows are read in
        // one batch while the layers before the first PLE layer run.
        const bool reads_rows =
            table && std::any_of(segments.begin(), segments.end(),
                                 [](const Segment& segment) { return segment.rows; });
        const std::size_t heads = config.ngram_heads();
        if (reads_rows) { span_row_ids.resize(n * heads); }
        for (std::size_t k = 0; k < chunks; ++k) {
            const std::int32_t t = count_of(k);
            const Part part{.sequence = &sequence, .column = 0, .count = t};
            stage_inputs(std::span(&part, 1), tokens.subspan(k * c, std::size_t(t)), false);
            if (reads_rows) {
                std::copy(row_ids.begin(), row_ids.end(),
                          span_row_ids.begin() + std::ptrdiff_t(k * c * heads));
            }
            Tensor mixed(rank.mixed, DType::BF16, {h, t});
            ops::embedding(Tensor(rank.ids, DType::I32, {t}), embedding_table, mixed, st);
            Tensor stack(chunk_stack(k), DType::FP32, {h, hc, t});
            ops::hyper_connection_expand(mixed, stack, st);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_positions.p) + k * c,
                                       rank.positions, std::size_t(t) * 4,
                                       cudaMemcpyDeviceToDevice, st));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(span_rope.p) + k * c, rank.rope,
                                       std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
            sequence.position += std::uint32_t(t);
        }
        sequence.position = start;
        if (reads_rows) {
            table->submit(span_row_ids,
                          std::span(static_cast<std::uint8_t*>(span_host_rows->data()),
                                    n * span_row_bytes));
            rows_waited = false;
        }
        // The next step's rows are read while the device runs this span.
        hint_next(sequence, next);
        void* const own_mixed   = rank.mixed;
        float* const own_inject = rank.inject;
        // Chunk k's MoE input and injection, kept between a layer's two phases.
        const auto chunk_planes = [&](std::size_t k) {
            rank.mixed  = static_cast<std::byte*>(span_mixed.p) + k * c * std::size_t(h) * 2;
            rank.inject = static_cast<float*>(span_inject.p) + k * c * std::size_t(hc);
        };
        try {
            // Each layer runs in two phases: attention over every chunk of the span, then the
            // experts over every chunk. The next layer's experts are copied into the pool while its
            // attention phase runs, so its first expert call rarely waits for them.
            for (std::size_t i = 0; i < layers.size(); ++i) {
                const LayerPlan& plan = layers[i];
                LayerState& state     = sequence.layers[i];
                if (plan.ple && !rows_waited) {
                    await_rows(rank);
                    CUDA_CHECK(cudaMemcpyAsync(span_rows.p, span_host_rows->data(),
                                               n * span_row_bytes, cudaMemcpyHostToDevice, st));
                }
                for (std::size_t k = 0; k < chunks; ++k) {
                    const std::int32_t t = count_of(k);
                    const Part part{.sequence = &sequence, .column = 0, .count = t};
                    CUDA_CHECK(cudaMemcpyAsync(rank.positions,
                                               static_cast<std::int32_t*>(span_positions.p) + k * c,
                                               std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    CUDA_CHECK(cudaMemcpyAsync(rank.rope,
                                               static_cast<std::int32_t*>(span_rope.p) + k * c,
                                               std::size_t(t) * 4, cudaMemcpyDeviceToDevice, st));
                    if (plan.ple) {
                        CUDA_CHECK(cudaMemcpyAsync(rank.rows,
                                                   static_cast<std::byte*>(span_rows.p) +
                                                       k * c * span_row_bytes,
                                                   std::size_t(t) * span_row_bytes,
                                                   cudaMemcpyDeviceToDevice, st));
                    }
                    rank.stack = chunk_stack(k);
                    chunk_planes(k);
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    Tensor mixed(rank.mixed, DType::BF16, {h, t});
                    Tensor inject(rank.inject, DType::FP32, {hc, t});
                    if (plan.ple) { run_ple(plan, state, rank, part); }
                    {
                        auto scope = rank.workspace->scope();
                        ops::hyper_connection_read(stack, plan.attn_hc.weights(),
                                                   config.rms_norm_eps, *rank.workspace, mixed,
                                                   &inject, st);
                    }
                    if (plan.gdn) {
                        run_gdn(plan, i, state, rank, part);
                    } else {
                        // The span holds the sequence at its start; chunk k begins k chunks on.
                        AttentionInputs in = target_inputs(i, state, rank, part);
                        in.first           = start + std::uint32_t(k * c);
                        run_qsa(plan, state, rank, part, in);
                    }
                    {
                        auto scope = rank.workspace->scope();
                        ops::hyper_connection_write_read(
                            stack, Tensor(rank.y, DType::BF16, {h, t}), inject,
                            plan.mlp_hc.weights(), config.rms_norm_eps, *rank.workspace, mixed,
                            &inject, st);
                    }
                }
                for (std::size_t k = 0; k < chunks;) {
                    const std::size_t last = group_end(k, chunks, n, c);
                    const auto t = static_cast<std::int32_t>(std::min(last * c, n) - k * c);
                    rank.stack = chunk_stack(k);
                    chunk_planes(k);
                    span_hold = last < chunks;
                    Tensor stack(rank.stack, DType::FP32, {h, hc, t});
                    Tensor inject(rank.inject, DType::FP32, {hc, t});
                    if (last == k + 1) {
                        run_moe(plan, rank, t, i);
                        ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, t}),
                                                    inject, st);
                    } else {
                        run_group(plan, rank, i, k, last, n, c);
                        ops::hyper_connection_write(stack, Tensor(span_y.p, DType::FP32, {h, t}),
                                                    inject, st);
                        if (last == chunks) {
                            // record_routes() reads the last chunk's routes from the layer's
                            // record, where run_moe() leaves them.
                            const auto top = std::size_t(config.num_experts_per_tok);
                            CUDA_CHECK(cudaMemcpyAsync(
                                route_records[i],
                                static_cast<std::int32_t*>(span_ids.p) + (chunks - 1 - k) * c * top,
                                top * std::size_t(count_of(chunks - 1)) * 4,
                                cudaMemcpyDeviceToDevice, st));
                        }
                    }
                    k = last;
                }
            }
            span_hold   = false;
            rank.mixed  = own_mixed;
            rank.inject = own_inject;
            // The head reads the last chunk's last rows.
            rank.stack       = chunk_stack(chunks - 1);
            const auto rows  = static_cast<std::int32_t>(logit_rows);
            const Tensor last(rank.stack + width * std::size_t(std::int32_t(last_count) - rows),
                              DType::FP32, {h, hc, rows});
            Tensor head_mixed(rank.mixed, DType::BF16, {h, rows});
            {
                auto scope = rank.workspace->scope();
                ops::hyper_connection_read(last, final_mixer.weights(), config.rms_norm_eps,
                                           *rank.workspace, head_mixed, nullptr, st);
            }
            Tensor logits(rank.logits, DType::BF16,
                          {static_cast<std::int32_t>(config.vocab_size), rows});
            project(head, head_mixed, logits, *rank.workspace, st);
            sequence.position = start + std::uint32_t(n);
            record_routes(last_count);
            // The MTP block catches up chunk by chunk over each chunk's own final stacks.
            if (mtp) {
                for (std::size_t k = 0; k < chunks; ++k) {
                    rank.stack = chunk_stack(k);
                    const auto t = count_of(k);
                    const Committed committed{&sequence, 0, t, start + std::uint32_t(k * c),
                                              tokens.subspan(k * c, std::size_t(t))};
                    mtp_catch_up(std::span(&committed, 1));
                }
                // The next pass's MTP cells start from the last chunk's stack, which it copied.
            }
        } catch (...) {
            rank.stack  = own_stack;
            rank.mixed  = own_mixed;
            rank.inject = own_inject;
            span_hold   = false;
            throw;
        }
        rank.stack = own_stack;
    }

    // Tokens one prompt call may take: a span with spans on, else a chunk.
    [[nodiscard]] std::uint32_t prompt_step() const noexcept {
        return span_tokens != 0 ? span_tokens : options.prefill_chunk;
    }

    void forward(std::uint32_t s, std::span<const std::int32_t> tokens, std::uint32_t logit_rows,
                 std::span<const std::int32_t> next) {
        if (next.size() > prompt_step()) {
            throw std::invalid_argument("qwen4_exp forward: more next tokens than a prompt step");
        }
        if (tokens.size() > options.prefill_chunk) {
            forward_span(s, tokens, logit_rows, next);
            return;
        }
        check_tokens(tokens, logit_rows);
        SequenceState& sequence      = sequences.at(s);
        const std::uint32_t position = sequence.position;
        const Part part{
            .sequence = &sequence, .column = 0, .count = static_cast<std::int32_t>(tokens.size())};
        pass(std::span(&part, 1), tokens, logit_rows);
        hint_next(sequence, next);
        if (mtp) {
            const Committed committed{&sequence, 0, part.count, position, tokens};
            mtp_catch_up(std::span(&committed, 1));
        }
    }

    // Whether forward_branches() runs: every sparse-attention layer keeps its KV on the device.
    [[nodiscard]] bool runs_branches() const noexcept { return stream_slots == 0; }

    void forward_branches(std::uint32_t s, const SequenceSnapshot& at,
                          std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths) {
        if (!runs_branches()) {
            throw std::logic_error("qwen4_exp forward_branches: a streamed KV layer");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(lengths.size()));
        std::vector<Part> parts;
        parts.reserve(lengths.size());
        std::uint32_t column = 0;
        for (const std::uint32_t length : lengths) {
            if (length == 0) {
                throw std::invalid_argument("qwen4_exp forward_branches: an empty branch");
            }
            parts.push_back({.sequence = &sequences.at(s),
                             .column   = static_cast<std::int32_t>(column),
                             .count    = static_cast<std::int32_t>(length)});
            column += length;
        }
        if (column != tokens.size()) {
            throw std::invalid_argument("qwen4_exp forward_branches: lengths unlike the tokens");
        }
        restore(s, at);
        branch_origin = &at;
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(parts.size()));
        } catch (...) {
            branch_origin = nullptr;
            throw;
        }
        branch_origin = nullptr;
        // The pass advanced the position by every branch; the state is the last branch's, which
        // the MTP block did not follow.
        sequences.at(s).position    = at.position + lengths.back();
        sequences.at(s).mtp_follows = false;
    }

    void forward_prompt_branches(std::uint32_t s, std::uint32_t prompt, SequenceSnapshot& at,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths) {
        if (!runs_branches()) {
            throw std::logic_error("qwen4_exp forward_prompt_branches: a streamed KV layer");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(lengths.size()));
        SequenceState& sequence = sequences.at(s);
        if (prompt == 0 || lengths.empty()) {
            throw std::invalid_argument("qwen4_exp forward_prompt_branches: no prompt or branch");
        }
        std::vector<Part> parts;
        parts.reserve(lengths.size() + 1);
        parts.push_back(
            {.sequence = &sequence, .column = 0, .count = static_cast<std::int32_t>(prompt)});
        std::uint32_t column = prompt;
        for (const std::uint32_t length : lengths) {
            if (length == 0) {
                throw std::invalid_argument("qwen4_exp forward_prompt_branches: an empty branch");
            }
            parts.push_back({.sequence = &sequence,
                             .column   = static_cast<std::int32_t>(column),
                             .count    = static_cast<std::int32_t>(length),
                             .offset   = prompt});
            column += length;
        }
        if (column != tokens.size()) {
            throw std::invalid_argument(
                "qwen4_exp forward_prompt_branches: lengths unlike the tokens");
        }
        size_snapshot(sequence, at);
        const std::uint32_t position = sequence.position;
        at.context                   = sequence.context;
        branch_origin                = &at;
        branch_capture               = &at;
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(lengths.size()));
        } catch (...) {
            branch_origin  = nullptr;
            branch_capture = nullptr;
            throw;
        }
        branch_origin  = nullptr;
        branch_capture = nullptr;
        // The MTP block followed none of the pass's tokens; its state goes along as snapshot()
        // keeps it.
        if (mtp) {
            RankBinding bind(device, model.head_rank());
            const cudaStream_t stream = ranks[model.head_rank()].stream;
            CUDA_CHECK(cudaMemcpyAsync(at.mtp[0].p, sequence.mtp.tail.p, at.mtp[0].bytes,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(at.mtp[1].p, sequence.mtp_pending.p, at.mtp[1].bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        synchronize_ranks();
        at.position          = position + prompt;
        at.mtp_follows       = false;
        sequence.position    = at.position + lengths.back();
        sequence.mtp_follows = false;
    }

    // Each part after a branch pass's first starts a layer from the origin's state of it: its
    // mixers' (Gated DeltaNet, convolution, indexer tail) before them, its PLE history before the
    // PLE (recurrent_regions order).
    static constexpr std::array<bool, 4> kMixerRegions{true, true, true, false};
    static constexpr std::array<bool, 4> kPleRegions{false, false, false, true};

    // The prompt part's state of a layer, into the snapshot the pass's branches start from.
    void capture_branch_layer(const LayerState& state, std::size_t i, RankState& rank,
                              const std::array<bool, 4>& which) {
        if (branch_capture == nullptr) { return; }
        const auto regions = recurrent_regions(state);
        auto& kept         = branch_capture->layers.at(i).buffers;
        for (std::size_t k = 0; k < regions.size(); ++k) {
            if (!which[k] || regions[k].first == nullptr || kept[k].bytes == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(kept[k].p, regions[k].first, kept[k].bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
        }
    }

    void restore_branch_layer(const LayerState& state, std::size_t i, RankState& rank,
                              const std::array<bool, 4>& which) {
        if (branch_origin == nullptr) { return; }
        const auto regions = recurrent_regions(state);
        const auto& kept   = branch_origin->layers.at(i).buffers;
        for (std::size_t k = 0; k < regions.size(); ++k) {
            if (!which[k] || regions[k].first == nullptr || kept[k].bytes == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(regions[k].first, kept[k].p, kept[k].bytes,
                                       cudaMemcpyDeviceToDevice, rank.stream));
        }
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
        std::vector<Committed> committed;
        for (const Part& part : parts) {
            committed.push_back({part.sequence, part.column, 1, part.sequence->position,
                                 tokens.subspan(std::size_t(part.column), 1)});
        }
        pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
        if (mtp) { mtp_catch_up(committed); }
    }

    void warm_up() {
        // Any token; the n-gram rows it addresses are read as a request's would be. The expert
        // cache does not count the warm-up's routes, which would fill it with one token's experts.
        constexpr std::int32_t kToken = 13;
        const bool device_experts     = model.options().experts == ExpertResidency::Device;
        const std::uint32_t room      = options.max_context > 1 ? options.max_context - 1 : 1;
        const std::uint32_t chunk = std::min(device_experts ? options.prefill_chunk : 16U, room);
        std::vector<std::int32_t> tokens(std::max<std::uint32_t>(chunk, 16), kToken);
        const auto prefill = [&](std::uint32_t count) {
            reset(0);
            count = std::min(count, room);
            forward(0, std::span(tokens).first(count), 1, {});
        };
        warming = true;
        prefill(1);
        if (verify_width > 1) { prefill(verify_width); }
        prefill(chunk);
        reset(0);
        prefill(2);
        const std::uint32_t sequence = 0;
        if (sequences.front().position + 1 < options.max_context) {
            decode(std::span(&sequence, 1), std::span(tokens).first(1));
        }
        if (mtp && can_draft(0) &&
            sequences.front().position + verify_width <= options.max_context) {
            std::vector<std::int32_t> drafts(options.draft_tokens);
            draft(std::span(&sequence, 1), std::span(tokens).first(1), drafts);
            std::vector<std::int32_t> round{kToken};
            round.insert(round.end(), drafts.begin(), drafts.end());
            verify(std::span(&sequence, 1), round);
            const std::uint32_t kept = 1;
            commit(std::span(&sequence, 1), std::span(&kept, 1));
        }
        reset(0);
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
        warming = false;
    }

    void grow_expert_cache(std::uint64_t keep) {
        if (!cache) { return; }
        std::vector<std::uint64_t> bytes(device.size(), 0);
        for (std::size_t r = 0; r < device.size(); ++r) {
            RankBinding bind(device, r);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::size_t free_bytes = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
            bytes[r] = free_bytes > keep ? free_bytes - keep : 0;
        }
        const auto added = cache->grow(bytes);
        for (std::size_t r = 0; r < added.size(); ++r) {
            memory.ranks[r].expert_cache_bytes += added[r];
            memory.expert_cache_bytes += added[r];
        }
        std::vector<ExpertCache::Loan> loans;
        for (std::size_t r = 0; r < slot_pools.size(); ++r) {
            for (const auto& buffer : slot_pools[r].buffers) {
                loans.push_back({.rank = r, .base = buffer.storage.p, .bytes = buffer.storage.bytes});
            }
        }
        if (!loans.empty()) {
            const auto lent = cache->lend(loans);
            for (std::size_t r = 0; r < lent.size(); ++r) {
                memory.ranks[r].expert_cache_bytes += lent[r];
                memory.ranks[r].workspace_bytes -= lent[r];
                memory.expert_cache_bytes += lent[r];
                memory.workspace_bytes -= lent[r];
            }
            pools_lent = true;
        }
        if (misses) {
            for (std::size_t i = 0; i < expert_layers().size(); ++i) {
                misses->set_resident(i, cache->storage(i));
            }
        }
    }

    // ---- MTP speculative decoding ------------------------------------------------------------

    void require_mtp(const char* what) const {
        if (!mtp) {
            throw std::logic_error(std::string("qwen4_exp ") + what +
                                   ": the executor has no drafts");
        }
    }

    [[nodiscard]] bool can_draft(std::uint32_t s) const {
        const SequenceState& sequence = sequences.at(s);
        return mtp && sequence.mtp_follows && sequence.position > 0;
    }

    // Uploads an MTP pass's token ids and cell positions to the head rank's chain buffers, at the
    // given offsets.
    void stage_mtp(std::span<const std::int32_t> ids, std::size_t ids_at,
                   std::span<const std::int32_t> positions, std::size_t positions_at) {
        RankState& rank = ranks[model.head_rank()];
        CUDA_CHECK(cudaEventSynchronize(mtp_staged));
        auto* host                 = static_cast<std::int32_t*>(mtp_staging->data());
        const std::size_t capacity = mtp_ids.bytes / 4;
        std::memcpy(host, ids.data(), ids.size() * 4);
        std::memcpy(host + capacity, positions.data(), positions.size() * 4);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(mtp_ids.p) + ids_at, host,
                                   ids.size() * 4, cudaMemcpyHostToDevice, rank.stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(mtp_positions.p) + positions_at,
                                   host + capacity, positions.size() * 4, cudaMemcpyHostToDevice,
                                   rank.stream));
        CUDA_CHECK(cudaEventRecord(mtp_staged, rank.stream));
    }

    // Puts `count` FP32 stacks into an MTP pass's input plane (BF16, plane a of the head rank) from
    // column `column` on.
    void stage_hidden(const void* source, std::int32_t count, std::int32_t column) {
        RankState& rank  = ranks[model.head_rank()];
        const auto width = static_cast<std::int32_t>(config.hc_count * config.hidden_size);
        const Tensor from(const_cast<void*>(source), DType::FP32, {width, count});
        Tensor to(columns(rank.a, width, column, 2), DType::BF16, {width, count});
        ops::cast_fp32_to_bf16(from, to, rank.stream);
    }

    // The token embeddings of `n` cells into `out` (BF16 [hidden, n] on the head rank) from their
    // ids at `ids` (device I32 [n], head rank). The table lives on rank 0, so a split model looks
    // the rows up there and copies them over.
    void embed_cells(const std::int32_t* ids, std::int32_t n, void* out) {
        const std::size_t head = model.head_rank();
        const auto h           = static_cast<std::int32_t>(config.hidden_size);
        RankState& last        = ranks[head];
        if (head == 0) {
            Tensor rows(out, DType::BF16, {h, n});
            ops::embedding(Tensor(const_cast<std::int32_t*>(ids), DType::I32, {n}), embedding_table,
                           rows, last.stream);
            return;
        }
        RankState& first = ranks[0];
        CUDA_CHECK(cudaEventRecord(last.done, last.stream));
        {
            RankBinding bind(device, 0);
            CUDA_CHECK(cudaStreamWaitEvent(first.stream, last.done, 0));
            CUDA_CHECK(cudaMemcpyPeerAsync(first.ids, device.rank(0).device, ids,
                                           device.rank(head).device, std::size_t(n) * 4,
                                           first.stream));
            Tensor rows(first.mixed, DType::BF16, {h, n});
            ops::embedding(Tensor(first.ids, DType::I32, {n}), embedding_table, rows, first.stream);
            CUDA_CHECK(cudaEventRecord(first.done, first.stream));
        }
        CUDA_CHECK(cudaStreamWaitEvent(last.stream, first.done, 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(out, device.rank(head).device, first.mixed,
                                       device.rank(0).device, std::size_t(h) * n * 2, last.stream));
        // Catch-up returns with this copy queued. The next prompt chunk can start on rank 0
        // before the head consumes its embeddings, so protect the source until the copy ends.
        CUDA_CHECK(cudaEventRecord(last.done, last.stream));
        {
            RankBinding bind(device, 0);
            CUDA_CHECK(cudaStreamWaitEvent(first.stream, last.done, 0));
        }
    }

    // One MTP pass on the head rank over `n` cells whose starting stacks are in its input plane,
    // their token ids at `ids` and positions at `positions` (device, head rank): the token's
    // normalised embedding, projected, joins every stream of the normalised stack (one norm over
    // all of a cell's streams, then a projection stream by stream) with unit weight, and the
    // block's sparse-attention layer advances that stack, which stays in the head rank's stack
    // plane. Drafting cells run the indexer on the sequences' scratch tails.
    void mtp_pass(std::span<const MtpPart> parts, std::int32_t n, const std::int32_t* ids,
                  const std::int32_t* positions, bool drafting) {
        RankState& rank      = ranks[model.head_rank()];
        const cudaStream_t s = rank.stream;
        WorkspaceArena& ws   = *rank.workspace;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const auto hc        = static_cast<std::int32_t>(config.hc_count);
        const float eps      = config.rms_norm_eps;
        embed_cells(ids, n, rank.mixed);
        Tensor embedded(rank.mixed, DType::BF16, {h, n});
        Tensor embedded_norm(rank.d, DType::BF16, {h, n});
        ops::rmsnorm(embedded, mtp->embedding_norm, eps, true, embedded_norm, s);
        Tensor token(rank.e, DType::BF16, {h, n});
        project(mtp->fc_embedding, embedded_norm, token, ws, s);
        Tensor hidden(rank.a, DType::BF16, {hc * h, n});
        Tensor hidden_norm(rank.b, DType::BF16, {hc * h, n});
        ops::rmsnorm(hidden, mtp->hidden_norm, eps, true, hidden_norm, s);
        Tensor streams(rank.c, DType::BF16, {h, hc * n});
        project(mtp->fc_hidden, Tensor(rank.b, DType::BF16, {h, hc * n}), streams, ws, s);
        Tensor stack(rank.stack, DType::FP32, {h, hc, n});
        ops::hyper_connection_expand(token, Tensor(rank.c, DType::BF16, {h, hc, n}), stack, s);

        const LayerPlan& plan = mtp->layer;
        Tensor mixed(rank.mixed, DType::BF16, {h, n});
        Tensor inject(rank.inject, DType::FP32, {hc, n});
        {
            auto scope = ws.scope();
            ops::hyper_connection_read(stack, plan.attn_hc.weights(), eps, ws, mixed, &inject, s);
        }
        // MTP cells rotate on the one text axis.
        const bool media = std::exchange(mrope, false);
        for (const MtpPart& part : parts) {
            SequenceState& sequence = *part.sequence;
            const Part cells{.sequence = &sequence, .column = part.column, .count = part.count};
            run_qsa(plan, sequence.mtp, rank, cells,
                    {.positions = positions + part.column,
                     .rope      = positions + part.column,
                     .index     = nullptr,
                     .tail      = drafting ? sequence.mtp_scratch_tail.p : sequence.mtp.tail.p,
                     .first     = part.first});
        }
        mrope = media;
        {
            auto scope = ws.scope();
            ops::hyper_connection_write_read(stack, Tensor(rank.y, DType::BF16, {h, n}), inject,
                                             plan.mlp_hc.weights(), eps, ws, mixed, &inject, s);
        }
        run_moe(plan, rank, n, layers.size());
        ops::hyper_connection_write(stack, Tensor(rank.y, DType::FP32, {h, n}), inject, s);
        record_mtp_routes(n);
    }

    // The MTP layer's routes for the expert cache, gathered over the passes between two target
    // passes (beyond a chunk of them the rest go uncounted).
    void record_mtp_routes(std::int32_t n) {
        if (!cache || warming || capturing_draft ||
            mtp_routes + std::uint32_t(n) > options.prefill_chunk) {
            return;
        }
        RankState& rank           = ranks[model.head_rank()];
        const std::uint64_t top   = config.num_experts_per_tok;
        const std::uint64_t pairs = top * options.prefill_chunk;
        auto* host = static_cast<std::int32_t*>(route_host->data()) + layers.size() * pairs +
                     top * mtp_routes;
        CUDA_CHECK(cudaMemcpyAsync(host, route_records[layers.size()], std::size_t(top) * n * 4,
                                   cudaMemcpyDeviceToHost, rank.stream));
        CUDA_CHECK(cudaEventRecord(rank.routes, rank.stream));
        mtp_routes += std::uint32_t(n);
    }

    // The greedy draft of each of the `n` single-cell parts of the last MTP pass: the block's
    // final mixer, the head, and the argmax over the public tokens into `out` (device I32 [n],
    // head rank).
    void mtp_head(std::int32_t n, std::int32_t* out, std::int32_t* second = nullptr,
                  float* logprobs = nullptr) {
        RankState& rank      = ranks[model.head_rank()];
        const cudaStream_t s = rank.stream;
        WorkspaceArena& ws   = *rank.workspace;
        const auto h         = static_cast<std::int32_t>(config.hidden_size);
        const Tensor stack(rank.stack, DType::FP32,
                           {h, static_cast<std::int32_t>(config.hc_count), n});
        Tensor mixed(rank.mixed, DType::BF16, {h, n});
        {
            auto scope = ws.scope();
            ops::hyper_connection_read(stack, mtp->final_mixer.weights(), config.rms_norm_eps, ws,
                                       mixed, nullptr, s);
        }
        Tensor logits(rank.logits, DType::BF16, {static_cast<std::int32_t>(config.vocab_size), n});
        project(mtp->head, mixed, logits, ws, s);
        Tensor tokens(out, DType::I32, {n});
        if (second) {
            Tensor alternative(second, DType::I32, {n});
            ops::argmax_top2(logits, tokens, alternative, domain, s);
        } else {
            ops::argmax(logits, tokens, domain, s);
        }
        if (logprobs) {
            Tensor probabilities(logprobs, DType::FP32, {n});
            ops::target_logprobs(logits, tokens, domain, probabilities, s);
        }
    }

    // Advances the MTP block over what a target pass committed: for each sequence the cells from
    // its last position before the pass (whose stack it kept) to the second to last committed
    // position, each pairing that position's stack with the next token, and keeps the last
    // committed position's stack for the next cell. The target's stacks are still in the head
    // rank's stack plane.
    void mtp_catch_up(std::span<const Committed> committed) {
        RankBinding bind(device, model.head_rank());
        RankState& rank  = ranks[model.head_rank()];
        const auto width = std::size_t(config.hc_count) * config.hidden_size;
        std::vector<MtpPart> parts;
        std::vector<std::int32_t> ids, positions;
        std::int32_t n = 0;
        for (const Committed& c : committed) {
            SequenceState& sequence = *c.sequence;
            if (!sequence.mtp_follows || c.count == 0) { continue; }
            // A sequence's first position has no stack before it.
            const bool kept           = c.position > 0;
            const std::int32_t cells  = c.count - (kept ? 0 : 1);
            const std::uint32_t first = kept ? c.position - 1 : 0;
            if (cells == 0) { continue; }
            if (kept) { stage_hidden(sequence.mtp_pending.p, 1, n); }
            if (c.count > 1) {
                stage_hidden(rank.stack + width * std::size_t(c.column), c.count - 1,
                             n + (kept ? 1 : 0));
            }
            parts.push_back({&sequence, n, cells, first});
            ids.insert(ids.end(), c.tokens.begin() + (kept ? 0 : 1), c.tokens.end());
            for (std::int32_t q = 0; q < cells; ++q) {
                positions.push_back(static_cast<std::int32_t>(first) + q);
            }
            n += cells;
        }
        // The stacks the next cells start from, once the casts above have read the old ones.
        for (const Committed& c : committed) {
            if (!c.sequence->mtp_follows || c.count == 0) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(c.sequence->mtp_pending.p,
                                       rank.stack + width * std::size_t(c.column + c.count - 1),
                                       width * 4, cudaMemcpyDeviceToDevice, rank.stream));
        }
        if (n == 0) { return; }
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_mtp(ids, 0, positions, 0);
        if (misses) { misses->keep_alive(); }
        mtp_pass(parts, n, static_cast<const std::int32_t*>(mtp_ids.p),
                 static_cast<const std::int32_t*>(mtp_positions.p), false);
        if (hybrid) { hybrid->finish(); }
    }

    std::uint32_t draft(std::span<const std::uint32_t> batch, std::span<const std::int32_t> anchors,
                        std::span<std::int32_t> out, std::span<std::uint32_t> extents = {},
                        std::uint32_t steps = 0) {
        require_mtp("draft");
        const std::size_t b      = batch.size();
        const std::size_t stride = options.draft_tokens;
        const std::size_t k      = steps == 0 ? stride : steps;
        if (b == 0 || b > sequences.size() || anchors.size() != b || out.size() != b * stride ||
            (!extents.empty() && extents.size() != b) || k > stride) {
            throw std::invalid_argument("qwen4_exp draft: one anchor and its drafts per sequence");
        }
        std::vector<SequenceState*> drafting;
        for (std::size_t j = 0; j < b; ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            if (std::find(drafting.begin(), drafting.end(), sequence) != drafting.end()) {
                throw std::invalid_argument("qwen4_exp draft: a sequence appears twice");
            }
            if (!can_draft(batch[j])) {
                throw std::invalid_argument("qwen4_exp draft: the sequence's MTP state does not "
                                            "follow its tokens");
            }
            if (sequence->position + verify_width > options.max_context) {
                throw std::invalid_argument("qwen4_exp draft: the round passes max_context");
            }
            if (anchors[j] < 0 || std::uint32_t(anchors[j]) >= config.vocab_size) {
                throw std::invalid_argument("qwen4_exp draft: anchor outside the vocabulary");
            }
            drafting.push_back(sequence);
        }
        RankBinding bind(device, model.head_rank());
        RankState& rank = ranks[model.head_rank()];
        // Step i's cell of a sequence sits at its position - 1 + i and pairs the stack before it
        // (the kept one at step 0) with the step's token (the anchor at step 0).
        std::vector<std::int32_t> positions(k * b);
        std::vector<MtpPart> parts;
        for (std::size_t j = 0; j < b; ++j) {
            const std::uint32_t first = drafting[j]->position - 1;
            for (std::size_t i = 0; i < k; ++i) {
                positions[i * b + j] = static_cast<std::int32_t>(first + i);
            }
            parts.push_back({drafting[j], std::int32_t(j), 1, first});
        }
        if (hybrid) { hybrid->begin(cancelled, !warming); }
        stage_mtp(anchors, 0, positions, 0);
        if (draft_prefetch) {
            std::vector<NgramContext> contexts;
            for (const auto* sequence : drafting) { contexts.push_back(sequence->context); }
            draft_prefetch->begin(std::move(contexts), anchors);
        }
        auto* ids        = static_cast<std::int32_t*>(mtp_ids.p);
        const auto* at   = static_cast<const std::int32_t*>(mtp_positions.p);
        const auto n     = static_cast<std::int32_t>(b);
        const auto second = [&](std::size_t i) {
            return draft_prefetch ? static_cast<std::int32_t*>(ngram_prefetch_tokens.p) + i * b
                                  : nullptr;
        };
        auto* confidence = static_cast<float*>(mtp_logprobs.p);
        // Step i: the MTP pass over the step's cells (the anchors at step 0, with the stacks the
        // sequences kept) and the head's pick for the next one.
        const auto step = [&](std::size_t i) {
            if (i == 0) {
                for (std::size_t j = 0; j < b; ++j) {
                    CUDA_CHECK(cudaMemcpyAsync(drafting[j]->mtp_scratch_tail.p,
                                               drafting[j]->mtp.tail.p, drafting[j]->mtp.tail.bytes,
                                               cudaMemcpyDeviceToDevice, rank.stream));
                    stage_hidden(drafting[j]->mtp_pending.p, 1, std::int32_t(j));
                }
            } else {
                stage_hidden(rank.stack, n, 0);
            }
            mtp_pass(parts, n, ids + i * b, at + i * b, true);
            mtp_head(n, ids + (i + 1) * b, second(i), confidence ? confidence + i * b : nullptr);
        };
        const auto prefetch = [&](std::size_t i) {
            if (draft_prefetch) {
                draft_prefetch->enqueue(i, ids + (i + 1) * b, second(i), rank.stream);
            }
        };
        // Tokens and positions stay in the staged buffers. Native hybrid experts use the same
        // captured exchange as text decode; GGUF host experts replay too, without counting the
        // draft steps' routes. A graph is captured at its second use.
        const bool graph = options.cuda_graphs && b == 1 && device.size() == 1 && !stream;
        const auto replay = [&](SequenceState::DraftGraph& slot, const auto& body) {
            if (graph && !slot.chain.ready() && slot.runs++ > 0) {
                DecodeGraphDefinition definition;
                capturing_draft = true;
                try {
                    definition.capture(rank.stream, body);
                } catch (...) {
                    capturing_draft = false;
                    throw;
                }
                capturing_draft = false;
                slot.chain.instantiate(definition);
            }
            if (graph && slot.chain.ready()) {
                slot.chain.launch(rank.stream);
            } else {
                body();
            }
        };
        if (misses) { misses->keep_alive(); }
        auto* host     = static_cast<std::int32_t*>(mtp_staging->data());
        auto* logprobs = reinterpret_cast<float*>(host + mtp_ids.bytes / 4);
        std::vector<std::uint32_t> lengths(b, static_cast<std::uint32_t>(k));
        std::size_t ran = k;
        if (confidence == nullptr || extents.empty()) {
            replay(drafting.front()->draft.at(k), [&] {
                for (std::size_t i = 0; i < k; ++i) {
                    step(i);
                    prefetch(i);
                }
                if (draft_prefetch) { draft_prefetch->join(rank.stream); }
            });
            // The expert cache takes the last verification's routes while the device drafts; the
            // table uploads it queues follow the draft on the stream, before the next pass.
            settle_routes();
        } else {
            // The chain ends at the first step after which every sequence has drafted at or below
            // the floor: each step runs from its own graph, and the host reads the step's
            // confidences before it starts the next.
            const float floor = std::log(options.draft_min_p);
            std::vector<char> open(b, 1);
            for (std::size_t i = 0; i < k; ++i) {
                replay(drafting.front()->step.at(i), [&] { step(i); });
                prefetch(i);
                CUDA_CHECK(cudaMemcpyAsync(logprobs + i * b, confidence + i * b,
                                           b * sizeof(float), cudaMemcpyDeviceToHost, rank.stream));
                // The expert cache takes the last verification's routes while the first step
                // runs; the MTP block's table uploads it queues precede the next step.
                if (i == 0) { settle_routes(); }
                CUDA_CHECK(cudaStreamSynchronize(rank.stream));
                bool more = false;
                for (std::size_t j = 0; j < b; ++j) {
                    if (!std::isfinite(logprobs[i * b + j])) {
                        throw std::runtime_error("qwen4_exp: non-finite MTP draft confidence");
                    }
                    if (open[j] && logprobs[i * b + j] <= floor) {
                        open[j]    = 0;
                        lengths[j] = static_cast<std::uint32_t>(i + 1);
                    }
                    more = more || open[j];
                }
                if (!more) {
                    ran = i + 1;
                    break;
                }
            }
            if (draft_prefetch) { draft_prefetch->join(rank.stream); }
        }
        CUDA_CHECK(cudaMemcpyAsync(host, ids + b, ran * b * 4, cudaMemcpyDeviceToHost, rank.stream));
        CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        if (hybrid) { hybrid->finish(); }
        if (draft_prefetch) { draft_prefetch->check(); }
        for (std::size_t j = 0; j < b; ++j) {
            // Past the steps run no draft exists; no verification reaches those columns.
            for (std::size_t i = 0; i < k; ++i) {
                out[j * stride + i] = i < ran ? host[i * b + j] : -1;
            }
            if (!extents.empty()) { extents[j] = lengths[j]; }
        }
        return static_cast<std::uint32_t>(ran);
    }

    void verify(std::span<const std::uint32_t> batch, std::span<const std::int32_t> tokens) {
        require_mtp("verify");
        const std::size_t b = batch.size();
        const std::size_t w = b == 0 ? 0 : tokens.size() / b;
        if (b == 0 || b > sequences.size() || w < 2 || w > verify_width || tokens.size() != b * w) {
            throw std::invalid_argument("qwen4_exp verify: an anchor and its drafts per sequence");
        }
        check_tokens(tokens, static_cast<std::uint32_t>(tokens.size()));
        active_verify_width = static_cast<std::uint32_t>(w);
        std::vector<Part> parts;
        verified.clear();
        for (std::size_t j = 0; j < b; ++j) {
            SequenceState* sequence = &sequences.at(batch[j]);
            if (std::find(verified.begin(), verified.end(), sequence) != verified.end()) {
                verified.clear();
                throw std::invalid_argument("qwen4_exp verify: a sequence appears twice");
            }
            parts.push_back({.sequence = sequence,
                             .column   = static_cast<std::int32_t>(j * w),
                             .count    = static_cast<std::int32_t>(w)});
            sequence->verify_context = sequence->context;
            sequence->verify_tokens.assign(tokens.begin() + std::ptrdiff_t(j * w),
                                           tokens.begin() + std::ptrdiff_t((j + 1) * w));
            verified.push_back(sequence);
        }
        verifying = true;
        try {
            pass(parts, tokens, static_cast<std::uint32_t>(tokens.size()));
            if (draft_prefetch) {
                RankBinding bind(device, model.head_rank());
                RankState& rank = ranks[model.head_rank()];
                std::vector<NgramContext> contexts;
                for (const auto* sequence : verified) { contexts.push_back(sequence->context); }
                draft_prefetch->begin_verification(std::move(contexts), tokens);
                const auto count = static_cast<std::int32_t>(tokens.size());
                auto* first = static_cast<std::int32_t*>(ngram_prefetch_tokens.p);
                auto* second = first + std::size_t(options.sequences) * w;
                Tensor logits(rank.logits, DType::BF16,
                              {static_cast<std::int32_t>(config.vocab_size), count});
                Tensor winners(first, DType::I32, {count}), alternatives(second, DType::I32, {count});
                ops::argmax_top2(logits, winners, alternatives, domain, rank.stream);
                draft_prefetch->enqueue_verification(first, second, rank.stream);
                draft_prefetch->join(rank.stream);
            }
        } catch (...) {
            verifying = false;
            verified.clear();
            throw;
        }
        verifying = false;
    }

    void commit(std::span<const std::uint32_t> batch, std::span<const std::uint32_t> kept) {
        require_mtp("commit");
        const std::uint32_t w = active_verify_width;
        if (batch.size() != verified.size() || kept.size() != batch.size()) {
            throw std::invalid_argument("qwen4_exp commit: the last verification's sequences");
        }
        for (std::size_t j = 0; j < batch.size(); ++j) {
            if (&sequences.at(batch[j]) != verified[j] || kept[j] == 0 || kept[j] > w) {
                throw std::invalid_argument("qwen4_exp commit: the last verification's sequences, "
                                            "1 to draft_tokens + 1 tokens each");
            }
        }
        if (draft_prefetch) {
            RankBinding bind(device, model.head_rank());
            // Captured events encode graph dependencies; wait on the joined stream before
            // reading host callback state. Host synchronization of a captured event is invalid.
            CUDA_CHECK(cudaStreamSynchronize(ranks[model.head_rank()].stream));
            draft_prefetch->check();
        }
        const auto di = static_cast<std::int32_t>(config.indexer_head_dim);
        const std::uint64_t index_rows =
            std::uint64_t(config.indexer_n_heads + 1) * config.indexer_head_dim;
        const std::uint64_t columns = std::uint64_t(verify_width) * options.sequences;
        const auto stack_width      = std::int32_t(config.hc_count * config.hidden_size);
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerPlan& plan = layers[i];
            RankBinding bind(device, plan.rank);
            RankState& rank = ranks[plan.rank];
            for (std::size_t j = 0; j < verified.size(); ++j) {
                LayerState& state = verified[j]->layers[i];
                const auto m      = static_cast<std::int32_t>(kept[j]);
                if (plan.qsa) {
                    // The indexer's tail advances over the kept positions' recorded projections;
                    // the verification's first position is still staged.
                    const QsaPlan& a = *plan.qsa;
                    const Tensor projection(static_cast<std::byte*>(rank.index_records.p) +
                                                (qsa_local[i] * columns + j * w) * index_rows * 2,
                                            DType::BF16, {a.index.rows, m});
                    const Tensor first(rank.positions + j * w, DType::I32, {1});
                    Tensor pooled(state.pooled.p, DType::FP32,
                                  {di, static_cast<std::int32_t>(pooled_slots)});
                    Tensor tail(state.tail.p, DType::FP32,
                                {di, static_cast<std::int32_t>(config.indexer_compress_ratio - 1)});
                    const ops::QsaIndexerWeights iw{&a.index_query_norm, &a.index_key_norm};
                    ops::qsa_indexer_append(projection, first, iw, config.rms_norm_eps,
                                            options.rope_yarn, pooled, tail, rank.stream);
                }
                if (plan.ple) {
                    // The PLE history advances over the kept positions' recorded input.
                    Tensor history(
                        state.history.p, DType::FP32,
                        {stack_width, static_cast<std::int32_t>((config.ple_conv_kernel_size - 1) *
                                                                config.ngram.ngram_size)});
                    const Tensor normalized(static_cast<float*>(rank.ple_records.p) +
                                                std::size_t(stack_width) * j * w,
                                            DType::FP32, {stack_width, m});
                    ops::ple_history_advance(history, normalized, rank.stream);
                }
            }
        }
        // The Gated DeltaNet layers replay the kept transitions into each sequence's slot.
        std::vector<ops::GdnReplayFoldRow> rows;
        for (std::size_t j = 0; j < verified.size(); ++j) {
            const auto slot = static_cast<std::int32_t>(verified[j]->slot);
            rows.push_back({slot, slot, static_cast<std::int32_t>(kept[j])});
        }
        for (RankState& rank : ranks) {
            if (!rank.fold) { continue; }
            RankBinding bind(device, rank.rank);
            rank.fold->execute(rows, static_cast<std::int32_t>(w), rank.stream);
        }
        std::vector<Committed> committed;
        std::vector<std::uint64_t> scratch;
        for (std::size_t j = 0; j < verified.size(); ++j) {
            SequenceState& sequence = *verified[j];
            const std::span<const std::int32_t> tokens(sequence.verify_tokens.data(), kept[j]);
            if (table) {
                scratch.resize(tokens.size() * config.ngram_heads());
                ngram_row_ids(ngram, tokens, config.eos_token_id, config.vocab_size,
                              sequence.context, scratch);
            }
            committed.push_back({&sequence, static_cast<std::int32_t>(j * w),
                                 static_cast<std::int32_t>(kept[j]), sequence.position, tokens});
            sequence.position += kept[j];
        }
        verified.clear();
        mtp_catch_up(committed);
    }

    std::uint64_t snapshot_bytes() const {
        const SequenceState& sequence = sequences.at(0);
        std::uint64_t bytes = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            for (const auto& region : recurrent_regions(sequence.layers[i])) {
                if (region.first != nullptr) { bytes += region.second; }
            }
        }
        if (mtp) { bytes += sequence.mtp.tail.bytes + sequence.mtp_pending.bytes; }
        return bytes;
    }

    void snapshot(std::uint32_t s, SequenceSnapshot& out) {
        SequenceState& sequence = sequences.at(s);
        size_snapshot(sequence, out);
        copy_state(sequence, out, true);
        out.position    = sequence.position;
        out.context     = sequence.context;
        out.mtp_follows = mtp && sequence.mtp_follows;
    }

    // Gives `out` its buffers, sized as `sequence`'s regions, unless it has them.
    void size_snapshot(const SequenceState& sequence, SequenceSnapshot& out) {
        if (out.layers.size() != layers.size()) {
            out.layers.clear();
            for (std::size_t i = 0; i < layers.size(); ++i) {
                RankBinding bind(device, layers[i].rank);
                SequenceSnapshot::Layer layer;
                for (const auto& [data, bytes] : recurrent_regions(sequence.layers[i])) {
                    layer.buffers.push_back(data != nullptr ? DeviceBuffer(bytes) : DeviceBuffer());
                }
                out.layers.push_back(std::move(layer));
            }
        }
        if (mtp && out.mtp.empty()) {
            RankBinding bind(device, model.head_rank());
            out.mtp.emplace_back(sequence.mtp.tail.bytes);
            out.mtp.emplace_back(sequence.mtp_pending.bytes);
        }
    }

    void restore(std::uint32_t s, const SequenceSnapshot& from) {
        if (from.layers.size() != layers.size()) {
            throw std::invalid_argument("qwen4_exp restore: the snapshot holds no state");
        }
        SequenceState& sequence = sequences.at(s);
        copy_state(sequence, const_cast<SequenceSnapshot&>(from), false);
        sequence.position = from.position;
        sequence.context  = from.context;
        // A snapshot taken before the MTP block followed the sequence leaves it behind.
        sequence.mtp_follows = mtp && !from.mtp.empty() && from.mtp_follows;
        // Snapshots hold text prompts only: a media prompt is not reused.
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
    }

    // One region of a sequence image, on its rank: device memory, or a streamed layer's host copy.
    struct ImageRegion {
        void* device      = nullptr;
        std::size_t bytes = 0;
        std::size_t rank  = 0;
    };

    // A sequence image's regions at `position`, in the image's byte order: each layer's recurrent
    // state (taken from `at` when given), its KV pages and pooled keys of the positions before
    // `position`, then the MTP block's. Positions past `position` in its last page and pooled
    // block come along; the sequence rewrites them before it reads them.
    [[nodiscard]] std::vector<ImageRegion> image_regions(const SequenceState& sequence,
                                                         const SequenceSnapshot* at,
                                                         std::uint32_t position) const {
        std::vector<ImageRegion> out;
        const std::uint64_t used =
            (std::uint64_t(position) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::uint64_t blocks = position / config.indexer_compress_ratio;
        const auto attention       = [&](const LayerState& state, std::size_t rank) {
            for (const KvPlane& plane : state.kv) {
                if (plane.p == nullptr) { continue; }
                out.push_back({plane.p, std::size_t(plane.bytes / pages * used), rank});
            }
            out.push_back(
                {state.pooled.p, std::size_t(config.indexer_head_dim) * 4 * blocks, rank});
        };
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const LayerState& state = sequence.layers[i];
            const auto regions      = recurrent_regions(state);
            for (std::size_t k = 0; k < regions.size(); ++k) {
                if (regions[k].first == nullptr) { continue; }
                void* source = at != nullptr ? at->layers[i].buffers[k].p : regions[k].first;
                out.push_back({source, regions[k].second, layers[i].rank});
            }
            if (layers[i].qsa) { attention(state, layers[i].rank); }
        }
        if (mtp) {
            const std::size_t head = model.head_rank();
            attention(sequence.mtp, head);
            out.push_back({at != nullptr ? at->mtp[0].p : sequence.mtp.tail.p,
                           sequence.mtp.tail.bytes, head});
            out.push_back({at != nullptr ? at->mtp[1].p : sequence.mtp_pending.p,
                           sequence.mtp_pending.bytes, head});
        }
        return out;
    }

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const ImageRegion& region : image_regions(sequences.front(), nullptr, position)) {
            bytes += region.bytes;
        }
        return bytes;
    }

    // Copies between a sequence image's regions and host pages (to the host when `save`) on the
    // regions' streams, and waits for the copies. The image runs through the pages in order, each
    // holding `page_bytes` of it.
    void copy_image(const std::vector<ImageRegion>& regions, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save) {
        std::uint64_t total = 0;
        for (const ImageRegion& region : regions) { total += region.bytes; }
        if (page_bytes == 0 || pages.size() < (total + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("qwen4_exp image: the pages do not hold the image");
        }
        std::uint64_t at = 0;
        for (const ImageRegion& region : regions) {
            RankBinding bind(device, region.rank);
            auto* device_bytes = static_cast<std::byte*>(region.device);
            for (std::size_t left = region.bytes; left > 0;) {
                const std::size_t in_page = std::min<std::size_t>(left, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                CUDA_CHECK(cudaMemcpyAsync(save ? static_cast<void*>(host) : device_bytes,
                                           save ? static_cast<void*>(device_bytes) : host, in_page,
                                           cudaMemcpyDefault, ranks[region.rank].stream));
                device_bytes += in_page;
                left -= in_page;
                at += in_page;
            }
        }
        for (auto& rank : ranks) {
            RankBinding bind(device, rank.rank);
            CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        }
    }

    SequenceImage save_image(std::uint32_t s, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const SequenceState& sequence = sequences.at(s);
        if (at != nullptr && (at->layers.size() != layers.size() || (mtp && at->mtp.size() != 2))) {
            throw std::invalid_argument("qwen4_exp image: the snapshot holds no state");
        }
        const std::uint32_t position = at != nullptr ? at->position : sequence.position;
        copy_image(image_regions(sequence, at, position), pages, page_bytes, true);
        return SequenceImage{.position = position,
                             .context  = at != nullptr ? at->context : sequence.context,
                             .mtp_follows =
                                 mtp && (at != nullptr ? at->mtp_follows : sequence.mtp_follows)};
    }

    void load_image(std::uint32_t s, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        SequenceState& sequence = sequences.at(s);
        if (image.position > options.max_context) {
            throw std::invalid_argument("qwen4_exp image: the image does not fit the sequence");
        }
        copy_image(image_regions(sequence, nullptr, image.position), pages, page_bytes, false);
        forget_streamed_kv(sequence);
        sequence.position    = image.position;
        sequence.context     = image.context;
        sequence.mtp_follows = mtp && image.mtp_follows;
        sequence.media_columns.clear();
        sequence.media_rope.clear();
        sequence.rope_delta = 0;
    }

    // Copies the recurrent state between a sequence and a snapshot on the layers' streams, and
    // waits for the copies: the paged KV and the indexer's pooled keys stay where they are, since
    // a sequence only writes positions at or past its own. The MTP block's tail and kept stack
    // go with them.
    void copy_state(SequenceState& sequence, SequenceSnapshot& snapshot, bool save) {
        const auto copy = [&](void* live, DeviceBuffer& kept, cudaStream_t stream) {
            if (live == nullptr || kept.bytes == 0) { return; }
            CUDA_CHECK(cudaMemcpyAsync(save ? kept.p : live, save ? live : kept.p, kept.bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        };
        for (std::size_t i = 0; i < layers.size(); ++i) {
            RankBinding bind(device, layers[i].rank);
            const auto regions = recurrent_regions(sequence.layers[i]);
            for (std::size_t k = 0; k < regions.size(); ++k) {
                copy(regions[k].first, snapshot.layers[i].buffers[k], ranks[layers[i].rank].stream);
            }
        }
        if (mtp && snapshot.mtp.size() == 2) {
            RankBinding bind(device, model.head_rank());
            const cudaStream_t stream = ranks[model.head_rank()].stream;
            copy(sequence.mtp.tail.p, snapshot.mtp[0], stream);
            copy(sequence.mtp_pending.p, snapshot.mtp[1], stream);
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

NgramTableStats Executor::ngram_stats() { return impl_->ngram_stats(); }

std::uint64_t Executor::ngram_resident_bytes() const noexcept {
    return impl_->table ? impl_->table->resident_bytes() : 0;
}

void Executor::reset(std::uint32_t sequence) { impl_->reset(sequence); }

void Executor::warm_up() { impl_->warm_up(); }
std::uint32_t Executor::prompt_step() const noexcept { return impl_->prompt_step(); }
void Executor::grow_expert_cache(std::uint64_t keep_free_bytes) {
    impl_->grow_expert_cache(keep_free_bytes);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->sequences.at(sequence).position;
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, std::span<const std::int32_t> next) {
    impl_->forward(sequence, tokens, logit_rows, next);
}

void Executor::forward_branches(std::uint32_t sequence, const SequenceSnapshot& at,
                                std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->forward_branches(sequence, at, tokens, lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence, std::uint32_t prompt,
                                       SequenceSnapshot& at, std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    impl_->forward_prompt_branches(sequence, prompt, at, tokens, lengths);
}

bool Executor::runs_branches() const noexcept { return impl_->runs_branches(); }

std::uint32_t Executor::max_logit_rows() const noexcept { return impl_->max_logit_rows; }

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

std::uint64_t Executor::snapshot_bytes() const { return impl_->snapshot_bytes(); }

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

std::uint64_t Executor::image_bytes(std::uint32_t position) const {
    return impl_->image_bytes(position);
}

SequenceImage Executor::save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes) {
    return impl_->save_image(sequence, at, pages, page_bytes);
}

void Executor::load_image(std::uint32_t sequence, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->load_image(sequence, image, pages, page_bytes);
}

void Executor::set_media(std::uint32_t sequence, std::span<const MediaItem> items,
                         std::vector<std::int32_t> rope_positions, std::int32_t rope_delta) {
    impl_->set_media(sequence, items, std::move(rope_positions), rope_delta);
}

bool Executor::vision() const noexcept { return impl_->vision_context != nullptr; }

std::uint32_t Executor::draft_tokens() const noexcept {
    return impl_->mtp ? impl_->options.draft_tokens : 0;
}

bool Executor::can_draft(std::uint32_t sequence) const { return impl_->can_draft(sequence); }

void Executor::stop_drafting(std::uint32_t sequence) {
    impl_->sequences.at(sequence).mtp_follows = false;
}

std::uint32_t Executor::draft(std::span<const std::uint32_t> sequences,
                              std::span<const std::int32_t> anchors, std::span<std::int32_t> out,
                              std::span<std::uint32_t> extents, std::uint32_t steps) {
    return impl_->draft(sequences, anchors, out, extents, steps);
}

void Executor::verify(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->verify(sequences, tokens);
}

void Executor::commit(std::span<const std::uint32_t> sequences,
                      std::span<const std::uint32_t> columns) {
    impl_->commit(sequences, columns);
}

Tensor Executor::logits(std::uint32_t rows) const {
    return Tensor(
        impl_->ranks[impl_->model.head_rank()].logits, DType::BF16,
        {static_cast<std::int32_t>(impl_->config.vocab_size), static_cast<std::int32_t>(rows)});
}

std::size_t Executor::head_rank() const noexcept { return impl_->model.head_rank(); }

ExpertCacheStats Executor::expert_cache_stats() const noexcept {
    if (impl_->hybrid) { return impl_->hybrid->stats(); }
    if (impl_->stream) {
        const auto stream = impl_->stream->stats();
        return ExpertCacheStats{.routes       = stream.routes,
                                .hits         = stream.hits,
                                .admitted     = stream.routes - stream.hits,
                                .copied_bytes = stream.read_bytes,
                                .slots        = stream.slots};
    }
    ExpertCacheStats out = impl_->cache ? impl_->cache->stats() : ExpertCacheStats{};
    if (impl_->misses) {
        const auto staged = impl_->misses->stats();
        out.cpu_routes += staged.cpu_pairs;
        out.copied_bytes += staged.staged_bytes;
    }
    return out;
}

cudaStream_t Executor::head_stream() const noexcept {
    return impl_->ranks[impl_->model.head_rank()].stream;
}

std::string Executor::expert_execution_profile() const {
    // Native hybrid banks compute some routed pairs in other arithmetic; a GGUF CPU share computes
    // the GPU's bits (gguf_expert_cpu), so its states are the GPU's.
    if (impl_->hybrid) { return impl_->hybrid->execution_profile(); }
    return "gpu";
}

bool Executor::hybrid_experts() const noexcept { return impl_->hybrid != nullptr; }
std::string Executor::expert_cpu_description() const {
    return impl_->misses ? impl_->misses->cpu_description() : std::string();
}
void Executor::save_expert_profile() const {
    const auto& path = impl_->options.hybrid_experts.record_profile;
    if (path.empty()) { return; }
    if (impl_->hybrid) {
        write_expert_profile(path, impl_->model.info().artifact_id, impl_->hybrid->route_counts());
    } else if (impl_->cache) {
        write_expert_profile(path, impl_->model.info().artifact_id, impl_->cache->totals());
    }
}
void Executor::bind_cancellation(const std::atomic<bool>* cancelled) noexcept {
    // Also drains partially enqueued work on exception paths before the request's borrowed flag
    // can die. Normal pass/draft boundaries have already drained and propagated any failure.
    if (!cancelled && impl_->hybrid) { try { impl_->hybrid->finish(); } catch (...) {} }
    impl_->cancelled = cancelled;
}

void Executor::raise_service_failure() {
    if (impl_->misses) { impl_->misses->raise_failure(); }
}

void Executor::abort(std::uint32_t sequence) {
    if (impl_->table) { impl_->table->wait(); }
    impl_->rows_waited = true;
    impl_->device.synchronize();
    impl_->pending_routes = 0;
    impl_->mtp_routes = 0;
    impl_->verifying = false;
    impl_->verified.clear();
    impl_->reset(sequence);
}

} // namespace ninfer::models::qwen4_exp
