#include "models/gemma4/executor.h"

#include "core/decode_graph.h"
#include "core/growable_region.h"
#include "core/vmm.h"
#include "core/weight_view.h"
#include "models/gemma4/cpu_experts.h"
#include "models/gemma4/kernels.h"
#include "models/gemma4/vision.h"
#include "models/gemma4/vision_kernels.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/ggml_fattn.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

namespace gguf = ops::gguf;

constexpr std::size_t kAlign = 256;

std::size_t round_up(std::size_t value, std::size_t to) { return (value + to - 1) / to * to; }

const float* fp32(const Model& model, WeightId id, std::int32_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {count});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

Weight matrix(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() == 1) { return ops::prepare_linear_weight(inputs.front()).weight; }
    if (!ops::joins(inputs)) {
        throw std::invalid_argument("gemma4: projection parts are not one stored parent");
    }
    return ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight;
}

// The matrices of `ids`: one when they are rows of one stored parent, else each on its own (a
// mixed-format GGUF stores a block's Q, K and V in different formats).
std::vector<Weight> matrices(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() > 1 && ops::joins(inputs)) {
        return {ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight};
    }
    std::vector<Weight> out;
    for (const auto& input : inputs) { out.push_back(ops::prepare_linear_weight(input).weight); }
    return out;
}

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

// A context-sized cache plane: rows of `row_bytes`, `rows` of them for each owner (a sequence's
// KV heads; the prompt chunk's scratch has one owner), `position_bytes` a position. Owned whole,
// or, beside an elastic expert cache, a growable region whose rows are backed only as far as their
// owner has reached (Impl::back), so the expert cache holds the rest of the device until then.
struct KvArea {
    DeviceBuffer owned;
    GrowableRegion region;
    void* p = nullptr;
    std::size_t bytes = 0, row_bytes = 0, position_bytes = 0;
    int rows = 0;
    std::vector<std::uint32_t> backed; // positions backed, by owner

    KvArea() = default;
    KvArea(std::size_t owners, int rows_per_owner, std::size_t row, std::size_t position,
           bool growing, int device)
        : bytes(owners * std::size_t(rows_per_owner) * row), row_bytes(row), position_bytes(position),
          rows(rows_per_owner), backed(owners, 0) {
        if (growing) {
            region = GrowableRegion(bytes, {.device = device});
            p      = region.data();
        } else {
            owned = DeviceBuffer(bytes);
            owned.fill(0);
            p = owned.p;
        }
    }
};

struct LayerPlan {
    bool full    = false;
    int head_dim = 0, kv_heads = 0, qkv_rows = 0;
    std::vector<Weight> qkv; // one [Q; K (; V)] parent, or the parts of a mixed-format block
    Weight output, gate_up, down;
    const float *input_norm = nullptr, *post_attention_norm = nullptr, *ffn_norm = nullptr,
                *post_ffn_norm = nullptr, *layer_scalar = nullptr, *q_norm = nullptr,
                *k_norm = nullptr;
    // MoE.
    bool moe = false;
    int delivery_layer = -1;
    const float *router = nullptr, *router_scale = nullptr, *expert_scale = nullptr,
                *pre_norm = nullptr, *post_norm = nullptr, *mlp_post_norm = nullptr;
    gguf::GgmlType gate_up_type{}, down_type{};
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // pointers: [E] gate, [E] up, [E] gate_up, [E] down
    // The F16 K/V cache: [sequences][kv heads][slots][D]. A full-attention block (K = V) keeps
    // V only, and its keys are derived from it where they are read.
    DeviceBuffer k;
    KvArea v;
    int slots = 0, ring = 0;
    std::int64_t seq_stride = 0, head_stride = 0; // elements of an F16 cache (and of its staging)
    // A quantized cache (kv_format.h) in place of k and v: the key plane (sliding blocks) and the
    // value plane, codes and scales [sequences][kv heads][slots][row bytes].
    DeviceBuffer k_codes, k_scales;
    KvArea v_codes, v_scales;
    kernels::KvPlaneView kq, vq;
    // A sliding block's checkpoint of every sequence's rings (snapshot()), allocated with the cache:
    // one buffer per ring buffer above (F16: K, V; quantized: K codes, K scales, V codes, V scales).
    std::vector<DeviceBuffer> checkpoint;
};

// Bytes of one cache row of a plane: head_dim / 256 blocks.
std::size_t plane_row_bytes(KvPlane plane, int head_dim) {
    return std::size_t(head_dim / kKvBlockDims) *
           std::size_t(kv_code_bytes(plane.coding) + kv_scale_bytes(plane.coding));
}

// A block of the assistant: its queries attend over the caches of the target block `target`.
struct DraftLayer {
    bool full = false;
    int head_dim = 0, target = 0;
    Weight query, output, gate_up, down;
    const float *input_norm = nullptr, *post_attention_norm = nullptr, *ffn_norm = nullptr,
                *post_ffn_norm = nullptr, *layer_scalar = nullptr, *q_norm = nullptr;
};

// Keys of the decode graphs: the batch size.
struct GraphEntry {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false; // ran once eagerly, so every kernel is loaded before the capture
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    const TextConfig& config;
    cudaStream_t stream = nullptr;
    int hidden = 0, query_heads = 0, vocab = 0, ring = 0, global_slots = 0;
    // A prompt chunk's sliding-window attention needs the window before it and the chunk itself:
    // a ring of prefill_ring slots that every sequence's chunks share, one block at a time
    // (staging_k/v), while each sequence keeps only its last `ring` (the window) positions.
    int prefill_ring = 0;
    DeviceBuffer staging_k, staging_v;
    KvArea derived_keys; // a prompt chunk's full-attention keys: [kv heads][slots][D]
    // A quantized cache: the format; a prompt chunk's full-attention values in F16 (as derived_keys);
    // a decode step's F16 K and V rows before they are stored ([kv heads][kKvRows][D] each).
    KvFormat kv;
    KvArea full_values;
    DeviceBuffer kv_rows_k, kv_rows_v;
    KvArea mask_full; // a prompt chunk's full-attention mask, [rows][full_slots]
    // Beside an elastic expert cache, the full blocks' value planes (a row per sequence and KV head)
    // and the prompt chunk's scratch above grow as far as the sequences reach: the expert cache
    // gives memory back when they need it and takes it again when a sequence is reset.
    bool growing = false;
    std::vector<KvArea*> sequence_areas, chunk_areas;
    std::atomic<std::uint64_t> kv_backed{0};
    std::uint64_t kv_fixed = 0; // the caches' bytes that do not grow
    static constexpr int kKvRows = 8;
    float eps = 1e-6f;
    std::vector<LayerPlan> layers;
    // The CPU's share of missing experts; it outlives the delivery that calls it.
    std::unique_ptr<CpuExperts> cpu_experts;
    std::unique_ptr<ExpertDelivery> delivery;
    mutable std::mutex expert_stats_mutex;
    ExpertDeliveryStats expert_startup_stats;
    Weight head; // tied token table
    const float* final_norm = nullptr;
    DeviceBuffer inv_freq_sliding, inv_freq_full;
    // The full-attention rotations of every position, (cos, sin) [slots][rotary pairs]: the keys a
    // full block derives from its values turn by them.
    DeviceBuffer rope_full;
    // What the full-attention blocks' rotated queries are multiplied by: the square of YaRN's
    // attention factor, one without YaRN.
    float full_query_scale = 1.0f;
    std::vector<std::uint32_t> positions;
    // Device staging: tokens, positions, sequences for up to max(prefill_chunk, sequences) rows.
    // d_keys: the assistant's last visible key position per row. d_blocks: a prompt chunk row's
    // media unit (-1: text), d_unit_rows: each unit's first row, for the chunk's media. A branch
    // pass's rows: d_slots, the position whose cache slot each takes; d_starts, its branch's first
    // row.
    DeviceBuffer d_tokens, d_positions, d_sequences, d_keys, d_blocks, d_unit_rows, d_slots,
        d_starts;
    // The vision tower (prompt chunks with media), and events around the last chunk's media.
    std::optional<VisionTower> vision;
    bool vision_media = false;
    cudaEvent_t media_begin = nullptr, media_end = nullptr;
    bool media_timed = false;
    std::unique_ptr<PinnedHostBuffer> h_staging;
    // Recorded after the staging copies: the next stage() rewrites the pinned buffer only once
    // the previous copies have read it (a prompt chunk that keeps no logits is not synchronized).
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    std::uint64_t kv_bytes      = 0;
    int decode_splits_full = 1, decode_splits_sliding = 1;
    // Decode graphs by row count, verification graphs (which leave the anchors alone) by 16 + rows.
    std::map<std::uint32_t, GraphEntry> graphs;

    // The assistant (options.draft_tokens > 0).
    std::vector<DraftLayer> draft_layers;
    Weight draft_head, pre_projection, post_projection;
    const float* draft_final_norm = nullptr;
    int draft_hidden = 0, draft_width = 0, draft_domain = 0;
    float draft_eps = 1e-6f;
    // The target's final-normed hidden state after each sequence's last fed token (BF16 [S][H]),
    // whether it follows the sequence, and the rows of the last decode or verification ([8][H]).
    DeviceBuffer anchor_hidden, row_hidden;
    std::vector<char> anchored;
    // A chain's drafts [steps][rows] (the device feeds each step the last one) and the host copy.
    DeviceBuffer d_drafts;
    std::unique_ptr<PinnedHostBuffer> h_drafts;
    std::uint32_t verify_width = 0; // rows per sequence of the last verify()
    std::map<std::uint32_t, GraphEntry> draft_graphs; // by rows << 8 | steps

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(o), config(m.config()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("gemma4 executor: invalid options");
        }
        if (device.size() != 1) {
            throw std::invalid_argument("Gemma 4 runs on one device");
        }
        stream      = device.stream;
        kv          = kv_format(options.kv_cache);
        // Beside an automatic cache of host or disk experts, the full-attention caches grow with
        // their sequences and the cache is elastic (ExpertDeliveryOptions::elastic).
        const int device_id = device.rank(0).device;
        growing = !model.expert_banks().empty() && !options.expert_cache_bytes &&
                  vmm::granularity(device_id) != 0;
        hidden      = int(config.hidden_size);
        query_heads = int(config.num_attention_heads);
        vocab       = int(config.vocab_size);
        eps         = config.rms_norm_eps;
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        if (options.draft_tokens > std::uint32_t(gguf::kMaxVectorColumns) - 1) {
            throw std::invalid_argument("gemma4 executor: at most 7 drafts a round");
        }
        if (options.draft_tokens > 0 && !model.assistant()) {
            throw std::invalid_argument("gemma4 executor: drafting needs the model's assistant");
        }
        // A verification writes a round's drafts past the window before keeping only some: the
        // ring holds them without overwriting a key the window still needs.
        ring         = int(round_up(config.sliding_window + options.draft_tokens, 256));
        prefill_ring = int(round_up(config.sliding_window + options.prefill_chunk, 256));
        global_slots = int(round_up(options.max_context + 1, 256));
        positions.assign(options.sequences, 0);
        const auto& w = model.weights();
        head          = matrix(model, {w.output_head});
        final_norm    = fp32(model, w.final_norm, hidden);
        // RoPE frequencies, as the reference computes them in FP32. Under YaRN the full-attention
        // blocks' proportional schedule (over the whole head, its first rotary_pairs_full pairs
        // rotated) takes YaRN's frequencies, and their queries the square of its attention factor.
        {
            std::vector<float> sliding(config.head_dim / 2), full(config.global_head_dim / 2, 0.0f);
            for (std::size_t i = 0; i < sliding.size(); ++i) {
                sliding[i] = float(1.0 / std::pow(double(config.rope_theta_sliding),
                                                  double(2 * i) / double(config.head_dim)));
            }
            if (options.rope_yarn.factor > 1.0f) {
                const ops::RopeYarnFrequencies yarn = ops::rope_yarn_frequencies(
                    int(config.global_head_dim), int(config.rotary_pairs_full),
                    config.rope_theta_full, options.rope_yarn);
                std::copy_n(yarn.inverse_frequency, config.rotary_pairs_full, full.begin());
                full_query_scale = yarn.attention_factor * yarn.attention_factor;
            } else {
                for (std::uint32_t i = 0; i < config.rotary_pairs_full; ++i) {
                    full[i] = float(1.0 / std::pow(double(config.rope_theta_full),
                                                   double(2 * i) / double(config.global_head_dim)));
                }
            }
            inv_freq_sliding = DeviceBuffer(sliding.size() * 4);
            inv_freq_sliding.copy_from_host(sliding.data(), sliding.size() * 4);
            inv_freq_full = DeviceBuffer(full.size() * 4);
            inv_freq_full.copy_from_host(full.data(), full.size() * 4);
            rope_full = DeviceBuffer(std::size_t(global_slots) * config.rotary_pairs_full * 8);
            kernels::rope_table(static_cast<const float*>(inv_freq_full.p),
                                int(config.rotary_pairs_full), global_slots,
                                static_cast<float2*>(rope_full.p), stream);
        }
        const int sequences = int(options.sequences);
        {
            // The caches are sized up front for every sequence's whole context: refuse with the
            // numbers rather than fail inside an allocation. Counted here: what grows with the
            // context or the sequences; the reserve covers the rest of the workspace, the CUDA
            // graphs (some captured while serving) and a margin. The narrower prompt chunks a card
            // without that room falls back to (gemma4_core.cpp) keep less, and their planned
            // workspace is checked against what is left once it is known (below).
            std::size_t needed = 0, staging = 0, chunk_bytes = 0;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                const bool full         = config.full_attention[i];
                const std::size_t slots = full ? global_slots : ring;
                const int d             = int(config.layer_head_dim(i));
                const std::size_t heads = config.layer_kv_heads(i);
                // Sliding rings also hold a checkpoint each; a full block stores V only.
                if (full) {
                    needed += slots * heads * plane_row_bytes(kv.full, d) * std::size_t(sequences);
                    // A prompt chunk's derived keys, and with a quantized cache its values, in F16.
                    chunk_bytes = std::max(chunk_bytes, (kv.quantized() ? 2 : 1) * slots * heads * d * 2);
                } else {
                    needed += (options.checkpoints ? 2 : 1) * slots * heads *
                              (plane_row_bytes(kv.key, d) + plane_row_bytes(kv.value, d)) *
                              std::size_t(sequences);
                    staging = std::max(staging, std::size_t(prefill_ring) * 2 * heads * d * 2);
                }
            }
            // The prompt chunk's full-attention mask (plan_workspace()).
            needed += staging + chunk_bytes + std::size_t(options.prefill_chunk) * global_slots * 2;
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = start_reserve();
            if (needed + reserve > free_bytes) {
                throw InsufficientMemory(
                    "Gemma 4: the KV cache, checkpoints and prompt buffers of " +
                    std::to_string(sequences) + " sequence(s) of " +
                    std::to_string(options.max_context) + " tokens need " +
                    std::to_string(needed >> 20) + " MiB beside " + std::to_string(reserve >> 20) +
                    " MiB for the workspace and CUDA graphs, and " +
                    std::to_string(free_bytes >> 20) +
                    " MiB are free after the weights; lower --max-concurrency or --max-context");
            }
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            const LayerWeights& lw = w.layers[i];
            LayerPlan p;
            p.full     = config.full_attention[i];
            p.head_dim = int(config.layer_head_dim(i));
            p.kv_heads = int(config.layer_kv_heads(i));
            p.qkv = lw.attention.value
                        ? matrices(model, {lw.attention.query, lw.attention.key, *lw.attention.value})
                        : matrices(model, {lw.attention.query, lw.attention.key});
            p.qkv_rows = 0;
            for (const Weight& part : p.qkv) { p.qkv_rows += part.n; }
            p.output   = matrix(model, {lw.attention.output});
            p.gate_up  = matrix(model, {lw.mlp_gate, lw.mlp_up});
            p.down     = matrix(model, {lw.mlp_down});
            p.input_norm          = fp32(model, lw.input_norm, hidden);
            p.post_attention_norm = fp32(model, lw.post_attention_norm, hidden);
            p.ffn_norm            = fp32(model, lw.ffn_norm, hidden);
            p.post_ffn_norm       = fp32(model, lw.post_ffn_norm, hidden);
            p.layer_scalar        = fp32(model, lw.layer_scalar, 1);
            p.q_norm              = fp32(model, lw.attention.query_norm, p.head_dim);
            p.k_norm              = fp32(model, lw.attention.key_norm, p.head_dim);
            for (const Weight* m : {&p.output, &p.gate_up, &p.down}) {
                if (!is_gguf(m->qtype)) {
                    throw std::invalid_argument("Gemma 4 runs GGUF block matrices");
                }
            }
            for (const Weight& m : p.qkv) {
                if (!is_gguf(m.qtype)) {
                    throw std::invalid_argument("Gemma 4 runs GGUF block matrices");
                }
            }
            if (lw.moe) {
                const MoeWeights& mw = *lw.moe;
                const int e = int(config.num_experts), width = int(config.moe_intermediate_size);
                p.moe           = true;
                p.router        = fp32(model, mw.router, e * hidden);
                p.router_scale  = fp32(model, mw.router_scale, hidden);
                p.expert_scale  = fp32(model, mw.expert_scale, e);
                p.pre_norm      = fp32(model, mw.pre_norm, hidden);
                p.post_norm     = fp32(model, mw.post_norm, hidden);
                p.mlp_post_norm = fp32(model, mw.mlp_post_norm, hidden);
                Weight gate_up, down;
                if (model.expert_residency() == ExpertResidency::Device) {
                    gate_up = matrix(model, {*mw.gate_up});
                    down = matrix(model, {*mw.down});
                    if (!is_gguf(gate_up.qtype) || !is_gguf(down.qtype)) {
                        throw std::invalid_argument("Gemma 4 experts must be GGUF block matrices");
                    }
                    p.gate_up_type = ops::detail::gguf_type(gate_up.qtype);
                    p.down_type = ops::detail::gguf_type(down.qtype);
                    p.gate_up_row_bytes = row_bytes(gate_up);
                    p.down_row_bytes = row_bytes(down);
                    std::vector<const void*> table(4 * std::size_t(e));
                    for (int x = 0; x < e; ++x) {
                        const auto* gu = static_cast<const std::byte*>(gate_up.qdata) +
                                         std::int64_t(x) * 2 * width * p.gate_up_row_bytes;
                        table[x] = gu;
                        table[e + x] = gu + std::int64_t(width) * p.gate_up_row_bytes;
                        table[2 * e + x] = gu;
                        table[3 * e + x] = static_cast<const std::byte*>(down.qdata) +
                                           std::int64_t(x) * hidden * p.down_row_bytes;
                    }
                    p.tables = DeviceBuffer(table.size() * sizeof(void*));
                    p.tables.copy_from_host(table.data(), table.size() * sizeof(void*));
                } else {
                    const auto& bank = model.expert_banks()[i];
                    p.delivery_layer = int(i);
                    p.gate_up_type = ops::detail::gguf_type(bank.gate_up_type);
                    p.down_type = ops::detail::gguf_type(bank.down_type);
                    p.gate_up_row_bytes = bank.gate_up_row_bytes;
                    p.down_row_bytes = bank.down_row_bytes;
                }
                if (!gguf::moe_matrix_supported(p.gate_up_type, 2 * width, hidden, true) ||
                    !gguf::moe_matrix_supported(p.down_type, hidden, width, true)) {
                    throw std::invalid_argument(
                        "Gemma 4: no integer expert kernel for these expert formats");
                }
            }
            p.slots       = p.full ? global_slots : ring;
            p.ring        = p.full ? 0 : ring;
            p.head_stride = std::int64_t(p.slots) * p.head_dim;
            p.seq_stride  = p.head_stride * p.kv_heads;
            if (!kv.quantized()) {
                const std::size_t bytes = std::size_t(p.seq_stride) * sequences * sizeof(__half);
                if (!p.full) {
                    p.k = DeviceBuffer(bytes);
                    p.k.fill(0);
                    kv_bytes += bytes;
                }
                p.v = KvArea(std::size_t(sequences), p.kv_heads, std::size_t(p.head_stride) * sizeof(__half),
                             std::size_t(p.head_dim) * sizeof(__half), growing && p.full, device_id);
                kv_bytes += bytes;
            } else {
                const auto plane = [&](KvPlane format, auto& codes, auto& scales) {
                    kernels::KvPlaneView view;
                    view.coding     = format.coding;
                    view.rotated    = format.rotated;
                    view.blocks     = p.head_dim / kKvBlockDims;
                    view.code_row   = std::int64_t(view.blocks) * kv_code_bytes(format.coding);
                    view.scale_row  = std::int64_t(view.blocks) * kv_scale_bytes(format.coding);
                    view.code_head  = view.code_row * p.slots;
                    view.scale_head = view.scale_row * p.slots;
                    view.code_seq   = view.code_head * p.kv_heads;
                    view.scale_seq  = view.scale_head * p.kv_heads;
                    if constexpr (std::is_same_v<std::decay_t<decltype(codes)>, KvArea>) {
                        const bool grows = growing && p.full;
                        codes  = KvArea(std::size_t(sequences), p.kv_heads, std::size_t(view.code_head),
                                        std::size_t(view.code_row), grows, device_id);
                        scales = KvArea(std::size_t(sequences), p.kv_heads, std::size_t(view.scale_head),
                                        std::size_t(view.scale_row), grows, device_id);
                    } else {
                        codes  = DeviceBuffer(std::size_t(view.code_seq) * sequences);
                        scales = DeviceBuffer(std::size_t(view.scale_seq) * sequences);
                        codes.fill(0);
                        scales.fill(0);
                    }
                    view.codes  = static_cast<std::uint8_t*>(codes.p);
                    view.scales = static_cast<std::uint8_t*>(scales.p);
                    kv_bytes += codes.bytes + scales.bytes;
                    return view;
                };
                if (!p.full) { p.kq = plane(kv.key, p.k_codes, p.k_scales); }
                p.vq = plane(p.full ? kv.full : kv.value, p.v_codes, p.v_scales);
            }
            if (!p.full && options.checkpoints) {
                // Sized up front with the cache, so a configuration that cannot hold them is refused
                // at startup instead of failing a request's first checkpoint.
                for (const auto& [ring, bytes] : ring_buffers(p, 0)) {
                    (void)ring;
                    p.checkpoint.emplace_back(bytes * sequences);
                    kv_bytes += bytes * sequences;
                }
            }
            layers.push_back(std::move(p));
        }
        {
            std::size_t staging = 0;
            for (const LayerPlan& p : layers) {
                if (!p.full) {
                    staging = std::max(staging, std::size_t(prefill_ring) * p.kv_heads *
                                                    p.head_dim * sizeof(__half));
                }
            }
            // A prompt chunk's full-attention keys, derived from one sequence's values. Zeroed (or
            // zeroed as they are backed): the slots past a chunk are masked, and a masked score must
            // stay finite.
            const LayerPlan* full = nullptr;
            for (const LayerPlan& p : layers) {
                if (!p.full) { continue; }
                if (full != nullptr && (p.kv_heads != full->kv_heads || p.head_dim != full->head_dim)) {
                    throw std::invalid_argument("Gemma 4: full-attention blocks of different shapes");
                }
                full = &p;
            }
            if (full != nullptr) {
                const std::size_t row = std::size_t(full->head_stride) * sizeof(__half);
                const std::size_t position = std::size_t(full->head_dim) * sizeof(__half);
                derived_keys = KvArea(1, full->kv_heads, row, position, growing, device_id);
                kv_bytes += derived_keys.bytes;
                if (kv.quantized()) {
                    full_values = KvArea(1, full->kv_heads, row, position, growing, device_id);
                    kv_bytes += full_values.bytes;
                }
            }
            mask_full = KvArea(1, 1, std::size_t(options.prefill_chunk) * global_slots * 2,
                               std::size_t(options.prefill_chunk) * 2, growing, device_id);
            if (kv.quantized()) {
                // A decode step's rows: at most 8 (a batch or a verification) of every block.
                std::size_t row_bytes = 0;
                for (const LayerPlan& p : layers) {
                    row_bytes = std::max(row_bytes, std::size_t(p.kv_heads) * p.head_dim * 2);
                }
                kv_rows_k = DeviceBuffer(row_bytes * kKvRows);
                kv_rows_v = DeviceBuffer(row_bytes * kKvRows);
                kv_bytes += 2 * row_bytes * kKvRows;
            }
            if (staging != 0) {
                staging_k = DeviceBuffer(staging);
                staging_v = DeviceBuffer(staging);
                staging_k.fill(0);
                staging_v.fill(0);
                kv_bytes += 2 * staging;
            }
        }
        const int sms = device.multiprocessor_count();
        // Upper bounds: a step uses as many as its keys fill (a tile each), so the sliding
        // window's 1024 keys take 16 and a long full-attention context fills one wave of the
        // eight-warp kernel (two blocks per SM).
        decode_splits_full    = std::clamp(2 * sms / std::max(1, int(config.num_global_key_value_heads)), 1, 128);
        decode_splits_sliding = 16;
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, sequences);
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        d_keys      = DeviceBuffer(rows * 4);
        d_blocks    = DeviceBuffer(rows * 4);
        d_unit_rows = DeviceBuffer(rows * 4);
        d_slots                = DeviceBuffer(rows * 4);
        d_starts               = DeviceBuffer(rows * 4);
        h_staging   = std::make_unique<PinnedHostBuffer>(kStagingRegions * rows * 4);
        if (options.vision_max_patches > 0) {
            if (!model.vision_config()) {
                throw std::invalid_argument("gemma4 executor: media needs the model's vision tower");
            }
            vision_media = true;
            CUDA_CHECK(cudaEventCreate(&media_begin));
            CUDA_CHECK(cudaEventCreate(&media_end));
            if (model.vision_placement() == VisionPlacement::Device ||
                model.vision_placement() == VisionPlacement::Pinned) {
                // One batch holds the largest unit, or as many smaller ones (video frames).
                vision.emplace(model, options.vision_max_patches);
            }
        }
        if (options.draft_tokens > 0) { plan_assistant(); }
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        for (LayerPlan& p : layers) {
            if (!p.full) { continue; }
            if (kv.quantized()) {
                sequence_areas.push_back(&p.v_codes);
                sequence_areas.push_back(&p.v_scales);
            } else {
                sequence_areas.push_back(&p.v);
            }
        }
        for (KvArea* area : {&derived_keys, &full_values, &mask_full}) {
            if (area->bytes != 0) { chunk_areas.push_back(area); }
        }
        kv_fixed = kv_bytes;
        if (growing) {
            for (KvArea* area : sequence_areas) { kv_fixed -= area->bytes; }
            kv_fixed -= derived_keys.bytes + full_values.bytes;
        }
        kv_backed = kv_fixed;
        workspace_bytes = plan_workspace();
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (options.prefill_chunk < kFullChunk &&
                workspace_bytes + kGraphReserve + unbacked() > free_bytes) {
                // The destructor does not run for a constructor that throws.
                if (media_begin != nullptr) { cudaEventDestroy(media_begin); }
                if (media_end != nullptr) { cudaEventDestroy(media_end); }
                throw InsufficientMemory(
                    "Gemma 4: the workspace of " + std::to_string(workspace_bytes >> 20) +
                    " MiB and " + std::to_string(kGraphReserve >> 20) +
                    " MiB for the CUDA graphs need more than the " +
                    std::to_string(free_bytes >> 20) +
                    " MiB free after the caches; lower --max-concurrency or --max-context");
            }
        }
        workspace = std::make_unique<WorkspaceArena>(workspace_bytes);
        if (!model.expert_banks().empty()) {
            std::vector<ExpertDeliveryLayer> sources;
            std::vector<std::uint64_t> sizes;
            for (const auto& bank : model.expert_banks()) {
                sources.push_back(bank.delivery);
                for (const auto& expert : bank.delivery.experts) {
                    sizes.resize(std::max(sizes.size(), expert.size()), 0);
                    for (std::size_t b = 0; b < expert.size(); ++b) {
                        sizes[b] = std::max(sizes[b], std::uint64_t(round_up(
                            expert[b].bytes + expert[b].zero_tail, kAlign)));
                    }
                }
            }
            // One whole-layer bank at least (the transport adds a second where memory allows).
            std::uint64_t slot = 0;
            for (auto bytes : sizes) { slot += bytes; }
            const auto bank = slot * config.num_experts;
            const auto metadata = std::uint64_t(options.prefill_chunk) * config.top_k_experts * 8 +
                std::uint64_t(config.num_hidden_layers) * config.num_experts * 4 * sizeof(void*) +
                config.num_experts * 4;
            std::size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            const auto cache = options.expert_cache_bytes.value_or(0);
            // The caches that grow later need their memory too: the elastic cache gives it back.
            if (bank + metadata + cache + unbacked() + kGraphReserve > free) {
                throw InsufficientMemory("Gemma 4: expert delivery needs " +
                    std::to_string((bank + metadata + cache) >> 20) + " MiB beside " +
                    std::to_string(unbacked() >> 20) +
                    " MiB for the growing caches and the CUDA Graph reserve after workspace allocation");
            }
            ExpertDeliveryOptions transport;
            transport.pair_capacity = options.prefill_chunk * config.top_k_experts;
            transport.cache_bytes = options.expert_cache_bytes;
            // A growing cache's next positions come from the headroom before the expert cache
            // has to give memory back.
            transport.keep_free_bytes = kGraphReserve + (growing ? kKvHeadroom : 0);
            transport.elastic = growing;
            const float dma_share = options.expert_dma_share.value_or(0.0F);
            if (!(dma_share >= 0 && dma_share <= 1) || options.expert_cpu_threads > 256) {
                throw std::invalid_argument("Gemma 4: the DMA share must be 0..1 and CPU threads 0..256");
            }
            std::vector<CpuExpertLayer> cpu_layers;
            bool computes = false;
            for (const auto& bank : model.expert_banks()) {
                cpu_layers.push_back({bank.gate_up_type, bank.down_type,
                                      std::int64_t(bank.gate_up_row_bytes),
                                      std::int64_t(bank.down_row_bytes)});
                computes = computes || CpuExperts::computes(cpu_layers.back(), hidden,
                                                            int(config.moe_intermediate_size));
            }
            if (computes && options.expert_dma_share.value_or(0.0F) < 1) {
                cpu_experts = std::make_unique<CpuExperts>(std::move(cpu_layers), hidden,
                    int(config.moe_intermediate_size), int(config.top_k_experts));
                transport.cpu.products      = cpu_experts.get();
                transport.cpu.input_bytes   = cpu_experts->input_bytes();
                transport.cpu.product_bytes = cpu_experts->product_bytes();
                transport.cpu.share         = 1.0 - double(dma_share);
                transport.cpu.balance       = !options.expert_dma_share;
                transport.cpu.threads       = options.expert_cpu_threads;
            }
            delivery = std::make_unique<ExpertDelivery>(device.rank(0), model.expert_files(),
                                                        std::move(sources), transport);
        }
    }

    // The startup check's room for the workspace and the CUDA graphs: 1.5 GiB at the full
    // 2048-token chunk (measured to leave 0.35-0.46 GiB for the graphs captured while serving
    // eight sequences); a narrower chunk keeps the least its workspace takes and then checks the
    // planned workspace with kGraphReserve beside it.
    static constexpr std::uint32_t kFullChunk  = 2048;
    static constexpr std::size_t kGraphReserve = std::size_t(320) << 20;
    static constexpr std::size_t kKvHeadroom   = std::size_t(256) << 20;

    // ---- Growing caches -------------------------------------------------------------------------

    // Device bytes the growing caches may still take.
    [[nodiscard]] std::size_t unbacked() const {
        std::size_t bytes = 0;
        for (const auto* list : {&sequence_areas, &chunk_areas}) {
            for (const KvArea* area : *list) {
                if (area->region) { bytes += area->region.bytes() - area->region.mapped_bytes(); }
            }
        }
        return bytes;
    }

    void count_backed() {
        std::uint64_t bytes = kv_fixed;
        for (const KvArea* area : sequence_areas) { bytes += area->region.mapped_bytes(); }
        for (const KvArea* area : {&derived_keys, &full_values}) {
            if (area->region) { bytes += area->region.mapped_bytes(); }
        }
        kv_backed.store(bytes, std::memory_order_relaxed);
    }

    // A growing area's rows of `owner` up to `positions`: the bytes still to back, and backing them
    // (each newly backed granule zeroed on the stream). False when the device has no room.
    [[nodiscard]] std::size_t missing(const KvArea& a, std::size_t owner, std::uint32_t positions) const {
        if (!a.region || positions <= a.backed[owner]) { return 0; }
        std::size_t bytes = 0;
        for (int r = 0; r < a.rows; ++r) {
            bytes += a.region.unmapped_bytes((owner * a.rows + r) * a.row_bytes,
                                             std::size_t(positions) * a.position_bytes);
        }
        return bytes;
    }
    [[nodiscard]] bool extend(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || positions <= a.backed[owner]) { return true; }
        for (int r = 0; r < a.rows; ++r) {
            if (!a.region.map((owner * a.rows + r) * a.row_bytes,
                              std::size_t(positions) * a.position_bytes, stream)) {
                return false;
            }
        }
        a.backed[owner] = positions;
        return true;
    }

    // Unbacks `owner`'s rows past `positions`, except granules another row still reaches into.
    // The stream has finished with them.
    void trim(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || a.backed[owner] <= positions) { return; }
        const std::size_t g   = a.region.granule();
        const std::uint32_t was = a.backed[owner];
        a.backed[owner]       = positions;
        const auto reached = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = 0; o < a.backed.size(); ++o) {
                for (int r = 0; r < a.rows; ++r) {
                    const std::size_t row = (o * a.rows + r) * a.row_bytes;
                    if (row < hi && lo < row + std::size_t(a.backed[o]) * a.position_bytes) { return true; }
                }
            }
            return false;
        };
        for (int r = 0; r < a.rows; ++r) {
            const std::size_t row = (owner * a.rows + r) * a.row_bytes;
            const std::size_t hi  = row + std::size_t(was) * a.position_bytes;
            for (std::size_t at = (row + std::size_t(positions) * a.position_bytes) / g * g; at < hi; at += g) {
                if (!reached(at, at + g)) { a.region.unmap(at, g); }
            }
        }
    }

    // Before a call that writes or reads `sequence`'s positions below `end`: its full-attention
    // rows (and with `chunk`, the prompt chunk's scratch) are backed up to the 256-slot tile that
    // holds `end`, the expert cache giving memory back when the device keeps less than the CUDA
    // Graph reserve beside them. Outside capture.
    void back(std::uint32_t sequence, std::uint32_t end, bool chunk) {
        if (!growing) { return; }
        const auto positions = std::uint32_t(round_up(end, 256));
        std::size_t need = 0;
        for (const KvArea* area : sequence_areas) { need += missing(*area, sequence, positions); }
        if (chunk) {
            for (const KvArea* area : chunk_areas) { need += missing(*area, 0, positions); }
        }
        const auto apply = [&] {
            bool ok = true;
            for (KvArea* area : sequence_areas) { ok = ok && extend(*area, sequence, positions); }
            if (chunk) {
                for (KvArea* area : chunk_areas) { ok = ok && extend(*area, 0, positions); }
            }
            return ok;
        };
        if (need == 0) {
            (void)apply();
            return;
        }
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (free < need + kGraphReserve) { delivery->shrink_cache(need + kGraphReserve - free); }
        while (!apply()) {
            if (delivery->shrink_cache(std::max(need, std::size_t(64) << 20)) == 0) {
                count_backed();
                throw InsufficientMemory("Gemma 4: no device memory left to grow the KV cache to " +
                                         std::to_string(positions) + " positions");
            }
        }
        count_backed();
    }

    // A reset sequence's rows go back to the device, the prompt scratch to what the longest
    // remaining sequence reaches, and the expert cache takes what is free beyond its headroom.
    void release(std::uint32_t sequence) {
        if (!growing || sequence_areas.empty() || sequence_areas.front()->backed[sequence] == 0) { return; }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (KvArea* area : sequence_areas) { trim(*area, sequence, 0); }
        std::uint32_t longest = 0;
        for (const std::uint32_t backed : sequence_areas.front()->backed) { longest = std::max(longest, backed); }
        for (KvArea* area : chunk_areas) { trim(*area, 0, longest); }
        count_backed();
        delivery->grow_cache();
    }

    std::size_t start_reserve() const {
        return options.prefill_chunk >= kFullChunk ? std::size_t(1536) << 20
                                                   : std::size_t(512) << 20;
    }

    ~Impl() {
        if (delivery || growing) { cudaStreamSynchronize(stream); }
        if (media_begin != nullptr) { cudaEventDestroy(media_begin); }
        if (media_end != nullptr) { cudaEventDestroy(media_end); }
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    void plan_assistant() {
        const AssistantConfig& ac = *model.assistant_config();
        const AssistantWeights& aw = *model.assistant();
        draft_hidden     = int(ac.hidden_size);
        draft_width      = int(ac.intermediate_size);
        draft_eps        = ac.rms_norm_eps;
        draft_domain     = int(model.resources().public_token_count);
        draft_head       = matrix(model, {aw.output_head});
        pre_projection   = matrix(model, {aw.pre_projection});
        post_projection  = matrix(model, {aw.post_projection});
        draft_final_norm = fp32(model, aw.final_norm, draft_hidden);
        // The target's last block of each kind holds the keys the assistant reads.
        int last_full = -1, last_sliding = -1;
        for (int i = 0; i < int(layers.size()); ++i) { (layers[i].full ? last_full : last_sliding) = i; }
        for (std::size_t i = 0; i < aw.layers.size(); ++i) {
            const AssistantLayerWeights& lw = aw.layers[i];
            DraftLayer d;
            d.full     = ac.full_attention[i];
            d.head_dim = int(d.full ? config.global_head_dim : config.head_dim);
            d.target   = d.full ? last_full : last_sliding;
            d.query    = matrix(model, {lw.query});
            d.output   = matrix(model, {lw.output});
            d.gate_up  = matrix(model, {lw.mlp_gate, lw.mlp_up});
            d.down     = matrix(model, {lw.mlp_down});
            d.input_norm          = fp32(model, lw.input_norm, draft_hidden);
            d.post_attention_norm = fp32(model, lw.post_attention_norm, draft_hidden);
            d.ffn_norm            = fp32(model, lw.ffn_norm, draft_hidden);
            d.post_ffn_norm       = fp32(model, lw.post_ffn_norm, draft_hidden);
            d.layer_scalar        = fp32(model, lw.layer_scalar, 1);
            d.q_norm              = fp32(model, lw.query_norm, d.head_dim);
            for (const Weight* m : {&d.query, &d.output, &d.gate_up, &d.down}) {
                if (!is_gguf(m->qtype)) {
                    throw std::invalid_argument("the Gemma 4 assistant runs GGUF block matrices");
                }
            }
            draft_layers.push_back(std::move(d));
        }
        for (const Weight* m : {&draft_head, &pre_projection, &post_projection}) {
            if (!is_gguf(m->qtype)) {
                throw std::invalid_argument("the Gemma 4 assistant runs GGUF block matrices");
            }
        }
        anchor_hidden = DeviceBuffer(std::size_t(options.sequences) * hidden * 2);
        row_hidden    = DeviceBuffer(std::size_t(gguf::kMaxVectorColumns) * hidden * 2);
        anchored.assign(options.sequences, 0);
        d_drafts = DeviceBuffer(std::size_t(options.draft_tokens) * gguf::kMaxVectorColumns * 4);
        h_drafts = std::make_unique<PinnedHostBuffer>(std::size_t(options.draft_tokens) *
                                                      gguf::kMaxVectorColumns * 4);
    }

    // ---- Workspace -----------------------------------------------------------------------------

    std::size_t plan_workspace() const {
        const std::size_t t     = options.prefill_chunk;
        const std::size_t h     = hidden;
        std::size_t qkv_rows    = 0, q_floats = 0;
        std::size_t gguf_scratch = 0;
        for (const LayerPlan& p : layers) {
            qkv_rows = std::max<std::size_t>(qkv_rows, p.qkv_rows);
            q_floats = std::max<std::size_t>(q_floats, std::size_t(query_heads) * p.head_dim);
            std::vector<const Weight*> products{&p.output, &p.down};
            for (const Weight& part : p.qkv) { products.push_back(&part); }
            for (const Weight* m : products) {
                const ops::detail::GgufShape shape{m->qtype, m->n, m->k};
                gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                          {&shape, 1}, 1, int(t)));
            }
            const ops::detail::GgufShape gu{p.gate_up.qtype, p.gate_up.n, p.gate_up.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_swiglu_workspace_bytes(gu, nullptr, 1, 8));
            gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                      {&gu, 1}, 1, int(t)));
        }
        const ops::detail::GgufShape hs{head.qtype, std::min<int>(head.n, vocab), head.k};
        gguf_scratch = std::max(gguf_scratch,
                                ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        std::size_t bytes = 0;
        const auto add    = [&](std::size_t n) { bytes += round_up(n, kAlign); };
        add(t * h * 4);              // residual
        add(t * h * 2);              // normalized input
        add(t * qkv_rows * 2);       // qkv
        add(t * q_floats * 4);       // q
        add(t * q_floats * 4);       // attention out FP32
        add(t * q_floats * 2);       // attention out BF16
        add(t * h * 2);              // o
        add(t * h * 2 * 2);          // mlp / moe inputs
        add(t * config.intermediate_size * 2);
        add(t * config.intermediate_size * 2 * 4); // [gate; up] FP32 plane
        add(t * h * 2);              // dense out
        add(t * std::size_t(prefill_ring) * 2); // sliding-window mask
        add(gguf::flash_attention_workspace_bytes(512, query_heads, 1, int(t)));
        add(gguf::flash_attention_workspace_bytes(256, query_heads, 1, int(t)));
        add(std::size_t(kHeadRows) * vocab * 4); // head FP32 plane
        add(kHeadRows * h * 2);
        std::size_t moe = 0;
        if (config.moe) {
            const std::size_t pairs = t * config.top_k_experts;
            const std::size_t w     = config.moe_intermediate_size;
            moe += round_up(t * config.top_k_experts * 8, kAlign) * 2; // ids, weights
            moe += round_up(t * (h + config.num_experts) * 4, kAlign);   // router input, logits
            moe += round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign);
            moe += round_up(pairs * 2 * w * 4, kAlign);
            moe += round_up(pairs * w * 2, kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(w), int(pairs)), kAlign);
            moe += round_up(pairs * h * 4, kAlign);
            moe += round_up(t * h * 8, kAlign);
        }
        const std::size_t decode =
            kernels::decode_attention_workspace_floats(int(options.sequences), query_heads, 512,
                                                       decode_splits_full) * 4 +
            kernels::decode_attention_workspace_floats(int(options.sequences), query_heads, 256,
                                                       decode_splits_sliding) * 4;
        std::size_t total = bytes + moe + round_up(decode, kAlign) + gguf_scratch + (16u << 20);
        if (vision) {
            // A chunk's embeddings, then one tower batch at a time beside them.
            total = std::max(total, round_up(t * h * 4, kAlign) + vision->workspace_bytes());
        } else if (vision_media) {
            // Soft tokens computed on the host: one unit's at a time beside the embeddings.
            total = std::max(total, round_up(t * h * 4, kAlign) + round_up(t * h * 2, kAlign) +
                                        (std::size_t(4) << 20));
        }
        return total;
    }

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    // ---- Shared pieces ---------------------------------------------------------------------------

    void project(const Tensor& x, const Weight& w, Tensor& out) {
        ops::linear(x, w, out, ops::LinearPolicy::AllowA8, *workspace, stream);
    }

    // Up to kMaxVectorColumns rows of width k in the vector kernels' q8_1 form, written by the
    // kernel that produces them instead of a BF16 row and a separate quantization.
    struct Q8Act {
        void* base = nullptr;
        kernels::Q8Rows rows;
    };
    Q8Act alloc_q8(int k, int t) {
        Q8Act q;
        q.base    = workspace->alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
        q.rows.qs = static_cast<std::int8_t*>(q.base);
        q.rows.ds = reinterpret_cast<half2*>(static_cast<std::uint8_t*>(q.base) +
                                             gguf::vector_activation_scales_offset(k, t));
        return q;
    }
    void project_q8(const Q8Act& x, const Weight& w, int t, Tensor& out, int row = 0,
                    int rows = 0) {
        gguf::VectorOutput o;
        o.bf16          = static_cast<__nv_bfloat16*>(out.data) + row;
        o.column_stride = rows != 0 ? rows : w.n;
        gguf::vector_product(ops::detail::gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, w.k,
                             x.base, t, o, stream);
    }

    // A block's Q, K (and V) rows from q8_1 inputs, part by part when they are not one parent.
    void project_qkv_q8(const Q8Act& x, const LayerPlan& p, int t, Tensor& out) {
        int row = 0;
        for (const Weight& part : p.qkv) {
            project_q8(x, part, t, out, row, p.qkv_rows);
            row += part.n;
        }
    }

    static constexpr std::size_t kStagingRegions = 8;

    // Tokens, positions, and when given sequences, key positions, a chunk's media unit per row and
    // its units' first rows, and a branch pass's slots and branch starts, to the device.
    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c, std::span<const std::int32_t> d = {},
               std::span<const std::int32_t> e = {}, std::span<const std::int32_t> f = {},
               std::span<const std::int32_t> g = {}, std::span<const std::int32_t> k = {}) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, options.sequences);
        if (e.size() > rows || f.size() > rows || g.size() > rows || k.size() > rows) {
            throw std::invalid_argument("gemma4 executor: staging overflow");
        }
        std::copy(g.begin(), g.end(), host + 6 * rows);
        std::copy(k.begin(), k.end(), host + 7 * rows);
        if (!g.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_slots.p, host + 6 * rows, g.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!k.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_starts.p, host + 7 * rows, k.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        std::copy(a.begin(), a.end(), host);
        std::copy(b.begin(), b.end(), host + rows);
        std::copy(c.begin(), c.end(), host + 2 * rows);
        std::copy(d.begin(), d.end(), host + 3 * rows);
        std::copy(e.begin(), e.end(), host + 4 * rows);
        std::copy(f.begin(), f.end(), host + 5 * rows);
        if (!e.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_blocks.p, host + 4 * rows, e.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!f.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_unit_rows.p, host + 5 * rows, f.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_tokens.p, host, a.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_positions.p, host + rows, b.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        if (!c.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_sequences.p, host + 2 * rows, c.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!d.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_keys.p, host + 3 * rows, d.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        staged->record(stream);
    }

    // The embeddings of `tokens` ids (the staged tokens by default), scaled, into FP32 h [T][H].
    void embed(int tokens, float* h, const std::int32_t* ids = nullptr) {
        gguf::dequantize_rows(ops::detail::gguf_type(head.qtype), head.qdata, row_bytes(head),
                              hidden, ids != nullptr ? ids : static_cast<const std::int32_t*>(d_tokens.p),
                              tokens, h, hidden, stream);
        kernels::scale(h, std::int64_t(tokens) * hidden, std::sqrt(float(hidden)), stream);
    }

    // The rest of block `p` after its attention output o (BF16 [T][H]): the attention residual,
    // the dense MLP and the routed experts into the residual h, and the normalized input of what
    // follows (`next_norm`: the next block's input norm, or the final norm) into `next`.
    void block_tail(const LayerPlan& p, float* h, int t, const Tensor& o, const float* next_norm,
                    __nv_bfloat16* next, const Q8Act* next_q) {
        auto scope    = workspace->scope();
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor mlp_in = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act mlp_q, moe_q;
        kernels::AttnResidualArgs residual;
        residual.h        = h;
        residual.o        = static_cast<const __nv_bfloat16*>(o.data);
        residual.w_post   = p.post_attention_norm;
        residual.hidden   = hidden;
        residual.tokens   = t;
        residual.eps      = eps;
        residual.ffn_norm = p.ffn_norm;
        if (q8) {
            mlp_q          = alloc_q8(hidden, t);
            residual.mlp_q = mlp_q.rows;
        } else {
            residual.mlp_in = static_cast<__nv_bfloat16*>(mlp_in.data);
        }
        if (p.moe) {
            residual.pre_norm = p.pre_norm;
            if (q8) {
                moe_q          = alloc_q8(hidden, t);
                residual.moe_q = moe_q.rows;
            } else {
                residual.moe_in = static_cast<__nv_bfloat16*>(
                    workspace->alloc_bytes(std::size_t(t) * hidden * 2).data);
            }
            residual.router_scale = p.router_scale;
            residual.router_x     = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        }
        kernels::attn_residual(residual, stream);
        const int width = int(config.intermediate_size);
        Tensor middle   = workspace->alloc(DType::BF16, {width, t});
        if (q8) {
            gguf::vector_swiglu(ops::detail::gguf_type(p.gate_up.qtype), p.gate_up.qdata,
                                row_bytes(p.gate_up), width, hidden, mlp_q.base, t,
                                static_cast<__nv_bfloat16*>(middle.data), width, stream,
                                gguf::GateActivation::GeluTanh);
        } else {
            // The integer matrix kernel takes whole 128-row tiles: the [gate; up] parent at once
            // (2 x 2112 rows), then GeGLU over its FP32 rows.
            auto* plane = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(t) * 2 * width * 4).data);
            ops::detail::GgufProduct product;
            product.weight   = &p.gate_up;
            product.f32      = plane;
            product.f32_rows = 2 * width;
            ops::detail::gguf_project(mlp_in, {&product, 1}, *workspace, stream);
            kernels::geglu_rows(plane, t, width, static_cast<__nv_bfloat16*>(middle.data), stream);
        }
        Tensor dense = workspace->alloc(DType::BF16, {hidden, t});
        project(middle, p.down, dense);
        kernels::FfnCombineArgs combine;
        combine.h         = h;
        combine.hidden    = hidden;
        combine.tokens    = t;
        combine.eps       = eps;
        combine.dense     = static_cast<const __nv_bfloat16*>(dense.data);
        combine.w_post    = p.post_ffn_norm;
        combine.scalar    = p.layer_scalar;
        combine.next_norm = next_norm;
        combine.next_out  = next;
        if (next_q != nullptr) { combine.next_q = next_q->rows; }
        if (p.moe) {
            combine.w_dense = p.mlp_post_norm;
            combine.w_moe   = p.post_norm;
            experts(p, residual.moe_in, q8 ? &moe_q : nullptr, residual.router_x, t, combine);
        }
        kernels::ffn_combine(combine, stream);
    }

    // The routed experts of block `p`; their output lands where `combine` reads it.
    void experts(const LayerPlan& p, const __nv_bfloat16* x, const Q8Act* xq, const float* router_x,
                 int t, kernels::FfnCombineArgs& combine) {
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* ids      = static_cast<std::int32_t*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* weights  = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* logits   = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * e * 4).data);
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        gguf::MoeRouting routing;
        if (t <= gguf::kMaxVectorColumns) {
            auto* base = static_cast<std::int32_t*>(routing_space);
            routing.bounds       = base;
            routing.sorted       = base + e + 1;
            routing.active       = base + e + 1 + pairs;
            routing.active_count = base + e + 1 + pairs + e;
            kernels::route_select(router_x, hidden, t, p.router, p.expert_scale, e, k, logits, ids,
                                  weights, base, base + e + 1, base + e + 1 + pairs,
                                  base + e + 1 + pairs + e, stream);
        } else {
            kernels::route(router_x, hidden, t, p.router, p.expert_scale, e, k, logits, ids,
                           weights, stream);
            routing = gguf::moe_sort_routes(ids, pairs, e, routing_space, stream);
        }
        // A decode or verification call may leave misses to the CPU, which reads the quantized
        // activation and the route weights.
        const bool cpu = cpu_experts && t <= CpuExperts::kMaxTokens && t <= gguf::kMaxVectorColumns;
        void* act = nullptr;
        if (t <= gguf::kMaxVectorColumns) {
            act = xq != nullptr ? xq->base : nullptr;
            if (act == nullptr) {
                act = workspace->alloc_bytes(gguf::vector_activation_bytes(hidden, t)).data;
                gguf::quantize_vector_activation(x, hidden, t, nullptr, act, stream);
            }
        }
        std::optional<ExpertDelivery::Call> call;
        if (delivery) {
            const std::array<ExpertDelivery::Input, 2> inputs{
                ExpertDelivery::Input{act, cpu ? cpu_experts->activation_bytes(t) : 0},
                ExpertDelivery::Input{weights, cpu ? cpu_experts->weight_bytes(t) : 0}};
            call = delivery->begin(std::size_t(p.delivery_layer), ids, std::uint32_t(pairs), stream,
                                   cpu ? std::span<const ExpertDelivery::Input>(inputs)
                                       : std::span<const ExpertDelivery::Input>());
            if (t > gguf::kMaxVectorColumns) { delivery->await(stream); }
        }
        const auto* tables = delivery ? delivery->tables(std::size_t(p.delivery_layer))
                                      : static_cast<const void* const*>(p.tables.p);
        const gguf::MoeTable gate{tables, p.gate_up_row_bytes, width, hidden};
        const gguf::MoeTable up{tables + e, p.gate_up_row_bytes, width, hidden};
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * width, hidden};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, width};
        const int max_active = std::min(pairs, e);
        auto* middle = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(pairs) * width * 2).data);
        if (t <= gguf::kMaxVectorColumns) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2 = workspace->alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            auto* fixed = static_cast<unsigned long long*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 8).data);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights = weights;
            const auto compute = [&](const gguf::MoeRouting& part) {
                gguf::moe_vector_swiglu(p.gate_up_type, gate, up, part, max_active, k, act, t,
                                        middle, chunk, stream, gguf::GateActivation::GeluTanh);
                gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, stream);
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, stream);
            };
            if (call) {
                // Both stages add to the same exact fixed-point sum, zeroed once above, each over
                // its experts of the call's sorted routes. Quantization reads every pair: the
                // cached stage finds the other pairs cleared, the missing stage the cached pairs'
                // middles.
                const auto stage = [&](const std::int32_t* set) {
                    gguf::MoeRouting part = routing;
                    part.active_count     = set;
                    part.active           = set + 1;
                    return part;
                };
                CUDA_CHECK(cudaMemsetAsync(middle, 0, std::size_t(pairs) * width * 2, stream));
                compute(stage(call->cached_experts));
                delivery->await(stream);
                compute(stage(call->missing_experts));
                if (call->cpu != nullptr) {
                    kernels::add_fixed(fixed, call->products, call->cpu, std::int64_t(t) * hidden, stream);
                }
            } else { compute(routing); }
            combine.moe_fixed = fixed;
            return;
        }
        void* rows_act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.gate_up_type, x, hidden, routing, pairs, k, rows_act,
                                             stream);
        auto* gu = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true, rows_act,
                                 gu, stream);
        kernels::geglu_rows(gu, pairs, width, middle, stream);
        void* act2 = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(width, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.down_type, middle, width, routing, pairs, 1, act2,
                                             stream);
        auto* rows = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * hidden * 4).data);
        gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act2, rows,
                                 stream);
        combine.moe_rows    = rows;
        combine.moe_weights = weights;
        combine.top_k       = k;
    }

    // The head of `rows` final-normed rows x into logits_bf16 columns [first, first + rows).
    void head_logits(const __nv_bfloat16* rows_x, int rows, int first) {
        for (int at = 0; at < rows; at += kHeadRows) {
            const int n  = std::min(kHeadRows, rows - at);
            auto scope   = workspace->scope();
            Tensor x(const_cast<__nv_bfloat16*>(rows_x) + std::int64_t(at) * hidden, DType::BF16,
                     {hidden, n});
            auto* plane = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(n) * vocab * 4).data);
            ops::detail::GgufProduct product;
            product.weight   = &head;
            product.f32      = plane;
            product.f32_rows = vocab;
            ops::detail::gguf_project(x, {&product, 1}, *workspace, stream);
            kernels::softcap(plane, std::int64_t(n) * vocab, config.final_logit_softcapping,
                             static_cast<__nv_bfloat16*>(logits_bf16.p) +
                                 std::int64_t(first + at) * vocab,
                             stream);
        }
    }

    // The head of the last `rows` of t <= kMaxVectorColumns final-normed q8_1 rows into
    // logits_bf16 columns [0, rows).
    void head_logits_q8(const Q8Act& x, int t, int rows) {
        auto scope  = workspace->scope();
        auto* plane = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * vocab * 4).data);
        gguf::VectorOutput o;
        o.f32           = plane;
        o.column_stride = vocab;
        gguf::vector_product(ops::detail::gguf_type(head.qtype), head.qdata, row_bytes(head), vocab,
                             head.k, x.base, t, o, stream);
        kernels::softcap(plane + std::int64_t(t - rows) * vocab, std::int64_t(rows) * vocab,
                         config.final_logit_softcapping,
                         static_cast<__nv_bfloat16*>(logits_bf16.p), stream);
    }

    // ---- Prompt chunks -----------------------------------------------------------------------------

    // The soft tokens of a chunk's media into its FP32 embeddings h, item by item; the device
    // tower encodes as many of an item's units at once as its capacity holds.
    void place_media(std::span<const ChunkMedia> media, float* h) {
        const VisionConfig& vc = *model.vision_config();
        const int pooling      = int(vc.pooling_kernel_size);
        std::size_t unit_row   = 0; // this item's first entry of d_unit_rows
        for (const ChunkMedia& item : media) {
            const int per_unit = item.grid_w * item.grid_h;
            const int tokens   = (item.grid_w / pooling) * (item.grid_h / pooling);
            const int units    = int(item.rows.size());
            const int* rows    = static_cast<const std::int32_t*>(d_unit_rows.p) + unit_row;
            unit_row += item.rows.size();
            if (!item.embeddings.empty()) {
                if (item.embeddings.size() != std::size_t(units) * tokens * hidden) {
                    throw std::invalid_argument("gemma4 forward: media embeddings do not match");
                }
                for (int u = 0; u < units; ++u) {
                    auto unit_scope = workspace->scope();
                    auto* soft      = static_cast<__nv_bfloat16*>(
                        workspace->alloc_bytes(std::size_t(tokens) * hidden * 2).data);
                    CUDA_CHECK(cudaMemcpyAsync(
                        soft, item.embeddings.data() + std::size_t(u) * tokens * hidden,
                        std::size_t(tokens) * hidden * 2, cudaMemcpyHostToDevice, stream));
                    vision_kernels::scatter_rows(soft, tokens, tokens, hidden, rows + u, h,
                                                 stream);
                }
                continue;
            }
            if (!vision) { throw std::logic_error("gemma4 forward: no device vision tower"); }
            if (item.patches.size() != std::size_t(units) * per_unit * vc.patch_features() ||
                per_unit <= 0 || std::uint32_t(per_unit) > vision->max_patches()) {
                throw std::invalid_argument("gemma4 forward: media patches do not match");
            }
            const int batch = int(vision->max_patches()) / per_unit;
            for (int u = 0; u < units; u += batch) {
                const int n     = std::min(batch, units - u);
                auto batch_scope = workspace->scope();
                const std::size_t values = std::size_t(n) * per_unit * vc.patch_features();
                auto* patches            = static_cast<__nv_bfloat16*>(
                    workspace->alloc_bytes(values * 2).data);
                CUDA_CHECK(cudaMemcpyAsync(
                    patches,
                    item.patches.data() + std::size_t(u) * per_unit * vc.patch_features(),
                    values * 2, cudaMemcpyHostToDevice, stream));
                auto* soft = static_cast<__nv_bfloat16*>(
                    workspace->alloc_bytes(std::size_t(n) * tokens * hidden * 2).data);
                vision->encode(patches, n, item.grid_w, item.grid_h, soft, *workspace, stream);
                vision_kernels::scatter_rows(soft, n * tokens, tokens, hidden, rows + u, h,
                                             stream);
            }
        }
    }

    // With `branches` (forward_branches): the chunk is those branches' tokens one after another,
    // each continuing the sequence at its position, and `logit_rows` is their count.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const ChunkMedia> media = {},
                 std::span<const std::uint32_t> branches = {}) {
        const int t = int(tokens.size());
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows == 0 || logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows)) {
            throw std::invalid_argument("gemma4 forward: invalid call");
        }
        const bool branching = !branches.empty();
        if (branching && (!media.empty() || logit_rows != branches.size() ||
                          std::accumulate(branches.begin(), branches.end(), std::uint64_t{0}) !=
                              std::uint64_t(t) ||
                          std::find(branches.begin(), branches.end(), 0U) != branches.end())) {
            throw std::invalid_argument("gemma4 forward_branches: invalid branches");
        }
        if (!media.empty() && !vision_media) {
            throw std::invalid_argument("gemma4 forward: media needs an executor with vision");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("gemma4 forward: context capacity exceeded");
        }
        back(sequence, first + std::uint32_t(t), true);
        std::vector<std::int32_t> pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        // A branch pass: each row rotates at its branch's position, and takes the cache slot of
        // its place in the chunk (`slots`, the positions a plain chunk has); `starts` its branch's
        // first row, for the masks.
        std::vector<std::int32_t> slots, starts;
        if (branching) {
            slots = pos;
            starts.resize(t);
            for (std::size_t b = 0, row = 0; b < branches.size(); row += branches[b++]) {
                for (std::uint32_t j = 0; j < branches[b]; ++j) {
                    pos[row + j]    = std::int32_t(first + j);
                    starts[row + j] = std::int32_t(row);
                }
            }
        }
        // Each row's media unit (-1 for text) and each unit's first row.
        std::vector<std::int32_t> blocks, unit_rows;
        if (!media.empty()) {
            const int pooling = int(model.vision_config()->pooling_kernel_size);
            blocks.assign(t, -1);
            for (const ChunkMedia& item : media) {
                const int tokens_per_unit = (item.grid_w / pooling) * (item.grid_h / pooling);
                if (item.grid_w <= 0 || item.grid_h <= 0 || item.grid_w % pooling != 0 ||
                    item.grid_h % pooling != 0) {
                    throw std::invalid_argument("gemma4 forward: invalid media grid");
                }
                for (const std::uint32_t row : item.rows) {
                    if (row + std::uint32_t(tokens_per_unit) > std::uint32_t(t)) {
                        throw std::invalid_argument("gemma4 forward: a media unit leaves the chunk");
                    }
                    const std::int32_t unit = std::int32_t(unit_rows.size());
                    for (int i = 0; i < tokens_per_unit; ++i) {
                        if (blocks[row + i] != -1) {
                            throw std::invalid_argument("gemma4 forward: media units overlap");
                        }
                        blocks[row + i] = unit;
                    }
                    unit_rows.push_back(std::int32_t(row));
                }
            }
        }
        stage(tokens, pos, {}, {}, blocks, unit_rows, slots, starts);
        const auto* d_branch_starts =
            branching ? static_cast<const std::int32_t*>(d_starts.p) : nullptr;
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        embed(t, h);
        media_timed = !media.empty();
        if (media_timed) {
            CUDA_CHECK(cudaEventRecord(media_begin, stream));
            place_media(media, h);
            CUDA_CHECK(cudaEventRecord(media_end, stream));
        }
        // Masks: the sliding ring and the full-attention prefix. With bidirectional vision
        // attention, a media unit's soft tokens see each other in the sliding ring.
        const bool bidirectional = !media.empty() && model.vision_config()->bidirectional_attention;
        const int full_slots = int(round_up(first + t, 256));
        auto* mask_ring = static_cast<__half*>(
            workspace->alloc_bytes(std::size_t(t) * prefill_ring * 2).data);
        auto* mask_full = static_cast<__half*>(this->mask_full.p);
        kernels::build_mask(mask_ring, t, prefill_ring, int(first), prefill_ring,
                            int(config.sliding_window), stream,
                            bidirectional ? static_cast<const std::int32_t*>(d_blocks.p) : nullptr,
                            d_branch_starts);
        const int window = int(config.sliding_window);
        kernels::build_mask(mask_full, t, full_slots, int(first), 0, 0, stream, nullptr,
                            d_branch_starts);
        // Each block's normalized input rows; after the last block, the final-normed rows (q8_1
        // for a vector-width chunk).
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor x      = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act xq;
        if (q8) { xq = alloc_q8(hidden, t); }
        kernels::rms_norm(h, layers.front().input_norm, hidden, t, eps,
                          q8 ? nullptr : static_cast<__nv_bfloat16*>(x.data), stream, xq.rows);
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            Tensor qkv = workspace->alloc(DType::BF16, {p.qkv_rows, t});
            if (q8) {
                project_qkv_q8(xq, p, t, qkv);
            } else if (p.qkv.size() == 1) {
                project(x, p.qkv.front(), qkv);
            } else {
                std::vector<ops::detail::GgufProduct> parts(p.qkv.size());
                for (std::size_t i = 0, row = 0; i < parts.size(); row += p.qkv[i].n, ++i) {
                    parts[i].weight = &p.qkv[i];
                    parts[i].out    = &qkv;
                    parts[i].row    = std::int32_t(row);
                }
                ops::detail::gguf_project(x, parts, *workspace, stream);
            }
            const int q_floats = query_heads * p.head_dim;
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = static_cast<const __nv_bfloat16*>(qkv.data);
            qa.tokens      = t;
            qa.query_heads = query_heads;
            qa.kv_heads    = p.kv_heads;
            qa.head_dim    = p.head_dim;
            qa.has_v       = !p.full;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = static_cast<const float*>(p.full ? inv_freq_full.p : inv_freq_sliding.p);
            qa.q_scale     = p.full ? full_query_scale : 1.0f;
            qa.q_out       = q;
            qa.slots       = branching ? static_cast<const std::int32_t*>(d_slots.p) : nullptr;
            // A sliding block attends over the staging ring: the window before the chunk first
            // comes over from the sequence's ring, the chunk's keys land there, and the window that
            // ends the chunk goes back.
            const std::int64_t seq_offset   = std::int64_t(sequence) * p.seq_stride;
            const std::int64_t stage_stride = std::int64_t(prefill_ring) * p.head_dim;
            const bool quantized = kv.quantized();
            kernels::RingCopyArgs keep;
            keep.kv_heads = p.kv_heads;
            keep.head_dim = p.head_dim;
            if (!p.full && quantized) {
                // The window before the chunk, decoded from the sequence's quantized ring.
                for (const auto& [plane, dst] : {std::pair{&p.kq, &staging_k}, std::pair{&p.vq, &staging_v}}) {
                    kernels::KvLoadArgs load;
                    load.src             = *plane;
                    load.src_ring        = p.ring;
                    load.sequence        = int(sequence);
                    load.begin           = std::max(0, int(first) - window);
                    load.end             = int(first);
                    load.heads           = p.kv_heads;
                    load.dst             = static_cast<__half*>(dst->p);
                    load.dst_head_stride = stage_stride;
                    load.dst_ring        = prefill_ring;
                    kernels::kv_load(load, stream);
                }
            } else if (!p.full) {
                keep.src_k           = static_cast<const __half*>(p.k.p) + seq_offset;
                keep.src_v           = static_cast<const __half*>(p.v.p) + seq_offset;
                keep.dst_k           = static_cast<__half*>(staging_k.p);
                keep.dst_v           = static_cast<__half*>(staging_v.p);
                keep.src_head_stride = p.head_stride;
                keep.dst_head_stride = stage_stride;
                keep.src_ring        = p.ring;
                keep.dst_ring        = prefill_ring;
                keep.begin           = std::max(0, int(first) - window);
                keep.end             = int(first);
                kernels::ring_copy(keep, stream);
            }
            // A quantized full block's chunk values land in F16 first (full_values, linear).
            __half* full_v = quantized ? static_cast<__half*>(full_values.p) : static_cast<__half*>(p.v.p);
            qa.k_cache     = p.full ? nullptr : static_cast<__half*>(staging_k.p);
            qa.v_cache     = p.full ? full_v : static_cast<__half*>(staging_v.p);
            qa.seq_stride  = p.full && !quantized ? p.seq_stride : 0;
            qa.head_stride = p.full ? p.head_stride : stage_stride;
            qa.sequence    = p.full && !quantized ? int(sequence) : 0;
            qa.ring        = p.full ? 0 : prefill_ring;
            kernels::qkv_post(qa, stream);
            if (quantized) {
                // The chunk's rows into the sequence's planes: a sliding block's into its ring (the
                // staging rows become the values they represent, which the chunk then attends to),
                // a full block's at their positions, after which the block's whole prefix comes
                // back from the plane.
                for (const auto& [plane, src] :
                     {std::pair{&p.kq, &staging_k}, std::pair{&p.vq, &staging_v}}) {
                    if (p.full && plane == &p.kq) { continue; }
                    kernels::KvStoreArgs store;
                    store.src             = p.full ? full_v : static_cast<__half*>(src->p);
                    store.src_head_stride = p.full ? p.head_stride : stage_stride;
                    store.src_ring        = p.full ? global_slots : prefill_ring;
                    store.positions =
                        static_cast<const std::int32_t*>(branching ? d_slots.p : d_positions.p);
                    store.sequence        = int(sequence);
                    store.rows            = t;
                    store.heads           = p.kv_heads;
                    store.dst             = *plane;
                    store.dst_ring        = p.ring;
                    // A branch pass keeps the sequence's window as it was: its rows go nowhere.
                    store.store_from      = p.full      ? 0
                                            : branching ? std::numeric_limits<int>::max()
                                                        : int(first) + t - p.ring;
                    store.write_back      = !p.full;
                    kernels::kv_store(store, stream);
                }
                if (p.full) {
                    kernels::KvLoadArgs load;
                    load.src             = p.vq;
                    load.sequence        = int(sequence);
                    load.begin           = 0;
                    load.end             = int(first) + t;
                    load.heads           = p.kv_heads;
                    load.dst             = full_v;
                    load.dst_head_stride = p.head_stride;
                    kernels::kv_load(load, stream);
                }
            }
            if (p.full) {
                kernels::derive_keys(
                    full_v + (quantized ? 0 : seq_offset), p.head_stride, p.kv_heads, p.head_dim,
                    int(first) + t, p.k_norm, static_cast<const float2*>(rope_full.p),
                    int(config.rotary_pairs_full), static_cast<__half*>(derived_keys.p), stream,
                    branching ? static_cast<const std::int32_t*>(d_positions.p) : nullptr,
                    int(first));
            }
            auto* attn = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            const std::size_t fa_bytes =
                gguf::flash_attention_workspace_bytes(p.head_dim, query_heads, p.kv_heads, t);
            void* fa_space = workspace->alloc_bytes(fa_bytes).data;
            gguf::FlashAttentionArgs fa;
            fa.head_dim        = p.head_dim;
            fa.query_heads     = query_heads;
            fa.kv_heads        = p.kv_heads;
            fa.tokens          = t;
            fa.q               = q;
            fa.q_token_stride  = q_floats;
            fa.k = p.full ? static_cast<const __half*>(derived_keys.p)
                          : static_cast<const __half*>(staging_k.p);
            fa.v = p.full ? full_v + (quantized ? 0 : seq_offset)
                          : static_cast<const __half*>(staging_v.p);
            fa.row_bytes       = std::int64_t(p.head_dim) * 2;
            fa.head_bytes      = (p.full ? p.head_stride : stage_stride) * 2;
            fa.slots           = p.full ? full_slots : prefill_ring;
            fa.mask            = p.full ? mask_full : mask_ring;
            fa.mask_row_halves = fa.slots;
            fa.out             = attn;
            fa.scale           = 1.0f;
            gguf::flash_attention(fa, fa_space, fa_bytes, stream);
            if (!p.full && !quantized && !branching) {
                kernels::RingCopyArgs back = keep;
                std::swap(back.src_head_stride, back.dst_head_stride);
                std::swap(back.src_ring, back.dst_ring);
                back.src_k = static_cast<const __half*>(staging_k.p);
                back.src_v = static_cast<const __half*>(staging_v.p);
                back.dst_k = static_cast<__half*>(p.k.p) + seq_offset;
                back.dst_v = static_cast<__half*>(p.v.p) + seq_offset;
                back.begin = std::max(0, int(first) + t - window);
                back.end   = int(first) + t;
                kernels::ring_copy(back, stream);
            }
            Tensor attn_bf16 = workspace->alloc(DType::BF16, {q_floats, t});
            kernels::f32_to_bf16(attn, static_cast<__nv_bfloat16*>(attn_bf16.data),
                                 std::int64_t(t) * q_floats, stream);
            Tensor o = workspace->alloc(DType::BF16, {hidden, t});
            project(attn_bf16, p.output, o);
            const bool last = li + 1 == layers.size();
            // The final rows in BF16 too when the assistant reads the last of them, or a branch
            // pass gathers its branches' last ones.
            block_tail(p, h, t, o, last ? final_norm : layers[li + 1].input_norm,
                       q8 && !(last && (drafting() || branching))
                           ? nullptr
                           : static_cast<__nv_bfloat16*>(x.data),
                       q8 ? &xq : nullptr);
        }
        if (branching) {
            // Each branch's last row to the front, in branch order; a branch's last row lies at or
            // past its index, so no copy overwrites a later one's source. The sequence stays at
            // its position, its window untouched.
            auto* rows      = static_cast<__nv_bfloat16*>(x.data);
            std::size_t row = 0;
            for (std::size_t b = 0; b < branches.size(); ++b) {
                row += branches[b];
                if (row - 1 == b) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(
                    rows + std::int64_t(b) * hidden, rows + std::int64_t(row - 1) * hidden,
                    std::size_t(hidden) * 2, cudaMemcpyDeviceToDevice, stream));
            }
            head_logits(rows, int(logit_rows), 0);
            return;
        }
        if (drafting()) {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<__nv_bfloat16*>(anchor_hidden.p) +
                                           std::int64_t(sequence) * hidden,
                                       static_cast<const __nv_bfloat16*>(x.data) +
                                           std::int64_t(t - 1) * hidden,
                                       std::size_t(hidden) * 2, cudaMemcpyDeviceToDevice, stream));
            anchored[sequence] = 1;
        }
        if (q8) {
            head_logits_q8(xq, t, int(logit_rows));
        } else {
            head_logits(static_cast<const __nv_bfloat16*>(x.data) +
                            std::int64_t(t - int(logit_rows)) * hidden,
                        int(logit_rows), 0);
        }
        positions[sequence] = first + std::uint32_t(t);
    }

    // ---- Decode ---------------------------------------------------------------------------------------

    [[nodiscard]] bool drafting() const noexcept { return options.draft_tokens > 0; }

    // One token per row; tokens, positions and sequences already staged on the device. With the
    // assistant, the rows' final hidden states land in row_hidden, and a decode step (not a
    // verification, whose kept row commit() picks) makes them its sequences' anchors.
    // A quantized cache: the b decode rows' F16 K and V (kv_rows) into the block's planes at their
    // positions, and the rows' queries turned for a rotated key plane, in one launch.
    void store_rows(const LayerPlan& p, int b, float* q) {
        if (b > kKvRows) { throw std::invalid_argument("gemma4: too many decode rows"); }
        kernels::KvRowsArgs rows;
        rows.k           = static_cast<const __half*>(kv_rows_k.p);
        rows.v           = static_cast<const __half*>(kv_rows_v.p);
        rows.head_stride = std::int64_t(kKvRows) * p.head_dim;
        rows.positions   = static_cast<const std::int32_t*>(d_positions.p);
        rows.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
        rows.rows        = b;
        rows.heads       = p.kv_heads;
        rows.ring        = p.ring;
        if (!p.full) { rows.k_plane = p.kq; }
        rows.v_plane = p.vq;
        if (!p.full && p.kq.rotated) {
            rows.q      = q;
            rows.q_rows = b * query_heads;
        }
        kernels::kv_store_rows(rows, stream);
    }

    // Queries meeting a sliding block's rotated key plane turn with it.
    void turn_queries(const LayerPlan& p, float* q, int rows) {
        if (!p.full && kv.quantized() && p.kq.rotated) {
            kernels::rotate_rows(q, rows * query_heads, stream);
        }
    }

    void set_caches(kernels::DecodeAttentionArgs& da, const LayerPlan& p) {
        if (kv.quantized()) {
            if (!p.full) { da.k_plane = p.kq; }
            da.v_plane = p.vq;
        } else {
            da.k_cache = p.full ? nullptr : static_cast<const __half*>(p.k.p);
            da.v_cache = static_cast<const __half*>(p.v.p);
        }
    }

    void decode_body(int b, bool verifying = false) {
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        embed(b, h);
        // Each block's normalized input rows in q8_1 form; after the last block, the final-normed
        // rows.
        const Q8Act xq = alloc_q8(hidden, b);
        kernels::rms_norm(h, layers.front().input_norm, hidden, b, eps, nullptr, stream, xq.rows);
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            Tensor qkv = workspace->alloc(DType::BF16, {p.qkv_rows, b});
            project_qkv_q8(xq, p, b, qkv);
            const int q_floats = query_heads * p.head_dim;
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = static_cast<const __nv_bfloat16*>(qkv.data);
            qa.tokens      = b;
            qa.query_heads = query_heads;
            qa.kv_heads    = p.kv_heads;
            qa.head_dim    = p.head_dim;
            qa.has_v       = !p.full;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = static_cast<const float*>(p.full ? inv_freq_full.p : inv_freq_sliding.p);
            qa.q_scale     = p.full ? full_query_scale : 1.0f;
            qa.q_out       = q;
            qa.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            if (kv.quantized()) {
                // The rows in F16 first, then stored into the planes.
                qa.k_cache     = p.full ? nullptr : static_cast<__half*>(kv_rows_k.p);
                qa.v_cache     = static_cast<__half*>(kv_rows_v.p);
                qa.head_stride = std::int64_t(kKvRows) * p.head_dim;
                qa.row_slots   = true;
                kernels::qkv_post(qa, stream);
                store_rows(p, b, q);
            } else {
                qa.k_cache     = p.full ? nullptr : static_cast<__half*>(p.k.p);
                qa.v_cache     = static_cast<__half*>(p.v.p);
                qa.seq_stride  = p.seq_stride;
                qa.head_stride = p.head_stride;
                qa.ring        = p.ring;
                kernels::qkv_post(qa, stream);
            }
            const int splits = p.full ? decode_splits_full : decode_splits_sliding;
            auto* partial    = static_cast<float*>(
                workspace->alloc_bytes(kernels::decode_attention_workspace_floats(
                                           b, query_heads, p.head_dim, splits) * 4).data);
            const Q8Act attn = alloc_q8(q_floats, b);
            kernels::DecodeAttentionArgs da;
            da.q           = q;
            set_caches(da, p);
            da.k_norm      = p.k_norm;
            da.rope        = static_cast<const float2*>(rope_full.p);
            da.rope_pairs  = int(config.rotary_pairs_full);
            da.seq_stride  = p.seq_stride;
            da.head_stride = p.head_stride;
            da.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            da.positions   = static_cast<const std::int32_t*>(d_positions.p);
            da.batch       = b;
            da.query_heads = query_heads;
            da.kv_heads    = p.kv_heads;
            da.head_dim    = p.head_dim;
            da.ring        = p.ring;
            da.window      = int(config.sliding_window);
            da.splits      = splits;
            da.scale       = 1.0f;
            da.partial     = partial;
            da.out_q       = attn.rows;
            kernels::decode_attention(da, stream);
            Tensor o = workspace->alloc(DType::BF16, {hidden, b});
            project_q8(attn, p.output, b, o);
            const bool last = li + 1 == layers.size();
            block_tail(p, h, b, o, last ? final_norm : layers[li + 1].input_norm,
                       last && drafting() ? static_cast<__nv_bfloat16*>(row_hidden.p) : nullptr,
                       &xq);
        }
        head_logits_q8(xq, b, b);
        if (drafting() && !verifying) {
            kernels::copy_rows(static_cast<const __nv_bfloat16*>(row_hidden.p), nullptr,
                               static_cast<__nv_bfloat16*>(anchor_hidden.p),
                               static_cast<const std::int32_t*>(d_sequences.p), b, hidden, stream);
        }
    }

    // Runs `body` from the graph of `key`: eagerly the first time, so every kernel is loaded
    // before the capture, then captured and replayed.
    template <typename Body>
    void run_graph(std::map<std::uint32_t, GraphEntry>& table, std::uint32_t key, Body&& body) {
        if (!options.cuda_graphs) {
            body();
            return;
        }
        GraphEntry& entry = table[key];
        if (!entry.warmed) {
            body();
            entry.warmed = true;
            return;
        }
        if (!entry.executable.ready()) {
            entry.definition.capture(stream, [&] { body(); });
            entry.executable.instantiate(entry.definition);
        }
        entry.executable.launch(stream);
    }

    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int b = int(sequences.size());
        if (b <= 0 || std::uint32_t(b) > options.sequences || tokens.size() != sequences.size() ||
            b > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("gemma4 decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences) {
                throw std::invalid_argument("gemma4 decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("gemma4 decode: context capacity exceeded");
            }
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        for (int i = 0; i < b; ++i) { back(sequences[i], positions[sequences[i]] + 1, false); }
        stage(tokens, pos, seq);
        run_graph(graphs, std::uint32_t(b), [&] { decode_body(b); });
        for (int i = 0; i < b; ++i) {
            positions[sequences[i]] += 1;
            if (drafting()) { anchored[sequences[i]] = 1; }
        }
    }

    // ---- Speculative rounds ------------------------------------------------------------------------

    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int n = int(sequences.size());
        if (!drafting() || n <= 0 || tokens.size() % sequences.size() != 0) {
            throw std::invalid_argument("gemma4 verify: invalid call");
        }
        const int w = int(tokens.size()) / n, rows = n * w;
        if (w < 1 || std::uint32_t(w) > options.draft_tokens + 1 || rows > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("gemma4 verify: invalid width");
        }
        std::vector<std::int32_t> pos(rows), seq(rows);
        for (int j = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + j, s) != 0) {
                throw std::invalid_argument("gemma4 verify: invalid sequence");
            }
            if (positions[s] + std::uint32_t(w) > options.max_context) {
                throw std::invalid_argument("gemma4 verify: context capacity exceeded");
            }
            for (int i = 0; i < w; ++i) {
                pos[j * w + i] = std::int32_t(positions[s]) + i;
                seq[j * w + i] = std::int32_t(s);
            }
        }
        for (const std::uint32_t s : sequences) { back(s, positions[s] + std::uint32_t(w), false); }
        stage(tokens, pos, seq);
        run_graph(graphs, 16 + std::uint32_t(rows), [&] { decode_body(rows, true); });
        verify_width = std::uint32_t(w);
    }

    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept) {
        if (kept.size() != sequences.size() || verify_width == 0) {
            throw std::invalid_argument("gemma4 commit: invalid call");
        }
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || kept[j] < 1 || kept[j] > verify_width) {
                throw std::invalid_argument("gemma4 commit: invalid row");
            }
            positions[s] += kept[j];
            // The anchor's hidden state: the row of the last token kept.
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<__nv_bfloat16*>(anchor_hidden.p) + std::int64_t(s) * hidden,
                static_cast<const __nv_bfloat16*>(row_hidden.p) +
                    std::int64_t(j * verify_width + kept[j] - 1) * hidden,
                std::size_t(hidden) * 2, cudaMemcpyDeviceToDevice, stream));
            anchored[s] = 1;
        }
        verify_width = 0;
    }

    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::span<std::int32_t> out, std::uint32_t steps) {
        const int n = int(sequences.size());
        if (!drafting() || n <= 0 || n > gguf::kMaxVectorColumns ||
            anchors.size() != sequences.size() || steps < 1 || steps > options.draft_tokens ||
            out.size() < std::size_t(n) * steps) {
            throw std::invalid_argument("gemma4 draft: invalid call");
        }
        std::vector<std::int32_t> pos(n), seq(n), keys(n);
        for (int j = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || !anchored[s] || positions[s] == 0) {
                throw std::invalid_argument("gemma4 draft: the sequence has no anchor state");
            }
            // The chain runs at the anchor's position and sees the keys before it.
            pos[j]  = std::int32_t(positions[s]);
            keys[j] = std::int32_t(positions[s]) - 1;
            seq[j]  = std::int32_t(s);
        }
        stage(anchors, pos, seq, keys);
        run_graph(draft_graphs, std::uint32_t(n) << 8 | steps, [&] { draft_body(n, int(steps)); });
        auto* host = static_cast<std::int32_t*>(h_drafts->data());
        CUDA_CHECK(cudaMemcpyAsync(host, d_drafts.p, std::size_t(n) * steps * 4,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int j = 0; j < n; ++j) {
            for (std::uint32_t i = 0; i < steps; ++i) { out[j * steps + i] = host[i * n + j]; }
        }
    }

    // The assistant's chain over n rows: anchors, positions, sequences and key limits staged.
    void draft_body(int n, int steps) {
        workspace->reset();
        auto scope        = workspace->scope();
        const int a       = draft_hidden;
        auto* hidden_in   = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(n) * hidden * 2).data);
        auto* embedding   = static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * hidden * 4).data);
        auto* stem        = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(n) * 2 * hidden * 2).data);
        void* stem_q      = workspace->alloc_bytes(gguf::vector_activation_bytes(2 * hidden, n)).data;
        auto* h           = static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * a * 4).data);
        const Q8Act xq    = alloc_q8(a, n);
        auto* draft_logits = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(n) * vocab * 2).data);
        kernels::copy_rows(static_cast<const __nv_bfloat16*>(anchor_hidden.p),
                           static_cast<const std::int32_t*>(d_sequences.p), hidden_in, nullptr, n,
                           hidden, stream);
        const std::int32_t* ids = static_cast<const std::int32_t*>(d_tokens.p);
        for (int step = 0; step < steps; ++step) {
            embed(n, embedding, ids);
            kernels::stem_rows(embedding, hidden_in, n, hidden, stem, stream);
            gguf::quantize_vector_activation(stem, 2 * hidden, n, nullptr, stem_q, stream);
            gguf::VectorOutput stem_out;
            stem_out.f32           = h;
            stem_out.column_stride = a;
            gguf::vector_product(ops::detail::gguf_type(pre_projection.qtype), pre_projection.qdata,
                                 row_bytes(pre_projection), a, 2 * hidden, stem_q, n, stem_out,
                                 stream);
            kernels::rms_norm(h, draft_layers.front().input_norm, a, n, draft_eps, nullptr, stream,
                              xq.rows);
            for (std::size_t li = 0; li < draft_layers.size(); ++li) {
                draft_layer(li, n, h, xq);
            }
            // xq now holds the final-normed rows: the head's argmax is the draft.
            gguf::VectorOutput head_out;
            head_out.bf16          = draft_logits;
            head_out.column_stride = vocab;
            gguf::vector_product(ops::detail::gguf_type(draft_head.qtype), draft_head.qdata,
                                 row_bytes(draft_head), vocab, a, xq.base, n, head_out, stream);
            auto* drafted = static_cast<std::int32_t*>(d_drafts.p) + std::int64_t(step) * n;
            const Tensor logits(draft_logits, DType::BF16, {vocab, n});
            Tensor tokens(drafted, DType::I32, {n});
            ops::argmax(logits, tokens, draft_domain, stream);
            ids = drafted;
            if (step + 1 < steps) {
                gguf::VectorOutput next;
                next.bf16          = hidden_in;
                next.column_stride = hidden;
                gguf::vector_product(ops::detail::gguf_type(post_projection.qtype),
                                     post_projection.qdata, row_bytes(post_projection), hidden, a,
                                     xq.base, n, next, stream);
            }
        }
    }

    // Block `li` of the assistant over its residual h; xq holds the block's normalized input in q8_1
    // form and receives the next block's (after the last, the final-normed rows).
    void draft_layer(std::size_t li, int n, float* h, const Q8Act& xq) {
        const DraftLayer& d = draft_layers[li];
        const LayerPlan& t  = layers[d.target];
        const int a         = draft_hidden;
        auto scope          = workspace->scope();
        const int q_floats  = query_heads * d.head_dim;
        Tensor qb           = workspace->alloc(DType::BF16, {q_floats, n});
        project_q8(xq, d.query, n, qb);
        auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * q_floats * 4).data);
        kernels::QkvArgs qa;
        qa.qkv         = static_cast<const __nv_bfloat16*>(qb.data);
        qa.tokens      = n;
        qa.query_heads = query_heads;
        qa.kv_heads    = 0; // the queries only
        qa.head_dim    = d.head_dim;
        qa.q_norm      = d.q_norm;
        qa.eps         = draft_eps;
        qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
        qa.inv_freq    = static_cast<const float*>(d.full ? inv_freq_full.p : inv_freq_sliding.p);
        qa.q_scale     = d.full ? full_query_scale : 1.0f;
        qa.q_out       = q;
        kernels::qkv_post(qa, stream);
        turn_queries(t, q, n);
        const int splits = d.full ? decode_splits_full : decode_splits_sliding;
        auto* partial    = static_cast<float*>(
            workspace->alloc_bytes(kernels::decode_attention_workspace_floats(
                                       n, query_heads, d.head_dim, splits) * 4).data);
        const Q8Act attn = alloc_q8(q_floats, n);
        kernels::DecodeAttentionArgs da;
        da.q           = q;
        set_caches(da, t);
        da.k_norm      = t.k_norm;
        da.rope        = static_cast<const float2*>(rope_full.p);
        da.rope_pairs  = int(config.rotary_pairs_full);
        da.seq_stride  = t.seq_stride;
        da.head_stride = t.head_stride;
        da.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
        da.positions   = static_cast<const std::int32_t*>(d_keys.p);
        da.batch       = n;
        da.query_heads = query_heads;
        da.kv_heads    = t.kv_heads;
        da.head_dim    = d.head_dim;
        da.ring        = t.ring;
        da.window      = int(config.sliding_window);
        da.splits      = splits;
        da.scale       = 1.0f;
        da.partial     = partial;
        da.out_q       = attn.rows;
        kernels::decode_attention(da, stream);
        Tensor o = workspace->alloc(DType::BF16, {a, n});
        project_q8(attn, d.output, n, o);
        const Q8Act mlp_q = alloc_q8(a, n);
        kernels::AttnResidualArgs residual;
        residual.h        = h;
        residual.o        = static_cast<const __nv_bfloat16*>(o.data);
        residual.w_post   = d.post_attention_norm;
        residual.hidden   = a;
        residual.tokens   = n;
        residual.eps      = draft_eps;
        residual.ffn_norm = d.ffn_norm;
        residual.mlp_q    = mlp_q.rows;
        kernels::attn_residual(residual, stream);
        Tensor middle = workspace->alloc(DType::BF16, {draft_width, n});
        gguf::vector_swiglu(ops::detail::gguf_type(d.gate_up.qtype), d.gate_up.qdata,
                            row_bytes(d.gate_up), draft_width, a, mlp_q.base, n,
                            static_cast<__nv_bfloat16*>(middle.data), draft_width, stream,
                            gguf::GateActivation::GeluTanh);
        Tensor dense = workspace->alloc(DType::BF16, {a, n});
        project(middle, d.down, dense);
        kernels::FfnCombineArgs combine;
        combine.h         = h;
        combine.hidden    = a;
        combine.tokens    = n;
        combine.eps       = draft_eps;
        combine.dense     = static_cast<const __nv_bfloat16*>(dense.data);
        combine.w_post    = d.post_ffn_norm;
        combine.scalar    = d.layer_scalar;
        combine.next_norm = li + 1 < draft_layers.size() ? draft_layers[li + 1].input_norm
                                                         : draft_final_norm;
        combine.next_q    = xq.rows;
        kernels::ffn_combine(combine, stream);
    }

    void warm_up() {
        // The prompt route once (kernels load at first launch), then each decode batch's graph.
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        forward(0, prompt, 1);
        if (vision) {
            // The tower's routes on one unit of a 70-token image and on two (video frames).
            const int grid_w = 30, grid_h = 21, per_unit = grid_w * grid_h;
            const VisionConfig& vc = *model.vision_config();
            for (int units = 1; units <= 2 && std::uint32_t(units * per_unit) <= vision->max_patches();
                 ++units) {
                auto scope         = workspace->scope();
                const std::size_t values = std::size_t(units) * per_unit * vc.patch_features();
                auto* patches = static_cast<__nv_bfloat16*>(workspace->alloc_bytes(values * 2).data);
                CUDA_CHECK(cudaMemsetAsync(patches, 0, values * 2, stream));
                const std::size_t tokens = std::size_t(units) * per_unit / vc.patches_per_token();
                auto* soft = static_cast<__nv_bfloat16*>(
                    workspace->alloc_bytes(tokens * std::size_t(hidden) * 2).data);
                vision->encode(patches, units, grid_w, grid_h, soft, *workspace, stream);
            }
        }
        if (drafting()) {
            // The assistant's and the verification's kernels, eagerly.
            const std::uint32_t s = 0;
            std::vector<std::int32_t> drafts(options.draft_tokens);
            draft({&s, 1}, std::vector<std::int32_t>{2}, drafts, options.draft_tokens);
            std::vector<std::int32_t> tokens(options.draft_tokens + 1, 2);
            verify({&s, 1}, tokens);
            const std::uint32_t one = 1;
            commit({&s, 1}, {&one, 1});
            anchored[0] = 0;
        }
        positions[0] = 0;
        std::vector<std::uint32_t> sequences;
        std::vector<std::int32_t> tokens;
        for (std::uint32_t b = 1; b <= std::min<std::uint32_t>(options.sequences, 8); ++b) {
            sequences.push_back(b - 1);
            tokens.push_back(2);
            for (int pass = 0; pass < 2; ++pass) { // eagerly, then captured
                decode(sequences, tokens);
                for (const std::uint32_t s : sequences) { positions[s] = 0; }
            }
        }
        std::fill(anchored.begin(), anchored.end(), 0);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (delivery) {
            // Drained, so warm-up deliveries and their asynchronous prefetches stay out of the
            // request counters, and the cache starts from the routing profile rather than the
            // warm-up's routes. Runtime readers use the same mutex and report request work only.
            delivery->drain();
            delivery->prime(options.expert_profile ? &*options.expert_profile : nullptr);
            std::lock_guard lock(expert_stats_mutex);
            expert_startup_stats = delivery->stats();
        }
    }

    // A sliding block's ring buffers of one sequence, (start, bytes): F16 K and V, or a quantized
    // cache's K codes, K scales, V codes and V scales.
    std::vector<std::pair<std::uint8_t*, std::size_t>> ring_buffers(const LayerPlan& p,
                                                                    std::uint32_t sequence) const {
        if (!kv.quantized()) {
            const std::size_t bytes = std::size_t(p.seq_stride) * sizeof(__half);
            return {{static_cast<std::uint8_t*>(p.k.p) + sequence * bytes, bytes},
                    {static_cast<std::uint8_t*>(p.v.p) + sequence * bytes, bytes}};
        }
        const auto at = [&](const kernels::KvPlaneView& v, bool scales) {
            const std::int64_t seq = scales ? v.scale_seq : v.code_seq;
            return std::pair{(scales ? v.scales : v.codes) + sequence * seq, std::size_t(seq)};
        };
        return {at(p.kq, false), at(p.kq, true), at(p.vq, false), at(p.vq, true)};
    }

    void copy_rings(std::uint32_t sequence, bool save) {
        if (!options.checkpoints) { throw std::logic_error("gemma4: an executor without checkpoints"); }
        for (const LayerPlan& p : layers) {
            if (p.full) { continue; }
            const auto rings = ring_buffers(p, sequence);
            for (std::size_t i = 0; i < rings.size(); ++i) {
                auto* kept = static_cast<std::uint8_t*>(p.checkpoint[i].p) + sequence * rings[i].second;
                CUDA_CHECK(cudaMemcpyAsync(save ? kept : rings[i].first, save ? rings[i].first : kept,
                                           rings[i].second, cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    // A full-attention block's row buffers of one sequence, (start, bytes per row, bytes between
    // heads): F16 values, or a quantized plane's codes and scales.
    struct FullRows {
        std::uint8_t* base;
        std::size_t row, head;
    };
    std::vector<FullRows> full_rows(const LayerPlan& p, std::uint32_t sequence) const {
        if (!kv.quantized()) {
            const std::size_t head = std::size_t(p.head_stride) * sizeof(__half);
            return {{static_cast<std::uint8_t*>(p.v.p) + sequence * head * p.kv_heads,
                     std::size_t(p.head_dim) * sizeof(__half), head}};
        }
        const kernels::KvPlaneView& v = p.vq;
        return {{v.codes + sequence * v.code_seq, std::size_t(v.code_row), std::size_t(v.code_head)},
                {v.scales + sequence * v.scale_seq, std::size_t(v.scale_row), std::size_t(v.scale_head)}};
    }

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const LayerPlan& p : layers) {
            if (p.full) {
                for (const FullRows& r : full_rows(p, 0)) { bytes += std::uint64_t(p.kv_heads) * position * r.row; }
            } else {
                for (const auto& ring : ring_buffers(p, 0)) { bytes += ring.second; }
            }
        }
        return bytes;
    }

    // Copies a sequence's image between the device and the host pages: the rings (from or to its
    // checkpoint with `checkpoint`) and the full-attention rows below `position`.
    void copy_image(std::uint32_t sequence, std::uint32_t position, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save, bool checkpoint) {
        if (page_bytes == 0 || pages.size() < (image_bytes(position) + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("gemma4 image: pages do not hold the image");
        }
        std::size_t at = 0;
        const auto copy = [&](std::uint8_t* device, std::size_t bytes) {
            while (bytes > 0) {
                const std::size_t in_page = std::min(bytes, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                if (save) {
                    CUDA_CHECK(cudaMemcpyAsync(host, device, in_page, cudaMemcpyDeviceToHost, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(device, host, in_page, cudaMemcpyHostToDevice, stream));
                }
                device += in_page;
                bytes -= in_page;
                at += in_page;
            }
        };
        for (const LayerPlan& p : layers) {
            if (p.full) {
                for (const FullRows& r : full_rows(p, sequence)) {
                    for (int h = 0; h < p.kv_heads; ++h) { copy(r.base + h * r.head, std::size_t(position) * r.row); }
                }
                continue;
            }
            const auto rings = ring_buffers(p, sequence);
            for (std::size_t i = 0; i < rings.size(); ++i) {
                std::uint8_t* device =
                    checkpoint ? static_cast<std::uint8_t*>(p.checkpoint[i].p) + sequence * rings[i].second
                               : rings[i].first;
                copy(device, rings[i].second);
            }
        }
    }

    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const std::uint32_t position = at != nullptr ? at->position : positions.at(sequence);
        copy_image(sequence, position, pages, page_bytes, true, at != nullptr);
        return {position};
    }

    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        if (image.position > options.max_context) {
            throw std::invalid_argument("gemma4 load_image: invalid image");
        }
        back(sequence, image.position, false);
        copy_image(sequence, image.position, pages, page_bytes, false, false);
        positions.at(sequence) = image.position;
        if (drafting()) { anchored[sequence] = 0; }
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        copy_rings(sequence, true);
        out.position = positions[sequence];
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        copy_rings(sequence, false);
        positions[sequence] = from.position;
        if (drafting()) { anchored[sequence] = 0; }
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    const auto& tower = impl_->vision;
    return ExecutorMemory{.kv_bytes            = impl_->kv_bytes,
                          .kv_backed_bytes     = impl_->kv_backed.load(std::memory_order_relaxed),
                          .kv_grows            = impl_->growing,
                          .workspace_bytes     = impl_->workspace_bytes + impl_->mask_full.bytes,
                          .vision_encode_bytes = tower ? tower->workspace_bytes() : 0,
                          .vision_owned_bytes  = tower ? tower->owned_bytes() : 0,
                          .experts = impl_->delivery ? impl_->delivery->memory() : ExpertDeliveryMemory{}};
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

void Executor::reset(std::uint32_t sequence) {
    impl_->positions.at(sequence) = 0;
    if (impl_->drafting()) { impl_->anchored.at(sequence) = 0; }
    impl_->release(sequence);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->positions.at(sequence);
}

void Executor::warm_up() { impl_->warm_up(); check_experts(); }

void Executor::check_experts() const { if (impl_->delivery) { impl_->delivery->check(); } }
ExpertDeliveryStats Executor::expert_stats() const {
    if (!impl_->delivery) { return {}; }
    std::lock_guard lock(impl_->expert_stats_mutex);
    return impl_->delivery->stats() - impl_->expert_startup_stats;
}
ExpertRouteCounts Executor::expert_totals() const {
    return impl_->delivery ? impl_->delivery->totals() : ExpertRouteCounts{};
}
std::string Executor::expert_cpu_description() const {
    return impl_->delivery ? impl_->delivery->cpu_description() : std::string();
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, std::span<const ChunkMedia> media) {
    impl_->forward(sequence, tokens, logit_rows, media);
}

void Executor::forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->forward(sequence, tokens, static_cast<std::uint32_t>(lengths.size()), {}, lengths);
}

bool Executor::vision() const noexcept { return impl_->vision_media; }

double Executor::last_media_seconds() const {
    if (!impl_->media_timed) { return 0.0; }
    CUDA_CHECK(cudaEventSynchronize(impl_->media_end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->media_begin, impl_->media_end));
    return double(ms) / 1000.0;
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("gemma4 logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16,
                  {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

bool Executor::can_draft(std::uint32_t sequence) const {
    return impl_->drafting() && impl_->anchored.at(sequence) != 0 &&
           impl_->positions.at(sequence) > 0;
}

void Executor::draft(std::span<const std::uint32_t> sequences,
                     std::span<const std::int32_t> anchors, std::span<std::int32_t> drafts,
                     std::uint32_t steps) {
    impl_->draft(sequences, anchors, drafts, steps);
}

void Executor::verify(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->verify(sequences, tokens);
}

void Executor::commit(std::span<const std::uint32_t> sequences,
                      std::span<const std::uint32_t> kept) {
    impl_->commit(sequences, kept);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

} // namespace ninfer::models::gemma4
