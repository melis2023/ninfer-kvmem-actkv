#include "models/flint/executor.h"

#include "core/decode_graph.h"
#include "core/growable_region.h"
#include "core/vmm.h"
#include "core/weight_view.h"
#include "models/flint/cpu_experts.h"
#include "models/flint/kernels.h"
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
#include <utility>
#include <vector>

namespace ninfer::models::flint {
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
        throw std::invalid_argument("flint: projection parts are not one stored parent");
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
// KV heads), `position_bytes` a position. Owned whole, or, beside an elastic expert cache, a
// growable region whose rows are backed only as far as their owner has reached (Impl::back), so
// the expert cache holds the rest of the device until then.
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
    bool full = false, rope = false, moe = false;
    int qkv_rows = 0;
    std::vector<Weight> qkv; // one [Q; K; V] parent, or the parts of a mixed-format block
    // The dense MLP's, or a MoE block's shared expert's, [gate; up] parent and down rows.
    Weight output, gate_up, down;
    int width = 0;
    const float *input_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr,
                *sinks = nullptr;
    // MoE.
    int delivery_layer = -1;
    const float* router = nullptr;
    gguf::GgmlType gate_up_type{}, down_type{};
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // pointers: [E] gate, [E] up, [E] gate_up, [E] down
    // The F16 K and V caches: [sequences][kv heads][slots][D].
    KvArea k, v;
    int slots = 0, ring = 0;
    std::int64_t seq_stride = 0, head_stride = 0; // elements of a cache (and of its staging)
    // A sliding block's checkpoint of every sequence's rings (snapshot()), allocated with the
    // cache: K, then V.
    std::vector<DeviceBuffer> checkpoint;
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
    int hidden = 0, query_heads = 0, kv_heads = 0, head_dim = 0, vocab = 0, ring = 0,
        global_slots = 0, window = 0;
    // A prompt chunk's sliding-window attention needs the window before it and the chunk itself:
    // a ring of prefill_ring slots that every sequence's chunks share, one block at a time
    // (staging_k/v), while each sequence keeps only its last `ring` (the window) positions.
    int prefill_ring = 0;
    DeviceBuffer staging_k, staging_v;
    KvArea mask_full; // a prompt chunk's full-attention mask, [rows][full_slots]
    // Beside an elastic expert cache, the full blocks' caches (a row per sequence and KV head) and
    // the prompt chunk's mask grow as far as the sequences reach: the expert cache gives memory
    // back when they need it and takes it again when a sequence is reset.
    bool growing = false;
    std::vector<KvArea*> sequence_areas, chunk_areas;
    std::atomic<std::uint64_t> kv_backed{0};
    std::uint64_t kv_fixed = 0; // the caches' bytes that do not grow
    float eps = 1e-6f;
    std::vector<LayerPlan> layers;
    // The CPU's share of missing experts; it outlives the delivery that calls it.
    std::unique_ptr<CpuExperts> cpu_experts;
    std::unique_ptr<ExpertDelivery> delivery;
    mutable std::mutex expert_stats_mutex;
    ExpertDeliveryStats expert_startup_stats;
    Weight embedding, head;
    const float* final_norm = nullptr;
    DeviceBuffer inv_freq;
    std::vector<std::uint32_t> positions;
    // Device staging: tokens, positions, sequences for up to max(prefill_chunk, sequences) rows. A
    // branch pass's rows: d_slots, the position whose cache slot each takes; d_starts, its branch's
    // first row.
    DeviceBuffer d_tokens, d_positions, d_sequences, d_slots, d_starts;
    DeviceBuffer route_done; // kernels::route_select's block count
    std::unique_ptr<PinnedHostBuffer> h_staging;
    // Recorded after the staging copies: the next stage() rewrites the pinned buffer only once
    // the previous copies have read it (a prompt chunk that keeps no logits is not synchronized).
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    std::uint64_t kv_bytes      = 0;
    int decode_splits_full = 1, decode_splits_sliding = 1;
    // Decode graphs by row count, verification graphs by 16 + rows.
    std::map<std::uint32_t, GraphEntry> graphs;
    std::uint32_t verify_width = 0; // rows per sequence of the last verify()

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(o), config(m.config()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("flint executor: invalid options");
        }
        if (device.size() != 1) { throw std::invalid_argument("Flint runs on one device"); }
        if (options.draft_tokens > std::uint32_t(gguf::kMaxVectorColumns) - 1) {
            throw std::invalid_argument("flint executor: at most 7 drafts a round");
        }
        stream = device.stream;
        // Beside an automatic cache of host or disk experts, the full-attention caches grow with
        // their sequences and the cache is elastic (ExpertDeliveryOptions::elastic).
        const int device_id = device.rank(0).device;
        growing = !model.expert_banks().empty() && !options.expert_cache_bytes &&
                  vmm::granularity(device_id) != 0;
        hidden      = int(config.hidden_size);
        query_heads = int(config.num_attention_heads);
        kv_heads    = int(config.num_key_value_heads);
        head_dim    = int(config.head_dim);
        vocab       = int(config.vocab_size);
        window      = int(config.sliding_window);
        eps         = config.rms_norm_eps;
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        // A verification writes a round's drafts past the window before keeping only some: the
        // ring holds them without overwriting a key the window still needs.
        ring         = int(round_up(config.sliding_window + options.draft_tokens, 256));
        prefill_ring = int(round_up(config.sliding_window + options.prefill_chunk, 256));
        global_slots = int(round_up(options.max_context + 1, 256));
        positions.assign(options.sequences, 0);
        const auto& w = model.weights();
        embedding     = native_weight(model.weight(w.token_embedding).view);
        head          = matrix(model, {w.output_head});
        final_norm    = fp32(model, w.final_norm, hidden);
        if (!is_gguf(embedding.qtype) || !is_gguf(head.qtype)) {
            throw std::invalid_argument("Flint runs GGUF block matrices");
        }
        {
            // The rotation of the sliding-window blocks, as the reference computes it in FP32.
            std::vector<float> freq(head_dim / 2);
            for (std::size_t i = 0; i < freq.size(); ++i) {
                freq[i] = float(1.0 / std::pow(double(config.rope_theta),
                                               double(2 * i) / double(head_dim)));
            }
            inv_freq = DeviceBuffer(freq.size() * 4);
            inv_freq.copy_from_host(freq.data(), freq.size() * 4);
        }
        const int sequences = int(options.sequences);
        const std::size_t row_f16 = std::size_t(head_dim) * sizeof(__half);
        {
            // The caches are sized up front for every sequence's whole context: refuse with the
            // numbers rather than fail inside an allocation. Counted here: what grows with the
            // context or the sequences; the reserve covers the rest of the workspace, the CUDA
            // graphs (some captured while serving) and a margin.
            std::size_t needed = 0;
            for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
                const std::size_t kv = 2 * std::size_t(kv_heads) * row_f16 * std::size_t(sequences);
                needed += config.full_attention[i]
                              ? std::size_t(global_slots) * kv
                              : (options.checkpoints ? 2 : 1) * std::size_t(ring) * kv;
            }
            needed += std::size_t(prefill_ring) * 2 * kv_heads * row_f16 +
                      std::size_t(options.prefill_chunk) * global_slots * 2;
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = start_reserve();
            if (needed + reserve > free_bytes) {
                throw InsufficientMemory(
                    "Flint: the KV cache, checkpoints and prompt buffers of " +
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
            p.full = config.full_attention[i];
            p.rope = config.rope[i];
            p.moe  = lw.moe.has_value();
            p.qkv  = matrices(model, {lw.attention.query, lw.attention.key, lw.attention.value});
            for (const Weight& part : p.qkv) { p.qkv_rows += part.n; }
            p.output     = matrix(model, {lw.attention.output});
            p.input_norm = fp32(model, lw.input_norm, hidden);
            p.ffn_norm   = fp32(model, lw.ffn_norm, hidden);
            p.q_norm     = fp32(model, lw.attention.query_norm, head_dim);
            p.k_norm     = fp32(model, lw.attention.key_norm, head_dim);
            p.sinks      = fp32(model, lw.attention.sinks, query_heads);
            if (!p.moe) {
                p.gate_up = matrix(model, {*lw.mlp_gate, *lw.mlp_up});
                p.down    = matrix(model, {*lw.mlp_down});
                p.width   = int(config.intermediate_size);
            } else {
                const MoeWeights& mw = *lw.moe;
                const int e = int(config.num_experts), width = int(config.moe_intermediate_size);
                p.gate_up = matrix(model, {mw.shared_gate, mw.shared_up});
                p.down    = matrix(model, {mw.shared_down});
                p.width   = int(config.shared_intermediate_size);
                p.router  = fp32(model, mw.router, e * hidden);
                if (model.expert_residency() == ExpertResidency::Device) {
                    const Weight gate_up = matrix(model, {*mw.gate_up});
                    const Weight down    = matrix(model, {*mw.down});
                    if (!is_gguf(gate_up.qtype) || !is_gguf(down.qtype)) {
                        throw std::invalid_argument("Flint experts must be GGUF block matrices");
                    }
                    p.gate_up_type      = ops::detail::gguf_type(gate_up.qtype);
                    p.down_type         = ops::detail::gguf_type(down.qtype);
                    p.gate_up_row_bytes = row_bytes(gate_up);
                    p.down_row_bytes    = row_bytes(down);
                    std::vector<const void*> table(4 * std::size_t(e));
                    for (int x = 0; x < e; ++x) {
                        const auto* gu = static_cast<const std::byte*>(gate_up.qdata) +
                                         std::int64_t(x) * 2 * width * p.gate_up_row_bytes;
                        table[x]         = gu;
                        table[e + x]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
                        table[2 * e + x] = gu;
                        table[3 * e + x] = static_cast<const std::byte*>(down.qdata) +
                                           std::int64_t(x) * hidden * p.down_row_bytes;
                    }
                    p.tables = DeviceBuffer(table.size() * sizeof(void*));
                    p.tables.copy_from_host(table.data(), table.size() * sizeof(void*));
                } else {
                    p.delivery_layer    = int(i - config.dense_layers);
                    const auto& bank    = model.expert_banks()[std::size_t(p.delivery_layer)];
                    p.gate_up_type      = ops::detail::gguf_type(bank.gate_up_type);
                    p.down_type         = ops::detail::gguf_type(bank.down_type);
                    p.gate_up_row_bytes = std::int64_t(bank.gate_up_row_bytes);
                    p.down_row_bytes    = std::int64_t(bank.down_row_bytes);
                }
                if (!gguf::moe_matrix_supported(p.gate_up_type, 2 * width, hidden, true) ||
                    !gguf::moe_matrix_supported(p.down_type, hidden, width, true)) {
                    throw std::invalid_argument(
                        "Flint: no integer expert kernel for these expert formats");
                }
            }
            for (const Weight* m : {&p.output, &p.gate_up, &p.down}) {
                if (!is_gguf(m->qtype)) { throw std::invalid_argument("Flint runs GGUF block matrices"); }
            }
            for (const Weight& m : p.qkv) {
                if (!is_gguf(m.qtype)) { throw std::invalid_argument("Flint runs GGUF block matrices"); }
            }
            p.slots       = p.full ? global_slots : ring;
            p.ring        = p.full ? 0 : ring;
            p.head_stride = std::int64_t(p.slots) * head_dim;
            p.seq_stride  = p.head_stride * kv_heads;
            const bool grows = growing && p.full;
            for (KvArea* area : {&p.k, &p.v}) {
                *area = KvArea(std::size_t(sequences), kv_heads,
                               std::size_t(p.head_stride) * sizeof(__half), row_f16, grows,
                               device_id);
                kv_bytes += area->bytes;
            }
            if (!p.full && options.checkpoints) {
                // Sized up front with the cache, so a configuration that cannot hold them is refused
                // at startup instead of failing a request's first checkpoint.
                for (int c = 0; c < 2; ++c) {
                    p.checkpoint.emplace_back(std::size_t(p.seq_stride) * sizeof(__half) * sequences);
                    kv_bytes += p.checkpoint.back().bytes;
                }
            }
            layers.push_back(std::move(p));
        }
        {
            const std::size_t staging = std::size_t(prefill_ring) * kv_heads * row_f16;
            staging_k = DeviceBuffer(staging);
            staging_v = DeviceBuffer(staging);
            staging_k.fill(0);
            staging_v.fill(0);
            kv_bytes += 2 * staging;
            mask_full = KvArea(1, 1, std::size_t(options.prefill_chunk) * global_slots * 2,
                               std::size_t(options.prefill_chunk) * 2, growing, device_id);
        }
        const int sms = device.multiprocessor_count();
        // Upper bounds: a step uses as many as its keys fill (a tile of 16 each), so the sliding
        // window's 513 keys take 16 splits of two tiles and a long full-attention context fills a
        // wave of the kernel.
        decode_splits_full    = std::clamp(2 * sms / std::max(1, kv_heads), 1, 128);
        decode_splits_sliding = 16;
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, sequences);
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        d_slots     = DeviceBuffer(rows * 4);
        d_starts    = DeviceBuffer(rows * 4);
        route_done             = DeviceBuffer(4);
        route_done.fill();
        h_staging   = std::make_unique<PinnedHostBuffer>(kStagingRegions * rows * 4);
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        for (LayerPlan& p : layers) {
            if (!p.full) { continue; }
            sequence_areas.push_back(&p.k);
            sequence_areas.push_back(&p.v);
        }
        chunk_areas.push_back(&mask_full);
        kv_fixed = kv_bytes;
        if (growing) {
            for (KvArea* area : sequence_areas) { kv_fixed -= area->bytes; }
        }
        kv_backed       = kv_fixed;
        workspace_bytes = plan_workspace();
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (options.prefill_chunk < kFullChunk &&
                workspace_bytes + kGraphReserve + unbacked() > free_bytes) {
                throw InsufficientMemory(
                    "Flint: the workspace of " + std::to_string(workspace_bytes >> 20) +
                    " MiB and " + std::to_string(kGraphReserve >> 20) +
                    " MiB for the CUDA graphs need more than the " +
                    std::to_string(free_bytes >> 20) +
                    " MiB free after the caches; lower --max-concurrency or --max-context");
            }
        }
        workspace = std::make_unique<WorkspaceArena>(workspace_bytes);
        if (!model.expert_banks().empty()) { plan_delivery(); }
    }

    void plan_delivery() {
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
        const auto bank     = slot * config.num_experts;
        const auto metadata = std::uint64_t(options.prefill_chunk) * config.top_k_experts * 8 +
            std::uint64_t(config.moe_layers()) * config.num_experts * 4 * sizeof(void*) +
            config.num_experts * 4;
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto cache = options.expert_cache_bytes.value_or(0);
        // The caches that grow later need their memory too: the elastic cache gives it back.
        if (bank + metadata + cache + unbacked() + kGraphReserve > free) {
            throw InsufficientMemory("Flint: expert delivery needs " +
                std::to_string((bank + metadata + cache) >> 20) + " MiB beside " +
                std::to_string(unbacked() >> 20) +
                " MiB for the growing caches and the CUDA Graph reserve after workspace allocation");
        }
        ExpertDeliveryOptions transport;
        transport.pair_capacity = options.prefill_chunk * config.top_k_experts;
        transport.cache_bytes   = options.expert_cache_bytes;
        // A growing cache's next positions come from the headroom before the expert cache has to
        // give memory back.
        transport.keep_free_bytes = kGraphReserve + (growing ? kKvHeadroom : 0);
        transport.elastic         = growing;
        const float dma_share     = options.expert_dma_share.value_or(0.0F);
        if (!(dma_share >= 0 && dma_share <= 1) || options.expert_cpu_threads > 256) {
            throw std::invalid_argument("Flint: the DMA share must be 0..1 and CPU threads 0..256");
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
        if (computes && dma_share < 1) {
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

    // The startup check's room for the workspace and the CUDA graphs: 1.5 GiB at the full
    // 2048-token chunk; a narrower chunk keeps the least its workspace takes and then checks the
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
        const std::size_t g     = a.region.granule();
        const std::uint32_t was = a.backed[owner];
        a.backed[owner]         = positions;
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
    // rows (and with `chunk`, the prompt chunk's mask) are backed up to the 256-slot tile that
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
                throw InsufficientMemory("Flint: no device memory left to grow the KV cache to " +
                                         std::to_string(positions) + " positions");
            }
        }
        count_backed();
    }

    // A reset sequence's rows go back to the device, the prompt mask to what the longest remaining
    // sequence reaches, and the expert cache takes what is free beyond its headroom.
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
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    // ---- Workspace -----------------------------------------------------------------------------

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    // Rows of a decode or verification call.
    [[nodiscard]] int decode_rows() const {
        return std::min<int>(gguf::kMaxVectorColumns,
                             std::max<int>(int(options.sequences), int(options.draft_tokens) + 1));
    }

    std::size_t plan_workspace() const {
        const std::size_t t = options.prefill_chunk;
        const std::size_t h = hidden;
        const std::size_t q_floats = std::size_t(query_heads) * head_dim;
        std::size_t qkv_rows = 0, width = 0, gguf_scratch = 0;
        for (const LayerPlan& p : layers) {
            qkv_rows = std::max<std::size_t>(qkv_rows, p.qkv_rows);
            width    = std::max<std::size_t>(width, p.width);
            std::vector<const Weight*> products{&p.output, &p.down};
            for (const Weight& part : p.qkv) { products.push_back(&part); }
            for (const Weight* m : products) {
                const ops::detail::GgufShape shape{m->qtype, m->n, m->k};
                gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                          {&shape, 1}, 1, int(t)));
            }
            const ops::detail::GgufShape gu{p.gate_up.qtype, p.gate_up.n, p.gate_up.k};
            gguf_scratch = std::max(gguf_scratch,
                                    ops::detail::gguf_swiglu_workspace_bytes(gu, nullptr, 1, int(t)));
        }
        const ops::detail::GgufShape hs{head.qtype, std::min<int>(head.n, vocab), head.k};
        gguf_scratch = std::max(gguf_scratch,
                                ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        std::size_t bytes = 0;
        const auto add    = [&](std::size_t n) { bytes += round_up(n, kAlign); };
        add(t * h * 4);              // residual
        add(t * h * 2);              // normalized input
        add(t * qkv_rows * 4);       // qkv FP32
        add(t * q_floats * 4);       // q
        add(t * q_floats * 4);       // attention out FP32
        add(t * q_floats * 2);       // attention out BF16
        add(t * h * 2);              // FFN input
        add(t * h * 4);              // router input
        add(t * width * 2);          // middle
        add(t * std::size_t(prefill_ring) * 2); // sliding-window mask
        add(gguf::flash_attention_workspace_bytes(head_dim, query_heads, kv_heads, int(t)));
        add(std::size_t(kHeadRows) * vocab * 4); // head FP32 plane
        add(kHeadRows * h * 2);
        std::size_t moe = 0;
        {
            const std::size_t pairs = t * config.top_k_experts;
            const std::size_t w     = config.moe_intermediate_size;
            moe += round_up(pairs * 4, kAlign) * 2;                   // ids, weights
            moe += round_up(t * config.num_experts * 4, kAlign);      // logits
            moe += round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign);
            moe += round_up(pairs * 2 * w * 4, kAlign);
            moe += round_up(pairs * w * 2, kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(w), int(pairs)), kAlign);
            moe += round_up(pairs * h * 4, kAlign);
        }
        const std::size_t decode = kernels::decode_attention_workspace_floats(
            decode_rows(), query_heads, head_dim, std::max(decode_splits_full, decode_splits_sliding)) * 4;
        return bytes + moe + round_up(decode, kAlign) + gguf_scratch + (16u << 20);
    }

    // ---- Shared pieces ---------------------------------------------------------------------------

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

    // out[:, c] (+)= W x_c for the q8_1 rows x: FP32 columns `column_stride` apart, from row `row`.
    void vector_f32(const Q8Act& x, const Weight& w, int t, float* out, int column_stride,
                    int row = 0, bool accumulate = false) {
        gguf::VectorOutput o;
        o.f32           = out + row;
        o.column_stride = column_stride;
        o.epilogue      = accumulate ? gguf::Epilogue::Accumulate : gguf::Epilogue::Store;
        gguf::vector_product(ops::detail::gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, w.k,
                             x.base, t, o, stream);
    }

    // out[:, c] (+)= W x_c for BF16 rows x [K, T]: FP32 columns `column_stride` apart, from `row`.
    void project_f32(const Tensor& x, const Weight& w, float* out, int column_stride, int row = 0,
                     bool accumulate = false) {
        ops::detail::GgufProduct product;
        product.weight   = &w;
        product.f32      = out;
        product.f32_rows = column_stride;
        product.row      = row;
        product.epilogue = accumulate ? ops::detail::GgufEpilogue::Accumulate
                                      : ops::detail::GgufEpilogue::Store;
        ops::detail::gguf_project(x, {&product, 1}, *workspace, stream);
    }

    // h[:, c] += W x_c for BF16 rows x [K, T]. The vector and the matrix kernels' q8_1 rows keep
    // each 32 values' sum in F16 for some types (Q5_K among Flint's): there the columns whose sums
    // could pass F16's range are scaled by a power of two, with their h columns, and h's columns
    // restored after the product (kernels::scale_columns).
    void project_residual(const Tensor& x, const Weight& w, float* h, int t) {
        const auto type = ops::detail::gguf_type(w.qtype);
        if (!(t <= gguf::kMaxVectorColumns ? gguf::vector_activation_f16_sums(type)
                                           : gguf::matrix_activation_f16_sums(type))) {
            project_f32(x, w, h, hidden, 0, true);
            return;
        }
        auto scope  = workspace->scope();
        auto* scale = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * 4).data);
        kernels::scale_columns(static_cast<__nv_bfloat16*>(x.data), w.k, t, h, hidden, scale,
                               stream);
        project_f32(x, w, h, hidden, 0, true);
        kernels::unscale_columns(h, hidden, t, scale, stream);
    }

    // A block's Q, K and V rows into FP32 [T][qkv_rows], part by part when they are not one parent.
    void project_qkv(const Tensor* x, const Q8Act* xq, const LayerPlan& p, int t, float* out) {
        if (xq != nullptr) {
            int row = 0;
            for (const Weight& part : p.qkv) {
                vector_f32(*xq, part, t, out, p.qkv_rows, row);
                row += part.n;
            }
            return;
        }
        std::vector<ops::detail::GgufProduct> parts(p.qkv.size());
        for (std::size_t i = 0, row = 0; i < parts.size(); row += p.qkv[i].n, ++i) {
            parts[i].weight   = &p.qkv[i];
            parts[i].f32      = out;
            parts[i].f32_rows = p.qkv_rows;
            parts[i].row      = std::int32_t(row);
        }
        ops::detail::gguf_project(*x, parts, *workspace, stream);
    }

    static constexpr std::size_t kStagingRegions = 5;

    // Tokens, positions, and when given sequences, a branch pass's slots and branch starts, to the
    // device.
    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c, std::span<const std::int32_t> g = {},
               std::span<const std::int32_t> k = {}) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, options.sequences);
        if (a.size() > rows || b.size() > rows || c.size() > rows || g.size() > rows ||
            k.size() > rows) {
            throw std::invalid_argument("flint executor: staging overflow");
        }
        std::copy(a.begin(), a.end(), host);
        std::copy(b.begin(), b.end(), host + rows);
        std::copy(c.begin(), c.end(), host + 2 * rows);
        std::copy(g.begin(), g.end(), host + 3 * rows);
        std::copy(k.begin(), k.end(), host + 4 * rows);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens.p, host, a.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_positions.p, host + rows, b.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        if (!c.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_sequences.p, host + 2 * rows, c.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!g.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_slots.p, host + 3 * rows, g.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!k.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_starts.p, host + 4 * rows, k.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        staged->record(stream);
    }

    // The embeddings of the staged tokens into FP32 h [T][H].
    void embed(int tokens, float* h) {
        gguf::dequantize_rows(ops::detail::gguf_type(embedding.qtype), embedding.qdata,
                              row_bytes(embedding), hidden,
                              static_cast<const std::int32_t*>(d_tokens.p), tokens, h, hidden,
                              stream);
    }

    // The FFN of block `p` after its attention has been added to the residual h: the dense MLP or
    // the routed and shared experts into h, and the normalized input of what follows (`next_norm`:
    // the next block's input norm, or the final norm) into `next` (BF16) and/or `next_q`.
    void block_tail(const LayerPlan& p, float* h, int t, const float* next_norm,
                    __nv_bfloat16* next, const Q8Act* next_q) {
        auto scope    = workspace->scope();
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor y      = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act yq;
        float* router_x = nullptr;
        if (q8) { yq = alloc_q8(hidden, t); }
        if (p.moe) {
            router_x = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        }
        kernels::rms_norm(h, p.ffn_norm, hidden, t, eps,
                          q8 ? nullptr : static_cast<__nv_bfloat16*>(y.data), stream, yq.rows,
                          router_x);
        // The dense MLP or the shared expert: SwiGLU over the [gate; up] parent, then the down rows
        // added to the residual.
        const auto mlp = [&] {
            auto mlp_scope = workspace->scope();
            Tensor middle  = workspace->alloc(DType::BF16, {p.width, t});
            if (q8) {
                gguf::vector_swiglu(ops::detail::gguf_type(p.gate_up.qtype), p.gate_up.qdata,
                                    row_bytes(p.gate_up), p.width, hidden, yq.base, t,
                                    static_cast<__nv_bfloat16*>(middle.data), p.width, stream);
            } else {
                ops::detail::gguf_swiglu(y, p.gate_up, nullptr, middle, *workspace, stream);
            }
            project_residual(middle, p.down, h, t);
        };
        kernels::FfnCombineArgs combine;
        combine.h         = h;
        combine.hidden    = hidden;
        combine.tokens    = t;
        combine.eps       = eps;
        combine.next_norm = next_norm;
        combine.next_out  = next;
        if (next_q != nullptr) { combine.next_q = next_q->rows; }
        if (p.moe) {
            experts(p, static_cast<const __nv_bfloat16*>(y.data), q8 ? &yq : nullptr, router_x, t,
                    combine, mlp);
        } else {
            mlp();
        }
        kernels::ffn_combine(combine, stream);
    }

    // The routed experts of block `p` (their output lands where `combine` reads it) and `shared`,
    // the shared expert, run while missing experts are copied in.
    template <typename Shared>
    void experts(const LayerPlan& p, const __nv_bfloat16* x, const Q8Act* xq, const float* router_x,
                 int t, kernels::FfnCombineArgs& combine, Shared&& shared) {
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* ids      = static_cast<std::int32_t*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* weights  = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* logits   = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * e * 4).data);
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        const bool vector   = t <= gguf::kMaxVectorColumns;
        gguf::MoeRouting routing;
        if (vector) {
            auto* base = static_cast<std::int32_t*>(routing_space);
            routing.bounds       = base;
            routing.sorted       = base + e + 1;
            routing.active       = base + e + 1 + pairs;
            routing.active_count = base + e + 1 + pairs + e;
            kernels::route_select(router_x, hidden, t, p.router, e, k, logits, ids, weights, base,
                                  base + e + 1, base + e + 1 + pairs, base + e + 1 + pairs + e,
                                  static_cast<std::uint32_t*>(route_done.p), stream);
        } else {
            kernels::route(router_x, hidden, t, p.router, e, k, logits, ids, weights, stream);
            routing = gguf::moe_sort_routes(ids, pairs, e, routing_space, stream);
        }
        // A decode or verification call may leave misses to the CPU, which reads the quantized
        // activation and the route weights.
        const bool cpu = cpu_experts && t <= CpuExperts::kMaxTokens && vector;
        const void* act = vector ? xq->base : nullptr;
        std::optional<ExpertDelivery::Call> call;
        if (delivery) {
            const std::array<ExpertDelivery::Input, 2> inputs{
                ExpertDelivery::Input{act, cpu ? cpu_experts->activation_bytes(t) : 0},
                ExpertDelivery::Input{weights, cpu ? cpu_experts->weight_bytes(t) : 0}};
            call = delivery->begin(std::size_t(p.delivery_layer), ids, std::uint32_t(pairs), stream,
                                   cpu ? std::span<const ExpertDelivery::Input>(inputs)
                                       : std::span<const ExpertDelivery::Input>());
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
        if (vector) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2  = workspace->alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            auto* fixed = static_cast<unsigned long long*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 8).data);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            // Down rows whose dot reads the activation's F16 sums take the middles' guard, which
            // may scale a pair's weight by a power of two.
            const bool guard = gguf::vector_activation_f16_sums(p.down_type);
            const kernels::Q8Rows middle_q{
                static_cast<std::int8_t*>(act2),
                reinterpret_cast<half2*>(static_cast<std::byte*>(act2) +
                                         gguf::vector_activation_scales_offset(width, pairs))};
            const auto compute = [&](const gguf::MoeRouting& part) {
                gguf::moe_vector_swiglu(p.gate_up_type, gate, up, part, max_active, k, act, t,
                                        middle, chunk, stream);
                if (guard) {
                    kernels::quantize_pairs(middle, pairs, width, weights, middle_q, stream);
                } else {
                    gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, stream);
                }
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, stream);
            };
            if (call) {
                // Both stages add to the same exact fixed-point sum, zeroed once above, each over
                // its experts of the call's sorted routes; the shared expert runs between them,
                // while the missing experts arrive. Quantization reads every pair: the cached stage
                // finds the other pairs cleared, the missing stage the cached pairs' middles.
                const auto stage = [&](const std::int32_t* set) {
                    gguf::MoeRouting part = routing;
                    part.active_count     = set;
                    part.active           = set + 1;
                    return part;
                };
                CUDA_CHECK(cudaMemsetAsync(middle, 0, std::size_t(pairs) * width * 2, stream));
                compute(stage(call->cached_experts));
                shared();
                delivery->await(stream);
                compute(stage(call->missing_experts));
                if (call->cpu != nullptr) {
                    kernels::add_fixed(fixed, call->products, call->cpu, std::int64_t(t) * hidden, stream);
                }
            } else {
                compute(routing);
                shared();
            }
            combine.moe_fixed = fixed;
            return;
        }
        shared();
        if (delivery) { delivery->await(stream); }
        void* rows_act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.gate_up_type, x, hidden, routing, pairs, k, rows_act,
                                             stream);
        auto* gu = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true, rows_act,
                                 gu, stream);
        // The route weights take the power-of-two factors of rows scaled for the down product.
        kernels::swiglu_rows(gu, pairs, width, middle,
                             gguf::matrix_activation_f16_sums(p.down_type) ? weights : nullptr,
                             stream);
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
            project_f32(x, head, plane, vocab);
            kernels::f32_to_bf16(plane, static_cast<__nv_bfloat16*>(logits_bf16.p) +
                                            std::int64_t(first + at) * vocab,
                                 std::int64_t(n) * vocab, stream);
        }
    }

    // The head of the last `rows` of t <= kMaxVectorColumns final-normed q8_1 rows into
    // logits_bf16 columns [0, rows).
    void head_logits_q8(const Q8Act& x, int t, int rows) {
        auto scope  = workspace->scope();
        auto* plane = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * vocab * 4).data);
        vector_f32(x, head, t, plane, vocab);
        kernels::f32_to_bf16(plane + std::int64_t(t - rows) * vocab,
                             static_cast<__nv_bfloat16*>(logits_bf16.p),
                             std::int64_t(rows) * vocab, stream);
    }

    // ---- Prompt chunks -----------------------------------------------------------------------------

    // With `branches` (forward_branches): the chunk is those branches' tokens one after another,
    // each continuing the sequence at its position, and `logit_rows` is their count. With `shared`
    // (forward_prompt_branches) the chunk's first `shared` tokens continue the sequence ahead of the
    // branches, which continue it from there; the sequence keeps those rows and stops there.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const std::uint32_t> branches = {},
                 std::uint32_t shared = 0) {
        const int t = int(tokens.size());
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows == 0 || logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows)) {
            throw std::invalid_argument("flint forward: invalid call");
        }
        const bool branching = !branches.empty();
        if (branching && (logit_rows != branches.size() || shared >= std::uint32_t(t) ||
                          std::accumulate(branches.begin(), branches.end(), std::uint64_t{0}) !=
                              std::uint64_t(t) - shared ||
                          std::find(branches.begin(), branches.end(), 0U) != branches.end())) {
            throw std::invalid_argument("flint forward_branches: invalid branches");
        }
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("flint forward: context capacity exceeded");
        }
        back(sequence, first + std::uint32_t(t), true);
        std::vector<std::int32_t> pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        // A branch pass: each branch row rotates at its branch's position, and takes the cache
        // slot of its place in the chunk (`slots`, the positions a plain chunk has); `starts` its
        // branch's first row, for the masks. The shared rows keep their own positions.
        std::vector<std::int32_t> slots, starts;
        if (branching) {
            slots = pos;
            starts.resize(t);
            for (std::size_t b = 0, row = shared; b < branches.size(); row += branches[b++]) {
                for (std::uint32_t j = 0; j < branches[b]; ++j) {
                    pos[row + j]    = std::int32_t(first + shared + j);
                    starts[row + j] = std::int32_t(row);
                }
            }
        }
        stage(tokens, pos, {}, slots, starts);
        const auto* d_branch_starts =
            branching ? static_cast<const std::int32_t*>(d_starts.p) : nullptr;
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * hidden * 4).data);
        embed(t, h);
        // Masks: the sliding ring and the full-attention prefix.
        const int full_slots = int(round_up(first + t, 256));
        auto* mask_ring = static_cast<__half*>(
            workspace->alloc_bytes(std::size_t(t) * prefill_ring * 2).data);
        auto* mask_full = static_cast<__half*>(this->mask_full.p);
        kernels::build_mask(mask_ring, t, prefill_ring, int(first), prefill_ring, window, stream,
                            d_branch_starts, int(shared));
        kernels::build_mask(mask_full, t, full_slots, int(first), 0, 0, stream, d_branch_starts,
                            int(shared));
        // Each block's normalized input rows; after the last block, the final-normed rows (q8_1
        // for a vector-width chunk).
        const bool q8 = t <= gguf::kMaxVectorColumns;
        Tensor x      = workspace->alloc(DType::BF16, {hidden, t});
        Q8Act xq;
        if (q8) { xq = alloc_q8(hidden, t); }
        kernels::rms_norm(h, layers.front().input_norm, hidden, t, eps,
                          q8 ? nullptr : static_cast<__nv_bfloat16*>(x.data), stream, xq.rows);
        const int q_floats = query_heads * head_dim;
        const std::int64_t stage_stride = std::int64_t(prefill_ring) * head_dim;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            auto* qkv = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(t) * p.qkv_rows * 4).data);
            project_qkv(q8 ? nullptr : &x, q8 ? &xq : nullptr, p, t, qkv);
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = qkv;
            qa.tokens      = t;
            qa.query_heads = query_heads;
            qa.kv_heads    = kv_heads;
            qa.head_dim    = head_dim;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = p.rope ? static_cast<const float*>(inv_freq.p) : nullptr;
            qa.q_out       = q;
            qa.slots       = branching ? static_cast<const std::int32_t*>(d_slots.p) : nullptr;
            // A sliding block attends over the staging ring: the window before the chunk first
            // comes over from the sequence's ring, the chunk's keys land there, and the window that
            // ends the chunk goes back. A full block's keys and values land in the sequence's cache.
            const std::int64_t seq_offset = std::int64_t(sequence) * p.seq_stride;
            kernels::RingCopyArgs keep;
            keep.kv_heads = kv_heads;
            keep.head_dim = head_dim;
            if (!p.full) {
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
                qa.k_cache     = static_cast<__half*>(staging_k.p);
                qa.v_cache     = static_cast<__half*>(staging_v.p);
                qa.head_stride = stage_stride;
                qa.ring        = prefill_ring;
            } else {
                qa.k_cache     = static_cast<__half*>(p.k.p);
                qa.v_cache     = static_cast<__half*>(p.v.p);
                qa.seq_stride  = p.seq_stride;
                qa.head_stride = p.head_stride;
                qa.sequence    = int(sequence);
            }
            kernels::qkv_post(qa, stream);
            auto* attn = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
            const std::size_t fa_bytes =
                gguf::flash_attention_workspace_bytes(head_dim, query_heads, kv_heads, t);
            void* fa_space = workspace->alloc_bytes(fa_bytes).data;
            gguf::FlashAttentionArgs fa;
            fa.head_dim       = head_dim;
            fa.query_heads    = query_heads;
            fa.kv_heads       = kv_heads;
            fa.tokens         = t;
            fa.q              = q;
            fa.q_token_stride = q_floats;
            fa.k = p.full ? static_cast<const __half*>(p.k.p) + seq_offset
                          : static_cast<const __half*>(staging_k.p);
            fa.v = p.full ? static_cast<const __half*>(p.v.p) + seq_offset
                          : static_cast<const __half*>(staging_v.p);
            fa.row_bytes       = std::int64_t(head_dim) * 2;
            fa.head_bytes      = (p.full ? p.head_stride : stage_stride) * 2;
            fa.slots           = p.full ? full_slots : prefill_ring;
            fa.mask            = p.full ? mask_full : mask_ring;
            fa.mask_row_halves = fa.slots;
            fa.out             = attn;
            fa.scale           = 1.0f / std::sqrt(float(head_dim));
            fa.sinks           = p.sinks;
            gguf::flash_attention(fa, fa_space, fa_bytes, stream);
            // The window that ends the sequence's rows of the chunk goes back: all of them, or a
            // branch pass's shared rows (the branches' rows lie past them, later in the ring).
            const int kept = int(first) + (branching ? int(shared) : t);
            if (!p.full && kept > int(first)) {
                kernels::RingCopyArgs back = keep;
                std::swap(back.src_head_stride, back.dst_head_stride);
                std::swap(back.src_ring, back.dst_ring);
                back.src_k = static_cast<const __half*>(staging_k.p);
                back.src_v = static_cast<const __half*>(staging_v.p);
                back.dst_k = static_cast<__half*>(p.k.p) + seq_offset;
                back.dst_v = static_cast<__half*>(p.v.p) + seq_offset;
                back.begin = std::max(0, kept - window);
                back.end   = kept;
                kernels::ring_copy(back, stream);
            }
            Tensor attn_bf16 = workspace->alloc(DType::BF16, {q_floats, t});
            kernels::f32_to_bf16(attn, static_cast<__nv_bfloat16*>(attn_bf16.data),
                                 std::int64_t(t) * q_floats, stream);
            project_residual(attn_bf16, p.output, h, t);
            const bool last = li + 1 == layers.size();
            // The final rows in BF16 too when a branch pass gathers its branches' last ones.
            block_tail(p, h, t, last ? final_norm : layers[li + 1].input_norm,
                       q8 && !(last && branching) ? nullptr : static_cast<__nv_bfloat16*>(x.data),
                       q8 ? &xq : nullptr);
        }
        if (branching) {
            // Each branch's last row to the front, in branch order; a branch's last row lies at or
            // past its index, so no copy overwrites a later one's source. The sequence ends after
            // its shared rows, its window there.
            auto* rows      = static_cast<__nv_bfloat16*>(x.data);
            std::size_t row = shared;
            for (std::size_t b = 0; b < branches.size(); ++b) {
                row += branches[b];
                if (row - 1 == b) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(
                    rows + std::int64_t(b) * hidden, rows + std::int64_t(row - 1) * hidden,
                    std::size_t(hidden) * 2, cudaMemcpyDeviceToDevice, stream));
            }
            head_logits(rows, int(logit_rows), 0);
            positions[sequence] = first + shared;
            return;
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

    // One token per row; tokens, positions and sequences already staged on the device.
    void decode_body(int b) {
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        embed(b, h);
        // Each block's normalized input rows in q8_1 form; after the last block, the final-normed
        // rows.
        const Q8Act xq = alloc_q8(hidden, b);
        kernels::rms_norm(h, layers.front().input_norm, hidden, b, eps, nullptr, stream, xq.rows);
        const int q_floats = query_heads * head_dim;
        for (std::size_t li = 0; li < layers.size(); ++li) {
            const LayerPlan& p = layers[li];
            auto layer_scope   = workspace->scope();
            auto* qkv = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(b) * p.qkv_rows * 4).data);
            project_qkv(nullptr, &xq, p, b, qkv);
            auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * q_floats * 4).data);
            kernels::QkvArgs qa;
            qa.qkv         = qkv;
            qa.tokens      = b;
            qa.query_heads = query_heads;
            qa.kv_heads    = kv_heads;
            qa.head_dim    = head_dim;
            qa.q_norm      = p.q_norm;
            qa.k_norm      = p.k_norm;
            qa.eps         = eps;
            qa.positions   = static_cast<const std::int32_t*>(d_positions.p);
            qa.inv_freq    = p.rope ? static_cast<const float*>(inv_freq.p) : nullptr;
            qa.q_out       = q;
            qa.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            qa.k_cache     = static_cast<__half*>(p.k.p);
            qa.v_cache     = static_cast<__half*>(p.v.p);
            qa.seq_stride  = p.seq_stride;
            qa.head_stride = p.head_stride;
            qa.ring        = p.ring;
            kernels::qkv_post(qa, stream);
            const int splits = p.full ? decode_splits_full : decode_splits_sliding;
            auto* partial    = static_cast<float*>(
                workspace->alloc_bytes(kernels::decode_attention_workspace_floats(
                                           b, query_heads, head_dim, splits) * 4).data);
            const Q8Act attn = alloc_q8(q_floats, b);
            kernels::DecodeAttentionArgs da;
            da.q           = q;
            da.k_cache     = static_cast<const __half*>(p.k.p);
            da.v_cache     = static_cast<const __half*>(p.v.p);
            da.sinks       = p.sinks;
            da.seq_stride  = p.seq_stride;
            da.head_stride = p.head_stride;
            da.sequences   = static_cast<const std::int32_t*>(d_sequences.p);
            da.positions   = static_cast<const std::int32_t*>(d_positions.p);
            da.batch       = b;
            da.query_heads = query_heads;
            da.kv_heads    = kv_heads;
            da.head_dim    = head_dim;
            da.ring        = p.ring;
            da.window      = window;
            da.splits      = splits;
            da.scale       = 1.0f / std::sqrt(float(head_dim));
            da.partial     = partial;
            da.out_q       = attn.rows;
            kernels::decode_attention(da, stream);
            vector_f32(attn, p.output, b, h, hidden, 0, true);
            const bool last = li + 1 == layers.size();
            block_tail(p, h, b, last ? final_norm : layers[li + 1].input_norm, nullptr, &xq);
        }
        head_logits_q8(xq, b, b);
    }

    // Runs `body` from the graph of `key`: eagerly the first time, so every kernel is loaded
    // before the capture, then captured and replayed.
    template <typename Body>
    void run_graph(std::uint32_t key, Body&& body) {
        if (!options.cuda_graphs) {
            body();
            return;
        }
        GraphEntry& entry = graphs[key];
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
            throw std::invalid_argument("flint decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences) {
                throw std::invalid_argument("flint decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("flint decode: context capacity exceeded");
            }
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        for (int i = 0; i < b; ++i) { back(sequences[i], positions[sequences[i]] + 1, false); }
        stage(tokens, pos, seq);
        run_graph(std::uint32_t(b), [&] { decode_body(b); });
        for (int i = 0; i < b; ++i) { positions[sequences[i]] += 1; }
    }

    // ---- Speculative rounds ------------------------------------------------------------------------

    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int n = int(sequences.size());
        if (options.draft_tokens == 0 || n <= 0 || tokens.size() % sequences.size() != 0) {
            throw std::invalid_argument("flint verify: invalid call");
        }
        const int w = int(tokens.size()) / n, rows = n * w;
        if (w < 1 || std::uint32_t(w) > options.draft_tokens + 1 || rows > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("flint verify: invalid width");
        }
        std::vector<std::int32_t> pos(rows), seq(rows);
        for (int j = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + j, s) != 0) {
                throw std::invalid_argument("flint verify: invalid sequence");
            }
            if (positions[s] + std::uint32_t(w) > options.max_context) {
                throw std::invalid_argument("flint verify: context capacity exceeded");
            }
            for (int i = 0; i < w; ++i) {
                pos[j * w + i] = std::int32_t(positions[s]) + i;
                seq[j * w + i] = std::int32_t(s);
            }
        }
        for (const std::uint32_t s : sequences) { back(s, positions[s] + std::uint32_t(w), false); }
        stage(tokens, pos, seq);
        run_graph(16 + std::uint32_t(rows), [&] { decode_body(rows); });
        verify_width = std::uint32_t(w);
    }

    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept) {
        if (kept.size() != sequences.size() || verify_width == 0) {
            throw std::invalid_argument("flint commit: invalid call");
        }
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || kept[j] < 1 || kept[j] > verify_width) {
                throw std::invalid_argument("flint commit: invalid row");
            }
            positions[s] += kept[j];
        }
        verify_width = 0;
    }

    void warm_up() {
        // The prompt route once (kernels load at first launch), then each decode batch's graph and
        // each verification width's.
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        forward(0, prompt, 1);
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
        if (options.draft_tokens > 0) {
            const std::uint32_t s = 0;
            std::vector<std::int32_t> row(options.draft_tokens + 1, 2);
            for (int pass = 0; pass < 2; ++pass) {
                verify({&s, 1}, row);
                const std::uint32_t one = 1;
                commit({&s, 1}, {&one, 1});
                positions[0] = 0;
            }
        }
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

    // A sliding block's ring buffers of one sequence, (start, bytes): K and V.
    std::array<std::pair<std::uint8_t*, std::size_t>, 2> ring_buffers(const LayerPlan& p,
                                                                      std::uint32_t sequence) const {
        const std::size_t bytes = std::size_t(p.seq_stride) * sizeof(__half);
        return {{{static_cast<std::uint8_t*>(p.k.p) + sequence * bytes, bytes},
                 {static_cast<std::uint8_t*>(p.v.p) + sequence * bytes, bytes}}};
    }

    void copy_rings(std::uint32_t sequence, bool save) {
        if (!options.checkpoints) { throw std::logic_error("flint: an executor without checkpoints"); }
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

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = 0;
        for (const LayerPlan& p : layers) {
            if (p.full) {
                bytes += 2 * std::uint64_t(kv_heads) * position * head_dim * sizeof(__half);
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
            throw std::invalid_argument("flint image: pages do not hold the image");
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
        const std::size_t head_bytes = std::size_t(global_slots) * head_dim * sizeof(__half);
        for (const LayerPlan& p : layers) {
            if (p.full) {
                for (const KvArea* area : {&p.k, &p.v}) {
                    auto* base = static_cast<std::uint8_t*>(area->p) + sequence * head_bytes * kv_heads;
                    for (int h = 0; h < kv_heads; ++h) {
                        copy(base + h * head_bytes, std::size_t(position) * head_dim * sizeof(__half));
                    }
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
            throw std::invalid_argument("flint load_image: invalid image");
        }
        back(sequence, image.position, false);
        copy_image(sequence, image.position, pages, page_bytes, false, false);
        positions.at(sequence) = image.position;
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        copy_rings(sequence, true);
        out.position = positions[sequence];
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        copy_rings(sequence, false);
        positions[sequence] = from.position;
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    return ExecutorMemory{.kv_bytes        = impl_->kv_bytes,
                          .kv_backed_bytes = impl_->kv_backed.load(std::memory_order_relaxed),
                          .kv_grows        = impl_->growing,
                          .workspace_bytes = impl_->workspace_bytes + impl_->mask_full.bytes,
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
    impl_->release(sequence);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->positions.at(sequence);
}

void Executor::warm_up() {
    impl_->warm_up();
    check_experts();
}

void Executor::check_experts() const {
    if (impl_->delivery) { impl_->delivery->check(); }
}
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
                       std::uint32_t logit_rows) {
    impl_->forward(sequence, tokens, logit_rows);
}

void Executor::forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->forward(sequence, tokens, static_cast<std::uint32_t>(lengths.size()), lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence,
                                       std::span<const std::int32_t> prompt,
                                       std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    std::vector<std::int32_t> chunk(prompt.begin(), prompt.end());
    chunk.insert(chunk.end(), tokens.begin(), tokens.end());
    impl_->forward(sequence, chunk, static_cast<std::uint32_t>(lengths.size()), lengths,
                   static_cast<std::uint32_t>(prompt.size()));
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("flint logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16,
                  {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

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

} // namespace ninfer::models::flint
