#include "models/qwen4_exp/hybrid_experts.h"

#include "core/arena.h"
#include "core/cuda_host_handshake.h"
#include "models/qwen4_exp/read_pool.h"
#include "ninfer/ops/residual_add.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace ninfer::models::qwen4_exp {
namespace {
constexpr std::size_t kHidden = 2560, kTop = 10, kCpuTokens = 16;
constexpr std::uint64_t kMargin = 1536ULL << 20;
constexpr std::size_t aligned(std::size_t bytes) { return (bytes + 255) & ~std::size_t(255); }

// Native experts' CPU products differ from the GPU's, so only an explicit share moves misses to
// the CPU; unset copies every one.
HybridExpertOptions validated(HybridExpertOptions options) {
    const float share = options.dma_share.value_or(1.0F);
    if (!std::isfinite(share) || share < 0 || share > 1 || options.cpu_threads > 256) {
        throw std::invalid_argument("hybrid experts: DMA share must be 0..1 and CPU threads 0..256");
    }
    options.dma_share = share;
    return options;
}

struct ProjectionBytes {
    std::size_t low, high, scales;
    [[nodiscard]] std::size_t size() const { return aligned(low) + aligned(high) + aligned(scales); }
};

ProjectionBytes planes(const Weight& weight) {
    const auto elements = std::size_t(weight.n) * weight.k;
    switch (weight.qtype) {
    case QType::BF16: return {elements * 2, 0, 0};
    case QType::Q2_G64_FP16: return {elements / 4, 0, elements / 64 * 2};
    case QType::Q4_G64_FP16: return {elements / 2, 0, elements / 64 * 2};
    case QType::Q5_G64_FP16: return {elements / 2, elements / 8, elements / 64 * 2};
    case QType::Q6_G64_FP16: return {elements / 2, elements / 4, elements / 64 * 2};
    case QType::Q8_G32_FP16: return {elements, 0, elements / 32 * 2};
    default: throw std::invalid_argument("hybrid experts: unsupported native format");
    }
}

std::array<const ops::CpuExpertProjection*, 3> projections(const ops::CpuExpertWeights& expert) {
    return {&expert.gate, &expert.up, &expert.down};
}

std::size_t expert_bytes(const ops::CpuExpertWeights& expert) {
    std::size_t bytes = 0;
    for (const auto* projection : projections(expert)) { bytes += planes(projection->weight).size(); }
    return bytes;
}

// Physical copies preserve every code and scale. The CPU and GPU keep the same representation;
// only its addresses change. A stored projection's planes need not be adjacent in the Model.
std::array<Weight, 3> pack(const ops::CpuExpertWeights& expert, std::byte* host,
                           std::byte* device) {
    std::array<Weight, 3> output;
    std::size_t offset = 0;
    const auto inputs = projections(expert);
    for (std::size_t p = 0; p < inputs.size(); ++p) {
        const Weight& source = inputs[p]->weight;
        const auto bytes = planes(source);
        Weight& target = output[p];
        target = source;
        target.payload = device + offset;
        target.payload_bytes = bytes.size();
        target.high_plane_bytes = bytes.high;
        const std::array<const void*, 3> pointers{source.qdata, source.qhigh, source.scales};
        const std::array<std::size_t, 3> sizes{bytes.low, bytes.high, bytes.scales};
        std::array<const void*, 3> relocated{};
        for (std::size_t plane = 0; plane < sizes.size(); ++plane) {
            if (sizes[plane] != 0) {
                if (!pointers[plane]) { throw std::invalid_argument("hybrid experts: missing weight plane"); }
                if (host) { std::memcpy(host + offset, pointers[plane], sizes[plane]); }
                relocated[plane] = device + offset;
                offset += aligned(sizes[plane]);
            }
        }
        target.qdata = relocated[0];
        target.qhigh = relocated[1];
        target.scales = relocated[2];
    }
    return output;
}

std::array<Weight, 3> copy_registered(const ops::CpuExpertWeights& expert, std::byte* device,
                                      cudaStream_t stream) {
    const auto output = pack(expert, nullptr, device);
    const auto source = projections(expert);
    for (std::size_t p = 0; p < source.size(); ++p) {
        const auto bytes = planes(source[p]->weight);
        const std::array<const void*, 3> from{source[p]->weight.qdata, source[p]->weight.qhigh, source[p]->weight.scales};
        const std::array<const void*, 3> to{output[p].qdata, output[p].qhigh, output[p].scales};
        const std::array<std::size_t, 3> sizes{bytes.low, bytes.high, bytes.scales};
        for (std::size_t plane = 0; plane < sizes.size(); ++plane) {
            if (sizes[plane]) {
                // Each copy stays within one stored plane and therefore one registration.
                CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(to[plane]), from[plane], sizes[plane],
                                           cudaMemcpyHostToDevice, stream));
            }
        }
    }
    return output;
}
} // namespace

struct HybridExperts::Impl {
    struct Layer {
        HybridExpertLayer inputs;
        std::size_t slot_bytes = 0;
        DeviceBuffer cache;
        std::vector<std::int32_t> slot_of, expert_of;
        std::vector<double> score;
    };
    struct Rank {
        DeviceBuffer slots, tables, parts, cpu_sum, sequence;
        std::array<std::unique_ptr<CudaCompletionEvent>, 2> staged;
        std::unique_ptr<CudaHostHandshake> handshake;
        std::atomic<std::uint32_t> served{0};
        std::uint64_t cache_bytes = 0;
        std::size_t slot_bytes = 0;
        std::size_t experts = 0;
    };
    struct Worker {
        ops::CpuExpertWorkspace workspace{static_cast<std::uint32_t>(kCpuTokens)};
        std::vector<std::uint16_t> input = std::vector<std::uint16_t>(kCpuTokens * kHidden);
        std::vector<float> output = std::vector<float>(kCpuTokens * kHidden);
    };

    DeviceContext& device;
    HybridExpertOptions options;
    std::uint32_t capacity;
    std::vector<Layer> layers;
    std::vector<Rank> ranks;
    std::vector<Worker> workers;
    ReadPool pool;
    std::function<void()> before_cpu;
    std::unique_ptr<PinnedHostBuffer> routes, activations, route_weights, selection, entries, sum;
    // Two staging blocks alternate; waiting before reuse keeps pageable Windows weights out of
    // large pinned allocations and lets CPU packing overlap the previous block's DMA.
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> staging;
    std::size_t staging_bytes = 0;
    std::vector<float> products = std::vector<float>(kCpuTokens * kTop * kHidden);
    std::vector<std::int32_t> cpu_jobs, dma_jobs, used;
    std::atomic<std::size_t> next_job{0};
    ExpertCacheStats counters;
    std::atomic<const std::atomic<bool>*> cancellation{nullptr};
    const std::atomic<bool>* cancelled = nullptr; // supervisor's borrow; released before retirement
    mutable std::mutex state_mutex;
    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::atomic<bool> active{false}, observing{false};
    std::atomic<bool> gpu_finished{false};
    std::mutex wake_mutex;
    std::condition_variable wake;
    std::atomic<bool> stopping{false};
    std::thread supervisor;

    void check_cancelled() const {
        if (cancelled && cancelled->load(std::memory_order_relaxed)) { throw ops::CpuExpertCancelled(); }
    }

    Impl(DeviceContext& d, std::vector<HybridExpertLayer> inputs, std::uint32_t tokens,
         std::uint64_t requested_cache, HybridExpertOptions o, std::function<void()> hook)
        : device(d), options(validated(o)), capacity(tokens), ranks(d.size()),
          workers(options.cpu_threads ? options.cpu_threads :
                      std::clamp(std::thread::hardware_concurrency(), 1U, 16U)),
          pool(workers.size()), before_cpu(std::move(hook)) {
        if (tokens == 0 || inputs.empty()) {
            throw std::invalid_argument("hybrid experts: invalid capacity, DMA share or worker count");
        }
        // Each rank's pool can hold one complete layer; cache allocations use only the memory
        // left after these required prefill resources have been allocated.
        std::vector<std::size_t> layer_counts(d.size(), 0);
        std::size_t max_experts = 0, max_slot_bytes = 0;
        for (auto& input : inputs) {
            if (input.rank >= ranks.size() || input.experts.size() < kTop || input.experts.size() > 512) {
                throw std::invalid_argument("hybrid experts: invalid layer or expert count");
            }
            const auto bytes = expert_bytes(input.experts.front());
            if (input.route_counts.empty()) { input.route_counts.resize(input.experts.size()); }
            if (input.route_counts.size() != input.experts.size()) {
                throw std::invalid_argument("hybrid experts: routing profile has the wrong expert count");
            }
            for (const auto& expert : input.experts) {
                if (expert_bytes(expert) != bytes) {
                    throw std::invalid_argument("hybrid experts: a bank must share its stored geometry");
                }
            }
            auto& rank = ranks[input.rank];
            rank.slot_bytes = std::max(rank.slot_bytes, bytes);
            rank.experts = std::max(rank.experts, input.experts.size());
            ++layer_counts[input.rank];
            max_experts = std::max(max_experts, input.experts.size());
            max_slot_bytes = std::max(max_slot_bytes, bytes);
            Layer layer;
            layer.slot_bytes = bytes;
            layer.slot_of.assign(input.experts.size(), -1);
            layer.score.assign(input.experts.size(), 0);
            layer.inputs = std::move(input);
            layers.push_back(std::move(layer));
        }
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            RankBinding bind(device, r);
            auto& rank = ranks[r];
            if (rank.experts == 0) { continue; }
            rank.slots = DeviceBuffer(rank.experts * rank.slot_bytes);
            rank.tables = DeviceBuffer(3 * rank.experts * sizeof(Weight));
            rank.parts = DeviceBuffer(kTop * tokens);
            rank.cpu_sum = DeviceBuffer(kCpuTokens * kHidden * sizeof(float));
            rank.sequence = DeviceBuffer(sizeof(std::uint32_t));
            rank.sequence.fill();
            rank.handshake = std::make_unique<CudaHostHandshake>(
                device.rank(r), static_cast<std::uint32_t*>(rank.sequence.p));
            for (auto& event : rank.staged) { event = std::make_unique<CudaCompletionEvent>(device.rank(r)); }
        }
        routes = std::make_unique<PinnedHostBuffer>(kTop * tokens * sizeof(std::int32_t));
        activations = std::make_unique<PinnedHostBuffer>(kCpuTokens * kHidden * sizeof(std::uint16_t));
        route_weights = std::make_unique<PinnedHostBuffer>(kCpuTokens * kTop * sizeof(float));
        selection = std::make_unique<PinnedHostBuffer>(kTop * tokens);
        entries = std::make_unique<PinnedHostBuffer>(3 * max_experts * sizeof(Weight));
        sum = std::make_unique<PinnedHostBuffer>(kCpuTokens * kHidden * sizeof(float));
        staging_bytes = std::max<std::size_t>(32U << 20, max_slot_bytes);
        if (std::any_of(layers.begin(), layers.end(), [](const auto& layer) { return !layer.inputs.registered; })) {
            for (auto& block : staging) { block = std::make_unique<PinnedHostBuffer>(staging_bytes); }
        }
        cpu_jobs.reserve(max_experts);
        dma_jobs.reserve(max_experts);
        used.reserve(max_experts);
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            if (!layer_counts[r]) { continue; }
            RankBinding bind(device, r);
            std::size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            const auto available = free > kMargin ? free - kMargin : 0;
            const auto budget = requested_cache == ~std::uint64_t{0} ? available :
                std::min<std::uint64_t>(available, requested_cache / ranks.size());
            for (auto& layer : layers) {
                if (layer.inputs.rank != r) { continue; }
                const auto slots = std::min<std::uint64_t>(layer.slot_of.size(),
                    budget / layer_counts[r] / layer.slot_bytes);
                std::vector<std::int32_t> order(layer.slot_of.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                    return layer.inputs.route_counts[a] > layer.inputs.route_counts[b];
                });
                layer.expert_of.assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(slots));
                if (slots) { layer.cache = DeviceBuffer(slots * layer.slot_bytes); }
                ranks[r].cache_bytes += slots * layer.slot_bytes;
                counters.slots += static_cast<std::uint32_t>(slots);
                // Equal counts keep ascending expert IDs; the selected placement is frozen
                // unless adaptive caching was requested and is part of the context image key.
                fill_cache(layer);
            }
        }
        supervisor = std::thread([this] { supervise(); });
    }

    ~Impl() {
        try { device.synchronize(); } catch (...) {}
        stopping.store(true, std::memory_order_release);
        wake.notify_one();
        if (supervisor.joinable()) { supervisor.join(); }
    }

    void fill_cache(Layer& layer) {
        auto& rank = ranks[layer.inputs.rank];
        const auto stream = device.rank(layer.inputs.rank).transfer_stream;
        const auto per_batch = std::max<std::size_t>(1, staging_bytes / layer.slot_bytes);
        std::size_t at = 0, batch = 0;
        while (at < layer.expert_of.size()) {
            if (layer.inputs.registered) {
                copy_registered(layer.inputs.experts[layer.expert_of[at]],
                    static_cast<std::byte*>(layer.cache.p) + at * layer.slot_bytes, stream);
                ++at;
                continue;
            }
            const auto ring = batch++ % staging.size();
            if (batch > staging.size()) { rank.staged[ring]->synchronize(); }
            const auto n = std::min(per_batch, layer.expert_of.size() - at);
            auto* host = static_cast<std::byte*>(staging[ring]->data());
            auto* target = static_cast<std::byte*>(layer.cache.p) + at * layer.slot_bytes;
            for (std::size_t j = 0; j < n; ++j) {
                pack(layer.inputs.experts[layer.expert_of[at + j]], host + j * layer.slot_bytes,
                     target + j * layer.slot_bytes);
            }
            CUDA_CHECK(cudaMemcpyAsync(target, host, n * layer.slot_bytes, cudaMemcpyHostToDevice, stream));
            rank.staged[ring]->record(stream);
            at += n;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t slot = 0; slot < layer.expert_of.size(); ++slot) {
            layer.slot_of[layer.expert_of[slot]] = static_cast<std::int32_t>(slot);
        }
        counters.admitted += layer.expert_of.size();
        counters.copied_bytes += layer.cache.bytes;
    }

    void admit(Layer& layer, std::int32_t expert, std::size_t slot) {
        auto* target = static_cast<std::byte*>(layer.cache.p) + slot * layer.slot_bytes;
        const auto stream = device.rank(layer.inputs.rank).transfer_stream;
        if (layer.inputs.registered) { copy_registered(layer.inputs.experts[expert], target, stream); }
        else {
            auto* host = static_cast<std::byte*>(staging[0]->data());
            pack(layer.inputs.experts[expert], host, target);
            CUDA_CHECK(cudaMemcpyAsync(target, host, layer.slot_bytes, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Publish residency only after the bytes are ready, and after the old consumer finished.
        const auto victim = layer.expert_of[slot];
        if (victim >= 0) { layer.slot_of[victim] = -1; }
        layer.expert_of[slot] = expert;
        layer.slot_of[expert] = static_cast<std::int32_t>(slot);
        ++counters.admitted;
        counters.copied_bytes += layer.slot_bytes;
    }

    void stage(Layer& layer, Rank& rank, Weight* tables) {
        const auto stream = device.rank(layer.inputs.rank).transfer_stream;
        const auto count = layer.inputs.experts.size();
        const auto per_batch = std::max<std::size_t>(1, staging_bytes / layer.slot_bytes);
        std::size_t at = 0, batch = 0;
        while (at < dma_jobs.size()) {
            check_cancelled();
            if (layer.inputs.registered) {
                const auto e = dma_jobs[at];
                const auto copied = copy_registered(layer.inputs.experts[e],
                    static_cast<std::byte*>(rank.slots.p) + at * layer.slot_bytes, stream);
                for (std::size_t p = 0; p < copied.size(); ++p) { tables[p * count + e] = copied[p]; }
                counters.copied_bytes += layer.slot_bytes;
                ++at;
                continue;
            }
            const auto ring = batch++ % staging.size();
            if (batch > staging.size()) { rank.staged[ring]->synchronize(); }
            const auto n = std::min(per_batch, dma_jobs.size() - at);
            auto* host = static_cast<std::byte*>(staging[ring]->data());
            auto* target = static_cast<std::byte*>(rank.slots.p) + at * layer.slot_bytes;
            for (std::size_t j = 0; j < n; ++j) {
                const auto e = dma_jobs[at + j];
                const auto weights = pack(layer.inputs.experts[e], host + j * layer.slot_bytes,
                                          target + j * layer.slot_bytes);
                for (std::size_t p = 0; p < weights.size(); ++p) { tables[p * count + e] = weights[p]; }
            }
            CUDA_CHECK(cudaMemcpyAsync(target, host, n * layer.slot_bytes, cudaMemcpyHostToDevice, stream));
            rank.staged[ring]->record(stream);
            counters.copied_bytes += n * layer.slot_bytes;
            at += n;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void cpu(Layer& layer, std::size_t worker, std::size_t tokens) {
        if (before_cpu) { before_cpu(); }
        auto& scratch = workers[worker];
        const auto* input = static_cast<const std::uint16_t*>(activations->data());
        const auto* ids = static_cast<const std::int32_t*>(routes->data());
        for (;;) {
            check_cancelled();
            const auto index = next_job.fetch_add(1, std::memory_order_relaxed);
            if (index >= cpu_jobs.size()) { return; }
            const auto expert = cpu_jobs[index];
            std::array<std::size_t, kCpuTokens> positions{};
            std::size_t columns = 0;
            for (std::size_t t = 0; t < tokens; ++t) {
                for (std::size_t k = 0; k < kTop; ++k) {
                    if (ids[t * kTop + k] != expert) { continue; }
                    positions[columns] = t * kTop + k;
                    std::memcpy(scratch.input.data() + columns * kHidden, input + t * kHidden,
                                kHidden * sizeof(std::uint16_t));
                    ++columns;
                    break; // moe_route selects each expert at most once per token.
                }
            }
            ops::moe_expert_cpu(std::span(scratch.input).first(columns * kHidden),
                                layer.inputs.experts[expert], scratch.workspace,
                                std::span(scratch.output).first(columns * kHidden),
                                ops::CpuExpertBackend::Automatic, cancelled);
            for (std::size_t j = 0; j < columns; ++j) {
                std::memcpy(products.data() + positions[j] * kHidden,
                            scratch.output.data() + j * kHidden, kHidden * sizeof(float));
            }
        }
    }

    void prepare(Layer& layer, std::size_t tokens, bool cpu_allowed, bool observe) {
        auto* host_ids = static_cast<const std::int32_t*>(routes->data());
        auto* parts = static_cast<std::uint8_t*>(selection->data());
        auto* tables = static_cast<Weight*>(entries->data());
        const auto count = layer.inputs.experts.size();
        std::fill_n(tables, 3 * count, Weight{});
        used.clear(); cpu_jobs.clear(); dma_jobs.clear();
        std::array<bool, 512> routed{};
        for (std::size_t pair = 0; pair < kTop * tokens; ++pair) {
            const auto e = host_ids[pair];
            if (e < 0 || std::size_t(e) >= count) { throw std::runtime_error("hybrid experts: invalid routed expert"); }
            routed[e] = true;
            if (observe) {
                ++counters.routes;
                if (layer.slot_of[e] >= 0) { ++counters.hits; }
                auto& count = layer.inputs.route_counts[e];
                if (count != std::numeric_limits<std::uint64_t>::max()) { ++count; }
            }
        }
        std::size_t misses = 0;
        for (std::size_t e = 0; e < count; ++e) { misses += routed[e] && layer.slot_of[e] < 0; }
        std::size_t dma_left = cpu_allowed ? static_cast<std::size_t>(std::ceil(misses * *options.dma_share)) : misses;
        std::array<bool, 512> gpu{};
        for (std::size_t e = 0; e < count; ++e) {
            if (!routed[e]) { continue; }
            used.push_back(static_cast<std::int32_t>(e));
            const auto slot = layer.slot_of[e];
            if (slot >= 0) {
                auto* target = static_cast<std::byte*>(layer.cache.p) + std::size_t(slot) * layer.slot_bytes;
                const auto resident = pack(layer.inputs.experts[e], nullptr, target);
                for (std::size_t p = 0; p < resident.size(); ++p) { tables[p * count + e] = resident[p]; }
                gpu[e] = true;
            } else if (dma_left) {
                --dma_left;
                dma_jobs.push_back(static_cast<std::int32_t>(e));
                gpu[e] = true;
            } else { cpu_jobs.push_back(static_cast<std::int32_t>(e)); }
        }
        for (std::size_t pair = 0; pair < kTop * tokens; ++pair) {
            const auto e = host_ids[pair];
            parts[pair] = gpu[e];
            if (observe && layer.slot_of[e] < 0) {
                if (gpu[e]) { ++counters.dma_routes; } else { ++counters.cpu_routes; }
            }
        }
    }

    void reduce(std::size_t tokens) {
        auto* total = static_cast<float*>(sum->data());
        const auto* parts = static_cast<const std::uint8_t*>(selection->data());
        const auto* probability = static_cast<const float*>(route_weights->data());
        std::fill_n(total, tokens * kHidden, 0.0F);
        // Worker completion order never controls the reduction order.
        for (std::size_t pair = 0; pair < tokens * kTop; ++pair) {
            if (parts[pair]) { continue; }
            for (std::size_t h = 0; h < kHidden; ++h) {
                total[pair / kTop * kHidden + h] += products[pair * kHidden + h] * probability[pair];
            }
        }
    }

    void adapt(Layer& layer, std::size_t tokens, bool observe) {
        if (observe && options.adaptive_cache && !layer.expert_of.empty()) {
            const auto* host_ids = static_cast<const std::int32_t*>(routes->data());
            const auto decay = std::pow(0.999, double(tokens));
            for (auto& score : layer.score) { score *= decay; }
            for (std::size_t pair = 0; pair < kTop * tokens; ++pair) { layer.score[host_ids[pair]] += 1; }
            std::int32_t best = -1;
            for (const auto e : used) {
                if (layer.slot_of[e] < 0 && (best < 0 || layer.score[e] > layer.score[best])) { best = e; }
            }
            const auto victim = std::min_element(layer.expert_of.begin(), layer.expert_of.end(), [&](auto a, auto b) {
                return layer.score[a] < layer.score[b];
            });
            if (best >= 0 && layer.score[best] > 1.25 * layer.score[*victim] + 0.5) {
                admit(layer, best, static_cast<std::size_t>(victim - layer.expert_of.begin()));
            }
        }
    }

    void remember_failure() noexcept {
        std::lock_guard lock(failure_mutex);
        if (!failure) { failure = std::current_exception(); }
    }

    void service(Rank& rank, const CudaHostHandshake::Request& request) noexcept {
        auto& handshake = *rank.handshake;
        const auto tokens = request.width;
        const bool cpu_allowed = request.flags & 1U;
        const bool observe = observing.load(std::memory_order_acquire);
        bool computing = false, prepared = false, succeeded = false;
        const std::lock_guard state_lock(state_mutex);
        try {
            // A failed graph still drains every subsequent exchange with inert operands. Its
            // incomplete output/state is discarded at the executor's transaction boundary.
            {
                const std::lock_guard lock(failure_mutex);
                if (failure) { std::rethrow_exception(failure); }
            }
            auto& layer = layers.at(request.tag);
            RankBinding bind(device, layer.inputs.rank);
            cancelled = cancellation.load(std::memory_order_acquire);
            check_cancelled();
            prepare(layer, tokens, cpu_allowed, observe);
            next_job.store(0, std::memory_order_relaxed);
            if (!cpu_jobs.empty()) {
                pool.start(std::min(workers.size(), cpu_jobs.size()),
                           [this, &layer, tokens](std::size_t worker) { cpu(layer, worker, tokens); });
                computing = true;
            }
            if (!dma_jobs.empty()) { stage(layer, rank, static_cast<Weight*>(entries->data())); }
            handshake.prepared(request.sequence);
            prepared = true;
            if (computing) { pool.finish(); computing = false; }
            check_cancelled();
            if (cpu_allowed) { reduce(tokens); }
            succeeded = true;
        } catch (...) {
            remember_failure();
            if (computing) { try { pool.finish(); } catch (...) {} }
            try {
                const auto& layer = layers.at(request.tag);
                RankBinding bind(device, layer.inputs.rank);
                CUDA_CHECK(cudaStreamSynchronize(device.rank(layer.inputs.rank).transfer_stream));
            } catch (...) { remember_failure(); }
            // If preparation has already been published, parts/tables belong to the GPU until
            // retirement. Otherwise mask every routed lane before letting the graph proceed.
            if (!prepared) {
                std::memset(selection->data(), 0, kTop * tokens);
                std::memset(entries->data(), 0, entries->size());
            }
            if (cpu_allowed) { std::memset(sum->data(), 0, tokens * kHidden * sizeof(float)); }
        }
        cancelled = nullptr;
        handshake.prepared(request.sequence);
        handshake.completed(request.sequence);
        // Never synchronize the compute stream here: its graph may already contain the next
        // request, which only this supervisor can prepare. Retire acknowledges just this layer.
        while (!handshake.retired(request.sequence) && !gpu_finished.load(std::memory_order_acquire) &&
               !stopping.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (succeeded && handshake.retired(request.sequence) && !stopping.load(std::memory_order_acquire)) {
            try {
                auto& layer = layers[request.tag];
                RankBinding bind(device, layer.inputs.rank);
                adapt(layer, tokens, observe);
            } catch (...) { remember_failure(); }
        }
        handshake.release(request.sequence);
    }

    void supervise() noexcept {
        while (!stopping.load(std::memory_order_acquire)) {
            if (!active.load(std::memory_order_acquire)) {
                std::unique_lock lock(wake_mutex);
                wake.wait(lock, [this] {
                    return active.load(std::memory_order_acquire) || stopping.load(std::memory_order_acquire);
                });
                continue;
            }
            bool worked = false;
            for (auto& rank : ranks) {
                if (!rank.handshake) { continue; }
                if (const auto request = rank.handshake->pending(rank.served.load(std::memory_order_acquire))) {
                    service(rank, *request);
                    rank.served.store(request->sequence, std::memory_order_release);
                    worked = true;
                }
            }
            if (!worked) { std::this_thread::yield(); }
        }
    }

    void run(std::size_t index, const Tensor& x, const Tensor& ids, const Tensor& weights,
             const Tensor& shared, ops::NativeMoeWeights banks, bool cpu_allowed,
             WorkspaceArena& workspace, Tensor& output) {
        auto& layer = layers.at(index);
        auto& rank = ranks[layer.inputs.rank];
        RankBinding bind(device, layer.inputs.rank);
        const auto stream = device.rank(layer.inputs.rank).stream;
        const auto tokens = static_cast<std::size_t>(x.ne[1]);
        if (tokens == 0 || tokens > capacity) { throw std::invalid_argument("hybrid experts: invalid token count"); }
        cpu_allowed = cpu_allowed && tokens <= kCpuTokens;
        CUDA_CHECK(cudaMemcpyAsync(routes->data(), ids.data, kTop * tokens * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        if (cpu_allowed) {
            CUDA_CHECK(cudaMemcpyAsync(activations->data(), x.data, tokens * kHidden * 2,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(route_weights->data(), weights.data, tokens * kTop * 4,
                                       cudaMemcpyDeviceToHost, stream));
        }
        rank.handshake->publish(static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(tokens),
                                cpu_allowed ? 1U : 0U, stream);
        rank.handshake->wait_prepared(stream);
        const auto count = layer.inputs.experts.size();
        CUDA_CHECK(cudaMemcpyAsync(rank.tables.p, entries->data(), 3 * count * sizeof(Weight),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(rank.parts.p, selection->data(), kTop * tokens,
                                   cudaMemcpyHostToDevice, stream));
        banks.gate.experts = static_cast<const Weight*>(rank.tables.p);
        banks.up.experts = banks.gate.experts + count;
        banks.down.experts = banks.up.experts + count;
        const Tensor mask(rank.parts.p, DType::U8, {static_cast<std::int32_t>(kTop), static_cast<std::int32_t>(tokens)});
        {
            auto scope = workspace.scope();
            ops::moe_experts_native(x, ids, weights, shared, banks, &mask, workspace, output, stream);
        }
        rank.handshake->wait_completed(stream);
        if (cpu_allowed) {
            CUDA_CHECK(cudaMemcpyAsync(rank.cpu_sum.p, sum->data(), tokens * kHidden * 4,
                                       cudaMemcpyHostToDevice, stream));
            const Tensor partial(rank.cpu_sum.p, DType::FP32, {static_cast<std::int32_t>(kHidden), static_cast<std::int32_t>(tokens)});
            ops::residual_add(partial, output, stream);
        }
        rank.handshake->retire(stream);
    }

    void finish() {
        if (!active.load(std::memory_order_acquire)) { return; }
        std::exception_ptr error;
        try { device.synchronize(); } catch (...) { error = std::current_exception(); }
        // An enqueue failure can leave a published request without a retire kernel. Once all
        // queued GPU work ends, let that request drain too before releasing borrowed pointers.
        gpu_finished.store(true, std::memory_order_release);
        for (auto& rank : ranks) {
            if (!rank.handshake) { continue; }
            while (rank.handshake->pending(rank.served.load(std::memory_order_acquire))) {
                std::this_thread::yield();
            }
        }
        active.store(false, std::memory_order_release);
        cancellation.store(nullptr, std::memory_order_release);
        {
            const std::lock_guard lock(failure_mutex);
            const auto worker_error = std::exchange(failure, nullptr);
            if (!error) { error = worker_error; }
        }
        if (error) { std::rethrow_exception(error); }
    }
};

HybridExperts::HybridExperts(DeviceContext& d, std::vector<HybridExpertLayer> layers,
                             std::uint32_t tokens, std::uint64_t cache_bytes,
                             HybridExpertOptions options, std::function<void()> before_cpu)
    : impl_(std::make_unique<Impl>(d, std::move(layers), tokens, cache_bytes, options,
                                   std::move(before_cpu))) {}
HybridExperts::~HybridExperts() = default;
void HybridExperts::run(std::size_t layer, const Tensor& x, const Tensor& ids, const Tensor& weights,
                         const Tensor& shared, ops::NativeMoeWeights banks, bool cpu_allowed,
                         WorkspaceArena& workspace, Tensor& output) {
    impl_->run(layer, x, ids, weights, shared, banks, cpu_allowed, workspace, output);
}
void HybridExperts::begin(const std::atomic<bool>* cancelled, bool observe) {
    const std::lock_guard lock(impl_->wake_mutex);
    impl_->cancellation.store(cancelled, std::memory_order_release);
    impl_->observing.store(observe, std::memory_order_release);
    impl_->gpu_finished.store(false, std::memory_order_release);
    impl_->active.store(true, std::memory_order_release);
    impl_->wake.notify_one();
}
void HybridExperts::finish() { impl_->finish(); }
ExpertRouteCounts HybridExperts::route_counts() const {
    const std::lock_guard lock(impl_->state_mutex);
    ExpertRouteCounts out;
    out.reserve(impl_->layers.size());
    for (const auto& layer : impl_->layers) { out.push_back(layer.inputs.route_counts); }
    return out;
}
ExpertCacheStats HybridExperts::stats() const noexcept {
    const std::lock_guard lock(impl_->state_mutex);
    return impl_->counters;
}
std::uint64_t HybridExperts::cache_bytes(std::size_t rank) const { return impl_->ranks.at(rank).cache_bytes; }
std::uint64_t HybridExperts::working_bytes(std::size_t index) const {
    const auto& r = impl_->ranks.at(index);
    return r.slots.bytes + r.tables.bytes + r.parts.bytes + r.cpu_sum.bytes + r.sequence.bytes;
}
std::string HybridExperts::execution_profile() const {
    const std::lock_guard lock(impl_->state_mutex);
    std::uint64_t hash = 1469598103934665603ULL;
    const auto word = [&](std::uint32_t value) {
        for (int i = 0; i < 4; ++i) { hash = (hash ^ ((value >> (8 * i)) & 255U)) * 1099511628211ULL; }
    };
    word(std::bit_cast<std::uint32_t>(*impl_->options.dma_share));
    word(impl_->options.adaptive_cache);
    for (const auto& layer : impl_->layers) {
        word(static_cast<std::uint32_t>(layer.inputs.rank));
        word(static_cast<std::uint32_t>(layer.slot_of.size()));
        for (const auto slot : layer.slot_of) { word(static_cast<std::uint32_t>(slot)); }
    }
    return "native-hybrid-" + std::string(ops::cpu_expert_backend_name(ops::CpuExpertBackend::Automatic)) +
           "-" + std::to_string(hash);
}

} // namespace ninfer::models::qwen4_exp
