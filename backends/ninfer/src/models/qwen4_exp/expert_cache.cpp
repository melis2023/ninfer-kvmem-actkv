#include "models/qwen4_exp/expert_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::uint64_t kAlign = 256;
// Per-token decay of an expert's route count: a half-life of about 700 tokens.
constexpr double kDecay = 0.999;
// A candidate displaces a cached expert only when it is clearly hotter, so near-ties do not thrash.
constexpr double kHysteresis = 1.25;
constexpr double kFloor      = 0.5;
// Zero bytes after every slot's down matrix, which the expert matrix kernel reads past a down
// row's end (expert_stream.cpp says why they must be zeros, not the next slot's gate).
constexpr std::uint64_t kTail = 256;

std::uint64_t aligned(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

} // namespace

struct ExpertCache::Layer {
    ExpertCacheLayer banks;
    std::uint32_t slots      = 0;
    std::uint64_t slot_bytes = 0;
    DeviceBuffer storage;
    std::vector<std::int32_t> slot_of;
    std::vector<std::int32_t> expert_of;
    std::vector<double> score;
    std::vector<const void*> current[3];
    std::size_t staging = 0; // byte offset of its three tables in the pinned staging

    [[nodiscard]] const ExpertBank& bank(int k) const {
        return k == 0 ? banks.gate : k == 1 ? banks.up : banks.down;
    }

    [[nodiscard]] std::uint64_t offset(int k) const {
        return k == 0   ? 0
               : k == 1 ? aligned(banks.gate.expert_bytes)
                        : aligned(banks.gate.expert_bytes) + aligned(banks.up.expert_bytes);
    }
};

ExpertCache::ExpertCache(DeviceContext& device, std::vector<ExpertCacheLayer> layers,
                         std::span<const std::uint64_t> bytes_by_rank)
    : device_(device) {
    std::map<std::size_t, std::size_t> layers_on_rank;
    for (const auto& layer : layers) { ++layers_on_rank[layer.rank]; }
    std::size_t staging_bytes = 0;
    for (auto& banks : layers) {
        const std::size_t experts = banks.gate.host.size();
        if (experts == 0 || banks.up.host.size() != experts || banks.down.host.size() != experts) {
            throw std::invalid_argument("expert cache: a layer's banks differ in expert count");
        }
        Layer layer;
        layer.banks      = std::move(banks);
        layer.slot_bytes = aligned(layer.banks.gate.expert_bytes) +
                           aligned(layer.banks.up.expert_bytes) +
                           aligned(layer.banks.down.expert_bytes + kTail);
        const std::uint64_t share =
            layer.banks.rank < bytes_by_rank.size()
                ? bytes_by_rank[layer.banks.rank] / layers_on_rank[layer.banks.rank]
                : 0;
        layer.slots =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(share / layer.slot_bytes, experts));
        layer.slot_of.assign(experts, -1);
        layer.expert_of.assign(layer.slots, -1);
        layer.score.assign(experts, 0.0);
        for (int k = 0; k < 3; ++k) { layer.current[k] = layer.bank(k).host; }
        if (layer.slots != 0) {
            RankBinding bind(device_, layer.banks.rank);
            layer.storage = DeviceBuffer(std::size_t(layer.slots) * layer.slot_bytes);
            CUDA_CHECK(cudaMemset(layer.storage.p, 0, layer.storage.bytes));
        }
        layer.staging = staging_bytes;
        staging_bytes += 3 * experts * sizeof(void*);
        stats_.slots += layer.slots;
        layers_.push_back(std::move(layer));
    }
    staging_ = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(staging_bytes, 64));
}

ExpertCache::~ExpertCache() = default;

void ExpertCache::observe(std::size_t index, std::span<const std::int32_t> ids,
                          std::uint32_t tokens) {
    Layer& layer       = layers_.at(index);
    const double decay = std::pow(kDecay, double(tokens));
    for (double& score : layer.score) { score *= decay; }
    for (const std::int32_t id : ids) {
        if (id < 0 || std::size_t(id) >= layer.score.size()) { continue; }
        layer.score[id] += 1.0;
        ++stats_.routes;
        if (layer.slot_of[id] >= 0) { ++stats_.hits; }
    }
}

void ExpertCache::rebalance(std::uint64_t byte_budget) {
    if (layers_.empty() || stats_.slots == 0) { return; }
    const std::uint64_t per_layer = byte_budget / layers_.size();
    auto* staging                 = static_cast<std::byte*>(staging_->data());
    for (Layer& layer : layers_) {
        if (layer.slots == 0) { continue; }
        const std::size_t experts = layer.score.size();
        std::vector<std::int32_t> candidates;
        for (std::size_t e = 0; e < experts; ++e) {
            if (layer.slot_of[e] < 0 && layer.score[e] > 0.0) {
                candidates.push_back(std::int32_t(e));
            }
        }
        if (candidates.empty()) { continue; }
        const std::size_t limit =
            std::max<std::size_t>(1, static_cast<std::size_t>(per_layer / layer.slot_bytes));
        const std::size_t considered = std::min(limit, candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + std::ptrdiff_t(considered),
                          candidates.end(), [&](std::int32_t a, std::int32_t b) {
                              return layer.score[a] != layer.score[b]
                                         ? layer.score[a] > layer.score[b]
                                         : a < b;
                          });
        candidates.resize(considered);
        // Free slots first, then the coldest cached experts, coldest first.
        std::vector<std::int32_t> free_slots, victims;
        for (std::uint32_t s = 0; s < layer.slots; ++s) {
            if (layer.expert_of[s] < 0) {
                free_slots.push_back(std::int32_t(s));
            } else {
                victims.push_back(layer.expert_of[s]);
            }
        }
        std::sort(victims.begin(), victims.end(), [&](std::int32_t a, std::int32_t b) {
            return layer.score[a] != layer.score[b] ? layer.score[a] < layer.score[b] : a > b;
        });
        std::size_t next_free = 0, next_victim = 0;
        bool changed = false;
        RankBinding bind(device_, layer.banks.rank);
        for (const std::int32_t expert : candidates) {
            std::int32_t slot = -1;
            if (next_free < free_slots.size()) {
                slot = free_slots[next_free++];
            } else if (next_victim < victims.size()) {
                const std::int32_t victim = victims[next_victim];
                if (!(layer.score[expert] > kHysteresis * layer.score[victim] + kFloor)) { break; }
                ++next_victim;
                slot                  = layer.slot_of[victim];
                layer.slot_of[victim] = -1;
                for (int k = 0; k < 3; ++k) {
                    layer.current[k][victim] = layer.bank(k).host[victim];
                }
            } else {
                break;
            }
            auto* base =
                static_cast<std::byte*>(layer.storage.p) + std::size_t(slot) * layer.slot_bytes;
            for (int k = 0; k < 3; ++k) {
                const ExpertBank& bank = layer.bank(k);
                std::byte* target      = base + layer.offset(k);
                CUDA_CHECK(cudaMemcpyAsync(target, bank.host[expert],
                                           std::size_t(bank.expert_bytes), cudaMemcpyHostToDevice,
                                           layer.banks.stream));
                layer.current[k][expert] = target;
                stats_.copied_bytes += std::uint64_t(bank.expert_bytes);
            }
            layer.slot_of[expert] = slot;
            layer.expert_of[slot] = expert;
            ++stats_.admitted;
            changed = true;
        }
        if (!changed) { continue; }
        for (int k = 0; k < 3; ++k) {
            const std::size_t bytes = experts * sizeof(void*);
            std::byte* mirror       = staging + layer.staging + std::size_t(k) * bytes;
            std::memcpy(mirror, layer.current[k].data(), bytes);
            CUDA_CHECK(cudaMemcpyAsync(layer.bank(k).table, mirror, bytes, cudaMemcpyHostToDevice,
                                       layer.banks.stream));
        }
    }
}

const void* ExpertCache::cached(std::size_t index, int k, std::int32_t expert) const {
    const Layer& layer = layers_.at(index);
    return layer.slot_of.at(static_cast<std::size_t>(expert)) >= 0 ? layer.current[k][expert]
                                                                   : nullptr;
}

ExpertCacheStats ExpertCache::stats() const noexcept { return stats_; }

} // namespace ninfer::models::qwen4_exp
