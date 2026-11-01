#include "models/qwen3_5/execution/expert_runtime.h"
#include "core/vmm.h"
#include "ops/sparse_moe/streamed.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

NativeExpertRuntime::NativeExpertRuntime(const Model& model, DeviceContext& device,
                                           std::span<const std::uint64_t> keep_free_by_rank,
                                           std::optional<std::uint64_t> cache_bytes,
                                           NativeExpertProfile profile, NativeExpertCpu cpu)
    : profile_(std::move(profile)) {
    const auto& source = model.expert_sources();
    if (profile_.counts && profile_.counts->size() != source.layers.size()) {
        throw std::invalid_argument("expert profile does not describe this model's expert layers");
    }
    if (keep_free_by_rank.size() != device.size()) {
        throw std::invalid_argument("native expert reservation must cover every device rank");
    }
    const float dma_share = cpu.dma_share.value_or(0.0F);
    if (!(dma_share >= 0 && dma_share <= 1) || cpu.threads > 256) {
        throw std::invalid_argument("native experts: the DMA share must be 0..1 and CPU threads 0..256");
    }
    std::vector<std::vector<ExpertDeliveryLayer>> layers(device.size());
    std::vector<std::vector<ops::detail::StreamCpuLayer>> cpu_layers(device.size());
    for (const auto& layer : source.layers) {
        auto& rank = layers.at(layer.rank);
        handles_.push_back({layer.rank, rank.size()});
        rank.push_back(layer.delivery);
        ops::detail::StreamCpuLayer cpu_layer{layer.gate_up.format, layer.down.format, {}};
        if (layer.delivery.aliases.size() != cpu_layer.aliases.size()) {
            throw std::logic_error("native expert layer has no eight plane aliases");
        }
        std::copy(layer.delivery.aliases.begin(), layer.delivery.aliases.end(), cpu_layer.aliases.begin());
        cpu_layers.at(layer.rank).push_back(cpu_layer);
    }
    ranks_.resize(device.size());
    cpu_.resize(device.size());
    elastic_ = device.size() == 1 && vmm::granularity(device.rank(0).device) != 0;
    for (std::size_t rank = 0; rank < layers.size(); ++rank) {
        if (layers[rank].empty()) { continue; }
        ExpertDeliveryOptions options;
        options.pair_capacity = ops::detail::kSparseMoeStreamPairs;
        options.suspendable = model.weights_suspendable();
        if (cache_bytes) { options.cache_bytes = *cache_bytes / device.size(); }
        options.keep_free_bytes = expert_add(keep_free_by_rank[rank], 512ULL << 20);
        options.elastic = elastic_;
        const bool computes = std::any_of(cpu_layers[rank].begin(), cpu_layers[rank].end(),
                                          ops::detail::SparseMoeStreamCpu::computes);
        // One device only: every stage's delivery would start its own CPU pool.
        if (computes && dma_share < 1 && device.size() == 1) {
            cpu_[rank] = std::make_unique<ops::detail::SparseMoeStreamCpu>(std::move(cpu_layers[rank]));
            options.cpu.products      = cpu_[rank].get();
            options.cpu.input_bytes   = ops::detail::SparseMoeStreamCpu::input_bytes();
            options.cpu.product_bytes = ops::detail::SparseMoeStreamCpu::product_bytes();
            options.cpu.share         = 1.0 - double(dma_share);
            options.cpu.balance       = !cpu.dma_share;
            options.cpu.threads       = cpu.threads;
        }
        ranks_[rank] = std::make_unique<ExpertDelivery>(device.rank(rank), source.files,
                                                         std::move(layers[rank]), options);
    }
}

NativeExpertRuntime::~NativeExpertRuntime() {
    if (profile_.record.empty()) { return; }
    try {
        drain();
        write_expert_profile(profile_.record, profile_.artifact, totals());
    } catch (...) {
        // A destructor cannot report the failure; the file stays as it was.
    }
}

std::vector<std::size_t> NativeExpertRuntime::widths(const Model& model) {
    std::vector<std::size_t> out;
    for (const auto& layer : model.expert_sources().layers) { out.push_back(layer.delivery.experts.size()); }
    return out;
}

void NativeExpertRuntime::prime() {
    std::vector<ExpertRouteCounts> by_rank(ranks_.size());
    if (profile_.counts) {
        for (std::size_t index = 0; index < handles_.size(); ++index) {
            auto& rank = by_rank[handles_[index].rank];
            rank.resize(std::max(rank.size(), handles_[index].local + 1));
            rank[handles_[index].local] = (*profile_.counts)[index];
        }
    }
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
        if (ranks_[rank]) { ranks_[rank]->prime(profile_.counts ? &by_rank[rank] : nullptr); }
    }
}

ExpertRouteCounts NativeExpertRuntime::totals() const {
    std::vector<ExpertRouteCounts> by_rank(ranks_.size());
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
        if (ranks_[rank]) { by_rank[rank] = ranks_[rank]->totals(); }
    }
    ExpertRouteCounts out;
    for (const auto& handle : handles_) { out.push_back(by_rank[handle.rank].at(handle.local)); }
    return out;
}

ExpertDelivery& NativeExpertRuntime::layer(std::size_t index) const {
    return *ranks_.at(handles_.at(index).rank);
}
std::size_t NativeExpertRuntime::local_layer(std::size_t index) const {
    return handles_.at(index).local;
}
ExpertDeliveryMemory NativeExpertRuntime::memory(std::size_t rank) const noexcept {
    return rank < ranks_.size() && ranks_[rank] ? ranks_[rank]->memory() : ExpertDeliveryMemory{};
}
void NativeExpertRuntime::check() const {
    for (const auto& rank : ranks_) { if (rank) { rank->check(); } }
}
void NativeExpertRuntime::drain() {
    for (const auto& rank : ranks_) { if (rank) { rank->drain(); } }
}
void NativeExpertRuntime::release_device_backing() {
    for (const auto& rank : ranks_) { if (rank) { rank->release_device_backing(); } }
}
void NativeExpertRuntime::restore_device_backing() {
    try {
        for (const auto& rank : ranks_) { if (rank) { rank->restore_device_backing(); } }
    } catch (...) {
        for (const auto& rank : ranks_) {
            if (rank) { try { rank->release_device_backing(); } catch (...) {} }
        }
        throw;
    }
}
std::uint64_t NativeExpertRuntime::device_backing_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const auto& rank : ranks_) { if (rank) { bytes += rank->state_backing_bytes(); } }
    return bytes;
}
std::uint64_t NativeExpertRuntime::grow_cache() {
    if (!elastic_ || !ranks_.front()) { throw std::logic_error("native expert cache is not elastic"); }
    return ranks_.front()->grow_cache();
}
std::uint64_t NativeExpertRuntime::shrink_cache(std::uint64_t bytes) {
    if (!elastic_ || !ranks_.front()) { throw std::logic_error("native expert cache is not elastic"); }
    return ranks_.front()->shrink_cache(bytes);
}
void NativeExpertRuntime::set_keep_free_bytes(std::uint64_t bytes) {
    for (const auto& rank : ranks_) { if (rank) { rank->set_keep_free_bytes(bytes); } }
}
std::string NativeExpertRuntime::cpu_description() const {
    for (std::size_t rank = 0; rank < ranks_.size(); ++rank) {
        if (ranks_[rank] && cpu_[rank]) {
            return ranks_[rank]->cpu_description() + ", " + cpu_[rank]->backend_name();
        }
    }
    return {};
}
ExpertDeliveryStats NativeExpertRuntime::stats() const {
    ExpertDeliveryStats out;
    for (const auto& rank : ranks_) {
        if (rank) { out = out + rank->stats(); }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::execution
