#include "models/qwen4_exp/expert_cache.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::uint64_t kAlign = 256;
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
    // Slots from base_slots on live in `extra`, allocated by grow() once startup is done, and the
    // extra_slots after those in the lent buffers' blocks, from lent_first on (lend()).
    DeviceBuffer extra;
    std::uint32_t base_slots = 0, extra_slots = 0, lent_first = 0;
    struct Block {
        std::byte* base     = nullptr;
        std::uint32_t count = 0;
    };
    std::array<Block, 2> lent{};
    std::vector<std::int32_t> slot_of;   // a held or pending expert's slot, else -1
    std::vector<std::int32_t> expert_of; // a slot's held or pending expert, else -1
    std::vector<std::int32_t> free_slots;
    std::vector<unsigned char> pending;  // its copy is in flight: the tables still read the host
    std::vector<std::uint64_t> total;
    // Admitted by the misses since the last observe: its routes then were misses.
    std::vector<unsigned char> fresh;
    bool handed_over = false;
    bool dirty       = false;
    std::vector<const void*> current[3];
    std::size_t staging = 0; // byte offset of its three tables in the pinned staging

    [[nodiscard]] const ExpertBank& bank(int k) const {
        return k == 0 ? banks.gate : k == 1 ? banks.up : banks.down;
    }

    [[nodiscard]] std::byte* slot_base(std::int64_t slot) const {
        if (slot < std::int64_t(base_slots)) {
            return static_cast<std::byte*>(storage.p) + std::size_t(slot) * slot_bytes;
        }
        slot -= base_slots;
        if (slot < std::int64_t(extra_slots)) {
            return static_cast<std::byte*>(extra.p) + std::size_t(slot) * slot_bytes;
        }
        slot -= extra_slots;
        if (slot >= std::int64_t(lent[0].count)) {
            return lent[1].base + std::size_t(slot - lent[0].count) * slot_bytes;
        }
        return lent[0].base + std::size_t(slot) * slot_bytes;
    }

    [[nodiscard]] std::uint32_t lent_end() const { return lent_first + lent[0].count + lent[1].count; }

    [[nodiscard]] std::uint64_t offset(int k) const {
        return k == 0   ? 0
               : k == 1 ? aligned(banks.gate.expert_bytes)
                        : aligned(banks.gate.expert_bytes) + aligned(banks.up.expert_bytes);
    }

    [[nodiscard]] std::size_t experts() const { return slot_of.size(); }

    void point_slot(std::int32_t expert, std::int32_t slot) {
        for (int k = 0; k < 3; ++k) {
            current[k][std::size_t(expert)] = slot_base(slot) + offset(k);
        }
        dirty = true;
    }
    void point_host(std::int32_t expert) {
        for (int k = 0; k < 3; ++k) { current[k][std::size_t(expert)] = bank(k).host[std::size_t(expert)]; }
        dirty = true;
    }
};

namespace {

// Slots each layer of one rank takes from `bytes` more, given the slots `have` it holds: an even
// share for every layer (the same number of slots each), unless a profile ranks the text layers'
// (layer, expert) pairs: then the last layer when it is the MTP block's keeps the even share, and
// the text layers take the ranked pairs in turn (each layer's `have` best already placed) while a
// slot fits the bytes left, then even shares of what remains.
std::vector<std::uint32_t> share_slots(std::uint64_t bytes, const std::vector<std::uint64_t>& slot_bytes,
                                       const std::vector<std::size_t>& experts,
                                       const std::vector<std::uint32_t>& have,
                                       const std::vector<const std::vector<std::uint64_t>*>& profile,
                                       std::size_t mtp) {
    const std::size_t n = slot_bytes.size();
    std::vector<std::uint32_t> out(n, 0);
    std::uint64_t per_slot = 0;
    for (const auto b : slot_bytes) { per_slot += b; }
    if (n == 0 || per_slot == 0) { return out; }
    const auto room = [&](std::size_t l) {
        return experts[l] > have[l] + out[l] ? experts[l] - have[l] - out[l] : std::size_t(0);
    };
    const bool ranked = std::any_of(profile.begin(), profile.end(), [](auto* p) { return p != nullptr; });
    const std::uint64_t even = bytes / per_slot;
    if (!ranked) {
        for (std::size_t l = 0; l < n; ++l) { out[l] = std::uint32_t(std::min<std::uint64_t>(even, room(l))); }
        return out;
    }
    if (mtp < n) {
        out[mtp] = std::uint32_t(std::min<std::uint64_t>(even, room(mtp)));
        bytes -= std::uint64_t(out[mtp]) * slot_bytes[mtp];
    }
    // (count, layer, expert rank within its layer), most routed first.
    std::vector<std::tuple<std::uint64_t, std::size_t, std::size_t>> pairs;
    for (std::size_t l = 0; l < n; ++l) {
        if (l == mtp || profile[l] == nullptr) { continue; }
        std::vector<std::uint64_t> counts = *profile[l];
        std::sort(counts.begin(), counts.end(), std::greater<>());
        for (std::size_t j = have[l]; j < counts.size() && counts[j] > 0; ++j) {
            pairs.emplace_back(counts[j], l, j);
        }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) > std::get<0>(b);
    });
    for (const auto& [count, l, j] : pairs) {
        if (room(l) == 0 || slot_bytes[l] > bytes) { continue; }
        ++out[l];
        bytes -= slot_bytes[l];
    }
    // What the ranking leaves: one slot a layer in turn.
    for (bool placed = true; placed;) {
        placed = false;
        for (std::size_t l = 0; l < n; ++l) {
            if (l == mtp || room(l) == 0 || slot_bytes[l] > bytes) { continue; }
            ++out[l];
            bytes -= slot_bytes[l];
            placed = true;
        }
    }
    return out;
}

} // namespace

ExpertCache::ExpertCache(DeviceContext& device, std::vector<ExpertCacheLayer> layers,
                         std::span<const std::uint64_t> bytes_by_rank,
                         const std::vector<std::vector<std::uint64_t>>* profile, bool last_is_mtp,
                         ExpertTierOptions tier)
    : device_(device) {
    const auto slot_bytes_of = [](const ExpertCacheLayer& banks) {
        return aligned(banks.gate.expert_bytes) + aligned(banks.up.expert_bytes) +
               aligned(banks.down.expert_bytes + kTail);
    };
    if (profile != nullptr && profile->size() != layers.size()) {
        throw std::invalid_argument("expert cache: the profile describes other layers");
    }
    std::size_t staging_bytes = 0;
    std::vector<std::uint32_t> experts_per_layer, pools;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& banks               = layers[index];
        const std::size_t experts = banks.gate.host.size();
        if (experts == 0 || banks.up.host.size() != experts || banks.down.host.size() != experts) {
            throw std::invalid_argument("expert cache: a layer's banks differ in expert count");
        }
        if (profile != nullptr && (*profile)[index].size() != experts) {
            throw std::invalid_argument("expert cache: the profile's bank width differs");
        }
        Layer layer;
        layer.slot_bytes = slot_bytes_of(banks);
        layer.banks      = std::move(banks);
        layer.slot_of.assign(experts, -1);
        layer.pending.assign(experts, 0);
        layer.total.assign(experts, 0);
        layer.fresh.assign(experts, 0);
        for (int k = 0; k < 3; ++k) { layer.current[k] = layer.bank(k).host; }
        layer.staging = staging_bytes;
        staging_bytes += 3 * experts * sizeof(void*);
        experts_per_layer.push_back(std::uint32_t(experts));
        pools.push_back(std::uint32_t(index));
        layers_.push_back(std::move(layer));
    }
    std::map<std::size_t, std::vector<std::size_t>> on_rank;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        on_rank[layers_[index].banks.rank].push_back(index);
    }
    for (const auto& [rank, members] : on_rank) {
        std::vector<std::uint64_t> slot_bytes;
        std::vector<std::size_t> experts;
        std::vector<const std::vector<std::uint64_t>*> ranked;
        std::size_t mtp = members.size();
        for (std::size_t i = 0; i < members.size(); ++i) {
            const Layer& layer = layers_[members[i]];
            slot_bytes.push_back(layer.slot_bytes);
            experts.push_back(layer.experts());
            ranked.push_back(profile != nullptr ? &(*profile)[members[i]] : nullptr);
            if (last_is_mtp && members[i] + 1 == layers_.size()) { mtp = i; }
        }
        const std::uint64_t bytes = rank < bytes_by_rank.size() ? bytes_by_rank[rank] : 0;
        const auto counts = share_slots(bytes, slot_bytes, experts,
                                        std::vector<std::uint32_t>(members.size(), 0), ranked, mtp);
        for (std::size_t i = 0; i < members.size(); ++i) {
            Layer& layer = layers_[members[i]];
            layer.slots  = counts[i];
            layer.expert_of.assign(layer.slots, -1);
            for (std::int32_t s = std::int32_t(layer.slots) - 1; s >= 0; --s) { layer.free_slots.push_back(s); }
            if (layer.slots != 0) {
                RankBinding bind(device_, layer.banks.rank);
                layer.storage = DeviceBuffer(std::size_t(layer.slots) * layer.slot_bytes);
                layer.storage.fill(0);
            }
            layer.base_slots = layer.slots;
            stats_.slots += layer.slots;
        }
    }
    staging_ = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(staging_bytes, 64));
    tier_    = std::make_unique<ExpertTier>(experts_per_layer, pools, tier);
    if (last_is_mtp && !layers_.empty()) { mtp_ = layers_.size() - 1; }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        tier_->set_free(std::uint32_t(index), layers_[index].slots);
    }
    side_.assign(device_.size(), nullptr);
    evicted_.assign(device_.size(), nullptr);
    uploaded_.assign(device_.size(), nullptr);
    handover_.assign(device_.size(), nullptr);
    holding_.assign(device_.size(), 0);
    for (const auto& [rank, members] : on_rank) {
        RankBinding bind(device_, rank);
        CUDA_CHECK(cudaStreamCreateWithFlags(&side_[rank], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evicted_[rank], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&uploaded_[rank], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&handover_[rank], cudaEventDisableTiming));
    }
}

ExpertCache::~ExpertCache() {
    for (std::size_t r = 0; r < side_.size(); ++r) {
        if (side_[r] == nullptr) { continue; }
        RankBinding bind(device_, r);
        cudaStreamSynchronize(side_[r]);
        cudaStreamDestroy(side_[r]);
        cudaEventDestroy(evicted_[r]);
        cudaEventDestroy(uploaded_[r]);
        cudaEventDestroy(handover_[r]);
    }
    for (const auto& copy : copies_) {
        if (copy.done != nullptr && std::find(spare_.begin(), spare_.end(), copy.done) == spare_.end()) {
            spare_.push_back(copy.done);
        }
    }
    for (auto event : spare_) { cudaEventDestroy(event); }
}

void ExpertCache::observe(std::size_t index, std::span<const std::int32_t> ids, bool learn) {
    std::lock_guard lock(mutex_);
    Layer& layer = layers_.at(index);
    for (const std::int32_t id : ids) {
        if (id < 0 || std::size_t(id) >= layer.experts()) { continue; }
        ++layer.total[std::size_t(id)];
        ++stats_.routes;
        if (layer.slot_of[std::size_t(id)] >= 0 && !layer.pending[std::size_t(id)] &&
            !layer.fresh[std::size_t(id)]) {
            ++stats_.hits;
        }
    }
    if (learn) { tier_->route(index, ids); }
    std::fill(layer.fresh.begin(), layer.fresh.end(), 0);
}

void ExpertCache::upload(Layer& layer, cudaStream_t stream) {
    auto* staging           = static_cast<std::byte*>(staging_->data());
    const std::size_t bytes = layer.experts() * sizeof(void*);
    for (int k = 0; k < 3; ++k) {
        std::byte* mirror = staging + layer.staging + std::size_t(k) * bytes;
        std::memcpy(mirror, layer.current[k].data(), bytes);
        CUDA_CHECK(cudaMemcpyAsync(layer.bank(k).table, mirror, bytes, cudaMemcpyHostToDevice, stream));
    }
    layer.dirty = false;
}

void ExpertCache::land() {
    std::vector<cudaEvent_t> landed;
    for (auto it = copies_.begin(); it != copies_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        Layer& layer = layers_[it->layer];
        layer.pending[std::size_t(it->expert)] = 0;
        layer.point_slot(it->expert, it->slot);
        tier_->set(it->layer, it->expert, ExpertTier::State::Held);
        if (std::find(landed.begin(), landed.end(), it->done) == landed.end()) { landed.push_back(it->done); }
        it = copies_.erase(it);
    }
    for (const auto event : landed) {
        const bool used = std::any_of(copies_.begin(), copies_.end(),
                                      [&](const Copy& copy) { return copy.done == event; });
        if (!used) { spare_.push_back(event); }
    }
}

void ExpertCache::adapt() {
    std::lock_guard lock(mutex_);
    // Candidates whose copies have landed: their entries move to their slots.
    land();
    // A round's swaps start over the passes until the next round, an even share each pass: a
    // whole round at once would hold the bus past the draft and delay the next pass's misses. A
    // started swap's victim leaves the tables now, its candidate's copy starts once it has.
    auto round = tier_->end_pass();
    if (!round.empty()) { queued_ = std::move(round); }
    const std::uint32_t every = std::max<std::uint32_t>(tier_->options().every, 1);
    const std::size_t share   = (std::size_t(tier_->options().swaps) + every - 1) / every;
    const std::size_t take    = std::min(share, queued_.size());
    const std::vector<ExpertTier::Swap> swaps(queued_.begin(),
                                              queued_.begin() + std::ptrdiff_t(take));
    queued_.erase(queued_.begin(), queued_.begin() + std::ptrdiff_t(take));
    std::vector<Copy> started;
    for (const auto& swap : swaps) {
        Layer& layer = layers_[swap.layer];
        // The misses may have admitted the candidate or taken the victim since the round.
        if (layer.slot_of[std::size_t(swap.in)] >= 0) { continue; }
        std::int32_t slot = -1;
        if (swap.out >= 0) {
            slot = layer.slot_of[std::size_t(swap.out)];
            if (slot < 0 || layer.pending[std::size_t(swap.out)]) { continue; }
            layer.slot_of[std::size_t(swap.out)] = -1;
            layer.point_host(swap.out);
            tier_->set(swap.layer, swap.out, ExpertTier::State::Out);
        } else {
            if (layer.free_slots.empty()) { continue; }
            slot = layer.free_slots.back();
            layer.free_slots.pop_back();
        }
        layer.expert_of[std::size_t(slot)]  = swap.in;
        layer.slot_of[std::size_t(swap.in)] = slot;
        layer.pending[std::size_t(swap.in)] = 1;
        tier_->set(swap.layer, swap.in, ExpertTier::State::Pending);
        started.push_back({swap.layer, swap.in, slot, nullptr});
    }
    // adapt runs once the last pass has ended, and the draft that may still run reads only the MTP
    // block's tables and slots. So the text layers' tables go up on the side stream at once, ahead
    // of its copies, and the compute stream waits for them before the next pass; the MTP block's
    // follow the draft on the compute stream.
    std::map<std::size_t, cudaStream_t> side_tables;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        Layer& layer = layers_[index];
        if (!layer.dirty) { continue; }
        const std::size_t rank = layer.banks.rank;
        RankBinding bind(device_, rank);
        if (index == mtp_) {
            upload(layer, layer.banks.stream);
        } else {
            upload(layer, side_[rank]);
            side_tables[rank] = layer.banks.stream;
        }
    }
    for (const auto& [rank, compute] : side_tables) {
        RankBinding bind(device_, rank);
        CUDA_CHECK(cudaEventRecord(uploaded_[rank], side_[rank]));
        CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[rank], 0));
    }
    // A text layer's candidates are copied right after: no kernel reads a text victim's slot
    // before the next pass, which reads its entries on the host. The MTP block's candidates wait
    // until the compute stream is done with its victims.
    std::map<std::size_t, cudaEvent_t> round_done;
    std::map<std::size_t, bool> after_draft;
    for (const bool mtp : {false, true}) {
        for (auto copy : started) {
            if ((copy.layer == mtp_) != mtp) { continue; }
            Layer& layer           = layers_[copy.layer];
            const std::size_t rank = layer.banks.rank;
            RankBinding bind(device_, rank);
            if (round_done.count(rank) == 0) {
                cudaEvent_t done = nullptr;
                if (!spare_.empty()) {
                    done = spare_.back();
                    spare_.pop_back();
                } else {
                    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
                }
                round_done[rank] = done;
            }
            if (mtp && !after_draft[rank]) {
                CUDA_CHECK(cudaEventRecord(evicted_[rank], layer.banks.stream));
                CUDA_CHECK(cudaStreamWaitEvent(side_[rank], evicted_[rank], 0));
                after_draft[rank] = true;
            }
            for (int k = 0; k < 3; ++k) {
                const ExpertBank& bank = layer.bank(k);
                CUDA_CHECK(cudaMemcpyAsync(layer.slot_base(copy.slot) + layer.offset(k),
                                           bank.host[std::size_t(copy.expert)],
                                           std::size_t(bank.expert_bytes), cudaMemcpyHostToDevice,
                                           side_[rank]));
                stats_.copied_bytes += std::uint64_t(bank.expert_bytes);
            }
            ++stats_.admitted;
            copy.done = round_done[rank];
            copies_.push_back(copy);
        }
    }
    for (const auto& [rank, done] : round_done) {
        RankBinding bind(device_, rank);
        CUDA_CHECK(cudaEventRecord(done, side_[rank]));
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        tier_->set_free(std::uint32_t(index), std::uint32_t(layers_[index].free_slots.size()));
    }
}

void ExpertCache::seed(const std::vector<std::vector<std::uint64_t>>& counts) {
    std::lock_guard lock(mutex_);
    if (counts.size() != layers_.size()) {
        throw std::invalid_argument("expert cache: the profile describes other layers");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        Layer& layer              = layers_[index];
        const std::size_t experts = layer.experts();
        if (counts[index].size() != experts) {
            throw std::invalid_argument("expert cache: the profile's bank width differs");
        }
        std::vector<std::int32_t> order(experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
            return counts[index][std::size_t(a)] > counts[index][std::size_t(b)];
        });
        RankBinding bind(device_, layer.banks.rank);
        for (const std::int32_t expert : order) {
            if (layer.free_slots.empty() || counts[index][std::size_t(expert)] == 0) { break; }
            if (layer.slot_of[std::size_t(expert)] >= 0) { continue; }
            const std::int32_t slot = layer.free_slots.back();
            layer.free_slots.pop_back();
            for (int k = 0; k < 3; ++k) {
                const ExpertBank& bank = layer.bank(k);
                CUDA_CHECK(cudaMemcpyAsync(layer.slot_base(slot) + layer.offset(k),
                                           bank.host[std::size_t(expert)],
                                           std::size_t(bank.expert_bytes), cudaMemcpyHostToDevice,
                                           layer.banks.stream));
                stats_.copied_bytes += std::uint64_t(bank.expert_bytes);
            }
            layer.slot_of[std::size_t(expert)] = slot;
            layer.expert_of[std::size_t(slot)] = expert;
            layer.point_slot(expert, slot);
            tier_->set(index, expert, ExpertTier::State::Held);
            ++stats_.admitted;
        }
        if (layer.dirty) { upload(layer, layer.banks.stream); }
        CUDA_CHECK(cudaStreamSynchronize(layer.banks.stream));
        tier_->set_free(std::uint32_t(index), std::uint32_t(layer.free_slots.size()));
    }
}

std::vector<std::vector<std::uint64_t>> ExpertCache::totals() const {
    std::lock_guard lock(mutex_);
    std::vector<std::vector<std::uint64_t>> out;
    out.reserve(layers_.size());
    for (const Layer& layer : layers_) { out.push_back(layer.total); }
    return out;
}

const void* ExpertCache::cached(std::size_t index, int k, std::int32_t expert) const {
    std::lock_guard lock(mutex_);
    const Layer& layer = layers_.at(index);
    const auto e       = static_cast<std::size_t>(expert);
    return layer.slot_of.at(e) >= 0 && !layer.pending[e] ? layer.current[k][e] : nullptr;
}

ExpertCacheStats ExpertCache::stats() const noexcept {
    std::lock_guard lock(mutex_);
    return stats_;
}

void ExpertCache::hand_over(std::size_t index) {
    std::lock_guard lock(mutex_);
    layers_.at(index).handed_over = true;
}

bool ExpertCache::admit(std::size_t index, std::int32_t expert, int pairs,
                        std::span<const unsigned char> in_use, Admitted& out) {
    std::lock_guard lock(mutex_);
    Layer& layer = layers_.at(index);
    if (!layer.handed_over || layer.slots == 0 || expert < 0 ||
        std::size_t(expert) >= layer.experts() || layer.slot_of[std::size_t(expert)] >= 0 ||
        in_use.size() < layer.experts()) {
        return false;
    }
    std::int32_t slot = -1;
    out               = Admitted{};
    if (!layer.free_slots.empty()) {
        slot = layer.free_slots.back();
        layer.free_slots.pop_back();
        tier_->set_free(std::uint32_t(index), std::uint32_t(layer.free_slots.size()));
    } else {
        const auto victim = tier_->admission_victim(index, expert, float(std::max(pairs, 1)), in_use);
        if (victim.expert < 0) { return false; }
        slot = layer.slot_of[std::size_t(victim.expert)];
        if (slot < 0 || layer.pending[std::size_t(victim.expert)]) { return false; }
        out.victim                                = victim.expert;
        layer.slot_of[std::size_t(victim.expert)] = -1;
        for (int k = 0; k < 3; ++k) { out.victim_host[k] = layer.bank(k).host[std::size_t(victim.expert)]; }
        layer.point_host(victim.expert);
        tier_->set(index, victim.expert, ExpertTier::State::Out);
    }
    // The misses copy the expert and count its bytes.
    for (int k = 0; k < 3; ++k) { out.slot[k] = layer.slot_base(slot) + layer.offset(k); }
    layer.point_slot(expert, slot);
    // The misses patch the device tables themselves; the mirror only keeps them for later uploads.
    layer.dirty                        = false;
    layer.slot_of[std::size_t(expert)] = slot;
    layer.expert_of[std::size_t(slot)] = expert;
    layer.fresh[std::size_t(expert)]   = 1;
    tier_->set(index, expert, ExpertTier::State::Held);
    ++stats_.admitted;
    return true;
}

std::array<void*, 3> ExpertCache::tables(std::size_t index) const {
    const Layer& layer = layers_.at(index);
    return {layer.banks.gate.table, layer.banks.up.table, layer.banks.down.table};
}

ExpertCache::Ranges ExpertCache::storage(std::size_t index) const {
    std::lock_guard lock(mutex_);
    const Layer& layer = layers_.at(index);
    const auto range   = [](const DeviceBuffer& buffer) {
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.p);
        return std::pair<std::uintptr_t, std::uintptr_t>{begin, begin + buffer.bytes};
    };
    const auto block = [&](const Layer::Block& lent) {
        const auto begin = reinterpret_cast<std::uintptr_t>(lent.base);
        return std::pair<std::uintptr_t, std::uintptr_t>{begin, begin + std::uint64_t(lent.count) * layer.slot_bytes};
    };
    return {range(layer.storage), range(layer.extra), block(layer.lent[0]), block(layer.lent[1])};
}

std::vector<std::uint64_t> ExpertCache::grow(std::span<const std::uint64_t> bytes_by_rank) {
    std::lock_guard lock(mutex_);
    if (!loans_.empty()) { throw std::logic_error("expert cache: grow before lending buffers"); }
    std::vector<std::uint64_t> added(bytes_by_rank.size(), 0);
    // As at construction, every growing layer of a rank gains the same number of slots.
    std::map<std::size_t, std::vector<std::size_t>> on_rank;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const Layer& layer = layers_[index];
        if (layer.extra.p == nullptr && layer.slots < layer.experts()) {
            on_rank[layer.banks.rank].push_back(index);
        }
    }
    for (const auto& [rank, members] : on_rank) {
        if (rank >= bytes_by_rank.size()) { continue; }
        std::vector<std::uint64_t> slot_bytes;
        std::vector<std::size_t> experts;
        std::vector<std::uint32_t> have;
        for (const auto index : members) {
            slot_bytes.push_back(layers_[index].slot_bytes);
            experts.push_back(layers_[index].experts());
            have.push_back(layers_[index].slots);
        }
        const auto more = share_slots(bytes_by_rank[rank], slot_bytes, experts, have,
                                      std::vector<const std::vector<std::uint64_t>*>(members.size(), nullptr),
                                      members.size());
        for (std::size_t i = 0; i < members.size(); ++i) {
            if (more[i] == 0) { continue; }
            const std::size_t index = members[i];
            Layer& layer            = layers_[index];
            RankBinding bind(device_, rank);
            layer.extra = DeviceBuffer(std::size_t(more[i]) * layer.slot_bytes);
            layer.extra.fill(0);
            layer.base_slots  = layer.slots;
            layer.extra_slots = more[i];
            for (std::uint32_t s = 0; s < more[i]; ++s) {
                layer.free_slots.insert(layer.free_slots.begin(), std::int32_t(layer.slots + s));
            }
            layer.slots += more[i];
            layer.expert_of.resize(layer.slots, -1);
            stats_.slots += more[i];
            added[rank] += std::uint64_t(more[i]) * layer.slot_bytes;
            tier_->set_free(std::uint32_t(index), std::uint32_t(layer.free_slots.size()));
        }
    }
    return added;
}

std::vector<std::uint64_t> ExpertCache::lend(std::span<const Loan> loans) {
    std::lock_guard lock(mutex_);
    if (!loans_.empty()) { throw std::logic_error("expert cache: buffers are lent once"); }
    std::vector<std::uint64_t> added(device_.size(), 0);
    std::map<std::size_t, std::vector<std::size_t>> on_rank;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        on_rank[layers_[index].banks.rank].push_back(index);
    }
    std::map<std::size_t, std::size_t> lent_by_rank;
    for (const Loan& loan : loans) {
        if (loan.rank >= device_.size() || loan.base == nullptr || loan.bytes == 0) {
            throw std::invalid_argument("expert cache: a lent buffer needs a rank and bytes");
        }
        const std::size_t k = lent_by_rank[loan.rank]++;
        if (k >= 2) { throw std::invalid_argument("expert cache: at most two lent buffers a rank"); }
        const auto it = on_rank.find(loan.rank);
        if (it == on_rank.end()) { continue; }
        // As grow shares: the same number of slots for every layer that is not full.
        std::vector<std::size_t> members;
        std::vector<std::uint64_t> slot_bytes;
        std::vector<std::size_t> experts;
        std::vector<std::uint32_t> have;
        for (const auto index : it->second) {
            const Layer& layer = layers_[index];
            if (layer.slots >= layer.experts()) { continue; }
            members.push_back(index);
            slot_bytes.push_back(layer.slot_bytes);
            experts.push_back(layer.experts());
            have.push_back(layer.slots);
        }
        const auto more = share_slots(loan.bytes, slot_bytes, experts, have,
                                      std::vector<const std::vector<std::uint64_t>*>(members.size(), nullptr),
                                      members.size());
        RankBinding bind(device_, loan.rank);
        CUDA_CHECK(cudaMemset(loan.base, 0, loan.bytes));
        auto* at = static_cast<std::byte*>(loan.base);
        for (std::size_t i = 0; i < members.size(); ++i) {
            if (more[i] == 0) { continue; }
            const std::size_t index = members[i];
            Layer& layer            = layers_[index];
            if (layer.lent[0].count + layer.lent[1].count == 0) { layer.lent_first = layer.slots; }
            layer.lent[k] = {at, more[i]};
            at += std::size_t(more[i]) * layer.slot_bytes;
            // Taken after the cache's own free slots, which survive the next prompt.
            for (std::uint32_t s = 0; s < more[i]; ++s) {
                layer.free_slots.insert(layer.free_slots.begin(), std::int32_t(layer.slots + s));
            }
            layer.slots += more[i];
            layer.expert_of.resize(layer.slots, -1);
            stats_.slots += more[i];
            added[loan.rank] += std::uint64_t(more[i]) * layer.slot_bytes;
            tier_->set_free(std::uint32_t(index), std::uint32_t(layer.free_slots.size()));
        }
    }
    loans_.assign(loans.begin(), loans.end());
    for (const auto& [rank, count] : lent_by_rank) { holding_[rank] = 1; }
    return added;
}

void ExpertCache::reclaim(std::size_t rank, cudaStream_t transfer) {
    std::lock_guard lock(mutex_);
    if (rank >= holding_.size() || !holding_[rank]) { return; }
    RankBinding bind(device_, rank);
    // A round's copies may be writing lent slots: they land first.
    CUDA_CHECK(cudaStreamSynchronize(side_[rank]));
    land();
    cudaStream_t compute = nullptr;
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        Layer& layer = layers_[index];
        if (layer.banks.rank != rank || layer.lent_end() == layer.lent_first) { continue; }
        compute = layer.banks.stream;
        const auto count = [&](std::int32_t expert) { return tier_->count(index, expert); };
        std::vector<std::int32_t> leaving;
        for (auto slot = layer.lent_first; slot < layer.lent_end(); ++slot) {
            if (layer.expert_of[slot] >= 0) { leaving.push_back(layer.expert_of[slot]); }
            layer.expert_of[slot] = -1;
        }
        std::stable_sort(leaving.begin(), leaving.end(),
                         [&](std::int32_t a, std::int32_t b) { return count(a) > count(b); });
        // The cache's own slots a leaving expert may take: the free ones, then the coldest held.
        std::vector<std::int32_t> own_free, colder;
        for (const auto slot : layer.free_slots) {
            if (std::uint32_t(slot) < layer.lent_first) { own_free.push_back(slot); }
        }
        for (std::uint32_t slot = 0; slot < layer.lent_first; ++slot) {
            const auto expert = layer.expert_of[slot];
            if (expert >= 0 && !layer.pending[std::size_t(expert)]) { colder.push_back(std::int32_t(slot)); }
        }
        std::stable_sort(colder.begin(), colder.end(), [&](std::int32_t a, std::int32_t b) {
            return count(layer.expert_of[std::size_t(a)]) < count(layer.expert_of[std::size_t(b)]);
        });
        std::size_t next = 0;
        for (const auto expert : leaving) {
            const auto from = layer.slot_of[std::size_t(expert)];
            std::int32_t to = -1;
            if (!own_free.empty()) {
                to = own_free.back();
                own_free.pop_back();
            } else if (next < colder.size() &&
                       count(layer.expert_of[std::size_t(colder[next])]) < count(expert)) {
                to                 = colder[next++];
                const auto victim  = layer.expert_of[std::size_t(to)];
                layer.slot_of[std::size_t(victim)] = -1;
                layer.point_host(victim);
                tier_->set(index, victim, ExpertTier::State::Out);
            }
            if (to < 0) {
                layer.slot_of[std::size_t(expert)] = -1;
                layer.point_host(expert);
                tier_->set(index, expert, ExpertTier::State::Out);
                continue;
            }
            CUDA_CHECK(cudaMemcpyAsync(layer.slot_base(to), layer.slot_base(from), layer.slot_bytes,
                                       cudaMemcpyDeviceToDevice, layer.banks.stream));
            layer.slot_of[std::size_t(expert)] = to;
            layer.expert_of[std::size_t(to)]   = expert;
            layer.point_slot(expert, to);
        }
        layer.free_slots = std::move(own_free);
        if (layer.dirty) { upload(layer, layer.banks.stream); }
        tier_->set_free(std::uint32_t(index), std::uint32_t(layer.free_slots.size()));
    }
    if (compute != nullptr) {
        CUDA_CHECK(cudaEventRecord(handover_[rank], compute));
        CUDA_CHECK(cudaStreamWaitEvent(transfer, handover_[rank], 0));
    }
    holding_[rank] = 0;
}

void ExpertCache::give_back(std::size_t rank, cudaStream_t transfer) {
    std::lock_guard lock(mutex_);
    if (rank >= holding_.size() || holding_[rank] || loans_.empty()) { return; }
    RankBinding bind(device_, rank);
    cudaStream_t compute = nullptr;
    for (const Layer& layer : layers_) {
        if (layer.banks.rank == rank) { compute = layer.banks.stream; }
    }
    if (compute != nullptr) {
        CUDA_CHECK(cudaEventRecord(handover_[rank], transfer));
        CUDA_CHECK(cudaStreamWaitEvent(compute, handover_[rank], 0));
        // A slot's down matrix is followed by zeros, which the wide calls' copies overwrote.
        for (const Loan& loan : loans_) {
            if (loan.rank == rank) { CUDA_CHECK(cudaMemsetAsync(loan.base, 0, loan.bytes, compute)); }
        }
        CUDA_CHECK(cudaEventRecord(handover_[rank], compute));
        CUDA_CHECK(cudaStreamWaitEvent(side_[rank], handover_[rank], 0));
    }
    // The returned slots take the most routed experts outside the cache at once, rather than over
    // the rounds of the decode that follows: their copies run on the side stream after the
    // zeroing, and a layer's experts serve once its copies have landed. The rest stay free.
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        Layer& layer = layers_[index];
        if (layer.banks.rank != rank || layer.lent_end() == layer.lent_first) { continue; }
        const auto count = [&](std::int32_t expert) { return tier_->count(index, expert); };
        std::vector<std::int32_t> missing;
        for (std::int32_t expert = 0; expert < std::int32_t(layer.experts()); ++expert) {
            if (layer.slot_of[std::size_t(expert)] < 0 &&
                count(expert) >= tier_->options().min_count) {
                missing.push_back(expert);
            }
        }
        const std::size_t take =
            std::min<std::size_t>(layer.lent_end() - layer.lent_first, missing.size());
        std::partial_sort(missing.begin(), missing.begin() + std::ptrdiff_t(take), missing.end(),
                          [&](std::int32_t a, std::int32_t b) { return count(a) > count(b); });
        cudaEvent_t done = nullptr;
        if (take != 0) {
            if (!spare_.empty()) {
                done = spare_.back();
                spare_.pop_back();
            } else {
                CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            }
        }
        auto slot = layer.lent_first;
        for (std::size_t i = 0; i < take; ++i, ++slot) {
            const std::int32_t expert           = missing[i];
            layer.expert_of[slot]               = expert;
            layer.slot_of[std::size_t(expert)]  = std::int32_t(slot);
            layer.pending[std::size_t(expert)]  = 1;
            tier_->set(index, expert, ExpertTier::State::Pending);
            for (int k = 0; k < 3; ++k) {
                const ExpertBank& bank = layer.bank(k);
                CUDA_CHECK(cudaMemcpyAsync(layer.slot_base(std::int32_t(slot)) + layer.offset(k),
                                           bank.host[std::size_t(expert)],
                                           std::size_t(bank.expert_bytes), cudaMemcpyHostToDevice,
                                           side_[rank]));
                stats_.copied_bytes += std::uint64_t(bank.expert_bytes);
            }
            ++stats_.admitted;
            copies_.push_back({index, expert, std::int32_t(slot), done});
        }
        if (done != nullptr) { CUDA_CHECK(cudaEventRecord(done, side_[rank])); }
        for (; slot < layer.lent_end(); ++slot) {
            layer.free_slots.insert(layer.free_slots.begin(), std::int32_t(slot));
        }
        tier_->set_free(std::uint32_t(index), std::uint32_t(layer.free_slots.size()));
    }
    holding_[rank] = 1;
}

} // namespace ninfer::models::qwen4_exp
