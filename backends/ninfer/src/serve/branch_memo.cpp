#include "serve/branch_memo.h"

namespace ninfer::serve {

namespace {

// Two lanes, each a bijective mixer of its state with the next word folded in: splitmix64's
// finalizer and MurmurHash3's fmix64. A word that differs always changes a lane's state, and the
// lengths written ahead of each list keep concatenations apart.
struct Lanes {
    std::uint64_t first  = 0x243F6A8885A308D3ULL;
    std::uint64_t second = 0x13198A2E03707344ULL;

    void fold(std::uint64_t word) noexcept {
        std::uint64_t a = first ^ word;
        a               = (a ^ (a >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        a               = (a ^ (a >> 27U)) * 0x94D049BB133111EBULL;
        first           = a ^ (a >> 31U);
        std::uint64_t b = second ^ word;
        b               = (b ^ (b >> 33U)) * 0xFF51AFD7ED558CCDULL;
        b               = (b ^ (b >> 33U)) * 0xC4CEB9FE1A85EC53ULL;
        second          = b ^ (b >> 33U);
    }

    void fold(std::span<const TokenId> tokens) noexcept {
        fold(tokens.size());
        for (const TokenId token : tokens) { fold(static_cast<std::uint32_t>(token)); }
    }
};

} // namespace

BranchKey branch_key(std::span<const TokenId> prompt, std::span<const TokenId> readout,
                     std::optional<int> top_logprobs) {
    Lanes lanes;
    lanes.fold(prompt);
    lanes.fold(readout);
    lanes.fold(top_logprobs ? static_cast<std::uint64_t>(*top_logprobs) + 1U : 0U);
    return BranchKey{.first = lanes.first, .second = lanes.second};
}

std::optional<BranchResult> BranchMemo::find(const BranchKey& key) {
    const std::lock_guard lock(mutex_);
    const auto found = index_.find(key);
    if (found == index_.end()) { return std::nullopt; }
    entries_.splice(entries_.begin(), entries_, found->second);
    return found->second->second;
}

void BranchMemo::keep(const BranchKey& key, const BranchResult& record) {
    if (capacity_ == 0) { return; }
    const std::lock_guard lock(mutex_);
    if (const auto found = index_.find(key); found != index_.end()) {
        found->second->second = record;
        entries_.splice(entries_.begin(), entries_, found->second);
        return;
    }
    if (entries_.size() == capacity_) {
        index_.erase(entries_.back().first);
        entries_.pop_back();
    }
    entries_.emplace_front(key, record);
    index_.emplace(key, entries_.begin());
}

std::size_t BranchMemo::size() const {
    const std::lock_guard lock(mutex_);
    return entries_.size();
}

} // namespace ninfer::serve
