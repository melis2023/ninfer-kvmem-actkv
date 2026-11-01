#pragma once

#include "ninfer/request.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>

namespace ninfer::serve {

// The 128-bit digest of one branched read: its whole prompt, its readout tokens and the
// alternatives its record reports. Branched reads are greedy and take no penalties, so a read's
// record depends on nothing else in its request.
struct BranchKey {
    std::uint64_t first                     = 0;
    std::uint64_t second                    = 0;
    bool operator==(const BranchKey&) const = default;
};

[[nodiscard]] BranchKey branch_key(std::span<const TokenId> prompt,
                                   std::span<const TokenId> readout,
                                   std::optional<int> top_logprobs);

// The records of the branched reads run last, by BranchKey: an exact repeat of a read answers
// from here, and the Engine runs only the reads the memo does not hold. It keeps the `capacity`
// most recently used records.
class BranchMemo {
public:
    explicit BranchMemo(std::size_t capacity) : capacity_(capacity) {}

    [[nodiscard]] std::optional<BranchResult> find(const BranchKey& key);
    void keep(const BranchKey& key, const BranchResult& record);
    [[nodiscard]] std::size_t size() const;

private:
    struct KeyHash {
        std::size_t operator()(const BranchKey& key) const noexcept {
            return static_cast<std::size_t>(key.first ^ key.second);
        }
    };

    using Entries = std::list<std::pair<BranchKey, BranchResult>>;

    std::size_t capacity_;
    mutable std::mutex mutex_;
    // Most recently used first.
    Entries entries_;
    std::unordered_map<BranchKey, Entries::iterator, KeyHash> index_;
};

} // namespace ninfer::serve
