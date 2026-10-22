#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kvmem {
enum class CheckpointPurpose : uint32_t {
    None = 0, ReuseAnchor = 1, QueryReplay = 2, InputFallback = 4,
    Reconstruction = 8, Endpoint = 16, MediaBoundary = 32
};
inline CheckpointPurpose operator|(CheckpointPurpose a, CheckpointPurpose b) {
    return CheckpointPurpose(uint32_t(a) | uint32_t(b));
}
inline bool has_purpose(CheckpointPurpose value, CheckpointPurpose purpose) {
    return (uint32_t(value) & uint32_t(purpose)) != 0;
}
struct CheckpointInfo {
    uint64_t frontier = 0;
    CheckpointPurpose purposes = CheckpointPurpose::None;
    const void * storage = nullptr;
    uint64_t bytes = 0;
};
struct CheckpointLimits {
    size_t count = 4;
    uint64_t bytes = std::numeric_limits<uint64_t>::max();
};
struct PrefixRange { uint64_t begin = 0, end = 0; };

// These are evaluated/accepted frontiers, never physical cache addresses.
uint64_t prefix_restore_limit(uint64_t lcp, uint64_t live, uint64_t eval_end,
                              bool needs_logits);
uint64_t prefix_anchor(uint64_t base, uint64_t lcp, uint64_t query,
                       const std::vector<PrefixRange> & indivisible = {});
// Returns retained indices in original order; aliases count their bytes once.
std::vector<size_t> retain_checkpoints(const std::vector<CheckpointInfo> &,
                                      const CheckpointLimits &);

// Native records supply checkpoint_info() and a purposes member. Their payload,
// input proof, completion fence and restore operation remain engine-owned.
template<class Record> class CheckpointCatalog {
public:
    CheckpointCatalog() : CheckpointCatalog(CheckpointLimits{}) {}
    explicit CheckpointCatalog(CheckpointLimits limits) : limits_(limits) {
        records_.reserve(limits.count);
    }
    auto begin() noexcept { return records_.begin(); }
    auto end() noexcept { return records_.end(); }
    auto begin() const noexcept { return records_.begin(); }
    auto end() const noexcept { return records_.end(); }
    bool empty() const noexcept { return records_.empty(); }
    size_t size() const noexcept { return records_.size(); }
    size_t capacity() const noexcept { return records_.capacity(); }
    const Record & front() const { return records_.front(); }
    const Record & back() const { return records_.back(); }
    void clear() noexcept { records_.clear(); }
    void swap(CheckpointCatalog & other) noexcept {
        records_.swap(other.records_); std::swap(limits_, other.limits_);
    }
    void truncate(uint64_t frontier) {
        records_.erase(std::remove_if(records_.begin(), records_.end(), [&](const Record & r) {
            return r.checkpoint_info().frontier > frontier;
        }), records_.end());
    }
    template<class Valid> const Record * select(uint64_t limit, Valid valid) const {
        return select(limit, std::move(valid), [](const Record &, const Record &) { return false; });
    }
    template<class Valid, class Prefer> const Record * select(uint64_t limit, Valid valid, Prefer prefer) const {
        const Record * best = nullptr;
        uint64_t frontier = 0;
        for (const auto & record : records_) {
            const auto info = record.checkpoint_info();
            if (info.storage && info.frontier <= limit && valid(record) &&
                    (!best || info.frontier > frontier || (info.frontier == frontier && prefer(record, *best)))) {
                best = &record; frontier = info.frontier;
            }
        }
        return best;
    }
    // Native fixed-purpose slots can share this directory without exposing their
    // mutable StateImage representation. The owner calls prune after publication.
    Record & slot(CheckpointPurpose purpose) {
        if (purpose == CheckpointPurpose::None) throw std::invalid_argument("checkpoint slot has no purpose");
        for (auto & r : records_) if (has_purpose(r.purposes, purpose)) return r;
        if (records_.size() >= limits_.count) prune();
        if (records_.size() >= limits_.count) throw std::length_error("checkpoint slot count exceeded");
        records_.emplace_back(); records_.back().purposes = purpose;
        return records_.back();
    }
    const Record * find(CheckpointPurpose purpose) const {
        for (const auto & r : records_) if (has_purpose(r.purposes, purpose)) return &r;
        return nullptr;
    }
    void remember(Record record) {
        const auto info = record.checkpoint_info();
        if (!info.storage) throw std::invalid_argument("checkpoint has no completed native state");
        for (auto & old : records_) {
            const auto previous = old.checkpoint_info();
            if (previous.frontier == info.frontier) record.purposes = record.purposes | old.purposes;
            else if (has_purpose(record.purposes, CheckpointPurpose::ReuseAnchor))
                old.purposes = CheckpointPurpose(uint32_t(old.purposes) & ~uint32_t(CheckpointPurpose::ReuseAnchor));
        }
        records_.erase(std::remove_if(records_.begin(), records_.end(), [&](const Record & old) {
            return old.checkpoint_info().frontier >= info.frontier;
        }), records_.end());
        records_.push_back(std::move(record));
        prune();
    }
    void prune() {
        std::vector<CheckpointInfo> infos;
        for (const auto & record : records_) infos.push_back(record.checkpoint_info());
        const auto keep = retain_checkpoints(infos, limits_);
        size_t destination = 0;
        for (auto source : keep) {
            if (destination != source) records_[destination] = std::move(records_[source]);
            ++destination;
        }
        records_.resize(destination);
    }
    uint64_t bytes() const {
        uint64_t total = 0;
        std::vector<const void *> seen;
        for (const auto & r : records_) {
            const auto info = r.checkpoint_info();
            if (info.storage && std::find(seen.begin(), seen.end(), info.storage) == seen.end()) {
                if (info.bytes > std::numeric_limits<uint64_t>::max() - total)
                    throw std::overflow_error("checkpoint bytes overflow");
                total += info.bytes; seen.push_back(info.storage);
            }
        }
        return total;
    }
private:
    CheckpointLimits limits_;
    std::vector<Record> records_;
};
} // namespace kvmem
