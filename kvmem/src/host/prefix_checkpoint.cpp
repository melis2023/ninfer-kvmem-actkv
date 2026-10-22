#include "kvmem/prefix_checkpoint.hpp"
#include <numeric>

namespace kvmem {
uint64_t prefix_restore_limit(uint64_t lcp, uint64_t live, uint64_t eval_end, bool needs_logits) {
    return std::min({lcp, live, eval_end - uint64_t(needs_logits && eval_end != 0)});
}
uint64_t prefix_anchor(uint64_t base, uint64_t lcp, uint64_t query,
                       const std::vector<PrefixRange> & indivisible) {
    if (base > query) throw std::invalid_argument("query precedes restored prefix");
    uint64_t anchor = std::clamp(lcp, base, query);
    // Floor to an input boundary; repeatedly floor overlapping native groups.
    bool changed;
    do {
        changed = false;
        for (const auto & range : indivisible) {
            if (range.begin > range.end) throw std::invalid_argument("invalid input group");
            if (range.begin < anchor && anchor < range.end) {
                if (range.begin < base) throw std::invalid_argument("restored prefix splits an input group");
                anchor = range.begin; changed = true;
            }
        }
    } while (changed);
    return anchor;
}
std::vector<size_t> retain_checkpoints(const std::vector<CheckpointInfo> & infos,
                                      const CheckpointLimits & limits) {
    std::vector<size_t> candidates(infos.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    const auto priority = [](CheckpointPurpose purpose) {
        if (has_purpose(purpose, CheckpointPurpose::ReuseAnchor)) return 3;
        if (has_purpose(purpose, CheckpointPurpose::QueryReplay | CheckpointPurpose::InputFallback |
                        CheckpointPurpose::Reconstruction | CheckpointPurpose::MediaBoundary)) return 2;
        return 1;
    };
    std::stable_sort(candidates.begin(), candidates.end(), [&](size_t a, size_t b) {
        const auto pa = priority(infos[a].purposes), pb = priority(infos[b].purposes);
        return pa != pb ? pa > pb : infos[a].frontier > infos[b].frontier;
    });
    std::vector<size_t> keep;
    std::vector<const void *> seen;
    uint64_t bytes = 0;
    for (auto index : candidates) {
        const auto & info = infos[index];
        if (!info.storage || keep.size() == limits.count) continue;
        const bool alias = std::find(seen.begin(), seen.end(), info.storage) != seen.end();
        if (!alias && info.bytes > limits.bytes - bytes) continue;
        if (!alias) { bytes += info.bytes; seen.push_back(info.storage); }
        keep.push_back(index);
    }
    std::sort(keep.begin(), keep.end());
    return keep;
}
} // namespace kvmem
