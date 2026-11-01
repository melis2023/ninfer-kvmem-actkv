#include "kvmem/checkpoint_policy.hpp"
#include <cerrno>
#include <cstring>
#include <limits>

namespace kvmem {
CheckpointPolicy CheckpointPolicy::environment() {
    CheckpointPolicy p;
    const auto mode = std::getenv("KVMEM_CHECKPOINT_POLICY");
    if (!mode || std::strcmp(mode, "legacy") == 0) return p;
    if (std::strcmp(mode, "spread") != 0) throw std::invalid_argument("KVMEM_CHECKPOINT_POLICY must be legacy or spread");
    p.enabled = true;
    auto read = [](const char * name, uint64_t value, uint64_t maximum, bool allow_zero = false) {
        const auto raw = std::getenv(name);
        if (!raw) return value;
        if (!*raw || *raw == '-') throw std::invalid_argument(std::string(name) + " must be positive");
        char * end = nullptr; errno = 0;
        const auto result = std::strtoull(raw, &end, 10);
        if (errno || *end || (!result && !allow_zero) || result > maximum) throw std::invalid_argument(std::string(name) + " out of range");
        return uint64_t(result);
    };
    p.interval = read("KVMEM_CHECKPOINT_INTERVAL", p.interval, UINT32_MAX);
    p.min_gap = read("KVMEM_CHECKPOINT_MIN_GAP", p.min_gap, UINT32_MAX, true);
    p.count = size_t(read("KVMEM_CHECKPOINT_COUNT", p.count, 256));
    p.bytes = read("KVMEM_CHECKPOINT_MIB", p.bytes >> 20, UINT32_MAX) << 20;
    p.hot_count = size_t(read("KVMEM_CHECKPOINT_HOT", std::min(p.hot_count, p.count), p.count));
    p.host_bytes = read("KVMEM_CHECKPOINT_HOST_MIB", p.host_bytes >> 20, UINT32_MAX) << 20;
    return p;
}

std::vector<size_t> retain_checkpoint_coverage(const std::vector<CoverageInfo> & info,
        size_t count, uint64_t budget, uint64_t interval, uint64_t min_gap) {
    std::vector<size_t> keep;
    for (size_t i = 0; i < info.size(); ++i) if (info[i].identity && info[i].bytes <= budget) keep.push_back(i);
    std::stable_sort(keep.begin(), keep.end(), [&](size_t a, size_t b) { return info[a].frontier < info[b].frontier; });
    if (min_gap) {
        size_t endpoint = info.size();
        for (auto i : keep) if (info[i].purpose == CoveragePurpose::Endpoint) endpoint = i;
        size_t destination = 0;
        uint64_t previous = 0;
        bool have_previous = false;
        for (auto i : keep) {
            // Only the newest endpoint bypasses spacing. It does not move the
            // ordinary representative, so chains of close points cannot drift.
            if (i == endpoint) { keep[destination++] = i; continue; }
            if (!have_previous || info[i].frontier - previous >= min_gap) {
                keep[destination++] = i;
                previous = info[i].frontier;
                have_previous = true;
            }
        }
        keep.resize(destination);
    }
    auto fits = [&] {
        uint64_t bytes = 0;
        std::vector<const void *> seen;
        for (auto i : keep) if (std::find(seen.begin(), seen.end(), info[i].identity) == seen.end()) {
            if (info[i].bytes > budget - bytes) return false;
            bytes += info[i].bytes; seen.push_back(info[i].identity);
        }
        return keep.size() <= count;
    };
    while (!keep.empty() && !fits()) {
        const uint64_t newest = info[keep.back()].frontier;
        // Only the latest anchor and endpoint are protected, not every historical one.
        size_t anchor = info.size(), root = info.size();
        for (auto i : keep) {
            if (info[i].purpose == CoveragePurpose::Anchor || (info[i].protections & 2)) anchor = i;
            if ((info[i].purpose == CoveragePurpose::Root || (info[i].protections & 1)) && root == info.size()) root = i;
        }
        size_t victim = keep.size();
        double lowest = std::numeric_limits<double>::infinity();
        for (size_t j = 0; j + 1 < keep.size(); ++j) {
            const auto i = keep[j];
            if (i == anchor || i == root) continue;
            const uint64_t left = j ? info[keep[j-1]].frontier : 0;
            const uint64_t right = info[keep[j+1]].frontier;
            const double cost = double(right - left) / double(std::max<uint64_t>(1, std::max(interval, newest - left)));
            if (cost < lowest || (cost == lowest && (victim == keep.size() || info[i].used < info[keep[victim]].used))) {
                lowest = cost; victim = j;
            }
        }
        if (victim == keep.size()) {
            // Hard execution leases live outside this directory. Under a tiny
            // optional budget retain newest first, then anchor, then root.
            victim = 0;
            if (keep.size() > 1 && keep[0] == anchor && root != info.size())
                victim = size_t(std::find(keep.begin(), keep.end(), root) - keep.begin());
        }
        keep.erase(keep.begin() + victim);
    }
    std::sort(keep.begin(), keep.end());
    return keep;
}

CheckpointCapturePlan::CheckpointCapturePlan(uint64_t base, uint64_t stride, uint64_t root_at, uint64_t tail_at)
    : interval(stride), root(root_at), tail(tail_at), root_done(root_at <= base), tail_done(tail_at <= base) {
    next = stride <= UINT64_MAX - base ? base + stride : UINT64_MAX;
}
uint64_t CheckpointCapturePlan::cut(uint64_t begin, uint64_t end) const {
    if (!root_done && root > begin) end = std::min(end, root);
    if (!tail_done && tail > begin) end = std::min(end, tail);
    return end;
}
std::optional<CoveragePurpose> CheckpointCapturePlan::due(uint64_t frontier) const {
    if (!root_done && frontier == root) return CoveragePurpose::Root;
    if (!tail_done && frontier == tail) return CoveragePurpose::Tail;
    if (interval && frontier >= next) return CoveragePurpose::Periodic;
    return std::nullopt;
}
void CheckpointCapturePlan::saved(uint64_t frontier) {
    if (frontier >= root) root_done = true;
    if (frontier >= tail) tail_done = true;
    next = interval <= UINT64_MAX - frontier ? frontier + interval : UINT64_MAX;
}
} // namespace kvmem
