#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>

namespace ninfer::detail {
// Adapted from melis2023/ninfer-kvmem-actkv b4d8096 and 591537b.
// Measurements are driver deltas, not an exact allocator ledger. Keep the donor's
// floor and confirm above-floor readings to filter WDDM residency spikes.
inline constexpr std::size_t kGraphProbeFloor = 256ULL << 20;

struct GraphProbeResult {
    std::optional<std::size_t> first;
    std::optional<std::size_t> second;
    std::optional<std::size_t> allowance;
};

template<class Measure>
GraphProbeResult calibrate_graph_allowance(Measure&& measure) {
    GraphProbeResult result;
    result.first = measure();
    if (!result.first) { return result; }
    auto measured = *result.first;
    if (measured > kGraphProbeFloor) {
        result.second = measure();
        if (result.second) { measured = std::min(measured, *result.second); }
    }
    result.allowance = std::max(measured, kGraphProbeFloor);
    return result;
}
} // namespace ninfer::detail
