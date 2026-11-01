#include "ops/linear_swiglu/q4cublas/cublas_prefill_gate.h"

#include "ninfer/ops/linear.h"
#include "ops/common/device_route.h"

#include <algorithm>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// The integer A8 mainloop's token tile: it takes exactly the positive multiples of it.
constexpr std::int32_t kIntegerTile = 128;

} // namespace

const CublasPrefillFamily* cublas_linear_add_family(std::int32_t input_rows) noexcept {
    if (input_rows == 17408) { return &kCublasDown; }
    if (input_rows == 6144) { return &kCublasOut; }
    return nullptr;
}

bool cublas_prefill_takes(const CublasPrefillFamily& family, std::int32_t tokens) {
    if (tokens <= kCublasPrefillFloorTokens) { return false; }
    const std::string_view key    = tokens % kIntegerTile == 0 ? family.integer_key : family.key;
    const std::string_view routed = device_route_schedule(key, tokens);
    if (routed == "on") { return true; }
    if (routed == "off") { return false; }
    return tokens >= kCublasPrefillMinTokens;
}

std::int32_t cublas_prefill_first_width(const CublasPrefillFamily& family, std::int32_t min_tokens,
                                        std::int32_t max_tokens) {
    const std::int32_t first = std::max(min_tokens, kCublasPrefillFloorTokens + 1);
    if (first > max_tokens) { return 0; }
    // The answer changes only where a band of either key ends or at the compiled gate, and within
    // each stretch between those it differs only between multiples of the tile and other widths,
    // so each stretch's first width of both kinds settles it.
    std::vector<std::int32_t> starts = {first, kCublasPrefillMinTokens};
    if (const auto profile = installed_device_route_profile()) {
        for (const std::string_view key : {family.key, family.integer_key}) {
            const auto found = profile->routes.find(key);
            if (found == profile->routes.end()) { continue; }
            for (const DeviceRouteBand& band : found->second) {
                if (band.last < max_tokens) { starts.push_back(band.last + 1); }
            }
        }
    }
    std::erase_if(starts, [&](std::int32_t start) { return start < first || start > max_tokens; });
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    for (std::size_t index = 0; index < starts.size(); ++index) {
        const std::int32_t begin   = starts[index];
        const std::int32_t end     = index + 1 < starts.size() ? starts[index + 1] - 1 : max_tokens;
        const std::int32_t other   = begin % kIntegerTile != 0 ? begin : begin + 1;
        const std::int32_t aligned = (begin + kIntegerTile - 1) / kIntegerTile * kIntegerTile;
        std::int32_t taken         = 0;
        if (other <= end && cublas_prefill_takes(family, other)) { taken = other; }
        if (aligned <= end && cublas_prefill_takes(family, aligned) &&
            (taken == 0 || aligned < taken)) {
            taken = aligned;
        }
        if (taken != 0) { return taken; }
    }
    return 0;
}

} // namespace ninfer::ops::detail
