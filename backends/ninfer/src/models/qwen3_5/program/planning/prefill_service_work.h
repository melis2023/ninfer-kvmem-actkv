#pragma once

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// The initial proposal must leave room for the sampled Begin token and correction.
[[nodiscard]] inline std::uint32_t initial_mtp_prefill_extent(
    std::uint32_t draft_window, std::uint32_t output_tokens,
    std::uint32_t capacity, std::uint32_t prompt_tokens) noexcept {
    return std::min({draft_window, output_tokens > 1 ? output_tokens - 2 : 0U,
                     capacity > prompt_tokens ? capacity - prompt_tokens - 1 : 0U});
}

// Count worker-visible prefill steps, not internal forward calls. KVMem resumes on
// the canonical chunk grid; its last step also leaves room in R for the MTP lead.
// Capture-separated intervals use the same grid. An empty suffix is accounted for
// by the caller's single Begin step.
[[nodiscard]] inline std::uint64_t prefill_service_units(
    std::uint32_t begin, std::uint32_t end, std::uint32_t chunk,
    std::uint32_t window_reserve = 0, std::uint32_t mtp_extent = 0) noexcept {
    if (begin >= end) return 0;
    if (window_reserve == 0) return 1ULL + (end - begin - 1ULL) / chunk;
    const std::uint32_t last_begin = std::max(begin, ((end - 1U) / chunk) * chunk);
    const std::uint32_t lead = mtp_extent == 0 ? 0U : mtp_extent - 1U;
    const std::uint32_t final_limit = window_reserve - lead;
    return 1ULL + (end - 1U) / chunk - begin / chunk +
           (end - last_begin - 1ULL) / final_limit;
}

} // namespace ninfer::models::qwen3_5::detail
