#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::ops::detail {

// A frontier may consume Small-T through T=46, but cannot extend its compiled domain.
// Zero means that the schedule is absent or invalid and the compiled policy applies.
[[nodiscard]] constexpr std::int32_t parse_sparse_moe_frontier(std::string_view value) noexcept {
    if (value.empty() || value.size() > 2 || value.front() == '0') { return 0; }
    std::int32_t result = 0;
    for (const char digit : value) {
        if (digit < '0' || digit > '9') { return 0; }
        result = 10 * result + digit - '0';
    }
    return result >= 2 && result <= 47 ? result : 0;
}

[[nodiscard]] constexpr bool sparse_moe_policy_candidate_valid(std::string_view key,
                                                               std::string_view schedule) noexcept {
    if (key == "sparse_moe/prefill/q8-q8") { return parse_sparse_moe_frontier(schedule) != 0; }
    if (key == "sparse_moe/prefill-tiles/q4-q5" || key == "sparse_moe/prefill-tiles/q4-q6") {
        return schedule == "narrow" || schedule == "wide";
    }
    return false;
}

} // namespace ninfer::ops::detail
