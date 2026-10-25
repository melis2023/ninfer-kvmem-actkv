#pragma once

#include "ninfer/types.h"

namespace ninfer::runtime {

// Normalize a valid copy width to the startup-fixed batch geometry. Invalid widths and
// concurrency are left intact for validation; they must not become valid by normalization.
[[nodiscard]] inline std::uint32_t concurrent_ngram_draft_tokens(
    std::uint32_t requested, std::uint32_t max_concurrency) noexcept {
    if (max_concurrency > 1 && max_concurrency <= kMaximumConcurrency &&
        requested > 15 && requested <= 63) {
        return 15;
    }
    return requested;
}

} // namespace ninfer::runtime
