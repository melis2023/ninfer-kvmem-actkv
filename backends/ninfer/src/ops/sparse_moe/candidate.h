#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ninfer::ops::detail {

// Internal laboratory controls. Production callers use Automatic.
enum class SparseMoeCandidate : std::uint8_t {
    Automatic, Decode, SmallT, Prefill, PrefillNarrow, PrefillWide,
};

[[nodiscard]] inline std::string_view sparse_moe_candidate_name(SparseMoeCandidate candidate) {
    switch (candidate) {
    case SparseMoeCandidate::Automatic: return "auto";
    case SparseMoeCandidate::Decode: return "decode";
    case SparseMoeCandidate::SmallT: return "small-t";
    case SparseMoeCandidate::Prefill: return "prefill";
    case SparseMoeCandidate::PrefillNarrow: return "prefill-narrow";
    case SparseMoeCandidate::PrefillWide: return "prefill-wide";
    }
    throw std::invalid_argument("sparse_moe: invalid candidate");
}

[[nodiscard]] inline SparseMoeCandidate parse_sparse_moe_candidate(std::string_view name) {
    for (const auto candidate : {SparseMoeCandidate::Automatic, SparseMoeCandidate::Decode,
                                SparseMoeCandidate::SmallT, SparseMoeCandidate::Prefill,
                                SparseMoeCandidate::PrefillNarrow, SparseMoeCandidate::PrefillWide}) {
        if (sparse_moe_candidate_name(candidate) == name) { return candidate; }
    }
    throw std::invalid_argument("sparse_moe: unknown candidate");
}

[[nodiscard]] constexpr bool sparse_moe_candidate_supports_tokens(SparseMoeCandidate candidate,
                                                                  std::int32_t tokens) {
    if (tokens < 1) { return false; }
    switch (candidate) {
    case SparseMoeCandidate::SmallT: return tokens >= 2 && tokens <= 46;
    case SparseMoeCandidate::Automatic:
    case SparseMoeCandidate::Decode:
    case SparseMoeCandidate::Prefill:
    case SparseMoeCandidate::PrefillNarrow:
    case SparseMoeCandidate::PrefillWide: return true;
    }
    return false;
}

} // namespace ninfer::ops::detail
