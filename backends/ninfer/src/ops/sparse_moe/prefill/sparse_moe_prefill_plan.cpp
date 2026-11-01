#include "core/weight.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/layout.h"
#include "ops/common/device_route.h"
#include "ops/sparse_moe/policy.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct CodecPrefillPolicy {
    std::int32_t minimum;
    std::string_view key;
};

CodecPrefillPolicy codec_prefill_policy(QType routed_gate_up, QType routed_down) noexcept {
    if (routed_gate_up == QType::Q4_G64_FP16) {
        if (routed_down == QType::Q5_G64_FP16) {
            return {kSparseMoePrefillQ4Q5Min, "sparse_moe/prefill/q4-q5"};
        }
        if (routed_down == QType::Q6_G64_FP16) {
            return {kSparseMoePrefillQ4Q6Min, "sparse_moe/prefill/q4-q6"};
        }
    }
    if (routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16) {
        return {kSparseMoePrefillQ8Q8Min, "sparse_moe/prefill/q8-q8"};
    }
    if (routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4) {
        return {kSparseMoePrefillNvfp4Min, "sparse_moe/prefill/nvfp4"};
    }
    return {0, {}};
}
} // namespace

std::int32_t sparse_moe_prefill_min_tokens(QType routed_gate_up, QType routed_down) noexcept {
    const auto policy = codec_prefill_policy(routed_gate_up, routed_down);
    if (policy.minimum == 0) { return 0; }
    const std::int32_t selected = parse_sparse_moe_frontier(device_route_schedule(policy.key, 1));
    // The NVFP4 lower frontier also protects its A16/A4 accuracy tradeoff.
    const std::int32_t minimum = routed_gate_up == QType::NVFP4 ? kSparseMoePrefillNvfp4Min : 2;
    return selected >= minimum ? selected : policy.minimum;
}

SparseMoeCandidate sparse_moe_prefill_tile_candidate(std::int32_t tokens, QType gate,
                                                      QType down, SparseMoeCandidate requested) {
    if (requested != SparseMoeCandidate::Automatic || gate != QType::Q4_G64_FP16) { return requested; }
    const auto key = down == QType::Q5_G64_FP16 ? "sparse_moe/prefill-tiles/q4-q5"
                                               : "sparse_moe/prefill-tiles/q4-q6";
    const std::string_view tile = device_route_schedule(key, tokens);
    if (tile == "narrow") { return SparseMoeCandidate::PrefillNarrow; }
    if (tile == "wide") { return SparseMoeCandidate::PrefillWide; }
    return requested;
}

bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                             QType routed_down) noexcept {
    const std::int32_t minimum = sparse_moe_prefill_min_tokens(routed_gate_up, routed_down);
    return minimum != 0 && tokens >= minimum;
}

std::size_t sparse_moe_prefill_workspace_bytes(std::int32_t max_tokens, bool nvfp4) {
    if (max_tokens < kSparseMoePrefillWorkspaceMin) {
        throw std::invalid_argument("sparse_moe prefill: max_tokens must be at least 20");
    }
    const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_prefill_workspace(layout, capacity_tokens, nvfp4);
    return layout.peak_bytes(1);
}

namespace {
SparseMoePrefillPlan resolve_prefill_plan(std::int32_t tokens, QType routed_gate_up,
                                         std::int32_t minimum, SparseMoeCandidate candidate) {
    if (minimum == 0) {
        throw std::invalid_argument("sparse_moe prefill: unsupported routed codec profile");
    }
    if (!sparse_moe_candidate_supports_tokens(candidate, tokens) ||
        (candidate == SparseMoeCandidate::Automatic && tokens < minimum)) {
        throw std::invalid_argument("sparse_moe prefill: unsupported token count");
    }
    if (candidate != SparseMoeCandidate::Automatic && candidate != SparseMoeCandidate::Prefill &&
        candidate != SparseMoeCandidate::PrefillNarrow && candidate != SparseMoeCandidate::PrefillWide) {
        throw std::invalid_argument("sparse_moe prefill: invalid candidate");
    }
    if (routed_gate_up == QType::NVFP4 &&
        (candidate == SparseMoeCandidate::PrefillNarrow || candidate == SparseMoeCandidate::PrefillWide)) {
        throw std::invalid_argument("sparse_moe prefill: NVFP4 has one fixed tile width");
    }

    // The workspace query has a floor of twenty tokens, and NVFP4 enters prefill at thirteen, so a
    // call in [13,19] has to allocate to that floor or the execution high-water mark falls short of
    // what the query promised. Slicing is unaffected: the loop still steps by the real token
    // count.
    const std::int32_t slice_tokens =
        std::max(std::min(tokens, kSparseMoePrefillSliceMax), kSparseMoePrefillWorkspaceMin);
    const bool nvfp4 = routed_gate_up == QType::NVFP4;
    return {tokens, slice_tokens, sparse_moe_prefill_workspace_bytes(slice_tokens, nvfp4), nvfp4,
            candidate};
}
} // namespace

SparseMoePrefillPlan resolve_sparse_moe_prefill_plan(std::int32_t tokens, QType routed_gate_up,
                                                     QType routed_down,
                                                     SparseMoeCandidate candidate) {
    return resolve_prefill_plan(tokens, routed_gate_up,
                               sparse_moe_prefill_min_tokens(routed_gate_up, routed_down), candidate);
}

} // namespace ninfer::ops::detail
