#pragma once

// Host plan of the fast INT8-family prompt kernel (prompt_i8_fast.cuh): its CTA shape and key
// splits, and the FP32 partials a split launch publishes. The launcher and the workspace capacity
// query read the same plan, so a launch never asks for more partials than the workspace reserved.
//
// The cost model follows ashalliants' prompt_i8_fa2_plan.h (ninfer-3090 d8a85a66d, after
// Infernix's fast_tiled_plan.h, Apache-2.0), with its RTX 3090 constants. The SM count is the
// executing device's, so the wave terms follow the card; the device profile's
// "attn_prompt_split" switch turns splits off on a card where they measured slower.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// A prompt launch: warps per CTA (16 query rows each) and key splits. More than one split divides
// every row block's key pages among CTAs and merges their FP32 partial rows, so a launch whose row
// blocks alone would leave SMs idle still fills them.
struct CausalPromptI8FastPlan {
    std::int32_t warps  = 8;
    std::int32_t splits = 1;

    friend bool operator==(const CausalPromptI8FastPlan&, const CausalPromptI8FastPlan&) = default;
};

// Splits only for widths no graph-captured call requests: verification and copy rounds stay at
// most 64 columns, so their captured topology never gains the merge kernel.
inline constexpr std::int32_t kCausalPromptI8FastMinSplitWidth = 65;
inline constexpr std::int32_t kCausalPromptI8FastMaxSplits     = 8;
// Every split keeps at least this many 64-key pages.
inline constexpr std::int32_t kCausalPromptI8FastMinPagesPerSplit = 8;
// Ceiling on the FP32 partials of one split launch.
inline constexpr std::size_t kCausalPromptI8FastSplitBudgetBytes = std::size_t{64} << 20U;

// Cost model, in units of one eight-warp CTA's sweep over one visible key; a launch takes
// waves x (keys / splits) x rate.
//   * A CTA whose rows occupy at most four warps -- every four-warp CTA, and an eight-warp CTA of
//     a launch of at most 64 columns -- sweeps a key at kCausalPromptI8FastNarrowSweep.
//   * A split sweeps its keys at kCausalPromptI8FastSplitSweep and also writes and merges its
//     FP32 row, priced per (column, split, query head) at kCausalPromptI8FastSplitPerRowHead.
//   * Waves blend the whole-wave count with the fractional one (kCausalPromptI8FastWholeWaves).
// ashalliants derived these on the RTX 3090 (82 SMs, 2026-10-10) from every CTA shape and split
// count 1-8 forced at 17-2048 columns over 4K/32K/128K cached keys: against the fastest forced
// candidate the plan was 1.5 % slower on average and never slower than the best unsplit launch.
inline constexpr double kCausalPromptI8FastNarrowSweep     = 0.65;
inline constexpr double kCausalPromptI8FastSplitSweep      = 1.05;
inline constexpr double kCausalPromptI8FastSplitPerRowHead = 1.0 / 24.0;
inline constexpr double kCausalPromptI8FastWholeWaves      = 0.85;

struct CausalPromptI8FastPartials {
    Tensor rows;
    Tensor stats;
};

// The normalized FP32 D256 row of every (query head, column, split) and its (max, sum) pair.
template <class Allocator>
CausalPromptI8FastPartials
allocate_causal_prompt_i8_fast_partials(Allocator& workspace, std::int32_t q_heads,
                                        std::int32_t width, std::int32_t splits) {
    return {workspace.alloc(DType::FP32, {256, q_heads, width, splits}),
            workspace.alloc(DType::FP32, {2, q_heads, width, splits})};
}

inline std::size_t causal_prompt_i8_fast_split_bytes(std::int32_t q_heads, std::int32_t width,
                                                     std::int32_t splits) {
    if (splits <= 1) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_causal_prompt_i8_fast_partials(layout, q_heads, width, splits);
    return layout.peak_bytes(1);
}

// What a launch may split: `maximum` 1 turns splits off; `forced` (with a maximum above 1) takes
// exactly that many splits where the width and the budget allow, whatever the cost model says.
struct CausalPromptI8FastSplitPolicy {
    std::int32_t maximum = kCausalPromptI8FastMaxSplits;
    std::int32_t forced  = 0;
};

// Plan for one launch of `width` columns whose last row sweeps `visible_keys` keys on a device of
// `multiprocessors` SMs. `wide` says whether the eight-warp CTA launches in this shape set (it does
// not on Turing with INT8 values). Every candidate cost is linear in the key count, with a constant
// term (the split merge) that grows with the split count, so fewer keys never select more splits:
// the plan at an envelope's maximum bounds every launch inside it.
inline CausalPromptI8FastPlan causal_prompt_i8_fast_plan(std::int32_t q_heads, std::int32_t width,
                                                         std::uint32_t visible_keys,
                                                         std::int32_t multiprocessors, bool wide,
                                                         CausalPromptI8FastSplitPolicy policy) {
    const std::int64_t sms = multiprocessors > 0 ? multiprocessors : 1;
    const auto waves       = [&](std::int64_t ctas) {
        const double whole    = static_cast<double>((ctas + sms - 1) / sms);
        const double fraction = static_cast<double>(ctas) / static_cast<double>(sms);
        return kCausalPromptI8FastWholeWaves * whole +
               (1.0 - kCausalPromptI8FastWholeWaves) * fraction;
    };
    const double keys = static_cast<double>(visible_keys);
    const std::int64_t pages =
        (static_cast<std::int64_t>(visible_keys) + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const std::int32_t max_splits = width >= kCausalPromptI8FastMinSplitWidth ? policy.maximum : 1;
    const auto fits               = [&](std::int32_t splits) {
        return causal_prompt_i8_fast_split_bytes(q_heads, width, splits) <=
               kCausalPromptI8FastSplitBudgetBytes;
    };
    // The split launches' CTA shape: eight warps where it launches, else four.
    const std::int32_t split_warps = wide ? 8 : 4;
    const std::int64_t split_rows  = 16 * split_warps;
    const std::int64_t ctas =
        static_cast<std::int64_t>((width + split_rows - 1) / split_rows) * q_heads;
    const double rate = split_warps == 4 || width <= 64 ? kCausalPromptI8FastNarrowSweep : 1.0;

    if (policy.forced > 1 && max_splits > 1) {
        std::int32_t splits = std::min(policy.forced, max_splits);
        while (splits > 1 && !fits(splits)) { --splits; }
        if (splits > 1) { return {split_warps, splits}; }
    }

    // Up to 64 columns both CTA shapes launch the same CTAs at the same rate, and the eight-warp
    // one spreads the Q encoding over twice the warps.
    const std::int64_t narrow_ctas = static_cast<std::int64_t>((width + 63) / 64) * q_heads;
    CausalPromptI8FastPlan best{wide && width <= 64 ? 8 : 4, 1};
    double best_cost = waves(narrow_ctas) * kCausalPromptI8FastNarrowSweep * keys;
    for (std::int32_t splits = 1; splits <= max_splits; ++splits) {
        if (splits > 1 &&
            (pages < static_cast<std::int64_t>(splits) * kCausalPromptI8FastMinPagesPerSplit ||
             !fits(splits))) {
            break;
        }
        double cost = waves(ctas * splits) * rate * keys / splits;
        if (splits > 1) {
            cost = cost * kCausalPromptI8FastSplitSweep +
                   kCausalPromptI8FastSplitPerRowHead * width * splits * q_heads;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best      = {split_warps, splits};
        }
    }
    return best;
}

} // namespace ninfer::ops::detail
