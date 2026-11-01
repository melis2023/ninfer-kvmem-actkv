#include "ops/linear/common/route_table.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"
#include "core/weight.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

// The device-profile key this plan consults: the A16 chain's, or the int8 family's own. The int8
// band chooses which chain's mainloop answers (own = MMQ, unified = grouped), and keeping it
// separate means calibrating it does not have to share a band with the A16 chain choice.
constexpr const char* kRouteKeyA16  = "unified/nvfp4_linear_swiglu";
constexpr const char* kRouteKeyInt8 = "nvfp4_int8/linear_swiglu";

constexpr const char* route_key(LinearPolicy policy) {
    return policy == LinearPolicy::AllowNvfp4Int8 ? kRouteKeyInt8 : kRouteKeyA16;
}

} // namespace
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedA16Mma,
    Int8Grouped,
    FusedW4A4,
    LinearW4A4Post,
    TmaFusedW4A4,
};

constexpr std::int32_t kFusedMaxTokens = 128;

// A ragged width is worth the fused route once its partial M tile is not half of the grid. While
// the grid is two tiles the fused kernel and the public linear + silu_mul composition are close
// enough on an RTX 5090 that the sign of the difference does not survive a second session, so
// that band keeps the composition it already had.
constexpr std::int32_t kRaggedTmaFloor = 2 * kNvfp4TmaBlockM;
// The capacity function walks one step down from this floor to find the widest width the
// composition still serves. That walk is only correct while the floor is at least one whole tile:
// below that the composition owns widths above the floor as well, and one step does not reach
// them.
static_assert(kRaggedTmaFloor >= kNvfp4TmaBlockM,
              "a ragged floor inside the first M tile would leave the composition band above it "
              "out of the workspace capacity");

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family. The int8
        // launch pads its token tile to 128 while the A16 routes need no transient storage, so
        // the plan, the capacity query and the launch have to agree that int8 is what ran at
        // every width -- including the narrow widths the default cascade prefers A16 on. This
        // table runs MMQ; the unified table runs the grouped ladder.
        return Nvfp4LinearSwiGluRoute::Int8Grouped;
    }
    if (!allows_a4(policy)) { // A16Only and AllowA8
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        // Widest reachable band on a card without a native FP4 tensor core. A16 is the only
        // policy NVFP4 weights have there, so the wide fused A16 MMA owns everything above the
        // SIMT small-T band instead of refusing it; see nvfp4_linear_swiglu_a16_mma.cu. The
        // launcher is shared with the unified plan, and the kernel is sized for the 64 KiB of a
        // Turing block, so the own chain runs it wherever its table is the only one selectable.
        return Nvfp4LinearSwiGluRoute::FusedA16Mma;
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    if (tokens <= kFusedMaxTokens) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition - a whole tile, or a ragged width with enough behind it; the call site below
    // forces the matching scale layout either way.
    if (tokens >= kNvfp4TmaBlockM &&
        ((tokens % kNvfp4TmaBlockM) == 0 || tokens >= kRaggedTmaFloor)) {
        return Nvfp4LinearSwiGluRoute::TmaFusedW4A4;
    }
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
}

struct Nvfp4LinearSwiGluWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

template <class Allocator>
Nvfp4LinearSwiGluWorkspace allocate_baseline_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4LinearSwiGluWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {Nvfp4N34816K5120::kOutputRows, tokens}, 256);
    const std::size_t linear_bytes = linear_workspace_capacity_bytes(
        QType::NVFP4, Nvfp4N34816K5120::kOutputRows, Nvfp4N34816K5120::kInputRows,
        LinearPolicy::AllowA4, tokens, tokens);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

template <class Allocator>
Nvfp4W4a4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_w4a4_workspace(allocator, tokens, Nvfp4N34816K5120::kInputRows);
}

std::size_t baseline_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_baseline_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

} // namespace

static std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes_own(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // The flag is the whole decision, so the int8 launch pads to a whole 128-token tile at
        // every width and the capacity must never answer zero for it. This table's own answer below
        // the crossover is the ported row-form kernel, whose planes are the tile's -- the narrow band
        // that sizes them to the call belongs to the unified table.
        return unified::nvfp4_linear_swiglu_int8_workspace_capacity_bytes(
            min_tokens, max_tokens, Nvfp4Int8Mainloop::Mmq);
    }
    if (!allows_a4(policy) || max_tokens <= 4) {
        return 0;
    }

    std::size_t maximum = 0;
    if (min_tokens <= kFusedMaxTokens && max_tokens >= 5) {
        maximum = fused_workspace_bytes(std::min(max_tokens, kFusedMaxTokens));
    }
    // From the ragged floor upward the TMA route owns every width, so the widest it can be asked
    // for is the interval's own maximum; below the floor it owns only whole tiles, and the guard
    // then rejects an interval that contains none.
    const std::int32_t largest_fused =
        max_tokens >= kRaggedTmaFloor ? max_tokens : max_tokens - (max_tokens % kNvfp4TmaBlockM);
    if (largest_fused >= std::max(min_tokens, kNvfp4TmaBlockM)) {
        maximum = std::max(maximum, fused_workspace_bytes(largest_fused));
    }

    // The composition keeps the gap below the floor, so it no longer has to be sized for the
    // interval's maximum. Capping at the floor puts the search where the only TMA widths are whole
    // tiles, and no two consecutive widths are both whole tiles, so one step is always enough.
    std::int32_t last_baseline = std::min(max_tokens, kRaggedTmaFloor - 1);
    if (resolve_route(policy, last_baseline) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) {
        --last_baseline;
    }
    if (last_baseline >= std::max(min_tokens, kFusedMaxTokens + 1)) {
        maximum = std::max(maximum, baseline_workspace_bytes(last_baseline));
    }
    return maximum;
}

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    return capacity_by_table(
        min_tokens, max_tokens,
        [policy](std::int32_t width) { return fused_route_table(route_key(policy), width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_swiglu_workspace_capacity_bytes(
                             unified_policy(policy), first, last)
                       : nvfp4_linear_swiglu_workspace_capacity_bytes_own(policy, first, last);
        });
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    if (fused_route_table(route_key(policy), x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_linear_swiglu_dispatch(x, weight, out, unified_policy(policy), workspace,
                                              stream);
        return;
    }
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        // resolve_route puts 2..16 in this case, which is exactly the range this line's own small-T
        // kernel is instantiated for (nvfp4_linear_swiglu_small_t.cu's make_index_sequence<16-2+1>,
        // indexed by tokens - 2 with no bounds check), so the band stays on it and only T > 16 goes
        // to the shared wide-T A16 MMA below. Consequence, deliberate: inside 3..16 the two tables
        // run different kernels and fold the global weight divisor at different points, so they
        // agree exactly only on power-of-two divisors and can differ by one BF16 ulp otherwise --
        // and this band is where a speculative verify pass lands (T = K+1). Upstream's contract
        // does not freeze bitwise equality (docs/maintainer/op-development.md). T = 1, T = 2 and
        // everything above 16 share a launch and so stay byte-identical. Decode (T=1) staying on
        // this line's kernel is also what keeps the Turing build on a path it always ran.
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedA16Mma:
        unified::nvfp4_linear_swiglu_a16_mma_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::Int8Grouped:
        unified::nvfp4_linear_swiglu_int8_grouped_launch(x, weight, out, workspace,
                                                         Nvfp4Int8Mainloop::Mmq, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedW4A4: {
        auto scope                       = workspace.scope();
        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled, stream);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_linear_swiglu_w4a4_tma(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            x.ne[1], alpha, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::LinearW4A4Post:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4LinearSwiGluWorkspace scratch = allocate_baseline_workspace(workspace, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    constexpr std::int32_t kIntermediate = Nvfp4N34816K5120::kOutputRows / 2;
    silu_mul(scratch.projected.slice(0, 0, kIntermediate),
             scratch.projected.slice(0, kIntermediate, kIntermediate), out, stream);
}

bool nvfp4_linear_swiglu_int8_route_active(LinearPolicy policy, std::int32_t tokens) {
    // The policy IS the switch: --nvfp4-w4a8-int registers AllowNvfp4Int8 and no width falls back
    // to A16 while it is set on a card that runs the family. The query and the resolve above have
    // to agree, or the capacity a caller reserved and the route the launch took would differ.
    return policy == LinearPolicy::AllowNvfp4Int8 && tokens > 0;
}

// Whether this int8 call would be answered by the unified chain, the only chain carrying the fused
// [RMSNorm -> quantise] edge. Same shape as the attention-side twin: the predicate is the one
// nvfp4_linear_swiglu_dispatch asks before it forks, so a caller that skips its standalone rmsnorm
// on this answer cannot disagree with the route the projection then takes. Every width the unified
// route serves qualifies: the edge writes one activation scale per 32 columns, which is the plane
// both chains rescale against.
bool nvfp4_linear_swiglu_int8_fused_norm_eligible(LinearPolicy policy, std::int32_t tokens) {
    if (policy != LinearPolicy::AllowNvfp4Int8 || tokens <= 0) {
        return false;
    }
    return fused_route_table(route_key(policy), tokens) == LinearRouteTable::Unified;
}

} // namespace ninfer::ops::detail
