#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"

#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedA16Mma,
    Int8Grouped,
    FusedA4,
    TmaFusedA4,
};

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family, so the
        // narrow-width A16 preferences below do not apply to an int8 call: the launch pads its
        // token tile to 128 while those routes need no transient storage, and a plan that chose
        // one of them would leave the capacity query and the launch disagreeing about which route
        // ran -- a buffer overrun, not an accounting error. The swiglu exact-workspace high-water
        // check is what caught that once. This table runs the grouped ladder; the own table runs
        // MMQ.
        return Nvfp4LinearSwiGluRoute::Int8Grouped;
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        // Above the SIMT band the A16-only card needs the fused wide MMA; see
        // nvfp4_linear_swiglu_a16_mma.cu. There is no NVFP4 tensor-core route here to prefer.
        return Nvfp4LinearSwiGluRoute::FusedA16Mma;
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition; the call site below forces the matching scale layout.
    if (tokens >= 256) { return Nvfp4LinearSwiGluRoute::TmaFusedA4; }
    return Nvfp4LinearSwiGluRoute::FusedA4;
}

template <class Allocator>
Nvfp4A4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_a4_workspace(allocator, tokens, Nvfp4N34816K5120::kInputRows);
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    // The `max_tokens <= 4` short-circuit is the same narrow-width A16 preference resolve_route
    // carries, restated in bytes: at those widths the A16 routes need no transient storage. An
    // int8 call never takes them -- the flag is the whole decision -- and it always takes planes out
    // of the caller's arena, so it must never be answered with "zero bytes"; the high-water check is
    // what proves the two sides agree.
    const bool int8_policy = policy == LinearPolicy::AllowNvfp4Int8;
    if (!int8_policy &&
        ((policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) || max_tokens <= 4)) {
        return 0;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // The route's capacity function reports exact bytes for a single T and the widest reach of an
        // interval otherwise, and it is asked with THIS table because the two chains size their
        // planes differently below the crossover: this one runs the narrow kernel, whose planes are
        // the call's width.
        return nvfp4_linear_swiglu_int8_workspace_capacity_bytes(min_tokens, max_tokens,
                                                                 Nvfp4Int8Mainloop::Grouped);
    }

    return fused_workspace_bytes(max_tokens);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedA16Mma:
        nvfp4_linear_swiglu_a16_mma_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::Int8Grouped:
        nvfp4_linear_swiglu_int8_grouped_launch(x, weight, out, workspace,
                                                Nvfp4Int8Mainloop::Grouped, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedA4:
        nvfp4_linear_swiglu_a4_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedA4: {
        auto scope                     = workspace.scope();
        const Nvfp4A4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled256, stream);
        launch_nvfp4_linear_swiglu_a4_tma(
            nvfp4_a4_operands(weight, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled256),
            static_cast<__nv_bfloat16*>(out.data), stream);
        return;
    }
    }
    throw std::logic_error("unreachable NVFP4 SwiGLU route");
}
} // namespace ninfer::ops::detail::unified
