#include "ops/linear/common/route_table.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan_unified.h"
#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

// The device-profile key this plan consults: the A16 chain's, or the int8 family's own. The int8
// band chooses which chain's mainloop answers (own = MMQ, unified = grouped), and keeping it
// separate means calibrating it does not have to share a band with the A16 chain choice.
constexpr const char* kRouteKeyA16  = "unified/nvfp4_linear_add";
constexpr const char* kRouteKeyInt8 = "nvfp4_int8/linear_add";

constexpr const char* route_key(LinearPolicy policy) {
    return policy == LinearPolicy::AllowNvfp4Int8 ? kRouteKeyInt8 : kRouteKeyA16;
}

// Widest token count this line's own small-T launcher is instantiated for: its table in
// nvfp4_linear_add_small_t.cu is make_index_sequence<32 - 2 + 1>, indexed by tokens - 2 with no
// bounds check. A capability boundary of the kernel, not a measured crossover.
constexpr std::int32_t kOwnSmallTMaxTokens = 32;

} // namespace
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    Int8Grouped,
    W4A4,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
        allows_a8_int(policy)) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family, so the
        // call takes int8 at every width. This table runs MMQ; the unified table runs grouped.
        return Nvfp4LinearAddRoute::Int8Grouped;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    const std::int32_t first_w4a4 = input_rows == 6144 ? 7 : 8;
    return tokens >= first_w4a4 ? Nvfp4LinearAddRoute::W4A4 : Nvfp4LinearAddRoute::A16;
}

} // namespace

static std::size_t nvfp4_linear_add_workspace_capacity_bytes_own(std::int32_t output_rows,
                                                                 std::int32_t input_rows,
                                                                 LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Nvfp4LinearAddRoute widest = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (widest == Nvfp4LinearAddRoute::Int8Grouped) {
        return unified::nvfp4_int8_grouped_workspace_bytes(max_tokens, input_rows);
    }
    return widest == Nvfp4LinearAddRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    return capacity_by_table(
        min_tokens, max_tokens,
        [policy](std::int32_t width) { return fused_route_table(route_key(policy), width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_linear_add_workspace_capacity_bytes(
                             output_rows, input_rows, unified_policy(policy), first, last)
                       : nvfp4_linear_add_workspace_capacity_bytes_own(output_rows, input_rows,
                                                                       policy, first, last);
        });
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    if (fused_route_table(route_key(policy), x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_linear_add_dispatch(x, weight, residual, unified_policy(policy), workspace,
                                           stream);
        return;
    }
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::Int8Grouped) {
        unified::nvfp4_linear_add_int8_launch(x, weight, residual, workspace,
                                              Nvfp4Int8Mainloop::Mmq, stream);
        return;
    }
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::A16) {
        // This table keeps its own kernels up to the width they are instantiated for and hands
        // everything wider to one unified launch, rather than looping its small-T kernel over
        // 32-token chunks that each re-read the whole weight. See nvfp4_attn_input_plan.cpp for the
        // measured figures, the one-ulp consequence inside 3..32, and the sm_75 shared-memory note.
        const std::int32_t tokens = x.ne[1];
        if (tokens == 1) {
            nvfp4_linear_add_decode_launch(x, weight, residual, stream);
        } else if (tokens <= kOwnSmallTMaxTokens) {
            nvfp4_linear_add_small_t_launch(x, weight, residual, stream);
        } else {
            unified::nvfp4_linear_add_a16_launch(x, weight, residual, stream);
        }
        return;
    }
    auto scope                       = workspace.scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
