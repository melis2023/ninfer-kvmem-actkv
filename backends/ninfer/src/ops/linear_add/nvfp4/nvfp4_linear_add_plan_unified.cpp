#include "core/weight.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan_unified.h"

#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    Int8Grouped,
    A4,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4LinearAddRoute::A16;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family: the call
        // takes int8 at every width. This table runs grouped; the own table runs MMQ.
        return Nvfp4LinearAddRoute::Int8Grouped;
    }
    if (!allows_a4(policy)) { throw std::invalid_argument("nvfp4 linear_add: unsupported policy"); }
    // With the three-stage A4 tile, K=6144 is ahead of A16 from 8 tokens too: 21.1 us against
    // 23.1-31.3 us at 8-16 tokens on RTX 5090.
    const std::int32_t first_a4 = 8;
    return tokens >= first_a4 ? Nvfp4LinearAddRoute::A4 : Nvfp4LinearAddRoute::A16;
}


} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Nvfp4LinearAddRoute widest = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (widest == Nvfp4LinearAddRoute::A4) {
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
    if (widest == Nvfp4LinearAddRoute::Int8Grouped) {
        // The width the launch addresses: below the crossover this table answers with the narrow
        // kernel, whose planes are the call's own rows; everywhere else it is the tile envelope.
        return narrow::workspace_bytes(max_tokens, input_rows, Nvfp4Int8Mainloop::Grouped);
    }
    return 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    switch (resolve_route(weight.n, weight.k, policy, x.ne[1])) {
    case Nvfp4LinearAddRoute::A16:
        nvfp4_linear_add_a16_launch(x, weight, residual, stream);
        return;
    case Nvfp4LinearAddRoute::Int8Grouped:
        nvfp4_linear_add_int8_launch(x, weight, residual, workspace, Nvfp4Int8Mainloop::Grouped,
                                     stream);
        return;
    case Nvfp4LinearAddRoute::A4:
        break;
    }
    auto scope                     = workspace.scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail::unified
