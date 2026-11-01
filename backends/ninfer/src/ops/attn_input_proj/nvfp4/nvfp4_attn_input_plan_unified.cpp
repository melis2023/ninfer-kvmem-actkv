#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"

#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    Int8Grouped,
    A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family: the call
        // takes int8 at every width and pads its last token tile. This table runs the grouped
        // mainloop; the own table runs MMQ.
        return Nvfp4AttnInputRoute::Int8Grouped;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::A4 : Nvfp4AttnInputRoute::A16;
}


} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4AttnInputRoute widest = resolve_route(policy, max_tokens);
    if (widest == Nvfp4AttnInputRoute::A4) {
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows);
    }
    if (widest == Nvfp4AttnInputRoute::Int8Grouped) {
        // The width the launch addresses, which this table's answer below the crossover is the narrow
        // kernel's call width and everywhere else the tile envelope -- the same question the launch
        // asks, because a plane sized for columns the quantiser never filled is read back uninitialised.
        return narrow::workspace_bytes(max_tokens, Nvfp4N14336K5120::kInputRows,
                                       Nvfp4Int8Mainloop::Grouped);
    }
    return 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4AttnInputRoute::A16:
        nvfp4_attn_input_a16_launch(x, weight, q, gate, k, v, stream);
        return;
    case Nvfp4AttnInputRoute::Int8Grouped:
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 int8 attn_input_proj requires caller workspace");
        }
        nvfp4_attn_input_int8_launch(x, weight, q, gate, k, v, *workspace,
                                     Nvfp4Int8Mainloop::Grouped, stream);
        return;
    case Nvfp4AttnInputRoute::A4:
        break;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 attn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail::unified
