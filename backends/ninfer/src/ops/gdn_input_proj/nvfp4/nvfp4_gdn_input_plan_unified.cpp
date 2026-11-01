#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.h"

#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    Int8Grouped,
    A4,
};

Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4GdnInputRoute::A16;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family: the call
        // takes int8 at every width. This table runs grouped; the own table runs MMQ.
        return Nvfp4GdnInputRoute::Int8Grouped;
    }
    if (allows_a4(policy)) { return Nvfp4GdnInputRoute::A4; }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}


} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4GdnInputRoute widest = resolve_route(policy, max_tokens);
    if (widest == Nvfp4GdnInputRoute::A4) {
        return nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N16384K5120::kInputRows);
    }
    if (widest == Nvfp4GdnInputRoute::Int8Grouped) {
        // The width the launch addresses: below the crossover this table answers with the narrow
        // kernel, whose planes are the call's own rows; everywhere else it is the tile envelope. The
        // query asks the same question the launch does, because the two must never disagree about
        // which columns exist.
        return narrow::workspace_bytes(max_tokens, Nvfp4N16384K5120::kInputRows,
                                       Nvfp4Int8Mainloop::Grouped);
    }
    return 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4GdnInputRoute::A16:
        nvfp4_gdn_input_a16_launch(x, weight, qkv, z, stream);
        return;
    case Nvfp4GdnInputRoute::Int8Grouped:
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 int8 gdn_input_proj requires caller workspace");
        }
        nvfp4_gdn_input_int8_launch(x, weight, qkv, z, *workspace, Nvfp4Int8Mainloop::Grouped,
                                    stream);
        return;
    case Nvfp4GdnInputRoute::A4:
        break;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 gdn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace ninfer::ops::detail::unified
