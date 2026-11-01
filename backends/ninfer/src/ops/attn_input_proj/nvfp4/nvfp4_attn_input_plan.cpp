#include "ops/linear/common/route_table.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"
#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

// The device-profile key this plan consults: the A16 chain's, or the int8 family's own. The int8
// band chooses which chain's mainloop answers (own = MMQ, unified = grouped), and keeping it
// separate means calibrating it does not have to share a band with the A16 chain choice.
constexpr const char* kRouteKeyA16  = "unified/nvfp4_attn_input";
constexpr const char* kRouteKeyInt8 = "nvfp4_int8/attn_input";

constexpr const char* route_key(LinearPolicy policy) {
    return policy == LinearPolicy::AllowNvfp4Int8 ? kRouteKeyInt8 : kRouteKeyA16;
}

// Widest token count this line's own small-T launcher is instantiated for: its table in
// nvfp4_attn_input_small_t.cu is make_index_sequence<32 - 2 + 1>, indexed by tokens - 2 with no
// bounds check. A capability boundary of the kernel, not a measured crossover.
constexpr std::int32_t kOwnSmallTMaxTokens = 32;

} // namespace
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    Int8Grouped,
    W4A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (policy == LinearPolicy::AllowNvfp4Int8) {
        // --nvfp4-w4a8-int registers this policy and nothing else switches the family, so the
        // call takes int8 at every width and pads its last token tile. The table decides the
        // mainloop: this one runs MMQ, the unified table runs the grouped ladder.
        return Nvfp4AttnInputRoute::Int8Grouped;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::W4A4 : Nvfp4AttnInputRoute::A16;
}

} // namespace

static std::size_t nvfp4_attn_input_workspace_capacity_bytes_own(LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4AttnInputRoute widest = resolve_route(policy, max_tokens);
    if (widest == Nvfp4AttnInputRoute::Int8Grouped) {
        // Padded-width bytes: what a launch addresses, and what the unified plan asks for.
        return unified::nvfp4_int8_grouped_workspace_bytes(max_tokens, Nvfp4N14336K5120::kInputRows);
    }
    return widest == Nvfp4AttnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    return capacity_by_table(
        min_tokens, max_tokens,
        [policy](std::int32_t width) { return fused_route_table(route_key(policy), width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) {
            return table == LinearRouteTable::Unified
                       ? unified::nvfp4_attn_input_workspace_capacity_bytes(policy, first, last)
                       : nvfp4_attn_input_workspace_capacity_bytes_own(policy, first, last);
        });
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (fused_route_table(route_key(policy), x.ne[1]) == LinearRouteTable::Unified) {
        unified::nvfp4_attn_input_dispatch(x, weight, q, gate, k, v, policy, workspace, stream);
        return;
    }
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::Int8Grouped) {
        if (workspace == nullptr) {
            throw std::invalid_argument("nvfp4 int8 attn_input_proj requires caller workspace");
        }
        unified::nvfp4_attn_input_int8_launch(x, weight, q, gate, k, v, *workspace,
                                              Nvfp4Int8Mainloop::Mmq, stream);
        return;
    }
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        // This table keeps its own kernels up to the width they are instantiated for and hands
        // everything wider to the unified wide-T A16 MMA:
        //   * T = 1 stays on this line's decode kernel, which is also what keeps the Turing build
        //     on a path it always ran;
        //   * 2..32 stay on this line's small-T kernel, 32 being where its instantiations end;
        //   * above 32 one unified launch answers. The alternative -- looping this line's small-T
        //     kernel over 32-token chunks -- re-reads the whole weight once per chunk, so its cost
        //     grows with the number of chunks a width needs rather than with the width itself.
        // Numerical consequence, deliberate: inside 3..32 the two tables run genuinely different
        // kernels (this line's small-T against the unified sliced-K), and they fold the global
        // weight divisor at different points, so they agree exactly only while that divisor is a
        // power of two and can differ by one BF16 ulp otherwise. Upstream's contract does not
        // freeze bitwise equality (docs/maintainer/op-development.md); the measured worst case is
        // ~0.4 % of the reference peak. T = 1, T = 2 and everything above 32 share a launch and so
        // stay byte-identical. The widest shared-memory request on the unified ladder is 52.5 KiB,
        // under sm_75's 64 KiB block limit, with mma_bf16/ldmatrix both carrying sm_75 forms.
        const std::int32_t tokens = x.ne[1];
        if (tokens == 1) {
            nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        } else if (tokens <= kOwnSmallTMaxTokens) {
            nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        } else {
            unified::nvfp4_attn_input_a16_launch(x, weight, q, gate, k, v, stream);
        }
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

// Whether this int8 call would be answered by the unified chain, which is the only chain carrying
// the fused [RMSNorm -> quantise] edge. The predicate is the one nvfp4_attn_input_dispatch above
// asks before it forks, so a caller that skips its standalone rmsnorm on this answer cannot
// disagree with the route the projection then takes. DeviceRouteForce reaches it too, which is how
// the op tests and calibrate pin a candidate under it.
//
// Every width qualifies on the unified chain: the edge writes one activation scale per 32 columns,
// which is what both chains' mainloops consume. The own chain stays on the standalone pair, and the
// interleaved MMQ plane it builds at wide widths is refused by the fused entry itself.
bool nvfp4_attn_input_int8_fused_norm_eligible(LinearPolicy policy, std::int32_t tokens) {
    if (policy != LinearPolicy::AllowNvfp4Int8 || tokens <= 0) {
        return false;
    }
    return fused_route_table(route_key(policy), tokens) == LinearRouteTable::Unified;
}

} // namespace ninfer::ops::detail
