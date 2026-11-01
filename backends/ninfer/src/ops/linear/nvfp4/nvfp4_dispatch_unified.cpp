#include "ops/linear/nvfp4/nvfp4_dispatch_unified.h"
#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::unified {
namespace {
const std::array kShapes{&kNvfp4N14336K5120, &kNvfp4N16384K5120, &kNvfp4N34816K5120,
                         &kNvfp4N5120K6144, &kNvfp4N5120K17408, &kNvfp4DFlash2Feature, &kNvfp4DFlash2FeatureT, &kNvfp4DFlash2Qkv, &kNvfp4DFlash2AttnOut, &kNvfp4DFlash2ConvProj, &kNvfp4DFlash2Selector, &kNvfp4N8192K4096};

const Nvfp4LinearShape* find_shape(std::int32_t n, std::int32_t k) {
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return shape;
    return nullptr;
}

// A geometry is served either by its own table row or -- under AllowNvfp4Int8 only -- by the plain
// int8 route's alignment contract. The int8 route is shape-generic: its grid comes from N, its
// k-tile count from K at launch time, so a new geometry costs a table row (and its routes'
// instantiation) rather than a kernel. What it cannot do is predicate a partial tile, which is
// exactly what the alignment contract states.
const Nvfp4LinearShape* resolve_geometry(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (const Nvfp4LinearShape* shape = find_shape(n, k)) { return shape; }
    if (allows_nvfp4_int8(policy) && nvfp4_shape_tile_aligned(n, k)) { return nullptr; }
    const std::string geometry = std::to_string(n) + ',' + std::to_string(k);
    if (allows_nvfp4_int8(policy)) {
        throw std::invalid_argument("nvfp4 int8 linear: unsupported geometry [" + geometry +
                                    "]: the tile-aligned route needs N % 128 == 0 and K % 64 == 0");
    }
    throw std::invalid_argument("nvfp4 linear: no registered shape [" + geometry +
                                "]; register it, or use AllowNvfp4Int8 for a tile-aligned geometry");
}

// Whether the plain int8 route takes this call: the table row's own admission for a registered
// geometry, the alignment contract plus the named route's width rule for an unregistered one.
bool int8_admits(const Nvfp4LinearShape* shape, std::int32_t min_tokens, std::int32_t max_tokens) {
    if (shape == nullptr) {
        return min_tokens == max_tokens ? nvfp4_int8_route_admits(min_tokens)
                                       : nvfp4_int8_route_interval_admits(min_tokens, max_tokens);
    }
    return shape->int8 != nullptr && shape->uses_int8(min_tokens, max_tokens);
}

// The scratch the int8 route needs for an interval: the bytes at the widest width the interval can
// be asked for, at the width THAT launch addresses. Plain Linear's int8 route is the unified table's,
// so below the crossover it addresses the call's own columns and above it whole tiles -- rounding the
// interval down instead would under-reserve for a ragged max under a forced route.
std::size_t int8_workspace_bytes(std::int32_t max_tokens, std::int32_t k) {
    return narrow::workspace_bytes(max_tokens, k, Nvfp4Int8Mainloop::Grouped);
}
} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("nvfp4 linear workspace: unsupported policy");
    const Nvfp4LinearShape* shape = resolve_geometry(n, k, policy);
    if (allows_nvfp4_int8(policy)) {
        // A registered geometry can always answer "no scratch" for a width its route does not take,
        // because the A16 composition it falls back to needs none. An unregistered geometry has no
        // A16 composition to fall back to, so a width its int8 route does not admit is not "zero
        // bytes of scratch" -- it is a width with no route, and the plan layer must hear that here
        // rather than at the launch.
        if (!int8_admits(shape, min_tokens, max_tokens)) {
            if (shape == nullptr)
                throw std::invalid_argument(
                    std::string("nvfp4 int8 linear: an unregistered geometry has no A16 route to "
                                "fall back to, and the int8 family does not admit the interval [") +
                    std::to_string(min_tokens) + ',' + std::to_string(max_tokens) +
                    "] at geometry " + std::to_string(n) + ',' + std::to_string(k));
            return 0;
        }
        return int8_workspace_bytes(max_tokens, k);
    }
    return allows_a4(policy) && shape->uses_a4(min_tokens, max_tokens)
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("nvfp4 linear: T must be positive");
    const Nvfp4LinearShape* shape = resolve_geometry(weight.n, weight.k, policy);
    // The int8 family is registered by --nvfp4-w4a8-int at the model layer and answers every
    // positive width (the wide mainloops pad their token tile), so a policy that allows it takes it
    // at whatever width arrives. Which wide mainloop answers -- grouped here, MMQ on the own table
    // -- is the table's business, not this dispatcher's.
    if (allows_nvfp4_int8(policy) && int8_admits(shape, x.ne[1], x.ne[1])) {
        if (workspace == nullptr)
            throw std::invalid_argument("nvfp4 int8 linear requires caller workspace");
        nvfp4_linear_int8_grouped_launch(x, weight, out, *workspace, stream);
        return;
    }
    if (shape == nullptr) {
        // No table row means no A16 composition to fall back to, so a width the int8 route refused
        // here is genuinely unroutable rather than a slower answer.
        throw std::invalid_argument(
            std::string("nvfp4 int8 linear: an unregistered geometry has no A16 route to fall back "
                        "to, and the int8 family does not admit T=") +
            std::to_string(x.ne[1]) + " at geometry " + std::to_string(weight.n) + ',' +
            std::to_string(weight.k));
    }
    if (!allows_a4(policy) || !shape->uses_a4(x.ne[1], x.ne[1]))
        return shape->a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("nvfp4 A4 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    shape->a4(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail::unified
