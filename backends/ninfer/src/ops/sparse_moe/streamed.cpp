#include "ops/sparse_moe/streamed.h"

#include "core/layout.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kHidden = 2048;

// Small-T views and the FP32 products of the eight routed and the shared path per column.
template <class Arena> Tensor allocate_products(Arena& arena, std::int32_t tokens) {
    return arena.alloc(DType::FP32, {kHidden, 9 * tokens});
}

} // namespace

std::size_t sparse_moe_stream_workspace_bytes(std::int32_t tokens, QType routed_gate_up,
                                              QType routed_down) {
    if (tokens < 1) { throw std::invalid_argument("streamed SparseMoe: tokens must be positive"); }
    if (sparse_moe_uses_prefill(tokens, routed_gate_up, routed_down)) {
        return resolve_sparse_moe_prefill_plan(tokens, routed_gate_up, routed_down).workspace_bytes;
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_small_t_workspace(layout, tokens);
    (void)allocate_products(layout, tokens);
    return layout.peak_bytes();
}

void sparse_moe_stream(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                       ExpertDelivery& delivery, std::size_t layer, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(x.ne[1]);
    const QType gate = weights.routed_gate_up.qtype, down = weights.routed_down.qtype;
    auto scope = workspace.scope();
    if (sparse_moe_uses_prefill(tokens, gate, down)) {
        const auto plan  = resolve_sparse_moe_prefill_plan(tokens, gate, down);
        const auto views = allocate_sparse_moe_prefill_workspace(workspace, plan.slice_tokens, plan.nvfp4);
        sparse_moe_stream_prefill(x, weights, destination, plan, views, delivery, layer, stream);
        return;
    }
    if (tokens > kSparseMoeSmallTMax) {
        throw std::invalid_argument("streamed SparseMoe: no Small-T route at this token count");
    }
    const auto views = allocate_sparse_moe_small_t_workspace(workspace, tokens);
    Tensor products  = allocate_products(workspace, tokens);
    sparse_moe_stream_route(x, weights.router_shared_gate, views, stream);
    const auto* ids   = static_cast<const std::int32_t*>(views.token_ids.data);
    const auto* alpha = static_cast<const float*>(views.token_alpha.data);
    // The CPU's operands, should the delivery leave it a share (streamed_cpu.h).
    const std::array<ExpertDelivery::Input, 2> inputs{{
        {x.data, std::size_t(tokens) * kHidden * 2},
        {alpha, std::size_t(tokens) * 8 * sizeof(float)},
    }};
    const auto split = delivery.begin(layer, ids, std::uint32_t(8 * tokens), stream, inputs);
    const auto* shared = static_cast<const float*>(views.shared_scale.data);
    auto* activations  = static_cast<float*>(views.scratch.data);
    auto* values       = static_cast<float*>(products.data);
    sparse_moe_stream_experts(x, weights, delivery.tables(layer), split.cached_ids, alpha, shared,
                              activations, values, true, stream);
    delivery.await(stream);
    sparse_moe_stream_experts(x, weights, delivery.tables(layer), split.missing_ids, alpha, shared,
                              activations, values, false, stream);
    if (split.cpu != nullptr) {
        sparse_moe_stream_place(ids, split.cached_ids, split.missing_ids, split.cpu, split.products,
                                8 * tokens, values, stream);
    }
    sparse_moe_stream_merge(values, destination, stream);
}

} // namespace ninfer::ops::detail
