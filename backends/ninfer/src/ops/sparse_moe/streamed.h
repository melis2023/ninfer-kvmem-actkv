#pragma once

#include "ninfer/ops/sparse_moe.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "core/expert_delivery.h"

namespace ninfer::ops::detail {

// SparseMoe over delivered routed experts: tables hold [gate/up code, high, scale, divisor,
// down ...][256] per layer. One delivery call per slice: below the prefill frontier the Small-T
// arithmetic runs the cached pairs while the rest arrive, then the missing ones; at and above it
// the grouped prefill waits for the whole slice's experts (one call per bounded native slice).
inline constexpr std::int32_t kSparseMoeStreamPairs = 8 * kSparseMoePrefillSliceMax;

// Workspace of one streamed call of `tokens` columns.
[[nodiscard]] std::size_t sparse_moe_stream_workspace_bytes(std::int32_t tokens, QType routed_gate_up,
                                                            QType routed_down);
void sparse_moe_stream(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                       ExpertDelivery& delivery, std::size_t layer, WorkspaceArena& workspace,
                       cudaStream_t stream);

void sparse_moe_stream_prefill(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, ExpertDelivery& delivery,
                               std::size_t layer, cudaStream_t stream);

void sparse_moe_stream_route(const Tensor& x, const Weight& router,
                              const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);
void sparse_moe_stream_decode_route(const Tensor& x, const Weight& router,
                                     const SparseMoeSmallTWorkspace& workspace,
                                     cudaStream_t stream);
void sparse_moe_stream_experts(const Tensor& x, const SparseMoeWeights& weights,
                                const void* const* tables, const std::int32_t* ids,
                                const float* alpha, const float* shared_scale,
                                float* activations, float* products, bool shared,
                                cudaStream_t stream);
// After the missing pairs: the CPU's product rows of the call (ExpertDelivery::Call::products,
// when *cpu is nonzero) into their pairs' slots of `products`.
void sparse_moe_stream_place(const std::int32_t* ids, const std::int32_t* cached,
                             const std::int32_t* missing, const std::int32_t* cpu,
                             const void* rows, std::int32_t pairs, float* products,
                             cudaStream_t stream);
void sparse_moe_stream_merge(const float* products, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
