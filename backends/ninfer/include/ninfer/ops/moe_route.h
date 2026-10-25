#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The router of Qwen3.8-Flash-Next's MoE (512 experts, or the 256 an expert-pruned release keeps)
 * with its shared-expert gate, for `tokens` columns of the block input m (BF16 or FP32
 * [2560, tokens]; FP32 keeps the activation unrounded, which the selection is sensitive to):
 *
 *   logits = router . m                       router BF16 [2560, E], 10 <= E <= 512
 *   p      = softmax(logits)                  over the E experts
 *   ids    = the 10 largest p (ties to the lower expert index), in decreasing p
 *   weights[j] = p[ids[j]] / sum_k p[ids[k]]
 *   shared = sigmoid(shared_gate . m)         shared_gate BF16 [2560]
 *
 * Outputs: `ids` I32 [10, tokens], `weights` FP32 [10, tokens], `shared` FP32 [tokens]. The oracle
 * evaluates in FP64 from the represented inputs; ids must match it up to experts whose FP64
 * logits tie the tenth within 1e-5, weights and shared are compared as FP32. The logits pass
 * through FP32 workspace.
 */
[[nodiscard]] std::size_t moe_route_workspace_bytes(std::int32_t tokens, std::int32_t experts);

void moe_route(const Tensor& m, const Tensor& router, const Tensor& shared_gate,
               WorkspaceArena& workspace, Tensor& ids, Tensor& weights, Tensor& shared,
               cudaStream_t stream);

} // namespace ninfer::ops
