#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Hyper-connection read and write of Qwen3.8-Flash-Next (transformers Qwen4ExpTextGatedResidual):
 * the residual is a stack of `streams` copies of the hidden width, and each block reads one mixed
 * input from it and writes its output back into every stream with a per-stream weight.
 *
 * `stack` is FP32 [hidden, streams, tokens] (stream c of token t at column-major offset
 * (t * streams + c) * hidden). The weights are BF16, stored as each row's input contiguous:
 * `norm` [streams * hidden], `down` [streams * hidden, lowrank], `up` [lowrank, streams * hidden],
 * and `inject` [streams * hidden, streams] (absent for the final mixer). With n = streams and
 * xs = stack[:, :, t] flattened stream-major, the read computes in exact arithmetic
 *
 *   xn[c, d]  = xs[c, d] / sqrt(mean_d xs[c, d]^2 + eps) * (1 + norm[c * hidden + d])
 *   lo        = silu((down . xn) / n)                                         (lowrank)
 *   gate      = sigmoid(up . lo)                                              (streams * hidden)
 *   mixed[d]  = (1 / n) * sum_c gate[c * hidden + d] * xn[c, d]               BF16 [hidden, tokens]
 *   inject[c] = 2 * sigmoid((inject . xn)[c] / n)                             FP32 [streams, tokens]
 *
 * and the write updates stack[c, d] += y[d] * inject[c] for the block output y (BF16 or FP32
 * [hidden, tokens]; the MoE's output is FP32). The oracle evaluates these in FP64 from the represented inputs; `mixed` is
 * compared after its BF16 store, `inject` and the written stack as FP32. Private arithmetic,
 * including the FP32 intermediates in workspace, is implementation-defined. The shapes this
 * implements are streams 4, hidden 2560 and lowrank 320.
 */
struct HyperConnectionWeights {
    const Tensor* norm   = nullptr;
    const Tensor* down   = nullptr;
    const Tensor* up     = nullptr;
    const Tensor* inject = nullptr; // the final mixer has none
};

[[nodiscard]] std::size_t hyper_connection_read_workspace_bytes(std::int32_t streams,
                                                                std::int32_t hidden,
                                                                std::int32_t lowrank,
                                                                std::int32_t tokens);

// `inject_weights` must be null exactly when `weights.inject` is.
void hyper_connection_read(const Tensor& stack, const HyperConnectionWeights& weights, float eps,
                           WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                           cudaStream_t stream);

void hyper_connection_write(Tensor& stack, const Tensor& y, const Tensor& inject_weights,
                            cudaStream_t stream);

// The stack's start (transformers repeats the embedding into every stream): stack[c, d] = x[d] for
// each stream c, x BF16 [hidden, tokens] widened exactly to FP32.
void hyper_connection_expand(const Tensor& x, Tensor& stack, cudaStream_t stream);

} // namespace ninfer::ops
