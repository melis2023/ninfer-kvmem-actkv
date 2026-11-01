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
 * (t * streams + c) * hidden). `norm` [streams * hidden] is BF16; the matrices are stored as each
 * row's input contiguous, in BF16 words or in ggml Q8_0 blocks (32 consecutive inputs of a row as
 * a binary16 scale d and 32 signed codes q, value d * q): `down` [streams * hidden, lowrank], `up`
 * [lowrank, streams * hidden], and `inject` [streams * hidden, streams] (absent for the final
 * mixer). With n = streams and xs = stack[:, :, t] flattened stream-major, the read computes in
 * exact arithmetic over the represented values
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
// One matrix of a hyper-connection, 16-byte aligned, in the geometry above.
struct HyperConnectionMatrix {
    enum class Format { BF16, Q8_0 };
    const void* data = nullptr; // null: absent
    Format format    = Format::BF16;
};

struct HyperConnectionWeights {
    const Tensor* norm = nullptr;
    HyperConnectionMatrix down;
    HyperConnectionMatrix up;
    HyperConnectionMatrix inject; // the final mixer has none
};

[[nodiscard]] std::size_t hyper_connection_read_workspace_bytes(std::int32_t streams,
                                                                std::int32_t hidden,
                                                                std::int32_t lowrank,
                                                                std::int32_t tokens);

// `inject_weights` must be null exactly when `weights.inject` is absent. Up to eight tokens the
// products accumulate in FP32 over FP32 activations; wider calls multiply BF16 activations by BF16
// weights (Q8_0 values rounded to BF16) on the tensor cores.
void hyper_connection_read(const Tensor& stack, const HyperConnectionWeights& weights, float eps,
                           WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                           cudaStream_t stream);

void hyper_connection_write(Tensor& stack, const Tensor& y, const Tensor& inject_weights,
                            cudaStream_t stream);

// Write y with previous_inject, then read the updated stack with weights. The written FP32
// stack is an observable rounding boundary before the read. Uses the read workspace query.
// previous_inject may be the same tensor as inject_weights: the read consumes its old values
// before publishing the new ones. All other inputs/outputs and workspace must not overlap.
void hyper_connection_write_read(Tensor& stack, const Tensor& y, const Tensor& previous_inject,
                                 const HyperConnectionWeights& weights, float eps,
                                 WorkspaceArena& workspace, Tensor& mixed, Tensor* inject_weights,
                                 cudaStream_t stream);

// The stack's start (transformers repeats the embedding into every stream): stack[c, d] = x[d] for
// each stream c, x BF16 [hidden, tokens] widened exactly to FP32.
void hyper_connection_expand(const Tensor& x, Tensor& stack, cudaStream_t stream);

// The MTP layer's start (vLLM Qwen4ExpMultiTokenPredictor, whose fused input joins the token's
// projection to every stream's with unit weight): stack[c, d] = streams[d, c] + x[d], with
// `streams` BF16 [hidden, streams, tokens] (each stream's own projection) and x BF16
// [hidden, tokens] (the token's, shared by the streams), both widened exactly to FP32 and summed in
// FP32. The oracle is the FP64 sum; the stack is compared as FP32.
void hyper_connection_expand(const Tensor& x, const Tensor& streams, Tensor& stack,
                             cudaStream_t stream);

} // namespace ninfer::ops
