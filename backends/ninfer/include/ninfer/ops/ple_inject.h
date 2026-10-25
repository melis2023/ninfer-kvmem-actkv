#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The PLE block of Qwen3.8-Flash-Next (transformers Qwen4ExpTextPLELayer) after its two
 * projections: it gates the hashed n-gram embedding's value by its key's agreement with each
 * residual stream, adds that and a dilated causal convolution of its normalised form to the
 * stack, and advances the convolution history.
 *
 * For one sequence of `tokens` consecutive positions: `stack` FP32 [hidden, streams, tokens],
 * `key` BF16 [streams * hidden, tokens] (key_proj of the embedding, before its norm), `value`
 * BF16 [hidden, tokens], the zero-centred norm weights `norm_key`, `norm_query`, `norm_conv` BF16
 * [streams * hidden], `conv` BF16 [taps, streams * hidden] (each channel's taps contiguous, oldest
 * first) and `history` FP32 [streams * hidden, (taps - 1) * dilation] holding the normalised
 * convolution input of the preceding positions, oldest first (zeros at a sequence start). With
 * groupnorm(x, w)[c, d] = x[c, d] / sqrt(mean_d x[c, d]^2 + eps) * (1 + w[c * hidden + d]):
 *
 *   k = groupnorm(key, norm_key),  q = groupnorm(stack, norm_query)
 *   s[c]  = sum_d k[c, d] q[c, d] / sqrt(hidden),   s'[c] = sign(s) sqrt(max(|s|, 1e-6))
 *   G[c, d] = sigmoid(s'[c]) * value[d],   N = groupnorm(G, norm_conv)
 *   conv[c, d] = silu(sum_j conv[j, c * hidden + d] * N_{t - dilation * (taps - 1 - j)}[c, d])
 *   stack[c, d] += G[c, d] + conv[c, d]
 *
 * and `history` becomes the last (taps - 1) * dilation positions of N. The oracle evaluates this
 * in FP64 from the represented inputs; the stack and history are compared as FP32. Splitting a
 * sequence into consecutive calls gives the same result as one call up to FP32 rounding. The
 * implemented geometry is streams 4, hidden 2560, taps 4, dilation 3.
 */
struct PleInjectWeights {
    const Tensor* norm_key   = nullptr;
    const Tensor* norm_query = nullptr;
    const Tensor* norm_conv  = nullptr;
    const Tensor* conv       = nullptr;
};

[[nodiscard]] std::size_t ple_inject_workspace_bytes(std::int32_t tokens);

void ple_inject(Tensor& stack, const Tensor& key, const Tensor& value,
                const PleInjectWeights& weights, float eps, Tensor& history,
                WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
