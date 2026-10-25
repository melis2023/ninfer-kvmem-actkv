#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// The gate activation: SiLU for Qwen3.5/3.6/3.8 Gated DeltaNet, sigmoid for Qwen3.8-Flash-Next's.
enum class GateActivation : unsigned char { Silu, Sigmoid };

/**
 * Applies RMS normalization over ne[0] and an elementwise gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r * weight[d] * act(z[d,r]),
 *
 * where act is SiLU(z) = z / (1 + exp(-z)) or sigmoid(z) = 1 / (1 + exp(-z)) as `gate` selects.
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors, `weight` is contiguous BF16 [D],
 * and eps is positive and finite. This form does not apply a unit offset to weight. Inputs and
 * output must not overlap. The oracle evaluates `ideal` naively in FP64 from the represented
 * inputs. The BF16 output is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Kernel reduction, staging,
 * and accumulator precision are implementation choices. There is no workspace or persistent state
 * side effect.
 */
void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, GateActivation gate,
                   float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
