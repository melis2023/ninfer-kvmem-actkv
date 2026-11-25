#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: cast_fp32_to_bf16
 *
 * Math / indexing:
 *   destination[i] = round_to_bf16_rne(source[i]) for every logical element i.
 *
 * Logical shapes:
 *   Source and destination have identical shapes and contiguous element order.
 *
 * Supported domain:
 *   Source is FP32 and destination is BF16. Contiguous aligned element counts divisible by four
 *   or two use vector routes; all other contiguous shapes use the scalar route.
 *
 * Numeric:
 *   Each FP32 value is converted independently with round-to-nearest-even.
 *
 * Effects:
 *   Writes the full destination. Source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no state side effect beyond writing destination.
 */
void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream);

/**
 * Op: cast_bf16_to_fp32
 *
 * Math / indexing:
 *   destination[i] = widen_to_fp32(source[i]) for every logical element i.
 *
 * Logical shapes:
 *   Source and destination have identical shapes and contiguous element order.
 *
 * Supported domain:
 *   Source is BF16 and destination is FP32. Element counts divisible by four with 8-byte
 *   aligned source and 16-byte aligned destination use the vector route; all other contiguous
 *   shapes use the scalar route.
 *
 * Numeric:
 *   Each BF16 value widens exactly; no rounding is introduced.
 *
 * Effects:
 *   Writes the full destination. Source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no state side effect beyond writing destination.
 */
void cast_bf16_to_fp32(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops
