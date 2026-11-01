#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 or FP32 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// The two highest finite BF16 vocabulary logits per column, ordered by descending value then
// ascending token id. Both outputs are contiguous I32 [T]; second excludes the token in first.
// Requires 2 <= valid_rows <= physical_rows. Inputs and outputs must not overlap. No workspace.
void argmax_top2(const Tensor& logits, Tensor& first, Tensor& second,
                 std::int32_t valid_rows, cudaStream_t stream);

} // namespace ninfer::ops
