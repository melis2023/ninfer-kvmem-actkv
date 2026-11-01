#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// The most tokens one readout_logprobs() call reads.
inline constexpr std::int32_t kReadoutLogprobsMaximum = 32;

/**
 * Op: Readout token log-probabilities
 *
 * The log probabilities of a caller-chosen set of tokens under one logits column, the typed
 * decision readout: a request names the tokens its answer can be and reads their probabilities
 * at the first generated position, without sampling them.
 *
 * Math / indexing:
 *   Let l[r] be the exact real value represented by logits[r]. For every k in [0,count),
 *
 *     ideal[k] = l[token_ids[k]] - log(sum_{r=0..valid_rows-1} exp(l[r])).
 *
 * Logical shapes:
 *   logits is [physical_rows] or [physical_rows,1], token_ids is [capacity] and output is
 *   [capacity], with 1<=count<=capacity<=kReadoutLogprobsMaximum and
 *   1<=valid_rows<=physical_rows. The first `count` values of token_ids are in [0,valid_rows);
 *   the rest of token_ids and output are not read or written. Physical rows
 *   [valid_rows,physical_rows) take no part in the denominator.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, token_ids is contiguous I32, and output is contiguous
 *   FP32. Storage has its dtype's natural alignment.
 *
 * Numeric:
 *   output is the FP32 numerical approximation of ideal. Reduction association and private
 *   accumulator precision are implementation choices; the independent oracle evaluates the full
 *   formula in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes output[0..count) and preserves both inputs. Output must not overlap logits or
 *   token_ids.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void readout_logprobs(const Tensor& logits, const Tensor& token_ids, std::int32_t count,
                      std::int32_t valid_rows, Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops
