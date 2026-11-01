#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops {
// Sum represented BF16 values over token rows, producing FP32 statistics.
// input is contiguous [channels,T], origin is an I32 scalar containing the
// nonnegative original position of row zero, range is I32 [begin,end).
// Only rows whose original position lies in range contribute. If block_tokens
// is positive, output bucket j corresponds to floor(origin/block_tokens)+j;
// output is [channels,ceil((T+block_tokens-1)/block_tokens)], including zero
// padding buckets. block_tokens==0 produces one sum over the entire range.
// Outputs are overwritten, not accumulated. All buffers must be disjoint;
// origin/range contents can change between graph replays. No persistent state,
// frontier update, normalization, quantization, allocation or host copy occurs.
// Numerical reference: naive FP64 sums of the represented BF16 inputs.
void token_sums(const Tensor& input, const Tensor& origin, const Tensor& range,
                std::uint32_t block_tokens, Tensor& output, cudaStream_t stream);

// Copy the represented BF16 rows in the intersection of [origin,origin+T) and
// range into contiguous output [channels,capacity], starting at row zero. The
// intersection must fit capacity; unused output rows are zeroed. No conversion
// or token reduction occurs. Input, origin and range have the token_sums layout;
// every buffer is contiguous and disjoint. Origin/range may change on replay.
// The reference is an exact copy of the selected represented input bits.
void capture_query_rows(const Tensor& input, const Tensor& origin, const Tensor& range,
                        Tensor& output, cudaStream_t stream);

// Add that intersection, in increasing token order, to an FP32 [channels,1]
// accumulator. The caller seeds it with the accepted Query prefix before each
// invocation (including graph replay); this preserves the sequential FP32 sum
// across chunk boundaries and interleaved sequences. Non-Query rows contribute
// nothing. All buffers are disjoint and the Op owns no allocation or state.
// Numerical reference: FP64 sum including the represented FP32 seed. The
// chronological FP32 additions are also a state boundary across chunk splits.
void accumulate_query_sum(const Tensor& input, const Tensor& origin, const Tensor& range,
                          Tensor& output, cudaStream_t stream);
} // namespace ninfer::ops
