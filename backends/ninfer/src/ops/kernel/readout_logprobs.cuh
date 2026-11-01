#pragma once

// Implements: include/ninfer/ops/readout_logprobs.h
// Match: contiguous BF16 [physical_rows], I32 [capacity], and FP32 [capacity].
// Algorithm assumptions: one 1024-thread CTA performs a single-pass online logsumexp over the
// column, then thread k writes token k's log probability.

#include "ops/common/warp.cuh"
#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kReadoutLogprobsBlock = 1024;

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void readout_logprobs_kernel(const __nv_bfloat16* logits, const std::int32_t* token_ids,
                                 float* output, std::int32_t count, std::int32_t valid_rows) {
    float local_max = -CUDART_INF_F;
    float local_sum = 0.0f;
    // Requires finite logits (see readout_logprobs.h): -inf before any finite value in a
    // thread's subsequence NaN-poisons local_sum through expf(-inf - -inf).
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value   = __bfloat162float(logits[row]);
        const float new_max = fmaxf(local_max, value);
        if (new_max > local_max) {
            local_sum *= expf(local_max - new_max);
        }
        local_max = new_max;
        local_sum += expf(value - local_max);
    }
    const float maximum = target_logprobs_block_max<BlockSize>(local_max);

    local_sum *= expf(local_max - maximum);
    __shared__ float warp_sums[BlockSize / kWarpSize];
    // block_reduce_sum leaves the total in lane 0 of warp 0; every reader (k < count <= 32)
    // sits in warp 0, so one shuffle broadcasts it there.
    const float sum        = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    const float lse        = maximum + logf(__shfl_sync(kFullWarpMask, sum, 0));
    const auto k           = static_cast<std::int32_t>(threadIdx.x);
    if (k < count) {
        output[k] = __bfloat162float(logits[token_ids[k]]) - lse;
    }
}

} // namespace ninfer::ops
