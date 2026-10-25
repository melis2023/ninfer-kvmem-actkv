#pragma once

// Implements: include/ninfer/ops/logprob_topk.h
// Match: contiguous BF16 [physical_rows,columns,batch], `batch` device SamplingConfig records, an
// optional I32 [columns-1,batch] draft overlay, and contiguous I32/FP32 outputs.
// Algorithm assumptions: one pass over the vocabulary split across CTAs per row: every thread
// keeps an online (maximum, scaled sum) pair and a sorted list of its best kLogprobTopK tokens,
// the CTA merges them into one partial, and one CTA per row combines the partials.

#include "ninfer/ops/logprob_topk.h"

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kLogprobTopkBlock = 128;
// Partial CTAs per row: a lone row spreads over the device, several rows share it.
inline constexpr int kLogprobTopkSplitSingle = 128;
inline constexpr int kLogprobTopkSplitMulti  = 64;

[[nodiscard]] constexpr int logprob_topk_split(std::int32_t rows) noexcept {
    return rows == 1 ? kLogprobTopkSplitSingle : kLogprobTopkSplitMulti;
}

struct LogprobCandidate {
    float value;
    std::int32_t id;
};

// Inserts (value,id) into a list sorted by sampling_better(), dropping its last entry.
__device__ __forceinline__ void logprob_insert(LogprobCandidate (&top)[kLogprobTopK], float value,
                                               std::int32_t id) {
    if (!sampling_better(value, id, top[kLogprobTopK - 1].value, top[kLogprobTopK - 1].id)) {
        return;
    }
    int position = kLogprobTopK - 1;
    while (position > 0 &&
           sampling_better(value, id, top[position - 1].value, top[position - 1].id)) {
        top[position] = top[position - 1];
        --position;
    }
    top[position] = LogprobCandidate{value, id};
}

// Adds exp(value - maximum) to an online (maximum, sum) pair.
__device__ __forceinline__ void logprob_accumulate(float& maximum, float& sum, float value) {
    if (value <= maximum) {
        sum += __expf(value - maximum);
        return;
    }
    sum     = sum * __expf(maximum - value) + 1.0f;
    maximum = value;
}

// Merges another online (maximum, sum) pair into this one.
__device__ __forceinline__ void logprob_merge(float& maximum, float& sum, float other_maximum,
                                              float other_sum) {
    if (other_maximum == -CUDART_INF_F) { return; }
    if (maximum == -CUDART_INF_F) {
        maximum = other_maximum;
        sum     = other_sum;
        return;
    }
    if (other_maximum > maximum) {
        sum     = sum * __expf(maximum - other_maximum) + other_sum;
        maximum = other_maximum;
    } else {
        sum += other_sum * __expf(other_maximum - maximum);
    }
}

// Shared state of one CTA's reduction.
struct LogprobBlockStorage {
    LogprobCandidate candidates[kLogprobTopkBlock * kLogprobTopK];
    float maxima[kLogprobTopkBlock];
    float sums[kLogprobTopkBlock];
};

// Reduces every thread's (maximum, sum) pair and sorted list into thread 0's: the pair into
// storage.maxima[0]/sums[0], the list into candidates[0..kLogprobTopK). All threads must call.
__device__ __forceinline__ void logprob_block_reduce(LogprobBlockStorage& storage,
                                                     const LogprobCandidate (&top)[kLogprobTopK],
                                                     float maximum, float sum) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) { storage.candidates[tid * kLogprobTopK + k] = top[k]; }
    storage.maxima[tid] = maximum;
    storage.sums[tid]   = sum;
    __syncthreads();
    for (int step = kLogprobTopkBlock / 2; step > 0; step >>= 1) {
        if (tid < step) {
            float merged_maximum = storage.maxima[tid];
            float merged_sum     = storage.sums[tid];
            logprob_merge(merged_maximum, merged_sum, storage.maxima[tid + step],
                          storage.sums[tid + step]);
            storage.maxima[tid] = merged_maximum;
            storage.sums[tid]   = merged_sum;

            const LogprobCandidate* left  = storage.candidates + tid * kLogprobTopK;
            const LogprobCandidate* right = storage.candidates + (tid + step) * kLogprobTopK;
            LogprobCandidate out[kLogprobTopK];
            int i = 0;
            int j = 0;
#pragma unroll
            for (int k = 0; k < kLogprobTopK; ++k) {
                const bool take_left =
                    sampling_better(left[i].value, left[i].id, right[j].value, right[j].id);
                out[k] = take_left ? left[i] : right[j];
                i += take_left ? 1 : 0;
                j += take_left ? 0 : 1;
            }
#pragma unroll
            for (int k = 0; k < kLogprobTopK; ++k) {
                storage.candidates[tid * kLogprobTopK + k] = out[k];
            }
        }
        __syncthreads();
    }
}

// Partial for vocabulary chunk blockIdx.x of row blockIdx.y = column + batch_row * columns.
__global__ void __launch_bounds__(kLogprobTopkBlock)
    logprob_topk_partials_kernel(const __nv_bfloat16* __restrict__ logits,
                                 const SamplingConfig* __restrict__ configs,
                                 const std::int32_t* __restrict__ drafts, std::int32_t columns,
                                 std::int32_t token_domain, std::int64_t row_stride,
                                 std::int32_t split, const std::int32_t* __restrict__ active,
                                 float* __restrict__ partial_maxima,
                                 float* __restrict__ partial_sums,
                                 LogprobCandidate* __restrict__ partial_top) {
    if (*active == 0) { return; }
    __shared__ LogprobBlockStorage storage;

    const int row       = static_cast<int>(blockIdx.y);
    const int column    = row % columns;
    const int batch_row = row / columns;
    const int chunk     = (token_domain + split - 1) / split;
    const int begin     = static_cast<int>(blockIdx.x) * chunk;
    const int end       = min(begin + chunk, token_domain);

    const SamplingConfig config  = configs[batch_row];
    const float inverse_t        = 1.0f / (config.temperature > 0.0f ? config.temperature : 1.0f);
    const std::int32_t* overlay  = drafts != nullptr && columns > 1
                                       ? drafts + static_cast<std::int64_t>(batch_row) * (columns - 1)
                                       : nullptr;
    const __nv_bfloat16* source  = logits + static_cast<std::int64_t>(row) * row_stride;

    LogprobCandidate top[kLogprobTopK];
#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) { top[k] = LogprobCandidate{-CUDART_INF_F, INT_MAX}; }
    float maximum = -CUDART_INF_F;
    float sum     = 0.0f;
    for (int v = begin + static_cast<int>(threadIdx.x); v < end; v += kLogprobTopkBlock) {
        const float scaled =
            sampling_adjusted_logit(__bfloat162float(source[v]), v, config, overlay, column) *
            inverse_t;
        // A masked or diverged token has no probability to report.
        if (!isfinite(scaled)) { continue; }
        logprob_accumulate(maximum, sum, scaled);
        logprob_insert(top, scaled, v);
    }
    logprob_block_reduce(storage, top, maximum, sum);

    if (threadIdx.x == 0) {
        const std::int64_t partial = static_cast<std::int64_t>(row) * split + blockIdx.x;
        partial_maxima[partial]    = storage.maxima[0];
        partial_sums[partial]      = storage.sums[0];
#pragma unroll
        for (int k = 0; k < kLogprobTopK; ++k) {
            partial_top[partial * kLogprobTopK + k] = storage.candidates[k];
        }
    }
}

// Combines row blockIdx.x's partials into its log-sum-exp and reported tokens.
__global__ void __launch_bounds__(kLogprobTopkBlock)
    logprob_topk_combine_kernel(const float* __restrict__ partial_maxima,
                                const float* __restrict__ partial_sums,
                                const LogprobCandidate* __restrict__ partial_top,
                                std::int32_t split, const std::int32_t* __restrict__ active,
                                std::int32_t* __restrict__ top_ids, float* __restrict__ top_values,
                                float* __restrict__ lse) {
    if (*active == 0) { return; }
    __shared__ LogprobBlockStorage storage;

    const int row             = static_cast<int>(blockIdx.x);
    const std::int64_t first  = static_cast<std::int64_t>(row) * split;
    LogprobCandidate top[kLogprobTopK];
#pragma unroll
    for (int k = 0; k < kLogprobTopK; ++k) { top[k] = LogprobCandidate{-CUDART_INF_F, INT_MAX}; }
    float maximum = -CUDART_INF_F;
    float sum     = 0.0f;
    for (int s = static_cast<int>(threadIdx.x); s < split; s += kLogprobTopkBlock) {
        logprob_merge(maximum, sum, partial_maxima[first + s], partial_sums[first + s]);
    }
    for (int i = static_cast<int>(threadIdx.x); i < split * kLogprobTopK; i += kLogprobTopkBlock) {
        const LogprobCandidate candidate = partial_top[first * kLogprobTopK + i];
        if (candidate.id != INT_MAX) { logprob_insert(top, candidate.value, candidate.id); }
    }
    logprob_block_reduce(storage, top, maximum, sum);

    if (threadIdx.x == 0) {
        const float row_lse = storage.maxima[0] + logf(storage.sums[0]);
#pragma unroll
        for (int k = 0; k < kLogprobTopK; ++k) {
            const LogprobCandidate candidate = storage.candidates[k];
            const bool empty                 = candidate.id == INT_MAX;
            const std::int64_t slot = static_cast<std::int64_t>(row) * kLogprobTopK + k;
            top_ids[slot]           = empty ? -1 : candidate.id;
            top_values[slot]        = empty ? -CUDART_INF_F : candidate.value - row_lse;
        }
        lse[row] = row_lse;
    }
}

} // namespace ninfer::ops
