// ninfer::ops - the block indexer of Qwen3.8-Flash-Next's sparse attention (contract in
// include/ninfer/ops/qsa_indexer.h). Pooling runs one CTA per completed block. Selection scores the
// blocks into workspace with CTAs of 64 blocks each (a grid sized by the capacity, so a graph
// replays it at any position), then one CTA per query runs a four-pass radix select of the 512th
// key and two ordered passes that keep ties at the lower block index and write the blocks in
// increasing order.
#include "ninfer/ops/qsa_indexer.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 128;
constexpr int kHeads       = 4;
constexpr int kRotaryPairs = 32;
constexpr int kBlock       = 4;
constexpr int kTopBlocks   = 512;
constexpr int kProjection  = kHeads * kHeadDim + kHeadDim;
constexpr int kThreads     = 512;
constexpr int kScoreBlocks = 64; // blocks one scoring CTA covers

// fl32(1e7^(-2i/64)) for i < 32: the text RoPE frequencies of theta 1e7 over 64 rotary dims.
__device__ __forceinline__ float rope_frequency(int pair) {
    return static_cast<float>(exp(-(2.0 * pair / 64.0) * 16.11809565095832)); // ln(1e7)
}

// Zero-centred RMSNorm of a 128-vector held one value per thread of a 128-thread group, then the
// rotate-half rotation of its first 64 dims at `position`. `values` is the group's shared row.
__device__ void norm_rope_row(float* values, const __nv_bfloat16* weight, float eps, int position,
                              int lane128, float* reduce) {
    const float x = values[lane128];
    float square  = x * x;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) square += __shfl_xor_sync(0xffffffffu, square, offset);
    const int warp = lane128 >> 5;
    if ((lane128 & 31) == 0) reduce[warp] = square;
    __syncwarp();
    __syncthreads();
    const float total = reduce[0] + reduce[1] + reduce[2] + reduce[3];
    const float normed =
        x * rsqrtf(total / kHeadDim + eps) * (1.0f + __bfloat162float(weight[lane128]));
    __syncthreads();
    values[lane128] = normed;
    __syncthreads();
    if (lane128 < kRotaryPairs) {
        const float angle = static_cast<float>(position) * rope_frequency(lane128);
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
        const float x1 = values[lane128], x2 = values[lane128 + kRotaryPairs];
        values[lane128]                = x1 * cosine - x2 * sine;
        values[lane128 + kRotaryPairs] = x2 * cosine + x1 * sine;
    }
    __syncthreads();
}

// One 128-thread CTA per block completing in the call: mean of its four raw keys (from the tail
// or the projection), norm, rope at the block's first position. The grid covers every block a call
// of `tokens` can complete; a CTA past the call's completed blocks has nothing to do.
__global__ void __launch_bounds__(kHeadDim)
    qsa_pool_kernel(const __nv_bfloat16* __restrict__ projection, const int* __restrict__ first,
                    int tokens, const __nv_bfloat16* __restrict__ key_norm, float eps,
                    const float* __restrict__ tail, int capacity, float* __restrict__ pooled) {
    __shared__ float row[kHeadDim];
    __shared__ float reduce[4];
    const int first_position = *first;
    const int block          = first_position / kBlock + static_cast<int>(blockIdx.x);
    if (block >= (first_position + tokens) / kBlock || block >= capacity) { return; }
    const int d     = threadIdx.x;
    float sum       = 0.0f;
#pragma unroll
    for (int i = 0; i < kBlock; ++i) {
        const int position = block * kBlock + i;
        sum += position < first_position
                   ? tail[(position % kBlock) * kHeadDim + d]
                   : __bfloat162float(projection[static_cast<std::int64_t>(position - first_position) *
                                                     kProjection +
                                                 kHeads * kHeadDim + d]);
    }
    row[d] = sum * 0.25f;
    __syncthreads();
    norm_rope_row(row, key_norm, eps, block * kBlock, d, reduce);
    pooled[static_cast<std::int64_t>(block) * kHeadDim + d] = row[d];
}

// The raw keys of the incomplete block after the call, at slot position mod 4. Positions of that
// block before the call are already in place.
__global__ void qsa_tail_kernel(const __nv_bfloat16* __restrict__ projection,
                                const int* __restrict__ first, int tokens,
                                float* __restrict__ tail) {
    const int first_position = *first;
    const int last      = first_position + tokens - 1;
    const int start     = (last + 1) / kBlock * kBlock; // first position of the incomplete block
    const int d         = threadIdx.x;
    for (int position = max(start, first_position); position <= last; ++position) {
        tail[(position % kBlock) * kHeadDim + d] = __bfloat162float(
            projection[static_cast<std::int64_t>(position - first_position) * kProjection +
                       kHeads * kHeadDim + d]);
    }
}

// Orderable key of a non-negative or negative float: larger floats give larger keys.
__device__ __forceinline__ unsigned order_key(float value) {
    const unsigned bits = __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

// The scores of blocks [64 x, 64 x + 64) for query y (the 1/sqrt(128) scale does not change the
// order), as orderable keys. A query that sees at most 512 blocks needs none.
__global__ void __launch_bounds__(kThreads)
    qsa_score_kernel(const __nv_bfloat16* __restrict__ projection, const int* __restrict__ first,
                     const __nv_bfloat16* __restrict__ query_norm, float eps,
                     const float* __restrict__ pooled, unsigned* __restrict__ keys_workspace,
                     int capacity) {
    __shared__ __align__(16) float query[kHeads * kHeadDim];
    __shared__ float reduce[kHeads][4];
    const int t        = blockIdx.y;
    const int position = *first + t;
    const int blocks   = min((position + 1) / kBlock, capacity);
    const int b0       = blockIdx.x * kScoreBlocks;
    if (blocks <= kTopBlocks || b0 >= blocks) { return; }
    const int tid = threadIdx.x;

    // Normalised, rotated query heads: four 128-thread groups.
    const int head = tid / kHeadDim, lane = tid % kHeadDim;
    query[tid] = __bfloat162float(projection[static_cast<std::int64_t>(t) * kProjection + tid]);
    __syncthreads();
    {
        // norm_rope_row synchronises the whole CTA; every group runs it together.
        float* row = query + head * kHeadDim;
        const float x = row[lane];
        float square  = x * x;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) square += __shfl_xor_sync(0xffffffffu, square, offset);
        if ((lane & 31) == 0) reduce[head][lane >> 5] = square;
        __syncthreads();
        const float total = reduce[head][0] + reduce[head][1] + reduce[head][2] + reduce[head][3];
        const float normed =
            x * rsqrtf(total / kHeadDim + eps) * (1.0f + __bfloat162float(query_norm[lane]));
        __syncthreads();
        row[lane] = normed;
        __syncthreads();
        if (lane < kRotaryPairs) {
            const float angle = static_cast<float>(position) * rope_frequency(lane);
            float sine, cosine;
            sincosf(angle, &sine, &cosine);
            const float x1 = row[lane], x2 = row[lane + kRotaryPairs];
            row[lane]                = x1 * cosine - x2 * sine;
            row[lane + kRotaryPairs] = x2 * cosine + x1 * sine;
        }
        __syncthreads();
    }

    // One warp per block, four dims per lane against every head.
    unsigned* keys = keys_workspace + static_cast<std::int64_t>(t) * capacity;
    const int warp = tid >> 5, warp_lane = tid & 31;
    float4 q[kHeads];
#pragma unroll
    for (int h = 0; h < kHeads; ++h) {
        q[h] = reinterpret_cast<const float4*>(query + h * kHeadDim)[warp_lane];
    }
    for (int b = b0 + warp; b < min(b0 + kScoreBlocks, blocks); b += kThreads / 32) {
        const float4 k = reinterpret_cast<const float4*>(pooled + static_cast<std::int64_t>(b) *
                                                                      kHeadDim)[warp_lane];
        float score    = 0.0f;
#pragma unroll
        for (int h = 0; h < kHeads; ++h) {
            float dot = q[h].x * k.x + q[h].y * k.y + q[h].z * k.z + q[h].w * k.w;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, offset);
            score += fmaxf(dot, 0.0f);
        }
        if (warp_lane == 0) keys[b] = order_key(score);
    }
}

__global__ void __launch_bounds__(kThreads)
    qsa_select_kernel(const int* __restrict__ first, const unsigned* __restrict__ keys_workspace,
                      int capacity, int* __restrict__ selected, int* __restrict__ counts) {
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ int histogram[256];
    __shared__ unsigned prefix_shared;
    __shared__ int remaining_shared;
    __shared__ int carry;

    const int t        = blockIdx.x;
    const int position = *first + t;
    const int blocks   = min((position + 1) / kBlock, capacity);
    int* out           = selected + static_cast<std::int64_t>(t) * kTopBlocks;
    const int tid      = threadIdx.x;
    if (blocks <= kTopBlocks) {
        for (int b = tid; b < blocks; b += kThreads) out[b] = b;
        if (tid == 0) counts[t] = blocks;
        return;
    }
    const unsigned* keys = keys_workspace + static_cast<std::int64_t>(t) * capacity;

    // Radix select of the kTopBlocks-th largest key, eight bits at a time from the top.
    if (tid == 0) {
        prefix_shared    = 0;
        remaining_shared = kTopBlocks;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += kThreads) histogram[i] = 0;
        __syncthreads();
        const unsigned prefix   = prefix_shared;
        const unsigned high_mask = shift == 24 ? 0u : ~0u << (shift + 8);
        for (int b = tid; b < blocks; b += kThreads) {
            const unsigned key = keys[b];
            if ((key & high_mask) == prefix) atomicAdd(&histogram[(key >> shift) & 0xff], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int remaining = remaining_shared;
            int digit     = 255;
            for (; digit > 0; --digit) {
                if (histogram[digit] >= remaining) break;
                remaining -= histogram[digit];
            }
            prefix_shared    = prefix | (static_cast<unsigned>(digit) << shift);
            remaining_shared = remaining; // how many keys equal to the threshold to keep
        }
        __syncthreads();
    }
    const unsigned threshold = prefix_shared;
    const int keep_equal     = remaining_shared;

    // Ordered compaction: every key above the threshold, and the first keep_equal keys equal to
    // it by block index, written in increasing block order.
    if (tid == 0) carry = 0;
    int equal_seen = 0; // equal keys before this chunk (uniform across the CTA)
    __syncthreads();
    for (int base = 0; base < blocks; base += kThreads) {
        const int b         = base + tid;
        const unsigned key  = b < blocks ? keys[b] : 0u;
        const int is_equal  = b < blocks && key == threshold;
        int equal_rank      = 0, equal_total = 0;
        Scan(scan_storage).ExclusiveSum(is_equal, equal_rank, equal_total);
        __syncthreads();
        const int take = b < blocks && (key > threshold || (is_equal && equal_seen + equal_rank < keep_equal));
        int slot = 0, taken = 0;
        Scan(scan_storage).ExclusiveSum(take, slot, taken);
        if (take) out[carry + slot] = b;
        __syncthreads();
        if (tid == 0) carry += taken;
        equal_seen += equal_total;
        __syncthreads();
    }
    if (tid == 0) counts[t] = carry;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa_indexer: ") + message); }
}

void require_projection(const Tensor& projection) {
    require(projection.dtype == DType::BF16 && projection.is_contiguous() &&
                projection.data != nullptr && projection.ne[0] == kProjection &&
                projection.ne[1] > 0 && projection.ne[2] == 1 && projection.ne[3] == 1,
            "projection must be contiguous BF16 [640, tokens]");
}

const int* require_position(const Tensor& first_position) {
    require(first_position.dtype == DType::I32 && first_position.data != nullptr &&
                first_position.numel() >= 1,
            "first_position must be a device I32 word");
    return static_cast<const int*>(first_position.data);
}

void require_norm(const Tensor* norm, const char* message) {
    require(norm != nullptr && norm->data != nullptr && norm->dtype == DType::BF16 &&
                norm->is_contiguous() && norm->ne[0] == kHeadDim && norm->ne[1] == 1,
            message);
}

void require_pooled(const Tensor& pooled) {
    require(pooled.dtype == DType::FP32 && pooled.is_contiguous() && pooled.data != nullptr &&
                pooled.ne[0] == kHeadDim && pooled.ne[1] > 0 && pooled.ne[2] == 1,
            "pooled must be contiguous FP32 [128, capacity]");
}

} // namespace

void qsa_indexer_append(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, Tensor& pooled, Tensor& tail,
                        cudaStream_t stream) {
    require_projection(projection);
    const int* first = require_position(first_position);
    require_norm(weights.key_norm, "key_norm must be BF16 [128]");
    require(eps > 0.0f, "eps must be positive");
    require_pooled(pooled);
    require(tail.dtype == DType::FP32 && tail.is_contiguous() && tail.data != nullptr &&
                tail.ne[0] == kHeadDim && tail.ne[1] == kBlock - 1,
            "tail must be contiguous FP32 [128, 3]");
    const std::int32_t tokens = projection.ne[1];
    // A call of `tokens` positions completes at most tokens / 4 + 1 blocks, wherever it starts.
    qsa_pool_kernel<<<tokens / kBlock + 1, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection.data), first, tokens,
        static_cast<const __nv_bfloat16*>(weights.key_norm->data), eps,
        static_cast<const float*>(tail.data), pooled.ne[1], static_cast<float*>(pooled.data));
    CUDA_CHECK(cudaGetLastError());
    qsa_tail_kernel<<<1, kHeadDim, 0, stream>>>(static_cast<const __nv_bfloat16*>(projection.data),
                                                first, tokens, static_cast<float*>(tail.data));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t qsa_indexer_select_workspace_bytes(std::int32_t tokens, std::int32_t capacity) {
    require(tokens > 0 && capacity > 0, "tokens and capacity must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I32, {capacity, tokens});
    return layout.peak_bytes(1);
}

void qsa_indexer_select(const Tensor& projection, const Tensor& first_position,
                        const QsaIndexerWeights& weights, float eps, const Tensor& pooled,
                        WorkspaceArena& workspace, Tensor& selected, Tensor& counts,
                        cudaStream_t stream) {
    require_projection(projection);
    const int* first = require_position(first_position);
    require_norm(weights.query_norm, "query_norm must be BF16 [128]");
    require(eps > 0.0f, "eps must be positive");
    require_pooled(pooled);
    const std::int32_t tokens = projection.ne[1];
    require(selected.dtype == DType::I32 && selected.is_contiguous() && selected.data != nullptr &&
                selected.ne[0] == kTopBlocks && selected.ne[1] == tokens,
            "selected must be contiguous I32 [512, tokens]");
    require(counts.dtype == DType::I32 && counts.is_contiguous() && counts.data != nullptr &&
                counts.ne[0] == tokens,
            "counts must be contiguous I32 [tokens]");
    const std::int32_t capacity = pooled.ne[1];
    auto scope                  = workspace.scope();
    Tensor keys                 = workspace.alloc(DType::I32, {capacity, tokens});
    const dim3 score_grid(static_cast<unsigned>((capacity + kScoreBlocks - 1) / kScoreBlocks),
                          static_cast<unsigned>(tokens));
    qsa_score_kernel<<<score_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection.data), first,
        static_cast<const __nv_bfloat16*>(weights.query_norm->data), eps,
        static_cast<const float*>(pooled.data), static_cast<unsigned*>(keys.data), capacity);
    CUDA_CHECK(cudaGetLastError());
    qsa_select_kernel<<<tokens, kThreads, 0, stream>>>(
        first, static_cast<const unsigned*>(keys.data), capacity, static_cast<int*>(selected.data),
        static_cast<int*>(counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
