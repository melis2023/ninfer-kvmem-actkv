// ninfer::ops - candidate_topk: each column's 16 largest logits (contract in
// include/ninfer/ops/candidate_selector.h). A column's vocabulary splits over kSplit CTAs: every
// thread keeps its best 16 in a sorted list, the CTA merges its threads' lists in shared memory,
// and one CTA per column merges the CTAs' lists.
#include "ninfer/ops/candidate_selector.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kTop = 16, kBlock = 128, kSplit = 32;

struct Candidate {
    float value;
    std::int32_t id;
};

// A larger value first, the lower id first on a tie; NaN is never better than anything.
__device__ __forceinline__ bool better(const Candidate& a, const Candidate& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ __forceinline__ void insert(Candidate (&top)[kTop], const Candidate& c) {
    if (!better(c, top[kTop - 1])) { return; }
    int p = kTop - 1;
    while (p > 0 && better(c, top[p - 1])) {
        top[p] = top[p - 1];
        --p;
    }
    top[p] = c;
}

// Merges every thread's sorted list into storage[0..kTop). All threads call.
__device__ void block_merge(Candidate* storage, const Candidate (&top)[kTop]) {
    const int tid = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int k = 0; k < kTop; ++k) { storage[tid * kTop + k] = top[k]; }
    __syncthreads();
    for (int step = kBlock / 2; step > 0; step >>= 1) {
        if (tid < step) {
            const Candidate* left  = storage + tid * kTop;
            const Candidate* right = storage + (tid + step) * kTop;
            Candidate out[kTop];
            int i = 0, j = 0;
#pragma unroll
            for (int k = 0; k < kTop; ++k) {
                const bool take_left = better(left[i], right[j]) || !better(right[j], left[i]);
                out[k]               = take_left ? left[i] : right[j];
                i += take_left ? 1 : 0;
                j += take_left ? 0 : 1;
            }
#pragma unroll
            for (int k = 0; k < kTop; ++k) { storage[tid * kTop + k] = out[k]; }
        }
        __syncthreads();
    }
}

template <typename T>
__device__ __forceinline__ float logit(const T* p, std::int64_t i) {
    if constexpr (std::is_same_v<T, float>) {
        return p[i];
    } else {
        return __bfloat162float(p[i]);
    }
}

// Partial blockIdx.x of column blockIdx.y.
template <typename T>
__global__ void __launch_bounds__(kBlock)
    candidate_topk_partials(const T* __restrict__ logits, std::int64_t rows,
                            std::int32_t valid_rows, Candidate* __restrict__ partials) {
    __shared__ Candidate storage[kBlock * kTop];
    const int column = static_cast<int>(blockIdx.y);
    const int chunk  = (valid_rows + kSplit - 1) / kSplit;
    const int begin  = static_cast<int>(blockIdx.x) * chunk;
    const int end    = min(begin + chunk, valid_rows);
    const T* source  = logits + std::int64_t(column) * rows;
    Candidate top[kTop];
#pragma unroll
    for (int k = 0; k < kTop; ++k) { top[k] = Candidate{-CUDART_INF_F, INT_MAX}; }
    for (int v = begin + static_cast<int>(threadIdx.x); v < end; v += kBlock) {
        insert(top, Candidate{logit(source, v), v});
    }
    block_merge(storage, top);
    if (threadIdx.x < kTop) {
        partials[(std::int64_t(column) * kSplit + blockIdx.x) * kTop + threadIdx.x] =
            storage[threadIdx.x];
    }
}

// Column blockIdx.x's partials into its candidates. A slot no finite or infinite logit filled (a
// column of NaN) names token 0, so a consumer never reads past the vocabulary.
__global__ void __launch_bounds__(kBlock)
    candidate_topk_combine(const Candidate* __restrict__ partials, std::int32_t* __restrict__ ids,
                           float* __restrict__ scores) {
    __shared__ Candidate storage[kBlock * kTop];
    const int column = static_cast<int>(blockIdx.x);
    Candidate top[kTop];
#pragma unroll
    for (int k = 0; k < kTop; ++k) {
        top[k] = threadIdx.x < kSplit
                     ? partials[(std::int64_t(column) * kSplit + threadIdx.x) * kTop + k]
                     : Candidate{-CUDART_INF_F, INT_MAX};
    }
    block_merge(storage, top);
    if (threadIdx.x < kTop) {
        const Candidate c                          = storage[threadIdx.x];
        ids[std::int64_t(column) * kTop + threadIdx.x]    = c.id == INT_MAX ? 0 : c.id;
        scores[std::int64_t(column) * kTop + threadIdx.x] = c.value;
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("candidate_topk: ") + message); }
}

} // namespace

std::size_t candidate_topk_workspace_capacity_bytes(std::int32_t columns) {
    require(columns > 0, "columns must be positive");
    return std::size_t(columns) * kSplit * kTop * sizeof(Candidate) + 256;
}

void candidate_topk(const Tensor& logits, std::int32_t valid_rows, Tensor& candidate_ids,
                    Tensor& candidate_scores, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t rows = logits.ne[0], columns = logits.ne[1];
    require(logits.data != nullptr && logits.is_contiguous() && logits.ne[2] == 1 &&
                logits.ne[3] == 1 && (logits.dtype == DType::BF16 || logits.dtype == DType::FP32),
            "logits must be contiguous BF16 or FP32 [rows,U]");
    require(columns > 0 && valid_rows >= kTop && valid_rows <= rows,
            "U must be positive and valid_rows in [16,rows]");
    require(candidate_ids.dtype == DType::I32 && candidate_ids.is_contiguous() &&
                candidate_ids.ne[0] == kTop &&
                std::int64_t(candidate_ids.ne[1]) * candidate_ids.ne[2] * candidate_ids.ne[3] ==
                    columns,
            "candidate_ids must be contiguous I32 [16,U]");
    require(candidate_scores.dtype == DType::FP32 && candidate_scores.is_contiguous() &&
                candidate_scores.ne[0] == kTop &&
                std::int64_t(candidate_scores.ne[1]) * candidate_scores.ne[2] *
                        candidate_scores.ne[3] ==
                    columns,
            "candidate_scores must be contiguous FP32 [16,U]");
    auto scope         = workspace.scope();
    Tensor raw         = workspace.alloc(DType::I32, {columns * kSplit * kTop * 2});
    auto* partials     = static_cast<Candidate*>(raw.data);
    const dim3 grid(kSplit, static_cast<unsigned>(columns));
    if (logits.dtype == DType::FP32) {
        candidate_topk_partials<float><<<grid, kBlock, 0, stream>>>(
            static_cast<const float*>(logits.data), rows, valid_rows, partials);
    } else {
        candidate_topk_partials<__nv_bfloat16><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), rows, valid_rows, partials);
    }
    CUDA_CHECK(cudaGetLastError());
    candidate_topk_combine<<<static_cast<unsigned>(columns), kBlock, 0, stream>>>(
        partials, static_cast<std::int32_t*>(candidate_ids.data),
        static_cast<float*>(candidate_scores.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
