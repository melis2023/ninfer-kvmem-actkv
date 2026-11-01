#include "core/weight.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                  = 48;
constexpr int kK                  = 5120;
constexpr int kThreads            = 256;
constexpr int kLogicalRows        = 2 * kN;
constexpr int kSmallTMax          = 8;
constexpr int kSmallTKSlice       = 512;
constexpr int kSmallTSplits       = kK / kSmallTKSlice;
constexpr int kSmallTRowsPerBlock = 4;
constexpr int kSmallTThreads      = kSmallTRowsPerBlock * 32;
static_assert(kK % kSmallTKSlice == 0, "small-T K split must divide K");

constexpr int k35N           = 32;
constexpr int k35K           = 2048;
constexpr int k35LogicalRows = 2 * k35N;

template <int TokenTile, int KSlice, int RowsPerBlock>
__global__ void bf16_gdn_gating_proj_small_t_partial_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, float* __restrict__ partial, std::int32_t t) {
    static_assert(TokenTile == kSmallTMax, "small-T token tile is fixed to 8");
    static_assert(KSlice == kSmallTKSlice, "small-T K split is fixed to 512");
    static_assert(RowsPerBlock == kSmallTRowsPerBlock, "small-T rows/block mismatch");
    constexpr int kVecsPerCol = KSlice / 8;

    __shared__ __align__(16) __nv_bfloat16 x_sh[TokenTile][KSlice];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int token0 = static_cast<int>(blockIdx.z) * TokenTile;
    if (token0 >= t) { return; }
    const int ncols = min(TokenTile, t - token0);
    const int split = static_cast<int>(blockIdx.y);
    const int k0    = split * KSlice;

    auto* x_sh_v = reinterpret_cast<uint4*>(x_sh);
    for (int i = static_cast<int>(threadIdx.x); i < ncols * kVecsPerCol;
         i += static_cast<int>(blockDim.x)) {
        const int col = i / kVecsPerCol;
        const int vec = i - col * kVecsPerCol;
        x_sh_v[col * kVecsPerCol + vec] =
            load_vec<uint4>(x + static_cast<std::int64_t>(token0 + col) * kK + k0 + vec * 8);
    }
    __syncthreads();

    const int logical_row = static_cast<int>(blockIdx.x) * RowsPerBlock + warp;
    if (logical_row >= kLogicalRows) { return; }
    const bool is_b = logical_row >= kN;
    const int row   = is_b ? logical_row - kN : logical_row;
    const __nv_bfloat16* wrow =
        (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * kK + k0;

    float acc[TokenTile];
#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) { acc[tt] = 0.0f; }

    for (int vec = lane; vec < kVecsPerCol; vec += 32) {
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);

#pragma unroll
        for (int tt = 0; tt < TokenTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv   = x_sh_v[tt * kVecsPerCol + vec];
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
                acc[tt]          = fmaf(wf0.x, xf0.x, acc[tt]);
                acc[tt]          = fmaf(wf0.y, xf0.y, acc[tt]);
                acc[tt]          = fmaf(wf1.x, xf1.x, acc[tt]);
                acc[tt]          = fmaf(wf1.y, xf1.y, acc[tt]);
                acc[tt]          = fmaf(wf2.x, xf2.x, acc[tt]);
                acc[tt]          = fmaf(wf2.y, xf2.y, acc[tt]);
                acc[tt]          = fmaf(wf3.x, xf3.x, acc[tt]);
                acc[tt]          = fmaf(wf3.y, xf3.y, acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) {
        if (tt < ncols) {
            float sum = warp_reduce_sum(acc[tt]);
            if (lane == 0) {
                const int token      = token0 + tt;
                partial[(static_cast<std::int64_t>(split) * t + token) * kLogicalRows +
                        logical_row] = sum;
            }
        }
    }
}

__global__ void bf16_gdn_gating_proj_small_t_reduce_kernel(const float* __restrict__ partial,
                                                           const float* __restrict__ A_log,
                                                           const float* __restrict__ dt_bias,
                                                           float* __restrict__ g,
                                                           float* __restrict__ beta,
                                                           std::int32_t t) {
    const int i =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int elems = kN * t;
    if (i >= elems) { return; }

    const int row   = i % kN;
    const int token = i / kN;
    float acc_a     = 0.0f;
    float acc_b     = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallTSplits; ++split) {
        const std::int64_t base = (static_cast<std::int64_t>(split) * t + token) * kLogicalRows;
        acc_a += partial[base + row];
        acc_b += partial[base + kN + row];
    }

    const std::int64_t out_index = static_cast<std::int64_t>(token) * kN + row;
    const float sp               = softplus(acc_a + dt_bias[row]);
    g[out_index]                 = -expf(A_log[row]) * sp;
    beta[out_index]              = sigmoid(acc_b);
}

__global__ void bf16_gdn_gating_proj_gemv_kernel(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* a_weight,
                                                 const __nv_bfloat16* b_weight, const float* A_log,
                                                 const float* dt_bias, float* g, float* beta) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= kN;
    const int row        = is_b ? global_row - kN : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;
    __shared__ float warp_sums[kThreads / kWarpSize];

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf16x2_to_float2(w2[p]);
        const float2 xf = bf16x2_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc, warp_sums);
    if (threadIdx.x == 0) {
        if (is_b) {
            beta[row] = sigmoid(acc);
        } else {
            const float sp = softplus(acc + dt_bias[row]);
            g[row]         = -expf(A_log[row]) * sp;
        }
    }
}

template <int ColsPerTile>
__global__ void bf16_gdn_gating_proj_35_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                    const __nv_bfloat16* __restrict__ a_weight,
                                                    const __nv_bfloat16* __restrict__ b_weight,
                                                    const float* __restrict__ A_log,
                                                    const float* __restrict__ dt_bias,
                                                    float* __restrict__ g, float* __restrict__ beta,
                                                    std::int32_t t) {
    static_assert(ColsPerTile == 4 || ColsPerTile == 8);
    const int lane        = static_cast<int>(threadIdx.x);
    const int logical_row = static_cast<int>(blockIdx.x);
    const bool is_b       = logical_row >= k35N;
    const int row         = is_b ? logical_row - k35N : logical_row;
    const int col0        = static_cast<int>(blockIdx.y) * ColsPerTile;
    const int ncols       = min(ColsPerTile, t - col0);
    const auto* weight    = is_b ? b_weight : a_weight;
    const auto* wrow      = weight + static_cast<std::int64_t>(row) * k35K;

    float acc[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) { acc[col] = 0.0f; }

    constexpr int kVecs = k35K / 8;
    for (int vec = lane; vec < kVecs; vec += 32) {
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if (col < ncols) {
                const uint4 xv =
                    load_vec<uint4>(x + static_cast<std::int64_t>(col0 + col) * k35K + vec * 8);
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
                acc[col]         = fmaf(wf0.x, xf0.x, acc[col]);
                acc[col]         = fmaf(wf0.y, xf0.y, acc[col]);
                acc[col]         = fmaf(wf1.x, xf1.x, acc[col]);
                acc[col]         = fmaf(wf1.y, xf1.y, acc[col]);
                acc[col]         = fmaf(wf2.x, xf2.x, acc[col]);
                acc[col]         = fmaf(wf2.y, xf2.y, acc[col]);
                acc[col]         = fmaf(wf3.x, xf3.x, acc[col]);
                acc[col]         = fmaf(wf3.y, xf3.y, acc[col]);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        if (col < ncols) {
            const float sum = warp_reduce_sum(acc[col]);
            if (lane == 0) {
                const std::int64_t out_index = static_cast<std::int64_t>(col0 + col) * k35N + row;
                if (is_b) {
                    beta[out_index] = sigmoid(sum);
                } else {
                    g[out_index] = -expf(A_log[row]) * softplus(sum + dt_bias[row]);
                }
            }
        }
    }
}

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [48,5120]");
    }
}

void require_shape35(const Weight& w, const char* name) {
    if (w.n != k35N || w.k != k35K || w.shape[0] != k35N || w.shape[1] != k35K) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [32,2048]");
    }
}

// CTAs of one cooperative split-K kernel the current device holds per SM, the lesser of its
// full-tile and predicated forms: the occupancy of the compiled kernel (its registers, block size
// and shared memory) on that device, so a Turing SM's 64 KiB of shared memory and an sm_86 SM's
// 100 KiB each give their own count. A cooperative grid larger than this times the SM count is
// refused by the driver. Zero when the query fails.
template <class Geometry, int SplitK, int Warps, bool NormalizeInput, int NormTokenCapacity>
std::int32_t cooperative_resident_ctas_per_sm() noexcept {
    static_assert(SplitK > 1);
    constexpr int kCachedDevices = 64;
    static std::array<std::atomic<std::int32_t>, kCachedDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
    const bool cached = device >= 0 && device < kCachedDevices;
    if (cached) {
        const std::int32_t known =
            cache[static_cast<std::size_t>(device)].load(std::memory_order_relaxed);
        if (known > 0) { return known; }
    }
    constexpr int kSmemBytes = kBf16GdnSmemBytes<Geometry::kBlockN>;
    const auto occupancy     = [](auto kernel) -> int {
        int blocks = 0;
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes) !=
                cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, Warps * 32,
                                                              kSmemBytes) != cudaSuccess) {
            return 0;
        }
        return blocks;
    };
    const std::int32_t ctas = std::min(
        occupancy(bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, true, Warps,
                                                       NormalizeInput, NormTokenCapacity>),
        occupancy(bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, false, Warps,
                                                       NormalizeInput, NormTokenCapacity>));
    if (cached && ctas > 0) {
        cache[static_cast<std::size_t>(device)].store(ctas, std::memory_order_relaxed);
    }
    return ctas;
}

template <class Geometry, int SplitK, int Warps = kBf16GdnWarps, bool NormalizeInput = false,
          int NormTokenCapacity = 0>
bool launch_bf16_prefill_mma(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                             const Tensor* norm_weight, float norm_eps, Tensor* normalized_x,
                             const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                             const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
                             cudaStream_t stream, std::int32_t multiprocessor_count = 0) {
    constexpr int kBlockN    = Geometry::kBlockN;
    constexpr int kSmemBytes = kBf16GdnSmemBytes<kBlockN>;
    const dim3 block(Warps * 32);
    auto launch_problem = [&](Bf16GdnGatingTokenVariant launch_variant, const Tensor& launch_x,
                              Tensor* launch_normalized_x, Tensor& launch_g, Tensor& launch_beta) {
        const std::int32_t launch_t = launch_x.ne[1];
        const dim3 grid(static_cast<unsigned>(div_up(launch_t, kBlockN)),
                        static_cast<unsigned>(Geometry::kHeads / kBf16GdnBlockM),
                        static_cast<unsigned>(SplitK));
        auto launch = [&](auto full_tokens) {
            constexpr bool FullTokens     = decltype(full_tokens)::value;
            raise_dynamic_shared_memory(
                bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                     NormalizeInput, NormTokenCapacity>,
                kSmemBytes);
            if constexpr (SplitK > 1) {
                cudaLaunchConfig_t config{};
                config.gridDim          = grid;
                config.blockDim         = block;
                config.dynamicSmemBytes = kSmemBytes;
                config.stream           = stream;
                cudaLaunchAttribute cooperative{};
                cooperative.id              = cudaLaunchAttributeCooperative;
                cooperative.val.cooperative = 1;
                config.attrs                = &cooperative;
                config.numAttrs             = 1;
                CUDA_CHECK(cudaLaunchKernelEx(
                    &config,
                    bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                         NormalizeInput, NormTokenCapacity>,
                    static_cast<const __nv_bfloat16*>(launch_x.data),
                    norm_weight != nullptr ? static_cast<const __nv_bfloat16*>(norm_weight->data)
                                           : static_cast<const __nv_bfloat16*>(nullptr),
                    launch_normalized_x != nullptr
                        ? static_cast<__nv_bfloat16*>(launch_normalized_x->data)
                        : static_cast<__nv_bfloat16*>(nullptr),
                    norm_eps, static_cast<const __nv_bfloat16*>(a_weight.qdata),
                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                    static_cast<float*>(workspace), static_cast<float*>(launch_g.data),
                    static_cast<float*>(launch_beta.data), launch_t));
            } else {
                bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                     NormalizeInput, NormTokenCapacity>
                    <<<grid, block, kSmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(launch_x.data),
                        norm_weight != nullptr
                            ? static_cast<const __nv_bfloat16*>(norm_weight->data)
                            : static_cast<const __nv_bfloat16*>(nullptr),
                        launch_normalized_x != nullptr
                            ? static_cast<__nv_bfloat16*>(launch_normalized_x->data)
                            : static_cast<__nv_bfloat16*>(nullptr),
                        norm_eps, static_cast<const __nv_bfloat16*>(a_weight.qdata),
                        static_cast<const __nv_bfloat16*>(b_weight.qdata),
                        static_cast<const float*>(A_log.data),
                        static_cast<const float*>(dt_bias.data), static_cast<float*>(workspace),
                        static_cast<float*>(launch_g.data), static_cast<float*>(launch_beta.data),
                        launch_t);
            }
        };
        if (launch_variant == Bf16GdnGatingTokenVariant::Full) {
            launch(std::true_type{});
        } else if (launch_variant == Bf16GdnGatingTokenVariant::Predicated) {
            launch(std::false_type{});
        } else {
            throw std::invalid_argument(
                "BF16 GDN gating MMA requires Full or Predicated token variant");
        }
    };

    if constexpr (SplitK == 1) {
        launch_problem(variant, x, normalized_x, g, beta);
    } else {
        constexpr std::int64_t kCtasPerTokenTile =
            static_cast<std::int64_t>(Geometry::kHeads / kBf16GdnBlockM) * SplitK;
        const std::int64_t resident_ctas =
            static_cast<std::int64_t>(multiprocessor_count) *
            cooperative_resident_ctas_per_sm<Geometry, SplitK, Warps, NormalizeInput,
                                             NormTokenCapacity>();
        const std::int64_t max_token_tiles = resident_ctas / kCtasPerTokenTile;
        if (max_token_tiles < 1) { return false; }

        const std::int32_t t = x.ne[1];
        const std::int64_t total_token_tiles =
            div_up(static_cast<std::int64_t>(t), static_cast<std::int64_t>(kBlockN));
        if (total_token_tiles <= max_token_tiles) {
            launch_problem(variant, x, normalized_x, g, beta);
        } else {
            // Token tiles have no cross-tile reduction. Rebase each tensor to a disjoint token
            // interval and reuse the call-scoped partial workspace in stream order.
            const std::int64_t launch_token_capacity = max_token_tiles * kBlockN;
            for (std::int32_t token_begin = 0; token_begin < t;) {
                const std::int32_t launch_t = static_cast<std::int32_t>(std::min<std::int64_t>(
                    static_cast<std::int64_t>(t) - token_begin, launch_token_capacity));
                Tensor launch_x             = x.slice(1, token_begin, launch_t);
                Tensor launch_g             = g.slice(1, token_begin, launch_t);
                Tensor launch_beta          = beta.slice(1, token_begin, launch_t);
                Tensor launch_normalized_storage{};
                Tensor* launch_normalized_x = nullptr;
                if (normalized_x != nullptr) {
                    launch_normalized_storage = normalized_x->slice(1, token_begin, launch_t);
                    launch_normalized_x       = &launch_normalized_storage;
                }
                const Bf16GdnGatingTokenVariant launch_variant =
                    launch_t % kBlockN == 0 ? Bf16GdnGatingTokenVariant::Full
                                            : Bf16GdnGatingTokenVariant::Predicated;
                launch_problem(launch_variant, launch_x, launch_normalized_x, launch_g,
                               launch_beta);
                token_begin += launch_t;
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    bf16_gdn_gating_proj_gemv_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    const std::int32_t t       = x.ne[1];
    const std::size_t required = static_cast<std::size_t>(kSmallTSplits) *
                                 static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(kLogicalRows) * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
        throw std::invalid_argument("gdn_gating_proj: small-T workspace is too small");
    }

    dim3 partial_block(kSmallTThreads);
    dim3 partial_grid(div_up(kLogicalRows, kSmallTRowsPerBlock), kSmallTSplits,
                      div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_partial_kernel<kSmallTMax, kSmallTKSlice, kSmallTRowsPerBlock>
        <<<partial_grid, partial_block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace), t);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kReduceThreads = 128;
    const int reduce_elems       = kN * t;
    const int reduce_blocks      = div_up(reduce_elems, kReduceThreads);
    bf16_gdn_gating_proj_small_t_reduce_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(
        static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}

bool bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            std::int32_t multiprocessor_count,
                                            cudaStream_t stream) {
    return launch_bf16_prefill_mma<Bf16Gdn27Geometry, 8, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

bool bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            std::int32_t multiprocessor_count,
                                            cudaStream_t stream) {
    // Eight warps, matching every other MMA route in this file. At the default 16 warps the CTA
    // needs 512 threads' worth of registers, which admits a single CTA per sm_86 SM and collapses
    // the cooperative budget to just the SM count. See the residency note in the plan's route
    // table.
    return launch_bf16_prefill_mma<Bf16Gdn27Geometry, 4, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

bool bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            std::int32_t multiprocessor_count,
                                            cudaStream_t stream) {
    return launch_bf16_prefill_mma<Bf16Gdn27Geometry, 2, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream) {
    (void)launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8>(variant, x, nullptr, 0.0F, nullptr,
                                                           a_weight, b_weight, A_log, dt_bias,
                                                           nullptr, g, beta, stream);
}

template <int ColsPerTile>
void launch_35_simt(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                    const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                    cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    const dim3 grid(static_cast<unsigned>(k35LogicalRows),
                    static_cast<unsigned>(div_up(x.ne[1], ColsPerTile)), 1u);
    bf16_gdn_gating_proj_35_simt_kernel<ColsPerTile><<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_35_simt<4>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
}

void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_35_simt<8>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
}

bool bf16_gdn_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                std::int32_t multiprocessor_count,
                                                cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    return launch_bf16_prefill_mma<Bf16Gdn35Geometry, 32, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

bool bf16_gdn_norm_gating_proj_35_mma_split32_launch(
    Bf16GdnGatingTokenVariant variant, const Tensor& x, const Tensor& norm_weight, float eps,
    Tensor& h, const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
    const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
    std::int32_t multiprocessor_count, cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    const auto launch = [&](auto token_capacity) -> bool {
        constexpr int TokenCapacity = decltype(token_capacity)::value;
        return launch_bf16_prefill_mma<Bf16Gdn35Geometry, 32, 8, true, TokenCapacity>(
            variant, x, &norm_weight, eps, &h, a_weight, b_weight, A_log, dt_bias, workspace, g,
            beta, stream, multiprocessor_count);
    };
    if (x.ne[1] <= 6) {
        return launch(std::integral_constant<int, 6>{});
    } else if (x.ne[1] <= 8) {
        return launch(std::integral_constant<int, 8>{});
    } else if (x.ne[1] <= 12) {
        return launch(std::integral_constant<int, 12>{});
    } else if (x.ne[1] <= 16) {
        return launch(std::integral_constant<int, 16>{});
    } else {
        throw std::invalid_argument("fused BF16 GDN norm/control requires T=1..16");
    }
}

bool bf16_gdn_gating_proj_35_mma_split16_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                std::int32_t multiprocessor_count,
                                                cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    return launch_bf16_prefill_mma<Bf16Gdn35Geometry, 16, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

bool bf16_gdn_gating_proj_35_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               std::int32_t multiprocessor_count,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    return launch_bf16_prefill_mma<Bf16Gdn35Geometry, 8, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

bool bf16_gdn_gating_proj_35_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               std::int32_t multiprocessor_count,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    return launch_bf16_prefill_mma<Bf16Gdn35Geometry, 4, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

bool bf16_gdn_gating_proj_35_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               std::int32_t multiprocessor_count,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    return launch_bf16_prefill_mma<Bf16Gdn35Geometry, 2, 8>(
        variant, x, nullptr, 0.0F, nullptr, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
        stream, multiprocessor_count);
}

void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    (void)launch_bf16_prefill_mma<Bf16Gdn35Geometry, 1, 8>(variant, x, nullptr, 0.0F, nullptr,
                                                           a_weight, b_weight, A_log, dt_bias,
                                                           nullptr, g, beta, stream);
}

std::int32_t bf16_gdn_gating_proj_mma_resident_ctas_per_sm(bool geometry35,
                                                           std::int32_t split_k) noexcept {
    if (geometry35) {
        switch (split_k) {
        case 32:
            return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 32, 8, false, 0>();
        case 16:
            return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 16, 8, false, 0>();
        case 8:
            return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 8, 8, false, 0>();
        case 4:
            return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 4, 8, false, 0>();
        case 2:
            return cooperative_resident_ctas_per_sm<Bf16Gdn35Geometry, 2, 8, false, 0>();
        default:
            return 0;
        }
    }
    switch (split_k) {
    case 8:
        return cooperative_resident_ctas_per_sm<Bf16Gdn27Geometry, 8, 8, false, 0>();
    case 4:
        return cooperative_resident_ctas_per_sm<Bf16Gdn27Geometry, 4, 8, false, 0>();
    case 2:
        return cooperative_resident_ctas_per_sm<Bf16Gdn27Geometry, 2, 8, false, 0>();
    default:
        return 0;
    }
}


namespace {
// One warp computes both FP32 control reductions for a (token, value-head) pair.
__global__ void gdn_control_general(const __nv_bfloat16* x, const __nv_bfloat16* a,
    const __nv_bfloat16* b, const float* alog, const float* bias, float* g, float* beta,
    int hidden, int heads, int tokens) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * 8 + threadIdx.x / 32;
    if (row >= static_cast<std::int64_t>(heads) * tokens) return;
    const int lane = threadIdx.x % 32, head = row % heads;
    const auto input = x + (row / heads) * hidden;
    const auto wa = a + static_cast<std::int64_t>(head) * hidden;
    const auto wb = b + static_cast<std::int64_t>(head) * hidden;
    float sum_a = 0, sum_b = 0;
    for (int k = lane; k < hidden; k += 32) {
        const float value = __bfloat162float(input[k]);
        sum_a = fmaf(value, __bfloat162float(wa[k]), sum_a);
        sum_b = fmaf(value, __bfloat162float(wb[k]), sum_b);
    }
    for (int delta = 16; delta; delta /= 2) {
        sum_a += __shfl_down_sync(0xffffffff, sum_a, delta);
        sum_b += __shfl_down_sync(0xffffffff, sum_b, delta);
    }
    if (lane == 0) {
        g[row] = -expf(alog[head]) * softplus(sum_a + bias[head]);
        beta[row] = sigmoid(sum_b);
    }
}
} // namespace

void bf16_gdn_gating_proj_general_launch(const Tensor& x, const Weight& a_weight,
    const Weight& b_weight, const Tensor& alog, const Tensor& bias,
    Tensor& g, Tensor& beta, cudaStream_t stream) {
    const auto blocks = (static_cast<std::int64_t>(g.ne[0]) * x.ne[1] + 7) / 8;
    if (blocks > 2147483647) throw std::invalid_argument("GDN control grid exceeds CUDA limit");
    gdn_control_general<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(alog.data),
        static_cast<const float*>(bias.data), static_cast<float*>(g.data), static_cast<float*>(beta.data),
        x.ne[0], g.ne[0], x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
