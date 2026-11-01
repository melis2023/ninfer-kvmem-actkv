#include "ninfer/ops/moe_experts.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHidden = 2560, kWidth = 640, kTop = 10, kSlots = 11, kWarps = 8;
constexpr int kMmaTokens = 16, kMmaColumns = 32, kMaxExperts = 512;

struct RouteGroups {
    int* counts = nullptr;
    int* offsets = nullptr;
    int* cursors = nullptr;
    int* total_tiles = nullptr;
    int* pairs = nullptr;
    int2* tiles = nullptr;
    int capacity = 0;
};

int tile_capacity(int tokens) { return (tokens * kTop + kMmaColumns - 1) / kMmaColumns + kMaxExperts; }

__global__ void count_routes(const int* ids, int count, int* counts) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && ids[i] >= 0) { atomicAdd(counts + ids[i], 1); }
}

__global__ void select_parts(const int* ids, const unsigned char* parts, int count, int* selected) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { selected[i] = parts[i] ? ids[i] : -1; }
}

__global__ void plan_groups(RouteGroups groups, int experts) {
    __shared__ int rows[kMaxExperts], tiles[kMaxExperts];
    const int e = threadIdx.x;
    const int count = e < experts ? groups.counts[e] : 0;
    const int tile_count = (count + kMmaColumns - 1) / kMmaColumns;
    rows[e] = count;
    tiles[e] = tile_count;
    __syncthreads();
    for (int stride = 1; stride < kMaxExperts; stride *= 2) {
        const int add_rows = e >= stride ? rows[e - stride] : 0;
        const int add_tiles = e >= stride ? tiles[e - stride] : 0;
        __syncthreads();
        rows[e] += add_rows;
        tiles[e] += add_tiles;
        __syncthreads();
    }
    groups.offsets[e] = rows[e] - count;
    groups.cursors[e] = 0;
    for (int tile = 0; tile < tile_count; ++tile) {
        groups.tiles[tiles[e] - tile_count + tile] = make_int2(e, rows[e] - count + tile * kMmaColumns);
    }
    if (e == kMaxExperts - 1) { *groups.total_tiles = tiles[e]; }
}

__global__ void scatter_routes(const int* ids, int count, RouteGroups groups) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int e = ids[i];
    if (e < 0) { return; }
    const int at = groups.offsets[e] + atomicAdd(groups.cursors + e, 1);
    groups.pairs[at] = (i / kTop) * kSlots + i % kTop;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_experts_native: ") + message); }
}

bool supported(QType format) {
    return format == QType::BF16 || format == QType::Q2_G64_FP16 ||
           format == QType::Q4_G64_FP16 || format == QType::Q5_G64_FP16 ||
           format == QType::Q6_G64_FP16 || format == QType::Q8_G32_FP16;
}

bool shaped(const Tensor& tensor, DType dtype, int rows, int columns) {
    return tensor.data && tensor.dtype == dtype && tensor.is_contiguous() &&
           tensor.ne[0] == rows && tensor.ne[1] == columns && tensor.ne[2] == 1 &&
           tensor.ne[3] == 1;
}

__device__ float warp_sum(float value) {
#pragma unroll
    for (int shift = 16; shift; shift >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, shift); }
    return value;
}

// A weight's scale plane stays row-major even when its code/high planes are panel-major.
template <QType Format, int K>
__device__ int packed_four(const Weight& w, int row, int column, float& scale) {
    constexpr int group_size = Format == QType::Q8_G32_FP16 ? 32 : 64;
    constexpr int groups = K / group_size;
    const int group = column / group_size, lane = column % group_size;
    const std::int64_t record = w.layout == QuantLayout::RowSplitPanel
        ? (std::int64_t(row / 64) * groups + group) * 64 + row % 64
        : std::int64_t(row) * groups + group;
    scale = __half2float(static_cast<const __half*>(w.scales)[std::int64_t(row) * groups + group]);
    const auto* codes = static_cast<const unsigned char*>(w.qdata);
    if constexpr (Format == QType::Q8_G32_FP16) {
        return *reinterpret_cast<const int*>(codes + record * 32 + lane);
    } else {
        unsigned low, high = 0;
        if constexpr (Format == QType::Q2_G64_FP16) {
            low = codes[record * 16 + lane / 4];
        } else {
            low = *reinterpret_cast<const unsigned short*>(codes + record * 32 + lane / 2);
            if constexpr (Format == QType::Q5_G64_FP16) {
                high = static_cast<const unsigned char*>(w.qhigh)[record * 8 + lane / 8] >> (lane % 8);
            } else if constexpr (Format == QType::Q6_G64_FP16) {
                high = static_cast<const unsigned char*>(w.qhigh)[record * 16 + lane / 4];
            }
        }
        unsigned packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q;
            if constexpr (Format == QType::Q2_G64_FP16) {
                q = int((low >> (2 * i)) & 3) - 1;
            } else {
                unsigned u = (low >> (4 * i)) & 15;
                constexpr int bits = Format == QType::Q4_G64_FP16 ? 4 :
                                     Format == QType::Q5_G64_FP16 ? 5 : 6;
                if constexpr (bits == 5) { u |= ((high >> i) & 1) << 4; }
                if constexpr (bits == 6) { u |= ((high >> (2 * i)) & 3) << 4; }
                q = int(u) - ((u & (1u << (bits - 1))) ? (1 << bits) : 0);
            }
            packed |= unsigned(static_cast<unsigned char>(q)) << (8 * i);
        }
        return static_cast<int>(packed);
    }
}

template <typename Source>
__device__ float value(const Source* x, int index) { return float(x[index]); }

// Every MMA sums one private 32-value activation group exactly in int32. Rescaling and the
// FP32 sum over groups follow in a fixed order, independent of route scatter/column order.
template <QType Format, bool Shared, bool Middle, bool Fused>
__global__ void grouped_mma(const signed char* q, const float* scales, const Weight* first,
                            const Weight* second, RouteGroups groups, int tokens, float* output) {
    constexpr int K = Middle ? kHidden : kWidth;
    constexpr int N = Middle ? kWidth : kHidden;
    constexpr int row_tiles = N / 64;
    const int tile = int(blockIdx.x) / row_tiles;
    if constexpr (Shared) {
        if (tile >= (tokens + kMmaColumns - 1) / kMmaColumns) { return; }
    } else {
        if (tile >= *groups.total_tiles) { return; }
    }
    const int2 selected = Shared ? make_int2(0, tile * kMmaColumns) : groups.tiles[tile];
    __shared__ int pairs[kMmaColumns];
    if (threadIdx.x < kMmaColumns) {
        const int at = selected.y + threadIdx.x;
        if constexpr (Shared) {
            pairs[threadIdx.x] = at < tokens ? at * kSlots + kTop : -1;
        } else {
            pairs[threadIdx.x] = at < groups.offsets[selected.x] + groups.counts[selected.x]
                ? groups.pairs[at] : -1;
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, gid = lane / 4, quarter = lane % 4;
    const int row = (int(blockIdx.x) % row_tiles) * 64 + (threadIdx.x / 32) * 16 + gid;
    float a[kMmaColumns / 8][4] = {}, b[kMmaColumns / 8][4] = {};
    for (int group = 0; group < K / 32; ++group) {
        const int column = group * 32 + quarter * 8;
        float w0, w1, unused;
        const unsigned a0 = packed_four<Format, K>(first[selected.x], row, column, w0);
        const unsigned a1 = packed_four<Format, K>(first[selected.x], row + 8, column, w1);
        const unsigned a2 = packed_four<Format, K>(first[selected.x], row, column + 4, unused);
        const unsigned a3 = packed_four<Format, K>(first[selected.x], row + 8, column + 4, unused);
        unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        float u0 = 0, u1 = 0;
        if constexpr (Fused) {
            b0 = packed_four<Format, K>(second[selected.x], row, column, u0);
            b1 = packed_four<Format, K>(second[selected.x], row + 8, column, u1);
            b2 = packed_four<Format, K>(second[selected.x], row, column + 4, unused);
            b3 = packed_four<Format, K>(second[selected.x], row + 8, column + 4, unused);
        }
#pragma unroll
        for (int nt = 0; nt < kMmaColumns / 8; ++nt) {
            if (pairs[nt * 8] < 0) { continue; }
            const int load_pair = pairs[nt * 8 + gid];
            const int pair0 = pairs[nt * 8 + quarter * 2], pair1 = pairs[nt * 8 + quarter * 2 + 1];
            const int load_input = Middle ? load_pair / kSlots : load_pair;
            const int input0 = Middle ? pair0 / kSlots : pair0;
            const int input1 = Middle ? pair1 / kSlots : pair1;
            unsigned x0 = 0, x1 = 0;
            if (load_pair >= 0) {
                const auto* x = reinterpret_cast<const unsigned*>(q + std::int64_t(load_input) * K + column);
                x0 = x[0]; x1 = x[1];
            }
            const float xs0 = pair0 < 0 ? 0 : scales[std::int64_t(input0) * (K / 32) + group];
            const float xs1 = pair1 < 0 ? 0 : scales[std::int64_t(input1) * (K / 32) + group];
            int sums[4] = {};
            mma_s8(sums[0], sums[1], sums[2], sums[3], a0, a1, a2, a3, x0, x1);
            a[nt][0] = fmaf(float(sums[0]), w0 * xs0, a[nt][0]);
            a[nt][1] = fmaf(float(sums[1]), w0 * xs1, a[nt][1]);
            a[nt][2] = fmaf(float(sums[2]), w1 * xs0, a[nt][2]);
            a[nt][3] = fmaf(float(sums[3]), w1 * xs1, a[nt][3]);
            if constexpr (Fused) {
                int upper[4] = {};
                mma_s8(upper[0], upper[1], upper[2], upper[3], b0, b1, b2, b3, x0, x1);
                b[nt][0] = fmaf(float(upper[0]), u0 * xs0, b[nt][0]);
                b[nt][1] = fmaf(float(upper[1]), u0 * xs1, b[nt][1]);
                b[nt][2] = fmaf(float(upper[2]), u1 * xs0, b[nt][2]);
                b[nt][3] = fmaf(float(upper[3]), u1 * xs1, b[nt][3]);
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < kMmaColumns / 8; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int pair = pairs[nt * 8 + quarter * 2 + i % 2];
            if (pair < 0) { continue; }
            float result = a[nt][i];
            if constexpr (Fused) { result = result / (1.0f + expf(-result)) * b[nt][i]; }
            output[std::int64_t(pair) * N + row + (i / 2) * 8] = result;
        }
    }
}

template <QType Format, bool Shared, bool Middle, bool Fused>
void launch_grouped(const signed char* q, const float* scales, const Weight* first,
                     const Weight* second, RouteGroups groups, int tokens, float* output,
                     cudaStream_t stream) {
    constexpr int rows = Middle ? kWidth : kHidden;
    const int tiles = Shared ? (tokens + kMmaColumns - 1) / kMmaColumns : groups.capacity;
    grouped_mma<Format, Shared, Middle, Fused><<<tiles * (rows / 64), 128, 0, stream>>>(
        q, scales, first, second, groups, tokens, output);
    CUDA_CHECK(cudaGetLastError());
}

// Integer weight codes are exactly representable in BF16. Keep their FP16 scales outside
// each 32-value dot product instead of rounding the dequantized weights to BF16.
template <QType Format, int K>
__device__ unsigned weight_pair(const Weight& w, int row, int column, float& scale) {
    if constexpr (Format == QType::BF16) {
        scale = 1.0f;
        const auto* words = static_cast<const __nv_bfloat16*>(w.qdata);
        return *reinterpret_cast<const unsigned*>(words + std::int64_t(row) * K + column);
    } else {
        const unsigned packed = unsigned(packed_four<Format, K>(w, row, column & ~3, scale));
        const int shift = (column & 2) * 8;
        const float a = float(static_cast<signed char>((packed >> shift) & 255));
        const float b = float(static_cast<signed char>((packed >> (shift + 8)) & 255));
        return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
               (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
    }
}

__device__ unsigned bf16_pair(float a, float b) {
    return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
           (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
}

template <bool Middle>
__device__ void activation_pair(const void* input, std::int64_t at, unsigned& high, unsigned& low) {
    if constexpr (Middle) {
        high = *reinterpret_cast<const unsigned*>(static_cast<const __nv_bfloat16*>(input) + at);
        low = 0;
    } else {
        const auto* x = static_cast<const float*>(input);
        const float a = x[at], b = x[at + 1];
        high = bf16_pair(a, b);
        const float ah = __bfloat162float(__ushort_as_bfloat16(high & 65535));
        const float bh = __bfloat162float(__ushort_as_bfloat16(high >> 16));
        // The FP32 SwiGLU product is private arithmetic. Two BF16 components retain its
        // low part instead of introducing a new single-BF16 materialization boundary.
        low = bf16_pair(a - ah, b - bh);
    }
}

template <QType Format, bool Shared, bool Middle, bool Fused>
__global__ void grouped_a16(const void* input, const Weight* first, const Weight* second,
                            RouteGroups groups, int tokens, float* output) {
    constexpr int K = Middle ? kHidden : kWidth;
    constexpr int N = Middle ? kWidth : kHidden;
    constexpr int row_tiles = N / 64;
    const int tile = int(blockIdx.x) / row_tiles;
    if constexpr (Shared) {
        if (tile >= (tokens + kMmaColumns - 1) / kMmaColumns) { return; }
    } else {
        if (tile >= *groups.total_tiles) { return; }
    }
    const int2 selected = Shared ? make_int2(0, tile * kMmaColumns) : groups.tiles[tile];
    __shared__ int pairs[kMmaColumns];
    if (threadIdx.x < kMmaColumns) {
        const int at = selected.y + threadIdx.x;
        if constexpr (Shared) {
            pairs[threadIdx.x] = at < tokens ? at * kSlots + kTop : -1;
        } else {
            pairs[threadIdx.x] = at < groups.offsets[selected.x] + groups.counts[selected.x]
                ? groups.pairs[at] : -1;
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, gid = lane / 4, quarter = lane % 4;
    const int row = (int(blockIdx.x) % row_tiles) * 64 + (threadIdx.x / 32) * 16 + gid;
    float a[kMmaColumns / 8][4] = {}, b[kMmaColumns / 8][4] = {};
    for (int group = 0; group < K / 32; ++group) {
        float sums[kMmaColumns / 8][4] = {}, upper[kMmaColumns / 8][4] = {};
        float w0 = 1, w1 = 1, u0 = 1, u1 = 1, unused;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int column = group * 32 + half * 16 + quarter * 2;
            const unsigned a0 = weight_pair<Format, K>(first[selected.x], row, column, w0);
            const unsigned a1 = weight_pair<Format, K>(first[selected.x], row + 8, column, w1);
            const unsigned a2 = weight_pair<Format, K>(first[selected.x], row, column + 8, unused);
            const unsigned a3 = weight_pair<Format, K>(first[selected.x], row + 8, column + 8, unused);
            unsigned b0 = 0, b1 = 0, b2 = 0, b3 = 0;
            if constexpr (Fused) {
                b0 = weight_pair<Format, K>(second[selected.x], row, column, u0);
                b1 = weight_pair<Format, K>(second[selected.x], row + 8, column, u1);
                b2 = weight_pair<Format, K>(second[selected.x], row, column + 8, unused);
                b3 = weight_pair<Format, K>(second[selected.x], row + 8, column + 8, unused);
            }
#pragma unroll
            for (int nt = 0; nt < kMmaColumns / 8; ++nt) {
                if (pairs[nt * 8] < 0) { continue; }
                const int pair = pairs[nt * 8 + gid];
                unsigned x0 = 0, x1 = 0, l0 = 0, l1 = 0;
                if (pair >= 0) {
                    const int source = Middle ? pair / kSlots : pair;
                    const auto at = std::int64_t(source) * K + column;
                    activation_pair<Middle>(input, at, x0, l0);
                    activation_pair<Middle>(input, at + 8, x1, l1);
                }
                mma_bf16(sums[nt][0], sums[nt][1], sums[nt][2], sums[nt][3],
                         a0, a1, a2, a3, x0, x1);
                if constexpr (!Middle) {
                    mma_bf16(sums[nt][0], sums[nt][1], sums[nt][2], sums[nt][3],
                             a0, a1, a2, a3, l0, l1);
                }
                if constexpr (Fused) {
                    mma_bf16(upper[nt][0], upper[nt][1], upper[nt][2], upper[nt][3],
                             b0, b1, b2, b3, x0, x1);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < kMmaColumns / 8; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[nt][i] = fmaf(sums[nt][i], i < 2 ? w0 : w1, a[nt][i]);
                if constexpr (Fused) {
                    b[nt][i] = fmaf(upper[nt][i], i < 2 ? u0 : u1, b[nt][i]);
                }
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < kMmaColumns / 8; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int pair = pairs[nt * 8 + quarter * 2 + i % 2];
            if (pair < 0) { continue; }
            float result = a[nt][i];
            if constexpr (Fused) { result = result / (1.0f + expf(-result)) * b[nt][i]; }
            output[std::int64_t(pair) * N + row + (i / 2) * 8] = result;
        }
    }
}

template <QType Format, bool Shared, bool Middle, bool Fused>
void launch_grouped_a16(const void* input, const Weight* first, const Weight* second,
                         RouteGroups groups, int tokens, float* output, cudaStream_t stream) {
    constexpr int rows = Middle ? kWidth : kHidden;
    const int tiles = Shared ? (tokens + kMmaColumns - 1) / kMmaColumns : groups.capacity;
    grouped_a16<Format, Shared, Middle, Fused><<<tiles * (rows / 64), 128, 0, stream>>>(
        input, first, second, groups, tokens, output);
    CUDA_CHECK(cudaGetLastError());
}

template <>
__device__ float value(const __nv_bfloat16* x, int index) { return __bfloat162float(x[index]); }

template <typename Source, int K>
__global__ void quantize_groups(const Source* x, int columns, signed char* codes, float* scales) {
    const int lane = threadIdx.x & 31;
    const int group = int(blockIdx.x) * kWarps + int(threadIdx.x) / 32;
    if (group >= columns * (K / 32)) { return; }
    const float v = value(x, group * 32 + lane);
    float maximum = fabsf(v);
#pragma unroll
    for (int shift = 16; shift; shift >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, shift));
    }
    const float scale = maximum / 127.0f;
    codes[group * 32 + lane] = static_cast<signed char>(
        scale == 0.0f ? 0 : max(-127, min(127, __float2int_rn(v / scale))));
    if (lane == 0) { scales[group] = scale; }
}

template <QType Format, int K, bool A8, typename Source>
__device__ float dot(const Weight& w, int row, const Source* x, const signed char* q,
                     const float* x_scales, int lane) {
    float sum = 0;
    for (int column = lane * 4; column < K; column += 128) {
        if constexpr (Format == QType::BF16) {
            const auto* weights = static_cast<const __nv_bfloat16*>(w.qdata) + std::int64_t(row) * K;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum = fmaf(__bfloat162float(weights[column + i]), value(x, column + i), sum);
            }
        } else {
            float weight_scale;
            const int words = packed_four<Format, K>(w, row, column, weight_scale);
            if constexpr (A8) {
                const int input = *reinterpret_cast<const int*>(q + column);
                sum = fmaf(float(__dp4a(words, input, 0)),
                           weight_scale * x_scales[column / 32], sum);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int code = static_cast<signed char>((unsigned(words) >> (i * 8)) & 255);
                    sum = fmaf(float(code) * weight_scale, value(x, column + i), sum);
                }
            }
        }
    }
    return warp_sum(sum);
}

template <QType Format, bool A8, bool Shared>
__global__ void middle_kernel(const __nv_bfloat16* m, const int* ids, const Weight* gate,
                             const Weight* up, const signed char* q, const float* scales,
                             float* middle) {
    const int row = int(blockIdx.x) * kWarps + int(threadIdx.x) / 32;
    const int slot = Shared ? kTop : int(blockIdx.y), token = int(blockIdx.z);
    const int expert = Shared ? 0 : ids[token * kTop + slot];
    const auto* x = m + std::int64_t(token) * kHidden;
    if (expert < 0) {
        if ((threadIdx.x & 31) == 0) {
            middle[(std::int64_t(token) * kSlots + slot) * kWidth + row] = 0;
        }
        return;
    }
    const auto* codes = q ? q + std::int64_t(token) * kHidden : nullptr;
    const auto* xs = scales ? scales + std::int64_t(token) * (kHidden / 32) : nullptr;
    const float g = dot<Format, kHidden, A8>(gate[expert], row, x, codes, xs, threadIdx.x & 31);
    const float u = dot<Format, kHidden, A8>(up[expert], row, x, codes, xs, threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) {
        middle[(std::int64_t(token) * kSlots + slot) * kWidth + row] =
            g / (1.0f + expf(-g)) * u;
    }
}

template <QType Format, bool A8, bool Shared>
__global__ void down_kernel(const int* ids, const Weight* down, const float* middle,
                           const signed char* q, const float* scales, float* products) {
    const int row = int(blockIdx.x) * kWarps + int(threadIdx.x) / 32;
    const int slot = Shared ? kTop : int(blockIdx.y), token = int(blockIdx.z);
    const int expert = Shared ? 0 : ids[token * kTop + slot];
    const std::int64_t pair = std::int64_t(token) * kSlots + slot;
    if (expert < 0) {
        if ((threadIdx.x & 31) == 0) { products[pair * kHidden + row] = 0; }
        return;
    }
    const float sum = dot<Format, kWidth, A8>(down[expert], row, middle + pair * kWidth,
        q ? q + pair * kWidth : nullptr, scales ? scales + pair * (kWidth / 32) : nullptr,
        threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) { products[pair * kHidden + row] = sum; }
}

template <QType Format, bool A8, bool Shared>
__global__ void projection_kernel(const __nv_bfloat16* m, const int* ids, const Weight* bank,
                                 const signed char* q, const float* scales, float* output) {
    const int row = int(blockIdx.x) * kWarps + int(threadIdx.x) / 32;
    const int slot = Shared ? kTop : int(blockIdx.y), token = int(blockIdx.z);
    const int expert = Shared ? 0 : ids[token * kTop + slot];
    if (expert < 0) {
        if ((threadIdx.x & 31) == 0) {
            output[(std::int64_t(token) * kSlots + slot) * kWidth + row] = 0;
        }
        return;
    }
    const float sum = dot<Format, kHidden, A8>(bank[expert], row,
        m + std::int64_t(token) * kHidden, q ? q + std::int64_t(token) * kHidden : nullptr,
        scales ? scales + std::int64_t(token) * (kHidden / 32) : nullptr, threadIdx.x & 31);
    if ((threadIdx.x & 31) == 0) {
        output[(std::int64_t(token) * kSlots + slot) * kWidth + row] = sum;
    }
}

template <bool Shared>
__global__ void activate_middle(float* gate, const float* up, int tokens) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr int slots = Shared ? 1 : kTop;
    if (i >= tokens * slots * kWidth) { return; }
    const int token = i / (slots * kWidth), slot = Shared ? kTop : i / kWidth % kTop;
    const std::int64_t at = (std::int64_t(token) * kSlots + slot) * kWidth + i % kWidth;
    const float g = gate[at];
    gate[at] = g / (1.0f + expf(-g)) * up[at];
}

template <QType Format, bool Shared>
void launch_projection(const Tensor& m, const Tensor& ids, const NativeExpertTable& bank,
                       const signed char* q, const float* scales, float* output,
                       RouteGroups groups, cudaStream_t stream) {
    if (!bank.integer_a8 && m.ne[1] >= kMmaTokens) {
        return launch_grouped_a16<Format, Shared, true, false>(
            m.data, bank.experts, nullptr, groups, m.ne[1], output, stream);
    }
    if constexpr (Format != QType::BF16) {
        if (bank.integer_a8 && m.ne[1] >= kMmaTokens) {
            return launch_grouped<Format, Shared, true, false>(
                q, scales, bank.experts, nullptr, groups, m.ne[1], output, stream);
        }
    }
    const dim3 grid(kWidth / kWarps, Shared ? 1 : kTop, m.ne[1]);
    if (bank.integer_a8) {
        projection_kernel<Format, true, Shared><<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(m.data), static_cast<const int*>(ids.data),
            bank.experts, q, scales, output);
    } else {
        projection_kernel<Format, false, Shared><<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(m.data), static_cast<const int*>(ids.data),
            bank.experts, nullptr, nullptr, output);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Shared>
void dispatch_projection(const Tensor& m, const Tensor& ids, const NativeExpertTable& bank,
                         const signed char* q, const float* scales, float* output,
                         RouteGroups groups, cudaStream_t stream) {
#define PROJECTION_CASE(qtype) case QType::qtype: \
    return launch_projection<QType::qtype, Shared>(m, ids, bank, q, scales, output, groups, stream)
    switch (bank.format) {
        PROJECTION_CASE(BF16);
        PROJECTION_CASE(Q2_G64_FP16);
        PROJECTION_CASE(Q4_G64_FP16);
        PROJECTION_CASE(Q5_G64_FP16);
        PROJECTION_CASE(Q6_G64_FP16);
        PROJECTION_CASE(Q8_G32_FP16);
        default: throw std::invalid_argument("moe_experts_native: unsupported projection format");
    }
#undef PROJECTION_CASE
}

__global__ void merge_kernel(const float* products, const float* weights, const float* shared,
                            const unsigned char* parts, std::int64_t count, float* y) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t token = i / kHidden, row = i % kHidden;
    float sum = 0;
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
        if (slot < kTop && parts && !parts[token * kTop + slot]) { continue; }
        const float scale = slot == kTop ? shared[token] : weights[token * kTop + slot];
        sum = fmaf(scale, products[(token * kSlots + slot) * kHidden + row], sum);
    }
    y[i] = sum;
}

template <QType Format, bool Shared>
void launch_bank(const Tensor& m, const Tensor& ids, const NativeExpertTable& gate,
                 const NativeExpertTable& up, const NativeExpertTable& down, float* middle,
                 float* products, const signed char* input_q, const float* input_scales,
                 signed char* middle_q, float* middle_scales, RouteGroups groups,
                 cudaStream_t stream, bool middle_pass) {
    if (m.ne[1] >= kMmaTokens) {
        if (middle_pass && !gate.integer_a8) {
            return launch_grouped_a16<Format, Shared, true, true>(
                m.data, gate.experts, up.experts, groups, m.ne[1], middle, stream);
        }
        if (!middle_pass && !down.integer_a8) {
            return launch_grouped_a16<Format, Shared, false, false>(
                middle, down.experts, nullptr, groups, m.ne[1], products, stream);
        }
    }
    if constexpr (Format != QType::BF16) {
        if (m.ne[1] >= kMmaTokens) {
            if (middle_pass && gate.integer_a8) {
                return launch_grouped<Format, Shared, true, true>(
                    input_q, input_scales, gate.experts, up.experts, groups, m.ne[1], middle, stream);
            }
            if (!middle_pass && down.integer_a8) {
                return launch_grouped<Format, Shared, false, false>(
                    middle_q, middle_scales, down.experts, nullptr, groups, m.ne[1], products, stream);
            }
        }
    }
    const dim3 grid(middle_pass ? kWidth / kWarps : kHidden / kWarps, Shared ? 1 : kTop, m.ne[1]);
    if (middle_pass) {
        if (gate.integer_a8) {
            middle_kernel<Format, true, Shared><<<grid, kWarps * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(m.data), static_cast<const int*>(ids.data),
                gate.experts, up.experts, input_q, input_scales, middle);
        } else {
            middle_kernel<Format, false, Shared><<<grid, kWarps * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(m.data), static_cast<const int*>(ids.data),
                gate.experts, up.experts, nullptr, nullptr, middle);
        }
    } else if (down.integer_a8) {
        down_kernel<Format, true, Shared><<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const int*>(ids.data), down.experts, middle, middle_q, middle_scales, products);
    } else {
        down_kernel<Format, false, Shared><<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const int*>(ids.data), down.experts, middle, nullptr, nullptr, products);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Shared>
void dispatch(const Tensor& m, const Tensor& ids, const NativeExpertTable& gate,
              const NativeExpertTable& up, const NativeExpertTable& down, float* middle,
              float* products, const signed char* input_q, const float* input_scales,
              signed char* middle_q, float* middle_scales, RouteGroups groups,
              cudaStream_t stream, bool middle_pass) {
    if (middle_pass && (gate.format != up.format || gate.integer_a8 != up.integer_a8)) {
        dispatch_projection<Shared>(m, ids, gate, input_q, input_scales, middle, groups, stream);
        // Down products are not live until both middle passes finish. Reuse their storage.
        dispatch_projection<Shared>(m, ids, up, input_q, input_scales, products, groups, stream);
        const int count = m.ne[1] * (Shared ? 1 : kTop) * kWidth;
        activate_middle<Shared><<<(count + 255) / 256, 256, 0, stream>>>(middle, products, m.ne[1]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const QType format = middle_pass ? gate.format : down.format;
#define NATIVE_CASE(qtype) case QType::qtype: \
    return launch_bank<QType::qtype, Shared>(m, ids, gate, up, down, middle, products, input_q, \
                                      input_scales, middle_q, middle_scales, groups, stream, middle_pass)
    switch (format) {
        NATIVE_CASE(BF16);
        NATIVE_CASE(Q2_G64_FP16);
        NATIVE_CASE(Q4_G64_FP16);
        NATIVE_CASE(Q5_G64_FP16);
        NATIVE_CASE(Q6_G64_FP16);
        NATIVE_CASE(Q8_G32_FP16);
        default: throw std::invalid_argument("moe_experts_native: unsupported format");
    }
#undef NATIVE_CASE
}

void validate(const NativeExpertTable& bank) {
    require(bank.experts && supported(bank.format), "unsupported or empty native expert bank");
    require(!bank.integer_a8 || bank.format != QType::BF16, "BF16 experts require A16 compute");
}

} // namespace

Weight prepare_native_expert(const WeightInput& input, int rows, int columns, bool integer_a8) {
    require((rows == kWidth && columns == kHidden) || (rows == kHidden && columns == kWidth),
            "expert role must be [640,2560] or [2560,640]");
    require(valid_linear_policy(input.policy) && !input.hadamard_signs.data &&
                !input.input_columns.data, "native expert requires a valid, unrotated input use");
    require(input.weight.shape == std::vector<std::uint64_t>{std::uint64_t(rows), std::uint64_t(columns)},
            "expert projection shape differs from its registered role");
    const Weight w = native_weight(input.weight);
    require(supported(w.qtype), "native experts require BF16 or Q2/Q4/Q5/Q6/Q8");
    if (w.qtype == QType::BF16) {
        require(w.layout == QuantLayout::Contiguous && !integer_a8,
                "BF16 expert requires contiguous A16 operands");
    } else {
        require(is_row_split(w.layout) && w.padded_shape[1] == columns && w.scales &&
                    ((w.qtype != QType::Q5_G64_FP16 && w.qtype != QType::Q6_G64_FP16) || w.qhigh),
                "native expert requires complete row-split planes");
    }
    require(!integer_a8 || input.policy != LinearPolicy::A16Only,
            "integer A8 expert arithmetic exceeds the source activation permission");
    return w;
}

std::size_t moe_experts_native_workspace_bytes(int tokens) {
    require(tokens > 0 && tokens <= 65535, "tokens must be in [1,65535]");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I32, {kTop, tokens}); // optional CPU/GPU route selection
    (void)layout.alloc(DType::FP32, {kWidth, kSlots, tokens});
    (void)layout.alloc(DType::FP32, {kHidden, kSlots, tokens});
    (void)layout.alloc(DType::I8, {kHidden, tokens});
    (void)layout.alloc(DType::FP32, {kHidden / 32, tokens});
    (void)layout.alloc(DType::I8, {kWidth, kSlots, tokens});
    (void)layout.alloc(DType::FP32, {kWidth / 32, kSlots, tokens});
    if (tokens >= kMmaTokens) {
        (void)layout.alloc(DType::I32, {kMaxExperts * 3 + 1});
        (void)layout.alloc(DType::I32, {tokens * kTop});
        (void)layout.alloc(DType::I32, {tile_capacity(tokens) * 2});
    }
    return layout.peak_bytes(1);
}

void moe_experts_native(const Tensor& m, const Tensor& ids, const Tensor& weights,
                        const Tensor& shared, const NativeMoeWeights& banks,
                        const Tensor* parts, WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    const int tokens = m.ne[1];
    require(tokens > 0 && tokens <= 65535 && shaped(m, DType::BF16, kHidden, tokens),
            "m must be BF16 [2560,tokens], tokens in [1,65535]");
    require(shaped(ids, DType::I32, kTop, tokens) && shaped(weights, DType::FP32, kTop, tokens),
            "route ids and weights must be contiguous I32/FP32 [10,tokens]");
    require(!parts || shaped(*parts, DType::U8, kTop, tokens), "parts must be U8 [10,tokens]");
    require(shaped(shared, DType::FP32, tokens, 1) && shaped(y, DType::FP32, kHidden, tokens),
            "shared/output must be FP32 [tokens]/[2560,tokens]");
    require(banks.experts >= kTop && banks.experts <= 512, "expert count must be in [10,512]");
    for (const auto* bank : {&banks.gate, &banks.up, &banks.down, &banks.shared_gate,
                             &banks.shared_up, &banks.shared_down}) { validate(*bank); }
    auto scope = workspace.scope();
    Tensor selected = ids;
    if (parts) {
        selected = workspace.alloc(DType::I32, {kTop, tokens});
        select_parts<<<(tokens * kTop + 255) / 256, 256, 0, stream>>>(
            static_cast<const int*>(ids.data), static_cast<const unsigned char*>(parts->data),
            tokens * kTop, static_cast<int*>(selected.data));
        CUDA_CHECK(cudaGetLastError());
    }
    Tensor middle = workspace.alloc(DType::FP32, {kWidth, kSlots, tokens});
    Tensor products = workspace.alloc(DType::FP32, {kHidden, kSlots, tokens});
    Tensor iq = workspace.alloc(DType::I8, {kHidden, tokens});
    Tensor is = workspace.alloc(DType::FP32, {kHidden / 32, tokens});
    Tensor mq = workspace.alloc(DType::I8, {kWidth, kSlots, tokens});
    Tensor ms = workspace.alloc(DType::FP32, {kWidth / 32, kSlots, tokens});
    auto* middle_p = static_cast<float*>(middle.data);
    auto* products_p = static_cast<float*>(products.data);
    auto* input_q = static_cast<signed char*>(iq.data);
    auto* input_scales = static_cast<float*>(is.data);
    auto* middle_q = static_cast<signed char*>(mq.data);
    auto* middle_scales = static_cast<float*>(ms.data);
    if (parts && tokens >= kMmaTokens) {
        // Grouped kernels visit only included pairs; quantization also visits the excluded
        // middle rows. Initialize those rows without dereferencing any excluded weight.
        CUDA_CHECK(cudaMemsetAsync(middle_p, 0, std::size_t(kWidth) * kSlots * tokens * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(products_p, 0, std::size_t(kHidden) * kSlots * tokens * sizeof(float), stream));
    }
    RouteGroups groups;
    if (tokens >= kMmaTokens) {
        Tensor metadata = workspace.alloc(DType::I32, {kMaxExperts * 3 + 1});
        Tensor pairs = workspace.alloc(DType::I32, {tokens * kTop});
        Tensor tiles = workspace.alloc(DType::I32, {tile_capacity(tokens) * 2});
        auto* data = static_cast<int*>(metadata.data);
        groups = {data, data + kMaxExperts, data + 2 * kMaxExperts, data + 3 * kMaxExperts,
                  static_cast<int*>(pairs.data), static_cast<int2*>(tiles.data), tile_capacity(tokens)};
        CUDA_CHECK(cudaMemsetAsync(groups.counts, 0, kMaxExperts * sizeof(int), stream));
        count_routes<<<(tokens * kTop + 255) / 256, 256, 0, stream>>>(
            static_cast<const int*>(selected.data), tokens * kTop, groups.counts);
        plan_groups<<<1, kMaxExperts, 0, stream>>>(groups, banks.experts);
        scatter_routes<<<(tokens * kTop + 255) / 256, 256, 0, stream>>>(
            static_cast<const int*>(selected.data), tokens * kTop, groups);
        CUDA_CHECK(cudaGetLastError());
    }
    if (banks.gate.integer_a8 || banks.up.integer_a8 ||
        banks.shared_gate.integer_a8 || banks.shared_up.integer_a8) {
        const int groups = tokens * (kHidden / 32);
        quantize_groups<__nv_bfloat16, kHidden><<<(groups + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(m.data), tokens, input_q, input_scales);
        CUDA_CHECK(cudaGetLastError());
    }
    dispatch<false>(m, selected, banks.gate, banks.up, banks.down, middle_p, products_p,
                    input_q, input_scales, middle_q, middle_scales, groups, stream, true);
    dispatch<true>(m, ids, banks.shared_gate, banks.shared_up, banks.shared_down, middle_p,
                   products_p, input_q, input_scales, middle_q, middle_scales, groups, stream, true);
    if (banks.down.integer_a8 || banks.shared_down.integer_a8) {
        const int groups = tokens * kSlots * (kWidth / 32);
        quantize_groups<float, kWidth><<<(groups + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(
            middle_p, tokens * kSlots, middle_q, middle_scales);
        CUDA_CHECK(cudaGetLastError());
    }
    dispatch<false>(m, selected, banks.gate, banks.up, banks.down, middle_p, products_p,
                    input_q, input_scales, middle_q, middle_scales, groups, stream, false);
    dispatch<true>(m, ids, banks.shared_gate, banks.shared_up, banks.shared_down, middle_p,
                   products_p, input_q, input_scales, middle_q, middle_scales, groups, stream, false);
    const std::int64_t count = std::int64_t(kHidden) * tokens;
    merge_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(products_p,
        static_cast<const float*>(weights.data), static_cast<const float*>(shared.data),
        parts ? static_cast<const unsigned char*>(parts->data) : nullptr, count,
        static_cast<float*>(y.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
