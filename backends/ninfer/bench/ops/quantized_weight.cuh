#pragma once

#include "core/weight.h"
#include "ninfer_bench_common.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::bench {

// Benchmark weights are format-correct pseudo-random codes and scales, reproducible from a seed.
// A constant byte pattern compresses, caches and toggles differently from a real checkpoint, and
// on Ampere constant data also draws less power, which can raise clocks: route tuning measured on
// it was measured on a workload the Engine never runs. `extent` is the largest logical |weight|.
inline constexpr std::uint64_t kDefaultWeightSeed = 0x51U;
inline constexpr float kDefaultWeightExtent       = 0.05F;

struct PackedQuantizedWeight {
    DeviceBuffer storage;
    Weight weight{};
    std::uint64_t low_bytes    = 0;
    std::uint64_t high_offset  = 0;
    std::uint64_t high_bytes   = 0;
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_bytes  = 0;

    [[nodiscard]] std::uint64_t model_weight_bytes() const noexcept {
        return low_bytes + high_bytes + scale_bytes;
    }
};

namespace detail {

struct QuantizedGeometry {
    std::int32_t group_size;
    std::int32_t high_bytes_per_group;
};

inline QuantizedGeometry quantized_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 0};
    case QType::Q5_G64_FP16:
        return {64, 8};
    case QType::Q6_G64_FP16:
        return {64, 16};
    case QType::Q8_G32_FP16:
        return {32, 0};
    case QType::T2_G128_FP16:
        return {128, 0};
    default:
        throw std::invalid_argument("unsupported benchmark quantized format");
    }
}

inline std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(label);
    }
    return left * right;
}

inline std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw std::overflow_error("benchmark quantized alignment overflow");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ inline std::uint32_t weight_bits(std::uint64_t index, std::uint64_t seed) {
    std::uint64_t value = index ^ (seed * 0x9e3779b97f4a7c15ULL + 0x632be59bd9b4e019ULL);
    value               = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value               = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<std::uint32_t>((value ^ (value >> 31)) >> 32);
}

__device__ inline float weight_uniform(std::uint64_t index, std::uint64_t seed, float low,
                                       float high) {
    return low + (high - low) * (static_cast<float>(weight_bits(index, seed) >> 8) / 16777216.0F);
}

// One thread per group: draw the group's signed logical codes once, then write the low plane
// (Q8 bytes, T2 two-bit fields, or nibbles) and the high plane (Q5 one bit, Q6 two bits per code)
// in the RowSplit layout. Codes past the logical K are zero, as the converter pads them.
static __global__ void fill_row_split_kernel(std::uint8_t* low, std::uint8_t* high, __half* scales,
                                             std::uint64_t groups, int k, int padded_k, int bits,
                                             int group_size, std::uint64_t seed, float extent) {
    const int limit      = bits == 2 ? 1 : (1 << (bits - 1)) - 1;
    const int high_bytes = bits == 5 ? 8 : bits == 6 ? 16 : 0;
    for (auto g = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < groups;
         g += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const int start =
            static_cast<int>(g % static_cast<std::uint64_t>(padded_k / group_size)) * group_size;
        std::uint8_t codes[128];
        for (int j = 0; j < group_size; ++j) {
            const int code = start + j < k
                                 ? static_cast<int>(weight_bits(g * group_size + j, seed) %
                                                    static_cast<std::uint32_t>(2 * limit + 1)) -
                                       limit
                                 : 0;
            codes[j] = static_cast<std::uint8_t>(static_cast<unsigned>(code) & ((1U << bits) - 1));
        }
        std::uint8_t* low_group = low + g * 32;
        for (int j = 0; j < 32; ++j) {
            if (bits == 8) {
                low_group[j] = codes[j];
            } else if (bits == 2) {
                low_group[j] =
                    static_cast<std::uint8_t>(codes[4 * j] | (codes[4 * j + 1] << 2) |
                                              (codes[4 * j + 2] << 4) | (codes[4 * j + 3] << 6));
            } else {
                low_group[j] = static_cast<std::uint8_t>((codes[2 * j] & 15U) |
                                                         ((codes[2 * j + 1] & 15U) << 4));
            }
        }
        for (int j = 0; j < high_bytes; ++j) {
            const int per_byte = 8 / (bits - 4);
            unsigned packed    = 0;
            for (int lane = 0; lane < per_byte; ++lane) {
                packed |= static_cast<unsigned>(codes[j * per_byte + lane] >> 4)
                          << (lane * (bits - 4));
            }
            high[g * high_bytes + j] = static_cast<std::uint8_t>(packed);
        }
        scales[g] = __float2half_rn(extent / static_cast<float>(limit) *
                                    weight_uniform(g, seed ^ 0x51ca1eU, 0.5F, 1.5F));
    }
}

static __global__ void fill_fp8_weight_kernel(std::uint8_t* codes, __nv_bfloat16* scales,
                                              std::uint64_t count, int rows, std::uint64_t seed,
                                              float extent) {
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        codes[i] = __nv_fp8_e4m3(weight_uniform(i, seed, -224.0F, 224.0F)).__x;
        if (i < static_cast<std::uint64_t>(rows)) {
            scales[i] = __float2bfloat16_rn(extent / 224.0F *
                                            weight_uniform(i, seed ^ 0x51ca1eU, 0.5F, 1.5F));
        }
    }
}

// Codes are any E2M1 nibble; each 16-element group's E4M3 scale lands in the BlockScaleK16M128x4
// swizzle at the position the converter writes it.
static __global__ void fill_nvfp4_weight_kernel(std::uint8_t* codes, std::uint8_t* scales,
                                                std::uint64_t elements, int k, std::uint64_t seed,
                                                float extent, float divisor) {
    for (auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements / 2; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        codes[i] = static_cast<std::uint8_t>((weight_bits(2 * i, seed) & 15U) |
                                             ((weight_bits(2 * i + 1, seed) & 15U) << 4));
        if (i < elements / 16) {
            const std::uint64_t row   = i / static_cast<std::uint64_t>(k / 16);
            const std::uint64_t group = i % static_cast<std::uint64_t>(k / 16);
            const std::uint64_t offset =
                ((row / 128) * static_cast<std::uint64_t>(k / 64) + group / 4) * 512 +
                (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
            scales[offset] = __nv_fp8_e4m3(extent * divisor / 6.0F *
                                           weight_uniform(i, seed ^ 0x51ca1eU, 0.5F, 1.5F))
                                 .__x;
        }
    }
}

inline int launch_grid(std::uint64_t elements) {
    return static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (elements + 255) / 256)));
}

} // namespace detail

inline PackedQuantizedWeight make_row_split_weight(QType qtype, std::int32_t n, std::int32_t k,
                                                   std::int32_t padded_k,
                                                   std::uint64_t seed = kDefaultWeightSeed,
                                                   float extent       = kDefaultWeightExtent) {
    const detail::QuantizedGeometry geometry = detail::quantized_geometry(qtype);
    if (n <= 0 || k <= 0 || padded_k < k || padded_k % geometry.group_size != 0) {
        throw std::invalid_argument("invalid benchmark RowSplit weight shape");
    }

    const std::uint64_t groups = detail::checked_mul(
        static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(padded_k / geometry.group_size),
        "benchmark weight group count overflow");
    const std::uint64_t low_bytes =
        detail::checked_mul(groups, 32, "benchmark low plane size overflow");
    const std::uint64_t high_bytes =
        detail::checked_mul(groups, static_cast<std::uint64_t>(geometry.high_bytes_per_group),
                            "benchmark high plane size overflow");
    const std::uint64_t scale_bytes =
        detail::checked_mul(groups, 2, "benchmark scale plane size overflow");
    const std::uint64_t high_offset  = detail::align_up(low_bytes, 256);
    const std::uint64_t scale_offset = detail::checked_add(
        high_offset, detail::align_up(high_bytes, 256), "benchmark scale plane offset overflow");
    const std::uint64_t payload_bytes =
        detail::checked_add(scale_offset, scale_bytes, "benchmark payload size overflow");
    if (payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark payload does not fit size_t");
    }

    PackedQuantizedWeight result{
        DeviceBuffer(static_cast<std::size_t>(payload_bytes)),
        {},
        low_bytes,
        high_offset,
        high_bytes,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.p, 0, result.storage.bytes));
    const int bits = qtype == QType::Q4_G64_FP16    ? 4
                     : qtype == QType::Q5_G64_FP16  ? 5
                     : qtype == QType::Q6_G64_FP16  ? 6
                     : qtype == QType::T2_G128_FP16 ? 2
                                                    : 8;
    detail::fill_row_split_kernel<<<detail::launch_grid(groups), 256>>>(
        static_cast<std::uint8_t*>(result.storage.p),
        high_bytes != 0 ? static_cast<std::uint8_t*>(result.storage.p) + high_offset : nullptr,
        reinterpret_cast<__half*>(static_cast<std::uint8_t*>(result.storage.p) + scale_offset),
        groups, k, padded_k, bits, geometry.group_size, seed, extent);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.payload          = result.storage.p;
    weight.payload_bytes    = payload_bytes;
    weight.high_plane_bytes = high_bytes;
    weight.qtype            = qtype;
    weight.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = padded_k;
    weight.ndim             = 2;
    weight.qdata            = result.storage.p;
    weight.qhigh            = high_bytes == 0
                                  ? nullptr
                                  : static_cast<const std::uint8_t*>(result.storage.p) + result.high_offset;
    weight.scales      = static_cast<const std::uint8_t*>(result.storage.p) + result.scale_offset;
    weight.n           = n;
    weight.k           = k;
    weight.group       = geometry.group_size;
    weight.layout      = QuantLayout::RowSplit;
    weight.scale_dtype = DType::FP16;
    return result;
}

// `divisor_rows` is how many consecutive rows one stored divisor covers: the whole plane for a
// matrix quantised on its own, and one source matrix's share for a plane stacked from several.
inline PackedQuantizedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                               std::int32_t divisor_rows = 0,
                                               std::uint64_t seed        = kDefaultWeightSeed,
                                               float extent              = kDefaultWeightExtent) {
    if (divisor_rows == 0) { divisor_rows = n; }
    if (n <= 0 || k <= 0 || (n % 128) != 0 || (k % 64) != 0 || divisor_rows <= 0 ||
        (n % divisor_rows) != 0 || (divisor_rows % 128) != 0) {
        throw std::invalid_argument("invalid benchmark NVFP4 weight shape");
    }
    const std::int32_t divisors = n / divisor_rows;
    const std::uint64_t elements =
        detail::checked_mul(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k),
                            "benchmark NVFP4 element count overflow");
    const std::uint64_t code_bytes   = elements / 2;
    const std::uint64_t scale_offset = detail::align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = elements / 16;
    const std::uint64_t divisor_offset =
        detail::checked_add(scale_offset, scale_bytes, "benchmark NVFP4 divisor offset overflow");
    const std::uint64_t payload_bytes =
        detail::checked_add(divisor_offset, static_cast<std::uint64_t>(divisors) * sizeof(float),
                            "benchmark NVFP4 payload size overflow");
    if (payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark NVFP4 payload does not fit size_t");
    }

    PackedQuantizedWeight result{
        DeviceBuffer(static_cast<std::size_t>(payload_bytes)),
        {},
        code_bytes,
        0,
        0,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.p, 0, result.storage.bytes));
    // A published checkpoint quantises every source matrix on its own, so the divisors differ.
    // Equal ones would let a per-row lookup pass while reading only the first word. The base of
    // 128 keeps the E4M3 group scales of extent-sized weights normal (about 1.07, not subnormal).
    constexpr float kWeightDivisor = 128.0F;
    std::vector<float> divisor_words(static_cast<std::size_t>(divisors));
    for (std::int32_t index = 0; index < divisors; ++index) {
        divisor_words[static_cast<std::size_t>(index)] =
            kWeightDivisor * (1.0F + 0.75F * static_cast<float>(index % 5));
    }
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(result.storage.p) + divisor_offset,
                          divisor_words.data(), divisor_words.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    // The scales assume the first divisor; later planes' divisors only rescale their output.
    detail::fill_nvfp4_weight_kernel<<<detail::launch_grid(code_bytes), 256>>>(
        static_cast<std::uint8_t*>(result.storage.p),
        static_cast<std::uint8_t*>(result.storage.p) + scale_offset, elements, k, seed, extent,
        kWeightDivisor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight              = result.weight;
    weight.payload              = result.storage.p;
    weight.payload_bytes        = payload_bytes;
    weight.qtype                = QType::NVFP4;
    weight.layout               = QuantLayout::BlockScaleK16M128x4;
    weight.scale_dtype          = DType::FP8_E4M3FN;
    weight.group_size           = 16;
    weight.group                = 16;
    weight.ndim                 = 2;
    weight.shape[0]             = n;
    weight.shape[1]             = k;
    weight.padded_shape[0]      = n;
    weight.padded_shape[1]      = k;
    weight.qdata                = result.storage.p;
    weight.qhigh                = nullptr;
    weight.scales               = static_cast<std::uint8_t*>(result.storage.p) + scale_offset;
    weight.n                    = n;
    weight.k                    = k;
    weight.weight_scale_divisor = divisor_words[0];
    weight.input_scale_divisor  = 3.5F;
    weight.weight_divisors      = static_cast<std::uint8_t*>(result.storage.p) + divisor_offset;
    weight.weight_divisor_rows  = divisor_rows;
    return result;
}

inline PackedQuantizedWeight make_fp8_weight(std::int32_t n, std::int32_t k,
                                             std::uint64_t seed = kDefaultWeightSeed,
                                             float extent       = kDefaultWeightExtent) {
    if (n <= 0 || k <= 0) { throw std::invalid_argument("invalid benchmark FP8 weight shape"); }
    const std::uint64_t code_bytes =
        detail::checked_mul(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k),
                            "benchmark FP8 code size overflow");
    const std::uint64_t scale_offset = detail::align_up(code_bytes, 256);
    const std::uint64_t scale_bytes =
        detail::checked_mul(static_cast<std::uint64_t>(n), 2, "benchmark FP8 scale size overflow");
    const std::uint64_t payload_bytes =
        detail::checked_add(scale_offset, scale_bytes, "benchmark FP8 payload size overflow");
    if (payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark FP8 payload does not fit size_t");
    }

    PackedQuantizedWeight result{
        DeviceBuffer(static_cast<std::size_t>(payload_bytes)),
        {},
        code_bytes,
        0,
        0,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.p, 0, result.storage.bytes));
    detail::fill_fp8_weight_kernel<<<detail::launch_grid(code_bytes), 256>>>(
        static_cast<std::uint8_t*>(result.storage.p),
        reinterpret_cast<__nv_bfloat16*>(static_cast<std::uint8_t*>(result.storage.p) +
                                         scale_offset),
        code_bytes, n, seed, extent);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.payload          = result.storage.p;
    weight.payload_bytes    = payload_bytes;
    weight.high_plane_bytes = 0;
    weight.qtype            = QType::FP8_E4M3FN_ROW_BF16;
    weight.layout           = QuantLayout::RowScale;
    weight.scale_dtype      = DType::BF16;
    weight.group_size       = static_cast<std::uint32_t>(k);
    weight.group            = k;
    weight.ndim             = 2;
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = k;
    weight.qdata            = result.storage.p;
    weight.qhigh            = nullptr;
    weight.scales           = static_cast<std::uint8_t*>(result.storage.p) + scale_offset;
    weight.n                = n;
    weight.k                = k;
    weight.scale_ne[0]      = n;
    weight.scale_nb[0]      = 2;
    weight.scale_nb[1]      = static_cast<std::int64_t>(n) * 2;
    weight.scale_nb[2]      = weight.scale_nb[1];
    weight.scale_nb[3]      = weight.scale_nb[1];
    return result;
}

inline Weight row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (parent.layout != QuantLayout::RowSplit || row_begin < 0 || rows <= 0 ||
        row_begin > parent.n - rows) {
        throw std::invalid_argument("benchmark RowSplit row view is out of range");
    }
    const detail::QuantizedGeometry geometry = detail::quantized_geometry(parent.qtype);
    const std::uint64_t groups_per_row =
        static_cast<std::uint64_t>(parent.padded_shape[1] / geometry.group_size);
    const std::uint64_t low_row_bytes = groups_per_row * 32;
    const std::uint64_t high_row_bytes =
        groups_per_row * static_cast<std::uint64_t>(geometry.high_bytes_per_group);
    const std::uint64_t scale_row_bytes = groups_per_row * 2;

    Weight view = parent;
    view.qdata  = static_cast<const std::uint8_t*>(parent.qdata) +
                 static_cast<std::uint64_t>(row_begin) * low_row_bytes;
    view.qhigh  = high_row_bytes == 0 ? nullptr
                                      : static_cast<const std::uint8_t*>(parent.qhigh) +
                                           static_cast<std::uint64_t>(row_begin) * high_row_bytes;
    view.scales = static_cast<const std::uint8_t*>(parent.scales) +
                  static_cast<std::uint64_t>(row_begin) * scale_row_bytes;
    view.n               = rows;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    return view;
}

} // namespace ninfer::bench
