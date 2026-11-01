#pragma once

#include "core/weight.h"
#include "ninfer/ops/moe_expert_cpu.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ninfer::ops::cpu_expert_detail {

struct Activations {
    const float* values;
    const std::int8_t* codes;
    const float* scales;
    int tokens;
    const std::atomic<bool>* cancelled = nullptr;
};
using Project = void (*)(const Weight&, const Activations&, bool, float*);
void project_scalar(const Weight&, const Activations&, bool, float*);
#if defined(NINFER_CPU_EXPERT_X86)
void project_avx2(const Weight&, const Activations&, bool, float*);
void project_avx512(const Weight&, const Activations&, bool, float*);
#endif

inline float half(std::uint16_t bits) {
    unsigned exponent = (bits >> 10) & 31U, mantissa = bits & 1023U;
    const auto sign = std::uint32_t(bits & 0x8000U) << 16;
    if (exponent == 0) {
        if (mantissa == 0) { return std::bit_cast<float>(sign); }
        int shift = 0;
        while ((mantissa & 1024U) == 0) { mantissa <<= 1; ++shift; }
        return std::bit_cast<float>(sign | (std::uint32_t(113 - shift) << 23) |
                                    ((mantissa & 1023U) << 13));
    }
    return std::bit_cast<float>(sign | ((exponent == 31 ? 255U : exponent + 112U) << 23) |
                                (mantissa << 13));
}

inline std::uint16_t word(const void* pointer, std::size_t index) {
    std::uint16_t value;
    std::memcpy(&value, static_cast<const std::uint8_t*>(pointer) + index * 2, 2);
    return value;
}

template<int Bits>
float decode32(const Weight& weight, int row, int block, std::int8_t* decoded) {
    constexpr int group_size = Bits == 8 ? 32 : 64;
    const int groups = weight.k / group_size;
    const int group = block * 32 / group_size, start = block * 32 % group_size;
    const std::size_t record = weight.layout == QuantLayout::RowSplitPanel
        ? (std::size_t(row / 64) * groups + group) * 64 + row % 64
        : std::size_t(row) * groups + group;
    const auto* low = static_cast<const std::uint8_t*>(weight.qdata) +
                      record * (Bits == 2 ? 16 : 32);
    const auto* high = static_cast<const std::uint8_t*>(weight.qhigh);
    if constexpr (Bits == 5 || Bits == 6) { high += record * (Bits == 5 ? 8 : 16); }
    for (int lane = 0; lane < 32; ++lane) {
        const int column = start + lane;
        unsigned code;
        if constexpr (Bits == 8) { code = low[column]; }
        else if constexpr (Bits == 2) { code = (low[column / 4] >> (2 * (column % 4))) & 3U; }
        else {
            code = (low[column / 2] >> (4 * (column % 2))) & 15U;
            if constexpr (Bits == 5) { code |= ((high[column / 8] >> (column % 8)) & 1U) << 4; }
            if constexpr (Bits == 6) { code |= ((high[column / 4] >> (2 * (column % 4))) & 3U) << 4; }
        }
        const int value = Bits == 2 ? int(code) - 1 :
            int(code) - ((code & (1U << (Bits - 1))) ? (1 << Bits) : 0);
        decoded[lane] = static_cast<std::int8_t>(value);
    }
    return half(word(weight.scales, std::size_t(row) * groups + group));
}

template<class Dot, int Bits>
void quantized(const Weight& weight, const Activations& x, bool a8, float* output) {
    const int n = weight.n, k = weight.k, blocks = k / 32;
    alignas(64) std::int8_t codes[32];
    // Decode a weight group once for every token in a verify window. Workers split experts;
    // a matrix's stored bytes are shared by its columns rather than reread per token.
    for (int row = 0; row < n; ++row) {
        if (x.cancelled && x.cancelled->load(std::memory_order_relaxed)) { throw CpuExpertCancelled(); }
        for (int t = 0; t < x.tokens; ++t) { output[std::size_t(t) * n + row] = 0; }
        for (int block = 0; block < blocks; ++block) {
            const float scale = decode32<Bits>(weight, row, block, codes);
            for (int t = 0; t < x.tokens; ++t) {
                const auto at = std::size_t(t) * k + block * 32;
                const float product = a8 ?
                    float(Dot::integer(codes, x.codes + at)) * x.scales[std::size_t(t) * blocks + block] :
                    Dot::floating(codes, x.values + at);
                output[std::size_t(t) * n + row] += product * scale;
            }
        }
    }
}

template<class Dot>
void project(const Weight& weight, const Activations& x, bool a8, float* output) {
    switch (weight.qtype) {
    case QType::Q2_G64_FP16: return quantized<Dot, 2>(weight, x, a8, output);
    case QType::Q4_G64_FP16: return quantized<Dot, 4>(weight, x, a8, output);
    case QType::Q5_G64_FP16: return quantized<Dot, 5>(weight, x, a8, output);
    case QType::Q6_G64_FP16: return quantized<Dot, 6>(weight, x, a8, output);
    case QType::Q8_G32_FP16: return quantized<Dot, 8>(weight, x, a8, output);
    default: break; // BF16, validated at the Op boundary.
    }
    for (int row = 0; row < weight.n; ++row) {
        if (x.cancelled && x.cancelled->load(std::memory_order_relaxed)) { throw CpuExpertCancelled(); }
        for (int t = 0; t < x.tokens; ++t) { output[std::size_t(t) * weight.n + row] = 0; }
        for (int column = 0; column < weight.k; ++column) {
            const float value = std::bit_cast<float>(std::uint32_t(
                word(weight.qdata, std::size_t(row) * weight.k + column)) << 16);
            for (int t = 0; t < x.tokens; ++t) {
                output[std::size_t(t) * weight.n + row] += value * x.values[std::size_t(t) * weight.k + column];
            }
        }
    }
}
} // namespace ninfer::ops::cpu_expert_detail
