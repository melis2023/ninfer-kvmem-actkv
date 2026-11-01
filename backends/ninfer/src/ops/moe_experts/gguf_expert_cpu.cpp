#include "ops/moe_experts/gguf_expert_cpu.h"

#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_cpu_tables.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(NINFER_CPU_EXPERT_X86) && (defined(__GNUC__) || defined(__clang__))
#    include <cpuid.h>
#endif
#if defined(_MSC_VER) && defined(NINFER_CPU_EXPERT_X86)
#    include <intrin.h>
#endif

namespace ninfer::ops {
namespace gguf_cpu_detail {

float half_to_float(std::uint16_t bits) noexcept {
    unsigned exponent = (bits >> 10) & 31U, mantissa = bits & 1023U;
    const auto sign   = std::uint32_t(bits & 0x8000U) << 16;
    if (exponent == 0) {
        if (mantissa == 0) { return std::bit_cast<float>(sign); }
        int shift = 0;
        while ((mantissa & 1024U) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        return std::bit_cast<float>(sign | (std::uint32_t(113 - shift) << 23) |
                                    ((mantissa & 1023U) << 13));
    }
    return std::bit_cast<float>(sign | ((exponent == 31 ? 255U : exponent + 112U) << 23) |
                                (mantissa << 13));
}

float round_to_half(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = bits & 0x80000000U, magnitude = bits & 0x7FFFFFFFU;
    if (magnitude >= 0x7F800000U) { return value; } // infinity or NaN
    if (magnitude < 0x38800000U) {
        // Below binary16's smallest normal 2^-14 the spacing is 2^-24 (exact scalings).
        return std::nearbyint(value * 0x1p24F) * 0x1p-24F;
    }
    // Ten of the 23 mantissa bits, rounded to nearest, ties to even; a carry may raise the exponent.
    const std::uint32_t rounded = (magnitude + 0xFFFU + ((magnitude >> 13) & 1U)) & ~0x1FFFU;
    if (rounded > 0x477FE000U) { return std::bit_cast<float>(sign | 0x7F800000U); } // past 65504
    return std::bit_cast<float>(sign | rounded);
}

std::uint16_t float_to_bf16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7FFFFFFFU) > 0x7F800000U) { return std::uint16_t((bits >> 16) | 0x40U); } // NaN
    return std::uint16_t((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

namespace {
std::uint16_t load16(const std::uint8_t* p) {
    std::uint16_t value;
    std::memcpy(&value, p, 2);
    return value;
}

std::uint32_t load32(const std::uint8_t* p) {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

std::int8_t signed_by(int value, unsigned negative) {
    return static_cast<std::int8_t>(negative ? -value : value);
}
} // namespace

// ggml's dequantize_row_* (ggml-quants.c) for one 32-value slice, as integers, with the GPU
// decoder's slice factor: the block's binary16 multiplier times an integer of at most six bits and
// a power of two, so every factor is exact.
void decode_slice(QType format, const std::uint8_t* row, int slice, GgufCpuSlice& out) noexcept {
    out.factor[1] = 0.0F;
    switch (format) {
    case QType::GGUF_Q8_0: {
        const std::uint8_t* block = row + std::size_t(slice) * 34;
        out.factor[0]             = half_to_float(load16(block));
        for (int j = 0; j < 32; ++j) { out.weights[j] = static_cast<std::int8_t>(block[2 + j]); }
        return;
    }
    case QType::GGUF_IQ4_NL: {
        // d then 16 bytes of two 4-bit indices into ggml's 16 values, the low nibbles first.
        const std::uint8_t* block = row + std::size_t(slice) * 18;
        out.factor[0]             = half_to_float(load16(block));
        for (int j = 0; j < 16; ++j) {
            out.weights[j]      = kvalues_iq4nl[block[2 + j] & 15U];
            out.weights[j + 16] = kvalues_iq4nl[block[2 + j] >> 4];
        }
        return;
    }
    case QType::GGUF_Q4_0: {
        // d then 16 bytes of codes c in 0..15 standing for c - 8: value j the low nibble of byte j,
        // value j + 16 its high nibble.
        const std::uint8_t* block = row + std::size_t(slice) * 18;
        out.factor[0]             = half_to_float(load16(block));
        for (int j = 0; j < 16; ++j) {
            out.weights[j]      = static_cast<std::int8_t>(int(block[2 + j] & 15U) - 8);
            out.weights[j + 16] = static_cast<std::int8_t>(int(block[2 + j] >> 4) - 8);
        }
        return;
    }
    case QType::GGUF_Q5_0: {
        // Q4_0's nibbles with a fifth bit per value in the word after d (bit j for value j),
        // standing for c - 16.
        const std::uint8_t* block = row + std::size_t(slice) * 22;
        out.factor[0]             = half_to_float(load16(block));
        const std::uint32_t qh    = load32(block + 2);
        for (int j = 0; j < 16; ++j) {
            const unsigned lo   = (block[6 + j] & 15U) | (((qh >> j) & 1U) << 4);
            const unsigned hi   = (block[6 + j] >> 4) | (((qh >> (j + 16)) & 1U) << 4);
            out.weights[j]      = static_cast<std::int8_t>(int(lo) - 16);
            out.weights[j + 16] = static_cast<std::int8_t>(int(hi) - 16);
        }
        return;
    }
    case QType::GGUF_Q2_0: {
        // A 64-value block is two slices; codes c in 0..3, value v at bits 2 (v % 4) of byte v / 4,
        // stand for c - 1.
        const std::uint8_t* block = row + std::size_t(slice / 2) * 18;
        out.factor[0]             = half_to_float(load16(block));
        for (int j = 0; j < 32; ++j) {
            const int v    = 32 * (slice % 2) + j;
            out.weights[j] = static_cast<std::int8_t>(((block[2 + v / 4] >> (2 * (v % 4))) & 3) - 1);
        }
        return;
    }
    default: break;
    }
    const std::uint8_t* block = row + std::size_t(slice / 8) * gguf_block_shape(format).bytes;
    const int sub             = slice % 8;
    switch (format) {
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K: {
        // d and dmin, twelve bytes of 6-bit scales and minimums, (Q5_K: 32 bytes of fifth bits,)
        // 128 bytes of nibbles: slice u's are the low (u even) or high nibbles of bytes
        // 32 (u / 2) .. 32 (u / 2) + 31.
        const bool five = format == QType::GGUF_Q5_K;
        const std::uint8_t* s6 = block + 4;
        int sc = 0, m = 0;
        if (sub < 4) {
            sc = s6[sub] & 63;
            m  = s6[sub + 4] & 63;
        } else {
            sc = (s6[sub + 4] & 15) | ((s6[sub - 4] >> 6) << 4);
            m  = (s6[sub + 4] >> 4) | ((s6[sub] >> 6) << 4);
        }
        out.factor[0] = half_to_float(load16(block)) * float(sc);
        out.factor[1] = half_to_float(load16(block + 2)) * float(m);
        const std::uint8_t* qs = block + (five ? 48 : 16) + 32 * (sub / 2);
        for (int j = 0; j < 32; ++j) {
            unsigned value = (qs[j] >> (4 * (sub & 1))) & 15U;
            if (five) { value |= ((block[16 + j] >> sub) & 1U) << 4; }
            out.weights[j] = static_cast<std::int8_t>(value);
        }
        return;
    }
    case QType::GGUF_Q6_K: {
        // 128 bytes of low nibbles, 64 of high bit pairs, 16 signed scales, then d. Slice u = 4 h
        // + q: the low (q < 2) or high nibbles of bytes 64 h + 32 (q % 2) ..., bits 2 q of bytes
        // 128 + 32 h ..., scales 8 h + 2 q (values 0..15) and 8 h + 2 q + 1 (16..31), less 32.
        const int h = sub / 4, q = sub % 4;
        const float d            = half_to_float(load16(block + 208));
        const auto* scales       = reinterpret_cast<const std::int8_t*>(block + 192 + 8 * h);
        const std::uint8_t* ql   = block + 64 * h + 32 * (q & 1);
        const std::uint8_t* qh   = block + 128 + 32 * h;
        for (int j = 0; j < 32; ++j) {
            const unsigned value = ((ql[j] >> (4 * (q >> 1))) & 15U) | (((qh[j] >> (2 * q)) & 3U) << 4);
            out.weights[j]       = static_cast<std::int8_t>(int(value) - 32);
        }
        out.factor[0] = d * float(scales[2 * q]);
        out.factor[1] = d * float(scales[2 * q + 1]);
        return;
    }
    default: break;
    }
    const float d = half_to_float(load16(block));
    switch (format) {
    case QType::GGUF_IQ2_S: {
        // d, 32 grid-index bytes, 32 sign bytes, 8 high-index bytes, 8 scale bytes.
        const std::uint8_t* qs    = block + 2 + 4 * sub;
        const std::uint8_t* signs = block + 34 + 4 * sub;
        const unsigned qh         = block[66 + sub];
        const unsigned scale      = block[74 + sub];
        for (int l = 0; l < 4; ++l) {
            const auto* grid = reinterpret_cast<const std::uint8_t*>(
                iq2s_grid + (qs[l] | ((qh << (8 - 2 * l)) & 0x300U)));
            for (int j = 0; j < 8; ++j) {
                out.weights[8 * l + j] = signed_by(grid[j], signs[l] & (1U << j));
            }
        }
        out.factor[0] = d * float(2 * (scale & 15U) + 1) * 0.125F;
        out.factor[1] = d * float(2 * (scale >> 4) + 1) * 0.125F;
        return;
    }
    case QType::GGUF_IQ3_XXS: {
        // d, 64 grid-index bytes, then per sub-block a word of four 7-bit sign indices and a
        // 4-bit scale.
        const std::uint8_t* qs  = block + 2 + 8 * sub;
        const std::uint32_t aux = load32(block + 66 + 4 * sub);
        for (int l = 0; l < 4; ++l) {
            const unsigned signs = ksigns_iq2xs[(aux >> (7 * l)) & 127U];
            for (int g = 0; g < 2; ++g) {
                const auto* grid = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + g]);
                for (int j = 0; j < 4; ++j) {
                    out.weights[8 * l + 4 * g + j] = signed_by(grid[j], signs & (1U << (4 * g + j)));
                }
            }
        }
        out.factor[0] = d * float(2 * (aux >> 28) + 1) * 0.25F;
        return;
    }
    case QType::GGUF_IQ3_S: {
        // d, 64 grid-index bytes, 8 high-bit bytes, 32 sign bytes, 4 bytes of two 4-bit scales.
        const std::uint8_t* qs    = block + 2 + 8 * sub;
        const unsigned qh         = block[66 + sub];
        const std::uint8_t* signs = block + 74 + 4 * sub;
        const unsigned scale      = (block[106 + sub / 2] >> (4 * (sub & 1))) & 15U;
        for (int g = 0; g < 8; ++g) {
            const auto* grid = reinterpret_cast<const std::uint8_t*>(
                iq3s_grid + (qs[g] | (((qh >> g) & 1U) << 8)));
            for (int j = 0; j < 4; ++j) {
                const int at    = 4 * g + j;
                out.weights[at] = signed_by(grid[j], signs[at / 8] & (1U << (at % 8)));
            }
        }
        out.factor[0] = d * float(2 * scale + 1);
        return;
    }
    case QType::GGUF_IQ4_XS: {
        // d, a 16-bit word of the scales' high bits, 4 bytes of their low nibbles, 128 code bytes.
        const unsigned high    = unsigned(load16(block + 2));
        const unsigned low     = (block[4 + sub / 2] >> (4 * (sub & 1))) & 15U;
        const std::uint8_t* qs = block + 8 + 16 * sub;
        for (int j = 0; j < 16; ++j) {
            out.weights[j]      = kvalues_iq4nl[qs[j] & 15U];
            out.weights[j + 16] = kvalues_iq4nl[qs[j] >> 4];
        }
        out.factor[0] = d * float(int(low | (((high >> (2 * sub)) & 3U) << 4)) - 32);
        return;
    }
    default: return;
    }
}

float butterfly(float* lanes) noexcept {
    for (int o = 16; o > 0; o >>= 1) {
        for (int l = 0; l < o; ++l) { lanes[l] += lanes[l + o]; }
    }
    return lanes[0];
}

void rows_scalar(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                 float* y, std::size_t y_stride) {
    const int tokens = x.tokens();
    const int slices = w.k / 32;
    const bool q2    = w.format == QType::GGUF_Q2_0;
    const bool two   = w.format == QType::GGUF_IQ2_S;
    const bool six   = w.format == QType::GGUF_Q6_K;
    const bool min   = w.format == QType::GGUF_Q4_K || w.format == QType::GGUF_Q5_K;
    std::vector<float> lanes(std::size_t(tokens) * 32);
    GgufCpuSlice slice;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* r = w.data + std::size_t(row) * w.row_bytes;
        std::fill(lanes.begin(), lanes.end(), 0.0F);
        for (int s = 0; s < slices; ++s) {
            decode_slice(w.format, r, s, slice);
            for (int t = 0; t < tokens; ++t) {
                // Q2_0 activations sit in the vector decoders' lane order.
                const std::int8_t* a = x.codes(t) + (q2 ? (s / 2) * 64 : 32 * s);
                std::int32_t dot[2]  = {0, 0};
                for (int j = 0; j < 32; ++j) {
                    dot[j / 16] += slice.weights[j] * a[q2 ? q2_lane_position(32 * (s % 2) + j) : j];
                }
                const float scale = x.scales(t)[s];
                float product     = 0;
                if (two) {
                    product = scale * std::fma(slice.factor[0], float(dot[0]), slice.factor[1] * float(dot[1]));
                } else if (six) {
                    const float low  = slice.factor[0] * float(dot[0]);
                    const float high = slice.factor[1] * float(dot[1]);
                    product          = scale * (low + high);
                } else {
                    product = slice.factor[0] * scale * float(dot[0] + dot[1]);
                    if (min) {
                        const float offset = slice.factor[1] * x.value_sums(t)[s];
                        product            = product - offset;
                    }
                }
                lanes[std::size_t(t) * 32 + s % 32] += product;
            }
        }
        for (int t = 0; t < tokens; ++t) {
            y[std::size_t(t) * y_stride + row] = butterfly(lanes.data() + std::size_t(t) * 32);
        }
    }
}

void swiglu_scalar(const float* gate, const float* up, std::size_t count, std::uint16_t* out) {
    for (std::size_t i = 0; i < count; ++i) { out[i] = float_to_bf16(portable_silu(gate[i]) * up[i]); }
}

void geglu_scalar(const float* gate, const float* up, std::size_t count, std::uint16_t* out) {
    for (std::size_t i = 0; i < count; ++i) { out[i] = float_to_bf16(gelu_tanh(gate[i]) * up[i]); }
}

} // namespace gguf_cpu_detail

namespace {
#if defined(NINFER_CPU_EXPERT_X86) && defined(_MSC_VER)
// MSVC has no __builtin_cpu_supports: CPUID leaves 1 and 7 and the register state the OS saves
// (XCR0) answer the same questions.
struct CpuFeatures {
    unsigned leaf1_ecx = 0, leaf7_ebx = 0;
    unsigned long long xcr0 = 0;
};

CpuFeatures cpu_features() noexcept {
    CpuFeatures out;
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) { return out; }
    __cpuid(regs, 1);
    out.leaf1_ecx = static_cast<unsigned>(regs[2]);
    if ((out.leaf1_ecx & (1U << 27)) == 0) { return out; } // no OSXSAVE: no AVX state
    out.xcr0 = _xgetbv(0);
    __cpuidex(regs, 7, 0);
    out.leaf7_ebx = static_cast<unsigned>(regs[1]);
    return out;
}

constexpr bool all_bits(unsigned value, unsigned mask) noexcept { return (value & mask) == mask; }
#endif

bool cpu_has_avx2() noexcept {
#if defined(NINFER_CPU_EXPERT_X86) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("f16c");
#elif defined(NINFER_CPU_EXPERT_X86) && defined(_MSC_VER)
    const CpuFeatures cpu = cpu_features();
    // XCR0 SSE and AVX state; leaf 1 FMA, AVX, F16C; leaf 7 AVX2.
    return (cpu.xcr0 & 0x6U) == 0x6U &&
           all_bits(cpu.leaf1_ecx, (1U << 12) | (1U << 28) | (1U << 29)) &&
           all_bits(cpu.leaf7_ebx, 1U << 5);
#else
    return false;
#endif
}

} // namespace

bool gguf_cpu_supported(QType format) noexcept {
    switch (format) {
    case QType::GGUF_Q2_0:
    case QType::GGUF_Q4_0:
    case QType::GGUF_Q5_0:
    case QType::GGUF_Q8_0:
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K:
    case QType::GGUF_Q6_K:
    case QType::GGUF_IQ4_NL:
    case QType::GGUF_IQ2_S:
    case QType::GGUF_IQ3_XXS:
    case QType::GGUF_IQ3_S:
    case QType::GGUF_IQ4_XS: return true;
    default: return false;
    }
}

bool gguf_cpu_backend_available(GgufCpuBackend backend) noexcept {
    switch (backend) {
    case GgufCpuBackend::Automatic:
    case GgufCpuBackend::Scalar: return true;
    case GgufCpuBackend::Avx2: return cpu_has_avx2();
    }
    return false;
}

const char* gguf_cpu_backend_name(GgufCpuBackend backend) noexcept {
    return gguf_cpu_backend_selected(backend) == GgufCpuBackend::Avx2 ? "avx2" : "scalar";
}

GgufCpuBackend gguf_cpu_backend_selected(GgufCpuBackend requested) noexcept {
    if (requested != GgufCpuBackend::Automatic) { return requested; }
    if (cpu_has_avx2()) { return GgufCpuBackend::Avx2; }
    return GgufCpuBackend::Scalar;
}

GgufCpuActivation::GgufCpuActivation(int token_capacity, int k)
    : capacity_(token_capacity), k_(k), groups_(k / 32), codes_(std::size_t(token_capacity) * k),
      scales_(std::size_t(token_capacity) * (k / 32)),
      value_sums_(std::size_t(token_capacity) * (k / 32)),
      sums_(std::size_t(token_capacity) * (k / 32)) {
    if (token_capacity <= 0 || k <= 0 || k % 32 != 0) {
        throw std::invalid_argument("GGUF CPU activation: k must be a positive multiple of 32");
    }
}

// ggml's quantize_q8_1 as the GPU computes it (ggml_bridge_vec.cuh decode_quantize): per 32
// values d = amax / 127 and codes round(v / d), halves away from zero; the scale used is d in
// binary16, and s the values' warp sum in binary16.
void GgufCpuActivation::quantize_group(int token, int g, const float* v) {
    std::int8_t* codes = codes_.data() + std::size_t(token) * k_;
    const bool lanes   = format_ == QType::GGUF_Q2_0;
    float warp[32];
    float amax = 0;
    for (int i = 0; i < 32; ++i) {
        if (!std::isfinite(v[i])) {
            throw std::runtime_error("GGUF CPU expert: nonfinite activation");
        }
        amax    = std::max(amax, std::abs(v[i]));
        warp[i] = v[i];
    }
    const std::size_t at = std::size_t(token) * groups_ + g;
    value_sums_[at]      = gguf_cpu_detail::round_to_half(gguf_cpu_detail::butterfly(warp));
    const float d        = amax / 127.0F;
    std::int32_t sum     = 0;
    for (int i = 0; i < 32; ++i) {
        const auto code =
            amax == 0 ? std::int8_t(0) : static_cast<std::int8_t>(std::round(v[i] / d));
        sum += code;
        const int j  = 32 * g + i;
        const int to = lanes ? j / 64 * 64 + gguf_cpu_detail::q2_lane_position(j % 64) : j;
        codes[to]    = code;
    }
    scales_[at] = gguf_cpu_detail::round_to_half(d);
    sums_[at]   = sum;
}

void GgufCpuActivation::reset(QType format, int tokens) {
    if (tokens <= 0 || tokens > capacity_) {
        throw std::invalid_argument("GGUF CPU activation: token count exceeds capacity");
    }
    if (format == QType::GGUF_Q2_0 && k_ % 64 != 0) {
        throw std::invalid_argument("GGUF CPU activation: Q2_0 needs whole 64-value blocks");
    }
    format_ = format;
    tokens_ = tokens;
}

void GgufCpuActivation::quantize_bf16(int token, const std::uint16_t* column, int group_begin,
                                      int group_end) {
    if (token < 0 || token >= tokens_ || group_begin < 0 || group_end > groups_) {
        throw std::invalid_argument("GGUF CPU activation: token or group outside the activation");
    }
    float values[32];
    for (int g = group_begin; g < group_end; ++g) {
        for (int i = 0; i < 32; ++i) {
            values[i] = std::bit_cast<float>(std::uint32_t(column[32 * g + i]) << 16);
        }
        quantize_group(token, g, values);
    }
}

void GgufCpuActivation::copy_token(int token, const GgufCpuActivation& from, int from_token) {
    if (token < 0 || token >= tokens_ || from_token < 0 || from_token >= from.tokens_ ||
        from.k_ != k_ || from.format_ != format_) {
        throw std::invalid_argument("GGUF CPU activation: copied token outside the activations");
    }
    std::memcpy(codes_.data() + std::size_t(token) * k_, from.codes(from_token), std::size_t(k_));
    const std::size_t to = std::size_t(token) * groups_, at = std::size_t(from_token) * groups_;
    std::copy_n(from.scales_.data() + at, groups_, scales_.data() + to);
    std::copy_n(from.value_sums_.data() + at, groups_, value_sums_.data() + to);
    std::copy_n(from.sums_.data() + at, groups_, sums_.data() + to);
}

void GgufCpuActivation::prepare_bf16(QType format, const std::uint16_t* values, std::size_t stride,
                                     int tokens) {
    reset(format, tokens);
    for (int t = 0; t < tokens; ++t) {
        quantize_bf16(t, values + std::size_t(t) * stride, 0, groups_);
    }
}

void GgufCpuActivation::load_q8_1(QType format, const std::int8_t* codes, const std::uint16_t* ds,
                                   std::span<const int> columns) {
    const int tokens = int(columns.size());
    if (tokens <= 0 || tokens > capacity_) {
        throw std::invalid_argument("GGUF CPU activation: token count exceeds capacity");
    }
    if (format == QType::GGUF_Q2_0 && k_ % 64 != 0) {
        throw std::invalid_argument("GGUF CPU activation: Q2_0 needs whole 64-value blocks");
    }
    format_ = format;
    tokens_ = tokens;
    const bool lanes = format == QType::GGUF_Q2_0;
    for (int t = 0; t < tokens; ++t) {
        const std::int8_t* from = codes + std::size_t(columns[std::size_t(t)]) * k_;
        const std::uint16_t* pair = ds + std::size_t(columns[std::size_t(t)]) * groups_ * 2;
        std::int8_t* to = codes_.data() + std::size_t(t) * k_;
        for (int g = 0; g < groups_; ++g) {
            std::int32_t sum = 0;
            for (int i = 0; i < 32; ++i) {
                const int j = 32 * g + i;
                sum += from[j];
                to[lanes ? j / 64 * 64 + gguf_cpu_detail::q2_lane_position(j % 64) : j] = from[j];
            }
            scales_[std::size_t(t) * groups_ + g]     = gguf_cpu_detail::half_to_float(pair[2 * g]);
            value_sums_[std::size_t(t) * groups_ + g] = gguf_cpu_detail::half_to_float(pair[2 * g + 1]);
            sums_[std::size_t(t) * groups_ + g]       = sum;
        }
    }
}

void gguf_cpu_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                   float* y, std::size_t y_stride, [[maybe_unused]] GgufCpuBackend backend) {
    if (!gguf_cpu_supported(w.format) || w.format != x.format() || w.k != x.k() || !w.data ||
        row_begin < 0 || row_end > w.rows || row_begin > row_end) {
        throw std::invalid_argument("GGUF CPU expert: mismatched matrix, activation or rows");
    }
#if defined(NINFER_CPU_EXPERT_X86)
    if (gguf_cpu_backend_selected(backend) == GgufCpuBackend::Avx2) {
        return gguf_cpu_detail::rows_avx2(w, x, row_begin, row_end, y, y_stride);
    }
#endif
    gguf_cpu_detail::rows_scalar(w, x, row_begin, row_end, y, y_stride);
}

void gguf_cpu_swiglu_bf16(const float* gate, const float* up, std::size_t count, std::uint16_t* out,
                          [[maybe_unused]] GgufCpuBackend backend) {
#if defined(NINFER_CPU_EXPERT_X86)
    if (gguf_cpu_backend_selected(backend) == GgufCpuBackend::Avx2) {
        return gguf_cpu_detail::swiglu_avx2(gate, up, count, out);
    }
#endif
    gguf_cpu_detail::swiglu_scalar(gate, up, count, out);
}

void gguf_cpu_geglu_bf16(const float* gate, const float* up, std::size_t count, std::uint16_t* out,
                         [[maybe_unused]] GgufCpuBackend backend) {
#if defined(NINFER_CPU_EXPERT_X86)
    if (gguf_cpu_backend_selected(backend) == GgufCpuBackend::Avx2) {
        return gguf_cpu_detail::geglu_avx2(gate, up, count, out);
    }
#endif
    gguf_cpu_detail::geglu_scalar(gate, up, count, out);
}

} // namespace ninfer::ops
