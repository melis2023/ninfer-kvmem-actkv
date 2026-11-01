// GGUF CPU expert rows (Q2_0, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_XS) against FP64 from independently decoded stored blocks, at the Flash-Next and Gemma 4
// expert shapes, for every backend this host supports. Blocks are decoded by ggml's reference
// dequantizers. Two oracles: the exact product bounds the int8 activation's error, and the q8_1
// product (the activation quantized per 32 values as the GPU does, d = amax / 127 rounded to
// binary16, codes round(v / d); a K-quant's minimum times the values' binary16 sum s) must agree
// to FP32 summation, so the CPU computes the GPU's arithmetic; every vector backend must give the
// scalar products' bits (test_moe_experts_gguf compares those with the GPU's), and so must the
// GPU's planar q8_1 activation (load_q8_1). SiLU and the tanh-approximated GELU
// (ops/common/silu.h) against FP64, and the BF16 middle of every backend.
#include "ops/common/silu.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

std::uint16_t half_bits(float value) {
    // Positive normal values only, rounded to nearest by truncating a rounded mantissa.
    const auto word = std::bit_cast<std::uint32_t>(value);
    const int exponent = int((word >> 23) & 255) - 127 + 15;
    const std::uint32_t mantissa = ((word & 0x7fffff) + 0x1000) >> 13;
    return static_cast<std::uint16_t>((exponent << 10) + mantissa);
}

std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    return static_cast<std::uint16_t>(word >> 16);
}

// The binary16 value nearest `value` (ties to even), from its exponent's spacing in FP64.
double nearest_half(float value) {
    const double v = std::abs(double(value));
    if (v == 0) { return 0; }
    const int exponent   = std::max(std::ilogb(v), -14); // below 2^-14 the spacing stays 2^-24
    const double spacing = std::ldexp(1.0, exponent - 10);
    const double rounded = std::nearbyint(v / spacing) * spacing; // FE_TONEAREST: ties to even
    return value < 0 ? -rounded : rounded;
}

// The binary16 word of a value binary16 holds exactly.
std::uint16_t exact_half_bits(float value) {
    static const std::vector<std::pair<float, std::uint16_t>> table = [] {
        std::vector<std::pair<float, std::uint16_t>> out;
        for (unsigned bits = 0; bits < 0x10000; ++bits) {
            if (((bits >> 10) & 31) == 31) { continue; }
            out.emplace_back(float(exact_half(bits)), std::uint16_t(bits));
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
            return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
        return out;
    }();
    const auto it = std::lower_bound(table.begin(), table.end(), std::pair{value, std::uint16_t(0)});
    if (it == table.end() || it->first != value) { throw std::runtime_error("not a binary16 value"); }
    return std::signbit(value) && value == 0 ? std::uint16_t(0x8000) : it->second;
}

struct Matrix {
    QType format;
    int rows, k;
    std::int64_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<double> values; // decoded independently from the stored bytes
    // Q4_K and Q5_K: each 32-value slice's minimum dmin * m, which its values subtract.
    std::vector<double> minimums;
};

std::uint16_t load16(const std::uint8_t* p) {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

// ggml-quants.c's get_scale_min_k4.
void scale_min_k4(int j, const std::uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ggml-quants.c's dequantize_row_q4_0, _q5_0, _q4_K, _q5_K and _q6_K (llama.cpp a25c9865, MIT),
// with FP64 output, from the stored bytes (ggml-common.h's layouts): one block to its values, and
// the K-quants' minimum per 32 values.
void reference_plain_block(QType format, const std::uint8_t* x, double* y, double* minimum) {
    if (format == QType::GGUF_Q4_0) {
        const double d = exact_half(load16(x));
        for (int j = 0; j < 16; ++j) {
            y[j]      = ((x[2 + j] & 0x0F) - 8) * d;
            y[j + 16] = ((x[2 + j] >> 4) - 8) * d;
        }
        return;
    }
    if (format == QType::GGUF_Q5_0) {
        const double d = exact_half(load16(x));
        std::uint32_t qh;
        std::memcpy(&qh, x + 2, 4);
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const std::uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            y[j]      = (((x[6 + j] & 0x0F) | xh_0) - 16) * d;
            y[j + 16] = (((x[6 + j] >> 4) | xh_1) - 16) * d;
        }
        return;
    }
    if (format == QType::GGUF_Q6_K) {
        const double d          = exact_half(load16(x + 208));
        const std::uint8_t* ql  = x;
        const std::uint8_t* qh  = x + 128;
        const auto* sc          = reinterpret_cast<const std::int8_t*>(x + 192);
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = int((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        return;
    }
    // Q4_K, Q5_K
    const bool five         = format == QType::GGUF_Q5_K;
    const double d          = exact_half(load16(x));
    const double min        = exact_half(load16(x + 2));
    const std::uint8_t* scales = x + 4;
    const std::uint8_t* qh  = x + 16;
    const std::uint8_t* q   = x + (five ? 48 : 16);
    int is = 0;
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        int sc, m;
        scale_min_k4(is + 0, scales, sc, m);
        const double d1 = d * sc, m1 = min * m;
        scale_min_k4(is + 1, scales, sc, m);
        const double d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            y[l] = d1 * ((q[l] & 0xF) + (five && (qh[l] & u1) ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            y[l + 32] = d2 * ((q[l] >> 4) + (five && (qh[l] & u2) ? 16 : 0)) - m2;
        }
        minimum[is]     = m1;
        minimum[is + 1] = m2;
        y += 64;
        q += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

bool k_minimum(QType format) { return format == QType::GGUF_Q4_K || format == QType::GGUF_Q5_K; }

// ggml-quants.c's dequantize_row_iq2_s, _iq3_xxs, _iq3_s, _iq4_nl and _iq4_xs (llama.cpp
// a25c9865, MIT), with FP64 output: one stored block of `format` to its values.
void reference_block(QType format, const std::uint8_t* bytes, double* y) {
    if (format == QType::GGUF_IQ4_NL) {
        const auto* x  = reinterpret_cast<const block_iq4_nl*>(bytes);
        const double d = exact_half(x->d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j]              = d * kvalues_iq4nl[x->qs[j] & 0xf];
            y[j + QK4_NL / 2] = d * kvalues_iq4nl[x->qs[j] >> 4];
        }
        return;
    }
    if (format == QType::GGUF_IQ4_XS) {
        const auto* x      = reinterpret_cast<const block_iq4_xs*>(bytes);
        const std::uint8_t* qs = x->qs;
        const double d     = exact_half(x->d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0]  = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
        return;
    }
    if (format == QType::GGUF_IQ2_S) {
        const auto* x             = reinterpret_cast<const block_iq2_s*>(bytes);
        const double d            = exact_half(x->d);
        const std::uint8_t* qs    = x->qs;
        const std::uint8_t* qh    = x->qh;
        const std::uint8_t* signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const double db[2] = {d * (0.5 + (x->scales[ib32] & 0xf)) * 0.25,
                                  d * (0.5 + (x->scales[ib32] >> 4)) * 0.25};
            for (int l = 0; l < 4; ++l) {
                const double dl = db[l / 2];
                const auto* grid = reinterpret_cast<const std::uint8_t*>(
                    iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
        return;
    }
    if (format == QType::GGUF_IQ3_XXS) {
        const auto* x       = reinterpret_cast<const block_iq3_xxs*>(bytes);
        const double d      = exact_half(x->d);
        const std::uint8_t* qs = x->qs;
        const std::uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(aux32));
            const double db = d * (0.5 + (aux32 >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const auto* grid1 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 0]);
                const auto* grid2 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.0 : 1.0);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 8;
        }
        return;
    }
    // IQ3_S
    const auto* x             = reinterpret_cast<const block_iq3_s*>(bytes);
    const double d            = exact_half(x->d);
    const std::uint8_t* qs    = x->qs;
    const std::uint8_t* qh    = x->qh;
    const std::uint8_t* signs = x->signs;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const double db1 = d * (1 + 2 * (x->scales[ib32 / 2] & 0xf));
        const double db2 = d * (1 + 2 * (x->scales[ib32 / 2] >> 4));
        for (int half = 0; half < 2; ++half) {
            const double db = half == 0 ? db1 : db2;
            for (int l = 0; l < 4; ++l) {
                const auto* grid1 = reinterpret_cast<const std::uint8_t*>(
                    iq3s_grid + (qs[2 * l + 0] | ((qh[half] << (8 - 2 * l)) & 256)));
                const auto* grid2 = reinterpret_cast<const std::uint8_t*>(
                    iq3s_grid + (qs[2 * l + 1] | ((qh[half] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.0 : 1.0);
                    y[j + 4] = db * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0 : 1.0);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

Matrix make(QType format, int rows, int k, std::mt19937& rng) {
    if (format == QType::GGUF_Q4_0 || format == QType::GGUF_Q5_0 || format == QType::GGUF_Q4_K ||
        format == QType::GGUF_Q5_K || format == QType::GGUF_Q6_K) {
        // Random codes and scales; the block multipliers are set to sensible sizes.
        const auto [group, block] = gguf_block_shape(format);
        Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}, {}};
        m.bytes.resize(std::size_t(rows) * m.row_bytes);
        for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
        const bool k_quant = group == 256;
        std::uniform_real_distribution<float> scale(k_quant ? 0.0001F : 0.002F, k_quant ? 0.0008F : 0.02F);
        m.values.resize(std::size_t(rows) * k);
        m.minimums.resize(std::size_t(rows) * (k / 32));
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < k / group; ++b) {
                std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
                const auto d    = half_bits(scale(rng));
                const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
                std::memcpy(p + (format == QType::GGUF_Q6_K ? 208 : 0), &signed_d, 2);
                if (k_minimum(format)) {
                    const auto dmin = half_bits(scale(rng));
                    std::memcpy(p + 2, &dmin, 2);
                }
                reference_plain_block(format, p,
                                      m.values.data() + std::size_t(r) * k + std::size_t(b) * group,
                                      m.minimums.data() + std::size_t(r) * (k / 32) + std::size_t(b) * (group / 32));
            }
        }
        return m;
    }
    if (format != QType::GGUF_Q2_0 && format != QType::GGUF_Q8_0) {
        // Every byte pattern is a valid i-quant or IQ4_NL block; d is set to a sensible size.
        const auto [group, block] = gguf_block_shape(format);
        Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}, {}};
        m.bytes.resize(std::size_t(rows) * m.row_bytes);
        for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
        std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
        m.values.resize(std::size_t(rows) * k);
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < k / group; ++b) {
                std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
                const auto d    = half_bits(scale(rng));
                const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
                std::memcpy(p, &signed_d, 2);
                reference_block(format, p, m.values.data() + std::size_t(r) * k + std::size_t(b) * group);
            }
        }
        return m;
    }
    const int group = format == QType::GGUF_Q2_0 ? 64 : 32;
    const int block = format == QType::GGUF_Q2_0 ? 18 : 34;
    Matrix m{format, rows, k, std::int64_t(k / group) * block, {}, {}, {}};
    m.bytes.resize(std::size_t(rows) * m.row_bytes);
    std::uniform_real_distribution<float> scale(0.002F, 0.05F);
    for (auto& byte : m.bytes) { byte = static_cast<std::uint8_t>(rng()); }
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            std::uint8_t* p     = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const auto d        = half_bits(scale(rng));
            const std::uint16_t signed_d = (rng() % 4 == 0) ? std::uint16_t(d | 0x8000) : d;
            std::memcpy(p, &signed_d, 2);
        }
    }
    m.values.resize(std::size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / group; ++b) {
            const std::uint8_t* p = m.bytes.data() + std::size_t(r) * m.row_bytes + std::size_t(b) * block;
            const double d        = exact_half(unsigned(p[0]) | (unsigned(p[1]) << 8));
            for (int j = 0; j < group; ++j) {
                const double code = format == QType::GGUF_Q2_0
                                        ? double(((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1)
                                        : double(static_cast<std::int8_t>(p[2 + j]));
                m.values[std::size_t(r) * k + std::size_t(b) * group + j] = d * code;
            }
        }
    }
    return m;
}

void check(const Matrix& m, int tokens, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(std::size_t(tokens) * m.k);
    for (auto& v : x) { v = bf16(normal(rng)); }
    std::vector<double> oracle(std::size_t(tokens) * m.rows);
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < m.rows; ++r) {
            double sum = 0;
            for (int i = 0; i < m.k; ++i) {
                sum += m.values[std::size_t(r) * m.k + i] *
                       double(std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + i]) << 16));
            }
            oracle[std::size_t(t) * m.rows + r] = sum;
        }
    }
    // q8_1 per 32 values, and the product it gives with every slice's dot exact in FP64. The value
    // sum s is the warp's: lane l adds lane l + 16, then + 8, + 4, + 2, + 1, in FP32.
    std::vector<double> q8(oracle.size()), magnitude(oracle.size());
    std::vector<int> codes(m.k);
    std::vector<double> scales(m.k / 32), sums(m.k / 32);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < m.k / 32; ++s) {
            float amax = 0;
            float warp[32];
            for (int i = 0; i < 32; ++i) {
                warp[i] = std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + 32 * s + i]) << 16);
                amax    = std::max(amax, std::abs(warp[i]));
            }
            for (int o = 16; o > 0; o >>= 1) {
                for (int l = 0; l < o; ++l) { warp[l] += warp[l + o]; }
            }
            sums[s]       = nearest_half(warp[0]);
            const float d = amax / 127.0F;
            scales[s]     = nearest_half(d);
            for (int i = 0; i < 32; ++i) {
                const float v = std::bit_cast<float>(std::uint32_t(x[std::size_t(t) * m.k + 32 * s + i]) << 16);
                codes[32 * s + i] = amax == 0 ? 0 : int(std::lround(v / d));
            }
        }
        for (int r = 0; r < m.rows; ++r) {
            double sum = 0, size = 0;
            for (int s = 0; s < m.k / 32; ++s) {
                // A K-quant's minimum comes off as minimum * s, not through the codes.
                const double minimum = k_minimum(m.format) ? m.minimums[std::size_t(r) * (m.k / 32) + s] : 0;
                double dot = 0;
                for (int i = 32 * s; i < 32 * s + 32; ++i) {
                    dot += (m.values[std::size_t(r) * m.k + i] + minimum) * codes[i];
                }
                sum += scales[s] * dot - minimum * sums[s];
                size += std::abs(scales[s] * dot) + std::abs(minimum * sums[s]);
            }
            q8[std::size_t(t) * m.rows + r]        = sum;
            magnitude[std::size_t(t) * m.rows + r] = size;
        }
    }
    GgufCpuActivation a(16, m.k);
    a.prepare_bf16(m.format, x.data(), m.k, tokens);
    const GgufCpuMatrix w{m.format, m.bytes.data(), m.row_bytes, m.rows, m.k};
    std::vector<float> scalar(oracle.size());
    gguf_cpu_rows(w, a, 0, m.rows, scalar.data(), m.rows, GgufCpuBackend::Scalar);
    // The GPU's planar q8_1 activation of the same columns, read back in reverse column order.
    {
        const int groups = m.k / 32;
        std::vector<std::int8_t> planar(std::size_t(tokens) * m.k);
        std::vector<std::uint16_t> ds(std::size_t(tokens) * groups * 2);
        for (int t = 0; t < tokens; ++t) {
            const int c = tokens - 1 - t;
            for (int i = 0; i < m.k; ++i) {
                planar[std::size_t(c) * m.k + i] =
                    a.codes(t)[m.format == QType::GGUF_Q2_0 ? i / 64 * 64 + gguf_cpu_detail::q2_lane_position(i % 64) : i];
            }
            for (int g = 0; g < groups; ++g) {
                ds[(std::size_t(c) * groups + g) * 2]     = exact_half_bits(a.scales(t)[g]);
                ds[(std::size_t(c) * groups + g) * 2 + 1] = exact_half_bits(a.value_sums(t)[g]);
            }
        }
        std::vector<int> columns(tokens);
        for (int t = 0; t < tokens; ++t) { columns[t] = tokens - 1 - t; }
        GgufCpuActivation loaded(16, m.k);
        loaded.load_q8_1(m.format, planar.data(), ds.data(), columns);
        std::vector<float> y(oracle.size());
        gguf_cpu_rows(w, loaded, 0, m.rows, y.data(), m.rows, GgufCpuBackend::Automatic);
        for (std::size_t i = 0; i < y.size(); ++i) {
            require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(scalar[i]),
                    "the planar q8_1 activation's products differ");
        }
    }
    for (const auto backend : {GgufCpuBackend::Scalar, GgufCpuBackend::Avx2}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        std::vector<float> y(oracle.size()), again(oracle.size());
        // Two row ranges, as workers split them.
        gguf_cpu_rows(w, a, 0, m.rows / 3, y.data(), m.rows, backend);
        gguf_cpu_rows(w, a, m.rows / 3, m.rows, y.data(), m.rows, backend);
        gguf_cpu_rows(w, a, 0, m.rows, again.data(), m.rows, backend);
        double error = 0, norm = 0, peak = 0, worst = 0, q8_drift = 0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            q8_drift = std::max(q8_drift, std::abs(double(y[i]) - q8[i]) / (magnitude[i] + 1e-30));
            require(std::isfinite(y[i]), "nonfinite output");
            require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(again[i]),
                    "a row split changed the result or a repeat differed");
            require(std::bit_cast<std::uint32_t>(y[i]) == std::bit_cast<std::uint32_t>(scalar[i]),
                    "a vector backend's bits differ from the scalar products'");
            error += (double(y[i]) - oracle[i]) * (double(y[i]) - oracle[i]);
            norm += oracle[i] * oracle[i];
            peak  = std::max(peak, std::abs(oracle[i]));
            worst = std::max(worst, std::abs(double(y[i]) - oracle[i]));
        }
        const double relative = std::sqrt(error / norm);
        std::cout << gguf_cpu_backend_name(backend) << " format " << int(m.format) << " "
                  << m.rows << "x" << m.k << " T=" << tokens << ": relative L2 " << relative
                  << ", max error/peak " << worst / peak << ", vs q8_1 " << q8_drift << '\n';
        require(relative < 0.03 && worst / peak < 0.1, "error exceeds the int8-activation bound");
        require(q8_drift < 2e-5, "the product is not the GPU's q8_1 arithmetic");
    }
}

// Units in the last place of |exact| as a float, between `got` and FP64's `exact`.
double ulps(float got, double exact) {
    const float near = static_cast<float>(exact);
    const double ulp = std::nextafter(std::abs(near), std::numeric_limits<float>::infinity()) - std::abs(near);
    return std::abs(double(got) - exact) / ulp;
}

double silu64(double x) { return x / (1.0 + std::exp(-x)); }
// 0.5 x (1 + tanh(u)) = x / (1 + e^-2u), the second form keeping its precision for negative u.
double gelu_u(double x) {
    return std::sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x * x * x);
}
double gelu64(double x) { return x / (1.0 + std::exp(-2.0 * gelu_u(x))); }

void check_silu(std::mt19937& rng) {
    // Within four units of FP64 (the exponential's polynomial, the sum and the quotient); where
    // e^-x passes the float range silu is a zero of x's sign.
    std::vector<float> x = {0.0F, -0.0F, 1e-30F, -1e-30F, 1e-3F, -1e-3F, 0.5F, -0.5F, 1.0F, -1.0F,
                            20.0F, -20.0F, 87.0F, -87.0F, 87.3365479F, -87.3365479F, 88.0F,
                            -88.0F, 88.7228394F, -88.7228394F, 100.0F, -100.0F, 3e38F, -3e38F};
    std::normal_distribution<float> normal(0.0F, 4.0F);
    std::uniform_real_distribution<float> wide(-87.0F, 87.0F);
    for (int i = 0; i < 200000; ++i) { x.push_back(i % 2 ? normal(rng) : wide(rng)); }
    double worst = 0;
    for (const float v : x) {
        const float got     = portable_silu(v);
        const double exact  = silu64(double(v));
        require(std::isfinite(got), "nonfinite SiLU");
        if (v <= -88.7228394F) {
            require(got == 0 && std::signbit(got), "SiLU past e^-x's range");
            continue;
        }
        worst = std::max(worst, ulps(got, exact));
    }
    std::cout << "silu: " << x.size() << " values, worst " << worst << " ulp\n";
    require(worst <= 4, "SiLU strays from FP64");
    // The BF16 middle: every backend gives the same words, each within one BF16 unit of FP64's.
    std::vector<float> gate(x.begin(), x.begin() + 100003), up(gate.size());
    std::normal_distribution<float> unit(0.0F, 1.0F);
    for (auto& v : up) { v = unit(rng); }
    std::vector<std::uint16_t> scalar(gate.size());
    gguf_cpu_swiglu_bf16(gate.data(), up.data(), gate.size(), scalar.data(), GgufCpuBackend::Scalar);
    for (std::size_t i = 0; i < gate.size(); ++i) {
        if (std::abs(gate[i]) >= 88.0F) { continue; } // the backends' agreement covers these
        const double exact = silu64(double(gate[i])) * double(up[i]);
        const double got   = double(std::bit_cast<float>(std::uint32_t(scalar[i]) << 16));
        const double step  = std::ldexp(1.0, std::ilogb(std::max(std::abs(exact), 1e-38)) - 7);
        require(std::abs(got - exact) <= step, "the BF16 middle strays from FP64");
    }
    for (const auto backend : {GgufCpuBackend::Avx2}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        std::vector<std::uint16_t> y(gate.size());
        gguf_cpu_swiglu_bf16(gate.data(), up.data(), gate.size(), y.data(), backend);
        require(y == scalar, "a vector backend's BF16 middle differs from the scalar one");
        std::cout << gguf_cpu_backend_name(backend) << " middle: " << y.size() << " words equal\n";
    }
    // GELU: x / (1 + e^-2u) in FP32 against FP64. The FP32 2u carries a relative error of up to
    // four units (the cube's three products, the sum and the scale), which e^-2u scales by |2u|:
    // within (4 |2u| + 8) units of FP64; past e^-2u's range a zero of x's sign.
    double gelu_worst = 0;
    for (const float v : x) {
        const float got    = gelu_tanh(v);
        const double exact = gelu64(double(v));
        require(!std::isnan(got), "NaN GELU");
        const double u = gelu_u(double(v));
        if (-2 * u >= 88.8 || !std::isfinite(u)) {
            require(got == 0 && std::signbit(got) == std::signbit(v), "GELU past e^-2u's range");
            continue;
        }
        if (std::abs(exact) < 1e-30) { continue; }
        const double bound = 4 * std::abs(2 * u) + 8;
        gelu_worst = std::max(gelu_worst, ulps(got, exact) / bound);
    }
    std::cout << "gelu_tanh: worst " << gelu_worst << " of the bound\n";
    require(gelu_worst <= 1, "GELU strays from FP64");
    std::vector<std::uint16_t> geglu(gate.size());
    gguf_cpu_geglu_bf16(gate.data(), up.data(), gate.size(), geglu.data(), GgufCpuBackend::Scalar);
    for (std::size_t i = 0; i < gate.size(); ++i) {
        // Where e^-2u leaves FP32's range GELU is a zero; the backends' agreement covers these.
        if (-2 * gelu_u(double(gate[i])) >= 88.0) { continue; }
        const double exact = gelu64(double(gate[i])) * double(up[i]);
        const double got   = double(std::bit_cast<float>(std::uint32_t(geglu[i]) << 16));
        const double step  = std::ldexp(1.0, std::ilogb(std::max(std::abs(exact), 1e-38)) - 7);
        require(std::abs(got - exact) <= step, "the GeGLU BF16 middle strays from FP64");
    }
    for (const auto backend : {GgufCpuBackend::Avx2}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        std::vector<std::uint16_t> y(gate.size());
        gguf_cpu_geglu_bf16(gate.data(), up.data(), gate.size(), y.data(), backend);
        require(y == geglu, "a vector backend's GeGLU middle differs from the scalar one");
        std::cout << gguf_cpu_backend_name(backend) << " GeGLU middle: " << y.size() << " words equal\n";
    }
}

} // namespace

int main() {
    try {
        std::mt19937 rng(20261009);
        check_silu(rng);
        for (const QType format : {QType::GGUF_Q2_0, QType::GGUF_Q8_0, QType::GGUF_IQ4_NL}) {
            const Matrix gate = make(format, 640, 2560, rng);
            const Matrix down = make(format, 2560, 640, rng);
            for (const int tokens : {1, 2, 3, 4, 5, 8, 9}) {
                check(gate, tokens, rng);
                check(down, tokens, rng);
            }
        }
        // Gemma 4 26B-A4B: [gate; up] 1408 x 2816 and down 2816 x 704 (22 slices, so no K-quant).
        for (const QType format : {QType::GGUF_Q4_0, QType::GGUF_Q5_0, QType::GGUF_Q8_0}) {
            const Matrix gate_up = make(format, 1408, 2816, rng);
            const Matrix down    = make(format, 2816, 704, rng);
            for (const int tokens : {1, 2, 3, 8}) {
                check(gate_up, tokens, rng);
                check(down, tokens, rng);
            }
        }
        for (const QType format : {QType::GGUF_Q4_K, QType::GGUF_Q5_K, QType::GGUF_Q6_K}) {
            const Matrix gate_up = make(format, 1408, 2816, rng);
            for (const int tokens : {1, 2, 3, 8, 9}) { check(gate_up, tokens, rng); }
            require(gguf_cpu_supported(format), "a K-quant is not supported");
        }
        // The i-quants hold the gate and up projections (k = 2560, ten 256-value blocks).
        for (const QType format : {QType::GGUF_IQ2_S, QType::GGUF_IQ3_XXS, QType::GGUF_IQ3_S,
                                   QType::GGUF_IQ4_XS}) {
            const Matrix gate = make(format, 640, 2560, rng);
            for (const int tokens : {1, 2, 3, 4, 5, 8, 9}) { check(gate, tokens, rng); }
            require(gguf_cpu_supported(format), "an i-quant is not supported");
        }
        require(!gguf_cpu_supported(QType::GGUF_IQ2_XS), "an unimplemented format is supported");
        bool refused = false;
        try {
            GgufCpuActivation a(1, 2560);
            std::vector<std::uint16_t> x(2560 * 2);
            a.prepare_bf16(QType::GGUF_Q2_0, x.data(), 2560, 2);
        } catch (const std::invalid_argument&) { refused = true; }
        require(refused, "an activation above its capacity was accepted");
        std::cout << "gguf_expert_cpu: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gguf_expert_cpu: " << error.what() << '\n';
        return 1;
    }
}
