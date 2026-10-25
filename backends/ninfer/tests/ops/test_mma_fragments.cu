// The warp MMA helpers of ops/common/mma.cuh against an FP64 (or exact integer) oracle, through the
// fragment layouts the PTX ISA specifies for each shape. On sm_8x and newer this exercises the
// native instructions; on sm_75 it qualifies the Turing implementations: FP16 m16n8k16 as two
// m16n8k8, INT8 m16n8k32 as four m8n8k16, and BF16 / TF32 tiles that run on FP16 m16n8k8 when every
// operand is exactly an FP16 value and in FP32 otherwise. The cases cover random tiles, the exact
// edges of the FP16-representable exponent window on both sides, out-of-window values in every lane
// or in a single lane (the warp has to agree on one route), subnormals, signed zeros and
// non-finite values.
#include "ops/common/mma.cuh"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer::test;

constexpr int kM = 16;
constexpr int kN = 8;

// ---- Fragment layouts (PTX ISA, "Matrix fragments for mma.m16n8k*") ----------------------------
// g = lane / 4, t = lane % 4.

// m16n8k16 .f16/.bf16 A: register r holds two consecutive k of row g + 8 * (r & 1), starting at
// k = 2t + 8 * (r >> 1). B: register r holds k = 2t + 8r, 2t + 8r + 1 of column g.
__device__ unsigned pack16(const std::uint16_t* m, int stride, int row, int col) {
    return unsigned(m[row * stride + col]) | (unsigned(m[row * stride + col + 1]) << 16);
}

__device__ unsigned frag_a_k16(const std::uint16_t* a, int lane, int r) {
    const int g = lane / 4, t = lane % 4;
    return pack16(a, 16, g + 8 * (r & 1), 2 * t + 8 * (r >> 1));
}

// bt is B transposed, [n][k].
__device__ unsigned frag_b_k16(const std::uint16_t* bt, int lane, int r) {
    const int g = lane / 4, t = lane % 4;
    return pack16(bt, 16, g, 2 * t + 8 * r);
}

// m16n8k32 .s8 A: register r holds four consecutive k of row g + 8 * (r & 1), starting at
// k = 4t + 16 * (r >> 1). B: register r holds k = 4t + 16r .. + 3 of column g.
__device__ unsigned pack8(const std::int8_t* m, int stride, int row, int col) {
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= unsigned(std::uint8_t(m[row * stride + col + i])) << (8 * i);
    }
    return v;
}

__device__ unsigned frag_a_s8(const std::int8_t* a, int lane, int r) {
    const int g = lane / 4, t = lane % 4;
    return pack8(a, 32, g + 8 * (r & 1), 4 * t + 16 * (r >> 1));
}

__device__ unsigned frag_b_s8(const std::int8_t* bt, int lane, int r) {
    const int g = lane / 4, t = lane % 4;
    return pack8(bt, 32, g, 4 * t + 16 * r);
}

// m16n8k8 .tf32 A: a0 (row g, k t), a1 (row g + 8, k t), a2 (row g, k t + 4),
// a3 (row g + 8, k t + 4). B: b0 (k t, column g), b1 (k t + 4, column g).
__device__ unsigned frag_a_tf32(const std::uint32_t* a, int lane, int r) {
    const int g = lane / 4, t = lane % 4;
    return a[(g + 8 * (r & 1)) * 8 + t + 4 * (r >> 1)];
}

__device__ unsigned frag_b_tf32(const std::uint32_t* bt, int lane, int r) {
    const int g = lane / 4, t = lane % 4;
    return bt[g * 8 + t + 4 * r];
}

// C/D: c0, c1 at row g, columns 2t, 2t + 1; c2, c3 at row g + 8.
__device__ int frag_c_index(int lane, int i) {
    const int g = lane / 4, t = lane % 4;
    return (g + 8 * (i >> 1)) * kN + 2 * t + (i & 1);
}

enum class Kind { bf16, f16, f16_f16acc, tf32, s8 };

struct Tiles {
    std::vector<std::uint16_t> a16, b16t; // 16-bit operands, A [16][16], B^T [8][16]
    std::vector<std::uint32_t> a32, b32t; // TF32 operands, A [16][8], B^T [8][8]
    std::vector<std::int8_t> a8, b8t;     // INT8 operands, A [16][32], B^T [8][32]
    std::vector<float> c;                 // [16][8] accumulator input
    std::vector<int> ci;
};

__global__ void run(Kind kind, int tiles, const std::uint16_t* a16, const std::uint16_t* b16t,
                    const std::uint32_t* a32, const std::uint32_t* b32t, const std::int8_t* a8,
                    const std::int8_t* b8t, const float* c_in, const int* ci_in, float* c_out,
                    int* ci_out) {
    const int tile = blockIdx.x;
    const int lane = threadIdx.x;
    if (tile >= tiles) { return; }
    const float* c = c_in + tile * kM * kN;
    float* out     = c_out + tile * kM * kN;
    float d[4];
    for (int i = 0; i < 4; ++i) { d[i] = c[frag_c_index(lane, i)]; }
    if (kind == Kind::bf16 || kind == Kind::f16 || kind == Kind::f16_f16acc) {
        const std::uint16_t* a = a16 + tile * 16 * 16;
        const std::uint16_t* b = b16t + tile * 8 * 16;
        const unsigned a0 = frag_a_k16(a, lane, 0), a1 = frag_a_k16(a, lane, 1);
        const unsigned a2 = frag_a_k16(a, lane, 2), a3 = frag_a_k16(a, lane, 3);
        const unsigned b0 = frag_b_k16(b, lane, 0), b1 = frag_b_k16(b, lane, 1);
        if (kind == Kind::bf16) {
            ninfer::ops::mma_bf16(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b0, b1);
        } else if (kind == Kind::f16) {
            ninfer::ops::mma_f16(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b0, b1);
        } else {
            const __half2 lo = __floats2half2_rn(d[0], d[1]), hi = __floats2half2_rn(d[2], d[3]);
            unsigned h0 = *reinterpret_cast<const unsigned*>(&lo);
            unsigned h1 = *reinterpret_cast<const unsigned*>(&hi);
            ninfer::ops::mma_f16_f16acc(h0, h1, a0, a1, a2, a3, b0, b1);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
            d[0]            = f0.x;
            d[1]            = f0.y;
            d[2]            = f1.x;
            d[3]            = f1.y;
        }
    } else if (kind == Kind::tf32) {
        const std::uint32_t* a = a32 + tile * 16 * 8;
        const std::uint32_t* b = b32t + tile * 8 * 8;
        ninfer::ops::mma_tf32_bits(d[0], d[1], d[2], d[3], frag_a_tf32(a, lane, 0),
                                   frag_a_tf32(a, lane, 1), frag_a_tf32(a, lane, 2),
                                   frag_a_tf32(a, lane, 3), frag_b_tf32(b, lane, 0),
                                   frag_b_tf32(b, lane, 1));
    } else {
        const std::int8_t* a = a8 + tile * 16 * 32;
        const std::int8_t* b = b8t + tile * 8 * 32;
        const int* ci        = ci_in + tile * kM * kN;
        int* cio             = ci_out + tile * kM * kN;
        int s[4];
        for (int i = 0; i < 4; ++i) { s[i] = ci[frag_c_index(lane, i)]; }
        ninfer::ops::mma_s8(s[0], s[1], s[2], s[3], frag_a_s8(a, lane, 0), frag_a_s8(a, lane, 1),
                            frag_a_s8(a, lane, 2), frag_a_s8(a, lane, 3), frag_b_s8(b, lane, 0),
                            frag_b_s8(b, lane, 1));
        for (int i = 0; i < 4; ++i) { cio[frag_c_index(lane, i)] = s[i]; }
        return;
    }
    for (int i = 0; i < 4; ++i) { out[frag_c_index(lane, i)] = d[i]; }
}

// ---- Host values -------------------------------------------------------------------------------

float bf16_value(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t(bits) << 16); }

float f16_value(std::uint16_t bits) { return __half2float(std::bit_cast<__half>(bits)); }

// The value a TF32 operand represents: its low 13 mantissa bits are not part of it.
float tf32_value(std::uint32_t bits) { return std::bit_cast<float>(bits & 0xffffe000u); }

std::uint16_t f16_bits(float v) { return std::bit_cast<std::uint16_t>(__float2half_rn(v)); }

// A value with the given unbiased exponent and random significand bits; the sign is random.
std::uint32_t float_bits_with_exponent(std::mt19937& rng, int exponent) {
    const std::uint32_t biased = std::uint32_t(exponent + 127);
    return (rng() & 0x807fffffu) | (biased << 23);
}

struct Case {
    std::string name;
    Kind kind;
    // Fills one tile's operands; `tile` indexes the tile inside the case.
    std::function<void(std::mt19937&, int tile, Tiles&, std::size_t)> fill;
};

double reference(const Tiles& t, Kind kind, int tile, int row, int col, double& magnitude) {
    double sum = 0.0;
    magnitude  = 0.0;
    if (kind == Kind::tf32) {
        for (int k = 0; k < 8; ++k) {
            const double p = double(tf32_value(t.a32[(tile * 16 + row) * 8 + k])) *
                             double(tf32_value(t.b32t[(tile * 8 + col) * 8 + k]));
            sum += p;
            magnitude += std::fabs(p);
        }
    } else {
        for (int k = 0; k < 16; ++k) {
            const std::uint16_t x = t.a16[(tile * 16 + row) * 16 + k];
            const std::uint16_t y = t.b16t[(tile * 8 + col) * 16 + k];
            const double p = kind == Kind::bf16 ? double(bf16_value(x)) * double(bf16_value(y))
                                                : double(f16_value(x)) * double(f16_value(y));
            sum += p;
            magnitude += std::fabs(p);
        }
    }
    const double c = t.c[(tile * kM + row) * kN + col];
    magnitude += std::fabs(c);
    return sum + c;
}

int check_case(const Case& test, int tiles, std::mt19937& rng) {
    Tiles t;
    t.a16.assign(std::size_t(tiles) * 16 * 16, 0);
    t.b16t.assign(std::size_t(tiles) * 8 * 16, 0);
    t.a32.assign(std::size_t(tiles) * 16 * 8, 0);
    t.b32t.assign(std::size_t(tiles) * 8 * 8, 0);
    t.a8.assign(std::size_t(tiles) * 16 * 32, 0);
    t.b8t.assign(std::size_t(tiles) * 8 * 32, 0);
    t.c.assign(std::size_t(tiles) * kM * kN, 0.0F);
    t.ci.assign(std::size_t(tiles) * kM * kN, 0);
    for (int tile = 0; tile < tiles; ++tile) { test.fill(rng, tile, t, std::size_t(tile)); }

    auto upload = [](const auto& host) {
        using T   = typename std::decay_t<decltype(host)>::value_type;
        T* device = nullptr;
        cuda_check(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
        cuda_check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
                   "upload");
        return device;
    };
    auto* da16  = upload(t.a16);
    auto* db16  = upload(t.b16t);
    auto* da32  = upload(t.a32);
    auto* db32  = upload(t.b32t);
    auto* da8   = upload(t.a8);
    auto* db8   = upload(t.b8t);
    auto* dc    = upload(t.c);
    auto* dci   = upload(t.ci);
    float* dout = nullptr;
    int* dciout = nullptr;
    cuda_check(cudaMalloc(&dout, t.c.size() * sizeof(float)), "cudaMalloc out");
    cuda_check(cudaMalloc(&dciout, t.ci.size() * sizeof(int)), "cudaMalloc int out");
    run<<<tiles, 32>>>(test.kind, tiles, da16, db16, da32, db32, da8, db8, dc, dci, dout, dciout);
    cuda_check_last_launch(test.name.c_str());
    std::vector<float> out(t.c.size());
    std::vector<int> iout(t.ci.size());
    cuda_check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "download");
    cuda_check(cudaMemcpy(iout.data(), dciout, iout.size() * sizeof(int), cudaMemcpyDeviceToHost),
               "download int");
    for (void* p : {static_cast<void*>(da16), static_cast<void*>(db16), static_cast<void*>(da32),
                    static_cast<void*>(db32), static_cast<void*>(da8), static_cast<void*>(db8),
                    static_cast<void*>(dc), static_cast<void*>(dci), static_cast<void*>(dout),
                    static_cast<void*>(dciout)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }

    std::size_t failures = 0;
    double worst         = 0.0;
    for (int tile = 0; tile < tiles; ++tile) {
        for (int row = 0; row < kM; ++row) {
            for (int col = 0; col < kN; ++col) {
                const std::size_t at = (std::size_t(tile) * kM + row) * kN + col;
                if (test.kind == Kind::s8) {
                    std::int64_t expected = t.ci[at];
                    for (int k = 0; k < 32; ++k) {
                        expected += std::int64_t(t.a8[(tile * 16 + row) * 32 + k]) *
                                    std::int64_t(t.b8t[(tile * 8 + col) * 32 + k]);
                    }
                    if (iout[at] != expected && ++failures <= 8) {
                        std::cerr << test.name << " tile " << tile << " (" << row << "," << col
                                  << "): " << iout[at] << " expected " << expected << '\n';
                    }
                    continue;
                }
                double magnitude      = 0.0;
                const double expected = reference(t, test.kind, tile, row, col, magnitude);
                const double got      = out[at];
                // Non-finite policy: a non-finite oracle value must come back as the same
                // non-finite class (NaN, or an infinity of the same sign); finite must stay finite.
                if (!std::isfinite(expected) || !std::isfinite(got)) {
                    const bool same = (std::isnan(expected) && std::isnan(got)) ||
                                      (std::isinf(expected) && std::isinf(got) &&
                                       std::signbit(expected) == std::signbit(got));
                    if (!same && ++failures <= 8) {
                        std::cerr << test.name << " tile " << tile << " (" << row << "," << col
                                  << "): " << got << " expected " << expected << '\n';
                    }
                    continue;
                }
                // Exact products summed in FP32 (FP16 for the FP16-accumulating form): the error of
                // a correct tile is bounded by its absolute sum times a few ulps per addition. A
                // wrong fragment layout misplaces whole products, orders of magnitude above this.
                const double unit =
                    test.kind == Kind::f16_f16acc ? std::ldexp(1.0, -8) : std::ldexp(1.0, -18);
                const double bound = unit * magnitude +
                                     (test.kind == Kind::f16_f16acc ? std::ldexp(1.0, -24) : 1e-38);
                const double error = std::fabs(got - expected);
                if (magnitude > 0.0) { worst = std::fmax(worst, error / magnitude); }
                if (error > bound && ++failures <= 8) {
                    std::cerr << test.name << " tile " << tile << " (" << row << "," << col
                              << "): " << got << " expected " << expected << " |error| " << error
                              << " bound " << bound << '\n';
                }
            }
        }
    }
    std::cout << test.name << ": " << tiles << " tiles, worst |error|/sum|terms| " << worst
              << (failures == 0 ? "" : " FAILED") << '\n';
    return failures == 0 ? 0 : 1;
}

// ---- Operand generators ------------------------------------------------------------------------

std::uint16_t random_bf16(std::mt19937& rng, int lo_exp, int hi_exp) {
    const int e = lo_exp + int(rng() % unsigned(hi_exp - lo_exp + 1));
    return std::uint16_t(float_bits_with_exponent(rng, e) >> 16);
}

std::uint32_t random_tf32(std::mt19937& rng, int lo_exp, int hi_exp) {
    const int e = lo_exp + int(rng() % unsigned(hi_exp - lo_exp + 1));
    return float_bits_with_exponent(rng, e); // low bits left set: the MMA must ignore them
}

std::uint16_t random_f16(std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    return f16_bits(normal(rng));
}

void fill_c(std::mt19937& rng, Tiles& t, std::size_t tile, float scale) {
    std::normal_distribution<float> normal(0.0F, scale);
    for (int i = 0; i < kM * kN; ++i) { t.c[tile * kM * kN + i] = normal(rng); }
}

// Operand of one 16-bit tile: `pick(rng, index)` returns its bits.
template <class Pick>
void fill16(Tiles& t, std::size_t tile, std::mt19937& rng, Pick pick) {
    for (int i = 0; i < 16 * 16; ++i) { t.a16[tile * 256 + i] = pick(rng, i); }
    for (int i = 0; i < 8 * 16; ++i) { t.b16t[tile * 128 + i] = pick(rng, 256 + i); }
}

template <class Pick>
void fill32(Tiles& t, std::size_t tile, std::mt19937& rng, Pick pick) {
    for (int i = 0; i < 16 * 8; ++i) { t.a32[tile * 128 + i] = pick(rng, i); }
    for (int i = 0; i < 8 * 8; ++i) { t.b32t[tile * 64 + i] = pick(rng, 128 + i); }
}

std::vector<Case> cases() {
    using Rng  = std::mt19937;
    using Fill = decltype(Case::fill);
    std::vector<Case> all;
    const auto add = [&all](const char* name, Kind kind, Fill fill) {
        all.push_back({name, kind, std::move(fill)});
    };
    // BF16: inside the exact FP16 window [-17, 15]; at both edges; past them; subnormal BF16; one
    // out-of-window value in a single lane's fragment; zeros of both signs; non-finite values.
    add("bf16 normal range", Kind::bf16, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng, [](Rng& r, int) { return random_bf16(r, -6, 4); });
        fill_c(rng, t, s, 4.0F);
    });
    add("bf16 window edges", Kind::bf16, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng,
               [](Rng& r, int i) { return random_bf16(r, i % 2 ? 15 : -17, i % 2 ? 15 : -17); });
        fill_c(rng, t, s, 1.0F);
    });
    add("bf16 past the window", Kind::bf16, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng, [](Rng& r, int) { return random_bf16(r, -40, 30); });
        fill_c(rng, t, s, 1.0F);
    });
    add("bf16 just past the window", Kind::bf16, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng,
               [](Rng& r, int i) { return random_bf16(r, i % 2 ? 16 : -18, i % 2 ? 16 : -18); });
        fill_c(rng, t, s, 1.0F);
    });
    add("bf16 subnormals and zeros", Kind::bf16, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng, [](Rng& r, int i) -> std::uint16_t {
            switch (r() % 4) {
            case 0:
                return std::uint16_t((r() & 0x8000u) | (r() & 0x7fu)); // subnormal
            case 1:
                return std::uint16_t(r() & 0x8000u); // +-0
            default:
                return random_bf16(r, -5, 5);
            }
        });
        fill_c(rng, t, s, 1.0F);
    });
    add("bf16 one lane past the window", Kind::bf16,
        [](Rng& rng, int tile, Tiles& t, std::size_t s) {
            fill16(t, s, rng, [](Rng& r, int) { return random_bf16(r, -4, 4); });
            // One element per tile, at a different position each tile: only its owner lane sees
            // it, the warp must still take one route.
            const int at            = (tile * 37) % (256 + 128);
            const std::uint16_t big = std::uint16_t(float_bits_with_exponent(rng, 20) >> 16);
            if (at < 256) {
                t.a16[s * 256 + at] = big;
            } else {
                t.b16t[s * 128 + at - 256] = big;
            }
            fill_c(rng, t, s, 1.0F);
        });
    add("bf16 non-finite", Kind::bf16, [](Rng& rng, int tile, Tiles& t, std::size_t s) {
        fill16(t, s, rng, [](Rng& r, int) { return random_bf16(r, -3, 3); });
        const std::uint16_t special[]      = {0x7f80u, 0xff80u, 0x7fc0u};
        t.a16[s * 256 + (tile * 11) % 256] = special[tile % 3];
        fill_c(rng, t, s, 1.0F);
    });
    // FP16 operands (attention): normal values, plus subnormals and zeros.
    add("f16", Kind::f16, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng, [](Rng& r, int i) -> std::uint16_t {
            if (i % 13 == 0) { return std::uint16_t(r() & 0x83ffu); } // subnormal or zero
            return random_f16(r);
        });
        fill_c(rng, t, s, 4.0F);
    });
    add("f16 f16-accumulate", Kind::f16_f16acc, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill16(t, s, rng, [](Rng& r, int) { return random_f16(r); });
        fill_c(rng, t, s, 2.0F);
        // The accumulator starts from FP16 values.
        for (int i = 0; i < kM * kN; ++i) {
            t.c[s * kM * kN + i] = f16_value(f16_bits(t.c[s * kM * kN + i]));
        }
    });
    // TF32: inside [-14, 15], at both edges, past them, one lane, zeros/subnormals, non-finite.
    add("tf32 normal range", Kind::tf32, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill32(t, s, rng, [](Rng& r, int) { return random_tf32(r, -8, 6); });
        fill_c(rng, t, s, 4.0F);
    });
    add("tf32 window edges", Kind::tf32, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill32(t, s, rng,
               [](Rng& r, int i) { return random_tf32(r, i % 2 ? 15 : -14, i % 2 ? 15 : -14); });
        fill_c(rng, t, s, 1.0F);
    });
    add("tf32 just past the window", Kind::tf32, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill32(t, s, rng,
               [](Rng& r, int i) { return random_tf32(r, i % 2 ? 16 : -15, i % 2 ? 16 : -15); });
        fill_c(rng, t, s, 1.0F);
    });
    add("tf32 past the window", Kind::tf32, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill32(t, s, rng, [](Rng& r, int) { return random_tf32(r, -60, 40); });
        fill_c(rng, t, s, 1.0F);
    });
    add("tf32 one lane past the window", Kind::tf32,
        [](Rng& rng, int tile, Tiles& t, std::size_t s) {
            fill32(t, s, rng, [](Rng& r, int) { return random_tf32(r, -4, 4); });
            const int at             = (tile * 29) % (128 + 64);
            const std::uint32_t tiny = float_bits_with_exponent(rng, -30);
            if (at < 128) {
                t.a32[s * 128 + at] = tiny;
            } else {
                t.b32t[s * 64 + at - 128] = tiny;
            }
            fill_c(rng, t, s, 1.0F);
        });
    add("tf32 zeros and subnormals", Kind::tf32, [](Rng& rng, int, Tiles& t, std::size_t s) {
        fill32(t, s, rng, [](Rng& r, int) -> std::uint32_t {
            switch (r() % 4) {
            case 0:
                return (r() & 0x80000000u) | (r() & 0x7fffffu); // subnormal
            case 1:
                return r() & 0x80000000u; // +-0
            default:
                return random_tf32(r, -5, 5);
            }
        });
        fill_c(rng, t, s, 1.0F);
    });
    add("tf32 non-finite", Kind::tf32, [](Rng& rng, int tile, Tiles& t, std::size_t s) {
        fill32(t, s, rng, [](Rng& r, int) { return random_tf32(r, -3, 3); });
        const std::uint32_t special[]    = {0x7f800000u, 0xff800000u, 0x7fc00000u};
        t.b32t[s * 64 + (tile * 7) % 64] = special[tile % 3];
        fill_c(rng, t, s, 1.0F);
    });
    // INT8: the whole range, exact.
    add("s8", Kind::s8, [](Rng& rng, int, Tiles& t, std::size_t s) {
        for (int i = 0; i < 16 * 32; ++i) { t.a8[s * 512 + i] = std::int8_t(rng()); }
        for (int i = 0; i < 8 * 32; ++i) { t.b8t[s * 256 + i] = std::int8_t(rng()); }
        for (int i = 0; i < kM * kN; ++i) { t.ci[s * kM * kN + i] = int(rng() % 200001u) - 100000; }
    });
    return all;
}

// An output's bits must depend on its own row of A and column of B only: the record and snapshot
// executions of ReplaySSM tile the same product differently and must agree bit for bit. Each pair
// of tiles differs in one out-of-FP16-window operand of A (row 3) and one of B (column 5); every
// output outside that row and column must come back identical in both tiles.
int check_neighbor_independence(Kind kind, std::mt19937& rng) {
    constexpr int kPairs = 256;
    Tiles t;
    t.a16.assign(std::size_t(2 * kPairs) * 16 * 16, 0);
    t.b16t.assign(std::size_t(2 * kPairs) * 8 * 16, 0);
    t.a32.assign(std::size_t(2 * kPairs) * 16 * 8, 0);
    t.b32t.assign(std::size_t(2 * kPairs) * 8 * 8, 0);
    t.a8.assign(std::size_t(2 * kPairs) * 16 * 32, 0);
    t.b8t.assign(std::size_t(2 * kPairs) * 8 * 32, 0);
    t.c.assign(std::size_t(2 * kPairs) * kM * kN, 0.0F);
    t.ci.assign(std::size_t(2 * kPairs) * kM * kN, 0);
    constexpr int kRow = 3, kCol = 5;
    for (int p = 0; p < kPairs; ++p) {
        const std::size_t clean = 2 * p, mixed = 2 * p + 1;
        fill_c(rng, t, clean, 2.0F);
        std::copy_n(t.c.begin() + clean * kM * kN, kM * kN, t.c.begin() + mixed * kM * kN);
        if (kind == Kind::bf16) {
            fill16(t, clean, rng, [](std::mt19937& r, int) { return random_bf16(r, -6, 4); });
            std::copy_n(t.a16.begin() + clean * 256, 256, t.a16.begin() + mixed * 256);
            std::copy_n(t.b16t.begin() + clean * 128, 128, t.b16t.begin() + mixed * 128);
            t.a16[mixed * 256 + kRow * 16 + p % 16]        = random_bf16(rng, 20, 20);
            t.b16t[mixed * 128 + kCol * 16 + (p * 7) % 16] = random_bf16(rng, -30, -30);
        } else {
            fill32(t, clean, rng, [](std::mt19937& r, int) { return random_tf32(r, -6, 4); });
            std::copy_n(t.a32.begin() + clean * 128, 128, t.a32.begin() + mixed * 128);
            std::copy_n(t.b32t.begin() + clean * 64, 64, t.b32t.begin() + mixed * 64);
            t.a32[mixed * 128 + kRow * 8 + p % 8]       = random_tf32(rng, 20, 20);
            t.b32t[mixed * 64 + kCol * 8 + (p * 3) % 8] = random_tf32(rng, -30, -30);
        }
    }
    auto upload = [](const auto& host) {
        using T   = typename std::decay_t<decltype(host)>::value_type;
        T* device = nullptr;
        cuda_check(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
        cuda_check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
                   "upload");
        return device;
    };
    auto* da16  = upload(t.a16);
    auto* db16  = upload(t.b16t);
    auto* da32  = upload(t.a32);
    auto* db32  = upload(t.b32t);
    auto* da8   = upload(t.a8);
    auto* db8   = upload(t.b8t);
    auto* dc    = upload(t.c);
    auto* dci   = upload(t.ci);
    float* dout = nullptr;
    int* dciout = nullptr;
    cuda_check(cudaMalloc(&dout, t.c.size() * sizeof(float)), "cudaMalloc out");
    cuda_check(cudaMalloc(&dciout, t.ci.size() * sizeof(int)), "cudaMalloc int out");
    run<<<2 * kPairs, 32>>>(kind, 2 * kPairs, da16, db16, da32, db32, da8, db8, dc, dci, dout,
                            dciout);
    cuda_check_last_launch("neighbor independence");
    std::vector<float> out(t.c.size());
    cuda_check(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "download");
    for (void* q : {static_cast<void*>(da16), static_cast<void*>(db16), static_cast<void*>(da32),
                    static_cast<void*>(db32), static_cast<void*>(da8), static_cast<void*>(db8),
                    static_cast<void*>(dc), static_cast<void*>(dci), static_cast<void*>(dout),
                    static_cast<void*>(dciout)}) {
        cuda_check(cudaFree(q), "cudaFree");
    }
    std::size_t failures = 0;
    for (int p = 0; p < kPairs; ++p) {
        for (int row = 0; row < kM; ++row) {
            for (int col = 0; col < kN; ++col) {
                if (row == kRow || col == kCol) { continue; }
                const float clean = out[(std::size_t(2 * p) * kM + row) * kN + col];
                const float mixed = out[(std::size_t(2 * p + 1) * kM + row) * kN + col];
                if (std::bit_cast<std::uint32_t>(clean) != std::bit_cast<std::uint32_t>(mixed) &&
                    ++failures <= 8) {
                    std::cerr << (kind == Kind::bf16 ? "bf16" : "tf32") << " pair " << p << " ("
                              << row << "," << col << "): " << clean << " vs " << mixed
                              << " beside an out-of-window row and column\n";
                }
            }
        }
    }
    std::cout << (kind == Kind::bf16 ? "bf16" : "tf32") << " neighbor independence: " << kPairs
              << " tile pairs" << (failures == 0 ? "" : " FAILED") << '\n';
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        std::mt19937 rng(20261010);
        int failed = 0;
        for (const Case& test : cases()) { failed += check_case(test, 512, rng); }
        failed += check_neighbor_independence(Kind::bf16, rng);
        failed += check_neighbor_independence(Kind::tf32, rng);
        if (failed != 0) {
            std::cerr << failed << " MMA fragment case(s) failed\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mma fragments test: " << error.what() << '\n';
        return 1;
    }
}
