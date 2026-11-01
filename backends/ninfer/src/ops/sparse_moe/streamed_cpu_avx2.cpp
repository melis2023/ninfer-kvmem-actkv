// AVX2 rows of streamed_cpu.h: eight of a row's 32 warp lanes per register, so that every lane
// keeps its own chain of fused multiply-adds in the scalar backend's order, and the lanes meet in
// the same tree. Built with AVX2, FMA and F16C, without floating-point contraction.

#include "ops/sparse_moe/streamed_cpu_detail.h"

#include <immintrin.h>

#include <cstring>
#include <vector>

namespace ninfer::ops::detail::stream_cpu_detail {
namespace {

constexpr int kMaxColumns = 64;

std::uint16_t load16(const std::uint8_t* p) noexcept {
    std::uint16_t value;
    std::memcpy(&value, p, 2);
    return value;
}

// The vector F16C form: MSVC does not declare the scalar _cvtsh_ss.
float half(std::uint16_t bits) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// Signed integer codes to weights: (code ^ bias) - bias, times the scale.
__m256 weight(__m256i code, int bias, __m256 scale) noexcept {
    const __m256i b = _mm256_set1_epi32(bias);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_xor_si256(code, b), b)), scale);
}

// Lanes 8 j .. 8 j + 7 of step `step` of `row`: their item-k weights in out[k].
void decode(const StreamCpuPlane& w, int row, int step, int j, __m256 (&out)[8]) noexcept {
    const __m256i nibble = _mm256_set1_epi32(15);
    switch (w.format) {
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16: {
        const std::size_t group = std::size_t(row) * std::size_t(w.k / 64) + std::size_t(4 * step + j);
        const __m256i words =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.codes + group * 32));
        const __m256 scale = _mm256_set1_ps(half(load16(w.scales + group * 2)));
        if (w.format == QType::Q4_G64_FP16) {
            for (int k = 0; k < 8; ++k) {
                out[k] = weight(_mm256_and_si256(_mm256_srli_epi32(words, 4 * k), nibble), 8, scale);
            }
        } else if (w.format == QType::Q5_G64_FP16) {
            const __m256i high = _mm256_cvtepu8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.high + group * 8)));
            const __m256i one = _mm256_set1_epi32(1);
            for (int k = 0; k < 8; ++k) {
                const __m256i low = _mm256_and_si256(_mm256_srli_epi32(words, 4 * k), nibble);
                const __m256i top =
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(high, k), one), 4);
                out[k] = weight(_mm256_or_si256(low, top), 16, scale);
            }
        } else {
            const __m256i high = _mm256_cvtepu16_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(w.high + group * 16)));
            const __m256i three = _mm256_set1_epi32(3);
            for (int k = 0; k < 8; ++k) {
                const __m256i low = _mm256_and_si256(_mm256_srli_epi32(words, 4 * k), nibble);
                const __m256i top =
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(high, 2 * k), three), 4);
                out[k] = weight(_mm256_or_si256(low, top), 32, scale);
            }
        }
        return;
    }
    case QType::Q8_G32_FP16: {
        // Lanes 8 j .. 8 j + 3 read group 8 step + 2 j, the next four the following group: 64
        // contiguous code bytes, eight per lane.
        const std::size_t group = std::size_t(row) * std::size_t(w.k / 32) + std::size_t(8 * step + 2 * j);
        const auto* codes       = w.codes + group * 32;
        const __m256i first     = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
        const __m256i second    = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes + 32));
        const float s0 = half(load16(w.scales + group * 2)), s1 = half(load16(w.scales + group * 2 + 2));
        const __m256 scale = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);
        for (int k = 0; k < 8; ++k) {
            // Byte k of each 8-byte lane into the top byte of its own dword: lanes 0 and 1 of each
            // 128-bit half of `first`, then the same of `second` beside them.
            const char z = char(0x80), a = char(k), b = char(8 + k);
            const __m256i pick = _mm256_setr_epi8(z, z, z, a, z, z, z, b, z, z, z, z, z, z, z, z,
                                                  z, z, z, a, z, z, z, b, z, z, z, z, z, z, z, z);
            const __m256i mixed = _mm256_unpacklo_epi64(_mm256_shuffle_epi8(first, pick),
                                                        _mm256_shuffle_epi8(second, pick));
            // [l0 l1 l4 l5 | l2 l3 l6 l7] -> [l0 .. l7], each byte sign-extended.
            const __m256i ordered = _mm256_permute4x64_epi64(mixed, _MM_SHUFFLE(3, 1, 2, 0));
            out[k] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(ordered, 24)), scale);
        }
        return;
    }
    case QType::NVFP4: {
        const int group = 4 * step + j;
        const __m256i words = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            w.codes + std::size_t(row) * std::size_t(w.k / 2) + std::size_t(group) * 32));
        const float divisor = 1.0F / w.divisors[row / w.divisor_rows];
        float c[4];
        for (int b = 0; b < 4; ++b) {
            c[b] = e4m3_to_float(w.scales[nvfp4_scale_byte(row, group * 4 + b, w.k / 64)]) * divisor;
        }
        const __m256 coefficient = _mm256_setr_ps(c[0], c[0], c[1], c[1], c[2], c[2], c[3], c[3]);
        const __m256 magnitudes  = _mm256_setr_ps(0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F);
        const __m256i seven = _mm256_set1_epi32(7), eight = _mm256_set1_epi32(8);
        for (int k = 0; k < 8; ++k) {
            const __m256i code = _mm256_and_si256(_mm256_srli_epi32(words, 4 * k), nibble);
            const __m256 value = _mm256_permutevar8x32_ps(magnitudes, _mm256_and_si256(code, seven));
            const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(code, eight), 28);
            out[k] = _mm256_mul_ps(_mm256_xor_ps(value, _mm256_castsi256_ps(sign)), coefficient);
        }
        return;
    }
    default: return;
    }
}

// The 32 lanes as the warp's shfl_down tree adds them: lane l takes l + 16, then l + 8, l + 4,
// l + 2 and l + 1.
float tree(const float* lanes) noexcept {
    const __m256 low  = _mm256_add_ps(_mm256_load_ps(lanes), _mm256_load_ps(lanes + 16));
    const __m256 high = _mm256_add_ps(_mm256_load_ps(lanes + 8), _mm256_load_ps(lanes + 24));
    const __m256 eight = _mm256_add_ps(low, high);
    const __m128 four  = _mm_add_ps(_mm256_castps256_ps128(eight), _mm256_extractf128_ps(eight, 1));
    const __m128 two   = _mm_add_ps(four, _mm_movehl_ps(four, four));
    return _mm_cvtss_f32(_mm_add_ss(two, _mm_shuffle_ps(two, two, 1)));
}

} // namespace

void rows_avx2(const StreamCpuPlane& w, const StreamCpuInput& x, int row_begin, int row_end,
               const float* coefficients, std::span<float* const> y) {
    const int steps   = w.k / 256;
    const int columns = x.columns();
    if (columns > kMaxColumns) {
        rows_scalar(w, x, row_begin, row_end, coefficients, y);
        return;
    }
    alignas(32) float lanes[kMaxColumns][32];
    for (int row = row_begin; row < row_end; ++row) {
        for (int t = 0; t < columns; ++t) {
            for (int j = 0; j < 4; ++j) { _mm256_store_ps(lanes[t] + 8 * j, _mm256_setzero_ps()); }
        }
        for (int step = 0; step < steps; ++step) {
            for (int j = 0; j < 4; ++j) {
                __m256 weights[8];
                decode(w, row, step, j, weights);
                for (int t = 0; t < columns; ++t) {
                    const float* values = x.lanes(t) + std::size_t(step) * 256 + 8 * j;
                    __m256 acc          = _mm256_load_ps(lanes[t] + 8 * j);
                    for (int k = 0; k < 8; ++k) {
                        acc = _mm256_fmadd_ps(weights[k], _mm256_loadu_ps(values + 32 * k), acc);
                    }
                    _mm256_store_ps(lanes[t] + 8 * j, acc);
                }
            }
        }
        for (int t = 0; t < columns; ++t) {
            const float dot        = tree(lanes[t]);
            y[std::size_t(t)][row] = coefficients ? coefficients[t] * dot : dot;
        }
    }
}

} // namespace ninfer::ops::detail::stream_cpu_detail
