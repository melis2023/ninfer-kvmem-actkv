// Built with AVX2 and FMA enabled (sources.cmake); reached only when the CPU has both.

#include "models/gemma4/vision_cpu_kernels.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ninfer::models::gemma4::cpu_vision {
namespace {

constexpr int kTileRows   = 4;
constexpr int kTileTokens = 16;

void pack(const float* x, std::size_t x_stride, int k, int token, int width, float* panel) {
    for (int inner = 0; inner < k; ++inner) {
        const float* source = x + std::size_t(inner) * x_stride + token;
        float* target       = panel + std::size_t(inner) * kTileTokens;
        std::copy_n(source, width, target);
        std::fill(target + width, target + kTileTokens, 0.0F);
    }
}

// Four rows of w by a panel of 16 columns: eight accumulators, two FMAs per row and value.
void gemm(const float* w, std::size_t w_stride, int n, int k, const float* x, std::size_t x_stride,
          int tokens, float* y, std::size_t y_stride) {
    thread_local std::vector<float> panel;
    panel.resize(std::size_t(k) * kTileTokens);
    for (int token = 0; token < tokens; token += kTileTokens) {
        const int width = std::min(kTileTokens, tokens - token);
        pack(x, x_stride, k, token, width, panel.data());
        for (int row = 0; row < n; row += kTileRows) {
            const int rows = std::min(kTileRows, n - row);
            // Rows past n repeat the last one; their results are dropped.
            const float* w0 = w + std::size_t(row) * w_stride;
            const float* w1 = w + std::size_t(row + std::min(1, rows - 1)) * w_stride;
            const float* w2 = w + std::size_t(row + std::min(2, rows - 1)) * w_stride;
            const float* w3 = w + std::size_t(row + std::min(3, rows - 1)) * w_stride;
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
            __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
            __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
            const float* xs = panel.data();
            for (int inner = 0; inner < k; ++inner, xs += kTileTokens) {
                const __m256 x0 = _mm256_loadu_ps(xs), x1 = _mm256_loadu_ps(xs + 8);
                __m256 v = _mm256_broadcast_ss(w0 + inner);
                a00      = _mm256_fmadd_ps(v, x0, a00);
                a01      = _mm256_fmadd_ps(v, x1, a01);
                v        = _mm256_broadcast_ss(w1 + inner);
                a10      = _mm256_fmadd_ps(v, x0, a10);
                a11      = _mm256_fmadd_ps(v, x1, a11);
                v        = _mm256_broadcast_ss(w2 + inner);
                a20      = _mm256_fmadd_ps(v, x0, a20);
                a21      = _mm256_fmadd_ps(v, x1, a21);
                v        = _mm256_broadcast_ss(w3 + inner);
                a30      = _mm256_fmadd_ps(v, x0, a30);
                a31      = _mm256_fmadd_ps(v, x1, a31);
            }
            alignas(32) float out[kTileRows][kTileTokens];
            _mm256_store_ps(out[0], a00);
            _mm256_store_ps(out[0] + 8, a01);
            _mm256_store_ps(out[1], a10);
            _mm256_store_ps(out[1] + 8, a11);
            _mm256_store_ps(out[2], a20);
            _mm256_store_ps(out[2] + 8, a21);
            _mm256_store_ps(out[3], a30);
            _mm256_store_ps(out[3] + 8, a31);
            for (int r = 0; r < rows; ++r) {
                std::copy_n(out[r], width, y + std::size_t(row + r) * y_stride + token);
            }
        }
    }
}

// exp(x) for x <= 0 as Cephes' expf computes it: x = n ln2 + f with |f| <= ln2 / 2, a degree-6
// polynomial for e^f and 2^n built in the exponent bits, a relative error of about 1e-7; below
// -87.3 the result is about the smallest normal, which a softmax sum cannot notice.
__m256 exp_nonpositive(__m256 x) {
    x              = _mm256_max_ps(x, _mm256_set1_ps(-87.3F));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341F)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375F), x);
    x = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4F), x);
    __m256 p = _mm256_set1_ps(1.9875691500e-4F);
    p        = _mm256_fmadd_ps(p, x, _mm256_set1_ps(1.3981999507e-3F));
    p        = _mm256_fmadd_ps(p, x, _mm256_set1_ps(8.3334519073e-3F));
    p        = _mm256_fmadd_ps(p, x, _mm256_set1_ps(4.1665795894e-2F));
    p        = _mm256_fmadd_ps(p, x, _mm256_set1_ps(1.6666665459e-1F));
    p        = _mm256_fmadd_ps(p, x, _mm256_set1_ps(5.0000001201e-1F));
    p        = _mm256_fmadd_ps(p, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0F)));
    const __m256i scale = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(scale));
}

void softmax_columns(float* s, std::size_t stride, int rows, int cols, float* sums) {
    thread_local std::vector<float> maxima;
    maxima.assign(std::size_t(cols), -INFINITY);
    const int vector_cols = cols / 8 * 8;
    for (int r = 0; r < rows; ++r) {
        const float* row = s + std::size_t(r) * stride;
        int c            = 0;
        for (; c < vector_cols; c += 8) {
            _mm256_storeu_ps(maxima.data() + c, _mm256_max_ps(_mm256_loadu_ps(maxima.data() + c),
                                                              _mm256_loadu_ps(row + c)));
        }
        for (; c < cols; ++c) { maxima[c] = std::max(maxima[c], row[c]); }
    }
    std::fill_n(sums, cols, 0.0F);
    for (int r = 0; r < rows; ++r) {
        float* row = s + std::size_t(r) * stride;
        int c      = 0;
        for (; c < vector_cols; c += 8) {
            const __m256 e = exp_nonpositive(
                _mm256_sub_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(maxima.data() + c)));
            _mm256_storeu_ps(row + c, e);
            _mm256_storeu_ps(sums + c, _mm256_add_ps(_mm256_loadu_ps(sums + c), e));
        }
        for (; c < cols; ++c) {
            row[c] = std::exp(row[c] - maxima[c]);
            sums[c] += row[c];
        }
    }
}

// tanh(z) = sign(z) (1 - t) / (1 + t) with t = exp(-2 |z|): an absolute error of about 1e-7.
void geglu(const float* gate, const float* up, float* out, std::size_t count) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0F);
    std::size_t i         = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 x = _mm256_loadu_ps(gate + i);
        const __m256 cube = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        const __m256 inner = _mm256_mul_ps(
            _mm256_set1_ps(0.7978845608028654F),
            _mm256_fmadd_ps(cube, _mm256_set1_ps(0.044715F), x));
        const __m256 magnitude = _mm256_andnot_ps(sign_bit, inner);
        const __m256 t = exp_nonpositive(_mm256_mul_ps(magnitude, _mm256_set1_ps(-2.0F)));
        const __m256 one = _mm256_set1_ps(1.0F);
        __m256 tanh = _mm256_div_ps(_mm256_sub_ps(one, t), _mm256_add_ps(one, t));
        tanh        = _mm256_or_ps(tanh, _mm256_and_ps(sign_bit, inner));
        const __m256 gelu = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5F), x),
                                          _mm256_add_ps(one, tanh));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(gelu, _mm256_loadu_ps(up + i)));
    }
    for (; i < count; ++i) {
        const float x     = gate[i];
        const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
        out[i]            = 0.5F * x * (1.0F + std::tanh(inner)) * up[i];
    }
}

} // namespace

const Kernels& avx2_kernels() noexcept {
    static const Kernels kernels{gemm, softmax_columns, geglu, "avx2"};
    return kernels;
}

} // namespace ninfer::models::gemma4::cpu_vision
