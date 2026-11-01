#include "models/gemma4/vision_cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_MSC_VER) && defined(NINFER_VISION_CPU_X86)
#    include <intrin.h>
#endif

namespace ninfer::models::gemma4::cpu_vision {
namespace {

constexpr int kTileRows   = 4;
constexpr int kTileTokens = 16;

// Columns [token, token + width) of x as a contiguous [k][kTileTokens] panel, zero-padded.
void pack(const float* x, std::size_t x_stride, int k, int token, int width, float* panel) {
    for (int inner = 0; inner < k; ++inner) {
        const float* source = x + std::size_t(inner) * x_stride + token;
        float* target       = panel + std::size_t(inner) * kTileTokens;
        std::copy_n(source, width, target);
        std::fill(target + width, target + kTileTokens, 0.0F);
    }
}

void gemm(const float* w, std::size_t w_stride, int n, int k, const float* x, std::size_t x_stride,
          int tokens, float* y, std::size_t y_stride) {
    thread_local std::vector<float> panel;
    panel.resize(std::size_t(k) * kTileTokens);
    for (int token = 0; token < tokens; token += kTileTokens) {
        const int width = std::min(kTileTokens, tokens - token);
        pack(x, x_stride, k, token, width, panel.data());
        for (int row = 0; row < n; row += kTileRows) {
            const int rows = std::min(kTileRows, n - row);
            float acc[kTileRows][kTileTokens] = {};
            for (int inner = 0; inner < k; ++inner) {
                const float* xs = panel.data() + std::size_t(inner) * kTileTokens;
                for (int r = 0; r < rows; ++r) {
                    const float weight = w[std::size_t(row + r) * w_stride + inner];
                    for (int t = 0; t < kTileTokens; ++t) { acc[r][t] += weight * xs[t]; }
                }
            }
            for (int r = 0; r < rows; ++r) {
                std::copy_n(acc[r], width, y + std::size_t(row + r) * y_stride + token);
            }
        }
    }
}

void softmax_columns(float* s, std::size_t stride, int rows, int cols, float* sums) {
    thread_local std::vector<float> maxima;
    maxima.assign(std::size_t(cols), -INFINITY);
    for (int r = 0; r < rows; ++r) {
        const float* row = s + std::size_t(r) * stride;
        for (int c = 0; c < cols; ++c) { maxima[c] = std::max(maxima[c], row[c]); }
    }
    std::fill_n(sums, cols, 0.0F);
    for (int r = 0; r < rows; ++r) {
        float* row = s + std::size_t(r) * stride;
        for (int c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - maxima[c]);
            sums[c] += row[c];
        }
    }
}

void geglu(const float* gate, const float* up, float* out, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float x     = gate[i];
        const float inner = 0.7978845608028654F * (x + 0.044715F * x * x * x);
        out[i]            = 0.5F * x * (1.0F + std::tanh(inner)) * up[i];
    }
}

bool avx2_fma() noexcept {
#if defined(NINFER_VISION_CPU_X86)
#    if defined(_MSC_VER)
    int basic[4], extended[4];
    __cpuid(basic, 0);
    if (basic[0] < 7) { return false; }
    __cpuid(basic, 1);
    // FMA, OSXSAVE and AVX, with the OS saving the YMM state.
    if ((basic[2] & (1 << 12)) == 0 || (basic[2] & (1 << 27)) == 0 ||
        (basic[2] & (1 << 28)) == 0 || (_xgetbv(0) & 6) != 6) {
        return false;
    }
    __cpuidex(extended, 7, 0);
    return (extended[1] & (1 << 5)) != 0;
#    else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#    endif
#else
    return false;
#endif
}

} // namespace

const Kernels& portable_kernels() noexcept {
    static const Kernels kernels{gemm, softmax_columns, geglu, "portable"};
    return kernels;
}

const Kernels& kernels() noexcept {
#if defined(NINFER_VISION_CPU_X86)
    static const Kernels& chosen = avx2_fma() ? avx2_kernels() : portable_kernels();
    return chosen;
#else
    return portable_kernels();
#endif
}

} // namespace ninfer::models::gemma4::cpu_vision
