#pragma once

// The FP32 arithmetic of Gemma 4's CPU vision tower (vision_cpu.cpp): one thread's matrix product
// and the column softmax of attention scores, in portable C++ and, on x86-64 CPUs that have them,
// with AVX2 and FMA instructions (vision_cpu_kernels_avx2.cpp, built for that instruction set and
// chosen at run time).

#include <cstddef>

namespace ninfer::models::gemma4::cpu_vision {

struct Kernels {
    // y[r][t] = sum_i w[r][i] x[i][t] for rows r < n and columns t < tokens, overwriting y: row r
    // of w at w + r * w_stride, row i of x at x + i * x_stride, row r of y at y + r * y_stride.
    void (*gemm)(const float* w, std::size_t w_stride, int n, int k, const float* x,
                 std::size_t x_stride, int tokens, float* y, std::size_t y_stride);
    // Softmax numerators of the columns of s [rows][cols] (row stride `stride`): each value
    // becomes exp(value - its column's maximum), and sums[c] the column's sum of them.
    void (*softmax_columns)(float* s, std::size_t stride, int rows, int cols, float* sums);
    // out[i] = gelu_tanh(gate[i]) * up[i] for i < count.
    void (*geglu)(const float* gate, const float* up, float* out, std::size_t count);
    const char* name;
};

[[nodiscard]] const Kernels& portable_kernels() noexcept;
#if defined(NINFER_VISION_CPU_X86)
[[nodiscard]] const Kernels& avx2_kernels() noexcept;
#endif

// The AVX2 kernels when the CPU has AVX2 and FMA, else the portable ones.
[[nodiscard]] const Kernels& kernels() noexcept;

} // namespace ninfer::models::gemma4::cpu_vision
