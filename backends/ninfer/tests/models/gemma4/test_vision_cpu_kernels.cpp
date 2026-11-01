// The CPU vision tower's kernels against FP64 oracles, the portable ones and, on a CPU with AVX2
// and FMA, the AVX2 ones:
//  - gemm over strided operands at tile tails (rows not a multiple of 4, columns not of 16) and the
//    tower's widths, leaving everything past its rows and columns untouched;
//  - softmax_columns over scores spread as attention's unscaled ones are (tens of units), every
//    column width from one vector's to past a vector multiple;
//  - geglu over gates through tanh's saturation.

#include "models/gemma4/vision_cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace cv = ninfer::models::gemma4::cpu_vision;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

void test_gemm(const cv::Kernels& kernels, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    struct Shape {
        int n, k, tokens;
    };
    for (const Shape shape : {Shape{1, 3, 1}, Shape{5, 72, 17}, Shape{72, 2340, 128},
                              Shape{1155, 72, 33}, Shape{9, 1152, 16}}) {
        const std::size_t w_stride = shape.k + 5, x_stride = shape.tokens + 3,
                          y_stride = shape.tokens + 7;
        std::vector<float> w(shape.n * w_stride), x(shape.k * x_stride);
        for (float& value : w) { value = normal(rng); }
        for (float& value : x) { value = normal(rng); }
        std::vector<float> y(shape.n * y_stride + 9, 7.0F);
        kernels.gemm(w.data(), w_stride, shape.n, shape.k, x.data(), x_stride, shape.tokens,
                     y.data(), y_stride);
        double worst = 0.0;
        bool kept    = true;
        for (int r = 0; r < shape.n; ++r) {
            for (std::size_t t = 0; t < y_stride; ++t) {
                const float got = y[r * y_stride + t];
                if (t >= std::size_t(shape.tokens)) {
                    kept &= got == 7.0F;
                    continue;
                }
                double want = 0.0, scale = 0.0;
                for (int i = 0; i < shape.k; ++i) {
                    const double product = double(w[r * w_stride + i]) * x[i * x_stride + t];
                    want += product;
                    scale += std::abs(product);
                }
                worst = std::max(worst, std::abs(got - want) / scale);
            }
        }
        kept &= std::all_of(y.end() - 9, y.end(), [](float v) { return v == 7.0F; });
        const std::string label = std::string(kernels.name) + " gemm " + std::to_string(shape.n) +
                                  "x" + std::to_string(shape.k) + "x" + std::to_string(shape.tokens);
        expect(worst <= 1e-5, label + ": error " + std::to_string(worst));
        expect(kept, label + ": wrote past its rows or columns");
    }
}

void test_softmax(const cv::Kernels& kernels, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 30.0F);
    for (const int cols : {1, 8, 13, 128}) {
        const int rows = 300;
        const std::size_t stride = cols + 4;
        std::vector<float> s(rows * stride);
        for (float& value : s) { value = normal(rng); }
        // A column whose scores are all far below zero still normalizes.
        for (int r = 0; r < rows; ++r) { s[r * stride] -= 500.0F; }
        const std::vector<float> scores = s;
        std::vector<float> sums(cols);
        kernels.softmax_columns(s.data(), stride, rows, cols, sums.data());
        // The FP32 difference s - max carries up to an ulp of the larger magnitude, which exp turns
        // into the same relative error; beyond it each value keeps 1e-6 of itself.
        double worst = 0.0;
        for (int c = 0; c < cols; ++c) {
            double best = -INFINITY, sum = 0.0, magnitude = 0.0;
            for (int r = 0; r < rows; ++r) {
                best      = std::max(best, double(scores[r * stride + c]));
                magnitude = std::max(magnitude, std::abs(double(scores[r * stride + c])));
            }
            const double allowance = 1e-6 + 2.0 * magnitude * 0x1p-24;
            for (int r = 0; r < rows; ++r) {
                const double want  = std::exp(double(scores[r * stride + c]) - best);
                const double error = std::abs(s[r * stride + c] - want);
                sum += want;
                // Relative where FP32 represents the value, absolute below that.
                worst = std::max(worst, (want > 1e-30 ? error / want : error) / allowance);
            }
            worst = std::max(worst, std::abs(sums[c] - sum) / sum / allowance);
        }
        expect(worst <= 1.0, std::string(kernels.name) + " softmax_columns over " +
                                 std::to_string(cols) + " columns: error " + std::to_string(worst) +
                                 " of the allowance");
    }
}

void test_geglu(const cv::Kernels& kernels, std::mt19937& rng) {
    std::uniform_real_distribution<float> gate(-12.0F, 12.0F), up(-3.0F, 3.0F);
    const std::size_t count = 1003;
    std::vector<float> g(count), u(count), out(count);
    for (std::size_t i = 0; i < count; ++i) {
        g[i] = i < 40 ? float(i) * 1e-3F - 0.02F : gate(rng); // small gates: tanh's cancellation
        u[i] = up(rng);
    }
    kernels.geglu(g.data(), u.data(), out.data(), count);
    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double x    = g[i];
        const double want = 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) * u[i];
        worst = std::max(worst, std::abs(out[i] - want) / (0.5 * std::abs(x * u[i]) + 1e-30));
    }
    expect(worst <= 1e-6, std::string(kernels.name) + " geglu: error " + std::to_string(worst));
}

void test(const cv::Kernels& kernels) {
    std::mt19937 rng(20261011);
    test_gemm(kernels, rng);
    test_softmax(kernels, rng);
    test_geglu(kernels, rng);
}

} // namespace

int main() {
    test(cv::portable_kernels());
    std::cout << "selected kernels: " << cv::kernels().name << '\n';
#if defined(NINFER_VISION_CPU_X86)
    if (&cv::kernels() == &cv::avx2_kernels()) { test(cv::avx2_kernels()); }
#endif
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " gemma4 vision CPU kernels\n";
    return failures == 0 ? 0 : 1;
}
