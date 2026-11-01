#include "ops/linear/linear_test_common.h"
#include "ops/linear/ptq1/ptq1_dispatch.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;
namespace quantized_weight = ninfer::test::quantized_weight;

quantized_weight::PackedWeight make_ptq1_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_ptq1_patterned_weight(n, k, seed);
}

constexpr Invocation a8(std::int32_t t) {
    return {t, CallForm::Policy, ops::LinearPolicy::AllowA8Int};
}

// PTQ1 keeps A16 below 33 tokens and uses its own padded INT8 tile above that boundary.
int ptq1_a8_conformance() {
    int failures = 0;

    constexpr std::array kWidths{a8(1),   a8(2),   a8(3),    a8(4),   a8(5),   a8(7),   a8(8),
                                 a8(9),   a8(12),  a8(15),   a8(16),  a8(17),  a8(24),  a8(32),
                                 a8(33),  a8(40),  a8(63),   a8(64),  a8(65),  a8(95),  a8(100),
                                 a8(127), a8(128), a8(129),  a8(192), a8(193), a8(256), a8(300),
                                 a8(385), a8(600), a8(1007), a8(1024), Invocation{33, CallForm::Policy, ops::LinearPolicy::AllowA8Int, true},
                                 Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8Int, true}};

    failures += run_shape("PTQ1_A8", ActivationCompute::A8, make_ptq1_weight,
                          {1024, 5120, 301U, Comparison::Full, true, kWidths});
    failures += run_shape("PTQ1_A8", ActivationCompute::A8, make_ptq1_weight,
                          {4096, 5120, 303U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("PTQ1_A8", ActivationCompute::A8, make_ptq1_weight,
                          {7168, 5120, 305U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("PTQ1_A8", ActivationCompute::A8, make_ptq1_weight,
                          {34816, 5120, 307U, Comparison::SampledRows, false, kWidths});
    failures += run_shape("PTQ1_A8", ActivationCompute::A8, make_ptq1_weight,
                          {5120, 6144, 309U, Comparison::Full, false, kWidths});
    failures += run_shape("PTQ1_A8", ActivationCompute::A8, make_ptq1_weight,
                          {5120, 17408, 311U, Comparison::Full, false, kWidths});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        if (!ops::detail::ptq1_a8_shape_supported(128, 5120) ||
            ops::detail::ptq1_a8_shape_supported(64, 5120) ||
            ops::detail::ptq1_a8_shape_supported(129, 5120) ||
            ops::detail::ptq1_a8_shape_supported(128, 4096)) {
            throw std::runtime_error("PTQ1 tile shape admission mismatch");
        }
        const int failures = ptq1_a8_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " PTQ1_A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "PTQ1_A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
