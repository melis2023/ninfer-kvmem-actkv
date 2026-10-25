#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // The ternary gate_up of Ternary Bonsai 2 at the decode and verify widths (one column, an
        // MTP round of four, a DFlash2 round of eight and the widths between) and past them, under
        // both integer policies; every case is held to the A8 activation allowance whichever route
        // ran, in CUDA graphs too at the verify widths.
        constexpr std::array<std::int32_t, 11> kTokenCases{1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64};
        constexpr std::array<std::int32_t, 2> kGraphCases{4, 8};
        int failures = 0;
        failures += run_profile(
            "LinearSwiGLU T2_A8INT",
            {QType::T2_G128_FP16, 34816, 5120, 17408, 1411U, ActivationCompute::A8Int},
            kTokenCases, kGraphCases);
        failures += run_profile(
            "LinearSwiGLU T2_A8INT_DECODE",
            {QType::T2_G128_FP16, 34816, 5120, 17408, 1413U, ActivationCompute::A8IntDecode},
            kTokenCases, kGraphCases);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU T2_A8INT correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU T2_A8INT test failed: " << error.what() << '\n';
        return 1;
    }
}
