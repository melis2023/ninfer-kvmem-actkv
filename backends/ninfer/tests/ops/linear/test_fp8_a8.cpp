#include "core/weight.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/linear_test_common.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>

// The A8 workspace contracts describe the parts that run FP8 A8 (its TMA split-K partials are
// sized there); an sm_8x build never admits A8 and does not carry those schedules.
#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_FP8)
constexpr bool kA8Executable = false;
#else
constexpr bool kA8Executable = true;
#endif

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_fp8_a8() {
    constexpr std::array attn_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures =
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {14336, 5120, 827U, Comparison::SampledRows, true, attn_decode_invocations});
    constexpr std::array attn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 829U, Comparison::SampledRows, true, attn_invocations});
    constexpr std::array attn_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 833U, Comparison::SampledRows, true, attn_bulk_invocations});
    constexpr std::array gdn_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 837U, Comparison::SampledRows, true, gdn_decode_invocations});
    constexpr std::array gdn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 839U, Comparison::SampledRows, true, gdn_invocations});
    constexpr std::array gdn_bulk_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 841U, Comparison::SampledRows, true, gdn_bulk_invocations});
    constexpr std::array mlp_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 851U, Comparison::SampledRows, true, mlp_decode_invocations});
    constexpr std::array mlp_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 853U, Comparison::SampledRows, true, mlp_invocations});

    constexpr std::array mlp_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{511, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 863U, Comparison::SampledRows, true, mlp_bulk_invocations});

    constexpr std::array residual6144_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 6144, 855U, Comparison::SampledRows, true, residual6144_decode_invocations});
    constexpr std::array residual6144_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {5120, 6144, 857U, Comparison::SampledRows, true, residual6144_invocations});
    constexpr std::array residual17408_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 17408, 867U, Comparison::SampledRows, true, residual17408_decode_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape(
        "FP8_A8", ActivationCompute::A8, make_fp8_weight,
        {5120, 17408, 859U, Comparison::SampledRows, true, residual17408_invocations});

    struct Problem {
        std::int32_t rows;
        std::int32_t input_rows;
        bool a8_at_one;
        bool a8_at_two;
    };

    for (const Problem problem :
         {Problem{14336, 5120, false, false}, Problem{16384, 5120, false, false},
          Problem{34816, 5120, false, false}, Problem{5120, 6144, false, false},
          Problem{5120, 17408, false, false}}) {
        const auto capacity = [&](std::int32_t first, std::int32_t last,
                                  ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
            return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16, problem.rows,
                                                        problem.input_rows, policy, first, last);
        };
        // An interval reserves what its widest-reserving width uses. A device profile may give
        // neighbouring widths different route tables, and then a narrower width can need more
        // than the widest one (a split-K schedule's partials on one table, none on the other).
        const auto widest = [&](std::int32_t first, std::int32_t last) {
            std::size_t most = 0;
            for (std::int32_t width = first; width <= last; ++width) {
                most = std::max(most, capacity(width, width));
            }
            return most;
        };
        const auto table = [](std::int32_t width) {
            return ops::detail::linear_route_table(ops::detail::LinearRouteFamily::Fp8, width);
        };
        const std::size_t one         = capacity(1, 1);
        const std::size_t two         = capacity(2, 2);
        const std::size_t forty_eight = capacity(48, 48);
        const std::size_t exact_1024  = capacity(1024, 1024);
        const std::size_t exact_1048  = capacity(1048, 1048);
        // On one table a wider A8 launch reserves more; across tables only their own sizes count.
        const bool wider_reserves_more =
            table(1024) != table(1048) || exact_1048 > exact_1024;
        if (kA8Executable &&
            ((one != 0) != problem.a8_at_one || (two != 0) != problem.a8_at_two ||
             capacity(2, 4) != 0 || forty_eight <= two || capacity(1, 48) != widest(1, 48) ||
             exact_1024 <= forty_eight || !wider_reserves_more ||
             capacity(1000, 1048) != widest(1000, 1048) ||
             capacity(1, 2048, ops::LinearPolicy::A16Only) != 0)) {
            std::cerr << "FP8 A8 workspace interval contract mismatch for N=" << problem.rows
                      << " K=" << problem.input_rows << ": 1=" << one << " 2=" << two
                      << " 2..4=" << capacity(2, 4) << " 48=" << forty_eight
                      << " 1..48=" << capacity(1, 48) << " (widest " << widest(1, 48)
                      << ") 1024=" << exact_1024 << " 1048=" << exact_1048
                      << " 1000..1048=" << capacity(1000, 1048) << " (widest "
                      << widest(1000, 1048) << ")\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
