#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {
// Upstream's routes keep the gate and up projections in FP32 through the activation, where most
// of this line's round them to BF16 first, so the check runs on the unified table.
//
// The int8 mainloop is pinned here too, for the same reason the table is: the wide mainloop
// follows the route table, and a binary that computes int8 workspace sizes and builds arenas
// around them has to say which kernel it is sizing for. Nesting-safe, so the per-profile pin in the
// harness and any case-local pin still win.
struct UnifiedInt8Route {
    UnifiedInt8Route()
        : previous(ninfer::ops::detail::force_nvfp4_int8_mainloop(
              ninfer::ops::detail::Nvfp4Int8Mainloop::Grouped)) {
        ninfer::ops::detail::force_linear_route_table(
            ninfer::ops::detail::LinearRouteTable::Unified);
    }

    ~UnifiedInt8Route() {
        ninfer::ops::detail::force_linear_route_table(std::nullopt);
        ninfer::ops::detail::force_nvfp4_int8_mainloop(previous);
    }

    UnifiedInt8Route(const UnifiedInt8Route&)            = delete;
    UnifiedInt8Route& operator=(const UnifiedInt8Route&) = delete;

    std::optional<ninfer::ops::detail::Nvfp4Int8Mainloop> previous;
};

int check_negative_gate() {
    using namespace ninfer;
    // A card that cannot select the unified table (unified_linear_routes) has no such route.
    if (!ops::detail::unified_linear_routes_here()) { return 0; }
    const UnifiedInt8Route unified;
    namespace qw = test::quantized_weight;
    if (test::cuda_unavailable()) return 0;
    constexpr int rows = 34816, k = 5120, half = rows / 2, max_tokens = 257;
    auto packed = qw::make_patterned_weight(
        QType::NVFP4, rows, k, 1837U, {.weight_scale_divisor = 1.0f, .input_scale_divisor = 6.0f});
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    // Preserve the scalar metadata, while constructing the two independently scaled groups.
    const float divisor = 1.0f;
    std::memcpy(packed.payload.data() + packed.weight_divisor_offset, &divisor, 4);
    const auto scale_offset = [&](int row, int group) {
        return packed.scale_plane_offset +
               static_cast<std::size_t>((row / 128) * (k / 64) + group / 4) * 512 +
               (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    };
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * (k / 2);
        if (row < half) {
            packed.payload[base]                 = 0x0f; // -6
            packed.payload[base + 8]             = 0x01; // +0.5 in the next G16 group
            packed.payload[scale_offset(row, 0)] = 0x46; // 3.5
            packed.payload[scale_offset(row, 1)] = 0x18; // 0.0625
            // k=65 = +0.5 * 0.9375 = 0.46875 (k=64 stays zero). Every gate row carries it so
            // the int8-vs-A16 divergence below is a whole-output one-ulp shift rather than a
            // row subset diluted into the L2 aggregates.
            packed.payload[base + 32]            = 0x10; // k64 +0, k65 +0.5
            packed.payload[scale_offset(row, 4)] = 0x37; // 0.9375
        } else {
            packed.payload[base]                 = 0x07; // +6
            packed.payload[scale_offset(row, 0)] = 0x7e; // 448
        }
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * max_tokens, 0);
    for (int token = 0; token < max_tokens; ++token) {
        activation[static_cast<std::size_t>(token) * k] =
            activation[static_cast<std::size_t>(token) * k + 16] = test::f32_to_bf16(1.0F);
        // Group 1 (disjoint from the probe group): k=64 pins the 64-column amax at 1.0 with a
        // zero weight, so it never reaches the output; k=65 = 0.5 then encodes under that amax
        // as s8 code 64 instead of round-tripping exactly. A lone nonzero channel would be its
        // own amax (the per-(token, 64) starvation the quantiser is built around) and the int8
        // reconstruction would be bit-exact again.
        activation[static_cast<std::size_t>(token) * k + 64] = test::f32_to_bf16(1.0F);
        activation[static_cast<std::size_t>(token) * k + 65] = test::f32_to_bf16(0.5F);
    }
    test::GuardedDeviceBuffer weights(packed.payload.size()), x(activation.size() * 2);
    test::GuardedDeviceBuffer y(static_cast<std::size_t>(half) * max_tokens * 2);
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    x.copy_from_host(activation.data(), activation.size() * 2);
    const Weight weight = packed.device_weight(weights.data());
    // The k=65 channel joins the probe terms in the oracle gate; the k=64 amax setter has a
    // zero weight and contributes nothing. The gate stays large-negative, so the complete
    // nonlinear formula keeps validating silu's argument order.
    const double gate = qw::logical_weight_fp64(packed, 0, 0) +
                        qw::logical_weight_fp64(packed, 0, 16) +
                        qw::logical_weight_fp64(packed, 0, 65) * 0.5;
    const double up    = qw::logical_weight_fp64(packed, half, 0);
    const double ideal = gate / (1.0 + std::exp(-gate)) * up;
    const auto bytes   = ops::linear_swiglu_workspace_capacity_bytes(
        weight.qtype, rows, k, ops::LinearPolicy::AllowA4, 1, max_tokens);
    WorkspaceArena workspace(bytes);
    int failures = 0;
    auto run_negative_gate = [&](const std::string& label, int tokens,
                                 ops::LinearPolicy policy, WorkspaceArena& arena,
                                 std::vector<std::uint16_t>* bits_out) {
        return test::run_case_allowing_arch_skip(label, [&] {
            Tensor input(x.data(), DType::BF16, {k, tokens});
            Tensor output(y.data(), DType::BF16, {half, tokens});
            ops::linear_swiglu(input, weight, output, policy, arena, nullptr);
            test::cuda_synchronize();
            std::vector<std::uint16_t> bits(static_cast<std::size_t>(half) * tokens);
            y.copy_to_host(bits.data(), bits.size() * 2);
            if (bits_out != nullptr) { *bits_out = bits; }
            std::vector<double> got(bits.size()), reference(bits.size(), ideal);
            for (std::size_t i = 0; i < bits.size(); ++i) got[i] = test::bf16_to_f32(bits[i]);
            // Every activation value here is exactly representable by A4 (the probe 1.0s and
            // k=65's 0.5, which encodes as e2m1 3.0 under its group's unit scale), so A4
            // reproduces the ideal gate bit-exactly. The negative gate keeps loss of
            // projection precision observable in the complete nonlinear formula.
            return test::verify_reduction(label, got, reference, {1.0 / 256.0, 0.0, 2.0 / 256.0});
        });
    };
    for (int tokens : {1, 64, 129, 256, 257}) {
        const std::string label = "NVFP4 SwiGLU negative gate T=" + std::to_string(tokens);
        failures += run_negative_gate(label, tokens, ops::LinearPolicy::AllowA4, workspace, nullptr);
    }
    // The AllowNvfp4Int8 route admits whole 128-token tiles only, so 256 is the one token count
    // here that drives the integer path through the negative-gate formula. The probe channels
    // quantize exactly (per-(token, 64) amax == 1.0, q == 127 with a 1-ulp dequant), but k=65
    // does not: starved by the weight-zero amax setter at k=64, 0.5 encodes as s8 code 64
    // (rn(63.5), half-to-even) and reconstructs as 64/127, +0.79% -- a loss the A16 route does
    // not have. 0.5 is deliberate: A4 stays exact on it (see above), so only the integer route
    // loses precision on this input, and the FP64 oracle keeps its tolerances untouched. The
    // resulting +0.18% output shift crosses one bf16 rounding midpoint (verified offline: A16
    // stores 231*2^-22, int8 stores 232*2^-22), which is what the A16-baseline guard below
    // measures.
    {
        const auto int8_bytes = ops::linear_swiglu_workspace_capacity_bytes(
            weight.qtype, rows, k, ops::LinearPolicy::AllowNvfp4Int8, 256, 256);
        WorkspaceArena int8_workspace(int8_bytes);
        // The unified table owns the integer route; without a forced table (or an installed
        // device profile) dispatch silently keeps the integer-A8 policies on the A16
        // composition. The forced table is what makes 256 actually take the integer kernel,
        // and the A16-baseline guard below fails loudly if that ever stops being true.
        if (!ninfer::ops::detail::nvfp4_linear_swiglu_int8_route_active(
                ninfer::ops::LinearPolicy::AllowNvfp4Int8, 256)) {
            std::cerr << "NVFP4 SwiGLU negative gate T=256 int8 grouped: int8 route not active\n";
            ++failures;
        }
        std::vector<std::uint16_t> int8_bits, a16_bits;
        failures += run_negative_gate("NVFP4 SwiGLU negative gate T=256 int8 grouped", 256,
                                      ops::LinearPolicy::AllowNvfp4Int8, int8_workspace,
                                      &int8_bits);
        WorkspaceArena a16_workspace(256);
        failures += run_negative_gate("NVFP4 SwiGLU negative gate T=256 int8 grouped A16 baseline",
                                      256, ops::LinearPolicy::A16Only, a16_workspace, &a16_bits);
        if (int8_bits.size() == a16_bits.size() && !int8_bits.empty()) {
            double diff_sq = 0.0;
            double base_sq = 0.0;
            for (std::size_t i = 0; i < int8_bits.size(); ++i) {
                const double diff =
                    static_cast<double>(test::bf16_to_f32(int8_bits[i])) -
                    static_cast<double>(test::bf16_to_f32(a16_bits[i]));
                diff_sq += diff * diff;
                base_sq += static_cast<double>(test::bf16_to_f32(a16_bits[i])) *
                           static_cast<double>(test::bf16_to_f32(a16_bits[i]));
            }
            const double relative = base_sq > 0.0 ? std::sqrt(diff_sq / base_sq) : 0.0;
            if (relative < 1.0e-4 || relative > 5.0e-2) {
                std::cerr << "NVFP4 SwiGLU negative gate T=256 int8 grouped: int8 output vs A16 "
                             "baseline rel difference "
                          << relative << " outside [1e-4, 5e-2] -- the integer route did not "
                                         "execute\n";
                ++failures;
            }
        }
    }
    return failures;
}
} // namespace

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // The A16 ladder stops at the width the fused wide MMA takes over from the SIMT
        // small-T band: 16|17 is the band boundary, and 128/256/512/1024/1025 carry the
        // wide route over whole tiles and the ragged width just past four of them. TODO.md
        // section 1 asks for exactly this re-add (the committed ladder stopped at 16, which
        // is how a route that cannot run on sm_86 at all once had a passing Op test).
        constexpr std::array<std::int32_t, 10> kA16Cases{1, 4, 8, 16, 17, 128, 256, 512, 1024, 1025};
        // 511 and 513 straddle the ragged floor: the first still reaches the composition, the
        // second is the narrowest ragged width the fused route admits and leaves one real token
        // in a third M tile. 767 leaves that tile all but full, and 1025 leaves one after four
        // whole tiles. 255/256 straddle the whole-tile floor, which does not move; 257 stays as
        // the width just past it that the composition keeps.
        constexpr std::array<std::int32_t, 21> kA4Cases{2,   4,   5,   16,  56,  64,   65,
                                                        96,  97,  112, 128, 129, 255,  256,
                                                        257, 511, 512, 513, 767, 1024, 1025};
        int failures = check_negative_gate();
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
        failures += run_profile("LinearSwiGLU NVFP4_A4",
                                {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4},
                                kA4Cases, std::array<std::int32_t, 4>{65, 97, 128, 129});
        // Tier-2 int8 grouped route: whole 128-token tiles (128/256/384/512/1024) take the
        // integer kernel, while 1/4/16/64 fall under the tile floor and 129/257 just past tile
        // boundaries -- all of which fall back to the A16 composition under the same policy.
        // The integer profile is a unified-table route: dispatch's default route conditions
        // (no forced table, no installed device profile) keep the integer policies on the
        // legacy plan's A16 composition, so without the forced table the admitted widths above
        // would pass without the integer kernel ever running -- the vacuous-green failure mode
        // the execution guard in run_profile now catches.
        constexpr std::array<std::int32_t, 11> kInt8Cases{1,  4,   16,  64,  128, 129, 256,
                                                          257, 384, 512, 1024};
        {
            const UnifiedInt8Route unified;
            failures += run_profile("LinearSwiGLU NVFP4_Int8Grouped",
                                    {QType::NVFP4, 34816, 5120, 17408, 1807U,
                                     ActivationCompute::Int8Grouped},
                                    kInt8Cases, std::array<std::int32_t, 2>{128, 1024});
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
