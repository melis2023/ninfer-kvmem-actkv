#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::test::linear_swiglu {
namespace {

// The criterion belongs to the activation-compute profile, not the weight storage format or a
// private materialized/fused implementation.
constexpr ReductionCriterion tolerance_for(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return {3.3e-3, 5.0e-3, kBf16GrossRelativeFloor};
    case ActivationCompute::A8:
        // Both independently A8-quantized projections feed the nonlinear product, so this profile
        // allows twice Linear's relative-L2 quantization allowance plus a bounded gross tail.
        return {8.0e-2, 1.0e-2, 1.2e-1};
    case ActivationCompute::A8Int:
    case ActivationCompute::A8IntDecode:
        // Same structure as A8: gate and up are each quantized independently before feeding the
        // nonlinear product, so the same allowance applies. Only the activation format differs --
        // s8 with one scale per weight group rather than FP8.
        return {8.0e-2, 1.0e-2, 1.2e-1};
    case ActivationCompute::Int8Grouped:
        // NVFP4 Tier-2 integer route: activations are s8 at one scale per (token, 64) block and
        // weights are e2m1 at one E4M3 scale per 16 columns, so both projections are quantized
        // independently before the nonlinear product -- the same structure as A8/A8Int with a
        // tighter weight grid, so the same allowance applies.
        return {8.0e-2, 1.0e-2, 1.2e-1};
    case ActivationCompute::A4:
        return {1.6e-1, 1.0e-2, 1.6e-1};
    }
    throw std::invalid_argument("linear_swiglu test: unknown activation compute profile");
}

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_swiglu test: invalid ") + label);
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_swiglu test: ") + label + " overflows");
    }
    return a * b;
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint16_t> make_activation(const Profile& profile, std::int32_t tokens) {
    std::vector<std::uint16_t> activation(
        checked_elements(profile.input_rows, tokens, "activation size"), 0);

    // Token zero is dense, so every logical K contributes to every production implementation.
    // Later tokens use exact zeros plus a rotating set of nonzeros. This keeps a full-output,
    // full-formula oracle practical at large registered T boundaries without adopting any
    // production staging or reduction behavior.
    const bool native_float_weight =
        profile.qtype == QType::NVFP4 || profile.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const float dense_scale =
        profile.qtype == QType::Q4_G64_FP16 ? 1.25e-4F : (native_float_weight ? 1.0e-3F : 1.0e-5F);
    for (std::int32_t column = 0; column < profile.input_rows; ++column) {
        const std::uint64_t mixed = mix64((static_cast<std::uint64_t>(profile.seed) << 32) |
                                          static_cast<std::uint32_t>(column));
        int numerator             = static_cast<int>((mixed >> 48) % 63U) - 31;
        if (numerator == 0) { numerator = (column & 1) == 0 ? 1 : -1; }
        const float value                            = static_cast<float>(numerator) * dense_scale;
        activation[static_cast<std::size_t>(column)] = test::f32_to_bf16(value);
    }

    constexpr std::int32_t kNonzerosPerSparseToken = 4;
    const float sparse_scale =
        profile.qtype == QType::Q4_G64_FP16 ? 1.5e-2F : (native_float_weight ? 2.0e-2F : 1.5e-3F);
    for (std::int32_t token = 1; token < tokens; ++token) {
        for (std::int32_t lane = 0; lane < kNonzerosPerSparseToken; ++lane) {
            const std::uint64_t mixed =
                mix64((static_cast<std::uint64_t>(profile.seed + 1U) << 32) |
                      static_cast<std::uint32_t>(token * kNonzerosPerSparseToken + lane));
            const std::int32_t column =
                static_cast<std::int32_t>(mixed % static_cast<std::uint64_t>(profile.input_rows));
            int numerator = static_cast<int>((mixed >> 48) % 31U) - 15;
            if (numerator == 0) { numerator = (lane & 1) == 0 ? 1 : -1; }
            const float value = static_cast<float>(numerator) * sparse_scale;
            activation[static_cast<std::size_t>(token) * profile.input_rows + column] =
                test::f32_to_bf16(value);
        }
    }
    return activation;
}

struct ActiveValue {
    std::int32_t token;
    double value;
};

std::vector<std::vector<ActiveValue>>
index_nonzero_activations(const std::vector<std::uint16_t>& activation, std::int32_t input_rows,
                          std::int32_t tokens) {
    std::vector<std::vector<ActiveValue>> by_column(static_cast<std::size_t>(input_rows));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < input_rows; ++column) {
            const std::uint16_t bits =
                activation[static_cast<std::size_t>(token) * input_rows + column];
            if ((bits & 0x7fffU) == 0U) { continue; }
            by_column[static_cast<std::size_t>(column)].push_back(
                {token, static_cast<double>(test::bf16_to_f32(bits))});
        }
    }
    return by_column;
}

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<double> linear_swiglu_oracle_fp64(const Profile& profile,
                                              const quantized_weight::PackedWeight& weight,
                                              const std::vector<std::uint16_t>& activation,
                                              std::int32_t tokens) {
    const auto active_by_column = index_nonzero_activations(activation, profile.input_rows, tokens);
    std::vector<double> output(checked_elements(profile.output_rows, tokens, "oracle output size"));

    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count =
        std::min(profile.output_rows, static_cast<std::int32_t>(hardware_threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * thread / thread_count);
        const std::int32_t row_end = static_cast<std::int32_t>(
            static_cast<std::int64_t>(profile.output_rows) * (thread + 1) / thread_count);
        workers.emplace_back([&, row_begin, row_end] {
            std::vector<double> gate(static_cast<std::size_t>(tokens));
            std::vector<double> up(static_cast<std::size_t>(tokens));
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                std::fill(gate.begin(), gate.end(), 0.0);
                std::fill(up.begin(), up.end(), 0.0);
                const std::int32_t up_row = profile.output_rows + row;
                for (std::int32_t column = 0; column < profile.input_rows; ++column) {
                    const double gate_weight =
                        quantized_weight::logical_weight_fp64(weight, row, column);
                    const double up_weight =
                        quantized_weight::logical_weight_fp64(weight, up_row, column);
                    for (const ActiveValue active :
                         active_by_column[static_cast<std::size_t>(column)]) {
                        gate[static_cast<std::size_t>(active.token)] += gate_weight * active.value;
                        up[static_cast<std::size_t>(active.token)] += up_weight * active.value;
                    }
                }
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const double fused = silu_fp64(gate[static_cast<std::size_t>(token)]) *
                                         up[static_cast<std::size_t>(token)];
                    output[static_cast<std::size_t>(token) * profile.output_rows + row] = fused;
                }
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return output;
}

std::vector<double> read_bf16_output(const test::GuardedDeviceBuffer& output,
                                     std::size_t elements) {
    std::vector<std::uint16_t> bits(elements);
    cuda_check(cudaMemcpy(bits.data(), output.data(), elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy LinearSwiGLU output");
    std::vector<double> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        values[index] = static_cast<double>(test::bf16_to_f32(bits[index]));
    }
    return values;
}

int compare_output(std::string_view label, const std::vector<double>& actual,
                   const double* reference, ActivationCompute activation_compute) {
    if (actual.empty() || reference == nullptr) {
        std::cerr << label << ": invalid numerical comparison\n";
        return 1;
    }

    return verify_reduction(label, actual, std::span<const double>(reference, actual.size()),
                            tolerance_for(activation_compute));
}

int verify_unchanged(std::string_view label, const test::GuardedDeviceBuffer& device,
                     const void* expected, std::size_t bytes) {
    if (device.bytes() != bytes) {
        std::cerr << label << ": read-only input size mismatch\n";
        return 1;
    }
    std::vector<std::uint8_t> actual(bytes);
    device.copy_to_host(actual.data(), actual.size());
    if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
    std::cerr << label << ": read-only input was modified\n";
    return 1;
}

void validate_profile(const Profile& profile) {
    const bool q4 = profile.qtype == QType::Q4_G64_FP16 && profile.gate_up_rows == 34816 &&
                    profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool t2 = is_ternary(profile.qtype) && profile.gate_up_rows == 34816 &&
                    profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool q8_companion = profile.qtype == QType::Q8_G32_FP16 &&
                              profile.gate_up_rows == 12288 && profile.input_rows == 2048 &&
                              profile.output_rows == 6144;
    const bool q8_dflash2 = profile.qtype == QType::Q8_G32_FP16 && profile.gate_up_rows == 34816 &&
                            profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool nvfp4 = profile.qtype == QType::NVFP4 && profile.gate_up_rows == 34816 &&
                       profile.input_rows == 5120 && profile.output_rows == 17408;
    const bool fp8 = profile.qtype == QType::FP8_E4M3FN_ROW_BF16 && profile.gate_up_rows == 34816 &&
                     profile.input_rows == 5120 && profile.output_rows == 17408;
    if ((!q4 && !t2 && !q8_companion && !q8_dflash2 && !nvfp4 && !fp8) ||
        profile.gate_up_rows != 2 * profile.output_rows) {
        throw std::invalid_argument("linear_swiglu test: profile is not registered");
    }
    if ((nvfp4 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A4 &&
         profile.activation_compute != ActivationCompute::Int8Grouped) ||
        (fp8 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A8) ||
        (!nvfp4 && !fp8 && profile.activation_compute != ActivationCompute::A16 &&
         profile.activation_compute != ActivationCompute::A8Int &&
         profile.activation_compute != ActivationCompute::A8IntDecode)) {
        throw std::invalid_argument("linear_swiglu test: invalid activation-compute profile");
    }
}

} // namespace

bool cuda_available() { return !test::cuda_unavailable(); }

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                std::span<const std::int32_t> graph_cases) {
    validate_profile(profile);
    // An Int8Grouped profile names the policy that admits the int8 family, and that policy is the
    // whole switch: the plan, the capacity query and the launch all read it, at every width. The
    // wide mainloop follows the route table, so a case that wants the other one pins it itself; the
    // vacuous-green guard below compares against the policy, so the two cannot drift apart.
    if (token_cases.empty()) { throw std::invalid_argument("linear_swiglu test: no token cases"); }
    for (std::size_t index = 0; index < token_cases.size(); ++index) {
        if (token_cases[index] <= 0 ||
            (index != 0 && token_cases[index] <= token_cases[index - 1])) {
            throw std::invalid_argument(
                "linear_swiglu test: token cases must be positive and increasing");
        }
    }
    const std::int32_t maximum_tokens = token_cases.back();
    for (int tokens : graph_cases)
        if (tokens <= 0 || tokens > maximum_tokens)
            throw std::invalid_argument("graph case outside oracle extent");

    quantized_weight::PatternedWeightOptions weight_options;
    if (profile.qtype == QType::NVFP4) {
        weight_options.weight_scale_divisor = 0.125F;
        weight_options.input_scale_divisor  = 3.5F;
    }
    quantized_weight::PackedWeight host_weight = quantized_weight::make_patterned_weight(
        profile.qtype, profile.gate_up_rows, profile.input_rows, profile.seed, weight_options);
    const std::vector<std::uint16_t> host_activation = make_activation(profile, maximum_tokens);
    const std::vector<double> reference =
        linear_swiglu_oracle_fp64(profile, host_weight, host_activation, maximum_tokens);
    std::vector<std::uint16_t> negative_activation;
    std::vector<double> negative_reference;
    std::optional<DeviceContext> graph_context;
    if (!graph_cases.empty()) {
        negative_activation = host_activation;
        for (auto& bits : negative_activation) bits ^= 0x8000;
        negative_reference =
            linear_swiglu_oracle_fp64(profile, host_weight, negative_activation, maximum_tokens);
        graph_context.emplace();
    }

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    test::GuardedDeviceBuffer device_activation(host_activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));

    const ops::LinearPolicy policy =
        profile.activation_compute == ActivationCompute::A4
            ? ops::LinearPolicy::AllowA4
            : (profile.activation_compute == ActivationCompute::A8
                   ? ops::LinearPolicy::AllowA8
                   : (profile.activation_compute == ActivationCompute::A8Int
                          ? ops::LinearPolicy::AllowA8Int
                          : (profile.activation_compute == ActivationCompute::A8IntDecode
                                 ? ops::LinearPolicy::AllowA8IntDecode
                                 : (profile.activation_compute ==
                                            ActivationCompute::Int8Grouped
                                        ? ops::LinearPolicy::AllowNvfp4Int8
                                        : ops::LinearPolicy::A16Only))));
    const std::size_t workspace_bytes = ops::linear_swiglu_workspace_capacity_bytes(
        profile.qtype, profile.gate_up_rows, profile.input_rows, policy, 1, maximum_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    int failures        = 0;
    bool any_graph_replay_ran = false;
    const auto run_case = [&](int tokens, bool replay) {
        const auto elements = checked_elements(profile.output_rows, tokens, "output size");
        test::GuardedDeviceBuffer output(elements * sizeof(std::uint16_t));
        Tensor x(device_activation.data(), DType::BF16, {profile.input_rows, tokens});
        Tensor destination(output.data(), DType::BF16, {profile.output_rows, tokens});
        workspace.reset();
        workspace.reset_peak();
        const cudaStream_t stream = replay ? graph_context->stream : nullptr;
        const auto launch         = [&] {
            ops::linear_swiglu(x, weight, destination, policy, workspace, stream);
        };
        const std::string label_case =
            std::string(label) + " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
        std::vector<double> last_actual;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        try {
            test::cuda_check(cudaDeviceSynchronize(), "finish fixture initialization");
            if (replay) {
                definition.capture(stream, launch);
                graph.instantiate(definition);
            }
            for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
                const auto& input_bits = phase ? negative_activation : host_activation;
                const auto& expected   = phase ? negative_reference : reference;
                if (replay)
                    device_activation.copy_from_host(input_bits.data(),
                                                     input_bits.size() * sizeof(std::uint16_t));
                output.fill(0xff);
                test::cuda_check(cudaDeviceSynchronize(), "finish replay input update");
                if (replay)
                    graph.launch(stream);
                else
                    launch();
                test::cuda_check(cudaStreamSynchronize(stream), "synchronize LinearSwiGLU");
                const auto exact = ops::linear_swiglu_workspace_capacity_bytes(
                    profile.qtype, profile.gate_up_rows, profile.input_rows, policy, tokens,
                    tokens);
                if (workspace.used() != 0 || workspace.peak_used() != exact) {
                    std::cerr << label_case << ": exact workspace query/execution mismatch\n";
                    ++failures;
                }
                failures += output.verify_guards(label_case);
                const auto actual = read_bf16_output(output, elements);
                failures +=
                    compare_output(label_case, actual, expected.data(), profile.activation_compute);
                last_actual = actual;
                if (replay)
                    failures += verify_unchanged(label_case + " input", device_activation,
                                                 input_bits.data(),
                                                 input_bits.size() * sizeof(std::uint16_t));
            }
            if (replay) any_graph_replay_ran = true;
        } catch (const std::exception& error) {
            if (unsupported_arch_refusal(error)) {
                std::cout << "SKIP " << label_case << ": " << error.what() << '\n';
                return;
            }
            std::cerr << label_case << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            return;
        }
        const std::size_t exact_workspace = ops::linear_swiglu_workspace_capacity_bytes(
            profile.qtype, profile.gate_up_rows, profile.input_rows, policy, tokens, tokens);
        if (workspace.used() != 0 || workspace.peak_used() != exact_workspace) {
            std::cerr << label_case << ": exact workspace query/execution high-water mismatch\n";
            ++failures;
        }
        // Vacuous-green guard for the integer profile. The int8 policy falls back to the A16
        // composition whenever the route table says so or the named route does not admit the width,
        // and a fallback that still satisfies the quantisation tolerance is exactly how an
        // Int8Grouped profile passes without the integer kernel ever running. First, dispatch
        // has to promise exactly the admission rule; second, the integer output has to differ
        // from the A16 composition of the same weights at quantisation scale.
        if (profile.activation_compute == ActivationCompute::Int8Grouped && !last_actual.empty()) {
            const bool admitted = ops::detail::unified::nvfp4_int8_route_admits(tokens);
            if (ops::detail::nvfp4_linear_swiglu_int8_route_active(policy, tokens) != admitted) {
                std::cerr << label_case << ": int8 route/dispatch admission mismatch\n";
                ++failures;
            }
            if (admitted) {
                try {
                    test::GuardedDeviceBuffer baseline(elements * sizeof(std::uint16_t));
                    Tensor baseline_destination(baseline.data(), DType::BF16,
                                                {profile.output_rows, tokens});
                    workspace.reset();
                    workspace.reset_peak();
                    ops::linear_swiglu(x, weight, baseline_destination, ops::LinearPolicy::A16Only,
                                       workspace, nullptr);
                    test::cuda_check(cudaDeviceSynchronize(),
                                     "synchronize the A16 baseline of the int8 guard");
                    const auto baseline_actual = read_bf16_output(baseline, elements);
                    double diff_sq = 0.0;
                    double base_sq = 0.0;
                    for (std::size_t index = 0; index < elements; ++index) {
                        const double diff = last_actual[index] - baseline_actual[index];
                        diff_sq += diff * diff;
                        base_sq += baseline_actual[index] * baseline_actual[index];
                    }
                    const double relative = base_sq > 0.0 ? std::sqrt(diff_sq / base_sq) : 0.0;
                    if (relative < 1.0e-4 || relative > 5.0e-2) {
                        std::cerr << label_case << ": int8 output vs A16 baseline rel difference "
                                  << relative << " outside [1e-4, 5e-2] -- the integer route did "
                                     "not execute\n";
                        ++failures;
                    }
                } catch (const std::exception& error) {
                    std::cerr << label_case << ": int8 execution guard failed: " << error.what()
                              << '\n';
                    ++failures;
                }
            }
        }
    };
    for (int tokens : token_cases) run_case(tokens, false);
    if (!graph_cases.empty()) {
        // Check eager inputs before graph fixtures overwrite them with the next represented input.
        failures += verify_unchanged(std::string(label) + " eager input", device_activation,
                                     host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));
    }
    for (int tokens : graph_cases) run_case(tokens, true);
    const auto& final_input = any_graph_replay_ran ? negative_activation : host_activation;
    failures += device_activation.verify_guards(std::string(label) + " activation");
    failures += device_weight.verify_guards(std::string(label) + " weight");
    failures += verify_unchanged(std::string(label) + " activation", device_activation,
                                 final_input.data(), final_input.size() * sizeof(std::uint16_t));
    failures += verify_unchanged(std::string(label) + " weight", device_weight,
                                 host_weight.payload.data(), host_weight.payload.size());
    return failures;
}

} // namespace ninfer::test::linear_swiglu
