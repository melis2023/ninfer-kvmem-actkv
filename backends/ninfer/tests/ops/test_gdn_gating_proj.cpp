#include "core/weight.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/weight_input.h"

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Geometry {
    const char* label;
    std::int32_t hidden;
    std::int32_t heads;
    bool parent_weight;
};

constexpr Geometry kQwen27{"qwen3_6_27b", 5120, 48, false};
constexpr Geometry kQwen38Parent{"qwen3_8_27b_parent", 5120, 48, true};
constexpr Geometry kQwen35{"qwen3_6_35b_a3b", 2048, 32, true};

#if defined(NINFER_SFU_SOFTPLUS) && NINFER_SFU_SOFTPLUS
// The SFU softplus carries lg2.approx's error, about 2e-6 of the result.
constexpr ReductionCriterion kGdnProjectionFp32{/*relative_l2=*/4.0e-6,
                                                /*gross_absolute=*/5.0e-7,
                                                /*gross_relative_to_max_reference=*/5.0e-6};
#else
// Sized for the unsplit MMA route, the least accurate registered projection route: it chains all
// 320 K tiles into one fp32 accumulator where a split-K route chains K/SplitK and reduces across
// the grid, so its rounding error measures 5-8x theirs. It covers T >= 4097 everywhere, and the
// planner also falls back to it when a split cooperative grid exceeds device residency: on an
// 82-SM laptop RTX 5090 that is T = 1024, whose split8 grid of 192 CTAs does not fit 164.
constexpr ReductionCriterion kGdnProjectionFp32{/*relative_l2=*/2.0e-6,
                                                /*gross_absolute=*/5.0e-7,
                                                /*gross_relative_to_max_reference=*/5.0e-6};
#endif
constexpr ReductionCriterion kGdnNormOutputBf16{/*relative_l2=*/1.75e-3,
                                                /*gross_absolute=*/1.0e-4,
                                                /*gross_relative_to_max_reference=*/kBf16GrossRelativeFloor};
// FP32 public controls permit 16-bit private operands/materialization. The gross cap includes
// propagated BF16 staging error; relative L2 remains the accuracy gate for the full formula.
constexpr ReductionCriterion kGdnNormControlFp32{/*relative_l2=*/8.0e-4,
                                                 /*gross_absolute=*/1.5e-4,
                                                 /*gross_relative_to_max_reference=*/2.0e-3};

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double e = std::exp(value);
    return e / (1.0 + e);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> read_fp32(const void* device, std::size_t elements) {
    const std::vector<float> values = from_device<float>(device, elements);
    return {values.begin(), values.end()};
}

int verify_normwise(const std::string& label, const std::vector<double>& actual,
                    const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, expected, criterion);
}

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

std::vector<std::int32_t> oracle_tokens(std::int32_t tokens) {
    std::vector<std::int32_t> selected;
    if (tokens <= 128) {
        selected.reserve(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) { selected.push_back(token); }
    } else {
        selected = {0, tokens / 2, tokens - 1};
    }
    return selected;
}

void projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                       const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                       const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                       const std::vector<std::int32_t>& selected_tokens, std::vector<double>& g,
                       std::vector<double>& beta) {
    const std::size_t output_elements =
        static_cast<std::size_t>(geometry.heads) * selected_tokens.size();
    g.resize(output_elements);
    beta.resize(output_elements);

    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t x_base =
            static_cast<std::size_t>(selected_tokens[sample]) * geometry.hidden;
        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = sample * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

void norm_projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                            const std::vector<float>& norm_weight,
                            const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                            const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                            std::int32_t tokens, double eps, std::vector<double>& h,
                            std::vector<double>& g, std::vector<double>& beta) {
    h.resize(static_cast<std::size_t>(geometry.hidden) * tokens);
    g.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    beta.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    std::vector<double> normalized(static_cast<std::size_t>(geometry.hidden));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t x_base = static_cast<std::size_t>(token) * geometry.hidden;
        double sum_squares       = 0.0;
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
            sum_squares += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(sum_squares / static_cast<double>(geometry.hidden) + eps);
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]) *
                                 inverse_rms * (1.0 + static_cast<double>(norm_weight[k]));
            normalized[static_cast<std::size_t>(k)] = value;
            h[x_base + static_cast<std::size_t>(k)] = value;
        }

        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = normalized[static_cast<std::size_t>(k)];
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = static_cast<std::size_t>(token) * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

std::vector<double> select_tokens(const std::vector<double>& full,
                                  const std::vector<std::int32_t>& selected_tokens,
                                  std::int32_t rows) {
    std::vector<double> selected(static_cast<std::size_t>(rows) * selected_tokens.size());
    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t source = static_cast<std::size_t>(selected_tokens[sample]) * rows;
        std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(source), rows,
                    selected.begin() + static_cast<std::ptrdiff_t>(sample * rows));
    }
    return selected;
}

int require_all_finite(const std::string& label, const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::cerr << label << ": output element " << i
                      << " was not written to a finite value\n";
            return 1;
        }
    }
    return 0;
}

int verify_inputs_unchanged(const std::string& label, const DeviceBuffer& device_x,
                            const std::vector<std::uint16_t>& x_bits,
                            const DeviceBuffer& device_weight,
                            const std::vector<std::uint16_t>& weight_bits,
                            const DeviceBuffer& device_a_log, const std::vector<float>& a_log,
                            const DeviceBuffer& device_dt_bias, const std::vector<float>& dt_bias) {
    int failures = 0;
    failures += verify_exact((label + " x immutable").c_str(),
                             from_device<std::uint16_t>(device_x, x_bits.size()), x_bits);
    failures +=
        verify_exact((label + " weight immutable").c_str(),
                     from_device<std::uint16_t>(device_weight, weight_bits.size()), weight_bits);
    failures += verify_exact((label + " A_log immutable").c_str(),
                             from_device<float>(device_a_log, a_log.size()), a_log);
    failures += verify_exact((label + " dt_bias immutable").c_str(),
                             from_device<float>(device_dt_bias, dt_bias.size()), dt_bias);
    return failures;
}

int run_projection_case(const Geometry& geometry, std::int32_t tokens, std::uint32_t seed,
                        DeviceExecutionView execution) {
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(a_weight, seed + 1u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 3u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 4u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::int32_t> selected = oracle_tokens(tokens);
    std::vector<double> reference_g, reference_beta;
    projection_oracle(geometry, x, a_weight, b_weight, a_log, dt_bias, selected, reference_g,
                      reference_beta);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(x);
    std::vector<std::uint16_t> weight_bits         = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_weight_bits = bf16_bits(b_weight);
    if (geometry.parent_weight) {
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    }
    DeviceBuffer device_x      = to_device(x_bits);
    DeviceBuffer device_weight = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) { device_b_weight = to_device(b_weight_bits); }
    DeviceBuffer device_a_log         = to_device(a_log);
    DeviceBuffer device_dt_bias       = to_device(dt_bias);
    const std::size_t output_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer device_g(output_elements * sizeof(float));
    GuardedDeviceBuffer device_beta(output_elements * sizeof(float));
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    const std::size_t workspace_bytes = ops::gdn_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (geometry.parent_weight) {
        // A/B use a contiguous reshape of the same stored words, with a different parent K.
        const std::array shape{static_cast<std::uint64_t>(4 * geometry.heads),
                               static_cast<std::uint64_t>(geometry.hidden / 2)};
        const WeightParent parent{weight_geometry(QType::BF16, QuantLayout::Contiguous, shape),
                                  static_cast<const std::byte*>(device_weight.p)};
        const auto count = static_cast<std::uint64_t>(geometry.heads) * geometry.hidden;
        const std::vector<std::uint64_t> logical_shape{static_cast<std::uint64_t>(geometry.heads),
                                                       static_cast<std::uint64_t>(geometry.hidden)};
        const WeightView a{logical_shape, {{&parent, 0, count}}};
        const WeightView b{logical_shape, {{&parent, count, 2 * count}}};
        const auto prepared =
            std::get<ops::SingleProjectionWeight>(ops::prepare_gdn_gating_proj_weights({a}, {b}));
        ops::gdn_gating_proj(tensor_x, prepared.weight, tensor_a_log, tensor_dt_bias, workspace,
                             tensor_g, tensor_beta, execution);
    } else {
        Weight weight_a = bf16_weight(device_weight.p, geometry.heads, geometry.hidden);
        Weight weight_b = bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
        ops::gdn_gating_proj(tensor_x, weight_a, weight_b, tensor_a_log, tensor_dt_bias, workspace,
                             tensor_g, tensor_beta, execution);
    }
    cuda_synchronize();

    const std::vector<double> full_g    = read_fp32(device_g.data(), output_elements);
    const std::vector<double> full_beta = read_fp32(device_beta.data(), output_elements);
    const std::string label =
        std::string("gdn_gating_proj ") + geometry.label + " T=" + std::to_string(tokens);
    int failures = 0;
    failures += require_all_finite(label + " g", full_g);
    failures += require_all_finite(label + " beta", full_beta);
    failures += verify_normwise(label + " g", select_tokens(full_g, selected, geometry.heads),
                                reference_g, kGdnProjectionFp32);
    failures += verify_normwise(label + " beta", select_tokens(full_beta, selected, geometry.heads),
                                reference_beta, kGdnProjectionFp32);
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                        device_a_log, a_log, device_dt_bias, dt_bias);
    if (!geometry.parent_weight) {
        failures += verify_exact((label + " b_weight immutable").c_str(),
                                 from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()),
                                 b_weight_bits);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_norm_projection_case(const Geometry& geometry, std::int32_t tokens, std::uint32_t seed,
                             DeviceExecutionView execution, bool replay = false, int mode = 0) {
    constexpr float kEps               = 1.0e-6f;
    const std::size_t h_elements       = std::size_t(geometry.hidden) * tokens;
    const std::size_t control_elements = std::size_t(geometry.heads) * tokens;
    std::vector<float> x(h_elements), norm_weight(geometry.hidden);
    std::vector<float> a_weight(std::size_t(geometry.heads) * geometry.hidden),
        b_weight(a_weight.size());
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0f, 1.0f);
    fill_uniform(norm_weight, seed + 1, -0.2f, 0.2f);
    fill_uniform(a_weight, seed + 2, -0.015f, 0.015f);
    fill_uniform(b_weight, seed + 3, -0.015f, 0.015f);
    fill_uniform(a_log, seed + 4, -2.0f, 1.0f);
    fill_uniform(dt_bias, seed + 5, -1.0f, 1.0f);
    norm_weight[0] = -1.0f;
    norm_weight[1] = -1.5f;
    if (mode == 1) std::fill(x.begin(), x.end(), 0.0f);
    if (mode == 2)
        for (auto& v : x) v *= 1.0e-5f;
    if (mode == 3)
        for (auto& v : x) v *= 512.0f;
    if (mode == 4) std::fill(norm_weight.begin(), norm_weight.end(), -1.0f);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);
    auto x_bits = bf16_bits(x), norm_weight_bits = bf16_bits(norm_weight),
         weight_bits         = bf16_bits(a_weight);
    const auto b_weight_bits = bf16_bits(b_weight);
    if (geometry.parent_weight)
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    DeviceBuffer device_x = to_device(x_bits), device_norm_weight = to_device(norm_weight_bits),
                 device_weight = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) device_b_weight = to_device(b_weight_bits);
    DeviceBuffer device_a_log = to_device(a_log), device_dt_bias = to_device(dt_bias);
    GuardedDeviceBuffer device_h(h_elements * 2), device_g(control_elements * 4),
        device_beta(control_elements * 4);
    Tensor tx(device_x.p, DType::BF16, {geometry.hidden, tokens}),
        tn(device_norm_weight.p, DType::BF16, {geometry.hidden});
    Tensor th(device_h.data(), DType::BF16, {geometry.hidden, tokens}),
        tg(device_g.data(), DType::FP32, {geometry.heads, tokens}),
        tb(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    Tensor ta(device_a_log.p, DType::FP32, {geometry.heads}),
        td(device_dt_bias.p, DType::FP32, {geometry.heads});
    const auto capacity = ops::gdn_norm_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 256)});
    const auto wa = bf16_weight(device_weight.p, (geometry.parent_weight ? 2 : 1) * geometry.heads,
                                geometry.hidden);
    const auto wb = geometry.parent_weight
                        ? Weight{}
                        : bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
    const auto launch = [&] {
        if (geometry.parent_weight)
            ops::gdn_norm_gating_proj(tx, tn, kEps, wa, ta, td, workspace, th, tg, tb, execution);
        else
            ops::gdn_norm_gating_proj(tx, tn, kEps, wa, wb, ta, td, workspace, th, tg, tb,
                                      execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) {
        launch();
        cuda_synchronize(execution.stream);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 3 : 1); ++phase) {
        if (phase) {
            for (auto& v : x) v *= -0.5f;
            for (auto& v : norm_weight) v = -v;
            for (auto& v : a_log) v += 0.125f;
            for (auto& v : dt_bias) v = -v;
            round_to_bf16(x);
            round_to_bf16(norm_weight);
            x_bits           = bf16_bits(x);
            norm_weight_bits = bf16_bits(norm_weight);
            device_x.copy_from_host(x_bits.data(), x_bits.size() * 2);
            device_norm_weight.copy_from_host(norm_weight_bits.data(), norm_weight_bits.size() * 2);
            device_a_log.copy_from_host(a_log.data(), a_log.size() * 4);
            device_dt_bias.copy_from_host(dt_bias.data(), dt_bias.size() * 4);
        }
        device_h.fill(0xff);
        device_g.fill(0xff);
        device_beta.fill(0xff);
        scratch.fill(0xff);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(execution.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(execution.stream);
        else
            launch();
        cuda_synchronize(execution.stream);
        std::vector<double> rh, rg, rb;
        norm_projection_oracle(geometry, x, norm_weight, a_weight, b_weight, a_log, dt_bias, tokens,
                               kEps, rh, rg, rb);
        const std::string label = std::string("gdn_norm_gating_proj ") + geometry.label +
                                  " T=" + std::to_string(tokens) + " mode=" + std::to_string(mode) +
                                  " phase=" + std::to_string(phase);
        failures += verify_normwise(label + " h", from_device_bf16(device_h.data(), h_elements), rh,
                                    kGdnNormOutputBf16);
        failures += verify_normwise(label + " g", read_fp32(device_g.data(), control_elements), rg,
                                    kGdnNormControlFp32);
        failures +=
            verify_normwise(label + " beta", read_fp32(device_beta.data(), control_elements), rb,
                            kGdnNormControlFp32);
        failures += device_h.verify_guards((label + " h").c_str());
        failures += device_g.verify_guards((label + " g").c_str());
        failures += device_beta.verify_guards((label + " beta").c_str());
        failures += scratch.verify_guards((label + " scratch").c_str());
        failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                            device_a_log, a_log, device_dt_bias, dt_bias);
        failures +=
            verify_exact((label + " norm weight immutable").c_str(),
                         from_device<std::uint16_t>(device_norm_weight, norm_weight_bits.size()),
                         norm_weight_bits);
        if (!geometry.parent_weight)
            failures += verify_exact(
                (label + " b weight immutable").c_str(),
                from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()), b_weight_bits);
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/peak mismatch\n";
            ++failures;
        }
    }
    return failures;
}

int verify_workspace_capacity_contract(const Geometry& geometry,
                                       const std::vector<std::int32_t>& route_endpoints) {
    const std::int32_t last = *std::max_element(route_endpoints.begin(), route_endpoints.end());
    const std::size_t interval =
        ops::gdn_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 1, last);
    std::size_t witness = 0;
    for (const std::int32_t tokens : route_endpoints) {
        witness = std::max(witness, ops::gdn_gating_proj_workspace_capacity_bytes(
                                        geometry.heads, geometry.hidden, tokens, tokens));
    }
    int failures = 0;
    if (interval != witness) {
        std::cerr << geometry.label << ": GDN control interval missed a route endpoint\n";
        ++failures;
    }
    for (const auto bounds :
         std::vector<std::pair<int, int>>{{1, 42}, {40, 43}, {1, 128}, {120, 129}, {1, 256}}) {
        std::size_t witness = 0;
        for (int t = bounds.first; t <= bounds.second; ++t)
            witness = std::max(witness, ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                            geometry.heads, geometry.hidden, t, t));
        const auto capacity = ops::gdn_norm_gating_proj_workspace_capacity_bytes(
            geometry.heads, geometry.hidden, bounds.first, bounds.second);
        if (capacity != witness) {
            std::cerr << geometry.label << ": norm/control interval missed an extent\n";
            ++failures;
        }
    }
    return failures;
}

// Both sides of every width where this device's plan changes schedule, with the interval's ends.
std::vector<std::int32_t> plan_endpoints(const Geometry& geometry, std::int32_t sm_count,
                                         std::int32_t last) {
    const auto schedule = [&](std::int32_t cols) {
        return ninfer::ops::detail::bf16_gdn_gating_resolve_plan(
                   {geometry.heads, geometry.hidden, cols}, sm_count)
            .schedule;
    };
    std::vector<std::int32_t> endpoints{1, last};
    for (std::int32_t cols = 2; cols <= last; ++cols) {
        if (schedule(cols) != schedule(cols - 1)) {
            endpoints.push_back(cols - 1);
            endpoints.push_back(cols);
        }
    }
    return endpoints;
}

// Which schedule the projection gets on devices with different SM counts. The route table was
// tuned on 82 SMs and its cooperative schedules need their whole grid resident, so a smaller device
// (a 48-SM RTX A4000), or one whose SMs hold fewer CTAs, must fall through to a less-split schedule
// while a route that fits is kept exactly.
namespace sm_count_routes {

using ninfer::ops::detail::Bf16GdnGatingPlan;
using Id = ninfer::ops::detail::Bf16GdnGatingScheduleId;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

struct Geometry {
    const char* label;
    std::int32_t heads;
    std::int32_t rows;
    std::int32_t tile_cols;
    std::int32_t row_tiles;
};

constexpr Geometry k27{"27B", 48, 5120, 128, 3};
constexpr Geometry k35{"35B-A3B", 32, 2048, 64, 2};

// The tuned table on 82 SMs, restated from the tuning notes rather than read back from the code.
Id tuned(const Geometry& g, std::int32_t cols) {
    if (g.heads == 48) {
        if (cols == 1) { return Id::GemvPairedRows; }
        if (cols <= 8) { return Id::SmallTSplit10; }
        if (cols <= 768) { return Id::MmaCooperativeSplit8; }
        if (cols <= 1664) { return Id::MmaCooperativeSplit4; }
        if (cols <= 3456) { return Id::MmaCooperativeSplit2; }
        return Id::MmaUnsplit;
    }
    if (cols <= 127) { return Id::MmaCooperativeSplit16; }
    if (cols <= 960) { return Id::MmaCooperativeSplit8; }
    if (cols <= 1920) { return Id::MmaCooperativeSplit4; }
    if (cols <= 3904) { return Id::MmaCooperativeSplit2; }
    return Id::MmaUnsplit;
}

std::int32_t split_of(Id schedule) {
    switch (schedule) {
    case Id::MmaCooperativeSplit32: return 32;
    case Id::MmaCooperativeSplit16: return 16;
    case Id::MmaCooperativeSplit8: return 8;
    case Id::MmaCooperativeSplit4: return 4;
    case Id::MmaCooperativeSplit2: return 2;
    default: return 0; // not cooperative
    }
}

// Resident CTAs per SM of each cooperative schedule on this device: the compiled kernels'
// occupancy, which the plan reads too (two 27B CTAs and two to four 35B ones on sm_86, fewer on a
// Turing SM's 64 KiB of shared memory).
std::int32_t ctas_per_sm(const Geometry& g, Id schedule) {
    return ninfer::ops::detail::bf16_gdn_gating_proj_mma_resident_ctas_per_sm(g.heads == 32,
                                                                              split_of(schedule));
}

bool grid_fits(const Geometry& g, Id schedule, std::int32_t cols, std::int32_t sms) {
    const std::int32_t split = split_of(schedule);
    if (split == 0) { return true; }
    const std::int64_t grid =
        static_cast<std::int64_t>((cols + g.tile_cols - 1) / g.tile_cols) * g.row_tiles * split;
    return grid <= static_cast<std::int64_t>(ctas_per_sm(g, schedule)) * sms;
}

int verify_plan_across_sm_counts() {
    constexpr std::int32_t kWidths = 6000;
    // 82 SMs is the device the table was tuned on: where this device's occupancy holds a tuned
    // route there (every width on sm_86), the plan must keep it.
    constexpr std::int32_t kSmCounts[] = {84, 82, 72, 68, 48, 40, 30, 28};
    // The tuned occupancy, restated from the tuning notes: on an sm_86 part the compiled kernels
    // must still hold it, or the tuned table no longer fits the device it was tuned on.
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
        major == 8 && minor == 6) {
        for (const Id schedule :
             {Id::MmaCooperativeSplit8, Id::MmaCooperativeSplit4, Id::MmaCooperativeSplit2}) {
            check(ctas_per_sm(k27, schedule) == 2, "27B: sm_86 occupancy of a cooperative split");
        }
        check(ctas_per_sm(k35, Id::MmaCooperativeSplit32) == 2, "35B: sm_86 split-32 occupancy");
        check(ctas_per_sm(k35, Id::MmaCooperativeSplit16) == 4, "35B: sm_86 split-16 occupancy");
        for (const Id schedule :
             {Id::MmaCooperativeSplit8, Id::MmaCooperativeSplit4, Id::MmaCooperativeSplit2}) {
            check(ctas_per_sm(k35, schedule) == 3, "35B: sm_86 occupancy of a cooperative split");
        }
    }
    for (const Geometry& g : {k27, k35}) {
        // A device with fewer SMs never gets a grid it cannot hold, and only leaves the tuned route
        // when that route does not fit, moving to a schedule that splits K no more.
        for (const std::int32_t sms : kSmCounts) {
            for (std::int32_t cols = 1; cols <= kWidths; ++cols) {
                const Bf16GdnGatingPlan plan = ninfer::ops::detail::bf16_gdn_gating_resolve_plan(
                    {g.heads, g.rows, cols}, sms);
                const std::string where = std::string(g.label) + ": " + std::to_string(sms) +
                                          " SMs, " + std::to_string(cols) + " columns";
                check(grid_fits(g, plan.schedule, cols, sms), where + " got a grid that is not resident");
                const Id preferred = tuned(g, cols);
                if (grid_fits(g, preferred, cols, sms)) {
                    check(plan.schedule == preferred, where + " left a tuned route that fits");
                } else {
                    check(split_of(plan.schedule) < split_of(preferred),
                          where + " did not move to a less-split schedule");
                }
            }
        }
        // Naming a schedule that does not fit is an error, not a silent substitution.
        bool threw = false;
        try {
            (void)ninfer::ops::detail::bf16_gdn_gating_resolve_candidate(
                Id::MmaCooperativeSplit8, {g.heads, g.rows, 1024}, 30);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, std::string(g.label) + ": an over-large candidate was accepted");
    }
    // The fused 35B norm + gating route launches split 32 only where its grid is resident (an RTX
    // 3060's 28 SMs hold 56 of its 64 CTAs); otherwise the norm runs first and the gating keeps
    // its resident plan.
    using NormId = ninfer::ops::detail::Bf16GdnNormGatingScheduleId;
    for (const std::int32_t sms : kSmCounts) {
        for (std::int32_t cols = 1; cols <= 64; ++cols) {
            const auto plan = ninfer::ops::detail::bf16_gdn_norm_gating_resolve_plan(
                {k35.heads, k35.rows, cols}, sms);
            const std::string where = "35B norm gating: " + std::to_string(sms) + " SMs, " +
                                      std::to_string(cols) + " columns";
            const bool fused = cols <= 16 && grid_fits(k35, Id::MmaCooperativeSplit32, cols, sms);
            check(plan.schedule == (fused ? NormId::MmaCooperativeSplit32 : NormId::Composed),
                  where + " chose the wrong route");
            check(grid_fits(k35, plan.control.schedule, cols, sms),
                  where + " got a control grid that is not resident");
        }
    }
    if (failures != 0) { std::cerr << failures << " gating plan failures\n"; }
    return failures;
}

} // namespace sm_count_routes

} // namespace

// gdn_norm_gating_proj_rotated must write g and beta word for word as gdn_norm_gating_proj does,
// and h as gdn_norm_gating_proj followed by hadamard_transform, on every norm/control route.
int run_rotated_norm_case(std::int32_t tokens, std::uint32_t seed, DeviceExecutionView execution) {
    constexpr float kEps               = 1.0e-6f;
    const Geometry& geometry           = kQwen27;
    const std::size_t h_elements       = std::size_t(geometry.hidden) * tokens;
    const std::size_t control_elements = std::size_t(geometry.heads) * tokens;
    std::vector<float> x(h_elements), norm_weight(geometry.hidden), signs(geometry.hidden);
    std::vector<float> a_weight(std::size_t(geometry.heads) * geometry.hidden),
        b_weight(a_weight.size());
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0f, 1.0f);
    fill_uniform(norm_weight, seed + 1, -0.2f, 0.2f);
    fill_uniform(a_weight, seed + 2, -0.015f, 0.015f);
    fill_uniform(b_weight, seed + 3, -0.015f, 0.015f);
    fill_uniform(a_log, seed + 4, -2.0f, 1.0f);
    fill_uniform(dt_bias, seed + 5, -1.0f, 1.0f);
    fill_uniform(signs, seed + 6, -1.0f, 1.0f);
    for (auto& sign : signs) sign = sign < 0.0f ? -1.0f : 1.0f;
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);
    DeviceBuffer device_x     = to_device(bf16_bits(x)),
                 device_norm  = to_device(bf16_bits(norm_weight)),
                 device_a     = to_device(bf16_bits(a_weight)),
                 device_b     = to_device(bf16_bits(b_weight)),
                 device_signs = to_device(bf16_bits(signs)), device_a_log = to_device(a_log),
                 device_dt_bias = to_device(dt_bias);
    DeviceBuffer h_primal(h_elements * 2), h_composed(h_elements * 2),
        g_primal(control_elements * 4), beta_primal(control_elements * 4);
    GuardedDeviceBuffer h_rotated(h_elements * 2), g_rotated(control_elements * 4),
        beta_rotated(control_elements * 4);
    h_rotated.fill(0xff);
    g_rotated.fill(0xff);
    beta_rotated.fill(0xff);
    Tensor tx(device_x.p, DType::BF16, {geometry.hidden, tokens}),
        tn(device_norm.p, DType::BF16, {geometry.hidden}),
        ts(device_signs.p, DType::BF16, {geometry.hidden}),
        ta(device_a_log.p, DType::FP32, {geometry.heads}),
        td(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor thp(h_primal.p, DType::BF16, {geometry.hidden, tokens}),
        thc(h_composed.p, DType::BF16, {geometry.hidden, tokens}),
        tgp(g_primal.p, DType::FP32, {geometry.heads, tokens}),
        tbp(beta_primal.p, DType::FP32, {geometry.heads, tokens});
    Tensor thr(h_rotated.data(), DType::BF16, {geometry.hidden, tokens}),
        tgr(g_rotated.data(), DType::FP32, {geometry.heads, tokens}),
        tbr(beta_rotated.data(), DType::FP32, {geometry.heads, tokens});
    const auto wa = bf16_weight(device_a.p, geometry.heads, geometry.hidden);
    const auto wb = bf16_weight(device_b.p, geometry.heads, geometry.hidden);
    const auto capacity =
        std::max<std::size_t>(ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                  geometry.heads, geometry.hidden, tokens, tokens),
                              256);
    DeviceBuffer scratch(capacity);
    WorkspaceArena workspace(DeviceSpan{scratch.p, capacity});
    ops::gdn_norm_gating_proj(tx, tn, kEps, wa, wb, ta, td, workspace, thp, tgp, tbp, execution);
    ops::hadamard_transform(thp, ts, false, thc, execution.stream);
    ops::gdn_norm_gating_proj_rotated(tx, tn, kEps, wa, wb, ta, td, ts, workspace, thr, tgr, tbr,
                                      execution);
    cuda_synchronize(execution.stream);
    const std::string label = "gdn_norm_gating_proj_rotated T=" + std::to_string(tokens);
    int failures            = verify_exact((label + " h").c_str(),
                                           from_device<std::uint16_t>(h_rotated.data(), h_elements),
                                           from_device<std::uint16_t>(h_composed, h_elements));
    failures += verify_exact((label + " g").c_str(),
                             from_device<std::uint32_t>(g_rotated.data(), control_elements),
                             from_device<std::uint32_t>(g_primal, control_elements));
    failures += verify_exact((label + " beta").c_str(),
                             from_device<std::uint32_t>(beta_rotated.data(), control_elements),
                             from_device<std::uint32_t>(beta_primal, control_elements));
    failures += h_rotated.verify_guards(label + " h guards");
    return failures;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    DeviceContext device;
    const DeviceExecutionView execution{nullptr, device.multiprocessor_count()};
    const DeviceExecutionView norm_execution{device.stream, device.multiprocessor_count()};
    int failures = sm_count_routes::verify_plan_across_sm_counts();
    // Both sides of each route boundary of this device's plan (768/769, 1664/1665 and 3456/3457 on
    // an 82-SM sm_86 part; residency moves them with the SM count and the SM's occupancy).
    failures += verify_workspace_capacity_contract(
        kQwen27, plan_endpoints(kQwen27, device.multiprocessor_count(), 3457));
    failures += verify_workspace_capacity_contract(
        kQwen35, plan_endpoints(kQwen35, device.multiprocessor_count(), 3905));

    for (const auto geometry : {Geometry{"dense_0_8b", 1024, 16, false},
                                Geometry{"dense_2b", 2048, 16, true},
                                Geometry{"dense_4b", 2560, 32, false},
                                Geometry{"dense_9b", 4096, 32, true}}) {
        for (const int tokens : {1, 2, 8, 17, 128, 257}) {
            failures += run_projection_case(geometry, tokens, 0xA000u + tokens, execution);
            failures += run_norm_projection_case(geometry, tokens, 0xB000u + tokens, norm_execution);
        }
    }

    // Every registered 27B projection route, including predicated and full token tiles, and both
    // sides of each sm_86 residency boundary (768/769, 1664/1665, 3456/3457).
    for (const std::int32_t tokens :
         {1, 8, 9, 768, 769, 1024, 1664, 1665, 2049, 3456, 3457, 4097}) {
        failures += run_projection_case(kQwen27, tokens,
                                        0x1000u + static_cast<std::uint32_t>(tokens), execution);
    }
    // The Qwen3.8 parent changes only the public storage boundary. One direct oracle case proves
    // its [A,B] row partition; the split 27B cases above cover every unchanged execution route.
    failures += run_projection_case(kQwen38Parent, 1, 0x1801u, execution);
    // Every registered 35B projection route and its contiguous-parent storage contract.
    for (const std::int32_t tokens : {1, 127, 128, 1024, 1025, 2049, 4097}) {
        failures += run_projection_case(kQwen35, tokens,
                                        0x2000u + static_cast<std::uint32_t>(tokens), execution);
    }

    // Direct complete norm/control oracles across variable widths and the existing 35B profile.
    failures += run_norm_projection_case(kQwen27, 1, 0x3001u, norm_execution);
    failures += run_norm_projection_case(kQwen27, 9, 0x3009u, norm_execution);
    failures += run_norm_projection_case(kQwen27, 64, 0x3040u, norm_execution);
    for (int tokens = 1; tokens <= 128; ++tokens)
        failures +=
            run_norm_projection_case(kQwen38Parent, tokens, 0x3800u + tokens, norm_execution);
    for (int tokens : {1, 2, 3, 8, 9, 14, 15, 16, 28, 29, 32, 42, 43, 64, 96, 128, 129, 256})
        failures +=
            run_norm_projection_case(kQwen38Parent, tokens, 0x4800u + tokens, norm_execution, true);
    for (int mode : {1, 2, 3, 4})
        failures +=
            run_norm_projection_case(kQwen38Parent, 16, 0x5800u + mode, norm_execution, true, mode);
    failures += run_norm_projection_case(kQwen35, 1, 0x4001u, norm_execution);
    failures += run_norm_projection_case(kQwen35, 16, 0x4010u, norm_execution);
    failures += run_norm_projection_case(kQwen35, 17, 0x4011u, norm_execution);
    failures += run_norm_projection_case(kQwen35, 64, 0x4040u, norm_execution);

    // Requalify the retained BF16-staging profile at every prefill reduction boundary.
    for (int tokens : {1024, 1025, 2048, 2049, 4097})
        failures +=
            run_norm_projection_case(kQwen38Parent, tokens, 0x6800u + tokens, norm_execution);
    for (int tokens : {2, 8, 15, 127, 128, 1024, 1025, 2048, 2049, 4097})
        failures += run_norm_projection_case(kQwen35, tokens, 0x7800u + tokens, norm_execution,
                                             tokens == 15);

    for (int tokens : {1, 2, 3, 8, 14, 15, 28, 29, 64, 200, 1024})
        failures += run_rotated_norm_case(tokens, 0x9000u + tokens, norm_execution);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating_proj correctness\n";
    return failures == 0 ? 0 : 1;
}
