// Complete CPU SwiGLU against FP64 from independently decoded physical weight bytes.
#include "ninfer/ops/moe_expert_cpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;
constexpr int H = 2560, I = 640;

void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
template<class F> void refused(F&& call) {
    try { call(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("CPU expert accepted invalid operands");
}
double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25) :
                                    std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}
std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    return static_cast<std::uint16_t>(word >> 16);
}

struct Bank {
    Weight weight;
    std::vector<std::uint8_t> data, high, scales;
    std::vector<double> exact;

    Bank(QType format, int rows, int columns, bool panel, unsigned seed)
        : exact(std::size_t(rows) * columns) {
        weight.qtype = format;
        weight.n = rows;
        weight.k = columns;
        weight.scale_dtype = DType::FP16;
        std::mt19937 random(seed);
        if (format == QType::BF16) {
            weight.layout = QuantLayout::Contiguous;
            data.resize(exact.size() * 2);
            for (std::size_t i = 0; i < exact.size(); ++i) {
                const auto word = bf16((int(random() % 2001) - 1000) * 0.000025F);
                data[i * 2] = word & 255;
                data[i * 2 + 1] = word >> 8;
                exact[i] = std::bit_cast<float>(std::uint32_t(word) << 16);
            }
            weight.qdata = data.data();
            return;
        }
        const int bits = format == QType::Q2_G64_FP16 ? 2 : format == QType::Q4_G64_FP16 ? 4 :
            format == QType::Q5_G64_FP16 ? 5 : format == QType::Q6_G64_FP16 ? 6 : 8;
        const int group = bits == 8 ? 32 : 64, groups = columns / group;
        const int low_bytes = bits == 2 ? 16 : 32;
        const int high_bytes = bits == 5 ? 8 : bits == 6 ? 16 : 0;
        weight.group = group;
        weight.layout = panel ? QuantLayout::RowSplitPanel : QuantLayout::RowSplit;
        data.resize(std::size_t(rows) * groups * low_bytes);
        high.resize(std::size_t(rows) * groups * high_bytes);
        scales.resize(std::size_t(rows) * groups * 2);
        for (int row = 0; row < rows; ++row) for (int g = 0; g < groups; ++g) {
            const unsigned choices[] = {0, 0x8000, 1, 0x9400,
                bits == 2 ? 0x2400U : bits == 8 ? 0x0800U : 0x1400U};
            const unsigned scale = choices[(row * 7 + g) % 5];
            const std::size_t s = (std::size_t(row) * groups + g) * 2;
            scales[s] = scale & 255; scales[s + 1] = scale >> 8;
            const std::size_t record = panel ? (std::size_t(row / 64) * groups + g) * 64 + row % 64 :
                                               std::size_t(row) * groups + g;
            for (int lane = 0; lane < group; ++lane) {
                const unsigned u = random() & ((1U << bits) - 1);
                if (bits == 2) { data[record * 16 + lane / 4] |= u << (2 * (lane % 4)); }
                else if (bits == 8) { data[record * 32 + lane] = u; }
                else {
                    data[record * 32 + lane / 2] |= (u & 15) << (4 * (lane % 2));
                    if (bits == 5) { high[record * 8 + lane / 8] |= (u >> 4) << (lane % 8); }
                    if (bits == 6) { high[record * 16 + lane / 4] |= (u >> 4) << (2 * (lane % 4)); }
                }
            }
            // Decode the stored payload independently; production helpers are not the oracle.
            for (int lane = 0; lane < group; ++lane) {
                unsigned u;
                if (bits == 2) { u = (data[record * 16 + lane / 4] >> (2 * (lane % 4))) & 3; }
                else if (bits == 8) { u = data[record * 32 + lane]; }
                else {
                    u = (data[record * 32 + lane / 2] >> (4 * (lane % 2))) & 15;
                    if (bits == 5) { u |= ((high[record * 8 + lane / 8] >> (lane % 8)) & 1) << 4; }
                    if (bits == 6) { u |= ((high[record * 16 + lane / 4] >> (2 * (lane % 4))) & 3) << 4; }
                }
                const int code = bits == 2 ? int(u) - 1 : int(u) -
                    ((u & (1U << (bits - 1))) ? (1 << bits) : 0);
                const unsigned stored_scale = scales[s] | (unsigned(scales[s + 1]) << 8);
                exact[std::size_t(row) * columns + g * group + lane] = double(code) * exact_half(stored_scale);
            }
        }
        weight.qdata = data.data(); weight.qhigh = high.empty() ? nullptr : high.data(); weight.scales = scales.data();
    }
};

std::vector<double> oracle(const Bank& gate, const Bank& up, const Bank& down,
                           std::span<const std::uint16_t> input) {
    std::vector<double> middle(I), output(H);
    for (int row = 0; row < I; ++row) {
        double g = 0, u = 0;
        for (int column = 0; column < H; ++column) {
            const double x = std::bit_cast<float>(std::uint32_t(input[column]) << 16);
            g += gate.exact[std::size_t(row) * H + column] * x;
            u += up.exact[std::size_t(row) * H + column] * x;
        }
        middle[row] = g / (1.0 + std::exp(-g)) * u;
    }
    for (int row = 0; row < H; ++row) {
        for (int column = 0; column < I; ++column) {
            output[row] += down.exact[std::size_t(row) * I + column] * middle[column];
        }
    }
    return output;
}

void run_case(QType gate_type, QType up_type, QType down_type, bool panel) {
    Bank gate(gate_type, I, H, panel, 17), up(up_type, I, H, panel, 31), down(down_type, H, I, panel, 59);
    std::array<std::vector<std::uint16_t>, 3> input;
    std::array<std::vector<double>, 3> expected;
    for (int variant = 0; variant < 3; ++variant) {
        input[variant].resize(H);
        for (int i = 0; i < H; ++i) {
            input[variant][i] = bf16(variant == 2 ? 0.0F : std::sin(float(i * 13 + variant * 7)) * 1.3F);
        }
        expected[variant] = oracle(gate, up, down, input[variant]);
    }
    CpuExpertWorkspace workspace(16);
    for (const auto backend : {CpuExpertBackend::Scalar, CpuExpertBackend::Avx2, CpuExpertBackend::Avx512Vnni}) {
        if (!cpu_expert_backend_available(backend)) { continue; }
        for (const bool a8 : {false, true}) {
            if (a8 && gate_type == QType::BF16 && up_type == QType::BF16 && down_type == QType::BF16) { continue; }
            CpuExpertWeights weights{{gate.weight, a8 && gate_type != QType::BF16},
                                      {up.weight, a8 && up_type != QType::BF16},
                                      {down.weight, a8 && down_type != QType::BF16}};
            for (const int tokens : {16, 1, 3}) {
                std::vector<std::uint16_t> x(std::size_t(tokens) * H);
                std::vector<float> actual(x.size()), repeat(x.size());
                for (int t = 0; t < tokens; ++t) {
                    std::copy(input[t % 3].begin(), input[t % 3].end(), x.begin() + t * H);
                }
                moe_expert_cpu(x, weights, workspace, actual, backend);
                if (tokens == 3) {
                    moe_expert_cpu(x, weights, workspace, repeat, backend);
                    require(std::memcmp(actual.data(), repeat.data(), actual.size() * sizeof(float)) == 0,
                            "CPU fixed-mode repeat differs");
                    std::atomic<bool> cancelled{true};
                    bool stopped = false;
                    try { moe_expert_cpu(x, weights, workspace, repeat, backend, &cancelled); }
                    catch (const CpuExpertCancelled&) { stopped = true; }
                    require(stopped, "CPU expert ignored cancellation");
                    require(std::memcmp(actual.data(), repeat.data(), actual.size() * sizeof(float)) == 0,
                            "already-cancelled CPU expert wrote output");
                    cancelled.store(false);
                    moe_expert_cpu(x, weights, workspace, repeat, backend, &cancelled);
                    require(std::memcmp(actual.data(), repeat.data(), actual.size() * sizeof(float)) == 0,
                            "CPU workspace did not recover after cancellation");
                }
                double error = 0, norm = 0, largest = 0, peak = 0;
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    require(std::isfinite(actual[i]), "CPU expert produced nonfinite output");
                    const double reference = expected[(i / H) % 3][i % H];
                    const double delta = actual[i] - reference;
                    error += delta * delta; norm += reference * reference;
                    largest = std::max(largest, std::abs(delta)); peak = std::max(peak, std::abs(reference));
                    if ((i / H) % 3 == 2) { require(actual[i] == 0, "zero CPU activation changed"); }
                }
                const double relative = std::sqrt(error / std::max(norm, 1e-30));
                const double gross = largest / std::max(peak, 1e-15);
                std::cout << cpu_expert_backend_name(backend) << " formats=" << int(gate_type) << ','
                          << int(up_type) << ',' << int(down_type) << " panel=" << panel << " a8=" << a8
                          << " T=" << tokens << " relative_l2=" << relative << " max_over_peak=" << gross << '\n';
                require(relative < (a8 ? 0.04 : 0.001) && gross < (a8 ? 0.16 : 0.004),
                        "CPU expert exceeds complete FP64 oracle bound");
            }
        }
    }
    CpuExpertWeights weights{{gate.weight, false}, {up.weight, false}, {down.weight, false}};
    std::vector<float> output(H);
    refused([&] { moe_expert_cpu({}, weights, workspace, {}, CpuExpertBackend::Scalar); });
    refused([&] { moe_expert_cpu(std::span(input[0]).first(H - 1), weights, workspace, output); });
    weights.gate.weight.n = I - 1;
    refused([&] { moe_expert_cpu(input[0], weights, workspace, output); });
    weights.gate.weight = gate.weight;
    if (gate_type == QType::BF16) {
        weights.gate.integer_a8 = true;
        refused([&] { moe_expert_cpu(input[0], weights, workspace, output); });
    } else if (gate_type == QType::Q5_G64_FP16) {
        weights.gate.weight.qhigh = nullptr;
        refused([&] { moe_expert_cpu(input[0], weights, workspace, output); });
    }
}
} // namespace

int main() try {
    for (const bool panel : {false, true}) {
        run_case(QType::Q2_G64_FP16, QType::Q2_G64_FP16, QType::Q2_G64_FP16, panel);
    }
    run_case(QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16, true);
    run_case(QType::Q5_G64_FP16, QType::Q4_G64_FP16, QType::Q8_G32_FP16, false);
    run_case(QType::BF16, QType::Q2_G64_FP16, QType::Q8_G32_FP16, true);
    run_case(QType::BF16, QType::BF16, QType::BF16, false);
    std::cout << "CPU_EXPERT_ORACLE_PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
