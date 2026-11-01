// Native Flash-Next experts against independently decoded stored words and complete FP64 math.
#include "core/device.h"
#include "ninfer/ops/moe_experts.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
constexpr int H = 2560, I = 640, TOP = 10, E = 512, DISTINCT = 3;

void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

double half_value(unsigned word) {
    const unsigned exponent = (word >> 10) & 31, mantissa = word & 1023;
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -24)
        : std::ldexp(1.0 + double(mantissa) / 1024, int(exponent) - 15);
    return word & 0x8000 ? -magnitude : magnitude;
}

// Packing and oracle decode use literal contract offsets, independent of device codecs.
struct Bank {
    QType format;
    int rows, k, distinct;
    std::vector<DeviceBuffer> payloads;
    DeviceBuffer operands;
    std::vector<std::vector<double>> exact;

    Bank(QType q, int n, int columns, int count, int entries, bool panel, unsigned seed)
        : format(q), rows(n), k(columns), distinct(count), operands(entries * sizeof(Weight)) {
        const int bits = q == QType::Q2_G64_FP16 ? 2 : q == QType::Q4_G64_FP16 ? 4 :
                         q == QType::Q5_G64_FP16 ? 5 : q == QType::Q6_G64_FP16 ? 6 : 8;
        const int group = q == QType::Q8_G32_FP16 ? 32 : 64;
        const int groups = k / group, base_per = bits == 2 ? 16 : 32;
        const int high_per = bits == 5 ? 8 : bits == 6 ? 16 : 0;
        const std::size_t base_bytes = std::size_t(n) * groups * base_per;
        const std::size_t high_offset = (base_bytes + 255) / 256 * 256;
        const std::size_t scale_offset = high_offset + (std::size_t(n) * groups * high_per + 255) / 256 * 256;
        const std::size_t bytes = q == QType::BF16 ? std::size_t(n) * k * 2 :
                                  scale_offset + std::size_t(n) * groups * 2;
        std::mt19937 random(seed);
        std::vector<Weight> prepared;
        for (int e = 0; e < count; ++e) {
            std::vector<unsigned char> raw(bytes, 0);
            exact.emplace_back(std::size_t(n) * k);
            if (q == QType::BF16) {
                for (std::size_t j = 0; j < std::size_t(n) * k; ++j) {
                    const auto word = ninfer::test::f32_to_bf16((int(random() % 2001) - 1000) * 0.000025f);
                    raw[j * 2] = word & 255; raw[j * 2 + 1] = word >> 8;
                    std::uint32_t fp32 = std::uint32_t(word) << 16;
                    float v; std::memcpy(&v, &fp32, sizeof(v));
                    exact.back()[j] = v;
                }
            } else {
                const unsigned scales[] = {0, 0x8000, 1, 0x9400,
                    bits == 2 ? 0x2400u : bits == 8 ? 0x0800u : 0x1400u};
                for (int r = 0; r < n; ++r) for (int g = 0; g < groups; ++g) {
                    unsigned scale = scales[(r * 7 + g + e) % 5] |
                        (((r + g) % 3 == 0 && (r * 7 + g + e) % 5 >= 2) ? 0x8000 : 0);
                    if (bits != 2 && scale == 0x8000) { scale = 0; }
                    const std::size_t si = scale_offset + (std::size_t(r) * groups + g) * 2;
                    raw[si] = scale & 255; raw[si + 1] = scale >> 8;
                    const std::size_t record = panel ? (std::size_t(r / 64) * groups + g) * 64 + r % 64
                                                     : std::size_t(r) * groups + g;
                    for (int lane = 0; lane < group; ++lane) {
                        unsigned u = random() & ((1u << bits) - 1);
                        if (bits != 2 && scale == 0) { u = 0; }
                        if (bits == 8 && u == 128) { u = 129; }
                        if (bits == 2) { raw[record * 16 + lane / 4] |= u << (2 * (lane % 4)); }
                        else if (bits == 8) { raw[record * 32 + lane] = u; }
                        else {
                            raw[record * 32 + lane / 2] |= (u & 15) << (4 * (lane % 2));
                            if (bits == 5) { raw[high_offset + record * 8 + lane / 8] |= (u >> 4) << (lane % 8); }
                            if (bits == 6) { raw[high_offset + record * 16 + lane / 4] |= (u >> 4) << (2 * (lane % 4)); }
                        }
                    }
                    // Decode the actual payload, including the independently permuted plane.
                    for (int lane = 0; lane < group; ++lane) {
                        unsigned u;
                        if (bits == 2) { u = (raw[record * 16 + lane / 4] >> (2 * (lane % 4))) & 3; }
                        else if (bits == 8) { u = raw[record * 32 + lane]; }
                        else {
                            u = (raw[record * 32 + lane / 2] >> (4 * (lane % 2))) & 15;
                            if (bits == 5) { u |= ((raw[high_offset + record * 8 + lane / 8] >> (lane % 8)) & 1) << 4; }
                            if (bits == 6) { u |= ((raw[high_offset + record * 16 + lane / 4] >> (2 * (lane % 4))) & 3) << 4; }
                        }
                        const int code = bits == 2 ? int(u) - 1 :
                            int(u) - ((u & (1u << (bits - 1))) ? (1 << bits) : 0);
                        const unsigned word = raw[si] | (unsigned(raw[si + 1]) << 8);
                        exact.back()[std::size_t(r) * k + g * group + lane] = code * half_value(word);
                    }
                }
            }
            payloads.emplace_back(bytes);
            payloads.back().copy_from_host(raw.data(), raw.size());
            const auto layout = q == QType::BF16 ? QuantLayout::Contiguous :
                                panel ? QuantLayout::RowSplitPanel : QuantLayout::RowSplit;
            WeightParent parent{weight_geometry(q, layout, std::array<std::uint64_t, 2>{unsigned(n), unsigned(k)}),
                                 static_cast<const std::byte*>(payloads.back().p)};
            const WeightView view{{unsigned(n), unsigned(k)}, {{&parent, 0, std::uint64_t(n) * k}}};
            const ops::WeightInput input{view, ops::LinearPolicy::AllowA8};
            prepared.push_back(ops::prepare_native_expert(input, n, k, false));
            if (q != QType::BF16) {
                bool refused = false;
                try { (void)ops::prepare_native_expert({view, ops::LinearPolicy::A16Only}, n, k, true); }
                catch (const std::invalid_argument&) { refused = true; }
                require(refused, "A16-only expert admitted integer A8");
            }
        }
        std::vector<Weight> table(entries);
        for (int e = 0; e < entries; ++e) { table[e] = prepared[e % count]; }
        operands.copy_from_host(table.data(), operands.bytes);
    }

    ops::NativeExpertTable view(bool a8) const {
        return {format, static_cast<const Weight*>(operands.p), a8 && format != QType::BF16};
    }

    const std::vector<double>& of(int e) const { return exact[e % distinct]; }
};

std::vector<double> expert(const Bank& gate, const Bank& up, const Bank& down, int e,
                           const std::vector<float>& x) {
    std::vector<double> middle(I), output(H);
    for (int r = 0; r < I; ++r) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) {
            g += gate.of(e)[std::size_t(r) * H + c] * x[c];
            u += up.of(e)[std::size_t(r) * H + c] * x[c];
        }
        middle[r] = g / (1 + std::exp(-g)) * u;
    }
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < I; ++c) { output[r] += down.of(e)[std::size_t(r) * I + c] * middle[c]; }
    }
    return output;
}

void run(QType gu, QType d, bool panel, bool mixed = false) {
    const QType shared_format = panel ? QType::Q8_G32_FP16 : QType::BF16;
    const QType up_format = mixed ? QType::Q8_G32_FP16 : gu;
    const QType shared_up_format = mixed ? QType::Q2_G64_FP16 : shared_format;
    Bank gate(gu, I, H, DISTINCT, E, panel, 21), up(up_format, I, H, DISTINCT, E, panel, 35),
         down(d, H, I, DISTINCT, E, panel, 49), sg(shared_format, I, H, 1, 1, panel, 63),
         su(shared_up_format, I, H, 1, 1, panel, 77), sd(shared_format, H, I, 1, 1, panel, 91);
    std::array<std::vector<float>, 3> input;
    std::array<std::vector<std::vector<double>>, 3> routed;
    std::array<std::vector<double>, 3> shared_output;
    for (int variant = 0; variant < 3; ++variant) {
        input[variant].resize(H);
        for (int c = 0; c < H; ++c) {
            input[variant][c] = variant == 2 ? 0 : std::sin(float(c * 13 + variant * 7)) * 1.3f;
        }
        ninfer::test::round_to_bf16(input[variant]);
        for (int e = 0; e < DISTINCT; ++e) { routed[variant].push_back(expert(gate, up, down, e, input[variant])); }
        shared_output[variant] = expert(sg, su, sd, 0, input[variant]);
    }
    for (bool a8 : {false, true}) {
        if (a8 && gu == QType::BF16 && !mixed) { continue; }
        const ops::NativeMoeWeights banks{gate.view(a8), up.view(a8), down.view(a8),
            sg.view(a8), su.view(a8), sd.view(a8), E};
        for (int tokens : {1, 2, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129}) {
            std::vector<__nv_bfloat16> x(std::size_t(H) * tokens);
            std::vector<int> ids(TOP * tokens);
            std::vector<float> weights(ids.size()), shared(tokens);
            std::vector<double> expected(std::size_t(H) * tokens);
            for (int t = 0; t < tokens; ++t) {
                for (int c = 0; c < H; ++c) { x[std::size_t(t) * H + c] = __float2bfloat16(input[t % 3][c]); }
                shared[t] = t == 0 ? 0 : 0.1f + (t % 7) * 0.1f;
                for (int slot = 0; slot < TOP; ++slot) {
                    const int e = t == 0 && slot == 0 ? 511 : (t * 13 + slot * 51) % E;
                    ids[t * TOP + slot] = e;
                    weights[t * TOP + slot] = float(slot + 1) / 55;
                    for (int c = 0; c < H; ++c) {
                        expected[std::size_t(t) * H + c] += weights[t * TOP + slot] * routed[t % 3][e % DISTINCT][c];
                    }
                }
                for (int c = 0; c < H; ++c) { expected[std::size_t(t) * H + c] += shared[t] * shared_output[t % 3][c]; }
            }
            DeviceBuffer dx(x.size() * 2), di(ids.size() * 4), dw(weights.size() * 4),
                ds(shared.size() * 4), dy(expected.size() * 4);
            dx.copy_from_host(x.data(), dx.bytes); di.copy_from_host(ids.data(), di.bytes);
            dw.copy_from_host(weights.data(), dw.bytes); ds.copy_from_host(shared.data(), ds.bytes);
            Tensor tx(dx.p, DType::BF16, {H, tokens}), ti(di.p, DType::I32, {TOP, tokens}),
                tw(dw.p, DType::FP32, {TOP, tokens}), ts(ds.p, DType::FP32, {tokens}),
                ty(dy.p, DType::FP32, {H, tokens});
            WorkspaceArena workspace(ops::moe_experts_native_workspace_bytes(tokens));
            const auto execute = [&] { ops::moe_experts_native(tx, ti, tw, ts, banks, nullptr, workspace, ty, nullptr); };
            if (tokens == 1) {
                const auto refused = [&](const ops::NativeMoeWeights& invalid) {
                    bool rejected = false;
                    try { ops::moe_experts_native(tx, ti, tw, ts, invalid, nullptr, workspace, ty, nullptr); }
                    catch (const std::invalid_argument&) { rejected = true; }
                    require(rejected, "native expert accepted an unsupported bank contract");
                };
                auto invalid = banks;
                invalid.experts = 9; refused(invalid);
                invalid = banks; invalid.gate.format = QType::GGUF_Q2_0; refused(invalid);
                invalid = banks; invalid.shared_down.experts = nullptr; refused(invalid);
                invalid = banks; invalid.up.format = QType::BF16;
                invalid.up.integer_a8 = true; refused(invalid);
            }
            execute(); check(cudaDeviceSynchronize());
            std::vector<float> actual(expected.size()), repeat(expected.size());
            dy.copy_to_host(actual.data(), dy.bytes);
            execute(); check(cudaDeviceSynchronize()); dy.copy_to_host(repeat.data(), dy.bytes);
            require(std::memcmp(actual.data(), repeat.data(), dy.bytes) == 0, "native expert repeat differs");
            if (tokens == 9 || tokens == 17 || tokens == 33) {
                cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
                check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                ops::moe_experts_native(tx, ti, tw, ts, banks, nullptr, workspace, ty, stream);
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (int replay = 0; replay < 3; ++replay) { check(cudaGraphLaunch(executable, stream)); }
                check(cudaStreamSynchronize(stream)); dy.copy_to_host(repeat.data(), dy.bytes);
                require(std::memcmp(actual.data(), repeat.data(), dy.bytes) == 0, "native expert graph replay differs");
                check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
            }
            double error = 0, norm = 0, largest = 0, peak = 0;
            for (std::size_t j = 0; j < actual.size(); ++j) {
                require(std::isfinite(actual[j]), "native expert output is not finite");
                const double delta = actual[j] - expected[j];
                error += delta * delta; norm += expected[j] * expected[j];
                largest = std::max(largest, std::abs(delta)); peak = std::max(peak, std::abs(expected[j]));
                if ((j / H) % 3 == 2) { require(actual[j] == 0, "zero activation changed during native expert arithmetic"); }
            }
            const double relative = std::sqrt(error / std::max(norm, 1e-30));
            const double gross = largest / std::max(peak, 1e-15);
            std::cout << "native gu=" << int(gu) << " down=" << int(d) << " panel=" << panel
                      << " mixed=" << mixed << " a8=" << a8 << " T=" << tokens << " relative_l2=" << relative
                      << " max_over_peak=" << gross << '\n';
            require(relative < (a8 ? 0.04 : 0.001) && gross < (a8 ? 0.16 : 0.004),
                    "native experts exceed their complete FP64 oracle bound");
            if (gu == QType::Q2_G64_FP16 && d == QType::Q2_G64_FP16 && !mixed &&
                (tokens == 1 || tokens == 16 || tokens == 17)) {
                DeviceBuffer selected(TOP * tokens), empty_operands(E * sizeof(Weight));
                empty_operands.fill(0);
                Tensor parts(selected.p, DType::U8, {TOP, tokens});
                for (const bool none : {false, true}) {
                    std::vector<unsigned char> mask(TOP * tokens);
                    for (std::size_t p = 0; p < mask.size(); ++p) { mask[p] = none ? 0 : p % 3 != 0; }
                    selected.copy_from_host(mask.data(), mask.size());
                    auto partitioned = banks;
                    if (none) {
                        // An excluded expert's operand table may have no valid weight pointer.
                        partitioned.gate.experts = static_cast<const Weight*>(empty_operands.p);
                        partitioned.up.experts = static_cast<const Weight*>(empty_operands.p);
                        partitioned.down.experts = static_cast<const Weight*>(empty_operands.p);
                    }
                    const auto partition = [&] {
                        ops::moe_experts_native(tx, ti, tw, ts, partitioned, &parts, workspace, ty, nullptr);
                    };
                    partition();
                    dy.copy_to_host(actual.data(), dy.bytes);
                    partition();
                    dy.copy_to_host(repeat.data(), dy.bytes);
                    require(std::memcmp(actual.data(), repeat.data(), dy.bytes) == 0,
                            "partitioned native repeat differs");
                    double part_error = 0, part_norm = 0, part_largest = 0, part_peak = 0;
                    for (int t = 0; t < tokens; ++t) for (int row = 0; row < H; ++row) {
                        double reference = shared[t] * shared_output[t % 3][row];
                        for (int slot = 0; slot < TOP; ++slot) {
                            const int p = t * TOP + slot;
                            if (mask[p]) { reference += weights[p] * routed[t % 3][ids[p] % DISTINCT][row]; }
                        }
                        const float value = actual[std::size_t(t) * H + row];
                        require(std::isfinite(value), "partitioned native nonfinite output");
                        const double delta = value - reference;
                        part_error += delta * delta; part_norm += reference * reference;
                        part_largest = std::max(part_largest, std::abs(delta));
                        part_peak = std::max(part_peak, std::abs(reference));
                    }
                    const double part_relative = std::sqrt(part_error / std::max(part_norm, 1e-30));
                    const double part_gross = part_largest / std::max(part_peak, 1e-15);
                    std::cout << "native parts none=" << none << " T=" << tokens << " a8=" << a8
                              << " panel=" << panel << " relative_l2=" << part_relative
                              << " max_over_peak=" << part_gross << '\n';
                    require(part_relative < (a8 ? 0.04 : 0.001) && part_gross < (a8 ? 0.16 : 0.004),
                            "partitioned native exceeds complete FP64 oracle bound");
                }
            }
        }
    }
}
} // namespace

int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    check(cudaSetDevice(0));
    for (bool panel : {false, true}) {
        run(QType::Q2_G64_FP16, QType::Q2_G64_FP16, panel);
        run(QType::Q4_G64_FP16, QType::Q5_G64_FP16, panel);
        run(QType::Q4_G64_FP16, QType::Q6_G64_FP16, panel);
        run(QType::Q5_G64_FP16, QType::Q5_G64_FP16, panel);
        run(QType::Q6_G64_FP16, QType::Q6_G64_FP16, panel);
        run(QType::Q8_G32_FP16, QType::Q8_G32_FP16, panel);
    }
    run(QType::BF16, QType::BF16, false);
    run(QType::Q2_G64_FP16, QType::Q2_G64_FP16, false, true);
    run(QType::Q2_G64_FP16, QType::Q2_G64_FP16, true, true);
    run(QType::BF16, QType::BF16, false, true);
    std::cout << "NATIVE_EXPERTS_PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
