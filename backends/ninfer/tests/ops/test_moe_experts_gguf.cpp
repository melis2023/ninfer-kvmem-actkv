// Qwen3.8-Flash-Next's MoE experts over GGUF block banks (moe_experts_gguf) against an FP64 product
// of the exactly dequantized weights. The tables point several experts at one buffer, so a few
// distinct banks cover all 512 experts while the routing still groups by expert id. The products
// quantize their activations to q8_1 as llama.cpp does, so they are held to that arithmetic's
// error. Up to eight tokens take the vector kernel; wider calls over device-resident banks (each
// matrix followed by 256 zero bytes, which the matrix kernel's last K step of a 640-value down row
// reads) take the matrix kernel wherever the block type has one (IQ1_M stays on the vector
// kernel), with one or several column tiles per expert, and the same calls over banks that are not
// declared device resident take the vector kernel. Where the CPU has the bank's block types, the
// decode form's second stage also comes from CPU products (gguf_expert_cpu, its automatically
// selected backend), as a miss service with a CPU share supplies it, and the sum must equal the
// GPU's bit for bit.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/moe_experts.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kHidden = 2560, kWidth = 640, kExperts = 512, kTopK = 10;

// Independent stored Q2_0 decoder: binary16 multiplier and four adjacent 2-bit codes per byte.
// Do not use the production GPU decoder to construct this format's mathematical oracle.
float exact_q2_0(const std::uint8_t* block, int element) {
    const std::uint16_t scale = std::uint16_t(block[0]) | (std::uint16_t(block[1]) << 8U);
    const int exponent = (scale >> 10U) & 31, fraction = scale & 1023;
    if (exponent == 31) { throw std::runtime_error("nonfinite Q2_0 scale in fixture"); }
    const double magnitude = exponent ? std::ldexp(1024.0 + fraction, exponent - 25)
                                      : std::ldexp(double(fraction), -24);
    const double multiplier = scale & 0x8000 ? -magnitude : magnitude;
    const int code = (block[2 + element / 4] >> (2 * (element % 4))) & 3;
    return float(multiplier * (code - 1));
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Byte offsets of each type's binary16 multipliers, set to finite values (gguf linear test).
std::vector<int> half_fields(QType q) {
    switch (q) {
    case QType::GGUF_Q2_K:
        return {80, 82};
    case QType::GGUF_Q3_K:
        return {108};
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K:
        return {0, 2};
    case QType::GGUF_Q6_K:
        return {208};
    case QType::GGUF_IQ1_M:
        return {};
    default:
        return {0};
    }
}

std::vector<std::uint8_t> random_blocks(QType q, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
    const auto block         = gguf_block_shape(q);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : half_fields(q)) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (q == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] =
                    static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
        }
    }
    return bytes;
}

// `distinct` random matrices of one projection, the device table that maps expert e to matrix
// e % distinct, and their exact values.
// Zero bytes after every matrix, as a device-resident bank keeps them.
constexpr std::size_t kTail = 256;

struct Bank {
    QType format;
    std::int64_t row_bytes = 0;
    std::vector<DeviceBuffer> matrices;
    DeviceBuffer table;
    std::vector<std::vector<float>> exact; // [distinct][rows * k]
    std::vector<std::vector<std::uint8_t>> host; // [distinct] stored blocks
    int distinct = 1;

    Bank(QType q, int rows, int k, int count, int experts, std::mt19937& rng)
        : format(q), table(std::size_t(experts) * sizeof(void*)), distinct(count) {
        const auto block = gguf_block_shape(q);
        row_bytes        = std::int64_t(k / block.elements) * block.bytes;
        std::vector<const void*> pointers(experts);
        for (int i = 0; i < count; ++i) {
            const auto bytes = random_blocks(q, rows, k, rng);
            matrices.emplace_back(bytes.size() + kTail);
            check(cudaMemset(matrices.back().p, 0, bytes.size() + kTail), "zero a bank");
            matrices.back().copy_from_host(bytes.data(), bytes.size());
            host.push_back(bytes);
            exact.emplace_back(std::size_t(rows) * k);
            if (q == QType::GGUF_Q2_0) {
                for (std::size_t element = 0; element < exact.back().size(); ++element) {
                    exact.back()[element] = exact_q2_0(bytes.data() + (element / 64) * 18,
                                                       int(element % 64));
                }
            } else {
                DeviceBuffer values(std::size_t(rows) * k * sizeof(float));
                ops::gguf::dequantize_rows(ops::detail::gguf_type(q), matrices.back().p, row_bytes, k,
                                           nullptr, rows, static_cast<float*>(values.p), k, nullptr);
                check(cudaDeviceSynchronize(), "dequantize");
                values.copy_to_host(exact.back().data(), exact.back().size() * sizeof(float));
            }
        }
        for (int e = 0; e < experts; ++e) { pointers[e] = matrices[e % count].p; }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] ops::GgufExpertTable view() const {
        return {format, static_cast<const void* const*>(table.p), row_bytes};
    }

    [[nodiscard]] const float* rows_of(int expert) const { return exact[expert % distinct].data(); }
};

double silu(double x) { return x / (1.0 + std::exp(-x)); }

// One expert's product for one token (BF16 words) on the CPU, as the miss service computes it.
std::vector<float> cpu_expert(const Bank& gate, const Bank& up, const Bank& down, int expert,
                              const std::uint16_t* x) {
    const auto matrix = [&](const Bank& bank, int rows, int k) {
        return ops::GgufCpuMatrix{bank.format, bank.host[expert % bank.distinct].data(),
                                  bank.row_bytes, rows, k};
    };
    constexpr auto backend = ops::GgufCpuBackend::Automatic;
    ops::GgufCpuActivation in(1, kHidden), in_up(1, kHidden), middle(1, kWidth);
    in.prepare_bf16(gate.format, x, kHidden, 1);
    in_up.prepare_bf16(up.format, x, kHidden, 1);
    std::vector<float> g(kWidth), u(kWidth), y(kHidden);
    ops::gguf_cpu_rows(matrix(gate, kWidth, kHidden), in, 0, kWidth, g.data(), kWidth, backend);
    ops::gguf_cpu_rows(matrix(up, kWidth, kHidden), in_up, 0, kWidth, u.data(), kWidth, backend);
    std::vector<std::uint16_t> m(kWidth);
    ops::gguf_cpu_swiglu_bf16(g.data(), u.data(), kWidth, m.data(), backend);
    middle.prepare_bf16(down.format, m.data(), kWidth, 1);
    ops::gguf_cpu_rows(matrix(down, kHidden, kWidth), middle, 0, kHidden, y.data(), kHidden,
                       backend);
    return y;
}

// down . (silu(gate . x) * (up . x)) in FP64.
std::vector<double> expert_output(const float* gate, const float* up, const float* down,
                                  const std::vector<float>& x) {
    std::vector<double> middle(kWidth);
    for (int r = 0; r < kWidth; ++r) {
        double g = 0, u = 0;
        const float* gr = gate + std::size_t(r) * kHidden;
        const float* ur = up + std::size_t(r) * kHidden;
        for (int i = 0; i < kHidden; ++i) {
            g += double(gr[i]) * x[i];
            u += double(ur[i]) * x[i];
        }
        middle[r] = silu(g) * u;
    }
    std::vector<double> out(kHidden);
    for (int r = 0; r < kHidden; ++r) {
        double acc      = 0;
        const float* dr = down + std::size_t(r) * kWidth;
        for (int i = 0; i < kWidth; ++i) { acc += double(dr[i]) * middle[i]; }
        out[r] = acc;
    }
    return out;
}

struct Case {
    const char* name;
    QType gate, up, down, shared_gate, shared_up, shared_down;
    int distinct;
    std::vector<int> tokens;
    int expert_pool; // routes draw from experts [0, expert_pool)
};

int run_case(const Case& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const Bank gate(c.gate, kWidth, kHidden, c.distinct, kExperts, rng);
    const Bank up(c.up, kWidth, kHidden, c.distinct, kExperts, rng);
    const Bank down(c.down, kHidden, kWidth, c.distinct, kExperts, rng);
    const Bank sgate(c.shared_gate, kWidth, kHidden, 1, 1, rng);
    const Bank sup(c.shared_up, kWidth, kHidden, 1, 1, rng);
    const Bank sdown(c.shared_down, kHidden, kWidth, 1, 1, rng);
    int failures = 0;
    std::vector<std::pair<int, bool>> calls; // tokens, device resident
    for (const int tokens : c.tokens) {
        calls.emplace_back(tokens, true);
        if (tokens > 8) { calls.emplace_back(tokens, false); }
    }
    for (const auto& [tokens, device_resident] : calls) {
        const ops::GgufMoeWeights banks{gate.view(),  up.view(),  down.view(), sgate.view(),
                                        sup.view(),   sdown.view(), kExperts,  device_resident};
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> unit(0.05f, 1.0f);
        std::vector<__nv_bfloat16> m(std::size_t(kHidden) * tokens);
        for (auto& v : m) { v = __float2bfloat16(normal(rng)); }
        std::vector<std::int32_t> ids(std::size_t(kTopK) * tokens);
        std::vector<float> weights(ids.size()), shared(tokens);
        std::vector<int> pool(c.expert_pool);
        std::iota(pool.begin(), pool.end(), 0);
        for (int t = 0; t < tokens; ++t) {
            std::shuffle(pool.begin(), pool.end(), rng);
            float total = 0;
            for (int j = 0; j < kTopK; ++j) {
                ids[t * kTopK + j]     = pool[j];
                weights[t * kTopK + j] = unit(rng);
                total += weights[t * kTopK + j];
            }
            for (int j = 0; j < kTopK; ++j) { weights[t * kTopK + j] /= total; }
            shared[t] = unit(rng);
        }
        DeviceBuffer d_m(m.size() * 2), d_ids(ids.size() * 4), d_w(weights.size() * 4),
            d_s(shared.size() * 4), d_y(std::size_t(kHidden) * tokens * 4);
        d_m.copy_from_host(m.data(), m.size() * 2);
        d_ids.copy_from_host(ids.data(), ids.size() * 4);
        d_w.copy_from_host(weights.data(), weights.size() * 4);
        d_s.copy_from_host(shared.data(), shared.size() * 4);
        Tensor t_m(d_m.p, DType::BF16, {kHidden, tokens}),
            t_ids(d_ids.p, DType::I32, {kTopK, tokens}), t_w(d_w.p, DType::FP32, {kTopK, tokens}),
            t_s(d_s.p, DType::FP32, {tokens}), t_y(d_y.p, DType::FP32, {kHidden, tokens});
        WorkspaceArena workspace(ops::moe_experts_gguf_workspace_bytes(tokens) + (1U << 20));
        ops::moe_experts_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        check(cudaDeviceSynchronize(), "moe_experts_gguf");
        std::vector<float> got(std::size_t(kHidden) * tokens);
        d_y.copy_to_host(got.data(), got.size() * 4);
        // A second run must reproduce every bit: the accumulation is order independent.
        ops::moe_experts_gguf(t_m, t_ids, t_w, t_s, banks, workspace, t_y, nullptr);
        check(cudaDeviceSynchronize(), "moe_experts_gguf repeat");
        std::vector<float> again(got.size());
        d_y.copy_to_host(again.data(), again.size() * 4);
        if (std::memcmp(got.data(), again.data(), got.size() * 4) != 0) {
            std::cerr << "FAIL " << c.name << " T=" << tokens
                      << (device_resident ? " matrix" : " vector") << ": a repeat differs\n";
            ++failures;
        }
        if (ops::moe_experts_gguf_decode_supported(banks, tokens)) {
            // The decode form must equal the separate kernels bit for bit, whole and split into a
            // first stage and a second (the even and odd pairs) as missing experts are served.
            const auto compare = [&](const char* what) {
                check(cudaDeviceSynchronize(), what);
                std::vector<float> other(got.size());
                d_y.copy_to_host(other.data(), other.size() * 4);
                if (std::memcmp(got.data(), other.data(), got.size() * 4) != 0) {
                    std::cerr << "FAIL " << c.name << " T=" << tokens << ": " << what
                              << " differs from the decode form\n";
                    ++failures;
                }
            };
            {
                auto scope       = workspace.scope();
                const auto stage = ops::moe_experts_gguf_begin(t_m, workspace, nullptr);
                ops::moe_experts_gguf_add(stage, t_m, t_ids, t_w, &t_s, banks, workspace, nullptr);
                ops::moe_experts_gguf_finish(stage, t_y, nullptr);
            }
            compare("the separate kernels");
            std::vector<std::int32_t> first(ids.size()), second(ids.size());
            for (std::size_t p = 0; p < ids.size(); ++p) {
                const bool odd = ids[p] % 2 != 0;
                first[p]       = odd ? -1 : ids[p];
                second[p]      = odd ? ids[p] : -1;
            }
            DeviceBuffer d_first(ids.size() * 4), d_second(ids.size() * 4);
            d_first.copy_from_host(first.data(), first.size() * 4);
            d_second.copy_from_host(second.data(), second.size() * 4);
            const Tensor t_first(d_first.p, DType::I32, {kTopK, tokens});
            const Tensor t_second(d_second.p, DType::I32, {kTopK, tokens});
            {
                auto scope       = workspace.scope();
                const auto state = ops::moe_experts_gguf_decode_begin(t_m, t_first, t_s, banks,
                                                                      workspace, nullptr);
                ops::moe_experts_gguf_decode_missing(state, t_m, t_second, banks, nullptr);
                ops::moe_experts_gguf_decode_finish(state, t_first, &t_second, t_w, banks, &banks,
                                                    nullptr, nullptr, nullptr, t_y, nullptr);
            }
            compare("the two-stage decode form");
            {
                // The shared expert on a side stream, joined before the sum is read.
                ops::GgufMoeSide side;
                check(cudaStreamCreateWithFlags(&side.stream, cudaStreamNonBlocking), "side stream");
                check(cudaEventCreateWithFlags(&side.fork, cudaEventDisableTiming), "fork event");
                check(cudaEventCreateWithFlags(&side.join, cudaEventDisableTiming), "join event");
                {
                    auto scope       = workspace.scope();
                    const auto state = ops::moe_experts_gguf_decode_begin(
                        t_m, t_first, t_s, banks, workspace, nullptr, &side);
                    ops::moe_experts_gguf_decode_missing(state, t_m, t_second, banks, nullptr);
                    ops::moe_experts_gguf_decode_finish(state, t_first, &t_second, t_w, banks,
                                                        &banks, nullptr, nullptr, nullptr, t_y,
                                                        nullptr);
                }
                compare("the decode form with a side stream");
                check(cudaEventDestroy(side.fork), "fork event");
                check(cudaEventDestroy(side.join), "join event");
                check(cudaStreamDestroy(side.stream), "side stream");
            }
            {
                auto scope       = workspace.scope();
                const auto stage = ops::moe_experts_gguf_begin(t_m, workspace, nullptr);
                ops::moe_experts_gguf_add(stage, t_m, t_first, t_w, &t_s, banks, workspace, nullptr);
                ops::moe_experts_gguf_add(stage, t_m, t_second, t_w, nullptr, banks, workspace,
                                          nullptr);
                ops::moe_experts_gguf_finish(stage, t_y, nullptr);
            }
            compare("the two-stage separate kernels");
            if (ops::gguf_cpu_supported(c.gate) && ops::gguf_cpu_supported(c.up) &&
                ops::gguf_cpu_supported(c.down)) {
                // The second stage's pairs as CPU products, the GPU running only the first.
                std::vector<float> products;
                std::vector<std::int32_t> pairs;
                for (std::size_t p = 0; p < ids.size(); ++p) {
                    if (second[p] < 0) { continue; }
                    const auto* x = reinterpret_cast<const std::uint16_t*>(m.data()) +
                                    (p / kTopK) * std::size_t(kHidden);
                    const auto y = cpu_expert(gate, up, down, second[p], x);
                    products.insert(products.end(), y.begin(), y.end());
                    pairs.push_back(std::int32_t(p));
                }
                const std::int32_t count = std::int32_t(pairs.size());
                const std::vector<std::int32_t> none(ids.size(), -1);
                DeviceBuffer d_products(std::max<std::size_t>(products.size(), 1) * 4),
                    d_pairs(std::max<std::size_t>(pairs.size(), 1) * 4), d_count(4),
                    d_none(ids.size() * 4);
                if (count > 0) {
                    d_products.copy_from_host(products.data(), products.size() * 4);
                    d_pairs.copy_from_host(pairs.data(), pairs.size() * 4);
                }
                d_count.copy_from_host(&count, 4);
                d_none.copy_from_host(none.data(), none.size() * 4);
                const Tensor t_none(d_none.p, DType::I32, {kTopK, tokens});
                {
                    auto scope       = workspace.scope();
                    const auto state = ops::moe_experts_gguf_decode_begin(t_m, t_first, t_s, banks,
                                                                          workspace, nullptr);
                    ops::moe_experts_gguf_decode_missing(state, t_m, t_none, banks, nullptr);
                    ops::moe_experts_gguf_decode_finish(
                        state, t_first, &t_none, t_w, banks, &banks,
                        static_cast<const float*>(d_products.p),
                        static_cast<const std::int32_t*>(d_pairs.p),
                        static_cast<const std::int32_t*>(d_count.p), t_y, nullptr);
                }
                check(cudaDeviceSynchronize(), "the decode form with CPU products");
                std::vector<float> hybrid(got.size());
                d_y.copy_to_host(hybrid.data(), hybrid.size() * 4);
                std::size_t differ = 0;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    differ += std::memcmp(&hybrid[i], &got[i], 4) != 0 ? 1 : 0;
                }
                std::cout << (differ == 0 ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                          << " with " << count << " CPU products ("
                          << ops::gguf_cpu_backend_name(ops::GgufCpuBackend::Automatic)
                          << "): " << differ << " of " << got.size() << " outputs differ from the GPU's\n";
                failures += differ == 0 ? 0 : 1;
            }
        }
        double num = 0, den = 0, worst = 0, scale = 0;
        for (int t = 0; t < tokens; ++t) {
            std::vector<float> x(kHidden);
            for (int i = 0; i < kHidden; ++i) {
                x[i] = __bfloat162float(m[std::size_t(t) * kHidden + i]);
            }
            std::vector<double> want(kHidden, 0.0);
            // Experts sharing a matrix share its output for this token.
            std::vector<std::vector<double>> outputs(c.distinct);
            for (int j = 0; j < kTopK; ++j) {
                const int e = ids[t * kTopK + j];
                auto& o     = outputs[e % c.distinct];
                if (o.empty()) {
                    o = expert_output(gate.rows_of(e), up.rows_of(e), down.rows_of(e), x);
                }
                for (int r = 0; r < kHidden; ++r) { want[r] += weights[t * kTopK + j] * o[r]; }
            }
            const auto s = expert_output(sgate.rows_of(0), sup.rows_of(0), sdown.rows_of(0), x);
            for (int r = 0; r < kHidden; ++r) { want[r] += shared[t] * s[r]; }
            for (int r = 0; r < kHidden; ++r) {
                const double d = double(got[std::size_t(t) * kHidden + r]) - want[r];
                num += d * d;
                den += want[r] * want[r];
                worst = std::max(worst, std::abs(d));
                scale = std::max(scale, std::abs(want[r]));
            }
        }
        const double rms = std::sqrt(num / std::max(den, 1e-30)),
                     max = worst / std::max(scale, 1e-30);
        // Two q8_1 activations (the token and the middle) per product chain.
        const bool ok = std::isfinite(rms) && rms < 4e-2 && max < 1.6e-1;
        std::cout << (ok ? "OK   " : "FAIL ") << c.name << " T=" << tokens
                  << (tokens > 8 ? (device_resident ? " matrix" : " vector") : "") << " rms=" << rms
                  << " max=" << max << '\n';
        failures += ok ? 0 : 1;
    }
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        using Q                       = QType;
        const std::vector<Case> cases = {
            // The GSQ-RCO Q2_0 release: fused Q2_0 gate/up, Q2_0 down, a mixed shared expert.
            {"q2_0 bank",
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_IQ4_XS,
             Q::GGUF_Q3_K,
             Q::GGUF_Q5_0,
             8,
             {1, 2, 3, 8, 9, 16, 24, 40, 48},
             kExperts},
            // Experts concentrated on a few ids: several columns per pass (chunks 4 and 8).
            {"q2_0 dense routing",
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q2_0,
             Q::GGUF_Q4_K,
             Q::GGUF_Q4_K,
             Q::GGUF_Q4_0,
             2,
             {160, 260},
             12},
            // Gate and up in different types take the unfused path.
            {"iq2_s/iq2_xxs unfused",
             Q::GGUF_IQ2_S,
             Q::GGUF_IQ2_XXS,
             Q::GGUF_IQ4_NL,
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ3_S,
             Q::GGUF_Q8_0,
             4,
             {1, 5, 24},
             kExperts},
            // IQ2_S's two half-slice factors on the decode form, against the CPU's products.
            {"iq2_s bank",
             Q::GGUF_IQ2_S,
             Q::GGUF_IQ2_S,
             Q::GGUF_Q8_0,
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_NL,
             4,
             {1, 2, 8},
             kExperts},
            {"iq1_m/iq3_xxs",
             Q::GGUF_IQ1_M,
             Q::GGUF_IQ1_M,
             Q::GGUF_Q2_0,
             Q::GGUF_IQ3_XXS,
             Q::GGUF_IQ3_XXS,
             Q::GGUF_IQ4_NL,
             4,
             {1, 4, 16},
             kExperts},
            // The GSQ-RCO IQ3_S release's routed types on the matrix path: its IQ3_S layers, and
            // routes concentrated on a few experts (several columns per expert) in the IQ3_XXS
            // and IQ4_XS layers, with both of its down types.
            {"iq3_s layer",
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ3_S,
             Q::GGUF_IQ4_NL,
             Q::GGUF_Q6_K,
             Q::GGUF_Q6_K,
             Q::GGUF_IQ4_NL,
             4,
             {1, 8, 9, 24},
             kExperts},
            {"iq3_xxs dense routing",
             Q::GGUF_IQ3_XXS,
             Q::GGUF_IQ3_XXS,
             Q::GGUF_Q2_0,
             Q::GGUF_Q4_K,
             Q::GGUF_Q4_K,
             Q::GGUF_IQ4_NL,
             2,
             {3, 24, 160},
             12},
            {"iq4_xs dense routing",
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_XS,
             Q::GGUF_IQ4_NL,
             Q::GGUF_Q5_K,
             Q::GGUF_Q5_K,
             Q::GGUF_IQ4_NL,
             2,
             {8, 24},
             12},
        };
        int failures       = 0;
        std::uint32_t seed = 4242;
        for (const auto& c : cases) { failures += run_case(c, seed++); }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " MoE experts GGUF\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "MoE experts GGUF: " << error.what() << '\n';
        return 1;
    }
}
