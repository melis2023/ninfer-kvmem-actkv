// moe_experts_bf16 against an FP64 oracle at the model's shapes: full 512-expert banks on the
// device (only the experts the routes name hold data), decode and verify widths, repeated experts
// across tokens.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/moe_experts.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden = 2560, kWidth = 640, kExperts = 512, kTop = 10;

std::vector<std::uint16_t> encode(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

int run(int tokens, std::uint32_t seed) {
    std::mt19937 random(seed);
    std::vector<int> ids(static_cast<std::size_t>(kTop) * tokens);
    std::vector<float> weights(ids.size()), shared(tokens), m(static_cast<std::size_t>(kHidden) * tokens);
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> chosen;
        while (chosen.size() < kTop) {
            const int e = static_cast<int>(random() % 24) * 21; // a small pool, so tokens share experts
            if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
        }
        float sum = 0;
        for (int k = 0; k < kTop; ++k) {
            ids[static_cast<std::size_t>(t) * kTop + k] = chosen[k];
            sum += (weights[static_cast<std::size_t>(t) * kTop + k] = 0.1f + (random() % 1000) / 1000.0f);
        }
        for (int k = 0; k < kTop; ++k) weights[static_cast<std::size_t>(t) * kTop + k] /= sum;
        shared[t] = (random() % 1000) / 1000.0f;
    }
    fill_uniform(m, seed + 1, -2.0f, 2.0f);
    round_to_bf16(m);
    std::map<int, std::pair<std::vector<float>, std::vector<float>>> experts; // gate_up, down
    for (int e : ids) {
        if (experts.count(e)) continue;
        std::vector<float> gu(static_cast<std::size_t>(2) * kWidth * kHidden), dn(static_cast<std::size_t>(kHidden) * kWidth);
        fill_uniform(gu, seed + 100 + e, -0.04f, 0.04f);
        fill_uniform(dn, seed + 2000 + e, -0.06f, 0.06f);
        round_to_bf16(gu);
        round_to_bf16(dn);
        experts.emplace(e, std::make_pair(std::move(gu), std::move(dn)));
    }
    std::vector<float> sgu(static_cast<std::size_t>(2) * kWidth * kHidden), sdn(static_cast<std::size_t>(kHidden) * kWidth);
    fill_uniform(sgu, seed + 3, -0.04f, 0.04f);
    fill_uniform(sdn, seed + 4, -0.06f, 0.06f);
    round_to_bf16(sgu);
    round_to_bf16(sdn);

    const auto expert = [&](const std::vector<float>& gu, const std::vector<float>& dn, int t) {
        std::vector<double> h(kWidth), out(kHidden);
        for (int i = 0; i < kWidth; ++i) {
            double g = 0, u = 0;
            for (int k = 0; k < kHidden; ++k) {
                const double x = m[static_cast<std::size_t>(t) * kHidden + k];
                g += double(gu[static_cast<std::size_t>(i) * kHidden + k]) * x;
                u += double(gu[static_cast<std::size_t>(kWidth + i) * kHidden + k]) * x;
            }
            h[i] = g / (1.0 + std::exp(-g)) * u;
        }
        for (int d = 0; d < kHidden; ++d) {
            double s = 0;
            for (int i = 0; i < kWidth; ++i) s += double(dn[static_cast<std::size_t>(d) * kWidth + i]) * h[i];
            out[d] = s;
        }
        return out;
    };
    std::vector<double> expected(static_cast<std::size_t>(kHidden) * tokens, 0.0);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < kTop; ++k) {
            const auto& [gu, dn] = experts.at(ids[static_cast<std::size_t>(t) * kTop + k]);
            const auto out = expert(gu, dn, t);
            for (int d = 0; d < kHidden; ++d) expected[static_cast<std::size_t>(t) * kHidden + d] += weights[static_cast<std::size_t>(t) * kTop + k] * out[d];
        }
        const auto out = expert(sgu, sdn, t);
        for (int d = 0; d < kHidden; ++d) expected[static_cast<std::size_t>(t) * kHidden + d] += shared[t] * out[d];
    }

    const std::size_t expert_gu = static_cast<std::size_t>(2) * kWidth * kHidden, expert_dn = static_cast<std::size_t>(kHidden) * kWidth;
    DeviceBuffer d_gu(expert_gu * kExperts * 2), d_dn(expert_dn * kExperts * 2);
    CUDA_CHECK(cudaMemset(d_gu.p, 0, d_gu.bytes));
    CUDA_CHECK(cudaMemset(d_dn.p, 0, d_dn.bytes));
    for (const auto& [e, banks] : experts) {
        const auto gb = encode(banks.first), db = encode(banks.second);
        CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(d_gu.p) + e * expert_gu, gb.data(), expert_gu * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(d_dn.p) + e * expert_dn, db.data(), expert_dn * 2, cudaMemcpyHostToDevice));
    }
    GuardedDeviceBuffer d_m(m.size() * 2), d_ids(ids.size() * 4), d_w(weights.size() * 4), d_shared(shared.size() * 4),
        d_sgu(sgu.size() * 2), d_sdn(sdn.size() * 2), d_y(static_cast<std::size_t>(kHidden) * tokens * 4);
    const auto mb = encode(m), sgb = encode(sgu), sdb = encode(sdn);
    d_m.copy_from_host(mb.data(), d_m.bytes());
    d_ids.copy_from_host(ids.data(), d_ids.bytes());
    d_w.copy_from_host(weights.data(), d_w.bytes());
    d_shared.copy_from_host(shared.data(), d_shared.bytes());
    d_sgu.copy_from_host(sgb.data(), d_sgu.bytes());
    d_sdn.copy_from_host(sdb.data(), d_sdn.bytes());
    Tensor t_m(d_m.data(), DType::BF16, {kHidden, tokens});
    Tensor t_ids(d_ids.data(), DType::I32, {kTop, tokens});
    Tensor t_w(d_w.data(), DType::FP32, {kTop, tokens});
    Tensor t_shared(d_shared.data(), DType::FP32, {tokens});
    Tensor t_gu(d_gu.p, DType::BF16, {kHidden, 2 * kWidth, kExperts});
    Tensor t_dn(d_dn.p, DType::BF16, {kWidth, kHidden, kExperts});
    Tensor t_sgu(d_sgu.data(), DType::BF16, {kHidden, 2 * kWidth});
    Tensor t_sdn(d_sdn.data(), DType::BF16, {kWidth, kHidden});
    Tensor t_y(d_y.data(), DType::FP32, {kHidden, tokens});
    WorkspaceArena workspace(ops::moe_experts_bf16_workspace_bytes(tokens));
    ops::moe_experts_bf16(t_m, t_ids, t_w, t_shared, t_gu, t_dn, t_sgu, t_sdn, workspace, t_y, nullptr);
    cuda_synchronize();
    const auto got = from_device<float>(d_y.data(), expected.size());
    const std::string label = "moe_experts_bf16 T=" + std::to_string(tokens);
    int failures = verify_reduction(label, std::vector<double>(got.begin(), got.end()), expected, {2e-5, 1e-5, 1e-4});
    for (auto* buffer : {&d_m, &d_ids, &d_w, &d_shared, &d_sgu, &d_sdn, &d_y}) failures += buffer->verify_guards(label.c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run(1, 9300u);
    failures += run(5, 9301u);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " moe_experts_bf16\n";
    return failures == 0 ? 0 : 1;
}
