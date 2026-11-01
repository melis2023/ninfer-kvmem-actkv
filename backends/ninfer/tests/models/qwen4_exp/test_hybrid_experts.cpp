#include "models/qwen4_exp/hybrid_experts.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen4_exp;
constexpr int H = 2560, I = 640, E = 16, TOP = 10;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

struct Projection {
    int bits, n, k;
    bool panel;
    std::vector<std::uint8_t> low, high;
    std::vector<std::uint16_t> scales;
    Weight weight;

    Projection(int b, int rows, int columns, bool panels, unsigned seed)
        : bits(b), n(rows), k(columns), panel(panels), low(std::size_t(n) * k / (b == 2 ? 4 : 2)),
          high(b == 5 ? std::size_t(n) * k / 8 : 0), scales(std::size_t(n) * k / 64) {
        for (int r = 0; r < n; ++r) for (int g = 0; g < k / 64; ++g) {
            // Binary16 1/32 (Q2) or 1/256 (Q4/Q5), including negative scales.
            scales[std::size_t(r) * (k / 64) + g] =
                std::uint16_t((b == 2 ? 0x2800 : 0x1c00) | ((r + g) % 3 == 0 ? 0x8000 : 0));
            const auto record = panel ? (std::size_t(r / 64) * (k / 64) + g) * 64 + r % 64
                                      : std::size_t(r) * (k / 64) + g;
            for (int c = 0; c < 64; ++c) {
                seed = 1664525U * seed + 1013904223U;
                const auto code = (seed >> 21) & ((1U << b) - 1);
                if (b == 2) { low[record * 16 + c / 4] |= std::uint8_t(code << (2 * (c % 4))); }
                else {
                    low[record * 32 + c / 2] |= std::uint8_t((code & 15) << (4 * (c % 2)));
                    if (b == 5) { high[record * 8 + c / 8] |= std::uint8_t((code >> 4) << (c % 8)); }
                }
            }
        }
        weight.qtype = b == 2 ? QType::Q2_G64_FP16 : b == 4 ? QType::Q4_G64_FP16 : QType::Q5_G64_FP16;
        weight.qdata = low.data(); weight.qhigh = high.empty() ? nullptr : high.data();
        weight.scales = scales.data(); weight.group = weight.group_size = 64;
        weight.n = weight.shape[0] = weight.padded_shape[0] = n;
        weight.k = weight.shape[1] = weight.padded_shape[1] = k;
        weight.ndim = 2;
        weight.scale_dtype = DType::FP16;
        weight.layout = panel ? QuantLayout::RowSplitPanel : QuantLayout::RowSplit;
    }

    // Decode independently from actual stored bytes; no CPU/GPU production decoder is called.
    double at(int row, int column) const {
        const int group = column / 64, lane = column % 64;
        const auto record = panel ? (std::size_t(row / 64) * (k / 64) + group) * 64 + row % 64
                                  : std::size_t(row) * (k / 64) + group;
        unsigned code = bits == 2 ? (low[record * 16 + lane / 4] >> (2 * (lane % 4))) & 3U
                                  : (low[record * 32 + lane / 2] >> (4 * (lane % 2))) & 15U;
        if (bits == 5) { code |= ((high[record * 8 + lane / 8] >> (lane % 8)) & 1U) << 4; }
        const int value = bits == 2 ? int(code) - 1 :
            int(code) - ((code & (1U << (bits - 1))) ? (1 << bits) : 0);
        const auto scale = scales[std::size_t(row) * (k / 64) + group];
        const double magnitude = std::ldexp(1.0 + double(scale & 1023) / 1024, int((scale >> 10) & 31) - 15);
        return value * ((scale & 0x8000) ? -magnitude : magnitude);
    }
    std::size_t bytes() const { return low.size() + high.size() + scales.size() * 2; }
};

std::vector<double> oracle(const Projection& gate, const Projection& up, const Projection& down,
                           std::span<const std::uint16_t> input) {
    std::vector<double> middle(I), out(H);
    for (int row = 0; row < I; ++row) {
        double g = 0, u = 0;
        for (int c = 0; c < H; ++c) {
            const auto x = std::bit_cast<float>(std::uint32_t(input[c]) << 16);
            g += gate.at(row, c) * x;
            u += up.at(row, c) * x;
        }
        middle[row] = g / (1 + std::exp(-g)) * u;
    }
    for (int row = 0; row < H; ++row) {
        for (int c = 0; c < I; ++c) { out[row] += down.at(row, c) * middle[c]; }
    }
    return out;
}

void run(DeviceContext& device, std::array<int, 3> bits, bool a8, bool panel,
         unsigned cached, float share, bool adaptive) {
    Projection gate(bits[0], I, H, panel, 17), up(bits[1], I, H, panel, 29),
               down(bits[2], H, I, panel, 41);
    Projection gate_b(bits[0], I, H, panel, 59), up_b(bits[1], I, H, panel, 73),
               down_b(bits[2], H, I, panel, 89);
    std::vector<std::unique_ptr<PinnedHostBuffer>> pinned;
    const bool direct = share == 0.5F;
    const auto memory = [&](const Projection& source) {
        Weight weight = source.weight;
        if (!direct) { return weight; }
        auto allocation = std::make_unique<PinnedHostBuffer>(source.bytes());
        auto* bytes = static_cast<std::byte*>(allocation->data());
        std::memcpy(bytes, source.low.data(), source.low.size());
        weight.qdata = bytes;
        bytes += source.low.size();
        if (!source.high.empty()) {
            std::memcpy(bytes, source.high.data(), source.high.size());
            weight.qhigh = bytes;
            bytes += source.high.size();
        }
        std::memcpy(bytes, source.scales.data(), source.scales.size() * 2);
        weight.scales = bytes;
        pinned.push_back(std::move(allocation));
        return weight;
    };
    const ops::CpuExpertWeights operands{{memory(gate), a8}, {memory(up), a8}, {memory(down), a8}};
    const ops::CpuExpertWeights other{{memory(gate_b), a8}, {memory(up_b), a8}, {memory(down_b), a8}};
    HybridExpertLayer layer;
    for (int e = 0; e < E; ++e) { layer.experts.push_back(e % 2 ? other : operands); }
    layer.registered = direct;
    if (direct) {
        layer.route_counts.resize(E);
        for (int e = 6; e < 10; ++e) { layer.route_counts[e] = 100 + 9 - e; }
    }
    const auto slot = gate.bytes() + up.bytes() + down.bytes(); // all planes are 256-byte multiples
    std::atomic<bool> stall{false}, entered{false}, cancelled{false};
    HybridExperts hybrid(device, {layer}, 17, cached * slot,
                          {.dma_share = share, .cpu_threads = 3, .adaptive_cache = adaptive}, [&] {
        if (!stall.load()) { return; }
        entered.store(true);
        while (!cancelled.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    });
    require(hybrid.cache_bytes(0) == cached * slot, "hybrid cache ignored its byte budget");

    // A zero BF16 shared expert still exercises the actual shared GPU route and merge.
    DeviceBuffer zero(H * I * 2); zero.fill(0);
    Weight shared;
    shared.qdata = zero.p; shared.qtype = QType::BF16; shared.layout = QuantLayout::Contiguous;
    shared.n = I; shared.k = H;
    std::array<Weight, 3> shared_weights{shared, shared, shared};
    shared_weights[2].n = H; shared_weights[2].k = I;
    DeviceBuffer shared_table(sizeof(shared_weights));
    shared_table.copy_from_host(shared_weights.data(), shared_table.bytes);
    const auto* table = static_cast<const Weight*>(shared_table.p);
    const ops::NativeMoeWeights banks{
        {gate.weight.qtype, table, a8}, {up.weight.qtype, table, a8}, {down.weight.qtype, table, a8},
        {QType::BF16, table, false}, {QType::BF16, table + 1, false}, {QType::BF16, table + 2, false}, E};
    WorkspaceArena workspace(ops::moe_experts_native_workspace_bytes(17));
    std::array<std::vector<std::uint16_t>, 2> input;
    std::array<std::array<std::vector<double>, 2>, 2> expected_expert;
    for (int variant = 0; variant < 2; ++variant) {
        input[variant].resize(H);
        for (int h = 0; h < H; ++h) {
            input[variant][h] = test::f32_to_bf16(std::sin(float(h * 13 + variant * 7)) * 0.6F);
        }
        expected_expert[variant][0] = oracle(gate, up, down, input[variant]);
        expected_expert[variant][1] = oracle(gate_b, up_b, down_b, input[variant]);
    }
    const auto initial_admissions = hybrid.stats().admitted;
    std::uint64_t expected_routes = 0;
    for (int tokens : {1, 3, 16, 17}) {
        std::vector<std::uint16_t> x(std::size_t(tokens) * H);
        std::vector<std::int32_t> ids(tokens * TOP);
        std::vector<float> weights(tokens * TOP), shared_gate(tokens, 0.5F);
        std::vector<double> expected(x.size());
        for (int t = 0; t < tokens; ++t) {
            std::copy(input[t % 2].begin(), input[t % 2].end(), x.begin() + std::size_t(t) * H);
            for (int k = 0; k < TOP; ++k) {
                ids[t * TOP + k] = (t + k + 3) % E;
                weights[t * TOP + k] = float(k + 1) / 55;
                for (int h = 0; h < H; ++h) {
                    expected[std::size_t(t) * H + h] += weights[t * TOP + k] *
                        expected_expert[t % 2][ids[t * TOP + k] % 2][h];
                }
            }
        }
        DeviceBuffer dx(x.size() * 2), di(ids.size() * 4), dw(weights.size() * 4), ds(tokens * 4), dy(x.size() * 4);
        dx.copy_from_host(x.data(), dx.bytes); di.copy_from_host(ids.data(), di.bytes);
        dw.copy_from_host(weights.data(), dw.bytes); ds.copy_from_host(shared_gate.data(), ds.bytes);
        const Tensor tx(dx.p, DType::BF16, {H, tokens}), ti(di.p, DType::I32, {TOP, tokens}),
                     tw(dw.p, DType::FP32, {TOP, tokens}), ts(ds.p, DType::FP32, {tokens});
        Tensor ty(dy.p, DType::FP32, {H, tokens});
        std::vector<float> first(x.size()), actual(x.size());
        DecodeGraphExecutable graph;
        for (int repeat = 0; repeat < 3; ++repeat) {
            hybrid.begin(nullptr, true);
            const auto call = [&] { hybrid.run(0, tx, ti, tw, ts, banks, tokens <= 16, workspace, ty); };
            if (repeat == 0) { call(); }
            else {
                if (!graph.ready()) {
                    DecodeGraphDefinition definition;
                    definition.capture(device.stream, [&] {
                        call();
                        // Two exchanges in one graph expose premature staging reuse and waits
                        // that accidentally synchronize the entire graph from its CPU worker.
                        if (tokens == 1) { call(); }
                    });
                    graph.instantiate(definition);
                }
                graph.launch(device.stream);
            }
            hybrid.finish();
            if (direct && tokens == 1 && repeat == 0) {
                require(hybrid.stats().hits == 4, "initial cache ignored the routing profile");
            }
            expected_routes += TOP * tokens * ((repeat && tokens == 1) ? 2 : 1);
            dy.copy_to_host(actual.data(), dy.bytes);
            double error = 0, norm = 0, peak = 0, max_error = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double delta = actual[i] - expected[i];
                error += delta * delta; norm += expected[i] * expected[i];
                peak = std::max(peak, std::abs(expected[i])); max_error = std::max(max_error, std::abs(delta));
            }
            const double relative = std::sqrt(error / norm), gross = max_error / peak;
            require(std::isfinite(relative) && relative < (a8 ? 0.04 : 0.001), "hybrid FP64 relative error");
            require(gross < (a8 ? 0.16 : 0.004), "hybrid FP64 pointwise error");
            if (repeat == 0) { first = actual; }
            else if (!adaptive) {
                require(std::memcmp(first.data(), actual.data(), dy.bytes) == 0, "fixed hybrid partition did not repeat");
            }
            std::cout << "hybrid bits=" << bits[0] << '/' << bits[1] << '/' << bits[2]
                      << " a8=" << a8 << " cached=" << cached << " dma=" << share
                      << " adaptive=" << adaptive << " T=" << tokens << " relative_l2=" << relative << '\n';
        }
        if (tokens == 1 && cached == 0 && share == 0 && bits[0] == 2) {
            stall.store(true);
            hybrid.begin(&cancelled, false);
            graph.launch(device.stream);
            std::thread cancel_worker([&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!entered.load() && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                cancelled.store(true);
            });
            bool rejected = false;
            try { hybrid.finish(); } catch (const ops::CpuExpertCancelled&) { rejected = true; }
            catch (...) { cancelled.store(true); cancel_worker.join(); throw; }
            cancel_worker.join();
            require(entered.load() && rejected, "captured hybrid did not cancel its stalled CPU worker");
            stall.store(false); cancelled.store(false);
            hybrid.begin(&cancelled, false);
            graph.launch(device.stream);
            hybrid.finish();
            dy.copy_to_host(actual.data(), dy.bytes);
            require(std::memcmp(first.data(), actual.data(), dy.bytes) == 0,
                    "hybrid graph retained incomplete work after cancellation");
            std::cout << "HYBRID_STALLED_WORKER_CANCEL_PASS\n";
        }
    }
    require(hybrid.stats().routes == expected_routes, "hybrid route accounting differs");
    require(hybrid.stats().routes == hybrid.stats().hits + hybrid.stats().cpu_routes + hybrid.stats().dma_routes,
            "hybrid route partition does not cover every routed expert");
    require((share < 1) == (hybrid.stats().cpu_routes > 0), "hybrid did not execute its selected CPU share");
    require(hybrid.stats().dma_routes > 0, "hybrid did not stream uncached prefill experts");
    if (cached) { require(hybrid.stats().hits > 0, "hybrid never read its resident cache"); }
    if (adaptive) { require(hybrid.stats().admitted > initial_admissions, "hybrid adaptive cache never admitted a miss"); }
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    std::cout << std::unitbuf;
    try {
        DeviceContext device;
        run(device, {2, 2, 2}, true, false, 0, 0, false);
        run(device, {2, 5, 4}, false, true, 4, 0.5F, false);
        run(device, {5, 4, 5}, true, false, 0, 1, false);
        run(device, {4, 4, 4}, true, true, 4, 0, true);
        std::cout << "HYBRID_EXPERT_ORACLE_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
