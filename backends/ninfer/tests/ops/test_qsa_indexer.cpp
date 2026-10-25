// qsa_indexer_append/select against an FP64 oracle of Qwen3.8-Flash-Next's block indexer: a
// sequence long enough for the selection to bind (650 blocks), fed as a prefill chunk that starts
// mid-block, decode steps across block boundaries and a verify-width chunk; the pooled keys
// pointwise and every selection as a set, up to near-ties of the 512th score. The decode steps
// replay one captured CUDA graph, the position advancing only in the device word the ops read.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim = 128, kHeads = 4, kProjection = 640, kTop = 512;
constexpr float kEps = 1e-6f;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> norm_rope(std::vector<double> x, const std::vector<float>& w, int position) {
    double sum = 0;
    for (double v : x) sum += v * v;
    const double scale = 1.0 / std::sqrt(sum / kDim + double(kEps));
    for (int d = 0; d < kDim; ++d) x[d] = x[d] * scale * (1.0 + w[d]);
    for (int i = 0; i < 32; ++i) {
        const float frequency = static_cast<float>(std::exp(-(2.0 * i / 64.0) * std::log(1e7)));
        const float angle     = static_cast<float>(position) * frequency;
        const double c = std::cos(double(angle)), s = std::sin(double(angle));
        const double x1 = x[i], x2 = x[i + 32];
        x[i]      = x1 * c - x2 * s;
        x[i + 32] = x2 * c + x1 * s;
    }
    return x;
}

struct Sequence {
    int length;
    std::vector<float> projection; // [length][640]
    std::vector<float> query_norm, key_norm;
};

std::vector<double> pooled_oracle(const Sequence& seq, int block) {
    std::vector<double> mean(kDim, 0.0);
    for (int i = 0; i < 4; ++i) {
        for (int d = 0; d < kDim; ++d) {
            mean[d] += seq.projection[static_cast<std::size_t>(block * 4 + i) * kProjection + kHeads * kDim + d];
        }
    }
    for (double& v : mean) v /= 4.0;
    return norm_rope(mean, seq.key_norm, block * 4);
}

int check_selection(const std::string& label, const Sequence& seq, const std::vector<std::vector<double>>& pooled,
                    int position, const int* got, int got_count) {
    const int blocks = (position + 1) / 4;
    std::vector<int> expected;
    std::vector<double> score(blocks);
    if (blocks <= kTop) {
        expected.resize(blocks);
        std::iota(expected.begin(), expected.end(), 0);
    } else {
        std::vector<std::vector<double>> q(kHeads);
        for (int h = 0; h < kHeads; ++h) {
            std::vector<double> raw(kDim);
            for (int d = 0; d < kDim; ++d) raw[d] = seq.projection[static_cast<std::size_t>(position) * kProjection + h * kDim + d];
            q[h] = norm_rope(raw, seq.query_norm, position);
        }
        for (int b = 0; b < blocks; ++b) {
            double s = 0;
            for (int h = 0; h < kHeads; ++h) {
                double dot = 0;
                for (int d = 0; d < kDim; ++d) dot += q[h][d] * pooled[b][d];
                s += std::max(dot, 0.0);
            }
            score[b] = s;
        }
        std::vector<int> order(blocks);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
        expected.assign(order.begin(), order.begin() + kTop);
        std::sort(expected.begin(), expected.end());
    }
    std::vector<int> actual(got, got + got_count);
    if (!std::is_sorted(actual.begin(), actual.end())) {
        std::cerr << label << ": selection not in increasing block order\n";
        return 1;
    }
    if (static_cast<int>(actual.size()) != static_cast<int>(expected.size())) {
        std::cerr << label << ": selected " << actual.size() << " blocks, expected " << expected.size() << '\n';
        return 1;
    }
    if (actual == expected) return 0;
    // Only blocks tied with the 512th score within 1e-5 relative may differ.
    std::vector<double> sorted(score);
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const double edge = sorted[kTop - 1];
    std::set<int> a(actual.begin(), actual.end()), e(expected.begin(), expected.end());
    for (int b = 0; b < blocks; ++b) {
        if (a.count(b) != e.count(b) && std::abs(score[b] - edge) > 1e-5 * std::abs(edge)) {
            std::cerr << label << ": block " << b << " (score " << score[b] << ", 512th " << edge
                      << ") selected=" << a.count(b) << " expected=" << e.count(b) << '\n';
            return 1;
        }
    }
    return 0;
}

int run(std::uint32_t seed) {
    Sequence seq{2600, std::vector<float>(static_cast<std::size_t>(2600) * kProjection), std::vector<float>(kDim), std::vector<float>(kDim)};
    fill_uniform(seq.projection, seed, -2.0f, 2.0f);
    fill_uniform(seq.query_norm, seed + 1, -0.5f, 0.5f);
    fill_uniform(seq.key_norm, seed + 2, -0.5f, 0.5f);
    round_to_bf16(seq.projection);
    round_to_bf16(seq.query_norm);
    round_to_bf16(seq.key_norm);
    std::vector<std::vector<double>> pooled(seq.length / 4);
    for (int b = 0; b < seq.length / 4; ++b) pooled[b] = pooled_oracle(seq, b);

    const int capacity = 700;
    GuardedDeviceBuffer d_projection(seq.projection.size() * 2), d_qn(kDim * 2), d_kn(kDim * 2),
        d_pooled(static_cast<std::size_t>(capacity) * kDim * 4), d_tail(3 * kDim * 4),
        d_selected(kProjection * 2 + static_cast<std::size_t>(kTop) * 64 * 4), d_counts(64 * 4);
    const auto bits = encode_bf16(seq.projection);
    d_projection.copy_from_host(bits.data(), d_projection.bytes());
    const auto qn = encode_bf16(seq.query_norm), kn = encode_bf16(seq.key_norm);
    d_qn.copy_from_host(qn.data(), d_qn.bytes());
    d_kn.copy_from_host(kn.data(), d_kn.bytes());
    CUDA_CHECK(cudaMemset(d_pooled.data(), 0, d_pooled.bytes()));
    CUDA_CHECK(cudaMemset(d_tail.data(), 0, d_tail.bytes()));
    Tensor t_qn(d_qn.data(), DType::BF16, {kDim}), t_kn(d_kn.data(), DType::BF16, {kDim});
    Tensor t_pooled(d_pooled.data(), DType::FP32, {kDim, capacity});
    Tensor t_tail(d_tail.data(), DType::FP32, {kDim, 3});
    GuardedDeviceBuffer d_first(4);
    const Tensor t_first(d_first.data(), DType::I32, {1});
    const ops::QsaIndexerWeights weights{&t_qn, &t_kn};
    WorkspaceArena workspace(ops::qsa_indexer_select_workspace_bytes(64, capacity));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraphExec_t decode = nullptr;

    int failures = 0;
    // Chunks: a prefill that ends mid-block, decode steps across block edges, a verify width, the
    // rest as chunks that start and end mid-block.
    std::vector<int> chunks{2050, 1, 1, 1, 1, 1, 9, 37, 64, 64, 64, 64, 64, 64, 64, 64, 2};
    int begin = 0;
    for (const int count : chunks) {
        if (begin + count > seq.length) break;
        // Decode steps read their projection column from one fixed place, as a graph replay does.
        auto* column = static_cast<std::uint16_t*>(d_projection.data()) + static_cast<std::size_t>(begin) * kProjection;
        if (count == 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_selected.data(), column, kProjection * 2, cudaMemcpyDeviceToDevice, stream));
            column = static_cast<std::uint16_t*>(d_selected.data());
        }
        CUDA_CHECK(cudaMemcpyAsync(d_first.data(), &begin, 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        Tensor t_projection(column, DType::BF16, {kProjection, count});
        Tensor t_selected(static_cast<std::byte*>(d_selected.data()) + (count == 1 ? kProjection * 2 : 0), DType::I32, {kTop, count});
        Tensor t_counts(d_counts.data(), DType::I32, {count});
        const auto step = [&] {
            ops::qsa_indexer_append(t_projection, t_first, weights, kEps, t_pooled, t_tail, stream);
            if (count <= 64) {
                ops::qsa_indexer_select(t_projection, t_first, weights, kEps, t_pooled, workspace, t_selected, t_counts, stream);
            }
        };
        if (count == 1) {
            if (decode == nullptr) {
                cudaGraph_t graph = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                step();
                CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&decode, graph, 0));
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
            CUDA_CHECK(cudaGraphLaunch(decode, stream));
        } else {
            step();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (count <= 64) {
            const std::byte* selected_base = static_cast<const std::byte*>(t_selected.data);
            const auto selected = from_device<int>(selected_base, static_cast<std::size_t>(kTop) * count);
            const auto counts   = from_device<int>(d_counts.data(), count);
            for (int t = 0; t < count; ++t) {
                failures += check_selection("qsa select p=" + std::to_string(begin + t), seq, pooled, begin + t,
                                            selected.data() + static_cast<std::size_t>(t) * kTop, counts[t]);
            }
        }
        begin += count;
    }
    if (decode != nullptr) CUDA_CHECK(cudaGraphExecDestroy(decode));
    CUDA_CHECK(cudaStreamDestroy(stream));
    const int complete = begin / 4;
    const auto got = from_device<float>(d_pooled.data(), static_cast<std::size_t>(complete) * kDim);
    std::vector<double> expected;
    for (int b = 0; b < complete; ++b) expected.insert(expected.end(), pooled[b].begin(), pooled[b].end());
    failures += verify_pointwise("qsa pooled keys", std::vector<double>(got.begin(), got.end()), expected, {2e-5, 2e-5});
    for (auto* buffer : {&d_projection, &d_qn, &d_kn, &d_pooled, &d_tail, &d_selected, &d_counts, &d_first}) {
        failures += buffer->verify_guards("qsa_indexer");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const int failures = run(8100u);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " qsa_indexer\n";
    return failures == 0 ? 0 : 1;
}
