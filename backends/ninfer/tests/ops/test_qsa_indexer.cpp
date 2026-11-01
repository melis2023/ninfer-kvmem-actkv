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

// YaRN over a native window (factor above one), as Hugging Face's `yarn` rope type gives it with
// beta_fast 32 and beta_slow 1; factor one leaves the rotation unscaled.
struct Yarn {
    double factor = 1.0;
    double native = 0.0;
    [[nodiscard]] ops::RopeYarn op() const {
        return {static_cast<float>(factor), static_cast<std::uint32_t>(native)};
    }
};

// Pair i's frequency of 64 rotated dims at theta 1e7, rounded to FP32 as the contract states.
float pair_frequency(int i, const Yarn& yarn) {
    if (!(yarn.factor > 1.0)) return static_cast<float>(std::exp(-(2.0 * i / 64.0) * std::log(1e7)));
    constexpr double kPi = 3.14159265358979323846;
    const auto correction = [&](double rotations) {
        return 64.0 * std::log(yarn.native / (rotations * 2.0 * kPi)) / (2.0 * std::log(1e7));
    };
    const double lo = std::max(std::floor(correction(32.0)), 0.0);
    double hi       = std::min(std::ceil(correction(1.0)), 63.0);
    if (hi == lo) hi += 0.001;
    const double ramp = std::clamp((i - lo) / (hi - lo), 0.0, 1.0);
    return static_cast<float>(std::pow(1e7, -2.0 * i / 64.0) * ((1.0 - ramp) + ramp / yarn.factor));
}

std::vector<double> norm_rope(std::vector<double> x, const std::vector<float>& w, int position,
                              const Yarn& yarn) {
    double sum = 0;
    for (double v : x) sum += v * v;
    const double scale = 1.0 / std::sqrt(sum / kDim + double(kEps));
    for (int d = 0; d < kDim; ++d) x[d] = x[d] * scale * (1.0 + w[d]);
    const double attention = yarn.factor > 1.0 ? 0.1 * std::log(yarn.factor) + 1.0 : 1.0;
    for (int i = 0; i < 32; ++i) {
        const float frequency = pair_frequency(i, yarn);
        const float angle     = static_cast<float>(position) * frequency;
        const double c = std::cos(double(angle)) * attention, s = std::sin(double(angle)) * attention;
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

std::vector<double> pooled_oracle(const Sequence& seq, int block, const Yarn& yarn) {
    std::vector<double> mean(kDim, 0.0);
    for (int i = 0; i < 4; ++i) {
        for (int d = 0; d < kDim; ++d) {
            mean[d] += seq.projection[static_cast<std::size_t>(block * 4 + i) * kProjection + kHeads * kDim + d];
        }
    }
    for (double& v : mean) v /= 4.0;
    return norm_rope(mean, seq.key_norm, block * 4, yarn);
}

int check_selection(const std::string& label, const Sequence& seq, const std::vector<std::vector<double>>& pooled,
                    int position, const int* got, int got_count, const Yarn& yarn, int column = -1) {
    const int blocks = std::min((position + 1) / 4, static_cast<int>(pooled.size()));
    std::vector<int> expected;
    std::vector<double> score(blocks);
    if (blocks <= kTop) {
        expected.resize(blocks);
        std::iota(expected.begin(), expected.end(), 0);
    } else {
        std::vector<std::vector<double>> q(kHeads);
        for (int h = 0; h < kHeads; ++h) {
            std::vector<double> raw(kDim);
            for (int d = 0; d < kDim; ++d) raw[d] = seq.projection[
                static_cast<std::size_t>(column < 0 ? position : column) * kProjection + h * kDim + d];
            q[h] = norm_rope(raw, seq.query_norm, position, yarn);
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
    if (got_count != static_cast<int>(expected.size())) {
        std::cerr << label << ": selected " << got_count << " blocks, expected " << expected.size() << '\n';
        return 1;
    }
    std::vector<int> actual(got, got + got_count);
    if (!std::is_sorted(actual.begin(), actual.end()) ||
        std::adjacent_find(actual.begin(), actual.end()) != actual.end() ||
        (!actual.empty() && (actual.front() < 0 || actual.back() >= blocks))) {
        std::cerr << label << ": selection not in increasing block order\n";
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

int run(std::uint32_t seed, const Yarn& yarn = {}) {
    Sequence seq{2600, std::vector<float>(static_cast<std::size_t>(2600) * kProjection), std::vector<float>(kDim), std::vector<float>(kDim)};
    fill_uniform(seq.projection, seed, -2.0f, 2.0f);
    fill_uniform(seq.query_norm, seed + 1, -0.5f, 0.5f);
    fill_uniform(seq.key_norm, seed + 2, -0.5f, 0.5f);
    round_to_bf16(seq.projection);
    round_to_bf16(seq.query_norm);
    round_to_bf16(seq.key_norm);
    std::vector<std::vector<double>> pooled(seq.length / 4);
    for (int b = 0; b < seq.length / 4; ++b) pooled[b] = pooled_oracle(seq, b, yarn);

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
            ops::qsa_indexer_append(t_projection, t_first, weights, kEps, yarn.op(), t_pooled, t_tail, stream);
            if (count <= 64) {
                ops::qsa_indexer_select(t_projection, t_first, weights, kEps, yarn.op(), t_pooled, workspace, t_selected, t_counts, stream);
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
                                            selected.data() + static_cast<std::size_t>(t) * kTop, counts[t], yarn);
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
    failures += verify_pointwise(yarn.factor > 1.0 ? "qsa pooled keys yarn" : "qsa pooled keys",
                                 std::vector<double>(got.begin(), got.end()), expected, {2e-5, 2e-5});
    for (auto* buffer : {&d_projection, &d_qn, &d_kn, &d_pooled, &d_tail, &d_selected, &d_counts, &d_first}) {
        failures += buffer->verify_guards("qsa_indexer");
    }
    return failures;
}

int selection_case(int tokens, int capacity, int first, bool ties = false, const Yarn& yarn = {}) {
    Sequence seq{tokens, std::vector<float>(std::size_t(tokens) * kProjection),
                 std::vector<float>(kDim), std::vector<float>(kDim)};
    fill_uniform(seq.projection, 9100 + tokens, -2.0f, 2.0f);
    fill_uniform(seq.query_norm, 9110, -0.5f, 0.5f);
    round_to_bf16(seq.projection);
    round_to_bf16(seq.query_norm);
    if (ties) { std::fill(seq.projection.begin(), seq.projection.end(), 0.0f); }
    std::vector<float> keys(std::size_t(capacity) * kDim);
    fill_uniform(keys, 9111, -2.0f, 2.0f);
    // Public pooled keys are FP32 inputs to select; the oracle uses those exact values.
    std::vector<std::vector<double>> pooled(capacity, std::vector<double>(kDim));
    for (int b = 0; b < capacity; ++b) {
        std::copy_n(keys.data() + std::size_t(b) * kDim, kDim, pooled[b].begin());
    }
    GuardedDeviceBuffer d_projection(seq.projection.size() * 2), d_qn(kDim * 2),
        d_pooled(keys.size() * 4), d_first(4), d_selected(std::size_t(tokens) * kTop * 4),
        d_counts(tokens * 4);
    const auto bits = encode_bf16(seq.projection), norms = encode_bf16(seq.query_norm);
    d_projection.copy_from_host(bits.data(), d_projection.bytes());
    d_qn.copy_from_host(norms.data(), d_qn.bytes());
    d_pooled.copy_from_host(keys.data(), d_pooled.bytes());
    Tensor projection(d_projection.data(), DType::BF16, {kProjection, tokens});
    Tensor norm(d_qn.data(), DType::BF16, {kDim});
    Tensor positions(d_first.data(), DType::I32, {1});
    Tensor pool(d_pooled.data(), DType::FP32, {kDim, capacity});
    Tensor selected(d_selected.data(), DType::I32, {kTop, tokens});
    Tensor counts(d_counts.data(), DType::I32, {tokens});
    const ops::QsaIndexerWeights weights{&norm, nullptr};
    WorkspaceArena workspace(ops::qsa_indexer_select_workspace_bytes(tokens, capacity));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int failures = 0;
    {
        const auto step = [&] {
            ops::qsa_indexer_select(projection, positions, weights, kEps, yarn.op(), pool, workspace,
                                   selected, counts, stream);
        };
        cudaGraphExec_t graph = nullptr;
        for (int replay = 0; replay < 2; ++replay) {
            const int position = first + replay * 3;
            d_first.copy_from_host(&position, 4);
            if (replay == 0) {
                step();
            } else {
                cudaGraph_t captured = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                step();
                CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
                CUDA_CHECK(cudaGraphInstantiate(&graph, captured, 0));
                CUDA_CHECK(cudaGraphDestroy(captured));
                CUDA_CHECK(cudaGraphLaunch(graph, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto got = from_device<int>(d_selected.data(), std::size_t(tokens) * kTop);
            const auto sizes = from_device<int>(d_counts.data(), tokens);
            for (int t = 0; t < tokens; ++t) {
                failures += check_selection("qsa T=" + std::to_string(tokens) +
                    " p=" + std::to_string(position + t) + (yarn.factor > 1.0 ? " yarn" : ""),
                    seq, pooled, position + t, got.data() + std::size_t(t) * kTop, sizes[t], yarn, t);
                if (ties) {
                    for (int i = 0; i < sizes[t]; ++i) {
                        if (got[std::size_t(t) * kTop + i] != i) { ++failures; break; }
                    }
                }
            }
            if (replay == 1) {
                CUDA_CHECK(cudaGraphLaunch(graph, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto same = from_device<int>(d_selected.data(), std::size_t(tokens) * kTop);
                const auto same_sizes = from_device<int>(d_counts.data(), tokens);
                if (same_sizes != sizes) { ++failures; }
                for (int t = 0; t < tokens; ++t) {
                    if (sizes[t] >= 0 && sizes[t] <= kTop &&
                        !std::equal(got.begin() + std::size_t(t) * kTop,
                                    got.begin() + std::size_t(t) * kTop + sizes[t],
                                    same.begin() + std::size_t(t) * kTop)) { ++failures; }
                }
                // Changed device position after capture must change causal visibility and RoPE.
                const int next = position + 2;
                d_first.copy_from_host(&next, 4);
                CUDA_CHECK(cudaGraphLaunch(graph, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto repeated = from_device<int>(d_selected.data(), std::size_t(tokens) * kTop);
                const auto repeated_sizes = from_device<int>(d_counts.data(), tokens);
                for (int t = 0; t < tokens; ++t) {
                    failures += check_selection("qsa changed-position graph", seq, pooled, next + t,
                        repeated.data() + std::size_t(t) * kTop, repeated_sizes[t], yarn, t);
                }
            }
        }
        CUDA_CHECK(cudaGraphExecDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto* buffer : {&d_projection, &d_qn, &d_pooled, &d_first, &d_selected, &d_counts}) {
        failures += buffer->verify_guards("qsa selection routes");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run(8100u);
    for (int tokens : {1, 3, 4, 5, 8, 9, 15, 16, 17, 37, 127, 128, 129, 512}) {
        failures += selection_case(tokens, 700, 2050);
    }
    failures += selection_case(5, 512, 2047);
    failures += selection_case(5, 700, 2400, true);
    failures += selection_case(128, 700, 2400, true);
    failures += selection_case(128, 8192, 32700);
    failures += selection_case(129, 32768, 130970);
    failures += selection_case(512, 32768, 0);
    failures += selection_case(512, 32768, 1800);
    // YaRN at the factor of Flash-Next's 589,824-token window: the whole prefill / decode / verify
    // flow, both scoring routes, and queries near that window's end.
    const Yarn yarn{2.25, 262144.0};
    failures += run(8200u, yarn);
    failures += selection_case(5, 700, 2050, false, yarn);
    failures += selection_case(129, 32768, 130970, false, yarn);
    failures += selection_case(5, 147500, 589000, false, yarn);
    failures += selection_case(128, 147500, 589000, false, yarn);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " qsa_indexer\n";
    return failures == 0 ? 0 : 1;
}
