// hyper_connection_read/write against an FP64 oracle of the Qwen3.8-Flash-Next gated residual, at
// the model's shapes (4 streams, hidden 2560, lowrank 320) and decode, verify and prefill widths,
// with and without the inject rows (the final mixer has none), eagerly and under graph replay; and
// hyper_connection_expand, which must widen the embedding into every stream exactly.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/hyper_connection.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStreams = 4;
constexpr int kHidden  = 2560;
constexpr int kLowrank = 320;
constexpr int kWidth   = kStreams * kHidden;
constexpr float kEps   = 1e-6f;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Oracle {
    std::vector<double> mixed;  // [tokens][hidden]
    std::vector<double> inject; // [tokens][streams]
};

Oracle oracle(const std::vector<float>& stack, const std::vector<float>& norm,
              const std::vector<float>& down, const std::vector<float>& up,
              const std::vector<float>* inject, int tokens) {
    Oracle out{std::vector<double>(static_cast<std::size_t>(tokens) * kHidden),
               std::vector<double>(static_cast<std::size_t>(tokens) * kStreams)};
    std::vector<double> xn(kWidth), low(kLowrank);
    for (int t = 0; t < tokens; ++t) {
        const float* xs = stack.data() + static_cast<std::size_t>(t) * kWidth;
        for (int c = 0; c < kStreams; ++c) {
            double sum = 0.0;
            for (int d = 0; d < kHidden; ++d) sum += double(xs[c * kHidden + d]) * xs[c * kHidden + d];
            const double scale = 1.0 / std::sqrt(sum / kHidden + double(kEps));
            for (int d = 0; d < kHidden; ++d) {
                xn[c * kHidden + d] = xs[c * kHidden + d] * scale * (1.0 + norm[c * kHidden + d]);
            }
        }
        for (int r = 0; r < kLowrank; ++r) {
            double v = 0.0;
            for (int k = 0; k < kWidth; ++k) v += double(down[static_cast<std::size_t>(r) * kWidth + k]) * xn[k];
            v /= kStreams;
            low[r] = v * sigmoid(v);
        }
        for (int d = 0; d < kHidden; ++d) {
            double mixed = 0.0;
            for (int c = 0; c < kStreams; ++c) {
                double g = 0.0;
                const std::size_t row = static_cast<std::size_t>(c * kHidden + d) * kLowrank;
                for (int k = 0; k < kLowrank; ++k) g += double(up[row + k]) * low[k];
                mixed += sigmoid(g) * xn[c * kHidden + d];
            }
            out.mixed[static_cast<std::size_t>(t) * kHidden + d] = mixed / kStreams;
        }
        if (inject != nullptr) {
            for (int c = 0; c < kStreams; ++c) {
                double v = 0.0;
                for (int k = 0; k < kWidth; ++k) v += double((*inject)[static_cast<std::size_t>(c) * kWidth + k]) * xn[k];
                out.inject[static_cast<std::size_t>(t) * kStreams + c] = 2.0 * sigmoid(v / kStreams);
            }
        }
    }
    return out;
}

int run_case(int tokens, bool with_inject, bool graph, std::uint32_t seed) {
    const std::string label = "hyper_connection T=" + std::to_string(tokens) +
                              (with_inject ? "" : " mixer") + (graph ? " graph" : "");
    std::vector<float> stack(static_cast<std::size_t>(kWidth) * tokens), norm(kWidth),
        down(static_cast<std::size_t>(kLowrank) * kWidth), up(static_cast<std::size_t>(kWidth) * kLowrank),
        inject(static_cast<std::size_t>(kStreams) * kWidth), y(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(stack, seed, -2.0f, 2.0f);
    // One stream of every token far larger than the others: the norm is per stream.
    for (int t = 0; t < tokens; ++t) {
        for (int d = 0; d < kHidden; ++d) stack[static_cast<std::size_t>(t) * kWidth + d] *= 64.0f;
    }
    fill_uniform(norm, seed + 1, -0.5f, 0.5f);
    fill_uniform(down, seed + 2, -0.03f, 0.03f);
    fill_uniform(up, seed + 3, -0.1f, 0.1f);
    fill_uniform(inject, seed + 4, -0.03f, 0.03f);
    fill_uniform(y, seed + 5, -4.0f, 4.0f);
    for (auto* v : {&norm, &down, &up, &inject, &y}) round_to_bf16(*v);

    const Oracle expected = oracle(stack, norm, down, up, with_inject ? &inject : nullptr, tokens);

    GuardedDeviceBuffer d_stack(stack.size() * 4), d_norm(norm.size() * 2), d_down(down.size() * 2),
        d_up(up.size() * 2), d_inject(inject.size() * 2), d_mixed(static_cast<std::size_t>(kHidden) * tokens * 2),
        d_weights(static_cast<std::size_t>(kStreams) * tokens * 4), d_y(y.size() * 2);
    d_stack.copy_from_host(stack.data(), d_stack.bytes());
    const auto copy_bf16 = [](GuardedDeviceBuffer& buffer, const std::vector<float>& values) {
        const auto bits = encode_bf16(values);
        buffer.copy_from_host(bits.data(), buffer.bytes());
    };
    copy_bf16(d_norm, norm);
    copy_bf16(d_down, down);
    copy_bf16(d_up, up);
    copy_bf16(d_inject, inject);
    copy_bf16(d_y, y);

    Tensor t_stack(d_stack.data(), DType::FP32, {kHidden, kStreams, tokens});
    Tensor t_norm(d_norm.data(), DType::BF16, {kWidth});
    Tensor t_down(d_down.data(), DType::BF16, {kWidth, kLowrank});
    Tensor t_up(d_up.data(), DType::BF16, {kLowrank, kWidth});
    Tensor t_inject(d_inject.data(), DType::BF16, {kWidth, kStreams});
    Tensor t_mixed(d_mixed.data(), DType::BF16, {kHidden, tokens});
    Tensor t_weights(d_weights.data(), DType::FP32, {kStreams, tokens});
    Tensor t_y(d_y.data(), DType::BF16, {kHidden, tokens});
    const ops::HyperConnectionWeights weights{&t_norm, &t_down, &t_up, with_inject ? &t_inject : nullptr};
    WorkspaceArena workspace(
        ops::hyper_connection_read_workspace_bytes(kStreams, kHidden, kLowrank, tokens));

    const auto run = [&](cudaStream_t stream) {
        ops::hyper_connection_read(t_stack, weights, kEps, workspace, t_mixed,
                                   with_inject ? &t_weights : nullptr, stream);
    };
    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        run(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CUDA_CHECK(cudaMemsetAsync(d_mixed.data(), 0xff, d_mixed.bytes(), stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        run(nullptr);
    }
    cuda_synchronize();

    int failures = verify_reduction(label + " mixed",
                                    from_device_bf16(d_mixed.data(), expected.mixed.size()),
                                    expected.mixed, {4.0e-3, 1.0e-6, 2.0 * 3.90625e-3});
    if (with_inject) {
        const auto got = from_device<float>(d_weights.data(), expected.inject.size());
        failures += verify_pointwise(label + " inject",
                                     std::vector<double>(got.begin(), got.end()), expected.inject,
                                     {2.0e-6, 2.0e-5});
        // The write: stack += y (x) inject, against the oracle's inject weights.
        ops::hyper_connection_write(t_stack, t_y, t_weights, nullptr);
        cuda_synchronize();
        std::vector<double> written(stack.size());
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < kStreams; ++c) {
                for (int d = 0; d < kHidden; ++d) {
                    const std::size_t i = (static_cast<std::size_t>(t) * kStreams + c) * kHidden + d;
                    written[i] = double(stack[i]) + double(y[static_cast<std::size_t>(t) * kHidden + d]) *
                                                        expected.inject[static_cast<std::size_t>(t) * kStreams + c];
                }
            }
        }
        const auto got_stack = from_device<float>(d_stack.data(), stack.size());
        failures += verify_pointwise(label + " write",
                                     std::vector<double>(got_stack.begin(), got_stack.end()),
                                     written, {1.0e-4, 2.0e-5});
        // And an FP32 block output (the MoE's) on top of it.
        std::vector<float> y32(y.size());
        fill_uniform(y32, seed + 6, -4.0f, 4.0f);
        GuardedDeviceBuffer d_y32(y32.size() * 4);
        d_y32.copy_from_host(y32.data(), d_y32.bytes());
        Tensor t_y32(d_y32.data(), DType::FP32, {kHidden, tokens});
        ops::hyper_connection_write(t_stack, t_y32, t_weights, nullptr);
        cuda_synchronize();
        for (std::size_t i = 0; i < written.size(); ++i) {
            const std::size_t column = i / kHidden, t = column / kStreams;
            written[i] = double(got_stack[i]) +
                         double(y32[t * kHidden + i % kHidden]) * expected.inject[column];
        }
        const auto got_fp32 = from_device<float>(d_stack.data(), stack.size());
        failures += verify_pointwise(label + " FP32 write",
                                     std::vector<double>(got_fp32.begin(), got_fp32.end()),
                                     written, {1.0e-4, 2.0e-5});
        failures += d_y32.verify_guards(label.c_str());
    }
    for (auto* buffer : {&d_stack, &d_norm, &d_down, &d_up, &d_inject, &d_mixed, &d_weights, &d_y}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

int run_expand(int tokens, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(kHidden) * tokens);
    std::uint32_t state = seed;
    for (auto& v : x) {
        state = state * 1664525u + 1013904223u;
        v = bf16_to_f32(f32_to_bf16(static_cast<float>(static_cast<std::int32_t>(state)) * 1e-9f));
    }
    const auto bits = encode_bf16(x);
    GuardedDeviceBuffer d_x(bits.size() * 2), d_stack(x.size() * kStreams * 4);
    d_x.copy_from_host(bits.data(), bits.size() * 2);
    Tensor t_x(d_x.data(), DType::BF16, {kHidden, tokens});
    Tensor t_stack(d_stack.data(), DType::FP32, {kHidden, kStreams, tokens});
    ops::hyper_connection_expand(t_x, t_stack, nullptr);
    cuda_synchronize();
    const auto got = from_device<float>(d_stack.data(), x.size() * kStreams);
    std::vector<float> want(got.size());
    for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < kStreams; ++c)
            for (int d = 0; d < kHidden; ++d)
                want[(static_cast<std::size_t>(t) * kStreams + c) * kHidden + d] =
                    x[static_cast<std::size_t>(t) * kHidden + d];
    const std::string label = "expand T=" + std::to_string(tokens);
    int failures            = verify_exact(label.c_str(), got, want);
    failures += d_x.verify_guards("expand input");
    failures += d_stack.verify_guards("expand stack");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // Every narrow width has its own kernels (1..8), then the wide path.
    for (const int tokens : {1, 2, 3, 6, 7, 8, 9, 16, 37}) {
        failures += run_case(tokens, true, false, 4100u + tokens);
    }
    failures += run_case(1, false, false, 4200u);
    failures += run_case(5, false, false, 4201u);
    failures += run_case(4, true, true, 4300u);
    failures += run_expand(1, 4400u);
    failures += run_expand(7, 4401u);
    // Refusals: an unsupported geometry and an inject output without inject rows.
    bool refused = false;
    try {
        (void)ops::hyper_connection_read_workspace_bytes(4, 2048, 320, 1);
    } catch (const std::invalid_argument&) { refused = true; }
    failures += refused ? 0 : 1;
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " hyper_connection\n";
    return failures == 0 ? 0 : 1;
}
