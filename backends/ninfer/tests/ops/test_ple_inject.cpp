// ple_inject against an FP64 oracle of Qwen3.8-Flash-Next's PLE block at the model's shapes: one
// call per sequence chunk and the same sequence split across calls (the convolution history
// carries the positions before a call), from a zero and from a populated history, under graph
// replay, and a token whose gate score is exactly zero; and the record form of a speculative
// verification with the history advanced past a committed prefix.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/ple_inject.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStreams  = 4;
constexpr int kHidden   = 2560;
constexpr int kWidth    = kStreams * kHidden;
constexpr int kTaps     = 4;
constexpr int kDilation = 3;
constexpr int kHistory  = 9;
constexpr float kEps    = 1e-6f;

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

struct Inputs {
    std::vector<float> stack, key, value, norm_key, norm_query, norm_conv, conv, history;
};

// Advances `stack` and `history` (FP64) through `tokens` positions starting at `first`.
void oracle(const Inputs& in, int first, int tokens, std::vector<double>& stack,
            std::vector<double>& history) {
    const auto groupnorm = [](const std::vector<double>& x, const std::vector<float>& w) {
        std::vector<double> out(kWidth);
        for (int c = 0; c < kStreams; ++c) {
            double sum = 0;
            for (int d = 0; d < kHidden; ++d) sum += x[c * kHidden + d] * x[c * kHidden + d];
            const double scale = 1.0 / std::sqrt(sum / kHidden + double(kEps));
            for (int d = 0; d < kHidden; ++d) out[c * kHidden + d] = x[c * kHidden + d] * scale * (1.0 + w[c * kHidden + d]);
        }
        return out;
    };
    std::vector<std::vector<double>> window; // N of the last kHistory positions, oldest first
    for (int j = 0; j < kHistory; ++j) {
        window.emplace_back(history.begin() + static_cast<std::ptrdiff_t>(j) * kWidth,
                            history.begin() + static_cast<std::ptrdiff_t>(j + 1) * kWidth);
    }
    for (int t = first; t < first + tokens; ++t) {
        std::vector<double> key(kWidth), xs(kWidth), value(kHidden);
        for (int i = 0; i < kWidth; ++i) {
            key[i] = in.key[static_cast<std::size_t>(t) * kWidth + i];
            xs[i]  = stack[static_cast<std::size_t>(t) * kWidth + i];
        }
        for (int d = 0; d < kHidden; ++d) value[d] = in.value[static_cast<std::size_t>(t) * kHidden + d];
        const auto k = groupnorm(key, in.norm_key);
        const auto q = groupnorm(xs, in.norm_query);
        std::vector<double> g(kWidth);
        for (int c = 0; c < kStreams; ++c) {
            double s = 0;
            for (int d = 0; d < kHidden; ++d) s += k[c * kHidden + d] * q[c * kHidden + d];
            s /= std::sqrt(double(kHidden));
            const double root = s == 0 ? 0 : std::copysign(std::sqrt(std::max(std::abs(s), 1e-6)), s);
            const double gate = 1.0 / (1.0 + std::exp(-root));
            for (int d = 0; d < kHidden; ++d) g[c * kHidden + d] = gate * value[d];
        }
        const auto n = groupnorm(g, in.norm_conv);
        window.push_back(n); // window[kHistory] is position t
        for (int i = 0; i < kWidth; ++i) {
            double sum = 0;
            for (int j = 0; j < kTaps; ++j) {
                sum += double(in.conv[static_cast<std::size_t>(i) * kTaps + j]) *
                       window[kHistory - kDilation * (kTaps - 1 - j)][i];
            }
            stack[static_cast<std::size_t>(t) * kWidth + i] += g[i] + sum / (1.0 + std::exp(-sum));
        }
        window.erase(window.begin());
    }
    for (int j = 0; j < kHistory; ++j) {
        std::copy(window[j].begin(), window[j].end(), history.begin() + static_cast<std::ptrdiff_t>(j) * kWidth);
    }
}

int run_case(const std::string& label, int tokens, const std::vector<int>& splits, bool zero_history,
             bool graph, std::uint32_t seed) {
    Inputs in;
    in.stack.resize(static_cast<std::size_t>(kWidth) * tokens);
    in.key.resize(static_cast<std::size_t>(kWidth) * tokens);
    in.value.resize(static_cast<std::size_t>(kHidden) * tokens);
    in.norm_key.resize(kWidth);
    in.norm_query.resize(kWidth);
    in.norm_conv.resize(kWidth);
    in.conv.resize(static_cast<std::size_t>(kWidth) * kTaps);
    in.history.assign(static_cast<std::size_t>(kWidth) * kHistory, 0.0f);
    fill_uniform(in.stack, seed, -3.0f, 3.0f);
    fill_uniform(in.key, seed + 1, -2.0f, 2.0f);
    fill_uniform(in.value, seed + 2, -2.0f, 2.0f);
    fill_uniform(in.norm_key, seed + 3, -0.5f, 0.5f);
    fill_uniform(in.norm_query, seed + 4, -0.5f, 0.5f);
    fill_uniform(in.norm_conv, seed + 5, -0.5f, 0.5f);
    fill_uniform(in.conv, seed + 6, -0.6f, 0.6f);
    if (!zero_history) fill_uniform(in.history, seed + 7, -1.5f, 1.5f);
    // Token 0's key is zero: its score is exactly zero and its gate one half.
    for (int i = 0; i < kWidth; ++i) in.key[i] = 0.0f;
    for (auto* v : {&in.key, &in.value, &in.norm_key, &in.norm_query, &in.norm_conv, &in.conv}) {
        round_to_bf16(*v);
    }
    std::vector<double> expected_stack(in.stack.begin(), in.stack.end());
    std::vector<double> expected_history(in.history.begin(), in.history.end());
    oracle(in, 0, tokens, expected_stack, expected_history);

    GuardedDeviceBuffer d_stack(in.stack.size() * 4), d_key(in.key.size() * 2),
        d_value(in.value.size() * 2), d_nk(kWidth * 2), d_nq(kWidth * 2), d_nc(kWidth * 2),
        d_conv(in.conv.size() * 2), d_history(in.history.size() * 4);
    const auto copy_bf16 = [](GuardedDeviceBuffer& buffer, const std::vector<float>& values) {
        const auto bits = encode_bf16(values);
        buffer.copy_from_host(bits.data(), buffer.bytes());
    };
    d_stack.copy_from_host(in.stack.data(), d_stack.bytes());
    d_history.copy_from_host(in.history.data(), d_history.bytes());
    copy_bf16(d_key, in.key);
    copy_bf16(d_value, in.value);
    copy_bf16(d_nk, in.norm_key);
    copy_bf16(d_nq, in.norm_query);
    copy_bf16(d_nc, in.norm_conv);
    copy_bf16(d_conv, in.conv);
    Tensor t_nk(d_nk.data(), DType::BF16, {kWidth});
    Tensor t_nq(d_nq.data(), DType::BF16, {kWidth});
    Tensor t_nc(d_nc.data(), DType::BF16, {kWidth});
    Tensor t_conv(d_conv.data(), DType::BF16, {kTaps, kWidth});
    Tensor t_history(d_history.data(), DType::FP32, {kWidth, kHistory});
    const ops::PleInjectWeights weights{&t_nk, &t_nq, &t_nc, &t_conv};
    WorkspaceArena workspace(ops::ple_inject_workspace_bytes(tokens));

    const auto run = [&](cudaStream_t stream) {
        int begin = 0;
        for (const int count : splits) {
            Tensor t_stack(static_cast<float*>(d_stack.data()) + static_cast<std::size_t>(begin) * kWidth,
                           DType::FP32, {kHidden, kStreams, count});
            Tensor t_key(static_cast<std::uint16_t*>(d_key.data()) + static_cast<std::size_t>(begin) * kWidth,
                         DType::BF16, {kWidth, count});
            Tensor t_value(static_cast<std::uint16_t*>(d_value.data()) + static_cast<std::size_t>(begin) * kHidden,
                           DType::BF16, {kHidden, count});
            ops::ple_inject(t_stack, t_key, t_value, weights, kEps, t_history, workspace, stream);
            begin += count;
        }
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
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        run(nullptr);
    }
    cuda_synchronize();
    const auto got_stack   = from_device<float>(d_stack.data(), in.stack.size());
    const auto got_history = from_device<float>(d_history.data(), in.history.size());
    int failures = verify_pointwise(label + " stack", std::vector<double>(got_stack.begin(), got_stack.end()),
                                    expected_stack, {5.0e-5, 5.0e-5});
    failures += verify_pointwise(label + " history",
                                 std::vector<double>(got_history.begin(), got_history.end()),
                                 expected_history, {5.0e-5, 5.0e-5});
    for (auto* buffer : {&d_stack, &d_key, &d_value, &d_nk, &d_nq, &d_nc, &d_conv, &d_history}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

// The record form over `tokens` positions, then the history advanced by the first `commit` of
// them: the stack must match the oracle over every position, the history must be untouched by the
// record and afterwards equal bit for bit what ple_inject over the committed prefix leaves, whose
// stack must equal the record's first positions bit for bit (a verification that keeps its
// prefix).
int run_record(int tokens, int commit, std::uint32_t seed) {
    const std::string label =
        "ple_inject_record T=" + std::to_string(tokens) + " commit " + std::to_string(commit);
    Inputs in;
    in.stack.resize(static_cast<std::size_t>(kWidth) * tokens);
    in.key.resize(static_cast<std::size_t>(kWidth) * tokens);
    in.value.resize(static_cast<std::size_t>(kHidden) * tokens);
    in.norm_key.resize(kWidth);
    in.norm_query.resize(kWidth);
    in.norm_conv.resize(kWidth);
    in.conv.resize(static_cast<std::size_t>(kWidth) * kTaps);
    in.history.resize(static_cast<std::size_t>(kWidth) * kHistory);
    fill_uniform(in.stack, seed, -3.0f, 3.0f);
    fill_uniform(in.key, seed + 1, -2.0f, 2.0f);
    fill_uniform(in.value, seed + 2, -2.0f, 2.0f);
    fill_uniform(in.norm_key, seed + 3, -0.5f, 0.5f);
    fill_uniform(in.norm_query, seed + 4, -0.5f, 0.5f);
    fill_uniform(in.norm_conv, seed + 5, -0.5f, 0.5f);
    fill_uniform(in.conv, seed + 6, -0.6f, 0.6f);
    fill_uniform(in.history, seed + 7, -1.5f, 1.5f);
    for (auto* v : {&in.key, &in.value, &in.norm_key, &in.norm_query, &in.norm_conv, &in.conv}) {
        round_to_bf16(*v);
    }
    std::vector<double> expected_stack(in.stack.begin(), in.stack.end());
    std::vector<double> scratch_history(in.history.begin(), in.history.end());
    oracle(in, 0, tokens, expected_stack, scratch_history);

    GuardedDeviceBuffer d_stack(in.stack.size() * 4), d_prefix(in.stack.size() * 4),
        d_key(in.key.size() * 2), d_value(in.value.size() * 2), d_nk(kWidth * 2), d_nq(kWidth * 2),
        d_nc(kWidth * 2), d_conv(in.conv.size() * 2), d_history(in.history.size() * 4),
        d_direct(in.history.size() * 4),
        d_normalized(static_cast<std::size_t>(kWidth) * tokens * 4);
    const auto copy_bf16 = [](GuardedDeviceBuffer& buffer, const std::vector<float>& values) {
        const auto bits = encode_bf16(values);
        buffer.copy_from_host(bits.data(), buffer.bytes());
    };
    d_stack.copy_from_host(in.stack.data(), d_stack.bytes());
    d_prefix.copy_from_host(in.stack.data(), d_prefix.bytes());
    d_history.copy_from_host(in.history.data(), d_history.bytes());
    d_direct.copy_from_host(in.history.data(), d_direct.bytes());
    copy_bf16(d_key, in.key);
    copy_bf16(d_value, in.value);
    copy_bf16(d_nk, in.norm_key);
    copy_bf16(d_nq, in.norm_query);
    copy_bf16(d_nc, in.norm_conv);
    copy_bf16(d_conv, in.conv);
    Tensor t_nk(d_nk.data(), DType::BF16, {kWidth});
    Tensor t_nq(d_nq.data(), DType::BF16, {kWidth});
    Tensor t_nc(d_nc.data(), DType::BF16, {kWidth});
    Tensor t_conv(d_conv.data(), DType::BF16, {kTaps, kWidth});
    Tensor t_history(d_history.data(), DType::FP32, {kWidth, kHistory});
    Tensor t_direct(d_direct.data(), DType::FP32, {kWidth, kHistory});
    const ops::PleInjectWeights weights{&t_nk, &t_nq, &t_nc, &t_conv};
    WorkspaceArena workspace(ops::ple_inject_workspace_bytes(tokens));

    Tensor t_stack(d_stack.data(), DType::FP32, {kHidden, kStreams, tokens});
    Tensor t_key(d_key.data(), DType::BF16, {kWidth, tokens});
    Tensor t_value(d_value.data(), DType::BF16, {kHidden, tokens});
    Tensor t_normalized(d_normalized.data(), DType::FP32, {kWidth, tokens});
    ops::ple_inject_record(t_stack, t_key, t_value, weights, kEps, t_history, t_normalized,
                           workspace, nullptr);
    cuda_synchronize();
    const auto untouched = from_device<float>(d_history.data(), in.history.size());
    int failures = verify_exact((label + " history untouched").c_str(), untouched, in.history);
    const auto got_stack = from_device<float>(d_stack.data(), in.stack.size());
    failures +=
        verify_pointwise(label + " stack", std::vector<double>(got_stack.begin(), got_stack.end()),
                         expected_stack, {5.0e-5, 5.0e-5});

    Tensor committed(d_normalized.data(), DType::FP32, {kWidth, commit});
    ops::ple_history_advance(t_history, committed, nullptr);
    Tensor t_prefix(d_prefix.data(), DType::FP32, {kHidden, kStreams, commit});
    Tensor t_key_prefix(d_key.data(), DType::BF16, {kWidth, commit});
    Tensor t_value_prefix(d_value.data(), DType::BF16, {kHidden, commit});
    ops::ple_inject(t_prefix, t_key_prefix, t_value_prefix, weights, kEps, t_direct, workspace,
                    nullptr);
    cuda_synchronize();
    failures += verify_exact((label + " committed history").c_str(),
                             from_device<float>(d_history.data(), in.history.size()),
                             from_device<float>(d_direct.data(), in.history.size()));
    const auto prefix =
        from_device<float>(d_prefix.data(), static_cast<std::size_t>(kWidth) * commit);
    failures += verify_exact(
        (label + " committed stack").c_str(), prefix,
        std::vector<float>(got_stack.begin(),
                           got_stack.begin() + static_cast<std::ptrdiff_t>(prefix.size())));
    for (auto* buffer : {&d_stack, &d_prefix, &d_key, &d_value, &d_nk, &d_nq, &d_nc, &d_conv,
                         &d_history, &d_direct, &d_normalized}) {
        failures += buffer->verify_guards(label.c_str());
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case("ple_inject T=1 start", 1, {1}, true, false, 5100u);
    failures += run_case("ple_inject T=1", 1, {1}, false, false, 5101u);
    failures += run_case("ple_inject T=12", 12, {12}, false, false, 5102u);
    failures += run_case("ple_inject T=12 as 5+1+6", 12, {5, 1, 6}, false, false, 5102u);
    failures += run_case("ple_inject T=40 start", 40, {40}, true, false, 5103u);
    failures += run_case("ple_inject T=7 as 1x7 graph", 7, {1, 1, 1, 1, 1, 1, 1}, false, true, 5104u);
    // Verification widths: a prefix shorter than the history window, one past it, and all of it.
    failures += run_record(4, 2, 5200u);
    failures += run_record(16, 11, 5201u);
    failures += run_record(9, 9, 5202u);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " ple_inject\n";
    return failures == 0 ? 0 : 1;
}
