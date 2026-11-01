#include "ninfer/ops/argmax.h"
#include "core/device.h"
#include <algorithm>
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            logits[base + row] = f32_to_bf16(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first]  = f32_to_bf16(token % 3 == 0 ? -0.5f : 32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (token == 0) {
            std::fill(logits.begin() + base, logits.begin() + base + valid_rows, f32_to_bf16(-4.0f));
        }
        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = f32_to_bf16(32768.0f);
            logits[base + physical_rows - 1] = f32_to_bf16(65536.0f);
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    auto logits   = make_logits(physical_rows, tokens, valid_rows);
    auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    if (physical_rows == 248320 && tokens == 128) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax(logits_tensor, output_tensor, valid_rows, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int t = 0; t < tokens; ++t)
            logits[static_cast<std::size_t>(t) * physical_rows + valid_rows - 1] = f32_to_bf16(8192.0f);
        CUDA_CHECK(cudaMemcpyAsync(device_logits.data(), logits.data(), device_logits.bytes(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    }

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint16_t>(device_logits.data(), logits.size());
    const std::string label = "argmax rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

// FP32 logits rank values BF16 would round together: a later row a little above the earlier
// maximum wins, and an exact tie still selects the lower row.
int run_fp32_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    const auto rounded = make_logits(physical_rows, tokens, valid_rows);
    std::vector<float> logits(rounded.size());
    for (std::size_t i = 0; i < rounded.size(); ++i) { logits[i] = bf16_to_f32(rounded[i]); }
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        const std::int32_t low  = (13 + token * 4099) % (valid_rows - 1);
        const std::int32_t high = valid_rows - 1 - token % 2;
        const float peak        = 64.0f + static_cast<float>(token);
        logits[base + low]      = peak;
        logits[base + high]     = token % 3 == 0 ? peak : peak + 1.0f / 4096.0f;
        expected[static_cast<std::size_t>(token)] = token % 3 == 0 ? low : high;
    }

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(float));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_output.fill(0xcd);
    Tensor logits_tensor(device_logits.data(), DType::FP32, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    const std::string label = "argmax fp32 rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(label.c_str(),
                             from_device<std::int32_t>(device_output.data(),
                                                       static_cast<std::size_t>(tokens)),
                             expected);
    failures += verify_exact((label + " preserves logits").c_str(),
                             from_device<float>(device_logits.data(), logits.size()), logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

int run_top2(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    auto logits = make_logits(physical_rows, tokens, valid_rows);
    GuardedDeviceBuffer input(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer first(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    GuardedDeviceBuffer second(first.bytes());
    Tensor x(input.data(), DType::BF16, {physical_rows, tokens});
    Tensor a(first.data(), DType::I32, {tokens}), b(second.data(), DType::I32, {tokens});
    input.copy_from_host(logits.data(), input.bytes());
    ops::argmax_top2(x, a, b, valid_rows, nullptr);
    cuda_synchronize();
    int failures = 0;
    const auto compare = [&] {
        const auto winners = argmax_oracle(logits, physical_rows, tokens, valid_rows);
        std::vector<std::int32_t> runners(tokens);
        for (int t = 0; t < tokens; ++t) {
            const auto base = static_cast<std::size_t>(t) * physical_rows;
            int best = winners[t] == 0 ? 1 : 0;
            for (int v = 0; v < valid_rows; ++v) {
                // Ascending traversal retains the lowest token id on a tie.
                if (v != winners[t] && double(bf16_to_f32(logits[base + v])) >
                                           double(bf16_to_f32(logits[base + best]))) { best = v; }
            }
            runners[t] = best;
        }
        failures += verify_exact("top2 first", from_device<std::int32_t>(first.data(), tokens), winners);
        failures += verify_exact("top2 second", from_device<std::int32_t>(second.data(), tokens), runners);
    };
    compare();
    if (physical_rows == 248320 && tokens == 8) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax_top2(x, a, b, valid_rows, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int t = 0; t < tokens; ++t) {
                const auto base = static_cast<std::size_t>(t) * physical_rows;
                logits[base + valid_rows - 1] = f32_to_bf16(repeat == 0 ? 1024.0F : -1024.0F);
                logits[base + valid_rows - 2] = f32_to_bf16(512.0F);
            }
            CUDA_CHECK(cudaMemcpyAsync(input.data(), logits.data(), input.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            compare();
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    const auto refuse = [&](auto&& call) {
        try { call(); } catch (const std::invalid_argument&) { return; }
        throw std::runtime_error("top2 accepted invalid output/domain");
    };
    refuse([&] { ops::argmax_top2(x, a, b, 1, nullptr); });
    refuse([&] { ops::argmax_top2(x, a, a, valid_rows, nullptr); });
    Tensor wrong(second.data(), DType::I32, {tokens + 1});
    refuse([&] { ops::argmax_top2(x, a, wrong, valid_rows, nullptr); });
    failures += verify_exact("top2 preserves logits", from_device<std::uint16_t>(input.data(), logits.size()), logits);
    failures += input.verify_guards("top2 input");
    failures += first.verify_guards("top2 first");
    failures += second.verify_guards("top2 second");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 2);
    failures += run_case(248320, 248077, 8);
    failures += run_case(248320, 248077, 9);
    failures += run_case(248320, 248077, 64);
    failures += run_case(248320, 248077, 15);
    failures += run_case(248320, 248077, 128);
    failures += run_case(131072, 131072, 1);
    failures += run_case(131072, 131072, 15);
    failures += run_case(131072, 131072, 120);
    for (const auto tokens : {1, 8, 9, 64}) {
        failures += run_fp32_case(248320, 248077, tokens);
        failures += run_fp32_case(131072, 131072, tokens);
    }
    failures += run_fp32_case(511, 511, 3);
    for (const auto rows : {2, 3, 511, 512, 513, 248320}) {
        for (const auto tokens : {1, 8, 17}) {
            failures += run_top2(rows, rows == 248320 ? 248077 : rows, tokens);
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
