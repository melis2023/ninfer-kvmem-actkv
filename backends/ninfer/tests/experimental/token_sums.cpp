#include "ninfer/ops/token_sums.h"
#include "ops/op_tester.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

void check_case(int channels, int tokens, int first, int block, bool clipped, bool graph) {
    std::mt19937 rng(913U + channels + tokens);
    std::uniform_real_distribution<float> random(-4.0F, 4.0F);
    std::vector<std::uint16_t> represented(static_cast<std::size_t>(channels) * tokens);
    for (auto& value : represented) { value = f32_to_bf16(random(rng)); }
    const auto buckets = block ? (tokens + 2 * block - 2) / block : 1;
    auto x = to_device(represented);
    auto origin_buffer = to_device(std::vector<int>{first});
    auto range_buffer = to_device(std::vector<int>{0, 1000000});
    DeviceBuffer result_buffer(static_cast<std::size_t>(channels) * buckets * sizeof(float));
    Tensor input(x.p, DType::BF16, {channels, tokens});
    Tensor origin(origin_buffer.p, DType::I32, {1});
    Tensor range(range_buffer.p, DType::I32, {2});
    Tensor output(result_buffer.p, DType::FP32, {channels, buckets});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream");
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture begin");
        ops::token_sums(input, origin, range, block, output, stream);
        cuda_check(cudaStreamEndCapture(stream, &captured), "capture end");
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0), "instantiate");
    }
    for (int replay = 0; replay < (graph ? 3 : 1); ++replay) {
        // The original position changes across page boundaries without recapture.
        const int start = first + replay * 37;
        const int limits[] = {clipped ? start + 7 : start,
                              clipped ? start + tokens - 13 : start + tokens};
        origin_buffer.copy_from_host(&start, sizeof(start));
        range_buffer.copy_from_host(limits, sizeof(limits));
        if (graph) { cuda_check(cudaGraphLaunch(executable, stream), "graph launch"); }
        else { ops::token_sums(input, origin, range, block, output, stream); }
        cuda_check(cudaStreamSynchronize(stream), "sum completion");
        const auto actual = from_device<float>(result_buffer, output.numel());
        std::vector<double> expected(actual.size(), 0), absolute(actual.size(), 0);
        // Independent oracle bins each represented input row by original position.
        for (int row = tokens - 1; row >= 0; --row) {
            const int position = start + row;
            if (position < limits[0] || position >= limits[1]) { continue; }
            const int bucket = block ? position / block - start / block : 0;
            for (int channel = 0; channel < channels; ++channel) {
                const double value = bf16_to_f32(represented[static_cast<std::size_t>(row) * channels + channel]);
                const auto index = static_cast<std::size_t>(bucket) * channels + channel;
                expected[index] += value;
                absolute[index] += std::abs(value);
            }
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            // FP32 summation forward-error bound from the represented BF16 inputs.
            const double bound = 2.0 * tokens * std::numeric_limits<float>::epsilon() * absolute[i];
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > std::max(1e-6, bound)) {
                throw std::runtime_error("token_sums differs from the FP64 oracle");
            }
        }
    }
    if (executable) { cuda_check(cudaGraphExecDestroy(executable), "destroy graph executable"); }
    if (captured) { cuda_check(cudaGraphDestroy(captured), "destroy graph"); }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
}

void check_query_capture(int channels, int tokens, int capacity, bool graph) {
    std::mt19937 rng(117U + channels + tokens);
    std::vector<std::uint16_t> represented(std::size_t(channels) * tokens);
    // Copying represented Q must preserve all bits, including tiny values and
    // signed zero. No conversion or arithmetic belongs at this boundary.
    for (auto& value : represented) value = static_cast<std::uint16_t>(rng());
    auto input_buffer = to_device(represented);
    auto origin_buffer = to_device(std::vector<int>{0});
    auto range_buffer = to_device(std::vector<int>{0, 0});
    const auto count = std::size_t(channels) * capacity;
    constexpr std::uint16_t guard = 0xA55A;
    auto result_buffer = to_device(std::vector<std::uint16_t>(count + 64, guard));
    Tensor input(input_buffer.p, DType::BF16, {channels, tokens});
    Tensor origin(origin_buffer.p, DType::I32, {1});
    Tensor range(range_buffer.p, DType::I32, {2});
    Tensor output(result_buffer.p, DType::BF16, {channels, capacity});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "query stream");
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "query capture begin");
        ops::capture_query_rows(input, origin, range, output, stream);
        cuda_check(cudaStreamEndCapture(stream, &captured), "query capture end");
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0), "query instantiate");
    }
    // End-aligned large chunk, left/right clipping, and empty spans overwrite
    // previous capture contents. Output capacity may be much smaller than T.
    const int spans[][2] = {{tokens-capacity, tokens}, {-7, capacity/2},
                            {tokens-capacity/2, tokens+19}, {tokens+1, tokens+2}, {-8, -1}};
    for (int pass = 0; pass < 5; ++pass) {
        const int start = 262000 + 37 * pass;
        const int limits[] = {start+spans[pass][0], start+spans[pass][1]};
        origin_buffer.copy_from_host(&start, sizeof(start));
        range_buffer.copy_from_host(limits, sizeof(limits));
        if (graph) cuda_check(cudaGraphLaunch(executable, stream), "query graph launch");
        else ops::capture_query_rows(input, origin, range, output, stream);
        cuda_check(cudaStreamSynchronize(stream), "query completion");
        const auto actual = from_device<std::uint16_t>(result_buffer, count+64);
        std::vector<std::uint16_t> expected(count, 0);
        std::size_t written = 0;
        for (int row = 0; row < tokens; ++row) {
            if (start+row < limits[0] || start+row >= limits[1]) continue;
            for (int channel = 0; channel < channels; ++channel)
                expected.at(written++) = represented[std::size_t(row)*channels+channel];
        }
        if (!std::equal(expected.begin(), expected.end(), actual.begin()) ||
            !std::all_of(actual.begin()+count, actual.end(), [](auto value) { return value == guard; }))
            throw std::runtime_error("compact Query capture differs from exact represented rows");
    }
    if (executable) cuda_check(cudaGraphExecDestroy(executable), "destroy query executable");
    if (captured) cuda_check(cudaGraphDestroy(captured), "destroy query graph");
    cuda_check(cudaStreamDestroy(stream), "destroy query stream");
}

void check_query_accumulation(int channels, int tokens, bool graph) {
    std::mt19937 rng(931U + channels + tokens);
    std::uniform_real_distribution<float> random(-4.0F, 4.0F);
    std::vector<std::uint16_t> represented(std::size_t(channels) * tokens);
    for (auto& value : represented) value = f32_to_bf16(random(rng));
    for (int row = 0; row < tokens; ++row)
        represented[std::size_t(row)*channels] = row%3 ? 0x0001 : 0x8001;
    auto input_buffer = to_device(represented);
    auto origin_buffer = to_device(std::vector<int>{0});
    auto range_buffer = to_device(std::vector<int>{0, 0});
    auto result_buffer = to_device(std::vector<float>(channels+64, 12345.0F));
    Tensor input(input_buffer.p, DType::BF16, {channels, tokens});
    Tensor origin(origin_buffer.p, DType::I32, {1});
    Tensor range(range_buffer.p, DType::I32, {2});
    Tensor output(result_buffer.p, DType::FP32, {channels, 1});
    std::vector<float> accepted[2] = {std::vector<float>(channels), std::vector<float>(channels)};
    for (int lane = 0; lane < 2; ++lane) for (int channel = 0; channel < channels; ++channel)
        accepted[lane][channel] = channel%3 ? float(lane+1) : (lane ? -1048576.0F : 1048576.0F);
    accepted[0][0] = accepted[1][0] = 0;
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "sum stream");
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "sum capture begin");
        ops::accumulate_query_sum(input, origin, range, output, stream);
        cuda_check(cudaStreamEndCapture(stream, &captured), "sum capture end");
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0), "sum instantiate");
    }
    for (int pass = 0; pass < 5; ++pass) {
        const int lane = pass == 1 ? 1 : 0;
        const int start = 262000 + pass*tokens;
        const int limits[] = {start+(pass == 3 ? tokens+1 : 7), start+tokens-3};
        origin_buffer.copy_from_host(&start, sizeof(start));
        range_buffer.copy_from_host(limits, sizeof(limits));
        result_buffer.copy_from_host(accepted[lane].data(), std::size_t(channels)*sizeof(float));
        if (graph) cuda_check(cudaGraphLaunch(executable, stream), "sum graph launch");
        else ops::accumulate_query_sum(input, origin, range, output, stream);
        cuda_check(cudaStreamSynchronize(stream), "sum completion");
        const auto actual = from_device<float>(result_buffer, channels+64);
        auto sequential = accepted[lane];
        std::vector<double> reference(sequential.begin(), sequential.end()), absolute(channels);
        for (int channel = 0; channel < channels; ++channel) absolute[channel] = std::abs(reference[channel]);
        for (int row = 0; row < tokens; ++row) {
            if (start+row < limits[0] || start+row >= limits[1]) continue;
            for (int channel = 0; channel < channels; ++channel) {
                const float value = bf16_to_f32(represented[std::size_t(row)*channels+channel]);
                sequential[channel] += value;
                reference[channel] += value;
                absolute[channel] += std::abs(value);
            }
        }
        for (int channel = 0; channel < channels; ++channel) {
            const double bound = 2.0 * tokens * std::numeric_limits<float>::epsilon() * absolute[channel];
            if (!std::isfinite(actual[channel]) ||
                std::abs(actual[channel]-reference[channel]) > std::max(1e-6, bound) ||
                actual[channel] != sequential[channel])
                throw std::runtime_error("Mean-Q differs from FP64 oracle or chronological FP32 accumulation");
        }
        if (!std::all_of(actual.begin()+channels, actual.end(), [](auto value) { return value == 12345.0F; }))
            throw std::runtime_error("Mean-Q writes outside its single-column output");
        // Pass 1 is another sequence; pass 2 is aborted. Neither may contaminate
        // the next accepted chunk when the Program reuses the same graph/buffer.
        if (pass != 2) accepted[lane] = std::move(sequential);
    }
    if (executable) cuda_check(cudaGraphExecDestroy(executable), "destroy sum executable");
    if (captured) cuda_check(cudaGraphDestroy(captured), "destroy sum graph");
    cuda_check(cudaStreamDestroy(stream), "destroy sum stream");
}

int main(int argc, char** argv) {
    try {
        if (cuda_unavailable()) { return 77; }
        int cases = 0;
        if (argc == 2 && std::string(argv[1]) == "candidates") {
            for (int channels : {512, 1024}) {
                for (int tokens : {2, 4, 8, 12, 16, 32, 64}) {
                    for (bool graph : {false, true}) {
                        check_case(channels, tokens, 0, 1, false, graph);
                        ++cases;
                    }
                }
            }
            std::cout << "TOKEN_SUMS_CANDIDATES PASS cases=" << cases << '\n';
            return 0;
        }
        for (int channels : {512, 1024, 4096, 6144}) {
            for (int tokens : {1, 63, 64, 65, 128, 511, 512, 513}) {
                for (int first : {0, 63, 262000}) {
                    for (int block : {0, 1, 64}) {
                        for (bool clipped : {false, true}) {
                            check_case(channels, tokens, first, block, clipped, false);
                            ++cases;
                        }
                    }
                }
            }
            check_case(channels, 65, 262000, 64, false, true);
            check_case(channels, 65, 262000, 0, true, true);
            for (int tokens : {1, 2, 3, 4}) {
                check_case(channels, tokens, 63, 1, false, true);
            }
            cases += 6;
        }
        for (int channels : {513, 1024, 6144}) {
            for (bool graph : {false, true}) {
                check_query_capture(channels, 2048, 512, graph);
                check_query_capture(channels, 128, 64, graph);
                check_query_capture(channels, 65, 512, graph);
                check_query_capture(channels, 1, 1, graph);
                check_query_accumulation(channels, 2048, graph);
                check_query_accumulation(channels, 128, graph);
                cases += 6;
            }
        }
        std::cout << "TOKEN_SUMS PASS cases=" << cases << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TOKEN_SUMS FAIL " << error.what() << '\n';
        return 1;
    }
}
