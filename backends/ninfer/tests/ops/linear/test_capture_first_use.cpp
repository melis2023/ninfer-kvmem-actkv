// A wide projection whose first call in the process is inside a CUDA graph capture, as the verify
// graph of several requests makes it at startup before any prefill has run. On sm_70 that width
// takes the cuBLAS route, whose handle is created on first use; the capture must survive that and
// the replay must equal an eager call of the same route.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

int main() {
    using namespace ninfer;
    if (test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        constexpr std::int32_t kRows   = 1024;
        constexpr std::int32_t kInput  = 5120;
        constexpr std::int32_t kTokens = 80;
        constexpr ops::LinearPolicy kPolicy = ops::LinearPolicy::A16Only;

        const test::quantized_weight::PackedWeight packed =
            test::quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, kRows, kInput, 71U);
        DeviceBuffer weight_payload(packed.payload.size());
        weight_payload.copy_from_host(packed.payload.data(), packed.payload.size());
        const Weight weight = packed.device_weight(weight_payload.p);

        std::vector<std::uint16_t> activation(static_cast<std::size_t>(kInput) * kTokens);
        for (std::size_t i = 0; i < activation.size(); ++i) {
            activation[i] = test::f32_to_bf16(static_cast<float>(static_cast<int>(i % 61) - 30) *
                                              1.0e-2F);
        }
        DeviceBuffer x_buffer(activation.size() * sizeof(std::uint16_t));
        x_buffer.copy_from_host(activation.data(), x_buffer.bytes);
        const std::size_t out_bytes = static_cast<std::size_t>(kRows) * kTokens * 2;
        DeviceBuffer graph_out(out_bytes);
        DeviceBuffer eager_out(out_bytes);
        graph_out.fill(0);
        eager_out.fill(0);

        const std::size_t workspace_bytes = ops::linear_workspace_capacity_bytes(
            QType::Q4_G64_FP16, kRows, kInput, kPolicy, kTokens, kTokens);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
        const Tensor x(x_buffer.p, DType::BF16, {kInput, kTokens});
        Tensor graph_result(graph_out.p, DType::BF16, {kRows, kTokens});
        Tensor eager_result(eager_out.p, DType::BF16, {kRows, kTokens});

        cudaStream_t stream = nullptr;
        test::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                         "cudaStreamCreateWithFlags");
        test::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                         "cudaStreamBeginCapture");
        {
            auto scope = workspace.scope();
            ops::linear(x, weight, graph_result, kPolicy, workspace, stream);
        }
        cudaGraph_t graph = nullptr;
        test::cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
        cudaGraphExec_t exec = nullptr;
        test::cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
        test::cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
        {
            auto scope = workspace.scope();
            ops::linear(x, weight, eager_result, kPolicy, workspace, stream);
        }
        test::cuda_synchronize(stream);

        std::vector<std::uint8_t> from_graph(out_bytes);
        std::vector<std::uint8_t> from_eager(out_bytes);
        graph_out.copy_to_host(from_graph.data(), out_bytes);
        eager_out.copy_to_host(from_eager.data(), out_bytes);
        test::cuda_check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
        test::cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
        test::cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");

        bool nonzero = false;
        for (std::size_t i = 0; i < from_eager.size() && !nonzero; ++i) {
            nonzero = from_eager[i] != 0;
        }
        if (!nonzero || std::memcmp(from_graph.data(), from_eager.data(), out_bytes) != 0) {
            std::cerr << "FAIL: the replay of a projection first called under capture differs "
                         "from the eager call\n";
            return 1;
        }
        std::cout << "OK Linear first use under graph capture\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Linear first use under graph capture: " << error.what() << '\n';
        return 1;
    }
}
