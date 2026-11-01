// Complete public indexer selection, including query preparation and top-512 selection.
#include "ninfer_bench_common.h"
#include "ninfer/ops/qsa_indexer.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
void run(int tokens, int capacity, cudaStream_t stream, DeviceBuffer& flush) {
    auto projection = make_bf16(std::size_t(640) * tokens, 1101);
    auto norm = make_bf16(128, 1102);
    auto pooled = make_f32(std::size_t(128) * capacity, 1103, -2.0f, 2.0f);
    DeviceBuffer first(4), selected(std::size_t(512) * tokens * 4), counts(tokens * 4);
    const int start = capacity * 4 - tokens;
    CUDA_CHECK(cudaMemcpy(first.p, &start, 4, cudaMemcpyHostToDevice));
    Tensor p(projection.p, DType::BF16, {640, tokens});
    Tensor n(norm.p, DType::BF16, {128}), k(pooled.p, DType::FP32, {128, capacity});
    Tensor position(first.p, DType::I32, {1}), out(selected.p, DType::I32, {512, tokens});
    Tensor count(counts.p, DType::I32, {tokens});
    WorkspaceArena workspace(ops::qsa_indexer_select_workspace_bytes(tokens, capacity));
    const ops::QsaIndexerWeights weights{&n, nullptr};
    TimedGraph graph;
    const auto call = [&](cudaStream_t s) {
        ops::qsa_indexer_select(p, position, weights, 1e-6f, ops::RopeYarn{}, k, workspace, out,
                                count, s);
    };
    call(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    graph.capture(stream, call);
    for (int warm = 0; warm < 5; ++warm) { graph.launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (bool cold : {false, true}) {
        std::vector<double> times;
        for (int sample = 0; sample < 33; ++sample) {
            if (cold) { flush_l2(flush, stream); }
            times.push_back(graph.launch_timed(stream));
        }
        const auto summary = summarize_timings(times);
        std::printf("{\"tokens\":%d,\"capacity\":%d,\"first_position\":%d,\"cache\":\"%s\","
                    "\"median_us\":%.3f,\"p95_us\":%.3f,\"nodes\":%zu,\"workspace_bytes\":%zu}\n",
                    tokens, capacity, start, cold ? "evicted" : "reused",
                    summary.median_us, summary.p95_us, graph.nodes(),
                    ops::qsa_indexer_select_workspace_bytes(tokens, capacity));
    }
}
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    std::printf("QSA selection: %s; CUDA %d; 33 public-Op graph samples; FP32 pooled keys\n",
                device.name, CUDART_VERSION);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto flush = make_f32(16 * 1024 * 1024, 1104);
    for (int capacity : {700, 8192, 32768}) {
        for (int tokens : {1, 4, 8, 16, 127, 128, 129, 512}) { run(tokens, capacity, stream, flush); }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
