// Public HC write/read composition versus the fused public Op, and the fused Op over BF16 versus ggml
// Q8_0 matrices, at the model's real geometry.
#include "ninfer_bench_common.h"
#include "ninfer/ops/hyper_connection.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(int tokens, bool fp32, cudaStream_t stream, DeviceBuffer& flush) {
    constexpr int hidden = 2560, streams = 4, width = hidden * streams, lowrank = 320;
    auto initial = make_f32(std::size_t(width) * tokens, 101, -2.0f, 2.0f);
    DeviceBuffer stack(initial.bytes), mixed(std::size_t(hidden) * tokens * 2);
    auto previous = make_f32(std::size_t(streams) * tokens, 102, 0.0f, 2.0f);
    DeviceBuffer injection(previous.bytes);
    auto y = fp32 ? make_f32(std::size_t(hidden) * tokens, 103)
                  : make_bf16(std::size_t(hidden) * tokens, 103);
    auto norm = make_bf16(width, 104);
    auto down = make_bf16(std::size_t(width) * lowrank, 105);
    auto up = make_bf16(std::size_t(lowrank) * width, 106);
    auto inject = make_bf16(std::size_t(width) * streams, 107);
    // Q8_0 blocks are 34 bytes, 17 BF16 words: each block's scale is a finite binary16 this way.
    auto down_q8   = make_bf16(std::size_t(width) * lowrank / 32 * 17, 109);
    auto up_q8     = make_bf16(std::size_t(lowrank) * width / 32 * 17, 110);
    auto inject_q8 = make_bf16(std::size_t(width) * streams / 32 * 17, 111);
    Tensor xs(stack.p, DType::FP32, {hidden, streams, tokens});
    Tensor output(y.p, fp32 ? DType::FP32 : DType::BF16, {hidden, tokens});
    Tensor iw(injection.p, DType::FP32, {streams, tokens});
    Tensor out(mixed.p, DType::BF16, {hidden, tokens});
    Tensor wn(norm.p, DType::BF16, {width});
    const ops::HyperConnectionWeights weights{&wn, {down.p}, {up.p}, {inject.p}};
    constexpr auto kQ8 = ops::HyperConnectionMatrix::Format::Q8_0;
    const ops::HyperConnectionWeights weights_q8{
        &wn, {down_q8.p, kQ8}, {up_q8.p, kQ8}, {inject_q8.p, kQ8}};
    WorkspaceArena workspace(ops::hyper_connection_read_workspace_bytes(streams, hidden, lowrank, tokens));
    const auto reset = [&] {
        CUDA_CHECK(cudaMemcpyAsync(stack.p, initial.p, initial.bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(injection.p, previous.p, previous.bytes, cudaMemcpyDeviceToDevice, stream));
    };
    const auto separate = [&](cudaStream_t s) {
        ops::hyper_connection_write(xs, output, iw, s);
        ops::hyper_connection_read(xs, weights, 1e-6f, workspace, out, &iw, s);
    };
    const auto fused = [&](cudaStream_t s) {
        ops::hyper_connection_write_read(xs, output, iw, weights, 1e-6f, workspace, out, &iw, s);
    };
    const auto fused_q8 = [&](cudaStream_t s) {
        ops::hyper_connection_write_read(xs, output, iw, weights_q8, 1e-6f, workspace, out, &iw, s);
    };
    // Initialize cuBLAS and both kernel sets before capture and measurement.
    reset();
    separate(stream);
    reset();
    fused(stream);
    reset();
    fused_q8(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TimedGraph baseline, candidate, quantized;
    baseline.capture(stream, separate);
    candidate.capture(stream, fused);
    quantized.capture(stream, fused_q8);
    const TimedGraph* graphs[] = {&baseline, &candidate, &quantized};
    for (int sample = 0; sample < 10; ++sample) {
        for (const TimedGraph* graph : graphs) {
            reset();
            graph->launch(stream);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const bool cold : {false, true}) {
        std::vector<double> times[3];
        for (int sample = 0; sample < 61; ++sample) {
            // Rotate the order; restore identical inputs and optionally evict L2 outside timing.
            for (int turn = 0; turn < 3; ++turn) {
                const int which = (sample + turn) % 3;
                reset();
                if (cold) { flush_l2(flush, stream); }
                times[which].push_back(graphs[which]->launch_timed(stream));
            }
        }
        const auto a = summarize_timings(times[0]), b = summarize_timings(times[1]),
                   c = summarize_timings(times[2]);
        std::printf("{\"tokens\":%d,\"y\":\"%s\",\"cache\":\"%s\","
                    "\"separate_us\":%.3f,\"fused_us\":%.3f,\"delta_pct\":%.3f,"
                    "\"separate_p95_us\":%.3f,\"fused_p95_us\":%.3f,"
                    "\"fused_q8_us\":%.3f,\"q8_delta_pct\":%.3f,\"fused_q8_p95_us\":%.3f,"
                    "\"separate_nodes\":%zu,\"fused_nodes\":%zu,\"fused_q8_nodes\":%zu}\n",
                    tokens, fp32 ? "fp32" : "bf16", cold ? "evicted" : "reused",
                    a.median_us, b.median_us, 100.0 * (b.median_us / a.median_us - 1.0),
                    a.p95_us, b.p95_us, c.median_us, 100.0 * (c.median_us / b.median_us - 1.0),
                    c.p95_us, baseline.nodes(), candidate.nodes(), quantized.nodes());
    }
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    std::printf("HC write/read: %s; CUDA %d; rotated CUDA Graph samples=61; reset outside timing\n",
                device.name, CUDART_VERSION);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto flush = make_f32(16 * 1024 * 1024, 108);
    for (const bool fp32 : {false, true}) {
        for (const int tokens : {1, 4, 8, 9, 16, 128, 512}) { run(tokens, fp32, stream, flush); }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
