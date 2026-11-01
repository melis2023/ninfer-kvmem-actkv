// GPU counterpart of q2_cpu_probe over the same stored real layer and public BF16 inputs.
// GPU timings include the public Op's shared-expert computation at zero contribution weight;
// CPU timings exclude it. Neither measurement includes routing, transfers or whole inference.
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_experts.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
constexpr int H = 2560, I = 640, TOP = 10, E = 80;
constexpr std::size_t matrix_bytes = std::size_t(H) * I / 64 * 18;
void check(cudaError_t rc) {
    if (rc != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(rc)); }
}
struct Bank {
    DeviceBuffer data{E * matrix_bytes + 256};
    DeviceBuffer table{E * sizeof(Weight)};
    explicit Bank(std::ifstream& source, int width, bool native) {
        std::vector<std::uint8_t> bytes(E * matrix_bytes);
        source.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        if (!source) { throw std::runtime_error("truncated expert fixture"); }
        const int rows = width == H ? I : H;
        const auto geometry = weight_geometry(QType::Q2_G64_FP16, QuantLayout::RowSplit,
            std::array<std::uint64_t, 2>{unsigned(rows), unsigned(width)});
        if (native) {
            if (geometry.bytes != matrix_bytes) { throw std::runtime_error("unexpected Q2 geometry"); }
            std::vector<std::uint8_t> planes(bytes.size());
            for (int e = 0; e < E; ++e) {
                for (std::size_t group = 0; group < std::size_t(H) * I / 64; ++group) {
                    const auto* block = bytes.data() + e * matrix_bytes + group * 18;
                    std::memcpy(planes.data() + e * matrix_bytes + group * 16, block + 2, 16);
                    std::memcpy(planes.data() + e * matrix_bytes + geometry.scale_offset + group * 2,
                                block, 2);
                }
            }
            bytes.swap(planes);
        }
        data.fill(0);
        data.copy_from_host(bytes.data(), bytes.size());
        std::vector<const void*> pointers(E);
        for (int e = 0; e < E; ++e) {
            pointers[e] = static_cast<const std::byte*>(data.p) + e * matrix_bytes;
        }
        if (native) {
            std::vector<Weight> weights(E);
            for (int e = 0; e < E; ++e) {
                WeightParent parent{geometry, static_cast<const std::byte*>(pointers[e])};
                const WeightView view{{unsigned(rows), unsigned(width)},
                    {{&parent, 0, std::uint64_t(rows) * width}}};
                weights[e] = ops::prepare_native_expert({view, ops::LinearPolicy::AllowA8}, rows, width, false);
            }
            table.copy_from_host(weights.data(), weights.size() * sizeof(Weight));
        } else {
            table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
        }
    }
    ops::GgufExpertTable view(int width) const {
        return {QType::GGUF_Q2_0, static_cast<const void* const*>(table.p), width / 64 * 18};
    }
    ops::NativeExpertTable native_view(bool a8) const {
        return {QType::Q2_G64_FP16, static_cast<const Weight*>(table.p), a8};
    }
};
}

int main(int argc, char** argv) try {
    if (argc != 3 && argc != 4) {
        throw std::runtime_error("expected expert fixture, FP64 oracle file and optional native-a16/native-a8");
    }
    const std::string route = argc == 4 ? argv[3] : "gguf";
    if (route != "gguf" && route != "native-a16" && route != "native-a8") {
        throw std::runtime_error("unknown expert probe route");
    }
    const bool native = route != "gguf", a8 = route == "native-a8";
    std::cout << "GPU_PROBE_ROUTE " << route << '\n';
    check(cudaSetDevice(0));
    std::ifstream source(argv[1], std::ios::binary), oracle(argv[2], std::ios::binary);
    std::array<std::uint32_t, 3> header{};
    source.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (header != std::array<std::uint32_t, 3>{H, I, E} || !oracle) {
        throw std::runtime_error("invalid probe inputs");
    }
    Bank gate(source, H, native), up(source, H, native), down(source, I, native);
    const ops::GgufMoeWeights banks{gate.view(H), up.view(H), down.view(I),
        gate.view(H), up.view(H), down.view(I), E, true};
    const ops::NativeMoeWeights native_banks{gate.native_view(a8), up.native_view(a8), down.native_view(a8),
        gate.native_view(a8), up.native_view(a8), down.native_view(a8), E};
    DeviceBuffer flush(32U << 20);
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    std::map<std::pair<int, bool>, std::vector<double>> references;
    for (int t : {1, 2, 5, 8}) for (bool reuse : {true, false}) {
        std::array<std::uint32_t, 3> record{};
        oracle.read(reinterpret_cast<char*>(record.data()), sizeof(record));
        if (record != std::array<std::uint32_t, 3>{std::uint32_t(t), std::uint32_t(reuse),
                                                  std::uint32_t(H * t)}) {
            throw std::runtime_error("oracle record does not match the workload");
        }
        auto& expected = references[{t, reuse}];
        expected.resize(H * t);
        oracle.read(reinterpret_cast<char*>(expected.data()), expected.size() * 8);
        if (!oracle) { throw std::runtime_error("truncated FP64 oracle"); }
    }
    int failures = 0;
    std::cout << "Wide workloads repeat eight BF16 input columns; reuse=0 cycles over 80 experts\n";
    for (int t : {1, 2, 5, 8, 16, 31, 32, 33, 64, 128, 512}) for (bool reuse : {true, false}) {
        const auto& reference = references.at({std::min(t, 8), reuse});
        std::vector<double> expected(H * t);
        std::vector<__nv_bfloat16> x(H * t);
        std::vector<std::int32_t> ids(TOP * t);
        std::vector<float> weights(TOP * t, 0.1F), shared(t, 0.0F);
        for (int c = 0; c < t; ++c) {
            for (int i = 0; i < H; ++i) {
                x[c * H + i] = __float2bfloat16(std::sin(float(i * 13 + (c % 8) * 7)) * 1.3F);
                expected[c * H + i] = reference[(c % 8) * H + i];
            }
            for (int e = 0; e < TOP; ++e) { ids[c * TOP + e] = reuse ? e : (c % 8) * TOP + e; }
        }
        DeviceBuffer dx(x.size() * 2), di(ids.size() * 4), dw(weights.size() * 4),
            ds(shared.size() * 4), dy(H * t * 4);
        dx.copy_from_host(x.data(), dx.bytes);
        di.copy_from_host(ids.data(), di.bytes);
        dw.copy_from_host(weights.data(), dw.bytes);
        ds.copy_from_host(shared.data(), ds.bytes);
        Tensor tx(dx.p, DType::BF16, {H, t}), ti(di.p, DType::I32, {TOP, t}),
            tw(dw.p, DType::FP32, {TOP, t}), ts(ds.p, DType::FP32, {t}),
            ty(dy.p, DType::FP32, {H, t});
        WorkspaceArena workspace(native ? ops::moe_experts_native_workspace_bytes(t) :
                                          ops::moe_experts_gguf_workspace_bytes(t));
        const auto execute = [&] {
            if (native) { ops::moe_experts_native(tx, ti, tw, ts, native_banks, nullptr, workspace, ty, stream); }
            else { ops::moe_experts_gguf(tx, ti, tw, ts, banks, workspace, ty, stream); }
        };
        execute();
        check(cudaStreamSynchronize(stream));
        std::vector<float> got(H * t), again(H * t);
        dy.copy_to_host(got.data(), dy.bytes);
        execute();
        check(cudaStreamSynchronize(stream));
        dy.copy_to_host(again.data(), dy.bytes);
        double error = 0, norm = 0, largest = 0, peak = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double diff = double(got[i]) - expected[i];
            if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) { ++failures; }
            error += diff * diff; norm += expected[i] * expected[i];
            largest = std::max(largest, std::abs(diff));
            peak = std::max(peak, std::abs(expected[i]));
        }
        const double relative = std::sqrt(error / std::max(norm, 1e-30));
        const double gross = largest / std::max(peak, 1e-15);
        const bool repeated = std::memcmp(got.data(), again.data(), dy.bytes) == 0;
        std::cout << "oracle T=" << t << " reuse=" << reuse << " relative_l2=" << relative
                  << " max_over_peak=" << gross << " repeated=" << repeated << '\n';
        if (!std::isfinite(relative) || !std::isfinite(gross) || relative >= 0.04 || gross >= 0.16 ||
            !repeated) { ++failures; }
        for (bool cold : {false, true}) {
            std::vector<float> samples;
            for (int repetition = 0; repetition < 10; ++repetition) {
                if (cold) { check(cudaMemsetAsync(flush.p, repetition, flush.bytes, stream)); }
                check(cudaEventRecord(begin, stream));
                execute();
                check(cudaEventRecord(end, stream));
                check(cudaEventSynchronize(end));
                float milliseconds = 0;
                check(cudaEventElapsedTime(&milliseconds, begin, end));
                if (repetition) { samples.push_back(milliseconds); }
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "gpu T=" << t << " reuse=" << reuse << " cold=" << cold
                      << " median_ms=" << samples[4] << " min_ms=" << samples.front()
                      << " max_ms=" << samples.back() << '\n';
        }
    }
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    check(cudaStreamDestroy(stream));
    std::cout << "GPU_PROBE_" << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
