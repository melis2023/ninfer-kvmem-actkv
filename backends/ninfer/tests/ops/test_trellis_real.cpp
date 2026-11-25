// Real-weight gate for the NariQ coded-linear Op: reads the raw planes that
// `python -m tools.trellis.dump_probe` wrote out of a converted `.ninfer` and runs
// `ops::trellis_linear` on the GPU against the host reference
// (`ops/trellis/trellis_reference.cpp`, itself an adaptation of ggml-trellis.cpp).
//
// The point is that the tile bytes, the descriptor and the side planes are the ones of the
// shipped Qwen3.8-27B-NariQ-3.0bpw model: a passing run says the device schedule reproduces the
// host definition on real data at 2, 3 and 6 bits, with both descriptor kinds in play.
//
// Usage: test_trellis_real <probe directory>   (skips with 77 when the directory is absent)
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/trellis.h"
#include "ops/trellis/trellis_reference.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kTokens = 4;

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { return {}; }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(stream),
                                     std::istreambuf_iterator<char>());
}

// A matrix and its planes, host copies plus the device copies the Op reads.
struct Probe {
    std::string name;
    std::int32_t n = 0;
    std::int32_t k = 0;
    int bits = 0;
    std::vector<std::uint8_t> payload;
    std::vector<float> s16;  // the GGUF F16 promoted to F32, as the artifact stores it
    std::vector<float> pre;
    std::vector<float> post;
    std::vector<std::int32_t> pin;
    std::vector<std::int32_t> pout;
    std::vector<std::int32_t> desc;

    DeviceBuffer d_payload, d_s16, d_pre, d_post, d_pin, d_pout, d_desc;

    Weight host_weight() const {
        Weight w;
        w.qtype         = static_cast<QType>(static_cast<int>(QType::TRELLIS_1BIT) + bits - 1);
        w.payload       = payload.data();
        w.payload_bytes = payload.size();
        w.qdata         = payload.data();
        w.n             = n;
        w.k             = k;
        w.group         = 16;
        w.layout        = QuantLayout::TrellisTile16;
        w.shape[0]      = n;
        w.shape[1]      = k;
        w.ndim          = 2;
        w.trellis_s16   = s16.data();
        w.trellis_pre   = pre.data();
        w.trellis_post  = post.data();
        w.trellis_pin   = pin.empty() ? nullptr : pin.data();
        w.trellis_pout  = pout.empty() ? nullptr : pout.data();
        w.trellis_desc  = desc.data();
        w.trellis_desc_len = static_cast<std::int32_t>(desc.size());
        return w;
    }

    Weight device_weight() const {
        Weight w = host_weight();
        w.payload      = d_payload.p;
        w.qdata        = d_payload.p;
        w.trellis_s16  = d_s16.p;
        w.trellis_pre  = d_pre.p;
        w.trellis_post = d_post.p;
        w.trellis_pin  = pin.empty() ? nullptr : static_cast<const std::int32_t*>(d_pin.p);
        w.trellis_pout = pout.empty() ? nullptr : static_cast<const std::int32_t*>(d_pout.p);
        // The descriptor is host metadata (the dispatcher validates and plans on the host).
        w.trellis_desc = desc.data();
        return w;
    }

    void upload() {
        d_payload = DeviceBuffer(payload.size());
        d_payload.copy_from_host(payload.data(), payload.size());
        d_s16  = DeviceBuffer(s16.size() * sizeof(float));
        d_s16.copy_from_host(s16.data(), s16.size() * sizeof(float));
        d_pre  = DeviceBuffer(pre.size() * sizeof(float));
        d_pre.copy_from_host(pre.data(), pre.size() * sizeof(float));
        d_post = DeviceBuffer(post.size() * sizeof(float));
        d_post.copy_from_host(post.data(), post.size() * sizeof(float));
        if (!pin.empty()) {
            d_pin = DeviceBuffer(pin.size() * sizeof(std::int32_t));
            d_pin.copy_from_host(pin.data(), pin.size() * sizeof(std::int32_t));
        }
        if (!pout.empty()) {
            d_pout = DeviceBuffer(pout.size() * sizeof(std::int32_t));
            d_pout.copy_from_host(pout.data(), pout.size() * sizeof(std::int32_t));
        }
        d_desc = DeviceBuffer(desc.size() * sizeof(std::int32_t));
        d_desc.copy_from_host(desc.data(), desc.size() * sizeof(std::int32_t));
    }
};

bool load(const std::string& directory, Probe& probe) {
    probe.payload = read_bytes(directory + "/payload.bin");
    if (probe.payload.empty()) { return false; }
    const auto s16 = read_bytes(directory + "/s16.f32");
    probe.s16.assign(reinterpret_cast<const float*>(s16.data()),
                     reinterpret_cast<const float*>(s16.data()) + s16.size() / sizeof(float));
    const auto pre = read_bytes(directory + "/pre.f32");
    probe.pre.assign(reinterpret_cast<const float*>(pre.data()),
                     reinterpret_cast<const float*>(pre.data()) + pre.size() / sizeof(float));
    const auto post = read_bytes(directory + "/post.f32");
    probe.post.assign(reinterpret_cast<const float*>(post.data()),
                      reinterpret_cast<const float*>(post.data()) + post.size() / sizeof(float));
    const auto pin = read_bytes(directory + "/pin.i32");
    probe.pin.assign(reinterpret_cast<const std::int32_t*>(pin.data()),
                     reinterpret_cast<const std::int32_t*>(pin.data()) + pin.size() / 4);
    const auto pout = read_bytes(directory + "/pout.i32");
    probe.pout.assign(reinterpret_cast<const std::int32_t*>(pout.data()),
                      reinterpret_cast<const std::int32_t*>(pout.data()) + pout.size() / 4);
    const auto desc = read_bytes(directory + "/desc.i32");
    probe.desc.assign(reinterpret_cast<const std::int32_t*>(desc.data()),
                      reinterpret_cast<const std::int32_t*>(desc.data()) + desc.size() / 4);
    // meta.json is one line per field; the four numbers the test needs are parsed here.
    std::ifstream meta(directory + "/meta.json");
    std::string text((std::istreambuf_iterator<char>(meta)), std::istreambuf_iterator<char>());
    auto field = [&text](const std::string& key) {
        const auto at = text.find("\"" + key + "\"");
        if (at == std::string::npos) { return std::string{}; }
        const auto colon = text.find(':', at);
        const auto begin = text.find_first_not_of(" \t\n", colon + 1);
        const auto end   = text.find_first_of(",}\n", begin);
        return text.substr(begin, end - begin);
    };
    probe.name = field("name");
    probe.n    = std::stoi(field("n"));
    probe.k    = std::stoi(field("k"));
    probe.bits = std::stoi(field("bits"));
    if (probe.n % 16 || probe.k % 16 || probe.bits < 1 || probe.bits > 6) { return false; }
    const auto expected = std::size_t(probe.n) / 16 * (std::size_t(probe.k) / 16) * 32 * probe.bits;
    return probe.payload.size() == expected && probe.pin.size() == (probe.pin.empty() ? 0 : probe.k);
}

__nv_bfloat16 to_bf16(float value) { return __float2bfloat16_rn(value); }

// Runs one `trellis_linear` at `tokens` and compares the device result to the host reference.
// tokens == 1 exercises the split-K decode GEMV route; tokens > 1 exercises the prefill GEMM route.
int check_tokens(const Probe& probe, int tokens, const Weight& host_weight, const Weight& device_weight) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(static_cast<std::size_t>(probe.k) * tokens);
    for (auto& value : x) { value = __bfloat162float(to_bf16(normal(rng))); }
    std::vector<__nv_bfloat16> x_bf16(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { x_bf16[i] = to_bf16(x[i]); }
    DeviceBuffer d_x(x.size() * sizeof(__nv_bfloat16));
    d_x.copy_from_host(x_bf16.data(), x.size() * sizeof(__nv_bfloat16));
    DeviceBuffer d_y(static_cast<std::size_t>(probe.n) * tokens * sizeof(__nv_bfloat16));

    WorkspaceArena workspace(ops::trellis_linear_workspace_bytes(device_weight, tokens));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    Tensor x_tensor(d_x.p, DType::BF16, {probe.k, tokens});
    Tensor y_tensor(d_y.p, DType::BF16, {probe.n, tokens});
    ops::trellis_linear(x_tensor, device_weight, y_tensor, tokens, workspace, stream);
    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        std::cout << "FAIL T=" << tokens << " " << probe.name << ": kernel " << cudaGetErrorString(status) << "\n";
        cudaStreamDestroy(stream);
        return 1;
    }

    std::vector<__nv_bfloat16> y_bf16(static_cast<std::size_t>(probe.n) * tokens);
    d_y.copy_to_host(y_bf16.data(), y_bf16.size() * sizeof(__nv_bfloat16));
    std::vector<float> y(static_cast<std::size_t>(probe.n) * tokens);
    for (std::size_t i = 0; i < y.size(); ++i) { y[i] = __bfloat162float(y_bf16[i]); }

    std::vector<float> reference(y.size());
    if (!ops::detail::trellis::project_host(x.data(), tokens, host_weight, reference.data())) {
        std::cout << "FAIL T=" << tokens << " " << probe.name << ": host reference rejected the weight\n";
        cudaStreamDestroy(stream);
        return 1;
    }
    double squared_error = 0, squared_reference = 0;
    float largest = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double error = double(y[i]) - double(reference[i]);
        squared_error += error * error;
        squared_reference += double(reference[i]) * double(reference[i]);
        largest = std::max(largest, std::abs(static_cast<float>(error)));
    }
    const double relative = squared_reference > 0 ? std::sqrt(squared_error / squared_reference) : 0;
    const double tolerance = 2.0 / 256.0;
    const bool ok = relative <= tolerance;
    std::cout << (ok ? "ok   " : "FAIL ") << probe.name << " T=" << tokens << " TRL" << probe.bits << " "
              << probe.n << "x" << probe.k << " relative L2 " << relative << " absmax " << largest << "\n";
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}

} // namespace

int run(const std::string& root) {
    const std::vector<std::string> names = {"blk.0.ffn_up.weight", "blk.0.attn_qkv.weight",
                                            "output.band00.weight"};
    int failures = 0;
    for (const auto& name : names) {
        Probe probe;
        const std::string directory = root + "/" + name;
        std::cout << "loading " << directory << std::endl;
        if (!load(directory, probe)) {
            std::cout << "skip " << name << ": no probe dump at " << directory << std::endl;
            failures += 1;
            continue;
        }
        std::cout << "loaded; uploading" << std::endl;
        probe.upload();
        std::cout << "uploaded" << std::endl;

        const Weight device_weight = probe.device_weight();
        const Weight host_weight   = probe.host_weight();
        failures += check_tokens(probe, kTokens, host_weight, device_weight);
        failures += check_tokens(probe, 1, host_weight, device_weight);
    }
    std::cout << (failures == 0 ? "trellis real-weight gate ok\n"
                                : "trellis real-weight gate FAILED\n");
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::cout << "trellis real probe: begin" << std::endl;
    const std::string root = argc > 1 ? argv[1] : "I:\\model\\trellis-probe";
    try {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        std::cout << "cuda devices " << devices << " " << cudaGetErrorString(status) << std::endl;
        if (status != cudaSuccess) {
            std::cout << "no CUDA device: " << cudaGetErrorString(status) << std::endl;
            return 1;
        }
        std::cout << "cuda ok" << std::endl;
        return run(root);
    } catch (const std::exception& error) {
        std::cout << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}



