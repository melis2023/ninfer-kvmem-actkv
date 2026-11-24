// Real-weight gate for `ops::trellis_banded_head`: two shards of the shipped Qwen3.8-27B-NariQ
// head (a TRL2 and a TRL6 band at different widths) are concatenated in band order, the artifact's
// vocab map is simulated by a rotation, and the result is compared row by row against the host
// reference projection of the same two shards.
//
// A passing run says: the shard slices of the concatenated plane land on the right offsets, the
// per-band `trellis_linear` calls keep their K, and the final vocab read is a row gather (the
// direction ggml uses) rather than a scatter.
//
// Usage: test_trellis_head <probe directory>   (skips with 77 when the probes are absent)
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
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

constexpr int kTokens = 2;

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { return {}; }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(stream),
                                     std::istreambuf_iterator<char>());
}

struct Probe {
    std::string name;
    std::int32_t n = 0;
    std::int32_t k = 0;
    int bits = 0;
    std::vector<std::uint8_t> payload;
    std::vector<float> s16, pre, post;
    std::vector<std::int32_t> pin, pout, desc;
    DeviceBuffer d_payload, d_s16, d_pre, d_post, d_pin, d_pout;

    Weight host_weight() const {
        Weight w;
        w.qtype            = static_cast<QType>(static_cast<int>(QType::TRELLIS_1BIT) + bits - 1);
        w.payload          = payload.data();
        w.payload_bytes    = payload.size();
        w.qdata            = payload.data();
        w.n                = n;
        w.k                = k;
        w.group            = 16;
        w.layout           = QuantLayout::TrellisTile16;
        w.shape[0]         = n;
        w.shape[1]         = k;
        w.ndim             = 2;
        w.trellis_s16      = s16.data();
        w.trellis_pre      = pre.data();
        w.trellis_post     = post.data();
        w.trellis_pin      = pin.empty() ? nullptr : pin.data();
        w.trellis_pout     = pout.empty() ? nullptr : pout.data();
        w.trellis_desc     = desc.data();
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
        return w;
    }

    void upload() {
        d_payload = DeviceBuffer(payload.size());
        d_payload.copy_from_host(payload.data(), payload.size());
        d_s16 = DeviceBuffer(s16.size() * sizeof(float));
        d_s16.copy_from_host(s16.data(), s16.size() * sizeof(float));
        d_pre = DeviceBuffer(pre.size() * sizeof(float));
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
    }
};

bool load(const std::string& directory, Probe& probe) {
    probe.payload = read_bytes(directory + "/payload.bin");
    if (probe.payload.empty()) { return false; }
    const auto read_floats = [](const std::string& path, std::vector<float>& out) {
        const auto raw = read_bytes(path);
        out.assign(reinterpret_cast<const float*>(raw.data()),
                   reinterpret_cast<const float*>(raw.data()) + raw.size() / sizeof(float));
    };
    const auto read_ints = [](const std::string& path, std::vector<std::int32_t>& out) {
        const auto raw = read_bytes(path);
        out.assign(reinterpret_cast<const std::int32_t*>(raw.data()),
                   reinterpret_cast<const std::int32_t*>(raw.data()) + raw.size() / 4);
    };
    read_floats(directory + "/s16.f32", probe.s16);
    read_floats(directory + "/pre.f32", probe.pre);
    read_floats(directory + "/post.f32", probe.post);
    read_ints(directory + "/pin.i32", probe.pin);
    read_ints(directory + "/pout.i32", probe.pout);
    read_ints(directory + "/desc.i32", probe.desc);
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
    return probe.payload.size() == expected;
}

} // namespace

int run(const std::string& root, const std::vector<std::string>& names) {
    std::vector<Probe> probes(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (!load(root + "/" + names[i], probes[i])) {
            std::cout << "skip: no probe dump for " << names[i] << std::endl;
            return 77;
        }
        probes[i].upload();
    }
    if (probes[0].k != probes[1].k) {
        std::cout << "FAIL: bands disagree on K (" << probes[0].k << " vs " << probes[1].k << ")\n";
        return 1;
    }
    const std::int32_t k     = probes[0].k;
    const std::int64_t total = std::int64_t(probes[0].n) + probes[1].n;
    std::cout << "bands " << probes[0].n << "x" << k << " TRL" << probes[0].bits << " + "
              << probes[1].n << "x" << k << " TRL" << probes[1].bits << std::endl;

    std::mt19937 rng(99);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(static_cast<std::size_t>(k) * kTokens);
    std::vector<__nv_bfloat16> x_bf16(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x_bf16[i] = __float2bfloat16_rn(normal(rng));
        x[i]     = __bfloat162float(x_bf16[i]);
    }
    DeviceBuffer d_x(x.size() * sizeof(__nv_bfloat16));
    d_x.copy_from_host(x_bf16.data(), x.size() * sizeof(__nv_bfloat16));
    DeviceBuffer d_logits(static_cast<std::size_t>(total) * kTokens * sizeof(__nv_bfloat16));

    // vocab -> concatenated-band row; a rotation keeps both bands' rows in play.
    std::vector<std::int32_t> map(static_cast<std::size_t>(total));
    for (std::int64_t v = 0; v < total; ++v) {
        map[static_cast<std::size_t>(v)] = static_cast<std::int32_t>((v + probes[1].n) % total);
    }
    DeviceBuffer d_map(map.size() * sizeof(std::int32_t));
    d_map.copy_from_host(map.data(), map.size() * sizeof(std::int32_t));

    Weight device_weights[2] = {probes[0].device_weight(), probes[1].device_weight()};
    const Weight* bands[2]   = {&device_weights[0], &device_weights[1]};
    WorkspaceArena workspace(ops::trellis_banded_head_workspace_bytes(bands, 2, kTokens));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    Tensor x_tensor(d_x.p, DType::BF16, {k, kTokens});
    Tensor map_tensor(d_map.p, DType::I32, {static_cast<std::int32_t>(total)});
    Tensor logits(d_logits.p, DType::BF16, {static_cast<std::int32_t>(total), kTokens});
    ops::trellis_banded_head(x_tensor, bands, 2, map_tensor, logits, kTokens, workspace, stream);
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        std::cout << "FAIL: kernel " << cudaGetErrorString(status) << std::endl;
        cudaStreamDestroy(stream);
        return 1;
    }

    std::vector<__nv_bfloat16> y_bf16(static_cast<std::size_t>(total) * kTokens);
    d_logits.copy_to_host(y_bf16.data(), y_bf16.size() * sizeof(__nv_bfloat16));
    std::vector<float> y(y_bf16.size());
    for (std::size_t i = 0; i < y.size(); ++i) { y[i] = __bfloat162float(y_bf16[i]); }

    // Host reference of the same concatenation, band by band, in band order.
    std::vector<float> reference(static_cast<std::size_t>(total) * kTokens);
    std::int64_t offset = 0;
    for (auto& probe : probes) {
        std::vector<float> band_reference(static_cast<std::size_t>(probe.n) * kTokens);
        const Weight host_weight = probe.host_weight();
        if (!ops::detail::trellis::project_host(x.data(), kTokens, host_weight,
                                                band_reference.data())) {
            std::cout << "FAIL: host reference rejected " << probe.name << std::endl;
            cudaStreamDestroy(stream);
            return 1;
        }
        std::copy(band_reference.begin(), band_reference.end(),
                  reference.begin() + static_cast<std::size_t>(offset) * kTokens);
        offset += probe.n;
    }

    double squared_error = 0, squared_reference = 0;
    float largest = 0;
    for (std::int64_t v = 0; v < total; ++v) {
        const std::size_t source = static_cast<std::size_t>(map[static_cast<std::size_t>(v)]);
        for (int t = 0; t < kTokens; ++t) {
            const std::size_t at  = static_cast<std::size_t>(v) * kTokens + t;
            const double expected = reference[source * kTokens + t];
            const double error    = double(y[at]) - expected;
            squared_error += error * error;
            squared_reference += expected * expected;
            largest = std::max(largest, std::abs(static_cast<float>(error)));
        }
    }
    const double relative =
        squared_reference > 0 ? std::sqrt(squared_error / squared_reference) : 0;
    // Both paths use the same FP32 arithmetic; only the per-band BF16 output rounding differs.
    const double tolerance = 2.0 / 256.0;
    const bool ok = relative <= tolerance;
    std::cout << (ok ? "ok   " : "FAIL ") << "banded head: " << total << " rows, " << kTokens
              << " tokens, relative L2 " << relative << " absmax " << largest << std::endl;
    cudaStreamDestroy(stream);
    std::cout << (ok ? "trellis banded-head gate ok\n" : "trellis banded-head gate FAILED\n");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string root = argc > 1 ? argv[1] : "I:\\model\\trellis-probe";
    std::vector<std::string> names = {"blk.0.ffn_up.weight", "output.band00.weight"};
    try {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess) {
            std::cout << "no CUDA device: " << cudaGetErrorString(status) << std::endl;
            return 1;
        }
        return run(root, names);
    } catch (const std::exception& error) {
        std::cout << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
