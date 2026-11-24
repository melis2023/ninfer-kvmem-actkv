// Real-weight gate for `ops::trellis_input_proj`: one coded parent (`blk.0.attn_qkv`, 10240 rows)
// is projected once and its rows are distributed to three logical outputs - one by the identity
// (a block-aligned split) and two by explicit maps (a reversed range and a 256-row interleave, the
// shape a full-attention parent's query/output-gate turns take).
//
// A passing run says the fused-projection path lands rows at the right destination offsets and
// reads the right parent rows, which is what "one coded matrix, many logical projections" needs.
//
// Usage: test_trellis_proj <probe directory>   (skips with 77 when the probe is absent)
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
#include <numeric>
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
    std::int32_t n = 0, k = 0;
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
    if (probe.n % 16 || probe.k % 16) { return false; }
    return probe.payload.size() == std::size_t(probe.n) / 16 * (std::size_t(probe.k) / 16) * 32 *
                                    std::size_t(probe.bits);
}

} // namespace

int run(const std::string& root) {
    Probe probe;
    if (!load(root + "/blk.0.attn_qkv.weight", probe)) {
        std::cout << "skip: no blk.0.attn_qkv.weight probe" << std::endl;
        return 77;
    }
    probe.upload();
    const std::int32_t k = probe.k;
    const std::int32_t n = probe.n;
    std::cout << "parent " << n << "x" << k << " TRL" << probe.bits << std::endl;

    // Three logical outputs: an identity split (rows 0..4095), a reversed tail (rows 6143..4096),
    // and a 256-row interleave taking every other turn of rows 0..8191.
    const std::int32_t rows0 = 4096, rows1 = 2048, rows2 = 8192;
    std::vector<std::int32_t> map0, map1, map2;
    for (std::int32_t i = 0; i < rows0; ++i) { map0.push_back(i); }
    for (std::int32_t i = 0; i < rows1; ++i) { map1.push_back(6143 - i); }
    // 256-row interleave over the first 8192 rows: the second half of each 512-row turn first, so
    // the map is a real permutation rather than the identity.
    for (std::int32_t pair = 0; pair < 16; ++pair) {
        for (std::int32_t r = 0; r < 256; ++r) { map2.push_back(pair * 512 + 256 + r); }
        for (std::int32_t r = 0; r < 256; ++r) { map2.push_back(pair * 512 + r); }
    }

    std::mt19937 rng(4321);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(static_cast<std::size_t>(k) * kTokens);
    std::vector<__nv_bfloat16> x_bf16(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x_bf16[i] = __float2bfloat16_rn(normal(rng));
        x[i]     = __bfloat162float(x_bf16[i]);
    }
    DeviceBuffer d_x(x.size() * sizeof(__nv_bfloat16));
    d_x.copy_from_host(x_bf16.data(), x.size() * sizeof(__nv_bfloat16));

    const std::int32_t shapes[3] = {rows0, rows1, rows2};
    const std::vector<std::int32_t>* maps[3] = {&map0, &map1, &map2};
    std::vector<DeviceBuffer> d_out(3);
    std::vector<DeviceBuffer> d_map(3);
    for (int i = 0; i < 3; ++i) {
        d_out[i] = DeviceBuffer(static_cast<std::size_t>(shapes[i]) * kTokens *
                                sizeof(__nv_bfloat16));
        d_map[i] = DeviceBuffer(maps[i]->size() * sizeof(std::int32_t));
        d_map[i].copy_from_host(maps[i]->data(), maps[i]->size() * sizeof(std::int32_t));
    }
    Tensor outputs[3] = {
        Tensor(d_out[0].p, DType::BF16, {rows0, kTokens}),
        Tensor(d_out[1].p, DType::BF16, {rows1, kTokens}),
        Tensor(d_out[2].p, DType::BF16, {rows2, kTokens}),
    };
    Tensor* output_ptrs[3] = {&outputs[0], &outputs[1], &outputs[2]};

    const Weight device_weight = probe.device_weight();
    ops::TrellisProjectionPart parts[3];
    for (int i = 0; i < 3; ++i) {
        parts[i].weight  = &device_weight;
        parts[i].output  = i;
        parts[i].dst_row = 0;
        parts[i].rows    = shapes[i];
        parts[i].row_map = i == 0 ? nullptr : static_cast<const std::int32_t*>(d_map[i].p);
    }

    WorkspaceArena workspace(ops::trellis_input_proj_workspace_bytes(parts, 3, kTokens));
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    Tensor x_tensor(d_x.p, DType::BF16, {k, kTokens});
    ops::trellis_input_proj(x_tensor, parts, 3, output_ptrs, 3, kTokens, workspace, stream);
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        std::cout << "FAIL: kernel " << cudaGetErrorString(status) << std::endl;
        cudaStreamDestroy(stream);
        return 1;
    }

    // Host reference of the whole parent, then the same row selection.
    std::vector<float> parent(static_cast<std::size_t>(n) * kTokens);
    if (!ops::detail::trellis::project_host(x.data(), kTokens, probe.host_weight(), parent.data())) {
        std::cout << "FAIL: host reference rejected the parent" << std::endl;
        cudaStreamDestroy(stream);
        return 1;
    }

    double squared_error = 0, squared_reference = 0;
    float largest = 0;
    int failures = 0;
    for (int i = 0; i < 3; ++i) {
        const auto* map = maps[i];
        std::vector<__nv_bfloat16> got(static_cast<std::size_t>(shapes[i]) * kTokens);
        d_out[i].copy_to_host(got.data(), got.size() * sizeof(__nv_bfloat16));
        double band_error = 0, band_reference = 0;
        for (std::int32_t r = 0; r < shapes[i]; ++r) {
            const std::size_t from = static_cast<std::size_t>(map == &map0
                                                                  ? r
                                                                  : (*map)[static_cast<std::size_t>(r)]);
            for (int t = 0; t < kTokens; ++t) {
                const std::size_t at       = static_cast<std::size_t>(r) * kTokens + t;
                const double      expected = parent[from * kTokens + t];
                const double      error    = double(__bfloat162float(got[at])) - expected;
                band_error += error * error;
                band_reference += expected * expected;
                largest = std::max(largest, std::abs(static_cast<float>(error)));
            }
        }
        squared_error += band_error;
        squared_reference += band_reference;
        const bool ok = band_reference > 0 && std::sqrt(band_error / band_reference) <= 2.0 / 256.0;
        std::cout << (ok ? "ok   " : "FAIL ") << "output " << i << ": " << shapes[i] << " rows, "
                  << (map == &map0 ? "identity" : "mapped") << std::endl;
        failures += ok ? 0 : 1;
    }
    const double relative = squared_reference > 0 ? std::sqrt(squared_error / squared_reference) : 0;
    std::cout << "relative L2 " << relative << " absmax " << largest << std::endl;
    cudaStreamDestroy(stream);
    std::cout << (failures == 0 ? "trellis input-proj gate ok\n" : "trellis input-proj gate FAILED\n");
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string root = argc > 1 ? argv[1] : "I:\\model\\trellis-probe";
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess) {
            std::cout << "no CUDA device" << std::endl;
            return 1;
        }
        return run(root);
    } catch (const std::exception& error) {
        std::cout << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
