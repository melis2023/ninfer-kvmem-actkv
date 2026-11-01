// Gemma 4's vision kernels against FP64 oracles, at the tower's widths (16 heads of 72, hidden
// 1152, GeGLU 4304, projection 2816) and two units of a 6 x 3 patch grid, the projections' FP32
// results in:
//  - embed: the patch projection plus the x and y position rows, and the first block's input;
//  - qkv: per-head norms (Q and K with gammas, V without), the axial RoPE by column and row, Q's
//    scale;
//  - residual: the post-normed delta into the residual, with and without the next block's input;
//  - geglu, pool (mean, scale, standardization, unscaled norm) and scatter_rows;
//  - build_mask with media units: a unit's soft tokens see each other ahead in the sliding ring,
//    text rows and other units do not.

#include "models/gemma4/kernels.h"
#include "models/gemma4/vision_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace vk = ninfer::models::gemma4::vision_kernels;
namespace kk = ninfer::models::gemma4::kernels;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

template <class T>
struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { check(cudaMalloc(&p, count * sizeof(T)), "malloc"); }
    explicit Device(const std::vector<T>& host) : Device(host.size()) { upload(host); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& host) {
        check(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> download() const {
        std::vector<T> host(n);
        check(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return host;
    }
};

double f(__nv_bfloat16 v) { return double(__bfloat162float(v)); }

std::vector<__nv_bfloat16> random_bf16(std::size_t n, std::mt19937& rng, float scale = 1.0f) {
    std::normal_distribution<float> normal(0.0f, scale);
    std::vector<__nv_bfloat16> out(n);
    for (auto& v : out) { v = __float2bfloat16(normal(rng)); }
    return out;
}

std::vector<float> random_f32(std::size_t n, std::mt19937& rng, float mean, float scale) {
    std::normal_distribution<float> normal(mean, scale);
    std::vector<float> out(n);
    for (auto& v : out) { v = normal(rng); }
    return out;
}

// BF16 outputs against an FP64 value: within BF16's rounding of the value plus a small absolute
// floor.
bool near_bf16(double got, double want) {
    return std::abs(got - want) <= 1.0 / 128.0 * std::abs(want) + 2e-3;
}

constexpr int kHidden = 1152, kHeads = 16, kD = 72, kWidth = 4304, kOut = 2816;
constexpr int kGridW = 6, kGridH = 3, kUnits = 2, kPerUnit = kGridW * kGridH;
constexpr int kPatches = kUnits * kPerUnit;
constexpr float kEps   = 1e-6f;

std::vector<double> rms_ref(const std::vector<double>& x, const float* gamma) {
    double ss = 0.0;
    for (const double v : x) { ss += v * v; }
    const double rs = 1.0 / std::sqrt(ss / double(x.size()) + kEps);
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { out[i] = x[i] * rs * (gamma ? gamma[i] : 1.0); }
    return out;
}

void test_embed(std::mt19937& rng) {
    constexpr int positions = 16;
    const auto x     = random_f32(std::size_t(kPatches) * kHidden, rng, 0.0f, 1.0f);
    const auto table = random_bf16(std::size_t(2) * positions * kHidden, rng, 0.5f);
    const auto norm  = random_f32(kHidden, rng, 1.0f, 0.2f);
    Device<__nv_bfloat16> dtable(table), dxn(std::size_t(kPatches) * kHidden);
    Device<float> dx(x), dnorm(norm), dh(std::size_t(kPatches) * kHidden);
    vk::EmbedArgs a;
    a.x = dx.p, a.table = dtable.p, a.positions = positions, a.patches = kPatches;
    a.per_unit = kPerUnit, a.grid_w = kGridW, a.hidden = kHidden, a.norm = dnorm.p, a.eps = kEps;
    a.h = dh.p, a.xn = dxn.p;
    vk::embed(a, nullptr);
    const auto h = dh.download();
    const auto xn = dxn.download();
    bool ok_h = true, ok_xn = true;
    for (int p = 0; p < kPatches; ++p) {
        const int local = p % kPerUnit, column = local % kGridW, row = local / kGridW;
        std::vector<double> want(kHidden);
        for (int c = 0; c < kHidden; ++c) {
            want[c] = double(x[std::size_t(p) * kHidden + c]) + f(table[std::size_t(column) * kHidden + c]) +
                      f(table[std::size_t(positions + row) * kHidden + c]);
            ok_h &= std::abs(h[std::size_t(p) * kHidden + c] - want[c]) <= 1e-5 * (1 + std::abs(want[c]));
        }
        const auto normed = rms_ref(want, norm.data());
        for (int c = 0; c < kHidden; ++c) {
            ok_xn &= near_bf16(f(xn[std::size_t(p) * kHidden + c]), normed[c]);
        }
    }
    expect(ok_h, "embed residual");
    expect(ok_xn, "embed normed input");
}

void test_qkv(std::mt19937& rng) {
    const auto qkv    = random_f32(std::size_t(kPatches) * 3 * kHidden, rng, 0.0f, 2.0f);
    const auto q_norm = random_f32(kD, rng, 1.0f, 0.3f);
    const auto k_norm = random_f32(kD, rng, 1.0f, 0.3f);
    std::vector<float> freq(kD / 4);
    for (int i = 0; i < kD / 4; ++i) { freq[i] = float(1.0 / std::pow(100.0, double(2 * i) / (kD / 2))); }
    Device<__nv_bfloat16> dq(std::size_t(kPatches) * kHidden), dk(std::size_t(kPatches) * kHidden),
        dv(std::size_t(kPatches) * kHidden);
    Device<float> dqkv(qkv), dqn(q_norm), dkn(k_norm), dfreq(freq);
    const float q_scale = std::sqrt(float(kD));
    vk::QkvArgs a;
    a.qkv = dqkv.p, a.patches = kPatches, a.per_unit = kPerUnit, a.grid_w = kGridW;
    a.heads = kHeads, a.head_dim = kD, a.q_norm = dqn.p, a.k_norm = dkn.p, a.inv_freq = dfreq.p;
    a.eps = kEps, a.q_scale = q_scale, a.q = dq.p, a.k = dk.p, a.v = dv.p;
    vk::qkv(a, nullptr);
    const std::vector<__nv_bfloat16> got[3] = {dq.download(), dk.download(), dv.download()};
    bool ok[3] = {true, true, true};
    for (int p = 0; p < kPatches; ++p) {
        const int local = p % kPerUnit;
        const double column = local % kGridW, row = local / kGridW;
        for (int part = 0; part < 3; ++part) {
            for (int head = 0; head < kHeads; ++head) {
                std::vector<double> x(kD);
                for (int i = 0; i < kD; ++i) {
                    x[i] = qkv[std::size_t(p) * 3 * kHidden + part * kHidden + head * kD + i];
                }
                const auto n = rms_ref(x, part == 0 ? q_norm.data() : part == 1 ? k_norm.data() : nullptr);
                for (int i = 0; i < kD; ++i) {
                    double want = n[i];
                    if (part < 2) {
                        const int half = i / (kD / 2), li = i % (kD / 2), pair = li % (kD / 4);
                        const double angle = (half == 0 ? column : row) * double(freq[pair]);
                        want = li < kD / 4 ? n[i] * std::cos(angle) - n[i + kD / 4] * std::sin(angle)
                                           : n[i] * std::cos(angle) + n[i - kD / 4] * std::sin(angle);
                    }
                    if (part == 0) { want *= q_scale; }
                    ok[part] &= near_bf16(f(got[part][std::size_t(p) * kHidden + head * kD + i]), want);
                }
            }
        }
    }
    expect(ok[0], "qkv: Q norm, RoPE and scale");
    expect(ok[1], "qkv: K norm and RoPE");
    expect(ok[2], "qkv: V norm");
}

void test_residual(std::mt19937& rng) {
    for (const bool with_next : {true, false}) {
        auto h          = random_f32(std::size_t(kPatches) * kHidden, rng, 0.0f, 3.0f);
        const auto o    = random_f32(std::size_t(kPatches) * kHidden, rng, 0.0f, 1.0f);
        const auto post = random_f32(kHidden, rng, 1.0f, 0.2f);
        const auto next = random_f32(kHidden, rng, 1.0f, 0.2f);
        Device<float> dh(h), dout(o), dpost(post), dnext(next);
        Device<__nv_bfloat16> dxn(std::size_t(kPatches) * kHidden);
        vk::residual(dh.p, dout.p, kPatches, kHidden, dpost.p, with_next ? dnext.p : nullptr, kEps,
                     dxn.p, nullptr);
        const auto got_h = dh.download();
        const auto xn    = dxn.download();
        bool ok_h = true, ok_xn = true;
        for (int p = 0; p < kPatches; ++p) {
            std::vector<double> delta(kHidden), sum(kHidden);
            for (int c = 0; c < kHidden; ++c) { delta[c] = o[std::size_t(p) * kHidden + c]; }
            const auto normed = rms_ref(delta, post.data());
            for (int c = 0; c < kHidden; ++c) {
                sum[c] = h[std::size_t(p) * kHidden + c] + normed[c];
                ok_h &= std::abs(got_h[std::size_t(p) * kHidden + c] - sum[c]) <= 1e-4 * (1 + std::abs(sum[c]));
            }
            if (with_next) {
                const auto in = rms_ref(sum, next.data());
                for (int c = 0; c < kHidden; ++c) {
                    ok_xn &= near_bf16(f(xn[std::size_t(p) * kHidden + c]), in[c]);
                }
            }
        }
        expect(ok_h, std::string("residual sum") + (with_next ? " (with next)" : ""));
        expect(ok_xn, "residual next input");
    }
}

void test_geglu(std::mt19937& rng) {
    const auto gu = random_f32(std::size_t(kPatches) * 2 * kWidth, rng, 0.0f, 2.0f);
    Device<float> dgu(gu);
    Device<__nv_bfloat16> dmid(std::size_t(kPatches) * kWidth);
    vk::geglu(dgu.p, kPatches, kWidth, dmid.p, nullptr);
    const auto mid = dmid.download();
    bool ok = true;
    for (int p = 0; p < kPatches; ++p) {
        for (int i = 0; i < kWidth; ++i) {
            const double g = gu[std::size_t(p) * 2 * kWidth + i];
            const double u = gu[std::size_t(p) * 2 * kWidth + kWidth + i];
            const double want =
                0.5 * g *
                (1.0 + std::tanh(std::sqrt(2.0 / std::numbers::pi) * (g + 0.044715 * g * g * g))) *
                u;
            ok &= near_bf16(f(mid[std::size_t(p) * kWidth + i]), want);
        }
    }
    expect(ok, "geglu");
}

void test_pool_and_scatter(std::mt19937& rng) {
    constexpr int k = 3, tokens_per_unit = (kGridW / k) * (kGridH / k), tokens = kUnits * tokens_per_unit;
    const auto h     = random_f32(std::size_t(kPatches) * kHidden, rng, 0.0f, 2.0f);
    const auto bias  = random_f32(kHidden, rng, 0.0f, 5.0f);
    const auto scale = random_f32(kHidden, rng, 0.05f, 0.01f);
    Device<float> dh(h), dbias(bias), dscale(scale);
    Device<__nv_bfloat16> dout(std::size_t(tokens) * kHidden);
    vk::PoolArgs a;
    a.h = dh.p, a.units = kUnits, a.grid_w = kGridW, a.grid_h = kGridH, a.pooling = k;
    a.hidden = kHidden, a.scale = std::sqrt(float(kHidden)), a.eps = kEps;
    a.std_bias = dbias.p, a.std_scale = dscale.p, a.out = dout.p;
    vk::pool(a, nullptr);
    const auto out = dout.download();
    bool ok = true;
    for (int t = 0; t < tokens; ++t) {
        const int unit = t / tokens_per_unit, local = t % tokens_per_unit;
        const int ty = local / (kGridW / k), tx = local % (kGridW / k);
        std::vector<double> v(kHidden, 0.0);
        for (int dy = 0; dy < k; ++dy) {
            for (int dx = 0; dx < k; ++dx) {
                const int p = unit * kPerUnit + (ty * k + dy) * kGridW + tx * k + dx;
                for (int c = 0; c < kHidden; ++c) { v[c] += h[std::size_t(p) * kHidden + c]; }
            }
        }
        for (int c = 0; c < kHidden; ++c) {
            v[c] = (v[c] / (k * k) * std::sqrt(double(kHidden)) - bias[c]) * scale[c];
        }
        const auto want = rms_ref(v, nullptr);
        for (int c = 0; c < kHidden; ++c) { ok &= near_bf16(f(out[std::size_t(t) * kHidden + c]), want[c]); }
    }
    expect(ok, "pool");

    // Two units of three tokens into rows 5.. and 20.. of a 32-row chunk.
    const auto soft = random_bf16(std::size_t(6) * kOut, rng);
    const std::vector<std::int32_t> rows{5, 20};
    std::vector<float> dst(std::size_t(32) * kOut, -7.0f);
    Device<__nv_bfloat16> dsoft(soft);
    Device<std::int32_t> drows(rows);
    Device<float> ddst(dst);
    vk::scatter_rows(dsoft.p, 6, 3, kOut, drows.p, ddst.p, nullptr);
    const auto got = ddst.download();
    bool scattered = true;
    for (int r = 0; r < 32; ++r) {
        const int unit = r >= 5 && r < 8 ? 0 : r >= 20 && r < 23 ? 1 : -1;
        for (int c = 0; c < kOut; ++c) {
            const double want = unit < 0 ? -7.0 : f(soft[std::size_t(unit * 3 + (r - rows[unit])) * kOut + c]);
            scattered &= got[std::size_t(r) * kOut + c] == float(want);
        }
    }
    expect(scattered, "scatter_rows");
}

void test_block_mask() {
    // A 12-token chunk at position 100 in a 512-slot ring with a 64-token window: rows 2..4 are
    // unit 0, rows 7..10 unit 1, the rest text.
    constexpr int tokens = 12, ring = 512, first = 100, window = 64;
    std::vector<std::int32_t> blocks(tokens, -1);
    for (int r = 2; r <= 4; ++r) { blocks[r] = 0; }
    for (int r = 7; r <= 10; ++r) { blocks[r] = 1; }
    Device<std::int32_t> dblocks(blocks);
    Device<__half> dmask(std::size_t(tokens) * ring);
    kk::build_mask(dmask.p, tokens, ring, first, ring, window, nullptr, dblocks.p);
    check(cudaDeviceSynchronize(), "mask");
    const auto mask = dmask.download();
    bool ok = true;
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < ring; ++s) {
            const int p = s; // the chunk and its window sit below the ring's size
            bool want = p <= first + t && p > first + t - window;
            if (p > first + t && p < first + tokens && blocks[t] >= 0 && blocks[p - first] == blocks[t]) {
                want = true;
            }
            if (p >= first + tokens) { want = false; }
            ok &= (__half2float(mask[std::size_t(t) * ring + s]) == 0.0f) == want;
        }
    }
    expect(ok, "build_mask with media units");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    std::mt19937 rng(20261011);
    test_embed(rng);
    test_qkv(rng);
    test_residual(rng);
    test_geglu(rng);
    test_pool_and_scatter(rng);
    test_block_mask();
    check(cudaDeviceSynchronize(), "sync");
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " gemma4 vision kernels\n";
    return failures == 0 ? 0 : 1;
}
