// llama.cpp's tensor-core flash attention behind the GGUF bridge (ops/linear/gguf/ggml_fattn.h),
// the prompt attention of Gemma 4's sliding-window (width 256) and full-attention (width 512)
// blocks and of Flint's blocks (width 128 with a learned sink per query head): FP32 queries against
// an F16 key/value cache under an F16 additive mask, at their head groups (16 query heads over 8
// key/value heads at width 256, over 2 at widths 128 and 512). Each width
// launches the tile that the build's device code has and its shared memory holds, so this also
// checks that both tiles run at all on the build's target. Each row is held against an FP64 softmax
// of the unrounded inputs, at query counts inside one tile, across tile boundaries and past one
// stream-k wave, and each launch is repeated and held to the first one's bits.

#include "core/arena.h"
#include "ops/linear/gguf/ggml_fattn.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ops::gguf;

constexpr int kQueryHeads = 16;

// Normwise relative error of one output row and a gross pointwise cap against the row's largest
// reference value: the kernel rounds the queries and the probabilities to FP16.
constexpr double kRowRelative = 4e-3;
constexpr double kPointwise   = 2e-2;
// Identical launches after the first, held to its bits.
constexpr int kReruns = 8;

int run(int head_dim, int kv_heads, int tokens, int base, std::mt19937& rng,
        int slots = 512, bool limit_kv = false, bool sinks = false) {
    const int group = kQueryHeads / kv_heads;
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> q(std::size_t(tokens) * kQueryHeads * head_dim);
    for (float& value : q) { value = unit(rng); }
    std::vector<__half> k(std::size_t(kv_heads) * slots * head_dim);
    std::vector<__half> v(k.size());
    for (__half& value : k) { value = __float2half(unit(rng)); }
    for (__half& value : v) { value = __float2half(unit(rng)); }
    // Query row t sees slots 0 .. base + t.
    std::vector<__half> mask(std::size_t(tokens) * slots);
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < slots; ++s) {
            mask[std::size_t(t) * slots + s] =
                __float2half(s <= base + t ? 0.0f : -std::numeric_limits<float>::infinity());
        }
    }
    const float scale = 1.0f / std::sqrt(float(head_dim)) * 2.0f;
    // Sink logits around the scores' scale, some above every score of a row.
    std::vector<float> sink(kQueryHeads);
    for (float& value : sink) { value = 3.0f * unit(rng); }

    DeviceBuffer d_q(q.size() * 4), d_k(k.size() * 2), d_v(v.size() * 2), d_mask(mask.size() * 2);
    DeviceBuffer d_out(q.size() * 4), d_sinks(sink.size() * 4);
    d_sinks.copy_from_host(sink.data(), sink.size() * 4);
    d_q.copy_from_host(q.data(), q.size() * 4);
    d_k.copy_from_host(k.data(), k.size() * 2);
    d_v.copy_from_host(v.data(), v.size() * 2);
    d_mask.copy_from_host(mask.data(), mask.size() * 2);
    const std::size_t workspace_bytes =
        gguf::flash_attention_workspace_bytes(head_dim, kQueryHeads, kv_heads, tokens);
    DeviceBuffer workspace(std::max<std::size_t>(workspace_bytes, 256));
    const int tile_rows = gguf::flash_attention_tile_rows(head_dim);
    std::vector<int> kv_max((tokens + tile_rows - 1) / tile_rows);
    for (int tile = 0; tile < static_cast<int>(kv_max.size()); ++tile) {
        const int visible = std::min(slots, base + std::min(tokens, (tile + 1) * tile_rows));
        kv_max[tile] = std::min(slots, ((visible + 255) / 256) * 256);
    }
    DeviceBuffer d_kv_max(kv_max.size() * sizeof(int));
    d_kv_max.copy_from_host(kv_max.data(), kv_max.size() * sizeof(int));

    gguf::FlashAttentionArgs args;
    args.head_dim        = head_dim;
    args.query_heads     = kQueryHeads;
    args.kv_heads        = kv_heads;
    args.tokens          = tokens;
    args.q               = static_cast<const float*>(d_q.p);
    args.q_token_stride  = std::int64_t(kQueryHeads) * head_dim;
    args.k               = d_k.p;
    args.v               = d_v.p;
    args.row_bytes       = std::int64_t(head_dim) * 2;
    args.head_bytes      = std::int64_t(slots) * head_dim * 2;
    args.slots           = slots;
    args.mask            = d_mask.p;
    args.mask_row_halves = slots;
    args.out             = static_cast<float*>(d_out.p);
    args.scale           = scale;
    args.kv_max          = limit_kv ? static_cast<const int*>(d_kv_max.p) : nullptr;
    args.sinks           = sinks ? static_cast<const float*>(d_sinks.p) : nullptr;
    gguf::flash_attention(args, workspace.p, workspace.bytes, nullptr);
    if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
        throw std::runtime_error(std::string("flash attention: ") + cudaGetErrorString(status));
    }
    std::vector<float> got(q.size());
    d_out.copy_to_host(got.data(), got.size() * 4);

    // A race between the kernel's warps shows as outputs that differ between identical launches
    // (Turing's 16- and 32-column tiles at width 512 gave such outputs): every rerun must give the
    // same bits.
    int reruns_differing = 0;
    std::vector<float> again(q.size());
    for (int rerun = 0; rerun < kReruns; ++rerun) {
        gguf::flash_attention(args, workspace.p, workspace.bytes, nullptr);
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
            throw std::runtime_error(std::string("flash attention: ") + cudaGetErrorString(status));
        }
        d_out.copy_to_host(again.data(), again.size() * 4);
        reruns_differing += std::memcmp(again.data(), got.data(), got.size() * 4) != 0;
    }

    int failures          = 0;
    double worst_relative = 0.0, worst_point = 0.0;
    std::vector<double> scores(slots), want(head_dim);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < kQueryHeads; ++h) {
            const int kv       = h / group;
            const float* q_row = q.data() + (std::size_t(t) * kQueryHeads + h) * head_dim;
            const int visible  = std::min(slots, base + t + 1);
            double peak        = -std::numeric_limits<double>::infinity();
            for (int s = 0; s < visible; ++s) {
                const __half* k_row = k.data() + (std::size_t(kv) * slots + s) * head_dim;
                double dot          = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    dot += double(q_row[d]) * double(__half2float(k_row[d]));
                }
                scores[s] = dot * scale;
                peak      = std::max(peak, scores[s]);
            }
            if (sinks) { peak = std::max(peak, double(sink[h])); }
            double sum = sinks ? std::exp(double(sink[h]) - peak) : 0.0;
            std::fill(want.begin(), want.end(), 0.0);
            for (int s = 0; s < visible; ++s) {
                const double p      = std::exp(scores[s] - peak);
                const __half* v_row = v.data() + (std::size_t(kv) * slots + s) * head_dim;
                sum += p;
                for (int d = 0; d < head_dim; ++d) {
                    want[d] += p * double(__half2float(v_row[d]));
                }
            }
            const float* got_row = got.data() + (std::size_t(t) * kQueryHeads + h) * head_dim;
            double err2 = 0.0, ref2 = 0.0, ref_max = 0.0, point = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                const double ref = want[d] / sum;
                const double err = double(got_row[d]) - ref;
                if (!std::isfinite(got_row[d])) { point = std::numeric_limits<double>::infinity(); }
                err2 += err * err;
                ref2 += ref * ref;
                ref_max = std::max(ref_max, std::abs(ref));
                point   = std::max(point, std::abs(err));
            }
            const double relative  = std::sqrt(err2 / std::max(ref2, 1e-30));
            const double pointwise = point / std::max(ref_max, 1e-30);
            worst_relative         = std::max(worst_relative, relative);
            worst_point            = std::max(worst_point, pointwise);
            if (!(relative <= kRowRelative && pointwise <= kPointwise)) { ++failures; }
        }
    }
    const bool ok = failures == 0 && reruns_differing == 0;
    std::cout << (ok ? "OK" : "FAIL") << " width " << head_dim << ", " << kv_heads << " KV heads, "
              << tokens << " queries after " << base << ", " << slots << " slots, kv_max="
              << limit_kv << ", sinks=" << sinks << ", tile_rows=" << tile_rows
              << ": worst row error " << worst_relative
              << ", worst point " << worst_point << " (" << failures << " rows over, "
              << reruns_differing << " of " << kReruns << " reruns differ)\n";
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        bool racecheck = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--racecheck") == 0) racecheck = true;
            else throw std::invalid_argument("usage: ninfer_gguf_flash_attention_test [--racecheck]");
        }
        const auto matrix = [&] {
            std::mt19937 rng(5120);
            int failures = 0;
            if (racecheck) {
                for (const int tokens : {3, 5}) {
                    failures += run(256, 8, tokens, 253, rng);
                    failures += run(512, 2, tokens, 253, rng);
                    failures += run(512, 2, tokens, 253, rng, 1024, true);
                    failures += run(128, 2, tokens, 253, rng, 1024, true, true);
                }
                return failures == 0 ? 0 : 1;
            }
            for (const int tokens : {1, 3, 4, 5, 9, 33, 130}) {
                failures += run(256, 8, tokens, 300, rng);
                failures += run(512, 2, tokens, 300, rng);
                failures += run(128, 2, tokens, 300, rng, 512, false, true);
            }
            failures += run(128, 2, 64, 0, rng);
            failures += run(128, 2, 64, 0, rng, 512, false, true);
            failures += run(512, 2, 64, 0, rng);
            // Cross the 256-slot mask boundary and exercise the combine loop at long contexts.
            // Both the full scan and kv_max must match the independent softmax, including tile tails.
            for (const int slots : {256, 1024, 4096, 16384}) {
                for (const bool limit_kv : {false, true}) {
                    failures += run(512, 2, 5, slots - 9, rng, slots, limit_kv);
                    failures += run(512, 2, 3, 253, rng, slots, limit_kv);
                    failures += run(128, 2, 5, slots - 9, rng, slots, limit_kv, true);
                }
            }
            std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF flash attention\n";
            return failures == 0 ? 0 : 1;
        };
        return matrix();
    } catch (const std::exception& error) {
        std::cerr << "GGUF flash attention: " << error.what() << '\n';
        return 1;
    }
}
