// Gemma 4's quantized KV caches (kv_format.h) against independent oracles, for every --kv-dtype:
//  - kv_store: the stored codes, decoded on the host with their stored scales (and unrotated in
//    FP64 for a rotated plane), stay near the rows stored (a coding's own error bound);
//  - kv_load and kv_store's write-back: the device's F16 rows equal those host-decoded values;
//  - decode_attention over the planes (a sliding-window ring of D 256 with rotated keys, turned
//    queries and, for NVFP4 values, a turned-back output; a full-attention block of D 512 whose keys
//    derive from its one unrotated plane) against FP64 attention over the host-decoded values.

#include "../../ops/kv_cache_e8_root_host.h"
#include "../../ops/kv_cache_lloyd4_oracle.h"
#include "core/arena.h"
#include "models/gemma4/kernels.h"
#include "models/gemma4/kv_format.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::gemma4;
namespace k = ninfer::models::gemma4::kernels;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int failures = 0;

void report(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

double half_value(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, 2);
    return double(__half2float(h));
}

double e4m3(std::uint8_t c) {
    const int sign = c >> 7, exp = (c >> 3) & 15, man = c & 7;
    if (exp == 15 && man == 7) { return NAN; }
    const double v = exp == 0 ? std::ldexp(double(man), -9) : std::ldexp(1.0 + man / 8.0, exp - 7);
    return sign ? -v : v;
}

double e2m1(int nibble) {
    static constexpr double kValues[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (nibble & 8 ? -1.0 : 1.0) * kValues[nibble & 7];
}

int signed_nibble(const std::uint8_t* codes, int d) {
    const int n = d % 2 ? codes[d / 2] >> 4 : codes[d / 2] & 15;
    return (n ^ 8) - 8;
}

// One 256-wide block's represented values from its codes and scales, in the stored (possibly
// rotated) domain.
std::vector<double> decode_block(KvCoding coding, const std::uint8_t* codes, const std::uint8_t* scales) {
    std::vector<double> x(256);
    const auto scale16 = [&](int i) {
        std::uint16_t b;
        std::memcpy(&b, scales + 2 * i, 2);
        return half_value(b);
    };
    for (int d = 0; d < 256; ++d) {
        switch (coding) {
        case KvCoding::Int8G64: x[d] = double(std::int8_t(codes[d])) * scale16(d / 64); break;
        case KvCoding::Int4G32: x[d] = signed_nibble(codes, d) * scale16(d / 32); break;
        case KvCoding::Int4E8: x[d] = signed_nibble(codes, d) * scale16(d / 64); break;
        case KvCoding::Lloyd4: {
            const int index = d % 2 ? codes[d / 2] >> 4 : codes[d / 2] & 15;
            x[d] = test::lloyd4_code(std::uint8_t(index)) * scale16(d / 64);
            break;
        }
        case KvCoding::RootE8: {
            const int block = d / 8;
            const auto c = test::e8_root::decode({codes[2 * block], codes[2 * block + 1]});
            x[d] = double(c[d % 8]) * scale16(d / 64);
            break;
        }
        case KvCoding::Fp8Row: x[d] = e4m3(codes[d]) * scale16(0); break;
        case KvCoding::Nvfp4G16: {
            const int n = d % 2 ? codes[d / 2] >> 4 : codes[d / 2] & 15;
            x[d] = e2m1(n) * e4m3(scales[d / 16]);
            break;
        }
        case KvCoding::F16: break;
        }
    }
    return x;
}

// The normalized Sylvester-Hadamard transform, H[i][j] = (-1)^popcount(i & j) / 16 (its own inverse).
std::vector<double> hadamard(const std::vector<double>& x) {
    std::vector<double> y(256, 0.0);
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) { y[i] += (std::popcount(unsigned(i & j)) % 2 ? -x[j] : x[j]); }
        y[i] /= 16.0;
    }
    return y;
}

// A coding's relative RMS error bound on Gaussian rows (loose: it catches layout errors, not
// accuracy regressions).
double error_bound(KvCoding c) {
    switch (c) {
    case KvCoding::Int8G64: return 0.02;
    case KvCoding::Fp8Row: return 0.06;
    case KvCoding::Int4G32:
    case KvCoding::Int4E8:
    case KvCoding::Lloyd4:
    case KvCoding::Nvfp4G16: return 0.25;
    case KvCoding::RootE8: return 0.5;
    case KvCoding::F16: break;
    }
    return 0.0;
}

const char* storage_name(KvCacheStorage s) {
    switch (s) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rk8v4";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value: return "rk4v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8: return "rk4v4-e8";
    case KvCacheStorage::RotatedE8RootKeyInt4Value: return "rk2v4-e8";
    }
    return "?";
}

// A device plane of `sequences` x `heads` x `slots` rows of `blocks` blocks, and its host decode.
struct Plane {
    k::KvPlaneView view;
    DeviceBuffer codes, scales;
    Plane(KvPlane format, int blocks, int slots, int heads, int sequences) {
        view.coding     = format.coding;
        view.rotated    = format.rotated;
        view.blocks     = blocks;
        view.code_row   = std::int64_t(blocks) * kv_code_bytes(format.coding);
        view.scale_row  = std::int64_t(blocks) * kv_scale_bytes(format.coding);
        view.code_head  = view.code_row * slots;
        view.scale_head = view.scale_row * slots;
        view.code_seq   = view.code_head * heads;
        view.scale_seq  = view.scale_head * heads;
        codes  = DeviceBuffer(std::size_t(view.code_seq) * sequences);
        scales = DeviceBuffer(std::size_t(view.scale_seq) * sequences);
        codes.fill(0);
        scales.fill(0);
        view.codes  = static_cast<std::uint8_t*>(codes.p);
        view.scales = static_cast<std::uint8_t*>(scales.p);
    }
    // Row (sequence, head, slot) decoded and unrotated: blocks * 256 values.
    std::vector<double> row(const std::vector<std::uint8_t>& hc, const std::vector<std::uint8_t>& hs,
                            int sequence, int head, int slot) const {
        std::vector<double> out;
        const std::uint8_t* c = hc.data() + sequence * view.code_seq + head * view.code_head + slot * view.code_row;
        const std::uint8_t* s = hs.data() + sequence * view.scale_seq + head * view.scale_head + slot * view.scale_row;
        for (int b = 0; b < view.blocks; ++b) {
            auto x = decode_block(view.coding, c + b * kv_code_bytes(view.coding),
                                  s + b * kv_scale_bytes(view.coding));
            if (view.rotated) { x = hadamard(x); }
            out.insert(out.end(), x.begin(), x.end());
        }
        return out;
    }
};

// One block kind of one format: rows of `positions` positions of two sequences stored, loaded,
// written back and attended.
struct Case {
    KvCacheStorage storage;
    bool full;            // D 512, one plane, keys derived; else D 256, ring 1280, window 1024
    std::vector<int> rows; // decode rows' positions (one per sequence)
};

void run(const Case& c, std::mt19937& rng) {
    const KvFormat format = kv_format(c.storage);
    const int d = c.full ? 512 : 256, heads = c.full ? 2 : 8, query_heads = 16;
    const int ring = c.full ? 0 : 1280, window = 1024, sequences = int(c.rows.size());
    const int max_pos = *std::max_element(c.rows.begin(), c.rows.end());
    const int slots = c.full ? max_pos + 1 : ring;
    const int blocks = d / 256;
    const std::string name = std::string(storage_name(c.storage)) + (c.full ? " full D512" : " sliding D256");
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Source rows: every position of every sequence a ring or a linear cache keeps, in F16,
    // [heads][rows][d] with rows = sequence-major positions (row r: sequence r / n, position r % n).
    const int n = max_pos + 1;
    const int rows = sequences * n;
    std::vector<__half> src_k(std::size_t(heads) * rows * d), src_v(src_k.size());
    for (auto& v : src_k) { v = __float2half(normal(rng)); }
    for (auto& v : src_v) { v = __float2half(normal(rng)); }
    std::vector<std::int32_t> positions(rows), seqs(rows);
    for (int r = 0; r < rows; ++r) {
        positions[r] = r % n;
        seqs[r]      = r / n;
    }
    DeviceBuffer dk(src_k.size() * 2), dv(src_v.size() * 2), dpos(rows * 4), dseq(rows * 4);
    dk.copy_from_host(src_k.data(), src_k.size() * 2);
    dv.copy_from_host(src_v.data(), src_v.size() * 2);
    dpos.copy_from_host(positions.data(), rows * 4);
    dseq.copy_from_host(seqs.data(), rows * 4);

    std::vector<Plane> planes;
    if (!c.full) { planes.emplace_back(format.key, blocks, slots, heads, sequences); }
    planes.emplace_back(c.full ? format.full : format.value, blocks, slots, heads, sequences);
    Plane* kp = c.full ? nullptr : &planes[0];
    Plane& vp = planes.back();
    // A ring keeps its last `ring` positions: the store skips the older ones of each sequence
    // (they would share slots). Store sequence by sequence so store_from applies per sequence.
    for (int s = 0; s < sequences; ++s) {
        for (Plane* p : {kp, &vp}) {
            if (p == nullptr) { continue; }
            k::KvStoreArgs a;
            a.src             = static_cast<__half*>((p == kp ? dk : dv).p) + std::size_t(s) * n * d;
            a.src_head_stride = std::int64_t(rows) * d;
            a.positions       = static_cast<const std::int32_t*>(dpos.p) + s * n;
            a.sequence        = s;
            a.rows            = n;
            a.heads           = heads;
            a.dst             = p->view;
            a.dst_ring        = ring;
            a.store_from      = ring != 0 ? n - ring : 0;
            k::kv_store(a, nullptr);
        }
    }
    check(cudaDeviceSynchronize(), "kv_store");

    // Host decode of every stored row, and its error against the source.
    std::vector<std::vector<std::uint8_t>> hc, hs;
    for (Plane& p : planes) {
        hc.emplace_back(p.codes.bytes);
        hs.emplace_back(p.scales.bytes);
        p.codes.copy_to_host(hc.back().data(), p.codes.bytes);
        p.scales.copy_to_host(hs.back().data(), p.scales.bytes);
    }
    const int lowest = ring != 0 ? std::max(0, n - ring) : 0;
    const auto decoded = [&](int plane, int s, int h, int pos) {
        return planes[plane].row(hc[plane], hs[plane], s, h, ring != 0 ? pos % ring : pos);
    };
    for (std::size_t pi = 0; pi < planes.size(); ++pi) {
        const auto& src = (kp != nullptr && pi == 0) ? src_k : src_v;
        double num = 0, den = 0;
        for (int s = 0; s < sequences; ++s) {
            for (int h = 0; h < heads; ++h) {
                for (int pos = lowest; pos < n; pos += 7) {
                    const auto x = decoded(int(pi), s, h, pos);
                    for (int i = 0; i < d; ++i) {
                        const double want = __half2float(src[(std::size_t(h) * rows + s * n + pos) * d + i]);
                        num += (x[i] - want) * (x[i] - want);
                        den += want * want;
                    }
                }
            }
        }
        const double rel = std::sqrt(num / den);
        report(rel < error_bound(planes[pi].view.coding),
               name + (planes.size() == 2 && pi == 0 ? " keys" : " values") + " stored, rel " + std::to_string(rel));
    }

    // kv_load: the device's F16 rows of sequence 1 equal the host decode.
    {
        const int s = sequences - 1;
        double worst = 0;
        for (std::size_t pi = 0; pi < planes.size(); ++pi) {
            DeviceBuffer out(std::size_t(heads) * n * d * 2);
            k::KvLoadArgs a;
            a.src             = planes[pi].view;
            a.src_ring        = ring;
            a.sequence        = s;
            a.begin           = lowest;
            a.end             = n;
            a.heads           = heads;
            a.dst             = static_cast<__half*>(out.p);
            a.dst_head_stride = std::int64_t(n) * d;
            k::kv_load(a, nullptr);
            check(cudaDeviceSynchronize(), "kv_load");
            std::vector<__half> got(std::size_t(heads) * n * d);
            out.copy_to_host(got.data(), got.size() * 2);
            for (int h = 0; h < heads; ++h) {
                for (int pos = lowest; pos < n; pos += 5) {
                    const auto x = decoded(int(pi), s, h, pos);
                    for (int i = 0; i < d; ++i) {
                        const double g = __half2float(got[(std::size_t(h) * n + pos) * d + i]);
                        worst = std::max(worst, std::abs(g - x[i]) / (1.0 + std::abs(x[i])));
                    }
                }
            }
        }
        report(worst < 4e-3, name + " kv_load equals the host decode, worst " + std::to_string(worst));
    }

    // Write-back: a prompt chunk's rows become their represented values.
    for (const KvPlane plane_format : c.full ? std::vector<KvPlane>{format.full}
                                             : std::vector<KvPlane>{format.key, format.value}) {
        std::vector<__half> chunk(std::size_t(heads) * 64 * d);
        for (auto& v : chunk) { v = __float2half(normal(rng)); }
        DeviceBuffer dc(chunk.size() * 2);
        dc.copy_from_host(chunk.data(), chunk.size() * 2);
        Plane scratch(plane_format, blocks, 64, heads, 1);
        k::KvStoreArgs a;
        a.src             = static_cast<__half*>(dc.p);
        a.src_head_stride = std::int64_t(64) * d;
        a.first           = 0;
        a.rows            = 64;
        a.heads           = heads;
        a.dst             = scratch.view;
        a.write_back      = true;
        k::kv_store(a, nullptr);
        check(cudaDeviceSynchronize(), "kv_store write-back");
        std::vector<__half> got(chunk.size());
        dc.copy_to_host(got.data(), got.size() * 2);
        std::vector<std::uint8_t> sc(scratch.codes.bytes), ss(scratch.scales.bytes);
        scratch.codes.copy_to_host(sc.data(), sc.size());
        scratch.scales.copy_to_host(ss.data(), ss.size());
        double worst = 0;
        for (int h = 0; h < heads; ++h) {
            for (int r = 0; r < 64; ++r) {
                const auto x = scratch.row(sc, ss, 0, h, r);
                for (int i = 0; i < d; ++i) {
                    const double g = __half2float(got[(std::size_t(h) * 64 + r) * d + i]);
                    worst = std::max(worst, std::abs(g - x[i]) / (1.0 + std::abs(x[i])));
                }
            }
        }
        report(worst < 4e-3, name + (plane_format.rotated ? " rotated" : "") +
                                 " write-back equals the host decode, worst " + std::to_string(worst));
    }

    // Decode attention over the planes, one row per sequence at c.rows[s].
    const int b = sequences;
    std::vector<float> q(std::size_t(b) * query_heads * d);
    for (auto& v : q) { v = normal(rng); }
    std::vector<float> k_norm(d), inv_freq(d / 2, 0.0f);
    for (auto& w : k_norm) { w = 1.0f + 0.25f * normal(rng); }
    for (int i = 0; i < 64 && i < d / 2; ++i) {
        inv_freq[i] = float(1.0 / std::pow(1e6, double(2 * i) / double(d)));
    }
    DeviceBuffer dw(k_norm.size() * 4), df(inv_freq.size() * 4), dr(std::size_t(slots) * 64 * 8);
    dw.copy_from_host(k_norm.data(), k_norm.size() * 4);
    df.copy_from_host(inv_freq.data(), inv_freq.size() * 4);
    k::rope_table(static_cast<const float*>(df.p), 64, slots, static_cast<float2*>(dr.p), nullptr);
    std::vector<std::int32_t> rpos(c.rows.begin(), c.rows.end()), rseq(b);
    for (int r = 0; r < b; ++r) { rseq[r] = r; }
    DeviceBuffer dq(q.size() * 4), drp(b * 4), drs(b * 4);
    dq.copy_from_host(q.data(), q.size() * 4);
    drp.copy_from_host(rpos.data(), b * 4);
    drs.copy_from_host(rseq.data(), b * 4);
    if (kp != nullptr && kp->view.rotated) {
        k::rotate_rows(static_cast<float*>(dq.p), b * query_heads, nullptr);
    }
    const int splits = c.full ? 37 : 16;
    DeviceBuffer partial(k::decode_attention_workspace_floats(b, query_heads, d, splits) * 4);
    DeviceBuffer out(std::size_t(b) * query_heads * d * 2);
    k::DecodeAttentionArgs a;
    a.q = static_cast<const float*>(dq.p);
    if (kp != nullptr) { a.k_plane = kp->view; }
    a.v_plane     = vp.view;
    a.k_norm      = static_cast<const float*>(dw.p);
    a.rope        = static_cast<const float2*>(dr.p);
    a.rope_pairs  = 64;
    a.sequences   = static_cast<const std::int32_t*>(drs.p);
    a.positions   = static_cast<const std::int32_t*>(drp.p);
    a.batch       = b;
    a.query_heads = query_heads;
    a.kv_heads    = heads;
    a.head_dim    = d;
    a.ring        = ring;
    a.window      = window;
    a.splits      = splits;
    a.scale       = 1.0f;
    a.partial     = static_cast<float*>(partial.p);
    a.out         = static_cast<__nv_bfloat16*>(out.p);
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention");
    std::vector<__nv_bfloat16> got(std::size_t(b) * query_heads * d);
    out.copy_to_host(got.data(), got.size() * 2);

    const int group = query_heads / heads;
    double num = 0, den = 0;
    for (int r = 0; r < b; ++r) {
        const int p  = c.rows[r];
        const int lo = ring != 0 ? std::max(0, p - window + 1) : 0;
        for (int h = 0; h < query_heads; ++h) {
            const int kv = h / group;
            const float* qh = q.data() + (std::size_t(r) * query_heads + h) * d;
            std::vector<std::vector<double>> vals;
            std::vector<double> score;
            double m = -1e300;
            for (int j = lo; j <= p; ++j) {
                const auto v = decoded(int(planes.size()) - 1, r, kv, j);
                std::vector<double> key;
                if (c.full) {
                    // RoPE(V * k_norm), rounded to F16 as the kernel derives it from the F16 value.
                    key.assign(d, 0.0);
                    for (int i = 0; i < d / 2; ++i) {
                        float co = 1.0f, si = 0.0f;
                        if (inv_freq[i] != 0.0f) {
                            const float angle = float(j) * inv_freq[i];
                            co                = std::cos(angle);
                            si                = std::sin(angle);
                        }
                        const double v0 = __half2float(__float2half(float(v[i])));
                        const double v1 = __half2float(__float2half(float(v[i + d / 2])));
                        const double n0 = v0 * k_norm[i], n1 = v1 * k_norm[i + d / 2];
                        key[i]         = __half2float(__float2half(float(n0 * co - n1 * si)));
                        key[i + d / 2] = __half2float(__float2half(float(n1 * co + n0 * si)));
                    }
                } else {
                    key = decoded(0, r, kv, j);
                }
                double s = 0;
                for (int i = 0; i < d; ++i) { s += double(qh[i]) * key[i]; }
                score.push_back(s);
                m = std::max(m, s);
                vals.push_back(v);
            }
            double l = 0;
            for (auto& s : score) { s = std::exp(s - m); l += s; }
            for (int i = 0; i < d; ++i) {
                double o = 0;
                for (std::size_t j = 0; j < vals.size(); ++j) { o += score[j] * vals[j][i]; }
                o /= l;
                const double e = double(__bfloat162float(got[(std::size_t(r) * query_heads + h) * d + i])) - o;
                num += e * e;
                den += o * o;
            }
        }
    }
    const double rms = std::sqrt(num / den);
    report(rms < 8e-3, name + " decode_attention rms " + std::to_string(rms));
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(91);
        for (const KvCacheStorage s :
             {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
              KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, KvCacheStorage::Nvfp4Group16,
              KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::RotatedLloyd4KeyInt4Value,
              KvCacheStorage::RotatedInt4KeyInt4ValueE8, KvCacheStorage::RotatedE8RootKeyInt4Value}) {
            // A ring past its first wrap (positions to 1500 in 1280 slots), rows near both ends.
            run({s, false, {1500, 1300}}, rng);
            run({s, true, {1100, 40}}, rng);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 KV formats\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma 4 KV formats: " << error.what() << '\n';
        return 1;
    }
}
