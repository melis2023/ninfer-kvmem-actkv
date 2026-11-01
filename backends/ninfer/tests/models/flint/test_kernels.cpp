// Flint's own kernels against FP64 oracles, at Flint's shapes (hidden 2048, 32 query heads over 4
// of width 128, 256 experts choosing 6, a 513-position window):
//  - decode_attention over an F16 cache with each query head's sink in its softmax: linear and
//    ring caches, one and many splits, short and long contexts, several sequences and a
//    verification's consecutive rows of one sequence; the BF16 rows and the q8_1 rows; sinks far
//    below the scores, near them and above them.
//  - qkv_post: per-head RMSNorm of Q and K, RoPE of the sliding blocks (none on the full ones), the
//    K and V rows at their cache slots (ring, two sequences, a branch pass's own slots).
//  - route and route_select: sigmoid scores of the top six logits renormalized, the expert
//    grouping of the pairs, the decode and the tiled prompt logits; route_select's logits, experts
//    and weights equal to route()'s bit for bit, its block count zero again after each launch.
//  - rms_norm and ffn_combine with their BF16, q8_1 and FP32 rows; the weighted FP32 expert rows
//    and the 2^-32 fixed-point sums.
//  - build_mask of a prompt chunk (linear and ring) and of a branch pass, alone or after a
//    prompt's last rows, ring_copy.
//  - swiglu_rows, alone and with the route weights taking the power of two of a row whose 32-value
//    sums pass the F16 limit; scale_columns and unscale_columns: the scaled values are the exact
//    multiples, the limit holds after them, and h returns bit for bit.
//  - quantize_pairs: the decode middles' guard (the same powers of two, the rows within the limit
//    and their weights untouched, a second stage's pass changing nothing) and q8_1 rows equal to
//    ggml's quantization of the rows it leaves, F16 sums finite.

#include "core/arena.h"
#include "models/flint/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace k = ninfer::models::flint::kernels;

constexpr int kD = 128, kHidden = 2048, kExperts = 256, kTopK = 6;

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

double q8_value(const std::vector<std::int8_t>& qs, const std::vector<half2>& ds, std::size_t at) {
    return double(qs[at]) * __low2float(ds[at / 32]);
}

struct AttentionCase {
    int query_heads, kv_heads, ring, window, splits;
    std::vector<int> positions; // one per batch row
    int group         = 1;      // rows per sequence: a verification's consecutive positions
    float sink_offset = 0.0f;   // added to every sink: far below or above the scores
};

void attention(const AttentionCase& c, std::mt19937& rng) {
    const int b       = int(c.positions.size());
    const int max_pos = *std::max_element(c.positions.begin(), c.positions.end());
    const int slots   = c.ring != 0 ? c.ring : max_pos + 1;
    const int seqs_n  = (b + c.group - 1) / c.group;
    const std::int64_t head_stride = std::int64_t(slots) * kD;
    const std::int64_t seq_stride  = head_stride * c.kv_heads;
    const float scale              = 1.0f / std::sqrt(float(kD));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__half> kc(seq_stride * seqs_n), vc(seq_stride * seqs_n);
    for (auto& v : kc) { v = __float2half(2.0f * normal(rng)); }
    for (auto& v : vc) { v = __float2half(normal(rng)); }
    std::vector<float> q(std::size_t(b) * c.query_heads * kD), sinks(c.query_heads);
    for (auto& v : q) { v = normal(rng); }
    for (auto& v : sinks) { v = 2.0f * normal(rng) + c.sink_offset; }
    std::vector<std::int32_t> seqs(b), pos(c.positions.begin(), c.positions.end());
    for (int r = 0; r < b; ++r) { seqs[r] = r / c.group; }

    DeviceBuffer dk(kc.size() * 2), dv(vc.size() * 2), dq(q.size() * 4), ds(b * 4), dp(b * 4),
        dsink(sinks.size() * 4);
    dk.copy_from_host(kc.data(), kc.size() * 2);
    dv.copy_from_host(vc.data(), vc.size() * 2);
    dq.copy_from_host(q.data(), q.size() * 4);
    ds.copy_from_host(seqs.data(), b * 4);
    dp.copy_from_host(pos.data(), b * 4);
    dsink.copy_from_host(sinks.data(), sinks.size() * 4);
    const std::size_t floats = k::decode_attention_workspace_floats(b, c.query_heads, kD, c.splits);
    const int width          = c.query_heads * kD;
    DeviceBuffer partial(floats * 4), out(std::size_t(b) * width * 2),
        q8(std::size_t(b) * width + std::size_t(b) * (width / 32) * 4 + 64);

    k::DecodeAttentionArgs a;
    a.q           = static_cast<const float*>(dq.p);
    a.k_cache     = static_cast<const __half*>(dk.p);
    a.v_cache     = static_cast<const __half*>(dv.p);
    a.sinks       = static_cast<const float*>(dsink.p);
    a.seq_stride  = seq_stride;
    a.head_stride = head_stride;
    a.sequences   = static_cast<const std::int32_t*>(ds.p);
    a.positions   = static_cast<const std::int32_t*>(dp.p);
    a.batch       = b;
    a.query_heads = c.query_heads;
    a.kv_heads    = c.kv_heads;
    a.head_dim    = kD;
    a.ring        = c.ring;
    a.window      = c.window;
    a.splits      = c.splits;
    a.scale       = scale;
    a.partial     = static_cast<float*>(partial.p);
    a.out         = static_cast<__nv_bfloat16*>(out.p);
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention");
    std::vector<__nv_bfloat16> got(std::size_t(b) * width);
    out.copy_to_host(got.data(), got.size() * 2);
    a.out      = nullptr;
    a.out_q.qs = static_cast<std::int8_t*>(q8.p);
    a.out_q.ds = reinterpret_cast<half2*>(static_cast<std::uint8_t*>(q8.p) + std::size_t(b) * width);
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention q8");
    std::vector<std::int8_t> got_q(std::size_t(b) * width);
    std::vector<half2> got_ds(std::size_t(b) * (width / 32));
    check(cudaMemcpy(got_q.data(), a.out_q.qs, got_q.size(), cudaMemcpyDeviceToHost), "q8 qs");
    check(cudaMemcpy(got_ds.data(), a.out_q.ds, got_ds.size() * 4, cudaMemcpyDeviceToHost), "q8 ds");

    const int group = c.query_heads / c.kv_heads;
    double num = 0, den = 0, qnum = 0;
    for (int r = 0; r < b; ++r) {
        const int p  = c.positions[r];
        const int lo = c.ring != 0 ? std::max(0, p - c.window + 1) : 0;
        for (int h = 0; h < c.query_heads; ++h) {
            const int kv    = h / group;
            const float* qh = q.data() + (std::size_t(r) * c.query_heads + h) * kD;
            const std::int64_t base = seqs[r] * seq_stride + kv * head_stride;
            std::vector<double> score;
            double m = sinks[h];
            for (int j = lo; j <= p; ++j) {
                const int slot   = c.ring != 0 ? j % c.ring : j;
                const __half* kr = kc.data() + base + std::int64_t(slot) * kD;
                double s         = 0;
                for (int i = 0; i < kD; ++i) { s += double(qh[i]) * __half2float(kr[i]); }
                score.push_back(s * scale);
                m = std::max(m, score.back());
            }
            // The sink is a key without a value: it joins the denominator only.
            double l = std::exp(double(sinks[h]) - m);
            for (auto& s : score) {
                s = std::exp(s - m);
                l += s;
            }
            for (int i = 0; i < kD; ++i) {
                double o = 0;
                for (int j = lo; j <= p; ++j) {
                    const int slot = c.ring != 0 ? j % c.ring : j;
                    o += score[j - lo] * __half2float(vc[base + std::int64_t(slot) * kD + i]);
                }
                o /= l;
                const std::size_t at = std::size_t(r) * width + std::size_t(h) * kD + i;
                const double e       = double(__bfloat162float(got[at])) - o;
                num += e * e;
                den += o * o;
                const double eq = q8_value(got_q, got_ds, at) - o;
                qnum += eq * eq;
            }
        }
    }
    const double rms = std::sqrt(num / den), qrms = std::sqrt(qnum / den);
    const std::string name =
        "decode_attention Hq " + std::to_string(c.query_heads) + " ring " + std::to_string(c.ring) +
        " splits " + std::to_string(c.splits) + " pos " + std::to_string(max_pos) +
        (c.group > 1 ? " rows of one sequence " + std::to_string(c.group) : "") +
        (c.sink_offset != 0.0f ? " sinks " + std::to_string(int(c.sink_offset)) : "") +
        " rms=" + std::to_string(rms) + " q8 rms=" + std::to_string(qrms);
    report(rms < 6e-3 && qrms < 1.5e-2, name);
}

// qkv_post against an FP64 oracle: Q and K RMS-normalized per head with their gammas, rotated
// when `rotate` (pairs (i, i + 64) at position * inv_freq[i], theta 10000), Q as FP32 rows, K and
// V as F16 rows at their slots: position (or stored[t]) modulo the ring, in their sequence.
void qkv(bool rotate, int ring, const std::vector<int>& positions, std::mt19937& rng,
         const std::vector<int>& stored = {}, int sequences = 1) {
    const std::vector<int>& at_slot = stored.empty() ? positions : stored;
    const int t = int(positions.size()), hq = 32, hkv = 4;
    const int rows  = (hq + 2 * hkv) * kD;
    const int slots = ring != 0 ? ring : *std::max_element(at_slot.begin(), at_slot.end()) + 1;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> qkv(std::size_t(t) * rows);
    for (auto& v : qkv) { v = 3.0f * normal(rng); }
    std::vector<float> q_norm(kD), k_norm(kD), inv_freq(kD / 2);
    for (auto& w : q_norm) { w = 1.0f + 0.25f * normal(rng); }
    for (auto& w : k_norm) { w = 1.0f + 0.25f * normal(rng); }
    for (int i = 0; i < kD / 2; ++i) {
        inv_freq[i] = float(1.0 / std::pow(10000.0, double(2 * i) / double(kD)));
    }
    std::vector<std::int32_t> pos(positions.begin(), positions.end()),
        slot_of(at_slot.begin(), at_slot.end()), seq(t);
    for (int i = 0; i < t; ++i) { seq[i] = i % sequences; }
    const std::int64_t head_stride = std::int64_t(slots) * kD, seq_stride = head_stride * hkv;
    const std::size_t cache        = std::size_t(seq_stride) * sequences;
    DeviceBuffer d_qkv(qkv.size() * 4), d_qn(kD * 4), d_kn(kD * 4), d_f(kD * 2), d_pos(t * 4),
        d_slots(t * 4), d_seq(t * 4), d_q(std::size_t(t) * hq * kD * 4), d_k(cache * 2),
        d_v(cache * 2);
    d_qkv.copy_from_host(qkv.data(), qkv.size() * 4);
    d_qn.copy_from_host(q_norm.data(), kD * 4);
    d_kn.copy_from_host(k_norm.data(), kD * 4);
    d_f.copy_from_host(inv_freq.data(), kD * 2);
    d_pos.copy_from_host(pos.data(), t * 4);
    d_slots.copy_from_host(slot_of.data(), t * 4);
    d_seq.copy_from_host(seq.data(), t * 4);
    d_k.fill(0);
    d_v.fill(0);
    k::QkvArgs a;
    a.qkv         = static_cast<const float*>(d_qkv.p);
    a.tokens      = t;
    a.query_heads = hq;
    a.kv_heads    = hkv;
    a.head_dim    = kD;
    a.q_norm      = static_cast<const float*>(d_qn.p);
    a.k_norm      = static_cast<const float*>(d_kn.p);
    a.eps         = 1e-6f;
    a.positions   = static_cast<const std::int32_t*>(d_pos.p);
    a.inv_freq    = rotate ? static_cast<const float*>(d_f.p) : nullptr;
    a.q_out       = static_cast<float*>(d_q.p);
    a.k_cache     = static_cast<__half*>(d_k.p);
    a.v_cache     = static_cast<__half*>(d_v.p);
    a.seq_stride  = seq_stride;
    a.head_stride = head_stride;
    a.sequences   = sequences > 1 ? static_cast<const std::int32_t*>(d_seq.p) : nullptr;
    a.ring        = ring;
    if (!stored.empty()) { a.slots = static_cast<const std::int32_t*>(d_slots.p); }
    k::qkv_post(a, nullptr);
    check(cudaDeviceSynchronize(), "qkv_post");
    std::vector<float> q(std::size_t(t) * hq * kD);
    std::vector<__half> kk(cache), vv(cache);
    d_q.copy_to_host(q.data(), q.size() * 4);
    d_k.copy_to_host(kk.data(), cache * 2);
    d_v.copy_to_host(vv.data(), cache * 2);
    double worst_q = 0.0, worst_k = 0.0;
    int wrong_v = 0;
    // One head of `x` normalized with `gamma` and rotated at `position`: element i.
    const auto expect = [&](const float* x, const std::vector<float>& gamma, int position, int i) {
        double sum = 0.0;
        for (int j = 0; j < kD; ++j) { sum += double(x[j]) * x[j]; }
        const double inv = 1.0 / std::sqrt(sum / kD + 1e-6);
        const int pair   = i % (kD / 2);
        const double n0 = x[pair] * inv * gamma[pair], n1 = x[pair + kD / 2] * inv * gamma[pair + kD / 2];
        double c = 1.0, s = 0.0;
        if (rotate) {
            const double angle = double(float(position) * inv_freq[pair]);
            c = std::cos(angle);
            s = std::sin(angle);
        }
        return i < kD / 2 ? n0 * c - n1 * s : n1 * c + n0 * s;
    };
    for (int token = 0; token < t; ++token) {
        const float* row = qkv.data() + std::size_t(token) * rows;
        for (int h = 0; h < hq; ++h) {
            for (int i = 0; i < kD; ++i) {
                const double e   = expect(row + h * kD, q_norm, positions[token], i);
                const double got = q[(std::size_t(token) * hq + h) * kD + i];
                worst_q          = std::max(worst_q, std::fabs(got - e) / (std::fabs(e) + 1e-2));
            }
        }
        const int slot = ring != 0 ? at_slot[token] % ring : at_slot[token];
        for (int h = 0; h < hkv; ++h) {
            const std::size_t at = seq[token] * seq_stride + h * head_stride + std::size_t(slot) * kD;
            for (int i = 0; i < kD; ++i) {
                const double e = expect(row + (hq + h) * kD, k_norm, positions[token], i);
                worst_k = std::max(worst_k, std::fabs(__half2float(kk[at + i]) - e) / (std::fabs(e) + 1e-1));
                const float v = row[(hq + hkv + h) * kD + i];
                wrong_v += __half2float(vv[at + i]) != __half2float(__float2half(v)) ? 1 : 0;
            }
        }
    }
    const std::string name = std::string("qkv_post ") + (rotate ? "rotated" : "unrotated") +
                             " ring " + std::to_string(ring) + " sequences " +
                             std::to_string(sequences) + (stored.empty() ? "" : " branch slots");
    report(worst_q <= 2e-5 && worst_k <= 2e-3 && wrong_v == 0,
           name + " q " + std::to_string(worst_q) + " k " + std::to_string(worst_k) + " v wrong " +
               std::to_string(wrong_v));
}

// The oracle's top six of one token: the logits by FP64 product, larger first, lower id on ties,
// with their sigmoid scores renormalized.
std::map<int, double> top_experts(const std::vector<float>& x, const std::vector<float>& router,
                                  int t) {
    std::vector<std::pair<double, int>> l(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        double s = 0;
        for (int i = 0; i < kHidden; ++i) {
            s += double(x[std::size_t(t) * kHidden + i]) * router[std::size_t(e) * kHidden + i];
        }
        l[e] = {s, e};
    }
    std::sort(l.begin(), l.end(), [](auto a, auto b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    double sum = 0;
    std::map<int, double> want;
    for (int j = 0; j < kTopK; ++j) {
        want[l[j].second] = 1.0 / (1.0 + std::exp(-l[j].first));
        sum += want[l[j].second];
    }
    for (auto& [e, w] : want) { w /= sum; }
    return want;
}

void route(int tokens, bool select, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(std::size_t(tokens) * kHidden), router(std::size_t(kExperts) * kHidden);
    for (auto& v : x) { v = normal(rng); }
    for (auto& v : router) { v = 0.02f * normal(rng); }
    const int pairs = tokens * kTopK;
    DeviceBuffer dx(x.size() * 4), dr(router.size() * 4),
        logits(std::size_t(tokens) * kExperts * 4), ids(pairs * 4), weights(pairs * 4),
        routing((kExperts + 1 + pairs + kExperts + 1) * 4), done(4);
    dx.copy_from_host(x.data(), x.size() * 4);
    dr.copy_from_host(router.data(), router.size() * 4);
    done.fill();
    auto* base = static_cast<std::int32_t*>(routing.p);
    if (select) {
        // Twice: the block count must be zero again after a launch.
        for (int launch = 0; launch < 2; ++launch) {
            k::route_select(static_cast<const float*>(dx.p), kHidden, tokens,
                            static_cast<const float*>(dr.p), kExperts, kTopK,
                            static_cast<float*>(logits.p), static_cast<std::int32_t*>(ids.p),
                            static_cast<float*>(weights.p), base, base + kExperts + 1,
                            base + kExperts + 1 + pairs, base + kExperts + 1 + pairs + kExperts,
                            static_cast<std::uint32_t*>(done.p), nullptr);
        }
    } else {
        k::route(static_cast<const float*>(dx.p), kHidden, tokens, static_cast<const float*>(dr.p),
                 kExperts, kTopK, static_cast<float*>(logits.p), static_cast<std::int32_t*>(ids.p),
                 static_cast<float*>(weights.p), nullptr);
    }
    check(cudaDeviceSynchronize(), "route");
    std::vector<std::int32_t> got_ids(pairs), arrays(kExperts + 1 + pairs + kExperts + 1);
    std::vector<float> got_w(pairs);
    ids.copy_to_host(got_ids.data(), pairs * 4);
    weights.copy_to_host(got_w.data(), pairs * 4);
    routing.copy_to_host(arrays.data(), arrays.size() * 4);
    bool ok = true;
    for (int t = 0; t < tokens; ++t) {
        // Near ties may order differently in FP32: compare as sets with their weights.
        std::map<int, double> want = top_experts(x, router, t);
        double sum                 = 0.0;
        for (int j = 0; j < kTopK; ++j) {
            const int e = got_ids[t * kTopK + j];
            sum += got_w[t * kTopK + j];
            if (!want.contains(e) || std::abs(want[e] - got_w[t * kTopK + j]) > 1e-5) { ok = false; }
        }
        ok = ok && std::abs(sum - 1.0) < 1e-5;
    }
    if (select) {
        // A decode step routes its rows as a verification pass of the same rows does.
        DeviceBuffer logits_r(logits.bytes), ids_r(ids.bytes), weights_r(weights.bytes);
        k::route(static_cast<const float*>(dx.p), kHidden, tokens, static_cast<const float*>(dr.p),
                 kExperts, kTopK, static_cast<float*>(logits_r.p),
                 static_cast<std::int32_t*>(ids_r.p), static_cast<float*>(weights_r.p), nullptr);
        check(cudaDeviceSynchronize(), "route");
        std::vector<std::byte> a(logits.bytes), b(logits.bytes);
        const auto same = [&](const DeviceBuffer& got, const DeviceBuffer& want) {
            a.resize(got.bytes);
            b.resize(want.bytes);
            got.copy_to_host(a.data(), got.bytes);
            want.copy_to_host(b.data(), want.bytes);
            return a == b;
        };
        ok = ok && same(logits, logits_r) && same(ids, ids_r) && same(weights, weights_r);
        std::uint32_t count_left = 1;
        done.copy_to_host(&count_left, 4);
        ok = ok && count_left == 0;
        // Every pair once, under its own expert; active experts ascending with their pairs.
        const std::int32_t* bounds = arrays.data();
        const std::int32_t* sorted = bounds + kExperts + 1;
        const std::int32_t* active = sorted + pairs;
        const std::int32_t count   = active[kExperts];
        std::set<int> seen;
        int listed = 0;
        for (int e = 0; e < kExperts; ++e) {
            const int n = bounds[e + 1] - bounds[e];
            if (n < 0) { ok = false; }
            if (n > 0) {
                if (listed >= count || active[listed] != e) { ok = false; }
                ++listed;
            }
            for (int i = bounds[e]; i < bounds[e + 1]; ++i) {
                if (got_ids[sorted[i]] != e || !seen.insert(sorted[i]).second) { ok = false; }
            }
        }
        ok = ok && listed == count && int(seen.size()) == pairs && bounds[0] == 0 &&
             bounds[kExperts] == pairs;
    }
    report(ok, std::string(select ? "route_select" : "route") + " T=" + std::to_string(tokens));
}

// rms_norm, then ffn_combine adding the routed rows (weighted FP32 rows, or fixed-point sums) to
// the residual and normalizing it for the next block: BF16, q8_1 and FP32 rows.
void norms(int tokens, bool fixed, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const std::size_t n = std::size_t(tokens) * kHidden, pairs = std::size_t(tokens) * kTopK;
    std::vector<float> h(n), w(kHidden), rows(pairs * kHidden), weights(pairs);
    std::vector<unsigned long long> sums(n);
    for (auto& v : h) { v = 4.0f * normal(rng); }
    for (auto& v : w) { v = 1.0f + 0.25f * normal(rng); }
    for (auto& v : rows) { v = normal(rng); }
    for (auto& v : weights) { v = std::abs(normal(rng)); }
    std::vector<double> moe(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        if (fixed) {
            const double value = normal(rng);
            const auto s       = static_cast<long long>(std::llround(value * 4294967296.0));
            sums[i]            = static_cast<unsigned long long>(s);
            moe[i]             = double(s) / 4294967296.0;
        } else {
            const std::size_t t = i / kHidden, c = i % kHidden;
            for (int j = 0; j < kTopK; ++j) {
                moe[i] += double(weights[t * kTopK + j]) * rows[(t * kTopK + j) * kHidden + c];
            }
        }
    }
    const std::size_t q8_bytes = n + (n / 32) * 4;
    DeviceBuffer dh(n * 4), dw(kHidden * 4), drows(rows.size() * 4), dweights(pairs * 4),
        dsums(n * 8), out(n * 2), q8(q8_bytes), f32(n * 4);
    dh.copy_from_host(h.data(), n * 4);
    dw.copy_from_host(w.data(), kHidden * 4);
    drows.copy_from_host(rows.data(), rows.size() * 4);
    dweights.copy_from_host(weights.data(), pairs * 4);
    dsums.copy_from_host(sums.data(), n * 8);
    const k::Q8Rows q{static_cast<std::int8_t*>(q8.p),
                      reinterpret_cast<half2*>(static_cast<std::uint8_t*>(q8.p) + n)};
    std::vector<__nv_bfloat16> got(n);
    std::vector<std::int8_t> got_q(n);
    std::vector<half2> got_ds(n / 32);
    std::vector<float> got_f(n);
    const auto read = [&]() {
        check(cudaDeviceSynchronize(), "norm");
        out.copy_to_host(got.data(), n * 2);
        q8.copy_to_host(got_q.data(), n);
        q8.copy_to_host(got_ds.data(), (n / 32) * 4, n);
    };
    // The normalized rows of `x` against the three outputs.
    const auto compare = [&](const std::vector<double>& x, bool with_f32, const std::string& name) {
        double num = 0, den = 0, qnum = 0, fnum = 0;
        for (int t = 0; t < tokens; ++t) {
            double sum = 0;
            for (int i = 0; i < kHidden; ++i) { sum += x[std::size_t(t) * kHidden + i] * x[std::size_t(t) * kHidden + i]; }
            const double inv = 1.0 / std::sqrt(sum / kHidden + 1e-6);
            for (int i = 0; i < kHidden; ++i) {
                const std::size_t at = std::size_t(t) * kHidden + i;
                const double e       = x[at] * inv * w[i];
                num += std::pow(__bfloat162float(got[at]) - e, 2);
                qnum += std::pow(q8_value(got_q, got_ds, at) - e, 2);
                if (with_f32) { fnum += std::pow(got_f[at] - e, 2); }
                den += e * e;
            }
        }
        const double rms = std::sqrt(num / den), qrms = std::sqrt(qnum / den),
                     frms = std::sqrt(fnum / den);
        report(rms < 4e-3 && qrms < 1.5e-2 && frms < 3e-6,
               name + " rms=" + std::to_string(rms) + " q8 rms=" + std::to_string(qrms) +
                   (with_f32 ? " f32 rms=" + std::to_string(frms) : ""));
    };

    k::rms_norm(static_cast<const float*>(dh.p), static_cast<const float*>(dw.p), kHidden, tokens,
                1e-6f, static_cast<__nv_bfloat16*>(out.p), nullptr, q, static_cast<float*>(f32.p));
    read();
    f32.copy_to_host(got_f.data(), n * 4);
    compare(std::vector<double>(h.begin(), h.end()), true, "rms_norm T=" + std::to_string(tokens));

    k::FfnCombineArgs a;
    a.h      = static_cast<float*>(dh.p);
    a.hidden = kHidden;
    a.tokens = tokens;
    a.eps    = 1e-6f;
    if (fixed) {
        a.moe_fixed = static_cast<const unsigned long long*>(dsums.p);
    } else {
        a.moe_rows    = static_cast<const float*>(drows.p);
        a.moe_weights = static_cast<const float*>(dweights.p);
        a.top_k       = kTopK;
    }
    a.next_norm = static_cast<const float*>(dw.p);
    a.next_out  = static_cast<__nv_bfloat16*>(out.p);
    a.next_q    = q;
    k::ffn_combine(a, nullptr);
    read();
    std::vector<float> got_h(n);
    dh.copy_to_host(got_h.data(), n * 4);
    std::vector<double> want_h(n);
    double hnum = 0, hden = 0;
    for (std::size_t i = 0; i < n; ++i) {
        want_h[i] = double(h[i]) + moe[i];
        hnum += std::pow(got_h[i] - want_h[i], 2);
        hden += want_h[i] * want_h[i];
    }
    report(std::sqrt(hnum / hden) < 1e-6, std::string("ffn_combine residual ") +
                                               (fixed ? "fixed-point sums" : "weighted rows") +
                                               " T=" + std::to_string(tokens));
    compare(want_h, false, std::string("ffn_combine next norm ") + (fixed ? "fixed" : "rows") +
                               " T=" + std::to_string(tokens));
}

// The prompt mask of a chunk of `tokens` at `first` (ring 0: a linear cache) and, with branch
// starts, of a branch pass: a branch row sees the positions before `first` as its own position
// does and the rows of its branch up to itself. A ring slot holds the latest position written.
// With `branches`, a branch pass of three branches; `shared` prompt rows run ahead of them.
void mask(int tokens, int first, int ring, int window, bool branches, int shared = 0) {
    std::vector<std::int32_t> starts;
    if (branches) {
        for (int row = 0; row < shared; ++row) { starts.push_back(row); }
        int row = shared;
        for (const int length : {5, 8, 7}) {
            for (int j = 0; j < length; ++j) { starts.push_back(row); }
            row += length;
        }
        tokens = int(starts.size());
    }
    const int last  = first + tokens - 1;
    const int slots = ring != 0 ? ring : last + 1 + 37;
    std::vector<int> held(slots, -1);
    for (int p = 0; p <= last; ++p) { held[ring != 0 ? p % ring : p] = p; }
    DeviceBuffer dstarts(std::max<std::size_t>(starts.size(), 1) * 4),
        dmask(std::size_t(tokens) * slots * 2);
    if (branches) { dstarts.copy_from_host(starts.data(), starts.size() * 4); }
    k::build_mask(static_cast<__half*>(dmask.p), tokens, slots, first, ring, window, nullptr,
                  branches ? static_cast<const std::int32_t*>(dstarts.p) : nullptr, shared);
    check(cudaDeviceSynchronize(), "build_mask");
    std::vector<__half> m(std::size_t(tokens) * slots);
    dmask.copy_to_host(m.data(), m.size() * 2);
    int wrong = 0;
    const int base = first + shared;
    for (int t = 0; t < tokens; ++t) {
        const bool branch  = branches && t >= shared;
        const int position = branch ? base + t - starts[t] : first + t;
        for (int s = 0; s < slots; ++s) {
            const int p = held[s];
            bool want   = false;
            if (p >= 0 && (!branch || p < base)) {
                want = p <= position && (ring == 0 || p > position - window);
            } else if (p >= base) {
                const int row = p - first;
                want = starts[row] == starts[t] && row <= t && (ring == 0 || t - row < window);
            }
            const float v = __half2float(m[std::size_t(t) * slots + s]);
            wrong += (v == 0.0f) != want || (!want && !std::isinf(v)) ? 1 : 0;
        }
    }
    report(wrong == 0, "build_mask T=" + std::to_string(tokens) + " first " + std::to_string(first) +
                           " ring " + std::to_string(ring) + (branches ? " branches" : "") +
                           (shared ? " after " + std::to_string(shared) + " prompt rows" : "") +
                           ": " + std::to_string(wrong) + " wrong");
}

// ring_copy: the rows of positions [begin, end) from one ring into another, both bit for bit.
void ring_copy(std::mt19937& rng) {
    constexpr int heads = 4, src_ring = 1280, dst_ring = 768, begin = 1500, end = 2013;
    std::uniform_int_distribution<int> bits(0, 0x7bff);
    std::vector<__half> sk(std::size_t(heads) * src_ring * kD), sv(sk.size());
    for (auto& v : sk) { v = __ushort_as_half(static_cast<unsigned short>(bits(rng))); }
    for (auto& v : sv) { v = __ushort_as_half(static_cast<unsigned short>(bits(rng))); }
    const std::size_t dst = std::size_t(heads) * dst_ring * kD;
    DeviceBuffer dsk(sk.size() * 2), dsv(sv.size() * 2), ddk(dst * 2), ddv(dst * 2);
    dsk.copy_from_host(sk.data(), sk.size() * 2);
    dsv.copy_from_host(sv.data(), sv.size() * 2);
    ddk.fill(0);
    ddv.fill(0);
    k::RingCopyArgs a;
    a.src_k           = static_cast<const __half*>(dsk.p);
    a.src_v           = static_cast<const __half*>(dsv.p);
    a.dst_k           = static_cast<__half*>(ddk.p);
    a.dst_v           = static_cast<__half*>(ddv.p);
    a.src_head_stride = std::int64_t(src_ring) * kD;
    a.dst_head_stride = std::int64_t(dst_ring) * kD;
    a.src_ring        = src_ring;
    a.dst_ring        = dst_ring;
    a.kv_heads        = heads;
    a.head_dim        = kD;
    a.begin           = begin;
    a.end             = end;
    k::ring_copy(a, nullptr);
    check(cudaDeviceSynchronize(), "ring_copy");
    std::vector<__half> dk(dst), dv(dst);
    ddk.copy_to_host(dk.data(), dst * 2);
    ddv.copy_to_host(dv.data(), dst * 2);
    int wrong = 0;
    for (int h = 0; h < heads; ++h) {
        for (int p = begin; p < end; ++p) {
            const std::size_t s = (std::size_t(h) * src_ring + p % src_ring) * kD;
            const std::size_t d = (std::size_t(h) * dst_ring + p % dst_ring) * kD;
            for (int i = 0; i < kD; ++i) {
                wrong += __half_as_ushort(dk[d + i]) != __half_as_ushort(sk[s + i]) ? 1 : 0;
                wrong += __half_as_ushort(dv[d + i]) != __half_as_ushort(sv[s + i]) ? 1 : 0;
            }
        }
    }
    report(wrong == 0, "ring_copy " + std::to_string(wrong) + " wrong");
}

void swiglu(std::mt19937& rng) {
    constexpr int pairs = 13, width = 1024;
    std::normal_distribution<float> normal(0.0f, 3.0f);
    std::vector<float> x(std::size_t(pairs) * 2 * width);
    for (auto& v : x) { v = normal(rng); }
    DeviceBuffer dx(x.size() * 4), out(std::size_t(pairs) * width * 2);
    dx.copy_from_host(x.data(), x.size() * 4);
    k::swiglu_rows(static_cast<const float*>(dx.p), pairs, width,
                   static_cast<__nv_bfloat16*>(out.p), nullptr, nullptr);
    check(cudaDeviceSynchronize(), "swiglu_rows");
    std::vector<__nv_bfloat16> got(std::size_t(pairs) * width);
    out.copy_to_host(got.data(), got.size() * 2);
    double worst = 0;
    for (int p = 0; p < pairs; ++p) {
        for (int r = 0; r < width; ++r) {
            const double g = x[std::size_t(p) * 2 * width + r];
            const double e = g / (1.0 + std::exp(-g)) * x[std::size_t(p) * 2 * width + width + r];
            worst = std::max(worst, std::fabs(__bfloat162float(got[std::size_t(p) * width + r]) - e) /
                                        (std::fabs(e) + 1e-2));
        }
    }
    report(worst < 8e-3, "swiglu_rows worst " + std::to_string(worst));
}

double largest_sum(const __nv_bfloat16* row, int n) {
    double largest = 0;
    for (int b = 0; b < n; b += 32) {
        double sum = 0;
        for (int i = b; i < b + 32; ++i) { sum += __bfloat162float(row[i]); }
        largest = std::max(largest, std::fabs(sum));
    }
    return largest;
}

// k with value == unscaled * 2^-k exactly for every value of a row, or -1.
int exact_exponent(const __nv_bfloat16* scaled, const __nv_bfloat16* unscaled, int n) {
    for (int k = 0; k < 32; ++k) {
        bool all = true;
        for (int i = 0; i < n && all; ++i) {
            all = __bfloat162float(scaled[i]) == std::ldexp(__bfloat162float(unscaled[i]), -k);
        }
        if (all) { return k; }
    }
    return -1;
}

// Routed SwiGLU rows at Flint's width (512), some of them past the F16 sum limit: each such row is
// the unscaled row times 2^-k with its largest sum under the limit and its weight times 2^k; every
// other row and weight is unchanged.
void swiglu_guard(std::mt19937& rng) {
    constexpr int pairs = 24, width = 512;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(std::size_t(pairs) * 2 * width), weights(pairs);
    for (int p = 0; p < pairs; ++p) {
        // Gates and ups of magnitude up to 300 give values near 90,000, as in Flint's block 46.
        const float spread = p % 3 == 0 ? 300.0f : p % 3 == 1 ? 40.0f : 2.0f;
        for (int r = 0; r < 2 * width; ++r) {
            x[std::size_t(p) * 2 * width + r] = spread * std::fabs(normal(rng)) * (r % 7 == 0 ? -1 : 1);
        }
        weights[p] = 0.05f + 0.9f * std::fabs(normal(rng)) / 3.0f;
    }
    DeviceBuffer dx(x.size() * 4), plain(std::size_t(pairs) * width * 2),
        guarded(std::size_t(pairs) * width * 2), dw(pairs * 4);
    dx.copy_from_host(x.data(), x.size() * 4);
    dw.copy_from_host(weights.data(), pairs * 4);
    k::swiglu_rows(static_cast<const float*>(dx.p), pairs, width,
                   static_cast<__nv_bfloat16*>(plain.p), nullptr, nullptr);
    k::swiglu_rows(static_cast<const float*>(dx.p), pairs, width,
                   static_cast<__nv_bfloat16*>(guarded.p), static_cast<float*>(dw.p), nullptr);
    check(cudaDeviceSynchronize(), "swiglu_rows guard");
    std::vector<__nv_bfloat16> a(std::size_t(pairs) * width), b(a.size());
    std::vector<float> w(pairs);
    plain.copy_to_host(a.data(), a.size() * 2);
    guarded.copy_to_host(b.data(), b.size() * 2);
    dw.copy_to_host(w.data(), pairs * 4);
    int wrong = 0, scaled = 0;
    for (int p = 0; p < pairs; ++p) {
        const __nv_bfloat16* ra = a.data() + std::size_t(p) * width;
        const __nv_bfloat16* rb = b.data() + std::size_t(p) * width;
        const double before = largest_sum(ra, width);
        const int e         = exact_exponent(rb, ra, width);
        const int expected  = before > k::kF16SumLimit ? int(std::ceil(std::log2(before / k::kF16SumLimit))) : 0;
        // The kernel sums in another order; its k may differ by one where a sum sits at a power of two.
        const bool ok = e >= 0 && std::abs(e - expected) <= 1 && largest_sum(rb, width) <= k::kF16SumLimit &&
                        w[p] == std::ldexp(weights[p], e) && (before > k::kF16SumLimit / 2 || e == 0);
        wrong += ok ? 0 : 1;
        scaled += e > 0 ? 1 : 0;
    }
    report(wrong == 0 && scaled >= pairs / 3,
           "swiglu_rows guard " + std::to_string(scaled) + " rows scaled, " + std::to_string(wrong) + " wrong");
}

// ggml's q8_1 of one 32-value group of BF16 values: the codes, d and the warp butterfly's sum, both
// rounded to F16.
void q8_group(const __nv_bfloat16* v, std::int8_t* codes, half2& ds) {
    float lanes[32], amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        lanes[i] = __bfloat162float(v[i]);
        amax     = std::max(amax, std::fabs(lanes[i]));
    }
    const float d = amax / 127.0f;
    for (int i = 0; i < 32; ++i) {
        codes[i] = amax == 0.0f ? 0 : static_cast<std::int8_t>(std::round(lanes[i] / d));
    }
    for (int o = 16; o > 0; o >>= 1) {
        for (int l = 0; l < o; ++l) { lanes[l] += lanes[l + o]; }
    }
    ds = __floats2half2_rn(d, lanes[0]);
}

// Decode middles at Flint's width (512), a third of them past the F16 sum limit, quantized with the
// guard twice (the two stages of a call with experts in transit).
void pair_guard(std::mt19937& rng) {
    constexpr int pairs = 30, width = 512, groups = width / 32;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> middle(std::size_t(pairs) * width);
    std::vector<float> weights(pairs);
    for (int p = 0; p < pairs; ++p) {
        const float spread = p % 3 == 0 ? 6000.0f : p % 3 == 1 ? 300.0f : 1.0f;
        for (int r = 0; r < width; ++r) {
            middle[std::size_t(p) * width + r] =
                __float2bfloat16(spread * std::fabs(normal(rng)) * (r % 5 == 0 ? -1 : 1));
        }
        weights[p] = 0.05f + 0.9f * std::fabs(normal(rng)) / 3.0f;
    }
    const std::size_t qs_bytes = std::size_t(pairs) * width,
                      ds_bytes = std::size_t(pairs) * groups * 4;
    DeviceBuffer dm(middle.size() * 2), dw(pairs * 4), dq(qs_bytes), dd(ds_bytes), dq2(qs_bytes),
        dd2(ds_bytes);
    dm.copy_from_host(middle.data(), middle.size() * 2);
    dw.copy_from_host(weights.data(), pairs * 4);
    const k::Q8Rows first{static_cast<std::int8_t*>(dq.p), static_cast<half2*>(dd.p)};
    const k::Q8Rows second{static_cast<std::int8_t*>(dq2.p), static_cast<half2*>(dd2.p)};
    k::quantize_pairs(static_cast<__nv_bfloat16*>(dm.p), pairs, width, static_cast<float*>(dw.p),
                      first, nullptr);
    std::vector<__nv_bfloat16> once(middle.size()), twice(middle.size());
    std::vector<float> w_once(pairs), w_twice(pairs);
    check(cudaDeviceSynchronize(), "quantize_pairs");
    dm.copy_to_host(once.data(), once.size() * 2);
    dw.copy_to_host(w_once.data(), pairs * 4);
    k::quantize_pairs(static_cast<__nv_bfloat16*>(dm.p), pairs, width, static_cast<float*>(dw.p),
                      second, nullptr);
    check(cudaDeviceSynchronize(), "quantize_pairs again");
    dm.copy_to_host(twice.data(), twice.size() * 2);
    dw.copy_to_host(w_twice.data(), pairs * 4);
    std::vector<std::int8_t> q(qs_bytes), q2(qs_bytes);
    std::vector<half2> d(std::size_t(pairs) * groups), d2(d.size());
    dq.copy_to_host(q.data(), qs_bytes);
    dd.copy_to_host(d.data(), ds_bytes);
    dq2.copy_to_host(q2.data(), qs_bytes);
    dd2.copy_to_host(d2.data(), ds_bytes);
    int wrong = 0, scaled = 0;
    for (int p = 0; p < pairs; ++p) {
        const __nv_bfloat16* before = middle.data() + std::size_t(p) * width;
        const __nv_bfloat16* after  = once.data() + std::size_t(p) * width;
        const int e                 = exact_exponent(after, before, width);
        bool ok = e >= 0 && w_once[p] == std::ldexp(weights[p], e) && w_twice[p] == w_once[p] &&
                  largest_sum(after, width) <= k::kF16SumLimit &&
                  (largest_sum(before, width) > k::kF16SumLimit / 2 || e == 0);
        for (int r = 0; r < width && ok; ++r) {
            const std::size_t i = std::size_t(p) * width + r;
            ok = __bfloat162float(twice[i]) == __bfloat162float(once[i]) && q2[i] == q[i];
        }
        for (int g = 0; g < groups && ok; ++g) {
            std::int8_t codes[32];
            half2 ds;
            q8_group(after + 32 * g, codes, ds);
            const std::size_t at = std::size_t(p) * groups + g;
            ok = std::equal(codes, codes + 32, q.begin() + std::ptrdiff_t(at * 32)) &&
                 __half2float(__low2half(d[at])) == __half2float(__low2half(ds)) &&
                 __half2float(__high2half(d[at])) == __half2float(__high2half(ds)) &&
                 std::isfinite(__half2float(__high2half(d[at]))) &&
                 __half2float(__high2half(d2[at])) == __half2float(__high2half(d[at]));
        }
        wrong += ok ? 0 : 1;
        scaled += e > 0 ? 1 : 0;
    }
    report(wrong == 0 && scaled >= pairs / 3, "quantize_pairs " + std::to_string(scaled) +
                                                  " rows scaled, " + std::to_string(wrong) +
                                                  " wrong");
}

// Columns of a prompt product's activation, some past the F16 sum limit, with their residual columns.
void column_guard(std::mt19937& rng) {
    constexpr int columns = 37, rows = 1024;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(std::size_t(columns) * rows);
    std::vector<float> h(std::size_t(columns) * kHidden);
    for (int c = 0; c < columns; ++c) {
        const float spread = c % 4 == 0 ? 5000.0f : c % 4 == 1 ? 400.0f : 1.0f;
        for (int r = 0; r < rows; ++r) {
            x[std::size_t(c) * rows + r] = __float2bfloat16(spread * (std::fabs(normal(rng)) + 0.1f));
        }
    }
    for (auto& v : h) { v = 30.0f * normal(rng); }
    DeviceBuffer dx(x.size() * 2), dh(h.size() * 4), scale(columns * 4);
    dx.copy_from_host(x.data(), x.size() * 2);
    dh.copy_from_host(h.data(), h.size() * 4);
    k::scale_columns(static_cast<__nv_bfloat16*>(dx.p), rows, columns, static_cast<float*>(dh.p),
                     kHidden, static_cast<float*>(scale.p), nullptr);
    check(cudaDeviceSynchronize(), "scale_columns");
    std::vector<__nv_bfloat16> xs(x.size());
    std::vector<float> hs(h.size()), s(columns);
    dx.copy_to_host(xs.data(), xs.size() * 2);
    dh.copy_to_host(hs.data(), hs.size() * 4);
    scale.copy_to_host(s.data(), columns * 4);
    k::unscale_columns(static_cast<float*>(dh.p), kHidden, columns, static_cast<const float*>(scale.p),
                       nullptr);
    check(cudaDeviceSynchronize(), "unscale_columns");
    std::vector<float> back(h.size());
    dh.copy_to_host(back.data(), back.size() * 4);
    int wrong = 0, scaled = 0;
    for (int c = 0; c < columns; ++c) {
        const __nv_bfloat16* before = x.data() + std::size_t(c) * rows;
        const __nv_bfloat16* after  = xs.data() + std::size_t(c) * rows;
        const int e = exact_exponent(after, before, rows);
        bool ok     = e >= 0 && s[c] == std::ldexp(1.0f, e) && largest_sum(after, rows) <= k::kF16SumLimit &&
                  (largest_sum(before, rows) > k::kF16SumLimit / 2 || e == 0);
        for (int r = 0; r < kHidden && ok; ++r) {
            const std::size_t i = std::size_t(c) * kHidden + r;
            ok = hs[i] == std::ldexp(h[i], -e) && back[i] == h[i];
        }
        wrong += ok ? 0 : 1;
        scaled += e > 0 ? 1 : 0;
    }
    report(wrong == 0 && scaled >= columns / 4,
           "scale_columns " + std::to_string(scaled) + " columns scaled, " + std::to_string(wrong) + " wrong");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(1913);
        // Sliding blocks: a 513-position window in the decode ring (768 slots), short and long
        // contexts, one and 16 splits, two sequences.
        attention({32, 4, 768, 513, 16, {0}}, rng);
        attention({32, 4, 768, 513, 16, {5, 512}}, rng);
        attention({32, 4, 768, 513, 16, {513, 4000}}, rng);
        attention({32, 4, 768, 513, 1, {3001}}, rng);
        // A verification's rows: one sequence's consecutive positions, across the window's edge.
        attention({32, 4, 768, 513, 16, {511, 512, 513, 514, 515, 516, 517, 518}, 8}, rng);
        attention({32, 4, 768, 513, 16, {2000, 2001, 2002, 900, 901, 902}, 3}, rng);
        // Full blocks: a linear cache, splits from one to 128 with empty ones past the keys.
        attention({32, 4, 0, 0, 1, {0}}, rng);
        attention({32, 4, 0, 0, 64, {40, 4999}}, rng);
        attention({32, 4, 0, 0, 128, {6000}}, rng);
        attention({32, 4, 0, 0, 60, {999}}, rng);
        attention({8, 1, 0, 0, 7, {100, 101, 102, 103}, 4}, rng);
        // Sinks far below the scores (no weight), near them, and above them (most of the weight;
        // far above, the output's q8_1 scale would underflow F16, as any tiny row's does).
        attention({32, 4, 768, 513, 16, {700}, 1, -40.0f}, rng);
        attention({32, 4, 0, 0, 16, {3, 300}, 1, 8.0f}, rng);
        attention({32, 4, 0, 0, 16, {3, 300}, 1, 10.0f}, rng);

        qkv(true, 768, {0, 1, 700, 4000, 4001}, rng);
        qkv(false, 0, {0, 1, 700, 4000, 4001}, rng);
        qkv(true, 768, {10, 10, 11, 11}, rng, {}, 2);
        // A branch pass: rows rotate at their branch positions and store at their own slots.
        qkv(true, 768, {300, 301, 302, 300, 301}, rng, {300, 301, 302, 303, 304});
        qkv(false, 0, {300, 301, 302, 300, 301}, rng, {300, 301, 302, 303, 304});

        for (const int tokens : {1, 3, 8}) { route(tokens, true, rng); }
        for (const int tokens : {1, 8, 9, 100}) { route(tokens, false, rng); }

        norms(1, false, rng);
        norms(5, true, rng);
        norms(70, false, rng);

        mask(64, 0, 0, 0, false);
        mask(64, 1000, 0, 0, false);
        mask(512, 300, 1536, 513, false);
        mask(512, 2000, 1536, 513, false);
        mask(0, 300, 1024, 513, true);
        mask(0, 1500, 1024, 513, true);
        mask(0, 300, 0, 0, true);
        // A prompt's last rows and the branches after them in one pass: the branches see the
        // prompt rows, not each other (after 700 of them, a branch row's window ends among them).
        mask(0, 300, 1024, 513, true, 40);
        mask(0, 1500, 1536, 513, true, 700);
        mask(0, 300, 0, 0, true, 40);
        mask(0, 0, 1024, 513, true, 1);
        ring_copy(rng);
        swiglu(rng);
        swiglu_guard(rng);
        pair_guard(rng);
        column_guard(rng);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flint kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flint kernels: " << error.what() << '\n';
        return 1;
    }
}
