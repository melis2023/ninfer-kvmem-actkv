// Gemma 4's own decode kernels against FP64 oracles:
//  - decode_attention over an F16 cache, both head geometries (D 256 with two query heads per KV
//    head, D 512 with eight), linear and sliding-window ring caches, one and many splits, short
//    and long contexts and two sequences in one batch; the BF16 rows and the q8_1 rows; and keys
//    derived from the values (K = V blocks: RoPE(V * k_norm), proportional over 64 of 256 pairs).
//  - route_select: the logits, the top-k ids and weights and the expert grouping of the pairs.
//  - a branch pass's rows (several branches one after another past the state, each continuing it
//    at the state's end): qkv_post's rotation at the branch position with the value stored at the
//    row's own slot, keys derived at the branch positions, and the prompt masks of a sliding ring
//    and of a linear cache, where a row sees the state as its position does and its own branch.

#include "core/arena.h"
#include "models/gemma4/kernels.h"
#include "ninfer/ops/rope.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace k = ninfer::models::gemma4::kernels;

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

struct AttentionCase {
    int head_dim, query_heads, kv_heads, ring, window, splits;
    std::vector<int> positions; // one per batch row
    bool derived = false;       // keys derived from the values
    int group    = 1;           // rows per sequence: a verification's consecutive positions
    float yarn   = 1.0f;        // above one: the derived keys turn at YaRN's frequencies
};

void attention(const AttentionCase& c, std::mt19937& rng) {
    const int d = c.head_dim, b = int(c.positions.size());
    const int max_pos = *std::max_element(c.positions.begin(), c.positions.end());
    const int slots   = c.ring != 0 ? c.ring : max_pos + 1;
    const std::int64_t head_stride = std::int64_t(slots) * d;
    const std::int64_t seq_stride  = head_stride * c.kv_heads;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__half> kc(seq_stride * b), vc(seq_stride * b);
    for (auto& v : kc) { v = __float2half(0.25f * normal(rng)); }
    for (auto& v : vc) { v = __float2half(normal(rng)); }
    std::vector<float> q(std::size_t(b) * c.query_heads * d);
    for (auto& v : q) { v = normal(rng); }
    std::vector<std::int32_t> seqs(b), pos(c.positions.begin(), c.positions.end());
    for (int r = 0; r < b; ++r) { seqs[r] = r / c.group; }
    // Derived keys: k_norm gammas near one, Gemma's full-attention frequencies (theta 1e6, the
    // first 64 pairs turn), and the oracle's keys RoPE(V * k_norm) rounded to F16 from the stored V.
    std::vector<float> k_norm(d), inv_freq(d / 2, 0.0f);
    for (auto& w : k_norm) { w = 1.0f + 0.25f * normal(rng); }
    if (c.yarn > 1.0f) {
        const ninfer::ops::RopeYarnFrequencies yarn =
            ninfer::ops::rope_yarn_frequencies(d, std::min(64, d / 2), 1e6f, {c.yarn, 262144});
        std::copy_n(yarn.inverse_frequency, yarn.pairs, inv_freq.begin());
    } else {
        for (int i = 0; i < 64 && i < d / 2; ++i) {
            inv_freq[i] = float(1.0 / std::pow(1e6, double(2 * i) / double(d)));
        }
    }
    if (c.derived) {
        for (int r = 0; r < b; ++r) {
            for (int kv = 0; kv < c.kv_heads; ++kv) {
                for (int j = 0; j < slots; ++j) {
                    const std::int64_t at = r * seq_stride + kv * head_stride + std::int64_t(j) * d;
                    for (int i = 0; i < d / 2; ++i) {
                        float co = 1.0f, si = 0.0f;
                        if (inv_freq[i] != 0.0f) {
                            const float angle = float(j) * inv_freq[i];
                            co                = std::cos(angle);
                            si                = std::sin(angle);
                        }
                        const double n0 = double(__half2float(vc[at + i])) * k_norm[i];
                        const double n1 = double(__half2float(vc[at + i + d / 2])) * k_norm[i + d / 2];
                        kc[at + i]         = __float2half(float(n0 * co - n1 * si));
                        kc[at + i + d / 2] = __float2half(float(n1 * co + n0 * si));
                    }
                }
            }
        }
    }
    DeviceBuffer dw(k_norm.size() * 4), df(inv_freq.size() * 4), dr(std::size_t(slots) * 64 * 8);
    dw.copy_from_host(k_norm.data(), k_norm.size() * 4);
    df.copy_from_host(inv_freq.data(), inv_freq.size() * 4);
    k::rope_table(static_cast<const float*>(df.p), 64, slots, static_cast<float2*>(dr.p), nullptr);

    DeviceBuffer dk(kc.size() * 2), dv(vc.size() * 2), dq(q.size() * 4), ds(b * 4), dp(b * 4);
    dk.copy_from_host(kc.data(), kc.size() * 2);
    dv.copy_from_host(vc.data(), vc.size() * 2);
    dq.copy_from_host(q.data(), q.size() * 4);
    ds.copy_from_host(seqs.data(), b * 4);
    dp.copy_from_host(pos.data(), b * 4);
    const std::size_t floats =
        k::decode_attention_workspace_floats(b, c.query_heads, d, c.splits);
    DeviceBuffer partial(floats * 4), out(std::size_t(b) * c.query_heads * d * 2);
    const int width = c.query_heads * d;
    DeviceBuffer q8(std::size_t(b) * width + std::size_t(b) * (width / 32) * 4 + 64);

    k::DecodeAttentionArgs a;
    a.q = static_cast<const float*>(dq.p);
    a.k_cache = c.derived ? nullptr : static_cast<const __half*>(dk.p);
    a.k_norm = static_cast<const float*>(dw.p);
    a.rope = static_cast<const float2*>(dr.p);
    a.rope_pairs = 64;
    a.v_cache = static_cast<const __half*>(dv.p);
    a.seq_stride = seq_stride;
    a.head_stride = head_stride;
    a.sequences = static_cast<const std::int32_t*>(ds.p);
    a.positions = static_cast<const std::int32_t*>(dp.p);
    a.batch = b;
    a.query_heads = c.query_heads;
    a.kv_heads = c.kv_heads;
    a.head_dim = d;
    a.ring = c.ring;
    a.window = c.window;
    a.splits = c.splits;
    a.scale = 1.0f;
    a.partial = static_cast<float*>(partial.p);
    a.out = static_cast<__nv_bfloat16*>(out.p);
    k::decode_attention(a, nullptr);
    check(cudaDeviceSynchronize(), "decode_attention");
    std::vector<__nv_bfloat16> got(std::size_t(b) * width);
    out.copy_to_host(got.data(), got.size() * 2);
    // The q8_1 form of the same rows.
    a.out   = nullptr;
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
            const int kv = h / group;
            const float* qh = q.data() + (std::size_t(r) * c.query_heads + h) * d;
            std::vector<double> score;
            double m = -1e300;
            for (int j = lo; j <= p; ++j) {
                const int slot = c.ring != 0 ? j % c.ring : j;
                const __half* kr = kc.data() + seqs[r] * seq_stride + kv * head_stride + std::int64_t(slot) * d;
                double s = 0;
                for (int i = 0; i < d; ++i) { s += double(qh[i]) * __half2float(kr[i]); }
                score.push_back(s);
                m = std::max(m, s);
            }
            double l = 0;
            for (auto& s : score) { s = std::exp(s - m); l += s; }
            for (int i = 0; i < d; ++i) {
                double o = 0;
                for (int j = lo; j <= p; ++j) {
                    const int slot = c.ring != 0 ? j % c.ring : j;
                    o += score[j - lo] *
                         __half2float(vc[seqs[r] * seq_stride + kv * head_stride + std::int64_t(slot) * d + i]);
                }
                o /= l;
                const std::size_t at = std::size_t(r) * width + std::size_t(h) * d + i;
                const double e = double(__bfloat162float(got[at])) - o;
                num += e * e;
                den += o * o;
                const float dq = __low2float(got_ds[at / 32]);
                const double eq = double(got_q[at]) * dq - o;
                qnum += eq * eq;
            }
        }
    }
    const double rms = std::sqrt(num / den), qrms = std::sqrt(qnum / den);
    const std::string name = "decode_attention D" + std::to_string(d) + " ring " + std::to_string(c.ring) +
                             " splits " + std::to_string(c.splits) + " pos " + std::to_string(max_pos) +
                             (c.derived ? " derived keys" : "") +
                             (c.group > 1 ? " rows of one sequence " + std::to_string(c.group) : "");
    const bool ok = rms < 6e-3 && qrms < 1.5e-2;
    std::cout << (ok ? "OK   " : "FAIL ") << name << " rms=" << rms << " q8 rms=" << qrms << '\n';
    failures += ok ? 0 : 1;
}

void route(int tokens, std::mt19937& rng) {
    constexpr int kHidden = 2816, kExperts = 128, kTopK = 8;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(std::size_t(tokens) * kHidden), router(std::size_t(kExperts) * kHidden),
        scale(kExperts);
    for (auto& v : x) { v = normal(rng) * 0.02f; }
    for (auto& v : router) { v = normal(rng); }
    std::uniform_real_distribution<float> unit(0.5f, 1.5f);
    for (auto& v : scale) { v = unit(rng); }
    const int pairs = tokens * kTopK;
    DeviceBuffer dx(x.size() * 4), dr(router.size() * 4), dsc(scale.size() * 4),
        logits(std::size_t(tokens) * kExperts * 4), ids(pairs * 4), weights(pairs * 4),
        routing((kExperts + 1 + pairs + kExperts + 1) * 4);
    dx.copy_from_host(x.data(), x.size() * 4);
    dr.copy_from_host(router.data(), router.size() * 4);
    dsc.copy_from_host(scale.data(), scale.size() * 4);
    auto* base = static_cast<std::int32_t*>(routing.p);
    k::route_select(static_cast<const float*>(dx.p), kHidden, tokens, static_cast<const float*>(dr.p),
                    static_cast<const float*>(dsc.p), kExperts, kTopK, static_cast<float*>(logits.p),
                    static_cast<std::int32_t*>(ids.p), static_cast<float*>(weights.p), base,
                    base + kExperts + 1, base + kExperts + 1 + pairs,
                    base + kExperts + 1 + pairs + kExperts, nullptr);
    check(cudaDeviceSynchronize(), "route_select");
    std::vector<std::int32_t> got_ids(pairs), arrays(kExperts + 1 + pairs + kExperts + 1);
    std::vector<float> got_w(pairs);
    ids.copy_to_host(got_ids.data(), pairs * 4);
    weights.copy_to_host(got_w.data(), pairs * 4);
    routing.copy_to_host(arrays.data(), arrays.size() * 4);
    const std::int32_t* bounds = arrays.data();
    const std::int32_t* sorted = bounds + kExperts + 1;
    const std::int32_t* active = sorted + pairs;
    const std::int32_t count   = active[kExperts];
    bool ok = true;
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::pair<double, int>> l(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            double s = 0;
            for (int i = 0; i < kHidden; ++i) {
                s += double(x[std::size_t(t) * kHidden + i]) * router[std::size_t(e) * kHidden + i];
            }
            l[e] = {s, e};
        }
        std::sort(l.begin(), l.end(), [](auto a, auto b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        double sum = 0;
        for (int j = 0; j < kTopK; ++j) { sum += std::exp(l[j].first - l[0].first); }
        // Near ties can order differently in FP32; compare as sets with their weights.
        std::map<int, double> want;
        for (int j = 0; j < kTopK; ++j) {
            want[l[j].second] = std::exp(l[j].first - l[0].first) / sum * scale[l[j].second];
        }
        for (int j = 0; j < kTopK; ++j) {
            const int e = got_ids[t * kTopK + j];
            if (!want.contains(e) || std::abs(want[e] - got_w[t * kTopK + j]) > 1e-4) { ok = false; }
        }
    }
    // The grouping: every pair once, under its own expert, active experts ascending with pairs.
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
    ok = ok && listed == count && int(seen.size()) == pairs && bounds[0] == 0 && bounds[kExperts] == pairs;
    std::cout << (ok ? "OK   " : "FAIL ") << "route_select T=" << tokens << '\n';
    failures += ok ? 0 : 1;
}

} // namespace

// qkv_post against an FP64 oracle: per-head RMSNorm of Q and K, Q rotated at its position by the
// given frequencies and multiplied by q_scale (the square of YaRN's attention factor), K = V
// blocks storing the RMS-normalized K rows as the values of their positions (a null k_cache).
// With `stored`, token t's value goes to slot stored[t] (a branch pass's rows) while it rotates
// at positions[t].
void qkv(int head_dim, float q_scale, const std::vector<int>& positions, std::mt19937& rng,
         const std::vector<int>& stored = {}) {
    const std::vector<int>& at_slot = stored.empty() ? positions : stored;
    const int d = head_dim, t = int(positions.size()), hq = 4, hkv = 1;
    const int rows  = (hq + hkv) * d;
    const int slots = *std::max_element(at_slot.begin(), at_slot.end()) + 1;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> qkv(std::size_t(t) * rows);
    std::vector<float> qkv_f(qkv.size());
    for (std::size_t i = 0; i < qkv.size(); ++i) {
        qkv[i]   = __float2bfloat16(normal(rng));
        qkv_f[i] = __bfloat162float(qkv[i]);
    }
    std::vector<float> q_norm(d), k_norm(d), inv_freq(d / 2, 0.0f);
    for (auto& w : q_norm) { w = 1.0f + 0.25f * normal(rng); }
    for (auto& w : k_norm) { w = 1.0f + 0.25f * normal(rng); }
    const ninfer::ops::RopeYarnFrequencies yarn =
        ninfer::ops::rope_yarn_frequencies(d, std::min(64, d / 2), 1e6f, {2.0f, 262144});
    std::copy_n(yarn.inverse_frequency, yarn.pairs, inv_freq.begin());
    std::vector<std::int32_t> pos(positions.begin(), positions.end());
    const std::vector<std::int32_t> slot_of(at_slot.begin(), at_slot.end());
    DeviceBuffer d_qkv(qkv.size() * 2), d_qn(d * 4), d_kn(d * 4), d_f(inv_freq.size() * 4),
        d_pos(t * 4), d_slots(t * 4), d_q(std::size_t(t) * hq * d * 4),
        d_v(std::size_t(slots) * d * 2);
    d_qkv.copy_from_host(qkv.data(), qkv.size() * 2);
    d_qn.copy_from_host(q_norm.data(), d * 4);
    d_kn.copy_from_host(k_norm.data(), d * 4);
    d_f.copy_from_host(inv_freq.data(), inv_freq.size() * 4);
    d_pos.copy_from_host(pos.data(), t * 4);
    d_slots.copy_from_host(slot_of.data(), t * 4);
    k::QkvArgs a;
    a.qkv         = static_cast<const __nv_bfloat16*>(d_qkv.p);
    a.tokens      = t;
    a.query_heads = hq;
    a.kv_heads    = hkv;
    a.head_dim    = d;
    a.has_v       = false;
    a.q_norm      = static_cast<const float*>(d_qn.p);
    a.k_norm      = static_cast<const float*>(d_kn.p);
    a.eps         = 1e-6f;
    a.positions   = static_cast<const std::int32_t*>(d_pos.p);
    a.inv_freq    = static_cast<const float*>(d_f.p);
    a.q_scale     = q_scale;
    a.q_out       = static_cast<float*>(d_q.p);
    a.k_cache     = nullptr;
    a.v_cache     = static_cast<__half*>(d_v.p);
    a.seq_stride  = std::int64_t(slots) * d;
    a.head_stride = std::int64_t(slots) * d;
    if (!stored.empty()) { a.slots = static_cast<const std::int32_t*>(d_slots.p); }
    k::qkv_post(a, nullptr);
    check(cudaDeviceSynchronize(), "qkv_post");
    std::vector<float> q(std::size_t(t) * hq * d);
    std::vector<__half> v(std::size_t(slots) * d);
    check(cudaMemcpy(q.data(), d_q.p, q.size() * 4, cudaMemcpyDeviceToHost), "q copy");
    check(cudaMemcpy(v.data(), d_v.p, v.size() * 2, cudaMemcpyDeviceToHost), "v copy");
    const std::string label = "qkv_post D" + std::to_string(d) + " scale " +
                              std::to_string(q_scale) + (stored.empty() ? "" : " branch rows");
    double worst = 0.0;
    for (int token = 0; token < t; ++token) {
        for (int h = 0; h < hq; ++h) {
            const float* x = qkv_f.data() + std::size_t(token) * rows + h * d;
            double sum     = 0.0;
            for (int i = 0; i < d; ++i) { sum += double(x[i]) * x[i]; }
            const double inv = 1.0 / std::sqrt(sum / d + 1e-6);
            for (int i = 0; i < d / 2; ++i) {
                const double n0 = x[i] * inv * q_norm[i], n1 = x[i + d / 2] * inv * q_norm[i + d / 2];
                double c = 1.0, s = 0.0;
                if (inv_freq[i] != 0.0f) {
                    const float angle = float(positions[token]) * inv_freq[i];
                    c                 = std::cos(angle);
                    s                 = std::sin(angle);
                }
                const double e0 = (n0 * c - n1 * s) * q_scale, e1 = (n1 * c + n0 * s) * q_scale;
                const float* got = q.data() + (std::size_t(token) * hq + h) * d;
                worst = std::max({worst, std::fabs(got[i] - e0) / (std::fabs(e0) + 1e-2),
                                  std::fabs(got[i + d / 2] - e1) / (std::fabs(e1) + 1e-2)});
            }
        }
        // The stored value of the token's position: its K row (has_v false) RMS-normalized without
        // a weight, in F16; the key is derived from it where it is read.
        const float* kr = qkv_f.data() + std::size_t(token) * rows + hq * d;
        double ksum     = 0.0;
        for (int i = 0; i < d; ++i) { ksum += double(kr[i]) * kr[i]; }
        const double kinv = 1.0 / std::sqrt(ksum / d + 1e-6);
        for (int i = 0; i < d; ++i) {
            const float kept = __half2float(v[std::size_t(at_slot[token]) * d + i]);
            const double e   = kr[i] * kinv;
            if (std::fabs(kept - e) > 1e-3 * (std::fabs(e) + 1e-3)) {
                expect(false, label + ": value row of slot " + std::to_string(at_slot[token]) +
                                  " at " + std::to_string(i) + ": " + std::to_string(kept) +
                                  " for " + std::to_string(e));
                break;
            }
        }
    }
    expect(worst <= 2e-5, label + ": rotated queries off by " + std::to_string(worst));
}

// The rows of three branches (5, 8 and 7 tokens) one after another past a state of `first`
// positions: row t of the branch starting at row b rotates as position first + t - b.
const std::vector<int> kBranchLengths{5, 8, 7};

std::vector<std::int32_t> branch_starts() {
    std::vector<std::int32_t> starts;
    int row = 0;
    for (const int length : kBranchLengths) {
        for (int j = 0; j < length; ++j) { starts.push_back(row); }
        row += length;
    }
    return starts;
}

// Keys derived from stored values: slots from `turned` on are a branch pass's rows and turn at
// their branch positions, the slots before at their own.
void branch_keys(std::mt19937& rng) {
    constexpr int d = 512, heads = 2, turned = 30;
    const std::vector<std::int32_t> starts = branch_starts();
    const int count                        = turned + int(starts.size());
    std::vector<std::int32_t> turns;
    for (int t = 0; t < int(starts.size()); ++t) { turns.push_back(turned + t - starts[t]); }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__half> v(std::size_t(heads) * count * d);
    for (auto& x : v) { x = __float2half(normal(rng)); }
    std::vector<float> k_norm(d), inv_freq(64);
    for (auto& w : k_norm) { w = 1.0f + 0.25f * normal(rng); }
    for (int i = 0; i < 64; ++i) {
        inv_freq[i] = float(1.0 / std::pow(1e6, double(2 * i) / double(d)));
    }
    DeviceBuffer dv(v.size() * 2), dk(v.size() * 2), dw(d * 4), df(64 * 4),
        dr(std::size_t(count) * 64 * 8), dt(turns.size() * 4);
    dv.copy_from_host(v.data(), v.size() * 2);
    dw.copy_from_host(k_norm.data(), d * 4);
    df.copy_from_host(inv_freq.data(), 64 * 4);
    dt.copy_from_host(turns.data(), turns.size() * 4);
    k::rope_table(static_cast<const float*>(df.p), 64, count, static_cast<float2*>(dr.p), nullptr);
    k::derive_keys(static_cast<const __half*>(dv.p), std::int64_t(count) * d, heads, d, count,
                   static_cast<const float*>(dw.p), static_cast<const float2*>(dr.p), 64,
                   static_cast<__half*>(dk.p), nullptr, static_cast<const std::int32_t*>(dt.p),
                   turned);
    check(cudaDeviceSynchronize(), "derive_keys");
    std::vector<__half> got(v.size());
    dk.copy_to_host(got.data(), got.size() * 2);
    double worst = 0.0;
    for (int h = 0; h < heads; ++h) {
        for (int p = 0; p < count; ++p) {
            const int position   = p < turned ? p : turned + (p - turned) - starts[p - turned];
            const std::size_t at = (std::size_t(h) * count + p) * d;
            for (int i = 0; i < d / 2; ++i) {
                double c = 1.0, s = 0.0;
                if (i < 64) {
                    const double angle = double(position) * inv_freq[i];
                    c                  = std::cos(angle);
                    s                  = std::sin(angle);
                }
                const double n0 = double(__half2float(v[at + i])) * k_norm[i];
                const double n1 = double(__half2float(v[at + i + d / 2])) * k_norm[i + d / 2];
                const double e0 = n0 * c - n1 * s, e1 = n1 * c + n0 * s;
                worst = std::max(
                    {worst, std::fabs(__half2float(got[at + i]) - e0) / (std::fabs(e0) + 1e-1),
                     std::fabs(__half2float(got[at + i + d / 2]) - e1) / (std::fabs(e1) + 1e-1)});
            }
        }
    }
    expect(worst <= 2e-3, "derive_keys at branch positions off by " + std::to_string(worst));
}

// A branch pass's prompt mask over a sliding ring (`ring`, `window`) or a linear cache (ring 0):
// row t, of the branch starting at row b, is at position first + t - b. It sees the state's
// positions up to its own, within the window on a ring, and its branch's rows up to itself, within
// the window; no other branch's rows. A ring slot holds the latest position written to it.
void branch_mask(int ring, int window) {
    constexpr int first                    = 300;
    const std::vector<std::int32_t> starts = branch_starts();
    const int tokens                       = int(starts.size());
    const int last                         = first + tokens - 1;
    const int slots                        = ring != 0 ? ring : 512;
    std::vector<int> held(slots, -1);
    for (int p = 0; p <= last; ++p) { held[ring != 0 ? p % ring : p] = p; }
    DeviceBuffer dstarts(starts.size() * 4), dmask(std::size_t(tokens) * slots * 2);
    dstarts.copy_from_host(starts.data(), starts.size() * 4);
    k::build_mask(static_cast<__half*>(dmask.p), tokens, slots, first, ring, window, nullptr,
                  nullptr, static_cast<const std::int32_t*>(dstarts.p));
    check(cudaDeviceSynchronize(), "build_mask");
    std::vector<__half> mask(std::size_t(tokens) * slots);
    dmask.copy_to_host(mask.data(), mask.size() * 2);
    int wrong = 0;
    for (int t = 0; t < tokens; ++t) {
        const int position = first + t - starts[t];
        for (int s = 0; s < slots; ++s) {
            const int p = held[s];
            bool want   = false;
            if (p >= 0 && p < first) {
                want = p <= position && (ring == 0 || p > position - window);
            } else if (p >= first) {
                const int row = p - first;
                want = starts[row] == starts[t] && row <= t && (ring == 0 || t - row < window);
            }
            const bool visible = __half2float(mask[std::size_t(t) * slots + s]) == 0.0f;
            wrong += visible != want ? 1 : 0;
        }
    }
    expect(wrong == 0, "build_mask of branch rows, ring " + std::to_string(ring) + ": " +
                           std::to_string(wrong) + " wrong entries");
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(77);
        // Sliding layers: D 256, 16 query heads over 8, a 1024-token window in a ring of exactly
        // the window (as a sequence keeps it) and in a wider one.
        attention({256, 16, 8, 1024, 1024, 16, {5}}, rng);
        attention({256, 16, 8, 1024, 1024, 16, {700, 1900}}, rng);
        attention({256, 16, 8, 1280, 1024, 1, {3000}}, rng);
        // A verification's rows: groups of one sequence's consecutive positions sharing a block.
        attention({256, 16, 8, 1280, 1024, 16, {3000, 3001, 3002, 3003}, false, 4}, rng);
        attention({256, 16, 8, 1280, 1024, 1, {500, 501, 2000, 2001}, false, 2}, rng);
        attention({256, 16, 8, 1280, 1024, 16, {0, 1, 2, 3}, false, 4}, rng);
        attention({256, 16, 8, 1280, 1024, 16, {1022, 1023, 1024}, false, 3}, rng);
        // Full layers: D 512, 16 query heads over 2, a linear cache.
        attention({512, 16, 2, 0, 0, 123, {0}}, rng);
        attention({512, 16, 2, 0, 0, 123, {40, 4999}}, rng);
        attention({512, 16, 2, 0, 0, 1, {777}}, rng);
        attention({512, 16, 2, 0, 0, 123, {0}, true}, rng);
        attention({512, 16, 2, 0, 0, 123, {40, 4999}, true}, rng);
        attention({512, 16, 2, 0, 0, 1, {777}, true}, rng);
        // Under YaRN at factor two the derived keys turn at its frequencies.
        attention({512, 16, 2, 0, 0, 123, {40, 4999}, true, 1, 2.0f}, rng);
        attention({512, 16, 2, 0, 0, 1, {4999, 5000}, true, 2, 2.0f}, rng);
        // The queries' rotation and scale, Gemma 4's head widths: scale one, and the square of
        // YaRN's attention factor at factor two, (0.1 ln 2 + 1)^2.
        qkv(512, 1.0f, {5, 1000, 7}, rng);
        qkv(512, 1.1434f, {5, 1000, 7}, rng);
        qkv(256, 1.21f, {0, 1023, 300}, rng);
        for (const int tokens : {1, 3, 8}) { route(tokens, rng); }
        // A branch pass: rows 0..4, 5..12 and 13..19 continue a 300-position state.
        qkv(512, 1.0f, {300, 301, 302, 303, 304, 300, 301, 302}, rng,
            {300, 301, 302, 303, 304, 305, 306, 307});
        branch_keys(rng);
        branch_mask(256, 128);
        branch_mask(256, 4);
        branch_mask(0, 0);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma 4 kernels: " << error.what() << '\n';
        return 1;
    }
}
