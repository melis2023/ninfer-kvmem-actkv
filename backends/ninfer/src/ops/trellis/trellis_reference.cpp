#include "ops/trellis/trellis_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ninfer::ops::detail::trellis {
namespace {

// Paley matrices H_m (m = 12, 20, 68): H = I + S with S[0][j] = 1, S[j][0] = -1 (j > 0) and
// S[i][j] = chi(j - i), chi the quadratic character mod m - 1.
struct Paley {
    signed char h[68 * 68];
};

constexpr int chi(int d, int q) {
    d = ((d % q) + q) % q;
    if (d == 0) {
        return 0;
    }
    for (int i = 1; i < q; ++i) {
        if ((i * i) % q == d) {
            return 1;
        }
    }
    return -1;
}

constexpr Paley make_paley(int m) {
    Paley p{};
    const int q = m - 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            const int s = i == 0 ? (j > 0 ? 1 : 0) : (j == 0 ? -1 : chi(j - i, q));
            p.h[i * m + j] = static_cast<signed char>(s + (i == j ? 1 : 0));
        }
    }
    return p;
}

constexpr Paley kH12 = make_paley(12);
constexpr Paley kH20 = make_paley(20);
constexpr Paley kH68 = make_paley(68);

const signed char* hm(int m) {
    return m == 12 ? kH12.h : m == 20 ? kH20.h : kH68.h;
}

// 1/sqrt(n) in double, rounded once to fp32.
float inv_sqrt(std::int64_t n) {
    return static_cast<float>(std::pow(static_cast<double>(n), -0.5));
}

bool is_pow2(std::int64_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint16_t magnitude = static_cast<std::uint16_t>(bits & 0x7FFFu);
    if (magnitude == 0) {
        std::uint32_t zero = sign;
        float f;
        std::memcpy(&f, &zero, 4);
        return f;
    }
    if (magnitude >= 0x7C00u) {
        std::uint32_t inf = sign | 0x7F800000u | (static_cast<std::uint32_t>(magnitude & 0x3FFu) << 13);
        float f;
        std::memcpy(&f, &inf, 4);
        return f;
    }
    int exponent = (magnitude >> 10) & 0x1F;
    std::uint32_t mantissa = magnitude & 0x3FFu;
    if (exponent == 0) {  // subnormal
        int shift = 0;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 0x3FFu;
        exponent = 1 - shift;
    }
    std::uint32_t word = sign | (static_cast<std::uint32_t>(exponent + 112) << 23) | (mantissa << 13);
    float f;
    std::memcpy(&f, &word, 4);
    return f;
}

// the fused parts of a whole-slice side: slice r is shard r of every part, side by side
struct Parts {
    int n = 0;
    std::int64_t start[kMaxParts] = {};
    std::int64_t shard[kMaxParts] = {};
    std::int64_t cum[kMaxParts + 1] = {};
};

bool make_parts(Parts& p, std::int64_t n, const std::int32_t* parts, int nparts, int shards) {
    if (shards < 1 || (parts != nullptr && (nparts < 1 || nparts > kMaxParts))) {
        return false;
    }
    p.n = parts != nullptr ? nparts : 1;
    std::int64_t start = 0, cum = 0;
    for (int i = 0; i < p.n; ++i) {
        const std::int64_t size = parts != nullptr ? parts[i] : n;
        if (size <= 0 || size % shards) {
            return false;
        }
        p.start[i] = start;
        p.shard[i] = size / shards;
        p.cum[i] = cum;
        start += size;
        cum += size / shards;
    }
    p.cum[p.n] = cum;
    return start == n;
}

// the stored index of slice r's local entry l
std::int64_t stored_index(const Parts& p, int r, std::int64_t l) {
    int i = 0;
    while (i + 1 < p.n && l >= p.cum[i + 1]) {
        ++i;
    }
    return p.start[i] + static_cast<std::int64_t>(r) * p.shard[i] + (l - p.cum[i]);
}

// out[a*run + j] = c sum_b H[a][b] in[b*run + j] (transpose: H[b][a])
void cross(const float* in, float* out, int m, std::int64_t run, bool transpose, float c) {
    const signed char* h = hm(m);
    for (int a = 0; a < m; ++a) {
        float* o = out + static_cast<std::int64_t>(a) * run;
        std::fill(o, o + run, 0.0f);
        for (int b = 0; b < m; ++b) {
            const int s = transpose ? h[b * m + a] : h[a * m + b];
            const float* x = in + static_cast<std::int64_t>(b) * run;
            if (s > 0) {
                for (std::int64_t j = 0; j < run; ++j) {
                    o[j] += x[j];
                }
            } else {
                for (std::int64_t j = 0; j < run; ++j) {
                    o[j] -= x[j];
                }
            }
        }
        for (std::int64_t j = 0; j < run; ++j) {
            o[j] *= c;
        }
    }
}

bool side_ok(std::int64_t n, int kind, int block, int shards, const std::int32_t* parts, int nparts) {
    switch (kind) {
    case 0:
        return true;
    case 1:
        return is_pow2(block) && block >= 2 && n % block == 0;
    case 2: {
        Parts p;
        return make_parts(p, n, nparts > 0 ? parts : nullptr, nparts, shards) &&
               hadfull_m(n / shards) != 0;
    }
    default:
        return false;
    }
}

} // namespace

bool is_trellis(QType qtype) noexcept {
    return qtype >= QType::TRELLIS_1BIT && qtype <= QType::TRELLIS_6BIT;
}

int bits_of(QType qtype) noexcept {
    return is_trellis(qtype) ? static_cast<int>(qtype) - static_cast<int>(QType::TRELLIS_1BIT) + 1 : 0;
}

void paley_table(int m, signed char* out) {
    if (m == 1) {
        out[0] = 1;
        return;
    }
    if (m != 12 && m != 20 && m != 68) {
        return;
    }
    std::memcpy(out, hm(m), static_cast<std::size_t>(m) * m);
}

int hadfull_m(std::int64_t len) noexcept {
    for (const int b : {1, 12, 20, 68}) {
        if (len >= b && len % b == 0 && is_pow2(len / b)) {
            return b;
        }
    }
    return 0;
}

void fwht(float* v, std::int64_t n, std::int64_t block) {
    const float c = inv_sqrt(block);
    for (std::int64_t b0 = 0; b0 < n; b0 += block) {
        float* u = v + b0;
        for (std::int64_t h = 1; h < block; h *= 2) {
            for (std::int64_t i = 0; i < block; i += 2 * h) {
                for (std::int64_t j = i; j < i + h; ++j) {
                    const float a = u[j], b = u[j + h];
                    u[j] = a + b;
                    u[j + h] = a - b;
                }
            }
        }
        for (std::int64_t i = 0; i < block; ++i) {
            u[i] *= c;
        }
    }
}

bool hadfull_slice(float* v, std::int64_t n, const std::int32_t* parts, int nparts, int shards, int r,
                   bool inverse, float* tmp) {
    Parts p;
    if (!make_parts(p, n, parts, nparts, shards) || r < 0 || r >= shards) {
        return false;
    }
    const std::int64_t len = n / shards;
    const int m = hadfull_m(len);
    if (m == 0) {
        return false;
    }
    const std::int64_t run = len / m;
    const float c_run = inv_sqrt(run), c_m = inv_sqrt(m);
    float* a = tmp;
    float* b = tmp + len;
    for (std::int64_t l = 0; l < len; ++l) {
        a[l] = v[stored_index(p, r, l)];
    }
    float* res = a;
    if (m == 1) {
        fwht(a, len, len);
    } else if (!inverse) {
        for (int i = 0; i < m; ++i) {
            fwht(a + static_cast<std::int64_t>(i) * run, run, run);
        }
        cross(a, b, m, run, false, c_m);
        res = b;
    } else {
        cross(a, b, m, run, true, c_m);
        for (int i = 0; i < m; ++i) {
            fwht(b + static_cast<std::int64_t>(i) * run, run, run);
        }
        res = b;
    }
    for (std::int64_t l = 0; l < len; ++l) {
        v[stored_index(p, r, l)] = res[l];
    }
    return true;
}

bool side_apply(float* v, std::int64_t n, int kind, int block, int shards, const std::int32_t* parts,
                int nparts, bool inverse, float* tmp) {
    if (kind == 0) {
        return true;
    }
    if (kind == 1) {
        fwht(v, n, block);
        return true;
    }
    for (int r = 0; r < shards; ++r) {
        if (!hadfull_slice(v, n, parts, nparts, shards, r, inverse, tmp)) {
            return false;
        }
    }
    return true;
}

bool parse_desc(const Weight& w, std::int64_t K, std::int64_t N, Desc& out) {
    if (w.trellis_desc == nullptr || w.trellis_desc_len < 7 || K % 16 || N % 16) {
        return false;
    }
    const std::int32_t* d = w.trellis_desc;
    const std::int32_t nparts = d[6];
    if (nparts < 0 || nparts > kMaxParts || w.trellis_desc_len != 7 + nparts) {
        return false;
    }
    out.rows_kind = d[0];
    out.rows_block = d[1];
    out.rows_shards = d[2];
    out.cols_kind = d[3];
    out.cols_block = d[4];
    out.cols_shards = d[5];
    out.nparts = nparts;
    for (int i = 0; i < nparts; ++i) {
        out.parts[i] = d[7 + i];
    }
    // The output side carries the fused parts; an input-side whole-slice Hadamard takes one part.
    if (!side_ok(N, out.rows_kind, out.rows_block, out.rows_shards, nparts > 0 ? out.parts : nullptr, nparts)) {
        return false;
    }
    if (!side_ok(K, out.cols_kind, out.cols_block, out.cols_shards, nullptr, 0)) {
        return false;
    }
    return true;
}

void decode_rows(QType qtype, const void* data, std::int64_t K, std::int64_t n0, std::int64_t n_rows,
                 const void* s16_f16, float* out) {
    const int k = bits_of(qtype);
    const int nw = 8 * k, total = 256 * k;
    const std::int64_t kt = K / 16;
    const std::uint32_t* words = static_cast<const std::uint32_t*>(data);
    const std::uint16_t* scales = static_cast<const std::uint16_t*>(s16_f16);
    for (std::int64_t J = n0 / 16; J < (n0 + n_rows) / 16; ++J) {
        for (std::int64_t i = 0; i < kt; ++i) {
            const std::uint32_t* t = words + (J * kt + i) * nw;  // tile (i,J), output-tile-major
            const float s = scales != nullptr ? half_to_float(scales[i]) : 1.0f;
            for (int p = 0; p < 256; ++p) {
                const int start = ((p + 1) * k - 16 + total) % total;
                const int wi = start >> 5, o = start & 31;
                const std::uint64_t pair =
                    (static_cast<std::uint64_t>(t[wi]) << 32) | t[(wi + 1) % nw];
                const std::uint32_t state = static_cast<std::uint32_t>(pair >> (48 - o)) & 0xFFFFu;
                const std::uint32_t x = state * 0x83DCD12Du;  // the INT8 grid
                const int v = static_cast<int>(((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) +
                                              ((x >> 26) & 63)) - 126;
                const int a = p >> 3, jj = p & 7;
                const int row = (a >> 2) + 8 * ((jj >> 2) & 1);
                const int col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
                out[(16 * J + row - n0) * K + 16 * i + col] = static_cast<float>(v) * s;
            }
        }
    }
}

bool dequantize(const Weight& w, float* out) {
    const std::int64_t K = w.k, N = w.n;
    Desc desc;
    if (!is_trellis(w.qtype) || !parse_desc(w, K, N, desc) || w.trellis_pre == nullptr ||
        w.trellis_post == nullptr) {
        return false;
    }
    const std::int32_t* pin = w.trellis_pin;
    const std::int32_t* pout = w.trellis_pout;
    if (pin != nullptr &&
        !std::all_of(pin, pin + K, [K](std::int32_t j) { return j >= 0 && j < K; })) {
        return false;
    }
    if (pout != nullptr &&
        !std::all_of(pout, pout + N, [N](std::int32_t i) { return i >= 0 && i < N; })) {
        return false;
    }
    decode_rows(w.qtype, w.qdata, K, 0, N, w.trellis_s16, out);
    const float* pre = static_cast<const float*>(w.trellis_pre);
    const float* post = static_cast<const float*>(w.trellis_post);
    const std::int64_t nmax = K > N ? K : N;
    std::vector<float> tmp(2 * nmax), row(K);
    for (std::int64_t n = 0; n < N; ++n) {  // w -> (w T_cols) diag(pre) P_in
        float* wt = out + n * K;
        if (!side_apply(wt, K, desc.cols_kind, desc.cols_block, desc.cols_shards, nullptr, 0, true,
                        tmp.data())) {
            return false;
        }
        for (std::int64_t j = 0; j < K; ++j) {
            wt[j] *= pre[j];
        }
        if (pin != nullptr) {  // column pin[j] of W collects column j
            std::fill(row.begin(), row.end(), 0.0f);
            for (std::int64_t j = 0; j < K; ++j) {
                row[pin[j]] += wt[j];
            }
            std::memcpy(wt, row.data(), K * sizeof(float));
        }
    }
    std::vector<float> cols(16 * N);
    for (std::int64_t c0 = 0; c0 < K; c0 += 16) {
        for (std::int64_t n = 0; n < N; ++n) {
            for (int c = 0; c < 16; ++c) {
                cols[c * N + n] = out[n * K + c0 + c];
            }
        }
        for (int c = 0; c < 16; ++c) {
            float* col = cols.data() + c * N;
            if (!side_apply(col, N, desc.rows_kind, desc.rows_block, desc.rows_shards,
                            desc.nparts > 0 ? desc.parts : nullptr, desc.nparts, true, tmp.data())) {
                return false;
            }
        }
        for (std::int64_t n = 0; n < N; ++n) {
            for (int c = 0; c < 16; ++c) {
                out[n * K + c0 + c] = cols[c * N + n] * post[n];
            }
        }
    }
    if (pout != nullptr) {  // row i of W is row pout[i]
        std::vector<float> rows(static_cast<std::size_t>(N) * K);
        for (std::int64_t i = 0; i < N; ++i) {
            std::memcpy(rows.data() + i * K, out + static_cast<std::int64_t>(pout[i]) * K,
                        K * sizeof(float));
        }
        std::memcpy(out, rows.data(), rows.size() * sizeof(float));
    }
    return true;
}

bool project_host(const float* x, int tokens, const Weight& w, float* y) {
    std::vector<float> dense(static_cast<std::size_t>(w.n) * w.k);
    if (!dequantize(w, dense.data())) {
        return false;
    }
    for (int t = 0; t < tokens; ++t) {
        for (std::int64_t n = 0; n < w.n; ++n) {
            double acc = 0.0;
            for (std::int64_t k = 0; k < w.k; ++k) {
                acc += static_cast<double>(dense[n * w.k + k]) *
                       static_cast<double>(x[k + static_cast<std::int64_t>(w.k) * t]);
            }
            y[n + static_cast<std::int64_t>(w.n) * t] = static_cast<float>(acc);
        }
    }
    return true;
}

bool embedding_host(const std::int32_t* ids, int tokens, const Weight& w, float* rows) {
    const std::int64_t K = w.k, N = w.n;
    Desc desc;
    if (!is_trellis(w.qtype) || !parse_desc(w, K, N, desc) || w.trellis_pre == nullptr) {
        return false;
    }
    if (desc.rows_kind != 0) {  // a coded embedding has no row side
        return false;
    }
    const std::int32_t* pin = w.trellis_pin;
    const float* pre = static_cast<const float*>(w.trellis_pre);
    std::vector<float> tmp(2 * K), gathered(K), row(K);
    std::vector<float> decoded(static_cast<std::size_t>(N) * K);
    for (int t = 0; t < tokens; ++t) {
        const std::int32_t id = ids[t];
        if (id < 0 || id >= N) {
            return false;
        }
        // 16-row group holding the id; at most one group is decoded per lookup
        const std::int64_t base = (id / 16) * 16;
        decode_rows(w.qtype, w.qdata, K, base, 16, w.trellis_s16, decoded.data());
        float* e = rows + static_cast<std::int64_t>(K) * t;
        const float* src = decoded.data() + (id - base) * K;
        std::memcpy(gathered.data(), src, K * sizeof(float));
        if (!side_apply(gathered.data(), K, desc.cols_kind, desc.cols_block, desc.cols_shards,
                        nullptr, 0, true, tmp.data())) {
            return false;
        }
        for (std::int64_t j = 0; j < K; ++j) {
            gathered[j] *= pre[j];
        }
        if (pin != nullptr) {
            std::fill(row.begin(), row.end(), 0.0f);
            for (std::int64_t j = 0; j < K; ++j) {
                row[pin[j]] += gathered[j];
            }
            std::memcpy(e, row.data(), K * sizeof(float));
        } else {
            std::memcpy(e, gathered.data(), K * sizeof(float));
        }
    }
    return true;
}

bool build_side_plan(const Weight& w, std::int64_t n, bool rows_side, SidePlan& out) {
    Desc desc;
    if (!parse_desc(w, w.k, w.n, desc)) {
        return false;
    }
    out.kind = rows_side ? desc.rows_kind : desc.cols_kind;
    out.block = rows_side ? desc.rows_block : desc.cols_block;
    out.shards = rows_side ? desc.rows_shards : desc.cols_shards;
    out.nparts = rows_side ? desc.nparts : 0;
    for (int i = 0; i < out.nparts; ++i) {
        out.parts[i] = desc.parts[i];
    }
    out.n = n;
    if (out.kind == 0) {
        return true;
    }
    if (out.kind == 1) {
        return is_pow2(out.block) && out.block >= 2 && n % out.block == 0;
    }
    if (out.kind != 2) {
        return false;
    }
    out.shard_len = n / out.shards;
    out.m = hadfull_m(out.shard_len);
    if (out.m == 0) {
        return false;
    }
    out.run = out.shard_len / out.m;
    Parts p;
    if (!make_parts(p, n, out.nparts > 0 ? out.parts : nullptr, out.nparts, out.shards)) {
        return false;
    }
    out.perm.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < out.shards; ++r) {
        for (std::int64_t l = 0; l < out.shard_len; ++l) {
            out.perm[static_cast<std::size_t>(r) * out.shard_len + l] =
                static_cast<std::int32_t>(stored_index(p, r, l));
        }
    }
    return true;
}

} // namespace ninfer::ops::detail::trellis
