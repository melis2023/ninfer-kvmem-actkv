#include "models/gemma4/kernels.h"

#include "core/device.h"
#include "models/gemma4/kv_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::gemma4::kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float s = lane < int(blockDim.x >> 5) ? partial[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) { total = s; }
    }
    __syncthreads();
    return total;
}

// Two sums over the block at once (blockDim.x a multiple of 32, at most 1024).
__device__ float2 block_sum2(float a, float b) {
    __shared__ float2 partial2[32];
    __shared__ float2 total2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    a = warp_sum(a);
    b = warp_sum(b);
    __syncthreads();
    if (lane == 0) { partial2[warp] = make_float2(a, b); }
    __syncthreads();
    if (warp == 0) {
        float2 v = lane < int(blockDim.x >> 5) ? partial2[lane] : make_float2(0.0f, 0.0f);
        v.x      = warp_sum(v.x);
        v.y      = warp_sum(v.y);
        if (lane == 0) { total2 = v; }
    }
    __syncthreads();
    return total2;
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Row kernels hold one float4 of a [hidden] row per thread (hidden % 4 == 0, hidden / 4 <= 1024).
__device__ __forceinline__ float4 load_bf16x4(const __nv_bfloat16* p) {
    const uint2 w   = *reinterpret_cast<const uint2*>(p);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w.y));
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}
__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
    const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 w;
    w.x = *reinterpret_cast<const unsigned*>(&lo);
    w.y = *reinterpret_cast<const unsigned*>(&hi);
    *reinterpret_cast<uint2*>(p) = w;
}
__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__device__ __forceinline__ float4 scale4(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 ld4(const float* p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}
__device__ __forceinline__ float4 fixed4(const unsigned long long* p, int i) {
    const auto* v  = p + 4 * std::int64_t(i);
    const double s = 1.0 / 4294967296.0;
    return make_float4(float(double(static_cast<long long>(v[0])) * s),
                       float(double(static_cast<long long>(v[1])) * s),
                       float(double(static_cast<long long>(v[2])) * s),
                       float(double(static_cast<long long>(v[3])) * s));
}

// The q8_1 form of this thread's four values, elements 4i..4i+3 of row `row` of width k: eight
// consecutive threads hold one 32-value block (hidden % 32 == 0 keeps every group of eight either
// in or out of the row). Every lane of the warp calls it; `valid` lanes store.
__device__ __forceinline__ void store_q8x4(const Q8Rows& q, int k, std::int64_t row, int i, float4 v,
                                           bool valid) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float sum  = v.x + v.y + v.z + v.w;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
        sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    }
    if (!valid) { return; }
    const float d = amax / 127.0f;
    char4 c       = make_char4(0, 0, 0, 0);
    if (amax != 0.0f) {
        c = make_char4(static_cast<signed char>(roundf(v.x / d)), static_cast<signed char>(roundf(v.y / d)),
                       static_cast<signed char>(roundf(v.z / d)), static_cast<signed char>(roundf(v.w / d)));
    }
    *reinterpret_cast<char4*>(q.qs + row * k + 4 * i) = c;
    if ((i & 7) == 0) { q.ds[row * (k / 32) + i / 8] = make_half2(__float2half(d), __float2half(sum)); }
}

__global__ void scale_kernel(float* x, std::int64_t n, float s) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { x[i] *= s; }
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                int hidden, float eps, __nv_bfloat16* __restrict__ out, Q8Rows q8) {
    const int q      = hidden >> 2, i = threadIdx.x;
    const float* row = x + std::int64_t(blockIdx.x) * hidden;
    const float4 v   = i < q ? ld4(row, i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float inv  = rsqrtf(block_sum(dot4(v, v)) / float(hidden) + eps);
    float4 y         = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < q) {
        const float4 wv = w != nullptr ? ld4(w, i) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        y               = mul4(scale4(v, inv), wv);
        if (out != nullptr) { store_bf16x4(out + std::int64_t(blockIdx.x) * hidden + 4 * i, y); }
    }
    if (q8.qs != nullptr) { store_q8x4(q8, hidden, blockIdx.x, i, y, i < q); }
}

// Block (t, head): Hq query heads then Hkv key/value heads; D/2 threads, thread i the pair
// (i, i + D/2).
__global__ void qkv_post_kernel(QkvArgs a) {
    const int t      = blockIdx.x;
    const int head   = blockIdx.y;
    const int d      = a.head_dim;
    const int half   = d >> 1;
    const int i      = threadIdx.x;
    const int rows   = (a.query_heads + a.kv_heads * (a.has_v ? 2 : 1)) * d;
    const __nv_bfloat16* row = a.qkv + std::int64_t(t) * rows;
    const int position = a.positions[t];
    const float angle  = float(position) * a.inv_freq[i];
    float c = 1.0f, s = 0.0f;
    if (a.inv_freq[i] != 0.0f) { sincosf(angle, &s, &c); }
    if (head < a.query_heads) {
        const __nv_bfloat16* x = row + head * d;
        const float x0 = __bfloat162float(x[i]), x1 = __bfloat162float(x[i + half]);
        const float inv = rsqrtf(block_sum(x0 * x0 + x1 * x1) / float(d) + a.eps);
        const float n0 = x0 * inv * a.q_norm[i], n1 = x1 * inv * a.q_norm[i + half];
        float* q = a.q_out + (std::int64_t(t) * a.query_heads + head) * d;
        q[i]        = (n0 * c - n1 * s) * a.q_scale;
        q[i + half] = (n1 * c + n0 * s) * a.q_scale;
        return;
    }
    const int kh = head - a.query_heads;
    const __nv_bfloat16* k = row + (a.query_heads + kh) * d;
    const __nv_bfloat16* v = a.has_v ? row + (a.query_heads + a.kv_heads + kh) * d : k;
    const float k0 = __bfloat162float(k[i]), k1 = __bfloat162float(k[i + half]);
    const float v0 = __bfloat162float(v[i]), v1 = __bfloat162float(v[i + half]);
    const float kinv = rsqrtf(block_sum(k0 * k0 + k1 * k1) / float(d) + a.eps);
    const float vinv = rsqrtf(block_sum(v0 * v0 + v1 * v1) / float(d) + a.eps);
    const float n0 = k0 * kinv * a.k_norm[i], n1 = k1 * kinv * a.k_norm[i + half];
    const int sequence = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const int stored   = a.slots != nullptr ? a.slots[t] : position;
    const int slot     = a.row_slots ? t : a.ring != 0 ? stored % a.ring : stored;
    const std::int64_t at =
        std::int64_t(sequence) * a.seq_stride + std::int64_t(kh) * a.head_stride +
        std::int64_t(slot) * d;
    if (a.k_cache != nullptr) {
        a.k_cache[at + i]        = __float2half(n0 * c - n1 * s);
        a.k_cache[at + i + half] = __float2half(n1 * c + n0 * s);
    }
    a.v_cache[at + i]        = __float2half(v0 * vinv);
    a.v_cache[at + i + half] = __float2half(v1 * vinv);
}

__global__ void build_mask_kernel(__half* mask, int tokens, int slots, int first, int ring,
                                  int window, const std::int32_t* __restrict__ blocks,
                                  const std::int32_t* __restrict__ starts) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (s >= slots) { return; }
    const int query = first + t;
    bool visible    = false;
    if (starts != nullptr) {
        // A branch row: the positions before `first` as its own position sees them (the window
        // ends at the same distance), and its branch's rows, which lie at positions first +
        // starts[t] onwards, up to itself.
        const int own  = first + starts[t];
        const int last = first + tokens - 1;
        const int p    = ring == 0 ? s : s <= last ? s + ((last - s) / ring) * ring : -1;
        if (p >= 0 && p < first) {
            visible = ring == 0 || p > query - (own - first) - window;
        } else if (p >= own && p <= query) {
            visible = ring == 0 || p > query - window;
        }
    } else if (ring == 0) {
        visible = s <= query;
    } else {
        const int last = first + tokens - 1;
        if (s <= last) {
            const int p = s + ((last - s) / ring) * ring; // latest position in slot s
            visible     = p <= query && p > query - window;
            if (!visible && blocks != nullptr && p > query && p <= last && blocks[t] >= 0) {
                visible = blocks[p - first] == blocks[t];
            }
        }
    }
    mask[std::int64_t(t) * slots + s] = visible ? __float2half(0.0f) : __float2half(-INFINITY);
}

// The key pair (i, i + D/2) from its stored value pair and its rotation (cos, sin), as qkv_post
// rotates a key.
__device__ __forceinline__ void derived_key_pair(float v0, float v1, float w0, float w1, float2 cs,
                                                 __half& k0, __half& k1) {
    const float n0 = v0 * w0, n1 = v1 * w1;
    k0 = __float2half(n0 * cs.x - n1 * cs.y);
    k1 = __float2half(n1 * cs.x + n0 * cs.y);
}

__global__ void rope_table_kernel(const float* __restrict__ freq, int pairs, int positions,
                                  float2* __restrict__ rope) {
    const std::int64_t j = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= std::int64_t(positions) * pairs) { return; }
    const int p = int(j / pairs), i = int(j % pairs);
    const float angle = float(p) * freq[i];
    float c = 1.0f, s = 0.0f;
    if (freq[i] != 0.0f) { sincosf(angle, &s, &c); }
    rope[j] = make_float2(c, s);
}

// Block (position, head): D/2 threads, one pair each.
__global__ void derive_keys_kernel(const __half* __restrict__ v, std::int64_t head_stride, int d,
                                   const float* __restrict__ w, const float2* __restrict__ rope,
                                   int pairs, __half* __restrict__ k,
                                   const std::int32_t* __restrict__ turns, int turned_from) {
    const int p = blockIdx.x, h = blockIdx.y, i = threadIdx.x, half = d >> 1;
    const std::int64_t at = h * head_stride + std::int64_t(p) * d;
    const int turn        = turns != nullptr && p >= turned_from ? turns[p - turned_from] : p;
    const float2 cs = i < pairs ? rope[std::int64_t(turn) * pairs + i] : make_float2(1.0f, 0.0f);
    derived_key_pair(__half2float(v[at + i]), __half2float(v[at + i + half]), w[i], w[i + half],
                     cs, k[at + i], k[at + i + half]);
}

// Block (position, head): the position's K and V rows, 16-byte words.
__global__ void ring_copy_kernel(RingCopyArgs a) {
    const int p = a.begin + blockIdx.x, h = blockIdx.y;
    const std::int64_t src = h * a.src_head_stride + std::int64_t(p % a.src_ring) * a.head_dim;
    const std::int64_t dst = h * a.dst_head_stride + std::int64_t(p % a.dst_ring) * a.head_dim;
    for (int w = threadIdx.x; w < a.head_dim / 8; w += blockDim.x) {
        reinterpret_cast<uint4*>(a.dst_k + dst)[w] = reinterpret_cast<const uint4*>(a.src_k + src)[w];
        reinterpret_cast<uint4*>(a.dst_v + dst)[w] = reinterpret_cast<const uint4*>(a.src_v + src)[w];
    }
}

__global__ void f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2bfloat16(in[i]); }
}

// ---- Decode attention ---------------------------------------------------------------------------
// Block (row, kv head, split): the G query heads of the KV head over the split's keys, in tiles. Four
// warps; warp w scores heads [w * HPW, (w + 1) * HPW) over keys of its key group (G < 4: the tile's
// keys are split between 4 / G warps per head). Every thread then accumulates D / 128 dimensions of
// every head's output.
constexpr int kDecodeThreads = 128;

// The sums of 16 per-lane values v[k] over the warp, by halving exchanges: afterwards lanes 2j and
// 2j + 1 both hold the total of key (lane & 16 ? 8 : 0) + (lane & 8 ? 4 : 0) + (lane & 4 ? 2 : 0) +
// (lane & 2 ? 1 : 0).
__device__ __forceinline__ float key_reduce16(float (&v)[16], int lane) {
#pragma unroll
    for (int width = 8, offset = 16; width >= 1; width >>= 1, offset >>= 1) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            const float send = upper ? v[i] : v[i + width];
            const float keep = upper ? v[i + width] : v[i];
            v[i]             = keep + __shfl_xor_sync(0xFFFFFFFFu, send, offset);
        }
    }
    return v[0] + __shfl_xor_sync(0xFFFFFFFFu, v[0], 1);
}

// The splits a row's keys actually use: at least a tile of keys each, so a short context does not
// pay for empty blocks and their partial rows. The attention and its combine agree on it.
__device__ __forceinline__ int decode_active_splits(int keys, int splits, int tile) {
    return min(splits, max(1, (keys + tile - 1) / tile));
}

// A tile's K or V words in registers between their fetch and their store to shared memory: F16 16-byte
// words as they are, or a quantized plane's chunk bytes and scales, decoded at the store.
template <KvCoding C, int N>
struct TileRegs {
    typename kv::Chunk<C>::Raw raw[N];
    typename kv::Chunk<C>::Scale scale[N];
};
template <int N>
struct TileRegs<KvCoding::F16, N> {
    uint4 raw[N];
};

// Word u of the tile: chunk c of the row at `slot` (a quantized plane), or the F16 word at element
// `at` of `cache`.
template <KvCoding C, int N>
__device__ __forceinline__ void tile_fetch(TileRegs<C, N>& t, int u, bool valid, const __half* cache,
                                           std::int64_t at, const KvPlaneView& plane, int sequence,
                                           int head, int slot, int c) {
    if constexpr (C == KvCoding::F16) {
        t.raw[u] = valid ? *reinterpret_cast<const uint4*>(cache + at) : make_uint4(0, 0, 0, 0);
    } else {
        using K = kv::Chunk<C>;
        t.raw[u]   = typename K::Raw{};
        t.scale[u] = typename K::Scale{};
        if (valid) {
            const std::uint8_t* codes = plane.codes + sequence * plane.code_seq +
                                        head * plane.code_head + slot * plane.code_row;
            const std::uint8_t* scales = plane.scales + sequence * plane.scale_seq +
                                         head * plane.scale_head + slot * plane.scale_row;
            t.raw[u]   = K::load(codes, c);
            t.scale[u] = K::scale(scales, c);
        }
    }
}

template <KvCoding C, int N>
__device__ __forceinline__ uint4 tile_word(const TileRegs<C, N>& t, int u) {
    if constexpr (C == KvCoding::F16) {
        return t.raw[u];
    } else {
        float f[8];
        kv::Chunk<C>::decode(t.raw[u], t.scale[u], f);
        return kv::pack8(f);
    }
}

template <int D, int G>
struct DecodeShape {
    static constexpr int kTile      = D == 512 ? 16 : 32;
    static constexpr int kHpw       = G >= 4 ? G / 4 : 1;   // heads per warp
    static constexpr int kGroups    = G >= 4 ? 1 : 4 / G;   // key groups per head
    static constexpr int kDpl       = D / 32;               // dims per lane
    static constexpr int kDpt       = D / kDecodeThreads;   // dims per thread (PV)
};

template <int D, int G, KvCoding KC = KvCoding::F16, KvCoding VC = KvCoding::F16>
__launch_bounds__(kDecodeThreads) __global__ void decode_attention_kernel(DecodeAttentionArgs a) {
    constexpr int V      = G;
    using S              = DecodeShape<D, V>;
    constexpr int kTile  = S::kTile;
    // Rows vary fastest: a verification's rows of one sequence read the same keys at once.
    const int b          = blockIdx.x;
    const int kv_head    = blockIdx.y;
    const int split      = blockIdx.z;
    const int warp       = threadIdx.x >> 5;
    const int lane       = threadIdx.x & 31;
    const int position   = a.positions[b];
    const int sequence   = a.sequences[b];
    const int lo_all     = a.ring != 0 ? max(0, position - a.window + 1) : 0;
    const int n_all      = position + 1 - lo_all;
    const int active     = decode_active_splits(n_all, a.splits, kTile);
    if (split >= active) { return; }
    const int chunk      = ((n_all + active - 1) / active + kTile - 1) / kTile * kTile;
    const int lo         = lo_all + split * chunk;
    const int hi         = min(position + 1, lo + chunk);

    __shared__ __align__(16) __half k_s[kTile][D];
    __shared__ __align__(16) __half v_s[kTile][D];
    __shared__ float p_s[V][kTile];
    __shared__ float m_s[V], l_s[V], scale_s[V];

    if (threadIdx.x < V) {
        m_s[threadIdx.x] = -INFINITY;
        l_s[threadIdx.x] = 0.0f;
    }
    float acc[V][S::kDpt];
#pragma unroll
    for (int g = 0; g < V; ++g) {
#pragma unroll
        for (int j = 0; j < S::kDpt; ++j) { acc[g][j] = 0.0f; }
    }
    __syncthreads();

    // This lane's slice of each of its heads' queries.
    const int head_base = (warp / S::kGroups) * S::kHpw;
    const int key_group = warp % S::kGroups;
    float q_r[S::kHpw][S::kDpl];
#pragma unroll
    for (int h = 0; h < S::kHpw; ++h) {
        const float* q = a.q + (std::int64_t(b) * a.query_heads + kv_head * G + head_base + h) * D +
                         lane * S::kDpl;
#pragma unroll
        for (int j = 0; j < S::kDpl; ++j) { q_r[h][j] = q[j] * a.scale; }
    }

    const std::int64_t base = std::int64_t(sequence) * a.seq_stride +
                              std::int64_t(kv_head) * a.head_stride;
    // The K and V rows of a tile in 16-byte words, kLoads per thread; the next tile's words are
    // fetched into registers while the current one is scored.
    constexpr int kWords = D / 8;
    constexpr int kLoads = kTile * kWords / kDecodeThreads;
    static_assert(kTile * kWords % kDecodeThreads == 0);
    TileRegs<KC, kLoads> k_next;
    TileRegs<VC, kLoads> v_next;
    const auto fetch = [&](int tile, int count) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int w = threadIdx.x + u * kDecodeThreads;
            const int r = w / kWords, c = w % kWords;
            const int p    = tile + r;
            const int slot = a.ring != 0 ? p % a.ring : p;
            const std::int64_t at = base + std::int64_t(slot) * D + c * 8;
            tile_fetch(k_next, u, r < count, a.k_cache, at, a.k_plane, sequence, kv_head, slot, c);
            tile_fetch(v_next, u, r < count, a.v_cache, at, a.v_plane, sequence, kv_head, slot, c);
        }
    };
    if (lo < hi) { fetch(lo, min(kTile, hi - lo)); }
    for (int tile = lo; tile < hi; tile += kTile) {
        const int count = min(kTile, hi - tile);
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int w = threadIdx.x + u * kDecodeThreads;
            const int r = w / kWords, c = w % kWords;
            *reinterpret_cast<uint4*>(&k_s[r][c * 8]) = tile_word(k_next, u);
            *reinterpret_cast<uint4*>(&v_s[r][c * 8]) = tile_word(v_next, u);
        }
        __syncthreads();
        if (tile + kTile < hi) { fetch(tile + kTile, min(kTile, hi - tile - kTile)); }
        // Scores: each lane's partial dot products over its kDpl dimensions for the warp's 16 keys,
        // then one butterfly that leaves every key's sum on a pair of lanes (16 shuffles per head
        // instead of 16 full reductions).
        constexpr int kKeysPerGroup = kTile / S::kGroups;
        static_assert(kKeysPerGroup == 16, "the score butterfly reduces 16 keys");
        float part[S::kHpw][kKeysPerGroup];
#pragma unroll
        for (int kk = 0; kk < kKeysPerGroup; ++kk) {
            const int key = key_group * kKeysPerGroup + kk;
            float kf[S::kDpl];
#pragma unroll
            for (int j = 0; j < S::kDpl; j += 2) {
                const float2 f = __half22float2(
                    *reinterpret_cast<const __half2*>(&k_s[key][lane * S::kDpl + j]));
                kf[j]     = f.x;
                kf[j + 1] = f.y;
            }
#pragma unroll
            for (int h = 0; h < S::kHpw; ++h) {
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < S::kDpl; ++j) { dot += q_r[h][j] * kf[j]; }
                part[h][kk] = dot;
            }
        }
        const int lane_key = key_group * kKeysPerGroup + ((lane & 16) ? 8 : 0) + ((lane & 8) ? 4 : 0) +
                             ((lane & 4) ? 2 : 0) + ((lane & 2) ? 1 : 0);
#pragma unroll
        for (int h = 0; h < S::kHpw; ++h) {
            const float dot = key_reduce16(part[h], lane);
            if ((lane & 1) == 0) {
                p_s[head_base + h][lane_key] = lane_key < count ? dot : -INFINITY;
            }
        }
        __syncthreads();
        // Online softmax: warp w takes heads w, w + 4, ...; lane k the tile's key k.
        static_assert(kTile <= 32, "one lane per key of a tile");
        for (int g = warp; g < V; g += kDecodeThreads / 32) {
            const float p     = lane < count ? p_s[g][lane] : -INFINITY;
            float m_tile      = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, o)); }
            const float m_old = m_s[g];
            const float m_new = fmaxf(m_old, m_tile);
            const float scale = expf(m_old - m_new);
            const float e     = lane < count ? expf(p - m_new) : 0.0f;
            if (lane < kTile) { p_s[g][lane] = e; }
            const float sum = warp_sum(e);
            if (lane == 0) {
                l_s[g]     = l_s[g] * scale + sum;
                m_s[g]     = m_new;
                scale_s[g] = scale;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < V; ++g) {
            const float scale = scale_s[g];
#pragma unroll
            for (int j = 0; j < S::kDpt; ++j) { acc[g][j] *= scale; }
        }
        for (int k = 0; k < count; ++k) {
#pragma unroll
            for (int j = 0; j < S::kDpt; ++j) {
                const float v = __half2float(v_s[k][threadIdx.x + j * kDecodeThreads]);
#pragma unroll
                for (int g = 0; g < V; ++g) { acc[g][j] += p_s[g][k] * v; }
            }
        }
        __syncthreads();
    }
    // Partials: [B][Hq][splits] rows of D values, then [B][Hq][splits] (m, l) pairs.
    const std::int64_t rows = std::int64_t(a.batch) * a.query_heads * a.splits;
#pragma unroll
    for (int g = 0; g < V; ++g) {
        const std::int64_t row =
            (std::int64_t(b) * a.query_heads + kv_head * G + g) * a.splits + split;
#pragma unroll
        for (int j = 0; j < S::kDpt; ++j) {
            a.partial[row * D + threadIdx.x + j * kDecodeThreads] = acc[g][j];
        }
        if (threadIdx.x == 0) {
            a.partial[rows * D + 2 * row]     = m_s[g];
            a.partial[rows * D + 2 * row + 1] = l_s[g];
        }
    }
}

// Width 512, eight query heads per KV head (the full-attention blocks): block (row, kv head, split)
// of eight warps, warp h the group's query head h. A tile's 16 keys and values pass through shared
// memory once for all eight heads; each lane holds dimensions [8 lane, 8 lane + 8) and [256 +
// 8 lane, 256 + 8 lane + 8) (conflict-free 16-byte reads), and every warp keeps its own online
// softmax, so a tile costs two barriers (three when the keys are derived from the values).
constexpr int kWideThreads = 256;

template <bool Derived, KvCoding VC = KvCoding::F16>
__launch_bounds__(kWideThreads, 2) __global__ void decode_attention_wide_kernel(DecodeAttentionArgs a) {
    constexpr int D = 512, kTile = 16, kHalf = D / 2;
    const int b        = blockIdx.x; // rows fastest, as decode_attention_kernel
    const int kv_head  = blockIdx.y;
    const int split    = blockIdx.z;
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int position = a.positions[b];
    const int sequence = a.sequences[b];
    const int n_all    = position + 1;
    const int active   = decode_active_splits(n_all, a.splits, kTile);
    if (split >= active) { return; }
    const int chunk = ((n_all + active - 1) / active + kTile - 1) / kTile * kTile;
    const int lo    = split * chunk;
    const int hi    = min(position + 1, lo + chunk);

    __shared__ __align__(16) __half k_s[kTile][D];
    __shared__ __align__(16) __half v_s[kTile][D];

    const int head = kv_head * (kWideThreads / 32) + warp;
    float q_r[16], acc[16];
    {
        const float* q = a.q + (std::int64_t(b) * a.query_heads + head) * D;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q_r[j]     = q[8 * lane + j] * a.scale;
            q_r[8 + j] = q[kHalf + 8 * lane + j] * a.scale;
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) { acc[j] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    const std::int64_t base = std::int64_t(sequence) * a.seq_stride +
                              std::int64_t(kv_head) * a.head_stride;
    constexpr int kWords = D / 8;                         // 16-byte words per row
    constexpr int kLoads = kTile * kWords / kWideThreads; // per thread and matrix
    uint4 k_next[kLoads];
    TileRegs<VC, kLoads> v_next;
    // Derived keys: thread t < 128 turns the 8 pairs [8 (t % 8), 8 (t % 8) + 8) of row t / 8 (the
    // first 64 pairs turn, rope_pairs == 64), with their rotations fetched beside the values.
    static_assert(kTile * 8 <= kWideThreads, "one thread per turned 8-pair chunk");
    float2 cs_next[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { cs_next[j] = make_float2(1.0f, 0.0f); }
    const auto fetch = [&](int tile, int count) {
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int w = threadIdx.x + u * kWideThreads;
            const int r = w / kWords, c = w % kWords;
            const std::int64_t at = base + std::int64_t(tile + r) * D + c * 8;
            k_next[u] = make_uint4(0, 0, 0, 0);
            if constexpr (!Derived) {
                if (r < count) { k_next[u] = *reinterpret_cast<const uint4*>(a.k_cache + at); }
            }
            tile_fetch(v_next, u, r < count, a.v_cache, at, a.v_plane, sequence, kv_head, tile + r, c);
        }
        if constexpr (Derived) {
            const int r = threadIdx.x >> 3, c = threadIdx.x & 7;
            if (threadIdx.x < kTile * 8 && r < count) {
                const float4* src = reinterpret_cast<const float4*>(
                    a.rope + std::int64_t(tile + r) * 64 + 8 * c);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float4 v = src[j];
                    cs_next[2 * j]     = make_float2(v.x, v.y);
                    cs_next[2 * j + 1] = make_float2(v.z, v.w);
                }
            }
        }
    };
    // After the score butterfly lanes 2j and 2j + 1 hold key `lane_key`.
    const int lane_key = ((lane & 16) ? 8 : 0) + ((lane & 8) ? 4 : 0) + ((lane & 4) ? 2 : 0) +
                         ((lane & 2) ? 1 : 0);
    if (lo < hi) { fetch(lo, min(kTile, hi - lo)); }
    for (int tile = lo; tile < hi; tile += kTile) {
        const int count = min(kTile, hi - tile);
#pragma unroll
        for (int u = 0; u < kLoads; ++u) {
            const int w = threadIdx.x + u * kWideThreads;
            const int r = w / kWords, c = w % kWords;
            if constexpr (!Derived) { *reinterpret_cast<uint4*>(&k_s[r][c * 8]) = k_next[u]; }
            *reinterpret_cast<uint4*>(&v_s[r][c * 8]) = tile_word(v_next, u);
        }
        float2 cs[8];
        if constexpr (Derived) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { cs[j] = cs_next[j]; }
        }
        __syncthreads();
        if (tile + kTile < hi) { fetch(tile + kTile, min(kTile, hi - tile - kTile)); }
        if constexpr (Derived) {
            // The tile's keys from its values, 8 pairs (i, i + 256) at a time: the turned chunks
            // (pairs below 64), then the chunks that only scale (keys past `count` are never
            // scored).
            const auto chunk_keys = [&](int r, int i, const float2* turn) {
                const uint4 w0 = *reinterpret_cast<const uint4*>(&v_s[r][i]);
                const uint4 w1 = *reinterpret_cast<const uint4*>(&v_s[r][i + kHalf]);
                const __half2* h0 = reinterpret_cast<const __half2*>(&w0);
                const __half2* h1 = reinterpret_cast<const __half2*>(&w1);
                uint4 o0, o1;
                __half2* k0 = reinterpret_cast<__half2*>(&o0);
                __half2* k1 = reinterpret_cast<__half2*>(&o1);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f0 = __half22float2(h0[j]), f1 = __half22float2(h1[j]);
                    const float n0 = f0.x * a.k_norm[i + 2 * j], n1 = f0.y * a.k_norm[i + 2 * j + 1];
                    const float m0 = f1.x * a.k_norm[i + kHalf + 2 * j];
                    const float m1 = f1.y * a.k_norm[i + kHalf + 2 * j + 1];
                    if (turn != nullptr) {
                        const float2 c0 = turn[2 * j], c1 = turn[2 * j + 1];
                        k0[j] = __floats2half2_rn(n0 * c0.x - m0 * c0.y, n1 * c1.x - m1 * c1.y);
                        k1[j] = __floats2half2_rn(m0 * c0.x + n0 * c0.y, m1 * c1.x + n1 * c1.y);
                    } else {
                        k0[j] = __floats2half2_rn(n0, n1);
                        k1[j] = __floats2half2_rn(m0, m1);
                    }
                }
                *reinterpret_cast<uint4*>(&k_s[r][i])         = o0;
                *reinterpret_cast<uint4*>(&k_s[r][i + kHalf]) = o1;
            };
            if (threadIdx.x < kTile * 8) { chunk_keys(threadIdx.x >> 3, 8 * (threadIdx.x & 7), cs); }
            // The 24 scaled chunks (pairs 64..255) of each of the 16 rows.
            for (int w = threadIdx.x; w < kTile * 24; w += kWideThreads) {
                chunk_keys(w / 24, 64 + 8 * (w % 24), nullptr);
            }
            __syncthreads();
        }
        // The 16 keys' scores: the lane's 16 dimensions each, then one butterfly.
        float part[kTile];
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const uint4 w0 = *reinterpret_cast<const uint4*>(&k_s[kk][8 * lane]);
            const uint4 w1 = *reinterpret_cast<const uint4*>(&k_s[kk][kHalf + 8 * lane]);
            const __half2* h0 = reinterpret_cast<const __half2*>(&w0);
            const __half2* h1 = reinterpret_cast<const __half2*>(&w1);
            float dot = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f0 = __half22float2(h0[j]), f1 = __half22float2(h1[j]);
                dot += q_r[2 * j] * f0.x + q_r[2 * j + 1] * f0.y;
                dot += q_r[8 + 2 * j] * f1.x + q_r[8 + 2 * j + 1] * f1.y;
            }
            part[kk] = dot;
        }
        const float score = key_reduce16(part, lane);
        const float sc    = lane_key < count ? score : -INFINITY;
        float m_tile      = sc;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, o));
        }
        const float m_new = fmaxf(m, m_tile);
        const float scale = expf(m - m_new);
        const float p     = lane_key < count ? expf(sc - m_new) : 0.0f;
        // Every key's probability sits on two lanes.
        l = l * scale + 0.5f * warp_sum(p);
        m = m_new;
#pragma unroll
        for (int j = 0; j < 16; ++j) { acc[j] *= scale; }
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const int src  = ((kk & 8) ? 16 : 0) | ((kk & 4) ? 8 : 0) | ((kk & 2) ? 4 : 0) |
                             ((kk & 1) ? 2 : 0);
            const float pk = __shfl_sync(0xFFFFFFFFu, p, src);
            const uint4 w0 = *reinterpret_cast<const uint4*>(&v_s[kk][8 * lane]);
            const uint4 w1 = *reinterpret_cast<const uint4*>(&v_s[kk][kHalf + 8 * lane]);
            const __half2* h0 = reinterpret_cast<const __half2*>(&w0);
            const __half2* h1 = reinterpret_cast<const __half2*>(&w1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f0 = __half22float2(h0[j]), f1 = __half22float2(h1[j]);
                acc[2 * j] += pk * f0.x;
                acc[2 * j + 1] += pk * f0.y;
                acc[8 + 2 * j] += pk * f1.x;
                acc[8 + 2 * j + 1] += pk * f1.y;
            }
        }
        __syncthreads();
    }
    // Partials as decode_attention_kernel leaves them: [B][Hq][splits] rows of D, then (m, l).
    const std::int64_t rows = std::int64_t(a.batch) * a.query_heads * a.splits;
    const std::int64_t row  = (std::int64_t(b) * a.query_heads + head) * a.splits + split;
    float* out = a.partial + row * D;
#pragma unroll
    for (int j = 0; j < 8; j += 4) {
        *reinterpret_cast<float4*>(out + 8 * lane + j) =
            make_float4(acc[j], acc[j + 1], acc[j + 2], acc[j + 3]);
        *reinterpret_cast<float4*>(out + kHalf + 8 * lane + j) =
            make_float4(acc[8 + j], acc[8 + j + 1], acc[8 + j + 2], acc[8 + j + 3]);
    }
    if (lane == 0) {
        a.partial[rows * D + 2 * row]     = m;
        a.partial[rows * D + 2 * row + 1] = l;
    }
}

// Block (head, row, 128 dimensions): merges the splits of one query head. Rotated (a rotated value
// plane, D 256): block (head, row) of 256 threads, which turn the merged row back before its output.
constexpr int kDecodeCombineThreads = 128; // at least the largest split count

template <int D, bool Rotated = false>
__global__ void decode_combine_kernel(DecodeAttentionArgs a) {
    static_assert(!Rotated || D == kKvBlockDims, "a rotated row is one 256-wide block");
    const int head = blockIdx.x, b = blockIdx.y;
    const int d    = blockIdx.z * kDecodeCombineThreads + threadIdx.x;
    const std::int64_t rows  = std::int64_t(a.batch) * a.query_heads * a.splits;
    const std::int64_t first = (std::int64_t(b) * a.query_heads + head) * a.splits;
    const int position = a.positions[b];
    const int lo_all   = a.ring != 0 ? max(0, position - a.window + 1) : 0;
    const int active   = decode_active_splits(position + 1 - lo_all, a.splits, D == 512 ? 16 : 32);
    // Each split's weight exp(m_s - m) once, in shared memory (blockDim >= splits).
    __shared__ float weight[Rotated ? kKvBlockDims : kDecodeCombineThreads];
    __shared__ float max_s, inv_l;
    const int s0   = threadIdx.x;
    const float ms = s0 < active ? a.partial[rows * D + 2 * (first + s0)] : -INFINITY;
    float m        = ms;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o)); }
    if ((threadIdx.x & 31) == 0) { weight[threadIdx.x >> 5] = m; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float all = -INFINITY;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) { all = fmaxf(all, weight[w]); }
        max_s = all;
    }
    __syncthreads();
    const float w_s = ms != -INFINITY ? expf(ms - max_s) : 0.0f;
    const float l_s = w_s != 0.0f ? a.partial[rows * D + 2 * (first + s0) + 1] * w_s : 0.0f;
    const float l   = block_sum(l_s);
    weight[threadIdx.x] = w_s;
    if (threadIdx.x == 0) { inv_l = 1.0f / l; }
    __syncthreads();
    {
        float o = 0.0f;
        for (int s = 0; s < active; ++s) {
            const float w = weight[s];
            if (w != 0.0f) { o += a.partial[(first + s) * D + d] * w; }
        }
        float y = o * inv_l;
        if constexpr (Rotated) {
            __shared__ float row[kKvBlockDims];
            row[threadIdx.x] = y;
            __syncthreads();
            if (threadIdx.x < 32) {
                float v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) { v[i] = row[threadIdx.x + 32 * i]; }
                ops::normalized_hadamard_d256_inplace(v, int(threadIdx.x));
#pragma unroll
                for (int i = 0; i < 8; ++i) { row[threadIdx.x + 32 * i] = v[i]; }
            }
            __syncthreads();
            y = row[threadIdx.x];
        }
        if (a.out_q.qs == nullptr) {
            a.out[(std::int64_t(b) * a.query_heads + head) * D + d] = __float2bfloat16(y);
            return;
        }
        // One warp holds 32 consecutive values: one q8_1 block of the o-projection input.
        const int width = a.query_heads * D, at = head * D + d;
        float amax = fabsf(y), sum = y;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
            sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
        }
        const float dq = amax / 127.0f;
        a.out_q.qs[std::int64_t(b) * width + at] =
            amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(y / dq));
        if ((threadIdx.x & 31) == 0) {
            a.out_q.ds[std::int64_t(b) * (width / 32) + at / 32] =
                make_half2(__float2half(dq), __float2half(sum));
        }
    }
}

// h += rms_norm(o) * w_post; then the FFN inputs of the new h (see AttnResidualArgs).
__global__ void attn_residual_kernel(AttnResidualArgs a) {
    const int q = a.hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * a.hidden;
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 o    = i < q ? load_bf16x4(a.o + row + 4 * i) : zero;
    const float inv_o = rsqrtf(block_sum(dot4(o, o)) / float(a.hidden) + a.eps);
    float4 h = zero;
    if (i < q) {
        h = add4(ld4(a.h + row, i), mul4(scale4(o, inv_o), ld4(a.w_post, i)));
        reinterpret_cast<float4*>(a.h + row)[i] = h;
    }
    const float inv = rsqrtf(block_sum(dot4(h, h)) / float(a.hidden) + a.eps);
    const bool valid = i < q;
    const float4 n   = scale4(h, inv);
    const float4 mlp = valid ? mul4(n, ld4(a.ffn_norm, i)) : zero;
    if (valid && a.mlp_in != nullptr) { store_bf16x4(a.mlp_in + row + 4 * i, mlp); }
    if (a.mlp_q.qs != nullptr) { store_q8x4(a.mlp_q, a.hidden, blockIdx.x, i, mlp, valid); }
    if (a.pre_norm != nullptr) {
        const float4 moe = valid ? mul4(n, ld4(a.pre_norm, i)) : zero;
        if (valid && a.moe_in != nullptr) { store_bf16x4(a.moe_in + row + 4 * i, moe); }
        if (a.moe_q.qs != nullptr) { store_q8x4(a.moe_q, a.hidden, blockIdx.x, i, moe, valid); }
    }
    if (valid && a.router_x != nullptr) {
        reinterpret_cast<float4*>(a.router_x + row)[i] =
            scale4(mul4(n, ld4(a.router_scale, i)), rsqrtf(float(a.hidden)));
    }
}

// Router logits of a prompt chunk, a tiled FP32 product: block (expert tile, token tile) computes
// kRouteTileT tokens x kRouteTileE experts over K steps of kRouteTileK staged in shared memory;
// thread (row group ty, expert tx) keeps kRouteTileT / 8 token sums of one expert.
constexpr int kRouteTileT = 64, kRouteTileE = 32, kRouteTileK = 32;
constexpr int kRouteTokens = 8; // decode: the small kernel below covers up to this many tokens

__global__ void route_logits_kernel(const float* __restrict__ x, int hidden, int tokens,
                                    const float* __restrict__ router, int experts,
                                    float* __restrict__ logits) {
    __shared__ float xs[kRouteTileT][kRouteTileK + 1];
    __shared__ float rs[kRouteTileE][kRouteTileK + 1];
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5; // 8 warps
    const int e0 = blockIdx.x * kRouteTileE, t0 = blockIdx.y * kRouteTileT;
    constexpr int kPer = kRouteTileT / 8;
    // Routing turns on near ties, so each K tile's partial sum enters a compensated total: the
    // logits keep the accuracy of the decode kernel's lane-split sums.
    float acc[kPer] = {}, err[kPer] = {};
    for (int k0 = 0; k0 < hidden; k0 += kRouteTileK) {
        for (int i = threadIdx.x; i < kRouteTileT * kRouteTileK; i += 256) {
            const int r = i / kRouteTileK, c = i % kRouteTileK;
            const int t = t0 + r, kk = k0 + c;
            xs[r][c] = t < tokens && kk < hidden ? x[std::int64_t(t) * hidden + kk] : 0.0f;
        }
        for (int i = threadIdx.x; i < kRouteTileE * kRouteTileK; i += 256) {
            const int r = i / kRouteTileK, c = i % kRouteTileK;
            const int e = e0 + r, kk = k0 + c;
            rs[r][c] = e < experts && kk < hidden ? router[std::int64_t(e) * hidden + kk] : 0.0f;
        }
        __syncthreads();
        float part[kPer] = {};
#pragma unroll 8
        for (int c = 0; c < kRouteTileK; ++c) {
            const float w = rs[tx][c];
#pragma unroll
            for (int i = 0; i < kPer; ++i) { part[i] = fmaf(xs[ty * kPer + i][c], w, part[i]); }
        }
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const float y = part[i] - err[i];
            const float t = acc[i] + y;
            err[i]        = (t - acc[i]) - y;
            acc[i]        = t;
        }
        __syncthreads();
    }
    const int e = e0 + tx;
    if (e >= experts) { return; }
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int t = t0 + ty * kPer + i;
        if (t < tokens) { logits[std::int64_t(t) * experts + e] = acc[i]; }
    }
}

// Router logits for up to kRouteTokens tokens (decode): one block per expert, its row split over
// four warps, so every SM streams part of the router.
constexpr int kRouteSmallThreads = 256;

__global__ void route_logits_small_kernel(const float* __restrict__ x, int hidden, int tokens,
                                          const float* __restrict__ router, int experts,
                                          float* __restrict__ logits) {
    __shared__ float part[kRouteSmallThreads / 32][kRouteTokens];
    const int e = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float4* w = reinterpret_cast<const float4*>(router + std::int64_t(e) * hidden);
    const int quads = hidden / 4;
    float dot[kRouteTokens] = {};
    for (int i = threadIdx.x; i < quads; i += kRouteSmallThreads) {
        const float4 wv = w[i];
#pragma unroll
        for (int k = 0; k < kRouteTokens; ++k) {
            if (k < tokens) {
                dot[k] += dot4(wv, reinterpret_cast<const float4*>(x)[std::int64_t(k) * quads + i]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kRouteTokens; ++k) {
        const float v = warp_sum(dot[k]);
        if (lane == 0) { part[warp][k] = v; }
    }
    __syncthreads();
    if (threadIdx.x < tokens) {
        float v = 0.0f;
#pragma unroll
        for (int w4 = 0; w4 < kRouteSmallThreads / 32; ++w4) { v += part[w4][threadIdx.x]; }
        logits[std::int64_t(threadIdx.x) * experts + e] = v;
    }
}

// One block per token, one thread per expert: each expert's rank among the logits (larger first,
// lower id on ties) selects the top_k; their softmax times expert_scale.
constexpr int kTopkMaxExperts = 256;

__global__ void route_topk_kernel(const float* __restrict__ logits, int experts, int top_k,
                                  const float* __restrict__ expert_scale,
                                  std::int32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ float v[kTopkMaxExperts];
    __shared__ float chosen[8];
    __shared__ int chosen_id[8];
    const int t = blockIdx.x, e = threadIdx.x;
    if (e < experts) { v[e] = logits[std::int64_t(t) * experts + e]; }
    __syncthreads();
    if (e < experts) {
        const float mine = v[e];
        int rank         = 0;
        for (int j = 0; j < experts; ++j) {
            const float other = v[j];
            rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
        }
        if (rank < top_k) {
            chosen[rank]    = mine;
            chosen_id[rank] = e;
        }
    }
    __syncthreads();
    if (e < top_k) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += expf(chosen[k] - chosen[0]); }
        ids[std::int64_t(t) * top_k + e]     = chosen_id[e];
        weights[std::int64_t(t) * top_k + e] =
            expf(chosen[e] - chosen[0]) / sum * expert_scale[chosen_id[e]];
    }
}

// Decode routing after the logits, in one block of at least max(experts, tokens * top_k)
// threads: each token's top_k as route_topk_kernel picks them, then the pairs grouped by expert as
// moe_sort_routes lists them (bounds, pairs per expert, active experts ascending).
__global__ void route_select_kernel(const float* __restrict__ logits, int tokens, int experts,
                                    int top_k, const float* __restrict__ expert_scale,
                                    std::int32_t* __restrict__ ids, float* __restrict__ weights,
                                    std::int32_t* __restrict__ bounds,
                                    std::int32_t* __restrict__ sorted,
                                    std::int32_t* __restrict__ active,
                                    std::int32_t* __restrict__ active_count) {
    __shared__ float v[kTopkMaxExperts];
    __shared__ float chosen[kRouteTokens * 8];
    __shared__ int chosen_id[kRouteTokens * 8];
    __shared__ int cursor[kTopkMaxExperts];
    __shared__ int warp_total[kTopkMaxExperts / 32][2];
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    for (int t = 0; t < tokens; ++t) {
        if (e < experts) { v[e] = logits[std::int64_t(t) * experts + e]; }
        __syncthreads();
        if (e < experts) {
            const float mine = v[e];
            int rank         = 0;
            for (int j = 0; j < experts; ++j) {
                const float other = v[j];
                rank += (other > mine || (other == mine && j < e)) ? 1 : 0;
            }
            if (rank < top_k) {
                chosen[t * top_k + rank]    = mine;
                chosen_id[t * top_k + rank] = e;
            }
        }
        __syncthreads();
    }
    const int pairs = tokens * top_k;
    if (e < pairs) {
        const int base = (e / top_k) * top_k;
        float sum      = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += expf(chosen[base + k] - chosen[base]); }
        ids[e]     = chosen_id[e];
        weights[e] = expf(chosen[e] - chosen[base]) / sum * expert_scale[chosen_id[e]];
    }
    // Exclusive scans over the experts of their pair counts and of whether they have any.
    int count = 0;
    if (e < experts) {
        for (int p = 0; p < pairs; ++p) { count += chosen_id[p] == e ? 1 : 0; }
    }
    int c = count, f = count != 0 ? 1 : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int uc = __shfl_up_sync(0xFFFFFFFFu, c, o), uf = __shfl_up_sync(0xFFFFFFFFu, f, o);
        if (lane >= o) {
            c += uc;
            f += uf;
        }
    }
    if (lane == 31) {
        warp_total[warp][0] = c;
        warp_total[warp][1] = f;
    }
    __syncthreads();
    int before_c = 0, before_f = 0;
    for (int w = 0; w < warp; ++w) {
        before_c += warp_total[w][0];
        before_f += warp_total[w][1];
    }
    const int start = before_c + c - count, slot = before_f + f - (count != 0 ? 1 : 0);
    if (e < experts) {
        bounds[e] = start;
        cursor[e] = start;
        if (count != 0) { active[slot] = e; }
        if (e == experts - 1) {
            bounds[experts] = start + count;
            *active_count   = slot + (count != 0 ? 1 : 0);
        }
    }
    __syncthreads();
    if (e < pairs) { sorted[atomicAdd(cursor + chosen_id[e], 1)] = e; }
}

__global__ void geglu_rows_kernel(const float* __restrict__ x, int pairs, int width,
                                  __nv_bfloat16* __restrict__ out) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::int64_t(pairs) * width) { return; }
    const std::int64_t p = i / width, r = i % width;
    const float* row     = x + p * 2 * width;
    out[i]               = __float2bfloat16(gelu_tanh(row[r]) * row[width + r]);
}

__global__ void ffn_combine_kernel(FfnCombineArgs a) {
    const int q = a.hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * a.hidden;
    const float hidden = float(a.hidden);
    const float4 zero  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 y = i < q ? load_bf16x4(a.dense + row + 4 * i) : zero;
    const bool moe = a.moe_rows != nullptr || a.moe_fixed != nullptr;
    float4 m       = zero;
    if (moe && i < q) {
        if (a.moe_rows != nullptr) {
            for (int k = 0; k < a.top_k; ++k) {
                const std::int64_t pair = std::int64_t(blockIdx.x) * a.top_k + k;
                m = add4(m, scale4(ld4(a.moe_rows + pair * a.hidden, i), a.moe_weights[pair]));
            }
        } else {
            m = fixed4(a.moe_fixed + row, i);
        }
    }
    if (a.w_dense != nullptr || moe) {
        // The dense and expert norms in one reduction.
        const float2 ss = block_sum2(dot4(y, y), dot4(m, m));
        if (i < q) {
            if (a.w_dense != nullptr) { y = mul4(scale4(y, rsqrtf(ss.x / hidden + a.eps)), ld4(a.w_dense, i)); }
            if (moe) { y = add4(y, mul4(scale4(m, rsqrtf(ss.y / hidden + a.eps)), ld4(a.w_moe, i))); }
        }
    }
    const float inv_y = rsqrtf(block_sum(dot4(y, y)) / hidden + a.eps);
    float4 h = zero;
    if (i < q) {
        h = scale4(add4(ld4(a.h + row, i), mul4(scale4(y, inv_y), ld4(a.w_post, i))), a.scalar[0]);
        reinterpret_cast<float4*>(a.h + row)[i] = h;
    }
    if (a.next_norm == nullptr) { return; }
    const float inv  = rsqrtf(block_sum(dot4(h, h)) / hidden + a.eps);
    const float4 nxt = i < q ? mul4(scale4(h, inv), ld4(a.next_norm, i)) : zero;
    if (i < q && a.next_out != nullptr) { store_bf16x4(a.next_out + row + 4 * i, nxt); }
    if (a.next_q.qs != nullptr) { store_q8x4(a.next_q, a.hidden, blockIdx.x, i, nxt, i < q); }
}

__global__ void add_fixed_kernel(unsigned long long* sums, const unsigned long long* products,
                                 const std::int32_t* flag, std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || *flag == 0) { return; }
    sums[i] += products[i];
}

__global__ void softcap_kernel(const float* x, std::int64_t n, float cap, __nv_bfloat16* out) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const float v = cap > 0.0f ? cap * tanhf(x[i] / cap) : x[i];
    out[i]        = __float2bfloat16(v);
}

unsigned blocks_for(std::int64_t n, int threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
}

// Block r: one row in 16-byte words.
__global__ void copy_rows_kernel(const __nv_bfloat16* __restrict__ src,
                                 const std::int32_t* __restrict__ src_rows,
                                 __nv_bfloat16* __restrict__ dst,
                                 const std::int32_t* __restrict__ dst_rows, int width) {
    const int r  = blockIdx.x;
    const auto* from = reinterpret_cast<const uint4*>(
        src + std::int64_t(src_rows != nullptr ? src_rows[r] : r) * width);
    auto* to = reinterpret_cast<uint4*>(dst + std::int64_t(dst_rows != nullptr ? dst_rows[r] : r) *
                                                  width);
    for (int w = threadIdx.x; w < width / 8; w += blockDim.x) { to[w] = from[w]; }
}

__global__ void stem_rows_kernel(const float* __restrict__ embedding,
                                 const __nv_bfloat16* __restrict__ hidden, int width,
                                 __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x;
    __nv_bfloat16* row = out + std::int64_t(r) * 2 * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        row[i]         = __float2bfloat16(embedding[std::int64_t(r) * width + i]);
        row[width + i] = hidden[std::int64_t(r) * width + i];
    }
}

} // namespace

void scale(float* x, std::int64_t n, float s, cudaStream_t stream) {
    if (n <= 0) { return; }
    scale_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, n, s);
    check_launch("scale");
}

int row_threads(int hidden) {
    if (hidden % 4 != 0 || hidden / 4 > 1024) {
        throw std::invalid_argument("gemma4 row kernels: hidden must be whole float4s, at most 4096");
    }
    return (hidden / 4 + 31) / 32 * 32;
}

void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q) {
    rms_norm_kernel<<<tokens, row_threads(hidden), 0, stream>>>(x, w, hidden, eps, out, q);
    check_launch("rms_norm");
}

void qkv_post(const QkvArgs& args, cudaStream_t stream) {
    if (args.head_dim != 256 && args.head_dim != 512) {
        throw std::invalid_argument("gemma4 qkv_post: head width must be 256 or 512");
    }
    const dim3 grid(args.tokens, args.query_heads + args.kv_heads, 1);
    qkv_post_kernel<<<grid, args.head_dim / 2, 0, stream>>>(args);
    check_launch("qkv_post");
}

void build_mask(__half* mask, int tokens, int slots, int first, int ring, int window,
                cudaStream_t stream, const std::int32_t* blocks, const std::int32_t* starts) {
    const dim3 grid(blocks_for(slots, 256), tokens, 1);
    build_mask_kernel<<<grid, 256, 0, stream>>>(mask, tokens, slots, first, ring, window, blocks,
                                                starts);
    check_launch("build_mask");
}

void ring_copy(const RingCopyArgs& a, cudaStream_t stream) {
    if (a.end <= a.begin) { return; }
    if (a.head_dim % 8 != 0) { throw std::invalid_argument("gemma4 ring_copy: head width"); }
    ring_copy_kernel<<<dim3(a.end - a.begin, a.kv_heads, 1), 64, 0, stream>>>(a);
    check_launch("ring_copy");
}

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    f32_to_bf16_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(in, out, n);
    check_launch("f32_to_bf16");
}

std::size_t decode_attention_workspace_floats(int batch, int query_heads, int head_dim,
                                              int splits) {
    const std::size_t rows = std::size_t(batch) * query_heads * splits;
    return rows * head_dim + 2 * rows;
}

namespace {

template <KvCoding KC, KvCoding VC>
void launch_sliding(const DecodeAttentionArgs& args, dim3 grid, cudaStream_t stream) {
    decode_attention_kernel<256, 2, KC, VC><<<grid, kDecodeThreads, 0, stream>>>(args);
}

// The sliding blocks' (key, value) codings of the KV formats (kv_format.h).
void dispatch_sliding(const DecodeAttentionArgs& args, dim3 grid, cudaStream_t stream) {
    const KvCoding k = args.k_plane.codes != nullptr ? args.k_plane.coding : KvCoding::F16;
    const KvCoding v = args.v_plane.codes != nullptr ? args.v_plane.coding : KvCoding::F16;
    using C = KvCoding;
    if (k == C::F16 && v == C::F16) { return launch_sliding<C::F16, C::F16>(args, grid, stream); }
    if (k == C::Int8G64 && v == C::Int8G64) { return launch_sliding<C::Int8G64, C::Int8G64>(args, grid, stream); }
    if (k == C::Fp8Row && v == C::Fp8Row) { return launch_sliding<C::Fp8Row, C::Fp8Row>(args, grid, stream); }
    if (k == C::Int8G64 && v == C::Int4G32) { return launch_sliding<C::Int8G64, C::Int4G32>(args, grid, stream); }
    if (k == C::Nvfp4G16 && v == C::Nvfp4G16) { return launch_sliding<C::Nvfp4G16, C::Nvfp4G16>(args, grid, stream); }
    if (k == C::Fp8Row && v == C::Nvfp4G16) { return launch_sliding<C::Fp8Row, C::Nvfp4G16>(args, grid, stream); }
    if (k == C::Lloyd4 && v == C::Int4G32) { return launch_sliding<C::Lloyd4, C::Int4G32>(args, grid, stream); }
    if (k == C::Int4E8 && v == C::Int4G32) { return launch_sliding<C::Int4E8, C::Int4G32>(args, grid, stream); }
    if (k == C::RootE8 && v == C::Int4G32) { return launch_sliding<C::RootE8, C::Int4G32>(args, grid, stream); }
    throw std::invalid_argument("gemma4 decode attention: no kernel for this key/value coding");
}

// The full-attention blocks' value plane, keys derived from it.
void dispatch_derived(const DecodeAttentionArgs& args, dim3 grid, cudaStream_t stream) {
    const KvCoding v = args.v_plane.codes != nullptr ? args.v_plane.coding : KvCoding::F16;
    using C = KvCoding;
    switch (v) {
    case C::F16: decode_attention_wide_kernel<true><<<grid, kWideThreads, 0, stream>>>(args); return;
    case C::Int8G64: decode_attention_wide_kernel<true, C::Int8G64><<<grid, kWideThreads, 0, stream>>>(args); return;
    case C::Int4G32: decode_attention_wide_kernel<true, C::Int4G32><<<grid, kWideThreads, 0, stream>>>(args); return;
    case C::Fp8Row: decode_attention_wide_kernel<true, C::Fp8Row><<<grid, kWideThreads, 0, stream>>>(args); return;
    case C::Nvfp4G16: decode_attention_wide_kernel<true, C::Nvfp4G16><<<grid, kWideThreads, 0, stream>>>(args); return;
    default: break;
    }
    throw std::invalid_argument("gemma4 decode attention: no derived-key kernel for this coding");
}

} // namespace

void decode_attention(const DecodeAttentionArgs& args, cudaStream_t stream) {
    const int group = args.query_heads / args.kv_heads;
    if (args.splits > kDecodeCombineThreads) {
        throw std::invalid_argument("gemma4 decode attention: too many splits");
    }
    // Block (row, kv head, split): the rows of a verification, one sequence's consecutive
    // positions, run side by side over the same key ranges and share them through L2.
    const dim3 grid(args.batch, args.kv_heads, args.splits);
    const bool k_planed = args.k_plane.codes != nullptr;
    const bool v_planed = args.v_plane.codes != nullptr;
    const bool derived  = args.k_cache == nullptr && !k_planed;
    if (derived && (args.k_norm == nullptr || args.rope == nullptr || args.ring != 0 ||
                    args.rope_pairs <= 0 || args.rope_pairs > args.head_dim / 2)) {
        throw std::invalid_argument("gemma4 decode attention: derived keys need a linear cache, "
                                    "k_norm and a rope table");
    }
    if (args.head_dim == 256 && group == 2 && !derived) {
        dispatch_sliding(args, grid, stream);
        check_launch("decode_attention");
        if (v_planed && args.v_plane.rotated) {
            decode_combine_kernel<256, true>
                <<<dim3(args.query_heads, args.batch, 1), kKvBlockDims, 0, stream>>>(args);
        } else {
            decode_combine_kernel<256>
                <<<dim3(args.query_heads, args.batch, 256 / kDecodeCombineThreads),
                   kDecodeCombineThreads, 0, stream>>>(args);
        }
    } else if (args.head_dim == 512 && group == 8 && args.ring == 0) {
        if (derived) {
            if (args.rope_pairs != 64) {
                throw std::invalid_argument("gemma4 decode attention: derived keys turn 64 pairs");
            }
            if (v_planed && args.v_plane.rotated) {
                throw std::invalid_argument("gemma4 decode attention: derived keys need an "
                                            "unrotated value plane");
            }
            dispatch_derived(args, grid, stream);
        } else if (!k_planed && !v_planed) {
            decode_attention_wide_kernel<false><<<grid, kWideThreads, 0, stream>>>(args);
        } else {
            throw std::invalid_argument("gemma4 decode attention: width 512 stores no quantized keys");
        }
        check_launch("decode_attention");
        decode_combine_kernel<512><<<dim3(args.query_heads, args.batch, 512 / kDecodeCombineThreads),
                                     kDecodeCombineThreads, 0, stream>>>(args);
    } else {
        throw std::invalid_argument("gemma4 decode attention: unsupported head geometry");
    }
    check_launch("decode_combine");
}

void attn_residual(const AttnResidualArgs& args, cudaStream_t stream) {
    attn_residual_kernel<<<args.tokens, row_threads(args.hidden), 0, stream>>>(args);
    check_launch("attn_residual");
}

void route(const float* x, int hidden, int tokens, const float* router, const float* expert_scale,
           int experts, int top_k, float* logits, std::int32_t* ids, float* weights,
           cudaStream_t stream) {
    if (top_k > 8 || top_k > experts || experts > kTopkMaxExperts || hidden % 4 != 0) {
        throw std::invalid_argument("gemma4 route: at most 8 of 256 experts over whole float4s");
    }
    if (tokens <= kRouteTokens) {
        route_logits_small_kernel<<<experts, kRouteSmallThreads, 0, stream>>>(x, hidden, tokens,
                                                                             router, experts, logits);
    } else {
        const dim3 grid(blocks_for(experts, kRouteTileE), blocks_for(tokens, kRouteTileT), 1);
        route_logits_kernel<<<grid, 256, 0, stream>>>(x, hidden, tokens, router, experts, logits);
    }
    check_launch("route logits");
    route_topk_kernel<<<tokens, (experts + 31) / 32 * 32, 0, stream>>>(logits, experts, top_k,
                                                                       expert_scale, ids, weights);
    check_launch("route top-k");
}

void route_select(const float* x, int hidden, int tokens, const float* router,
                  const float* expert_scale, int experts, int top_k, float* logits,
                  std::int32_t* ids, float* weights, std::int32_t* bounds, std::int32_t* sorted,
                  std::int32_t* active, std::int32_t* active_count, cudaStream_t stream) {
    if (tokens > kRouteTokens || top_k > 8 || top_k > experts || experts > kTopkMaxExperts ||
        hidden % 4 != 0) {
        throw std::invalid_argument("gemma4 route_select: at most 8 tokens, 8 of 256 experts");
    }
    route_logits_small_kernel<<<experts, kRouteSmallThreads, 0, stream>>>(x, hidden, tokens, router,
                                                                         experts, logits);
    check_launch("route logits");
    const int threads = (std::max(experts, tokens * top_k) + 31) / 32 * 32;
    route_select_kernel<<<1, threads, 0, stream>>>(logits, tokens, experts, top_k, expert_scale, ids,
                                                   weights, bounds, sorted, active, active_count);
    check_launch("route select");
}

void geglu_rows(const float* x, int pairs, int width, __nv_bfloat16* out, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(pairs) * width;
    if (n <= 0) { return; }
    geglu_rows_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, pairs, width, out);
    check_launch("geglu_rows");
}

void ffn_combine(const FfnCombineArgs& args, cudaStream_t stream) {
    ffn_combine_kernel<<<args.tokens, row_threads(args.hidden), 0, stream>>>(args);
    check_launch("ffn_combine");
}

void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    add_fixed_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        sums, static_cast<const unsigned long long*>(products), flag, n);
    check_launch("add_fixed");
}

void softcap(const float* x, std::int64_t n, float cap, __nv_bfloat16* out, cudaStream_t stream) {
    if (n <= 0) { return; }
    softcap_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, n, cap, out);
    check_launch("softcap");
}

void rope_table(const float* inv_freq, int pairs, int positions, float2* rope, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(pairs) * positions;
    if (n <= 0) { return; }
    rope_table_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(inv_freq, pairs, positions, rope);
    check_launch("rope_table");
}

void derive_keys(const __half* v, std::int64_t head_stride, int kv_heads, int head_dim, int count,
                 const float* k_norm, const float2* rope, int rope_pairs, __half* k,
                 cudaStream_t stream, const std::int32_t* turns, int turned_from) {
    if (count <= 0) { return; }
    derive_keys_kernel<<<dim3(count, kv_heads, 1), head_dim / 2, 0, stream>>>(
        v, head_stride, head_dim, k_norm, rope, rope_pairs, k, turns, turned_from);
    check_launch("derive_keys");
}

void copy_rows(const __nv_bfloat16* src, const std::int32_t* src_rows, __nv_bfloat16* dst,
               const std::int32_t* dst_rows, int rows, int width, cudaStream_t stream) {
    if (rows <= 0) { return; }
    if (width % 8 != 0) { throw std::invalid_argument("gemma4 copy_rows: width"); }
    copy_rows_kernel<<<rows, 128, 0, stream>>>(src, src_rows, dst, dst_rows, width);
    check_launch("copy_rows");
}

void stem_rows(const float* embedding, const __nv_bfloat16* hidden, int rows, int width,
               __nv_bfloat16* out, cudaStream_t stream) {
    if (rows <= 0) { return; }
    stem_rows_kernel<<<rows, 256, 0, stream>>>(embedding, hidden, width, out);
    check_launch("stem_rows");
}

} // namespace ninfer::models::gemma4::kernels
