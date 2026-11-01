#include "models/flint/kernels.h"

#include "core/device.h"
#include "ops/common/silu.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::flint::kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("flint ") + what + ": " + cudaGetErrorString(status));
    }
}

unsigned blocks_for(std::int64_t n, int threads) {
    return static_cast<unsigned>((n + threads - 1) / threads);
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

// Maximum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it.
__device__ float block_max(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o)); }
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float m = lane < int(blockDim.x >> 5) ? partial[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o)); }
        if (lane == 0) { total = m; }
    }
    __syncthreads();
    return total;
}

// Row kernels hold one float4 of a [hidden] row per thread (hidden % 4 == 0, hidden / 4 <= 1024).
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

int row_threads(int hidden) {
    if (hidden % 4 != 0 || hidden / 4 > 1024) {
        throw std::invalid_argument("flint row kernels: hidden must be whole float4s, at most 4096");
    }
    return (hidden / 4 + 31) / 32 * 32;
}

__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                int hidden, float eps, __nv_bfloat16* __restrict__ out, Q8Rows q8,
                                float* __restrict__ f32) {
    const int q      = hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * hidden;
    const float4 v   = i < q ? ld4(x + row, i) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float inv  = rsqrtf(block_sum(dot4(v, v)) / float(hidden) + eps);
    float4 y         = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < q) {
        y = mul4(scale4(v, inv), ld4(w, i));
        if (out != nullptr) { store_bf16x4(out + row + 4 * i, y); }
        if (f32 != nullptr) { reinterpret_cast<float4*>(f32 + row)[i] = y; }
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
    const int rows   = (a.query_heads + 2 * a.kv_heads) * d;
    const float* row = a.qkv + std::int64_t(t) * rows;
    float c = 1.0f, s = 0.0f;
    if (a.inv_freq != nullptr) { sincosf(float(a.positions[t]) * a.inv_freq[i], &s, &c); }
    if (head < a.query_heads) {
        const float* x = row + head * d;
        const float x0 = x[i], x1 = x[i + half];
        const float inv = rsqrtf(block_sum(x0 * x0 + x1 * x1) / float(d) + a.eps);
        const float n0 = x0 * inv * a.q_norm[i], n1 = x1 * inv * a.q_norm[i + half];
        float* q = a.q_out + (std::int64_t(t) * a.query_heads + head) * d;
        q[i]        = n0 * c - n1 * s;
        q[i + half] = n1 * c + n0 * s;
        return;
    }
    const int kh = head - a.query_heads;
    const float* k = row + (a.query_heads + kh) * d;
    const float* v = row + (a.query_heads + a.kv_heads + kh) * d;
    const float k0 = k[i], k1 = k[i + half];
    const float kinv = rsqrtf(block_sum(k0 * k0 + k1 * k1) / float(d) + a.eps);
    const float n0 = k0 * kinv * a.k_norm[i], n1 = k1 * kinv * a.k_norm[i + half];
    const int sequence = a.sequences != nullptr ? a.sequences[t] : a.sequence;
    const int stored   = a.slots != nullptr ? a.slots[t] : a.positions[t];
    const int slot     = a.ring != 0 ? stored % a.ring : stored;
    const std::int64_t at =
        std::int64_t(sequence) * a.seq_stride + std::int64_t(kh) * a.head_stride +
        std::int64_t(slot) * d;
    a.k_cache[at + i]        = __float2half(n0 * c - n1 * s);
    a.k_cache[at + i + half] = __float2half(n1 * c + n0 * s);
    a.v_cache[at + i]        = __float2half(v[i]);
    a.v_cache[at + i + half] = __float2half(v[i + half]);
}

__global__ void build_mask_kernel(__half* mask, int tokens, int slots, int first, int ring,
                                  int window, const std::int32_t* __restrict__ starts, int shared) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (s >= slots) { return; }
    const int query = first + t;
    bool visible    = false;
    if (starts != nullptr && t >= shared) {
        // A branch row: the positions before the branches (the sequence's and the chunk's first
        // `shared` rows) as its own position sees them (the window ends at the same distance),
        // and its branch's rows, which lie at positions first + starts[t] onwards, up to itself.
        const int base = first + shared;
        const int own  = first + starts[t];
        const int last = first + tokens - 1;
        const int p    = ring == 0 ? s : s <= last ? s + ((last - s) / ring) * ring : -1;
        if (p >= 0 && p < base) {
            visible = ring == 0 || p > query - (own - base) - window;
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
        }
    }
    mask[std::int64_t(t) * slots + s] = visible ? __float2half(0.0f) : __float2half(-INFINITY);
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
// Width 128, eight query heads per KV head: block (row, kv head, split) of eight warps, warp h the
// group's query head h. A tile's 16 keys and values pass through shared memory once for all eight
// heads; lane l holds dimensions [4 l, 4 l + 4) (8-byte reads, conflict-free), and every warp keeps
// its own online softmax, so a tile costs two barriers.
constexpr int kDecodeThreads = 256;
constexpr int kTile          = 16;
constexpr int kD             = 128;

// The splits a row's keys actually use: at least a tile of keys each, so a short context does not
// pay for empty blocks and their partial rows. The attention and its combine agree on it.
__device__ __forceinline__ int decode_active_splits(int keys, int splits) {
    return min(splits, max(1, (keys + kTile - 1) / kTile));
}

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

__launch_bounds__(kDecodeThreads) __global__ void decode_attention_kernel(DecodeAttentionArgs a) {
    const int b        = blockIdx.x; // rows fastest: a verification's rows read the same keys
    const int kv_head  = blockIdx.y;
    const int split    = blockIdx.z;
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int position = a.positions[b];
    const int sequence = a.sequences[b];
    const int lo_all   = a.ring != 0 ? max(0, position - a.window + 1) : 0;
    const int n_all    = position + 1 - lo_all;
    const int active   = decode_active_splits(n_all, a.splits);
    if (split >= active) { return; }
    const int chunk = ((n_all + active - 1) / active + kTile - 1) / kTile * kTile;
    const int lo    = lo_all + split * chunk;
    const int hi    = min(position + 1, lo + chunk);

    __shared__ __align__(16) __half k_s[kTile][kD];
    __shared__ __align__(16) __half v_s[kTile][kD];

    const int head = kv_head * (kDecodeThreads / 32) + warp;
    float q_r[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    {
        const float4 q = *reinterpret_cast<const float4*>(
            a.q + (std::int64_t(b) * a.query_heads + head) * kD + 4 * lane);
        q_r[0] = q.x * a.scale;
        q_r[1] = q.y * a.scale;
        q_r[2] = q.z * a.scale;
        q_r[3] = q.w * a.scale;
    }
    float m = -INFINITY, l = 0.0f;

    const std::int64_t base = std::int64_t(sequence) * a.seq_stride +
                              std::int64_t(kv_head) * a.head_stride;
    // One 16-byte word of K and one of V per thread and tile: row w / 16, words w % 16.
    constexpr int kWords = kD / 8;
    static_assert(kTile * kWords == kDecodeThreads, "one word per thread and matrix");
    const int r = threadIdx.x / kWords, c = threadIdx.x % kWords;
    uint4 k_next = make_uint4(0, 0, 0, 0), v_next = make_uint4(0, 0, 0, 0);
    const auto fetch = [&](int tile, int count) {
        k_next = v_next = make_uint4(0, 0, 0, 0);
        if (r < count) {
            const int p    = tile + r;
            const int slot = a.ring != 0 ? p % a.ring : p;
            const std::int64_t at = base + std::int64_t(slot) * kD + c * 8;
            k_next = *reinterpret_cast<const uint4*>(a.k_cache + at);
            v_next = *reinterpret_cast<const uint4*>(a.v_cache + at);
        }
    };
    // After the score butterfly lanes 2j and 2j + 1 hold key `lane_key`.
    const int lane_key = ((lane & 16) ? 8 : 0) + ((lane & 8) ? 4 : 0) + ((lane & 4) ? 2 : 0) +
                         ((lane & 2) ? 1 : 0);
    if (lo < hi) { fetch(lo, min(kTile, hi - lo)); }
    for (int tile = lo; tile < hi; tile += kTile) {
        const int count = min(kTile, hi - tile);
        *reinterpret_cast<uint4*>(&k_s[r][c * 8]) = k_next;
        *reinterpret_cast<uint4*>(&v_s[r][c * 8]) = v_next;
        __syncthreads();
        if (tile + kTile < hi) { fetch(tile + kTile, min(kTile, hi - tile - kTile)); }
        float part[kTile];
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const uint2 w = *reinterpret_cast<const uint2*>(&k_s[kk][4 * lane]);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
            part[kk] = q_r[0] * f0.x + q_r[1] * f0.y + q_r[2] * f1.x + q_r[3] * f1.y;
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
        for (int j = 0; j < 4; ++j) { acc[j] *= scale; }
#pragma unroll
        for (int kk = 0; kk < kTile; ++kk) {
            const int src  = ((kk & 8) ? 16 : 0) | ((kk & 4) ? 8 : 0) | ((kk & 2) ? 4 : 0) |
                             ((kk & 1) ? 2 : 0);
            const float pk = __shfl_sync(0xFFFFFFFFu, p, src);
            const uint2 w = *reinterpret_cast<const uint2*>(&v_s[kk][4 * lane]);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&w.x));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&w.y));
            acc[0] += pk * f0.x;
            acc[1] += pk * f0.y;
            acc[2] += pk * f1.x;
            acc[3] += pk * f1.y;
        }
        __syncthreads();
    }
    // Partials: [B][Hq][splits] rows of D, then [B][Hq][splits] (m, l) pairs.
    const std::int64_t rows = std::int64_t(a.batch) * a.query_heads * a.splits;
    const std::int64_t row  = (std::int64_t(b) * a.query_heads + head) * a.splits + split;
    *reinterpret_cast<float4*>(a.partial + row * kD + 4 * lane) =
        make_float4(acc[0], acc[1], acc[2], acc[3]);
    if (lane == 0) {
        a.partial[rows * kD + 2 * row]     = m;
        a.partial[rows * kD + 2 * row + 1] = l;
    }
}

// Block (head, row): merges the splits of one query head and its sink, thread d the output's
// dimension d. Every split counts with weight exp(m_s - M), the sink as a key without a value:
// M = max(max_s m_s, sink), l = sum_s l_s exp(m_s - M) + exp(sink - M).
constexpr int kCombineThreads = kD; // at least the largest split count

__global__ void decode_combine_kernel(DecodeAttentionArgs a) {
    const int head = blockIdx.x, b = blockIdx.y;
    const int d    = threadIdx.x;
    const std::int64_t rows  = std::int64_t(a.batch) * a.query_heads * a.splits;
    const std::int64_t first = (std::int64_t(b) * a.query_heads + head) * a.splits;
    const int position = a.positions[b];
    const int lo_all   = a.ring != 0 ? max(0, position - a.window + 1) : 0;
    const int active   = decode_active_splits(position + 1 - lo_all, a.splits);
    __shared__ float weight[kCombineThreads];
    __shared__ float max_s, inv_l;
    const int s0   = threadIdx.x;
    const float ms = s0 < active ? a.partial[rows * kD + 2 * (first + s0)] : -INFINITY;
    float m        = ms;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o)); }
    if ((threadIdx.x & 31) == 0) { weight[threadIdx.x >> 5] = m; }
    __syncthreads();
    const float sink = a.sinks[head];
    if (threadIdx.x == 0) {
        float all = sink;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) { all = fmaxf(all, weight[w]); }
        max_s = all;
    }
    __syncthreads();
    const float w_s = ms != -INFINITY ? expf(ms - max_s) : 0.0f;
    const float l_s = w_s != 0.0f ? a.partial[rows * kD + 2 * (first + s0) + 1] * w_s : 0.0f;
    const float l   = block_sum(l_s);
    weight[threadIdx.x] = w_s;
    if (threadIdx.x == 0) { inv_l = 1.0f / (l + expf(sink - max_s)); }
    __syncthreads();
    float o = 0.0f;
    for (int s = 0; s < active; ++s) {
        const float w = weight[s];
        if (w != 0.0f) { o += a.partial[(first + s) * kD + d] * w; }
    }
    const float y = o * inv_l;
    if (a.out_q.qs == nullptr) {
        a.out[(std::int64_t(b) * a.query_heads + head) * kD + d] = __float2bfloat16(y);
        return;
    }
    // One warp holds 32 consecutive values: one q8_1 block of the o-projection input.
    const int width = a.query_heads * kD, at = head * kD + d;
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

// ---- Routing ------------------------------------------------------------------------------------

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
// eight warps, so every SM streams part of the router.
constexpr int kRouteSmallThreads = 256;

// Expert e's logits of the tokens by a kRouteSmallThreads-thread block (route_logits_small_kernel's
// and route_decode_kernel's blocks).
__device__ __forceinline__ void router_row(const float* __restrict__ x, int hidden, int tokens,
                                           const float* __restrict__ router, int experts, int e,
                                           float* __restrict__ logits) {
    __shared__ float part[kRouteSmallThreads / 32][kRouteTokens];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
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

__global__ void route_logits_small_kernel(const float* __restrict__ x, int hidden, int tokens,
                                          const float* __restrict__ router, int experts,
                                          float* __restrict__ logits) {
    router_row(x, hidden, tokens, router, experts, blockIdx.x, logits);
}

constexpr int kTopkMaxExperts = 256;

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// One block per token, one thread per expert: each expert's rank among the logits (larger first,
// lower id on ties) selects the top_k; their sigmoid scores renormalized over the top_k.
__global__ void route_topk_kernel(const float* __restrict__ logits, int experts, int top_k,
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
            chosen[rank]    = sigmoid(mine);
            chosen_id[rank] = e;
        }
    }
    __syncthreads();
    if (e < top_k) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += chosen[k]; }
        ids[std::int64_t(t) * top_k + e]     = chosen_id[e];
        weights[std::int64_t(t) * top_k + e] = chosen[e] / (sum + 1e-20f);
    }
}

// The rank of `mine`, expert e's logit, among values[0, kTopkMaxExperts) (larger first, lower id on
// equal values), as route_topk_kernel ranks it: the values past the experts are -inf, which never
// rank before an expert. Unrolled float4 reads keep a lone block's loads in flight.
__device__ __forceinline__ int rank_among(const float* values, float mine, int e) {
    const float4* v4 = reinterpret_cast<const float4*>(values);
    int rank         = 0;
#pragma unroll
    for (int j4 = 0; j4 < kTopkMaxExperts / 4; ++j4) {
        const float4 o = v4[j4];
        const int j    = 4 * j4;
        rank += (o.x > mine) | ((o.x == mine) & (j < e));
        rank += (o.y > mine) | ((o.y == mine) & (j + 1 < e));
        rank += (o.z > mine) | ((o.z == mine) & (j + 2 < e));
        rank += (o.w > mine) | ((o.w == mine) & (j + 3 < e));
    }
    return rank;
}

// Decode routing after the logits, by one block of kTopkMaxExperts threads: each token's top_k as
// route_topk_kernel picks them, then the pairs grouped by expert as moe_sort_routes lists them
// (bounds, pairs per expert, active experts ascending). Other blocks wrote the logits: they are
// read from L2.
__device__ void select_routes(const float* logits, int tokens, int experts, int top_k,
                              std::int32_t* __restrict__ ids, float* __restrict__ weights,
                              std::int32_t* __restrict__ bounds, std::int32_t* __restrict__ sorted,
                              std::int32_t* __restrict__ active,
                              std::int32_t* __restrict__ active_count) {
    __shared__ __align__(16) float v[kTopkMaxExperts];
    __shared__ float chosen[kRouteTokens * 8];
    __shared__ int chosen_id[kRouteTokens * 8];
    __shared__ int cursor[kTopkMaxExperts];
    __shared__ int warp_total[kTopkMaxExperts / 32][2];
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    for (int t = 0; t < tokens; ++t) {
        v[e] = e < experts ? __ldcg(logits + std::int64_t(t) * experts + e) : -INFINITY;
        __syncthreads();
        if (e < experts) {
            const float mine = v[e];
            const int rank   = rank_among(v, mine, e);
            if (rank < top_k) {
                chosen[t * top_k + rank]    = sigmoid(mine);
                chosen_id[t * top_k + rank] = e;
            }
        }
        __syncthreads();
    }
    const int pairs = tokens * top_k;
    if (e < pairs) {
        const int base = (e / top_k) * top_k;
        float sum      = 0.0f;
        for (int k = 0; k < top_k; ++k) { sum += chosen[base + k]; }
        ids[e]     = chosen_id[e];
        weights[e] = chosen[e] / (sum + 1e-20f);
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

static_assert(kRouteSmallThreads == kTopkMaxExperts, "a routing block computes a row, then selects");

// Decode routing in one launch: block e computes expert e's logits as route_logits_small_kernel
// does, and the last block to finish selects (select_routes). `done` counts the finished blocks;
// the last one sets it back to zero.
__global__ void route_decode_kernel(const float* __restrict__ x, int hidden, int tokens,
                                    const float* __restrict__ router, int experts, int top_k,
                                    float* __restrict__ logits, std::int32_t* __restrict__ ids,
                                    float* __restrict__ weights, std::int32_t* __restrict__ bounds,
                                    std::int32_t* __restrict__ sorted,
                                    std::int32_t* __restrict__ active,
                                    std::int32_t* __restrict__ active_count,
                                    unsigned* __restrict__ done) {
    __shared__ bool last;
    router_row(x, hidden, tokens, router, experts, blockIdx.x, logits);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { last = atomicAdd(done, 1u) == gridDim.x - 1; }
    __syncthreads();
    if (!last) { return; }
    __threadfence();
    select_routes(logits, tokens, experts, top_k, ids, weights, bounds, sorted, active,
                  active_count);
    if (threadIdx.x == 0) { *done = 0; }
}

constexpr int kGuardThreads = 128;

// One block per pair. A warp's step covers 32 aligned values, the span of one F16 sum.
__global__ void swiglu_rows_kernel(const float* __restrict__ x, int width,
                                   __nv_bfloat16* __restrict__ out, float* __restrict__ weights) {
    const std::int64_t p = blockIdx.x;
    const float* row     = x + p * 2 * width;
    __nv_bfloat16* o     = out + p * width;
    float largest        = 0.0f;
    for (int r = threadIdx.x; r < width; r += blockDim.x) {
        const __nv_bfloat16 v = __float2bfloat16(ops::portable_silu(row[r]) * row[width + r]);
        o[r]                  = v;
        if (weights != nullptr) { largest = fmaxf(largest, fabsf(warp_sum(__bfloat162float(v)))); }
    }
    if (weights == nullptr) { return; }
    const int k = f16_sum_exponent(block_max(largest));
    if (k == 0) { return; }
    const float down = ldexpf(1.0f, -k);
    for (int r = threadIdx.x; r < width; r += blockDim.x) {
        o[r] = __float2bfloat16(__bfloat162float(o[r]) * down);
    }
    if (threadIdx.x == 0) { weights[p] *= ldexpf(1.0f, k); }
}

// One block per column.
__global__ void scale_columns_kernel(__nv_bfloat16* __restrict__ x, int rows,
                                     float* __restrict__ h, int h_rows, float* __restrict__ scale) {
    const std::int64_t c   = blockIdx.x;
    __nv_bfloat16* column  = x + c * rows;
    float largest          = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        largest = fmaxf(largest, fabsf(warp_sum(__bfloat162float(column[r]))));
    }
    const int k = f16_sum_exponent(block_max(largest));
    if (threadIdx.x == 0) { scale[c] = ldexpf(1.0f, k); }
    if (k == 0) { return; }
    const float down = ldexpf(1.0f, -k);
    for (int r = threadIdx.x; r < rows; r += blockDim.x) {
        column[r] = __float2bfloat16(__bfloat162float(column[r]) * down);
    }
    float* hc = h + c * h_rows;
    for (int r = threadIdx.x; r < h_rows; r += blockDim.x) { hc[r] *= down; }
}

// One block of `width` threads per pair: each warp's 32 values are one F16 sum's.
__global__ void quantize_pairs_kernel(__nv_bfloat16* __restrict__ middle, int width,
                                      float* __restrict__ weights, Q8Rows out) {
    __shared__ float largest[32];
    __shared__ int exponent;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const std::int64_t at = std::int64_t(blockIdx.x) * width + threadIdx.x;
    float x               = __bfloat162float(middle[at]);
    float sum             = warp_sum(x);
    if (lane == 0) { largest[warp] = fabsf(sum); }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) { m = fmaxf(m, largest[w]); }
        exponent = f16_sum_exponent(m);
        if (exponent != 0) { weights[blockIdx.x] = ldexpf(weights[blockIdx.x], exponent); }
    }
    __syncthreads();
    if (exponent != 0) {
        const __nv_bfloat16 v = __float2bfloat16(ldexpf(x, -exponent));
        middle[at]            = v;
        x                     = __bfloat162float(v);
        sum                   = warp_sum(x);
    }
    // quantize_vector_kernel's numbers: its warp's maximum and butterfly sum.
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o)); }
    const float d = amax / 127.0f;
    out.qs[at]    = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(x / d));
    if (lane == 0) { out.ds[at / 32] = __floats2half2_rn(d, sum); }
}

__global__ void unscale_columns_kernel(float* __restrict__ h, int h_rows,
                                       const float* __restrict__ scale) {
    const float s = scale[blockIdx.x];
    if (s == 1.0f) { return; }
    float* hc = h + std::int64_t(blockIdx.x) * h_rows;
    for (int r = threadIdx.x; r < h_rows; r += blockDim.x) { hc[r] *= s; }
}

__global__ void add_fixed_kernel(unsigned long long* sums, const unsigned long long* products,
                                 const std::int32_t* flag, std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || *flag == 0) { return; }
    sums[i] += products[i];
}

__global__ void ffn_combine_kernel(FfnCombineArgs a) {
    const int q = a.hidden >> 2, i = threadIdx.x;
    const std::int64_t row = std::int64_t(blockIdx.x) * a.hidden;
    const float4 zero      = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 h               = zero;
    if (i < q) {
        h = ld4(a.h + row, i);
        if (a.moe_rows != nullptr) {
            for (int k = 0; k < a.top_k; ++k) {
                const std::int64_t pair = std::int64_t(blockIdx.x) * a.top_k + k;
                h = add4(h, scale4(ld4(a.moe_rows + pair * a.hidden, i), a.moe_weights[pair]));
            }
        } else if (a.moe_fixed != nullptr) {
            h = add4(h, fixed4(a.moe_fixed + row, i));
        }
        reinterpret_cast<float4*>(a.h + row)[i] = h;
    }
    const float inv  = rsqrtf(block_sum(dot4(h, h)) / float(a.hidden) + a.eps);
    const float4 nxt = i < q ? mul4(scale4(h, inv), ld4(a.next_norm, i)) : zero;
    if (i < q && a.next_out != nullptr) { store_bf16x4(a.next_out + row + 4 * i, nxt); }
    if (a.next_q.qs != nullptr) { store_q8x4(a.next_q, a.hidden, blockIdx.x, i, nxt, i < q); }
}

} // namespace

void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q, float* f32) {
    if (tokens <= 0) { return; }
    rms_norm_kernel<<<tokens, row_threads(hidden), 0, stream>>>(x, w, hidden, eps, out, q, f32);
    check_launch("rms_norm");
}

void qkv_post(const QkvArgs& args, cudaStream_t stream) {
    if (args.head_dim != kD) { throw std::invalid_argument("flint qkv_post: head width must be 128"); }
    const dim3 grid(args.tokens, args.query_heads + args.kv_heads, 1);
    qkv_post_kernel<<<grid, args.head_dim / 2, 0, stream>>>(args);
    check_launch("qkv_post");
}

void build_mask(__half* mask, int tokens, int slots, int first, int ring, int window,
                cudaStream_t stream, const std::int32_t* starts, int shared) {
    const dim3 grid(blocks_for(slots, 256), tokens, 1);
    build_mask_kernel<<<grid, 256, 0, stream>>>(mask, tokens, slots, first, ring, window, starts,
                                                shared);
    check_launch("build_mask");
}

void ring_copy(const RingCopyArgs& a, cudaStream_t stream) {
    if (a.end <= a.begin) { return; }
    if (a.head_dim % 8 != 0) { throw std::invalid_argument("flint ring_copy: head width"); }
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

void decode_attention(const DecodeAttentionArgs& args, cudaStream_t stream) {
    if (args.head_dim != kD || args.query_heads != 8 * args.kv_heads || args.sinks == nullptr ||
        args.splits < 1 || args.splits > kCombineThreads) {
        throw std::invalid_argument("flint decode attention: width 128, eight query heads per KV "
                                    "head, sinks and 1..128 splits");
    }
    decode_attention_kernel<<<dim3(args.batch, args.kv_heads, args.splits), kDecodeThreads, 0,
                              stream>>>(args);
    check_launch("decode_attention");
    decode_combine_kernel<<<dim3(args.query_heads, args.batch, 1), kCombineThreads, 0, stream>>>(
        args);
    check_launch("decode_combine");
}

void route(const float* x, int hidden, int tokens, const float* router, int experts, int top_k,
           float* logits, std::int32_t* ids, float* weights, cudaStream_t stream) {
    if (top_k > 8 || top_k > experts || experts > kTopkMaxExperts || hidden % 4 != 0) {
        throw std::invalid_argument("flint route: at most 8 of 256 experts over whole float4s");
    }
    if (tokens <= kRouteTokens) {
        route_logits_small_kernel<<<experts, kRouteSmallThreads, 0, stream>>>(x, hidden, tokens,
                                                                             router, experts, logits);
    } else {
        const dim3 grid(blocks_for(experts, kRouteTileE), blocks_for(tokens, kRouteTileT), 1);
        route_logits_kernel<<<grid, 256, 0, stream>>>(x, hidden, tokens, router, experts, logits);
    }
    check_launch("route logits");
    route_topk_kernel<<<tokens, (experts + 31) / 32 * 32, 0, stream>>>(logits, experts, top_k, ids,
                                                                       weights);
    check_launch("route top-k");
}

void route_select(const float* x, int hidden, int tokens, const float* router, int experts,
                  int top_k, float* logits, std::int32_t* ids, float* weights, std::int32_t* bounds,
                  std::int32_t* sorted, std::int32_t* active, std::int32_t* active_count,
                  std::uint32_t* done, cudaStream_t stream) {
    if (tokens > kRouteTokens || top_k > 8 || top_k > experts || experts > kTopkMaxExperts ||
        hidden % 4 != 0) {
        throw std::invalid_argument("flint route_select: at most 8 tokens, 8 of 256 experts");
    }
    route_decode_kernel<<<experts, kTopkMaxExperts, 0, stream>>>(
        x, hidden, tokens, router, experts, top_k, logits, ids, weights, bounds, sorted, active,
        active_count, done);
    check_launch("route_select");
}

void swiglu_rows(const float* x, int pairs, int width, __nv_bfloat16* out, float* weights,
                 cudaStream_t stream) {
    if (width % 32 != 0) { throw std::invalid_argument("flint swiglu_rows: width % 32 != 0"); }
    if (pairs <= 0 || width <= 0) { return; }
    swiglu_rows_kernel<<<pairs, kGuardThreads, 0, stream>>>(x, width, out, weights);
    check_launch("swiglu_rows");
}

void scale_columns(__nv_bfloat16* x, int rows, int columns, float* h, int h_rows, float* scale,
                   cudaStream_t stream) {
    if (rows % 32 != 0) { throw std::invalid_argument("flint scale_columns: rows % 32 != 0"); }
    if (columns <= 0) { return; }
    scale_columns_kernel<<<columns, kGuardThreads, 0, stream>>>(x, rows, h, h_rows, scale);
    check_launch("scale_columns");
}

void quantize_pairs(__nv_bfloat16* middle, int pairs, int width, float* weights, Q8Rows out,
                    cudaStream_t stream) {
    if (width % 32 != 0 || width > 1024) {
        throw std::invalid_argument("flint quantize_pairs: whole 32-value groups, at most 1024");
    }
    if (pairs <= 0) { return; }
    quantize_pairs_kernel<<<pairs, width, 0, stream>>>(middle, width, weights, out);
    check_launch("quantize_pairs");
}

void unscale_columns(float* h, int h_rows, int columns, const float* scale, cudaStream_t stream) {
    if (columns <= 0) { return; }
    unscale_columns_kernel<<<columns, kGuardThreads, 0, stream>>>(h, h_rows, scale);
    check_launch("unscale_columns");
}

void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream) {
    if (n <= 0) { return; }
    add_fixed_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        sums, static_cast<const unsigned long long*>(products), flag, n);
    check_launch("add_fixed");
}

void ffn_combine(const FfnCombineArgs& args, cudaStream_t stream) {
    if (args.next_norm == nullptr) { throw std::invalid_argument("flint ffn_combine: next norm"); }
    ffn_combine_kernel<<<args.tokens, row_threads(args.hidden), 0, stream>>>(args);
    check_launch("ffn_combine");
}

} // namespace ninfer::models::flint::kernels
