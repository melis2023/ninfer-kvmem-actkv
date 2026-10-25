#include "ops/softmax_attention/dense/causal_cache/prompt_sm70.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t_sm70.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim     = 256;
constexpr int kRows        = 64; // query rows per CTA
constexpr int kWarps       = 8;  // eight rows each
constexpr int kTile        = 32; // keys per tile
constexpr int kThreads     = kWarps * 32;
constexpr int kStride      = sm70_attention::kQStride; // halves per staged Q, K or V row
constexpr int kPStride     = sm70_attention::kPStride;
constexpr int kSharedBytes = (kRows * kStride + 2 * kTile * kStride + kWarps * 8 * kPStride) *
                             static_cast<int>(sizeof(__half));
static_assert(kRows * kHeadDim * sizeof(float) <= (kRows + 2 * kTile) * kStride * sizeof(__half),
              "the rotated epilogue's FP32 rows reuse the Q, K and V stages");

// How a cache stores its K or V plane. Every plane is [page][head][page offset][row], so a token's
// row (page * kv_heads + head) * page size + offset addresses each of them.
enum class Coding : int { Bf16, Int8, Lloyd4, Int4E8, RootE8, Int4, Fp8, Nvfp4 };

template <Coding C>
__host__ __device__ constexpr int row_bytes() {
    if constexpr (C == Coding::Bf16) {
        return kHeadDim * 2;
    } else if constexpr (C == Coding::Int8 || C == Coding::Fp8) {
        return kHeadDim;
    } else if constexpr (C == Coding::RootE8) {
        return kKVCacheE8RootRowBytes;
    } else {
        return kHeadDim / 2;
    }
}

// One loader thread's 32 dimensions of one key row as stored: up to 64 bytes and the scale bits
// that cover them (an FP16 scale, or two E4M3 bytes for NVFP4).
struct Raw {
    uint4 w[4];
    unsigned scale;
};

// Dimensions [32 part, 32 part + 32) of plane row `row`.
template <Coding C>
__device__ __forceinline__ Raw read_raw(const void* plane, const void* scales, std::int64_t row,
                                        int part) {
    Raw r{};
    const auto* base = static_cast<const std::uint8_t*>(plane) + row * row_bytes<C>();
    if constexpr (C == Coding::Bf16) {
        const auto* p = reinterpret_cast<const uint4*>(base + part * 64);
#pragma unroll
        for (int i = 0; i < 4; ++i) { r.w[i] = p[i]; }
    } else if constexpr (C == Coding::Int8 || C == Coding::Fp8) {
        const auto* p = reinterpret_cast<const uint4*>(base + part * 32);
        r.w[0]        = p[0];
        r.w[1]        = p[1];
    } else if constexpr (C == Coding::RootE8) {
        const uint2 v = *reinterpret_cast<const uint2*>(base + part * 8);
        r.w[0]        = make_uint4(v.x, v.y, 0u, 0u);
    } else {
        r.w[0] = *reinterpret_cast<const uint4*>(base + part * 16);
    }
    const auto* half_scales = static_cast<const __half*>(scales);
    if constexpr (C == Coding::Int8 || C == Coding::Lloyd4 || C == Coding::Int4E8 ||
                  C == Coding::RootE8) {
        r.scale = __half_as_ushort(half_scales[row * kKVCacheInt8Groups + part / 2]);
    } else if constexpr (C == Coding::Int4) {
        r.scale = __half_as_ushort(half_scales[row * kKVCacheInt4ValueGroups + part]);
    } else if constexpr (C == Coding::Fp8) {
        r.scale = __half_as_ushort(half_scales[row]);
    } else if constexpr (C == Coding::Nvfp4) {
        r.scale = *reinterpret_cast<const std::uint16_t*>(static_cast<const std::uint8_t*>(scales) +
                                                          row * kKVCacheNvfp4Groups + 2 * part);
    }
    return r;
}

__device__ __forceinline__ unsigned clamped_f16x2(unsigned bf16x2) {
    const float lo  = fminf(fmaxf(__uint_as_float(bf16x2 << 16), -65504.0f), 65504.0f);
    const float hi  = fminf(fmaxf(__uint_as_float(bf16x2 & 0xffff0000u), -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ __half2 fp16_scale(unsigned bits) {
    return __half2half2(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// Eight words of four signed int8 codes times `scale` as sixteen FP16 pairs.
__device__ __forceinline__ void codes_to_f16(const unsigned (&codes)[8], __half2 scale,
                                             unsigned (&out)[16]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm70_attention::int8x4_to_f16x2(codes[i], scale, out[2 * i], out[2 * i + 1]);
    }
}

// The 32 dimensions of a Raw in FP16 pairs, dimension order.
template <Coding C>
__device__ __forceinline__ void decode_raw(const Raw& r, unsigned (&out)[16]) {
    if constexpr (C == Coding::Bf16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[4 * i + 0] = clamped_f16x2(r.w[i].x);
            out[4 * i + 1] = clamped_f16x2(r.w[i].y);
            out[4 * i + 2] = clamped_f16x2(r.w[i].z);
            out[4 * i + 3] = clamped_f16x2(r.w[i].w);
        }
    } else if constexpr (C == Coding::Int8) {
        const unsigned codes[8] = {r.w[0].x, r.w[0].y, r.w[0].z, r.w[0].w,
                                   r.w[1].x, r.w[1].y, r.w[1].z, r.w[1].w};
        codes_to_f16(codes, fp16_scale(r.scale), out);
    } else if constexpr (C == Coding::Lloyd4 || C == Coding::Int4E8 || C == Coding::Int4) {
        const uint2 first       = make_uint2(r.w[0].x, r.w[0].y);
        const uint2 second      = make_uint2(r.w[0].z, r.w[0].w);
        const int4 lo           = C == Coding::Lloyd4 ? kv_cache_lloyd4_expand16(first)
                                                      : kv_cache_int4_unpack_i8x16(first);
        const int4 hi           = C == Coding::Lloyd4 ? kv_cache_lloyd4_expand16(second)
                                                      : kv_cache_int4_unpack_i8x16(second);
        const unsigned codes[8] = {static_cast<unsigned>(lo.x), static_cast<unsigned>(lo.y),
                                   static_cast<unsigned>(lo.z), static_cast<unsigned>(lo.w),
                                   static_cast<unsigned>(hi.x), static_cast<unsigned>(hi.y),
                                   static_cast<unsigned>(hi.z), static_cast<unsigned>(hi.w)};
        codes_to_f16(codes, fp16_scale(r.scale), out);
    } else if constexpr (C == Coding::RootE8) {
        unsigned codes[8];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const unsigned word  = b < 2 ? r.w[0].x : r.w[0].y;
            const unsigned block = word >> (16 * (b & 1));
            const uint2 decoded =
                kv_cache_e8_root_decode_block(block & 0xffu, (block >> 8) & 0xffu);
            codes[2 * b]     = decoded.x;
            codes[2 * b + 1] = decoded.y;
        }
        codes_to_f16(codes, fp16_scale(r.scale), out);
    } else if constexpr (C == Coding::Fp8) {
        // The bit-level decode is the code times 2^-8; the scale carries the 2^8.
        const __half2 scale     = __hmul2(fp16_scale(r.scale), __float2half2_rn(256.0f));
        const unsigned words[8] = {r.w[0].x, r.w[0].y, r.w[0].z, r.w[0].w,
                                   r.w[1].x, r.w[1].y, r.w[1].z, r.w[1].w};
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const __half2 v =
                __hmul2(kv_cache_fp8_code2_to_half2_x2m8(words[i >> 1] >> (16 * (i & 1))), scale);
            out[i] = *reinterpret_cast<const unsigned*>(&v);
        }
    } else {
        // NVFP4: E2M1 pairs over one E4M3 scale per 16 dimensions, exact in FP16 (every E4M3 scale
        // is an FP16, and the bit-level decode times 2^8 restores it).
        const unsigned words[4] = {r.w[0].x, r.w[0].y, r.w[0].z, r.w[0].w};
        const __half2 two8      = __float2half2_rn(256.0f);
        const unsigned s0       = r.scale & 0xffu;
        const unsigned s1       = (r.scale >> 8) & 0xffu;
        const __half2 scale0    = __hmul2(kv_cache_fp8_code2_to_half2_x2m8(s0 | (s0 << 8)), two8);
        const __half2 scale1    = __hmul2(kv_cache_fp8_code2_to_half2_x2m8(s1 | (s1 << 8)), two8);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const __half2 v = __hmul2(kv_cache_e2m1x2_to_half2(words[i >> 2] >> (8 * (i & 3))),
                                      i < 8 ? scale0 : scale1);
            out[i]          = *reinterpret_cast<const unsigned*>(&v);
        }
    }
}

// K and V of the CTA's current key tile [k0, k0 + 32), in shared memory, as attend_tile reads
// them.
struct SharedRows {
    const __half* k_s;
    const __half* v_s;
    int k0;

    __device__ __forceinline__ const __half* key(int k) const { return k_s + (k - k0) * kStride; }

    __device__ __forceinline__ uint4 key8(const __half* row, int d) const {
        return *reinterpret_cast<const uint4*>(row + d);
    }

    __device__ __forceinline__ const __half* value(int k, int dim0) const {
        return v_s + (k - k0) * kStride + dim0;
    }

    __device__ __forceinline__ uint2 value4(const __half* row, int n) const {
        return *reinterpret_cast<const uint2*>(row + 32 * n);
    }
};

__device__ __forceinline__ void store_row(__half* dst, const unsigned (&words)[16]) {
    auto* p = reinterpret_cast<uint4*>(dst);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        p[i] = make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
    }
}

// The rows that carry a query: all of them, or the first valid_columns[0] of a masked row.
__device__ __forceinline__ std::int32_t valid_tokens(const std::int32_t* valid_columns,
                                                     std::int32_t tokens) {
    return valid_columns == nullptr ? tokens : max(0, min(tokens, valid_columns[0]));
}

// Packed row r of a KV head is token r / G and query head kv_head * G + r % G (G = the group's
// query heads). Rows of the tokens past the valid ones are written as zeros.
template <class Metadata, Coding K, Coding V, bool RotatedKeys, bool RotatedValues>
__launch_bounds__(kThreads) __global__
    void prompt_kernel(const __nv_bfloat16* __restrict__ q,
                       const std::int32_t* __restrict__ positions,
                       const std::int32_t* __restrict__ valid_columns, Metadata metadata,
                       const void* __restrict__ cache_k, const void* __restrict__ cache_v,
                       const void* __restrict__ scale_k, const void* __restrict__ scale_v,
                       std::int32_t tokens, std::int32_t q_heads, std::int32_t kv_heads,
                       float scale, __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    __half* q_s = reinterpret_cast<__half*>(shared_bytes);
    __half* k_s = q_s + kRows * kStride;
    __half* v_s = k_s + kTile * kStride;
    __half* p_s = v_s + kTile * kStride;

    const int group   = q_heads / kv_heads;
    const int kv_head = static_cast<int>(blockIdx.y);
    // Later blocks see more keys; they start first.
    const int block     = static_cast<int>(gridDim.x - 1 - blockIdx.x);
    const int row0      = block * kRows;
    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int valid     = valid_tokens(valid_columns, tokens);
    const int row_count = valid * group;
    const auto q_index  = [&](int row, int d) {
        return (static_cast<std::int64_t>(row / group) * q_heads + kv_head * group + row % group) *
                   kHeadDim +
               d;
    };

    // The CTA's query rows in FP16 (rotated like the keys), zero past the valid ones.
    for (int r = warp; r < kRows; r += kWarps) {
        const int row   = row0 + r;
        const bool live = row < row_count;
        float values[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            values[i] = live ? __bfloat162float(q[q_index(row, lane + 32 * i)]) : 0.0f;
        }
        if constexpr (RotatedKeys) { normalized_hadamard_d256_inplace(values, lane); }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q_s[r * kStride + lane + 32 * i] =
                __float2half_rn(fminf(fmaxf(values[i], -65504.0f), 65504.0f));
        }
    }
    // The keys the block's rows see: up to the largest position among its valid tokens.
    int window = 0;
    for (int t = row0 / group; t <= (min(row0 + kRows, row_count) - 1) / group && t < valid; ++t) {
        window = max(window, positions[t] + 1);
    }
    const int lane_row_a = (lane & 1) + 4 * (lane >> 4);
    const int row_a      = row0 + warp * 8 + lane_row_a;
    const int row_b      = row_a + 2;
    const int qabs_a     = row_a < row_count ? positions[row_a / group] : -1;
    const int qabs_b     = row_b < row_count ? positions[row_b / group] : -1;
    int warp_position    = max(qabs_a, qabs_b);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_position = max(warp_position, __shfl_xor_sync(0xffffffffu, warp_position, mask));
    }
    const __half* q_row = q_s + (warp * 8 + (lane & 3) + 4 * (lane >> 4)) * kStride;
    __half* p_tile      = p_s + warp * 8 * kPStride;
    sm70_attention::RowState st;

    // Loader thread: key k0 + tid / 8, dimensions [32 (tid % 8), +32).
    const int load_key        = tid >> 3;
    const int part            = tid & 7;
    const std::int32_t* table = metadata.block_table();
    const auto read           = [&](int k0, Raw& key_raw, Raw& value_raw) {
        const int key = k0 + load_key;
        if (key < window) {
            const std::int64_t page = table[key >> kPagedKVPageShift];
            const std::int64_t row =
                (page * kv_heads + kv_head) * kPagedKVPageSize + (key & kPagedKVPageMask);
            key_raw   = read_raw<K>(cache_k, scale_k, row, part);
            value_raw = read_raw<V>(cache_v, scale_v, row, part);
        } else {
            key_raw   = Raw{};
            value_raw = Raw{};
        }
    };
    const int tiles = (window + kTile - 1) / kTile;
    Raw key_raw{};
    Raw value_raw{};
    if (tiles > 0) { read(0, key_raw, value_raw); }
    __syncthreads();
    for (int tile = 0; tile < tiles; ++tile) {
        const int k0 = tile * kTile;
        unsigned words[16];
        decode_raw<K>(key_raw, words);
        store_row(k_s + load_key * kStride + 32 * part, words);
        decode_raw<V>(value_raw, words);
        store_row(v_s + load_key * kStride + 32 * part, words);
        __syncthreads();
        if (tile + 1 < tiles) { read(k0 + kTile, key_raw, value_raw); }
        if (warp_position >= k0) {
            sm70_attention::attend_tile(SharedRows{k_s, v_s, k0}, q_row, p_tile, k0, 0, window,
                                        qabs_a, qabs_b, scale, st);
        }
        __syncthreads();
    }

    // Element i of product n: row (lane & 1) + (i & 2) + 4h, dimension 32n + 8qp + (i & 4) +
    // (lane & 2) + (i & 1).
    const float inv_a = st.l_a > 0.0f ? 1.0f / st.l_a : 0.0f;
    const float inv_b = st.l_b > 0.0f ? 1.0f / st.l_b : 0.0f;
    const int qp      = (lane >> 2) & 3;
    const int rows    = tokens * group;
    if constexpr (RotatedValues) {
        // The values were stored rotated: each row goes back through the (self-inverse)
        // transform, from FP32 rows in the stages the tiles no longer need.
        float* rows_s = reinterpret_cast<float*>(shared_bytes) + warp * 8 * kHeadDim;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int local              = lane_row_a + (i & 2);
                const int d                  = 32 * n + 8 * qp + (i & 4) + (lane & 2) + (i & 1);
                rows_s[local * kHeadDim + d] = st.acc[n][i] * ((i & 2) ? inv_b : inv_a);
            }
        }
        __syncwarp();
        for (int local = 0; local < 8; ++local) {
            const int row = row0 + warp * 8 + local;
            if (row >= rows) { break; }
            float values[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) { values[i] = rows_s[local * kHeadDim + lane + 32 * i]; }
            normalized_hadamard_d256_inplace(values, lane);
            const bool live = row < row_count;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[q_index(row, lane + 32 * i)] = __float2bfloat16_rn(live ? values[i] : 0.0f);
            }
        }
    } else {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (i & 2) ? row_b : row_a;
                if (row >= rows) { continue; }
                const float inv = row < row_count ? ((i & 2) ? inv_b : inv_a) : 0.0f;
                const int d     = 32 * n + 8 * qp + (i & 4) + (lane & 2);
                *reinterpret_cast<__nv_bfloat162*>(out + q_index(row, d)) =
                    __floats2bfloat162_rn(st.acc[n][i] * inv, st.acc[n][i + 1] * inv);
            }
        }
    }
}

// The plane codings of each storage, and which planes it stores rotated (every quantized key
// coding; NVFP4 values).
template <class F>
void with_codings(KvCacheStorage storage, F&& f) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return f.template operator()<Coding::Bf16, Coding::Bf16, false, false>();
    case KvCacheStorage::Int8Group64:
        return f.template operator()<Coding::Int8, Coding::Int8, true, false>();
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return f.template operator()<Coding::Int8, Coding::Int4, true, false>();
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return f.template operator()<Coding::Lloyd4, Coding::Int4, true, false>();
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return f.template operator()<Coding::Int4E8, Coding::Int4, true, false>();
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return f.template operator()<Coding::RootE8, Coding::Int4, true, false>();
    case KvCacheStorage::Fp8E4M3Row256:
        return f.template operator()<Coding::Fp8, Coding::Fp8, true, false>();
    case KvCacheStorage::Nvfp4Group16:
        return f.template operator()<Coding::Nvfp4, Coding::Nvfp4, true, true>();
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return f.template operator()<Coding::Fp8, Coding::Nvfp4, true, true>();
    }
    throw std::invalid_argument("causal_attention_prompt_sm70: unknown KV storage");
}

template <class View, class Metadata>
void launch(const Tensor& q, const Tensor& positions, const std::int32_t* valid_columns,
            float scale, const View& cache, Metadata metadata, Tensor& out, cudaStream_t stream) {
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = cache.num_kv_heads;
    if (cache.head_dim != kHeadDim || q.ne[0] != kHeadDim || kv_heads <= 0 ||
        q_heads % kv_heads != 0 || tokens <= 0) {
        throw std::invalid_argument("causal_attention_prompt_sm70: unsupported geometry");
    }
    const std::int32_t rows = tokens * (q_heads / kv_heads);
    const dim3 grid(static_cast<unsigned>((rows + kRows - 1) / kRows),
                    static_cast<unsigned>(kv_heads));
    with_codings(cache.storage, [&]<Coding K, Coding V, bool RotatedKeys, bool RotatedValues>() {
        auto kernel = prompt_kernel<Metadata, K, V, RotatedKeys, RotatedValues>;
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kSharedBytes);
        });
        kernel<<<grid, kThreads, kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data), valid_columns, metadata,
            cache.k_pages.data, cache.v_pages.data, cache.k_scale_pages.data,
            cache.v_scale_pages.data, tokens, q_heads, kv_heads, scale,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace

bool causal_attention_prompt_sm70_applies(KvCacheStorage storage, std::int32_t batch) {
#if defined(NINFER_SM70)
    (void)storage; // every KV storage has a coding here
    return batch == 1;
#else
    (void)storage;
    (void)batch;
    return false;
#endif
}

void causal_attention_prompt_sm70_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache, Tensor& out,
                                         cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    launch(q, positions, nullptr, scale, cache, metadata, out, stream);
}

void causal_attention_prompt_sm70_launch(const Tensor& q, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& table_rows,
                                         float scale, const PagedKVBatchLayerView& cache,
                                         Tensor& out, cudaStream_t stream) {
    const PagedKVBatchMetadata<false> metadata{
        .tables        = static_cast<const std::int32_t*>(cache.block_tables.data),
        .valid_columns = nullptr,
        .table_rows    = static_cast<const std::int32_t*>(table_rows.data),
        .table_stride  = cache.block_tables.ne[0],
    };
    launch(q, positions, static_cast<const std::int32_t*>(valid_columns.data), scale, cache,
           metadata, out, stream);
}

} // namespace ninfer::ops::detail
