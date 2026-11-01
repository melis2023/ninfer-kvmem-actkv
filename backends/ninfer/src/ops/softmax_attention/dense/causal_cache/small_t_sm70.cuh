#pragma once

// ninfer::ops - split-KV causal small-T attention over a BF16 KV cache on sm_70 (Tesla V100), the
// partial kernel of a 70 build. The Ampere kernel (small_t_bf16.cuh) runs there on emulated
// ldmatrix and m16n8k16 tiles; this one uses Volta's mma.m8n8k4 on FP16 operands. It keeps that
// kernel's grid, split ranges, cache writes and partial outputs, so the split reducer is shared.
//
// A warp owns eight query rows (a row group: G query heads times T tokens, eight at a time) and
// walks its split's keys in tiles of 32: S = Q K^T as m8n8k4.row.col products (the four quad pairs
// take eight keys each, Q's fragment mirrored), the online softmax in FP32, P through shared memory
// into m8n8k4.row.row products with V (the quad pairs take eight of each 32 head dimensions). Each
// lane reads its key's K row and the V rows of its products straight from the pages, converting
// BF16 to FP16 (saturated at +-65504) as it loads them. One row group splits its tiles over up to
// four warps, combined in shared memory before the partial is written. A tile whose keys (or the
// CTA's queries) hold a BF16 value beyond FP16 forms its scores again in FP32 from the BF16 values
// (exact_scores), and one whose values do adds what their saturation took from P V
// (add_value_excess), so the products keep BF16's range.
//
// The INT8-family caches (causal_attention_small_t_sm70_i8_kernel: int8, rk8v4, rk4v4, rk4v4-e8,
// rk2v4-e8) take the same products: the fused append rotates and quantizes the new rows exactly as
// small_t_i8.cuh does, Q is rotated by the same normalized D256 Hadamard transform and kept in
// FP16, and the key codings and value codes are decoded to FP16 with their group scales as they
// load.

#include "core/device.h"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops {

namespace sm70_attention {

template <bool RowB>
__device__ __forceinline__ void hmma(float (&c)[8], unsigned a0, unsigned a1, unsigned b0,
                                     unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    if constexpr (RowB) {
        asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                       "+f"(c[6]), "+f"(c[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                       "+f"(c[6]), "+f"(c[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    }
#else
    (void)c;
    (void)a0;
    (void)a1;
    (void)b0;
    (void)b1;
#endif
}

__device__ __forceinline__ unsigned to_f16x2(unsigned bf16x2) {
    return detail::sm70::bf16x2_to_f16x2(bf16x2);
}

// Four signed int8 codes as two FP16 pairs (code 0 in the low lane of the first), times `scale`.
__device__ __forceinline__ void int8x4_to_f16x2(unsigned codes, __half2 scale, unsigned& lo,
                                                unsigned& hi) {
    const unsigned offset = codes ^ 0x80808080u; // 128 + code per byte
    const __half2 bias    = __float2half2_rn(1152.0f);
    const unsigned m0     = __byte_perm(offset, 0x64646464u, 0x4140u);
    const unsigned m1     = __byte_perm(offset, 0x64646464u, 0x4342u);
    const __half2 v0      = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&m0), bias), scale);
    const __half2 v1      = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&m1), bias), scale);
    lo                    = *reinterpret_cast<const unsigned*>(&v0);
    hi                    = *reinterpret_cast<const unsigned*>(&v1);
}

template <int Rows>
struct Shape {
    static constexpr int kRowGroups = (Rows + 7) / 8;
    // Warps splitting one row group's key tiles; more row groups share the keys' loads instead.
    static constexpr int kKeyWarps = kRowGroups == 1 ? 4 : (kRowGroups == 2 ? 2 : 1);
    static constexpr int kWarps    = kRowGroups * kKeyWarps;
};

inline constexpr int kQStride = kCausalHeadDim + 8; // halves; pads rows apart by 16 bytes
inline constexpr int kPStride = 32 + 8;             // halves

// The shared memory of causal_attention_small_t_sm70_i8_kernel for TokenTile tokens (its appending
// form, the larger), and the CTAs of it an SM holds: the kernel is bounded to that many, which keeps
// it within 128 registers a thread at four warps, and sm_70 has 96 KiB of shared memory an SM. The
// host sizes its whole-wave split counts from the same number (i8_partial_ctas_per_sm).
template <int TokenTile, int Rows>
constexpr int i8_shared_bytes() {
    using S = Shape<Rows>;
    return S::kRowGroups * 8 * kQStride * 2 + S::kWarps * 8 * kPStride * 2 + 64 * 4 +
           2 * S::kWarps * 8 * 4 + (S::kKeyWarps > 1 ? S::kRowGroups * 8 * kCausalHeadDim * 4 : 4) +
           TokenTile * kCausalHeadDim * 4;
}

template <int TokenTile, int Rows>
constexpr int i8_resident_ctas() {
    const int by_registers = 16 / Shape<Rows>::kWarps;
    const int by_shared    = 96 * 1024 / i8_shared_bytes<TokenTile, Rows>();
    const int ctas         = by_registers < by_shared ? by_registers : by_shared;
    return ctas > 0 ? ctas : 1;
}

// Flags BF16 values beyond FP16 in `bits` (detail::sm70::bf16x2_beyond_f16, accumulated): a
// value is beyond where bits & kBeyondBits is nonzero.
inline constexpr unsigned kBeyondBits = 0x80008000u;
__device__ __forceinline__ unsigned beyond_bits(unsigned bf16x2) {
    return (bf16x2 & 0x7f807f80u) + 0x38803880u;
}

// K and V of a BF16 cache: the current step's rows from the append input, the rest from the pages.
// kExact: the rows also give the BF16 values of the CTA's queries (query) and keys and values
// (key_bf16, value_bf16), and key8 and value4 flag what they saturate.
template <typename Geometry, typename CacheInput>
struct Bf16Rows {
    static constexpr bool kExact = true;
    const __nv_bfloat16* cache_k;
    const __nv_bfloat16* cache_v;
    CacheInput input;
    const std::int32_t* pages;
    const std::int32_t* block_table;
    int first_page;
    bool staged_pages;
    int kv_head;
    int first_pos;
    int valid_tokens;
    const __nv_bfloat16* q; // the call's queries from the CTA's column on
    int tokens;
    int row_count;
    bool query_beyond; // some query of the CTA is beyond FP16

    __device__ __forceinline__ const __nv_bfloat16* query(int row) const {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        return row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head)
                   ? q + causal_q_index<Geometry>(q_head, 0, token)
                   : nullptr;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_bf16(int k) const { return key(k); }
    __device__ __forceinline__ const __nv_bfloat16* value_bf16(int k) const {
        return value(k, 0);
    }
    __device__ __forceinline__ bool keys_beyond(unsigned bits) const {
        return __any_sync(0xffffffffu, (bits & kBeyondBits) != 0u);
    }
    __device__ __forceinline__ bool values_beyond(unsigned bits) const {
        return __any_sync(0xffffffffu, (bits & kBeyondBits) != 0u);
    }

    __device__ __forceinline__ const __nv_bfloat16* address(const __nv_bfloat16* plane, bool keys,
                                                            int key, int d) const {
        if constexpr (CacheInput::writes_cache) {
            const int new_token = key - first_pos;
            if (key >= first_pos && new_token < valid_tokens) {
                return (keys ? input.k : input.v) +
                       kv_cache_int8_new_index<Geometry>(kv_head, d, new_token);
            }
        }
        const int page = staged_pages ? pages[(key >> kPagedKVPageShift) - first_page]
                                      : block_table[key >> kPagedKVPageShift];
        return plane + causal_cache_index<Geometry>(page, kv_head, d, key & kPagedKVPageMask);
    }

    __device__ __forceinline__ const __nv_bfloat16* key(int k) const {
        return address(cache_k, true, k, 0);
    }

    __device__ __forceinline__ uint4 key8(const __nv_bfloat16* row, int d, unsigned& bits) const {
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(row + d));
        bits |= beyond_bits(b.x) | beyond_bits(b.y) | beyond_bits(b.z) | beyond_bits(b.w);
        return make_uint4(to_f16x2(b.x), to_f16x2(b.y), to_f16x2(b.z), to_f16x2(b.w));
    }

    __device__ __forceinline__ const __nv_bfloat16* value(int k, int dim0) const {
        return address(cache_v, false, k, dim0);
    }

    __device__ __forceinline__ uint2 value4(const __nv_bfloat16* row, int n, unsigned& bits) const {
        const uint2 b = __ldg(reinterpret_cast<const uint2*>(row + 32 * n));
        bits |= beyond_bits(b.x) | beyond_bits(b.y);
        return make_uint2(to_f16x2(b.x), to_f16x2(b.y));
    }
};

// Eight int8 codes (dimension order) of eight key dimensions from their stored bytes in `w`: eight
// INT8 codes, a packed 4-bit word (the Lloyd-Max levels or the E8-snapped signed int4 codes) in
// w.x, or one E8 root block in the low half of w.x.
template <KvKeyCoding Keys>
__device__ __forceinline__ uint2 key_codes8(uint2 w) {
    if constexpr (Keys == KvKeyCoding::Int8) {
        return w;
    } else if constexpr (Keys == KvKeyCoding::RootE8) {
        return kv_cache_e8_root_decode_block(w.x & 0xffu, (w.x >> 8) & 0xffu);
    } else {
        const unsigned word = w.x;
        if constexpr (Keys == KvKeyCoding::Lloyd4) {
            return make_uint2(kv_cache_lloyd4_expand4(word), kv_cache_lloyd4_expand4(word >> 16));
        } else {
            const unsigned low  = __vsub4((word & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
            const unsigned high = __vsub4(((word >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
            return make_uint2(__byte_perm(low, high, 0x5140), __byte_perm(low, high, 0x7362));
        }
    }
}

// K and V of an INT8-family cache (rotated keys): INT8 G64 codes or a packed key coding over the
// G64 key scales, INT8 G64 values or packed signed int4 values over G32 scales. A lane reads 32
// dimensions of a row at a time in one or two 16-byte loads (kChunks): attend_tile's key chunks,
// and the 32 consecutive value dimensions of its value products.
template <typename Geometry, KvKeyCoding Keys, bool PackedValues>
struct Int8Rows {
    static constexpr bool kChunks = true;
    static constexpr int kKeyChunksInFlight = 8;
    const std::uint8_t* cache_k;
    const std::uint8_t* cache_v;
    const __half* scale_k;
    const __half* scale_v;
    const std::int32_t* pages;
    const std::int32_t* block_table;
    int first_page;
    bool staged_pages;
    int kv_head;

    struct KeyRow {
        const std::uint8_t* codes;
        uint2 scales; // the row's four G64 scales
    };

    // The stored bytes of 32 key dimensions: 32 INT8 codes, 16 bytes of a 4-bit coding, or four
    // E8 root blocks (8 bytes); `hi` holds INT8 keys' second 16 bytes only.
    struct KeyChunk {
        uint4 lo;
        uint4 hi;
    };

    // A lane's 32 value dimensions [dim0, dim0 + 32) of one key: 32 INT8 codes or 16 bytes of
    // packed int4 codes, and their one scale (G64 or G32).
    struct ValueRow {
        uint4 lo;
        uint4 hi;
        __half2 scale;
    };

    __device__ __forceinline__ int page_of(int k) const {
        return staged_pages ? pages[(k >> kPagedKVPageShift) - first_page]
                            : block_table[k >> kPagedKVPageShift];
    }

    __device__ __forceinline__ static __half2 half_of(unsigned word, int upper) {
        return __half2half2(
            __ushort_as_half(static_cast<unsigned short>(upper ? (word >> 16) : (word & 0xffffu))));
    }

    __device__ __forceinline__ KeyRow key(int k) const {
        const int page   = page_of(k);
        const int offset = k & kPagedKVPageMask;
        std::int64_t at  = 0;
        if constexpr (Keys == KvKeyCoding::Int8) {
            at = kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, offset);
        } else {
            at = kv_cache_packed_key_chunk_index<Geometry, Keys>(page, kv_head, 0, offset);
        }
        return {cache_k + at,
                *reinterpret_cast<const uint2*>(
                    scale_k + kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, offset))};
    }

    // The stored bytes of dimensions [d0, d0 + 32), d0 a multiple of 32.
    __device__ __forceinline__ KeyChunk key_chunk(const KeyRow& r, int d0) const {
        KeyChunk c{};
        if constexpr (Keys == KvKeyCoding::Int8) {
            const auto* p = reinterpret_cast<const uint4*>(r.codes + d0);
            c.lo          = p[0];
            c.hi          = p[1];
        } else if constexpr (Keys == KvKeyCoding::RootE8) {
            const uint2 v = *reinterpret_cast<const uint2*>(r.codes + d0 / 4);
            c.lo          = make_uint4(v.x, v.y, 0, 0);
        } else {
            c.lo = *reinterpret_cast<const uint4*>(r.codes + d0 / 2);
        }
        return c;
    }

    // Dimensions d0 + 8j..+7 of a chunk in FP16.
    __device__ __forceinline__ uint4 key8(const KeyRow& r, const KeyChunk& c, int d0, int j) const {
        uint2 w;
        if constexpr (Keys == KvKeyCoding::Int8) {
            const uint4& half = j < 2 ? c.lo : c.hi;
            w                 = (j & 1) ? make_uint2(half.z, half.w) : make_uint2(half.x, half.y);
        } else if constexpr (Keys == KvKeyCoding::RootE8) {
            const unsigned word = j < 2 ? c.lo.x : c.lo.y;
            w                   = make_uint2((j & 1) ? word >> 16 : word & 0xffffu, 0u);
        } else {
            w = make_uint2(j == 0 ? c.lo.x : j == 1 ? c.lo.y : j == 2 ? c.lo.z : c.lo.w, 0u);
        }
        const int group = (d0 + 8 * j) / kKVCacheInt8Group;
        const __half2 s = half_of(group < 2 ? r.scales.x : r.scales.y, group & 1);
        const uint2 codes = key_codes8<Keys>(w);
        uint4 out;
        int8x4_to_f16x2(codes.x, s, out.x, out.y);
        int8x4_to_f16x2(codes.y, s, out.z, out.w);
        return out;
    }

    // Dimensions [dim0, dim0 + 32) of key k, dim0 a multiple of 32.
    __device__ __forceinline__ ValueRow value(int k, int dim0) const {
        const int page   = page_of(k);
        const int offset = k & kPagedKVPageMask;
        ValueRow out{};
        if constexpr (PackedValues) {
            out.lo = *reinterpret_cast<const uint4*>(
                cache_v +
                kv_cache_int4_value_code_index<Geometry>(page, kv_head, dim0 / 2, offset));
            out.scale = __half2half2(scale_v[kv_cache_int4_value_scale_index<Geometry>(
                page, kv_head, dim0 / 32, offset)]);
        } else {
            const auto* p = reinterpret_cast<const uint4*>(
                cache_v + kv_cache_int8_quant_code_index<Geometry>(page, kv_head, dim0, offset));
            out.lo    = p[0];
            out.hi    = p[1];
            out.scale = __half2half2(scale_v[kv_cache_int8_quant_scale_index<Geometry>(
                page, kv_head, dim0 / kKVCacheInt8Group, offset)]);
        }
        return out;
    }

    // Dimensions dim0 + 4n..+3 of a value row.
    __device__ __forceinline__ uint2 value4(const ValueRow& r, int n) const {
        unsigned codes = 0;
        if constexpr (PackedValues) {
            const unsigned word = n < 2 ? r.lo.x : n < 4 ? r.lo.y : n < 6 ? r.lo.z : r.lo.w;
            const unsigned pair = (n & 1) ? word >> 16 : word & 0xffffu;
            const unsigned low  = __vsub4((pair & 0x0f0fu) ^ 0x0808u, 0x0808u);
            const unsigned high = __vsub4(((pair >> 4) & 0x0f0fu) ^ 0x0808u, 0x0808u);
            codes               = __byte_perm(low, high, 0x5140);
        } else {
            const uint4& half = n < 4 ? r.lo : r.hi;
            const int m       = n & 3;
            codes = m == 0 ? half.x : m == 1 ? half.y : m == 2 ? half.z : half.w;
        }
        uint2 out;
        int8x4_to_f16x2(codes, r.scale, out.x, out.y);
        return out;
    }
};

// K and V of the FP8 (row-scaled E4M3) and NVFP4 (G16 E2M1 over E4M3 scales) caches: fp8, nvfp4,
// and k8v4's FP8 keys with NVFP4 values. Keys are stored rotated in both codings, NVFP4 values too
// (their reducer rotates the output back). Chunked as Int8Rows: 32 dimensions a load.
template <typename Geometry, bool KeyFp8, bool ValueFp8, int KeyChunksInFlight = 8>
struct QuantRows {
    static constexpr bool kChunks = true;
    static constexpr int kKeyChunksInFlight = KeyChunksInFlight;
    const std::uint8_t* cache_k;
    const std::uint8_t* cache_v;
    const void* scale_k;
    const void* scale_v;
    const std::int32_t* pages;
    const std::int32_t* block_table;
    int first_page;
    bool staged_pages;
    int kv_head;

    struct KeyRow {
        const std::uint8_t* codes;
        uint4 scales; // FP8: the row's FP16 scale in the low half of x; NVFP4: its 16 E4M3 bytes
    };

    // The stored bytes of 32 dimensions: 32 FP8 codes or 16 bytes of E2M1 pairs (`hi` FP8 only).
    struct KeyChunk {
        uint4 lo;
        uint4 hi;
    };

    // A lane's 32 value dimensions [dim0, dim0 + 32) of one key and the row's scales.
    struct ValueRow {
        uint4 lo;
        uint4 hi;
        uint4 scales;
        int group0; // NVFP4: the G16 group of dim0
    };

    __device__ __forceinline__ int page_of(int k) const {
        return staged_pages ? pages[(k >> kPagedKVPageShift) - first_page]
                            : block_table[k >> kPagedKVPageShift];
    }

    __device__ __forceinline__ static unsigned bits(__half2 v) {
        return *reinterpret_cast<const unsigned*>(&v);
    }

    __device__ __forceinline__ static unsigned fp8x2(unsigned codes, __half2 s) {
        return bits(__hmul2(kv_cache_fp8_code2_to_half2_x2m8(codes), s));
    }

    __device__ __forceinline__ static unsigned e2m1x2(unsigned byte, __half2 s) {
        return bits(__hmul2(kv_cache_e2m1x2_to_half2(byte), s));
    }

    // The row's FP16 scale times 2^8 for fp8x2 (exact, and at most 65504 / 448 * 256).
    __device__ __forceinline__ static __half2 fp16_scale(const uint4& scales) {
        const __half s = __ushort_as_half(static_cast<unsigned short>(scales.x & 0xffffu));
        return __half2half2(__hmul(s, __float2half_rn(256.0f)));
    }

    __device__ __forceinline__ static __half2 e4m3_scale(const uint4& scales, int group) {
        const unsigned word =
            group < 8 ? (group < 4 ? scales.x : scales.y) : (group < 12 ? scales.z : scales.w);
        const unsigned code = (word >> (8 * (group & 3))) & 0xffu;
        // Every E4M3 scale is an exact FP16, and the bit-level decode times 2^8 restores it.
        return __hmul2(kv_cache_fp8_code2_to_half2_x2m8(code | (code << 8)),
                       __float2half2_rn(256.0f));
    }

    template <bool Fp8>
    __device__ __forceinline__ static uint4 load_scales(const void* plane, int page, int kv_head,
                                                        int offset) {
        if constexpr (Fp8) {
            const __half s = static_cast<const __half*>(
                plane)[kv_cache_fp8_scale_index<Geometry>(page, kv_head, offset)];
            return make_uint4(__half_as_ushort(s), 0, 0, 0);
        } else {
            return *reinterpret_cast<const uint4*>(
                static_cast<const std::uint8_t*>(plane) +
                kv_cache_nvfp4_scale_index<Geometry>(page, kv_head, 0, offset));
        }
    }

    __device__ __forceinline__ KeyRow key(int k) const {
        const int page        = page_of(k);
        const int offset      = k & kPagedKVPageMask;
        const std::int64_t at = KeyFp8
                                    ? kv_cache_fp8_code_index<Geometry>(page, kv_head, 0, offset)
                                    : kv_cache_nvfp4_code_index<Geometry>(page, kv_head, 0, offset);
        return {cache_k + at, load_scales<KeyFp8>(scale_k, page, kv_head, offset)};
    }

    // The stored bytes of dimensions [d0, d0 + 32), d0 a multiple of 32.
    __device__ __forceinline__ KeyChunk key_chunk(const KeyRow& r, int d0) const {
        KeyChunk c{};
        if constexpr (KeyFp8) {
            const auto* p = reinterpret_cast<const uint4*>(r.codes + d0);
            c.lo          = p[0];
            c.hi          = p[1];
        } else {
            c.lo = *reinterpret_cast<const uint4*>(r.codes + d0 / 2);
        }
        return c;
    }

    // Dimensions d0 + 8j..+7 of a chunk in FP16.
    __device__ __forceinline__ uint4 key8(const KeyRow& r, const KeyChunk& c, int d0, int j) const {
        if constexpr (KeyFp8) {
            const uint4& half = j < 2 ? c.lo : c.hi;
            const uint2 w = (j & 1) ? make_uint2(half.z, half.w) : make_uint2(half.x, half.y);
            const __half2 s = fp16_scale(r.scales);
            return make_uint4(fp8x2(w.x, s), fp8x2(w.x >> 16, s), fp8x2(w.y, s),
                              fp8x2(w.y >> 16, s));
        } else {
            const unsigned w = j == 0 ? c.lo.x : j == 1 ? c.lo.y : j == 2 ? c.lo.z : c.lo.w;
            const __half2 s  = e4m3_scale(r.scales, (d0 + 8 * j) / kKVCacheNvfp4Group);
            return make_uint4(e2m1x2(w, s), e2m1x2(w >> 8, s), e2m1x2(w >> 16, s),
                              e2m1x2(w >> 24, s));
        }
    }

    // Dimensions [dim0, dim0 + 32) of key k, dim0 a multiple of 32.
    __device__ __forceinline__ ValueRow value(int k, int dim0) const {
        const int page   = page_of(k);
        const int offset = k & kPagedKVPageMask;
        ValueRow out{};
        if constexpr (ValueFp8) {
            const auto* p = reinterpret_cast<const uint4*>(
                cache_v + kv_cache_fp8_code_index<Geometry>(page, kv_head, dim0, offset));
            out.lo = p[0];
            out.hi = p[1];
        } else {
            out.lo = *reinterpret_cast<const uint4*>(
                cache_v + kv_cache_nvfp4_code_index<Geometry>(page, kv_head, dim0, offset));
        }
        out.scales = load_scales<ValueFp8>(scale_v, page, kv_head, offset);
        out.group0 = dim0 / kKVCacheNvfp4Group;
        return out;
    }

    // Dimensions dim0 + 4n..+3 of a value row: NVFP4 group group0 + n / 4.
    __device__ __forceinline__ uint2 value4(const ValueRow& r, int n) const {
        if constexpr (ValueFp8) {
            const uint4& half = n < 4 ? r.lo : r.hi;
            const int m       = n & 3;
            const unsigned c  = m == 0 ? half.x : m == 1 ? half.y : m == 2 ? half.z : half.w;
            const __half2 s   = fp16_scale(r.scales);
            return make_uint2(fp8x2(c, s), fp8x2(c >> 16, s));
        } else {
            const unsigned word = n < 2 ? r.lo.x : n < 4 ? r.lo.y : n < 6 ? r.lo.z : r.lo.w;
            const unsigned c    = (n & 1) ? word >> 16 : word & 0xffffu;
            const __half2 s     = e4m3_scale(r.scales, r.group0 + n / 4);
            return make_uint2(e2m1x2(c, s), e2m1x2(c >> 8, s));
        }
    }
};

// NVFP4's scalar row access for the one/two-tile H16 verification splits. Keep the same
// represented FP16 operands, but read each MMA fragment separately and use its logical output
// dimensions. The chunked provider still owns the common pages and stored-scale decoding.
template <typename Geometry>
struct Nvfp4ScalarRows {
    static constexpr bool kChunks = false;
    using PackedRows = QuantRows<Geometry, false, false>;
    using KeyRow = typename PackedRows::KeyRow;
    PackedRows packed;

    struct ValueRow {
        const std::uint8_t* codes;
        uint4 scales;
        int group0;
    };

    __device__ __forceinline__ KeyRow key(int k) const { return packed.key(k); }

    __device__ __forceinline__ uint4 key8(const KeyRow& row, int d) const {
        const unsigned codes = *reinterpret_cast<const unsigned*>(row.codes + d / 2);
        const __half2 scale = PackedRows::e4m3_scale(row.scales, d / kKVCacheNvfp4Group);
        return make_uint4(PackedRows::e2m1x2(codes, scale),
                          PackedRows::e2m1x2(codes >> 8, scale),
                          PackedRows::e2m1x2(codes >> 16, scale),
                          PackedRows::e2m1x2(codes >> 24, scale));
    }

    __device__ __forceinline__ ValueRow value(int k, int dim0) const {
        const int page = packed.page_of(k);
        const int offset = k & kPagedKVPageMask;
        const auto at = kv_cache_nvfp4_code_index<Geometry>(page, packed.kv_head, dim0, offset);
        return {packed.cache_v + at,
                PackedRows::template load_scales<false>(packed.scale_v, page, packed.kv_head,
                                                        offset),
                dim0 / kKVCacheNvfp4Group};
    }

    __device__ __forceinline__ uint2 value4(const ValueRow& row, int n) const {
        const unsigned codes = *reinterpret_cast<const std::uint16_t*>(row.codes + 16 * n);
        const __half2 scale = PackedRows::e4m3_scale(row.scales, row.group0 + 2 * n);
        return make_uint2(PackedRows::e2m1x2(codes, scale),
                          PackedRows::e2m1x2(codes >> 8, scale));
    }
};

// Rows whose lanes read 32 dimensions of a row at a time (Int8Rows::kChunks).
template <class Rows, class = void>
struct chunked_rows : std::false_type {};
template <class Rows>
struct chunked_rows<Rows, std::void_t<decltype(Rows::kChunks)>>
    : std::bool_constant<Rows::kChunks> {};

// Rows of BF16 values that FP16 may not hold (Bf16Rows::kExact).
template <class Rows, class = void>
struct exact_rows : std::false_type {};
template <class Rows>
struct exact_rows<Rows, std::void_t<decltype(Rows::kExact)>> : std::bool_constant<Rows::kExact> {};

// Eight BF16 values of a and of b: sum a_e b_e in FP32.
__device__ __forceinline__ float dot8_bf16(uint4 a, uint4 b) {
    const unsigned aw[4] = {a.x, a.y, a.z, a.w};
    const unsigned bw[4] = {b.x, b.y, b.z, b.w};
    float sum            = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sum = fmaf(__uint_as_float(aw[i] << 16), __uint_as_float(bw[i] << 16), sum);
        sum = fmaf(__uint_as_float(aw[i] & 0xffff0000u), __uint_as_float(bw[i] & 0xffff0000u), sum);
    }
    return sum;
}

// The tile's unscaled scores in FP32 from the BF16 queries and keys, in place of the FP16 products:
// element i is row (lane & 1) + (i & 2) + 4h of the warp's rows from q_row0 and key k0 + 8 qp +
// (i & 4) + (lane & 2) + (i & 1). Keys outside [key_lo, key_hi) and rows without a query score 0
// (the caller masks them).
template <class Rows>
__device__ __forceinline__ void exact_scores(const Rows& rows, int k0, int key_lo, int key_hi,
                                          int q_row0, float (&s)[8]) {
    constexpr int D = kCausalHeadDim;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int qp    = (lane >> 2) & 3;
    const int h     = lane >> 4;
    const __nv_bfloat16* query[2] = {rows.query(q_row0 + (lane & 1) + 4 * h),
                                     rows.query(q_row0 + (lane & 1) + 4 * h + 2)};
    const __nv_bfloat16* key[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int k = k0 + qp * 8 + 4 * (e >> 1) + (lane & 2) + (e & 1);
        key[e]      = k >= key_lo && k < key_hi ? rows.key_bf16(k) : nullptr;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) { s[i] = 0.0f; }
#pragma unroll 1
    for (int d = 0; d < D; d += 8) {
        uint4 q8[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            q8[r] = query[r] != nullptr ? __ldg(reinterpret_cast<const uint4*>(query[r] + d))
                                        : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (key[e] == nullptr) { continue; }
            const uint4 k8 = __ldg(reinterpret_cast<const uint4*>(key[e] + d));
            const int i    = 4 * (e >> 1) + (e & 1);
            s[i] += dot8_bf16(q8[0], k8);
            s[i + 2] += dot8_bf16(q8[1], k8);
        }
    }
}

// Adds to the warp's outputs what saturating the tile's BF16 values to FP16 took from P V: the
// FP16 probability in p_tile times the value's excess (detail::sm70::saturated_excess), for each
// output element of the lane (row (lane & 1) + (i & 2) + 4h, dimension 32n + 8qp + (i & 4) +
// (lane & 2) + (i & 1)). A zero probability adds nothing, whatever the value.
template <class Rows>
__device__ __forceinline__ void add_value_excess(const Rows& rows, const __half* p_tile, int k0,
                                              int key_lo, int key_hi, float (&acc)[8][8]) {
    constexpr int Bc      = 32;
    constexpr int PStride = kPStride;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int qp          = (lane >> 2) & 3;
    const int row_a       = (lane & 1) + 4 * (lane >> 4);
#pragma unroll 1
    for (int kk = 0; kk < Bc; ++kk) {
        const int k = k0 + kk;
        if (k < key_lo || k >= key_hi) { continue; }
        const float p[2] = {__half2float(p_tile[row_a * PStride + kk]),
                            __half2float(p_tile[(row_a + 2) * PStride + kk])};
        const __nv_bfloat16* v = rows.value_bf16(k);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int d         = 32 * n + 8 * qp + 4 * half + (lane & 2);
                const unsigned word = *reinterpret_cast<const unsigned*>(v + d);
                const float lo      = detail::sm70::saturated_excess(__uint_as_float(word << 16));
                const float hi =
                    detail::sm70::saturated_excess(__uint_as_float(word & 0xffff0000u));
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (p[r] == 0.0f) { continue; }
                    const int i = 4 * half + 2 * r;
                    acc[n][i] = fmaf(p[r], lo, acc[n][i]);
                    acc[n][i + 1] = fmaf(p[r], hi, acc[n][i + 1]);
                }
            }
        }
    }
}

// The value dimension of accumulator element i of product n in a lane: column c = (i & 4) +
// (lane & 2) + (i & 1) of product n is dimension 32n + 8qp + c, or, for chunked rows, 64qp +
// 32(c / 4) + 4n + c % 4, which gives each lane's B operands 32 consecutive dimensions.
template <bool Chunks>
__device__ __forceinline__ int acc_dim(int lane, int n, int i) {
    const int qp     = (lane >> 2) & 3;
    const int column = (lane & 2) + (i & 1);
    return Chunks ? 64 * qp + 8 * (i & 4) + 4 * n + column : 32 * n + 8 * qp + (i & 4) + column;
}

// The online-softmax state of a warp's eight query rows in one lane: rows a ((lane & 1) + 4h)
// and b (a + 2) of the group, and the lane's share of their FP32 outputs (element i of product n is
// row (lane & 1) + (i & 2) + 4h, dimension acc_dim(lane, n, i)).
struct RowState {
    float m_a;
    float m_b;
    float l_a;
    float l_b;
    float acc[8][8];

    __device__ __forceinline__ RowState() : l_a(0.0f), l_b(0.0f) {
        m_a = -CUDART_INF_F;
        m_b = -CUDART_INF_F;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
        }
    }
};

// One key tile [k0, k0 + 32) of a warp's eight query rows: S = Q K^T as m8n8k4.row.col products
// (q_row is this lane's A row, Q in FP16 QStride halves apart), the masked online softmax in FP32,
// P through the warp's p_tile into m8n8k4.row.row products with V. Keys outside [key_lo, key_hi)
// and past a row's position (qabs_a, qabs_b; -1 for no row) are masked. `rows` supplies K and V in
// FP16: key(k) and key8(row, d) the eight dimensions from d of key k (chunked rows: key_chunk(row,
// d0) the stored bytes of 32 and key8(row, chunk, d0, j) the eight from d0 + 8j), value(k, dim0)
// and value4(row, n) the four from dim0 + 32 n (chunked rows: from dim0 + 4n). Exact rows
// (exact_rows) also flag saturated keys and values (keys_beyond, values_beyond, query_beyond) and
// give their BF16 values, the warp's rows numbered from q_row0.
template <class Rows>
__device__ __forceinline__ void attend_tile(const Rows& rows, const __half* q_row, __half* p_tile,
                                            int k0, int key_lo, int key_hi, int qabs_a, int qabs_b,
                                            float scale, RowState& st, int q_row0 = 0) {
    constexpr int D             = kCausalHeadDim;
    constexpr int Bc            = 32;
    constexpr int PStride       = kPStride;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int qp                = (lane >> 2) & 3;
    const int h                 = lane >> 4;
    const int a_row             = (lane & 3) + 4 * h;

    constexpr bool Chunks = chunked_rows<Rows>::value;
    constexpr bool Exact  = exact_rows<Rows>::value;

    // S = Q K^T: this lane supplies the K row of key k0 + 8 qp + (lane % 4) + 4h.
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    unsigned key_bits = 0u;
    {
        const int key    = k0 + qp * 8 + (lane & 3) + 4 * h;
        const bool live  = key >= key_lo && key < key_hi;
        const auto k_row = rows.key(live ? key : k0);
        if constexpr (Chunks) {
#pragma unroll (Rows::kKeyChunksInFlight)
            for (int d0 = 0; d0 < D; d0 += 32) {
                const auto chunk = rows.key_chunk(k_row, d0);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint4 k8 = live ? rows.key8(k_row, chunk, d0, j) : make_uint4(0, 0, 0, 0);
                    const uint4 q8 = *reinterpret_cast<const uint4*>(q_row + d0 + 8 * j);
                    hmma<false>(s, q8.x, q8.y, k8.x, k8.y);
                    hmma<false>(s, q8.z, q8.w, k8.z, k8.w);
                }
            }
        } else {
#pragma unroll 8
            for (int d = 0; d < D; d += 8) {
                uint4 k8 = make_uint4(0, 0, 0, 0);
                if (live) {
                    if constexpr (Exact) {
                        k8 = rows.key8(k_row, d, key_bits);
                    } else {
                        k8 = rows.key8(k_row, d);
                    }
                }
                const uint4 q8 = *reinterpret_cast<const uint4*>(q_row + d);
                hmma<false>(s, q8.x, q8.y, k8.x, k8.y);
                hmma<false>(s, q8.z, q8.w, k8.z, k8.w);
            }
        }
    }
    if constexpr (Exact) {
        if (rows.keys_beyond(key_bits) || rows.query_beyond) {
            exact_scores(rows, k0, key_lo, key_hi, q_row0, s);
        }
    }
    // Masked, scaled scores and the tile's row maxima (rows a: i & 2 == 0, b: i & 2 == 2).
    float bm_a = -CUDART_INF_F, bm_b = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key    = k0 + qp * 8 + (i & 4) + (lane & 2) + (i & 1);
        const bool upper = (i & 2) != 0;
        const int qabs   = upper ? qabs_b : qabs_a;
        const bool keep  = qabs >= 0 && key >= key_lo && key < key_hi && key <= qabs;
        s[i]             = keep ? s[i] * scale : -CUDART_INF_F;
        if (upper) {
            bm_b = fmaxf(bm_b, s[i]);
        } else {
            bm_a = fmaxf(bm_a, s[i]);
        }
    }
#pragma unroll
    for (int mask = 2; mask <= 8; mask <<= 1) {
        bm_a = fmaxf(bm_a, __shfl_xor_sync(FullMask, bm_a, mask));
        bm_b = fmaxf(bm_b, __shfl_xor_sync(FullMask, bm_b, mask));
    }
    const float nm_a    = fmaxf(st.m_a, bm_a);
    const float nm_b    = fmaxf(st.m_b, bm_b);
    const float alpha_a = st.m_a == -CUDART_INF_F ? 0.0f : exp2_approx((st.m_a - nm_a) * Log2E);
    const float alpha_b = st.m_b == -CUDART_INF_F ? 0.0f : exp2_approx((st.m_b - nm_b) * Log2E);
    float bl_a = 0.0f, bl_b = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool upper = (i & 2) != 0;
        const float nm   = upper ? nm_b : nm_a;
        const float p =
            (nm > -CUDART_INF_F && s[i] > -CUDART_INF_F) ? exp2_approx((s[i] - nm) * Log2E) : 0.0f;
        if (upper) {
            bl_b += p;
        } else {
            bl_a += p;
        }
        const int row               = (lane & 1) + (i & 2) + 4 * h;
        const int col               = qp * 8 + (i & 4) + (lane & 2) + (i & 1);
        p_tile[row * PStride + col] = __float2half_rn(p);
    }
#pragma unroll
    for (int mask = 2; mask <= 8; mask <<= 1) {
        bl_a += __shfl_xor_sync(FullMask, bl_a, mask);
        bl_b += __shfl_xor_sync(FullMask, bl_b, mask);
    }
    st.l_a = st.l_a * alpha_a + bl_a;
    st.l_b = st.l_b * alpha_b + bl_b;
    st.m_a = nm_a;
    st.m_b = nm_b;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { st.acc[n][i] *= (i & 2) ? alpha_b : alpha_a; }
    }
    __syncwarp();
    // O += P V: four keys per product; the quad pairs take eight of each 32 dimensions, or, for
    // chunked rows, 64 consecutive dimensions each.
    const int value_dim0 = Chunks ? 64 * qp + 32 * h : 8 * qp + 4 * h;
    unsigned value_bits  = 0u;
#pragma unroll 2
    for (int step = 0; step < Bc / 4; ++step) {
        const uint2 p4   = *reinterpret_cast<const uint2*>(p_tile + a_row * PStride + 4 * step);
        const int key    = k0 + 4 * step + (lane & 3);
        const bool live  = key >= key_lo && key < key_hi;
        const auto v_row = rows.value(live ? key : k0, value_dim0);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            uint2 v4 = make_uint2(0, 0);
            if (live) {
                if constexpr (Exact) {
                    v4 = rows.value4(v_row, n, value_bits);
                } else {
                    v4 = rows.value4(v_row, n);
                }
            }
            hmma<true>(st.acc[n], p4.x, p4.y, v4.x, v4.y);
        }
    }
    if constexpr (Exact) {
        if (rows.values_beyond(value_bits)) {
            add_value_excess(rows, p_tile, k0, key_lo, key_hi, st.acc);
        }
    }
    __syncwarp();
}

// The products, softmax and partial outputs of one CTA, after its query rows are in q_s (FP16,
// QStride halves apart, zero past row_count). `rows` supplies K and V as attend_tile reads them.
template <typename Geometry, int TokenTile, class Rows>
__device__ __forceinline__ void
attend(const Rows& rows, const __half* q_s, __half* p_s, float (*stat_m)[8], float (*stat_l)[8],
       float* combined, int tokens, int row_count, int kv_head, int split, int split_start,
       int split_end, int first_tile, int key_blocks, const std::int32_t* pos, float scale,
       float* partial_acc, float* partial_m, float* partial_l) {
    using Dims             = Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D        = kCausalHeadDim;
    constexpr int Bc       = 32;
    constexpr int Groups   = Dims::kRowGroups;
    constexpr int KeyWarps = Dims::kKeyWarps;
    constexpr int Threads  = 32 * Dims::kWarps;
    constexpr int QStride  = kQStride;
    constexpr int PStride  = kPStride;
    constexpr float Log2E  = 1.4426950408889634074f;
    constexpr bool Chunks  = chunked_rows<Rows>::value;
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;

    const int group    = warp / KeyWarps;
    const int key_warp = warp % KeyWarps;
    const int h        = lane >> 4;
    const int a_row    = (lane & 3) + 4 * h;
    // The accumulator rows of this lane: element i is row (lane & 1) + (i & 2) + 4h of the group.
    const int row_a           = group * 8 + (lane & 1) + 4 * h;
    const int row_b           = row_a + 2;
    const auto query_position = [&](int row) {
        return row < row_count ? pos[row / Geometry::GroupSize] : -1;
    };
    const int qabs_a    = query_position(row_a);
    const int qabs_b    = query_position(row_b);
    const __half* q_row = q_s + (group * 8 + a_row) * QStride;
    __half* p_tile      = p_s + warp * 8 * PStride;

    RowState st;
    for (int block = key_warp; block < key_blocks; block += KeyWarps) {
        attend_tile(rows, q_row, p_tile, first_tile + block * Bc, split_start, split_end, qabs_a,
                    qabs_b, scale, st, group * 8);
    }
    float& m_a         = st.m_a;
    float& m_b         = st.m_b;
    float& l_a         = st.l_a;
    float& l_b         = st.l_b;
    float (&acc)[8][8] = st.acc;

    // Element i of product n: row (lane & 1) + (i & 2) + 4h, dimension acc_dim(lane, n, i).
    const auto write_row = [&](int row, float m, float l) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m;
        partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l;
    };
    if constexpr (KeyWarps == 1) {
        if ((lane & 14) == 0) {
            if (row_a < row_count) { write_row(row_a, m_a, l_a); }
            if (row_b < row_count) { write_row(row_b, m_b, l_b); }
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (i & 2) ? row_b : row_a;
                if (row >= row_count) { continue; }
                int q_head = 0;
                int token  = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
                const int d                            = acc_dim<Chunks>(lane, n, i);
                *reinterpret_cast<float2*>(&partial_acc[causal_partial_acc_index<Geometry>(
                    q_head, d, token, split, tokens)]) = make_float2(acc[n][i], acc[n][i + 1]);
            }
        }
    } else {
        // Combine the row group's key warps: their maxima, then each warp's share rescaled to
        // the common maximum and added in shared memory, warp by warp in a fixed order (a repeated
        // call reproduces every bit).
        if ((lane & 14) == 0) {
            const int local_a         = (lane & 1) + 4 * h;
            stat_m[warp][local_a]     = m_a;
            stat_m[warp][local_a + 2] = m_b;
            stat_l[warp][local_a]     = l_a;
            stat_l[warp][local_a + 2] = l_b;
        }
        __syncthreads();
        const auto common_max = [&](int local) {
            float m = -CUDART_INF_F;
#pragma unroll
            for (int w = 0; w < KeyWarps; ++w) {
                m = fmaxf(m, stat_m[group * KeyWarps + w][local]);
            }
            return m;
        };
        const int local_a = (lane & 1) + 4 * h;
        const float max_a = common_max(local_a);
        const float max_b = common_max(local_a + 2);
        const float f_a   = m_a == -CUDART_INF_F ? 0.0f : exp2_approx((m_a - max_a) * Log2E);
        const float f_b   = m_b == -CUDART_INF_F ? 0.0f : exp2_approx((m_b - max_b) * Log2E);
        float* group_acc  = combined + group * 8 * D;
        // Chunked rows keep a row's dimension d at slot_of(d, row): their quad pairs' dimensions
        // are 64 apart, one bank, so bits 6-7 move to 3-4, and rows are D apart, so odd rows flip
        // bit 2. A warp's adds then meet at most two to a bank.
        const auto slot_of = [](int d, int row) {
            return Chunks ? d ^ ((row & 1) << 2) ^ (((d >> 6) & 3) << 3) : d;
        };
        for (int w = 0; w < KeyWarps; ++w) {
            if (key_warp == w) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int local   = (lane & 1) + (i & 2) + 4 * h;
                        const int d       = acc_dim<Chunks>(lane, n, i);
                        const float share = acc[n][i] * ((i & 2) ? f_b : f_a);
                        float& slot       = group_acc[local * D + slot_of(d, local)];
                        slot              = w == 0 ? share : slot + share;
                    }
                }
            }
            __syncthreads();
        }
        // One thread per (row, two dimensions) writes the combined partial.
        for (int idx = tid; idx < Groups * 8 * (D / 2); idx += Threads) {
            const int row = idx / (D / 2);
            const int d   = (idx - row * (D / 2)) * 2;
            if (row >= row_count) { continue; }
            const int g     = row / 8;
            const int local = row % 8;
            int q_head      = 0;
            int token       = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (!causal_valid_q_head<Geometry>(kv_head, q_head)) { continue; }
            float m = -CUDART_INF_F;
#pragma unroll
            for (int w = 0; w < KeyWarps; ++w) { m = fmaxf(m, stat_m[g * KeyWarps + w][local]); }
            if (d == 0) {
                float l = 0.0f;
#pragma unroll
                for (int w = 0; w < KeyWarps; ++w) {
                    const float mw = stat_m[g * KeyWarps + w][local];
                    l += mw == -CUDART_INF_F
                             ? 0.0f
                             : stat_l[g * KeyWarps + w][local] * exp2_approx((mw - m) * Log2E);
                }
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l;
            }
            const float* src = combined + (g * 8 + local) * D + slot_of(d, local);
            *reinterpret_cast<float2*>(
                &partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) =
                make_float2(src[0], src[1]);
        }
    }
}

} // namespace sm70_attention

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
__launch_bounds__(32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps) __global__
    void causal_attention_small_t_sm70_bf16_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
        __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
        std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
        float scale, float* partial_acc, float* partial_m, float* partial_l) {
    using Shape            = sm70_attention::Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D        = kCausalHeadDim;
    constexpr int Bc       = 32;
    constexpr int Groups   = Shape::kRowGroups;
    constexpr int KeyWarps = Shape::kKeyWarps;
    constexpr int Threads  = 32 * Shape::kWarps;
    constexpr int QStride  = sm70_attention::kQStride;
    constexpr int PStride  = sm70_attention::kPStride;
    constexpr int PageIds  = 64;

    __shared__ __align__(16) __half q_s[Groups * 8 * QStride];
    __shared__ __align__(16) __half p_s[Shape::kWarps * 8 * PStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float stat_m[Shape::kWarps][8];
    __shared__ float stat_l[Shape::kWarps][8];
    __shared__ float combined[KeyWarps > 1 ? Groups * 8 * D : 1];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    int valid_tokens      = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Groups * 8 || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int live_end = last_pos + 1;
    const int window   = causal_small_t_tile_window(first_pos, tokens, logical_capacity);
    const int active_split_count =
        causal_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    // Keys at or past the live end are beyond every live column: none is loaded.
    const int load_end = split_end < live_end ? split_end : live_end;

    const int first_tile    = (split_start / Bc) * Bc;
    const int key_blocks    = div_up(split_end - first_tile, Bc);
    const int first_page    = first_tile >> kPagedKVPageShift;
    const int page_count    = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    const bool staged_pages = page_count <= PageIds;
    if (staged_pages) {
        for (int page = tid; page < page_count; page += Threads) {
            physical_pages_s[page] = block_table[first_page + page];
        }
    }
    __syncthreads();

    if constexpr (CacheInput::writes_cache) {
        // The owning split writes each new row. Current attention reads those rows directly from
        // input below, so no split depends on another split's cache write.
        for (int chunk = tid; chunk < valid_tokens * (D / 8); chunk += Threads) {
            const int token = chunk / (D / 8);
            const int d     = (chunk - token * (D / 8)) * 8;
            const int p_tok = pos[token];
            if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                p_tok < logical_capacity) {
                const std::int64_t new_off = kv_cache_int8_new_index<Geometry>(kv_head, d, token);
                const std::int64_t cache_off =
                    causal_cache_index<Geometry>(paged_kv_physical_page(block_table, p_tok),
                                                 kv_head, d, p_tok & kPagedKVPageMask);
                store_vec(&cache_k[cache_off], load_vec<int4>(&input.k[new_off]));
                store_vec(&cache_v[cache_off], load_vec<int4>(&input.v[new_off]));
            }
        }
    }

    // The CTA's query rows in FP16, zero past row_count.
    unsigned query_bits = 0u;
    for (int idx = tid; idx < Groups * 8 * (D / 2); idx += Threads) {
        const int row = idx / (D / 2);
        const int d   = (idx - row * (D / 2)) * 2;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        unsigned value = 0;
        if (row < row_count && causal_valid_q_head<Geometry>(kv_head, q_head)) {
            const unsigned word =
                *reinterpret_cast<const unsigned*>(&q[causal_q_index<Geometry>(q_head, d, token)]);
            query_bits |= sm70_attention::beyond_bits(word);
            value = sm70_attention::to_f16x2(word);
        }
        *reinterpret_cast<unsigned*>(&q_s[row * QStride + d]) = value;
    }
    const bool query_beyond = __syncthreads_or((query_bits & sm70_attention::kBeyondBits) != 0u);

    const sm70_attention::Bf16Rows<Geometry, CacheInput> rows{
        cache_k,    cache_v,      input,   physical_pages_s, block_table,
        first_page, staged_pages, kv_head, first_pos,        valid_tokens,
        q,          tokens,       row_count, query_beyond};
    sm70_attention::attend<Geometry, TokenTile>(
        rows, q_s, p_s, stat_m, stat_l, combined, tokens, row_count, kv_head, split, split_start,
        load_end, first_tile, key_blocks, pos, scale, partial_acc, partial_m, partial_l);
}

// The INT8-family caches (rotated keys in any key coding, INT8 or packed int4 values), launched by
// small_t_i8_launch.cuh in place of the tiled kernel: the same grid, active splits (wave_splits
// included), fused append and partial outputs, with this file's products.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool PackedValues, KvKeyCoding Keys>
__launch_bounds__(32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps,
                  sm70_attention::i8_resident_ctas<TokenTile, TokenTile * Geometry::GroupSize>())
    __global__ void causal_attention_small_t_sm70_i8_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t wave_splits,
        float scale, float* partial_acc, float* partial_m, float* partial_l) {
    using Shape                 = sm70_attention::Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D             = kCausalHeadDim;
    constexpr int Bc            = 32;
    constexpr int Groups        = Shape::kRowGroups;
    constexpr int KeyWarps      = Shape::kKeyWarps;
    constexpr int Warps         = Shape::kWarps;
    constexpr int Threads       = 32 * Warps;
    constexpr int QStride       = sm70_attention::kQStride;
    constexpr int PStride       = sm70_attention::kPStride;
    constexpr int RowCount      = TokenTile * Geometry::GroupSize;
    constexpr int KeyGroups     = kKVCacheInt8Groups;
    constexpr int PageIds       = 64;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) __half q_s[Groups * 8 * QStride];
    __shared__ __align__(16) __half p_s[Warps * 8 * PStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float stat_m[Warps][8];
    __shared__ float stat_l[Warps][8];
    __shared__ float combined[KeyWarps > 1 ? Groups * 8 * D : 1];
    __shared__ float k_h64_s[CacheInput::writes_cache ? TokenTile * D : 1];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               TokenTile)] = 0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int live_end = last_pos + 1;
    const int window   = causal_small_t_tile_window(first_pos, TokenTile, logical_capacity);
    const int active_split_count =
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile, wave_splits);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    // Keys at or past the live end are beyond every live column: none is loaded.
    const int load_end = split_end < live_end ? split_end : live_end;

    const int first_tile    = (split_start / Bc) * Bc;
    const int key_blocks    = div_up(split_end - first_tile, Bc);
    const int first_page    = first_tile >> kPagedKVPageShift;
    const int page_count    = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    const bool staged_pages = page_count <= PageIds;
    if (staged_pages) {
        for (int page = tid; page < page_count; page += Threads) {
            physical_pages_s[page] = block_table[first_page + page];
        }
    }
    __syncthreads();
    const auto page_at = [&](int key) {
        return staged_pages ? physical_pages_s[(key >> kPagedKVPageShift) - first_page]
                            : block_table[key >> kPagedKVPageShift];
    };

    if constexpr (CacheInput::writes_cache) {
        // The fused append of small_t_i8.cuh: the owning split rotates (H64 fragments, then the
        // H4 leaf) and quantizes each new key row and quantizes its value row; every key, the new
        // ones included, is then read back from the cache.
        for (int pair = warp; pair < valid_tokens * KeyGroups; pair += Warps) {
            const int token    = pair / KeyGroups;
            const int grp      = pair - token * KeyGroups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            k_h64_s[token * D + d0] = k_h64[0];
            k_h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();

        for (int pair = warp; pair < valid_tokens * KeyGroups; pair += Warps) {
            const int token    = pair / KeyGroups;
            const int grp      = pair - token * KeyGroups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = page_at(position);
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }
            kv_cache_i8_family_store_key_group<Geometry, Keys>(
                cache_k_i8, cache_k_scale, physical_page, kv_head, grp, page_offset, lane, k_out[0],
                k_out[1]);
            // The vv0 lanes span dimensions [64g, 64g+32) and the vv1 lanes the next 32, so the
            // packed coding's two G32 groups fall out of the lane assignment directly.
            const float vv0       = __bfloat162float(input.v[src0]);
            const float vv1       = __bfloat162float(input.v[src1]);
            const float vamax_lo  = warp_max(fabsf(vv0), FullMask);
            const float vamax_hi  = warp_max(fabsf(vv1), FullMask);
            const auto v_quant    = PackedValues
                                        ? kv_cache_int4_quant_params(vamax_lo)
                                        : kv_cache_int8_quant_params(fmaxf(vamax_lo, vamax_hi));
            const auto v_quant_hi = PackedValues ? kv_cache_int4_quant_params(vamax_hi) : v_quant;
            if constexpr (PackedValues) {
                // A packed byte holds the adjacent dimension pair, which spans lanes l and l^1.
                const std::int8_t c0 = kv_cache_int4_quant_code(vv0, v_quant.inverse_scale);
                const std::int8_t c1 = kv_cache_int4_quant_code(vv1, v_quant_hi.inverse_scale);
                const int partner0   = __shfl_xor_sync(FullMask, static_cast<int>(c0), 1);
                const int partner1   = __shfl_xor_sync(FullMask, static_cast<int>(c1), 1);
                if ((lane & 1) == 0) {
                    auto* packed_v                 = reinterpret_cast<std::uint8_t*>(cache_v_i8);
                    const std::int64_t packed_base = kv_cache_int4_value_code_index<Geometry>(
                        physical_page, kv_head, grp * (kKVCacheInt8Group / 2), page_offset);
                    packed_v[packed_base + (lane >> 1)] =
                        kv_cache_int4_pack(c0, static_cast<std::int8_t>(partner0));
                    packed_v[packed_base + (lane >> 1) + 16] =
                        kv_cache_int4_pack(c1, static_cast<std::int8_t>(partner1));
                }
            } else {
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                    page_offset)] =
                    kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
                cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                    page_offset)] =
                    kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            }
            if (lane == 0) {
                if constexpr (PackedValues) {
                    cache_v_scale[kv_cache_int4_value_scale_index<Geometry>(
                        physical_page, kv_head, 2 * grp, page_offset)]     = v_quant.scale;
                    cache_v_scale[kv_cache_int4_value_scale_index<Geometry>(
                        physical_page, kv_head, 2 * grp + 1, page_offset)] = v_quant_hi.scale;
                } else {
                    cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                        physical_page, kv_head, grp, page_offset)] = v_quant.scale;
                }
            }
        }
        __syncthreads();
    }

    // Q rotated by the keys' normalized D256 transform, in FP16 (zero past RowCount).
    for (int row = warp; row < Groups * 8; row += Warps) {
        float values[8];
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        const bool live = row < RowCount && causal_valid_q_head<Geometry>(kv_head, q_head);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] =
                live ? __bfloat162float(q[causal_q_index<Geometry>(q_head, lane + 32 * r, token)])
                     : 0.0f;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            q_s[row * QStride + lane + 32 * r] = __float2half_rn(live ? values[r] : 0.0f);
        }
    }
    __syncthreads();

    const sm70_attention::Int8Rows<Geometry, Keys, PackedValues> rows{
        reinterpret_cast<const std::uint8_t*>(cache_k_i8),
        reinterpret_cast<const std::uint8_t*>(cache_v_i8),
        cache_k_scale,
        cache_v_scale,
        physical_pages_s,
        block_table,
        first_page,
        staged_pages,
        kv_head};
    sm70_attention::attend<Geometry, TokenTile>(
        rows, q_s, p_s, stat_m, stat_l, combined, TokenTile, RowCount, kv_head, split, split_start,
        load_end, first_tile, key_blocks, pos, scale, partial_acc, partial_m, partial_l);
}

// The FP8 and NVFP4 caches (fp8: FP8 keys and values; nvfp4: NVFP4 keys and values; k8v4: FP8
// keys, NVFP4 values), launched by small_t_fp8.cu, small_t_nvfp4.cu and small_t_k8v4.cu in place
// of their tiled kernels: the same grid, quantized active splits and split ranges, the fused
// append of those kernels (one warp per new row, the standalone append's operations and rounding
// order) and partial outputs, NVFP4 values left rotated for their reducers.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool KeyFp8, bool ValueFp8>
__launch_bounds__(32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps) __global__
    void causal_attention_small_t_sm70_quantized_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
        std::uint8_t* cache_k, std::uint8_t* cache_v,
        std::conditional_t<KeyFp8, __half, std::uint8_t>* cache_k_scale,
        std::conditional_t<ValueFp8, __half, std::uint8_t>* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float attention_scale,
        float* partial_acc, float* partial_m, float* partial_l) {
    using Shape                 = sm70_attention::Shape<TokenTile * Geometry::GroupSize>;
    constexpr int D             = kCausalHeadDim;
    constexpr int Bc            = 32;
    constexpr int Groups        = Shape::kRowGroups;
    constexpr int KeyWarps      = Shape::kKeyWarps;
    constexpr int Warps         = Shape::kWarps;
    constexpr int Threads       = 32 * Warps;
    constexpr int QStride       = sm70_attention::kQStride;
    constexpr int PStride       = sm70_attention::kPStride;
    constexpr int RowCount      = TokenTile * Geometry::GroupSize;
    constexpr int PageIds       = 64;
    constexpr bool Nvfp4Append  = !KeyFp8 || !ValueFp8;
    constexpr unsigned FullMask = 0xffffffffu;

    __shared__ __align__(16) __half q_s[Groups * 8 * QStride];
    __shared__ __align__(16) __half p_s[Warps * 8 * PStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ float stat_m[Warps][8];
    __shared__ float stat_l[Warps][8];
    __shared__ float combined[KeyWarps > 1 ? Groups * 8 * D : 1];
    __shared__ float quant_scratch[CacheInput::writes_cache && Nvfp4Append ? Warps * D : 1];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    positions += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split,
                                                               TokenTile)] = 0.0f;
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = positions[0];
    const std::int32_t last_pos  = positions[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int live_end = last_pos + 1;
    const int window   = causal_small_t_tile_window(first_pos, TokenTile, logical_capacity);
    const int active_split_count =
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    int split_start         = 0;
    int split_end           = 0;
    if constexpr (TokenTile == 1) {
        const int first_owned_tile = split * logical_tiles / active_split_count;
        const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
        split_start                = first_owned_tile * Bc;
        split_end                  = min(end_owned_tile * Bc, window);
    } else {
        const int units_per_split = tile_split ? div_up(logical_tiles, active_split_count)
                                               : div_up(window, active_split_count);
        split_start               = split * units_per_split * (tile_split ? Bc : 1);
        split_end = min(split_start + units_per_split * (tile_split ? Bc : 1), window);
    }
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    // Keys at or past the live end are beyond every live column: none is loaded.
    const int load_end = split_end < live_end ? split_end : live_end;

    const int first_tile    = (split_start / Bc) * Bc;
    const int key_blocks    = div_up(split_end - first_tile, Bc);
    const int first_page    = first_tile >> kPagedKVPageShift;
    const int page_count    = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    const bool staged_pages = page_count <= PageIds;
    if (staged_pages) {
        for (int page = tid; page < page_count; page += Threads) {
            physical_pages_s[page] = block_table[first_page + page];
        }
    }
    __syncthreads();
    const auto page_at = [&](int key) {
        return staged_pages ? physical_pages_s[(key >> kPagedKVPageShift) - first_page]
                            : block_table[key >> kPagedKVPageShift];
    };

    if constexpr (CacheInput::writes_cache) {
        // One warp owns the complete D256 K row and then the complete V row, with the standalone
        // append's operations and rounding order (FP8: the represented FP16 scale feeds back
        // before the codes; NVFP4: G16 groups of the rotated row).
        for (int token = warp; token < valid_tokens; token += Warps) {
            const int position = positions[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = page_at(position);
            const int page_offset   = position & kPagedKVPageMask;
            float* row_s            = Nvfp4Append ? quant_scratch + warp * D : nullptr;
            const auto store_fp8    = [&](std::uint8_t* plane, __half* scales,
                                          const float (&values)[8]) {
                float local_absmax = 0.0f;
#pragma unroll
                for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
                const auto quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    plane[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, lane + 32 * r,
                                                            page_offset)] =
                        kv_cache_fp8_quant_code(values[r], quant.inverse_scale);
                }
                if (lane == 0) {
                    scales[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head,
                                                              page_offset)] = quant.scale;
                }
            };
            const auto store_nvfp4 = [&](std::uint8_t* plane, std::uint8_t* scales,
                                         const float (&values)[8]) {
#pragma unroll
                for (int r = 0; r < 8; ++r) { row_s[lane + 32 * r] = values[r]; }
                __syncwarp();
                if (lane < kKVCacheNvfp4Groups) {
                    const auto quantized =
                        kv_cache_nvfp4_quantize_group16(row_s + lane * kKVCacheNvfp4Group);
                    const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
                        physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset);
                    store_vec(plane + code_offset,
                              make_uint2(quantized.codes_lo, quantized.codes_hi));
                    scales[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, lane,
                                                                page_offset)] = quantized.scale;
                }
                __syncwarp();
            };
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                values[r] = __bfloat162float(
                    input.k[kv_cache_fp8_src_index<Geometry>(kv_head, lane + 32 * r, token)]);
            }
            normalized_hadamard_d256_inplace(values, lane);
            if constexpr (KeyFp8) {
                store_fp8(cache_k, cache_k_scale, values);
            } else {
                store_nvfp4(cache_k, cache_k_scale, values);
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                values[r] = __bfloat162float(
                    input.v[kv_cache_fp8_src_index<Geometry>(kv_head, lane + 32 * r, token)]);
            }
            if constexpr (ValueFp8) {
                store_fp8(cache_v, cache_v_scale, values);
            } else {
                normalized_hadamard_d256_inplace(values, lane);
                store_nvfp4(cache_v, cache_v_scale, values);
            }
        }
        __syncthreads();
    }

    // Q rotated by the keys' normalized D256 transform, in FP16 (zero past RowCount).
    for (int row = warp; row < Groups * 8; row += Warps) {
        float values[8];
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        const bool live = row < RowCount && causal_valid_q_head<Geometry>(kv_head, q_head);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            values[r] =
                live ? __bfloat162float(q[causal_q_index<Geometry>(q_head, lane + 32 * r, token)])
                     : 0.0f;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            q_s[row * QStride + lane + 32 * r] = __float2half_rn(live ? values[r] : 0.0f);
        }
    }
    __syncthreads();

    const sm70_attention::QuantRows<Geometry, KeyFp8, ValueFp8> rows{
        cache_k,     cache_v,    cache_k_scale, cache_v_scale, physical_pages_s,
        block_table, first_page, staged_pages,  kv_head};
    if constexpr (!MultiBatch && !KeyFp8 && !ValueFp8 && TokenTile == 4) {
        // These choices are measured over the complete Op on V100. Use the actual split work,
        // not the host envelope: one captured graph serves every position inside that envelope.
        if constexpr (Geometry::QHeads == 16) {
            if (key_blocks <= 2) {
                const sm70_attention::Nvfp4ScalarRows<Geometry> scalar{rows};
                sm70_attention::attend<Geometry, TokenTile>(
                    scalar, q_s, p_s, stat_m, stat_l, combined, TokenTile, RowCount, kv_head, split,
                    split_start, load_end, first_tile, key_blocks, positions, attention_scale,
                    partial_acc, partial_m, partial_l);
                return;
            }
        }
        // Fused append benefits from the shorter QK unroll. Cached-only H16 splits at 9K/12K/
        // 24K keep eight chunks in flight; above 256 active CTAs, two chunks win again.
        if (CacheInput::writes_cache || Geometry::KVHeads * active_split_count > 256) {
            const sm70_attention::QuantRows<Geometry, false, false, 2> limited{
                cache_k, cache_v, cache_k_scale, cache_v_scale, physical_pages_s,
                block_table, first_page, staged_pages, kv_head};
            sm70_attention::attend<Geometry, TokenTile>(
                limited, q_s, p_s, stat_m, stat_l, combined, TokenTile, RowCount, kv_head, split,
                split_start, load_end, first_tile, key_blocks, positions, attention_scale,
                partial_acc, partial_m, partial_l);
            return;
        }
    }
    sm70_attention::attend<Geometry, TokenTile>(rows, q_s, p_s, stat_m, stat_l, combined, TokenTile,
                                                RowCount, kv_head, split, split_start, load_end,
                                                first_tile, key_blocks, positions, attention_scale,
                                                partial_acc, partial_m, partial_l);
}

namespace detail {

// Host launch of causal_attention_small_t_sm70_quantized_kernel for the FP8 and NVFP4 small-T
// launchers.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool KeyFp8, bool ValueFp8>
void launch_small_t_sm70_quantized(const Tensor& q, CacheInput input, const Tensor& positions,
                                   float scale, const PagedKVBatchLayerView& cache,
                                   const CausalSmallTInvocation& invocation,
                                   std::int32_t logical_capacity, std::int32_t splits,
                                   Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                   cudaStream_t stream) {
    using KeyScale   = std::conditional_t<KeyFp8, __half, std::uint8_t>;
    using ValueScale = std::conditional_t<ValueFp8, __half, std::uint8_t>;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    constexpr int kBlock = 32 * sm70_attention::Shape<TokenTile * Geometry::GroupSize>::kWarps;
    causal_attention_small_t_sm70_quantized_kernel<Geometry, TokenTile, MultiBatch, Masked,
                                                   CacheInput, KeyFp8, ValueFp8>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(positions.data),
            static_cast<std::uint8_t*>(cache.k_pages.data),
            static_cast<std::uint8_t*>(cache.v_pages.data),
            static_cast<KeyScale*>(cache.k_scale_pages.data),
            static_cast<ValueScale*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
            logical_capacity, scale, static_cast<float*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace detail

} // namespace ninfer::ops
