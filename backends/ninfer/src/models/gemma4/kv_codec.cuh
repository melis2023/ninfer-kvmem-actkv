#pragma once

// Device side of Gemma 4's quantized KV planes (kv_format.h): reading eight consecutive dimensions of
// a row as F16 (the decode attention's tile loads), and a warp encoding or decoding one 256-wide
// block. The codings are the engine's D256 codecs; only the addressing is Gemma's own: a row's codes
// and scales are contiguous, block after block, and chunk c (dimensions [8c, 8c + 8)) of a row has
// its codes at byte c * bytes-per-chunk and its scale at index c / chunks-per-group of the row's
// scales, whatever block it is in.

#include "models/gemma4/kv_format.h"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::models::gemma4::kv {

// Eight F16 values as one 16-byte word.
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
    uint4 w;
    __half2* h = reinterpret_cast<__half2*>(&w);
#pragma unroll
    for (int j = 0; j < 4; ++j) { h[j] = __floats2half2_rn(f[2 * j], f[2 * j + 1]); }
    return w;
}

__device__ __forceinline__ void int8x8_to_float(uint2 raw, float s, float (&f)[8]) {
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { f[j] = float(c[j]) * s; }
}

// Signed nibbles, dimension order (low nibble first).
__device__ __forceinline__ void int4x8_to_float(unsigned raw, float s, float (&f)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int nibble = int((raw >> (4 * j)) & 0xfu);
        f[j]             = float((nibble ^ 8) - 8) * s;
    }
}

// Per coding: Raw, the bytes of one chunk; Scale, the chunk's group scale; load(), the chunk's
// bytes and scale from its row; decode(), the eight values.
template <KvCoding C>
struct Chunk;

template <>
struct Chunk<KvCoding::Int8G64> {
    using Raw = uint2;
    using Scale = __half;
    static constexpr int kBytes = 8, kPerGroup = 8;
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) { int8x8_to_float(r, __half2float(s), f); }
};

template <>
struct Chunk<KvCoding::Int4G32> {
    using Raw = unsigned;
    using Scale = __half;
    static constexpr int kBytes = 4, kPerGroup = 4;
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) { int4x8_to_float(r, __half2float(s), f); }
};

template <>
struct Chunk<KvCoding::Int4E8> : Chunk<KvCoding::Int4G32> {
    static constexpr int kPerGroup = 8;
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
};

template <>
struct Chunk<KvCoding::Lloyd4> {
    using Raw = unsigned;
    using Scale = __half;
    static constexpr int kBytes = 4, kPerGroup = 8;
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const uint2 codes = make_uint2(ops::kv_cache_lloyd4_expand4(r & 0xffffu),
                                       ops::kv_cache_lloyd4_expand4(r >> 16));
        int8x8_to_float(codes, __half2float(s), f);
    }
};

template <>
struct Chunk<KvCoding::RootE8> {
    using Raw = unsigned short;
    using Scale = __half;
    static constexpr int kBytes = 2, kPerGroup = 8;
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned short*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        int8x8_to_float(ops::kv_cache_e8_root_decode_block(r & 0xffu, r >> 8), __half2float(s), f);
    }
};

template <>
struct Chunk<KvCoding::Fp8Row> {
    using Raw = uint2;
    using Scale = __half;
    static constexpr int kBytes = 8, kPerGroup = 32;
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const std::uint16_t* pairs = reinterpret_cast<const std::uint16_t*>(&r);
        const float sf             = __half2float(s);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = __half22float2(ops::kv_cache_fp8_code2_to_half2(pairs[j]));
            f[2 * j]       = v.x * sf;
            f[2 * j + 1]   = v.y * sf;
        }
    }
};

template <>
struct Chunk<KvCoding::Nvfp4G16> {
    using Raw = unsigned;
    using Scale = std::uint8_t;
    static constexpr int kBytes = 4, kPerGroup = 2;
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) { return scales[c / kPerGroup]; }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        // E2M1 times an E4M3 scale is exact in FP16 (nvfp4_group16_codec.cuh) and in FP32.
        __nv_fp8_e4m3 scale;
        scale.__x      = s;
        const float sf = static_cast<float>(scale);
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&r);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_fp4x2_e2m1 pair;
            pair.__x       = bytes[j];
            const float2 v = static_cast<float2>(pair);
            f[2 * j]       = v.x * sf;
            f[2 * j + 1]   = v.y * sf;
        }
    }
};

// ---- One block per warp ---------------------------------------------------------------------------
// Lane l holds dimensions l + 32 r of the block in x[r] (the layout of the engine's append kernels
// and of normalized_hadamard_d256_inplace). `scratch` is the warp's 256 floats of shared memory.

// Encodes x (already rotated for a rotated plane) into the block's codes and scales.
template <KvCoding C>
__device__ __forceinline__ void encode_block(const float (&x)[8], std::uint8_t* codes,
                                             std::uint8_t* scales, float* scratch, int lane) {
    constexpr unsigned kAll = 0xffffffffu;
    if constexpr (C == KvCoding::Int8G64) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float a = ops::warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q  = ops::kv_cache_int8_quant_params(a);
            auto* c       = reinterpret_cast<std::int8_t*>(codes);
            c[64 * g + lane]      = ops::kv_cache_int8_quant_code(x[2 * g], q.inverse_scale);
            c[64 * g + 32 + lane] = ops::kv_cache_int8_quant_code(x[2 * g + 1], q.inverse_scale);
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Int4G32) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const auto q         = ops::kv_cache_int4_quant_params(ops::warp_max(fabsf(x[r]), kAll));
            const std::int8_t c  = ops::kv_cache_int4_quant_code(x[r], q.inverse_scale);
            const int partner    = __shfl_xor_sync(kAll, int(c), 1);
            if ((lane & 1) == 0) {
                codes[16 * r + (lane >> 1)] = ops::kv_cache_int4_pack(c, std::int8_t(partner));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[r] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Lloyd4) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const auto e       = ops::kv_cache_lloyd4_encode_group(x[2 * g], x[2 * g + 1]);
            const int partner0 = __shfl_xor_sync(kAll, e.index0, 1);
            const int partner1 = __shfl_xor_sync(kAll, e.index1, 1);
            if ((lane & 1) == 0) {
                codes[32 * g + (lane >> 1)]      = std::uint8_t(e.index0 | (partner0 << 4));
                codes[32 * g + 16 + (lane >> 1)] = std::uint8_t(e.index1 | (partner1 << 4));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = e.scale; }
        }
    } else if constexpr (C == KvCoding::Int4E8) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float a        = ops::warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q         = ops::kv_cache_int4_quant_params(a);
            const std::int8_t c0 = ops::kv_cache_int4_e8_key_code(x[2 * g], q.inverse_scale, lane);
            const std::int8_t c1 = ops::kv_cache_int4_e8_key_code(x[2 * g + 1], q.inverse_scale, lane);
            const int partner0   = __shfl_xor_sync(kAll, int(c0), 1);
            const int partner1   = __shfl_xor_sync(kAll, int(c1), 1);
            if ((lane & 1) == 0) {
                codes[32 * g + (lane >> 1)]      = ops::kv_cache_int4_pack(c0, std::int8_t(partner0));
                codes[32 * g + 16 + (lane >> 1)] = ops::kv_cache_int4_pack(c1, std::int8_t(partner1));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::RootE8) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float a = ops::warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q  = ops::kv_cache_int4_quant_params(a);
            ops::kv_cache_e8_root_store_key_group(codes, 0, g, x[2 * g], x[2 * g + 1],
                                                  __half2float(q.scale), lane);
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Fp8Row) {
        float a = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) { a = fmaxf(a, fabsf(x[r])); }
        const auto q = ops::kv_cache_fp8_quant_params(ops::warp_max(a, kAll));
#pragma unroll
        for (int r = 0; r < 8; ++r) { codes[32 * r + lane] = ops::kv_cache_fp8_quant_code(x[r], q.inverse_scale); }
        if (lane == 0) { reinterpret_cast<__half*>(scales)[0] = q.scale; }
    } else if constexpr (C == KvCoding::Nvfp4G16) {
#pragma unroll
        for (int r = 0; r < 8; ++r) { scratch[32 * r + lane] = x[r]; }
        __syncwarp();
        if (lane < 16) {
            const auto q = ops::kv_cache_nvfp4_quantize_group16(scratch + 16 * lane);
            *reinterpret_cast<uint2*>(codes + 8 * lane) = make_uint2(q.codes_lo, q.codes_hi);
            scales[lane] = q.scale;
        }
        __syncwarp();
    }
}

// The block's represented values, in the encoder's layout (rotated still, for a rotated plane).
template <KvCoding C>
__device__ __forceinline__ void decode_block(const std::uint8_t* codes, const std::uint8_t* scales,
                                             float (&x)[8], float* scratch, int lane) {
    using K = Chunk<C>;
    float f[8];
    K::decode(K::load(codes, lane), K::scale(scales, lane), f); // chunk `lane`: dims [8l, 8l + 8)
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 8; ++j) { scratch[8 * lane + j] = f[j]; }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < 8; ++r) { x[r] = scratch[32 * r + lane]; }
    __syncwarp();
}

} // namespace ninfer::models::gemma4::kv
