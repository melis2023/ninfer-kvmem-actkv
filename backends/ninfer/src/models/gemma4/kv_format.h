#pragma once

// Gemma 4's KV-cache storage (--kv-dtype). A cache row is one KV head of one position, in blocks of
// 256 dimensions (one for a sliding-window block, two for a full-attention block), each block's
// codes and scales in two planes. The codings are the engine's own D256 codecs
// (ops/kv_cache/*_codec.cuh), with their bytes and scale boundaries:
//
//   Int8G64   signed int8, one FP16 scale per 64 dimensions, FP16-RNE(absmax / 127)
//   Int4G32   two signed 4-bit codes per byte, one FP16 scale per 32, FP16-RNE(absmax / 7)
//   Fp8Row    E4M3FN codes, one FP16 scale per 256
//   Nvfp4G16  two E2M1 codes per byte, one E4M3 scale per 16
//   Lloyd4    4-bit Lloyd-Max indices with a G64 FP16 least-squares scale (rk4v4 keys)
//   Int4E8    4-bit codes of the nearest E8 point, G64 FP16 scale (rk4v4-e8 keys)
//   RootE8    two bytes per 8 dimensions: E8 root, radius and residual axis, G64 FP16 scale
//             (rk2v4-e8 keys)
//
// A rotated plane stores the normalized Sylvester-Hadamard transform of each block, as the engine's
// formats store keys (and NVFP4 values): the attention turns its queries (and its output, for
// rotated values) instead of every row.
//
// A sliding-window block keeps K and V in the format's own codings. A full-attention block keeps one
// plane, its values, and derives each key from it (RoPE(v * k_norm)): the per-position rotation
// rules out a Hadamard-rotated plane, so the plane takes an unrotated coding at least as fine as the
// format's keys -- INT8 for int8 and rk8v4, FP8 for fp8 and k8v4, NVFP4 for nvfp4, and the packed
// int4 value coding for the 4-bit and 2-bit key formats (rk4v4, rk4v4-e8, rk2v4-e8).

#include "ninfer/kv_cache_storage.h"

#include <cstdint>

// The sizes below are also compile-time constants of the CUDA kernels.
#if defined(__CUDACC__)
#    define NINFER_GEMMA4_KV_HOST_DEVICE __host__ __device__
#else
#    define NINFER_GEMMA4_KV_HOST_DEVICE
#endif

namespace ninfer::models::gemma4 {

enum class KvCoding : std::uint8_t { F16, Int8G64, Int4G32, Fp8Row, Nvfp4G16, Lloyd4, Int4E8, RootE8 };

struct KvPlane {
    KvCoding coding = KvCoding::F16;
    bool rotated    = false;
};

struct KvFormat {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    KvPlane key, value; // sliding-window blocks
    KvPlane full;       // the full-attention blocks' one plane
    [[nodiscard]] bool quantized() const noexcept { return storage != KvCacheStorage::BFloat16; }
};

[[nodiscard]] KvFormat kv_format(KvCacheStorage storage);

inline constexpr int kKvBlockDims = 256;

// Bytes of one block's codes and of its scales.
[[nodiscard]] NINFER_GEMMA4_KV_HOST_DEVICE constexpr int kv_code_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 512;
    case KvCoding::Int8G64:
    case KvCoding::Fp8Row: return 256;
    case KvCoding::Int4G32:
    case KvCoding::Nvfp4G16:
    case KvCoding::Lloyd4:
    case KvCoding::Int4E8: return 128;
    case KvCoding::RootE8: return 64;
    }
    return 0;
}

[[nodiscard]] NINFER_GEMMA4_KV_HOST_DEVICE constexpr int kv_scale_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 0;
    case KvCoding::Int8G64:
    case KvCoding::Lloyd4:
    case KvCoding::Int4E8:
    case KvCoding::RootE8: return 4 * 2;
    case KvCoding::Int4G32: return 8 * 2;
    case KvCoding::Fp8Row: return 2;
    case KvCoding::Nvfp4G16: return 16;
    }
    return 0;
}

} // namespace ninfer::models::gemma4
