#pragma once

// Physical KV-cache storage formats (--kv-dtype). Kernels, the paged KV store and the Engine options
// share it; it depends on nothing else in the public interface.

#include <cstdint>

namespace ninfer {

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane over a group-32
    // scale. Ported onto the kv_cache_append Op that owns KV quantization, and opt-in through
    // --kv-dtype rk8v4. See docs/cli.md, "Context and memory".
    RotatedInt8KeyInt4ValueGroup64,
    // NVFP4 e2m1, group-16 scale, both K and V planes packed two codes per byte.
    Nvfp4Group16,
    // K8V4: FP8 E4M3 key plane (same coding as Fp8E4M3Row256) paired with an NVFP4 value plane.
    Fp8KeyNvfp4Value,
    // rk4v4 (this fork only): rotated keys as 4-bit indices into a Lloyd-Max codebook with a
    // group-64 FP16 scale, paired with rk8v4's packed signed int4 group-32 value plane. Opt-in
    // through --kv-dtype rk4v4.
    RotatedLloyd4KeyInt4Value,
    // rk4v4-e8: rotated keys snapped per 8-dimension block to the E8 lattice and stored as packed
    // signed int4 over a group-64 scale, paired with the rk8v4 packed int4 value plane. Opt-in
    // through --kv-dtype rk4v4-e8. Appended last so the existing values keep their numbering.
    RotatedInt4KeyInt4ValueE8,
    // rk2v4-e8: rotated keys stored per 8-dimension block as the nearest E8 root, a log-radius and
    // a residual axis (two bytes per block) over a group-64 scale, paired with the rk8v4 packed
    // int4 value plane. Opt-in through --kv-dtype rk2v4-e8.
    RotatedE8RootKeyInt4Value,
};

} // namespace ninfer
