#pragma once

// Wide products on Tesla V100 (sm_70). Volta's tensor cores multiply FP16 operands only: there are
// no BF16 and no integer ones, so the A16 and A8 prefill kernels written for Ampere run there on
// emulated warp primitives at a fraction of the card's rate. A wide product instead converts its
// BF16 activations to FP16 once, dequantizes the weight to FP16 a slab of rows at a time, and hands
// each slab to a cuBLAS FP16 GEMM with an FP32 accumulator (the card's full tensor rate, 125
// TFLOP/s on an SXM2 part); an epilogue then stores the FP32 slab. The dequantization is
// weight-sized and the GEMM token-sized, so the route pays from a width on, below which the
// format's own kernels stay. llama.cpp makes the same choice on Volta (cuBLAS above 64 columns).
//
// FP16 holds every BF16 activation of magnitude below 65504 with at least BF16's precision; larger
// values saturate at +-65504 in the FP16 operands. The routes then add what saturation took: the
// conversion flags each column (or chunk of one) holding such a value, and for those the product of
// the FP16 weights with x minus its saturated value is added in FP32 (saturated_excess). A product
// therefore keeps BF16's range; values below FP16's normal range keep FP16's absolute precision
// (2^-25).

#include "ops/common/device_tier.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::sm70 {

#if defined(__CUDACC__)
// Two BF16 values in one word (lower k in the low half) as two FP16 values in one word, each
// saturated to +-65504.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned word) {
    const float lo  = fminf(fmaxf(__uint_as_float(word << 16), -65504.0f), 65504.0f);
    const float hi  = fminf(fmaxf(__uint_as_float(word & 0xffff0000u), -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Whether either BF16 half of `word` has a magnitude of 2^16 or more (infinities and NaNs too),
// which bf16x2_to_f16x2 saturates. Below 2^16 a BF16 value is at most 65280. With the signs and
// mantissas masked off, adding 113 to an exponent field of 143 or more carries into its sign bit.
__device__ __forceinline__ bool bf16x2_beyond_f16(unsigned word) {
    return (((word & 0x7f807f80u) + 0x38803880u) & 0x80008000u) != 0u;
}

// What bf16x2_to_f16x2's saturation takes from one BF16 value: zero unless it is beyond FP16.
__device__ __forceinline__ float saturated_excess(float x) {
    return x - fminf(fmaxf(x, -65504.0f), 65504.0f);
}

// sum_e w[e] * saturated_excess(x[e]) over a row's FP16 weights and a column's BF16 values.
__device__ __forceinline__ float saturated_part(const __half* w, const __nv_bfloat16* x,
                                                std::int32_t k) {
    float sum = 0.0f;
    for (std::int32_t e = 0; e < k; e += 2) {
        const unsigned word = *reinterpret_cast<const unsigned*>(x + e);
        if (!bf16x2_beyond_f16(word)) { continue; }
        const float2 weights = __half22float2(*reinterpret_cast<const __half2*>(w + e));
        sum = fmaf(weights.x, saturated_excess(__uint_as_float(word << 16)), sum);
        sum = fmaf(weights.y, saturated_excess(__uint_as_float(word & 0xffff0000u)), sum);
    }
    return sum;
}
#endif

// Whether the calling thread's current device runs Volta's code (the route exists only there).
[[nodiscard]] inline bool enabled() { return current_device_volta(); }

// The narrowest product the route takes: 65 columns, or NINFER_SM70_CUBLAS_MIN. Decode and verify
// batches stay below it on the formats' own kernels, whose launches CUDA graphs capture.
inline constexpr std::int32_t kCublasMinTokens = 65;
[[nodiscard]] std::int32_t cublas_min_tokens();

// Whether a product of `tokens` columns takes the route on the current device.
[[nodiscard]] inline bool takes(std::int32_t tokens) {
    return enabled() && tokens >= cublas_min_tokens();
}

// A workspace query over [min_tokens, max_tokens] for an Op whose widths from the route's gate up
// take the route: `routes(lo, hi)` sizes the Op's own routes over the widths below the gate, and
// `route_bytes` is the route's need at max_tokens (zero when the route does not apply). Exact for a
// single width, as the Op contracts require.
template <class Routes>
[[nodiscard]] std::size_t split_capacity(std::int32_t min_tokens, std::int32_t max_tokens,
                                         std::size_t route_bytes, Routes&& routes) {
    if (route_bytes == 0 || !takes(max_tokens)) { return routes(min_tokens, max_tokens); }
    const std::int32_t gate = cublas_min_tokens();
    std::size_t bytes       = route_bytes;
    if (min_tokens < gate) {
        const std::size_t below = routes(min_tokens, gate - 1);
        bytes                   = below > bytes ? below : bytes;
    }
    return bytes;
}

// The FP16 weight rows one GEMM covers: a multiple of 128 rows within kWeightSlabBytes, or every
// row of a smaller matrix.
inline constexpr std::size_t kWeightSlabBytes = std::size_t{32} << 20;
[[nodiscard]] std::int32_t slab_rows(std::int32_t rows, std::int32_t k);

// Workspace of one product: the FP16 activation plane and its column flags, then the FP16 weight
// slab and its FP32 output plane. Each part is rounded to 256 bytes.
[[nodiscard]] std::size_t activation_bytes(std::int32_t k, std::int32_t tokens);
[[nodiscard]] std::size_t slab_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens);

// out[c * k + i] = x[c * k + i] in FP16 (saturating) for `tokens` columns of width k, and
// beyond_columns(out, k, tokens)[c] nonzero where column c holds a value beyond FP16.
void convert_activation(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens, __half* out,
                        cudaStream_t stream);
[[nodiscard]] const std::uint32_t* beyond_columns(const __half* plane, std::int32_t k,
                                                  std::int32_t tokens);

// What a wide product adds back for saturated activations (sm70_gemm.cuh): the BF16 columns the
// FP16 plane was converted from and its column flags.
struct Saturation {
    const __nv_bfloat16* x        = nullptr;
    const std::uint32_t* columns = nullptr;
    std::int32_t k                = 0;
};

// out[c * ldc + r] = sum_i w[r * k + i] * x[c * k + i] in FP32 through cuBLAS on `stream`.
void gemm(const __half* w, std::int32_t rows, std::int32_t k, const __half* x, std::int32_t tokens,
          float* out, std::int64_t ldc, cudaStream_t stream);

} // namespace ninfer::ops::detail::sm70
