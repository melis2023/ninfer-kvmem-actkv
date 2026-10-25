#pragma once

// MMA operands of the BF16 attention kernels on Turing. sm_80 and newer multiply BF16 Q, K and V
// and BF16-rounded probabilities on BF16 m16n8k16. Turing has no BF16 MMA: mma_bf16's emulation
// converts every fragment and votes on its exactness at every call, and runs a product on FP16
// only where every operand is an FP16 value. Two things make that slow in attention:
//
// - Softmax probabilities span far more binades than FP16's, so as BF16 operands nearly every PV
//   tile of a long context fell back to FP32. A Turing build rounds the probabilities to FP16
//   (three more significant bits from 2^-14 to one, FP16 subnormals below, zero under 2^-24, as the
//   INT8-family kernels round them on every architecture): pack_attention_p.
// - The per-call conversions and votes cost more than the MMAs, and an exactness vote fails for a
//   whole tile over one value under 2^-17. The causal-cache kernels multiply Q, K and V as their
//   FP16 roundings instead: the BF16 value itself from 2^-17 up, an FP16 subnormal or zero below,
//   a change of at most 2^-25, so a product moves by at most 2^-25 times its other operand (the
//   INT8-family kernels hold V as FP16 too). They stage a tile as FP16 when every value is below
//   2^16 (store_bf16x8_as_half) and run it on native FP16 MMA. A tile with a larger value, which
//   FP16 cannot hold, stages as BF16 and multiplies on mma_bf16_rounded and mma_f16_bf16_rounded,
//   which take the same rounding of every value below 2^16 and keep the larger ones: the outputs
//   whose row and column fit run on the same FP16 MMA, the others in FP32. An output's bits
//   therefore depend on its own row and column alone, whichever way its tile was staged.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

#if defined(NINFER_SM75)
inline constexpr bool kAttentionPvHalfProbabilities = true;
#else
inline constexpr bool kAttentionPvHalfProbabilities = false;
#endif

// Two probabilities as the A operand of attention_pv_mma.
__device__ __forceinline__ std::uint32_t pack_attention_p(float lo, float hi) {
    if constexpr (kAttentionPvHalfProbabilities) {
        return pack_f16x2(lo, hi);
    } else {
        return pack_bf16x2(lo, hi);
    }
}

#if defined(NINFER_SM75)
namespace sm75 {

// Both BF16 halves are below 2^16 in magnitude (a biased exponent of at most 142; zero and BF16
// subnormals included), so their FP16 roundings are finite.
__device__ __forceinline__ bool bf16x2_fits_fp16(unsigned x) {
    return ((x >> 7) & 0xffu) <= 142u && ((x >> 23) & 0xffu) <= 142u;
}

// Each BF16 half below 2^16 as its FP16 rounding (still a BF16 value: at most eight significant
// bits), a larger one unchanged.
__device__ __forceinline__ unsigned bf16x2_fp16_rounded(unsigned x) {
    const auto rounded = [](unsigned bits) {
        if (((bits >> 7) & 0xffu) > 142u) { return bits; }
        return __float_as_uint(__half2float(__float2half_rn(__uint_as_float(bits << 16)))) >> 16;
    };
    return rounded(x & 0xffffu) | (rounded(x >> 16) << 16);
}

} // namespace sm75

// m16n8k16 with FP16 A and BF16 B. B runs on FP16 m16n8k8 where it is exactly an FP16 value; the
// outputs of a column that holds another value take the FP32 product of the same operands, so an
// output's bits depend on its own column alone (sm75::take_inexact_outputs).
__device__ __forceinline__ void mma_f16_bf16(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned cols =
        __ballot_sync(kAll, sm75::bf16x2_exact_in_fp16(b0) && sm75::bf16x2_exact_in_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, a0, a1, sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, a2, a3, sm75::bf16x2_to_half2(b1));
    if (cols != kAll) {
        sm75::take_inexact_outputs(
            c0, c1, c2, c3,
            sm75::mma_k16_fp32<pre_sm80::F16Operands, pre_sm80::Bf16Operands>(acc, a0, a1, a2, a3,
                                                                              b0, b1),
            kAll, kAll, cols);
    }
}

// m16n8k16 of BF16 fragments as the causal-cache kernels multiply Q and K: every value below 2^16
// as its FP16 rounding, on FP16 m16n8k8 for the outputs whose row of A and column of B fit FP16
// and in FP32 for the others (mma_bf16's per-output routing).
__device__ __forceinline__ void mma_bf16_rounded(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned rows_lo =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(a0) && sm75::bf16x2_fits_fp16(a2));
    const unsigned rows_hi =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(a1) && sm75::bf16x2_fits_fp16(a3));
    const unsigned cols =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(b0) && sm75::bf16x2_fits_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::bf16x2_to_half2(a0), sm75::bf16x2_to_half2(a1),
                     sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, sm75::bf16x2_to_half2(a2), sm75::bf16x2_to_half2(a3),
                     sm75::bf16x2_to_half2(b1));
    if ((rows_lo & rows_hi & cols) != kAll) {
        const float4 exact = sm75::mma_k16_fp32<pre_sm80::Bf16Operands, pre_sm80::Bf16Operands>(
            acc, sm75::bf16x2_fp16_rounded(a0), sm75::bf16x2_fp16_rounded(a1),
            sm75::bf16x2_fp16_rounded(a2), sm75::bf16x2_fp16_rounded(a3),
            sm75::bf16x2_fp16_rounded(b0), sm75::bf16x2_fp16_rounded(b1));
        sm75::take_inexact_outputs(c0, c1, c2, c3, exact, rows_lo, rows_hi, cols);
    }
}

// mma_f16_bf16 as the causal-cache kernels multiply P and V: every value of B below 2^16 as its
// FP16 rounding, on FP16 m16n8k8 for the columns that fit FP16 and in FP32 for the others.
__device__ __forceinline__ void mma_f16_bf16_rounded(float& c0, float& c1, float& c2, float& c3,
                                                     unsigned a0, unsigned a1, unsigned a2,
                                                     unsigned a3, unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned cols =
        __ballot_sync(kAll, sm75::bf16x2_fits_fp16(b0) && sm75::bf16x2_fits_fp16(b1));
    const float4 acc = make_float4(c0, c1, c2, c3);
    sm75::mma_f16_k8(c0, c1, c2, c3, a0, a1, sm75::bf16x2_to_half2(b0));
    sm75::mma_f16_k8(c0, c1, c2, c3, a2, a3, sm75::bf16x2_to_half2(b1));
    if (cols != kAll) {
        const float4 exact = sm75::mma_k16_fp32<pre_sm80::F16Operands, pre_sm80::Bf16Operands>(
            acc, a0, a1, a2, a3, sm75::bf16x2_fp16_rounded(b0), sm75::bf16x2_fp16_rounded(b1));
        sm75::take_inexact_outputs(c0, c1, c2, c3, exact, kAll, kAll, cols);
    }
}

// Stores eight BF16 values (16 bytes) as their FP16 roundings and says whether every one was
// within FP16's range; when one was not, the destination holds values the caller must replace.
__device__ __forceinline__ bool store_bf16x8_as_half(void* dst, uint4 raw) {
    const bool fits = sm75::bf16x2_fits_fp16(raw.x) && sm75::bf16x2_fits_fp16(raw.y) &&
                      sm75::bf16x2_fits_fp16(raw.z) && sm75::bf16x2_fits_fp16(raw.w);
    store_vec(dst, make_uint4(sm75::bf16x2_to_half2(raw.x), sm75::bf16x2_to_half2(raw.y),
                              sm75::bf16x2_to_half2(raw.z), sm75::bf16x2_to_half2(raw.w)));
    return fits;
}

__device__ __forceinline__ bool copy_bf16x8_as_half(void* dst, const void* src) {
    return store_bf16x8_as_half(dst, load_vec<uint4>(src));
}
#endif

// O += P V for one m16n8k16 fragment: probabilities from pack_attention_p, BF16 values.
__device__ __forceinline__ void attention_pv_mma(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned p0, unsigned p1, unsigned p2, unsigned p3,
                                                 unsigned v0, unsigned v1) {
#if defined(NINFER_SM75)
    mma_f16_bf16(c0, c1, c2, c3, p0, p1, p2, p3, v0, v1);
#else
    mma_bf16(c0, c1, c2, c3, p0, p1, p2, p3, v0, v1);
#endif
}

} // namespace ninfer::ops
