#pragma once

// BF16-cache causal prompt kernel of a Turing (sm_75) build, included by prompt_bf16.cuh after its
// staging helper; the sm_80+ kernel there keeps this one's contract, tiles and grid. Turing has 64
// KiB of shared memory per block and no cp.async: one buffer holds K(kb), then V(kb), beside Q, and
// the kernel takes (Br + Bc) * D * 2 bytes, exactly 64 KiB. It has no BF16 MMA either, so a CTA
// sweeps its rows on FP16 MMA with Q, K and V as their FP16 roundings
// (common/bf16_attention.cuh) and sweeps them again on the BF16 emulation, an out-of-line function,
// only when a tile holds a value FP16 cannot hold: in one loop body with the fast path, the
// emulation's code cost the fast path most of its speed. Nothing is written before a sweep ends.

#include <math_constants.h>

#include "ops/common/device_limits.h"
#include "ops/softmax_attention/common/bf16_attention.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

namespace ninfer::ops {

inline constexpr int kCausalPromptBf16SmemBytes = (kCausalPromptBr + kCausalPromptBc) *
                                                  kCausalPromptHeadDim *
                                                  static_cast<int>(sizeof(__nv_bfloat16));
static_assert(kCausalPromptBf16SmemBytes <= kMaxSharedBytesPerBlock);

namespace sm75_prompt_bf16 {

// causal_prompt_stage_kv as FP16 roundings; false when one of this thread's values is beyond FP16's
// range.
template <typename Geometry>
__device__ __forceinline__ bool stage_kv_half(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                              int kv_head, int k0, int max_query_abs,
                                              int physical_page, int tid) {
    constexpr int D         = kCausalPromptHeadDim;
    constexpr int Bc        = kCausalPromptBc;
    constexpr int Threads   = kCausalPromptThreads;
    constexpr int VecPerRow = D / 8;
    const __nv_bfloat16* cache_block =
        cache + paged_kv_element_offset<kCausalPromptHeadDim, Geometry::KVHeads>(
                    physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    bool fits = true;
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key_l  = chunk >> 5;
        const int d      = (chunk & 31) << 3;
        __nv_bfloat16* p = &dst[key_l * D + causal_prompt_swz(key_l, d)];
        if ((k0 + key_l) <= max_query_abs) {
            fits = copy_bf16x8_as_half(p, &cache_block[key_l * D + d]) && fits;
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
    return fits;
}

// The CTA's Q rows into q_s, as FP16 roundings (Half; the return says whether every value the
// thread copied fits FP16) or as BF16.
template <typename Geometry, bool Half>
__device__ __forceinline__ bool stage_q(__nv_bfloat16* q_s, const __nv_bfloat16* q, int q_head,
                                        int q0, int tokens, int tid) {
    constexpr int D              = kCausalPromptHeadDim;
    constexpr int Br             = kCausalPromptBr;
    constexpr int Threads        = kCausalPromptThreads;
    constexpr int VecPerRow      = D / 8;
    constexpr int QRowStride     = D * Geometry::QHeads;
    const __nv_bfloat16* q_block = q + causal_prompt_q_index<Geometry>(q_head, 0, q0);
    bool fits                    = true;
#pragma unroll
    for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
        const int row    = chunk >> 5;
        const int d      = (chunk & 31) << 3;
        __nv_bfloat16* p = &q_s[row * D + causal_prompt_swz(row, d)];
        if (q0 + row < tokens) {
            if constexpr (Half) {
                fits = copy_bf16x8_as_half(p, &q_block[row * QRowStride + d]) && fits;
            } else {
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
    return fits;
}

// One sweep of the CTA's rows over its keys, ending with their outputs, with Q already in q_s.
// Fast: K and V tiles as FP16 roundings on FP16 MMA; false, with nothing written, once a tile holds
// a value FP16 cannot hold. Exact: tiles as BF16 on mma_bf16_rounded and mma_f16_bf16_rounded,
// whose products equal the fast sweep's for every operand FP16 holds. The tile math is the sm_80+
// kernel's (prompt_bf16.cuh).
template <typename Geometry, bool Exact>
__device__ __forceinline__ bool
sweep(__nv_bfloat16* q_s, const __nv_bfloat16* __restrict__ cache_k,
      const __nv_bfloat16* __restrict__ cache_v, const std::int32_t* block_table, int base_pos,
      float scale, __nv_bfloat16* __restrict__ out, int q_head, int q0, int tokens, int width) {
    constexpr int Br      = kCausalPromptBr;
    constexpr int Bc      = kCausalPromptBc;
    constexpr float Log2E = 1.4426950408889634074f;

    __nv_bfloat16* kv_s = q_s + Br * kCausalPromptHeadDim; // K(kb), then V(kb)
    const int tid       = static_cast<int>(threadIdx.x);
    const int lane      = tid & 31;
    const int kv_head   = q_head / Geometry::GroupSize;
    const int warp_row0 = (tid >> 5) * 16;
    const CausalPromptWarpTiles tiles =
        causal_prompt_warp_tiles(smem_addr(q_s), smem_addr(kv_s), smem_addr(kv_s), warp_row0, lane);
    const auto qk_mma = [](float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                           unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
        if constexpr (Exact) {
            mma_bf16_rounded(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        } else {
            mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        }
    };
    const auto pv_mma = [](float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                           unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
        if constexpr (Exact) {
            mma_f16_bf16_rounded(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        } else {
            mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
        }
    };

    float acc[kCausalPromptPVNt][4];
#pragma unroll
    for (int n = 0; n < kCausalPromptPVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1;
    const float scale_l2    = scale * Log2E;
    int physical_page       = block_table[0];

    // Stages one K or V tile into kv_s and publishes it; false (fast sweep) when it does not fit.
    const auto stage = [&](const __nv_bfloat16* cache, int k0) -> bool {
        if constexpr (Exact) {
            causal_prompt_stage_kv<Geometry>(kv_s, cache, kv_head, k0, max_query_abs, physical_page,
                                             tid);
            cp_wait<0>();
            __syncthreads();
            return true;
        } else {
            const bool fits = stage_kv_half<Geometry>(kv_s, cache, kv_head, k0, max_query_abs,
                                                      physical_page, tid);
            return __syncthreads_and(fits ? 1 : 0) != 0;
        }
    };

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;
        if (kb != 0) {
            physical_page = block_table[kb];
            __syncthreads(); // every warp's PV is done with V(kb - 1)
        }
        if (!stage(cache_k, k0)) { return false; }

        float score[kCausalPromptQKNt][4];
        causal_prompt_qk(score, tiles, qk_mma);
        unsigned p_frag[kCausalPromptPVKs][4];
        causal_prompt_softmax(score, p_frag, acc, m0, m1, l0, l1, q0, k0, tokens, base_pos,
                              scale_l2, warp_row0, lane);

        // V(kb) replaces K(kb) once every warp's QK is done with it.
        __syncthreads();
        if (!stage(cache_v, k0)) { return false; }
        causal_prompt_pv(acc, p_frag, tiles, pv_mma);
    }

    causal_prompt_store<Geometry>(acc, l0, l1, out, q_head, q0, tokens, width, warp_row0, lane,
                                  tid);
    return true;
}

// The BF16 emulation's sweep, out of line: stages Q as BF16 over q_s and sweeps every key block
// again, over shared memory nothing else still reads.
template <typename Geometry>
__device__ __noinline__ void
sweep_exact(__nv_bfloat16* q_s, const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
            const __nv_bfloat16* cache_v, const std::int32_t* block_table, int base_pos,
            float scale, __nv_bfloat16* out, int q_head, int q0, int tokens, int width) {
    const int tid = static_cast<int>(threadIdx.x);
    __syncthreads(); // every warp is done with the fast sweep's q_s and K/V buffer
    (void)stage_q<Geometry, false>(q_s, q, q_head, q0, tokens, tid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    (void)sweep<Geometry, true>(q_s, cache_k, cache_v, block_table, base_pos, scale, out, q_head,
                                q0, tokens, width);
}

} // namespace sm75_prompt_bf16

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
template <typename Geometry, typename Metadata>
__launch_bounds__(kCausalPromptThreads, 1) __global__
    void causal_attention_prompt_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                             const __nv_bfloat16* __restrict__ cache_k,
                                             const __nv_bfloat16* __restrict__ cache_v,
                                             Metadata metadata,
                                             const std::int32_t* __restrict__ positions,
                                             float scale, __nv_bfloat16* __restrict__ out,
                                             std::int32_t width) {
    constexpr int Br      = kCausalPromptBr;
    constexpr int Threads = kCausalPromptThreads;
    static_assert(Threads == 128);

    extern __shared__ __align__(16) __nv_bfloat16 causal_smem[];
    __nv_bfloat16* q_s = causal_smem; // [Br, D] swizzled, then the K/V buffer [Bc, D]

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int q0      = q_block * Br;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const bool q_fits = sm75_prompt_bf16::stage_q<Geometry, true>(q_s, q, q_head, q0, tokens, tid);
    if (__syncthreads_and(q_fits ? 1 : 0) != 0 &&
        sm75_prompt_bf16::sweep<Geometry, false>(q_s, cache_k, cache_v, block_table, base_pos,
                                                 scale, out, q_head, q0, tokens, width)) {
        return;
    }
    sm75_prompt_bf16::sweep_exact<Geometry>(q_s, q, cache_k, cache_v, block_table, base_pos, scale,
                                            out, q_head, q0, tokens, width);
}

} // namespace ninfer::ops
