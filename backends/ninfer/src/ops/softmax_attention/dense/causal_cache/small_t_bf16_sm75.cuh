#pragma once

// ninfer::ops - split-KV causal small-T attention, BF16 KV-cache partial kernel of the Turing shape
// set (ops/common/device_tier.h), included by small_t_bf16.cuh after the standard kernel; this one
// keeps its contract, split geometry and partial layout. Turing has no BF16 MMA, so a CTA sweeps
// its split on FP16 MMA with Q, K and V as their FP16 roundings (common/bf16_attention.cuh), and
// sweeps it again on the BF16 emulation, an out-of-line function, only when a tile holds a value
// FP16 cannot hold: in the fast sweep's loop body, the emulation's registers made it spill. The
// probabilities stay in registers (the QK accumulator is the PV A fragment) and page IDs come from
// the block table, so the K and V tiles are all of a CTA's shared memory: 32 KiB, two CTAs per SM.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "ops/softmax_attention/common/bf16_attention.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>

namespace ninfer::ops {
// Turing's forms (sm75_mma.cuh) exist in every image but sm_70's, which never runs this kernel and
// compiles it empty.
#if defined(NINFER_SM75_FORMS)
namespace sm75_small_t_bf16 {

inline constexpr int kBc = 32;
// Loads of K and V a thread issues before it stores any: Turing copies synchronously.
inline constexpr int kStageBatch = 2;

// The split one CTA sweeps (causal_small_t_bf16_split) and the cache and scale it reads.
template <typename CacheInput>
struct Split : CausalSmallTBf16Split<CacheInput> {
    const __nv_bfloat16* cache_k;
    const __nv_bfloat16* cache_v;
    float scale;
};

// The CTA's Q rows (zero padded) through qkv_s into BF16 A fragments.
template <typename Geometry, int Wc, typename CacheInput>
__device__ __forceinline__ void load_q(const Split<CacheInput>& s, __nv_bfloat16* qkv_s,
                                       unsigned (&af_q)[kCausalHeadDim / 16][4]) {
    constexpr int D       = kCausalHeadDim;
    constexpr int Br      = Wc * 16;
    constexpr int Threads = Wc * 32;
    const int tid         = static_cast<int>(threadIdx.x);
    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, s.tokens, s.kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < s.row_count && causal_valid_q_head<Geometry>(s.kv_head, q_head)) {
            value = s.q[causal_q_index<Geometry>(q_head, d, token)];
        }
        qkv_s[row * D + causal_small_t_tc_swz(row, d)] = value;
    }
    __syncthreads();
    const int lane     = tid & 31;
    const int a_mat    = lane >> 3;
    const int a_rowoff = (lane & 7) + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int arow     = (tid >> 5) * 16 + a_rowoff;
#pragma unroll
    for (int k = 0; k < D / 16; ++k) {
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + causal_small_t_tc_swz(arow, acol)]));
    }
}

// One sweep of the split, ending with its partials. Fast (Q as FP16 roundings): tiles staged as
// FP16 roundings on FP16 MMA; false, with nothing written, once a tile holds a value FP16 cannot
// hold. Exact (Q as BF16): tiles staged as BF16 on mma_bf16_rounded and mma_f16_bf16_rounded,
// whose products equal the fast sweep's for every operand FP16 holds.
template <typename Geometry, int Wc, bool Exact, typename CacheInput>
__device__ __forceinline__ bool sweep(const Split<CacheInput>& s, __nv_bfloat16* qkv_s,
                                      const unsigned (&af_q)[kCausalHeadDim / 16][4]) {
    constexpr int Bc            = kBc;
    constexpr int D             = kCausalHeadDim;
    constexpr int Threads       = Wc * 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert((Bc * (D / 8)) % (kStageBatch * Threads) == 0);

    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;
    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int row0 = (tid >> 5) * 16 + gid;
    const int row1 = row0 + 8;
    int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
    causal_small_t_tc_row_to_qt<Geometry>(row0, s.tokens, s.kv_head, q_head0, token0);
    causal_small_t_tc_row_to_qt<Geometry>(row1, s.tokens, s.kv_head, q_head1, token1);
    const int qabs0 = (row0 < s.row_count) ? s.pos[token0] : -1;
    const int qabs1 = (row1 < s.row_count) ? s.pos[token1] : -1;

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    int physical_page = s.block_table[s.first_tile >> kPagedKVPageShift];

    for (int kb = 0; kb < s.key_blocks; ++kb) {
        const int k0 = s.first_tile + kb * Bc;
        if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
            physical_page = s.block_table[k0 >> kPagedKVPageShift];
        }
        // The tile's K and V: current-step tokens from the input, tail slots zeroed.
        bool fits = true;
#pragma unroll 1
        for (int base = tid; base < Bc * (D / 8); base += kStageBatch * Threads) {
            int4 k_raw[kStageBatch];
            int4 v_raw[kStageBatch];
#pragma unroll
            for (int j = 0; j < kStageBatch; ++j) {
                const int chunk = base + j * Threads;
                const int key_l = chunk / (D / 8);
                const int d     = (chunk - key_l * (D / 8)) * 8;
                const int key   = k0 + key_l;
                k_raw[j]        = make_int4(0, 0, 0, 0);
                v_raw[j]        = make_int4(0, 0, 0, 0);
                if (key >= s.split_start && key < s.load_end) {
                    const __nv_bfloat16* k_src = s.cache_k;
                    const __nv_bfloat16* v_src = s.cache_v;
                    std::int64_t off = causal_cache_index<Geometry>(physical_page, s.kv_head, d,
                                                                    key & kPagedKVPageMask);
                    if constexpr (CacheInput::writes_cache) {
                        const int new_token = key - s.first_pos;
                        if (new_token >= 0 && new_token < s.valid_tokens && key >= s.first_pos) {
                            k_src = s.input.k;
                            v_src = s.input.v;
                            off   = kv_cache_int8_new_index<Geometry>(s.kv_head, d, new_token);
                        }
                    }
                    k_raw[j] = load_vec<int4>(k_src + off);
                    v_raw[j] = load_vec<int4>(v_src + off);
                }
            }
#pragma unroll
            for (int j = 0; j < kStageBatch; ++j) {
                const int chunk = base + j * Threads;
                const int key_l = chunk / (D / 8);
                const int d     = (chunk - key_l * (D / 8)) * 8;
                const int at    = key_l * D + causal_small_t_tc_swz(key_l, d);
                if constexpr (Exact) {
                    store_vec(&k_s[at], k_raw[j]);
                    store_vec(&v_s[at], v_raw[j]);
                } else {
                    const uint4 k_bits = *reinterpret_cast<const uint4*>(&k_raw[j]);
                    const uint4 v_bits = *reinterpret_cast<const uint4*>(&v_raw[j]);
                    fits               = store_bf16x8_as_half(&k_s[at], k_bits) && fits;
                    fits               = store_bf16x8_as_half(&v_s[at], v_bits) && fits;
                }
            }
        }
        if constexpr (Exact) {
            __syncthreads();
        } else if (__syncthreads_and(fits ? 1 : 0) == 0) {
            return false;
        }

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + causal_small_t_tc_swz(brow, bcol)]));
                if constexpr (Exact) {
                    mma_bf16_rounded(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                     af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
                } else {
                    mma_f16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                            af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
                }
            }
        }

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int key0 = k0 + nt * 8 + 2 * lid;
            const int key1 = key0 + 1;
            const bool in0 = key0 >= s.split_start && key0 < s.split_end;
            const bool in1 = key1 >= s.split_start && key1 < s.split_end;
            score[nt][0]   = (row0 < s.row_count && in0 && key0 <= qabs0) ? score[nt][0] * s.scale
                                                                          : -CUDART_INF_F;
            score[nt][1]   = (row0 < s.row_count && in1 && key1 <= qabs0) ? score[nt][1] * s.scale
                                                                          : -CUDART_INF_F;
            score[nt][2]   = (row1 < s.row_count && in0 && key0 <= qabs1) ? score[nt][2] * s.scale
                                                                          : -CUDART_INF_F;
            score[nt][3]   = (row1 < s.row_count && in1 && key1 <= qabs1) ? score[nt][3] * s.scale
                                                                          : -CUDART_INF_F;
            bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        // P as the PV A fragment: n-tile nt holds keys 8 nt to 8 nt + 7, the low (even nt) or
        // high (odd nt) half of contraction step nt / 2.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int slot            = (nt & 1) * 2;
            p_frag[nt >> 1][slot]     = pack_attention_p(p00, p01);
            p_frag[nt >> 1][slot + 1] = pack_attention_p(p10, p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
                if constexpr (Exact) {
                    mma_f16_bf16_rounded(acc[n][0], acc[n][1], acc[n][2], acc[n][3], p_frag[k][0],
                                         p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[0], vf[1]);
                } else {
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], p_frag[k][0], p_frag[k][1],
                            p_frag[k][2], p_frag[k][3], vf[0], vf[1]);
                }
            }
        }
        __syncthreads();
    }

    causal_small_t_bf16_store<Geometry>(s, acc, m0, m1, l0, l1, (tid >> 5) * 16, lane);
    return true;
}

// The BF16 emulation's sweep, out of line: reloads Q as BF16 and sweeps the split from its first
// tile, over shared memory nothing else still reads.
template <typename Geometry, int Wc, typename CacheInput>
__device__ __noinline__ void sweep_exact(Split<CacheInput> s, __nv_bfloat16* qkv_s) {
    unsigned af_q[kCausalHeadDim / 16][4];
    load_q<Geometry, Wc>(s, qkv_s, af_q);
    __syncthreads(); // every warp holds its Q fragments before the K and V tiles overwrite qkv_s
    sweep<Geometry, Wc, true>(s, qkv_s, af_q);
}

} // namespace sm75_small_t_bf16
#endif

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void causal_attention_small_t_tc_partial_bf16_turing_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, __nv_bfloat16* cache_k,
    __nv_bfloat16* cache_v, const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    float* partial_acc, float* partial_m, float* partial_l) {
    if constexpr (!shape_set_in_image(true)) { return; } // not this image's set
#if defined(NINFER_SM75_FORMS)
    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 48);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc      = WarpsPerCta;
    constexpr int Br      = Wc * 16;
    constexpr int Bc      = sm75_small_t_bf16::kBc;
    constexpr int D       = kCausalHeadDim;
    constexpr int Threads = Wc * 32;
    constexpr int QKKs    = D / 16;
    constexpr int QkvRows = 2 * Bc;

    static_assert(QkvRows >= Br && QkvRows * D * 2 == 32 * 1024);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];

    causal_small_t_bf16_split<Geometry, TokenTile, Br, Bc, Threads, MultiBatch, Masked>(
        q, input, pos, cache_k, cache_v, block_tables, valid_columns, table_rows, table_stride,
        tokens, full_width, column_begin, logical_capacity, partial_acc, partial_m, partial_l,
        [&](const CausalSmallTBf16Split<CacheInput>& split) {
            const sm75_small_t_bf16::Split<CacheInput> s{split, cache_k, cache_v, scale};

            // Q keeps its fragments for the fast sweep (the K and V tiles overwrite qkv_s).
            unsigned af_q[QKKs][4];
            sm75_small_t_bf16::load_q<Geometry, Wc>(s, qkv_s, af_q);
            bool q_fits = true;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    q_fits = q_fits && sm75::bf16x2_fits_fp16(af_q[k][i]);
                }
            }
            if (__syncthreads_and(q_fits ? 1 : 0) != 0) {
#pragma unroll
                for (int k = 0; k < QKKs; ++k) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) { af_q[k][i] = sm75::bf16x2_to_half2(af_q[k][i]); }
                }
                if (sm75_small_t_bf16::sweep<Geometry, Wc, false>(s, qkv_s, af_q)) { return; }
            }
            sm75_small_t_bf16::sweep_exact<Geometry, Wc>(s, qkv_s);
        });
#endif
}

} // namespace ninfer::ops
