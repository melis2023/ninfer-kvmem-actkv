#pragma once

// ninfer::ops - split-KV causal small-T attention over an INT8 G64 cache, pipelined tier
// ("8x2x32p" of the attn_i8_small route keys).
//
// The arithmetic of causal_attention_small_t_i8_tiled_kernel (small_t_i8.cuh) with its INT8 keys
// and values: the same fused append, the same rotated INT8 queries, native INT8 QK per G64 group
// with FP32 scales, the same online softmax over 32-key tiles with FP16 P, and PV on FP16 Tensor
// Cores in the same key order, under the same split partition. What differs is the staging:
//
//   * V codes stay INT8 in shared memory. Each of the eight warps owns a 32-dimension slice of the
//     output for every row tile and widens its own V codes in registers (ldmatrix.trans on byte
//     pairs, value_fragments.cuh), exactly as the FP16 staging tile held them, so every code is
//     widened once and outputs and FP32 partials equal the staged kernel's bit for bit.
//   * Without the FP16 V tile, two INT8 K/V stages fit beside two CTAs per SM: the next tile's
//     codes stream by cp.async through the current tile's QK and PV, and a split that holds none
//     of the call's appended positions issues its first tile before the append and the query
//     preparation.
//
// Ported from Wallawalla/ninfer ee2e1d223 (grouped_pipelined.cuh, Ian Ranson), adapted to this
// line's partition, append and tile helpers.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"
#include "ops/softmax_attention/dense/causal_cache/value_fragments.cuh"

#include <cstdint>

namespace ninfer::ops {

struct CausalSmallTI8PipelinedShape {
    static constexpr int kWarps   = 8;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kKeys    = 32;
    // K codes, V codes, then the G64 key and value scales of one tile.
    static constexpr int kStageBytes =
        2 * kKeys * kCausalHeadDim + 2 * kKeys * kKVCacheInt8Groups * 2;
    static constexpr int kArenaBytes = 2 * kStageBytes;
};

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput,
          bool PvF16>
__launch_bounds__(CausalSmallTI8PipelinedShape::kThreads, 2) __global__
    void causal_attention_small_t_i8_pipelined_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, std::int32_t wave_splits,
        float scale, float* partial_acc, float* partial_m, float* partial_l) {
#if !defined(NINFER_SM70_DEVICE)
    using Shape                 = CausalSmallTI8PipelinedShape;
    constexpr int Wc            = Shape::kWarps;
    constexpr int RowCount      = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles      = (RowCount + 15) / 16;
    constexpr int Br            = RowTiles * 16;
    constexpr int Bc            = Shape::kKeys;
    constexpr int D             = kCausalHeadDim;
    constexpr int DB16          = D / 2;
    constexpr int Threads       = Shape::kThreads;
    constexpr int Groups        = kKVCacheInt8Groups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKKs          = D / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int StageBytes    = Shape::kStageBytes;
    constexpr int PageIds       = 64;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(TokenTile >= 1 && RowCount <= 48);
    static_assert(RowTiles >= 1 && RowTiles <= 3 && RowTiles <= Wc);
    static_assert(D / Wc == 32);
    static_assert(QKKs == Groups * GroupKc);
    static_assert(Bc <= kPagedKVPageSize && kPagedKVPageSize % Bc == 0);
    // The append's FP32 exchange and the query scales live in stage one before its first tile.
    static_assert(TokenTile * D * 4 <= StageBytes && RowCount * Groups * 4 <= StageBytes);

    __shared__ __align__(16) std::int8_t q_s[Br * D];
    extern __shared__ __align__(16) std::int8_t arena[];
    __shared__ __align__(16) __half p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ std::int32_t physical_pages_s[PageIds];
    std::int8_t* q_i8     = q_s;
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_i8);
    float* stage1_scratch = reinterpret_cast<float*>(arena + StageBytes);
    float* q_scale_tmp    = stage1_scratch;
    const auto stage_k    = [&](int s) { return arena + s * StageBytes; };
    const auto stage_v    = [&](int s) { return arena + s * StageBytes + Bc * D; };
    const auto stage_ks   = [&](int s) {
        return reinterpret_cast<__half*>(arena + s * StageBytes + 2 * Bc * D);
    };
    const auto stage_vs = [&](int s) {
        return reinterpret_cast<__half*>(arena + s * StageBytes + 2 * Bc * D) + Bc * Groups;
    };

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
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

    // The staged kernel's partition, so both tiers split a window identically.
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

    // K keeps the staged kernel's swizzled B16-pair layout for QK; V takes the 16-byte chunk XOR
    // swizzle that ldmatrix.trans reads without bank conflicts.
    auto issue_kv_tile = [&](int stage, int tile_k0) {
        const int physical_page = page_at(tile_k0);
        std::int8_t* k_i8       = stage_k(stage);
        std::int8_t* v_i8       = stage_v(stage);
        __half* k_scale_s       = stage_ks(stage);
        __half* v_scale_s       = stage_vs(stage);
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < load_end) {
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<8>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int2(0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int2(0, 0));
            }
        }
#    pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l    = chunk / (D / 16);
            const int dc       = chunk - key_l * (D / 16);
            const int d        = dc * 16;
            const int key      = tile_k0 + key_l;
            std::int8_t* k_dst = &k_i8[key_l * D + causal_small_t_tc_swz(key_l, dc * 8) * 2];
            std::int8_t* v_dst = &v_i8[key_l * D + ((dc ^ (key_l & 7)) << 4)];
            if (key >= split_start && key < load_end) {
                const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                    physical_page, kv_head, d, key & kPagedKVPageMask);
                ninfer::ops::cp_async<16>(k_dst, &cache_k_i8[off]);
                ninfer::ops::cp_async<16>(v_dst, &cache_v_i8[off]);
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    // A split that holds none of the call's appended positions reads only settled cache rows, so
    // its first tile streams while the append and the query preparation run.
    bool owns_append = false;
    if constexpr (CacheInput::writes_cache) {
        owns_append = !(pos[valid_tokens - 1] < split_start || pos[0] >= split_end);
    }
    if (!owns_append) { issue_kv_tile(0, first_tile); }

    if constexpr (CacheInput::writes_cache) {
        // The staged kernel's append (INT8 keys and values): H256 as H4 over four H64 groups,
        // exchanged through stage one, which no tile occupies yet.
        float* k_h64_s = stage1_scratch;
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
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

        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = page_at(position);
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);

            float k_out[2];
#    pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }
            kv_cache_i8_family_store_key_group<Geometry, KvKeyCoding::Int8>(
                cache_k_i8, cache_k_scale, physical_page, kv_head, grp, page_offset, lane, k_out[0],
                k_out[1]);
            const float vv0      = __bfloat162float(input.v[src0]);
            const float vv1      = __bfloat162float(input.v[src1]);
            const float vamax_lo = warp_max(fabsf(vv0), FullMask);
            const float vamax_hi = warp_max(fabsf(vv1), FullMask);
            const auto v_quant   = kv_cache_int8_quant_params(fmaxf(vamax_lo, vamax_hi));
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            if (lane == 0) {
                cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(
                    physical_page, kv_head, grp, page_offset)] = v_quant.scale;
            }
        }
        __syncthreads();
    }
    if (owns_append) { issue_kv_tile(0, first_tile); }

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float q_values[8];
#    pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)]);
        }
        normalized_hadamard_d256_inplace(q_values, lane);

#    pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float amax      = fmaxf(fabsf(x0), fabsf(x1));
            amax            = warp_max(amax, FullMask);
            const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_small_t_store_byte_swizzled(q_i8, row, d0, DB16,
                                               kv_cache_int8_quant_code(x0, inv));
            causal_small_t_store_byte_swizzled(q_i8, row, d1, DB16,
                                               kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0[Groups];
    float q_scale_r1[Groups];
    if (warp < RowTiles) {
        const int producer_row0 = warp * 16 + gid;
#    pragma unroll
        for (int g = 0; g < Groups; ++g) {
            q_scale_r0[g] =
                producer_row0 < RowCount ? q_scale_tmp[producer_row0 * Groups + g] : 0.0f;
            q_scale_r1[g] =
                producer_row0 + 8 < RowCount ? q_scale_tmp[(producer_row0 + 8) * Groups + g] : 0.0f;
        }
    }
    // Stage one is about to take the second tile: the scales above are read first.
    __syncthreads();

    // acc[row tile][16-dimension block][even/odd n8 tile]: this warp owns dimensions
    // 32 * warp .. 32 * warp + 31 of every row.
    float acc[RowTiles][2][2][4];
#    pragma unroll
    for (int t = 0; t < RowTiles; ++t) {
#    pragma unroll
        for (int b = 0; b < 2; ++b) {
#    pragma unroll
            for (int e = 0; e < 2; ++e) {
#    pragma unroll
                for (int i = 0; i < 4; ++i) { acc[t][b][e][i] = 0.0f; }
            }
        }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;
    const int v_group = (warp * 32) / kKVCacheInt8Group;

#    pragma unroll 1
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0    = first_tile + kb * Bc;
        const int stage = kb & 1;
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) { issue_kv_tile(stage ^ 1, k0 + Bc); }

        if (warp < RowTiles) {
            const __nv_bfloat16* k_b16  = reinterpret_cast<const __nv_bfloat16*>(stage_k(stage));
            const __half* k_scale_s     = stage_ks(stage);
            const int producer_row_base = warp * 16;
            __half* p_sw                = &p_s[producer_row_base * Bc];
            float score[QKNt][4];
#    pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#    pragma unroll
            for (int g = 0; g < Groups; ++g) {
                unsigned af[GroupKc][4];
#    pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(
                        af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                        smem_addr(
                            &q_b16[(producer_row_base + a_rowoff) * DB16 +
                                   causal_small_t_tc_swz(producer_row_base + a_rowoff, acol)]));
                }

#    pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#    pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(
                            bf[0], bf[1],
                            smem_addr(&k_b16[brow * DB16 + causal_small_t_tc_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya  = nt * 8 + 2 * lid;
                    const int keyb  = keya + 1;
                    const float ka  = __half2float(k_scale_s[keya * Groups + g]);
                    const float kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                    score[nt][0] += q_scale_r0[g] * ka * static_cast<float>(c0);
                    score[nt][1] += q_scale_r0[g] * kb2 * static_cast<float>(c1);
                    score[nt][2] += q_scale_r1[g] * ka * static_cast<float>(c2);
                    score[nt][3] += q_scale_r1[g] * kb2 * static_cast<float>(c3);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#    pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] =
                    (row0 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs0)
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (row0 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs0)
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (row1 < RowCount && key0 >= split_start && key0 < split_end && key0 <= qabs1)
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (row1 < RowCount && key1 >= split_start && key1 < split_end && key1 <= qabs1)
                        ? score[nt][3] * scale
                        : -CUDART_INF_F;
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

            float bl0 = 0.0f, bl1 = 0.0f;
#    pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
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
                p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col0)] = __float2half_rn(p00);
                p_sw[gid * Bc + causal_small_t_tc_swz32(gid, col1)] = __float2half_rn(p01);
                p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col0)] =
                    __float2half_rn(p10);
                p_sw[(gid + 8) * Bc + causal_small_t_tc_swz32(gid + 8, col1)] =
                    __float2half_rn(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            l0 = l0 * alpha0 + bl0;
            l1 = l1 * alpha1 + bl1;
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        }
        __syncthreads();

        // PV: this warp's 32 dimensions for every row tile. The even/odd n8 tiles of a
        // 16-dimension block hold dimensions 16b + 2n and 16b + 2n + 1.
        const std::int8_t* v_i8 = stage_v(stage);
        const __half* v_scale_s = stage_vs(stage);
#    pragma unroll
        for (int t = 0; t < RowTiles; ++t) {
            const float alpha0 = alpha_s[t * 16 + gid];
            const float alpha1 = alpha_s[t * 16 + gid + 8];
#    pragma unroll
            for (int b = 0; b < 2; ++b) {
#    pragma unroll
                for (int e = 0; e < 2; ++e) {
                    acc[t][b][e][0] *= alpha0;
                    acc[t][b][e][1] *= alpha0;
                    acc[t][b][e][2] *= alpha1;
                    acc[t][b][e][3] *= alpha1;
                }
            }
        }
        // PvF16: a tile's products are summed in FP16 per output fragment and folded into the
        // FP32 accumulator once, as the staged kernel does.
        unsigned tile_acc[PvF16 ? RowTiles : 1][2][2][2];
        if constexpr (PvF16) {
#    pragma unroll
            for (int t = 0; t < RowTiles; ++t) {
#    pragma unroll
                for (int b = 0; b < 2; ++b) {
#    pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        tile_acc[t][b][e][0] = 0u;
                        tile_acc[t][b][e][1] = 0u;
                    }
                }
            }
        }
#    pragma unroll
        for (int j = 0; j < PVKs; ++j) {
            const int key          = j * 16 + 2 * lid;
            const __half2 lo_scale = __halves2half2(v_scale_s[key * Groups + v_group],
                                                    v_scale_s[(key + 1) * Groups + v_group]);
            const __half2 hi_scale = __halves2half2(v_scale_s[(key + 8) * Groups + v_group],
                                                    v_scale_s[(key + 9) * Groups + v_group]);
            const unsigned lo      = load_vec<unsigned>(&lo_scale);
            const unsigned hi      = load_vec<unsigned>(&hi_scale);
            unsigned r[4];
            {
                const int vkey  = j * 16 + ((a_mat & 1) << 3) + a_rin;
                const int chunk = warp * 2 + (a_mat >> 1);
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(&v_i8[vkey * D + ((chunk ^ (vkey & 7)) << 4)]));
            }
            unsigned vf[2][2][2]; // [16-dimension block][even/odd][low/high keys]
#    pragma unroll
            for (int b = 0; b < 2; ++b) {
                causal_i8_value_pair_fragments(r[2 * b], lo, vf[b][0][0], vf[b][1][0]);
                causal_i8_value_pair_fragments(r[2 * b + 1], hi, vf[b][0][1], vf[b][1][1]);
            }
#    pragma unroll
            for (int t = 0; t < RowTiles; ++t) {
                const __half* p_consumer = &p_s[t * 16 * Bc];
                unsigned pf[4];
                const int pcol = j * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(
                        &p_consumer[a_rowoff * Bc + causal_small_t_tc_swz32(a_rowoff, pcol)]));
#    pragma unroll
                for (int b = 0; b < 2; ++b) {
#    pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        if constexpr (PvF16) {
                            mma_f16_f16acc(tile_acc[t][b][e][0], tile_acc[t][b][e][1], pf[0], pf[1],
                                           pf[2], pf[3], vf[b][e][0], vf[b][e][1]);
                        } else {
                            mma_f16(acc[t][b][e][0], acc[t][b][e][1], acc[t][b][e][2],
                                    acc[t][b][e][3], pf[0], pf[1], pf[2], pf[3], vf[b][e][0],
                                    vf[b][e][1]);
                        }
                    }
                }
            }
        }
        if constexpr (PvF16) {
#    pragma unroll
            for (int t = 0; t < RowTiles; ++t) {
#    pragma unroll
                for (int b = 0; b < 2; ++b) {
#    pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float2 top = __half22float2(
                            *reinterpret_cast<const __half2*>(&tile_acc[t][b][e][0]));
                        const float2 bottom = __half22float2(
                            *reinterpret_cast<const __half2*>(&tile_acc[t][b][e][1]));
                        acc[t][b][e][0] += top.x;
                        acc[t][b][e][1] += top.y;
                        acc[t][b][e][2] += bottom.x;
                        acc[t][b][e][3] += bottom.y;
                    }
                }
            }
        }
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

    // Lane (g, t) holds dimensions 16b + 4t .. 16b + 4t + 3 of rows g and g + 8 of each tile:
    // the even tile's pair is dimensions 4t and 4t + 2, the odd tile's 4t + 1 and 4t + 3.
#    pragma unroll
    for (int t = 0; t < RowTiles; ++t) {
        const int row0 = t * 16 + gid;
        const int row1 = row0 + 8;
#    pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int d0 = (warp * 2 + b) * 16 + 4 * lid;
            if (row0 < RowCount) {
                int q_head = 0;
                int token  = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
                const std::int64_t dst =
                    causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
                *reinterpret_cast<float2*>(&partial_acc[dst]) =
                    make_float2(acc[t][b][0][0], acc[t][b][1][0]);
                *reinterpret_cast<float2*>(&partial_acc[dst + 2]) =
                    make_float2(acc[t][b][0][1], acc[t][b][1][1]);
            }
            if (row1 < RowCount) {
                int q_head = 0;
                int token  = 0;
                causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
                const std::int64_t dst =
                    causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
                *reinterpret_cast<float2*>(&partial_acc[dst]) =
                    make_float2(acc[t][b][0][2], acc[t][b][1][2]);
                *reinterpret_cast<float2*>(&partial_acc[dst + 2]) =
                    make_float2(acc[t][b][0][3], acc[t][b][1][3]);
            }
        }
    }
#endif
}

} // namespace ninfer::ops
