#pragma once

// Gemma 4's own elementwise, normalization, routing and decode-attention kernels. Activations are
// token-major: an FP32 residual h[t][i], BF16 projection inputs x[t][k] (a [K, T] Tensor).

#include "models/gemma4/kv_format.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::gemma4::kernels {

// Rows in the GGUF vector kernels' q8_1 activation form (ggml_bridge.h's vector activation):
// int8 values [T][k] and (d, sum) half2 pairs [T][k / 32]. A kernel given one writes it instead of
// leaving a BF16 row for a separate quantization; k must be a whole number of 32-value blocks.
struct Q8Rows {
    std::int8_t* qs = nullptr;
    half2* ds       = nullptr;
};

// x[i] *= s over n values.
void scale(float* x, std::int64_t n, float s, cudaStream_t stream);

// out[t][i] = bf16(x[t][i] * rsqrt(mean_i x[t]^2 + eps) * w[i]); w null: no gamma. With q, the
// same rows also (or, with out null, only) in q8_1 form.
void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q = {});

// The Q/K/V of one attention block, after its projection: per head RMSNorm (Q and K with their
// gammas, V without), RoPE on Q and K (pair (i, i + D/2) turns by position * inv_freq[i]; a zero
// frequency leaves the pair), Q to FP32 [T][Hq][D], K and V to the F16 cache slots of their
// positions. V is the block's own rows, or with `has_v` false the raw K rows (K = V attention).
// A null k_cache stores V only: the keys are then RoPE(V * k_norm), derived where they are read
// (derive_keys, decode_attention) with the same rotations (rope_table).
struct QkvArgs {
    const __nv_bfloat16* qkv = nullptr; // [T][Hq*D + Hkv*D (+ Hkv*D)]
    int tokens = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    bool has_v = true;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    float eps           = 1e-6f;
    const std::int32_t* positions = nullptr; // [T]
    const float* inv_freq         = nullptr; // [D/2]
    // Every value of the rotated Q is multiplied by it: the square of YaRN's attention factor, so
    // that Q and K carry the factor together as the reference's cos and sin do (one, unscaled).
    float q_scale = 1.0f;
    float* q_out  = nullptr; // [T][Hq][D]
    // Cache element (d, slot, head, sequence) at base + sequence * seq_stride + head * head_stride
    // + slot * D + d; slot = position, or position % ring for a ring.
    __half* k_cache = nullptr;
    __half* v_cache = nullptr;
    std::int64_t seq_stride = 0, head_stride = 0;
    const std::int32_t* sequences = nullptr; // [T]; null: every token is `sequence`'s
    int sequence = 0;
    int ring     = 0;
    // Token t's K and V at slot t rather than at its position's (rows a quantized cache stores from,
    // kv_store).
    bool row_slots = false;
    // [T]; the position whose slot token t's K and V take, when it is not the token's own (a
    // branch pass's tokens rotate at their branch's positions and lie one branch after another).
    const std::int32_t* slots = nullptr;
};
void qkv_post(const QkvArgs& args, cudaStream_t stream);

// A quantized cache plane (kv_format.h). Row (sequence, head, slot) has its codes at codes +
// sequence * code_seq + head * code_head + slot * code_row and its scales likewise; a row is
// `blocks` 256-wide blocks.
struct KvPlaneView {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    std::int64_t code_row = 0, scale_row = 0, code_head = 0, scale_head = 0, code_seq = 0,
                 scale_seq = 0;
    KvCoding coding = KvCoding::F16;
    bool rotated    = false;
    int blocks      = 1;
};

// Stores F16 rows into a quantized plane: the row of position p (positions[r], or first + r) of
// sequence sequences[r] (or `sequence`) at slot p % dst_ring (dst_ring 0: slot p), turned first for a
// rotated plane. A source row is head h's at src + h * src_head_stride + slot * D, slot p % src_ring
// (src_ring 0: slot r). Positions below store_from are not stored (a prompt chunk longer than a
// ring keeps its last ones); with write_back every source row becomes the value its codes
// represent, unrotated, so a prompt chunk attends to what its decode steps will read.
struct KvStoreArgs {
    __half* src = nullptr;
    std::int64_t src_head_stride = 0;
    int src_ring = 0;
    const std::int32_t* positions = nullptr;
    int first = 0;
    const std::int32_t* sequences = nullptr;
    int sequence = 0;
    int rows = 0, heads = 0;
    KvPlaneView dst;
    int dst_ring   = 0;
    int store_from = 0;
    bool write_back = false;
};
void kv_store(const KvStoreArgs& args, cudaStream_t stream);

// The F16 values of positions [begin, end) of one sequence's quantized plane, unrotated, into dst:
// head h's row of position p at dst + h * dst_head_stride + slot * D, slot p % dst_ring (dst_ring 0:
// slot p); the plane's row at slot p % src_ring (src_ring 0: slot p).
struct KvLoadArgs {
    KvPlaneView src;
    int src_ring = 0, sequence = 0, begin = 0, end = 0, heads = 0;
    __half* dst = nullptr;
    std::int64_t dst_head_stride = 0;
    int dst_ring = 0;
};
void kv_load(const KvLoadArgs& args, cudaStream_t stream);

// x[r] = H x[r] for rows of 256 floats, H the normalized Sylvester-Hadamard transform of the engine's
// rotated planes: queries meeting rotated keys.
void rotate_rows(float* x, int rows, cudaStream_t stream);

// A decode step's rows into a block's quantized planes in one launch: row r's K (unless k_plane has
// no codes) and V, head h at k / v + h * head_stride + r * D, stored at slot positions[r] % ring
// (ring 0: the position) of sequence sequences[r]; and, with q, q_rows rows of 256 floats turned as
// rotate_rows turns them.
struct KvRowsArgs {
    const __half* k = nullptr;
    const __half* v = nullptr;
    std::int64_t head_stride = 0;
    const std::int32_t* positions = nullptr;
    const std::int32_t* sequences = nullptr;
    int rows = 0, heads = 0, ring = 0;
    KvPlaneView k_plane, v_plane;
    float* q   = nullptr;
    int q_rows = 0;
};
void kv_store_rows(const KvRowsArgs& args, cudaStream_t stream);

// The prompt attention mask, F16 [tokens][slots]: 0 where the query at position first + t sees
// slot s, -inf elsewhere. Linear (ring 0): slot s holds position s, visible when s <= first + t.
// Ring: slot s holds the latest position p = s (mod ring) at or before first + tokens - 1, visible
// when first + t - window < p <= first + t.
// With `blocks` ([tokens], the media unit of each chunk row, -1 for text) a ring's slot holding a
// chunk position of the query's own unit is visible too, ahead of the query (a media unit's soft
// tokens attend each other in both directions).
// With `starts` ([tokens]) the chunk holds branches one after another, each continuing the sequence
// at `first`: row t belongs to the branch whose first row is starts[t], sees the positions before
// `first` as a query at first + t - starts[t] would and the rows of its own branch up to itself,
// and no other branch's.
void build_mask(__half* mask, int tokens, int slots, int first, int ring, int window,
                cudaStream_t stream, const std::int32_t* blocks = nullptr,
                const std::int32_t* starts = nullptr);

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream);

// Copies the K and V rows of positions [begin, end) of one sequence from one ring to another:
// position p lives in slot p % ring of each ([kv heads][slots][D] halves, heads head_stride
// elements apart).
struct RingCopyArgs {
    const __half* src_k = nullptr;
    const __half* src_v = nullptr;
    __half* dst_k       = nullptr;
    __half* dst_v       = nullptr;
    std::int64_t src_head_stride = 0, dst_head_stride = 0;
    int src_ring = 0, dst_ring = 0, kv_heads = 0, head_dim = 0, begin = 0, end = 0;
};
void ring_copy(const RingCopyArgs& args, cudaStream_t stream);

// One token per row b of `batch` at positions[b] of sequence sequences[b]: attention over its
// sequence's cache (linear: positions 0..p; ring: the window ending at p), split over `splits`
// key ranges and merged. Head widths 256 (two query heads per KV head) and 512 (eight). With a
// null k_cache (and no k_plane) the key of position j is RoPE_j(V_j * k_norm), the rotation of pair
// i < rope_pairs read from rope[j * rope_pairs + i] (rope_table), rounded to F16 as a stored key is.
// Quantized planes take the place of k_cache / v_cache: q must already be turned for a rotated key
// plane (rotate_rows), and a rotated value plane's output is turned back here.
struct DecodeAttentionArgs {
    const float* q         = nullptr; // [B][Hq][D]
    const __half* k_cache  = nullptr;
    const __half* v_cache  = nullptr;
    KvPlaneView k_plane, v_plane; // quantized, codes non-null
    const float* k_norm    = nullptr; // [D], with k_cache null
    const float2* rope     = nullptr; // [positions][rope_pairs] (cos, sin), with k_cache null
    int rope_pairs         = 0;
    std::int64_t seq_stride = 0, head_stride = 0; // elements, as QkvArgs
    const std::int32_t* sequences = nullptr;      // [B]
    const std::int32_t* positions = nullptr;      // [B]
    int batch = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    int ring = 0, window = 0;
    int splits  = 1;
    float scale = 1.0f;
    float* partial = nullptr;       // workspace: decode_attention_workspace_floats
    __nv_bfloat16* out = nullptr;   // [B][Hq * D], or
    Q8Rows out_q;                   // the same rows in q8_1 form
};
[[nodiscard]] std::size_t decode_attention_workspace_floats(int batch, int query_heads,
                                                            int head_dim, int splits);
void decode_attention(const DecodeAttentionArgs& args, cudaStream_t stream);

// The attention output into the residual and the FFN inputs of the result, per token row:
//   h       += rms_norm(o) * w_post
//   mlp_in   = bf16(rms_norm(h) * ffn_norm)
//   moe_in   = bf16(rms_norm(h) * pre_norm)                    (when given)
//   router_x = rms_norm(h) * router_scale / sqrt(H), FP32      (when given)
struct AttnResidualArgs {
    float* h               = nullptr;
    const __nv_bfloat16* o = nullptr;
    const float* w_post    = nullptr;
    int hidden = 0, tokens = 0;
    float eps  = 1e-6f;
    const float* ffn_norm     = nullptr;
    __nv_bfloat16* mlp_in     = nullptr;
    const float* pre_norm     = nullptr;
    __nv_bfloat16* moe_in     = nullptr;
    const float* router_scale = nullptr;
    float* router_x           = nullptr;
    // q8_1 forms of mlp_in / moe_in, written when given (the BF16 rows then only when non-null).
    Q8Rows mlp_q, moe_q;
};
void attn_residual(const AttnResidualArgs& args, cudaStream_t stream);

// The router of the MoE block over its input rows x (attn_residual's router_x, FP32 [T][H]):
// logits = router . x (FP32 router [E][H]); the top_k experts by softmax, their weights
// renormalized over the top_k and multiplied by expert_scale. ids/weights [T][top_k]; logits is
// FP32 scratch [T][E].
void route(const float* x, int hidden, int tokens, const float* router, const float* expert_scale,
           int experts, int top_k, float* logits, std::int32_t* ids, float* weights,
           cudaStream_t stream);

// route() for at most 8 tokens, with the pairs also grouped by expert into the arrays of a
// gguf::MoeRouting (bounds [E + 1], sorted [T * top_k], active [E], active_count [1]), as
// moe_sort_routes would group them, without its separate launch.
void route_select(const float* x, int hidden, int tokens, const float* router,
                  const float* expert_scale, int experts, int top_k, float* logits,
                  std::int32_t* ids, float* weights, std::int32_t* bounds, std::int32_t* sorted,
                  std::int32_t* active, std::int32_t* active_count, cudaStream_t stream);

// out[p][r] = bf16(gelu_tanh(x[p][r]) * x[p][width + r]) for pairs p of [2 * width] FP32 rows.
void geglu_rows(const float* x, int pairs, int width, __nv_bfloat16* out, cudaStream_t stream);

// sums[i] += products[i] for i < n when *flag is nonzero: the 2^-32 fixed-point sums of the
// routed experts the CPU computed for a call (ExpertDelivery), added as the expert kernels add
// theirs, modulo 2^64.
void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream);

// The block's FFN output into the residual:
//   y    = rms_norm(dense) * w_dense            (w_dense null: y = dense, a dense model)
//   y   += rms_norm(moe) * w_moe                (moe absent in a dense model)
//   h[t] = (h[t] + rms_norm(y) * w_post) * scalar
//   next_out[t] = bf16(rms_norm(h[t]) * next_norm)   (when given: the next block's input)
// (next_norm given with next_out and/or next_q).
// The MoE rows are the 2^-32 fixed-point sums of the expert products (`moe_fixed`, [T][H]) or
// their FP32 rows weighted here: moe[t] = sum_k moe_weights[t * top_k + k] * moe_rows[t * top_k + k].
struct FfnCombineArgs {
    float* h = nullptr;
    int hidden = 0, tokens = 0;
    float eps  = 1e-6f;
    const __nv_bfloat16* dense = nullptr;
    const float* w_dense       = nullptr;
    const float* moe_rows      = nullptr; // [T * top_k][H]
    const float* moe_weights   = nullptr; // [T * top_k]
    int top_k                  = 0;
    const unsigned long long* moe_fixed = nullptr;
    const float* w_moe  = nullptr;
    const float* w_post = nullptr;
    const float* scalar = nullptr; // [1]
    const float* next_norm  = nullptr;
    __nv_bfloat16* next_out = nullptr;
    Q8Rows next_q; // next_out in q8_1 form, written when given (next_out may then be null)
};
void ffn_combine(const FfnCombineArgs& args, cudaStream_t stream);

// rope[p * pairs + i] = (cos, sin) of p * inv_freq[i] for positions [0, positions), computed as
// qkv_post computes a rotation, so a derived key equals the key qkv_post would have stored from the
// same (F16) value.
void rope_table(const float* inv_freq, int pairs, int positions, float2* rope, cudaStream_t stream);

// The keys of positions [0, count) of one sequence from its values: k[h][p] = RoPE_p(v[h][p] *
// k_norm) in F16, both [kv heads][slots][D] with heads head_stride elements apart (linear caches).
// With `turns`, slot p from `turned_from` on rotates as position turns[p - turned_from] (a branch
// pass's rows).
void derive_keys(const __half* v, std::int64_t head_stride, int kv_heads, int head_dim, int count,
                 const float* k_norm, const float2* rope, int rope_pairs, __half* k,
                 cudaStream_t stream, const std::int32_t* turns = nullptr, int turned_from = 0);

// out = bf16(cap * tanh(x / cap)) (cap 0: bf16(x)).
void softcap(const float* x, std::int64_t n, float cap, __nv_bfloat16* out, cudaStream_t stream);

// dst row (dst_rows ? dst_rows[r] : r) = src row (src_rows ? src_rows[r] : r) for r < rows, rows of
// `width` BF16 values (a multiple of 8).
void copy_rows(const __nv_bfloat16* src, const std::int32_t* src_rows, __nv_bfloat16* dst,
               const std::int32_t* dst_rows, int rows, int width, cudaStream_t stream);

// The assistant's input rows: out[r] = [bf16(embedding[r]) ; hidden[r]], BF16 [rows][2 width].
void stem_rows(const float* embedding, const __nv_bfloat16* hidden, int rows, int width,
               __nv_bfloat16* out, cudaStream_t stream);

} // namespace ninfer::models::gemma4::kernels
