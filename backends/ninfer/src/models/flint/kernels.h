#pragma once

// Flint's own normalization, attention-input, routing and decode-attention kernels. Activations are
// token-major: an FP32 residual h[t][i], BF16 projection inputs x[t][k] (a [K, T] Tensor).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::flint::kernels {

// Rows in the GGUF vector kernels' q8_1 activation form (ggml_bridge.h's vector activation):
// int8 values [T][k] and (d, sum) half2 pairs [T][k / 32]. A kernel given one writes it instead of
// leaving a BF16 row for a separate quantization; k must be a whole number of 32-value blocks.
struct Q8Rows {
    std::int8_t* qs = nullptr;
    half2* ds       = nullptr;
};

// y[t][i] = x[t][i] * rsqrt(mean_i x[t]^2 + eps) * w[i], written as BF16 rows (out), q8_1 rows (q)
// and FP32 rows (f32), each when given.
void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q = {}, float* f32 = nullptr);

// The Q/K/V of one attention block, after its projection (FP32 rows): per head RMSNorm of Q and K
// with their gammas (V as it is), RoPE on Q and K when the block turns (pair (i, i + D/2) by
// position * inv_freq[i]), Q to FP32 [T][Hq][D], K and V to the F16 cache slots of their positions.
struct QkvArgs {
    const float* qkv = nullptr; // [T][Hq*D + 2 * Hkv*D]
    int tokens = 0, query_heads = 0, kv_heads = 0, head_dim = 0;
    const float* q_norm = nullptr;
    const float* k_norm = nullptr;
    float eps           = 1e-6f;
    const std::int32_t* positions = nullptr; // [T]
    const float* inv_freq         = nullptr; // [D/2]; null: no rotation
    float* q_out                  = nullptr; // [T][Hq][D]
    // Cache element (d, slot, head, sequence) at base + sequence * seq_stride + head * head_stride
    // + slot * D + d; slot = position, or position % ring for a ring.
    __half* k_cache = nullptr;
    __half* v_cache = nullptr;
    std::int64_t seq_stride = 0, head_stride = 0;
    const std::int32_t* sequences = nullptr; // [T]; null: every token is `sequence`'s
    int sequence = 0;
    int ring     = 0;
    // [T]; the position whose slot token t's K and V take, when it is not the token's own (a branch
    // pass's tokens rotate at their branch's positions and lie one branch after another).
    const std::int32_t* slots = nullptr;
};
void qkv_post(const QkvArgs& args, cudaStream_t stream);

// The prompt attention mask, F16 [tokens][slots]: 0 where the query at position first + t sees
// slot s, -inf elsewhere. Linear (ring 0): slot s holds position s, visible when s <= first + t.
// Ring: slot s holds the latest position p = s (mod ring) at or before first + tokens - 1, visible
// when first + t - window < p <= first + t.
// With `starts` ([tokens]) the chunk's rows from `shared` on hold branches one after another, each
// continuing the sequence at first + shared, after the chunk's first `shared` rows (a prompt's
// last rows, masked as above): row t belongs to the branch whose first row is starts[t], sees the
// positions before first + shared as a query at first + shared + t - starts[t] would and the rows
// of its own branch up to itself, and no other branch's.
void build_mask(__half* mask, int tokens, int slots, int first, int ring, int window,
                cudaStream_t stream, const std::int32_t* starts = nullptr, int shared = 0);

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
// sequence's F16 cache (linear: positions 0..p; ring: the window ending at p), each query head's
// sink logit joining its softmax denominator, split over `splits` key ranges and merged. Width 128
// with eight query heads per KV head.
struct DecodeAttentionArgs {
    const float* q         = nullptr; // [B][Hq][D]
    const __half* k_cache  = nullptr;
    const __half* v_cache  = nullptr;
    const float* sinks     = nullptr; // [Hq]
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

// The router of a MoE block over its input rows x (FP32 [T][H]): logits = router . x (FP32 router
// [E][H]); the top_k experts by logit (lower id first on equal logits), weighted by their sigmoid
// scores renormalized over the top_k (s / (sum + 1e-20)). ids/weights [T][top_k]; logits is FP32
// scratch [T][E].
void route(const float* x, int hidden, int tokens, const float* router, int experts, int top_k,
           float* logits, std::int32_t* ids, float* weights, cudaStream_t stream);

// route() for at most 8 tokens in one launch (the same logits, experts and weights bit for bit),
// with the pairs also grouped by expert into the arrays of a gguf::MoeRouting (bounds [E + 1],
// sorted [T * top_k], active [E], active_count [1]), as moe_sort_routes would group them. `done`: a
// zeroed device word, zero again when the launch ends (its blocks count themselves there).
void route_select(const float* x, int hidden, int tokens, const float* router, int experts,
                  int top_k, float* logits, std::int32_t* ids, float* weights,
                  std::int32_t* bounds, std::int32_t* sorted, std::int32_t* active,
                  std::int32_t* active_count, std::uint32_t* done, cudaStream_t stream);

// A matrix activation that keeps each 32 values' sum in F16 (Q4_K, Q5_K and the like,
// gguf::matrix_activation_f16_sums) is kept to sums of at most this magnitude: a row or column with
// a larger one is first multiplied by a power of two (swiglu_rows, scale_columns). Flint's routed
// SwiGLU rows reach sums near 90,000, past F16's 65504.
inline constexpr float kF16SumLimit = 16384.0f;

// k of the least power of two 2^k with largest * 2^-k under kF16SumLimit; 0 when `largest` is
// within the limit or not finite. The guards below and Flint's CPU expert products decide alike.
__host__ __device__ inline int f16_sum_exponent(float largest) {
    if (!(largest > kF16SumLimit) || !(largest <= 3.40282347e38f)) { return 0; }
    int k = 0;
    frexpf(largest / kF16SumLimit, &k);
    return k;
}

// out[p][r] = bf16(silu(x[p][r]) * x[p][width + r]) for pairs p of [2 * width] FP32 rows
// (width % 32 == 0). With `weights` [pairs], a row whose largest sum of 32 values (r = 32i to
// 32i + 31) passes kF16SumLimit is written times 2^-k, the least power of two that brings it under
// the limit, and weights[p] is multiplied by 2^k: both factors are exact, so the weighted down
// product is unchanged and its activation's F16 sums stay finite.
void swiglu_rows(const float* x, int pairs, int width, __nv_bfloat16* out, float* weights,
                 cudaStream_t stream);

// The vector activation (Q8Rows of [pairs][width]) of a decode or verification call's BF16 middle
// rows [pairs][width] for a down product whose vector dot reads the F16 sums
// (gguf::vector_activation_f16_sums), with the same guard: a row whose largest sum of 32 values
// passes kF16SumLimit is first multiplied in place by 2^-k (f16_sum_exponent) and weights[p] by
// 2^k. A row within the limit (one an earlier stage of the call scaled among them) keeps its values
// and weight. The codes and (d, sum) pairs are gguf::quantize_vector_activation's of the rows it
// leaves (width % 32 == 0, width <= 1024).
void quantize_pairs(__nv_bfloat16* middle, int pairs, int width, float* weights, Q8Rows out,
                    cudaStream_t stream);

// The same guard for a product of the BF16 rows x [columns][rows] (rows % 32 == 0) accumulated into
// h [columns][h_rows]: each column c of x whose largest sum of 32 values passes kF16SumLimit, and
// column c of h, are multiplied by 2^-k as above, and scale[c] is set to 2^k (1 for the others).
// unscale_columns multiplies h's columns by scale after the product, which returns the sum h
// would have received without the factors.
void scale_columns(__nv_bfloat16* x, int rows, int columns, float* h, int h_rows, float* scale,
                   cudaStream_t stream);
void unscale_columns(float* h, int h_rows, int columns, const float* scale, cudaStream_t stream);

// sums[i] += products[i] for i < n when *flag is nonzero: the 2^-32 fixed-point sums of the
// routed experts the CPU computed for a call (ExpertDelivery), added as the expert kernels add
// theirs, modulo 2^64.
void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream);

// The routed experts' output into the residual and the normalized input of what follows:
//   h[t]       += moe[t]                                  (when given)
//   next_out[t] = bf16(rms_norm(h[t]) * next_norm)        (and/or next_q, its q8_1 form)
// The MoE rows are the 2^-32 fixed-point sums of the expert products (`moe_fixed`, [T][H]) or
// their FP32 rows weighted here: moe[t] = sum_k moe_weights[t * top_k + k] * moe_rows[t * top_k + k].
struct FfnCombineArgs {
    float* h = nullptr;
    int hidden = 0, tokens = 0;
    float eps  = 1e-6f;
    const float* moe_rows    = nullptr; // [T * top_k][H]
    const float* moe_weights = nullptr; // [T * top_k]
    int top_k                = 0;
    const unsigned long long* moe_fixed = nullptr;
    const float* next_norm  = nullptr;
    __nv_bfloat16* next_out = nullptr;
    Q8Rows next_q;
};
void ffn_combine(const FfnCombineArgs& args, cudaStream_t stream);

} // namespace ninfer::models::flint::kernels
