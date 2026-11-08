#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// DSpine selected-prefix bridge, one write after each draft layer.
// All matrices use the engine's feature-major layout [features, columns].

void dspine_bridge_read(const Tensor& hidden, const Tensor* residual, float eps, Tensor& raw,
                        Tensor& normalized, Tensor& rms, cudaStream_t stream);

void dspine_bridge_linear(const Tensor& x, const Tensor& weight, Tensor& out,
                          cudaStream_t stream);

// projected[c, t] = sum_h normalized[h, t] * weight[layer, c, h].
// early is [4, C, H]; final is [C, H]; layer < 4 selects early[layer].
void dspine_bridge_read_proj(const Tensor& normalized, const Tensor& early,
                             const Tensor& final, std::int32_t layer, Tensor& projected,
                             cudaStream_t stream);

void dspine_bridge_codes(const Tensor& projected, const Tensor& codebook, const Tensor& anchors,
                         std::int32_t block_size, std::int32_t batch, float sqrt_c, float eps,
                         Tensor& codes, cudaStream_t stream);

void dspine_bridge_message(const Tensor& q, const Tensor& s, const Tensor& v,
                           const Tensor& gate_bias, std::int32_t block_size, std::int32_t batch,
                           Tensor& message, cudaStream_t stream);

// delta[h, t] = sum_r message[r, t] * write[layer, h, r].
void dspine_bridge_write(const Tensor& message, const Tensor& write, std::int32_t layer,
                         Tensor& delta, cudaStream_t stream);

void dspine_bridge_finish(const Tensor& raw, const Tensor& delta, const Tensor& rms,
                          const Tensor& next_norm, std::int32_t block_size, std::int32_t batch,
                          float eps, Tensor& out, Tensor& normed, cudaStream_t stream);

// `out [R,H] = norm[h] * write[last_layer][h,r]`, ready as a [H,R] linear weight.
void dspine_write_scaled(const Tensor& write, const Tensor& norm, std::int32_t layer,
                         Tensor& out, cudaStream_t stream);

// Stable top-16 over each column of a BF16 logits matrix. `logits` is consumed in place.
void dspine_top16(Tensor& logits, Tensor& ids, std::int32_t valid_rows, cudaStream_t stream);

// Serial last-write refinement over the proposal rows.
// cand_ids [16,rows,batch] rank-fastest, rms [width,batch], raw [H,width,batch],
// q [rank,width,batch], anchors [batch], write_scaled [R,H], head_rows [H,columns],
// norm [H]. message [rank,columns] and delta [H,columns] are transient.
// scratch is FP32 [2*rank,columns]. drafts is I32 [rows,batch].
void dspine_refine(const Tensor& cand_ids, const Tensor& rms, const Tensor& raw, const Tensor& q,
                   const Tensor& anchors, const Tensor& codebook, const Tensor& predecessor_gate,
                   const Tensor& prev, const Tensor& gate_bias, const Tensor& write_scaled,
                   const Tensor& head_rows, const Tensor& norm, std::int32_t width,
                   std::int32_t batch, Tensor& message, Tensor& delta, Tensor& scratch,
                   Tensor& drafts, cudaStream_t stream);

// Causal block attention over the sliding context plus the current proposal block.
// q [D,Hq,width,batch], k/v [D,Hkv,width,batch], context K/V
// [D,padded,Hkv,lanes], positions [width,batch], valid_columns/lanes [batch].
void dspine_causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid_columns,
                             const Tensor& lanes, const Tensor& context_k,
                             const Tensor& context_v, std::int32_t capacity,
                             std::int32_t padded_capacity, float scale, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::ops
