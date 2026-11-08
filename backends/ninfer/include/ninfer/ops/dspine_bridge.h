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

} // namespace ninfer::ops
