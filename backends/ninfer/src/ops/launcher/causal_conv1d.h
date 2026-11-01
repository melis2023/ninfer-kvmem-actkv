#pragma once

// ninfer::ops::detail - private launch prototypes for causal_conv1d.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Registered split partitions, named by their row profile. The wrapper resolves one of these from
// the destination row counts and refuses anything else. The 1024 and 512 profiles are a 27B
// layer's share on each of two and four tensor-parallel ranks.
enum class CausalConvSplitGeometry {
    Rows2048x2048x4096,
    Rows2048x2048x6144,
    Rows1024x1024x3072,
    Rows512x512x1536,
};

inline constexpr std::int32_t kCausalConvParallelMaxTokens = 32;

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_prefill_split_launch(const Tensor& x, const Tensor& weight,
                                        const Tensor& conv_state_in, Tensor& conv_state_out,
                                        Tensor& out0, Tensor& out1, Tensor& out2,
                                        CausalConvSplitGeometry geometry, cudaStream_t stream);
void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_smallt_split_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out0, Tensor& out1, Tensor& out2,
                                       CausalConvSplitGeometry geometry, cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_snapshot_launch(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                   const Tensor& valid_columns, const Tensor& initial_state_slots,
                                   const Tensor& snapshot_base_slots, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
