#pragma once

// ninfer::ops::detail - private launch prototype for readout_logprobs.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void readout_logprobs_launch(const Tensor& logits, const Tensor& token_ids, std::int32_t count,
                             std::int32_t valid_rows, Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops::detail
