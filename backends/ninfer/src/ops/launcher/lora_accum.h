#pragma once

// ninfer::ops::detail - private launch prototype for lora_accum. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void lora_accum_launch(const Tensor& x, const Tensor& a, const Tensor& b, Tensor& out,
                       const Tensor& temp, float scale, cudaStream_t stream);

} // namespace ninfer::ops::detail
