#pragma once

// ninfer::ops::detail - private launch prototype for lora_swiglu. Included by the wrapper (host)
// and defined by the launcher (.cu). Not part of the public api.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void lora_swiglu_launch(const Tensor& x, const Tensor& a_gate, const Tensor& b_gate,
                        const Tensor& a_up, const Tensor& b_up, const Tensor& plane,
                        Tensor& activation, const Tensor& scratch, float scale_gate,
                        float scale_up, cudaStream_t stream);

} // namespace ninfer::ops::detail
