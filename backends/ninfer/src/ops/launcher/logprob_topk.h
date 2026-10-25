#pragma once

// ninfer::ops::detail - private launch prototype for logprob_topk.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t logprob_topk_workspace_exact_bytes(std::int32_t rows);

void logprob_topk_launch(const Tensor& logits, const SamplingConfig* configs,
                         const std::int32_t* drafts, std::int32_t token_domain, Tensor& top_ids,
                         Tensor& top_values, Tensor& lse, const Tensor& active,
                         DeviceSpan workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
