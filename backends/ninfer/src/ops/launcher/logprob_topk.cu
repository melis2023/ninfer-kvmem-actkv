// Implements: include/ninfer/ops/logprob_topk.h
// Match: wrapper-validated contiguous tensors, `batch` device SamplingConfig records, and a
// caller-owned scratch span of logprob_topk_workspace_exact_bytes().
// Algorithm assumptions: split-CTA partials, then one combining CTA per row (kernel header).
#include "ops/launcher/logprob_topk.h"

#include "core/device.h"
#include "ops/kernel/logprob_topk.cuh"

namespace ninfer::ops::detail {

std::size_t logprob_topk_workspace_exact_bytes(std::int32_t rows) {
    const std::size_t partials =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(logprob_topk_split(rows));
    return partials * (2 * sizeof(float) + kLogprobTopK * sizeof(LogprobCandidate));
}

void logprob_topk_launch(const Tensor& logits, const SamplingConfig* configs,
                         const std::int32_t* drafts, std::int32_t token_domain, Tensor& top_ids,
                         Tensor& top_values, Tensor& lse, const Tensor& active,
                         DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t columns = logits.ne[1];
    const std::int32_t rows    = logits.ne[1] * logits.ne[2];
    const std::int32_t split   = logprob_topk_split(rows);
    const std::size_t partials = static_cast<std::size_t>(rows) * static_cast<std::size_t>(split);

    auto* maxima = static_cast<float*>(workspace.data);
    float* sums  = maxima + partials;
    auto* top    = reinterpret_cast<LogprobCandidate*>(sums + partials);
    const auto* flag = static_cast<const std::int32_t*>(active.data);

    logprob_topk_partials_kernel<<<dim3(static_cast<unsigned int>(split),
                                        static_cast<unsigned int>(rows)),
                                   kLogprobTopkBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), configs, drafts, columns, token_domain,
        logits.ne[0], split, flag, maxima, sums, top);
    CUDA_CHECK(cudaGetLastError());
    logprob_topk_combine_kernel<<<static_cast<unsigned int>(rows), kLogprobTopkBlock, 0, stream>>>(
        maxima, sums, top, split, flag, static_cast<std::int32_t*>(top_ids.data),
        static_cast<float*>(top_values.data), static_cast<float*>(lse.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
