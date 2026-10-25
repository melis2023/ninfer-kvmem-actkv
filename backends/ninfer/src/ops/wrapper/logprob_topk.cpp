// ninfer::ops - logprob_topk wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/logprob_topk.h"

#include "ops/launcher/logprob_topk.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void reject(const std::string& message) {
    throw std::invalid_argument("logprob_topk: " + message);
}

void require_shape(const Tensor& tensor, std::int32_t d0, std::int32_t d1, std::int32_t d2,
                   const char* label) {
    if (tensor.ne[0] != d0 || tensor.ne[1] != d1 || tensor.ne[2] != d2 || tensor.ne[3] != 1) {
        reject(std::string(label) + " must have shape [" + std::to_string(d0) + "," +
               std::to_string(d1) + "," + std::to_string(d2) + "]");
    }
}

void require_accessible(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.dtype != dtype) { reject(std::string(label) + " has the wrong dtype"); }
    if (tensor.data == nullptr) { reject(std::string(label) + " data must be non-null"); }
    if (!tensor.is_contiguous()) { reject(std::string(label) + " must be contiguous"); }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) % dtype_size(dtype)) != 0) {
        reject(std::string(label) + " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

std::size_t logprob_topk_workspace_capacity_bytes(std::int32_t token_domain, std::int32_t rows) {
    if (token_domain < kLogprobTopK || rows <= 0) {
        throw std::invalid_argument("logprob_topk workspace: invalid profile");
    }
    return detail::logprob_topk_workspace_exact_bytes(rows);
}

void logprob_topk(const Tensor& logits, const SamplingConfig* configs, const Tensor* drafts,
                  std::int32_t token_domain, Tensor& top_ids, Tensor& top_values, Tensor& lse,
                  const Tensor& active, WorkspaceArena& workspace, cudaStream_t stream) {
    require_accessible(logits, DType::BF16, "logits");
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0 || logits.ne[2] <= 0 || logits.ne[3] != 1) {
        reject("logits must be [physical_rows,columns,batch] with positive dimensions");
    }
    const std::int32_t columns = logits.ne[1];
    const std::int32_t batch   = logits.ne[2];
    if (token_domain < kLogprobTopK || token_domain > logits.ne[0]) {
        reject("token_domain must be in [kLogprobTopK,physical_rows]");
    }
    if (configs == nullptr) { reject("sampling configs must be non-null"); }
    const std::int32_t* draft_data = nullptr;
    if (columns > 1) {
        if (drafts == nullptr) { reject("a multi-column call needs the drafts overlay"); }
        require_accessible(*drafts, DType::I32, "drafts");
        require_shape(*drafts, columns - 1, batch, 1, "drafts");
        draft_data = static_cast<const std::int32_t*>(drafts->data);
    }
    require_accessible(top_ids, DType::I32, "top_ids");
    require_accessible(top_values, DType::FP32, "top_values");
    require_accessible(lse, DType::FP32, "lse");
    require_accessible(active, DType::I32, "active");
    require_shape(top_ids, kLogprobTopK, columns, batch, "top_ids");
    require_shape(top_values, kLogprobTopK, columns, batch, "top_values");
    require_shape(lse, columns, batch, 1, "lse");
    require_shape(active, 1, 1, 1, "active");
    const Tensor* outputs[] = {&top_ids, &top_values, &lse};
    for (const Tensor* output : outputs) {
        if (overlaps(*output, logits) || overlaps(*output, active) ||
            (columns > 1 && overlaps(*output, *drafts))) {
            reject("outputs must not overlap the inputs");
        }
    }
    if (overlaps(top_ids, top_values) || overlaps(top_ids, lse) || overlaps(top_values, lse)) {
        reject("outputs must not overlap one another");
    }

    auto scope = workspace.scope();
    const DeviceSpan scratch =
        workspace.alloc_bytes(detail::logprob_topk_workspace_exact_bytes(columns * batch));
    detail::logprob_topk_launch(logits, configs, draft_data, token_domain, top_ids, top_values,
                                lse, active, scratch, stream);
}

} // namespace ninfer::ops
