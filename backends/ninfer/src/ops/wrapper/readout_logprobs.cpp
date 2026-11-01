// ninfer::ops - readout_logprobs wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/readout_logprobs.h"

#include "ops/launcher/readout_logprobs.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_vector(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("readout_logprobs: ") + label +
                                    " must be a vector with a positive length");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("readout_logprobs: ") + label +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("readout_logprobs: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("readout_logprobs: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void readout_logprobs(const Tensor& logits, const Tensor& token_ids, std::int32_t count,
                      std::int32_t valid_rows, Tensor& output, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("readout_logprobs: logits must be BF16");
    }
    if (token_ids.dtype != DType::I32) {
        throw std::invalid_argument("readout_logprobs: token_ids must be I32");
    }
    if (output.dtype != DType::FP32) {
        throw std::invalid_argument("readout_logprobs: output must be FP32");
    }

    require_vector(logits, "logits");
    require_vector(token_ids, "token_ids");
    require_vector(output, "output");
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("readout_logprobs: valid_rows must be in [1, physical_rows]");
    }
    if (count <= 0 || count > token_ids.ne[0] || count > output.ne[0] ||
        count > kReadoutLogprobsMaximum) {
        throw std::invalid_argument(
            "readout_logprobs: count must be in [1, min(capacity, kReadoutLogprobsMaximum)]");
    }

    (void)logits.bytes();
    (void)token_ids.bytes();
    (void)output.bytes();
    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(token_ids, alignof(std::int32_t), "token_ids");
    require_accessible(output, alignof(float), "output");
    if (overlaps(output, logits) || overlaps(output, token_ids)) {
        throw std::invalid_argument("readout_logprobs: output must not overlap either input");
    }

    detail::readout_logprobs_launch(logits, token_ids, count, valid_rows, output, stream);
}

} // namespace ninfer::ops
