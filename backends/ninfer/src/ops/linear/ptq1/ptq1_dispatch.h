#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/t2/t2_launch.h"

namespace ninfer::ops::detail {

inline constexpr std::int32_t kPtq1TileMinTokens = 33;
[[nodiscard]] bool ptq1_a8_shape_supported(std::int32_t n, std::int32_t k);
void validate_ptq1_shape(std::int32_t n, std::int32_t k, std::int32_t tokens, LinearPolicy policy);
void ptq1_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail
