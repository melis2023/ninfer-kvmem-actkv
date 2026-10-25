#pragma once

#include "ops/linear/q6/q6_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q6Launch select_q6_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2048(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n1152_k1536(std::int32_t tokens);

[[nodiscard]] Q6Launch select_q6_n248320_k5120_unified(std::int32_t tokens);

// The vocabulary-head schedule the device profile's "q6_head/248320x5120" entry names for this
// width ("gemv" through 2 tokens, "small_t" through 32), or null to keep the route tables.
[[nodiscard]] Q6Launch routed_q6_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2048_unified(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n1152_k1536_unified(std::int32_t tokens);

} // namespace ninfer::ops::detail
