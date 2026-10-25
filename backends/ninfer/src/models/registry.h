#pragma once

#include <string_view>

namespace ninfer::models {

// Qwen4Exp (Qwen3.8-Flash-Next) is its own family (models/qwen4_exp); the Qwen3.5 config parser
// never resolves it.
enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
