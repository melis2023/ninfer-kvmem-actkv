#pragma once

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr std::int32_t kStateDim  = 128;
inline constexpr std::int32_t kChunkSize = 64;

[[nodiscard]] constexpr bool are_head_counts_valid(std::int64_t qk_heads,
                                                   std::int64_t value_heads) noexcept {
    return qk_heads > 0 && value_heads >= qk_heads && (value_heads % qk_heads) == 0;
}

// Which GDN value-head layout the loaded checkpoint uses: the `repeat_interleave` form older packs
// store (value head h reads q/k head h / group) or the HF `repeat` form (h % H_qk). The loader
// resolves it from the artifact and sets it once before any forward pass.
inline bool& head_interleave_flag() noexcept {
    static bool value = false;
    return value;
}
inline bool head_interleave() noexcept { return head_interleave_flag(); }
inline void set_head_interleave(bool value) noexcept { head_interleave_flag() = value; }

} // namespace ninfer::ops::detail::gated_delta_net
