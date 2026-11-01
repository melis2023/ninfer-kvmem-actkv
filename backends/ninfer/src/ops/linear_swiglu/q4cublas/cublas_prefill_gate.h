#pragma once

// Where the cuBLAS prefill route (w4_cublas_prefill.h) takes a call that its policy admits.
//
// The route dequantises the whole weight before its GEMM, a pass that costs the same whatever the
// token count, so it pays only past some width. That width depends on the card and on the route
// it replaces: the integer A8 mainloop at multiples of its 128-column tile, the A16 mainloop at
// every other width. Each Op family therefore has two device-profile keys, `<family>` for the
// widths the integer route cannot take and `<family>/a8` for the ones it can. A band's schedule is
// "on" or "off"; where no band reaches a width, the compiled gate kCublasPrefillMinTokens answers.
//
// No profile can route a call of kCublasPrefillFloorTokens columns or fewer: decode and verify
// rounds run the same Ops and policies over up to that many columns, and they stay off the route.
// Workspace planning asks cublas_prefill_first_width() under the same installed profile as
// execution, so a profile that moves the gate also moves the reservation.

#include <cstdint>
#include <string_view>

namespace ninfer::ops::detail {

struct CublasPrefillFamily {
    std::string_view key;         // widths the integer A8 route cannot take
    std::string_view integer_key; // multiples of its tile
};

inline constexpr CublasPrefillFamily kCublasGateUp{"w4_cublas/q4_linear_swiglu",
                                                   "w4_cublas/q4_linear_swiglu/a8"};
inline constexpr CublasPrefillFamily kCublasDown{"w4_cublas/linear_add/5120x17408",
                                                 "w4_cublas/linear_add/5120x17408/a8"};
inline constexpr CublasPrefillFamily kCublasOut{"w4_cublas/linear_add/5120x6144",
                                                "w4_cublas/linear_add/5120x6144/a8"};
inline constexpr CublasPrefillFamily kCublasGdnInput{"w4_cublas/q4_q5_gdn_input",
                                                     "w4_cublas/q4_q5_gdn_input/a8"};
inline constexpr CublasPrefillFamily kCublasAttnInput{"w4_cublas/q4_q5_attn_input",
                                                      "w4_cublas/q4_q5_attn_input/a8"};

// The LinearAdd family of a residual projection's input width; null for other shapes.
[[nodiscard]] const CublasPrefillFamily* cublas_linear_add_family(std::int32_t input_rows) noexcept;

// Whether the route takes a call of `tokens` columns of `family` on the current device.
[[nodiscard]] bool cublas_prefill_takes(const CublasPrefillFamily& family, std::int32_t tokens);

// The narrowest width in [min_tokens, max_tokens] the route takes for `family` on the current
// device; 0 when it takes none of them.
[[nodiscard]] std::int32_t cublas_prefill_first_width(const CublasPrefillFamily& family,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens);

} // namespace ninfer::ops::detail
