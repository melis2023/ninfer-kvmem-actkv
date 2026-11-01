#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

// One column of the sampler allow-bitset. Bit 0 forbids `token`. The column covers
// token ids in [0, domain). High bits past `domain` in the last word stay set; the
// sampler does not read them. A stride of zero makes every speculative column use
// this one column.
inline void fill_forbidden_token_mask(std::span<std::uint32_t> words, std::uint32_t token,
                                      std::uint32_t domain) {
    if (domain == 0 || token >= domain) {
        throw std::invalid_argument("forbidden token is outside the sampling domain");
    }
    const std::size_t needed = (static_cast<std::size_t>(domain) + 31U) / 32U;
    if (words.size() < needed) {
        throw std::invalid_argument("forbidden-token mask is shorter than the sampling domain");
    }
    for (std::size_t index = 0; index < needed; ++index) { words[index] = 0xFFFFFFFFU; }
    words[token / 32U] &= ~(1U << (token % 32U));
}

} // namespace ninfer::models::qwen3_5
