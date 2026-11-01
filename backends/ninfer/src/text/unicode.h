#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text::unicode_internal {

struct CodepointSpan {
    std::int32_t value = 0;
    std::size_t offset = 0;
    std::size_t length = 0;
};

std::string normalize_nfc(std::string_view text);
CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context);
std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context);
std::string codepoint_to_utf8(std::int32_t codepoint);

bool is_letter(std::int32_t codepoint) noexcept;
bool is_mark(std::int32_t codepoint) noexcept;
bool is_number(std::int32_t codepoint) noexcept;
bool is_whitespace(std::int32_t codepoint) noexcept;
// Letter general categories: Lu or Lt, Ll, and Lm or Lo.
bool is_upper_or_title_letter(std::int32_t codepoint) noexcept;
bool is_lower_letter(std::int32_t codepoint) noexcept;
bool is_modifier_or_other_letter(std::int32_t codepoint) noexcept;
bool is_punctuation(std::int32_t codepoint) noexcept; // P*
bool is_symbol(std::int32_t codepoint) noexcept;      // S*

} // namespace ninfer::text::unicode_internal
