#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <optional>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// The vocabulary id whose decoded text is exactly `text`, when that id is unique.
// Zero matches and more than one match both return empty: a missing or ambiguous
// spelling is left to the model instead of guessing which id to forbid.
template <typename Lookup>
std::optional<TokenId> unique_exact_token(std::size_t vocab_size, Lookup&& decoded_bytes,
                                          std::string_view text) {
    std::optional<TokenId> found;
    for (std::size_t id = 0; id < vocab_size; ++id) {
        if (decoded_bytes(id) != text) { continue; }
        if (found) { return {}; }
        found = static_cast<TokenId>(id);
    }
    return found;
}

} // namespace ninfer::models::qwen3_5::frontend
