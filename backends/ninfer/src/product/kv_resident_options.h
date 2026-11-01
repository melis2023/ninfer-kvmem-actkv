#pragma once

// --kv-resident, which ninfer, ninfer-serve and ninfer-perplexity share: how much of each
// Qwen3.8-Flash-Next sparse-attention layer's KV stays in device memory.

#include <charconv>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// The help lines, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kKvResidentHelp =
    "  --kv-resident N|all|auto      Qwen3.8-Flash-Next: positions of each sparse-\n"
    "                                attention layer's KV a sequence keeps on the GPU;\n"
    "                                a longer context keeps that KV in pinned RAM and\n"
    "                                the GPU the blocks it reads, freeing memory for\n"
    "                                experts (auto: 32768 with host or disk experts,\n"
    "                                all with device experts)\n";

// Applies `option` when it is --kv-resident, taking its argument from `value()`; false leaves it
// to the caller.
template <class Value>
bool parse_kv_resident_option(std::string_view option, std::optional<std::uint32_t>& out,
                              Value&& value) {
    if (option != "--kv-resident") { return false; }
    const std::string text(value());
    if (text == "all") {
        out = 0;
    } else if (text == "auto") {
        out.reset();
    } else {
        std::uint32_t positions = 0;
        const auto [end, error] =
            std::from_chars(text.data(), text.data() + text.size(), positions);
        if (error != std::errc{} || end != text.data() + text.size() || positions == 0) {
            throw std::invalid_argument("--kv-resident takes a positive position count, all or "
                                        "auto");
        }
        out = positions;
    }
    return true;
}

} // namespace ninfer::product
