#pragma once

// The Qwen3.8-Flash-Next n-gram table options that ninfer, ninfer-serve and ninfer-perplexity
// share: where the table comes from, where its rows live and how they are read.

#include "ninfer/engine_options.h"
#include "ninfer/model_enums.h"

#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// The help lines of the options, in the 32-column layout of ninfer's and ninfer-serve's help.
inline constexpr const char* kNgramTableHelp =
    "  --ngram-table PATH            Qwen3.8-Flash-Next: the n-gram table artifact,\n"
    "                                for a model published without its table\n"
    "  --ngram-residency disk|ram|ram-hot\n"
    "                                where the table's rows come from: its file, 16\n"
    "                                rows per token (default); all of it in RAM; or\n"
    "                                the rows a hot-row profile ranks first in RAM\n"
    "                                and the rest from the file\n"
    "  --ngram-io buffered|direct|mmap  how rows are read from the file: through the\n"
    "                                page cache (default), past it, or mapped\n"
    "  --ngram-io-depth N            row reads in flight, 1..1024 (default 64)\n"
    "  --ngram-hot-profile PATH      ram-hot: override the table's embedded profile\n"
    "                                with a ninfer-ngram-profile file\n"
    "  --ngram-ram-mib N             disk, ram-hot: RAM for rows and their index\n"
    "                                (disk default 0/off; ram-hot default 4096)\n"
    "  --ngram-lock                  ram, ram-hot: lock the rows in physical memory\n"
    "  --no-ngram-table              Qwen3.8-Flash-Next: run without the n-gram table.\n"
    "                                Non-standard experimental mode: the model was\n"
    "                                trained with the table and degrades badly\n"
    "                                without it (WikiText-2 perplexity 2.66 -> 5.01)\n";

namespace detail {

inline std::uint64_t parse_ngram_count(std::string_view text, std::string_view option,
                                       std::uint64_t low, std::uint64_t high) {
    std::uint64_t value     = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < low || value > high) {
        throw std::invalid_argument(std::string(option) + " takes an integer in " +
                                    std::to_string(low) + ".." + std::to_string(high));
    }
    return value;
}

} // namespace detail

// Applies `option` when it is one of the n-gram table options, taking its argument from
// `value()`; false leaves it to the caller.
template <class Value>
bool parse_ngram_table_option(std::string_view option, NgramTableOptions& out, Value&& value) {
    if (option == "--ngram-table") {
        out.path = std::string(value());
    } else if (option == "--no-ngram-table") {
        out.disabled = true;
    } else if (option == "--ngram-residency") {
        const std::string residency(value());
        if (residency == "disk") {
            out.residency = NgramResidency::Disk;
        } else if (residency == "ram") {
            out.residency = NgramResidency::Ram;
        } else if (residency == "ram-hot") {
            out.residency = NgramResidency::RamHot;
        } else {
            throw std::invalid_argument("--ngram-residency must be disk, ram or ram-hot");
        }
    } else if (option == "--ngram-io") {
        const std::string io(value());
        if (io == "buffered") {
            out.io = NgramIo::Buffered;
        } else if (io == "direct") {
            out.io = NgramIo::Direct;
        } else if (io == "mmap") {
            out.io = NgramIo::Mapped;
        } else {
            throw std::invalid_argument("--ngram-io must be buffered, direct or mmap");
        }
    } else if (option == "--ngram-io-depth") {
        out.io_depth = static_cast<std::uint32_t>(
            detail::parse_ngram_count(std::string(value()), option, 1, 1024));
    } else if (option == "--ngram-hot-profile") {
        out.hot_profile = std::string(value());
    } else if (option == "--ngram-ram-mib") {
        out.ram_budget_bytes =
            detail::parse_ngram_count(std::string(value()), option, 0, std::uint64_t{1} << 30U)
            << 20U;
    } else if (option == "--ngram-lock") {
        out.lock = true;
    } else {
        return false;
    }
    return true;
}

} // namespace ninfer::product
