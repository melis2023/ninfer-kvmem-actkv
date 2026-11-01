#pragma once

// A hot-row profile of the Qwen3.8-Flash-Next n-gram table: the rows a text corpus addresses, the
// most frequent first, counted from the tokens alone (ninfer-ngram-profile). The RamHot residency
// keeps the leading rows that fit its budget in RAM.
//
// File (little-endian): the magic "NFNGHOT1", then u64 table rows, u64 fingerprint of the hash
// constants (ngram_hash_fingerprint), u64 corpus tokens, u64 row count, and that many u32 row ids.

#include "models/qwen4_exp/ngram_hash.h"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct NgramProfile {
    std::uint64_t table_rows  = 0;
    std::uint64_t fingerprint = 0;
    std::uint64_t tokens      = 0; // of the corpus the rows were counted over
    std::vector<std::uint32_t> rows;
};

// Throws std::runtime_error for a file that is not a whole profile.
[[nodiscard]] NgramProfile read_ngram_profile(const std::filesystem::path& path);
[[nodiscard]] NgramProfile decode_ngram_profile(std::span<const std::byte> bytes,
                                                std::string_view label);
void write_ngram_profile(const std::filesystem::path& path, const NgramProfile& profile);

// Refuses (std::invalid_argument) a profile counted for other hash constants than `constants`.
void check_ngram_profile(const NgramProfile& profile, const NgramHashConstants& constants,
                         const std::filesystem::path& path);

} // namespace ninfer::models::qwen4_exp
