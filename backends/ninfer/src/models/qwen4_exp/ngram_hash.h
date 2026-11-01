#pragma once

// The hashed n-gram rows of Qwen3.8-Flash-Next's per-layer embedding (PLE, transformers
// Qwen4ExpTextPLEEmbedding): every token addresses one row of a shared table per hash head, a head
// per (n-gram order, slot), from the token and its predecessors within the current EOS-delimited
// segment. The constants are derived exactly as the reference derives them; the checkpoint also
// stores them, and a loader must refuse a table whose stored constants differ.

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct NgramHashSpec {
    std::uint32_t vocab_size      = 0;  // token vocabulary; bounds the multipliers
    std::uint32_t ngram_size      = 0;  // largest order; orders 2..ngram_size are hashed
    std::uint32_t heads_per_ngram = 0;  // hash heads per order
    std::uint32_t ple_layer_index = 0;  // zero-based position among the PLE layers
    std::uint64_t vocab_base      = 0;  // each head's table is the next prime at or above it
    std::uint64_t divisible_by    = 0;  // the table's row count rounds up to a multiple of it
    std::uint64_t seed            = 0;
};

struct NgramHashConstants {
    std::uint32_t order = 0;                 // ngram_size
    std::uint32_t heads_per_order = 0;       // heads_per_ngram
    std::vector<std::uint64_t> multipliers;  // one per context position, newest first
    std::vector<std::uint64_t> head_vocab;   // rows of each head, head-major over orders
    std::vector<std::uint64_t> head_offset;  // first table row of each head
    std::uint64_t rows = 0;                  // table rows, padded

    [[nodiscard]] std::uint32_t heads() const noexcept {
        return static_cast<std::uint32_t>(head_vocab.size());
    }
};

// Throws std::invalid_argument for a specification the hash cannot represent.
[[nodiscard]] NgramHashConstants derive_ngram_hash_constants(const NgramHashSpec& spec);

// FNV-1a 64 over the constants (order, heads per order, multipliers, head rows and offsets, table
// rows, each a little-endian u64): what a hot-row profile records of the hash it counted rows of.
[[nodiscard]] std::uint64_t ngram_hash_fingerprint(const NgramHashConstants& constants);

// The predecessors of the next token, newest first, as the hash reads them: a sequence starts
// after `order - 1` EOS tokens, and an EOS predecessor cuts every older one to EOS too.
struct NgramContext {
    std::vector<std::int32_t> previous;

    [[nodiscard]] static NgramContext sequence_start(const NgramHashConstants& constants,
                                                     std::int32_t eos);
};

// Writes heads() row ids per token, token-major, into `rows` and advances `context` past
// `tokens`. A token at or past the vocabulary, or a negative one, throws std::invalid_argument.
void ngram_row_ids(const NgramHashConstants& constants, std::span<const std::int32_t> tokens,
                   std::int32_t eos, std::uint32_t vocab_size, NgramContext& context,
                   std::span<std::uint64_t> rows);

} // namespace ninfer::models::qwen4_exp
