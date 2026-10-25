// The PLE n-gram hash of Qwen3.8-Flash-Next. Oracles: the constants the checkpoint stores as
// buffers (layers.1.ple.ple_embedding.{layer_multipliers, ngram_heads_vocab_sizes,
// ngram_heads_offsets}) for the derivation, and for the rows an independent transcription of the
// reference's whole-sequence form (segment starts from the last EOS, products checked below 2^63) against the
// streamed, chunked one.
#include "models/qwen4_exp/ngram_hash.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

constexpr std::uint32_t kVocab = 248320;
constexpr std::int32_t kEos    = 248044;

NgramHashSpec flash_next_spec() {
    return NgramHashSpec{.vocab_size      = kVocab,
                         .ngram_size      = 3,
                         .heads_per_ngram = 8,
                         .ple_layer_index = 0,
                         .vocab_base      = 20000000,
                         .divisible_by    = 128,
                         .seed            = 1234};
}

// The checkpoint's stored buffers (read from the safetensors by byte range).
constexpr std::array<std::uint64_t, 3> kStoredMultipliers{23703573157769ULL, 20109073645365ULL,
                                                           8052911324071ULL};
constexpr std::array<std::uint64_t, 16> kStoredVocab{
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr std::array<std::uint64_t, 16> kStoredOffsets{
    0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};

// The reference's form: a token's context positions before its segment start (one past the last
// EOS strictly before it, or the sequence start) read as EOS.
std::vector<std::uint64_t> reference_rows(const std::vector<std::int32_t>& tokens) {
    std::vector<std::uint64_t> rows;
    std::size_t segment_start = 0;
    for (std::size_t p = 0; p < tokens.size(); ++p) {
        if (p > 0 && tokens[p - 1] == kEos) { segment_start = p; }
        std::array<std::uint64_t, 3> context{};
        for (std::size_t s = 0; s < 3; ++s) {
            const bool inside = p >= s && p - s >= segment_start;
            context[s]        = static_cast<std::uint64_t>(inside ? tokens[p - s] : kEos);
        }
        const auto product = [&](std::size_t s) {
            constexpr std::uint64_t kMaxLong = 0x7fffffffffffffffULL;
            require(context[s] <= kMaxLong / kStoredMultipliers[s], "product past 2^63");
            return context[s] * kStoredMultipliers[s];
        };
        const std::uint64_t mixed2 = product(0) ^ product(1);
        const std::uint64_t mixed3 = mixed2 ^ product(2);
        for (std::size_t h = 0; h < 16; ++h) {
            rows.push_back((h < 8 ? mixed2 : mixed3) % kStoredVocab[h] + kStoredOffsets[h]);
        }
    }
    return rows;
}

int run() {
    const NgramHashConstants constants = derive_ngram_hash_constants(flash_next_spec());
    require(constants.heads() == 16, "16 hash heads");
    require(std::equal(constants.multipliers.begin(), constants.multipliers.end(),
                       kStoredMultipliers.begin(), kStoredMultipliers.end()),
            "multipliers differ from the checkpoint's buffer");
    require(std::equal(constants.head_vocab.begin(), constants.head_vocab.end(),
                       kStoredVocab.begin(), kStoredVocab.end()),
            "head vocabularies differ from the checkpoint's buffer");
    require(std::equal(constants.head_offset.begin(), constants.head_offset.end(),
                       kStoredOffsets.begin(), kStoredOffsets.end()),
            "head offsets differ from the checkpoint's buffer");
    require(constants.rows == 320001536ULL, "table rows (320,001,446 rounded to 128)");

    // A second PLE layer takes the next sixteen primes and its own multipliers.
    NgramHashSpec second = flash_next_spec();
    second.ple_layer_index = 1;
    const NgramHashConstants next = derive_ngram_hash_constants(second);
    require(next.head_vocab.front() > constants.head_vocab.back(), "layer 1 primes follow layer 0");
    require(next.multipliers != constants.multipliers, "layer 1 has its own multipliers");

    // Sequences with EOS at the start, adjacent EOS, EOS at the end and long runs without one,
    // streamed in chunks of every size from 1 to 7 against the whole-sequence reference.
    std::mt19937 random(20261002u);
    std::uniform_int_distribution<std::int32_t> token(0, static_cast<std::int32_t>(kVocab) - 1);
    for (int trial = 0; trial < 64; ++trial) {
        std::vector<std::int32_t> tokens(1 + trial * 3);
        for (auto& id : tokens) { id = random() % 5 == 0 ? kEos : token(random); }
        if (trial % 4 == 0) { tokens.front() = kEos; }
        if (trial % 4 == 1) { tokens.back() = kEos; }
        if (trial == 2) { tokens.assign(tokens.size(), kVocab - 1); }
        const std::vector<std::uint64_t> expected = reference_rows(tokens);
        for (std::size_t chunk = 1; chunk <= 7; ++chunk) {
            NgramContext context = NgramContext::sequence_start(constants, kEos);
            std::vector<std::uint64_t> rows(tokens.size() * 16);
            for (std::size_t begin = 0; begin < tokens.size(); begin += chunk) {
                const std::size_t count = std::min(chunk, tokens.size() - begin);
                ngram_row_ids(constants, std::span(tokens).subspan(begin, count), kEos, kVocab,
                              context, std::span(rows).subspan(begin * 16, count * 16));
            }
            require(rows == expected, "streamed rows differ from the reference at trial " +
                                          std::to_string(trial) + ", chunk " +
                                          std::to_string(chunk));
        }
        for (const std::uint64_t row : expected) {
            require(row < constants.rows, "row past the table");
        }
    }

    // The token's own EOS does not cut its own context; the next token's predecessors are cut.
    {
        const std::vector<std::int32_t> tokens{11, 12, kEos, 13};
        const auto rows = reference_rows(tokens);
        NgramContext context = NgramContext::sequence_start(constants, kEos);
        std::vector<std::uint64_t> streamed(tokens.size() * 16);
        ngram_row_ids(constants, tokens, kEos, kVocab, context, streamed);
        require(streamed == rows, "EOS cut");
        require(context.previous == std::vector<std::int32_t>{13, kEos},
                "context carries the last two raw tokens");
    }

    // Refusals.
    bool refused = false;
    try {
        NgramContext context = NgramContext::sequence_start(constants, kEos);
        std::vector<std::uint64_t> rows(16);
        const std::int32_t outside = static_cast<std::int32_t>(kVocab);
        ngram_row_ids(constants, std::span(&outside, 1), kEos, kVocab, context, rows);
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "a token past the vocabulary is refused");
    refused = false;
    try {
        NgramHashSpec invalid = flash_next_spec();
        invalid.ngram_size    = 1;
        (void)derive_ngram_hash_constants(invalid);
    } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "an order below two is refused");
    return 0;
}

} // namespace

int main() {
    try {
        const int result = run();
        std::cout << "PASS qwen4_exp n-gram hash\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
