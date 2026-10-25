#include "models/qwen4_exp/ngram_hash.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

using u64 = std::uint64_t;

// a * b mod m without a 128-bit type (MSVC has none): double-and-add, overflow-safe for any m.
u64 mul_mod(u64 a, u64 b, u64 m) {
    a %= m;
    b %= m;
    if (a < (1ULL << 32U) && b < (1ULL << 32U)) { return a * b % m; }
    u64 result = 0;
    while (b) {
        if (b & 1U) { result = result >= m - a ? result - (m - a) : result + a; }
        a = a >= m - a ? a - (m - a) : a + a;
        b >>= 1U;
    }
    return result;
}

u64 pow_mod(u64 base, u64 exponent, u64 m) {
    u64 result = 1;
    base %= m;
    while (exponent) {
        if (exponent & 1U) { result = mul_mod(result, base, m); }
        base = mul_mod(base, base, m);
        exponent >>= 1U;
    }
    return result;
}

// Deterministic Miller-Rabin for every 64-bit value (the first twelve prime bases suffice).
bool is_prime(u64 n) {
    if (n < 2) { return false; }
    for (const u64 p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL,
                        37ULL}) {
        if (n % p == 0) { return n == p; }
    }
    u64 d       = n - 1;
    unsigned s  = 0;
    while ((d & 1U) == 0) {
        d >>= 1U;
        ++s;
    }
    for (const u64 a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL,
                        37ULL}) {
        u64 x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) { continue; }
        bool composite = true;
        for (unsigned r = 1; r < s; ++r) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) { return false; }
    }
    return true;
}

u64 splitmix64(u64 value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

} // namespace

NgramHashConstants derive_ngram_hash_constants(const NgramHashSpec& spec) {
    if (spec.vocab_size == 0 || spec.ngram_size < 2 || spec.heads_per_ngram == 0 ||
        spec.vocab_base < 2 || spec.divisible_by == 0) {
        throw std::invalid_argument("n-gram hash: invalid specification");
    }
    NgramHashConstants out;
    out.order           = spec.ngram_size;
    out.heads_per_order = spec.heads_per_ngram;

    // Every multiplier keeps token * multiplier below 2^63, so the reference's signed 64-bit
    // products and these unsigned ones agree, and so do their XORs.
    constexpr u64 kMaxLong = static_cast<u64>(std::numeric_limits<std::int64_t>::max());
    const u64 half         = (kMaxLong / spec.vocab_size) / 2;
    const u64 base_seed    = spec.seed + 10007ULL * spec.ple_layer_index;
    for (std::uint32_t i = 0; i < spec.ngram_size; ++i) {
        const u64 mixed = splitmix64(base_seed + 0x9E3779B97F4A7C15ULL * (i + 1ULL));
        out.multipliers.push_back(2 * (mixed % half) + 1);
    }

    // Head h of this PLE layer takes the (global head + 1)-th prime at or above the base, so the
    // heads of every layer have distinct moduli.
    const std::uint32_t heads = (spec.ngram_size - 1) * spec.heads_per_ngram;
    const u64 skip            = static_cast<u64>(spec.ple_layer_index) * heads;
    u64 candidate             = spec.vocab_base;
    for (u64 found = 0; found < skip + heads; ++candidate) {
        if (!is_prime(candidate)) { continue; }
        if (found++ >= skip) { out.head_vocab.push_back(candidate); }
    }
    u64 offset = 0;
    for (const u64 vocab : out.head_vocab) {
        out.head_offset.push_back(offset);
        if (offset > std::numeric_limits<u64>::max() - vocab) {
            throw std::invalid_argument("n-gram hash: table rows overflow");
        }
        offset += vocab;
    }
    out.rows = (offset + spec.divisible_by - 1) / spec.divisible_by * spec.divisible_by;
    return out;
}

NgramContext NgramContext::sequence_start(const NgramHashConstants& constants, std::int32_t eos) {
    return NgramContext{std::vector<std::int32_t>(constants.order - 1, eos)};
}

void ngram_row_ids(const NgramHashConstants& constants, std::span<const std::int32_t> tokens,
                   std::int32_t eos, std::uint32_t vocab_size, NgramContext& context,
                   std::span<std::uint64_t> rows) {
    const std::uint32_t heads = constants.heads();
    if (context.previous.size() + 1 != constants.order ||
        constants.multipliers.size() != constants.order ||
        heads != (constants.order - 1) * constants.heads_per_order) {
        throw std::invalid_argument("n-gram hash: context does not match the constants");
    }
    if (rows.size() != tokens.size() * heads) {
        throw std::invalid_argument("n-gram hash: row output size mismatch");
    }
    if (eos < 0 || static_cast<std::uint32_t>(eos) >= vocab_size) {
        throw std::invalid_argument("n-gram hash: EOS outside the vocabulary");
    }
    std::vector<std::int32_t> window(constants.order);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        const std::int32_t token = tokens[t];
        if (token < 0 || static_cast<std::uint32_t>(token) >= vocab_size) {
            throw std::invalid_argument("n-gram hash: token " + std::to_string(token) +
                                        " outside the vocabulary");
        }
        window[0] = token;
        bool cut  = false;
        for (std::uint32_t s = 1; s < constants.order; ++s) {
            const std::int32_t previous = context.previous[s - 1];
            window[s]                   = cut ? eos : previous;
            cut                         = cut || previous == eos;
        }
        u64 mixed = static_cast<u64>(window[0]) * constants.multipliers[0];
        for (std::uint32_t order = 2; order <= constants.order; ++order) {
            mixed ^= static_cast<u64>(window[order - 1]) * constants.multipliers[order - 1];
            for (std::uint32_t slot = 0; slot < constants.heads_per_order; ++slot) {
                const std::uint32_t head = (order - 2) * constants.heads_per_order + slot;
                rows[t * heads + head] =
                    mixed % constants.head_vocab[head] + constants.head_offset[head];
            }
        }
        for (std::uint32_t s = constants.order - 1; s > 1; --s) {
            context.previous[s - 1] = context.previous[s - 2];
        }
        context.previous[0] = token;
    }
}

} // namespace ninfer::models::qwen4_exp
