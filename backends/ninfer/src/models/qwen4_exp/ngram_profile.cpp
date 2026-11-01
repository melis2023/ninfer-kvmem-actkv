#include "models/qwen4_exp/ngram_profile.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

static_assert(std::endian::native == std::endian::little, "profiles are little-endian");

constexpr char kMagic[8]             = {'N', 'F', 'N', 'G', 'H', 'O', 'T', '1'};
constexpr std::uint64_t kHeaderBytes = 8 + 4 * 8;

} // namespace

NgramProfile read_ngram_profile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("cannot open the hot-row profile " + path.string()); }
    const auto size = file.tellg();
    if (size < 0) { throw std::runtime_error("cannot size the hot-row profile " + path.string()); }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) { throw std::runtime_error("cannot read the hot-row profile " + path.string()); }
    return decode_ngram_profile(bytes, path.string());
}

NgramProfile decode_ngram_profile(std::span<const std::byte> bytes, std::string_view label) {
    const std::string name(label);
    if (bytes.size() < kHeaderBytes || std::memcmp(bytes.data(), kMagic, sizeof kMagic) != 0) {
        throw std::runtime_error(name + " is not an n-gram hot-row profile");
    }
    std::uint64_t header[4] = {};
    std::memcpy(header, bytes.data() + sizeof kMagic, sizeof header);
    NgramProfile out{.table_rows = header[0], .fingerprint = header[1], .tokens = header[2]};
    const std::uint64_t count = header[3];
    const auto payload = bytes.size() - kHeaderBytes;
    if (payload % sizeof(std::uint32_t) != 0 || count != payload / sizeof(std::uint32_t)) {
        throw std::runtime_error("the hot-row profile " + name + " has an invalid row count");
    }
    out.rows.resize(static_cast<std::size_t>(count));
    if (!out.rows.empty()) { std::memcpy(out.rows.data(), bytes.data() + kHeaderBytes, payload); }
    return out;
}

void write_ngram_profile(const std::filesystem::path& path, const NgramProfile& profile) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t header[4] = {profile.table_rows, profile.fingerprint, profile.tokens,
                                     profile.rows.size()};
    file.write(kMagic, sizeof kMagic);
    file.write(reinterpret_cast<const char*>(header), sizeof header);
    file.write(reinterpret_cast<const char*>(profile.rows.data()),
               static_cast<std::streamsize>(profile.rows.size() * 4));
    file.close();
    if (!file) { throw std::runtime_error("cannot write the hot-row profile " + path.string()); }
}

void check_ngram_profile(const NgramProfile& profile, const NgramHashConstants& constants,
                         const std::filesystem::path& path) {
    if (profile.table_rows != constants.rows ||
        profile.fingerprint != ngram_hash_fingerprint(constants)) {
        throw std::invalid_argument("the hot-row profile " + path.string() +
                                    " was counted for another n-gram hash than the model's");
    }
    auto sorted = profile.rows;
    std::sort(sorted.begin(), sorted.end());
    if ((!sorted.empty() && sorted.back() >= constants.rows) ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("the hot-row profile " + path.string() +
                                    " has duplicate or out-of-range rows");
    }
}

} // namespace ninfer::models::qwen4_exp
