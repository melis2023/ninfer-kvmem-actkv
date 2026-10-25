// NgramTableReader reads the rows the hash addresses from a table file, or a table split across two
// files inside a row, from disk and from RAM, byte for byte, and refuses a short file or a row past
// the table.
#include "models/qwen4_exp/ngram_table.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
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

int run() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("ninfer_ngram_table_" + std::to_string(std::random_device{}()) + ".bin");
    constexpr std::uint64_t kOffset = 4096 + 17;
    constexpr std::uint32_t kRowBytes = 162;
    constexpr std::uint64_t kRows = 5000;
    std::vector<std::uint8_t> payload(kRows * kRowBytes);
    std::mt19937 random(7001u);
    for (auto& byte : payload) byte = static_cast<std::uint8_t>(random());
    {
        std::ofstream file(path, std::ios::binary);
        const std::vector<char> header(kOffset, 'h');
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(reinterpret_cast<const char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
    }
    const NgramTableLayout layout{{{path, kOffset, kRows * kRowBytes}}, kRowBytes, kRows};
    // The same table split across two files at a byte that falls inside a row, as an artifact's
    // part boundary may.
    const auto second         = std::filesystem::path(path.string() + ".part");
    const std::uint64_t split = 2500 * kRowBytes + 61;
    {
        std::ofstream file(second, std::ios::binary);
        const std::vector<char> header(4096, 'p');
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(reinterpret_cast<const char*>(payload.data() + split),
                   static_cast<std::streamsize>(payload.size() - split));
    }
    const NgramTableLayout split_layout{
        {{path, kOffset, split}, {second, 4096, kRows * kRowBytes - split}}, kRowBytes, kRows};
    std::vector<std::uint64_t> ids;
    for (int i = 0; i < 300; ++i) ids.push_back(random() % kRows);
    ids.push_back(0);
    ids.push_back(kRows - 1);
    ids.push_back(ids.front()); // a repeat
    std::vector<std::uint8_t> expected;
    for (const auto id : ids) {
        expected.insert(expected.end(), payload.begin() + static_cast<std::ptrdiff_t>(id * kRowBytes),
                        payload.begin() + static_cast<std::ptrdiff_t>((id + 1) * kRowBytes));
    }
    for (const auto residency : {NgramResidency::Disk, NgramResidency::Ram})
        for (const NgramTableLayout* table : {&layout, &split_layout}) {
            const NgramTableReader reader(*table, residency);
            std::vector<std::uint8_t> out(ids.size() * kRowBytes);
            reader.read_rows(ids, out);
            require(out == expected, "rows differ from the file");
            bool refused = false;
            try {
                const std::uint64_t past = kRows;
                std::vector<std::uint8_t> one(kRowBytes);
                reader.read_rows(std::span(&past, 1), one);
            } catch (const std::out_of_range&) { refused = true; }
            require(refused, "a row past the table is refused");
        }
    bool refused = false;
    try {
        const NgramTableReader reader(
            NgramTableLayout{{{path, kOffset, kRows * kRowBytes}}, kRowBytes, kRows + 1},
            NgramResidency::Disk);
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a file shorter than the table is refused");
    std::filesystem::remove(path);
    std::filesystem::remove(second);
    return 0;
}

} // namespace

int main() {
    try {
        const int result = run();
        std::cout << "PASS qwen4_exp n-gram table reader\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
