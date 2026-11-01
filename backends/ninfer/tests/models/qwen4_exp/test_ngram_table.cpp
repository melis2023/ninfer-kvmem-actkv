// NgramTableReader returns the file's bytes for the rows a batch addresses in every residency (the
// file, all of it in RAM, a profile's hot rows in RAM and the rest from the file) and every I/O
// mode (buffered, direct, mapped), from a table in one file or split across two inside a row, and
// counts the rows RAM served. It refuses a short file, a row past the table, a hot-row budget below
// its index, and a profile naming a row twice or past the table; a row read that fails raises
// FileReadError. A hot-row profile reads back as written and is refused for other hash constants.
// Hints, including the batches of many pages a worker thread issues, leave demand reads unchanged.
#include "core/file_read_error.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/ngram_profile.h"
#include "models/qwen4_exp/ngram_table.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Error, class F>
void refused(F&& f, const std::string& what) {
    try {
        f();
    } catch (const Error&) { return; }
    throw std::runtime_error("not refused: " + what);
}

constexpr std::uint32_t kRowBytes = 162;
constexpr std::uint64_t kRows     = 5000;

bool stages_hints(NgramIo io) {
#ifdef _WIN32
    return io == NgramIo::Direct || io == NgramIo::Buffered;
#else
    return io == NgramIo::Direct;
#endif
}

// The bytes the hot set's index takes for kRows rows (one bit per row, a u32 per 512 rows).
constexpr std::uint64_t kIndexBytes = (kRows + 63) / 64 * 8 + (kRows + 511) / 512 * 4;

int run() {
    // The build directory rather than a temporary one, which is often a RAM file system that
    // refuses direct I/O.
    const auto directory = std::filesystem::current_path();
    const auto path =
        directory / ("ninfer_ngram_table_" + std::to_string(std::random_device{}()) + ".bin");
    constexpr std::uint64_t kOffset = 4096 + 17;
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
    ids.push_back(2500);        // the row the split cuts
    ids.push_back(ids.front()); // a repeat
    std::vector<std::uint8_t> expected;
    for (const auto id : ids) {
        expected.insert(expected.end(), payload.begin() + static_cast<std::ptrdiff_t>(id * kRowBytes),
                        payload.begin() + static_cast<std::ptrdiff_t>((id + 1) * kRowBytes));
    }
    // A profile: every row in a shuffled order, of which a budget keeps the first 1,700.
    std::vector<std::uint32_t> profile(kRows);
    std::iota(profile.begin(), profile.end(), 0U);
    std::shuffle(profile.begin(), profile.end(), random);
    constexpr std::uint64_t kHot = 1700;
    std::vector<bool> hot(kRows, false);
    for (std::uint64_t i = 0; i < kHot; ++i) hot[profile[i]] = true;
    std::uint64_t hot_reads = 0;
    for (const auto id : ids) hot_reads += hot[id];

    bool direct_skipped = false;
    for (const auto io : {NgramIo::Buffered, NgramIo::Direct, NgramIo::Mapped})
        for (const auto residency :
             {NgramResidency::Disk, NgramResidency::Ram, NgramResidency::RamHot})
            for (const NgramTableLayout* table : {&layout, &split_layout}) {
                NgramReadOptions options{.residency = residency, .io = io, .depth = 5};
                if (residency == NgramResidency::RamHot) {
                    options.budget_bytes = kIndexBytes + kHot * kRowBytes + kRowBytes - 1;
                    options.hot_rows     = profile;
                }
                const std::string name = "io " + std::to_string(int(io)) + ", residency " +
                                         std::to_string(int(residency)) + ", " +
                                         std::to_string(table->segments.size()) + " file(s)";
                std::unique_ptr<NgramTableReader> reader;
                try {
                    reader = std::make_unique<NgramTableReader>(*table, options);
                } catch (const std::runtime_error& error) {
                    if (io == NgramIo::Direct &&
                        std::string(error.what()).find("without direct I/O") != std::string::npos) {
                        direct_skipped = true;
                        continue;
                    }
                    throw;
                }
                const std::uint64_t resident = residency == NgramResidency::Ram      ? kRows
                                               : residency == NgramResidency::RamHot ? kHot
                                                                                     : 0;
                require(reader->resident_rows() == resident, name + ": resident rows");
                std::cout << name << ": " << reader->io_backend() << '\n';
                for (int pass = 0; pass < 2; ++pass) {
                    std::vector<std::uint8_t> out(ids.size() * kRowBytes);
                    reader->prefetch(ids);
                    if (pass == 0) {
                        reader->read_rows(ids, out);
                    } else {
                        reader->submit(ids, out);
                        refused<std::logic_error>([&] { reader->submit(ids, out); },
                                                   name + ": overlapping batch");
                        reader->wait();
                        reader->wait(); // completed batches do not leave native I/O pending
                    }
                    require(out == expected, name + ": rows differ from the file");
                }
                const NgramTableStats stats = reader->counters();
                const std::uint64_t served  = residency == NgramResidency::Ram      ? ids.size()
                                              : residency == NgramResidency::RamHot ? hot_reads
                                                                                    : 0;
                const bool staged = stages_hints(io) && residency != NgramResidency::Ram;
                require(stats.rows == 2 * ids.size() && stats.batches == 2 &&
                            stats.resident_rows >= 2 * served &&
                            stats.resident_rows <= (staged ? 2 * ids.size() : 2 * served),
                        name + ": counters");
                const std::uint64_t past = kRows;
                std::vector<std::uint8_t> one(kRowBytes);
                refused<std::out_of_range>([&] { reader->read_rows(std::span(&past, 1), one); },
                                           name + ": a row past the table");
                refused<std::out_of_range>([&] { reader->prefetch(std::span(&past, 1)); },
                                           name + ": prefetch past the table");
            }
    if (direct_skipped) { std::cout << "SKIP direct I/O: the file system has none\n"; }

    // A small budget forces eviction rather than relying on enough room for the table. Exercise
    // duplicate reads, the split row, concurrent misses, and reuse after the files served them.
    for (const auto io : {NgramIo::Buffered, NgramIo::Direct, NgramIo::Mapped}) {
        if (io == NgramIo::Direct && direct_skipped) { continue; }
        for (const auto budget : {std::uint64_t{1}, std::uint64_t{1024}, std::uint64_t{8192}}) {
            NgramTableReader reader(split_layout, {.io = io, .budget_bytes = budget, .depth = 5});
            require(reader.resident_rows() == 0 && reader.resident_bytes() <= budget,
                    "disk cache starts empty within its budget");
            const std::vector<std::uint64_t> repeated{0, 2500, kRows - 1, 7, 2500};
            std::vector<std::uint8_t> small(repeated.size() * kRowBytes);
            reader.prefetch(repeated);
            require(reader.counters().rows == 0 && reader.resident_rows() == 0,
                    "prefetch neither counts a read nor admits a row");
            for (int pass = 0; pass < 2; ++pass) {
                reader.read_rows(repeated, small);
                for (std::size_t i = 0; i < repeated.size(); ++i) {
                    require(std::equal(small.begin() + i * kRowBytes,
                                       small.begin() + (i + 1) * kRowBytes,
                                       payload.begin() + repeated[i] * kRowBytes),
                            "cached split/duplicate row differs from file");
                }
            }
            const auto minimum_hits = budget == 1 ? 0 : repeated.size();
            require(reader.counters().resident_rows >= minimum_hits &&
                        reader.counters().resident_rows <=
                            (stages_hints(io) ? 2 * repeated.size() : minimum_hits),
                    "a repeated batch hits the cache when its working set fits");
            for (int pass = 0; pass < 3; ++pass) {
                std::vector<std::uint8_t> out(expected.size());
                reader.prefetch(ids);
                reader.read_rows(ids, out);
                require(out == expected, "disk cache eviction changes row bytes");
                require(reader.resident_rows() * kRowBytes <= reader.resident_bytes() &&
                            reader.resident_bytes() <= budget,
                        "disk cache exceeds its budget after eviction");
            }
        }
    }

    for (const auto io : {NgramIo::Direct, NgramIo::Buffered}) {
        if (!stages_hints(io) || (io == NgramIo::Direct && direct_skipped)) { continue; }
        // A completed staged hint serves exact bytes without a resident cache. Poll only for
        // the asynchronous completion, using observable demand hits, with a bounded deadline.
        NgramTableReader reader(split_layout, {.io = io, .depth = 3});
        const std::vector<std::uint64_t> hinted{0, 2500, kRows - 1};
        std::vector<std::uint8_t> out(hinted.size() * kRowBytes);
        reader.prefetch(hinted);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        for (;;) {
            const auto before = reader.counters().resident_rows;
            reader.read_rows(hinted, out);
            for (std::size_t i = 0; i < hinted.size(); ++i) {
                require(std::equal(out.begin() + i * kRowBytes, out.begin() + (i + 1) * kRowBytes,
                                   payload.begin() + hinted[i] * kRowBytes), "staged hint row bytes");
            }
            if (reader.counters().resident_rows - before == hinted.size()) { break; }
            require(std::chrono::steady_clock::now() < deadline, "staged hints never became usable");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(reader.resident_rows() == 0 && reader.resident_bytes() == 0,
                "lookahead staging was admitted without a demand-cache budget");
        // Hint lists may exceed the I/O depth; truncation changes only which reads can hit.
        reader.prefetch(ids);
        std::vector<std::uint8_t> all(expected.size());
        reader.read_rows(ids, all);
        require(all == expected, "bounded staged hints changed demand output");
        // Destruction drains a still-running speculative read before closing files or buffers.
        reader.prefetch(hinted);
    }

    {
        // OS hints of more than kInlineHints page ranges run on a worker thread, which drops a
        // batch that arrives while it still hints the previous one; destruction waits for it.
        // Rows of a page each make every row its own range. Hints never change demand output.
        constexpr std::uint32_t kPageRow = 4096;
        constexpr std::uint64_t kPageRows = NgramTableReader::kInlineHints * 2 + 3;
        const auto pages = std::filesystem::path(path.string() + ".pages");
        std::vector<std::uint8_t> page_payload(kPageRows * kPageRow);
        for (auto& byte : page_payload) byte = static_cast<std::uint8_t>(random());
        {
            std::ofstream file(pages, std::ios::binary);
            file.write(reinterpret_cast<const char*>(page_payload.data()),
                       static_cast<std::streamsize>(page_payload.size()));
        }
        const NgramTableLayout page_layout{{{pages, 0, kPageRows * kPageRow}}, kPageRow, kPageRows};
        std::vector<std::uint64_t> all(kPageRows);
        std::iota(all.begin(), all.end(), 0U);
        std::reverse(all.begin(), all.end());
        for (const auto io : {NgramIo::Buffered, NgramIo::Mapped}) {
            if (stages_hints(io)) { continue; }
            for (int pass = 0; pass < 3; ++pass) {
                NgramTableReader reader(page_layout, {.io = io, .depth = 4});
                reader.prefetch(all);
                reader.prefetch(all);
                require(reader.counters().rows == 0, "worker hints count no read");
                std::vector<std::uint8_t> out(page_payload.size());
                reader.read_rows(all, out);
                for (std::size_t i = 0; i < all.size(); ++i) {
                    require(std::equal(out.begin() + i * kPageRow, out.begin() + (i + 1) * kPageRow,
                                       page_payload.begin() + all[i] * kPageRow),
                            "rows read beside worker hints differ from the file");
                }
                if (pass == 2) { reader.prefetch(all); } // destroyed while the worker may hint
            }
        }
        std::filesystem::remove(pages);
    }

#ifndef _WIN32
    // A failed multi-row batch must not admit even the rows that happened to finish first.
    // Windows opens immutable artifacts without write sharing; mapped truncation is not valid.
    for (const auto io : {NgramIo::Buffered, NgramIo::Direct}) {
        if (io == NgramIo::Direct && direct_skipped) { continue; }
        NgramTableReader reader(layout, {.io = io, .budget_bytes = 8192, .depth = 2});
        std::filesystem::resize_file(path, kOffset + 2 * kRowBytes);
        const std::vector<std::uint64_t> rows{0, kRows - 1};
        std::vector<std::uint8_t> out(rows.size() * kRowBytes);
        reader.prefetch(rows);
        refused<FileReadError>([&] { reader.read_rows(rows, out); }, "read after truncation");
        require(reader.resident_rows() == 0, "failed batch admitted partial data");
        {
            std::ofstream file(path, std::ios::binary | std::ios::in);
            file.seekp(kOffset);
            file.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        }
        reader.read_rows(rows, out);
        require(reader.counters().resident_rows == 0 && reader.resident_rows() == rows.size(),
                "successful retry did not read and admit both rows");
        reader.read_rows(rows, out);
        require(reader.counters().resident_rows == rows.size(), "successful retry is not cached");
        for (std::size_t i = 0; i < rows.size(); ++i) {
            require(std::equal(out.begin() + i * kRowBytes, out.begin() + (i + 1) * kRowBytes,
                               payload.begin() + rows[i] * kRowBytes), "retry row bytes");
        }
    }
#endif

    refused<std::runtime_error>(
        [&] {
            const NgramTableReader reader(
                NgramTableLayout{{{path, kOffset, kRows * kRowBytes}}, kRowBytes, kRows + 1}, {});
        },
        "a file shorter than the table");
    const auto hot_reader = [&](std::uint64_t budget, std::vector<std::uint32_t> rows) {
        const NgramTableReader reader(layout, {.residency    = NgramResidency::RamHot,
                                               .budget_bytes = budget,
                                               .hot_rows     = std::move(rows)});
    };
    refused<std::invalid_argument>([&] { hot_reader(kIndexBytes, profile); },
                                   "a hot-row budget below its index");
    refused<std::invalid_argument>([&] { hot_reader(1 << 20, {3, 9, 3}); }, "a row named twice");
    refused<std::invalid_argument>([&] { hot_reader(1 << 20, {3, std::uint32_t(kRows)}); },
                                   "a hot row past the table");
    refused<std::invalid_argument>([&] { const NgramTableReader reader(layout, {.lock = true}); },
                                   "locking the disk residency");
    refused<std::invalid_argument>([&] { const NgramTableReader reader(layout, {.depth = 0}); },
                                   "no reads in flight");
    std::filesystem::remove(path);
    std::filesystem::remove(second);

    // A profile reads back as written and belongs to the hash it was counted for.
    const NgramHashConstants constants =
        derive_ngram_hash_constants(NgramHashSpec{.vocab_size      = 248320,
                                                  .ngram_size      = 3,
                                                  .heads_per_ngram = 8,
                                                  .vocab_base      = 20000000,
                                                  .divisible_by    = 128,
                                                  .seed            = 1234});
    const NgramProfile written{.table_rows  = constants.rows,
                               .fingerprint = ngram_hash_fingerprint(constants),
                               .tokens      = 12345,
                               .rows        = {7, 1, 99, 320001445}};
    const auto file =
        directory / ("ninfer_ngram_profile_" + std::to_string(std::random_device{}()) + ".bin");
    write_ngram_profile(file, written);
    const NgramProfile read = read_ngram_profile(file);
    require(read.table_rows == written.table_rows && read.fingerprint == written.fingerprint &&
                read.tokens == written.tokens && read.rows == written.rows,
            "the profile reads back as written");
    check_ngram_profile(read, constants, file);
    std::ifstream input(file, std::ios::binary);
    std::vector<std::byte> encoded(40 + written.rows.size() * 4 + 1);
    input.read(reinterpret_cast<char*>(encoded.data() + 1),
               static_cast<std::streamsize>(encoded.size() - 1));
    input.close();
    const auto profile_payload = std::span<const std::byte>(encoded).subspan(1);
    require(decode_ngram_profile(profile_payload, "embedded profile").rows == written.rows,
            "an unaligned embedded profile preserves frequency order");
    auto malformed = encoded;
    std::fill(malformed.begin() + 33, malformed.begin() + 41, std::byte{0xff});
    refused<std::runtime_error>([&] {
        (void)decode_ngram_profile(std::span<const std::byte>(malformed).subspan(1), "bad count");
    }, "an overflowing embedded row count");
    refused<std::runtime_error>([&] {
        (void)decode_ngram_profile(profile_payload.first(12), "short header");
    }, "an incomplete embedded header");
    malformed = encoded;
    malformed[1] = std::byte{0};
    refused<std::runtime_error>([&] {
        (void)decode_ngram_profile(std::span<const std::byte>(malformed).subspan(1), "bad magic");
    }, "an invalid embedded magic");
    for (const auto row : {written.rows[0], static_cast<std::uint32_t>(constants.rows)}) {
        auto invalid = written;
        invalid.rows.back() = row;
        refused<std::invalid_argument>([&] { check_ngram_profile(invalid, constants, file); },
                                       "a duplicate or out-of-range profile row");
    }
    NgramHashConstants other = constants;
    other.multipliers[1] += 2;
    refused<std::invalid_argument>([&] { check_ngram_profile(read, other, file); },
                                   "a profile of other hash constants");
    std::filesystem::resize_file(file, std::filesystem::file_size(file) - 2);
    refused<std::runtime_error>([&] { (void)read_ngram_profile(file); }, "a truncated profile");
    std::filesystem::remove(file);
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
