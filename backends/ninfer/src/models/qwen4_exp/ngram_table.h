#pragma once

// Row reads from the Qwen3.8-Flash-Next n-gram table. By default the table stays in its files and
// the rows each pass addresses are read from there, `depth` at once: positioned reads through the
// OS page cache (NgramIo::Buffered), aligned reads past it (Direct), or copies out of a read-only
// mapping (Mapped). Disk caches recently read rows within its RAM budget. The Ram residency reads
// the whole table into RAM at startup; RamHot reads the
// rows a hot-row profile ranks most frequent, as many as a budget holds, and reads the rest from
// the files. Every mode returns the files' bytes.

#include "core/file_read_queue.h"
#include "models/qwen4_exp/read_pool.h"
#include "ninfer/model_enums.h"
#include "ninfer/observability/resources.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

// One file's share of the table's bytes; a table split across an artifact's part files has one
// per part, in order, and a row may straddle two of them.
struct NgramTableSegment {
    std::filesystem::path path;
    std::uint64_t file_offset = 0; // where the segment's first table byte is in the file
    std::uint64_t bytes       = 0;
};

struct NgramTableLayout {
    std::vector<NgramTableSegment> segments; // the table's bytes, in order
    std::uint32_t row_bytes = 0;
    std::uint64_t rows      = 0; // addressable rows
};

struct NgramReadOptions {
    NgramResidency residency = NgramResidency::Disk;
    NgramIo io               = NgramIo::Buffered;
    // Disk: cached rows and their index, zero disables caching. RamHot: resident rows and index.
    std::uint64_t budget_bytes = 0;
    // RamHot: most frequent first; the leading rows that fit become resident.
    std::vector<std::uint32_t> hot_rows;
    bool lock           = false; // keep the resident rows in physical memory
    std::uint32_t depth = 64;    // reads in flight
};

class NgramTableReader {
public:
    // Opens the files and checks that they hold every row; Ram and RamHot read their rows now.
    NgramTableReader(NgramTableLayout layout, const NgramReadOptions& options);
    ~NgramTableReader();
    NgramTableReader(const NgramTableReader&)            = delete;
    NgramTableReader& operator=(const NgramTableReader&) = delete;

    [[nodiscard]] const NgramTableLayout& layout() const noexcept { return layout_; }

    [[nodiscard]] const NgramReadOptions& options() const noexcept { return options_; }

    // Rows held in RAM and allocated capacity including the index; Disk fills on demand.
    [[nodiscard]] std::uint64_t resident_rows() const noexcept {
        return resident_rows_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;

    // Physical reader selected on this OS; a restricted Linux host may disable io_uring.
    [[nodiscard]] const char* io_backend() const noexcept;

    // Starts copying the rows `row_ids` addresses, in order, into `out` (row_ids.size() *
    // row_bytes); the resident rows are copied before it returns. Both spans stay untouched by
    // the caller until wait(). One batch at a time. A row id past the table throws
    // std::out_of_range.
    void submit(std::span<const std::uint64_t> row_ids, std::span<std::uint8_t> out);
    // Waits for the batch submit() started, reading with the pool, and rethrows a failed read as
    // std::runtime_error.
    void wait();
    // submit() and wait().
    void read_rows(std::span<const std::uint64_t> row_ids, std::span<std::uint8_t> out);

    // Advisory lookahead: OS hints for Linux buffered/mapped and Windows mapped reads;
    // up to depth direct reads (also Windows buffered reads) into
    // bounded staging. Hints do not change demand counters or admit rows to the resident cache.
    // A batch of more than kInlineHints page ranges is hinted by a worker thread and dropped
    // while the worker still hints the previous one. Like submit/wait, called by the reader's
    // owner, never concurrently with them.
    void prefetch(std::span<const std::uint64_t> row_ids);
    static constexpr std::size_t kInlineHints = 256;

    // The counters of every finished batch; any thread may read them.
    [[nodiscard]] NgramTableStats counters() const;

private:
    struct File;
    class Memory;
    class RowCache;
    class Lookahead;
    class Hints;

    // Reads `bytes` of the table at table offset `offset`, across segments, from the files.
    void read(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const;
    // Copies [first, first + count) table rows into `destination` for a residency's load.
    void load(std::uint64_t first, std::uint64_t count, std::uint8_t* destination) const;
    void load_resident();
    void load_hot();
    [[nodiscard]] std::int64_t hot_slot(std::uint64_t row) const noexcept;

    NgramTableLayout layout_;
    NgramReadOptions options_;
    std::vector<std::unique_ptr<File>> files_; // one per segment
    std::vector<std::uint64_t> starts_;        // table offset of each segment
    std::unique_ptr<Memory> resident_;         // Ram: the table; RamHot: the hot rows by row id
    std::unique_ptr<RowCache> cache_;          // Disk: bounded rows, populated after successful reads
    std::atomic<std::uint64_t> resident_rows_{0};
    // RamHot: bit per table row, and the hot rows before each 512-row block.
    std::vector<std::uint64_t> hot_bits_;
    std::vector<std::uint32_t> hot_blocks_;
    std::unique_ptr<ReadPool> pool_;
    std::unique_ptr<FileReadQueue> queue_;
    std::vector<QueuedFileRead> reads_;
    std::unique_ptr<Lookahead> lookahead_;
    std::unique_ptr<Hints> hints_;

    // The batch in flight: the rows the files serve (index into the batch), and its start.
    std::vector<std::uint32_t> misses_;
    std::vector<std::uint32_t> admissions_;
    std::span<const std::uint64_t> batch_ids_;
    std::span<std::uint8_t> batch_out_;
    std::chrono::steady_clock::time_point batch_start_;
    bool pending_ = false;

    std::atomic<std::uint64_t> rows_{0}, resident_hits_{0}, batches_{0}, latency_ns_{0};
    std::array<std::atomic<std::uint64_t>, NgramTableStats::kLatencyBuckets> histogram_{};
};

} // namespace ninfer::models::qwen4_exp
