#pragma once

// Row reads from the Qwen3.8-Flash-Next n-gram table, which by default stays on disk: rows are
// read where the file stores them, through the OS page cache, as the hash addresses them, several
// at once. The RAM residency loads the whole payload once and serves rows from memory.

#include "models/qwen4_exp/read_pool.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

enum class NgramResidency : std::uint8_t { Disk, Ram };

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

class NgramTableReader {
  public:
      // Opens the files and checks that they hold every row; Ram reads the payload now.
      NgramTableReader(NgramTableLayout layout, NgramResidency residency);
      ~NgramTableReader();
      NgramTableReader(const NgramTableReader&)            = delete;
      NgramTableReader& operator=(const NgramTableReader&) = delete;

      [[nodiscard]] const NgramTableLayout& layout() const noexcept { return layout_; }
      [[nodiscard]] NgramResidency residency() const noexcept { return residency_; }

      // Copies the rows `row_ids` addresses, in order, into `out` (row_ids.size() * row_bytes).
      // A row id past the table throws std::out_of_range; a failed read throws std::runtime_error.
      void read_rows(std::span<const std::uint64_t> row_ids, std::span<std::uint8_t> out) const;

  private:
    struct File;
    // Reads `bytes` of the table at table offset `offset`, across segments.
    void read(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const;

    NgramTableLayout layout_;
    NgramResidency residency_;
    std::vector<std::unique_ptr<File>> files_; // one per segment
    std::vector<std::uint64_t> starts_;        // table offset of each segment
    std::vector<std::uint8_t> resident_;
    std::unique_ptr<ReadPool> pool_; // disk residency
};

} // namespace ninfer::models::qwen4_exp
