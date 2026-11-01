// Pieces of stored objects (artifact/selection.h), the weights a tensor-parallel rank uploads. The
// oracle is exact: the piece's bytes, gathered by the runs from a stored object whose every byte
// names its row and column unit, must be exactly the selected units in row-major order, and the
// piece's geometry must be what a stored object of the piece's shape would have.

#include "artifact/selection.h"

#include "artifact/schema.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

WeightGeometry stored(QType format, QuantLayout layout, std::vector<std::uint64_t> shape) {
    return weight_geometry(format, layout, std::span<const std::uint64_t>(shape));
}

// Gathers the piece through `runs` from a stored object whose unit (u, row) holds the bytes
// `row * 100000 + u` in a fixed pattern, then checks each piece unit against its source.
void check_gather(const WeightGeometry& whole, const ObjectSelection& selection,
                  std::uint64_t rows, std::uint64_t units, std::uint64_t unit_bytes,
                  std::uint64_t column_unit) {
    std::vector<std::uint32_t> source(rows * units * unit_bytes);
    for (std::uint64_t r = 0; r < rows; ++r) {
        for (std::uint64_t u = 0; u < units; ++u) {
            for (std::uint64_t b = 0; b < unit_bytes; ++b) {
                source[(r * units + u) * unit_bytes + b] =
                    static_cast<std::uint32_t>((r * 100000 + u) * 64 + b);
            }
        }
    }
    const auto piece = selected_geometry(whole, selection);
    std::uint64_t picked_units = 0;
    for (const auto& [begin, end] : selection.columns) { picked_units += (end - begin) / column_unit; }
    const std::uint64_t piece_rows = selection.row_end - selection.row_begin;
    std::vector<std::uint32_t> got(piece_rows * picked_units * unit_bytes, 0xffffffffU);
    std::uint64_t covered = 0;
    for (const auto& run : selected_runs(whole, selection)) {
        require(run.source % sizeof(std::uint32_t) == 0 && run.bytes % sizeof(std::uint32_t) == 0,
                "runs stay on the test's 4-byte cells");
        for (std::uint64_t i = 0; i < run.bytes / 4; ++i) {
            got[run.destination / 4 + i] = source[run.source / 4 + i];
        }
        covered += run.bytes;
    }
    require(covered == got.size() * 4, "the runs cover the piece exactly once");
    for (std::uint64_t r = 0; r < piece_rows; ++r) {
        std::uint64_t unit = 0;
        for (const auto& [begin, end] : selection.columns) {
            for (std::uint64_t u = begin / column_unit; u < end / column_unit; ++u, ++unit) {
                for (std::uint64_t b = 0; b < unit_bytes; ++b) {
                    const std::uint64_t at = (r * picked_units + unit) * unit_bytes + b;
                    require(got[at] == source[((selection.row_begin + r) * units + u) * unit_bytes + b],
                            "a piece unit holds its source unit");
                }
            }
        }
    }
    require(piece.bytes == got.size() * 4, "the piece geometry counts the gathered bytes");
}

void contiguous_matrix() {
    // FP32 [6, 8]: one unit is one column of four bytes, cut as a cell of the gather.
    const auto whole = stored(QType::FP32, QuantLayout::Contiguous, {6, 8});
    const ObjectSelection rows_only{.row_begin = 2, .row_end = 5, .columns = {{0, 8}}};
    check_gather(whole, rows_only, 6, 8, 1, 1);
    require(selected_runs(whole, rows_only).size() == 1, "whole rows gather in one run");
    const ObjectSelection columns{.row_begin = 0, .row_end = 6, .columns = {{1, 3}, {5, 8}}};
    check_gather(whole, columns, 6, 8, 1, 1);
    const auto piece = selected_geometry(whole, columns);
    require(piece.shape == std::vector<std::uint64_t>({6, 5}), "a column piece keeps the rows");
}

void vector_object() {
    const auto whole = stored(QType::FP32, QuantLayout::Contiguous, {12});
    const ObjectSelection part{.row_begin = 0, .row_end = 1, .columns = {{4, 8}}};
    check_gather(whole, part, 1, 12, 1, 1);
    require(selected_geometry(whole, part).shape == std::vector<std::uint64_t>({4}),
            "a vector piece is a vector");
}

void ggml_blocks() {
    // Q8_0 [4, 256]: blocks of 32 values in 34 bytes; the test cells are bytes / 2 wide, so the
    // gather check runs on 2-byte multiples via a Q8_0 row of 8 blocks = 272 bytes = 68 cells.
    const auto whole = stored(QType::GGUF_Q8_0, QuantLayout::GgufBlocks, {4, 256});
    require(whole.code_bytes_per_row == 272, "a Q8_0 row of 256 values is 272 bytes");
    const ObjectSelection blocks{.row_begin = 1, .row_end = 3, .columns = {{32, 96}, {192, 256}}};
    const auto runs = selected_runs(whole, blocks);
    require(runs.size() == 4, "two rows of two column ranges");
    require(runs[0].source == 272 + 34 && runs[0].bytes == 68 && runs[0].destination == 0,
            "the first run starts at row 1, block 1");
    require(runs[1].source == 272 + 6 * 34 && runs[1].bytes == 68 && runs[1].destination == 68,
            "the second range follows the first");
    require(runs[2].source == 2 * 272 + 34 && runs[2].destination == 136, "row 2 follows row 1");
    const auto piece = selected_geometry(whole, blocks);
    require(piece.shape == std::vector<std::uint64_t>({2, 128}) && piece.bytes == 2 * 136,
            "the piece is a Q8_0 [2, 128]");
    bool refused = false;
    try {
        (void)selected_runs(whole, {.row_begin = 0, .row_end = 1, .columns = {{16, 64}}});
    } catch (const ArtifactError&) { refused = true; }
    require(refused, "a column bound inside a block is refused");
}

void refusals() {
    const auto whole = stored(QType::FP32, QuantLayout::Contiguous, {6, 8});
    const ObjectSelection bad[] = {
        {.row_begin = 3, .row_end = 3, .columns = {{0, 8}}},
        {.row_begin = 0, .row_end = 7, .columns = {{0, 8}}},
        {.row_begin = 0, .row_end = 2, .columns = {}},
        {.row_begin = 0, .row_end = 2, .columns = {{4, 6}, {2, 3}}},
        {.row_begin = 0, .row_end = 2, .columns = {{0, 9}}},
    };
    for (const auto& selection : bad) {
        bool refused = false;
        try {
            (void)selected_geometry(whole, selection);
        } catch (const ArtifactError&) { refused = true; }
        require(refused, "an empty, outside or unordered selection is refused");
    }
    // Only an INT32 vector's values can be rebased.
    const auto indices = stored(QType::INT32, QuantLayout::Contiguous, {16});
    const ObjectSelection rebased{.row_begin = 0, .row_end = 1, .columns = {{4, 8}}, .rebase = 4};
    require(selected_geometry(indices, rebased).shape == std::vector<std::uint64_t>({4}),
            "an INT32 vector piece can be rebased");
    bool refused = false;
    try {
        (void)selected_geometry(stored(QType::FP32, QuantLayout::Contiguous, {16}), rebased);
    } catch (const ArtifactError&) { refused = true; }
    require(refused, "a rebased FP32 piece is refused");
}

} // namespace

int main() {
    try {
        contiguous_matrix();
        vector_object();
        ggml_blocks();
        refusals();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK artifact selection\n";
    return 0;
}
