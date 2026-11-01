#pragma once

#include "core/weight_view.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::artifact {

// A piece of a stored matrix that a device object holds on its own: rows [row_begin, row_end) and,
// of each, the columns of every range in order, side by side. A vector is one row. A tensor-parallel
// rank holds such pieces of the projections it computes (models/qwen3_5/load/text.cpp).
//
// Contiguous matrices and vectors, and ggml blocks with column bounds on whole blocks: in both, a
// piece of a row is a run of that row's bytes, so the piece is a gather of byte runs.
struct ObjectSelection {
    std::uint64_t row_begin = 0;
    std::uint64_t row_end   = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> columns;
    // INT32 vectors only: subtracted from every selected value at upload, so a piece of an index
    // vector can index a piece of what it indexes.
    std::int32_t rebase = 0;

    bool operator==(const ObjectSelection&) const = default;
};

// One run of a selection: `bytes` from `source` (an offset in the stored object) to `destination`
// (an offset in the selected object).
struct SelectedRun {
    std::uint64_t source      = 0;
    std::uint64_t bytes       = 0;
    std::uint64_t destination = 0;
};

// The geometry the selected piece takes. Throws ArtifactError for a selection outside the object,
// an empty or unordered range, a layout no byte gather can cut, or a ggml column bound inside a
// block.
[[nodiscard]] WeightGeometry selected_geometry(const WeightGeometry& stored,
                                               const ObjectSelection& selection);

// The byte runs that copy the piece out of the stored object, in destination order. Whole rows that
// follow each other are one run.
[[nodiscard]] std::vector<SelectedRun> selected_runs(const WeightGeometry& stored,
                                                     const ObjectSelection& selection);

} // namespace ninfer::artifact
