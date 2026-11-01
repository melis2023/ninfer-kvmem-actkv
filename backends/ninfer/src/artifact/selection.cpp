#include "artifact/selection.h"

#include "artifact/schema.h"

namespace ninfer::artifact {
namespace {

// A stored object as rows of bytes: a vector is one row.
struct RowBytes {
    std::uint64_t rows        = 0;
    std::uint64_t columns     = 0;
    std::uint64_t row_bytes   = 0;
    std::uint64_t column_unit = 1; // columns per indivisible run of bytes
};

RowBytes row_bytes(const WeightGeometry& stored) {
    if (stored.shape.empty() || stored.shape.size() > 2) {
        throw ArtifactError("a selection needs a vector or a matrix");
    }
    RowBytes out;
    out.rows    = stored.shape.size() == 1 ? 1 : stored.shape[0];
    out.columns = stored.shape.back();
    if (stored.layout == QuantLayout::Contiguous) {
        out.row_bytes = stored.shape.size() == 1 ? stored.bytes : stored.code_bytes_per_row;
    } else if (stored.layout == QuantLayout::GgufBlocks) {
        out.row_bytes   = stored.code_bytes_per_row;
        out.column_unit = stored.group_size;
    } else {
        throw ArtifactError("a selection cuts only contiguous matrices and ggml blocks");
    }
    if (out.rows == 0 || out.columns == 0 || out.row_bytes % (out.columns / out.column_unit) != 0) {
        throw ArtifactError("a selection needs whole runs of bytes per column unit");
    }
    return out;
}

void check(const RowBytes& stored, const ObjectSelection& selection) {
    if (selection.row_begin >= selection.row_end || selection.row_end > stored.rows ||
        selection.columns.empty()) {
        throw ArtifactError("a selection names no rows, rows outside the object or no columns");
    }
    std::uint64_t previous_end = 0;
    for (const auto& [begin, end] : selection.columns) {
        if (begin >= end || end > stored.columns || begin < previous_end) {
            throw ArtifactError("selection columns must be ordered, disjoint and inside the object");
        }
        if (begin % stored.column_unit != 0 || end % stored.column_unit != 0) {
            throw ArtifactError("selection columns must fall on whole ggml blocks");
        }
        previous_end = end;
    }
}

std::uint64_t selected_columns(const ObjectSelection& selection) {
    std::uint64_t columns = 0;
    for (const auto& [begin, end] : selection.columns) {
        columns = checked_add(columns, end - begin, "selected columns");
    }
    return columns;
}

} // namespace

WeightGeometry selected_geometry(const WeightGeometry& stored, const ObjectSelection& selection) {
    const RowBytes layout = row_bytes(stored);
    check(layout, selection);
    if (selection.rebase != 0 &&
        (stored.format != QType::INT32 || stored.layout != QuantLayout::Contiguous ||
         stored.shape.size() != 1)) {
        throw ArtifactError("only a piece of an INT32 vector is rebased");
    }
    const std::uint64_t columns = selected_columns(selection);
    if (stored.shape.size() == 1) {
        const std::uint64_t shape[] = {columns};
        return weight_geometry(stored.format, stored.layout, shape, stored.divisor_count);
    }
    const std::uint64_t shape[] = {selection.row_end - selection.row_begin, columns};
    return weight_geometry(stored.format, stored.layout, shape, stored.divisor_count);
}

std::vector<SelectedRun> selected_runs(const WeightGeometry& stored,
                                       const ObjectSelection& selection) {
    const RowBytes layout = row_bytes(stored);
    check(layout, selection);
    const std::uint64_t unit_bytes = layout.row_bytes / (layout.columns / layout.column_unit);
    const auto column_bytes        = [&](std::uint64_t column) {
        return checked_mul(column / layout.column_unit, unit_bytes, "selection column bytes");
    };
    const std::uint64_t rows = selection.row_end - selection.row_begin;
    std::vector<SelectedRun> out;
    if (selection.columns.size() == 1 && selection.columns.front().first == 0 &&
        selection.columns.front().second == layout.columns) {
        out.push_back({.source      = checked_mul(selection.row_begin, layout.row_bytes, "row"),
                       .bytes       = checked_mul(rows, layout.row_bytes, "rows"),
                       .destination = 0});
        return out;
    }
    const std::uint64_t destination_row = column_bytes(selected_columns(selection));
    out.reserve(static_cast<std::size_t>(rows * selection.columns.size()));
    for (std::uint64_t row = 0; row < rows; ++row) {
        const std::uint64_t source_row =
            checked_mul(selection.row_begin + row, layout.row_bytes, "selection row");
        std::uint64_t destination = checked_mul(row, destination_row, "selection row");
        for (const auto& [begin, end] : selection.columns) {
            const std::uint64_t bytes = column_bytes(end) - column_bytes(begin);
            out.push_back({.source      = source_row + column_bytes(begin),
                           .bytes       = bytes,
                           .destination = destination});
            destination += bytes;
        }
    }
    return out;
}

} // namespace ninfer::artifact
