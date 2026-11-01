#pragma once

#include "core/weight.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Offsets describe stored bytes, not logical elements. Delivery copies each range unchanged.
struct ExpertByteRange {
    std::uint64_t source = 0;
    std::uint64_t destination = 0;
    std::uint64_t bytes = 0;
};

struct ExpertProjectionLayout {
    QType format = QType::Q4_G64_FP16;
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::uint64_t codes = 0, high = 0, scales = 0, divisors = 0;
    std::uint64_t code_bytes = 0, high_bytes = 0, scale_bytes = 0;
    std::uint32_t divisor_rows = 0;
    std::uint64_t bytes = 0;
};

struct ExpertParentLayout {
    QType format;
    QuantLayout layout;
    std::uint64_t rows, columns;
    std::uint64_t high_offset, scale_offset, divisor_offset;
    std::uint64_t divisor_count = 1;
};

inline std::uint64_t expert_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("native expert byte offset overflows");
    }
    return a + b;
}

inline std::uint64_t expert_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("native expert byte count overflows");
    }
    return a * b;
}

inline std::uint64_t expert_align(std::uint64_t n) {
    return expert_add(n, 255) / 256 * 256;
}

// Gate and up retain separate divisors in a joined NVFP4 slot. All supported row spans start
// on complete 128-row scale tiles, so their swizzled scale bytes can be copied without a permute.
inline ExpertProjectionLayout expert_projection_layout(QType format, std::uint32_t rows,
                                                        std::uint32_t columns,
                                                        std::uint32_t divisor_rows = 0) {
    if (!rows || !columns || columns % 128) {
        throw std::invalid_argument("native expert needs positive rows and K divisible by 128");
    }
    ExpertProjectionLayout out;
    out.format = format;
    out.rows = rows;
    out.columns = columns;
    const auto elements = expert_mul(rows, columns);
    switch (format) {
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        out.code_bytes = elements / 2;
        out.high_bytes = format == QType::Q5_G64_FP16 ? elements / 8
                       : format == QType::Q6_G64_FP16 ? elements / 4 : 0;
        out.scale_bytes = elements / 64 * 2;
        break;
    case QType::Q8_G32_FP16:
        out.code_bytes = elements;
        out.scale_bytes = elements / 32 * 2;
        break;
    case QType::NVFP4:
        if (rows % 128 || !divisor_rows || divisor_rows % 128 || rows % divisor_rows) {
            throw std::invalid_argument("native NVFP4 expert needs complete scale/divisor tiles");
        }
        out.code_bytes = elements / 2;
        out.scale_bytes = elements / 16;
        out.divisor_rows = divisor_rows;
        break;
    default:
        throw std::invalid_argument("unsupported native expert format");
    }
    out.high = expert_align(out.code_bytes);
    out.scales = expert_add(out.high, expert_align(out.high_bytes));
    out.divisors = expert_align(expert_add(out.scales, out.scale_bytes));
    out.bytes = out.divisors;
    if (out.divisor_rows) {
        out.bytes = expert_align(expert_add(out.bytes, std::uint64_t(rows / divisor_rows) * 4));
    }
    return out;
}

// A logical projection can span several parents. The caller invokes this once per row part,
// with its destination row origin. NVFP4 tile and divisor boundaries are validated explicitly.
inline std::vector<ExpertByteRange>
expert_projection_ranges(const ExpertParentLayout& parent, std::uint64_t source_row,
                          std::uint64_t rows, const ExpertProjectionLayout& slot,
                          std::uint64_t destination_row = 0) {
    if (!rows || source_row > parent.rows || rows > parent.rows - source_row ||
        destination_row > slot.rows || rows > slot.rows - destination_row ||
        parent.columns != slot.columns || parent.format != slot.format) {
        throw std::invalid_argument("native expert row range exceeds its source or slot");
    }
    const bool nvfp4 = parent.format == QType::NVFP4;
    if ((!nvfp4 && parent.layout != QuantLayout::RowSplit) ||
        (nvfp4 && parent.layout != QuantLayout::BlockScaleK16M128x4)) {
        throw std::invalid_argument("native expert source layout is unsupported");
    }
    const auto code_row = slot.code_bytes / slot.rows;
    const auto high_row = slot.high_bytes / slot.rows;
    const auto scale_row = slot.scale_bytes / slot.rows;
    std::vector<ExpertByteRange> result;
    result.push_back({expert_mul(source_row, code_row),
                      expert_add(slot.codes, expert_mul(destination_row, code_row)),
                      expert_mul(rows, code_row)});
    if (high_row) {
        result.push_back({expert_add(parent.high_offset, expert_mul(source_row, high_row)),
                          expert_add(slot.high, expert_mul(destination_row, high_row)),
                          expert_mul(rows, high_row)});
    }
    result.push_back({expert_add(parent.scale_offset, expert_mul(source_row, scale_row)),
                      expert_add(slot.scales, expert_mul(destination_row, scale_row)),
                      expert_mul(rows, scale_row)});
    if (nvfp4) {
        if (source_row % 128 || destination_row % 128 || rows % 128 ||
            !parent.divisor_count || parent.rows % parent.divisor_count) {
            throw std::invalid_argument("native NVFP4 range splits a scale tile");
        }
        const auto parent_divisor_rows = parent.rows / parent.divisor_count;
        if (!parent_divisor_rows || parent_divisor_rows % 128) {
            throw std::invalid_argument("native NVFP4 source has invalid divisor geometry");
        }
        // A slot may expand one shared parent divisor into several row groups. Copying the
        // scalar more than once preserves its represented bits without interpreting it here.
        if (!slot.divisor_rows || destination_row % slot.divisor_rows ||
            rows % slot.divisor_rows) {
            throw std::invalid_argument("native NVFP4 range splits a slot divisor");
        }
        for (std::uint64_t row = 0; row < rows; row += slot.divisor_rows) {
            const auto source = source_row + row;
            if (slot.divisor_rows > parent_divisor_rows - source % parent_divisor_rows) {
                throw std::invalid_argument("native NVFP4 slot divisor merges source divisors");
            }
            result.push_back({expert_add(parent.divisor_offset,
                                          expert_mul(source / parent_divisor_rows, 4)),
                              expert_add(slot.divisors,
                                          expert_mul((destination_row + row) / slot.divisor_rows, 4)),
                              4});
        }
    }
    return result;
}

} // namespace ninfer::models::qwen3_5
