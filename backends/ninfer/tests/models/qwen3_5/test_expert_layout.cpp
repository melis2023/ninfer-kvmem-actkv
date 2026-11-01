#include "models/qwen3_5/expert_layout.h"
#include "models/qwen3_5/expert_admission.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen3_5;

namespace {
int failures = 0;
void expect(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
template<class F> void rejected(F&& f) {
    try { f(); ++failures; std::cerr << "FAIL: invalid range accepted\n"; }
    catch (const std::invalid_argument&) {}
    catch (const std::overflow_error&) {}
}

std::uint64_t scale_index(std::uint64_t row, std::uint64_t group, std::uint64_t k) {
    // Independent decode of the on-disk K16M128x4 permutation.
    return ((row / 128) * (k / 64) + group / 4) * 512 +
           (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

void row_split(QType format, std::uint32_t k) {
    constexpr std::uint64_t rows = 512, parent_rows = 3 * rows;
    const unsigned group = format == QType::Q8_G32_FP16 ? 32 : 64;
    const unsigned code_group = 32;
    const unsigned high_group = format == QType::Q5_G64_FP16 ? 8
                              : format == QType::Q6_G64_FP16 ? 16 : 0;
    const auto code_row = k / group * code_group;
    const auto high_row = k / group * high_group;
    const auto scale_row = k / group * 2;
    const auto high_offset = parent_rows * code_row;
    const auto scale_offset = high_offset + parent_rows * high_row;
    const ExpertParentLayout parent{format, QuantLayout::RowSplit, parent_rows, k,
                                     high_offset, scale_offset, 0};
    const auto slot = expert_projection_layout(format, rows, k);
    const auto ranges = expert_projection_ranges(parent, rows, rows, slot);
    std::vector<std::uint8_t> source(scale_offset + parent_rows * scale_row);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::uint8_t>((i * 19 + (i >> 9) * 43) % 251);
    }
    std::vector<std::uint8_t> target(slot.bytes, 0xFD);
    for (const auto& range : ranges) {
        std::memcpy(target.data() + range.destination, source.data() + range.source, range.bytes);
    }
    // Decode every code byte, high byte and FP16 scale bit at both K shapes. This detects a
    // source-row error even when codes, high bits and scales have different physical strides.
    const std::array<std::uint64_t, 3> row_bytes{code_row, high_row, scale_row};
    const std::array<std::uint64_t, 3> original{0, high_offset, scale_offset};
    const std::array<std::uint64_t, 3> relocated{slot.codes, slot.high, slot.scales};
    for (unsigned plane = 0; plane < 3; ++plane) {
        for (std::uint64_t r = 0; r < rows; ++r) {
            for (std::uint64_t c = 0; c < row_bytes[plane]; ++c) {
                if (source[original[plane] + (rows + r) * row_bytes[plane] + c] !=
                    target[relocated[plane] + r * row_bytes[plane] + c]) {
                    expect(false, "row-split relocation changes represented bytes"); return;
                }
            }
        }
    }
    rejected([&] { (void)expert_projection_ranges(parent, parent_rows, 1, slot); });
    auto panel = parent; panel.layout = QuantLayout::RowSplitPanel;
    rejected([&] { (void)expert_projection_ranges(panel, 0, rows, slot); });
}

void nvfp4() {
    constexpr std::uint64_t k = 2048, parent_rows = 3072, rows = 1024, origin = 1024;
    constexpr auto scale_offset = parent_rows * k / 2;
    constexpr auto divisor_offset = scale_offset + parent_rows * k / 16;
    const ExpertParentLayout parent{QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                     parent_rows, k, 0, scale_offset, divisor_offset, 6};
    const auto slot = expert_projection_layout(QType::NVFP4, rows, k, 512);
    std::vector<std::uint8_t> source(divisor_offset + 24);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::uint8_t>((i * 17 + (i >> 7)) % 251);
    }
    std::vector<std::uint8_t> target(slot.bytes, 0xFD);
    for (const auto& range : expert_projection_ranges(parent, origin, rows, slot)) {
        std::memcpy(target.data() + range.destination, source.data() + range.source, range.bytes);
    }
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t group = 0; group < k / 16; ++group) {
            expect(source[scale_offset + scale_index(origin + row, group, k)] ==
                       target[slot.scales + scale_index(row, group, k)],
                   "NVFP4 swizzled scale changed");
        }
        expect(std::memcmp(source.data() + divisor_offset + (origin + row) / 512 * 4,
                            target.data() + slot.divisors + row / 512 * 4, 4) == 0,
               "NVFP4 source divisor changed");
    }
    rejected([&] { (void)expert_projection_ranges(parent, 64, 512, slot); });
    rejected([&] { (void)expert_projection_ranges(parent, origin, 128, slot); });
    auto merged = expert_projection_layout(QType::NVFP4, rows, k, 1024);
    rejected([&] { (void)expert_projection_ranges(parent, origin, rows, merged); });
    auto shared = parent; shared.divisor_count = 1;
    const auto expanded = expert_projection_ranges(shared, origin, rows, slot);
    expect(expanded.size() == 4 && expanded[2].source == expanded[3].source,
           "one NVFP4 parent divisor must be duplicated into both slot row groups");
}
} // namespace

int main() {
    for (const bool automatic_kv : {false, true}) {
        for (const auto cache : {std::optional<std::uint64_t>{},
                                 std::optional<std::uint64_t>{0},
                                 std::optional<std::uint64_t>{1024}}) {
            const auto reserved = native_expert_runtime_reservation(automatic_kv, cache, 8192, 2048);
            // The cache comes first unless KV is explicit or the cache is turned off.
            const auto expected = automatic_kv && (!cache || *cache > 0) ? 2048 : 8192;
            expect(reserved == std::uint64_t(expected),
                   "automatic/explicit KV and cache priority differs");
        }
    }
    for (const auto format : {QType::Q4_G64_FP16, QType::Q5_G64_FP16,
                              QType::Q6_G64_FP16, QType::Q8_G32_FP16}) {
        for (const std::uint32_t k : {512, 2048}) { row_split(format, k); }
    }
    nvfp4();
    rejected([] { (void)expert_projection_layout(QType::GGUF_Q4_0, 1024, 2048); });
    rejected([] { (void)expert_mul(UINT64_MAX, 2); });
    rejected([] { (void)expert_align(UINT64_MAX); });
    if (failures) { return 1; }
    std::cout << "native expert plane relocation checks passed\n";
    return 0;
}
