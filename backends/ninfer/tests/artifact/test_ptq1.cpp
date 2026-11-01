#include "artifact/formats.h"
#include "core/weight_view.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ninfer;
    const auto format = artifact::parse_format("ptq1_g128_fp16");
    if (format != QType::PTQ1_G128_FP16 || artifact::format_name(format) != "ptq1_g128_fp16") return 1;
    const auto geometry = weight_geometry(format, QuantLayout::RowSplit,
                                          std::array<std::uint64_t, 2>{7, 5120});
    // Three independently aligned planes: 6720 base bytes, 560 tail bytes and 560 scale bytes.
    if (geometry.code_bytes != 6720 || geometry.high_offset != 6912 || geometry.high_bytes != 560 ||
        geometry.scale_offset != 7680 || geometry.bytes != 8240) return 1;
    std::vector<std::byte> payload(geometry.bytes);
    const WeightParent parent{geometry, payload.data()};
    const WeightView rows{{2, 5120}, {{&parent, 3 * 5120, 5 * 5120}}};
    const Weight weight = native_weight(rows);
    if (weight.n != 2 || weight.k != 5120 || weight.group != 128 ||
        weight.qdata != payload.data() + 3 * 960 ||
        weight.qhigh != payload.data() + 6912 + 3 * 80 ||
        weight.scales != payload.data() + 7680 + 3 * 80 || weight.high_plane_bytes != 560) return 1;
    for (auto layout : {QuantLayout::RowSplit, QuantLayout::RowSplitPanel}) {
        bool rejected = false;
        try {
            (void)weight_geometry(format, layout, std::array<std::uint64_t, 2>{7, 129});
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 1;
    }
    bool rejected_panel = false;
    try {
        (void)weight_geometry(format, QuantLayout::RowSplitPanel,
                              std::array<std::uint64_t, 2>{7, 5120});
    } catch (const std::invalid_argument&) { rejected_panel = true; }
    if (!rejected_panel) return 1;
    std::cout << "OK PTQ1 format, geometry and native row binding\n";
    return 0;
}
