#include "models/qwen3_5/program/planning/prefill_service_work.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using ninfer::models::qwen3_5::detail::initial_mtp_prefill_extent;
using ninfer::models::qwen3_5::detail::prefill_service_units;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Independent step-by-step execution model: stop at the next grid boundary,
// leave proposal space at the prompt end, and yield at an offered capture.
std::uint64_t executed(std::uint32_t cursor, std::uint32_t end, std::uint32_t chunk,
                       std::uint32_t reserve, std::uint32_t extent, std::uint32_t capture) {
    std::uint64_t units = 0;
    while (cursor < end) {
        auto count = reserve ? chunk - cursor % chunk : chunk;
        if (reserve && end - cursor <= count)
            count = std::min(count, reserve - (extent ? extent - 1U : 0U));
        count = std::min(count, end - cursor);
        if (capture > cursor) count = std::min(count, capture - cursor);
        require(count != 0, "prefill made no progress");
        cursor += count;
        ++units;
    }
    return units;
}
}

int main() {
    try {
        require(initial_mtp_prefill_extent(4, 129, 4096, 1919) == 4, "MTP extent");
        require(initial_mtp_prefill_extent(4, 2, 4096, 1919) == 0, "short output extent");
        require(initial_mtp_prefill_extent(15, 129, 4096, 4095) == 0, "context edge extent");
        require(initial_mtp_prefill_extent(15, 129, 4096, 4092) == 3, "context clipping");
        require(prefill_service_units(0, 1919, 128, 128, 4) == 16,
                "1919-token prompt needs 16 service units, not 15");
        require(prefill_service_units(0, 125, 128, 128, 4) == 1 &&
                prefill_service_units(0, 126, 128, 128, 4) == 2 &&
                prefill_service_units(0, 128, 128, 128, 4) == 2,
                "MTP tail split boundary");
        require(prefill_service_units(0, 1919, 128, 256, 4) == 15,
                "roomy reserve should not split the tail");
        require(prefill_service_units(63, 129, 128, 128) == 2,
                "reused/query suffix must retain canonical boundaries");
        for (std::uint32_t chunk : {128U, 256U}) {
            for (std::uint32_t reserve : {0U, chunk, chunk + 64U}) {
                for (std::uint32_t extent : {0U, 1U, 4U, 15U}) {
                    for (std::uint32_t end = 1; end <= 3 * chunk; ++end) {
                        for (std::uint32_t begin : {0U, end / 2, end - 1U}) {
                            const auto projected = prefill_service_units(begin, end, chunk, reserve, extent);
                            require(projected == executed(begin, end, chunk, reserve, extent, 0),
                                    "projection disagrees with worker-visible steps");
                            for (std::uint32_t capture : {begin + 1U, (begin + end) / 2, end - 1U}) {
                                if (capture <= begin || capture >= end) continue;
                                const auto with_capture =
                                    prefill_service_units(begin, capture, chunk, reserve) +
                                    prefill_service_units(capture, end, chunk, reserve) + projected -
                                    prefill_service_units(begin, end, chunk, reserve);
                                require(with_capture >= executed(begin, end, chunk, reserve, extent, capture),
                                        "capture before prompt end exhausted the service budget");
                            }
                        }
                    }
                }
            }
        }
        std::cout << "PREFILL_SERVICE_WORK PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
