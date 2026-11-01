#include "runtime/engine/graph_probe.h"
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

using Reading = std::optional<std::size_t>;
constexpr std::size_t MiB = 1ULL << 20;

void check(std::initializer_list<Reading> readings, Reading expected, std::size_t calls) {
    std::vector<Reading> values(readings);
    std::size_t index = 0;
    const auto result = ninfer::detail::calibrate_graph_allowance([&]() -> Reading {
        if (index == values.size()) { throw std::runtime_error("unexpected extra probe"); }
        return values[index++];
    });
    if (result.allowance != expected || index != calls) {
        throw std::runtime_error("invalid calibrated capacity allowance");
    }
}

int main() {
    try {
        check({std::nullopt}, std::nullopt, 1); // Preserve static planning on failure.
        check({0}, 256 * MiB, 1); // Warm driver caches must not remove the floor.
        check({30 * MiB}, 256 * MiB, 1);
        check({256 * MiB}, 256 * MiB, 1);
        check({300 * MiB, 40 * MiB}, 256 * MiB, 2);
        check({300 * MiB, 290 * MiB}, 290 * MiB, 2);
        check({300 * MiB, 400 * MiB}, 300 * MiB, 2);
        check({300 * MiB, std::nullopt}, 300 * MiB, 2);
        bool propagated = false;
        try {
            ninfer::detail::calibrate_graph_allowance([]() -> Reading {
                throw std::runtime_error("execution fault");
            });
        } catch (const std::runtime_error&) { propagated = true; }
        if (!propagated) { throw std::runtime_error("execution fault was swallowed"); }
        std::cout << "graph probe calibration policy: 9 cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
