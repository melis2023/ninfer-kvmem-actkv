#include "ops/common/device_route.h"
#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string>

int main() {
    using namespace ninfer::test::linear_add;
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        // Route starts follow the 2026-09-17 sm_86 retune of select_q4_linear_add.
        constexpr std::array<std::int32_t, 8> route_starts{2, 5, 9, 25, 65, 81, 97, 129};
        constexpr std::array<std::int32_t, 13> interiors{1,   4,   8,   24,  32,  64, 80,
                                                         96,  128, 160, 161, 192, 193};
        constexpr std::array<std::int32_t, 6> graph_tokens{1, 4, 33, 97, 193, 512};
        constexpr std::array<std::int32_t, 3> full_tokens{1, 4, 8};
        int failures = run_shape("Q4_A16 LinearAdd", WeightFormat::Q4G64F16S,
                                 {5120, 6144, 429U, route_starts, interiors, graph_tokens});
        failures += run_shape("Q4_A16 LinearAdd full", WeightFormat::Q4G64F16S,
                              {5120, 6144, 431U, {}, full_tokens, {}, true});
        // MLP down: the same route table at K=17408.
        failures += run_shape("Q4_A16 LinearAdd down", WeightFormat::Q4G64F16S,
                              {5120, 17408, 433U, route_starts, interiors, graph_tokens});
        // The small-T MMA variants the device profile may route to, forced over their width
        // domains at both K.
        constexpr std::array<std::int32_t, 4> small_c8{1, 2, 7, 8};
        constexpr std::array<std::int32_t, 4> small_c16{9, 12, 15, 16};
        constexpr std::array<std::int32_t, 5> small_c32{17, 24, 25, 31, 32};
        constexpr std::array<std::int32_t, 2> graph_c16{9, 16};
        constexpr std::array<std::int32_t, 2> graph_c32{17, 32};
        for (const std::int32_t k : {6144, 17408}) {
            const std::string key = "q4_linear_add/5120x" + std::to_string(k);
            const auto seed       = static_cast<std::uint32_t>(437U + k);
            {
                const ninfer::ops::DeviceRouteForce force(key, "small_t_c8");
                failures += run_shape("Q4_A16 LinearAdd small-T c8", WeightFormat::Q4G64F16S,
                                      {5120, k, seed, {}, small_c8, {}});
            }
            {
                const ninfer::ops::DeviceRouteForce force(key, "small_t_c16");
                failures += run_shape("Q4_A16 LinearAdd small-T c16", WeightFormat::Q4G64F16S,
                                      {5120, k, seed + 1U, {}, small_c16, graph_c16});
            }
            {
                const ninfer::ops::DeviceRouteForce force(key, "small_t_c32");
                failures += run_shape("Q4_A16 LinearAdd small-T c32", WeightFormat::Q4G64F16S,
                                      {5120, k, seed + 2U, {}, small_c32, graph_c32});
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
