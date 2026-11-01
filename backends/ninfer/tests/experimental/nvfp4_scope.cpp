#include "ninfer/ops/linear.h"
#include "ops/linear/common/route_table.h"
#include "ops/common/device_route.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

void check_route_selection() {
    using namespace ninfer;
    using namespace ops::detail;
    if (!unified_linear_routes_here()) return;
    const char* value = std::getenv("NINFER_LINEAR_ROUTES");
    const std::string_view spec = value == nullptr ? "" : value;
    const bool int8_scope = spec.starts_with("int8:");
    const bool a16_scope = spec.starts_with("a16:");
    if (!spec.empty() && !int8_scope && !a16_scope) {
        throw std::invalid_argument("routing check requires default or scoped NVFP4 environment");
    }
    const auto forced = spec.ends_with(":unified") ? LinearRouteTable::Unified
                                                  : LinearRouteTable::Legacy;
    const auto int8_table = int8_scope ? forced : LinearRouteTable::Unified;
    const auto a16_table = a16_scope ? forced : LinearRouteTable::Legacy;
    for (int width : {1, 2, 8, 9, 32, 129, 2048}) {
        if (linear_route_table(LinearRouteFamily::Nvfp4Int8, width) != int8_table) {
            throw std::runtime_error("plain W4A8 selected an unexpected route");
        }
    }
    if (linear_route_table(LinearRouteFamily::Nvfp4, 2048) != a16_table ||
        linear_route_table(LinearRouteFamily::Q4, 2048) != LinearRouteTable::Legacy) {
        throw std::runtime_error("NVFP4 override changed a different plain Linear family");
    }
    // Give both fused families the opposite profile schedule so a leaked scope is observable.
    const auto baseline = forced == LinearRouteTable::Unified ? LinearRouteTable::Legacy
                                                             : LinearRouteTable::Unified;
    const char* schedule = baseline == LinearRouteTable::Unified ? "unified" : "";
    for (const auto& [key, expected] : {
             std::pair{"unified/nvfp4_linear_add", a16_scope ? forced : baseline},
             std::pair{"nvfp4_int8/linear_add", int8_scope ? forced : baseline},
             std::pair{"unified/fp8_linear_add", baseline}}) {
        const ops::DeviceRouteForce route(key, schedule);
        if (fused_route_table(key, 1) != expected) {
            throw std::runtime_error("NVFP4 override crossed fused family boundaries");
        }
    }
    bool rejected = false;
    try {
        const auto bytes = ops::linear_workspace_capacity_bytes(
            QType::NVFP4, 14336, 5120, ops::LinearPolicy::AllowNvfp4Int8, 1, 129);
        if (bytes == 0) throw std::runtime_error("W4A8 projection needs activation workspace");
    } catch (const std::invalid_argument&) { rejected = true; }
    if (rejected != (int8_table == LinearRouteTable::Legacy)) {
        throw std::runtime_error("MTP-shaped W4A8 workspace admission disagrees with selected route");
    }
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    if (!ninfer::ops::detail::unified_linear_routes_here()) return 77;
    try {
        check_route_selection();
        std::cout << "NVFP4 default/scoped routing passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
