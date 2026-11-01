#pragma once

#include "core/expert_delivery.h"
#include "models/qwen3_5/expert_layout.h"

namespace ninfer::models::qwen3_5 {

// Aliases are gate/up codes, high bits, scales, divisors, then the same four down planes.
// An unused high/divisor alias points at codes; its codec never reads it.
struct NativeExpertLayer {
    ExpertDeliveryLayer delivery;
    ExpertProjectionLayout gate_up, down;
    std::size_t rank = 0;
};

struct NativeExpertSources {
    std::vector<std::filesystem::path> files;
    std::vector<NativeExpertLayer> layers;
};

} // namespace ninfer::models::qwen3_5
