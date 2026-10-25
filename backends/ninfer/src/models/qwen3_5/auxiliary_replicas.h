#pragma once

// Per-device copies of the auxiliaries a weight's Uses read: input-column gathers and Hadamard sign
// vectors. An artifact stores one object for every Use with the same auxiliary value (one gather
// serves every Gated DeltaNet output projection of a GGUF import), and one object is materialized
// on one device -- rank 0, since nothing places an auxiliary with a layer. A pipeline stage whose
// weights read such an auxiliary gets its own copy, so no Use reads another GPU's memory, which a
// device without peer access cannot address.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight_view.h"
#include "models/qwen3_5/weights.h"

#include <cstddef>
#include <map>
#include <span>
#include <utility>

namespace ninfer::models::qwen3_5 {

class AuxiliaryReplicas {
public:
    AuxiliaryReplicas() = default;
    // Copies every auxiliary to each rank whose weights read it from another rank.
    AuxiliaryReplicas(std::span<const BoundWeight> bound, DeviceContext& device);

    // The auxiliary `id` of `bound`, a 1-D tensor in the memory of device `rank`.
    [[nodiscard]] Tensor on_rank(std::span<const BoundWeight> bound, WeightId id,
                                 std::size_t rank) const;

    [[nodiscard]] std::size_t copies() const noexcept { return copies_.size(); }

private:
    std::map<std::pair<std::size_t, std::size_t>, DeviceBuffer> copies_; // by (auxiliary, rank)
};

} // namespace ninfer::models::qwen3_5
