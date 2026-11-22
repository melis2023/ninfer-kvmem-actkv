#pragma once

#include "core/weight_view.h"

namespace ninfer::artifact {

struct TensorObject;

// Interpret and validate a requested tensor's complete encoded parent. `source_alignment` is the
// alignment the container promises for its object offsets: a `.ninfer` writes every object at 256
// bytes, while a GGUF aligns its data section to as little as 32, so an adapter reading one passes
// the alignment that file actually keeps rather than the layout's.
[[nodiscard]] WeightGeometry describe_tensor(const TensorObject& object,
                                             std::uint64_t source_alignment = 256);

} // namespace ninfer::artifact
