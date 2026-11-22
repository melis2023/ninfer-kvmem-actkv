#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::models::dspine {
class Drafter;
}

namespace ninfer::models::qwen3_5::detail {

struct DFlashPersistentState {
    CyclicKVCache& local;
    std::optional<qwen3_5::PagedKVCache> full;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;
    // --spec dspine: the head-write table (dspine::Drafter); empty otherwise.
    Tensor dspine_head_write;
    // --spec dspine: the drafter that materializes the context into `local` and proposes; the
    // Program owns it. Null for DFlash and DFlash2, whose weights are the model's.
    const dspine::Drafter* dspine = nullptr;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout,
                          CyclicKVCache& local_state);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    void save_rewrite_checkpoint(std::int32_t source_slot, std::int32_t destination_slot,
                                 cudaStream_t stream);
};

} // namespace ninfer::models::qwen3_5::detail
