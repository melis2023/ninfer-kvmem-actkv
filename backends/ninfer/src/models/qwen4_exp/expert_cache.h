#pragma once

// A device cache of the routed experts of host-resident banks. Every expert of a layer is reached
// through device tables of base pointers (one per projection) that the expert kernels read; a
// cached expert's entries point at its copy in a device slot, every other entry at its bytes in the
// pinned host block. Between forward passes the cache counts the routes the last pass took, with an
// exponential decay per token, and swaps the experts it would most often have needed into the
// slots of the ones it needed least, by copy and table update on the layer's stream. The kernels
// never see the difference, so no pass waits on the cache.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExpertBank {
    std::int64_t expert_bytes = 0; // one expert's rows
    std::vector<const void*> host; // each expert's rows in the pinned host block
    void* table = nullptr;         // the device table the kernels read
};

struct ExpertCacheLayer {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    ExpertBank gate, up, down;
};

struct ExpertCacheStats {
    std::uint64_t routes       = 0; // (token, expert) pairs observed
    std::uint64_t hits         = 0; // of them, served from a slot
    std::uint64_t admitted     = 0; // experts copied into slots
    std::uint64_t copied_bytes = 0;
    std::uint32_t slots        = 0;
};

class ExpertCache {
public:
    // `bytes_by_rank` is the device memory each rank lends its layers' slots.
    ExpertCache(DeviceContext& device, std::vector<ExpertCacheLayer> layers,
                std::span<const std::uint64_t> bytes_by_rank);
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // The routes one pass took through `layer`: `ids` holds every (token, slot) pair's expert.
    void observe(std::size_t layer, std::span<const std::int32_t> ids, std::uint32_t tokens);
    // Copies up to `byte_budget` bytes of experts into slots and updates the tables, on each
    // layer's stream.
    void rebalance(std::uint64_t byte_budget);
    // Projection k (0 gate, 1 up, 2 down) of `expert` in its device slot, or null when the expert
    // is not cached. Slots are zeroed when allocated and 256 zero bytes follow the last, as the
    // expert matrix kernel needs (see moe_experts_gguf).
    [[nodiscard]] const void* cached(std::size_t layer, int k, std::int32_t expert) const;
    [[nodiscard]] ExpertCacheStats stats() const noexcept;

private:
    struct Layer;
    DeviceContext& device_;
    std::vector<Layer> layers_;
    std::unique_ptr<PinnedHostBuffer> staging_;
    ExpertCacheStats stats_;
};

} // namespace ninfer::models::qwen4_exp
