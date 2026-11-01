#pragma once

#include "models/expert_profile.h"
#include "models/qwen3_5/model.h"
#include "ops/sparse_moe/streamed_cpu.h"

#include <filesystem>
#include <optional>
#include <string>

namespace ninfer::models::qwen3_5::execution {

// The routing profile that fills the expert caches once warm-up is done, and the file that
// receives the counts the engine's decode and verification calls record when it stops.
struct NativeExpertProfile {
    std::optional<ExpertRouteCounts> counts;
    std::filesystem::path record;
    artifact::ArtifactId artifact{};
};

// The CPU's share of a decode or verification call's missing experts (ops/sparse_moe/streamed_cpu.h,
// the GPU's products bit for bit). dma_share: the share 0..1 the GPU copies, the CPU computing the
// rest; unset, each call splits its misses by measured CPU and bus times. threads: the CPU pool,
// 0 for pool_cpu_count().
struct NativeExpertCpu {
    std::optional<float> dma_share;
    std::uint32_t threads = 0;
};

// Program-owned service. Immutable sources are borrowed from Model. No service or slot is shared
// between Programs, even when their Parameters refer to the same Model.
class NativeExpertRuntime {
public:
    NativeExpertRuntime(const Model& model, DeviceContext& device,
                         std::span<const std::uint64_t> keep_free_by_rank,
                         std::optional<std::uint64_t> cache_bytes = {},
                         NativeExpertProfile profile = {}, NativeExpertCpu cpu = {});
    ~NativeExpertRuntime();
    NativeExpertRuntime(const NativeExpertRuntime&) = delete;
    NativeExpertRuntime& operator=(const NativeExpertRuntime&) = delete;
    // The routed experts of each of the model's expert layers.
    [[nodiscard]] static std::vector<std::size_t> widths(const Model& model);
    // Once warm-up is done: empties every rank's cache, forgets the warm-up's routes and fills
    // the cache from the profile.
    void prime();
    // Every route the decode and verification calls took through each expert layer since prime().
    [[nodiscard]] ExpertRouteCounts totals() const;
    [[nodiscard]] ExpertDelivery& layer(std::size_t index) const;
    [[nodiscard]] std::size_t local_layer(std::size_t index) const;
    [[nodiscard]] ExpertDeliveryMemory memory(std::size_t rank) const noexcept;
    [[nodiscard]] ExpertDeliveryStats stats() const;
    void check() const;
    // Waits for every rank's delivery work, prefetches included.
    void drain();
    void release_device_backing();
    void restore_device_backing();
    [[nodiscard]] std::uint64_t device_backing_bytes() const noexcept;
    // One device with virtual memory management: the cache grows into free device memory beyond
    // the bytes set_keep_free_bytes keeps and gives memory back (ExpertDelivery::grow_cache,
    // shrink_cache), so free KV pages can fund it (core/kv_expert_loan.h).
    [[nodiscard]] bool elastic() const noexcept { return elastic_; }
    std::uint64_t grow_cache();
    std::uint64_t shrink_cache(std::uint64_t bytes);
    void set_keep_free_bytes(std::uint64_t bytes);
    // With a CPU share: its threads and vector backend, e.g. "15 pinned threads, avx2"; else empty.
    [[nodiscard]] std::string cpu_description() const;

private:
    struct Handle { std::size_t rank, local; };
    std::vector<Handle> handles_;
    std::vector<std::unique_ptr<ops::detail::SparseMoeStreamCpu>> cpu_;
    std::vector<std::unique_ptr<ExpertDelivery>> ranks_;
    NativeExpertProfile profile_;
    bool elastic_ = false;
};

} // namespace ninfer::models::qwen3_5::execution
