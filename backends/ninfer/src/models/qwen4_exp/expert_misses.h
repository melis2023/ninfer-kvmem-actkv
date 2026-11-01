#pragma once

// The routed experts a decode or verification layer needs but its device cache lacks, served while
// the cached ones run. After a layer's routing a kernel splits its pairs: the cached ones run at once
// from the device tables; the missing experts are posted to pinned host memory (one "doorbell" per
// device) without any host synchronization, so the whole pass can replay as one CUDA graph. A host
// thread picks the request up immediately and, for each missing expert, either copies it into a
// device staging region (or, when the cache's expert tier admits it, straight into a slot) by the
// copy engine, or has the CPU compute its pairs from the bytes in RAM (host experts) or from the
// file (disk experts); an expert the CPU computes crosses no bus and takes no slot. The second
// stage of the layer waits for the request on the GPU, then runs the staged experts and adds the
// CPU's products. The staged experts are reached through a second table that only the second
// stage reads, so the resident stage needs no host at all.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "models/qwen4_exp/expert_cache.h"
#include "models/qwen4_exp/expert_location.h"
#include "ninfer/ops/moe_experts.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct MissLayer {
    std::size_t rank   = 0;
    std::size_t experts = 0;
    struct Projection {
        QType format              = QType::GGUF_Q2_0;
        std::int64_t row_bytes    = 0;
        std::int32_t rows         = 0;
        std::int32_t k            = 0;
        std::int64_t expert_bytes = 0;
        std::vector<const void*> host;           // host experts: each expert's bytes in RAM
        std::span<const ExpertLocation> located; // disk experts: where each expert's bytes are
    };
    std::array<Projection, 3> projections;   // gate, up, down
    const void* const* gate_table = nullptr; // the device table the cached stage reads
    // Device memory holding this layer's cached experts (ExpertCache::storage): a gate entry
    // inside any of its blocks is resident.
    ExpertCache::Ranges resident{};
    // Host experts whose layer the cache handed over: a missing expert the GPU copies anyway goes
    // straight into a slot when the cache's tier admits it (ExpertCache::admit), and stays.
    bool admit = false;
};

struct MissOptions {
    std::uint32_t token_capacity = 8;    // tokens of one call
    double cpu_share             = 0.0;  // share of a request's missing experts the CPU computes
    // Instead of a fixed share, each request gives the CPU the experts that make its copies and
    // its CPU products end soonest, by times fitted to the earlier requests.
    bool cpu_balance             = false;
    std::uint32_t cpu_threads    = 0;    // 0: one per physical core (pool_cpu_count())
    std::uint64_t staging_bytes  = 128ULL << 20; // device staging per rank, two halves
    // 1..9: a call leaves out each missing expert that all its tokens rank past this many places
    // of their top ten (lossy); 0 serves every miss.
    std::uint32_t tail_skip_rank = 0;
};

struct MissStats {
    std::uint64_t requests    = 0; // layers that had missing experts
    std::uint64_t staged      = 0; // experts copied to device staging or straight into slots
    std::uint64_t mapped      = 0; // experts the GPU read across the bus (staging full)
    std::uint64_t computed    = 0; // experts the CPU computed
    std::uint64_t admitted    = 0; // of the staged, copied straight into cache slots
    std::uint64_t cpu_pairs   = 0;
    std::uint64_t staged_bytes = 0;
    double service_seconds     = 0; // host time from a request to its readiness, summed
};

class ExpertMisses {
public:
    // `cache` (may be null) holds the slots of the layers that admit; its layer indices are the
    // misses' own.
    ExpertMisses(DeviceContext& device, std::vector<MissLayer> layers,
                 std::vector<std::filesystem::path> files, MissOptions options, ExpertCache* cache);
    ~ExpertMisses();
    ExpertMisses(const ExpertMisses&)            = delete;
    ExpertMisses& operator=(const ExpertMisses&) = delete;

    // Device views a layer's call uses: the cached pairs' ids, then the missing pairs' ids, tables
    // and CPU products the host fills.
    struct Call {
        Tensor cached_ids;
        Tensor missing_ids;
        ops::GgufMoeWeights missing_banks;
        const float* products        = nullptr;
        const std::int32_t* pairs    = nullptr;
        const std::int32_t* count    = nullptr;
        std::int32_t max_products    = 0;
    };
    // Enqueued on `stream` (capturable): splits `ids` of `tokens` tokens and posts the missing
    // experts. `m` is the layer's BF16 input, copied to the host when the CPU takes a share.
    [[nodiscard]] Call begin(std::size_t layer, const Tensor& ids, const Tensor& m,
                             std::int32_t tokens, const ops::GgufMoeWeights& banks,
                             cudaStream_t stream);
    // Enqueued, any token count: the pairs of `ids` (I32 [10, tokens]) whose expert is cached,
    // the others -1, and their `weights` (FP32 [10, tokens]) renormalized per token to its whole
    // routed sum: for a layer whose output only steers drafts, which verification checks.
    void drop_missing(std::size_t layer, const Tensor& ids, const Tensor& weights,
                      Tensor& kept_ids, Tensor& kept_weights, cudaStream_t stream);
    // Enqueued: waits until the request's experts are staged and its CPU products written.
    void await(std::size_t layer, cudaStream_t stream);
    // Keeps the service (and the CPU workers) spinning for the next half second; a pass calls it
    // before its first layer. Rethrows a service failure.
    void keep_alive();
    // Rethrows a failure the service recorded while serving a call. A failed call released the
    // GPU with its missing experts left out, so a pass checks this once it has completed, before
    // anything it computed is used.
    void raise_failure();
    // The layer's cached blocks changed (the cache grew); the device copy the kernels read follows.
    void set_resident(std::size_t layer, const ExpertCache::Ranges& resident);
    [[nodiscard]] MissStats stats() const;
    [[nodiscard]] bool cpu_enabled() const noexcept;
    // With a CPU share: its threads and vector backend, e.g. "15 pinned threads, avx2"; else
    // empty.
    [[nodiscard]] std::string cpu_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
