#pragma once

// Engine-wide choices of purpose, KV capacity, speculative drafting and weight residency that
// the Engine options, the statistics and the model loaders share.

#include <cstdint>

namespace ninfer {

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

enum class ProposalHead : std::uint8_t {
    // The artifact's optimized proposal head when it stores one, the full output head otherwise;
    // resolved when the model loads.
    Auto,
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

// How many of its drafts an MTP round verifies: always all of them, or a width the Engine adapts
// to the drafts' measured survival and the measured round cost. Greedy output does not depend on
// it.
enum class MtpDraftPolicy : std::uint8_t {
    Fixed,
    Adaptive,
};

enum class VisionResidency : std::uint8_t {
    Resident, // fixed Vision GPU allocations for the process lifetime
    Overlay,  // tower host-pinned; each image borrows device memory inside a bounded window
    Cpu,      // tower decoded to FP32 host memory and run on CPU threads; no device Vision memory
};

// Where supported MoE families keep routed expert banks (see each family guide).
enum class ExpertResidency : std::uint8_t {
    // In the weight arena of each layer's stage device.
    Device,
    // In host RAM: experts the device cache lacks are copied to the GPU or computed on the CPU
    // (HybridExpertOptions).
    Host,
    // Left in the artifact's files and copied into a device cache when a pass routes to them,
    // through the OS page cache: the least host memory, at the cost of a wait for every expert a
    // pass needs that the cache does not hold.
    Disk,
};

// Qwen3.8-Flash-Next: where the n-gram table's rows come from.
enum class NgramResidency : std::uint8_t {
    // The table's file, with a bounded cache of rows read as the hash addresses them.
    Disk,
    // RAM: the whole table is read at startup.
    Ram,
    // RAM for the rows a hot-row profile ranks most frequent, as many as the RAM budget holds,
    // read at startup; the other rows come from the file.
    RamHot,
};

// Qwen3.8-Flash-Next: how rows are read from the table's file.
enum class NgramIo : std::uint8_t {
    // Positioned reads through the OS page cache.
    Buffered,
    // Reads that bypass the page cache (O_DIRECT, FILE_FLAG_NO_BUFFERING): the table takes no
    // RAM beyond what the residency keeps, and every cache miss reaches the drive.
    Direct,
    // Copies out of a read-only mapping of the file, faulted in through the page cache.
    Mapped,
};

} // namespace ninfer
