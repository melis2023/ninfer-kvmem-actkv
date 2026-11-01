#include "core/expert_delivery.h"

#include "core/arena.h"
#include "core/cpu_spin_pool.h"
#include "core/cuda_host_handshake.h"
#include "core/cuda_host_handshake.cuh"
#include "core/expert_split.h"
#include "core/file_read_queue.h"
#include "core/growable_region.h"
#include "core/vmm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace ninfer {
namespace {

// A round's copies a call's post work issues at most: experts, and bytes read from files.
constexpr std::size_t kRoundExperts = 8;
constexpr std::uint64_t kRoundFileBytes = 8ULL << 20;

// needed[e] of a call: its pairs routed to expert e, and whether the table entry the split read
// lies in the persistent cache.
constexpr std::uint32_t kCountMask = (1u << 30) - 1;
// Request flag: begin() received the family's CPU inputs, so the call may leave misses to the CPU.
constexpr std::uint32_t kCpuCall = 1;
constexpr std::uint32_t kHit     = 1u << 30;
constexpr std::uint32_t kMiss    = 1u << 31;

std::uint64_t aligned(std::uint64_t n) {
    if (n > std::numeric_limits<std::uint64_t>::max() - 255) {
        throw std::invalid_argument("expert delivery: byte extent overflow");
    }
    return (n + 255) / 256 * 256;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument("expert delivery: byte count overflow");
    }
    return a * b;
}

// Use the core's existing fixed-address allocation when the Program supports model suspend. An
// elastic cache is a growable region: addresses for every expert, `bytes` of them in slots.
struct Storage {
    DeviceBuffer owned;
    VmmRegion region;
    GrowableRegion growable;
    void* p = nullptr;
    std::size_t bytes = 0;
    Storage() = default;
    Storage(std::size_t n, int device, bool suspendable) : bytes(n) {
        if (suspendable) {
            region = VmmRegion(n, {.device = device});
            p = region.span().data;
        } else {
            owned = DeviceBuffer(n);
            p = owned.p;
        }
    }
    void fill() {
        if (growable) {
            // An elastic cache maps a run of granules for each of its pools: zero every run.
            const std::size_t g = growable.granule();
            for (std::size_t at = 0; at < growable.bytes();) {
                if (!growable.mapped(at, g)) {
                    at += g;
                    continue;
                }
                std::size_t end = at + g;
                while (end < growable.bytes() && growable.mapped(end, g)) { end += g; }
                CUDA_CHECK(cudaMemset(static_cast<std::byte*>(p) + at, 0, end - at));
                at = end;
            }
        } else {
            CUDA_CHECK(cudaMemset(p, 0, bytes));
        }
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    [[nodiscard]] std::uint64_t backing_bytes() const noexcept {
        return growable ? growable.mapped_bytes() : region ? region.reserved_bytes() : bytes;
    }
    void release_backing() {
        if (growable) { growable.release_backing(); }
        else { region.release_backing(); }
    }
    void restore_backing() {
        if (growable) { growable.restore_backing(); }
        else { region.restore_backing(); }
    }
};

// Device ranges a call hands the host service (the CPU's operands), into mapped host memory.
struct HostCopies {
    const std::byte* source[3]{};
    std::byte* target[3]{};
    std::uint32_t bytes[3]{};
};

// One block: a shared histogram of the call's experts, then each pair's cached/missing id and each
// set's experts (their count, then the experts ascending) by the table entry the expert has now.
// It writes the mask and `copies` into mapped host memory and publishes the request itself.
// Tables change only at missing entries or at experts absent from this call, so cached consumers
// can read them while the host prepares the rest. An entry in a lent bank (`lent`: [bank][low,
// high), empty when not lent) hits in a narrow call only: a wide call, which routes half the
// layer's experts or more, may take that bank back while its cached pairs run.
__global__ void split_routes(const std::int32_t* ids, std::uint32_t pairs, int experts,
                             const void* const* tables, std::uintptr_t low, std::uintptr_t high,
                             const std::uintptr_t* lent, std::int32_t* cached,
                             std::int32_t* missing, std::int32_t* cached_experts,
                             std::int32_t* missing_experts, std::uint32_t* needed,
                             HostCopies copies, handshake::DeviceMailbox mailbox, std::uint32_t tag,
                             std::uint32_t flags) {
    extern __shared__ std::uint32_t counts[];
    __shared__ std::uint32_t routes, sets[2][32], listed[2];
    if (threadIdx.x == 0) {
        routes    = 0;
        listed[0] = listed[1] = 0;
    }
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { counts[e] = 0; }
    __syncthreads();
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { atomicAdd(counts + e, 1u); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        if (counts[e]) { atomicAdd(&routes, 1u); }
    }
    __syncthreads();
    const bool wide = 2 * routes >= std::uint32_t(experts);
    const std::uintptr_t lent_low[2]{lent[0], lent[2]}, lent_high[2]{lent[1], lent[3]};
    const auto hit = [&](const void* entry) {
        const auto address = reinterpret_cast<std::uintptr_t>(entry);
        if (address >= low && address < high) { return true; }
        return !wide && ((address >= lent_low[0] && address < lent_high[0]) ||
                         (address >= lent_low[1] && address < lent_high[1]));
    };
    for (std::uint32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e       = ids[p];
        const bool valid  = e >= 0 && e < experts;
        const bool cached_pair = valid && hit(tables[e]);
        cached[p]  = cached_pair ? e : -1;
        missing[p] = valid && !cached_pair ? e : -1;
    }
    // Each warp ballots its experts of a block-wide pass; earlier warps and passes place them.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int first = 0; first < experts; first += blockDim.x) {
        const int e               = first + threadIdx.x;
        const std::uint32_t count = e < experts ? counts[e] : 0;
        const bool in_cache       = count != 0 && hit(tables[e]);
        const unsigned hits       = __ballot_sync(~0u, in_cache);
        const unsigned misses     = __ballot_sync(~0u, count != 0 && !in_cache);
        if (e < experts) {
            needed[e] = count == 0 ? 0u : (count & kCountMask) | (in_cache ? kHit : kMiss);
        }
        if (lane == 0) {
            sets[0][warp] = __popc(hits);
            sets[1][warp] = __popc(misses);
        }
        __syncthreads();
        std::uint32_t at[2]{listed[0], listed[1]};
        for (int w = 0; w < warp; ++w) {
            at[0] += sets[0][w];
            at[1] += sets[1][w];
        }
        const unsigned below = (1u << lane) - 1;
        if (count != 0) {
            if (in_cache) {
                cached_experts[1 + at[0] + __popc(hits & below)] = e;
            } else {
                missing_experts[1 + at[1] + __popc(misses & below)] = e;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < int(blockDim.x >> 5); ++w) {
                listed[0] += sets[0][w];
                listed[1] += sets[1][w];
            }
        }
        __syncthreads();
    }
    for (int c = 0; c < 3; ++c) {
        for (std::uint32_t i = threadIdx.x; i < copies.bytes[c]; i += blockDim.x) {
            copies.target[c][i] = copies.source[c][i];
        }
    }
    // The host reads the mask and the copies once it sees the request.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        cached_experts[0]  = std::int32_t(listed[0]);
        missing_experts[0] = std::int32_t(listed[1]);
        handshake::publish(mailbox, tag, pairs, flags);
    }
}

} // namespace

struct ExpertDelivery::Impl {
    struct Layer {
        ExpertDeliveryLayer source;
        Storage tables;
        std::unique_ptr<PinnedHostBuffer> mirror;
        // Whole-table uploads alternate between two pinned copies, so the mirror can change while
        // an earlier upload still reads its copy.
        std::unique_ptr<PinnedHostBuffer> upload[2];
        cudaEvent_t uploaded[2]{};
        int next_upload = 0;
        bool dirty      = false;
        std::vector<std::int32_t> slot_of;
        std::uint32_t cached = 0; // slots holding this layer's experts
        std::uint32_t pool   = 0; // the pool of slots its experts fit
        // Bank stride of each blob: the blob's bytes when every expert's are equal (the resident
        // layout, one copy per run of experts), else its aligned maximum.
        std::vector<std::uint64_t> stride;
        std::int64_t successor = -1;
        bool successor_wide    = false;
        std::vector<std::uint64_t> total; // the narrow calls' routes of each expert
        bool cpu = false;                 // the family computes its experts on the CPU
        // Where each expert's entries point besides a cache slot: the lent slot holding it, and
        // the bank whose layout they address (-1: none).
        std::vector<std::int32_t> lent_of;
        std::vector<std::int8_t> bank_of;
    };
    struct Bank {
        std::int64_t layer = -1;
        std::vector<unsigned char> valid;
    };
    struct Slot {
        std::int32_t layer = -1, expert = -1;
        std::uint32_t pool = 0;
        // Its table entries still point elsewhere: a promotion's are patched at the next call, a
        // round's once its copy has landed.
        bool pending = false;
        std::vector<std::uint64_t> written;
    };
    // A slot of a lent bank: bank k holds slots [k * per_bank, (k + 1) * per_bank). `used`: the
    // last call that routed its expert.
    struct LentSlot {
        std::int32_t layer = -1, expert = -1;
        std::uint64_t used = 0;
        std::vector<std::uint64_t> written;
    };

    // The cache slots of the layers whose experts' blobs need the same room (each blob's largest
    // bytes and tail in those layers, 256-byte aligned): the tier's pools (core/expert_tier.h). A
    // pool's slots are ids [first, first + capacity), the first `count` of them backed, at `base`
    // in the cache one after another; an elastic cache reserves `span` bytes for each, a whole
    // number of granules.
    struct Pool {
        std::vector<std::uint64_t> offset, size; // per blob, within a slot
        std::uint64_t slot_bytes = 0, base = 0, span = 0;
        std::int32_t first     = 0;
        std::uint32_t capacity = 0, count = 0, layers = 0;
        std::vector<std::int32_t> free; // popped lowest first
    };
    struct Deferred {
        std::size_t layer;
        std::int32_t expert, slot;
        cudaEvent_t landed = nullptr; // a round's copy; null: a promotion, ordered on `copies`
    };
    // A host-to-device copy of contiguous bytes from one pinned range or a run of file extents.
    struct Piece {
        std::byte* target = nullptr;
        const std::byte* pinned = nullptr;
        std::uintptr_t region = 0;
        std::vector<ExpertExtent> extents;
        std::uint64_t bytes = 0;
    };

    RankContext rank;
    ExpertDeliveryOptions options;
    std::vector<Layer> layers;
    std::uint32_t max_experts = 0;
    // Per blob: a lent slot's offset/size (aligned for the largest blob and tail of any layer), the
    // bank region offset/size and the zero tail kept after a bank run.
    std::vector<std::uint64_t> slot_offset, slot_size, bank_offset, bank_size, tail;
    std::uint64_t slot_bytes = 0, bank_bytes = 0;
    std::vector<Pool> pools;
    // What backs a pool's slots: an elastic cache maps whole granules for each pool; one region
    // holds every pool's slots and rounds once.
    std::uint64_t unit         = 1;
    std::uint64_t expert_count = 0;
    Storage cache, cached_ids, missing_ids;
    // The call's expert lists, cached then missing (each its count, then the experts), and last
    // the handshake's sequence word.
    Storage lists;
    std::int32_t* cached_experts  = nullptr;
    std::int32_t* missing_experts = nullptr;
    // The addresses a table entry lies in when it points into the cache: an elastic cache's whole
    // reservation, so graphs captured before it grew still find its later slots.
    std::uint64_t cache_span = 0;
    std::vector<Storage> banks;
    std::vector<Bank> bank_state;
    int last_bank = 0;
    // Lent banks (ExpertDeliveryOptions::lend_banks): their slots, the slots one bank holds (0:
    // banks are never lent), which banks are lent, the device copy of their address ranges the
    // split reads and its pinned source, the narrow passes since the last wide call, and the
    // served calls' count.
    std::vector<LentSlot> lent_slots;
    std::uint32_t per_bank = 0;
    std::vector<unsigned char> bank_lent;
    Storage lent_ranges;
    std::unique_ptr<PinnedHostBuffer> host_ranges;
    bool ranges_dirty         = false;
    std::uint32_t quiet_passes = 0;
    std::uint64_t stamp        = 0;
    std::unique_ptr<PinnedHostBuffer> needed_host, staging;
    void* needed_mapped = nullptr;
    std::vector<Slot> slots; // every pool's, backed or not
    std::vector<Deferred> deferred;
    // The tier over the pools, the last narrow call's layer (a lower one starts a pass), and a
    // round's candidates whose copies post work has not issued yet.
    std::unique_ptr<ExpertTier> tier;
    std::int64_t last_narrow = -1;
    std::vector<Deferred> queued;
    cudaStream_t rounds = nullptr;
    std::vector<cudaEvent_t> spare_events;
    std::unique_ptr<PinnedHostBuffer> round_staging; // disk experts: a call's round reads
    cudaEvent_t round_staged = nullptr;               // the staging's last copies
    // A CPU share: the pool, the split, the call's inputs and ids on the host, the CPU's
    // products and flag, the corrected missing ids, and disk experts' bytes read for the CPU.
    std::unique_ptr<CpuSpinPool> pool;
    ExpertSplit split;
    std::unique_ptr<PinnedHostBuffer> host_inputs, host_ids, host_products, host_flag, host_missing;
    void* host_inputs_mapped = nullptr;
    void* host_ids_mapped    = nullptr;
    Storage products, cpu_flag;
    std::vector<std::byte> cpu_bytes;
    cudaEvent_t copies_begun = nullptr, copies_done = nullptr; // a call's copies, timed
    bool spinning = false;
    std::optional<CudaHostHandshake> handshake;
    cudaStream_t copies = nullptr;
    cudaEvent_t copied[2]{};
    std::uint64_t half_bytes = 0;
    int half = 0;
    std::uint64_t staging_cursor = 0;
    std::vector<QueuedFileRead> batch_reads;
    struct StagedCopy { std::byte* target; const std::uint8_t* source; std::size_t bytes; };
    std::vector<StagedCopy> batch_copies;
    std::vector<std::ifstream> files;
    std::unique_ptr<FileReadQueue> read_queue;
    std::atomic<bool> stopping{false};
    std::atomic<std::uint32_t> finished{0}; // the last request whose service, post work included, ended
    std::thread service;
    mutable std::mutex mutex;
    std::exception_ptr failure;
    ExpertDeliveryStats counters;
    ExpertDeliveryMemory allocation;
    bool released = false;
    // The call being served and its post work: layer, bank, whether wide, and its routed experts.
    // Between a call's ready signal and the next request the GPU may read any layer's table (the
    // next split, then that call's cached pairs), so post work writes only free slots. Cached
    // experts are replaced only while a request is served, never one it routes.
    std::int64_t previous_layer = -1;
    std::size_t call_layer      = 0;
    int call_bank               = 0;
    bool call_wide              = false;
    std::vector<unsigned char> call_routed, call_cpu;
    std::vector<std::uint32_t> call_count;
    std::vector<Piece> pieces;
    std::vector<std::pair<std::byte*, std::uint64_t>> zeroes;

    Impl(const RankContext& r, std::vector<std::filesystem::path> paths,
         std::vector<ExpertDeliveryLayer> specs, ExpertDeliveryOptions o)
        : rank(r), options(o) {
        DeviceBinding bind(rank.device);
        if (specs.empty() || options.pair_capacity == 0) {
            throw std::invalid_argument("expert delivery: layers and pair capacity are required");
        }
        if (options.banks < 1 || options.banks > 2) {
            throw std::invalid_argument("expert delivery: one or two layer banks");
        }
        bool disk = false;
        std::vector<std::uint64_t> largest, largest_tail;
        // Per layer and blob: the room a slot needs (its largest blob and tail, aligned).
        std::vector<std::vector<std::uint64_t>> rooms;
        for (const auto& layer : specs) {
            expert_count += layer.experts.size();
            if (layer.experts.empty() || layer.aliases.empty()) {
                throw std::invalid_argument("expert delivery: empty expert or alias table");
            }
            if (layer.experts.size() > 8192) {
                throw std::invalid_argument("expert delivery: at most 8192 experts per layer");
            }
            max_experts = std::max(max_experts, std::uint32_t(layer.experts.size()));
            const std::size_t blobs = layer.experts.front().size();
            std::vector<std::uint64_t> layer_largest(blobs, 0), layer_tail(blobs, 0);
            for (const auto& expert : layer.experts) {
                if (expert.empty() || expert.size() != blobs) {
                    throw std::invalid_argument("expert delivery: every expert of a layer needs its blobs");
                }
                largest.resize(std::max(largest.size(), expert.size()), 0);
                largest_tail.resize(largest.size(), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    const auto& blob = expert[b];
                    if (!blob.bytes || (blob.pinned != nullptr) == !blob.extents.empty()) {
                        throw std::invalid_argument("expert delivery: choose pinned bytes or extents");
                    }
                    std::uint64_t extent_bytes = 0;
                    for (const auto& extent : blob.extents) {
                        if (extent.file >= paths.size() || !extent.bytes ||
                            extent.bytes > blob.bytes - std::min(blob.bytes, extent_bytes)) {
                            throw std::invalid_argument("expert delivery: invalid file extent");
                        }
                        extent_bytes += extent.bytes;
                    }
                    if (!blob.pinned && extent_bytes != blob.bytes) {
                        throw std::invalid_argument("expert delivery: extents do not cover blob");
                    }
                    if (blob.zero_tail > std::numeric_limits<std::uint64_t>::max() - blob.bytes) {
                        throw std::invalid_argument("expert delivery: blob tail overflow");
                    }
                    largest[b]      = std::max(largest[b], blob.bytes);
                    largest_tail[b] = std::max(largest_tail[b], blob.zero_tail);
                    layer_largest[b] = std::max(layer_largest[b], blob.bytes);
                    layer_tail[b]    = std::max(layer_tail[b], blob.zero_tail);
                    disk |= !blob.extents.empty();
                }
                for (const auto& alias : layer.aliases) {
                    if (alias.blob >= expert.size() || alias.offset >= expert[alias.blob].bytes) {
                        throw std::invalid_argument("expert delivery: alias outside blob");
                    }
                }
            }
            auto& room = rooms.emplace_back(blobs);
            for (std::size_t b = 0; b < blobs; ++b) {
                room[b] = aligned(checked_add(layer_largest[b], layer_tail[b]));
            }
        }
        for (std::size_t b = 0; b < largest.size(); ++b) {
            slot_offset.push_back(slot_bytes);
            slot_size.push_back(aligned(checked_add(largest[b], largest_tail[b])));
            slot_bytes = checked_add(slot_bytes, slot_size.back());
        }
        tail = largest_tail;
        bank_size.assign(largest.size(), 0);
        if (options.cpu.products != nullptr &&
            (options.cpu.product_bytes == 0 || !(options.cpu.share >= 0 && options.cpu.share <= 1))) {
            throw std::invalid_argument("expert delivery: a CPU share needs its products and a share in [0, 1]");
        }
        std::uint64_t largest_expert = 0;
        for (std::size_t index = 0; index < specs.size(); ++index) {
            auto& spec = specs[index];
            Layer layer;
            for (const auto& expert : spec.experts) {
                std::uint64_t bytes = 0;
                for (const auto& blob : expert) { bytes += blob.bytes; }
                largest_expert = std::max(largest_expert, bytes);
            }
            layer.total.assign(spec.experts.size(), 0);
            layer.cpu = options.cpu.products != nullptr && options.cpu.products->supports(index);
            layer.pool = std::uint32_t(
                std::find_if(pools.begin(), pools.end(),
                             [&](const Pool& pool) { return pool.size == rooms[index]; }) -
                pools.begin());
            if (layer.pool == pools.size()) {
                auto& pool = pools.emplace_back();
                pool.size  = rooms[index];
                for (const auto size : pool.size) {
                    pool.offset.push_back(pool.slot_bytes);
                    pool.slot_bytes = checked_add(pool.slot_bytes, size);
                }
            }
            pools[layer.pool].capacity += std::uint32_t(spec.experts.size());
            ++pools[layer.pool].layers;
            const auto count = spec.experts.size();
            const auto blobs = spec.experts.front().size();
            for (std::size_t b = 0; b < blobs; ++b) {
                const auto first = spec.experts.front()[b].bytes;
                const bool equal = std::all_of(spec.experts.begin(), spec.experts.end(),
                    [&](const auto& expert) { return expert[b].bytes == first; });
                // Kernels read these planes as the resident bank: every expert at its index.
                const bool resident = equal && first % 4 == 0;
                layer.stride.push_back(resident ? first : aligned(largest[b]));
                bank_size[b] = std::max(bank_size[b],
                                        checked_add(checked_mul(count, layer.stride.back()), tail[b]));
            }
            layer.slot_of.assign(count, -1);
            layer.lent_of.assign(count, -1);
            layer.bank_of.assign(count, -1);
            const auto bytes = count * spec.aliases.size() * sizeof(void*);
            layer.mirror = std::make_unique<PinnedHostBuffer>(bytes);
            std::memset(layer.mirror->data(), 0, bytes);
            for (int u = 0; u < 2; ++u) {
                layer.upload[u] = std::make_unique<PinnedHostBuffer>(bytes);
                CUDA_CHECK(cudaEventCreateWithFlags(&layer.uploaded[u], cudaEventDisableTiming));
                allocation.pinned_bytes += bytes;
            }
            layer.source = std::move(spec);
            allocation.pinned_bytes += bytes;
            layers.push_back(std::move(layer));
        }
        for (const auto size : bank_size) {
            bank_offset.push_back(bank_bytes);
            bank_bytes = checked_add(bank_bytes, aligned(size));
        }
        cached_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        missing_ids = Storage(std::size_t(options.pair_capacity) * 4, rank.device, options.suspendable);
        lists =
            Storage((2 * std::size_t(1 + max_experts) + 1) * 4, rank.device, options.suspendable);
        lists.fill();
        cached_experts  = static_cast<std::int32_t*>(lists.p);
        missing_experts = cached_experts + 1 + max_experts;
        handshake.emplace(rank,
                          reinterpret_cast<std::uint32_t*>(missing_experts + 1 + max_experts));
        // Mapped: the split writes it.
        needed_host = std::make_unique<PinnedHostBuffer>(std::size_t(max_experts) * 4);
        std::memset(needed_host->data(), 0, needed_host->size());
        CUDA_CHECK(cudaHostGetDevicePointer(&needed_mapped, needed_host->data(), 0));
        allocation.metadata_bytes =
            cached_ids.backing_bytes() + missing_ids.backing_bytes() + lists.backing_bytes();
        allocation.pinned_bytes += needed_host->size() + handshake->pinned_bytes();
        for (auto& layer : layers) {
            const auto bytes = layer.mirror->size();
            layer.tables = Storage(bytes, rank.device, options.suspendable);
            layer.tables.fill();
            allocation.metadata_bytes += layer.tables.backing_bytes();
        }
        if (disk) {
            half_bytes = aligned(std::max<std::uint64_t>(options.staging_bytes / 2, 256));
            staging = std::make_unique<PinnedHostBuffer>(2 * half_bytes);
            allocation.pinned_bytes += staging->size();
            for (const auto& path : paths) {
                files.emplace_back(path, std::ios::binary);
                if (!files.back()) {
                    throw std::runtime_error("expert delivery: cannot open " + path.string());
                }
            }
            read_queue = FileReadQueue::open(paths, false, 16);
        }
        const std::uint64_t granule = options.suspendable ? vmm::granularity(rank.device) : 1;
        if (!granule) {
            throw std::invalid_argument("expert delivery: suspend needs VMM allocation granularity");
        }
        const std::uint64_t cache_granule = options.elastic ? vmm::granularity(rank.device) : granule;
        if (!cache_granule) {
            throw std::invalid_argument("expert delivery: an elastic cache needs VMM allocation");
        }
        unit                = options.elastic ? cache_granule : 1;
        const auto physical = [&](std::uint64_t bytes) {
            return checked_add(bytes, granule - 1) / granule * granule;
        };
        const auto available = [&] {
            std::size_t free = 0, total = 0;
            CUDA_CHECK(cudaMemGetInfo(&free, &total));
            return free > options.keep_free_bytes ? free - options.keep_free_bytes : 0;
        };
        // Banks before the cache: every miss lands in one. A second bank is dropped before the
        // last one, and an explicit cache keeps its size by dropping the second bank first.
        std::uint32_t bank_count = options.banks;
        std::uint64_t explicit_cache = 0;
        if (options.cache_bytes) {
            const auto counts =
                fill_pools(*options.cache_bytes, std::vector<std::uint32_t>(pools.size(), 0));
            for (std::size_t p = 0; p < pools.size(); ++p) {
                explicit_cache = checked_add(explicit_cache, backing(pools[p], counts[p]));
            }
            explicit_cache = physical(explicit_cache);
        }
        while (bank_count > 1 && bank_count * physical(bank_bytes) + explicit_cache > available()) {
            --bank_count;
        }
        if (physical(bank_bytes) > available()) {
            throw std::invalid_argument(
                "expert delivery: a layer bank needs " + std::to_string(physical(bank_bytes)) +
                " device bytes, but only " + std::to_string(available()) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        banks.reserve(bank_count);
        for (std::uint32_t b = 0; b < bank_count; ++b) {
            banks.emplace_back(bank_bytes, rank.device, options.suspendable);
            banks.back().fill();
            bank_state.push_back({-1, std::vector<unsigned char>(max_experts, 0)});
            allocation.bank_bytes += banks.back().backing_bytes();
        }
        allocation.banks = bank_count;
        // A narrow call routes fewer than half a layer's experts, so one lent bank whose slots
        // number at least that many always has a slot for a miss that its call does not route.
        bank_lent.assign(bank_count, 0);
        if (options.lend_banks && 2 * (bank_bytes / slot_bytes) >= max_experts) {
            per_bank = std::uint32_t(bank_bytes / slot_bytes);
            lent_slots.resize(std::size_t(per_bank) * bank_count);
            for (auto& slot : lent_slots) { slot.written.assign(slot_offset.size(), 0); }
        }
        lent_ranges = Storage(4 * sizeof(std::uintptr_t), rank.device, options.suspendable);
        lent_ranges.fill();
        host_ranges = std::make_unique<PinnedHostBuffer>(4 * sizeof(std::uintptr_t));
        std::memset(host_ranges->data(), 0, host_ranges->size());
        allocation.metadata_bytes += lent_ranges.backing_bytes();
        allocation.pinned_bytes += host_ranges->size();
        const std::uint64_t remaining = available();
        // Automatic slots may use only whole physical granules inside the remaining budget.
        // An explicit request must preserve the same reserve, including VMM rounding.
        const auto budget = options.cache_bytes.value_or(remaining / cache_granule * cache_granule);
        const auto counts = fill_pools(budget, std::vector<std::uint32_t>(pools.size(), 0));
        std::uint64_t cache_bytes = 0, reserve = 0;
        std::int32_t first = 0;
        for (std::size_t p = 0; p < pools.size(); ++p) {
            auto& pool = pools[p];
            pool.first = first;
            first += std::int32_t(pool.capacity);
            pool.count = counts[p];
            if (options.elastic) {
                // Each pool's reservation starts at a granule, so its slots map and unmap alone.
                pool.span =
                    checked_add(checked_mul(pool.capacity, pool.slot_bytes), cache_granule - 1) /
                    cache_granule * cache_granule;
                pool.base = reserve;
                reserve   = checked_add(reserve, pool.span);
            } else {
                pool.span = checked_mul(pool.count, pool.slot_bytes);
                pool.base = cache_bytes;
            }
            cache_bytes = checked_add(cache_bytes, backing(pool, pool.count));
        }
        const auto backing_bytes = checked_add(cache_bytes, cache_granule - 1) / cache_granule * cache_granule;
        if (backing_bytes > remaining) {
            throw std::invalid_argument(
                "expert delivery: cache needs " + std::to_string(backing_bytes) +
                " device bytes, but only " + std::to_string(remaining) +
                " remain after keep_free_bytes=" + std::to_string(options.keep_free_bytes));
        }
        if (options.elastic) {
            cache.growable = GrowableRegion(reserve, {.device = rank.device});
            cache.p = cache.growable.data();
            for (const auto& pool : pools) {
                if (!cache.growable.map(pool.base, pool.count * pool.slot_bytes)) {
                    throw std::invalid_argument(
                        "expert delivery: the device cannot back the cache");
                }
            }
            cache.bytes = cache_bytes;
        }
        if (cache_bytes) {
            if (!options.elastic) { cache = Storage(cache_bytes, rank.device, options.suspendable); }
            cache.fill();
        }
        slots.resize(std::size_t(first));
        for (std::uint32_t p = 0; p < pools.size(); ++p) {
            auto& pool = pools[p];
            for (std::uint32_t i = 0; i < pool.capacity; ++i) {
                auto& slot = slots[std::size_t(pool.first) + i];
                slot.pool  = p;
                slot.written.assign(pool.offset.size(), 0);
            }
            reset_free(pool);
        }
        cache_span = options.elastic ? cache.growable.bytes() : cache.bytes;
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = cache_slots();
        call_routed.assign(max_experts, 0);
        call_cpu.assign(max_experts, 0);
        call_count.assign(max_experts, 0);
        std::vector<std::uint32_t> widths, layer_pools;
        for (const auto& layer : layers) {
            widths.push_back(std::uint32_t(layer.source.experts.size()));
            layer_pools.push_back(layer.pool);
        }
        tier = std::make_unique<ExpertTier>(widths, layer_pools, options.tier);
        set_free();
        if (disk) {
            round_staging = std::make_unique<PinnedHostBuffer>(
                std::size_t(std::max(kRoundFileBytes, largest_expert)));
            allocation.pinned_bytes += round_staging->size();
        }
        if (options.cpu.products != nullptr) {
            const auto pinned = [&](std::size_t bytes) {
                auto buffer = std::make_unique<PinnedHostBuffer>(std::max<std::size_t>(bytes, 64));
                std::memset(buffer->data(), 0, buffer->size());
                allocation.pinned_bytes += buffer->size();
                return buffer;
            };
            host_inputs   = pinned(options.cpu.input_bytes);
            host_ids      = pinned(std::size_t(options.pair_capacity) * 4);
            // Mapped: the split writes them.
            CUDA_CHECK(cudaHostGetDevicePointer(&host_inputs_mapped, host_inputs->data(), 0));
            CUDA_CHECK(cudaHostGetDevicePointer(&host_ids_mapped, host_ids->data(), 0));
            host_missing  = pinned(std::size_t(options.pair_capacity + 1 + max_experts) * 4);
            host_products = pinned(options.cpu.product_bytes);
            host_flag     = pinned(4);
            products = Storage(options.cpu.product_bytes, rank.device, options.suspendable);
            cpu_flag = Storage(4, rank.device, options.suspendable);
            products.fill();
            cpu_flag.fill();
            allocation.metadata_bytes += products.backing_bytes() + cpu_flag.backing_bytes();
            pool = std::make_unique<CpuSpinPool>(
                options.cpu.threads != 0 ? options.cpu.threads : pool_cpu_count(), true);
        }
        CUDA_CHECK(cudaFuncSetAttribute(split_routes, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        int(max_experts * 4)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&rounds, cudaStreamNonBlocking));
        try {
            for (auto& event : copied) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
            CUDA_CHECK(cudaEventCreateWithFlags(&round_staged, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreate(&copies_begun));
            CUDA_CHECK(cudaEventCreate(&copies_done));
            service = std::thread([this] { run(); });
        } catch (...) {
            for (auto event : copied) { if (event) { cudaEventDestroy(event); } }
            for (auto event : {round_staged, copies_begun, copies_done}) {
                if (event) { cudaEventDestroy(event); }
            }
            cudaStreamDestroy(copies);
            cudaStreamDestroy(rounds);
            throw;
        }
    }

    ~Impl() {
        DeviceBinding bind(rank.device);
        // Keep the worker alive until the last graph's waits and consumers finish.
        cudaStreamSynchronize(rank.stream);
        stopping.store(true, std::memory_order_release);
        if (service.joinable()) { service.join(); }
        cudaStreamSynchronize(copies);
        cudaStreamSynchronize(rounds);
        for (auto event : copied) { cudaEventDestroy(event); }
        for (auto& layer : layers) {
            for (auto event : layer.uploaded) { if (event) { cudaEventDestroy(event); } }
        }
        for (const auto& d : deferred) {
            if (d.landed && std::find(spare_events.begin(), spare_events.end(), d.landed) ==
                                spare_events.end()) {
                spare_events.push_back(d.landed);
            }
        }
        for (auto event : spare_events) { cudaEventDestroy(event); }
        for (auto event : {round_staged, copies_begun, copies_done}) { cudaEventDestroy(event); }
        cudaStreamDestroy(copies);
        cudaStreamDestroy(rounds);
    }

    void check() const {
        std::lock_guard lock(mutex);
        if (failure) { std::rethrow_exception(failure); }
    }

    template <typename Function> void each_storage(Function&& function) {
        for (Storage* storage : {&cache, &cached_ids, &missing_ids, &lists, &products, &cpu_flag,
                                 &lent_ranges}) {
            if (storage->bytes) { function(*storage); }
        }
        for (auto& bank : banks) { function(bank); }
        for (auto& layer : layers) { function(layer.tables); }
    }

    [[nodiscard]] std::byte* slot_base(std::int32_t slot) const {
        const auto& pool = pools[slots[std::size_t(slot)].pool];
        return static_cast<std::byte*>(cache.p) + pool.base +
               std::uint64_t(slot - pool.first) * pool.slot_bytes;
    }

    [[nodiscard]] std::byte* slot_blob(std::int32_t slot, std::size_t b) const {
        return slot_base(slot) + pools[slots[std::size_t(slot)].pool].offset[b];
    }

    [[nodiscard]] Pool& pool_of(std::size_t li) { return pools[layers[li].pool]; }

    // The bytes that back n slots of `pool` (`unit`).
    [[nodiscard]] std::uint64_t backing(const Pool& pool, std::uint64_t n) const {
        return checked_add(checked_mul(n, pool.slot_bytes), unit - 1) / unit * unit;
    }

    [[nodiscard]] std::uint32_t cache_slots() const {
        std::uint32_t n = 0;
        for (const auto& pool : pools) { n += pool.count; }
        return n;
    }

    // Every backed slot of `pool` free, popped lowest first.
    static void reset_free(Pool& pool) {
        pool.free.clear();
        for (std::int64_t i = std::int64_t(pool.count) - 1; i >= 0; --i) {
            pool.free.push_back(pool.first + std::int32_t(i));
        }
    }

    void set_free() {
        for (std::uint32_t p = 0; p < pools.size(); ++p) {
            tier->set_free(p, std::uint32_t(pools[p].free.size()));
        }
    }

    // Whether na slots of pool a give its layers fewer each than nb of pool b give b's (the lower
    // pool first among equals).
    [[nodiscard]] bool fewer(std::size_t a, std::uint32_t na, std::size_t b,
                             std::uint32_t nb) const {
        const auto x = std::uint64_t(na) * pools[b].layers, y = std::uint64_t(nb) * pools[a].layers;
        return x != y ? x < y : a < b;
    }

    // From `counts` (each pool's slots), one slot at a time to the pool whose layers hold the
    // fewest each, among those with experts left whose next slot keeps the backing within
    // `bytes`: every layer about as many slots as any other, whatever its experts' size.
    [[nodiscard]] std::vector<std::uint32_t> fill_pools(std::uint64_t bytes,
                                                        std::vector<std::uint32_t> counts) const {
        std::uint64_t used = 0;
        for (std::size_t p = 0; p < pools.size(); ++p) { used += backing(pools[p], counts[p]); }
        for (;;) {
            std::size_t best = pools.size();
            for (std::size_t p = 0; p < pools.size(); ++p) {
                if (counts[p] >= pools[p].capacity) { continue; }
                const auto more = backing(pools[p], counts[p] + 1) - backing(pools[p], counts[p]);
                if (more > bytes - std::min(bytes, used)) { continue; }
                if (best == pools.size() || fewer(p, counts[p], best, counts[best])) { best = p; }
            }
            if (best == pools.size()) { return counts; }
            used += backing(pools[best], counts[best] + 1) - backing(pools[best], counts[best]);
            ++counts[best];
        }
    }

    // From `counts`, one slot at a time from the pool whose layers hold the most each, until the
    // backing is within `bytes`.
    [[nodiscard]] std::vector<std::uint32_t> trim_pools(std::uint64_t bytes,
                                                        std::vector<std::uint32_t> counts) const {
        std::uint64_t used = 0;
        for (std::size_t p = 0; p < pools.size(); ++p) { used += backing(pools[p], counts[p]); }
        while (used > bytes) {
            std::size_t worst = pools.size();
            for (std::size_t p = 0; p < pools.size(); ++p) {
                if (counts[p] == 0) { continue; }
                if (worst == pools.size() || fewer(worst, counts[worst], p, counts[p])) {
                    worst = p;
                }
            }
            if (worst == pools.size()) { break; }
            used -= backing(pools[worst], counts[worst]) - backing(pools[worst], counts[worst] - 1);
            --counts[worst];
        }
        return counts;
    }
    [[nodiscard]] std::byte* bank_blob(int bank, std::size_t li, std::int32_t e, std::size_t b) const {
        return static_cast<std::byte*>(banks[std::size_t(bank)].p) + bank_offset[b] +
               std::uint64_t(e) * layers[li].stride[b];
    }

    // The table entries of (li, e) point at base(b) for each alias's blob, or at nothing.
    template <typename Base> void point(std::size_t li, std::int32_t e, Base&& base) {
        auto& layer = layers[li];
        const auto n = layer.source.experts.size();
        auto** mirror = static_cast<const void**>(layer.mirror->data());
        for (std::size_t a = 0; a < layer.source.aliases.size(); ++a) {
            const auto alias = layer.source.aliases[a];
            std::byte* blob = base(alias.blob);
            mirror[a * n + std::size_t(e)] = blob ? blob + alias.offset : nullptr;
        }
        layer.dirty = true;
    }
    // Before (li, e)'s entries point elsewhere: the lent slot holding it is free again.
    void leave(std::size_t li, std::int32_t e) {
        auto& layer = layers[li];
        layer.bank_of[std::size_t(e)] = -1;
        const auto l = layer.lent_of[std::size_t(e)];
        if (l < 0) { return; }
        layer.lent_of[std::size_t(e)] = -1;
        lent_slots[std::size_t(l)].layer = lent_slots[std::size_t(l)].expert = -1;
    }
    void point_slot(std::size_t li, std::int32_t e, std::int32_t slot) {
        leave(li, e);
        point(li, e, [&](std::size_t b) { return slot_blob(slot, b); });
    }
    void point_bank(std::size_t li, std::int32_t e, int bank) {
        leave(li, e);
        point(li, e, [&](std::size_t b) { return bank_blob(bank, li, e, b); });
        layers[li].bank_of[std::size_t(e)] = std::int8_t(bank);
    }
    void point_nowhere(std::size_t li, std::int32_t e) {
        leave(li, e);
        point(li, e, [](std::size_t) { return static_cast<std::byte*>(nullptr); });
    }
    [[nodiscard]] std::byte* lent_base(std::int32_t l) const {
        const auto bank = std::uint32_t(l) / per_bank;
        return static_cast<std::byte*>(banks[bank].p) + std::uint64_t(std::uint32_t(l) % per_bank) * slot_bytes;
    }
    void point_lent(std::size_t li, std::int32_t e, std::int32_t l) {
        leave(li, e);
        std::byte* base = lent_base(l);
        point(li, e, [&](std::size_t b) { return base + slot_offset[b]; });
        layers[li].lent_of[std::size_t(e)] = l;
        auto& slot = lent_slots[std::size_t(l)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.used = stamp;
    }

    // Lends bank k: nothing points into its layout any more, and its zeroed bytes hold free slots.
    // The served call's split read every such entry as a miss, and earlier calls' consumers are done.
    void lend_bank(int k) {
        for (std::size_t li = 0; li < layers.size(); ++li) {
            for (std::size_t e = 0; e < layers[li].bank_of.size(); ++e) {
                if (layers[li].bank_of[e] == k) { point_nowhere(li, std::int32_t(e)); }
            }
        }
        bank_state[std::size_t(k)].layer = -1;
        std::fill(bank_state[std::size_t(k)].valid.begin(), bank_state[std::size_t(k)].valid.end(), 0);
        CUDA_CHECK(cudaMemsetAsync(banks[std::size_t(k)].p, 0, banks[std::size_t(k)].bytes, copies));
        for (std::uint32_t i = 0; i < per_bank; ++i) {
            auto& slot = lent_slots[std::size_t(k) * per_bank + i];
            slot.layer = slot.expert = -1;
            std::fill(slot.written.begin(), slot.written.end(), 0);
        }
        bank_lent[std::size_t(k)] = 1;
        auto* ranges = static_cast<std::uintptr_t*>(host_ranges->data());
        ranges[2 * k]     = reinterpret_cast<std::uintptr_t>(banks[std::size_t(k)].p);
        ranges[2 * k + 1] = ranges[2 * k] + std::uint64_t(per_bank) * slot_bytes;
        ranges_dirty = true;
        set_lent_count();
    }

    // A wide call takes bank k back: the experts of its slots leave (the call read their entries
    // as misses).
    void reclaim_bank(int k, ExpertDeliveryStats& delta) {
        for (std::uint32_t i = 0; i < per_bank; ++i) {
            const auto& slot = lent_slots[std::size_t(k) * per_bank + i];
            if (slot.layer < 0) { continue; }
            point_nowhere(std::size_t(slot.layer), slot.expert);
            ++delta.lent_evictions;
        }
        bank_lent[std::size_t(k)] = 0;
        auto* ranges = static_cast<std::uintptr_t*>(host_ranges->data());
        ranges[2 * k] = ranges[2 * k + 1] = 0;
        ranges_dirty = true;
        set_lent_count();
    }

    void set_lent_count() {
        std::uint32_t lent = 0;
        for (const auto flag : bank_lent) { lent += flag ? per_bank : 0; }
        std::lock_guard lock(mutex);
        allocation.lent_slots = lent;
    }

    [[nodiscard]] bool any_lent() const {
        return std::any_of(bank_lent.begin(), bank_lent.end(), [](unsigned char flag) { return flag != 0; });
    }

    // The lent slot a narrow call of layer li fills next, other than those it took (`taken`): a
    // free one, else the least routed expert (the tier's counts, then the oldest use) that the call
    // does not route, the lower slot first among equals. -1: none. A call's misses take them one
    // by one; nothing they change moves another slot's place in this order.
    [[nodiscard]] std::int32_t next_lent(std::size_t li,
                                         std::span<const std::int32_t> taken) const {
        std::int32_t best = -1;
        std::tuple<int, float, std::uint64_t> best_key{};
        for (std::size_t k = 0; k < bank_lent.size(); ++k) {
            if (!bank_lent[k]) { continue; }
            for (std::uint32_t i = 0; i < per_bank; ++i) {
                const auto l = std::int32_t(k * per_bank + i);
                const auto& slot = lent_slots[std::size_t(l)];
                if (slot.layer == std::int32_t(li) && call_routed[std::size_t(slot.expert)]) { continue; }
                if (std::find(taken.begin(), taken.end(), l) != taken.end()) { continue; }
                const auto key =
                    slot.layer < 0
                        ? std::tuple{0, 0.0F, std::uint64_t(0)}
                        : std::tuple{1, tier->count(std::size_t(slot.layer), slot.expert),
                                     slot.used};
                if (best < 0 || key < best_key) {
                    best     = l;
                    best_key = key;
                }
            }
        }
        return best;
    }

    // Copies a narrow call's missing expert e of layer li into lent slot l.
    void admit_lent(std::int32_t l, std::size_t li, std::int32_t e, ExpertDeliveryStats& delta) {
        const auto& slot = lent_slots[std::size_t(l)];
        if (slot.layer >= 0) {
            point_nowhere(std::size_t(slot.layer), slot.expert);
            ++delta.lent_evictions;
        }
        auto& written = lent_slots[std::size_t(l)].written;
        std::byte* base = lent_base(l);
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto& blob = layers[li].source.experts[std::size_t(e)][b];
            add_piece(base + slot_offset[b], blob);
            if (written[b] > blob.bytes) {
                zeroes.push_back({base + slot_offset[b] + blob.bytes, written[b] - blob.bytes});
            }
            written[b] = blob.bytes;
        }
        point_lent(li, e, l);
        ++delta.lent_admissions;
    }

    // One copy per changed table, from a pinned copy no earlier upload still reads.
    void upload_tables() {
        for (auto& layer : layers) {
            if (!layer.dirty) { continue; }
            const int u = layer.next_upload;
            layer.next_upload = 1 - u;
            CUDA_CHECK(cudaEventSynchronize(layer.uploaded[u]));
            std::memcpy(layer.upload[u]->data(), layer.mirror->data(), layer.mirror->size());
            CUDA_CHECK(cudaMemcpyAsync(layer.tables.p, layer.upload[u]->data(), layer.mirror->size(),
                                       cudaMemcpyHostToDevice, copies));
            CUDA_CHECK(cudaEventRecord(layer.uploaded[u], copies));
            layer.dirty = false;
        }
    }

    // Queue a host-to-device copy, extending the previous piece when both its source and its
    // target continue it.
    void add_piece(std::byte* target, const ExpertBlob& blob) {
        if (!pieces.empty()) {
            auto& last = pieces.back();
            const bool target_follows = last.target + last.bytes == target;
            if (target_follows && blob.pinned && last.pinned && blob.pinned_region != 0 &&
                blob.pinned_region == last.region &&
                last.pinned + last.bytes == static_cast<const std::byte*>(blob.pinned)) {
                last.bytes += blob.bytes;
                return;
            }
            if (target_follows && !blob.pinned && !last.pinned) {
                for (const auto& extent : blob.extents) {
                    auto& end = last.extents.back();
                    if (end.file == extent.file && end.offset + end.bytes == extent.offset) {
                        end.bytes += extent.bytes;
                    } else {
                        last.extents.push_back(extent);
                    }
                }
                last.bytes += blob.bytes;
                return;
            }
        }
        Piece piece;
        piece.target = target;
        piece.pinned = static_cast<const std::byte*>(blob.pinned);
        piece.region = blob.pinned_region;
        piece.extents = blob.extents;
        piece.bytes = blob.bytes;
        pieces.push_back(std::move(piece));
    }

    // Several reads share one bounded half. FileReadQueue can submit their extents together.
    // Its fallback reads the same ranges through the OS page cache.
    void read_now(std::span<const QueuedFileRead> reads) {
        if (read_queue) {
            read_queue->read(reads);
            return;
        }
        for (const auto& read : reads) {
            auto& file = files.at(read.file);
            file.clear();
            file.seekg(std::streamoff(read.offset));
            file.read(reinterpret_cast<char*>(read.destination.data()),
                      std::streamsize(read.destination.size()));
            if (!file) { throw std::runtime_error("expert delivery: incomplete file read"); }
        }
    }

    void flush_reads() {
        if (batch_reads.empty()) { return; }
        CUDA_CHECK(cudaEventSynchronize(copied[half]));
        read_now(batch_reads);
        for (const auto& copy : batch_copies) {
            CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                       cudaMemcpyHostToDevice, copies));
        }
        CUDA_CHECK(cudaEventRecord(copied[half], copies));
        batch_reads.clear();
        batch_copies.clear();
        staging_cursor = 0;
        half = 1 - half;
    }

    void stage(const Piece& piece, ExpertDeliveryStats& delta) {
        std::uint64_t destination = 0;
        for (const auto& extent : piece.extents) {
            std::uint64_t within = 0;
            while (within < extent.bytes) {
                if (staging_cursor == half_bytes) { flush_reads(); }
                auto* buffer = static_cast<std::uint8_t*>(staging->data()) +
                               half * half_bytes + staging_cursor;
                const auto bytes = std::min(extent.bytes - within, half_bytes - staging_cursor);
                batch_reads.push_back({extent.file, extent.offset + within,
                                       {buffer, std::size_t(bytes)}});
                batch_copies.push_back({piece.target + destination, buffer, std::size_t(bytes)});
                staging_cursor += bytes;
                destination += bytes;
                within += bytes;
                delta.read_bytes += bytes;
            }
        }
    }

    void flush_pieces(ExpertDeliveryStats& delta) {
        for (const auto& piece : pieces) {
            if (piece.pinned) {
                CUDA_CHECK(cudaMemcpyAsync(piece.target, piece.pinned, piece.bytes,
                                           cudaMemcpyHostToDevice, copies));
            } else {
                stage(piece, delta);
            }
            delta.copied_bytes += piece.bytes;
        }
        flush_reads();
        for (const auto& [target, bytes] : zeroes) {
            CUDA_CHECK(cudaMemsetAsync(target, 0, bytes, copies));
        }
        pieces.clear();
        zeroes.clear();
    }

    // Bytes of blob b written into slot s: zero what a larger earlier blob left past them, so the
    // slot's tail stays zero.
    void slot_written(std::int32_t s, std::size_t b, std::uint64_t bytes) {
        auto& written = slots[std::size_t(s)].written[b];
        if (written > bytes) { zeroes.push_back({slot_blob(s, b) + bytes, written - bytes}); }
        written = bytes;
    }

    void evict(std::int32_t s, ExpertDeliveryStats& delta) {
        auto& slot = slots[std::size_t(s)];
        auto& layer = layers[std::size_t(slot.layer)];
        layer.slot_of[std::size_t(slot.expert)] = -1;
        --layer.cached;
        point_nowhere(std::size_t(slot.layer), slot.expert);
        tier->set(std::size_t(slot.layer), slot.expert, ExpertTier::State::Out);
        slot.layer = slot.expert = -1;
        slot.pending = false;
        ++delta.evictions;
    }

    // A free slot for the served call's missing expert e of layer li, routed `pairs` times, else
    // the slot of the coldest expert the call does not route when the tier admits e. -1: none.
    std::int32_t admit_slot(std::size_t li, std::int32_t e, std::uint32_t pairs,
                            ExpertDeliveryStats& delta) {
        auto& free = pool_of(li).free;
        if (!free.empty()) {
            const auto s = free.back();
            free.pop_back();
            return s;
        }
        const auto victim = tier->admission_victim(li, e, float(pairs), call_routed);
        if (victim.expert < 0) { return -1; }
        const auto s = layers[victim.layer].slot_of[std::size_t(victim.expert)];
        if (s < 0 || slots[std::size_t(s)].pending) { return -1; }
        evict(s, delta);
        return s;
    }

    void occupy(std::int32_t s, std::size_t li, std::int32_t e, bool pending) {
        auto& slot = slots[std::size_t(s)];
        slot.layer = std::int32_t(li);
        slot.expert = e;
        slot.pending = pending;
        layers[li].slot_of[std::size_t(e)] = s;
        ++layers[li].cached;
        tier->set(li, e, pending ? ExpertTier::State::Pending : ExpertTier::State::Held);
    }

    cudaEvent_t take_event() {
        if (!spare_events.empty()) {
            const auto event = spare_events.back();
            spare_events.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return event;
    }

    // Promotions of an earlier wide call (their copies precede the next table upload on
    // `copies`) and round candidates whose copies have landed: their entries move to their slots.
    void land() {
        std::vector<cudaEvent_t> done;
        for (auto it = deferred.begin(); it != deferred.end();) {
            if (it->landed) {
                const cudaError_t status = cudaEventQuery(it->landed);
                if (status == cudaErrorNotReady) {
                    ++it;
                    continue;
                }
                CUDA_CHECK(status);
                if (std::find(done.begin(), done.end(), it->landed) == done.end()) {
                    done.push_back(it->landed);
                }
            }
            auto& slot = slots[std::size_t(it->slot)];
            if (slot.layer == std::int32_t(it->layer) && slot.expert == it->expert && slot.pending) {
                slot.pending = false;
                point_slot(it->layer, it->expert, it->slot);
                tier->set(it->layer, it->expert, ExpertTier::State::Held);
            }
            it = deferred.erase(it);
        }
        for (const auto event : done) {
            const bool used = std::any_of(deferred.begin(), deferred.end(),
                                          [&](const Deferred& d) { return d.landed == event; });
            if (!used) { spare_events.push_back(event); }
        }
    }

    // A pass has ended: the tier's round. Victims leave now (never an expert the served call
    // routes); candidates hold their slots pending until post work has copied them in.
    void round(ExpertDeliveryStats& delta) {
        set_free();
        for (const auto& swap : tier->end_pass()) {
            auto& layer = layers[swap.layer];
            if (layer.slot_of[std::size_t(swap.in)] >= 0) { continue; }
            std::int32_t s = -1;
            if (swap.out < 0) {
                auto& free = pools[layer.pool].free;
                if (free.empty()) { continue; }
                s = free.back();
                free.pop_back();
            } else {
                s = layers[swap.out_layer].slot_of[std::size_t(swap.out)];
                if (s < 0 || slots[std::size_t(s)].pending ||
                    (swap.out_layer == call_layer && call_routed[std::size_t(swap.out)])) {
                    continue;
                }
                evict(s, delta);
            }
            occupy(s, swap.layer, swap.in, true);
            queued.push_back({swap.layer, swap.in, s, nullptr});
            ++delta.admissions;
        }
    }

    // Post work of a narrow call: copies of the oldest queued candidates on the side stream, a few
    // experts and file bytes at a time, so the next call's service is not delayed.
    void issue_rounds(ExpertDeliveryStats& delta) {
        if (queued.empty()) { return; }
        const cudaEvent_t done = take_event();
        std::uint64_t staged = 0;
        std::size_t issued = 0;
        std::vector<QueuedFileRead> reads;
        std::vector<StagedCopy> staged_copies;
        auto* staging = round_staging ? static_cast<std::uint8_t*>(round_staging->data()) : nullptr;
        for (; issued < queued.size() && issued < kRoundExperts; ++issued) {
            const auto& d = queued[issued];
            const auto& expert = layers[d.layer].source.experts[std::size_t(d.expert)];
            std::uint64_t file_bytes = 0;
            for (const auto& blob : expert) { file_bytes += blob.pinned ? 0 : blob.bytes; }
            if (file_bytes != 0 && staged != 0 &&
                (staged + file_bytes > round_staging->size() || staged + file_bytes > kRoundFileBytes)) {
                break;
            }
            if (file_bytes != 0 && staged == 0) { CUDA_CHECK(cudaEventSynchronize(round_staged)); }
            for (std::size_t b = 0; b < expert.size(); ++b) {
                const auto& blob = expert[b];
                std::byte* target = slot_blob(d.slot, b);
                if (blob.pinned) {
                    CUDA_CHECK(cudaMemcpyAsync(target, blob.pinned, blob.bytes, cudaMemcpyHostToDevice,
                                               rounds));
                } else {
                    std::uint64_t at = 0;
                    for (const auto& extent : blob.extents) {
                        reads.push_back({extent.file, extent.offset,
                                         {staging + staged + at, std::size_t(extent.bytes)}});
                        at += extent.bytes;
                    }
                    staged_copies.push_back({target, staging + staged, std::size_t(blob.bytes)});
                    staged += blob.bytes;
                    delta.read_bytes += blob.bytes;
                }
                auto& written = slots[std::size_t(d.slot)].written[b];
                if (written > blob.bytes) {
                    CUDA_CHECK(cudaMemsetAsync(target + blob.bytes, 0, written - blob.bytes, rounds));
                }
                written = blob.bytes;
                delta.copied_bytes += blob.bytes;
            }
        }
        if (!reads.empty()) {
            read_now(reads);
            for (const auto& copy : staged_copies) {
                CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                           cudaMemcpyHostToDevice, rounds));
            }
            CUDA_CHECK(cudaEventRecord(round_staged, rounds));
        }
        CUDA_CHECK(cudaEventRecord(done, rounds));
        for (std::size_t i = 0; i < issued; ++i) {
            auto d = queued[i];
            d.landed = done;
            deferred.push_back(d);
        }
        queued.erase(queued.begin(), queued.begin() + std::ptrdiff_t(issued));
        if (issued == 0) { spare_events.push_back(done); }
    }

    [[nodiscard]] int bank_holding(std::size_t li) const {
        for (std::size_t b = 0; b < bank_state.size(); ++b) {
            if (bank_state[b].layer == std::int64_t(li)) { return int(b); }
        }
        return -1;
    }

    // A served call may write any bank: the preceding call's consumers finished before it was
    // published. It takes the bank the last bank-using call did not use, which keeps that layer
    // for an immediate repeat (a second prefill slice, the next draft of the same layer).
    [[nodiscard]] int pick_bank(int avoid) const {
        return bank_state.size() == 1 ? 0 : avoid >= 0 ? 1 - avoid : 1 - last_bank;
    }
    void claim_bank(int k, std::size_t li) {
        auto& bank = bank_state[std::size_t(k)];
        bank.layer = std::int64_t(li);
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }

    // Copy experts of layer li into bank k at their indices, one piece per contiguous run, and keep
    // zeros after the last expert of each run that the bank does not already continue.
    void fill_bank(int k, std::size_t li, const std::vector<std::int32_t>& experts) {
        auto& layer = layers[li];
        auto& bank = bank_state[std::size_t(k)];
        const auto blobs = layer.source.experts.front().size();
        for (std::size_t b = 0; b < blobs; ++b) {
            for (const auto e : experts) {
                add_piece(bank_blob(k, li, e, b), layer.source.experts[std::size_t(e)][b]);
            }
        }
        for (const auto e : experts) { bank.valid[std::size_t(e)] = 1; }
        for (std::size_t b = 0; b < blobs; ++b) {
            if (!tail[b]) { continue; }
            for (const auto e : experts) {
                const auto next = std::size_t(e) + 1;
                if (next < layer.source.experts.size() && bank.valid[next]) { continue; }
                const auto end = std::uint64_t(e) * layer.stride[b] +
                                 layer.source.experts[std::size_t(e)][b].bytes;
                const auto room = bank_size[b] - std::min(bank_size[b], end);
                if (room) {
                    zeroes.push_back({static_cast<std::byte*>(banks[std::size_t(k)].p) +
                                          bank_offset[b] + end,
                                      std::min(tail[b], room)});
                }
            }
        }
    }

    // Device copies of bank expert e into slot s.
    void promote(std::int32_t s, std::size_t li, std::int32_t e, int bank, ExpertDeliveryStats& delta) {
        for (std::size_t b = 0; b < layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto bytes = layers[li].source.experts[std::size_t(e)][b].bytes;
            CUDA_CHECK(cudaMemcpyAsync(slot_blob(s, b), bank_blob(bank, li, e, b), bytes,
                                       cudaMemcpyDeviceToDevice, copies));
            slot_written(s, b, bytes);
            delta.promoted_bytes += bytes;
        }
        ++delta.admissions;
    }

    // The CPU's share of a narrow call's misses that neither a slot nor the layer's bank holds:
    // the experts with the fewest pairs (a copy serves all of an expert's pairs at once, the CPU's
    // work grows with each). Marks them in call_cpu.
    std::vector<std::int32_t> cpu_share(std::size_t li, const std::uint32_t* needed, int held) {
        const auto& layer = layers[li];
        std::vector<std::pair<std::uint32_t, std::int32_t>> order;
        for (int e = 0; e < int(layer.source.experts.size()); ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            const auto s = layer.slot_of[std::size_t(e)];
            if (s >= 0 && !slots[std::size_t(s)].pending) { continue; }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) { continue; }
            order.emplace_back(needed[e] & kCountMask, e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t count = 0;
        if (options.cpu.balance) {
            std::vector<int> pairs;
            for (const auto& [n, e] : order) { pairs.push_back(int(n)); }
            count = split.cpu_count(pairs, megabytes(li));
        } else {
            count = std::min(order.size(),
                             std::size_t(std::lround(double(order.size()) * options.cpu.share)));
        }
        std::vector<std::int32_t> out;
        for (std::size_t i = order.size() - count; i < order.size(); ++i) {
            out.push_back(order[i].second);
            call_cpu[std::size_t(order[i].second)] = 1;
        }
        return out;
    }

    [[nodiscard]] double megabytes(std::size_t li) const {
        double bytes = 0;
        for (const auto& blob : layers[li].source.experts.front()) { bytes += double(blob.bytes); }
        return bytes / 1e6;
    }

    // The CPU's products of `cpu` for the served call of `pairs` pairs while this thread issues
    // the call's copies (`issue`; before a disk expert's reads, which they would wait behind),
    // then its missing ids without the CPU's pairs, its products and flag for the device.
    void compute_cpu(std::size_t li, const std::vector<std::int32_t>& cpu, std::uint32_t pairs,
                     const std::uint32_t* needed, ExpertDeliveryStats& delta,
                     const std::function<void()>& issue) {
        auto* flag = static_cast<std::int32_t*>(host_flag->data());
        if (cpu.empty()) {
            issue();
            if (*flag != 0) {
                *flag = 0;
                CUDA_CHECK(cudaMemcpyAsync(cpu_flag.p, flag, 4, cudaMemcpyHostToDevice, copies));
            }
            return;
        }
        const auto& layer = layers[li];
        std::vector<std::vector<const std::byte*>> blobs(cpu.size());
        std::vector<QueuedFileRead> reads;
        std::uint64_t file_bytes = 0;
        for (const auto e : cpu) {
            for (const auto& blob : layer.source.experts[std::size_t(e)]) {
                file_bytes += blob.pinned ? 0 : blob.bytes;
            }
        }
        cpu_bytes.resize(std::size_t(file_bytes));
        std::uint64_t at = 0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            for (const auto& blob : layer.source.experts[std::size_t(cpu[i])]) {
                if (blob.pinned) {
                    blobs[i].push_back(static_cast<const std::byte*>(blob.pinned));
                    continue;
                }
                blobs[i].push_back(cpu_bytes.data() + at);
                for (const auto& extent : blob.extents) {
                    reads.push_back({extent.file, extent.offset,
                                     {reinterpret_cast<std::uint8_t*>(cpu_bytes.data() + at),
                                      std::size_t(extent.bytes)}});
                    at += extent.bytes;
                }
            }
        }
        if (!reads.empty()) {
            issue();
            read_now(reads);
        }
        const auto* ids = static_cast<const std::int32_t*>(host_ids->data());
        ExpertCpuProducts::Work work;
        work.layer    = li;
        work.experts  = cpu;
        work.blobs    = blobs;
        work.ids      = {ids, pairs};
        work.inputs   = static_cast<const std::byte*>(host_inputs->data());
        work.products = static_cast<std::byte*>(host_products->data());
        work.pool     = pool.get();
        if (reads.empty()) { work.side = issue; }
        const auto [used, microseconds] = options.cpu.products->compute(work);
        if (used > options.cpu.product_bytes) {
            throw std::logic_error("expert delivery: CPU products exceed their buffer");
        }
        // The missing set without the CPU's experts: per pair, then as a list.
        auto* missing = static_cast<std::int32_t*>(host_missing->data());
        auto* listed      = missing + options.pair_capacity;
        const int experts = int(layer.source.experts.size());
        const auto left   = [&](std::int32_t e) {
            return e >= 0 && e < experts && (needed[e] & kMiss) && !call_cpu[std::size_t(e)];
        };
        for (std::uint32_t p = 0; p < pairs; ++p) { missing[p] = left(ids[p]) ? ids[p] : -1; }
        listed[0] = 0;
        for (int e = 0; e < experts; ++e) {
            if (left(e)) { listed[1 + listed[0]++] = e; }
        }
        CUDA_CHECK(cudaMemcpyAsync(missing_ids.p, missing, std::size_t(pairs) * 4, cudaMemcpyHostToDevice,
                                   copies));
        CUDA_CHECK(cudaMemcpyAsync(missing_experts, listed, std::size_t(1 + listed[0]) * 4,
                                   cudaMemcpyHostToDevice, copies));
        if (used != 0) {
            CUDA_CHECK(cudaMemcpyAsync(products.p, host_products->data(), used, cudaMemcpyHostToDevice,
                                       copies));
        }
        *flag = 1;
        CUDA_CHECK(cudaMemcpyAsync(cpu_flag.p, flag, 4, cudaMemcpyHostToDevice, copies));
        double work_done = 0;
        for (const auto e : cpu) {
            const int n = int(needed[e] & kCountMask);
            delta.cpu_pairs += std::uint64_t(n);
            work_done += ExpertSplit::cpu_work(megabytes(li), n);
        }
        delta.cpu_experts += cpu.size();
        delta.read_bytes += file_bytes;
        if (options.cpu.balance) { split.computed(work_done, microseconds); }
    }

    void serve(const CudaHostHandshake::Request& request) {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        land();
        const std::size_t li = request.tag;
        auto& layer = layers.at(li);
        call_layer = li;
        const auto* needed = static_cast<const std::uint32_t*>(needed_host->data());
        const int experts = int(layer.source.experts.size());
        delta.requests = 1;
        std::uint32_t routes = 0;
        std::fill(call_routed.begin(), call_routed.end(), 0);
        std::fill(call_cpu.begin(), call_cpu.end(), 0);
        for (int e = 0; e < experts; ++e) {
            const std::uint32_t count = needed[e] & kCountMask;
            call_count[std::size_t(e)] = count;
            if (!count) { continue; }
            ++routes;
            call_routed[std::size_t(e)] = 1;
            delta.pairs += count;
            if (needed[e] & kHit) {
                ++delta.hits;
                delta.pair_hits += count;
            }
        }
        delta.routes = routes;
        const bool wide = 2 * routes >= std::uint32_t(experts);
        ++stamp;
        for (int e = 0; e < experts; ++e) {
            const auto l = layer.lent_of[std::size_t(e)];
            if (call_routed[std::size_t(e)] && l >= 0) { lent_slots[std::size_t(l)].used = stamp; }
        }
        if (previous_layer >= 0) {
            layers[std::size_t(previous_layer)].successor = std::int64_t(li);
            layers[std::size_t(previous_layer)].successor_wide = wide;
        }
        previous_layer = std::int64_t(li);
        const bool new_pass = !wide && last_narrow >= 0 && std::int64_t(li) < last_narrow;
        if (wide) { quiet_passes = 0; }
        if (per_bank && new_pass && ++quiet_passes >= 2) {
            // Only narrow calls ran for two passes: the banks become cache slots.
            for (std::size_t b = 0; b < banks.size(); ++b) {
                if (!bank_lent[b]) { lend_bank(int(b)); }
            }
        }
        const int held = bank_holding(li);
        int k = held;
        if (wide && per_bank) {
            // The wide call's bank: a lent one is taken back first.
            if (k < 0) { k = pick_bank(-1); }
            if (bank_lent[std::size_t(k)]) { reclaim_bank(k, delta); }
        }
        const bool lent = !wide && any_lent();
        std::vector<std::int32_t> lent_taken;
        const auto cpu = !wide && layer.cpu && (request.flags & kCpuCall) ? cpu_share(li, needed, held)
                                                                          : std::vector<std::int32_t>{};
        std::vector<std::int32_t> to_bank;
        for (int e = 0; e < experts; ++e) {
            if (!(needed[e] & kMiss)) { continue; }
            ++delta.misses;
            if (call_cpu[std::size_t(e)]) { continue; }
            const auto s = layer.slot_of[std::size_t(e)];
            if (s >= 0 && !slots[std::size_t(s)].pending) {
                // Promoted or landed after the split read its entry.
                point_slot(li, e, s);
                continue;
            }
            if (held >= 0 && bank_state[std::size_t(held)].valid[std::size_t(e)]) {
                ++delta.bank_reuses;
                point_bank(li, e, held);
                continue;
            }
            // A wide call reads a lent slot's entry as a miss; the slot keeps its bytes until a
            // later call takes its bank back.
            if (layer.lent_of[std::size_t(e)] >= 0) { continue; }
            // A round's candidate whose copy is still in flight is served from the bank.
            if (!wide && s < 0) {
                const auto slot = admit_slot(li, e, needed[e] & kCountMask, delta);
                if (slot >= 0) {
                    occupy(slot, li, e, false);
                    for (std::size_t b = 0; b < layer.source.experts[std::size_t(e)].size(); ++b) {
                        const auto& blob = layer.source.experts[std::size_t(e)][b];
                        add_piece(slot_blob(slot, b), blob);
                        slot_written(slot, b, blob.bytes);
                    }
                    ++delta.admissions;
                    point_slot(li, e, slot);
                    continue;
                }
            }
            if (lent) {
                // Lent banks take every miss no cache slot took, so none needs a bank.
                const auto l = next_lent(li, lent_taken);
                if (l < 0) {
                    throw std::logic_error("expert delivery: no lent slot for a narrow call's miss");
                }
                lent_taken.push_back(l);
                admit_lent(l, li, e, delta);
                continue;
            }
            to_bank.push_back(e);
        }
        // The tier learns from decode and verification only; a call below the last narrow one
        // starts a pass (a draft layer's repeated calls do not). It learns once the call's copies
        // are issued, so a pass's round does not hold them back; the call's tables, uploaded
        // after it, carry the round's evictions before any later layer runs.
        const auto learn = [&] {
            if (wide) { return; }
            if (new_pass) { round(delta); }
            last_narrow = std::int64_t(li);
            std::lock_guard lock(mutex);
            for (int e = 0; e < experts; ++e) {
                const auto count = call_count[std::size_t(e)];
                if (!count) { continue; }
                tier->route(li, e, float(count));
                layer.total[std::size_t(e)] += count;
            }
        };
        if (!to_bank.empty()) {
            if (k < 0) { k = pick_bank(-1); }
            if (k != held) { claim_bank(k, li); }
        }
        for (const auto e : to_bank) { point_bank(li, e, k); }
        if (k >= 0 && (held >= 0 || !to_bank.empty())) { last_bank = k; }
        call_bank = (held >= 0 || !to_bank.empty()) ? k : -1;
        call_wide = wide;
        if (!to_bank.empty()) { fill_bank(k, li, to_bank); }
        const bool timed = options.cpu.balance && !cpu.empty() && !pieces.empty();
        const auto copied_before = delta.copied_bytes;

        const auto issue = [&] {
            if (timed) { CUDA_CHECK(cudaEventRecord(copies_begun, copies)); }
            flush_pieces(delta);
            if (timed) { CUDA_CHECK(cudaEventRecord(copies_done, copies)); }
            learn();
        };
        if (layer.cpu && host_flag) {
            compute_cpu(li, cpu, request.width, needed, delta, issue);
        } else {
            issue();
        }
        if (ranges_dirty) {
            CUDA_CHECK(cudaMemcpyAsync(lent_ranges.p, host_ranges->data(), host_ranges->size(),
                                       cudaMemcpyHostToDevice, copies));
            ranges_dirty = false;
        }
        upload_tables();
        CUDA_CHECK(cudaStreamSynchronize(copies));
        if (timed) {
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, copies_begun, copies_done));
            split.copied(double(delta.copied_bytes - copied_before) / 1e6, 1e3 * double(milliseconds));
        }
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    // After the call's experts are ready: a wide call promotes its most routed bank experts into
    // free cache slots by device copies, up to the layer's share of the cache, so a first prompt
    // fills the cache across every layer and evicts nothing. It also starts the bank of the layer
    // that followed it last time.
    void post() {
        const auto started = std::chrono::steady_clock::now();
        ExpertDeliveryStats delta;
        if (!call_wide) {
            if (queued.empty()) { return; }
            issue_rounds(delta);
            delta.service_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            add(delta);
            return;
        }
        const std::size_t li = call_layer;
        auto& layer = layers[li];
        if (pool_of(li).count != 0 && call_bank >= 0) {
            const auto& bank = bank_state[std::size_t(call_bank)];
            std::vector<std::int32_t> candidates;
            for (std::size_t e = 0; e < layer.source.experts.size(); ++e) {
                if (call_routed[e] && bank.valid[e] && layer.slot_of[e] < 0) {
                    candidates.push_back(std::int32_t(e));
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](std::int32_t a, std::int32_t b) {
                const auto ca = call_count[std::size_t(a)], cb = call_count[std::size_t(b)];
                return ca != cb ? ca > cb : a < b;
            });
            auto& pool       = pool_of(li);
            const auto share = std::max<std::size_t>(1, pool.count / pool.layers);
            for (const auto e : candidates) {
                if (pool.free.empty() || layer.cached >= share) { break; }
                const auto s = pool.free.back();
                pool.free.pop_back();
                occupy(s, li, e, true);
                promote(s, li, e, call_bank, delta);
                deferred.push_back({li, e, s});
            }
        }
        if (bank_state.size() == 2 && layer.successor >= 0 && layer.successor_wide) {
            const auto next = std::size_t(layer.successor);
            const bool held = std::any_of(bank_state.begin(), bank_state.end(),
                                          [&](const Bank& b) { return b.layer == std::int64_t(next); });
            std::vector<std::int32_t> experts;
            for (std::size_t e = 0; e < layers[next].source.experts.size() && !held; ++e) {
                if (layers[next].slot_of[e] < 0) { experts.push_back(std::int32_t(e)); }
            }
            // Never the bank this call's consumers are reading, nor a lent one: the next call may
            // be narrow and read its slots.
            const int k = pick_bank(call_bank);
            if (!experts.empty() && !bank_lent[std::size_t(k)]) {
                claim_bank(k, next);
                fill_bank(k, next, experts);
                ++delta.prefetched_layers;
            }
        }
        flush_pieces(delta); // tables are unchanged: promoted entries are patched at the next request
        delta.service_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        add(delta);
    }

    void add(const ExpertDeliveryStats& d) {
        std::lock_guard lock(mutex);
        counters = counters + d;
    }

    void fail(const CudaHostHandshake::Request* request) {
        {
            std::lock_guard lock(mutex);
            if (!failure) { failure = std::current_exception(); }
        }
        batch_reads.clear();
        batch_copies.clear();
        pieces.clear();
        zeroes.clear();
        staging_cursor = 0;
        if (request) {
            // Captured decode can finish safely; the caller checks before publishing output.
            cudaMemsetAsync(missing_ids.p, 0xff, std::size_t(request->width) * 4, copies);
            cudaMemsetAsync(missing_experts, 0, 4, copies);
            if (cpu_flag.bytes) {
                cudaMemsetAsync(cpu_flag.p, 0, 4, copies);
                *static_cast<std::int32_t*>(host_flag->data()) = 0;
            }
        }
        cudaStreamSynchronize(copies);
    }

    void run() {
        DeviceBinding bind(rank.device);
        std::uint32_t served = 0;
        auto active_until = std::chrono::steady_clock::now();
        while (!stopping.load(std::memory_order_acquire)) {
            const auto request = handshake->pending(served);
            if (!request) {
                if (std::chrono::steady_clock::now() < active_until) { std::this_thread::yield(); }
                else {
                    // Idle: the CPU workers sleep until the next request.
                    if (spinning) {
                        pool->spin(false);
                        spinning = false;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                continue;
            }
            if (pool && !spinning) {
                pool->spin(true);
                spinning = true;
            }
            bool served_ok = false;
            try { check(); serve(*request); served_ok = true; }
            catch (...) { fail(&*request); }
            handshake->prepared(request->sequence);
            served = request->sequence;
            if (served_ok) {
                try { post(); }
                catch (...) { fail(nullptr); }
            }
            finished.store(served, std::memory_order_release);
            active_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        }
    }

    // After the compute stream is idle: the worker has finished every published request,
    // post work included, and its copies are done.
    void quiesce() {
        DeviceBinding bind(rank.device);
        CUDA_CHECK(cudaStreamSynchronize(rank.stream));
        while (const auto request = handshake->pending(finished.load(std::memory_order_acquire))) {
            (void)request;
            std::this_thread::yield();
        }
        CUDA_CHECK(cudaStreamSynchronize(copies));
        CUDA_CHECK(cudaStreamSynchronize(rounds));
    }

    // An elastic cache, quiesced and locked: gives back at least `bytes` of whole granules, or
    // every slot, taking slots from the pools whose layers hold the most each (trim_pools). Each
    // pool keeps the most routed experts it holds (the tier's decayed counts, then the routes since
    // prime()), moved into its remaining slots when they sat past them; the others leave, and so
    // do a round's candidates whose copies were not issued yet.
    std::uint64_t shrink(std::uint64_t bytes) {
        const std::uint64_t before = cache.growable.mapped_bytes();
        if (cache_slots() == 0 || bytes == 0) { return 0; }
        const std::uint64_t g    = cache.growable.granule();
        const std::uint64_t drop = (std::min(bytes, before) + g - 1) / g * g;
        std::vector<std::uint32_t> counts;
        for (const auto& pool : pools) { counts.push_back(pool.count); }
        const auto keep = trim_pools(before - std::min(before, drop), counts);
        ExpertDeliveryStats delta;
        land();
        for (const auto& d : queued) {
            if (slots[std::size_t(d.slot)].layer >= 0) { evict(d.slot, delta); }
        }
        queued.clear();
        const auto rank_of = [&](std::int32_t s) {
            const auto& slot = slots[std::size_t(s)];
            return std::pair{tier->count(std::size_t(slot.layer), slot.expert),
                             layers[std::size_t(slot.layer)].total[std::size_t(slot.expert)]};
        };
        for (std::size_t p = 0; p < pools.size(); ++p) {
            auto& pool     = pools[p];
            const auto end = pool.first + std::int32_t(keep[p]);
            std::vector<std::int32_t> held;
            for (std::int32_t s = pool.first; s < pool.first + std::int32_t(pool.count); ++s) {
                if (slots[std::size_t(s)].layer >= 0) { held.push_back(s); }
            }
            std::stable_sort(held.begin(), held.end(), [&](std::int32_t a, std::int32_t b) {
                return rank_of(a) > rank_of(b);
            });
            for (std::size_t i = keep[p]; i < held.size(); ++i) { evict(held[i], delta); }
            held.resize(std::min<std::size_t>(held.size(), keep[p]));
            std::vector<std::int32_t> inside;
            for (std::int32_t s = end - 1; s >= pool.first; --s) {
                if (slots[std::size_t(s)].layer < 0) { inside.push_back(s); }
            }
            for (const auto from : held) {
                if (from < end) { continue; }
                const auto to = inside.back();
                inside.pop_back();
                CUDA_CHECK(cudaMemcpyAsync(slot_base(to), slot_base(from), pool.slot_bytes,
                                           cudaMemcpyDeviceToDevice, copies));
                auto& source   = slots[std::size_t(from)];
                auto& target   = slots[std::size_t(to)];
                target.layer   = source.layer;
                target.expert  = source.expert;
                target.pending = false;
                target.written = source.written;
                layers[std::size_t(source.layer)].slot_of[std::size_t(source.expert)] = to;
                point_slot(std::size_t(source.layer), source.expert, to);
                source.layer = source.expert = -1;
            }
            pool.free  = std::move(inside);
            pool.count = keep[p];
        }
        upload_tables();
        CUDA_CHECK(cudaStreamSynchronize(copies));
        cache.bytes = 0;
        for (const auto& pool : pools) {
            const std::uint64_t end = std::uint64_t(pool.count) * pool.slot_bytes;
            cache.growable.unmap(pool.base + end, pool.span - end);
            cache.bytes += backing(pool, pool.count);
        }
        set_free();
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = cache_slots();
        counters = counters + delta;
        return before - cache.growable.mapped_bytes();
    }

    // An elastic cache, quiesced and locked: empty slots in the free device memory beyond
    // keep_free_bytes, up to every expert (or an explicit cache size), as much of it as the device
    // supplies, to the pools whose layers hold the fewest each (fill_pools).
    std::uint64_t grow() {
        const std::uint64_t before = cache.growable.mapped_bytes();
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const std::uint64_t g    = cache.growable.granule();
        const std::uint64_t room = free > options.keep_free_bytes
                                       ? (free - options.keep_free_bytes) / g * g : 0;
        std::uint64_t budget     = before + room;
        if (options.cache_bytes) { budget = std::min(budget, *options.cache_bytes); }
        std::vector<std::uint32_t> counts;
        for (const auto& pool : pools) { counts.push_back(pool.count); }
        const auto target = fill_pools(budget, counts);
        for (std::size_t p = 0; p < pools.size(); ++p) {
            auto& pool     = pools[p];
            const auto old = pool.count;
            auto want      = target[p];
            // The slots that fit in the granules mapped now and in `room` more.
            while (want > old &&
                   !cache.growable.map(pool.base + std::uint64_t(old) * pool.slot_bytes,
                                       std::uint64_t(want - old) * pool.slot_bytes)) {
                want = old + (want - old) / 2;
            }
            if (want <= old) { continue; }
            CUDA_CHECK(cudaMemsetAsync(slot_base(pool.first + std::int32_t(old)), 0,
                                       std::uint64_t(want - old) * pool.slot_bytes, copies));
            for (auto s = pool.first + std::int32_t(old); s < pool.first + std::int32_t(want);
                 ++s) {
                std::fill(slots[std::size_t(s)].written.begin(),
                          slots[std::size_t(s)].written.end(), 0);
                pool.free.push_back(s);
            }
            std::sort(pool.free.begin(), pool.free.end(), std::greater<>());
            pool.count = want;
        }
        CUDA_CHECK(cudaStreamSynchronize(copies));
        cache.bytes = 0;
        for (const auto& pool : pools) { cache.bytes += backing(pool, pool.count); }
        set_free();
        allocation.cache_bytes = cache.backing_bytes();
        allocation.cache_slots = cache_slots();
        return cache.growable.mapped_bytes() - before;
    }
};

ExpertDelivery::ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                               std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options)
    : impl_(std::make_unique<Impl>(rank, std::move(files), std::move(layers), options)) {}
ExpertDelivery::~ExpertDelivery() = default;

ExpertDelivery::Call ExpertDelivery::begin(std::size_t index, const std::int32_t* ids,
                                           std::uint32_t pairs, cudaStream_t stream,
                                           std::span<const Input> inputs) {
    auto& p = *impl_;
    p.check();
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    const auto& layer = p.layers.at(index);
    if (!ids || !pairs || pairs > p.options.pair_capacity || stream != p.rank.stream) {
        throw std::invalid_argument("expert delivery: invalid call or compute stream");
    }
    const auto low = reinterpret_cast<std::uintptr_t>(p.cache.p);
    const int experts = int(layer.source.experts.size());
    Call call{static_cast<const std::int32_t*>(p.cached_ids.p),
              static_cast<const std::int32_t*>(p.missing_ids.p), p.cached_experts,
              p.missing_experts};
    const bool cpu = p.host_flag && layer.cpu && !inputs.empty();
    HostCopies copies;
    if (cpu) {
        // The CPU's operands: the pairs' experts and the family's inputs.
        if (inputs.size() > 2) {
            throw std::invalid_argument("expert delivery: more than two CPU inputs");
        }
        copies.source[0] = reinterpret_cast<const std::byte*>(ids);
        copies.target[0] = static_cast<std::byte*>(p.host_ids_mapped);
        copies.bytes[0]  = pairs * 4;
        std::size_t at = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (input.bytes > p.options.cpu.input_bytes - std::min(at, p.options.cpu.input_bytes)) {
                throw std::invalid_argument("expert delivery: CPU inputs exceed their buffer");
            }
            copies.source[1 + i] = static_cast<const std::byte*>(input.data);
            copies.target[1 + i] = static_cast<std::byte*>(p.host_inputs_mapped) + at;
            copies.bytes[1 + i]  = std::uint32_t(input.bytes);
            at += input.bytes;
        }
        call.products = p.products.p;
        call.cpu = static_cast<const std::int32_t*>(p.cpu_flag.p);
    }
    split_routes<<<1, 1024, std::size_t(experts) * 4, stream>>>(
        ids, pairs, experts, static_cast<const void* const*>(layer.tables.p), low,
        low + p.cache_span, static_cast<const std::uintptr_t*>(p.lent_ranges.p),
        static_cast<std::int32_t*>(p.cached_ids.p), static_cast<std::int32_t*>(p.missing_ids.p),
        p.cached_experts, p.missing_experts, static_cast<std::uint32_t*>(p.needed_mapped), copies,
        p.handshake->device(), std::uint32_t(index), cpu ? kCpuCall : 0);
    CUDA_CHECK(cudaGetLastError());
    return call;
}
void ExpertDelivery::await(cudaStream_t stream) {
    impl_->handshake->wait_prepared(stream);
    impl_->check();
}
const void* const* ExpertDelivery::tables(std::size_t layer) const {
    return static_cast<const void* const*>(impl_->layers.at(layer).tables.p);
}
void ExpertDelivery::check() const { impl_->check(); }
void ExpertDelivery::drain() {
    impl_->quiesce();
    impl_->check();
}
void ExpertDelivery::invalidate() {
    auto& p = *impl_;
    if (p.released) { throw std::logic_error("expert delivery: restore backing before invalidation"); }
    drain();
    for (auto& layer : p.layers) {
        layer.tables.fill();
        std::memset(layer.mirror->data(), 0, layer.mirror->size());
        std::fill(layer.slot_of.begin(), layer.slot_of.end(), -1);
        layer.cached = 0;
        layer.dirty = false;
    }
    for (auto& slot : p.slots) {
        slot.layer = slot.expert = -1;
        slot.pending = false;
    }
    for (auto& pool : p.pools) { Impl::reset_free(pool); }
    for (auto& bank : p.bank_state) {
        bank.layer = -1;
        std::fill(bank.valid.begin(), bank.valid.end(), 0);
    }
    // No bank stays lent: the split reads an empty range for each.
    for (auto& layer : p.layers) {
        std::fill(layer.lent_of.begin(), layer.lent_of.end(), -1);
        std::fill(layer.bank_of.begin(), layer.bank_of.end(), std::int8_t(-1));
    }
    for (auto& slot : p.lent_slots) { slot.layer = slot.expert = -1; }
    std::fill(p.bank_lent.begin(), p.bank_lent.end(), 0);
    std::memset(p.host_ranges->data(), 0, p.host_ranges->size());
    p.lent_ranges.fill();
    p.ranges_dirty = false;
    p.quiet_passes = 0;
    p.set_lent_count();
    for (const auto& d : p.deferred) {
        if (d.landed && std::find(p.spare_events.begin(), p.spare_events.end(), d.landed) ==
                            p.spare_events.end()) {
            p.spare_events.push_back(d.landed);
        }
    }
    p.deferred.clear();
    p.queued.clear();
    for (std::size_t li = 0; li < p.layers.size(); ++li) {
        for (std::size_t e = 0; e < p.layers[li].source.experts.size(); ++e) {
            p.tier->set(li, std::int32_t(e), ExpertTier::State::Out);
        }
    }
    p.set_free();
    p.previous_layer = -1;
    p.last_narrow = -1;
    if (p.host_flag) {
        *static_cast<std::int32_t*>(p.host_flag->data()) = 0;
        p.cpu_flag.fill();
    }
}
void ExpertDelivery::release_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (p.released) { return; }
    drain();
    DeviceBinding bind(p.rank.device);
    p.each_storage([](Storage& storage) { storage.release_backing(); });
    p.released = true;
}
void ExpertDelivery::restore_device_backing() {
    auto& p = *impl_;
    if (!p.options.suspendable) { throw std::logic_error("expert delivery: suspend was not enabled"); }
    if (!p.released) { return; }
    DeviceBinding bind(p.rank.device);
    try {
        p.each_storage([](Storage& storage) { storage.restore_backing(); });
        p.each_storage([](Storage& storage) { storage.fill(); });
        p.handshake->resume();
        for (auto& slot : p.slots) { std::fill(slot.written.begin(), slot.written.end(), 0); }
        p.released = false;
        invalidate();
    } catch (...) {
        p.each_storage([](Storage& storage) { storage.release_backing(); });
        p.released = true;
        throw;
    }
}
std::uint64_t ExpertDelivery::state_backing_bytes() const noexcept {
    const auto memory = this->memory();
    return memory.cache_bytes + memory.bank_bytes + memory.metadata_bytes;
}
ExpertDeliveryStats ExpertDelivery::stats() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
ExpertDeliveryMemory ExpertDelivery::memory() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->allocation;
}

void ExpertDelivery::prime(const std::vector<std::vector<std::uint64_t>>* profile) {
    auto& p = *impl_;
    invalidate();
    p.tier->reset();
    {
        std::lock_guard lock(p.mutex);
        for (auto& layer : p.layers) { std::fill(layer.total.begin(), layer.total.end(), 0); }
    }
    p.set_free();
    if (profile == nullptr) { return; }
    const auto& counts = *profile;
    if (counts.size() != p.layers.size()) {
        throw std::invalid_argument("expert delivery: the profile describes other layers");
    }
    // Most routed first over every layer, each into its pool while it has slots; ties keep layer
    // and expert order.
    std::vector<std::tuple<std::uint64_t, std::size_t, std::int32_t>> ranked;
    for (std::size_t li = 0; li < p.layers.size(); ++li) {
        if (counts[li].size() != p.layers[li].source.experts.size()) {
            throw std::invalid_argument("expert delivery: the profile's bank width differs");
        }
        for (std::size_t e = 0; e < counts[li].size(); ++e) {
            if (counts[li][e] != 0 && p.layers[li].slot_of[e] < 0) {
                ranked.emplace_back(counts[li][e], li, std::int32_t(e));
            }
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    DeviceBinding bind(p.rank.device);
    // The service is idle before the first call; the lock orders these changes before it serves.
    std::lock_guard lock(p.mutex);
    ExpertDeliveryStats delta;
    for (const auto& [count, li, e] : ranked) {
        auto& free = p.pool_of(li).free;
        if (free.empty()) { continue; }
        const auto s = free.back();
        free.pop_back();
        p.occupy(s, li, e, false);
        for (std::size_t b = 0; b < p.layers[li].source.experts[std::size_t(e)].size(); ++b) {
            const auto& blob = p.layers[li].source.experts[std::size_t(e)][b];
            p.add_piece(p.slot_blob(s, b), blob);
            p.slot_written(s, b, blob.bytes);
        }
        p.point_slot(li, e, s);
        ++delta.admissions;
    }
    p.flush_pieces(delta);
    p.upload_tables();
    CUDA_CHECK(cudaStreamSynchronize(p.copies));
    p.set_free();
    p.counters = p.counters + delta;
}

std::vector<std::vector<std::uint64_t>> ExpertDelivery::totals() const {
    std::lock_guard lock(impl_->mutex);
    std::vector<std::vector<std::uint64_t>> out;
    for (const auto& layer : impl_->layers) { out.push_back(layer.total); }
    return out;
}

std::uint64_t ExpertDelivery::shrink_cache(std::uint64_t bytes) {
    auto& p = *impl_;
    if (!p.options.elastic) { throw std::logic_error("expert delivery: the cache is not elastic"); }
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    drain();
    DeviceBinding bind(p.rank.device);
    std::lock_guard lock(p.mutex);
    return p.shrink(bytes);
}

std::uint64_t ExpertDelivery::grow_cache() {
    auto& p = *impl_;
    if (!p.options.elastic) { throw std::logic_error("expert delivery: the cache is not elastic"); }
    if (p.released) { throw std::logic_error("expert delivery: device backing is released"); }
    drain();
    DeviceBinding bind(p.rank.device);
    std::lock_guard lock(p.mutex);
    return p.grow();
}

void ExpertDelivery::set_keep_free_bytes(std::uint64_t bytes) {
    std::lock_guard lock(impl_->mutex);
    impl_->options.keep_free_bytes = bytes;
}

std::string ExpertDelivery::cpu_description() const {
    if (!impl_->pool) { return {}; }
    return std::to_string(impl_->pool->threads()) +
           (impl_->pool->pinned() ? " pinned threads" : " threads");
}

} // namespace ninfer
