#include "core/device_snapshot.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/mman.h>
#endif

namespace ninfer {
namespace {

using Clock = std::chrono::steady_clock;

// Four 64 MiB slots keep a DMA in flight while the host copies the previous one; larger slots buy
// nothing once the bus, not the copy, is the limit.
constexpr std::size_t kSlotBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kSlots     = 4;
// A host copy between a staging slot and ordinary memory runs on this many threads: one thread
// copies at well under bus speed, and the first touch of a fresh pageable page faults it in.
constexpr std::size_t kCopyThreads = 4;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void parallel_copy(std::byte* destination, const std::byte* source, std::size_t bytes) {
    const std::size_t threads = bytes >= (8ULL << 20) ? kCopyThreads : 1;
    if (threads == 1) {
        std::memcpy(destination, source, bytes);
        return;
    }
    const std::size_t share = (bytes + threads - 1) / threads;
    std::vector<std::future<void>> copies;
    for (std::size_t index = 1; index < threads; ++index) {
        const std::size_t begin = index * share;
        if (begin >= bytes) { break; }
        const std::size_t count = std::min(share, bytes - begin);
        copies.push_back(std::async(std::launch::async, [=] {
            std::memcpy(destination + begin, source + begin, count);
        }));
    }
    std::memcpy(destination, source, std::min(share, bytes));
    for (auto& copy : copies) { copy.get(); }
}

// Ordinary host memory for a pageable snapshot. On Linux it asks for transparent huge pages, so
// the first touch faults in 2 MiB at a time rather than 4 KiB.
class HostBlock {
public:
    HostBlock() noexcept = default;
    explicit HostBlock(std::size_t bytes) : bytes_(bytes) {
#ifdef _WIN32
        data_ = static_cast<std::byte*>(
            VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (data_ == nullptr) { throw std::bad_alloc(); }
#else
        void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                             -1, 0);
        if (mapping == MAP_FAILED) { throw std::bad_alloc(); }
#    ifdef MADV_HUGEPAGE
        (void)madvise(mapping, bytes, MADV_HUGEPAGE);
#    endif
        data_ = static_cast<std::byte*>(mapping);
#endif
    }
    ~HostBlock() { reset(); }
    HostBlock(const HostBlock&)            = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock(HostBlock&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    HostBlock& operator=(HostBlock&& other) noexcept {
        if (this != &other) {
            reset();
            data_  = std::exchange(other.data_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    void reset() noexcept {
        if (data_ == nullptr) { return; }
#ifdef _WIN32
        (void)VirtualFree(data_, 0, MEM_RELEASE);
#else
        (void)munmap(data_, bytes_);
#endif
        data_  = nullptr;
        bytes_ = 0;
    }

private:
    std::byte* data_   = nullptr;
    std::size_t bytes_ = 0;
};

// One page-locked staging buffer and the copies packed into it: a slot carries any number of
// ranges, so a snapshot made of many small ranges pays one event per 64 MiB, not one per range.
struct StagingSlot {
    struct Piece {
        std::byte* host     = nullptr;
        std::size_t offset  = 0;
        std::size_t bytes   = 0;
    };

    PinnedHostBuffer buffer{kSlotBytes};
    cudaEvent_t event = nullptr;
    std::vector<Piece> pieces; // captures waiting to land in host memory once the DMA completes
    std::size_t used = 0;
    bool busy        = false;

    StagingSlot() { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    ~StagingSlot() {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    StagingSlot(const StagingSlot&)            = delete;
    StagingSlot& operator=(const StagingSlot&) = delete;

    [[nodiscard]] std::byte* at(std::size_t offset) const noexcept {
        return static_cast<std::byte*>(buffer.data()) + offset;
    }

    // Waits for the slot's DMA and lands its pending captures in host memory; the slot is empty
    // afterwards.
    void settle() {
        if (busy) {
            CUDA_CHECK(cudaEventSynchronize(event));
            for (const Piece& piece : pieces) {
                parallel_copy(piece.host, at(piece.offset), piece.bytes);
            }
        }
        pieces.clear();
        used = 0;
        busy = false;
    }
};

} // namespace

struct DeviceSnapshot::Impl {
    Memory memory;
    // Pageable mode: each rank's staging ring, made by the capture and kept until the snapshot is
    // cleared, so the restore does not pin its slots again (pinning costs about as much as moving
    // a few hundred MiB).
    std::vector<std::vector<std::unique_ptr<StagingSlot>>> staging;
    std::unique_ptr<PinnedHostBuffer> pinned;
    HostBlock pageable;
    std::vector<Range> ranges;
    std::size_t bytes = 0;

    std::byte* host_base() const noexcept {
        if (memory == Memory::Pinned) { return static_cast<std::byte*>(pinned->data()); }
        return pageable.data();
    }
};

DeviceSnapshot::DeviceSnapshot(Memory memory, std::size_t pinned_capacity)
    : impl_(std::make_unique<Impl>()) {
    impl_->memory = memory;
    if (memory == Memory::Pinned) {
        if (pinned_capacity == 0) {
            throw std::invalid_argument("a pinned device snapshot needs a reserved capacity");
        }
        impl_->pinned = std::make_unique<PinnedHostBuffer>(pinned_capacity);
    } else if (pinned_capacity != 0) {
        throw std::invalid_argument("a pageable device snapshot reserves no pinned memory");
    }
}

DeviceSnapshot::~DeviceSnapshot()                                    = default;
DeviceSnapshot::DeviceSnapshot(DeviceSnapshot&&) noexcept            = default;
DeviceSnapshot& DeviceSnapshot::operator=(DeviceSnapshot&&) noexcept = default;

DeviceSnapshot::Memory DeviceSnapshot::memory() const noexcept { return impl_->memory; }

std::size_t DeviceSnapshot::bytes() const noexcept { return impl_->bytes; }

std::size_t DeviceSnapshot::host_capacity_bytes() const noexcept {
    return impl_->memory == Memory::Pinned ? impl_->pinned->size() : impl_->pageable.size();
}

bool DeviceSnapshot::empty() const noexcept { return impl_->ranges.empty(); }

void DeviceSnapshot::clear() noexcept {
    impl_->staging.clear();
    impl_->ranges.clear();
    impl_->bytes = 0;
    impl_->pageable.reset();
}

DeviceSnapshot::Stats DeviceSnapshot::capture(DeviceContext& device,
                                              std::span<const Range> ranges) {
    const auto start  = Clock::now();
    Impl& impl        = *impl_;
    std::size_t total = 0;
    for (const Range& range : ranges) {
        if (range.rank >= device.size()) {
            throw std::invalid_argument("device snapshot range names an unknown rank");
        }
        total += range.bytes;
    }
    clear();
    if (impl.memory == Memory::Pinned) {
        if (total > impl.pinned->size()) {
            throw std::runtime_error("device snapshot of " + std::to_string(total) +
                                     " bytes exceeds its pinned reservation of " +
                                     std::to_string(impl.pinned->size()));
        }
    } else if (total != 0) {
        impl.pageable = HostBlock(total);
    }

    impl.ranges.assign(ranges.begin(), ranges.end());
    impl.bytes = total;
    try {
        transfer(device, /*to_host=*/true);
    } catch (...) {
        clear();
        throw;
    }
    return Stats{.bytes = total, .seconds = seconds_since(start)};
}

void DeviceSnapshot::transfer(DeviceContext& device, bool to_host) {
    Impl& impl = *impl_;
    // Host offsets follow the range order; the copies themselves are grouped by rank, because a
    // staging slot's completion event must belong to the device whose stream records it.
    std::vector<std::size_t> offsets(impl.ranges.size());
    std::size_t running = 0;
    for (std::size_t index = 0; index < impl.ranges.size(); ++index) {
        offsets[index] = running;
        running += impl.ranges[index].bytes;
    }
    std::byte* const host = impl.host_base();
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        const RankBinding bound(device, rank);
        const cudaStream_t stream = device.transfer_stream_for_rank(rank);
        if (impl.staging.size() < device.size()) { impl.staging.resize(device.size()); }
        std::vector<std::unique_ptr<StagingSlot>>& slots = impl.staging[rank];
        std::size_t current = 0;
        for (std::size_t index = 0; index < impl.ranges.size(); ++index) {
            const Range& range = impl.ranges[index];
            if (range.rank != rank || range.bytes == 0) { continue; }
            auto* const device_bytes = static_cast<std::byte*>(range.address);
            std::byte* const host_bytes = host + offsets[index];
            if (impl.memory == Memory::Pinned) {
                CUDA_CHECK(cudaMemcpyAsync(to_host ? static_cast<void*>(host_bytes) : device_bytes,
                                           to_host ? static_cast<const void*>(device_bytes)
                                                   : host_bytes,
                                           range.bytes,
                                           to_host ? cudaMemcpyDeviceToHost
                                                   : cudaMemcpyHostToDevice,
                                           stream));
                continue;
            }
            if (slots.empty()) {
                for (std::size_t slot = 0; slot < kSlots; ++slot) {
                    slots.push_back(std::make_unique<StagingSlot>());
                }
                slots[0]->settle();
            }
            for (std::size_t done = 0; done < range.bytes;) {
                StagingSlot* slot = slots[current].get();
                if (slot->used == kSlotBytes) {
                    // Full: its copies go out under one event and the next slot takes over.
                    CUDA_CHECK(cudaEventRecord(slot->event, stream));
                    slot->busy = true;
                    current    = (current + 1) % kSlots;
                    slot       = slots[current].get();
                    slot->settle();
                }
                const std::size_t chunk = std::min(kSlotBytes - slot->used, range.bytes - done);
                if (to_host) {
                    CUDA_CHECK(cudaMemcpyAsync(slot->at(slot->used), device_bytes + done, chunk,
                                               cudaMemcpyDeviceToHost, stream));
                    slot->pieces.push_back({host_bytes + done, slot->used, chunk});
                } else {
                    parallel_copy(slot->at(slot->used), host_bytes + done, chunk);
                    CUDA_CHECK(cudaMemcpyAsync(device_bytes + done, slot->at(slot->used), chunk,
                                               cudaMemcpyHostToDevice, stream));
                }
                slot->used += chunk;
                done += chunk;
            }
        }
        if (!slots.empty() && slots[current]->used != 0) {
            CUDA_CHECK(cudaEventRecord(slots[current]->event, stream));
            slots[current]->busy = true;
        }
        for (const auto& slot : slots) { slot->settle(); }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}


DeviceSnapshot::Stats DeviceSnapshot::restore(DeviceContext& device) {
    const auto start = Clock::now();
    if (impl_->ranges.empty()) { return Stats{}; }
    transfer(device, /*to_host=*/false);
    return Stats{.bytes = impl_->bytes, .seconds = seconds_since(start)};
}

} // namespace ninfer
