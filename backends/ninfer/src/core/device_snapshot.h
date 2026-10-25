#pragma once

#include "core/arena.h"
#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

// Device bytes copied out to host memory and back to the same addresses: what a suspended model
// keeps of its device state while the physical memory behind it is released.
//
// Pageable memory is allocated per capture, sized to what it holds, and freed once the copy is back
// on the device, so a suspend costs host memory only while the model is suspended. Transfers then
// go through a small ring of page-locked staging buffers whose host copies overlap the next DMA.
// Pinned memory is one page-locked block reserved up front: the device copies straight into it, at
// full bus speed, and the block stays allocated for the life of the snapshot.
class DeviceSnapshot {
public:
    enum class Memory : std::uint8_t { Pageable, Pinned };

    // One device range to preserve, on the device of pipeline rank `rank`.
    struct Range {
        std::size_t rank  = 0;
        void* address     = nullptr;
        std::size_t bytes = 0;
    };

    struct Stats {
        std::size_t bytes = 0;
        double seconds    = 0.0;
    };

    // `pinned_capacity` sizes the reserved block in Pinned mode and must be zero otherwise.
    DeviceSnapshot(Memory memory, std::size_t pinned_capacity = 0);
    ~DeviceSnapshot();

    DeviceSnapshot(const DeviceSnapshot&)            = delete;
    DeviceSnapshot& operator=(const DeviceSnapshot&) = delete;
    DeviceSnapshot(DeviceSnapshot&&) noexcept;
    DeviceSnapshot& operator=(DeviceSnapshot&&) noexcept;

    [[nodiscard]] Memory memory() const noexcept;
    // Bytes the snapshot holds, zero when empty.
    [[nodiscard]] std::size_t bytes() const noexcept;
    // Host bytes allocated for it: the pinned reservation, or the pageable copy while one exists.
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    // Copies every range to host memory, replacing what the snapshot held. Ranges are copied on
    // their rank's transfer stream; the call returns once every copy is complete. The caller has
    // drained all work that writes them.
    Stats capture(DeviceContext& device, std::span<const Range> ranges);

    // Copies the captured bytes back to the addresses they came from and waits for the copies. The
    // host copy survives a failure, so a restore can be retried once the device memory is back.
    Stats restore(DeviceContext& device);

    // Frees the host copy (a pinned reservation stays reserved and becomes empty).
    void clear() noexcept;

private:
    struct Impl;

    void transfer(DeviceContext& device, bool to_host);

    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
