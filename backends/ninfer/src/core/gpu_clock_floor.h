#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ninfer {

// Holds the graphics clock of CUDA devices at or above a floor (NVML locked clocks between the
// floor and the device's maximum) while work runs, and gives them back to the driver once no work
// has run for kRelease. A layer pipeline leaves each stage waiting on its neighbours between short
// kernels, and the driver reads the waits as idle: on four RTX 3060 under Windows a Flash-Next
// decode ran its kernels at 210-480 MHz, and the longer the answer the lower its rate. A held
// clock also stays up while the device has no work (an idle RTX 4090 held at 1500 MHz drew 33 W in
// P5 against 22-28 W in P8), so it is held only around work. NVML takes about 10 ms a device to
// lock or reset, so a thread of this object does it and the work never waits for it.
//
// Changing clocks needs administrator rights (root on Linux); a device whose clocks cannot be held
// is reported and left as it is. Giving the clocks back resets the device's locked clocks, so a
// lock set before this object existed is not restored. A process that is killed rather than
// stopped leaves the floor in place until the driver is told otherwise (`nvidia-smi -rgc`) or the
// machine restarts.
class GpuClockFloor {
public:
    // How long the clocks stay held after the last holder ends, so consecutive requests share one
    // lock.
    static constexpr std::chrono::seconds kRelease{2};

    // Holds nothing and reports nothing.
    GpuClockFloor() noexcept = default;
    // Loads NVML and checks each device once by locking and resetting its clocks; holds nothing
    // until hold(). A zero floor does nothing.
    GpuClockFloor(std::span<const int> devices, std::uint32_t floor_mhz);
    ~GpuClockFloor();

    GpuClockFloor(const GpuClockFloor&)            = delete;
    GpuClockFloor& operator=(const GpuClockFloor&) = delete;

    // One line per device the floor was asked for: the range it holds, or why it is left alone.
    [[nodiscard]] const std::vector<std::string>& report() const noexcept { return report_; }
    // Whether every device asked for can be held.
    [[nodiscard]] bool complete() const noexcept { return complete_; }

    // Holds the floor until the returned token is destroyed; tokens nest. Null when no device can
    // be held. Neither this nor the token's end waits for NVML.
    [[nodiscard]] std::shared_ptr<void> hold();

private:
    struct Nvml;
    struct Device {
        void* handle          = nullptr;
        unsigned int low_mhz  = 0;
        unsigned int high_mhz = 0;
    };

    static std::shared_ptr<Nvml> load_nvml(std::string& problem);
    void govern();
    void leave() noexcept;

    std::shared_ptr<Nvml> nvml_;
    std::vector<Device> devices_;
    std::vector<std::string> report_;
    bool complete_ = false;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t holders_ = 0;
    bool stopping_       = false;
    std::thread governor_;
};

} // namespace ninfer
