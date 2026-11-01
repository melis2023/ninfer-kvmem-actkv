#pragma once

// Bookkeeping around the Engine worker's failure recovery. Header-only and free of CUDA, so its
// policy is unit-tested without a device.

#include <chrono>
#include <cstdint>
#include <exception>
#include <string>

namespace ninfer::runtime {

// Counts worker failures that came in a row and decides when one more must fail every request.
// Up to `maximum` failures in a row are recovered from; the next one is not. The streak ends in two
// ways:
//   * a work unit (control batch, prefill step or decode round) completes (`record_success`);
//   * `healthy_window` passes after a failure with no further failure. Each recovery is verified
//     (device synchronized, physical baseline empty, grafts reinstalled), so failures that far
//     apart are not the tight loop the cap exists to stop. Without the window, a client that keeps
//     resending one request whose admission runs out of memory, with no work unit in between,
//     would leave the next recoverable failure to fail every request.
class RecoveryStreak {
public:
    using Clock = std::chrono::steady_clock;

    RecoveryStreak(std::uint32_t maximum, Clock::duration healthy_window) noexcept
        : maximum_(maximum), healthy_window_(healthy_window) {}

    // Counts the failure at `now`; false when it exceeds the streak `maximum` allows.
    [[nodiscard]] bool record_failure(Clock::time_point now) noexcept {
        if (count_ != 0 && now - last_failure_ >= healthy_window_) { count_ = 0; }
        last_failure_ = now;
        ++count_;
        return count_ <= maximum_;
    }

    void record_success() noexcept { count_ = 0; }

    // Failures in the current streak, the last one counted included.
    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }

private:
    std::uint32_t maximum_;
    Clock::duration healthy_window_;
    std::uint32_t count_ = 0;
    Clock::time_point last_failure_{};
};

// The text of an exception for the operator log. Never throws.
[[nodiscard]] inline std::string exception_text(std::exception_ptr error) noexcept {
    try {
        if (error) { std::rethrow_exception(error); }
    } catch (const std::exception& exception) {
        try {
            return exception.what();
        } catch (...) {}
    } catch (...) {}
    return "unknown exception";
}

} // namespace ninfer::runtime
