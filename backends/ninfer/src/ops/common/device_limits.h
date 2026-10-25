#pragma once

namespace ninfer::ops {

// The largest dynamic shared memory one block may opt into on the build's architecture: 64 KiB on
// Turing (sm_75), 99 KiB on sm_86, sm_89 and sm_120 (sm_80 allows more, but its kernels are the
// sm_86 ones). Kernel shapes assert against it, so a schedule that cannot launch on the target
// fails to compile instead of at its first launch.
#if defined(NINFER_SM75)
inline constexpr int kMaxSharedBytesPerBlock = 64 * 1024;
#else
inline constexpr int kMaxSharedBytesPerBlock = 99 * 1024;
#endif

// The most threads one SM holds resident on the build's architecture: 1,024 on Turing, 1,536 on
// sm_86, sm_89 and sm_120 (a 70 or 80 build's parts hold 2,048, and their kernels ask for at most
// the sm_86 figure).
#if defined(NINFER_SM75)
inline constexpr int kMaxThreadsPerSm = 1024;
#else
inline constexpr int kMaxThreadsPerSm = 1536;
#endif

// The minimum resident CTAs a __launch_bounds__ of `threads` a CTA asks for: `requested`, or as
// many as an SM of the build's architecture holds where that is fewer. ptxas ignores a minimum it
// cannot meet, which leaves the kernel's registers unbounded instead of sized for full occupancy.
[[nodiscard]] constexpr int resident_ctas_bound(int threads, int requested) {
    return requested * threads <= kMaxThreadsPerSm ? requested : kMaxThreadsPerSm / threads;
}

// The sm_8x bound, for the unified-template Linear schedules (namespace detail::unified) and the
// Q8 schedule templates they share with the legacy table. A Turing build compiles the unified table
// but never selects it (kUnifiedLinearRoutes in ops/linear/common/route_table.h), so those
// schedules keep the 99 KiB they were sized for; the legacy Q8 routes that do run on Turing are
// qualified by their tests, where a schedule past 64 KiB fails its first launch.
inline constexpr int kMaxSharedBytesPerBlockSm8x = 99 * 1024;

// The depth of a cp.async staging ring. Turing copies synchronously (ops/common/memory.cuh), so a
// ring deeper than two stages -- one being computed, one being filled -- only spends shared memory
// there. Every ring that takes its depth from here is written for any depth of at least two.
[[nodiscard]] constexpr int pipeline_stages(int requested) {
#if defined(NINFER_SM75)
    return requested < 2 ? requested : 2;
#else
    return requested;
#endif
}

} // namespace ninfer::ops
