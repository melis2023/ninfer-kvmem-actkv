#pragma once

#include "core/paged_kv_cache.h"
#include "core/host_kv_arena.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Opt-in transfer diagnostics. All event reads follow the caller's existing batch fence.
// Event spans include stream idle time during host submission; they are not copy-engine busy
// time. Across ranks report the longest span per batch, rather than adding concurrent spans.
class KvmemTransferTrace {
    using Clock = std::chrono::steady_clock;
    struct RankTimer {
        int device;
        cudaStream_t stream;
        cudaEvent_t start = nullptr, stop = nullptr;
        RankTimer(int device, cudaStream_t stream) : device(device), stream(stream) {
            const DeviceBinding bind(device);
            CUDA_CHECK(cudaEventCreate(&start));
            const auto status = cudaEventCreate(&stop);
            if (status != cudaSuccess) {
                (void)cudaEventDestroy(start);
                CUDA_CHECK(status);
            }
        }
        ~RankTimer() {
            int previous = device;
            (void)cudaGetDevice(&previous);
            (void)cudaSetDevice(device);
            (void)cudaEventDestroy(stop);
            (void)cudaEventDestroy(start);
            (void)cudaSetDevice(previous);
        }
    };
public:
    KvmemTransferTrace(const DeviceContext& device, RankStreams streams,
                       const char* direction, std::size_t pages, std::size_t staging_bytes,
                       bool asynchronous)
        : enabled_(enabled()), direction_(direction), pages_(pages), staging_bytes_(staging_bytes),
          asynchronous_(asynchronous) {
        if (!enabled_) return;
        started_ = last_ = Clock::now();
        for (std::size_t rank = 0; rank < device.size(); ++rank) {
            timers_.push_back(std::make_unique<RankTimer>(device.rank(rank).device, streams[rank]));
        }
    }
    void add_copy(const DeviceKVPagePool& pool, std::span<const DeviceKVPageHandle> pages,
                   std::span<const std::byte* const> records,
                   const HostKVPageLayout& layout,
                   std::span<const std::uint32_t> groups = {}) {
        if (!enabled_) return;
        const auto stats = pool.host_record_copy_stats(pages, records, groups, layout);
        runs_ += stats.runs;
        copies_ += stats.copies;
        payload_bytes_ += stats.payload_bytes;
    }
    void host_done() { checkpoint(host_us_); }
    void prepare_done() { checkpoint(prepare_us_); }
    void begin_copy() {
        if (!enabled_) return;
        prepare_done();
        for (const auto& timer : timers_) {
            const DeviceBinding bind(timer->device);
            CUDA_CHECK(cudaEventRecord(timer->start, timer->stream));
        }
    }
    void end_copy() {
        if (!enabled_) return;
        for (const auto& timer : timers_) {
            const DeviceBinding bind(timer->device);
            CUDA_CHECK(cudaEventRecord(timer->stop, timer->stream));
        }
        checkpoint(submit_us_);
    }
    void batch_complete() {
        if (!enabled_) return;
        checkpoint(wait_us_);
        float longest = 0;
        for (const auto& timer : timers_) {
            const DeviceBinding bind(timer->device);
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, timer->start, timer->stop));
            longest = std::max(longest, ms);
        }
        gpu_span_ms_ += longest;
        ++batches_;
        prepare_done();
    }
    void finish() {
        if (!enabled_) return;
        prepare_done();
        const auto total = std::chrono::duration_cast<std::chrono::microseconds>(last_ - started_).count();
        std::fprintf(stderr,
            "KVMEM_TRANSFER direction=%s async=%d pages=%zu batches=%llu pool_runs=%llu "
            "copy_calls=%llu payload_bytes=%llu staging_bytes=%zu prepare_us=%lld host_us=%lld "
            "submit_us=%lld wait_us=%lld gpu_span_ms=%.3f total_us=%lld\n",
            direction_, int(asynchronous_), pages_, static_cast<unsigned long long>(batches_),
            static_cast<unsigned long long>(runs_), static_cast<unsigned long long>(copies_),
            static_cast<unsigned long long>(payload_bytes_), staging_bytes_, prepare_us_, host_us_,
            submit_us_, wait_us_, gpu_span_ms_, static_cast<long long>(total));
    }
private:
    static bool enabled() {
        static const bool value = [] {
            const char* flag = std::getenv("KVMEM_TRANSFER_TRACE");
            return flag && std::strcmp(flag, "1") == 0;
        }();
        return value;
    }
    void checkpoint(long long& accumulated) {
        if (!enabled_) return;
        const auto now = Clock::now();
        accumulated += std::chrono::duration_cast<std::chrono::microseconds>(now - last_).count();
        last_ = now;
    }
    bool enabled_;
    const char* direction_;
    std::size_t pages_, staging_bytes_;
    bool asynchronous_;
    Clock::time_point started_{}, last_{};
    std::vector<std::unique_ptr<RankTimer>> timers_;
    std::uint64_t batches_ = 0, runs_ = 0, copies_ = 0, payload_bytes_ = 0;
    long long prepare_us_ = 0, host_us_ = 0, submit_us_ = 0, wait_us_ = 0;
    double gpu_span_ms_ = 0;
};

} // namespace ninfer::models::qwen3_5::detail
