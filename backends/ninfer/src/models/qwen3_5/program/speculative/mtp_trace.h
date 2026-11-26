#pragma once

// MTP round section timing, enabled by NINFER_MTP_TRACE=1. The trace forces the eager
// decode body (see decode.cpp) and times its GPU sections with stream events: the same
// kernel sequence a captured graph replays, so the section times are the graph's.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] inline bool mtp_trace_requested() {
    static const bool on = [] {
        const char* flag = std::getenv("NINFER_MTP_TRACE");
        return flag && flag[0] == '1';
    }();
    return on;
}

namespace mtp_trace {

inline cudaEvent_t& event(unsigned index) {
    static cudaEvent_t events[4] = {};
    return events[index];
}

inline void record(unsigned index, cudaStream_t stream) {
    // Graph capture replays this body at startup: an event on a capturing stream is
    // illegal, and the capture's section times would say nothing about steady state anyway.
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &status);
    if (status != cudaStreamCaptureStatusNone) { return; }
    cudaEvent_t& slot = event(index);
    if (slot == nullptr) { cudaEventCreate(&slot); }
    cudaEventRecord(slot, stream);
}

// Prints the section times of the last recorded round and re-arms every event.
inline void report(unsigned window) {
    cudaEventSynchronize(event(3));
    float sections[3] = {};
    cudaEventElapsedTime(&sections[0], event(0), event(1));
    cudaEventElapsedTime(&sections[1], event(1), event(2));
    cudaEventElapsedTime(&sections[2], event(2), event(3));
    std::fprintf(stderr,
                 "MTP_GPU_TRACE window=%u target_verify=%.3fms draft_block=%.3fms ar_propose=%.3fms\n",
                 window, sections[0], sections[1], sections[2]);
}

} // namespace mtp_trace

} // namespace ninfer::models::qwen3_5::execution
