#pragma once

// MTP round section timing, enabled by NINFER_MTP_TRACE=1. The trace forces the eager
// decode body (see decode.cpp) and times its GPU sections with stream events: the same
// kernel sequence a captured graph replays, so the section times are the graph's.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] inline bool mtp_trace_requested() {
    static const bool on = [] {
        const char* flag = std::getenv("NINFER_MTP_TRACE");
        return flag && flag[0] == '1';
    }();
    return on;
}

namespace mtp_trace {

// Section boundaries of one decode round: ingress -> verified -> drafted -> proposed -> copied.
inline cudaEvent_t& event(unsigned index) {
    static cudaEvent_t events[4] = {};
    return events[index];
}

// The per-AR-step begin/end pairs; report() prints and frees them.
inline std::vector<cudaEvent_t>& step_events() {
    static std::vector<cudaEvent_t> steps;
    return steps;
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

// One begin/end pair around a single AR step of the draft block.
inline void step_record(unsigned index, cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &status);
    if (status != cudaStreamCaptureStatusNone) { return; }
    cudaEvent_t slot = nullptr;
    cudaEventCreate(&slot);
    cudaEventRecord(slot, stream);
    step_events().push_back(slot);
}

// Same, on a separate list (the propose call inside one AR step).
inline std::vector<cudaEvent_t>& propose_events() {
    static std::vector<cudaEvent_t> events;
    return events;
}

inline void propose_record(unsigned index, cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &status);
    if (status != cudaStreamCaptureStatusNone) { return; }
    cudaEvent_t slot = nullptr;
    cudaEventCreate(&slot);
    cudaEventRecord(slot, stream);
    propose_events().push_back(slot);
}

// Prints the section times of the last recorded round and frees every event.
inline void report(unsigned window) {
    cudaEventSynchronize(event(3));
    float sections[3] = {};
    cudaEventElapsedTime(&sections[0], event(0), event(1));
    cudaEventElapsedTime(&sections[1], event(1), event(2));
    cudaEventElapsedTime(&sections[2], event(2), event(3));
    std::fprintf(stderr,
                 "MTP_GPU_TRACE window=%u target_verify=%.3fms draft_block=%.3fms ar_propose=%.3fms\n",
                 window, sections[0], sections[1], sections[2]);
    const std::vector<cudaEvent_t>& steps = step_events();
    for (std::size_t i = 0; i + 1 < steps.size(); i += 2) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, steps[i], steps[i + 1]) == cudaSuccess) {
            std::fprintf(stderr, "MTP_GPU_TRACE ar_step[%zu]=%.3fms\n", i / 2, ms);
        }
    }
    for (const cudaEvent_t slot : steps) { cudaEventDestroy(slot); }
    step_events().clear();
    const std::vector<cudaEvent_t>& propose = propose_events();
    for (std::size_t i = 0; i + 1 < propose.size(); i += 2) {
        float ms = 0;
        if (cudaEventElapsedTime(&ms, propose[i], propose[i + 1]) == cudaSuccess) {
            std::fprintf(stderr, "MTP_GPU_TRACE propose[%zu]=%.3fms\n", i / 2, ms);
        }
    }
    for (const cudaEvent_t slot : propose) { cudaEventDestroy(slot); }
    propose_events().clear();
}

} // namespace mtp_trace

} // namespace ninfer::models::qwen3_5::execution
