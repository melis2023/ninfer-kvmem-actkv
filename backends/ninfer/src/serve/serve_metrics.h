#pragma once

// GET /metrics in the Prometheus text format. The llamacpp:-prefixed series keep llama.cpp's
// --metrics names and meanings, so dashboards and autoscalers built against it read this server
// unchanged: computed prefill tokens (prefix-cache hits excluded) against prefill execution time,
// committed decode tokens against decode execution time. Both come from the Engine's live
// RuntimeStats, so rates advance during a long request instead of jumping at its completion. The
// ninfer:-prefixed series report what llama.cpp has no name for.

#include "serve/generation_service.h"
#include "serve/load_report.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

class ServeMetrics {
public:
    // Accumulates one completed request. Called from the request-done funnel, so every protocol
    // and both streaming modes count exactly once.
    void record_done(const GenerationOutcome& outcome);
    // Counts one accepted request that ended in an error instead of an outcome.
    void record_failure();
    // Counts one request refused before it reached the Engine (one per request_rejected event):
    // overload, an invalid or oversized prompt or media.
    void record_rejection();

    [[nodiscard]] std::string render(const LoadCapacity& capacity, const LoadSample& sample) const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                = 0;
    std::uint64_t requests_failed_total_         = 0;
    std::uint64_t requests_rejected_total_       = 0;
    std::uint64_t prefix_cache_hit_tokens_total_ = 0;
    std::uint64_t draft_tokens_total_            = 0;
    std::uint64_t draft_accepted_tokens_total_   = 0;
};

} // namespace ninfer::serve
