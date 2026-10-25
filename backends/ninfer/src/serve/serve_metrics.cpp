#include "serve/serve_metrics.h"

#include <cmath>
#include <cstdio>

namespace ninfer::serve {
namespace {

void append(std::string& out, const char* name, const char* type, const char* help, double value) {
    char line[640];
    const double finite = std::isfinite(value) ? value : 0.0;
    std::snprintf(line, sizeof(line), "# HELP %s %s\n# TYPE %s %s\n%s %.17g\n", name, help, name,
                  type, name, finite);
    out += line;
}

double ratio(double numerator, double denominator) {
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

} // namespace

void ServeMetrics::record_done(const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    prefix_cache_hit_tokens_total_ += metrics.prefix_cache_hit_tokens;
    draft_tokens_total_ += metrics.speculative_draft_tokens;
    draft_accepted_tokens_total_ += metrics.speculative_accepted_tokens;
}

void ServeMetrics::record_failure() {
    const std::lock_guard lock(mutex_);
    ++requests_failed_total_;
}

void ServeMetrics::record_rejection() {
    const std::lock_guard lock(mutex_);
    ++requests_rejected_total_;
}

std::string ServeMetrics::render(const LoadCapacity& capacity, const LoadSample& sample) const {
    const ninfer::RuntimeStats& stats = sample.stats;
    const double page_tokens =
        capacity.kv_capacity_pages == 0
            ? 0.0
            : static_cast<double>(capacity.kv_capacity_tokens) / capacity.kv_capacity_pages;
    const double kv_tokens = static_cast<double>(stats.device_main_kv_occupied_pages) * page_tokens;
    // Tokens and seconds both come from the Engine's per-unit counters, so they cover the same
    // work (in-flight, failed and cancelled requests included) and their ratio is a real rate.
    const auto prompt_tokens    = static_cast<double>(stats.computed_prefill_tokens);
    const auto predicted_tokens = static_cast<double>(stats.committed_decode_tokens);

    const std::lock_guard lock(mutex_);
    std::string out;
    out.reserve(4096);
    append(out, "llamacpp:prompt_tokens_total", "counter",
           "Prompt tokens evaluated by prefill; reused prefix tokens are excluded.", prompt_tokens);
    append(out, "llamacpp:prompt_seconds_total", "counter", "Prefill execution time in seconds.",
           stats.prefill_seconds_total);
    append(out, "llamacpp:tokens_predicted_total", "counter",
           "Tokens committed by decode rounds.", predicted_tokens);
    append(out, "llamacpp:tokens_predicted_seconds_total", "counter",
           "Decode execution time in seconds.", stats.decode_seconds_total);
    append(out, "llamacpp:n_decode_total", "counter", "Decode rounds executed.",
           static_cast<double>(stats.decode_rounds));
    append(out, "llamacpp:n_busy_slots_per_decode", "gauge",
           "Average number of requests per decode round.",
           ratio(static_cast<double>(stats.decode_row_rounds),
                 static_cast<double>(stats.decode_rounds)));
    append(out, "llamacpp:prompt_tokens_seconds", "gauge",
           "Average prefill throughput in tokens/s.",
           ratio(prompt_tokens, stats.prefill_seconds_total));
    append(out, "llamacpp:predicted_tokens_seconds", "gauge",
           "Average generation throughput in tokens/s.",
           ratio(predicted_tokens, stats.decode_seconds_total));
    append(out, "llamacpp:kv_cache_usage_ratio", "gauge",
           "Occupied share of the device KV cache; 1 means full.",
           ratio(static_cast<double>(stats.device_main_kv_occupied_pages),
                 static_cast<double>(capacity.kv_capacity_pages)));
    append(out, "llamacpp:kv_cache_tokens", "gauge", "Tokens held in the device KV cache.",
           kv_tokens);
    append(out, "llamacpp:requests_processing", "gauge", "Requests holding an execution lane.",
           static_cast<double>(stats.running_requests));
    append(out, "llamacpp:requests_deferred", "gauge", "Requests waiting for admission.",
           static_cast<double>(stats.waiting_requests));
    append(out, "ninfer:requests_total", "counter", "Requests completed with an outcome.",
           static_cast<double>(requests_total_));
    append(out, "ninfer:requests_failed_total", "counter",
           "Accepted requests that ended in an error.",
           static_cast<double>(requests_failed_total_));
    append(out, "ninfer:requests_rejected_total", "counter",
           "Generation requests rejected during preparation, one per request_rejected log event "
           "(overload, invalid or oversized prompt or media). Unparseable and oversized HTTP "
           "bodies are not counted; failures after acceptance, including a queue timeout after "
           "submission, count in ninfer:requests_failed_total.",
           static_cast<double>(requests_rejected_total_));
    append(out, "ninfer:requests_admitted", "gauge",
           "Requests holding server ingress capacity, in any phase.",
           static_cast<double>(sample.admitted_requests));
    append(out, "ninfer:prefix_cache_hit_tokens_total", "counter",
           "Prompt tokens of completed requests served from a cached prefix.",
           static_cast<double>(prefix_cache_hit_tokens_total_));
    append(out, "ninfer:reused_prompt_tokens_total", "counter",
           "Prompt tokens restored from the context cache instead of prefilled.",
           static_cast<double>(stats.reused_prompt_tokens));
    append(out, "ninfer:draft_tokens_total", "counter",
           "Speculative draft tokens proposed for completed requests.",
           static_cast<double>(draft_tokens_total_));
    append(out, "ninfer:draft_accepted_tokens_total", "counter",
           "Speculative draft tokens accepted for completed requests.",
           static_cast<double>(draft_accepted_tokens_total_));
    append(out, "ninfer:context_cache_exhausted_requests_total", "counter",
           "Requests failed because the context cache had no placement for them.",
           static_cast<double>(stats.context_cache_exhausted_requests));
    append(out, "ninfer:engine_recoveries_total", "counter",
           "Host-side worker failures the Engine survived instead of latching unavailable.",
           static_cast<double>(stats.engine_recoveries));
    append(out, "ninfer:uptime_seconds", "gauge", "Seconds since the Engine became ready.",
           sample.uptime_seconds);
    return out;
}

} // namespace ninfer::serve
