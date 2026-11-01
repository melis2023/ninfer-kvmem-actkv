#include "serve/serve_metrics.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

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

// A labelled family: one HELP/TYPE header, then one sample per label set.
void append_family(std::string& out, const char* name, const char* type, const char* help) {
    out.append("# HELP ").append(name).append(" ").append(help).append("\n");
    out.append("# TYPE ").append(name).append(" ").append(type).append("\n");
}

void append_sample(std::string& out, const char* name, const std::string& labels,
                   std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%llu", static_cast<unsigned long long>(value));
    out.append(name).append("{").append(labels).append("} ").append(text).append("\n");
}

// The context cache's Engine counters. Selections count admissions by the source they started
// from: `root` is a miss (a prefill from token zero), every other source a reuse, so the hit rate
// is 1 - root / all. Pressure events are what planning did to inactive owners to make room.
void append_context_cache(std::string& out, const ninfer::RuntimeStats& stats) {
    const char* selections = "ninfer:context_selections_total";
    append_family(out, selections, "counter",
                  "Admissions by the context-cache source they started from; root is a miss.");
    append_sample(out, selections, "source=\"root\"", stats.root_selections);
    append_sample(out, selections, "source=\"private_endpoint\"",
                  stats.private_endpoint_selections);
    append_sample(out, selections, "source=\"private_turn_closure\"",
                  stats.private_turn_closure_selections);
    append_sample(out, selections, "source=\"private_response_replay\"",
                  stats.private_response_replay_selections);
    append_sample(out, selections, "source=\"private_long_anchor\"",
                  stats.private_long_anchor_selections);
    append_sample(out, selections, "source=\"shared_stable_prefix\"",
                  stats.shared_stable_prefix_selections);

    const char* pressure = "ninfer:context_pressure_events_total";
    append_family(out, pressure, "counter",
                  "What pressure planning did to inactive context-cache owners.");
    append_sample(out, pressure, "event=\"private_owner_evicted\"",
                  stats.pressure_private_owners_evicted);
    append_sample(out, pressure, "event=\"private_owner_degraded\"",
                  stats.pressure_private_owners_degraded);
    append_sample(out, pressure, "event=\"shared_owner_evicted\"",
                  stats.pressure_shared_owners_evicted);
    append_sample(out, pressure, "event=\"shared_owner_degraded\"",
                  stats.pressure_shared_owners_degraded);
    append_sample(out, pressure, "event=\"checkpoint_dropped\"",
                  stats.pressure_checkpoints_dropped);

    const char* searches = "ninfer:context_pressure_searches_total";
    append_family(out, searches, "counter", "Pressure planning searches by how they ended.");
    append_sample(out, searches, "result=\"started\"", stats.pressure_searches);
    append_sample(out, searches, "result=\"budget_exhausted\"",
                  stats.pressure_search_budget_exhaustions);
    append_sample(out, searches, "result=\"maximal_fallback\"",
                  stats.pressure_maximal_fallback_selections);

    const char* transfers = "ninfer:context_transfer_bytes_total";
    append_family(out, transfers, "counter",
                  "Context-cache bytes moved between Device and Host, by object and direction.");
    const auto transfer = [&](const char* object, const char* direction, std::uint64_t bytes) {
        append_sample(out, transfers,
                      std::string("object=\"") + object + "\",direction=\"" + direction + "\"",
                      bytes);
    };
    transfer("state", "d2h", stats.state_d2h_bytes);
    transfer("state", "h2d", stats.state_h2d_bytes);
    transfer("main_kv", "d2h", stats.main_kv_d2h_bytes);
    transfer("main_kv", "h2d", stats.main_kv_h2d_bytes);
    transfer("backend_kv", "d2h", stats.backend_kv_d2h_bytes);
    transfer("backend_kv", "h2d", stats.backend_kv_h2d_bytes);
    append(out, "ninfer:context_transfer_seconds_total", "counter",
           "Time admissions waited for context-cache transfers.",
           stats.actual_context_transfer_seconds);
    append(out, "ninfer:context_historical_fork_hits_total", "counter",
           "Admissions that forked a historical checkpoint instead of the latest endpoint.",
           static_cast<double>(stats.historical_fork_hits));

    const char* occupancy = "ninfer:context_occupancy";
    append_family(out, occupancy, "gauge",
                  "Context-cache occupancy by pool: state slots and KV pages on the Device, state "
                  "slots and KV bytes on the Host.");
    append_sample(out, occupancy, "pool=\"device_state_slots\"",
                  stats.device_state_occupied_slots);
    append_sample(out, occupancy, "pool=\"host_state_slots\"", stats.host_state_occupied_slots);
    append_sample(out, occupancy, "pool=\"device_main_kv_pages\"",
                  stats.device_main_kv_occupied_pages);
    append_sample(out, occupancy, "pool=\"device_backend_kv_pages\"",
                  stats.device_backend_kv_occupied_pages);
    append_sample(out, occupancy, "pool=\"host_kv_bytes\"",
                  static_cast<std::uint64_t>(stats.host_kv_occupied_bytes));
}

} // namespace

void ServeMetrics::record_done(const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    prefix_cache_hit_tokens_total_ += metrics.prefix_cache_hit_tokens;
    draft_tokens_total_ += metrics.speculative_draft_tokens;
    draft_accepted_tokens_total_ += metrics.speculative_accepted_tokens;
    if (metrics.speculative_backend != SpeculativeBackend::None) {
        kvmem::SpeculativeStatistics stats;
        stats.requests = 1; stats.rounds = metrics.speculative_rounds;
        stats.drafted_tokens = metrics.speculative_draft_tokens;
        stats.accepted_tokens = metrics.speculative_accepted_tokens;
        stats.generated_tokens = outcome.tokens.size();
        stats.fallback_steps = metrics.speculative_fallback_steps;
        speculative_statistics_.add(stats);
    }
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
    append(out, "ninfer:waiting_cancelled_requests_total", "counter",
           "Requests the client cancelled while they waited for admission.",
           static_cast<double>(stats.waiting_cancelled_requests));
    append(out, "ninfer:waiting_expired_requests_total", "counter",
           "Requests that reached the pending timeout before admission.",
           static_cast<double>(stats.waiting_expired_requests));
    append(out, "ninfer:waiting_abandoned_seconds_total", "counter",
           "Time cancelled and expired requests had waited before they left the queue.",
           stats.waiting_abandoned_seconds);
    append(out, "ninfer:cancelled_prefills_total", "counter",
           "Requests cancelled while their prompt prefilled.",
           static_cast<double>(stats.cancelled_prefills));
    append(out, "ninfer:cancelled_prefill_computed_tokens_total", "counter",
           "Prompt tokens those requests had computed when they were cancelled.",
           static_cast<double>(stats.cancelled_prefill_computed_tokens));
    append(out, "ninfer:cancelled_prefills_salvaged_total", "counter",
           "Cancelled prefills the context cache kept, so a retry resumes where they stopped.",
           static_cast<double>(stats.cancelled_prefills_salvaged));
    append_context_cache(out, stats);
    return out;
}

} // namespace ninfer::serve
