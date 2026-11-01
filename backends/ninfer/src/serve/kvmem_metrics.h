#pragma once

#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "product/speculative_options.h"
#include <nlohmann/json.hpp>

namespace ninfer::serve {

// Additive diagnostics; the native usage, timings and Prometheus series stay intact.
inline nlohmann::json kvmem_generation_metrics(const GenerationOutcome& outcome) {
    using Json = nlohmann::json;
    const auto& m = outcome.metrics;
    const auto rounds = m.engine_timing.decode_rounds;
    const auto tokens = outcome.completion_tokens > 0 ? outcome.completion_tokens - 1 : 0;
    Json spec{{"backend", product::speculative_backend_name(m.speculative_backend)},
              {"drafted_tokens", m.speculative_draft_tokens},
              {"accepted_tokens", m.speculative_accepted_tokens},
              {"verification_steps", m.speculative_rounds},
              {"fallback_rounds", m.speculative_fallback_steps},
              {"ngram_drafted_tokens", m.ngram_drafted_tokens},
              {"ngram_accepted_tokens", m.ngram_accepted_tokens},
              {"accepted_per_position", m.speculative_accepted_per_position},
              {"generation_rounds", rounds}, {"output_tokens", tokens},
              {"wall_ms", m.generation_wall_seconds * 1000.0},
              {"timing_scope", "first_to_last_token"},
              {"acceptance_rate", m.speculative_draft_tokens
                   ? Json(double(m.speculative_accepted_tokens) / m.speculative_draft_tokens) : Json(nullptr)},
              {"tokens_per_round", rounds ? Json(double(tokens) / rounds) : Json(nullptr)},
              {"ms_per_round", rounds ? Json(m.generation_wall_seconds * 1000.0 / rounds) : Json(nullptr)}};
    if (!rounds) spec["unavailable_reason"] = "no_decode_rounds_observed";
    if (m.speculative_backend == SpeculativeBackend::Mtp &&
        m.speculative_draft_tokens >= m.ngram_drafted_tokens &&
        m.speculative_accepted_tokens >= m.ngram_accepted_tokens) {
        spec["mtp_drafted_tokens"] = m.speculative_draft_tokens - m.ngram_drafted_tokens;
        spec["mtp_accepted_tokens"] = m.speculative_accepted_tokens - m.ngram_accepted_tokens;
    }
    return {{"speculative", std::move(spec)}};
}

// Native request results, independent of whether the client requested SSE timing
// chunks. TTFT execution is admission-to-first-token; request TTFT also includes
// preparation and queue wait. Keep both boundaries explicit.
inline nlohmann::json kvmem_monitor_outcome(const GenerationOutcome& outcome) {
    using Json = nlohmann::json;
    const auto timings = completion_timings_json(outcome);
    const auto& m = outcome.metrics;
    const bool cancelled = outcome.finish_reason == ninfer::FinishReason::Cancelled;
    const char* finish = cancelled ? "cancelled" : !outcome.tool_calls.empty()
        ? "tool_calls" : openai_finish_reason(outcome.finish_reason);
    auto spec = kvmem_generation_metrics(outcome).at("speculative");
    spec["ngram_archive_drafted_tokens"] = m.ngram_archive_drafted_tokens;
    spec["ngram_archive_accepted_tokens"] = m.ngram_archive_accepted_tokens;
    Json result{{"prompt_tokens", outcome.prompt_tokens},
        {"cached_tokens", std::min<int>(outcome.prompt_tokens, m.prefix_cache_hit_tokens)},
        {"prompt_new_tokens", timings.at("prompt_n")},
        {"generated_tokens", outcome.completion_tokens},
        {"prefill_tps", timings.at("prompt_per_second")},
        {"prefill_ms", timings.at("prompt_ms")},
        {"decode_tps", timings.at("predicted_per_second")},
        {"decode_tps_mean", timings.at("predicted_per_second")},
        {"decode_ms", timings.at("predicted_ms")},
        {"ttft_ms", outcome.completion_tokens > 0 ? timings.at("prompt_ms") : Json(nullptr)},
        {"ttft_request_ms", outcome.completion_tokens > 0 ? Json(m.ttft_seconds * 1000) : Json(nullptr)},
        {"ttft_scope", "execution"}, {"finish_reason", finish},
        {"drafts_offered", m.speculative_draft_tokens},
        {"drafts_accepted", m.speculative_accepted_tokens}, {"speculative", std::move(spec)}};
    // Native total speculation includes copy drafts. Do not label them as MTP.
    if (m.speculative_backend == SpeculativeBackend::Mtp &&
        m.speculative_draft_tokens >= m.ngram_drafted_tokens &&
        m.speculative_accepted_tokens >= m.ngram_accepted_tokens) {
        result["mtp_drafts_offered"] = m.speculative_draft_tokens - m.ngram_drafted_tokens;
        result["mtp_drafts_accepted"] = m.speculative_accepted_tokens - m.ngram_accepted_tokens;
    }
    return result;
}

} // namespace ninfer::serve
