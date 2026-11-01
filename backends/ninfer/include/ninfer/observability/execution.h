#pragma once

// Per-request execution statistics: timings, speculative and thinking-budget counters, prefix reuse
// and context materialization diagnostics.

#include "ninfer/model_enums.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer {

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    // Overlay vision residency: per-request sums over the image windows.
    std::uint32_t overlay_windows      = 0;
    double overlay_window_seconds      = 0.0;
    double overlay_evict_seconds       = 0.0;
    double overlay_restore_seconds     = 0.0;
    std::size_t overlay_evicted_bytes  = 0;
    std::size_t overlay_staged_bytes   = 0;
    // Windows that had to borrow the text weights, which stalls every other lane.
    std::uint32_t overlay_exclusive_windows = 0;
    double decode_seconds      = 0.0;
    // Request wall phases at committed model-state boundaries. Prompt begins when admission and
    // its exact reuse choice are published and ends at the first accepted output token, or
    // at termination when prefill is cancelled before output. Generation
    // spans the first through last accepted output token and therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // Adaptive MTP: rounds verified at each width 1..draft_window, and width changes.
    bool adaptive                    = false;
    std::uint64_t window_transitions = 0;
    std::vector<std::uint64_t> rounds_per_window;
    std::uint64_t ngram_rounds                  = 0;
    std::uint64_t ngram_drafted_tokens          = 0;
    std::uint64_t ngram_accepted_tokens         = 0;
    std::uint64_t ngram_archive_rounds          = 0;
    std::uint64_t ngram_archive_drafted_tokens  = 0;
    std::uint64_t ngram_archive_accepted_tokens = 0;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> requested_budget;
    std::optional<std::uint32_t> effective_budget;
    // Model-origin tokens accepted while thinking remained open, with or without a budget.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
    // The request switched to its post-thinking sampling when its reasoning block closed.
    bool post_thinking_sampling = false;
    // The reasoning-loop guard fired: after this many model-origin thinking tokens, at this
    // repeated-passage coverage (ThinkingControlOptions::loop decided the action; `applied` tells
    // whether it concluded).
    bool loop_detected                 = false;
    std::uint32_t loop_thinking_tokens = 0;
    float loop_coverage                = 0.0F;
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why bounded pressure planning stopped for the materialization decision committed to one request.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    TimeBudget,
    InsufficientExpectedGain,
    WorkBudget,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::TimeBudget:
        return "time_budget";
    case MaterializationStopReason::InsufficientExpectedGain:
        return "insufficient_expected_gain";
    case MaterializationStopReason::WorkBudget:
        return "work_budget";
    }
    return "no_pressure";
}

enum class MaterializationSearchPhase : std::uint8_t {
    None,
    Setup,
    Construction,
    Assessment,
    Expansion,
    Refinement,
};

[[nodiscard]] inline constexpr const char*
materialization_search_phase_name(MaterializationSearchPhase phase) noexcept {
    switch (phase) {
    case MaterializationSearchPhase::None:
        return "none";
    case MaterializationSearchPhase::Setup:
        return "setup";
    case MaterializationSearchPhase::Construction:
        return "construction";
    case MaterializationSearchPhase::Assessment:
        return "assessment";
    case MaterializationSearchPhase::Expansion:
        return "expansion";
    case MaterializationSearchPhase::Refinement:
        return "refinement";
    }
    return "none";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns           = 0;
    std::uint64_t predicted_future_loss_ns   = 0;
    std::uint64_t predicted_total_ns         = 0;
    std::uint32_t targets_evaluated          = 0;
    std::uint64_t projection_work            = 0;
    std::uint64_t planning_elapsed_ns        = 0;
    std::uint64_t search_elapsed_ns          = 0;
    MaterializationStopReason stop_reason    = MaterializationStopReason::NoPressure;
    bool budget_exhausted                    = false;
    std::uint32_t selected_degradation_units = 0;
    bool selected_maximal_fallback           = false;

    std::uint64_t initial_predicted_total_ns = 0;
    std::optional<std::uint64_t> first_improvement_ns;
    std::uint32_t incumbent_improvements         = 0;
    std::uint64_t search_work                    = 0;
    std::uint64_t search_granted_ns              = 0;
    std::uint32_t search_renewals                = 0;
    bool search_discovery_used                   = false;
    std::uint64_t search_overshoot_ns            = 0;
    MaterializationSearchPhase search_stop_phase = MaterializationSearchPhase::None;
    bool search_boundary_limited                 = false;
    // The most prompt reuse any admission candidate offered, independent of the plan that won.
    // Beside a root plan, 0 points at prefix matching (nothing was on the table); a large value
    // points at the planner's pricing.
    std::uint32_t best_reuse_prompt_tokens = 0;

    // Hybrid prefix cache admission. `cached_prefix_tokens` is the longest prompt prefix held as
    // cached KV blocks whether or not it was reusable: reuse also needs a state snapshot inside
    // it, so a gap to the reused token count is prefix lost to snapshot placement.
    // `restored_host_bytes` is what the admission copied back from the Host tier; the copies
    // overlap the request's first prefill pass, so their time is part of its prefill.
    std::uint32_t cached_prefix_tokens = 0;
    std::uint64_t restored_host_bytes  = 0;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct NgramArchiveStats {
    bool enabled              = false;
    bool bound                = false;
    bool published            = false;
    std::uint64_t generation  = 0;
    std::size_t sources       = 0;
    std::size_t session_bytes = 0;
    std::size_t total_bytes   = 0;
    // Effective seed after request-domain separation; archive state is also needed for replay.
    std::optional<std::uint64_t> sampling_seed;
};

} // namespace ninfer
