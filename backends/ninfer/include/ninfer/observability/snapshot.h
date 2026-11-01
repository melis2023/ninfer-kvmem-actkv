#pragma once

// Engine snapshots: runtime statistics, context cost, model metadata and residency, slot state
// and the load summary.

#include "ninfer/observability/resources.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {

enum class ModelResidency : std::uint8_t {
    Resident,
    Suspended,
};

// One suspend or resume, as it was carried out.
struct ResidencyTransition {
    double seconds = 0.0;
    // Device state copied to host memory (suspend) or back (resume).
    std::uint64_t state_bytes   = 0;
    double state_seconds        = 0.0;
    // Weight bytes copied to host memory (suspend, Host weights), or uploaded again (resume).
    std::uint64_t weight_bytes  = 0;
    double weight_seconds       = 0.0;
    // Payload read from the artifact by a resume.
    std::uint64_t artifact_read_bytes = 0;
};

struct ResidencyStatus {
    bool enabled          = false;
    ModelResidency state  = ModelResidency::Resident;
    bool auto_resume      = true;
    // Device memory a suspend gives back: the model's fixed-address regions on every device.
    std::uint64_t releasable_device_bytes = 0;
    // Host memory held for the suspended model now: the retained state and any weight copy (the
    // pinned reservation counts while it exists, suspended or not).
    std::uint64_t host_snapshot_bytes = 0;
    std::uint64_t host_reserved_bytes = 0;
    std::uint64_t suspend_count       = 0;
    std::uint64_t resume_count        = 0;
    std::optional<ResidencyTransition> last_suspend;
    std::optional<ResidencyTransition> last_resume;
    // The last failed resume; cleared by the next successful one. The model stays suspended and a
    // later resume retries.
    std::string last_error;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    KvmemMemorySummary kvmem;
    // Worker-published: monitoring must not query CUDA/PDH or take the execution lock.
    CudaResidencySummary cuda_residency;
    RuntimeHostWorkStats host_work;
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Execution time of prefill units and decode rounds (host submission, device wait and host
    // post-processing). Advances per unit, so a scraper sees rates move during a long request.
    double prefill_seconds_total = 0.0;
    double decode_seconds_total  = 0.0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    // Prefill passes that computed the last prompt chunks of several requests at once, and the
    // sum of the requests in them.
    std::uint64_t shared_prefill_passes     = 0;
    std::uint64_t shared_prefill_requests   = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;

    // The first kQueueReportCap waiting requests in submission order, with how long each had
    // waited when the snapshot was published; the Engine republishes at least once a second
    // while requests wait.
    struct QueueEntry {
        std::uint64_t request_id = 0;
        double wait_seconds      = 0.0;
    };

    static constexpr std::size_t kQueueReportCap = 16;
    std::array<QueueEntry, kQueueReportCap> queue{};
    std::uint32_t queue_entries             = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;
    // Materializations aborted because a context-cache store rejected the placement.
    std::uint64_t context_cache_exhausted_requests = 0;
    // A capture the Program declined because the offer was not physically feasible. Unlike an
    // abort this is a silent retention loss, so it needs its own counter to be observable.
    std::uint64_t active_captures_skipped = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t private_endpoint_selections        = 0;
    std::uint64_t private_turn_closure_selections    = 0;
    std::uint64_t private_response_replay_selections = 0;
    std::uint64_t private_long_anchor_selections     = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;
    // Aborted requests whose live state was published as a continuation endpoint.
    std::uint64_t salvaged_continuations = 0;
    // Requests that left the queue without being admitted, and the time they had waited: the
    // client gave up (cancelled) or the pending timeout fired (expired).
    std::uint64_t waiting_cancelled_requests = 0;
    std::uint64_t waiting_expired_requests   = 0;
    double waiting_abandoned_seconds         = 0.0;
    // Requests cancelled while their prompt prefilled, the prompt tokens they had computed, and
    // how many of them the context cache salvaged so a retry resumes where they stopped.
    std::uint64_t cancelled_prefills                = 0;
    std::uint64_t cancelled_prefill_computed_tokens = 0;
    std::uint64_t cancelled_prefills_salvaged       = 0;

    std::uint64_t state_moves     = 0;
    std::uint64_t state_forks     = 0;
    std::uint64_t state_restores  = 0;
    std::uint64_t state_d2h_count = 0;
    std::uint64_t state_h2d_count = 0;
    std::uint64_t state_d2d_count = 0;
    std::uint64_t state_d2h_bytes = 0;
    std::uint64_t state_h2d_bytes = 0;
    std::uint64_t state_d2d_bytes = 0;
    double state_d2h_seconds      = 0.0;
    double state_h2d_seconds      = 0.0;
    double state_d2d_seconds      = 0.0;

    std::uint64_t main_kv_d2h_pages    = 0;
    std::uint64_t main_kv_h2d_pages    = 0;
    std::uint64_t main_kv_d2d_pages    = 0;
    std::uint64_t main_kv_d2h_bytes    = 0;
    std::uint64_t main_kv_h2d_bytes    = 0;
    std::uint64_t main_kv_d2d_bytes    = 0;
    double main_kv_d2h_seconds         = 0.0;
    double main_kv_h2d_seconds         = 0.0;
    double main_kv_d2d_seconds         = 0.0;
    std::uint64_t backend_kv_d2h_pages = 0;
    std::uint64_t backend_kv_h2d_pages = 0;
    std::uint64_t backend_kv_d2d_pages = 0;
    std::uint64_t backend_kv_d2h_bytes = 0;
    std::uint64_t backend_kv_h2d_bytes = 0;
    std::uint64_t backend_kv_d2d_bytes = 0;
    double backend_kv_d2h_seconds      = 0.0;
    double backend_kv_h2d_seconds      = 0.0;
    double backend_kv_d2d_seconds      = 0.0;

    std::uint64_t pressure_spill_pages                 = 0;
    std::uint64_t partial_tail_cow_pages               = 0;
    std::uint32_t device_state_occupied_slots          = 0;
    std::uint32_t host_state_occupied_slots            = 0;
    std::uint32_t device_main_kv_occupied_pages        = 0;
    std::uint32_t device_backend_kv_occupied_pages     = 0;
    // Un-written growth reservation held by active requests, split out of the occupied totals
    // above. Occupancy alone cannot distinguish KV that exists from KV a request is merely still
    // entitled to, which is what made context-cache starvation invisible in the request log.
    std::uint32_t device_main_kv_lease_pages           = 0;
    std::uint32_t device_backend_kv_lease_pages        = 0;
    std::size_t host_kv_occupied_bytes                 = 0;
    std::uint64_t pressure_private_owners_degraded     = 0;
    std::uint64_t pressure_private_owners_demoted      = 0;
    std::uint64_t pressure_private_owners_evicted      = 0;
    std::uint64_t pressure_shared_owners_degraded      = 0;
    std::uint64_t pressure_shared_owners_evicted       = 0;
    std::uint64_t pressure_checkpoints_dropped         = 0;
    std::uint64_t pressure_searches                    = 0;
    std::uint64_t pressure_search_budget_exhaustions   = 0;
    std::uint64_t pressure_maximal_fallback_selections = 0;
    std::uint32_t shared_active_references             = 0;
    std::uint64_t historical_fork_hits                 = 0;
    double actual_context_transfer_seconds             = 0.0;

    // Hybrid prefix cache (ContextCacheMode::Hybrid); zero in Legacy mode. Block and snapshot
    // gauges are absolute; the rest are cumulative event counters.
    std::uint32_t hybrid_cached_blocks           = 0; // Device-resident tree blocks
    std::uint32_t hybrid_evictable_blocks        = 0;
    std::uint32_t hybrid_tree_blocks             = 0; // Device or Host
    std::uint32_t hybrid_snapshots               = 0;
    std::uint64_t hybrid_host_capacity_bytes     = 0;
    std::uint64_t hybrid_host_used_bytes         = 0;
    std::uint64_t hybrid_snapshot_hits           = 0;
    std::uint64_t hybrid_reused_tokens           = 0;
    std::uint64_t hybrid_blocks_inserted         = 0;
    std::uint64_t hybrid_blocks_reattached       = 0;
    std::uint64_t hybrid_blocks_duplicate        = 0;
    std::uint64_t hybrid_taps_created            = 0;
    std::uint64_t hybrid_taps_skipped            = 0;
    std::uint64_t hybrid_endpoints_created       = 0;
    std::uint64_t hybrid_host_image_writes       = 0;
    std::uint64_t hybrid_host_block_writes       = 0;
    std::uint64_t hybrid_host_image_restores     = 0;
    std::uint64_t hybrid_host_block_restores     = 0;
    std::uint64_t hybrid_host_write_bytes        = 0;
    std::uint64_t hybrid_host_restore_bytes      = 0;
    std::uint64_t hybrid_evicted_blocks          = 0;
    std::uint64_t hybrid_host_snapshot_evictions = 0;
    std::uint64_t hybrid_host_dead_reclaims      = 0;
    std::uint64_t hybrid_unbacked_node_losses    = 0;
    // Host-side failures the worker survived by failing the in-flight requests and clearing the
    // context cache instead of latching the Engine unavailable.
    std::uint64_t engine_recoveries = 0;
    NgramTableStats ngram_table;
    ExpertResidencyStats experts;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

// Static facts about the loaded model, independent of the current request context and memory
// layout. The Engine derives the model identity (model_id) and the dimension facts
// (vocab_size, embedding_size, native_context) from the loaded model, and measures the
// parameters, weight bytes, and weights profile from the artifact's tensor inventory. Serving
// renders these into the OpenAI-compatible /v1/models model object and its llama.cpp-compatible
// `meta` field.
struct ModelMetadata {
    std::string model_id;    // Artifact model name (directory metadata "name").
    std::string weights_id;  // Encoded formats of the artifact tensors (meta ftype).
    std::uint64_t vocab_size     = 0; // Tokenizer token domain (meta n_vocab).
    std::uint64_t embedding_size = 0; // Model embedding width (meta n_embd).
    std::uint64_t native_context = 0; // Model native/training context (meta n_ctx_train).
    std::uint64_t parameters     = 0; // Total logical weight elements (meta n_params).
    std::uint64_t weight_bytes   = 0; // Encoded weight payload bytes (meta size).
};

// Session persistence. A slot is one private context-cache catalog cell; a retained session in it
// can be saved to a file and a saved file restored into it. Session digests are FNV-1a 64 over the
// token ledger as 16 lowercase hex characters.
struct SlotCheckpoint {
    std::uint32_t frontier = 0;
    std::string session_digest;
};

struct SlotState {
    // An active request will publish into this cell.
    bool processing = false;
    // The cell holds a retained session.
    bool retained = false;
    // Retained: the session depth. Processing: the request's prompt tokens.
    std::uint32_t prompt_tokens = 0;
    // Retained: the session depth. Processing: the prompt tokens reused from the cache.
    std::uint32_t cached_tokens = 0;
    std::string session_digest;
    // Restorable checkpoints of a retained session, ascending by frontier.
    std::vector<SlotCheckpoint> checkpoints;
};

struct SlotSaveResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    std::string session_digest;
};

struct SlotRestoreResult {
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    std::string session_digest;
};

// A slot operation's expected session digest did not match the slot's resident session.
class SlotSessionMismatch final : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// How the Engine rotates positions past the model's native window, as it resolved the rope
// options for the loaded family (runtime/contract/rope_scaling.h): unscaled RoPE, YaRN at
// `factor` over the `native_context` positions, or linear position interpolation by `factor` past
// the `native_context` threshold.
enum class RopeScalingType : std::uint8_t { None, Yarn, Interpolation };

struct RopeScalingSummary {
    RopeScalingType type         = RopeScalingType::None;
    float factor                 = 1.0F;
    std::uint32_t native_context = 0;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::uint64_t pinned_weight_bytes  = 0; // host-pinned model weights (experts or overlay vision tower)
    std::uint64_t overlay_window_bytes = 0; // device bytes one overlay vision window borrows
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;
    // The rotation past the native window, as every family resolves it from the rope options.
    RopeScalingSummary rope;

    // Hybrid prefix cache restored from its persistent file at startup.
    struct PrefixCacheRestore {
        bool attempted = false;
        bool restored  = false;
        // Why nothing was restored (no file yet, incompatible file, I/O error).
        std::string message;
        std::uint64_t blocks    = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t bytes     = 0;
        double seconds          = 0.0;
        // What the file holds and the Host tier bytes all of it takes, against this Engine's
        // tier. When the tier is smaller, only the snapshots it values most were restored.
        std::uint64_t saved_blocks        = 0;
        std::uint64_t saved_snapshots     = 0;
        std::uint64_t required_host_bytes = 0;
        std::uint64_t host_bytes          = 0;
    } prefix_cache;
};

} // namespace ninfer
