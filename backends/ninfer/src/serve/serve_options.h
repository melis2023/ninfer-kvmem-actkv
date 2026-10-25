#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    // Set only when a valid requested copy width was reduced for concurrent execution.
    std::optional<std::uint32_t> requested_ngram_draft_tokens;
    bool help_requested = false;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    // --stats-port: also serve /health, /stats, /v1/load and /metrics on this port with one
    // worker of their own (0 = off).
    int stats_port = 0;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    // --request-log-max-mib: rotate the request log at this size (0 keeps one unbounded file),
    // keeping --request-log-keep rotated copies.
    std::uint32_t request_log_max_mib  = 0;
    std::uint32_t request_log_keep     = 4;
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    KvmemOptions kvmem;
    std::optional<std::size_t> kv_headroom_mib;
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    // Admission waits behind the active set, so the deadline has to cover the generation
    // time of the requests ahead in the FIFO. One 1K-token response already runs past 15 s
    // at C1 on an RTX 3090, and a 6.5K-token one past 100 s; a 30 s deadline expired those
    // callers before they were ever admitted.
    std::uint32_t pending_timeout_ms   = 600000;
    std::uint32_t prefill_chunk        = 1024;
    bool fast_prefill_kernel           = false;
    std::filesystem::path context_cost_presets;
    std::string device_profile = "auto";
    std::filesystem::path device_profile_path;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    // Ordered CUDA devices, one pipeline stage each: primary first, also holding the embedding,
    // head and round state. Empty keeps the single-device route selected by `device`. Mutually
    // exclusive with --device.
    std::vector<int> devices;
    // Layers per stage, one count per entry of `devices`. Empty lets the engine choose.
    std::vector<std::uint32_t> stage_layers;
    // Qwen3.8-Flash-Next: where the expert banks live, and where its n-gram table comes from.
    ExpertResidency expert_residency = ExpertResidency::Device;
    std::optional<std::uint64_t> expert_cache_bytes;
    NgramTableOptions ngram_table;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    bool ngram_native_sessions = false;
    ContextCacheOptions context_cache;
    bool enable_vision      = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph     = true;
    bool lm_head_q4         = false;
    bool lm_head_q6         = false;
    bool embedding_q4       = false;
    bool embedding_q6       = false;
    bool mtp_experts_q4     = false;
    bool gdn_state_fp16     = false;
    bool rope_yarn          = false;
    float rope_yarn_factor  = 1.0F;
    float rope_scaling_factor                   = 1.0F;
    std::uint32_t rope_scaling_original_context = 0;
    bool wddm_evictable_budget = false;
    CudaMemoryPolicy cuda_memory_policy = CudaMemoryPolicy::DriverDefault;
    std::size_t cuda_vram_reserve_bytes = 64ULL << 20;
    std::size_t cuda_memory_probe_step_bytes = 128ULL << 20;
    // --model-suspend and its options: the residency API (suspend, resume) over fixed-address
    // device memory.
    ModelSuspendOptions suspend;
    // Router mode: no artifact argument; --models-dir and/or --models-preset name the models, which
    // load on demand. Every other option on the command line applies to each model.
    std::optional<std::filesystem::path> models_dir;
    std::optional<std::filesystem::path> models_preset;
    std::uint32_t models_max = 1;
    bool models_autoload     = true;
    // Seconds a model may sit idle before it is put to sleep (needs --model-suspend) or unloaded;
    // zero keeps it loaded. Both modes.
    std::uint32_t sleep_idle_seconds  = 0;
    std::uint32_t unload_idle_seconds = 0;
    // The arguments after the artifact path, router options removed: what router mode starts each
    // model with.
    std::vector<std::string> model_arguments;

    [[nodiscard]] bool router() const noexcept { return models_dir || models_preset; }
    bool mlp_a8_decode      = false;
    bool prefill_a8         = true;
    bool prefill_cublas     = false;
    bool prefill_cublas_projections = true;
    // Explicit total CUDA Graph driver-state allowance in MiB; 0 keeps the computed allowance.
    std::uint64_t cuda_graph_allowance_mib = 0;
    bool allow_prefix_reuse = true;
    // Offer shared-prefix candidates on a content-independent token grid so unrelated callers whose
    // prompts merely start alike converge on the same frontier. Off by default: it adds host-side
    // candidate work to every request and only pays for itself on a multi-tenant preamble.
    bool auto_prefix_grid = false;
    // --derive-session-keys: a request that names no session (Chat Completions, Messages) gets
    // one derived from its instructions and first user message, so its conversation keeps a
    // session lineage and LiveSession retention like a Responses conversation.
    bool derive_session_keys = false;
    // --lenient-assistant-history: Responses input whose assistant content or reasoning follows
    // function_call Items joins that run's turn instead of failing with invalid_assistant_history.
    bool lenient_assistant_history = false;
    // Directory for /slots session files; empty disables slot save/restore.
    std::filesystem::path slot_save_path;
    // Spill an involuntarily evicted session back to the slot file it was last saved to or
    // restored from. Requires slot_save_path.
    bool auto_save_evicted = false;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    // --graft NAME=PATH, repeatable: phantom-kv grafts a request may select with "graft": NAME.
    std::vector<GraftSource> grafts;
    // --default-graft NAME: graft applied to a request that states none ("graft": "" opts out).
    // Empty means no default; otherwise it names an entry of `grafts`.
    std::string default_graft;
    std::optional<std::uint32_t> default_thinking_budget;
    // Effort for requests that name none; a request that disables thinking is left alone.
    std::optional<RequestedReasoningEffort> default_reasoning_effort;
    // End-of-thinking message fed to the model when it hits the thinking budget; empty
    // preserves the model's built-in control suffix.
    std::string thinking_budget_message;
    // Output limit for a request that omits one. Unset means the Engine's concurrent lane budget:
    // see request_limits().
    std::optional<int> default_max_tokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    // --log-colours on|off: on colours the console log's levels and statistics, off keeps it plain;
    // unset colours the levels on a console only.
    std::optional<bool> log_colours;
    // --log-stats-panel on|off: pin the session statistics beneath the console log (terminal only).
    bool log_stats_panel   = false;
    bool enable_webui      = true;  // serve a WebUI compiled in with NINFER_WEBUI_DIR
    // --webui-mcp-proxy: relay the WebUI's MCP traffic at /cors-proxy. It reaches any http host it
    // is given and carries no API key, so it stays off unless asked for by name.
    bool webui_mcp_proxy = false;
    bool structured_output = false; // accept JSON/JSON Schema constrained requests
    bool concurrent_prefill = false; // admit to free lanes while other requests prefill
    // --recover-invariant-failures: a worker logic_error fails the active requests, not the Engine.
    bool recover_invariant_failures = false;
    // --unconstrained-response-format: without structured_output, generate a JSON or JSON Schema
    // request unconstrained instead of refusing it, for clients that always send a format.
    bool unconstrained_response_format = false;
    // --assistant-prefill: a Chat Completions request whose last message is the assistant's
    // continues that message in place, as /v1/messages always does.
    bool assistant_prefill = false;
    // --usage-chunk-choice: emit the streaming usage chunk with a zero-delta choice instead of the
    // OpenAI-conformant empty choices array. Strict client parsers (GitHub Copilot) reject the
    // empty array as "Response contained no choices"; the extra choice is inert for other clients.
    bool usage_chunk_choice = false;
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    // --post-thinking and --post-thinking-*: thinking requests switch to these overrides (omitted
    // fields from the model's post-thinking preset) from the token after reasoning closes. Without
    // them a request opts in with its own `post_thinking` object.
    std::optional<SamplingOverrides> post_thinking_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

// Parse-time limits. A request that omits max_tokens / max_completion_tokens / max_output_tokens
// gets --default-max-tokens when set; otherwise GenerationService asks the Engine for the largest
// budget that still lets every configured lane be admitted at once (the remaining context with one
// lane).
[[nodiscard]] inline RequestLimits request_limits(const ServeOptions& options) noexcept {
    return RequestLimits{.default_max_tokens        = options.default_max_tokens,
                         .max_context               = static_cast<int>(options.max_context),
                         .assistant_prefill         = options.assistant_prefill,
                         .lenient_assistant_history = options.lenient_assistant_history};
}

ServeOptions parse_serve_options(int argc, char** argv);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);

} // namespace ninfer::serve
