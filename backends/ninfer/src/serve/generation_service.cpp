#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "kvmem/monitor.hpp"
#include "kvmem/monitor_rate.hpp"

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "serve/structured_output.h"
#include "serve/translate.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    std::size_t peak   = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidToolConstraint:
        error.status = 400;
        error.param = "tools";
        error.code = "invalid_tool_constraint";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

// A raw prompt has no message structure to checkpoint, so the same prompt sent again (a regenerated
// completion) resumes only from a private anchor one token short of its end. The anchor costs one
// more prefill pass -- about 20 ms for a 27B model on the RTX 3090 -- and saves the whole prefill
// when the prompt comes back (1,602 tokens: 590 ms to 26 ms), so only a prompt whose prefill is
// several times that pass gets one.
constexpr std::size_t kRawPromptAnchorMinimumTokens = 512;

// The branched reads' records the memo keeps, about 1 KiB each.
constexpr std::size_t kBranchMemoRecords = 8192;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    // Translation already refuses a graft the server did not load; the Engine's own check is the
    // same contract, so it keeps the protocol code.
    if (std::string_view(error.message).starts_with("unknown graft '")) {
        error.param = "graft";
        error.code  = "unknown_graft";
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    media.image_detail        = part.image_detail;
    return media;
}

void trim_cache_markers(std::vector<ninfer::PromptCacheMarker>& markers, std::uint32_t maximum) {
    std::vector<ninfer::PromptCacheMarker> unique;
    unique.reserve(markers.size());
    for (const ninfer::PromptCacheMarker& marker : markers) {
        if (std::find(unique.begin(), unique.end(), marker) == unique.end()) {
            unique.push_back(marker);
        }
    }
    if (unique.size() > maximum) {
        unique.erase(unique.begin(), unique.end() - static_cast<std::ptrdiff_t>(maximum));
    }
    markers = std::move(unique);
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink, const PreparedRequest& prepared)
        : sink_(&sink), monitor_(prepared.monitor_update), prompt_(prepared.prompt_tokens) {}

    void start(ninfer::GenerationStart start) override {
        prompt_ = start.prompt.prompt_tokens;
        cached_ = start.reused_prompt_tokens;
        if (monitor_) monitor_({{"status", "reading"}, {"prompt_tokens", prompt_}, {"cached_tokens", cached_}});
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninfer::PromptProgress progress) override {
        cached_ = progress.reused_prompt_tokens;
        if (monitor_) monitor_({{"status", "reading"}, {"prompt_tokens", progress.total_prompt_tokens}, {"cached_tokens", cached_}});
        if (monitor_) monitor_({{"prompt_processed_tokens", progress.processed_prompt_tokens},
            {"prefill_elapsed_ms", progress.elapsed_ns / 1e6}});
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        const auto native = completion_timings_json(prompt_, cached_, timing);
        nlohmann::json values{{"status", "generating"}, {"generated_tokens", timing.generated_tokens},
            {"decode_tps", native.at("predicted_per_second")},
            {"decode_tps_mean", native.at("predicted_per_second")},
            {"prefill_tps", native.at("prompt_per_second")}, {"prefill_ms", native.at("prompt_ms")},
            {"ttft_ms", native.at("prompt_ms")}, {"ttft_scope", "execution"},
            {"decode_ms", native.at("predicted_ms")}, {"prompt_new_tokens", prompt_ - cached_}};
        values.update(rate_.sample(timing.generated_tokens, timing.generation_elapsed_ns, kvmem::monitor::now_ms()));
        if (monitor_) monitor_(values);
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (monitor_ && !delta.text.empty()) {
            const auto phase = delta.channel == ninfer::OutputChannel::Reasoning ? "thinking" : "answering";
            if (phase_ != phase) { phase_ = phase; monitor_({{"phase", phase}}); }
        }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (!delta.text.empty() && sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
            return;
        }
        if (delta.text.empty() && delta.logprobs.empty()) { return; }
        if (sink_->on_content) { sink_->on_content(delta.text, delta.logprobs); }
    }

private:
    const StreamSink* sink_ = nullptr;
    std::function<void(const nlohmann::json&)> monitor_;
    int prompt_ = 0, cached_ = 0;
    kvmem::monitor_rate_window rate_;
    std::string phase_;
};

} // namespace

ninfer::EngineOptions make_engine_options(const ServeOptions& options) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = options.artifact_path;
    engine_options.chat_template_path       = options.chat_template_path;
    engine_options.device                   = options.device;
    engine_options.max_context              = options.max_context;
    engine_options.kv_capacity              = options.kv_capacity;
    engine_options.kvmem                    = options.kvmem;
    engine_options.max_concurrency          = options.max_concurrency;
    engine_options.max_pending_requests     = options.max_pending_requests;
    engine_options.pending_timeout_ms       = options.pending_timeout_ms;
    engine_options.prefill_chunk            = options.prefill_chunk;
    engine_options.decode_rounds_per_prefill = options.decode_rounds_per_prefill;
    engine_options.fast_prefill_kernel      = options.fast_prefill_kernel;
    engine_options.kv_cache                 = options.kv_cache;
    engine_options.enable_vision            = options.enable_vision;
    engine_options.grafts                   = options.grafts;
    engine_options.vision_residency         = options.vision_residency;
    engine_options.cpu_vision_idle_seconds   = options.cpu_vision_idle_seconds;
    engine_options.vision_max_merged_tokens = options.vision_max_merged_tokens;
    engine_options.use_cuda_graph           = options.use_cuda_graph;
    engine_options.gpu_clock_floor_mhz      = options.gpu_clock_floor_mhz;
    engine_options.lm_head_q4               = options.lm_head_q4;
    engine_options.lm_head_q6               = options.lm_head_q6;
    engine_options.embedding_q4             = options.embedding_q4;
    engine_options.embedding_q6             = options.embedding_q6;
    engine_options.mtp_experts_q4           = options.mtp_experts_q4;
    engine_options.gdn_state_fp16           = options.gdn_state_fp16;
    engine_options.rope_yarn                = options.rope_yarn;
    engine_options.rope_yarn_factor         = options.rope_yarn_factor;
    engine_options.rope_scaling_factor           = options.rope_scaling_factor;
    engine_options.rope_scaling_original_context = options.rope_scaling_original_context;
    engine_options.structured_output        = options.structured_output;
    engine_options.concurrent_prefill       = options.concurrent_prefill;
    engine_options.recover_invariant_failures = options.recover_invariant_failures;
    engine_options.thinking_budget_message  = options.thinking_budget_message;
    engine_options.cuda_graph_allowance_bytes =
        static_cast<std::size_t>(options.cuda_graph_allowance_mib) << 20;
    engine_options.wddm_evictable_budget    = options.wddm_evictable_budget;
    engine_options.cuda_memory_policy = options.cuda_memory_policy;
    engine_options.cuda_vram_reserve_bytes = options.cuda_vram_reserve_bytes;
    engine_options.cuda_memory_probe_step_bytes = options.cuda_memory_probe_step_bytes;
    engine_options.suspend                  = options.suspend;
    engine_options.mlp_a8_decode            = options.mlp_a8_decode;
    engine_options.prefill_a8               = options.prefill_a8;
    engine_options.nvfp4_w4a8_int           = options.nvfp4_w4a8_int;
    engine_options.nvfp4_w4a16              = options.nvfp4_w4a16;
    engine_options.prefill_cublas           = options.prefill_cublas;
    engine_options.prefill_cublas_projections = options.prefill_cublas_projections;
    engine_options.speculative              = options.speculative;
    engine_options.context_cache            = options.context_cache;
    engine_options.slot_auto_save.enabled   = options.auto_save_evicted;
    engine_options.devices                  = options.devices;
    engine_options.stage_layers             = options.stage_layers;
    engine_options.tensor_parallel          = options.tensor_parallel;
    engine_options.expert_residency           = options.expert_residency;
    engine_options.ngram_table                = options.ngram_table;
    engine_options.kv_resident_positions      = options.kv_resident_positions;
    engine_options.expert_cache_bytes         = options.expert_cache_bytes;
    engine_options.hybrid_experts             = options.hybrid_experts;
    engine_options.context_cost.preset_path = options.context_cost_presets;
    engine_options.device_profile           = options.device_profile;
    engine_options.device_profile_path      = options.device_profile_path;
    engine_options.media_cache_bytes        = options.media_cache_bytes;
    engine_options.media_live_bytes         = options.media_live_bytes;
    engine_options.media_preprocess_threads = options.media_preprocess_threads;
    return engine_options;
}

GenerationService::GenerationService(
    ServeOptions options, StartupObserver startup_observer, DiagnosticObserver diagnostic_observer,
    std::function<void(const ninfer::SlotAutoSaveEvent&)> auto_save_listener)
    : options_(std::move(options)) {
    // Inline ECC on GDDR6X GeForce cards reserves ~6.25% of VRAM for checksums and taxes
    // memory bandwidth on every access. Decode is bandwidth-bound, so an ECC-enabled card
    // silently loses a large share of its published throughput and KV capacity while looking
    // exactly like an engine regression. ECC is off by default on GeForce; warn loudly when
    // someone (or some tool) left it on.
    {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, options_.device) == cudaSuccess &&
            props.ECCEnabled != 0) {
            write_console_log(ConsoleLogLevel::Warning,
                              std::string("ECC is ENABLED on ") + props.name +
                                  ": GDDR6X stores ECC checksums in VRAM, costing ~6.25% of "
                                  "capacity (23,028 vs 24,564 MiB on a 24 GB card) and ~12% of "
                                  "memory bandwidth (measured 803 vs 902 GB/s on an RTX 3090 Ti). "
                                  "KV capacity and prefill suffer accordingly. ECC is off by "
                                  "default on GeForce; if this is not a deliberate reliability "
                                  "choice, disable it with `nvidia-smi -e 0` and reboot.");
        }
    }
    ninfer::EngineOptions engine_options = make_engine_options(options_);
    engine_options.startup_observer      = std::move(startup_observer);
    engine_options.diagnostic_observer   = std::move(diagnostic_observer);
    engine_options.slot_auto_save.listener = std::move(auto_save_listener);
    engine_           = std::make_unique<ninfer::Engine>(std::move(engine_options));
    // A request that says nothing about thinking gets its template's default. Qwen's templates
    // think unless told not to; Gemma 4's chat template keeps thinking off unless enable_thinking
    // asks for it, so without --no-thinking or a request flag a Gemma 4 server answers directly.
    // Flint's template has no switch: thinking on, the default, leaves reasoning to the model, and
    // thinking off closes an empty reasoning block, after which it answers at once.
    if (!options_.enable_thinking && engine_->load_summary().architecture == "Gemma4ForCausalLM") {
        options_.enable_thinking = false;
    }
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    count_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    branch_memo_ = std::make_shared<BranchMemo>(kBranchMemoRecords);
}

std::size_t GenerationService::admitted_requests() const {
    std::lock_guard lock(request_capacity_->mutex);
    return request_capacity_->active;
}

std::size_t GenerationService::peak_admitted_requests() const {
    std::lock_guard lock(request_capacity_->mutex);
    return request_capacity_->peak;
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    return acquire_lifetime(request_capacity_, deadline_policy, "inference request queue is full");
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_lifetime(const std::shared_ptr<RequestCapacity>& capacity,
                                    DeadlinePolicy deadline_policy,
                                    const char* full_message) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(capacity->mutex);
        if (capacity->active >= capacity->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded, full_message));
        }
        ++capacity->active;
        capacity->peak = std::max(capacity->peak, capacity->active);
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(capacity, started, deadline);
    } catch (...) {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    std::vector<StagedRequest> one;
    one.push_back(stage(request, consumer_mode, observation, std::move(is_cancelled),
                        std::move(context_cache)));
    return std::move(submit_group(std::move(one)).front());
}

StagedRequest GenerationService::stage(const GenerationRequest& request,
                                       GenerationConsumerMode consumer_mode,
                                       ninfer::GenerationObservationOptions observation,
                                       std::function<bool()> is_cancelled,
                                       ContextCacheHints context_cache) const {
    if (consumer_mode == GenerationConsumerMode::Streaming) {
        observation.live_timings = true;
        observation.prompt_progress = true;
    }
    return stage_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout);
}

std::vector<PreparedRequest>
GenerationService::submit_group(std::vector<StagedRequest> staged) const {
    std::vector<ninfer::GenerationSubmission> submissions;
    submissions.reserve(staged.size());
    for (StagedRequest& request : staged) { submissions.push_back(std::move(request.submission)); }
    std::vector<ninfer::GenerationHandle> handles;
    try {
        handles = engine_->submit_group(std::move(submissions));
    } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    std::vector<PreparedRequest> prepared;
    prepared.reserve(staged.size());
    for (std::size_t index = 0; index < staged.size(); ++index) {
        PreparedRequest& request          = staged[index].prepared;
        request.generation                = std::move(handles[index]);
        request.sampling                  = request.generation.resolved_sampling();
        request.effective_thinking_budget = request.generation.effective_thinking_budget();
        prepared.push_back(std::move(request));
    }
    return prepared;
}

StagedRequest GenerationService::stage_read(const BranchRead& read,
                                            ninfer::GenerationObservationOptions observation,
                                            std::function<bool()> is_cancelled,
                                            const ContextCacheHints& context_cache) const {
    StagedRequest staged = stage(*read.request, GenerationConsumerMode::Aggregate, observation,
                                 std::move(is_cancelled), context_cache);
    if (!read.continuation.empty()) {
        const std::span<const ninfer::TokenId> base = staged.submission.prompt.token_ids();
        std::vector<ninfer::TokenId> tokens(base.begin(), base.end());
        tokens.insert(tokens.end(), read.continuation.begin(), read.continuation.end());
        staged.prepared.prompt_tokens = static_cast<int>(tokens.size());
        try {
            staged.submission.prompt = engine_->prepare_tokens(std::move(tokens), true, false,
                                                               read.request->speculative_drafting);
        } catch (const std::out_of_range& exception) {
            throw_invalid_input(std::invalid_argument(exception.what()), "invalid_prompt");
        }
    }
    staged.submission.options.execution.readout_tokens = read.readout;
    return staged;
}

PreparedRequest GenerationService::submit_branches(std::span<const BranchRead> reads,
                                                   ninfer::GenerationObservationOptions observation,
                                                   std::function<bool()> is_cancelled,
                                                   const ContextCacheHints& context_cache) const {
    if (reads.empty()) { throw std::invalid_argument("submit_branches needs one read or more"); }
    StagedRequest first = stage_read(BranchRead{.request = reads.front().request}, observation,
                                     is_cancelled, context_cache);
    ninfer::RequestOptions options = std::move(first.submission.options);
    options.execution.readout_tokens.clear();
    // The prompt of the request whose reads come next, without any read's continuation.
    const GenerationRequest* base_request               = reads.front().request;
    const std::span<const ninfer::TokenId> first_tokens = first.submission.prompt.token_ids();
    std::vector<ninfer::TokenId> base_tokens(first_tokens.begin(), first_tokens.end());
    const auto continued = [&](std::span<const ninfer::TokenId> continuation) {
        std::vector<ninfer::TokenId> tokens = base_tokens;
        tokens.insert(tokens.end(), continuation.begin(), continuation.end());
        try {
            // A branch's prompt only supplies its tokens: it drafts nothing.
            return engine_->prepare_tokens(std::move(tokens), true, false, false);
        } catch (const std::out_of_range& exception) {
            throw_invalid_input(std::invalid_argument(exception.what()), "invalid_prompt");
        }
    };
    std::vector<ninfer::GenerationBranch> branches;
    branches.reserve(reads.size());
    branches.push_back(ninfer::GenerationBranch{
        .prompt = reads.front().continuation.empty() ? std::move(first.submission.prompt)
                                                     : continued(reads.front().continuation),
        .readout_tokens = reads.front().readout});
    for (std::size_t index = 1; index < reads.size(); ++index) {
        const BranchRead& read = reads[index];
        if (read.request != base_request) {
            StagedRequest next = stage(*read.request, GenerationConsumerMode::Aggregate,
                                       observation, is_cancelled, context_cache);
            const std::span<const ninfer::TokenId> tokens = next.submission.prompt.token_ids();
            base_tokens.assign(tokens.begin(), tokens.end());
            base_request = read.request;
            if (read.continuation.empty()) {
                branches.push_back(ninfer::GenerationBranch{
                    .prompt = std::move(next.submission.prompt), .readout_tokens = read.readout});
                continue;
            }
        }
        branches.push_back(ninfer::GenerationBranch{.prompt         = continued(read.continuation),
                                                    .readout_tokens = read.readout});
    }
    // The memo answers the reads it holds; the Engine runs the rest.
    const std::optional<int> top_logprobs = reads.front().request->reported_top_logprobs();
    HeldBranches held;
    held.records.resize(reads.size());
    std::vector<ninfer::GenerationBranch> pending;
    pending.reserve(branches.size());
    for (std::size_t index = 0; index < branches.size(); ++index) {
        std::optional<BranchKey> key;
        if (reads[index].memo && reads[index].request->cache_prompt) {
            key = branch_key(branches[index].prompt.token_ids(), branches[index].readout_tokens,
                             top_logprobs);
            if (std::optional<ninfer::BranchResult> record = branch_memo_->find(*key)) {
                held.records[index] = std::move(record);
                continue;
            }
        }
        held.run.push_back(key);
        pending.push_back(std::move(branches[index]));
    }
    ninfer::GenerationHandle handle;
    if (!pending.empty()) {
        try {
            handle = engine_->submit_branches(
                std::move(pending), std::move(options), first.submission.consumer_mode,
                first.submission.observation, first.submission.pending_deadline);
        } catch (const ninfer::RequestError& exception) {
            throw_request_error(exception);
        } catch (const std::invalid_argument& exception) {
            throw_invalid_input(exception, "invalid_prompt");
        }
    }
    PreparedRequest prepared           = std::move(first.prepared);
    prepared.generation                = std::move(handle);
    prepared.sampling                  = prepared.generation.resolved_sampling();
    prepared.effective_thinking_budget = prepared.generation.effective_thinking_budget();
    prepared.held_branches             = std::move(held);
    return prepared;
}

StagedRequest GenerationService::stage_impl(const GenerationRequest& incoming,
                                            GenerationConsumerMode consumer_mode,
                                            ninfer::GenerationObservationOptions observation,
                                            std::function<bool()> is_cancelled,
                                            ContextCacheHints context_cache,
                                            CacheParticipation cache_participation,
                                            DeadlinePolicy deadline_policy) const {
    const std::optional<GenerationRequest> unconstrained = drop_unavailable_constraints(
        incoming, options_.structured_output, options_.unconstrained_response_format);
    const GenerationRequest& request = unconstrained ? *unconstrained : incoming;
    if (!request.cache_prompt) { cache_participation = CacheParticipation::Disabled; }
    StagedRequest staged;
    PreparedRequest& prepared               = staged.prepared;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    request_options.ngram_session        = request.ngram_session;
    prepared.thinking_budget             = request_options.execution.thinking.budget;
    prepared.reasoning_effort            = semantics.reasoning_effort;
    prepared.requested_reasoning_effort  = semantics.requested_reasoning_effort;
    prepared.preserve_thinking           = semantics.preserve_thinking;
    prepared.requested_preserve_thinking = semantics.requested_preserve_thinking;
    prepared.parallel_tool_calls         = request.parallel_tool_calls;
    const bool request_has_media         = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        ninfer::PreparedPrompt prompt;
        if (request.raw_prompt) {
            std::vector<ninfer::TokenId> tokens;
            for (const RawPromptPiece& piece : *request.raw_prompt) {
                if (piece.token) {
                    tokens.push_back(*piece.token);
                    continue;
                }
                const std::vector<ninfer::TokenId> encoded = engine_->tokenize_text(piece.text);
                tokens.insert(tokens.end(), encoded.begin(), encoded.end());
            }
            const auto invalid_prompt = [](std::string message) {
                ApiError error;
                error.status  = 400;
                error.param   = "prompt";
                error.code    = "invalid_prompt";
                error.message = std::move(message);
                throw ApiException(std::move(error));
            };
            if (tokens.empty()) { invalid_prompt("prompt must not be empty"); }
            const bool cached = cache_participation == CacheParticipation::ReadWrite;
            const bool anchor = cached && tokens.size() >= kRawPromptAnchorMinimumTokens;
            try {
                prompt = engine_->prepare_tokens(std::move(tokens), cached, anchor,
                                                 request.speculative_drafting);
            } catch (const std::out_of_range& exception) {
                // A token id outside the vocabulary is the client's input, not a server fault.
                invalid_prompt(exception.what());
            }
            prepared.acquisition_seconds =
                std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        } else {
            std::size_t remaining_media_bytes =
                std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
            ninfer::PromptInput input =
                to_prompt_input(request, semantics, [&](const ContentPart& part) {
                    return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                         remaining_media_bytes);
                });
            std::vector<PromptCacheMarker> protocol_markers =
                std::move(input.context_cache.markers);
            const bool protocol_allows_engine_automatic =
                input.context_cache.allow_engine_automatic_shared_prefixes;
            input.context_cache = std::move(context_cache);
            input.context_cache.markers.insert(input.context_cache.markers.end(),
                                               std::make_move_iterator(protocol_markers.begin()),
                                               std::make_move_iterator(protocol_markers.end()));
            input.context_cache.allow_engine_automatic_shared_prefixes =
                input.context_cache.allow_engine_automatic_shared_prefixes &&
                protocol_allows_engine_automatic;
            input.context_cache.allow_engine_prefix_grid =
                input.context_cache.allow_engine_prefix_grid || options_.auto_prefix_grid;
            if (options_.derive_session_keys && !input.context_cache.session_key &&
                cache_participation == CacheParticipation::ReadWrite) {
                input.context_cache.session_key = derived_session_key(request);
            }
            trim_cache_markers(
                input.context_cache.markers,
                engine_->options().context_cache.max_cache_markers_per_request.value());
            prepared.acquisition_seconds =
                std::chrono::duration<double>(Clock::now() - acquisition_started).count();
            check_preparation_control(prepared.lifetime->deadline, is_cancelled);
            const PreparationControl control{
                .deadline     = prepared.lifetime->deadline,
                .cancellation = CancellationView(is_cancelled),
            };
            prompt = engine_->prepare(std::move(input), control);
        }
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
            prepared.effective_thinking_budget.reset();
        }
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        if (request.derive_output_budget) {
            request_options.execution.requested_output_tokens =
                engine_->concurrent_output_budget(prompt);
        }
        prepared.requested_output_tokens =
            static_cast<int>(request_options.execution.requested_output_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        staged.submission = ninfer::GenerationSubmission{
            .prompt           = std::move(prompt),
            .options          = std::move(request_options),
            .consumer_mode    = consumer_mode == GenerationConsumerMode::Streaming
                                    ? ninfer::OutputConsumerMode::Streaming
                                    : ninfer::OutputConsumerMode::Aggregate,
            .observation      = observation,
            .pending_deadline = prepared.lifetime->deadline};
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return staged;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const std::shared_ptr<RequestLifetime> lifetime = acquire_lifetime(
        count_capacity_, DeadlinePolicy::ClientPendingTimeout, "token count queue is full");
    const Clock::time_point deadline        = lifetime->deadline;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

std::string GenerationService::render_prompt(const GenerationRequest& request,
                                             std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const std::shared_ptr<RequestLifetime> lifetime = acquire_lifetime(
        count_capacity_, DeadlinePolicy::ClientPendingTimeout, "token count queue is full");
    const Clock::time_point deadline        = lifetime->deadline;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        std::string text;
        for (const ninfer::TokenId token : prompt.token_ids()) {
            text += engine_->token_bytes(token);
        }
        return text;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink, prepared); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    if (prepared.held_branches && !prepared.generation) {
        // The memo held every read: no Engine request ran, and none of the prompt was computed.
        GenerationOutcome outcome;
        outcome.prompt_tokens = prepared.prompt_tokens;
        outcome.metrics.prefix_cache_hit_tokens =
            static_cast<std::uint32_t>(prepared.prompt_tokens);
        outcome.metrics.prepare_seconds = prepared.prepare_seconds;
        outcome.metrics.total_seconds   = prepared.prepare_seconds;
        for (std::optional<ninfer::BranchResult>& record : prepared.held_branches->records) {
            outcome.branches.push_back(std::move(*record));
        }
        return outcome;
    }
    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) { throw_request_error(exception); }
    if (prepared.held_branches && result.branches.size() == prepared.held_branches->run.size()) {
        // The Engine's records in read order between the held ones, and into the memo.
        HeldBranches& held                         = *prepared.held_branches;
        std::vector<ninfer::BranchResult> computed = std::move(result.branches);
        result.branches.clear();
        result.branches.reserve(held.records.size());
        std::size_t next = 0;
        for (std::optional<ninfer::BranchResult>& record : held.records) {
            if (record) {
                result.branches.push_back(std::move(*record));
                continue;
            }
            if (const std::optional<BranchKey>& key = held.run[next]) {
                branch_memo_->keep(*key, computed[next]);
            }
            result.branches.push_back(std::move(computed[next++]));
        }
    }
    GenerationOutcome outcome;
    outcome.tokens              = result.generated_token_ids;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.content_logprobs    = std::move(result.content_logprobs);
    outcome.branches            = std::move(result.branches);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.overlay_windows           = result.timings.overlay_windows;
    outcome.metrics.overlay_exclusive_windows = result.timings.overlay_exclusive_windows;
    outcome.metrics.overlay_window_seconds  = result.timings.overlay_window_seconds;
    outcome.metrics.overlay_evict_seconds   = result.timings.overlay_evict_seconds;
    outcome.metrics.overlay_restore_seconds = result.timings.overlay_restore_seconds;
    outcome.metrics.overlay_evicted_bytes   = result.timings.overlay_evicted_bytes;
    outcome.metrics.overlay_staged_bytes    = result.timings.overlay_staged_bytes;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.engine_timing                = result.engine_timing;
    outcome.metrics.computed_prefill_tokens      = result.computed_prompt_tokens;
    outcome.metrics.prefix_cache_hit_tokens      = result.reused_prompt_tokens;
    outcome.id_slot                              = result.slot;
    outcome.session_digest                       = result.session_digest;
    outcome.metrics.prefix_reuse_path            = result.prefix_reuse_path;
    outcome.metrics.materialization              = result.materialization;
    outcome.metrics.speculative_backend          = result.speculative.backend;
    outcome.metrics.speculative_draft_window     = result.speculative.draft_window;
    outcome.metrics.speculative_rounds           = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens     = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens  = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps   = result.speculative.fallback_steps;
    outcome.metrics.ngram_rounds                 = result.speculative.ngram_rounds;
    outcome.metrics.ngram_drafted_tokens         = result.speculative.ngram_drafted_tokens;
    outcome.metrics.ngram_accepted_tokens        = result.speculative.ngram_accepted_tokens;
    outcome.metrics.ngram_archive_rounds         = result.speculative.ngram_archive_rounds;
    outcome.metrics.ngram_archive_drafted_tokens = result.speculative.ngram_archive_drafted_tokens;
    outcome.metrics.ngram_archive_accepted_tokens =
        result.speculative.ngram_archive_accepted_tokens;
    outcome.metrics.ngram_archive = result.ngram_archive;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);
    outcome.metrics.speculative_adaptive           = result.speculative.adaptive;
    outcome.metrics.speculative_window_transitions = result.speculative.window_transitions;
    outcome.metrics.speculative_rounds_per_window = std::move(result.speculative.rounds_per_window);

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    // parallel_tool_calls=false promises the caller at most one tool call per assistant turn.
    // Decoding is not constrained, so enforce it here: keep the first call and drop the rest. The
    // model can ask for the next one on the following turn, which is how a sequential executor
    // consumes them anyway.
    if (!prepared.parallel_tool_calls && outcome.tool_calls.size() > 1) {
        outcome.tool_calls.resize(1);
    }
    return outcome;
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    std::vector<StagedRequest> staged;
    staged.push_back(stage_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                                CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup));
    PreparedRequest prepared = std::move(submit_group(std::move(staged)).front());
    run(prepared, nullptr);
}

} // namespace ninfer::serve
