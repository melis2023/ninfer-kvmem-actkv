#include "runtime/engine/qwen4_exp_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/startup.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen4_exp/ngram_component.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/target_logprobs.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/effective_thinking_budget.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/model_instance.h"
#include "text/structured_output.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iterator>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <variant>

namespace ninfer::runtime {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point from, Clock::time_point to) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

double seconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

// Requests fail alone on their own errors; anything else may have left the device in an unknown
// state, so it fails the Engine.
bool request_error(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const RequestError&) { return true; } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {}
    return false;
}

} // namespace

bool is_qwen4_exp_artifact(const std::filesystem::path& path) {
    if (path.extension() != ".ninfer") { return false; }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) { return false; }
    std::optional<artifact::Reader> reader;
    try {
        reader.emplace(path);
    } catch (const std::exception&) { return false; }
    if (models::qwen4_exp::is_ngram_table_artifact(*reader)) {
        throw std::invalid_argument(path.string() +
                                    " is a Qwen3.8-Flash-Next n-gram table, not a model; pass it "
                                    "with --ngram-table next to the model");
    }
    return models::qwen4_exp::is_qwen4_exp(*reader);
}

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device) {
    const auto start = Clock::now();
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.enable_vision && options.vision_residency != VisionResidency::Resident) {
        throw std::invalid_argument(
            "Qwen3.8-Flash-Next runs its Vision tower resident on the device only");
    }
    if (options.speculative.backend != SpeculativeBackend::None ||
        options.speculative.ngram_draft_tokens != 0) {
        throw std::invalid_argument("speculative decoding is not available for Qwen3.8-Flash-Next");
    }
    if (options.context_cache.enabled && !options.context_cache.disk_kv_path.empty()) {
        throw std::invalid_argument(
            "the context cache's disk KV tier is not available for Qwen3.8-Flash-Next");
    }
    const NgramTableOptions& table = options.ngram_table;
    if (table.disabled && (!table.path.empty() || table.ram)) {
        throw std::invalid_argument("--no-ngram-table excludes --ngram-table and --ngram-ram");
    }
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    // The n-gram table is located before anything else starts, so a model without one fails at
    // once.
    std::optional<models::qwen4_exp::NgramTableSource> ngram;
    if (!table.disabled) {
        ngram = models::qwen4_exp::ngram_table_source(
            reader, options.artifact_path,
            models::qwen4_exp::parse_text_config(reader.directory().component("text").config),
            table.path);
    }
    inspect.complete();
    install_device_route_profile_for(options, device);
    models::qwen4_exp::LoadOptions load;
    load.artifact     = options.artifact_path;
    load.ranks        = device.size();
    load.stage_layers = options.stage_layers;
    load.experts      = options.expert_residency;
    load.vision       = options.enable_vision;
    StartupPhaseScope materialize(options.startup_observer, StartupPhase::TargetPlan);
    auto model = models::qwen4_exp::load_model(reader, load, device, &options.startup_observer);
    device.synchronize();
    materialize.complete();
    const auto free_bytes = [&] {
        RankBinding bind(device, 0);
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    };
    const std::size_t free_after_weights = free_bytes();

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(Qwen4ExpInstance{
        .model    = nullptr,
        .frontend = models::qwen3_5::make_frontend(
            model->resources(), {.chat_template_path      = options.chat_template_path,
                                 .architecture            = models::Architecture::Qwen4Exp,
                                 .vision_enabled          = options.enable_vision,
                                 .max_context             = options.max_context,
                                 .media_cache_bytes       = options.media_cache_bytes,
                                 .media_live_bytes        = options.media_live_bytes,
                                 .thinking_budget_message = options.thinking_budget_message}),
        .executor = nullptr,
        .capacity = options.max_context});
    frontend_phase.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    models::qwen4_exp::ExecutorOptions executor;
    executor.max_context     = options.max_context;
    executor.prefill_chunk   = std::clamp<std::uint32_t>(options.prefill_chunk, 64, 4096);
    executor.sequences       = options.max_concurrency;
    executor.ngram           = std::move(ngram);
    executor.ngram_residency = table.ram ? models::qwen4_exp::NgramResidency::Ram
                                         : models::qwen4_exp::NgramResidency::Disk;
    executor.expert_cache_bytes =
        options.expert_cache_bytes.value_or(models::qwen4_exp::ExecutorOptions::kAutoExpertCache);
    executor.cuda_graphs = options.use_cuda_graph;
    executor.vision_max_merged_tokens = options.vision_max_merged_tokens;
    executor.kv_cache                 = options.kv_cache;
    instance->executor = std::make_unique<models::qwen4_exp::Executor>(*model, device, executor);
    device.synchronize();
    program.complete();
    instance->free_after_weights = free_after_weights;
    instance->free_after_startup = free_bytes();
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                       "Qwen3.8-Flash-Next: %zu stage(s), experts in %s memory, n-gram table "
                       "%s, state %.0f MiB, workspace %.0f MiB, expert cache %.0f MiB",
                       model->stages().stages(),
                       options.expert_residency == ExpertResidency::Host   ? "pinned host"
                       : options.expert_residency == ExpertResidency::Disk ? "the artifact's files"
                                                                           : "device",
                       table.disabled      ? "off"
                       : table.ram         ? "in RAM"
                       : table.path.empty() ? "read from the artifact"
                                            : "read from its table artifact",
                       double(instance->executor->memory().state_bytes) / 1048576.0,
                       double(instance->executor->memory().workspace_bytes) / 1048576.0,
                       double(instance->executor->memory().expert_cache_bytes) / 1048576.0);
    if (table.disabled) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                           "Qwen3.8-Flash-Next runs WITHOUT its n-gram table (--no-ngram-table): "
                           "a non-standard experimental mode. The model was trained with the "
                           "table and degrades badly without it (WikiText-2 perplexity 2.66 -> "
                           "5.01 on GSQ-RCO Q2_0); use it only for experiments");
    }

    ConstructedQwen4Exp out;
    const auto& stats     = model->storage_stats();
    out.load.architecture = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    out.load.model_name   = model->info().name;
    out.load.prefill_signature = "qwen4_exp";
    std::set<std::string> formats;
    for (const auto& weight : model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    out.load.weight_formats.assign(formats.begin(), formats.end());
    out.load.load_seconds             = seconds(start, Clock::now());
    out.load.upload_seconds           = stats.upload_seconds;
    out.load.artifact_bytes_read      = stats.read_bytes;
    out.load.host_to_device_bytes     = stats.h2d_bytes;
    out.load.peak_staging_bytes       = stats.peak_staging_bytes;
    out.load.pinned_weight_bytes      = stats.pinned_bytes;
    out.load.device_object_count      = stats.device_object_count;
    out.load.host_object_count        = stats.host_object_count;
    const auto& config                = model->config();
    out.model_metadata.model_id       = out.load.model_name;
    out.model_metadata.vocab_size     = config.vocab_size;
    out.model_metadata.embedding_size = config.hidden_size;
    out.model_metadata.native_context = config.max_position_embeddings;
    std::set<std::string> tensor_formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; }
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        out.model_metadata.parameters += elements;
        out.model_metadata.weight_bytes += tensor->bytes;
        tensor_formats.emplace(tensor->format);
    }
    for (const auto& format : tensor_formats) {
        if (!out.model_metadata.weights_id.empty()) { out.model_metadata.weights_id += "+"; }
        out.model_metadata.weights_id += format;
    }
    instance->model = std::move(model);
    out.instance    = std::move(instance);
    return out;
}

struct Qwen4ExpCore::Request {
    std::uint64_t id = 0;
    models::qwen3_5::PreparedPrompt prompt;
    std::vector<TokenId> prompt_tokens;
    models::qwen3_5::OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    double vision_seconds  = 0.0; // the Vision tower's run over the prompt's media
    ResolvedRequestOptions options;
    OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;
    GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::atomic<bool> cancelled{false};

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<std::variant<OutputDelta, GenerationTimingObservation>> events;
    bool response_done = false;
    std::exception_ptr error;
    GenerationResult result;
    std::string content, reasoning;
    std::vector<TokenLogprob> content_logprobs;

    // Where the context cache snapshots the sequence: the prompt's turn closure (the end of the
    // last user turn, which the next turn's prompt repeats although it renders this turn's answer
    // differently), else the prompt's end; and whether the prompt may be reused at all.
    std::uint32_t anchor_at = 0;
    bool reusable           = true;
    bool media              = false; // images or video: encoded at the prompt's first chunk

    // The worker's progress with an admitted request.
    std::uint32_t slot      = 0;
    std::uint32_t reused    = 0; // prompt tokens its sequence already held
    PrefixReusePath reuse_path = PrefixReusePath::Root; // where they came from
    std::uint32_t prefilled = 0; // prompt tokens its sequence holds, the reused ones included
    bool decoding           = false;
    bool penalties          = false;
    bool post_thinking      = false;
    std::uint32_t position  = 0; // tokens fed: the logical position of the next sampled token
    std::vector<TokenId> feed;   // what the next decode step feeds
    std::vector<TokenId> generated;
    std::optional<GenerationBudget> budget;
    Clock::time_point admitted, prefill_start, prefill_end, first_token, last_token;
};

struct Qwen4ExpCore::Impl {
    // One executor sequence, the request it serves and what the context cache keeps of it: the
    // tokens its live state holds, and a snapshot of its state at the end of the last prompt it
    // prefilled, which a later prompt that starts with that one resumes from.
    struct Slot {
        std::shared_ptr<Request> request;
        std::vector<TokenId> fed;
        std::vector<TokenId> anchor;
        models::qwen4_exp::SequenceSnapshot snapshot;
        std::uint64_t last_used = 0;
    };

    // One row of a sampling call: the request's parameters at its position.
    struct SampleRow {
        const ResolvedSamplingParameters* params = nullptr;
        std::uint32_t position                   = 0;
        std::uint32_t slot                       = 0;
        bool counts                              = false;
        bool logprobs                            = false;
        const text::GrammarState* grammar        = nullptr;
    };

    Qwen4ExpInstance& instance;
    DeviceContext& device;
    const std::uint32_t max_context;
    const std::size_t max_outstanding;
    const std::chrono::milliseconds pending_timeout;
    const std::uint32_t domain;
    const bool structured_output;
    const bool reuse_prefixes;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::shared_ptr<Request>> pending;
    std::size_t outstanding = 0;
    bool stopping           = false;
    bool failed             = false;
    std::uint64_t next_id   = 1;

    mutable std::mutex stats_mutex;
    RuntimeStats stats;

    // Worker-owned.
    std::vector<Slot> slots; // by executor sequence
    std::uint64_t use_clock = 0;
    bool prefill_turn       = true; // a prefill chunk and a decode step take turns

    // Head-device sampling planes, a row per slot; with structured output the grammars' token
    // bitmasks too. The host side is staged in pinned memory: configs, positions, sampled tokens.
    DeviceBuffer sample_config, sample_position, sample_out, token_counts, score_targets, score_out;
    DeviceBuffer token_mask;
    // The logprob gather of a sampling call whose rows ask for it: [kLogprobTopK, rows] ids and
    // values, [rows] log-sum-exps and the gather's enabling flag, with the host copy.
    DeviceBuffer logprob_ids, logprob_values, logprob_lse, logprob_flag;
    std::unique_ptr<PinnedHostBuffer> host_mask, host_sample, host_logprobs;
    std::unique_ptr<WorkspaceArena> sample_workspace;

    std::thread worker;

    Impl(Qwen4ExpInstance& i, DeviceContext& d, const EngineOptions& options)
        : instance(i), device(d), max_context(options.max_context),
          max_outstanding(std::size_t(options.max_concurrency) + options.max_pending_requests),
          pending_timeout(options.pending_timeout_ms),
          domain(i.model->resources().public_token_count),
          structured_output(options.structured_output),
          reuse_prefixes(options.context_cache.enabled), slots(i.executor->options().sequences) {
        RankBinding bind(device, i.executor->head_rank());
        const std::size_t rows = slots.size();
        { // Native tool grammars share these masks even when JSON formats are disabled.
            token_mask = DeviceBuffer(rows * mask_words() * sizeof(std::uint32_t));
            host_mask =
                std::make_unique<PinnedHostBuffer>(rows * mask_words() * sizeof(std::uint32_t));
        }
        sample_config   = DeviceBuffer(rows * sizeof(ops::SamplingConfig));
        sample_position = DeviceBuffer(rows * sizeof(std::int32_t));
        sample_out      = DeviceBuffer(rows * sizeof(std::int32_t));
        host_sample     = std::make_unique<PinnedHostBuffer>(
            rows * (sizeof(ops::SamplingConfig) + 2 * sizeof(std::int32_t)));
        token_counts     = DeviceBuffer(rows * domain * sizeof(std::int32_t));
        score_targets    = DeviceBuffer(4096 * sizeof(std::int32_t));
        score_out        = DeviceBuffer(4096 * sizeof(float));
        const std::size_t top = rows * kMaximumTokenLogprobs;
        logprob_ids           = DeviceBuffer(top * sizeof(std::int32_t));
        logprob_values        = DeviceBuffer(top * sizeof(float));
        logprob_lse           = DeviceBuffer(rows * sizeof(float));
        logprob_flag          = DeviceBuffer(sizeof(std::int32_t));
        host_logprobs =
            std::make_unique<PinnedHostBuffer>(top * (sizeof(std::int32_t) + sizeof(float)));
        const std::int32_t enabled = 1;
        CUDA_CHECK(cudaMemcpy(logprob_flag.p, &enabled, sizeof(enabled), cudaMemcpyHostToDevice));
        sample_workspace = std::make_unique<WorkspaceArena>(std::max<std::size_t>(
            {ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(domain), 1,
                                                    static_cast<std::int32_t>(rows)),
             ops::logprob_topk_workspace_capacity_bytes(static_cast<std::int32_t>(domain),
                                                        static_cast<std::int32_t>(rows)),
             std::size_t{256}}));
        worker = std::thread([this] {
            device.bind_to_current_thread();
            loop();
        });
    }

    ~Impl() {
        {
            std::lock_guard lock(queue_mutex);
            stopping = true;
        }
        queue_cv.notify_all();
        if (worker.joinable()) { worker.join(); }
    }

    [[nodiscard]] std::size_t mask_words() const noexcept {
        return (std::size_t(domain) + 31) / 32;
    }

    void complete(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->error         = std::move(error);
                request->response_done = true;
            }
        }
        request->cv.notify_all();
        std::lock_guard lock(queue_mutex);
        --outstanding;
    }

    void complete(const std::shared_ptr<Request>& request, GenerationResult result) {
        {
            std::lock_guard lock(request->mutex);
            if (!request->response_done) {
                request->result        = std::move(result);
                request->response_done = true;
            }
        }
        request->cv.notify_all();
        std::lock_guard lock(queue_mutex);
        --outstanding;
    }

    [[nodiscard]] bool any_active() const {
        return std::any_of(slots.begin(), slots.end(),
                           [](const Slot& slot) { return slot.request != nullptr; });
    }

    void loop() {
        for (;;) {
            {
                std::unique_lock lock(queue_mutex);
                queue_cv.wait(lock, [&] { return stopping || !pending.empty() || any_active(); });
                if (stopping) {
                    auto waiting = std::move(pending);
                    pending.clear();
                    lock.unlock();
                    const auto unavailable = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is stopping"));
                    for (auto& item : waiting) { complete(item, unavailable); }
                    for (auto& slot : slots) {
                        if (slot.request) { complete(std::exchange(slot.request, {}), unavailable); }
                    }
                    publish_stats();
                    return;
                }
            }
            try {
                admit();
                step();
            } catch (...) {
                // Not a request's own error: the device may be in an unknown state, so the Engine
                // fails with every request it holds.
                const auto error = std::current_exception();
                {
                    std::lock_guard lock(queue_mutex);
                    failed   = true;
                    stopping = true;
                }
                for (auto& slot : slots) {
                    if (slot.request) { complete(std::exchange(slot.request, {}), error); }
                }
                queue_cv.notify_all();
            }
            publish_stats();
        }
    }

    // The queue depth, from a submitting thread.
    void publish_queue() {
        std::lock_guard lock(queue_mutex);
        std::lock_guard stats_lock(stats_mutex);
        stats.waiting_requests = static_cast<std::uint32_t>(pending.size());
    }

    // Every request gauge, from the worker, which owns the slots.
    void publish_stats() {
        std::uint32_t running = 0, prefilling = 0, decoding = 0;
        for (const Slot& slot : slots) {
            if (!slot.request) { continue; }
            ++running;
            ++(slot.request->decoding ? decoding : prefilling);
        }
        std::lock_guard lock(queue_mutex);
        std::lock_guard stats_lock(stats_mutex);
        stats.waiting_requests      = static_cast<std::uint32_t>(pending.size());
        stats.running_requests      = running;
        stats.prefilling_requests   = prefilling;
        stats.decode_ready_requests = decoding;
    }

    // A request failed on its own: it completes with the error, and its sequence keeps nothing
    // for the context cache, since the sequence may have stopped anywhere.
    void fail(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        if (!request_error(error)) { std::rethrow_exception(error); }
        Slot& slot = slots.at(request->slot);
        if (slot.request == request) {
            slot.request = nullptr;
            slot.fed.clear();
            slot.anchor.clear();
        }
        complete(request, std::move(error));
    }

    // Tokens of `held` a prompt can start from: all of them when they are a strict prefix of the
    // prompt (its last token is always fed, since the first sample needs its logits), else none.
    static std::uint32_t prefix_reuse(const std::vector<TokenId>& held,
                                      const std::vector<TokenId>& prompt) {
        if (held.empty() || held.size() >= prompt.size()) { return 0; }
        return std::equal(held.begin(), held.end(), prompt.begin())
                   ? static_cast<std::uint32_t>(held.size())
                   : 0;
    }

    [[nodiscard]] std::uint32_t reuse(const Slot& slot, const std::vector<TokenId>& prompt) const {
        if (!reuse_prefixes) { return 0; }
        return std::max(prefix_reuse(slot.fed, prompt), prefix_reuse(slot.anchor, prompt));
    }

    // Moves queued requests into free sequences, oldest first: each into the free sequence whose
    // cached state serves the most of its prompt, else the least recently used.
    void admit() {
        for (;;) {
            std::optional<std::uint32_t> best;
            std::shared_ptr<Request> request;
            {
                std::lock_guard lock(queue_mutex);
                if (pending.empty()) { return; }
                for (std::uint32_t s = 0; s < slots.size(); ++s) {
                    if (slots[s].request) { continue; }
                    if (!best) {
                        best = s;
                        continue;
                    }
                    const auto& prompt   = pending.front()->prompt_tokens;
                    const auto candidate = reuse(slots[s], prompt), current = reuse(slots[*best], prompt);
                    if (candidate > current ||
                        (candidate == current && slots[s].last_used < slots[*best].last_used)) {
                        best = s;
                    }
                }
                if (!best) { return; }
                request = std::move(pending.front());
                pending.pop_front();
            }
            if (Clock::now() > request->deadline) {
                complete(request, std::make_exception_ptr(
                                      RequestError(RequestErrorKind::QueueTimeout,
                                                   "inference request expired in the queue")));
                continue;
            }
            try {
                begin(request, *best);
            } catch (...) { fail(request, std::current_exception()); }
        }
    }

    void begin(const std::shared_ptr<Request>& request, std::uint32_t s) {
        Request& r          = *request;
        Slot& slot          = slots[s];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        r.slot              = s;
        if (prompt_n == 0) { throw std::invalid_argument("prepared prompt is empty"); }
        if (prompt_n > max_context) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                               "prepared prompt exceeds Engine max_context");
        }
        r.admitted = Clock::now();
        // Resume from the sequence's live state when the prompt continues it, else from the
        // snapshot at the end of its last prompt when the prompt continues that.
        std::uint32_t reused = 0;
        if (reuse_prefixes && r.reusable) {
            const std::uint32_t live     = prefix_reuse(slot.fed, r.prompt_tokens);
            const std::uint32_t anchored = prefix_reuse(slot.anchor, r.prompt_tokens);
            if (anchored > live) {
                executor.restore(s, slot.snapshot);
                slot.fed     = slot.anchor;
                reused       = anchored;
                r.reuse_path = PrefixReusePath::PrivateTurnClosure;
            } else if (live > 0) {
                reused       = live;
                r.reuse_path = PrefixReusePath::PrivateEndpoint;
            }
        }
        if (reused == 0) {
            executor.reset(s);
            slot.fed.clear();
        }
        slot.request   = request;
        slot.last_used = ++use_clock;
        r.reused = r.prefilled = reused;
        r.prefill_start        = Clock::now();
        {
            std::lock_guard lock(stats_mutex);
            stats.reused_prompt_tokens += reused;
        }
        if (r.consumer_mode == OutputConsumerMode::Streaming) {
            {
                std::lock_guard lock(r.mutex);
                r.stream_start =
                    GenerationStart{.prompt = r.prompt_summary, .reused_prompt_tokens = reused};
            }
            r.cv.notify_all();
        }
    }

    void step() {
        std::shared_ptr<Request> prefilling;
        bool decoding = false;
        for (const Slot& slot : slots) {
            if (!slot.request) { continue; }
            if (slot.request->decoding) {
                decoding = true;
            } else if (!prefilling || slot.request->id < prefilling->id) {
                prefilling = slot.request;
            }
        }
        if (prefilling && (prefill_turn || !decoding)) {
            prefill_turn = false;
            try {
                prefill_step(prefilling);
            } catch (...) { fail(prefilling, std::current_exception()); }
        } else if (decoding) {
            prefill_turn = true;
            decode_step();
        }
    }

    // Publishes a committed preview with the logprob records it released, which a streaming
    // request receives on the commit's content delta.
    void push_events(Request& r, models::qwen3_5::PublishedOutput published,
                     std::optional<GenerationTimingObservation> timing) {
        const bool streaming               = r.consumer_mode == OutputConsumerMode::Streaming;
        std::vector<TokenLogprob> logprobs = r.output.take_content_logprobs();
        if (published.empty() && !timing && logprobs.empty()) { return; }
        if (streaming && !logprobs.empty()) {
            OutputDelta* content = nullptr;
            for (OutputDelta& delta : published) {
                if (delta.channel == OutputChannel::Content) { content = &delta; }
            }
            if (content == nullptr) {
                published.push_back(OutputDelta{.channel = OutputChannel::Content});
                content = &published.back();
            }
            content->logprobs = logprobs;
        }
        {
            std::lock_guard lock(r.mutex);
            if (streaming && timing) { r.events.emplace_back(*timing); }
            for (OutputDelta& delta : published) {
                (delta.channel == OutputChannel::Reasoning ? r.reasoning : r.content) += delta.text;
                if (streaming) { r.events.emplace_back(std::move(delta)); }
            }
            r.content_logprobs.insert(r.content_logprobs.end(),
                                      std::make_move_iterator(logprobs.begin()),
                                      std::make_move_iterator(logprobs.end()));
        }
        if (streaming) { r.cv.notify_all(); }
    }

    // Samples one token per row from the first rows.size() columns of the head logits; a grammar
    // restricts its row to the tokens its current state allows. A row that asks for logprobs gets
    // the sampled token's record in `logprobs`, under the distribution its token was drawn from.
    void sample(std::span<const SampleRow> rows, std::int32_t purpose, std::span<TokenId> out,
                std::span<runtime::RawTokenLogprob> logprobs) {
        RankBinding bind(device, instance.executor->head_rank());
        const cudaStream_t stream = instance.executor->head_stream();
        const std::size_t n       = rows.size();
        auto* configs             = static_cast<ops::SamplingConfig*>(host_sample->data());
        auto* positions           = reinterpret_cast<std::int32_t*>(configs + slots.size());
        auto* tokens              = positions + slots.size();
        bool masked               = false;
        for (std::size_t b = 0; b < n; ++b) {
            const SampleRow& row = rows[b];
            ops::SamplingConfig config;
            config.temperature       = row.params->temperature;
            config.top_k             = row.params->top_k;
            config.top_p             = row.params->top_p;
            config.min_p             = row.params->min_p;
            config.presence_penalty  = row.params->presence_penalty;
            config.frequency_penalty = row.params->frequency_penalty;
            config.seed              = row.params->seed;
            config.token_counts =
                row.counts ? static_cast<std::int32_t*>(token_counts.p) + std::size_t(row.slot) * domain
                           : nullptr;
            if (row.grammar != nullptr) {
                auto* words = static_cast<std::uint32_t*>(host_mask->data()) + b * mask_words();
                row.grammar->fill_masks(std::span(words, mask_words()), {});
                config.token_mask =
                    static_cast<const std::uint32_t*>(token_mask.p) + b * mask_words();
                config.token_mask_stride = static_cast<std::int32_t>(mask_words());
                masked                   = true;
            }
            configs[b]   = config;
            positions[b] = static_cast<std::int32_t>(row.position);
        }
        if (masked) {
            CUDA_CHECK(cudaMemcpyAsync(token_mask.p, host_mask->data(),
                                       n * mask_words() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(sample_config.p, configs, n * sizeof(ops::SamplingConfig),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(sample_position.p, positions, n * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        const auto width = static_cast<std::int32_t>(n);
        Tensor sampled(sample_out.p, DType::I32, {width});
        const Tensor logical(sample_position.p, DType::I32, {width});
        const Tensor logits = instance.executor->logits(static_cast<std::uint32_t>(n));
        const auto* device_configs = static_cast<const ops::SamplingConfig*>(sample_config.p);
        const bool gather = std::any_of(rows.begin(), rows.end(),
                                        [](const SampleRow& row) { return row.logprobs; });
        const std::size_t top = n * kMaximumTokenLogprobs;
        auto* host_ids        = static_cast<std::int32_t*>(host_logprobs->data());
        auto* host_values = reinterpret_cast<float*>(host_ids + slots.size() * kMaximumTokenLogprobs);
        if (gather) {
            // Before sampling adds the tokens to the penalty counts.
            Tensor ids(logprob_ids.p, DType::I32, {ops::kLogprobTopK, 1, width});
            Tensor values(logprob_values.p, DType::FP32, {ops::kLogprobTopK, 1, width});
            Tensor lse(logprob_lse.p, DType::FP32, {1, width});
            const Tensor flag(logprob_flag.p, DType::I32, {1});
            ops::logprob_topk(logits.view({logits.ne[0], 1, width}), device_configs, nullptr,
                              static_cast<std::int32_t>(domain), ids, values, lse, flag,
                              *sample_workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(host_ids, logprob_ids.p, top * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_values, logprob_values.p, top * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        {
            auto scope = sample_workspace->scope();
            ops::sample(logits, sampled, static_cast<std::int32_t>(domain), device_configs,
                        logical, purpose, *sample_workspace, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(tokens, sample_out.p, n * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t b = 0; b < n; ++b) {
            const std::int32_t token = tokens[b];
            if (token == ops::kSamplerNonFiniteToken || token < 0 ||
                std::uint32_t(token) >= domain) {
                throw std::runtime_error("Qwen3.8-Flash-Next produced non-finite logits");
            }
            out[b] = token;
            if (!rows[b].logprobs) { continue; }
            runtime::RawTokenLogprob& record = logprobs[b];
            record = runtime::RawTokenLogprob{.id = token, .logprob = kLogprobSentinel};
            for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
                const std::size_t slot = b * kMaximumTokenLogprobs + k;
                record.top_ids[k]      = host_ids[slot];
                record.top_values[k]   = host_values[slot];
                if (host_ids[slot] == token) { record.logprob = host_values[slot]; }
            }
        }
    }

    [[nodiscard]] SampleRow sample_row(Request& r) {
        const auto& exec = r.options.execution;
        if (!r.post_thinking && exec.post_thinking_sampling && r.output.reasoning_closed()) {
            r.post_thinking = true;
        }
        return SampleRow{.params   = r.post_thinking ? &*exec.post_thinking_sampling
                                                     : &exec.sampling,
                         .position = r.position,
                         .slot     = r.slot,
                         .counts   = r.penalties,
                         .logprobs = exec.logprobs,
                         .grammar  = r.output.grammar_state().get()};
    }

    // Frees the request's sequence (keeping its state for the context cache) and completes it.
    void finish(const std::shared_ptr<Request>& request, FinishReason reason) {
        Request& r = *request;
        GenerationResult result;
        result.prompt              = r.prompt_summary;
        result.generated_token_ids = std::move(r.generated);
        {
            std::lock_guard lock(r.mutex);
            result.content          = std::move(r.content);
            result.reasoning        = std::move(r.reasoning);
            result.content_logprobs = std::move(r.content_logprobs);
        }
        const auto now = Clock::now();
        if (r.first_token == Clock::time_point{}) { r.first_token = r.last_token = now; }
        if (r.prefill_end == Clock::time_point{}) { r.prefill_end = now; }
        result.tool_calls                      = r.output.take_tool_calls();
        result.tool_call_parse                 = r.output.tool_call_parse_diagnostics();
        result.reasoning_tokens                = r.output.reasoning_tokens();
        result.finish_reason                   = reason;
        result.matched_stop_string             = r.output.matched_stop_string();
        result.thinking                        = r.output.thinking_stats();
        result.thinking.post_thinking_sampling = r.post_thinking;
        result.reused_prompt_tokens            = r.reused;
        result.prefix_reuse_path               = r.reuse_path;
        result.timings.prepare_seconds         = r.prepare_seconds;
        result.timings.vision_seconds          = r.vision_seconds;
        result.timings.prefill_seconds = seconds(r.prefill_start, r.prefill_end) - r.vision_seconds;
        result.timings.decode_seconds          = seconds(r.prefill_end, r.last_token);
        result.timings.first_token_seconds = r.prepare_seconds + seconds(r.submitted, r.first_token);
        if (r.observation.phase_timings) {
            result.timings.prompt_wall_seconds     = seconds(r.admitted, r.first_token);
            result.timings.generation_wall_seconds = seconds(r.first_token, r.last_token);
        }
        result.timings.total_seconds = r.prepare_seconds + seconds(r.submitted, now);
        Slot& slot                   = slots.at(r.slot);
        slot.request                 = nullptr;
        if (r.media) {
            // Its state depends on media its tokens do not identify.
            slot.fed.clear();
            slot.anchor.clear();
        }
        complete(request, std::move(result));
    }

    void finish_now(const std::shared_ptr<Request>& request, FinishReason reason) {
        (void)request->output.preview_terminal(reason);
        push_events(*request, request->output.commit_preview(), std::nullopt);
        finish(request, reason);
    }

    // Feeds the next chunk of a prompt, which ends at the anchor when one lies ahead, where the
    // sequence's state is kept for the context cache; at the prompt's end samples the first token.
    void prefill_step(const std::shared_ptr<Request>& request) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
            return;
        }
        if (r.media && r.prefilled == 0) {
            // The tower's embeddings stay until another prompt's media replace them; prompts
            // prefill one at a time, so this one is done first.
            const auto& data = models::qwen3_5::PreparedPromptAccess::view(r.prompt);
            const auto& vision = *instance.model->vision_config();
            const auto control = models::qwen3_5::build_vision_control(
                data, models::qwen3_5::plan_vision_control(data, vision), 0);
            std::vector<models::qwen4_exp::MediaItem> items;
            for (std::size_t i = 0; i < control.items.size(); ++i) {
                items.push_back({.patches = data.media_payloads.at(i)->span(),
                                 .control = &control.items[i]});
            }
            const auto vision_start = Clock::now();
            executor.set_media(r.slot, items, data.positions, data.rope_delta);
            r.vision_seconds = seconds(vision_start, Clock::now());
        }
        const std::uint32_t until = r.prefilled < r.anchor_at ? r.anchor_at : prompt_n;
        const std::uint32_t n     = std::min(executor.options().prefill_chunk, until - r.prefilled);
        const auto chunk = std::span<const TokenId>(r.prompt_tokens).subspan(r.prefilled, n);
        executor.forward(r.slot, chunk, 1);
        slot.fed.insert(slot.fed.end(), chunk.begin(), chunk.end());
        r.prefilled += n;
        if (reuse_prefixes && r.reusable && r.prefilled == r.anchor_at) {
            executor.snapshot(r.slot, slot.snapshot);
            slot.anchor = slot.fed;
        }
        if (r.observation.prompt_progress) {
            CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
            {
                std::lock_guard lock(r.mutex);
                r.stream_progress = PromptProgress{.total_prompt_tokens     = prompt_n,
                                                   .reused_prompt_tokens    = r.reused,
                                                   .processed_prompt_tokens = r.prefilled,
                                                   .elapsed_ns = elapsed_ns(r.admitted, Clock::now())};
            }
            r.cv.notify_all();
        }
        if (r.prefilled < prompt_n) { return; }
        CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
        r.prefill_end = Clock::now();
        {
            std::lock_guard lock(stats_mutex);
            stats.computed_prefill_tokens += prompt_n - r.reused;
            stats.prefill_seconds_total += seconds(r.prefill_start, r.prefill_end);
        }
        const auto& exec = r.options.execution;
        const std::uint32_t capacity =
            effective_output_capacity(exec.requested_output_tokens, max_context, prompt_n);
        r.budget.emplace(capacity, exec.requested_output_tokens <= capacity
                                       ? FinishReason::OutputLimit
                                       : FinishReason::ContextCapacity);
        r.penalties = exec.sampling.presence_penalty != 0.0F ||
                      exec.sampling.frequency_penalty != 0.0F ||
                      (exec.post_thinking_sampling &&
                       (exec.post_thinking_sampling->presence_penalty != 0.0F ||
                        exec.post_thinking_sampling->frequency_penalty != 0.0F));
        if (r.penalties) {
            RankBinding bind(device, executor.head_rank());
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts.p) +
                                           std::size_t(r.slot) * domain,
                                       0, std::size_t(domain) * sizeof(std::int32_t),
                                       executor.head_stream()));
        }
        r.position         = prompt_n;
        r.decoding         = true;
        const SampleRow row = sample_row(r);
        TokenId token       = 0;
        runtime::RawTokenLogprob logprob;
        sample(std::span(&row, 1), ops::kSamplePurposePrefill, std::span(&token, 1),
               std::span(&logprob, 1));
        accept(request, token, row.logprobs ? &logprob : nullptr);
    }

    // Applies the output policy to a sampled token: publishes what it accepts, then either
    // finishes the request or queues what its next decode step feeds. `logprob` is the token's
    // record when the request asked for logprobs.
    void accept(const std::shared_ptr<Request>& request, TokenId token,
                const runtime::RawTokenLogprob* logprob) {
        Request& r                    = *request;
        const OutputDecision decision = r.output.preview_model(
            std::span<const TokenId>(&token, 1), r.budget->remaining(), r.budget->limit_reason(),
            logprob != nullptr ? std::span<const runtime::RawTokenLogprob>(logprob, 1)
                               : std::span<const runtime::RawTokenLogprob>{});
        const auto now = Clock::now();
        if (r.first_token == Clock::time_point{}) { r.first_token = now; }
        r.last_token = now;
        if (decision.accepted_tokens > 1) {
            throw std::logic_error("output policy accepted more than the sampled token");
        }
        if (decision.accepted_tokens == 1) {
            r.generated.push_back(token);
            r.budget->commit(1);
        }
        std::optional<GenerationTimingObservation> timing;
        if (r.observation.live_timings) {
            timing = GenerationTimingObservation{
                .generated_tokens      = static_cast<std::uint32_t>(r.generated.size()),
                .prompt_elapsed_ns     = elapsed_ns(r.admitted, r.first_token),
                .generation_elapsed_ns = elapsed_ns(r.first_token, now)};
        }
        push_events(r, r.output.commit_preview(), timing);
        {
            std::lock_guard lock(stats_mutex);
            stats.committed_decode_tokens += decision.accepted_tokens;
        }
        if (decision.finished()) {
            finish(request, decision.finish_reason);
            return;
        }
        r.feed.assign(1, token);
        if (decision.continuation == ContinuationAction::ApplyTargetControl) {
            const auto pending_control = r.output.pending_control_tokens();
            const std::vector<TokenId> control(pending_control.begin(), pending_control.end());
            const OutputDecision forced = r.output.preview_control(control, r.budget->remaining());
            if (forced.accepted_tokens != control.size() || forced.finished()) {
                throw std::logic_error("thinking control preview returned an invalid decision");
            }
            r.generated.insert(r.generated.end(), control.begin(), control.end());
            r.budget->commit(static_cast<std::uint32_t>(control.size()));
            push_events(r, r.output.commit_preview(), std::nullopt);
            r.feed.insert(r.feed.end(), control.begin(), control.end());
        }
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
            return;
        }
        if (r.position + r.feed.size() > max_context) {
            finish_now(request, FinishReason::ContextCapacity);
        }
    }

    // One step of every decoding request: those that feed one token run as one batch, whose
    // experts read their weights once; one feeding a thinking-control suffix runs alone.
    void decode_step() {
        auto& executor = *instance.executor;
        std::vector<std::shared_ptr<Request>> batch, alone;
        for (const Slot& slot : slots) {
            if (!slot.request || !slot.request->decoding) { continue; }
            (slot.request->feed.size() == 1 ? batch : alone).push_back(slot.request);
        }
        const auto start = Clock::now();
        for (const auto& request : alone) {
            Request& r = *request;
            executor.forward(r.slot, r.feed, 1);
            Slot& slot = slots[r.slot];
            slot.fed.insert(slot.fed.end(), r.feed.begin(), r.feed.end());
            r.position += static_cast<std::uint32_t>(r.feed.size());
            const SampleRow row = sample_row(r);
            TokenId token       = 0;
            runtime::RawTokenLogprob logprob;
            sample(std::span(&row, 1), ops::kSamplePurposeDecode, std::span(&token, 1),
                   std::span(&logprob, 1));
            try {
                accept(request, token, row.logprobs ? &logprob : nullptr);
            } catch (...) { fail(request, std::current_exception()); }
        }
        if (!batch.empty()) {
            std::vector<std::uint32_t> sequences;
            std::vector<TokenId> tokens;
            for (const auto& request : batch) {
                sequences.push_back(request->slot);
                tokens.push_back(request->feed.front());
            }
            executor.decode(sequences, tokens);
            std::vector<SampleRow> rows;
            for (const auto& request : batch) {
                slots[request->slot].fed.push_back(request->feed.front());
                request->position += 1;
                rows.push_back(sample_row(*request));
            }
            std::vector<TokenId> sampled(batch.size());
            std::vector<runtime::RawTokenLogprob> logprobs(batch.size());
            sample(rows, ops::kSamplePurposeDecode, sampled, logprobs);
            for (std::size_t b = 0; b < batch.size(); ++b) {
                try {
                    accept(batch[b], sampled[b], rows[b].logprobs ? &logprobs[b] : nullptr);
                } catch (...) { fail(batch[b], std::current_exception()); }
            }
        }
        std::lock_guard lock(stats_mutex);
        stats.decode_rounds += 1;
        stats.decode_row_rounds += batch.size() + alone.size();
        stats.decode_seconds_total += seconds(start, Clock::now());
    }

    std::vector<float> score(const std::vector<TokenId>& tokens, std::uint32_t first_target) {
        auto& executor        = *instance.executor;
        const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
        if (n < 2 || first_target == 0 || first_target >= n || n > max_context) {
            throw std::invalid_argument("score: invalid token window");
        }
        // Logits for position p predict token p + 1; the executor keeps at most 512 logit rows.
        const std::uint32_t chunk = std::min<std::uint32_t>(executor.options().prefill_chunk, 512);
        executor.reset(0);
        slots[0].fed.clear();
        slots[0].anchor.clear();
        std::vector<float> out;
        out.reserve(n - first_target);
        RankBinding bind(device, executor.head_rank());
        for (std::uint32_t at = 0; at + 1 < n; at += chunk) {
            const std::uint32_t count = std::min(chunk, n - 1 - at);
            executor.forward(0, std::span<const TokenId>(tokens).subspan(at, count), count);
            // Rows predicting targets [at + 1, at + count].
            const std::uint32_t first_row = first_target > at + 1 ? first_target - at - 1 : 0;
            if (first_row >= count) { continue; }
            const std::uint32_t rows = count - first_row;
            std::vector<std::int32_t> targets(rows);
            for (std::uint32_t i = 0; i < rows; ++i) {
                targets[i] = tokens[at + 1 + first_row + i];
            }
            const cudaStream_t stream = executor.head_stream();
            CUDA_CHECK(cudaMemcpyAsync(score_targets.p, targets.data(), rows * 4,
                                       cudaMemcpyHostToDevice, stream));
            const Tensor all = executor.logits(count);
            const Tensor logits(static_cast<__nv_bfloat16*>(all.data) +
                                    std::size_t(first_row) * all.ne[0],
                                DType::BF16, {all.ne[0], static_cast<std::int32_t>(rows)});
            const Tensor target_ids(score_targets.p, DType::I32, {static_cast<std::int32_t>(rows)});
            Tensor result(score_out.p, DType::FP32, {static_cast<std::int32_t>(rows)});
            ops::target_logprobs(logits, target_ids, static_cast<std::int32_t>(domain), result,
                                 stream);
            std::vector<float> host(rows);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), score_out.p, rows * 4, cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            out.insert(out.end(), host.begin(), host.end());
        }
        return out;
    }
};

Qwen4ExpCore::Submission::Submission(Qwen4ExpCore& owner, std::shared_ptr<Request> request,
                                     std::optional<std::uint32_t> budget) noexcept
    : owner_(&owner), request_(std::move(request)), effective_thinking_budget_(budget) {}

Qwen4ExpCore::Submission::~Submission() {
    if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
}

Qwen4ExpCore::Submission::Submission(Submission&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)),
      effective_thinking_budget_(other.effective_thinking_budget_) {}

Qwen4ExpCore::Submission& Qwen4ExpCore::Submission::operator=(Submission&& other) noexcept {
    if (this != &other) {
        if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
        owner_                     = std::exchange(other.owner_, nullptr);
        request_                   = std::move(other.request_);
        effective_thinking_budget_ = other.effective_thinking_budget_;
    }
    return *this;
}

GenerationResult Qwen4ExpCore::Submission::wait(OutputSink* sink,
                                                const CancellationView& cancellation) {
    if (request_ == nullptr) { throw std::logic_error("submission is empty"); }
    const std::shared_ptr<Request> request = std::move(request_);
    const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
    if (streaming != (sink != nullptr)) {
        request->cancelled.store(true, std::memory_order_release);
        throw std::invalid_argument(
            "GenerationHandle wait sink does not match its submitted consumer mode");
    }
    std::exception_ptr caller_error;
    for (;;) {
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<std::variant<OutputDelta, GenerationTimingObservation>> events;
        bool done = false;
        {
            std::unique_lock lock(request->mutex);
            request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                return request->response_done || request->stream_start.has_value() ||
                       request->stream_progress.has_value() || !request->events.empty();
            });
            start    = std::exchange(request->stream_start, std::nullopt);
            progress = std::exchange(request->stream_progress, std::nullopt);
            events.swap(request->events);
            done = request->response_done;
        }
        if (caller_error == nullptr && sink != nullptr) {
            try {
                if (start) { sink->start(std::move(*start)); }
                if (progress) { sink->progress(std::move(*progress)); }
                for (auto& event : events) {
                    if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                        sink->timing(*timing);
                    } else {
                        sink->publish(std::move(std::get<OutputDelta>(event)));
                    }
                }
            } catch (...) {
                caller_error = std::current_exception();
                request->cancelled.store(true, std::memory_order_release);
            }
        }
        if (caller_error == nullptr) {
            try {
                if (cancellation.requested()) {
                    request->cancelled.store(true, std::memory_order_release);
                }
            } catch (...) {
                caller_error = std::current_exception();
                request->cancelled.store(true, std::memory_order_release);
            }
        }
        if (!done) { continue; }
        if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
        std::lock_guard lock(request->mutex);
        if (request->error != nullptr) { std::rethrow_exception(request->error); }
        return std::move(request->result);
    }
}

Qwen4ExpCore::Qwen4ExpCore(Qwen4ExpInstance& instance, DeviceContext& device,
                           const EngineOptions& options)
    : impl_(std::make_unique<Impl>(instance, device, options)) {
    if (options.max_concurrency == 0 || options.max_pending_requests == 0 ||
        options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine core bounds are invalid");
    }
}

Qwen4ExpCore::~Qwen4ExpCore() = default;

void Qwen4ExpCore::stop() noexcept {
    {
        std::lock_guard lock(impl_->queue_mutex);
        impl_->stopping = true;
    }
    impl_->queue_cv.notify_all();
}

Qwen4ExpCore::Submission Qwen4ExpCore::submit(models::qwen3_5::PreparedPrompt prompt,
                                              PromptSummary prompt_summary, double prepare_seconds,
                                              ResolvedRequestOptions options,
                                              OutputConsumerMode consumer_mode,
                                              GenerationObservationOptions observation,
                                              Clock::time_point pending_deadline) {
    const auto submitted = Clock::now();
    if (options.execution.structured_output.kind != StructuredOutputKind::None &&
        !impl_->structured_output) {
        throw std::invalid_argument(
            "structured output requires an Engine started with structured_output");
    }
    if (pending_deadline == Clock::time_point{}) {
        pending_deadline = submitted + impl_->pending_timeout;
    }
    auto request = std::make_shared<Request>();
    request->prompt_tokens =
        std::vector<TokenId>(prompt.token_ids().begin(), prompt.token_ids().end());
    {
        const auto& data     = models::qwen3_5::PreparedPromptAccess::view(prompt);
        const auto& identity = data.identity;
        const auto prompt_n  = static_cast<std::uint32_t>(request->prompt_tokens.size());
        request->media       = data.has_media();
        if (request->media && !impl_->instance.executor->vision()) {
            throw std::invalid_argument("media need an Engine started with Vision");
        }
        // A prompt's media are not part of its tokens, so a media prompt is never reused.
        request->reusable = identity.reusable && !request->media;
        request->anchor_at   = prompt_n;
        if (identity.rewrite_checkpoint &&
            identity.rewrite_checkpoint->kind ==
                models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
            identity.rewrite_checkpoint->frontier > 0 &&
            identity.rewrite_checkpoint->frontier < prompt_n) {
            request->anchor_at = identity.rewrite_checkpoint->frontier;
        }
    }
    request->prompt_summary  = prompt_summary;
    request->prepare_seconds = prepare_seconds;
    request->consumer_mode   = consumer_mode;
    request->observation     = observation;
    request->deadline        = pending_deadline;
    request->submitted       = submitted;
    apply_effective_thinking_budget(
        options.execution.thinking,
        effective_output_capacity(options.execution.requested_output_tokens, impl_->max_context,
                                  prompt_summary.prompt_tokens),
        impl_->instance.frontend.thinking_control_token_count());
    request->output = impl_->instance.frontend.make_output_session(
        prompt, options.stop, options.output, options.execution.thinking,
        options.execution.structured_output, options.tool_choice);
    const std::optional<std::uint32_t> budget = request->output.thinking_stats().effective_budget;
    request->prompt                           = std::move(prompt);
    request->options                          = std::move(options);
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding >= impl_->max_outstanding) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        ++impl_->outstanding;
        request->id = impl_->next_id++;
        impl_->pending.push_back(request);
    }
    impl_->queue_cv.notify_one();
    impl_->publish_queue();
    return Submission(*this, std::move(request), budget);
}

std::vector<float> Qwen4ExpCore::score(models::qwen3_5::PreparedPrompt prompt,
                                       std::uint32_t first_target) {
    const std::vector<TokenId> tokens(prompt.token_ids().begin(), prompt.token_ids().end());
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding != 0) {
            throw std::logic_error("score requires an idle Qwen3.8-Flash-Next Engine");
        }
        ++impl_->outstanding;
    }

    struct Release {
        Impl& impl;

        ~Release() {
            std::lock_guard lock(impl.queue_mutex);
            --impl.outstanding;
        }
    } release{*impl_};

    impl_->device.bind_to_current_thread();
    return impl_->score(tokens, first_target);
}

MemorySummary Qwen4ExpCore::memory_summary() const {
    const auto& stats  = impl_->instance.model->storage_stats();
    const auto memory  = impl_->instance.executor->memory();
    const auto arena   = [](std::uint64_t bytes) {
        return ArenaMemorySummary{bytes, bytes, bytes};
    };
    std::vector<DeviceMemorySummary> devices;
    for (std::size_t r = 0; r < impl_->device.size(); ++r) {
        const auto& rank = memory.ranks.at(r);
        devices.push_back({.device    = impl_->device.rank(r).device,
                           .weights   = arena(r < stats.device_capacity_by_rank.size()
                                                  ? stats.device_capacity_by_rank[r]
                                                  : 0),
                           .sequence  = arena(rank.state_bytes),
                           .workspace = arena(rank.workspace_bytes),
                           .expert_cache_bytes = rank.expert_cache_bytes});
    }
    MemorySummary out;
    out.device                    = devices.front().device;
    out.max_context               = impl_->max_context;
    out.kv_capacity               = impl_->max_context;
    out.kv_cache                      = impl_->instance.executor->options().kv_cache;
    out.weights                   = devices.front().weights;
    out.sequence                  = devices.front().sequence;
    out.workspace                 = devices.front().workspace;
    out.expert_cache_bytes        = memory.expert_cache_bytes;
    out.kv_payload_bytes          = memory.kv_bytes;
    out.runtime_reservation_bytes = memory.state_bytes + memory.workspace_bytes;
    out.available_after_weights_bytes = impl_->instance.free_after_weights;
    out.available_after_startup_bytes = impl_->instance.free_after_startup;
    if (devices.size() > 1) { out.devices = std::move(devices); }
    return out;
}

RuntimeStats Qwen4ExpCore::runtime_stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->stats;
}

bool Qwen4ExpCore::is_available() const {
    std::lock_guard lock(impl_->queue_mutex);
    return !impl_->stopping && !impl_->failed;
}

bool Qwen4ExpCore::has_failed() const {
    std::lock_guard lock(impl_->queue_mutex);
    return impl_->failed;
}

} // namespace ninfer::runtime
