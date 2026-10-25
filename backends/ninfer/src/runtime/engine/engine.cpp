#include "ninfer/engine.h"
#include "text/structured_output.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/context_cache/hybrid_resource_manager.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/qwen4_exp_core.h"
#include "runtime/engine/slot_spill_guard.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    if (options.devices.empty()) {
        DeviceContext device(options.device);
        phase.complete();
        return device;
    }
    DeviceContext device{std::span<const int>(options.devices)};
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    text::validate_structured_output(options.execution.structured_output);
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    if (mode == SamplingMode::Thinking && options.execution.post_thinking_sampling) {
        SamplingOverrides post_thinking = *options.execution.post_thinking_sampling;
        if (!post_thinking.seed) { post_thinking.seed = resolved.execution.sampling.seed; }
        resolved.execution.post_thinking_sampling =
            runtime::resolve_sampling(defaults.post_thinking, post_thinking);
    }
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.execution.structured_output       = std::move(options.execution.structured_output);
    resolved.execution.logprobs                = options.execution.logprobs;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    resolved.tool_choice                       = std::move(options.tool_choice);
    resolved.ngram_session                      = std::move(options.ngram_session);
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

CudaResidencySummary residency_summary(const runtime::ModelInstance& instance) {
    if (!instance.residency) { return {}; }
    // stats() reads the session's cached snapshot. HTTP threads never enter CUDA or PDH.
    const auto stats = instance.residency->stats();
    return {.enabled = true,
            .verified = stats.verified,
            .cuda_free_bytes = stats.cuda_free_bytes,
            .cuda_total_bytes = stats.cuda_total_bytes,
            .dedicated_bytes = stats.dedicated_bytes,
            .shared_bytes = stats.shared_bytes,
            .shared_baseline_bytes = stats.shared_baseline_bytes,
            .device_allocated_bytes = stats.device_allocated_bytes,
            .host_pool_bytes = stats.host_pool_bytes,
            .host_used_bytes = stats.host_used_bytes,
            .verified_reserve_bytes = stats.verified_reserve_bytes};
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         models::qwen3_5::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    models::qwen3_5::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

std::span<const TokenId> PreparedPrompt::token_ids() const noexcept {
    return impl_ != nullptr ? impl_->value.token_ids() : std::span<const TokenId>{};
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
        virtual std::optional<std::uint32_t> effective_thinking_budget() const noexcept       = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

        std::optional<std::uint32_t> effective_thinking_budget() const noexcept override {
            return submission_.effective_thinking_budget();
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

    [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
        return state_->effective_thinking_budget();
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

std::optional<std::uint32_t> GenerationHandle::effective_thinking_budget() const noexcept {
    return impl_ != nullptr ? impl_->effective_thinking_budget() : std::nullopt;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

namespace {

// What a session snapshot binds to: the model, its weight formats and quantization, and the
// artifact size, since the prefill signature describes weight geometry but not weight values.
std::string slot_model_binding(const EngineOptions& options, const LoadSummary& load) {
    std::string binding = load.architecture + '\n' + load.model_name + '\n';
    for (const std::string& format : load.weight_formats) { binding += format + ','; }
    binding += '\n' + load.prefill_signature + '\n';
    std::error_code size_error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, size_error);
    binding += size_error ? std::string("?") : std::to_string(size);
    return binding;
}

// Write-then-rename, so a torn write never shadows a good snapshot at `path`. The staging name
// embeds the thread id so concurrent writers of one path never share a temporary file.
void write_snapshot_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ostringstream staging_name;
    staging_name << path << ".tmp." << std::this_thread::get_id();
    const std::string staging = staging_name.str();
    {
        std::ofstream file(staging, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        if (!file.good()) {
            file.close();
            (void)std::remove(staging.c_str());
            throw std::invalid_argument("failed to write session snapshot file");
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(staging, path, rename_error);
    if (rename_error) {
        (void)std::remove(staging.c_str());
        throw std::invalid_argument("failed to publish session snapshot file: " +
                                    rename_error.message());
    }
}

// Auto-save spills queue at most this many snapshots. Each holds a whole session, several GB for
// a deep one; beyond the bound a spill is dropped and reported rather than growing host memory.
constexpr std::size_t kMaximumPendingSlotWrites = 2;

} // namespace

class Engine::Impl {
public:
    using GenerationCore = runtime::EngineCore<runtime::ModelInstance>;
    using HybridGenerationCore =
        runtime::EngineCore<runtime::ModelInstance,
                            runtime::HybridResourceManager<runtime::ModelInstance::ModelContract>>;
    using ScoringCore = runtime::CausalScoreCore<runtime::ModelInstance>;
    // Qwen3.8-Flash-Next: its own model family and core, which serves generation and scoring.
    using Qwen4ExpCore = runtime::Qwen4ExpCore;
    using Core = std::variant<std::monostate, std::unique_ptr<GenerationCore>,
                              std::unique_ptr<HybridGenerationCore>, std::unique_ptr<ScoringCore>,
                              std::unique_ptr<Qwen4ExpCore>>;

    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        if (runtime::is_qwen4_exp_artifact(options.artifact_path)) {
            if (options.kvmem.selected_tokens != 0) {
                throw std::invalid_argument("KVMem is not supported for qwen4_exp");
            }
            if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault) {
                throw std::invalid_argument("mixed/strict CUDA memory is not supported for qwen4_exp");
            }
            auto constructed    = runtime::construct_qwen4_exp(options, device);
            flash               = std::move(constructed.instance);
            load                = std::move(constructed.load);
            load.cuda_sync_mode = device.sync_mode();
            model_metadata      = std::move(constructed.model_metadata);
            sampling_defaults   = flash->frontend.sampling_defaults();
            StartupPhaseScope finalize_phase(options.startup_observer,
                                             StartupPhase::EngineFinalize);
            core = std::make_unique<Qwen4ExpCore>(*flash, device, options);
            finalize_phase.complete();
            return;
        }
        auto constructed  = runtime::construct_model(options, device);
        // construct_model returns the resolved options for this instance. Anything the model had
        // to derive (the single host RAM budget's Host split and long-anchor count) is only known
        // after planning, so the Engine adopts the resolved copy here — before the core that
        // sizes its admission capacity from it exists — and reports it through options().
        options             = std::move(constructed.options);
        active            = std::move(constructed.instance);
        load              = std::move(constructed.load);
        load.cuda_sync_mode = device.sync_mode();
        model_metadata    = std::move(constructed.model_metadata);
        sampling_defaults = active->frontend.sampling_defaults();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core = std::make_unique<ScoringCore>(*active, device);
        } else if (options.context_cache.enabled &&
                   options.context_cache.mode == ContextCacheMode::Hybrid) {
            core = std::make_unique<HybridGenerationCore>(*active, device, options,
                                                          std::move(constructed.context_cost));
        } else {
            auto generation = std::make_unique<GenerationCore>(
                *active, device, options, std::move(constructed.context_cost));
            if (options.slot_auto_save.enabled) {
                generation->set_eviction_sink(
                    slot_model_binding(options, load),
                    [this](std::string path, runtime::ModelInstance::ModelContract::SessionSnapshot&& snapshot) {
                        enqueue_write(std::move(path), std::move(snapshot));
                    });
            }
            core = std::move(generation);
        }
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core::ResidentMemoryBinding resident_binding(active ? active->residency : nullptr);
        const bool persists = persists_prefix_cache();
        stop();
        // Joins the generation core's orderly stop, which saves the Host tier before dropping it.
        core.emplace<std::monostate>();
        stop_writer();
        try {
            device.synchronize();
        } catch (...) {}
        if (persists) { report_prefix_cache_save(); }
        active.reset();
    }

    void stop() noexcept {
        if (stop_requested.exchange(true)) { return; }
        if (persists_prefix_cache()) {
            runtime::publish_diagnostic(
                options.diagnostic_observer, DiagnosticLevel::Info, "saving the prefix cache to %s",
                options.context_cache.hybrid.persistent_file.string().c_str());
        }
        std::visit(
            [](auto& state) {
                using CoreState = std::remove_cvref_t<decltype(state)>;
                if constexpr (!std::is_same_v<CoreState, std::monostate>) { state->stop(); }
            },
            core);
    }

    [[nodiscard]] bool persists_prefix_cache() const noexcept {
        return std::holds_alternative<std::unique_ptr<HybridGenerationCore>>(core) &&
               !options.context_cache.hybrid.persistent_file.empty();
    }

    void report_prefix_cache_save() const noexcept {
        try {
            const std::optional<models::qwen3_5::HybridCachePersistence> result =
                active->program->hybrid_shutdown_save();
            if (!result) {
                runtime::publish_diagnostic(
                    options.diagnostic_observer, DiagnosticLevel::Warning,
                    "prefix cache not saved: the Engine did not stop cleanly");
                return;
            }
            const models::qwen3_5::HybridCachePersistence& saved = *result;
            if (saved.ok) {
                runtime::publish_diagnostic(
                    options.diagnostic_observer, DiagnosticLevel::Info,
                    "prefix cache saved: %llu blocks, %llu snapshots, %.1f MiB in %.1f s",
                    static_cast<unsigned long long>(saved.blocks),
                    static_cast<unsigned long long>(saved.snapshots),
                    static_cast<double>(saved.bytes) / 1048576.0, saved.seconds);
            } else {
                runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                            "prefix cache not saved: %s", saved.message.c_str());
            }
        } catch (const std::exception& error) {
            runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                        "prefix cache not saved: %s", error.what());
        } catch (...) {}
    }

    // File I/O on slot paths is serialized by slot_io_mutex: the writer holds it for each spill,
    // from popping the item to publishing the file, and every explicit save, restore and erase
    // holds it for its whole operation. An explicit operation first claims its path, which
    // advances the path's generation; a spill queued under an older generation is then skipped as
    // superseded, since the client has just declared the file's content. Spills queued after the
    // claim are newer states of the session and are written after the explicit operation.

    // Writes the spills still queued for `path`, so a restore reads the newest saved state instead
    // of the file a pending spill was about to replace. Called with slot_io_mutex held; the writer
    // cannot be part-way through one of them, because it pops items only under that mutex.
    void flush_pending_for_path(const std::string& path) {
        std::deque<PendingWrite> matching;
        {
            std::scoped_lock lock(writer_mutex);
            for (auto it = pending_writes.begin(); it != pending_writes.end();) {
                if (it->path == path) {
                    matching.push_back(std::move(*it));
                    it = pending_writes.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (PendingWrite& item : matching) { write_spill(item); }
    }

    // Session slots are private catalog cells, which only the Legacy context cache has.
    [[nodiscard]] GenerationCore& generation_core() {
        auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
        if (generation == nullptr || *generation == nullptr) {
            throw std::invalid_argument(
                "session slots require a generation Engine with the legacy context cache");
        }
        return **generation;
    }

    [[nodiscard]] const GenerationCore* legacy_generation_core() const noexcept {
        const auto* generation = std::get_if<std::unique_ptr<GenerationCore>>(&core);
        return generation == nullptr ? nullptr : generation->get();
    }

    // The model's frontend and prompt capacity, whichever family it is.
    [[nodiscard]] const models::qwen3_5::Frontend& frontend() const {
        return flash ? flash->frontend : active->frontend;
    }

    [[nodiscard]] std::uint32_t capacity() const {
        return flash ? flash->capacity : active->capacity;
    }

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    std::unique_ptr<runtime::Qwen4ExpInstance> flash;
    LoadSummary load;
    ModelMetadata model_metadata;
    ModelSamplingDefaults sampling_defaults;
    Core core;
    std::atomic<bool> stop_requested{false};
    SlotSpillGuard spill_guard;
    std::mutex slot_io_mutex;

private:
    struct PendingWrite {
        std::string path;
        runtime::ModelInstance::ModelContract::SessionSnapshot snapshot;
        // The path's generation when the spill was queued; a later explicit claim supersedes it.
        std::uint64_t generation = 0;
    };

    void enqueue_write(std::string path, runtime::ModelInstance::ModelContract::SessionSnapshot&& snapshot) {
        std::unique_lock lock(writer_mutex);
        if (pending_writes.size() >= kMaximumPendingSlotWrites) {
            SlotAutoSaveEvent event;
            event.path   = std::move(path);
            event.tokens = snapshot.tokens;
            event.bytes  = snapshot.bytes.size();
            event.error  = "auto-save queue is full; the evicted session was not saved";
            lock.unlock();
            notify(event);
            return;
        }
        if (!writer.joinable()) { writer = std::thread([this] { writer_loop(); }); }
        const std::uint64_t generation = spill_guard.generation(path);
        pending_writes.push_back(PendingWrite{std::move(path), std::move(snapshot), generation});
        lock.unlock();
        writer_cv.notify_one();
    }

    void notify(const SlotAutoSaveEvent& event) const noexcept {
        if (!options.slot_auto_save.listener) { return; }
        try {
            options.slot_auto_save.listener(event);
        } catch (...) {}
    }

    // One spill, with slot_io_mutex held: skipped when an explicit operation has claimed the path
    // since it was queued, or when the file already holds a deeper state of the session.
    void write_spill(PendingWrite& item) {
        SlotAutoSaveEvent event;
        event.path         = item.path;
        event.tokens       = item.snapshot.tokens;
        event.bytes        = item.snapshot.bytes.size();
        const auto started = std::chrono::steady_clock::now();
        try {
            if (spill_guard.generation(item.path) != item.generation) {
                event.superseded = true;
            } else if (const std::optional<std::uint32_t> deeper =
                           spill_guard.blocks(item.path, item.snapshot.tokens)) {
                event.skipped_behind_tokens = deeper;
            } else {
                write_snapshot_file(item.path, item.snapshot.bytes);
                spill_guard.note_spilled(item.path, item.snapshot.tokens);
            }
        } catch (const std::exception& error) {
            event.error = error.what();
        } catch (...) { event.error = "unknown auto-save failure"; }
        event.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        notify(event);
    }

    void writer_loop() {
        for (;;) {
            {
                std::unique_lock lock(writer_mutex);
                writer_cv.wait(lock, [this] { return writer_stop || !pending_writes.empty(); });
                if (pending_writes.empty()) { return; }
            }
            // Take slot_io_mutex before popping, so an explicit operation holding it sees every
            // spill still queued and none half-taken.
            std::scoped_lock io(slot_io_mutex);
            std::optional<PendingWrite> item;
            {
                std::scoped_lock lock(writer_mutex);
                if (pending_writes.empty()) { continue; }
                item.emplace(std::move(pending_writes.front()));
                pending_writes.pop_front();
            }
            write_spill(*item);
        }
    }

    // Pending spills are flushed before the thread exits.
    void stop_writer() noexcept {
        {
            std::scoped_lock lock(writer_mutex);
            writer_stop = true;
        }
        writer_cv.notify_all();
        if (writer.joinable()) {
            try {
                writer.join();
            } catch (...) {}
        }
    }

    std::mutex writer_mutex;
    std::condition_variable writer_cv;
    std::deque<PendingWrite> pending_writes;
    bool writer_stop = false;
    std::thread writer;
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared      = impl_->frontend().prepare(std::move(input), control);
    PromptSummary info = prepared.summary();
    const SamplingMode sampling_mode =
        info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (info.prompt_tokens > impl_->capacity()) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(info, preparation, sampling_mode,
                                                                 std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids, bool allow_prefix_identity,
                                      bool anchor_prompt_end) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->capacity()) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), impl_->capacity()));
    }
    auto prepared = impl_->frontend().prepare_tokens(std::move(token_ids), allow_prefix_identity,
                                                     anchor_prompt_end);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > impl_->capacity()) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, preparation, SamplingMode::Thinking, std::move(prepared)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text, bool parse_special) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().tokenize_text(text, parse_special);
}

std::string Engine::token_bytes(TokenId token) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().token_bytes(token);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    PreparedPrompt prompt      = prepare_tokens(std::move(tokens), false);
    const std::size_t expected = prompt.summary().prompt_tokens - first_target;
    std::vector<float> result  = std::visit(
        [&](auto& core) -> std::vector<float> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::Qwen4ExpCore>>) {
                return core->score(std::move(prompt.impl_->value), first_target);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.size() != expected) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }

            [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
                return result.thinking.effective_budget;
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                    = prompt_summary;
        immediate.result.finish_reason             = FinishReason::OutputLimit;
        immediate.result.thinking.requested_budget = resolved_options.execution.thinking.budget;
        // No output is licensed, so the cap never binds: effective equals requested.
        immediate.result.thinking.effective_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission = core->submit(std::move(prompt.impl_->value), prompt_summary,
                                               prepare_seconds, std::move(resolved_options),
                                               consumer_mode, observation, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

ModelMetadata Engine::model_metadata() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->model_metadata;
}

std::uint32_t Engine::concurrent_output_budget(const PreparedPrompt& prompt) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("concurrent_output_budget requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (impl_->flash) {
        // One sequence at a time: the whole window past the prompt.
        const std::uint32_t prompt_tokens = prompt.impl_->summary.prompt_tokens;
        return prompt_tokens >= impl_->capacity() ? 0U : impl_->capacity() - prompt_tokens + 1U;
    }
    return impl_->active->program->concurrent_output_budget(prompt.impl_->summary.prompt_tokens);
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    MemorySummary summary = std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
    if (impl_->active) { summary.cuda_residency = residency_summary(*impl_->active); }
    return summary;
}

SlotSaveResult Engine::save_slot(std::uint32_t slot, const std::string& path,
                                 const std::string& expected_digest) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto started = std::chrono::steady_clock::now();
    std::scoped_lock io(impl_->slot_io_mutex);
    auto snapshot = impl_->generation_core().save_slot(
        slot, slot_model_binding(impl_->options, impl_->load), expected_digest, path,
        [&] { impl_->spill_guard.claim(path); });
    write_snapshot_file(path, snapshot.bytes);
    impl_->spill_guard.note_authoritative(path, snapshot.tokens);

    SlotSaveResult result;
    result.tokens         = snapshot.tokens;
    result.bytes          = snapshot.bytes.size();
    result.session_digest = std::move(snapshot.session_digest);
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

SlotRestoreResult Engine::restore_slot(std::uint32_t slot, const std::string& path) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const auto started = std::chrono::steady_clock::now();
    std::scoped_lock io(impl_->slot_io_mutex);
    impl_->flush_pending_for_path(path);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { throw std::invalid_argument("session snapshot file is unavailable"); }
    const std::streamsize size = file.tellg();
    if (size <= 0) { throw std::invalid_argument("session snapshot file is empty"); }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!file.good()) { throw std::invalid_argument("failed to read session snapshot file"); }
    file.close();

    auto [tokens, digest] = impl_->generation_core().restore_slot(
        slot, std::span<const std::uint8_t>(bytes.data(), bytes.size()),
        slot_model_binding(impl_->options, impl_->load), path,
        [&] { impl_->spill_guard.claim(path); });
    impl_->spill_guard.note_authoritative(path, tokens);

    SlotRestoreResult result;
    result.tokens         = tokens;
    result.bytes          = bytes.size();
    result.session_digest = std::move(digest);
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

std::uint32_t Engine::erase_slot(std::uint32_t slot, const std::string& expected_digest) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    std::scoped_lock io(impl_->slot_io_mutex);
    return impl_->generation_core().erase_slot(
        slot, expected_digest,
        [&](const std::string& bound_path) { impl_->spill_guard.claim(bound_path); });
}

std::vector<SlotState> Engine::slot_states() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    // A Hybrid context cache or a scoring Engine has no catalog cells to list.
    const auto* generation = impl_->legacy_generation_core();
    return generation != nullptr ? generation->slot_states() : std::vector<SlotState>{};
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().media_cache_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    RuntimeStats summary = std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
    if (impl_->active) { summary.cuda_residency = residency_summary(*impl_->active); }
    return summary;
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

bool Engine::has_failed() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->has_failed();
            }
        },
        impl_->core);
}

ResidencyStatus Engine::suspend(std::optional<bool> auto_resume) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](auto& core) -> ResidencyStatus {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::GenerationCore>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::HybridGenerationCore>>) {
                return core->suspend(auto_resume);
            } else {
                throw std::invalid_argument("model suspend applies to Generation Engines");
            }
        },
        impl_->core);
}

ResidencyStatus Engine::resume() {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](auto& core) -> ResidencyStatus {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::GenerationCore>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::HybridGenerationCore>>) {
                return core->resume();
            } else {
                throw std::invalid_argument("model suspend applies to Generation Engines");
            }
        },
        impl_->core);
}

ResidencyStatus Engine::residency() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& core) -> ResidencyStatus {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::GenerationCore>> ||
                          std::is_same_v<CoreState, std::unique_ptr<Impl::HybridGenerationCore>>) {
                return core->residency();
            } else {
                return ResidencyStatus{};
            }
        },
        impl_->core);
}

void Engine::stop() noexcept {
    if (impl_ != nullptr) { impl_->stop(); }
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
