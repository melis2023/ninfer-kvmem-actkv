#include "runtime/engine/flint_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/startup.h"
#include "models/expert_profile.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/processor.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/readout_logprobs.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/grammar_masks.h"
#include "runtime/contract/lookup_draft.h"
#include "runtime/contract/rope_scaling.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/worker_recovery.h"
#include "runtime/engine/effective_thinking_budget.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/host_memory.h"
#include "runtime/engine/model_instance.h"
#include "text/structured_output.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
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

bool request_error(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const RequestError&) { return true; } catch (const std::invalid_argument&) {
        return true;
    } catch (const text::GrammarViolation&) {
        // Raised by the output policy before anything of the round is committed.
        return true;
    } catch (...) {}
    return false;
}

} // namespace

bool is_flint_artifact(const std::filesystem::path& path) {
    if (path.extension() != ".ninfer") { return false; }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) { return false; }
    try {
        const artifact::Reader reader(path);
        return models::flint::is_flint(reader);
    } catch (const std::exception&) { return false; }
}

ConstructedFlint construct_flint(const EngineOptions& options, DeviceContext& device) {
    const auto start = Clock::now();
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    if (options.enable_vision) { throw std::invalid_argument("Flint has no vision tower"); }
    // Flint publishes no draft head: a speculative round verifies what a context lookup proposes.
    const SpeculativeOptions& speculative = options.speculative;
    if (speculative.backend != SpeculativeBackend::None) {
        throw std::invalid_argument("Flint has no draft model; it drafts by context lookup only "
                                    "(--lookup-ngram, without --spec)");
    }
    const std::uint32_t lookup_drafts =
        speculative.lookup_ngram == 0 ? 0
        : speculative.draft_tokens != 0 ? speculative.draft_tokens
                                        : SpeculativeOptions::kAutomaticDraftTokens;
    if (lookup_drafts > 7) {
        throw std::invalid_argument("Flint: --lookup-ngram verifies 1 to 7 drafts (--draft-tokens)");
    }
    if (speculative.draft_min_p != 0 || speculative.mtp_attention_window != 0 ||
        speculative.ngram_draft_tokens != 0 || speculative.ngram_archive_bytes != 0 ||
        speculative.proposal_head != ProposalHead::Auto ||
        speculative.mtp_policy == MtpDraftPolicy::Adaptive) {
        throw std::invalid_argument("Flint has no --draft-min-p, --adaptive-mtp, n-gram copy "
                                    "proposals or archive, --mtp-attention-window or proposal head");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument("Flint keeps its KV cache in F16 (no --kv-dtype formats)");
    }
    const auto& hybrid = options.hybrid_experts;
    if (hybrid.mapped_misses || hybrid.tail_skip_rank != 0 || hybrid.prefill_buffers != 0 ||
        hybrid.adaptive_cache) {
        throw std::invalid_argument("Flint has no mapped misses, tail skip, prefill buffers or "
                                    "adaptive expert cache (Qwen3.8-Flash-Next options)");
    }
    if (options.expert_residency == ExpertResidency::Device &&
        (hybrid.dma_share || hybrid.cpu_threads != 0 || !hybrid.routing_profile.empty() ||
         !hybrid.record_profile.empty())) {
        throw std::invalid_argument("expert DMA shares, CPU threads and routing profiles require "
                                    "host or disk experts (--expert-residency host|disk)");
    }
    if (device.size() != 1 || options.tensor_parallel) {
        throw std::invalid_argument("Flint runs on one device");
    }
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    inspect.complete();
    install_device_route_profile_for(options, device);
    StartupPhaseScope materialize(options.startup_observer, StartupPhase::TargetPlan);
    auto model = models::flint::load_model(
        reader, {.artifact = options.artifact_path, .experts = options.expert_residency}, device,
        &options.startup_observer);
    device.synchronize();
    materialize.complete();
    const auto free_bytes = [&] {
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    };
    const std::size_t free_after_weights = free_bytes();

    StartupPhaseScope frontend_phase(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<FlintInstance>(FlintInstance{
        .model    = nullptr,
        .frontend = models::qwen3_5::make_frontend(
            model->resources(), {.chat_template_path = options.chat_template_path,
                                 .architecture       = models::Architecture::Flint,
                                 .vision_enabled     = false,
                                 .max_context        = options.max_context}),
        .executor = nullptr,
        .capacity = options.max_context});
    frontend_phase.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    models::flint::ExecutorOptions executor;
    executor.max_context   = options.max_context;
    executor.sequences     = std::clamp<std::uint32_t>(options.max_concurrency, 1, 8);
    executor.cuda_graphs   = options.use_cuda_graph;
    executor.draft_tokens  = lookup_drafts;
    executor.expert_cache_bytes = options.expert_cache_bytes;
    executor.expert_dma_share   = hybrid.dma_share;
    executor.expert_cpu_threads = hybrid.cpu_threads;
    // The full-attention blocks carry no rotation and the sliding ones never rotate a distance past
    // their window, so no position takes scaling.
    if (options.rope_yarn || options.rope_yarn_factor > 1.0F || options.rope_scaling_factor > 1.0F) {
        throw std::invalid_argument("Flint takes no rotation scaling: its full-attention blocks "
                                    "have no rotary positions");
    }
    runtime::RopeScalingPlan rope;
    rope.summary.native_context = model->config().max_position_embeddings;
    if (options.max_context > model->config().max_position_embeddings) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                           "Flint: --max-context %u is past the %u positions it was trained on",
                           options.max_context, model->config().max_position_embeddings);
    }
    if (options.expert_residency != ExpertResidency::Device) {
        const std::vector<std::size_t> widths(model->expert_banks().size(),
                                              std::size_t(model->config().num_experts));
        if (!hybrid.routing_profile.empty()) {
            executor.expert_profile = models::read_expert_profile(hybrid.routing_profile,
                                                                  model->info().artifact_id, widths);
        } else {
            executor.expert_profile = models::builtin_expert_profile("flint", widths);
        }
    }
    // Causal scoring keeps no context between requests.
    executor.checkpoints   = options.context_cache.enabled &&
                           options.purpose == EnginePurpose::Generation;
    // Prompt chunks of 2048 tokens whatever was requested: the routed
    // experts' matrix kernel fills its widest tile only at wide chunks, and larger chunks grow the
    // sliding-window rings (window + chunk). A device without room for that chunk's workspace takes
    // the widest of 1024 and 512 that fits.
    for (const std::uint32_t chunk : {2048u, 1024u, 512u}) {
        executor.prefill_chunk = chunk;
        try {
            instance->executor =
                std::make_unique<models::flint::Executor>(*model, device, executor);
            break;
        } catch (const models::flint::InsufficientMemory&) {
            if (chunk == 512) { throw; }
        }
    }
    device.synchronize();
    program.complete();
    {
        StartupPhaseScope warm(options.startup_observer, StartupPhase::CudaGraphPrepare);
        instance->executor->warm_up();
        warm.complete();
    }
    instance->free_after_weights = free_after_weights;
    instance->free_after_startup = free_bytes();
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                       "Flint: %u sequence(s), prompt chunks of %u tokens, F16 KV %.0f MiB%s, "
                       "workspace %.0f MiB",
                       executor.sequences, executor.prefill_chunk,
                       double(instance->executor->memory().kv_bytes) / 1048576.0,
                       instance->executor->memory().kv_grows
                           ? " at most, growing with the sequences beside an elastic expert cache"
                           : "",
                       double(instance->executor->memory().workspace_bytes) / 1048576.0);
    if (options.expert_residency != ExpertResidency::Device) {
        const auto memory = instance->executor->memory().experts;
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
            "Flint: experts in %s, cache %.0f MiB (%u slots), %u layer bank(s) %.0f MiB, "
            "metadata %.1f MiB, pinned delivery %.1f MiB, pinned expert sources %.0f MiB",
            options.expert_residency == ExpertResidency::Host ? "pinned host memory" : "artifact files",
            double(memory.cache_bytes) / 1048576, memory.cache_slots, memory.banks,
            double(memory.bank_bytes) / 1048576, double(memory.metadata_bytes) / 1048576,
            double(memory.pinned_bytes) / 1048576, double(model->expert_pinned_bytes()) / 1048576);
        const auto cpu = instance->executor->expert_cpu_description();
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
            "Flint: missing experts %s; cache seeded %s",
            cpu.empty() ? "copied to the GPU"
                        : ("split between copies and " + cpu + (hybrid.dma_share
                               ? " (fixed share)" : " (by measured speeds)")).c_str(),
            executor.expert_profile ? (hybrid.routing_profile.empty() ? "from the built-in profile"
                                                                      : "from --expert-profile")
                                    : "by the first requests");
    }
    instance->record_profile = hybrid.record_profile;
    if (lookup_drafts != 0) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "Flint: context-lookup speculative decoding, %u-token match, up to %u "
                           "drafts a round",
                           speculative.lookup_ngram, lookup_drafts);
    }
    ConstructedFlint out;
    const auto& stats              = model->storage_stats();
    out.load.architecture          = std::string(models::architecture_name(models::Architecture::Flint));
    out.load.model_name            = model->info().name;
    out.load.prefill_signature     = "flint";
    std::set<std::string> formats;
    for (const auto& weight : model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    out.load.weight_formats.assign(formats.begin(), formats.end());
    out.load.load_seconds         = seconds(start, Clock::now());
    out.load.upload_seconds       = stats.upload_seconds;
    out.load.artifact_bytes_read  = stats.read_bytes;
    out.load.host_to_device_bytes = stats.h2d_bytes;
    out.load.peak_staging_bytes   = stats.peak_staging_bytes;
    out.load.pinned_weight_bytes  = stats.pinned_bytes;
    out.load.device_object_count  = stats.device_object_count;
    out.load.host_object_count    = stats.host_object_count;
    out.load.rope                 = rope.summary;
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

struct FlintCore::Request {
    std::uint64_t id = 0;
    models::qwen3_5::PreparedPrompt prompt;
    std::vector<TokenId> prompt_tokens;
    models::qwen3_5::OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
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

    std::uint32_t anchor_at = 0;
    bool reusable           = true;
    // CacheRetentionHint::Disposable: no request continues where this one ends (a decision read),
    // so the store keeps no image of its end.
    bool disposable = false;
    // Engine::submit_branches: the prompt is the branches' shared prefix, and once it is in, each
    // branch runs from the snapshot at its end and its first token is read. branch_fed counts the
    // current branch's tokens fed, branch_results the branches read; the first branch's token and
    // record are what the content publishes.
    std::shared_ptr<const std::vector<runtime::ReadBranch>> branches;
    std::uint32_t branch_fed = 0;
    std::vector<BranchResult> branch_results;
    TokenId branch_token = 0;
    runtime::RawTokenLogprob branch_record;
    // With a grammar (structured output): the mask of the token it samples next, started once a
    // token is accepted when the request decodes plainly, else by the plain step that needs it. A
    // decode step runs without waiting for it: when the mask is late, the row's logits are kept
    // (saved_logits) and the token is sampled once the mask is ready. A speculative round fills
    // its own masks, one per column after its drafts.
    bool constrained = false;
    std::shared_ptr<runtime::GrammarMaskJob> mask;
    runtime::GrammarMaskCost mask_cost;
    bool awaiting_sample           = false;
    std::int32_t awaiting_purpose  = 0;
    // The last prompt chunk or accepted token, which the progress watchdog measures from.
    Clock::time_point progress;
    // Prefixes other prompts are expected to share (the end of the leading system messages, of
    // tool declarations, of a declared cache boundary), ascending and below anchor_at: a prompt
    // chunk ends at each, and the state there goes to the host store for them.
    std::vector<std::uint32_t> shared_at;

    std::uint32_t slot         = 0;
    std::uint32_t reused       = 0;
    PrefixReusePath reuse_path = PrefixReusePath::Root;
    std::uint32_t prefilled    = 0;
    bool decoding              = false;
    bool penalties             = false;
    bool post_thinking         = false;
    std::uint32_t position     = 0;
    std::vector<TokenId> feed;
    std::vector<TokenId> generated;
    std::optional<GenerationBudget> budget;
    Clock::time_point admitted, prefill_start, prefill_end, first_token, last_token;
    SpeculativeStats speculative; // an Engine with --lookup-ngram only
};

struct FlintCore::Impl {
    struct Slot {
        std::shared_ptr<Request> request;
        std::vector<TokenId> fed;
        std::vector<TokenId> anchor;
        models::flint::SequenceSnapshot snapshot;
        std::uint64_t last_used = 0;
        bool disposable         = false; // its last request's
    };

    struct SampleRow {
        const ResolvedSamplingParameters* params = nullptr;
        std::uint32_t position                   = 0;
        std::uint32_t slot                       = 0;
        bool counts                              = false;
        bool logprobs                            = false;
        const std::uint32_t* mask                = nullptr; // a grammar's next-token mask
        // The typed readout's token ids, on a request's first generated token only.
        const std::vector<TokenId>* readout = nullptr;
    };

    FlintInstance& instance;
    DeviceContext& device;
    const std::uint32_t max_context;
    const std::size_t max_outstanding;
    const std::chrono::milliseconds pending_timeout;
    const std::uint32_t domain;
    const bool structured_output;
    const bool reuse_prefixes;
    const std::uint32_t drafts;       // a lookup's drafts a round; 0 without speculation
    const std::uint32_t lookup_ngram; // --lookup-ngram: tokens a context lookup matches; 0 is off
    const DiagnosticObserver diagnostics;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::shared_ptr<Request>> pending;
    std::size_t outstanding = 0;
    bool stopping           = false;
    bool failed             = false;
    std::uint64_t next_id   = 1;

    mutable std::mutex stats_mutex;
    RuntimeStats stats;

    std::vector<Slot> slots;
    std::uint64_t use_clock                 = 0;

    // The context cache past the sequences: images of the prefixes they give up (the state a turn
    // closed at, and where a sequence ended) and of shared prefixes, in pinned host memory up to a
    // budget, the least recently used dropped first. A prompt resumes from the longest stored
    // prefix when no free sequence serves more of it.
    struct Stored {
        std::vector<TokenId> tokens;
        models::flint::SequenceImage header;
        PrefixReusePath path = PrefixReusePath::PrivateEndpoint;
        std::vector<std::byte*> pages; // slices of host_pages
        std::uint64_t bytes     = 0;
        std::uint64_t last_used = 0;
    };
    // The images live in pages of one buffer pinned at startup: pinning and unpinning each image
    // would stall every sequence, and since nothing is freed, the copies need no host wait (the
    // executor's stream orders a page's reuse after the copies before it).
    static constexpr std::size_t kHostPageBytes = std::size_t(8) << 20;
    std::unique_ptr<PinnedHostBuffer> host_pages;
    std::vector<std::byte*> free_pages;
    // An image holds the sliding rings whatever its length (53 MiB in F16), which copy back in a
    // few milliseconds against about 0.2 ms of prefill per token: the bound is the budget. A
    // conversation's prefixes are many and keep 1024 tokens or more; a shared prefix, which many
    // requests reuse, once its image pays (image_pays).
    static constexpr std::size_t kMinStoredTokens = 1024;
    // Host copies of an image each way, about PCIe 4.0 x16's; and the prefill time per token.
    static constexpr double kImageCopyBytesPerSecond = 12e9;
    static constexpr double kPrefillSecondsPerToken  = 2e-4;
    std::vector<Stored> stored;
    std::uint64_t host_budget = 0, host_used = 0;
    std::uint32_t decode_run                = 0;
    std::uint32_t decode_rounds_per_prefill = 1;

    DeviceBuffer sample_config, sample_position, sample_out, token_counts, score_targets, score_out;
    // A row's logits kept while its grammar mask is late (one per sequence), and those of the rows
    // sampled together afterwards, side by side.
    DeviceBuffer saved_logits, deferred_logits;
    std::size_t vocab_rows = 0; // logits per row
    DeviceBuffer token_mask;
    DeviceBuffer logprob_ids, logprob_values, logprob_lse, logprob_flag;
    DeviceBuffer readout_ids, readout_values;
    std::unique_ptr<PinnedHostBuffer> host_readout;
    std::unique_ptr<PinnedHostBuffer> host_mask, host_sample, host_logprobs;
    std::unique_ptr<WorkspaceArena> sample_workspace;
    // A speculative round's acceptance, a row per verified sequence: its drafts, the verification's
    // greedy targets, the live draft count and the tokens fed before the round, and the licensed
    // run (tokens, count, accepted drafts, last token); and every slot's token counts before the
    // round, which a round that keeps fewer tokens than it licensed restores.
    DeviceBuffer spec_drafts, spec_targets, spec_extents, spec_lengths, spec_licensed, spec_counts,
        spec_accepted, spec_anchors, counts_backup;
    std::unique_ptr<PinnedHostBuffer> host_spec;

    std::thread worker;

    Impl(FlintInstance& i, DeviceContext& d, const EngineOptions& options)
        : instance(i), device(d), max_context(options.max_context),
          max_outstanding(std::size_t(options.max_concurrency) + options.max_pending_requests),
          pending_timeout(options.pending_timeout_ms),
          domain(i.model->resources().public_token_count),
          structured_output(options.structured_output),
          reuse_prefixes(i.executor->options().checkpoints), drafts(i.executor->draft_tokens()),
          lookup_ngram(options.speculative.lookup_ngram), diagnostics(options.diagnostic_observer),
          slots(i.executor->options().sequences),
          decode_rounds_per_prefill(
              options.decode_rounds_per_prefill != 0
                  ? options.decode_rounds_per_prefill
                  : std::max<std::uint32_t>(1, i.executor->options().prefill_chunk / 64)) {
        if (reuse_prefixes) {
            // The host tier's budget, sized here once the weights are loaded when it is
            // automatic, and pinned now in whole pages.
            const ContextCacheOptions& cache = options.context_cache;
            const ContextCacheOptions sized  = resolve_auto_host_cache_now(cache, 0U);
            const std::uint64_t budget =
                sized.host_cache_budget_bytes.value_or(cache.host_kv_capacity_bytes);
            const std::size_t pages = static_cast<std::size_t>(budget / kHostPageBytes);
            host_budget             = std::uint64_t(pages) * kHostPageBytes;
            if (pages > 0) {
                host_pages = std::make_unique<PinnedHostBuffer>(host_budget);
                auto* base = static_cast<std::byte*>(host_pages->data());
                free_pages.reserve(pages);
                for (std::size_t i = pages; i-- > 0;) { free_pages.push_back(base + i * kHostPageBytes); }
            }
        }
        const std::size_t rows = slots.size();
        // A sampling call's columns per row: one, or a verification's drafts and bonus.
        const std::size_t width = std::size_t(drafts) + 1;
        if (structured_output) {
            token_mask = DeviceBuffer(rows * width * mask_words() * sizeof(std::uint32_t));
            host_mask  = std::make_unique<PinnedHostBuffer>(rows * width * mask_words() *
                                                            sizeof(std::uint32_t));
        }
        if (structured_output) {
            vocab_rows      = static_cast<std::size_t>(instance.executor->logits(1).ne[0]);
            saved_logits    = DeviceBuffer(rows * vocab_rows * sizeof(__nv_bfloat16));
            deferred_logits = DeviceBuffer(rows * vocab_rows * sizeof(__nv_bfloat16));
        }
        sample_config   = DeviceBuffer(rows * sizeof(ops::SamplingConfig));
        sample_position = DeviceBuffer(rows * sizeof(std::int32_t));
        sample_out      = DeviceBuffer(rows * sizeof(std::int32_t));
        host_sample     = std::make_unique<PinnedHostBuffer>(
            rows * (sizeof(ops::SamplingConfig) + 2 * sizeof(std::int32_t)));
        token_counts  = DeviceBuffer(rows * domain * sizeof(std::int32_t));
        score_targets = DeviceBuffer(models::flint::Executor::kMaxLogitRows * sizeof(std::int32_t));
        score_out     = DeviceBuffer(models::flint::Executor::kMaxLogitRows * sizeof(float));
        const std::size_t top = rows * width * kMaximumTokenLogprobs;
        logprob_ids    = DeviceBuffer(top * sizeof(std::int32_t));
        logprob_values = DeviceBuffer(top * sizeof(float));
        logprob_lse    = DeviceBuffer(rows * width * sizeof(float));
        logprob_flag   = DeviceBuffer(sizeof(std::int32_t));
        readout_ids           = DeviceBuffer(rows * kMaximumReadoutTokens * sizeof(std::int32_t));
        readout_values        = DeviceBuffer(rows * kMaximumReadoutTokens * sizeof(float));
        host_readout = std::make_unique<PinnedHostBuffer>(rows * kMaximumReadoutTokens *
                                                          (sizeof(std::int32_t) + sizeof(float)));
        host_logprobs =
            std::make_unique<PinnedHostBuffer>(top * (sizeof(std::int32_t) + sizeof(float)));
        const std::int32_t enabled = 1;
        CUDA_CHECK(cudaMemcpy(logprob_flag.p, &enabled, sizeof(enabled), cudaMemcpyHostToDevice));
        std::size_t workspace = std::max<std::size_t>(
            {ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(domain), 1,
                                                    static_cast<std::int32_t>(rows)),
             ops::logprob_topk_workspace_capacity_bytes(static_cast<std::int32_t>(domain),
                                                        static_cast<std::int32_t>(rows * width)),
             std::size_t{256}});
        if (drafts > 0) {
            workspace =
                std::max(workspace, ops::speculative_accept_coupled_drafts_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(domain), 1,
                                        static_cast<std::int32_t>(drafts), 1,
                                        static_cast<std::int32_t>(rows)));
            const auto plane = [&](std::size_t count) {
                return DeviceBuffer(rows * count * sizeof(std::int32_t));
            };
            spec_drafts   = plane(drafts);
            spec_targets  = plane(width);
            spec_licensed = plane(width);
            spec_extents  = plane(1);
            spec_lengths  = plane(1);
            spec_counts   = plane(1);
            spec_accepted = plane(1);
            spec_anchors  = plane(1);
            counts_backup = DeviceBuffer(rows * domain * sizeof(std::int32_t));
            host_spec     = std::make_unique<PinnedHostBuffer>(rows * (drafts + width + 4) *
                                                               sizeof(std::int32_t));
        }
        sample_workspace = std::make_unique<WorkspaceArena>(workspace);
        load_sampling_kernels();
        worker           = std::thread([this] {
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

    [[nodiscard]] std::size_t mask_words() const noexcept { return (std::size_t(domain) + 31) / 32; }

    // Launches every sampling kernel a step or a speculative round may launch once now, on scratch
    // inputs. CUDA loads a kernel at its first launch, and a load while a decode or verification
    // graph waits on the device for the host experts' service waits for that graph, whose service
    // needs the driver lock the load holds: the first round after startup would never finish.
    void load_sampling_kernels() {
        const cudaStream_t stream = instance.executor->head_stream();
        const ops::SamplingConfig greedy{};
        CUDA_CHECK(cudaMemcpy(sample_config.p, &greedy, sizeof(greedy), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(sample_position.p, 0, sizeof(std::int32_t)));
        const auto* config  = static_cast<const ops::SamplingConfig*>(sample_config.p);
        const auto vocab    = static_cast<std::int32_t>(domain);
        const Tensor flag(logprob_flag.p, DType::I32, {1});
        const auto gather = [&](const Tensor& logits, std::int32_t columns, const Tensor* drafted) {
            Tensor ids(logprob_ids.p, DType::I32, {ops::kLogprobTopK, columns, 1});
            Tensor values(logprob_values.p, DType::FP32, {ops::kLogprobTopK, columns, 1});
            Tensor lse(logprob_lse.p, DType::FP32, {columns, 1});
            ops::logprob_topk(logits.view({logits.ne[0], columns, 1}), config, drafted, vocab, ids,
                              values, lse, flag, *sample_workspace, stream);
        };
        {
            const Tensor logits = instance.executor->logits(1);
            Tensor sampled(sample_out.p, DType::I32, {1});
            const Tensor logical(sample_position.p, DType::I32, {1});
            gather(logits, 1, nullptr);
            auto scope = sample_workspace->scope();
            ops::sample(logits, sampled, vocab, config, logical, ops::kSamplePurposeDecode,
                        *sample_workspace, stream);
            // The sampled id may be the non-finite marker; the zeroed position is a valid id.
            Tensor counts(token_counts.p, DType::I32, {vocab});
            ops::increment_token_counts(logical, counts, stream);
        }
        if (drafts > 0) {
            for (DeviceBuffer* plane : {&spec_drafts, &spec_extents, &spec_lengths, &spec_anchors}) {
                CUDA_CHECK(cudaMemset(plane->p, 0, plane->bytes));
            }
            // A round's width is its drafts and the bonus column, one draft to all of them.
            for (std::uint32_t k = 1; k <= drafts; ++k) {
                const auto columns  = static_cast<std::int32_t>(k + 1);
                const Tensor logits = instance.executor->logits(k + 1);
                const Tensor drafted(spec_drafts.p, DType::I32, {static_cast<std::int32_t>(k), 1});
                gather(logits, columns, &drafted);
                Tensor targets(spec_targets.p, DType::I32, {columns});
                ops::argmax(logits, targets, vocab, stream);
                Tensor extents(spec_extents.p, DType::I32, {1});
                Tensor lengths(spec_lengths.p, DType::I32, {1});
                Tensor anchors(spec_anchors.p, DType::I32, {1});
                Tensor licensed(spec_licensed.p, DType::I32, {columns, 1});
                Tensor counts(spec_counts.p, DType::I32, {1});
                Tensor accepted(spec_accepted.p, DType::I32, {1});
                auto scope = sample_workspace->scope();
                ops::speculative_accept_coupled_drafts(
                    targets.view({columns, 1}), logits.view({logits.ne[0], columns, 1}), drafted,
                    extents, lengths, anchors, licensed, counts, accepted, vocab, config,
                    *sample_workspace, stream);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Each request clears its own row of the counts; the scratch increment leaves no trace.
        CUDA_CHECK(cudaMemset(token_counts.p, 0, token_counts.bytes));
    }

    // A request that neither computes a prompt chunk nor accepts a token for this long fails with
    // an explicit error rather than holding its sequence, so a stall surfaces as errors.
    static constexpr auto kProgressTimeout = std::chrono::seconds(120);
    Clock::time_point progress_checked{};

    // Starts the mask of a structured-output request's next token, from its grammar's state now.
    void schedule_mask(Request& r) {
        const auto grammar = r.output.grammar_state();
        r.mask = grammar ? runtime::GrammarMaskJob::start(*grammar, mask_words()) : nullptr;
    }

    // After a structured request's accepted tokens: one that decodes plainly starts its next mask
    // now, a step ahead; one that speculates fills its round's columns during the verification.
    void next_mask(const std::shared_ptr<Request>& request) {
        Request& r = *request;
        if (r.constrained && active(request) && !(drafts > 0 && r.mask_cost.speculates())) {
            schedule_mask(r);
        } else {
            r.mask.reset();
        }
    }

    [[nodiscard]] static bool mask_ready(const Request& r) { return !r.mask || r.mask->ready(); }

    // After its mask is ready: false (the request failed) when the grammar had no admissible token
    // or took over the budget.
    bool mask_usable(const std::shared_ptr<Request>& request) {
        const auto& job = request->mask;
        if (!job) { return true; }
        request->mask_cost.observe(job->seconds());
        const std::string problem = runtime::grammar_mask_problem(*job);
        if (problem.empty()) { return true; }
        fail(request, std::make_exception_ptr(RequestError(
                          RequestErrorKind::Unavailable, "structured output: " + problem)));
        return false;
    }

    // Fails each request that has made no progress for kProgressTimeout.
    void check_progress() {
        const auto now = Clock::now();
        if (now - progress_checked < std::chrono::seconds(1)) { return; }
        progress_checked = now;
        for (const Slot& slot : slots) {
            const auto request = slot.request;
            if (!request || now - request->progress < kProgressTimeout) { continue; }
            fail(request, std::make_exception_ptr(RequestError(
                              RequestErrorKind::Unavailable,
                              "inference request made no progress for " +
                                  std::to_string(kProgressTimeout.count()) + " s")));
        }
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

    // Not a request's own error: the Engine stops serving, and the cause is logged here, where it
    // is the operator's only lead.
    void fail_engine(std::exception_ptr error) {
        const auto running = std::count_if(
            slots.begin(), slots.end(), [](const Slot& slot) { return slot.request != nullptr; });
        publish_diagnostic(diagnostics, DiagnosticLevel::Error,
                           "worker failure: %s - failing %td running request(s), the Engine "
                           "stops serving",
                           exception_text(error).c_str(), running);
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
            bool worked = true;
            try {
                admit();
                worked = step();
                check_progress();
            } catch (...) { fail_engine(std::current_exception()); }
            publish_stats();
            if (!worked) {
                // Every active request waits for its grammar mask.
                std::unique_lock lock(queue_mutex);
                queue_cv.wait_for(lock, std::chrono::milliseconds(1),
                                  [&] { return stopping || !pending.empty(); });
            }
        }
    }

    void publish_queue() {
        std::lock_guard lock(queue_mutex);
        std::lock_guard stats_lock(stats_mutex);
        stats.waiting_requests = static_cast<std::uint32_t>(pending.size());
    }

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

    // The tokens `held` the prompt starts with.
    static std::uint32_t prefix_reuse(const std::vector<TokenId>& held, const Request& prompt) {
        // A prompt's last token is fed, since its first sample needs the logits there; a
        // branched read's branches bring their own, so it can resume at its prompt's end.
        const auto& tokens     = prompt.prompt_tokens;
        const std::size_t most = prompt.branches ? tokens.size() : tokens.size() - 1;
        if (held.empty() || tokens.empty() || held.size() > most ||
            !std::equal(held.begin(), held.end(), tokens.begin())) {
            return 0;
        }
        return static_cast<std::uint32_t>(held.size());
    }

    // Frees pages for an image of `bytes`, dropping the least recently used images; false when
    // the budget cannot hold it at all.
    bool make_host_room(std::uint64_t bytes) {
        const std::size_t pages = static_cast<std::size_t>((bytes + kHostPageBytes - 1) / kHostPageBytes);
        if (pages > host_budget / kHostPageBytes) { return false; }
        while (free_pages.size() < pages) {
            const auto oldest = std::min_element(stored.begin(), stored.end(), [](const Stored& a, const Stored& b) {
                return a.last_used < b.last_used;
            });
            if (oldest == stored.end()) { return false; }
            free_pages.insert(free_pages.end(), oldest->pages.begin(), oldest->pages.end());
            host_used -= oldest->pages.size() * kHostPageBytes;
            stored.erase(oldest);
        }
        return true;
    }

    [[nodiscard]] std::optional<std::size_t> stored_index(const std::vector<TokenId>& tokens) const {
        for (std::size_t i = 0; i < stored.size(); ++i) {
            if (stored[i].tokens == tokens) { return i; }
        }
        return std::nullopt;
    }

    // The stored image that serves the most of the prompt, and how much.
    [[nodiscard]] std::pair<std::size_t, std::uint32_t> find_stored(const Request& prompt) const {
        std::pair<std::size_t, std::uint32_t> best{0, 0};
        for (std::size_t i = 0; i < stored.size(); ++i) {
            const std::uint32_t length = prefix_reuse(stored[i].tokens, prompt);
            if (length > best.second) { best = {i, length}; }
        }
        return best;
    }

    // Whether a shared prefix of `tokens` is worth an image: saving it and copying it back take
    // less time than computing its tokens again. The prompt's chunks end where such images lie,
    // so the answer depends on the prefix alone, never on what ran before or beside it: a prompt
    // computes in the same chunks, and to the same values, under any load.
    [[nodiscard]] bool image_pays(std::uint32_t tokens) const {
        const double copy =
            2.0 * double(instance.executor->image_bytes(tokens)) / kImageCopyBytesPerSecond;
        return double(tokens) * kPrefillSecondsPerToken > copy;
    }

    // Keeps the state of sequence `s` up to `tokens` (its live state, or its checkpoint `at`) in
    // the store, unless the store already holds that prefix or it is too short to be worth it.
    void store_image(std::uint32_t s, const std::vector<TokenId>& tokens,
                     const models::flint::SequenceSnapshot* at, PrefixReusePath path) {
        if (host_budget == 0) { return; }
        if (path == PrefixReusePath::SharedStablePrefix
                ? !image_pays(static_cast<std::uint32_t>(tokens.size()))
                : tokens.size() < kMinStoredTokens) {
            return;
        }
        if (const auto known = stored_index(tokens)) {
            stored[*known].last_used = ++use_clock;
            return;
        }
        auto& executor      = *instance.executor;
        const auto position = at != nullptr ? at->position : executor.position(s);
        if (position != tokens.size()) { return; } // the state holds other tokens
        const std::uint64_t bytes = executor.image_bytes(position);
        if (!make_host_room(bytes)) { return; }
        Stored image;
        image.tokens = tokens;
        image.path   = path;
        image.bytes  = bytes;
        const auto count = static_cast<std::ptrdiff_t>((bytes + kHostPageBytes - 1) / kHostPageBytes);
        image.pages.assign(free_pages.end() - count, free_pages.end());
        free_pages.erase(free_pages.end() - count, free_pages.end());
        image.header    = executor.save_image(s, at, image.pages, kHostPageBytes);
        image.last_used = ++use_clock;
        host_used += image.pages.size() * kHostPageBytes;
        stored.push_back(std::move(image));
        std::lock_guard lock(stats_mutex);
        stats.state_d2h_count += 1;
        stats.state_d2h_bytes += bytes;
        stats.host_kv_occupied_bytes = host_used;
    }

    // Keeps what sequence `s` holds before its state goes: the prefix its last prompt's turn
    // closed at, and where it ended.
    void retire(std::uint32_t s) {
        Slot& slot          = slots[s];
        const bool anchored = !slot.anchor.empty() && slot.anchor.size() <= slot.fed.size() &&
                              std::equal(slot.anchor.begin(), slot.anchor.end(), slot.fed.begin());
        if (anchored) {
            store_image(s, slot.anchor, &slot.snapshot, PrefixReusePath::PrivateTurnClosure);
        }
        if (slot.fed != slot.anchor && !slot.disposable) {
            store_image(s, slot.fed, nullptr, PrefixReusePath::PrivateEndpoint);
        }
    }

    // Loads stored image `index` into sequence `s`.
    void restore_stored(std::uint32_t s, std::size_t index) {
        Stored& image = stored.at(index);
        instance.executor->load_image(s, image.header, image.pages, kHostPageBytes);
        image.last_used = ++use_clock;
        std::lock_guard lock(stats_mutex);
        stats.state_h2d_count += 1;
        stats.state_h2d_bytes += image.bytes;
    }

    [[nodiscard]] std::uint32_t reuse(const Slot& slot, const Request& prompt) const {
        if (!reuse_prefixes) { return 0; }
        return std::max(prefix_reuse(slot.fed, prompt), prefix_reuse(slot.anchor, prompt));
    }

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
                    const Request& prompt = *pending.front();
                    const auto candidate  = reuse(slots[s], prompt);
                    const auto current   = reuse(slots[*best], prompt);
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
        r.admitted           = Clock::now();
        // Resume from the sequence's live state when the prompt continues it, else from its
        // checkpoint at the end of its last prompt when the prompt continues that, else from the
        // longest prefix the store keeps; what the sequence held goes to the store first.
        std::uint32_t reused = 0;
        if (reuse_prefixes && r.reusable) {
            const std::uint32_t live       = prefix_reuse(slot.fed, r);
            const std::uint32_t anchored   = prefix_reuse(slot.anchor, r);
            const auto [image, from_store] = find_stored(r);
            if (from_store > std::max(live, anchored)) {
                std::vector<TokenId> tokens = stored[image].tokens;
                const PrefixReusePath path  = stored[image].path;
                retire(s);
                // Retiring may have dropped or moved the image: find it again.
                if (const auto index = stored_index(tokens)) {
                    restore_stored(s, *index);
                    slot.fed = std::move(tokens);
                    slot.anchor.clear();
                    reused       = from_store;
                    r.reuse_path = path;
                }
            } else if (anchored > live) {
                retire(s);
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
            if (reuse_prefixes) { retire(s); }
            executor.reset(s);
            slot.fed.clear();
            slot.anchor.clear();
        }
        slot.request    = request;
        slot.last_used  = ++use_clock;
        slot.disposable = r.disposable;
        r.reused = r.prefilled = reused;
        if (drafts > 0) {
            r.speculative.enabled      = true;
            r.speculative.draft_window = drafts;
            r.speculative.accepted_per_position.assign(drafts, 0);
        }
        r.prefill_start        = Clock::now();
        r.progress             = r.prefill_start;
        r.constrained          = r.output.grammar_state() != nullptr;
        if (r.constrained) { schedule_mask(r); } // the first token's, while the prompt prefills
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

    // The shortest remaining prompt first, so a short or cached request is not held behind a
    // long one; ties go to the earlier request.
    static bool prefills_before(const Request& a, const Request& b) {
        const auto left = [](const Request& r) {
            const std::size_t n = r.prompt_tokens.size();
            return n - std::min<std::size_t>(r.prefilled, n);
        };
        if (left(a) != left(b)) { return left(a) < left(b); }
        return a.id < b.id;
    }

    // One prompt chunk or one decode round; false when there was nothing to run (every active
    // request waits for its grammar mask).
    bool step() {
        std::shared_ptr<Request> prefilling;
        bool decoding = false;
        for (const Slot& slot : slots) {
            if (!slot.request) { continue; }
            if (slot.request->decoding) {
                // A request runs a decode step unless its kept logits still wait for their mask.
                decoding = decoding || !slot.request->awaiting_sample || mask_ready(*slot.request);
            } else if (!prefilling || prefills_before(*slot.request, *prefilling)) {
                prefilling = slot.request;
            }
        }
        if (prefilling && (!decoding || decode_run >= decode_rounds_before(*prefilling))) {
            decode_run = 0;
            try {
                prefill_step(prefilling);
            } catch (...) { fail(prefilling, std::current_exception()); }
            return true;
        }
        if (!decoding) { return false; }
        decode_run += decode_run != UINT32_MAX;
        decode_step();
        return true;
    }

    // The prompt tokens the request's next prefill step computes.
    [[nodiscard]] std::uint32_t prefill_step_tokens(const Request& r) const {
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        if (r.branches && r.prefilled == prompt_n) {
            const std::size_t left =
                (*r.branches)[r.branch_results.size()].tokens.size() - r.branch_fed;
            return static_cast<std::uint32_t>(
                std::min<std::size_t>(left, instance.executor->options().prefill_chunk));
        }
        std::uint32_t until = r.prefilled < r.anchor_at ? r.anchor_at : prompt_n;
        for (const std::uint32_t frontier : r.shared_at) {
            if (frontier > r.prefilled) {
                until = std::min(until, frontier);
                break;
            }
        }
        return std::min(instance.executor->options().prefill_chunk, until - r.prefilled);
    }

    // Decode rounds that run before a prefill step while other requests generate: the configured
    // rounds per full chunk, proportionally fewer (at least one) before a shorter step, so a short
    // prompt is not held behind a long chunk's worth of decode rounds.
    [[nodiscard]] std::uint32_t decode_rounds_before(const Request& r) const {
        const std::uint64_t chunk = std::max<std::uint32_t>(
            instance.executor->options().prefill_chunk, 1);
        return static_cast<std::uint32_t>(std::max<std::uint64_t>(
            1, (std::uint64_t(decode_rounds_per_prefill) * prefill_step_tokens(r) + chunk - 1) /
                   chunk));
    }

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

    void sample(std::span<const SampleRow> rows, std::int32_t purpose, std::span<TokenId> out,
                std::span<runtime::RawTokenLogprob> logprobs, const Tensor* source = nullptr) {
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
            config.token_counts      = row.counts ? static_cast<std::int32_t*>(token_counts.p) +
                                                        std::size_t(row.slot) * domain
                                                  : nullptr;
            if (row.mask != nullptr) {
                auto* words = static_cast<std::uint32_t*>(host_mask->data()) + b * mask_words();
                std::copy_n(row.mask, mask_words(), words);
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
        const Tensor logits =
            source != nullptr ? *source : instance.executor->logits(static_cast<std::uint32_t>(n));
        const auto* device_configs = static_cast<const ops::SamplingConfig*>(sample_config.p);
        const bool gather = std::any_of(rows.begin(), rows.end(),
                                        [](const SampleRow& row) { return row.logprobs; });
        const std::size_t top = n * kMaximumTokenLogprobs;
        auto* host_ids        = static_cast<std::int32_t*>(host_logprobs->data());
        auto* host_values =
            reinterpret_cast<float*>(host_ids + slots.size() * kMaximumTokenLogprobs);
        if (gather) {
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
        auto* readout_host_ids = static_cast<std::int32_t*>(host_readout->data());
        auto* readout_host_values =
            reinterpret_cast<float*>(readout_host_ids + slots.size() * kMaximumReadoutTokens);
        const bool read = std::any_of(rows.begin(), rows.end(),
                                      [](const SampleRow& row) { return row.readout != nullptr; });
        if (read) {
            for (std::size_t b = 0; b < n; ++b) {
                if (rows[b].readout == nullptr) { continue; }
                std::copy(rows[b].readout->begin(), rows[b].readout->end(),
                          readout_host_ids + b * kMaximumReadoutTokens);
            }
            CUDA_CHECK(cudaMemcpyAsync(readout_ids.p, readout_host_ids,
                                       n * kMaximumReadoutTokens * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            for (std::size_t b = 0; b < n; ++b) {
                if (rows[b].readout == nullptr) { continue; }
                const auto offset = b * kMaximumReadoutTokens;
                const Tensor ids(static_cast<std::int32_t*>(readout_ids.p) + offset, DType::I32,
                                 {static_cast<std::int32_t>(kMaximumReadoutTokens)});
                Tensor values(static_cast<float*>(readout_values.p) + offset, DType::FP32,
                              {static_cast<std::int32_t>(kMaximumReadoutTokens)});
                ops::readout_logprobs(logits.slice(1, static_cast<std::int32_t>(b), 1), ids,
                                      static_cast<std::int32_t>(rows[b].readout->size()),
                                      static_cast<std::int32_t>(domain), values, stream);
            }
            CUDA_CHECK(cudaMemcpyAsync(readout_host_values, readout_values.p,
                                       n * kMaximumReadoutTokens * sizeof(float),
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
        instance.executor->check_experts();
        for (std::size_t b = 0; b < n; ++b) {
            const std::int32_t token = tokens[b];
            if (token == ops::kSamplerNonFiniteToken || token < 0 ||
                std::uint32_t(token) >= domain) {
                throw std::runtime_error("Flint produced non-finite logits");
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
            if (rows[b].readout != nullptr) {
                record.readout_count = static_cast<std::uint32_t>(rows[b].readout->size());
                std::copy_n(readout_host_values + b * kMaximumReadoutTokens, record.readout_count,
                            record.readout_values.begin());
            }
        }
    }

    [[nodiscard]] SampleRow sample_row(Request& r) {
        const auto& exec = r.options.execution;
        if (!r.post_thinking && exec.post_thinking_sampling && r.output.reasoning_closed()) {
            r.post_thinking = true;
        }
        return SampleRow{.params = r.post_thinking ? &*exec.post_thinking_sampling : &exec.sampling,
                         .position = r.position,
                         .slot     = r.slot,
                         .counts   = r.penalties,
                         .logprobs = exec.logprobs,
                         .mask     = r.mask ? r.mask->masks() : nullptr,
                         .readout  = exec.logprobs && !exec.readout_tokens.empty() &&
                                             r.first_token == Clock::time_point{}
                                         ? &exec.readout_tokens
                                         : nullptr};
    }

    // --expert-profile-out: every route since warm-up, rewritten at each request boundary.
    void save_expert_profile() {
        if (instance.record_profile.empty()) { return; }
        models::write_expert_profile(instance.record_profile, instance.model->info().artifact_id,
                                     instance.executor->expert_totals());
    }

    void finish(const std::shared_ptr<Request>& request, FinishReason reason) {
        save_expert_profile();
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
        result.computed_prompt_tokens          = r.prefilled - r.reused;
        result.reused_prompt_tokens            = r.reused;
        result.prefix_reuse_path               = r.reuse_path;
        result.branches                        = std::move(r.branch_results);
        if (drafts > 0) { result.speculative = r.speculative; }
        result.timings.prepare_seconds         = r.prepare_seconds;
        result.timings.prefill_seconds         = seconds(r.prefill_start, r.prefill_end);
        result.timings.decode_seconds          = seconds(r.prefill_end, r.last_token);
        result.timings.first_token_seconds = r.prepare_seconds + seconds(r.submitted, r.first_token);
        if (r.observation.phase_timings) {
            result.timings.prompt_wall_seconds     = seconds(r.admitted, r.first_token);
            result.timings.generation_wall_seconds = seconds(r.first_token, r.last_token);
        }
        result.timings.total_seconds = r.prepare_seconds + seconds(r.submitted, now);
        slots.at(r.slot).request     = nullptr;
        complete(request, std::move(result));
    }

    void finish_now(const std::shared_ptr<Request>& request, FinishReason reason) {
        (void)request->output.preview_terminal(reason);
        push_events(*request, request->output.commit_preview(), std::nullopt);
        finish(request, reason);
    }

    void prefill_step(const std::shared_ptr<Request>& request) {
        Request& r          = *request;
        Slot& slot          = slots[r.slot];
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
            return;
        }
        // A prompt admitted with nothing to resume may find a prefix another request stored while
        // it waited (the shared system prompt of requests that arrived together): start there.
        if (reuse_prefixes && r.reusable && r.prefilled == 0) {
            const auto [image, from_store] = find_stored(r);
            if (from_store > 0) {
                restore_stored(r.slot, image);
                slot.fed     = stored[image].tokens;
                r.reuse_path = stored[image].path;
                r.reused = r.prefilled = from_store;
                std::lock_guard lock(stats_mutex);
                stats.reused_prompt_tokens += from_store;
            }
        }
        if (r.branches && r.prefilled == prompt_n) {
            branch_step(request);
            return;
        }
        const std::uint32_t n = prefill_step_tokens(r);
        if (n == 0) { throw std::logic_error("flint: an empty prefill step"); }
        const auto chunk = std::span<const TokenId>(r.prompt_tokens).subspan(r.prefilled, n);
        // A branched read's last chunk takes the first branches into its pass when the first fits
        // beside it: the sequence keeps the chunk's rows and the branches run from its end.
        std::vector<TokenId> branch_tokens;
        std::vector<std::uint32_t> lengths;
        if (r.branches && r.prefilled + n == prompt_n) {
            next_branches(r, executor.options().prefill_chunk - n, branch_tokens, lengths);
        }
        if (lengths.empty()) {
            executor.forward(r.slot, chunk, 1);
        } else {
            executor.forward_prompt_branches(r.slot, chunk, branch_tokens, lengths);
        }
        slot.fed.insert(slot.fed.end(), chunk.begin(), chunk.end());
        r.prefilled += n;
        r.progress = Clock::now();
        if (reuse_prefixes && r.reusable &&
            std::binary_search(r.shared_at.begin(), r.shared_at.end(), r.prefilled)) {
            store_image(r.slot, slot.fed, nullptr, PrefixReusePath::SharedStablePrefix);
        }
        if (reuse_prefixes && r.reusable && r.prefilled == r.anchor_at && r.prefilled < prompt_n) {
            executor.snapshot(r.slot, slot.snapshot);
            slot.anchor = slot.fed;
        }
        if (r.observation.prompt_progress) {
            CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
            executor.check_experts();
            {
                std::lock_guard lock(r.mutex);
                r.stream_progress =
                    PromptProgress{.total_prompt_tokens     = prompt_n,
                                   .reused_prompt_tokens    = r.reused,
                                   .processed_prompt_tokens = r.prefilled,
                                   .elapsed_ns = elapsed_ns(r.admitted, Clock::now())};
            }
            r.cv.notify_all();
        }
        if (r.prefilled < prompt_n) { return; }
        if (!lengths.empty()) {
            read_branches(request, lengths);
            return;
        }
        if (r.branches) { return; } // the next steps run the branches
        finish_prompt(request);
    }

    // One step of a branched read: the current branch's next tokens, from the snapshot at the
    // prompt's end when the branch begins, and once they are in, its first token read. When the
    // last branch is read, the request publishes the first branch's token and finishes.
    void branch_step(const std::shared_ptr<Request>& request) {
        Request& r                        = *request;
        Slot& slot                        = slots[r.slot];
        auto& executor                    = *instance.executor;
        const auto prompt_n               = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const runtime::ReadBranch& branch = (*r.branches)[r.branch_results.size()];
        // Branches that fit a prompt chunk run in shared passes, which leave the sequence at the
        // prompt's end (branch_pass); a longer one runs alone, chunk by chunk, from the snapshot
        // it leaves at the prompt's end.
        const std::size_t room = std::min<std::size_t>(executor.options().prefill_chunk,
                                                       executor.options().max_context - prompt_n);
        if (r.branch_fed == 0 && branch.tokens.size() <= room) {
            if (slot.fed != r.prompt_tokens) {
                if (slot.anchor != r.prompt_tokens) {
                    throw std::logic_error("a branched read lost the state at its prompt's end");
                }
                executor.restore(r.slot, slot.snapshot);
                slot.fed = slot.anchor;
            }
            branch_pass(request);
            return;
        }
        // A branch starts from the state at the prompt's end: the sequence's own at the first
        // branch, kept in its snapshot (the anchor) for the others.
        if (r.branch_fed == 0) {
            if (slot.fed == r.prompt_tokens) {
                if (slot.anchor != slot.fed) {
                    executor.snapshot(r.slot, slot.snapshot);
                    slot.anchor = slot.fed;
                }
            } else {
                if (slot.anchor != r.prompt_tokens) {
                    throw std::logic_error("a branched read lost the state at its prompt's end");
                }
                executor.restore(r.slot, slot.snapshot);
                slot.fed = slot.anchor;
            }
        }
        const std::uint32_t n = prefill_step_tokens(r);
        const auto chunk      = std::span<const TokenId>(branch.tokens).subspan(r.branch_fed, n);
        executor.forward(r.slot, chunk, 1);
        slot.fed.insert(slot.fed.end(), chunk.begin(), chunk.end());
        r.branch_fed += n;
        r.progress = Clock::now();
        if (r.branch_fed < branch.tokens.size()) { return; }
        SampleRow row = sample_row(r);
        row.readout   = branch.readout_tokens.empty() ? nullptr : &branch.readout_tokens;
        TokenId token = 0;
        runtime::RawTokenLogprob record;
        sample(std::span(&row, 1), ops::kSamplePurposePrefill, std::span(&token, 1),
               std::span(&record, 1));
        if (r.branch_results.empty()) {
            r.branch_token  = token;
            r.branch_record = record;
        }
        r.branch_results.push_back(BranchResult{
            .prompt_tokens = prompt_n + static_cast<std::uint32_t>(branch.tokens.size()),
            .first         = instance.frontend.token_logprob(record)});
        r.branch_fed = 0;
        if (r.branch_results.size() < r.branches->size()) { return; }
        finish_prompt(request);
    }

    // The read's next branches that fit `room` tokens of a pass together; the branches take cache
    // slots one after another past the prompt.
    void next_branches(const Request& r, std::size_t room, std::vector<TokenId>& tokens,
                       std::vector<std::uint32_t>& lengths) const {
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const std::vector<runtime::ReadBranch>& branches = *r.branches;
        room = std::min<std::size_t>(room, instance.executor->options().max_context - prompt_n);
        for (std::size_t b = r.branch_results.size();
             b < branches.size() && lengths.size() < models::flint::Executor::kMaxLogitRows &&
             tokens.size() + branches[b].tokens.size() <= room;
             ++b) {
            tokens.insert(tokens.end(), branches[b].tokens.begin(), branches[b].tokens.end());
            lengths.push_back(static_cast<std::uint32_t>(branches[b].tokens.size()));
        }
    }

    // The next branches of a branched read that fit one prompt chunk together, in one pass from
    // the sequence at the prompt's end (Executor::forward_branches), which leaves it there.
    void branch_pass(const std::shared_ptr<Request>& request) {
        Request& r     = *request;
        auto& executor = *instance.executor;
        std::vector<TokenId> tokens;
        std::vector<std::uint32_t> lengths;
        next_branches(r, executor.options().prefill_chunk, tokens, lengths);
        executor.forward_branches(r.slot, tokens, lengths);
        r.progress = Clock::now();
        read_branches(request, lengths);
    }

    // The first tokens of the branches a pass just ran (`lengths`, the read's next ones), a lane's
    // worth of rows at a time; the request finishes after its last branch.
    void read_branches(const std::shared_ptr<Request>& request,
                       std::span<const std::uint32_t> lengths) {
        Request& r          = *request;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        const std::vector<runtime::ReadBranch>& branches = *r.branches;
        const std::size_t first                          = r.branch_results.size();
        const Tensor logits =
            instance.executor->logits(static_cast<std::uint32_t>(lengths.size()));
        const std::size_t width = slots.size();
        for (std::size_t at = 0; at < lengths.size(); at += width) {
            const std::size_t n = std::min(width, lengths.size() - at);
            std::vector<SampleRow> rows(n, sample_row(r));
            for (std::size_t k = 0; k < n; ++k) {
                const runtime::ReadBranch& branch = branches[first + at + k];
                rows[k].readout = branch.readout_tokens.empty() ? nullptr : &branch.readout_tokens;
            }
            std::vector<TokenId> sampled(n);
            std::vector<runtime::RawTokenLogprob> records(n);
            const Tensor source =
                logits.slice(1, static_cast<std::int32_t>(at), static_cast<std::int32_t>(n));
            sample(rows, ops::kSamplePurposePrefill, sampled, records, &source);
            for (std::size_t k = 0; k < n; ++k) {
                if (r.branch_results.empty()) {
                    r.branch_token  = sampled[k];
                    r.branch_record = records[k];
                }
                const auto own = static_cast<std::uint32_t>(branches[first + at + k].tokens.size());
                r.branch_results.push_back(
                    BranchResult{.prompt_tokens = prompt_n + own,
                                 .first         = instance.frontend.token_logprob(records[k])});
            }
        }
        if (r.branch_results.size() < branches.size()) { return; }
        finish_prompt(request);
    }

    // The prompt is in (and a branched read's branches): the request takes its first token and
    // decodes from there.
    void finish_prompt(const std::shared_ptr<Request>& request) {
        Request& r          = *request;
        auto& executor      = *instance.executor;
        const auto prompt_n = static_cast<std::uint32_t>(r.prompt_tokens.size());
        CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
        executor.check_experts();
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
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts.p) +
                                           std::size_t(r.slot) * domain,
                                       0, std::size_t(domain) * sizeof(std::int32_t),
                                       executor.head_stream()));
        }
        r.position = prompt_n;
        r.decoding = true;
        // The first token's mask began with the request; when it is not done yet, the prompt's
        // last logits wait for it instead of the engine thread.
        if (r.branches) {
            accept(request, r.branch_token, &r.branch_record);
            return;
        }
        if (!mask_ready(r)) {
            defer_sample(r, executor.logits(1), 0, ops::kSamplePurposePrefill);
            return;
        }
        if (!mask_usable(request)) { return; }
        const SampleRow row = sample_row(r);
        TokenId token       = 0;
        runtime::RawTokenLogprob logprob;
        sample(std::span(&row, 1), ops::kSamplePurposePrefill, std::span(&token, 1),
               std::span(&logprob, 1));
        accept(request, token, row.logprobs ? &logprob : nullptr);
        next_mask(request);
    }

    [[nodiscard]] bool active(const std::shared_ptr<Request>& request) const {
        return slots.at(request->slot).request == request;
    }

    // Applies the output policy to a sampled token: publishes what it accepts, then either
    // finishes the request or queues what its next decode step feeds.
    void accept(const std::shared_ptr<Request>& request, TokenId token,
                const runtime::RawTokenLogprob* logprob) {
        (void)accept(request, std::span<const TokenId>(&token, 1),
                     logprob != nullptr ? std::span<const runtime::RawTokenLogprob>(logprob, 1)
                                        : std::span<const runtime::RawTokenLogprob>{});
        check_capacity(request);
    }

    // A request whose next feed would pass the context finishes.
    void check_capacity(const std::shared_ptr<Request>& request) {
        if (active(request) && request->position + request->feed.size() > max_context) {
            finish_now(request, FinishReason::ContextCapacity);
        }
    }

    // The same for a run of sampled tokens (a speculative round's licensed tokens), without the
    // capacity check, which the caller makes once the round's position is known: returns how many
    // of them the policy accepted. The next step feeds the last accepted one (the first when none
    // was, as a single sampled token is fed whatever the policy decides).
    std::uint32_t accept(const std::shared_ptr<Request>& request, std::span<const TokenId> tokens,
                         std::span<const runtime::RawTokenLogprob> logprobs) {
        Request& r                    = *request;
        const OutputDecision decision = r.output.preview_model(tokens, r.budget->remaining(),
                                                               r.budget->limit_reason(), logprobs);
        const auto now = Clock::now();
        if (r.first_token == Clock::time_point{}) { r.first_token = now; }
        r.last_token = now;
        r.progress   = now;
        if (decision.accepted_tokens > tokens.size()) {
            throw std::logic_error("output policy accepted more than the sampled tokens");
        }
        const std::uint32_t accepted = decision.accepted_tokens;
        if (accepted > 0) {
            r.generated.insert(r.generated.end(), tokens.begin(), tokens.begin() + accepted);
            r.budget->commit(accepted);
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
            stats.committed_decode_tokens += accepted;
        }
        if (decision.finished()) {
            finish(request, decision.finish_reason);
            return accepted;
        }
        r.feed.assign(1, tokens[accepted > 0 ? accepted - 1 : 0]);
        if (r.cancelled.load(std::memory_order_acquire)) {
            finish_now(request, FinishReason::Cancelled);
        }
        return accepted;
    }

    // Whether the request's next step can be a speculative round: an Engine with drafts, room for
    // the round in the context and the output, and no switch of sampling parameters pending (a
    // reasoning block that has not closed yet when the request samples its answer differently).
    [[nodiscard]] bool speculates(const Request& r) const {
        // A structured row speculates while its masks are cheap enough for the round to fill its
        // columns during the verification; otherwise it decodes plainly, a mask ahead.
        return drafts > 0 && (!r.constrained || r.mask_cost.speculates()) && r.feed.size() == 1 &&
               r.position + drafts + 1 <= max_context &&
               r.output.model_token_budget_remaining(r.budget->remaining()) > 1 &&
               !(r.options.execution.post_thinking_sampling && !r.post_thinking);
    }

    // The drafts a round takes while `decoding` requests decode at once. Each verified row routes to
    // its own experts, so the rows a round adds cost more as requests join; Gemma 4's measurement of
    // this on an RTX 3090 (two requests gain with one draft each and lose with three, four lose with
    // one) sets the same bounds here.
    [[nodiscard]] std::uint32_t round_drafts(std::size_t decoding) const noexcept {
        return decoding == 1 ? drafts : decoding <= 3 ? 1U : 0U;
    }

    // Context lookup (--lookup-ngram N): when the request's last N tokens, its anchor last, appeared
    // earlier in its sequence, up to `steps` tokens that followed then; verification keeps the
    // output exact. The count proposed, 0 without a match.
    std::uint32_t lookup(const Request& r, std::uint32_t steps, TokenId* out) {
        std::vector<TokenId>& ledger = slots[r.slot].fed;
        ledger.push_back(r.feed.front());
        const std::uint32_t found = runtime::lookup_draft(ledger, lookup_ngram, steps, out);
        ledger.pop_back();
        return found;
    }

    // One step of every decoding request. Those whose context proposes a continuation run one
    // speculative round together, the rest one batched decode step.
    void decode_step() {
        std::vector<std::shared_ptr<Request>> batch, speculative;
        std::vector<std::vector<TokenId>> proposals;
        std::size_t decoding = 0;
        for (const Slot& slot : slots) {
            if (slot.request && slot.request->decoding) { ++decoding; }
        }
        const std::uint32_t steps = round_drafts(decoding);
        std::vector<std::shared_ptr<Request>> deferred;
        for (const Slot& slot : slots) {
            // A copy: finishing or failing the request clears its slot.
            const std::shared_ptr<Request> request = slot.request;
            if (!request || !request->decoding) { continue; }
            if (request->awaiting_sample) {
                // Its logits wait for its grammar mask; a cancellation need not.
                if (mask_ready(*request)) {
                    if (mask_usable(request)) { deferred.push_back(request); }
                } else if (request->cancelled.load(std::memory_order_acquire)) {
                    finish_now(request, FinishReason::Cancelled);
                }
                continue;
            }
            if (steps > 0 && speculates(*request)) {
                std::vector<TokenId> proposal(steps);
                if (const std::uint32_t found = lookup(*request, steps, proposal.data())) {
                    proposal.resize(found);
                    proposals.push_back(std::move(proposal));
                    speculative.push_back(request);
                    continue;
                }
            }
            batch.push_back(request);
        }
        if (batch.empty() && speculative.empty() && deferred.empty()) { return; }
        const auto start = Clock::now();
        if (!deferred.empty()) { sample_deferred(deferred); }
        if (!batch.empty()) { plain_step(batch); }
        if (!speculative.empty()) { speculative_round(speculative, proposals); }
        std::lock_guard lock(stats_mutex);
        stats.decode_rounds += 1;
        stats.decode_row_rounds += batch.size() + speculative.size();
        stats.decode_seconds_total += seconds(start, Clock::now());
    }

    // Keeps logits row `row` of `source` for the request until its grammar mask is ready.
    void defer_sample(Request& r, const Tensor& source, std::size_t row, std::int32_t purpose) {
        const std::size_t bytes = vocab_rows * sizeof(__nv_bfloat16);
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(saved_logits.p) + std::size_t(r.slot) * bytes,
                                   static_cast<const std::uint8_t*>(source.data) + row * bytes, bytes,
                                   cudaMemcpyDeviceToDevice, instance.executor->head_stream()));
        r.awaiting_sample  = true;
        r.awaiting_purpose = purpose;
    }

    // Samples requests whose kept logits now have their masks.
    void sample_deferred(const std::vector<std::shared_ptr<Request>>& ready) {
        const std::size_t bytes = vocab_rows * sizeof(__nv_bfloat16);
        for (const std::int32_t purpose : {ops::kSamplePurposePrefill, ops::kSamplePurposeDecode}) {
            std::vector<std::shared_ptr<Request>> group;
            for (const auto& request : ready) {
                if (request->awaiting_purpose == purpose) { group.push_back(request); }
            }
            if (group.empty()) { continue; }
            std::vector<SampleRow> rows;
            for (std::size_t j = 0; j < group.size(); ++j) {
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint8_t*>(deferred_logits.p) + j * bytes,
                    static_cast<const std::uint8_t*>(saved_logits.p) + std::size_t(group[j]->slot) * bytes,
                    bytes, cudaMemcpyDeviceToDevice, instance.executor->head_stream()));
                rows.push_back(sample_row(*group[j]));
            }
            const Tensor source(deferred_logits.p, DType::BF16,
                                {static_cast<std::int32_t>(vocab_rows), static_cast<std::int32_t>(group.size())});
            std::vector<TokenId> sampled(group.size());
            std::vector<runtime::RawTokenLogprob> logprobs(group.size());
            sample(rows, purpose, sampled, logprobs, &source);
            for (std::size_t j = 0; j < group.size(); ++j) {
                group[j]->awaiting_sample = false;
                if (purpose == ops::kSamplePurposeDecode && drafts > 0) {
                    ++group[j]->speculative.fallback_steps;
                }
                try {
                    accept(group[j], sampled[j], rows[j].logprobs ? &logprobs[j] : nullptr);
                    next_mask(group[j]);
                } catch (...) { fail(group[j], std::current_exception()); }
            }
        }
    }

    void plain_step(const std::vector<std::shared_ptr<Request>>& batch) {
        auto& executor = *instance.executor;
        std::vector<std::uint32_t> sequences;
        std::vector<TokenId> tokens;
        for (const auto& request : batch) {
            sequences.push_back(request->slot);
            tokens.push_back(request->feed.front());
            // A structured row without its mask yet starts it now, to fill while the step runs.
            if (request->constrained && !request->mask) { schedule_mask(*request); }
        }
        executor.decode(sequences, tokens);
        // A grammar row whose mask is still being computed keeps its logits for later; the decode
        // step itself did not wait for the mask.
        const Tensor logits = executor.logits(static_cast<std::uint32_t>(batch.size()));
        std::vector<std::shared_ptr<Request>> now;
        std::vector<std::size_t> columns;
        for (std::size_t b = 0; b < batch.size(); ++b) {
            const auto& request = batch[b];
            slots[request->slot].fed.push_back(request->feed.front());
            request->position += 1;
            if (mask_ready(*request)) {
                now.push_back(request);
                columns.push_back(b);
            } else {
                defer_sample(*request, logits, b, ops::kSamplePurposeDecode);
            }
        }
        std::vector<SampleRow> rows;
        for (const auto& request : now) { rows.push_back(sample_row(*request)); }
        std::optional<Tensor> source;
        if (now.size() != batch.size() && !now.empty()) {
            const std::size_t bytes = vocab_rows * sizeof(__nv_bfloat16);
            for (std::size_t j = 0; j < now.size(); ++j) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(deferred_logits.p) + j * bytes,
                                           static_cast<const std::uint8_t*>(logits.data) + columns[j] * bytes,
                                           bytes, cudaMemcpyDeviceToDevice, executor.head_stream()));
            }
            source.emplace(Tensor(deferred_logits.p, DType::BF16,
                                  {static_cast<std::int32_t>(vocab_rows), static_cast<std::int32_t>(now.size())}));
        }
        std::vector<TokenId> sampled(now.size());
        std::vector<runtime::RawTokenLogprob> logprobs(now.size());
        if (!now.empty()) {
            sample(rows, ops::kSamplePurposeDecode, sampled, logprobs, source ? &*source : nullptr);
        }
        for (std::size_t b = 0; b < now.size(); ++b) {
            if (drafts > 0) { ++now[b]->speculative.fallback_steps; }
            try {
                if (!mask_usable(now[b])) { continue; }
                accept(now[b], sampled[b], rows[b].logprobs ? &logprobs[b] : nullptr);
                next_mask(now[b]);
            } catch (...) { fail(now[b], std::current_exception()); }
        }
    }

    // One round of each request: the target verifies the anchor and its context's proposal at once,
    // the acceptance licenses a run of tokens on the device (the accepted drafts and a correction or
    // bonus, each the target's own keyed draw, so the output is what plain decoding samples), the
    // output policy takes what it accepts of the run, and the executor commits the tokens fed up to
    // the last of them.
    void speculative_round(const std::vector<std::shared_ptr<Request>>& batch,
                           const std::vector<std::vector<TokenId>>& proposals) {
        auto& executor      = *instance.executor;
        const std::size_t b = batch.size();
        std::vector<std::uint32_t> sequences;
        std::vector<TokenId> anchors;
        for (const auto& request : batch) {
            sequences.push_back(request->slot);
            anchors.push_back(request->feed.front());
        }
        std::vector<TokenId> all_drafts(b * drafts);
        std::vector<std::uint32_t> draft_extents(b);
        for (std::size_t j = 0; j < b; ++j) {
            draft_extents[j] = static_cast<std::uint32_t>(proposals[j].size());
            std::copy(proposals[j].begin(), proposals[j].end(),
                      all_drafts.begin() + std::ptrdiff_t(j * drafts));
        }
        const std::size_t k = *std::max_element(draft_extents.begin(), draft_extents.end());
        const std::size_t w = k + 1;
        const auto columns  = static_cast<std::int32_t>(w);
        std::vector<TokenId> proposed;
        proposed.reserve(b * k);
        for (std::size_t j = 0; j < b; ++j) {
            proposed.insert(proposed.end(), all_drafts.begin() + std::ptrdiff_t(j * drafts),
                            all_drafts.begin() + std::ptrdiff_t(j * drafts + k));
        }
        std::vector<TokenId> tokens;
        for (std::size_t j = 0; j < b; ++j) {
            tokens.push_back(anchors[j]);
            tokens.insert(tokens.end(), proposed.begin() + std::ptrdiff_t(j * k),
                          proposed.begin() + std::ptrdiff_t((j + 1) * k));
        }
        // A structured row's masks, one per column after its own drafts, fill in parallel while
        // the verification runs.
        std::vector<std::shared_ptr<runtime::GrammarMaskJob>> column_masks(b);
        for (std::size_t j = 0; j < b; ++j) {
            if (!batch[j]->constrained) { continue; }
            const std::size_t own = std::min<std::size_t>(draft_extents[j], k);
            column_masks[j] = runtime::GrammarMaskJob::start(
                *batch[j]->output.grammar_state(), mask_words(),
                std::vector<TokenId>(proposed.begin() + std::ptrdiff_t(j * k),
                                     proposed.begin() + std::ptrdiff_t(j * k + own)));
        }
        executor.verify(sequences, tokens);

        const cudaStream_t stream = executor.head_stream();
        std::vector<SampleRow> rows;
        for (const auto& request : batch) { rows.push_back(sample_row(*request)); }
        // The rows' sampling, each column of a structured row masked as its drafts before it leave
        // the grammar. A row whose grammar cannot serve samples unmasked and fails after the round.
        auto* configs = static_cast<ops::SamplingConfig*>(host_sample->data());
        std::vector<std::string> mask_failures(b);
        bool masked = false;
        for (std::size_t j = 0; j < b; ++j) {
            const SampleRow& row = rows[j];
            ops::SamplingConfig config;
            config.temperature       = row.params->temperature;
            config.top_k             = row.params->top_k;
            config.top_p             = row.params->top_p;
            config.min_p             = row.params->min_p;
            config.presence_penalty  = row.params->presence_penalty;
            config.frequency_penalty = row.params->frequency_penalty;
            config.seed              = row.params->seed;
            config.token_counts      = row.counts ? static_cast<std::int32_t*>(token_counts.p) +
                                                        std::size_t(row.slot) * domain
                                                  : nullptr;
            if (const auto& job = column_masks[j]) {
                job->wait();
                batch[j]->mask_cost.observe(job->seconds());
                mask_failures[j] = runtime::grammar_mask_problem(*job);
                if (mask_failures[j].empty()) {
                    // Columns past the row's own drafts are never taken; they stay unrestricted.
                    auto* words = static_cast<std::uint32_t*>(host_mask->data()) + j * w * mask_words();
                    std::copy_n(job->masks(), job->columns() * mask_words(), words);
                    std::fill(words + job->columns() * mask_words(), words + w * mask_words(),
                              ~std::uint32_t{0});
                    config.token_mask =
                        static_cast<const std::uint32_t*>(token_mask.p) + j * w * mask_words();
                    config.token_mask_stride = static_cast<std::int32_t>(mask_words());
                    masked                   = true;
                }
            }
            configs[j] = config;
        }
        if (masked) {
            CUDA_CHECK(cudaMemcpyAsync(token_mask.p, host_mask->data(),
                                       b * w * mask_words() * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(sample_config.p, configs, b * sizeof(ops::SamplingConfig),
                                   cudaMemcpyHostToDevice, stream));
        // Drafts, live draft counts and the tokens each request fed before the round.
        auto* staged = static_cast<std::int32_t*>(host_spec->data());
        std::copy(proposed.begin(), proposed.end(), staged);
        for (std::size_t j = 0; j < b; ++j) {
            // Acceptance respects each sequence's own extent and its output budget, correction
            // included.
            const auto remaining = batch[j]->output.model_token_budget_remaining(
                batch[j]->budget->remaining());
            staged[b * k + j] = static_cast<std::int32_t>(
                std::min<std::size_t>(draft_extents[j], remaining - 1));
            staged[b * k + b + j] = static_cast<std::int32_t>(batch[j]->position);
        }
        CUDA_CHECK(
            cudaMemcpyAsync(spec_drafts.p, staged, b * k * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(spec_extents.p, staged + b * k, b * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(spec_lengths.p, staged + b * k + b, b * 4,
                                   cudaMemcpyHostToDevice, stream));
        const auto rows_n         = static_cast<std::int32_t>(b);
        const Tensor logits       = executor.logits(static_cast<std::uint32_t>(b * w));
        const auto* device_config = static_cast<const ops::SamplingConfig*>(sample_config.p);
        const Tensor drafted(spec_drafts.p, DType::I32, {static_cast<std::int32_t>(k), rows_n});
        // The tokens' logprob records, from the distributions their columns are drawn from,
        // before the acceptance adds the round's tokens to the penalty counts.
        const bool gather     = std::any_of(rows.begin(), rows.end(),
                                            [](const SampleRow& row) { return row.logprobs; });
        const std::size_t top = b * w * kMaximumTokenLogprobs;
        auto* host_ids        = static_cast<std::int32_t*>(host_logprobs->data());
        auto* host_values     = reinterpret_cast<float*>(
            host_ids + slots.size() * (std::size_t(drafts) + 1) * kMaximumTokenLogprobs);
        if (gather) {
            Tensor ids(logprob_ids.p, DType::I32, {ops::kLogprobTopK, columns, rows_n});
            Tensor values(logprob_values.p, DType::FP32, {ops::kLogprobTopK, columns, rows_n});
            Tensor lse(logprob_lse.p, DType::FP32, {columns, rows_n});
            const Tensor flag(logprob_flag.p, DType::I32, {1});
            ops::logprob_topk(logits.view({logits.ne[0], columns, rows_n}), device_config, &drafted,
                              static_cast<std::int32_t>(domain), ids, values, lse, flag,
                              *sample_workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(host_ids, logprob_ids.p, top * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_values, logprob_values.p, top * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        for (std::size_t j = 0; j < b; ++j) {
            if (!rows[j].counts) { continue; }
            const std::size_t at = std::size_t(rows[j].slot) * domain * sizeof(std::int32_t);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(counts_backup.p) + at,
                                       static_cast<const std::byte*>(token_counts.p) + at,
                                       std::size_t(domain) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        Tensor targets(spec_targets.p, DType::I32, {columns * rows_n});
        ops::argmax(logits, targets, static_cast<std::int32_t>(domain), stream);
        Tensor extents(spec_extents.p, DType::I32, {rows_n});
        Tensor lengths(spec_lengths.p, DType::I32, {rows_n});
        Tensor round_anchors(spec_anchors.p, DType::I32, {rows_n});
        Tensor licensed(spec_licensed.p, DType::I32, {columns, rows_n});
        Tensor counts(spec_counts.p, DType::I32, {rows_n});
        Tensor accepted(spec_accepted.p, DType::I32, {rows_n});
        {
            auto scope = sample_workspace->scope();
            ops::speculative_accept_coupled_drafts(
                targets.view({columns, rows_n}), logits.view({logits.ne[0], columns, rows_n}),
                drafted, extents, lengths, round_anchors, licensed, counts, accepted,
                static_cast<std::int32_t>(domain), device_config, *sample_workspace, stream);
        }
        auto* host = staged + b * (k + 2);
        CUDA_CHECK(
            cudaMemcpyAsync(host, spec_licensed.p, b * w * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(host + b * w, spec_counts.p, b * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host + b * w + b, spec_accepted.p, b * 4, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        instance.executor->check_experts();

        // The output policy takes its part of each licensed run; the executor then keeps the tokens
        // fed up to the last one taken.
        std::vector<std::uint32_t> kept(b, 1);
        std::vector<char> failed(b, 0);
        for (std::size_t j = 0; j < b; ++j) {
            const auto& request           = batch[j];
            Request& r                    = *request;
            const std::int32_t licensed_n = host[b * w + j];
            const std::int32_t drafts_n   = host[b * w + b + j];
            const std::int32_t* run       = host + j * w;
            if (licensed_n < 1 || licensed_n > columns || run[0] == ops::kSamplerNonFiniteToken) {
                throw std::runtime_error("Flint produced non-finite logits");
            }
            for (std::int32_t i = 0; i < licensed_n; ++i) {
                if (run[i] < 0 || std::uint32_t(run[i]) >= domain) {
                    throw std::runtime_error("Flint licensed a token outside the vocabulary");
                }
            }
            std::vector<runtime::RawTokenLogprob> records;
            if (rows[j].logprobs) {
                for (std::int32_t i = 0; i < licensed_n; ++i) {
                    runtime::RawTokenLogprob record{.id = run[i], .logprob = kLogprobSentinel};
                    for (std::size_t t = 0; t < kMaximumTokenLogprobs; ++t) {
                        const std::size_t at = (j * w + std::size_t(i)) * kMaximumTokenLogprobs + t;
                        record.top_ids[t]    = host_ids[at];
                        record.top_values[t] = host_values[at];
                        if (host_ids[at] == run[i]) { record.logprob = host_values[at]; }
                    }
                    records.push_back(record);
                }
            }
            r.speculative.rounds += 1;
            r.speculative.drafted_tokens += draft_extents[j];
            r.speculative.accepted_tokens += std::uint32_t(drafts_n);
            // Reported with the n-gram proposals as well: the context, not a model, proposed them.
            ++r.speculative.ngram_rounds;
            r.speculative.ngram_drafted_tokens += draft_extents[j];
            r.speculative.ngram_accepted_tokens += std::uint32_t(drafts_n);
            for (std::int32_t i = 0; i < drafts_n; ++i) {
                ++r.speculative.accepted_per_position[i];
            }
            const std::span<const TokenId> run_tokens(run, std::size_t(licensed_n));
            std::uint32_t taken = 0;
            if (!mask_failures[j].empty()) {
                fail(request, std::make_exception_ptr(RequestError(
                                  RequestErrorKind::Unavailable, "structured output: " + mask_failures[j])));
                failed[j] = 1;
            } else {
                try {
                    taken = accept(request, run_tokens, records);
                } catch (...) {
                    fail(request, std::current_exception());
                    failed[j] = 1;
                }
            }
            // The tokens counted as sampled: those taken, or the first when none was.
            kept[j] = std::max<std::uint32_t>(taken, 1);
            if (kept[j] < std::uint32_t(licensed_n) && active(request) && rows[j].counts) {
                // The acceptance counted the whole run; the request keeps only its prefix.
                const std::size_t at = std::size_t(r.slot) * domain * sizeof(std::int32_t);
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(token_counts.p) + at,
                                           static_cast<const std::byte*>(counts_backup.p) + at,
                                           std::size_t(domain) * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToDevice, stream));
                const Tensor ids(static_cast<std::int32_t*>(spec_licensed.p) + j * w, DType::I32,
                                 {static_cast<std::int32_t>(kept[j])});
                Tensor row_counts(static_cast<std::int32_t*>(token_counts.p) +
                                      std::size_t(r.slot) * domain,
                                  DType::I32, {static_cast<std::int32_t>(domain)});
                ops::increment_token_counts(ids, row_counts, stream);
            }
        }
        executor.commit(sequences, kept);
        for (std::size_t j = 0; j < b; ++j) {
            Request& r = *batch[j];
            if (failed[j]) { continue; }
            Slot& slot = slots[r.slot];
            // The verification fed the anchor and the drafts; the sequence keeps the first ones.
            slot.fed.insert(slot.fed.end(), tokens.begin() + std::ptrdiff_t(j * w),
                            tokens.begin() + std::ptrdiff_t(j * w + kept[j]));
            r.position += kept[j];
            try {
                check_capacity(batch[j]);
            } catch (...) { fail(batch[j], std::current_exception()); }
            if (r.constrained) { next_mask(batch[j]); }
        }
    }

    std::vector<float> score(const std::vector<TokenId>& window, std::uint32_t window_target) {
        auto& executor = *instance.executor;
        if (window.size() < 2 || window_target == 0 || window_target >= window.size() ||
            window.size() > max_context) {
            throw std::invalid_argument("score: invalid token window");
        }
        // Flint reads a sequence as it is: its prompts open with no <bos>.
        const std::vector<TokenId>& tokens = window;
        const std::uint32_t first_target   = window_target;
        const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
        const std::uint32_t chunk = std::min<std::uint32_t>(
            executor.options().prefill_chunk, models::flint::Executor::kMaxLogitRows);
        executor.reset(0);
        slots[0].fed.clear();
        slots[0].anchor.clear();
        std::vector<float> out;
        out.reserve(n - first_target);
        for (std::uint32_t at = 0; at + 1 < n; at += chunk) {
            const std::uint32_t count = std::min(chunk, n - 1 - at);
            executor.forward(0, std::span<const TokenId>(tokens).subspan(at, count), count);
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
            instance.executor->check_experts();
            out.insert(out.end(), host.begin(), host.end());
        }
        save_expert_profile();
        return out;
    }
};

FlintCore::Submission::Submission(FlintCore& owner, std::shared_ptr<Request> request,
                                   std::optional<std::uint32_t> budget) noexcept
    : owner_(&owner), request_(std::move(request)), effective_thinking_budget_(budget) {}

FlintCore::Submission::~Submission() {
    if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
}

FlintCore::Submission::Submission(Submission&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)),
      effective_thinking_budget_(other.effective_thinking_budget_) {}

FlintCore::Submission& FlintCore::Submission::operator=(Submission&& other) noexcept {
    if (this != &other) {
        if (request_ != nullptr) { request_->cancelled.store(true, std::memory_order_release); }
        owner_                     = std::exchange(other.owner_, nullptr);
        request_                   = std::move(other.request_);
        effective_thinking_budget_ = other.effective_thinking_budget_;
    }
    return *this;
}

GenerationResult FlintCore::Submission::wait(OutputSink* sink,
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

FlintCore::FlintCore(FlintInstance& instance, DeviceContext& device,
                       const EngineOptions& options)
    : impl_(std::make_unique<Impl>(instance, device, options)) {
    if (options.max_concurrency == 0 || options.max_pending_requests == 0 ||
        options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine core bounds are invalid");
    }
}

FlintCore::~FlintCore() = default;

void FlintCore::stop() noexcept {
    {
        std::lock_guard lock(impl_->queue_mutex);
        impl_->stopping = true;
    }
    impl_->queue_cv.notify_all();
}

std::shared_ptr<FlintCore::Request>
FlintCore::make_request(GroupMember& member, std::optional<std::uint32_t>& budget_out) {
    models::qwen3_5::PreparedPrompt& prompt         = member.prompt;
    const PromptSummary& prompt_summary             = member.prompt_summary;
    const double prepare_seconds                    = member.prepare_seconds;
    ResolvedRequestOptions& options                 = member.options;
    const OutputConsumerMode consumer_mode          = member.consumer_mode;
    const GenerationObservationOptions& observation = member.observation;
    Clock::time_point& pending_deadline             = member.pending_deadline;
    const auto submitted                            = Clock::now();
    for (const TokenId id : options.execution.readout_tokens) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= impl_->domain) {
            throw std::invalid_argument("readout token is outside the public token domain");
        }
    }
    if (const auto& branches = options.execution.branches) {
        if (branches->empty() || options.execution.grammar ||
            prompt.token_ids().size() + 1 > impl_->max_context) {
            throw std::invalid_argument("a branched read needs branches and a text prompt");
        }
        for (const runtime::ReadBranch& branch : *branches) {
            if (branch.tokens.empty() ||
                prompt.token_ids().size() + branch.tokens.size() > impl_->max_context) {
                throw std::invalid_argument("a branch needs tokens within the context");
            }
            for (const TokenId id : branch.readout_tokens) {
                if (id < 0 || static_cast<std::uint32_t>(id) >= impl_->domain) {
                    throw std::invalid_argument("readout token is outside the public token domain");
                }
            }
        }
    }
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
        if (data.has_media()) { throw std::invalid_argument("Flint reads text prompts only"); }
        const auto prompt_n  = static_cast<std::uint32_t>(request->prompt_tokens.size());
        request->reusable   = identity.reusable;
        request->disposable = data.tap_hints.disposable;
        request->anchor_at  = prompt_n;
        if (identity.rewrite_checkpoint &&
            identity.rewrite_checkpoint->kind ==
                models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
            identity.rewrite_checkpoint->frontier > 0 &&
            identity.rewrite_checkpoint->frontier < prompt_n) {
            request->anchor_at = identity.rewrite_checkpoint->frontier;
        }
        // Shared prefixes worth an image of their own: a structural or declared boundary, not one
        // the engine merely observed (the prompt's end, a grid point).
        for (const auto& opportunity : data.context_cache.opportunities) {
            const auto evidence   = opportunity.evidence;
            const bool structural = has_shared_candidate_evidence(
                evidence, SharedCandidateEvidence::ExplicitBoundary |
                              SharedCandidateEvidence::RequestedAutomatic |
                              SharedCandidateEvidence::DefaultAutomatic |
                              SharedCandidateEvidence::EngineStructural);
            const std::uint32_t frontier = opportunity.frontier;
            if (opportunity.kind == PromptCacheMarkerKind::SharedStablePrefix && structural &&
                frontier > 0 && frontier < request->anchor_at && impl_->image_pays(frontier)) {
                request->shared_at.push_back(frontier);
            }
        }
        std::sort(request->shared_at.begin(), request->shared_at.end());
        request->shared_at.erase(std::unique(request->shared_at.begin(), request->shared_at.end()),
                                 request->shared_at.end());
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
        options.execution.structured_output);
    // Strict tool arguments bring a grammar without a structured final format.
    if (request->output.grammar_state() && !impl_->structured_output) {
        throw std::invalid_argument(
            "strict tool arguments require an Engine started with structured_output");
    }
    const std::optional<std::uint32_t> budget = request->output.thinking_stats().effective_budget;
    request->prompt                           = std::move(prompt);
    request->options                          = std::move(options);
    request->branches                         = request->options.execution.branches;
    budget_out                                = budget;
    return request;
}

FlintCore::Submission FlintCore::submit(models::qwen3_5::PreparedPrompt prompt,
                                          PromptSummary prompt_summary, double prepare_seconds,
                                          ResolvedRequestOptions options,
                                          OutputConsumerMode consumer_mode,
                                          GenerationObservationOptions observation,
                                          Clock::time_point pending_deadline) {
    std::vector<GroupMember> one;
    one.push_back(GroupMember{.prompt           = std::move(prompt),
                              .prompt_summary   = prompt_summary,
                              .prepare_seconds  = prepare_seconds,
                              .options          = std::move(options),
                              .consumer_mode    = consumer_mode,
                              .observation      = observation,
                              .pending_deadline = pending_deadline});
    return std::move(submit_group(std::move(one)).front());
}

std::vector<FlintCore::Submission> FlintCore::submit_group(std::vector<GroupMember> group) {
    std::vector<std::shared_ptr<Request>> requests;
    std::vector<std::optional<std::uint32_t>> budgets(group.size());
    requests.reserve(group.size());
    for (std::size_t k = 0; k < group.size(); ++k) {
        requests.push_back(make_request(group[k], budgets[k]));
    }
    if (requests.empty()) { return {}; }
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding + requests.size() > impl_->max_outstanding) {
            throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
        }
        impl_->outstanding += requests.size();
        for (const auto& request : requests) {
            request->id = impl_->next_id++;
            impl_->pending.push_back(request);
        }
    }
    impl_->queue_cv.notify_one();
    impl_->publish_queue();
    std::vector<Submission> submissions;
    submissions.reserve(requests.size());
    for (std::size_t k = 0; k < requests.size(); ++k) {
        submissions.push_back(Submission(*this, std::move(requests[k]), budgets[k]));
    }
    return submissions;
}

std::vector<float> FlintCore::score(models::qwen3_5::PreparedPrompt prompt,
                                     std::uint32_t first_target) {
    if (models::qwen3_5::PreparedPromptAccess::view(prompt).has_media()) {
        throw std::invalid_argument("Flint scores text prompts only");
    }
    const std::vector<TokenId> tokens(prompt.token_ids().begin(), prompt.token_ids().end());
    {
        std::lock_guard lock(impl_->queue_mutex);
        if (impl_->stopping || impl_->failed) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable");
        }
        if (impl_->outstanding != 0) {
            throw std::logic_error("score requires an idle Flint Engine");
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

MemorySummary FlintCore::memory_summary() const {
    const auto& stats = impl_->instance.model->storage_stats();
    const auto memory = impl_->instance.executor->memory();
    const auto arena  = [](std::uint64_t bytes) { return ArenaMemorySummary{bytes, bytes, bytes}; };
    MemorySummary out;
    out.device      = impl_->device.rank(0).device;
    out.max_context = impl_->max_context;
    out.kv_capacity = impl_->max_context;
    out.kv_cache    = KvCacheStorage::BFloat16; // F16 rows of the same width
    out.host_kv_capacity_bytes = impl_->host_budget;
    {
        std::lock_guard lock(impl_->stats_mutex);
        out.host_kv_occupied_bytes = impl_->stats.host_kv_occupied_bytes;
    }
    out.weights   = arena(stats.device_capacity_by_rank.empty() ? 0 : stats.device_capacity_by_rank[0]);
    out.sequence  = {memory.kv_bytes, memory.kv_backed_bytes, memory.kv_backed_bytes};
    out.workspace = arena(memory.workspace_bytes + memory.experts.bank_bytes +
                          memory.experts.metadata_bytes);
    out.kv_payload_bytes              = memory.kv_bytes;
    out.expert_cache_bytes = memory.experts.cache_bytes;
    out.runtime_reservation_bytes = memory.kv_bytes + memory.workspace_bytes +
        memory.experts.cache_bytes + memory.experts.bank_bytes + memory.experts.metadata_bytes;
    out.available_after_weights_bytes = impl_->instance.free_after_weights;
    out.available_after_startup_bytes = impl_->instance.free_after_startup;
    return out;
}

RuntimeStats FlintCore::runtime_stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    auto stats = impl_->stats;
    const auto experts = impl_->instance.executor->expert_stats();
    stats.experts.routes = experts.pairs;
    stats.experts.hits = experts.pair_hits;
    stats.experts.cpu_routes = experts.cpu_pairs;
    stats.experts.admitted = experts.admissions + experts.lent_admissions;
    const auto memory = impl_->instance.executor->memory().experts;
    stats.experts.slots = memory.cache_slots + memory.lent_slots;
    stats.experts.transferred_bytes = experts.copied_bytes;
    return stats;
}

bool FlintCore::is_available() const {
    std::lock_guard lock(impl_->queue_mutex);
    return !impl_->stopping && !impl_->failed;
}

bool FlintCore::has_failed() const {
    std::lock_guard lock(impl_->queue_mutex);
    return impl_->failed;
}

} // namespace ninfer::runtime
