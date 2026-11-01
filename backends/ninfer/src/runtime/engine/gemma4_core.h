#pragma once
// The Engine core of a Gemma 4 (Gemma4ForCausalLM) artifact. The model family has its own load and
// execution (models/gemma4) and serves prompts through the shared frontend in its Gemma dialect.
//
// Up to max_concurrency requests run at once, each on its own executor sequence, admitted in FIFO
// order. A prompt is prefilled in chunks, one request at a time; between its chunks every decoding
// request advances one token, all of them in one batched pass. With the context cache on, a
// sequence keeps its state when its request ends, and a snapshot of its sliding-window rings where
// the prompt's last user turn closes: a later prompt that starts with what the sequence holds, or
// with the prompt up to that point, resumes there instead of prefilling it again. What a sequence
// holds before another request takes it, and its state at the end of a prompt's shared prefixes
// (the leading system messages and tool declarations, a declared cache boundary), go to images in
// pinned host memory within --host-cache-mib; a prompt on any sequence resumes from the longest
// image it starts with, including one stored while it waited for its first prompt chunk.
//
// With --spec mtp the model's assistant drafts: requests that can run a speculative round draft
// after their next token (the configured drafts for a request alone, one each while two or three
// decode, none beyond) and verify it and the drafts in one pass; the acceptance keeps drafts while
// they equal the request's own keyed draw, so the output is plain decoding's.
#include "core/device.h"
#include "models/gemma4/executor.h"
#include "models/gemma4/model.h"
#include "models/gemma4/vision_cpu.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "ninfer/observability/resources.h"
#include "ninfer/observability/snapshot.h"
#include "ninfer/request.h"
#include "ninfer/types_fwd.h"
#include "runtime/contract/request.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::runtime {

struct Gemma4Instance {
    std::unique_ptr<models::gemma4::Model> model;
    models::qwen3_5::Frontend frontend;
    std::unique_ptr<models::gemma4::Executor> executor;
    // With --vision-residency cpu: the vision tower on CPU threads.
    std::unique_ptr<models::gemma4::CpuVisionTower> cpu_vision;
    // With --vision: the most soft tokens one image or video frame takes.
    std::uint32_t vision_budget = 0;
    std::uint32_t capacity = 0;
    std::size_t free_after_weights = 0;
    std::size_t free_after_startup = 0;
    // --expert-profile-out: the routing profile written at request boundaries.
    std::filesystem::path record_profile;
};

struct ConstructedGemma4 {
    std::unique_ptr<Gemma4Instance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
};

[[nodiscard]] bool is_gemma4_artifact(const std::filesystem::path& path);
[[nodiscard]] ConstructedGemma4 construct_gemma4(const EngineOptions& options,
                                                 DeviceContext& device);

class Gemma4Core {
public:
    using Clock = std::chrono::steady_clock;
    struct Request;

    class Submission {
    public:
        Submission() noexcept = default;
        ~Submission();
        Submission(Submission&& other) noexcept;
        Submission& operator=(Submission&& other) noexcept;
        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation);

        [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
            return effective_thinking_budget_;
        }

    private:
        friend class Gemma4Core;
        Submission(Gemma4Core& owner, std::shared_ptr<Request> request,
                   std::optional<std::uint32_t> budget) noexcept;
        Gemma4Core* owner_ = nullptr;
        std::shared_ptr<Request> request_;
        std::optional<std::uint32_t> effective_thinking_budget_;
    };

    Gemma4Core(Gemma4Instance& instance, DeviceContext& device, const EngineOptions& options);
    ~Gemma4Core();
    Gemma4Core(const Gemma4Core&)            = delete;
    Gemma4Core& operator=(const Gemma4Core&) = delete;

    void stop() noexcept;

    Submission submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {});

    // One request of submit_group: the arguments of submit.
    struct GroupMember {
        models::qwen3_5::PreparedPrompt prompt;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;
        GenerationObservationOptions observation;
        Clock::time_point pending_deadline;
    };

    // Queues the members together, in order, under one queue lock, so the worker admits them at
    // one boundary: the first to reach a prefix they share stores its state there and the others
    // resume from it. Capacity is reserved for the whole group, and an error for one member
    // queues none.
    std::vector<Submission> submit_group(std::vector<GroupMember> group);

    // Natural-log probabilities of tokens [first_target, n) given their prefixes.
    [[nodiscard]] std::vector<float> score(models::qwen3_5::PreparedPrompt prompt,
                                           std::uint32_t first_target);

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] bool is_available() const;
    [[nodiscard]] bool has_failed() const;

    void reset_memory_peaks() noexcept {}

private:
    struct Impl;
    // A member's request, validated and ready to queue, and its effective thinking budget.
    std::shared_ptr<Request> make_request(GroupMember& member,
                                          std::optional<std::uint32_t>& budget);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::runtime
