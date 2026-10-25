#pragma once

// The Engine core of a Qwen3.8-Flash-Next (Qwen4ExpForCausalLM) artifact. The model family has its
// own load and execution (models/qwen4_exp) and shares the Qwen3.5 frontend: prompts are prepared,
// and output is decoded, stopped and split into reasoning and content exactly as for Qwen3.5.
//
// Up to max_concurrency requests run at once, each on its own executor sequence, admitted in FIFO
// order. A prompt is prefilled in chunks, one request at a time; between its chunks every decoding
// request advances one token, all of them in one batched pass whose experts read their weights
// once, and each feeds its sampled token (with any thinking-budget control suffix the frontend asks
// for) back in. With the context cache on, a sequence keeps its state when its request ends, and
// a snapshot of it where the prompt's last user turn closes: a later prompt that starts with what
// the sequence holds, or with the prompt up to that point, resumes there instead of prefilling it
// again.

#include "core/device.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen4_exp/executor.h"
#include "models/qwen4_exp/model.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::runtime {

struct Qwen4ExpInstance {
    std::unique_ptr<models::qwen4_exp::Model> model;
    models::qwen3_5::Frontend frontend;
    std::unique_ptr<models::qwen4_exp::Executor> executor;
    std::uint32_t capacity = 0;
    // Free memory of the primary device once the weights were placed, and after startup.
    std::size_t free_after_weights = 0;
    std::size_t free_after_startup = 0;
};

struct ConstructedQwen4Exp {
    std::unique_ptr<Qwen4ExpInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
};

[[nodiscard]] bool is_qwen4_exp_artifact(const std::filesystem::path& path);
[[nodiscard]] ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options,
                                                      DeviceContext& device);

class Qwen4ExpCore {
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
        friend class Qwen4ExpCore;
        Submission(Qwen4ExpCore& owner, std::shared_ptr<Request> request,
                   std::optional<std::uint32_t> budget) noexcept;
        Qwen4ExpCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;
        std::optional<std::uint32_t> effective_thinking_budget_;
    };

    Qwen4ExpCore(Qwen4ExpInstance& instance, DeviceContext& device, const EngineOptions& options);
    ~Qwen4ExpCore();
    Qwen4ExpCore(const Qwen4ExpCore&)            = delete;
    Qwen4ExpCore& operator=(const Qwen4ExpCore&) = delete;

    void stop() noexcept;

    Submission submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {});

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
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::runtime
