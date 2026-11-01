#pragma once

#include "ninfer/base.h"
#include "ninfer/observability/execution.h"
#include "ninfer/request.h"
#include <atomic>
#include <cstdint>
#include <optional>

#include <memory>
#include <vector>

namespace ninfer::text {
class GrammarState;
}

namespace ninfer::runtime {

// One branch of Engine::submit_branches: the tokens that continue the request's prompt, and the
// token ids whose exact log probabilities the record of its first generated token carries.
struct ReadBranch {
    std::vector<TokenId> tokens;
    std::vector<TokenId> readout_tokens;
};

// Engine has already selected the model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    StructuredOutputOptions structured_output;
    std::shared_ptr<text::GrammarState> grammar;
    // Default tool_choice none may name the one vocabulary id the sampler must not emit.
    std::optional<TokenId> suppressed_tool_open;
    ResolvedSamplingParameters sampling;
    // Applied once the model closes its reasoning block.
    std::optional<ResolvedSamplingParameters> post_thinking_sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    // Gather each generated token's log probability and its likely alternatives.
    bool logprobs = false;
    // With logprobs: the token ids whose exact log probabilities the first generated token's
    // record carries (ExecutionOptions::readout_tokens).
    std::vector<TokenId> readout_tokens;
    // ExecutionOptions::speculative_drafting.
    bool speculative_drafting = true;
    // Engine::submit_branches: after the prompt, each branch runs as a block over it and its first
    // generated token is read; null for an ordinary request. Such a request is greedy, reads one
    // token and drafts nothing; its own readout_tokens are empty.
    std::shared_ptr<const std::vector<ReadBranch>> branches;
    // Engine-assigned publication order; never supplied by an external client.
    std::uint64_t publication_order = 0;
};

struct ResolvedRequestOptions {
    ToolChoice tool_choice;
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    NgramSessionHints ngram_session;
};

enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

struct OutputDecision {
    std::uint32_t accepted_tokens   = 0;
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;
    // Empty or one token-aligned frontier within the accepted span where model-history
    // reconstruction gains an execution split. Frontend owns detection; Engine only transports
    // this relative position.
    std::optional<std::uint32_t> prefix_execution_split_after;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Non-owning cancellation observation used while the worker advances a context transaction. The
// request record owns the flag for longer than Program can retain this view.
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    std::uint64_t service_work_quanta     = 0;
    bool publish_continuation             = true;
};

} // namespace ninfer::runtime
