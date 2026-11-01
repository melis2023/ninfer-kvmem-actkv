#pragma once

// Requests and their results: sampling, stops, thinking control, structured output, media,
// tool calls, chat messages, prompts and context-cache hints, request errors, streamed output
// and the generation result.

#include "ninfer/base.h"
#include "ninfer/observability/execution.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;
    // Applies from the token after a thinking request closes its reasoning block, when the
    // request or the server asks for post-thinking sampling.
    SamplingPreset post_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

// What the reasoning-loop guard does when the open thinking keeps repeating itself (every 512
// thinking tokens: the last 2,048 characters repeat a period of at most 64 characters, or the share
// of the last 2,000 words inside a 12-word passage seen three times reaches 25 %).
enum class ReasoningLoopAction : std::uint8_t {
    Off,      // no check
    Stop,     // end the request there (FinishReason::OutputLimit)
    Conclude, // inject the thinking-close control (as a reached thinking budget does) and let the
              // model answer; Stop when early close is unavailable (always on Gemma 4 and
              // Flint) or the remaining output budget cannot hold the control span and one more
              // token
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
    // Effective thinking budget derived from capacity and control requirements.
    // When omitted, defaults to budget.
    std::optional<std::uint32_t> effective_budget;
    // Whether canonical early-close guidance and control tokens can be inserted.
    bool early_close_available = true;
    ReasoningLoopAction loop   = ReasoningLoopAction::Off;
};

enum class StructuredOutputKind : std::uint8_t {
    None,
    JsonObject,
    JsonSchema,
    Regex,
    Choice,
    Grammar
};

struct StructuredOutputOptions {
    StructuredOutputKind kind = StructuredOutputKind::None;
    std::string schema; // JSON Schema for JsonSchema, otherwise empty
    // JsonSchema only. When true, object schemas without additionalProperties (or
    // unevaluatedProperties) admit only their declared properties, following the OpenAI/Anthropic
    // strict structured-output convention. When false, standard JSON Schema defaults apply.
    bool strict = false;
    // Regex: a regular expression the whole final response matches. Grammar: a GBNF grammar
    // (llama.cpp's syntax) whose `root` rule the whole final response matches. Otherwise empty.
    std::string grammar;
    // Choice: the final response is exactly one of these strings (at least one, each non-empty).
    std::vector<std::string> choices;
};

struct ExecutionOptions {
    StructuredOutputOptions structured_output;
    SamplingOverrides sampling;
    // Present on a thinking request: from the token after the model closes its reasoning block,
    // sampling switches to these values, omitted ones taken from the post-thinking preset and an
    // omitted seed from the request. Ignored when thinking is off.
    std::optional<SamplingOverrides> post_thinking_sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    // Opt-in log probabilities: for every generated content token, its log probability and the
    // kMaximumTokenLogprobs most likely tokens under the distribution the sampler draws from
    // (see GenerationResult::content_logprobs). Sampling is unaffected.
    bool logprobs = false;
    // Typed readout, with logprobs: the exact log probability of each of these token ids at the
    // first generated token, under the softmax of the head's logits (the distribution of a greedy
    // request without penalties), whether or not the id is among the top alternatives
    // (TokenLogprob::readout). At most kMaximumReadoutTokens ids, each within the model's public
    // token domain. A core without the readout leaves TokenLogprob::readout empty.
    std::vector<TokenId> readout_tokens;
    // False keeps the Engine's speculative drafter (--spec) off this request: a one-token request
    // drafts nothing, and the drafter's work over the prompt then stops at the deepest cache entry
    // the request publishes (its last explicit cache boundary), so the entries up to there still
    // serve drafting requests and the request leaves no continuation of its own. Qwen3.8-Flash-Next
    // runs no MTP block over such a prompt at all; its entries of it serve only requests that do
    // not draft. Requires requested_output_tokens == 1. A core without this control runs its
    // drafter as configured.
    bool speculative_drafting = true;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

struct NgramSessionHints {
    // Explicit local conversation identity, at most 256 bytes. Never inferred from
    // prompt similarity or the KV session key. Empty means request-local drafting.
    std::string key;
    bool reset = false;
    std::string parent;
    std::uint64_t parent_generation = 0;
    // Retention-enabled requests mix the sampling seed with fresh request entropy:
    // prior generated proposals must not reuse the random draws that produced them.
};

enum class ToolChoiceMode : std::uint8_t { Auto, None, Required };
enum class ToolConstraintMode : std::uint8_t { Automatic, Basic };
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::optional<std::vector<std::string>> allowed_names;
    bool parallel = true;
    ToolConstraintMode constraints = ToolConstraintMode::Automatic;
};

struct RequestOptions {
    ToolChoice tool_choice;
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    NgramSessionHints ngram_session;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

// How finely an image is seen (OpenAI's image detail). Low bounds it at the area of a 512 x 512
// picture, 256 merged Vision tokens (or the server's own bound if that is smaller); Auto and High
// use the server's bound (--vision-max-merged).
enum class ImageDetail : std::uint8_t {
    Auto,
    Low,
    High,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
    ImageDetail image_detail              = ImageDetail::Auto;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Terminal interpretation of model-origin tool-call markup. Parameter schemas guide JSON
// normalization but do not validate the call; only a structure/identity failure can return a
// complete marker region to ordinary content.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen = false;
    // A forced call whose closing tag the model never emitted was closed by the decoder. Only the
    // outer tag is ever supplied, so no argument byte is invented.
    bool forced_call_closed                     = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t empty_arguments_omitted       = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    // Repeated parameters of a call, resolved to their last value.
    std::uint32_t duplicate_parameters_repaired = 0;
    // Why the strict pass rejected the region. With recovered set, the region still produced
    // calls and nothing went out as text.
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;
    bool recovered                              = false;
    // Calls the recovery pass kept although the strict pass would not: an undeclared name, a
    // parameter closed by its function, a call missing only its outer tag.
    std::uint32_t recovered_call_count = 0;
    // A call the recovery pass could not read either was replaced by a call to the reserved error
    // tool, so the client answers with an error and the model gets a turn to retry.
    bool malformed_call_reported = false;
    // Text after the last call; the model is predicting a tool result there, so it is dropped.
    bool trailing_content_dropped = false;
    // The structured calls came from a turn that ended on its stop token inside unclosed thinking.
    bool recovered_from_reasoning = false;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    // OpenAI-shaped function definitions in declaration order. `function.strict` configures the
    // output contract and is not rendered into the prompt.
    std::vector<std::string> tool_jsons;
    // Function the caller selected. Its call opener is appended to the generation prompt, so the
    // answer can only continue inside that call. Requires a new assistant turn with thinking off.
    std::string forced_tool_name;
    // Name of an EngineOptions::grafts entry to place in front of the rendered prompt; empty for
    // none. The rendered prompt should carry no system turn of its own: the graft already holds one.
    std::string graft;
    // The request drafts (ExecutionOptions::speculative_drafting). False prepares no n-gram
    // proposal index or sources: a request that does not draft reads one token.
    bool speculative_drafting = true;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
    EngineObserved     = 1U << 4U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Structural shared candidates the Engine derives from the prompt's own shape: after the
    // leading System/Developer block, after the tool definitions, and at the full prompt frontier.
    // A protocol that carries its own write policy still wants these, because its policy only
    // governs where *its* marker goes; turning them off leaves a request whose only candidate sits
    // at the end of its own prompt, which no differing request can ever match.
    bool allow_engine_automatic_shared_prefixes = true;
    // Propose additional shared candidates on a content-independent token grid so two prompts that
    // merely start alike converge on the same frontier. Grid candidates carry EngineObserved
    // evidence only, so they are never speculatively materialized: a grid frontier is published
    // solely once two distinct reuse domains have both asked for it.
    bool allow_engine_prefix_grid = false;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

// A context-cache store (Device or Host StateImage slots, Host KV extents, Paged KV pages, transfer
// lanes) could not honor a placement the planner had already accounted for. Materialization aborts,
// only the affected request fails (Overloaded), and the Engine keeps serving. Derives from
// std::bad_alloc so existing exhaustion handling keeps working; what() names the exhausted store.
class ContextCacheExhausted final : public std::bad_alloc {
public:
    explicit ContextCacheExhausted(std::string message) : message_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    InvalidMedia,
    InvalidToolConstraint,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

// One generated content token: its id, log probability and decoded bytes (possibly part of a UTF-8
// sequence), and the kMaximumTokenLogprobs most likely tokens at its position in descending order,
// the lower id first on a tie. Slots past the last token the mask admits hold id -1.
struct TokenLogprob {
    TokenId id        = 0;
    float logprob     = 0.0f;
    std::string bytes;
    std::array<TokenId, kMaximumTokenLogprobs> top_ids{};
    std::array<float, kMaximumTokenLogprobs> top_values{};
    // Decoded byte string for each top-k alternative, aligned with top_ids.
    std::array<std::string, kMaximumTokenLogprobs> top_bytes;
    // The log probability of each ExecutionOptions::readout_tokens id, in the request's order,
    // under the softmax of the head's logits at the first generated token; empty on every other
    // token and when the request named no readout tokens or the core has no typed readout.
    std::vector<float> readout;
};

// One prompt of Engine::submit_branches: its length, shared prefix included, and the record of the
// first token the model generates after it. TokenLogprob::readout holds the branch's readout
// tokens' log probabilities and `bytes` the generated token's.
struct BranchResult {
    std::uint32_t prompt_tokens = 0;
    TokenLogprob first;
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
    // The log probability records of the content tokens whose text this delta publishes, in token
    // order. Empty unless the request asked for logprobs.
    std::vector<TokenLogprob> logprobs;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live observations require a Streaming consumer;
// phase timings may also be retained for an Aggregate terminal response.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    // One record per content token, in generation order, when the request enabled logprobs.
    std::vector<TokenLogprob> content_logprobs;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    // Logical uncached prompt tokens accepted at completed prefill boundaries.
    // Excludes reused prefixes and internal replay; may be partial or zero on cancellation.
    std::uint32_t computed_prompt_tokens = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    // The private catalog cell the finished session was retained in and its session digest, or
    // -1 and empty when the session was not retained.
    std::int32_t slot = -1;
    std::string session_digest;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    NgramArchiveStats ngram_archive;
    ThinkingBudgetStats thinking;
    // Engine::submit_branches: one record per branch, in order; empty for every other request.
    std::vector<BranchResult> branches;
};

} // namespace ninfer
