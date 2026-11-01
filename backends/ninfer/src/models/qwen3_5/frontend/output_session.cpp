#include "models/qwen3_5/frontend/output_session.h"
#include "text/structured_output.h"
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/flint_output.h"
#include "models/qwen3_5/frontend/gemma_output.h"
#include "models/qwen3_5/frontend/reasoning_loop.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/unicode.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {
namespace fi                                = frontend;
constexpr std::string_view kThinkClose      = "</think>";
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // Replace one maximal subpart. The first byte that cannot continue this sequence
                // is deliberately left for the next iteration, so valid following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

// The model serializes its reasoning close as "\n</think>\n\n", so only a line break confirms a
// marker. Prose that quotes the marker ("the </think> tag") goes on along the same line.
constexpr bool is_close_separator(char byte) noexcept { return byte == '\r' || byte == '\n'; }

struct ReasoningCloseScan {
    std::size_t close = std::string::npos;
    std::size_t hold  = 0;
};

// Generated prose can quote the close marker while discussing the protocol, so a marker counts as
// the reasoning close only when it is followed by a line break, or by the implicit end of the
// turn. A marker inside quoted text is followed by a space, punctuation or an escaped newline and
// stays in the reasoning channel. Markers at the end of the available bytes stay pending until
// their following byte arrives, which keeps a quoted marker from closing the channel across a
// token round. Under a grammar (`exact`) the reasoning text excludes the marker, so its first
// occurrence is the close whatever follows it.
ReasoningCloseScan scan_reasoning_close(std::string_view text, bool exact, bool implicit_end) {
    if (exact) {
        const std::size_t marker = text.find(kThinkClose);
        if (marker != std::string_view::npos) { return ReasoningCloseScan{.close = marker}; }
        return ReasoningCloseScan{.hold = longest_suffix_prefix(text, kThinkClose)};
    }
    std::size_t search = 0;
    for (;;) {
        const std::size_t marker = text.find(kThinkClose, search);
        if (marker == std::string_view::npos) { break; }
        const std::size_t after = marker + kThinkClose.size();
        if (after == text.size()) {
            if (implicit_end) { return ReasoningCloseScan{.close = marker}; }
            return ReasoningCloseScan{.hold = text.size() - marker};
        }
        if (is_close_separator(text[after])) { return ReasoningCloseScan{.close = marker}; }
        search = after;
    }
    return ReasoningCloseScan{.hold = longest_suffix_prefix(text, kThinkClose, true)};
}

template <std::size_t Size>
consteval std::array<std::size_t, Size> make_prefix_failure_table(std::string_view pattern) {
    std::array<std::size_t, Size> failure{};
    for (std::size_t index = 1; index < Size; ++index) {
        std::size_t matched = failure[index - 1U];
        while (matched != 0 && pattern[index] != pattern[matched]) {
            matched = failure[matched - 1U];
        }
        if (pattern[index] == pattern[matched]) { ++matched; }
        failure[index] = matched;
    }
    return failure;
}

struct PrefixExecutionTracker {
    static constexpr std::string_view kBoundary = fi::kCanonicalReasoningCloseSerialization;
    static constexpr auto kFailure = make_prefix_failure_table<kBoundary.size()>(kBoundary);

    // Returns the byte offset immediately after the first completed boundary in this token.
    [[nodiscard]] std::optional<std::size_t> feed(std::string_view bytes) noexcept {
        if (!tracking) { return std::nullopt; }
        for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
            const char byte = bytes[offset];
            while (matched != 0 && byte != kBoundary[matched]) { matched = kFailure[matched - 1U]; }
            if (byte == kBoundary[matched]) { ++matched; }
            if (matched != kBoundary.size()) { continue; }
            tracking = false;
            matched  = 0;
            return offset + 1U;
        }
        return std::nullopt;
    }

    std::size_t matched = 0;
    bool tracking       = false;
};

// A content token's logprob record, held until the content channel publishes the token's first
// byte, so that a record never precedes its text and a stop string's cut drops the records of the
// bytes it removes.
struct HeldLogprob {
    TokenLogprob record;
    std::uint64_t start = 0; // the token's first byte in content-channel bytes
};

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    // Gemma: inside a channel's header (`thought\n` after <|channel>), and its bytes so far.
    bool gemma_header = false;
    std::string gemma_header_text;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning              = false;
    // A grammar constrains the output: the first close marker ends the reasoning.
    bool exact_close               = false;
    bool strip_content_leading     = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
    // Bytes fed to the content channel after the leading-whitespace strip, and the part of them
    // published; the difference is held for a possible stop string.
    std::uint64_t content_fed       = 0;
    std::uint64_t content_published = 0;
    std::vector<HeldLogprob> held_logprobs;
    // Records of content tokens whose bytes still wait in utf8_pending for the rest of a code point.
    std::vector<TokenLogprob> utf8_logprobs;
};

struct SemanticThinkingState {
    std::optional<std::uint32_t> budget;
    std::string close_pending;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool in_reasoning                   = false;
    bool exact_close                    = false;
    bool control_pending                = false;
    bool applied                        = false;
    // Reasoning-loop guard: the words of the round being previewed (the committed history lives in
    // OutputSession::Impl, outside the per-round copy), a word cut by the last token, the recent
    // reasoning the periodic-tail check reads, the next check and what fired.
    ReasoningLoopAction loop = ReasoningLoopAction::Off;
    std::string loop_partial;
    std::string loop_tail;
    std::vector<std::uint64_t> loop_round;
    std::uint32_t loop_next_check      = kLoopCheckTokens;
    bool loop_detected                 = false;
    std::uint32_t loop_thinking_tokens = 0;
    float loop_coverage                = 0.0F;
};

void feed_semantic_thinking(SemanticThinkingState& state, std::string_view bytes) {
    if (!state.in_reasoning || bytes.empty()) { return; }
    state.close_pending.append(bytes);
    const ReasoningCloseScan scan =
        scan_reasoning_close(state.close_pending, state.exact_close, false);
    if (scan.close != std::string::npos) {
        state.close_pending.clear();
        state.in_reasoning    = false;
        state.control_pending = false;
        return;
    }
    state.close_pending.erase(0, state.close_pending.size() - scan.hold);
}

// A complete free-output `</think>` waiting for the byte after it. The token that confirms it is
// not thinking, and a held marker is the model's own close at a budget boundary.
bool holds_semantic_close(const SemanticThinkingState& state) noexcept {
    return state.in_reasoning && !state.exact_close &&
           std::string_view(state.close_pending).ends_with(kThinkClose);
}

bool holds_presentation_close(const DecoderState& state) noexcept {
    return state.in_reasoning && !state.exact_close && state.utf8_pending.empty() &&
           std::string_view(state.think_marker_pending).ends_with(kThinkClose);
}

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    // The content-channel bytes that remain published after the cut.
    std::uint64_t content_cut = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            // The combined text starts at the content channel's first unpublished byte.
            const std::uint64_t content_cut =
                state.content_published +
                (channel == OutputChannel::Content
                     ? found + (stop.include_in_output ? stop.text.size() : 0U)
                     : 0U);
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .content_cut       = content_cut,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
    if (channel == OutputChannel::Content) {
        state.content_fed += text.size();
        state.content_published = state.content_fed - hold;
    }
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
    if (channel == OutputChannel::Content) { state.content_published = state.content_fed; }
}

// Moves the held records whose first byte the content channel has published to `released`, in
// generation order; a terminal state has published everything it will.
void release_logprobs(DecoderState& state, std::vector<TokenLogprob>& released) {
    std::size_t count = 0;
    while (count < state.held_logprobs.size() &&
           (state.terminal || state.held_logprobs[count].start < state.content_published)) {
        released.push_back(std::move(state.held_logprobs[count].record));
        ++count;
    }
    state.held_logprobs.erase(state.held_logprobs.begin(),
                              state.held_logprobs.begin() + static_cast<std::ptrdiff_t>(count));
    if (state.terminal) {
        for (TokenLogprob& record : state.utf8_logprobs) { released.push_back(std::move(record)); }
        state.utf8_logprobs.clear();
    }
}

// Ends the records at a stop string's cut: the records of tokens with a byte before the cut are
// released, the others belong to the cut bytes and are dropped.
void cut_logprobs(DecoderState& state, std::uint64_t content_cut,
                  std::vector<TokenLogprob>& released) {
    for (HeldLogprob& held : state.held_logprobs) {
        if (held.start < content_cut) { released.push_back(std::move(held.record)); }
    }
    state.held_logprobs.clear();
    state.utf8_logprobs.clear();
}

void feed_content(DecoderState& state, std::string text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best_match) {
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens,
                 best_match);
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       PublishedOutput& emitted, std::uint32_t committed_tokens,
                       StopMatch* best_match) {
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, best_match);
        return;
    }

    state.think_marker_pending.append(text);
    const ReasoningCloseScan scan =
        scan_reasoning_close(state.think_marker_pending, state.exact_close, false);
    if (scan.close != std::string::npos) {
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, scan.close), policy,
                     emitted, committed_tokens, best_match);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(scan.close + kThinkClose.size());
        state.think_marker_pending.clear();
        state.in_reasoning          = false;
        state.strip_content_leading = true;
        feed_content(state, std::move(content), policy, emitted, committed_tokens, best_match);
        return;
    }

    const std::size_t safe = state.think_marker_pending.size() - scan.hold;
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, best_match);
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string_view bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match) {
    state.utf8_pending.append(bytes);
    const std::string text = consume_generated_utf8(state.utf8_pending);
    feed_decoded_text(state, text, policy, emitted, committed_tokens, best_match);
}

// The model's stop token after a held `</think>` ends the turn right after it, so the held marker
// is the close; it is resolved before the stop token's own bytes (published when the caller asks
// for them) reach the decoder.
void close_held_reasoning(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                          std::uint32_t committed_tokens) {
    if (!holds_presentation_close(state)) { return; }
    const std::size_t split = state.think_marker_pending.size() - kThinkClose.size();
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, split), policy, emitted,
                 committed_tokens, nullptr);
    close_channel(state, OutputChannel::Reasoning, emitted);
    state.think_marker_pending.clear();
    state.in_reasoning          = false;
    state.strip_content_leading = true;
}

void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens) {
    if (!state.utf8_pending.empty()) {
        // A token budget can end between byte-level tokens of one code point.
        // Publish the standard replacement character rather than an invalid
        // UTF-8 suffix; the logical token prefix remains exact.
        state.utf8_pending.clear();
        feed_decoded_text(state, kUtf8Replacement, policy, emitted, committed_tokens, nullptr);
    }
    if (state.in_reasoning) {
        // A close marker still pending at the end of the turn is the model's implicit close.
        const ReasoningCloseScan scan =
            scan_reasoning_close(state.think_marker_pending, state.exact_close, true);
        const std::size_t split =
            scan.close != std::string::npos ? scan.close : state.think_marker_pending.size();
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, split), policy, emitted,
                     committed_tokens, nullptr);
        close_channel(state, OutputChannel::Reasoning, emitted);
        if (scan.close != std::string::npos) {
            std::string content =
                state.think_marker_pending.substr(scan.close + kThinkClose.size());
            state.think_marker_pending.clear();
            state.in_reasoning          = false;
            state.strip_content_leading = true;
            feed_content(state, std::move(content), policy, emitted, committed_tokens, nullptr);
        } else {
            state.think_marker_pending.clear();
        }
    }
    close_channel(state, OutputChannel::Content, emitted);
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.think_marker_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    state.held_logprobs.clear();
    state.utf8_logprobs.clear();
    return state;
}

} // namespace

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning, ThinkingControlOptions thinking,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::shared_ptr<const fi::ToolCallOutputContract> tool_call_output_,
         std::shared_ptr<const fi::GemmaMarkers> gemma_,
         std::shared_ptr<const fi::FlintMarkers> flint_)
        : gemma(output.raw ? nullptr : std::move(gemma_)),
          flint(output.raw ? nullptr : std::move(flint_)), tokenizer(std::move(tokenizer_)),
          policy(std::move(policy_)),
          thinking_control_tokens(std::move(thinking_control_tokens_)),
          preserve_special(output.raw || output.preserve_special_tokens ||
                           (tool_call_output_ && tool_call_output_->constrained)),
          split_reasoning(starts_in_reasoning && !output.raw),
          // Copied, not moved: the body still asks whether the request declared tools. Gemma's
          // and Flint's calls go through their own decoders; the Qwen decoder would seed its own
          // opener for a forced call.
          tool_call_output(output.raw || gemma || flint ? nullptr : tool_call_output_,
                           output.tool_name_max_length) {
        if (thinking.budget && *thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        requested_budget          = thinking.budget;
        early_close_available     = thinking.early_close_available;
        state.in_reasoning        = split_reasoning;
        prefix_execution.tracking = starts_in_reasoning;
        started_in_reasoning      = starts_in_reasoning;
        semantic.budget           = thinking.effective_budget.has_value()
                                        ? thinking.effective_budget
                                        : thinking.budget;
        semantic.loop             = thinking.loop;
        // The semantic tracker counts the model's own thinking tokens for every response that
        // starts in thinking: the request log reports that count (model_thinking_tokens), so it
        // must run without a budget too (upstream Neroued/ninfer#373). The budget caps it and the
        // loop guard reads it. Cost: one close-marker scan per generated thinking token.
        semantic.in_reasoning = starts_in_reasoning;
        collect_open_reasoning = split_reasoning && gemma == nullptr && flint == nullptr &&
                                 tool_call_output_ != nullptr;
        if (gemma && tool_call_output_ != nullptr) {
            // A forced call's opener, `<|tool_call>call:NAME{`, is already in the prompt.
            gemma_tools.emplace(tool_call_output_->forced_tool_name);
        }
        if (flint && tool_call_output_ != nullptr) { flint_tools.emplace(tool_call_output_); }
    }

    // Gemma's dialect: channel markers open and close reasoning, tool-call markup stays in the
    // content for gemma_tools (present when the request declared tools).
    std::shared_ptr<const fi::GemmaMarkers> gemma;
    std::optional<fi::GemmaToolCallDecoder> gemma_tools;
    // Flint's dialect: the model's own <think> and </think> tokens open and close reasoning,
    // tool-call markup stays in the content for flint_tools (present when the request declared
    // tools).
    std::shared_ptr<const fi::FlintMarkers> flint;
    std::optional<fi::FlintToolCallDecoder> flint_tools;

    // Whether the model opens its reasoning itself (Gemma's thought channel, Flint's <think>), so
    // the decoder's channel, not the prompt-opened semantic state, says when it reasons.
    [[nodiscard]] bool model_opens_reasoning() const noexcept { return gemma || flint; }

    // The repeated-passage coverage of the committed thinking plus the previewed round's words.
    [[nodiscard]] double loop_coverage(const std::vector<std::uint64_t>& round) const {
        const std::size_t keep =
            kLoopHistoryWords > round.size() ? kLoopHistoryWords - round.size() : 0;
        std::vector<std::uint64_t> words(
            loop_words.end() - static_cast<std::ptrdiff_t>(std::min(keep, loop_words.size())),
            loop_words.end());
        words.insert(words.end(), round.begin(), round.end());
        return repeated_passage_coverage(words);
    }

    // Whether the loop guard reads this point of the output: Qwen's thinking, opened by the prompt
    // and tracked by the semantic state, or the reasoning a Gemma or Flint model opened itself.
    [[nodiscard]] bool loop_reasoning(const DecoderState& decoder,
                                      const SemanticThinkingState& thinking) const noexcept {
        return model_opens_reasoning() ? decoder.in_reasoning : thinking.in_reasoning;
    }

    // Moves the committed round's words into the history (bounded); drops it once thinking ends.
    void commit_loop_words() {
        if (!loop_reasoning(state, semantic) || semantic.loop == ReasoningLoopAction::Off ||
            semantic.loop_detected) {
            loop_words.clear();
            loop_words.shrink_to_fit();
            semantic.loop_round.clear();
            semantic.loop_tail.clear();
            return;
        }
        loop_words.insert(loop_words.end(), semantic.loop_round.begin(), semantic.loop_round.end());
        semantic.loop_round.clear();
        if (loop_words.size() > 2 * kLoopHistoryWords) {
            loop_words.erase(loop_words.begin(),
                             loop_words.end() - static_cast<std::ptrdiff_t>(kLoopHistoryWords));
        }
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    bool preserve_special      = false;
    bool split_reasoning       = false;
    bool started_in_reasoning  = false;
    bool early_close_available = true;
    std::optional<std::uint32_t> requested_budget;
    DecoderState state;
    DecoderState preview_state;
    SemanticThinkingState semantic;
    SemanticThinkingState preview_semantic;
    std::vector<std::uint64_t> loop_words; // committed thinking words (reasoning-loop guard)
    PrefixExecutionTracker prefix_execution;
    PrefixExecutionTracker preview_prefix_execution;
    std::optional<std::uint32_t> preview_execution_split_after;
    PublishedOutput preview_output;
    // Records the preview released with their bytes, and those the last commit published.
    std::vector<TokenLogprob> preview_logprobs;
    std::vector<TokenLogprob> committed_logprobs;
    fi::ToolCallOutputDecoder tool_call_output;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    // Free tool output keeps the open thinking, so a turn that ends inside it can still return the
    // calls stranded there; the text is dropped once the thinking closes.
    bool collect_open_reasoning = false;
    std::string open_reasoning;
    FinishReason preview_finish_reason = FinishReason::None;

    // Holds the record of a token the decoder just fed when its bytes went to the content channel:
    // they raised content_fed, or they wait outside the reasoning block for the rest of a code
    // point. `fed_before` is content_fed before the token.
    void hold_logprob(const runtime::RawTokenLogprob& raw, std::string_view bytes,
                      std::uint64_t fed_before) {
        DecoderState& decoder = preview_state;
        const bool fed        = decoder.content_fed > fed_before;
        if (!fed && (decoder.in_reasoning || decoder.utf8_pending.empty())) { return; }
        TokenLogprob record;
        record.id         = raw.id;
        record.logprob    = raw.logprob;
        record.bytes      = std::string(bytes);
        record.top_ids    = raw.top_ids;
        record.top_values = raw.top_values;
        for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
            if (raw.top_ids[k] >= 0) {
                record.top_bytes[k] = std::string(tokenizer->decoded_token(raw.top_ids[k]).bytes);
            }
        }
        if (raw.readout_count > 0) {
            const std::size_t count = std::min<std::size_t>(raw.readout_count, kMaximumReadoutTokens);
            record.readout.assign(raw.readout_values.begin(), raw.readout_values.begin() + count);
        }
        if (!fed) {
            decoder.utf8_logprobs.push_back(std::move(record));
            return;
        }
        // Tokens that began this token's code point come first.
        for (TokenLogprob& earlier : decoder.utf8_logprobs) {
            decoder.held_logprobs.push_back(
                HeldLogprob{.record = std::move(earlier), .start = fed_before});
        }
        decoder.utf8_logprobs.clear();
        decoder.held_logprobs.push_back(
            HeldLogprob{.record = std::move(record), .start = fed_before});
    }
    bool preview_ready = false;
    std::shared_ptr<text::GrammarState> grammar;
    std::optional<TokenId> suppressed_tool_open;
    std::unique_ptr<text::GrammarState> preview_grammar;
};

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(
    std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy, OutputOptions output,
    bool starts_in_reasoning, ThinkingControlOptions thinking,
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
    std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output,
    std::shared_ptr<text::GrammarState> grammar,
    std::shared_ptr<const frontend::GemmaMarkers> gemma,
    std::shared_ptr<const frontend::FlintMarkers> flint,
    std::optional<TokenId> suppressed_tool_open)
    : impl_(std::make_unique<Impl>(std::move(tokenizer), std::move(policy), output,
                                   starts_in_reasoning, thinking,
                                   std::move(thinking_control_tokens), std::move(tool_call_output),
                                   std::move(gemma), std::move(flint))) {
    impl_->grammar              = std::move(grammar);
    impl_->state.exact_close    = impl_->grammar != nullptr;
    impl_->semantic.exact_close = impl_->grammar != nullptr;
    // A grammar admits the stop token only after the thinking closes, so a constrained turn
    // cannot end inside it.
    if (impl_->grammar) { impl_->collect_open_reasoning = false; }
    impl_->suppressed_tool_open = suppressed_tool_open;
}

std::shared_ptr<text::GrammarState> OutputSession::grammar_state() const {
    return impl_ ? impl_->grammar : nullptr;
}

std::optional<TokenId> OutputSession::suppressed_tool_open() const {
    return impl_ ? impl_->suppressed_tool_open : std::nullopt;
}

runtime::OutputDecision
OutputSession::preview_model(std::span<const TokenId> tokens, std::uint32_t total_budget_remaining,
                             FinishReason limit_reason,
                             std::span<const runtime::RawTokenLogprob> logprobs) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!logprobs.empty() && logprobs.size() != tokens.size()) {
        throw std::invalid_argument("logprob records do not align with the generated-token round");
    }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (impl_->semantic.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    impl_->preview_logprobs.clear();

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        if (reason != FinishReason::None) { impl_->preview_semantic.control_pending = false; }
        impl_->preview_finish_reason = reason;
        if (impl_->preview_execution_split_after && *impl_->preview_execution_split_after > count) {
            throw std::logic_error("prefix execution split exceeds the accepted token prefix");
        }
        if (impl_->grammar) {
            // Installed only once it admits the round: a rejected round leaves no preview grammar
            // for a later terminal preview to commit.
            auto grammar = impl_->grammar->fork();
            grammar->accept(tokens.first(count));
            impl_->preview_grammar = std::move(grammar);
        }
        impl_->preview_ready = true;
        return runtime::OutputDecision{
            .accepted_tokens              = count,
            .finish_reason                = reason,
            .continuation                 = continuation,
            .prefix_execution_split_after = impl_->preview_execution_split_after,
        };
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count          = static_cast<std::uint32_t>(index + 1);
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);

        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = count;
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();

        // A turn-level stop token emitted before the reasoning block closes is the model's
        // implicit reasoning boundary, not the end of the turn. Stop strings already have an
        // output-channel guard; keep the token path consistent and continue decoding content.
        // Preserve the upstream terminal recovery when complete, declared tool calls are
        // stranded in open reasoning. Otherwise retain KVMem's implicit reasoning boundary.
        bool recover_open_calls = false;
        if (stop_token && impl_->preview_state.in_reasoning && impl_->collect_open_reasoning &&
            !holds_presentation_close(impl_->preview_state)) {
            std::string reasoning = impl_->open_reasoning;
            for (const auto& delta : impl_->preview_output)
                if (delta.channel == OutputChannel::Reasoning) reasoning += delta.text;
            reasoning += impl_->preview_state.think_marker_pending;
            auto probe = impl_->tool_call_output;
            recover_open_calls = probe.finish(FinishReason::StopToken, reasoning)
                                     .diagnostics.recovered_from_reasoning;
        }
        if (!impl_->gemma && stop_token && impl_->preview_state.in_reasoning &&
            !holds_presentation_close(impl_->preview_state) && !recover_open_calls) {
            DecoderState& state = impl_->preview_state;
            // This token itself closes thinking; unlike a token confirming an already held
            // close marker, it still belongs to the model's thinking-token budget.
            ++state.reasoning_tokens;
            SemanticThinkingState& thinking = impl_->preview_semantic;
            if (thinking.in_reasoning) {
                ++thinking.model_thinking_tokens;
                if (impl_->early_close_available && thinking.budget &&
                    thinking.model_thinking_tokens > *thinking.budget) {
                    throw std::logic_error("model output exceeded the licensed thinking budget");
                }
            }
            if (!state.utf8_pending.empty()) {
                state.utf8_pending.clear();
                feed_decoded_text(state, kUtf8Replacement, impl_->policy, impl_->preview_output,
                                  count, nullptr);
            }
            if (state.in_reasoning) {
                const ReasoningCloseScan scan =
                    scan_reasoning_close(state.think_marker_pending, state.exact_close, true);
                const std::size_t split = scan.close != std::string::npos
                                              ? scan.close
                                              : state.think_marker_pending.size();
                feed_channel(state, OutputChannel::Reasoning,
                             std::string_view(state.think_marker_pending).substr(0, split),
                             impl_->policy, impl_->preview_output, count, nullptr);
                close_channel(state, OutputChannel::Reasoning, impl_->preview_output);
                state.think_marker_pending.clear();
                state.in_reasoning          = false;
                state.strip_content_leading = true;
            }

            // The presentation decoder, thinking-budget controller and prefix executor are
            // independent state machines. Commit the implicit boundary in all three so a later
            // content token cannot re-arm control or leave prefix execution open.
            impl_->preview_semantic.close_pending.clear();
            impl_->preview_semantic.in_reasoning    = false;
            impl_->preview_semantic.control_pending = false;
            if (impl_->preview_prefix_execution.tracking) {
                impl_->preview_prefix_execution.tracking = false;
                impl_->preview_prefix_execution.matched  = 0;
                impl_->preview_execution_split_after     = count;
            }
            release_logprobs(state, impl_->preview_logprobs);
            continue;
        }

        // A token that confirms a held close (its first byte is a line break, or it ends the
        // turn) is the first one after the thinking, so neither count includes it.
        const bool presentation_held = holds_presentation_close(impl_->preview_state);
        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        SemanticThinkingState& thinking = impl_->preview_semantic;
        if (thinking.in_reasoning) {
            const bool held = holds_semantic_close(thinking);
            if (held) {
                if (stop_token) {
                    thinking.close_pending.clear();
                    thinking.in_reasoning = false;
                } else {
                    feed_semantic_thinking(thinking, decoded.bytes);
                }
            }
            if (thinking.in_reasoning) {
                ++thinking.model_thinking_tokens;
                if (impl_->early_close_available && thinking.budget &&
                    thinking.model_thinking_tokens > *thinking.budget) {
                    throw std::logic_error("model output exceeded the licensed thinking budget");
                }
                if (thinking.loop != ReasoningLoopAction::Off && !thinking.loop_detected) {
                    append_words(decoded.bytes, thinking.loop_partial, thinking.loop_round);
                    append_tail(decoded.bytes, thinking.loop_tail);
                }
                if (!held) { feed_semantic_thinking(thinking, decoded.bytes); }
            }
        }
        const auto uncount_confirming_token = [&] {
            if (presentation_held && !impl_->preview_state.in_reasoning) {
                --impl_->preview_state.reasoning_tokens;
            }
        };
        DecoderState before_state;
        PublishedOutput before_output;
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        if (stop_token && presentation_held) {
            close_held_reasoning(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        }
        StopMatch match;
        std::string_view bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        if (impl_->gemma) {
            const fi::GemmaMarkers& markers = *impl_->gemma;
            DecoderState& decoder           = impl_->preview_state;
            if (token == markers.channel_open) {
                decoder.gemma_header = true;
                decoder.gemma_header_text.clear();
                bytes = {};
            } else if (decoder.gemma_header) {
                // The channel's name and its newline; whatever follows in the token is reasoning.
                decoder.gemma_header_text.append(decoded.bytes);
                const std::size_t newline = decoder.gemma_header_text.find('\n');
                bytes                     = {};
                if (newline != std::string::npos) {
                    decoder.gemma_header = false;
                    decoder.in_reasoning = true;
                    const std::size_t rest =
                        decoder.gemma_header_text.size() - newline - 1;
                    bytes = decoded.bytes.substr(decoded.bytes.size() - rest);
                }
            } else if (token == markers.channel_close) {
                bytes = decoder.in_reasoning ? std::string_view("</think>\n") : std::string_view{};
            } else if (markers.markup(token)) {
                bytes = decoded.bytes;
            }
            if (thinking.loop != ReasoningLoopAction::Off && !thinking.loop_detected &&
                decoder.in_reasoning && token != markers.channel_close) {
                append_words(bytes, thinking.loop_partial, thinking.loop_round);
                append_tail(bytes, thinking.loop_tail);
            }
        } else if (impl_->flint) {
            const fi::FlintMarkers& markers = *impl_->flint;
            DecoderState& decoder           = impl_->preview_state;
            if (token == markers.think_open && !decoder.in_reasoning) {
                // The model decided to reason: what follows is reasoning, the marker is neither.
                decoder.in_reasoning = true;
                bytes                = {};
            } else if (token == markers.think_close && decoder.in_reasoning) {
                // The close marker followed by a line break closes the channel at once.
                bytes = std::string_view("</think>\n");
            }
            if (thinking.loop != ReasoningLoopAction::Off && !thinking.loop_detected &&
                decoder.in_reasoning && token != markers.think_close) {
                append_words(bytes, thinking.loop_partial, thinking.loop_round);
                append_tail(bytes, thinking.loop_tail);
            }
        }
        const std::uint64_t fed_before = impl_->preview_state.content_fed;
        feed_token_bytes(impl_->preview_state, bytes, impl_->policy, impl_->preview_output, count,
                         &match);
        if (!stop_token) { uncount_confirming_token(); }
        if (!logprobs.empty() && !bytes.empty()) {
            impl_->hold_logprob(logprobs[index], bytes, fed_before);
        }

        if (match.found) {
            cut_logprobs(impl_->preview_state, match.content_cut, impl_->preview_logprobs);
            impl_->preview_state = terminal_state(std::move(impl_->preview_state));
            impl_->preview_state.matched_stop_order = match.declaration_order;
            impl_->preview_output                   = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
            }
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            uncount_confirming_token();
            release_logprobs(impl_->preview_state, impl_->preview_logprobs);
            return complete(count, FinishReason::StopToken);
        }
        release_logprobs(impl_->preview_state, impl_->preview_logprobs);
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        release_logprobs(impl_->preview_state, impl_->preview_logprobs);
        return complete(count, limit_reason);
    }
    // A close marker that is the last licensed thinking token is the model's own close at the
    // boundary, not a quote: the budget is spent either way, so no control follows it.
    if (impl_->preview_semantic.budget &&
        impl_->preview_semantic.model_thinking_tokens == *impl_->preview_semantic.budget &&
        holds_semantic_close(impl_->preview_semantic)) {
        impl_->preview_semantic.close_pending.clear();
        impl_->preview_semantic.in_reasoning = false;
    }
    if (impl_->early_close_available && impl_->preview_semantic.in_reasoning &&
        impl_->preview_semantic.budget &&
        impl_->preview_semantic.model_thinking_tokens == *impl_->preview_semantic.budget) {
        impl_->preview_semantic.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    // Reasoning-loop guard: once per kLoopCheckTokens thinking tokens, at the end of a round. Gemma
    // and Flint count the tokens of the reasoning they opened; they have no early-close control,
    // so they stop. A held close marker defers the check to the next round, which resolves it.
    SemanticThinkingState& semantic     = impl_->preview_semantic;
    const std::uint32_t thinking_tokens = impl_->model_opens_reasoning()
                                              ? impl_->preview_state.reasoning_tokens
                                              : semantic.model_thinking_tokens;
    if (semantic.loop != ReasoningLoopAction::Off &&
        impl_->loop_reasoning(impl_->preview_state, semantic) && !semantic.applied &&
        !semantic.loop_detected && !holds_semantic_close(semantic) &&
        thinking_tokens >= semantic.loop_next_check) {
        semantic.loop_next_check = (thinking_tokens / kLoopCheckTokens + 1U) * kLoopCheckTokens;
        // Strata reads a periodic tail as full coverage before its word measure.
        const double coverage =
            periodic_tail(semantic.loop_tail) ? 1.0 : impl_->loop_coverage(semantic.loop_round);
        if (coverage >= kLoopCoverage) {
            semantic.loop_detected        = true;
            semantic.loop_thinking_tokens = thinking_tokens;
            semantic.loop_coverage        = static_cast<float>(coverage);
            const std::uint64_t control =
                impl_->thinking_control_tokens ? impl_->thinking_control_tokens->size() : 0U;
            const std::uint64_t remaining = total_budget_remaining - count;
            // Conclude needs the control span and one post-close model token; a request whose
            // capacity left no room for early close (early_close_available) stops instead.
            if (semantic.loop == ReasoningLoopAction::Conclude && impl_->early_close_available &&
                control > 0 && remaining >= control + 1U) {
                semantic.control_pending = true;
                return complete(count, FinishReason::None,
                                runtime::ContinuationAction::ApplyTargetControl);
            }
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            release_logprobs(impl_->preview_state, impl_->preview_logprobs);
            return complete(count, FinishReason::OutputLimit);
        }
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    // A pending control span (a reached budget's or the loop guard's) licenses no model token.
    if (impl_ != nullptr && impl_->semantic.control_pending) { return 0; }
    if (impl_ == nullptr || !impl_->semantic.budget || !impl_->semantic.in_reasoning ||
        impl_->semantic.applied || !impl_->early_close_available) {
        return total_budget_remaining;
    }
    if (impl_->semantic.control_pending ||
        impl_->semantic.model_thinking_tokens >= *impl_->semantic.budget) {
        return 0;
    }
    return std::min(total_budget_remaining,
                    *impl_->semantic.budget - impl_->semantic.model_thinking_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->semantic.control_pending || !impl_->thinking_control_tokens) {
        return {};
    }
    return std::span<const TokenId>(*impl_->thinking_control_tokens)
        .subspan(impl_->semantic.injected_tokens);
}

std::uint32_t OutputSession::control_suffix_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->thinking_control_tokens || !impl_->semantic.budget ||
        impl_->semantic.applied) {
        return 0;
    }
    return impl_->semantic.control_pending || impl_->semantic.in_reasoning
               ? static_cast<std::uint32_t>(impl_->thinking_control_tokens->size()) - impl_->semantic.injected_tokens
               : 0U;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (tokens.empty() || tokens.size() > expected.size() ||
        !std::equal(tokens.begin(), tokens.end(), expected.begin())) {
        throw std::invalid_argument("thinking control preview requires a prefix of the pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    impl_->preview_logprobs.clear();
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = static_cast<std::uint32_t>(index + 1U);
        }
        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        feed_semantic_thinking(impl_->preview_semantic, decoded.bytes);
        const std::string_view presentation_bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, presentation_bytes, impl_->policy,
                         impl_->preview_output, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    const bool complete = tokens.size() == expected.size();
    if (complete && impl_->preview_semantic.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the thinking phase");
    }
    if (complete && impl_->split_reasoning && impl_->preview_state.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the reasoning channel");
    }
    impl_->preview_semantic.control_pending = !complete;
    impl_->preview_semantic.applied         = complete;
    impl_->preview_semantic.injected_tokens += static_cast<std::uint32_t>(tokens.size());
    impl_->preview_finish_reason = FinishReason::None;
    if (impl_->grammar) {
        auto grammar = impl_->grammar->fork();
        grammar->accept(tokens);
        impl_->preview_grammar = std::move(grammar);
    }
    impl_->preview_ready = true;
    return runtime::OutputDecision{
        .accepted_tokens              = static_cast<std::uint32_t>(tokens.size()),
        .continuation                 = complete ? runtime::ContinuationAction::Decode
                                                 : runtime::ContinuationAction::ApplyTargetControl,
        .prefix_execution_split_after = impl_->preview_execution_split_after,
    };
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_semantic.control_pending = false;
    impl_->preview_output.clear();
    impl_->preview_logprobs.clear();
    impl_->preview_grammar.reset();
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, 0);
    release_logprobs(impl_->preview_state, impl_->preview_logprobs);
    impl_->preview_finish_reason = reason;
    impl_->preview_ready         = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    if (impl_->preview_grammar) {
        *impl_->grammar = std::move(*impl_->preview_grammar);
        impl_->preview_grammar.reset();
    }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    swap(impl_->semantic, impl_->preview_semantic);
    swap(impl_->prefix_execution, impl_->preview_prefix_execution);
    impl_->commit_loop_words();
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->committed_logprobs = std::move(impl_->preview_logprobs);
    impl_->preview_logprobs.clear();
    impl_->preview_ready = false;

    for (OutputDelta& delta : output) {
        if (delta.channel == OutputChannel::Content) {
            delta.text = impl_->gemma_tools   ? impl_->gemma_tools->feed(delta.text)
                         : impl_->flint_tools ? impl_->flint_tools->feed(delta.text)
                                              : impl_->tool_call_output.feed(delta.text);
        } else if (delta.channel == OutputChannel::Reasoning && impl_->collect_open_reasoning) {
            impl_->open_reasoning.append(delta.text);
        }
    }
    if (impl_->collect_open_reasoning && !impl_->state.in_reasoning) {
        impl_->collect_open_reasoning = false;
        impl_->open_reasoning         = {};
    }
    if (impl_->state.terminal) {
        fi::ToolCallOutputDecoder::Terminal terminal;
        if (impl_->gemma_tools) {
            auto gemma          = impl_->gemma_tools->finish();
            terminal.content    = std::move(gemma.content);
            terminal.tool_calls = std::move(gemma.tool_calls);
            terminal.diagnostics = gemma.diagnostics;
        } else if (impl_->flint_tools) {
            auto flint           = impl_->flint_tools->finish();
            terminal.content     = std::move(flint.content);
            terminal.tool_calls  = std::move(flint.tool_calls);
            terminal.diagnostics = flint.diagnostics;
        } else {
            // The collected thinking survives only when the turn ended without closing it.
            terminal =
                impl_->tool_call_output.finish(impl_->preview_finish_reason, impl_->open_reasoning);
        }
        impl_->tool_calls                            = std::move(terminal.tool_calls);
        impl_->tool_call_parse                       = terminal.diagnostics;
        if (!terminal.content.empty()) {
            OutputDelta* content = nullptr;
            for (OutputDelta& delta : output) {
                if (delta.channel == OutputChannel::Content) { content = &delta; }
            }
            if (content != nullptr) {
                content->text += terminal.content;
            } else {
                output.push_back(OutputDelta{.channel = OutputChannel::Content,
                                             .text    = std::move(terminal.content)});
            }
        }
    }
    return output;
}

std::vector<TokenLogprob> OutputSession::take_content_logprobs() noexcept {
    return impl_ != nullptr ? std::exchange(impl_->committed_logprobs, {})
                            : std::vector<TokenLogprob>{};
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->tool_call_parse : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

bool OutputSession::reasoning_closed() const noexcept {
    return impl_ != nullptr && impl_->started_in_reasoning && !impl_->prefix_execution.tracking;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{
        .requested_budget = impl_->requested_budget,
        // Without early close the cap is never enforced, so there is no effective budget.
        .effective_budget = impl_->early_close_available ? impl_->semantic.budget : std::nullopt,
        .model_thinking_tokens = impl_->semantic.model_thinking_tokens,
        .injected_tokens       = impl_->semantic.injected_tokens,
        .applied               = impl_->semantic.applied,
        .loop_detected         = impl_->semantic.loop_detected,
        .loop_thinking_tokens  = impl_->semantic.loop_thinking_tokens,
        .loop_coverage         = impl_->semantic.loop_coverage,
    };
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->state.matched_stop_order;
    if (index >= impl_->policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->policy.strings[index].text;
}

} // namespace ninfer::models::qwen3_5
