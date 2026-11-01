#pragma once

// Gemma 4's output markup in the shared frontend: the reasoning channel `<|channel>thought\n ...
// <channel|>` and tool calls `<|tool_call>call:NAME{key:value,...}<tool_call|>`, whose strings are
// delimited by `<|"|>`.

#include "ninfer/request.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;

// The special tokens of the dialect; -1 where the vocabulary lacks one.
struct GemmaMarkers {
    std::int32_t channel_open  = -1; // <|channel>
    std::int32_t channel_close = -1; // <channel|>
    std::int32_t call_open     = -1; // <|tool_call>
    std::int32_t call_close    = -1; // <tool_call|>
    std::int32_t quote         = -1; // <|"|>

    [[nodiscard]] static GemmaMarkers from(const Tokenizer& tokenizer);
    // Whether the token's text is kept in the content stream (tool-call markup).
    [[nodiscard]] bool markup(std::int32_t token) const noexcept {
        return token == call_open || token == call_close || token == quote;
    }
};

inline constexpr std::string_view kGemmaCallOpen  = "<|tool_call>";
inline constexpr std::string_view kGemmaCallClose = "<tool_call|>";
inline constexpr std::string_view kGemmaQuote     = "<|\"|>";
// A call's text between the markers is `call:NAME{...}`: the name follows kGemmaCallHead and the
// argument object opens with kGemmaArgumentsOpen.
inline constexpr std::string_view kGemmaCallHead      = "call:";
inline constexpr std::string_view kGemmaArgumentsOpen = "{";

// Pulls Gemma tool calls out of the content stream: text outside calls is published as it
// arrives (a possible opener's prefix is held back), each complete call becomes a structured
// call with JSON arguments. An unterminated or unreadable call returns to content at the end.
//
// A forced call is one the prompt already opened: it ends with `<|tool_call>call:NAME{`, so
// generation resumes inside the call's arguments and the decoder starts in that call. The model
// never emits those bytes, so they join the call but never the content a failed call returns.
class GemmaToolCallDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    // `forced_tool_name` is the function the prompt's opener names; empty when the model chooses.
    explicit GemmaToolCallDecoder(std::string_view forced_tool_name = {});

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    std::string pending_;
    std::string region_;
    // Bytes at the head of region_ that came from the prompt rather than from the model; non-zero
    // only while the region being read is the forced call.
    std::size_t seeded_bytes_ = 0;
    bool in_call_             = false;
    std::vector<GeneratedToolCall> calls_;
    ToolCallParseDiagnostics diagnostics_;
};

// `call:NAME{...}` (the text between the call markers) as a structured call; false when unreadable.
[[nodiscard]] bool parse_gemma_call(std::string_view region, GeneratedToolCall& out);

} // namespace ninfer::models::qwen3_5::frontend
