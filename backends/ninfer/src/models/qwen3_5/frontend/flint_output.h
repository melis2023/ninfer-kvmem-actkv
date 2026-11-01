#pragma once

// Flint's output markup in the shared frontend: a reasoning block `<think> ... </think>` that the
// model opens itself when it decides to reason (its chat template has no thinking switch), and
// GLM-4.7 tool calls `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...
// </tool_call>`, whose values are text: a string as it is, anything else as its JSON.

#include "ninfer/request.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class Tokenizer;
struct ToolCallOutputContract;

// The reasoning markers, single tokens of Flint's vocabulary; -1 where the vocabulary lacks one.
struct FlintMarkers {
    std::int32_t think_open  = -1; // <think>
    std::int32_t think_close = -1; // </think>

    [[nodiscard]] static FlintMarkers from(const Tokenizer& tokenizer);
};

inline constexpr std::string_view kFlintThinkOpen  = "<think>";
inline constexpr std::string_view kFlintThinkClose = "</think>";
inline constexpr std::string_view kFlintCallOpen   = "<tool_call>";
inline constexpr std::string_view kFlintCallClose  = "</tool_call>";
inline constexpr std::string_view kFlintKeyOpen    = "<arg_key>";
inline constexpr std::string_view kFlintKeyClose   = "</arg_key>";
inline constexpr std::string_view kFlintValueOpen  = "<arg_value>";
inline constexpr std::string_view kFlintValueClose = "</arg_value>";

// Pulls Flint tool calls out of the content stream: text outside calls is published as it arrives
// (a possible opener's prefix is held back), each complete call becomes a structured call whose
// arguments the request's declared parameter types read (as Qwen's untyped arguments are read).
// An unterminated or unreadable call returns to content at the end.
//
// A forced call is one the prompt already opened: it ends with `<tool_call>NAME`, so generation
// resumes after the function name and the decoder starts in that call. The model never emits those
// bytes, so they join the call but never the content a failed call returns.
class FlintToolCallDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    explicit FlintToolCallDecoder(std::shared_ptr<const ToolCallOutputContract> contract);

    [[nodiscard]] std::string feed(std::string_view text);
    [[nodiscard]] Terminal finish();

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string pending_;
    std::string region_;
    // Bytes at the head of region_ that came from the prompt rather than from the model; non-zero
    // only while the region being read is the forced call.
    std::size_t seeded_bytes_ = 0;
    bool in_call_             = false;
    std::vector<GeneratedToolCall> calls_;
    ToolCallParseDiagnostics diagnostics_;
};

// `NAME<arg_key>...` (the text between the call markers) as a structured call; false when
// unreadable.
[[nodiscard]] bool parse_flint_call(std::string_view region,
                                    const ToolCallOutputContract& contract, GeneratedToolCall& out,
                                    ToolCallParseDiagnostics& diagnostics);

} // namespace ninfer::models::qwen3_5::frontend
