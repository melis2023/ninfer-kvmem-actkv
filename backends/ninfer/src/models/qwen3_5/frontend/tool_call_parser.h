#pragma once

#include "ninfer/request.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    enum class Encoding : std::uint8_t { NonStrict, RawString, Json };
    struct Parameter {
        Encoding encoding = Encoding::NonStrict;
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
        // Declared `strict: true`: its arguments are generated under its parameters' JSON Schema
        // (schema_json) rather than as free text.
        bool strict = false;
        std::string schema_json;
        std::size_t declaration_index = 0;
    };

    std::vector<Tool> tools;
    bool constrained = false;
    bool required = false;
    bool parallel = true;
    bool enforce_declared_names = false;
    bool duplicate_names = false;
    // Function whose call opener the rendered prompt already carries, empty when the model chooses
    // freely. It lives here because the prompt that carries the opener is the only thing that knows
    // it: a second, independently supplied copy could name a different tool and the continuation
    // would be attributed to that one. The Qwen decoder and, for a Gemma 4 request, the Gemma
    // decoder both start inside that call; the syntax of the opener is the model's own.
    std::string forced_tool_name;
    std::string continuation;
};

// Markdown code state of a text read so far: an open fence (a run of three or more backticks or
// tildes; a run of the same character at least as long, alone on its line, closes it) or an odd
// count of backticks in the current paragraph (inline code). A tool-call marker of free output
// inside code is a quoted example and stays text. Only the text before a marker decides, so
// streamed and whole outputs agree.
class MarkdownCodeTracker {
public:
    void feed(std::string_view text);
    // Whether the next byte lies inside a code fence or inline code.
    [[nodiscard]] bool in_code() const noexcept;

private:
    void end_line();

    std::string line_;               // the current, unfinished line
    char fence_              = '\0'; // the open fence's character, '\0' when none is open
    std::uint32_t fence_len_ = 0;    // the open fence's opening run length
    std::uint32_t ticks_     = 0;    // backticks of the current paragraph's finished lines
};
[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice);

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled,
                                std::string_view forced_tool_name = {});

// A call whose arguments came as untyped text, (name, value) pairs in order (Qwen's syntax and
// Flint's), with each value read by its declared parameter's types; a mismatch or an undeclared
// parameter is counted in `diagnostics` and kept.
[[nodiscard]] GeneratedToolCall normalize_untyped_tool_call(
    std::string_view name, std::span<const std::pair<std::string_view, std::string_view>> arguments,
    const ToolCallOutputContract& contract, ToolCallParseDiagnostics& diagnostics);

// Native Qwen tool-call envelope for a structured final-response request. A strict tool's
// arguments follow its parameters' JSON Schema; another's remain non-strict values, with each
// declared parameter emitted at most once in declaration order.
[[nodiscard]] std::string structured_tool_call_format(const ToolCallOutputContract& contract);

[[nodiscard]] bool has_strict_tools(const ToolCallOutputContract& contract) noexcept;

// For a response without a structured final format: free text in which every tool call follows
// the envelope above, starting inside the forced call when the prompt opened one. An XGrammar
// structural-tag format. A strict tool's schema that strict arguments cannot honor (unsupported,
// or uniqueItems, which only the final-response path enforces) throws std::invalid_argument.
[[nodiscard]] std::string tool_call_text_format(const ToolCallOutputContract& contract);

// `code` is the Markdown code state of the text before `text`: a marker inside code is skipped.
[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract, MarkdownCodeTracker code = {});

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix; a marker inside Markdown code is published as text. At terminal time, valid calls are
// retained structurally. A region that opens a function but fails the strict reader goes through
// recovery: readable calls are kept and an unreadable one becomes a call to the reserved error
// tool. Only markup that never opens a function is restored verbatim. When the prompt already
// carries the opener of a forced call, the decoder is seeded with that opener, because the model
// never emits it.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    [[nodiscard]] std::string feed(std::string_view text);
    // `open_reasoning` is the thinking of a turn that ended without closing it. When the turn ended
    // on the model's stop token and no content was fed, complete declared calls that end that
    // thinking become the structured turn.
    [[nodiscard]] Terminal finish(FinishReason reason             = FinishReason::StopToken,
                                  std::string_view open_reasoning = {});

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    // A '<' and what follows while it can still grow into a tool-call region opener.
    std::string pending_marker_;
    // Markdown code state of the published text, and of the text before the held region.
    MarkdownCodeTracker code_;
    MarkdownCodeTracker region_code_;
    // Bytes of tool_region_ that came from the prompt's opener rather than from the model. The
    // fallback path returns the region as ordinary content, and these must not appear there.
    std::size_t seeded_prefix_bytes_  = 0;
    std::size_t max_tool_name_length_ = 0;
    bool saw_tool_marker_             = false;
    bool fed_content_                 = false;
    bool forced_                      = false;
    bool finished_                    = false;
};

} // namespace ninfer::models::qwen3_5::frontend
