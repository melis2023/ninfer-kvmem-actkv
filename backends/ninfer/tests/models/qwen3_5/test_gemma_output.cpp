// Gemma 4's tool-call markup in the shared frontend: GemmaToolCallDecoder pulls
// `<|tool_call>call:NAME{key:value,...}<tool_call|>` regions out of streamed content (markers split
// across feeds included) and returns them as structured calls with JSON arguments; text around
// them stays content, and an unreadable or unterminated call returns to content. A forced call is
// already open in the prompt (`<|tool_call>call:NAME{`): the decoder starts inside it, a turn that
// ends on the closed arguments completes it, and a failed one returns only the model's own bytes.

#include "models/qwen3_5/frontend/gemma_output.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::ToolCallParseFallbackReason;
using ninfer::models::qwen3_5::frontend::GemmaToolCallDecoder;
using ninfer::models::qwen3_5::frontend::parse_gemma_call;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

// `streamed` collects what feed() published during the stream; the terminal content is added to it.
GemmaToolCallDecoder::Terminal run(const std::vector<std::string>& pieces, std::string& streamed,
                                   std::string_view forced_tool_name = {}) {
    GemmaToolCallDecoder decoder(forced_tool_name);
    for (const auto& piece : pieces) { streamed += decoder.feed(piece); }
    auto terminal = decoder.finish();
    streamed += terminal.content;
    return terminal;
}

// Feeds the text one byte at a time: every marker is split at every possible point.
GemmaToolCallDecoder::Terminal run_bytewise(const std::string& text, std::string& streamed,
                                            std::string_view forced_tool_name = {}) {
    std::vector<std::string> pieces;
    for (const char byte : text) { pieces.emplace_back(1, byte); }
    return run(pieces, streamed, forced_tool_name);
}

} // namespace

int main() {
    const std::string q = "<|\"|>";
    {
        ninfer::GeneratedToolCall call;
        expect(parse_gemma_call("call:get_weather{city:" + q + "Paris" + q + ",days:3}", call),
               "parse a call");
        expect(call.name == "get_weather", "call name");
        const auto args = nlohmann::json::parse(call.arguments_json);
        expect(args == nlohmann::json({{"city", "Paris"}, {"days", 3}}), "call arguments");
    }
    {
        // Nested object and array values, a boolean, a string holding braces and commas.
        ninfer::GeneratedToolCall call;
        expect(parse_gemma_call("call:f{a:{b:[1,2," + q + "x,}" + q + "]},flag:true}", call),
               "parse nested values");
        const auto args = nlohmann::json::parse(call.arguments_json);
        expect(args == nlohmann::json({{"a", {{"b", {1, 2, "x,}"}}}}, {"flag", true}}),
               "nested arguments");
    }
    {
        ninfer::GeneratedToolCall call;
        expect(!parse_gemma_call("get_weather{city:Paris}", call), "reject a region without call:");
    }
    {
        // Streamed with the markers split across feeds, text before the call kept as content.
        std::string streamed;
        const auto terminal = run({"Checking. <|tool", "_call>call:get_weather{city:" + q, "Paris",
                                   q + ",days:3}<tool_", "call|>"},
                                  streamed);
        expect(terminal.tool_calls.size() == 1, "one streamed call");
        expect(streamed == "Checking. ", "content around the call: '" + streamed + "'");
        if (!terminal.tool_calls.empty()) {
            expect(terminal.tool_calls[0].name == "get_weather", "streamed call name");
        }
    }
    {
        // Two calls in one turn.
        std::string streamed;
        const auto terminal =
            run({"<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:" + q + "z" + q +
                 "}<tool_call|>"},
                streamed);
        expect(terminal.tool_calls.size() == 2, "two calls");
        expect(streamed.empty(), "no content beside two calls");
    }
    {
        // An unterminated call returns to content.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1"}, streamed);
        expect(terminal.tool_calls.empty(), "unterminated call is not a call");
        expect(streamed.find("call:a{x:1") != std::string::npos, "unterminated call stays content");
    }
    {
        // An unterminated call that does not parse reports why it went back to content.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1"}, streamed);
        expect(terminal.diagnostics.fallback_reason == ToolCallParseFallbackReason::MalformedStructure,
               "unterminated unreadable call reports malformed_structure");
        expect(!terminal.diagnostics.forced_call_closed, "a free call is never a closed forced call");
    }
    {
        // A free call that ends on its closing brace is read as before, and is not a forced one.
        std::string streamed;
        const auto terminal = run({"<|tool_call>call:a{x:1}"}, streamed);
        expect(terminal.tool_calls.size() == 1 && streamed.empty(), "free call without marker");
        expect(!terminal.diagnostics.forced_call_closed, "free call is not a closed forced call");
    }

    // Forced call: the prompt supplied `<|tool_call>call:NAME{`, the model continues with arguments.
    {
        const std::string text = "caption:" + q + "a cat, <b>}" + q + ",count:3,ratio:-0.5,ok:true,"
                                 "off:false,none:null,tags:[" + q + "x" + q + ",2],"
                                 "opts:{n:{k:" + q + "v" + q + "}}}<tool_call|>";
        const auto expected    = nlohmann::json::parse(
            R"({"caption":"a cat, <b>}","count":3,"ratio":-0.5,"ok":true,"off":false,)"
            R"("none":null,"tags":["x",2],"opts":{"n":{"k":"v"}}})");
        std::string whole;
        const auto one = run({text}, whole, "draw_image");
        expect(one.tool_calls.size() == 1 && one.tool_calls[0].name == "draw_image",
               "forced call is one call with the forced name");
        if (!one.tool_calls.empty()) {
            expect(nlohmann::json::parse(one.tool_calls[0].arguments_json) == expected,
                   "forced call arguments: " + one.tool_calls[0].arguments_json);
        }
        expect(whole.empty(), "a forced call leaves no content: '" + whole + "'");
        expect(one.diagnostics.marker_seen && one.diagnostics.structured_call_count == 1 &&
                   !one.diagnostics.forced_call_closed &&
                   one.diagnostics.fallback_reason == ToolCallParseFallbackReason::None,
               "forced call diagnostics");

        // The same text split at every byte, including inside <|"|> and <tool_call|>.
        std::string bytewise;
        const auto split = run_bytewise(text, bytewise, "draw_image");
        expect(split.tool_calls.size() == 1 && bytewise.empty(), "byte-split forced call");
        if (!split.tool_calls.empty()) {
            expect(nlohmann::json::parse(split.tool_calls[0].arguments_json) == expected,
                   "byte-split forced call arguments");
        }
        expect(!split.diagnostics.forced_call_closed, "a terminated forced call was not closed");

        // Fragments that end mid-marker at several points.
        for (const std::size_t cut : {text.size() - 1, text.size() - 6, text.size() - 11}) {
            std::string pieces_streamed;
            const auto pieces = run({text.substr(0, cut), text.substr(cut)}, pieces_streamed,
                                    "draw_image");
            expect(pieces.tool_calls.size() == 1 && pieces_streamed.empty() &&
                       !pieces.diagnostics.forced_call_closed,
                   "forced call split at " + std::to_string(cut));
        }
    }
    {
        // The turn ends on the closing brace: the call is complete without its marker.
        std::string streamed;
        const auto terminal = run({"caption:" + q + "x" + q + "}"}, streamed, "draw_image");
        expect(terminal.tool_calls.size() == 1 && terminal.tool_calls[0].name == "draw_image",
               "unterminated forced call completes");
        expect(terminal.diagnostics.forced_call_closed, "forced_call_closed on a missing marker");
        expect(terminal.diagnostics.fallback_reason == ToolCallParseFallbackReason::None &&
                   streamed.empty(),
               "a completed forced call leaves no fallback and no content");
        std::string bytewise;
        const auto split = run_bytewise("caption:" + q + "x" + q + "}", bytewise, "draw_image");
        expect(split.tool_calls.size() == 1 && split.diagnostics.forced_call_closed &&
                   bytewise.empty(),
               "byte-split unterminated forced call");
    }
    {
        // A function with no arguments.
        std::string streamed;
        const auto terminal = run({"}"}, streamed, "ping");
        expect(terminal.tool_calls.size() == 1 && terminal.tool_calls[0].arguments_json == "{}" &&
                   terminal.diagnostics.forced_call_closed,
               "forced call without arguments");
    }
    {
        // Less than a complete call is never completed: the model's own bytes return as content,
        // without the prompt's `<|tool_call>call:NAME{`.
        const std::vector<std::string> unfinished = {
            "caption:" + q + "x" + q + ",cou", "caption:" + q + "x", "caption:" + q + "x" + q + ",",
            "a:{b:1}", std::string{}, "caption:" + q + "x" + q + ",opts:{a:1}"};
        for (const std::string& text : unfinished) {
            std::string streamed;
            const auto terminal = run({text}, streamed, "draw_image");
            expect(terminal.tool_calls.empty(), "unfinished forced call is not a call: '" + text + "'");
            expect(streamed == text, "unfinished forced call returns the model's bytes: '" + text +
                                         "' -> '" + streamed + "'");
            expect(!terminal.diagnostics.forced_call_closed &&
                       terminal.diagnostics.fallback_reason ==
                           ToolCallParseFallbackReason::MalformedStructure,
                   "unfinished forced call reports malformed_structure: '" + text + "'");
        }
        // A terminated region that does not read keeps the model's closing marker.
        std::string streamed;
        const auto terminal = run({"caption " + q + "x" + q + "}<tool_call|>"}, streamed, "draw_image");
        expect(terminal.tool_calls.empty() &&
                   streamed == "caption " + q + "x" + q + "}<tool_call|>" &&
                   terminal.diagnostics.fallback_reason ==
                       ToolCallParseFallbackReason::MalformedStructure,
               "unreadable terminated forced call: '" + streamed + "'");
    }
    {
        // The forced call comes first; later text and calls are the model's own. A later call that
        // lacks its marker is a free call, not a closed forced one.
        std::string streamed;
        const auto terminal = run({"id:1}<tool_call|>Done. <|tool_call>call:b{y:2}"}, streamed, "a");
        expect(terminal.tool_calls.size() == 2 && terminal.tool_calls[0].name == "a" &&
                   terminal.tool_calls[1].name == "b",
               "forced call followed by a free call");
        expect(streamed == "Done. ", "text between calls stays content: '" + streamed + "'");
        expect(!terminal.diagnostics.forced_call_closed, "only the forced call can be closed");

        // A failed forced region does not leak its seeded head into the next region's content.
        std::string failed;
        const auto second = run({"junk<tool_call|><|tool_call>call:b{y:2}<tool_call|>"}, failed, "a");
        expect(second.tool_calls.size() == 1 && second.tool_calls[0].name == "b" &&
                   failed == "junk<tool_call|>",
               "failed forced region then a free call: '" + failed + "'");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma output\n";
    return failures == 0 ? 0 : 1;
}
