// Flint in the shared frontend.
//
// Pre-tokenizer: the word ends of Flint's three splits (digits in groups of up to three, CJK runs,
// o200k's words) against Hugging Face tokenizers' own pre-tokenization of the same texts
// (tests/fixtures/flint/pretokenizer_cases.json, generated from the release's tokenizer.json). A
// byte vocabulary without merges makes a token of every byte, so the stable frontier of a boundary
// at every byte is the last word end at or before it. A long run that continues a kept one reuses
// its words only where the next code point decides them: the same text encodes alike with and
// without the kept run.
//
// Chat template: the release's template (tests/fixtures/flint/chat_template.jinja, Apache-2.0,
// RecursiveSI/flint) renders the turns as Hugging Face renders them, each turn ending where the
// next role marker begins; disabled thinking closes an empty reasoning block after the assistant
// marker, and a forced call opens `<tool_call>NAME` after it.
//
// Tool calls: FlintToolCallDecoder pulls GLM-4.7 calls out of streamed content (markers split at
// every byte), types their text values by the declared parameters, returns an unreadable or
// unterminated call to content, and starts a forced call inside its opener.

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/flint_output.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#error "NINFER_SOURCE_DIR is required"
#endif

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
using Json   = nlohmann::json;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::string read_fixture(const std::string& name) {
    std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/flint/" + name,
                       std::ios::binary);
    if (!file) { throw std::runtime_error("missing fixture " + name); }
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

// GPT-2's byte-to-symbol table, which byte-level BPE vocabularies are written in.
std::string byte_level_symbol(std::uint8_t byte) {
    const auto visible = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int code = byte;
    if (!visible(byte)) {
        int next = 256;
        for (int b = 0; b < byte; ++b) {
            if (!visible(b)) { ++next; }
        }
        code = next;
    }
    std::string out;
    if (code < 0x80) {
        out.push_back(static_cast<char>(code));
    } else {
        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    return out;
}

// A byte vocabulary without merges behind Flint's pipeline: no normalizer and its pre-tokenizer.
fi::Tokenizer byte_tokenizer() {
    Json vocab = Json::object();
    for (int byte = 0; byte < 256; ++byte) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(byte))] = byte;
    }
    const Json split = [] {
        Json splits = Json::array();
        for (const char* pattern :
             {"\\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+",
              "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
              "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}"
              "\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+"
              "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}) {
            splits.push_back({{"type", "Split"},
                              {"pattern", {{"Regex", pattern}}},
                              {"behavior", "Isolated"},
                              {"invert", false}});
        }
        splits.push_back({{"type", "ByteLevel"},
                          {"add_prefix_space", false},
                          {"trim_offsets", true},
                          {"use_regex", false}});
        return splits;
    }();
    const std::string tokenizer_json =
        Json{{"normalizer", nullptr},
             {"pre_tokenizer", {{"type", "Sequence"}, {"pretokenizers", split}}},
             {"decoder", {{"type", "ByteLevel"}}},
             {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
             {"added_tokens", Json::array()}}
            .dump();
    const std::string config = Json{{"added_tokens_decoder", Json::object()}}.dump();
    return fi::Tokenizer({.tokenizer_json         = tokenizer_json,
                          .tokenizer_config_json  = config,
                          .generation_config_json = R"({"eos_token_id":0})"});
}

// The word ends of `text`: the distinct stable frontiers of a boundary at every byte.
std::vector<std::size_t> word_ends(const fi::Tokenizer& tokenizer, const std::string& text) {
    std::vector<std::size_t> boundaries(text.size() + 1);
    for (std::size_t i = 0; i <= text.size(); ++i) { boundaries[i] = i; }
    const fi::BoundaryEncodedText encoded = tokenizer.encode_with_boundaries(text, boundaries);
    std::set<std::size_t> ends;
    for (const auto& boundary : encoded.boundaries) {
        if (boundary.stable_frontier != 0) { ends.insert(boundary.stable_frontier); }
    }
    return {ends.begin(), ends.end()};
}

void test_pretokenizer() {
    const fi::Tokenizer tokenizer = byte_tokenizer();
    const Json cases              = Json::parse(read_fixture("pretokenizer_cases.json"));
    for (const Json& item : cases) {
        const std::string text = item.at("text").get<std::string>();
        const auto expected    = item.at("word_ends").get<std::vector<std::size_t>>();
        const auto actual      = word_ends(tokenizer, text);
        if (actual != expected) {
            std::ostringstream got;
            for (const std::size_t end : actual) { got << end << ' '; }
            expect(false, "word ends of " + Json(text).dump() + ": " + got.str());
        }
    }
    // Encoding keeps every byte: no normalization runs (a decomposed accent stays decomposed).
    const std::string decomposed = "e\xcc\x81";
    expect(tokenizer.encode(decomposed).size() == decomposed.size(), "no NFC before Flint's BPE");
}

void test_run_reuse() {
    // A kept run ends its words where a lower-case letter later on would join a capital run to the
    // word before it; a run that shares it up to that letter must not reuse that word end.
    std::string prefix;
    while (prefix.size() < 6000) { prefix += "word, more text; "; }
    prefix += "\xe0\xb9\x84\xe0\xb8\x97\xe0\xb8\xa2" "ABCDEFGHIJ"; // Thai, then capitals
    const std::string first  = prefix + " tail, end.";
    const std::string second = prefix + "k tail, end.";
    const fi::Tokenizer warm = byte_tokenizer();
    (void)word_ends(warm, first);
    const fi::Tokenizer cold = byte_tokenizer();
    expect(word_ends(warm, second) == word_ends(cold, second),
           "a kept run's word end before a capital was reused");
    // Its punctuation-led word ends are reused, and agree with a fresh encoding.
    const std::string third = prefix.substr(0, 3000) + "different continuation, here.";
    expect(word_ends(warm, third) == word_ends(cold, third), "reused punctuation word ends");
}

void test_chat_template() {
    const auto compiled = fi::CompiledChatTemplate::resolve(
        read_fixture("chat_template.jinja"), "flint",
        Json{{"bos_token", "<|begin_of_text|>"}, {"eos_token", "<|end_of_text|>"}});
    std::vector<fi::ChatMessage> messages(4);
    messages[0].role = ninfer::ChatRole::System;
    messages[0].parts.push_back(fi::ChatPart::text_part("Be brief."));
    messages[1].role = ninfer::ChatRole::User;
    messages[1].parts.push_back(fi::ChatPart::text_part("Weather in Paris?"));
    messages[2].role              = ninfer::ChatRole::Assistant;
    messages[2].reasoning_content = "Call the tool.";
    messages[2].tool_calls.push_back(
        {.id = "c1", .name = "get_weather", .arguments_json = R"({"city":"Paris","days":3})"});
    messages[3].role = ninfer::ChatRole::Tool;
    messages[3].parts.push_back(fi::ChatPart::text_part("sunny"));
    fi::ChatRenderOptions options;
    options.tool_call_syntax = fi::ToolCallSyntax::Flint; // the frontend's choice for Flint
    options.tool_jsons       = {
        R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}})"};
    const fi::RenderedChat rendered = compiled.render(messages, options);
    const std::string tools =
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user "
        "query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", "
        "\"properties\": {\"city\": {\"type\": \"string\"}}}}</tools>\n\nFor each function "
        "call, output the function name and "
        "arguments within the following XML format:\n<tool_call>{function-name}<arg_key>"
        "{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n";
    const std::string expected =
        "<|system|>Be brief." + tools + "<|user|>Weather in Paris?<|assistant|><think>Call the "
        "tool.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call><|observation|>"
        "<tool_response>sunny</tool_response><|assistant|>";
    expect(rendered.text == expected, "rendered chat:\n" + rendered.text);
    expect(!rendered.starts_in_reasoning, "a Flint prompt does not open reasoning");
    // Each turn ends where the next role marker begins.
    const auto at = [&](std::string_view marker, std::size_t from = 0) {
        return std::optional<std::size_t>(rendered.text.find(marker, from));
    };
    expect(rendered.message_boundaries.size() == 5, "message boundary count");
    if (rendered.message_boundaries.size() == 5) {
        expect(rendered.message_boundaries[1] == at("<|system|>", 1), "system message boundary");
        expect(rendered.message_boundaries[2] == at("<|assistant|>"), "user message boundary");
        expect(rendered.message_boundaries[3] == at("<|observation|>"),
               "assistant message boundary");
    }
    // Disabled thinking: an empty reasoning block after the assistant marker, by enable_thinking or
    // by reasoning effort none; the boundaries stay where they were.
    options.enable_thinking            = false;
    const fi::RenderedChat no_thinking = compiled.render(messages, options);
    expect(no_thinking.text == expected + "<think></think>", "thinking off: " + no_thinking.text);
    expect(no_thinking.message_boundaries == rendered.message_boundaries,
           "thinking off keeps the message boundaries");
    options.enable_thinking  = std::nullopt;
    options.reasoning_effort = ninfer::ReasoningEffort::None;
    expect(compiled.render(messages, options).text == no_thinking.text, "reasoning effort none");
    options.reasoning_effort = std::nullopt;
    // A forced call (thinking off) opens after the empty block, in Flint's syntax.
    options.enable_thinking       = false;
    options.forced_tool_name      = "get_weather";
    const fi::RenderedChat forced = compiled.render(messages, options);
    expect(forced.text == expected + "<think></think><tool_call>get_weather",
           "forced call opener: " + forced.text);
}

std::shared_ptr<const fi::ToolCallOutputContract> weather_contract(std::string_view forced = {}) {
    const std::vector<std::string> tools = {
        R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"code":{"type":"string"},"options":{"type":"object"}}}}})"};
    return fi::build_tool_call_output_contract(tools, true, forced);
}

fi::FlintToolCallDecoder::Terminal run(const std::string& text, std::string& streamed,
                                       std::string_view forced = {}) {
    fi::FlintToolCallDecoder decoder(weather_contract(forced));
    for (const char byte : text) { streamed += decoder.feed(std::string(1, byte)); }
    auto terminal = decoder.finish();
    streamed += terminal.content;
    return terminal;
}

void test_tool_calls() {
    {
        std::string streamed;
        const auto terminal =
            run("Checking. <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"
                "</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>code"
                "</arg_key><arg_value>007</arg_value><arg_key>options</arg_key><arg_value>"
                "{\"units\": \"metric\"}</arg_value></tool_call> Done.",
                streamed);
        expect(streamed == "Checking.  Done.", "text around a call: " + streamed);
        expect(terminal.tool_calls.size() == 1, "one call");
        if (terminal.tool_calls.size() == 1) {
            expect(terminal.tool_calls[0].name == "get_weather", "call name");
            expect(Json::parse(terminal.tool_calls[0].arguments_json) ==
                       Json({{"city", "Paris"},
                             {"days", 3},
                             {"code", "007"},
                             {"options", {{"units", "metric"}}}}),
                   "typed arguments: " + terminal.tool_calls[0].arguments_json);
        }
        expect(terminal.diagnostics.structured_call_count == 1, "call count");
    }
    {
        // Two calls, a call without arguments, whitespace between the name and its arguments.
        std::string streamed;
        const auto terminal =
            run("<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Oslo</arg_value>"
                "</tool_call><tool_call>get_weather</tool_call>",
                streamed);
        expect(streamed.empty(), "nothing but calls");
        expect(terminal.tool_calls.size() == 2, "two calls");
        if (terminal.tool_calls.size() == 2) {
            expect(Json::parse(terminal.tool_calls[0].arguments_json) ==
                       Json({{"city", "Oslo"}}),
                   "first call arguments");
            expect(terminal.tool_calls[1].arguments_json == "{}", "a call without arguments");
        }
    }
    {
        // An unreadable call returns to content as it was written.
        std::string streamed;
        const std::string broken = "<tool_call>get_weather<arg_key>city</arg_key>Paris</tool_call>";
        const auto terminal      = run("a " + broken + " b", streamed);
        expect(terminal.tool_calls.empty(), "no call from broken markup");
        expect(streamed == "a " + broken + " b", "broken call returns to content: " + streamed);
        expect(terminal.diagnostics.fallback_reason ==
                   ninfer::ToolCallParseFallbackReason::MalformedStructure,
               "broken call diagnostics");
    }
    {
        // An unterminated call that reads completely is a call; one that does not is content.
        std::string streamed;
        auto terminal = run("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome"
                            "</arg_value>",
                            streamed);
        expect(terminal.tool_calls.size() == 1 && streamed.empty(), "complete unterminated call");
        streamed.clear();
        terminal = run("<tool_call>get_weather<arg_key>city</arg_key><arg_val", streamed);
        expect(terminal.tool_calls.empty() &&
                   streamed == "<tool_call>get_weather<arg_key>city</arg_key><arg_val",
               "incomplete call returns to content");
    }
    {
        // A forced call: the prompt holds `<tool_call>get_weather`; the model writes the rest.
        std::string streamed;
        auto terminal = run("<arg_key>city</arg_key><arg_value>Lima</arg_value>", streamed,
                            "get_weather");
        expect(terminal.tool_calls.size() == 1 && streamed.empty(), "forced call");
        expect(terminal.diagnostics.forced_call_closed, "forced call closed by the decoder");
        streamed.clear();
        terminal = run("<arg_key>city", streamed, "get_weather");
        expect(terminal.tool_calls.empty() && streamed == "<arg_key>city",
               "a failed forced call returns only the model's bytes: " + streamed);
    }
}

} // namespace

int main() {
    try {
        test_pretokenizer();
        test_run_reuse();
        test_chat_template();
        test_tool_calls();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
