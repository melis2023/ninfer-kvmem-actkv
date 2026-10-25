// Raw-prompt completion on the wire, without an Engine: llama.cpp's /completion and OpenAI's legacy
// /v1/completions. Covered: every prompt form llama.cpp accepts (and the several-prompt form it
// refuses here), llama.cpp's sampler controls at and away from their neutral values, the output
// limit spellings, json_schema, and the response objects and stream sequences of both dialects.

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/text_completion.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

RequestLimits limits() {
    RequestLimits out;
    out.default_max_tokens = 256;
    out.max_context        = 4096;
    return out;
}

TextCompletionRequest parse(const char* body,
                            TextCompletionDialect dialect = TextCompletionDialect::LlamaCpp) {
    return parse_text_completion_request(RequestJson::parse(body), limits(), dialect);
}

// The error code a body is refused with, or "" when it parses.
std::string refusal(const char* body,
                    TextCompletionDialect dialect = TextCompletionDialect::LlamaCpp) {
    try {
        (void)parse(body, dialect);
    } catch (const ApiException& error) {
        return error.error().code.empty() ? error.error().message : error.error().code;
    }
    return {};
}

void prompt_forms() {
    const auto text = parse(R"({"prompt": "Hello"})");
    expect(text.generation.raw_prompt && text.generation.raw_prompt->size() == 1 &&
               text.generation.raw_prompt->front().text == "Hello",
           "a string prompt is one text piece");
    const auto tokens = parse(R"({"prompt": [1, 2, 3]})");
    expect(tokens.generation.raw_prompt->size() == 3 &&
               tokens.generation.raw_prompt->at(1).token == 2,
           "token ids pass through");
    const auto mixed = parse(R"({"prompt": [7, "text", 9]})");
    expect(mixed.generation.raw_prompt->size() == 3 &&
               mixed.generation.raw_prompt->at(1).text == "text" &&
               !mixed.generation.raw_prompt->at(1).token,
           "token ids and text mix in one prompt");
    const auto nested = parse(R"({"prompt": [[5, "x"]]})");
    expect(nested.generation.raw_prompt->size() == 2, "an array holding one prompt is that prompt");
    const auto listed = parse(R"({"prompt": ["only"]})");
    expect(listed.generation.raw_prompt->front().text == "only",
           "a list of one string prompt is that prompt");
    expect(refusal(R"({"prompt": ["a", "b"]})") == "multiple_prompts_not_supported",
           "several prompts are refused");
    expect(refusal(R"({"prompt": [[1], [2]]})") == "multiple_prompts_not_supported",
           "several token prompts are refused");
    expect(!refusal(R"({"prompt": ""})").empty(), "an empty prompt is refused");
    expect(!refusal(R"({"prompt": []})").empty(), "an empty token prompt is refused");
    expect(!refusal(R"({"prompt": [-1]})").empty(), "a negative token id is refused");
    expect(!refusal(R"({})").empty(), "a missing prompt is refused");
    expect(refusal(R"({"prompt": {"prompt_string": "x", "multimodal_data": []}})") ==
               "multimodal_prompt_not_supported",
           "a multimodal prompt object is refused");
}

void controls() {
    expect(refusal(R"({"prompt": "x", "repeat_penalty": 1.0, "mirostat": 0, "typical_p": 1,
                      "top_n_sigma": -1, "n_probs": 0, "grammar": "", "lora": [],
                      "logit_bias": [], "samplers": ["top_k"], "n_keep": 4, "id_slot": -1})")
               .empty(),
           "neutral llama.cpp controls are accepted");
    expect(refusal(R"({"prompt": "x", "repeat_penalty": 1.1})") == "repeat_penalty_not_supported",
           "a repeat penalty is refused");
    expect(refusal(R"({"prompt": "x", "mirostat": 2})") == "mirostat_not_supported",
           "mirostat is refused");
    expect(refusal(R"({"prompt": "x", "n_probs": 5})") == "n_probs_not_supported",
           "per-token probabilities are refused");
    expect(refusal(R"({"prompt": "x", "grammar": "root ::= \"a\""})") ==
               "constrained_decoding_not_supported",
           "a GBNF grammar is refused");

    const auto sampling = parse(R"({"prompt": "x", "temperature": -1, "top_k": -1, "seed": -1,
                                   "min_p": 0.1})");
    expect(sampling.generation.sampling.temperature == 0.0, "a negative temperature is greedy");
    expect(sampling.generation.sampling.top_k == 0, "top_k <= 0 is the widest set");
    expect(!sampling.generation.sampling.seed, "a negative seed is random");
    expect(sampling.generation.sampling.min_p == 0.1, "min_p");

    const auto unlimited = parse(R"({"prompt": "x", "n_predict": -1})");
    expect(unlimited.generation.derive_output_budget && unlimited.output_tokens_explicit,
           "n_predict -1 is no limit");
    const auto limited = parse(R"({"prompt": "x", "n_predict": 12, "max_tokens": 99})");
    expect(limited.generation.max_tokens == 12, "n_predict wins over max_tokens");
    const auto defaulted = parse(R"({"prompt": "x"})");
    expect(defaulted.generation.max_tokens == 256 && !defaulted.output_tokens_explicit,
           "an omitted limit takes the server default");
    expect(!refusal(R"({"prompt": "x", "max_tokens": -2})", TextCompletionDialect::OpenAI).empty(),
           "OpenAI's max_tokens accepts only -1 below zero");

    const auto stops = parse(R"({"prompt": "x", "stop": ["a", "b", "c", "d", "e"],
                                "ignore_eos": true, "cache_prompt": false})");
    expect(stops.generation.stop_strings.size() == 5 && stops.generation.ignore_eos &&
               !stops.generation.cache_prompt,
           "llama.cpp stop lists, ignore_eos and cache_prompt");

    const auto schema = parse(R"({"prompt": "x", "json_schema": {"type": "object"}})");
    expect(schema.generation.structured_output.kind == ninfer::StructuredOutputKind::JsonSchema,
           "json_schema constrains the output");
    const auto any_json = parse(R"({"prompt": "x", "json_schema": {}})");
    expect(any_json.generation.structured_output.kind == ninfer::StructuredOutputKind::JsonObject,
           "an empty json_schema admits any JSON object");

    expect(refusal(R"({"prompt": "x", "n": 2})", TextCompletionDialect::OpenAI) ==
               "n_not_supported",
           "OpenAI n > 1 is refused");
    expect(refusal(R"({"prompt": "x", "suffix": "tail"})", TextCompletionDialect::OpenAI) ==
               "suffix_not_supported",
           "fill-in-the-middle is refused");
    expect(refusal(R"({"prompt": "x", "logprobs": 3})", TextCompletionDialect::OpenAI) ==
               "logprobs_not_supported",
           "OpenAI logprobs are refused");
}

GenerationOutcome outcome() {
    GenerationOutcome out;
    out.text                            = "Hello\nworld";
    out.tokens                          = {11, 12, 13};
    out.prompt_tokens                   = 5;
    out.completion_tokens               = 3;
    out.finish_reason                   = ninfer::FinishReason::StopString;
    out.matched_stop_string             = "###";
    out.metrics.prefix_cache_hit_tokens = 2;
    out.metrics.prompt_wall_seconds     = 0.01;
    out.metrics.generation_wall_seconds = 0.02;
    return out;
}

TextCompletionContext context(const TextCompletionRequest& request) {
    PreparedRequest prepared;
    prepared.requested_output_tokens = 64;
    prepared.sampling.temperature    = 0.7F;
    TextCompletionContext out        = make_text_completion_context(request, "Say:", prepared);
    out.model                        = "m";
    return out;
}

std::vector<Json> events(const std::vector<std::string>& chunks) {
    std::vector<Json> out;
    for (const std::string& chunk : chunks) {
        if (chunk == "data: [DONE]\n\n") {
            out.push_back("[DONE]");
            continue;
        }
        out.push_back(Json::parse(chunk.substr(6)));
    }
    return out;
}

void llama_responses() {
    const auto request = parse(R"({"prompt": "Say:", "return_tokens": true, "stop": ["###"]})");
    const Json full    = Json::parse(make_text_completion_response(context(request), outcome()));
    expect(full["content"] == "Hello\nworld" && full["stop"] == true, "content and stop");
    expect(full["stop_type"] == "word" && full["stopping_word"] == "###", "a stop word");
    expect(full["tokens"] == Json::array({11, 12, 13}), "return_tokens");
    expect(full["tokens_predicted"] == 3 && full["tokens_evaluated"] == 5 &&
               full["tokens_cached"] == 2,
           "token counts");
    expect(full["prompt"] == "Say:" && full["has_new_line"] == true && full["model"] == "m",
           "prompt, new line and model");
    expect(full["generation_settings"]["n_predict"] == 64 &&
               full["generation_settings"]["stop"] == Json::array({"###"}),
           "generation settings");
    expect(full["timings"]["cache_n"] == 2 && full["timings"]["prompt_n"] == 3, "timings");

    const auto fields = parse(R"({"prompt": "x", "response_fields": ["content",
                                 "generation_settings/n_predict", "missing"]})");
    const Json picked = Json::parse(make_text_completion_response(context(fields), outcome()));
    expect(picked.size() == 2 && picked["content"] == "Hello\nworld" &&
               picked["generation_settings/n_predict"] == 64,
           "response_fields keeps the named fields under their paths");

    const auto streamed = parse(R"({"prompt": "Say:", "stream": true})");
    TextCompletionStream stream(context(streamed));
    expect(stream.start().empty(), "a llama.cpp stream opens with no chunk");
    stream.note_start(ninfer::GenerationStart{.prompt = {.prompt_tokens = 5}});
    stream.note_timing(ninfer::GenerationTimingObservation{.generated_tokens = 1});
    const Json first = Json::parse(stream.content_delta("Hello").substr(6));
    expect(first["content"] == "Hello" && first["stop"] == false &&
               first["tokens_predicted"] == 1 && first["tokens_evaluated"] == 5,
           "a llama.cpp chunk carries text and running counts");
    const auto rest = events(stream.finish(outcome()));
    expect(rest.size() == 2 && rest[0]["content"] == "\nworld" && rest[1]["stop"] == true &&
               rest[1]["content"] == "" && rest.back() != "[DONE]",
           "the rest of the text, then the final object, and no [DONE]");
}

void openai_responses() {
    const auto request =
        parse(R"({"prompt": "Say:", "echo": true})", TextCompletionDialect::OpenAI);
    auto limited          = outcome();
    limited.finish_reason = ninfer::FinishReason::OutputLimit;
    const Json full       = Json::parse(make_text_completion_response(context(request), limited));
    expect(full["object"] == "text_completion" &&
               full["id"].get<std::string>().starts_with("cmpl-"),
           "a text_completion object");
    expect(full["choices"][0]["text"] == "Say:Hello\nworld", "echo precedes the completion");
    expect(full["choices"][0]["finish_reason"] == "length", "an output limit is length");
    expect(full["usage"]["prompt_tokens"] == 5 && full["usage"]["completion_tokens"] == 3 &&
               full["usage"]["total_tokens"] == 8,
           "usage");

    const auto streamed = parse(R"({"prompt": "Say:", "stream": true, "echo": true,
                                   "stream_options": {"include_usage": true}})",
                                TextCompletionDialect::OpenAI);
    TextCompletionStream stream(context(streamed));
    const auto opening = events(stream.start());
    expect(opening.size() == 1 && opening[0]["choices"][0]["text"] == "Say:",
           "an echoed stream opens with the prompt");
    stream.note_start(ninfer::GenerationStart{.prompt = {.prompt_tokens = 5}});
    (void)stream.content_delta("Hello\nworld");
    const auto closing = events(stream.finish(outcome()));
    expect(closing.size() == 3 && closing[0]["choices"][0]["finish_reason"] == "stop" &&
               closing[1]["choices"].empty() && closing[1]["usage"]["total_tokens"] == 8 &&
               closing[2] == "[DONE]",
           "finish chunk, usage chunk, then [DONE]");
}

} // namespace

int main() {
    try {
        prompt_forms();
        controls();
        llama_responses();
        openai_responses();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "text completion: ok\n";
    return 0;
}
