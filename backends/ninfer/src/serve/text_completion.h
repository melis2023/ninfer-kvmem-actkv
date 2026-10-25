#pragma once

// Raw-prompt completion in two wire dialects: llama.cpp's native POST /completion and OpenAI's
// legacy POST /v1/completions. The prompt is text, token ids, or both mixed in one array;
// generation continues it directly, with no chat template and no reasoning block. Parsing produces
// the protocol envelope plus an executable GenerationRequest; the renderers consume
// protocol-neutral GenerationOutcome values.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;
struct PreparedRequest;

enum class TextCompletionDialect : std::uint8_t {
    LlamaCpp, // POST /completion
    OpenAI,   // POST /v1/completions
};

// How a response is rendered: the request's response options.
struct TextCompletionResponseOptions {
    TextCompletionDialect dialect = TextCompletionDialect::LlamaCpp;
    bool stream                   = false;
    bool timings_per_token        = false;
    bool return_progress          = false;
    // OpenAI: stream_options.include_usage, and echo (the prompt text precedes the completion).
    bool include_usage = false;
    bool echo          = false;
    // llama.cpp: the generated token ids in the response (return_tokens), and the response fields
    // to keep (response_fields; "a/b" names a nested one). Empty keeps every field.
    bool return_tokens = false;
    std::vector<std::string> response_fields;
};

struct TextCompletionRequest {
    std::string model;
    GenerationRequest generation;
    TextCompletionResponseOptions response;
    bool output_tokens_explicit = false;
};

TextCompletionRequest parse_text_completion_request(const RequestJson& body,
                                                    const RequestLimits& limits,
                                                    TextCompletionDialect dialect);

// Everything a response reports besides the outcome: its options and identity, the prompt as text
// (llama.cpp's `prompt`, OpenAI's echo) and the settings the request resolved to.
struct TextCompletionContext {
    TextCompletionResponseOptions response;
    std::string id;
    std::string model;
    std::int64_t created = 0;
    std::string prompt_text;
    ninfer::ResolvedSamplingParameters sampling;
    int requested_output_tokens = 0;
    bool ignore_eos             = false;
    std::vector<std::string> stop;
};

TextCompletionContext make_text_completion_context(const TextCompletionRequest& request,
                                                   std::string prompt_text,
                                                   const PreparedRequest& prepared);

std::string make_text_completion_response(const TextCompletionContext& context,
                                          const GenerationOutcome& outcome);

// One streamed completion. llama.cpp chunks carry the text with running token counts and end with
// the final object; OpenAI chunks are text_completion objects ending in [DONE].
class TextCompletionStream {
public:
    explicit TextCompletionStream(TextCompletionContext context);

    // The OpenAI echo chunk, or nothing.
    std::vector<std::string> start();
    void note_start(const ninfer::GenerationStart& start);
    void note_timing(const ninfer::GenerationTimingObservation& timing);
    // A llama.cpp return_progress chunk.
    std::string prompt_progress(const ninfer::PromptProgress& progress);
    std::string content_delta(const std::string& text);
    std::vector<std::string> finish(const GenerationOutcome& outcome);

private:
    [[nodiscard]] nlohmann::json live_timings() const;

    TextCompletionContext context_;
    std::string content_;
    std::optional<ninfer::GenerationTimingObservation> live_timing_;
    std::uint32_t prompt_tokens_ = 0;
    std::uint32_t cached_tokens_ = 0;
    bool started_                = false;
    bool admitted_               = false;
    bool finished_               = false;
};

} // namespace ninfer::serve
