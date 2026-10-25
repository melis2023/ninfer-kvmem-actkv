#pragma once

// OpenAI Chat Completions wire adapter. Parsing produces one protocol envelope plus an executable
// GenerationRequest; response builders consume protocol-neutral GenerationOutcome values.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

struct OpenAIChatRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool include_usage          = false;
    bool output_tokens_explicit = false;
    // llama.cpp-compatible response observations. Final timings remain unconditional;
    // timings_per_token controls cumulative timing snapshots on streamed output chunks and,
    // together with return_progress, on prompt-progress chunks.
    bool timings_per_token = false;
    bool return_progress   = false;
};

OpenAIChatRequest parse_chat_completion_request(const RequestJson& body,
                                                const RequestLimits& limits);

struct OpenAIChatResponseIdentity {
    std::string id;
    std::string model;
    std::int64_t created = 0;
};

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model);
// `top_logprobs` is absent unless the request asked for logprobs, and is then the number of
// alternatives each token reports.
std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome,
                                          std::optional<int> top_logprobs = std::nullopt);

// Pieces the text-completion endpoints share with Chat Completions: llama.cpp's `timings` object of
// a finished generation and the live snapshot of one under way, its `prompt_progress` object, and
// the OpenAI `usage` object and finish_reason of a finished generation.
nlohmann::json completion_timings_json(const GenerationOutcome& outcome);
nlohmann::json completion_timings_json(std::uint32_t prompt_tokens, std::uint32_t cached_tokens,
                                       const ninfer::GenerationTimingObservation& observation);
nlohmann::json completion_prompt_progress_json(const ninfer::PromptProgress& progress);
nlohmann::json openai_usage_json(const GenerationOutcome& outcome);
const char* openai_finish_reason(ninfer::FinishReason reason) noexcept;

class OpenAIChatStream {
public:
    OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                     bool timings_per_token = false, bool return_progress = false,
                     bool usage_chunk_choice = false,
                     std::optional<int> top_logprobs = std::nullopt);

    std::string start();
    void note_start(const ninfer::GenerationStart& start);
    std::string initial_prompt_progress();
    std::string prompt_progress(const ninfer::PromptProgress& progress);
    void note_timing(const ninfer::GenerationTimingObservation& timing);
    std::string reasoning_delta(const std::string& text);
    // A content chunk carries the logprob records of the tokens whose text it publishes.
    std::string content_delta(const std::string& text,
                              std::span<const ninfer::TokenLogprob> logprobs = {});
    std::vector<std::string> finish(const GenerationOutcome& outcome);

private:
    nlohmann::json live_timings_json() const;

    OpenAIChatResponseIdentity identity_;
    std::optional<int> top_logprobs_;
    std::string reasoning_;
    std::string content_;
    std::optional<ninfer::GenerationTimingObservation> live_timing_;
    std::uint32_t prompt_tokens_            = 0;
    std::uint32_t cached_tokens_            = 0;
    std::uint32_t last_progress_tokens_     = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool include_usage_                     = false;
    bool timings_per_token_                 = false;
    bool return_progress_                   = false;
    bool usage_chunk_choice_                = false;
    bool started_                           = false;
    bool admitted_                          = false;
    bool progress_started_                  = false;
    bool content_started_                   = false;
    bool finished_                          = false;
};

} // namespace ninfer::serve
