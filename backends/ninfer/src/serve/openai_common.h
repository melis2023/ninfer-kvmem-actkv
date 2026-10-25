#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

// Recognized image detail levels all use the fixed, startup-configured Vision preprocessing.
// Validate the wire value before lowering it to that same effective "auto" behavior.
void validate_openai_image_detail(const RequestJson& image, std::string_view field,
                                  std::string_view param);

// True for OpenAI tool types the *server* would have executed - hosted search, hosted code
// execution, hosted file search and the like. NInfer has no executor for any of them, and a client
// never waits on one itself, so declaring one is silently dropped rather than failing the request:
// the model is simply never told the tool exists, which is the same outcome as not declaring it.
//
// Client-executed types stay rejected. Dropping one of those would leave the caller waiting for a
// call that can never arrive, which is worse than a clear error.
[[nodiscard]] bool is_hosted_openai_tool_type(std::string_view type) noexcept;

// Generated tokens' log probability records in the OpenAI shape: each token's text (its bytes,
// with U+FFFD for a sequence that is not valid UTF-8 on its own), logprob (OpenAI's -9999 for a
// token outside its top set), and up to `top_logprobs` alternatives; with `bytes`, every entry
// also carries its exact bytes.
[[nodiscard]] nlohmann::json openai_token_logprobs_json(std::span<const ninfer::TokenLogprob> records,
                                                        int top_logprobs, bool bytes);

// What /v1/models advertises about the one resident model.
struct ModelDescription {
    std::string id;
    std::uint32_t max_model_len = 0; // --max-context, each sequence's ceiling
    bool vision                 = false;
    ninfer::ModelMetadata metadata;
    // llama.cpp router status: loaded, sleeping, loading, unloading or unloaded. A model that is
    // not loaded has no metadata to report; its object carries the id, path and status only.
    std::string status = "loaded";
    bool loaded_facts  = true;
    std::string path;
    std::vector<std::string> args;
    std::vector<std::string> aliases;
    bool failed = false;
    std::string last_error;
};

std::string make_models_list(const ModelDescription& model, std::int64_t created);
std::string make_models_list(const std::vector<ModelDescription>& models, std::int64_t created);
std::string make_model_object(const ModelDescription& model, std::int64_t created);
// The /v1 discovery document: startup announces the API base as a URL, so the bare base answers
// with the configured model alias and the endpoints this build serves instead of a 404.
nlohmann::json make_api_index(const std::string& model_id);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

void validate_openai_model(std::string_view requested, std::string_view available);

std::string new_openai_chat_completion_id();
std::string new_openai_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
