#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <string_view>
#include <utility>

namespace ninfer::serve {

namespace {

std::string chat_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + buffer.data();
}

// A token's bytes as JSON text: an invalid or truncated UTF-8 sequence becomes U+FFFD, since a
// token can end inside a multi-byte character. The exact bytes travel beside it.
std::string lossy_utf8(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto lead       = static_cast<unsigned char>(bytes[index]);
        std::size_t length    = 0;
        std::uint32_t minimum = 0;
        if (lead < 0x80) {
            length = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length  = 2;
            minimum = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length  = 3;
            minimum = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length  = 4;
            minimum = 0x10000;
        }
        bool valid              = length != 0 && index + length <= bytes.size();
        std::uint32_t codepoint = length == 1 ? lead : lead & (0x7FU >> length);
        for (std::size_t offset = 1; valid && offset < length; ++offset) {
            const auto next = static_cast<unsigned char>(bytes[index + offset]);
            valid           = (next & 0xC0U) == 0x80U;
            codepoint       = (codepoint << 6U) | (next & 0x3FU);
        }
        valid = valid && codepoint >= minimum && codepoint <= 0x10FFFF &&
                (codepoint < 0xD800 || codepoint > 0xDFFF);
        if (valid) {
            out.append(bytes.substr(index, length));
            index += length;
        } else {
            out.append("\xEF\xBF\xBD");
            ++index;
        }
    }
    return out;
}

std::string responses_identifier(std::string_view prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx%016llx",
                  static_cast<unsigned long long>(distribution(random)),
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + "_" + buffer.data();
}

} // namespace

using Json = nlohmann::json;

Json openai_token_logprobs_json(std::span<const ninfer::TokenLogprob> records, int top_logprobs,
                                bool bytes) {
    const auto entry = [bytes](std::string_view token, float logprob) {
        Json out = {{"token", lossy_utf8(token)}, {"logprob", logprob}};
        if (bytes) {
            Json array = Json::array();
            for (const char byte : token) { array.push_back(static_cast<unsigned char>(byte)); }
            out["bytes"] = std::move(array);
        }
        return out;
    };
    Json out = Json::array();
    for (const ninfer::TokenLogprob& record : records) {
        Json token    = entry(record.bytes, record.logprob);
        Json top      = Json::array();
        const auto n  = std::min<std::size_t>(static_cast<std::size_t>(std::max(top_logprobs, 0)),
                                              ninfer::kMaximumTokenLogprobs);
        for (std::size_t k = 0; k < n && record.top_ids[k] >= 0; ++k) {
            top.push_back(entry(record.top_bytes[k], record.top_values[k]));
        }
        token["top_logprobs"] = std::move(top);
        out.push_back(std::move(token));
    }
    return out;
}

bool is_hosted_openai_tool_type(std::string_view type) noexcept {
    // OpenAI versions some of these with a dated suffix (web_search_preview_2025_03_11), so match
    // the family by prefix rather than pinning every spelling.
    static constexpr std::string_view kHostedPrefixes[] = {
        "web_search", "code_interpreter", "file_search",
        "image_generation", "computer_use", "mcp",
    };
    for (const std::string_view prefix : kHostedPrefixes) {
        if (type.size() >= prefix.size() && type.substr(0, prefix.size()) == prefix) {
            return true;
        }
    }
    return false;
}

bool parse_openai_prompt_cache_breakpoint(const RequestJson& value, std::string_view param) {
    if (!value.contains("prompt_cache_breakpoint") ||
        value.at("prompt_cache_breakpoint").is_null()) {
        return false;
    }
    const RequestJson& breakpoint = value.at("prompt_cache_breakpoint");
    if (!breakpoint.is_object() || !breakpoint.contains("mode") ||
        !breakpoint.at("mode").is_string() ||
        breakpoint.at("mode").get<std::string>() != "explicit") {
        bad_request("prompt_cache_breakpoint must be {mode:'explicit'}", std::string(param),
                    "invalid_cache_breakpoint");
    }
    return true;
}

void validate_openai_image_detail(const RequestJson& image, std::string_view field,
                                  std::string_view param) {
    if (!image.contains("detail") || image.at("detail").is_null()) { return; }
    if (!image.at("detail").is_string()) {
        bad_request(std::string(field) + " must be a string", std::string(param));
    }
    const std::string& detail = image.at("detail").get_ref<const std::string&>();
    if (detail != "auto" && detail != "low" && detail != "high" && detail != "original") {
        bad_request(std::string(field) + " must be 'auto', 'low', 'high', or 'original'",
                    std::string(param), "invalid_value");
    }
}

OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body) {
    auto require_string_hint = [&](const char* field, std::optional<std::size_t> maximum = {}) {
        if (!body.contains(field) || body.at(field).is_null()) { return; }
        if (!body.at(field).is_string()) {
            bad_request(std::string(field) + " must be a string", field);
        }
        if (maximum && body.at(field).get_ref<const std::string&>().size() > *maximum) {
            bad_request(std::string(field) + " must be at most " + std::to_string(*maximum) +
                            " characters",
                        field);
        }
    };
    require_string_hint("prompt_cache_key", 64U);
    require_string_hint("safety_identifier", 64U);
    require_string_hint("user");

    if (body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null()) {
        if (!body.at("prompt_cache_retention").is_string()) {
            bad_request("prompt_cache_retention must be a string", "prompt_cache_retention");
        }
        const std::string value = body.at("prompt_cache_retention").get<std::string>();
        if (value != "in_memory" && value != "24h") {
            bad_request("prompt_cache_retention must be 'in_memory' or '24h'",
                        "prompt_cache_retention");
        }
    }

    OpenAIPromptCachePolicy policy;
    if (!body.contains("prompt_cache_options") || body.at("prompt_cache_options").is_null()) {
        return policy;
    }
    const RequestJson& options = body.at("prompt_cache_options");
    if (!options.is_object()) {
        bad_request("prompt_cache_options must be an object", "prompt_cache_options");
    }
    policy.automatic = OpenAIPromptCacheAutomatic::Requested;
    if (options.contains("mode") && !options.at("mode").is_null()) {
        if (!options.at("mode").is_string()) {
            bad_request("prompt_cache_options.mode must be a string", "prompt_cache_options");
        }
        const std::string mode = options.at("mode").get<std::string>();
        if (mode == "explicit") {
            policy.automatic = OpenAIPromptCacheAutomatic::Disabled;
        } else if (mode != "implicit") {
            bad_request("prompt_cache_options.mode must be 'implicit' or 'explicit'",
                        "prompt_cache_options");
        }
    }
    if (options.contains("ttl") && !options.at("ttl").is_null() &&
        (!options.at("ttl").is_string() || options.at("ttl").get<std::string>() != "30m")) {
        bad_request("prompt_cache_options.ttl must be '30m'", "prompt_cache_options");
    }
    return policy;
}

void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy) {
    std::vector<std::optional<CacheBoundary>*> explicit_boundaries;
    for (ToolDefinition& tool : request.tools) {
        if (tool.cache_boundary_after) {
            explicit_boundaries.push_back(&tool.cache_boundary_after);
        }
    }
    for (ChatTurn& turn : request.messages) {
        for (ContentPart& part : turn.content) {
            if (part.cache_boundary_after) {
                explicit_boundaries.push_back(&part.cache_boundary_after);
            }
        }
        if (turn.cache_boundary_after) {
            explicit_boundaries.push_back(&turn.cache_boundary_after);
        }
    }

    std::optional<CacheBoundary>* automatic_target = nullptr;
    for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
        if (!turn->tool_calls.empty()) {
            automatic_target = &turn->cache_boundary_after;
            break;
        }
        if (!turn->content.empty()) {
            automatic_target = &turn->content.back().cache_boundary_after;
            break;
        }
    }
    if (automatic_target == nullptr && !request.tools.empty()) {
        automatic_target = &request.tools.back().cache_boundary_after;
    }

    const bool automatic_enabled =
        policy.automatic != OpenAIPromptCacheAutomatic::Disabled && automatic_target != nullptr;
    const bool automatic_merges_explicit   = automatic_enabled && automatic_target->has_value();
    const std::size_t explicit_write_slots = !automatic_enabled || automatic_merges_explicit
                                                 ? kMaximumExplicitPromptCacheMarkers
                                                 : kMaximumExplicitPromptCacheMarkers - 1U;
    const std::size_t first_selected       = explicit_boundaries.size() > explicit_write_slots
                                                 ? explicit_boundaries.size() - explicit_write_slots
                                                 : 0U;
    for (std::size_t index = 0; index < explicit_boundaries.size(); ++index) {
        if (index < first_selected) {
            explicit_boundaries[index]->reset();
        } else {
            explicit_boundaries[index]->value().kind =
                ninfer::PromptCacheMarkerKind::SharedStablePrefix;
            explicit_boundaries[index]->value().evidence =
                ninfer::SharedCandidateEvidence::ExplicitBoundary;
        }
    }

    if (automatic_enabled) {
        const ninfer::SharedCandidateEvidence evidence =
            policy.automatic == OpenAIPromptCacheAutomatic::Default
                ? ninfer::SharedCandidateEvidence::DefaultAutomatic
                : ninfer::SharedCandidateEvidence::RequestedAutomatic;
        if (*automatic_target) {
            automatic_target->value().evidence |= evidence;
        } else {
            *automatic_target = CacheBoundary{.evidence = evidence};
        }
    }
    // OpenAI's policy governs one boundary: the automatic marker this function just placed at the
    // end of the prompt. It says nothing about where a prompt's structure already exposes a
    // reusable prefix, so the Engine's structural boundaries stay enabled. Disabling them left a
    // request whose only shared candidate sat at the end of its own prompt, which no differing
    // request can ever match: ten identical-preamble requests each recomputed their whole prompt.
    request.allow_engine_automatic_shared_prefixes = true;
}

namespace {

// Discovery metadata for the one resident model. No client ecosystem agrees on one name for the
// context limit: max_model_len is vLLM/llama.cpp's discovery field, context_window is Anthropic's
// Models API field, and context_length is the OpenRouter/Ollama convention. OpenAI's own /v1/models
// spec has none of them. The same per-request ceiling is mirrored under all three.
//
// Modalities are reported twice: llama.cpp's `modalities.vision` flag, and the OpenRouter
// `architecture` object that llama.cpp's router-mode /models also emits, so clients that gate image
// attachments on either see image and video input only when the server was started with --vision.
// llama.cpp's `meta` carries facts about the registered artifact behind the alias: n_ctx is the
// per-request context ceiling, n_ctx_train the model's native context, ftype the registered
// weights profile.
Json model_status_json(const ModelDescription& model) {
    Json status = {{"value", model.status}};
    if (!model.args.empty()) { status["args"] = model.args; }
    if (model.failed) {
        status["failed"] = true;
        if (!model.last_error.empty()) { status["error"] = model.last_error; }
    }
    return status;
}

Json model_json(const ModelDescription& model, std::int64_t created) {
    if (!model.loaded_facts) {
        Json out = {{"id", model.id},
                    {"object", "model"},
                    {"created", created},
                    {"owned_by", "ninfer"},
                    {"status", model_status_json(model)}};
        if (!model.path.empty()) { out["path"] = model.path; }
        if (!model.aliases.empty()) { out["aliases"] = model.aliases; }
        return out;
    }
    Json input = Json::array({"text"});
    if (model.vision) {
        input.push_back("image");
        input.push_back("video");
    }
    const ninfer::ModelMetadata& metadata = model.metadata;
    return Json{{"id", model.id},
                {"object", "model"},
                {"created", created},
                {"owned_by", "ninfer"},
                {"max_model_len", model.max_model_len},
                {"context_window", model.max_model_len},
                {"context_length", model.max_model_len},
                {"modalities", Json{{"vision", model.vision}}},
                {"architecture", Json{{"input_modalities", std::move(input)},
                                      {"output_modalities", Json::array({"text"})}}},
                {"status", model_status_json(model)},
                {"meta", Json{{"n_vocab", metadata.vocab_size},
                              {"n_ctx", model.max_model_len},
                              {"n_ctx_train", metadata.native_context},
                              {"n_embd", metadata.embedding_size},
                              {"n_params", metadata.parameters},
                              {"size", metadata.weight_bytes},
                              {"ftype", metadata.weights_id}}}};
}

} // namespace

std::string make_models_list(const ModelDescription& model, std::int64_t created) {
    return Json{{"object", "list"}, {"data", Json::array({model_json(model, created)})}}.dump();
}

std::string make_models_list(const std::vector<ModelDescription>& models, std::int64_t created) {
    Json data = Json::array();
    for (const ModelDescription& model : models) { data.push_back(model_json(model, created)); }
    return Json{{"object", "list"}, {"data", std::move(data)}}.dump();
}

Json make_api_index(const std::string& model_id) {
    const auto endpoint = [](const char* method, const char* path, const char* description) {
        return Json{{"method", method}, {"path", path}, {"description", description}};
    };
    return Json{
        {"object", "api_base"},
        {"service", "ninfer-serve"},
        {"model", model_id},
        {"endpoints",
         Json::array(
             {endpoint("GET", "/health", "process health"),
              endpoint("GET", "/v1/load", "serving capacity, current load, and token counters"),
              endpoint("GET", "/metrics", "Prometheus text metrics"),
              endpoint("GET", "/slots", "llama.cpp-shaped listing of the context-cache cells"),
              endpoint("POST", "/slots/{id}?action=save|restore|erase",
                       "retained-session persistence (with --slot-save-path)"),
              endpoint("GET", "/props", "llama.cpp-shaped server properties"),
              endpoint("GET", "/v1/models", "configured OpenAI model alias"),
              endpoint("GET", "/v1/models/{id}", "lookup of the configured alias"),
              endpoint("POST", "/v1/chat/completions", "OpenAI-style chat generation"),
              endpoint("POST", "/v1/completions", "OpenAI legacy raw-prompt completion"),
              endpoint("POST", "/completion", "llama.cpp raw-prompt completion"),
              endpoint("POST", "/tokenize", "llama.cpp tokenization with the model's tokenizer"),
              endpoint("POST", "/detokenize", "llama.cpp token ids back to text"),
              endpoint("POST", "/apply-template", "llama.cpp chat template rendering"),
              endpoint("POST", "/v1/rerank", "documents ranked by relevance to a query"),
              endpoint("GET", "/models", "llama.cpp router: every configured model and its state"),
              endpoint("POST", "/models/load|unload|sleep", "llama.cpp router model lifecycle"),
              endpoint("GET", "/models/sse", "llama.cpp router model state events"),
              endpoint("POST", "/v1/responses", "OpenAI Responses generation, state, and SSE"),
              endpoint("POST", "/v1/responses/input_tokens",
                       "Responses prompt-token count without generation"),
              endpoint("POST", "/v1/responses/compact", "Responses conversation compaction"),
              endpoint("POST", "/v1/responses/{id}/cancel", "cancel a background Response"),
              endpoint("GET", "/v1/responses/{id}", "retrieve a locally stored terminal Response"),
              endpoint("DELETE", "/v1/responses/{id}", "delete a locally stored Response"),
              endpoint("GET", "/v1/responses/{id}/input_items",
                       "list that Response's normalized input Items"),
              endpoint("POST", "/v1/messages", "Anthropic-style message generation"),
              endpoint("POST", "/v1/messages/count_tokens",
                       "checkpoint-native expanded input-token count")})}};
}

std::string make_model_object(const ModelDescription& model, std::int64_t created) {
    return model_json(model, created).dump();
}

std::string make_error_body(const ApiError& error) {
    Json rendered     = {{"message", error.message}, {"type", error.type}};
    rendered["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    rendered["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", std::move(rendered)}}.dump();
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void validate_openai_model(std::string_view requested, std::string_view available) {
    if (requested == available) { return; }
    ApiError error;
    error.status  = 404;
    error.type    = "invalid_request_error";
    error.param   = "model";
    error.code    = "model_not_found";
    error.message = "model '" + std::string(requested) + "' not found";
    throw ApiException(std::move(error));
}

std::string new_openai_chat_completion_id() { return chat_identifier("chatcmpl-"); }

std::string new_openai_completion_id() { return chat_identifier("cmpl-"); }

std::string new_openai_chat_tool_call_id() { return chat_identifier("call_"); }

std::string new_openai_request_id() { return responses_identifier("req"); }

std::string new_openai_response_id() { return responses_identifier("resp"); }

std::string new_openai_response_item_id(std::string_view prefix) {
    return responses_identifier(prefix);
}

} // namespace ninfer::serve
