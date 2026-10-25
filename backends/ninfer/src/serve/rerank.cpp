#include "serve/rerank.h"

#include "serve/generation_service.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using Json    = RequestJson;
using OutJson = nlohmann::json;

// Qwen3-Reranker's judging prompt and default instruction.
constexpr const char* kJudge = "Judge whether the Document meets the requirements based on the "
                               "Query and the Instruct provided. Note that the answer can only be "
                               "\"yes\" or \"no\".";
constexpr const char* kDefaultInstruction =
    "Given a web search query, retrieve relevant passages that answer the query";

bool present(const Json& body, const char* key) {
    return body.contains(key) && !body.at(key).is_null();
}

std::string document_text(const Json& document, const char* param) {
    if (document.is_string()) { return document.get<std::string>(); }
    if (document.is_object() && document.contains("text") && document.at("text").is_string()) {
        return document.at("text").get<std::string>();
    }
    bad_request(std::string(param) + " entries must be strings or objects with a string text",
                param);
}

ChatTurn text_turn(ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    turn.content.push_back(ContentPart{.kind = ContentKind::Text, .text = std::move(text)});
    return turn;
}

} // namespace

RerankRequest parse_rerank_request(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    RerankRequest out;
    if (present(body, "model")) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        out.model = body.at("model").get<std::string>();
    }
    if (!present(body, "query") || !body.at("query").is_string() ||
        body.at("query").get_ref<const std::string&>().empty()) {
        bad_request("query must be a non-empty string", "query");
    }
    out.query         = body.at("query").get<std::string>();
    out.tei           = present(body, "texts");
    const char* param = out.tei ? "texts" : "documents";
    if (!present(body, param) || !body.at(param).is_array() || body.at(param).empty()) {
        bad_request(std::string(param) + " must be a non-empty array", param);
    }
    const Json& documents = body.at(param);
    if (documents.size() > kMaximumRerankDocuments) {
        bad_request(std::string(param) + " holds more than " +
                        std::to_string(kMaximumRerankDocuments) + " documents",
                    param);
    }
    out.documents.reserve(documents.size());
    for (const Json& document : documents) {
        out.documents.push_back(document_text(document, param));
    }
    out.top_n = out.documents.size();
    if (const std::optional<int> top_n = optional_int(body, "top_n")) {
        if (*top_n < 1) { bad_request("top_n must be at least 1", "top_n"); }
        out.top_n = std::min(out.documents.size(), static_cast<std::size_t>(*top_n));
    }
    out.return_documents = out.tei ? optional_bool(body, "return_text", false)
                                   : optional_bool(body, "return_documents", true);
    out.instruction      = kDefaultInstruction;
    if (present(body, "instruction")) {
        if (!body.at("instruction").is_string()) {
            bad_request("instruction must be a string", "instruction");
        }
        out.instruction = body.at("instruction").get<std::string>();
    }
    return out;
}

GenerationRequest make_rerank_judgement(const RerankRequest& request, const std::string& document) {
    GenerationRequest out;
    out.messages.push_back(text_turn(ChatRole::System, kJudge));
    out.messages.push_back(text_turn(ChatRole::User, "<Instruct>: " + request.instruction +
                                                         "\n<Query>: " + request.query +
                                                         "\n<Document>: " + document));
    out.enable_thinking      = false;
    out.max_tokens           = 1;
    out.sampling.temperature = 0.0;
    out.logprobs             = true;
    // The server's default graft would be a hidden prefix the judging prompt does not expect.
    out.graft = std::string();
    return out;
}

double rerank_score(const GenerationOutcome& outcome) {
    if (outcome.content_logprobs.empty() || outcome.content_logprobs.front().top_ids[0] < 0) {
        throw std::runtime_error("the rerank judgement produced no answer token");
    }
    // The answer token's most likely alternatives, under the greedy judge's raw distribution.
    const ninfer::TokenLogprob& answer = outcome.content_logprobs.front();
    double yes                         = 0.0;
    double no                          = 0.0;
    double least                       = 0.0;
    bool has_yes                       = false;
    bool has_no                        = false;
    for (std::size_t k = 0; k < ninfer::kMaximumTokenLogprobs && answer.top_ids[k] >= 0; ++k) {
        const std::string& bytes = answer.top_bytes[k];
        const double probability = std::exp(static_cast<double>(answer.top_values[k]));
        least                    = probability;
        if (bytes == "yes" || bytes == "Yes") {
            yes += probability;
            has_yes = true;
        } else if (bytes == "no" || bytes == "No") {
            no += probability;
            has_no = true;
        }
    }
    if (!has_yes) { yes = least; }
    if (!has_no) { no = least; }
    return yes + no > 0.0 ? yes / (yes + no) : 0.5;
}

std::string make_rerank_response(const RerankRequest& request, std::vector<RerankScore> scores,
                                 int prompt_tokens) {
    std::stable_sort(scores.begin(), scores.end(), [](const RerankScore& a, const RerankScore& b) {
        return a.score != b.score ? a.score > b.score : a.index < b.index;
    });
    scores.resize(std::min(scores.size(), request.top_n));
    OutJson results = OutJson::array();
    for (const RerankScore& entry : scores) {
        OutJson result                                    = {{"index", entry.index}};
        result[request.tei ? "score" : "relevance_score"] = entry.score;
        if (request.return_documents) {
            const std::string& text = request.documents.at(entry.index);
            result[request.tei ? "text" : "document"] =
                request.tei ? OutJson(text) : OutJson{{"text", text}};
        }
        results.push_back(std::move(result));
    }
    const auto dump = [](const OutJson& value) {
        return value.dump(-1, ' ', false, OutJson::error_handler_t::replace);
    };
    if (request.tei) { return dump(results); }
    return dump(
        OutJson{{"model", request.model},
                {"object", "list"},
                {"usage", {{"prompt_tokens", prompt_tokens}, {"total_tokens", prompt_tokens}}},
                {"results", std::move(results)}});
}

} // namespace ninfer::serve
