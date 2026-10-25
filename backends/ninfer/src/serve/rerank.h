#pragma once

// Document reranking (POST /rerank, /v1/rerank and their /reranking spellings): Jina's request and
// response, and llama.cpp's TEI variant, scored by the served chat model as a relevance judge. Each
// document is one prompt asking whether it meets the query -- Qwen3-Reranker's formulation -- and
// its score is P(yes) / (P(yes) + P(no)) over the first answer token's log probabilities.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

// Cohere's bound. Each document is one prompt the model reads.
inline constexpr std::size_t kMaximumRerankDocuments = 1000;

struct RerankRequest {
    std::string model;
    std::string query;
    std::vector<std::string> documents;
    // What relevance means (Qwen3-Reranker's "Instruct"); a web-search default when omitted.
    std::string instruction;
    // Results kept, best first; every document by default.
    std::size_t top_n     = 0;
    bool return_documents = true;
    // TEI's shape (llama.cpp's /rerank with `texts`): a bare array of {index, score[, text]}.
    bool tei = false;
};

[[nodiscard]] RerankRequest parse_rerank_request(const RequestJson& body);

// The request that judges `document`: the judging instruction as the system turn, the instruction,
// query and document as the user turn, thinking off, one output token with its alternatives.
[[nodiscard]] GenerationRequest make_rerank_judgement(const RerankRequest& request,
                                                      const std::string& document);

// P(yes) / (P(yes) + P(no)) over the first token's alternatives, "Yes" and "No" counted with them.
// An answer missing from the alternatives counts as likely as the least likely of them, its upper
// bound, so a judgement that answers neither scores 0.5.
[[nodiscard]] double rerank_score(const GenerationOutcome& outcome);

struct RerankScore {
    std::size_t index = 0;
    double score      = 0.0;
};

// Best first, the lower index first among equal scores, `top_n` kept.
[[nodiscard]] std::string make_rerank_response(const RerankRequest& request,
                                               std::vector<RerankScore> scores, int prompt_tokens);

} // namespace ninfer::serve
