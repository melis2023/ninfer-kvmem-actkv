#pragma once
#include "serve/request.h"
#include "serve/request_json.h"

#include <optional>

namespace ninfer::serve {
// Chat nests schema metadata under json_schema; Responses and Anthropic use a flat format.
StructuredOutputOptions parse_structured_output(const RequestJson& format, bool nested,
                                                const std::string& param);

// The constrained-decoding fields vLLM and llama.cpp add to OpenAI's request: `structured_outputs`
// (vLLM), the earlier `guided_json`, `guided_regex`, `guided_choice` and `guided_grammar` (vLLM),
// and `grammar`, a GBNF grammar (llama.cpp; empty means none). At most one of them may constrain a
// request, and not beside a response format already in `request`. Sets the request's
// structured_output and the field that carried it.
void parse_constrained_decoding_extensions(const RequestJson& body, GenerationRequest& request);

// --structured-output gates final response formats. Without it, those formats are refused unless
// --unconstrained-response-format explicitly allows dropping them. Native tool constraints remain
// available and strict tool intent is retained. Returns the changed request, or nothing.
std::optional<GenerationRequest> drop_unavailable_constraints(const GenerationRequest& request,
                                                              bool structured_output,
                                                              bool unconstrained_response_format);
} // namespace ninfer::serve
