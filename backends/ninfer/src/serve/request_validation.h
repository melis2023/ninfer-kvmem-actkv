#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

// The NInfer `graft` extension field: a graft name, "" for explicitly none, or absent/null (nullopt)
// to take the server's default graft. Whether the name is loaded is checked against the server's
// grafts during translation.
std::optional<std::string> parse_graft_field(const RequestJson& body);

// The NInfer `thinking_budget` extension field: a positive cap on model-origin thinking tokens for
// this request, or absent/null (nullopt) to take the server default. Ignored when the request does
// not think. Anthropic Messages carries the same cap as `thinking.budget_tokens`.
std::optional<std::uint32_t> parse_thinking_budget_field(const RequestJson& body);

void parse_tool_constraints(const RequestJson& body, GenerationRequest& request);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// An optional `post_thinking` object: temperature (up to `max_temperature`), top_p, top_k, min_p,
// presence_penalty, frequency_penalty and seed, applied from the token after the reasoning block
// closes. Absent or null is nullopt; an empty object selects the post-thinking preset.
[[nodiscard]] std::optional<SamplingParams> parse_post_thinking(const RequestJson& body,
                                                                double max_temperature);

// Name of a JSON value's type for error messages ("string", "array", ...).
[[nodiscard]] const char* request_json_type_name(const RequestJson& value) noexcept;

// ASCII-safe, bounded preview of a client-supplied value for error messages. Every byte outside
// printable ASCII is escaped as \xNN, so the result can be embedded in the JSON error body and in
// logs without control characters or invalid UTF-8.
[[nodiscard]] std::string ascii_preview(std::string_view value, std::size_t limit = 64);

} // namespace ninfer::serve
