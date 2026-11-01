#include "serve/structured_output.h"
#include "text/structured_output.h"

#include <algorithm>
#include <optional>
#include <string>

namespace ninfer::serve {
StructuredOutputOptions parse_structured_output(const RequestJson& format, bool nested,
                                                const std::string& param) {
    const auto fail = [&](const std::string& message) -> void {
        throw ApiException(
            ApiError{.message = message, .param = param, .code = "invalid_response_format"});
    };
    if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
        fail(param + " must contain a string type");
    }
    const auto type = format.at("type").get<std::string>();
    StructuredOutputOptions options;
    if (type == "text" || type == "json_object") {
        if (format.size() != 1) { fail("unexpected members in " + param); }
        options.kind =
            type == "text" ? StructuredOutputKind::None : StructuredOutputKind::JsonObject;
    } else if (type == "json_schema") {
        const auto& spec =
            nested && format.contains("json_schema") ? format.at("json_schema") : format;
        if (nested && (format.size() != 2 || !format.contains("json_schema"))) {
            fail("json_schema requires a json_schema object");
        }
        if (!spec.is_object() || !spec.contains("schema") ||
            (!spec.at("schema").is_object() && !spec.at("schema").is_boolean())) {
            fail("json_schema requires an object or boolean schema");
        }
        for (const auto& [key, value] : spec.items()) {
            if (key == "schema" || (!nested && key == "type")) { continue; }
            if ((key == "name" || key == "description") && value.is_string()) { continue; }
            if (key == "strict" && (value.is_boolean() || value.is_null())) { continue; }
            fail("unsupported or invalid schema format member: " + key);
        }
        options.kind   = StructuredOutputKind::JsonSchema;
        options.schema = spec.at("schema").dump();
        // OpenAI strict mode: an object schema without additionalProperties admits only its
        // declared properties. Without it, standard JSON Schema defaults apply.
        options.strict = spec.contains("strict") && spec.at("strict").is_boolean() &&
                         spec.at("strict").get<bool>();
    } else {
        fail("unsupported response format type: " + type);
    }
    try {
        text::validate_structured_output(options);
    } catch (const std::invalid_argument& e) { fail(e.what()); }
    return options;
}

namespace {

[[noreturn]] void invalid_constraint(const std::string& message, const std::string& param) {
    throw ApiException(
        ApiError{.message = message, .param = param, .code = "invalid_response_format"});
}

std::string string_value(const RequestJson& value, const std::string& param) {
    if (!value.is_string()) { invalid_constraint(param + " must be a string", param); }
    return value.get<std::string>();
}

// A JSON Schema given as an object, a boolean, or that JSON as a string (vLLM accepts each).
StructuredOutputOptions json_schema_constraint(const RequestJson& value, const std::string& param) {
    RequestJson schema = value;
    if (value.is_string()) {
        schema = RequestJson::parse(value.get_ref<const std::string&>(), nullptr, false);
        if (schema.is_discarded()) { invalid_constraint(param + " is not valid JSON", param); }
    }
    if (!schema.is_object() && !schema.is_boolean()) {
        invalid_constraint(param + " must be a JSON Schema object", param);
    }
    StructuredOutputOptions options;
    options.kind   = StructuredOutputKind::JsonSchema;
    options.schema = schema.dump();
    return options;
}

StructuredOutputOptions text_constraint(StructuredOutputKind kind, const RequestJson& value,
                                        const std::string& param) {
    StructuredOutputOptions options;
    options.kind = kind;
    if (kind == StructuredOutputKind::Choice) {
        if (!value.is_array()) {
            invalid_constraint(param + " must be an array of strings", param);
        }
        for (const auto& choice : value) { options.choices.push_back(string_value(choice, param)); }
    } else {
        options.grammar = string_value(value, param);
    }
    return options;
}

// vLLM's StructuredOutputsParams: one of json, regex, choice, grammar or json_object, and the
// options that keep their defaults here.
StructuredOutputOptions structured_outputs_constraint(const RequestJson& value) {
    const std::string param = "structured_outputs";
    if (!value.is_object()) { invalid_constraint(param + " must be an object", param); }
    std::optional<StructuredOutputOptions> options;
    bool closed_objects = false;
    const auto take     = [&](StructuredOutputOptions next, const std::string& key) {
        if (options) {
            invalid_constraint(param + " names more than one constraint", param + "." + key);
        }
        options = std::move(next);
    };
    for (const auto& [key, member] : value.items()) {
        const std::string field = param + "." + key;
        if (member.is_null()) { continue; }
        if (key == "json") {
            take(json_schema_constraint(member, field), key);
        } else if (key == "regex") {
            take(text_constraint(StructuredOutputKind::Regex, member, field), key);
        } else if (key == "choice") {
            take(text_constraint(StructuredOutputKind::Choice, member, field), key);
        } else if (key == "grammar") {
            take(text_constraint(StructuredOutputKind::Grammar, member, field), key);
        } else if (key == "json_object") {
            if (!member.is_boolean()) { invalid_constraint(field + " must be a boolean", field); }
            if (member.get<bool>()) {
                StructuredOutputOptions object;
                object.kind = StructuredOutputKind::JsonObject;
                take(std::move(object), key);
            }
        } else if (key == "disable_additional_properties") {
            if (!member.is_boolean()) { invalid_constraint(field + " must be a boolean", field); }
            closed_objects = member.get<bool>();
        } else if (key == "disable_fallback" || key == "disable_any_whitespace") {
            // NInfer has no fallback backend, and it bounds whitespace runs in any case.
            if (!member.is_boolean()) { invalid_constraint(field + " must be a boolean", field); }
        } else {
            invalid_constraint("unsupported structured_outputs member: " + key, field);
        }
    }
    if (!options) { invalid_constraint(param + " names no constraint", param); }
    if (closed_objects) {
        if (options->kind != StructuredOutputKind::JsonSchema) {
            invalid_constraint("disable_additional_properties applies to a JSON schema",
                               param + ".disable_additional_properties");
        }
        options->strict = true;
    }
    return *options;
}

} // namespace

void parse_constrained_decoding_extensions(const RequestJson& body, GenerationRequest& request) {
    std::optional<StructuredOutputOptions> found;
    std::string found_param;
    const auto take = [&](const std::string& param, StructuredOutputOptions options) {
        if (found) { invalid_constraint(param + " and " + found_param + " are exclusive", param); }
        found       = std::move(options);
        found_param = param;
    };
    const auto present = [&](const char* key) {
        return body.contains(key) && !body.at(key).is_null();
    };
    if (present("structured_outputs")) {
        take("structured_outputs", structured_outputs_constraint(body.at("structured_outputs")));
    }
    if (present("guided_json")) {
        take("guided_json", json_schema_constraint(body.at("guided_json"), "guided_json"));
    }
    if (present("guided_regex")) {
        take("guided_regex",
             text_constraint(StructuredOutputKind::Regex, body.at("guided_regex"), "guided_regex"));
    }
    if (present("guided_choice")) {
        take("guided_choice", text_constraint(StructuredOutputKind::Choice,
                                              body.at("guided_choice"), "guided_choice"));
    }
    if (present("guided_grammar")) {
        take("guided_grammar", text_constraint(StructuredOutputKind::Grammar,
                                               body.at("guided_grammar"), "guided_grammar"));
    }
    if (present("grammar")) {
        auto grammar =
            text_constraint(StructuredOutputKind::Grammar, body.at("grammar"), "grammar");
        if (!grammar.grammar.empty()) { take("grammar", std::move(grammar)); }
    }
    if (present("guided_whitespace_pattern")) {
        invalid_constraint("guided_whitespace_pattern is not supported; NInfer bounds whitespace "
                           "runs itself",
                           "guided_whitespace_pattern");
    }
    if (!found) { return; }
    if (request.structured_output.kind != StructuredOutputKind::None) {
        invalid_constraint(found_param + " and " + request.structured_output_param +
                               " are exclusive",
                           found_param);
    }
    try {
        text::validate_structured_output(*found);
    } catch (const std::invalid_argument& e) { invalid_constraint(e.what(), found_param); }
    request.structured_output       = std::move(*found);
    request.structured_output_param = found_param;
}

std::optional<GenerationRequest> drop_unavailable_constraints(const GenerationRequest& request,
                                                              bool structured_output,
                                                              bool unconstrained_response_format) {
    if (structured_output) { return std::nullopt; }
    std::optional<GenerationRequest> dropped;
    if (request.structured_output.kind != StructuredOutputKind::None) {
        if (!unconstrained_response_format) {
            throw ApiException(ApiError{
                .message =
                    "structured output requires the server to start with --structured-output",
                .param = request.structured_output_param,
                .code  = "response_format_not_supported"});
        }
        dropped.emplace(request);
        dropped->structured_output = {};
    }
    // Native tool constraints always have mask storage in this integration. The server flag
    // gates final response formats only; strict tool intent must never be silently dropped.
    return dropped;
}
} // namespace ninfer::serve
