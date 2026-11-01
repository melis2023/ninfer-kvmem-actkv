#include "text/structured_output.h"
#include "ninfer/request.h"
#include "text/schema_normalization.h"
#include "text/string_constraints.h"
#include <bit>
#include <xgrammar/xgrammar.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;

bool schema_value_equal(const Json& left, const Json& right) {
    if (left.is_number() && right.is_number())
        return left.get<long double>() == right.get<long double>();
    if (left.is_object() && right.is_object()) {
        if (left.size() != right.size()) return false;
        for (const auto& [key, value] : left.items()) {
            if (!right.contains(key) || !schema_value_equal(value, right.at(key))) return false;
        }
        return true;
    }
    if (left.is_array() && right.is_array()) {
        if (left.size() != right.size()) return false;
        for (std::size_t index = 0; index < left.size(); ++index)
            if (!schema_value_equal(left[index], right[index])) return false;
        return true;
    }
    return left == right;
}

bool contains_schema_reference(const Json& schema) {
    if (!schema.is_object()) return false;
    if (schema.contains("$ref") || schema.contains("$dynamicRef") || schema.contains("$recursiveRef")) return true;
    for (const auto& [key, value] : schema.items()) {
        if (key == "properties" || key == "$defs" || key == "definitions" || key == "patternProperties" || key == "dependentSchemas") {
            if (value.is_object()) for (const auto& child : value)
                if (contains_schema_reference(child)) return true;
        } else if (key == "anyOf" || key == "allOf" || key == "oneOf" || key == "prefixItems") {
            if (value.is_array()) for (const auto& child : value)
                if (contains_schema_reference(child)) return true;
        } else if (key == "items" || key == "additionalProperties" || key == "propertyNames" || key == "not" || key == "contains" || key == "if" || key == "then" || key == "else") {
            if (contains_schema_reference(value)) return true;
        }
    }
    return false;
}

void schema_check(const Json& s) {
    if (s.is_boolean()) { return; }
    if (!s.is_object()) { throw std::invalid_argument("JSON schema must be an object or boolean"); }
    static const std::unordered_set<std::string> annotations = {
        "$schema",  "title",    "description", "default",
        "examples", "$comment", "$defs",       "definitions", "$id"};
    static const std::unordered_set<std::string> supported = {"type",
                                                              "properties",
                                                              "required",
                                                              "additionalProperties",
                                                              "items",
                                                              "uniqueItems",
                                                              "prefixItems",
                                                              "minItems",
                                                              "maxItems",
                                                              "maxProperties",
                                                              "pattern",
                                                              "format",
                                                              "minLength",
                                                              "maxLength",
                                                              "enum",
                                                              "const",
                                                              "anyOf",
                                                              "$ref",
                                                              "minimum",
                                                              "maximum",
                                                              "exclusiveMinimum",
                                                              "exclusiveMaximum"};
    for (const auto& [key, value] : s.items()) {
        if (key == "$schema" && value != "https://json-schema.org/draft/2020-12/schema" &&
            value != "http://json-schema.org/draft-07/schema#") {
            throw std::invalid_argument("unsupported JSON Schema dialect");
        }
        if ((key == "minLength" || key == "maxLength" || key == "minItems" || key == "maxItems" || key == "maxProperties") &&
            (!value.is_number_integer() || value < 0 || value > 2147483647)) {
            throw std::invalid_argument(key + " must be a nonnegative 32-bit integer");
        }
        if (key == "minimum" || key == "maximum" || key == "exclusiveMinimum" ||
            key == "exclusiveMaximum") {
            // The pinned compiler represents bounds as doubles. Keep integer bounds exact
            // and reject non-finite values before they reach its range arithmetic.
            if (!value.is_number() || !std::isfinite(value.get<double>()) ||
                std::abs(value.get<long double>()) > 9007199254740991.0L) {
                throw std::invalid_argument(key + " requires a finite bound within +/- (2^53-1)");
            }
            if (!s.contains("type") || (s.at("type") != "integer" && s.at("type") != "number")) {
                throw std::invalid_argument(
                    "numeric bounds require explicit integer or number type");
            }
        }
        if (!annotations.contains(key) && !supported.contains(key)) {
            throw std::invalid_argument("unsupported JSON schema keyword: " + key);
        }
        if (key == "$id") {
            if (!value.is_string() || contains_schema_reference(s))
                throw std::invalid_argument("schema resource identifiers with references need explicit scoped resolution");
        }
        if (key == "maxProperties") {
            const bool object_type = s.contains("type") &&
                (s.at("type") == "object" || (s.at("type").is_array() &&
                 std::any_of(s.at("type").begin(), s.at("type").end(), [](const auto& type) { return type == "object"; })));
            if (!object_type && !s.contains("properties") && !s.contains("additionalProperties"))
                throw std::invalid_argument("maxProperties requires a represented object branch");
        }
        if (key == "uniqueItems" && !value.is_boolean())
            throw std::invalid_argument("uniqueItems must be boolean");
        if (key == "required") {
            if (!value.is_array()) throw std::invalid_argument("required must be an array of unique property names");
            std::unordered_set<std::string> names;
            for (const auto& name : value) {
                if (!name.is_string() || !names.insert(name.get<std::string>()).second)
                    throw std::invalid_argument("required must be an array of unique property names");
                const bool declared = s.contains("properties") && s.at("properties").is_object() &&
                                      s.at("properties").contains(name.get<std::string>());
                if (!declared && s.contains("additionalProperties") && s.at("additionalProperties") == false)
                    throw std::invalid_argument("required property is forbidden by additionalProperties: " + name.get<std::string>());
            }
        }
        if (key == "pattern" || key == "format") {
            if (!value.is_string() || !s.contains("type") || s.at("type") != "string") {
                throw std::invalid_argument(key + " requires a string value and explicit string type");
            }
            // The pinned converter chooses one string representation. Refuse conjunctions
            // until their intersection is implemented; never silently discard a bound.
            if (s.contains("minLength") || s.contains("maxLength") ||
                (s.contains("pattern") && s.contains("format"))) {
                throw std::invalid_argument("pattern/format cannot be combined with length bounds or each other");
            }
            if (key == "format" && value != "date" && value != "date-time" && value != "time") {
                throw std::invalid_argument("unsupported asserted string format: " + value.get<std::string>());
            }
        }
        if (key == "$ref" &&
            (!value.is_string() || (value != "#" && !value.get<std::string>().starts_with("#/")))) {
            throw std::invalid_argument("JSON schema supports only local fragment $ref values");
        }
        if (key == "$ref" && value != "#") {
            const auto ref = value.get<std::string>();
            // The pinned compiler interprets literal object paths, not RFC 6901 escapes.
            // Reject ambiguous spellings rather than resolving a different schema silently.
            if (ref.find_first_of("~%") != std::string::npos || ref.ends_with('/') ||
                ref.find("//") != std::string::npos) {
                throw std::invalid_argument(
                    "$ref requires nonempty, unescaped object path segments");
            }
        }
        if (key == "properties" || key == "$defs" || key == "definitions") {
            if (!value.is_object()) { throw std::invalid_argument(key + " must be an object"); }
            for (const auto& child : value) { schema_check(child); }
        } else if (key == "items" || key == "additionalProperties") {
            schema_check(value);
        } else if (key == "anyOf" || key == "prefixItems") {
            if (!value.is_array()) { throw std::invalid_argument(key + " must be an array"); }
            for (const auto& child : value) { schema_check(child); }
        }
    }
    // XGrammar prioritizes these branches over their siblings. Disallow combinations that
    // would otherwise silently discard constraints (annotations and definitions are harmless).
    for (const char* branch : {"$ref", "const", "enum", "anyOf"}) {
        if (!s.contains(branch)) { continue; }
        for (const auto& [key, value] : s.items()) {
            if (key != branch && !annotations.contains(key)) {
                // Pydantic Literal may emit both assertions. Compile the constant only
                // when it satisfies enum as well; object order and number kinds are semantic.
                if (((std::string_view(branch) == "const" && key == "enum") ||
                     (std::string_view(branch) == "enum" && key == "const")) &&
                    s.at("enum").is_array() &&
                    std::any_of(s.at("enum").begin(), s.at("enum").end(),
                                [&](const auto& candidate) { return schema_value_equal(s.at("const"), candidate); })) {
                    continue;
                }
                if (key == "type" &&
                    (std::string_view(branch) == "enum" || std::string_view(branch) == "const")) {
                    const auto matches = [&](const Json& v) {
                        if (!value.is_string()) { return false; }
                        const auto type = value.get<std::string>();
                        return (type == "string" && v.is_string()) ||
                               (type == "integer" && v.is_number_integer()) ||
                               (type == "number" && v.is_number()) ||
                               (type == "boolean" && v.is_boolean()) ||
                               (type == "null" && v.is_null()) ||
                               (type == "array" && v.is_array()) ||
                               (type == "object" && v.is_object());
                    };
                    if (s.contains("const") && matches(s.at("const"))) { continue; }
                    if (s.contains("enum") && s.at("enum").is_array() &&
                        std::all_of(s.at("enum").begin(), s.at("enum").end(), matches)) {
                        continue;
                    }
                }
                throw std::invalid_argument(std::string(branch) + " cannot be combined with " +
                                            key);
            }
        }
    }
}

// An EBNF string literal of `text`: the grammar parser reads UTF-8 as is, so only the quote, the
// backslash and control bytes are escaped.
std::string ebnf_literal(std::string_view text) {
    std::string out = "\"";
    for (const char ch : text) {
        const auto byte = static_cast<unsigned char>(ch);
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (byte < 0x20 || byte == 0x7f) {
            char escape[5];
            std::snprintf(escape, sizeof(escape), "\\x%02x", byte);
            out += escape;
        } else {
            out += ch;
        }
    }
    return out + "\"";
}

// The grammar of a Choice: the response is exactly one of the strings.
std::string choice_grammar(const std::vector<std::string>& choices) {
    std::string ebnf = "root ::= ";
    for (std::size_t index = 0; index < choices.size(); ++index) {
        if (index != 0) { ebnf += " | "; }
        ebnf += ebnf_literal(choices[index]);
    }
    return ebnf;
}

const char* kind_name(StructuredOutputKind kind) {
    switch (kind) {
    case StructuredOutputKind::Regex:
        return "regex";
    case StructuredOutputKind::Choice:
        return "choices";
    case StructuredOutputKind::Grammar:
        return "grammar";
    default:
        return "JSON schema";
    }
}
} // namespace

void validate_structured_output(const StructuredOutputOptions& options) {
    const bool text_grammar = options.kind == StructuredOutputKind::Regex ||
                              options.kind == StructuredOutputKind::Grammar;
    if (options.kind == StructuredOutputKind::JsonSchema) {
        try {
            const auto normalized_schema = normalize_json_schema(options.schema);
            schema_check(Json::parse(normalized_schema));
            StringConstraintState::AnalyzeSchema(normalized_schema);
        } catch (const Json::exception& e) {
            throw std::invalid_argument(std::string("invalid JSON schema: ") + e.what());
        }
    } else if (!options.schema.empty()) {
        throw std::invalid_argument("schema requires JsonSchema mode");
    }
    if (text_grammar) {
        if (options.grammar.empty()) {
            throw std::invalid_argument(std::string(kind_name(options.kind)) + " is empty");
        }
        // Parsing needs no vocabulary, so a malformed constraint fails before admission.
        try {
            if (options.kind == StructuredOutputKind::Regex) {
                (void)xgrammar::Grammar::FromRegex(options.grammar);
            } else {
                (void)xgrammar::Grammar::FromEBNF(options.grammar, "root");
            }
        } catch (const std::exception& e) {
            throw std::invalid_argument(std::string("invalid ") + kind_name(options.kind) + ": " +
                                        e.what());
        }
    } else if (!options.grammar.empty()) {
        throw std::invalid_argument("grammar text requires Regex or Grammar mode");
    }
    if (options.kind == StructuredOutputKind::Choice) {
        if (options.choices.empty()) { throw std::invalid_argument("choices are empty"); }
        if (std::any_of(options.choices.begin(), options.choices.end(),
                        [](const std::string& choice) { return choice.empty(); })) {
            throw std::invalid_argument("a choice is empty");
        }
    } else if (!options.choices.empty()) {
        throw std::invalid_argument("choices require Choice mode");
    }
}

struct GrammarState::Impl {
    xgrammar::GrammarMatcher matcher;
    int vocab_size;
    std::shared_ptr<const std::vector<std::string>> vocabulary;
    std::shared_ptr<const std::vector<std::uint8_t>> stops;
    std::unique_ptr<StringConstraintState> strings;

    Impl(xgrammar::GrammarMatcher matcher, int vocab_size,
         std::shared_ptr<const std::vector<std::string>> vocabulary,
         std::shared_ptr<const std::vector<std::uint8_t>> stops,
         std::unique_ptr<StringConstraintState> strings)
        : matcher(std::move(matcher)), vocab_size(vocab_size), vocabulary(std::move(vocabulary)),
          stops(std::move(stops)), strings(std::move(strings)) {}
    std::string_view semantic_bytes(TokenId token) const {
        return (*stops)[token] ? std::string_view{} : std::string_view((*vocabulary)[token]);
    }
};

GrammarState::GrammarState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GrammarState::~GrammarState() = default;
GrammarState::GrammarState(GrammarState&&) noexcept = default;
GrammarState& GrammarState::operator=(GrammarState&&) noexcept = default;

std::unique_ptr<GrammarState> GrammarState::fork() const {
    auto strings = impl_->strings ? std::make_unique<StringConstraintState>(*impl_->strings) : nullptr;
    return std::unique_ptr<GrammarState>(new GrammarState(
        std::make_unique<Impl>(impl_->matcher.Fork(), impl_->vocab_size, impl_->vocabulary,
                               impl_->stops, std::move(strings))));
}

void GrammarState::accept(std::span<const TokenId> tokens) {
    for (TokenId token : tokens) {
        if (token < 0 || token >= impl_->vocab_size || impl_->matcher.IsTerminated())
            throw GrammarViolation("generated token violates the structured output grammar");
        auto next = impl_->strings ? std::make_unique<StringConstraintState>(*impl_->strings) : nullptr;
        if ((next && !next->accept(impl_->semantic_bytes(token))) || !impl_->matcher.AcceptToken(token))
            throw GrammarViolation("generated token " + std::to_string(token) +
                                   " violates the structured output grammar");
        if (next) impl_->strings = std::move(next);
    }
}

void GrammarState::fill_masks(std::span<std::uint32_t> masks,
                              std::span<const TokenId> drafts) const {
    const int words = xgrammar::GetBitmaskSize(impl_->vocab_size);
    if (masks.size() != (drafts.size() + 1) * words)
        throw std::logic_error("incorrect grammar mask shape");
    std::fill(masks.begin(), masks.end(), ~std::uint32_t{0});
    // The grammar after each prefix of the drafts the grammar admits, made in order (accepting a
    // token is cheap); the masks, the costly part, then fill one column per thread. A draft the
    // grammar rejects leaves the columns after it unrestricted.
    struct Column {
        xgrammar::GrammarMatcher matcher;
        std::unique_ptr<StringConstraintState> strings;
    };
    std::vector<Column> columns;
    columns.reserve(drafts.size() + 1);
    auto matcher = impl_->matcher.Fork();
    auto strings = impl_->strings ? std::make_unique<StringConstraintState>(*impl_->strings) : nullptr;
    for (std::size_t col = 0; col <= drafts.size(); ++col) {
        if (matcher.IsTerminated()) break;
        columns.push_back({matcher.Fork(), strings ? std::make_unique<StringConstraintState>(*strings) : nullptr});
        if (col == drafts.size()) break;
        const TokenId token = drafts[col];
        if (token < 0 || token >= impl_->vocab_size || !matcher.AcceptToken(token)) break;
        if (strings && !strings->accept(impl_->semantic_bytes(token))) break;
    }
    const auto fill = [&](std::size_t col) {
        Column& column        = columns[col];
        std::int64_t shape[2] = {1, words};
        DLTensor tensor{};
        tensor.data = masks.data() + col * words;
        tensor.device = {kDLCPU, 0}; tensor.ndim = 2;
        tensor.dtype = {kDLInt, 32, 1}; tensor.shape = shape;
        column.matcher.FillNextTokenBitmask(&tensor);
        if (impl_->vocab_size % 32)
            masks[(col + 1) * words - 1] &= (1U << (impl_->vocab_size % 32)) - 1;
        auto row = masks.subspan(col * words, words);
        // Semantic work is absent for ordinary schemas.
        const auto& constraint = column.strings;
        if (constraint) {
            for (int word = 0; word < words; ++word) {
                auto candidates = row[word];
                while (candidates) {
                    const int bit = std::countr_zero(candidates);
                    const TokenId token = word * 32 + bit;
                    const auto bytes = impl_->semantic_bytes(token);
                    if (!(*impl_->stops)[token] && constraint->needs_check(bytes) && !constraint->accepts(bytes))
                        row[word] &= ~(1U << bit);
                    candidates &= candidates - 1;
                }
            }
        }
        if (std::none_of(row.begin(), row.end(), [](auto word) { return word != 0; }))
            throw std::runtime_error("structured output grammar has no admissible next token");
    };
    std::vector<std::exception_ptr> errors(columns.size());
    std::vector<std::thread> threads;
    for (std::size_t col = 1; col < columns.size(); ++col) {
        threads.emplace_back([&, col] {
            try { fill(col); } catch (...) { errors[col] = std::current_exception(); }
        });
    }
    if (!columns.empty()) {
        try { fill(0); } catch (...) { errors[0] = std::current_exception(); }
    }
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
}

struct StructuredCompiler::Impl {
    std::shared_ptr<const std::vector<std::string>> vocabulary;
    std::shared_ptr<const std::vector<std::uint8_t>> stops;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
    std::mutex mutex;

    Impl(std::vector<std::string> vocab, std::vector<int> stop_ids)
        : vocabulary(std::make_shared<const std::vector<std::string>>(std::move(vocab))),
          tokenizer(*vocabulary, xgrammar::VocabType::RAW, static_cast<int>(vocabulary->size()), stop_ids),
          compiler(tokenizer, 4, true, 256 * 1024 * 1024) {
        auto flags = std::make_shared<std::vector<std::uint8_t>>(vocabulary->size());
        for (int token : stop_ids) {
            if (token < 0 || static_cast<std::size_t>(token) >= flags->size())
                throw std::invalid_argument("stop token is outside the vocabulary");
            (*flags)[token] = 1;
        }
        stops = std::move(flags);
    }
};

StructuredCompiler::StructuredCompiler(std::vector<std::string> vocab, std::vector<int> stops)
    : impl_(std::make_unique<Impl>(std::move(vocab), std::move(stops))) {}

StructuredCompiler::~StructuredCompiler() = default;

std::shared_ptr<GrammarState>
StructuredCompiler::compile(const StructuredOutputOptions& options,
                            const StructuredOutputEnvelope& envelope) {
    validate_structured_output(options);
    if (options.kind == StructuredOutputKind::None) { return {}; }
    std::lock_guard lock(impl_->mutex);
    try {
        const bool enveloped =
            !envelope.reasoning_close.empty() || !envelope.alternative_format.empty();
        const bool json = options.kind == StructuredOutputKind::JsonObject ||
                          options.kind == StructuredOutputKind::JsonSchema;
        std::unique_ptr<StringConstraintState> strings;
        // The final response's grammar on its own, and as a structural-tag format for an envelope.
        std::optional<xgrammar::CompiledGrammar> grammar;
        Json content;
        if (json) {
            const auto normalized_schema = options.kind == StructuredOutputKind::JsonSchema
                                               ? normalize_json_schema(options.schema)
                                               : std::string("{\"type\":\"object\"}");
            strings = std::make_unique<StringConstraintState>(normalized_schema,
                                                              envelope.reasoning_close);
            if (!strings->has_constraints()) strings.reset();
            // strict_mode=false retains JSON Schema defaults for additional properties/items; a
            // strict JsonSchema request closes undeclared object properties.
            const bool strict = options.kind == StructuredOutputKind::JsonSchema && options.strict;
            grammar = impl_->compiler.CompileJSONSchema(normalized_schema, true, std::nullopt,
                                                        std::nullopt, strict, 8);
            if (enveloped) {
                std::ostringstream ebnf;
                ebnf << grammar->GetGrammar();
                content = Json{{"type", "grammar"}, {"grammar", ebnf.str()}};
            }
        } else if (options.kind == StructuredOutputKind::Regex) {
            content = Json{{"type", "regex"}, {"pattern", options.grammar}};
            if (!enveloped) { grammar = impl_->compiler.CompileRegex(options.grammar); }
        } else {
            const std::string ebnf = options.kind == StructuredOutputKind::Choice
                                         ? choice_grammar(options.choices)
                                         : options.grammar;
            content                = Json{{"type", "grammar"}, {"grammar", ebnf}};
            if (!enveloped) { grammar = impl_->compiler.CompileGrammar(ebnf, "root"); }
        }
        if (enveloped) {
            // The final response's root starts at its first byte; Qwen's reasoning close is
            // followed by whitespace (including the canonical budget-control suffix's two
            // newlines), which the output session strips from the content.
            content = Json{
                {"type", "sequence"},
                {"elements", Json::array({Json{{"type", "regex"}, {"pattern", "[ \\t\\r\\n]{0,8}"}},
                                          content})}};
            if (!envelope.alternative_format.empty()) {
                content = Json{
                    {"type", "or"},
                    {"elements", Json::array({content, Json::parse(envelope.alternative_format)})}};
            }
            if (!envelope.reasoning_close.empty()) {
                // any_text excludes the first closing delimiter, including split-token and
                // overlapping prefixes. A wildcard repetition would allow reasoning to consume
                // the delimiter and bypass the final-content constraint.
                content =
                    Json{{"type", "sequence"},
                         {"elements",
                          Json::array(
                              {Json{{"type", "any_text"},
                                    {"excludes", Json::array({envelope.reasoning_close})}},
                               Json{{"type", "const_string"}, {"value", envelope.reasoning_close}},
                               content})}};
            }
            grammar = impl_->compiler.CompileStructuralTag(
                Json{{"type", "structural_tag"}, {"format", content}}.dump());
        }
        return std::shared_ptr<GrammarState>(new GrammarState(std::make_unique<GrammarState::Impl>(
            xgrammar::GrammarMatcher(*grammar), impl_->tokenizer.GetVocabSize(), impl_->vocabulary,
            impl_->stops, std::move(strings))));
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("cannot compile ") + kind_name(options.kind) +
                                    ": " + e.what());
    }
}

std::shared_ptr<GrammarState>
StructuredCompiler::compile_format(const std::string& format,
                                   const StructuredOutputEnvelope& envelope) {
    std::lock_guard lock(impl_->mutex);
    try {
        Json content = Json::parse(format);
        if (!envelope.reasoning_close.empty()) {
            content = Json{
                {"type", "sequence"},
                {"elements",
                 Json::array({Json{{"type", "any_text"},
                                   {"excludes", Json::array({envelope.reasoning_close})}},
                              Json{{"type", "const_string"}, {"value", envelope.reasoning_close}},
                              content})}};
        }
        auto grammar = impl_->compiler.CompileStructuralTag(
            Json{{"type", "structural_tag"}, {"format", content}}.dump());
        return std::shared_ptr<GrammarState>(new GrammarState(std::make_unique<GrammarState::Impl>(
            xgrammar::GrammarMatcher(grammar), impl_->tokenizer.GetVocabSize(), impl_->vocabulary,
            impl_->stops, nullptr)));
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("cannot compile tool calls: ") + e.what());
    }
}
std::shared_ptr<GrammarState> StructuredCompiler::compile_model(
    const std::string& source, const StructuredOutputOptions& body, const std::string& close,
    const std::string& continuation) {
    validate_structured_output(body);
    std::lock_guard lock(impl_->mutex);
    auto compiled = impl_->compiler.CompileGrammar(source);
    if (!close.empty()) {
        Json content{{"type", "grammar"}, {"grammar", source}};
        content = Json{{"type", "sequence"}, {"elements", Json::array({
            Json{{"type", "any_text"}, {"excludes", Json::array({close})}},
            Json{{"type", "const_string"}, {"value", close}},
            Json{{"type", "regex"}, {"pattern", "[ \\t\\r\\n]{0,8}"}}, content})}};
        compiled = impl_->compiler.CompileStructuralTag(
            Json{{"type", "structural_tag"}, {"format", content}}.dump());
    }
    std::unique_ptr<StringConstraintState> strings;
    if (body.kind == StructuredOutputKind::JsonSchema) {
        strings = std::make_unique<StringConstraintState>(normalize_json_schema(body.schema), close);
        if (!strings->has_constraints()) strings.reset();
    }
    xgrammar::GrammarMatcher matcher(compiled);
    if (!continuation.empty() && !matcher.AcceptString(continuation))
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "assistant continuation is not a prefix of the tool grammar");
    return std::shared_ptr<GrammarState>(new GrammarState(std::make_unique<GrammarState::Impl>(
        std::move(matcher), impl_->tokenizer.GetVocabSize(),
        impl_->vocabulary, impl_->stops, std::move(strings))));
}
} // namespace ninfer::text
