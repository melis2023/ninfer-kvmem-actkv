#include "models/qwen3_5/frontend/tool_grammar.h"
#include "text/structured_output.h"
#include "text/schema_normalization.h"
#include <sstream>

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"

#include <nlohmann/json.hpp>
#include <optional>

namespace ninfer::models::qwen3_5::frontend {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolCallOutputContract;

xgrammar::Grammar arguments(const Contract::Tool& tool) {
    try {
        std::string source;
        if (tool.strict)
            source = text::normalize_json_schema(tool.schema_json);
        else
            source = R"({"type":"object","additionalProperties":{"type":"string"}})";
        return xgrammar::JSONSchemaToGrammar(
            source, false, std::nullopt, std::pair<std::string, std::string>{", ", ": "}, false,
            std::nullopt, false, xgrammar::JSONFormat::kQwenXML);
    } catch (const std::exception& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool '" + tool.name + "': " + error.what());
    }
}

xgrammar::Grammar build(const Contract& contract, const StructuredOutputOptions& body) {
    std::optional<xgrammar::Grammar> json;
    if (body.kind != StructuredOutputKind::None) {
        text::validate_structured_output(body);
        json = xgrammar::JSONSchemaToGrammar(
            body.kind == StructuredOutputKind::JsonObject ? "{\"type\":\"object\"}"
                : text::normalize_json_schema(body.schema), true, std::nullopt, std::nullopt,
            body.strict, 8);
    }
    if (json && contract.tools.empty()) return *json;
    xgrammar::GrammarBuilder builder;
    if (contract.tools.empty())
        return builder.Get(
            builder.AddRuleWithHint("root", builder.AddTagDispatch({{}, false, {"<tool_call>"}})));
    std::vector<int32_t> choices;
    std::optional<int32_t> basic_arguments;
    for (const auto& tool : contract.tools) {
        const auto body = !tool.strict && basic_arguments
                              ? *basic_arguments
                              : xgrammar::SubGrammarAdder::Apply(&builder, arguments(tool));
        if (!tool.strict) basic_arguments = body;
        choices.push_back(builder.AddSequence(
            {builder.AddByteString("\n<function=" + tool.name + ">\n"), builder.AddRuleRef(body),
             builder.AddByteString("</function>\n</tool_call>")}));
    }
    const auto call_body = builder.AddRuleWithHint("call_body", builder.AddChoices(choices));
    const auto next_call = builder.AddSequence(
        {builder.AddByteString("\n<tool_call>"), builder.AddRuleRef(call_body)});
    const auto tail  = contract.parallel ? builder.AddRepeatFromExpr("calls", next_call, 0, -1)
                                         : builder.AddEmptyStr();
    const auto first = builder.AddRuleWithHint(
        "calls", builder.AddSequence({builder.AddRuleRef(call_body), tail}));
    const auto root =
        (contract.required || json.has_value())
            ? builder.AddSequence({builder.AddByteString("<tool_call>"), builder.AddRuleRef(first)})
            : builder.AddTagDispatch({{{"<tool_call>", first}}, false, {}});
    auto calls =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    return json && !contract.required ? xgrammar::Grammar::Union({*json, calls}) : calls;
}
} // namespace

std::string tool_grammar(const Contract& contract, const StructuredOutputOptions& body) {
    std::ostringstream source;
    source << build(contract, body);
    return source.str();
}
} // namespace ninfer::models::qwen3_5::frontend
