#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/qwen3_5/frontend/tool_grammar.h"
#include "text/structured_output.h"
#ifndef NINFER_TOOL_GRAMMAR_ONLY
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/anthropic_messages.h"
#include "serve/translate.h"
#endif
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
namespace fi = ninfer::models::qwen3_5::frontend;
using Json = nlohmann::ordered_json;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::vector<TokenId> bytes(std::string_view value) {
    std::vector<TokenId> out;
    for (unsigned char c : value) out.push_back(c);
    return out;
}
int main() try {
    std::vector<std::string> vocab;
    for (int i = 0; i < 256; ++i) vocab.emplace_back(1, static_cast<char>(i));
    vocab.push_back("<eos>");
    text::StructuredCompiler compiler(std::move(vocab), {256});
    const std::vector<std::string> declarations = {
        R"({"type":"function","function":{"name":"alpha","parameters":{"type":"object","properties":{"s":{"type":"string"},"n":{"type":"integer"}}}}})",
        R"({"type":"function","function":{"name":"beta","parameters":{"type":"object","properties":{}}}})"};
    auto base = fi::build_tool_call_output_contract(declarations, true);
    const std::string call = "<tool_call>\n<function=beta>\n</function>\n</tool_call>";
    auto compile = [&](ToolChoice choice, StructuredOutputOptions body = {}, std::string close = "") {
        auto contract = fi::select_tool_call_contract(base, choice);
        return compiler.compile_model(fi::tool_grammar(*contract, body), body, close);
    };
    auto accepts = [&](ToolChoice choice, std::string_view value, bool eos = true,
                       StructuredOutputOptions body = {}, std::string close = "") {
        auto grammar = compile(choice, body, close);
        try {
            grammar->accept(bytes(value));
            if (eos) { const TokenId end = 256; grammar->accept({&end, 1}); }
            return true;
        } catch (const std::logic_error&) { return false; }
    };
    ToolChoice required; required.mode = ToolChoiceMode::Required;
    require(accepts(required, call), "required multi-tool choice");
    require(!accepts(required, ""), "required blocks EOS before a tool call");
    require(!accepts(required, "text"), "required must block text and early EOS");
    require(!accepts(required, "<tool_call>\n<function=unknown>"), "declared names only");
    require(accepts({}, "prefix " + call), "auto permits text before first call");
    require(!accepts({}, call + "garbage"), "terminal call sequence forbids trailing text");
    required.allowed_names = std::vector<std::string>{"alpha"};
    require(!accepts(required, call), "allowed subset enforced");
    required.allowed_names = std::vector<std::string>{"beta"};
    require(accepts(required, call), "required permits the single allowed tool");
    require(!accepts(required, ""), "single allowed tool is still required");
    ToolChoice allowed_auto; allowed_auto.allowed_names = std::vector<std::string>{"alpha"};
    require(accepts(allowed_auto, "normal answer"), "auto allowed subset permits a text answer");
    require(!accepts(allowed_auto, call), "auto also blocks tools outside allowed subset");
    required.allowed_names.reset(); required.parallel = false;
    require(!accepts(required, call + "\n" + call), "parallel=false enforced in generation");
    ToolChoice none; none.mode = ToolChoiceMode::None;
    require(accepts(none, "normal answer"), "none admits text");
    require(!accepts(none, call), "none blocks tool marker");
    require(accepts(required, "reasoning</think>\n\n" + call, true, {}, "</think>"), "required with thinking");
    StructuredOutputOptions json{StructuredOutputKind::JsonSchema,
        R"({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false})", true};
    require(accepts({}, "{\"ok\":true}", true, json), "JSON body composed with auto tools");
    require(accepts({}, call, true, json), "JSON body tool alternative");
    require(!accepts(required, "{\"ok\":true}", true, json), "required overrides JSON body alternative");
    require(accepts(none, "{\"ok\":true}", true, json), "none retains JSON body constraint");
    StructuredOutputOptions unique{StructuredOutputKind::JsonSchema,
        R"({"type":"array","items":{"type":"string"},"uniqueItems":true})",false};
    require(accepts({}, "[\"a\",\"b\"]",true,unique), "composed final JSON keeps uniqueItems validator");
    require(!accepts({}, "[\"a\",\"a\"]",true,unique), "composed final JSON blocks duplicate strings");
    auto transaction = compile(required);
    std::vector<std::uint32_t> before(9), after(9), speculative(27);
    transaction->fill_masks(before, {});
    transaction->fill_masks(speculative, bytes("<t"));
    transaction->fill_masks(after, {});
    require(before == after, "draft mask preview must not advance committed state");
    auto fork = transaction->fork(); fork->accept(bytes(call));
    transaction->fill_masks(after, {});
    require(before == after, "rejected/foreign grammar fork must not leak state");
    auto continued = compiler.compile_model(fi::tool_grammar(*fi::select_tool_call_contract(base, required), {}), {}, "", "<tool_call>\n<function=beta>\n");
    continued->accept(bytes("</function>\n</tool_call>"));
    const TokenId end = 256; continued->accept({&end,1});
    const std::vector<std::string> strict_def = {R"({"function":{"name":"typed","strict":true,"parameters":{"type":"object","properties":{"s":{"type":"string","pattern":"^[a-z ]+$"},"n":{"type":"integer","minimum":1,"maximum":9},"o":{"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false}},"required":["s","n","o"],"additionalProperties":false}}})"};
    auto typed = fi::select_tool_call_contract(fi::build_tool_call_output_contract(strict_def, true), required);
    auto strict_accepts = [&](std::string value) {
        auto grammar = compiler.compile_model(fi::tool_grammar(*typed, {}), {}, "");
        try { grammar->accept(bytes(value)); const TokenId end = 256; grammar->accept({&end,1}); return true; }
        catch (const std::logic_error&) { return false; }
    };
    const std::string strict_call = "<tool_call>\n<function=typed>\n<parameter=s>\n  abc  \n</parameter>\n<parameter=n>\n3\n</parameter>\n<parameter=o>\n{\"ok\": true}\n</parameter>\n</function>\n</tool_call>";
    require(strict_accepts(strict_call), "strict raw string, bounded integer and nested JSON");
    auto bad = strict_call; bad.replace(bad.find("\n3\n"), 3, "\n0\n");
    require(!strict_accepts(bad), "strict bound violation blocked");
    bad = strict_call; bad.replace(bad.find("abc"),3,"ABC");
    require(!strict_accepts(bad), "raw string pattern blocked");
    auto unsupported = Json::parse(strict_def[0]);
    unsupported["function"]["parameters"]["properties"]["values"] =
        {{"type", "array"}, {"items", {{"type", "string"}}}, {"uniqueItems", true}};
    bool rejected_unsupported = false;
    const std::vector<std::string> unsupported_def = {unsupported.dump()};
    try {
        (void)fi::select_tool_call_contract(
            fi::build_tool_call_output_contract(unsupported_def, true), required);
    } catch (const RequestError& error) {
        rejected_unsupported = error.kind() == RequestErrorKind::InvalidToolConstraint &&
            std::string(error.what()).find("uniqueItems") != std::string::npos;
    }
    require(rejected_unsupported, "unsupported strict tool schema fails explicitly");
    fi::ToolCallOutputDecoder decoder(typed,128);
    require(decoder.feed(strict_call).empty(), "tool markup withheld from content");
    auto parsed = decoder.finish();
    require(parsed.tool_calls.size()==1, "strict complete call published");
    auto arguments=Json::parse(parsed.tool_calls[0].arguments_json);
    require(arguments["s"]=="  abc  " && arguments["n"]==3 && arguments["o"]["ok"]==true, "strict codecs preserve exact argument values");
    fi::ToolCallOutputDecoder truncated(typed,128); (void)truncated.feed(strict_call.substr(0, strict_call.size()-1));
    require(truncated.finish().tool_calls.empty(), "truncated call never repaired or invented");
    auto parallel = fi::select_tool_call_contract(base, {});
    fi::ToolCallOutputDecoder partial(parallel,128); (void)partial.feed(call+"\n<tool_call>\n<function=alpha>\n");
    require(partial.finish().tool_calls.size()==1, "keep completed call before truncated second call");
    ToolChoice automatic; automatic.constraints = ToolConstraintMode::Automatic;
    require(!fi::select_tool_call_contract(base, automatic)->constrained, "auto opt-out retains free decoder");
    require(fi::select_tool_call_contract(fi::build_tool_call_output_contract(strict_def,true), automatic)->constrained, "strict remains mandatory under auto opt-out");
#ifndef NINFER_TOOL_GRAMMAR_ONLY
    using namespace ninfer::serve;
    const RequestLimits limits;
    Json chat={{"model","test"},{"messages",Json::array({{{"role","user"},{"content","hi"}}})},
               {"tools",Json::array({Json::parse(declarations[0]),Json::parse(declarations[1])})}, {"tool_choice","required"}};
    auto cr=parse_chat_completion_request(chat,limits).generation;
    require(cr.tool_choice.mode==ToolChoiceMode::Required && cr.tools.size()==2, "Chat required preserves declarations");
    chat["tool_choice"]={{"type","allowed_tools"},{"allowed_tools",{{"mode","required"},{"tools",Json::array({{{"type","function"},{"name","beta"}}})}}}};
    cr=parse_chat_completion_request(chat,limits).generation;
    require(cr.tools.size()==2 && cr.tool_choice.allowed_names->size()==1, "Chat allowed selection keeps prompt declarations");
    chat["tool_choice"]={{"type","function"},{"function",{{"name","beta"}}}}; chat["enable_thinking"]=true;
    cr=parse_chat_completion_request(chat,limits).generation;
    ServeOptions server;
    auto semantics=resolve_prompt_semantics(cr,server);
    auto options=to_request_options(cr,server,semantics,true);
    require(options.tool_choice.mode==ToolChoiceMode::Required && !options.tool_choice.parallel && !options.output.preserve_special_tokens, "Chat named choice enforced without forced prompt opener");
    auto prompt=to_prompt_input(cr,semantics,{});
    require(prompt.options.forced_tool_name.empty() && prompt.options.tool_jsons.size()==2, "selection absent from reusable prompt");
    Json responses={{"model","test"},{"input","hi"},{"tools",Json::array({{{"type","function"},{"name","typed"},{"strict",true},{"parameters",Json::parse(strict_def[0])["function"]["parameters"]}}})},{"tool_choice","required"},{"tool_constraints","basic"}};
    auto rr=parse_openai_responses_create_request(responses,limits);
    require(rr.prompt.generation.tools[0].strict && rr.prompt.generation.tool_choice.mode==ToolChoiceMode::Required, "Responses strict and required transport");
    Json anth={{"model","test"},{"max_tokens",128},{"messages",Json::array({{{"role","user"},{"content","hi"}}})},
               {"tools",Json::array({{{"name","alpha"},{"input_schema",Json::object()}},{{"name","beta"},{"input_schema",Json::object()}}})},{"tool_choice",{{"type","any"}}}};
    auto ar=parse_anthropic_messages_request(anth,limits).generation;
    require(ar.tool_choice.mode==ToolChoiceMode::Required && ar.tools.size()==2, "Anthropic any over multiple tools");
    anth["tool_choice"]={{"type","tool"},{"name","beta"}};
    ar=parse_anthropic_messages_request(anth,limits).generation;
    options=to_request_options(ar,server,resolve_prompt_semantics(ar,server),true);
    require(options.tool_choice.parallel && options.tool_choice.mode==ToolChoiceMode::Required, "Anthropic named choice preserves parallel semantics");
#endif
    std::cout << "core tool constraints: PASS (grammar, strict codecs, transactional masks"
#ifndef NINFER_TOOL_GRAMMAR_ONLY
              << ", three protocols"
#endif
              << ")\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what()<<"\n"; return 1; }
