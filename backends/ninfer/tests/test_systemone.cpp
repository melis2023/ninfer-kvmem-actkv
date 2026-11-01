// Typed decisions on the wire, without an Engine: System One's and OpenAI's requests, the labels
// and the prompt each question becomes, the answer read from the first token's readout (or its
// top alternatives), and both responses.

#include "serve/generation_service.h"
#include "serve/systemone.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

bool close_to(double a, double b) { return std::abs(a - b) < 1e-6; }

// The user turn's text as the chat template renders it: its parts back to back.
std::string user_text(const GenerationRequest& read) {
    std::string text;
    for (const ContentPart& part : read.messages.at(1).content) { text += part.text; }
    return text;
}

struct Refusal {
    int status = 0;
    std::string param;
};

Refusal systemone_refusal(const char* body) {
    try {
        (void)parse_systemone_request(RequestJson::parse(body));
    } catch (const ApiException& error) {
        return {.status = error.error().status, .param = error.error().param};
    }
    return {};
}

Refusal decisions_refusal(const char* body) {
    try {
        (void)parse_decisions_request(RequestJson::parse(body));
    } catch (const ApiException& error) {
        return {.status = error.error().status, .param = error.error().param};
    }
    return {};
}

const char* kSystemOne = R"({
  "model": "m",
  "state": {"ticket": "My card was charged twice for one order."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "payments and refunds", "shipping": null}},
    "urgent": {"type": "noul", "instructions": "Reply within the hour?",
               "criteria": {"true": "money is moving"}},
    "tone": {"type": "score", "instructions": "How upset?",
             "criteria": ["calm", "annoyed", "furious"]}
  }
})";

const char* kOpenAI = R"({
  "model": "m",
  "input": [{"role": "user", "content": [{"type": "input_text", "text": "Broken screen."}]},
            {"role": "user", "content": "Second message."}],
  "questions": [
    {"type": "predicate", "name": "damaged", "instructions": "Damaged item?"},
    {"type": "choice", "name": "department", "instructions": "Which department?",
     "choices": [{"value": "returns", "description": "RMA"}, {"value": true}]},
    {"type": "score", "instructions": "How severe?",
     "levels": [{"label": "cosmetic"}, {"label": "broken"}]}
  ]
})";

// A first-token record whose readout holds the given label log probabilities and whose top
// alternatives are the given (id, probability) pairs, most likely first.
ninfer::TokenLogprob read(std::vector<double> readout,
                          std::vector<std::pair<ninfer::TokenId, double>> top) {
    ninfer::TokenLogprob first;
    first.top_ids.fill(-1);
    for (std::size_t k = 0; k < top.size(); ++k) {
        first.top_ids[k]    = top[k].first;
        first.top_values[k] = static_cast<float>(std::log(top[k].second));
    }
    first.id      = first.top_ids[0];
    first.logprob = first.top_values[0];
    first.readout.assign(readout.begin(), readout.end());
    return first;
}

void systemone_parsing() {
    const DecisionRequest request = parse_systemone_request(RequestJson::parse(kSystemOne));
    expect(request.dialect == DecisionDialect::SystemOne && request.model == "m",
           "System One dialect and model");
    expect(request.state == R"({"ticket":"My card was charged twice for one order."})",
           "a JSON state becomes its compact text");
    expect(request.questions.size() == 3 && request.questions[0].id == "team" &&
               request.questions[1].id == "urgent" && request.questions[2].id == "tone",
           "questions keep the request's order");
    const DecisionQuestion& team = request.questions[0];
    expect(team.type == DecisionType::Choice && team.options.size() == 2 &&
               team.options[0].text == "billing" &&
               team.options[0].description == "payments and refunds" &&
               team.options[1].text == "shipping" && !team.options[1].description,
           "a choice's options with and without descriptions");
    const DecisionQuestion& urgent = request.questions[1];
    expect(urgent.type == DecisionType::Noul && urgent.options.size() == 2 &&
               urgent.options[0].text == "yes" && urgent.options[0].description == "money is moving" &&
               urgent.options[1].text == "no" && !urgent.options[1].description,
           "a noul's yes and no with the true description");
    const DecisionQuestion& tone = request.questions[2];
    expect(tone.type == DecisionType::Score && tone.options.size() == 3 &&
               tone.options[2].text == "furious" && tone.options[2].value == Json(2),
           "a score's levels, lowest first");
    const DecisionRequest plain = parse_systemone_request(RequestJson::parse(
        R"({"state": "s", "instructions": "ctx",
            "questions": {"q": {"type": "noul", "instructions": {"k": 1}}}})"));
    expect(plain.state == "s" && plain.instructions == "ctx" &&
               plain.questions[0].instructions == R"({"k":1})" &&
               !plain.questions[0].options[0].description,
           "a string state, context, JSON instructions, a noul without criteria");

    expect(systemone_refusal(R"({"questions": {"q": {"type": "noul", "instructions": "i"}}})")
                   .param == "state",
           "a state is required");
    expect(systemone_refusal(R"({"state": "s", "questions": {}})").param == "questions",
           "at least one question");
    expect(systemone_refusal(R"({"state": "s", "extra": 1, "questions": {}})").param == "extra",
           "unknown top-level fields are refused");
    expect(systemone_refusal(
               R"({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "x": 1}}})")
                   .param == "questions.q.x",
           "unknown question fields are refused");
    expect(systemone_refusal(
               R"({"state": "s", "questions": {"q": {"type": "choice", "instructions": "i",
                   "criteria": {"only": null}}}})")
                   .param == "questions.q.criteria",
           "a choice needs two options");
    expect(systemone_refusal(
               R"({"state": "s", "questions": {"q": {"type": "score", "instructions": "i",
                   "criteria": ["a"]}}})")
                   .param == "questions.q.criteria",
           "a score needs two levels");
    expect(systemone_refusal(
               R"({"state": "s", "questions": {"q": {"type": "predicate", "instructions": "i"}}})")
                   .param == "questions.q.type",
           "OpenAI's predicate is not a System One type");
    expect(systemone_refusal(
               R"({"state": "s", "joint": true, "questions": {"q": {"type": "noul", "instructions": "i"}}})")
                   .param == "joint",
           "joint reads are refused");
    expect(systemone_refusal(
               R"({"state": "s", "audio": ["data:audio/wav;base64,AA=="],
                   "questions": {"q": {"type": "noul", "instructions": "i"}}})")
                   .status == 501,
           "audio is not implemented");
    std::string many = R"({"state": "s", "questions": {)";
    for (int q = 0; q < 201; ++q) {
        many += (q ? "," : "") + std::string("\"q") + std::to_string(q) +
                R"(": {"type": "noul", "instructions": "i"})";
    }
    many += "}}";
    expect(systemone_refusal(many.c_str()).param == "questions", "at most 200 questions");
}

void decisions_parsing() {
    const DecisionRequest request = parse_decisions_request(RequestJson::parse(kOpenAI));
    expect(request.dialect == DecisionDialect::OpenAI && request.model == "m",
           "OpenAI dialect and model");
    expect(request.state == "Broken screen.\n\nSecond message.",
           "user messages with input_text parts and strings join the state");
    expect(request.questions.size() == 3, "three questions");
    const DecisionQuestion& damaged = request.questions[0];
    expect(damaged.type == DecisionType::Noul && damaged.named && damaged.id == "damaged" &&
               damaged.options.size() == 2,
           "a predicate is a noul");
    const DecisionQuestion& department = request.questions[1];
    expect(department.type == DecisionType::Choice && department.options.size() == 2 &&
               department.options[0].text == "returns" && department.options[0].description == "RMA" &&
               department.options[1].text == "true" && department.options[1].value == Json(true),
           "choice values keep their type");
    const DecisionQuestion& severity = request.questions[2];
    expect(severity.type == DecisionType::Score && !severity.named &&
               severity.options[1].text == "broken",
           "an unnamed score with labelled levels");

    expect(decisions_refusal(R"({"input": "i", "questions": [{"type": "noul", "instructions": "i"}]})")
                   .param == "questions[0].type",
           "System One's noul is not an OpenAI type");
    expect(decisions_refusal(
               R"({"input": "i", "questions": [{"type": "choice", "instructions": "i",
                   "choices": [{"value": "a"}, {"value": "a"}]}]})")
                   .param == "questions[0].choices[1]",
           "choice values are unique");
    expect(decisions_refusal(
               R"({"input": [{"role": "system", "content": "s"}],
                   "questions": [{"type": "predicate", "instructions": "i"}]})")
                   .param == "input[0].role",
           "only user messages");
    expect(decisions_refusal(
               R"({"input": "i", "stream": true, "questions": [{"type": "predicate", "instructions": "i"}]})")
                   .param == "stream",
           "unknown fields are refused");
    expect(decisions_refusal(
               R"({"input": "i", "questions": [{"type": "predicate", "instructions": "i",
                   "levels": [{"label": "a"}, {"label": "b"}]}]})")
                   .param == "questions[0]",
           "a predicate takes no levels");
}

void prompts() {
    const DecisionRequest request = parse_systemone_request(RequestJson::parse(kSystemOne));
    const DecisionLabels team     = decision_labels(request.questions[0]);
    const DecisionLabels urgent   = decision_labels(request.questions[1]);
    const DecisionLabels tone     = decision_labels(request.questions[2]);
    expect(team.texts == std::vector<std::string>({"A", "B"}) &&
               team.option == std::vector<std::size_t>({0, 1}) &&
               tone.texts == std::vector<std::string>({"0", "1", "2"}) &&
               tone.option == std::vector<std::size_t>({0, 1, 2}),
           "letters and digits label the options");
    // A choice wider than the letters: fixed-width numbers read a digit at a time.
    DecisionQuestion wide_choice;
    wide_choice.type = DecisionType::Choice;
    wide_choice.options.resize(120);
    const DecisionLabels numbers = decision_labels(wide_choice);
    expect(numbers.width == 3 && numbers.texts.front() == "000" && numbers.texts.back() == "119",
           "a choice of 120 options is labelled 000 to 119");
    const std::vector<DecisionReadNode> nodes = decision_read_nodes(
        numbers, [](const std::string& piece) { return static_cast<ninfer::TokenId>(piece[0]); });
    // The root, the hundreds 0 and 1, then 10 tens under 0 and 2 under 1.
    expect(nodes.size() == 1 + 2 + 12 &&
               nodes.front().next == std::vector<std::string>({"0", "1"}) &&
               nodes[2].prefix == "1" && nodes[2].next == std::vector<std::string>({"0", "1"}) &&
               nodes.back().prefix == "11" && nodes.back().next.size() == 10 &&
               nodes.back().continuation == std::vector<ninfer::TokenId>({'1', '1'}),
           "the label trie's nodes, each continued by its prefix");
    std::vector<ninfer::TokenLogprob> records(nodes.size());
    std::vector<const ninfer::TokenLogprob*> pointers;
    for (std::size_t n = 0; n < nodes.size(); ++n) {
        records[n].top_ids.fill(-1);
        records[n].top_ids[0] = nodes[n].readout[0];
        // Uniform over each node's digits.
        records[n].readout.assign(nodes[n].readout.size(),
                                  static_cast<float>(-std::log(double(nodes[n].readout.size()))));
        pointers.push_back(&records[n]);
    }
    // Make 105 likelier: its path 1, 0, 5.
    records[0].readout = {static_cast<float>(std::log(0.2)), static_cast<float>(std::log(0.8))};
    const DecisionAnswer digits = read_decision(pointers, nodes, numbers);
    expect(digits.exact && digits.probabilities.size() == 120 &&
               close_to(digits.probabilities[105], 0.8 * 0.5 * 0.1) &&
               close_to(digits.probabilities[5], 0.2 * 0.1 * 0.1) &&
               close_to(digits.label_mass, 1.0),
           "a wide label's probability is the product along its digits");
    expect(urgent.texts == std::vector<std::string>({"yes", "Yes", "no", "No"}) &&
               urgent.option == std::vector<std::size_t>({0, 0, 1, 1}),
           "a noul reads yes and no in both cases");
    const GenerationRequest choice = make_decision_read(request, request.questions[0], {32, 33});
    expect(choice.messages.size() == 2 && choice.messages[0].role == ninfer::ChatRole::System &&
               choice.messages[0].content.at(0).text.find("one label only") != std::string::npos &&
               choice.messages[0].cache_boundary_after == CacheBoundary{} &&
               choice.messages[1].role == ninfer::ChatRole::User,
           "the system turn asking for one label, with an explicit cache boundary, then the "
           "user turn");
    const std::vector<ContentPart>& parts = choice.messages[1].content;
    expect(parts.size() == 2 && parts[0].text.rfind("State:\n{\"ticket\"", 0) == 0 &&
               parts[0].text.ends_with("}\n\n") &&
               parts[0].cache_boundary_after == CacheBoundary{} &&
               parts[1].text.rfind("Question: Which team?\n", 0) == 0 &&
               !parts[1].cache_boundary_after,
           "the state ends the first part, with an explicit cache boundary, and the question "
           "starts the second");
    const ninfer::PromptInput input =
        to_prompt_input(choice, resolve_prompt_semantics(choice, ServeOptions{}),
                        [](const ContentPart&) -> ninfer::OwnedMedia {
                            throw std::logic_error("a decision read has no media");
                        });
    const ninfer::PromptCacheMarker after_system{
        .after_message_count = 1,
        .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
    };
    const ninfer::PromptCacheMarker after_state{
        .after_message_count      = 2,
        .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence                 = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 1,
    };
    expect(input.messages.size() == 2 && input.messages[1].parts.size() == 2 &&
               input.context_cache.markers ==
                   std::vector<ninfer::PromptCacheMarker>({after_system, after_state}),
           "the Engine sees shared-prefix markers after the system turn and the state part");
    const std::string text = user_text(choice);
    expect(text.find("State:\n{\"ticket\"") == 0 &&
               text.find("Question: Which team?\nOptions:\nA. billing - payments and refunds\n"
                         "B. shipping\nReply with a single letter (A or B) and nothing else.") !=
                   std::string::npos,
           "the state, the question and the labelled options");
    expect(choice.enable_thinking == false && choice.max_tokens == 1 && choice.logprobs &&
               choice.readout_tokens == std::vector<ninfer::TokenId>({32, 33}) && choice.graft &&
               choice.graft->empty() && choice.sampling.temperature == 0.0 &&
               choice.sampling.presence_penalty == 0.0 && !choice.speculative_drafting,
           "one greedy token with the labels read out, no thinking, no graft, no penalties, no "
           "drafter");
    const GenerationRequest noul = make_decision_read(request, request.questions[1], {1, 2, 3, 4});
    expect(user_text(noul).find(
               "Question: Reply within the hour?\nyes: money is moving\n"
               "Answer with one word, yes or no, in lower case.") != std::string::npos,
           "a noul shows the described answers");
    const GenerationRequest score = make_decision_read(request, request.questions[2], {1, 2, 3});
    expect(user_text(score).find(
               "Levels, lowest first:\n0. calm\n1. annoyed\n2. furious\n"
               "Reply with a single number (0, 1 or 2) and nothing else.") != std::string::npos,
           "a score lists its levels under their numbers");
    const DecisionRequest with_context = parse_systemone_request(RequestJson::parse(
        R"({"state": "s", "instructions": "ctx",
            "questions": {"q": {"type": "noul", "instructions": "i"},
                          "wide": {"type": "choice", "instructions": "i",
                                   "criteria": {"a": null, "b": null, "c": null, "d": null, "e": null}}}})"));
    const GenerationRequest first = make_decision_read(with_context, with_context.questions[0],
                                                       {1, 2, 3, 4});
    const GenerationRequest wide = make_decision_read(with_context, with_context.questions[1],
                                                      {1, 2, 3, 4, 5});
    expect(first.messages[1].content.at(0).text == "ctx\n\nState:\ns\n\n" &&
               wide.messages[1].content.at(0).text == first.messages[1].content.at(0).text,
           "context goes ahead of the state, and every question shares that part");
    expect(user_text(wide).find("Reply with a single letter (A to E) and nothing else.") !=
               std::string::npos,
           "many options are spanned");
}

void reading() {
    const DecisionLabels three{.texts = {"A", "B", "C"}, .option = {0, 1, 2}};
    const std::vector<ninfer::TokenId> labels{7, 8, 9};
    // Exact readout: 0.6, 0.2 and 0.1 of the vocabulary on the labels.
    const DecisionAnswer exact =
        read_decision(read({std::log(0.6), std::log(0.2), std::log(0.1)}, {{7, 0.6}, {8, 0.2}}),
                      three, labels);
    expect(exact.exact && close_to(exact.label_mass, 0.9) &&
               close_to(exact.probabilities[0], 0.6 / 0.9) &&
               close_to(exact.probabilities[2], 0.1 / 0.9) && exact.argmax_is_label,
           "softmax over the exact label log probabilities, their total as the label mass");
    const DecisionAnswer other = read_decision(
        read({std::log(0.3), std::log(0.1), std::log(0.1)}, {{5, 0.5}, {7, 0.3}}), three, labels);
    expect(!other.argmax_is_label && close_to(other.label_mass, 0.5),
           "the model's favourite token outside the labels is reported");
    // No readout: the top alternatives, a missing label as likely as the least likely of them.
    const DecisionAnswer top =
        read_decision(read({}, {{7, 0.5}, {4, 0.2}, {9, 0.1}}), three, labels);
    expect(!top.exact && close_to(top.label_mass, 0.7) && close_to(top.probabilities[0], 0.5 / 0.7) &&
               close_to(top.probabilities[1], 0.1 / 0.7) && close_to(top.probabilities[2], 0.1 / 0.7),
           "without a readout the labels are read from the top alternatives");
    const DecisionAnswer short_readout = read_decision(read({0.0, 0.0}, {{7, 0.5}}), three, labels);
    expect(!short_readout.exact, "a readout of another size is not this question's");
    // A noul's two cases add up per option.
    const DecisionLabels noul{.texts = {"yes", "Yes", "no", "No"}, .option = {0, 0, 1, 1}};
    const std::vector<ninfer::TokenId> noul_tokens{11, 12, 13, 14};
    const DecisionAnswer merged = read_decision(
        read({std::log(0.5), std::log(0.2), std::log(0.1), std::log(0.1)}, {{7, 0.5}}), noul,
        noul_tokens);
    expect(merged.exact && merged.probabilities.size() == 2 &&
               close_to(merged.probabilities[0], 0.7 / 0.9) &&
               close_to(merged.probabilities[1], 0.2 / 0.9) && close_to(merged.label_mass, 0.9),
           "yes and Yes answer yes, no and No answer no");
    // Runoff: options judged alone in text order, the consideration set, the listings' merge.
    DecisionQuestion choice;
    choice.type = DecisionType::Choice;
    for (const char* text : {"delta", "alpha", "charlie", "bravo"}) {
        choice.options.push_back({.value = RequestJson(text), .text = text, .description = {}});
    }
    expect(runoff_order(choice) == std::vector<std::size_t>({1, 3, 2, 0}),
           "runoff reads the options in text order");
    // Odds of yes over no, in text order (alpha, bravo, charlie, delta): 8, 1, 1/8, 1/1000.
    const auto judged_answer = [](double yes, double no) {
        DecisionAnswer answer;
        answer.probabilities = {yes / (yes + no), no / (yes + no)};
        return answer;
    };
    const std::vector<DecisionAnswer> judged{judged_answer(8, 1), judged_answer(1, 1),
                                             judged_answer(1, 8), judged_answer(1, 1000)};
    const RunoffCandidates candidates = runoff_consider(choice, judged);
    const double total                = 8.0 + 1.0 + 0.125 + 0.001;
    expect(candidates.ranking == std::vector<std::size_t>({1, 3, 2}) &&
               close_to(candidates.pointwise[1], 8.0 / total) &&
               close_to(candidates.pointwise[0], 0.001 / total),
           "the consideration set keeps the top three by pointwise probability");
    const DecisionQuestion shifted = runoff_listing(choice, candidates, 1);
    expect(shifted.options.size() == 3 && shifted.options[0].text == "bravo" &&
               shifted.options[2].text == "alpha",
           "a listing rotates the canonical ranking");
    // Every listing puts charlie first at 0.5 and splits the rest evenly: charlie wins.
    std::vector<DecisionAnswer> listings(3);
    for (std::size_t shift = 0; shift < 3; ++shift) {
        listings[shift].probabilities.assign(3, 0.25);
        listings[shift].label_mass                         = 0.9;
        listings[shift].probabilities[(2 + 3 - shift) % 3] = 0.5;
    }
    const DecisionAnswer chosen = runoff_choose(choice, candidates, listings);
    const double set            = (8.0 + 1.0 + 0.125) / total;
    expect(close_to(chosen.probabilities[2], set * 0.5 / 1.0) &&
               close_to(chosen.probabilities[0], 0.001 / total) && close_to(chosen.label_mass, 0.9),
           "the candidates share the set's probability by their listings' geometric mean");
    bool refused = false;
    try {
        ninfer::TokenLogprob empty;
        empty.top_ids.fill(-1);
        (void)read_decision(empty, three, labels);
    } catch (const std::runtime_error&) { refused = true; }
    expect(refused, "a read without an answer token is an error, not an answer");
}

void responses() {
    const DecisionRequest request = parse_systemone_request(RequestJson::parse(kSystemOne));
    std::vector<DecisionAnswer> answers(3);
    answers[0] = {.probabilities = {0.25, 0.75}, .label_mass = 0.8, .argmax_is_label = true};
    answers[1] = {.probabilities = {0.9, 0.1}, .label_mass = 0.5, .argmax_is_label = false};
    answers[2] = {.probabilities = {0.2, 0.5, 0.3},
                  .label_mass    = 1.0,
                  .argmax_is_label = true,
                  .exact         = false};
    const Json body = Json::parse(make_decision_response(request, answers, 120, 3));
    expect(body["object"] == "structured_decision" && body["model"] == "m" &&
               body["id"].get<std::string>().rfind("decision-", 0) == 0 &&
               body["usage"]["input_tokens"] == 120 && body["usage"]["output_tokens"] == 3,
           "a structured decision with usage");
    const Json& team = body["answers"]["team"];
    expect(team["type"] == "choice" && team["choice"] == "shipping" &&
               close_to(team["probabilities"]["billing"].get<double>(), 0.25) &&
               close_to(team["confidence"].get<double>(), 0.6),
           "the choice, its probabilities and the winner's share of the vocabulary");
    const Json& urgent = body["answers"]["urgent"];
    expect(urgent["type"] == "noul" && close_to(urgent["noul"].get<double>(), 0.9) &&
               close_to(urgent["probabilities"]["no"].get<double>(), 0.1) &&
               close_to(urgent["confidence"].get<double>(), 0.45),
           "a noul's probability of yes");
    const Json& tone = body["answers"]["tone"];
    expect(tone["type"] == "score" && close_to(tone["score"].get<double>(), 1.1) &&
               tone["legend"]["2"] == "furious" &&
               close_to(tone["probabilities"]["1"].get<double>(), 0.5) &&
               close_to(tone["confidence"].get<double>(), 0.5),
           "a score's expected level, legend and probabilities by level index");
    expect(close_to(body["diagnostics"]["team"]["label_mass"].get<double>(), 0.8) &&
               body["diagnostics"]["urgent"]["argmax_is_label"] == false &&
               body["diagnostics"]["team"]["readout"] == "exact" &&
               body["diagnostics"]["tone"]["readout"] == "top_k",
           "diagnostics per question");

    const DecisionRequest openai = parse_decisions_request(RequestJson::parse(kOpenAI));
    std::vector<DecisionAnswer> openai_answers(3);
    openai_answers[0] = {.probabilities = {0.3, 0.7}, .label_mass = 1.0};
    openai_answers[1] = {.probabilities = {0.4, 0.6}, .label_mass = 0.5};
    openai_answers[2] = {.probabilities = {0.8, 0.2}, .label_mass = 1.0};
    const Json decision = Json::parse(make_decision_response(openai, openai_answers, 40, 3));
    expect(decision["object"] == "decision" && decision["answers"].is_array() &&
               decision["answers"].size() == 3 && decision["usage"]["total_tokens"] == 43,
           "an OpenAI decision with answers in question order");
    const Json& damaged = decision["answers"][0];
    expect(damaged["type"] == "predicate" && damaged["name"] == "damaged" &&
               close_to(damaged["probability"].get<double>(), 0.3),
           "a predicate's probability");
    const Json& department = decision["answers"][1];
    expect(department["type"] == "choice" && department["choice"] == true &&
               department["probabilities"][0]["value"] == "returns" &&
               close_to(department["probabilities"][1]["probability"].get<double>(), 0.6) &&
               close_to(department["confidence"].get<double>(), 0.3),
           "a choice echoes its values with their types");
    const Json& severity = decision["answers"][2];
    expect(severity["type"] == "score" && severity["name"].is_null() &&
               close_to(severity["score"].get<double>(), 0.2) &&
               severity["probabilities"][1]["label"] == "broken" &&
               severity["probabilities"][1]["value"] == 1,
           "an unnamed score with labelled, indexed levels");
}

} // namespace

// Images and video as data URLs: where each endpoint takes them, their order, and their place in
// every read, ahead of the state inside the part its cache boundary closes.
void media() {
    const DecisionRequest system_one = parse_systemone_request(RequestJson::parse(R"({
        "state": [{"role": "user", "content": [
            {"type": "text", "text": "Receipt attached."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Cw=="}}]}],
        "images": ["data:image/png;base64,Bw=="],
        "files": ["data:image/jpeg;base64,AQ==", "data:video/mp4;base64,Aw=="],
        "questions": {"q": {"type": "noul", "instructions": "Is it paid?"}}})"));
    expect(system_one.media.size() == 4 &&
               system_one.media[0].source.value == "data:image/jpeg;base64,AQ==" &&
               system_one.media[1].kind == ContentKind::Video &&
               system_one.media[2].source.value == "data:image/png;base64,Bw==" &&
               system_one.media[3].source.value == "data:image/png;base64,Cw==" &&
               system_one.media[3].kind == ContentKind::Image,
           "files, then images, then the state's image_url parts");
    expect(system_one.state.find("Receipt attached.") != std::string::npos &&
               system_one.state.find("image_url") == std::string::npos,
           "a state's image parts leave its text");

    const GenerationRequest read =
        make_decision_read(system_one, system_one.questions[0], {1, 2, 3, 4});
    const std::vector<ContentPart>& parts = read.messages.at(1).content;
    expect(parts.size() == 6 && parts[0].kind == ContentKind::Image &&
               parts[1].kind == ContentKind::Video && parts[3].kind == ContentKind::Image &&
               !parts[3].cache_boundary_after && parts[4].kind == ContentKind::Text &&
               parts[4].text.starts_with("State:\n") && parts[4].cache_boundary_after &&
               parts[5].text.starts_with("Question: Is it paid?"),
           "the media precede the state, inside the part its boundary closes");

    const DecisionRequest alone = parse_systemone_request(RequestJson::parse(
        R"({"images": ["data:image/png;base64,Bw=="],
            "questions": {"q": {"type": "noul", "instructions": "Is there a table?"}}})"));
    const GenerationRequest image_only =
        make_decision_read(alone, alone.questions[0], {1, 2, 3, 4});
    expect(user_text(image_only).find("State:") == std::string::npos &&
               image_only.messages.at(1).content.front().kind == ContentKind::Image,
           "a request about its images alone has no state block");

    const DecisionRequest text = parse_systemone_request(RequestJson::parse(kSystemOne));
    expect(
        text.media.empty() &&
            make_decision_read(text, text.questions[0], {32, 33}).messages.at(1).content.size() ==
                2,
        "a text request keeps its two parts");

    expect(systemone_refusal(R"({"state": "s", "images": ["https://example.com/a.png"],
                                 "questions": {"q": {"type": "noul", "instructions": "i"}}})")
                   .param == "images[0]",
           "System One takes data URLs only");
    expect(systemone_refusal(R"({"state": "s", "files": ["data:audio/wav;base64,AA=="],
                                 "questions": {"q": {"type": "noul", "instructions": "i"}}})")
                   .status == 501,
           "an audio file is not implemented");
    expect(systemone_refusal(R"({"state": "s", "videos": ["data:video/mp4;base64,AA=="],
                                 "questions": {"q": {"type": "noul", "instructions": "i"}}})")
                   .status == 501,
           "System One's videos field is not implemented");
    expect(
        systemone_refusal(R"({"questions": {"q": {"type": "noul", "instructions": "i"}}})").param ==
            "state",
        "without media or a conversation the state is required");

    const DecisionRequest openai = parse_decisions_request(RequestJson::parse(R"({
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Which form is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,Bw==", "detail": "low"}]},
            {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,Cw=="}]}],
        "questions": [{"type": "predicate", "instructions": "Is it signed?"}]})"));
    expect(openai.media.size() == 2 && openai.media[0].image_detail == ninfer::ImageDetail::Low &&
               openai.media[1].source.value == "data:image/png;base64,Cw==" &&
               openai.state == "Which form is this?",
           "input_image parts in order; a message of images alone adds no text");
    expect(decisions_refusal(R"({"input": [{"role": "user", "content": [
                                   {"type": "input_image", "image_url": "https://example.com/a.png"}]}],
                                 "questions": [{"type": "predicate", "instructions": "i"}]})")
                   .param == "input[0].content[0].image_url",
           "OpenAI's input_image takes a data URL");
    expect(decisions_refusal(R"({"input": [{"role": "user", "content": [
                                   {"type": "input_image", "file_id": "file-1"}]}],
                                 "questions": [{"type": "predicate", "instructions": "i"}]})")
                   .param == "input[0].content[0].file_id",
           "input_image.file_id needs a Files API");
    expect(decisions_refusal(R"({"input": "i", "images": ["data:image/png;base64,Bw=="],
                                 "questions": [{"type": "predicate", "instructions": "i"}]})")
                   .status == 501,
           "OpenAI's request has no images field");
}

int main() {
    try {
        systemone_parsing();
        decisions_parsing();
        prompts();
        media();
        reading();
        responses();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "systemone: ok\n";
    return 0;
}
