#pragma once

// Typed decisions: POST /v1/systemone (TypeSafe's System One API, as llama.cpp, vLLM and Ollama
// serve it) and POST /v1/decisions (OpenAI's Decisions API), answered by the served chat model
// through a label readout. Each question is one prompt -- a fixed system turn asking for one
// label, then the state, the question and its options under one-token labels (A.. for a choice,
// yes/no for a noul, 0.. for a score), thinking off -- and the answer is read from the first reply
// token's exact log probabilities of those labels (a noul's yes and no in both cases): softmax
// over the options gives the probabilities, their total over the vocabulary is the label mass,
// and the winner's probability over the whole vocabulary is the confidence.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

// OpenAI's bounds on questions and choices; vLLM's on levels. A choice of up to 26 options is
// labelled with letters, a wider one with fixed-width numbers read a digit at a time.
inline constexpr std::size_t kMaximumDecisionQuestions = 200;
inline constexpr std::size_t kMaximumDecisionOptions   = 255;
inline constexpr std::size_t kMaximumLetterOptions     = 26;
inline constexpr std::size_t kMaximumDecisionLevels    = 10;
inline constexpr std::size_t kMaximumDecisionIdLength  = 100;
// Each read's request holds its own copy of the media's data URLs: their size times the reads.
inline constexpr std::size_t kMaximumDecisionMediaCopies = std::size_t{512} << 20;

enum class DecisionDialect { SystemOne, OpenAI };

// Where a read's prompt puts the questions: after the state, each read holding its own question
// and options, or every question with its options ahead of the state, shared by every read and by
// requests that ask the same questions, each read then naming the one to answer.
enum class DecisionLayout { StateFirst, SchemaFirst };

// System One's choice / noul / score; OpenAI's predicate is a noul.
enum class DecisionType { Choice, Noul, Score };

struct DecisionOption {
    // The option's wire value: a System One option name or score level, or an OpenAI choice value
    // (a string or a boolean, echoed with its type) or level label.
    RequestJson value;
    // The value as the prompt shows it.
    std::string text;
    std::optional<std::string> description;
};

struct DecisionQuestion {
    // System One's question id, or OpenAI's optional name (`named` false echoes null).
    std::string id;
    bool named        = true;
    DecisionType type = DecisionType::Choice;
    std::string instructions;
    // A choice's options or a score's levels, lowest first; a noul's optional yes and no
    // descriptions as two entries (text "yes" and "no").
    std::vector<DecisionOption> options;
};

struct DecisionRequest {
    DecisionDialect dialect = DecisionDialect::SystemOne;
    std::string model;
    // The state as the prompt shows it: a string as given, other JSON as its compact text.
    std::string state;
    // Optional context placed ahead of the state (System One's `instructions`).
    std::string instructions;
    // Images and videos as data URLs, placed ahead of the state in every read: System One's
    // `files` and its alias `images` in that order, then the image_url parts of a state made of
    // chat messages (removed from the state's text); OpenAI's input_image parts in input order.
    std::vector<ContentPart> media;
    std::vector<DecisionQuestion> questions;
    // A live conversation the questions are asked about: an OpenAI chat completion request body
    // (`messages`, and `tools` and `chat_template_kwargs` as the agent sent them). Its messages
    // render as that request's prompt does, so a session the context cache holds for it serves
    // every question, which then costs only its own tokens. `state` is optional with it.
    std::optional<RequestJson> conversation;
    DecisionLayout layout = DecisionLayout::StateFirst;
    // Choice questions answered by Runoff (arXiv 2610.18233), whose answer does not depend on the
    // options' order: each option is first judged alone (a yes or no read), the few most likely
    // are kept, and those are compared in listings where each takes every position.
    bool runoff = false;
    // An answer whose margin lies below this is read again alone, outside any shared pass, so
    // near ties do not depend on what else ran with them; 0 reads every answer once.
    double recheck_margin = 0.0;
};

// Strict: unknown fields are refused, as vLLM refuses them. Media come as data URLs (llama.cpp's
// System One contract); audio and System One's `audio` and `videos` fields are refused as not
// implemented, other media sources as invalid.
[[nodiscard]] DecisionRequest parse_systemone_request(const RequestJson& body);
[[nodiscard]] DecisionRequest parse_decisions_request(const RequestJson& body);

struct DecisionLabels {
    // The label texts the read looks up: A..Z for a choice of up to 26 options and 00.. or 000..
    // for a wider one, yes, Yes, no and No for a noul, 0..9 for a score.
    std::vector<std::string> texts;
    // The option each label answers for, aligned with texts.
    std::vector<std::size_t> option;
    // Characters per label, each one token: 1, or a wider choice's digits.
    std::size_t width = 1;
};

[[nodiscard]] DecisionLabels decision_labels(const DecisionQuestion& question);

// One read of a question: the label characters already given after its prompt, as tokens, and
// the tokens that may come next, each with the character it adds. A question of one-token labels
// is one read; wider labels are read as the nodes of their trie, a digit per read.
struct DecisionReadNode {
    std::string prefix;
    std::vector<ninfer::TokenId> continuation;
    std::vector<ninfer::TokenId> readout;
    std::vector<std::string> next;
};

// The reads of a question, the root first. `token` maps one label character to its token.
[[nodiscard]] std::vector<DecisionReadNode>
decision_read_nodes(const DecisionLabels& labels,
                    const std::function<ninfer::TokenId(const std::string&)>& token);

// The request that reads `question`: the system turn, then one user turn with the state, the
// question and its options under the labels, thinking off, one output token without the
// speculative drafter, with the labels' token ids (aligned with decision_labels(question).texts)
// read out exactly. Explicit cache boundaries follow the system turn and the user turn's part
// that holds the context and the state, which the request's media precede; the last part holds
// the question.
// With `conversation` (the request's parsed conversation) its messages and tools come first, an
// explicit cache boundary after its last message, and the decision instructions open the user turn
// that holds the state and the question.
[[nodiscard]] GenerationRequest make_decision_read(const DecisionRequest& request,
                                                   const DecisionQuestion& question,
                                                   std::vector<ninfer::TokenId> label_tokens,
                                                   const GenerationRequest* conversation = nullptr);

struct DecisionAnswer {
    // Over the options, in order; sums to 1.
    std::vector<double> probabilities;
    // The labels' total probability over the whole vocabulary.
    double label_mass = 0.0;
    // The model's most likely token is one of the labels.
    bool argmax_is_label = false;
    // Read from the exact readout; false when the core has none and the labels were read from the
    // top alternatives (a label outside them counts as likely as the least likely alternative).
    bool exact = true;
    // The most likely option's probability less the second's: a client escalates an answer below
    // its own threshold (to a generation with reasoning, for example).
    double margin = 0.0;
};

// The answer of one read from its first token's record, `label_tokens` aligned with
// `labels.texts`, or std::runtime_error when the read produced no answer token.
[[nodiscard]] DecisionAnswer read_decision(const ninfer::TokenLogprob& first,
                                           const DecisionLabels& labels,
                                           std::span<const ninfer::TokenId> label_tokens);

// The answer of a question read as `nodes` (decision_read_nodes), `records` aligned with them: a
// label's log probability is the sum of its characters' along its path.
[[nodiscard]] DecisionAnswer read_decision(std::span<const ninfer::TokenLogprob* const> records,
                                           std::span<const DecisionReadNode> nodes,
                                           const DecisionLabels& labels);

// Runoff's first step, Consider: the request that judges `option` of `question` alone, a noul
// read whose labels are decision_labels of a noul.
[[nodiscard]] GenerationRequest
make_runoff_pointwise(const DecisionRequest& request, const DecisionQuestion& question,
                      const DecisionOption& option, std::vector<ninfer::TokenId> label_tokens,
                      const GenerationRequest* conversation = nullptr);

struct RunoffCandidates {
    // The pointwise distribution over the question's options: each option's odds of yes over no,
    // normalized.
    std::vector<double> pointwise;
    // The consideration set in its canonical ranking: by decreasing pointwise probability, equal
    // ones by option text. The smallest top set holding `alpha` of the probability and at least
    // `minimum` options (ties kept together), or every option.
    std::vector<std::size_t> ranking;
};

// The options' order Runoff reads them in: by option text, so the reads do not depend on the
// request's order.
[[nodiscard]] std::vector<std::size_t> runoff_order(const DecisionQuestion& question);

// `judged` holds the pointwise answers in runoff_order(question).
[[nodiscard]] RunoffCandidates runoff_consider(const DecisionQuestion& question,
                                               std::span<const DecisionAnswer> judged,
                                               double alpha = 0.99, std::size_t minimum = 3);

// Choose's listing `shift`: the question with only the candidates, the ranking rotated by `shift`.
[[nodiscard]] DecisionQuestion runoff_listing(const DecisionQuestion& question,
                                              const RunoffCandidates& candidates,
                                              std::size_t shift);

// The answer from the listings' answers, one per shift in order: the candidates share the
// pointwise probability of the set by the geometric mean of their listing probabilities, and
// the other options keep their pointwise probability.
[[nodiscard]] DecisionAnswer runoff_choose(const DecisionQuestion& question,
                                           const RunoffCandidates& candidates,
                                           std::span<const DecisionAnswer> listings);

// One response, the answers aligned with the request's questions; `input_tokens` and
// `output_tokens` summed over the reads.
[[nodiscard]] std::string make_decision_response(const DecisionRequest& request,
                                                 std::span<const DecisionAnswer> answers,
                                                 int input_tokens, int output_tokens);

} // namespace ninfer::serve
