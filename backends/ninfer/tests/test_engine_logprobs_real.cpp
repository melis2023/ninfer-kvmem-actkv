#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// Token log probabilities on a real model through the public Engine, for any model family:
// NINFER_LOGPROBS_ARTIFACT names the artifact, NINFER_LOGPROBS_NGRAM_TABLE a Qwen3.8-Flash-Next
// table artifact, and NINFER_LOGPROBS_SPECULATIVE=mtp|dflash2 verifies the tokens in speculative
// rounds.
namespace {

int g_failures = 0;

void check(bool condition, std::string_view what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++g_failures;
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk        = 1024;
    options.max_concurrency      = 3;
    options.max_pending_requests = 3;
    options.structured_output    = true;
    if (const char* table = std::getenv("NINFER_LOGPROBS_NGRAM_TABLE"); table && *table) {
        options.ngram_table.path = table;
    }
    if (const char* spec = std::getenv("NINFER_LOGPROBS_SPECULATIVE"); spec && *spec) {
        const std::string_view backend(spec);
        options.speculative.backend      = backend == "mtp" ? ninfer::SpeculativeBackend::Mtp
                                                            : ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = backend == "mtp" ? 3 : 15;
    }
    return options;
}

ninfer::PromptInput prompt(std::string text) {
    ninfer::PromptInput input;
    input.messages.push_back(ninfer::ChatMessage{
        .role  = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions request(bool logprobs, std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.execution.logprobs                = logprobs;
    return options;
}

std::string joined(const std::vector<ninfer::TokenLogprob>& records) {
    std::string out;
    for (const ninfer::TokenLogprob& record : records) { out += record.bytes; }
    return out;
}

// What every record must satisfy: a finite log probability (the sampler keeps at most its 20 most
// likely tokens, so a drawn token always has one), alternatives in descending order with distinct
// ids, and, under greedy decoding, the token is its position's most likely one.
void check_records(const std::vector<ninfer::TokenLogprob>& records, bool greedy,
                   std::string_view label) {
    std::size_t bad = 0;
    for (const ninfer::TokenLogprob& record : records) {
        bool ok = std::isfinite(record.logprob) && record.logprob <= 0.0F &&
                  record.logprob != ninfer::kLogprobSentinel && !record.bytes.empty();
        std::set<ninfer::TokenId> ids;
        for (std::size_t k = 0; k < ninfer::kMaximumTokenLogprobs; ++k) {
            if (record.top_ids[k] < 0) { break; }
            ok = ok && ids.insert(record.top_ids[k]).second && std::isfinite(record.top_values[k]);
            if (k > 0) { ok = ok && record.top_values[k] <= record.top_values[k - 1]; }
        }
        if (greedy) {
            ok = ok && record.top_ids[0] == record.id && record.top_values[0] == record.logprob &&
                 record.top_bytes[0] == record.bytes;
        }
        bad += ok ? 0U : 1U;
    }
    check(bad == 0, std::string(label) + ": " + std::to_string(bad) + " of " +
                        std::to_string(records.size()) + " records break the invariants");
}

class Collector final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel != ninfer::OutputChannel::Content) { return; }
        text += delta.text;
        records.insert(records.end(), delta.logprobs.begin(), delta.logprobs.end());
        // A record arrives with its token's first byte: what the records spell so far agrees with
        // the text so far and runs past it by less than one token.
        const std::string spelled = joined(records);
        const std::size_t common  = std::min(spelled.size(), text.size());
        ordered                   = ordered && spelled.compare(0, common, text, 0, common) == 0 &&
                  (records.empty() || spelled.size() < text.size() + records.back().bytes.size());
    }
    std::string text;
    std::vector<ninfer::TokenLogprob> records;
    bool ordered = true;
};

int run() {
    const char* artifact = std::getenv("NINFER_LOGPROBS_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    ninfer::Engine engine(engine_options(artifact));
    const std::string question =
        "Write three sentences about the history of the printing press, plainly.";

    // Greedy: the records spell the content exactly and repeat run to run; gathering them changes
    // nothing that is generated.
    const ninfer::GenerationResult greedy = engine.generate(engine.prepare(prompt(question)),
                                                            request(true, 96));
    std::cout << "greedy: " << greedy.generated_token_ids.size() << " tokens, "
              << greedy.content_logprobs.size() << " records, speculative rounds "
              << greedy.speculative.rounds << '\n';
    check(!greedy.content_logprobs.empty(), "greedy request produced records");
    check(greedy.content_logprobs.size() <= greedy.generated_token_ids.size(),
          "no more records than generated tokens");
    check(joined(greedy.content_logprobs) == greedy.content,
          "the records' bytes spell the content");
    check_records(greedy.content_logprobs, true, "greedy");
    const ninfer::GenerationResult again = engine.generate(engine.prepare(prompt(question)),
                                                           request(true, 96));
    bool same = again.content_logprobs.size() == greedy.content_logprobs.size();
    for (std::size_t i = 0; same && i < greedy.content_logprobs.size(); ++i) {
        same = again.content_logprobs[i].id == greedy.content_logprobs[i].id &&
               std::abs(again.content_logprobs[i].logprob - greedy.content_logprobs[i].logprob) <
                   1e-4F;
    }
    check(same, "greedy records repeat run to run");
    const ninfer::GenerationResult plain = engine.generate(engine.prepare(prompt(question)),
                                                           request(false, 96));
    check(plain.content_logprobs.empty(), "a request without logprobs gathers none");
    check(plain.generated_token_ids == greedy.generated_token_ids,
          "gathering logprobs does not change the generated tokens");

    // Sampling with penalties: the drawn token is always among the reported alternatives.
    ninfer::RequestOptions sampled             = request(true, 96);
    sampled.execution.sampling.temperature      = 0.8F;
    sampled.execution.sampling.top_p            = 0.95F;
    sampled.execution.sampling.top_k            = 20;
    sampled.execution.sampling.presence_penalty = 1.0F;
    sampled.execution.sampling.seed             = 7;
    const ninfer::GenerationResult drawn =
        engine.generate(engine.prepare(prompt(question)), sampled);
    check_records(drawn.content_logprobs, false, "sampled");
    check(joined(drawn.content_logprobs) == drawn.content, "sampled records spell the content");

    // Streaming carries the same records, each no earlier than its text.
    Collector collector;
    auto handle = engine.submit(engine.prepare(prompt(question)), request(true, 96),
                                ninfer::OutputConsumerMode::Streaming);
    const ninfer::GenerationResult streamed = handle.wait(&collector);
    check(collector.ordered, "a streamed record arrived before its text");
    check(joined(collector.records) == joined(streamed.content_logprobs) &&
              collector.records.size() == streamed.content_logprobs.size(),
          "streamed records match the aggregate ones");

    // A stop string drops the records of the tokens it cuts.
    if (greedy.content.size() > 40) {
        ninfer::RequestOptions stopped = request(true, 96);
        stopped.stop.strings.push_back(ninfer::StopString{.text = greedy.content.substr(30, 4)});
        const ninfer::GenerationResult cut =
            engine.generate(engine.prepare(prompt(question)), stopped);
        const std::string spelled = joined(cut.content_logprobs);
        check(cut.finish_reason == ninfer::FinishReason::StopString, "the stop string matched");
        check(spelled.starts_with(cut.content) &&
                  (cut.content_logprobs.empty() ||
                   spelled.size() < cut.content.size() + cut.content_logprobs.back().bytes.size()),
              "only the token the cut runs through keeps its record past the cut");
    }

    // Requests decoded in one batch each read their own row: the records of two of them spell their
    // own content, and a third beside them that does not ask gathers none.
    auto left  = engine.submit(engine.prepare(prompt(question)), request(true, 64));
    auto right = engine.submit(engine.prepare(prompt("Count from one to twenty in words.")),
                               request(true, 64));
    auto quiet = engine.submit(engine.prepare(prompt("Name the planets of the solar system.")),
                               request(false, 64));
    const ninfer::GenerationResult left_result  = left.wait();
    const ninfer::GenerationResult right_result = right.wait();
    const ninfer::GenerationResult quiet_result = quiet.wait();
    check_records(left_result.content_logprobs, true, "batched left");
    check_records(right_result.content_logprobs, true, "batched right");
    check(!left_result.content_logprobs.empty() && !right_result.content_logprobs.empty() &&
              joined(left_result.content_logprobs) == left_result.content &&
              joined(right_result.content_logprobs) == right_result.content,
          "batched records spell their own request's content");
    check(quiet_result.content_logprobs.empty() && !quiet_result.content.empty(),
          "a batched request without logprobs gathers none");

    // Under a JSON grammar the mask removes tokens from the distribution: every alternative for the
    // first token opens the object, after optional whitespace.
    ninfer::RequestOptions json           = request(true, 48);
    json.execution.structured_output.kind = ninfer::StructuredOutputKind::JsonObject;
    const ninfer::GenerationResult object =
        engine.generate(engine.prepare(prompt("Name one color.")), json);
    std::cout << "json: " << object.content << '\n';
    check_records(object.content_logprobs, true, "json");
    bool opens = !object.content_logprobs.empty();
    for (std::size_t k = 0; opens && k < ninfer::kMaximumTokenLogprobs; ++k) {
        const ninfer::TokenLogprob& first = object.content_logprobs.front();
        if (first.top_ids[k] < 0) { break; }
        const std::string& bytes = first.top_bytes[k];
        const std::size_t text   = bytes.find_first_not_of(" \t\r\n");
        opens                    = text == std::string::npos || bytes[text] == '{';
    }
    check(opens, "the grammar's mask reached the gather");
    return g_failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
