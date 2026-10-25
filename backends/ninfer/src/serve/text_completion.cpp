#include "serve/text_completion.h"

#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"
#include "serve/structured_output.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json    = RequestJson;
using OutJson = nlohmann::json;

// llama.cpp sets no limit on stop strings, and OpenAI's four would refuse the lists text-completion
// clients send. Bounded all the same: every stop string is scanned on every token.
constexpr std::size_t kMaximumStopStrings = 32;

bool present(const Json& body, const char* key) {
    return body.contains(key) && !body.at(key).is_null();
}

ninfer::TokenId prompt_token(const Json& value) {
    const bool in_range =
        value.is_number_unsigned()
            ? value.get<std::uint64_t>() <=
                  static_cast<std::uint64_t>(std::numeric_limits<ninfer::TokenId>::max())
            : value.get<std::int64_t>() >= 0 &&
                  value.get<std::int64_t>() <= std::numeric_limits<ninfer::TokenId>::max();
    if (!in_range) {
        bad_request("prompt token ids must be nonnegative 32-bit integers", "prompt");
    }
    return static_cast<ninfer::TokenId>(value.get<std::int64_t>());
}

// One prompt: text, or an array of token ids, or token ids and text mixed in one array.
std::vector<RawPromptPiece> prompt_pieces(const Json& prompt) {
    std::vector<RawPromptPiece> pieces;
    if (prompt.is_string()) {
        pieces.push_back(RawPromptPiece{.text = prompt.get<std::string>()});
        return pieces;
    }
    pieces.reserve(prompt.size());
    for (const Json& item : prompt) {
        if (item.is_string()) {
            pieces.push_back(RawPromptPiece{.text = item.get<std::string>()});
        } else if (item.is_number_integer()) {
            pieces.push_back(RawPromptPiece{.token = prompt_token(item)});
        } else {
            bad_request(
                "a prompt is a string, an array of token ids, or token ids and strings mixed",
                "prompt");
        }
    }
    return pieces;
}

// llama.cpp's prompt forms: a string; token ids; token ids and strings mixed; or an array of such
// prompts. An array of several prompts would ask for several completions, which one request does
// not give, so it must hold exactly one.
std::vector<RawPromptPiece> parse_prompt(const Json& body) {
    if (!present(body, "prompt")) { bad_request("prompt is required", "prompt"); }
    const Json& prompt = body.at("prompt");
    if (prompt.is_object()) {
        bad_request("multimodal prompt objects are not supported; send media to "
                    "/v1/chat/completions",
                    "prompt", "multimodal_prompt_not_supported");
    }
    if (!prompt.is_string() && !prompt.is_array()) {
        bad_request("prompt must be a string or an array", "prompt");
    }
    const Json* single = &prompt;
    if (prompt.is_array()) {
        const bool has_number  = std::any_of(prompt.begin(), prompt.end(),
                                             [](const Json& item) { return item.is_number(); });
        const bool scalar_only = std::all_of(prompt.begin(), prompt.end(), [](const Json& item) {
            return item.is_number() || item.is_string();
        });
        if (!(scalar_only && (has_number || prompt.empty()))) {
            if (prompt.size() != 1) {
                bad_request("prompt holds several prompts, while NInfer completes one prompt per "
                            "request",
                            "prompt", "multiple_prompts_not_supported");
            }
            single = &prompt.front();
            if (!single->is_string() && !single->is_array()) {
                bad_request("a prompt is a string, an array of token ids, or token ids and strings "
                            "mixed",
                            "prompt");
            }
        }
    }
    std::vector<RawPromptPiece> pieces = prompt_pieces(*single);
    const bool empty = std::all_of(pieces.begin(), pieces.end(), [](const RawPromptPiece& piece) {
        return !piece.token && piece.text.empty();
    });
    if (empty) { bad_request("prompt must not be empty", "prompt"); }
    return pieces;
}

std::optional<std::uint64_t> parse_seed(const Json& body) {
    if (!present(body, "seed")) { return std::nullopt; }
    const Json& value = body.at("seed");
    if (!value.is_number_integer()) { bad_request("seed must be an integer", "seed"); }
    if (value.is_number_unsigned()) { return value.get<std::uint64_t>(); }
    // llama.cpp's -1, and any negative seed, asks for a random one.
    const std::int64_t seed = value.get<std::int64_t>();
    if (seed < 0) { return std::nullopt; }
    return static_cast<std::uint64_t>(seed);
}

void parse_sampling(const Json& body, GenerationRequest& output) {
    SamplingParams& sampling = output.sampling;
    sampling.temperature     = optional_number(body, "temperature");
    // llama.cpp samples greedily below zero.
    if (sampling.temperature && *sampling.temperature < 0.0) { sampling.temperature = 0.0; }
    sampling.top_p             = optional_number(body, "top_p");
    sampling.min_p             = optional_number(body, "min_p");
    sampling.presence_penalty  = optional_number(body, "presence_penalty");
    sampling.frequency_penalty = optional_number(body, "frequency_penalty");
    sampling.seed              = parse_seed(body);
    // llama.cpp disables the cut at top_k <= 0; zero selects NInfer's widest candidate set.
    if (const std::optional<int> top_k = optional_int(body, "top_k")) {
        sampling.top_k = std::max(0, *top_k);
    }
}

// llama.cpp sampler and stopping controls NInfer does not have. The value that leaves generation
// unchanged is accepted, since clients send the defaults; any other is refused, not ignored.
void require_neutral_controls(const Json& body) {
    struct Control {
        const char* key;
        double neutral;
        bool at_most; // every value at or below `neutral` is neutral too
    };

    static constexpr Control controls[] = {
        {"repeat_penalty", 1.0, false},  {"typical_p", 1.0, false},
        {"typ_p", 1.0, false},           {"tfs_z", 1.0, false},
        {"dynatemp_range", 0.0, false},  {"mirostat", 0.0, false},
        {"xtc_probability", 0.0, false}, {"dry_multiplier", 0.0, false},
        {"top_n_sigma", 0.0, true},      {"n_probs", 0.0, false},
        {"n_indent", 0.0, false},        {"t_max_predict_ms", 0.0, true},
        {"t_max_prompt_ms", 0.0, true},
    };
    for (const Control& control : controls) {
        const std::optional<double> value = optional_number(body, control.key);
        if (!value) { continue; }
        const bool neutral =
            control.at_most ? *value <= control.neutral : *value == control.neutral;
        if (!neutral) {
            bad_request(std::string(control.key) +
                            " other than its neutral value requires a llama.cpp sampling or "
                            "stopping control that NInfer does not provide",
                        control.key, std::string(control.key) + "_not_supported");
        }
    }
    if (present(body, "grammar")) {
        const Json& grammar = body.at("grammar");
        if (!grammar.is_string()) { bad_request("grammar must be a string", "grammar"); }
        if (!grammar.get_ref<const std::string&>().empty()) {
            bad_request(
                "grammar requests constrained decoding by a GBNF grammar, which NInfer does "
                "not provide; use json_schema or response_format",
                "grammar", "constrained_decoding_not_supported");
        }
    }
    if (present(body, "lora")) {
        const Json& lora = body.at("lora");
        if (!lora.is_array()) { bad_request("lora must be an array", "lora"); }
        if (!lora.empty()) {
            bad_request("lora adapters are not supported", "lora", "lora_not_supported");
        }
    }
    if (present(body, "logit_bias")) {
        const Json& biases = body.at("logit_bias");
        if (!biases.is_object() && !biases.is_array()) {
            bad_request("logit_bias must be an object or an array", "logit_bias");
        }
        const bool neutral =
            biases.empty() ||
            (biases.is_object() && std::all_of(biases.begin(), biases.end(), [](const Json& value) {
                 return value.is_number() && value.get<double>() == 0.0;
             }));
        if (!neutral) {
            bad_request("nonzero logit_bias requires per-token logit modification, which NInfer "
                        "does not provide",
                        "logit_bias", "logit_bias_not_supported");
        }
    }
}

void parse_stop(const Json& body, GenerationRequest& output) {
    output.ignore_eos = optional_bool(body, "ignore_eos", false);
    if (!present(body, "stop")) { return; }
    const Json& stop  = body.at("stop");
    const auto append = [&](const Json& value) {
        if (!value.is_string()) { bad_request("stop entries must be strings", "stop"); }
        std::string text = value.get<std::string>();
        if (text.empty()) { bad_request("stop strings must not be empty", "stop"); }
        output.stop_strings.push_back(std::move(text));
    };
    if (stop.is_string()) {
        append(stop);
    } else if (stop.is_array()) {
        if (stop.size() > kMaximumStopStrings) {
            bad_request("stop supports at most " + std::to_string(kMaximumStopStrings) + " strings",
                        "stop");
        }
        for (const Json& value : stop) { append(value); }
    } else {
        bad_request("stop must be a string or an array of strings", "stop");
    }
}

void parse_output_limit(const Json& body, const RequestLimits& limits,
                        TextCompletionDialect dialect, TextCompletionRequest& output) {
    // llama.cpp reads n_predict, then max_tokens; OpenAI reads max_tokens.
    const char* param        = "max_tokens";
    std::optional<int> limit = std::nullopt;
    if (dialect == TextCompletionDialect::LlamaCpp) {
        limit = optional_int(body, "n_predict");
        if (limit) { param = "n_predict"; }
    }
    if (!limit) { limit = optional_int(body, "max_tokens"); }
    if (!limit) {
        apply_default_output_limit(output.generation, limits);
        return;
    }
    output.output_tokens_explicit = true;
    // llama.cpp's -1 (and -2, until the context fills) is no limit: the largest budget that still
    // lets every lane hold such a request, as on Chat Completions.
    const bool unlimited = dialect == TextCompletionDialect::LlamaCpp ? *limit < 0 : *limit == -1;
    if (unlimited) {
        output.generation.max_tokens           = limits.max_context;
        output.generation.derive_output_budget = true;
        return;
    }
    if (*limit < 0) {
        bad_request(std::string(param) + " must be nonnegative, or -1 for no limit", param);
    }
    output.generation.max_tokens = *limit;
}

void parse_structured_output_fields(const Json& body, TextCompletionDialect dialect,
                                    GenerationRequest& output) {
    if (present(body, "response_format")) {
        output.structured_output =
            parse_structured_output(body.at("response_format"), true, "response_format");
    }
    if (dialect != TextCompletionDialect::LlamaCpp || !present(body, "json_schema")) { return; }
    if (output.structured_output.kind != StructuredOutputKind::None) {
        bad_request("json_schema and response_format are exclusive", "json_schema");
    }
    // llama.cpp's json_schema is the bare schema; an empty one admits any JSON object.
    const Json& schema = body.at("json_schema");
    const Json format =
        schema.is_object() && schema.empty()
            ? Json{{"type", "json_object"}}
            : Json{{"type", "json_schema"},
                   {"json_schema", Json{{"name", "json_schema"}, {"schema", schema}}}};
    output.structured_output = parse_structured_output(format, true, "json_schema");
}

std::vector<std::string> parse_response_fields(const Json& body) {
    std::vector<std::string> fields;
    if (!present(body, "response_fields")) { return fields; }
    const Json& value = body.at("response_fields");
    if (!value.is_array()) {
        bad_request("response_fields must be an array of strings", "response_fields");
    }
    for (const Json& field : value) {
        if (!field.is_string()) {
            bad_request("response_fields must be an array of strings", "response_fields");
        }
        fields.push_back(field.get<std::string>());
    }
    return fields;
}

void parse_openai_controls(const Json& body, TextCompletionRequest& output) {
    if (const std::optional<int> count = optional_int(body, "n"); count && *count != 1) {
        bad_request("n requests multiple completions, while NInfer produces one completion per "
                    "request; only n=1 is supported",
                    "n", "n_not_supported");
    }
    if (const std::optional<int> best_of = optional_int(body, "best_of");
        best_of && *best_of != 1) {
        bad_request("best_of requests several candidate completions, while NInfer produces one; "
                    "only best_of=1 is supported",
                    "best_of", "best_of_not_supported");
    }
    if (const std::optional<int> logprobs = optional_int(body, "logprobs");
        logprobs && *logprobs != 0) {
        bad_request("logprobs requires per-token log probabilities in the response, which NInfer "
                    "does not provide",
                    "logprobs", "logprobs_not_supported");
    }
    if (present(body, "suffix")) {
        if (!body.at("suffix").is_string()) { bad_request("suffix must be a string", "suffix"); }
        if (!body.at("suffix").get_ref<const std::string&>().empty()) {
            bad_request("suffix requests fill-in-the-middle completion, which NInfer does not "
                        "provide",
                        "suffix", "suffix_not_supported");
        }
    }
    output.response.echo = optional_bool(body, "echo", false);
    if (present(body, "stream_options")) {
        const Json& options = body.at("stream_options");
        if (!options.is_object()) {
            bad_request("stream_options must be an object", "stream_options");
        }
        output.response.include_usage = optional_bool(options, "include_usage", false);
    }
}

std::string dump(const OutJson& value) {
    return value.dump(-1, ' ', false, OutJson::error_handler_t::replace);
}

std::string event(const OutJson& payload) { return "data: " + dump(payload) + "\n\n"; }

const char* stop_type(ninfer::FinishReason reason) noexcept {
    switch (reason) {
    case ninfer::FinishReason::StopToken:
        return "eos";
    case ninfer::FinishReason::StopString:
        return "word";
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "limit";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::Cancelled:
        return "none";
    }
    return "none";
}

OutJson generation_settings(const TextCompletionContext& context) {
    const ninfer::ResolvedSamplingParameters& sampling = context.sampling;
    return OutJson{{"n_predict", context.requested_output_tokens},
                   {"max_tokens", context.requested_output_tokens},
                   {"seed", sampling.seed},
                   {"temperature", sampling.temperature},
                   {"top_k", sampling.top_k},
                   {"top_p", sampling.top_p},
                   {"min_p", sampling.min_p},
                   {"presence_penalty", sampling.presence_penalty},
                   {"frequency_penalty", sampling.frequency_penalty},
                   {"stop", context.stop},
                   {"ignore_eos", context.ignore_eos},
                   {"stream", context.response.stream},
                   {"n_probs", 0},
                   {"timings_per_token", context.response.timings_per_token}};
}

// llama.cpp's response_fields: only the named fields, a nested one named by its "a/b" path and
// reported under that path.
OutJson select_fields(OutJson full, const std::vector<std::string>& fields) {
    if (fields.empty()) { return full; }
    OutJson out = OutJson::object();
    for (const std::string& field : fields) {
        const OutJson* node = &full;
        std::size_t begin   = 0;
        while (node != nullptr && begin <= field.size()) {
            const std::size_t end = std::min(field.find('/', begin), field.size());
            const std::string key = field.substr(begin, end - begin);
            node  = node->is_object() && node->contains(key) ? &node->at(key) : nullptr;
            begin = end + 1;
        }
        if (node != nullptr) { out[field] = *node; }
    }
    return out;
}

// The final llama.cpp object. Streamed, its content already went out in the chunks; the generated
// token ids, which the chunks do not carry, come here.
OutJson llama_final(const TextCompletionContext& context, const GenerationOutcome& outcome,
                    bool streamed) {
    OutJson payload = {
        {"index", 0},
        {"content", streamed ? std::string() : outcome.text},
        {"tokens", context.response.return_tokens ? OutJson(outcome.tokens) : OutJson::array()},
        {"id_slot", outcome.id_slot},
        {"stop", true},
        {"model", context.model},
        {"tokens_predicted", outcome.completion_tokens},
        {"tokens_evaluated", outcome.prompt_tokens},
        {"tokens_cached", outcome.metrics.prefix_cache_hit_tokens},
        {"generation_settings", generation_settings(context)},
        {"prompt", context.prompt_text},
        {"has_new_line", outcome.text.find('\n') != std::string::npos},
        {"truncated", false},
        {"stop_type", stop_type(outcome.finish_reason)},
        {"stopping_word", outcome.matched_stop_string.value_or(std::string())},
        {"timings", completion_timings_json(outcome)},
    };
    if (!outcome.session_digest.empty()) { payload["session_digest"] = outcome.session_digest; }
    return select_fields(std::move(payload), context.response.response_fields);
}

OutJson openai_base(const TextCompletionContext& context) {
    return OutJson{{"id", context.id},
                   {"object", "text_completion"},
                   {"created", context.created},
                   {"model", context.model}};
}

OutJson openai_choice(std::string text, OutJson finish_reason) {
    return OutJson{{"text", std::move(text)},
                   {"index", 0},
                   {"logprobs", nullptr},
                   {"finish_reason", std::move(finish_reason)}};
}

void add_slot_identity(OutJson& payload, const GenerationOutcome& outcome) {
    if (outcome.id_slot >= 0) { payload["id_slot"] = outcome.id_slot; }
    if (!outcome.session_digest.empty()) { payload["session_digest"] = outcome.session_digest; }
}

} // namespace

TextCompletionRequest parse_text_completion_request(const Json& body, const RequestLimits& limits,
                                                    TextCompletionDialect dialect) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    TextCompletionRequest output;
    output.response.dialect = dialect;
    // As on Chat Completions, an omitted, null or empty model names the default model.
    if (present(body, "model")) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        output.model = body.at("model").get<std::string>();
    }
    output.generation.raw_prompt = parse_prompt(body);
    parse_sampling(body, output.generation);
    require_neutral_controls(body);
    parse_stop(body, output.generation);
    parse_output_limit(body, limits, dialect, output);
    parse_structured_output_fields(body, dialect, output.generation);
    output.generation.cache_prompt    = optional_bool(body, "cache_prompt", true);
    output.response.stream            = optional_bool(body, "stream", false);
    output.response.timings_per_token = optional_bool(body, "timings_per_token", false);
    if (dialect == TextCompletionDialect::LlamaCpp) {
        output.response.return_progress = optional_bool(body, "return_progress", false);
        output.response.return_tokens   = optional_bool(body, "return_tokens", false);
        output.response.response_fields = parse_response_fields(body);
    } else {
        parse_openai_controls(body, output);
    }
    return output;
}

TextCompletionContext make_text_completion_context(const TextCompletionRequest& request,
                                                   std::string prompt_text,
                                                   const PreparedRequest& prepared) {
    TextCompletionContext context;
    context.response                = request.response;
    context.id                      = new_openai_completion_id();
    context.model                   = request.model;
    context.created                 = unix_time_now();
    context.prompt_text             = std::move(prompt_text);
    context.sampling                = prepared.sampling;
    context.requested_output_tokens = prepared.requested_output_tokens;
    context.ignore_eos              = request.generation.ignore_eos;
    context.stop                    = request.generation.stop_strings;
    return context;
}

std::string make_text_completion_response(const TextCompletionContext& context,
                                          const GenerationOutcome& outcome) {
    if (context.response.dialect == TextCompletionDialect::LlamaCpp) {
        return dump(llama_final(context, outcome, false));
    }
    OutJson payload    = openai_base(context);
    payload["choices"] = OutJson::array(
        {openai_choice(context.response.echo ? context.prompt_text + outcome.text : outcome.text,
                       openai_finish_reason(outcome.finish_reason))});
    payload["usage"]   = openai_usage_json(outcome);
    payload["timings"] = completion_timings_json(outcome);
    add_slot_identity(payload, outcome);
    return dump(payload);
}

TextCompletionStream::TextCompletionStream(TextCompletionContext context)
    : context_(std::move(context)) {}

std::vector<std::string> TextCompletionStream::start() {
    if (started_ || finished_) { throw std::logic_error("text completion stream already started"); }
    started_ = true;
    std::vector<std::string> events;
    if (context_.response.dialect == TextCompletionDialect::OpenAI && context_.response.echo &&
        !context_.prompt_text.empty()) {
        OutJson payload    = openai_base(context_);
        payload["choices"] = OutJson::array({openai_choice(context_.prompt_text, nullptr)});
        events.push_back(event(payload));
    }
    return events;
}

void TextCompletionStream::note_start(const ninfer::GenerationStart& start) {
    if (!started_ || admitted_ || finished_ ||
        start.reused_prompt_tokens > start.prompt.prompt_tokens) {
        throw std::logic_error("invalid text completion generation-start state");
    }
    admitted_      = true;
    prompt_tokens_ = start.prompt.prompt_tokens;
    cached_tokens_ = start.reused_prompt_tokens;
}

void TextCompletionStream::note_timing(const ninfer::GenerationTimingObservation& timing) {
    if (!started_ || !admitted_ || finished_ || timing.generated_tokens == 0) {
        throw std::logic_error("invalid text completion live-timing state");
    }
    if (live_timing_ && (timing.generated_tokens < live_timing_->generated_tokens ||
                         timing.generation_elapsed_ns < live_timing_->generation_elapsed_ns)) {
        throw std::logic_error("text completion live timings are not cumulative");
    }
    live_timing_ = timing;
}

OutJson TextCompletionStream::live_timings() const {
    if (!context_.response.timings_per_token || !live_timing_ || !admitted_) { return nullptr; }
    return completion_timings_json(prompt_tokens_, cached_tokens_, *live_timing_);
}

std::string TextCompletionStream::prompt_progress(const ninfer::PromptProgress& progress) {
    if (context_.response.dialect != TextCompletionDialect::LlamaCpp || !started_ || !admitted_ ||
        finished_) {
        throw std::logic_error("invalid text completion prompt-progress state");
    }
    return event(OutJson{{"index", 0},
                         {"content", ""},
                         {"tokens", OutJson::array()},
                         {"stop", false},
                         {"id_slot", -1},
                         {"tokens_predicted", 0},
                         {"tokens_evaluated", progress.total_prompt_tokens},
                         {"prompt_progress", completion_prompt_progress_json(progress)}});
}

std::string TextCompletionStream::content_delta(const std::string& text) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid text completion content delta state");
    }
    content_ += text;
    OutJson payload;
    if (context_.response.dialect == TextCompletionDialect::LlamaCpp) {
        payload = OutJson{{"index", 0},
                          {"content", text},
                          {"tokens", OutJson::array()},
                          {"stop", false},
                          {"id_slot", -1},
                          {"tokens_predicted", live_timing_ ? live_timing_->generated_tokens : 0U},
                          {"tokens_evaluated", prompt_tokens_}};
    } else {
        payload            = openai_base(context_);
        payload["choices"] = OutJson::array({openai_choice(text, nullptr)});
    }
    if (OutJson timings = live_timings(); !timings.is_null()) {
        payload["timings"] = std::move(timings);
    }
    return event(payload);
}

std::vector<std::string> TextCompletionStream::finish(const GenerationOutcome& outcome) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid text completion stream finish state");
    }
    finished_ = true;
    if (!outcome.text.starts_with(content_)) {
        throw std::logic_error("streamed content does not match terminal output");
    }
    std::vector<std::string> events;
    const std::string suffix = outcome.text.substr(content_.size());
    const bool llama         = context_.response.dialect == TextCompletionDialect::LlamaCpp;
    if (!suffix.empty()) {
        OutJson payload;
        if (llama) {
            payload = OutJson{{"index", 0},
                              {"content", suffix},
                              {"tokens", OutJson::array()},
                              {"stop", false},
                              {"id_slot", -1},
                              {"tokens_predicted", outcome.completion_tokens},
                              {"tokens_evaluated", outcome.prompt_tokens}};
        } else {
            payload            = openai_base(context_);
            payload["choices"] = OutJson::array({openai_choice(suffix, nullptr)});
        }
        events.push_back(event(payload));
    }
    if (llama) {
        // llama.cpp ends its native stream with the final object and no [DONE].
        events.push_back(event(llama_final(context_, outcome, true)));
        return events;
    }
    OutJson final_chunk = openai_base(context_);
    final_chunk["choices"] =
        OutJson::array({openai_choice(std::string(), openai_finish_reason(outcome.finish_reason))});
    if (!context_.response.include_usage) {
        final_chunk["timings"] = completion_timings_json(outcome);
        add_slot_identity(final_chunk, outcome);
    }
    events.push_back(event(final_chunk));
    if (context_.response.include_usage) {
        OutJson usage    = openai_base(context_);
        usage["choices"] = OutJson::array();
        usage["usage"]   = openai_usage_json(outcome);
        usage["timings"] = completion_timings_json(outcome);
        add_slot_identity(usage, outcome);
        events.push_back(event(usage));
    }
    events.emplace_back("data: [DONE]\n\n");
    return events;
}

} // namespace ninfer::serve
