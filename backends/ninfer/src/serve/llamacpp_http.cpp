// llama.cpp's native server endpoints -- POST /tokenize, /detokenize, /apply-template, /completion
// and /rerank -- and OpenAI's legacy POST /v1/completions, which shares the raw-prompt completion.

#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"
#include "serve/rerank.h"
#include "serve/text_completion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using OutJson = nlohmann::json;

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

std::string dump(const OutJson& value) {
    return value.dump(-1, ' ', false, OutJson::error_handler_t::replace);
}

RequestJson parse_object_body(const httplib::Request& req) {
    RequestJson body = parse_json_body(req);
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

// The `model` of a request body; empty names the default model.
std::string body_model(const RequestJson& body) {
    if (!body.contains("model") || body.at("model").is_null()) { return {}; }
    if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
    return body.at("model").get<std::string>();
}

bool valid_utf8(std::string_view bytes) {
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto lead       = static_cast<unsigned char>(bytes[index]);
        std::size_t length    = 0;
        std::uint32_t minimum = 0;
        if (lead < 0x80) {
            length = 1;
        } else if (lead >= 0xC2 && lead <= 0xDF) {
            length  = 2;
            minimum = 0x80;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            length  = 3;
            minimum = 0x800;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length  = 4;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (index + length > bytes.size()) { return false; }
        std::uint32_t codepoint = length == 1 ? lead : lead & (0x7FU >> length);
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto next = static_cast<unsigned char>(bytes[index + offset]);
            if ((next & 0xC0U) != 0x80U) { return false; }
            codepoint = (codepoint << 6U) | (next & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFF ||
            (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
            return false;
        }
        index += length;
    }
    return true;
}

// llama.cpp's token piece: the text, or its bytes when they are not whole UTF-8 on their own.
OutJson piece_json(const std::string& bytes) {
    if (valid_utf8(bytes)) { return bytes; }
    OutJson out = OutJson::array();
    for (const char byte : bytes) { out.push_back(static_cast<unsigned char>(byte)); }
    return out;
}

std::string token_bytes(const GenerationService& service, ninfer::TokenId token) {
    try {
        return service.token_bytes(token);
    } catch (const std::out_of_range&) {
        bad_request("token id " + std::to_string(token) + " is outside the model's vocabulary",
                    "tokens");
    }
}

// The prompt as text: text pieces as they are, token ids as the bytes they decode to.
std::string prompt_text(const GenerationService& service,
                        const std::vector<RawPromptPiece>& pieces) {
    std::string text;
    for (const RawPromptPiece& piece : pieces) {
        text += piece.token ? token_bytes(service, *piece.token) : piece.text;
    }
    return text;
}

} // namespace

ModelRegistry::Lease HttpServer::lease_host_model(std::string_view model,
                                                  const httplib::Request& req) const {
    // Host work (tokenizing, rendering a template) needs no device memory: a sleeping model serves
    // it as it is. A model that is not loaded loads as for generation.
    std::optional<ModelRegistry::Lease> observed;
    try {
        observed = registry_->observe(model);
    } catch (const ModelUnknown&) {
        // lease_model reports the unknown name.
    }
    if (observed) { return std::move(*observed); }
    return lease_model(model, req);
}

void HttpServer::handle_tokenize(const httplib::Request& req, httplib::Response& res) const {
    const RequestJson body           = parse_object_body(req);
    const ModelRegistry::Lease lease = lease_host_model(body_model(body), req);
    std::string content;
    if (body.contains("content") && !body.at("content").is_null()) {
        if (!body.at("content").is_string()) { bad_request("content must be a string", "content"); }
        content = body.at("content").get<std::string>();
    }
    // add_special adds the vocabulary's BOS, which Qwen checkpoints do not use.
    (void)optional_bool(body, "add_special", false);
    const bool parse_special         = optional_bool(body, "parse_special", true);
    const bool with_pieces           = optional_bool(body, "with_pieces", false);
    const GenerationService& service = lease.service();
    OutJson tokens                   = OutJson::array();
    for (const ninfer::TokenId token : service.tokenize(content, parse_special)) {
        if (with_pieces) {
            tokens.push_back(
                OutJson{{"id", token}, {"piece", piece_json(service.token_bytes(token))}});
        } else {
            tokens.push_back(token);
        }
    }
    res.set_content(dump(OutJson{{"tokens", std::move(tokens)}}), "application/json");
}

void HttpServer::handle_detokenize(const httplib::Request& req, httplib::Response& res) const {
    const RequestJson body           = parse_object_body(req);
    const ModelRegistry::Lease lease = lease_host_model(body_model(body), req);
    std::string content;
    if (body.contains("tokens") && !body.at("tokens").is_null()) {
        const RequestJson& tokens = body.at("tokens");
        if (!tokens.is_array()) { bad_request("tokens must be an array of token ids", "tokens"); }
        for (const RequestJson& token : tokens) {
            if (!token.is_number_integer() || token.get<std::int64_t>() < 0 ||
                token.get<std::int64_t>() > std::numeric_limits<ninfer::TokenId>::max()) {
                bad_request("tokens must be an array of token ids", "tokens");
            }
            content += token_bytes(lease.service(),
                                   static_cast<ninfer::TokenId>(token.get<std::int64_t>()));
        }
    }
    res.set_content(dump(OutJson{{"content", std::move(content)}}), "application/json");
}

void HttpServer::handle_apply_template(const httplib::Request& req, httplib::Response& res) const {
    const RequestJson body          = parse_object_body(req);
    const OpenAIChatRequest request = parse_chat_completion_request(body, request_limits(options_));
    const ModelRegistry::Lease lease = lease_host_model(request.model, req);
    const std::string prompt         = lease.service().render_prompt(
        request.generation, [&req] { return client_disconnected(req); });
    res.set_content(dump(OutJson{{"prompt", prompt}}), "application/json");
}

void HttpServer::handle_rerank(const httplib::Request& req, httplib::Response& res) {
    RerankRequest request;
    ModelRegistry::Lease lease;
    try {
        request       = parse_rerank_request(parse_json_body(req));
        lease         = lease_model(request.model, req);
        request.model = lease.model();
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService& service = lease.service();
    const auto cancelled       = [&req] { return client_disconnected(req); };
    // One judgement per document, submitted a lane's worth at a time: the Engine runs a window
    // together, and the ingress queue keeps room for other clients' requests.
    const std::size_t window =
        std::max<std::size_t>(1, static_cast<std::size_t>(service.options().max_concurrency));
    std::vector<RerankScore> scores;
    scores.reserve(request.documents.size());
    int prompt_tokens = 0;
    try {
        for (std::size_t begin = 0; begin < request.documents.size(); begin += window) {
            const std::size_t end = std::min(begin + window, request.documents.size());
            std::vector<GenerationRequest> judgements;
            std::vector<PreparedRequest> batch;
            std::vector<std::shared_ptr<RequestLifecycle>> lifecycles;
            for (std::size_t index = begin; index < end; ++index) {
                judgements.push_back(make_rerank_judgement(request, request.documents[index]));
                batch.push_back(service.prepare(judgements.back(),
                                                GenerationConsumerMode::Aggregate,
                                                {.phase_timings = true}, cancelled));
                const RequestLogMetadata metadata{.model = request.model};
                lifecycles.push_back(begin_request(make_request_log_context(
                    ++request_seq_, "rerank", judgements.back(), metadata, batch.back())));
            }
            for (std::size_t offset = 0; offset < batch.size(); ++offset) {
                GenerationOutcome outcome;
                try {
                    outcome = service.run(batch[offset], nullptr, cancelled);
                } catch (const ApiException& exception) {
                    lifecycles[offset]->failure(make_generation_request_failure(exception.error()));
                    throw;
                }
                lifecycles[offset]->done(outcome);
                prompt_tokens += outcome.prompt_tokens;
                scores.push_back({.index = begin + offset, .score = rerank_score(outcome)});
            }
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_rerank_response(request, std::move(scores), prompt_tokens),
                    "application/json");
}

void HttpServer::handle_text_completion(const httplib::Request& req, httplib::Response& res,
                                        TextCompletionDialect dialect) {
    const bool llama     = dialect == TextCompletionDialect::LlamaCpp;
    const char* protocol = llama ? "llamacpp_completion" : "openai_completions";
    TextCompletionRequest request;
    ModelRegistry::Lease lease;
    try {
        const auto body = parse_json_body(req);
        request         = parse_text_completion_request(body, request_limits(options_), dialect);
        request.generation.ngram_session = resolve_ngram_session(req, body, options_);
        lease                            = lease_model(request.model, req);
        request.model                    = lease.model();
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService* const service = &lease.service();

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.response.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    std::string prompt;
    try {
        const ninfer::GenerationObservationOptions observation{
            .phase_timings = true,
            // llama.cpp chunks carry the running count of generated tokens.
            .live_timings =
                request.response.stream && (llama || request.response.timings_per_token),
            .prompt_progress = request.response.stream && request.response.return_progress,
        };
        prepared = service->prepare(request.generation,
                                    request.response.stream ? GenerationConsumerMode::Streaming
                                                            : GenerationConsumerMode::Aggregate,
                                    observation, [&req] { return client_disconnected(req); });
        prepared.model_hold = std::make_shared<ModelRegistry::Lease>(std::move(lease));
        if (llama || request.response.echo) {
            prompt = prompt_text(*service, *request.generation.raw_prompt);
        }
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, protocol, request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, protocol, request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    TextCompletionContext context =
        make_text_completion_context(request, std::move(prompt), prepared);
    auto lifecycle = begin_request(
        make_request_log_context(req_id, protocol, request.generation, metadata, prepared));

    if (!request.response.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_openai_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_ngram_generation_header(res, outcome.metrics.ngram_archive);
            set_owned_json_content(res, make_text_completion_response(context, outcome),
                                   prepared.lifetime);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            write_openai_error(res, error);
        }
        return;
    }

    try {
        const bool return_progress = request.response.return_progress;
        const bool live_timings    = llama || request.response.timings_per_token;
        auto stream                = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder               = std::make_shared<TextCompletionStream>(std::move(context));

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [service, stream, encoder, lifecycle, return_progress,
             live_timings](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        render_and_write(transport, [&] { return sse_error_event(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };
                const auto internal_error = [](const char* message) {
                    ApiError error;
                    error.status  = 500;
                    error.type    = "internal_error";
                    error.message = message;
                    return error;
                };
                try {
                    transport.write(encoder->start());
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        encoder->note_start(start);
                    };
                    if (return_progress) {
                        output.on_progress = [&](const ninfer::PromptProgress& progress) {
                            render_and_write(transport,
                                             [&] { return encoder->prompt_progress(progress); });
                        };
                    }
                    if (live_timings) {
                        output.on_timing = [&](const ninfer::GenerationTimingObservation& timing) {
                            encoder->note_timing(timing);
                        };
                    }
                    output.on_content = [&](const std::string& text,
                                            std::span<const ninfer::TokenLogprob>) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };

                    outcome = service->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(internal_error(exception.what()));
                } catch (const ApiException& exception) {
                    lifecycle->failure(make_generation_request_failure(exception.error()));
                    return send_error(exception.error());
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    return send_error(internal_error(exception.what()));
                }

                lifecycle->done(outcome);
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                    if (auto comment = ngram_generation_comment(outcome.metrics.ngram_archive);
                        !comment.empty()) {
                        terminal.insert(terminal.begin(), std::move(comment));
                    }
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    return send_error(internal_error(exception.what()));
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
