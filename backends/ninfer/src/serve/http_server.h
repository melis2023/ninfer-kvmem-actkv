#pragma once

#include "serve/console_stats.h"
#include "serve/generation_service.h"
#include "serve/load_report.h"
#include "serve/model_registry.h"
#include "serve/openai_common.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_metrics.h"
#include "serve/serve_options.h"
#include "serve/text_completion.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

// Anthropic SDKs append /v1/<endpoint> to their base URL, so a client given the OpenAI-style base
// URL that already ends in /v1 requests /v1/v1/<endpoint>. Every /v1 API route also answers there:
// api_route_pattern("/messages") matches /v1/messages and /v1/v1/messages without adding a capture
// group, and canonical_api_path maps the doubled form back to the /v1/<endpoint> it names.
[[nodiscard]] std::string api_route_pattern(std::string_view endpoint);
[[nodiscard]] std::string_view canonical_api_path(std::string_view path) noexcept;

class HttpServer {
public:
    // `panel` is the console panel for the session statistics; it is drawn only when enabled by
    // the options and supported by the terminal, and may be null.
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger,
               std::shared_ptr<product::TerminalPanel> panel = nullptr);
    // Stops and joins the startup listener if it is still running. Without this, a failure between
    // start_serving_during_startup() and listen() -- the Engine throwing while loading weights, the
    // most likely failure there is -- would destroy a joinable std::thread and call std::terminate,
    // turning a clean diagnosable error into an abort.
    ~HttpServer();

    HttpServer(const HttpServer&)            = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Reserves the configured address before model loading. The registry is attached once the
    // models that load at startup are ready, then listen() enters the blocking accept loop on the
    // already-bound socket. In single-model mode the registry holds the one model, loaded; in
    // router mode it holds every model of the catalog, loaded or not.
    bool bind();
    void attach(ModelRegistry& registry);
    bool listen();
    // Closes the listening sockets and stops the attached service's Engine, whose queued and
    // running requests then fail, so listen() returns within about one unit of Engine work.
    // Does not block; callable from any thread.
    void stop();

    // Serve 503 while the Engine is still loading.
    //
    // bind() deliberately runs before the Engine is constructed, so a port clash fails in
    // milliseconds instead of after ten seconds of weight loading. The cost used to be a window --
    // measured at 0.27s to 9.5s on the 27B -- where the kernel accepted connections into the
    // backlog and nothing ever answered them: a TCP readiness probe called that "ready", and an
    // HTTP probe burned its whole timeout instead of failing fast.
    //
    // Calling this immediately after bind() starts the accept loop on a background thread with
    // every route answering 503 plus Retry-After. attach() then publishes the service and the same
    // loop begins serving normally, with no second bind and no handoff of the listening socket.
    void start_serving_during_startup();
    [[nodiscard]] bool serving_during_startup() const noexcept {
        return startup_listener_.joinable();
    }
    // Blocks until the background accept loop returns, which happens when stop() is called.
    // Mirrors listen()'s return: true when the loop exited cleanly.
    bool await_startup_listener();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

    // True once the engine watch stopped listen() because the one model's Engine failed
    // Engine-wide; the process then exits non-zero so a supervisor reloads the model.
    [[nodiscard]] bool engine_failed() const noexcept {
        return engine_failed_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool webui_enabled() const noexcept;

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    [[nodiscard]] httplib::Server::HandlerResponse pre_route(const httplib::Request& req,
                                                             httplib::Response& res) const;
    void register_routes();
    // Body-carrying routes are registered through httplib's content-reader form so that
    // --max-request-mib stays the only source of a 413; see buffer_request_body.
    void register_post(const std::string& pattern, httplib::Server::Handler handler);
    void register_delete(const std::string& pattern, httplib::Server::Handler handler);
    void register_stats_routes();
    void start_stats_listener();
    void stop_stats_listener();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_load(const httplib::Request& req, httplib::Response& res) const;
    void handle_metrics(const httplib::Request& req, httplib::Response& res) const;
    void handle_stats(const httplib::Request& req, httplib::Response& res) const;
    void handle_health(httplib::Response& res) const;
    void handle_slots(const httplib::Request& req, httplib::Response& res) const;
    void handle_model_residency(const httplib::Request& req, httplib::Response& res) const;
    void handle_model_suspend(const httplib::Request& req, httplib::Response& res);
    void handle_model_resume(const httplib::Request& req, httplib::Response& res);
    // llama.cpp router API: GET /models, POST /models/load and /models/unload, GET /models/sse; and
    // NInfer's POST /models/sleep.
    void handle_router_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_models_action(const httplib::Request& req, httplib::Response& res,
                              std::string_view action);
    void handle_models_sse(const httplib::Request& req, httplib::Response& res);
    void handle_slot_action(const httplib::Request& req, httplib::Response& res);
    // llama.cpp's native endpoints: POST /tokenize, /detokenize, /apply-template and /completion;
    // and OpenAI's legacy POST /v1/completions, the same raw-prompt completion.
    void handle_tokenize(const httplib::Request& req, httplib::Response& res) const;
    void handle_detokenize(const httplib::Request& req, httplib::Response& res) const;
    void handle_apply_template(const httplib::Request& req, httplib::Response& res) const;
    void handle_text_completion(const httplib::Request& req, httplib::Response& res,
                                TextCompletionDialect dialect);
    // POST /rerank, /reranking, /v1/rerank and /v1/reranking: the documents scored by the model.
    void handle_rerank(const httplib::Request& req, httplib::Response& res);
    void handle_props(const httplib::Request& req, httplib::Response& res) const;
    void handle_webui(const httplib::Request& req, httplib::Response& res) const;

    // What the server reports about one loaded model, computed once per Engine instance: an
    // unload and a later load start a new one.
    struct ModelFacts {
        const GenerationService* service = nullptr;
        LoadCapacity capacity;
        ninfer::ModelMetadata metadata;
        bool vision               = false;
        std::uint32_t max_context = 0;
    };

    // The model a request names (empty names the default model), leased for the call: loaded or
    // woken when autoload allows (the `autoload` query parameter overrides the server's choice).
    // Throws ApiException: 404 for an unknown model, 503 when it is not loaded and may not be or
    // failed to load.
    [[nodiscard]] ModelRegistry::Lease lease_model(std::string_view model,
                                                   const httplib::Request& req) const;
    // An Anthropic client names its own model families (Claude Code sends claude-*): a name no
    // model or alias answers to selects the default model instead of failing.
    [[nodiscard]] std::string anthropic_model(const std::string& requested) const;
    // The model of a request that only runs on the host (tokenizing, rendering a template): a
    // loaded or sleeping model serves it without waking; one not loaded loads as for generation.
    [[nodiscard]] ModelRegistry::Lease lease_host_model(std::string_view model,
                                                        const httplib::Request& req) const;
    // The `model` query parameter of a GET endpoint, leased the same way.
    [[nodiscard]] ModelRegistry::Lease lease_query_model(const httplib::Request& req) const;
    [[nodiscard]] ModelFacts model_facts(const ModelRegistry::Lease& lease) const;
    [[nodiscard]] LoadSample load_sample(GenerationService& service) const;
    [[nodiscard]] ModelDescription model_description(const ModelRegistry::Lease& lease) const;
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();
    void run_engine_watch();
    void stop_engine_watch();

    // Written once by attach() on the main thread and read by request handlers on httplib's worker
    // threads, so the publication has to be ordered. Handlers only ever test readiness through
    // ready_; registry_ itself is not read until ready_ has been observed true.
    ModelRegistry* registry_ = nullptr;
    std::atomic<bool> ready_{false};
    std::thread startup_listener_;
    std::atomic<bool> startup_listener_result_{false};
    ServeOptions options_;
    // The one model's id in single-model mode; empty in router mode, where requests name models.
    std::string public_model_id_;
    std::chrono::steady_clock::time_point attached_at_;
    mutable std::mutex facts_mutex_;
    mutable std::map<std::string, ModelFacts> facts_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    // Mutable: a model's server_start record is written when a const handler first sees it.
    mutable JsonlRequestLog request_jsonl_;
    ServeMetrics metrics_;
    std::unique_ptr<ConsoleStatsPanel> console_stats_;
    httplib::Server server_;
    // Present only with --stats-port.
    httplib::Server stats_server_;
    std::thread stats_listener_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    std::mutex watch_mutex_;
    std::condition_variable watch_cv_;
    std::thread watch_thread_;
    bool watch_stopping_ = false;
    std::atomic<bool> engine_failed_{false};
};

} // namespace ninfer::serve
