#include "serve/http_server.h"
#include "serve/slot_files.h"

#include "product/logging/logging.h"
#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/kvmem_status_json.h"
#include "serve/mcp_proxy.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"
#include "serve/webui.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr std::string_view kCorsAllowedHeaders =
    "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
    "anthropic-user-profile-id";

// API routes keep their own 404s. Every other GET path belongs to the WebUI, whose client-side
// router owns paths the server has no file for.
bool is_api_path(std::string_view path) {
    return path == "/v1" || path.starts_with("/v1/") || path == "/health" || path == "/metrics" ||
           path == "/stats" || path == "/slots" || path == "/props" || path == "/models" ||
           path.starts_with("/models/") || path == "/completion" || path == "/completions" ||
           path == "/tokenize" || path == "/detokenize" || path == "/apply-template" ||
           path == "/rerank" || path == "/reranking" || path == kMcpProxyPath;
}


void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) {
    return canonical_api_path(path).starts_with("/v1/messages");
}

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

// A plain-handler route has its body read by dispatch_request() through Server::read_content,
// which rejects any application/x-www-form-urlencoded body larger than 8 KiB (httplib's
// CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH) with a 413 that ignores --max-request-mib --
// and `curl -d` sends JSON with that content type. The check runs before any handler, so no route
// can correct it. The content-reader route form reads through read_content_core, which enforces
// only the configured payload limit; buffering the body here keeps a 413 meaning exactly "the raw
// request body exceeds --max-request-mib".
void buffer_request_body(const httplib::Request& request, httplib::Response& response,
                         const httplib::ContentReader& content_reader,
                         const httplib::Server::Handler& handler) {
    httplib::Request buffered = request;
    buffered.body.clear();
    const bool complete = content_reader([&](const char* data, std::size_t length) {
        buffered.body.append(data, length);
        return true;
    });
    // An aborted read already carries httplib's own status (413 for the payload limit); the
    // unrendered-error handler renders its documented envelope.
    if (!complete) { return; }
    handler(buffered, response);
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_demoted !=
               report.previous.pressure_private_owners_demoted ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.engine_recoveries != report.previous.engine_recoveries ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_main_kv_lease_pages !=
               report.previous.device_main_kv_lease_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.device_backend_kv_lease_pages !=
               report.previous.device_backend_kv_lease_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view request_path) noexcept {
    const std::string_view path = canonical_api_path(request_path);
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/completions") { return "openai_completions"; }
    if (path == "/completion" || path == "/completions") { return "llamacpp_completion"; }
    if (path == "/tokenize" || path == "/detokenize" || path == "/apply-template") {
        return "llamacpp_tokenizer";
    }
    if (path == "/rerank" || path == "/reranking" || path == "/v1/rerank" ||
        path == "/v1/reranking") {
        return "rerank";
    }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    if (path == "/v1/load") { return "load"; }
    if (path == "/stats") { return "stats"; }
    if (path == "/slots" || path.starts_with("/slots/")) { return "slots"; }
    if (path == "/models" || path.starts_with("/models/")) { return "models"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && is_anthropic_path(request.path)) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (is_anthropic_path(request.path)) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

std::string api_route_pattern(std::string_view endpoint) {
    return "(?:/v1)?/v1" + std::string(endpoint);
}

std::string_view canonical_api_path(std::string_view path) noexcept {
    constexpr std::string_view kVersion = "/v1";
    if (path.starts_with("/v1/v1/")) { path.remove_prefix(kVersion.size()); }
    return path;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger,
                       std::shared_ptr<product::TerminalPanel> panel)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger),
                     static_cast<std::uint64_t>(options_.request_log_max_mib) << 20U,
                     options_.request_log_keep) {
    if (options_.log_stats_panel && panel != nullptr && panel->enabled()) {
        console_stats_ = std::make_unique<ConsoleStatsPanel>(std::move(panel));
    }
    // cpp-httplib is thread-per-connection: a worker is held for a connection's whole
    // life, including the idle keep-alive window between requests. Sizing the pool to the
    // request-lifetime capacity alone lets idle pooled connections occupy every worker, at
    // which point a C8 server accepts and runs requests strictly one at a time. Base
    // workers cover the admissible request capacity; the headroom is grown on demand for
    // connections that are merely open, and those dynamic workers retire once idle.
    constexpr std::size_t kKeepAliveWorkerHeadroom = 64;
    const std::size_t request_workers =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests + 1;
    const std::size_t worker_limit = request_workers + kKeepAliveWorkerHeadroom;
    server_.new_task_queue         = [request_workers, worker_limit] {
        return new httplib::ThreadPool(request_workers, worker_limit, worker_limit);
    };
    server_.set_tcp_nodelay(true);
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
    if (options_.stats_port != 0) { register_stats_routes(); }
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    metrics_.record_rejection();
    operational_log_.request_rejected(context);
    if (console_stats_) {
        console_stats_->request_rejected(
            make_request_failure(RequestFailurePhase::Prepare, context.error));
    }
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    metrics_.record_done(outcome);
    operational_log_.request_done(context, outcome);
    if (console_stats_) { console_stats_->request_done(outcome); }
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    metrics_.record_failure();
    operational_log_.request_failure(context, failure);
    if (console_stats_) { console_stats_->request_failure(failure); }
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock = std::chrono::steady_clock;

    // Each model's counters since the previous report. A model whose Engine changed (unloaded and
    // loaded again) starts over, since its counters did.
    struct Previous {
        const GenerationService* service = nullptr;
        ninfer::RuntimeStats stats;
        Clock::time_point time;
    };

    std::map<std::string, Previous> previous;
    const auto sample = [&](bool tail) {
        registry_->for_each_service([&](const std::string& id, GenerationService& service) {
            const ninfer::RuntimeStats current = service.runtime_stats();
            const Clock::time_point now        = Clock::now();
            auto found                         = previous.find(id);
            if (found == previous.end() || found->second.service != &service) {
                previous[id] = Previous{&service, current, now};
                return;
            }
            const ThroughputReport report = make_throughput_report(
                found->second.stats, current,
                std::chrono::duration<double>(now - found->second.time).count());
            if (report_has_activity(report)) {
                // The exact partial interval remains useful to measurement consumers. Pretty
                // throughput is a fixed-cadence operational record without an irregular tail.
                if (tail) {
                    request_jsonl_.write_throughput(report);
                } else {
                    record_throughput(report);
                }
            }
            if (!tail && console_stats_ && (public_model_id_.empty() || id == public_model_id_)) {
                console_stats_->runtime(current);
            }
            found->second = Previous{&service, current, now};
        });
    };
    sample(false);
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = Clock::now() + interval;
    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }
        sample(false);
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }
    sample(true);
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

httplib::Server::HandlerResponse HttpServer::pre_route(const httplib::Request& req,
                                                       httplib::Response& res) const {
    ensure_openai_request_id(req, res);
    if (!ready_.load(std::memory_order_acquire)) {
        // Runs for every route, including /health and OPTIONS, so a caller cannot tell "not
        // ready" apart from "unauthenticated" -- and skips the API-key check below, since a
        // loading-status response carries nothing worth protecting.
        ApiError error;
        error.status  = 503;
        error.type    = "service_unavailable";
        error.code    = "model_loading";
        error.message = "The model is still loading. Retry shortly.";
        // Weight load plus warmup measured ~10s on the 27B and longer on the 35B. Two seconds
        // is a polite poll interval rather than a promise about when readiness arrives.
        res.set_header("Retry-After", "2");
        if (is_anthropic_path(req.path)) {
            write_anthropic_error(res, error, new_anthropic_request_id());
        } else {
            write_openai_error(res, error);
        }
        return httplib::Server::HandlerResponse::Handled;
    }
    // The MCP relay carries no API key: the WebUI prefixes every header it means for the MCP
    // server, its own Authorization included, so requiring the key would break the relay
    // rather than protect it. It is opt-in, and the bind address is its boundary.
    if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS" ||
        (options_.webui_mcp_proxy && req.path == kMcpProxyPath) ||
        (webui_enabled() && req.method == "GET" && !is_api_path(req.path))) {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    // Accept both the OpenAI-style bearer token and the Anthropic-style
    // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
    // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
    const bool bearer_ok =
        matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
    const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
    if (!bearer_ok && !x_api_key_ok) {
        ApiError error;
        error.status  = 401;
        error.type    = "invalid_request_error";
        error.code    = "invalid_api_key";
        error.message = "missing or invalid API key";
        // Render the 401 in the shape the target endpoint speaks.
        if (is_anthropic_path(req.path)) {
            write_anthropic_error(res, error, new_anthropic_request_id());
        } else {
            write_openai_error(res, error);
        }
        return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
}

void HttpServer::register_post(const std::string& pattern, httplib::Server::Handler handler) {
    server_.Post(pattern, [handler = std::move(handler)](const httplib::Request& request,
                                                         httplib::Response& response,
                                                         const httplib::ContentReader& reader) {
        buffer_request_body(request, response, reader, handler);
    });
}

void HttpServer::register_delete(const std::string& pattern, httplib::Server::Handler handler) {
    server_.Delete(pattern, [handler = std::move(handler)](const httplib::Request& request,
                                                           httplib::Response& response,
                                                           const httplib::ContentReader& reader) {
        buffer_request_body(request, response, reader, handler);
    });
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers", std::string(kCorsAllowedHeaders)},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key. Headers a
        // client asks for beyond the fixed list (a WebUI's x-conversation-id, say) are allowed
        // too, since the preflight only gates which headers the browser may send.
        server_.Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
            res.status            = 204;
            const auto& requested = req.get_header_value("Access-Control-Request-Headers");
            if (!requested.empty()) {
                res.headers.erase("Access-Control-Allow-Headers");
                res.set_header("Access-Control-Allow-Headers",
                               std::string(kCorsAllowedHeaders) + ", " + requested);
            }
        });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        return pre_route(req, res);
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (is_anthropic_path(req.path)) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (is_anthropic_path(req.path)) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (is_anthropic_path(req.path)) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health",
                [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
    server_.Get(api_route_pattern("/load"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_load(req, res);
                });
    server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });
    server_.Get("/stats", [this](const httplib::Request& req, httplib::Response& res) {
        handle_stats(req, res);
    });
    server_.Get("/slots", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slots(req, res);
    });
    register_post(R"(/slots/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_slot_action(req, res);
    });
    server_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res);
    });
    for (const char* base : {"/v1", "/v1/"}) {
        server_.Get(base, [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(
                make_api_index(public_model_id_.empty() ? std::string("router") : public_model_id_)
                    .dump(),
                "application/json");
        });
    }
    server_.Get(api_route_pattern("/models"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_models(req, res);
                });
    // llama.cpp's router API, in single-model mode too: one model is still a catalog.
    server_.Get("/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_router_models(req, res);
    });
    server_.Get("/models/sse", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models_sse(req, res);
    });
    for (const char* action : {"load", "unload", "sleep"}) {
        register_post(std::string("/models/") + action,
                      [this, action](const httplib::Request& req, httplib::Response& res) {
                          handle_models_action(req, res, action);
                      });
    }
    // Before the single-model route below, which would take "<id>/residency" for a model id.
    server_.Get(api_route_pattern(R"(/models/(.+)/residency)"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_model_residency(req, res);
                });
    register_post(api_route_pattern(R"(/models/(.+)/suspend)"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_model_suspend(req, res);
                  });
    register_post(api_route_pattern(R"(/models/(.+)/resume)"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_model_resume(req, res);
                  });
    server_.Get(api_route_pattern(R"(/models/(.+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_model(req, res);
                });
    register_post(api_route_pattern("/chat/completions"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_chat_completions(req, res);
                  });
    register_post(api_route_pattern("/completions"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_text_completion(req, res, TextCompletionDialect::OpenAI);
                  });
    for (const char* path : {"/completion", "/completions"}) {
        register_post(path, [this](const httplib::Request& req, httplib::Response& res) {
            handle_text_completion(req, res, TextCompletionDialect::LlamaCpp);
        });
    }
    register_post("/tokenize", [this](const httplib::Request& req, httplib::Response& res) {
        handle_tokenize(req, res);
    });
    register_post("/detokenize", [this](const httplib::Request& req, httplib::Response& res) {
        handle_detokenize(req, res);
    });
    register_post("/apply-template", [this](const httplib::Request& req, httplib::Response& res) {
        handle_apply_template(req, res);
    });
    for (const char* endpoint : {"/rerank", "/reranking"}) {
        const auto rerank = [this](const httplib::Request& req, httplib::Response& res) {
            handle_rerank(req, res);
        };
        register_post(endpoint, rerank);
        register_post(api_route_pattern(endpoint), rerank);
    }
    register_post(api_route_pattern("/responses"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_responses(req, res);
                  });
    register_post(api_route_pattern("/responses/input_tokens"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_response_input_tokens(req, res);
                  });
    register_post(api_route_pattern("/responses/compact"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_response_compact(req, res);
                  });
    register_post(api_route_pattern(R"(/responses/([^/]+)/cancel)"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_response_cancel(req, res);
                  });
    server_.Get(api_route_pattern(R"(/responses/([^/]+)/input_items)"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(api_route_pattern(R"(/responses/([^/]+))"),
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    register_delete(api_route_pattern(R"(/responses/([^/]+))"),
                    [this](const httplib::Request& req, httplib::Response& res) {
                        handle_response_delete(req, res);
                    });
    register_post(api_route_pattern("/messages/count_tokens"),
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_count_tokens(req, res);
                  });
    register_post(
        api_route_pattern("/messages"),
        [this](const httplib::Request& req, httplib::Response& res) { handle_messages(req, res); });
    if (options_.webui_mcp_proxy) {
        const auto relay = [](const httplib::Request& req, httplib::Response& res) {
            relay_mcp_proxy(req, res);
        };
        server_.Get(kMcpProxyPath, relay);
        register_post(kMcpProxyPath, relay);
        register_delete(kMcpProxyPath, relay);
    }
    // Registered last: httplib tries routes in order, so every API route above wins its path.
    if (webui_enabled()) {
        server_.Get(R"(/.*)", [this](const httplib::Request& req, httplib::Response& res) {
            handle_webui(req, res);
        });
    }
}

bool HttpServer::webui_enabled() const noexcept {
    return options_.enable_webui && !webui_assets().empty();
}

void HttpServer::handle_webui(const httplib::Request& req, httplib::Response& res) const {
    if (is_api_path(req.path)) {
        res.status = 404;
        return;
    }
    const bool gzip         = req.get_header_value("Accept-Encoding").find("gzip") != std::string::npos;
    const WebUiAsset* asset = req.path.size() > 1
                                  ? find_webui_asset(std::string_view(req.path).substr(1), gzip)
                                  : nullptr;
    if (asset == nullptr) { asset = find_webui_asset("index.html", gzip); }
    if (asset == nullptr) {
        res.status = 404;
        return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_header("ETag", std::string(asset->etag));
    res.set_header("Vary", "Accept-Encoding");
    if (!asset->encoding.empty()) {
        res.set_header("Content-Encoding", std::string(asset->encoding));
    }
    if (req.get_header_value("If-None-Match") == asset->etag) {
        res.status = 304;
        return;
    }
    res.set_content(reinterpret_cast<const char*>(asset->bytes.data()), asset->bytes.size(),
                    std::string(asset->content_type));
}

// llama.cpp-shaped slot listing: one entry per private context-cache catalog cell. A cell an
// active request will publish into reports that request's prompt and reused tokens; a retained
// cell reports the session depth as both, with its session digest and restorable checkpoints.
void HttpServer::handle_slots(const httplib::Request& req, httplib::Response& res) const {
    ModelRegistry::Lease lease;
    try {
        lease = lease_query_model(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService& service = lease.service();
    const bool speculative =
        service.options().speculative.backend != ninfer::SpeculativeBackend::None;
    const std::uint32_t context                 = service.options().max_context;
    const std::vector<ninfer::SlotState> states = service.slot_states();
    nlohmann::json slots                        = nlohmann::json::array();
    for (std::size_t index = 0; index < states.size(); ++index) {
        const ninfer::SlotState& state = states[index];
        nlohmann::json checkpoints     = nlohmann::json::array();
        for (const ninfer::SlotCheckpoint& checkpoint : state.checkpoints) {
            checkpoints.push_back({{"frontier", checkpoint.frontier},
                                   {"session_digest", checkpoint.session_digest}});
        }
        slots.push_back({{"id", index},
                         {"is_processing", state.processing},
                         {"retained", state.retained},
                         {"session_digest", state.session_digest},
                         {"checkpoints", std::move(checkpoints)},
                         {"n_ctx", context},
                         {"n_prompt_tokens", state.prompt_tokens},
                         {"n_prompt_tokens_cache", state.cached_tokens},
                         {"speculative", speculative}});
    }
    res.set_header("Cache-Control", "no-store");
    res.set_content(slots.dump(), "application/json");
}

// llama.cpp-shaped session persistence: POST /slots/{id}?action=save|restore|erase with
// {"filename": NAME} for save and restore and an optional {"if_digest": DIGEST} precondition on
// save and erase. Enabled only by --slot-save-path; names are confined to that directory.
void HttpServer::handle_slot_action(const httplib::Request& req, httplib::Response& res) {
    const auto fail = [&res](int status, std::string code, std::string message) {
        ApiError error;
        error.status  = status;
        error.type    = status >= 500 ? "server_error" : "invalid_request_error";
        error.code    = std::move(code);
        error.message = std::move(message);
        write_openai_error(res, error);
    };
    if (options_.slot_save_path.empty()) {
        fail(501, "slot_persistence_disabled",
             "this server was started without --slot-save-path; slot save/restore is disabled");
        return;
    }
    ModelRegistry::Lease lease;
    try {
        lease = lease_query_model(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService& service = lease.service();
    const std::string id_text = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(id_text);
    } catch (const std::exception&) {
        fail(400, "invalid_slot", "slot id is not a number");
        return;
    }
    // Range-checked before narrowing, so an id past 2^32 is refused rather than wrapped.
    const std::size_t slot_count = service.slot_states().size();
    if (parsed >= slot_count) {
        fail(400, "invalid_slot",
             "slot " + id_text + " is outside this server's " + std::to_string(slot_count) +
                 " slots");
        return;
    }
    const auto slot = static_cast<std::uint32_t>(parsed);
    const std::string action = req.get_param_value("action");

    std::string filename;
    std::string if_digest;
    try {
        const nlohmann::json body =
            req.body.empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);
        if (!body.is_object()) { throw std::invalid_argument("body is not an object"); }
        filename  = body.value("filename", std::string());
        if_digest = body.value("if_digest", std::string());
    } catch (const std::exception&) {
        fail(400, "invalid_request",
             "request body must be a JSON object with string filename and if_digest");
        return;
    }

    try {
        if (action == "erase") {
            const std::uint32_t erased = service.slot_erase(slot, if_digest);
            operational_log_.slot_erased(slot, erased);
            res.set_content(nlohmann::json{{"id_slot", slot}, {"n_erased", erased}}.dump(),
                            "application/json");
            return;
        }
        if (action != "save" && action != "restore") {
            fail(400, "invalid_action", "action must be save, restore, or erase");
            return;
        }
        const std::optional<std::string> sanitized = sanitize_slot_filename(filename);
        if (!sanitized) {
            fail(400, "invalid_filename",
                 "filename must be 1-" + std::to_string(kSlotFilenameMaxBytes) +
                     " characters of [A-Za-z0-9._-], must not start or end with a dot, and "
                     "must not name a device");
            return;
        }
        const std::string path = (options_.slot_save_path / *sanitized).string();
        if (action == "save") {
            const ninfer::SlotSaveResult saved = service.slot_save(slot, path, if_digest);
            operational_log_.slot_saved(slot, *sanitized, saved);
            res.set_content(nlohmann::json{{"id_slot", slot},
                                           {"filename", *sanitized},
                                           {"n_saved", saved.tokens},
                                           {"n_written", saved.bytes},
                                           {"session_digest", saved.session_digest},
                                           {"timings", {{"save_ms", saved.seconds * 1000.0}}}}
                                .dump(),
                            "application/json");
        } else {
            const ninfer::SlotRestoreResult restored = service.slot_restore(slot, path);
            operational_log_.slot_restored(slot, *sanitized, restored);
            res.set_content(
                nlohmann::json{{"id_slot", slot},
                               {"filename", *sanitized},
                               {"n_restored", restored.tokens},
                               {"n_read", restored.bytes},
                               {"session_digest", restored.session_digest},
                               {"timings", {{"restore_ms", restored.seconds * 1000.0}}}}
                    .dump(),
                "application/json");
        }
    } catch (const ninfer::RequestError& busy) {
        fail(409, "slot_busy", busy.what());
    } catch (const ninfer::SlotSessionMismatch& mismatch) {
        fail(409, "slot_session_mismatch", mismatch.what());
    } catch (const std::invalid_argument& rejected) {
        fail(400, "slot_" + action + "_failed", rejected.what());
    }
}

LoadSample HttpServer::load_sample(GenerationService& service) const {
    LoadSample sample;
    sample.uptime_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - attached_at_).count();
    sample.admitted_requests      = service.admitted_requests();
    sample.peak_admitted_requests = service.peak_admitted_requests();
    sample.stats                  = service.runtime_stats();
    return sample;
}

void HttpServer::handle_load(const httplib::Request& req, httplib::Response& res) const {
    try {
        const ModelRegistry::Lease lease = lease_query_model(req);
        res.set_header("Cache-Control", "no-store");
        res.set_content(make_load_report(model_facts(lease).capacity, load_sample(lease.service())),
                        "application/json");
    } catch (const ApiException& exception) { write_openai_error(res, exception.error()); }
}

void HttpServer::handle_stats(const httplib::Request& req, httplib::Response& res) const {
    try {
        const ModelRegistry::Lease lease = lease_query_model(req);
        res.set_header("Cache-Control", "no-store");
        res.set_content(
            make_stats_report(model_facts(lease).capacity, load_sample(lease.service())),
            "application/json");
    } catch (const ApiException& exception) { write_openai_error(res, exception.error()); }
}

// The pollers' own listener: one worker and no generation routes, so a dashboard or watchdog is
// never queued behind the connections the request pool is serving. It applies the same readiness
// and API-key rules as the main listener.
void HttpServer::register_stats_routes() {
    stats_server_.new_task_queue = [] { return new httplib::ThreadPool(1, 1, 64); };
    stats_server_.set_socket_options(configure_http_server_socket);
    stats_server_.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
            return pre_route(req, res);
        });
    stats_server_.Get(
        "/health", [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
    stats_server_.Get("/stats", [this](const httplib::Request& req, httplib::Response& res) {
        handle_stats(req, res);
    });
    stats_server_.Get("/v1/load", [this](const httplib::Request& req, httplib::Response& res) {
        handle_load(req, res);
    });
    stats_server_.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_metrics(req, res);
    });
}

void HttpServer::handle_health(httplib::Response& res) const {
    // A router is healthy while it runs: its models load on demand. One model is healthy while its
    // Engine serves, which includes a model that sleeps but wakes for the next request.
    bool available = registry_ != nullptr;
    if (available && !registry_->router()) {
        const auto status = registry_->status(public_model_id_);
        available = status && !status->held &&
                    (status->state == ModelState::Loaded || status->state == ModelState::Sleeping);
        if (available && status->state == ModelState::Loaded) {
            if (auto lease = registry_->observe(public_model_id_)) {
                available = lease->service().is_available();
            }
        }
    }
    res.status           = available ? 200 : 503;
    res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                    "application/json");
}

void HttpServer::handle_metrics(const httplib::Request& req, httplib::Response& res) const {
    try {
        const ModelRegistry::Lease lease = lease_query_model(req);
        res.set_header("Cache-Control", "no-store");
        res.set_content(metrics_.render(model_facts(lease).capacity, load_sample(lease.service())),
                        "text/plain; version=0.0.4; charset=utf-8");
    } catch (const ApiException& exception) { write_openai_error(res, exception.error()); }
}

void HttpServer::handle_props(const httplib::Request& req, httplib::Response& res) const {
    ModelRegistry::Lease lease;
    try {
        lease = lease_query_model(req);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }
    GenerationService& service          = lease.service();
    const ServeOptions& options         = service.options();
    const ModelFacts facts              = model_facts(lease);
    const ninfer::SamplingPreset preset = service.sampling_defaults().for_mode(
        options.enable_thinking == false ? ninfer::SamplingMode::NonThinking
                                         : ninfer::SamplingMode::Thinking);
    const ninfer::SamplingOverrides& overrides = options.sampling_overrides;
    // llama.cpp's -1 means "no fixed cap": without --default-max-tokens each request's budget is
    // derived from its prompt and the lane share, so no single number applies.
    const int n_predict   = options.default_max_tokens.value_or(-1);
    nlohmann::json params = {
        {"n_predict", n_predict},
        {"max_tokens", n_predict},
        {"temperature", options.greedy ? 0.0F : overrides.temperature.value_or(preset.temperature)},
        {"top_k", overrides.top_k.value_or(preset.top_k)},
        {"top_p", overrides.top_p.value_or(preset.top_p)},
        {"min_p", overrides.min_p.value_or(preset.min_p)},
        {"presence_penalty", overrides.presence_penalty.value_or(preset.presence_penalty)},
        {"frequency_penalty", overrides.frequency_penalty.value_or(preset.frequency_penalty)},
    };
    params["seed"] = overrides.seed ? nlohmann::json(*overrides.seed) : nlohmann::json(-1);
    nlohmann::json props = {
        {"default_generation_settings",
         {{"n_ctx", facts.capacity.max_context},
          {"speculative", options.speculative.backend != ninfer::SpeculativeBackend::None},
          {"params", std::move(params)}}},
        {"total_slots", facts.capacity.max_concurrency},
        {"model_alias", lease.model()},
        {"model_path", options.artifact_path},
        {"modalities", {{"vision", options.enable_vision}, {"audio", false}}},
        {"is_sleeping", service.residency().state == ninfer::ModelResidency::Suspended},
        {"endpoint_slots", true},
        {"endpoint_props", true},
        {"endpoint_metrics", true},
        // The WebUI ungreys its "Use llama-server proxy" option from this flag alone.
        {"cors_proxy_enabled", options_.webui_mcp_proxy},
    };
    props["backend"] = "ninfer";
    const auto& execution = service.engine_options();
    props["execution"] = {{"mode", execution.devices.size() > 1 ? "layer" : "single"},
        {"devices", execution.devices.empty() ? std::vector<int>{execution.device} : execution.devices},
        {"stage_layers", execution.stage_layers}};
    props["kvmem"] = {{"enabled", options.kvmem.selected_tokens != 0},
        {"selected_tokens", options.kvmem.selected_tokens},
        {"reserve_tokens", options.kvmem.reserve_tokens},
        {"sink_tokens", options.kvmem.sink_tokens},
        {"recent_tokens", options.kvmem.recent_tokens},
        {"host_payload_budget_bytes", options.kvmem.host_bytes}};
    if (options.kvmem.selected_tokens) {
        props["kvmem"]["usage"] = kvmem_status_json(service.runtime_stats().kvmem);
        props["kvmem"]["capabilities"] = {{"text_chat", true}, {"streaming", true},
            {"history_reuse", options.allow_prefix_reuse}, {"original_positions", true},
            {"disk_restore", false},
            {"active_kv_tiering", true},
            {"ssd_enabled", options.kvmem.disk_bytes != 0},
            {"ssd_persistence", "process"},
            {"speculative_decoding", options.speculative.backend == SpeculativeBackend::Mtp},
            {"mtp_drafts", options.speculative.draft_tokens}, {"multimodal", options.enable_vision},
            {"ngram_drafts", options.speculative.ngram_draft_tokens},
            {"ngram_min_match", options.speculative.ngram_min_match},
            {"max_active_requests", options.max_concurrency},
            {"retained_sessions", options.kvmem.retained_sessions},
            {"multi_gpu_mode", execution.devices.size() > 1 ? "layer" : "single"},
            {"shared_host_history", true},
            {"disk_budget_bytes", options.kvmem.disk_bytes},
            {"cross_backend_kv_transfer", false}};
    }
    res.set_content(props.dump(), "application/json");
}

ModelDescription HttpServer::model_description(const ModelRegistry::Lease& lease) const {
    const ModelFacts facts = model_facts(lease);
    ModelDescription out{.id            = lease.model(),
                         .max_model_len = facts.max_context,
                         .vision        = facts.vision,
                         .metadata      = facts.metadata};
    if (const auto status = registry_->status(lease.model())) {
        out.status  = model_state_name(status->state);
        out.aliases = status->aliases;
    }
    out.path = lease.service().options().artifact_path;
    return out;
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    std::vector<ModelDescription> models;
    for (const ModelStatusSnapshot& status : registry_->status()) {
        if (auto lease = registry_->observe(status.id)) {
            ModelDescription description = model_description(*lease);
            description.args             = status.arguments;
            description.failed           = status.failed;
            description.last_error       = status.last_error;
            models.push_back(std::move(description));
            continue;
        }
        models.push_back(ModelDescription{.id           = status.id,
                                          .status       = model_state_name(status.state),
                                          .loaded_facts = false,
                                          .args         = status.arguments,
                                          .aliases      = status.aliases,
                                          .failed       = status.failed,
                                          .last_error   = status.last_error});
    }
    res.set_content(make_models_list(models, unix_time_now()), "application/json");
}

namespace {

nlohmann::json residency_transition_json(const ninfer::ResidencyTransition& transition) {
    return nlohmann::json{{"ms", transition.seconds * 1000.0},
                          {"state_bytes", transition.state_bytes},
                          {"state_ms", transition.state_seconds * 1000.0},
                          {"weight_bytes", transition.weight_bytes},
                          {"weight_ms", transition.weight_seconds * 1000.0},
                          {"artifact_read_bytes", transition.artifact_read_bytes}};
}

nlohmann::json residency_json(const std::string& model, const ninfer::ResidencyStatus& status) {
    nlohmann::json out{
        {"object", "model.residency"},
        {"model", model},
        {"enabled", status.enabled},
        {"state", status.state == ninfer::ModelResidency::Suspended ? "suspended" : "resident"},
        {"auto_resume", status.auto_resume},
        {"releasable_device_bytes", status.releasable_device_bytes},
        {"host_snapshot_bytes", status.host_snapshot_bytes},
        {"host_reserved_bytes", status.host_reserved_bytes},
        {"suspend_count", status.suspend_count},
        {"resume_count", status.resume_count},
        {"last_suspend", status.last_suspend ? residency_transition_json(*status.last_suspend)
                                             : nlohmann::json(nullptr)},
        {"last_resume", status.last_resume ? residency_transition_json(*status.last_resume)
                                           : nlohmann::json(nullptr)},
        {"last_error", status.last_error.empty() ? nlohmann::json(nullptr)
                                                 : nlohmann::json(status.last_error)}};
    return out;
}

} // namespace

void HttpServer::handle_model_residency(const httplib::Request& req,
                                        httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_->resolve(id);
    if (!resolved) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    if (auto lease = registry_->observe(*resolved)) {
        nlohmann::json out = residency_json(*resolved, lease->service().residency());
        out["status"]      = model_state_name(registry_->status(*resolved)->state);
        res.set_content(out.dump(), "application/json");
        return;
    }
    res.set_content(nlohmann::json{{"object", "model.residency"},
                                   {"model", *resolved},
                                   {"status", "unloaded"},
                                   {"state", "unloaded"}}
                        .dump(),
                    "application/json");
}

namespace {

void fail_residency(httplib::Response& res, int status, std::string code, std::string message) {
    ApiError error;
    error.status  = status;
    error.type    = status >= 500 ? "server_error" : "invalid_request_error";
    error.code    = std::move(code);
    error.message = std::move(message);
    write_openai_error(res, error);
}

} // namespace

void HttpServer::handle_model_suspend(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_->resolve(id);
    if (!resolved) {
        fail_residency(res, 404, "model_not_found", "model '" + id + "' not found");
        return;
    }
    std::optional<bool> auto_resume;
    try {
        const nlohmann::json body =
            req.body.empty() ? nlohmann::json::object() : nlohmann::json::parse(req.body);
        if (!body.is_object()) { throw std::invalid_argument("body is not an object"); }
        for (const auto& [key, value] : body.items()) {
            if (key != "auto_resume" || !value.is_boolean()) {
                throw std::invalid_argument("unexpected field");
            }
            auto_resume = value.get<bool>();
        }
    } catch (const std::exception&) {
        fail_residency(res, 400, "invalid_request",
                       "request body must be a JSON object with at most a boolean auto_resume");
        return;
    }
    try {
        // Held when the body says so, or the model was started with --no-auto-resume.
        bool hold = auto_resume.has_value() && !*auto_resume;
        if (!auto_resume) {
            if (auto lease = registry_->observe(*resolved)) {
                hold = !lease->service().engine_options().suspend.auto_resume;
            }
        }
        registry_->sleep(*resolved, hold, /*wait=*/false);
        handle_model_residency(req, res);
    } catch (const ModelBusy& error) {
        fail_residency(res, 409, "model_busy", error.what());
    } catch (const ModelNotLoaded& error) {
        fail_residency(res, 400, "model_suspend_unavailable", error.what());
    } catch (const ninfer::RequestError& error) {
        if (error.kind() == ninfer::RequestErrorKind::Overloaded) {
            fail_residency(res, 409, "model_busy", error.what());
        } else {
            fail_residency(res, 503, "model_unavailable", error.what());
        }
    } catch (const std::invalid_argument& error) {
        fail_residency(res, 400, "model_suspend_disabled", error.what());
    } catch (const std::exception& error) {
        fail_residency(res, 500, "model_residency_error", error.what());
    }
}

void HttpServer::handle_model_resume(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_->resolve(id);
    if (!resolved) {
        fail_residency(res, 404, "model_not_found", "model '" + id + "' not found");
        return;
    }
    if (!req.body.empty()) {
        try {
            const nlohmann::json body = nlohmann::json::parse(req.body);
            if (!body.is_object() || !body.empty()) {
                throw std::invalid_argument("unexpected body");
            }
        } catch (const std::exception&) {
            fail_residency(res, 400, "invalid_request", "request body must be empty or {}");
            return;
        }
    }
    try {
        registry_->load(*resolved);
        handle_model_residency(req, res);
    } catch (const ModelLoadFailed& error) {
        fail_residency(res, 500, "model_residency_error", error.what());
    } catch (const std::exception& error) {
        fail_residency(res, 500, "model_residency_error", error.what());
    }
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_->resolve(id);
    if (!resolved) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    if (auto lease = registry_->observe(*resolved)) {
        res.set_content(make_model_object(model_description(*lease), unix_time_now()),
                        "application/json");
        return;
    }
    const auto status = registry_->status(*resolved);
    res.set_content(make_model_object(ModelDescription{.id     = *resolved,
                                                       .status = model_state_name(status->state),
                                                       .loaded_facts = false,
                                                       .args         = status->arguments,
                                                       .aliases      = status->aliases,
                                                       .failed       = status->failed,
                                                       .last_error   = status->last_error},
                                      unix_time_now()),
                    "application/json");
}

ModelRegistry::Lease HttpServer::lease_model(std::string_view model,
                                             const httplib::Request& req) const {
    std::optional<bool> autoload;
    if (req.has_param("autoload")) {
        const std::string value = req.get_param_value("autoload");
        autoload                = value == "1" || value == "true";
    }
    try {
        ModelRegistry::Lease lease = registry_->acquire(model, autoload);
        (void)model_facts(lease);
        return lease;
    } catch (const ModelUnknown& error) {
        ApiError api;
        api.status  = 404;
        api.param   = "model";
        api.code    = "model_not_found";
        api.message = error.what();
        throw ApiException(std::move(api));
    } catch (const ModelNotLoaded& error) {
        ApiError api;
        api.status  = 503;
        api.type    = "server_error";
        api.param   = "model";
        api.code    = "model_not_loaded";
        api.message = error.what();
        throw ApiException(std::move(api));
    } catch (const ModelLoadFailed& error) {
        ApiError api;
        api.status  = 503;
        api.type    = "server_error";
        api.param   = "model";
        api.code    = "model_load_failed";
        api.message = error.what();
        throw ApiException(std::move(api));
    }
}

std::string HttpServer::anthropic_model(const std::string& requested) const {
    return registry_->resolve(requested) ? requested : std::string();
}

ModelRegistry::Lease HttpServer::lease_query_model(const httplib::Request& req) const {
    const std::string model = req.has_param("model") ? req.get_param_value("model") : std::string();
    // Observing a model never loads or wakes it (llama.cpp exempts these endpoints too): a sleeping
    // model reports its state as it is.
    try {
        if (auto lease = registry_->observe(model)) { return std::move(*lease); }
    } catch (const ModelUnknown& error) {
        ApiError api;
        api.status  = 404;
        api.param   = "model";
        api.code    = "model_not_found";
        api.message = error.what();
        throw ApiException(std::move(api));
    }
    ApiError api;
    api.status  = 503;
    api.type    = "server_error";
    api.param   = "model";
    api.code    = "model_not_loaded";
    api.message = model.empty() ? "no model is loaded; name one with ?model="
                                : "model '" + model + "' is not loaded";
    throw ApiException(std::move(api));
}

HttpServer::ModelFacts HttpServer::model_facts(const ModelRegistry::Lease& lease) const {
    GenerationService& service = lease.service();
    {
        std::lock_guard lock(facts_mutex_);
        const auto found = facts_.find(lease.model());
        if (found != facts_.end() && found->second.service == &service) { return found->second; }
    }
    // memory_summary() takes the Engine execution lock: computed once per Engine instance.
    ModelFacts facts;
    facts.service                      = &service;
    facts.metadata                     = service.model_metadata();
    facts.vision                       = service.engine_options().enable_vision;
    facts.max_context                  = service.options().max_context;
    const ninfer::MemorySummary memory = service.memory_summary();
    facts.capacity = make_load_capacity(lease.model(), service.engine_options(), memory);
    request_jsonl_.write_server_start(service.options(), service.engine_options(),
                                      service.sampling_defaults(), lease.model(),
                                      service.load_summary(), memory);
    std::lock_guard lock(facts_mutex_);
    facts_[lease.model()] = facts;
    return facts;
}

void HttpServer::handle_router_models(const httplib::Request& req, httplib::Response& res) const {
    handle_models(req, res);
}

void HttpServer::handle_models_action(const httplib::Request& req, httplib::Response& res,
                                      std::string_view action) {
    std::string model;
    try {
        const nlohmann::json body = nlohmann::json::parse(req.body.empty() ? "{}" : req.body);
        if (!body.is_object() || !body.contains("model") || !body.at("model").is_string()) {
            throw std::invalid_argument("model");
        }
        model = body.at("model").get<std::string>();
    } catch (const std::exception&) {
        fail_residency(res, 400, "invalid_request",
                       "request body must be a JSON object with a string model");
        return;
    }
    const auto resolved = registry_->resolve(model);
    if (!resolved) {
        fail_residency(res, 404, "model_not_found", "model '" + model + "' not found");
        return;
    }
    try {
        if (action == "load") {
            registry_->load(*resolved);
        } else if (action == "unload") {
            registry_->unload(*resolved);
        } else {
            registry_->sleep(*resolved, false, /*wait=*/false);
        }
        res.set_content(R"({"success":true})", "application/json");
    } catch (const ModelBusy& error) {
        fail_residency(res, 409, "model_busy", error.what());
    } catch (const ModelNotLoaded& error) {
        fail_residency(res, 400, "model_not_loaded", error.what());
    } catch (const ModelLoadFailed& error) {
        fail_residency(res, 500, "model_load_failed", error.what());
    } catch (const std::exception& error) {
        fail_residency(res, 500, "model_residency_error", error.what());
    }
}

void HttpServer::handle_models_sse(const httplib::Request&, httplib::Response& res) {
    prepare_sse_response(res);
    auto cursor = std::make_shared<std::uint64_t>(0);
    res.set_chunked_content_provider(
        "text/event-stream", [this, cursor](std::size_t, httplib::DataSink& sink) -> bool {
            std::vector<ModelEvent> events;
            *cursor = registry_->events(*cursor, std::chrono::seconds(5), events);
            std::string out;
            for (const ModelEvent& event : events) {
                nlohmann::json data{{"status", model_state_name(event.state)}};
                if (event.failed) {
                    data["failed"] = true;
                    data["error"]  = event.error;
                }
                out += "data: " +
                       nlohmann::json{{"model", event.model},
                                      {"event", "model_status"},
                                      {"data", std::move(data)}}
                           .dump() +
                       "\n\n";
            }
            if (out.empty()) { out = ": keep-alive\n\n"; }
            return sink.write(out.data(), out.size());
        });
}

bool HttpServer::bind() {
    return server_.bind_to_port(options_.host, options_.port) &&
           (options_.stats_port == 0 ||
            stats_server_.bind_to_port(options_.host, options_.stats_port));
}

void HttpServer::start_stats_listener() {
    if (options_.stats_port == 0 || stats_listener_.joinable()) { return; }
    stats_listener_ = std::thread([this] {
        try {
            (void)stats_server_.listen_after_bind();
        } catch (const std::exception&) {}
    });
}

void HttpServer::stop_stats_listener() {
    if (!stats_listener_.joinable()) { return; }
    stats_server_.stop();
    stats_listener_.join();
}

void HttpServer::start_serving_during_startup() {
    if (startup_listener_.joinable()) {
        throw std::logic_error("HTTP startup listener is already running");
    }

    // The readiness gate lives in register_routes()'s single pre-routing handler, ahead of the
    // API-key check -- not here, so starting this listener never replaces (and thereby drops) the
    // auth/request-ID middleware for the remainder of the process's life.
    start_stats_listener();
    startup_listener_ = std::thread([this] {
        // listen_after_bind() blocks here for the whole life of the server, spanning the switch
        // from 503 to serving. stop() is what ends it. It can also throw before ever reaching
        // that loop -- the task queue's thread pool spawns its worker threads here, and
        // std::thread's constructor throws std::system_error under resource exhaustion. An
        // exception escaping a thread function is std::terminate, so it is caught and folded into
        // the same false result a synchronous listen() failure already produces.
        bool result = false;
        try {
            result = server_.listen_after_bind();
        } catch (const std::exception&) {
        }
        startup_listener_result_.store(result, std::memory_order_release);
    });
}

bool HttpServer::await_startup_listener() {
    if (startup_listener_.joinable()) { startup_listener_.join(); }
    return startup_listener_result_.load(std::memory_order_acquire);
}

void HttpServer::attach(ModelRegistry& registry) {
    if (registry_ != nullptr) { throw std::logic_error("HTTP model registry is already attached"); }
    registry_ = &registry;
    if (!registry.router()) { public_model_id_ = registry.status().front().id; }
    attached_at_ = std::chrono::steady_clock::now();
    // The server_start record of every model already loaded; later loads write theirs when a
    // request first leases them (model_facts).
    for (const ModelStatusSnapshot& status : registry.status()) {
        if (auto lease = registry.observe(status.id)) { (void)model_facts(*lease); }
    }
    // Release: everything above must be visible to a handler that observes ready_ as true. This is
    // the only write, and handlers acquire it in the pre-routing guard before touching registry_.
    ready_.store(true, std::memory_order_release);
}

// ENGINE WATCH. After an Engine-wide failure the Engine fails every queued request and refuses
// every new one, but never recovers, while /v1/models keeps answering: a single-model server would
// stay up as a dead endpoint. The watch polls the model while listen() runs and stops the accept
// loop the first time its Engine has failed, so main() exits with status 2 and a supervisor reloads
// the model. A router instead unloads a failed model, which its next request loads again.
void HttpServer::run_engine_watch() {
    constexpr auto kInterval = std::chrono::milliseconds(250);
    for (;;) {
        {
            std::unique_lock lock(watch_mutex_);
            if (watch_cv_.wait_for(lock, kInterval, [this] { return watch_stopping_; })) { return; }
        }
        bool failed = false;
        try {
            if (auto lease = registry_->observe(public_model_id_)) {
                failed = lease->service().has_failed();
            }
        } catch (...) {}
        if (failed) {
            engine_failed_.store(true, std::memory_order_release);
            operational_log_.engine_failure();
            server_.stop();
            return;
        }
    }
}

void HttpServer::stop_engine_watch() {
    if (!watch_thread_.joinable()) { return; }
    {
        std::lock_guard lock(watch_mutex_);
        watch_stopping_ = true;
    }
    watch_cv_.notify_one();
    watch_thread_.join();
}

bool HttpServer::listen() {
    if (registry_ == nullptr) { throw std::logic_error("HTTP model registry is not attached"); }
    if (!registry_->router() && public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (console_stats_) { console_stats_->show(); }
    try {
        start_stats_listener();
        if (options_.log_stats_interval_ms != 0) {
            stats_stopping_ = false;
            stats_thread_   = std::thread([this] { run_stats_reporter(); });
        }
        if (!registry_->router()) {
            watch_stopping_ = false;
            watch_thread_   = std::thread([this] { run_engine_watch(); });
        }
        // When the startup listener is running, the accept loop is already live on its thread and
        // has been since bind(); calling listen_after_bind() again would try to accept on the same
        // socket from two threads. Wait for that loop instead.
        const bool result =
            startup_listener_.joinable() ? await_startup_listener() : server_.listen_after_bind();
        stop_engine_watch();
        stop_stats_listener();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_engine_watch();
        stop_stats_listener();
        stop_stats_reporter();
        // Stop and join the startup listener before this exception unwinds past us. attach() has
        // already run by the time listen() can be called, so that thread is live against
        // registry_; the caller (main.cpp) destroys the attached registry, constructed after this
        // HttpServer, before this object -- leaving that thread dereferencing a dangling pointer
        // for however long stack unwinding takes if it is still running then.
        if (startup_listener_.joinable()) {
            server_.stop();
            startup_listener_.join();
        }
        throw;
    }
}

void HttpServer::stop() {
    stats_server_.stop();
    server_.stop();
    if (registry_ != nullptr) {
        registry_->for_each_service(
            [](const std::string&, GenerationService& service) { service.stop(); });
        registry_->stop();
    }
}

HttpServer::~HttpServer() {
    stop_stats_listener();
    if (startup_listener_.joinable()) {
        server_.stop();
        startup_listener_.join();
    }
    stop_stats_reporter();
}

} // namespace ninfer::serve
