#include "serve/mcp_proxy.h"

#include "serve/http_server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <thread>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

httplib::Request proxy_request(const std::string& url, const httplib::Headers& headers) {
    httplib::Request request;
    request.method  = "POST";
    request.path    = kMcpProxyPath;
    request.headers = headers;
    request.params.emplace(kMcpProxyUrlParam, url);
    return request;
}

// The byte at stream offset i of the upstream reply.
char pattern_byte(std::size_t i) { return static_cast<char>((i * 131 + i / 4096) & 0xff); }

// An httplib server on a free loopback port, listening on its own thread until destroyed.
struct LoopbackServer {
    httplib::Server server;
    std::thread thread;
    int port = 0;

    void start() {
        port   = server.bind_to_any_port("127.0.0.1");
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    ~LoopbackServer() {
        server.stop();
        if (thread.joinable()) { thread.join(); }
    }
};

// An MCP server whose reply to POST /mcp is `total` pattern bytes in 64 KiB chunks, as fast as the
// relay reads them; `sent` counts the bytes its socket accepted.
struct StreamingUpstream {
    static constexpr std::size_t kChunk = 64 * 1024;
    LoopbackServer loopback;
    std::size_t total;
    std::atomic<std::size_t> sent{0};
    std::atomic<bool> ended{false};

    explicit StreamingUpstream(std::size_t bytes) : total(bytes) {
        loopback.server.Post("/mcp", [this](const httplib::Request&, httplib::Response& response) {
            response.set_chunked_content_provider(
                "text/event-stream",
                [this](std::size_t offset, httplib::DataSink& sink) {
                    if (offset >= total) {
                        sink.done();
                        return true;
                    }
                    std::string chunk(std::min(kChunk, total - offset), '\0');
                    for (std::size_t i = 0; i < chunk.size(); ++i) {
                        chunk[i] = pattern_byte(offset + i);
                    }
                    if (!sink.write(chunk.data(), chunk.size())) { return false; }
                    sent += chunk.size();
                    return true;
                },
                [this](bool) { ended = true; });
        });
        loopback.start();
    }
};

// This server's relay route alone.
struct Relay {
    LoopbackServer loopback;

    Relay() {
        const auto relay = [](const httplib::Request& request, httplib::Response& response) {
            relay_mcp_proxy(request, response);
        };
        loopback.server.Post(kMcpProxyPath, relay);
        loopback.server.Delete(kMcpProxyPath, relay);
        loopback.start();
    }
};

std::string relay_path(int upstream_port) {
    return std::string(kMcpProxyPath) + "?" + kMcpProxyUrlParam + "=" +
           httplib::encode_query_component("http://127.0.0.1:" + std::to_string(upstream_port) +
                                           "/mcp");
}

// POSTs to the relay for the upstream's /mcp; `receive` sees every chunk of the relayed body and
// returns false to hang up.
httplib::Result post_through(const Relay& relay, const StreamingUpstream& upstream,
                             httplib::ContentReceiverWithProgress receive) {
    httplib::Client client("127.0.0.1", relay.loopback.port);
    client.set_read_timeout(std::chrono::seconds(30));
    httplib::Request request;
    request.method = "POST";
    request.path   = relay_path(upstream.loopback.port);
    request.body   = R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})";
    request.set_header("Content-Type", "application/json");
    request.content_receiver = std::move(receive);
    return client.send(request);
}

// A browser that stops reading must stop the relay reading the MCP server: the relay holds at most
// kMcpProxyBufferBytes, and the sockets between the three hold a few MiB more. Unbounded, the relay
// takes the whole 256 MiB into memory within the pause. After it, every byte arrives in order.
int test_slow_reader_bounds_the_relay() {
    constexpr std::size_t kTotal        = std::size_t{256} << 20;
    constexpr std::size_t kStallCeiling = std::size_t{64} << 20;
    StreamingUpstream upstream(kTotal);
    Relay relay;
    std::size_t received          = 0;
    std::size_t sent_during_stall = 0;
    bool in_order                 = true;
    const httplib::Result result  = post_through(
        relay, upstream, [&](const char* data, std::size_t length, std::size_t, std::size_t) {
            if (received == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                sent_during_stall = upstream.sent.load();
            }
            for (std::size_t i = 0; i < length && in_order; ++i) {
                in_order = data[i] == pattern_byte(received + i);
            }
            received += length;
            return true;
        });
    int failures = 0;
    failures += check(result && result->status == 200, "the relayed stream did not complete");
    failures += check(sent_during_stall < kStallCeiling,
                      "the relay kept reading the MCP server while the browser did not read");
    failures +=
        check(received == kTotal && in_order, "the relayed body lost, added or reordered bytes");
    if (failures != 0) {
        std::cerr << "  sent during the stall " << sent_during_stall << ", received " << received
                  << " of " << kTotal << '\n';
    }
    return failures;
}

// A browser that hangs up while the relay waits for it to drain must release the MCP server: the
// waiting upstream reader wakes and the upstream connection closes.
int test_hang_up_releases_the_upstream() {
    StreamingUpstream upstream(std::size_t{256} << 20);
    Relay relay;
    bool first = true;
    (void)post_through(relay, upstream, [&](const char*, std::size_t, std::size_t, std::size_t) {
        if (!first) { return false; }
        first = false;
        // Long enough for the relay's buffer to fill and its reader to wait.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return true;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!upstream.ended && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return check(upstream.ended && upstream.sent < upstream.total,
                 "the MCP server's reply kept streaming after the browser hung up");
}

// Replies without a body (204 to the DELETE that ends a session, 202 to a notification) reach the
// browser with their status and headers, and leave its kept-alive connection fit for the next
// request.
int test_bodyless_replies_pass() {
    LoopbackServer upstream;
    upstream.server.Delete("/mcp", [](const httplib::Request&, httplib::Response& response) {
        response.status = 204;
        response.set_header("Mcp-Session-Id", "ended");
    });
    upstream.server.Post("/mcp", [](const httplib::Request&, httplib::Response& response) {
        response.status = 202;
    });
    upstream.start();
    Relay relay;
    httplib::Client client("127.0.0.1", relay.loopback.port);
    client.set_keep_alive(true);
    const std::string path         = relay_path(upstream.port);
    const httplib::Result ended    = client.Delete(path);
    const httplib::Result accepted = client.Post(path, "{}", "application/json");
    int failures                   = 0;
    failures += check(ended && ended->status == 204 && ended->body.empty() &&
                          ended->get_header_value("Mcp-Session-Id") == "ended",
                      "a 204 reply did not reach the browser as it was");
    failures += check(accepted && accepted->status == 202 && accepted->body.empty(),
                      "a bodyless 202 after a 204 on the same connection did not arrive");
    return failures;
}

int test_relay_requires_the_api_key() {
    const auto request = [](const std::string& method, const std::string& path) {
        httplib::Request r;
        r.method = method;
        r.path   = path;
        return r;
    };
    int failures = 0;
    failures += check(requires_api_key("key", true, request("POST", kMcpProxyPath)) &&
                          requires_api_key("key", true, request("GET", kMcpProxyPath)) &&
                          requires_api_key("key", false, request("DELETE", kMcpProxyPath)),
                      "the MCP relay was reachable without the configured API key");
    failures += check(!requires_api_key("", true, request("POST", kMcpProxyPath)),
                      "the relay asked for a key no one configured");
    failures += check(!requires_api_key("key", true, request("OPTIONS", kMcpProxyPath)) &&
                          !requires_api_key("key", true, request("GET", "/health")) &&
                          !requires_api_key("key", true, request("GET", "/")),
                      "a preflight, /health or a WebUI page required the API key");
    failures += check(requires_api_key("key", true, request("POST", "/v1/chat/completions")) &&
                          requires_api_key("key", false, request("GET", "/")),
                      "an API route, or GET / without a WebUI, skipped the API key");
    return failures;
}

} // namespace

int main() {
    int failures = 0;

    const McpProxyRequest plain = parse_mcp_proxy_target("http://127.0.0.1:12315/mcp");
    failures += check(plain.ok && plain.target.host == "127.0.0.1" && plain.target.port == 12315 &&
                          plain.target.path == "/mcp",
                      "host, port and path were not split out of the target url");

    const McpProxyRequest default_port = parse_mcp_proxy_target("http://example.test/mcp");
    failures += check(default_port.ok && default_port.target.port == 80,
                      "a target url without a port did not default to 80");

    const McpProxyRequest with_query =
        parse_mcp_proxy_target("http://127.0.0.1:8931/sse?session=abc&x=1");
    failures += check(with_query.ok && with_query.target.path == "/sse?session=abc&x=1",
                      "the query string was dropped from the forwarded path");

    const McpProxyRequest rooted = parse_mcp_proxy_target("http://127.0.0.1:12315");
    failures += check(rooted.ok && rooted.target.path == "/",
                      "a target url without a path did not forward to /");

    const McpProxyRequest fragment = parse_mcp_proxy_target("http://127.0.0.1:9/mcp#frag");
    failures += check(fragment.ok && fragment.target.path == "/mcp",
                      "the fragment was sent upstream instead of being dropped");

    const McpProxyRequest ipv6 = parse_mcp_proxy_target("http://[::1]:12315/mcp");
    failures += check(ipv6.ok && ipv6.target.host == "::1" && ipv6.target.port == 12315,
                      "a bracketed IPv6 literal was not unwrapped into host and port");

    const McpProxyRequest ipv6_default = parse_mcp_proxy_target("http://[::1]/mcp");
    failures += check(ipv6_default.ok && ipv6_default.target.host == "::1" &&
                          ipv6_default.target.port == 80,
                      "a bracketed IPv6 literal without a port was misparsed");

    // https must be refused outright: silently downgrading to http would send the
    // caller's Authorization header over the wire in the clear.
    failures += check(!parse_mcp_proxy_target("https://example.test/mcp").ok,
                      "an https target was accepted despite there being no TLS client");
    failures += check(!parse_mcp_proxy_target("example.test/mcp").ok,
                      "a schemeless target was accepted");
    failures += check(!parse_mcp_proxy_target("http://").ok, "a hostless target was accepted");
    failures += check(!parse_mcp_proxy_target("http://host:70000/mcp").ok,
                      "an out-of-range port was accepted");
    failures += check(!parse_mcp_proxy_target("http://host:abc/mcp").ok,
                      "a non-numeric port was accepted");
    failures += check(!parse_mcp_proxy_target("http://::1:12315/mcp").ok,
                      "an unbracketed IPv6 literal was accepted");
    failures += check(!parse_mcp_proxy_target("").ok, "an empty target was accepted");

    const httplib::Headers headers = {
        {std::string(kMcpProxyHeaderPrefix) + "Authorization", "Bearer secret"},
        {std::string(kMcpProxyHeaderPrefix) + "Accept", "text/event-stream"},
        // Framing belongs to the upstream hop; httplib::Client owns it.
        {std::string(kMcpProxyHeaderPrefix) + "Host", "wrong.test"},
        {std::string(kMcpProxyHeaderPrefix) + "Content-Length", "17"},
        // Unprefixed headers belong to the browser-to-server hop only.
        {"Authorization", "Bearer server-api-key"},
        {"User-Agent", "browser"},
    };
    const McpProxyRequest forwarded =
        parse_mcp_proxy_request(proxy_request("http://127.0.0.1:12315/mcp", headers));
    failures += check(forwarded.ok, "a well-formed proxy request was rejected");
    failures += check(forwarded.headers.size() == 2,
                      "the forwarded header set was not limited to the prefixed, non-framing ones");
    failures += check(forwarded.headers.find("Authorization") != forwarded.headers.end() &&
                          forwarded.headers.find("Authorization")->second == "Bearer secret",
                      "the target's Authorization header was not un-prefixed and forwarded");
    failures += check(forwarded.headers.find("Accept") != forwarded.headers.end(),
                      "a prefixed Accept header was not forwarded");
    failures += check(forwarded.headers.find("Host") == forwarded.headers.end() &&
                          forwarded.headers.find("Content-Length") == forwarded.headers.end(),
                      "upstream framing headers were forwarded");
    failures += check(forwarded.headers.find("User-Agent") == forwarded.headers.end(),
                      "an unprefixed browser header leaked upstream");

    httplib::Request no_url;
    no_url.method = "POST";
    no_url.path   = kMcpProxyPath;
    failures += check(!parse_mcp_proxy_request(no_url).ok,
                      "a proxy request without a url parameter was accepted");

    // The bug this whole path exists to avoid: relaying an upstream CORS header
    // next to this server's own makes the browser see "*, *" and reject the reply.
    failures += check(is_hop_by_hop_response_header("Access-Control-Allow-Origin") &&
                          is_hop_by_hop_response_header("access-control-expose-headers"),
                      "upstream CORS headers would have been copied onto the relayed response");
    failures += check(is_hop_by_hop_response_header("Content-Length") &&
                          is_hop_by_hop_response_header("Transfer-Encoding") &&
                          is_hop_by_hop_response_header("Connection"),
                      "upstream framing headers would have been copied onto a chunked response");
    failures += check(!is_hop_by_hop_response_header("Mcp-Session-Id") &&
                          !is_hop_by_hop_response_header("Content-Type") &&
                          !is_hop_by_hop_response_header("Cache-Control"),
                      "a header the MCP client needs was dropped from the relayed response");

    failures += check(header_name_is("Content-Type", "content-type") &&
                          header_name_is("content-TYPE", "content-type") &&
                          !header_name_is("Content-Length", "content-type"),
                      "case-insensitive header matching is wrong");

    failures += test_relay_requires_the_api_key();
    failures += test_slow_reader_bounds_the_relay();
    failures += test_hang_up_releases_the_upstream();
    failures += test_bodyless_replies_pass();

    if (failures == 0) { std::cout << "mcp proxy ok\n"; }
    return failures == 0 ? 0 : 1;
}
