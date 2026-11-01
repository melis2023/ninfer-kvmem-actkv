#include "models/qwen3_5/frontend/flint_output.h"

#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <cctype>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

std::int32_t single_token(const Tokenizer& tokenizer, std::string_view text) {
    const std::vector<int> ids = tokenizer.encode(text);
    return ids.size() == 1 ? ids.front() : -1;
}

void skip_space(std::string_view text, std::size_t& at) {
    while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at])) != 0) { ++at; }
}

// The text between `open` at `at` and the next `close`, moving `at` past the close.
bool take_tagged(std::string_view text, std::size_t& at, std::string_view open,
                 std::string_view close, std::string_view& out) {
    skip_space(text, at);
    if (!text.substr(at).starts_with(open)) { return false; }
    const std::size_t begin = at + open.size();
    const std::size_t end   = text.find(close, begin);
    if (end == std::string_view::npos) { return false; }
    out = text.substr(begin, end - begin);
    at  = end + close.size();
    return true;
}

// The length of the longest suffix of `text` that begins `marker` (shorter than the marker).
std::size_t partial_suffix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

} // namespace

FlintMarkers FlintMarkers::from(const Tokenizer& tokenizer) {
    FlintMarkers out;
    out.think_open  = single_token(tokenizer, kFlintThinkOpen);
    out.think_close = single_token(tokenizer, kFlintThinkClose);
    return out;
}

bool parse_flint_call(std::string_view region, const ToolCallOutputContract& contract,
                      GeneratedToolCall& out, ToolCallParseDiagnostics& diagnostics) {
    // The name runs to the first argument (whitespace before it is not the name's), or to the end.
    const std::size_t first = region.find(kFlintKeyOpen);
    std::string_view name   = region.substr(0, first);
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front())) != 0) {
        name.remove_prefix(1);
    }
    while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())) != 0) {
        name.remove_suffix(1);
    }
    if (name.empty() || name.find('<') != std::string_view::npos) { return false; }
    std::vector<std::pair<std::string_view, std::string_view>> arguments;
    std::size_t at = first == std::string_view::npos ? region.size() : first;
    for (;;) {
        skip_space(region, at);
        if (at == region.size()) { break; }
        std::string_view key, value;
        if (!take_tagged(region, at, kFlintKeyOpen, kFlintKeyClose, key) ||
            !take_tagged(region, at, kFlintValueOpen, kFlintValueClose, value) || key.empty()) {
            return false;
        }
        arguments.emplace_back(key, value);
    }
    out = normalize_untyped_tool_call(name, arguments, contract, diagnostics);
    return true;
}

FlintToolCallDecoder::FlintToolCallDecoder(std::shared_ptr<const ToolCallOutputContract> contract)
    : contract_(std::move(contract)) {
    if (contract_ == nullptr || contract_->forced_tool_name.empty()) { return; }
    // The generation prompt ends with `<tool_call>NAME`, so generation resumes after the name and
    // the region the reader sees has to begin where the prompt left off.
    region_                  = contract_->forced_tool_name;
    seeded_bytes_            = region_.size();
    in_call_                 = true;
    diagnostics_.marker_seen = true;
}

std::string FlintToolCallDecoder::feed(std::string_view text) {
    std::string published;
    pending_.append(text);
    for (;;) {
        if (!in_call_) {
            const std::size_t open = pending_.find(kFlintCallOpen);
            if (open == std::string::npos) {
                const std::size_t hold = partial_suffix(pending_, kFlintCallOpen);
                published.append(pending_, 0, pending_.size() - hold);
                pending_.erase(0, pending_.size() - hold);
                return published;
            }
            published.append(pending_, 0, open);
            pending_.erase(0, open + kFlintCallOpen.size());
            in_call_                 = true;
            diagnostics_.marker_seen = true;
            region_.clear();
            seeded_bytes_ = 0;
        }
        const std::size_t close = pending_.find(kFlintCallClose);
        if (close == std::string::npos) {
            const std::size_t hold = partial_suffix(pending_, kFlintCallClose);
            region_.append(pending_, 0, pending_.size() - hold);
            pending_.erase(0, pending_.size() - hold);
            return published;
        }
        region_.append(pending_, 0, close);
        pending_.erase(0, close + kFlintCallClose.size());
        in_call_ = false;
        GeneratedToolCall call;
        if (parse_flint_call(region_, *contract_, call, diagnostics_)) {
            calls_.push_back(std::move(call));
            ++diagnostics_.structured_call_count;
        } else {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            // Only what the model produced: a forced call's head came from the prompt.
            if (seeded_bytes_ == 0) { published.append(kFlintCallOpen); }
            published.append(region_, seeded_bytes_, std::string::npos);
            published.append(kFlintCallClose);
        }
        region_.clear();
        seeded_bytes_ = 0;
    }
}

FlintToolCallDecoder::Terminal FlintToolCallDecoder::finish() {
    Terminal out;
    if (in_call_) {
        region_.append(pending_);
        GeneratedToolCall call;
        if (parse_flint_call(region_, *contract_, call, diagnostics_)) {
            calls_.push_back(std::move(call));
            ++diagnostics_.structured_call_count;
            // A forced turn that ends after its last argument is a complete call missing only its
            // closing marker. Supplying that marker invents no argument byte; anything less
            // complete does not parse and stays text.
            if (seeded_bytes_ != 0) { diagnostics_.forced_call_closed = true; }
        } else {
            diagnostics_.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            if (seeded_bytes_ == 0) { out.content.append(kFlintCallOpen); }
            out.content.append(region_, seeded_bytes_, std::string::npos);
        }
    } else {
        out.content = pending_;
    }
    pending_.clear();
    region_.clear();
    seeded_bytes_   = 0;
    in_call_        = false;
    out.tool_calls  = std::move(calls_);
    out.diagnostics = diagnostics_;
    return out;
}

} // namespace ninfer::models::qwen3_5::frontend
