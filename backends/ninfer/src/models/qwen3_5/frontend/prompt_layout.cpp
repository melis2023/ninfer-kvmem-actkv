#include "models/qwen3_5/frontend/prompt_layout.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::frontend {
namespace {

constexpr std::string_view kQwenStart   = "<|im_start|>";
constexpr std::string_view kQwenEnd     = "<|im_end|>";
// Gemma 4's turns: <|turn>role\n ... <turn|>, the assistant's role named "model".
constexpr std::string_view kGemmaStart  = "<|turn>";
constexpr std::string_view kGemmaEnd    = "<turn|>";
// Flint's turns: a role marker and the turn's text up to the next one; no marker closes a turn.
struct FlintRole {
    std::string_view marker;
    ChatRole role;
};
constexpr FlintRole kFlintRoles[] = {{"<|system|>", ChatRole::System},
                                     {"<|developer|>", ChatRole::Developer},
                                     {"<|user|>", ChatRole::User},
                                     {"<|assistant|>", ChatRole::Assistant},
                                     {"<|observation|>", ChatRole::Tool}};
constexpr std::string_view kVisionStart = "<|vision_start|>";
constexpr std::string_view kVisionEnd   = "<|vision_end|>";

bool template_bytes(const text::TemplateOutput& output, std::size_t begin, std::size_t end) {
    return !text::overlaps(output.literal_spans, begin, end);
}

std::size_t find_marker(const text::TemplateOutput& output, std::string_view marker,
                        std::size_t begin) {
    for (auto pos = output.text.find(marker, begin); pos != std::string::npos;
         pos      = output.text.find(marker, pos + marker.size())) {
        if (template_bytes(output, pos, pos + marker.size())) return pos;
    }
    return std::string::npos;
}

// The first Flint role marker at or after `begin`; null where none follows.
const FlintRole* find_flint_role(const text::TemplateOutput& output, std::size_t begin,
                                 std::size_t& at) {
    const FlintRole* found = nullptr;
    at                     = std::string::npos;
    for (const FlintRole& role : kFlintRoles) {
        const std::size_t pos = find_marker(output, role.marker, begin);
        if (pos < at) {
            at    = pos;
            found = &role;
        }
    }
    return found;
}

// Flint's layout: each turn ends where the next begins, which closes it, since a role marker is a
// special token a turn's text cannot run into. Its template writes no media.
PromptLayout inspect_flint_layout(const text::TemplateOutput& output,
                                  std::span<const Modality> media) {
    if (!media.empty()) { throw std::invalid_argument("Flint's chat template takes no media"); }
    PromptLayout result;
    const std::string_view source = output.text;
    std::size_t pos               = 0;
    const FlintRole* role         = find_flint_role(output, 0, pos);
    while (role != nullptr) {
        const std::size_t content_begin = pos + role->marker.size();
        std::size_t next                = std::string::npos;
        const FlintRole* following      = find_flint_role(output, content_begin, next);
        const bool closed               = following != nullptr;
        const std::size_t content_end   = closed ? next : source.size();
        result.messages.push_back(
            {role->role, pos, content_begin, content_end, content_end, closed});
        if (role->role == ChatRole::Assistant) {
            result.execution_boundaries.push_back(content_begin);
            if (!closed) {
                // A final assistant prefix that opened a reasoning block and left it open.
                const auto body = source.substr(content_begin, content_end - content_begin);
                result.starts_in_reasoning =
                    body.starts_with("<think>") &&
                    template_bytes(output, content_begin, content_begin + 7) &&
                    find_marker(output, "</think>", content_begin + 7) == std::string::npos;
            }
        }
        role = following;
        pos  = next;
    }
    return result;
}

} // namespace

std::optional<text::ByteSpan> unique_output_region(const text::TemplateOutput& output,
                                                   std::uint32_t tag) {
    std::optional<text::ByteSpan> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag) continue;
        if (result) return std::nullopt;
        result = text::ByteSpan{region.begin, region.end};
    }
    return result;
}

std::optional<std::size_t> source_boundary(const text::TemplateOutput& output, std::uint32_t tag,
                                           std::size_t offset) {
    std::optional<std::size_t> result;
    for (const auto& region : output.regions) {
        if (region.tag != tag || !region.source_offset) continue;
        const auto begin = *region.source_offset;
        if (offset < begin || offset - begin > region.end - region.begin) continue;
        const auto mapped = region.begin + offset - begin;
        if (result && *result != mapped) return std::nullopt;
        result = mapped;
    }
    for (const auto& mapping : output.boundary_mappings) {
        if (mapping.tag != tag || offset < mapping.source_begin || offset > mapping.source_end)
            continue;
        if (result && *result != mapping.offset) return std::nullopt;
        result = mapping.offset;
    }
    return result;
}

PromptLayout inspect_prompt_layout(const text::TemplateOutput& output,
                                   std::span<const Modality> media) {
    std::size_t flint_at = std::string::npos;
    if (find_marker(output, kQwenStart, 0) == std::string::npos &&
        find_marker(output, kGemmaStart, 0) == std::string::npos &&
        find_flint_role(output, 0, flint_at) != nullptr) {
        return inspect_flint_layout(output, media);
    }
    PromptLayout result;
    const std::string_view source = output.text;
    const bool gemma =
        find_marker(output, kQwenStart, 0) == std::string::npos &&
        find_marker(output, kGemmaStart, 0) != std::string::npos;
    const std::string_view kStart = gemma ? kGemmaStart : kQwenStart;
    const std::string_view kEnd   = gemma ? kGemmaEnd : kQwenEnd;
    for (auto pos = find_marker(output, kStart, 0); pos != std::string::npos;) {
        const auto header_end = source.find('\n', pos + kStart.size());
        if (header_end == std::string::npos) break;
        const auto role = source.substr(pos + kStart.size(), header_end - pos - kStart.size());
        ChatRole parsed;
        if (role == "assistant" || (gemma && role == "model"))
            parsed = ChatRole::Assistant;
        else if (role == "user")
            parsed = ChatRole::User;
        else if (role == "system")
            parsed = ChatRole::System;
        else if (role == "developer")
            parsed = ChatRole::Developer;
        else if (role == "tool")
            parsed = ChatRole::Tool;
        else {
            pos = find_marker(output, kStart, header_end + 1);
            continue;
        }
        const auto next  = find_marker(output, kStart, header_end + 1);
        const auto close = find_marker(output, kEnd, header_end + 1);
        const bool closed =
            close != std::string::npos && (next == std::string::npos || close < next);
        const auto content_end = closed ? close : next == std::string::npos ? source.size() : next;
        auto end               = closed ? close + kEnd.size() : content_end;
        if (closed && end < source.size() && source[end] == '\n') ++end;
        result.messages.push_back({parsed, pos, header_end + 1, content_end, end, closed});
        if (parsed == ChatRole::Assistant) {
            result.execution_boundaries.push_back(header_end + 1);
            constexpr std::string_view open = "<think>\n";
            const auto body = source.substr(header_end + 1, content_end - header_end - 1);
            if (body.starts_with(open) &&
                template_bytes(output, header_end + 1, header_end + 1 + open.size())) {
                result.execution_boundaries.push_back(header_end + 1 + open.size());
                const auto reasoning_close = find_marker(
                    output, kCanonicalReasoningCloseSerialization, header_end + 1 + open.size());
                if (reasoning_close != std::string::npos && reasoning_close < content_end) {
                    result.execution_boundaries.push_back(
                        reasoning_close + kCanonicalReasoningCloseSerialization.size());
                }
            }
            if (!closed && next == std::string::npos) {
                // Only the final assistant prefix controls the initial output channel.
                const auto open_end =
                    body.starts_with("<think>") &&
                            template_bytes(output, header_end + 1, header_end + 1 + 7)
                        ? header_end + 1 + 7
                        : std::string::npos;
                const auto close_think = open_end == std::string::npos
                                             ? std::string::npos
                                             : find_marker(output, "</think>", open_end);
                result.starts_in_reasoning =
                    open_end != std::string::npos && close_think == std::string::npos;
            }
        }
        pos = next;
    }

    // Gemma 4's template writes one <|image|> or <|video|> per media input, in input order; the
    // processor adds the begin and end markers around the soft tokens it expands it into.
    if (gemma) {
        for (std::size_t pos = 0; pos < source.size();) {
            const auto image = find_marker(output, "<|image|>", pos);
            const auto video = find_marker(output, "<|video|>", pos);
            const auto pad   = std::min(image, video);
            if (pad == std::string::npos) break;
            const auto modality = pad == image ? Modality::Image : Modality::Video;
            constexpr std::size_t size = 9; // <|image|> and <|video|>
            const auto index       = result.media_placeholders.size();
            if (index >= media.size() || media[index] != modality) {
                throw std::invalid_argument(
                    "chat template media placeholders must match input count, type and order");
            }
            result.media_placeholders.push_back({{pad, pad + size}, modality, index});
            pos = pad + size;
        }
        if (result.media_placeholders.size() != media.size()) {
            throw std::invalid_argument("chat template omitted an input media placeholder");
        }
        return result;
    }
    // Each media input contributes one complete Qwen placeholder in input order.
    for (std::size_t pos = 0; pos < source.size();) {
        const auto image = find_marker(output, "<|image_pad|>", pos);
        const auto video = find_marker(output, "<|video_pad|>", pos);
        const auto pad   = std::min(image, video);
        if (pad == std::string::npos) break;
        const auto modality = pad == image ? Modality::Image : Modality::Video;
        const std::string_view token =
            modality == Modality::Image ? "<|image_pad|>" : "<|video_pad|>";
        if (pad < kVisionStart.size() ||
            source.substr(pad - kVisionStart.size(), kVisionStart.size()) != kVisionStart ||
            source.substr(pad + token.size(), kVisionEnd.size()) != kVisionEnd ||
            !template_bytes(output, pad - kVisionStart.size(),
                            pad + token.size() + kVisionEnd.size())) {
            throw std::invalid_argument(
                "chat template media pad requires a complete Qwen vision wrapper");
        }
        const auto index = result.media_placeholders.size();
        if (index >= media.size() || media[index] != modality) {
            throw std::invalid_argument(
                "chat template media placeholders must match input count, type and order");
        }
        result.media_placeholders.push_back({{pad, pad + token.size()}, modality, index});
        pos = pad + token.size();
    }
    if (result.media_placeholders.size() != media.size()) {
        throw std::invalid_argument("chat template omitted an input media placeholder");
    }
    return result;
}

} // namespace ninfer::models::qwen3_5::frontend
