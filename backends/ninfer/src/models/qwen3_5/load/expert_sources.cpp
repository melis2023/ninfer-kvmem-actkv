#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/expert_sources.h"

#include <algorithm>
#include <array>
#include <optional>

namespace ninfer::models::qwen3_5::loading {
namespace {

struct Span {
    artifact::ObjectHandle object;
    std::uint64_t offset, bytes;
};
using Planes = std::array<std::vector<Span>, 4>;

void append(std::vector<Span>& spans, Span span) {
    if (!spans.empty() && spans.back().object == span.object &&
        spans.back().offset + spans.back().bytes == span.offset) {
        spans.back().bytes += span.bytes;
    } else {
        spans.push_back(span);
    }
}

ExpertProjectionLayout layout(const artifact::Reader& reader,
                                std::span<const PendingWeight> pending,
                                std::span<const WeightId> parameters, std::uint32_t rows,
                                std::uint32_t columns, std::uint32_t divisor_rows) {
    const auto& first = pending[parameters.front().index].reference.binding.parts.front();
    const auto format = reader.geometry(first.object).format;
    return expert_projection_layout(format, rows, columns,
                                      format == QType::NVFP4 ? divisor_rows : 0);
}

Planes planes(const artifact::Reader& reader, std::span<const PendingWeight> pending,
               std::span<const WeightId> parameters, const ExpertProjectionLayout& slot) {
    Planes out;
    std::uint64_t row = 0;
    for (const auto id : parameters) {
        const auto& parameter = pending[id.index];
        for (const auto& use : parameter.uses) {
            if (use.hadamard_signs || use.input_columns) {
                throw artifact::ArtifactError(parameter.reference.name +
                                              ": native streamed experts require unrotated rows");
            }
        }
        for (const auto& part : parameter.reference.binding.parts) {
            const auto& g = reader.geometry(part.object);
            reader.validate_object(part.object);
            if (g.shape.size() != 2 || g.padded_columns != slot.columns ||
                part.begin % slot.columns || part.end % slot.columns) {
                throw artifact::ArtifactError(parameter.reference.name +
                                              ": native streamed expert needs complete rows");
            }
            const ExpertParentLayout parent{g.format, g.layout, g.shape[0], g.shape[1],
                                             g.high_offset, g.scale_offset, g.divisor_offset,
                                             g.divisor_count};
            const auto count = (part.end - part.begin) / slot.columns;
            const auto ranges = expert_projection_ranges(parent, part.begin / slot.columns,
                                                           count, slot, row);
            for (const auto& range : ranges) {
                unsigned plane = range.destination >= slot.divisors ? 3
                               : range.destination >= slot.scales ? 2
                               : slot.high_bytes && range.destination >= slot.high ? 1 : 0;
                append(out[plane], {part.object, range.source, range.bytes});
            }
            row += count;
        }
    }
    if (row != slot.rows) { throw artifact::ArtifactError("native expert rows are incomplete"); }
    return out;
}

ExpertBlob blob(const artifact::Reader& reader, const artifact::MaterializedArtifact& backing,
                 std::span<const Span> spans, ExpertResidency residency) {
    ExpertBlob out;
    for (const auto& span : spans) { out.bytes = expert_add(out.bytes, span.bytes); }
    if (residency == ExpertResidency::Host) {
        if (spans.size() != 1) {
            throw artifact::ArtifactError(
                "native host expert plane must be contiguous; disk residency accepts fragments");
        }
        const auto* parent = backing.pinned_parent(spans.front().object).data;
        out.pinned = parent + spans.front().offset;
        out.pinned_region = reinterpret_cast<std::uintptr_t>(parent);
    } else {
        for (const auto& span : spans) {
            const auto offset = artifact::object_offset(reader.directory().object(span.object));
            for (const auto& segment : reader.segments(expert_add(offset, span.offset), span.bytes)) {
                out.extents.push_back({segment.file_index, segment.file_offset, segment.bytes});
            }
        }
    }
    return out;
}

void add_projection(ExpertDeliveryLayer& layer, std::vector<ExpertBlob>& expert,
                     const artifact::Reader& reader, const artifact::MaterializedArtifact& backing,
                     const Planes& source, ExpertResidency residency, bool first) {
    const auto code_blob = expert.size();
    for (unsigned plane = 0; plane < 4; ++plane) {
        if (source[plane].empty()) {
            if (first) { layer.aliases.push_back({code_blob, 0}); }
            continue;
        }
        if (first) { layer.aliases.push_back({expert.size(), 0}); }
        expert.push_back(blob(reader, backing, source[plane], residency));
    }
}

void add_layer(NativeExpertSources& out, MoeWeights& moe, const artifact::Reader& reader,
                 std::span<const PendingWeight> pending,
                 const artifact::MaterializedArtifact& backing, ExpertResidency residency) {
    if (pending[moe.experts.front().gate.index].reference.residency ==
        artifact::Residency::Device) { return; }
    NativeExpertLayer layer;
    layer.rank = pending[moe.router.index].rank;
    const std::array gu{moe.experts.front().gate, moe.experts.front().up};
    const std::array down{moe.experts.front().down};
    layer.gate_up = layout(reader, pending, gu, 1024, 2048, 512);
    layer.down = layout(reader, pending, down, 2048, 512, 2048);
    for (const auto& expert : moe.experts) {
        const std::array gu_ids{expert.gate, expert.up};
        const std::array down_ids{expert.down};
        std::vector<ExpertBlob> payload;
        const bool first = layer.delivery.experts.empty();
        add_projection(layer.delivery, payload, reader, backing,
                         planes(reader, pending, gu_ids, layer.gate_up), residency, first);
        add_projection(layer.delivery, payload, reader, backing,
                         planes(reader, pending, down_ids, layer.down), residency, first);
        layer.delivery.experts.push_back(std::move(payload));
    }
    moe.delivery_layer = out.layers.size();
    out.layers.push_back(std::move(layer));
}
} // namespace

NativeExpertSources expert_sources(const artifact::Reader& reader,
                                    std::span<const PendingWeight> pending, ModelWeights& weights,
                                    const artifact::MaterializedArtifact& backing,
                                    const LoadOptions& options,
                                    const std::filesystem::path& artifact_path) {
    NativeExpertSources out;
    if (options.expert_residency == ExpertResidency::Device) { return out; }
    if (options.expert_residency == ExpertResidency::Disk) {
        if (artifact_path.empty()) {
            throw std::invalid_argument("disk expert residency requires the artifact entry path");
        }
        for (const auto& file : reader.directory().files) {
            out.files.push_back(file.path ? artifact_path.parent_path() / *file.path
                                           : artifact_path);
        }
    }
    for (auto& block : weights.text.layers) {
        add_layer(out, std::get<MoeWeights>(block.ffn), reader, pending, backing,
                    options.expert_residency);
    }
    if (weights.mtp) {
        add_layer(out, std::get<MoeWeights>(weights.mtp->layer.ffn), reader, pending, backing,
                    options.expert_residency);
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
