#include "models/qwen3_5/load/bindings.h"

#include "artifact/transcode.h"
#include "artifact/views.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::loading {
#if defined(NINFER_SM8X_COMPAT)
namespace {

#    if defined(NINFER_SM120_NVFP4)
constexpr bool kNvfp4A4Built = true;
#    else
constexpr bool kNvfp4A4Built = false;
#    endif
#    if defined(NINFER_SM120_FP8)
constexpr bool kFp8A8Built = true;
#    else
constexpr bool kFp8A8Built = false;
#    endif

bool stored_as(const artifact::Binder& binder, const artifact::ParameterReference& reference,
               QType format) {
    for (const auto& part : reference.binding.parts) {
        if (binder.geometry(part.object).format != format) { return false; }
    }
    return !reference.binding.parts.empty();
}

bool keeps_permission(const artifact::Binder& binder, const artifact::ParameterReference& reference,
                      ops::LinearPolicy policy) {
    if (ops::allows_a8(policy) && stored_as(binder, reference, QType::T5_G128_FP16)) {
        return true; // Integer tensor cores; independent of FP8/FP4 architecture support.
    }
    switch (policy) {
    case ops::LinearPolicy::AllowA4:
        return kNvfp4A4Built && stored_as(binder, reference, QType::NVFP4);
    case ops::LinearPolicy::AllowA8:
        return kFp8A8Built && stored_as(binder, reference, QType::FP8_E4M3FN_ROW_BF16);
    default:
        return false;
    }
}

} // namespace
#endif

void apply_rank_policies(Bindings& bindings, std::span<const ops::DeviceTier> rank_tiers) {
    // A build without 120a narrowed these permissions when it bound the weights.
    if constexpr (!ops::kBuildBlackwell) {
        (void)bindings;
        (void)rank_tiers;
        return;
    }
    for (PendingWeight& weight : bindings.weights) {
        if (weight.rank >= rank_tiers.size()) {
            throw std::logic_error("rank policies: no device tier for pipeline rank " +
                                   std::to_string(weight.rank));
        }
        if (rank_tiers[weight.rank] == ops::DeviceTier::Blackwell) { continue; }
        const auto& parts = weight.reference.binding.parts;
        bool t5 = !parts.empty();
        for (const auto& part : parts) {
            t5 = t5 && bindings.binder.reader().geometry(part.object).format == QType::T5_G128_FP16;
        }
        if (t5) { continue; }
        for (WeightUse& use : weight.uses) {
            if (use.policy == ops::LinearPolicy::AllowA8 ||
                use.policy == ops::LinearPolicy::AllowA4) {
                use.policy = ops::LinearPolicy::A16Only;
            }
        }
    }
}

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format,
                             artifact::Residency residency) {
    if (parameters_.contains(name)) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
    PendingWeight pending;
    pending.reference = binder.parameter(name, std::move(shape), residency, exact_format);
    add_uses(pending, name, inputs);
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(binder.object_id(part.object));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::piece(const std::string& name, const artifact::Shape& shape,
                         const std::vector<std::string>& inputs, std::optional<QType> exact_format,
                         std::size_t rank, const ParameterPiece& piece) {
    const auto found = binder.reader().directory().bindings.find(name);
    if (found == binder.reader().directory().bindings.end()) {
        throw artifact::ArtifactError("missing logical parameter " + name);
    }
    const artifact::Binding& whole = found->second;
    if (shape.empty() || shape.size() > 2 || weight_element_count(shape) != whole.elements) {
        throw artifact::ArtifactError(name + ": logical shape differs from Binding coverage");
    }
    const std::uint64_t columns = shape.back();
    const bool matrix           = shape.size() == 2;
    if (piece.ranges.empty() || (piece.cut == ParameterPiece::Cut::Rows &&
                                 (!matrix || piece.ranges.size() != 1))) {
        throw std::invalid_argument(name + ": a row piece takes one range of a matrix's rows");
    }
    // A matrix whose input is gathered (`input_columns`) stores its columns in gathered order: a
    // piece of logical columns [lo, hi) is the stored columns that read an element of it, and its
    // own gather is those columns' entries less lo.
    std::optional<artifact::Binding> gather;
    for (const auto& input : inputs) {
        for (const auto& [role, binding] : binder.use(name, input).auxiliaries) {
            if (role == "input_columns") { gather = binding; }
        }
    }
    ParameterPiece stored_piece = piece;
    if (gather && piece.cut == ParameterPiece::Cut::Columns) {
        if (!matrix || piece.ranges.size() != 1) {
            throw std::invalid_argument(name + ": a gathered input splits by one range of columns");
        }
        const auto order        = binder.values(*gather, QType::INT32).integers();
        const auto [lo, hi]     = piece.ranges.front();
        stored_piece.ranges.clear();
        if (order.size() != columns) {
            throw artifact::ArtifactError(name + ": the input gather does not cover the columns");
        }
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto element = static_cast<std::uint64_t>(order[column]);
            if (element < lo || element >= hi) { continue; }
            if (!stored_piece.ranges.empty() && stored_piece.ranges.back().second == column) {
                ++stored_piece.ranges.back().second;
            } else {
                stored_piece.ranges.emplace_back(column, column + 1);
            }
        }
    }
    artifact::Binding binding;
    std::uint64_t logical   = 0; // the first logical row (matrix) or column (vector) of a part
    std::uint64_t picked    = 0; // rows (row piece) or columns of the piece
    for (const auto& part : whole.parts) {
        const auto stored = binder.reader().geometry(part.object);
        if (exact_format && stored.format != *exact_format) {
            throw artifact::ArtifactError(name +
                                          ": representation does not match its mathematical value type");
        }
        if (matrix && (stored.shape.size() != 2 || stored.shape[1] != columns ||
                       part.begin % columns != 0 || part.end % columns != 0)) {
            throw artifact::ArtifactError(name + ": a tensor-parallel piece needs parts of whole rows");
        }
        if (!matrix && stored.shape.size() != 1) {
            throw artifact::ArtifactError(name + ": a vector piece needs vector parts");
        }
        const std::uint64_t first = matrix ? part.begin / columns : part.begin;
        const std::uint64_t count = matrix ? (part.end - part.begin) / columns : part.end - part.begin;
        artifact::ObjectSelection selection;
        if (piece.cut == ParameterPiece::Cut::Rows) {
            const auto [begin, end] = piece.ranges.front();
            const std::uint64_t lo  = std::max(begin, logical);
            const std::uint64_t hi  = std::min(end, logical + count);
            if (lo < hi) {
                selection.row_begin = first + lo - logical;
                selection.row_end   = first + hi - logical;
                selection.columns   = {{0, columns}};
            }
        } else if (matrix) {
            selection.row_begin = first;
            selection.row_end   = first + count;
            selection.columns   = stored_piece.ranges;
        } else {
            selection.row_begin = 0;
            selection.row_end   = 1;
            for (const auto& [begin, end] : stored_piece.ranges) {
                const std::uint64_t lo = std::max(begin, logical);
                const std::uint64_t hi = std::min(end, logical + count);
                if (lo < hi) { selection.columns.emplace_back(first + lo - logical, first + hi - logical); }
            }
        }
        logical += count;
        if (selection.columns.empty()) { continue; }
        const artifact::ObjectHandle handle =
            binder.select_device(part.object, std::move(selection), rank);
        const auto elements = binder.geometry(handle).elements;
        binding.parts.push_back({handle, 0, elements});
        binding.elements += elements;
    }
    artifact::Shape piece_shape;
    if (piece.cut == ParameterPiece::Cut::Rows) {
        picked      = piece.ranges.front().second - piece.ranges.front().first;
        piece_shape = {picked, columns};
    } else {
        for (const auto& [begin, end] : piece.ranges) { picked += end - begin; }
        piece_shape = matrix ? artifact::Shape{shape[0], picked} : artifact::Shape{picked};
    }
    if (binding.elements != weight_element_count(piece_shape)) {
        throw artifact::ArtifactError(name + ": a tensor-parallel piece leaves the parameter");
    }
    PendingWeight pending;
    pending.reference = {name + "@" + std::to_string(rank), std::move(piece_shape),
                         std::move(binding), artifact::Residency::Device};
    add_uses(pending, name, inputs, piece.cut == ParameterPiece::Cut::Rows);
    for (WeightUse& use : pending.uses) {
        if (use.hadamard_signs) {
            throw std::invalid_argument(name + "@" + use.input +
                                        ": tensor parallelism cannot split a rotated input");
        }
        // A row piece reads the whole input and keeps the whole gather.
        if (gather && piece.cut == ParameterPiece::Cut::Columns) {
            use.input_columns = gather_piece(*gather, stored_piece.ranges,
                                             piece.ranges.front().first, rank, name + "@" + use.input);
        }
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(binder.object_id(part.object));
    }
    pending.rank = rank;
    const WeightId id{weights.size()};
    weights.push_back(std::move(pending));
    return id;
}

void Bindings::add_uses(PendingWeight& pending, const std::string& name,
                        const std::vector<std::string>& inputs, bool whole_gathers) {
    for (const auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        result.input = input;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
#if defined(NINFER_SM8X_COMPAT)
        // sm_86/sm_89 have no FP8 or FP4 tensor cores. A stored permission for A8/A4 activations
        // is an upper bound, not a requirement, so FP8 and NVFP4 weights run their A16 routes,
        // which dequantize the stored codes before the matmul. A 120a build on this path keeps the
        // FP8 A8 and NVFP4 W4A4 units, and there such a weight keeps its permission.
        if (!keeps_permission(binder, pending.reference, result.policy)) {
            result.policy = ops::LinearPolicy::A16Only;
        }
#endif
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role == "hadamard_signs") {
                if (pending.reference.shape.size() != 2) {
                    throw artifact::ArtifactError(name + "@" + input +
                                                  ": Hadamard signs require a matrix");
                }
                result.hadamard_signs =
                    hadamard_signs(binding, pending.reference.shape[1], name + "@" + input);
                continue;
            }
            if (role == "input_columns") {
                if (!whole_gathers) { continue; }
                if (pending.reference.shape.size() != 2) {
                    throw artifact::ArtifactError(name + "@" + input +
                                                  ": an input gather requires a matrix");
                }
                result.input_columns =
                    input_columns(binding, pending.reference.shape[1], name + "@" + input);
                continue;
            }
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        pending.uses.push_back(std::move(result));
    }
}

WeightId Bindings::hadamard_signs(const artifact::Binding& binding, std::uint64_t width,
                                  const std::string& use) {
    if (width == 0 || width % 1024 != 0) {
        throw artifact::ArtifactError(use + ": Hadamard rotation needs a multiple of 1024 columns");
    }
    std::string key = std::to_string(width);
    for (const auto& part : binding.parts) {
        const auto& object = binder.reader().directory().object(part.object);
        key += "|" + artifact::object_id(object) + ":" + std::to_string(part.begin) + "-" +
               std::to_string(part.end);
    }
    if (const auto found = signs_.find(key); found != signs_.end()) { return found->second; }
    PendingWeight pending;
    pending.reference = binder.binding("hadamard_signs/" + std::to_string(signs_.size()), binding,
                                       {width}, artifact::Residency::Device, QType::BF16);
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    weights.push_back(std::move(pending));
    signs_.emplace(std::move(key), id);
    return id;
}

WeightId Bindings::input_columns(const artifact::Binding& binding, std::uint64_t width,
                                 const std::string& use) {
    std::string key = std::to_string(width);
    for (const auto& part : binding.parts) {
        const auto& object = binder.reader().directory().object(part.object);
        key += "|" + artifact::object_id(object) + ":" + std::to_string(part.begin) + "-" +
               std::to_string(part.end);
    }
    if (const auto found = columns_.find(key); found != columns_.end()) { return found->second; }
    PendingWeight pending;
    pending.reference = binder.binding("input_columns/" + std::to_string(columns_.size()), binding,
                                       {width}, artifact::Residency::Device, QType::INT32);
    const auto columns = binder.values(pending.reference.binding, QType::INT32).integers();
    std::vector<bool> seen(width, false);
    for (const auto column : columns) {
        if (column < 0 || std::uint64_t(column) >= width || seen[column]) {
            throw artifact::ArtifactError(use + ": the input gather must permute [0, K)");
        }
        seen[column] = true;
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    weights.push_back(std::move(pending));
    columns_.emplace(std::move(key), id);
    return id;
}

WeightId Bindings::gather_piece(const artifact::Binding& gather,
                                const std::vector<std::pair<std::uint64_t, std::uint64_t>>& columns,
                                std::uint64_t base, std::size_t rank, const std::string& use) {
    std::string key = std::to_string(rank) + "|" + std::to_string(base);
    for (const auto& part : gather.parts) {
        key += "|" + binder.object_id(part.object) + ":" + std::to_string(part.begin) + "-" +
               std::to_string(part.end);
    }
    for (const auto& [begin, end] : columns) {
        key += "|" + std::to_string(begin) + "-" + std::to_string(end);
    }
    if (const auto found = gather_pieces_.find(key); found != gather_pieces_.end()) {
        return found->second;
    }
    // The selected entries, then their check: less `base`, they must permute [0, width).
    const auto order = binder.values(gather, QType::INT32).integers();
    std::uint64_t width = 0;
    std::vector<bool> seen;
    for (const auto& [begin, end] : columns) { width += end - begin; }
    seen.assign(width, false);
    for (const auto& [begin, end] : columns) {
        for (std::uint64_t column = begin; column < end; ++column) {
            const auto local = static_cast<std::uint64_t>(order[column]) - base;
            if (local >= width || seen[local]) {
                throw artifact::ArtifactError(use + ": the piece's gather must permute its columns");
            }
            seen[local] = true;
        }
    }
    if (base > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw artifact::ArtifactError(use + ": the gather base exceeds int32");
    }
    artifact::Binding binding;
    std::uint64_t logical = 0;
    for (const auto& part : gather.parts) {
        const std::uint64_t count = part.end - part.begin;
        artifact::ObjectSelection selection{.row_begin = 0, .row_end = 1, .columns = {},
                                            .rebase = static_cast<std::int32_t>(base)};
        for (const auto& [begin, end] : columns) {
            const std::uint64_t lo = std::max(begin, logical);
            const std::uint64_t hi = std::min(end, logical + count);
            if (lo < hi) {
                selection.columns.emplace_back(part.begin + lo - logical, part.begin + hi - logical);
            }
        }
        logical += count;
        if (selection.columns.empty()) { continue; }
        const artifact::ObjectHandle handle =
            binder.select_device(part.object, std::move(selection), rank);
        const auto elements = binder.geometry(handle).elements;
        binding.parts.push_back({handle, 0, elements});
        binding.elements += elements;
    }
    PendingWeight pending;
    pending.reference = {"input_columns/piece" + std::to_string(gather_pieces_.size()),
                         {width}, std::move(binding), artifact::Residency::Device};
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(binder.object_id(part.object));
    }
    pending.rank = rank;
    const WeightId id{weights.size()};
    weights.push_back(std::move(pending));
    gather_pieces_.emplace(std::move(key), id);
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format,
                          artifact::Residency residency) {
    return parameter(std::move(name), std::move(shape), {}, format, residency);
}

void Bindings::evict(WeightId id, std::uint32_t rank) {
    const auto& parameter = at(id);
    if (parameter.reference.residency != artifact::Residency::Device) {
        throw artifact::ArtifactError(parameter.reference.name +
                                      ": only a device parameter can be evictable");
    }
    for (const auto& part : parameter.reference.binding.parts) {
        binder.evict_device(part.object, rank);
    }
}

WeightId Bindings::auxiliary_on_rank(WeightId id, std::size_t rank) {
    if (rank == 0) { return id; }
    const auto key = std::pair{id.index, rank};
    if (const auto found = auxiliary_ranks_.find(key); found != auxiliary_ranks_.end()) {
        return found->second;
    }
    auto copy = at(id);
    copy.reference.name += "/rank/" + std::to_string(rank);
    copy.reference.device_replica = rank;
    for (const auto& part : copy.reference.binding.parts) {
        binder.replicate_device(part.object, rank);
    }
    const WeightId placed{weights.size()};
    weights.push_back(std::move(copy));
    auxiliary_ranks_.emplace(key, placed);
    return placed;
}

void Bindings::place(WeightId id, std::size_t rank) {
    // Rank 0 is recorded too, so an object shared between a rank-0 layer and a later stage's layer
    // is refused as a conflicting placement instead of silently living on one of them.
    const auto& parameter = at(id);
    if (parameter.reference.residency == artifact::Residency::Device) {
        for (const auto& part : parameter.reference.binding.parts) {
            binder.device_rank(part.object, rank);
        }
    }
    weights.at(id.index).rank = rank;
    // Input permutations and Hadamard signs may be shared by layers on different devices.
    // Retarget each Use to a rank-local immutable copy, deduplicated within that rank. Keep
    // indices across pushes: creating a copy can reallocate the pending-weight vector.
    const auto count = parameter.uses.size();
    for (std::size_t i = 0; i < count; ++i) {
        const auto signs = weights[id.index].uses[i].hadamard_signs;
        const auto columns = weights[id.index].uses[i].input_columns;
        if (signs) {
            const auto placed = auxiliary_on_rank(*signs, rank);
            weights[id.index].uses[i].hadamard_signs = placed;
        }
        if (columns) {
            const auto placed = auxiliary_on_rank(*columns, rank);
            weights[id.index].uses[i].input_columns = placed;
        }
    }
}

bool Bindings::transcode(WeightId id, QType target, std::string_view option) {
    const auto& parameter = at(id);
    bool requested        = false;
    for (const auto& part : parameter.reference.binding.parts) {
        const auto geometry = binder.geometry(part.object);
        if (geometry.format == target) { continue; }
        if (geometry.layout != QuantLayout::RowSplit ||
            !artifact::row_split_transcode_supported(geometry.format, target)) {
            throw std::invalid_argument(std::string(option) + " requires " +
                                        parameter.reference.name +
                                        " to be stored as row-split Q8_G32");
        }
        binder.transcode_device(part.object, target);
        requested = true;
    }
    return requested;
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        // Disk expert parameters retain geometry in NativeExpertSources. They have no resident
        // WeightInput and are never passed to a resident Op preparation routine.
        auto view = item.reference.residency == artifact::Residency::Values
                        ? WeightView{item.reference.shape, {}}
                        : artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses), item.rank});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
