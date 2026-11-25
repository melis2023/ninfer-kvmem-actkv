#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/rotation.h"

#include "core/device.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

namespace {

// copies the `dst.ne[0]` channel rows of a flat [channels, columns] plane that start at `begin`
// into a contiguous destination of the same row width.
void copy_channel_block(const Tensor& src, std::int32_t columns, std::int32_t begin, Tensor& dst,
                        cudaStream_t stream) {
    const std::size_t bytes =
        static_cast<std::size_t>(dst.ne[0]) * static_cast<std::size_t>(columns) * sizeof(std::uint16_t);
    const auto* from = static_cast<const std::uint16_t*>(src.data) +
                       static_cast<std::size_t>(begin) * static_cast<std::size_t>(columns);
    CUDA_CHECK(cudaMemcpyAsync(dst.data, from, bytes, cudaMemcpyDeviceToDevice, stream));
}

std::int32_t projection_input_rows(const ops::ProjectionWeights& projection) {
    if (const auto* single = std::get_if<LinearParameters>(&projection)) {
        return single->weight.k;
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&projection)) {
        return gguf->parts.front().weight.k;
    }
    if (const auto* coded = std::get_if<ops::CodedProjectionWeights>(&projection)) {
        return coded->leader().k;
    }
    return std::get<ops::PairedProjectionWeights>(projection).first.k;
}

// The native overlap contract takes a disjoint span even for a zero-scratch fused route.
std::size_t snapshot_scratch_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                   std::int32_t batch, std::int32_t first_width,
                                   std::int32_t last_width) {
    std::size_t bytes;
    if (const auto* coded = std::get_if<ops::CodedProjectionWeights>(&parameters.projection)) {
        // No fused convolution for a coded parent: one parent plane plus the projection's own
        // workspace, and the snapshot output plane.
        const std::int32_t columns = std::max(batch, 1) * last_width;
        const std::int32_t channels =
            dimension(static_cast<std::uint64_t>(coded->parents.front().weight.n));
        bytes = ops::coded_projection_workspace_bytes(*coded, columns) +
                static_cast<std::size_t>(channels) * static_cast<std::size_t>(columns) *
                    sizeof(std::uint16_t) * 2;
    } else if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(*gguf, batch,
                                                                           first_width, last_width);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        bytes = ops::gdn_input_proj_split_conv_snapshot_workspace_capacity_bytes(
            pair->first.qtype, pair->second.qtype, pair->policy, batch, first_width, last_width);
    } else if (const auto& single = std::get<LinearParameters>(parameters.projection);
               single.weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single.weight;
        bytes         = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single.policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    return std::max(std::size_t{1}, bytes);
}

std::size_t record_scratch_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                 std::int32_t batch, std::int32_t first_width,
                                 std::int32_t last_width) {
    std::size_t bytes;
    if (const auto* coded = std::get_if<ops::CodedProjectionWeights>(&parameters.projection)) {
        // The replay record's convolution is fused with the projection in every dense route; a
        // coded parent has no such route, so the record path is refuse-only for now.
        (void)coded;
        throw std::invalid_argument("coded NariQ weights do not support replay-record GDN");
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(*gguf, batch, first_width,
                                                                         last_width);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        bytes = ops::gdn_input_proj_split_conv_record_workspace_capacity_bytes(
            pair->first.qtype, pair->second.qtype, pair->policy, batch, first_width, last_width);
    } else if (const auto& single = std::get<LinearParameters>(parameters.projection);
               single.weight.qtype != QType::Q8_G32_FP16) {
        const auto& w = single.weight;
        bytes         = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            w.qtype, w.n, w.k, single.policy, batch, first_width, last_width);
    } else {
        bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.key_width()),
            static_cast<std::int32_t>(config.value_width()), batch, first_width, last_width);
    }
    return std::max(std::size_t{1}, bytes);
}

} // namespace

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("GDN projection: invalid column interval");
    }
    std::size_t inner;
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& w = single->weight;
        inner = ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, w.n, w.k, single->policy,
                                                             first, last);
    } else if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        inner = ops::gdn_input_proj_workspace_capacity_bytes(*gguf, first, last);
    } else if (const auto* coded =
                   std::get_if<ops::CodedProjectionWeights>(&parameters.projection)) {
        inner = ops::coded_projection_workspace_bytes(*coded, last);
    } else {
        const auto& pair = std::get<ops::PairedProjectionWeights>(parameters.projection);
        inner            = ops::gdn_input_proj_split_workspace_capacity_bytes(
            pair.first.qtype, pair.first.n, pair.second.qtype, pair.second.n, pair.first.k,
            pair.policy, first, last);
    }
    return rotated_workspace_bytes(projection_signs(parameters.projection),
                                   projection_input_rows(parameters.projection), last, inner);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                         std::int32_t batch, std::int32_t first_width,
                                         std::int32_t last_width) {
    return rotated_workspace_bytes(
        projection_signs(parameters.projection), projection_input_rows(parameters.projection),
        std::max(batch, 1) * last_width,
        snapshot_scratch_bytes(parameters, config, batch, first_width, last_width));
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, const GdnConfig& config,
                                       std::int32_t batch, std::int32_t first_width,
                                       std::int32_t last_width) {
    return rotated_workspace_bytes(
        projection_signs(parameters.projection), projection_input_rows(parameters.projection),
        std::max(batch, 1) * last_width,
        record_scratch_bytes(parameters, config, batch, first_width, last_width));
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis) {
    auto scope = workspace.scope();
    const Tensor x =
        rotated_input(hidden, projection_signs(parameters.projection), workspace, stream, basis);
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj(x, *gguf, qkv, z, workspace, stream);
    } else if (const auto* coded =
                   std::get_if<ops::CodedProjectionWeights>(&parameters.projection)) {
        Tensor* const outputs[2] = {&qkv, &z};
        ops::coded_projection(x, *coded, outputs, 2, x.ne[1], workspace, stream);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj(x, pair->first, pair->second, qkv, z, pair->policy, workspace, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj(x, single.weight, qkv, z, single.policy, workspace, stream);
    }
}

InputBasis gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                            bool norm_unit_offset, const GdnParameters& parameters, Tensor& hidden,
                            Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                            DeviceExecutionView execution) {
    const Tensor& signs = projection_signs(parameters.projection);
    if (const auto* coded = std::get_if<ops::CodedProjectionWeights>(&parameters.control)) {
        // The fused A/B router reads stored weights directly; a coded parent instead has to go
        // through one trellis linear per side, so the norm runs on its own first.
        const std::int32_t columns = residual.ne[1];
        const std::int32_t heads   = dimension(static_cast<std::uint64_t>(
            coded->parents.front().weight.n));
        auto scope                 = workspace.scope();
        ops::rmsnorm(residual, norm, epsilon, norm_unit_offset, hidden, execution.stream);
        Tensor a = workspace.alloc(DType::BF16, {heads, columns});
        Tensor b = workspace.alloc(DType::BF16, {heads, columns});
        Tensor* const outputs[2] = {&a, &b};
        ops::coded_projection(hidden, *coded, outputs, 2, columns, workspace, execution.stream);
        ops::gdn_gating(a, b, parameters.a_log, parameters.dt_bias, g, beta, execution.stream);
        return InputBasis::Primal;
    }
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.control)) {
        if (rotated(signs)) {
            ops::gdn_norm_gating_proj_rotated(residual, norm, epsilon, pair->first, pair->second,
                                              parameters.a_log, parameters.dt_bias, signs,
                                              workspace, hidden, g, beta, execution);
            return InputBasis::Rotated;
        }
        ops::gdn_norm_gating_proj(residual, norm, epsilon, pair->first, pair->second,
                                  parameters.a_log, parameters.dt_bias, workspace, hidden, g, beta,
                                  execution);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.control);
        ops::gdn_norm_gating_proj(residual, norm, epsilon, single.weight, parameters.a_log,
                                  parameters.dt_bias, workspace, hidden, g, beta, execution);
    }
    return InputBasis::Primal;
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream, InputBasis basis) {
    auto scope = workspace.scope();
    const Tensor x =
        rotated_input(hidden, projection_signs(parameters.projection), workspace, stream, basis);
    WorkspaceArena scratch(workspace.alloc_bytes(
        snapshot_scratch_bytes(parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* coded = std::get_if<ops::CodedProjectionWeights>(&parameters.projection)) {
        // No fused coded route exists for the batched snapshot: project into one qkv plane, run
        // the snapshot convolution over it, then copy the three channel blocks to their outputs.
        const std::int32_t columns = hidden.ne[1] * hidden.ne[2];
        const std::int32_t channels =
            dimension(static_cast<std::uint64_t>(coded->parents.front().weight.n));
        Tensor qkv_flat = workspace.alloc(DType::BF16, {channels, columns});
        auto scope      = workspace.scope();
        Tensor* const outputs[2] = {&qkv_flat, &z};
        ops::coded_projection(x, *coded, outputs, 2, columns, workspace, stream);
        Tensor qkv_view   = qkv_flat.view({channels, hidden.ne[1], hidden.ne[2]});
        Tensor conv_out   = workspace.alloc(DType::BF16, {channels, hidden.ne[1], hidden.ne[2]});
        ops::causal_conv1d_silu_snapshot(qkv_view, parameters.convolution, conv_states,
                                         valid_columns, initial_slots, destination_slots,
                                         conv_out, stream);
        copy_channel_block(conv_out, columns, 0, query, stream);
        copy_channel_block(conv_out, columns, static_cast<std::int32_t>(query.ne[0]), key, stream);
        copy_channel_block(conv_out, columns,
                           static_cast<std::int32_t>(query.ne[0] + key.ne[0]), value, stream);
    } else if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(x, *gguf, parameters.convolution, conv_states,
                                          valid_columns, initial_slots, destination_slots, query,
                                          key, value, z, scratch, stream);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_snapshot(
            x, pair->first, pair->second, parameters.convolution, conv_states, valid_columns,
            initial_slots, destination_slots, query, key, value, z, pair->policy, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_snapshot(x, single.weight, parameters.convolution, conv_states,
                                          valid_columns, initial_slots, destination_slots, query,
                                          key, value, z, single.policy, scratch, stream);
    }
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                           InputBasis basis) {
    auto scope = workspace.scope();
    const Tensor x =
        rotated_input(hidden, projection_signs(parameters.projection), workspace, stream, basis);
    WorkspaceArena scratch(workspace.alloc_bytes(
        record_scratch_bytes(parameters, config, hidden.ne[2], hidden.ne[1], hidden.ne[1])));
    if (const auto* coded = std::get_if<ops::CodedProjectionWeights>(&parameters.projection)) {
        (void)coded;
        throw std::invalid_argument("coded NariQ weights do not support replay-record GDN");
    }
    if (const auto* gguf = std::get_if<ops::GgufProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(x, *gguf, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, scratch, stream);
    } else if (const auto* pair =
                   std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::gdn_input_proj_conv_record(x, pair->first, pair->second, parameters.convolution,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, z, pair->policy, scratch, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::gdn_input_proj_conv_record(x, single.weight, parameters.convolution, conv_states,
                                        valid_columns, initial_slots, conv_record, query, key,
                                        value, z, single.policy, scratch, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
