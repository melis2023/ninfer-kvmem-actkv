#include "ninfer/ops/weight_input.h"
#include "ninfer/ops/trellis.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

// Rows fused into one native operand read one activation, so they must agree on its rotation.
Tensor common_signs(std::span<const WeightInput> inputs) {
    const Tensor signs = inputs.front().hadamard_signs;
    for (const auto& input : inputs) {
        require(input.hadamard_signs.data == signs.data &&
                    input.hadamard_signs.ne[0] == signs.ne[0],
                "fused projection rows must share one Hadamard rotation");
    }
    return signs;
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    auto view         = concatenate_rows(inputs);
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float divisor     = 0;
    if (region.parent->geometry.format == QType::NVFP4) {
        require(std::isfinite(region.parent->weight_scale_divisor) &&
                    region.parent->weight_scale_divisor > 0,
                "NVFP4 parent requires a positive weight divisor");
        for (const auto& input : inputs) {
            if (!input.activation_input_divisor) {
                require(!allows_a4(policy), "NVFP4 A4 native input requires an activation divisor");
                continue;
            }
            require(std::isfinite(*input.activation_input_divisor) &&
                        *input.activation_input_divisor > 0,
                    "NVFP4 native input requires a positive activation divisor");
            if (divisor == 0) {
                // A16 does not read this auxiliary; retain a stored positive value for the ABI.
                divisor = *input.activation_input_divisor;
            } else if (allows_a4(policy)) {
                // Current NVFP4 consumers use A16 for AllowA8 as well. Only their A4 route
                // quantizes the shared activation and therefore requires a common divisor.
                require(std::bit_cast<std::uint32_t>(divisor) ==
                            std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
                        "shared NVFP4 native input requires identical activation divisors");
            }
        }
        // The legacy native ABI validates this field even on A16, where it is never read.
        // Keep that ABI detail out of the artifact's optional Use auxiliaries.
        if (divisor == 0) { divisor = 1.0F; }
    }
    SingleProjectionWeight result{native_weight(view, divisor), policy, common_signs(inputs)};
    const Tensor& columns = inputs.front().input_columns;
    for (const auto& input : inputs) {
        require(input.input_columns.data == columns.data,
                "fused projection rows must share one input gather");
    }
    if (columns.data != nullptr) {
        require(columns.dtype == DType::I32 && columns.numel() == result.weight.k,
                "an input gather must be INT32 [K]");
        result.weight.input_columns = static_cast<const std::int32_t*>(columns.data);
    }
    // A coded weight carries its own side planes and cannot be assembled from several parameters:
    // its row Hadamard spans the whole matrix, so no consumer may slice it into logical pieces.
    const Tensor& descriptor = inputs.front().trellis_desc;
    for (const auto& input : inputs) {
        require(input.trellis_desc.data == descriptor.data,
                "a coded projection cannot be fused from several parameters");
    }
    if (descriptor.data != nullptr) {
        require(inputs.size() == 1, "a coded projection is a single parameter");
        require(is_coded(result.weight.qtype), "trellis side planes require a coded format");
        const auto& s16  = inputs.front().trellis_s16;
        const auto& pre  = inputs.front().trellis_pre;
        const auto& post = inputs.front().trellis_post;
        const auto& pin  = inputs.front().trellis_pin;
        const auto& pout = inputs.front().trellis_pout;
        require(s16.data != nullptr && s16.dtype == DType::FP32 && s16.numel() == result.weight.k / 16,
                "a coded weight needs an FP32 [K/16] scale plane");
        require(pre.data != nullptr && pre.dtype == DType::FP32 && pre.numel() == result.weight.k,
                "a coded weight needs an FP32 [K] input diagonal");
        require(post.data != nullptr && post.dtype == DType::FP32 && post.numel() == result.weight.n,
                "a coded weight needs an FP32 [N] output diagonal");
        require(pin.data == nullptr || (pin.dtype == DType::I32 && pin.numel() == result.weight.k),
                "a coded input gather must be INT32 [K]");
        require(pout.data == nullptr || (pout.dtype == DType::I32 && pout.numel() == result.weight.n),
                "a coded output gather must be INT32 [N]");
        require(descriptor.dtype == DType::I32 && descriptor.numel() >= 7,
                "a coded descriptor must be INT32 [7 + parts]");
        result.weight.trellis_s16   = s16.data;
        result.weight.trellis_pre   = pre.data;
        result.weight.trellis_post  = post.data;
        result.weight.trellis_pin   = pin.data == nullptr ? nullptr : static_cast<const std::int32_t*>(pin.data);
        result.weight.trellis_pout  = pout.data == nullptr ? nullptr : static_cast<const std::int32_t*>(pout.data);
        result.weight.trellis_desc  = static_cast<const std::int32_t*>(descriptor.data);
        result.weight.trellis_desc_len = static_cast<std::int32_t>(descriptor.numel());
    }
    return result;
}

bool gguf_parent(const WeightInput& input) {
    return !input.weight.parts.empty() && input.weight.parts.front().parent != nullptr &&
           is_gguf(input.weight.parts.front().parent->geometry.format);
}

// GGUF parts of a fused input projection, merged where consecutive logical inputs continue one
// parent's rows into one output's rows.
GgufProjectionWeights gguf_projection(std::span<const WeightInput> inputs,
                                      std::span<const std::int32_t> outputs,
                                      std::span<const std::int32_t> rows) {
    GgufProjectionWeights result;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        require(gguf_parent(inputs[i]), "input projection: GGUF parts cannot mix formats");
        auto part = single(inputs.subspan(i, 1));
        if (!result.parts.empty()) {
            auto& last = result.parts.back();
            const auto block = gguf_block_shape(last.weight.qtype);
            const std::int64_t row_bytes = std::int64_t(last.weight.k / block.elements) * block.bytes;
            const auto* end = static_cast<const std::byte*>(last.weight.qdata) + last.weight.n * row_bytes;
            if (part.weight.qtype == last.weight.qtype && part.weight.qdata == end &&
                outputs[i] == last.output && rows[i] == last.row + last.weight.n) {
                last.weight.n += part.weight.n;
                last.weight.shape[0] = last.weight.padded_shape[0] = last.weight.n;
                continue;
            }
        }
        result.parts.push_back({part.weight, outputs[i], rows[i]});
    }
    result.hadamard_signs = common_signs(inputs);
    return result;
}

ProjectionWeights input_projection(std::span<const WeightInput, 4> inputs, bool attention) {
    const auto& q      = matrix(inputs[0]);
    const auto& k      = matrix(inputs[1]);
    const auto& third  = matrix(inputs[2]);
    const auto& fourth = matrix(inputs[3]);
    const bool dense =
        attention ? q == std::vector<std::uint64_t>{6144, 5120} &&
                        k == std::vector<std::uint64_t>{1024, 5120} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 5120} && k == q &&
                        third == std::vector<std::uint64_t>{6144, 5120} && fourth == third;
    const bool moe =
        attention ? q == std::vector<std::uint64_t>{4096, 2048} &&
                        k == std::vector<std::uint64_t>{512, 2048} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 2048} && k == q &&
                        third == std::vector<std::uint64_t>{4096, 2048} && fourth == third;
    if (std::any_of(inputs.begin(), inputs.end(), gguf_parent)) {
        require(q[1] == k[1] && q[1] == third[1] && q[1] == fourth[1] &&
                    (attention ? third == q && fourth == k && q[0] % k[0] == 0
                               : k == q && fourth == third && third[0] % q[0] == 0),
                "input projection: inconsistent GGUF logical geometry");
        const auto q = static_cast<std::int32_t>(matrix(inputs[0])[0]);
        const auto k = static_cast<std::int32_t>(matrix(inputs[1])[0]);
        if (attention) {
            const std::array<std::int32_t, 4> outputs{0, 2, 1, 3};
            const std::array<std::int32_t, 4> rows{0, 0, 0, 0};
            return gguf_projection(inputs, outputs, rows);
        }
        const std::array<std::int32_t, 4> outputs{0, 0, 0, 1};
        const std::array<std::int32_t, 4> rows{0, q, q + k, 0};
        return gguf_projection(inputs, outputs, rows);
    }
    require(dense || moe, "input projection: unsupported native logical projection geometry");
    const auto joined = concatenate_rows(inputs);
    if (contiguous(joined)) {
        auto result       = single(inputs);
        const auto format = result.weight.qtype;
        const bool supported =
            (moe && format == QType::Q8_G32_FP16) ||
            (dense && (format == QType::NVFP4 || format == QType::FP8_E4M3FN_ROW_BF16 ||
                       (attention && format == QType::BF16)));
        require(supported, "input projection: unsupported single-parent format");
        return result;
    }
    require(dense, "input projection: unsupported multi-parent geometry");
    const auto first  = single(inputs.first<2>());
    const auto second = single(inputs.last<2>());
    const bool q4_q5 =
        first.weight.qtype == QType::Q4_G64_FP16 && second.weight.qtype == QType::Q5_G64_FP16;
    const bool ternary = is_ternary(first.weight.qtype) &&
                         first.weight.qtype == second.weight.qtype;
    require(q4_q5 || ternary,
            "input projection: requires Q4/Q5 or two ternary parents of the same format");
    const auto pair_policy = first.weight.qtype == QType::T5_G128_FP16
                                 ? common_policy(inputs) : LinearPolicy::A16Only;
    return PairedProjectionWeights{first.weight, second.weight, pair_policy,
                                   common_signs(inputs)};
}

} // namespace

bool joins(std::span<const WeightInput> inputs) { return contiguous(concatenate_rows(inputs)); }

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                       const WeightInput& key,
                                                       const WeightInput& value) {
    const auto& q = matrix(query);
    require((q == std::vector<std::uint64_t>{4096, 2048} ||
             q == std::vector<std::uint64_t>{4096, 5120}) &&
                matrix(key) == std::vector<std::uint64_t>{1024, q[1]} &&
                matrix(value) == matrix(key),
            "QKV input projection: unsupported logical geometry");
    const std::array inputs{query, key, value};
    auto result = single(inputs);
    require(result.weight.qtype == QType::Q8_G32_FP16,
            "QKV input projection: native form requires Q8");
    return result;
}

ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                  const WeightInput& gate,
                                                  const WeightInput& value) {
    const std::array inputs{query, key, gate, value};
    return input_projection(inputs, true);
}

ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query, const WeightInput& key,
                                                 const WeightInput& value, const WeightInput& z) {
    const std::array inputs{query, key, value, z};
    return input_projection(inputs, false);
}

ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b) && shape[0] > 0 && shape[0] % 16 == 0 &&
                shape[1] > 0 && shape[1] % 128 == 0,
            "GDN control: unsupported A/B geometry");
    const std::array inputs{a, b};
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
    return PairedProjectionWeights{first.weight, second.weight};
}

namespace {

CodedProjectionWeights coded_form(std::vector<SingleProjectionWeight> parents,
                                  std::vector<CodedProjectionReach> reaches) {
    CodedProjectionWeights result;
    result.parents = std::move(parents);
    result.reaches = std::move(reaches);
    if (result.parents.empty() || result.reaches.empty()) {
        throw std::invalid_argument("a coded projection needs parents and destinations");
    }
    for (const auto& parent : result.parents) {
        require(is_coded(parent.weight.qtype), "a coded projection needs coded parents");
        require(!(parent.hadamard_signs.data != nullptr),
                "a coded projection carries its rotation in the weight");
    }
    for (const auto& reach : result.reaches) {
        require(reach.parent >= 0 && static_cast<std::size_t>(reach.parent) < result.parents.size() &&
                    reach.output >= 0 && reach.dst_row >= 0 && reach.rows > 0,
                "coded projection destination is out of range");
    }
    return result;
}

// Rows of one coded parent, `rows` of them starting at logical `first`.
void reach_parent(CodedProjectionWeights& out, std::int32_t parent, std::int32_t output,
                  std::int32_t dst_row, std::int32_t rows) {
    out.reaches.push_back(CodedProjectionReach{parent, output, dst_row, rows});
}

// Rows of one coded parent read through a stride pattern: `rows` of them, interleaved head by head.
void reach_parent_interleaved(CodedProjectionWeights& out, std::int32_t parent, std::int32_t output,
                              std::int32_t rows, std::int32_t group, std::int32_t origin) {
    out.reaches.push_back(
        CodedProjectionReach{parent, output, 0, rows, group, 2 * group, origin});
}

} // namespace

ProjectionWeights prepare_coded_attn_projection(const WeightInput& packed, const WeightInput& key,
                                                const WeightInput& value, std::uint64_t query_rows,
                                                std::uint64_t key_rows, std::uint64_t head_width) {
    require(matrix(packed)[0] == 2 * query_rows && matrix(key)[0] == key_rows &&
                matrix(value)[0] == key_rows && matrix(key)[1] == matrix(packed)[1] &&
                matrix(value)[1] == matrix(packed)[1],
            "coded attention projection: unsupported packed geometry");
    require(head_width > 0 && query_rows % head_width == 0 && 2 * head_width <= matrix(packed)[0],
            "coded attention projection: query rows do not tile the head width");
    std::vector<SingleProjectionWeight> parents{prepare_linear_weight(packed),
                                                prepare_linear_weight(key),
                                                prepare_linear_weight(value)};
    CodedProjectionWeights out;
    // The fused parent stores the head's query block then its gate block, head after head.
    reach_parent_interleaved(out, 0, 0, static_cast<std::int32_t>(query_rows),
                             static_cast<std::int32_t>(head_width), 0);
    reach_parent_interleaved(out, 0, 1, static_cast<std::int32_t>(query_rows),
                             static_cast<std::int32_t>(head_width),
                             static_cast<std::int32_t>(head_width));
    reach_parent(out, 1, 2, 0, static_cast<std::int32_t>(key_rows));
    reach_parent(out, 2, 3, 0, static_cast<std::int32_t>(key_rows));
    out.hadamard_signs = packed.hadamard_signs;
    return coded_form(std::move(parents), std::move(out.reaches));
}

ProjectionWeights prepare_coded_gdn_projection(const WeightInput& packed, const WeightInput& z,
                                               std::uint64_t key_rows, std::uint64_t value_rows) {
    require(matrix(packed)[0] == 2 * key_rows + value_rows && matrix(z)[0] == value_rows &&
                matrix(z)[1] == matrix(packed)[1],
            "coded GDN projection: unsupported packed geometry");
    std::vector<SingleProjectionWeight> parents{prepare_linear_weight(packed),
                                                prepare_linear_weight(z)};
    CodedProjectionWeights out;
    reach_parent(out, 0, 0, 0, static_cast<std::int32_t>(key_rows));
    reach_parent(out, 0, 0, static_cast<std::int32_t>(key_rows),
                 static_cast<std::int32_t>(key_rows));
    reach_parent(out, 0, 0, static_cast<std::int32_t>(2 * key_rows),
                 static_cast<std::int32_t>(value_rows));
    reach_parent(out, 1, 1, 0, static_cast<std::int32_t>(value_rows));
    out.hadamard_signs = packed.hadamard_signs;
    return coded_form(std::move(parents), std::move(out.reaches));
}

ProjectionWeights prepare_coded_gating_projection(const WeightInput& a, const WeightInput& b) {
    require(matrix(a) == matrix(b), "coded GDN control: A/B geometry differs");
    require(matrix(a)[0] % 16 == 0 && matrix(a)[1] % 128 == 0,
            "coded GDN control: unsupported A/B geometry");
    std::vector<SingleProjectionWeight> parents{prepare_linear_weight(a),
                                                prepare_linear_weight(b)};
    CodedProjectionWeights out;
    reach_parent(out, 0, 0, 0, static_cast<std::int32_t>(matrix(a)[0]));
    reach_parent(out, 1, 1, 0, static_cast<std::int32_t>(matrix(b)[0]));
    out.hadamard_signs = a.hadamard_signs;
    return coded_form(std::move(parents), std::move(out.reaches));
}

std::size_t coded_projection_workspace_bytes(const CodedProjectionWeights& weights,
                                             std::int32_t max_tokens) {
    auto parts = std::vector<TrellisProjectionPart>(weights.reaches.size());
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const auto& reach = weights.reaches[i];
        parts[i] = TrellisProjectionPart{&weights.parents.at(reach.parent).weight, reach.output,
                                         reach.dst_row, reach.rows, nullptr,
                                         reach.src_group, reach.src_stride, reach.src_origin};
    }
    return trellis_input_proj_workspace_bytes(parts.data(), parts.size(), max_tokens);
}

void coded_projection(const Tensor& x, const CodedProjectionWeights& weights,
                      Tensor* const* outputs, std::size_t output_count, std::int32_t max_tokens,
                      WorkspaceArena& workspace, cudaStream_t stream) {
    auto parts = std::vector<TrellisProjectionPart>(weights.reaches.size());
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const auto& reach = weights.reaches[i];
        parts[i] = TrellisProjectionPart{&weights.parents.at(reach.parent).weight, reach.output,
                                         reach.dst_row, reach.rows, nullptr,
                                         reach.src_group, reach.src_stride, reach.src_origin};
    }
    trellis_input_proj(x, parts.data(), parts.size(), outputs, output_count, max_tokens,
                       workspace, stream);
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down) {
    require(expert_gate_up.size() == 512 && expert_down.size() == 256,
            "SparseMoe native input requires 256 experts");
    require(matrix(router) == std::vector<std::uint64_t>{256, 2048} &&
                matrix(shared_score) == std::vector<std::uint64_t>{1, 2048},
            "SparseMoe router geometry differs");
    for (const auto& input : expert_gate_up) {
        require(matrix(input) == std::vector<std::uint64_t>{512, 2048},
                "SparseMoe gate/up geometry differs");
    }
    for (const auto& input : expert_down) {
        require(matrix(input) == std::vector<std::uint64_t>{2048, 512},
                "SparseMoe down geometry differs");
    }
    require(matrix(shared_gate) == std::vector<std::uint64_t>{512, 2048} &&
                matrix(shared_up) == matrix(shared_gate) &&
                matrix(shared_down) == std::vector<std::uint64_t>{2048, 512},
            "SparseMoe shared expert geometry differs");
    const std::array router_inputs{router, shared_score};
    const auto router_bank        = single(router_inputs);
    const auto gate_up_bank       = single(expert_gate_up);
    const auto down_bank          = single(expert_down);
    const auto shared             = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto down               = prepare_linear_weight(shared_down);
    const auto gate_format        = gate_up_bank.weight.qtype;
    const auto down_format        = down_bank.weight.qtype;
    const auto shared_format      = shared.weight.qtype;
    const auto shared_down_format = down.weight.qtype;
    const bool groupwise =
        shared_format == QType::Q8_G32_FP16 && shared_down_format == QType::Q8_G32_FP16 &&
        ((gate_format == QType::Q4_G64_FP16 &&
          (down_format == QType::Q5_G64_FP16 || down_format == QType::Q6_G64_FP16)) ||
         (gate_format == QType::Q8_G32_FP16 && down_format == QType::Q8_G32_FP16));
    // The four expert matrices are one profile rather than two independent choices: the prefill
    // route quantises the activation as well as the weight and its gate/up epilogue emits the
    // encoded intermediate `down` consumes, so a mixture with a groupwise codec has no route
    // behind it at any token count.
    const bool nvfp4 = gate_format == QType::NVFP4 && down_format == QType::NVFP4 &&
                       shared_format == QType::NVFP4 && shared_down_format == QType::NVFP4;
    require(router_bank.weight.qtype == QType::BF16 && (groupwise || nvfp4),
            "SparseMoe native bank formats are unsupported");
#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)
    // A model prefills, and this profile prefills only through W4A4 on Blackwell tensor cores.
    require(!nvfp4, "SparseMoe NVFP4 expert banks need an sm_120a build");
#endif
    if (nvfp4) {
        // The prefill route of this profile quantises the hidden state to four bits, which is what
        // AllowA4 grants. The groupwise profiles consume it represented and need no permission.
        const auto permits = [](const WeightInput& input) { return allows_a4(input.policy); };
        require(std::all_of(expert_gate_up.begin(), expert_gate_up.end(), permits) &&
                    std::all_of(expert_down.begin(), expert_down.end(), permits) &&
                    permits(shared_gate) && permits(shared_up) && permits(shared_down),
                "SparseMoe NVFP4 requires AllowA4 on every expert input");
    }
    return {router_bank.weight, gate_up_bank.weight, down_bank.weight, shared.weight, down.weight};
}

} // namespace ninfer::ops
