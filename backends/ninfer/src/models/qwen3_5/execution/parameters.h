#pragma once

#include "models/qwen3_5/lora_adapter.h"
#include "models/qwen3_5/model.h"
#include "ninfer/ops/weight_input.h"

#include "core/arena.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline ops::LinearPolicy residual_projection_policy(const LinearParameters& p,
                                                                  bool wide_verification) {
    // Keep the neural path's residual activation precision in wide copy verification.
    return wide_verification && p.weight.qtype == QType::FP8_E4M3FN_ROW_BF16
               ? ops::LinearPolicy::A16Only
               : p.policy;
}

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

// One attached adapter projection: A [rank, in] and B [out, rank], both FP16, both views on the
// adapter's single device allocation. `scale` already folds alpha/rank and --lora-scale.
struct LoraPair {
    Tensor a;
    Tensor b;
    // FP32 [rank_capacity, max_tokens] shared by every factor of the adapter; a captured decode
    // graph keeps its address, so it cannot come from a per-call workspace arena.
    Tensor scratch;
    float scale = 0.0F;

    [[nodiscard]] bool present() const noexcept {
        return a.data != nullptr && b.data != nullptr && scratch.data != nullptr;
    }
};

struct AttentionLora {
    LoraPair query, key, value, output, gate;
};

struct DenseLora {
    LoraPair gate, up, down;
    // FP32 [2 * rank_capacity, tokens_capacity]: the fused gate/up delta and product needs both
    // projections' rank reductions at once, so it cannot share `LoraPair::scratch`.
    Tensor swiglu_scratch;
};

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
    // gate_up policy for Verify-phase calls. --mlp-a8-decode widens this to AllowA8IntDecode
    // independently of --prefill-a8, so long as the profile is eligible; prefill and scoring
    // always use gate_up.policy and never take the decode-width integer route.
    ops::LinearPolicy verify_gate_up_policy = ops::LinearPolicy::A16Only;
    // The up matrix when gate and up are separate GGUF parents; gate_up then holds the gate alone.
    std::optional<LinearParameters> up;
    // Attached adapter for this block's dense FFN, absent for an artifact without one.
    std::optional<DenseLora> lora;
};

struct StreamedMoeParameters {
    // Routed Weight fields contain only the codec profile; all operand addresses come from the
    // delivery tables. Router and shared expert remain immutable resident native operands.
    ops::SparseMoeWeights weights;
    std::size_t delivery_layer = 0;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights, StreamedMoeParameters>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    // Attached adapter for this block's attention, absent for an artifact without one.
    std::optional<AttentionLora> lora;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
    // Which stage's device holds this layer: its weights, its KV cache or recurrent state, and the
    // scratch it runs in. Always 0 on one GPU.
    std::size_t rank = 0;
};

struct TextParameters {
    Weight token_embedding;
    // A token table stored in the rotated basis (T2) is restored after the gather with the
    // hidden-width sign vector; empty for an ordinary table.
    Tensor token_embedding_signs;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
    // Tensor parallelism: `layers` holds rank 0's share of each layer and rank_layers[r - 1] rank
    // r's, whose attention layers carry no mixer (rank 0 runs attention alone). Empty otherwise.
    std::vector<std::vector<BlockParameters>> rank_layers;
    // How many stages the layer loop spans, or how many tensor-parallel ranks share every layer.
    // One is the identity plan and the fast path.
    std::size_t rank_count = 1;
    // Stage s owns layers [stage_begin[s], stage_end(s)).
    std::vector<std::uint32_t> stage_begin{0};

    [[nodiscard]] bool split_execution() const noexcept { return rank_count > 1; }
    [[nodiscard]] bool tensor_parallel() const noexcept { return !rank_layers.empty(); }
    [[nodiscard]] std::uint32_t stage_end(std::size_t stage) const {
        return stage + 1 < stage_begin.size() ? stage_begin[stage + 1]
                                              : static_cast<std::uint32_t>(layers.size());
    }

    // The DFlash feature taps among stage `stage`'s layers, in the drafter's order.
    [[nodiscard]] std::vector<std::uint32_t> stage_taps(std::size_t stage,
                                                        std::span<const std::uint32_t> taps) const {
        const std::uint32_t begin = stage_begin.at(stage);
        const std::uint32_t end   = stage_end(stage);
        std::vector<std::uint32_t> out;
        for (const std::uint32_t layer : taps) {
            if (layer >= begin && layer < end) { out.push_back(layer); }
        }
        return out;
    }
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DSpineBridgeParameters {
    Tensor gate_weight, predecessor_gate_weight, prev_weight;
    Tensor early_read_weights, read_weight;
    Tensor feature_write_weights, codebook, gate_bias;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    std::optional<DSpineBridgeParameters> dspine_bridge;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
    // The one device allocation every attached adapter factor is a view on. Empty without --lora.
    std::vector<DeviceBuffer> lora_owner;};

// Uploads `adapter` and points each text block's projection at its factors. `user_scale` multiplies
// the file's own alpha/rank so `--lora-scale 1` is llama.cpp's default. Throws when a factor's
// shape does not match the projection it names (an adapter built for another base) or when the
// process runs a staged/tensor-parallel plan, where one block does not own all of its weights.
void attach_lora(Parameters& parameters, const LoraAdapter& adapter, float user_scale);

} // namespace ninfer::models::qwen3_5::execution
