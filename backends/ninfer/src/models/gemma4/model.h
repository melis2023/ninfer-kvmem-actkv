#pragma once

// Gemma 4 (Gemma4ForCausalLM): the immutable model a v3 artifact describes. Matrices stay in the
// GGUF block formats the converter imported; vectors are FP32. The frontend resources are Gemma 4's
// tokenizer and chat template, which the shared frontend serves in its Gemma dialect.

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/expert_delivery.h"
#include "core/startup.h"
#include "models/gemma4/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::gemma4 {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;

struct AttentionWeights {
    WeightId query, key, output, query_norm, key_norm;
    std::optional<WeightId> value; // sliding-window blocks; full attention reads K as V
};

struct MoeWeights {
    WeightId router, router_scale, expert_scale, pre_norm, post_norm;
    std::optional<WeightId> gate_up, down; // resident banks only
    WeightId mlp_post_norm; // post_feedforward_layernorm_1 of the dense branch
};

struct ExpertBanks {
    QType gate_up_type{}, down_type{};
    std::uint64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    ExpertDeliveryLayer delivery;
};

struct LayerWeights {
    WeightId input_norm, post_attention_norm, ffn_norm, post_ffn_norm, layer_scalar;
    AttentionWeights attention;
    WeightId mlp_gate, mlp_up, mlp_down;
    std::optional<MoeWeights> moe;
};

struct TextWeights {
    WeightId output_head, final_norm; // the head is also the token table
    std::vector<LayerWeights> layers;
};

// The assistant's block: queries only, since it attends over the target's keys and values.
struct AssistantLayerWeights {
    WeightId input_norm, post_attention_norm, ffn_norm, post_ffn_norm, layer_scalar;
    WeightId query, query_norm, output;
    WeightId mlp_gate, mlp_up, mlp_down;
};

struct AssistantWeights {
    WeightId output_head, final_norm; // its own tied table, read as the head only
    WeightId pre_projection;          // [H_a, 2 H]: target embedding and hidden state in
    WeightId post_projection;         // [H, H_a]: the next step's hidden state out
    std::vector<AssistantLayerWeights> layers;
};

// The vision tower: BF16 matrices (Q, K and V of a block one parent, gate and up another), FP32
// norms, the BF16 patch embedding [hidden, patch features] and position table [2 x positions,
// hidden] (x rows, then y rows).
struct VisionLayerWeights {
    WeightId input_norm, post_attention_norm, ffn_norm, post_ffn_norm;
    WeightId query, key, value, output, query_norm, key_norm;
    WeightId mlp_gate, mlp_up, mlp_down;
};

struct VisionWeights {
    WeightId patch_embedding, position_table, projection;
    std::optional<WeightId> std_bias, std_scale; // with VisionConfig::standardize
    std::vector<VisionLayerWeights> layers;
};

// Where the vision tower's weights live: on the device for the tower's CUDA kernels; in pinned
// host memory, streamed block by block through a small device window as the tower runs; or in
// host memory for its CPU encoder.
enum class VisionPlacement : std::uint8_t {
    None,
    Device,
    Pinned,
    Host,
};

struct LoadOptions {
    std::filesystem::path artifact;
    // Binds the artifact's `mtp` component, which must then exist.
    bool assistant = false;
    // Binds the artifact's `vision` component, which must then exist, in this placement.
    VisionPlacement vision = VisionPlacement::None;
    ExpertResidency experts = ExpertResidency::Device;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] ExpertResidency expert_residency() const noexcept { return expert_residency_; }
    [[nodiscard]] std::span<const ExpertBanks> expert_banks() const noexcept { return expert_banks_; }
    [[nodiscard]] std::uint64_t expert_pinned_bytes() const noexcept {
        std::uint64_t bytes = 0;
        for (const auto& bank : expert_banks_) {
            for (const auto& expert : bank.delivery.experts) {
                for (const auto& blob : expert) { if (blob.pinned) { bytes += blob.bytes; } }
            }
        }
        return bytes;
    }
    [[nodiscard]] const std::vector<std::filesystem::path>& expert_files() const noexcept {
        return expert_files_;
    }
    // Present when loaded with LoadOptions::assistant.
    [[nodiscard]] const std::optional<AssistantConfig>& assistant_config() const noexcept {
        return assistant_config_;
    }
    [[nodiscard]] const std::optional<AssistantWeights>& assistant() const noexcept {
        return assistant_;
    }
    // Present when loaded with LoadOptions::vision.
    [[nodiscard]] const std::optional<VisionConfig>& vision_config() const noexcept {
        return vision_config_;
    }
    [[nodiscard]] const std::optional<VisionWeights>& vision() const noexcept { return vision_; }
    [[nodiscard]] VisionPlacement vision_placement() const noexcept { return vision_placement_; }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    struct VisionParts {
        std::optional<VisionConfig> config;
        std::optional<VisionWeights> weights;
        VisionPlacement placement = VisionPlacement::None;
    };
    Model(TextConfig config, TextWeights weights, std::optional<AssistantConfig> assistant_config,
          std::optional<AssistantWeights> assistant, VisionParts vision,
          std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::optional<AssistantConfig> assistant_config_;
    std::optional<AssistantWeights> assistant_;
    std::optional<VisionConfig> vision_config_;
    std::optional<VisionWeights> vision_;
    VisionPlacement vision_placement_ = VisionPlacement::None;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    ExpertResidency expert_residency_ = ExpertResidency::Device;
    std::vector<ExpertBanks> expert_banks_;
    std::vector<std::filesystem::path> expert_files_;
};

// Whether the artifact's text component is Gemma 4.
[[nodiscard]] bool is_gemma4(const artifact::Reader& reader);
// Whether it also holds the multi-token-prediction assistant.
[[nodiscard]] bool has_assistant(const artifact::Reader& reader);
// Whether it also holds the vision tower.
[[nodiscard]] bool has_vision(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::gemma4
