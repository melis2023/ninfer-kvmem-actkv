#pragma once

// Flint (FlintForCausalLM): the immutable model a v3 artifact describes. Matrices stay in the GGUF
// block formats the converter imported; vectors are FP32. The frontend resources are Flint's
// tokenizer and chat template, which the shared frontend serves in its Flint dialect.

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/expert_delivery.h"
#include "core/startup.h"
#include "models/flint/config.h"
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

namespace ninfer::models::flint {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;

struct AttentionWeights {
    WeightId query, key, value, output, query_norm, key_norm, sinks;
};

// A MoE block: the router, the shared expert's [gate; up] and down, and the routed experts' banks
// when they are resident on the device (gate and up rows interleaved per expert).
struct MoeWeights {
    WeightId router, shared_gate, shared_up, shared_down;
    std::optional<WeightId> gate_up, down;
};

struct ExpertBanks {
    QType gate_up_type{}, down_type{};
    std::uint64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    ExpertDeliveryLayer delivery;
};

struct LayerWeights {
    WeightId input_norm, ffn_norm;
    AttentionWeights attention;
    std::optional<WeightId> mlp_gate, mlp_up, mlp_down; // the dense blocks
    std::optional<MoeWeights> moe;                       // the rest
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    std::vector<LayerWeights> layers;
};

struct LoadOptions {
    std::filesystem::path artifact;
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
    // Host or disk experts: one bank per MoE block, in block order (block dense_layers + i).
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
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    ExpertResidency expert_residency_ = ExpertResidency::Device;
    std::vector<ExpertBanks> expert_banks_;
    std::vector<std::filesystem::path> expert_files_;
};

// Whether the artifact's text component is Flint.
[[nodiscard]] bool is_flint(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::flint
