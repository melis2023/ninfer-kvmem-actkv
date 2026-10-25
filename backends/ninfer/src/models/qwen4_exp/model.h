#pragma once

// Qwen3.8-Flash-Next (Qwen4ExpForCausalLM): the immutable model a v3 artifact describes. The text
// tower's parameters are bound as the artifact stores them (GGUF blocks, BF16 or NInfer's own
// formats); the frontend resources are the Qwen3.5 tokenizer and chat template, which this family
// shares. Expert banks live on the device of their layer's pipeline stage, or in the pinned Host
// block where the GPU reads them across the bus.

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/stage_plan.h"
#include "core/startup.h"
#include "models/qwen3_5/auxiliary_replicas.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "models/qwen4_exp/config.h"
#include "ninfer/ops/weight_input.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::qwen4_exp {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;
using qwen3_5::WeightUseId;

struct HyperConnectionWeights {
    WeightId norm, down, up;
    std::optional<WeightId> inject; // absent for the final mixer
};

struct SparseAttentionWeights {
    WeightId query, gate, key, value, output, query_norm, key_norm;
    WeightId index_query, index_key, index_query_norm, index_key_norm;
};

struct GdnWeights {
    WeightId query, key, value, z, a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

// Where one expert's rows of one projection sit in the artifact's files (disk-resident experts):
// one run of one file, or two when the rows straddle the boundary between two files.
struct ExpertLocation {
    struct Run {
        std::size_t file     = 0; // index into Model::files()
        std::uint64_t offset = 0; // byte offset in that file
        std::uint64_t bytes  = 0; // 0: unused
    };

    std::array<Run, 2> runs{};
    std::uint64_t bytes    = 0; // of all runs
    QType format           = QType::GGUF_Q8_0;
    std::int64_t row_bytes = 0;
    std::int32_t rows      = 0;
};

struct MoeWeights {
    WeightId router, shared_score;
    // One per expert; empty when the experts stay in the files, which `located` then describes.
    std::vector<WeightId> gate, up, down;
    std::vector<ExpertLocation> located_gate, located_up, located_down;
    WeightId shared_gate, shared_up, shared_down;
};

struct PleWeights {
    WeightId key, value, norm_key, norm_query, norm_conv, convolution;
};

struct LayerWeights {
    HyperConnectionWeights attn_hc, mlp_hc;
    std::variant<GdnWeights, SparseAttentionWeights> mixer;
    MoeWeights moe;
    std::optional<PleWeights> ple;
};

struct TextWeights {
    WeightId token_embedding, output_head;
    HyperConnectionWeights final_mixer;
    std::vector<LayerWeights> layers;
};


struct LoadOptions {
    // The artifact's entry file; its part files sit next to it.
    std::filesystem::path artifact;
    // Pipeline stages, one per device rank; `stage_layers` lists each one's layer count (empty:
    // balanced by the bytes each stage holds).
    std::size_t ranks = 1;
    std::vector<std::uint32_t> stage_layers;
    ExpertResidency experts = ExpertResidency::Device;
    // Loads the Vision tower the artifact carries (its `vision` component, the Qwen3.5 tower) onto
    // the first stage's device, beside the token embedding its output joins.
    bool vision = false;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const StagePlan& stages() const noexcept { return stages_; }

    // The stage that holds the embedding (the first) and the one that holds the final mixer and
    // the output head (the last).
    [[nodiscard]] std::size_t head_rank() const noexcept { return stages_.stages() - 1; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    // The weight's one mathematical use, as a projection operand, its auxiliaries on the weight's
    // device.
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    // The artifact's files, in its order (disk-resident experts are read from them).
    [[nodiscard]] const std::vector<std::filesystem::path>& files() const noexcept {
        return files_;
    }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    // The Vision tower when the model was loaded with it.
    [[nodiscard]] const std::optional<qwen3_5::VisionConfig>& vision_config() const noexcept {
        return vision_config_;
    }
    [[nodiscard]] const std::optional<qwen3_5::VisionWeights>& vision_weights() const noexcept {
        return vision_weights_;
    }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    Model(TextConfig config, LoadOptions options, TextWeights weights, StagePlan stages,
          std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    // Destroy every borrower before the backing.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    LoadOptions options_;
    TextWeights weights_;
    StagePlan stages_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::vector<std::filesystem::path> files_;
    qwen3_5::AuxiliaryReplicas replicas_;
    std::optional<qwen3_5::VisionConfig> vision_config_;
    std::optional<qwen3_5::VisionWeights> vision_weights_;
};

// Whether the artifact's text component is this family.
[[nodiscard]] bool is_qwen4_exp(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::qwen4_exp
