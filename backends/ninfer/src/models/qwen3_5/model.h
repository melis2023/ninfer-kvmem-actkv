#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/dspine/model.h"
#include "models/qwen3_5/auxiliary_replicas.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/expert_sources.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

class CpuVisionCache;

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const NativeExpertSources& expert_sources() const noexcept { return experts_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    // A Use whose matrix is Hadamard-rotated is only admitted where the execution layer rotates
    // the activation: input() refuses it, rotated_input() carries its sign vector. Auxiliaries
    // (sign vectors, input gathers) are read on the weight's own device.
    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] ops::WeightInput rotated_input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput rotated_input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    // --spec dspine: the standalone drafter, materialized on rank 0 beside the target; null
    // otherwise.
    [[nodiscard]] const dspine::Model* dspine() const noexcept { return dspine_.get(); }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Overlay Vision residency only: the pinned tower groups, the pinned block that holds them and
    // the eviction pool behind the weight arena. The pool is shared mutable device state; its
    // single window is arbitrated by the one Program that executes this Model.
    [[nodiscard]] const std::optional<VisionOverlayLayout>& vision_overlay() const noexcept {
        return vision_overlay_;
    }

    [[nodiscard]] std::span<const std::byte> pinned_weights() const noexcept {
        return backing_.pinned_block();
    }

    [[nodiscard]] EvictableWeightPool* weight_pool() const noexcept {
        return backing_.weight_pool();
    }

    // Model suspend: the weight arenas give their device memory back and later receive the same
    // bytes again from the artifact, at the same addresses, so every bound weight and captured
    // graph stays valid. The caller has drained every stream that reads them.
    // A DSpine drafter's weights take part in every step, and so does an external DFlash drafter's.
    [[nodiscard]] bool weights_suspendable() const noexcept {
        return backing_.suspendable() && (!dspine_ || dspine_->backing().suspendable()) &&
               (!draft_backing_ || draft_backing_->suspendable());
    }
    [[nodiscard]] bool weights_resident() const noexcept { return backing_.device_backed(); }
    [[nodiscard]] std::uint64_t weight_backing_bytes() const noexcept {
        return backing_.device_backing_bytes() +
               (dspine_ ? dspine_->backing().device_backing_bytes() : 0) +
               (draft_backing_ ? draft_backing_->device_backing_bytes() : 0);
    }
    [[nodiscard]] std::vector<artifact::MaterializedArtifact::RankArena>
    device_weight_arenas() const {
        auto arenas = backing_.device_arenas();
        if (dspine_) {
            for (const auto& arena : dspine_->backing().device_arenas()) { arenas.push_back(arena); }
        }
        if (draft_backing_) {
            for (const auto& arena : draft_backing_->device_arenas()) { arenas.push_back(arena); }
        }
        return arenas;
    }
    void release_device_weights() {
        backing_.release_device_backing();
        if (dspine_) { dspine_->backing().release_device_backing(); }
        if (draft_backing_) { draft_backing_->release_device_backing(); }
    }
    void restore_device_weight_backing(DeviceContext& device) {
        backing_.restore_device_backing(device);
        if (dspine_) { dspine_->backing().restore_device_backing(device); }
        if (draft_backing_) { draft_backing_->restore_device_backing(device); }
    }
    artifact::MaterializationStats reload_device_weights(DeviceContext& device) {
        auto stats = backing_.upload_device_objects_again(device);
        if (dspine_) {
            const auto drafter = dspine_->backing().upload_device_objects_again(device);
            stats.file_bytes += drafter.file_bytes;
            stats.read_bytes += drafter.read_bytes;
            stats.h2d_bytes += drafter.h2d_bytes;
            stats.upload_seconds += drafter.upload_seconds;
        }
        if (draft_backing_) {
            const auto drafter = draft_backing_->upload_device_objects_again(device);
            stats.file_bytes += drafter.file_bytes;
            stats.read_bytes += drafter.read_bytes;
            stats.h2d_bytes += drafter.h2d_bytes;
            stats.upload_seconds += drafter.upload_seconds;
        }
        return stats;
    }

    // CPU Vision capability and lazy Host residency, independent of whether weights are loaded.
    [[nodiscard]] const std::shared_ptr<CpuVisionCache>& cpu_vision() const noexcept {
        return cpu_vision_;
    }

private:
    // Attaches a coded weight's side planes to its WeightInput: device planes through the replica
    // helper, the descriptor straight from its host placement.
    void attach_trellis(ops::WeightInput& result, const WeightUse& use) const;
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*,
                                                    const artifact::MaterializeOptions&);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          AuxiliaryReplicas replicas, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing,
          std::optional<artifact::MaterializedArtifact> draft_backing,
          std::optional<VisionOverlayLayout> vision_overlay,
          std::shared_ptr<CpuVisionCache> cpu_vision,
          NativeExpertSources experts);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    // --draft-model on DFlash: the drafter's own weights, in their own arena. Absent when the
    // drafter rides in the target artifact or is a DSpine Model of its own.
    std::optional<artifact::MaterializedArtifact> draft_backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    AuxiliaryReplicas replicas_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<VisionOverlayLayout> vision_overlay_;
    std::shared_ptr<CpuVisionCache> cpu_vision_;
    NativeExpertSources experts_;
    std::unique_ptr<dspine::Model> dspine_;
};

} // namespace ninfer::models::qwen3_5
