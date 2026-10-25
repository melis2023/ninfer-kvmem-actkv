#pragma once

#include "core/device_snapshot.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"
#include "core/resident_memory.h"

#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// Installs each rank's GPU route profile (EngineOptions::device_profile) before any Op runs,
// calibrating a device that has no measured profile.
void install_device_route_profile_for(const EngineOptions& options, const DeviceContext& device);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    // Keep the fixed Host pool and device accounting alive through Program and weight destruction.
    std::shared_ptr<core::ResidentMemorySession> residency;
    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();

    // Writes every startup-pinned graft (all but PrefillKV) into the Program's shared prefixes. At
    // startup, and again after a worker recovery has released them with the rest of the cache.
    void inject_pinned_grafts();

    // Model suspend (EngineOptions::suspend). The caller holds the instance idle and owns the
    // snapshots: `state` receives the Program's live device state, `weights` (when given) a host
    // copy of the weights, which a resume then uploads instead of reading the artifact again. A
    // failure part-way through a suspend restores what was released before it throws, so the
    // instance stays resident; a failed resume leaves everything released and the snapshots
    // intact, so it may be retried.
    [[nodiscard]] bool suspendable() const noexcept;
    [[nodiscard]] std::uint64_t releasable_device_bytes() const noexcept;
    [[nodiscard]] std::uint64_t weight_host_copy_bytes() const noexcept;
    ResidencyTransition suspend(DeviceContext& device, DeviceSnapshot& state,
                                DeviceSnapshot* weights);
    ResidencyTransition resume(DeviceContext& device, DeviceSnapshot& state,
                               DeviceSnapshot* weights);
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    ContextMachineCostModel context_cost;
    // The options this instance was built from, with any value the model had to resolve (today the
    // single host RAM budget's Host split and long-anchor count) replaced by what the plan actually
    // uses. Carrying it keeps the Engine's copy, its ResourceManager and the frontend grid on one
    // resolved value instead of a plan that silently differs from the reported options.
    EngineOptions options;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
