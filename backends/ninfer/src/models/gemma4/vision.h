#pragma once

// Gemma 4's vision tower on the device: a batch of media units (an image, or frames of a video) of
// one patch grid in, each unit's soft tokens out as text-model input embeddings. The patch
// projection adds the x and y position rows; every block (sandwich RMSNorms, per-head Q/K/V norms,
// axial RoPE, unscaled attention over the unit's own patches, a GeGLU MLP) multiplies BF16 matrices
// by BF16 inputs into FP32 results over an FP32 residual; pooling averages pooling x pooling
// patches, scales by sqrt(hidden), standardizes and projects into the text model's width.
//
// The blocks' matrices are resident on the device, or (VisionPlacement::Pinned) stay in pinned
// host memory and pass through a window of two device slots: a block's matrices upload on a copy
// stream while the block before it computes. The rest of the tower (patch and output projections,
// position table, norms) is resident either way.

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "models/gemma4/model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::gemma4 {

class VisionTower {
public:
    // `max_patches`: the most patches one encode() batch takes, which sizes its scratch.
    VisionTower(const Model& model, std::uint32_t max_patches);
    ~VisionTower();
    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::uint32_t max_patches() const noexcept { return max_patches_; }
    // Whether the blocks stream through the device window.
    [[nodiscard]] bool streamed() const noexcept { return streamed_; }
    // Device bytes the tower owns: the window and, streamed, its resident copies.
    [[nodiscard]] std::size_t owned_bytes() const noexcept { return owned_bytes_; }
    // Scratch one encode() of up to max_patches patches takes from its workspace, the patches
    // and the soft tokens included.
    [[nodiscard]] std::size_t workspace_bytes() const;

    // `units` units of grid_w x grid_h patches: their patches (BF16 [patches][patch features] on
    // the device) to their soft tokens' text embeddings, BF16 [tokens][projection_dim] in `out`,
    // unit after unit, each row by row of its pooled grid.
    void encode(const __nv_bfloat16* patches, int units, int grid_w, int grid_h,
                __nv_bfloat16* out, WorkspaceArena& workspace, cudaStream_t stream) const;

private:
    struct Layer {
        Weight qkv, output, gate_up, down;
        const float *input_norm = nullptr, *post_attention_norm = nullptr, *ffn_norm = nullptr,
                    *post_ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    };

    // Streamed: layer i's matrices in window slot `slot`, uploading them first.
    [[nodiscard]] Layer staged(std::size_t i, int slot) const;
    void upload(std::size_t i, int slot, cudaStream_t stream) const;
    // A resident copy of `bytes` host bytes the tower owns.
    [[nodiscard]] const void* own(const void* host, std::size_t bytes);

    VisionConfig config_;
    std::uint32_t max_patches_ = 0;
    bool streamed_             = false;
    Weight patch_embedding_, projection_;
    const __nv_bfloat16* position_table_ = nullptr;
    const float *std_bias_ = nullptr, *std_scale_ = nullptr;
    std::vector<Layer> layers_;
    DeviceBuffer inv_freq_; // [head_dim / 4]
    std::vector<DeviceBuffer> owned_;
    std::size_t owned_bytes_ = 0;
    // The window: two slots of one block's matrices, the stream they upload on, and per slot the
    // upload's completion and the last use of what it held.
    DeviceBuffer slots_[2];
    std::size_t slot_bytes_    = 0;
    cudaStream_t copy_stream_  = nullptr;
    cudaEvent_t copied_[2]     = {nullptr, nullptr};
    cudaEvent_t used_[2]       = {nullptr, nullptr};
};

} // namespace ninfer::models::gemma4
