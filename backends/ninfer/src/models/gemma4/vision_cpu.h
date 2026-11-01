#pragma once

// Gemma 4's vision tower on CPU threads, for `--vision-residency cpu`: the device tower's
// arithmetic (vision.cpp) in FP32 over weights decoded from the host-resident artifact at load, its
// products and attention blocked into the matrix kernels of vision_cpu_kernels.h (AVX2 and FMA when
// the CPU has them). Only the soft tokens' embeddings reach the device, staged into the prompt chunk
// that holds them. One batch encodes at a time across the process, since each already spreads over
// every core.

#include "models/gemma4/model.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::gemma4 {

class CpuVisionTower {
public:
    // `threads`: the CPU threads an encode uses (0: all hardware threads).
    explicit CpuVisionTower(const Model& model, unsigned threads = 0);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

    // `units` units of grid_w x grid_h patches (BF16 [units][grid_h * grid_w][patch features])
    // to their soft tokens' text embeddings, BF16 [units][tokens][projection_dim], each unit's
    // tokens row by row of its pooled grid.
    [[nodiscard]] std::vector<std::uint16_t> encode(std::span<const std::uint16_t> patches,
                                                    int units, int grid_w, int grid_h) const;

private:
    struct Layer {
        std::vector<float> qkv, output, gate_up, down; // [rows][k] row-major
        std::vector<float> input_norm, post_attention_norm, ffn_norm, post_ffn_norm, q_norm,
            k_norm;
    };

    VisionConfig config_;
    unsigned threads_ = 0;
    std::vector<float> patch_embedding_, projection_, position_table_, std_bias_, std_scale_;
    std::vector<Layer> layers_;
    std::vector<float> inv_freq_;
};

} // namespace ninfer::models::gemma4
