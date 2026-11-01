#pragma once

// The text config of Gemma 4 (Gemma4ForCausalLM) as tools/convert/gemma4.py writes it into the
// artifact: sliding-window and full-attention blocks, full attention with K = V and proportional
// RoPE, a GeGLU MLP and, in the MoE models, a routed expert bank beside it in every block.

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::gemma4 {

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float final_logit_softcapping         = 0; // 0: none
    // true: full attention; false: sliding-window attention.
    std::vector<bool> full_attention;

    std::uint32_t num_attention_heads = 0;
    // Sliding-window blocks.
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t sliding_window      = 0;
    float rope_theta_sliding          = 0;
    // Full-attention blocks: K is also V; RoPE turns the first `rotary_pairs_full` pairs (i, i +
    // D/2) at theta^(-2i/D) and leaves the rest.
    std::uint32_t num_global_key_value_heads = 0;
    std::uint32_t global_head_dim            = 0;
    float rope_theta_full                    = 0;
    std::uint32_t rotary_pairs_full          = 0;

    std::uint32_t intermediate_size = 0;
    bool moe                        = false;
    std::uint32_t num_experts       = 0;
    std::uint32_t top_k_experts     = 0;
    std::uint32_t moe_intermediate_size = 0;

    [[nodiscard]] std::uint32_t layer_head_dim(std::uint32_t layer) const {
        return full_attention.at(layer) ? global_head_dim : head_dim;
    }
    [[nodiscard]] std::uint32_t layer_kv_heads(std::uint32_t layer) const {
        return full_attention.at(layer) ? num_global_key_value_heads : num_key_value_heads;
    }
};

// Throws artifact::ArtifactError naming the field that is missing, malformed or outside the
// implemented geometry.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);

// The multi-token-prediction assistant (Gemma4AssistantForCausalLM), the artifact's `mtp`
// component: a dense drafter of its own width whose blocks attend over the target's KV (its
// sliding-window blocks the target's last sliding-window block, its full-attention block the
// target's last full-attention block) with the target's head geometry, fed the target's embedding
// of a token beside the target's final hidden state.
struct AssistantConfig {
    std::uint32_t hidden_size          = 0;
    std::uint32_t backbone_hidden_size = 0; // the target's hidden_size
    std::uint32_t intermediate_size    = 0;
    std::uint32_t vocab_size           = 0;
    std::uint32_t num_attention_heads  = 0;
    float rms_norm_eps                 = 0;
    std::vector<bool> full_attention;
};

// Checked against the target: equal vocabulary and attention geometry, a backbone of the target's
// width, and both kinds of block.
[[nodiscard]] AssistantConfig parse_assistant_config(const artifact::Json& value,
                                                     const TextConfig& target);

// The vision tower (Gemma4VisionModel and the projection into the text model), the artifact's
// `vision` component. A media unit (an image, or one frame of a video) is a grid of patch x patch
// pixels; the tower's blocks attend over all of its patches, and its average over pooling x pooling
// neighbourhoods, standardized and projected, gives the unit's soft tokens, which take the place
// of the prompt's media placeholders. In the text model's sliding-window blocks one unit's soft
// tokens attend each other in both directions when `bidirectional_attention`.
struct VisionConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t head_dim                = 0;
    std::uint32_t patch_size              = 0;
    std::uint32_t pooling_kernel_size     = 0;
    std::uint32_t position_embedding_size = 0; // rows of each of the x and y tables
    std::uint32_t projection_dim          = 0; // the text model's hidden size
    std::uint32_t default_output_length   = 0; // soft tokens of an image by default
    float rope_theta                      = 0;
    float rms_norm_eps                    = 0;
    bool standardize                      = false;
    bool bidirectional_attention          = false;
    std::int32_t image_token_id = -1, video_token_id = -1, boi_token_id = -1, eoi_token_id = -1;

    // Values of one patch, (channel, y, x) with x fastest.
    [[nodiscard]] std::uint32_t patch_features() const noexcept {
        return 3 * patch_size * patch_size;
    }
    [[nodiscard]] std::uint32_t patches_per_token() const noexcept {
        return pooling_kernel_size * pooling_kernel_size;
    }
};

// Checked against the target and against the implemented kernels (a 1152-wide tower of 16 heads of
// 72, a projection to the target's width).
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& value,
                                               const TextConfig& target);

} // namespace ninfer::models::gemma4
