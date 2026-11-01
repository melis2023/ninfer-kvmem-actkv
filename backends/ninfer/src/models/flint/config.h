#pragma once

// The text config of Flint (FlintForCausalLM) as tools/convert/flint.py writes it into the
// artifact: grouped-query attention with per-head QK RMS norms and a learned sink per query head,
// sliding-window blocks with RoPE and full-attention blocks without, leading dense SwiGLU blocks,
// then routed SwiGLU experts (sigmoid scores, the top k renormalized) beside an ungated shared
// SwiGLU expert, and an untied output head.

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::flint {

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float rope_theta                      = 0;
    // Per block: true for full attention, false for a sliding window of `sliding_window` positions
    // (the query's own included).
    std::vector<bool> full_attention;
    // Per block: whether its queries and keys turn (NeoX pairs (i, i + D/2)).
    std::vector<bool> rope;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t sliding_window      = 0;
    // Blocks [0, dense_layers) have a dense SwiGLU MLP of intermediate_size; the rest are MoE.
    std::uint32_t dense_layers            = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_experts             = 0;
    std::uint32_t top_k_experts           = 0;
    std::uint32_t moe_intermediate_size   = 0;
    std::uint32_t shared_intermediate_size = 0;

    [[nodiscard]] bool moe(std::uint32_t layer) const noexcept { return layer >= dense_layers; }
    [[nodiscard]] std::uint32_t moe_layers() const noexcept { return num_hidden_layers - dense_layers; }
};

// Throws artifact::ArtifactError naming the field that is missing, malformed or outside the
// implemented geometry.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);

} // namespace ninfer::models::flint
