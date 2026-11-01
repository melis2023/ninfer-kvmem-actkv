#pragma once
#include "core/tensor.h"
#include <vector>

namespace ninfer::models::qwen3_5::execution {
// Stable Program-owned buffers, borrowed by eager execution and graph capture.
// Each shard holds its stage's attention layers. Host stores use global layer order.
// Key sums: FP32 [key_width,buckets,layers]. Query values are either compact
// BF16 [query_width,query_capacity,layers] or seeded FP32 [query_width,1,layers]
// for Mean-Q; attention feedback uses its separate post-RoPE capture.
// Ranges: [K begin,end,Q begin,end].
// Only normalized, represented BF16 values BEFORE RoPE enter these reductions.
struct MemoryStatisticsShard {
    Tensor key_sums;
    Tensor query_values;
    Tensor attention_queries; // BF16 post-RoPE target Q, three global spread3 slots
    Tensor attention_tokens;
    Tensor attention_rows, attention_positions, attention_scratch, attention_norms, attention_output;
    Tensor ranges;
    // Prefill aggregates complete logical blocks; speculative verification keeps
    // one column per candidate until the Frontend commits an exact prefix.
    std::uint32_t block_tokens = 64;
    Tensor candidate_origin;
    Tensor prefill_origin;
    std::uint32_t first_layer = 0;
};

struct MemoryStatistics {
    std::vector<MemoryStatisticsShard> ranks;
    std::uint32_t key_width = 0;
    std::uint32_t query_width = 0;
    std::uint32_t layers = 0;
    std::uint32_t columns = 0;
    std::uint32_t query_columns = 0;
    bool mean_query = false;

    [[nodiscard]] std::size_t query_element_bytes() const noexcept {
        return mean_query ? sizeof(float) : sizeof(std::uint16_t);
    }

    [[nodiscard]] std::size_t key_bytes() const noexcept {
        return std::size_t(key_width) * columns * layers * sizeof(float);
    }
    [[nodiscard]] std::size_t query_elements() const noexcept {
        return std::size_t(query_width) * query_columns * layers;
    }
    [[nodiscard]] std::size_t query_bytes() const noexcept {
        return query_elements() * query_element_bytes();
    }
};
} // namespace ninfer::models::qwen3_5::execution
