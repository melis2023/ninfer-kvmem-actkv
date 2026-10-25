#pragma once
#include "core/tensor.h"
#include <vector>

namespace ninfer::models::qwen3_5::execution {
// Stable Program-owned buffers, borrowed by eager execution and graph capture.
// Each shard holds its stage's attention layers. Host stores use global layer order.
// Key sums: [key_width,buckets,layers]; query sums: [query_width,1,layers].
// Ranges: [K begin,end,Q begin,end].
// Only normalized, represented BF16 values BEFORE RoPE enter these reductions.
struct MemoryStatisticsShard {
    Tensor key_sums;
    Tensor query_sums;
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

    [[nodiscard]] std::size_t key_bytes() const noexcept {
        return std::size_t(key_width) * columns * layers * sizeof(float);
    }
    [[nodiscard]] std::size_t query_elements() const noexcept {
        return std::size_t(query_width) * layers;
    }
    [[nodiscard]] std::size_t query_bytes() const noexcept {
        return query_elements() * sizeof(float);
    }
};
} // namespace ninfer::models::qwen3_5::execution
