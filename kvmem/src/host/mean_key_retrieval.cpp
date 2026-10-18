#include "kvmem/mean_key_retrieval.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace kvmem {
std::vector<double> mean_key_retrieval_scores(const RawKvStore& keys, uint32_t blocks,
                                               MeanKeyQueryView query) {
    const uint64_t key_width = uint64_t(query.kv_heads) * query.head_dim;
    const uint64_t query_width = uint64_t(query.query_heads) * query.head_dim;
    if (query.aggregation != GqaScoreAggregation::Mean && query.aggregation != GqaScoreAggregation::Max)
        throw std::invalid_argument("invalid GQA aggregation");
    if (!query.sums || !query.token_counts || !query.layers || !query.head_dim ||
        !query.kv_heads || !query.query_heads || query.query_heads % query.kv_heads ||
        key_width != keys.config().n_embd_k || query.layers != keys.config().n_layer ||
        query_width > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("mean-key retrieval: incompatible query/key geometry");
    }
    const auto group = query.query_heads / query.kv_heads;
    const bool max = query.aggregation == GqaScoreAggregation::Max;
    const uint32_t probes = max ? query.query_heads : query.kv_heads;
    const uint64_t probe_width = uint64_t(probes) * query.head_dim;
    std::vector<float> means(static_cast<size_t>(probe_width) * query.layers, 0.0f);
    std::vector<float> norms(static_cast<size_t>(probes) * query.layers, 0.0f);
    uint32_t active_layers = 0;
    for (uint32_t layer = 0; layer < query.layers; ++layer) {
        if (!query.token_counts[layer]) continue;
        ++active_layers;
        const float inv = max ? 1.0f : 1.0f / static_cast<float>(uint64_t(query.token_counts[layer]) * group);
        for (uint32_t head = 0; head < probes; ++head) {
            float norm = 0.0f;
            for (uint32_t d = 0; d < query.head_dim; ++d) {
                float sum = 0.0f;
                for (uint32_t g = 0; g < (max ? 1U : group); ++g) {
                    const float value = query.sums[layer * query_width +
                        (max ? head : head * group + g) * query.head_dim + d];
                    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite mean-key query");
                    sum += value;
                }
                const float mean = sum * inv;
                means[layer * probe_width + head * query.head_dim + d] = mean;
                norm += mean * mean;
            }
            norms[layer * probes + head] = std::sqrt(norm);
        }
    }
    std::vector<double> scores(blocks, 0.0);
    if (!active_layers) return scores;
    std::vector<float> key(static_cast<size_t>(key_width));
    for (uint32_t block = 0; block < blocks; ++block) {
        if (!keys.has_block(block)) continue;
        double total = 0.0;
        for (uint32_t layer = 0; layer < query.layers; ++layer) {
            if (!query.token_counts[layer]) continue;
            keys.mean_k(block, layer, key.data());
            double layer_score = 0.0;
            for (uint32_t head = 0; head < query.kv_heads; ++head) {
                float norm = 0.0f;
                for (uint32_t d = 0; d < query.head_dim; ++d) {
                    const float value = key[head * query.head_dim + d];
                    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite mean-key index");
                    norm += value * value;
                }
                const float key_norm = std::sqrt(norm);
                float best = -std::numeric_limits<float>::infinity();
                for (uint32_t g = 0; g < (max ? group : 1U); ++g) {
                    const uint32_t probe = max ? head * group + g : head;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < query.head_dim; ++d)
                        dot += means[layer * probe_width + probe * query.head_dim + d] *
                            key[head * query.head_dim + d];
                    const float cosine = dot / (norms[layer * probes + probe] * key_norm + 1e-6f);
                    best = std::max(best, cosine);
                }
                layer_score += best;
            }
            total += layer_score / query.kv_heads;
        }
        scores[block] = total / active_layers;
    }
    return scores;
}
} // namespace kvmem
