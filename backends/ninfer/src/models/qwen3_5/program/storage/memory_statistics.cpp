#include "models/qwen3_5/program/program_impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {
void read_statistics(DeviceContext& device, const execution::MemoryStatistics& statistics,
                     void* destination, bool include_query) {
    auto* host = static_cast<std::byte*>(destination);
    const auto key_stride = std::size_t(statistics.key_width) * statistics.columns * sizeof(float);
    const auto query_stride = std::size_t(statistics.query_width) * sizeof(float);
    for (std::size_t rank = 0; rank < statistics.ranks.size(); ++rank) {
        const auto& shard = statistics.ranks[rank];
        if (shard.key_sums.data == nullptr) { continue; }
        DeviceBinding binding(device.rank(rank).device);
        const auto stream = device.stream_for_rank(rank);
        CUDA_CHECK(cudaMemcpyAsync(host + shard.first_layer * key_stride,
            shard.key_sums.data, shard.key_sums.bytes(), cudaMemcpyDeviceToHost, stream));
        if (include_query) {
            CUDA_CHECK(cudaMemcpyAsync(host + statistics.key_bytes() + shard.first_layer * query_stride,
                shard.query_sums.data, shard.query_sums.bytes(), cudaMemcpyDeviceToHost, stream));
        }
    }
    device.synchronize();
}
} // namespace

void ProgramImpl::prepare_memory_statistics(SequenceState& sequence, std::uint32_t query_begin,
                                            std::uint32_t query_end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics) {
        const auto& config = parameters.model.config().text;
        state.statistics = std::make_unique<kvmem::RawKvStore>(kvmem::RawKvStoreConfig{
            .n_layer = config.full_attention_layers,
            .n_embd_k = memory_statistics->key_width,
            .block_tokens = 64});
        kvmem::KvMemStoreConfig selection;
        selection.lrfu = kvmem_options.lrfu;
        selection.lrfu_lambda = kvmem_options.lrfu_lambda;
        selection.lrfu_top_fraction = kvmem_options.lrfu_top_fraction;
        selection.block_tokens = 64;
        selection.select_budget = kvmem_options.selected_tokens;
        selection.sink_blocks = kvmem_options.sink_blocks();
        selection.recent_blocks = kvmem_options.recent_blocks();
        state.selector = std::make_unique<kvmem::KvMemStore>(selection);
        state.selector->begin_retrieval_round();
        state.query_sum.assign(memory_statistics->query_elements(), 0.0F);
        state.query_begin = query_begin;
        state.query_end = query_end;
    }
    if (state.query_begin != query_begin || state.query_end != query_end) {
        throw std::logic_error("KVMem query range changed inside one prefill");
    }
    memory_statistics_ranges = {0, static_cast<std::int32_t>(capacity),
        static_cast<std::int32_t>(query_begin), static_cast<std::int32_t>(query_end)};
    for (std::size_t rank = 0; rank < memory_statistics->ranks.size(); ++rank) {
        const auto& shard = memory_statistics->ranks[rank];
        if (shard.ranges.data == nullptr) { continue; }
        DeviceBinding binding(device.rank(rank).device);
        CUDA_CHECK(cudaMemcpyAsync(shard.ranges.data, memory_statistics_ranges.data(),
            sizeof(memory_statistics_ranges), cudaMemcpyHostToDevice, device.stream_for_rank(rank)));
    }
}

void ProgramImpl::commit_memory_statistics(SequenceState& sequence, std::uint32_t begin,
                                           std::uint32_t end) {
    if (!memory_statistics) { return; }
    auto& state = sequence.window;
    if (!state.statistics || begin != state.statistics_frontier || end <= begin ||
        end - begin > prefill_chunk) {
        throw std::logic_error("KVMem statistics require exactly one append at the evaluated frontier");
    }
    const auto& buffers = *memory_statistics;
    auto* host = static_cast<std::byte*>(memory_statistics_host->data());
    read_statistics(device, buffers, host, true);
    const auto width = buffers.key_width;
    const auto layer_stride = static_cast<std::size_t>(width) * buffers.columns;
    const auto* keys = reinterpret_cast<const float*>(host);
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto at = begin; at < end;) {
            const auto count = std::min(end - at, 64 - at % 64);
            const auto bucket = at / 64 - begin / 64;
            const auto* sum = keys + layer * layer_stride + bucket * width;
            for (std::uint32_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(sum[channel])) { throw std::runtime_error("Nonfinite KVMem key sum"); }
            }
            state.statistics->write_layer_mean_sum(at, count, layer, sum);
            at += count;
        }
    }
    const auto query_first = std::max(begin, state.query_begin);
    const auto query_last = std::min(end, state.query_end);
    if (!state.query_replayed && query_last > query_first) {
        const auto* query = reinterpret_cast<const float*>(host + buffers.key_bytes());
        for (std::size_t i = 0; i < state.query_sum.size(); ++i) {
            if (!std::isfinite(query[i])) { throw std::runtime_error("Nonfinite KVMem query sum"); }
            state.query_sum[i] += query[i];
        }
        state.query_tokens += query_last - query_first;
    }
    if (state.selector->total_tokens() != begin) {
        throw std::logic_error("KVMem block catalog lost its statistics frontier");
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
    if (kvmem_options.verify_transfers && (end == state.query_end || end % 64 == 0)) {
        std::fprintf(stderr, "KVMEM_STATISTICS frontier=%u query_tokens=%u query_begin=%u query_end=%u layers=%u mean_bytes=%zu\n",
            end, state.query_tokens, state.query_begin, state.query_end,
            state.statistics->config().n_layer, state.statistics->allocated_bytes());
    }
}
void ProgramImpl::read_memory_candidates() {
    if (!memory_candidate_statistics) return;
    // Also pull the current round's query aggregate for decode pressure. It is
    // one FP32 row per layer and costs far less than the candidate keys.
    read_statistics(device, *memory_candidate_statistics, memory_candidate_host->data(),
                    kvmem_options.decode_pressure);
}

void ProgramImpl::prepare_memory_feedback(SequenceState& sequence, std::uint32_t begin,
                                          std::uint32_t end) {
    if (!memory_candidate_statistics || !kvmem_options.decode_pressure ||
        !sequence.window.statistics || end <= begin) {
        return;
    }
    memory_statistics_ranges[2] = static_cast<std::int32_t>(begin);
    memory_statistics_ranges[3] = static_cast<std::int32_t>(end);
    for (std::size_t rank = 0; rank < memory_candidate_statistics->ranks.size(); ++rank) {
        auto& shard = memory_candidate_statistics->ranks[rank];
        // Decode rows are the only source for this aggregate. The persistent
        // prefill query_sum is copied to the host before the first decode round.
        shard.capture_query = true;
        if (shard.ranges.data == nullptr) { continue; }
        DeviceBinding binding(device.rank(rank).device);
        const auto stream = device.stream_for_rank(rank);
        CUDA_CHECK(cudaMemcpyAsync(shard.ranges.data, memory_statistics_ranges.data(),
            sizeof(memory_statistics_ranges), cudaMemcpyHostToDevice, stream));
    }
}

void ProgramImpl::note_accepted_tokens(SequenceState& sequence, std::uint32_t accepted,
                                       std::uint32_t width) {
    if (!kvmem_options.decode_pressure || !memory_candidate_statistics || accepted == 0 ||
        !sequence.window.statistics) {
        return;
    }
    auto& state = sequence.window;
    const auto interval = std::max<std::uint32_t>(1U, kvmem_options.decode_pressure_interval);
    state.accepted_since_sample += accepted;
    if (state.accepted_since_sample < interval) { return; }
    state.accepted_since_sample -= interval;

    const auto& config = parameters.model.config().text;
    const auto layers = config.full_attention_layers;
    const auto heads = config.attention->num_attention_heads;
    const auto kv_heads = config.attention->num_key_value_heads;
    const auto head_dim = config.attention->head_dim;
    const auto group = heads / kv_heads;
    if (!layers || !heads || !kv_heads || !group || !head_dim || !width) { return; }

    // Decode pressure is a one-hop-free, resident-only ranking: the current
    // round's accepted query rows score the pages that are still in the
    // working set. Rejected draft columns dilute the round mean, so the score
    // is an approximation of accepted attention rather than a per-column sum.
    const auto* bytes = static_cast<const float*>(memory_candidate_host->data());
    const auto* query = bytes + memory_candidate_statistics->key_bytes() / sizeof(float);
    const auto layer = layers - 1;
    const auto query_width = heads * head_dim;
    const float* q_layer = query + static_cast<std::size_t>(layer) * query_width;
    std::vector<float> mean_q(static_cast<std::size_t>(kv_heads) * head_dim, 0.0F);
    const float inv = 1.0F / static_cast<float>(static_cast<double>(width) * group);
    for (std::uint32_t head = 0; head < kv_heads; ++head) {
        for (std::uint32_t d = 0; d < head_dim; ++d) {
            float sum = 0.0F;
            for (std::uint32_t g = 0; g < group; ++g) {
                sum += q_layer[(head * group + g) * head_dim + d];
            }
            mean_q[head * head_dim + d] = sum * inv;
        }
    }

    const auto pages = kv_pages_for_tokens(sequence.text_kv_valid);
    if (!pages) { return; }
    auto& mass = state.attention_mass;
    if (mass.size() < pages) { mass.resize(pages, 0.0); }
    const double decay = std::exp(-static_cast<double>(interval) / 8192.0 * 0.6931471805599453);
    for (auto& value : mass) { value *= decay; }

    std::vector<float> key(static_cast<std::size_t>(kv_heads) * head_dim, 0.0F);
    std::vector<std::pair<std::uint32_t, double>> scored;
    scored.reserve(state.pages.size());
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto page : state.pages) {
        if (page >= pages || !state.statistics->has_block(page)) { continue; }
        state.statistics->mean_k(page, layer, key.data());
        double total = 0.0;
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            double dot = 0.0, qn = 0.0, kn = 0.0;
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                const double qv = mean_q[head * head_dim + d];
                const double kv = key[head * head_dim + d];
                dot += qv * kv;
                qn += qv * qv;
                kn += kv * kv;
            }
            total += dot / (std::sqrt(qn * kn) + 1e-6);
        }
        const double score = total / kv_heads;
        scored.emplace_back(page, score);
        maximum = std::max(maximum, score);
    }
    if (scored.empty()) { return; }
    double normalizer = 0.0;
    for (auto& entry : scored) {
        entry.second = std::exp((entry.second - maximum) * 8.0);
        normalizer += entry.second;
    }
    if (!(normalizer > 0.0)) { return; }
    for (const auto& entry : scored) {
        mass[entry.first] += entry.second / normalizer;
    }
    std::fprintf(stderr, "KVMEM_FEEDBACK frontier=%u resident=%zu pages=%u\n",
        sequence.text_kv_valid, scored.size(), pages);
}

void ProgramImpl::commit_memory_candidates(SequenceState& sequence, std::uint32_t begin,
                                            std::uint32_t end, std::uint32_t first_column,
                                            std::uint32_t row_width) {
    if (!memory_candidate_statistics) { return; }
    auto& state = sequence.window;
    const auto& keys = *memory_candidate_statistics;
    if (!state.statistics || !state.selector || begin != state.statistics_frontier ||
        state.selector->total_tokens() != begin || end <= begin ||
        end - begin > row_width || first_column + row_width > keys.columns) {
        throw std::logic_error("KVMem speculative statistics have an invalid committed prefix");
    }
    const auto width = static_cast<std::size_t>(keys.key_width);
    const auto stride = width * keys.columns;
    const auto* values = static_cast<const float*>(memory_candidate_host->data());
    // Each represented BF16 key was copied into its own FP32 column by
    // token_sums(block_tokens=1). Rejected and output-truncated columns never
    // enter the persistent mean-K store, including a partial logical page.
    for (std::uint32_t layer = 0; layer < state.statistics->config().n_layer; ++layer) {
        for (auto position = begin; position < end; ++position) {
            const auto* value = values + layer * stride + (first_column + position - begin) * width;
            for (std::size_t channel = 0; channel < width; ++channel) {
                if (!std::isfinite(value[channel])) {
                    throw std::runtime_error("Nonfinite committed KVMem candidate key");
                }
            }
            state.statistics->write_layer_mean_sum(position, 1, layer, value);
        }
    }
    state.selector->register_append(end - begin);
    state.statistics_frontier = end;
}
} // namespace ninfer::models::qwen3_5::detail
