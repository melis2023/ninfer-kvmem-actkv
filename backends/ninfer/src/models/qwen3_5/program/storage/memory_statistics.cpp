#include "models/qwen3_5/program/program_impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include "ninfer/ops/attention_page_mass.h"

namespace ninfer::models::qwen3_5::detail {
namespace {
void read_statistics(DeviceContext& device, const execution::MemoryStatistics& statistics,
                     void* destination, bool include_query,
                     std::uint32_t query_first = 0, std::uint32_t query_count = 0) {
    auto* host = static_cast<std::byte*>(destination);
    const auto key_stride = std::size_t(statistics.key_width) * statistics.columns * sizeof(float);
    const auto query_stride =
        std::size_t(statistics.query_width) * statistics.query_columns * sizeof(float);
    for (std::size_t rank = 0; rank < statistics.ranks.size(); ++rank) {
        const auto& shard = statistics.ranks[rank];
        if (shard.key_sums.data == nullptr) { continue; }
        DeviceBinding binding(device.rank(rank).device);
        const auto stream = device.stream_for_rank(rank);
        CUDA_CHECK(cudaMemcpyAsync(host + shard.first_layer * key_stride,
            shard.key_sums.data, shard.key_sums.bytes(), cudaMemcpyDeviceToHost, stream));
        if (include_query) {
            const auto count = query_count ? query_count : statistics.query_columns;
            const auto offset = std::size_t(query_first) * statistics.query_width * sizeof(float);
            CUDA_CHECK(cudaMemcpy2DAsync(
                host + statistics.key_bytes() + shard.first_layer * query_stride + offset,
                query_stride,
                static_cast<const std::byte*>(shard.query_sums.data) + offset, query_stride,
                std::size_t(count) * statistics.query_width * sizeof(float),
                shard.query_sums.ne[2], cudaMemcpyDeviceToHost, stream));
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
        state.graph.set_budget(std::make_shared<kvmem::GraphStatisticsBudget>(32u << 20,
                                                                             memory_graph_budget));
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
        state.query_rows.resize(memory_statistics->layers);
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
    const auto query_first = std::max(begin, state.query_begin);
    const auto query_last = std::min(end, state.query_end);
    const bool read_query = !state.query_replayed && query_last > query_first;
    read_statistics(device, buffers, host, read_query, read_query ? query_first - begin : 0,
        read_query ? query_last - query_first : 0);
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
    if (read_query) {
        const auto* query = reinterpret_cast<const float*>(host + buffers.key_bytes());
        for (std::uint32_t layer = 0; layer < buffers.layers; ++layer) {
            const auto* first = query + (std::size_t(layer) * buffers.query_columns +
                query_first - begin) * buffers.query_width;
            const auto elements = std::size_t(query_last - query_first) * buffers.query_width;
            for (std::size_t i = 0; i < elements; ++i) {
                if (!std::isfinite(first[i])) { throw std::runtime_error("Nonfinite KVMem Query row"); }
            }
            state.query_rows[layer].insert(state.query_rows[layer].end(), first, first + elements);
        }
        state.query_tokens += query_last - query_first;
    }
    if (state.selector->total_tokens() != begin) {
        throw std::logic_error("KVMem block catalog lost its statistics frontier");
    }
    state.selector->register_append(end - begin);
    observe_memory_attention(sequence, begin, end, 0, end - begin, true);
    state.statistics_frontier = end;
    if (kvmem_options.verify_transfers && (end == state.query_end || end % 64 == 0)) {
        std::fprintf(stderr, "KVMEM_STATISTICS frontier=%u query_tokens=%u query_begin=%u query_end=%u layers=%u mean_bytes=%zu\n",
            end, state.query_tokens, state.query_begin, state.query_end,
            state.statistics->config().n_layer, state.statistics->allocated_bytes());
    }
}

std::vector<double> ProgramImpl::score_memory_query(SequenceState& sequence,
                                     const std::vector<std::uint32_t>& mandatory) {
    auto& state = sequence.window;
    if (!state.query_tokens || !state.statistics || !state.selector) {
        throw std::logic_error("KVMem Query scoring has no accepted index");
    }
    const auto& config = parameters.model.config().text;
    std::vector<const float*> layers(config.full_attention_layers);
    std::vector<std::uint32_t> counts(config.full_attention_layers, state.query_tokens);
    const auto width = config.attention->query_width();
    if (state.query_rows.size() != layers.size()) {
        throw std::logic_error("KVMem Query layer coverage mismatch");
    }
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        if (state.query_rows[layer].size() != std::size_t(state.query_tokens) * width) {
            throw std::logic_error("KVMem Query token coverage mismatch");
        }
        layers[layer] = state.query_rows[layer].data();
    }
    const auto& policy = *state.selector;
    const auto& options = policy.config();
    const auto frame = kvmem::query_score_frame(state.statistics_frontier, 64,
        policy.budget_blocks(), options.sink_blocks, options.recent_blocks, mandatory);
    if (!memory_query_scorer) {
        throw std::logic_error("KVMem Query scorer lost its startup workspace");
    }
    const kvmem::PageQueryView query{nullptr, counts.data(), config.full_attention_layers,
        state.query_tokens, config.attention->num_attention_heads,
        config.attention->num_key_value_heads, config.attention->head_dim, layers.data()};
    const auto aggregation = kvmem_options.score_agg_max
        ? kvmem::GqaScoreAggregation::Max : kvmem::GqaScoreAggregation::Mean;
    auto scores = memory_query_scorer->score(*state.statistics, query, frame, aggregation);
    if (kvmem_options.graph_alpha > 0) {
        scores = kvmem::propagate_query_scores(scores, frame, state.graph,
                                               kvmem_options.graph_alpha);
    }
    state.selector->set_retrieval_scores(scores);
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr,
            "KVMEM_QUERY scorer=page-softmax agg=%s lrfu=%d graph_alpha=%.6g rows=%u blocks=%zu\n",
            kvmem_options.score_agg_max ? "max" : "mean", int(kvmem_options.lrfu),
            kvmem_options.graph_alpha, state.query_tokens, scores.size());
    }
    return scores;
}

void ProgramImpl::read_memory_candidates() {
    if (!memory_candidate_statistics) return;
    read_statistics(device, *memory_candidate_statistics, memory_candidate_host->data(), false);
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
    observe_memory_attention(sequence, begin, end, first_column, row_width, false);
    state.statistics_frontier = end;
}

void ProgramImpl::observe_memory_attention(SequenceState& sequence, std::uint32_t begin,
    std::uint32_t end, std::uint32_t first_column, std::uint32_t row_width, bool prefill) {
    auto& state = sequence.window;
    if (!memory_statistics || end <= begin || end <= state.graph.accepted_through) return;
    const auto& config = parameters.model.config().text;
    const auto accepted = end - begin;
    auto next_scope = state.attention_scope;
    std::vector<std::uint32_t> rows;
    const auto output_first = state.output_begin && state.output_begin > begin ?
        std::min(accepted, state.output_begin - begin) : 0u;
    const bool output = state.output_begin && output_first < accepted;
    if (output) {
        std::vector<std::int32_t> tokens(accepted);
        const auto& captured = memory_statistics->ranks.front().attention_tokens;
        DeviceBinding binding(device.rank(0).device);
        CUDA_CHECK(cudaMemcpyAsync(tokens.data(),
            static_cast<const std::int32_t*>(captured.data) + first_column,
            tokens.size() * sizeof(tokens[0]), cudaMemcpyDeviceToHost, device.stream_for_rank(0)));
        CUDA_CHECK(cudaStreamSynchronize(device.stream_for_rank(0)));
        auto plan = kvmem::accepted_output_samples(next_scope, tokens.data() + output_first,
            accepted - output_first, state.think_open, state.think_close);
        rows = std::move(plan.rows);
        next_scope = plan.state;
        for (auto& row : rows) row += output_first;
    } else if (prefill) {
        rows = kvmem::prefill_sample_rows(begin, accepted,
            state.feedback.snapshot().prefill_end, compact_position(sequence, begin),
            kvmem_window_tokens);
    }
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](auto row) {
        return begin + row < state.graph.accepted_through;
    }), rows.end());
    std::vector<kvmem::AcceptedAttentionRow> observations(rows.size());
    auto logical_pages = state.pages;
    for (auto id = begin / 64; id < (end + 63) / 64; ++id) {
        if (std::find(logical_pages.begin(), logical_pages.end(), id) == logical_pages.end()) {
            logical_pages.push_back(id);
        }
    }
    const std::int32_t selected[]{0, static_cast<std::int32_t>(config.full_attention_layers / 2),
        static_cast<std::int32_t>(config.full_attention_layers - 1)};
    const auto selected_count = std::min<std::uint32_t>(3, config.full_attention_layers);
    for (std::size_t i = 0; i < rows.size(); ++i) observations[i].source_position = begin + rows[i];
    for (std::size_t first = 0; first < rows.size(); first += 64) {
        const auto count = std::min<std::size_t>(64, rows.size() - first);
        std::vector<std::int32_t> columns(count), positions(count);
        for (std::size_t i = 0; i < count; ++i) {
            columns[i] = static_cast<std::int32_t>(first_column + rows[first + i]);
            positions[i] = static_cast<std::int32_t>(compact_position(sequence, begin + rows[first + i]));
        }
        for (std::int32_t slot = 0; slot < 3; ++slot) {
            if (slot && selected[slot] == selected[slot - 1]) continue;
            const auto layer = selected[slot];
            std::size_t rank = 0;
            for (; rank < memory_statistics->ranks.size(); ++rank) {
                const auto& shard = memory_statistics->ranks[rank];
                if (shard.attention_queries.data &&
                    layer >= static_cast<std::int32_t>(shard.first_layer) &&
                    layer < static_cast<std::int32_t>(shard.first_layer + shard.key_sums.ne[2])) {
                    break;
                }
            }
            if (rank == memory_statistics->ranks.size()) {
                throw std::logic_error("KVMem attention layer has no owning stage");
            }
            auto& shard = memory_statistics->ranks[rank];
            DeviceBinding binding(device.rank(rank).device);
            const auto stream = device.stream_for_rank(rank);
            auto row_tensor = shard.attention_rows.slice(0, 0, static_cast<std::int32_t>(count));
            auto position_tensor =
                shard.attention_positions.slice(0, 0, static_cast<std::int32_t>(count));
            CUDA_CHECK(cudaMemcpyAsync(row_tensor.data, columns.data(),
                count * sizeof(columns[0]), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(position_tensor.data, positions.data(),
                count * sizeof(positions[0]), cudaMemcpyHostToDevice, stream));
            const auto capacity = shard.attention_queries.ne[1];
            auto query = shard.attention_queries.slice(2, slot, 1).view(
                {static_cast<std::int32_t>(config.attention->head_dim),
                 static_cast<std::int32_t>(config.attention->num_attention_heads), capacity});
            auto scratch = shard.attention_scratch.slice(2, 0, static_cast<std::int32_t>(count));
            auto norms = shard.attention_norms.slice(2, 0, static_cast<std::int32_t>(count));
            auto mass = shard.attention_output.slice(1, 0, static_cast<std::int32_t>(count));
            auto cache = decoder->text_kv.batch_layer_view(static_cast<std::size_t>(layer));
            const auto table_row = text_kv_addresses->bound_row(sequence.kv->text);
            ops::attention_page_mass(query, row_tensor, position_tensor, table_row, cache,
                                     scratch, norms, mass, stream);
            std::vector<float> host(std::size_t(mass.ne[0]) * count);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), mass.data, host.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (std::size_t i = 0; i < count; ++i) {
                for (std::size_t p = 0; p < static_cast<std::size_t>(mass.ne[0]); ++p) {
                    const auto value = host[i * mass.ne[0] + p] / selected_count;
                    if (!std::isfinite(value) || value < 0) {
                        throw std::runtime_error("KVMem sampled invalid attention mass");
                    }
                    if (p >= logical_pages.size()) {
                        if (value > 1e-7) {
                            throw std::runtime_error("KVMem sampled an unmapped compact page");
                        }
                        continue;
                    }
                    if (value > 0) {
                        observations[first + i].mass.push_back(
                            {std::uint64_t(logical_pages[p]) * 64, value});
                    }
                }
            }
        }
    }
    if (observations.empty()) {
        state.graph = kvmem::prepare_graph_accept(state.graph, state.graph.accepted_through,
                                                  end, {}, 32u << 20);
        state.attention_scope = next_scope;
        return;
    }
    auto next_graph = kvmem::prepare_graph_accept(state.graph, state.graph.accepted_through,
                                                 end, observations, 32u << 20);
    size_t extra = state.feedback.snapshot().bytes();
    for (const auto& row : observations) {
        extra += row.mass.size() * sizeof(std::pair<std::uint64_t, double>) * 3;
    }
    next_graph.reserve_extra(extra);
    auto next_feedback = state.feedback;
    if (output) {
        for (const auto& row : observations) next_feedback.observe_output(row.mass);
    } else {
        std::vector<std::uint64_t> exposed;
        for (auto page : logical_pages) exposed.push_back(std::uint64_t(page) * 64);
        std::sort(exposed.begin(), exposed.end());
        exposed.erase(std::unique(exposed.begin(), exposed.end()), exposed.end());
        next_feedback.observe_prefill(begin, end, accepted, observations, exposed);
    }
    next_graph.include_feedback(next_feedback.snapshot().bytes());
    state.graph = std::move(next_graph);
    state.feedback = std::move(next_feedback);
    state.attention_scope = next_scope;
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr,
            "KVMEM_ATTENTION phase=%s begin=%u end=%u samples=%zu graph_revision=%llu\n",
            output ? "output" : "prefill", begin, end, rows.size(),
            static_cast<unsigned long long>(state.graph.revision));
    }
    (void)row_width;
}
} // namespace ninfer::models::qwen3_5::detail
