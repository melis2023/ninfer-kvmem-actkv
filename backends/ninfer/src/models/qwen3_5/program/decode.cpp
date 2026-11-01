#include "models/qwen3_5/program/structured_round.h"
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "core/nvtx.h"
#include "core/device.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/sampling.h"
#include "models/qwen3_5/program/tool_open_mask.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_5::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        card.set_memory_statistics(state.execution.memory_statistics);
        card.set_stage_runtime(state.execution.stages);
        card.set_expert_runtime(state.execution.experts);
        card.set_rope_yarn(state.execution.rope_yarn);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor sample_positions   = ordinary.sample_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        // The structured lanes' masks, computed while the forward pass ran.
        if (state.execution.io.structured != nullptr) {
            state.execution.io.structured->wait(state.execution.device.stream);
        }
        // Before sampling adds the round's tokens to the penalty counts.
        gather_logprobs(state.execution, logits.view({logits.ne[0], 1, batch_size}),
                        ordinary.sampling, nullptr);
        ops::sample(logits, sampled,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ordinary.sampling, sample_positions, ops::kSamplePurposeDecode,
                    state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_5::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

namespace {

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label,
                                         std::uint32_t verify_window = 0);

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label);

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label);

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label,
                                         std::uint32_t verify_window) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.verify_window == verify_window &&
                   profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

} // namespace

void ProgramImpl::update_sampling(SequenceHandle handle,
                                  const ResolvedSamplingParameters& sampling) {
    if (!valid_sequence(handle)) { throw std::logic_error("sampling update names no sequence"); }
    const std::uint32_t lane  = ContractAccess::lane(handle).value;
    RequestControl& request   = requests[lane];
    ops::SamplingConfig& host = request.sampling_host;
    host.temperature          = sampling.temperature;
    host.top_k                = sampling.top_k;
    host.top_p                = sampling.top_p;
    host.min_p                = sampling.min_p;
    host.presence_penalty     = sampling.presence_penalty;
    host.frequency_penalty    = sampling.frequency_penalty;
    // Penalties that start now count only the tokens sampled from here on.
    if ((host.presence_penalty != 0.0F || host.frequency_penalty != 0.0F) &&
        host.token_counts == nullptr) {
        Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                            .view({dimension(parameters.model.resources().public_token_count)});
        CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
        host.token_counts = static_cast<std::int32_t*>(counts.data);
    }
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &host, sizeof(host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImpl::ensure_suppressed_tool_open_mask(TokenId token) {
    if (suppressed_tool_open_id == token && suppressed_tool_open_mask.p != nullptr) { return; }
    const std::uint32_t domain = parameters.model.resources().public_token_count;
    if (token < 0 || static_cast<std::uint32_t>(token) >= domain) {
        throw std::logic_error("suppressed tool-open token is outside the public vocabulary");
    }
    const std::size_t words = (static_cast<std::size_t>(domain) + 31U) / 32U;
    std::vector<std::uint32_t> host(words);
    fill_forbidden_token_mask(host, static_cast<std::uint32_t>(token), domain);
    suppressed_tool_open_mask = DeviceBuffer(words * sizeof(std::uint32_t));
    suppressed_tool_open_mask.copy_from_host(host.data(), host.size() * sizeof(std::uint32_t));
    suppressed_tool_open_id = token;
}

void ProgramImpl::install_sampling(SequenceState& sequence, RequestControl& request,
                                   const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
    request.sampling_host     = config;
    if (request.grammar) {
        request.sampling_host.token_mask        = structured_round->device_mask(sequence.lane);
        request.sampling_host.token_mask_stride = structured_round->stride();
        structured_round->fill(sequence.lane, *request.grammar, device.stream);
    } else if (request.suppressed_tool_open && *request.suppressed_tool_open >= 0 &&
               static_cast<std::uint32_t>(*request.suppressed_tool_open) <
                   parameters.model.resources().public_token_count) {
        // One static column. Stride 0 makes every MTP draft column read it, so the ban does not
        // allocate a grammar mask or change this request's draft eligibility.
        ensure_suppressed_tool_open_mask(*request.suppressed_tool_open);
        request.sampling_host.token_mask =
            static_cast<const std::uint32_t*>(suppressed_tool_open_mask.p);
        request.sampling_host.token_mask_stride = 0;
    }
    request.next_mask.reset();
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(widest_verify_window(), 0),
        .adaptive              = mtp_controller.has_value(),
        .rounds_per_window =
            std::vector<std::uint64_t>(mtp_controller.has_value() ? draft_window : 0U, 0),
    };
    request.mtp_signal.reset();
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImpl::arm_logprobs(bool any) {
    if (logprobs_armed == any) { return; }
    set_device_i32(io.logprob_active, any ? 1 : 0);
    logprobs_armed = any;
}

void ProgramImpl::fetch_logprobs(std::int32_t rows) {
    static_assert(kMaximumTokenLogprobs == static_cast<std::size_t>(ops::kLogprobTopK));
    const auto capacity = static_cast<std::size_t>(io.logprob_ids.ne[1]);
    if (rows <= 0 || static_cast<std::size_t>(rows) > capacity) {
        throw std::logic_error("logprob gather rows exceed the round buffer");
    }
    if (!logprobs_host) {
        logprobs_host.emplace(capacity * kMaximumTokenLogprobs * (sizeof(TokenId) + sizeof(float)));
    }
    auto* ids                 = static_cast<unsigned char*>(logprobs_host->data());
    const std::size_t entries = static_cast<std::size_t>(rows) * kMaximumTokenLogprobs;
    CUDA_CHECK(cudaMemcpyAsync(ids, io.logprob_ids.data, entries * sizeof(TokenId),
                               cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(ids + capacity * kMaximumTokenLogprobs * sizeof(TokenId),
                               io.logprob_values.data, entries * sizeof(float),
                               cudaMemcpyDeviceToHost, device.stream));
}

runtime::RawTokenLogprob ProgramImpl::fetched_logprob(std::int32_t row, TokenId token) const {
    const auto capacity = static_cast<std::size_t>(io.logprob_ids.ne[1]);
    const auto* ids     = static_cast<const TokenId*>(logprobs_host->data());
    const auto* values  = reinterpret_cast<const float*>(ids + capacity * kMaximumTokenLogprobs);
    runtime::RawTokenLogprob record{.id = token, .logprob = kLogprobSentinel};
    for (std::size_t k = 0; k < kMaximumTokenLogprobs; ++k) {
        const std::size_t slot = static_cast<std::size_t>(row) * kMaximumTokenLogprobs + k;
        record.top_ids[k]      = ids[slot];
        record.top_values[k]   = values[slot];
        // A drawn token outside the top set keeps the sentinel.
        if (ids[slot] == token) { record.logprob = values[slot]; }
    }
    return record;
}

std::int32_t ProgramImpl::arm_readout(std::span<const TokenId> ids) {
    if (ids.empty()) { return 0; }
    if (ids.size() > kMaximumReadoutTokens ||
        static_cast<std::int64_t>(ids.size()) > io.readout_ids.ne[0]) {
        throw std::logic_error("readout token ids exceed the round buffer");
    }
    // A pageable source: the copy completes on the host before it returns.
    CUDA_CHECK(cudaMemcpyAsync(io.readout_ids.data, ids.data(), ids.size() * sizeof(TokenId),
                               cudaMemcpyHostToDevice, device.stream));
    return static_cast<std::int32_t>(ids.size());
}

void ProgramImpl::fetch_readout(std::int32_t count) {
    if (count <= 0) { return; }
    if (!readout_host) {
        readout_host.emplace(kMaximumConcurrency * kMaximumReadoutTokens * sizeof(float));
    }
    CUDA_CHECK(cudaMemcpyAsync(readout_host->data(), io.readout_values.data,
                               static_cast<std::size_t>(count) * sizeof(float),
                               cudaMemcpyDeviceToHost, device.stream));
}

void ProgramImpl::attach_readout(runtime::RawTokenLogprob& record, std::int32_t count,
                                 std::int32_t row) const {
    record.readout_count = 0;
    if (count <= 0) { return; }
    const auto* values = static_cast<const float*>(readout_host->data()) +
                         static_cast<std::size_t>(row) * kMaximumReadoutTokens;
    std::copy_n(values, count, record.readout_values.begin());
    record.readout_count = static_cast<std::uint32_t>(count);
}

void ProgramImpl::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 ||
        source.ne[0] != dimension(parameters.model.config().text.hidden_size) ||
        source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImpl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImpl::make_early_token_resources() {
    if (first_token_ready) { return; }
    first_token_ready.emplace(device);
    initial_drafts_ready.reserve(max_concurrency);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        initial_drafts_ready.emplace_back(device);
    }
    initial_drafts_host.emplace(static_cast<std::size_t>(max_concurrency) *
                                qwen3_5::kMtpDecodeMaximumDrafts * sizeof(TokenId));
}

void ProgramImpl::settle_initial_drafts(SequenceState& sequence) {
    if (!sequence.mtp_drafts_pending) { return; }
    initial_drafts_ready.at(sequence.lane).synchronize();
    const auto* row = static_cast<const TokenId*>(initial_drafts_host->data()) +
                      static_cast<std::size_t>(sequence.lane) * qwen3_5::kMtpDecodeMaximumDrafts;
    std::copy_n(row, sequence.mtp_draft_count, sequence.mtp_drafts.begin());
    sequence.mtp_drafts_pending = false;
}

void ProgramImpl::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

std::vector<NgramProposer::Match>
ProgramImpl::propose_ngram(std::span<const std::uint32_t> lanes,
                           std::span<const runtime::RoundBudget> budgets) {
    std::vector<NgramProposer::Match> matches(lanes.size());
    if (ngram_draft_window == 0) { return matches; }
    if (budgets.size() != lanes.size()) {
        throw std::invalid_argument("ngram budget count does not match lane count");
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        if (lanes[row] >= max_concurrency) {
            throw std::invalid_argument("ngram lane is out of range");
        }
        matches[row] = propose_ngram_one(lanes[row], budgets[row]);
    }
    return matches;
}

void ProgramImpl::take_ngram_index(RequestControl& request, PreparedPromptData& prompt) {
    // A prompt prepared for a request that does not draft carries none: that request reads one
    // token and proposes nothing, and request planning refuses a drafting request without one.
    request.ngram          = std::move(prompt.ngram_index);
    request.ngram_indexed  = prompt.token_ids.size();
    request.ngram_snapshot = std::move(prompt.ngram_snapshot);
}

NgramProposer::Match ProgramImpl::propose_ngram_one(std::uint32_t lane,
                                                    const runtime::RoundBudget& budget) {
    auto& request        = requests[lane];
    const auto& sequence = active_sequence(lane);
    const auto& ledger   = sequence.ledger;
    const auto remaining = budget.generated_tokens_remaining;
    const auto room =
        sequence.execution_frontier < capacity ? capacity - sequence.execution_frontier - 1U : 0U;
    const auto maximum = std::min({ngram_draft_window, remaining > 1 ? remaining - 1U : 0U, room});
    // The neural path still validates decode readiness. No draft fits an anchor-only tail.
    if (maximum == 0) { return {}; }
    if (!request.ngram || request.ngram_indexed > ledger.size()) {
        throw std::logic_error("ngram committed history is not initialized");
    }
    while (request.ngram_indexed < ledger.size()) {
        request.ngram->append(ledger[request.ngram_indexed++]);
    }
    const auto history =
        std::span<const TokenId>(ledger).last(std::min(kNgramMatchHistoryTokens, ledger.size()));
    NgramProposer::Match match;
    if (request.ngram_snapshot && request.ngram_copy_source != 0 &&
        ledger.size() >= request.ngram_copy_ledger) {
        const auto offset = request.ngram_copy_offset + ledger.size() - request.ngram_copy_ledger;
        if (offset <= std::numeric_limits<std::uint32_t>::max()) {
            match = request.ngram_snapshot->continue_copy(request.ngram_copy_source,
                                                          static_cast<std::uint32_t>(offset),
                                                          history, maximum + 2U, ngram_min_match);
        }
    }
    if (match.tokens.size() < maximum + 2U) {
        match = request.ngram->propose(history, maximum + 2U, ngram_min_match);
        // Equal candidates keep the live source; a maximal match cannot be improved.
        if (request.ngram_snapshot &&
            !NgramProposer::maximal(match, history.size(), maximum + 2U)) {
            auto retained = request.ngram_snapshot->propose(history, maximum + 2U, ngram_min_match);
            if (retained.matched > match.matched ||
                (retained.matched == match.matched &&
                 retained.tokens.size() > match.tokens.size())) {
                match = std::move(retained);
            }
        }
    }
    match = NgramProposer::finish_round(std::move(match), maximum, draft_window);
    // A cursor is only a candidate. Next round rechecks it against committed ledger
    // tokens, including correction/bonus/control tokens, before skipping lookup.
    request.ngram_copy_source = !match.tokens.empty() && match.archived ? match.source : 0;
    request.ngram_copy_offset = match.offset;
    request.ngram_copy_ledger = ledger.size();
    return match;
}

void ProgramImpl::record_ngram_round(RequestControl& request, const NgramProposer::Match& match,
                                     std::uint32_t drafted, std::uint32_t accepted) noexcept {
    request.speculative_stats.ngram_rounds += 1;
    request.speculative_stats.ngram_drafted_tokens += drafted;
    request.speculative_stats.ngram_accepted_tokens += accepted;
    if (match.archived) {
        request.speculative_stats.ngram_archive_rounds += 1;
        request.speculative_stats.ngram_archive_drafted_tokens += drafted;
        request.speculative_stats.ngram_archive_accepted_tokens += accepted;
    }
}

void ProgramImpl::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                std::span<const std::uint32_t> starts,
                                                std::span<const std::uint32_t> counts) {
    if (!is_masked_draft_backend(speculative_backend) || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    // A copy round leaves up to the widest window of target features to append.
    const std::uint32_t widest_count = widest_verify_window() + 1U;
    std::uint32_t minimum_count      = widest_count;
    std::uint32_t maximum_count      = 0;
    *dflash_host_ingress             = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > widest_count ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        // Context append writes only the draft caches. Target execution owns Main KV
        // coverage, including any uncommitted suffix that survives until the round is settled.
        // DFlash2 has only fixed cyclic state; DFlash also grows its Full backend KV here.
        if (sequence.kv->backend) {
            backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, end,
                                                          compute_streams);
        }
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows               = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {dimension(parameters.draft->feature_projection.weight.k),
                                 static_cast<std::int32_t>(widest_count), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    execution::DFlashAppendContext state{
        {device, parameters, work, state_images->linear(0),
         replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
         proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel, mtp_attention_window, experts.get()},
        *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    execution::dflash_append_context(state, features, positions, device_counts,
                                     state_destination_tensor, table_rows,
                                     {minimum_count, maximum_count});
}

StructuredReadiness ProgramImpl::structured_readiness(SequenceHandle handle) {
    if (!structured_round || !valid_sequence(handle)) { return {}; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    RequestControl& request  = requests[lane];
    if (!request.grammar || speculative_backend != SpeculativeBackend::None) {
        request.next_mask.reset();
        return {};
    }
    // The grammar moves on only when the Engine commits tokens, which advances the ledger: a job
    // started at another frontier is for another position.
    const std::uint32_t frontier = active_sequence(lane).ledger_frontier;
    if (request.next_mask && request.next_mask_frontier != frontier) { request.next_mask.reset(); }
    if (!request.next_mask) {
        request.next_mask = runtime::GrammarMaskJob::start(
            *request.grammar, static_cast<std::size_t>(structured_round->stride()));
        request.next_mask_frontier = frontier;
        request.next_mask_started  = Clock::now();
    }
    if (!request.next_mask->ready()) {
        const double waited =
            std::chrono::duration<double>(Clock::now() - request.next_mask_started).count();
        if (waited > runtime::kGrammarMaskBudgetSeconds) {
            return {.ready   = false,
                    .failure = "the grammar mask took over " +
                               std::to_string(runtime::kGrammarMaskBudgetSeconds) + " s"};
        }
        return {.ready = false};
    }
    std::string problem = runtime::grammar_mask_problem(*request.next_mask);
    if (!problem.empty()) { return {.ready = false, .failure = std::move(problem)}; }
    return {};
}

std::shared_ptr<runtime::GrammarMaskJob>
ProgramImpl::take_next_mask(RequestControl& request, const SequenceState& sequence,
                            std::uint32_t extent) {
    if (extent != 0 || !request.next_mask || !request.next_mask->ready() ||
        request.next_mask_frontier != sequence.ledger_frontier) {
        request.next_mask.reset();
        return nullptr;
    }
    return std::exchange(request.next_mask, nullptr);
}

void ProgramImpl::check_masks() {
    if (structured_round) { structured_round->check(); }
}

void ProgramImpl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= dimension(parameters.model.resources().public_token_count)) {
            throw std::runtime_error("target returned a token outside the public token domain");
        }
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                   std::span<const runtime::RoundBudget> budgets,
                                   runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        if (kvmem_window_tokens) {
            prepare_window(sequence, sequence.execution_frontier, sequence.execution_frontier + 1,
                false,false,kvmem::SelectionReason::DecodePressure);
        }
        maximum_frontier = std::max(maximum_frontier, compact_position(sequence, sequence.execution_frontier));
    }

    const auto start = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        bool logprobs = false;
        if (structured_round) { structured_round->begin(); }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = active_sequence(lanes[row]);
            RequestControl& request            = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(compact_position(sequence, frontier), "ordinary batch position");
            ordinary_host_ingress->sample_positions[row] =
                checked_i32(frontier + 1U, "ordinary batch sample position");
            if (kvmem_options.verify_transfers && frontier % 64 == 0) {
                std::fprintf(stderr, "KVMEM_ACTIVE_KV route=ordinary logical=%u history=%d capacity=%u\n",
                    frontier, ordinary_host_ingress->cache_positions[row], kvmem_window_tokens);
            }
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            const StateImageSelectors selectors                 = state_selectors(sequence);
            ordinary_host_ingress->state_source_slots[row]      = selectors.source;
            ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
            ordinary_host_ingress->sampling[row]                = request.sampling_host;
            if (request.grammar) {
                if (auto ready = take_next_mask(request, sequence, 0)) {
                    structured_round->adopt(sequence.lane, std::move(ready));
                } else {
                    structured_round->start(sequence.lane, *request.grammar, {});
                }
            }
            logprobs = logprobs || request.logprobs;
            ensure_sequence_kv_mapped(sequence, frontier + 1, 0);
        }
        if (structured_round) { structured_round->publish(); }
        arm_logprobs(logprobs);

        execution::OrdinaryBatchContext schedule_state{
            {device, parameters, work, state_images->linear(0),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
             mtp_attention_window, experts.get()},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            state_images->continuation_hidden_store()};

        schedule_state.execution.memory_statistics = memory_candidate_statistics ? &*memory_candidate_statistics : nullptr;
        mark_workspace_usage(workspace_plan.ordinary_round);
        execution::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                         envelope, executable);
        if (logprobs) { fetch_logprobs(static_cast<std::int32_t>(lanes.size())); }
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        check_masks();
        timing.end_wait();

        read_memory_candidates();
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = active_sequence(lanes[row]);
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            // A round gathers every row when any of its requests asks; a row keeps a record only
            // for a request that asked.
            if (request.logprobs) {
                round_logprobs[row] = fetched_logprob(static_cast<std::int32_t>(row), token);
            }
            commit_memory_candidates(sequence, base_E, base_E + 1, static_cast<std::uint32_t>(row), 1);
            sequence.text_kv_valid = base_E + 1;
            commit_sequence_kv(sequence, sequence.text_kv_valid, 0);
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .logprobs = logprobs ? std::span<const runtime::RawTokenLogprob>(round_logprobs.data(),
                                                                             lanes.size())
                                 : std::span<const runtime::RawTokenLogprob>{},
            .timing = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    std::array<const MtpAdaptiveSignal*, kMaximumConcurrency> signals{};
    std::array<std::uint32_t, kMaximumConcurrency> available{};
    std::array<std::uint32_t, kMaximumConcurrency> room{};
    std::uint64_t cohort_key = 1469598103934665603ULL;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        settle_initial_drafts(active_sequence(lane));
        SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        if (kvmem_window_tokens) {
            prepare_window(sequence, sequence.execution_frontier,
                std::min(capacity, sequence.execution_frontier +
                    std::max(draft_window, ngram_draft_window) + draft_window),false,false,kvmem::SelectionReason::DecodePressure);
        }
        maximum_frontier = std::max(maximum_frontier, compact_position(sequence, sequence.execution_frontier));
        signals[row]     = &request.mtp_signal;
        room[row]        = std::min(budgets[row].generated_tokens_remaining,
                                    capacity - sequence.execution_frontier);
        available[row]   = std::min(sequence.mtp_draft_count, room[row] > 0 ? room[row] - 1U : 0U);
        cohort_key ^= static_cast<std::uint64_t>(lane) << 32U | lane_epochs[lane];
        cohort_key *= 1099511628211ULL;
    }
    // A copy proposal for any row makes this a copy round, verified at the ngram window; rows
    // without one verify their neural drafts inside it. Otherwise a fixed policy verifies the
    // whole draft window, and adaptive MTP the width its controller selects, never below the
    // narrowest width it captured.
    const std::vector<NgramProposer::Match> matches = propose_ngram(lanes, budgets);
    const bool copy_round =
        std::any_of(matches.begin(), matches.end(),
                    [](const NgramProposer::Match& match) { return !match.tokens.empty(); });
    std::uint32_t verify_window = copy_round ? ngram_draft_window : draft_window;
    if (mtp_controller && !copy_round) {
        verify_window =
            std::clamp(mtp_controller->select(
                           std::span<const MtpAdaptiveSignal* const>(signals.data(), lanes.size()),
                           std::span<const std::uint32_t>(available.data(), lanes.size()),
                           std::span<const std::uint32_t>(room.data(), lanes.size()), cohort_key),
                       mtp_minimum_adaptive_window(draft_window), draft_window);
    }
    const std::uint32_t width = verify_window + 1;
    const auto attention_capacity = kvmem_window_tokens ? kvmem_window_tokens : capacity;

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        execution::MtpCausalAttentionEnvelopes envelopes =
            mtp_causal_attention_envelopes(maximum_frontier, verify_window, draft_window, attention_capacity);
        if (use_cuda_graph) {
            DecodeGraphFamily& family = copy_round ? ngram_graphs : mtp_graphs;
            DecodeGraphProfile& profile =
                select_graph_profile(family, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch", verify_window);
            executable = &install_graph_profile(family, profile, "MTP batch");
            envelopes  = mtp_causal_attention_envelopes(profile.max_execution_frontier,
                                                        verify_window, draft_window, attention_capacity);
        }

        bool logprobs = false;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            RequestControl& request           = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const NgramProposer::Match& match = matches[row];
            const bool row_copy               = !match.tokens.empty();
            const std::uint32_t extent        = std::min(
                {row_copy ? static_cast<std::uint32_t>(match.tokens.size())
                          : sequence.mtp_draft_count,
                 verify_window, max_by_budget, capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(compact_position(sequence, frontier), "MTP batch cache frontier");
            mtp_host_ingress->logical_frontiers[row] = checked_i32(frontier, "MTP batch logical frontier");
            if (kvmem_options.verify_transfers) {
                std::fprintf(stderr, "KVMEM_ACTIVE_KV route=mtp logical=%u history=%d verify_columns=%u capacity=%u\n",
                    frontier, mtp_host_ingress->base_frontiers[row], extent + 1, kvmem_window_tokens);
            }
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < verify_window; ++j) {
                mtp_host_ingress->current_drafts[row * verify_window + j] =
                    j < extent ? (row_copy ? match.tokens[j] : sequence.mtp_drafts[j])
                               : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            mtp_host_ingress->mtp_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            const StateImageSelectors selectors            = state_selectors(sequence);
            mtp_host_ingress->state_source_slots[row]      = selectors.source;
            mtp_host_ingress->state_destination_slots[row] = selectors.destination;
            mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta + checked_i32(sequence.window.removed_tokens, "MTP cache shift");
            mtp_host_ingress->sampling[row]                = request.sampling_host;
            logprobs = logprobs || request.logprobs;
            if (request.grammar) {
                if (auto ready = take_next_mask(request, sequence, extent)) {
                    structured_round->adopt(sequence.lane, std::move(ready));
                } else {
                    structured_round->start(
                        sequence.lane, *request.grammar,
                        {mtp_host_ingress->current_drafts.data() + row * verify_window, extent});
                }
            }
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1,
                                      std::min(capacity, frontier + extent + draft_window));
        }
        if (structured_round) { structured_round->publish(); }

        arm_logprobs(logprobs);
        qwen3_5::MtpDecodeState frame = io.mtp_decode->verification_view(verify_window);
        execution::MtpBatchContext schedule_state{
            {device, parameters, work, state_images->linear(0),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
             mtp_attention_window, experts.get()},
            decoder->text_kv,
            *decoder->mtp_cache(),
            frame,
            *mtp_host_ingress,
            *mtp_host_egress,
            state_images->continuation_hidden_store()};

        schedule_state.execution.memory_statistics = memory_candidate_statistics ? &*memory_candidate_statistics : nullptr;
        mark_workspace_usage(workspace_plan.mtp_round);
        execution::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                    verify_window, draft_window, envelopes, executable);
        if (logprobs) { fetch_logprobs(static_cast<std::int32_t>(lanes.size() * width)); }
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        check_masks();
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            // The gather row of verification column c in batch row r is c + r * width.
            for (std::size_t c = 0; request.logprobs && c < row_tokens.size(); ++c) {
                round_logprobs[row * width + c] =
                    fetched_logprob(static_cast<std::int32_t>(row * width + c), row_tokens[c]);
            }
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            // The adaptive controller measures the neural drafter only.
            if (mtp_controller && !copy_round) {
                request.speculative_stats.rounds_per_window[verify_window - 1U] += 1;
                if (mtp_controller->transitioned()) {
                    request.speculative_stats.window_transitions += 1;
                }
            }
            if (!matches[row].tokens.empty()) {
                record_ngram_round(request, matches[row], pcur,
                                   static_cast<std::uint32_t>(accepted_i));
            } else if (mtp_controller) {
                request.mtp_signal.observe(pcur, static_cast<std::uint32_t>(accepted_i));
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
                .record_width  = width,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        if (mtp_controller && !copy_round) {
            mtp_controller->observe_execution(static_cast<std::uint32_t>(lanes.size()),
                                              verify_window, seconds);
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .logprobs   = logprobs ? std::span<const runtime::RawTokenLogprob>(
                                         round_logprobs.data(), lanes.size() * width)
                                   : std::span<const runtime::RawTokenLogprob>{},
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeDFlashRound, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (!is_masked_draft_backend(speculative_backend) || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    // A copy proposal for any row makes this a copy round at the ngram window, on the frame
    // viewed at that width. At batch>1 the drafter also runs and the copy payload overlays it on
    // the device, so a row without a copy keeps its neural proposal; at batch one the copy alone
    // is verified.
    const std::vector<NgramProposer::Match> matches = propose_ngram(lanes, budgets);
    const bool copy_round =
        std::any_of(matches.begin(), matches.end(),
                    [](const NgramProposer::Match& match) { return !match.tokens.empty(); });
    const std::uint32_t verify_drafts = copy_round ? ngram_draft_window : draft_window;
    const bool drafter_runs           = !copy_round || lanes.size() > 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        dflash_host_ingress->copy_rows[row] = !matches[row].tokens.empty() ? 1 : 0;
    }
    for (std::size_t row = 0; copy_round && row < lanes.size(); ++row) {
        const auto row_extent = static_cast<std::uint32_t>(matches[row].tokens.size());
        for (std::uint32_t step = 0; step < verify_drafts; ++step) {
            const TokenId token = step < row_extent ? matches[row].tokens[step] : 0;
            dflash_host_ingress->ngram_tokens[row * verify_drafts + step] = token;
            for (std::uint32_t slot = 0; slot < ops::kSparseSpeculativeCandidates; ++slot) {
                const auto candidate = row * verify_drafts * ops::kSparseSpeculativeCandidates +
                                       step * ops::kSparseSpeculativeCandidates + slot;
                dflash_host_ingress->ngram_candidates[candidate] =
                    (token + static_cast<TokenId>(slot)) %
                    parameters.model.resources().public_token_count;
                dflash_host_ingress->ngram_q[candidate] = slot == 0 ? 1.0F : 0.0F;
            }
        }
    }
    const std::uint32_t width = verify_drafts + 1U;
    // The previous round may have been a wider copy round, whose context append is still owed.
    const std::uint32_t widest_width    = widest_verify_window() + 1U;
    std::uint32_t maximum_frontier      = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > widest_width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        // Reserve the complete verification width before binding compact target KV positions.
        if (kvmem_window_tokens) {
            prepare_window(sequence, sequence.execution_frontier,
                           std::min(capacity, sequence.execution_frontier + width),false,false,kvmem::SelectionReason::DecodePressure);
        }
        // Graph selection follows compact target KV; the draft ring keeps logical positions.
        maximum_frontier =
            std::max(maximum_frontier, compact_position(sequence, sequence.execution_frontier));
    }
    // Verification attention partitions each row by its full window, so the envelope covers the
    // window rather than the drafts a row actually carries: a round the output budget capped keeps
    // the partition of the uncapped one.
    const auto target_envelope_for = [&](std::uint32_t max_execution_frontier) {
        return ops::CausalAttentionExecutionEnvelope{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(max_execution_frontier) + width))};
    };

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeDFlashSubmit, nvtx::Category::DFlash,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable    = nullptr;
        execution::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, capacity);
        auto target_envelope                 = target_envelope_for(maximum_frontier);
        if (use_cuda_graph) {
            DecodeGraphFamily& family   = copy_round ? ngram_graphs : dflash_graphs;
            DecodeGraphProfile& profile = select_graph_profile(
                family, static_cast<std::uint32_t>(lanes.size()), maximum_frontier, "DFlash batch",
                copy_round ? verify_drafts : 0U);
            executable = &install_graph_profile(family, profile, "DFlash batch");
            envelopes =
                dflash_envelopes(profile.min_execution_frontier, profile.max_execution_frontier, capacity);
            target_envelope = target_envelope_for(profile.max_execution_frontier);
        }

        bool logprobs = false;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t proposed =
                !matches[row].tokens.empty()
                    ? static_cast<std::uint32_t>(matches[row].tokens.size())
                    : (drafter_runs ? draft_window : 0U);
            const std::uint32_t extent =
                std::min({proposed, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            // Compact coordinates: target_rope_positions below stays logical for RoPE.
            const bool logical_draft = speculative_backend == SpeculativeBackend::DFlash2;
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(compact_position(sequence, frontier), "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] = checked_i32(
                logical_draft ? sequence.dflash_context_frontier
                              : compact_position(sequence, sequence.dflash_context_frontier),
                "DFlash context frontier");
            dflash_host_ingress->draft_frontiers[row] = checked_i32(
                logical_draft ? frontier : compact_position(sequence, frontier),
                "DFlash draft frontier");
            dflash_host_ingress->draft_floors[row] = checked_i32(
                logical_draft ? sequence.dflash_context_floor : 0, "DFlash draft floor");
            dflash_host_ingress->proposal_valid_columns[row] =
                static_cast<std::int32_t>(draft_window + 1U);
            dflash_host_ingress->proposal_extents[row]       = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t position = frontier + std::min(column, extent);
                dflash_host_ingress->target_rope_positions[row * width + column] =
                    checked_i32(position, "DFlash target RoPE position") + sequence.rope_delta;
            }
            dflash_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
            dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors          = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[row] = selectors.source;
            dflash_host_ingress->state_destination_slots[row] = selectors.destination;
            dflash_host_ingress->sampling[row]                = request.sampling_host;
            logprobs = logprobs || request.logprobs;
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1U,
                                      backend_kv_cache() ? frontier : 0U);
        }
        arm_logprobs(logprobs);

        if (structured_round) {
            structured_round->begin();
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                RequestControl& request     = requests[lanes[row]];
                const std::uint32_t extent  = static_cast<std::uint32_t>(
                    dflash_host_ingress->proposal_extents[row]);
                auto ready = request.grammar
                                 ? take_next_mask(request, active_sequence(lanes[row]), extent)
                                 : nullptr;
                // A lane whose mask was computed ahead hands it over; the round starts no job
                // for it.
                structured_round->set_dflash_row(row, lanes[row], extent,
                                                 ready ? nullptr : request.grammar);
                if (ready) { structured_round->adopt(lanes[row], std::move(ready)); }
            }
        }
        execution::DFlashBatchContext schedule_state{
            {device, parameters, work, state_images->linear(0),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
             mtp_attention_window, experts.get()},
            decoder->text_kv,
            *dflash,
            *io.dflash_decode,
            *dflash_host_ingress,
            *dflash_host_egress,
            state_images->continuation_hidden_store()};
        schedule_state.ngram                  = copy_round;
        schedule_state.neural_proposal_drafts = draft_window;

        mark_workspace_usage(workspace_plan.dflash_round);
        execution::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                       verify_drafts, envelopes, target_envelope, executable);
        if (logprobs) { fetch_logprobs(static_cast<std::int32_t>(lanes.size() * width)); }
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        check_masks();
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            for (std::size_t c = 0; request.logprobs && c < row_tokens.size(); ++c) {
                round_logprobs[row * width + c] =
                    fetched_logprob(static_cast<std::int32_t>(row * width + c), row_tokens[c]);
            }
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            if (!matches[row].tokens.empty()) {
                record_ngram_round(request, matches[row], extent,
                                   static_cast<std::uint32_t>(accepted_i));
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
                .record_width  = width,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .logprobs   = logprobs ? std::span<const runtime::RawTokenLogprob>(
                                         round_logprobs.data(), lanes.size() * width)
                                   : std::span<const runtime::RawTokenLogprob>{},
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_raw(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets, failed_timing);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mtp_batch(lanes, budgets, failed_timing);
    }
    return decode_dflash_batch(lanes, budgets, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_non_speculative_pending(
    SequenceState& sequence, RequestControl& request, std::uint32_t accepted_tokens, bool terminal,
    std::optional<std::uint32_t> prefix_execution_split_after,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    const std::uint32_t base_ledger_frontier = request.pending.kind == PendingKind::Begin
                                                   ? request.pending.prompt_tokens
                                                   : request.pending.base_S;
    commit_generated_prefix_identity(
        sequence, base_ledger_frontier,
        std::span<const TokenId>(sequence.ledger).subspan(base_ledger_frontier, accepted_tokens),
        prefix_execution_split_after);

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        advance_rebuild_work(sequence, request.pending.base_E + request.pending.produced,
                             prefill_chunk);
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin publishes a sampled token but does not execute it through the target. An exact-hit
    // Fork therefore still names an immutable read source and an unwritten destination here; the
    // first state-mutating decode commit closes it. A suffix prefill already closed its Fork at
    // the committed prefill frontier.
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, compute_streams);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    capture_memory_reconstruction_if_ready(sequence);
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}


} // namespace ninfer::models::qwen3_5::detail
