#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/structured_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    card.set_memory_statistics(execution.memory_statistics);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    // The structured lanes' masks, computed while the verification ran.
    if (execution.io.structured != nullptr) { execution.io.structured->wait(execution.device.stream); }
    // Before acceptance adds this round's tokens to the penalty counts.
    gather_logprobs(execution, frame.target_logits, frame.sampling, &frame.drafts);
    // The sparse accept op's workspace and sampling domain are fixed at the trained 248077 rows;
    // a coded pack may carry the tokenizer's padded vocabulary (248320), so the accepted domain is
    // the smaller of the two. physical_rows still comes from the logits plane.
    const std::int32_t token_domain = std::min<std::int32_t>(
        dimension(execution.parameters.model.resources().public_token_count), 248077);
    if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts, token_domain,
            frame.sampling, {false}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_coupled_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, token_domain, frame.sampling,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::models::qwen3_5::execution
