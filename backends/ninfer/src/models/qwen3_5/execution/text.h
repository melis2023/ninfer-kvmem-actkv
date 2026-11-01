#pragma once
#include "models/qwen3_5/execution/memory_statistics.h"
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/execution/rotation.h"
#include "models/qwen3_5/execution/stage_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class NativeExpertRuntime;

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

// The most segments one batched prefill pass runs: a lane's last prompt chunk, or a branched
// read's prompt part and its branch blocks.
inline constexpr std::size_t kMaximumPrefillSegments = 64;

// What a batched prefill segment produces at its last column.
enum class SegmentOutput : std::uint8_t {
    // Samples its next token as a finalizing chunk does, with its logprob record and readout.
    Sample,
    // A branch block: the greedy token, its logprob record and readout, without the sampler.
    Read,
    // A branched read's prompt part: nothing.
    None,
};

// One segment of a batched prefill pass (TextContext::prefill_final_batch).
struct PrefillSegment {
    // The segment's tokens: a lane's prompt up to its frontier, or one branch's own tokens.
    std::span<const int> ids;
    // The absolute position of ids[0]; a branch continues its read's prompt frontier.
    std::uint32_t begin                 = 0;
    std::int32_t kv_table_row           = 0;
    std::int32_t state_source_slot      = 0;
    // Negative for a branch block: its GDN state goes to a transient buffer per layer and is not
    // kept, and its K/V rows past the prompt frontier are scratch the next block overwrites.
    std::int32_t state_destination_slot = 0;
    SegmentOutput output                = SegmentOutput::Sample;
    // The device sampling record; Read takes its logprob adjustments from it, None ignores it.
    const ops::SamplingConfig* sampling = nullptr;
    // The typed readout's token ids; empty reads nothing.
    std::span<const std::int32_t> readout;
};

// Host destinations of a batched prefill, one row per segment with output, in segment order: its
// token, kLogprobTopK logprob ids and values (written when logprobs are armed), and
// kReadoutLogprobsMaximum readout values, of which a row's first readout.size() are written.
struct PrefillBatchHost {
    std::int32_t* tokens  = nullptr;
    std::int32_t* top_ids = nullptr;
    float* top_values     = nullptr;
    float* readout        = nullptr;
};

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    void set_expert_runtime(NativeExpertRuntime* experts) noexcept { experts_ = experts; }

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_memory_statistics(const MemoryStatistics* value) noexcept { statistics_ = value; }
    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }
    void set_cache_position_shift(std::uint32_t shift) noexcept { cache_position_shift_ = shift; }

    // The typed readout a finalizing chunk performs on its sampled token's logits: the first
    // `count` ids of io.readout_ids, read into io.readout_values; zero reads nothing.
    void set_readout(std::int32_t count) noexcept { readout_count_ = count; }

    // Runs right after a finalizing chunk samples its token, before the drafter's work on the
    // chunk: it enqueues the token's copies to the host. The chunk then returns without waiting for
    // that work, which the stream orders before anything later.
    void set_first_token_hook(std::function<void()> hook) { first_token_hook_ = std::move(hook); }

    void set_rope_yarn(const ops::RopeYarn& yarn) noexcept { rope_yarn_ = yarn; }

    // Keys before its first query that the MTP draft attention reads past its one-page sink;
    // zero reads the whole history.
    void set_mtp_attention_window(std::uint32_t keys) noexcept { mtp_attention_window_ = keys; }

    // Present when the model is split over several devices; see StageRuntime.
    void set_stage_runtime(StageRuntime* runtime) noexcept { stage_runtime_ = runtime; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    // Route prefill prompt attention through the fast INT8-KV prompt kernel.
    void set_fast_prefill_kernel(bool enabled) noexcept { fast_prefill_kernel_ = enabled; }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    // One event per model layer: the next pass over the layer stack waits for each layer's event
    // before that layer's work (the layer's cached state is still being copied in).
    void set_layer_ready(std::span<const cudaEvent_t> events) noexcept { layer_ready_ = events; }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    // The last chunks of several requests' text prompts, and branched reads' blocks, in one pass
    // over the layer stack. Token-wise work runs over all columns at once; segment by segment, in
    // order, attention appends and reads the segment's KV row and each GDN layer advances its
    // state, as prefill_chunk would for that segment alone. A branch block follows its read's
    // prompt part: it shares the read's KV row past the prompt frontier and starts each GDN layer
    // from the state the prompt part left, so it sees the prompt and itself only. Segments with
    // output then read their last column: Sample samples as a finalizing chunk does (keyed by its
    // frontier), Read takes the greedy token; both gather the logprob record and the readout.
    // `host` receives every output row before this returns; prefill_hidden holds the final-norm
    // hidden of every column, segment after segment. No drafter work or pipeline stages.
    [[nodiscard]] PrefillChunkResult prefill_final_batch(std::span<const PrefillSegment> segments,
                                                         PrefillBatchHost host);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
private:
    void capture_attention_tokens(const Tensor&);
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;

    // Per-request width of a bound sequence batch; an unbatched call is one row of all columns.
    [[nodiscard]] std::int32_t row_width(std::int32_t columns) const noexcept {
        return active_sequence_batch_ != 0 ? active_sequence_width_ : columns;
    }
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    // The layers one stage owns, on the device that stage's context is currently bound to.
    template <class Tap>
    void run_stage_layers(std::size_t stage, Tensor& x, Phase phase, Tap& tap);
    // The whole layer stack across pipeline stages, ending with the residual back on rank 0. A tap
    // sees all of its layers on rank 0; a later stage's tapped outputs cross after the residual.
    template <class Tap>
    void run_staged(Tensor& x, Phase phase, Tap& tap);
    // The whole layer stack with every rank running its share of each layer; the residual ends on
    // rank 0 as on one device.
    template <class Tap>
    void run_tensor_parallel(Tensor& x, Phase phase, Tap& tap);
    // The GDN and FFN geometry of the rank a pass is issuing: a tensor-parallel share, or the model.
    [[nodiscard]] const TextConfig& layer_config() const noexcept {
        return stage_runtime_ != nullptr && stage_runtime_->tensor_parallel
                   ? stage_runtime_->tensor_parallel->shard
                   : config_;
    }
    // A mixer's output projection: added to the residual, or in a tensor-parallel pass written to
    // the rank's FP32 partial, which the exchange adds.
    void project_residual(const Tensor& input, const LinearParameters& p, Tensor& x,
                          InputBasis basis, bool wide_verification);
    // The Linear Attention state pool holding GDN layer `layer`, and its index within that pool.
    struct GdnStateRef {
        LinearAttentionStatePool* pool;
        std::uint32_t local;
    };
    [[nodiscard]] GdnStateRef gdn_state(std::uint32_t layer) const;
    // The shard holding GDN layer `layer` and the layer's index within it.
    struct GdnShard {
        std::size_t shard;
        std::uint32_t local;
    };
    [[nodiscard]] GdnShard gdn_shard(std::uint32_t layer) const;
    // ReplaySSM record storage for GDN layer `layer`, from the shard that holds it.
    [[nodiscard]] GdnReplayRecordLayer replay_layer(std::uint32_t layer, std::int32_t batch,
                                                    std::int32_t width) const;
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void window_mtp_attention(PagedKVBatchLayerView& view, Tensor& table_rows, Tensor& positions,
                              const Tensor& valid_columns);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);

    // A batched prefill segment's columns and sequence bindings within one pass. A negative
    // destination slot leaves the segment's GDN state in a transient buffer (a branch block).
    struct SegmentBinding {
        std::int32_t column  = 0;
        std::int32_t columns = 0;
        Tensor positions;
        Tensor kv_table_row;
        ops::CausalAttentionExecutionEnvelope envelope;
        std::int32_t state_source_slot      = 0;
        std::int32_t state_destination_slot = 0;
    };

    DeviceContext& ctx_;
    const Parameters& parameters_;
    NativeExpertRuntime* experts_ = nullptr;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    const MemoryStatistics* statistics_ = nullptr;
    bool fast_prefill_kernel_ = false;
    std::uint32_t text_kv_base_;
    std::uint32_t cache_position_shift_ = 0;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_memory_prefill_origin_                                    = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    // Non-empty during prefill_final_batch: a prefill pass's sequences, column range by range.
    std::span<const SegmentBinding> active_segments_;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;
    std::span<const cudaEvent_t> layer_ready_;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    std::int32_t readout_count_                 = 0;
    std::function<void()> first_token_hook_;
    ops::RopeYarn rope_yarn_;
    std::uint32_t mtp_attention_window_         = 0;
    const MtpParameters* mtp_                   = nullptr;
    StageRuntime* stage_runtime_                = nullptr;
    // A tensor-parallel pass: the rank whose share is being issued, and where its output
    // projections write their FP32 partial instead of adding to the residual.
    std::size_t tensor_rank_  = 0;
    Tensor* residual_partial_ = nullptr;
};

} // namespace ninfer::models::qwen3_5::execution
