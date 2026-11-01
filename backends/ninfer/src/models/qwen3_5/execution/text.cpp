#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/logprob_topk.h"
#include "ninfer/ops/readout_logprobs.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/paged_kv_window.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <optional>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        // pending_features is lane-owned at the frame's native width; a narrower round fills
        // the leading columns of each lane through the parent strides.
        if (target.ne[1] != batch_width) { target = target.slice(1, 0, batch_width); }
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv),
      batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
}

TextContext::~TextContext() = default;

void TextContext::capture_attention_tokens(const Tensor& ids) {
    if(!statistics_) return;
    auto tokens=statistics_->ranks.front().attention_tokens;
    if(!tokens.data || ids.dtype!=DType::I32 || !ids.is_contiguous() || ids.numel()>tokens.numel())
        throw std::logic_error("KVMem consumed token capture exceeds its planned capacity");
    CUDA_CHECK(cudaMemcpyAsync(tokens.data,ids.data,ids.bytes(),cudaMemcpyDeviceToDevice,ctx_.stream));
}

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        embed_tokens(flat_ids, *embed_, parameters_.text.token_embedding_signs, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});

    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_qk_norm_rope(rope_for_op, *config_.rope_parameters, *config_.attention,
                      config_.rms_norm_eps, mtp_->query_norm, mtp_->key_norm, q, k, qn, kn,
                      rope_yarn_, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor gate_batch     = gate.view({dimension(config_.attention->head_dim),
                                           dimension(config_.attention->num_attention_heads), width,
                                           active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        Tensor table_rows     = *active_backend_kv_table_rows_;
        auto view             = batch_mtp_kv_->batch_layer_view(0);
        if (mtp_attention_window_ != 0) {
            window_mtp_attention(view, table_rows, position_batch, *active_valid_columns_);
        }
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_, table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            view, envelope, work_, a_batch, s, &gate_batch);
    } else {
        Tensor position_rows = positions.view({T, 1});
        Tensor table_rows    = io_.backend_kv_table_row;
        auto view            = batch_mtp_kv_->batch_layer_view(0);
        if (mtp_attention_window_ != 0) {
            window_mtp_attention(view, table_rows, position_rows, Tensor{});
        }
        ops::causal_softmax_attention(
            qn, kn, v, position_rows.view({T}), Tensor{}, table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            view, envelope, work_, a, s, &gate);
    }

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, s, true, false, InputBasis::Primal, false, experts_);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

// The MTP draft attends to one sink page and the newest window, so its read stops growing with
// the context; target verification keeps full attention and alone decides what is committed. The
// K/V rows the draft appends land in their own pages through the windowed table.
void TextContext::window_mtp_attention(PagedKVBatchLayerView& view, Tensor& table_rows,
                                       Tensor& positions, const Tensor& valid_columns) {
    const auto window = workspace::paged_kv_window(work_, view.block_tables.ne[0], positions.ne[0],
                                                   table_rows.ne[0]);
    Tensor tables     = window.tables;
    Tensor rows       = window.rows;
    Tensor shifted    = window.positions;
    ops::paged_kv_window_rows(view.block_tables, table_rows, positions, valid_columns, 1,
                              static_cast<std::int32_t>(mtp_attention_window_), tables, rows,
                              shifted, ctx_.stream);
    view.block_tables = tables;
    table_rows        = rows;
    positions         = shifted;
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, rope_yarn_, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        mtp_query_gate_projection(ah_last, mtp_->projection, *config_.attention, q_flat, gate_flat,
                                  work_, s);
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        text_rope(last_rope_position, *config_.rope_parameters, rope_yarn_, qn, s);

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        ops::causal_softmax_attention_cached(
            qn, last_position,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o, work_,
                s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_last, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_last, {}, work_, s, true, false, InputBasis::Primal, false, experts_);
        }
        ops::rmsnorm(x_last, mtp_->final_norm, config_.rms_norm_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits =
            work_.alloc(fp32_proposal_logits(*proposal_head_) ? DType::FP32 : DType::BF16,
                        {proposal_head_n_, T});
        project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = fp32_proposal_logits(mtp_->output_head)
                                   ? work_.alloc(DType::FP32, {dimension(config_.vocab_size), T})
                                   : matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        embed_tokens(ids, *embed_, parameters_.text.token_embedding_signs, x, stream);
        capture_attention_tokens(ids);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden, stream);
        project(hidden, *lm_head_, logits, work_, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashVerifyMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        embed_tokens(flat_ids, *embed_, parameters_.text.token_embedding_signs, x, stream);
        capture_attention_tokens(flat_ids);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, flat_hidden, stream);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMtpVerifyMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph) {
    const auto& p  = std::get<AttentionParameters>(w.mixer);
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection        = workspace::text_attention_projection(work_, config_, T);
    const Tensor& input_signs    = projection_signs(p.projection);
    const InputBasis input_basis = rotated(input_signs) ? InputBasis::Rotated : InputBasis::Primal;
    // The NVFP4 W4A4 route can normalize while it quantizes, which skips the BF16 round trip of
    // the normalized hidden state, so only the other routes stage it.
    const auto* single     = std::get_if<LinearParameters>(&p.projection);
    const bool fused_nvfp4 = input_basis == InputBasis::Primal && single != nullptr &&
                             ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(single->weight,
                                                                               single->policy, T);
    // The integer route carries the same edge: one pass over the residual row produces the s8 codes
    // and per-32 scales the grouped or narrow mainloop reads, so the rmsnorm below is skipped
    // rather than run and then re-read. The policies are disjoint, and the band decides the width.
    const bool fused_int8 = !fused_nvfp4 && input_basis == InputBasis::Primal && single != nullptr &&
                            ops::attn_input_proj_fused_rmsnorm_int8_eligible(
                                single->weight, single->policy, T);
    Tensor h;
    if (!fused_nvfp4 && !fused_int8) {
        h = workspace::matrix(work_, DType::BF16, dimension(config_.hidden_size), T);
    }
    if (input_basis == InputBasis::Rotated) {
        ops::rmsnorm_hadamard(x, w.input_norm, config_.rms_norm_eps, true, input_signs, h, s);
    } else if (!fused_nvfp4 && !fused_int8) {
        ops::rmsnorm(x, w.input_norm, config_.rms_norm_eps, true, h, s);
    }

    Tensor q         = projection.query.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(config_.attention->head_dim),
                                            dimension(config_.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(config_.attention->head_dim),
                                              dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat    = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat    = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat    = v.view({dimension(config_.attention->key_width()), T});
    if (fused_nvfp4) {
        ops::attn_input_proj_fused_rmsnorm_nvfp4(x, w.input_norm, config_.rms_norm_eps,
                                                 single->weight, q_flat, gate_flat, k_flat, v_flat,
                                                 single->policy, work_, s);
    } else if (fused_int8) {
        ops::attn_input_proj_fused_rmsnorm_int8(x, w.input_norm, config_.rms_norm_eps,
                                                single->weight, q_flat, gate_flat, k_flat, v_flat,
                                                single->policy, work_, s);
    } else {
        attention_projection(h, p, q_flat, gate_flat, k_flat, v_flat, work_, s, input_basis);
    }

    const auto results = workspace::text_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    MemoryStatisticsShard local_statistics;
    const MemoryStatisticsShard* statistics = nullptr;
    std::int32_t statistics_layer = fidx;
    if (statistics_ != nullptr) {
        local_statistics = statistics_->ranks.at(ctx_.active_rank());
        if (active_memory_prefill_origin_ != nullptr) {
            local_statistics.prefill_origin = *active_memory_prefill_origin_;
        }
        statistics_layer -= static_cast<std::int32_t>(local_statistics.first_layer);
        if (statistics_layer < 0 || statistics_layer >= local_statistics.key_sums.ne[2]) {
            throw std::logic_error("KVMem statistics layer is outside its owning stage");
        }
        statistics = &local_statistics;
    }
    text_qk_norm_rope(rope_for_op, *config_.rope_parameters, *config_.attention,
                      config_.rms_norm_eps, p.query_norm, p.key_norm, q, k, qn, kn, rope_yarn_,
                      s, statistics, statistics_layer);
    if(statistics && statistics->attention_queries.data) {
        const std::int32_t selected[]{0,static_cast<std::int32_t>(config_.full_attention_layers/2),
            static_cast<std::int32_t>(config_.full_attention_layers-1)};
        for(std::int32_t slot=0;slot<3;++slot) {
            if(selected[slot]!=fidx || (slot && selected[slot]==selected[slot-1])) continue;
            auto capture=statistics->attention_queries.slice(2,slot,1).slice(1,0,T)
                .view({dimension(config_.attention->query_width()),T});
            CUDA_CHECK(cudaMemcpyAsync(capture.data,qn.data,capture.bytes(),cudaMemcpyDeviceToDevice,s));
        }
    }

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    const bool rotated_output = rotated(p.output.hadamard_signs);
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor gate_batch     = gate.view({dimension(config_.attention->head_dim),
                                           dimension(config_.attention->num_attention_heads), width,
                                           active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        // The gate rides along with the attention call: where the route can, the reduce epilogue
        // applies it at the store, and every other route applies it inside the Op. One contract
        // either way, and the same bytes the standalone multiply produced. A rotated output
        // projection takes the ungated attention, gated and rotated in one pass below.
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, s, rotated_output ? nullptr : &gate_batch);
    } else if (!active_segments_.empty()) {
        // Each segment attends over its own sequence exactly as its prefill alone would.
        for (const SegmentBinding& segment : active_segments_) {
            auto scope           = work_.scope();
            Tensor a_segment     = a.slice(2, segment.column, segment.columns);
            Tensor gate_segment  = gate.slice(2, segment.column, segment.columns);
            ops::causal_softmax_attention(
                qn.slice(2, segment.column, segment.columns),
                kn.slice(2, segment.column, segment.columns),
                v.slice(2, segment.column, segment.columns), segment.positions, Tensor{},
                segment.kv_table_row,
                {dimension(config_.attention->head_dim),
                 dimension(config_.attention->num_attention_heads),
                 dimension(config_.attention->num_key_value_heads)},
                static_cast<float>(1.0 /
                                   std::sqrt(static_cast<double>(config_.attention->head_dim))),
                batch_text_kv_->batch_layer_view(fidx), segment.envelope, work_, a_segment, s,
                rotated_output ? nullptr : &gate_segment);
        }
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a, s,
            rotated_output ? nullptr : &gate);
    }
    const bool wide = wide_residual_verification(ph, row_width(T), row_width(T));
    Tensor gated    = a.view({dimension(config_.attention->query_width()), T});
    if (rotated_output) {
        ops::sigmoid_mul_hadamard(gate, a, p.output.hadamard_signs, gated, s);
        project_residual(gated, p.output, x, InputBasis::Rotated, wide);
        return;
    }
    project_residual(gated, p.output, x, InputBasis::Primal, wide);
}

void TextContext::project_residual(const Tensor& input, const LinearParameters& p, Tensor& x,
                                   InputBasis basis, bool wide_verification) {
    if (residual_partial_ == nullptr) {
        project_add(input, p, x, work_, ctx_.stream, basis, wide_verification);
        return;
    }
    auto scope     = work_.scope();
    const Tensor v = rotated_input(input, p.hadamard_signs, work_, ctx_.stream, basis);
    ops::linear(v, p.weight, *residual_partial_, residual_projection_policy(p, wide_verification),
                work_, ctx_.stream);
}

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph) {
    const auto& p                = std::get<GdnParameters>(w.mixer);
    cudaStream_t s               = ctx_.stream;
    const int T                  = x.ne[1];
    const TextConfig& config     = layer_config();

    // A tensor-parallel rank gates every head, as one device would, and keeps its own.
    const auto control = workspace::gdn_control(work_, config_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    const InputBasis projection_basis = gdn_norm_control(x, w.input_norm, config.rms_norm_eps, p,
                                                         h, g, beta, work_, ctx_.execution_view());
    if (&config != &config_) {
        const auto heads       = static_cast<std::size_t>(config.gdn->linear_num_value_heads);
        const auto all_heads   = static_cast<std::size_t>(config_.gdn->linear_num_value_heads);
        const auto own_gates   = [&](const Tensor& all) {
            Tensor own = work_.alloc(DType::FP32, {static_cast<std::int32_t>(heads), T});
            CUDA_CHECK(cudaMemcpy2DAsync(own.data, heads * sizeof(float),
                                         static_cast<const float*>(all.data) + tensor_rank_ * heads,
                                         all_heads * sizeof(float), heads * sizeof(float),
                                         static_cast<std::size_t>(T), cudaMemcpyDeviceToDevice, s));
            return own;
        };
        g    = own_gates(g);
        beta = own_gates(beta);
    }

    const auto projection = workspace::gdn_projection(work_, config, T);
    Tensor z  = projection.output_gate.view({dimension(config.gdn->linear_value_head_dim),
                                             dimension(config.gdn->linear_num_value_heads), T});
    Tensor qc = projection.query;
    Tensor kc = projection.key;
    Tensor vc = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input =
            h.view({dimension(config.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(config.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(config.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(config.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(config.gdn->value_width()), width, active_sequence_batch_});
        const GdnStateRef gdn_pool = gdn_state(static_cast<std::uint32_t>(gidx));
        Tensor conv_states         = gdn_pool.pool->layer_view(gdn_pool.local).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records =
                replay_layer(static_cast<std::uint32_t>(gidx), active_sequence_batch_, width);
            gdn_projection_record(projection_input, p, *config.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s,
                                  projection_basis);
        } else {
            gdn_projection_snapshot(
                projection_input, p, *config.gdn, conv_states, valid,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                query_output, key_output, value_output, gate_output, work_, s, projection_basis);
        }
    } else {
        Tensor qkv    = workspace::gdn_prefill_conv(work_, config, T);
        Tensor z_flat = z.view({dimension(config.gdn->value_width()), T});
        gdn_projection(h, p, qkv, z_flat, work_, s, projection_basis);
        const GdnStateRef gdn_pool = gdn_state(static_cast<std::uint32_t>(gidx));
        if (!active_segments_.empty()) {
            for (const SegmentBinding& segment : active_segments_) {
                auto scope = work_.scope();
                const Tensor conv_state_in =
                    gdn_pool.pool->conv_slot(gdn_pool.local, segment.state_source_slot);
                // A branch block keeps no state: its convolution window goes to scratch.
                Tensor conv_state_out =
                    segment.state_destination_slot >= 0
                        ? gdn_pool.pool->conv_slot(gdn_pool.local, segment.state_destination_slot)
                        : work_.alloc(conv_state_in.dtype,
                                      {conv_state_in.ne[0], conv_state_in.ne[1]});
                Tensor q_segment = qc.slice(1, segment.column, segment.columns);
                Tensor k_segment = kc.slice(1, segment.column, segment.columns);
                Tensor v_segment = vc.slice(1, segment.column, segment.columns);
                ops::causal_conv1d_silu_split(qkv.slice(1, segment.column, segment.columns),
                                              p.convolution, conv_state_in, conv_state_out,
                                              q_segment, k_segment, v_segment, s);
            }
        } else {
            Tensor conv_state_in =
                gdn_pool.pool->conv_slot(gdn_pool.local, linear_state_source_slot_);
            Tensor conv_state_out =
                gdn_pool.pool->conv_slot(gdn_pool.local, linear_state_destination_slot_);
            ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc,
                                          kc, vc, s);
        }
    }

    Tensor q_recurrent = qc.view({dimension(config.gdn->linear_key_head_dim),
                                  dimension(config.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(config.gdn->linear_key_head_dim),
                                  dimension(config.gdn->linear_num_key_heads), T});

    Tensor vv = vc.view({dimension(config.gdn->linear_value_head_dim),
                         dimension(config.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, config, T)
                   .view({dimension(config.gdn->linear_value_head_dim),
                          dimension(config.gdn->linear_num_value_heads), T});
    if (ph == Phase::Verify) {
        const GdnStateRef recurrent_pool = gdn_state(static_cast<std::uint32_t>(gidx));
        Tensor recurrent_states  = recurrent_pool.pool->layer_view(recurrent_pool.local).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch           = q_recurrent.view({dimension(config.gdn->linear_key_head_dim),
                                                     dimension(config.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(config.gdn->linear_key_head_dim),
                                                     dimension(config.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(config.gdn->linear_value_head_dim),
                                            dimension(config.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g.view({dimension(config.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta.view(
            {dimension(config.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(config.gdn->linear_value_head_dim),
                    dimension(config.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records =
                replay_layer(static_cast<std::uint32_t>(gidx), active_sequence_batch_, width);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else if (!active_segments_.empty()) {
        const GdnStateRef recurrent_pool = gdn_state(static_cast<std::uint32_t>(gidx));
        for (const SegmentBinding& segment : active_segments_) {
            auto scope = work_.scope();
            const Tensor recurrent_state_in =
                recurrent_pool.pool->recurrent_slot(recurrent_pool.local, segment.state_source_slot);
            // A branch block starts from the state its read's prompt left and keeps nothing.
            Tensor recurrent_state_out =
                segment.state_destination_slot >= 0
                    ? recurrent_pool.pool->recurrent_slot(recurrent_pool.local,
                                                          segment.state_destination_slot)
                    : work_.alloc(recurrent_state_in.dtype,
                                  {recurrent_state_in.ne[0], recurrent_state_in.ne[1],
                                   recurrent_state_in.ne[2]});
            Tensor o_segment = o.slice(2, segment.column, segment.columns);
            ops::gated_delta_net(
                q_recurrent.slice(2, segment.column, segment.columns),
                k_recurrent.slice(2, segment.column, segment.columns),
                vv.slice(2, segment.column, segment.columns),
                g.slice(1, segment.column, segment.columns),
                beta.slice(1, segment.column, segment.columns),
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(config.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o_segment,
                s);
        }
    } else {
        const GdnStateRef recurrent_pool = gdn_state(static_cast<std::uint32_t>(gidx));
        Tensor recurrent_state_in =
            recurrent_pool.pool->recurrent_slot(recurrent_pool.local, linear_state_source_slot_);
        Tensor recurrent_state_out = recurrent_pool.pool->recurrent_slot(
            recurrent_pool.local, linear_state_destination_slot_);
        ops::gated_delta_net(
            q_recurrent, k_recurrent, vv, g, beta,
            static_cast<float>(1.0 /
                               std::sqrt(static_cast<double>(config.gdn->linear_key_head_dim))),
            /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o, s);
    }

    Tensor on = workspace::gdn_normalized_output(work_, config, T)
                    .view({dimension(config.gdn->linear_value_head_dim),
                           dimension(config.gdn->linear_num_value_heads), T});
    Tensor normalized = on.view({dimension(config.gdn->value_width()), T});
    if (rotated(p.output.hadamard_signs)) {
        ops::gated_rmsnorm_hadamard(o, p.norm, z, config.rms_norm_eps, p.output.hadamard_signs,
                                    normalized, s);
        project_residual(normalized, p.output, x, InputBasis::Rotated, false);
        return;
    }
    ops::gated_rmsnorm(o, p.norm, z, ops::GateActivation::Silu, config.rms_norm_eps, on, s);
    project_residual(normalized, p.output, x, InputBasis::Primal,
                     wide_residual_verification(ph, row_width(T), row_width(T)));
}

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                           const ops::SparseMoeHints& hints) {
    const TextConfig& config = layer_config();
    Tensor h                 = workspace::post_mixer_hidden(work_, config, x.ne[1]);
    if (const Tensor* signs = ffn_input_signs(weights.ffn)) {
        ops::rmsnorm_hadamard(x, weights.post_attention_norm, config.rms_norm_eps, true, *signs, h,
                              ctx_.stream);
        ffn(h, weights.ffn, x, hints, work_, ctx_.stream, false, phase == Phase::Verify,
            InputBasis::Rotated, false, experts_, nullptr, residual_partial_);
        return;
    }
    // The integer gate_up route normalises while it quantises, so the BF16 normed row never exists:
    // the residual goes into the FFN un-normed and carries the norm with it. The predicate is the one
    // the projection itself asks, evaluated with the policy this phase will pass down.
    const auto* dense = std::get_if<DenseParameters>(&weights.ffn);
    const bool fused_int8 =
        dense != nullptr && !dense->up.has_value() &&
        ops::linear_swiglu_fused_rmsnorm_int8_eligible(
            dense->gate_up.weight,
            phase == Phase::Verify ? dense->verify_gate_up_policy : dense->gate_up.policy,
            static_cast<std::int32_t>(x.ne[1]));
    const bool wide =
        wide_residual_verification(phase, row_width(x.ne[1]), row_width(x.ne[1]));
    if (fused_int8) {
        // The share writes the row it is given and the all-reduce sums those shares, so a deferred
        // norm has nowhere to go here. Refuse the combination instead of answering from x.
        if (residual_partial_ != nullptr) {
            throw std::logic_error("FFN: the fused int8 input norm has no tensor-parallel share");
        }
        const InputNorm norm{&weights.post_attention_norm, config.rms_norm_eps};
        ffn(x, weights.ffn, x, hints, work_, ctx_.stream, false, phase == Phase::Verify,
            InputBasis::Primal, wide, experts_, &norm, nullptr);
        return;
    }
    ops::rmsnorm(x, weights.post_attention_norm, config.rms_norm_eps, true, h, ctx_.stream);
    ffn(h, weights.ffn, x, hints, work_, ctx_.stream, false, phase == Phase::Verify,
        InputBasis::Primal, wide, experts_, nullptr, residual_partial_);
}

TextContext::GdnShard TextContext::gdn_shard(std::uint32_t layer) const {
    if (stage_runtime_ == nullptr) { return {0, layer}; }
    // Every rank holds its share of every layer.
    if (stage_runtime_->tensor_parallel) { return {tensor_rank_, layer}; }
    // The shard holding a layer is the last one that starts at or before it.
    for (std::size_t shard = stage_runtime_->state.size(); shard-- > 0;) {
        if (stage_runtime_->state_first_layer[shard] <= layer) {
            return {shard, layer - stage_runtime_->state_first_layer[shard]};
        }
    }
    throw std::logic_error("Linear Attention layer has no state shard");
}

TextContext::GdnStateRef TextContext::gdn_state(std::uint32_t layer) const {
    if (stage_runtime_ == nullptr) { return {&state_, layer}; }
    const GdnShard located = gdn_shard(layer);
    return {stage_runtime_->state[located.shard], located.local};
}

GdnReplayRecordLayer TextContext::replay_layer(std::uint32_t layer, std::int32_t batch,
                                               std::int32_t width) const {
    if (stage_runtime_ == nullptr || stage_runtime_->replay.empty()) {
        return replay_records_->layer(static_cast<std::int32_t>(layer), batch, width);
    }
    const GdnShard located = gdn_shard(layer);
    return stage_runtime_->replay[located.shard]->layer(static_cast<std::int32_t>(located.local),
                                                        batch, width);
}

// The layers of one stage, in order. On one device this is the whole model.
template <class Tap>
void TextContext::run_stage_layers(std::size_t stage, Tensor& x, Phase ph, Tap& tap) {
    const bool prefill      = ph == Phase::Prefill;
    const std::uint32_t end = parameters_.text.stage_end(stage);
    for (std::uint32_t layer = parameters_.text.stage_begin[stage]; layer < end; ++layer) {
        const auto& block  = parameters_.text.layers[layer];
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = dimension(config_.compact_layer_indices[layer]);
        nvtx::ScopedRange layer_range(
            full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
                 : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        try {
            if (layer < layer_ready_.size()) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx_.stream, layer_ready_[layer], 0));
            }
            {
                nvtx::ScopedRange mixer_range(
                    full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                         : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
                    full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
                auto scope = work_.scope();
                if (full) {
                    attn_mix(block, x, compact, ph);
                } else {
                    gdn_mix(block, x, compact, ph);
                }
            }
            {
                nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer
                                                : nvtx::Name::VerifyPostMixer,
                                        nvtx::Category::PostMixer, layer);
                auto scope = work_.scope();
                // The prefetch hint names pointers into the *next* layer's projection weights. It
                // only helps when that layer runs next on this device, so it is dropped at the last
                // layer of a stage exactly as it is at the last layer of the model.
                mlp_tail(block, x, ph,
                         layer + 1 < end ? next_projection_hints(static_cast<int>(layer))
                                         : ops::SparseMoeHints{});
            }
            if constexpr (Tap::enabled) {
                tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " verify") +
                                     " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
        }
    }
    // Later passes are stream-ordered behind this one.
    layer_ready_ = {};
}

namespace {

// The device tensors layer kernels read that live in rank 0's memory.
constexpr std::size_t kControlTensors = 7;

// Points a TextContext's active control tensors at stage-local copies for one stage, and puts the
// originals back after it.
class ScopedControl {
public:
    ScopedControl(std::array<const Tensor*, kControlTensors> current,
                  std::array<const Tensor*, kControlTensors> replacement,
                  std::array<const Tensor**, kControlTensors> slots)
        : current_(current), slots_(slots) {
        for (std::size_t i = 0; i < kControlTensors; ++i) { *slots_[i] = replacement[i]; }
    }
    ScopedControl(const ScopedControl&)            = delete;
    ScopedControl& operator=(const ScopedControl&) = delete;
    ~ScopedControl() {
        for (std::size_t i = 0; i < kControlTensors; ++i) { *slots_[i] = current_[i]; }
    }

private:
    std::array<const Tensor*, kControlTensors> current_;
    std::array<const Tensor**, kControlTensors> slots_;
};

// A later stage's DFlash taps, taken in that stage's memory: each tapped layer's output is copied
// into its own [hidden, columns] slab of `staging`, in the order of the link's layers, and leaves
// for rank 0 on the link's stream while the stage's later layers run.
struct StageFeatureTap {
    static constexpr bool enabled = true;

    StageRuntime::FeatureLink* link = nullptr;
    Tensor staging;
    std::uint32_t slot = 0;

    void capture_layer(int layer, const Tensor& value, cudaStream_t stream) const {
        const auto& layers = link->layers;
        const auto it = std::find(layers.begin(), layers.end(), static_cast<std::uint32_t>(layer));
        if (it == layers.end()) { return; }
        const auto index = static_cast<std::size_t>(it - layers.begin());
        if (value.dtype != staging.dtype || !value.is_contiguous() ||
            value.ne[0] != staging.ne[0] ||
            static_cast<std::int64_t>(value.ne[1]) * static_cast<std::int64_t>(layers.size()) !=
                staging.ne[1]) {
            throw std::logic_error("a pipeline stage feature tap does not match its staging");
        }
        Tensor slab = staging.slice(1, static_cast<std::int32_t>(index) * value.ne[1], value.ne[1]);
        CUDA_CHECK(cudaMemcpyAsync(slab.data, value.data, value.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
        link->ready[index].record(stream);
        link->ready[index].wait(link->stream.get());
        link->links[index].send(slab.data, slab.bytes(), slot, link->stream.get());
    }
};

} // namespace

// Runs the layers across pipeline stages. Stage 0 is rank 0, where the caller's residual already
// lives; each later stage receives the residual and the control tensors its layers read, runs, and
// passes the residual on; the last stage returns it to rank 0 for the head.
//
// A tap writes into rank 0's buffers, which a later stage's device cannot reach, so only stage 0
// hands its layers to the tap directly. A later stage stages each tapped output in its own scratch
// and sends it to rank 0 while its later layers run; rank 0, idle until the residual comes back,
// receives each output as it lands and gives it to the tap as if that layer had run there.
template <class Tap>
void TextContext::run_staged(Tensor& x, Phase ph, Tap& tap) {
    StageRuntime& runtime    = *stage_runtime_;
    const std::size_t stages = parameters_.text.rank_count;
    if (runtime.forward.size() + 1 != stages || runtime.control.size() + 1 != stages ||
        !runtime.back.has_value() || ctx_.active_rank() != 0) {
        throw std::logic_error("stage runtime does not match the model's stages");
    }
    if constexpr (Tap::enabled) {
        if (runtime.features.size() + 1 != stages) {
            throw std::logic_error("layer feature taps across pipeline stages have no links");
        }
    }
    const std::uint32_t slot = runtime.next_slot;
    runtime.next_slot        = (slot + 1U) % static_cast<std::uint32_t>(runtime.forward.front().slots());

    // Control tensors, packed in this order. Ones that are absent this pass (verify-only columns
    // and slots, say) are skipped.
    const std::array<const Tensor**, kControlTensors> slots = {
        &active_cache_positions_,        &active_rope_positions_,
        &active_kv_table_rows_,          &active_valid_columns_,
        &active_linear_state_source_slots_, &active_linear_state_destination_slots_,
        &active_memory_prefill_origin_};
    const std::array<const Tensor*, kControlTensors> fallback = {
        &io_.pos, &io_.rope_pos, &io_.text_kv_table_row, nullptr, nullptr, nullptr,
        statistics_ != nullptr && statistics_->ranks.front().block_tokens != 1
            ? &statistics_->ranks.front().prefill_origin : nullptr};
    std::array<const Tensor*, kControlTensors> current{};
    std::array<std::int64_t, kControlTensors> offset{};
    std::int64_t total = 0;
    for (std::size_t i = 0; i < kControlTensors; ++i) {
        const Tensor* source = *slots[i] != nullptr ? *slots[i] : fallback[i];
        current[i]           = *slots[i];
        if (source == nullptr || source->data == nullptr) {
            offset[i] = -1;
            continue;
        }
        if (source->dtype != DType::I32 || !source->is_contiguous()) {
            throw std::logic_error("a pipeline stage control tensor is not contiguous int32");
        }
        offset[i] = total;
        total += source->numel();
    }
    total                          = std::max<std::int64_t>((total + 3) & ~std::int64_t{3}, 4);
    const std::size_t control_bytes = static_cast<std::size_t>(total) * sizeof(std::int32_t);
    if (control_bytes > runtime.control.front().slot_bytes()) {
        throw std::runtime_error("pipeline stage control block exceeds its link");
    }

    const cudaStream_t entry_stream = ctx_.stream_for_rank(0);
    auto entry_scope                = work_.scope();
    Tensor pack                     = work_.alloc(DType::I32, {static_cast<std::int32_t>(total)});
    std::array<const Tensor*, kControlTensors> sources{};
    for (std::size_t i = 0; i < kControlTensors; ++i) {
        if (offset[i] < 0) { continue; }
        sources[i] = *slots[i] != nullptr ? *slots[i] : fallback[i];
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(pack.data) + offset[i],
                                   sources[i]->data, sources[i]->bytes(), cudaMemcpyDeviceToDevice,
                                   entry_stream));
    }
    // Every stage's control block leaves rank 0 up front, so it is on its way while stage 0 computes.
    for (std::size_t stage = 1; stage < stages; ++stage) {
        runtime.control[stage - 1].send(pack.data, control_bytes, slot, entry_stream);
    }

    run_stage_layers(0, x, ph, tap);
    runtime.forward[0].send(x.data, x.bytes(), slot, entry_stream);

    for (std::size_t stage = 1; stage < stages; ++stage) {
        // Rebinds the context's device, stream and workspace to this stage, and back afterwards.
        ScopedDeviceRank rank_guard(ctx_, stage);
        ScopedArenaRank arena_guard(work_, stage);
        auto scope = work_.scope();
        Tensor stage_x = work_.alloc(x.dtype, {x.ne[0], x.ne[1]});
        runtime.forward[stage - 1].recv(stage_x.data, stage_x.bytes(), slot, ctx_.stream);

        Tensor control = work_.alloc(DType::I32, {static_cast<std::int32_t>(total)});
        runtime.control[stage - 1].recv(control.data, control_bytes, slot, ctx_.stream);
        std::array<Tensor, kControlTensors> local{};
        std::array<const Tensor*, kControlTensors> replacement{};
        for (std::size_t i = 0; i < kControlTensors; ++i) {
            if (offset[i] < 0) { continue; }
            local[i]      = *sources[i];
            local[i].data = static_cast<std::int32_t*>(control.data) + offset[i];
            replacement[i] = &local[i];
        }
        ScopedControl bound(current, replacement, slots);

        StageRuntime::FeatureLink* features = nullptr;
        if constexpr (Tap::enabled) {
            if (runtime.features[stage - 1].has_value()) {
                features = &*runtime.features[stage - 1];
            }
        }
        if (features != nullptr) {
            const auto taps = static_cast<std::int32_t>(features->layers.size());
            StageFeatureTap stage_tap{
                .link    = features,
                .staging = work_.alloc(stage_x.dtype, {stage_x.ne[0], stage_x.ne[1] * taps}),
                .slot    = slot};
            run_stage_layers(stage, stage_x, ph, stage_tap);
        } else {
            NullTap none;
            run_stage_layers(stage, stage_x, ph, none);
        }
        if (stage + 1 < stages) {
            runtime.forward[stage].send(stage_x.data, stage_x.bytes(), slot, ctx_.stream);
        } else {
            runtime.back->send(stage_x.data, stage_x.bytes(), slot, ctx_.stream);
        }
        // The stage's stream takes the feature stream back, so the staging outlives its sends.
        if (features != nullptr) {
            features->sent.record(features->stream.get());
            features->sent.wait(ctx_.stream);
        }
    }

    if constexpr (Tap::enabled) {
        // Each tapped output is received as it lands, so all but the last stage's last ones have
        // crossed before the residual returns.
        auto scope = work_.scope();
        Tensor received;
        for (std::size_t stage = 1; stage < stages; ++stage) {
            if (!runtime.features[stage - 1].has_value()) { continue; }
            StageRuntime::FeatureLink& features = *runtime.features[stage - 1];
            if (received.data == nullptr) { received = work_.alloc(x.dtype, {x.ne[0], x.ne[1]}); }
            for (std::size_t index = 0; index < features.layers.size(); ++index) {
                features.links[index].recv(received.data, received.bytes(), slot, entry_stream);
                tap.capture_layer(static_cast<int>(features.layers[index]), received, entry_stream);
            }
        }
    }
    runtime.back->recv(x.data, x.bytes(), slot, entry_stream);
}

// Runs every layer with each rank issuing its share of it. Rank 0 sends the residual and the
// control tensors to the other ranks once, and each holds a replica of the residual from then on.
// A GDN mixer or an FFN is one exchange: every rank projects its share into an FP32 partial and
// every replica adds the sum of the partials. Attention runs on rank 0 alone, an exchange with one
// contributor. The replicas stay equal, and rank 0's ends the pass where one device's would.
template <class Tap>
void TextContext::run_tensor_parallel(Tensor& x, Phase ph, Tap& tap) {
    if constexpr (Tap::enabled) {
        (void)tap;
        throw std::logic_error("layer feature taps are not available with tensor parallelism");
    }
    StageRuntime& runtime            = *stage_runtime_;
    auto& tp                         = *runtime.tensor_parallel;
    TensorParallelExchange& exchange = *tp.exchange;
    const std::size_t ranks          = exchange.ranks();
    if (runtime.forward.size() + 1 != ranks || runtime.control.size() + 1 != ranks ||
        tp.joined.size() + 1 != ranks || parameters_.text.rank_layers.size() + 1 != ranks ||
        ctx_.active_rank() != 0) {
        throw std::logic_error("tensor-parallel runtime does not match the model's ranks");
    }
    const std::uint32_t slot = runtime.next_slot;
    runtime.next_slot = (slot + 1U) % static_cast<std::uint32_t>(runtime.forward.front().slots());

    // The control tensors, packed as for pipeline stages.
    const std::array<const Tensor**, kControlTensors> slots = {
        &active_cache_positions_,           &active_rope_positions_,
        &active_kv_table_rows_,             &active_valid_columns_,
        &active_linear_state_source_slots_, &active_linear_state_destination_slots_,
        &active_memory_prefill_origin_};
    const std::array<const Tensor*, kControlTensors> fallback = {
        &io_.pos, &io_.rope_pos, &io_.text_kv_table_row, nullptr, nullptr, nullptr,
        statistics_ != nullptr && statistics_->ranks.front().block_tokens != 1
            ? &statistics_->ranks.front().prefill_origin : nullptr};
    std::array<const Tensor*, kControlTensors> current{};
    std::array<const Tensor*, kControlTensors> sources{};
    std::array<std::int64_t, kControlTensors> offset{};
    std::int64_t total = 0;
    for (std::size_t i = 0; i < kControlTensors; ++i) {
        current[i] = *slots[i];
        sources[i] = *slots[i] != nullptr ? *slots[i] : fallback[i];
        if (sources[i] == nullptr || sources[i]->data == nullptr) {
            offset[i] = -1;
            continue;
        }
        if (sources[i]->dtype != DType::I32 || !sources[i]->is_contiguous()) {
            throw std::logic_error("a tensor-parallel control tensor is not contiguous int32");
        }
        offset[i] = total;
        total += sources[i]->numel();
    }
    total                           = std::max<std::int64_t>((total + 3) & ~std::int64_t{3}, 4);
    const std::size_t control_bytes = static_cast<std::size_t>(total) * sizeof(std::int32_t);
    const std::int64_t values = static_cast<std::int64_t>(x.ne[0]) * x.ne[1];
    const std::size_t partial_bytes = static_cast<std::size_t>(values) * sizeof(float);
    if (control_bytes > runtime.control.front().slot_bytes() ||
        x.bytes() > runtime.forward.front().slot_bytes() || partial_bytes > exchange.slot_bytes()) {
        throw std::runtime_error("a tensor-parallel pass exceeds its links");
    }

    const cudaStream_t entry_stream = ctx_.stream_for_rank(0);
    auto entry_scope                = work_.scope();
    Tensor pack = work_.alloc(DType::I32, {static_cast<std::int32_t>(total)});
    for (std::size_t i = 0; i < kControlTensors; ++i) {
        if (offset[i] < 0) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(pack.data) + offset[i],
                                   sources[i]->data, sources[i]->bytes(),
                                   cudaMemcpyDeviceToDevice, entry_stream));
    }
    for (std::size_t rank = 1; rank < ranks; ++rank) {
        runtime.control[rank - 1].send(pack.data, control_bytes, slot, entry_stream);
        runtime.forward[rank - 1].send(x.data, x.bytes(), slot, entry_stream);
    }

    // Each rank's replica and control copies, held in its own scratch for the whole pass.
    struct Replica {
        Tensor x;
        std::array<Tensor, kControlTensors> local{};
        std::array<const Tensor*, kControlTensors> control{};
    };
    std::vector<Replica> replicas(ranks);
    std::vector<WorkspaceArena::Scope> pass_scopes;
    pass_scopes.reserve(ranks);
    replicas[0].x = x;
    for (std::size_t rank = 1; rank < ranks; ++rank) {
        ScopedDeviceRank rank_guard(ctx_, rank);
        ScopedArenaRank arena_guard(work_, rank);
        pass_scopes.push_back(work_.scope());
        Replica& replica = replicas[rank];
        replica.x        = work_.alloc(x.dtype, {x.ne[0], x.ne[1]});
        runtime.forward[rank - 1].recv(replica.x.data, replica.x.bytes(), slot, ctx_.stream);
        Tensor control = work_.alloc(DType::I32, {static_cast<std::int32_t>(total)});
        runtime.control[rank - 1].recv(control.data, control_bytes, slot, ctx_.stream);
        for (std::size_t i = 0; i < kControlTensors; ++i) {
            if (offset[i] < 0) { continue; }
            replica.local[i]      = *sources[i];
            replica.local[i].data = static_cast<std::int32_t*>(control.data) + offset[i];
            replica.control[i]    = &replica.local[i];
        }
    }

    // Issues `share` for rank `rank` on its device, against its scratch, control and replica.
    const auto on_rank = [&](std::size_t rank, const auto& share) {
        std::optional<ScopedDeviceRank> device_guard;
        std::optional<ScopedArenaRank> arena_guard;
        std::optional<ScopedControl> control_guard;
        if (rank != 0) {
            device_guard.emplace(ctx_, rank);
            arena_guard.emplace(work_, rank);
            control_guard.emplace(current, replicas[rank].control, slots);
        }
        tensor_rank_ = rank;
        try {
            auto scope = work_.scope();
            share(replicas[rank].x);
        } catch (...) {
            tensor_rank_      = 0;
            residual_partial_ = nullptr;
            throw;
        }
        tensor_rank_ = 0;
    };
    // One exchange: ranks [0, contributors) project into their partials, every replica adds them.
    std::size_t next_exchange = 0;
    const auto exchange_step  = [&](std::size_t contributors, const auto& share) {
        const std::size_t exchange_slot = next_exchange;
        next_exchange ^= 1U;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            on_rank(rank, [&](Tensor& replica) {
                const bool contributes = rank < contributors;
                if (contributes) {
                    Tensor partial(exchange.partial(rank, exchange_slot), DType::FP32,
                                   {x.ne[0], x.ne[1]});
                    residual_partial_ = &partial;
                    share(rank, replica);
                    residual_partial_ = nullptr;
                }
                exchange.publish(rank, exchange_slot, contributes ? partial_bytes : 0,
                                 ctx_.stream);
            });
        }
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            on_rank(rank, [&](Tensor& replica) {
                exchange.reduce(rank, exchange_slot, static_cast<__nv_bfloat16*>(replica.data),
                                values, contributors, ctx_.stream);
            });
        }
    };

    const auto layers = static_cast<std::uint32_t>(parameters_.text.layers.size());
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const auto block = [&](std::size_t rank) -> const BlockParameters& {
            return rank == 0 ? parameters_.text.layers[layer]
                             : parameters_.text.rank_layers[rank - 1][layer];
        };
        const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
        const auto compact = dimension(config_.compact_layer_indices[layer]);
        try {
            if (layer < layer_ready_.size()) {
                CUDA_CHECK(cudaStreamWaitEvent(entry_stream, layer_ready_[layer], 0));
            }
            if (full) {
                exchange_step(1, [&](std::size_t, Tensor& replica) {
                    attn_mix(block(0), replica, compact, ph);
                });
            } else {
                exchange_step(ranks, [&](std::size_t rank, Tensor& replica) {
                    gdn_mix(block(rank), replica, compact, ph);
                });
            }
            exchange_step(ranks, [&](std::size_t rank, Tensor& replica) {
                mlp_tail(block(rank), replica, ph,
                         rank == 0 && layer + 1 < layers
                             ? next_projection_hints(static_cast<int>(layer))
                             : ops::SparseMoeHints{});
            });
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (ph == Phase::Prefill ? " prefill" : " verify") +
                                     " tensor-parallel columns=" + std::to_string(x.ne[1]) + ": " +
                                     error.what());
        }
    }
    // A captured pass ends with every rank joined back into rank 0's stream.
    for (std::size_t rank = 1; rank < ranks; ++rank) {
        tp.joined[rank - 1].record(ctx_.stream_for_rank(rank));
        tp.joined[rank - 1].wait(entry_stream);
    }
    layer_ready_ = {};
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    if (!parameters_.text.split_execution()) {
        run_stage_layers(0, x, ph, tap);
        return;
    }
    if (stage_runtime_ == nullptr) {
        throw std::logic_error("a split model needs its stage runtime");
    }
    if (stage_runtime_->tensor_parallel) {
        run_tensor_parallel(x, ph, tap);
        return;
    }
    run_staged(x, ph, tap);
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_ + static_cast<std::int32_t>(cache_position_shift_), s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0          = 0;
    bool token_left = false;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto prompt_t0_i32 = static_cast<std::int32_t>(prompt_t0);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0_i32);
                const auto end = std::lower_bound(begin, scatter.end(), prompt_t0_i32 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] = begin[i] - prompt_t0_i32;
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : ((rope_delta_ != 0 || cache_position_shift_ != 0) ? 1 : 0);
            const bool overlay_staging   = !vision_chunk.host_embeddings.empty();
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()), overlay_staging);
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0 - static_cast<int>(cache_position_shift_), s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (cache_position_shift_ != 0) {
                rope_positions = roots.rope_positions;
                ops::fill_i32_positions(rope_positions, base_i + t0 + rope_delta_, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len) - cache_position_shift_;
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{
                .min_visible_keys   = visible,
                .max_visible_keys   = visible,
                .fast_prompt_kernel = fast_prefill_kernel_,
                .small_prefill      = true};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            embed_tokens(ids_device, *embed_, parameters_.text.token_embedding_signs, x, s);
            capture_attention_tokens(ids_device);
            const auto visual_count = static_cast<std::int32_t>(local_scatter_indices.size());
            if (overlay_staging) {
                // Stage the shifted MTP column even when this unit contains only
                // the preceding Vision marker and no Main visual embedding.
                const std::size_t column_bytes = static_cast<std::size_t>(
                    roots.visual_embeddings.ne[0]) * sizeof(std::uint16_t);
                const auto merged = static_cast<std::int32_t>(vision_chunk.control->merged_count);
                const std::int32_t staged_columns =
                    std::min<std::int32_t>(visual_count + 1, merged - visual_begin);
                const std::size_t offset = static_cast<std::size_t>(visual_begin) * column_bytes;
                const std::size_t bytes = static_cast<std::size_t>(staged_columns) * column_bytes;
                if (staged_columns < visual_count || offset + bytes > vision_chunk.host_embeddings.size()) {
                    throw std::logic_error("Vision chunk columns exceed the item embeddings");
                }
                CUDA_CHECK(cudaMemcpyAsync(roots.visual_embeddings.data,
                                           vision_chunk.host_embeddings.data() + offset, bytes,
                                           cudaMemcpyHostToDevice, s));
            }
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = overlay_staging
                    ? roots.visual_embeddings.slice(1, 0, visual_count)
                    : vision_chunk.embeddings.slice(1, visual_begin, visual_count);
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            if (statistics_ != nullptr) {
                // Logical token indexing is independent of all three MRoPE axes
                // and of the compact KV address used by attention.
                auto origin = statistics_->ranks.front().prefill_origin;
                ops::set_i32_scalar(origin, base_i + t0, s);
            }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                // The token's logprob records, gathered before sampling adds it to the penalty
                // counts; a device no-op unless the request asked for them.
                if (sampling_config_ != nullptr && io_.logprob_active.data != nullptr) {
                    Tensor ids = io_.logprob_ids.slice(1, 0, 1).view({ops::kLogprobTopK, 1, 1});
                    Tensor values =
                        io_.logprob_values.slice(1, 0, 1).view({ops::kLogprobTopK, 1, 1});
                    Tensor lse = io_.logprob_lse.slice(0, 0, 1).view({1, 1});
                    ops::logprob_topk(logits.view({logits.ne[0], 1, 1}), sampling_config_, nullptr,
                                      dimension(parameters_.model.resources().public_token_count),
                                      ids, values, lse, io_.logprob_active, work_, s);
                }
                if (readout_count_ > 0) {
                    ops::readout_logprobs(logits, io_.readout_ids, readout_count_,
                                          dimension(parameters_.model.resources().public_token_count),
                                          io_.readout_values, s);
                }
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count),
                                sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count), s);
                }
                if (first_token_hook_) {
                    first_token_hook_();
                    token_left = true;
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings =
                        work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                    embed_tokens(mtp_ids, *embed_, parameters_.text.token_embedding_signs,
                                 mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            if (overlay_staging) {
                                qwen3_5::detail::scatter_shifted_visual_embeddings(
                                    mtp_input_embeddings, roots.visual_embeddings, overlap,
                                    shifted_indices, s, static_cast<std::size_t>(visual_begin));
                            } else {
                                qwen3_5::detail::scatter_shifted_visual_embeddings(
                                    mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                    shifted_indices, s);
                            }
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T - static_cast<int>(cache_position_shift_), s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i) - cache_position_shift_;
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    // A token that has left is waited for by the caller; the drafter's work behind it is not.
    if (!token_left) {
        timing.begin_wait();
        ctx_.synchronize();
        timing.end_wait();
    }
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_final_batch(std::span<const PrefillSegment> segments,
                                                    PrefillBatchHost host) {
    runtime::ExecutionTimingRecorder timing;
    if (segments.empty() || segments.size() > kMaximumPrefillSegments || host.tokens == nullptr ||
        host.top_ids == nullptr || host.top_values == nullptr || host.readout == nullptr) {
        throw std::invalid_argument("batched prefill needs one to 64 segments and host outputs");
    }
    if (prefill_hidden_.data == nullptr || mtp_enabled() || parameters_.text.split_execution()) {
        throw std::logic_error(
            "batched prefill runs without drafter work or stages and needs prefill hidden");
    }
    cudaStream_t s          = ctx_.stream;
    const auto count        = static_cast<std::int32_t>(segments.size());
    constexpr auto readouts = static_cast<std::int32_t>(ops::kReadoutLogprobsMaximum);
    std::vector<SegmentBinding> bindings(segments.size());
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> positions_host;
    std::vector<std::int32_t> rows_host(segments.size());
    // Output rows: the segments that read their last column, in segment order.
    std::vector<std::int32_t> output_segments;
    std::vector<std::int32_t> sample_rows;
    std::vector<std::int32_t> frontiers_host;
    std::vector<std::int32_t> readout_ids_host;
    bool any_readout     = false;
    std::int32_t columns = 0;
    for (std::int32_t i = 0; i < count; ++i) {
        const PrefillSegment& segment = segments[static_cast<std::size_t>(i)];
        const std::uint64_t frontier  = std::uint64_t(segment.begin) + segment.ids.size();
        if (segment.ids.empty() ||
            (segment.output != SegmentOutput::None && segment.sampling == nullptr) ||
            segment.readout.size() > static_cast<std::size_t>(readouts) ||
            (segment.output == SegmentOutput::None && !segment.readout.empty()) ||
            (segment.state_destination_slot < 0 && segment.output == SegmentOutput::Sample) ||
            frontier > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            std::uint64_t(columns) + segment.ids.size() > prefill_chunk_) {
            throw std::invalid_argument("batched prefill segment is outside the pass");
        }
        const auto length = static_cast<std::int32_t>(segment.ids.size());
        ids.insert(ids.end(), segment.ids.begin(), segment.ids.end());
        for (std::int32_t t = 0; t < length; ++t) {
            positions_host.push_back(static_cast<std::int32_t>(segment.begin) + t);
        }
        rows_host[static_cast<std::size_t>(i)] = segment.kv_table_row;
        if (segment.output != SegmentOutput::None) {
            const auto row = static_cast<std::int32_t>(output_segments.size());
            output_segments.push_back(i);
            frontiers_host.push_back(static_cast<std::int32_t>(frontier));
            readout_ids_host.resize(readout_ids_host.size() + static_cast<std::size_t>(readouts),
                                    0);
            std::copy(segment.readout.begin(), segment.readout.end(),
                      readout_ids_host.end() - readouts);
            any_readout = any_readout || !segment.readout.empty();
            if (segment.output == SegmentOutput::Sample) { sample_rows.push_back(row); }
        }
        const auto visible = static_cast<std::uint32_t>(frontier);
        bindings[static_cast<std::size_t>(i)] = SegmentBinding{
            .column                 = columns,
            .columns                = length,
            .envelope               = {.min_visible_keys   = visible,
                                       .max_visible_keys   = visible,
                                       .fast_prompt_kernel = fast_prefill_kernel_,
                                       .small_prefill      = true},
            .state_source_slot      = segment.state_source_slot,
            .state_destination_slot = segment.state_destination_slot,
        };
        columns += length;
    }
    if (sample_rows.size() > kMaximumConcurrency) {
        throw std::invalid_argument("batched prefill samples more rows than lanes");
    }

    work_.reset();
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(columns));
    rope_delta_ = 0;
    ops::set_i32_scalar(io_.rope_delta, 0, s);
    const auto roots = workspace::text_prefill_roots(work_, config_, columns, 0, 0);
    Tensor ids_device = roots.ids;
    copy_i32(ids.data(), ids_device, s);
    Tensor positions = roots.positions;
    copy_i32(positions_host.data(), positions, s);
    Tensor rows = work_.alloc(DType::I32, {count});
    copy_i32(rows_host.data(), rows, s);
    for (std::int32_t i = 0; i < count; ++i) {
        SegmentBinding& binding = bindings[static_cast<std::size_t>(i)];
        binding.positions       = positions.slice(0, binding.column, binding.columns);
        binding.kv_table_row    = rows.slice(0, i, 1);
    }

    Tensor x = roots.residual;
    {
        // Token-wise work reads the widest segment's envelope; attention uses each segment's own.
        const auto widest = std::max_element(
            bindings.begin(), bindings.end(), [](const SegmentBinding& a, const SegmentBinding& b) {
                return a.envelope.max_visible_keys < b.envelope.max_visible_keys;
            });
        ScopedPositions scoped_cache(active_cache_positions_, positions);
        ScopedPositions scoped_rope(active_rope_positions_, positions);
        ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, widest->envelope);
        ScopedValue<std::span<const SegmentBinding>> scoped_segments(
            active_segments_, std::span<const SegmentBinding>(bindings.data(), bindings.size()));
        embed_tokens(ids_device, *embed_, parameters_.text.token_embedding_signs, x, s);
        run_layers(x, Phase::Prefill);
    }

    Tensor xf = matrix_window(prefill_hidden_, columns);
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);
    const auto outputs = static_cast<std::int32_t>(output_segments.size());
    if (outputs != 0) {
        Tensor last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), outputs});
        for (std::int32_t row = 0; row < outputs; ++row) {
            const SegmentBinding& binding =
                bindings[static_cast<std::size_t>(output_segments[static_cast<std::size_t>(row)])];
            const Tensor source = xf.slice(1, binding.column + binding.columns - 1, 1);
            CUDA_CHECK(cudaMemcpyAsync(last.slice(1, row, 1).data, source.data, source.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
        }
        const std::int32_t physical_rows = io_.logits.ne[0];
        const auto token_domain = dimension(parameters_.model.resources().public_token_count);
        Tensor logits           = work_.alloc(DType::BF16, {physical_rows, outputs});
        project(last, *lm_head_, logits, work_, s);

        // The logprob gather reads one contiguous sampling record per row.
        const DeviceSpan configs_span =
            work_.alloc_bytes(sizeof(ops::SamplingConfig) * static_cast<std::size_t>(outputs));
        auto* configs = static_cast<ops::SamplingConfig*>(configs_span.data);
        for (std::int32_t row = 0; row < outputs; ++row) {
            const PrefillSegment& segment =
                segments[static_cast<std::size_t>(output_segments[static_cast<std::size_t>(row)])];
            CUDA_CHECK(cudaMemcpyAsync(configs + row, segment.sampling, sizeof(ops::SamplingConfig),
                                       cudaMemcpyDeviceToDevice, s));
        }
        // The rows' logprob records, gathered before sampling adds a sampled token to its
        // penalty counts; a device no-op unless logprobs are armed.
        Tensor top_ids    = work_.alloc(DType::I32, {ops::kLogprobTopK, 1, outputs});
        Tensor top_values = work_.alloc(DType::FP32, {ops::kLogprobTopK, 1, outputs});
        Tensor lse        = work_.alloc(DType::FP32, {1, outputs});
        if (io_.logprob_active.data != nullptr) {
            ops::logprob_topk(logits.view({physical_rows, 1, outputs}), configs, nullptr,
                              token_domain, top_ids, top_values, lse, io_.logprob_active, work_, s);
        }
        if (any_readout) {
            Tensor readout_ids = work_.alloc(DType::I32, {readouts, outputs});
            copy_i32(readout_ids_host.data(), readout_ids, s);
            Tensor readout_values = work_.alloc(DType::FP32, {readouts, outputs});
            for (std::int32_t row = 0; row < outputs; ++row) {
                const auto requested = static_cast<std::int32_t>(
                    segments[static_cast<std::size_t>(
                                 output_segments[static_cast<std::size_t>(row)])]
                        .readout.size());
                if (requested == 0) { continue; }
                Tensor values = readout_values.slice(1, row, 1).view({readouts});
                ops::readout_logprobs(logits.slice(1, row, 1),
                                      readout_ids.slice(1, row, 1).view({readouts}), requested,
                                      token_domain, values, s);
            }
            CUDA_CHECK(cudaMemcpyAsync(host.readout, readout_values.data, readout_values.bytes(),
                                       cudaMemcpyDeviceToHost, s));
        }
        // Sampled rows draw from their own logits, compacted so the sampler sees lanes only.
        Tensor tokens = work_.alloc(DType::I32, {outputs});
        if (!sample_rows.empty()) {
            const auto sampled   = static_cast<std::int32_t>(sample_rows.size());
            Tensor sample_logits = work_.alloc(DType::BF16, {physical_rows, sampled});
            const DeviceSpan sample_configs_span =
                work_.alloc_bytes(sizeof(ops::SamplingConfig) * static_cast<std::size_t>(sampled));
            auto* sample_configs = static_cast<ops::SamplingConfig*>(sample_configs_span.data);
            std::vector<std::int32_t> sample_positions_host;
            for (std::int32_t k = 0; k < sampled; ++k) {
                const std::int32_t row = sample_rows[static_cast<std::size_t>(k)];
                CUDA_CHECK(
                    cudaMemcpyAsync(sample_logits.slice(1, k, 1).data, logits.slice(1, row, 1).data,
                                    logits.slice(1, row, 1).bytes(), cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(sample_configs + k, configs + row,
                                           sizeof(ops::SamplingConfig), cudaMemcpyDeviceToDevice,
                                           s));
                sample_positions_host.push_back(frontiers_host[static_cast<std::size_t>(row)]);
            }
            Tensor sample_positions = work_.alloc(DType::I32, {sampled});
            copy_i32(sample_positions_host.data(), sample_positions, s);
            Tensor sampled_tokens = work_.alloc(DType::I32, {sampled});
            ops::sample(sample_logits, sampled_tokens, token_domain, sample_configs,
                        sample_positions, ops::kSamplePurposePrefill, work_, s);
            for (std::int32_t k = 0; k < sampled; ++k) {
                const std::int32_t row = sample_rows[static_cast<std::size_t>(k)];
                CUDA_CHECK(cudaMemcpyAsync(tokens.slice(0, row, 1).data,
                                           sampled_tokens.slice(0, k, 1).data, sizeof(std::int32_t),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }
        CUDA_CHECK(
            cudaMemcpyAsync(host.tokens, tokens.data, tokens.bytes(), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(host.top_ids, top_ids.data, top_ids.bytes(),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(host.top_values, top_values.data, top_values.bytes(),
                                   cudaMemcpyDeviceToHost, s));
    }

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    // A read's token is the most likely one, the first of its logprob record (the lower id on a
    // tie, as greedy sampling picks).
    for (std::int32_t row = 0; row < outputs; ++row) {
        if (segments[static_cast<std::size_t>(output_segments[static_cast<std::size_t>(row)])]
                .output == SegmentOutput::Read) {
            host.tokens[row] = host.top_ids[static_cast<std::size_t>(row) * ops::kLogprobTopK];
        }
    }
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(columns),
                              .finalized        = outputs != 0,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution
