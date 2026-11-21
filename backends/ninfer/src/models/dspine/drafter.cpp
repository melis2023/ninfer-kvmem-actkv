#include "models/dspine/drafter.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/weight.h"
#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dspine.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::dspine {
namespace {

constexpr std::size_t kSlack = 256; // per arena allocation

std::int32_t dim(std::uint64_t value) { return static_cast<std::int32_t>(value); }

std::size_t plane(std::uint64_t rows, std::uint64_t columns, std::size_t element) {
    return rows * columns * element + kSlack;
}

void project(const ops::SingleProjectionWeight& w, const Tensor& x, Tensor& out,
             WorkspaceArena& work, cudaStream_t stream) {
    ops::linear(x, w.weight, out, w.policy, work, stream);
}

std::size_t linear_bytes(const ops::SingleProjectionWeight& w, std::int32_t columns) {
    return ops::linear_workspace_capacity_bytes(w.weight.qtype, w.weight.n, w.weight.k, w.policy,
                                                1, std::max(columns, 1));
}

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("DSpine drafter: " + message); }
}

bool is_window(std::uint32_t window) {
    return window == 2048 || window == 4096 || window == 8192;
}

// The registered fused routes a layer takes where its weights fit them (the Ops' headers list
// the forms); any other format or geometry runs the separate projections.
bool fused_query_key_value(const std::optional<ops::SingleProjectionWeight>& w) {
    return w && w->weight.qtype == QType::Q8_G32_FP16 && w->weight.n == 6144 &&
           (w->weight.k == 5120 || w->weight.k == 2048);
}

bool fused_swiglu(const std::optional<ops::SingleProjectionWeight>& w) {
    return w && w->weight.qtype == QType::Q8_G32_FP16 &&
           ((w->weight.n == 34816 && w->weight.k == 5120) ||
            (w->weight.n == 12288 && w->weight.k == 2048));
}

bool fused_add(const ops::SingleProjectionWeight& w) {
    const auto n = w.weight.n, k = w.weight.k;
    switch (w.weight.qtype) {
    case QType::Q8_G32_FP16:
        return (n == 2048 && (k == 4096 || k == 6144)) || (n == 5120 && (k == 6144 || k == 17408));
    case QType::BF16:
        return n == 5120 && k == 6144;
    default:
        return false;
    }
}

// The pair form of rmsnorm_rope: q [128,32,W,B] and k [128,8,W,B], W 2..16, B 1..8.
bool fused_norm_rope(const Config& c, std::int32_t width, std::int32_t batch) {
    return ops::rmsnorm_rope_constants_match(c.rope_theta, c.rms_norm_eps) && c.head_dim == 128 &&
           c.num_attention_heads == 32 && c.num_key_value_heads == 8 && width >= 2 &&
           width <= 16 && batch >= 1 && batch <= 8;
}

// The single-K form: k [128,8,T], T 1..2048.
bool fused_key_norm_rope(const Config& c, std::int32_t columns) {
    return ops::rmsnorm_rope_constants_match(c.rope_theta, c.rms_norm_eps) && c.head_dim == 128 &&
           c.num_key_value_heads == 8 && columns >= 1 && columns <= 2048;
}

} // namespace

Drafter::Drafter(const Model& model, Target target, Tensor head_write, std::uint32_t window,
                 std::uint32_t max_context, std::int32_t max_batch, std::int32_t max_width)
    : model_(model), target_(std::move(target)), head_write_(head_write), window_(window),
      max_context_(max_context), max_batch_(max_batch), max_width_(max_width) {
    const Config& c = model_.config();
    check_target(c, target_.hidden, target_.vocab, target_.layers, "target");
    require(target_.domain >= ops::kDSpineCandidates &&
                std::uint32_t(target_.domain) <= target_.vocab,
            "the candidates' vocabulary must hold 16 ids and lie within the head's rows");
    require(target_.embed != nullptr && target_.head != nullptr, "the target's rows and head");
    require(target_.logits == DType::BF16 || target_.logits == DType::FP32,
            "the head writes BF16 or FP32 logits");
    require(head_write_.dtype == DType::BF16 && head_write_.ne[0] == dim(c.rank) &&
                head_write_.ne[1] == dim(c.vocab_size) &&
                head_write_.numel() == std::int64_t(c.rank) * c.vocab_size,
            "the head-write table must be BF16 [rank, vocab]");
    require(is_window(window_) && window_ <= c.sliding_window,
            "the ring's window must be 2048, 4096 or 8192, at most the trained one");
    require(max_batch_ >= 1 && max_batch_ <= 8, "at most eight rows a call");
    require(max_width_ >= 2 && std::uint32_t(max_width_) <= c.block_size,
            "a block of 2..block_size columns");
    require(model_.operands().layers.size() == c.num_hidden_layers, "one operand set per layer");
}

void Drafter::prepare(WorkspaceArena& work, cudaStream_t stream) const {
    const Config& c   = model_.config();
    const Operands& p = model_.operands();
    const std::int32_t h = dim(c.hidden_size), rank = dim(c.rank), v = dim(c.vocab_size);
    const std::int32_t chunk = prepare_columns(model_, max_batch_, max_width_);
    Tensor table             = head_write_;
    for (std::int32_t first = 0; first < rank; first += chunk) {
        auto scope        = work.scope();
        const std::int32_t n = std::min(chunk, rank - first);
        // Columns [first, first + n) of the last write, each through the final norm's weight.
        const Tensor columns = p.head_write_columns.slice(1, first, n);
        Tensor scaled        = work.alloc(DType::BF16, {h, n});
        ops::dspine_scale_rows(columns, p.final_norm, scaled, stream);
        Tensor logits = work.alloc(target_.logits, {v, n});
        target_.head(scaled, logits, work, stream);
        ops::dspine_store_head_write(logits, first, table, stream);
    }
}

void Drafter::materialize(const ContextCall& call, std::span<const CyclicKVCacheLayerView> ring,
                          WorkspaceArena& work, cudaStream_t stream) const {
    const Config& c   = model_.config();
    const Operands& p = model_.operands();
    const std::int32_t h = dim(c.hidden_size), w = call.features.ne[1], b = call.features.ne[2];
    const std::int32_t d = dim(c.head_dim), kv = dim(c.num_key_value_heads);
    // A commit takes up to eight rows whatever the proposals' batch (a decode step's sequences
    // commit one column each); the caller's workspace bounds the call.
    require(call.features.dtype == DType::BF16 &&
                call.features.ne[0] == dim(std::uint64_t(c.taps()) * c.hidden_size) && w > 0 &&
                b > 0 && b <= 8 && call.positions.dtype == DType::I32 &&
                call.positions.ne[0] == w && call.positions.ne[1] == b &&
                call.counts.dtype == DType::I32 && call.counts.ne[0] == b &&
                call.lanes.dtype == DType::I32 && call.lanes.ne[0] == b &&
                call.max_count <= std::uint32_t(w) && call.min_count <= call.max_count,
            "the context call's geometry");
    require(ring.size() == p.layers.size(), "one ring layer per drafter layer");
    for (const auto& layer : ring) { require(layer.capacity == window_, "the ring's capacity"); }
    nvtx::ScopedRange range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(w) * b);
    auto scope = work.scope();
    // A single row's columns past max_count are inert. A row commits at most the ring: a single
    // row of known count wider than it stores its last window of columns, which every later block
    // sees in full (the slots it overwrites are then all dead).
    std::int32_t first = 0, width = b == 1 ? std::min(w, dim(call.max_count)) : w;
    if (width == 0) { return; }
    Tensor counts = call.counts;
    ops::KVCacheAppendPrefixExecutionEnvelope envelope{call.min_count, call.max_count};
    if (call.max_count > window_) {
        require(b == 1 && call.min_count == call.max_count,
                "a commit wider than the ring must be one row of known count");
        first    = width - dim(window_);
        width    = dim(window_);
        counts   = work.alloc(DType::I32, {1});
        ops::set_i32_scalar(counts, width, stream);
        envelope = {window_, window_};
    }
    const std::int32_t columns = width * b;
    const Tensor features =
        width == w ? call.features : call.features.slice(1, first, width);
    const Tensor positions =
        width == w ? call.positions : call.positions.slice(0, first, width);
    Tensor projected = work.alloc(DType::BF16, {h, columns});
    project(p.feature_projection, features.view({features.ne[0], columns}), projected, work,
            stream);
    Tensor context = work.alloc(DType::BF16, {h, columns});
    ops::rmsnorm(projected, p.context_norm, c.rms_norm_eps, false, context, stream);
    const Tensor flat_positions = positions.view({columns});
    const bool norm_rope        = fused_key_norm_rope(c, columns);
    for (std::size_t l = 0; l < p.layers.size(); ++l) {
        auto layer_scope  = work.scope();
        const auto& layer = p.layers[l];
        Tensor value      = work.alloc(DType::BF16, {d, kv, width, b});
        Tensor value_flat = value.view({d * kv, columns});
        Tensor key        = work.alloc(DType::BF16, {d, kv, width, b});
        Tensor key_raw =
            norm_rope ? key.view({d * kv, columns}) : work.alloc(DType::BF16, {d * kv, columns});
        if (ops::linear_pair_admits(layer.context_key.weight, layer.context_value.weight)) {
            ops::linear_pair(context, layer.context_key.weight, layer.context_value.weight,
                             key_raw, value_flat, stream);
        } else {
            project(layer.context_key, context, key_raw, work, stream);
            project(layer.context_value, context, value_flat, work, stream);
        }
        Tensor rotated = key.view({d, kv, columns});
        if (norm_rope) {
            ops::rmsnorm_rope(flat_positions, layer.key_norm, rotated, stream);
        } else {
            Tensor key_rows = key.view({d, kv * columns});
            ops::rmsnorm(key_raw.view({d, kv * columns}), layer.key_norm, c.rms_norm_eps, false,
                         key_rows, stream);
            ops::rope(flat_positions, d, c.rope_theta, rotated, stream);
        }
        ops::kv_cache_append_prefix(key, value, positions.view({width, b}), counts, call.lanes,
                                    envelope, ring[l], stream);
    }
}

void Drafter::propose(const ProposalCall& call, std::span<const CyclicKVCacheLayerView> ring,
                      WorkspaceArena& work, cudaStream_t stream) const {
    const Config& c   = model_.config();
    const Operands& p = model_.operands();
    const std::int32_t h = dim(c.hidden_size), w = call.width, b = call.batch, k = w - 1;
    const std::int32_t columns = w * b, masks = k * b;
    const std::int32_t d = dim(c.head_dim), hq = dim(c.num_attention_heads),
                       kv = dim(c.num_key_value_heads);
    const std::int32_t inner = dim(c.intermediate_size), code = dim(c.code_size),
                       rank = dim(c.rank), v = dim(c.vocab_size);
    const float eps = c.rms_norm_eps;
    require(w >= 2 && w <= max_width_ && b >= 1 && b <= max_batch_ && call.configs != nullptr,
            "the proposal's geometry");
    require(ring.size() == p.layers.size(), "one ring layer per drafter layer");
    for (const auto& layer : ring) { require(layer.capacity == window_, "the ring's capacity"); }
    nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash,
                            static_cast<std::uint64_t>(columns));
    auto scope       = work.scope();
    Tensor ids       = work.alloc(DType::I32, {w, b});
    Tensor positions = work.alloc(DType::I32, {w, b});
    ops::prepare_masked_block(call.anchors, call.frontiers, call.valid_columns,
                              dim(c.mask_token_id), ids, positions, stream);
    const Tensor flat_positions = positions.view({columns});
    // The residual stream, the next layer's normalized input, and the last write's state the
    // refinement reads: its scale, packed rows, message and the final norm's denominator.
    Tensor residual    = work.alloc(DType::BF16, {h, w, b});
    Tensor normed      = work.alloc(DType::BF16, {h, w, b});
    Tensor rms         = work.alloc(DType::FP32, {w, b});
    Tensor denominator = work.alloc(DType::FP32, {w, b});
    Tensor packed      = work.alloc(DType::BF16, {3 * rank, w, b});
    Tensor message     = work.alloc(DType::BF16, {rank, w, b});
    Tensor residual_flat = residual.view({h, columns});
    Tensor normed_flat   = normed.view({h, columns});
    target_.embed(ids.view({columns}), residual_flat, stream);
    ops::rmsnorm(residual_flat, p.layers.front().input_norm, eps, false, normed_flat, stream);
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{0, max_context_};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d)));
    const bool norm_rope = fused_norm_rope(c, w, b);
    for (std::size_t l = 0; l < p.layers.size(); ++l) {
        const auto& layer = p.layers[l];
        const bool last   = l + 1 == p.layers.size();
        nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash, l);
        {
            // Attention over the normalized input, causal inside the block.
            auto branch   = work.scope();
            Tensor q      = work.alloc(DType::BF16, {d, hq, w, b});
            Tensor key    = work.alloc(DType::BF16, {d, kv, w, b});
            Tensor value  = work.alloc(DType::BF16, {d, kv, w, b});
            Tensor v_flat = value.view({d * kv, columns});
            // The fused norm and rotation work in place on the projections' outputs.
            Tensor q_raw = norm_rope ? q.view({d * hq, columns})
                                     : work.alloc(DType::BF16, {d * hq, columns});
            Tensor k_raw = norm_rope ? key.view({d * kv, columns})
                                     : work.alloc(DType::BF16, {d * kv, columns});
            if (fused_query_key_value(layer.query_key_value)) {
                ops::attn_input_proj(normed_flat, layer.query_key_value->weight, q_raw, k_raw,
                                     v_flat, stream);
            } else {
                project(layer.query, normed_flat, q_raw, work, stream);
                project(layer.key, normed_flat, k_raw, work, stream);
                project(layer.value, normed_flat, v_flat, work, stream);
            }
            if (norm_rope) {
                ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, q, key, stream);
            } else {
                Tensor q_rows = q.view({d, hq * columns});
                Tensor k_rows = key.view({d, kv * columns});
                ops::rmsnorm(q_raw.view({d, hq * columns}), layer.query_norm, eps, false, q_rows,
                             stream);
                ops::rmsnorm(k_raw.view({d, kv * columns}), layer.key_norm, eps, false, k_rows,
                             stream);
                Tensor q_heads = q.view({d, hq, columns});
                Tensor k_heads = key.view({d, kv, columns});
                ops::rope(flat_positions, d, c.rope_theta, q_heads, k_heads, stream);
            }
            Tensor attended = work.alloc(DType::BF16, {d, hq, w, b});
            // The bridge's rows are its own block, so a query row sees the rows at or before it;
            // no KVMem floor applies here (min_positions stays null).
            ops::sliding_window_attention(q, key, value, positions, call.valid_columns,
                                          call.lanes, Tensor{}, {d, hq, kv}, window_,
                                          scale, ring[l], envelope, work, attended, stream,
                                          ops::SlidingWindowMask::Causal);
            const Tensor attended_flat = attended.view({d * hq, columns});
            if (fused_add(layer.output)) {
                ops::linear_add(attended_flat, layer.output.weight, residual_flat,
                                layer.output.policy, work, stream);
            } else {
                Tensor output = work.alloc(DType::BF16, {h, columns});
                project(layer.output, attended_flat, output, work, stream);
                ops::residual_add(output, residual_flat, stream);
            }
        }
        {
            // The MLP over the post-attention norm; the residual then holds the bridge's raw row.
            auto branch    = work.scope();
            Tensor x       = work.alloc(DType::BF16, {h, columns});
            Tensor product = work.alloc(DType::BF16, {inner, columns});
            ops::rmsnorm(residual_flat, layer.post_attention_norm, eps, false, x, stream);
            if (fused_swiglu(layer.gate_up)) {
                ops::linear_swiglu(x, layer.gate_up->weight, product, layer.gate_up->policy, work,
                                   stream);
            } else {
                Tensor gate = work.alloc(DType::BF16, {inner, columns});
                Tensor up   = work.alloc(DType::BF16, {inner, columns});
                project(layer.gate, x, gate, work, stream);
                project(layer.up, x, up, work, stream);
                ops::silu_mul(gate, up, product, stream);
            }
            if (fused_add(layer.down)) {
                ops::linear_add(product, layer.down.weight, residual_flat, layer.down.policy, work,
                                stream);
            } else {
                Tensor down = work.alloc(DType::BF16, {h, columns});
                project(layer.down, product, down, work, stream);
                ops::residual_add(down, residual_flat, stream);
            }
        }
        {
            // The bridge's write, and the next layer's (or the final) norm of its output.
            auto branch       = work.scope();
            Tensor normalized = work.alloc(DType::BF16, {h, columns});
            Tensor rms_flat   = rms.view({columns});
            ops::dspine_bridge_read(residual_flat, c.bridge_eps, normalized, rms_flat, stream);
            Tensor codes = work.alloc(DType::BF16, {code, columns});
            project(layer.read, normalized, codes, work, stream);
            ops::dspine_bridge_codes(codes, c.bridge_eps, codes, stream);
            Tensor packed_flat = packed.view({3 * rank, columns});
            project(p.bridge_projection, codes, packed_flat, work, stream);
            ops::dspine_bridge_message(packed, call.anchors, p.sender_table, p.gate_bias,
                                       message, stream);
            Tensor delta = work.alloc(DType::BF16, {h, w, b});
            Tensor delta_flat = delta.view({h, columns});
            project(layer.write, message.view({rank, columns}), delta_flat, work, stream);
            const Tensor& next = last ? p.final_norm : p.layers[l + 1].input_norm;
            ops::dspine_bridge_finish(residual, rms, delta, next, eps, residual, normed,
                                      last ? &denominator : nullptr, stream);
        }
    }
    // The mask columns' final hidden, packed row after row: the head's candidates, then the
    // refinement's walk over them.
    Tensor hidden = work.alloc(DType::BF16, {h, masks});
    const std::size_t row = std::size_t(h) * dtype_size(DType::BF16);
    CUDA_CHECK(cudaMemcpy2DAsync(hidden.data, std::size_t(k) * row,
                                 static_cast<const std::byte*>(normed.data) + row,
                                 std::size_t(w) * row, std::size_t(k) * row, std::size_t(b),
                                 cudaMemcpyDeviceToDevice, stream));
    Tensor scores = work.alloc(DType::FP32, {ops::kDSpineCandidates, k, b});
    if (target_.head_topk) {
        auto head_scope   = work.scope();
        Tensor candidates = call.candidates.view({ops::kDSpineCandidates, masks});
        Tensor flat       = scores.view({ops::kDSpineCandidates, masks});
        target_.head_topk(hidden, candidates, flat, work, stream);
    } else {
        auto head_scope   = work.scope();
        Tensor logits     = work.alloc(target_.logits, {v, masks});
        target_.head(hidden, logits, work, stream);
        Tensor candidates = call.candidates;
        ops::candidate_topk(logits, target_.domain, candidates, scores, work, stream);
    }
    Tensor drafts     = call.drafts;
    Tensor proposal_q = call.proposal_q;
    Tensor confidence = call.confidence;
    ops::dspine_path(call.candidates, scores, denominator, rms, packed, message, head_write_,
                     p.sender_table, p.gate_bias, call.anchors, call.frontiers, call.configs,
                     drafts, proposal_q, confidence.data != nullptr ? &confidence : nullptr, work,
                     stream);
}

std::int32_t Drafter::prepare_columns(const Model& model, std::int32_t max_batch,
                                      std::int32_t max_width) noexcept {
    return std::max(std::min(dim(model.config().rank), (max_width - 1) * max_batch), 1);
}

std::size_t Drafter::prepare_workspace_bytes(const Model& model, DType logits,
                                             std::int32_t max_batch, std::int32_t max_width,
                                             std::size_t head_bytes) {
    const Config& c          = model.config();
    const std::uint64_t cols = std::uint64_t(prepare_columns(model, max_batch, max_width));
    return plane(c.hidden_size, cols, 2) + plane(c.vocab_size, cols, dtype_size(logits)) +
           head_bytes + kSlack;
}

std::size_t Drafter::context_workspace_bytes(const Model& model, std::uint32_t window,
                                             std::int32_t width, std::int32_t batch) {
    const Config& c   = model.config();
    const Operands& p = model.operands();
    // A single row wider than the ring keeps only its last window of columns.
    const std::uint64_t columns =
        batch == 1 ? std::min<std::uint64_t>(std::uint64_t(std::max(width, 1)), window)
                   : std::uint64_t(std::max(width, 1)) * std::uint64_t(batch);
    std::size_t layer_linear = 0;
    for (const auto& layer : p.layers) {
        layer_linear = std::max({layer_linear, linear_bytes(layer.context_key, dim(columns)),
                                 linear_bytes(layer.context_value, dim(columns))});
    }
    const std::size_t layer = 3 * plane(c.key_width(), columns, 2) + layer_linear;
    return plane(1, 1, 4) + 2 * plane(c.hidden_size, columns, 2) +
           std::max(linear_bytes(p.feature_projection, dim(columns)), layer);
}

std::size_t Drafter::proposal_workspace_bytes(const Model& model, DType logits,
                                              std::uint32_t window, std::uint32_t max_context,
                                              std::int32_t width, std::int32_t batch,
                                              std::size_t head_bytes) {
    const Config& c   = model.config();
    const Operands& p = model.operands();
    const std::uint64_t h = c.hidden_size, q = c.query_width(), kv = c.key_width();
    const std::uint64_t inner = c.intermediate_size, code = c.code_size, rank = c.rank;
    const std::uint64_t cells = std::uint64_t(width) * batch;
    const std::uint64_t masks = std::uint64_t(width - 1) * batch;
    std::size_t projections   = linear_bytes(p.bridge_projection, dim(cells));
    for (const auto& layer : p.layers) {
        for (const auto* weight : {&layer.query, &layer.key, &layer.value, &layer.output,
                                   &layer.gate, &layer.up, &layer.down, &layer.read,
                                   &layer.write}) {
            projections = std::max(projections, linear_bytes(*weight, dim(cells)));
        }
        if (fused_swiglu(layer.gate_up)) {
            const Weight& gu = layer.gate_up->weight;
            projections      = std::max(projections, ops::linear_swiglu_workspace_capacity_bytes(
                                                          gu.qtype, gu.n, gu.k,
                                                          layer.gate_up->policy, 1, dim(cells)));
        }
        for (const auto* weight : {&layer.output, &layer.down}) {
            if (fused_add(*weight)) {
                projections = std::max(projections, ops::linear_add_workspace_capacity_bytes(
                                                        weight->weight.qtype, weight->weight.n,
                                                        weight->weight.k, weight->policy, 1,
                                                        dim(cells)));
            }
        }
    }
    const std::size_t attend = ops::sliding_window_attention_workspace_capacity_bytes(
        {dim(c.head_dim), dim(c.num_attention_heads), dim(c.num_key_value_heads)}, window,
        {0, max_context}, 1, width, batch);
    const std::size_t base = 4 * plane(1, cells, 4) + 2 * plane(h, cells, 2) +
                             plane(3 * rank, cells, 2) + plane(rank, cells, 2);
    const std::size_t attention =
        3 * plane(q, cells, 2) + 3 * plane(kv, cells, 2) + plane(h, cells, 2) +
        std::max(projections, attend);
    const std::size_t mlp    = 2 * plane(h, cells, 2) + 3 * plane(inner, cells, 2) + projections;
    const std::size_t bridge = 2 * plane(h, cells, 2) + plane(code, cells, 2) + projections;
    std::size_t path         = 0;
    for (std::int32_t steps = 1; steps < width; ++steps) {
        path = std::max(path, ops::dspine_path_workspace_capacity_bytes(steps, batch));
    }
    const std::size_t tail =
        plane(h, masks, 2) + plane(ops::kDSpineCandidates, masks, 4) +
        std::max(plane(c.vocab_size, masks, dtype_size(logits)) +
                     std::max(head_bytes, ops::candidate_topk_workspace_capacity_bytes(dim(masks))),
                 path);
    return base + std::max({attention, mlp, bridge, tail}) + kSlack;
}

} // namespace ninfer::models::dspine
