#include "models/gemma4/vision.h"

#include "core/weight_view.h"
#include "models/gemma4/vision_kernels.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {
namespace {

namespace vk = vision_kernels;

constexpr std::size_t kAlign = 256;

std::size_t round_up(std::size_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

const float* fp32(const Model& model, WeightId id, std::int32_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {count});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

// One matrix over the rows of `ids`, which must be one stored BF16 parent.
Weight matrix(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() > 1 && !ops::joins(inputs)) {
        throw std::invalid_argument("gemma4 vision: projection parts are not one stored parent");
    }
    Weight w = inputs.size() == 1
                   ? ops::prepare_linear_weight(inputs.front()).weight
                   : ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight;
    if (w.qtype != QType::BF16) {
        throw std::invalid_argument("gemma4 vision: the tower runs BF16 matrices");
    }
    return w;
}

// The scratch of one encode of `patches` patches: the patches, the residual and a block's normed
// input, then either the attention phase or the MLP phase (each in its own scope), then the
// pooled tokens and their projection. Projections write FP32 and their BF16 inputs come from the
// kernels that consume those outputs.
struct ScratchPlan {
    std::size_t patches_in, residual, input, attention, mlp, pooled, tokens;

    [[nodiscard]] std::size_t total() const {
        return patches_in + residual + input + std::max(attention, mlp) + pooled + tokens;
    }
};

ScratchPlan scratch(const VisionConfig& c, std::size_t patches) {
    const std::size_t h = c.hidden_size, w = c.intermediate_size;
    const std::size_t tokens = patches / c.patches_per_token();
    ScratchPlan s;
    s.patches_in = round_up(patches * c.patch_features() * 2);
    s.residual   = round_up(patches * h * 4);
    s.input      = round_up(patches * h * 2);
    // qkv (FP32), then q, k and v, the attention output and the output projection (FP32); the
    // patch projection (FP32) fits within it.
    s.attention = round_up(patches * 3 * h * 4) + 3 * round_up(patches * h * 2) +
                  round_up(patches * h * 2) + round_up(patches * h * 4);
    // [gate; up] (FP32), their product, the down projection (FP32).
    s.mlp    = round_up(patches * 2 * w * 4) + round_up(patches * w * 2) + round_up(patches * h * 4);
    s.pooled = round_up(tokens * h * 2);
    s.tokens = round_up(tokens * c.projection_dim * 2);
    return s;
}

} // namespace

VisionTower::VisionTower(const Model& model, std::uint32_t max_patches)
    : config_(*model.vision_config()), max_patches_(max_patches),
      streamed_(model.vision_placement() == VisionPlacement::Pinned) {
    if (!model.vision() || (model.vision_placement() != VisionPlacement::Device && !streamed_)) {
        throw std::invalid_argument("gemma4 vision: the model holds no device vision tower");
    }
    if (max_patches == 0 || max_patches % config_.patches_per_token() != 0) {
        throw std::invalid_argument("gemma4 vision: invalid patch capacity");
    }
    const VisionWeights& w = *model.vision();
    const int h            = int(config_.hidden_size);
    // Streamed, the small parts are copied to the device once; resident, they are used in place.
    const auto vector = [&](WeightId id, int count) {
        const float* p = fp32(model, id, count);
        return streamed_ ? static_cast<const float*>(own(p, std::size_t(count) * 4)) : p;
    };
    const auto dense = [&](std::initializer_list<WeightId> ids) {
        Weight m = matrix(model, ids);
        if (streamed_) {
            m.qdata = own(m.qdata, std::size_t(m.n) * m.k * 2);
        }
        return m;
    };
    patch_embedding_ = dense({w.patch_embedding});
    projection_      = dense({w.projection});
    const Tensor table =
        weight_tensor(model.weight(w.position_table).view,
                      {h, int(2 * config_.position_embedding_size)});
    if (table.dtype != DType::BF16) {
        throw std::invalid_argument("gemma4 vision: the position table must be BF16");
    }
    position_table_ = static_cast<const __nv_bfloat16*>(
        streamed_ ? own(table.data, std::size_t(h) * 2 * config_.position_embedding_size * 2)
                  : table.data);
    if (config_.standardize) {
        std_bias_  = vector(*w.std_bias, h);
        std_scale_ = vector(*w.std_scale, h);
    }
    for (const VisionLayerWeights& lw : w.layers) {
        Layer layer;
        // Streamed, the matrices keep their host addresses; staged() points them at a slot.
        layer.qkv                 = matrix(model, {lw.query, lw.key, lw.value});
        layer.output              = matrix(model, {lw.output});
        layer.gate_up             = matrix(model, {lw.mlp_gate, lw.mlp_up});
        layer.down                = matrix(model, {lw.mlp_down});
        layer.input_norm          = vector(lw.input_norm, h);
        layer.post_attention_norm = vector(lw.post_attention_norm, h);
        layer.ffn_norm            = vector(lw.ffn_norm, h);
        layer.post_ffn_norm       = vector(lw.post_ffn_norm, h);
        layer.q_norm              = vector(lw.query_norm, int(config_.head_dim));
        layer.k_norm              = vector(lw.key_norm, int(config_.head_dim));
        layers_.push_back(layer);
    }
    if (streamed_) {
        const Layer& first = layers_.front();
        for (const Weight* m : {&first.qkv, &first.output, &first.gate_up, &first.down}) {
            slot_bytes_ += round_up(std::size_t(m->n) * m->k * 2);
        }
        for (int i = 0; i < 2; ++i) {
            slots_[i] = DeviceBuffer(slot_bytes_);
            CUDA_CHECK(cudaEventCreateWithFlags(&copied_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&used_[i], cudaEventDisableTiming));
        }
        owned_bytes_ += 2 * slot_bytes_;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    }
    // The axial RoPE frequencies over a quarter of the head: theta^(-2i / (D/2)), as the reference
    // computes them in FP32.
    std::vector<float> freq(config_.head_dim / 4);
    const double spatial = double(config_.head_dim / 2);
    for (std::size_t i = 0; i < freq.size(); ++i) {
        freq[i] = float(1.0 / std::pow(double(config_.rope_theta), double(2 * i) / spatial));
    }
    inv_freq_ = DeviceBuffer(freq.size() * sizeof(float));
    inv_freq_.copy_from_host(freq.data(), freq.size() * sizeof(float));
}

VisionTower::~VisionTower() {
    for (int i = 0; i < 2; ++i) {
        if (copied_[i] != nullptr) { cudaEventDestroy(copied_[i]); }
        if (used_[i] != nullptr) { cudaEventDestroy(used_[i]); }
    }
    if (copy_stream_ != nullptr) { cudaStreamDestroy(copy_stream_); }
}

const void* VisionTower::own(const void* host, std::size_t bytes) {
    DeviceBuffer& buffer = owned_.emplace_back(bytes);
    buffer.copy_from_host(host, bytes);
    owned_bytes_ += bytes;
    return buffer.p;
}

VisionTower::Layer VisionTower::staged(std::size_t i, int slot) const {
    Layer layer   = layers_[i];
    auto* cursor  = static_cast<std::byte*>(slots_[slot].p);
    for (Weight* m : {&layer.qkv, &layer.output, &layer.gate_up, &layer.down}) {
        m->qdata = cursor;
        cursor += round_up(std::size_t(m->n) * m->k * 2);
    }
    return layer;
}

void VisionTower::upload(std::size_t i, int slot, cudaStream_t stream) const {
    // The slot's last block must be done with it, and its compute waits for the upload.
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, used_[slot]));
    const Layer& from = layers_[i];
    const Layer to    = staged(i, slot);
    const Weight* src[] = {&from.qkv, &from.output, &from.gate_up, &from.down};
    const Weight* dst[] = {&to.qkv, &to.output, &to.gate_up, &to.down};
    for (int m = 0; m < 4; ++m) {
        CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(dst[m]->qdata), src[m]->qdata,
                                   std::size_t(src[m]->n) * src[m]->k * 2, cudaMemcpyHostToDevice,
                                   copy_stream_));
    }
    CUDA_CHECK(cudaEventRecord(copied_[slot], copy_stream_));
    (void)stream;
}

std::size_t VisionTower::workspace_bytes() const {
    return scratch(config_, max_patches_).total() + (std::size_t(4) << 20);
}

void VisionTower::encode(const __nv_bfloat16* patches, int units, int grid_w, int grid_h,
                         __nv_bfloat16* out, WorkspaceArena& workspace,
                         cudaStream_t stream) const {
    const int per_unit  = grid_w * grid_h;
    const int total     = units * per_unit;
    const int k         = int(config_.pooling_kernel_size);
    if (units <= 0 || grid_w <= 0 || grid_h <= 0 || grid_w % k != 0 || grid_h % k != 0 ||
        std::uint32_t(total) > max_patches_) {
        throw std::invalid_argument("gemma4 vision: invalid encode batch");
    }
    const int h       = int(config_.hidden_size);
    const int width   = int(config_.intermediate_size);
    const int heads   = int(config_.num_attention_heads);
    const int d       = int(config_.head_dim);
    const float eps   = config_.rms_norm_eps;
    const int tokens  = total / (k * k);
    auto scope        = workspace.scope();
    Tensor input(const_cast<__nv_bfloat16*>(patches), DType::BF16,
                 {int(config_.patch_features()), total});
    auto* residual = static_cast<float*>(workspace.alloc_bytes(std::size_t(total) * h * 4).data);
    Tensor x       = workspace.alloc(DType::BF16, {h, total});
    {
        auto embed_scope = workspace.scope();
        Tensor projected = workspace.alloc(DType::FP32, {h, total});
        ops::linear(input, patch_embedding_, projected, stream);
        vk::EmbedArgs e;
        e.x         = static_cast<const float*>(projected.data);
        e.table     = position_table_;
        e.positions = int(config_.position_embedding_size);
        e.patches   = total;
        e.per_unit  = per_unit;
        e.grid_w    = grid_w;
        e.hidden    = h;
        e.norm      = layers_.front().input_norm;
        e.eps       = eps;
        e.h         = residual;
        e.xn        = static_cast<__nv_bfloat16*>(x.data);
        vk::embed(e, stream);
    }
    const ops::AttentionHeadGeometry geometry{.head_dim = d, .query_heads = heads, .kv_heads = heads};
    // The attention kernel's profile scales scores by 1/sqrt(D); the tower's are unscaled, so Q
    // leaves its kernel multiplied by sqrt(D).
    const float profile_scale = 1.0f / std::sqrt(float(d));
    if (streamed_) { upload(0, 0, stream); }
    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const int slot = int(li % 2);
        if (streamed_) {
            // The next block uploads while this one computes.
            if (li + 1 < layers_.size()) { upload(li + 1, 1 - slot, stream); }
            CUDA_CHECK(cudaStreamWaitEvent(stream, copied_[slot]));
        }
        const Layer layer = streamed_ ? staged(li, slot) : layers_[li];
        const float* next = li + 1 < layers_.size() ? layers_[li + 1].input_norm : nullptr;
        {
            auto attention_scope = workspace.scope();
            Tensor qkv           = workspace.alloc(DType::FP32, {3 * h, total});
            ops::linear(x, layer.qkv, qkv, stream);
            Tensor q = workspace.alloc(DType::BF16, {d, heads, total});
            Tensor kk = workspace.alloc(DType::BF16, {d, heads, total});
            Tensor v = workspace.alloc(DType::BF16, {d, heads, total});
            vk::QkvArgs a;
            a.qkv      = static_cast<const float*>(qkv.data);
            a.patches  = total;
            a.per_unit = per_unit;
            a.grid_w   = grid_w;
            a.heads    = heads;
            a.head_dim = d;
            a.q_norm   = layer.q_norm;
            a.k_norm   = layer.k_norm;
            a.inv_freq = static_cast<const float*>(inv_freq_.p);
            a.eps      = eps;
            a.q_scale  = std::sqrt(float(d));
            a.q        = static_cast<__nv_bfloat16*>(q.data);
            a.k        = static_cast<__nv_bfloat16*>(kk.data);
            a.v        = static_cast<__nv_bfloat16*>(v.data);
            vk::qkv(a, stream);
            Tensor attended = workspace.alloc(DType::BF16, {d, heads, total});
            if (units == 1) {
                ops::softmax_attention(q, kk, v, geometry, profile_scale, workspace, attended,
                                       stream);
            } else {
                ops::packed_softmax_attention(q, kk, v, geometry, profile_scale, per_unit,
                                              attended, stream);
            }
            Tensor flat = attended.view({h, total});
            Tensor o    = workspace.alloc(DType::FP32, {h, total});
            ops::linear(flat, layer.output, o, stream);
            vk::residual(residual, static_cast<const float*>(o.data), total, h,
                         layer.post_attention_norm, layer.ffn_norm, eps,
                         static_cast<__nv_bfloat16*>(x.data), stream);
        }
        {
            auto mlp_scope = workspace.scope();
            Tensor gu      = workspace.alloc(DType::FP32, {2 * width, total});
            ops::linear(x, layer.gate_up, gu, stream);
            Tensor mid = workspace.alloc(DType::BF16, {width, total});
            vk::geglu(static_cast<const float*>(gu.data), total, width,
                      static_cast<__nv_bfloat16*>(mid.data), stream);
            Tensor down = workspace.alloc(DType::FP32, {h, total});
            ops::linear(mid, layer.down, down, stream);
            vk::residual(residual, static_cast<const float*>(down.data), total, h,
                         layer.post_ffn_norm, next, eps, static_cast<__nv_bfloat16*>(x.data),
                         stream);
        }
        if (streamed_) { CUDA_CHECK(cudaEventRecord(used_[slot], stream)); }
    }
    Tensor pooled = workspace.alloc(DType::BF16, {h, tokens});
    vk::PoolArgs p;
    p.h         = residual;
    p.units     = units;
    p.grid_w    = grid_w;
    p.grid_h    = grid_h;
    p.pooling   = k;
    p.hidden    = h;
    p.scale     = std::sqrt(float(h));
    p.eps       = eps;
    p.std_bias  = std_bias_;
    p.std_scale = std_scale_;
    p.out       = static_cast<__nv_bfloat16*>(pooled.data);
    vk::pool(p, stream);
    Tensor embeddings(out, DType::BF16, {int(config_.projection_dim), tokens});
    ops::linear(pooled, projection_, embeddings, stream);
}

} // namespace ninfer::models::gemma4
