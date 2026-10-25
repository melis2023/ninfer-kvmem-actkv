// Qwen3.8-Flash-Next forward over a prefix of blocks, composed from the engine's Ops on BF16
// checkpoint weights, against the FP64 reference (tools/reference/qwen4_exp.py) on the same
// tokens. Point NINFER_QWEN4_EXP_SLICE at a directory that tools/reference/fetch_slice.py filled
// with the blocks' tensors (plus the embedding, the final mixer and the head when golden logits
// exist) and its n-gram rows, and in which the reference wrote golden.json. Skips without it.
//
// One sequence from position 0 as one prefill call. The residual stack after each block is
// compared by relative L2 against the reference's, and the logits by top-1 agreement and
// relative L2. The Ops run their production paths; every weight stays BF16 as the checkpoint
// stores it (the debug representation).
#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"
#include "ninfer/ops/ngram_rows.h"
#include "ninfer/ops/ple_inject.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using artifact::Json;
namespace fs = std::filesystem;

namespace {

constexpr int kH = 2560, kHC = 4, kWidth = kH * kHC, kLowrank = 320, kVocab = 248320;
constexpr float kEps = 1e-6f;
const std::string kPrefix = "model.language_model.";

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, 4);
    return value;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, 4);
    word += 0x7fffu + ((word >> 16) & 1u);
    return static_cast<std::uint16_t>(word >> 16);
}

// Minimal reader of a safetensors shard set described by model.safetensors.index.json.
class Checkpoint {
  public:
    explicit Checkpoint(fs::path root) : root_(std::move(root)) {
        std::ifstream index(root_ / "model.safetensors.index.json");
        require(index.good(), "slice index missing");
        const Json json = Json::parse(index);
        for (const auto& [name, file] : json.at("weight_map").items()) files_[name] = file;
        std::ifstream config(root_ / "config.json");
        config_ = Json::parse(config).at("text_config");
    }
    [[nodiscard]] const Json& config() const { return config_; }
    [[nodiscard]] bool has(const std::string& name) const { return files_.count(name) != 0; }

    std::vector<std::uint8_t> bytes(const std::string& name, std::vector<std::int64_t>* shape = nullptr) const {
        const auto found = files_.find(name);
        require(found != files_.end(), "tensor missing from the slice: " + name);
        std::ifstream file(root_ / found->second, std::ios::binary);
        std::uint64_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), 8);
        std::string header(header_size, '\0');
        file.read(header.data(), static_cast<std::streamsize>(header_size));
        const Json info = Json::parse(header).at(name);
        const auto offsets = info.at("data_offsets");
        const std::uint64_t begin = offsets[0], end = offsets[1];
        if (shape) *shape = info.at("shape").get<std::vector<std::int64_t>>();
        std::vector<std::uint8_t> data(end - begin);
        file.seekg(static_cast<std::streamoff>(8 + header_size + begin));
        file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        require(static_cast<std::uint64_t>(file.gcount()) == data.size(), "short read of " + name);
        return data;
    }

  private:
    fs::path root_;
    std::map<std::string, std::string> files_;
    Json config_;
};

// A device copy of a host byte vector.
struct Device {
    DeviceBuffer buffer;
    explicit Device(const std::vector<std::uint8_t>& bytes) : buffer(std::max<std::size_t>(bytes.size(), 16)) {
        buffer.copy_from_host(bytes.data(), bytes.size());
    }
    explicit Device(std::size_t bytes) : buffer(std::max<std::size_t>(bytes, 16)) { buffer.fill(0); }
};

Weight bf16_weight(const void* data, std::int32_t n, std::int32_t k) {
    Weight weight{};
    weight.payload       = data;
    weight.payload_bytes = static_cast<std::uint64_t>(n) * k * 2;
    weight.qtype         = QType::BF16;
    weight.shape[0] = weight.padded_shape[0] = n;
    weight.shape[1] = weight.padded_shape[1] = k;
    weight.ndim  = 2;
    weight.qdata = data;
    weight.n     = n;
    weight.k     = k;
    weight.layout = QuantLayout::Contiguous;
    return weight;
}

std::vector<std::uint8_t> fp32_from_bf16(const std::vector<std::uint8_t>& bf16) {
    std::vector<std::uint8_t> out(bf16.size() * 2);
    for (std::size_t i = 0; i < bf16.size() / 2; ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, bf16.data() + 2 * i, 2);
        const float value = bf16_to_float(bits);
        std::memcpy(out.data() + 4 * i, &value, 4);
    }
    return out;
}

struct Context {
    const Checkpoint& ckpt;
    DeviceContext& device;
    int tokens;
    std::vector<std::int32_t> token_ids;
    Tensor stack;
    Tensor positions;
};

// x (BF16 [k, T]) through a BF16 checkpoint matrix [n, k] into out (BF16 [n, T]).
void project(Context& c, const std::string& name, const Tensor& x, Tensor& out, std::int32_t n, std::int32_t k) {
    Device weight(c.ckpt.bytes(name));
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_workspace_capacity_bytes(QType::BF16, n, k, ops::LinearPolicy::A16Only, c.tokens, c.tokens), 256));
    ops::linear(x, bf16_weight(weight.buffer.p, n, k), out, ops::LinearPolicy::A16Only, workspace, c.device.stream);
    c.device.synchronize();
}

// One hyper-connection read: mixed BF16 [H, T] and, with inject rows, the inject weights.
void hc_read(Context& c, const std::string& prefix, Tensor& mixed, Tensor* inject) {
    Device norm(c.ckpt.bytes(prefix + "hc_norm.weight"));
    Device down(c.ckpt.bytes(prefix + "input_mix_weight_down.weight"));
    Device up(c.ckpt.bytes(prefix + "input_mix_weight_up.weight"));
    std::unique_ptr<Device> inject_rows;
    Tensor t_norm(norm.buffer.p, DType::BF16, {kWidth});
    Tensor t_down(down.buffer.p, DType::BF16, {kWidth, kLowrank});
    Tensor t_up(up.buffer.p, DType::BF16, {kLowrank, kWidth});
    Tensor t_inject;
    ops::HyperConnectionWeights weights{&t_norm, &t_down, &t_up, nullptr};
    if (inject) {
        inject_rows = std::make_unique<Device>(c.ckpt.bytes(prefix + "block_inject_weight.weight"));
        t_inject = Tensor(inject_rows->buffer.p, DType::BF16, {kWidth, kHC});
        weights.inject = &t_inject;
    }
    WorkspaceArena workspace(ops::hyper_connection_read_workspace_bytes(kHC, kH, kLowrank, c.tokens));
    ops::hyper_connection_read(c.stack, weights, kEps, workspace, mixed, inject, c.device.stream);
    c.device.synchronize();
}

void ple(Context& c, int layer, const fs::path& rows_file) {
    const std::string p = kPrefix + "layers." + std::to_string(layer) + ".ple.";
    // Row ids from the hash, rows from the slice's row file.
    const auto& cfg = c.ckpt.config();
    models::qwen4_exp::NgramHashSpec spec{
        .vocab_size = kVocab, .ngram_size = cfg.at("ngram_size"), .heads_per_ngram = cfg.at("heads_per_ngram"),
        .ple_layer_index = 0, .vocab_base = cfg.at("ngram_vocab_size_base"),
        .divisible_by = cfg.at("make_ngram_vocab_size_divisible_by"), .seed = 1234};
    const auto constants = models::qwen4_exp::derive_ngram_hash_constants(spec);
    const std::int32_t eos = cfg.at("eos_token_id");
    auto context = models::qwen4_exp::NgramContext::sequence_start(constants, eos);
    std::vector<std::uint64_t> ids(static_cast<std::size_t>(c.tokens) * constants.heads());
    models::qwen4_exp::ngram_row_ids(constants, c.token_ids, eos, kVocab, context, ids);
    std::ifstream file(rows_file, std::ios::binary);
    require(file.good(), "n-gram rows file missing");
    std::uint64_t count = 0;
    file.read(reinterpret_cast<char*>(&count), 8);
    std::map<std::uint64_t, std::vector<std::uint8_t>> rows;
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t id = 0;
        file.read(reinterpret_cast<char*>(&id), 8);
        std::vector<std::uint8_t> row(320);
        file.read(reinterpret_cast<char*>(row.data()), 320);
        rows.emplace(id, std::move(row));
    }
    std::vector<std::uint8_t> staged;
    for (const auto id : ids) {
        const auto found = rows.find(id);
        require(found != rows.end(), "n-gram row " + std::to_string(id) + " not fetched");
        staged.insert(staged.end(), found->second.begin(), found->second.end());
    }
    Device d_rows(staged), d_emb(static_cast<std::size_t>(kH) * c.tokens * 2);
    Tensor t_rows(d_rows.buffer.p, DType::U8, {320, static_cast<std::int32_t>(ids.size())});
    Tensor emb(d_emb.buffer.p, DType::BF16, {kH, c.tokens});
    ops::ngram_embed_rows(t_rows, ops::NgramRowFormat::Bf16, 16, emb, c.device.stream);
    Device d_key(static_cast<std::size_t>(kWidth) * c.tokens * 2), d_value(static_cast<std::size_t>(kH) * c.tokens * 2);
    Tensor key(d_key.buffer.p, DType::BF16, {kWidth, c.tokens}), value(d_value.buffer.p, DType::BF16, {kH, c.tokens});
    project(c, p + "key_proj.weight", emb, key, kWidth, kH);
    project(c, p + "value_proj.weight", emb, value, kH, kH);
    Device nk(c.ckpt.bytes(p + "norm_key.weight")), nq(c.ckpt.bytes(p + "norm_query.weight")),
        nc(c.ckpt.bytes(p + "norm_conv.weight")), conv(c.ckpt.bytes(p + "conv1d.weight")),
        history(static_cast<std::size_t>(kWidth) * 9 * 4);
    Tensor t_nk(nk.buffer.p, DType::BF16, {kWidth}), t_nq(nq.buffer.p, DType::BF16, {kWidth}),
        t_nc(nc.buffer.p, DType::BF16, {kWidth}), t_conv(conv.buffer.p, DType::BF16, {4, kWidth}),
        t_history(history.buffer.p, DType::FP32, {kWidth, 9});
    WorkspaceArena workspace(ops::ple_inject_workspace_bytes(c.tokens));
    ops::ple_inject(c.stack, key, value, {&t_nk, &t_nq, &t_nc, &t_conv}, kEps, t_history, workspace, c.device.stream);
    c.device.synchronize();
}

void gdn(Context& c, int layer, const Tensor& a, Tensor& y) {
    const std::string p = kPrefix + "layers." + std::to_string(layer) + ".linear_attn.";
    const int T = c.tokens;
    Device qkv(static_cast<std::size_t>(10240) * T * 2), z(static_cast<std::size_t>(6144) * T * 2),
        ga(static_cast<std::size_t>(48) * T * 2), gb(static_cast<std::size_t>(48) * T * 2);
    Tensor t_qkv(qkv.buffer.p, DType::BF16, {10240, T}), t_z(z.buffer.p, DType::BF16, {6144, T}),
        t_a(ga.buffer.p, DType::BF16, {48, T}), t_b(gb.buffer.p, DType::BF16, {48, T});
    project(c, p + "in_proj_qkv.weight", a, t_qkv, 10240, kH);
    project(c, p + "in_proj_z.weight", a, t_z, 6144, kH);
    project(c, p + "in_proj_a.weight", a, t_a, 48, kH);
    project(c, p + "in_proj_b.weight", a, t_b, 48, kH);
    // Conv weight: the checkpoint's [C, 1, 4] (taps contiguous) to the Op's tap-major [C, 4].
    const auto conv_raw = c.ckpt.bytes(p + "conv1d.weight");
    std::vector<std::uint8_t> conv(conv_raw.size());
    for (int ch = 0; ch < 10240; ++ch)
        for (int j = 0; j < 4; ++j)
            std::memcpy(conv.data() + 2 * (static_cast<std::size_t>(j) * 10240 + ch), conv_raw.data() + 2 * (static_cast<std::size_t>(ch) * 4 + j), 2);
    Device d_conv(conv), state(static_cast<std::size_t>(10240) * 3 * 2), q(static_cast<std::size_t>(2048) * T * 2),
        k(static_cast<std::size_t>(2048) * T * 2), v(static_cast<std::size_t>(6144) * T * 2);
    Tensor t_conv(d_conv.buffer.p, DType::BF16, {10240, 4}), t_state(state.buffer.p, DType::BF16, {10240, 3}),
        t_q(q.buffer.p, DType::BF16, {2048, T}), t_k(k.buffer.p, DType::BF16, {2048, T}), t_v(v.buffer.p, DType::BF16, {6144, T});
    ops::causal_conv1d_silu_split(t_qkv, t_conv, t_state, t_state, t_q, t_k, t_v, c.device.stream);
    Device alog(fp32_from_bf16(c.ckpt.bytes(p + "A_log"))), dt(fp32_from_bf16(c.ckpt.bytes(p + "dt_bias"))),
        g(static_cast<std::size_t>(48) * T * 4), beta(static_cast<std::size_t>(48) * T * 4);
    Tensor t_alog(alog.buffer.p, DType::FP32, {48}), t_dt(dt.buffer.p, DType::FP32, {48}),
        t_g(g.buffer.p, DType::FP32, {48, T}), t_beta(beta.buffer.p, DType::FP32, {48, T});
    ops::gdn_gating(t_a, t_b, t_alog, t_dt, t_g, t_beta, c.device.stream);
    Device ssm(static_cast<std::size_t>(128) * 128 * 48 * 4), o(static_cast<std::size_t>(6144) * T * 2), gated(static_cast<std::size_t>(6144) * T * 2);
    Tensor q3(q.buffer.p, DType::BF16, {128, 16, T}), k3(k.buffer.p, DType::BF16, {128, 16, T}), v3(v.buffer.p, DType::BF16, {128, 48, T}),
        t_ssm(ssm.buffer.p, DType::FP32, {128, 128, 48}), t_o(o.buffer.p, DType::BF16, {128, 48, T});
    WorkspaceArena workspace(std::max<std::size_t>(ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, T, T), 256));
    ops::gated_delta_net(q3, k3, v3, t_g, t_beta, 1.0f / std::sqrt(128.0f), true, workspace, t_ssm, t_o, c.device.stream);
    Device norm(c.ckpt.bytes(p + "norm.weight"));
    Tensor t_norm(norm.buffer.p, DType::BF16, {128});
    Tensor rows_o(o.buffer.p, DType::BF16, {128, 48 * T}), rows_z(z.buffer.p, DType::BF16, {128, 48 * T}),
        rows_out(gated.buffer.p, DType::BF16, {128, 48 * T});
    ops::gated_rmsnorm(rows_o, t_norm, rows_z, ops::GateActivation::Sigmoid, kEps, rows_out, c.device.stream);
    Tensor t_gated(gated.buffer.p, DType::BF16, {6144, T});
    project(c, p + "out_proj.weight", t_gated, y, kH, 6144);
}

void qsa(Context& c, int layer, const Tensor& a, Tensor& y) {
    const std::string p = kPrefix + "layers." + std::to_string(layer) + ".self_attn.";
    const int T = c.tokens;
    // q_proj rows are per head [q (256) | gate (256)]: split them into two matrices.
    std::vector<std::int64_t> shape;
    const auto qg = c.ckpt.bytes(p + "q_proj.weight", &shape);
    std::vector<std::uint8_t> wq(qg.size() / 2), wg(qg.size() / 2);
    const std::size_t row = static_cast<std::size_t>(kH) * 2;
    for (int h = 0; h < 24; ++h) {
        std::memcpy(wq.data() + h * 256 * row, qg.data() + (h * 512) * row, 256 * row);
        std::memcpy(wg.data() + h * 256 * row, qg.data() + (h * 512 + 256) * row, 256 * row);
    }
    Device d_wq(wq), d_wg(wg);
    WorkspaceArena lws(std::max<std::size_t>(ops::linear_workspace_capacity_bytes(QType::BF16, 6144, kH, ops::LinearPolicy::A16Only, T, T), 256));
    Device q(static_cast<std::size_t>(6144) * T * 2), gate(static_cast<std::size_t>(6144) * T * 2),
        k(static_cast<std::size_t>(512) * T * 2), v(static_cast<std::size_t>(512) * T * 2), index(static_cast<std::size_t>(640) * T * 2);
    Tensor t_q(q.buffer.p, DType::BF16, {6144, T}), t_gate(gate.buffer.p, DType::BF16, {6144, T});
    ops::linear(a, bf16_weight(d_wq.buffer.p, 6144, kH), t_q, ops::LinearPolicy::A16Only, lws, c.device.stream);
    ops::linear(a, bf16_weight(d_wg.buffer.p, 6144, kH), t_gate, ops::LinearPolicy::A16Only, lws, c.device.stream);
    Tensor t_k(k.buffer.p, DType::BF16, {512, T}), t_v(v.buffer.p, DType::BF16, {512, T}), t_index(index.buffer.p, DType::BF16, {640, T});
    project(c, p + "k_proj.weight", a, t_k, 512, kH);
    project(c, p + "v_proj.weight", a, t_v, 512, kH);
    project(c, p + "indexer.index_qk_proj.weight", a, t_index, 640, kH);
    Device qn(c.ckpt.bytes(p + "q_norm.weight")), kn(c.ckpt.bytes(p + "k_norm.weight")),
        q_out(static_cast<std::size_t>(6144) * T * 2), k_out(static_cast<std::size_t>(512) * T * 2);
    Tensor t_qn(qn.buffer.p, DType::BF16, {256}), t_kn(kn.buffer.p, DType::BF16, {256});
    Tensor q3(q.buffer.p, DType::BF16, {256, 24, T}), k3(k.buffer.p, DType::BF16, {256, 2, T}),
        qo(q_out.buffer.p, DType::BF16, {256, 24, T}), ko(k_out.buffer.p, DType::BF16, {256, 2, T});
    ops::rmsnorm_rope(c.positions, t_qn, t_kn, q3, k3, qo, ko, c.device.stream);
    // A BF16 paged cache for the sequence, pages in order.
    const int pages = (T + kPagedKVPageSize - 1) / kPagedKVPageSize;
    Device kc(static_cast<std::size_t>(256) * kPagedKVPageSize * 2 * pages * 2), vc(static_cast<std::size_t>(256) * kPagedKVPageSize * 2 * pages * 2);
    std::vector<std::int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::vector<std::uint8_t> table_bytes(table.size() * 4);
    std::memcpy(table_bytes.data(), table.data(), table_bytes.size());
    Device d_table(table_bytes);
    PagedKVLayerView cache{};
    cache.k_pages = Tensor(kc.buffer.p, DType::BF16, {256, kPagedKVPageSize, 2, pages});
    cache.v_pages = Tensor(vc.buffer.p, DType::BF16, {256, kPagedKVPageSize, 2, pages});
    cache.block_table = Tensor(d_table.buffer.p, DType::I32, {pages});
    cache.head_dim = 256;
    cache.num_kv_heads = 2;
    cache.storage = KvCacheStorage::BFloat16;
    Tensor v3(v.buffer.p, DType::BF16, {256, 2, T});
    ops::kv_cache_append(ko, v3, c.positions, cache, c.device.stream);
    // Indexer and selection.
    Device iqn(c.ckpt.bytes(p + "indexer.q_layernorm.weight")), ikn(c.ckpt.bytes(p + "indexer.k_layernorm.weight"));
    Tensor t_iqn(iqn.buffer.p, DType::BF16, {128}), t_ikn(ikn.buffer.p, DType::BF16, {128});
    const int capacity = std::max(1, T / 4 + 1);
    Device pooled(static_cast<std::size_t>(128) * capacity * 4), tail(static_cast<std::size_t>(128) * 3 * 4),
        selected(static_cast<std::size_t>(512) * T * 4), counts(static_cast<std::size_t>(T) * 4);
    Tensor t_pooled(pooled.buffer.p, DType::FP32, {128, capacity}), t_tail(tail.buffer.p, DType::FP32, {128, 3}),
        t_selected(selected.buffer.p, DType::I32, {512, T}), t_counts(counts.buffer.p, DType::I32, {T});
    const ops::QsaIndexerWeights iw{&t_iqn, &t_ikn};
    ops::qsa_indexer_append(t_index, c.positions, iw, kEps, t_pooled, t_tail, c.device.stream);
    WorkspaceArena sws(ops::qsa_indexer_select_workspace_bytes(T, capacity));
    ops::qsa_indexer_select(t_index, c.positions, iw, kEps, t_pooled, sws, t_selected, t_counts, c.device.stream);
    Device attn(static_cast<std::size_t>(6144) * T * 2);
    Tensor t_attn(attn.buffer.p, DType::BF16, {256, 24, T});
    WorkspaceArena attention_ws(ops::sparse_softmax_attention_workspace_bytes(T));
    ops::sparse_softmax_attention(qo, c.positions, t_selected, t_counts, cache, 1.0f / 16.0f, attention_ws, t_attn,
                                  c.device.stream);
    Tensor flat_attn(attn.buffer.p, DType::BF16, {6144, T});
    ops::sigmoid_mul(t_gate, flat_attn, c.device.stream);
    c.device.synchronize();
    project(c, p + "o_proj.weight", flat_attn, y, kH, 6144);
}

void moe(Context& c, int layer, const Tensor& m, Tensor& y) {
    const std::string p = kPrefix + "layers." + std::to_string(layer) + ".mlp.";
    const int T = c.tokens;
    Device router(c.ckpt.bytes(p + "gate.weight")), shared_gate(c.ckpt.bytes(p + "shared_expert_gate.weight")),
        ids(static_cast<std::size_t>(10) * T * 4), weights(static_cast<std::size_t>(10) * T * 4), shared(static_cast<std::size_t>(T) * 4);
    Tensor t_router(router.buffer.p, DType::BF16, {kH, 512}), t_sg(shared_gate.buffer.p, DType::BF16, {kH}),
        t_ids(ids.buffer.p, DType::I32, {10, T}), t_w(weights.buffer.p, DType::FP32, {10, T}), t_shared(shared.buffer.p, DType::FP32, {T});
    WorkspaceArena route_ws(ops::moe_route_workspace_bytes(T, 512));
    ops::moe_route(m, t_router, t_sg, route_ws, t_ids, t_w, t_shared, c.device.stream);
    Device gate_up(c.ckpt.bytes(p + "experts.gate_up_proj")), down(c.ckpt.bytes(p + "experts.down_proj"));
    auto sgu = c.ckpt.bytes(p + "shared_expert.gate_proj.weight");
    const auto su = c.ckpt.bytes(p + "shared_expert.up_proj.weight");
    sgu.insert(sgu.end(), su.begin(), su.end());
    Device d_sgu(sgu), d_sdn(c.ckpt.bytes(p + "shared_expert.down_proj.weight"));
    Tensor t_gu(gate_up.buffer.p, DType::BF16, {kH, 1280, 512}), t_dn(down.buffer.p, DType::BF16, {640, kH, 512}),
        t_sgu(d_sgu.buffer.p, DType::BF16, {kH, 1280}), t_sdn(d_sdn.buffer.p, DType::BF16, {640, kH});
    WorkspaceArena workspace(ops::moe_experts_bf16_workspace_bytes(T));
    ops::moe_experts_bf16(m, t_ids, t_w, t_shared, t_gu, t_dn, t_sgu, t_sdn, workspace, y, c.device.stream);
    c.device.synchronize();
}

double relative_l2(const std::vector<float>& got, const std::vector<float>& want) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        num += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        den += double(want[i]) * want[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

std::vector<float> read_f32(const fs::path& path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * 4));
    require(static_cast<std::size_t>(file.gcount()) == count * 4, "short golden " + path.string());
    return values;
}

int run(const fs::path& root) {
    const Checkpoint ckpt(root);
    std::ifstream golden_file(root / "golden.json");
    require(golden_file.good(), "golden.json missing: run tools/reference/qwen4_exp.py first");
    const Json golden = Json::parse(golden_file);
    std::vector<std::int32_t> tokens = golden.at("tokens").get<std::vector<std::int32_t>>();
    const int layers = golden.at("layers");
    const int T      = static_cast<int>(tokens.size());
    DeviceContext device;
    // The stack: every stream starts as the token's embedding.
    std::vector<std::int64_t> shape;
    const auto embed = ckpt.bytes(kPrefix + "embed_tokens.weight", &shape);
    std::vector<float> stack_host(static_cast<std::size_t>(kWidth) * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < kHC; ++c)
            for (int d = 0; d < kH; ++d) {
                std::uint16_t bits;
                std::memcpy(&bits, embed.data() + 2 * (static_cast<std::size_t>(tokens[t]) * kH + d), 2);
                stack_host[(static_cast<std::size_t>(t) * kHC + c) * kH + d] = bf16_to_float(bits);
            }
    DeviceBuffer stack(stack_host.size() * 4);
    stack.copy_from_host(stack_host.data(), stack.bytes);
    std::vector<std::int32_t> positions(T);
    for (int t = 0; t < T; ++t) positions[t] = t;
    DeviceBuffer d_positions(T * 4);
    d_positions.copy_from_host(positions.data(), T * 4);
    Context c{ckpt, device, T, tokens, Tensor(stack.p, DType::FP32, {kH, kHC, T}), Tensor(d_positions.p, DType::I32, {T})};
    const auto& cfg = ckpt.config();
    std::vector<int> ple_blocks;
    for (const auto& id : cfg.at("ple_layer_ids")) ple_blocks.push_back(id.get<int>() - 1);
    int failures = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const std::string l = kPrefix + "layers." + std::to_string(layer) + ".";
        if (std::find(ple_blocks.begin(), ple_blocks.end(), layer) != ple_blocks.end()) ple(c, layer, root / "ngram_rows.bin");
        DeviceBuffer mixed(static_cast<std::size_t>(kH) * T * 2), inject(static_cast<std::size_t>(kHC) * T * 4),
            y(static_cast<std::size_t>(kH) * T * 4);
        Tensor t_mixed(mixed.p, DType::BF16, {kH, T}), t_inject(inject.p, DType::FP32, {kHC, T});
        hc_read(c, l + "attn_hyper_connection.", t_mixed, &t_inject);
        Tensor t_y(y.p, DType::BF16, {kH, T});
        if (cfg.at("layer_types")[layer] == "linear_attention") gdn(c, layer, t_mixed, t_y);
        else qsa(c, layer, t_mixed, t_y);
        ops::hyper_connection_write(c.stack, t_y, t_inject, device.stream);
        hc_read(c, l + "mlp_hyper_connection.", t_mixed, &t_inject);
        Tensor t_y32(y.p, DType::FP32, {kH, T});
        moe(c, layer, t_mixed, t_y32);
        ops::hyper_connection_write(c.stack, t_y32, t_inject, device.stream);
        device.synchronize();
        std::vector<float> got(stack_host.size());
        stack.copy_to_host(got.data(), got.size() * 4);
        const auto want = read_f32(root / golden.at("R")[layer].get<std::string>(), got.size());
        const double error = relative_l2(got, want);
        std::cout << "block " << layer << " (" << cfg.at("layer_types")[layer].get<std::string>()
                  << "): stack relative L2 " << error << '\n';
        if (!(error < 2e-2)) ++failures;
    }
    if (golden.contains("logits")) {
        DeviceBuffer mixed(static_cast<std::size_t>(kH) * T * 2), logits(static_cast<std::size_t>(kVocab) * T * 2);
        Tensor t_mixed(mixed.p, DType::BF16, {kH, T}), t_logits(logits.p, DType::BF16, {kVocab, T});
        hc_read(c, kPrefix + "hyper_connection_mixer.", t_mixed, nullptr);
        project(c, "lm_head.weight", t_mixed, t_logits, kVocab, kH);
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(kVocab) * T);
        logits.copy_to_host(bits.data(), bits.size() * 2);
        std::vector<float> got(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) got[i] = bf16_to_float(bits[i]);
        const auto want = read_f32(root / golden.at("logits").get<std::string>(), got.size());
        int agree = 0;
        for (int t = 0; t < T; ++t) {
            const auto a = got.begin() + static_cast<std::ptrdiff_t>(t) * kVocab;
            const auto b = want.begin() + static_cast<std::ptrdiff_t>(t) * kVocab;
            agree += std::max_element(a, a + kVocab) - a == std::max_element(b, b + kVocab) - b;
        }
        const double error = relative_l2(got, want);
        std::cout << "logits: top-1 agreement " << agree << "/" << T << ", relative L2 " << error << '\n';
        if (agree < T || !(error < 5e-2)) ++failures;
    }
    return failures;
}

} // namespace

int main() {
    const char* root = std::getenv("NINFER_QWEN4_EXP_SLICE");
    if (root == nullptr) {
        std::cout << "SKIP: NINFER_QWEN4_EXP_SLICE is not set\n";
        return 77;
    }
    try {
        const int failures = run(root);
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " qwen4_exp slice\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
