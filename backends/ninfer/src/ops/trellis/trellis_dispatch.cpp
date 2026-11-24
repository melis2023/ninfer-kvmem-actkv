#include "core/arena.h"
#include "ninfer/ops/trellis.h"
#include "ops/trellis/trellis_kernels.cuh"
#include "ops/trellis/trellis_reference.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::ops {
namespace {

using detail::trellis::Desc;
using detail::trellis::DevicePlan;
using detail::trellis::SidePlan;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// A weight's transform plans are immutable once the artifact is loaded, so a process-wide cache
// keyed by the descriptor plus geometry serves every call without touching the host per step.
struct PlanEntry {
    SidePlan rows;
    SidePlan cols;
    DeviceBuffer perm_rows;
    DeviceBuffer perm_cols;
    DeviceBuffer paley_rows;
    DeviceBuffer paley_cols;
};

std::string plan_key(const Weight& w) {
    std::string key = std::to_string(static_cast<int>(w.qtype)) + ":" + std::to_string(w.k) + ":" +
                      std::to_string(w.n) + ":";
    for (int i = 0; i < w.trellis_desc_len; ++i) {
        key += std::to_string(w.trellis_desc[i]) + ",";
    }
    return key;
}

std::mutex g_plan_mutex;
std::unordered_map<std::string, std::unique_ptr<PlanEntry>> g_plans;

const PlanEntry& plans_for(const Weight& w, const Desc& desc) {
    const std::string key = plan_key(w);
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    const auto found = g_plans.find(key);
    if (found != g_plans.end()) {
        return *found->second;
    }
    auto entry = std::make_unique<PlanEntry>();
    if (!detail::trellis::build_side_plan(w, w.n, true, entry->rows) ||
        !detail::trellis::build_side_plan(w, w.k, false, entry->cols)) {
        throw std::invalid_argument("trellis: unsupported side descriptor");
    }
    auto upload = [](const SidePlan& plan, DeviceBuffer& buffer, DeviceBuffer& paley) {
        if (plan.kind != 2 || plan.perm.empty()) {
            return;
        }
        buffer = DeviceBuffer(plan.perm.size() * sizeof(std::int32_t));
        buffer.copy_from_host(plan.perm.data(), buffer.bytes);
        // The Paley matrix of the side: 1x1, 12x12, 20x20 or 68x68 signed bytes.
        std::vector<signed char> table(static_cast<std::size_t>(plan.m) * plan.m, 0);
        detail::trellis::paley_table(plan.m, table.data());
        paley = DeviceBuffer(table.size());
        paley.copy_from_host(table.data(), paley.bytes);
    };
    upload(entry->rows, entry->perm_rows, entry->paley_rows);
    upload(entry->cols, entry->perm_cols, entry->paley_cols);
    (void)desc;
    const auto [it, inserted] = g_plans.emplace(key, std::move(entry));
    return *it->second;
}

DevicePlan device_view(const SidePlan& plan, const DeviceBuffer& perm, const DeviceBuffer& paley) {
    DevicePlan view;
    if (plan.kind == 2) {
        view.perm = static_cast<const std::int32_t*>(perm.p);
        view.paley = static_cast<const signed char*>(paley.p);
    }
    return view;
}

void validate_weight(const Weight& w) {
    Desc desc;
    if (!detail::trellis::is_trellis(w.qtype)) {
        throw std::invalid_argument("trellis: weight qtype is not a coded format");
    }
    if (w.trellis_s16 == nullptr || w.trellis_pre == nullptr || w.trellis_post == nullptr) {
        throw std::invalid_argument("trellis: missing s16/pre/post side tensors");
    }
    if (w.qdata == nullptr) {
        throw std::invalid_argument("trellis: missing coded payload");
    }
    if (!detail::trellis::parse_desc(w, w.k, w.n, desc)) {
        throw std::invalid_argument("trellis: unsupported descriptor");
    }
    const int bits = detail::trellis::bits_of(w.qtype);
    if (bits < 1 || bits > 6) {
        throw std::invalid_argument("trellis: weight qtype is not a coded format");
    }
    const std::int64_t expected = static_cast<std::int64_t>(w.n) * (w.k / 16) * (2 * bits);
    if (static_cast<std::int64_t>(w.payload_bytes) < expected) {
        throw std::invalid_argument("trellis: payload is shorter than the coded geometry");
    }
}

void apply_column_side(const Weight& w, const PlanEntry& plans, float* plane, float* tmp, float* tmp2,
                       int tokens, cudaStream_t stream) {
    if (plans.cols.kind == 0) {
        return;
    }
    if (plans.cols.kind == 1) {
        detail::trellis::rht_inplace(plane, w.k, plans.cols.block, tokens, stream);
        return;
    }
    detail::trellis::hadfull_pass(plane, plane, tmp, tmp2, plans.cols,
                                  device_view(plans.cols, plans.perm_cols, plans.paley_cols), false,
                                  tokens, stream);
}

} // namespace

std::size_t trellis_linear_workspace_bytes(const Weight& w, std::int32_t max_tokens) {
    validate_weight(w);
    if (max_tokens <= 0) {
        throw std::invalid_argument("trellis workspace: token count must be positive");
    }
    const std::int64_t wider = w.k > w.n ? w.k : w.n;
    // Rotated activation, the FP32 coded product, and two logical-order scratch planes.
    const std::int64_t floats = (w.k + w.n + 2 * wider) * static_cast<std::int64_t>(max_tokens);
    return static_cast<std::size_t>(floats) * sizeof(float);
}

void trellis_linear(const Tensor& x, const Weight& w, Tensor& y, std::int32_t max_tokens,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    validate_weight(w);
    if (x.dtype != DType::BF16 || y.dtype != DType::BF16) {
        throw std::invalid_argument("trellis: x and y must be BF16");
    }
    if (x.ne[2] != 1 || x.ne[3] != 1 || y.ne[2] != 1 || y.ne[3] != 1) {
        throw std::invalid_argument("trellis: x and y must have shape [K,T] and [N,T]");
    }
    if (x.ne[0] != w.k || y.ne[0] != w.n || x.ne[1] != y.ne[1]) {
        throw std::invalid_argument("trellis: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !y.is_contiguous()) {
        throw std::invalid_argument("trellis: x and y must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(y.data, 16)) {
        throw std::invalid_argument("trellis: x and y must be 16-byte aligned");
    }
    const int tokens = x.ne[1];
    if (tokens > max_tokens) {
        throw std::invalid_argument("trellis: token count exceeds the workspace capacity");
    }
    Desc desc;
    if (!detail::trellis::parse_desc(w, w.k, w.n, desc)) {
        throw std::invalid_argument("trellis: unsupported descriptor");
    }
    const PlanEntry& plans = plans_for(w, desc);

    const std::size_t bytes = trellis_linear_workspace_bytes(w, max_tokens);
    if (workspace.capacity() < bytes) {
        throw std::invalid_argument("trellis: workspace is smaller than the weight needs");
    }
    auto scope = workspace.scope();
    Tensor xa = workspace.alloc(DType::FP32, {w.k, tokens});
    Tensor yr = workspace.alloc(DType::FP32, {w.n, tokens});
    const std::int32_t wider = w.k > w.n ? w.k : w.n;
    Tensor tmp = workspace.alloc(DType::FP32, {wider, tokens});
    Tensor tmp2 = workspace.alloc(DType::FP32, {wider, tokens});

    detail::trellis::gather_scale(x, w, static_cast<float*>(xa.data), tokens, w.k, stream);
    apply_column_side(w, plans, static_cast<float*>(xa.data), static_cast<float*>(tmp.data),
                      static_cast<float*>(tmp2.data), tokens, stream);
    detail::trellis::tile_product(static_cast<const float*>(xa.data), w,
                                  static_cast<float*>(yr.data), tokens, stream);

    float* yrot = static_cast<float*>(yr.data);
    if (plans.rows.kind == 1) {
        detail::trellis::rht_inplace(yrot, w.n, plans.rows.block, tokens, stream);
    } else if (plans.rows.kind == 2) {
        detail::trellis::hadfull_pass(yrot, yrot, static_cast<float*>(tmp.data),
                                      static_cast<float*>(tmp2.data), plans.rows,
                                      device_view(plans.rows, plans.perm_rows, plans.paley_rows), true,
                                      tokens, stream);
    }
    detail::trellis::post_and_gather(yrot, w, y, tokens, stream);
}

void trellis_embedding(const Tensor& ids, const Weight& table, Tensor& rows, cudaStream_t stream) {
    validate_weight(table);
    Desc desc;
    if (!detail::trellis::parse_desc(table, table.k, table.n, desc)) {
        throw std::invalid_argument("trellis embedding: unsupported descriptor");
    }
    if (desc.rows_kind != 0) {
        throw std::invalid_argument("trellis embedding: a coded embedding carries no row side");
    }
    if (ids.dtype != DType::I32 || rows.dtype != DType::BF16) {
        throw std::invalid_argument("trellis embedding: ids must be I32 and rows BF16");
    }
    if (ids.ne[0] != rows.ne[1] || rows.ne[0] != table.k) {
        throw std::invalid_argument("trellis embedding: expected I32 [T] and BF16 [K,T]");
    }
    if (!ids.is_contiguous() || !rows.is_contiguous() || !aligned_to(rows.data, 16)) {
        throw std::invalid_argument("trellis embedding: tensors must be contiguous and aligned");
    }
    const int tokens = ids.ne[0];
    const PlanEntry& plans = plans_for(table, desc);
    const std::int64_t wider = table.k > table.n ? table.k : table.n;
    const std::int64_t floats = (table.k + 2 * wider) * static_cast<std::int64_t>(tokens);

    // The embedding Op carries no caller workspace; a process-wide scratch plane grows to the
    // largest lookup the engine has asked for and is reused across steps.
    static std::mutex scratch_mutex;
    static DeviceBuffer scratch;
    {
        std::lock_guard<std::mutex> lock(scratch_mutex);
        const std::size_t bytes = static_cast<std::size_t>(floats) * sizeof(float);
        if (scratch.bytes < bytes) {
            scratch = DeviceBuffer(bytes);
        }
    }
    float* g = static_cast<float*>(scratch.p);
    float* tmp = g + static_cast<std::int64_t>(table.k) * tokens;
    float* tmp2 = tmp + wider * tokens;

    detail::trellis::decode_row(ids, table, g, tokens, stream);
    if (plans.cols.kind == 1) {
        detail::trellis::rht_inplace(g, table.k, plans.cols.block, tokens, stream);
    } else if (plans.cols.kind == 2) {
        detail::trellis::hadfull_pass(g, g, tmp, tmp2, plans.cols,
                                      device_view(plans.cols, plans.perm_cols, plans.paley_cols), true,
                                      tokens, stream);
    }
    detail::trellis::pre_and_gather_bf16(g, table, rows, tokens, stream);
}

std::size_t trellis_banded_head_workspace_bytes(const Weight* const* bands, std::size_t band_count,
                                                std::int32_t max_tokens) {
    if (bands == nullptr || band_count == 0) {
        throw std::invalid_argument("trellis banded head: at least one band is required");
    }
    if (max_tokens <= 0) {
        throw std::invalid_argument("trellis banded head: token count must be positive");
    }
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < band_count; ++i) {
        if (bands[i] == nullptr) {
            throw std::invalid_argument("trellis banded head: null band weight");
        }
        bytes = std::max(bytes, trellis_linear_workspace_bytes(*bands[i], max_tokens));
    }
    return bytes;
}

void trellis_banded_head(const Tensor& x, const Weight* const* bands, std::size_t band_count,
                         const Tensor& vocab_gather, Tensor& logits, std::int32_t max_tokens,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::size_t bytes = trellis_banded_head_workspace_bytes(bands, band_count, max_tokens);
    if (workspace.capacity() < bytes) {
        throw std::invalid_argument("trellis banded head: workspace is smaller than the bands need");
    }
    if (x.dtype != DType::BF16 || logits.dtype != DType::BF16 || vocab_gather.dtype != DType::I32) {
        throw std::invalid_argument("trellis banded head: x/logits must be BF16 and the map I32");
    }
    if (x.ne[2] != 1 || x.ne[3] != 1 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("trellis banded head: x and logits must be two-dimensional");
    }
    const int tokens = x.ne[1];
    if (tokens > max_tokens || logits.ne[1] != tokens) {
        throw std::invalid_argument("trellis banded head: token counts disagree or exceed capacity");
    }
    if (!x.is_contiguous() || !logits.is_contiguous() || !vocab_gather.is_contiguous()) {
        throw std::invalid_argument("trellis banded head: tensors must be contiguous");
    }
    if (!aligned_to(logits.data, 16)) {
        throw std::invalid_argument("trellis banded head: logits must be 16-byte aligned");
    }
    std::int64_t total_rows = 0;
    for (std::size_t i = 0; i < band_count; ++i) {
        validate_weight(*bands[i]);
        if (bands[i]->k != x.ne[0]) {
            throw std::invalid_argument("trellis banded head: bands disagree on K");
        }
        total_rows += bands[i]->n;
    }
    const std::int32_t vocabulary = logits.ne[0];
    if (vocabulary <= 0 || vocabulary > total_rows || vocab_gather.ne[0] != vocabulary) {
        throw std::invalid_argument("trellis banded head: the map must have one entry per logit row");
    }

    // The concatenated band plane is the head's size, not a step's: keep it in one growing scratch
    // buffer instead of charging it to the per-step arena.
    static std::mutex scratch_mutex;
    static DeviceBuffer scratch;
    const std::size_t concat_bytes =
        static_cast<std::size_t>(total_rows) * static_cast<std::size_t>(tokens) * sizeof(std::uint16_t);
    {
        std::lock_guard<std::mutex> lock(scratch_mutex);
        if (scratch.bytes < concat_bytes) {
            scratch = DeviceBuffer(concat_bytes);
        }
    }
    auto* concat = static_cast<unsigned char*>(scratch.p);

    std::int64_t offset = 0;
    for (std::size_t i = 0; i < band_count; ++i) {
        const Weight& band = *bands[i];
        Tensor slice(concat + static_cast<std::size_t>(offset) * tokens * sizeof(std::uint16_t),
                     DType::BF16, {band.n, tokens});
        trellis_linear(x, band, slice, max_tokens, workspace, stream);
        offset += band.n;
    }
    detail::trellis::gather_rows_bf16(concat, static_cast<const std::int32_t*>(vocab_gather.data),
                                      vocabulary, tokens, logits.data, stream);
}

} // namespace ninfer::ops
