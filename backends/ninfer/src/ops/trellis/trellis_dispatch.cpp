#include "core/arena.h"
#include "ninfer/ops/trellis.h"
#include "ops/trellis/trellis_kernels.cuh"
#include "ops/trellis/trellis_reference.h"
#include "ops/trellis/trellis_profile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
namespace profile = detail::trellis::profile;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// NINFER_TRACE_DUMP=<dir>: write one FP32 plane of the coded pipeline as
// `<dir>/<seq>_<stage>_n<N>.f32`, so a whole pass keeps every call orderable.
void dump_plane(const void* device, std::int64_t count, const char* stage, std::int64_t n,
                std::int64_t k, cudaStream_t stream) {
    const char* dir = std::getenv("NINFER_TRACE_DUMP");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    static std::atomic<int> sequence{0};
    const int seq = sequence.fetch_add(1);
    cudaStreamSynchronize(stream);
    std::vector<float> host(static_cast<std::size_t>(count));
    cudaMemcpy(host.data(), device, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%03d_%s_n%lld_k%lld.f32", dir, seq, stage,
                  static_cast<long long>(n), static_cast<long long>(k));
    if (std::FILE* file = std::fopen(path, "wb")) {
        std::fwrite(host.data(), sizeof(float), host.size(), file);
        std::fclose(file);
    }
    std::fprintf(stderr, "plane | %03d | %s | n %lld k %lld | max %.5f\n", seq, stage,
                 static_cast<long long>(n), static_cast<long long>(k),
                 static_cast<double>(host.empty() ? 0.0F : host[0]));
}

// NINFER_TRACE_GDNPLANE=1: dump a BF16 [rows, cols] plane as FP32 for the coded-projection
// staging diagnostics (parent plane before a gather, destination after it).
void dump_bf16_plane(const void* device, std::int64_t rows, std::int64_t cols, const char* tag,
                     cudaStream_t stream) {
    const char* dir = std::getenv("NINFER_TRACE_DUMP");
    const char* on  = std::getenv("NINFER_TRACE_GDNPLANE");
    if (dir == nullptr || dir[0] == '\0' || on == nullptr || on[0] == '\0' || on[0] == '0') {
        return;
    }
    static std::atomic<int> sequence{0};
    const int seq = sequence.fetch_add(1);
    cudaStreamSynchronize(stream);
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows * cols));
    cudaMemcpy(host.data(), device, host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    std::vector<float> values(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        std::uint32_t word = static_cast<std::uint32_t>(host[i]) << 16;
        std::memcpy(&values[i], &word, sizeof(float));
    }
    char path[512];
    std::snprintf(path, sizeof(path), "%s/bf-%03d-%s-%lldx%lld.f32", dir, seq, tag,
                  static_cast<long long>(rows), static_cast<long long>(cols));
    if (std::FILE* file = std::fopen(path, "wb")) {
        std::fwrite(values.data(), sizeof(float), values.size(), file);
        std::fclose(file);
    }
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

namespace {
std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }
} // namespace

std::size_t trellis_coded_workspace_bytes(std::int32_t n, std::int32_t k,
                                          std::int32_t max_tokens) {
    if (n <= 0 || k <= 0 || max_tokens <= 0) {
        throw std::invalid_argument("trellis workspace: shape must be positive");
    }
    const std::int64_t tok    = max_tokens;
    const std::int64_t wider  = k > n ? k : n;
    // Prefill prologue scratch: fp16 B-fragment plane padded to 8-token groups (one 128-block per
    // group holds 1024 halves) + per-token 2^-e row scales.
    const std::int64_t groups = (tok + 7) / 8;
    const std::int64_t xf_half = static_cast<std::int64_t>(k / 128) * groups * 1024;
    // Splits run for verify windows of up to sixteen tokens; larger prefill chunks stop growing.
    const std::int64_t pf_tok = tok < 16 ? tok : 16;
    // Every arena plane is 256-byte aligned, so round each plane's size up independently; the
    // packed raw byte count otherwise under-reserves and the last alloc throws bad_alloc.
    const std::size_t xa       = round256(static_cast<std::size_t>(static_cast<std::int64_t>(k) * tok * 4));
    const std::size_t yr       = round256(static_cast<std::size_t>(static_cast<std::int64_t>(n) * tok * 4));
    const std::size_t tmp      = round256(static_cast<std::size_t>(wider * tok * 4));
    const std::size_t split    = round256(static_cast<std::size_t>(
        static_cast<std::int64_t>(detail::trellis::kDecodeSplits) * n * 4));
    const std::size_t pf_split = round256(static_cast<std::size_t>(
        static_cast<std::int64_t>(detail::trellis::kPfMaxSplits) * n * pf_tok * 4));
    const std::size_t xs       = round256(static_cast<std::size_t>(tok * 4));
    const std::size_t xf       = round256(static_cast<std::size_t>(xf_half * 2));
    return xa + yr + tmp + tmp + split + pf_split + xs + xf;
}

std::size_t trellis_linear_workspace_bytes(const Weight& w, std::int32_t max_tokens) {
    validate_weight(w);
    return trellis_coded_workspace_bytes(w.n, w.k, max_tokens);
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
    // NINFER_PROFILE=1: the whole linear and each of its stages, grouped by shape and by a token
    // bucket, so a slow shape or stage shows up without a profiler attached.
    const char* bucket = tokens == 1 ? "T1" : (tokens <= 16 ? "T16" : "T64");
    const std::string prof_key = "lin n" + std::to_string(w.n) + " k" + std::to_string(w.k) +
                                 " b" + std::to_string(detail::trellis::bits_of(w.qtype)) + " " + bucket;
    profile::Scope prof_whole(prof_key, stream);
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
    // Prefill prologue scratch: fp16 activation fragment plane (group-padded) and per-token row scale.
    const std::int64_t pf_groups = (tokens + 7) / 8;
    const std::size_t xf_bytes = static_cast<std::size_t>(w.k / 128) * pf_groups * 1024 *
                                 sizeof(std::uint16_t);
    DeviceSpan xf_span = workspace.alloc_bytes(xf_bytes, 256);
    Tensor xs = workspace.alloc(DType::FP32, {tokens});
    // One token cannot fill the GPU with N/16 warps, so the product splits K across blocks and
    // reduces; the accumulation planes come from the same arena.
    Tensor partials;
    const bool split_decode = tokens == 1 && !detail::trellis::trellis_pf_decode_enabled() &&
                              w.k >= 32 * detail::trellis::kDecodeSplits;
    if (split_decode) {
        partials = workspace.alloc(DType::FP32, {w.n * detail::trellis::kDecodeSplits});
    }
    // Multi-token coded MMA: small N runs K in parallel slices and reduces their FP32 planes.
    Tensor pf_partials;
    if (tokens > 1 && tokens <= 16 && w.k % 128 == 0) {
        pf_partials = workspace.alloc(DType::FP32,
                                      {w.n, tokens, detail::trellis::kPfMaxSplits});
    }

    // A kind-1 column side is a plain RHT, so the gather/scale folds into the same kernel. Kind 2
    // needs the gathered plane for its shard/paley passes and keeps the two stages.
    const bool fused_col = plans.cols.kind == 1;
    {
        profile::Scope stage(prof_key + (fused_col ? " gather+colside" : " gather"), stream);
        if (fused_col) {
            detail::trellis::gather_scale_rht(x, w, static_cast<float*>(xa.data), tokens,
                                              plans.cols.block, stream);
        } else {
            detail::trellis::gather_scale(x, w, static_cast<float*>(xa.data), tokens, w.k, stream);
        }
        dump_plane(xa.data, static_cast<std::int64_t>(w.k) * tokens, "gather", w.n, w.k, stream);
    }
    if (plans.cols.kind == 2) {
        profile::Scope stage(prof_key + " colside", stream);
        apply_column_side(w, plans, static_cast<float*>(xa.data), static_cast<float*>(tmp.data),
                      static_cast<float*>(tmp2.data), tokens, stream);
    }
    dump_plane(xa.data, static_cast<std::int64_t>(w.k) * tokens, "xrot", w.n, w.k, stream);
    {
        profile::Scope stage(prof_key + " product", stream);
    if (split_decode) {
        if (detail::trellis::trellis_gemv_enabled()) {
            detail::trellis::tile_product_split1_gemv(
                static_cast<const float*>(xa.data), w, static_cast<float*>(yr.data),
                static_cast<float*>(partials.data), static_cast<std::uint16_t*>(xf_span.data),
                static_cast<float*>(xs.data), stream);
        } else {
            detail::trellis::tile_product_split1(static_cast<const float*>(xa.data), w,
                                                 static_cast<float*>(yr.data),
                                                 static_cast<float*>(partials.data), stream);
        }
    } else {
        detail::trellis::tile_product(static_cast<const float*>(xa.data), w,
                                      static_cast<float*>(yr.data),
                                      pf_partials.data != nullptr
                                          ? static_cast<float*>(pf_partials.data)
                                          : nullptr,
                                      tokens,
                                      static_cast<std::uint16_t*>(xf_span.data),
                                      static_cast<float*>(xs.data), stream);
    }
    }
    dump_plane(yr.data, static_cast<std::int64_t>(w.n) * tokens, "tile", w.n, w.k, stream);
    float* yrot = static_cast<float*>(yr.data);
    if (plans.rows.kind != 0) {
        profile::Scope stage(prof_key + " rowside", stream);
    if (plans.rows.kind == 1) {
        detail::trellis::rht_inplace(yrot, w.n, plans.rows.block, tokens, stream);
    } else if (plans.rows.kind == 2) {
        detail::trellis::hadfull_pass(yrot, yrot, static_cast<float*>(tmp.data),
                                      static_cast<float*>(tmp2.data), plans.rows,
                                      device_view(plans.rows, plans.perm_rows, plans.paley_rows), true,
                                      tokens, stream);
    }
    }
    {
        profile::Scope stage(prof_key + " post", stream);
    detail::trellis::post_and_gather(yrot, w, y, tokens, stream);
    }
    dump_plane(yr.data, static_cast<std::int64_t>(w.n) * tokens, "post", w.n, w.k, stream);
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
    static std::vector<DeviceBuffer> retired;
    {
        std::lock_guard<std::mutex> lock(scratch_mutex);
        const std::size_t bytes = static_cast<std::size_t>(floats) * sizeof(float);
        if (scratch.bytes < bytes) {
            // A captured CUDA graph bakes this pointer, so a later, larger lookup must not free the
            // block an existing graph may still replay: retire it instead.
            retired.push_back(std::move(scratch));
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
    detail::trellis::pre_and_gather_bf16(g, table, ids, rows, tokens, stream);
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
    static std::vector<DeviceBuffer> retired;
    const std::size_t concat_bytes =
        static_cast<std::size_t>(total_rows) * static_cast<std::size_t>(tokens) * sizeof(std::uint16_t);
    {
        std::lock_guard<std::mutex> lock(scratch_mutex);
        if (scratch.bytes < concat_bytes) {
            // A captured CUDA graph bakes this pointer: a wider prefill must not free the block a
            // decode graph replays, so retire it rather than returning it to the allocator.
            retired.push_back(std::move(scratch));
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
        // NINFER_TRACE_LOGITS=1: the band's own top rows, so a band-level divergence from an
        // off-line decode is visible before the vocabulary gather hides it.
        static const bool trace_bands = [] {
            const char* value = std::getenv("NINFER_TRACE_LOGITS");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        if (trace_bands) {
            cudaStreamSynchronize(stream);
            std::vector<std::uint16_t> host(static_cast<std::size_t>(band.n));
            const void* source = static_cast<const std::uint8_t*>(slice.data) +
                                 static_cast<std::size_t>(tokens - 1) *
                                     static_cast<std::size_t>(band.n) * sizeof(std::uint16_t);
            cudaMemcpy(host.data(), source, host.size() * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost);
            std::vector<float> values(host.size());
            for (std::size_t r = 0; r < host.size(); ++r) {
                std::uint32_t word = static_cast<std::uint32_t>(host[r]) << 16;
                std::memcpy(&values[r], &word, sizeof(float));
            }
            const char* dump_dir = std::getenv("NINFER_TRACE_DUMP");
            if (dump_dir != nullptr && dump_dir[0] != '\0') {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/band%02zu.f32", dump_dir, i);
                if (std::FILE* file = std::fopen(path, "wb")) {
                    std::fwrite(values.data(), sizeof(float), values.size(), file);
                    std::fclose(file);
                }
            }
            std::vector<std::size_t> order(values.size());
            for (std::size_t r = 0; r < order.size(); ++r) { order[r] = r; }
            std::partial_sort(order.begin(), order.begin() + 3, order.end(),
                              [&](std::size_t a, std::size_t b) { return values[a] > values[b]; });
            std::fprintf(stderr, "band | %zu | n %d | top %zu=%.5f %zu=%.5f %zu=%.5f\n", i,
                         static_cast<int>(band.n), order[0], static_cast<double>(values[order[0]]),
                         order[1], static_cast<double>(values[order[1]]), order[2],
                         static_cast<double>(values[order[2]]));
        }
        offset += band.n;
    }
    std::vector<std::int32_t> band_rows(band_count);
    for (std::size_t i = 0; i < band_count; ++i) {
        band_rows[i] = static_cast<std::int32_t>(bands[i]->n);
    }
    detail::trellis::gather_rows_bf16(concat, static_cast<const std::int32_t*>(vocab_gather.data),
                                      vocabulary, tokens, band_rows.data(),
                                      static_cast<std::int32_t>(band_count), logits.data, stream);
    if (const char* dir = std::getenv("NINFER_TRACE_DUMP"); dir != nullptr && dir[0] != '\0') {
        cudaStreamSynchronize(stream);
        std::vector<std::uint16_t> host(static_cast<std::size_t>(vocabulary) *
                                        static_cast<std::size_t>(tokens));
        cudaMemcpy(host.data(), logits.data, host.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost);
        std::vector<float> values(host.size());
        for (std::size_t i = 0; i < host.size(); ++i) {
            std::uint32_t word = static_cast<std::uint32_t>(host[i]) << 16;
            std::memcpy(&values[i], &word, sizeof(float));
        }
        char path[512];
        std::snprintf(path, sizeof(path), "%s/headlogits-%lldx%d.f32",
                      std::getenv("NINFER_TRACE_DUMP"), static_cast<long long>(vocabulary), tokens);
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(values.data(), sizeof(float), values.size(), file);
            std::fclose(file);
        }
    }
}

std::size_t trellis_input_proj_workspace_bytes(const TrellisProjectionPart* parts,
                                               std::size_t part_count, std::int32_t max_tokens) {
    if (parts == nullptr || part_count == 0) {
        throw std::invalid_argument("trellis input proj: at least one part is required");
    }
    std::size_t bytes = 0;
    std::int64_t widest = 0;
    for (std::size_t i = 0; i < part_count; ++i) {
        if (parts[i].weight == nullptr) {
            throw std::invalid_argument("trellis input proj: null part weight");
        }
        widest = std::max<std::int64_t>(widest, parts[i].weight->n);
        bytes = std::max(bytes, trellis_linear_workspace_bytes(*parts[i].weight, max_tokens));
    }
    // The parent plane the parts gather from is carved from the same arena, ahead of the widest
    // parent's own linear scratch. A process-wide buffer instead would be reallocated by a later,
    // wider round and leave every captured CUDA graph pointing at freed memory.
    const std::size_t plane_bytes =
        static_cast<std::size_t>(widest) * static_cast<std::size_t>(max_tokens) *
        sizeof(std::uint16_t);
    return bytes + plane_bytes + 256;  // the arena aligns each allocation to 256 bytes
}

void trellis_input_proj(const Tensor& x, const TrellisProjectionPart* parts, std::size_t part_count,
                        Tensor* const* outputs, std::size_t output_count, std::int32_t max_tokens,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    const std::size_t bytes = trellis_input_proj_workspace_bytes(parts, part_count, max_tokens);
    if (workspace.capacity() < bytes) {
        throw std::invalid_argument("trellis input proj: workspace is smaller than a parent needs");
    }
    if (x.dtype != DType::BF16 || outputs == nullptr || output_count == 0) {
        throw std::invalid_argument("trellis input proj: BF16 activation and at least one output");
    }
    const int tokens = x.ne[1];
    if (tokens > max_tokens || !x.is_contiguous()) {
        throw std::invalid_argument("trellis input proj: tokens exceed capacity or x is not dense");
    }
    const std::int32_t k = x.ne[0];

    // Distinct parents are projected once, in first-appearance order; the widest decides the
    // parent plane, which lives in this round's arena so a captured graph keeps a stable pointer.
    std::int64_t widest = 0;
    for (std::size_t i = 0; i < part_count; ++i) {
        const Weight& w = *parts[i].weight;
        validate_weight(w);
        if (w.k != k) {
            throw std::invalid_argument("trellis input proj: part K disagrees with the activation");
        }
        widest = std::max<std::int64_t>(widest, w.n);
    }
    auto plane_scope = workspace.scope();
    Tensor plane_tensor =
        workspace.alloc(DType::BF16, {static_cast<std::int32_t>(widest), tokens});
    auto* plane = static_cast<unsigned char*>(plane_tensor.data);

    std::vector<const Weight*> done;
    for (std::size_t i = 0; i < part_count; ++i) {
        const TrellisProjectionPart& part = parts[i];
        if (part.output < 0 || static_cast<std::size_t>(part.output) >= output_count ||
            outputs[part.output] == nullptr) {
            throw std::invalid_argument("trellis input proj: part names an absent output");
        }
        Tensor& dst = *outputs[part.output];
        if (dst.dtype != DType::BF16 || !dst.is_contiguous() || dst.ne[1] != tokens ||
            !aligned_to(dst.data, 16)) {
            throw std::invalid_argument("trellis input proj: outputs must be dense BF16 [rows,T]");
        }
        if (part.rows <= 0 || part.dst_row < 0 || part.dst_row + part.rows > dst.ne[0]) {
            throw std::invalid_argument("trellis input proj: part rows fall outside its output");
        }
        const bool projected =
            std::find(done.begin(), done.end(), part.weight) != done.end();
        {
            const char* on = std::getenv("NINFER_TRACE_GDNPLANE");
            if (on != nullptr && on[0] != '\0' && on[0] != '0') {
                std::fprintf(stderr,
                             "part | %zu | wn %lld | rows %d | dst_row %d | out %d/%lld | grp %d "
                             "stride %d origin %d | map %d | proj %d\n",
                             i, static_cast<long long>(part.weight->n), part.rows, part.dst_row,
                             part.output, static_cast<long long>(dst.ne[0]), part.src_group,
                             part.src_stride, part.src_origin, part.row_map != nullptr ? 1 : 0,
                             projected ? 1 : 0);
            }
        }
        if (!projected) {
            Tensor parent(plane, DType::BF16, {part.weight->n, tokens});
            trellis_linear(x, *part.weight, parent, max_tokens, workspace, stream);
            done.push_back(part.weight);
            if (std::getenv("NINFER_TRACE_GDNPLANE") != nullptr) {
                dump_bf16_plane(plane, part.weight->n, tokens, "parent", stream);
            }
        }
        detail::trellis::gather_rows_into_bf16(plane, part.row_map, part.rows, part.dst_row,
                                               part.src_group, part.src_stride, part.src_origin,
                                               tokens, part.weight->n, dst.ne[0], dst.data, stream);
        if (part.rows == 6144) {
            dump_bf16_plane(dst.data, dst.ne[0], tokens, "dst", stream);
        }
        if (std::getenv("NINFER_TRACE_PROBE") != nullptr && dst.ne[0] == 10240) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "step%zu", i);
            dump_bf16_plane(dst.data, dst.ne[0], tokens, tag, stream);
            std::snprintf(tag, sizeof(tag), "pl%zu", i);
            dump_bf16_plane(plane, part.weight->n, tokens, tag, stream);
            std::fprintf(stderr, "step | %zu | plane %p | dst %p\n", i,
                         static_cast<const void*>(plane), static_cast<const void*>(dst.data));
        }
        if (part.rows == 6144 && part.weight->n == 10240 &&
            std::getenv("NINFER_TRACE_PROBE") != nullptr) {
            auto* base = static_cast<std::uint16_t*>(dst.data);
            const std::int64_t span = static_cast<std::int64_t>(dst.ne[0]) * tokens;
            for (std::int64_t at : {static_cast<std::int64_t>(131000), static_cast<std::int64_t>(200000),
                                    span - 1000}) {
                const std::uint16_t marker = 0x7F80;
                cudaMemcpy(base + at, &marker, sizeof(marker), cudaMemcpyHostToDevice);
                cudaStreamSynchronize(stream);
                std::uint16_t back = 0;
                cudaMemcpy(&back, base + at, sizeof(back), cudaMemcpyDeviceToHost);
                std::fprintf(stderr, "probe | dst %p rows %lld span %lld at %lld -> %04x\n",
                             static_cast<const void*>(base), static_cast<long long>(dst.ne[0]),
                             static_cast<long long>(span),
                             static_cast<long long>(at), static_cast<unsigned>(back));
            }
        }
    }
    if (std::getenv("NINFER_TRACE_PROBE") != nullptr && outputs[0]->ne[0] == 10240) {
        // Ground truth: compare the finished qkv against the parent plane on the host.
        const Tensor& out = *outputs[0];
        const std::int64_t n = out.ne[0], span = n * tokens;
        std::vector<std::uint16_t> got(static_cast<std::size_t>(span)),
            plane_copy(static_cast<std::size_t>(span));
        cudaMemcpy(got.data(), out.data, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(plane_copy.data(), plane, plane_copy.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost);
        auto diff = [&](std::int64_t got_row, std::int64_t src_row, std::int64_t rows) {
            double num = 0.0, den = 0.0;
            for (std::int64_t r = 0; r < rows; ++r) {
                for (int t = 0; t < tokens; ++t) {
                    const auto a = static_cast<float>(got[static_cast<std::size_t>((got_row + r) * tokens + t)]);
                    const auto b = static_cast<float>(plane_copy[static_cast<std::size_t>((src_row + r) * tokens + t)]);
                    num += static_cast<double>((a - b) * (a - b));
                    den += static_cast<double>(b * b);
                }
            }
            return den > 0.0 ? std::sqrt(num / den) : -1.0;
        };
        std::fprintf(stderr,
                     "qkvprobe | k-vs-q %.4f | k-vs-plane-k %.4f | v-vs-plane-v %.4f | "
                     "v-vs-plane-head %.4f\n",
                     diff(2048, 0, 2048), diff(2048, 2048, 2048), diff(4096, 4096, 6144),
                     diff(4096, 0, 6144));
    }
}

} // namespace ninfer::ops
