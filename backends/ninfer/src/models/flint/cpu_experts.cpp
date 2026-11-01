#include "models/flint/cpu_experts.h"

#include "core/cpu_spin_pool.h"
#include "core/expert_split.h"
#include "models/flint/kernels.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"
// After the bridge, which brings the CUDA host definitions fixed_point.cuh's qualifiers need.
#include "ops/common/fixed_point.cuh"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::models::flint {
namespace {

namespace gguf = ops::gguf;

// Rows per task are a multiple of 16 (64 bytes of FP32 outputs). A [gate; up] task takes rows of
// one expert's gate or up, a down task the same rows of every CPU expert, so that no two tasks add
// to the same sums.
constexpr int kRowQuantum = 16;

template <class Job, class Side>
void run(CpuSpinPool* pool, std::size_t count, Job&& job, Side&& side) {
    if (pool) { return pool->run(count, job, side); }
    side();
    for (std::size_t i = 0; i < count; ++i) { job(i); }
}

template <class Job>
void run(CpuSpinPool* pool, std::size_t count, Job&& job) {
    run(pool, count, job, [] {});
}

// kernels::quantize_pairs' guard of one BF16 middle row: its largest sum of 32 values, added as a
// warp adds them, past kF16SumLimit scales the row by 2^-k. Returns k.
int guard_middle(std::uint16_t* row, int width) {
    float largest = 0;
    float lanes[32];
    for (int g = 0; g < width / 32; ++g) {
        for (int i = 0; i < 32; ++i) {
            lanes[i] = std::bit_cast<float>(std::uint32_t(row[32 * g + i]) << 16);
        }
        largest = std::max(largest, std::abs(ops::gguf_cpu_detail::butterfly(lanes)));
    }
    const int k = kernels::f16_sum_exponent(largest);
    if (k != 0) {
        for (int i = 0; i < width; ++i) {
            const float value = std::bit_cast<float>(std::uint32_t(row[i]) << 16);
            row[i]            = ops::gguf_cpu_detail::float_to_bf16(std::ldexp(value, -k));
        }
    }
    return k;
}

} // namespace

CpuExperts::CpuExperts(std::vector<CpuExpertLayer> layers, int hidden, int width, int top_k)
    : layers_(std::move(layers)), hidden_(hidden), width_(width), top_k_(top_k),
      backend_(ops::gguf_cpu_backend_selected(ops::GgufCpuBackend::Automatic)) {
    if (hidden <= 0 || width <= 0 || hidden % 32 != 0 || width % 32 != 0 || top_k <= 0) {
        throw std::invalid_argument("Flint CPU experts: invalid expert geometry");
    }
}

bool CpuExperts::computes(const CpuExpertLayer& layer, int hidden, int width) noexcept {
    const auto fits = [](QType format, int k) {
        return ops::gguf_cpu_supported(format) && k % gguf_block_shape(format).elements == 0;
    };
    return fits(layer.gate_up, hidden) && fits(layer.down, width);
}

std::size_t CpuExperts::activation_bytes(int tokens) const {
    return gguf::vector_activation_bytes(hidden_, tokens);
}
std::size_t CpuExperts::weight_bytes(int tokens) const { return std::size_t(tokens) * top_k_ * 4; }
std::size_t CpuExperts::input_bytes() const {
    return activation_bytes(kMaxTokens) + weight_bytes(kMaxTokens);
}
std::size_t CpuExperts::product_bytes() const { return std::size_t(kMaxTokens) * hidden_ * 8; }

bool CpuExperts::supports(std::size_t layer) const {
    return layer < layers_.size() && computes(layers_[layer], hidden_, width_);
}

ExpertCpuProducts::Done CpuExperts::compute(const Work& work) {
    using Clock                 = std::chrono::steady_clock;
    const auto started          = Clock::now();
    const CpuExpertLayer& layer = layers_.at(work.layer);
    const int pairs  = int(work.ids.size());
    const int tokens = pairs / top_k_;
    if (pairs % top_k_ != 0 || tokens < 1 || tokens > kMaxTokens) {
        throw std::invalid_argument("Flint CPU experts: a call of 1 to 8 tokens");
    }
    const auto* base    = reinterpret_cast<const std::uint8_t*>(work.inputs);
    const auto* codes   = reinterpret_cast<const std::int8_t*>(base);
    const auto* ds      = reinterpret_cast<const std::uint16_t*>(
        base + gguf::vector_activation_scales_offset(hidden_, tokens));
    const auto* weights = reinterpret_cast<const float*>(base + activation_bytes(tokens));
    auto* sums          = reinterpret_cast<std::uint64_t*>(work.products);
    std::memset(sums, 0, std::size_t(tokens) * hidden_ * 8);

    const std::size_t count = work.experts.size();
    if (jobs_.size() < count) { jobs_.resize(count); }
    for (std::size_t i = 0; i < count; ++i) {
        Job& job   = jobs_[i];
        job.expert = work.experts[i];
        job.pairs.clear();
        job.tokens.clear();
        for (int p = 0; p < pairs; ++p) {
            if (work.ids[std::size_t(p)] != job.expert) { continue; }
            job.pairs.push_back(p);
            job.tokens.push_back(p / top_k_);
        }
        if (job.pairs.empty() || work.blobs[i].size() < 2) {
            throw std::invalid_argument("Flint CPU experts: an expert without pairs or blobs");
        }
        job.gate_up = reinterpret_cast<const std::uint8_t*>(work.blobs[i][0]);
        job.down    = reinterpret_cast<const std::uint8_t*>(work.blobs[i][1]);
        if (job.input.k() != hidden_) { job.input = ops::GgufCpuActivation(kMaxTokens, hidden_); }
        if (job.middle.k() != width_) { job.middle = ops::GgufCpuActivation(kMaxTokens, width_); }
        job.input.load_q8_1(layer.gate_up, codes, ds, job.tokens);
        const std::size_t n = job.pairs.size();
        job.middle.reset(layer.down, int(n));
        job.gate.resize(n * width_);
        job.up.resize(n * width_);
        job.middle_bf16.resize(n * width_);
        job.exponent.assign(n, 0);
        job.rows.resize(n * hidden_);
    }

    // Gate and up rows: the stored [gate; up] matrix's first and second `width` rows, while the
    // calling thread runs work.side.
    std::vector<RowPart> parts;
    double cost = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double one = ExpertSplit::cpu_work(1.0, int(jobs_[i].pairs.size()));
        parts.push_back({width_, one});
        parts.push_back({width_, one});
        cost += one;
    }
    const std::uint32_t threads = work.pool ? work.pool->threads() : 1;
    auto tasks                  = row_tasks(parts, threads, kRowQuantum);
    const auto side             = [&] {
        if (work.side) { work.side(); }
    };
    run(
        work.pool, tasks.size(),
        [&](std::size_t index) {
            const RowTask& task = tasks[index];
            Job& job            = jobs_[task.part / 2];
            const bool is_up    = (task.part & 1U) != 0;
            const ops::GgufCpuMatrix w{
                layer.gate_up,
                job.gate_up + (is_up ? std::size_t(width_) * layer.gate_up_row_bytes : 0),
                layer.gate_up_row_bytes, width_, hidden_};
            ops::gguf_cpu_rows(w, job.input, task.begin, task.end,
                               (is_up ? job.up : job.gate).data(), width_, backend_);
        },
        side);
    const auto rows_done = work.pool ? work.pool->finished() : Clock::now();
    const auto middle    = Clock::now();
    // The BF16 middle, as the GPU stores it before quantizing it for the down rows, a pair a task.
    const bool guard = gguf::vector_activation_f16_sums(ops::detail::gguf_type(layer.down));
    std::vector<std::pair<std::uint32_t, int>> columns;
    for (std::size_t i = 0; i < count; ++i) {
        for (int j = 0; j < int(jobs_[i].pairs.size()); ++j) { columns.emplace_back(i, j); }
    }
    run(work.pool, columns.size(), [&](std::size_t index) {
        const auto [i, j]    = columns[index];
        Job& job             = jobs_[i];
        const std::size_t at = std::size_t(j) * width_;
        ops::gguf_cpu_swiglu_bf16(job.gate.data() + at, job.up.data() + at, std::size_t(width_),
                                 job.middle_bf16.data() + at, backend_);
        if (guard) {
            job.exponent[std::size_t(j)] = guard_middle(job.middle_bf16.data() + at, width_);
        }
        job.middle.quantize_bf16(j, job.middle_bf16.data() + at, 0, job.middle.groups());
    });
    const RowPart down{hidden_, cost};
    tasks = row_tasks(std::span(&down, 1), threads, kRowQuantum);
    run(work.pool, tasks.size(), [&](std::size_t index) {
        const int begin = tasks[index].begin;
        const int end   = tasks[index].end;
        for (std::size_t i = 0; i < count; ++i) {
            Job& job = jobs_[i];
            const ops::GgufCpuMatrix w{layer.down, job.down, layer.down_row_bytes, hidden_, width_};
            ops::gguf_cpu_rows(w, job.middle, begin, end, job.rows.data(), std::size_t(hidden_), backend_);
            for (std::size_t j = 0; j < job.pairs.size(); ++j) {
                const float weight = std::ldexp(weights[job.pairs[j]], job.exponent[j]);
                const float* row   = job.rows.data() + j * hidden_;
                std::uint64_t* sum = sums + std::size_t(job.tokens[j]) * hidden_;
                // Two's complement: the GPU's unsigned atomic additions, modulo 2^64.
                for (int r = begin; r < end; ++r) {
                    sum[r] += static_cast<std::uint64_t>(ops::fixed_product(row[r], weight));
                }
            }
        }
    });
    const auto done = Clock::now();
    return {
        std::size_t(tokens) * hidden_ * 8,
        std::chrono::duration<double, std::micro>(rows_done - started + (done - middle)).count()};
}

} // namespace ninfer::models::flint
