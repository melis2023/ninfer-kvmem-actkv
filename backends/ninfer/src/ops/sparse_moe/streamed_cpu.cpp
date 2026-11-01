#include "ops/sparse_moe/streamed_cpu.h"

#include "core/cpu_spin_pool.h"
#include "core/expert_split.h"
#include "ops/common/silu.h"
#include "ops/sparse_moe/streamed_cpu_detail.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace stream_cpu_detail {
namespace {

std::uint16_t load16(const std::uint8_t* p) noexcept {
    std::uint16_t value;
    std::memcpy(&value, p, 2);
    return value;
}
std::uint32_t load32(const std::uint8_t* p) noexcept {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

constexpr float kE2m1[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

} // namespace

float e4m3_to_float(std::uint8_t bits) noexcept {
    const unsigned exponent = (bits >> 3) & 15U, mantissa = bits & 7U;
    const float sign        = (bits & 0x80U) ? -1.0F : 1.0F;
    if (exponent == 15 && mantissa == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    if (exponent == 0) { return sign * std::ldexp(float(mantissa), -9); }
    return sign * std::ldexp(float(8 + mantissa), int(exponent) - 10);
}

std::int64_t nvfp4_scale_byte(int row, int block, int tiles_per_row) noexcept {
    const int m_tile    = row / 128;
    const int row_inner = row - m_tile * 128;
    return (std::int64_t(m_tile) * tiles_per_row + block / 4) * 512 + (row_inner & 31) * 16 +
           (row_inner >> 5) * 4 + (block & 3);
}

void decode_lane(const StreamCpuPlane& w, int row, int step, int lane, float (&out)[8]) noexcept {
    switch (w.format) {
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16: {
        const std::size_t groups = std::size_t(w.k / 64);
        const std::size_t group  = std::size_t(row) * groups + std::size_t(4 * step + lane / 8);
        const int in_group       = lane % 8;
        const std::uint32_t word = load32(w.codes + group * 32 + std::size_t(in_group) * 4);
        const float scale        = gguf_cpu_detail::half_to_float(load16(w.scales + group * 2));
        unsigned high = 0;
        if (w.format == QType::Q5_G64_FP16) { high = w.high[group * 8 + std::size_t(in_group)]; }
        if (w.format == QType::Q6_G64_FP16) {
            high = load16(w.high + group * 16 + std::size_t(in_group) * 2);
        }
        for (int k = 0; k < 8; ++k) {
            const unsigned nibble = (word >> (4 * k)) & 15U;
            int value;
            if (w.format == QType::Q4_G64_FP16) {
                value = int(nibble ^ 8U) - 8;
            } else if (w.format == QType::Q5_G64_FP16) {
                value = int((nibble | (((high >> k) & 1U) << 4)) ^ 16U) - 16;
            } else {
                value = int((nibble | (((high >> (2 * k)) & 3U) << 4)) ^ 32U) - 32;
            }
            out[k] = float(value) * scale;
        }
        return;
    }
    case QType::Q8_G32_FP16: {
        const std::size_t groups = std::size_t(w.k / 32);
        const std::size_t group  = std::size_t(row) * groups + std::size_t(8 * step + lane / 4);
        const float scale        = gguf_cpu_detail::half_to_float(load16(w.scales + group * 2));
        const auto* codes = reinterpret_cast<const std::int8_t*>(w.codes) + group * 32 +
                            std::size_t(lane % 4) * 8;
        for (int k = 0; k < 8; ++k) { out[k] = float(codes[k]) * scale; }
        return;
    }
    case QType::NVFP4: {
        const int group          = 4 * step + lane / 8;
        const int in_group       = lane % 8;
        const std::uint32_t word = load32(w.codes + std::size_t(row) * std::size_t(w.k / 2) +
                                          std::size_t(group) * 32 + std::size_t(in_group) * 4);
        const std::uint8_t scale = w.scales[nvfp4_scale_byte(row, group * 4 + in_group / 2, w.k / 64)];
        const float coefficient =
            e4m3_to_float(scale) * (1.0F / w.divisors[row / w.divisor_rows]);
        for (int k = 0; k < 8; ++k) {
            const unsigned code = (word >> (4 * k)) & 15U;
            const float value   = (code & 8U) ? -kE2m1[code & 7U] : kE2m1[code & 7U];
            out[k]              = value * coefficient;
        }
        return;
    }
    default: return;
    }
}

void rows_scalar(const StreamCpuPlane& w, const StreamCpuInput& x, int row_begin, int row_end,
                 const float* coefficients, std::span<float* const> y) {
    const int steps   = w.k / 256;
    const int columns = x.columns();
    std::vector<float> lanes(std::size_t(columns) * 32);
    for (int row = row_begin; row < row_end; ++row) {
        std::fill(lanes.begin(), lanes.end(), 0.0F);
        for (int step = 0; step < steps; ++step) {
            for (int lane = 0; lane < 32; ++lane) {
                float weights[8];
                decode_lane(w, row, step, lane, weights);
                for (int t = 0; t < columns; ++t) {
                    const float* values = x.lanes(t) + std::size_t(step) * 256 + lane;
                    float& acc          = lanes[std::size_t(t) * 32 + std::size_t(lane)];
                    for (int k = 0; k < 8; ++k) { acc = std::fma(weights[k], values[k * 32], acc); }
                }
            }
        }
        for (int t = 0; t < columns; ++t) {
            const float dot = gguf_cpu_detail::butterfly(lanes.data() + std::size_t(t) * 32);
            y[std::size_t(t)][row] = coefficients ? coefficients[t] * dot : dot;
        }
    }
}

} // namespace stream_cpu_detail

bool stream_cpu_supported(QType gate_up, QType down) noexcept {
    return (gate_up == QType::Q4_G64_FP16 &&
            (down == QType::Q5_G64_FP16 || down == QType::Q6_G64_FP16)) ||
           (gate_up == QType::Q8_G32_FP16 && down == QType::Q8_G32_FP16) ||
           (gate_up == QType::NVFP4 && down == QType::NVFP4);
}

StreamCpuInput::StreamCpuInput(int capacity, int k)
    : capacity_(capacity), k_(k), lanes_(std::size_t(std::max(capacity, 0)) * std::size_t(std::max(k, 0))) {
    if (capacity <= 0 || k <= 0 || k % 256 != 0) {
        throw std::invalid_argument("streamed SparseMoe CPU input: k must be a positive multiple of 256");
    }
}

void StreamCpuInput::load_bf16(int column, const std::uint16_t* values) {
    if (column < 0 || column >= capacity_) { throw std::out_of_range("streamed SparseMoe CPU input column"); }
    float* lanes = lanes_.data() + std::size_t(column) * k_;
    for (int step = 0; step < k_ / 256; ++step) {
        for (int lane = 0; lane < 32; ++lane) {
            for (int k = 0; k < 8; ++k) {
                const std::uint32_t bits = std::uint32_t(values[step * 256 + lane * 8 + k]) << 16;
                lanes[step * 256 + k * 32 + lane] = std::bit_cast<float>(bits);
            }
        }
    }
    if (column >= columns_) { columns_ = column + 1; }
}

void StreamCpuInput::load_fp32(int column, const float* values) {
    if (column < 0 || column >= capacity_) { throw std::out_of_range("streamed SparseMoe CPU input column"); }
    float* lanes = lanes_.data() + std::size_t(column) * k_;
    for (int step = 0; step < k_ / 256; ++step) {
        for (int lane = 0; lane < 32; ++lane) {
            for (int k = 0; k < 8; ++k) {
                lanes[step * 256 + k * 32 + lane] = values[step * 256 + lane * 8 + k];
            }
        }
    }
    if (column >= columns_) { columns_ = column + 1; }
}

void StreamCpuInput::set_columns(int columns) {
    if (columns < 0 || columns > capacity_) {
        throw std::out_of_range("streamed SparseMoe CPU input columns");
    }
    columns_ = columns;
}

void stream_cpu_rows(const StreamCpuPlane& w, const StreamCpuInput& x, int row_begin, int row_end,
                     const float* coefficients, std::span<float* const> y, GgufCpuBackend backend) {
    if (w.k != x.k() || w.k % 256 != 0 || row_begin < 0 || row_end > w.rows || row_begin > row_end ||
        y.size() < std::size_t(x.columns()) ||
        (w.format == QType::NVFP4 && (w.divisors == nullptr || w.divisor_rows <= 0))) {
        throw std::invalid_argument("streamed SparseMoe CPU rows: invalid operands");
    }
#if defined(NINFER_CPU_EXPERT_X86)
    if (gguf_cpu_backend_selected(backend) == GgufCpuBackend::Avx2) {
        stream_cpu_detail::rows_avx2(w, x, row_begin, row_end, coefficients, y);
        return;
    }
#else
    (void)backend;
#endif
    stream_cpu_detail::rows_scalar(w, x, row_begin, row_end, coefficients, y);
}

void stream_cpu_swiglu(const float* gate, const float* up, std::size_t count, float* out) noexcept {
    for (std::size_t i = 0; i < count; ++i) { out[i] = ops::portable_silu(gate[i]) * up[i]; }
}

namespace {

constexpr int kHidden       = 2048;
constexpr int kIntermediate = 512;
constexpr int kTopK         = 8;
constexpr int kMaxTokens    = 46; // kSparseMoeSmallTMax: the widest streamed Small-T call
// Rows per task are a multiple of 16 (64 bytes of FP32 outputs).
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

StreamCpuPlane plane(QType format, const std::vector<const std::byte*>& blobs,
                     const ExpertAlias* aliases, int rows, int k, int divisor_rows) {
    const auto at = [&](int index) -> const std::uint8_t* {
        const auto& alias = aliases[index];
        if (alias.blob >= blobs.size()) { throw std::invalid_argument("streamed SparseMoe CPU: missing blob"); }
        return reinterpret_cast<const std::uint8_t*>(blobs[alias.blob]) + alias.offset;
    };
    StreamCpuPlane out;
    out.format = format;
    out.codes  = at(0);
    out.high   = format == QType::Q5_G64_FP16 || format == QType::Q6_G64_FP16 ? at(1) : nullptr;
    out.scales = at(2);
    if (format == QType::NVFP4) {
        out.divisors     = reinterpret_cast<const float*>(at(3));
        out.divisor_rows = divisor_rows;
    }
    out.rows = rows;
    out.k    = k;
    return out;
}

} // namespace

SparseMoeStreamCpu::SparseMoeStreamCpu(std::vector<StreamCpuLayer> layers, GgufCpuBackend backend)
    : layers_(std::move(layers)), backend_(gguf_cpu_backend_selected(backend)) {}

bool SparseMoeStreamCpu::computes(const StreamCpuLayer& layer) noexcept {
    return stream_cpu_supported(layer.gate_up, layer.down);
}

std::size_t SparseMoeStreamCpu::input_bytes() noexcept {
    return std::size_t(kMaxTokens) * (kHidden * 2 + kTopK * 4);
}

std::size_t SparseMoeStreamCpu::product_bytes() noexcept {
    return std::size_t(kMaxTokens) * kTopK * kHidden * 4;
}

const char* SparseMoeStreamCpu::backend_name() const noexcept {
    return gguf_cpu_backend_name(backend_);
}

bool SparseMoeStreamCpu::supports(std::size_t layer) const {
    return layer < layers_.size() && computes(layers_[layer]);
}

ExpertCpuProducts::Done SparseMoeStreamCpu::compute(const Work& work) {
    using Clock                 = std::chrono::steady_clock;
    const auto started          = Clock::now();
    const StreamCpuLayer& layer = layers_.at(work.layer);
    const int pairs             = int(work.ids.size());
    const int tokens            = pairs / kTopK;
    if (pairs % kTopK != 0 || tokens < 1 || tokens > kMaxTokens) {
        throw std::invalid_argument("streamed SparseMoe CPU: a call of 1 to 46 tokens");
    }
    const auto* x     = reinterpret_cast<const std::uint16_t*>(work.inputs);
    const auto* alpha = reinterpret_cast<const float*>(work.inputs + std::size_t(tokens) * kHidden * 2);
    auto* products    = reinterpret_cast<float*>(work.products);

    // Each CPU pair's product row, in the order of the pairs.
    std::vector<int> row_of(std::size_t(pairs), -1);
    int rows = 0;
    for (int p = 0; p < pairs; ++p) {
        if (std::find(work.experts.begin(), work.experts.end(), work.ids[std::size_t(p)]) !=
            work.experts.end()) {
            row_of[std::size_t(p)] = rows++;
        }
    }
    const std::size_t count = work.experts.size();
    if (jobs_.size() < count) { jobs_.resize(count); }
    for (std::size_t i = 0; i < count; ++i) {
        Job& job   = jobs_[i];
        job.expert = work.experts[i];
        job.pairs.clear();
        for (int p = 0; p < pairs; ++p) {
            if (work.ids[std::size_t(p)] == job.expert) { job.pairs.push_back(p); }
        }
        if (job.pairs.empty() || work.blobs[i].empty()) {
            throw std::invalid_argument("streamed SparseMoe CPU: an expert without pairs or blobs");
        }
        job.gate_up = plane(layer.gate_up, work.blobs[i], layer.aliases.data(), 2 * kIntermediate,
                            kHidden, kIntermediate);
        job.down    = plane(layer.down, work.blobs[i], layer.aliases.data() + 4, kHidden,
                            kIntermediate, kHidden);
        const int n = int(job.pairs.size());
        if (job.input.capacity() < n) {
            job.input  = StreamCpuInput(std::max(n, 8), kHidden);
            job.middle = StreamCpuInput(std::max(n, 8), kIntermediate);
        }
        job.input.set_columns(n);
        job.middle.set_columns(n);
        job.gate_up_rows.resize(std::size_t(n) * 2 * kIntermediate);
        job.middle_values.resize(std::size_t(n) * kIntermediate);
        job.coefficients.resize(std::size_t(n));
        job.gate_up_out.resize(std::size_t(n));
        job.products.resize(std::size_t(n));
        for (int j = 0; j < n; ++j) {
            const int p = job.pairs[std::size_t(j)];
            job.coefficients[std::size_t(j)] = alpha[p];
            job.gate_up_out[std::size_t(j)]  = job.gate_up_rows.data() + std::size_t(j) * 2 * kIntermediate;
            job.products[std::size_t(j)]     = products + std::size_t(row_of[std::size_t(p)]) * kHidden;
        }
    }
    // The pairs' inputs in the lanes' order, a pair a task.
    std::vector<std::pair<std::uint32_t, int>> columns;
    for (std::size_t i = 0; i < count; ++i) {
        for (int j = 0; j < int(jobs_[i].pairs.size()); ++j) { columns.emplace_back(i, j); }
    }
    run(work.pool, columns.size(), [&](std::size_t index) {
        const auto [i, j] = columns[index];
        Job& job          = jobs_[i];
        job.input.load_bf16(j, x + std::size_t(job.pairs[std::size_t(j)] / kTopK) * kHidden);
    });

    // Gate and up rows (the [gate; up] plane's first and second 512) while the calling thread runs
    // work.side.
    std::vector<RowPart> parts;
    for (std::size_t i = 0; i < count; ++i) {
        parts.push_back(
            {2 * kIntermediate, ExpertSplit::cpu_work(1.0, int(jobs_[i].pairs.size()))});
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
            Job& job            = jobs_[task.part];
            stream_cpu_rows(job.gate_up, job.input, task.begin, task.end, nullptr, job.gate_up_out,
                            backend_);
        },
        side);
    const auto rows_done = work.pool ? work.pool->finished() : Clock::now();
    const auto middle    = Clock::now();
    // The FP32 middle silu(gate) * up, as the d3 kernel stores it for d4, a pair a task.
    run(work.pool, columns.size(), [&](std::size_t index) {
        const auto [i, j]    = columns[index];
        Job& job             = jobs_[i];
        const float* gate_up = job.gate_up_rows.data() + std::size_t(j) * 2 * kIntermediate;
        float* out           = job.middle_values.data() + std::size_t(j) * kIntermediate;
        stream_cpu_swiglu(gate_up, gate_up + kIntermediate, kIntermediate, out);
        job.middle.load_fp32(j, out);
    });
    parts.clear();
    for (std::size_t i = 0; i < count; ++i) {
        parts.push_back({kHidden, ExpertSplit::cpu_work(1.0, int(jobs_[i].pairs.size()))});
    }
    tasks = row_tasks(parts, threads, kRowQuantum);
    run(work.pool, tasks.size(), [&](std::size_t index) {
        const RowTask& task = tasks[index];
        Job& job            = jobs_[task.part];
        stream_cpu_rows(job.down, job.middle, task.begin, task.end, job.coefficients.data(),
                        job.products, backend_);
    });
    const auto done = Clock::now();
    return {
        std::size_t(rows) * kHidden * 4,
        std::chrono::duration<double, std::micro>(rows_done - started + (done - middle)).count()};
}

} // namespace ninfer::ops::detail
