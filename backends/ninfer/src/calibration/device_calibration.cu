#include "calibration/device_calibration.h"

#include "calibration/route_catalog.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_add/q4/q4_linear_add_dispatch.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4cublas/cublas_prefill_gate.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::calibration {
namespace {

__global__ void fill_u16_kernel(std::uint16_t* values, std::uint64_t count, std::uint16_t bits) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        values[index] = bits;
    }
}

__global__ void fill_f32_kernel(float* values, std::uint64_t count, float value) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        values[index] = value;
    }
}

__global__ void fill_bf16_ramp_kernel(std::uint16_t* values, std::uint64_t count) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        // bf16 values in [-0.5, 0.5): exact magnitudes are irrelevant to timing.
        const float value = 0.5f - static_cast<float>(index % 251) / 250.0f;
        values[index]     = static_cast<std::uint16_t>(__float_as_uint(value) >> 16);
    }
}

__global__ void fill_random_kernel(std::uint32_t* words, std::uint64_t count, std::uint32_t seed) {
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        std::uint32_t x = static_cast<std::uint32_t>(index) * 0x9E3779B9u ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        words[index] = x;
    }
}

int fill_grid(std::uint64_t count) {
    return static_cast<int>(std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (count + 255) / 256)));
}

void fill_random(void* data, std::size_t bytes, std::uint32_t seed) {
    const std::uint64_t words = bytes / 4;
    fill_random_kernel<<<fill_grid(words), 256>>>(static_cast<std::uint32_t*>(data), words, seed);
    CUDA_CHECK(cudaGetLastError());
}

DeviceBuffer bf16_buffer(std::size_t elements) {
    DeviceBuffer buffer(std::max<std::size_t>(elements * 2, 256));
    fill_bf16_ramp_kernel<<<fill_grid(elements), 256>>>(static_cast<std::uint16_t*>(buffer.p),
                                                        elements);
    CUDA_CHECK(cudaGetLastError());
    return buffer;
}

// A row-split weight with the layout the artifacts use: code plane, optional high plane, FP16
// scales. Codes are a fixed byte pattern that every format decodes to valid values.
struct SyntheticWeight {
    DeviceBuffer storage;
    Weight weight{};
};

SyntheticWeight row_split_weight(QType qtype, std::int32_t n, std::int32_t k) {
    std::int32_t group      = 64;
    std::int32_t high_bytes = 0;
    switch (qtype) {
    case QType::Q4_G64_FP16:
        break;
    case QType::Q5_G64_FP16:
        high_bytes = 8;
        break;
    case QType::Q6_G64_FP16:
        high_bytes = 16;
        break;
    case QType::Q8_G32_FP16:
        group = 32;
        break;
    case QType::T2_G128_FP16:
        group = 128;
        break;
    case QType::PTQ1_G128_FP16:
        group = 128;
        high_bytes = 2;
        break;
    default:
        throw std::invalid_argument("calibration: unsupported synthetic weight format");
    }
    const auto align    = [](std::uint64_t value) { return (value + 255) / 256 * 256; };
    const std::uint64_t groups =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k / group);
    const std::uint64_t low_bytes    = groups * (qtype == QType::PTQ1_G128_FP16 ? 24 : 32);
    const std::uint64_t high_total   = groups * static_cast<std::uint64_t>(high_bytes);
    const std::uint64_t high_offset  = align(low_bytes);
    const std::uint64_t scale_offset = high_offset + align(high_total);
    const std::uint64_t total        = scale_offset + groups * 2;

    SyntheticWeight result{DeviceBuffer(static_cast<std::size_t>(total)), {}};
    auto* base = static_cast<std::uint8_t*>(result.storage.p);
    fill_random(base, low_bytes, 0x5eedu + static_cast<std::uint32_t>(n));
    if (high_total != 0) { fill_random(base + high_offset, high_total, 0xbeefu + static_cast<std::uint32_t>(k)); }
    fill_u16_kernel<<<fill_grid(groups), 256>>>(
        reinterpret_cast<std::uint16_t*>(base + scale_offset), groups, 0x3c00);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& w           = result.weight;
    w.payload           = base;
    w.payload_bytes     = total;
    w.high_plane_bytes  = high_total;
    w.qtype             = qtype;
    w.group_size        = static_cast<std::uint32_t>(group);
    w.shape[0]          = n;
    w.shape[1]          = k;
    w.padded_shape[0]   = n;
    w.padded_shape[1]   = k;
    w.ndim              = 2;
    w.qdata             = base;
    w.qhigh             = high_total == 0 ? nullptr : base + high_offset;
    w.scales            = base + scale_offset;
    w.n                 = n;
    w.k                 = k;
    w.group             = group;
    w.layout            = QuantLayout::RowSplit;
    w.scale_dtype       = DType::FP16;
    return result;
}

// An NVFP4 weight in the layout the artifacts use: an E2M1 code plane, one E4M3 block scale per
// 16 elements of a 128x4 tile, and one weight divisor past the scale plane. The field list is
// exactly what ops::detail::validate_nvfp4_weight requires, including N%128, K%64, the 16-byte
// plane alignment and the divisor row count.
SyntheticWeight block_scale_nvfp4_weight(std::int32_t n, std::int32_t k) {
    const auto align = [](std::uint64_t value) { return (value + 255) / 256 * 256; };
    const std::uint64_t elements = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k);
    const std::uint64_t code_plane_bytes   = elements / 2;
    const std::uint64_t scale_plane_offset = align(code_plane_bytes);
    const std::uint64_t scale_plane_bytes  = elements / 16;
    const std::uint64_t divisor_offset     = scale_plane_offset + scale_plane_bytes;
    const std::uint64_t total              = divisor_offset + sizeof(float);

    SyntheticWeight result{DeviceBuffer(static_cast<std::size_t>(total)), {}};
    auto* base = static_cast<std::uint8_t*>(result.storage.p);
    fill_random(base, code_plane_bytes, 0x4e56u + static_cast<std::uint32_t>(n));
    // E4M3 1.0 is 0x38: a finite scale of one keeps the chains' products comparable, where a
    // non-finite scale would make the output check reject every candidate.
    CUDA_CHECK(cudaMemsetAsync(base + scale_plane_offset, 0x38, scale_plane_bytes, nullptr));
    fill_f32_kernel<<<fill_grid(1), 256>>>(reinterpret_cast<float*>(base + divisor_offset), 1, 1.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& w              = result.weight;
    w.payload              = base;
    w.payload_bytes        = total;
    w.high_plane_bytes     = 0;
    w.qtype                = QType::NVFP4;
    w.group_size           = 16;
    w.shape[0]             = n;
    w.shape[1]             = k;
    w.padded_shape[0]      = n;
    w.padded_shape[1]      = k;
    w.ndim                 = 2;
    w.qdata                = base;
    w.qhigh                = nullptr;
    w.scales               = base + scale_plane_offset;
    w.n                    = n;
    w.k                    = k;
    w.group                = 16;
    w.layout               = QuantLayout::BlockScaleK16M128x4;
    w.scale_dtype          = DType::FP8_E4M3FN;
    w.weight_scale_divisor = 1.0F;
    w.input_scale_divisor  = 1.0F;
    w.weight_divisors      = base + divisor_offset;
    w.weight_divisor_rows  = n;
    return result;
}

class Timer {
public:
    explicit Timer(const CalibrationOptions& options)
        : options_(options), flush_(flush_bytes()) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~Timer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    Timer(const Timer&)            = delete;
    Timer& operator=(const Timer&) = delete;

    // Median microseconds of `launch` with L2 flushed before each sample, or infinity when the
    // launch throws (a candidate outside its domain or an unsupported shape).
    double median_us(const std::function<void(cudaStream_t)>& launch) {
        try {
            for (int index = 0; index < options_.warmup; ++index) {
                flush();
                launch(nullptr);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> samples;
            samples.reserve(static_cast<std::size_t>(options_.repeat));
            for (int index = 0; index < options_.repeat; ++index) {
                flush();
                CUDA_CHECK(cudaEventRecord(start_, nullptr));
                launch(nullptr);
                CUDA_CHECK(cudaEventRecord(stop_, nullptr));
                CUDA_CHECK(cudaEventSynchronize(stop_));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
                samples.push_back(ms);
            }
            std::sort(samples.begin(), samples.end());
            return 1000.0 * samples[samples.size() / 2];
        } catch (const std::exception&) {
            cudaGetLastError();
            cudaDeviceSynchronize();
            return std::numeric_limits<double>::infinity();
        }
    }

private:
    static std::size_t flush_bytes() {
        int device = 0;
        int l2     = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
        return std::max<std::size_t>(64ULL << 20, static_cast<std::size_t>(l2) * 4);
    }
    void flush() { CUDA_CHECK(cudaMemsetAsync(flush_.p, 0xa5, flush_.bytes, nullptr)); }

    const CalibrationOptions& options_;
    DeviceBuffer flush_;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
};

// One route key's measurement; its candidate schedules come from the route catalog.
struct Family {
    std::string key;
    std::vector<std::int32_t> widths;
    std::function<void(std::int32_t, cudaStream_t)> run;
    // The BF16 output a run at this width writes, compared between the compiled route and each
    // candidate: a schedule that computes something else must never win on speed.
    std::function<std::pair<const void*, std::size_t>(std::int32_t)> output;
    // Restores any state a run accumulates into (a residual) before a checked run.
    std::function<void()> reset;
};

std::vector<float> bf16_host(std::pair<const void*, std::size_t> output) {
    std::vector<std::uint16_t> raw(output.second);
    CUDA_CHECK(cudaMemcpy(raw.data(), output.first, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> values(raw.size());
    for (std::size_t index = 0; index < raw.size(); ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw[index]) << 16;
        float value              = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        values[index] = value;
    }
    return values;
}

// Largest element difference over the reference's largest magnitude; infinity for non-finite.
double relative_difference(const std::vector<float>& reference, const std::vector<float>& other) {
    double scale = 1e-6;
    double worst = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(other[index])) { return std::numeric_limits<double>::infinity(); }
        scale = std::max(scale, static_cast<double>(std::fabs(reference[index])));
        worst = std::max(worst, static_cast<double>(std::fabs(reference[index] - other[index])));
    }
    return worst / scale;
}

void report(const CalibrationOptions& options, const std::string& line) {
    if (options.log) { options.log(line); }
}

// Times the compiled choice and every catalog candidate at each width and turns the winners into
// bands. An empty schedule is the compiled route; a candidate replaces it only when faster by
// `margin`. A key whose every width keeps the compiled route is still returned (one compiled band):
// the stored entry then records that this device measured the compiled route best, which overrides
// a built-in profile's choice for the key. Empty only when `only` excludes the key.
std::vector<ops::DeviceRouteBand> sweep(const Family& family, Timer& timer,
                                        const CalibrationOptions& options) {
    std::vector<ops::DeviceRouteBand> bands;
    if (!options.only.empty() && family.key.rfind(options.only, 0) != 0) { return bands; }
    for (const std::int32_t width : family.widths) {
        const double compiled = timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
        std::vector<float> reference;
        if (family.output) {
            if (family.reset) { family.reset(); }
            family.run(width, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            reference = bf16_host(family.output(width));
        }
        double best_us = compiled;
        std::string best;
        for (const std::string& candidate :
             calibration_candidates(family.key, ops::current_device_tier())) {
            const ops::DeviceRouteForce force(family.key, candidate);
            if (family.output) {
                try {
                    if (family.reset) { family.reset(); }
                    family.run(width, nullptr);
                    CUDA_CHECK(cudaDeviceSynchronize());
                } catch (const std::exception&) {
                    cudaGetLastError();
                    continue;
                }
                const double difference = relative_difference(reference, bf16_host(family.output(width)));
                if (!(difference <= 0.05)) {
                    char line[256];
                    std::snprintf(line, sizeof(line), "%s w=%d %s rejected: output differs by %.3g",
                                  family.key.c_str(), width, candidate.c_str(), difference);
                    report(options, line);
                    continue;
                }
            }
            const double us = timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
            if (options.detail) {
                char line[256];
                std::snprintf(line, sizeof(line), "%s w=%d %s %.1f us (compiled %.1f us)",
                              family.key.c_str(), width, candidate.c_str(), us, compiled);
                report(options, line);
            }
            if (std::isfinite(us) && us < compiled * (1.0 - options.margin) && us < best_us) {
                best_us = us;
                best    = candidate;
            }
        }
        // A win must survive a second, interleaved measurement of both routes.
        if (!best.empty()) {
            double again_compiled = 0.0;
            double again_best     = 0.0;
            for (int round = 0; round < 2; ++round) {
                again_compiled += timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
                const ops::DeviceRouteForce force(family.key, best);
                again_best += timer.median_us([&](cudaStream_t stream) { family.run(width, stream); });
            }
            if (!(again_best < again_compiled * (1.0 - options.margin / 2))) { best.clear(); }
        }
        char line[256];
        std::snprintf(line, sizeof(line), "%s w=%d compiled %.1f us -> %s %.1f us", family.key.c_str(),
                      width, compiled, best.empty() ? "compiled" : best.c_str(),
                      best.empty() ? compiled : best_us);
        report(options, line);
        if (!bands.empty() && bands.back().schedule == best) {
            bands.back().last = width;
        } else {
            bands.push_back({width, best});
        }
    }
    return bands;
}

std::vector<std::int32_t> decode_widths() {
    return {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
}

// Ternary Bonsai 2 27B projections, in the launches the model makes: the GDN and attention input
// pairs, the output and down projections with their residual adds, the gate/up and the LM head.
void calibrate_ternary(ops::DeviceRouteProfile& profile, Timer& timer,
                       const CalibrationOptions& options) {
    constexpr std::int32_t hidden = 5120;
    constexpr std::int32_t max_tokens = 192;
    SyntheticWeight gdn_qk      = row_split_weight(QType::T2_G128_FP16, 4096, hidden);
    SyntheticWeight gdn_vz      = row_split_weight(QType::T2_G128_FP16, 12288, hidden);
    SyntheticWeight attn_qk     = row_split_weight(QType::T2_G128_FP16, 7168, hidden);
    SyntheticWeight attn_gv     = row_split_weight(QType::T2_G128_FP16, 7168, hidden);
    SyntheticWeight out_proj    = row_split_weight(QType::T2_G128_FP16, hidden, 6144);
    SyntheticWeight gate_up     = row_split_weight(QType::T2_G128_FP16, 34816, hidden);
    SyntheticWeight down        = row_split_weight(QType::T2_G128_FP16, hidden, 17408);
    SyntheticWeight head        = row_split_weight(QType::T2_G128_FP16, 248320, hidden);
    DeviceBuffer x_hidden       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer x_mixer        = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer x_intermediate = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    DeviceBuffer qkv            = bf16_buffer(static_cast<std::size_t>(10240) * max_tokens);
    DeviceBuffer z              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer q              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer gate           = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer k              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer v              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer residual       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer wide           = bf16_buffer(static_cast<std::size_t>(34816) * max_tokens);
    DeviceBuffer logits         = bf16_buffer(static_cast<std::size_t>(248320) * 16);
    WorkspaceArena workspace(256ULL << 20);

    const auto tensor = [](DeviceBuffer& buffer, std::int32_t rows, std::int32_t cols) {
        return Tensor(buffer.p, DType::BF16, {rows, cols});
    };
    const auto gdn_pair = [&](std::int32_t cols, cudaStream_t stream) {
        auto scope        = workspace.scope();
        Tensor x          = tensor(x_hidden, hidden, cols);
        Tensor qkv_plane  = tensor(qkv, 10240, cols);
        Tensor z_plane    = tensor(z, 6144, cols);
        const auto active = ops::detail::t2_a8_quantize(x, workspace, stream);
        ops::detail::t2_a8_project_split_pair(active, {gdn_qk.weight, qkv_plane, 0, 4096, qkv_plane, 0},
                                              {gdn_vz.weight, qkv_plane, 4096, 6144, z_plane, 0},
                                              stream);
    };
    const auto attn_pair = [&](std::int32_t cols, cudaStream_t stream) {
        auto scope        = workspace.scope();
        Tensor x          = tensor(x_hidden, hidden, cols);
        Tensor q_plane    = tensor(q, 6144, cols);
        Tensor gate_plane = tensor(gate, 6144, cols);
        Tensor k_plane    = tensor(k, 1024, cols);
        Tensor v_plane    = tensor(v, 1024, cols);
        const auto active = ops::detail::t2_a8_quantize(x, workspace, stream);
        ops::detail::t2_a8_project_split_pair(active, {attn_qk.weight, q_plane, 0, 6144, k_plane, 0},
                                              {attn_gv.weight, gate_plane, 0, 6144, v_plane, 0},
                                              stream);
    };
    const auto out_add = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_mixer, 6144, cols);
        Tensor res = tensor(residual, hidden, cols);
        ops::detail::t2_a8_linear_add(x, out_proj.weight, res, workspace, stream);
    };
    const auto gate_up_linear = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_hidden, hidden, cols);
        Tensor out = tensor(wide, 34816, cols);
        ops::detail::t2_a8_linear(x, gate_up.weight, out, workspace, stream);
    };
    const auto down_add = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_intermediate, 17408, cols);
        Tensor res = tensor(residual, hidden, cols);
        ops::detail::t2_a8_linear_add(x, down.weight, res, workspace, stream);
    };
    const auto head_linear = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x   = tensor(x_hidden, hidden, cols);
        Tensor out = tensor(logits, 248320, cols);
        ops::detail::t2_a8_linear(x, head.weight, out, workspace, stream);
    };

    DeviceBuffer residual_pristine = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    const auto view = [](DeviceBuffer& buffer, std::size_t rows) {
        return [&buffer, rows](std::int32_t cols) {
            return std::pair<const void*, std::size_t>{buffer.p, rows * static_cast<std::size_t>(cols)};
        };
    };
    const auto reset_residual = [&]() {
        CUDA_CHECK(cudaMemcpy(residual.p, residual_pristine.p, residual.bytes, cudaMemcpyDeviceToDevice));
    };
    const std::vector<Family> families = {
        {"t2_i8_small/4096+12288x5120", decode_widths(), gdn_pair, view(qkv, 10240), {}},
        {"t2_i8_small/7168+7168x5120", decode_widths(), attn_pair, view(q, 6144), {}},
        {"t2_i8_small/5120x6144", decode_widths(), out_add, view(residual, hidden), reset_residual},
        {"t2_i8_small/34816x5120", decode_widths(), gate_up_linear, view(wide, 34816), {}},
        {"t2_i8_small/5120x17408", decode_widths(), down_add, view(residual, hidden), reset_residual},
        {"t2_i8_small/248320x5120", {1, 2, 3, 4, 5, 6, 7, 8, 12, 16}, head_linear,
         view(logits, 248320), {}},
        {"t2_i8_route",
         {16, 24, 32, 48, 64, 96, 128, 160, 192},
         [&](std::int32_t cols, cudaStream_t stream) {
             gdn_pair(cols, stream);
             out_add(cols, stream);
             gate_up_linear(cols, stream);
             down_add(cols, stream);
         },
         view(wide, 34816),
         reset_residual},
    };
    for (const Family& family : families) {
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

// PTQ1's compact planes and 128-row kernel have their own measurements. A T2
// profile cannot steer this family, even when both artifacts describe Bonsai.
void calibrate_ptq1(ops::DeviceRouteProfile& profile, Timer& timer,
                    const CalibrationOptions& options) {
    constexpr std::int32_t max_tokens = 512;
    for (const auto [n, k] : {std::pair{4096, 5120}, std::pair{12288, 5120},
                             std::pair{7168, 5120}, std::pair{5120, 6144},
                             std::pair{34816, 5120}, std::pair{5120, 17408},
                             std::pair{248320, 5120}}) {
        const std::string key = "ptq1_i8_tile/" + std::to_string(n) + "x" + std::to_string(k);
        if (!options.only.empty() && key.rfind(options.only, 0) != 0) { continue; }
        auto weight = row_split_weight(QType::PTQ1_G128_FP16, n, k);
        auto input = bf16_buffer(static_cast<std::size_t>(k) * max_tokens);
        auto output = bf16_buffer(static_cast<std::size_t>(n) * max_tokens);
        WorkspaceArena workspace(ops::detail::t2_a8_workspace_bytes(
            n, k, ops::LinearPolicy::AllowA8Int, max_tokens, true));
        const Family family{key, {33, 64, 65, 128, 256, 512},
            [&](std::int32_t cols, cudaStream_t stream) {
                auto scope = workspace.scope();
                Tensor x(input.p, DType::BF16, {k, cols});
                Tensor y(output.p, DType::BF16, {n, cols});
                ops::detail::t2_a8_linear(x, weight.weight, y, workspace, stream);
            },
            [&](std::int32_t cols) {
                return std::pair<const void*, std::size_t>{output.p, static_cast<std::size_t>(n) * cols};
            }, {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            profile.routes[key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

// Qwen3.6/3.8-27B groupwise-int fused projections at their registered shapes.
void calibrate_groupwise(ops::DeviceRouteProfile& profile, Timer& timer,
                         const CalibrationOptions& options) {
    constexpr std::int32_t hidden     = 5120;
    constexpr std::int32_t max_tokens = 64;
    SyntheticWeight attn_qk = row_split_weight(QType::Q4_G64_FP16, 7168, hidden);
    SyntheticWeight attn_gv = row_split_weight(QType::Q5_G64_FP16, 7168, hidden);
    SyntheticWeight gdn_qk  = row_split_weight(QType::Q4_G64_FP16, 4096, hidden);
    SyntheticWeight gdn_vz  = row_split_weight(QType::Q5_G64_FP16, 12288, hidden);
    SyntheticWeight out     = row_split_weight(QType::Q5_G64_FP16, hidden, 6144);
    SyntheticWeight down    = row_split_weight(QType::Q5_G64_FP16, hidden, 17408);
    SyntheticWeight gate_up = row_split_weight(QType::Q4_G64_FP16, 34816, hidden);
    SyntheticWeight out_q4            = row_split_weight(QType::Q4_G64_FP16, hidden, 6144);
    SyntheticWeight down_q4           = row_split_weight(QType::Q4_G64_FP16, hidden, 17408);
    SyntheticWeight head_q6           = row_split_weight(QType::Q6_G64_FP16, 248320, hidden);
    DeviceBuffer x_hidden       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer x_mixer        = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer x_intermediate = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    DeviceBuffer q              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer gate           = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer k              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer v              = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer qkv            = bf16_buffer(static_cast<std::size_t>(10240) * max_tokens);
    DeviceBuffer z              = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer residual       = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer swiglu_out     = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    DeviceBuffer logits         = bf16_buffer(static_cast<std::size_t>(248320) * 32);
    WorkspaceArena workspace(256ULL << 20);
    const auto tensor = [](DeviceBuffer& buffer, std::int32_t rows, std::int32_t cols) {
        return Tensor(buffer.p, DType::BF16, {rows, cols});
    };
    std::vector<std::int32_t> widths = decode_widths();
    widths.insert(widths.end(), {40, 48, 56, 64});
    DeviceBuffer residual_pristine = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    const auto view = [](DeviceBuffer& buffer, std::size_t rows) {
        return [&buffer, rows](std::int32_t cols) {
            return std::pair<const void*, std::size_t>{buffer.p, rows * static_cast<std::size_t>(cols)};
        };
    };
    const auto reset_residual = [&]() {
        CUDA_CHECK(cudaMemcpy(residual.p, residual_pristine.p, residual.bytes, cudaMemcpyDeviceToDevice));
    };

    const std::vector<Family> families = {
        {"q4_q5_attn_input/5120x6144x1024",
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_hidden, hidden, cols);
             Tensor qp = tensor(q, 6144, cols), gp = tensor(gate, 6144, cols);
             Tensor kp = tensor(k, 1024, cols), vp = tensor(v, 1024, cols);
             ops::detail::q4_q5_attn_input_dispatch(x, attn_qk.weight, attn_gv.weight, qp, gp, kp, vp,
                                                    stream);
         },
         view(q, 6144),
         {}},
        {"q4_q5_gdn_input/5120x4096x12288",
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x  = tensor(x_hidden, hidden, cols);
             Tensor qp = tensor(qkv, 10240, cols), zp = tensor(z, 6144, cols);
             ops::detail::q4_q5_gdn_input_dispatch(x, gdn_qk.weight, gdn_vz.weight, qp, zp, stream);
         },
         view(qkv, 10240),
         {}},
        {"q5_linear_add/5120x6144",
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_mixer, 6144, cols), res = tensor(residual, hidden, cols);
             ops::detail::q5_linear_add_dispatch(x, out.weight, res, workspace, stream);
         },
         view(residual, hidden),
         reset_residual},
        {"q5_linear_add/5120x17408",
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_intermediate, 17408, cols), res = tensor(residual, hidden, cols);
             ops::detail::q5_linear_add_dispatch(x, down.weight, res, workspace, stream);
         },
         view(residual, hidden),
         reset_residual},
        {"q4_linear_swiglu/34816x17408x5120",
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_hidden, hidden, cols), o = tensor(swiglu_out, 17408, cols);
             ops::detail::q4_linear_swiglu_dispatch(x, gate_up.weight, o, workspace, stream);
         },
         view(swiglu_out, 17408),
         {}},
        {"q4_linear_add/5120x6144",
         decode_widths(),
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_mixer, 6144, cols), res = tensor(residual, hidden, cols);
             ops::detail::select_q4_linear_add(hidden, 6144, cols)(x, out_q4.weight, res, stream);
         },
         view(residual, hidden),
         reset_residual},
        {"q4_linear_add/5120x17408",
         decode_widths(),
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_intermediate, 17408, cols), res = tensor(residual, hidden, cols);
             ops::detail::select_q4_linear_add(hidden, 17408, cols)(x, down_q4.weight, res, stream);
         },
         view(residual, hidden),
         reset_residual},
        {"q6_head/248320x5120",
         decode_widths(),
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_hidden, hidden, cols), o = tensor(logits, 248320, cols);
             ops::linear(x, head_q6.weight, o, stream);
         },
         view(logits, 248320),
         {}},
    };
    // Upstream's move of these projections onto the unified templates, against the routes the
    // families above left: one switch per Op, measured over the widths its fused launches serve.
    const auto widths_to = [&](std::int32_t last) {
        std::vector<std::int32_t> out;
        for (const std::int32_t width : widths) {
            if (width <= last) { out.push_back(width); }
        }
        return out;
    };
    const std::vector<Family> unified_families = {
        {"unified/q4_q5_attn_input",
         widths_to(12),
         families[0].run,
         families[0].output,
         {}},
        {"unified/q4_q5_gdn_input",
         widths_to(15),
         families[1].run,
         families[1].output,
         {}},
        {"unified/q5_linear_add/5120x6144",
         widths,
         families[2].run,
         families[2].output,
         families[2].reset},
        {"unified/q5_linear_add/5120x17408",
         widths,
         families[3].run,
         families[3].output,
         families[3].reset},
        {"unified/q4_linear_swiglu",
         widths_to(32),
         families[4].run,
         families[4].output,
         {}},
        {"unified/q4_linear_add",
         widths,
         [&](std::int32_t cols, cudaStream_t stream) {
             Tensor x = tensor(x_mixer, 6144, cols), res = tensor(residual, hidden, cols);
             ops::detail::select_q4_linear_add(hidden, 6144, cols)(x, out_q4.weight, res, stream);
         },
         view(residual, hidden),
         reset_residual},
    };
    for (const auto* set : {&families, &unified_families}) {
        // A card that cannot select the unified table has no switch onto it to measure.
        if (set == &unified_families && !ops::detail::unified_linear_routes_here()) { continue; }
        for (const Family& family : *set) {
            auto bands = sweep(family, timer, options);
            if (!bands.empty()) {
                profile.routes[family.key] = std::move(bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
        }
    }
}

// The NVFP4 fused projections: the A16 chain choice under the four `unified/nvfp4_*` keys, and the
// mainloop choice of the int8 family under the four `nvfp4_int8_*` keys (--nvfp4-w4a8-int). Both
// families are measured through the same dispatch the serving call takes, so the route table under
// test decides the kernel in both arms.
//
// Under A16Only every chain is registered at every width a fused launch serves, so the A16 keys
// are a W4A16 chain choice and nothing else: the own chain's per-width kernels against the unified
// chain's own, which is a 32-token tiling loop against one launch at prefill widths and two
// distinct small-T kernel families below them. Past 16 columns both chains resolve to the same wide
// fused A16 MMA kernel, so the SwiGLU A16 key stops there: above it there is no switch to measure.
// The int8 keys cover every width -- that family pads rather than stopping at 16.
void calibrate_nvfp4_fused(ops::DeviceRouteProfile& profile, Timer& timer,
                           const CalibrationOptions& options) {
    // A card that cannot select the unified table has no switch onto it to measure.
    if (!ops::detail::unified_linear_routes_here()) { return; }
    constexpr std::int32_t hidden     = 5120;
    constexpr std::int32_t max_tokens = 1024;
    SyntheticWeight attn     = block_scale_nvfp4_weight(14336, hidden);
    SyntheticWeight gdn      = block_scale_nvfp4_weight(16384, hidden);
    SyntheticWeight out_proj = block_scale_nvfp4_weight(hidden, 6144);
    SyntheticWeight down     = block_scale_nvfp4_weight(hidden, 17408);
    SyntheticWeight gate_up  = block_scale_nvfp4_weight(34816, hidden);
    DeviceBuffer x_hidden   = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer x_mixer    = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer x_inter    = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    DeviceBuffer q          = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer gate       = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer k_plane    = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer v_plane    = bf16_buffer(static_cast<std::size_t>(1024) * max_tokens);
    DeviceBuffer qkv        = bf16_buffer(static_cast<std::size_t>(10240) * max_tokens);
    DeviceBuffer z_plane    = bf16_buffer(static_cast<std::size_t>(6144) * max_tokens);
    DeviceBuffer residual   = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    DeviceBuffer swiglu_out = bf16_buffer(static_cast<std::size_t>(17408) * max_tokens);
    // The row feeding an input projection is the RMSNorm of the residual, and the int8 route can
    // fold that norm into its own quantise pass. Both halves of the comparison therefore need the
    // norm's weight and the BF16 row the non-folding arm writes.
    DeviceBuffer norm_weight        = bf16_buffer(static_cast<std::size_t>(hidden));
    DeviceBuffer x_normed           = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    constexpr float kNormEps        = 1e-6F;
    WorkspaceArena workspace(256ULL << 20);

    const auto tensor = [](DeviceBuffer& buffer, std::int32_t rows, std::int32_t cols) {
        return Tensor(buffer.p, DType::BF16, {rows, cols});
    };
    const auto view = [](DeviceBuffer& buffer, std::size_t rows) {
        return [&buffer, rows](std::int32_t cols) {
            return std::pair<const void*, std::size_t>{buffer.p,
                                                       rows * static_cast<std::size_t>(cols)};
        };
    };
    DeviceBuffer residual_pristine = bf16_buffer(static_cast<std::size_t>(hidden) * max_tokens);
    const auto reset_residual      = [&]() {
        CUDA_CHECK(
            cudaMemcpy(residual.p, residual_pristine.p, residual.bytes, cudaMemcpyDeviceToDevice));
    };

    const auto attn_run = [&](ops::LinearPolicy policy, std::int32_t cols, cudaStream_t stream) {
        auto scope      = workspace.scope();
        Tensor q_out    = tensor(q, 6144, cols);
        Tensor gate_out = tensor(gate, 6144, cols);
        Tensor k_out    = tensor(k_plane, 1024, cols);
        Tensor v_out    = tensor(v_plane, 1024, cols);
        // Under the int8 policy the timed region is the edge: if the fused entry is eligible under
        // the route state this candidate is timed in it runs, otherwise the standalone rmsnorm is
        // issued first. Both arms are charged for what they really pay, and an A16 arm is a bare
        // projection because both of its sides pay the same norm.
        if (policy == ops::LinearPolicy::AllowNvfp4Int8) {
            Tensor normed = tensor(x_normed, hidden, cols);
            if (ops::attn_input_proj_fused_rmsnorm_int8_eligible(attn.weight, policy, cols)) {
                ops::attn_input_proj_fused_rmsnorm_int8(tensor(residual, hidden, cols),
                                                        tensor(norm_weight, hidden, 1), kNormEps,
                                                        attn.weight, q_out, gate_out, k_out, v_out,
                                                        policy, workspace, stream);
                return;
            }
            ops::rmsnorm(tensor(residual, hidden, cols), tensor(norm_weight, hidden, 1), kNormEps,
                         true, normed, stream);
            ops::detail::nvfp4_attn_input_dispatch(normed, attn.weight, q_out, gate_out, k_out,
                                                   v_out, policy, &workspace, stream);
            return;
        }
        ops::detail::nvfp4_attn_input_dispatch(tensor(x_hidden, hidden, cols), attn.weight, q_out,
                                               gate_out, k_out, v_out, policy, &workspace, stream);
    };
    const auto gdn_run = [&](ops::LinearPolicy policy, std::int32_t cols, cudaStream_t stream) {
        auto scope     = workspace.scope();
        Tensor x       = tensor(x_hidden, hidden, cols);
        Tensor qkv_out = tensor(qkv, 10240, cols);
        Tensor z_out   = tensor(z_plane, 6144, cols);
        ops::detail::nvfp4_gdn_input_dispatch(x, gdn.weight, qkv_out, z_out, policy, &workspace,
                                              stream);
    };
    // One key steers both residual adds, so a run exercises the pair the way a layer does.
    const auto add_run = [&](ops::LinearPolicy policy, std::int32_t cols, cudaStream_t stream) {
        auto scope = workspace.scope();
        Tensor x   = tensor(x_mixer, 6144, cols);
        Tensor res = tensor(residual, hidden, cols);
        ops::detail::nvfp4_linear_add_dispatch(x, out_proj.weight, res, policy, workspace, stream);
        Tensor x2   = tensor(x_inter, 17408, cols);
        Tensor res2 = tensor(residual, hidden, cols);
        ops::detail::nvfp4_linear_add_dispatch(x2, down.weight, res2, policy, workspace, stream);
    };
    const auto swiglu_run = [&](ops::LinearPolicy policy, std::int32_t cols,
                                cudaStream_t stream) {
        auto scope = workspace.scope();
        Tensor o   = tensor(swiglu_out, 17408, cols);
        if (policy == ops::LinearPolicy::AllowNvfp4Int8) {
            Tensor normed = tensor(x_normed, hidden, cols);
            if (ops::linear_swiglu_fused_rmsnorm_int8_eligible(gate_up.weight, policy, cols)) {
                ops::linear_swiglu_fused_rmsnorm_int8(tensor(residual, hidden, cols),
                                                      tensor(norm_weight, hidden, 1), kNormEps,
                                                      gate_up.weight, o, policy, workspace, stream);
                return;
            }
            ops::rmsnorm(tensor(residual, hidden, cols), tensor(norm_weight, hidden, 1), kNormEps,
                         true, normed, stream);
            ops::detail::nvfp4_linear_swiglu_dispatch(normed, gate_up.weight, o, policy, workspace,
                                                      stream);
            return;
        }
        ops::detail::nvfp4_linear_swiglu_dispatch(tensor(x_hidden, hidden, cols), gate_up.weight, o,
                                                  policy, workspace, stream);
    };

    // The decode step, the speculative verify widths, the small-T band, and the prefill chunks a
    // 27B serves: the bands a profile stores have to answer all of them.
    const std::vector<std::int32_t> widths = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 12, 14,
                                              16, 20, 24, 28, 32, 48, 64, 96, 128, 256, 512, 1024};
    const std::vector<std::int32_t> small_t_widths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16};
    const auto bind_policy = [](const auto& run, ops::LinearPolicy policy) {
        return std::function<void(std::int32_t, cudaStream_t)>(
            [&run, policy](std::int32_t cols, cudaStream_t stream) { run(policy, cols, stream); });
    };
    const std::vector<Family> families = {
        {"unified/nvfp4_attn_input", widths, bind_policy(attn_run, ops::LinearPolicy::A16Only),
         view(q, 6144), {}},
        {"unified/nvfp4_gdn_input", widths, bind_policy(gdn_run, ops::LinearPolicy::A16Only),
         view(qkv, 10240), {}},
        {"unified/nvfp4_linear_add", widths, bind_policy(add_run, ops::LinearPolicy::A16Only),
         view(residual, hidden), reset_residual},
        {"unified/nvfp4_linear_swiglu", small_t_widths,
         bind_policy(swiglu_run, ops::LinearPolicy::A16Only), view(swiglu_out, 17408), {}},
    };
    // The int8 family's own four keys: the same weights and buffers, only the policy differs, so
    // the sweep measures which chain's int8 mainloop answers at each width.
    const std::vector<Family> int8_families = {
        {"nvfp4_int8/attn_input", widths,
         bind_policy(attn_run, ops::LinearPolicy::AllowNvfp4Int8), view(q, 6144), {}},
        {"nvfp4_int8/gdn_input", widths,
         bind_policy(gdn_run, ops::LinearPolicy::AllowNvfp4Int8), view(qkv, 10240), {}},
        {"nvfp4_int8/linear_add", widths,
         bind_policy(add_run, ops::LinearPolicy::AllowNvfp4Int8), view(residual, hidden),
         reset_residual},
        {"nvfp4_int8/linear_swiglu", widths,
         bind_policy(swiglu_run, ops::LinearPolicy::AllowNvfp4Int8), view(swiglu_out, 17408), {}},
    };
    for (const auto* set : {&families, &int8_families}) {
        for (const Family& family : *set) {
            auto bands = sweep(family, timer, options);
            if (!bands.empty()) {
                profile.routes[family.key] = std::move(bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
        }
    }
}

// The cuBLAS prefill route's width gate (--prefill-cublas) per Op, through the public Ops under
// AllowPrefillCublas, at the 27B's registered shapes. Each Op is timed twice: at widths the integer
// A8 mainloop cannot take, where the A16 mainloop is the alternative, and at multiples of its
// 128-column tile under the `/a8` key. Widths start past the decode floor
// (kCublasPrefillFloorTokens) and stop where the route wins on every card measured; past them the
// compiled gate answers. One width sits just below that gate, so no band of compiled answers
// spans it: such a band would hand the widths before the gate the answer measured past it.
void calibrate_cublas_prefill(ops::DeviceRouteProfile& profile, Timer& timer,
                              const CalibrationOptions& options) {
    constexpr ops::LinearPolicy policy             = ops::LinearPolicy::AllowPrefillCublas;
    constexpr std::int32_t hidden                  = 5120;
    const std::vector<std::int32_t> other_widths   = {
        136, 200, 264, 328, 400, 464, ops::kCublasPrefillMinTokens - 1, 600, 776, 1000, 1500};
    const std::vector<std::int32_t> aligned_widths = {256, 384, 512, 640, 768, 1024, 1536, 2048};
    const std::int32_t max_tokens                  = aligned_widths.back();
    const std::int32_t first_tokens                = ops::kCublasPrefillFloorTokens + 1;
    if (!options.only.empty() && std::string_view("w4_cublas/").rfind(options.only, 0) != 0 &&
        options.only.rfind("w4_cublas/", 0) != 0) {
        return;
    }
    SyntheticWeight gate_up = row_split_weight(QType::Q4_G64_FP16, 34816, hidden);
    SyntheticWeight down    = row_split_weight(QType::Q5_G64_FP16, hidden, 17408);
    SyntheticWeight out     = row_split_weight(QType::Q5_G64_FP16, hidden, 6144);
    SyntheticWeight gdn_qk  = row_split_weight(QType::Q4_G64_FP16, 4096, hidden);
    SyntheticWeight gdn_vz  = row_split_weight(QType::Q5_G64_FP16, 12288, hidden);
    SyntheticWeight attn_qk = row_split_weight(QType::Q4_G64_FP16, 7168, hidden);
    SyntheticWeight attn_gv = row_split_weight(QType::Q5_G64_FP16, 7168, hidden);
    const auto columns      = [&](std::int32_t rows) {
        return bf16_buffer(static_cast<std::size_t>(rows) * max_tokens);
    };
    DeviceBuffer x_hidden          = columns(hidden);
    DeviceBuffer x_mixer           = columns(6144);
    DeviceBuffer x_intermediate    = columns(17408);
    DeviceBuffer swiglu_out        = columns(17408);
    DeviceBuffer residual          = columns(hidden);
    DeviceBuffer residual_pristine = columns(hidden);
    DeviceBuffer qkv               = columns(10240);
    DeviceBuffer z                 = columns(6144);
    DeviceBuffer q                 = columns(6144);
    DeviceBuffer gate              = columns(6144);
    DeviceBuffer k                 = columns(1024);
    DeviceBuffer v                 = columns(1024);
    // Every family's widest claimant over the measured widths, the route forced on from the floor.
    std::size_t workspace_bytes = 256;
    const auto reserve = [&](const ops::detail::CublasPrefillFamily& family, auto capacity) {
        const ops::DeviceRouteForce force(std::string(family.key), "on");
        workspace_bytes = std::max(workspace_bytes, capacity(first_tokens, max_tokens));
    };
    reserve(ops::detail::kCublasGateUp, [&](std::int32_t lo, std::int32_t hi) {
        return ops::linear_swiglu_workspace_capacity_bytes(QType::Q4_G64_FP16, 34816, hidden,
                                                           policy, lo, hi);
    });
    reserve(ops::detail::kCublasDown, [&](std::int32_t lo, std::int32_t hi) {
        return ops::linear_add_workspace_capacity_bytes(QType::Q5_G64_FP16, hidden, 17408, policy,
                                                        lo, hi);
    });
    reserve(ops::detail::kCublasOut, [&](std::int32_t lo, std::int32_t hi) {
        return ops::linear_add_workspace_capacity_bytes(QType::Q5_G64_FP16, hidden, 6144, policy,
                                                        lo, hi);
    });
    reserve(ops::detail::kCublasGdnInput, [&](std::int32_t lo, std::int32_t hi) {
        return ops::gdn_input_proj_split_workspace_capacity_bytes(
            QType::Q4_G64_FP16, 4096, QType::Q5_G64_FP16, 12288, hidden, policy, lo, hi);
    });
    reserve(ops::detail::kCublasAttnInput, [&](std::int32_t lo, std::int32_t hi) {
        return ops::attn_input_proj_split_workspace_capacity_bytes(
            QType::Q4_G64_FP16, 7168, QType::Q5_G64_FP16, 7168, hidden, policy, lo, hi);
    });
    WorkspaceArena workspace(workspace_bytes);
    const auto tensor = [](DeviceBuffer& buffer, std::int32_t rows, std::int32_t cols) {
        return Tensor(buffer.p, DType::BF16, {rows, cols});
    };
    const auto view = [](DeviceBuffer& buffer, std::size_t rows) {
        return [&buffer, rows](std::int32_t cols) {
            return std::pair<const void*, std::size_t>{buffer.p,
                                                       rows * static_cast<std::size_t>(cols)};
        };
    };
    const auto reset_residual = [&]() {
        CUDA_CHECK(
            cudaMemcpy(residual.p, residual_pristine.p, residual.bytes, cudaMemcpyDeviceToDevice));
    };
    const auto run_gate_up = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x = tensor(x_hidden, hidden, cols), o = tensor(swiglu_out, 17408, cols);
        ops::linear_swiglu(x, gate_up.weight, o, policy, workspace, stream);
    };
    const auto run_add = [&](DeviceBuffer& input, std::int32_t k_rows, SyntheticWeight& weight) {
        return [&, k_rows](std::int32_t cols, cudaStream_t stream) {
            Tensor x = tensor(input, k_rows, cols), res = tensor(residual, hidden, cols);
            ops::linear_add(x, weight.weight, res, policy, workspace, stream);
        };
    };
    const auto run_gdn = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x  = tensor(x_hidden, hidden, cols);
        Tensor qp = tensor(qkv, 10240, cols), zp = tensor(z, 6144, cols);
        ops::gdn_input_proj(x, gdn_qk.weight, gdn_vz.weight, qp, zp, policy, workspace, stream);
    };
    const auto run_attn = [&](std::int32_t cols, cudaStream_t stream) {
        Tensor x  = tensor(x_hidden, hidden, cols);
        Tensor qp = tensor(q, 6144, cols), gp = tensor(gate, 6144, cols);
        Tensor kp = tensor(k, 1024, cols), vp = tensor(v, 1024, cols);
        ops::attn_input_proj(x, attn_qk.weight, attn_gv.weight, qp, gp, kp, vp, policy, workspace,
                             stream);
    };

    struct Op {
        const ops::detail::CublasPrefillFamily& family;
        std::function<void(std::int32_t, cudaStream_t)> run;
        std::function<std::pair<const void*, std::size_t>(std::int32_t)> output;
        std::function<void()> reset;
    };

    const std::vector<Op> op_list = {
        {ops::detail::kCublasGateUp, run_gate_up, view(swiglu_out, 17408), {}},
        {ops::detail::kCublasDown, run_add(x_intermediate, 17408, down), view(residual, hidden),
         reset_residual},
        {ops::detail::kCublasOut, run_add(x_mixer, 6144, out), view(residual, hidden),
         reset_residual},
        {ops::detail::kCublasGdnInput, run_gdn, view(qkv, 10240), {}},
        {ops::detail::kCublasAttnInput, run_attn, view(q, 6144), {}},
    };
    for (const Op& op : op_list) {
        for (const bool aligned : {false, true}) {
            const Family family{std::string(aligned ? op.family.integer_key : op.family.key),
                                aligned ? aligned_widths : other_widths, op.run, op.output,
                                op.reset};
            auto bands = sweep(family, timer, options);
            if (!bands.empty()) {
                profile.routes[family.key] = std::move(bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
        }
    }
}

// The Q8 G32 K-split contractions' 8-column tiles on their four-K-warp form, at every registered
// geometry and decode width, through the Linear Op an MTP head, a DFlash2 drafter or an official
// artifact's head runs. One synthetic weight at a time bounds the memory this takes.
void calibrate_q8_ksplit(ops::DeviceRouteProfile& profile, Timer& timer,
                         const CalibrationOptions& options) {
    constexpr std::int32_t max_tokens      = 8;
    const std::vector<std::int32_t> widths = {1, 2, 3, 4, 5, 6, 7, 8};
    for (const auto& shape :
         {std::pair{6144, 5120}, std::pair{14336, 5120}, std::pair{17408, 5120},
          std::pair{34816, 5120}, std::pair{248320, 5120}, std::pair{5120, 6144},
          std::pair{5120, 10240}, std::pair{5120, 17408}, std::pair{5120, 25600},
          std::pair{2048, 4096}, std::pair{2048, 16384}}) {
        const std::int32_t rows = shape.first;
        const std::int32_t k    = shape.second;
        const std::string key   = "q8_ksplit/" + std::to_string(rows) + "x" + std::to_string(k);
        if (!options.only.empty() && key.rfind(options.only, 0) != 0) { continue; }
        SyntheticWeight weight = row_split_weight(QType::Q8_G32_FP16, rows, k);
        DeviceBuffer x         = bf16_buffer(static_cast<std::size_t>(k) * max_tokens);
        DeviceBuffer out       = bf16_buffer(static_cast<std::size_t>(rows) * max_tokens);
        const Family family{key,
                            widths,
                            [&](std::int32_t cols, cudaStream_t stream) {
                                const Tensor input(x.p, DType::BF16, {k, cols});
                                Tensor output(out.p, DType::BF16, {rows, cols});
                                ops::linear(input, weight.weight, output, stream);
                            },
                            [&](std::int32_t cols) {
                                return std::pair<const void*, std::size_t>{
                                    out.p, static_cast<std::size_t>(rows) *
                                               static_cast<std::size_t>(cols)};
                            },
                            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(
                std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

// The INT8-family small-T partial kernel's tiers. A serving process captures its graphs over an
// envelope as wide as the KV capacity, so the tier is chosen per envelope and has to serve every
// depth below it: each envelope is timed at its full depth and at a sixteenth of it.
// The two-stage GDN prefill against the WY/state-passing pipeline at the prompt widths prefill
// takes, for the 27B's 48 value heads and the 35B-A3B's 32 (16 query/key heads each).
void calibrate_linear_attention(ops::DeviceRouteProfile& profile, Timer& timer,
                                const CalibrationOptions& options) {
    constexpr std::int32_t state_dim       = 128;
    constexpr std::int32_t qk_heads        = 16;
    const std::vector<std::int32_t> widths = {16,  32,  48,   64,   96,   128,
                                              256, 512, 1024, 2048, 4096, 8192};
    const std::int32_t max_tokens          = widths.back();
    const float scale                      = 1.0f / std::sqrt(static_cast<float>(state_dim));
    for (const std::int32_t value_heads : {48, 32}) {
        const auto f32_buffer = [](std::size_t elements, float value) {
            DeviceBuffer buffer(std::max<std::size_t>(elements * 4, 256));
            fill_f32_kernel<<<fill_grid(elements), 256>>>(static_cast<float*>(buffer.p), elements,
                                                          value);
            CUDA_CHECK(cudaGetLastError());
            return buffer;
        };
        const std::size_t qk_elements = static_cast<std::size_t>(state_dim) * qk_heads * max_tokens;
        const std::size_t value_elements =
            static_cast<std::size_t>(state_dim) * value_heads * max_tokens;
        const std::size_t gate_elements = static_cast<std::size_t>(value_heads) * max_tokens;
        const std::size_t state_elements =
            static_cast<std::size_t>(state_dim) * state_dim * value_heads;
        DeviceBuffer q         = bf16_buffer(qk_elements);
        DeviceBuffer k         = bf16_buffer(qk_elements);
        DeviceBuffer v         = bf16_buffer(value_elements);
        DeviceBuffer out       = bf16_buffer(value_elements);
        DeviceBuffer g         = f32_buffer(gate_elements, -0.05f);
        DeviceBuffer beta      = f32_buffer(gate_elements, 0.5f);
        DeviceBuffer state_in  = f32_buffer(state_elements, 0.0f);
        DeviceBuffer state_out = f32_buffer(state_elements, 0.0f);
        WorkspaceArena workspace(
            std::max<std::size_t>(ops::gated_delta_net_workspace_capacity_bytes(
                                      qk_heads, value_heads, true, 1, max_tokens),
                                  256));
        const auto run = [&](std::int32_t tokens, cudaStream_t stream) {
            const Tensor qt(q.p, DType::BF16, {state_dim, qk_heads, tokens});
            const Tensor kt(k.p, DType::BF16, {state_dim, qk_heads, tokens});
            const Tensor vt(v.p, DType::BF16, {state_dim, value_heads, tokens});
            const Tensor gt(g.p, DType::FP32, {value_heads, tokens});
            const Tensor bt(beta.p, DType::FP32, {value_heads, tokens});
            const Tensor st_in(state_in.p, DType::FP32, {state_dim, state_dim, value_heads});
            Tensor st_out(state_out.p, DType::FP32, {state_dim, state_dim, value_heads});
            Tensor ot(out.p, DType::BF16, {state_dim, value_heads, tokens});
            ops::gated_delta_net(qt, kt, vt, gt, bt, scale, true, workspace, st_in, st_out, ot,
                                 stream);
        };
        const Family family{"gdn_two_stage/h" + std::to_string(value_heads),
                            widths,
                            run,
                            [&](std::int32_t tokens) {
                                return std::pair<const void*, std::size_t>{
                                    out.p,
                                    static_cast<std::size_t>(state_dim) * value_heads * tokens};
                            },
                            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            // Past the widest measured prompt both algorithms scale linearly in its chunks.
            bands.back().last          = std::numeric_limits<std::int32_t>::max();
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(
                std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

void calibrate_attention(ops::DeviceRouteProfile& profile, Timer& timer,
                         const CalibrationOptions& options, KvCacheStorage storage,
                         const char* coding) {
    constexpr std::int32_t kLongestEnvelope = 4 * 262144;
    constexpr std::int32_t head_dim = 256;
    constexpr std::int32_t q_heads  = 24;
    constexpr std::int32_t kv_heads = 4;
    const std::vector<std::int32_t> envelopes = {8192, 65536, 262144};
    const std::int32_t max_window            = envelopes.back();
    const PagedKVStorageLayout layout        = paged_kv_storage_layout(storage, head_dim);
    const std::int32_t pages                 = max_window / kPagedKVPageSize;
    const auto plane_bytes = [&](std::int32_t leading, DType dtype) {
        return static_cast<std::size_t>(leading) * kv_heads * kPagedKVPageSize * pages * dtype_size(dtype);
    };
    DeviceBuffer k_pages(plane_bytes(layout.key.data_leading_extent, layout.key.data_dtype));
    DeviceBuffer v_pages(plane_bytes(layout.value.data_leading_extent, layout.value.data_dtype));
    DeviceBuffer k_scale(layout.key.has_scale()
                             ? plane_bytes(layout.key.scale_leading_extent, layout.key.scale_dtype)
                             : 256);
    DeviceBuffer v_scale(layout.value.has_scale()
                             ? plane_bytes(layout.value.scale_leading_extent, layout.value.scale_dtype)
                             : 256);
    fill_random(k_pages.p, k_pages.bytes, 0x1234u);
    fill_random(v_pages.p, v_pages.bytes, 0x5678u);
    // Scales of about 1/64 keep the scores and values in a realistic range.
    fill_u16_kernel<<<fill_grid(k_scale.bytes / 2), 256>>>(static_cast<std::uint16_t*>(k_scale.p),
                                                           k_scale.bytes / 2, 0x2400);
    fill_u16_kernel<<<fill_grid(v_scale.bytes / 2), 256>>>(static_cast<std::uint16_t*>(v_scale.p),
                                                           v_scale.bytes / 2, 0x2400);
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::int32_t> identity(static_cast<std::size_t>(pages));
    for (std::int32_t page = 0; page < pages; ++page) { identity[static_cast<std::size_t>(page)] = page; }
    DeviceBuffer block_table(identity.size() * sizeof(std::int32_t));
    block_table.copy_from_host(identity.data(), block_table.bytes);
    DeviceBuffer q = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * 8);
    DeviceBuffer out = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * 8);
    DeviceBuffer positions(8 * sizeof(std::int32_t));

    const ops::AttentionHeadGeometry geometry{head_dim, q_heads, kv_heads};
    const auto cache_view = [&](std::int32_t window) {
        const std::int32_t logical = window / kPagedKVPageSize;
        return PagedKVLayerView{
            .k_pages = Tensor(k_pages.p, layout.key.data_dtype,
                              {layout.key.data_leading_extent, kPagedKVPageSize, kv_heads, pages}),
            .v_pages = Tensor(v_pages.p, layout.value.data_dtype,
                              {layout.value.data_leading_extent, kPagedKVPageSize, kv_heads, pages}),
            .k_scale_pages = layout.key.has_scale()
                                 ? Tensor(k_scale.p, layout.key.scale_dtype,
                                          {layout.key.scale_leading_extent, kPagedKVPageSize, kv_heads,
                                           pages})
                                 : Tensor(),
            .v_scale_pages = layout.value.has_scale()
                                 ? Tensor(v_scale.p, layout.value.scale_dtype,
                                          {layout.value.scale_leading_extent, kPagedKVPageSize,
                                           kv_heads, pages})
                                 : Tensor(),
            .block_table  = Tensor(block_table.p, DType::I32, {logical}),
            .head_dim     = head_dim,
            .num_kv_heads = kv_heads,
            .storage      = storage,
        };
    };

    // FP16-accumulated PV per key tile: one 1024-token prompt chunk at 32K and a decode step at 131K.
    if (storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        constexpr std::int32_t chunk = 1024;
        // The fast-kernel comparison runs the chunk the planner aligns to the fast kernel's waves:
        // whole waves of 128-row CTAs are whole waves of the standard kernel's 64-row ones too.
        const std::int32_t wave_chunk =
            ops::causal_softmax_attention_prompt_aligned_chunk(geometry, storage, true, chunk);
        const std::int32_t max_chunk = std::max(chunk, wave_chunk);
        DeviceBuffer prompt_q   = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * max_chunk);
        DeviceBuffer prompt_out = bf16_buffer(static_cast<std::size_t>(head_dim) * q_heads * max_chunk);
        DeviceBuffer prompt_positions(max_chunk * sizeof(std::int32_t));
        // The fast kernel's key splits take workspace; size it for the fast kernel too, which the
        // "attn_prompt_fast" sweep below may select.
        const auto prompt_capacity = [&] {
            return ops::causal_softmax_attention_workspace_capacity_bytes(
                geometry, storage, {1, static_cast<std::uint32_t>(max_window)}, 1, 1, max_chunk);
        };
        std::size_t workspace_bytes = prompt_capacity();
        {
            const ops::DeviceRouteForce fast_prompt("attn_prompt_fast", "on");
            workspace_bytes = std::max(workspace_bytes, prompt_capacity());
        }
        WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
        const auto run = [&](std::int32_t tokens, std::int32_t depth, cudaStream_t stream) {
            std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
            for (std::int32_t token = 0; token < tokens; ++token) {
                host[static_cast<std::size_t>(token)] = depth - tokens + token;
            }
            CUDA_CHECK(cudaMemcpyAsync(prompt_positions.p, host.data(),
                                       host.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       stream));
            Tensor qt(prompt_q.p, DType::BF16, {head_dim, q_heads, tokens});
            Tensor ot(prompt_out.p, DType::BF16, {head_dim, q_heads, tokens});
            Tensor pt(prompt_positions.p, DType::I32, {tokens});
            auto scope = workspace.scope();
            ops::causal_softmax_attention_cached(qt, pt, geometry, 0.0625f, cache_view(max_window),
                                                 {1, static_cast<std::uint32_t>(max_window)},
                                                 workspace, ot, stream);
        };
        const Family family{"attn_pv_f16",
                            {1},
                            [&](std::int32_t, cudaStream_t stream) {
                                run(chunk, 32768, stream);
                                run(1, 131072, stream);
                            },
                            [&](std::int32_t) {
                                return std::pair<const void*, std::size_t>{
                                    prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                            },
                            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
        // A card without the standard INT8 prompt kernel (causal_prompt_i8_standard) runs every
        // INT8-family prompt on the fast one and has neither switch.
        if (ops::detail::causal_prompt_i8_standard_here()) {
            // The standard prompt kernel with each KV head's query heads packed into its tiles, as
            // the PV choice above left it: one 1024-token chunk at 32K and one at 131K of context.
            const Family pack{"attn_pack_gqa",
                              {1},
                              [&](std::int32_t, cudaStream_t stream) {
                                  run(chunk, 32768, stream);
                                  run(chunk, 131072, stream);
                              },
                              [&](std::int32_t) {
                                  return std::pair<const void*, std::size_t>{
                                      prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                              },
                              {}};
            auto pack_bands = sweep(pack, timer, options);
            if (!pack_bands.empty()) {
                profile.routes[pack.key] = std::move(pack_bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
            // The fast prompt kernel against the standard one (as the PV and packing choices above
            // left it): one wave-aligned chunk at 32K and one at 131K of context.
            const Family fast{"attn_prompt_fast",
                              {1},
                              [&](std::int32_t, cudaStream_t stream) {
                                  run(wave_chunk, 32768, stream);
                                  run(wave_chunk, 131072, stream);
                              },
                              [&](std::int32_t) {
                                  return std::pair<const void*, std::size_t>{
                                      prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                              },
                              {}};
            auto fast_bands = sweep(fast, timer, options);
            if (!fast_bands.empty()) {
                profile.routes[fast.key] = std::move(fast_bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
        }
        // Key splits of the fast prompt kernel (prompt_i8_fast_plan.h) against unsplit launches,
        // where the fast kernel now serves prompts: a 256-token continuation at 32K and 131K of
        // context, where few row blocks leave SMs idle, and a wave-aligned chunk at 131K. The
        // split partials need their own workspace, sized with the fast kernel selected.
        if (ops::detail::causal_attention_prompt_fast_kernel(storage, false)) {
            constexpr std::int32_t kContinuation = 256;
            const ops::CausalAttentionExecutionEnvelope split_envelope{
                1, static_cast<std::uint32_t>(max_window)};
            WorkspaceArena split_workspace(std::max<std::size_t>(
                ops::causal_softmax_attention_workspace_capacity_bytes(
                    geometry, storage, split_envelope, 1, kContinuation, max_chunk),
                256));
            const auto attend = [&](std::int32_t tokens, std::int32_t depth, cudaStream_t stream) {
                std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
                for (std::int32_t token = 0; token < tokens; ++token) {
                    host[static_cast<std::size_t>(token)] = depth - tokens + token;
                }
                CUDA_CHECK(cudaMemcpyAsync(prompt_positions.p, host.data(),
                                           host.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, stream));
                Tensor qt(prompt_q.p, DType::BF16, {head_dim, q_heads, tokens});
                Tensor ot(prompt_out.p, DType::BF16, {head_dim, q_heads, tokens});
                Tensor pt(prompt_positions.p, DType::I32, {tokens});
                auto scope = split_workspace.scope();
                ops::causal_softmax_attention_cached(qt, pt, geometry, 0.0625f,
                                                     cache_view(max_window), split_envelope,
                                                     split_workspace, ot, stream);
            };
            const Family split{"attn_prompt_split",
                               {1},
                               [&](std::int32_t, cudaStream_t stream) {
                                   attend(kContinuation, 32768, stream);
                                   attend(kContinuation, 131072, stream);
                                   attend(wave_chunk, 131072, stream);
                               },
                               [&](std::int32_t) {
                                   return std::pair<const void*, std::size_t>{
                                       prompt_out.p, static_cast<std::size_t>(head_dim) * q_heads};
                               },
                               {}};
            auto split_bands = sweep(split, timer, options);
            if (!split_bands.empty()) {
                profile.routes[split.key] = std::move(split_bands);
                ops::install_device_route_profile(
                    std::make_shared<const ops::DeviceRouteProfile>(profile));
            }
        }
        // Parallel query tiles against the serial chunks for single-row verification: 16 and 32
        // columns (two and four 8-column tiles) at 32K and 131K of context. The serial and the
        // tiled workspaces differ, so the arena takes the larger of the two.
        constexpr std::int32_t kTiledWidth = 32;
        const ops::CausalAttentionExecutionEnvelope verify_envelope{
            .min_visible_keys = 1,
            .max_visible_keys = static_cast<std::uint32_t>(max_window),
            .wide_verification = true,
        };
        const auto verify_capacity = [&] {
            return ops::causal_softmax_attention_workspace_capacity_bytes(
                geometry, storage, verify_envelope, 1, 1, kTiledWidth);
        };
        std::size_t tiled_bytes = verify_capacity();
        {
            auto probe = std::make_shared<ops::DeviceRouteProfile>(profile);
            probe->routes["attn_parallel_tiles"] = {ops::DeviceRouteBand{1, "on"}};
            ops::install_device_route_profile(probe);
            tiled_bytes = std::max(tiled_bytes, verify_capacity());
            ops::install_device_route_profile(
                std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
        WorkspaceArena tiled_workspace(tiled_bytes);
        const auto verify = [&](std::int32_t tokens, std::int32_t depth, cudaStream_t stream) {
            std::vector<std::int32_t> host(static_cast<std::size_t>(tokens));
            for (std::int32_t token = 0; token < tokens; ++token) {
                host[static_cast<std::size_t>(token)] = depth - tokens + token;
            }
            CUDA_CHECK(cudaMemcpyAsync(prompt_positions.p, host.data(),
                                       host.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       stream));
            Tensor qt(prompt_q.p, DType::BF16, {head_dim, q_heads, tokens});
            Tensor ot(prompt_out.p, DType::BF16, {head_dim, q_heads, tokens});
            Tensor pt(prompt_positions.p, DType::I32, {tokens});
            auto scope = tiled_workspace.scope();
            ops::causal_softmax_attention_cached(qt, pt, geometry, 0.0625f, cache_view(max_window),
                                                 verify_envelope, tiled_workspace, ot, stream);
        };
        const Family tiles{"attn_parallel_tiles",
                           {1},
                           [&](std::int32_t, cudaStream_t stream) {
                               verify(16, 32768, stream);
                               verify(kTiledWidth, 32768, stream);
                               verify(16, 131072, stream);
                               verify(kTiledWidth, 131072, stream);
                           },
                           [&](std::int32_t) {
                               return std::pair<const void*, std::size_t>{
                                   prompt_out.p,
                                   static_cast<std::size_t>(head_dim) * q_heads * kTiledWidth};
                           },
                           {}};
        auto tile_bands = sweep(tiles, timer, options);
        if (!tile_bands.empty()) {
            profile.routes[tiles.key] = std::move(tile_bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }

    for (std::int32_t width = 1; width <= 8; ++width) {
        const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
            geometry, storage, {1, static_cast<std::uint32_t>(max_window)}, 1, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
        const auto attend = [&](std::int32_t envelope, std::int32_t depth, cudaStream_t stream) {
            std::vector<std::int32_t> host(static_cast<std::size_t>(width));
            for (std::int32_t token = 0; token < width; ++token) {
                host[static_cast<std::size_t>(token)] = depth - width + token;
            }
            CUDA_CHECK(cudaMemcpyAsync(positions.p, host.data(), host.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            Tensor qt(q.p, DType::BF16, {head_dim, q_heads, width});
            Tensor ot(out.p, DType::BF16, {head_dim, q_heads, width});
            Tensor pt(positions.p, DType::I32, {width});
            const PagedKVLayerView cache = cache_view(envelope);
            auto scope                   = workspace.scope();
            ops::causal_softmax_attention_cached(
                qt, pt, geometry, 0.0625f, cache,
                {1, static_cast<std::uint32_t>(envelope)}, workspace, ot, stream);
        };
        const Family family{
            "attn_i8_small/h24/" + std::string(coding) + "/w" + std::to_string(width), envelopes,
            [&](std::int32_t envelope, cudaStream_t stream) {
                attend(envelope, envelope, stream);
                attend(envelope, std::max(width + 1, envelope / 16), stream);
            },
            [&](std::int32_t) {
                return std::pair<const void*, std::size_t>{
                    out.p, static_cast<std::size_t>(head_dim) * q_heads * width};
            },
            {}};
        auto bands = sweep(family, timer, options);
        if (!bands.empty()) {
            // Past the native window (YaRN, up to four times it) the kernel is as memory-bound as at
            // the widest measured envelope, so that envelope's tier serves those windows too.
            if (bands.back().last == max_window) { bands.back().last = kLongestEnvelope; }
            profile.routes[family.key] = std::move(bands);
            ops::install_device_route_profile(std::make_shared<const ops::DeviceRouteProfile>(profile));
        }
    }
}

} // namespace

ops::DeviceRouteProfile calibrate_device_routes(const CalibrationOptions& options) {
    const auto started = std::chrono::steady_clock::now();
    ops::DeviceRouteProfile profile;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    profile.multiprocessors = properties.multiProcessorCount;
    profile.origin          = "calibration";
    Timer timer(options);
    // The families outside a Volta card's catalog (calibration/route_catalog.cpp) steer routes it
    // never takes.
    const bool projections = !ops::current_device_volta();
    const bool attention   = projections && options.attention;
    if (projections && options.ternary) { calibrate_ternary(profile, timer, options); }
    if (projections && options.ternary) { calibrate_ptq1(profile, timer, options); }
    if (projections && options.groupwise) {
        calibrate_groupwise(profile, timer, options);
        calibrate_cublas_prefill(profile, timer, options);
        calibrate_q8_ksplit(profile, timer, options);
    }
    if (projections && options.nvfp4) { calibrate_nvfp4_fused(profile, timer, options); }
    if (options.linear_attention) { calibrate_linear_attention(profile, timer, options); }
    if (attention) {
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                            "rk8v4");
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedLloyd4KeyInt4Value,
                            "rk4v4");
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedInt4KeyInt4ValueE8,
                            "rk4v4-e8");
        calibrate_attention(profile, timer, options, KvCacheStorage::RotatedE8RootKeyInt4Value,
                            "rk2v4-e8");
        calibrate_attention(profile, timer, options, KvCacheStorage::Int8Group64, "int8");
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::size_t routed = 0;
    for (const auto& [key, bands] : profile.routes) {
        routed += std::any_of(bands.begin(), bands.end(),
                              [](const ops::DeviceRouteBand& band) { return !band.schedule.empty(); });
    }
    char line[160];
    std::snprintf(line, sizeof(line),
                  "calibration: %zu keys measured, %zu routed off the compiled route in %.1f s",
                  profile.routes.size(), routed, seconds);
    report(options, line);
    return profile;
}

} // namespace ninfer::calibration
