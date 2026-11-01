#include "ninfer/ops/moe_expert_cpu.h"
#include "ops/moe_experts/cpu_projection.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

#if defined(_MSC_VER) && defined(NINFER_CPU_EXPERT_X86)
#    include <intrin.h>
#endif

namespace ninfer::ops {
namespace {
constexpr int kHidden = 2560, kWidth = 640;

struct ScalarDot {
    static int integer(const std::int8_t* weights, const std::int8_t* input) {
        int sum = 0;
        for (int i = 0; i < 32; ++i) { sum += int(weights[i]) * int(input[i]); }
        return sum;
    }
    static float floating(const std::int8_t* weights, const float* input) {
        float sum = 0;
        for (int i = 0; i < 32; ++i) { sum += float(weights[i]) * input[i]; }
        return sum;
    }
};

void validate(const CpuExpertProjection& projection, int rows, int columns) {
    const auto& w = projection.weight;
    if (w.n != rows || w.k != columns || !w.qdata || w.input_columns) {
        throw std::invalid_argument("CPU expert: invalid shape, bytes or input gather");
    }
    if (w.qtype == QType::BF16) {
        if (w.layout != QuantLayout::Contiguous || projection.integer_a8) {
            throw std::invalid_argument("CPU expert: BF16 requires contiguous A16 weights");
        }
        return;
    }
    if (w.qtype != QType::Q2_G64_FP16 && w.qtype != QType::Q4_G64_FP16 &&
        w.qtype != QType::Q5_G64_FP16 && w.qtype != QType::Q6_G64_FP16 &&
        w.qtype != QType::Q8_G32_FP16) {
        throw std::invalid_argument("CPU expert: unsupported weight format");
    }
    if ((w.layout != QuantLayout::RowSplit && w.layout != QuantLayout::RowSplitPanel) ||
        !w.scales || w.scale_dtype != DType::FP16 ||
        w.group != (w.qtype == QType::Q8_G32_FP16 ? 32 : 64) ||
        ((w.qtype == QType::Q5_G64_FP16 || w.qtype == QType::Q6_G64_FP16) && !w.qhigh)) {
        throw std::invalid_argument("CPU expert: incomplete native weight planes");
    }
}

void quantize(const float* values, std::size_t count, std::int8_t* codes, float* scales) {
    for (std::size_t start = 0; start < count; start += 32) {
        float maximum = 0;
        for (int i = 0; i < 32; ++i) {
            if (!std::isfinite(values[start + i])) {
                throw std::runtime_error("CPU expert: nonfinite activation");
            }
            maximum = std::max(maximum, std::abs(values[start + i]));
        }
        const float scale = maximum / 127.0F;
        scales[start / 32] = scale;
        for (int i = 0; i < 32; ++i) {
            codes[start + i] = scale == 0 ? 0 : static_cast<std::int8_t>(
                std::clamp(std::nearbyint(values[start + i] / scale), -127.0F, 127.0F));
        }
    }
}

CpuExpertBackend choose(CpuExpertBackend backend) noexcept {
    if (backend != CpuExpertBackend::Automatic) { return backend; }
    if (cpu_expert_backend_available(CpuExpertBackend::Avx512Vnni)) { return CpuExpertBackend::Avx512Vnni; }
    if (cpu_expert_backend_available(CpuExpertBackend::Avx2)) { return CpuExpertBackend::Avx2; }
    return CpuExpertBackend::Scalar;
}
} // namespace

void cpu_expert_detail::project_scalar(const Weight& w, const Activations& x, bool a8, float* y) {
    project<ScalarDot>(w, x, a8, y);
}

bool cpu_expert_backend_available(CpuExpertBackend backend) noexcept {
    if (backend == CpuExpertBackend::Automatic || backend == CpuExpertBackend::Scalar) { return true; }
    if (backend != CpuExpertBackend::Avx2 && backend != CpuExpertBackend::Avx512Vnni) { return false; }
#if defined(NINFER_CPU_EXPERT_X86)
#    if defined(_MSC_VER)
    int basic[4], extended[4];
    __cpuid(basic, 0);
    if (basic[0] < 7) { return false; }
    __cpuid(basic, 1);
    if ((basic[2] & (1 << 27)) == 0 || (basic[2] & (1 << 28)) == 0) { return false; }
    const auto enabled = _xgetbv(0);
    if ((enabled & 6) != 6) { return false; }
    __cpuidex(extended, 7, 0);
    if (backend == CpuExpertBackend::Avx2) { return (extended[1] & (1 << 5)) != 0; }
    return (enabled & 0xe6) == 0xe6 && (extended[1] & (1 << 16)) != 0 &&
        (extended[1] & (1 << 30)) != 0 && (extended[1] & (int(1U << 31))) != 0 &&
        (extended[2] & (1 << 11)) != 0;
#    else
    if (backend == CpuExpertBackend::Avx2) { return __builtin_cpu_supports("avx2"); }
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
#    endif
#else
    return false;
#endif
}

const char* cpu_expert_backend_name(CpuExpertBackend backend) noexcept {
    switch (choose(backend)) {
    case CpuExpertBackend::Avx2: return "avx2";
    case CpuExpertBackend::Avx512Vnni: return "avx512-vnni";
    default: return "scalar";
    }
}

CpuExpertWorkspace::CpuExpertWorkspace(std::uint32_t token_capacity) : capacity_(token_capacity) {
    if (capacity_ == 0 || capacity_ > 65535) { throw std::invalid_argument("CPU expert: capacity must be 1..65535"); }
    input_.resize(std::size_t(capacity_) * kHidden);
    gate_.resize(std::size_t(capacity_) * kWidth);
    middle_.resize(gate_.size());
    input_codes_.resize(input_.size());
    middle_codes_.resize(middle_.size());
    input_scales_.resize(input_.size() / 32);
    middle_scales_.resize(middle_.size() / 32);
}

void moe_expert_cpu(std::span<const std::uint16_t> x, const CpuExpertWeights& weights,
                    CpuExpertWorkspace& workspace, std::span<float> y, CpuExpertBackend backend,
                    const std::atomic<bool>* cancelled) {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) { throw CpuExpertCancelled(); }
    if (x.empty() || x.size() % kHidden != 0 || y.size() != x.size() ||
        x.size() / kHidden > workspace.capacity()) {
        throw std::invalid_argument("CPU expert: input, output or workspace shape mismatch");
    }
    validate(weights.gate, kWidth, kHidden);
    validate(weights.up, kWidth, kHidden);
    validate(weights.down, kHidden, kWidth);
    backend = choose(backend);
    if (!cpu_expert_backend_available(backend)) { throw std::invalid_argument("CPU expert: unsupported CPU backend"); }
    cpu_expert_detail::Project project = cpu_expert_detail::project_scalar;
#if defined(NINFER_CPU_EXPERT_X86)
    if (backend == CpuExpertBackend::Avx2) { project = cpu_expert_detail::project_avx2; }
    if (backend == CpuExpertBackend::Avx512Vnni) { project = cpu_expert_detail::project_avx512; }
#endif
    const int tokens = static_cast<int>(x.size() / kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) {
        workspace.input_[i] = std::bit_cast<float>(std::uint32_t(x[i]) << 16);
    }
    if (weights.gate.integer_a8 || weights.up.integer_a8) {
        quantize(workspace.input_.data(), x.size(), workspace.input_codes_.data(), workspace.input_scales_.data());
    }
    const cpu_expert_detail::Activations input{workspace.input_.data(), workspace.input_codes_.data(),
                                               workspace.input_scales_.data(), tokens, cancelled};
    project(weights.gate.weight, input, weights.gate.integer_a8, workspace.gate_.data());
    project(weights.up.weight, input, weights.up.integer_a8, workspace.middle_.data());
    const std::size_t middle_count = std::size_t(tokens) * kWidth;
    for (std::size_t i = 0; i < middle_count; ++i) {
        const float gate = workspace.gate_[i];
        workspace.middle_[i] *= gate / (1.0F + std::exp(-gate));
    }
    if (weights.down.integer_a8) {
        quantize(workspace.middle_.data(), middle_count, workspace.middle_codes_.data(), workspace.middle_scales_.data());
    }
    const cpu_expert_detail::Activations middle{workspace.middle_.data(), workspace.middle_codes_.data(),
                                                workspace.middle_scales_.data(), tokens, cancelled};
    project(weights.down.weight, middle, weights.down.integer_a8, y.data());
}
} // namespace ninfer::ops
