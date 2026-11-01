#pragma once

#include "core/weight.h"

#include <cstdint>
#include <atomic>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::ops {

struct CpuExpertCancelled final : std::runtime_error {
    CpuExpertCancelled() : std::runtime_error("CPU expert execution cancelled") {}
};

// Execution resource, selected once by the Program. Explicit choices allow every available
// production implementation to be qualified against the same independent oracle.
enum class CpuExpertBackend { Automatic, Scalar, Avx2, Avx512Vnni };
[[nodiscard]] bool cpu_expert_backend_available(CpuExpertBackend backend) noexcept;
[[nodiscard]] const char* cpu_expert_backend_name(CpuExpertBackend backend) noexcept;

struct CpuExpertProjection {
    Weight weight;
    // The caller must derive this permission from the projection's actual artifact Use.
    bool integer_a8 = false;
};

struct CpuExpertWeights {
    CpuExpertProjection gate, up, down;
};

class CpuExpertWorkspace {
public:
    explicit CpuExpertWorkspace(std::uint32_t token_capacity);
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

private:
    friend void moe_expert_cpu(std::span<const std::uint16_t>, const CpuExpertWeights&,
                               CpuExpertWorkspace&, std::span<float>, CpuExpertBackend,
                               const std::atomic<bool>*);
    std::uint32_t capacity_;
    std::vector<float> input_, gate_, middle_, input_scales_, middle_scales_;
    std::vector<std::int8_t> input_codes_, middle_codes_;
};

/**
 * One host-resident SwiGLU expert:
 *   y[t] = down . (silu(gate . x[t]) * (up . x[t])).
 *
 * x contains T contiguous BF16 rows of 2560 stored words; y receives T FP32 rows of 2560.
 * Gate/up are [640,2560], down [2560,640]. Host-readable Weight operands retain BF16 or
 * native Q2/Q4/Q5/Q6/Q8 planes, row-split or the native expert's 64-row panel layout.
 * Each projection's integer_a8 independently permits private group-32 int8 activations.
 * BF16 projections require A16. The middle stays private FP32, with no semantic cast.
 *
 * The oracle independently decodes the stored weights and evaluates the full formula in
 * FP64 from BF16 x; neither private activation quantization nor intermediate reductions
 * are oracle boundaries. Fixed operands/backend repeat exactly. T is 1..65535 and cannot
 * exceed the caller-owned workspace capacity. Inputs, output and workspace do not alias.
 * No allocation or worker creation occurs during successful execution. Program owns
 * expert selection, weighted merge, CPU/GPU partition and thread scheduling.
 * A borrowed cancellation flag is checked between output rows. Cancellation throws
 * CpuExpertCancelled, leaves output incomplete, and retains no reference to the flag.
 */
void moe_expert_cpu(std::span<const std::uint16_t> x, const CpuExpertWeights& weights,
                    CpuExpertWorkspace& workspace, std::span<float> y,
                    CpuExpertBackend backend = CpuExpertBackend::Automatic,
                    const std::atomic<bool>* cancelled = nullptr);

} // namespace ninfer::ops
