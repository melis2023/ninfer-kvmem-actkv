#pragma once

// The CPU's share of a streamed native SparseMoe call (streamed.h): the routed experts of a narrow
// call that ExpertDelivery leaves to the CPU, computed from their stored planes in host memory bit
// for bit as the GPU's sparse_moe_stream_d3 and sparse_moe_stream_d4 kernels compute them, so that
// how a call splits its misses never changes a result.
//
// Each output row is one warp's dot product. Lane l of 32 owns the values 256 i + 8 l + k
// (k = 0..7) of each 256-value step i and adds their products w * x in that order by fused
// multiply-adds; the lanes then meet as in the warp's shfl_down tree (lane l takes lane l + 16,
// then l + 8, l + 4, l + 2, l + 1). Every routed format has this ownership. The weights are the
// stored codes decoded exactly: Q4, Q5 and Q6 as signed integers times the 64-value group's
// binary16 scale, Q8 the same per 32 values, NVFP4 as e2m1 values times (the 16-value block's e4m3
// scale times the correctly rounded reciprocal of the row's divisor). Gate and up rows take the
// BF16 input, the middle is ops::portable_silu(gate) * up in FP32 (ops/common/silu.h, as the
// kernels compute it), the down rows take the middle, and a pair's product row is its route weight
// times each down dot. These translation units build without floating-point contraction, so no
// step fuses unless written as a fused multiply-add.

#include "core/expert_delivery.h"
#include "core/weight.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::ops::detail {

// One routed projection of one expert in host memory, laid out as a device slot
// (models/qwen3_5/expert_layout.h): codes, Q5/Q6 high bits and scales row by row, NVFP4 scales in
// their 128x4 tiles and one divisor per `divisor_rows` rows.
struct StreamCpuPlane {
    QType format               = QType::Q4_G64_FP16;
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* high   = nullptr;
    const std::uint8_t* scales = nullptr;
    const float* divisors      = nullptr;
    int divisor_rows           = 0;
    int rows = 0, k = 0;
};

// Whether the CPU computes routed experts of these gate/up and down formats (the streamed
// decode's pairs: Q4 with Q5 or Q6, Q8 with Q8, NVFP4 with NVFP4).
[[nodiscard]] bool stream_cpu_supported(QType gate_up, QType down) noexcept;

// Up to `capacity` columns of k values (a positive multiple of 256), each in the lanes' order:
// position (8 i + k) * 32 + l holds value 256 i + 8 l + k.
class StreamCpuInput {
public:
    StreamCpuInput() = default;
    StreamCpuInput(int capacity, int k);
    // Column `column` (below the capacity) from k BF16 words or FP32 values; columns() becomes at
    // least column + 1. Threads may load columns below columns() at once.
    void load_bf16(int column, const std::uint16_t* values);
    void load_fp32(int column, const float* values);
    void clear() noexcept { columns_ = 0; }

    void set_columns(int columns);
    [[nodiscard]] int columns() const noexcept { return columns_; }
    [[nodiscard]] int capacity() const noexcept { return capacity_; }
    [[nodiscard]] int k() const noexcept { return k_; }
    [[nodiscard]] const float* lanes(int column) const noexcept {
        return lanes_.data() + std::size_t(column) * k_;
    }

private:
    int capacity_ = 0, k_ = 0, columns_ = 0;
    std::vector<float> lanes_;
};

// y[t][r] = coefficients[t] * (row r . x[t]) for rows [row_begin, row_end) and every column t of
// x, or the dot itself where `coefficients` is null. backend Automatic takes AVX2 (with FMA and
// F16C) where the CPU has it; every backend gives the same bits.
void stream_cpu_rows(const StreamCpuPlane& w, const StreamCpuInput& x, int row_begin, int row_end,
                     const float* coefficients, std::span<float* const> y, GgufCpuBackend backend);
// out[i] = ops::portable_silu(gate[i]) * up[i].
void stream_cpu_swiglu(const float* gate, const float* up, std::size_t count, float* out) noexcept;

// The aliases of one expert layer's blobs (models/qwen3_5/expert_sources.h): gate/up codes, high
// bits, scales and divisors, then down's four. An unused high or divisor alias is never read.
struct StreamCpuLayer {
    QType gate_up = QType::Q4_G64_FP16, down = QType::Q6_G64_FP16;
    std::array<ExpertAlias, 8> aliases{};
};

// ExpertDelivery's CPU products for sparse_moe_stream. A call passes begin() its input x (BF16
// [tokens][2048]) and its route weights (FP32 [tokens][8]); compute() writes one FP32 product row
// of 2048 per CPU pair, in the order of the pairs, and sparse_moe_stream places each into its
// pair's slot before the merge.
class SparseMoeStreamCpu final : public ExpertCpuProducts {
public:
    explicit SparseMoeStreamCpu(std::vector<StreamCpuLayer> layers,
                                GgufCpuBackend backend = GgufCpuBackend::Automatic);

    // Whether any layer of these formats has CPU products.
    [[nodiscard]] static bool computes(const StreamCpuLayer& layer) noexcept;
    // ExpertCpuOptions' sizes: the widest Small-T call's inputs and products.
    [[nodiscard]] static std::size_t input_bytes() noexcept;
    [[nodiscard]] static std::size_t product_bytes() noexcept;
    [[nodiscard]] const char* backend_name() const noexcept;

    [[nodiscard]] bool supports(std::size_t layer) const override;
    Done compute(const Work& work) override;

private:
    struct Job {
        std::int32_t expert = -1;
        std::vector<int> pairs; // ascending; one per token routed to the expert
        StreamCpuPlane gate_up, down;
        StreamCpuInput input, middle;
        std::vector<float> gate_up_rows, middle_values, coefficients;
        std::vector<float*> gate_up_out, products;
    };

    std::vector<StreamCpuLayer> layers_;
    GgufCpuBackend backend_;
    std::vector<Job> jobs_;
};

} // namespace ninfer::ops::detail
