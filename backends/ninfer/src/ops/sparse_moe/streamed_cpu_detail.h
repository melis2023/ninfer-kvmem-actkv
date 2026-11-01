#pragma once

// The vector backends of streamed_cpu.h, which reproduce the scalar one bit for bit.

#include "ops/sparse_moe/streamed_cpu.h"

namespace ninfer::ops::detail::stream_cpu_detail {

// The exact value of an e4m3 code (NaN for its NaN codes).
[[nodiscard]] float e4m3_to_float(std::uint8_t bits) noexcept;
// The byte of NVFP4 scale block `block` (16 values) of `row` in the 128x4 tiles of a plane with
// `tiles_per_row` (k / 64) tiles per 128 rows (ops/linear/nvfp4/nvfp4_codec.cuh).
[[nodiscard]] std::int64_t nvfp4_scale_byte(int row, int block, int tiles_per_row) noexcept;
// Lane `lane`'s eight weights of 256-value step `step` of `row`, in item order.
void decode_lane(const StreamCpuPlane& w, int row, int step, int lane, float (&out)[8]) noexcept;

void rows_scalar(const StreamCpuPlane& w, const StreamCpuInput& x, int row_begin, int row_end,
                 const float* coefficients, std::span<float* const> y);
#if defined(NINFER_CPU_EXPERT_X86)
void rows_avx2(const StreamCpuPlane& w, const StreamCpuInput& x, int row_begin, int row_end,
               const float* coefficients, std::span<float* const> y);
#endif

} // namespace ninfer::ops::detail::stream_cpu_detail
