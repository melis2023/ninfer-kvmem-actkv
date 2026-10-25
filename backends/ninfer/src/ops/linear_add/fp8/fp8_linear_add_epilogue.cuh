#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8AddResidualEpilogue {
    const __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t token,
                                           float value) const {
        return value + __bfloat162float(residual[static_cast<std::int64_t>(token) * rows + row]);
    }

    // One rounding for the scaled residual update, as ops/linear/common/epilogue.cuh states, so a
    // column stores the same update in a full tile and in a partial one.
    __device__ __forceinline__ float apply_scaled(std::int32_t row, std::int32_t token, float value,
                                                  float scale) const {
        return __fmaf_rn(value, scale,
                         __bfloat162float(residual[static_cast<std::int64_t>(token) * rows + row]));
    }
};

} // namespace ninfer::ops::detail
