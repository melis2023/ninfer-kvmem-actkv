// Adapted from JGamboa/ninfer-4090-windows@e905787f (Apache-2.0); see NOTICE.
#include "ops/linear/t5/t5_project.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 1024;
constexpr int kThreads = 256; // one four-column code word per thread

// One CTA per (1024-column block, token). Thread tid decodes the four columns 4 tid .. 4 tid + 3,
// word w = tid % 16 of unit tid / 16 (t5_a8.cuh): trit m of unit bytes 4 g .. 4 g + 3 for
// w = 5 g + m < 15, byte 12 for w = 15. Then, for a rotated table (explicit signs argument), the
// logical W' H S row transform: the FP32 shared-memory Walsh-Hadamard butterfly of ops/hadamard,
// the 1/32 normalization and the signs. One BF16 rounding of the result.
__global__ void __launch_bounds__(kThreads)
    t5_embedding_kernel(const std::int32_t* __restrict__ ids, const std::uint8_t* __restrict__ codes,
                        const __half* __restrict__ scales, std::int64_t scale_row_halves,
                        const __nv_bfloat16* __restrict__ signs, int k,
                        __nv_bfloat16* __restrict__ out) {
    __shared__ float values[kBlock];
    const int tid     = static_cast<int>(threadIdx.x);
    const int column0 = static_cast<int>(blockIdx.x) * kBlock;
    const int token   = static_cast<int>(blockIdx.y);
    const std::int64_t row = ids[token];
    const int unit = column0 / 64 + tid / 16;
    const int word = tid % 16;
    const std::uint8_t* bytes = codes + row * (std::int64_t(k) / 64 * 13) + std::int64_t(unit) * 13;
    std::uint32_t trits = 0;
    if (word < 15) {
        const int g = word / 5, m = word % 5;
        std::uint32_t even = std::uint32_t(bytes[4 * g]) | (std::uint32_t(bytes[4 * g + 2]) << 16);
        std::uint32_t odd  = std::uint32_t(bytes[4 * g + 1]) | (std::uint32_t(bytes[4 * g + 3]) << 16);
        for (int step = 0; step <= m; ++step) {
            even *= 3u;
            odd *= 3u;
            trits = ((even >> 8) & 0x00030003u) | (odd & 0x03000300u);
            even &= 0x00ff00ffu;
            odd &= 0x00ff00ffu;
        }
    } else {
        std::uint32_t r = bytes[12];
        for (int m = 0; m < 4; ++m) {
            r *= 3u;
            trits |= (r >> 8) << (8 * m);
            r &= 0xffu;
        }
    }
    const float scale = __half2float(scales[row * scale_row_halves + (column0 + 4 * tid) / 128]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        values[4 * tid + j] = static_cast<float>(static_cast<int>((trits >> (8 * j)) & 0xffu) - 1) * scale;
    }
    __nv_bfloat16* destination = out + std::int64_t(token) * k + column0;
    if (signs == nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) destination[4 * tid + j] = __float2bfloat16_rn(values[4 * tid + j]);
        return;
    }
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < kBlock; stride <<= 1) {
#pragma unroll
        for (int pair = tid; pair < kBlock / 2; pair += kThreads) {
            const int low        = (pair / stride) * 2 * stride + pair % stride;
            const float a        = values[low];
            const float b        = values[low + stride];
            values[low]          = a + b;
            values[low + stride] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kBlock / kThreads; ++i) {
        const int index = tid + i * kThreads;
        destination[index] = __float2bfloat16_rn(values[index] * 0x1p-5f *
                                                 __bfloat162float(signs[column0 + index]));
    }
}

} // namespace

void t5_embedding(const Tensor& ids, const Weight& table, const Tensor& signs, Tensor& out, cudaStream_t stream) {
    validate_t5_weight(table, "embedding");
    const int tokens = ids.ne[0];
    if (ids.dtype != DType::I32 || out.dtype != DType::BF16 || out.ne[0] != table.k ||
        out.ne[1] != tokens || !ids.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("embedding: t5 expects I32 ids [T] and BF16 out [K,T]");
    }
    if (tokens == 0) return;
    const dim3 grid(static_cast<unsigned>(table.k / kBlock), static_cast<unsigned>(tokens));
    t5_embedding_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __half*>(table.scales), table.scale_nb[1] / 2,
        static_cast<const __nv_bfloat16*>(signs.data), table.k,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
