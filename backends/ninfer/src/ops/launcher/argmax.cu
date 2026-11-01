// Implements: include/ninfer/ops/argmax.h
// Match: validated contiguous BF16 or FP32 logits and I32 output.
// Algorithm assumptions: one tile uses a direct reduction; larger domains use
// zero-initialized atomic winners across route-selected row tiles.
#include "ops/launcher/argmax.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/argmax.cuh"
#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int tiled_block_for(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

template <typename Logit>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

template <typename Logit>
void argmax_typed_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                         cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    if (t_count == 0) { return; }

    constexpr int kTileElems = kArgmaxBlock * kArgmaxItemsPerThread;
    const int tiled_blocks   = div_up(valid_rows, kTileElems);
    if (tiled_blocks < 2) {
        argmax_kernel<Logit><<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const Logit*>(logits.data), static_cast<std::int32_t*>(out.data),
            valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    argmax_tiled_atomic_launch<Logit>(logits, out, valid_rows,
                                      tiled_block_for(physical_rows, valid_rows, t_count), stream);
}

} // namespace

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream) {
    if (logits.dtype == DType::FP32) {
        argmax_typed_launch<float>(logits, out, valid_rows, stream);
        return;
    }
    argmax_typed_launch<__nv_bfloat16>(logits, out, valid_rows, stream);
}

void argmax_second_launch(const Tensor& logits, const Tensor& first, Tensor& second,
                          std::int32_t valid_rows, cudaStream_t stream) {
    for_each_token_slice(logits.ne[1], 1, [&](int offset, int count) {
        const auto input = logits.slice(1, offset, count);
        const auto winner = first.slice(0, offset, count);
        auto runner = second.slice(0, offset, count);
        argmax_second_init<<<div_up(count, 256), 256, 0, stream>>>(
            static_cast<const std::int32_t*>(winner.data),
            static_cast<std::int32_t*>(runner.data), count);
        CUDA_CHECK(cudaGetLastError());
        const dim3 grid(div_up(valid_rows, kArgmaxBlock), count);
        argmax_second_kernel<<<grid, kArgmaxBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const std::int32_t*>(winner.data),
            static_cast<std::int32_t*>(runner.data), valid_rows, logits.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    });
}

namespace {

template <typename Logit>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * kArgmaxItemsPerThread);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(grid), dim3(block), 0, stream}, argmax_tiled_atomic_kernel<Logit>,
            static_cast<const Logit*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows));
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace
} // namespace ninfer::ops::detail
