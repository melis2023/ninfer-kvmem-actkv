#include "ops/linear/bf16/bf16_shapes.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/token_slices.h"
#include "ops/linear/bf16/bf16_n256_k5120.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch_chunk(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
                  std::int32_t tokens, cudaStream_t stream) {
    constexpr int kRowTiles = 256 / Schedule::kOutputRowsPerCta;
    const dim3 grid(kRowTiles, (tokens + Schedule::kTileTokens - 1) / Schedule::kTileTokens);
    raise_dynamic_shared_memory(bf16_n256_k5120_mma_kernel<Schedule>, Schedule::kSharedBytes);
    CUDA_CHECK(
        pdl::launch_consumer({dim3(grid), dim3(Schedule::kThreads), Schedule::kSharedBytes, stream},
                             bf16_n256_k5120_mma_kernel<Schedule>, x, weight, out, tokens));
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_grid(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kTileTokens, [&](int offset, int count) {
        launch_chunk<Schedule>(
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(offset) * 5120,
            static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<__nv_bfloat16*>(out.data) + static_cast<std::int64_t>(offset) * 256, count,
            stream);
    });
}
} // namespace

// Narrow widths split K sixteen ways where those warps' 96 KiB of staging fit a block, and eight
// ways in Turing's shape set (64 KiB).
template <bool TuringShapes>
constexpr int kNarrowKWarps =
    bf16_n256_k5120_staging_bytes(16, 8) <= shape_max_shared_bytes_per_block(TuringShapes) ? 16 : 8;

Bf16Launch select_bf16_n256_k5120(std::int32_t tokens) {
    if (tokens <= 76) {
        return with_shape_set([]<bool TuringShapes>() -> Bf16Launch {
            return launch_grid<Bf16N256K5120MmaSchedule<kNarrowKWarps<TuringShapes>, 8>>;
        });
    }
    return launch_grid<Bf16N256K5120MmaSchedule<8, 16>>;
}

} // namespace ninfer::ops::detail
