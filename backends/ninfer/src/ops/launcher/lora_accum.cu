// Implements: include/ninfer/ops/lora_accum.h
// Finite dispatch: one pass-1 launch over (rank, splits, column tiles) and one pass-2 launch whose
// column tile is a compile-time constant, chosen from the next width at or above the token count.
#include "ops/launcher/lora_accum.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/lora_accum.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

int grid_ceil(std::int64_t work, std::int64_t per_block) {
    return static_cast<int>(std::max<std::int64_t>(1, (work + per_block - 1) / per_block));
}

template <int COLUMNS>
void launch_pass2(const float* scratch, const __half* b, __nv_bfloat16* out, int rows, int rank,
                  int tokens, int temp_stride, int splits, int out_ld, float scale,
                  cudaStream_t stream) {
    const std::size_t shared_bytes = static_cast<std::size_t>(rank) * COLUMNS * sizeof(float);
    const dim3 grid(grid_ceil(rows, kLoraPass2Rows), grid_ceil(tokens, COLUMNS));
    lora_pass2_kernel<COLUMNS><<<grid, kLoraPass2Rows, shared_bytes, stream>>>(
        scratch, b, out, rows, rank, tokens, temp_stride, splits, out_ld, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void lora_accum_launch(const Tensor& x, const Tensor& a, const Tensor& b, Tensor& out,
                       const Tensor& temp, float scale, cudaStream_t stream) {
    const int inner       = x.ne[0];
    const int tokens      = x.ne[1];
    const int rank        = a.ne[1];
    const int rows        = out.ne[0];
    const int temp_stride = temp.ne[0];
    // A row region of a wider plane writes with the parent's column stride.
    const int out_ld = static_cast<int>(out.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    auto* scratch    = static_cast<float*>(temp.data);

    const int splits = std::max(1, temp.ne[0] / rank);
    const dim3 block1(kLoraTileColumns, kLoraTileInner);
    const dim3 grid1(rank, splits, grid_ceil(tokens, kLoraTileColumns));
    lora_pass1_kernel<<<grid1, block1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __half*>(a.data), scratch, rank,
        inner, tokens, temp_stride, splits);
    CUDA_CHECK(cudaGetLastError());

    const auto* b_ptr   = static_cast<const __half*>(b.data);
    auto* out_ptr       = static_cast<__nv_bfloat16*>(out.data);
    // Bench-only override so one binary can measure every tile width: NINFER_LORA_COLUMNS=1|2|4|8|16|32
    // forces that width (clamped up to the token count), 0 keeps the automatic choice. Read once:
    // this launcher runs 256 times per forward and getenv walks the environment block.
    static const int forced = [] {
        const char* text = std::getenv("NINFER_LORA_COLUMNS");
        return text == nullptr ? 0 : std::atoi(text);
    }();
    if (forced > 0) {
        if (tokens <= 1 || forced == 1) {
            launch_pass2<1>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                            scale, stream);
        } else if (forced == 2) {
            launch_pass2<2>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                            scale, stream);
        } else if (forced == 4) {
            launch_pass2<4>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                            scale, stream);
        } else if (forced == 8) {
            launch_pass2<8>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                            scale, stream);
        } else if (forced == 16) {
            launch_pass2<16>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits,
                             out_ld, scale, stream);
        } else {
            launch_pass2<32>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits,
                             out_ld, scale, stream);
        }
        return;
    }
    if (tokens <= 1) {
        launch_pass2<1>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                        scale, stream);
    } else if (tokens <= 2) {
        launch_pass2<2>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                        scale, stream);
    } else if (tokens <= 4) {
        launch_pass2<4>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                        scale, stream);
    } else if (tokens <= 8) {
        launch_pass2<8>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                        scale, stream);
    } else if (tokens <= 16) {
        launch_pass2<16>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                         scale, stream);
    } else {
        launch_pass2<32>(scratch, b_ptr, out_ptr, rows, rank, tokens, temp_stride, splits, out_ld,
                         scale, stream);
    }
}

} // namespace ninfer::ops::detail
