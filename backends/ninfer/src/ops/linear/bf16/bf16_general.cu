#include "ops/linear/bf16/bf16_general.cuh"
#include "core/device.h"
#include "ops/linear/bf16/bf16_dispatch.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <int T>
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    bf16_general_gemv_kernel<T>
        <<<static_cast<unsigned>((N + kGemvWarps - 1) / kGemvWarps), kGemvWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), static_cast<__nv_bfloat16*>(out.data),
            N, weight.k);
    CUDA_CHECK(cudaGetLastError());
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// Runtime-shape bf16 GEMM used for every linear projection the specialised bf16 kernels do not
// tile. x is [tokens, k], weight is [n, k] row-major, and out is [tokens, n] token-major, matching
// the contiguous layout the specialised kernels produce.
void bf16_general_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    const std::int32_t K = weight.k;
    const std::int32_t T = x.ne[1];
    if (T <= kGemvMaxTokens && K % 8 == 0 && aligned16(x.data) && aligned16(weight.qdata)) {
        switch (T) {
        case 1:
            return launch_gemv<1>(x, weight, out, stream);
        case 2:
            return launch_gemv<2>(x, weight, out, stream);
        case 3:
            return launch_gemv<3>(x, weight, out, stream);
        case 4:
            return launch_gemv<4>(x, weight, out, stream);
        case 5:
            return launch_gemv<5>(x, weight, out, stream);
        case 6:
            return launch_gemv<6>(x, weight, out, stream);
        case 7:
            return launch_gemv<7>(x, weight, out, stream);
        default:
            return launch_gemv<8>(x, weight, out, stream);
        }
    }
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((T + kTileT - 1) / kTileT));
    bf16_general_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), N, K, T);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens) {
    (void)tokens;
    return &bf16_general_gemm;
}

} // namespace ninfer::ops::detail
