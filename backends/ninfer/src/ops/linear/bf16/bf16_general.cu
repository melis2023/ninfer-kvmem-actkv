#include "ops/linear/bf16/bf16_general.cuh"
#include "core/device.h"
#include "ops/linear/bf16/bf16_dispatch.h"

#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// From this width a runtime-shape product takes cuBLAS's tensor-core BF16 GEMM (FP32 accumulator)
// instead of the SIMT tile below; narrower ones, the widths decode graphs capture, keep the kernels
// of this file.
constexpr std::int32_t kCublasMinTokens = 64;

cublasHandle_t blas_handle() {
    static std::array<cublasHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= int(handles.size())) {
        throw std::invalid_argument("bf16 linear: device ordinal out of range");
    }
    std::call_once(once[device], [&] {
        if (cublasCreate(&handles[device]) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("bf16 linear: cublasCreate failed");
        }
    });
    return handles[device];
}

// out[T, N] = x[T, K] * weight[N, K]^T: column-major, the weight is a K x N matrix read
// transposed, x a K x T one and out an N x T one, BF16 or FP32.
void bf16_cublas_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const cublasHandle_t blas = blas_handle();
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("bf16 linear: cublasSetStream failed");
    }
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, weight.n, x.ne[1], weight.k, &alpha, weight.qdata,
        CUDA_R_16BF, weight.k, x.data, CUDA_R_16BF, weight.k, &beta, out.data,
        out.dtype == DType::FP32 ? CUDA_R_32F : CUDA_R_16BF, weight.n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("bf16 linear: cublasGemmEx failed");
    }
}

template <int T, class Out>
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    bf16_general_gemv_kernel<T, Out>
        <<<static_cast<unsigned>((N + kGemvWarps - 1) / kGemvWarps), kGemvWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), static_cast<Out*>(out.data), N,
            weight.k);
    CUDA_CHECK(cudaGetLastError());
}

bool aligned16(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0; }

// Runtime-shape bf16 GEMM used for every linear projection the specialised bf16 kernels do not
// tile and every FP32 result. x is [tokens, k], weight is [n, k] row-major, and out is [tokens, n]
// token-major, matching the contiguous layout the specialised kernels produce.
template <class Out>
void bf16_general_gemm_as(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    const std::int32_t K = weight.k;
    const std::int32_t T = x.ne[1];
    if (T >= kCublasMinTokens) { return bf16_cublas_gemm(x, weight, out, stream); }
    if (T <= kGemvMaxTokens && K % 8 == 0 && aligned16(x.data) && aligned16(weight.qdata)) {
        switch (T) {
        case 1:
            return launch_gemv<1, Out>(x, weight, out, stream);
        case 2:
            return launch_gemv<2, Out>(x, weight, out, stream);
        case 3:
            return launch_gemv<3, Out>(x, weight, out, stream);
        case 4:
            return launch_gemv<4, Out>(x, weight, out, stream);
        case 5:
            return launch_gemv<5, Out>(x, weight, out, stream);
        case 6:
            return launch_gemv<6, Out>(x, weight, out, stream);
        case 7:
            return launch_gemv<7, Out>(x, weight, out, stream);
        default:
            return launch_gemv<8, Out>(x, weight, out, stream);
        }
    }
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((T + kTileT - 1) / kTileT));
    bf16_general_gemm_kernel<Out><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<Out*>(out.data), N, K, T);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_general_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::FP32) { return bf16_general_gemm_as<float>(x, weight, out, stream); }
    bf16_general_gemm_as<__nv_bfloat16>(x, weight, out, stream);
}

} // namespace

[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens) {
    (void)tokens;
    return &bf16_general_gemm;
}

} // namespace ninfer::ops::detail
