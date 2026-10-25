#include "ops/linear/sm70/sm70_gemm.h"

#include "core/device.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::sm70 {
namespace {

constexpr std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("sm70 GEMM: cuBLAS failure in ") + what + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// One handle per device, created on first use and kept for the process; the stream is set per call
// because the handle is shared. The first use can be inside a CUDA graph capture (a verify graph of
// several requests reaches the cuBLAS widths before any prefill has run); cublasCreate allocates,
// which relaxed capture mode lets this thread do without invalidating the capture.
cublasHandle_t handle_for_current_device() {
    static constexpr int kMaxDevices = 8;
    static cublasHandle_t handles[kMaxDevices]{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("sm70 GEMM: device index out of range");
    }
    if (!handles[device]) {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        const cublasStatus_t created = cublasCreate(&handles[device]);
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        check_blas(created, "cublasCreate");
        check_blas(cublasSetMathMode(handles[device], CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    }
    return handles[device];
}

__device__ __forceinline__ float saturate_f16(float value) {
    return fminf(fmaxf(value, -65504.0f), 65504.0f);
}

// Eight values per thread: one 16-byte BF16 load, one 16-byte FP16 store.
__global__ void convert_activation_kernel(const uint4* __restrict__ x, uint4* __restrict__ out,
                                          std::int64_t vectors) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vectors) { return; }
    const uint4 in = x[i];
    out[i]         = make_uint4(bf16x2_to_f16x2(in.x), bf16x2_to_f16x2(in.y), bf16x2_to_f16x2(in.z),
                                bf16x2_to_f16x2(in.w));
}

__global__ void convert_activation_tail_kernel(const __nv_bfloat16* __restrict__ x,
                                               __half* __restrict__ out, std::int64_t first,
                                               std::int64_t count) {
    const std::int64_t i = first + static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2half_rn(saturate_f16(__bfloat162float(x[i])));
}

} // namespace

std::int32_t cublas_min_tokens() {
    static const std::int32_t minimum = [] {
        const char* value = std::getenv("NINFER_SM70_CUBLAS_MIN");
        return value != nullptr ? std::max(1, std::atoi(value)) : kCublasMinTokens;
    }();
    return minimum;
}

std::int32_t slab_rows(std::int32_t rows, std::int32_t k) {
    if (rows <= 0 || k <= 0) { throw std::invalid_argument("sm70 GEMM: invalid matrix"); }
    const std::size_t fit = kWeightSlabBytes / (static_cast<std::size_t>(k) * sizeof(__half));
    const std::int32_t slab =
        static_cast<std::int32_t>(std::max<std::size_t>(128, fit / 128 * 128));
    return std::min(rows, slab);
}

std::size_t activation_bytes(std::int32_t k, std::int32_t tokens) {
    return round256(static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) *
                    sizeof(__half));
}

std::size_t slab_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    const std::size_t slab = static_cast<std::size_t>(slab_rows(rows, k));
    return round256(slab * static_cast<std::size_t>(k) * sizeof(__half)) +
           round256(slab * static_cast<std::size_t>(tokens) * sizeof(float));
}

void convert_activation(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens, __half* out,
                        cudaStream_t stream) {
    const std::int64_t count   = static_cast<std::int64_t>(k) * tokens;
    constexpr int kThreads     = 256;
    const bool aligned         = (reinterpret_cast<std::uintptr_t>(x) % 16) == 0 &&
                                 (reinterpret_cast<std::uintptr_t>(out) % 16) == 0;
    const std::int64_t vectors = aligned ? count / 8 : 0;
    if (vectors > 0) {
        convert_activation_kernel<<<static_cast<unsigned>((vectors + kThreads - 1) / kThreads),
                                    kThreads, 0, stream>>>(reinterpret_cast<const uint4*>(x),
                                                           reinterpret_cast<uint4*>(out), vectors);
        CUDA_CHECK(cudaGetLastError());
    }
    const std::int64_t first = vectors * 8;
    if (first < count) {
        convert_activation_tail_kernel<<<static_cast<unsigned>((count - first + kThreads - 1) /
                                                               kThreads),
                                         kThreads, 0, stream>>>(x, out, first, count);
        CUDA_CHECK(cudaGetLastError());
    }
}

void gemm(const __half* w, std::int32_t rows, std::int32_t k, const __half* x, std::int32_t tokens,
          float* out, std::int64_t ldc, cudaStream_t stream) {
    cublasHandle_t blas = handle_for_current_device();
    check_blas(cublasSetStream(blas, stream), "cublasSetStream");
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // Column-major view: the weight slab is K x rows (lda K), the activations K x tokens (ldb K)
    // and the output rows x tokens (ldc), so C = A^T B.
    check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, tokens, k, &alpha, w, CUDA_R_16F,
                            k, x, CUDA_R_16F, k, &beta, out, CUDA_R_32F, static_cast<int>(ldc),
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx");
}

} // namespace ninfer::ops::detail::sm70
