// ninfer::ops - decode of staged Qwen3.8-Flash-Next n-gram rows (contract in
// include/ninfer/ops/ngram_rows.h).
#include "ninfer/ops/ngram_rows.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kRowWidth = 160;

__global__ void __launch_bounds__(kRowWidth)
    ngram_fp8_rows_kernel(const std::uint8_t* __restrict__ rows, std::int64_t row_count,
                          __nv_bfloat16* __restrict__ embedding) {
    const std::int64_t row = blockIdx.x;
    const int j            = threadIdx.x;
    if (row >= row_count) return;
    const std::uint8_t* source = rows + row * (kRowWidth + 2);
    __half_raw scale_raw;
    scale_raw.x = static_cast<unsigned short>(source[kRowWidth] | (source[kRowWidth + 1] << 8));
    const float scale = __half2float(__half(scale_raw));
    __nv_fp8_e4m3 code;
    code.__x = source[j];
    embedding[row * kRowWidth + j] = __float2bfloat16_rn(static_cast<float>(code) * scale);
}

// ggml's IQ4_NL codebook.
__constant__ std::int8_t kIq4NlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                             1,    13,   25,  38,  53,  69,  89,  113};
constexpr std::int32_t kIq4NlBlockBytes   = 18;

__global__ void __launch_bounds__(kRowWidth)
    ngram_iq4nl_rows_kernel(const std::uint8_t* __restrict__ rows, std::int64_t row_count,
                            __nv_bfloat16* __restrict__ embedding) {
    const std::int64_t row = blockIdx.x;
    const int j            = threadIdx.x;
    if (row >= row_count) return;
    const std::uint8_t* block =
        rows + row * (kRowWidth / 32 * kIq4NlBlockBytes) + (j / 32) * kIq4NlBlockBytes;
    __half_raw scale_raw;
    scale_raw.x                    = static_cast<unsigned short>(block[0] | (block[1] << 8));
    const float scale              = __half2float(__half(scale_raw));
    const int i                    = j % 32;
    const int code                 = i < 16 ? (block[2 + i] & 0xF) : (block[2 + i - 16] >> 4);
    embedding[row * kRowWidth + j] = __float2bfloat16_rn(scale * float(kIq4NlValues[code]));
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ngram_embed_rows: ") + message); }
}

} // namespace

std::uint32_t ngram_row_bytes(NgramRowFormat format) {
    switch (format) {
    case NgramRowFormat::Bf16:
        return kRowWidth * 2;
    case NgramRowFormat::Fp8E4M3RowScale:
        return kRowWidth + 2;
    case NgramRowFormat::Iq4Nl:
        return kRowWidth / 32 * kIq4NlBlockBytes;
    }
    throw std::invalid_argument("ngram_row_bytes: unknown format");
}

void ngram_embed_rows(const Tensor& rows, NgramRowFormat format, std::int32_t heads,
                      Tensor& embedding, cudaStream_t stream) {
    require(heads > 0, "heads must be positive");
    const std::int32_t tokens = embedding.ne[1];
    require(tokens > 0, "tokens must be positive");
    require(rows.dtype == DType::U8 && rows.is_contiguous() && rows.data != nullptr &&
                rows.ne[0] == static_cast<std::int32_t>(ngram_row_bytes(format)) &&
                rows.ne[1] == heads * tokens && rows.ne[2] == 1 && rows.ne[3] == 1,
            "rows must be contiguous U8 [row_bytes, heads * tokens]");
    require(embedding.dtype == DType::BF16 && embedding.is_contiguous() &&
                embedding.data != nullptr && embedding.ne[0] == heads * kRowWidth &&
                embedding.ne[2] == 1 && embedding.ne[3] == 1,
            "embedding must be contiguous BF16 [heads * 160, tokens]");
    const std::int64_t row_count = static_cast<std::int64_t>(heads) * tokens;
    if (format == NgramRowFormat::Bf16) {
        // The rows already are the embedding's bytes, in its order.
        CUDA_CHECK(cudaMemcpyAsync(embedding.data, rows.data,
                                   static_cast<std::size_t>(row_count) * kRowWidth * 2,
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }
    if (format == NgramRowFormat::Iq4Nl) {
        ngram_iq4nl_rows_kernel<<<static_cast<unsigned>(row_count), kRowWidth, 0, stream>>>(
            static_cast<const std::uint8_t*>(rows.data), row_count,
            static_cast<__nv_bfloat16*>(embedding.data));
    } else {
        ngram_fp8_rows_kernel<<<static_cast<unsigned>(row_count), kRowWidth, 0, stream>>>(
            static_cast<const std::uint8_t*>(rows.data), row_count,
            static_cast<__nv_bfloat16*>(embedding.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
