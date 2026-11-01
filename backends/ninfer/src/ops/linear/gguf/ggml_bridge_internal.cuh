#pragma once

// Shared by the bridge's translation units only: the vendored ggml headers and the device facts the
// matrix kernel's host side needs.

#include "ggml_bridge.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include "../../common/silu.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::gguf::detail {

struct DeviceFacts {
    int cc         = 0; // ggml's compute capability: 100 * major + 10 * minor
    int sm_count   = 0;
    std::size_t shared_per_block_optin = 0;
};

[[nodiscard]] const DeviceFacts& device_facts();

inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gguf ") + what + ": " + cudaGetErrorString(status));
    }
}

[[nodiscard]] inline ggml_type to_ggml(GgmlType type) { return static_cast<ggml_type>(type); }

// Column tiles instantiated for the matrix kernel. A product is launched with the smallest number of
// column tiles these allow, which is where ggml's own selection lands for them as well.
inline constexpr int kMatrixColumnTiles[] = {16, 32, 64, 128};
// MoeRouting lists the matrix kernel's work for each of these widths.
inline constexpr int kMoeTileWidths = 4;

// The matrix kernel's activation carries one block_q8_1_mmq per 128 values per column; ggml over-reads
// up to one widest column tile past the end, so the plane keeps that much slack.
inline constexpr std::size_t kMatrixActivationSlack = 128 * sizeof(block_q8_1_mmq);

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns);

template <ggml_type type>
bool moe_matrix_fits_impl(int rows, int k, bool tail);

template <ggml_type type>
void moe_matrix_product_impl(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                             const MoeRouting& routing, int max_active, int pairs, bool tail,
                             const void* activation, float* out, cudaStream_t stream);

// Where and how the vector kernel (ggml_bridge_vec.cuh) writes its rows.
struct VecStore {
    __nv_bfloat16* bf16;
    float* f32;
    std::int64_t column_stride;
    Epilogue epilogue;
    const float* gate;
    std::int64_t gate_column_stride;
    int activation; // GateActivation of the gate product and of a fused [gate; up] product
};

struct VecArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows; // output rows; a fused [gate; up] parent holds 2 * rows
    int k;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    VecStore out;
};

// The vector kernel's MoE form (ggml_bridge_vec.cuh moe_kernel).
struct MoeVecArgs {
    const std::uint8_t* const* first;  // expert table (the gate table when fused)
    const std::uint8_t* const* second; // fused: the up table
    std::int64_t row_bytes;
    int rows;
    int k;
    const std::int32_t* bounds;
    const std::int32_t* sorted;
    const std::int32_t* active;
    const std::int32_t* active_count;
    int column_group;
    int per_token;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    __nv_bfloat16* out_bf16;
    float* out_f32;
    unsigned long long* weighted;
    const float* weights;
    const float* gate;
    int activation; // GateActivation of `gate` and of the fused form
};

template <ggml_type type>
void moe_launch(const MoeVecArgs& args, int max_active, int chunk, bool fused, cudaStream_t stream);
template <ggml_type type>
void moe_decode_up_launch(const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream);
template <ggml_type type>
void moe_decode_down_launch(const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream);
template <ggml_type type>
void moe_decode_prep_launch(const MoeDecodePrepArgs& args, cudaStream_t stream);

// Whether slice_dequantize serves the type (every type but IQ1_M).
template <ggml_type type>
inline constexpr bool kSliceDequantize = type != GGML_TYPE_IQ1_M;

// Rows of a GGUF matrix to dst_t through the vector kernels' slice decoders: one thread a 32-value
// slice, written as 16-byte stores (ggml_bridge_vec.cuh).
template <ggml_type type, typename dst_t>
void slice_dequantize(const void* weight, std::int64_t row_bytes, int k, const std::int32_t* row_ids,
                      int rows, dst_t* out, std::int64_t out_row_stride, cudaStream_t stream);

// Byte offset of the scales inside a planar vector activation of `columns` x `k`.
[[nodiscard]] constexpr std::size_t vec_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 15) / 16 * 16;
}

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, bool fused, cudaStream_t stream);

// SiLU and the tanh-approximated GELU from correctly rounded operations (ops/common/silu.h): the
// CPU's GGUF expert products compute the same bits.
__device__ __forceinline__ float vec_silu(float x) { return ops::portable_silu(x); }

// GateActivation: 0 SiLU, 1 tanh-approximated GELU.
__device__ __forceinline__ float vec_gate(float x, int activation) {
    return activation == 0 ? vec_silu(x) : ops::gelu_tanh(x);
}

__device__ __forceinline__ void vec_store(const VecStore& out, int row, int column, float value) {
    const std::int64_t at = column * out.column_stride + row;
    if (out.epilogue == Epilogue::GateProduct) {
        value *= vec_gate(out.gate[column * out.gate_column_stride + row], out.activation);
    }
    if (out.f32 != nullptr) {
        out.f32[at] = out.epilogue == Epilogue::Accumulate ? out.f32[at] + value : value;
        return;
    }
    if (out.epilogue == Epilogue::Accumulate) { value += __bfloat162float(out.bf16[at]); }
    out.bf16[at] = __float2bfloat16(value);
}

template <ggml_type type, typename Dst = float, bool accumulate = false>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, Dst* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                         bool tail, float* partial = nullptr);

} // namespace ninfer::ops::gguf::detail
