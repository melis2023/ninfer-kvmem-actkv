#pragma once

// Host side of ggml's integer tensor-core matrix kernel (mmq.cuh launch_mul_mat_q), without ggml's
// backend context: the caller owns the activation, the FP32 output and the stream-k fixup plane.

#include "ggml_bridge_internal.cuh"

#include <climits>
#include <mutex>

namespace ninfer::ops::gguf::detail {

struct MatrixLaunch {
    int J                   = 0;
    int I                   = 0;
    int threads             = 0;
    std::size_t shared      = 0;
    int row_tiles           = 0;
    int column_tiles        = 0;
    bool stream_k           = false;
    int blocks              = 0;
    bool fixup              = false;
};

template <ggml_type type>
MatrixLaunch select_matrix_launch(int rows, int columns) {
    const DeviceFacts& device = device_facts();
    MatrixLaunch best;
    int best_tiles = INT_MAX;
    for (const int J : kMatrixColumnTiles) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        const int tiles = (columns + J - 1) / J;
        if (tiles < best_tiles) {
            best_tiles        = tiles;
            best.J            = J;
            best.I            = config.I;
            best.threads      = config.nthreads;
            best.shared       = shared;
            best.stream_k     = config.stream_k;
        }
        if (tiles == 1) { break; }
    }
    if (best.J == 0) { throw std::invalid_argument("gguf matrix product: no tile fits this device"); }
    best.row_tiles    = (rows + best.I - 1) / best.I;
    best.column_tiles = (columns + best.J - 1) / best.J;
    const int tiles   = best.row_tiles * best.column_tiles;
    if (best.stream_k) {
        const int waves      = (tiles + device.sm_count - 1) / device.sm_count;
        const int efficiency = 100 * tiles / (device.sm_count * waves);
        best.blocks          = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns) {
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    return launch.fixup ? std::size_t(launch.blocks) * launch.J * launch.I * sizeof(float) : 0;
}

template <ggml_type type, int J>
void launch_matrix(const MatrixLaunch& launch, const char* weight, int stride_row_blocks, int rows,
                   int k, const int* activation, int columns, float* out,
                   std::int64_t out_column_stride, float* fixup, cudaStream_t stream) {
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(mul_mat_q<type, J, false>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "matrix shared memory");
    });
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    const uint3 blocks_fd   = init_fastdiv_values(k / qk);
    const uint3 one_fd      = init_fastdiv_values(1);
    const uint3 ntx_fd      = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride        = static_cast<int>(out_column_stride);
    if (!launch.stream_k) {
        mul_mat_q<type, J, false>
            <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                weight, activation, nullptr, nullptr, out, nullptr, nullptr, blocks_fd, rows,
                columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0, one_fd,
                one_fd, 0, 0, 0, ntx_fd);
        check(cudaGetLastError(), "matrix product launch");
        return;
    }
    if (launch.fixup && fixup == nullptr) {
        throw std::invalid_argument("gguf matrix product: this launch needs a fixup plane");
    }
    mul_mat_q<type, J, false><<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
        weight, activation, nullptr, nullptr, out, launch.fixup ? fixup : nullptr, nullptr,
        blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
        one_fd, one_fd, 0, 0, 0, ntx_fd);
    check(cudaGetLastError(), "matrix product launch");
    if (!launch.fixup) { return; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    mul_mat_q_stream_k_fixup<type, J, false><<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, fixup, blocks_fd, rows, columns, stride, one_fd, 0, one_fd, 0,
        ntx_fd);
    check(cudaGetLastError(), "matrix fixup launch");
}

template <ggml_type type>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, float* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    if (k % 256 != 0 || rows % 128 != 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf matrix product: unsupported geometry");
    }
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    const auto* x             = static_cast<const char*>(weight);
    const auto* y             = static_cast<const int*>(activation);
    auto* f                   = static_cast<float*>(fixup);
    const int stride_row      = static_cast<int>(row_bytes / block_bytes);
    switch (launch.J) {
    case 16:
        launch_matrix<type, 16>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    case 32:
        launch_matrix<type, 32>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    case 64:
        launch_matrix<type, 64>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    case 128:
        launch_matrix<type, 128>(launch, x, stride_row, rows, k, y, columns, out, out_column_stride, f, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("gguf matrix product: unexpected column tile");
}

// ggml's tile loop over the experts a router selected: one CTA per (row tile, column tile, active
// expert), the expert's rows reached through a table of base pointers (a bank, a cache slot or
// mapped host memory) instead of a fixed stride. Column c of the activation is the routing's c-th
// pair, and pair p's result row lands at out[p * rows].
template <ggml_type type, int J>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, false),
                  ggml_cuda_mmq_get_occupancy(type, J, false)) static __global__
    void moe_mul_mat_q(const void* const* __restrict__ experts, const int* __restrict__ y,
                       const std::int32_t* __restrict__ sorted,
                       const std::int32_t* __restrict__ bounds,
                       const std::int32_t* __restrict__ active,
                       const std::int32_t* __restrict__ active_count, float* __restrict__ dst,
                       const uint3 blocks_per_row, int rows, int stride_row_x, int ncols_y) {
    if (ggml_cuda_mmq_get_config(type, J, false).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, false) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, false);
    if (int(blockIdx.z) >= *active_count) { return; }
    const int expert   = active[blockIdx.z];
    const int col_low  = bounds[expert];
    const int col_diff = bounds[expert + 1] - col_low;
    const int jt       = blockIdx.y;
    const int it       = blockIdx.x;
    if (jt * J >= col_diff) { return; }
    // The tile's destination pairs head the dynamic shared memory, where the tile loop reads them.
    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > J && j >= J) { break; }
        ids_dst_shared[j] = jt * J + j < col_diff ? sorted[col_low + jt * J + j] : 0;
    }
    __syncthreads();
    const int offset_y     = (col_low + jt * J) * int(sizeof(block_q8_1_mmq) / sizeof(int));
    const int tile_x_max_i = rows - it * I - 1;
    const int tile_y_max_j = col_diff - jt * J - 1;
    mul_mat_q_process_tile<type, J, false, false, GGML_PREC_Q8>(
        static_cast<const char*>(experts[expert]), it * I * stride_row_x, y + offset_y,
        ids_dst_shared, dst + it * I, nullptr, nullptr, stride_row_x, ncols_y, rows, tile_x_max_i,
        tile_y_max_j, 0, blocks_per_row.z);
}

// The column tile for products whose experts take `columns` pairs on average: the narrowest tile
// that holds them, or the widest available. Zero when no tile of this type fits the geometry (the
// rows a whole number of row tiles; k a whole number of the kernel's K steps, or of blocks over a
// 256-value step when zeros follow the bank) or the device.
template <ggml_type type>
MatrixLaunch select_moe_launch(int rows, int k, int columns, bool tail) {
    const DeviceFacts& device = device_facts();
    constexpr int qk          = ggml_cuda_type_traits<type>::qk;
    MatrixLaunch best;
    for (const int J : kMatrixColumnTiles) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        const bool steps = k % config.K_vram == 0 ||
                           (tail && config.K_vram == MMQ_ITER_K && k % qk == 0 && k % 128 == 0);
        if (config.type == GGML_TYPE_COUNT || rows % config.I != 0 || !steps) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        best.J       = J;
        best.I       = config.I;
        best.threads = config.nthreads;
        best.shared  = shared;
        if (J >= columns) { break; }
    }
    return best;
}

template <ggml_type type>
bool moe_matrix_fits_impl(int rows, int k, bool tail) {
    return select_moe_launch<type>(rows, k, 1, tail).J != 0;
}

template <ggml_type type, int J>
void launch_moe_matrix(const MatrixLaunch& launch, const void* const* experts, int stride_row,
                       int rows, int k, const MoeRouting& routing, int max_active, int pairs,
                       int max_columns, const int* activation, float* out, cudaStream_t stream) {
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(moe_mul_mat_q<type, J>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "moe matrix shared memory");
    });
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const dim3 grid(rows / launch.I, (max_columns + J - 1) / J, max_active);
    const dim3 block(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    moe_mul_mat_q<type, J><<<grid, block, launch.shared, stream>>>(
        experts, activation, routing.sorted, routing.bounds, routing.active, routing.active_count,
        out, init_fastdiv_values(k / qk), rows, stride_row, pairs);
    check(cudaGetLastError(), "moe matrix product launch");
}

template <ggml_type type>
void moe_matrix_product_impl(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                             const MoeRouting& routing, int max_active, int pairs, int max_columns,
                             bool tail, const void* activation, float* out, cudaStream_t stream) {
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    constexpr int qk          = ggml_cuda_type_traits<type>::qk;
    if (max_active <= 0 || pairs <= 0 || max_columns <= 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf moe matrix product: invalid geometry");
    }
    const MatrixLaunch launch =
        select_moe_launch<type>(rows, k, (pairs + max_active - 1) / max_active, tail);
    if (launch.J == 0) {
        throw std::invalid_argument("gguf moe matrix product: no tile fits this geometry");
    }
    const auto* y        = static_cast<const int*>(activation);
    const int stride_row = static_cast<int>(row_bytes / block_bytes);
    switch (launch.J) {
    case 16:
        return launch_moe_matrix<type, 16>(launch, experts, stride_row, rows, k, routing,
                                           max_active, pairs, max_columns, y, out, stream);
    case 32:
        return launch_moe_matrix<type, 32>(launch, experts, stride_row, rows, k, routing,
                                           max_active, pairs, max_columns, y, out, stream);
    case 64:
        return launch_moe_matrix<type, 64>(launch, experts, stride_row, rows, k, routing,
                                           max_active, pairs, max_columns, y, out, stream);
    case 128:
        return launch_moe_matrix<type, 128>(launch, experts, stride_row, rows, k, routing,
                                            max_active, pairs, max_columns, y, out, stream);
    default:
        break;
    }
    throw std::invalid_argument("gguf moe matrix product: unexpected column tile");
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_MATRIX_INSTANCE(TYPE)                                                          \
    template std::size_t ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<TYPE>(int, int);       \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE>(                            \
        const void*, std::int64_t, int, int, const void*, int, float*, std::int64_t, void*,        \
        cudaStream_t);                                                                             \
    template bool ninfer::ops::gguf::detail::moe_matrix_fits_impl<TYPE>(int, int, bool);           \
    template void ninfer::ops::gguf::detail::moe_matrix_product_impl<TYPE>(                        \
        const void* const*, std::int64_t, int, int, const ninfer::ops::gguf::MoeRouting&, int,     \
        int, int, bool, const void*, float*, cudaStream_t)
