// llama.cpp's flash_attn_ext_f16 (fattn-mma-f16.cuh) launched without ggml's backend context: the
// host side of launch_fattn and ggml_cuda_flash_attn_ext_mma_f16_case, for the two Gemma 4 shapes.

#include "ggml_fattn.h"
#include "ggml_bridge_internal.cuh"

#include "ggml-cuda/fattn-mma-f16.cuh"
// This archive's include path holds only the bridge and ggml (sources.cmake); the limits header
// includes nothing.
#include "../../common/device_limits.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::gguf {
namespace {

using detail::check;
using detail::device_facts;

// Shared memory of one flash_attn_ext_f16 tile under configuration `config`, as
// ggml_cuda_flash_attn_ext_mma_f16_case sizes it: `turing_mma` selects ldmatrix swizzling and 16
// columns per warp (Volta takes 32), `cp_async` the configuration's multi-stage K/V ring.
constexpr std::size_t tile_shared_bytes(const fattn_mma_config& config, int D, int ncols1,
                                        int ncols2, bool turing_mma, bool cp_async) {
    const int ncols         = ncols1 * ncols2;
    const int nstages       = cp_async && ncols2 >= 2 ? config.nstages_target : 0;
    const int cols_per_warp = std::min(ncols, turing_mma ? 16 : 32);
    const int nwarps        = config.nthreads / WARP_SIZE;

    const bool swizzled = turing_mma && ggml_cuda_fattn_mma_bank_aligned(config.nbatch_K2) &&
                          ggml_cuda_fattn_mma_bank_aligned(config.nbatch_V2);

    const std::size_t stride_k = ggml_cuda_fattn_mma_get_stride_tile(config.nbatch_K2, swizzled);
    const std::size_t stride_v = ggml_cuda_fattn_mma_get_stride_tile(config.nbatch_V2, swizzled);
    const std::size_t kv_1     = std::size_t(config.nbatch_fa) * std::max(stride_k, stride_v) * 4;
    const std::size_t kv_2     = std::size_t(config.nbatch_fa) * (stride_k + stride_v) * 4;
    const std::size_t q_bytes  = std::size_t(ncols) * (D / 2 + 4) * 4;
    const std::size_t mask     = std::size_t(ncols1) * (config.nbatch_fa / 2 + 4) * 4;
    const std::size_t combine =
        std::size_t(nwarps) * cols_per_warp * (config.nbatch_combine + 4) * 4;
    const std::size_t kv = nstages <= 1 ? kv_1 : kv_2;
    return std::max(combine, config.Q_in_reg ? std::max(q_bytes, kv + mask) : q_bytes + kv + mask);
}

// The configuration table this build's device code compiles (ggml_cuda_fattn_mma_get_config's
// device form) and the instructions it has.
constexpr fattn_mma_config build_config(int D, int ncols) {
#if defined(NINFER_SM70)
    return ggml_cuda_fattn_mma_get_config_volta(D, D, ncols);
#elif defined(NINFER_SM75)
    return ggml_cuda_fattn_mma_get_config_turing(D, D, ncols);
#else
    return ggml_cuda_fattn_mma_get_config_ampere(D, D, ncols);
#endif
}
#if defined(NINFER_SM70)
inline constexpr bool kBuildTuringMma = false;
#else
inline constexpr bool kBuildTuringMma = true;
#endif
#if defined(NINFER_SM70) || defined(NINFER_SM75)
inline constexpr bool kBuildCpAsync = false;
#else
inline constexpr bool kBuildCpAsync = true;
#endif
// The tile widths (query rows times query heads per key/value head) flash_attn_ext_f16 has device
// code for on this build's architecture: at least 32 on Volta, at most 32 on Turing.
#if defined(NINFER_SM70)
inline constexpr int kBuildMinTileColumns = 32;
#else
inline constexpr int kBuildMinTileColumns = 1;
#endif
#if defined(NINFER_SM75)
inline constexpr int kBuildMaxTileColumns = 32;
#else
inline constexpr int kBuildMaxTileColumns = 1 << 20;
#endif
// The most query rows per tile, up to `widest`, whose tile this build's device code has and whose
// shared memory fits a block: ggml's prompt choices (32 rows at width 256, 8 at width 512) where
// they do, and on Turing 16 rows at width 256 and 4 at width 512 (its kernel has no tile wider than
// 32 columns). The vendored combine barrier (third_party/ggml-quants/README.ninfer.md) keeps the
// tiles that hold Q in shared memory free of the race the upstream kernel had at width 512.
template <int D, int ncols2>
constexpr int fitting_ncols1(int widest) {
    for (int ncols1 = widest; ncols1 >= 1; ncols1 /= 2) {
        const int ncols = ncols1 * ncols2;
        if (ncols < kBuildMinTileColumns) { break; }
        const fattn_mma_config config = build_config(D, ncols);
        if (ncols <= kBuildMaxTileColumns && config.nbatch_fa > 0 &&
            tile_shared_bytes(config, D, ncols1, ncols2, kBuildTuringMma, kBuildCpAsync) <=
                std::size_t(kMaxSharedBytesPerBlock)) {
            return ncols1;
        }
    }
    return 0;
}

// The configurations instantiated: width 256 with two query heads per key/value head, width 512
// with eight, each with the most query rows per tile that fit (fitting_ncols1).
template <int D> struct Shape;
template <> struct Shape<256> {
    static constexpr int ncols2 = 2;
    static constexpr int ncols1 = fitting_ncols1<256, ncols2>(32);
};
template <> struct Shape<512> {
    static constexpr int ncols2 = 8;
    static constexpr int ncols1 = fitting_ncols1<512, ncols2>(8);
};
static_assert(Shape<256>::ncols1 > 0 && Shape<512>::ncols1 > 0,
              "no flash-attention tile of this build fits Gemma 4's head widths");
#if defined(NINFER_SM75)
static_assert(Shape<256>::ncols1 == 16 && Shape<512>::ncols1 == 4);
#elif !defined(NINFER_SM70)
static_assert(Shape<256>::ncols1 == 32 && Shape<512>::ncols1 == 8);
#endif

struct Launch {
    fattn_kernel_t kernel = nullptr;
    int threads           = 0;
    std::size_t shared    = 0;
    int nbatch_fa         = 0;
    int blocks_per_sm     = 0;
};

template <int D>
const Launch& launch_for() {
    constexpr int ncols1 = Shape<D>::ncols1;
    constexpr int ncols2 = Shape<D>::ncols2;
    constexpr int ncols  = ncols1 * ncols2;
    static Launch launches[16];
    static std::once_flag once[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(once[device & 15], [&] {
        const int cc                  = device_facts().cc;
        const fattn_mma_config config = ggml_cuda_fattn_mma_get_config(D, D, ncols, cc);
        const std::size_t shared      = tile_shared_bytes(
            config, D, ncols1, ncols2, turing_mma_available(cc), cp_async_available(cc));
        const int nthreads  = config.nthreads;
        const int nbatch_fa = config.nbatch_fa;
        Launch& out = launches[device & 15];
        out.kernel  = flash_attn_ext_f16<D, D, ncols1, ncols2, false, false, false>;
        out.threads = nthreads;
        out.shared  = shared;
        out.nbatch_fa = nbatch_fa;
        check(cudaFuncSetAttribute(reinterpret_cast<const void*>(out.kernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shared)),
              "flash attention shared memory");
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, out.kernel, nthreads, shared),
              "flash attention occupancy");
        out.blocks_per_sm = std::max(blocks, 1);
    });
    return launches[device & 15];
}

struct Grid {
    int blocks     = 0;
    int ntiles_dst = 0;
    int ntiles_x   = 0;
    int ntiles_kv  = 0;
    bool fixup     = false;
};

template <int D>
Grid grid_for(const Launch& launch, int tokens, int kv_heads, int z_gqa, int slots) {
    constexpr int ncols1 = Shape<D>::ncols1;
    const auto& facts = device_facts();
    Grid g;
    g.ntiles_x   = (tokens + ncols1 - 1) / ncols1;
    g.ntiles_dst         = g.ntiles_x * z_gqa * kv_heads;
    g.ntiles_kv  = (slots + launch.nbatch_fa - 1) / launch.nbatch_fa;
    const int max_blocks = launch.blocks_per_sm * facts.sm_count;
    const int waves      = (g.ntiles_dst + max_blocks - 1) / max_blocks;
    const int efficiency = 100 * g.ntiles_dst / (max_blocks * waves);
    const bool stream_k  = facts.cc >= GGML_CUDA_CC_ADA_LOVELACE || efficiency < 75;
    g.blocks = g.ntiles_dst;
    if (stream_k) {
        const int raw     = std::min(max_blocks, g.ntiles_kv * g.ntiles_dst);
        const int rounded = (raw / g.ntiles_dst) * g.ntiles_dst;
        const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
        g.blocks          = loss <= 5 ? rounded : raw;
    }
    g.fixup = g.ntiles_dst % g.blocks != 0;
    return g;
}

template <int D>
std::size_t workspace_for(int query_heads, int kv_heads, int max_tokens) {
    constexpr int ncols = Shape<D>::ncols1 * Shape<D>::ncols2;
    (void)query_heads;
    (void)kv_heads;
    (void)max_tokens;
    // Stream-k never launches more than a wave of blocks; each keeps its partial rows and their
    // (max, sum) pairs.
    const Launch& launch = launch_for<D>();
    const std::size_t blocks = std::size_t(launch.blocks_per_sm) * device_facts().sm_count;
    return blocks * ncols * (2 + D / 2) * sizeof(float2) + 256;
}

template <int D>
void run(const FlashAttentionArgs& a, void* workspace, std::size_t workspace_bytes,
         cudaStream_t stream) {
    constexpr int ncols1 = Shape<D>::ncols1;
    constexpr int ncols2 = Shape<D>::ncols2;
    constexpr int ncols  = ncols1 * ncols2;
    if (a.query_heads % (a.kv_heads * ncols2) != 0) {
        throw std::invalid_argument("gguf flash attention: unsupported grouping for this width");
    }
    // Query heads per key/value head come in groups of ncols2, one tile column each.
    const int z_gqa      = a.query_heads / (a.kv_heads * ncols2);
    const Launch& launch = launch_for<D>();
    const Grid g         = grid_for<D>(launch, a.tokens, a.kv_heads, z_gqa, a.slots);
    float2* meta         = nullptr;
    if (g.fixup) {
        const std::size_t need = std::size_t(g.blocks) * ncols * (2 + D / 2) * sizeof(float2);
        if (workspace == nullptr || workspace_bytes < need) {
            throw std::invalid_argument("gguf flash attention: workspace too small");
        }
        meta = static_cast<float2*>(workspace);
    }
    const uint3 ne01 = init_fastdiv_values(a.tokens);
    const std::int32_t nb01 = static_cast<std::int32_t>(a.q_token_stride * sizeof(float));
    const std::int32_t nb02 = static_cast<std::int32_t>(D * sizeof(float));
    const std::int32_t nb03 = static_cast<std::int32_t>(nb01) * a.tokens;
    const std::int32_t nb31 = static_cast<std::int32_t>(a.mask_row_halves * sizeof(half));
    const dim3 block(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    launch.kernel<<<dim3(g.blocks, 1, 1), block, launch.shared, stream>>>(
        reinterpret_cast<const char*>(a.q), static_cast<const char*>(a.k),
        static_cast<const char*>(a.v), static_cast<const char*>(a.mask), nullptr, a.kv_max, a.out,
        meta, a.scale, 0.0f, 1.0f, 1.0f, 1u, 0.0f, D, ne01, a.query_heads, 1, nb01, nb02, nb03, D,
        a.slots, a.kv_heads, 1, static_cast<std::int32_t>(a.row_bytes),
        static_cast<std::int32_t>(a.head_bytes), a.head_bytes * a.kv_heads,
        static_cast<std::int32_t>(a.row_bytes), static_cast<std::int32_t>(a.head_bytes),
        a.head_bytes * a.kv_heads, a.tokens, 1, 1, nb31, nb31 * a.tokens,
        std::int64_t(nb31) * a.tokens);
    check(cudaGetLastError(), "flash attention launch");
    if (!g.fixup) { return; }
    const int gqa = a.query_heads / a.kv_heads;
    if (g.blocks % g.ntiles_dst == 0 && g.blocks > g.ntiles_dst) {
        const int per_tile = g.blocks / g.ntiles_dst;
        const uint3 fd0    = init_fastdiv_values(g.ntiles_x * z_gqa * a.kv_heads);
        const uint3 fd1    = init_fastdiv_values(g.ntiles_x * z_gqa);
        const uint3 fd2    = init_fastdiv_values(g.ntiles_x);
        flash_attn_stream_k_fixup_uniform<D, ncols1, ncols2>
            <<<dim3(g.ntiles_dst, ncols1, ncols2), dim3(D, 1, 1), 0, stream>>>(
                a.out, meta, a.tokens, a.query_heads, a.kv_heads, g.blocks, gqa, per_tile, fd0, fd1,
                fd2);
    } else {
        const int total = g.ntiles_kv * g.ntiles_dst;
        const uint3 fd0 = init_fastdiv_values(g.ntiles_kv * g.ntiles_x * z_gqa * a.kv_heads);
        const uint3 fd1 = init_fastdiv_values(g.ntiles_kv * g.ntiles_x * z_gqa);
        const uint3 fd2 = init_fastdiv_values(g.ntiles_kv * g.ntiles_x);
        const uint3 fd3 = init_fastdiv_values(g.ntiles_kv);
        flash_attn_stream_k_fixup_general<D, ncols1, ncols2>
            <<<dim3(g.blocks, ncols1, ncols2), dim3(D, 1, 1), 0, stream>>>(
                a.out, meta, a.tokens, a.query_heads, gqa, total, fd0, fd1, fd2, fd3);
    }
    check(cudaGetLastError(), "flash attention fixup launch");
}

} // namespace

std::size_t flash_attention_workspace_bytes(int head_dim, int query_heads, int kv_heads,
                                            int max_tokens) {
    switch (head_dim) {
    case 256: return workspace_for<256>(query_heads, kv_heads, max_tokens);
    case 512: return workspace_for<512>(query_heads, kv_heads, max_tokens);
    default: break;
    }
    throw std::invalid_argument("gguf flash attention: head width must be 256 or 512");
}

int flash_attention_tile_rows(int head_dim) {
    switch (head_dim) {
    case 256:
        return Shape<256>::ncols1;
    case 512:
        return Shape<512>::ncols1;
    default:
        break;
    }
    throw std::invalid_argument("gguf flash attention: head width must be 256 or 512");
}

void flash_attention(const FlashAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
                     cudaStream_t stream) {
    if (args.q == nullptr || args.k == nullptr || args.v == nullptr || args.mask == nullptr ||
        args.out == nullptr || args.tokens <= 0 || args.kv_heads <= 0 || args.slots <= 0 ||
        args.slots % FATTN_KQ_STRIDE != 0 || args.mask_row_halves < args.slots ||
        args.row_bytes % 16 != 0 || args.head_bytes % 16 != 0 ||
        (args.q_token_stride * sizeof(float)) % 16 != 0 || args.mask_row_halves % 8 != 0) {
        throw std::invalid_argument("gguf flash attention: invalid arguments");
    }
    switch (args.head_dim) {
    case 256: return run<256>(args, workspace, workspace_bytes, stream);
    case 512: return run<512>(args, workspace, workspace_bytes, stream);
    default: break;
    }
    throw std::invalid_argument("gguf flash attention: head width must be 256 or 512");
}

} // namespace ninfer::ops::gguf
