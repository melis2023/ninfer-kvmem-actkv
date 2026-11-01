// Adapted from JGamboa/ninfer-4090-windows@e905787f (Apache-2.0); see NOTICE.
#include "ops/linear/t5/t5_project.h"

#include "core/device.h"
#include "ops/linear/t5/t5_a8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

// Routes by T (design 9.1): the dp4a GEMV through T = 4, the small-T tensor-core route through
// T = 32 (MTP verification of up to eight lanes at draft 3), the prefill GEMM beyond.
constexpr int kSmallTMaxTokens = 8 * t5_a8::kSmallMaxTiles;

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0; }

std::size_t round_up_256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

struct QuantizedX {
    Tensor q, scale, group_sum, slice_sum;
};

// Bytes of code rows that a decode-width quantization asks L2 for ahead of the GEMV
// (t5_a8::WeightPrefetch). One MiB is what DRAM moves while the quantization runs: per MTP 2
// round on the RTX 4090, 0.75 MiB saved 1.5 %, 1-1.5 MiB 1.9 %, 4 MiB 0.9 % (design 9.1 item 33).
constexpr std::int64_t kGemvPrefetchBytes = std::int64_t(1) << 20;

t5_a8::WeightPrefetch gemv_prefetch(const Weight& w) {
    const std::int64_t row_bytes = std::int64_t(w.k / t5_a8::kUnitColumns) * t5_a8::kUnitBytes;
    const std::int64_t rows =
        std::min<std::int64_t>(w.n, std::max<std::int64_t>(1, kGemvPrefetchBytes / row_bytes));
    // Line i starts at base + 128 i, inside the first `bytes` of the plane for every i counted.
    const auto lines = [](std::int64_t bytes) { return static_cast<unsigned>((bytes + 127) / 128); };
    return {static_cast<const char*>(w.qdata), static_cast<const char*>(w.scales),
            lines(rows * row_bytes), lines(rows * w.scale_nb[1])};
}

// Quantizes represented BF16 input. At GEMV widths it also prefetches leading weight rows.
template <class Input>
QuantizedX quantize(const Input& input, const Weight& w, int tokens, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    const int k = w.k;
    QuantizedX result{workspace.alloc(DType::I8, {k, tokens}),
                      workspace.alloc(DType::FP32, {k / 128, tokens}),
                      workspace.alloc(DType::I32, {k / 128, tokens}),
                      workspace.alloc(DType::I32, {k / 32, tokens})};
    auto* q     = static_cast<std::uint32_t*>(result.q.data);
    auto* scale = static_cast<float*>(result.scale.data);
    auto* gsum  = static_cast<int*>(result.group_sum.data);
    auto* ssum  = static_cast<int*>(result.slice_sum.data);
    // The dp4a GEMV route (T <= 4) reads the GEMV activation layout and gets the prefetch row.
    const bool gemv = tokens <= t5_a8::kGemvMaxTokens;
    const t5_a8::WeightPrefetch lines =
        gemv ? gemv_prefetch(w) : t5_a8::WeightPrefetch{nullptr, nullptr, 0u, 0u};
    const dim3 grid(static_cast<unsigned>(k / t5_a8::kQuantizeBlock),
                    static_cast<unsigned>(tokens + (gemv ? 1 : 0)));
    t5_a8::quantize_kernel<Input><<<grid, t5_a8::kQuantizeThreads, 0, stream>>>(
        input, k, q, scale, gsum, ssum, tokens, gemv, lines);
    CUDA_CHECK(cudaGetLastError());
    return result;
}

template <int Tile, int R>
void launch_gemv_rows(const QuantizedX& q, const Weight& w, const t5_a8::Outputs& outputs,
                      bool accumulate, cudaStream_t stream) {
    constexpr int rows_per_cta = t5_a8::kGemvThreads / 16 * R;
    const unsigned grid        = static_cast<unsigned>((w.n + rows_per_cta - 1) / rows_per_cta);
    t5_a8::gemv_kernel<Tile, R><<<grid, t5_a8::kGemvThreads, 0, stream>>>(
        static_cast<const uint4*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.slice_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.n, w.k, outputs, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

// Rows per GEMV half-warp. The 5120-row weights (640 CTAs at two rows) keep two: at one row,
// down ran 8 % slower at T = 3. The taller ones (14336 rows and up) take one, twice the warps,
// 1-5 % faster at T = 3 (in_proj 24.2 -> 23.7 us, qkvg 22.5 -> 21.4, gate+up 46.0 -> 45.3).
constexpr int kGemvOneRowMinRows = 8192;

template <int Tile>
void launch_gemv(const QuantizedX& q, const Weight& w, const t5_a8::Outputs& outputs,
                 bool accumulate, cudaStream_t stream) {
    if (w.n >= kGemvOneRowMinRows) {
        launch_gemv_rows<Tile, 1>(q, w, outputs, accumulate, stream);
    } else {
        launch_gemv_rows<Tile, 2>(q, w, outputs, accumulate, stream);
    }
}

template <int NTiles>
void launch_small_t(const QuantizedX& q, const Weight& w, int tokens,
                    const t5_a8::Outputs& outputs, bool accumulate, cudaStream_t stream) {
    constexpr int kTokens = 8 * NTiles;
    const dim3 grid(static_cast<unsigned>((w.n + t5_a8::kSmallRows - 1) / t5_a8::kSmallRows),
                    static_cast<unsigned>((tokens + kTokens - 1) / kTokens));
    t5_a8::small_t_kernel<NTiles><<<grid, t5_a8::kSmallThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.n, w.k, tokens, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

void launch_gemm(const QuantizedX& q, const Weight& w, int tokens, const t5_a8::Outputs& outputs,
                 bool accumulate, cudaStream_t stream) {
    constexpr std::size_t kSmem = t5_a8::gemm_shared_bytes();
    static_assert(kSmem <= 48 * 1024);
    const int token_tiles = (tokens + t5_a8::kGemmTokens - 1) / t5_a8::kGemmTokens;
    const unsigned grid   = static_cast<unsigned>(w.n / t5_a8::kGemmRows * token_tiles);
    t5_a8::gemm_kernel<<<grid, t5_a8::kGemmThreads, kSmem, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, token_tiles, outputs,
        accumulate);
    CUDA_CHECK(cudaGetLastError());
}

// The token tiles of each 128-row block: `wide` 128-token tiles, then `narrow` 64-token tiles.
struct TallSplit {
    int wide, narrow;
};

// Uniform complete 128-token tiles and a 64-token tail. The donor's 4090 wave-cost
// heuristic is not assumed to describe SM120; this keeps arithmetic and bounds identical.
TallSplit choose_tall_split(int, int tokens) {
    return {tokens / 128, (tokens % 128 + 63) / 64};
}

void launch_tall_gemm(const QuantizedX& q, const Weight& w, int tokens,
                      const t5_a8::Outputs& outputs, bool accumulate, cudaStream_t stream) {
    constexpr std::size_t kSmem = t5_a8::gemm_tall_shared_bytes();
    raise_dynamic_shared_memory(t5_a8::gemm_tall_kernel, static_cast<int>(kSmem));
    const int row_blocks  = w.n / t5_a8::kTallRows;
    const TallSplit split = choose_tall_split(row_blocks, tokens);
    const unsigned grid   = static_cast<unsigned>(row_blocks * (split.wide + split.narrow));
    t5_a8::gemm_tall_kernel<<<grid, t5_a8::kTallThreads, kSmem, stream>>>(
        static_cast<const std::uint8_t*>(q.q.data), static_cast<const float*>(q.scale.data),
        static_cast<const int*>(q.group_sum.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const __half*>(w.scales), w.scale_nb[1] / 2, w.k, tokens, split.wide,
        split.narrow, outputs, accumulate);
    CUDA_CHECK(cudaGetLastError());
}

void require_input(const Tensor& x, const Weight& w, const char* what) {
    if (x.dtype != DType::BF16 || x.ne[0] != w.k || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        x.data == nullptr || !x.is_contiguous() || !aligned16(x.data)) {
        throw std::invalid_argument(std::string("t5_project: ") + what +
                                    " must be contiguous aligned BF16 [K,T]");
    }
}

// Validates the weight, policy and outputs, then quantizes the input and multiplies: dp4a GEMV
// through T = 4, the small-T MMA route through T = 32 (and at any T when N % 64 != 0, in 32-token
// tiles), the 64 x 64 GEMM through T = 64 (and beyond when N % 128 != 0 or an output is not made
// of aligned 8-row chunks), the pipelined 128-row GEMM beyond (every Bonsai weight).
template <class Input>
void project(const Input& input, int tokens, const Weight& w, std::span<Tensor* const> outputs,
             bool accumulate, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (!allows_a8(policy) || workspace == nullptr) {
        throw std::invalid_argument("t5_project: a T5 weight requires AllowA8 and a workspace");
    }
    if (outputs.empty() || outputs.size() > t5_a8::kMaxOutputs) {
        throw std::invalid_argument("t5_project: expected one to four outputs");
    }
    t5_a8::Outputs packed{};
    int end = 0;
    // The pipelined GEMM stores 8-row chunks as 16-byte words.
    bool chunked_outputs = true;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == nullptr) { throw std::invalid_argument("t5_project: null output"); }
        const Tensor& out = *outputs[i];
        if (out.dtype != DType::BF16 || out.ne[0] <= 0 || out.ne[0] > w.n - end ||
            out.ne[1] != tokens || out.ne[2] != 1 || out.ne[3] != 1 ||
            out.nb[0] != sizeof(__nv_bfloat16) || out.nb[1] < out.ne[0] * sizeof(__nv_bfloat16) || out.data == nullptr) {
            throw std::invalid_argument("t5_project: outputs must be contiguous BF16 [rows,T]");
        }
        end += out.ne[0];
        chunked_outputs = chunked_outputs && out.ne[0] % 8 == 0 && out.nb[1] % 16 == 0 && aligned16(out.data);
        packed.data[i] = static_cast<__nv_bfloat16*>(out.data);
        packed.end[i]  = end;
        packed.stride[i] = static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16));
    }
    for (std::size_t i = outputs.size(); i < t5_a8::kMaxOutputs; ++i) {
        packed.data[i] = packed.data[outputs.size() - 1];
        packed.end[i]  = end;
        packed.stride[i] = packed.stride[outputs.size() - 1];
    }
    if (end != w.n) {
        throw std::invalid_argument("t5_project: output rows must cover the weight rows");
    }

    auto scope         = workspace->scope();
    const QuantizedX q = quantize(input, w, tokens, *workspace, stream);
    switch (tokens) {
    case 1: launch_gemv<1>(q, w, packed, accumulate, stream); return;
    case 2: launch_gemv<2>(q, w, packed, accumulate, stream); return;
    case 3: launch_gemv<3>(q, w, packed, accumulate, stream); return;
    case 4: launch_gemv<4>(q, w, packed, accumulate, stream); return;
    default: break;
    }
    if (tokens <= kSmallTMaxTokens || w.n % t5_a8::kGemmRows != 0) {
        switch ((std::min(tokens, kSmallTMaxTokens) + 7) / 8) {
        case 1: launch_small_t<1>(q, w, tokens, packed, accumulate, stream); break;
        case 2: launch_small_t<2>(q, w, tokens, packed, accumulate, stream); break;
        case 3: launch_small_t<3>(q, w, tokens, packed, accumulate, stream); break;
        default: launch_small_t<4>(q, w, tokens, packed, accumulate, stream); break;
        }
    } else if (tokens > t5_a8::kGemmTokens && w.n % t5_a8::kTallRows == 0 && chunked_outputs) {
        launch_tall_gemm(q, w, tokens, packed, accumulate, stream);
    } else {
        launch_gemm(q, w, tokens, packed, accumulate, stream);
    }
}

} // namespace

void validate_t5_weight(const Weight& w, const char* op) {
    if (w.qtype != QType::T5_G128_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group_size != 128 || w.group != 128 ||
        w.n <= 0 || w.k <= 0 || w.k % 1024 || w.ndim != 2 ||
        w.shape[0] != w.n || w.shape[1] != w.k ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k ||
        w.qdata == nullptr || w.qhigh != nullptr || w.high_plane_bytes != 0 || w.scales == nullptr ||
        (reinterpret_cast<std::uintptr_t>(w.qdata) & 15) ||
        (reinterpret_cast<std::uintptr_t>(w.scales) & 15) || w.scale_nb[1] != w.k / 64) {
        throw std::invalid_argument(std::string(op) +
                                    ": weight must be T5_G128_FP16 ternary rows with K%1024=0");
    }
}

std::size_t t5_workspace_capacity_bytes(LinearPolicy policy, std::int32_t input_rows,
                                        std::int32_t max_tokens) {
    if (input_rows <= 0 || input_rows % 1024 || max_tokens <= 0) {
        throw std::invalid_argument("t5 workspace: requires K % 1024 == 0 and positive T");
    }
    if (!allows_a8(policy)) { throw std::invalid_argument("t5 workspace: requires AllowA8"); }
    const std::size_t k = static_cast<std::size_t>(input_rows);
    const std::size_t t = static_cast<std::size_t>(max_tokens);
    return round_up_256(k * t) + 2 * round_up_256(k / 128 * t * 4) + round_up_256(k / 32 * t * 4);
}

void t5_project(const Tensor& x, const Weight& w, std::span<Tensor* const> outputs, bool accumulate,
                LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_t5_weight(w, "t5_project");
    require_input(x, w, "x");
    project(t5_a8::PlainInput{static_cast<const __nv_bfloat16*>(x.data)}, x.ne[1], w, outputs,
            accumulate, policy, workspace, stream);
}


} // namespace ninfer::ops::detail
