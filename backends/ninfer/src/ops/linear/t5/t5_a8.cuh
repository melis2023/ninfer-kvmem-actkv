// Adapted from JGamboa/ninfer-4090-windows@e905787f (Apache-2.0); see NOTICE.
#pragma once

// A8 projections of T5_G128_FP16 weights, adapted from the JGamboa T5 route.
//
// Weight layout. A row is K / 64 units of 13 bytes. Unit byte 4 g + j (g = 0..2, j = 0..3)
// holds the trits t_m = code of column 20 g + 4 m + j (m = 0..4), byte 12 the columns 60 + m
// (m = 0..3, t_4 = 0), as q = ceil(256 v / 243) with v = sum_m t_m 3^(4 - m). Decoding one
// 32-bit group of bytes 4 g .. 4 g + 3 runs the even and odd bytes in 16-bit lanes: per m,
// r = 3 r yields the trit r >> 8 of every byte, so each m gives one word of four codes {0,1,2}
// for the natural-order columns 20 g + 4 m .. 20 g + 4 m + 3.
//
// Activation layout: int8 in natural column order with one FP32 scale per 128 columns, and the
// sums of q per 32-column slice and per 128-column group, so `(code - 1) . q = code . q - sum(q)`.
// This port accepts represented BF16 rows from the current model's fused rotation producers.
// It does not apply the rotation a second time or change the model's rounding boundary.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::t5_a8 {

__device__ __forceinline__ void mma_s8_from(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                            unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1, int c0, int c1, int c2, int c3) {
#if defined(NINFER_SM70_DEVICE) || defined(NINFER_SM75_DEVICE)
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
    mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3));
#endif
}

constexpr int kMaxOutputs = 4;

struct Outputs {
    __nv_bfloat16* data[kMaxOutputs];
    int stride[kMaxOutputs];
    int end[kMaxOutputs]; // exclusive parent row bound of each output
};

// The output of parent row `row`, resolved with unrolled selects (no local copy of Outputs).
struct OutputRow {
    __nv_bfloat16* data; // element (row, token 0)
    int rows;            // token stride
};

__device__ __forceinline__ OutputRow output_row(const Outputs& outputs, int row) {
    __nv_bfloat16* data = outputs.data[0];
    int begin = 0, stride = outputs.stride[0];
#pragma unroll
    for (int s = 1; s < kMaxOutputs; ++s) {
        if (row >= outputs.end[s - 1]) {
            data  = outputs.data[s];
            begin = outputs.end[s - 1];
            stride = outputs.stride[s];
        }
    }
    return {data + (row - begin), stride};
}

__device__ __forceinline__ void store_row(OutputRow out, int token, float value, bool accumulate) {
    __nv_bfloat16* p = out.data + std::int64_t(token) * out.rows;
    *p = __float2bfloat16_rn(accumulate ? value + __bfloat162float(*p) : value);
}

// Position of activation word `word` (columns 4 word .. 4 word + 3 of a token row) in the GEMV
// layout: each run of 16 units (1024 columns) stores its 16-byte chunks chunk-major, [chunk
// 0..3][unit 0..15], so the 16 lanes of a GEMV half-warp, which read the same chunk of 16
// consecutive units, load 256 contiguous bytes instead of 16 chunks 64 bytes apart.
__device__ __forceinline__ int gemv_word(int word) {
    const int unit = word >> 4, chunk = (word >> 2) & 3;
    return (unit >> 4) * 256 + chunk * 64 + (unit & 15) * 4 + (word & 3);
}

// One warp quantizes one (token, 128-column group); lane l holds columns 4 l .. 4 l + 3 in
// `value` and writes activation word l (natural order, or the GEMV layout above). q = rint(x *
// 127 / amax), scale = amax / 127 (zero for an all-zero group), then the sums of q per 32-column
// slice (eight lanes) and per group.
__device__ __forceinline__ void quantize_group(const float (&value)[4], int lane, int token,
                                               int group, int k, bool gemv_layout,
                                               std::uint32_t* __restrict__ qx,
                                               float* __restrict__ group_scale,
                                               int* __restrict__ group_sum,
                                               int* __restrict__ slice_sum) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(value[i]));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
    std::uint32_t word  = 0;
    int sum             = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = max(-127, min(127, __float2int_rn(value[i] * inverse)));
        word |= (static_cast<std::uint32_t>(q) & 0xffu) << (8 * i);
        sum += q;
    }
    const std::int64_t groups = k / 128;
    const int index = group * 32 + lane;
    qx[std::int64_t(token) * (k / 4) + (gemv_layout ? gemv_word(index) : index)] = word;
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    if ((lane & 7) == 0) slice_sum[std::int64_t(token) * (k / 32) + group * 4 + (lane >> 3)] = sum;
    sum += __shfl_xor_sync(0xffffffffu, sum, 8);
    sum += __shfl_xor_sync(0xffffffffu, sum, 16);
    if (lane == 0) {
        group_scale[std::int64_t(token) * groups + group] = amax / 127.0f;
        group_sum[std::int64_t(token) * groups + group]   = sum;
    }
}

constexpr int kUnitColumns = 64;
constexpr int kUnitBytes   = 13;

// One CTA quantizes one represented BF16 1024-column block of one token.
constexpr int kQuantizeBlock   = 1024;
constexpr int kQuantizeThreads = 256;

__device__ __forceinline__ void unpack_bf16x4(uint2 word, float (&value)[4]) {
    const float2 low  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word.x));
    const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&word.y));
    value[0]          = low.x;
    value[1]          = low.y;
    value[2]          = high.x;
    value[3]          = high.y;
}

// The input x [K,T] itself.
struct PlainInput {
    static constexpr bool kTokenState = false;
    const __nv_bfloat16* x;

    struct Token {};
    __device__ __forceinline__ Token prepare(int, int) const { return {}; }
    __device__ __forceinline__ void operator()(int k, int token, int column, Token,
                                               float (&value)[4]) const {
        unpack_bf16x4(load_ldg<uint2>(x + std::int64_t(token) * k + column), value);
    }
};

// The leading code lines and scale lines of the weight that a decode-width quantization asks L2
// for (prefetch.global.L2, one request per 128-byte line). The GEMV that follows reads them
// first (CTAs start in row order), so DRAM streams its first rows while the quantization runs
// instead of idling until the GEMV's first loads. Lines are counted from 128-byte-aligned bases.
struct WeightPrefetch {
    const char* codes;
    const char* scales;
    unsigned code_lines;
    unsigned scale_lines;
};

// Grid: (K / 1024, T) CTAs. For the
// GEMV (`gemv`, T <= 4) the activation words take the GEMV layout and one more row of gridDim.x
// CTAs (blockIdx.y == tokens) only issues `prefetch` and exits.
template <class Input>
__global__ void __launch_bounds__(kQuantizeThreads)
    quantize_kernel(Input input, int k,
                    std::uint32_t* __restrict__ qx, float* __restrict__ group_scale,
                    int* __restrict__ group_sum, int* __restrict__ slice_sum, int tokens, bool gemv,
                    WeightPrefetch prefetch) {
    if (static_cast<int>(blockIdx.y) >= tokens) {
        const unsigned threads = gridDim.x * blockDim.x;
        const unsigned first   = blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned line = first; line < prefetch.code_lines; line += threads) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(prefetch.codes + 128ull * line));
        }
        for (unsigned line = first; line < prefetch.scale_lines; line += threads) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(prefetch.scales + 128ull * line));
        }
        return;
    }
    const int tid                   = static_cast<int>(threadIdx.x);
    const int token                 = static_cast<int>(blockIdx.y);
    const typename Input::Token state = input.prepare(k, token);
    const int first = static_cast<int>(blockIdx.x);
    const int last = first + 1;
    for (int block = first; block < last; ++block) {
        const int column = block * kQuantizeBlock + 4 * tid;
        float value[4];
        input(k, token, column, state, value);
        quantize_group(value, tid & 31, token, column / 128, k, gemv, qx, group_scale, group_sum,
                       slice_sum);
    }
}

// The 13 bytes of one unit as four little-endian words (bytes 0-3, 4-7, 8-11, 12), read with
// four aligned 32-bit streaming loads and funnel shifts.
__device__ __forceinline__ void load_unit(const std::uint8_t* __restrict__ row_codes, int unit,
                                          std::uint32_t (&a)[4]) {
    const std::int64_t start = std::int64_t(unit) * kUnitBytes;
    const auto* base = reinterpret_cast<const std::uint32_t*>(row_codes + (start & ~std::int64_t(3)));
    const unsigned shift   = unsigned(start & 3) * 8u;
    const std::uint32_t w0 = __ldcs(base), w1 = __ldcs(base + 1), w2 = __ldcs(base + 2),
                        w3 = __ldcs(base + 3);
    a[0] = __funnelshift_r(w0, w1, shift);
    a[1] = __funnelshift_r(w1, w2, shift);
    a[2] = __funnelshift_r(w2, w3, shift);
    a[3] = w3 >> shift;
}

// The five code words of one 32-bit byte group. After r *= 3 of a byte r in a 16-bit lane, the
// lane's high byte is exactly the trit (3 * 255 < 768), so one byte permute gathers the word.
__device__ __forceinline__ void group_words(std::uint32_t bytes, std::uint32_t* words) {
    std::uint32_t even = bytes & 0x00ff00ffu;
    std::uint32_t odd  = __byte_perm(bytes, 0u, 0x4341); // bytes 1 and 3
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        even *= 3u;
        odd *= 3u;
        words[m] = __byte_perm(even, odd, 0x7351);
        even &= 0x00ff00ffu;
        odd &= 0x00ff00ffu;
    }
}

// The code word of unit byte 12 (columns 60..63).
__device__ __forceinline__ std::uint32_t last_word(std::uint32_t bytes) {
    std::uint32_t r = bytes & 0xffu;
    std::uint32_t x[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        r *= 3u;
        x[m] = r;
        r &= 0xffu;
    }
    return __byte_perm(__byte_perm(x[0], x[1], 0x0051), __byte_perm(x[2], x[3], 0x0051), 0x5410);
}

// words[w] = the codes of unit columns 4 w .. 4 w + 3, one per byte.
__device__ __forceinline__ void decode_unit(const std::uint32_t (&a)[4], std::uint32_t (&words)[16]) {
#pragma unroll
    for (int g = 0; g < 3; ++g) group_words(a[g], words + 5 * g);
    words[15] = last_word(a[3]);
}

// ---------------------------------------------------------------------------------------------
// dp4a GEMV for decode and single-lane MTP verification (T <= 4). A CTA of two warps owns 4 R
// rows: each half-warp R rows, its 16 lanes striding the units (K / 64 is a multiple of 16, so
// no lane idles). Per unit and row: four streaming loads, the arithmetic decode, and one dp4a
// per code word and token; the activation words of a unit are shared by the R rows. A row's
// arithmetic does not depend on R. The dp4a work grows with T while the decode does not, so
// wider T takes the tensor-core small-T route below.

constexpr int kGemvThreads   = 64;
constexpr int kGemvMaxTokens = 4;

template <int Tile, int R>
__global__ void __launch_bounds__(kGemvThreads)
    gemv_kernel(const uint4* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ slice_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int n, int k,
                Outputs outputs, bool accumulate) {
    static_assert(Tile >= 1 && Tile <= kGemvMaxTokens);
    const int warp  = static_cast<int>(threadIdx.x) / 32;
    const int lane  = static_cast<int>(threadIdx.x) % 32;
    const int hl    = lane & 15;
    const int row0 =
        ((static_cast<int>(blockIdx.x) * (kGemvThreads / 32) + warp) * 2 + (lane >> 4)) * R;
    const int units              = k / kUnitColumns;
    const std::int64_t row_bytes = std::int64_t(units) * kUnitBytes;
    const int slices = k / 32, groups = k / 128;
    // Rows past n alias the last row; their results are never stored.
    const std::uint8_t* row_codes[R];
    const __half* row_scales[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const std::int64_t row = min(row0 + r, n - 1);
        row_codes[r]           = codes + row * row_bytes;
        row_scales[r]          = scales + row * scale_row_halves;
    }
    float acc[R][Tile];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) acc[r][t] = 0.0f;
    }
    for (int u = hl; u < units; u += 16) {
        std::uint32_t words[R][16];
        float scale[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            std::uint32_t a[4];
            load_unit(row_codes[r], u, a);
            decode_unit(a, words[r]);
            scale[r] = __half2float(__ldcs(row_scales[r] + (u >> 1)));
        }
        uint4 xs[Tile][4];
        int offset[Tile];
        float step[Tile];
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            // GEMV layout (gemv_word): chunk w of unit u at 16-unit run u / 16, chunk-major.
            const uint4* row = qx + std::int64_t(t) * (k / 16) + (u >> 4) * 64 + (u & 15);
#pragma unroll
            for (int w = 0; w < 4; ++w) xs[t][w] = __ldg(row + 16 * w);
            const int2 sums = __ldg(reinterpret_cast<const int2*>(slice_sum + std::int64_t(t) * slices) + u);
            offset[t]       = sums.x + sums.y;
            step[t] = __ldg(group_scale + std::int64_t(t) * groups + (u >> 1));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int dot[Tile];
#pragma unroll
            for (int t = 0; t < Tile; ++t) dot[t] = 0;
#pragma unroll
            for (int w = 0; w < 16; ++w) {
#pragma unroll
                for (int t = 0; t < Tile; ++t) {
                    const std::uint32_t lanes[4] = {xs[t][w >> 2].x, xs[t][w >> 2].y,
                                                    xs[t][w >> 2].z, xs[t][w >> 2].w};
                    dot[t] = __dp4a(static_cast<int>(words[r][w]), static_cast<int>(lanes[w & 3]), dot[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                acc[r][t] = fmaf(scale[r] * step[t], static_cast<float>(dot[t] - offset[t]), acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[r][t] = v;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= n) continue;
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (hl == t) store_row(output_row(outputs, row), t, acc[r][t], accumulate);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// int8 tensor-core GEMMs for prefill, m16n8k32 s8 MMAs over 128-column steps (one scale group).
// Code words and activations are stored in shared memory in 16-byte chunks XORed with the row
// (token) index, so both the A and B fragments load with conflict-free `ldmatrix`. Each step's
// int32 sums start at kFloatMagic, so the FP32 update acc += (row scale * token scale) *
// (code . q - sum(q)) needs one exact FADD per sum instead of an I2F (quarter rate). The code
// bytes of a unit are loaded a step ahead as aligned words and shifted only at the decode, so
// the load latency hides behind the MMAs. The token tiles of a row block are adjacent in launch
// order and share its code bytes in L2. The arithmetic per output is the same in both kernels.
//
// gemm_kernel: 64 x 64 tiles for T = 33..64 (and wider T when N % 128 != 0). A CTA of four
// warps (wm, wn) of 32 x 32 owns 64 rows and 64 tokens, three CTAs per SM. The stage's 64 rows x
// 2 units are decoded into a single code-word buffer, one unit per thread, after the step's MMAs
// between two barriers. The activation and the token scales and sums are double-buffered with
// cp.async.
//
// gemm_tall_kernel: 128-row tiles of 128 or 64 tokens for wider T, below.

constexpr int kStageK = 128;

constexpr int kGemmRows    = 64;
constexpr int kGemmTokens  = 64;
constexpr int kGemmThreads = 128;

__device__ __forceinline__ int gemm_word(int row, int word) { return row * 32 + (word ^ ((row & 7) << 2)); }

template <int Tokens>
struct GemmStage {
    std::uint8_t x[Tokens * kStageK]; // 16-byte chunks swizzled per token row
    float scale[Tokens];
    int sum[Tokens];
};

// 1.5 * 2^23 as FP32 bits: kFloatMagic + x reinterpreted as a float is 12582912 + x exactly for
// |x| < 2^22 (a step's |code . q| <= 2 * 127 * 128), and 12582912 + sum(q) is exact as well.
constexpr int kFloatMagic = 0x4B400000;

// One unit's 13 bytes as aligned streaming loads of the 16-byte window that holds them;
// shift_unit_words aligns them to the unit.
__device__ __forceinline__ void load_unit_words(const std::uint8_t* __restrict__ row_codes, int unit,
                                                std::uint32_t (&r)[4]) {
    const auto* base = reinterpret_cast<const std::uint32_t*>(row_codes + ((unit * kUnitBytes) & ~3));
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = __ldcs(base + i);
}

// The unit's bytes as little-endian words (bytes 0-3, 4-7, 8-11, 12).
__device__ __forceinline__ void shift_unit_words(const std::uint32_t (&r)[4], int unit,
                                                 std::uint32_t (&a)[4]) {
    const unsigned shift = unsigned((unit * kUnitBytes) & 3) * 8u;
    a[0] = __funnelshift_r(r[0], r[1], shift);
    a[1] = __funnelshift_r(r[1], r[2], shift);
    a[2] = __funnelshift_r(r[2], r[3], shift);
    a[3] = r[3] >> shift;
}

// Decodes a unit and stores its 16 code words in the row's swizzled chunks.
__device__ __forceinline__ void decode_store_unit(const std::uint32_t (&a)[4],
                                                  std::uint32_t* __restrict__ code_words, int row,
                                                  int unit) {
    std::uint32_t* base = code_words + row * 32;
    const auto chunk    = [&](int c) { return base + ((unit * 4 + c) ^ (row & 7)) * 4; };
    std::uint32_t words[15];
#pragma unroll
    for (int g = 0; g < 3; ++g) group_words(a[g], words + 5 * g);
    *reinterpret_cast<uint4*>(chunk(0)) = make_uint4(words[0], words[1], words[2], words[3]);
    *reinterpret_cast<uint4*>(chunk(1)) = make_uint4(words[4], words[5], words[6], words[7]);
    *reinterpret_cast<uint4*>(chunk(2)) = make_uint4(words[8], words[9], words[10], words[11]);
    *reinterpret_cast<uint4*>(chunk(3)) = make_uint4(words[12], words[13], words[14], last_word(a[3]));
}

// Shared memory of one CTA: the double-buffered activation stages, then the code words.
constexpr std::size_t gemm_shared_bytes() {
    return 2 * sizeof(GemmStage<kGemmTokens>) + std::size_t(kGemmRows) * 32 * sizeof(std::uint32_t);
}

// Stores a warp's (16 MTiles) x 32 tile. Fragment C: [0], [1] are row gid, tokens 2 lid and
// 2 lid + 1; [2], [3] are row gid + 8.
template <int MTiles>
__device__ __forceinline__ void store_tile(const Outputs& outputs, const float (&acc)[MTiles][4][4],
                                           int row, int token_base, int token0, int live, int lane,
                                           bool accumulate) {
    const int gid = lane >> 2;
    const int lid = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const OutputRow out = output_row(outputs, row + mt * 16 + gid + half * 8);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int token = token_base + nt * 8 + 2 * lid + j;
                    if (token < live) store_row(out, token0 + token, acc[mt][nt][2 * half + j], accumulate);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kGemmThreads, 3)
    gemm_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                int tokens, int token_tiles, Outputs outputs, bool accumulate) {
    constexpr int kRows      = kGemmRows;
    constexpr int kTokens    = kGemmTokens;
    constexpr int kThreads   = kGemmThreads;
    extern __shared__ __align__(128) std::uint8_t gemm_smem[];
    auto* stages     = reinterpret_cast<GemmStage<kTokens>*>(gemm_smem);
    auto* code_words = reinterpret_cast<std::uint32_t*>(gemm_smem + 2 * sizeof(GemmStage<kTokens>));
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int wm     = warp % 2;
    const int wn     = warp / 2;
    const int row0   = static_cast<int>(blockIdx.x) / token_tiles * kRows;
    const int token0 = static_cast<int>(blockIdx.x) % token_tiles * kTokens;
    const int live   = min(kTokens, tokens - token0);
    const int steps  = k / kStageK;
    const std::int64_t row_bytes = std::int64_t(k / kUnitColumns) * kUnitBytes;
    // Decode item (row, unit) = (tid / 2, tid % 2).
    const int my_unit = tid & 1;
    const int my_row  = tid >> 1;
    const std::uint8_t* my_codes = codes + std::int64_t(row0 + my_row) * row_bytes;

    const auto stage_x = [&](int step, GemmStage<kTokens>& s) {
#pragma unroll
        for (int i = 0; i < kTokens * (kStageK / 16) / kThreads; ++i) {
            const int item   = tid + i * kThreads;
            const int token  = item >> 3;
            const int chunk  = item & 7;
            const int source = token < live ? token0 + token : token0;
            cp_async_zfill<16>(&s.x[token * kStageK + ((chunk ^ (token & 7)) << 4)],
                               qx + std::int64_t(source) * k + std::int64_t(step) * kStageK +
                                   chunk * 16,
                               token < live ? 16 : 0);
        }
        // kThreads >= 2 kTokens: the first kTokens threads stage the scales, the next the sums.
        static_assert(kThreads >= 2 * kTokens);
        const int token  = tid & (kTokens - 1);
        const int source = token < live ? token0 + token : token0;
        const std::int64_t index = std::int64_t(source) * (k / kStageK) + step;
        if (tid < kTokens) {
            cp_async_zfill<4>(&s.scale[token], group_scale + index, token < live ? 4 : 0);
        } else if (tid < 2 * kTokens) {
            cp_async_zfill<4>(&s.sum[token], group_sum + index, token < live ? 4 : 0);
        }
    };

    // ldmatrix row addresses. A (x4, per mt and ks): matrices rows 0-7 / 8-15 of chunk 2 ks,
    // then of chunk 2 ks + 1, i.e. fragment registers a0..a3. B (x4, per nt pair and ks): tokens
    // of nt at chunks 2 ks, 2 ks + 1, then of nt + 1.
    const int a_row    = wm * 32 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk  = lane >> 4;
    const int b_token  = wn * 32 + (lane >> 4) * 8 + (lane & 7);
    const int b_chunk  = (lane >> 3) & 1;

    float acc[2][4][4] = {};
    std::uint32_t raw[4] = {};
    load_unit_words(my_codes, my_unit, raw);
    std::uint32_t words[4] = {};
    shift_unit_words(raw, my_unit, words);
    decode_store_unit(words, code_words, my_row, my_unit);
    stage_x(0, stages[0]);
    cp_commit();
    for (int step = 0; step < steps; ++step) {
        if (step + 1 < steps) {
            stage_x(step + 1, stages[(step + 1) & 1]);
            load_unit_words(my_codes, 2 * (step + 1) + my_unit, raw);
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        const GemmStage<kTokens>& s = stages[step & 1];
        int group[2][4][4];
#pragma unroll
        for (int ks = 0; ks < kStageK / 32; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int token = b_token + np * 16;
                const int chunk = 2 * ks + b_chunk;
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            smem_addr(&s.x[token * kStageK + ((chunk ^ (token & 7)) << 4)]));
            }
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                            smem_addr(&code_words[gemm_word(a_row + mt * 16, 4 * (2 * ks + a_chunk))]));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int* g = group[mt][nt];
                    if (ks == 0) {
                        mma_s8_from(g[0], g[1], g[2], g[3], a0, a1, a2, a3, b[nt][0], b[nt][1],
                                    kFloatMagic, kFloatMagic, kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(g[0], g[1], g[2], g[3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
        float token_scale[4][2], token_bias[4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int token    = wn * 32 + nt * 8 + 2 * lid + j;
                token_scale[nt][j] = s.scale[token];
                token_bias[nt][j]  = __int_as_float(kFloatMagic + s.sum[token]);
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r        = row0 + wm * 32 + mt * 16 + gid;
            const float top    = __half2float(__ldg(scales + std::int64_t(r) * scale_row_halves + step));
            const float bottom =
                __half2float(__ldg(scales + std::int64_t(r + 8) * scale_row_halves + step));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j      = e & 1;
                    const float unit = (e < 2 ? top : bottom) * token_scale[nt][j];
                    acc[mt][nt][e]   = fmaf(unit, __int_as_float(group[mt][nt][e]) - token_bias[nt][j],
                                            acc[mt][nt][e]);
                }
            }
        }
        // Every warp has read this stage's code words before they are overwritten.
        __syncthreads();
        if (step + 1 < steps) {
            shift_unit_words(raw, 2 * (step + 1) + my_unit, words);
            decode_store_unit(words, code_words, my_row, my_unit);
        }
    }
    store_tile(outputs, acc, row0 + wm * 32, wn * 32, token0, live, lane, accumulate);
}

// ---------------------------------------------------------------------------------------------
// gemm_tall_kernel: tiles of 128 rows x 128 or 64 tokens for T > 64 (design 9.1). Eight warps of
// 64 x 32 (128 tokens) or 32 x 32 (64 tokens), one CTA per SM, one barrier per step: the
// activation stages and the code-word stages are both double-buffered, so the next step's
// activations (cp.async) and code words (decoded by all threads, one unit each) are written while
// this step's are read. A 64-token tile decodes as much per step for half the MMAs; the launch
// mixes both widths so that the last resident wave is filled (t5_project.cu).

template <int Rows>
struct GemmCodeStage {
    std::uint32_t words[Rows * 32]; // 16-byte chunks swizzled per row
    float scale[Rows];              // the stage's FP16 row scales
};

// Stages the activation tiles (tokens x 128 bytes, 16-byte chunks swizzled per token) and the
// token scales and sums of a step; `thread` of `Threads` participating threads. The source
// pointers are fixed per thread (tokens past `live` read token0 with zero fill), so a step only
// adds its column offset.
template <int Tokens, int Threads>
struct XStager {
    static_assert(Threads >= 2 * Tokens && Tokens * (kStageK / 16) % Threads == 0);
    static constexpr int kChunks = Tokens * (kStageK / 16) / Threads;
    const std::uint8_t* src[kChunks];
    unsigned dst[kChunks];
    int bytes[kChunks];
    const std::uint8_t* small_src; // token scale (thread < Tokens) or sum, of step 0
    unsigned small_dst;
    int small_bytes;

    __device__ __forceinline__ XStager(int thread, const std::uint8_t* __restrict__ qx,
                                       const float* __restrict__ group_scale,
                                       const int* __restrict__ group_sum, int k, int token0,
                                       int live) {
#pragma unroll
        for (int i = 0; i < kChunks; ++i) {
            const int item   = thread + i * Threads;
            const int token  = item >> 3;
            const int chunk  = item & 7;
            const int source = token < live ? token0 + token : token0;
            src[i]   = qx + std::int64_t(source) * k + chunk * 16;
            dst[i]   = unsigned(token * kStageK + ((chunk ^ (token & 7)) << 4));
            bytes[i] = token < live ? 16 : 0;
        }
        const int token          = thread & (Tokens - 1);
        const int source         = token < live ? token0 + token : token0;
        const std::int64_t index = std::int64_t(source) * (k / kStageK);
        small_src   = thread < Tokens ? reinterpret_cast<const std::uint8_t*>(group_scale + index)
                                      : reinterpret_cast<const std::uint8_t*>(group_sum + index);
        small_dst   = unsigned(thread < Tokens ? offsetof(GemmStage<Tokens>, scale)
                                               : offsetof(GemmStage<Tokens>, sum)) +
                    unsigned(token) * 4u;
        small_bytes = thread < 2 * Tokens && token < live ? 4 : 0;
    }

    __device__ __forceinline__ void issue(int step, GemmStage<Tokens>& stage, int thread) const {
        const unsigned base = smem_addr(&stage);
#pragma unroll
        for (int i = 0; i < kChunks; ++i) {
#if defined(NINFER_PRE_SM80_DEVICE)
            cp_async_zfill<16, Cache::cg>(reinterpret_cast<std::uint8_t*>(&stage) + dst[i],
                                          src[i] + std::int64_t(step) * kStageK, bytes[i]);
#else
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :
                         : "r"(base + dst[i]), "l"(src[i] + std::int64_t(step) * kStageK),
                           "r"(bytes[i]));
#endif
        }
        if (thread < 2 * Tokens) {
#if defined(NINFER_PRE_SM80_DEVICE)
            cp_async_zfill<4>(reinterpret_cast<std::uint8_t*>(&stage) + small_dst,
                              small_src + std::int64_t(step) * 4, small_bytes);
#else
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :
                         : "r"(base + small_dst), "l"(small_src + std::int64_t(step) * 4),
                           "r"(small_bytes));
#endif
        }
    }
};

// Byte offsets of a lane's ldmatrix rows per k-slice: A (x4 per 16-row tile) in the code words,
// matrices rows 0-7 / 8-15 of chunk 2 ks, then of chunk 2 ks + 1; B (x4 per token pair) in the
// activation tile, tokens of nt at chunks 2 ks, 2 ks + 1, then of nt + 1. Rows and tokens are
// 128-byte lines with 16-byte chunks XORed with the line index, and a row (token) index differs
// from the lane's line only in multiples of 8, so chunk 2 ks + c sits at ((c ^ l7) << 4) ^ (ks << 5).
struct FragOffsets {
    unsigned a[4];
    unsigned b[4];
};

__device__ __forceinline__ FragOffsets frag_offsets(int a_row, int b_token, int lane) {
    const unsigned l7 = unsigned(lane & 7);
    const unsigned ax = (unsigned(lane >> 4) ^ l7) << 4;
    const unsigned bx = (unsigned((lane >> 3) & 1) ^ l7) << 4;
    FragOffsets f;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        f.a[ks] = unsigned(a_row) * 128u + (ax ^ unsigned(ks << 5));
        f.b[ks] = unsigned(b_token) * 128u + (bx ^ unsigned(ks << 5));
    }
    return f;
}

constexpr int kTallRows    = 128;
constexpr int kTallThreads = 256;

// A tile of 128 rows x Tokens (128 or 64) tokens: eight warps of kWarpRows rows x 32 tokens.
template <int Tokens>
struct TallTile {
    static_assert(Tokens == 128 || Tokens == 64);
    static constexpr int kWarpsN   = Tokens / 32;
    static constexpr int kWarpsM   = kTallThreads / 32 / kWarpsN;
    static constexpr int kWarpRows = kTallRows / kWarpsM;
    static constexpr int kMTiles   = kWarpRows / 16; // 16-row MMA tiles per warp
};

// Sized for the 128-token tile; a 64-token tile uses the same stage offsets.
constexpr std::size_t gemm_tall_shared_bytes() {
    return 2 * sizeof(GemmStage<128>) + 2 * sizeof(GemmCodeStage<kTallRows>);
}

// The FP32 tile staged as [token][row], rows padded so that the fragment stores are conflict-free
// and every 8-row chunk is 16-byte aligned.
constexpr int kTallStageLd = kTallRows + 4;
static_assert(std::size_t(128) * kTallStageLd * sizeof(float) <= gemm_tall_shared_bytes());

// Stores a 128-row tile through shared memory: the warps stage their FP32 accumulators, then each
// thread writes 8-row chunks of one token as 16-byte stores (a token's 128 rows by 16 adjacent
// threads), adding the residual for accumulate. Per output this is store_row's arithmetic:
// bf16_rn(acc) or bf16_rn(acc + residual). The host routes here only outputs whose row bounds
// are multiples of 8 and whose data is 16-byte aligned.
template <int Tokens, int MTiles>
__device__ __forceinline__ void store_tall_tile(const Outputs& outputs,
                                                const float (&acc)[MTiles][4][4], float* staged,
                                                int row0, int token0, int live, int warp_row,
                                                int warp_token, int tid, bool accumulate) {
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row   = warp_row + mt * 16 + gid + (e >> 1) * 8;
                const int token = warp_token + nt * 8 + 2 * lid + (e & 1);
                staged[token * kTallStageLd + row] = acc[mt][nt][e];
            }
        }
    }
    __syncthreads();
    constexpr int kChunks = kTallRows / 8;
#pragma unroll
    for (int i = 0; i < Tokens * kChunks / kTallThreads; ++i) {
        const int item  = tid + i * kTallThreads;
        const int token = item / kChunks;
        const int chunk = item % kChunks;
        if (token >= live) continue;
        const float* source = &staged[token * kTallStageLd + chunk * 8];
        const float4 low    = *reinterpret_cast<const float4*>(source);
        const float4 high   = *reinterpret_cast<const float4*>(source + 4);
        float value[8]      = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        const OutputRow out = output_row(outputs, row0 + chunk * 8);
        auto* target = reinterpret_cast<uint4*>(out.data + std::int64_t(token0 + token) * out.rows);
        if (accumulate) {
            const uint4 old         = *target;
            const unsigned words[4] = {old.x, old.y, old.z, old.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[w]));
                value[2 * w]     = value[2 * w] + pair.x;
                value[2 * w + 1] = value[2 * w + 1] + pair.y;
            }
        }
        unsigned packed[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const __nv_bfloat162 pair = __floats2bfloat162_rn(value[2 * w], value[2 * w + 1]);
            packed[w]                 = *reinterpret_cast<const unsigned*>(&pair);
        }
        *target = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

// Each SM sub-partition holds one warp of each half of the CTA. Warps 0-3 ("ping") multiply, then
// decode the next step's code words and apply the step's FP32 update; warps 4-7 ("pong") first
// apply the previous step's update and decode, then multiply, carrying their int32 sums and the
// step's scales across the barrier. A sub-partition's tensor pipe then runs one warp's MMAs while
// the other warp does its integer and FP32 work, instead of both doing the same phase at once.
template <int Tokens>
__device__ __forceinline__ void gemm_tall_tile(const std::uint8_t* __restrict__ qx,
                                               const float* __restrict__ group_scale,
                                               const int* __restrict__ group_sum,
                                               const std::uint8_t* __restrict__ codes,
                                               const __half* __restrict__ scales,
                                               std::int64_t scale_row_halves, int k, int tokens,
                                               int row0, int token0, const Outputs& outputs,
                                               bool accumulate, std::uint8_t* gemm_smem) {
    using Tile             = TallTile<Tokens>;
    constexpr int kRows    = kTallRows;
    constexpr int kTokens  = Tokens;
    constexpr int kThreads = kTallThreads;
    constexpr int MTiles   = Tile::kMTiles;
    using XStage           = GemmStage<kTokens>;
    using CodeStage        = GemmCodeStage<kRows>;
    auto* xs = reinterpret_cast<XStage*>(gemm_smem);
    auto* cs = reinterpret_cast<CodeStage*>(gemm_smem + 2 * sizeof(GemmStage<128>));
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int wm      = warp % Tile::kWarpsM;
    const int wn      = warp / Tile::kWarpsM;
    const bool pong   = warp >= 4;
    const int live    = min(kTokens, tokens - token0);
    const int steps   = k / kStageK;
    // Decode item: row tid / 2, unit tid % 2 of each step.
    const int my_row  = tid >> 1;
    const int my_unit = tid & 1;
    const std::uint8_t* my_codes =
        codes + std::int64_t(row0 + my_row) * (std::int64_t(k / kUnitColumns) * kUnitBytes);
    const __half* my_scales = scales + std::int64_t(row0 + my_row) * scale_row_halves;
    std::uint32_t raw[4]    = {};
    __half raw_scale        = {};
    const auto load = [&](int step) {
        load_unit_words(my_codes, 2 * step + my_unit, raw);
        if (my_unit == 0) raw_scale = __ldg(my_scales + step);
    };
    const auto decode = [&](int step, CodeStage& c) {
        std::uint32_t a[4];
        shift_unit_words(raw, 2 * step + my_unit, a);
        decode_store_unit(a, c.words, my_row, my_unit);
        if (my_unit == 0) c.scale[my_row] = __half2float(raw_scale);
    };

    const FragOffsets f = frag_offsets(wm * Tile::kWarpRows + (lane & 7) + ((lane >> 3) & 1) * 8,
                                       wn * 32 + (lane >> 4) * 8 + (lane & 7), lane);
    const auto multiply = [&](const XStage& s, const CodeStage& c, int (&g)[MTiles][4][4]) {
        const unsigned x_base = smem_addr(s.x);
        const unsigned c_base = smem_addr(c.words);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned b[4][2];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                            x_base + f.b[ks] + np * 16 * kStageK);
            }
#pragma unroll
            for (int mt = 0; mt < MTiles; ++mt) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, c_base + f.a[ks] + mt * 16 * 128);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    int* d = g[mt][nt];
                    if (ks == 0) {
                        mma_s8_from(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1],
                                    kFloatMagic, kFloatMagic, kFloatMagic, kFloatMagic);
                    } else {
                        mma_s8(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[nt][0], b[nt][1]);
                    }
                }
            }
        }
    };
    // The step's token scales, token biases 12582912 + sum(q) (so that the int32 sum started at
    // kFloatMagic gives code . q - sum(q) with one exact FADD) and row scales.
    float token_scale[4][2], token_bias[4][2], row_scale[MTiles][2];
    const auto load_scales = [&](const XStage& s, const CodeStage& c) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int token    = wn * 32 + nt * 8 + 2 * lid + j;
                token_scale[nt][j] = s.scale[token];
                token_bias[nt][j]  = __int_as_float(kFloatMagic + s.sum[token]);
            }
        }
#pragma unroll
        for (int mt = 0; mt < MTiles; ++mt) {
            row_scale[mt][0] = c.scale[wm * Tile::kWarpRows + mt * 16 + gid];
            row_scale[mt][1] = c.scale[wm * Tile::kWarpRows + mt * 16 + gid + 8];
        }
    };
    float acc[MTiles][4][4] = {};
    const auto update = [&](const int (&g)[MTiles][4][4]) {
#pragma unroll
        for (int mt = 0; mt < MTiles; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j      = e & 1;
                    const float unit = row_scale[mt][e >> 1] * token_scale[nt][j];
                    acc[mt][nt][e]   = fmaf(unit, __int_as_float(g[mt][nt][e]) - token_bias[nt][j],
                                            acc[mt][nt][e]);
                }
            }
        }
    };

    const XStager<kTokens, kThreads> stager(tid, qx, group_scale, group_sum, k, token0, live);
    load(0);
    decode(0, cs[0]);
    if (steps > 1) load(1);
    stager.issue(0, xs[0], tid);
    cp_commit();
    int g[MTiles][4][4];
    for (int step = 0; step < steps; ++step) {
        cp_wait<0>();
        // Stage `step` is visible and every warp is done reading step - 1, whose buffers refill.
        __syncthreads();
        if (step + 1 < steps) stager.issue(step + 1, xs[(step + 1) & 1], tid);
        cp_commit();
        const XStage& s    = xs[step & 1];
        const CodeStage& c = cs[step & 1];
        const auto next    = [&] {
            if (step + 1 < steps) {
                decode(step + 1, cs[(step + 1) & 1]);
                if (step + 2 < steps) load(step + 2);
            }
        };
        if (pong) {
            if (step > 0) update(g);
            next();
            multiply(s, c, g);
            load_scales(s, c);
        } else {
            multiply(s, c, g);
            next();
            load_scales(s, c);
            update(g);
        }
    }
    if (pong) update(g);
    // Every warp is done with the stages, which now hold the FP32 tile.
    __syncthreads();
    store_tall_tile<Tokens, MTiles>(outputs, acc, reinterpret_cast<float*>(gemm_smem), row0,
                                    token0, live, wm * Tile::kWarpRows, wn * 32, tid, accumulate);
}

// A row block's tokens are covered by `wide` 128-token tiles, then `narrow` 64-token tiles. The
// grid holds every row block's wide tiles first, then every row block's narrow tiles, token
// tiles fastest within each part, so the cheaper narrow CTAs fill the last resident wave.
__global__ void __launch_bounds__(kTallThreads, 1)
    gemm_tall_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                     const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                     const __half* __restrict__ scales, std::int64_t scale_row_halves, int k,
                     int tokens, int wide, int narrow, Outputs outputs, bool accumulate) {
    extern __shared__ __align__(128) std::uint8_t gemm_smem[];
    const int row_blocks = static_cast<int>(gridDim.x) / (wide + narrow);
    const int tile       = static_cast<int>(blockIdx.x);
    if (tile < row_blocks * wide) {
        gemm_tall_tile<128>(qx, group_scale, group_sum, codes, scales, scale_row_halves, k, tokens,
                            tile / wide * kTallRows, tile % wide * 128, outputs, accumulate,
                            gemm_smem);
    } else {
        const int rest = tile - row_blocks * wide;
        gemm_tall_tile<64>(qx, group_scale, group_sum, codes, scales, scale_row_halves, k, tokens,
                           rest / narrow * kTallRows, wide * 128 + rest % narrow * 64, outputs,
                           accumulate, gemm_smem);
    }
}

// ---------------------------------------------------------------------------------------------
// Small-T tensor-core GEMV for the verification of concurrent MTP lanes (T = 5..32; also any T
// of a weight whose rows are not a multiple of the GEMM's 64). The dp4a GEMV's work grows with T
// and the prefill GEMM gives each 64-row CTA the whole K (80 CTAs for a 5120-row weight), so
// neither fits here. A CTA of four warps owns 16 rows and 8 NTiles tokens; warp w takes the
// 128-column groups w, w + 4, ... of the row block (K % 1024 == 0: every warp has K / 512 >= 2
// groups), and the four FP32 partial sums are added in shared memory in warp order, so the
// weights are read once per token tile with one CTA per 16 rows.
//
// Per group each lane decodes one (row, unit) = (lane / 2, lane % 2) as the GEMV does (the next
// two groups' bytes are in flight meanwhile) into the warp's shared code words, and the warp
// multiplies 16 rows x 128 columns x 8 NTiles tokens with m16n8k32 s8 MMAs, A by `ldmatrix`.
// The k order inside the MMAs is permuted so that lane `lid` reads the activation words
// 8 lid .. 8 lid + 7 of its token with two 16-byte loads: MMA step ks, fragment half h (a0/a1
// and b0 for h = 0, a2/a3 and b1 for h = 1) holds group word 8 lid + 2 ks + h, i.e. code word
// w of unit u (group word 16 u + w) is stored at chunk w % 8, word 2 u + w / 8 of its row (the
// chunk XORed with the row as in the GEMM). The group sums correct the code offset as there.

constexpr int kSmallWarps    = 4;
constexpr int kSmallThreads  = 32 * kSmallWarps;
constexpr int kSmallRows     = 16;
constexpr int kSmallMaxTiles = 4; // 32 tokens per CTA

template <int NTiles>
__global__ void __launch_bounds__(kSmallThreads, NTiles <= 2 ? 4 : 3)
    small_t_kernel(const std::uint8_t* __restrict__ qx, const float* __restrict__ group_scale,
                   const int* __restrict__ group_sum, const std::uint8_t* __restrict__ codes,
                   const __half* __restrict__ scales, std::int64_t scale_row_halves, int n, int k,
                   int tokens, Outputs outputs, bool accumulate) {
    static_assert(NTiles >= 1 && NTiles <= kSmallMaxTiles);
    constexpr int kTokens = 8 * NTiles;
    // Per warp: 16 rows x 32 code words per group; after the loop, the partial sums [token][row].
    __shared__ __align__(128) std::uint32_t warp_words[kSmallWarps][kSmallRows * 32];
    static_assert(kTokens * kSmallRows <= kSmallRows * 32);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int row0   = static_cast<int>(blockIdx.x) * kSmallRows;
    const int token0 = static_cast<int>(blockIdx.y) * kTokens;
    const int live   = min(kTokens, tokens - token0);
    const int groups = k / kStageK;
    const std::int64_t row_bytes = std::int64_t(k / kUnitColumns) * kUnitBytes;
    std::uint32_t* code_words    = warp_words[warp];

    // Rows past n alias the last row and tokens past `live` the last live token; their results
    // are never stored.
    const int my_row             = lane >> 1;
    const int my_unit            = lane & 1;
    const std::uint8_t* my_codes = codes + std::int64_t(min(row0 + my_row, n - 1)) * row_bytes;
    const __half* top_scales     = scales + std::int64_t(min(row0 + gid, n - 1)) * scale_row_halves;
    const __half* bottom_scales =
        scales + std::int64_t(min(row0 + gid + 8, n - 1)) * scale_row_halves;
    // B operand: token nt * 8 + gid, bytes 32 lid .. 32 lid + 31 of each group; C fragment
    // tokens nt * 8 + 2 lid + j for the group scales and sums.
    int x_offset[NTiles], token_offset[NTiles][2];
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
        x_offset[nt] = (token0 + min(nt * 8 + gid, live - 1)) * k + 32 * lid;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            token_offset[nt][j] = (token0 + min(nt * 8 + 2 * lid + j, live - 1)) * groups;
        }
    }
    // ldmatrix row addresses of the A fragments (as in the GEMM).
    const int a_row   = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_chunk = lane >> 4;

    float acc[NTiles][4];
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;
    }
    std::uint32_t raw[4], next[4];
    load_unit(my_codes, 2 * warp + my_unit, raw);
    load_unit(my_codes, 2 * (warp + kSmallWarps) + my_unit, next);
#pragma unroll 1
    for (int group = warp; group < groups; group += kSmallWarps) {
        uint4 x[NTiles][2];
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
            const auto* source = reinterpret_cast<const uint4*>(qx + x_offset[nt] + group * kStageK);
            x[nt][0]           = __ldg(source);
            x[nt][1]           = __ldg(source + 1);
        }
        {
            std::uint32_t words[16];
            decode_unit(raw, words);
            std::uint32_t* row_words = code_words + my_row * 32 + 2 * my_unit;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                *reinterpret_cast<uint2*>(row_words + ((c ^ (my_row & 7)) << 2)) =
                    make_uint2(words[c], words[c + 8]);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) raw[i] = next[i];
        if (group + 2 * kSmallWarps < groups) {
            load_unit(my_codes, 2 * (group + 2 * kSmallWarps) + my_unit, next);
        }
        __syncwarp();
        int dot[NTiles][4];
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) dot[nt][e] = 0;
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned a0, a1, a2, a3;
            ldmatrix_x4(a0, a1, a2, a3,
                        smem_addr(&code_words[gemm_word(a_row, 4 * (2 * ks + a_chunk))]));
#pragma unroll
            for (int nt = 0; nt < NTiles; ++nt) {
                // b0, b1 = group words 8 lid + 2 ks and 8 lid + 2 ks + 1 of the lane's token.
                const uint4& xs = x[nt][ks >> 1];
                mma_s8(dot[nt][0], dot[nt][1], dot[nt][2], dot[nt][3], a0, a1, a2, a3,
                       ks & 1 ? xs.z : xs.x, ks & 1 ? xs.w : xs.y);
            }
        }
        // Every lane has its fragments before the next group's code words are stored.
        __syncwarp();
        const float top    = __half2float(__ldg(top_scales + group));
        const float bottom = __half2float(__ldg(bottom_scales + group));
        // Fragment C: [0], [1] are row gid, tokens 2 lid and 2 lid + 1; [2], [3] row gid + 8.
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float step = __ldg(group_scale + token_offset[nt][j] + group);
                const int sum    = __ldg(group_sum + token_offset[nt][j] + group);
                acc[nt][j] = fmaf(top * step, static_cast<float>(dot[nt][j] - sum), acc[nt][j]);
                acc[nt][2 + j] =
                    fmaf(bottom * step, static_cast<float>(dot[nt][2 + j] - sum), acc[nt][2 + j]);
            }
        }
    }

    // The warp's partial sums replace its code words as [token][row]; then every thread adds
    // the four warps' sums of (row, token) items in warp order, rows fastest (coalesced stores).
    auto* partial = reinterpret_cast<float*>(code_words);
#pragma unroll
    for (int nt = 0; nt < NTiles; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            partial[(nt * 8 + 2 * lid + (e & 1)) * kSmallRows + gid + (e >> 1) * 8] = acc[nt][e];
        }
    }
    __syncthreads();
    for (int item = tid; item < kTokens * kSmallRows; item += kSmallThreads) {
        const int row   = item % kSmallRows;
        const int token = item / kSmallRows;
        if (token >= live || row0 + row >= n) continue;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kSmallWarps; ++w) sum += reinterpret_cast<const float*>(warp_words[w])[item];
        store_row(output_row(outputs, row0 + row), token0 + token, sum, accumulate);
    }
}

} // namespace ninfer::ops::detail::t5_a8
