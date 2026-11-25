#include "ops/trellis/trellis_kernels.cuh"
#include "ops/trellis/trellis_profile.h"
#include "ops/common/mma.cuh"
#include "ops/common/device_multiprocessors.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail::trellis {
namespace {

constexpr int kThreads = 256;

// The INT8 grid of a 16-bit code window: state * golden ratio, folded as four 6-bit limbs.
// The four limbs land in the four bytes of (x >> 2), so one dot-product-accumulate sums them; the
// byte-wise form this replaces cost four shifts, four masks and three adds per weight.
__device__ __forceinline__ int grid_value(std::uint32_t state) {
    const std::uint32_t x = (state * 0x83DCD12Du) >> 2;
    return __dp4a(static_cast<int>(x & 0x3F3F3F3Fu), 0x01010101, -126);
}

// The tile position of ring index p: p = 8a + j -> (row, column) in exllamav3's tensor-core order.
__device__ __forceinline__ void tile_position(int p, int& row, int& col) {
    const int a = p >> 3, jj = p & 7;
    row = (a >> 2) + 8 * ((jj >> 2) & 1);
    col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
}

// One 16-bit window of the tile's circular stream ending at bit (p+1)*bits; stream bit q is
// bit 31 - q % 32 of word q / 32. The two `%` operations of the original form have runtime
// divisors, which compile to a full integer division on every decoded code; both ranges are
// bounded (0 <= s < total, 0 <= wi < nw), so a conditional fix-up is enough.
__device__ __forceinline__ std::uint32_t window(const std::uint32_t* tile, int nw, int bits, int p) {
    const int total = 256 * bits;
    int start = (p + 1) * bits - 16;
    if (start < 0) { start += total; }
    const int wi = start >> 5, o = start & 31;
    const int wn = (wi + 1 == nw) ? 0 : wi + 1;
    const std::uint64_t pair = (static_cast<std::uint64_t>(tile[wi]) << 32) | tile[wn];
    return static_cast<std::uint32_t>(pair >> (48 - o)) & 0xFFFFu;
}

// Compile-time bit width variant: constant-folds the stream geometry for the hot tile product.
template <int BITS>
__device__ __forceinline__ std::uint32_t window_t(const std::uint32_t* tile, int p) {
    constexpr int total = 256 * BITS;
    constexpr int nw = 8 * BITS;
    int start = (p + 1) * BITS - 16;
    if (start < 0) { start += total; }
    const int wi = start >> 5, o = start & 31;
    const int wn = (wi + 1 == nw) ? 0 : wi + 1;
    const std::uint64_t pair = (static_cast<std::uint64_t>(tile[wi]) << 32) | tile[wn];
    return static_cast<std::uint32_t>(pair >> (48 - o)) & 0xFFFFu;
}

__global__ void gather_scale_kernel(const __nv_bfloat16* x, const float* pre, const std::int32_t* pin,
                                    float* out, std::int64_t k, std::int64_t total) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t col = i / k;
        const std::int64_t j = i - col * k;
        const std::int32_t source = pin != nullptr ? pin[j] : static_cast<std::int32_t>(j);
        out[i] = pre[j] * __bfloat162float(x[source + col * k]);
    }
}

// Normalised Sylvester butterflies of `block` entries, one block-column per CUDA block.
__global__ void rht_kernel(float* v, std::int64_t n, int block, int columns) {
    extern __shared__ float s[];
    const int col = blockIdx.x;
    const std::int64_t b0 = static_cast<std::int64_t>(blockIdx.y) * block;
    float* base = v + static_cast<std::int64_t>(col) * n + b0;
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        s[i] = base[i];
    }
    __syncthreads();
    for (int h = 1; h < block; h <<= 1) {
        for (int i = threadIdx.x; i < block / 2; i += blockDim.x) {
            const int group = i / h, j = i - group * h;
            const int p = group * 2 * h + j;
            const float a = s[p], b = s[p + h];
            s[p] = a + b;
            s[p + h] = a - b;
        }
        __syncthreads();
    }
    const float c = rsqrtf(static_cast<float>(block));
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        base[i] = s[i] * c;
    }
}

__global__ void permute_kernel(const float* src, float* dst, const std::int32_t* perm, std::int64_t n,
                               int columns, bool scatter) {
    const std::int64_t total = n * columns;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t col = i / n;
        const std::int64_t d = i - col * n;
        const std::int64_t at = static_cast<std::int64_t>(perm[d]) + col * n;
        if (scatter) {
            dst[at] = src[i];
        } else {
            dst[i] = src[at];
        }
    }
}

// The per-run butterflies of a whole-slice side, one run of one column per CUDA block.
__global__ void fwht_runs_kernel(float* v, std::int64_t n, int shards, int m, std::int64_t run,
                                 int columns) {
    extern __shared__ float s[];
    const int col = blockIdx.x;
    const std::int64_t job = blockIdx.y;
    const int r = static_cast<int>(job / m), q = static_cast<int>(job % m);
    const std::int64_t shard_len = static_cast<std::int64_t>(m) * run;
    float* base = v + static_cast<std::int64_t>(col) * n + static_cast<std::int64_t>(r) * shard_len +
                  static_cast<std::int64_t>(q) * run;
    for (std::int64_t i = threadIdx.x; i < run; i += blockDim.x) {
        s[i] = base[i];
    }
    __syncthreads();
    for (std::int64_t h = 1; h < run; h <<= 1) {
        for (std::int64_t i = threadIdx.x; i < run / 2; i += blockDim.x) {
            const std::int64_t group = i / h, j = i - group * h;
            const std::int64_t p = group * 2 * h + j;
            const float a = s[p], b = s[p + h];
            s[p] = a + b;
            s[p + h] = a - b;
        }
        __syncthreads();
    }
    const float c = rsqrtf(static_cast<float>(run));
    for (std::int64_t i = threadIdx.x; i < run; i += blockDim.x) {
        base[i] = s[i] * c;
    }
    (void)shards;
}

// The Paley cross of a whole-slice side: one output run of one column per CUDA block.
__global__ void paley_kernel(const float* in, float* out, const signed char* h, std::int64_t n,
                             int shards, int m, std::int64_t run, bool transpose, float c,
                             int columns) {
    const int col = blockIdx.x;
    const std::int64_t job = blockIdx.y;
    const int r = static_cast<int>(job / m), a = static_cast<int>(job % m);
    const std::int64_t shard_len = static_cast<std::int64_t>(m) * run;
    const float* src = in + static_cast<std::int64_t>(col) * n + static_cast<std::int64_t>(r) * shard_len;
    float* dst = out + static_cast<std::int64_t>(col) * n + static_cast<std::int64_t>(r) * shard_len +
                 static_cast<std::int64_t>(a) * run;
    for (std::int64_t j = threadIdx.x; j < run; j += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < m; ++b) {
            const int sign = transpose ? h[b * m + a] : h[a * m + b];
            const float v = src[static_cast<std::int64_t>(b) * run + j];
            acc += sign > 0 ? v : -v;
        }
        dst[j] = acc * c;
    }
    (void)shards;
}

// The coded tile product. One CUDA block owns 16 output rows and up to 64 token columns; each
// thread decodes one of the 256 tile positions per input tile and then multiplies the 16-row slab.
template <int CHUNK, int BITS>
__global__ void tile_product_kernel(const float* xa, const std::uint32_t* codes,
                                    const float* s16, float* yr, int K, int N, int T,
                                    int kt_total) {
    __shared__ signed char tile[256];
    const int n0 = static_cast<int>(blockIdx.x) * 16;
    const int t0 = static_cast<int>(blockIdx.y) * CHUNK;
    const int tid = threadIdx.x;  // 0..255
    const int r = tid >> 4, lane = tid & 15;
    constexpr int nw = 8 * BITS;

    float acc[CHUNK / 16];
#pragma unroll
    for (int c = 0; c < CHUNK / 16; ++c) {
        acc[c] = 0.0f;
    }

    for (int i = 0; i < kt_total; ++i) {
        const std::uint32_t* words =
            codes + (static_cast<std::int64_t>(n0 / 16) * kt_total + i) * nw;
        int row = 0, col = 0;
        tile_position(tid, row, col);
        tile[row * 16 + col] = static_cast<signed char>(grid_value(window_t<BITS>(words, tid)));
        __syncthreads();
        const float scale = s16[i];
        const std::int64_t kbase = static_cast<std::int64_t>(i) * 16;
#pragma unroll
        for (int c = 0; c < CHUNK / 16; ++c) {
            const int token = t0 + lane + 16 * c;
            if (token < T) {
                float partial = 0.0f;
#pragma unroll
                for (int cc = 0; cc < 16; ++cc) {
                    partial += static_cast<float>(tile[r * 16 + cc]) *
                               xa[kbase + cc + static_cast<std::int64_t>(K) * token];
                }
                acc[c] += partial * scale;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int c = 0; c < CHUNK / 16; ++c) {
        const int token = t0 + lane + 16 * c;
        if (token < T) {
            yr[(n0 + r) + static_cast<std::int64_t>(N) * token] = acc[c];
        }
    }
}

// Pack two fp16 values into one 32-bit mma operand register (low half = first value).
__device__ __forceinline__ unsigned pack_f16_2(float lo, float hi) {
    const unsigned a = __half_as_ushort(__float2half(lo));
    const unsigned b = __half_as_ushort(__float2half(hi));
    return a | (b << 16);
}

// The coded weights are small integers, so skipping the float round trip saves one conversion per
// weight in the hot loop.
__device__ __forceinline__ unsigned pack_i16_2(int lo, int hi) {
    const unsigned a = __half_as_ushort(__int2half_rn(lo));
    const unsigned b = __half_as_ushort(__int2half_rn(hi));
    return a | (b << 16);
}

// libtrellis' `h_grid` (common.cuh, Apache-2.0): the grid sum lands straight in the fp16 bit
// pattern of 1024 + 4v, because dp4a's accumulator seeds the exponent. No int->fp16 conversion and
// no per-weight -126; `sub_correction` takes 1528 off both halves to give the exact 4v (|4v| <= 504
// is exact in fp16), and the code's factor 4 folds into the activation scale.
__device__ __forceinline__ unsigned h_grid(std::uint32_t state) {
    return __dp4a((state * 0x83DCD12Du) & 0xFCFCFCFCu, 0x01010101u, 0x6400u);
}

__device__ __forceinline__ unsigned half2_sub_1528(unsigned packed) {
    const __half2 offset = __half2half2(__ushort_as_half(0x65F8u));  // 1528
    const __half2 value  = __hsub2(*reinterpret_cast<const __half2*>(&packed), offset);
    return *reinterpret_cast<const unsigned*>(&value);
}

// ================================================================================================
// Reference-style prefill GEMM (ported from libtrellis gemm.cu, Apache-2.0 / Nari Labs).
//
// 1. Prologue: one CTA per token row folds the per-16 s16 scale into the activation, picks one
//    power-of-two exponent 2^e for the whole row, and stores fp16 activations in the m16n8k16
//    B-fragment order; xs[t] = 2^-e. The GEMM inner loop then never repacks activations.
// 2. GEMM: a CTA owns 128 outputs x TT tokens. Each warp decodes its weight tile once per k-tile
//    into an exact 4v A fragment and multiplies it with every token group's staged B fragment.
//    cp.async stages codes and activations S stages deep; the next k-tile's decode precedes the
//    current products. The epilogue runs the row scale and 1/4 through shared memory.
// ================================================================================================

constexpr int kW5 = 8;  // warps = output tiles per CTA

__device__ __forceinline__ int frag_pos(int k) {
    return 4 * ((k & 7) >> 1) + 2 * (k >> 3) + (k & 1);
}

__device__ __forceinline__ void cp16s(std::uint32_t smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// Rolling-window plan for one lane's eight positions of a tile (libtrellis common.cuh Plan<K>).
template <int K>
struct TrePlan {
    static constexpr int NW = 8 * K, TOTAL = 256 * K, SPAN = 16 / K + 1,
                         ANCH = (8 + SPAN - 1) / SPAN;
    int hi[ANCH], lo[ANCH], sh[ANCH];
    __device__ __forceinline__ explicit TrePlan(int lane) {
#pragma unroll
        for (int a = 0; a < ANCH; ++a) {
            const int e = ((8 * lane + 7 - a * SPAN + 1) * K) % TOTAL;
            const int w0 = ((e - 32 + TOTAL) >> 5) % NW;
            hi[a] = w0;
            lo[a] = (w0 + 1) % NW;
            sh[a] = 32 - (e & 31);
        }
    }
};

template <int C>
__device__ __forceinline__ std::uint32_t low16(std::uint32_t r) {
    if constexpr (C == 0) return r & 0xFFFFu;
    else if constexpr (C == 16) return r >> 16;
    else if constexpr (C % 8 == 0) return __byte_perm(r, 0u, 0x4400u | ((C / 8 + 1) << 4) | (C / 8));
    else return (r >> C) & 0xFFFFu;
}

template <int K, int I>
__device__ __forceinline__ std::uint32_t state_at(const std::uint32_t (&r)[TrePlan<K>::ANCH]) {
    constexpr int A = (7 - I) / TrePlan<K>::SPAN, ANCHOR = 7 - A * TrePlan<K>::SPAN;
    return low16<(ANCHOR - I) * K>(r[A]);
}

// Proven window extraction (same scheme as the gated old MMA kernel), assembled into the m16n8k16
// A fragment: a0=(g,2q,2q+1), a1=(g+8,2q,2q+1), a2=(g,2q+8,2q+9), a3=(g+8,2q+8,2q+9).
template <int BITS>
__device__ __forceinline__ void decode_tile_proven(const std::uint32_t* tw, int d, int lane,
                                                   std::uint32_t (&af)[4]) {
    constexpr int nw = 8 * BITS, total = 256 * BITS, kWL = ((31 + 7 * BITS) >> 5) + 2;
    const std::uint32_t* tile = tw + d * nw;
    int start0 = (8 * lane + 1) * BITS - 16;
    if (start0 < 0) start0 += total;
    const int o0 = start0 & 31, wi0 = start0 >> 5;
    std::uint32_t quad[kWL];
#pragma unroll
    for (int w = 0; w < kWL; ++w) {
        const int at = wi0 + w;
        quad[w] = tile[at >= nw ? at - nw : at];
    }
    const auto win = [&](int j) {
        const int dd = o0 + j * BITS;
        return __funnelshift_l(quad[(dd >> 5) + 1], quad[dd >> 5], dd & 31) >> 16;
    };
    std::uint32_t f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = h_grid(win(j));
    af[0] = __byte_perm(f[0], f[1], 0x5410);
    af[1] = __byte_perm(f[4], f[5], 0x5410);
    af[2] = __byte_perm(f[2], f[3], 0x5410);
    af[3] = __byte_perm(f[6], f[7], 0x5410);
    const __half2 off = __half2half2(__ushort_as_half(0x65F8u));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 v = __hsub2(*reinterpret_cast<const __half2*>(&af[i]), off);
        af[i] = *reinterpret_cast<const std::uint32_t*>(&v);
    }
}

// Decode the lane's eight positions, exact 4v A fragment (1528 subtracted in the fragment).
template <int K>
__device__ __forceinline__ void trellis_decode_exact(const std::uint32_t (&w)[2 * TrePlan<K>::ANCH],
                                                      const TrePlan<K>& plan, std::uint32_t (&af)[4]) {
    constexpr int ANCH = TrePlan<K>::ANCH;
    std::uint32_t r[ANCH];
#pragma unroll
    for (int x = 0; x < ANCH; ++x) {
        r[x] = __funnelshift_rc(w[2 * x], w[2 * x + 1], plan.sh[x]);
    }
    const std::uint32_t f0 = h_grid(state_at<K, 0>(r)), f1 = h_grid(state_at<K, 1>(r));
    const std::uint32_t f2 = h_grid(state_at<K, 2>(r)), f3 = h_grid(state_at<K, 3>(r));
    const std::uint32_t f4 = h_grid(state_at<K, 4>(r)), f5 = h_grid(state_at<K, 5>(r));
    const std::uint32_t f6 = h_grid(state_at<K, 6>(r)), f7 = h_grid(state_at<K, 7>(r));
    af[0] = __byte_perm(f0, f1, 0x5410);
    af[1] = __byte_perm(f4, f5, 0x5410);
    af[2] = __byte_perm(f2, f3, 0x5410);
    af[3] = __byte_perm(f6, f7, 0x5410);
    const __half2 off = __half2half2(__ushort_as_half(0x65F8u));  // 1528
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 v = __hsub2(*reinterpret_cast<const __half2*>(&af[i]), off);
        af[i] = *reinterpret_cast<const std::uint32_t*>(&v);
    }
}

// Prologue: one CTA per token. Folds s16 into the activation, per-token 2^e scale, stores the
// fp16 B-fragment plane. Input xa is FP32 [K, T] (pre/pin and the column side already applied).
__global__ void trellis_prologue_f32_kernel(const float* __restrict__ xa, int K, int T,
                                             const float* __restrict__ s16, __half* __restrict__ xf,
                                             float* __restrict__ xs, int groups) {
    __shared__ float wmax[kW5];
    const int t = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32, nb = K / 128;
    const float* xr = xa + static_cast<std::int64_t>(t) * K;
    float mx = 0.f;
    for (int blk = warp; blk < nb; blk += kW5) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = 128 * blk + lane + 32 * i;
            const float v = xr[k] * s16[k / 16];
            mx = fmaxf(mx, fabsf(v));
        }
    }
#pragma unroll
    for (int o = 16; o; o /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) wmax[warp] = mx;
    __syncthreads();
    mx = wmax[0];
#pragma unroll
    for (int w = 1; w < kW5; ++w) mx = fmaxf(mx, wmax[w]);
    int x2 = 0;
    frexpf(mx > 0.f && mx <= 3.0e38f ? mx : 1.f, &x2);
    const int e = 14 - x2;
    if (threadIdx.x == 0) xs[t] = ldexpf(1.f, -e);
    __half* base = xf + static_cast<size_t>(t / 8) * 1024 + (t % 8) * 16;
    const size_t cstep = static_cast<size_t>(groups) * 1024;
    for (int blk = warp; blk < nb; blk += kW5) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = 128 * blk + lane + 32 * i;
            v[i] = xr[k] * s16[k / 16];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = lane + 32 * i;
            base[blk * cstep + (k / 16) * 128 + frag_pos(k % 16)] = __float2half_rn(ldexpf(v[i], e));
        }
    }
}

// Stage plan for the prefill GEMM. TT = tokens per CTA, CK = k-tiles per stage, CPS = CTAs/SM.
template <int K, int TT, int CK, int CPS>
struct PfTile {
    static constexpr int WORDS = 8 * K;
    static constexpr int CODE_W = kW5 * CK * WORDS;
    static constexpr int GT = TT / 8;
    static constexpr int STAGE = CODE_W + GT * CK * 64;
    static constexpr int STAGE_BYTES = STAGE * 4;
    static constexpr int SMEM_SM = 102400;
    static constexpr int SMEM_CTA = 101376 - 64;
    static constexpr int fit() {
        for (int s = 4; s >= 2; --s) {
            if (CPS * (((s * STAGE_BYTES + 255) / 256) * 256 + 1024) <= SMEM_SM &&
                s * STAGE_BYTES <= SMEM_CTA) {
                return s;
            }
        }
        return 2;
    }
    static constexpr int S = fit();
    static constexpr int SMEM = S * STAGE_BYTES > 32 * 132 * 4 ? S * STAGE_BYTES : 32 * 132 * 4;
};

template <int BITS, int TT, int CK, int CPS>
__global__ void __launch_bounds__(256, CPS)
    pf_mma_kernel(const std::uint32_t* __restrict__ codes, int kt, int nt,
                  const std::uint32_t* __restrict__ xf, const float* __restrict__ xs, int rows,
                  int groups, float* __restrict__ yr, int N) {
    using Tile = PfTile<BITS, TT, CK, CPS>;
    constexpr int WORDS = Tile::WORDS, CODE_W = Tile::CODE_W, STAGE = Tile::STAGE, S = Tile::S;
    constexpr int GT = Tile::GT;
    extern __shared__ __align__(16) std::uint32_t sm[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, q = lane & 3;
    const int tb = blockIdx.x, cb = blockIdx.y, j0 = cb * kW5, g0 = tb * GT;
    const int rg = min(GT, groups - g0);
    const bool active = j0 + warp < nt;

    constexpr int UT = CK * WORDS / 4, UC = kW5 * UT, NC = (UC + 255) / 256;
    constexpr int UA = CK * 16, NA = GT * UA / 256;
    static_assert(GT * UA % 256 == 0, "stage copies");
    const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(sm));
    const std::uint32_t* csrc[NC];
    std::uint32_t cdst[NC];
    bool cok[NC];
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int u = threadIdx.x + 256 * i, w = u / UT, uu = u % UT;
        const int d = 4 * uu / WORDS, word = 4 * uu % WORDS;
        cok[i] = u < UC && j0 + w < nt;
        csrc[i] = codes + (static_cast<std::int64_t>(j0 + w) * kt + d) * WORDS + word;
        cdst[i] = base + 4 * (w * CK * WORDS + 4 * uu);
    }
    const std::int64_t cstep = static_cast<std::int64_t>(CK) * WORDS;
    int asrc[NA];
    std::uint32_t adst[NA];
    bool aok[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int u = threadIdx.x + 256 * i, r = u / UA, w = u % UA;
        aok[i] = r < rg;
        asrc[i] = (g0 + r) * 512 + 4 * w;
        adst[i] = base + 4 * (CODE_W + r * CK * 64 + 4 * w);
    }
    const size_t xstep = static_cast<size_t>(groups) * 512;
    auto issue = [&](int cl, int st) {
        const std::uint32_t off = 4u * STAGE * st;
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            if (cok[i]) cp16s(cdst[i] + off, csrc[i] + cl * cstep);
        }
        const std::uint32_t* xc = xf + static_cast<size_t>((cl * CK / 8)) * xstep + (cl * CK % 8) * 64;
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            if (aok[i]) cp16s(adst[i] + off, xc + asrc[i]);
        }
    };
#pragma unroll 1
    for (int s = 0; s < S - 1; ++s) {
        if (s < kt / CK) issue(s, s);
        cp_commit();
    }
    float acc[GT][4];
#pragma unroll
    for (int r = 0; r < GT; ++r)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[r][e] = 0.f;

    auto decode_tile = [&](const std::uint32_t* tw, int d, std::uint32_t (&af)[4]) {
        decode_tile_proven<BITS>(tw, d, lane, af);
    };

    const int nch = kt / CK;
#pragma unroll 1
    for (int c = 0, cur = 0, next = S - 1; c < nch; ++c) {
        cp_wait<S - 2>();
        __syncthreads();
        if (c + S - 1 < nch) issue(c + S - 1, next);
        cp_commit();
        const std::uint32_t* st = sm + cur * STAGE;
        cur = cur + 1 == S ? 0 : cur + 1;
        next = next + 1 == S ? 0 : next + 1;
        if (!active) continue;
        const std::uint32_t* tw = st + warp * CK * WORDS;
        const uint2* xb = reinterpret_cast<const uint2*>(st + CODE_W) + lane;
        std::uint32_t af[4];
        decode_tile(tw, 0, af);
#pragma unroll
        for (int d = 0; d < CK; ++d) {
            std::uint32_t an[4];
            if (d + 1 < CK) decode_tile(tw, d + 1, an);
#pragma unroll
            for (int r = 0; r < GT; ++r) {
                if (r < rg) {
                    const uint2 bb = xb[(r * CK + d) * 32];
                    ninfer::ops::mma_f16(acc[r][0], acc[r][1], acc[r][2], acc[r][3], af[0], af[1],
                                         af[2], af[3], bb.x, bb.y);
                }
            }
            if (d + 1 < CK) {
#pragma unroll
                for (int e = 0; e < 4; ++e) af[e] = an[e];
            }
        }
    }
    cp_wait<0>();

    // Epilogue: 32 tokens at a time through shared memory; yr[n + N*t] = acc * 1/4 * xs[t].
    constexpr int RED5 = 132;
    const int n = nt * 16, o0 = cb * 128, nv = min(128, n - o0);
    float* red = reinterpret_cast<float*>(sm);
    __syncthreads();
#pragma unroll
    for (int p = 0; p < TT / 32; ++p) {
#pragma unroll
        for (int r = 0; r < GT; ++r) {
            if (r / 4 != p) continue;
            const int tl = (r % 4) * 8 + 2 * q, ol = warp * 16 + g;
            red[tl * RED5 + ol] = acc[r][0];
            red[(tl + 1) * RED5 + ol] = acc[r][1];
            red[tl * RED5 + ol + 8] = acc[r][2];
            red[(tl + 1) * RED5 + ol + 8] = acc[r][3];
        }
        __syncthreads();
        for (int tl = warp; tl < 32; tl += kW5) {
            const int t = tb * TT + p * 32 + tl;
            if (t >= rows) break;
            float v[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) v[m] = red[tl * RED5 + lane + 32 * m];
            const float sc = xs[t];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                if (lane + 32 * m < nv) {
                    yr[(o0 + lane + 32 * m) + static_cast<std::int64_t>(N) * t] = v[m] * 0.25f * sc;
                }
            }
        }
        __syncthreads();
    }
}

// Tensor-core path. One warp owns one 16-output-row tile and up to 16 token rows. Lane l decodes
// the eight coded values p = 8l..8l+7 that are exactly its two m16n8k16 B fragments, so a whole
// 16x16 tile goes from the packed stream into the tensor core without a shared-memory round-trip.
// This is the same path libtrellis/ExLlamaV3 uses for GEMV; the coded bytes are unchanged.
//
// One warp owns one 16-output-row tile over `kTokenTiles` consecutive 16-token tiles, so a decoded
// code tile feeds four MMAs instead of one: prompting reads the coded bytes 4x less often and
// amortises the codebook decode 4x, which is what the prefill route was paying for.
constexpr int kTokenTiles = 8;
constexpr int kActStages  = 3;

template <int BITS>
__global__ void
    tile_product_mma_kernel(const float* xa, const std::uint32_t* codes,
                            const float* s16, float* yr, int K, int N, int T,
                            int kt_total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int J = static_cast<int>(blockIdx.x) * (kThreads / 32) + warp;
    const bool mine = J * 16 < N;  // every warp still joins the block's barriers below
    const int t0 = static_cast<int>(blockIdx.y) * (16 * kTokenTiles);
    const int gq = lane & 3;    // selects k rows 2*gq, 2*gq+1, 2*gq+8, 2*gq+9
    const int gn = lane >> 2;   // selects one of the eight n columns

    float c[kTokenTiles][4] = {};  // output n columns 0..7 of the tile, per token tile
    float d[kTokenTiles][4] = {};  // output n columns 8..15
    bool live[kTokenTiles];
    int rs0[kTokenTiles], rs8[kTokenTiles];
    int row0[kTokenTiles], row8[kTokenTiles];
#pragma unroll
    for (int tt = 0; tt < kTokenTiles; ++tt) {
        const int r0 = t0 + tt * 16 + gn;
        const int r8 = r0 + 8;
        live[tt] = t0 + tt * 16 < T;  // uniform across the warp
        row0[tt] = r0;
        row8[tt] = r8;
        rs0[tt] = r0 < T ? r0 : T - 1;
        rs8[tt] = r8 < T ? r8 : T - 1;
    }
    // The eight windows this lane consumes are consecutive in the circular bit stream (they start
    // BITS apart), so four words cover all of them. The per-window form issued two global loads
    // each -- sixteen per lane per K-tile -- and this loop is the whole coded-linear cost.
    constexpr int nw = 8 * BITS;
    constexpr int total = 256 * BITS;
    constexpr int kWordsPerLane = ((31 + 7 * BITS) >> 5) + 2;
    int start0 = (8 * lane + 1) * BITS - 16;
    if (start0 < 0) { start0 += total; }
    const int o0 = start0 & 31;
    const int wi0 = start0 >> 5;
    // The activation rows this block needs are 20 KB apart in `xa`, so the per-tile loads hit eight
    // sectors each. Staging the whole k-tile for the block's tokens in shared memory turns that into
    // one cooperative coalesced pass (the warps of a block share the same tokens).
    constexpr int kTokensPerBlock = 16 * kTokenTiles;
    constexpr int kStages         = kActStages;
    constexpr int kWarpsPerBlock  = kThreads / 32;
    __shared__ float s_act[kStages][kTokensPerBlock * 16];
    // The CTA's eight code tiles ride the same cp.async group as the activations (libtrellis stages
    // "the CTA's code words and its rows' activations" together), so the decode never waits on a
    // global load that was issued in the same iteration.
    __shared__ std::uint32_t s_codes[kStages][kWarpsPerBlock * nw];
    const int base_J = static_cast<int>(blockIdx.x) * kWarpsPerBlock;
    const auto stage_activation = [&](int kt, int stage) {
        float* dst = s_act[stage];
        for (int idx = threadIdx.x; idx < kTokensPerBlock; idx += kThreads) {
            const int token = min(t0 + idx, T - 1);
            const float* src = xa + static_cast<std::int64_t>(kt) * 16 +
                               static_cast<std::int64_t>(K) * token;
            // cp.async moves 4, 8 or 16 bytes per call, so one token's 16 floats are four copies.
            for (int c = 0; c < 4; ++c) {
                __pipeline_memcpy_async(dst + idx * 16 + c * 4, src + c * 4, 16);
            }
        }
        std::uint32_t* dst_codes = s_codes[stage];
        constexpr int kCopiesPerTile = nw / 4;
        const int last_tile          = (N / 16) - 1;
        for (int idx = threadIdx.x; idx < kWarpsPerBlock * kCopiesPerTile; idx += kThreads) {
            const int w  = idx / kCopiesPerTile;
            const int c  = idx - w * kCopiesPerTile;
            const int Jw = min(base_J + w, last_tile);
            __pipeline_memcpy_async(
                dst_codes + w * nw + c * 4,
                codes + (static_cast<std::int64_t>(Jw) * kt_total + kt) * nw + c * 4, 16);
        }
        __pipeline_commit();
    };
    for (int s = 0; s < kStages; ++s) {
        if (s < kt_total) { stage_activation(s, s); } else { __pipeline_commit(); }
    }
    for (int i = 0; i < kt_total; ++i) {
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();
        const float* act = s_act[i % kStages];
        const std::uint32_t* tile = s_codes[i % kStages] + warp * nw;
        const float quarter = s16[i] * 0.25f;
        std::uint32_t quad[kWordsPerLane];
#pragma unroll
        for (int w = 0; w < kWordsPerLane; ++w) {
            const int at = wi0 + w;
            quad[w]   = tile[at >= nw ? at - nw : at];
        }
        const auto window_of = [&quad, o0](int j) {
            const int d = o0 + j * BITS;
            // Bits [s, s+16) of the pair (quad[k] << 32) | quad[k+1], in one funnel shift plus a
            // 16-bit slice; building the 64-bit pair and shifting by 48-s cost three more
            // instructions on every weight of every coded projection.
            return __funnelshift_l(quad[(d >> 5) + 1], quad[d >> 5], d & 31) >> 16;
        };
        // h_grid + one packed half2 subtract (libtrellis decode_exact): the fragments hold the exact
        // 4v, so the code's factor 4 moves into the activation scale below.
        const unsigned b0a = half2_sub_1528(__byte_perm(h_grid(window_of(0)), h_grid(window_of(1)), 0x5410));
        const unsigned b1a = half2_sub_1528(__byte_perm(h_grid(window_of(2)), h_grid(window_of(3)), 0x5410));
        const unsigned b0b = half2_sub_1528(__byte_perm(h_grid(window_of(4)), h_grid(window_of(5)), 0x5410));
        const unsigned b1b = half2_sub_1528(__byte_perm(h_grid(window_of(6)), h_grid(window_of(7)), 0x5410));

        const int kbase = 2 * gq;
#pragma unroll
        for (int tt = 0; tt < kTokenTiles; ++tt) {
            if (!live[tt]) { continue; }
            const float* row8v = act + (tt * 16 + gn) * 16;
            const unsigned a0 = pack_f16_2(row8v[kbase] * quarter, row8v[kbase + 1] * quarter);
            const unsigned a1 = pack_f16_2(row8v[128 + kbase] * quarter, row8v[128 + kbase + 1] * quarter);
            const unsigned a2 = pack_f16_2(row8v[kbase + 8] * quarter, row8v[kbase + 9] * quarter);
            const unsigned a3 = pack_f16_2(row8v[128 + kbase + 8] * quarter, row8v[128 + kbase + 9] * quarter);
            ninfer::ops::mma_f16(c[tt][0], c[tt][1], c[tt][2], c[tt][3], a0, a1, a2, a3, b0a, b1a);
            ninfer::ops::mma_f16(d[tt][0], d[tt][1], d[tt][2], d[tt][3], a0, a1, a2, a3, b0b, b1b);
        }
        __syncthreads();  // the buffer just consumed may be refilled below
        if (i + kStages < kt_total) {
            stage_activation(i + kStages, i % kStages);
        } else {
            __pipeline_commit();  // keep the group count aligned with wait_prior
        }
    }

    if (!mine) { return; }  // the block's barriers are the last block-wide work
    const int n0 = 16 * J + 2 * gq;
#pragma unroll
    for (int tt = 0; tt < kTokenTiles; ++tt) {
        const int r0 = row0[tt], r8 = row8[tt];
        if (r0 < T) {
            if (n0 < N) { yr[n0 + static_cast<std::int64_t>(N) * r0] = c[tt][0]; }
            if (n0 + 1 < N) { yr[n0 + 1 + static_cast<std::int64_t>(N) * r0] = c[tt][1]; }
            if (n0 + 8 < N) { yr[n0 + 8 + static_cast<std::int64_t>(N) * r0] = d[tt][0]; }
            if (n0 + 9 < N) { yr[n0 + 9 + static_cast<std::int64_t>(N) * r0] = d[tt][1]; }
        }
        if (r8 < T) {
            if (n0 < N) { yr[n0 + static_cast<std::int64_t>(N) * r8] = c[tt][2]; }
            if (n0 + 1 < N) { yr[n0 + 1 + static_cast<std::int64_t>(N) * r8] = c[tt][3]; }
            if (n0 + 8 < N) { yr[n0 + 8 + static_cast<std::int64_t>(N) * r8] = d[tt][2]; }
            if (n0 + 9 < N) { yr[n0 + 9 + static_cast<std::int64_t>(N) * r8] = d[tt][3]; }
        }
    }
}

// Single-token route. One token gives only N/16 warps, which cannot fill the GPU (the largest
// weight reaches 640 warps, ~28% occupancy), so the loop is latency-bound instead of
// bandwidth-bound. Several blocks therefore accumulate disjoint K ranges into per-split planes and
// a reduction adds them; the rows-side Hadamard, `post` and `pout` are linear, so summing first is
// exact.
template <int BITS>
__global__ void tile_product_mma_split_kernel(const float* xa, const std::uint32_t* codes,
                                              const float* s16, float* partials, int K, int N,
                                              int kt_total, int kt_per_split) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int J = static_cast<int>(blockIdx.x) * (kThreads / 32) + warp;
    if (J * 16 >= N) { return; }
    const int split = static_cast<int>(blockIdx.z);
    const int kt_begin = split * kt_per_split;
    const int kt_end   = min(kt_total, kt_begin + kt_per_split);
    if (kt_begin >= kt_end) { return; }
    const int gq = lane & 3;
    const int gn = lane >> 2;
    constexpr int nw = 8 * BITS;
    constexpr int total = 256 * BITS;
    constexpr int kWordsPerLane = ((31 + 7 * BITS) >> 5) + 2;
    int start0 = (8 * lane + 1) * BITS - 16;
    if (start0 < 0) { start0 += total; }
    const int o0 = start0 & 31;
    const int wi0 = start0 >> 5;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const unsigned zero_rows = pack_f16_2(0.0f, 0.0f);
#pragma unroll 4
    for (int i = kt_begin; i < kt_end; ++i) {
        const std::uint32_t* words = codes + (static_cast<std::int64_t>(J) * kt_total + i) * nw;
        const float quarter = s16[i] * 0.25f;
        std::uint32_t quad[kWordsPerLane];
#pragma unroll
        for (int w = 0; w < kWordsPerLane; ++w) {
            const int at = wi0 + w;
            quad[w]   = words[at >= nw ? at - nw : at];
        }
        const auto window_of = [&quad, o0](int j) {
            const int d = o0 + j * BITS;
            // Bits [s, s+16) of the pair (quad[k] << 32) | quad[k+1], in one funnel shift plus a
            // 16-bit slice; building the 64-bit pair and shifting by 48-s cost three more
            // instructions on every weight of every coded projection.
            return __funnelshift_l(quad[(d >> 5) + 1], quad[d >> 5], d & 31) >> 16;
        };
        const unsigned b0a = half2_sub_1528(__byte_perm(h_grid(window_of(0)), h_grid(window_of(1)), 0x5410));
        const unsigned b1a = half2_sub_1528(__byte_perm(h_grid(window_of(2)), h_grid(window_of(3)), 0x5410));
        const unsigned b0b = half2_sub_1528(__byte_perm(h_grid(window_of(4)), h_grid(window_of(5)), 0x5410));
        const unsigned b1b = half2_sub_1528(__byte_perm(h_grid(window_of(6)), h_grid(window_of(7)), 0x5410));

        const int kbase = i * 16 + 2 * gq;
        const unsigned a0 = pack_f16_2(xa[kbase] * quarter, xa[kbase + 1] * quarter);
        const unsigned a2 = pack_f16_2(xa[kbase + 8] * quarter, xa[kbase + 9] * quarter);
        // Rows 8..15 carry no token here, so their operands are zero.
        ninfer::ops::mma_f16(c0, c1, c2, c3, a0, zero_rows, a2, zero_rows, b0a, b1a);
        ninfer::ops::mma_f16(d0, d1, d2, d3, a0, zero_rows, a2, zero_rows, b0b, b1b);
    }

    if (gn != 0) { return; }  // only row 0 exists, and `gn` selects the row of the a fragments
    float* out = partials + static_cast<std::int64_t>(split) * N;
    const int n0 = 16 * J + 2 * gq;
    if (n0 < N) { out[n0] = c0; }
    if (n0 + 1 < N) { out[n0 + 1] = c1; }
    if (n0 + 8 < N) { out[n0 + 8] = d0; }
    if (n0 + 9 < N) { out[n0 + 9] = d1; }
}

// Staged single-token route: the block's eight code tiles for one k-tile are copied cooperatively
// (coalesced 16B units) through a three-stage shared-memory ring, so the decode never stalls on a
// scattered global word load. Decode, MMA operand order and split accumulation are identical to
// tile_product_mma_split_kernel.
template <int BITS>
__global__ void tile_product_mma_split_staged_kernel(const float* xa, const std::uint32_t* codes,
                                                     const float* s16, float* partials, int K, int N,
                                                     int kt_total, int kt_per_split) {
    constexpr int nw = 8 * BITS, total = 256 * BITS, kStages = 3, units = nw / 4;
    __shared__ std::uint32_t s_codes[kStages][8 * nw];
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int base_J = static_cast<int>(blockIdx.x) * 8;
    const int J     = base_J + warp;
    const bool active = J * 16 < N;
    const int split = static_cast<int>(blockIdx.z);
    const int kt_begin = split * kt_per_split;
    const int kt_end   = min(kt_total, kt_begin + kt_per_split);
    if (kt_begin >= kt_end) { return; }

    auto stage = [&](int kt, int st) {
        if (kt < kt_end) {
            for (int idx = threadIdx.x; idx < 8 * units; idx += kThreads) {
                const int w = idx / units, c = idx - w * units;
                const int Jw = min(base_J + w, N / 16 - 1);
                __pipeline_memcpy_async(s_codes[st] + w * nw + c * 4,
                                         codes + (static_cast<std::int64_t>(Jw) * kt_total + kt) * nw + c * 4,
                                         16);
            }
        }
        __pipeline_commit();
    };
    for (int s = 0; s < kStages; ++s) { stage(kt_begin + s, s % kStages); }

    constexpr int kWordsPerLane = ((31 + 7 * BITS) >> 5) + 2;
    int start0 = (8 * lane + 1) * BITS - 16;
    if (start0 < 0) { start0 += total; }
    const int o0 = start0 & 31;
    const int wi0 = start0 >> 5;
    const int gq = lane & 3;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    const unsigned zero_rows = pack_f16_2(0.0f, 0.0f);

    for (int i = kt_begin; i < kt_end; ++i) {
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();
        const int st = (i - kt_begin) % kStages;
        const std::uint32_t* tile = s_codes[st] + warp * nw;
        if (i + kStages < kt_end) { stage(i + kStages, (i - kt_begin + kStages) % kStages); }
        else { __pipeline_commit(); }

        const float quarter = s16[i] * 0.25f;
        std::uint32_t quad[kWordsPerLane];
#pragma unroll
        for (int w = 0; w < kWordsPerLane; ++w) {
            const int at = wi0 + w;
            quad[w] = tile[at >= nw ? at - nw : at];
        }
        const auto window_of = [&quad, o0](int j) {
            const int d = o0 + j * BITS;
            return __funnelshift_l(quad[(d >> 5) + 1], quad[d >> 5], d & 31) >> 16;
        };
        const unsigned b0a = half2_sub_1528(__byte_perm(h_grid(window_of(0)), h_grid(window_of(1)), 0x5410));
        const unsigned b1a = half2_sub_1528(__byte_perm(h_grid(window_of(2)), h_grid(window_of(3)), 0x5410));
        const unsigned b0b = half2_sub_1528(__byte_perm(h_grid(window_of(4)), h_grid(window_of(5)), 0x5410));
        const unsigned b1b = half2_sub_1528(__byte_perm(h_grid(window_of(6)), h_grid(window_of(7)), 0x5410));

        const int kbase = i * 16 + 2 * gq;
        const unsigned a0 = pack_f16_2(xa[kbase] * quarter, xa[kbase + 1] * quarter);
        const unsigned a2 = pack_f16_2(xa[kbase + 8] * quarter, xa[kbase + 9] * quarter);
        ninfer::ops::mma_f16(c0, c1, c2, c3, a0, zero_rows, a2, zero_rows, b0a, b1a);
        ninfer::ops::mma_f16(d0, d1, d2, d3, a0, zero_rows, a2, zero_rows, b0b, b1b);
        __syncthreads();
    }

    if (!active) { return; }
    float* out = partials + static_cast<std::int64_t>(split) * N;
    const int n0 = 16 * J + 2 * gq;
    if (n0 < N) { out[n0] = c0; }
    if (n0 + 1 < N) { out[n0 + 1] = c1; }
    if (n0 + 8 < N) { out[n0 + 8] = d0; }
    if (n0 + 9 < N) { out[n0 + 9] = d1; }
}

// ================================================================================================
// Reference-structured decode GEMV (one token), from libtrellis gemv.cu's SINGLE row.
//
// The weights are the A operand: a k-tile is decoded once into the exact 4v fragment
// (`decode_tile_proven`) and multiplied with the prologue's prepared fp16 B fragment, so a k-tile
// costs one m16n8k16 MMA instead of the old split kernel's two (its second MMA zeroes the fragment's
// rows 8..15). Codes and the chunk's activation block ride a CK = 8 k-tile cp.async ring and the
// copy of chunk c + 1 is issued before chunk c's products, so the decode never waits on a load
// issued in the same iteration -- the plain split kernel's per-tile LDG -> decode chain.
//
// B fragment addressing: the prologue puts token 0's 16 halves of a k-tile at words [0, 8) of the
// k-tile's 64-word area in fragment order, so lane q of every group reads uint2 q; the mma's eight
// columns then all hold the same token and only column 0 (lanes with q == 0) is stored.
// ================================================================================================
constexpr int kGemvCK = 8;  // k-tiles per chunk: one 128-input activation block
constexpr int kGemvStages = 2;

// One CTA is pf's eight warps, each owning one 16-row tile of the column block. A four-warp block
// (twice the CTAs, half the code ring) was 1.5% slower in A/B, so the shape stays at kW5.
template <int BITS>
__global__ void gemv_split_kernel(const std::uint32_t* __restrict__ codes,
                                  const std::uint32_t* __restrict__ xf,
                                  const float* __restrict__ xs, float* __restrict__ partials, int kt,
                                  int nt, int N, int kb, int groups) {
    constexpr int WORDS = 8 * BITS, CK = kGemvCK, S = kGemvStages, THREADS = kW5 * 32;
    constexpr int WPC = CK * WORDS;     // words per warp's chunk (CK consecutive k-tiles)
    constexpr int CODE_W = kW5 * WPC;   // the block's code area
    constexpr int ACT_W = CK * 64;      // one 128-input block of one row group
    constexpr int STAGE = CODE_W + ACT_W;
    extern __shared__ __align__(16) std::uint32_t sm[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = lane & 3;
    const int j0 = static_cast<int>(blockIdx.x) * kW5, j = j0 + warp;
    const bool active = j < nt;  // every warp still joins the block's barriers below
    const int split = static_cast<int>(blockIdx.y);
    const int i0 = split * kb, nk = min(kt - i0, kb);
    if (nk <= 0) { return; }
    const int nch = nk / CK;

    // This chunk's 16-byte copies: each warp's CK tiles are one contiguous run of WPC words.
    constexpr int UPW = WPC / 4, UC = kW5 * UPW, NC = (UC + THREADS - 1) / THREADS;
    const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(sm));
    const std::uint32_t* csrc[NC];
    std::uint32_t cdst[NC];
    bool cok[NC];
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const int u = threadIdx.x + THREADS * i, w = u / UPW, uu = u - w * UPW;
        cok[i] = u < UC && j0 + w < nt;
        csrc[i] = codes + (static_cast<std::int64_t>(min(j0 + w, nt - 1)) * kt + i0) * WORDS + 4 * uu;
        cdst[i] = base + 4 * (w * WPC + 4 * uu);
    }
    const std::uint32_t abase = base + 4 * CODE_W;
    const auto issue = [&](int c, int st) {
        const std::uint32_t off = 4u * STAGE * st;
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            if (cok[i]) { cp16s(cdst[i] + off, csrc[i] + c * WPC); }
        }
        // The chunk is exactly one 128-input block: its row-group slab is contiguous, so 128 threads
        // copy one 16-byte unit each and the fragment read keeps pf's uint2 addressing.
        if (threadIdx.x < ACT_W / 4) {
            cp16s(abase + off + 16u * threadIdx.x,
                  xf + static_cast<std::int64_t>(i0 / CK + c) * groups * 512 + 4 * threadIdx.x);
        }
        cp_commit();
    };
#pragma unroll 1
    for (int s = 0; s < S - 1; ++s) {
        if (s < nch) { issue(s, s); } else { cp_commit(); }
    }
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
#pragma unroll 1
    for (int c = 0, cur = 0, next = S - 1; c < nch; ++c) {
        cp_wait<S - 2>();
        __syncthreads();
        if (c + S - 1 < nch) { issue(c + S - 1, next); } else { cp_commit(); }
        const std::uint32_t* st = sm + cur * STAGE;
        cur = cur + 1 == S ? 0 : cur + 1;
        next = next + 1 == S ? 0 : next + 1;
        if (!active) { continue; }
        const std::uint32_t* tw = st + warp * WPC;
        const uint2* xb = reinterpret_cast<const uint2*>(st + CODE_W) + q;
        // The next k-tile's decode is issued before this one's product, so the tube stays fed.
        std::uint32_t af[4], an[4];
        decode_tile_proven<BITS>(tw, 0, lane, af);
#pragma unroll
        for (int d = 0; d < CK; ++d) {
            if (d + 1 < CK) { decode_tile_proven<BITS>(tw, d + 1, lane, an); }
            const uint2 bb = xb[d * 32];
            ninfer::ops::mma_f16(c0, c1, c2, c3, af[0], af[1], af[2], af[3], bb.x, bb.y);
            if (d + 1 < CK) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { af[e] = an[e]; }
            }
        }
    }
    cp_wait<0>();
    if (!active) { return; }
    // Column 0 is the only live mma column: lanes with q == 0 hold outputs lane/4 and lane/4 + 8.
    const float row_scale = 0.25f * __ldg(xs);
    if (q == 0) {
        const int o = 16 * j + (lane >> 2);
        float* part = partials + static_cast<std::int64_t>(split) * N;
        if (o < N) { part[o] = c0 * row_scale; }
        if (o + 8 < N) { part[o + 8] = c2 * row_scale; }
    }
}

__global__ void reduce_split_kernel(const float* partials, float* yr, int N, int splits) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += gridDim.x * blockDim.x) {
        float sum = partials[n];
        for (int s = 1; s < splits; ++s) {
            sum += partials[static_cast<std::int64_t>(s) * N + n];
        }
        yr[n] = sum;
    }
}

__global__ void post_gather_kernel(const float* yr, const float* post, const std::int32_t* pout,
                                   __nv_bfloat16* out, std::int64_t n, std::int64_t total) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t col = i / n;
        const std::int64_t row = i - col * n;
        const std::int32_t source = pout != nullptr ? pout[row] : static_cast<std::int32_t>(row);
        // Both the row scale and the tile row are in storage order: `post` belongs to the stored row
        // that `pout[row]` names, not to the output row (a banded head keeps one post per band row).
        out[i] = __float2bfloat16(post[source] * yr[static_cast<std::int64_t>(source) + col * n]);
    }
}

// One coded row per CUDA block: decode the 16-row group holding the id, keep the wanted row.
__global__ void decode_row_kernel(const std::int32_t* ids, const std::uint32_t* codes,
                                  const float* s16, float* g, int K, int N, int bits,
                                  int kt_total) {
    __shared__ signed char tile[256];
    const int t = blockIdx.x;
    const std::int32_t id = ids[t];
    if (id < 0 || id >= N) {
        return;
    }
    const std::int64_t J = id / 16;
    const int want = id - 16 * static_cast<int>(J);
    const int nw = 8 * bits;
    for (int i = 0; i < kt_total; ++i) {
        const std::uint32_t* words = codes + (J * kt_total + i) * nw;
        int row = 0, col = 0;
        tile_position(threadIdx.x, row, col);
        tile[row * 16 + col] = static_cast<signed char>(grid_value(window(words, nw, bits, threadIdx.x)));
        __syncthreads();
        const float scale = s16[i];
        if (threadIdx.x < 16) {
            g[static_cast<std::int64_t>(i) * 16 + threadIdx.x +
              static_cast<std::int64_t>(K) * t] = static_cast<float>(tile[want * 16 + threadIdx.x]) * scale;
        }
        __syncthreads();
    }
}

// A coded embedding row is the row of the linear map W, so it also carries that row's `post`
// scale: out[j] = post[row_id] * pre[j] * T_cols^T(tile row)[j], gathered by `pin`.
__global__ void pre_gather_kernel(const float* g, const float* pre, const std::int32_t* pin,
                                  const float* post, const std::int32_t* ids,
                                  __nv_bfloat16* rows, std::int64_t k, std::int64_t total) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t col = i / k;
        const std::int64_t j = i - col * k;
        const float scale = post != nullptr ? post[ids[col]] : 1.0F;
        const float v = scale * pre[j] * g[i];
        if (pin != nullptr) {
            rows[static_cast<std::int64_t>(pin[j]) + col * k] = __float2bfloat16(v);
        } else {
            rows[i] = __float2bfloat16(v);
        }
    }
}

int grid_for(std::int64_t total) {
    const std::int64_t blocks = (total + kThreads - 1) / kThreads;
    return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

// The banded head's plane is one token-major slab per band, so a global row `g` lives at
// `base(band) * tokens + t * rows(band) + (g - base(band))`. The band table rides in the launch
// parameters: nothing is allocated, so this stays legal inside a CUDA graph capture.
struct BandRows {
    std::int32_t count;
    std::int32_t rows[32];
};

__global__ void gather_banded_kernel(const __nv_bfloat16* src, const std::int32_t* index,
                                     __nv_bfloat16* dst, std::int64_t rows, int tokens,
                                     BandRows bands) {
    const std::int64_t total = rows * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t r = i / tokens;
        const std::int64_t t = i - r * tokens;
        const std::int64_t g = index != nullptr ? static_cast<std::int64_t>(index[r]) : r;
        std::int64_t base = 0;
        std::int64_t from = 0;
        for (std::int32_t b = 0; b < bands.count; ++b) {
            const std::int64_t n = bands.rows[b];
            if (g < base + n) {
                from = base * tokens + t * n + (g - base);
                break;
            }
            base += n;
        }
        dst[r + t * rows] = src[from];
    }
}

// The same read, displaced by `dst_row` rows and with a null map meaning the identity, or the
// interleaved pattern `(r / group) * stride + origin + r % group` when `group` is nonzero.
__global__ void gather_rows_into_kernel(const __nv_bfloat16* src, const std::int32_t* index,
                                        __nv_bfloat16* dst, std::int64_t rows, std::int64_t dst_row,
                                        std::int32_t group, std::int32_t stride, std::int32_t origin,
                                        int tokens, std::int64_t src_rows, std::int64_t dst_rows) {
    const std::int64_t total = rows * tokens;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t r = i / tokens;
        const std::int64_t t = i - r * tokens;
        std::int64_t from      = dst_row + r;
        if (index != nullptr) {
            from = static_cast<std::int64_t>(index[r]);
        } else if (group > 0) {
            from = (r / group) * stride + origin + r % group;
        }
        dst[(dst_row + r) + t * dst_rows] = src[from + t * src_rows];
    }
}

struct PfArgs {
    const float* xa;
    const std::uint32_t* codes;
    const float* s16;
    const std::uint16_t* xf;
    const float* xs;
    float* yr;
    int K, N, rows, groups, kt, nt;
};

int pick_pf_tiling(int rows, int nt, int sms) {
    const int cbs = (nt + kW5 - 1) / kW5;
    const int c128 = cbs * ((rows + 127) / 128), c64 = cbs * ((rows + 63) / 64);
    auto load = [&](int ctas, float pair, float single) {
        const int n = (ctas + sms - 1) / sms;
        return (n / 2) * pair + (n % 2) * single;
    };
    const float t_c8 = static_cast<float>(c128 + sms - 1) / sms;
    const float t_x2 = load(c128, 1.8f, 1.03f);
    const float t_64 = load(c64, 1.1f, 0.6f);
    if (t_64 < t_x2 && t_64 < t_c8) return 0;
    return t_c8 < t_x2 ? 1 : 2;
}

template <int BITS, int TT, int CK, int CPS>
void launch_pf(const PfArgs& a, cudaStream_t stream) {
    using Tile = PfTile<BITS, TT, CK, CPS>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(pf_mma_kernel<BITS, TT, CK, CPS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, Tile::SMEM);
        configured = true;
    }
    const dim3 grid((a.rows + TT - 1) / TT, (a.nt + kW5 - 1) / kW5, 1);
    pf_mma_kernel<BITS, TT, CK, CPS><<<grid, 256, Tile::SMEM, stream>>>(
        a.codes, a.kt, a.nt, reinterpret_cast<const std::uint32_t*>(a.xf), a.xs, a.rows, a.groups,
        a.yr, a.N);
}

template <int BITS>
void launch_pf_bits(const PfArgs& a, int tiling, cudaStream_t stream) {
    switch (tiling) {
        case 0: launch_pf<BITS, 64, 8, 2>(a, stream); break;
        case 1: launch_pf<BITS, 128, 8, 1>(a, stream); break;
        default: launch_pf<BITS, 128, 4, 2>(a, stream); break;
    }
}

} // namespace

bool trellis_pf_decode_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TRELLIS_PFDECODE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

// On by default: decode 20.4 tok/s against the plain split's 18.8 on the same artifact (T=1 gate
// unchanged). NINFER_TRELLIS_GEMV=0 restores the old route.
bool trellis_gemv_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TRELLIS_GEMV");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    return on;
}

namespace {
int decode_split_count() {
    static const int env = [] {
        const char* v = std::getenv("NINFER_TRELLIS_SPLITS");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    return env > 0 ? std::min(kDecodeSplits, env) : kDecodeSplits;
}

template <int BITS>
void launch_gemv_split(const std::uint32_t* codes, const std::uint32_t* xf, const float* xs,
                       float* partials, int kt, int nt, int N, int kb, int groups, int splits,
                       cudaStream_t stream) {
    constexpr int WORDS = 8 * BITS;
    constexpr std::size_t SMEM =
        static_cast<std::size_t>(kGemvStages) * (kW5 * kGemvCK * WORDS + kGemvCK * 64) * 4;
    const dim3 grid(static_cast<unsigned>((nt + kW5 - 1) / kW5), static_cast<unsigned>(splits));
    gemv_split_kernel<BITS><<<grid, kW5 * 32, SMEM, stream>>>(codes, xf, xs, partials, kt, nt, N, kb,
                                                             groups);
}
} // namespace

void gather_scale(const Tensor& x, const Weight& w, float* out, int tokens, std::int64_t K,
                  cudaStream_t stream) {
    gather_scale_kernel<<<grid_for(K * tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(w.trellis_pre),
        w.trellis_pin, out, K, K * tokens);
}

void rht_inplace(float* v, std::int64_t n, int block, int columns, cudaStream_t stream) {
    rht_kernel<<<dim3(static_cast<unsigned>(columns), static_cast<unsigned>(n / block)), 128,
                 static_cast<std::size_t>(block) * sizeof(float), stream>>>(v, n, block, columns);
}

void permute_side(const float* src, float* dst, const std::int32_t* perm, std::int64_t n, int columns,
                  bool scatter, cudaStream_t stream) {
    permute_kernel<<<grid_for(n * columns), kThreads, 0, stream>>>(src, dst, perm, n, columns, scatter);
}

void hadfull_pass(const float* src, float* dst, float* tmp, float* tmp2, const SidePlan& plan,
                  const DevicePlan& device, bool inverse, int columns, cudaStream_t stream) {
    const std::int64_t n = plan.n;
    // The logical order: perm[d] is the stored index of logical position d.
    permute_side(src, tmp, device.perm, n, columns, false, stream);
    const int work = plan.shards * plan.m;
    const float c_m = rsqrtf(static_cast<float>(plan.m));
    const unsigned nx = static_cast<unsigned>(columns), ny = static_cast<unsigned>(work);
    const std::size_t fwht_bytes = static_cast<std::size_t>(plan.run) * sizeof(float);
    if (!inverse) {
        fwht_runs_kernel<<<dim3(nx, ny), 128, fwht_bytes, stream>>>(tmp, n, plan.shards, plan.m,
                                                                    plan.run, columns);
        paley_kernel<<<dim3(nx, ny), 128, 0, stream>>>(tmp, tmp2, device.paley, n, plan.shards, plan.m,
                                                       plan.run, false, c_m, columns);
    } else {
        paley_kernel<<<dim3(nx, ny), 128, 0, stream>>>(tmp, tmp2, device.paley, n, plan.shards, plan.m,
                                                       plan.run, true, c_m, columns);
        fwht_runs_kernel<<<dim3(nx, ny), 128, fwht_bytes, stream>>>(tmp2, n, plan.shards, plan.m,
                                                                    plan.run, columns);
    }
    permute_side(tmp2, dst, device.perm, n, columns, true, stream);
}

void tile_product(const float* xa, const Weight& w, float* yr, int tokens, std::uint16_t* xf,
                  float* xs, cudaStream_t stream) {
    const int kt_total = w.k / 16;
    const auto* codes = static_cast<const std::uint32_t*>(w.qdata);
    const auto* s16 = static_cast<const float*>(w.trellis_s16);
    static const bool force_simt = [] {
        const char* v = std::getenv("NINFER_TRELLIS_SIMT");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    static const bool old_mma = [] {
        const char* v = std::getenv("NINFER_TRELLIS_OLDMMA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    const int bits = bits_of(w.qtype);

    if (!force_simt && !old_mma && (tokens > 1 || trellis_pf_decode_enabled()) && w.k % 128 == 0) {
        const int groups = (tokens + 7) / 8;
        trellis_prologue_f32_kernel<<<tokens, 256, 0, stream>>>(
            xa, w.k, tokens, s16, reinterpret_cast<__half*>(xf), xs, groups);
        const PfArgs a{xa, codes, s16, xf, xs,
                       yr, w.k, w.n, tokens, groups, kt_total, w.n / 16};
        const int sms = ::ninfer::ops::detail::current_device_multiprocessors(36);
        const int tiling = pick_pf_tiling(tokens, w.n / 16, sms);
        switch (bits) {
#define LAUNCH_PF(B) case B: launch_pf_bits<B>(a, tiling, stream); break
            LAUNCH_PF(1);
            LAUNCH_PF(2);
            LAUNCH_PF(3);
            LAUNCH_PF(4);
            LAUNCH_PF(5);
            LAUNCH_PF(6);
#undef LAUNCH_PF
        default: break;
        }
        return;
    }

    if (!force_simt) {
        const unsigned nx = static_cast<unsigned>((w.n / 16 + kThreads / 32 - 1) / (kThreads / 32));
        const unsigned ny =
            static_cast<unsigned>((tokens + 16 * kTokenTiles - 1) / (16 * kTokenTiles));
        const dim3 grid(nx, ny);
        switch (bits) {
#define LAUNCH_MMA(B) \
    case B: tile_product_mma_kernel<B><<<grid, kThreads, 0, stream>>>(xa, codes, s16, yr, w.k, w.n, tokens, kt_total); break
            LAUNCH_MMA(1);
            LAUNCH_MMA(2);
            LAUNCH_MMA(3);
            LAUNCH_MMA(4);
            LAUNCH_MMA(5);
            LAUNCH_MMA(6);
#undef LAUNCH_MMA
        default: break;
        }
        return;
    }

    constexpr int kChunk = 64;
    const unsigned nx = static_cast<unsigned>(w.n / 16);
    const unsigned ny = static_cast<unsigned>((tokens + kChunk - 1) / kChunk);
    const dim3 grid(nx, ny);
    switch (bits) {
#define LAUNCH_TILE(B) \
    case B: tile_product_kernel<kChunk, B><<<grid, kThreads, 0, stream>>>(xa, codes, s16, yr, w.k, w.n, tokens, kt_total); break
        LAUNCH_TILE(1);
        LAUNCH_TILE(2);
        LAUNCH_TILE(3);
        LAUNCH_TILE(4);
        LAUNCH_TILE(5);
        LAUNCH_TILE(6);
#undef LAUNCH_TILE
    default: break;
    }
}

void tile_product_split1(const float* xa, const Weight& w, float* yr, float* partials,
                         cudaStream_t stream) {
    const int kt_total = w.k / 16;
    const auto* codes = static_cast<const std::uint32_t*>(w.qdata);
    const auto* s16 = static_cast<const float*>(w.trellis_s16);
    const int bits = bits_of(w.qtype);
    static const bool staged = [] {
        const char* v = std::getenv("NINFER_TRELLIS_STAGEDEC");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    const int splits = decode_split_count();
    const int per = (kt_total + splits - 1) / splits;
    const unsigned nx = static_cast<unsigned>((w.n / 16 + kThreads / 32 - 1) / (kThreads / 32));
    const dim3 grid(nx, 1, static_cast<unsigned>(splits));
    switch (bits) {
#define LAUNCH_SPLIT(B)                                                                          \
    case B:                                                                                      \
        if (staged) {                                                                            \
            tile_product_mma_split_staged_kernel<B><<<grid, kThreads, 0, stream>>>(             \
                xa, codes, s16, partials, w.k, w.n, kt_total, per);                              \
        } else {                                                                                 \
            tile_product_mma_split_kernel<B><<<grid, kThreads, 0, stream>>>(xa, codes, s16,     \
                                                                             partials, w.k,     \
                                                                             w.n, kt_total, per);\
        }                                                                                        \
        break
        LAUNCH_SPLIT(1);
        LAUNCH_SPLIT(2);
        LAUNCH_SPLIT(3);
        LAUNCH_SPLIT(4);
        LAUNCH_SPLIT(5);
        LAUNCH_SPLIT(6);
#undef LAUNCH_SPLIT
    default: return;
    }
    reduce_split_kernel<<<grid_for(w.n), kThreads, 0, stream>>>(partials, yr, w.n, splits);
}

// Decode GEMV in the reference's orientation. `xf`/`xs` are the prologue scratch (fp16 B-fragment
// plane, per-token 2^-e); the prologue runs here for one token, then the chunked kernel and the same
// split reduction. Falls back to the plain split route when K does not cover whole 128-input blocks.
void tile_product_split1_gemv(const float* xa, const Weight& w, float* yr, float* partials,
                              std::uint16_t* xf, float* xs, cudaStream_t stream) {
    const int kt = w.k / 16, nt = w.n / 16;
    if (w.k % 128 != 0 || kt < kGemvCK) {
        tile_product_split1(xa, w, yr, partials, stream);
        return;
    }
    const int splits = decode_split_count();
    const int bits = bits_of(w.qtype);
    const auto* codes = static_cast<const std::uint32_t*>(w.qdata);
    const auto* s16 = static_cast<const float*>(w.trellis_s16);
    {
        profile::Scope stage("gemv prologue", stream);
        trellis_prologue_f32_kernel<<<1, kThreads, 0, stream>>>(
            xa, w.k, 1, s16, reinterpret_cast<__half*>(xf), xs, 1);
    }
    // Splits in whole chunks (k-tiles per split rounded up to CK), so every chunk is one 128-block.
    const int kb = ((kt + splits * kGemvCK - 1) / (splits * kGemvCK)) * kGemvCK;
    const int active = (kt + kb - 1) / kb;
    const auto* xfw = reinterpret_cast<const std::uint32_t*>(xf);
    {
        profile::Scope stage("gemv kernel", stream);
        switch (bits) {
#define LAUNCH_GEMV(B)                                                                              \
    case B:                                                                                         \
        launch_gemv_split<B>(codes, xfw, xs, partials, kt, nt, w.n, kb, 1, active, stream);          \
        break
            LAUNCH_GEMV(1);
            LAUNCH_GEMV(2);
            LAUNCH_GEMV(3);
            LAUNCH_GEMV(4);
            LAUNCH_GEMV(5);
            LAUNCH_GEMV(6);
#undef LAUNCH_GEMV
        default: return;
        }
    }
    reduce_split_kernel<<<grid_for(w.n), kThreads, 0, stream>>>(partials, yr, w.n, active);
}

void post_and_gather(const float* yr, const Weight& w, Tensor& out, int tokens, cudaStream_t stream) {
    post_gather_kernel<<<grid_for(static_cast<std::int64_t>(w.n) * tokens), kThreads, 0, stream>>>(
        yr, static_cast<const float*>(w.trellis_post), w.trellis_pout,
        static_cast<__nv_bfloat16*>(out.data), w.n, static_cast<std::int64_t>(w.n) * tokens);
}

void decode_row(const Tensor& ids, const Weight& w, float* g, int tokens, cudaStream_t stream) {
    const int bits = bits_of(w.qtype);
    decode_row_kernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint32_t*>(w.qdata),
        static_cast<const float*>(w.trellis_s16), g, w.k, w.n, bits, w.k / 16);
}

void pre_and_gather_bf16(const float* g, const Weight& w, const Tensor& ids, Tensor& rows, int tokens,
                         cudaStream_t stream) {
    pre_gather_kernel<<<grid_for(static_cast<std::int64_t>(w.k) * tokens), kThreads, 0, stream>>>(
        g, static_cast<const float*>(w.trellis_pre), w.trellis_pin,
        static_cast<const float*>(w.trellis_post), static_cast<const std::int32_t*>(ids.data),
        static_cast<__nv_bfloat16*>(rows.data), w.k, static_cast<std::int64_t>(w.k) * tokens);
}

void gather_rows_bf16(const void* src, const std::int32_t* index, std::int64_t rows, int tokens,
                      const std::int32_t* band_rows, std::int32_t band_count, void* dst,
                      cudaStream_t stream) {
    BandRows bands{};
    if (band_count > 32) { band_count = 32; }
    bands.count = band_count;
    for (std::int32_t i = 0; i < band_count; ++i) { bands.rows[i] = band_rows[i]; }
    gather_banded_kernel<<<grid_for(rows * tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src), index, static_cast<__nv_bfloat16*>(dst), rows, tokens,
        bands);
}

void gather_rows_into_bf16(const void* src, const std::int32_t* index, std::int64_t rows,
                           std::int64_t dst_row, std::int32_t group, std::int32_t stride,
                           std::int32_t origin, int tokens, std::int64_t src_rows,
                           std::int64_t dst_rows, void* dst, cudaStream_t stream) {
    gather_rows_into_kernel<<<grid_for(rows * tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src), index, static_cast<__nv_bfloat16*>(dst), rows,
        dst_row, group, stride, origin, tokens, src_rows, dst_rows);
    const char* dbg = std::getenv("NINFER_TRACE_GATHER");
    if (dbg == nullptr || dbg[0] == '\0' || dbg[0] == '0' || index != nullptr || group > 0) {
        return;
    }
    cudaStreamSynchronize(stream);
    const std::size_t count = static_cast<std::size_t>(rows) * tokens;
    std::vector<std::uint16_t> got(count), want(count), whole(count);
    cudaMemcpy(got.data(), dst, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(want.data(), src, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    {
        // The shifted plane is read row by row: the rows are one element apart, tokens `src_rows`.
        std::vector<std::uint16_t> plane(static_cast<std::size_t>(src_rows) * tokens);
        cudaMemcpy(plane.data(), src, plane.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        for (std::int64_t r = 0; r < rows; ++r) {
            for (int t = 0; t < tokens; ++t) {
                whole[static_cast<std::size_t>(r) * tokens + t] =
                    plane[static_cast<std::size_t>(dst_row + r) + static_cast<std::size_t>(t) * src_rows];
            }
        }
    }
    std::size_t miss_identity = 0, miss_shifted = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (got[i] != want[i]) { ++miss_identity; }
        if (got[i] != whole[i]) { ++miss_shifted; }
    }
    std::fprintf(stderr,
                 "gather | rows %lld dst_row %lld tokens %d grid %d | miss(head) %zu/%zu | "
                 "miss(shift) %zu/%zu\n",
                 static_cast<long long>(rows), static_cast<long long>(dst_row), tokens,
                 grid_for(rows * tokens), miss_identity, count, miss_shifted, count);
}

} // namespace ninfer::ops::detail::trellis

