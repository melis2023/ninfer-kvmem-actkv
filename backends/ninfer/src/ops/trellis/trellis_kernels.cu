#include "ops/trellis/trellis_kernels.cuh"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail::trellis {
namespace {

constexpr int kThreads = 256;

// The INT8 grid of a 16-bit code window: state * golden ratio, folded as four 6-bit limbs.
__device__ __forceinline__ int grid_value(std::uint32_t state) {
    const std::uint32_t x = state * 0x83DCD12Du;
    return static_cast<int>(((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) +
                            ((x >> 26) & 63)) - 126;
}

// The tile position of ring index p: p = 8a + j -> (row, column) in exllamav3's tensor-core order.
__device__ __forceinline__ void tile_position(int p, int& row, int& col) {
    const int a = p >> 3, jj = p & 7;
    row = (a >> 2) + 8 * ((jj >> 2) & 1);
    col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1);
}

// One 16-bit window of the tile's circular stream ending at bit (p+1)*bits; stream bit q is
// bit 31 - q % 32 of word q / 32.
__device__ __forceinline__ std::uint32_t window(const std::uint32_t* tile, int nw, int bits, int p) {
    const int total = 256 * bits;
    const int start = ((p + 1) * bits - 16 + total) % total;
    const int wi = start >> 5, o = start & 31;
    const std::uint64_t pair = (static_cast<std::uint64_t>(tile[wi]) << 32) |
                               tile[(wi + 1) % nw];
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
template <int CHUNK>
__global__ void tile_product_kernel(const float* xa, const std::uint32_t* codes,
                                    const float* s16, float* yr, int K, int N, int T,
                                    int bits, int kt_total) {
    __shared__ signed char tile[256];
    const int n0 = static_cast<int>(blockIdx.x) * 16;
    const int t0 = static_cast<int>(blockIdx.y) * CHUNK;
    const int tid = threadIdx.x;  // 0..255
    const int r = tid >> 4, lane = tid & 15;
    const int nw = 8 * bits;

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
        tile[row * 16 + col] = static_cast<signed char>(grid_value(window(words, nw, bits, tid)));
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

__global__ void pre_gather_kernel(const float* g, const float* pre, const std::int32_t* pin,
                                  __nv_bfloat16* rows, std::int64_t k, std::int64_t total) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t col = i / k;
        const std::int64_t j = i - col * k;
        const float v = pre[j] * g[i];
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

void tile_product(const float* xa, const Weight& w, float* yr, int tokens, cudaStream_t stream) {
    const int bits = bits_of(w.qtype);
    const int kt_total = w.k / 16;
    constexpr int kChunk = 64;
    const unsigned nx = static_cast<unsigned>(w.n / 16);
    const unsigned ny = static_cast<unsigned>((tokens + kChunk - 1) / kChunk);
    tile_product_kernel<kChunk><<<dim3(nx, ny), kThreads, 0, stream>>>(
        xa, static_cast<const std::uint32_t*>(w.qdata), static_cast<const float*>(w.trellis_s16),
        yr, w.k, w.n, tokens, bits, kt_total);
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

void pre_and_gather_bf16(const float* g, const Weight& w, Tensor& rows, int tokens,
                         cudaStream_t stream) {
    pre_gather_kernel<<<grid_for(static_cast<std::int64_t>(w.k) * tokens), kThreads, 0, stream>>>(
        g, static_cast<const float*>(w.trellis_pre), w.trellis_pin,
        static_cast<__nv_bfloat16*>(rows.data), w.k, static_cast<std::int64_t>(w.k) * tokens);
}

} // namespace ninfer::ops::detail::trellis

