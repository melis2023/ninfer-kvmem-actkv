#include "ninfer/ops/dspine_bridge.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_bf16.h>
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr float kF32Tiny = 1.1754943508222875e-38f;

// ninfer activations are column-major: element (row, column) of a [rows, columns]
// tensor lives at row + rows*column. Weight matrices are different -- a logical
// (out, in) matrix is bound as ne={in, out}, so its flat index stays out*in+in_idx.
// Every bridge kernel below must use cm() for activations; using the row-major
// row*columns+column offset transposes the whole residual stream and silently
// destroys the draft (this was the original 0% acceptance bug).
__device__ __forceinline__ std::int64_t cm(std::int64_t row, std::int64_t rows,
                                           std::int64_t column) {
    return row + rows * column;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warps[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) { warps[warp] = value; }
    __syncthreads();
    value = (threadIdx.x < (blockDim.x >> 5)) ? warps[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

__global__ void dspine_bridge_read_kernel(const __nv_bfloat16* hidden,
                                          const __nv_bfloat16* residual, int h, int t, float eps,
                                          __nv_bfloat16* raw, __nv_bfloat16* normalized,
                                          float* rms) {
    const int column = static_cast<int>(blockIdx.x);
    if (column >= t) { return; }
    float sum = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < h; index += blockDim.x) {
        const float base = __bfloat162float(hidden[cm(index, h, column)]);
        const float value =
            residual ? base + __bfloat162float(residual[cm(index, h, column)]) : base;
        raw[cm(index, h, column)] = __float2bfloat16(value);
        sum += value * value;
    }
    sum = block_sum(sum);
    __shared__ float shared_rms;
    if (threadIdx.x == 0) {
        shared_rms = sqrtf(fmaxf(sum / static_cast<float>(h), kF32Tiny));
        rms[column]  = shared_rms;
    }
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < h; index += blockDim.x) {
        const float value = __bfloat162float(raw[cm(index, h, column)]);
        normalized[cm(index, h, column)] = __float2bfloat16(value / (shared_rms + eps));
    }
}

// out[row, col] = sum_k weight[row, k] * x[k, col], for the t proposal columns.
//
// One warp owns one output row and strides the reduction axis, so a weight row is a
// coalesced read and is fetched once for all t columns. Every bridge projection used
// to map adjacent threads onto adjacent *rows* (`row = index % rows`) while looping
// over k, which made consecutive lanes touch addresses a full row apart: each 32-byte
// transaction served a single half, and the weight row was re-read per column. That
// turned a 512x5120 matvec -- a ~15 us read at HBM speed -- into a 500 us kernel, and
// the bridge projections together were 8.6 ms of a 14.4 ms draft round.
constexpr int kDSpineGemmWarps = 8;
// One GEMV launch covers at most this many columns: the body keeps its
// accumulators in a fixed float[16]. Columns are a tile, not a limit -- the
// launchers below walk them in groups of this size, so the bridge accepts any
// (draft tokens + 1) x concurrency up to kDSpineMaxColumns. The release runs
// batch 8 with 16 draft tokens, i.e. 128 columns.
constexpr int kDSpineGemmColumns = 16;
constexpr int kDSpineMaxColumns  = 128;

__device__ __forceinline__ void dspine_bridge_gemv_body(
    const __nv_bfloat16* weight, const __nv_bfloat16* x, int rows, int k, int t,
    int row, int lane, __nv_bfloat16* out);

__device__ __forceinline__ bool dspine_bridge_gemv_row(int rows, int& row, int& lane);

__global__ void dspine_bridge_linear_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            int n, int k, int t, __nv_bfloat16* out) {
    int row = 0, lane = 0;
    if (!dspine_bridge_gemv_row(n, row, lane)) { return; }
    dspine_bridge_gemv_body(weight, x, n, k, t, row, lane, out);
}

// q, s and v are three GEMVs over the same codes tensor with three different
// weight planes. One grid covers all 3 x rank output rows, so the bridge spends
// one launch here instead of three per draft layer. Every warp's reduce uses the
// same gemv body and the same accumulation order as the separate launches, so
// the result is bit-identical.
__global__ void dspine_bridge_qsv_kernel(const __nv_bfloat16* codes,
                                         const __nv_bfloat16* gate,
                                         const __nv_bfloat16* predecessor_gate,
                                         const __nv_bfloat16* prev, int rows, int k, int t,
                                         __nv_bfloat16* q, __nv_bfloat16* s,
                                         __nv_bfloat16* v) {
    const int global_row = static_cast<int>(blockIdx.x) * kDSpineGemmWarps +
                           (static_cast<int>(threadIdx.x) >> 5);
    const int plane = global_row / rows;
    if (plane >= 3) { return; }
    const int row  = global_row - plane * rows;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const __nv_bfloat16* weight =
        plane == 0 ? gate : (plane == 1 ? predecessor_gate : prev);
    __nv_bfloat16* out = plane == 0 ? q : (plane == 1 ? s : v);
    dspine_bridge_gemv_body(weight, codes, rows, k, t, row, lane, out);
}

// Eight bf16 values as one 16-byte load. The gemv body below used to step the reduction
// axis one bf16 at a time, so a warp instruction moved 64 B and each iteration's
// accumulate depended on its own load: a 512x5120 projection became a 160-deep chain of
// dependent 600 ns loads (152 us, ~34 GB/s) instead of a bandwidth-bound read (12 us).
// Every artifact tensor is bound at a 256-byte-aligned offset, so 16-byte access is safe
// whenever k is a multiple of 8.
__device__ __forceinline__ void dspine_bridge_load8(const __nv_bfloat16* p, float (&out)[8]) {
    const float4 raw = *reinterpret_cast<const float4*>(p);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void dspine_bridge_gemv_body(
    const __nv_bfloat16* weight, const __nv_bfloat16* x, int rows, int k, int t,
    int row, int lane, __nv_bfloat16* out) {
    float acc[16];
#pragma unroll
    for (int col = 0; col < 16; ++col) { acc[col] = 0.0f; }
    if ((k & 7) == 0) {
        for (int kk = 8 * lane; kk < k; kk += 256) {
            float w[8];
            dspine_bridge_load8(weight + static_cast<std::int64_t>(row) * k + kk, w);
            for (int col = 0; col < t; ++col) {
                float xs[8];
                dspine_bridge_load8(x + kk + static_cast<std::int64_t>(k) * col, xs);
#pragma unroll
                for (int i = 0; i < 8; ++i) { acc[col] += w[i] * xs[i]; }
            }
        }
    } else {
        for (int kk = lane; kk < k; kk += 32) {
            const float w =
                __bfloat162float(weight[static_cast<std::int64_t>(row) * k + kk]);
            for (int col = 0; col < t; ++col) {
                acc[col] += w * __bfloat162float(x[kk + static_cast<std::int64_t>(k) * col]);
            }
        }
    }
    for (int col = 0; col < t; ++col) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[col] += __shfl_xor_sync(kFullWarpMask, acc[col], offset);
        }
        if (lane == 0) {
            out[static_cast<std::int64_t>(row) + static_cast<std::int64_t>(rows) * col] =
                __float2bfloat16(acc[col]);
        }
    }
}

__device__ __forceinline__ bool dspine_bridge_gemv_row(int rows, int& row, int& lane) {
    lane = static_cast<int>(threadIdx.x) & 31;
    row  = static_cast<int>(blockIdx.x) * kDSpineGemmWarps +
          (static_cast<int>(threadIdx.x) >> 5);
    return row < rows;
}

__global__ void dspine_bridge_read_proj_kernel(const __nv_bfloat16* normalized,
                                               const __nv_bfloat16* early,
                                               const __nv_bfloat16* final, int layer,
                                               int first_exact, int c, int h, int t,
                                               __nv_bfloat16* projected) {
    int row = 0, lane = 0;
    if (!dspine_bridge_gemv_row(c, row, lane)) { return; }
    const __nv_bfloat16* weight =
        layer < first_exact && early != nullptr
            ? early + static_cast<std::int64_t>(layer) * c * h
            : final;
    dspine_bridge_gemv_body(weight, normalized, c, h, t, row, lane, projected);
}

__global__ void dspine_bridge_codes_kernel(const __nv_bfloat16* projected,
                                           const __nv_bfloat16* codebook,
                                           const std::int32_t* anchors, int c, int t,
                                           int block_size, int batch, float sqrt_c, float eps,
                                           __nv_bfloat16* codes) {
    const int column = static_cast<int>(blockIdx.x);
    if (column >= t) { return; }
    if (column % block_size == 0) {
        const int anchor = anchors[column / block_size];
        for (int index = static_cast<int>(threadIdx.x); index < c; index += blockDim.x) {
            codes[cm(index, c, column)] =
                __float2bfloat16(sqrt_c * __bfloat162float(codebook[anchor * c + index]));
        }
        return;
    }
    float sum = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < c; index += blockDim.x) {
        const float value = __bfloat162float(projected[cm(index, c, column)]);
        sum += value * value;
    }
    sum = block_sum(sum);
    __shared__ float shared_norm;
    // The reference bridge divides by (||projected|| + bridge.eps), not by
    // sqrt(||projected||^2 + eps). Adding eps under the root changes the scale
    // of near-zero projections by orders of magnitude.
    if (threadIdx.x == 0) { shared_norm = sqrtf(sum) + eps; }
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < c; index += blockDim.x) {
        const float value = __bfloat162float(projected[cm(index, c, column)]);
        codes[cm(index, c, column)] = __float2bfloat16(sqrt_c * value / shared_norm);
    }
}

__global__ void dspine_bridge_message_kernel(const __nv_bfloat16* q, const __nv_bfloat16* s,
                                             const __nv_bfloat16* v,
                                             const __nv_bfloat16* gate_bias, int rank, int t,
                                             int block_size, int batch,
                                             __nv_bfloat16* message) {
    const int column = static_cast<int>(blockIdx.x);
    if (column >= t) { return; }
    if (column % block_size == 0) {
        for (int index = static_cast<int>(threadIdx.x); index < rank; index += blockDim.x) {
            message[cm(index, rank, column)] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int previous = column - 1;
    for (int index = static_cast<int>(threadIdx.x); index < rank; index += blockDim.x) {
        const float qv = __bfloat162float(q[cm(index, rank, column)]);
        const float sv = __bfloat162float(s[cm(index, rank, previous)]);
        const float vv = __bfloat162float(v[cm(index, rank, previous)]);
        const float g =
            1.0f / (1.0f + expf(-(qv + sv + __bfloat162float(gate_bias[index]))));
        message[cm(index, rank, column)] = __float2bfloat16(g * vv);
    }
}

__global__ void dspine_bridge_write_kernel(const __nv_bfloat16* message,
                                           const __nv_bfloat16* write, int layer, int h, int r,
                                           int t, __nv_bfloat16* delta) {
    int row = 0, lane = 0;
    if (!dspine_bridge_gemv_row(h, row, lane)) { return; }
    const __nv_bfloat16* weight = write + static_cast<std::int64_t>(layer) * h * r;
    dspine_bridge_gemv_body(weight, message, h, r, t, row, lane, delta);
}

__global__ void dspine_bridge_finish_kernel(const __nv_bfloat16* raw,
                                            const __nv_bfloat16* delta, const float* rms,
                                            const __nv_bfloat16* next_norm, int h, int t,
                                            int block_size, int batch, float eps,
                                            __nv_bfloat16* out, __nv_bfloat16* normed) {
    const int column = static_cast<int>(blockIdx.x);
    if (column >= t) { return; }
    const bool anchor = column % block_size == 0;
    const float scale  = rms[column];
    float sum          = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < h; index += blockDim.x) {
        float value = __bfloat162float(raw[cm(index, h, column)]);
        if (!anchor) { value += scale * __bfloat162float(delta[cm(index, h, column)]); }
        out[cm(index, h, column)] = __float2bfloat16(value);
        sum += value * value;
    }
    sum = block_sum(sum);
    __shared__ float shared_inv;
    if (threadIdx.x == 0) {
        shared_inv = rsqrtf(sum / static_cast<float>(h) + eps);
    }
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < h; index += blockDim.x) {
        const float value = __bfloat162float(out[cm(index, h, column)]) * shared_inv *
                            __bfloat162float(next_norm[index]);
        normed[cm(index, h, column)] = __float2bfloat16(value);
    }
}

// Geometry comes from the caller so another Qwen drafter needs no rebuild. head_dim
// must be a power of two because the shared-memory reduction halves it each step.
//
// The context scan is split across kDSpineAttentionSplits warps per (head, row):
// one warp alone serialises 4-8K ring positions and leaves 36 SMs with ~1 warp
// each, so a round paid 17 ms of pure latency at 4K. Each warp now scans its own
// slice and the block merges the partial online-softmax states in shared memory.
constexpr int kDSpineAttentionSplits = 8;

__global__ void dspine_causal_attention_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const std::int32_t* positions, const std::int32_t* valid_columns,
    const std::int32_t* floors, const std::int32_t* lanes,
    const __nv_bfloat16* context_k, const __nv_bfloat16* context_v, int capacity,
    int padded, int width, int batch, int head_dim, int query_heads, int kv_heads,
    int window, float scale, __nv_bfloat16* out) {
    const int block = static_cast<int>(blockIdx.x);
    const int w     = block % width;
    const int rest  = block / width;
    const int b     = rest % batch;
    const int head  = rest / batch;
    // One warp covers one slice of the context; each lane holds head_dim/32 elements
    // of the query. A context position therefore costs one shuffle reduction with no
    // __syncthreads at all.
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int dims    = head_dim >> 5;
    const int kv_head = head / (query_heads / kv_heads);
    // All proposal rows share the same context frontier: positions[0, b]. The
    // ring holds context only; proposal K/V is the block below. KVMem reselection
    // invalidates everything below draft_floors[b], so mask those ring slots too.
    // The reference drafter is trained with an 8K sliding window, so the window --
    // not the ring capacity -- bounds the scan. Scanning the full 52,864-slot ring
    // fed the drafter positions it never saw and multiplied the cost.
    const int frontier = positions[width * b];
    const int floor    = floors != nullptr ? floors[b] : 0;
    const int span     = window > 0 && window < capacity ? window : capacity;
    const int begin    = max(floor, frontier - span);
    const int valid    = valid_columns[b];
    const std::int64_t q_base =
        static_cast<std::int64_t>(head_dim) *
            (head + query_heads * (w + width * b)) + lane;

    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        qv[i] = i < dims ? __bfloat162float(q[q_base + 32 * i]) : 0.0f;
    }
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0f; }
    float m = -INFINITY;
    float l = 0.0f;

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[b]) * head_dim * padded * kv_heads;
    const auto consume = [&](const __nv_bfloat16* key, const __nv_bfloat16* value,
                             std::int64_t index) {
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < dims) { score += qv[i] * __bfloat162float(key[index + 32 * i]); }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_xor_sync(kFullWarpMask, score, offset);
        }
        score *= scale;
        const float m_new     = fmaxf(m, score);
        const float e         = expf(score - m_new);
        const float scale_old = expf(m - m_new);
        l = l * scale_old + e;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < dims) {
                acc[i] = acc[i] * scale_old + e * __bfloat162float(value[index + 32 * i]);
            }
        }
        m = m_new;
    };
    const int total = frontier - begin;
    const int chunk = (total + kDSpineAttentionSplits - 1) / kDSpineAttentionSplits;
    const int stop  = min(frontier, begin + warp * chunk + chunk);
    for (int p = begin + warp * chunk; p < stop; ++p) {
        const int slot = p % capacity;
        const std::int64_t base = lane_base +
            static_cast<std::int64_t>(head_dim) * slot +
            static_cast<std::int64_t>(head_dim) * padded * kv_head;
        consume(context_k, context_v, base + lane);
    }
    // The proposal block is consumed exactly once, by the first warp.
    if (warp == 0) {
        for (int j = 0; j <= w && j < valid; ++j) {
            const std::int64_t base =
                static_cast<std::int64_t>(head_dim) *
                (kv_head + kv_heads * (j + width * b));
            consume(k, v, base + lane);
        }
    }

    // Merge the per-warp partial states. Softmax is associative, so rescaling every
    // partial by exp(m_s - M) reproduces the sequential result up to rounding.
    __shared__ float s_acc[kDSpineAttentionSplits * 256];
    __shared__ float s_m[kDSpineAttentionSplits];
    __shared__ float s_l[kDSpineAttentionSplits];
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i < dims) { s_acc[warp * 256 + lane + 32 * i] = acc[i]; }
    }
    __syncthreads();

    float peak = -INFINITY;
#pragma unroll
    for (int s = 0; s < kDSpineAttentionSplits; ++s) { peak = fmaxf(peak, s_m[s]); }
    float weight[kDSpineAttentionSplits];
    float denom = 0.0f;
#pragma unroll
    for (int s = 0; s < kDSpineAttentionSplits; ++s) {
        weight[s] = s_m[s] > -INFINITY ? expf(s_m[s] - peak) : 0.0f;
        denom += s_l[s] * weight[s];
    }
    const float inverse = denom > 0.0f ? 1.0f / denom : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i < dims) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < kDSpineAttentionSplits; ++s) {
                sum += s_acc[s * 256 + lane + 32 * i] * weight[s];
            }
            out[q_base + 32 * i] = __float2bfloat16(sum * inverse);
        }
    }
}

constexpr int kDSpineCandidates = kSparseSpeculativeCandidates;
constexpr int kDSpineMaximumRows   = 15;

__global__ void dspine_write_scaled_kernel(const __nv_bfloat16* write,
                                           const __nv_bfloat16* norm, int h, int r,
                                           __nv_bfloat16* out) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (index >= h * r) { return; }
    const int hh = index / r;
    out[index]   = __float2bfloat16(__bfloat162float(write[index]) *
                                  __bfloat162float(norm[hh]));
}

__global__ void dspine_refine_kernel(
    const std::int32_t* cand_ids, const float* rms, const __nv_bfloat16* raw,
    const __nv_bfloat16* q, const std::int32_t* anchors, const __nv_bfloat16* codebook,
    const __nv_bfloat16* predecessor_gate, const __nv_bfloat16* prev,
    const __nv_bfloat16* gate_bias, const __nv_bfloat16* write_scaled,
    const __nv_bfloat16* head_rows, const __nv_bfloat16* norm, int width, int batch,
    int code_size, int rank, int hidden, float sqrt_c, float eps, __nv_bfloat16* message,
    float* delta, float* scratch, float* table) {
    // One block per (row, sender) column. Refining a column is a 512-wide gate/value
    // reduction, a 512x512 write, and a 16-way head score -- all independent across
    // columns. The previous single-block launch serialised the whole 630M-MAC
    // refinement on one SM (334 ms per round, 80% of decode time); spreading the
    // columns over blocks lets every SM work in parallel. Accumulation order per
    // column is unchanged, so the published table is bit-identical.
    const int column  = static_cast<int>(blockIdx.x);
    const int b       = static_cast<int>(blockIdx.y);
    const int rows    = width - 1;
    const int columns = rows * kDSpineCandidates;
    const int threads = static_cast<int>(blockDim.x);
    const int tid     = static_cast<int>(threadIdx.x);
    const int R       = rank;
    const int C       = code_size;
    const int H       = hidden;
    // scratch/message/delta hold batch in their trailing length, so index them by
    // the batch-global column instead of the per-batch one.
    const int gcolumn = b * columns + column;

    const int i   = column / kDSpineCandidates;
    const int j   = column - i * kDSpineCandidates;
    const int sid = i == 0 ? anchors[b]
                           : cand_ids[(b * rows + i - 1) * kDSpineCandidates + j];
    const int q_column = b * width + i + 1;

    // Reduction stages below are warp-per-output-row with the lane striding the
    // reduction axis. Mapping threads straight onto the output axis made consecutive
    // lanes read addresses a whole row apart, so each 32-byte transaction served one
    // half; a 512-wide gate/value pair that should be a ~10 us read cost ~2 ms.
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int warps   = threads >> 5;

    for (int r = warp_id; r < R; r += warps) {
        float gate  = 0.0f;
        float value = 0.0f;
        for (int c = lane_id; c < C; c += 32) {
            const float e = sqrt_c * __bfloat162float(codebook[sid * C + c]);
            gate  += e * __bfloat162float(predecessor_gate[r * C + c]);
            value += e * __bfloat162float(prev[r * C + c]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate  += __shfl_xor_sync(kFullWarpMask, gate, offset);
            value += __shfl_xor_sync(kFullWarpMask, value, offset);
        }
        if (lane_id == 0) {
            scratch[r + 2 * R * gcolumn]     = gate;
            scratch[R + r + 2 * R * gcolumn] = value;
        }
    }
    __syncthreads();

    for (int r = tid; r < R; r += threads) {
        const float qv = __bfloat162float(q[r + R * q_column]);
        const float g  = 1.0f / (1.0f + expf(-(qv + scratch[r + 2 * R * gcolumn] +
                                              __bfloat162float(gate_bias[r]))));
        message[r + R * gcolumn] = __float2bfloat16(g * scratch[R + r + 2 * R * gcolumn]);
    }
    __syncthreads();

    for (int h = warp_id; h < H; h += warps) {
        float sum = 0.0f;
        for (int r = lane_id; r < R; r += 32) {
            sum += __bfloat162float(write_scaled[h * R + r]) *
                   __bfloat162float(message[r + R * gcolumn]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(kFullWarpMask, sum, offset);
        }
        if (lane_id == 0) { delta[h + H * gcolumn] = sum; }
    }
    __syncthreads();

    const float rm     = rms[q_column];
    const __nv_bfloat16* base = raw + static_cast<std::int64_t>(H) * q_column;
    const float* delta_column = delta + static_cast<std::int64_t>(H) * gcolumn;
    float score[kDSpineCandidates];
#pragma unroll
    for (int jp = 0; jp < kDSpineCandidates; ++jp) { score[jp] = 0.0f; }
    // llama.cpp's lattice normalises the refined residual before the head read:
    //   refined = rms_norm(delta * rms + raw) * output_norm
    // combined is norm_w * (raw + rms*write@msg), so divide by the state RMS.
    // The head read is H*16 multiply-adds; leaving it to a single thread parked 255
    // threads for the whole stage, so it is split across the block and reduced.
    float square_sum = 0.0f;
    for (int h = tid; h < H; h += threads) {
        const float norm_h = __bfloat162float(norm[h]);
        // delta already carries norm_w (write_scaled), so only the raw term is scaled.
        const float combined = norm_h * __bfloat162float(base[h]) + rm * delta_column[h];
        const float state    = norm_h != 0.0f ? combined / norm_h : 0.0f;
        square_sum += state * state;
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) {
            score[jp] += combined * __bfloat162float(
                head_rows[static_cast<std::int64_t>(H) *
                              (b * columns + i * kDSpineCandidates + jp) + h]);
        }
    }
#pragma unroll
    for (int jp = 0; jp < kDSpineCandidates; ++jp) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            score[jp] += __shfl_xor_sync(kFullWarpMask, score[jp], offset);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        square_sum += __shfl_xor_sync(kFullWarpMask, square_sum, offset);
    }
    __shared__ float s_score[8][kDSpineCandidates];
    __shared__ float s_square[8];
    if (lane_id == 0) {
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) { s_score[warp_id][jp] = score[jp]; }
        s_square[warp_id] = square_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total[kDSpineCandidates];
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) { total[jp] = 0.0f; }
        float sum_square = 0.0f;
        for (int w = 0; w < warps; ++w) {
#pragma unroll
            for (int jp = 0; jp < kDSpineCandidates; ++jp) { total[jp] += s_score[w][jp]; }
            sum_square += s_square[w];
        }
        const float inverse_rms = rsqrtf(sum_square / static_cast<float>(H) + eps);
        // Publish the whole predecessor-major transition table [row, sender, candidate]
        // so the speculative decoder can walk it greedily or under the request's
        // sampling temperature, exactly like llama.cpp's DSpine lattice.
        float* row = table + (static_cast<std::int64_t>(b) * rows + i) *
                                 kDSpineCandidates * kDSpineCandidates +
                             j * kDSpineCandidates;
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) {
            row[jp] = total[jp] * inverse_rms;
        }
    }
}

// One warp owns one batch row and walks the transition table from the anchor's
// sender rank. Temperature <= 0 is an exact greedy argmax (matching llama.cpp's
// default chain); a positive temperature samples the same softmax the verifier
// will use for rejection sampling.
//
// Measured: forcing the chain greedy at positive temperature (the release's
// `_serial_chain` behaviour, with q still carrying the row softmax) dropped
// acceptance from 24.6% to 21.5% and decode from 48.1 to 44.5 at 2,158 tokens and
// temperature 1.0. A chain drawn from q keeps p/q honest -- min(1, p/q) is exact
// when the draft really came from q -- while a greedy chain pairs an argmax draft
// with a q that overstates it. The release's two paths (`serial_chain` for its
// default temperature, `_sample_capped_chain` for sampled serving) are not
// interchangeable, so this one stays as the sampled form.
__device__ __forceinline__ int dspine_draw_rank(float edge, const SamplingConfig& config,
                                                unsigned long long seed, int position,
                                                float* q) {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const float temperature = config.temperature;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0f) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kDSpineCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kDSpineCandidates) q[lane] = lane == selected ? 1.0f : 0.0f;
        return selected;
    }
    const float weight =
        lane < kDSpineCandidates ? __expf((edge - maximum) / temperature) : 0.0f;
    float probability = weight / warp_sum(weight);
    // The reference samples each chain row under the request's filters
    // (`filter_probs`: joint top-k / top-p + min-p, then renormalise), and the
    // published q has to carry the filtered probabilities because that is the
    // distribution the verifier divides by. Only 16 candidates reach this row, so
    // the rank and the prefix mass come from two unrolled warp broadcasts.
    if (config.min_p > 0.0f || config.top_p < 1.0f ||
        (config.top_k > 0 && config.top_k < kDSpineCandidates)) {
        const float peak = warp_max(probability);
        int rank         = 0;
        float prefix     = 0.0f;
        const int limit  = config.top_k > 0 ? config.top_k : kDSpineCandidates;
        const bool valid = lane < kDSpineCandidates;
#pragma unroll
        for (int other_lane = 0; other_lane < kDSpineCandidates; ++other_lane) {
            const float other =
                __shfl_sync(kFullWarpMask, probability, other_lane);
            const bool ahead = other > probability ||
                               (other == probability && other_lane < lane);
            rank += ahead ? 1 : 0;
            prefix += ahead ? other : 0.0f;
        }
        const bool kept =
            valid && rank < limit &&
            (config.min_p <= 0.0f || probability >= config.min_p * peak) &&
            (config.top_p >= 1.0f || rank == 0 || prefix < config.top_p);
        probability = kept ? probability : 0.0f;
        const float total = warp_sum(probability);
        probability = total > 0.0f ? probability / total : probability;
    }
    if (lane < kDSpineCandidates) q[lane] = probability;
    float uniform = lane == 0
                        ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U)
                        : 0.0f;
    uniform = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kDSpineCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits =
        __ballot_sync(kFullWarpMask, lane < kDSpineCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kDSpineCandidates - 1;
}

__global__ __launch_bounds__(32) void dspine_lattice_walk_kernel(
    const std::int32_t* cand_ids, const float* table, const std::int32_t* positions,
    const SamplingConfig* configs, int rows, int plain, std::int32_t* drafts, float* q) {
    const int lane = static_cast<int>(threadIdx.x);
    const int b    = static_cast<int>(blockIdx.x);
    const float temperature = configs != nullptr ? configs[b].temperature : 0.0f;
    const unsigned long long seed = configs != nullptr ? configs[b].seed : 0ULL;
    const int position = positions != nullptr ? positions[b] : 0;
    // Diagnostic: NINFER_DSPINE_PLAIN=1 emits the parallel pass's own top-1 chain and
    // skips the transition table. It isolates what the serial last-write refinement is
    // worth, because everything else in the round is identical.
    if (plain != 0) {
        for (int i = 0; i < rows; ++i) {
            const std::int64_t column = static_cast<std::int64_t>(b) * rows + i;
            for (int lane_id = lane; lane_id < kDSpineCandidates; lane_id += 32) {
                q[column * kDSpineCandidates + lane_id] =
                    lane_id == 0 ? 1.0f : 0.0f;
            }
            if (lane == 0) { drafts[column] = cand_ids[column * kDSpineCandidates]; }
        }
        return;
    }
    int sender = 0;
    for (int i = 0; i < rows; ++i) {
        const std::int64_t column = static_cast<std::int64_t>(b) * rows + i;
        const float edge =
            lane < kDSpineCandidates
                ? table[(column * kDSpineCandidates + sender) * kDSpineCandidates + lane]
                : -CUDART_INF_F;
        const int selected =
            dspine_draw_rank(edge, configs != nullptr ? configs[b] : SamplingConfig{},
                             seed, position + i, q + column * kDSpineCandidates);
        sender = selected;
        if (lane == 0) drafts[column] = cand_ids[column * kDSpineCandidates + selected];
    }
}

// Each thread keeps its own descending top-kCandidates while the vocabulary is streamed
// once; the block then merges the 256 per-thread lists through the same candidate-rank
// scan. The previous version rescanned the whole vocabulary once per rank, so a 248k
// vocabulary cost 16 passes over 2.5 MB per column (40 MB per round, 0.54 ms) and the
// selection had only one block per column to hide that latency.
//
// Keeping kCandidates per thread is what makes the single pass exact: the union of the
// per-thread lists contains the global top kCandidates no matter how the winners cluster.
// The insertion bubbles the new value down with constant indices so both arrays stay in
// registers instead of spilling to local memory.
__global__ void dspine_top16_kernel(__nv_bfloat16* logits, int vocab, int valid_rows,
                                    std::int32_t* ids) {
    constexpr int kCandidates = kDSpineCandidates;
    const int column  = static_cast<int>(blockIdx.x);
    const int tid     = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    __nv_bfloat16* values = logits + static_cast<std::int64_t>(vocab) * column;
    __shared__ float s_value[256][kCandidates];
    __shared__ int s_index[256][kCandidates];
    __shared__ float r_value[256];
    __shared__ int r_index[256];

    float best[kCandidates];
    int best_index[kCandidates];
#pragma unroll
    for (int i = 0; i < kCandidates; ++i) {
        best[i]       = -INFINITY;
        best_index[i] = 0;
    }
    for (int v = tid; v < valid_rows; v += threads) {
        float value = __bfloat162float(values[v]);
        int index   = v;
        if (value <= best[kCandidates - 1]) { continue; }
#pragma unroll
        for (int i = kCandidates - 1; i >= 0; --i) {
            if (value > best[i]) {
                const float displaced = best[i];
                const int displaced_index = best_index[i];
                best[i]       = value;
                best_index[i] = index;
                value         = displaced;
                index         = displaced_index;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kCandidates; ++i) {
        s_value[tid][i] = best[i];
        s_index[tid][i] = best_index[i];
    }
    __syncthreads();
    for (int rank = 0; rank < kCandidates; ++rank) {
        float mine     = -INFINITY;
        int mine_index = 0;
#pragma unroll
        for (int i = 0; i < kCandidates; ++i) {
            if (s_value[tid][i] > mine) {
                mine       = s_value[tid][i];
                mine_index = s_index[tid][i];
            }
        }
        r_value[tid] = mine;
        r_index[tid] = mine_index;
        __syncthreads();
        for (int stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other     = r_value[tid + stride];
                const int other_index = r_index[tid + stride];
                if (other > r_value[tid] ||
                    (other == r_value[tid] && other_index < r_index[tid])) {
                    r_value[tid] = other;
                    r_index[tid] = other_index;
                }
            }
            __syncthreads();
        }
        const int winner = r_index[0];
        if (tid == 0) {
            ids[rank + kDSpineCandidates * column] = winner;
            values[winner] = __float2bfloat16(-INFINITY);
        }
        // Only the owning thread touches its own list, so no barrier is needed between the
        // retire and the next rank's scan.
#pragma unroll
        for (int i = 0; i < kCandidates; ++i) {
            if (s_index[tid][i] == winner) { s_value[tid][i] = -INFINITY; }
        }
    }
}

void require_bf16(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("dspine_bridge: invalid ") + name);
    }
}

} // namespace

void dspine_bridge_read(const Tensor& hidden, const Tensor* residual, float eps, Tensor& raw,
                        Tensor& normalized, Tensor& rms, cudaStream_t stream) {
    require_bf16(hidden, "hidden");
    if (residual != nullptr) { require_bf16(*residual, "residual"); }
    require_bf16(raw, "raw");
    require_bf16(normalized, "normalized");
    if (rms.dtype != DType::FP32) { throw std::invalid_argument("dspine_bridge: invalid rms"); }
    const int h = hidden.ne[0];
    const int t = hidden.ne[1];
    dspine_bridge_read_kernel<<<t, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        residual ? static_cast<const __nv_bfloat16*>(residual->data) : nullptr, h, t, eps,
        static_cast<__nv_bfloat16*>(raw.data),
        static_cast<__nv_bfloat16*>(normalized.data), static_cast<float*>(rms.data));
}

void dspine_bridge_linear(const Tensor& x, const Tensor& weight, Tensor& out,
                          cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(weight, "weight");
    require_bf16(out, "out");
    const int k = x.ne[0];
    const int t = x.ne[1];
    const int n = out.ne[0];
    if (out.ne[1] != t || weight.ne[0] != k || weight.ne[1] != n || t < 1 ||
        t > kDSpineMaxColumns) {
        throw std::invalid_argument("dspine_bridge: invalid linear tensors");
    }
    const auto* x_rows   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* w_rows   = static_cast<const __nv_bfloat16*>(weight.data);
    auto* out_rows       = static_cast<__nv_bfloat16*>(out.data);
    const unsigned grid  = static_cast<unsigned>((n + kDSpineGemmWarps - 1) / kDSpineGemmWarps);
    for (int col = 0; col < t; col += kDSpineGemmColumns) {
        const int width = std::min(kDSpineGemmColumns, t - col);
        dspine_bridge_linear_kernel<<<grid, 32 * kDSpineGemmWarps, 0, stream>>>(
            x_rows + static_cast<std::int64_t>(k) * col, w_rows, n, k, width,
            out_rows + static_cast<std::int64_t>(n) * col);
    }
}

void dspine_bridge_qsv(const Tensor& codes, const Tensor& gate, const Tensor& predecessor_gate,
                       const Tensor& prev, Tensor& q, Tensor& s, Tensor& v,
                       cudaStream_t stream) {
    require_bf16(codes, "codes");
    require_bf16(gate, "gate_weight");
    require_bf16(predecessor_gate, "predecessor_gate_weight");
    require_bf16(prev, "prev_weight");
    require_bf16(q, "q");
    require_bf16(s, "s");
    require_bf16(v, "v");
    const int k = codes.ne[0];
    const int t = codes.ne[1];
    const int r = q.ne[0];
    if (q.ne[1] != t || s.ne[0] != r || s.ne[1] != t || v.ne[0] != r || v.ne[1] != t ||
        gate.ne[0] != k || gate.ne[1] != r || predecessor_gate.ne[0] != k ||
        predecessor_gate.ne[1] != r || prev.ne[0] != k || prev.ne[1] != r || t < 1 ||
        t > kDSpineMaxColumns) {
        throw std::invalid_argument("dspine_bridge: invalid qsv tensors");
    }
    const auto* x_rows = static_cast<const __nv_bfloat16*>(codes.data);
    const auto* wq     = static_cast<const __nv_bfloat16*>(gate.data);
    const auto* ws     = static_cast<const __nv_bfloat16*>(predecessor_gate.data);
    const auto* wv     = static_cast<const __nv_bfloat16*>(prev.data);
    auto* q_rows       = static_cast<__nv_bfloat16*>(q.data);
    auto* s_rows       = static_cast<__nv_bfloat16*>(s.data);
    auto* v_rows       = static_cast<__nv_bfloat16*>(v.data);
    const unsigned grid = static_cast<unsigned>((3 * r + kDSpineGemmWarps - 1) / kDSpineGemmWarps);
    for (int col = 0; col < t; col += kDSpineGemmColumns) {
        const int width = std::min(kDSpineGemmColumns, t - col);
        dspine_bridge_qsv_kernel<<<grid, 32 * kDSpineGemmWarps, 0, stream>>>(
            x_rows + static_cast<std::int64_t>(k) * col, wq, ws, wv, r, k, width,
            q_rows + static_cast<std::int64_t>(r) * col,
            s_rows + static_cast<std::int64_t>(r) * col,
            v_rows + static_cast<std::int64_t>(r) * col);
    }
}

void dspine_bridge_read_proj(const Tensor& normalized, const Tensor& early,
                             const Tensor& final, std::int32_t layer, Tensor& projected,
                             cudaStream_t stream) {
    require_bf16(normalized, "normalized");
    require_bf16(early, "early_read_weights");
    require_bf16(final, "read_weight");
    require_bf16(projected, "projected");
    const int h = normalized.ne[0];
    const int t = normalized.ne[1];
    const int c = projected.ne[0];
    const int first_exact = early.ne[2];
    if (projected.ne[1] != t || early.ne[0] != h || early.ne[1] != c || first_exact < 1 ||
        layer > first_exact || final.ne[0] != h || final.ne[1] != c || t < 1 ||
        t > kDSpineMaxColumns) {
        throw std::invalid_argument("dspine_bridge: invalid read projection");
    }
    const auto* x_rows  = static_cast<const __nv_bfloat16*>(normalized.data);
    const auto* early_p = static_cast<const __nv_bfloat16*>(early.data);
    const auto* final_p = static_cast<const __nv_bfloat16*>(final.data);
    auto* out_rows      = static_cast<__nv_bfloat16*>(projected.data);
    const unsigned grid = static_cast<unsigned>((c + kDSpineGemmWarps - 1) / kDSpineGemmWarps);
    for (int col = 0; col < t; col += kDSpineGemmColumns) {
        const int width = std::min(kDSpineGemmColumns, t - col);
        dspine_bridge_read_proj_kernel<<<grid, 32 * kDSpineGemmWarps, 0, stream>>>(
            x_rows + static_cast<std::int64_t>(h) * col,
            layer < first_exact ? early_p : nullptr, final_p, layer, first_exact, c, h, width,
            out_rows + static_cast<std::int64_t>(c) * col);
    }
}

void dspine_bridge_codes(const Tensor& projected, const Tensor& codebook, const Tensor& anchors,
                         std::int32_t block_size, std::int32_t batch, float sqrt_c, float eps,
                         Tensor& codes, cudaStream_t stream) {
    require_bf16(projected, "projected");
    require_bf16(codebook, "codebook");
    require_bf16(codes, "codes");
    if (anchors.dtype != DType::I32 || batch <= 0 || block_size < batch) {
        throw std::invalid_argument("dspine_bridge: invalid anchors");
    }
    const int c = projected.ne[0];
    const int t = projected.ne[1];
    dspine_bridge_codes_kernel<<<t, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(codebook.data),
        static_cast<const std::int32_t*>(anchors.data), c, t, block_size, batch, sqrt_c, eps,
        static_cast<__nv_bfloat16*>(codes.data));
}

void dspine_bridge_message(const Tensor& q, const Tensor& s, const Tensor& v,
                           const Tensor& gate_bias, std::int32_t block_size, std::int32_t batch,
                           Tensor& message, cudaStream_t stream) {
    require_bf16(q, "q");
    require_bf16(s, "s");
    require_bf16(v, "v");
    require_bf16(gate_bias, "gate_bias");
    require_bf16(message, "message");
    const int rank = gate_bias.ne[0];
    const int t    = message.ne[1];
    if (q.ne[0] != rank || q.ne[1] != t || s.ne[0] != rank || s.ne[1] != t ||
        v.ne[0] != rank || v.ne[1] != t || batch <= 0 || block_size < batch) {
        throw std::invalid_argument("dspine_bridge: invalid message tensors");
    }
    dspine_bridge_message_kernel<<<t, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(s.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const __nv_bfloat16*>(gate_bias.data), rank, t, block_size, batch,
        static_cast<__nv_bfloat16*>(message.data));
}

void dspine_bridge_write(const Tensor& message, const Tensor& write, std::int32_t layer,
                         Tensor& delta, cudaStream_t stream) {
    require_bf16(message, "message");
    require_bf16(write, "feature_write_weights");
    require_bf16(delta, "delta");
    const int h = delta.ne[0];
    const int t = delta.ne[1];
    const int r = message.ne[0];
    if (message.ne[1] != t || write.ne[0] != r || write.ne[1] != h || write.ne[2] < layer + 1 ||
        t < 1 || t > kDSpineMaxColumns) {
        throw std::invalid_argument("dspine_bridge: invalid write tensors");
    }
    const auto* m_rows  = static_cast<const __nv_bfloat16*>(message.data);
    const auto* w_rows  = static_cast<const __nv_bfloat16*>(write.data);
    auto* out_rows      = static_cast<__nv_bfloat16*>(delta.data);
    const unsigned grid = static_cast<unsigned>((h + kDSpineGemmWarps - 1) / kDSpineGemmWarps);
    for (int col = 0; col < t; col += kDSpineGemmColumns) {
        const int width = std::min(kDSpineGemmColumns, t - col);
        dspine_bridge_write_kernel<<<grid, 32 * kDSpineGemmWarps, 0, stream>>>(
            m_rows + static_cast<std::int64_t>(r) * col, w_rows, layer, h, r, width,
            out_rows + static_cast<std::int64_t>(h) * col);
    }
}

void dspine_bridge_finish(const Tensor& raw, const Tensor& delta, const Tensor& rms,
                          const Tensor& next_norm, std::int32_t block_size, std::int32_t batch,
                          float eps, Tensor& out, Tensor& normed, cudaStream_t stream) {
    require_bf16(raw, "raw");
    require_bf16(delta, "delta");
    require_bf16(next_norm, "next_norm");
    require_bf16(out, "out");
    require_bf16(normed, "normed");
    if (rms.dtype != DType::FP32 || batch <= 0 || block_size < batch) {
        throw std::invalid_argument("dspine_bridge: invalid finish tensors");
    }
    const int h = raw.ne[0];
    const int t = raw.ne[1];
    dspine_bridge_finish_kernel<<<t, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw.data),
        static_cast<const __nv_bfloat16*>(delta.data), static_cast<const float*>(rms.data),
        static_cast<const __nv_bfloat16*>(next_norm.data), h, t, block_size, batch, eps,
        static_cast<__nv_bfloat16*>(out.data),
        static_cast<__nv_bfloat16*>(normed.data));
}

// One block per (head, batch): every proposal row lives in the same block, the context
// slice is staged through shared memory once per tile, and each row's share of the tile
// is scanned by its own warp.
//
// Two earlier layouts were measured against this one. The per-row split-K kernel put
// each proposal row in its own block, so the whole context slice was re-read once per
// row and per head (~960 MB per round at 4K). Co-locating the rows without staging cut
// that to one read per block but still re-read the slice once per row inside the block,
// because each row's warps walked the full position range: 32 blocks x 6 rows x 2.1 MB =
// 405 MB per layer per round, which is 943 us of a 56 ms round and the single largest
// DSpine cost. Staging a tile in shared memory makes the slice cost 32 x 2.1 MB = 67 MB
// for the whole block, ~6x less traffic, while the 24 warps keep the parallelism that a
// one-warp-per-row layout lacked.
//
// Partial online-softmax states are merged in shared memory with the lane dimension
// preserved, because each lane owns a different head_dim slice.
//
// Restricted to head_dim == 128 and Rows <= 8; the caller falls back to the general
// per-row kernel otherwise.
constexpr int kDSpineRowsMax       = 8;
constexpr int kDSpineRowDims       = 4;

constexpr int kDSpineTilePositions = 48;

// One block per head is all the parallelism a 32-head draft has, and a 32-block grid on a
// 30-SM part always costs two waves. Whatever is left lives in occupancy: the warps that
// scan the context are Rows * RowWarps, so the split is chosen to fill the 1024-thread
// budget instead of fixing it at four per row (24 warps, which measured 355 us per call
// against a 155 us traffic floor).
template <int Rows>
struct DSpineRowSplit {
    static constexpr int warps = (32 / Rows) < 1 ? 1 : (32 / Rows);
};

template <int Rows, int RowWarps>
__global__ void dspine_causal_attention_rows_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const std::int32_t* positions, const std::int32_t* valid_columns,
    const std::int32_t* floors, const std::int32_t* lanes,
    const __nv_bfloat16* context_k, const __nv_bfloat16* context_v, int capacity,
    int padded, int width, int batch, int head_dim, int query_heads, int kv_heads,
    int window, float scale, __nv_bfloat16* out) {
    const int head  = static_cast<int>(blockIdx.x);
    const int b     = static_cast<int>(blockIdx.y);
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int warps = Rows * RowWarps;
    const int row   = warp / RowWarps;
    const int sub   = warp - row * RowWarps;
    const int kv_head = head / (query_heads / kv_heads);

    const int frontier = positions[width * b];
    const int floor    = floors != nullptr ? floors[b] : 0;
    const int span     = window > 0 && window < capacity ? window : capacity;
    const int begin    = max(floor, frontier - span);
    const int valid    = valid_columns[b];
    // q/out are [head_dim, query_heads, width, batch]: a proposal row advances by
    // head_dim * query_heads.
    const std::int64_t row_base =
        static_cast<std::int64_t>(head_dim) *
            (head + static_cast<std::int64_t>(query_heads) * width * b) +
        static_cast<std::int64_t>(head_dim) * query_heads * row + lane;

    // Tile layout is [dim group][position][lane]: index i selects a group of 32 contiguous
    // head_dim values of one position, so a staging step is one coalesced 64-byte read and
    // the consuming warps read consecutive lanes out of consecutive banks.
    __shared__ __align__(16) __nv_bfloat16 s_k[kDSpineRowDims * kDSpineTilePositions * 32];
    __shared__ __align__(16) __nv_bfloat16 s_v[kDSpineRowDims * kDSpineTilePositions * 32];
    constexpr int kTileElements = kDSpineRowDims * kDSpineTilePositions * 32;
    // Staged 16 bytes per thread: one float4 covers eight contiguous head_dim values, so a
    // warp moves 512 B per instruction. Element-at-a-time staging moved 64 B per warp
    // instruction and left the tiles latency bound at 664 us per call.
    constexpr int kStageVector   = 8;
    constexpr int kStageVectors  = kTileElements / kStageVector;

    float qv[kDSpineRowDims];
    float acc[kDSpineRowDims];
#pragma unroll
    for (int i = 0; i < kDSpineRowDims; ++i) {
        qv[i]  = __bfloat162float(q[row_base + 32 * i]);
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    const int threads = warps * 32;
    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[b]) * head_dim * padded * kv_heads;
    const std::int64_t head_stride =
        static_cast<std::int64_t>(head_dim) * padded * kv_head;
    for (int tile = begin; tile < frontier; tile += kDSpineTilePositions) {
        const int count = min(kDSpineTilePositions, frontier - tile);
        for (int v = static_cast<int>(threadIdx.x); v < kStageVectors; v += threads) {
            const int i = v * kStageVector;
            const int p = (i >> 5) % kDSpineTilePositions;
            if (p >= count) { continue; }
            const int group = i / (kDSpineTilePositions * 32);
            const std::int64_t base = lane_base + head_stride +
                static_cast<std::int64_t>(head_dim) * ((tile + p) % capacity) +
                group * 32 + (i & 31);
            *reinterpret_cast<float4*>(&s_k[i]) =
                *reinterpret_cast<const float4*>(context_k + base);
            *reinterpret_cast<float4*>(&s_v[i]) =
                *reinterpret_cast<const float4*>(context_v + base);
        }
        __syncthreads();
        const int share = (count + RowWarps - 1) / RowWarps;
        const int p0    = sub * share;
        const int p1    = min(count, p0 + share);
        for (int p = p0; p < p1; ++p) {
            float key[kDSpineRowDims];
            float val[kDSpineRowDims];
#pragma unroll
            for (int i = 0; i < kDSpineRowDims; ++i) {
                key[i] = __bfloat162float(s_k[(i * kDSpineTilePositions + p) * 32 + lane]);
                val[i] = __bfloat162float(s_v[(i * kDSpineTilePositions + p) * 32 + lane]);
            }
            float score = 0.0f;
#pragma unroll
            for (int i = 0; i < kDSpineRowDims; ++i) { score += key[i] * qv[i]; }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(kFullWarpMask, score, offset);
            }
            score *= scale;
            const float m_new     = fmaxf(m, score);
            const float e         = expf(score - m_new);
            const float scale_old = expf(m - m_new);
            l = l * scale_old + e;
#pragma unroll
            for (int i = 0; i < kDSpineRowDims; ++i) {
                acc[i] = acc[i] * scale_old + e * val[i];
            }
            m = m_new;
        }
        __syncthreads();
    }

    // Proposal block: row `row` attends to proposal columns 0..row. Done once per row.
    if (sub == 0) {
        const int last = min(row + 1, valid);
        for (int j = 0; j < last; ++j) {
            const std::int64_t base =
                static_cast<std::int64_t>(head_dim) *
                    (kv_head + kv_heads * (j + width * b)) + lane;
            float key[kDSpineRowDims];
            float val[kDSpineRowDims];
#pragma unroll
            for (int i = 0; i < kDSpineRowDims; ++i) {
                key[i] = __bfloat162float(k[base + 32 * i]);
                val[i] = __bfloat162float(v[base + 32 * i]);
            }
            float score = 0.0f;
#pragma unroll
            for (int i = 0; i < kDSpineRowDims; ++i) { score += key[i] * qv[i]; }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(kFullWarpMask, score, offset);
            }
            score *= scale;
            const float m_new     = fmaxf(m, score);
            const float e         = expf(score - m_new);
            const float scale_old = expf(m - m_new);
            l = l * scale_old + e;
#pragma unroll
            for (int i = 0; i < kDSpineRowDims; ++i) {
                acc[i] = acc[i] * scale_old + e * val[i];
            }
            m = m_new;
        }
    }

    // Merge the RowWarps partials of each row. Lane indexes the head_dim slice,
    // so it must be part of the shared index -- collapsing it makes every lane write
    // the same slot and the draft silently degenerates. Sized by Rows rather than by the
    // dispatch maximum so the context tiles and the merge fit in 48 KB together.
    __shared__ float s_acc[Rows * RowWarps * 32 * kDSpineRowDims];
    __shared__ float s_m[Rows * RowWarps];
    __shared__ float s_l[Rows * RowWarps];
    const int slot_index = row * RowWarps + sub;
    if (lane == 0) {
        s_m[slot_index] = m;
        s_l[slot_index] = l;
    }
#pragma unroll
    for (int i = 0; i < kDSpineRowDims; ++i) {
        s_acc[(slot_index * 32 + lane) * kDSpineRowDims + i] = acc[i];
    }
    __syncthreads();
    if (sub == 0) {
        float peak = -INFINITY;
        for (int s = 0; s < RowWarps; ++s) {
            peak = fmaxf(peak, s_m[row * RowWarps + s]);
        }
        float denom = 0.0f;
        for (int s = 0; s < RowWarps; ++s) {
            const float ms = s_m[row * RowWarps + s];
            if (ms > -INFINITY) {
                denom += s_l[row * RowWarps + s] * expf(ms - peak);
            }
        }
        const float inverse = denom > 0.0f ? 1.0f / denom : 0.0f;
#pragma unroll
        for (int i = 0; i < kDSpineRowDims; ++i) {
            float sum = 0.0f;
            for (int s = 0; s < RowWarps; ++s) {
                const float ms = s_m[row * RowWarps + s];
                if (ms > -INFINITY) {
                    sum += s_acc[((row * RowWarps + s) * 32 + lane) *
                                     kDSpineRowDims + i] * expf(ms - peak);
                }
            }
            out[row_base + 32 * i] = __float2bfloat16(sum * inverse);
        }
    }
}

void dspine_causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid_columns,
                             const Tensor& floors, const Tensor& lanes,
                             const Tensor& context_k, const Tensor& context_v,
                             std::int32_t capacity, std::int32_t padded_capacity,
                             std::int32_t window, float scale,
                             Tensor& out, cudaStream_t stream) {
    require_bf16(q, "q");
    require_bf16(k, "k");
    require_bf16(v, "v");
    require_bf16(context_k, "context_k");
    require_bf16(context_v, "context_v");
    require_bf16(out, "out");
    const int head_dim    = q.ne[0];
    const int query_heads = q.ne[1];
    const int kv_heads    = k.ne[1];
    const int width       = q.ne[2];
    const int batch       = q.ne[3];
    const bool bad_geometry =
        head_dim < 2 || head_dim > 1024 || (head_dim & (head_dim - 1)) != 0 ||
        query_heads == 0 || kv_heads == 0 || query_heads % kv_heads != 0 ||
        k.ne[0] != head_dim || k.ne[2] != width || k.ne[3] != batch ||
        v.ne[0] != head_dim || v.ne[1] != kv_heads || v.ne[2] != width ||
        v.ne[3] != batch || context_k.ne[0] != head_dim ||
        context_k.ne[1] != padded_capacity || context_k.ne[2] != kv_heads ||
        context_v.ne[0] != head_dim || context_v.ne[1] != padded_capacity ||
        context_v.ne[2] != kv_heads || out.ne[0] != head_dim ||
        out.ne[1] != query_heads || out.ne[2] != width || out.ne[3] != batch ||
        positions.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        lanes.dtype != DType::I32 || positions.ne[0] != width ||
        positions.ne[1] != batch || valid_columns.ne[0] != batch ||
        lanes.ne[0] != batch || capacity <= 0 || window < 0;
    if (bad_geometry ||
        (floors.data != nullptr &&
         (floors.dtype != DType::I32 || floors.ne[0] != batch))) {
        throw std::invalid_argument("dspine_causal_attention: invalid geometry");
    }
    // The warp-per-row kernel holds head_dim/32 values per lane, so head_dim must
    // divide into whole warps and fit the eight-element register tile.
    if (head_dim % 32 != 0 || head_dim > 256) {
        throw std::invalid_argument(
            "dspine_causal_attention: head_dim must be a multiple of 32 up to 256");
    }
    const int threads = 32 * kDSpineAttentionSplits;
    // Shared-context path: one block owns all proposal rows of a head, so the context
    // slice is read once instead of once per row. It stages the slice with 16-byte
    // vector loads, which needs the context pointers themselves to be 16-byte aligned;
    // a lane-offset ring buffer that is not falls back to the general kernel.
    const bool context_aligned =
        reinterpret_cast<std::uintptr_t>(context_k.data) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(context_v.data) % 16 == 0;
    const bool shared_rows = context_aligned && head_dim == 32 * kDSpineRowDims &&
                             width >= 2 && width <= kDSpineRowsMax;
    if (shared_rows) {
        const dim3 grid(query_heads, batch);
        // Each Rows picks the widest per-row split that keeps the block within the
        // 1024-thread limit, so the warps scanning the context fill the SM.
#define NINFER_DSPINE_LAUNCH_ROWS(R)                                                    \
    dspine_causal_attention_rows_kernel<R, DSpineRowSplit<R>::warps>                        \
        <<<grid, 32 * R * DSpineRowSplit<R>::warps, 0, stream>>>(                          \
        static_cast<const __nv_bfloat16*>(q.data),                                      \
        static_cast<const __nv_bfloat16*>(k.data),                                      \
        static_cast<const __nv_bfloat16*>(v.data),                                      \
        static_cast<const std::int32_t*>(positions.data),                               \
        static_cast<const std::int32_t*>(valid_columns.data),                           \
        floors.data != nullptr ? static_cast<const std::int32_t*>(floors.data) : nullptr, \
        static_cast<const std::int32_t*>(lanes.data),                                   \
        static_cast<const __nv_bfloat16*>(context_k.data),                              \
        static_cast<const __nv_bfloat16*>(context_v.data), capacity, padded_capacity,    \
        width, batch, head_dim, query_heads, kv_heads, window, scale,                    \
        static_cast<__nv_bfloat16*>(out.data))
        switch (width) {
            case 2: NINFER_DSPINE_LAUNCH_ROWS(2); break;
            case 3: NINFER_DSPINE_LAUNCH_ROWS(3); break;
            case 4: NINFER_DSPINE_LAUNCH_ROWS(4); break;
            case 5: NINFER_DSPINE_LAUNCH_ROWS(5); break;
            case 6: NINFER_DSPINE_LAUNCH_ROWS(6); break;
            case 7: NINFER_DSPINE_LAUNCH_ROWS(7); break;
            default: NINFER_DSPINE_LAUNCH_ROWS(8); break;
        }
#undef NINFER_DSPINE_LAUNCH_ROWS
        return;
    }
    const int grid    = width * batch * query_heads;
    dspine_causal_attention_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        floors.data != nullptr ? static_cast<const std::int32_t*>(floors.data) : nullptr,
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const __nv_bfloat16*>(context_k.data),
        static_cast<const __nv_bfloat16*>(context_v.data), capacity, padded_capacity, width,
        batch, head_dim, query_heads, kv_heads, window, scale,
        static_cast<__nv_bfloat16*>(out.data));
}

void dspine_write_scaled(const Tensor& write, const Tensor& norm, std::int32_t layer,
                         Tensor& out, cudaStream_t stream) {
    require_bf16(write, "feature_write_weights");
    require_bf16(norm, "norm");
    require_bf16(out, "write_scaled");
    const int r = out.ne[0];
    const int h = out.ne[1];
    if (norm.ne[0] != h || norm.ne[1] != 1 || write.ne[0] != r || write.ne[1] != h ||
        write.ne[2] < layer + 1) {
        throw std::invalid_argument("dspine_write_scaled: invalid geometry");
    }
    const int total = h * r;
    dspine_write_scaled_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(write.data) +
            static_cast<std::int64_t>(layer) * h * r,
        static_cast<const __nv_bfloat16*>(norm.data), h, r,
        static_cast<__nv_bfloat16*>(out.data));
}

void dspine_top16(Tensor& logits, Tensor& ids, std::int32_t valid_rows, cudaStream_t stream) {
    require_bf16(logits, "logits");
    if (ids.dtype != DType::I32 || logits.ne[0] < kDSpineCandidates ||
        ids.ne[0] != kDSpineCandidates || ids.ne[1] != logits.ne[1] ||
        valid_rows < kDSpineCandidates || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("dspine_top16: invalid geometry");
    }
    dspine_top16_kernel<<<logits.ne[1], 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data), logits.ne[0], valid_rows,
        static_cast<std::int32_t*>(ids.data));
}

void dspine_refine(const Tensor& cand_ids, const Tensor& rms, const Tensor& raw, const Tensor& q,
                   const Tensor& anchors, const Tensor& codebook, const Tensor& predecessor_gate,
                   const Tensor& prev, const Tensor& gate_bias, const Tensor& write_scaled,
                   const Tensor& head_rows, const Tensor& norm, std::int32_t width,
                   std::int32_t batch, float eps, Tensor& message, Tensor& delta, Tensor& scratch,
                   Tensor& table, cudaStream_t stream) {
    require_bf16(raw, "raw");
    require_bf16(q, "q");
    require_bf16(codebook, "codebook");
    require_bf16(predecessor_gate, "predecessor_gate");
    require_bf16(prev, "prev");
    require_bf16(gate_bias, "gate_bias");
    require_bf16(write_scaled, "write_scaled");
    require_bf16(head_rows, "head_rows");
    require_bf16(norm, "norm");
    const std::int32_t rows = width - 1;
    if (cand_ids.dtype != DType::I32 || rms.dtype != DType::FP32 ||
        cand_ids.ne[0] != kDSpineCandidates || cand_ids.ne[1] != rows ||
        cand_ids.ne[2] != batch || rms.ne[0] != width || rms.ne[1] != batch ||
        anchors.dtype != DType::I32 || anchors.ne[0] != batch || width < 2 ||
        rows > kDSpineMaximumRows) {
        throw std::invalid_argument("dspine_refine: invalid proposal geometry");
    }
    const int rank = gate_bias.ne[0];
    const int code = codebook.ne[0];
    const int hidden = raw.ne[0];
    const int columns = rows * kDSpineCandidates;
    if (q.ne[0] != rank || q.ne[1] != width || q.ne[2] != batch ||
        predecessor_gate.ne[0] != code || predecessor_gate.ne[1] != rank ||
        prev.ne[0] != code || prev.ne[1] != rank || raw.ne[1] != width ||
        raw.ne[2] != batch || write_scaled.ne[0] != rank || write_scaled.ne[1] != hidden ||
        norm.ne[0] != hidden || head_rows.ne[0] != hidden ||
        head_rows.ne[1] != columns * batch || message.dtype != DType::BF16 ||
        message.ne[0] != rank || message.ne[1] != columns * batch ||
        delta.dtype != DType::FP32 || delta.ne[0] != hidden ||
        delta.ne[1] != columns * batch || scratch.dtype != DType::FP32 ||
        scratch.ne[0] != 2 * rank || scratch.ne[1] < columns ||
        table.dtype != DType::FP32 ||
        table.ne[0] != kDSpineCandidates * kDSpineCandidates || table.ne[1] != rows ||
        table.ne[2] != batch) {
        throw std::invalid_argument("dspine_refine: invalid weight geometry");
    }
    dspine_refine_kernel<<<dim3(static_cast<unsigned>(columns), static_cast<unsigned>(batch)),
                           256, 0, stream>>>(
        static_cast<const std::int32_t*>(cand_ids.data), static_cast<const float*>(rms.data),
        static_cast<const __nv_bfloat16*>(raw.data),
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const __nv_bfloat16*>(codebook.data),
        static_cast<const __nv_bfloat16*>(predecessor_gate.data),
        static_cast<const __nv_bfloat16*>(prev.data),
        static_cast<const __nv_bfloat16*>(gate_bias.data),
        static_cast<const __nv_bfloat16*>(write_scaled.data),
        static_cast<const __nv_bfloat16*>(head_rows.data),
        static_cast<const __nv_bfloat16*>(norm.data), width, batch, code, rank, hidden,
        std::sqrt(static_cast<float>(code)), eps, static_cast<__nv_bfloat16*>(message.data),
        static_cast<float*>(delta.data), static_cast<float*>(scratch.data),
        static_cast<float*>(table.data));
}

void dspine_lattice_walk(const Tensor& candidate_ids, const Tensor& table,
                         const Tensor& base_positions, const SamplingConfig* configs,
                         Tensor& drafts, Tensor& proposal_q, cudaStream_t stream) {
    const int rows  = table.ne[1];
    const int batch = table.ne[2];
    if (candidate_ids.dtype != DType::I32 ||
        candidate_ids.ne[0] != kDSpineCandidates || candidate_ids.ne[1] != rows ||
        candidate_ids.ne[2] != batch || table.dtype != DType::FP32 ||
        table.ne[0] != kDSpineCandidates * kDSpineCandidates || rows < 1 ||
        rows > kDSpineMaximumRows || batch < 1 || drafts.dtype != DType::I32 ||
        drafts.ne[0] != rows || drafts.ne[1] != batch ||
        proposal_q.dtype != DType::FP32 || proposal_q.ne[0] != kDSpineCandidates ||
        proposal_q.ne[1] != rows || proposal_q.ne[2] != batch ||
        base_positions.dtype != DType::I32 || base_positions.ne[0] != batch) {
        throw std::invalid_argument("dspine_lattice_walk: invalid lattice geometry");
    }
    const char* plain_env = std::getenv("NINFER_DSPINE_PLAIN");
    const int plain       = plain_env != nullptr && plain_env[0] != '0' ? 1 : 0;
    dspine_lattice_walk_kernel<<<batch, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(table.data),
        static_cast<const std::int32_t*>(base_positions.data), configs, rows, plain,
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data));
}

} // namespace ninfer::ops
