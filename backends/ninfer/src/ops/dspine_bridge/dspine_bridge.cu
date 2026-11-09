#include "ninfer/ops/dspine_bridge.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr float kF32Tiny = 1.1754943508222875e-38f;

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
        const float base = __bfloat162float(hidden[index * t + column]);
        const float value =
            residual ? base + __bfloat162float(residual[index * t + column]) : base;
        raw[index * t + column] = __float2bfloat16(value);
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
        const float value = __bfloat162float(raw[index * t + column]);
        normalized[index * t + column] = __float2bfloat16(value / (shared_rms + eps));
    }
}

__global__ void dspine_bridge_linear_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                            int n, int k, int t, __nv_bfloat16* out) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (index >= n * t) { return; }
    const int row = index / t;
    const int col = index - row * t;
    float sum = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        sum += __bfloat162float(x[kk * t + col]) *
               __bfloat162float(weight[row * k + kk]);
    }
    out[index] = __float2bfloat16(sum);
}

__global__ void dspine_bridge_read_proj_kernel(const __nv_bfloat16* normalized,
                                               const __nv_bfloat16* early,
                                               const __nv_bfloat16* final, int layer, int c, int h,
                                               int t, __nv_bfloat16* projected) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (index >= c * t) { return; }
    const int row = index / t;
    const int col = index - row * t;
    const __nv_bfloat16* weight =
        layer < 4 && early != nullptr ? early + static_cast<std::int64_t>(layer) * c * h : final;
    float sum = 0.0f;
    for (int hh = 0; hh < h; ++hh) {
        sum += __bfloat162float(normalized[hh * t + col]) *
               __bfloat162float(weight[row * h + hh]);
    }
    projected[index] = __float2bfloat16(sum);
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
            codes[index * t + column] =
                __float2bfloat16(sqrt_c * __bfloat162float(codebook[anchor * c + index]));
        }
        return;
    }
    float sum = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < c; index += blockDim.x) {
        const float value = __bfloat162float(projected[index * t + column]);
        sum += value * value;
    }
    sum = block_sum(sum);
    __shared__ float shared_norm;
    if (threadIdx.x == 0) { shared_norm = sqrtf(sum + eps); }
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < c; index += blockDim.x) {
        const float value = __bfloat162float(projected[index * t + column]);
        codes[index * t + column] = __float2bfloat16(sqrt_c * value / shared_norm);
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
            message[index * t + column] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int previous = column - 1;
    for (int index = static_cast<int>(threadIdx.x); index < rank; index += blockDim.x) {
        const float qv = __bfloat162float(q[index * t + column]);
        const float sv = __bfloat162float(s[index * t + previous]);
        const float vv = __bfloat162float(v[index * t + previous]);
        const float g =
            1.0f / (1.0f + expf(-(qv + sv + __bfloat162float(gate_bias[index]))));
        message[index * t + column] = __float2bfloat16(g * vv);
    }
}

__global__ void dspine_bridge_write_kernel(const __nv_bfloat16* message,
                                           const __nv_bfloat16* write, int layer, int h, int r,
                                           int t, __nv_bfloat16* delta) {
    const int index = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (index >= h * t) { return; }
    const int row = index / t;
    const int col = index - row * t;
    const __nv_bfloat16* weight = write + static_cast<std::int64_t>(layer) * h * r;
    float sum = 0.0f;
    for (int rr = 0; rr < r; ++rr) {
        sum += __bfloat162float(message[rr * t + col]) *
               __bfloat162float(weight[row * r + rr]);
    }
    delta[index] = __float2bfloat16(sum);
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
        float value = __bfloat162float(raw[index * t + column]);
        if (!anchor) { value += scale * __bfloat162float(delta[index * t + column]); }
        out[index * t + column] = __float2bfloat16(value);
        sum += value * value;
    }
    sum = block_sum(sum);
    __shared__ float shared_inv;
    if (threadIdx.x == 0) {
        shared_inv = rsqrtf(sum / static_cast<float>(h) + eps);
    }
    __syncthreads();
    for (int index = static_cast<int>(threadIdx.x); index < h; index += blockDim.x) {
        const float value = __bfloat162float(out[index * t + column]) * shared_inv *
                            __bfloat162float(next_norm[index]);
        normed[index * t + column] = __float2bfloat16(value);
    }
}

constexpr int kDSpineHeadDim = 128;
constexpr int kDSpineQueryHeads = 32;
constexpr int kDSpineKvHeads = 8;

__global__ void dspine_causal_attention_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const std::int32_t* positions, const std::int32_t* valid_columns,
    const std::int32_t* lanes, const __nv_bfloat16* context_k,
    const __nv_bfloat16* context_v, int capacity, int padded, int width, int batch,
    float scale, __nv_bfloat16* out) {
    __shared__ float reduction[kDSpineHeadDim];
    const int block = static_cast<int>(blockIdx.x);
    const int w     = block % width;
    const int rest  = block / width;
    const int b     = rest % batch;
    const int head  = rest / batch;
    const int dim   = static_cast<int>(threadIdx.x);
    const int kv_head = head / (kDSpineQueryHeads / kDSpineKvHeads);
    // All proposal rows share the same context frontier: positions[0, b]. The
    // ring holds context only; proposal K/V is the block below.
    const int frontier = positions[width * b];
    const int end      = frontier;
    const int begin    = max(0, frontier - capacity);
    const std::int64_t q_index =
        dim + static_cast<std::int64_t>(kDSpineHeadDim) *
                  (head + kDSpineQueryHeads * (w + width * b));
    const float qv = __bfloat162float(q[q_index]);
    const auto dot = [&](const __nv_bfloat16* key, std::int64_t index) {
        reduction[dim] = qv * __bfloat162float(key[index]);
        __syncthreads();
        for (int stride = kDSpineHeadDim / 2; stride > 0; stride >>= 1) {
            if (dim < stride) { reduction[dim] += reduction[dim + stride]; }
            __syncthreads();
        }
        const float result = reduction[0];
        __syncthreads();
        return result;
    };
    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;
    const auto add_key = [&](const __nv_bfloat16* key, const __nv_bfloat16* value,
                             std::int64_t key_index, std::int64_t value_index) {
        const float score = dot(key, key_index) * scale;
        const float m_new = fmaxf(m, score);
        const float e     = expf(score - m_new);
        const float scale_old = expf(m - m_new);
        l   = l * scale_old + e;
        acc = acc * scale_old + e * __bfloat162float(value[value_index]);
        m   = m_new;
    };
    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[b]) * kDSpineHeadDim * padded * kDSpineKvHeads;
    for (int p = begin; p < end; ++p) {
        const int slot = p & (capacity - 1);
        const std::int64_t base = lane_base +
            static_cast<std::int64_t>(kDSpineHeadDim) * slot +
            static_cast<std::int64_t>(kDSpineHeadDim) * padded * kv_head;
        add_key(context_k, context_v, base + dim, base + dim);
    }
    for (int j = 0; j <= w; ++j) {
        const std::int64_t base =
            static_cast<std::int64_t>(kDSpineHeadDim) *
            (kv_head + kDSpineKvHeads * (j + width * b));
        add_key(k, v, base + dim, base + dim);
    }
    out[q_index] = __float2bfloat16(acc / l);
}

constexpr int kDSpineCandidates = 16;
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
    float* delta, float* scratch, std::int32_t* drafts) {
    __shared__ std::int32_t choice[kDSpineMaximumRows * kDSpineCandidates];
    const int b       = static_cast<int>(blockIdx.x);
    const int rows    = width - 1;
    const int columns = rows * kDSpineCandidates;
    const int threads = static_cast<int>(blockDim.x);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int warps   = threads >> 5;
    const int R       = rank;
    const int C       = code_size;
    const int H       = hidden;

    // tvg/tvv for every (row, sender) pair. One warp owns one pair so the codebook row is
    // broadcast through L1 instead of being re-read per lane.
    for (int column = warp; column < columns; column += warps) {
        const int i   = column / kDSpineCandidates;
        const int j   = column - i * kDSpineCandidates;
        const int sid = i == 0 ? anchors[b]
                               : cand_ids[(b * rows + i - 1) * kDSpineCandidates + j];
        for (int r = lane; r < R; r += 32) {
            float gate  = 0.0f;
            float value = 0.0f;
            for (int c = 0; c < C; ++c) {
                const float e = sqrt_c * __bfloat162float(codebook[sid * C + c]);
                gate += e * __bfloat162float(predecessor_gate[r * C + c]);
                value += e * __bfloat162float(prev[r * C + c]);
            }
            scratch[r + 2 * R * column]     = gate;
            scratch[R + r + 2 * R * column] = value;
        }
    }
    __syncthreads();

    for (int index = tid; index < R * columns; index += threads) {
        const int r       = index % R;
        const int column  = index / R;
        const int i       = column / kDSpineCandidates;
        const int q_column = b * width + i + 1;
        const float qv    = __bfloat162float(q[r + R * q_column]);
        const float g     = 1.0f / (1.0f + expf(-(qv + scratch[r + 2 * R * column] +
                                                 __bfloat162float(gate_bias[r]))));
        message[index] =
            __float2bfloat16(g * scratch[R + r + 2 * R * column]);
    }
    __syncthreads();

    for (int index = tid; index < H * columns; index += threads) {
        const int h      = index % H;
        const int column = index / H;
        float sum        = 0.0f;
        for (int r = 0; r < R; ++r) {
            sum += __bfloat162float(write_scaled[h * R + r]) *
                   __bfloat162float(message[r + R * column]);
        }
        delta[index] = sum;
    }
    __syncthreads();

    for (int ij = tid; ij < columns; ij += threads) {
        const int i        = ij / kDSpineCandidates;
        const int q_column = b * width + i + 1;
        const float rm     = rms[q_column];
        const __nv_bfloat16* base = raw + static_cast<std::int64_t>(H) * q_column;
        const float* delta_column = delta + static_cast<std::int64_t>(H) * ij;
        float score[kDSpineCandidates];
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) { score[jp] = 0.0f; }
        // The reference normalises the refined residual before the head read:
        //   normalized = state / sqrt(mean(state^2)+eps) * norm_w
        // combined already carries norm_w, so the head dot is  norm_w * state,
        // and only the per-sender scale below is missing.
        float square_sum = 0.0f;
        for (int h = 0; h < H; ++h) {
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
        const float inverse_rms = rsqrtf(square_sum / static_cast<float>(H) + eps);
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) { score[jp] *= inverse_rms; }
        int best         = 0;
        float best_score = -INFINITY;
#pragma unroll
        for (int jp = 0; jp < kDSpineCandidates; ++jp) {
            if (score[jp] > best_score) {
                best_score = score[jp];
                best       = jp;
            }
        }
        choice[ij] = best;
    }
    __syncthreads();

    if (tid == 0) {
        std::int32_t sender = choice[0];
        for (int i = 0; i < rows; ++i) {
            const int column = b * rows + i;
            drafts[column]   = cand_ids[column * kDSpineCandidates + sender];
            if (i + 1 < rows) {
                sender = choice[(i + 1) * kDSpineCandidates + sender];
            }
        }
    }
}

__global__ void dspine_top16_kernel(__nv_bfloat16* logits, int vocab, int valid_rows,
                                    std::int32_t* ids) {
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    __nv_bfloat16* values = logits + static_cast<std::int64_t>(vocab) * column;
    __shared__ float shared_value[256];
    __shared__ int shared_index[256];
    for (int rank = 0; rank < kDSpineCandidates; ++rank) {
        float best     = -INFINITY;
        int best_index = 0;
        for (int v = tid; v < valid_rows; v += static_cast<int>(blockDim.x)) {
            const float value = __bfloat162float(values[v]);
            if (value > best) {
                best       = value;
                best_index = v;
            }
        }
        shared_value[tid] = best;
        shared_index[tid] = best_index;
        __syncthreads();
        for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other       = shared_value[tid + stride];
                const int other_index   = shared_index[tid + stride];
                if (other > shared_value[tid] ||
                    (other == shared_value[tid] && other_index < shared_index[tid])) {
                    shared_value[tid] = other;
                    shared_index[tid] = other_index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            ids[rank + kDSpineCandidates * column] = shared_index[0];
            values[shared_index[0]] = __float2bfloat16(-INFINITY);
        }
        __syncthreads();
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
    if (out.ne[1] != t || weight.ne[0] != k || weight.ne[1] != n) {
        throw std::invalid_argument("dspine_bridge: invalid linear tensors");
    }
    const int total = n * t;
    dspine_bridge_linear_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data), n, k, t,
        static_cast<__nv_bfloat16*>(out.data));
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
    if (projected.ne[1] != t || early.ne[0] != h || early.ne[1] != c || early.ne[2] < 4 ||
        final.ne[0] != h || final.ne[1] != c) {
        throw std::invalid_argument("dspine_bridge: invalid read projection");
    }
    const int total = c * t;
    dspine_bridge_read_proj_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        layer < 4 ? static_cast<const __nv_bfloat16*>(early.data) : nullptr,
        static_cast<const __nv_bfloat16*>(final.data), layer, c, h, t,
        static_cast<__nv_bfloat16*>(projected.data));
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
    if (message.ne[1] != t || write.ne[0] != r || write.ne[1] != h || write.ne[2] < layer + 1) {
        throw std::invalid_argument("dspine_bridge: invalid write tensors");
    }
    const int total = h * t;
    dspine_bridge_write_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(message.data),
        static_cast<const __nv_bfloat16*>(write.data), layer, h, r, t,
        static_cast<__nv_bfloat16*>(delta.data));
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

void dspine_causal_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid_columns,
                             const Tensor& lanes, const Tensor& context_k,
                             const Tensor& context_v, std::int32_t capacity,
                             std::int32_t padded_capacity, float scale, Tensor& out,
                             cudaStream_t stream) {
    require_bf16(q, "q");
    require_bf16(k, "k");
    require_bf16(v, "v");
    require_bf16(context_k, "context_k");
    require_bf16(context_v, "context_v");
    require_bf16(out, "out");
    const int width = q.ne[2];
    const int batch = q.ne[3];
    if (q.ne[0] != kDSpineHeadDim || q.ne[1] != kDSpineQueryHeads || k.ne[0] != kDSpineHeadDim ||
        k.ne[1] != kDSpineKvHeads || k.ne[2] != width || k.ne[3] != batch ||
        v.ne[0] != kDSpineHeadDim || v.ne[1] != kDSpineKvHeads || v.ne[2] != width ||
        v.ne[3] != batch || context_k.ne[0] != kDSpineHeadDim ||
        context_k.ne[1] != padded_capacity || context_k.ne[2] != kDSpineKvHeads ||
        context_v.ne[0] != kDSpineHeadDim || context_v.ne[1] != padded_capacity ||
        context_v.ne[2] != kDSpineKvHeads || out.ne[0] != kDSpineHeadDim ||
        out.ne[1] != kDSpineQueryHeads || out.ne[2] != width || out.ne[3] != batch ||
        positions.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        lanes.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
        valid_columns.ne[0] != batch || lanes.ne[0] != batch || capacity <= 0 ||
        (capacity & (capacity - 1)) != 0) {
        throw std::invalid_argument("dspine_causal_attention: invalid geometry");
    }
    const int grid = width * batch * kDSpineQueryHeads;
    dspine_causal_attention_kernel<<<grid, kDSpineHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const __nv_bfloat16*>(context_k.data),
        static_cast<const __nv_bfloat16*>(context_v.data), capacity, padded_capacity, width,
        batch, scale, static_cast<__nv_bfloat16*>(out.data));
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
                   Tensor& drafts, cudaStream_t stream) {
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
        drafts.dtype != DType::I32 || drafts.ne[0] != rows) {
        throw std::invalid_argument("dspine_refine: invalid weight geometry");
    }
    dspine_refine_kernel<<<batch, 256, 0, stream>>>(
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
        static_cast<std::int32_t*>(drafts.data));
}

} // namespace ninfer::ops
