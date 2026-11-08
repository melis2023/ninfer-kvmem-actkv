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

} // namespace ninfer::ops
