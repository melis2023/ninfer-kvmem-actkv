// ninfer::ops - the PLE block of Qwen3.8-Flash-Next after its projections (contract in
// include/ninfer/ops/ple_inject.h). Three launches: per-token statistics, gate and normalised
// convolution input; the convolution and stack update; the history advance, which the record form
// leaves to ple_history_advance.
#include "ninfer/ops/ple_inject.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kStreams  = 4;
constexpr std::int32_t kHidden   = 2560;
constexpr std::int32_t kWidth    = kStreams * kHidden;
constexpr std::int32_t kTaps     = 4;
constexpr std::int32_t kDilation = 3;
constexpr std::int32_t kHistory  = (kTaps - 1) * kDilation;
constexpr int kThreads           = 512;

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + __expf(-x)); }

template <int Count>
__device__ void block_sum(float (&values)[Count], float* shared) {
#pragma unroll
    for (int i = 0; i < Count; ++i) {
        float v = values[i];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
        if ((threadIdx.x & 31) == 0) shared[i * (kThreads / 32) + (threadIdx.x >> 5)] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (int i = 0; i < Count; ++i) {
            float v = threadIdx.x < kThreads / 32 ? shared[i * (kThreads / 32) + threadIdx.x] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
            if (threadIdx.x == 0) shared[Count * (kThreads / 32) + i] = v;
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < Count; ++i) values[i] = shared[Count * (kThreads / 32) + i];
    __syncthreads();
}

// One block per token: the gate per stream (`gate`, [tokens][streams]) and the normalised
// convolution input N ([tokens][streams * hidden]).
__global__ void __launch_bounds__(kThreads)
    ple_gate_kernel(const float* __restrict__ stack, const __nv_bfloat16* __restrict__ key,
                    const __nv_bfloat16* __restrict__ value,
                    const __nv_bfloat16* __restrict__ norm_key,
                    const __nv_bfloat16* __restrict__ norm_query,
                    const __nv_bfloat16* __restrict__ norm_conv, float eps,
                    float* __restrict__ gate, float* __restrict__ normalized) {
    __shared__ float shared[16 * (kThreads / 32) + 16];
    const int t                 = blockIdx.x;
    const float* xs             = stack + static_cast<std::int64_t>(t) * kWidth;
    const __nv_bfloat16* keys   = key + static_cast<std::int64_t>(t) * kWidth;
    const __nv_bfloat16* values = value + static_cast<std::int64_t>(t) * kHidden;

    // Sums of squares of key and stack per stream, and of the value.
    float squares[2 * kStreams + 1] = {};
    for (int i = threadIdx.x; i < kWidth; i += kThreads) {
        const int c   = i / kHidden;
        const float k = __bfloat162float(keys[i]);
        const float x = xs[i];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            if (s == c) {
                squares[s] += k * k;
                squares[kStreams + s] += x * x;
            }
        }
    }
    for (int d = threadIdx.x; d < kHidden; d += kThreads) {
        const float v = __bfloat162float(values[d]);
        squares[2 * kStreams] += v * v;
    }
    block_sum(squares, shared);
    float key_scale[kStreams], query_scale[kStreams];
#pragma unroll
    for (int c = 0; c < kStreams; ++c) {
        key_scale[c]   = rsqrtf(squares[c] / kHidden + eps);
        query_scale[c] = rsqrtf(squares[kStreams + c] / kHidden + eps);
    }
    const float value_mean_square = squares[2 * kStreams] / kHidden;

    float dots[kStreams] = {};
    for (int i = threadIdx.x; i < kWidth; i += kThreads) {
        const int c   = i / kHidden;
        const float k = __bfloat162float(keys[i]) * key_scale[c] *
                        (1.0f + __bfloat162float(norm_key[i]));
        const float q = xs[i] * query_scale[c] * (1.0f + __bfloat162float(norm_query[i]));
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            if (s == c) dots[s] += k * q;
        }
    }
    block_sum(dots, shared);
    float gates[kStreams], conv_scale[kStreams];
#pragma unroll
    for (int c = 0; c < kStreams; ++c) {
        const float s      = dots[c] * rsqrtf(static_cast<float>(kHidden));
        const float signed_root = s == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(s), 1e-6f)), s);
        gates[c]           = sigmoid_f(signed_root);
        // G = gate * value, so mean G^2 = gate^2 * mean value^2.
        conv_scale[c] = rsqrtf(gates[c] * gates[c] * value_mean_square + eps);
    }
    if (threadIdx.x < kStreams) gate[t * kStreams + threadIdx.x] = gates[threadIdx.x];
    for (int i = threadIdx.x; i < kWidth; i += kThreads) {
        const int c = i / kHidden;
        const float g = gates[c] * __bfloat162float(values[i - c * kHidden]);
        normalized[static_cast<std::int64_t>(t) * kWidth + i] =
            g * conv_scale[c] * (1.0f + __bfloat162float(norm_conv[i]));
    }
}

// One thread per (token, channel): the dilated causal convolution over N, reading the positions
// before the call from `history`, and the stack update.
__global__ void __launch_bounds__(256)
    ple_conv_kernel(float* __restrict__ stack, const __nv_bfloat16* __restrict__ value,
                    const __nv_bfloat16* __restrict__ conv, const float* __restrict__ history,
                    const float* __restrict__ gate, const float* __restrict__ normalized,
                    int tokens) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(tokens) * kWidth) return;
    const int t       = static_cast<int>(i / kWidth);
    const int channel = static_cast<int>(i - static_cast<std::int64_t>(t) * kWidth);
    const int c       = channel / kHidden;
    float sum         = 0.0f;
#pragma unroll
    for (int j = 0; j < kTaps; ++j) {
        const int position = t - kDilation * (kTaps - 1 - j);
        const float n      = position >= 0
                                 ? normalized[static_cast<std::int64_t>(position) * kWidth + channel]
                                 : history[static_cast<std::int64_t>(kHistory + position) * kWidth + channel];
        sum = fmaf(__bfloat162float(conv[static_cast<std::int64_t>(channel) * kTaps + j]), n, sum);
    }
    const float convolved = sum * sigmoid_f(sum);
    const float g         = gate[t * kStreams + c] *
                    __bfloat162float(value[static_cast<std::int64_t>(t) * kHidden + channel - c * kHidden]);
    stack[i] += g + convolved;
}

// One thread per channel: the history becomes the last kHistory positions of (history, N), in
// ascending order so a position is read before any thread overwrites it.
__global__ void __launch_bounds__(256)
    ple_history_kernel(float* __restrict__ history, const float* __restrict__ normalized,
                       int tokens) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kWidth) return;
    for (int j = 0; j < kHistory; ++j) {
        const int source = j + tokens; // position in (history, N)
        const float n    = source < kHistory
                               ? history[static_cast<std::int64_t>(source) * kWidth + channel]
                               : normalized[static_cast<std::int64_t>(source - kHistory) * kWidth + channel];
        history[static_cast<std::int64_t>(j) * kWidth + channel] = n;
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ple_inject: ") + message); }
}

void require_tensor(const Tensor* tensor, DType dtype, std::int32_t n0, std::int32_t n1,
                    const char* message) {
    require(tensor != nullptr && tensor->data != nullptr && tensor->dtype == dtype &&
                tensor->is_contiguous() && tensor->ne[0] == n0 && tensor->ne[1] == n1 &&
                tensor->ne[2] == 1 && tensor->ne[3] == 1,
            message);
}

// The gate and the stack update of `tokens` positions, N into `normalized`; the history is read.
void inject(Tensor& stack, const Tensor& key, const Tensor& value, const PleInjectWeights& weights,
            float eps, const Tensor& history, Tensor& gate, Tensor& normalized,
            cudaStream_t stream) {
    const std::int32_t tokens = stack.ne[2];
    const auto bf16           = [](const Tensor* tensor) {
        return static_cast<const __nv_bfloat16*>(tensor->data);
    };
    ple_gate_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const float*>(stack.data), bf16(&key), bf16(&value), bf16(weights.norm_key),
        bf16(weights.norm_query), bf16(weights.norm_conv), eps, static_cast<float*>(gate.data),
        static_cast<float*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t elements = static_cast<std::int64_t>(tokens) * kWidth;
    ple_conv_kernel<<<static_cast<unsigned>(div_up(elements, std::int64_t{256})), 256, 0, stream>>>(
        static_cast<float*>(stack.data), bf16(&value), bf16(weights.conv),
        static_cast<const float*>(history.data), static_cast<const float*>(gate.data),
        static_cast<const float*>(normalized.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void require_inputs(const Tensor& stack, const Tensor& key, const Tensor& value,
                    const PleInjectWeights& weights, float eps, const Tensor& history) {
    require(stack.dtype == DType::FP32 && stack.is_contiguous() && stack.data != nullptr &&
                stack.ne[0] == kHidden && stack.ne[1] == kStreams && stack.ne[3] == 1,
            "stack must be contiguous FP32 [2560, 4, tokens]");
    const std::int32_t tokens = stack.ne[2];
    require(tokens > 0, "tokens must be positive");
    require(eps > 0.0f, "eps must be positive");
    require_tensor(&key, DType::BF16, kWidth, tokens, "key must be BF16 [10240, tokens]");
    require_tensor(&value, DType::BF16, kHidden, tokens, "value must be BF16 [2560, tokens]");
    require_tensor(weights.norm_key, DType::BF16, kWidth, 1, "norm_key must be BF16 [10240]");
    require_tensor(weights.norm_query, DType::BF16, kWidth, 1, "norm_query must be BF16 [10240]");
    require_tensor(weights.norm_conv, DType::BF16, kWidth, 1, "norm_conv must be BF16 [10240]");
    require_tensor(weights.conv, DType::BF16, kTaps, kWidth, "conv must be BF16 [4, 10240]");
    require_tensor(&history, DType::FP32, kWidth, kHistory, "history must be FP32 [10240, 9]");
}

void advance(Tensor& history, const Tensor& normalized, std::int32_t tokens, cudaStream_t stream) {
    ple_history_kernel<<<div_up(kWidth, 256), 256, 0, stream>>>(
        static_cast<float*>(history.data), static_cast<const float*>(normalized.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t ple_inject_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kStreams, tokens});
    (void)layout.alloc(DType::FP32, {kWidth, tokens});
    return layout.peak_bytes(1);
}

void ple_inject(Tensor& stack, const Tensor& key, const Tensor& value,
                const PleInjectWeights& weights, float eps, Tensor& history,
                WorkspaceArena& workspace, cudaStream_t stream) {
    require_inputs(stack, key, value, weights, eps, history);
    const std::int32_t tokens = stack.ne[2];
    auto scope                = workspace.scope();
    Tensor gate               = workspace.alloc(DType::FP32, {kStreams, tokens});
    Tensor normalized         = workspace.alloc(DType::FP32, {kWidth, tokens});
    inject(stack, key, value, weights, eps, history, gate, normalized, stream);
    advance(history, normalized, tokens, stream);
}

void ple_inject_record(Tensor& stack, const Tensor& key, const Tensor& value,
                       const PleInjectWeights& weights, float eps, const Tensor& history,
                       Tensor& normalized, WorkspaceArena& workspace, cudaStream_t stream) {
    require_inputs(stack, key, value, weights, eps, history);
    const std::int32_t tokens = stack.ne[2];
    require_tensor(&normalized, DType::FP32, kWidth, tokens,
                   "normalized must be FP32 [10240, tokens]");
    auto scope  = workspace.scope();
    Tensor gate = workspace.alloc(DType::FP32, {kStreams, tokens});
    inject(stack, key, value, weights, eps, history, gate, normalized, stream);
}

void ple_history_advance(Tensor& history, const Tensor& normalized, cudaStream_t stream) {
    require_tensor(&history, DType::FP32, kWidth, kHistory, "history must be FP32 [10240, 9]");
    require(normalized.ne[1] > 0, "normalized needs at least one position");
    require_tensor(&normalized, DType::FP32, kWidth, normalized.ne[1],
                   "normalized must be FP32 [10240, tokens]");
    advance(history, normalized, normalized.ne[1], stream);
}

} // namespace ninfer::ops
