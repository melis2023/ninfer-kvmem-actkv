#pragma once

// The Gemma 4 vision tower's own kernels. Activations are patch-major: an FP32 residual h[p][i],
// BF16 projection inputs x[p][k] (a [K, P] Tensor) and FP32 projection outputs. A batch holds
// `units` media units (an image or one video frame each) of grid_w x grid_h patches, unit after
// unit, patches row by row.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::gemma4::vision_kernels {

// h[p] = x[p] + table[x_p] + table[positions + y_p], where (x_p, y_p) is patch p's column and row
// in its unit (table: [2 * positions][hidden], the x rows then the y rows); then the first block's
// input xn[p] = bf16(h[p] * rsqrt(mean h[p]^2 + eps) * norm).
struct EmbedArgs {
    const float* x             = nullptr; // [P][hidden], the patch projection
    const __nv_bfloat16* table = nullptr;
    int positions = 0, patches = 0, per_unit = 0, grid_w = 0, hidden = 0;
    const float* norm = nullptr;
    float eps         = 1e-6f;
    float* h          = nullptr;
    __nv_bfloat16* xn = nullptr;
};
void embed(const EmbedArgs& a, cudaStream_t stream);

// A block's Q, K and V [P][heads * head_dim] from its projection [P][3 * hidden]: per head RMSNorm
// (Q and K with their gammas, V without), then on Q and K the axial RoPE: the first half of a head
// turns by the patch's column, the second by its row, each half its pairs (i, i + D/4) by
// position * inv_freq[i] (inv_freq: [D/4]). Q leaves scaled by q_scale.
struct QkvArgs {
    const float* qkv = nullptr;
    int patches = 0, per_unit = 0, grid_w = 0, heads = 0, head_dim = 0;
    const float* q_norm   = nullptr;
    const float* k_norm   = nullptr;
    const float* inv_freq = nullptr;
    float eps = 1e-6f, q_scale = 1.0f;
    __nv_bfloat16 *q = nullptr, *k = nullptr, *v = nullptr;
};
void qkv(const QkvArgs& a, cudaStream_t stream);

// h[p] += rms_norm(o[p]) * post; then, with next, the next input xn[p] = bf16(rms_norm(h[p]) * next).
void residual(float* h, const float* o, int patches, int hidden, const float* post,
              const float* next, float eps, __nv_bfloat16* xn, cudaStream_t stream);

// mid[p][i] = bf16(gelu_tanh(gu[p][i]) * gu[p][width + i]).
void geglu(const float* gu, int patches, int width, __nv_bfloat16* mid,
           cudaStream_t stream);

// Each unit's soft tokens, row by row of its pooled grid: the mean of a pooling x pooling block of
// patches, times scale; standardized ((v - std_bias) * std_scale) when std_bias is given; then RMS
// normalized without a gamma, as the projection's input, BF16 [tokens][hidden].
struct PoolArgs {
    const float* h = nullptr;
    int units = 0, grid_w = 0, grid_h = 0, pooling = 0, hidden = 0;
    float scale = 1.0f, eps = 1e-6f;
    const float* std_bias  = nullptr;
    const float* std_scale = nullptr;
    __nv_bfloat16* out     = nullptr;
};
void pool(const PoolArgs& a, cudaStream_t stream);

// dst[rows[t / per_unit] + t % per_unit][i] = src[t][i] over `tokens` BF16 rows of `width`: each
// unit's soft tokens into the prompt chunk's FP32 embeddings from the row its run starts at.
void scatter_rows(const __nv_bfloat16* src, int tokens, int per_unit, int width,
                  const std::int32_t* rows, float* dst, cudaStream_t stream);

} // namespace ninfer::models::gemma4::vision_kernels
