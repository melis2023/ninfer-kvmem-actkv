#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/common.cuh"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <new>

namespace ninfer::ops::detail::gated_delta_net {
namespace chunked {
namespace {

// One warp per 128-element row: the same rsqrt(sum(x^2) + eps) the recurrent route applies.
constexpr int kNormRowsPerBlock = 8;

__global__ void qk_inverse_norms_kernel(const __nv_bfloat16* __restrict__ q,
                                        const __nv_bfloat16* __restrict__ k,
                                        float* __restrict__ q_inv_norm,
                                        float* __restrict__ k_inv_norm, std::int64_t heads,
                                        std::int64_t tokens, float eps) {
    const std::int64_t rows = heads * tokens;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t row  = static_cast<std::int64_t>(blockIdx.x) * kNormRowsPerBlock +
                              (static_cast<int>(threadIdx.x) >> 5);
    const bool is_k         = blockIdx.y != 0;
    if (row >= rows) { return; }
    const __nv_bfloat16* src = (is_k ? k : q) + row * kStateDim + lane * 4;
    const Bf16x4Pack packed  = load_vec<Bf16x4Pack>(src);
    const float2 lo          = bf16x2_to_float2(packed.pair[0]);
    const float2 hi          = bf16x2_to_float2(packed.pair[1]);
    float sum                = lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y;
    sum                      = warp_reduce_sum(sum);
    if (lane == 0) {
        // Input rows are token-major (row = token * heads + head); the table is head-major.
        const std::int64_t token                                = row / heads;
        const std::int64_t head                                 = row - token * heads;
        (is_k ? k_inv_norm : q_inv_norm)[head * tokens + token] = rsqrtf(sum + eps);
    }
}

} // namespace

cudaError_t launch_qk_inverse_norms(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                    float* q_inv_norm, float* k_inv_norm, std::int32_t heads,
                                    std::int32_t tokens, float eps, cudaStream_t stream) {
    static_assert(kStateDim == 128, "one warp covers a 128-element row with 4 values per lane");
    const std::int64_t rows = static_cast<std::int64_t>(heads) * tokens;
    const dim3 grid(static_cast<unsigned>((rows + kNormRowsPerBlock - 1) / kNormRowsPerBlock), 2,
                    1);
    qk_inverse_norms_kernel<<<grid, kNormRowsPerBlock * ninfer::ops::kWarpSize, 0, stream>>>(
        q, k, q_inv_norm, k_inv_norm, heads, tokens, eps);
    return cudaGetLastError();
}

} // namespace chunked

std::size_t chunked_workspace_bytes(std::int32_t qk_heads, std::int32_t value_heads,
                                    std::int32_t tokens, bool normalize_qk) {
    if (tokens <= 0) { return 0; }
    return chunked::workspace_bytes(qk_heads, value_heads, tokens, normalize_qk);
}

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, bool normalize_qk, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream) {
    const auto layout = chunked::compute_workspace_layout(q.ne[1], v.ne[1], q.ne[2], normalize_qk);
    if (workspace == nullptr || workspace_bytes < layout.total_bytes) { throw std::bad_alloc(); }

    const DeviceSpan backing{workspace, workspace_bytes};
    const Tensor g_cumsum = layout.g_cumsum.bind(backing);
    const Tensor W        = layout.W.bind(backing);
    const Tensor U        = layout.U.bind(backing);
    const Tensor v_new    = layout.v_new.bind(backing);
    const Tensor h_chunk  = layout.h_chunk.bind(backing);

    const auto* q_bits = static_cast<const __nv_bfloat16*>(q.data);
    const auto* k_bits = static_cast<const __nv_bfloat16*>(k.data);
    float* q_inv_norm  = nullptr;
    float* k_inv_norm  = nullptr;
    if (normalize_qk) {
        q_inv_norm = static_cast<float*>(layout.q_inv_norm.bind(backing).data);
        k_inv_norm = static_cast<float*>(layout.k_inv_norm.bind(backing).data);
        CUDA_CHECK(chunked::launch_qk_inverse_norms(q_bits, k_bits, q_inv_norm, k_inv_norm, q.ne[1],
                                                    q.ne[2], 1.0e-6f, stream));
    }

    chunked::prepare_wy_wu_config prepare{};
    prepare.H_qk         = q.ne[1];
    prepare.H_v          = v.ne[1];
    prepare.L            = q.ne[2];
    prepare.k            = k_bits;
    prepare.k_inv_norm   = k_inv_norm;
    prepare.v            = static_cast<const __nv_bfloat16*>(v.data);
    prepare.g_in         = static_cast<const float*>(g.data);
    prepare.beta         = static_cast<const float*>(beta.data);
    prepare.W            = static_cast<__nv_bfloat16*>(W.data);
    prepare.U            = static_cast<__half*>(U.data);
    prepare.g_cumsum_out = static_cast<float*>(g_cumsum.data);
    prepare.stream       = stream;
    CUDA_CHECK(chunked::launch_prepare_wy_wu(prepare));

    chunked::state_passing_config state{};
    state.H_qk       = q.ne[1];
    state.H_v        = v.ne[1];
    state.L          = q.ne[2];
    state.W          = static_cast<const __nv_bfloat16*>(W.data);
    state.U          = static_cast<const __half*>(U.data);
    state.k          = k_bits;
    state.k_inv_norm = k_inv_norm;
    state.g_cumsum   = static_cast<const float*>(g_cumsum.data);
    state.state_in   = static_cast<const float*>(ssm_state_in.data);
    state.v_new      = static_cast<__half*>(v_new.data);
    state.h_chunk    = static_cast<__nv_bfloat16*>(h_chunk.data);
    state.state_out  = static_cast<float*>(ssm_state_out.data);
    state.stream     = stream;
    CUDA_CHECK(chunked::launch_state_passing(state));

    chunked::chunk_output_config output{};
    output.H_qk       = q.ne[1];
    output.H_v        = v.ne[1];
    output.L          = q.ne[2];
    output.q          = q_bits;
    output.q_inv_norm = q_inv_norm;
    output.k          = k_bits;
    output.k_inv_norm = k_inv_norm;
    output.v_new      = static_cast<const __half*>(v_new.data);
    output.g_cumsum   = static_cast<const float*>(g_cumsum.data);
    output.h_chunk    = static_cast<const __nv_bfloat16*>(h_chunk.data);
    output.attn_out   = static_cast<__nv_bfloat16*>(out.data);
    output.scale      = scale;
    output.stream     = stream;
    CUDA_CHECK(chunked::launch_output(output));
}

} // namespace ninfer::ops::detail::gated_delta_net
