// ninfer::ops - Qwen3.8-Flash-Next MoE experts over GGUF block banks (contract in
// include/ninfer/ops/moe_experts.h). The router's pairs are grouped by expert on the device; the
// gate/up products then run once per active expert over all of its tokens, the down products
// accumulate weighted into a fixed-point plane, and the shared expert takes the same path as a
// one-expert bank that every token selects. Up to eight tokens run the vector kernel's MoE form
// (ops/linear/gguf); wider calls over device-resident banks run all three projections through the
// integer tensor-core matrix kernel over the routed pairs, where the block type has one.
#include "ninfer/ops/moe_experts.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kHidden  = 2560;
constexpr int kWidth   = 640;
constexpr int kExperts = 512;
constexpr int kTopK    = 10;
// Wider calls over device-resident banks take the matrix kernel.
constexpr int kVectorTokens = 8;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_experts_gguf: ") + message); }
}

bool shaped(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1) {
    return tensor.dtype == dtype && tensor.is_contiguous() && tensor.data != nullptr &&
           tensor.ne[0] == n0 && tensor.ne[1] == n1 && tensor.ne[2] == 1 && tensor.ne[3] == 1;
}

// How many of one expert's columns share a pass over its rows, from the average it gets.
int chunk_for(std::int64_t pairs, std::int64_t experts) {
    const std::int64_t average = (pairs + experts - 1) / experts;
    return average >= 6 ? 8 : average >= 3 ? 4 : average >= 2 ? 2 : 1;
}

gguf::MoeTable table(const GgufExpertTable& bank, int rows, int k) {
    require(bank.experts != nullptr && bank.row_bytes > 0, "an expert table is empty");
    return gguf::MoeTable{bank.experts, bank.row_bytes, rows, k};
}

bool fusable(const GgufExpertTable& gate, const GgufExpertTable& up) {
    return gate.format == up.format && gate.row_bytes == up.row_bytes;
}

// out = silu(gate) * up, rounded to BF16, over [pairs][rows] planes.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = gate[i];
    out[i]        = __float2bfloat16(g / (1.0f + expf(-g)) * up[i]);
}

// fixed[t][r] += each of token t's pairs' contribution weights[p] * down[p][r] in 2^-32 fixed
// point, rounded per pair as the vector products round it. One thread per (token, row).
__global__ void accumulate_fixed_kernel(const float* __restrict__ down,
                                        const float* __restrict__ weights, int per_token, int rows,
                                        std::int64_t count,
                                        unsigned long long* __restrict__ fixed) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / rows, r = i % rows;
    long long sum = 0;
    for (int j = 0; j < per_token; ++j) {
        const std::int64_t p = t * per_token + j;
        const double scaled  = fmin(
            fmax(double(down[p * rows + r]) * double(weights[p]) * 4294967296.0, -4.0e18), 4.0e18);
        sum += __double2ll_rn(scaled);
    }
    fixed[i] += static_cast<unsigned long long>(sum);
}

__global__ void store_fixed_kernel(const unsigned long long* __restrict__ fixed,
                                   float* __restrict__ y, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(fixed[i])) *
                              (1.0 / 4294967296.0));
}

struct Plan {
    std::int32_t tokens = 0;
    std::int32_t pairs  = 0;
};

bool matrix_path(std::int32_t tokens, bool device_resident, const GgufExpertTable& table, int rows,
                 int k) {
    return tokens > kVectorTokens && device_resident &&
           gguf::moe_matrix_supported(detail::gguf_type(table.format), rows, k, true);
}

// One bank's pass: the middle of every pair, then its weighted down product into `fixed`.
void run_bank(const Tensor& routing_ids, std::int32_t pairs, std::int32_t experts, int per_token,
              const float* pair_weights, const __nv_bfloat16* m, const void* activation,
              std::int32_t tokens, const GgufExpertTable& gate, const GgufExpertTable& up,
              const GgufExpertTable& down, bool device_resident, WorkspaceArena& workspace,
              unsigned long long* fixed, cudaStream_t stream) {
    auto scope           = workspace.scope();
    Tensor routing_bytes = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(experts, pairs))});
    const gguf::MoeRouting routing =
        gguf::moe_sort_routes(static_cast<const std::int32_t*>(routing_ids.data), pairs, experts,
                              routing_bytes.data, stream);
    const int active                = std::min(pairs, experts);
    const int chunk                 = chunk_for(pairs, active);
    Tensor middle                   = workspace.alloc(DType::BF16, {kWidth, pairs});
    auto* middle_p                  = static_cast<__nv_bfloat16*>(middle.data);
    const gguf::MoeTable gate_table = table(gate, kWidth, kHidden);
    const gguf::MoeTable up_table   = table(up, kWidth, kHidden);
    if (matrix_path(tokens, device_resident, gate, kWidth, kHidden) &&
        matrix_path(tokens, device_resident, up, kWidth, kHidden)) {
        // Each projection's pairs gathered in routing order and quantized for its schedule; an
        // expert receives at most one pair per token.
        auto planes          = workspace.scope();
        const auto gate_type = detail::gguf_type(gate.format),
                   up_type   = detail::gguf_type(up.format);
        const auto activation_bytes =
            static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(kHidden, pairs));
        Tensor gate_x = workspace.alloc(DType::U8, {activation_bytes});
        gguf::quantize_moe_matrix_activation(gate_type, m, kHidden, routing, pairs, per_token,
                                             gate_x.data, stream);
        Tensor up_x = gate_x;
        if (gguf::matrix_activation_layout(up_type) != gguf::matrix_activation_layout(gate_type)) {
            up_x = workspace.alloc(DType::U8, {activation_bytes});
            gguf::quantize_moe_matrix_activation(up_type, m, kHidden, routing, pairs, per_token,
                                                 up_x.data, stream);
        }
        Tensor gate_plane = workspace.alloc(DType::FP32, {kWidth, pairs});
        Tensor up_plane   = workspace.alloc(DType::FP32, {kWidth, pairs});
        gguf::moe_matrix_product(gate_type, gate_table, routing, active, pairs, tokens, true,
                                 gate_x.data, static_cast<float*>(gate_plane.data), stream);
        gguf::moe_matrix_product(up_type, up_table, routing, active, pairs, tokens, true, up_x.data,
                                 static_cast<float*>(up_plane.data), stream);
        const std::int64_t count = std::int64_t(kWidth) * pairs;
        swiglu_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<const float*>(gate_plane.data), static_cast<const float*>(up_plane.data),
            middle_p, count);
        CUDA_CHECK(cudaGetLastError());
    } else if (fusable(gate, up)) {
        gguf::moe_vector_swiglu(detail::gguf_type(gate.format), gate_table, up_table, routing,
                                active, per_token, activation, tokens, middle_p, chunk, stream);
    } else {
        Tensor gate_plane = workspace.alloc(DType::FP32, {kWidth, pairs});
        auto* gate_p      = static_cast<float*>(gate_plane.data);
        gguf::moe_vector_product(detail::gguf_type(gate.format), gate_table, routing, active,
                                 per_token, per_token, activation, tokens,
                                 gguf::MoeOutput{.f32 = gate_p}, chunk, stream);
        gguf::moe_vector_product(detail::gguf_type(up.format), up_table, routing, active, per_token,
                                 per_token, activation, tokens,
                                 gguf::MoeOutput{.bf16 = middle_p, .gate = gate_p}, chunk, stream);
    }
    const auto down_type = detail::gguf_type(down.format);
    if (matrix_path(tokens, device_resident, down, kHidden, kWidth)) {
        // The middle in routing order, its 640 values padded to three 256-value steps.
        Tensor middle_x = workspace.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(kWidth, pairs))});
        gguf::quantize_moe_matrix_activation(down_type, middle_p, kWidth, routing, pairs, 1,
                                             middle_x.data, stream);
        Tensor down_plane = workspace.alloc(DType::FP32, {kHidden, pairs});
        auto* down_p      = static_cast<float*>(down_plane.data);
        gguf::moe_matrix_product(down_type, table(down, kHidden, kWidth), routing, active, pairs,
                                 tokens, true, middle_x.data, down_p, stream);
        const std::int64_t count = std::int64_t(kHidden) * tokens;
        accumulate_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            down_p, pair_weights, per_token, kHidden, count, fixed);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    Tensor middle_q = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, pairs))});
    gguf::quantize_vector_activation(middle_p, kWidth, pairs, nullptr, middle_q.data, stream);
    gguf::moe_vector_product(
        down_type, table(down, kHidden, kWidth), routing, active, 1, per_token, middle_q.data,
        pairs, gguf::MoeOutput{.weighted = fixed, .weights = pair_weights}, chunk, stream);
}

} // namespace

std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    const std::int32_t pairs = tokens * kTopK;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    (void)layout.alloc(DType::I64, {kHidden, tokens});
    (void)layout.alloc(DType::I32, {tokens});
    // The larger of the two banks' passes, which run one after the other in the same scope: the
    // routing and the middle, then gate and up (the vector path's gate plane, or the matrix path's
    // activations and planes, released before the down product), then the down product's input
    // (and the matrix path's output plane).
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::moe_routing_bytes(kExperts, pairs))});
    (void)layout.alloc(DType::BF16, {kWidth, pairs});
    {
        auto planes = layout.scope();
        if (tokens > kVectorTokens) {
            const auto activation_bytes =
                static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(kHidden, pairs));
            (void)layout.alloc(DType::U8, {activation_bytes});
            (void)layout.alloc(DType::U8, {activation_bytes});
            (void)layout.alloc(DType::FP32, {kWidth, pairs});
        }
        (void)layout.alloc(DType::FP32, {kWidth, pairs});
    }
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, pairs))});
    if (tokens > kVectorTokens) {
        (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                          gguf::moe_matrix_activation_bytes(kWidth, pairs))});
        (void)layout.alloc(DType::FP32, {kHidden, pairs});
    }
    return layout.peak_bytes(1);
}

void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    require(tokens * kTopK <= 65535, "at most 6553 tokens per call");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(shaped(weights, DType::FP32, kTopK, tokens), "weights must be FP32 [10, tokens]");
    require(shared.dtype == DType::FP32 && shared.is_contiguous() && shared.numel() == tokens,
            "shared must be FP32 [tokens]");
    require(shaped(y, DType::FP32, kHidden, tokens), "y must be FP32 [2560, tokens]");
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(m.data), kHidden, tokens,
                                     nullptr, activation.data, stream);
    Tensor fixed = workspace.alloc(DType::I64, {kHidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(fixed.data, 0, fixed.bytes(), stream));
    auto* fixed_p = static_cast<unsigned long long*>(fixed.data);
    // Every token selects the shared expert (index 0 of its one-entry bank).
    Tensor shared_ids = workspace.alloc(DType::I32, {tokens});
    CUDA_CHECK(cudaMemsetAsync(shared_ids.data, 0, shared_ids.bytes(), stream));
    require(banks.experts >= kTopK && banks.experts <= kExperts, "experts must be in [10, 512]");
    const auto* m_p = static_cast<const __nv_bfloat16*>(m.data);
    run_bank(ids, tokens * kTopK, banks.experts, kTopK, static_cast<const float*>(weights.data),
             m_p, activation.data, tokens, banks.gate, banks.up, banks.down, banks.device_resident,
             workspace, fixed_p, stream);
    run_bank(shared_ids, tokens, 1, 1, static_cast<const float*>(shared.data), m_p, activation.data,
             tokens, banks.shared_gate, banks.shared_up, banks.shared_down, banks.device_resident,
             workspace, fixed_p, stream);
    const std::int64_t count = std::int64_t(kHidden) * tokens;
    store_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        fixed_p, static_cast<float*>(y.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
