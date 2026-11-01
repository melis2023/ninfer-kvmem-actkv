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
#include "ops/common/fixed_point.cuh"
#include "ops/common/silu.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

// out = silu(gate) * up, rounded to BF16, over [pairs][rows] planes: the vector kernels' SiLU
// (ops/common/silu.h), so every path computes the same middle.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = __float2bfloat16(portable_silu(gate[i]) * up[i]);
}

// fixed[t][r] += each of token t's pairs' contribution weights[p] * down[p][r] in 2^-32 fixed
// point, rounded per pair as the vector products round it. One thread per (token, row).
__global__ void accumulate_fixed_kernel(const float* __restrict__ down,
                                        const float* __restrict__ weights,
                                        const std::int32_t* __restrict__ ids, int per_token,
                                        int rows, std::int64_t count,
                                        unsigned long long* __restrict__ fixed) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / rows, r = i % rows;
    long long sum = 0;
    for (int j = 0; j < per_token; ++j) {
        const std::int64_t p = t * per_token + j;
        if (ids[p] < 0) { continue; } // a pair another stage adds
        sum += fixed_product(down[p * rows + r], weights[p]);
    }
    fixed[i] += static_cast<unsigned long long>(sum);
}

// y[t][r] = the 2^-32 fixed-point sum plus the external products of token t's pairs (each rounded
// per pair as the kernels round theirs). One thread per (token, row).
__global__ void decode_store_kernel(const unsigned long long* __restrict__ fixed,
                                    const float* __restrict__ products,
                                    const std::int32_t* __restrict__ pairs,
                                    const std::int32_t* __restrict__ count,
                                    const float* __restrict__ weights, int tokens,
                                    float* __restrict__ y) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::int64_t(tokens) * kHidden) { return; }
    unsigned long long sum = fixed[i];
    if (products != nullptr) {
        const int t = int(i / kHidden), r = int(i % kHidden);
        const int n = *count;
        for (int j = 0; j < n; ++j) {
            const int pair = pairs[j];
            if (pair / kTopK != t) { continue; }
            sum += static_cast<unsigned long long>(
                fixed_product(products[std::int64_t(j) * kHidden + r], weights[pair]));
        }
    }
    y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(sum)) *
                              (1.0 / 4294967296.0));
}

__global__ void store_fixed_kernel(const unsigned long long* __restrict__ fixed,
                                   float* __restrict__ y, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(fixed[i])) *
                              (1.0 / 4294967296.0));
}

// fixed[pair's token][r] += weights[pair] * values[i][r] for the listed pairs, rounded per pair
// as the vector products round theirs. One block row per listed pair.
__global__ void add_products_kernel(const float* __restrict__ values,
                                    const std::int32_t* __restrict__ pairs,
                                    const std::int32_t* __restrict__ count, int per_token,
                                    const float* __restrict__ weights,
                                    unsigned long long* __restrict__ fixed) {
    const int i = blockIdx.y;
    if (i >= *count) { return; }
    const int pair = pairs[i];
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < kHidden; r += gridDim.x * blockDim.x) {
        atomicAdd(fixed + std::int64_t(pair / per_token) * kHidden + r,
                  static_cast<unsigned long long>(
                      fixed_product(values[std::int64_t(i) * kHidden + r], weights[pair])));
    }
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
        gguf::moe_matrix_product(gate_type, gate_table, routing, active, pairs, true, gate_x.data,
                                 static_cast<float*>(gate_plane.data), stream);
        gguf::moe_matrix_product(up_type, up_table, routing, active, pairs, true, up_x.data,
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
                                 true, middle_x.data, down_p, stream);
        const std::int64_t count = std::int64_t(kHidden) * tokens;
        accumulate_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            down_p, pair_weights, static_cast<const std::int32_t*>(routing_ids.data), per_token,
            kHidden, count, fixed);
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

// The routing of a one-expert bank every token selects, laid out as moe_sort_routes lays it out
// (bounds [2], sorted [tokens], active [1], active count): no sort needed.
__global__ void shared_routing_kernel(std::int32_t* __restrict__ base, int tokens) {
    std::int32_t* bounds = base;
    std::int32_t* sorted = bounds + 2;
    std::int32_t* active = sorted + tokens;
    if (threadIdx.x == 0) {
        bounds[0] = 0;
        bounds[1] = tokens;
        active[0] = 0;
        active[1] = 1;
    }
    if (int(threadIdx.x) < tokens) { sorted[threadIdx.x] = int(threadIdx.x); }
}

// run_bank's vector path for the shared expert of up to eight tokens, weighted into `fixed`.
// Its allocations stay in the caller's scope: the work may still run on a side stream after the
// call returns.
void run_shared_vector(const float* shared_weights, const void* activation, std::int32_t tokens,
                       const GgufMoeWeights& banks, WorkspaceArena& workspace,
                       unsigned long long* fixed, cudaStream_t stream) {
    Tensor routing_bytes = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(1, tokens))});
    auto* base = static_cast<std::int32_t*>(routing_bytes.data);
    shared_routing_kernel<<<1, 32, 0, stream>>>(base, tokens);
    CUDA_CHECK(cudaGetLastError());
    const gguf::MoeRouting routing{base, base + 2, base + 2 + tokens, base + 3 + tokens};
    const int chunk                 = chunk_for(tokens, 1);
    Tensor middle                   = workspace.alloc(DType::BF16, {kWidth, tokens});
    auto* middle_p                  = static_cast<__nv_bfloat16*>(middle.data);
    const gguf::MoeTable gate_table = table(banks.shared_gate, kWidth, kHidden);
    const gguf::MoeTable up_table   = table(banks.shared_up, kWidth, kHidden);
    if (fusable(banks.shared_gate, banks.shared_up)) {
        gguf::moe_vector_swiglu(detail::gguf_type(banks.shared_gate.format), gate_table, up_table,
                                routing, 1, 1, activation, tokens, middle_p, chunk, stream);
    } else {
        Tensor gate_plane = workspace.alloc(DType::FP32, {kWidth, tokens});
        auto* gate_p      = static_cast<float*>(gate_plane.data);
        gguf::moe_vector_product(detail::gguf_type(banks.shared_gate.format), gate_table, routing,
                                 1, 1, 1, activation, tokens, gguf::MoeOutput{.f32 = gate_p}, chunk,
                                 stream);
        gguf::moe_vector_product(detail::gguf_type(banks.shared_up.format), up_table, routing, 1, 1,
                                 1, activation, tokens,
                                 gguf::MoeOutput{.bf16 = middle_p, .gate = gate_p}, chunk, stream);
    }
    Tensor middle_q = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, tokens))});
    gguf::quantize_vector_activation(middle_p, kWidth, tokens, nullptr, middle_q.data, stream);
    gguf::moe_vector_product(detail::gguf_type(banks.shared_down.format),
                             table(banks.shared_down, kHidden, kWidth), routing, 1, 1, 1,
                             middle_q.data, tokens,
                             gguf::MoeOutput{.weighted = fixed, .weights = shared_weights}, chunk,
                             stream);
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
    return std::max(layout.peak_bytes(1), moe_experts_gguf_decode_workspace_bytes(tokens));
}

namespace {
void check_stage_inputs(const Tensor& m, const Tensor& ids, const Tensor& weights) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    require(tokens * kTopK <= 65535, "at most 6553 tokens per call");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(shaped(weights, DType::FP32, kTopK, tokens), "weights must be FP32 [10, tokens]");
}
} // namespace

GgufMoeStage moe_experts_gguf_begin(const Tensor& m, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0 && shaped(m, DType::BF16, kHidden, tokens), "m must be BF16 [2560, tokens]");
    GgufMoeStage stage;
    stage.tokens      = tokens;
    Tensor activation = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(m.data), kHidden, tokens,
                                     nullptr, activation.data, stream);
    Tensor fixed = workspace.alloc(DType::I64, {kHidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(fixed.data, 0, fixed.bytes(), stream));
    stage.activation = activation.data;
    stage.fixed      = static_cast<unsigned long long*>(fixed.data);
    return stage;
}

void moe_experts_gguf_add(const GgufMoeStage& stage, const Tensor& m, const Tensor& ids,
                          const Tensor& weights, const Tensor* shared,
                          const GgufMoeWeights& banks, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    check_stage_inputs(m, ids, weights);
    require(m.ne[1] == stage.tokens && stage.fixed != nullptr, "the stage began with other tokens");
    require(banks.experts >= kTopK && banks.experts <= kExperts, "experts must be in [10, 512]");
    const std::int32_t tokens = stage.tokens;
    const auto* m_p           = static_cast<const __nv_bfloat16*>(m.data);
    run_bank(ids, tokens * kTopK, banks.experts, kTopK, static_cast<const float*>(weights.data),
             m_p, stage.activation, tokens, banks.gate, banks.up, banks.down,
             banks.device_resident, workspace, stage.fixed, stream);
    if (shared != nullptr) {
        require(shared->dtype == DType::FP32 && shared->is_contiguous() &&
                    shared->numel() == tokens,
                "shared must be FP32 [tokens]");
        auto scope        = workspace.scope();
        Tensor shared_ids = workspace.alloc(DType::I32, {tokens});
        CUDA_CHECK(cudaMemsetAsync(shared_ids.data, 0, shared_ids.bytes(), stream));
        run_bank(shared_ids, tokens, 1, 1, static_cast<const float*>(shared->data), m_p,
                 stage.activation, tokens, banks.shared_gate, banks.shared_up, banks.shared_down,
                 banks.device_resident, workspace, stage.fixed, stream);
    }
}

void moe_experts_gguf_add_products(const GgufMoeStage& stage, const float* values,
                                   const std::int32_t* pairs, const std::int32_t* count,
                                   std::int32_t max_count, const Tensor& weights,
                                   cudaStream_t stream) {
    require(stage.fixed != nullptr && max_count >= 0 && max_count <= stage.tokens * kTopK,
            "invalid product count");
    if (max_count == 0) { return; }
    add_products_kernel<<<dim3(5, static_cast<unsigned>(max_count)), 512, 0, stream>>>(
        values, pairs, count, kTopK, static_cast<const float*>(weights.data), stage.fixed);
    CUDA_CHECK(cudaGetLastError());
}

void moe_experts_gguf_finish(const GgufMoeStage& stage, Tensor& y, cudaStream_t stream) {
    require(stage.fixed != nullptr && shaped(y, DType::FP32, kHidden, stage.tokens),
            "y must be FP32 [2560, tokens]");
    const std::int64_t count = std::int64_t(kHidden) * stage.tokens;
    store_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        stage.fixed, static_cast<float*>(y.data), count);
    CUDA_CHECK(cudaGetLastError());
}

namespace {
int decode_chunk(std::int32_t tokens) { return tokens <= 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8; }

const std::uint8_t* const* table_of(const GgufExpertTable& table) {
    return reinterpret_cast<const std::uint8_t* const*>(table.experts);
}
} // namespace

namespace {
bool down_fits(const GgufExpertTable& table) {
    return kWidth % gguf_block_shape(table.format).elements == 0;
}

// The shared expert takes the fused decode kernels too when its gate and up share a block type
// and its down rows split into whole blocks; otherwise the generic vector path.
bool shared_decodes(const GgufMoeWeights& banks) {
    return fusable(banks.shared_gate, banks.shared_up) && down_fits(banks.shared_down);
}

// The prepared inputs of a decode call: q8_1 values, (d, sum) and integer sums per 32 values.
std::int32_t prepared_bytes(std::int32_t k, std::int32_t columns) {
    return columns * (k + (k / 32) * std::int32_t(sizeof(half2) + sizeof(int)));
}

struct Prepared {
    std::int8_t* qs;
    half2* ds;
    int* qsum;
};

Prepared prepared_at(void* base, std::int32_t k, std::int32_t columns) {
    auto* qs  = static_cast<std::int8_t*>(base);
    auto* ds  = reinterpret_cast<half2*>(qs + std::size_t(columns) * k);
    auto* sum = reinterpret_cast<int*>(ds + std::size_t(columns) * (k / 32));
    return {qs, ds, sum};
}
} // namespace

bool moe_experts_gguf_decode_supported(const GgufMoeWeights& banks, std::int32_t tokens) {
    return tokens > 0 && tokens <= kVectorTokens && fusable(banks.gate, banks.up) &&
           down_fits(banks.down);
}

std::size_t moe_experts_gguf_decode_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {kWidth, tokens * kTopK});
    (void)layout.alloc(DType::BF16, {kWidth, tokens});
    (void)layout.alloc(DType::I64, {kHidden * tokens + 1});
    (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(gguf::MoeDecodeRecord)) *
                                   tokens * kTopK});
    (void)layout.alloc(DType::U8, {prepared_bytes(kHidden, tokens)});
    (void)layout.alloc(DType::U8, {prepared_bytes(kWidth, tokens * kTopK)});
    // A shared expert on the generic path: its activation, ids and the bank pass's workspace.
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    (void)layout.alloc(DType::I32, {tokens});
    (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(1, tokens))});
    (void)layout.alloc(DType::BF16, {kWidth, tokens});
    (void)layout.alloc(DType::FP32, {kWidth, tokens});
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, tokens))});
    return layout.peak_bytes(1);
}

namespace {
// The shared expert of a decode call, weighted into its fixed-point sum: the fused decode kernels
// when its banks allow them, else the generic vector path.
void run_shared(const Tensor& m, const Tensor& shared, const GgufMoeWeights& banks,
                WorkspaceArena& workspace, const GgufMoeDecode& out, int chunk,
                cudaStream_t stream) {
    const std::int32_t tokens = out.tokens;
    if (!shared_decodes(banks)) {
        Tensor activation = workspace.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(m.data), kHidden,
                                         tokens, nullptr, activation.data, stream);
        run_shared_vector(static_cast<const float*>(shared.data), activation.data, tokens,
                          banks, workspace, out.fixed, stream);
        return;
    }
    // The shared expert quantizes its inputs and middles itself: its block types may differ.
    gguf::MoeDecodeUpArgs shared_up;
    shared_up.m         = static_cast<const __nv_bfloat16*>(m.data);
    shared_up.slots     = tokens;
    shared_up.per_token = 1;
    shared_up.gate      = table_of(banks.shared_gate);
    shared_up.up        = table_of(banks.shared_up);
    shared_up.row_bytes = banks.shared_gate.row_bytes;
    shared_up.middle    = static_cast<__nv_bfloat16*>(out.shared_middle);
    gguf::moe_decode_up(detail::gguf_type(banks.shared_gate.format), shared_up, chunk, stream);
    gguf::MoeDecodeDownArgs down;
    down.ids_a     = nullptr;
    down.slots     = tokens;
    down.per_token = 1;
    down.table_a   = table_of(banks.shared_down);
    down.row_bytes = banks.shared_down.row_bytes;
    down.middle    = static_cast<const __nv_bfloat16*>(out.shared_middle);
    down.weights   = static_cast<const float*>(shared.data);
    down.fixed     = out.fixed;
    down.tokens    = tokens;
    gguf::moe_decode_down(detail::gguf_type(banks.shared_down.format), down, chunk, stream);
}
} // namespace

GgufMoeDecode moe_experts_gguf_decode_begin(const Tensor& m, const Tensor& ids,
                                            const Tensor& shared, const GgufMoeWeights& banks,
                                            WorkspaceArena& workspace, cudaStream_t stream,
                                            const GgufMoeSide* side) {
    const std::int32_t tokens = m.ne[1];
    require(shaped(m, DType::BF16, kHidden, tokens) && moe_experts_gguf_decode_supported(banks, tokens),
            "decode form: m must be BF16 [2560, 1..8] over fusable banks");
    require(shaped(ids, DType::I32, kTopK, tokens), "ids must be I32 [10, tokens]");
    require(shared.dtype == DType::FP32 && shared.numel() == tokens, "shared must be FP32 [tokens]");
    GgufMoeDecode out;
    out.tokens = tokens;
    out.middle        = workspace.alloc(DType::BF16, {kWidth, tokens * kTopK}).data;
    out.shared_middle = workspace.alloc(DType::BF16, {kWidth, tokens}).data;
    out.fixed = static_cast<unsigned long long*>(workspace.alloc(DType::I64, {kHidden * tokens + 1}).data);
    out.records = workspace.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(gguf::MoeDecodeRecord)) *
                                              tokens * kTopK}).data;
    out.inputs  = workspace.alloc(DType::U8, {prepared_bytes(kHidden, tokens)}).data;
    out.middles = workspace.alloc(DType::U8, {prepared_bytes(kWidth, tokens * kTopK)}).data;
    const int chunk = decode_chunk(tokens);
    // Once for every block: the slots' pairs, the inputs quantized, the sum zeroed.
    const Prepared inputs = prepared_at(out.inputs, kHidden, tokens);
    gguf::MoeDecodePrepArgs prep;
    prep.m          = static_cast<const __nv_bfloat16*>(m.data);
    prep.tokens     = tokens;
    prep.ids        = static_cast<const std::int32_t*>(ids.data);
    prep.slots      = tokens * kTopK;
    prep.per_token  = kTopK;
    prep.records    = static_cast<gguf::MoeDecodeRecord*>(out.records);
    prep.qs         = inputs.qs;
    prep.ds         = inputs.ds;
    prep.qsum       = inputs.qsum;
    prep.zero       = out.fixed;
    prep.zero_count = kHidden * tokens + 1;
    gguf::moe_decode_prep(detail::gguf_type(banks.gate.format), prep, stream);
    const Prepared middles = prepared_at(out.middles, kWidth, tokens * kTopK);
    gguf::MoeDecodeUpArgs up;
    up.m          = static_cast<const __nv_bfloat16*>(m.data);
    up.ids        = static_cast<const std::int32_t*>(ids.data);
    up.slots      = tokens * kTopK;
    up.per_token  = kTopK;
    up.gate       = table_of(banks.gate);
    up.up         = table_of(banks.up);
    up.row_bytes  = banks.gate.row_bytes;
    up.middle     = static_cast<__nv_bfloat16*>(out.middle);
    up.records    = prep.records;
    up.qs         = inputs.qs;
    up.ds         = inputs.ds;
    up.qsum       = inputs.qsum;
    up.middle_qs         = middles.qs;
    up.middle_ds         = middles.ds;
    up.middle_qsum       = middles.qsum;
    up.middle_transposed = gguf::moe_decode_transposed(detail::gguf_type(banks.down.format));
    // The shared expert after the preparation has zeroed the sum, on the side stream when given:
    // it reads only the inputs and adds into the sum, whose integer additions commute.
    cudaStream_t shared_stream = stream;
    if (side != nullptr) {
        CUDA_CHECK(cudaEventRecord(side->fork, stream));
        CUDA_CHECK(cudaStreamWaitEvent(side->stream, side->fork, 0));
        shared_stream = side->stream;
        out.join      = side->join;
    }
    run_shared(m, shared, banks, workspace, out, chunk, shared_stream);
    if (side != nullptr) { CUDA_CHECK(cudaEventRecord(side->join, side->stream)); }
    gguf::moe_decode_up(detail::gguf_type(banks.gate.format), up, chunk, stream);
    return out;
}


void moe_experts_gguf_decode_missing(const GgufMoeDecode& state, const Tensor& m,
                                     const Tensor& ids, const GgufMoeWeights& banks,
                                     cudaStream_t stream) {
    require(state.fixed != nullptr && shaped(ids, DType::I32, kTopK, state.tokens),
            "decode form: missing ids must be I32 [10, tokens]");
    const Prepared inputs = prepared_at(state.inputs, kHidden, state.tokens);
    gguf::MoeDecodeUpArgs up;
    up.m         = static_cast<const __nv_bfloat16*>(m.data);
    up.ids       = static_cast<const std::int32_t*>(ids.data);
    up.slots     = state.tokens * kTopK;
    up.per_token = kTopK;
    up.gate      = table_of(banks.gate);
    up.up        = table_of(banks.up);
    up.row_bytes = banks.gate.row_bytes;
    up.middle    = static_cast<__nv_bfloat16*>(state.middle);
    up.qs        = inputs.qs;
    up.ds        = inputs.ds;
    up.qsum      = inputs.qsum;
    // The second stage lists its slots and quantizes their middles.
    const Prepared middles = prepared_at(state.middles, kWidth, state.tokens * kTopK);
    up.middle_qs         = middles.qs;
    up.middle_ds         = middles.ds;
    up.middle_qsum       = middles.qsum;
    up.middle_transposed = gguf::moe_decode_transposed(detail::gguf_type(banks.down.format));
    up.write_records     = static_cast<gguf::MoeDecodeRecord*>(state.records);
    gguf::moe_decode_up(detail::gguf_type(banks.gate.format), up, decode_chunk(state.tokens), stream);
}

void moe_experts_gguf_decode_finish(const GgufMoeDecode& state, const Tensor& ids,
                                    const Tensor* missing_ids, const Tensor& weights,
                                    const GgufMoeWeights& banks, const GgufMoeWeights* missing_banks,
                                    const float* products, const std::int32_t* product_pairs,
                                    const std::int32_t* product_count, Tensor& y,
                                    cudaStream_t stream) {
    require(state.fixed != nullptr && shaped(y, DType::FP32, kHidden, state.tokens),
            "y must be FP32 [2560, tokens]");
    require(shaped(weights, DType::FP32, kTopK, state.tokens), "weights must be FP32 [10, tokens]");
    require((missing_ids == nullptr) == (missing_banks == nullptr), "missing ids need their banks");
    // The up launches listed every slot and quantized its pairs' middles.
    const Prepared middles = prepared_at(state.middles, kWidth, state.tokens * kTopK);
    gguf::MoeDecodeDownArgs down;
    down.ids_a     = static_cast<const std::int32_t*>(ids.data);
    down.ids_b     = missing_ids ? static_cast<const std::int32_t*>(missing_ids->data) : nullptr;
    down.records   = static_cast<const gguf::MoeDecodeRecord*>(state.records);
    down.qs        = middles.qs;
    down.ds        = middles.ds;
    down.qsum      = middles.qsum;
    down.slots     = state.tokens * kTopK;
    down.per_token = kTopK;
    down.table_a   = table_of(banks.down);
    down.table_b   = missing_banks ? table_of(missing_banks->down) : nullptr;
    down.row_bytes = banks.down.row_bytes;
    down.middle    = static_cast<const __nv_bfloat16*>(state.middle);
    down.weights   = static_cast<const float*>(weights.data);
    down.fixed     = state.fixed;
    down.tokens    = state.tokens;
    // The store runs as its own launch: one block finishing it serially was the call's tail.
    down.y         = nullptr;
    gguf::moe_decode_down(detail::gguf_type(banks.down.format), down, decode_chunk(state.tokens),
                          stream);
    if (state.join != nullptr) { CUDA_CHECK(cudaStreamWaitEvent(stream, state.join, 0)); }
    const std::int64_t count = std::int64_t(kHidden) * state.tokens;
    decode_store_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        state.fixed, products, product_pairs, product_count, static_cast<const float*>(weights.data),
        state.tokens, static_cast<float*>(y.data));
    CUDA_CHECK(cudaGetLastError());
}

void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream) {
    check_stage_inputs(m, ids, weights);
    require(shaped(y, DType::FP32, kHidden, m.ne[1]), "y must be FP32 [2560, tokens]");
    auto scope = workspace.scope();
    if (moe_experts_gguf_decode_supported(banks, m.ne[1])) {
        const auto state = moe_experts_gguf_decode_begin(m, ids, shared, banks, workspace, stream);
        moe_experts_gguf_decode_finish(state, ids, nullptr, weights, banks, nullptr, nullptr,
                                       nullptr, nullptr, y, stream);
        return;
    }
    const auto stage  = moe_experts_gguf_begin(m, workspace, stream);
    moe_experts_gguf_add(stage, m, ids, weights, &shared, banks, workspace, stream);
    moe_experts_gguf_finish(stage, y, stream);
}

} // namespace ninfer::ops
