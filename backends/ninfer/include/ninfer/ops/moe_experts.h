#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * The routed and shared experts of Qwen3.8-Flash-Next's MoE over BF16 weights (the debug and
 * oracle representation), after moe_route:
 *
 *   e_j(x) = down_j . (silu(gate_j . x) * (up_j . x))          expert width 640
 *   y      = sum_{k<10} weights[k] e_{ids[k]}(m) + shared * e_shared(m)
 *
 * `m` BF16 [2560, tokens]; `ids` I32 [10, tokens], `weights` FP32 [10, tokens], `shared` FP32
 * [tokens] from moe_route; `gate_up` BF16 [2560, 1280, 512] (each expert's 640 gate rows then 640
 * up rows), `down` BF16 [640, 2560, 512], `shared_gate_up` BF16 [2560, 1280], `shared_down` BF16
 * [640, 2560]; `y` FP32 [2560, tokens]. The oracle evaluates in FP64 from the represented inputs;
 * y is compared as FP32 by relative L2 and gross error.
 */
[[nodiscard]] std::size_t moe_experts_bf16_workspace_bytes(std::int32_t tokens);

void moe_experts_bf16(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const Tensor& gate_up, const Tensor& down,
                      const Tensor& shared_gate_up, const Tensor& shared_down,
                      WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

/**
 * The same experts over GGUF block banks (an imported GSQ-RCO release), reached through tables of
 * expert base pointers, so an expert may sit in a device bank, a cache slot or mapped host memory.
 * `gate` and `up` hold each expert's 640 rows of 2560 values, `down` its 2560 rows of 640; the
 * routed tables have `experts` entries (up to 512), the shared expert's tables one. Every entry
 * points at device-readable memory in the table's block type, rows `row_bytes` apart.
 *
 * The products quantize their activation to ggml's q8_1 as llama.cpp does (the format's
 * arithmetic): m once per token, the middle silu(gate . m) * (up . m) after its BF16 store. Each
 * expert's contribution weights[k] * e (shared * e_shared for the shared expert) is accumulated in
 * 2^-32 fixed point, so y does not depend on the order the experts run in, and stored as FP32.
 * The oracle independently decodes stored blocks and evaluates the complete expert mathematics
 * in FP64 from the public BF16 input, without the implementation's internal q8_1 quantization or
 * BF16 middle cast; y is compared as FP32 by relative L2 and gross error. Q2_0 has an independent
 * CPU decoder in its qualification; other formats' existing fixtures still use the GPU decoder.
 *
 * Up to eight tokens run ggml-style vector products. Wider calls over `device_resident` banks run
 * all three projections through ggml's integer tensor-core matrix kernel (with the same q8_1
 * arithmetic): the tables then point at device memory only, and at least 256 zero bytes follow the
 * last expert of every down bank (and of a slot pool), since the kernel reads a 640-value row in
 * three 256-value steps. Banks in mapped host memory stay on the vector products, which read them
 * across the bus better.
 */
struct GgufExpertTable {
    QType format               = QType::GGUF_Q8_0;
    const void* const* experts = nullptr; // device array of expert base pointers
    std::int64_t row_bytes     = 0;
};

struct GgufMoeWeights {
    GgufExpertTable gate, up, down;
    GgufExpertTable shared_gate, shared_up, shared_down;
    // Entries of the routed tables: 512, or the 256 an expert-pruned release keeps.
    std::int32_t experts = 512;
    // Every entry in device memory, zeros past each down bank (see above).
    bool device_resident = false;
};

[[nodiscard]] std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens);
void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream);

/**
 * The same sum in stages, for banks whose missing experts arrive in device memory, or are computed
 * elsewhere, while the resident ones run. begin quantizes m and zeroes the 2^-32 fixed-point sum in
 * `workspace`; the caller keeps that workspace scope open until finish. Each add sums the routed
 * pairs whose `ids` entry is non-negative (a negative entry is a pair another stage adds) through
 * `banks`, and the shared expert when `shared` is given. add_products adds `*count` (at most
 * `max_count`) externally computed pair values, FP32 [count][2560] with their pair indices in
 * `pairs`, weighted by `weights` and rounded per pair as the kernels round theirs; its arrays may be
 * mapped host memory. finish stores y. Pairs split across adds give exactly the single call's sum;
 * external products carry their own arithmetic.
 */
struct GgufMoeStage {
    std::int32_t tokens       = 0;
    void* activation          = nullptr;
    unsigned long long* fixed = nullptr;
};

[[nodiscard]] GgufMoeStage moe_experts_gguf_begin(const Tensor& m, WorkspaceArena& workspace,
                                                  cudaStream_t stream);
void moe_experts_gguf_add(const GgufMoeStage& stage, const Tensor& m, const Tensor& ids,
                          const Tensor& weights, const Tensor* shared,
                          const GgufMoeWeights& banks, WorkspaceArena& workspace,
                          cudaStream_t stream);
void moe_experts_gguf_add_products(const GgufMoeStage& stage, const float* values,
                                   const std::int32_t* pairs, const std::int32_t* count,
                                   std::int32_t max_count, const Tensor& weights,
                                   cudaStream_t stream);
void moe_experts_gguf_finish(const GgufMoeStage& stage, Tensor& y, cudaStream_t stream);

// A second stream on which the decode form runs the shared expert concurrently with the routed
// experts, and the events that fork it from the call's stream and join it back before the sum is
// read. All three are the caller's, reused call after call.
struct GgufMoeSide {
    cudaStream_t stream = nullptr;
    cudaEvent_t fork    = nullptr;
    cudaEvent_t join    = nullptr;
};

/**
 * The decode form, up to eight tokens over banks whose gate and up share a block type: the same
 * sum in a few launches instead of a sort, memsets, activation passes and a store per bank, with
 * every row's arithmetic unchanged (moe_experts_gguf takes it whenever it applies). begin runs the
 * pairs whose `ids` entry is non-negative and the shared expert; decode_missing adds the up rows of
 * the pairs `ids` of a second stage lists, through `banks` (missing experts staged elsewhere);
 * finish runs every down row from the two id sets and their banks, adds external products and
 * stores y. The workspace scope stays open from begin to finish. Given a side stream, begin runs
 * the shared expert there and finish waits for it.
 */
struct GgufMoeDecode {
    std::int32_t tokens       = 0;
    void* middle              = nullptr; // BF16 [tokens * 10][640]
    void* shared_middle       = nullptr; // BF16 [tokens][640]
    unsigned long long* fixed = nullptr; // [tokens][2560] and a completion counter
    // The call's preparation: every slot's pairs, the tokens' inputs and the pairs' middles
    // quantized once for all blocks.
    void* records = nullptr;
    void* inputs  = nullptr;
    void* middles = nullptr;
    cudaEvent_t join = nullptr; // the shared expert's side stream, when it ran on one
};
[[nodiscard]] bool moe_experts_gguf_decode_supported(const GgufMoeWeights& banks,
                                                     std::int32_t tokens);
[[nodiscard]] std::size_t moe_experts_gguf_decode_workspace_bytes(std::int32_t tokens);
[[nodiscard]] GgufMoeDecode moe_experts_gguf_decode_begin(const Tensor& m, const Tensor& ids,
                                                          const Tensor& shared,
                                                          const GgufMoeWeights& banks,
                                                          WorkspaceArena& workspace,
                                                          cudaStream_t stream,
                                                          const GgufMoeSide* side = nullptr);
void moe_experts_gguf_decode_missing(const GgufMoeDecode& state, const Tensor& m,
                                     const Tensor& ids, const GgufMoeWeights& banks,
                                     cudaStream_t stream);
void moe_experts_gguf_decode_finish(const GgufMoeDecode& state, const Tensor& ids,
                                    const Tensor* missing_ids, const Tensor& weights,
                                    const GgufMoeWeights& banks, const GgufMoeWeights* missing_banks,
                                    const float* products, const std::int32_t* product_pairs,
                                    const std::int32_t* product_count, Tensor& y,
                                    cudaStream_t stream);

/**
 * One native expert projection bank. `experts` is a caller-owned device array of prepared
 * Weight operands, all in `format`: BF16 or Q2/Q4/Q5/Q6/Q8 row-split weights. The operands
 * retain their actual code/high/scale planes and layout; no selected-weight repack is needed.
 * `integer_a8` selects private group-32 activation quantization and signed-byte dot products;
 * BF16 banks take A16 only. Preparation validates the represented shapes and storage.
 */
struct NativeExpertTable {
    QType format = QType::BF16;
    const Weight* experts = nullptr;
    bool integer_a8 = false;
};

struct NativeMoeWeights {
    NativeExpertTable gate, up, down;
    NativeExpertTable shared_gate, shared_up, shared_down;
    std::int32_t experts = 512;
};

[[nodiscard]] Weight prepare_native_expert(const WeightInput& input, std::int32_t rows,
                                           std::int32_t columns, bool integer_a8);
[[nodiscard]] std::size_t moe_experts_native_workspace_bytes(std::int32_t tokens);

/**
 * Residual-free Flash-Next experts with the same represented inputs, top-10 routing and FP32
 * output as moe_experts_bf16. Routed banks have `experts` entries (10..512), shared banks one;
 * gate/up have 640 rows of 2560 values, down 2560 rows of 640. Each projection keeps its own
 * format and compute profile. Each call supports 1..65535 tokens and enqueues without host synchronization.
 *
 * The independent oracle decodes the original stored words and evaluates complete SwiGLU and
 * weighted merge in FP64 from public BF16 m. Middle products, activation quantization and
 * reduction order are private arithmetic, not oracle boundaries. Repeats with fixed operands
 * and profile are bit-exact. Optional `parts` is U8 [10,tokens]: zero excludes that routed
 * contribution, nonzero evaluates it; null includes every route. The shared expert always
 * contributes. Excluded experts' weight operands are never read. A caller may compute those
 * experts on the CPU and add their weighted sum after both parts finish.
 * Caller-owned workspace and operand tables are graph-stable;
 * live inputs, weights, output and workspace do not overlap.
 */
void moe_experts_native(const Tensor& m, const Tensor& ids, const Tensor& weights,
                        const Tensor& shared, const NativeMoeWeights& banks,
                        const Tensor* parts, WorkspaceArena& workspace, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
