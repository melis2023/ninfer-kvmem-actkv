#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/expert_runtime.h"
#include "ops/sparse_moe/streamed.h"
#include "models/qwen3_5/execution/rotation.h"

#include "core/layout.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

bool rotated_dense(const DenseParameters& p) {
    return rotated(p.gate_up.hadamard_signs) || rotated(p.down.hadamard_signs);
}

// A Hadamard-rotated Dense FFN projects gate/up into one plane from the rotated input; when down
// is rotated, silu_mul_hadamard writes its input already in the rotated basis.
std::size_t rotated_dense_workspace_bytes(const DenseParameters& p, std::int32_t first,
                                          std::int32_t last, bool wide_verification) {
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            rotated_workspace_bytes(p.gate_up.hadamard_signs, gu.k, last,
                                    ops::linear_workspace_capacity_bytes(
                                        gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last)));
    }
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
        down.qtype, down.n, down.k, residual_projection_policy(p.down, wide_verification), first,
        last));
    return layout.peak_bytes(1);
}

void rotated_dense_ffn(const Tensor& hidden, const DenseParameters& p, Tensor& residual,
                       WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis,
                       bool wide_verification) {
    const auto columns      = hidden.ne[1];
    const auto& gu          = p.gate_up.weight;
    const auto intermediate = gu.n / 2;
    Tensor plane            = workspace.alloc(DType::BF16, {gu.n, columns});
    {
        auto call      = workspace.scope();
        const Tensor x = rotated_input(hidden, p.gate_up.hadamard_signs, workspace, stream, basis);
        ops::linear(x, gu, plane, p.gate_up.policy, workspace, stream);
    }
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, columns});
    if (rotated(p.down.hadamard_signs)) {
        ops::silu_mul_hadamard(plane, p.down.hadamard_signs, activation, stream);
    } else {
        ops::silu_mul(plane.slice(0, 0, intermediate), plane.slice(0, intermediate, intermediate),
                      activation, stream);
    }
    ops::linear_add(activation, p.down.weight, residual,
                    residual_projection_policy(p.down, wide_verification), workspace, stream);
}

} // namespace

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp, bool verify, bool wide_verification) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* streamed = std::get_if<StreamedMoeParameters>(&parameters)) {
        std::size_t bytes = 0;
        for (const std::int32_t tokens : {first, last}) {
            bytes = std::max(bytes, ops::detail::sparse_moe_stream_workspace_bytes(
                                        tokens, streamed->weights.routed_gate_up.qtype,
                                        streamed->weights.routed_down.qtype));
        }
        // Small-T grows with the column count up to the prefill frontier.
        for (std::int32_t tokens = first; tokens <= std::min(last, ops::detail::kSparseMoeSmallTMax);
             ++tokens) {
            bytes = std::max(bytes, ops::detail::sparse_moe_stream_workspace_bytes(
                                        tokens, streamed->weights.routed_gate_up.qtype,
                                        streamed->weights.routed_down.qtype));
        }
        return bytes;
    }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p = std::get<DenseParameters>(parameters);
    if (rotated_dense(p)) {
        return rotated_dense_workspace_bytes(p, first, last, wide_verification);
    }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (is_gguf(gu.qtype)) {
        WorkspaceLayoutBuilder layout;
        const std::int32_t rows = p.up ? gu.n : gu.n / 2;
        (void)layout.alloc(DType::BF16, {rows, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(
                p.up ? ops::linear_swiglu_pair_workspace_capacity_bytes(
                           gu.qtype, p.up->weight.qtype, gu.n, gu.k, p.gate_up.policy, first, last)
                     : ops::linear_swiglu_workspace_capacity_bytes(gu.qtype, gu.n, gu.k,
                                                                   p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
        return layout.peak_bytes(1);
    }
    WorkspaceLayoutBuilder layout;
    if (mtp || gu.qtype == QType::Q6_G64_FP16) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        if (mtp) {
            (void)layout.alloc(DType::BF16, {down.n, last});
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        } else {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, verify ? p.verify_gate_up_policy : p.gate_up.policy, first,
                last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, residual_projection_policy(p.down, wide_verification),
                first, last));
        }
    }
    return layout.peak_bytes(1);
}

const Tensor* ffn_input_signs(const FfnParameters& parameters) {
    const auto* dense = std::get_if<DenseParameters>(&parameters);
    return dense != nullptr && rotated(dense->gate_up.hadamard_signs)
               ? &dense->gate_up.hadamard_signs
               : nullptr;
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream, bool mtp,
         bool verify, InputBasis basis, bool wide_verification, NativeExpertRuntime* experts,
         const InputNorm* input_norm, Tensor* partial) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (input_norm != nullptr) {
        // Only the composition at the bottom of this function folds the norm, so a caller that hands
        // an un-normed row to any other path must not be answered.
        const auto* dense = std::get_if<DenseParameters>(&parameters);
        const bool fusable =
            dense != nullptr && !mtp && basis == InputBasis::Primal && !rotated_dense(*dense) &&
            !dense->up.has_value() && !is_gguf(dense->gate_up.weight.qtype) &&
            dense->gate_up.weight.qtype != QType::Q6_G64_FP16 && experts == nullptr;
        if (!fusable) { throw std::invalid_argument("FFN: this path cannot fold its input norm"); }
    }
    if (partial != nullptr) {
        // The share reads the row it is handed, so it cannot take a caller that deferred its norm.
        if (input_norm != nullptr) {
            throw std::invalid_argument("FFN: a tensor-parallel share cannot fold its input norm");
        }
        const auto* dense = std::get_if<DenseParameters>(&parameters);
        if (dense == nullptr || mtp || rotated_dense(*dense)) {
            throw std::logic_error("a tensor-parallel FFN share is an unrotated dense FFN");
        }
        const auto& gu    = dense->gate_up.weight;
        Tensor activation = workspace.alloc(DType::BF16, {dense->up ? gu.n : gu.n / 2, columns});
        {
            auto call = workspace.scope();
            if (dense->up) {
                ops::linear_swiglu(hidden, gu, dense->up->weight, activation,
                                   dense->gate_up.policy, workspace, stream);
            } else {
                ops::linear_swiglu(hidden, gu, activation,
                                   verify && !is_gguf(gu.qtype) ? dense->verify_gate_up_policy
                                                                : dense->gate_up.policy,
                                   workspace, stream);
            }
        }
        ops::linear(activation, dense->down.weight, *partial,
                    residual_projection_policy(dense->down, wide_verification), workspace, stream);
        return;
    }
    if (const auto* moe = std::get_if<StreamedMoeParameters>(&parameters)) {
        if (experts == nullptr) { throw std::logic_error("streamed SparseMoe has no Program delivery service"); }
        if (hidden.dtype != DType::BF16 || residual.dtype != DType::BF16 ||
            hidden.ne[0] != 2048 || residual.ne[0] != 2048 || residual.ne[1] != columns ||
            columns <= 0 || !hidden.is_contiguous() || !residual.is_contiguous()) {
            throw std::invalid_argument("streamed SparseMoe needs contiguous BF16 [2048,T]");
        }
        auto& delivery = experts->layer(moe->delivery_layer);
        delivery.check();
        ops::detail::sparse_moe_stream(hidden, moe->weights, residual, delivery,
                                       experts->local_layer(moe->delivery_layer), workspace, stream);
        return;
    }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        stream);
        return;
    }
    const auto& p = std::get<DenseParameters>(parameters);
    if (rotated_dense(p)) {
        rotated_dense_ffn(hidden, p, residual, workspace, stream, basis, wide_verification);
        return;
    }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (is_gguf(gu.qtype)) {
        Tensor activation = workspace.alloc(DType::BF16, {p.up ? gu.n : gu.n / 2, columns});
        {
            auto call = workspace.scope();
            if (p.up) {
                ops::linear_swiglu(hidden, gu, p.up->weight, activation, p.gate_up.policy,
                                   workspace, stream);
            } else {
                ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
            }
        }
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    if (mtp || gu.qtype == QType::Q6_G64_FP16) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        if (mtp) {
            Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
            ops::linear(activation, down, delta, p.down.policy, workspace, stream);
            ops::residual_add(delta, residual, stream);
            return;
        }
        ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        const auto policy = verify ? p.verify_gate_up_policy : p.gate_up.policy;
        if (input_norm != nullptr) {
            ops::linear_swiglu_fused_rmsnorm_int8(hidden, *input_norm->weight, input_norm->eps, gu,
                                                  activation, policy, workspace, stream);
        } else {
            ops::linear_swiglu(hidden, gu, activation, policy, workspace, stream);
        }
    }
    ops::linear_add(activation, down, residual,
                    residual_projection_policy(p.down, wide_verification), workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
