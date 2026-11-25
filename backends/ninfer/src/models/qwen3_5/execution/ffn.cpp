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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

// NINFER_TRACE_DUMP=<dir>: one BF16 plane as `<dir>/<name>-<column>.f32`, FP32 per element.
void dump_bf16_plane(const Tensor& t, const char* name, cudaStream_t stream) {
    const char* dir = std::getenv("NINFER_TRACE_DUMP");
    if (dir == nullptr || dir[0] == '\0' || t.data == nullptr) { return; }
    cudaStreamSynchronize(stream);
    for (std::int32_t column = 0; column < t.ne[1]; ++column) {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(t.ne[0]));
        const void* source = static_cast<const std::uint8_t*>(t.data) +
                             static_cast<std::size_t>(column) *
                                 static_cast<std::size_t>(t.ne[0]) * sizeof(std::uint16_t);
        cudaMemcpy(host.data(), source, host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::vector<float> values(host.size());
        for (std::size_t i = 0; i < host.size(); ++i) {
            std::uint32_t word = static_cast<std::uint32_t>(host[i]) << 16;
            std::memcpy(&values[i], &word, sizeof(float));
        }
        char path[512];
        std::snprintf(path, sizeof(path), "%s/%s-%03d.f32", dir, name, column);
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(values.data(), sizeof(float), values.size(), file);
            std::fclose(file);
        }
    }
}

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
    if (is_coded(gu.qtype)) {
        // Coded FFN: one trellis linear per parent (gate, up, down), each needing the coded
        // workspace, plus the gate/up/activation/delta planes.
        if (!p.up) { throw std::logic_error("coded FFN needs a separate up projection"); }
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::BF16, {gu.n, last});
        (void)layout.alloc(DType::BF16, {p.up->weight.n, last});
        (void)layout.alloc(DType::BF16, {gu.n, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        for (const auto& [qtype, rows, cols] :
             {std::tuple{gu.qtype, gu.n, gu.k}, std::tuple{p.up->weight.qtype, p.up->weight.n,
                                                          p.up->weight.k},
              std::tuple{down.qtype, down.n, down.k}}) {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                qtype, rows, cols, ops::LinearPolicy::A16Only, first, last));
        }
        return layout.peak_bytes(1);
    }
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
         const InputNorm* input_norm, Tensor* partial, Tensor* accum) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (accum != nullptr) {
        // Only the coded dense FFN forms its down projection as a separate tensor, which is the
        // delta a wide residual accumulator needs. Every other route fuses the add into the
        // projection and cannot answer here, so refuse instead of silently dropping the update.
        const auto* dense = std::get_if<DenseParameters>(&parameters);
        if (dense == nullptr || !is_coded(dense->gate_up.weight.qtype)) {
            throw std::logic_error("FFN: a wide residual accumulator needs the coded dense path");
        }
    }
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
    if (is_coded(gu.qtype)) {
        // NariQ coded FFN: gate and up are two coded parents, not one fused gate_up plane, so each
        // is one trellis linear and the product is formed by an explicit silu_mul. The down
        // projection cannot use the fused add route either, for the same reason.
        if (!p.up) { throw std::logic_error("coded FFN needs a separate up projection"); }
        Tensor gate = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate, p.gate_up.policy, workspace, stream);
        }
        Tensor up = workspace.alloc(DType::BF16, {p.up->weight.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, p.up->weight, up, p.up->policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n, columns});
        ops::silu_mul(gate, up, activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        }
        ops::residual_add(delta, residual, stream);
        // The wide accumulator mirrors the same delta so the residual stream keeps the low bits a
        // BF16 in-place add would round away on every block.
        if (accum != nullptr) { ops::residual_add(delta, *accum, stream); }
        // NINFER_TRACE_DUMP=<dir>: the first coded FFN of a run, plane by plane, so the gate/up row
        // split, the SwiGLU order and the down projection can be checked against an off-line decode.
        {
            static const char* dump_dir = std::getenv("NINFER_TRACE_DUMP");
            static bool dumped = false;
            if (dump_dir != nullptr && dump_dir[0] != '\0' && !dumped) {
                dumped = true;
                dump_bf16_plane(hidden, "ffn-in", stream);
                dump_bf16_plane(gate, "ffn-gate", stream);
                dump_bf16_plane(up, "ffn-up", stream);
                dump_bf16_plane(activation, "ffn-act", stream);
                dump_bf16_plane(delta, "ffn-delta", stream);
            }
        }
        return;
    }
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
