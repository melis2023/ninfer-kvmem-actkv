#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"

namespace ninfer::models::qwen3_5::execution {

class NativeExpertRuntime;

// `verify` selects the Verify-phase gate_up policy (DenseParameters::verify_gate_up_policy).
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false,
                                              bool verify = false, bool wide_verification = false);
// A caller that did not normalise its input. The integer gate_up route folds that norm into its own
// quantise pass, so `hidden` is then the un-normed residual row and this names the norm that would
// have produced it. `ffn` honours it only on the Dense NVFP4 single-source path the eligibility
// predicate admits, and throws if anything else hands it over.
struct InputNorm {
    const Tensor* weight;
    float eps;
};

// `wide_verification` keeps the neural path's residual activation precision in a wide single-row
// copy verification (see residual_projection_policy). With `partial`, a dense FFN writes its down
// projection to that FP32 [hidden, T] matrix instead of adding it to `residual` (a tensor-parallel
// rank's share, which the exchange adds).
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false, bool verify = false, InputBasis basis = InputBasis::Primal,
         bool wide_verification = false, NativeExpertRuntime* experts = nullptr,
         const InputNorm* input_norm = nullptr, Tensor* partial = nullptr);
// The sign vector of a rotated Dense FFN's input projection, or null.
[[nodiscard]] const Tensor* ffn_input_signs(const FfnParameters& parameters);

} // namespace ninfer::models::qwen3_5::execution
