#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/** Native ternary weights use T2_G128_FP16, PTQ1_G128_FP16 or T5_G128_FP16 RowSplit,
 * group 128 and FP16 multipliers. Exact packing and the T5 compute contract are defined in
 * docs/maintainer/tensor-formats.md. Split parents must use the same packing. PTQ1 has a high
 * plane; T2/T5 do not. T5 requires A8 permission and workspace, with per-group s8 activations
 * and positive K divisible by 1024; it has no A16 fallback. Public activations remain BF16.
 */
/**
 * Returns the transient capacity required by LinearSwiGLU for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 through T=16; AllowA4 accepts every positive T. Row-scaled FP8 accepts all
 * policies, with A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered profiles are:
 *   - Q4_G64_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - Q8_G32_FP16 weight [12288,2048], x [2048,T], out [6144,T];
 *   - Q8_G32_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - NVFP4 BlockScaleK16M128x4 weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_BF16 RowScale weight [34816,5120], x [5120,T], out [17408,T].
 *   Inputs and output are contiguous BF16. Q4/Q8 scales are FP16, NVFP4 scales are E4M3FN, and
 *   row-scaled FP8 has one BF16 multiplier per gate/up parent row. Gate rows `[0,17408)` precede
 *   their matching up rows `[17408,34816)`.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named activation-compute criterion, not the oracle.
 *   Production routes may fuse or materialize gate/up and may choose their natural accumulator,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *   AllowA8/AllowA4 permit FP8 activation quantization; route thresholds are implementation
 * choices.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_capacity_bytes(),
 *   scoped to the call. Q8, NVFP4 A16, and row-scaled FP8 A16 require zero bytes; A4/A8 routes use
 *   caller-owned activation storage and may use private projection storage. There is no persistent
 *   state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream);

/**
 * A16-only convenience form. Q4/Q8 and row-scaled FP8 retain their complete positive-T domain.
 * NVFP4 is admitted only through T=16; larger NVFP4 extents require the policy-bearing AllowA4
 * form.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

/**
 * Two-parent form: out[:,t] = silu(gate_weight * x[:,t]) * (up_weight * x[:,t]) for gate and up
 * matrices of one shape [N,K] stored as separate parents. Registered for GGUF block matrices, whose
 * two halves may use different block types; their products quantize the activation to q8_1 as
 * llama.cpp does, whatever the policy. Size the workspace with
 * linear_swiglu_pair_workspace_capacity_bytes().
 */
void linear_swiglu(const Tensor& x, const Weight& gate_weight, const Weight& up_weight, Tensor& out,
                   LinearPolicy policy, WorkspaceArena& ws, cudaStream_t stream);

[[nodiscard]] std::size_t linear_swiglu_pair_workspace_capacity_bytes(
    QType gate_qtype, QType up_qtype, std::int32_t rows, std::int32_t input_rows,
    LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Whether the integer-activation NVFP4 SwiGLU route takes the fused norm for this gate_up parent at
 * this width. The answer is the device profile's `nvfp4_int8/linear_swiglu` band: true exactly when
 * the unified chain's mainloop answers the call, which is the only chain carrying the fused edge.
 * Every width the unified route answers qualifies: the edge writes per-32 activation groups, which
 * is the plane both chains rescale against.
 */
[[nodiscard]] bool linear_swiglu_fused_rmsnorm_int8_eligible(const Weight& gate_up_weight,
                                                              LinearPolicy policy,
                                                              std::int32_t tokens);

/**
 * Computes the Offset RMSNorm of `residual` and the SwiGLU projection of the result with one
 * activation pass: a single kernel turns the residual row into the per-32 s8 codes and FP32 scales
 * the mainloop then consumes, replacing the caller's separate rmsnorm and this route's separate
 * quantise. `residual` is BF16 [5120, tokens], `out` BF16 [gate_up.n / 2, tokens], and the same
 * workspace capacity the ordinary int8 call for this width already asks for (this entry allocates
 * strictly less: the caller's BF16 normed row never exists).
 * Bit-for-bit the composition of Offset rmsnorm() into BF16 storage followed by that route's own
 * per-32 quantiser; the norm half reproduces the D=5120 launcher's pair ownership, accumulation
 * order, block reduction and single BF16 rounding, and the quantise half calls the group function
 * both chains already share. Refuses the own chain's interleaved MMQ plane and any width whose
 * consumer folds per 64 columns.
 */
void linear_swiglu_fused_rmsnorm_int8(const Tensor& residual, const Tensor& norm_weight, float eps,
                                      const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                                      WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
