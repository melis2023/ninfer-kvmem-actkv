#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

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
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
    /// Admit either A16 or integer-A8 compute profiles. Distinct from AllowA8, which selects the
    /// FP8 activation path against FP8 weights; this one quantises activations to s8 with one
    /// scale per weight group and feeds groupwise-int weights to the integer tensor cores. Held
    /// to the same A8 activation allowance.
    AllowA8Int,
    /// AllowA8Int plus the integer small-T route for the 27B gate_up at decode and verify widths
    /// (ops/linear/q4/q4_small_t_mma_i8.cuh). Separate because that route is a quality trade the
    /// prefill one is not asked to carry: it applies s8 activation quantisation to every decode
    /// step rather than to full prefill tiles only, so it is opt-in per engine.
    AllowA8IntDecode,
    /// AllowA8Int plus the cuBLAS prefill route at wide token counts: the weight is materialised as
    /// int8 with one scale per row and the GEMM handed to cuBLAS, which runs it about twice as fast
    /// as this fork's own integer mainloop can. It is a further quality trade -- per-row weight
    /// scales and per-token activation scales, where the integer route carries per-group ones -- so
    /// it is opt-in per engine, and it falls back to AllowA8Int below the width where it pays.
    AllowPrefillCublas,
    /// AllowA8Int's counterpart for block-scaled NVFP4 weights: the stored E2M1 codes widen
    /// losslessly to s8 and feed the integer tensor cores with s8-quantised activations, one FP32
    /// scale per (token, 32 columns) and the per-16 E4M3 weight scales rescaling in FP32. It is the
    /// same A8-activation quality trade the Q4/Q5 integer routes carry, opt-in per engine with
    /// --nvfp4-w4a8-int. Which mainloop answers follows the route table -- the own table runs MMQ,
    /// the unified table runs the grouped ladder, and T <= 8 runs the MMVQ-shaped row-form kernel
    /// under either -- so a calibrated band chooses it (the nvfp4_int8/* keys).
    AllowNvfp4Int8,
};

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4 || policy == LinearPolicy::AllowA8Int ||
           policy == LinearPolicy::AllowA8IntDecode || policy == LinearPolicy::AllowPrefillCublas ||
           policy == LinearPolicy::AllowNvfp4Int8;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/// Integer-A8 (s8 activation, groupwise-int weight) profiles. Never implies the FP8 A8 path.
[[nodiscard]] constexpr bool allows_a8_int(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8Int || policy == LinearPolicy::AllowA8IntDecode ||
           policy == LinearPolicy::AllowPrefillCublas;
}

/// The NVFP4 int8-grouped route: the same activation trade as allows_a8_int, but for block-scaled
/// NVFP4 weights against their stored E2M1/E4M3 planes, and never for any other weight format.
[[nodiscard]] constexpr bool allows_nvfp4_int8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowNvfp4Int8;
}

/// The cuBLAS prefill route. Only ever admitted past a width gate, because its dequantise pass is
/// weight-sized and has to be amortised over the tokens in the call.
[[nodiscard]] constexpr bool allows_cublas_prefill(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowPrefillCublas;
}

/// The route's width gate where the device profile names none: from here it beats the integer
/// mainloop it replaces on every card measured. The dequantise pass costs the same whatever the
/// token count, so a narrow call pays for it without the GEMM to amortise it; where the narrower
/// gate lies depends on the card and on whether the width is a multiple of the integer route's
/// 128-column tile, so a calibrated profile places it per Op (`w4_cublas/...` keys).
inline constexpr std::int32_t kCublasPrefillMinTokens = 512;

/// No profile moves the gate to this width or below it: a decode or verify round runs up to
/// kMaximumConcurrency lanes of a 16-token window, 8 x 16 = 128 columns, through the same Ops and
/// policies, and those rounds stay off the route.
inline constexpr std::int32_t kCublasPrefillFloorTokens = 128;

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path; with a BF16 or GGUF weight it may instead be FP32,
 * the unrounded FP32 accumulation.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]` (FP32 `[N,T]` is accepted for a BF16
 * or GGUF `w` only). Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, or Q8_G32_FP16 weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16 weights, plus
 * registered contiguous BF16 problems. Each format owns a finite registry of exact physical
 * weight problems and selects its kernel internally; a valid encoding and alignment do not imply
 * arbitrary N/K support. FP8 currently registers `[N,K]` in `{[14336,5120], [16384,5120],
 * [34816,5120], [248320,5120], [5120,6144], [5120,17408]}` at every positive T. The current NVFP4
 * problems register the five non-vocabulary FP8 geometries and accept every positive T. Q8 also
 * registers `[5120,25600]` at every positive T. BF16 registers `[14336,5120]`,
 * `[5120,6144]`, and `[256,5120]` at every positive T; any other BF16 `[N,K]` (the Gemma 4 vision
 * tower's projections, for instance), and every BF16 problem with an FP32 `out`, runs a
 * runtime-shape GEMM, cuBLAS's tensor-core BF16 GEMM with an FP32 accumulator from T=64 on. Text and MTP packed-weight problems accept every positive
 * column extent T. Registered Vision problems accept raw-patch P in
 * `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix column does not inherently
 * represent a text token. FP32 `x`, and FP32 `out` with a weight neither BF16 nor GGUF, are
 * unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 or FP32 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. Every policy permits the existing A16 implementations of BF16 and Q4/Q5/Q6/Q8.
 * FP8 accepts all three policies; AllowA8 and AllowA4 permit the private resolver to select a
 * qualified A16 or A8 route for the registered problem and T. FP8 `[248320,5120]` retains A16
 * compute under every policy at every positive T. NVFP4 uses A16 for A16Only and
 * AllowA8; AllowA4 permits the private resolver to select either a qualified A16 route or
 * activation quantization to NVFP4 at every positive T. The selected route depends only on the
 * registered problem and T.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
