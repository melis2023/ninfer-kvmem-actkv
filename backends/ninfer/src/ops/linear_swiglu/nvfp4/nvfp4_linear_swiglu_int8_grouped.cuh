#pragma once

// Integer-activation route for the NVFP4 27B mlp/gate_up LinearSwiGLU on sm_86.
//
// Reads the artifact's native NVFP4 weight with no repack: the E2M1 code plane is row-major
// [n][k/2] and the E4M3 scale plane is the BlockScaleK16M128x4 tiling, exactly what the A16 routes
// consume. The E2M1 codes widen losslessly to s8 (v = 2*e2m1) and feed mma.m16n8k16.s8 with the
// per-16 E4M3 group scales rescaling in FP32; activations are quantised to s8 with one FP32 scale
// per (token, 32 columns) -- the same outlier confinement the q4a8 route chose, and the reason a
// per-token scalar is not used (a massive-activation channel would starve its whole token).
//
// gate_up is one stored [gate; up] plane. The grouped mainloop answers it with paired rows: a block
// stages 64 gate rows and their 64 up partners and computes silu(gate) * up from one thread's own
// FP32 accumulators, with a single BF16 rounding and no gate plane in global memory
// (Nvfp4Int8SwiGluRowPolicy). The MMQ / MMVQ routes below the crossover keep the two-projection
// form: the gate projection writes an FP32 plane and the up projection's store reads it back and
// applies silu(gate) * up in FP32 -- the same assembly ggml's GateProduct carries
// (ggml_bridge_internal.cuh), and what removes the paired-row / N=2*intermediate constraint from
// those kernels.
//
// This adds activation quantisation error the A16 route does not have: ~1.2-1.4% relative L2 on
// realistic distributions (measured against real artifact weight rows), held to the A8Int
// allowance. It is opt-in through LinearPolicy::AllowNvfp4Int8 (--nvfp4-w4a8-int) and serves every
// width, since the mainloop pads its last 128-token tile rather than refusing a width it does not
// divide exactly; the route table's nvfp4_int8 key chooses which chain's mainloop answers.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_int8_route.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::unified {

// True when this route can serve the call: the registered 27B gate_up profile as a single-source
// NVFP4 plane, and a token count the 128-wide tiling covers exactly.
[[nodiscard]] bool nvfp4_linear_swiglu_int8_grouped_supported(const Weight& gate_up,
                                                              std::int32_t tokens);

// Token half of the admission rule, for the workspace query, which sees the profile but not the
// Weight. A single-T capacity query must report the route that will actually run.
[[nodiscard]] bool nvfp4_linear_swiglu_int8_tokens_supported(std::int32_t tokens);

// Transient bytes this route needs for the quantised activation planes over [min,max] tokens.
// Exact when min==max, an upper bound over the admitted widths otherwise. The table is asked twice
// over, because it decides two things: how wide the planes are (the unified narrow band addresses the
// call's own width, the MMVQ-shaped kernel the tile envelope) and whether the FP32 gate plane exists
// at all (the two-projection routes write it for the up GEMM's GateProduct epilogue; the grouped
// mainloop pairs gate and up rows inside the block and allocates nothing).
[[nodiscard]] std::size_t
nvfp4_linear_swiglu_int8_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens,
                                                  Nvfp4Int8Mainloop table);

// x is contiguous BF16 [k, tokens]; out is contiguous BF16 [n/2, tokens]. Requires the exact
// workspace capacity above to be reserved in `workspace`.
void nvfp4_linear_swiglu_int8_grouped_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                             cudaStream_t stream);

// The same route with the caller's RMSNorm folded into the activation quantise: `residual` is the
// un-normed row, and one edge kernel writes the planes both projections read. Reachable only under
// the unified table, and only when the caller therefore skips its standalone rmsnorm -- see the
// eligibility query in the wrapper.
void nvfp4_linear_swiglu_int8_fused_norm_launch(const Tensor& residual, const Tensor& norm_weight,
                                                float eps, const Weight& weight, Tensor& out,
                                                WorkspaceArena& workspace, Nvfp4Int8Mainloop table,
                                                cudaStream_t stream);

} // namespace ninfer::ops::detail::unified
