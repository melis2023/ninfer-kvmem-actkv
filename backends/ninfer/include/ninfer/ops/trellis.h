#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op family: trellis
 *
 * The NariQ coded-linear format (nari-llama.cpp `nariq/FORMAT.md`, version 1) keeps a matrix in
 * 16x16 tiles of 1..6-bit integer codes over a fixed INT8 grid, and moves the rotation to the
 * activation side:
 *
 *   y = gather_pout(post * T_rows^T( W_hat * T_cols(pre * gather_pin(x)) ))
 *   W_hat[n][k] = s16[k/16] * v(state[n][k]),  v in [-126,126]
 *
 * `T_cols` / `T_rows` are either identity (kind 0), a block Sylvester Hadamard RHT (kind 1), or a
 * whole-slice Hadamard with Paley factors H12/H20/H68 (kind 2, up to eight fused parts). The
 * output side uses the inverse of the same transform. Weight formats are TRL1..TRL6 (1..6 bits per
 * weight) with the `trellis_tile16_v1` layout; side tensors live in the artifact as the Use
 * auxiliaries `s16` (F16 [K/16]), `pre` (F32 [K]), `post` (F32 [N]), and optionally `pin` (I32 [K])
 * and `pout` (I32 [N]), with the descriptor in the parameter's `trellis_desc` metadata.
 *
 * Numeric: activations are rotated in FP32, the tile product accumulates in FP32, and the output
 * side runs in FP32 before one BF16 rounding of `y`. The ideal result is the definition above
 * evaluated in FP64 from the represented FP32 activations; accumulation order and the decode
 * schedule are private.
 *
 * Domain: K and N are multiples of 16, `x` is contiguous BF16 [K,T] and `y` contiguous BF16 [N,T],
 * all 16-byte aligned. Every side tensor and the descriptor must pass `trellis_desc_supported`;
 * a gather that is not a permutation is rejected.
 *
 * Workspace: caller-owned, sized by `trellis_linear_workspace_bytes`. Layout of rotated
 * activations, the FP32 result plane, and a per-weight transform plan cache are internal.
 */
void trellis_linear(const Tensor& x, const Weight& w, Tensor& y, std::int32_t max_tokens,
                    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Capacity for `trellis_linear` at every T in [1,max_tokens] for this weight.
 */
[[nodiscard]] std::size_t trellis_linear_workspace_bytes(const Weight& w, std::int32_t max_tokens);

/**
 * Op: trellis_embedding
 *
 * Decodes the coded rows `ids[t]` of a coded table and restores them to the primal basis:
 *
 *   rows[:,t] = gather_pin(pre * T_cols^T row(ids[t]))
 *
 * which is the row form of `trellis_linear`'s input side. `table` is a TRL1..TRL6 weight with
 * shape [vocab,K]; `ids` is contiguous I32 [T] with every id in [0,vocab); `rows` is contiguous,
 * 16-byte aligned BF16 [K,T]. The table's descriptor carries the column side; a table row
 * transform is rejected because a coded embedding has none.
 *
 * Numeric: one BF16 rounding of the decoded FP32 row, as `embedding`'s contract. No workspace.
 */
void trellis_embedding(const Tensor& ids, const Weight& table, Tensor& rows, cudaStream_t stream);

} // namespace ninfer::ops
