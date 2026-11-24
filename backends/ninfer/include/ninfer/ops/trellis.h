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
 * auxiliaries `s16` (F32 [K/16], the GGUF's F16 promoted exactly), `pre` (F32 [K]), `post` (F32
 * [N]), and optionally `pin` (I32 [K]) and `pout` (I32 [N]), with the descriptor in the parameter's
 * `trellis_desc` metadata. The descriptor is host memory; the planes and the payload are device
 * memory.
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

/**
 * Op: trellis_banded_head
 *
 * The artifact splits the vocabulary head over shards of different widths and formats, each a coded
 * matrix, plus one `vocab_gather` map. The shards are concatenated in band order and the map's
 * per-vocab entry names the concatenated row that holds that vocab row:
 *
 *   concat[offset_b + i] = trellis_linear(x, band_b)[i],  offset_{b+1} = offset_b + band_b.N
 *   logits[v][:]        = concat[vocab_gather[v]][:]
 *
 * `x` is the contiguous BF16 hidden state [K,T]; every band shares one K. `vocab_gather` is
 * contiguous I32 [V] with every value in [0,sum(band.N)), read as host-model device data; `logits`
 * is contiguous, 16-byte aligned BF16 [V,T]. This is the direction ggml uses (`get_rows` of the
 * concatenated bands by the same map), not a scatter.
 *
 * Numeric: each band is one `trellis_linear` rounding, so a band row carries that Op's BF16 result;
 * the final read is an exact BF16 copy.
 *
 * Workspace: caller-owned, sized by `trellis_banded_head_workspace_bytes`, which is the largest
 * single band's `trellis_linear_workspace_bytes`. The concatenated band plane is a process-wide
 * scratch buffer that grows to the largest head the engine has asked for.
 */
void trellis_banded_head(const Tensor& x, const Weight* const* bands, std::size_t band_count,
                         const Tensor& vocab_gather, Tensor& logits, std::int32_t max_tokens,
                         WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Capacity for `trellis_banded_head` at every T in [1,max_tokens] for this band set.
 */
[[nodiscard]] std::size_t trellis_banded_head_workspace_bytes(const Weight* const* bands,
                                                             std::size_t band_count,
                                                             std::int32_t max_tokens);

/**
 * Op: trellis_input_proj
 *
 * The coded form of a fused input projection. A coded parent cannot be sliced by rows the way a
 * dense one can - its row side crosses the whole matrix (kind 1 in blocks of `rows_block`, kind 2
 * over the whole slice) - so one band of the parent is projected once and its rows are then
 * distributed to the logical outputs by a host-built row map:
 *
 *   parent = trellis_linear(x, part.weight)              // once per distinct parent weight
 *   outputs[part.output][part.dst_row + i][:] = parent[part.row_map[i]][:]
 *
 * `row_map` is contiguous I32 [part.rows] on the device with every entry in [0,parent.N); a null
 * map means the identity, i.e. `part.row_map[i] == i`. This covers both the block-aligned splits
 * (a GDN qkv parent cut at 2048-row boundaries) and the interleaved ones (a full-attention parent
 * storing query and output gate in 256-row turns) with one code path, and it never materialises a
 * second copy of the parent beyond the row copy into the destination.
 *
 * `x` is contiguous BF16 [K,T] and shared by every part; each output is contiguous BF16 [rows,T]
 * with 16-byte alignment, and the parts of one output must not overlap each other or each other's
 * parents. Numeric: one `trellis_linear` BF16 rounding per parent, then exact BF16 row copies.
 *
 * Workspace: caller-owned, sized by `trellis_input_proj_workspace_bytes` (the largest single
 * parent's `trellis_linear_workspace_bytes`). The packed parent plane is a process-wide scratch
 * buffer sized by the widest parent.
 */
struct TrellisProjectionPart {
    const Weight* weight  = nullptr;
    std::int32_t output   = 0;
    std::int32_t dst_row  = 0;
    std::int32_t rows     = 0;
    const std::int32_t* row_map = nullptr;
};

void trellis_input_proj(const Tensor& x, const TrellisProjectionPart* parts, std::size_t part_count,
                        Tensor* const* outputs, std::size_t output_count, std::int32_t max_tokens,
                        WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Capacity for `trellis_input_proj` at every T in [1,max_tokens] for this part set.
 */
[[nodiscard]] std::size_t trellis_input_proj_workspace_bytes(const TrellisProjectionPart* parts,
                                                            std::size_t part_count,
                                                            std::int32_t max_tokens);

} // namespace ninfer::ops
