#pragma once
#include "core/paged_kv_cache.h"
#include <cuda_runtime.h>

namespace ninfer::ops {
// Sample BF16 post-RoPE target Q against the represented persistent K cache.
// query:[D,H,T], rows/positions:I32[S] select Q columns and compact causal ends.
// The selected table row is immutable for the call. scratch:[pages,H,S],
// norms:[2,H,S], output:[pages,S], all FP32 and disjoint caller-owned buffers.
// Each page's token logsumexp is normalized across the COMPLETE visible cache,
// then heads are averaged. No V read, allocation, host copy or state mutation.
// D=256, page-major persistent cache; native BF16 and all D256 quantized K
// codings are decoded with stored scales. Input rows are in [0,T), positions
// are inclusive compact token ends in [0,pages*64). Operands and workspaces
// must not alias. An unmapped/out-of-range visible physical page yields NaN,
// which the owner must reject before accepting statistics.
void attention_page_mass(const Tensor& query, const Tensor& rows, const Tensor& positions,
    std::int32_t table_row, const PagedKVBatchLayerView& cache, Tensor& scratch,
    Tensor& norms, Tensor& output, cudaStream_t stream);
} // namespace ninfer::ops
