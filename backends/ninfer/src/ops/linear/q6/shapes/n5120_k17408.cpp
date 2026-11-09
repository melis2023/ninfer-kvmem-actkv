#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// The DSpine draft MLP down projection (hidden-width output over the wider
// intermediate). Same contract as n34816_k5120.cpp: the draft block runs it at
// T <= 16 and prefill chunks walk the same row tiles, so the ladder trades the
// row count for a wider token tile as T grows.
Q6Launch select_q6_n5120_k17408(std::int32_t tokens) {
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4_cg;
    if (tokens <= 5) return launch_q6_a16_simt_r8_t5_cg;
    if (tokens <= 6) return launch_q6_a16_simt_r8_t6_cg;
    if (tokens <= 7) return launch_q6_a16_simt_r8_t7_cg;
    if (tokens <= 16) return launch_q6_a16_mma_r32_t16_k256;
    if (tokens <= 24) return launch_q6_a16_mma_r32_t24_k256;
    if (tokens <= 32) return launch_q6_a16_mma_r32_t32_k256;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
