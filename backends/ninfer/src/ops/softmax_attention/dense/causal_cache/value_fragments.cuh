#pragma once

// FP16 PV B fragments decoded in registers from INT8-family value codes that ldmatrix.trans read
// from shared memory as byte pairs. The fast prompt kernel (prompt_i8_fast.cuh) and the pipelined
// small-T kernel (small_t_i8_pipelined.cuh) share them, so both widen a code exactly as the
// retired FP16 staging tile did: the code exactly to FP16, then one FP16 multiply by its key's
// represented group scale, which rounds like the FP32 product rounded to FP16.

#include <cuda_fp16.h>

#include "ops/common/memory.cuh"

namespace ninfer::ops {

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds the codes
// {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}. Returns the FP16 B-fragment halves
// {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly and multiplied
// once by its key's represented group scale (`scales` holds the FP16 scales of keys k and k + 1).
__device__ __forceinline__ void causal_i8_value_pair_fragments(unsigned codes, unsigned scales,
                                                               unsigned& even, unsigned& odd) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    unsigned e            = __byte_perm(biased, 0x64646464u, 0x4240);
    unsigned o            = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), s2);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), s2);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

// One ldmatrix.trans b16 lane of a packed int4 [key][d/2] tile holds dimensions 4c..4c+3 of keys k
// and k+1, one nibble each (low nibble first, byte order: k lo pair, k hi pair, k+1 lo, k+1 hi).
// Returns, for r = 0..3, the FP16 B-fragment half {V[k][4c+r], V[k+1][4c+r]}, each code widened
// exactly and multiplied once by its key's represented group scale.
__device__ __forceinline__ void causal_i4_value_quad_fragments(unsigned codes, unsigned scales,
                                                               unsigned (&out)[4]) {
    // Signed nibble ^ 8 is code + 8; 0x6400 | nibble is the FP16 value 1024 + nibble.
    const unsigned biased = codes ^ 0x88888888u;
    const __half2 offset  = __float2half2_rn(1032.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const unsigned widened = ((biased >> (4 * r)) & 0x000F000Fu) | 0x64006400u;
        const __half2 value    = __hmul2(__hsub2(load_vec<__half2>(&widened), offset), s2);
        out[r]                 = load_vec<unsigned>(&value);
    }
}

} // namespace ninfer::ops
