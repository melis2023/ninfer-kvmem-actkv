#include "ops/linear/q6/q6_shapes.h"

#include "ops/common/device_route.h"

#include <array>

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4_cg;
    if (tokens <= 5) return launch_q6_a16_simt_r8_t5_cg;
    if (tokens <= 6) return launch_q6_a16_simt_r8_t6_cg;
    if (tokens <= 7) return launch_q6_a16_simt_r8_t7_cg;
    if (tokens <= 16) return launch_q6_a16_mma_r64_t16_k128;
    if (tokens <= 24) return launch_q6_a16_mma_r64_t24_k128;
    if (tokens <= 32) return launch_q6_a16_mma_r64_t32_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    return launch_q6_a16_mma_r64_t128;
}

// ashalliants measured both against the legacy routes on an RTX 3090 (cold, us): the GEMV 1159 vs
// 1381 for the SIMT tile at T=1 and a tie at T=2; the small-T MMA 1154-1175 at T=3..8 (against
// 1651 at T=4 and 2450 at T=8) and 1793-2386 at T=17..32 (against 2602-2891 for the 64-row MMA).
// This line's unified sliced-K kernels, which the 3090 route table takes at T=4..18 and 21..40,
// were not in that race; the profile entry, set by calibration on the GPU, decides each width.
Q6Launch routed_q6_n248320_k5120(std::int32_t tokens) {
    enum class Head : std::uint8_t { Gemv, SmallT };
    static constexpr std::array<DeviceRouteCandidate<Head>, 2> kCandidates{{
        {"gemv", Head::Gemv, 2},
        {"small_t", Head::SmallT, 32},
    }};
    const auto* routed = routed_candidate<Head>("q6_head/248320x5120", tokens, kCandidates);
    if (routed == nullptr) { return nullptr; }
    if (routed->id == Head::Gemv) {
        return tokens == 1 ? launch_q6_a16_rowsplit_gemv_t1 : launch_q6_a16_rowsplit_gemv_t2;
    }
    if (tokens <= 8) { return launch_q6_a16_small_t_c8; }
    if (tokens <= 16) { return launch_q6_a16_small_t_c16; }
    return launch_q6_a16_small_t_c32;
}

} // namespace ninfer::ops::detail
