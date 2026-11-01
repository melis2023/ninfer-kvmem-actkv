// Wide fused A16 route for NVFP4 linear_swiglu: T > 16 on an architecture with no native FP4
// tensor-core path (sm_80/86/89), where A16 is the only available compute policy.
//
// This reuses the NVFP4 A16 MMA mainloop (ops/linear/nvfp4/nvfp4_a16_mma.cuh) with the SwiGlu
// paired-rows policy and epilogue, rather than duplicating it, because that mainloop is already
// the contract this route needs:
//
//   * weights: E2M1 codes are widened to BF16 in `weight_shared` with their per-16 E4M3 block
//     scale applied during the widening (nvfp4_scaled_pair_bf16), so no separate gate/up tensor
//     and no intermediate rounding of a projection is introduced;
//   * activations: the public BF16 `x`, staged unchanged -- no FP4/FP8 quantization of A;
//   * contraction: mma.sync m16n8k16 bf16.bf16.f32, FP32 accumulators;
//   * scale order: the block scales are folded in during widening, and the global weight divisor
//     (Nvfp4A16Operands::alpha) is applied once to the accumulators after the whole K loop;
//   * fused epilogue: SwiGluRowMajorMmaEpilogue computes silu(gate) * up in FP32 and the single
//     rounding to BF16 happens in LinearBf16Output::store.
//
// That last point is why this is fused at all: routing through a BF16 gate/up pair plus silu_mul
// (the LinearA16Post composition) rounds each projection once before SiLU and loses accuracy.

#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h"

#include "core/device.h"
#include "core/weight.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

using Geometry              = Nvfp4Geometry<34816, 5120>;
constexpr int kIntermediate = Geometry::kOutputRows / 2;

} // namespace

void nvfp4_linear_swiglu_a16_mma_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    if (weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 linear_swiglu A16 MMA: unsupported weight geometry");
    }
    const auto operands = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const auto launch = [&]<class Schedule>() {
        using Instance = Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>;
        launch_nvfp4_a16_mma<Instance>(operands, output, SwiGluRowMajorMmaEpilogue{}, stream,
                                      SwiGluRowMajorMmaRows<Instance>{});
    };
    // The tiling is the one the same geometry's plain NVFP4 Linear A16 route selects, over the
    // registered rows/K extents of this Op's only shape.
    if (x.ne[1] <= 64) {
        launch.template operator()<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    } else if (x.ne[1] <= 96) {
        launch.template operator()<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>();
    } else {
        launch.template operator()<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 1, 3>>();
    }
}

} // namespace ninfer::ops::detail::unified
