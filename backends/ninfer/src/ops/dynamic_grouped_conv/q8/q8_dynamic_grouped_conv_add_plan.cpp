#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_kernels.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct Plan {
    Q8DynamicConvAddSchedule schedule;
    std::size_t projected_bytes;      // the projected BF16 matrix this dispatch allocates
};

Plan resolve_plan(int input_rows, int width, int batch) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const int columns = width * batch;
    using S           = Q8DynamicConvAddSchedule;
    const S schedule  = columns <= 88 ? S::TiledMma : S::MmaK128;
    const std::size_t projected = static_cast<std::size_t>(5120) * columns * sizeof(std::uint16_t);
    return {schedule, projected};
}
} // namespace

std::size_t q8_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    QType projection_type, int input_rows, int min_width, int max_width, int min_batch, int max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    const auto projected = resolve_plan(input_rows, max_width, max_batch).projected_bytes;
    // The fused Q8 route needs only the projected matrix. Other formats use Linear with
    // caller-owned staging. The BF16 plane is a multiple of the arena's 256-byte alignment.
    return projected + (projection_type == QType::Q8_G32_FP16 ? 0 :
        linear_workspace_capacity_bytes(projection_type, 5120, input_rows,
                                        LinearPolicy::A16Only, min_width * min_batch,
                                        max_width * max_batch));
}

const char* q8_linear_dynamic_grouped_conv_add_route_name(int input_rows, int width, int batch) {
    switch (resolve_plan(input_rows, width, batch).schedule) {
    case Q8DynamicConvAddSchedule::TiledMma:
        return "dynamic_grouped_conv_add.q8.tiled_mma.materialized_bf16";
    case Q8DynamicConvAddSchedule::MmaK128:
        return "dynamic_grouped_conv_add.q8.r64_c64_k128.materialized_bf16";
    }
    throw std::logic_error("linear dynamic grouped conv add: invalid production schedule");
}

void q8_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base, const Tensor& delta,
                                                 Tensor& residual, WorkspaceArena& workspace,
                                                 cudaStream_t stream) {
    const Plan plan          = resolve_plan(x.ne[0], x.ne[1], x.ne[2]);
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.projected_bytes);
    Tensor projected(storage.data, DType::BF16, {5120, x.ne[1], x.ne[2]});
    q8_dynamic_grouped_conv_add_materialized_launch(plan.schedule, x, weight, base, delta, residual,
                                                    projected, stream);
}

void materialized_linear_dynamic_grouped_conv_add_dispatch(
    const Tensor& x, const Weight& projection_weight, const Tensor& base_kernel,
    const Tensor& finish_delta, Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    const Plan plan           = resolve_plan(x.ne[0], x.ne[1], x.ne[2]);
    auto scope                = workspace.scope();
    const DeviceSpan storage  = workspace.alloc_bytes(plan.projected_bytes);
    const std::int32_t tokens = x.ne[1] * x.ne[2];
    Tensor projected(storage.data, DType::BF16, {5120, tokens});
    linear(x.view({x.ne[0], tokens}), projection_weight, projected, LinearPolicy::A16Only,
           workspace, stream);
    dynamic_grouped_conv_finish_launch(projected, base_kernel, finish_delta, residual, x.ne[1],
                                       stream);
}
} // namespace ninfer::ops::detail
