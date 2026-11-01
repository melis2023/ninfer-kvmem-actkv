#include "ops/linear/t5/t5_project.h"

#include "core/weight.h"
#include "core/weight_view.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/sm70/sm70_integer.h"
#include "ops/linear/sm70/sm70_project.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/linear/t2/t2_weight_view.h"
#include "ops/linear_swiglu/q4cublas/cublas_prefill_gate.h"
#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_conv_tensor(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                         std::int32_t batch, const char* op, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != width ||
        tensor.ne[2] != batch || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string(op) + ": invalid " + label);
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_single_parent_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
    }
}

struct ConvGeometry {
    std::int32_t width;
    std::int32_t batch;
    std::int32_t aggregate_columns;
};

ConvGeometry require_snapshot_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatch || (batch > 1 && width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: unsupported B/W domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_snapshot", "x");
    return {width, batch, width * batch};
}

ConvGeometry require_record_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 64;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width < kMinimumWidth || width > kMaximumWidth || batch <= 0 || batch > kMaximumBatch ||
        (batch > 1 && width > 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: unsupported B/T domain");
    }
    require_conv_tensor(x, hidden, width, batch, "gdn_input_proj_conv_record", "x");
    return {width, batch, width * batch};
}

void require_snapshot_operands(const Tensor& conv_weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, std::int32_t channels,
                               ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] < geometry.aggregate_columns ||
        conv_states.ne[3] != 1 || !conv_states.is_contiguous() ||
        !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: invalid convolution snapshot state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots) || !valid_selector(snapshot_base_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid state selector");
    }
    if (valid_columns.data != nullptr) {
        if (!valid_selector(valid_columns)) {
            throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid valid columns");
        }
    }
}

Tensor flatten_columns(const Tensor& tensor, std::int32_t rows, ConvGeometry geometry) {
    return Tensor(tensor.data, tensor.dtype, {rows, geometry.aggregate_columns});
}

void require_record_operands(const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             std::int32_t channels, ConvGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] <= 0 || conv_states.ne[3] != 1 ||
        !conv_states.is_contiguous() || !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid convolution state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid initial state selector");
    }
    if (valid_columns.data != nullptr && !valid_selector(valid_columns)) {
        throw std::invalid_argument("gdn_input_proj_conv_record: invalid valid columns");
    }
}

bool overlaps_range(const Tensor& tensor, const void* base, std::size_t bytes) {
    if (tensor.data == nullptr || base == nullptr || bytes == 0) { return false; }
    const auto tensor_begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    const auto range_begin  = reinterpret_cast<std::uintptr_t>(base);
    return tensor_begin < range_begin + bytes && range_begin < tensor_begin + tensor.bytes();
}

void require_record_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots, const Tensor& conv_record,
                               const Tensor& query, const Tensor& key, const Tensor& value,
                               const Tensor& z, const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{
        &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
        &conv_record, &query,       &key,         &value,         &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            if (tensors[rhs]->data != nullptr && overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_record: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_record: tensor operand overlaps live workspace");
        }
    }
}

void require_snapshot_nonoverlap(const Tensor& x, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, const Tensor& query,
                                 const Tensor& key, const Tensor& value, const Tensor& z,
                                 const WorkspaceArena& workspace) {
    const std::array<const Tensor*, 10> tensors{&x,
                                                &conv_weight,
                                                &conv_states,
                                                &valid_columns,
                                                &initial_state_slots,
                                                &snapshot_base_slots,
                                                &query,
                                                &key,
                                                &value,
                                                &z};
    for (std::size_t lhs = 0; lhs < tensors.size(); ++lhs) {
        if (tensors[lhs]->data == nullptr) { continue; }
        for (std::size_t rhs = lhs + 1; rhs < tensors.size(); ++rhs) {
            const bool shared_state_selectors =
                tensors[lhs] == &initial_state_slots && tensors[rhs] == &snapshot_base_slots;
            if (!shared_state_selectors && tensors[rhs]->data != nullptr &&
                overlaps(*tensors[lhs], *tensors[rhs])) {
                throw std::invalid_argument(
                    "gdn_input_proj_conv_snapshot: tensor operands must not overlap");
            }
        }
        if (overlaps_range(*tensors[lhs], workspace.base(), workspace.capacity())) {
            throw std::invalid_argument(
                "gdn_input_proj_conv_snapshot: tensor operand overlaps live workspace");
        }
    }
}

template <std::size_t Count>
void require_parent_nonoverlap(const Weight& weight,
                               const std::array<const Tensor*, Count>& tensors,
                               const WorkspaceArena& workspace, const char* operation) {
    for (const Tensor* tensor : tensors) {
        if (tensor->data != nullptr &&
            overlaps_range(*tensor, weight.payload,
                           static_cast<std::size_t>(weight.payload_bytes))) {
            throw std::invalid_argument(std::string(operation) +
                                        ": tensor operand overlaps parent weight");
        }
    }
    if (weight.payload != nullptr && workspace.base() != nullptr && workspace.capacity() != 0) {
        const auto weight_begin    = reinterpret_cast<std::uintptr_t>(weight.payload);
        const auto workspace_begin = reinterpret_cast<std::uintptr_t>(workspace.base());
        if (weight_begin < workspace_begin + workspace.capacity() &&
            workspace_begin < weight_begin + weight.payload_bytes) {
            throw std::invalid_argument(std::string(operation) +
                                        ": parent weight overlaps live workspace");
        }
    }
}

void require_snapshot_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                      std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: invalid B/W domain");
    }
}

void require_record_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                    std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMinimumWidth = 2;
    constexpr std::int32_t kMaximumWidth = 64;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width < kMinimumWidth ||
        max_width < min_width || max_width > kMaximumWidth || (batch_size > 1 && max_width > 16)) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: invalid B/T domain");
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool no_high   = weight.qhigh == nullptr && weight.high_plane_bytes == 0;
    const bool q4_planes = qtype != QType::Q4_G64_FP16 || no_high;
    const bool ternary = is_ternary(qtype);
    const bool t2_planes = !ternary ||
        (weight.qtype != QType::PTQ1_G128_FP16 ? no_high
         : (weight.qhigh != nullptr && weight.high_plane_bytes != 0));
    const bool q5_planes =
        qtype != QType::Q5_G64_FP16 || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    const std::int32_t group = ternary ? 128 : 64;
    // The panel layout is RowSplit's bytes permuted (sm_70 loads, read by its routes).
    if ((weight.qtype != qtype && !(ternary && is_ternary(weight.qtype))) ||
        !is_row_split(weight.layout) ||
        weight.scale_dtype != DType::FP16 ||
        weight.group_size != static_cast<std::uint32_t>(group) || weight.group != group ||
        !t2_planes || weight.ndim != 2 || weight.n != rows || weight.k != 5120 ||
        weight.shape[0] != rows || weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5_G64_FP16 && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_q8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2048 || weight.shape[0] != rows ||
        weight.shape[1] != 2048 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2048 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    // The split Q4/Q5 pair has an integer-activation route; the single-parent forms decline it
    // below, where no such route is registered for their qtypes.
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    // The NVFP4 int8-grouped route is a single-parent concern; its resolvers decide what this Op
    // admits for it.
    case LinearPolicy::AllowNvfp4Int8:
    // The cuBLAS prefill route is registered only for linear_swiglu and linear_add. Everywhere else
    // this policy means exactly what AllowA8Int means, and is accepted rather than rejected so that
    // one engine-wide setting does not have to be threaded per Op.
    case LinearPolicy::AllowPrefillCublas:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

// No single-parent qtype registers an integer-activation route, and their resolvers reject a
// policy they do not know, so the integer policies read as A16Only here.
LinearPolicy without_integer(LinearPolicy policy) {
    return allows_a8_int(policy) ? LinearPolicy::A16Only : policy;
}

// sm_70: the parents' row ranges into qkv and z through one preparation of x (an FP16 conversion,
// or the integer route's quantization where the policy admits it for both parents). `first` fills
// qkv from its head; `second`, when present, fills the rest of qkv and then z.
void sm70_project_gdn(const Tensor& x, const Weight& first, const Weight* second, Tensor& qkv,
                      Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    using detail::sm70::Destination;
    auto scope = workspace.scope();
    const bool integer =
        detail::sm70::integer_takes(first, x.ne[1], policy) &&
        (second == nullptr || detail::sm70::integer_takes(*second, x.ne[1], policy));
    const auto activation = detail::sm70::prepare_for(
        x, std::max(first.n, second != nullptr ? second->n : 0), workspace, stream, integer);
    auto* qkv_plane             = static_cast<__nv_bfloat16*>(qkv.data);
    auto* z_plane               = static_cast<__nv_bfloat16*>(z.data);
    const std::int32_t qkv_rows = qkv.ne[0];
    const std::int32_t z_rows   = z.ne[0];
    if (second == nullptr) {
        const Destination parts[] = {{qkv_plane, qkv_rows, 0, 0, qkv_rows, false},
                                     {z_plane, z_rows, 0, qkv_rows, z_rows, false}};
        detail::sm70::project_for(activation, first, parts, stream);
        return;
    }
    const std::int32_t head          = first.n;
    const std::int32_t tail_rows     = qkv_rows - head;
    const Destination first_part[]   = {{qkv_plane, qkv_rows, 0, 0, head, false}};
    const Destination second_parts[] = {{qkv_plane, qkv_rows, head, 0, tail_rows, false},
                                        {z_plane, z_rows, 0, tail_rows, z_rows, false}};
    detail::sm70::project_for(activation, first, first_part, stream);
    detail::sm70::project_for(activation, *second, second_parts, stream);
}

bool sm70_takes(const Tensor& x, const Weight& first, const Weight* second) {
    return detail::sm70::covers(first, x.ne[1]) &&
           (second == nullptr || detail::sm70::covers(*second, x.ne[1]));
}

// The Q8 parent's projection for the conv forms: the A16 routes, or sm_70's wide route.
void project_single(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

// The Q4/Q5 pair's projection for the conv forms: the A16 routes, or sm_70's wide route.
void project_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                  cudaStream_t stream);

// Whether the sm_70 routes take both parents' format (they then take every width).
bool sm70_formats(QType first, QType second, std::int32_t input_rows) {
    return detail::sm70::enabled() && detail::sm70::covered_format(first, input_rows) &&
           detail::sm70::covered_format(second, input_rows);
}

// The sm_70 routes' workspace over [min_tokens, max_tokens] where sm70_formats holds; zero where it
// does not.
std::size_t sm70_projection_bytes(QType first, QType second, std::int32_t rows,
                                  std::int32_t input_rows, std::int32_t min_tokens,
                                  std::int32_t max_tokens,
                                  LinearPolicy policy = LinearPolicy::A16Only) {
    if (!sm70_formats(first, second, input_rows)) { return 0; }
    // The integer route quantizes x for both parents or for neither.
    const LinearPolicy shared =
        detail::sm70::integer_format(first, input_rows, min_tokens, policy) &&
                detail::sm70::integer_format(second, input_rows, min_tokens, policy)
            ? policy
            : LinearPolicy::A16Only;
    return detail::sm70::covered_bytes(first, rows, input_rows, min_tokens, max_tokens, shared);
}

// The registered conv-form shapes name their parents: query/key/value 2048/2048/6144 is the
// Q4/Q5 pair over 5120 inputs, 2048/2048/4096 the Q8 parent over 2048.
std::size_t shape_sm70_bytes(bool q4_q5, std::int32_t min_columns, std::int32_t max_columns) {
    return q4_q5 ? sm70_projection_bytes(QType::Q4_G64_FP16, QType::Q5_G64_FP16, 12288, 5120,
                                         min_columns, max_columns)
                 : sm70_projection_bytes(QType::Q8_G32_FP16, QType::Q8_G32_FP16, 12288, 2048,
                                         min_columns, max_columns);
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_policy(policy);
    policy = without_integer(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    // NVFP4 and FP8 parents keep their own composites (and workspace contracts) on sm_70; NVFP4
    // artifacts need a 120a build anyway.
    if (workspace != nullptr && weight.qtype != QType::NVFP4 &&
        weight.qtype != QType::FP8_E4M3FN_ROW_BF16 && sm70_takes(x, weight, nullptr) &&
        weight.n == qkv.ne[0] + z.ne[0] && qkv.ne[1] == cols && z.ne[1] == cols) {
        sm70_project_gdn(x, weight, nullptr, qkv, z, policy, *workspace, stream);
        return;
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj: unsupported weight shape");
        }
        detail::fp8_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden  = 2048;
    constexpr std::int32_t kQkvRows = 8192;
    constexpr std::int32_t kZRows   = 4096;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);
    require_q8_rowsplit(weight, kRows, "query/key/value/z weight");
    detail::q8_gdn_input_dispatch(x, weight, qkv, z, stream);
}

// The Q4/Q5 conv forms always materialize the projection through gdn_input_proj() and run the conv
// separately. A fused projection-epilogue conv (SIMT GEMV/GEMM, T=1..3 and 5..6) served batch 1
// until 2026-09-11 and lost at every width it took: record form, RTX 3090, graph replay, cold L2,
// median of 50 (gdn_input_proj_conv_snapshot_bench), us:
//
//   T               1      2      3      5      6
//   fused        95.2  104.4  123.9  162.8  191.5
//   materialized 80.9   80.9   81.9   85.0   86.0     (snapshot form at T=1)
//
// At T=5 that was 4.8 ms of a four-draft-token MTP round.
void require_q4_q5_conv_admitted(std::int32_t tokens, std::int32_t batch_size) {
    if (!detail::q4_q5_gdn_input_admits({5120, 4096, 12288, 10240, 6144, 5120, tokens}) ||
        batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument(
            "Q4/Q5 GDN input conv: exact problem or column count is not admitted");
    }
}

detail::Q8GdnInputConvPlan resolve_q8_conv_plan(std::int32_t tokens, std::int32_t batch_size) {
    return detail::q8_gdn_input_conv_resolve_plan({2048, 8192, 4096, 12288, 2048, tokens},
                                                  batch_size);
}

struct ProjectedWorkspace {
    Tensor projected;
};

template <class Allocator>
ProjectedWorkspace allocate_projected_workspace(Allocator& allocator, std::int32_t channels,
                                                std::int32_t tokens) {
    ProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {channels, tokens});
    return out;
}

std::size_t composed_snapshot_capacity(std::int32_t channels, std::int32_t aggregate_columns,
                                       std::size_t projection_workspace_bytes) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, aggregate_columns);
    if (projection_workspace_bytes != 0) { (void)layout.alloc_bytes(projection_workspace_bytes); }
    return layout.peak_bytes(1);
}

template <class Project>
void compose_batched_snapshot(const Tensor& x, const Tensor& conv_weight, Tensor& conv_states,
                              const Tensor& valid_columns, const Tensor& initial_state_slots,
                              const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                              Tensor& value, Tensor& z, std::int32_t query_rows,
                              std::int32_t key_rows, std::int32_t value_rows, ConvGeometry geometry,
                              WorkspaceArena& workspace, cudaStream_t stream, Project&& project) {
    const std::int32_t channels = query_rows + key_rows + value_rows;
    auto scope                  = workspace.scope();
    ProjectedWorkspace scratch =
        allocate_projected_workspace(workspace, channels, geometry.aggregate_columns);

    Tensor x_flat = flatten_columns(x, x.ne[0], geometry);
    Tensor z_flat = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, scratch.projected, z_flat);

    Tensor projected(scratch.projected.data, DType::BF16,
                     {channels, geometry.width, geometry.batch});
    detail::gdn_projected_conv_snapshot_launch(projected, conv_weight, conv_states, valid_columns,
                                               initial_state_slots, snapshot_base_slots, query, key,
                                               value, stream);
}

template <class Project>
void compose_record(const Tensor& x, const Tensor& conv_weight, const Tensor& conv_states,
                    const Tensor& valid_columns, const Tensor& initial_state_slots,
                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                    ConvGeometry geometry, WorkspaceArena& workspace, cudaStream_t stream,
                    Project&& project) {
    auto scope         = workspace.scope();
    Tensor x_flat      = flatten_columns(x, x.ne[0], geometry);
    Tensor record_flat = flatten_columns(conv_record, conv_record.ne[0], geometry);
    Tensor z_flat      = flatten_columns(z, z.ne[0], geometry);
    project(x_flat, record_flat, z_flat);
    detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
}

void dispatch_single_parent_snapshot(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                     Tensor& value, Tensor& z, LinearPolicy policy,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (geometry.batch > 1) {
            if (plan.schedule != detail::Nvfp4GdnConvScheduleId::Materialized) {
                throw std::logic_error("batched NVFP4 GDN conv selected a fused schedule");
            }
            compose_batched_snapshot(x, conv_weight, conv_states, valid_columns,
                                     initial_state_slots, snapshot_base_slots, query, key, value, z,
                                     kQueryRows, kKeyRows, kValueRows, geometry, workspace, stream,
                                     [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                         gdn_input_proj(x_flat, weight, projected, z_flat, policy,
                                                        workspace, stream);
                                     });
            return;
        }
        detail::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                            initial_state_slots, snapshot_base_slots, query, key,
                                            value, z, policy, workspace, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "fp8 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_snapshot", "z");
        require_snapshot_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                    snapshot_base_slots, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{&x,
                                                    &conv_weight,
                                                    &conv_states,
                                                    &valid_columns,
                                                    &initial_state_slots,
                                                    &snapshot_base_slots,
                                                    &query,
                                                    &key,
                                                    &value,
                                                    &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_snapshot");
        detail::fp8_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_state_slots, snapshot_base_slots, query, key,
                                          value, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_snapshot_input(x, kHidden);
    require_q8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");
    if (geometry.batch > 1) {
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, kQueryRows, kKeyRows, kValueRows, geometry, workspace, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                project_single(x_flat, weight, projected, z_flat, policy, workspace, stream);
            });
        return;
    }

    // sm_70 materializes every width: its projection routes beat the emulated fused kernels.
    const detail::Q8GdnInputConvPlan plan =
        detail::sm70::enabled() ? detail::Q8GdnInputConvPlan{}
                                : resolve_q8_conv_plan(geometry.width, geometry.batch);
    if (!detail::sm70::enabled() &&
        plan.schedule == detail::Q8GdnInputConvScheduleId::DecodeFused) {
        detail::q8_gdn_input_decode_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }
    if (!detail::sm70::enabled() &&
        plan.schedule == detail::Q8GdnInputConvScheduleId::SplitKMmaFused) {
        detail::q8_gdn_input_splitk_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope                 = workspace.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(workspace, kChannels, geometry.width);
    project_single(x, weight, scratch.projected, z, policy, workspace, stream);
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void dispatch_single_parent_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& conv_record,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   LinearPolicy policy, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);

        const detail::Nvfp4GdnConvPlan plan =
            detail::nvfp4_gdn_conv_resolve_plan(policy, geometry.width, geometry.batch);
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::Materialized && geometry.batch > 1) {
            compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                           conv_record, query, key, value, z, geometry, workspace, stream,
                           [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                               gdn_input_proj(x_flat, weight, record_flat, z_flat, policy,
                                              workspace, stream);
                           });
            return;
        }
        if (plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) {
            detail::nvfp4_gdn_record_small_t_launch(x, weight, conv_weight, conv_states,
                                                    valid_columns, initial_state_slots, conv_record,
                                                    query, key, value, z, stream);
            return;
        }

        auto scope = workspace.scope();
        gdn_input_proj(x, weight, conv_record, z, policy, workspace, stream);
        detail::nvfp4_gdn_record_post_launch(conv_record, conv_weight, conv_states, valid_columns,
                                             initial_state_slots, query, key, value, stream);
        return;
    }

    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const ConvGeometry geometry        = require_record_input(x, kHidden);
        detail::validate_fp8_weight(weight, "fp8 gdn_input_proj_conv_record");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument("fp8 gdn_input_proj_conv_record: unsupported weight shape");
        }
        require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                kChannels, geometry);
        require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "conv record");
        require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "query");
        require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "key");
        require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                            "gdn_input_proj_conv_record", "value");
        require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                            "z");
        require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                  conv_record, query, key, value, z, workspace);
        const std::array<const Tensor*, 10> tensors{
            &x,           &conv_weight, &conv_states, &valid_columns, &initial_state_slots,
            &conv_record, &query,       &key,         &value,         &z};
        require_parent_nonoverlap(weight, tensors, workspace, "fp8 gdn_input_proj_conv_record");
        detail::fp8_gdn_record_dispatch(x, weight, conv_weight, conv_states, valid_columns,
                                        initial_state_slots, conv_record, query, key, value, z,
                                        policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const ConvGeometry geometry       = require_record_input(x, kHidden);
    require_q8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    if (geometry.batch > 1 || geometry.width > 16 || detail::sm70::enabled()) {
        compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                       query, key, value, z, geometry, workspace, stream,
                       [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                           project_single(x_flat, weight, record_flat, z_flat, policy, workspace,
                                          stream);
                       });
        return;
    }
    const detail::Q8GdnInputConvPlan plan = resolve_q8_conv_plan(geometry.width, geometry.batch);
    if (plan.schedule != detail::Q8GdnInputConvScheduleId::SplitKMmaFused) {
        throw std::logic_error("Q8 ReplaySSM record domain selected a non-record schedule");
    }
    detail::q8_gdn_input_splitk_conv_record_launch(x, weight, conv_weight, conv_states,
                                                   valid_columns, initial_state_slots, conv_record,
                                                   query, key, value, z, stream);
}

} // namespace

namespace {

// Shared shape checks for the split form, which both overloads owe their callers.
void require_split_profile(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                           Tensor& qkv, Tensor& z) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t cols            = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_rowsplit(qk_weight, QType::Q4_G64_FP16, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5_G64_FP16, kParentRows, "value/z weight");
}

bool t2_two_parent(const Weight& qk_weight, const Weight& value_z_weight) {
    const bool qk = is_ternary(qk_weight.qtype);
    if (qk != is_ternary(value_z_weight.qtype) ||
        (qk && qk_weight.qtype != value_z_weight.qtype)) {
        throw std::invalid_argument("gdn_input_proj: the two ternary parents must share one format");
    }
    return qk;
}

// Ternary two-parent projection. On the integer route both parents take one quantisation of x and
// write straight into qkv and z. Otherwise q/k come from the whole qk parent and value/z from row
// views of the value/z parent, each through the T2 linear routes, and the q/k and value planes are
// assembled into the caller's [10240,T] qkv with two strided copies; z is written directly.
void t2_two_parent_project(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                           Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    constexpr std::int32_t kQkRows    = 4096;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kQkvRows   = kQkRows + kValueRows;
    constexpr std::size_t kBytes      = 2;
    const std::int32_t cols           = x.ne[1];
    require_matrix(x, 5120, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_rowsplit(qk_weight, QType::T2_G128_FP16, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::T2_G128_FP16, kValueRows + kZRows, "value/z weight");
    if (qk_weight.qtype == QType::T5_G128_FP16) {
        Tensor qk = qkv.slice(0, 0, kQkRows);
        Tensor value = qkv.slice(0, kQkRows, kValueRows);
        Tensor* first[]{&qk};
        Tensor* second[]{&value, &z};
        detail::t5_project(x, qk_weight, first, false, policy, &workspace, stream);
        detail::t5_project(x, value_z_weight, second, false, policy, &workspace, stream);
        return;
    }
    if (sm70_takes(x, qk_weight, &value_z_weight)) {
        sm70_project_gdn(x, qk_weight, &value_z_weight, qkv, z, policy, workspace, stream);
        return;
    }
    auto scope = workspace.scope();
    if (detail::t2_a8_admits(policy) && detail::t2_a8_supported(qk_weight, cols) &&
        detail::t2_a8_supported(value_z_weight, cols)) {
        // One quantisation of x; q/k and value land in qkv directly and z in its own plane.
        const auto activations = detail::t2_a8_quantize(x, workspace, stream, qk_weight.qtype == QType::PTQ1_G128_FP16);
        detail::t2_a8_project_split_pair(activations, {qk_weight, qkv, 0, kQkRows, qkv, 0},
                                         {value_z_weight, qkv, kQkRows, kValueRows, z, 0}, stream);
        return;
    }
    Tensor qk_plane    = workspace.alloc(DType::BF16, {kQkRows, cols});
    Tensor value_plane = workspace.alloc(DType::BF16, {kValueRows, cols});
    linear(x, qk_weight, qk_plane, policy, workspace, stream);
    linear(x, detail::t2_row_view(value_z_weight, 0, kValueRows), value_plane, policy, workspace,
           stream);
    linear(x, detail::t2_row_view(value_z_weight, kValueRows, kZRows), z, policy, workspace,
           stream);
    auto* destination = static_cast<std::uint8_t*>(qkv.data);
    CUDA_CHECK(cudaMemcpy2DAsync(destination, kQkvRows * kBytes, qk_plane.data, kQkRows * kBytes,
                                 kQkRows * kBytes, static_cast<std::size_t>(cols),
                                 cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(destination + kQkRows * kBytes, kQkvRows * kBytes,
                                 value_plane.data, kValueRows * kBytes, kValueRows * kBytes,
                                 static_cast<std::size_t>(cols), cudaMemcpyDeviceToDevice, stream));
}

// Transient storage of t2_two_parent_project over each width in [min_columns, max_columns]: the
// integer route's activation planes where it takes the width, otherwise the q/k and value planes of
// the A16 route (whose T2 linears take no workspace).
std::size_t t2_two_parent_projection_bytes(std::int32_t min_columns, std::int32_t max_columns,
                                           LinearPolicy policy = LinearPolicy::A16Only,
                                           QType qtype = QType::T2_G128_FP16) {
    if (qtype == QType::T5_G128_FP16) {
        return detail::t5_workspace_capacity_bytes(policy, 5120, max_columns);
    }
    // sm_70 projects both parents through its own routes at every width.
    if (detail::sm70::enabled() && qtype == QType::T2_G128_FP16) {
        return sm70_projection_bytes(QType::T2_G128_FP16, QType::T2_G128_FP16, 12288, 5120,
                                     min_columns, max_columns, policy);
    }
    std::size_t bytes = 0;
    for (std::int32_t columns = min_columns; columns <= max_columns; ++columns) {
        WorkspaceLayoutBuilder layout;
        if (!detail::t2_a8_admits(policy) ||
            !detail::t2_a8_layout_activations(layout, 5120, columns, qtype == QType::PTQ1_G128_FP16)) {
            (void)layout.alloc(DType::BF16, {4096, columns});
            (void)layout.alloc(DType::BF16, {6144, columns});
        }
        bytes = std::max(bytes, layout.peak_bytes(1));
    }
    return bytes;
}

} // namespace

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    validate_policy(policy);
    if (t2_two_parent(qk_weight, value_z_weight)) {
        if (x.ne[1] <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
        t2_two_parent_project(x, qk_weight, value_z_weight, qkv, z, policy, workspace, stream);
        return;
    }
    require_split_profile(x, qk_weight, value_z_weight, qkv, z);
    if (sm70_takes(x, qk_weight, &value_z_weight)) {
        sm70_project_gdn(x, qk_weight, &value_z_weight, qkv, z, policy, workspace, stream);
        return;
    }
    // qk fills the head of `qkv`; value_z fills its tail and, from its own second row range, `z`.
    // Both parents read the same activations, so one quantisation serves all three destinations.
    if (allows_cublas_prefill(policy) &&
        detail::cublas_prefill_takes(detail::kCublasGdnInput, x.ne[1])) {
        const std::int32_t qkv_rows   = qkv.ne[0];
        const std::int32_t z_rows     = z.ne[0];
        const std::int32_t qk_rows    = qk_weight.n;
        const std::int32_t value_rows = qkv_rows - qk_rows;
        const detail::CublasProjectionDestination qk_dests[] = {
            {qkv.data, 0, qk_rows, qkv_rows, 0}};
        const detail::CublasProjectionDestination vz_dests[] = {
            {qkv.data, 0, value_rows, qkv_rows, qk_rows},
            {z.data, value_rows, z_rows, z_rows, 0}};
        const detail::CublasProjection parents[] = {
            {&qk_weight, qk_dests, 1}, {&value_z_weight, vz_dests, 2}};
        if (value_rows > 0 && detail::w4_cublas_projection_supported(parents, 2, x.ne[1])) {
            detail::w4_cublas_projection_launch(x, parents, 2, workspace, stream);
            return;
        }
    }
    if (allows_a8_int(policy) &&
        detail::q4_q5_gdn_input_a8_supported(qk_weight, value_z_weight, x.ne[1])) {
        detail::q4_q5_gdn_input_a8_launch(x, qk_weight, value_z_weight, qkv, z, workspace, stream);
        return;
    }
    detail::q4_q5_gdn_input_dispatch(x, qk_weight, value_z_weight, qkv, z, stream);
}

std::size_t gdn_input_proj_split_workspace_capacity_bytes(
    QType qk_qtype, std::int32_t qk_rows, QType value_z_qtype, std::int32_t value_z_rows,
    std::int32_t input_rows, LinearPolicy policy, std::int32_t min_tokens,
    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (is_ternary(qk_qtype) || is_ternary(value_z_qtype)) {
        if (qk_qtype != value_z_qtype || qk_rows != 4096 || value_z_rows != 12288 ||
            input_rows != 5120) {
            throw std::invalid_argument("gdn_input_proj workspace: unregistered T2 pair");
        }
        return t2_two_parent_projection_bytes(min_tokens, max_tokens, policy, qk_qtype);
    }
    const bool registered = qk_qtype == QType::Q4_G64_FP16 && qk_rows == 4096 &&
                            value_z_qtype == QType::Q5_G64_FP16 && value_z_rows == 12288 &&
                            input_rows == 5120;
    const auto routes = [&](std::int32_t lo, std::int32_t hi) -> std::size_t {
        if (!allows_a8_int(policy) || !registered) { return 0; }
        const std::size_t a8 = detail::q4_q5_gdn_input_a8_workspace_capacity_bytes(lo, hi);
        const std::int32_t first =
            allows_cublas_prefill(policy)
                ? detail::cublas_prefill_first_width(detail::kCublasGdnInput, lo, hi)
                : 0;
        if (first != 0) {
            return std::max(a8, detail::w4_cublas_projection_workspace_capacity_bytes(
                                    std::max(qk_rows, value_z_rows), input_rows, first, hi));
        }
        return a8;
    };
    if (sm70_formats(qk_qtype, value_z_qtype, input_rows)) {
        return sm70_projection_bytes(qk_qtype, value_z_qtype, std::max(qk_rows, value_z_rows),
                                     input_rows, min_tokens, max_tokens, policy);
    }
    return routes(min_tokens, max_tokens);
}

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t cols            = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    if (t2_two_parent(qk_weight, value_z_weight)) {
        throw std::invalid_argument("gdn_input_proj: the T2 pair needs the workspace overload");
    }
    require_rowsplit(qk_weight, QType::Q4_G64_FP16, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5_G64_FP16, kParentRows, "value/z weight");

    detail::q4_q5_gdn_input_dispatch(x, qk_weight, value_z_weight, qkv, z, stream);
}

namespace {

std::size_t single_parent_route_bytes(QType parent_qtype, std::int32_t parent_rows,
                                      std::int32_t input_rows, LinearPolicy policy,
                                      std::int32_t min_tokens, std::int32_t max_tokens) {
    if (parent_qtype == QType::NVFP4) {
        if (parent_rows != detail::Nvfp4N16384K5120::kOutputRows ||
            input_rows != detail::Nvfp4N16384K5120::kInputRows) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (parent_rows != detail::Fp8N16384K5120::kOutputRows ||
            input_rows != detail::Fp8N16384K5120::kInputRows) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported FP8 profile");
        }
        return detail::fp8_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (parent_qtype == QType::Q8_G32_FP16 && parent_rows == 12288 && input_rows == 2048) {
        (void)detail::q8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, min_tokens});
        (void)detail::q8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, max_tokens});
        return 0;
    }
    throw std::invalid_argument("gdn_input_proj workspace: unsupported parent profile");
}

} // namespace

std::size_t gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    // NVFP4 and FP8 parents keep their own routes (dispatch_single_parent).
    if (parent_qtype != QType::NVFP4 && parent_qtype != QType::FP8_E4M3FN_ROW_BF16 &&
        sm70_formats(parent_qtype, parent_qtype, input_rows)) {
        return sm70_projection_bytes(parent_qtype, parent_qtype, parent_rows, input_rows,
                                     min_tokens, max_tokens, without_integer(policy));
    }
    return single_parent_route_bytes(parent_qtype, parent_rows, input_rows, policy, min_tokens,
                                     max_tokens);
}


void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, policy, &workspace, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only, nullptr,
                           stream);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool q8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !q8) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: unregistered shape");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t channels = query_rows + key_rows + value_rows;
    if (batch_size > 1) {
        if (q4_q5) {
            require_q4_q5_conv_admitted(min_width, batch_size);
            require_q4_q5_conv_admitted(max_width, batch_size);
        } else {
            (void)resolve_q8_conv_plan(min_width, batch_size);
            (void)resolve_q8_conv_plan(max_width, batch_size);
        }
        return composed_snapshot_capacity(
            channels, batch_size * max_width,
            shape_sm70_bytes(q4_q5, batch_size * min_width, batch_size * max_width));
    }

    std::int32_t largest_materialized_width = 0;
    if (q4_q5) {
        require_q4_q5_conv_admitted(min_width, 1);
        require_q4_q5_conv_admitted(max_width, 1);
        largest_materialized_width = max_width;
    } else {
        (void)resolve_q8_conv_plan(min_width, 1);
        (void)resolve_q8_conv_plan(max_width, 1);
        if (max_width >= 17 || detail::sm70::enabled()) { largest_materialized_width = max_width; }
    }
    if (largest_materialized_width == 0) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected_workspace(layout, channels, largest_materialized_width);
    if (const std::size_t sm70 = shape_sm70_bytes(q4_q5, min_width, largest_materialized_width);
        sm70 != 0) {
        (void)layout.alloc_bytes(sm70);
    }
    return layout.peak_bytes(1);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16 &&
        parent_rows == detail::Fp8N16384K5120::kOutputRows &&
        input_rows == detail::Fp8N16384K5120::kInputRows) {
        return detail::fp8_gdn_snapshot_workspace_capacity_bytes(policy, batch_size, min_width,
                                                                 max_width);
    }
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4N16384K5120::kOutputRows ||
        input_rows != detail::Nvfp4N16384K5120::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot workspace: unsupported single-parent profile");
    }
    if (batch_size == 1) {
        return detail::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, min_width, max_width);
    }

    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    (void)detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);

    constexpr std::int32_t kChannels       = 10240;
    const std::int32_t aggregate_columns   = batch_size * max_width;
    const std::size_t projection_workspace = gdn_input_proj_workspace_capacity_bytes(
        parent_qtype, parent_rows, input_rows, policy, batch_size * min_width, aggregate_columns);
    return composed_snapshot_capacity(kChannels, aggregate_columns, projection_workspace);
}

std::size_t gdn_input_proj_split_conv_snapshot_workspace_capacity_bytes(QType qk_qtype,
                                                                        QType value_z_qtype,
                                                                        std::int32_t batch_size,
                                                                        std::int32_t min_width,
                                                                        std::int32_t max_width) {
    return gdn_input_proj_split_conv_snapshot_workspace_capacity_bytes(
        qk_qtype, value_z_qtype, LinearPolicy::A16Only, batch_size, min_width, max_width);
}

std::size_t gdn_input_proj_split_conv_snapshot_workspace_capacity_bytes(
    QType qk_qtype, QType value_z_qtype, LinearPolicy policy, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    if (is_ternary(qk_qtype) && qk_qtype == value_z_qtype) {
        require_snapshot_capacity_domain(batch_size, min_width, max_width);
        const std::int32_t columns = std::max(batch_size, 1) * max_width;
        return composed_snapshot_capacity(
            2048 + 2048 + 6144, columns,
            t2_two_parent_projection_bytes(std::max(batch_size, 1) * min_width, columns, policy, qk_qtype));
    }
    if (qk_qtype == QType::Q4_G64_FP16 && value_z_qtype == QType::Q5_G64_FP16) {
        return gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, batch_size,
                                                                     min_width, max_width);
    }
    throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: unregistered pair");
}

std::size_t gdn_input_proj_split_conv_record_workspace_capacity_bytes(QType qk_qtype,
                                                                      QType value_z_qtype,
                                                                      std::int32_t batch_size,
                                                                      std::int32_t min_width,
                                                                      std::int32_t max_width) {
    return gdn_input_proj_split_conv_record_workspace_capacity_bytes(
        qk_qtype, value_z_qtype, LinearPolicy::A16Only, batch_size, min_width, max_width);
}

std::size_t gdn_input_proj_split_conv_record_workspace_capacity_bytes(
    QType qk_qtype, QType value_z_qtype, LinearPolicy policy, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    if (is_ternary(qk_qtype) && qk_qtype == value_z_qtype) {
        require_record_capacity_domain(batch_size, min_width, max_width);
        return t2_two_parent_projection_bytes(std::max(batch_size, 1) * min_width,
                                              std::max(batch_size, 1) * max_width, policy, qk_qtype);
    }
    if (qk_qtype == QType::Q4_G64_FP16 && value_z_qtype == QType::Q5_G64_FP16) {
        return gdn_input_proj_conv_record_workspace_capacity_bytes(2048, 2048, 6144, batch_size,
                                                                   min_width, max_width);
    }
    throw std::invalid_argument("gdn_input_proj_conv_record workspace: unregistered pair");
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool q8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !q8) {
        throw std::invalid_argument("gdn_input_proj_conv_record workspace: unregistered shape");
    }
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (q4_q5) {
        require_q4_q5_conv_admitted(min_width, batch_size);
        require_q4_q5_conv_admitted(max_width, batch_size);
    } else {
        (void)resolve_q8_conv_plan(min_width, batch_size);
        (void)resolve_q8_conv_plan(max_width, batch_size);
    }
    // Both records project through gdn_input_proj, which takes the sm_70 routes there.
    const std::int32_t batch = std::max(batch_size, 1);
    return shape_sm70_bytes(q4_q5, batch * min_width, batch * max_width);
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    require_record_capacity_domain(batch_size, min_width, max_width);
    if (parent_qtype == QType::FP8_E4M3FN_ROW_BF16 &&
        parent_rows == detail::Fp8N16384K5120::kOutputRows &&
        input_rows == detail::Fp8N16384K5120::kInputRows) {
        return detail::fp8_gdn_record_workspace_capacity_bytes(policy, batch_size, min_width,
                                                               max_width);
    }
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4N16384K5120::kOutputRows ||
        input_rows != detail::Nvfp4N16384K5120::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_record workspace: unsupported single-parent profile");
    }
    const detail::Nvfp4GdnConvPlan minimum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, min_width, batch_size);
    const detail::Nvfp4GdnConvPlan maximum_plan =
        detail::nvfp4_gdn_conv_resolve_plan(policy, max_width, batch_size);
    if (batch_size == 1) {
        if (minimum_plan.schedule == detail::Nvfp4GdnConvScheduleId::DecodeFusedA16) {
            throw std::logic_error("ReplaySSM record planner admitted NVFP4 decode");
        }
        if (maximum_plan.schedule == detail::Nvfp4GdnConvScheduleId::SmallTFusedA16) { return 0; }
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, std::max(min_width, 4),
                                                                max_width);
    }
    return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, batch_size * min_width,
                                                            batch_size * max_width);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    gdn_input_proj_conv_snapshot(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                 valid_columns, initial_state_slots, snapshot_base_slots, query,
                                 key, value, z, LinearPolicy::A16Only, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    validate_policy(policy);
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_snapshot_input(x, kHidden);
    const bool ternary                 = t2_two_parent(qk_weight, value_z_weight);
    require_rowsplit(qk_weight, ternary ? QType::T2_G128_FP16 : QType::Q4_G64_FP16,
                     kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, ternary ? QType::T2_G128_FP16 : QType::Q5_G64_FP16,
                     kParentRows, "value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_snapshot",
                        "z");

    if (geometry.batch > 1) {
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, kQueryRows, kKeyRows, kValueRows, geometry, ws, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                if (ternary) {
                    t2_two_parent_project(x_flat, qk_weight, value_z_weight, projected, z_flat,
                                          policy, ws, stream);
                } else {
                    project_pair(x_flat, qk_weight, value_z_weight, projected, z_flat, policy, ws,
                                 stream);
                }
            });
        return;
    }

    if (!ternary) { require_q4_q5_conv_admitted(geometry.width, geometry.batch); }
    auto scope                 = ws.scope();
    ProjectedWorkspace scratch = allocate_projected_workspace(ws, kChannels, geometry.width);
    if (ternary) {
        t2_two_parent_project(x, qk_weight, value_z_weight, scratch.projected, z, policy, ws,
                              stream);
    } else {
        project_pair(x, qk_weight, value_z_weight, scratch.projected, z, policy, ws, stream);
    }
    detail::gdn_projected_conv_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                               valid_columns, initial_state_slots,
                                               snapshot_base_slots, query, key, value, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    gdn_input_proj_conv_record(x, qk_weight, value_z_weight, conv_weight, conv_states,
                               valid_columns, initial_state_slots, conv_record, query, key, value,
                               z, LinearPolicy::A16Only, workspace, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    validate_policy(policy);
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const ConvGeometry geometry        = require_record_input(x, kHidden);
    const bool ternary                 = t2_two_parent(qk_weight, value_z_weight);
    require_rowsplit(qk_weight, ternary ? QType::T2_G128_FP16 : QType::Q4_G64_FP16,
                     kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, ternary ? QType::T2_G128_FP16 : QType::Q5_G64_FP16,
                     kParentRows, "value/z weight");
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots, kChannels,
                            geometry);
    require_conv_tensor(conv_record, kChannels, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, kQueryRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, kKeyRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "key");
    require_conv_tensor(value, kValueRows, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, kZRows, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);

    if (!ternary) { require_q4_q5_conv_admitted(geometry.width, geometry.batch); }
    compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                   query, key, value, z, geometry, workspace, stream,
                   [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                       if (ternary) {
                           t2_two_parent_project(x_flat, qk_weight, value_z_weight, record_flat,
                                                 z_flat, policy, workspace, stream);
                       } else {
                           project_pair(x_flat, qk_weight, value_z_weight, record_flat, z_flat,
                                        policy, workspace, stream);
                       }
                   });
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, policy, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, LinearPolicy::A16Only, ws, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, policy, workspace, stream);
}

void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent_record(x, query_key_value_z_weight, conv_weight, conv_states,
                                  valid_columns, initial_state_slots, conv_record, query, key,
                                  value, z, LinearPolicy::A16Only, workspace, stream);
}

namespace {

// The rows a GGUF projection's parts cover: q, k and value into the convolution input (output 0),
// z beside it (output 1). Value and z have the same rows, so q and k split the rest evenly.
struct GgufGdnGeometry {
    std::int32_t hidden    = 0;
    std::int32_t qkv       = 0;
    std::int32_t z         = 0;
    std::int32_t query_key = 0;
    std::int32_t value     = 0;
};

std::vector<detail::GgufShape> gguf_shapes(const GgufProjectionWeights& weights) {
    std::vector<detail::GgufShape> shapes;
    for (const auto& part : weights.parts) {
        shapes.push_back({part.weight.qtype, part.weight.n, part.weight.k});
    }
    return shapes;
}

// Checks that the parts tile q/k/v and z without gaps or overlaps and returns their geometry.
GgufGdnGeometry require_gguf_gdn(const GgufProjectionWeights& weights) {
    if (weights.parts.empty()) {
        throw std::invalid_argument("gdn_input_proj: a GGUF projection needs parts");
    }
    GgufGdnGeometry geometry;
    geometry.hidden = weights.parts.front().weight.k;
    std::int64_t covered[2] = {0, 0};
    for (const auto& part : weights.parts) {
        detail::require_gguf(part.weight, "gdn_input_proj GGUF part");
        if (part.weight.k != geometry.hidden || part.output < 0 || part.output > 1 ||
            part.row != covered[part.output] || part.weight.n <= 0) {
            throw std::invalid_argument("gdn_input_proj: GGUF parts do not tile q/k/v and z");
        }
        covered[part.output] += part.weight.n;
    }
    const auto key = (covered[0] - covered[1]) / 2;
    if (geometry.hidden <= 0 || key <= 0 || key % 128 || covered[1] <= 0 ||
        covered[1] % key || 2 * key + covered[1] != covered[0] || covered[0] > INT32_MAX) {
        throw std::invalid_argument("gdn_input_proj: invalid GGUF q/k/v/z geometry");
    }
    geometry.qkv       = static_cast<std::int32_t>(covered[0]);
    geometry.z         = static_cast<std::int32_t>(covered[1]);
    geometry.value     = geometry.z;
    geometry.query_key = (geometry.qkv - geometry.z) / 2;
    if (geometry.z <= 0 || geometry.query_key <= 0 ||
        2 * geometry.query_key + geometry.value != geometry.qkv) {
        throw std::invalid_argument("gdn_input_proj: GGUF parts do not cover q/k/v and z");
    }
    return geometry;
}

void gguf_gdn_project(const Tensor& x, const GgufProjectionWeights& weights, Tensor& qkv,
                      Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    std::vector<detail::GgufProduct> products;
    products.reserve(weights.parts.size());
    for (const auto& part : weights.parts) {
        detail::GgufProduct product;
        product.weight = &part.weight;
        product.out    = part.output == 0 ? &qkv : &z;
        product.row    = part.row;
        products.push_back(product);
    }
    detail::gguf_project(x, products, workspace, stream);
}

} // namespace

std::size_t gdn_input_proj_workspace_capacity_bytes(const GgufProjectionWeights& weights,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    (void)require_gguf_gdn(weights);
    const auto shapes = gguf_shapes(weights);
    return detail::gguf_project_workspace_bytes(shapes, min_tokens, max_tokens);
}

void gdn_input_proj(const Tensor& x, const GgufProjectionWeights& weights, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    const GgufGdnGeometry rows = require_gguf_gdn(weights);
    const std::int32_t cols    = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, rows.hidden, cols, "x");
    require_matrix(qkv, rows.qkv, cols, "qkv");
    require_matrix(z, rows.z, cols, "z");
    gguf_gdn_project(x, weights, qkv, z, workspace, stream);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    const GgufProjectionWeights& weights, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width) {
    const GgufGdnGeometry rows = require_gguf_gdn(weights);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const auto shapes          = gguf_shapes(weights);
    const std::int32_t batch   = std::max(batch_size, 1);
    const std::size_t project  = detail::gguf_project_workspace_bytes(
        shapes, batch * min_width, batch * max_width);
    return composed_snapshot_capacity(rows.qkv, batch * max_width, project);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const GgufProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    const GgufGdnGeometry rows  = require_gguf_gdn(weights);
    const ConvGeometry geometry = require_snapshot_input(x, rows.hidden);
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, rows.qkv, geometry);
    require_conv_tensor(query, rows.query_key, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "query");
    require_conv_tensor(key, rows.query_key, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "key");
    require_conv_tensor(value, rows.value, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "value");
    require_conv_tensor(z, rows.z, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_snapshot", "z");
    compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                             snapshot_base_slots, query, key, value, z, rows.query_key,
                             rows.query_key, rows.value, geometry, ws, stream,
                             [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                 gguf_gdn_project(x_flat, weights, projected, z_flat, ws, stream);
                             });
}

std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    const GgufProjectionWeights& weights, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width) {
    (void)require_gguf_gdn(weights);
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const auto shapes        = gguf_shapes(weights);
    const std::int32_t batch = std::max(batch_size, 1);
    return detail::gguf_project_workspace_bytes(shapes, batch * min_width, batch * max_width);
}

void gdn_input_proj_conv_record(const Tensor& x, const GgufProjectionWeights& weights,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    const GgufGdnGeometry rows  = require_gguf_gdn(weights);
    const ConvGeometry geometry = require_record_input(x, rows.hidden);
    require_record_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                            rows.qkv, geometry);
    require_conv_tensor(conv_record, rows.qkv, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "conv record");
    require_conv_tensor(query, rows.query_key, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "query");
    require_conv_tensor(key, rows.query_key, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "key");
    require_conv_tensor(value, rows.value, geometry.width, geometry.batch,
                        "gdn_input_proj_conv_record", "value");
    require_conv_tensor(z, rows.z, geometry.width, geometry.batch, "gdn_input_proj_conv_record",
                        "z");
    require_record_nonoverlap(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                              conv_record, query, key, value, z, workspace);
    compose_record(x, conv_weight, conv_states, valid_columns, initial_state_slots, conv_record,
                   query, key, value, z, geometry, workspace, stream,
                   [&](const Tensor& x_flat, Tensor& record_flat, Tensor& z_flat) {
                       gguf_gdn_project(x_flat, weights, record_flat, z_flat, workspace, stream);
                   });
}

namespace {

void project_single(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    if (sm70_takes(x, weight, nullptr)) {
        gdn_input_proj(x, weight, qkv, z, policy, workspace, stream);
        return;
    }
    gdn_input_proj(x, weight, qkv, z, stream);
}

void project_pair(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                  Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    if (sm70_takes(x, qk_weight, &value_z_weight)) {
        gdn_input_proj(x, qk_weight, value_z_weight, qkv, z, policy, workspace, stream);
        return;
    }
    gdn_input_proj(x, qk_weight, value_z_weight, qkv, z, stream);
}

} // namespace

} // namespace ninfer::ops
