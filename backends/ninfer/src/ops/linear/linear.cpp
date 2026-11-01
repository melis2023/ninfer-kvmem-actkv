#include "ops/linear/t5/t5_project.h"

#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/gguf/gguf_linear.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/ptq1/ptq1_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/sm70/sm70_integer.h"
#include "ops/linear/sm70/sm70_project.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/linear/t2/t2_dispatch.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// The integer-A8 policies reach a plain linear for T2 weights, whose integer prefill route lives
// here rather than in a fused op; every other format's dispatcher treats them as A16.
void validate_linear_policy(LinearPolicy policy) {
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("linear: invalid compute policy");
    }
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 ||
        (out.dtype != DType::BF16 &&
         !(out.dtype == DType::FP32 && (w.qtype == QType::BF16 || is_gguf(w.qtype))))) {
        throw std::invalid_argument(
            "linear: x must be BF16 and out BF16 (or FP32 for BF16 and GGUF weights)");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

// The sm_70 routes take every dense format; without an arena, the widths that need none.
// The sm_70 routes take every dense format, but not the NVFP4 int8 family: that policy is the
// caller telling this Op which kernels it wants, and sm_70's FP16 route is not one of them.
bool sm70_route(const Tensor& x, const Weight& w, const Tensor& out, LinearPolicy policy,
                const WorkspaceArena* workspace) {
    return policy != LinearPolicy::AllowNvfp4Int8 && out.dtype == DType::BF16 &&
           detail::sm70::covers(w, x.ne[1]) &&
           (workspace != nullptr || detail::sm70::direct_tokens(x.ne[1]));
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    if (sm70_route(x, w, out, policy, workspace)) {
        const detail::sm70::Destination all{
            static_cast<__nv_bfloat16*>(out.data), w.n, 0, 0, w.n, false};
        if (workspace == nullptr) {
            detail::sm70::project_for(detail::sm70::prepare_direct(x), w, {&all, 1}, stream);
            return;
        }
        auto scope            = workspace->scope();
        const auto activation = detail::sm70::prepare_for(
            x, w.n, *workspace, stream, detail::sm70::integer_takes(w, x.ne[1], policy));
        detail::sm70::project_for(activation, w, {&all, 1}, stream);
        return;
    }
    if (is_gguf(w.qtype)) {
        if (workspace == nullptr) {
            throw std::invalid_argument("linear: a GGUF weight needs the workspace overload");
        }
        detail::gguf_linear(x, w, out, *workspace, stream);
        return;
    }
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, stream);
        return;
    case QType::T5_G128_FP16: {
        Tensor* outputs[]{&out};
        detail::t5_project(x, w, outputs, false, policy, workspace, stream);
        return;
    }
    case QType::PTQ1_G128_FP16:
        if (workspace != nullptr && detail::t2_a8_admits(policy) && detail::t2_a8_supported(w, x.ne[1])) {
            detail::t2_a8_linear(x, w, out, *workspace, stream);
            return;
        }
        detail::ptq1_dispatch(x, w, out, policy, stream);
        return;
    case QType::T2_G128_FP16:
        if (workspace != nullptr && detail::t2_a8_admits(policy) &&
            detail::t2_a8_supported(w, x.ne[1])) {
            detail::t2_a8_linear(x, w, out, *workspace, stream);
            return;
        }
        detail::t2_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32:
    case QType::INT32:
    default:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

std::size_t linear_route_workspace_bytes(QType qtype, std::int32_t output_rows,
                                         std::int32_t input_rows, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::T5_G128_FP16:
        if (output_rows <= 0) { throw std::invalid_argument("linear workspace: T5 rows must be positive"); }
        return detail::t5_workspace_capacity_bytes(policy, input_rows, max_tokens);
    case QType::PTQ1_G128_FP16:
        detail::validate_ptq1_shape(output_rows, input_rows, min_tokens, policy);
        detail::validate_ptq1_shape(output_rows, input_rows, max_tokens, policy);
        return detail::t2_a8_workspace_bytes(output_rows, input_rows, policy, max_tokens, true);
    case QType::T2_G128_FP16:
        (void)detail::select_t2_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_t2_launch(output_rows, input_rows, max_tokens, policy);
        return detail::t2_a8_workspace_bytes(output_rows, input_rows, policy, max_tokens);
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32:
    case QType::INT32:
    default:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}


} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    if (is_gguf(qtype)) {
        const detail::GgufShape shape{qtype, output_rows, input_rows};
        return detail::gguf_project_workspace_bytes({&shape, 1}, min_tokens, max_tokens);
    }
    const auto routes = [&](std::int32_t lo, std::int32_t hi) {
        return linear_route_workspace_bytes(qtype, output_rows, input_rows, policy, lo, hi);
    };
    // Every width of a dense format takes an sm_70 route there, except the NVFP4 int8 family.
    if (policy == LinearPolicy::AllowNvfp4Int8) { return routes(min_tokens, max_tokens); }
    return detail::sm70::capacity(qtype, output_rows, input_rows, min_tokens, max_tokens, policy,
                                  routes);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

} // namespace ninfer::ops
