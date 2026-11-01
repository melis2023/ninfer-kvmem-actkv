// The rotation past a model's native window, resolved once for every family from the Engine's
// rope options: YaRN at a fixed --rope-yarn-factor, or at --max-context / native once --rope-yarn's
// window passes the native positions, to four times the window (the extension Qwen documents,
// 1,048,576 tokens on a 262,144-token window); or, where the family's rotations take it, linear
// position interpolation by --rope-scaling-factor past --rope-scaling-original-context. Each
// family declares what its rotations can take and applies the plan to every rotation it makes;
// the Engine reports the summary (LoadSummary::rope, the server_start record's engine.rope).
#pragma once

#include "ninfer/ops/rope.h"
#include "ninfer/engine_options.h"
#include "ninfer/observability/snapshot.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::runtime {

// What a family's rotations take past the native window. YaRN is taken by every family whose
// full-attention blocks rotate (Flint's do not, and it takes no scaling).
struct RopeScalingSupport {
    std::string_view family; // named in a refusal
    bool interpolation = false; // linear position interpolation (--rope-scaling-factor)
};

inline constexpr RopeScalingSupport kQwen35RopeScaling{"Qwen3.5", true};
inline constexpr RopeScalingSupport kQwen4ExpRopeScaling{"Qwen3.8-Flash-Next", false};
inline constexpr RopeScalingSupport kGemma4RopeScaling{"Gemma 4", false};

struct RopeScalingPlan {
    ops::RopeYarn yarn;         // what the family's rotations apply
    RopeScalingSummary summary; // what the Engine reports
};

inline constexpr std::uint32_t kRopePositionExtension = 4;

inline RopeScalingPlan resolve_rope_scaling(const EngineOptions& options,
                                            std::uint32_t native_context,
                                            const RopeScalingSupport& support) {
    if (native_context == 0) {
        throw std::invalid_argument("the model's native window is unknown");
    }
    RopeScalingPlan plan;
    plan.summary.native_context = native_context;
    if (options.rope_scaling_factor > 1.0F) {
        if (!support.interpolation) {
            throw std::invalid_argument(std::string(support.family) +
                                        " rotates past its native window with YaRN (--rope-yarn, "
                                        "--rope-yarn-factor), not --rope-scaling-factor");
        }
        const std::uint32_t threshold = options.rope_scaling_original_context != 0
                                            ? options.rope_scaling_original_context
                                            : native_context;
        plan.yarn.interpolation_factor    = options.rope_scaling_factor;
        plan.yarn.interpolation_threshold = threshold;
        plan.summary.type                 = RopeScalingType::Interpolation;
        plan.summary.factor               = options.rope_scaling_factor;
        plan.summary.native_context       = threshold;
        return plan;
    }
    float factor = 1.0F;
    if (options.rope_yarn_factor > 1.0F) {
        factor = options.rope_yarn_factor;
    } else if (options.rope_yarn && options.max_context > native_context) {
        factor = static_cast<float>(options.max_context) / static_cast<float>(native_context);
    }
    if (factor > static_cast<float>(kRopePositionExtension)) {
        throw std::invalid_argument(
            std::string(support.family) + "'s YaRN reaches four times its native window: "
                                          "--max-context " +
            std::to_string(std::uint64_t{kRopePositionExtension} * native_context) + " at most");
    }
    if (factor > 1.0F) {
        plan.yarn.factor         = factor;
        plan.yarn.native_context = native_context;
        plan.summary.type        = RopeScalingType::Yarn;
        plan.summary.factor      = factor;
    }
    return plan;
}

} // namespace ninfer::runtime
