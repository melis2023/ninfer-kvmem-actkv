// The rotation past the native window as every family resolves it from the Engine's rope options
// (runtime/contract/rope_scaling.h): the plan each family applies and the summary the Engine
// reports, the refusals, and the four-times-native cap.
#include "runtime/contract/rope_scaling.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

bool near(float a, double b) { return std::fabs(double(a) - b) <= 1e-6 * std::fabs(b) + 1e-9; }

ninfer::EngineOptions options(std::uint32_t max_context, bool yarn = false, float yarn_factor = 1.0F,
                              float scaling_factor = 1.0F, std::uint32_t original = 0) {
    ninfer::EngineOptions out;
    out.max_context                   = max_context;
    out.rope_yarn                     = yarn;
    out.rope_yarn_factor              = yarn_factor;
    out.rope_scaling_factor           = scaling_factor;
    out.rope_scaling_original_context = original;
    return out;
}

// Every family with the same options gives the same plan, apart from what it refuses.
const ninfer::runtime::RopeScalingSupport* const kFamilies[] = {
    &ninfer::runtime::kQwen35RopeScaling, &ninfer::runtime::kQwen4ExpRopeScaling,
    &ninfer::runtime::kGemma4RopeScaling};

void unscaled() {
    for (const auto* family : kFamilies) {
        const std::string name(family->family);
        // Within the window, with or without --rope-yarn.
        for (const bool yarn : {false, true}) {
            const auto plan =
                ninfer::runtime::resolve_rope_scaling(options(262144, yarn), 262144, *family);
            expect(!plan.yarn.active(), name + ": a window at native is unscaled");
            expect(plan.summary.type == ninfer::RopeScalingType::None &&
                       plan.summary.factor == 1.0F && plan.summary.native_context == 262144,
                   name + ": the unscaled summary names the native window");
        }
        // Past the window without --rope-yarn: plain RoPE, reported as none.
        const auto plain = ninfer::runtime::resolve_rope_scaling(options(524288), 262144, *family);
        expect(!plain.yarn.active() && plain.summary.type == ninfer::RopeScalingType::None,
               name + ": past the window without the flag stays plain RoPE");
    }
}

void yarn() {
    for (const auto* family : kFamilies) {
        const std::string name(family->family);
        // --rope-yarn: the factor is max-context / native.
        const auto derived =
            ninfer::runtime::resolve_rope_scaling(options(589824, true), 262144, *family);
        expect(near(derived.yarn.factor, 2.25) && derived.yarn.native_context == 262144 &&
                   derived.yarn.interpolation_factor == 1.0F,
               name + ": --rope-yarn takes max-context / native");
        expect(derived.summary.type == ninfer::RopeScalingType::Yarn &&
                   near(derived.summary.factor, 2.25) && derived.summary.native_context == 262144,
               name + ": the YaRN summary carries the factor and the window");
        // --rope-yarn-factor: fixed, whatever the window.
        const auto fixed = ninfer::runtime::resolve_rope_scaling(options(8192, false, 4.0F),
                                                                 262144, *family);
        expect(fixed.yarn.factor == 4.0F && fixed.yarn.native_context == 262144 &&
                   fixed.summary.type == ninfer::RopeScalingType::Yarn &&
                   fixed.summary.factor == 4.0F,
               name + ": --rope-yarn-factor applies below the window");
        // The fixed factor wins over the derived one.
        const auto both = ninfer::runtime::resolve_rope_scaling(options(589824, true, 3.0F),
                                                                262144, *family);
        expect(both.yarn.factor == 3.0F, name + ": a fixed factor wins over the derived one");
        // Four times the window at most.
        const auto cap = ninfer::runtime::resolve_rope_scaling(options(1048576, true), 262144,
                                                               *family);
        expect(near(cap.yarn.factor, 4.0), name + ": four times the window is the cap");
        bool refused = false;
        try {
            (void)ninfer::runtime::resolve_rope_scaling(options(1048577, true), 262144, *family);
        } catch (const std::invalid_argument& error) {
            refused = std::string(error.what()).find("1048576") != std::string::npos &&
                      std::string(error.what()).find(name) != std::string::npos;
        }
        expect(refused, name + ": past four times the window is refused with the limit");
        // A smaller window, as a slice's.
        const auto slice =
            ninfer::runtime::resolve_rope_scaling(options(16384, true), 4096, *family);
        expect(slice.yarn.factor == 4.0F && slice.yarn.native_context == 4096,
               name + ": the native window comes from the model");
    }
}

void interpolation() {
    // Qwen3.5 takes it, at the native window by default or at the given threshold.
    const auto at_native = ninfer::runtime::resolve_rope_scaling(
        options(1048576, false, 1.0F, 8.0F), 262144, ninfer::runtime::kQwen35RopeScaling);
    expect(at_native.yarn.factor == 1.0F && at_native.yarn.interpolation_factor == 8.0F &&
               at_native.yarn.interpolation_threshold == 262144,
           "Qwen3.5: interpolation past the native window by default");
    expect(at_native.summary.type == ninfer::RopeScalingType::Interpolation &&
               at_native.summary.factor == 8.0F && at_native.summary.native_context == 262144,
           "Qwen3.5: the interpolation summary carries the factor and the threshold");
    const auto at_given = ninfer::runtime::resolve_rope_scaling(
        options(65536, false, 1.0F, 2.0F, 32768), 262144, ninfer::runtime::kQwen35RopeScaling);
    expect(at_given.yarn.interpolation_threshold == 32768 &&
               at_given.summary.native_context == 32768,
           "Qwen3.5: --rope-scaling-original-context sets the threshold");
    // The other families refuse it, naming the flags that work.
    for (const auto* family : {&ninfer::runtime::kQwen4ExpRopeScaling,
                               &ninfer::runtime::kGemma4RopeScaling}) {
        bool refused = false;
        try {
            (void)ninfer::runtime::resolve_rope_scaling(options(524288, false, 1.0F, 2.0F),
                                                        262144, *family);
        } catch (const std::invalid_argument& error) {
            const std::string what(error.what());
            refused = what.find(std::string(family->family)) != std::string::npos &&
                      what.find("--rope-yarn") != std::string::npos &&
                      what.find("--rope-scaling-factor") != std::string::npos;
        }
        expect(refused, std::string(family->family) + ": position interpolation is refused");
    }
}

void unknown_window() {
    bool refused = false;
    try {
        (void)ninfer::runtime::resolve_rope_scaling(options(4096), 0,
                                                    ninfer::runtime::kGemma4RopeScaling);
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "a model without a native window is refused");
}

} // namespace

int main() {
    unscaled();
    yarn();
    interpolation();
    unknown_window();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " rope scaling\n";
    return failures == 0 ? 0 : 1;
}
