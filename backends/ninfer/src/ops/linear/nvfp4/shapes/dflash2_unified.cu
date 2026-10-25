// DFlash2 drafter matrices for the NVFP4-encoded draft module, unified routes.
// Weight-only A16 (gemv for one token, SIMT for the draft widths, MMA for wider chunks).
// The draft never quantizes activations, so no A4 route is registered.
#include "ops/linear/nvfp4/nvfp4_shapes_unified.h"
#include "ops/linear/nvfp4/nvfp4_launch_unified.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::unified {
namespace {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct DFlash2Routes {
    using Geometry = Nvfp4Geometry<OutputRows, InputRows>;
    using Gemv =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    using C2 = Nvfp4A16SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    using C4 = Nvfp4A16SimtSchedule<8, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                    Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                    Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    using FullChunk = Nvfp4A16SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                           Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                           Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    using T32 = Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>;
    using T64 = Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>;

    static Nvfp4Launch select(std::int32_t tokens) {
        if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
        if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
        if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
        if (tokens <= 32) return nvfp4_linear_a16_simt<Geometry, 32, FullChunk, false>;
        if (tokens <= 64) return nvfp4_linear_a16_mma<Geometry, T32>;
        return nvfp4_linear_a16_mma<Geometry, T64>;
    }

    static void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        constexpr std::int32_t kChunk = 32;
        for (std::int32_t offset = 0; offset < x.ne[1]; offset += kChunk) {
            const std::int32_t count = std::min(kChunk, x.ne[1] - offset);
            const Tensor input       = x.slice(1, offset, count);
            Tensor output            = out.slice(1, offset, count);
            select(count)(input, weight, output, stream);
        }
    }
};

bool dflash2_uses_a4(std::int32_t, std::int32_t) { return false; }

} // namespace

const Nvfp4LinearShape kNvfp4DFlash2Feature{25600, 5120, DFlash2Routes<25600, 5120>::launch,
                                            nullptr, dflash2_uses_a4};
// The artifact stores the feature expansion as [hidden, 5*hidden]; this route covers
// that ordering (the one above covers the transposed [5*hidden, hidden] form).
const Nvfp4LinearShape kNvfp4DFlash2FeatureT{5120, 25600, DFlash2Routes<5120, 25600>::launch,
                                            nullptr, dflash2_uses_a4};
const Nvfp4LinearShape kNvfp4DFlash2Qkv{6144, 5120, DFlash2Routes<6144, 5120>::launch, nullptr,
                                        dflash2_uses_a4};
const Nvfp4LinearShape kNvfp4DFlash2AttnOut{5120, 4096, DFlash2Routes<5120, 4096>::launch, nullptr,
                                            dflash2_uses_a4};
const Nvfp4LinearShape kNvfp4DFlash2ConvProj{1280, 5120, DFlash2Routes<1280, 5120>::launch,
                                             nullptr, dflash2_uses_a4};
const Nvfp4LinearShape kNvfp4DFlash2Selector{256, 5120, DFlash2Routes<256, 5120>::launch, nullptr,
                                             dflash2_uses_a4};

} // namespace ninfer::ops::detail::unified
