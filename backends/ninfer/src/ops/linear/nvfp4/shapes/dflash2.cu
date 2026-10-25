// DFlash2 drafter matrices for the NVFP4-encoded draft module: weight-only A16 routes
// (gemv for a single token, exact small-T families, 32-token chunks for wider extents).
// The draft never quantizes activations, so no A4 route is registered for these shapes.
//
// Geometries follow the DFlash2 drafter: feature expansion, fused QKV, attention output,
// convolution projection and the selector. They are registered through the same
// Nvfp4LinearShape table the text tower uses.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct DFlash2Routes {
    using Geometry = Nvfp4Geometry<OutputRows, InputRows>;
    using Gemv =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
    template <int Tokens>
    using Exact =
        Nvfp4SimtSchedule<(Tokens <= 12 ? 8 : (Tokens <= 16 ? 16 : 4)), 1, 2,
                          (Tokens >= 17 && Tokens <= 20) ? 8 : 16, Tokens, 1,
                          Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                          Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    using C2 = Nvfp4SimtSchedule<8, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    using C4 = Nvfp4SimtSchedule<8, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::SharedPhase,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
    using C32 = Nvfp4SimtSchedule<4, 1, 2, 16, 8, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                  Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                  Nvfp4SimtBlockOrder::TokenTilesContiguous, 4>;
    using FullChunk = Nvfp4SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                        Nvfp4SimtBlockOrder::RowsContiguous, 1>;

    static Nvfp4Launch select_a16(std::int32_t tokens) {
        if (tokens == 1) return launch_nvfp4_gemv<Geometry, Gemv>;
        if (tokens == 32) return launch_nvfp4_simt<Geometry, 32, FullChunk, true>;
        if (tokens >= 5 && tokens <= 28) return select_nvfp4_exact<Geometry, 5, 28, Exact>(tokens);
        if (tokens <= 2) return launch_nvfp4_simt<Geometry, 2, C2, true>;
        if (tokens <= 4) return launch_nvfp4_simt<Geometry, 4, C4, false>;
        if (tokens <= 32) return launch_nvfp4_simt<Geometry, 32, C32, false>;
        throw std::logic_error("nvfp4 DFlash2 A16 chunk exceeds shape capacity");
    }

    static void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
        launch_nvfp4_a16_chunks<32, select_a16>(x, weight, out, stream);
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

} // namespace ninfer::ops::detail
