#include "ops/linear/linear_test_common.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/nvfp4/nvfp4_e4m3_scale.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include "ops/linear/nvfp4/nvfp4_mmq_host.h"
#include "ops/op_tester.h" // test::ScopedNvfp4Int8Mainloop

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

// The unified chain's narrow band is what this file's last section puts next to the port, and it is
// spelled by its own namespace rather than by the port's.
namespace narrow = ninfer::ops::detail::narrow;

int run_generic_geometries();
int run_forced_padded_widths();
int run_narrow_band_width_ladder();
int run_narrow_band_bitexact_port();
int run_wide_band_bitexact_mmq();
int verify_token_tile_ladder();
int verify_geometry_diagnostics();

// Two things have to be said for an int8 measurement to be real, and they are the same shape of
// hazard. (1) Plain Linear's int8 route is registered by the unified shape table only: its shapes
// registry and its kernel instances are all `ops::linear::nvfp4::unified`, and an own-table call
// raises rather than quietly running A16. This binary installs no device profile, so without a table
// pin the schedule would be Legacy for most widths and every int8 invocation would fail the
// dispatcher's own check -- a configuration error masquerading as a test result. (2) The wide
// mainloop follows the table (own runs MMQ, unified runs grouped), so pin the table's answer
// explicitly rather than letting a future band change move the kernel under test. A case that wants
// the other mainloop says so locally and nests.
struct UnifiedInt8Route {
    UnifiedInt8Route()
        : previous(ops::detail::force_nvfp4_int8_mainloop(
              ops::detail::Nvfp4Int8Mainloop::Grouped)) {
        ops::detail::force_linear_route_table(ops::detail::LinearRouteTable::Unified);
    }
    ~UnifiedInt8Route() {
        ops::detail::force_linear_route_table(std::nullopt);
        ops::detail::force_nvfp4_int8_mainloop(previous);
    }
    UnifiedInt8Route(const UnifiedInt8Route&)            = delete;
    UnifiedInt8Route& operator=(const UnifiedInt8Route&) = delete;

    std::optional<ops::detail::Nvfp4Int8Mainloop> previous;
};

// Pin a mainloop for the duration of one block. Shared with the linear harnesses, which pin grouped
// for an Int8Grouped shape: save/restore rather than clear means the two nest instead of fighting.
using ForcedRoute = test::ScopedNvfp4Int8Mainloop;

// Intervals whose widest point has a whole 128-token tile inside them. A registered geometry also
// admits the empty intervals below (it has an A16 composition to fall back to); the alignment-only
// geometry has no fallback, so only the whole-tile points inside an interval are reachable --
// verify_aligned_only_envelopes states its envelope over exactly those.
constexpr std::pair<std::int32_t, std::int32_t> kRegisteredIntervals[] = {
    {1, 4}, {2, 4}, {1, 128}, {9, 25}, {24, 129}, {128, 512}};
constexpr std::pair<std::int32_t, std::int32_t> kAlignedOnlyIntervals[] = {
    {1, 128}, {1, 129}, {9, 129}, {129, 256}, {128, 512}};

// The shared verify_workspace_envelopes covers the A16/A8/A4 policies; the int8 route's
// admission gate makes its envelope worth checking under its own policy, where ragged points
// reserve nothing and tile-aligned points reserve their exact planes.
int verify_int8_workspace_envelopes(
    std::int32_t n, std::int32_t k,
    std::span<const std::pair<std::int32_t, std::int32_t>> intervals = kRegisteredIntervals) {
    int failures = 0;
    const auto policy = ops::LinearPolicy::AllowNvfp4Int8;
    for (auto [first, last] : intervals) {
        const auto capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, first, last);
        for (int t = first; t <= last; ++t) {
            const auto point =
                ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, t, t);
            if (point > capacity) {
                std::cerr << "Linear workspace interval [" << first << ',' << last
                          << "] cannot cover T=" << t << " for [" << n << ',' << k << "]\n";
                ++failures;
                break;
            }
        }
    }
    // The MMQ mainloop quantises the same activations into one interleaved plane instead of two.
    // It costs the same K + K/8 bytes per token, so every token count it can be called on has to fit
    // inside the envelope declared for it -- a silent overrun would only show up as a corrupted
    // activation plane on the shapes that actually reach the MMQ route. The envelope has to be asked
    // with THIS mainloop pinned, because the two mainloops now deliberately size their planes
    // differently below the crossover: the grouped one addresses the call's own columns, MMQ and the
    // port it shares the band with address the padded tile. The width filter is just "can a route get
    // here": MMQ pads like the grouped kernel, so the ragged widths are exactly the interesting ones,
    // and the only geometry condition left is the weight-side K % ITER_K == 0.
    const ForcedRoute mmq_mainloop(ops::detail::Nvfp4Int8Mainloop::Mmq);
    for (auto [first, last] : intervals) {
        const auto capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, first, last);
        for (int t = first; t <= last; ++t) {
            if (!(ops::detail::mmq::k_admitted(k) &&
                  ops::detail::unified::nvfp4_int8_route_admits(t))) {
                continue;
            }
            const auto tile = ops::detail::mmq::activation_bytes(t, k);
            if (tile > capacity) {
                std::cerr << "MMQ activation plane T=" << t << " needs " << tile
                          << " B, interval [" << first << ',' << last << "] declares " << capacity
                          << " for [" << n << ',' << k << "]\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// An unregistered alignment-only geometry has no A16 fallback, so a width its route does not admit
// has no route at all -- it throws, which is a different statement from "reserves nothing". Its
// envelope is therefore stated over the widths its route can reach: a declared interval still has to
// cover each of those, the property the A16 fallback would otherwise hide.
int verify_aligned_only_envelopes(
    std::int32_t n, std::int32_t k,
    std::span<const std::pair<std::int32_t, std::int32_t>> intervals) {
    int failures = 0;
    const auto policy = ops::LinearPolicy::AllowNvfp4Int8;
    for (auto [first, last] : intervals) {
        const auto capacity =
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, first, last);
        for (int t = first; t <= last; ++t) {
            if (!ops::detail::unified::nvfp4_int8_route_admits(t)) { continue; }
            const auto point =
                ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, policy, t, t);
            if (point > capacity) {
                std::cerr << "Linear workspace interval [" << first << ',' << last
                          << "] cannot cover T=" << t << " for [" << n << ',' << k << "]\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// The per-16 E4M3 scale table, enumerated against llama.cpp's ue4m3 decode.
//
// The reference below is an independent restatement of ggml_cuda_ue4m3_to_fp32
// (ggml/src/ggml-cuda/common.cuh:847, software fallback): ldexp-based, in double, bit-compared --
// not a copy of the production expression, which is written with power-of-two factors. Every one
// of the 256 bytes is checked, including the corners llama's two paths disagree on.
// ---------------------------------------------------------------------------
double llama_ue4m3_software(std::uint8_t byte) {
    if (byte == 0u || byte == 0x7Fu) { return 0.0; }
    const int exponent = (byte >> 3) & 0x0F;
    const int mantissa = byte & 0x07;
    const double raw   = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9)
                                       : std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0,
                                                    exponent - 7);
    return raw / 2.0;
}

// The signed E4M3FN reading this table used before it took llama's semantics -- the one whose
// values the old alpha = 0.5/divisor halved. Kept here as the other side of the half-factor check.
double signed_e4m3fn(std::uint8_t byte) {
    const bool negative = (byte & 0x80u) != 0;
    const int  exponent = (byte >> 3) & 0x0F;
    const int  mantissa = byte & 0x07;
    if (exponent == 0x0F && mantissa == 0x07) { return std::nan(""); }
    const double magnitude =
        exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9)
                      : std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
    return negative ? -magnitude : magnitude;
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int verify_ue4m3_scale_table() {
    int failures = 0;
    for (int byte = 0; byte < 256; ++byte) {
        const float produced = ops::detail::nvfp4::kNvfp4E4M3Scale[byte];
        const float expected =
            static_cast<float>(llama_ue4m3_software(static_cast<std::uint8_t>(byte)));
        if (float_bits(produced) != float_bits(expected)) {
            std::cerr << "NVFP4 E4M3 scale 0x" << std::hex << byte << std::dec << ": got "
                      << produced << ", llama.cpp ue4m3 gives " << expected << '\n';
            ++failures;
        }
    }
    // The half factor moved out of alpha and into the decode: over the reachable domain -- a block
    // scale is a magnitude, so bytes above 0x7F never occur -- the new entry is exactly half the
    // old one. 0x7F is the one reachable byte that changes for another reason: NaN -> 0.0f.
    int halved = 0;
    for (int byte = 0; byte <= 0x7E; ++byte) {
        const float produced = ops::detail::nvfp4::kNvfp4E4M3Scale[byte];
        const float old_half =
            static_cast<float>(signed_e4m3fn(static_cast<std::uint8_t>(byte)) / 2.0);
        if (float_bits(produced) != float_bits(old_half)) {
            std::cerr << "NVFP4 E4M3 scale 0x" << std::hex << byte << std::dec << ": got "
                      << produced << ", half the previous decode is " << old_half << '\n';
            ++failures;
        } else {
            ++halved;
        }
    }
    if (halved != 127) {
        std::cerr << "NVFP4 E4M3 scale: " << halved << "/127 reachable bytes are exactly halved\n";
        ++failures;
    }
    if (ops::detail::nvfp4::kNvfp4E4M3Scale[0x7F] != 0.0F) {
        std::cerr << "NVFP4 E4M3 scale 0x7F (NaN) must decode to 0.0f\n";
        ++failures;
    }

    // The A16 routes decode this table as `kNvfp4E4M3Scale[byte] * 2^15` and fold 2^14 into the
    // code side (see nvfp4_codec.cuh), so the guarantee those kernels rest on is "entry == the
    // previous decode / 2 over the reachable domain", asserted above for 0x00..0x7E. Two bytes are
    // outside the production domain (a block scale is a magnitude; the shipped weights top out at
    // 0x7E -- scan_scale_plane.py) and their behaviour was only written down in prose. Pinning it
    // here makes the prose checkable:
    //
    //   0x7F  the previous decode is NaN; this table is +0.0f (llama's byte-exact NaN guard).
    //   0x80+ bit 7 is ignored for the magnitude (exponent/mantissa extraction masks it), so the
    //         magnitude matches the low-7-bit byte -- EXCEPT 0xFF: the NaN guard tests the whole
    //         byte, so 0xFF slips past it and decodes to 240, while 0x7F decodes to 0. That
    //         asymmetry is a property of llama's expression, not of this table, and asserting it
    //         keeps a future "cleanup" from quietly changing what the kernels would read.
    {
        const auto old_at = [](int byte) {
            return static_cast<float>(signed_e4m3fn(static_cast<std::uint8_t>(byte)));
        };
        const float nan_probe = old_at(0x7F);
        const bool old_is_nan = !(nan_probe == nan_probe);
        if (!old_is_nan) {
            std::cerr << "NVFP4 E4M3 scale 0x7F: the previous decode was expected to be NaN, got "
                      << nan_probe << '\n';
            ++failures;
        }
        int magnitude_differences = 0;
        for (int byte = 0x80; byte <= 0xFF; ++byte) {
            const float produced = ops::detail::nvfp4::kNvfp4E4M3Scale[byte];
            const float low = ops::detail::nvfp4::kNvfp4E4M3Scale[byte & 0x7F];
            if (float_bits(produced) != float_bits(low)) { ++magnitude_differences; }
        }
        if (magnitude_differences != 1) {
            std::cerr << "NVFP4 E4M3 scale: 0x80..0xFF differs from its low-7-bit byte on "
                      << magnitude_differences << "/128 bytes; the documented model says exactly 1 "
                         "(0xFF, because the NaN guard is byte-exact)\n";
            ++failures;
        }
        if (float_bits(ops::detail::nvfp4::kNvfp4E4M3Scale[0xFF]) != float_bits(240.0F)) {
            std::cerr << "NVFP4 E4M3 scale 0xFF must decode to 240 (NaN guard is byte-exact), got "
                      << ops::detail::nvfp4::kNvfp4E4M3Scale[0xFF] << '\n';
            ++failures;
        }
        int sign_differences = 0;
        for (int byte = 0x80; byte <= 0xFF; ++byte) {
            if (float_bits(ops::detail::nvfp4::kNvfp4E4M3Scale[byte]) != float_bits(old_at(byte))) {
                ++sign_differences;
            }
        }
        std::cout << "NVFP4 E4M3 scale divergence pinned: 0x7F NaN -> 0.0f, 0xFF -> 240.0f "
                     "(byte-exact NaN guard), 0x80..0xFF differ from the previous signed decode on "
                  << sign_differences << "/128 bytes; reachable domain 0x00..0x7E is exact\n";
    }
    std::cout << "NVFP4 E4M3 scale table: 256/256 bytes match llama.cpp ue4m3, " << halved
              << "/127 reachable bytes are half the previous decode\n";
    return failures;
}

int run_nvfp4_int8() {
    const UnifiedInt8Route unified;
    std::vector<Invocation> invocations;
    // Ragged widths fall back to the A16 composition under the same policy: 1/4/16/64 sit under
    // the 128-token tile floor and 129/257 just past tile boundaries.
    for (int t : {1, 4, 16, 64, 129, 257})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    // Whole 128-token tiles take the grouped-integer route.
    for (int t : {128, 256, 384, 512})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    // Graph replay on both sides of the admission gate, plus one tile-aligned replay.
    for (int t : {128, 512})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8, true});
    invocations.push_back({129, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8, true});

    int failures = 0;
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {14336, 5120, 801U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {16384, 5120, 803U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {34816, 5120, 804U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {5120, 6144, 805U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {5120, 17408, 807U, Comparison::SampledRows, true, invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        failures += verify_int8_workspace_envelopes(n, k);
    }
    failures += run_generic_geometries();
    failures += run_forced_padded_widths();
    failures += run_narrow_band_width_ladder();
    failures += run_narrow_band_bitexact_port();
    failures += run_wide_band_bitexact_mmq();
    failures += verify_token_tile_ladder();
    failures += verify_ue4m3_scale_table();
    return failures;
}

// The registration contract: a geometry outside the 27B artifact's set reaches the same
// (kernels-free) int8 mainloop through one table row, and an aligned geometry with no row at all
// reaches it through the alignment admission. Neither has an A16 composition to hide behind, which
// is what keeps the oracle comparisons above from being vacuously green.
int run_generic_geometries() {
    int failures = 0;
    // [8192, 4096] has a row (shapes/n8192_k4096_unified.cu): ragged widths take its A16 rows, the
    // whole tiles take the shared int8 mainloop.
    std::vector<Invocation> registered;
    for (int t : {1, 64, 129}) {
        registered.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    for (int t : {128, 256, 512}) {
        registered.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {8192, 4096, 901U, Comparison::SampledRows, true, registered});
    failures += verify_int8_workspace_envelopes(8192, 4096);

    // [4096, 2048] has no row: the alignment admission is the only way in, and the int8 mainloop is
    // the only route behind it. An unregistered geometry therefore cannot pass by silently falling
    // back to A16 -- the policy overload refuses it.
    std::vector<Invocation> unregistered;
    for (int t : {128, 256, 384}) {
        unregistered.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped", ActivationCompute::Int8Grouped, make_nvfp4_weight,
                          {4096, 2048, 903U, Comparison::SampledRows, true, unregistered});
    failures += verify_aligned_only_envelopes(4096, 2048, kAlignedOnlyIntervals);

    failures += verify_geometry_diagnostics();
    return failures;
}

// The admission layer's diagnostics: a table miss under a policy that has no route for the
// geometry, a ragged width on the alignment-only geometry, and a geometry that misses either
// alignment axis, each named rather than reported as an unqualified "unsupported shape".
int verify_geometry_diagnostics() {
    // These are the diagnostics of a refusal: an alignment-only geometry has no table row, so the
    // only route that can serve it is the int8 family's alignment contract. Width is no longer part
    // of that contract -- the family answers every positive width by padding its last tile, which
    // run_forced_padded_widths is what exercises -- so what remains is the geometry: a shape that
    // misses either alignment axis.
    struct Case {
        std::int32_t n, k;
        ops::LinearPolicy policy;
        std::int32_t tokens;
        const char* expected;
    };
    const Case cases[] = {
        // No row and no A16 route under this policy.
        {4096, 2048, ops::LinearPolicy::A16Only, 128, "no registered shape"},
        // N is not a whole number of 128-row tiles.
        {4000, 2048, ops::LinearPolicy::AllowNvfp4Int8, 128, "N % 128 == 0"},
        // K is not a whole number of 64-column k-tiles.
        {4096, 2080, ops::LinearPolicy::AllowNvfp4Int8, 128, "K % 64 == 0"},
    };
    int failures = 0;
    for (const Case& probe : cases) {
        bool threw = false;
        std::string message;
        try {
            (void)ops::linear_workspace_capacity_bytes(QType::NVFP4, probe.n, probe.k, probe.policy,
                                                       probe.tokens, probe.tokens);
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        }
        if (!threw || message.find(probe.expected) == std::string::npos) {
            std::cerr << "NVFP4 int8 [" << probe.n << ',' << probe.k << "] T=" << probe.tokens
                      << " (" << static_cast<int>(probe.policy) << "): expected \"" << probe.expected
                      << "\", got " << (threw ? message : std::string("no exception")) << '\n';
            ++failures;
        }
    }
    return failures;
}

// The grouped route answers at every width, so what has to be right here is the padded tile: the
// widths the old cascade never sent to the int8 kernels -- one token, a handful, a width landing in
// the middle of a tile -- each run with the last tile full and its surplus columns discarded. Every
// column is compared against the FP64 oracle, and the input-preservation check catches the failure
// padding actually creates: a padding row reading past the caller's activation. The grouped cases run
// on the route run_nvfp4_int8 already pinned; only MMQ needs to say so itself.
int run_forced_padded_widths() {
    int failures = 0;

    // Every residue class under a tile, both sides of the 128 and 256 tile boundaries, and one
    // width past each -- a grid that floored instead of ceiling the token count would show up here
    // as missing columns, and one that ceiled the stores as corrupt ones.
    std::vector<Invocation> narrow_widths;
    for (int t : {1, 2, 3, 31, 32, 33, 64, 65, 100, 127}) {
        narrow_widths.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped_PaddedNarrow", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 911U, Comparison::SampledRows, true, narrow_widths});

    std::vector<Invocation> ragged;
    for (int t : {128, 129, 255, 256, 257, 300, 383}) {
        ragged.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped_PaddedRagged", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 913U, Comparison::SampledRows, true, ragged});

    // [4096, 2048] has no table row and no A16 composition, so a green run here can only mean the
    // padded int8 kernel answered these widths -- the positive-execution assertion the registered
    // geometry cannot offer on its own, where a refused width would quietly pass on A16 instead.
    std::vector<Invocation> alignment_only;
    for (int t : {1, 37, 129, 915}) {
        alignment_only.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Grouped_PaddedAlignedOnly", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {4096, 2048, 917U, Comparison::SampledRows, true, alignment_only});

    // The MMQ mainloop pads against a different shape of problem: its plane is window-major,
    // [window][token][row], so the STRIDE is a function of the token count and the writer and the
    // reader must both use the padded number. Same widths, same oracle, same guarded outputs.
    const ForcedRoute mmq_route(ops::detail::Nvfp4Int8Mainloop::Mmq);
    std::vector<Invocation> mmq_narrow;
    for (int t : {1, 2, 3, 31, 32, 33, 64, 65, 100, 127}) {
        mmq_narrow.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Mmq_PaddedNarrow", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 921U, Comparison::SampledRows, true, mmq_narrow});

    std::vector<Invocation> mmq_ragged;
    for (int t : {128, 129, 255, 256, 257, 300, 383}) {
        mmq_ragged.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Mmq_PaddedRagged", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 923U, Comparison::SampledRows, true, mmq_ragged});

    std::vector<Invocation> mmq_alignment_only;
    for (int t : {1, 37, 129, 915}) {
        mmq_alignment_only.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8Mmq_PaddedAlignedOnly", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {4096, 2048, 927U, Comparison::SampledRows, true, mmq_alignment_only});
    return failures;
}

// The int8 family's width ladder: a row-form kernel answers up to eight columns -- llama.cpp's own
// crossover -- and the MMA mainloops take everything above it, whichever int8 route is named. Which
// row-form kernel depends on the table this binary pins: the unified one runs its own narrow kernel
// at exactly the call's width, the legacy one runs the port, whose column stage is the width rounded
// up to a power of two plus six. Both halves of that are what run_narrow_band_bitexact_port below
// holds against each other; here the ladder only has to land on the FP64 oracle at the widths it
// serves, including both sides of each power-of-two column stage, so that whatever number of columns
// a stage retires, the call stores exactly the columns its caller asked for. A single column is where
// a token tile wastes the most, and the row-form kernel is the only thing that covers it. The wide
// half is here for the crossover itself: a width above kMaxTokens has to leave the row-form kernel,
// whose column stages stop at eight, rather than quietly keeping it.
int run_narrow_band_width_ladder() {
    int failures = 0;

    // Every width the row-form band serves, including both sides of each power-of-two column stage.
    // The route this binary pins is Grouped, which is the point: the narrow band answers under any
    // int8 route, not only a name of its own, and under this one it answers with the narrow kernel.
    std::vector<Invocation> narrow_widths;
    for (int t : {1, 2, 3, 4, 5, 6, 7, 8}) {
        narrow_widths.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8NarrowBand", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 931U, Comparison::SampledRows, true, narrow_widths});

    // Above the crossover the pinned route is the grouped MMA loop, at the J stage the width
    // selects. The geometry pair keeps K a whole ITER_K partition, and the case is the crossover
    // guard: these widths must reach the MMA loop instead of the row-form kernel.
    std::vector<Invocation> wide;
    for (int t : {9, 16, 256}) {
        wide.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8NarrowBand_Crossover", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {8192, 4096, 933U, Comparison::SampledRows, true, wide});

    // [4096, 2048] has no table row and no A16 composition, so a green run here can only mean the
    // ladder answered these widths itself.
    std::vector<Invocation> alignment_only;
    for (int t : {1, 3, 8, 37, 129}) {
        alignment_only.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowNvfp4Int8});
    }
    failures += run_shape("NVFP4_Int8NarrowBand_AlignedOnly", ActivationCompute::Int8Grouped,
                          make_nvfp4_weight,
                          {4096, 2048, 935U, Comparison::SampledRows, true, alignment_only});
    return failures;
}

// ---------------------------------------------------------------------------
// Layer 1 of the bit-exactness argument: the unified chain's narrow-band kernel and the port the
// legacy chain still runs below the crossover are not merely required to agree within the int8
// criterion -- they are required to leave the same bytes. They are the same contraction over the
// same quantisation (the port pads its column stage to a power of two, the narrow kernel takes the
// call's width, and the row factor only decides which block owns a row), so any difference is either
// an arithmetic drift in one of them or a host-side disagreement about the planes -- and the planes
// are the interesting half, because an uninitialised activation column routinely reads as zero and
// walks straight through an arithmetic oracle.
//
// The comparison is at ROUTE level rather than a direct call into the two kernels because the bug it
// exists to catch lives on the host: the plane allocation, the quantiser's grid and the swiglu gate
// plane all move with the column count, and only a call through the op touches all three at once. A
// direct kernel call could not be spelled here anyway -- `<<<>>>` needs a CUDA translation unit.
//
// Both pins resolve through the same dispatcher, so the case also has to prove the integer route
// actually ran instead of watching an A16 composition answer twice and report a vacuous equality. It
// reads that off the arena: the int8 routes are the only ones that take activation planes out of the
// caller's workspace, and the narrow one asks for strictly fewer bytes than the port whenever the
// width is under a whole padded tile.
// ---------------------------------------------------------------------------

// One run of one op: the bytes every output buffer held when it finished, the workspace it consumed,
// and whether the canaries around those buffers survived.
struct RouteSnapshot {
    std::vector<std::uint8_t> bytes;
    std::size_t peak = 0;
    int guard_failures = 0;
};

void append_output(RouteSnapshot& snapshot, const test::GuardedDeviceBuffer& out,
                   std::string_view label) {
    snapshot.guard_failures += out.verify_guards(label);
    const auto payload = test::from_device<std::uint8_t>(out.data(), out.bytes());
    snapshot.bytes.insert(snapshot.bytes.end(), payload.begin(), payload.end());
}

// The arena is the witness that the two pins really ran the two different kernels: below the widest
// stage the narrow route sizes its planes to the call and must reserve strictly less than the port,
// while at the widest stage the two agree because the port's padded tile is eight and so is the call.
// Equality of the two peaks below that width would mean both pins resolved to the SAME route -- an
// A16 or A4 composition answering twice, which is exactly the hollow green this layer exists to avoid.
int compare_narrow_port(std::string_view label, std::int32_t tokens, const RouteSnapshot& narrow_run,
                        const RouteSnapshot& port_run) {
    int failures = narrow_run.guard_failures + port_run.guard_failures;
    if (narrow_run.peak == 0 || port_run.peak == 0) {
        std::cerr << label << ": the int8 route did not run (workspace peak " << narrow_run.peak
                  << " narrow / " << port_run.peak << " port)\n";
        return failures + 1;
    }
    if (narrow_run.peak > port_run.peak) {
        std::cerr << label << ": narrow route reserved " << narrow_run.peak
                  << " B, the port it replaces " << port_run.peak
                  << " B -- its planes are not sized to the call\n";
        ++failures;
    } else if (tokens < narrow::kMaxTokens && narrow_run.peak == port_run.peak) {
        std::cerr << label << ": both pins reserved " << narrow_run.peak
                  << " B, so T=" << tokens << " ran one route twice rather than the narrow kernel and "
                     "the port\n";
        ++failures;
    }
    if (narrow_run.bytes.size() != port_run.bytes.size()) {
        std::cerr << label << ": output size mismatch, narrow " << narrow_run.bytes.size()
                  << " B vs port " << port_run.bytes.size() << " B\n";
        return failures + 1;
    }
    for (std::size_t index = 0; index < narrow_run.bytes.size(); ++index) {
        if (narrow_run.bytes[index] == port_run.bytes[index]) { continue; }
        std::cerr << label << ": narrow and port differ at byte " << index << ", "
                  << static_cast<int>(narrow_run.bytes[index]) << " vs "
                  << static_cast<int>(port_run.bytes[index]) << '\n';
        ++failures;
        break;
    }
    return failures;
}

// Runs `invoke` once on each mainloop and compares what the outputs ended up holding. `invoke` owns
// its buffers and appends them to the snapshot it is handed, so an op with four outputs compares all
// four and an op that reads its output before writing it (linear_add) can seed it identically both
// times. `capacity` is asked rather than passed in bytes because the two pins do not address the same
// width: the narrow kernel the Grouped pin runs sizes its planes to the call, the port the Mmq pin
// runs rounds them up to the tile the wide mainloops use. One arena sized for either alone would
// make the other pin fail to reserve, and asking inside the pin is also the plan layer's own
// contract stated at the widest point it has -- what the capacity query declares is what the launch
// takes out of it.
template <class Capacity, class Invoke>
int run_narrow_port_pair(std::string_view label, std::int32_t tokens, Capacity&& capacity,
                         Invoke&& invoke) {
    const ops::detail::Nvfp4Int8Mainloop pins[2] = {ops::detail::Nvfp4Int8Mainloop::Grouped,
                                                    ops::detail::Nvfp4Int8Mainloop::Mmq};
    RouteSnapshot snapshot[2];
    for (int index = 0; index < 2; ++index) {
        const ForcedRoute route(pins[index]);
        const std::size_t reserved = capacity();
        WorkspaceArena workspace(std::max<std::size_t>(reserved, 256));
        invoke(workspace, snapshot[index]);
        test::cuda_synchronize();
        snapshot[index].peak = workspace.peak_used();
        if (workspace.used() != 0) {
            std::cerr << label << ": route left " << workspace.used() << " B of workspace reserved\n";
            ++snapshot[index].guard_failures;
        }
    }
    return compare_narrow_port(label, tokens, snapshot[0], snapshot[1]);
}

// The activation both pins quantise: one deterministic, sign-mixed pattern per (token, column) at the
// widest stage, so a narrower call reads a prefix of the same buffer.
std::vector<std::uint16_t> make_band_activation(std::int32_t k, std::int32_t tokens,
                                                 std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    for (std::size_t index = 0; index < bits.size(); ++index) {
        const std::uint32_t hash = static_cast<std::uint32_t>(index) * 2654435761U + seed;
        const float represented =
            static_cast<float>(static_cast<int>(hash & 0x3ffU) - 512) * (1.0F / 512.0F);
        bits[index] = test::f32_to_bf16(represented);
    }
    return bits;
}

struct BandFixture {
    test::quantized_weight::PackedWeight host;
    DeviceBuffer weight;
    DeviceBuffer activation;
};

BandFixture make_band_fixture(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    BandFixture fixture{make_nvfp4_weight(n, k, seed), DeviceBuffer{}, DeviceBuffer{}};
    fixture.weight     = test::to_device(fixture.host.payload);
    fixture.activation = test::to_device(make_band_activation(k, narrow::kMaxTokens, seed + 1U));
    return fixture;
}

// ---------------------------------------------------------------------------
// The gate the wide band never had: byte equality between the unified chain's grouped mainloop and
// the own chain's MMQ, above the crossover.
//
// This is not a tolerance test and must not become one. The defect it exists for -- folding one
// activation scale per BK=64 tile instead of per 32-column group, through a per-tile FP32 temporary
// -- moved the FP64 oracle's worst case from rel_l2 0.004307 to 0.004364 against a limit of 0.04.
// An oracle with a 10x margin cannot see a 1.3% move, so every tolerance gate in this suite stayed
// green while the two chains disagreed on the bytes they wrote. Only raw comparison says.
//
// The same non-vacuity question as the narrow layer applies, but it cannot be answered off the arena
// here: above the crossover both routes reserve the same bytes for the same call, and the printed
// peaks below show exactly that. The control is the perturbation run in run_wide_mmq_pair.
// ---------------------------------------------------------------------------
int compare_wide_mmq(std::string_view label, const RouteSnapshot& grouped_run,
                     const RouteSnapshot& mmq_run) {
    int failures = grouped_run.guard_failures + mmq_run.guard_failures;
    if (grouped_run.peak == 0 || mmq_run.peak == 0) {
        std::cerr << label << ": the int8 route did not run (workspace peak " << grouped_run.peak
                  << " grouped / " << mmq_run.peak << " mmq)\n";
        return failures + 1;
    }
    std::cout << "  " << label << " peaks grouped=" << grouped_run.peak
              << " mmq=" << mmq_run.peak << '\n';
    if (grouped_run.bytes.size() != mmq_run.bytes.size()) {
        std::cerr << label << ": output size mismatch, grouped " << grouped_run.bytes.size()
                  << " B vs mmq " << mmq_run.bytes.size() << " B\n";
        return failures + 1;
    }
    std::size_t differing = 0;
    for (std::size_t index = 0; index < grouped_run.bytes.size(); ++index) {
        if (grouped_run.bytes[index] == mmq_run.bytes[index]) { continue; }
        if (differing < 3) {
            std::cerr << label << ": grouped and mmq differ at byte " << index << ", "
                      << static_cast<int>(grouped_run.bytes[index]) << " vs "
                      << static_cast<int>(mmq_run.bytes[index]) << '\n';
        }
        ++differing;
    }
    if (differing != 0) {
        std::cerr << label << ": " << differing << " of " << grouped_run.bytes.size()
                  << " output bytes differ\n";
        ++failures;
    }
    return failures;
}

// The witness that both mainloops really ran is NOT the arena: above the crossover the two chains
// reserve the same bytes for the same call (measured equal at all 25 cases), which is the capacity
// contract doing its job, not a route collapse. So this layer proves sensitivity instead: it flips
// the sign bit of one activation element, runs the grouped route again, and requires the output bytes
// to change. A comparison that reacts to one input element cannot be two pins answering from an
// uninitialised buffer or from the same kernel twice.
template <class Capacity, class Invoke>
int run_wide_mmq_pair(std::string_view label, std::int32_t tokens, void* activation,
                      Capacity&& capacity, Invoke&& invoke) {
    std::uint16_t base = 0;
    // This TU has no CUDA_CHECK macro; a failed copy here would mean the control cannot run at all,
    // so it is fatal rather than silently skipped.
    const auto copy = [&](void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
        if (cudaMemcpy(dst, src, bytes, kind) != cudaSuccess) {
            throw std::runtime_error(std::string(label) + ": control copy failed");
        }
    };
    copy(&base, activation, sizeof(base), cudaMemcpyDeviceToHost);
    const auto perturb = [&](bool on) {
        const std::uint16_t written = on ? static_cast<std::uint16_t>(base ^ 0x8000u) : base;
        copy(activation, &written, sizeof(written), cudaMemcpyHostToDevice);
    };

    const ops::detail::Nvfp4Int8Mainloop pins[3] = {ops::detail::Nvfp4Int8Mainloop::Grouped,
                                                    ops::detail::Nvfp4Int8Mainloop::Mmq,
                                                    ops::detail::Nvfp4Int8Mainloop::Grouped};
    RouteSnapshot snapshot[3];
    for (int index = 0; index < 3; ++index) {
        perturb(index == 2);
        const ForcedRoute route(pins[index]);
        const std::size_t reserved = capacity();
        WorkspaceArena workspace(std::max<std::size_t>(reserved, 256));
        invoke(workspace, snapshot[index]);
        test::cuda_synchronize();
        snapshot[index].peak = workspace.peak_used();
        if (workspace.used() != 0) {
            std::cerr << label << ": route left " << workspace.used()
                      << " B of workspace reserved\n";
            ++snapshot[index].guard_failures;
        }
    }
    perturb(false);
    int failures = compare_wide_mmq(label, snapshot[0], snapshot[1]);
    if (snapshot[0].bytes == snapshot[2].bytes) {
        std::cerr << label
                  << ": control failed -- flipping one activation element changed nothing, so the "
                     "byte comparison is not observing the kernel's output\n";
        ++failures;
    }
    return failures;
}

// The wide band's own whitelist of shapes: every one is a production projection of the 27B whose k is
// a multiple of 256, which is what the MMQ route admits (nvfp4_mmq_host.h:69) and the grouped route
// only asks for a multiple of 64. A shape the two chains disagree about admitting is not comparable
// here and is deliberately absent.
int run_wide_band_bitexact_mmq() {
    int failures = 0;
    constexpr std::int32_t kOut = 17408;
    const std::vector<std::int32_t> widths = {16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                                              129, 256, 1024};
    const std::int32_t widest = widths.back();

    // The fixtures the narrow layer builds stage only eight columns; widen the activation so a real
    // call reads rows the quantiser filled rather than the tail of a smaller buffer.
    const auto widen = [&](BandFixture& fixture, std::int32_t k, std::uint32_t seed) {
        fixture.activation = test::to_device(make_band_activation(k, widest, seed));
    };
    BandFixture linear_large = make_band_fixture(8192, 4096, 941U);
    BandFixture fused        = make_band_fixture(14336, 5120, 943U);
    BandFixture gate_up      = make_band_fixture(34816, 5120, 947U);
    BandFixture gdn          = make_band_fixture(16384, 5120, 967U);
    BandFixture down         = make_band_fixture(5120, 6144, 971U);
    widen(linear_large, 4096, 2026U);
    widen(fused, 5120, 2027U);
    widen(gate_up, 5120, 2028U);
    widen(gdn, 5120, 2030U);
    widen(down, 6144, 2029U);

    const Weight linear_large_weight = linear_large.host.device_weight(linear_large.weight.p);
    const Weight fused_weight        = fused.host.device_weight(fused.weight.p);
    const Weight gate_up_weight      = gate_up.host.device_weight(gate_up.weight.p);
    const Weight gdn_weight          = gdn.host.device_weight(gdn.weight.p);
    const Weight down_weight         = down.host.device_weight(down.weight.p);

    // linear_add's epilogue adds into the destination it also reads, so both pins must start from
    // the same residual bytes.
    const std::vector<std::uint16_t> residual = make_band_activation(5120, widest, 977U);

    for (const std::int32_t tokens : widths) {
        const std::string width = " T=" + std::to_string(tokens);

        const auto run_linear = [&](std::string_view name, BandFixture& fixture,
                                    const Weight& weight, std::int32_t n, std::int32_t k) {
            const auto capacity = [&] {
                return ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k,
                                                            ops::LinearPolicy::AllowNvfp4Int8,
                                                            tokens, tokens);
            };
            return run_wide_mmq_pair(
                std::string(name) + " [" + std::to_string(n) + "," + std::to_string(k) + "]" + width,
                tokens, fixture.activation.p, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer out(std::size_t(n) * tokens * sizeof(std::uint16_t));
                    out.fill(0);
                    const Tensor x(fixture.activation.p, DType::BF16, {k, tokens});
                    Tensor destination(out.data(), DType::BF16, {n, tokens});
                    ops::linear(x, weight, destination, ops::LinearPolicy::AllowNvfp4Int8,
                                workspace, nullptr);
                    append_output(snapshot, out, name);
                });
        };

        failures += run_linear("linear", linear_large, linear_large_weight, 8192, 4096);
        failures += run_linear("linear", fused, fused_weight, 14336, 5120);

        {
            const auto capacity = [&] {
                return ops::linear_swiglu_workspace_capacity_bytes(
                    QType::NVFP4, 34816, 5120, ops::LinearPolicy::AllowNvfp4Int8, tokens, tokens);
            };
            failures += run_wide_mmq_pair(
                "linear_swiglu [34816,5120]" + width, tokens, gate_up.activation.p, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer out(std::size_t(kOut) * tokens *
                                                  sizeof(std::uint16_t));
                    out.fill(0);
                    const Tensor x(gate_up.activation.p, DType::BF16, {5120, tokens});
                    Tensor destination(out.data(), DType::BF16, {kOut, tokens});
                    ops::linear_swiglu(x, gate_up_weight, destination,
                                       ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, out, "linear_swiglu");
                });
        }

        {
            constexpr std::int32_t q_rows = 6144, kv_rows = 1024;
            const auto capacity = [&] {
                return ops::attn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, 14336, 5120, ops::LinearPolicy::AllowNvfp4Int8, tokens, tokens);
            };
            failures += run_wide_mmq_pair(
                "attn_input_proj [14336,5120]" + width, tokens, fused.activation.p, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer query(q_rows * tokens * sizeof(std::uint16_t)),
                        gate(q_rows * tokens * sizeof(std::uint16_t)),
                        key(kv_rows * tokens * sizeof(std::uint16_t)),
                        value(kv_rows * tokens * sizeof(std::uint16_t));
                    query.fill(0);
                    gate.fill(0);
                    key.fill(0);
                    value.fill(0);
                    const Tensor x(fused.activation.p, DType::BF16, {5120, tokens});
                    Tensor q_tensor(query.data(), DType::BF16, {q_rows, tokens});
                    Tensor g_tensor(gate.data(), DType::BF16, {q_rows, tokens});
                    Tensor k_tensor(key.data(), DType::BF16, {kv_rows, tokens});
                    Tensor v_tensor(value.data(), DType::BF16, {kv_rows, tokens});
                    ops::attn_input_proj(x, fused_weight, q_tensor, g_tensor, k_tensor, v_tensor,
                                         ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, query, "attn q");
                    append_output(snapshot, gate, "attn gate");
                    append_output(snapshot, key, "attn k");
                    append_output(snapshot, value, "attn v");
                });
        }

        // The GDN parent's two-plane adapter ([qkv; z] over 16384 rows). This shape was NOT in the
        // first version of this gate, and the freshly calibrated device profile routes
        // gdn_input's whole 7..1024 band to the unified chain -- so it is the most likely place for
        // a wide-band disagreement that the 25 original cases could not see.
        {
            constexpr std::int32_t qkv_rows = 10240, z_rows = 6144;
            const auto capacity = [&] {
                return ops::gdn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowNvfp4Int8, tokens, tokens);
            };
            failures += run_wide_mmq_pair(
                "gdn_input_proj [16384,5120]" + width, tokens, gdn.activation.p, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer qkv(qkv_rows * tokens * sizeof(std::uint16_t)),
                        z(z_rows * tokens * sizeof(std::uint16_t));
                    qkv.fill(0);
                    z.fill(0);
                    const Tensor x(gdn.activation.p, DType::BF16, {5120, tokens});
                    Tensor qkv_out(qkv.data(), DType::BF16, {qkv_rows, tokens});
                    Tensor z_out(z.data(), DType::BF16, {z_rows, tokens});
                    ops::gdn_input_proj(x, gdn_weight, qkv_out, z_out,
                                        ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, qkv, "gdn qkv");
                    append_output(snapshot, z, "gdn z");
                });
        }

        {
            const std::size_t payload = std::size_t(5120) * tokens * sizeof(std::uint16_t);
            const auto capacity = [&] {
                return ops::linear_add_workspace_capacity_bytes(
                    QType::NVFP4, 5120, 6144, ops::LinearPolicy::AllowNvfp4Int8, tokens, tokens);
            };
            failures += run_wide_mmq_pair(
                "linear_add [5120,6144]" + width, tokens, down.activation.p, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer out(payload);
                    out.copy_from_host(residual.data(), payload);
                    const Tensor x(down.activation.p, DType::BF16, {6144, tokens});
                    Tensor destination(out.data(), DType::BF16, {5120, tokens});
                    ops::linear_add(x, down_weight, destination,
                                    ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, out, "linear_add");
                });
        }
    }
    return failures;
}

int run_narrow_band_bitexact_port() {
    int failures = 0;

    // The whitelist is a list, so what production launches is worth stating rather than assuming: one
    // column stage per width, a row factor that turns on at two columns, and the two widest exact
    // stages (seven and eight) run at four rows -- measured there and only there (EXECUTION-LOG R3-5).
    // `kMaxRowsPerBlock` is the ceiling the kernel is written for, not a stage either chain runs.
    for (int nc = 1; nc <= narrow::kMaxTokens; ++nc) {
        const int expected_rows = nc >= 7 ? 4 : (nc >= 2 ? 2 : 1);
        if (narrow::columns(nc) != nc || narrow::rows_per_block(nc) != expected_rows) {
            std::cerr << "NVFP4 int8 narrow stage NC=" << nc << ": columns=" << narrow::columns(nc)
                      << " rows=" << narrow::rows_per_block(nc) << ", expected rows="
                      << expected_rows << '\n';
            ++failures;
        }
    }

    constexpr std::int32_t kOut = 17408; // gate_up [34816,5120]: gate rows then up rows
    const BandFixture linear_large = make_band_fixture(8192, 4096, 941U);
    const BandFixture fused        = make_band_fixture(14336, 5120, 943U);
    const BandFixture gate_up      = make_band_fixture(34816, 5120, 947U);
    const BandFixture gdn          = make_band_fixture(16384, 5120, 967U);
    const BandFixture down         = make_band_fixture(5120, 6144, 971U);

    const Weight linear_large_weight = linear_large.host.device_weight(linear_large.weight.p);
    const Weight fused_weight        = fused.host.device_weight(fused.weight.p);
    const Weight gate_up_weight      = gate_up.host.device_weight(gate_up.weight.p);
    const Weight gdn_weight          = gdn.host.device_weight(gdn.weight.p);
    const Weight down_weight         = down.host.device_weight(down.weight.p);

    // The residual linear_add adds into, seeded identically before each pin runs because the op
    // reads it back out of the buffer it writes.
    const std::vector<std::uint16_t> residual =
        make_band_activation(5120, narrow::kMaxTokens, 977U);

    for (std::int32_t tokens = 1; tokens <= narrow::kMaxTokens; ++tokens) {
        const std::string width = " T=" + std::to_string(tokens);

        const auto run_plain_linear = [&](std::string_view name, const BandFixture& fixture,
                                         const Weight& weight, std::int32_t n, std::int32_t k) {
            const auto capacity = [&] {
                return ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k,
                                                            ops::LinearPolicy::AllowNvfp4Int8,
                                                            tokens, tokens);
            };
            return run_narrow_port_pair(
                std::string(name) + " [" + std::to_string(n) + "," + std::to_string(k) + "]" +
                    width,
                tokens, capacity, [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer out(std::size_t(n) * tokens * sizeof(std::uint16_t));
                    out.fill(0);
                    const Tensor x(fixture.activation.p, DType::BF16, {k, tokens});
                    Tensor destination(out.data(), DType::BF16, {n, tokens});
                    ops::linear(x, weight, destination, ops::LinearPolicy::AllowNvfp4Int8,
                                workspace, nullptr);
                    append_output(snapshot, out, name);
                });
        };

        failures += run_plain_linear("linear", linear_large, linear_large_weight, 8192, 4096);
        failures += run_plain_linear("linear", fused, fused_weight, 14336, 5120);

        // The swiglu pair: the case the gate plane is checked in, because the up half reads columns
        // the gate half wrote, so a plane sized for a width the quantiser never filled shows up here
        // before anywhere else.
        {
            const auto capacity = [&] {
                return ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, 34816, 5120,
                                                                   ops::LinearPolicy::AllowNvfp4Int8,
                                                                   tokens, tokens);
            };
            failures += run_narrow_port_pair(
                "linear_swiglu [34816,5120]" + width, tokens, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer out(std::size_t(kOut) * tokens *
                                                  sizeof(std::uint16_t));
                    out.fill(0);
                    const Tensor x(gate_up.activation.p, DType::BF16, {5120, tokens});
                    Tensor destination(out.data(), DType::BF16, {kOut, tokens});
                    ops::linear_swiglu(x, gate_up_weight, destination,
                                       ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, out, "linear_swiglu");
                });
        }

        // The fused attention parent's four-plane segmented adapter. Parent rows are
        // [query; key; gate; value] (see nvfp4_attn_input_int8.cu), which is why the four buffers
        // are two shapes rather than four.
        {
            constexpr std::int32_t q_rows = 6144, kv_rows = 1024;
            const auto capacity = [&] {
                return ops::attn_input_proj_workspace_capacity_bytes(QType::NVFP4, 14336, 5120,
                                                                     ops::LinearPolicy::AllowNvfp4Int8,
                                                                     tokens, tokens);
            };
            failures += run_narrow_port_pair(
                "attn_input_proj [14336,5120]" + width, tokens, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer query(q_rows * tokens * sizeof(std::uint16_t)),
                        gate(q_rows * tokens * sizeof(std::uint16_t)),
                        key(kv_rows * tokens * sizeof(std::uint16_t)),
                        value(kv_rows * tokens * sizeof(std::uint16_t));
                    query.fill(0);
                    gate.fill(0);
                    key.fill(0);
                    value.fill(0);
                    const Tensor x(fused.activation.p, DType::BF16, {5120, tokens});
                    Tensor q_tensor(query.data(), DType::BF16, {q_rows, tokens});
                    Tensor g_tensor(gate.data(), DType::BF16, {q_rows, tokens});
                    Tensor k_tensor(key.data(), DType::BF16, {kv_rows, tokens});
                    Tensor v_tensor(value.data(), DType::BF16, {kv_rows, tokens});
                    ops::attn_input_proj(x, fused_weight, q_tensor, g_tensor, k_tensor, v_tensor,
                                        ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, query, "attn q");
                    append_output(snapshot, gate, "attn gate");
                    append_output(snapshot, key, "attn k");
                    append_output(snapshot, value, "attn v");
                });
        }

        // The GDN parent's two-plane adapter ([qkv; z] over 16384 rows).
        {
            constexpr std::int32_t qkv_rows = 10240, z_rows = 6144;
            const auto capacity = [&] {
                return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, 5120,
                                                                    ops::LinearPolicy::AllowNvfp4Int8,
                                                                    tokens, tokens);
            };
            failures += run_narrow_port_pair(
                "gdn_input_proj [16384,5120]" + width, tokens, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer qkv(qkv_rows * tokens * sizeof(std::uint16_t)),
                        z(z_rows * tokens * sizeof(std::uint16_t));
                    qkv.fill(0);
                    z.fill(0);
                    const Tensor x(gdn.activation.p, DType::BF16, {5120, tokens});
                    Tensor qkv_out(qkv.data(), DType::BF16, {qkv_rows, tokens});
                    Tensor z_out(z.data(), DType::BF16, {z_rows, tokens});
                    ops::gdn_input_proj(x, gdn_weight, qkv_out, z_out,
                                        ops::LinearPolicy::AllowNvfp4Int8, workspace, nullptr);
                    append_output(snapshot, qkv, "gdn qkv");
                    append_output(snapshot, z, "gdn z");
                });
        }

        // The store that reads its own destination back: linear_add's epilogue adds the projection
        // into the residual, which is the one adapter whose guard against a stray padding column is
        // an out-of-bounds READ rather than only a write.
        {
            const std::size_t payload = std::size_t(5120) * tokens * sizeof(std::uint16_t);
            const auto capacity = [&] {
                return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, 5120, 6144,
                                                                ops::LinearPolicy::AllowNvfp4Int8,
                                                                tokens, tokens);
            };
            failures += run_narrow_port_pair(
                "linear_add [5120,6144]" + width, tokens, capacity,
                [&](WorkspaceArena& workspace, RouteSnapshot& snapshot) {
                    test::GuardedDeviceBuffer out(payload);
                    out.copy_from_host(residual.data(), payload);
                    const Tensor x(down.activation.p, DType::BF16, {6144, tokens});
                    Tensor destination(out.data(), DType::BF16, {5120, tokens});
                    ops::linear_add(x, down_weight, destination, ops::LinearPolicy::AllowNvfp4Int8,
                                    workspace, nullptr);
                    append_output(snapshot, out, "linear_add");
                });
        }
    }
    return failures;
}

// The ladder's stage selection is arithmetic, not a table lookup, so it is cheap to state outright:
// the tile is the smallest admitted stage covering the width, and the padded width is that stage
// rounded up -- which together are what make a narrow call read each weight element exactly once.
int verify_token_tile_ladder() {
    int failures = 0;
    struct Case {
        std::int32_t tokens;
        int j;
        std::int32_t padded;
    };
    const Case cases[] = {
        {1, 8, 8},     {7, 8, 8},      {8, 8, 8},      {9, 16, 16},    {16, 16, 16},
        {17, 24, 24},  {48, 48, 48},   {49, 64, 64},   {64, 64, 64},   {65, 80, 80},
        {100, 112, 112}, {127, 128, 128}, {128, 128, 128}, {129, 128, 256}, {256, 128, 256},
        {300, 128, 384}, {1024, 128, 1024},
    };
    for (const Case& probe : cases) {
        const int j = ops::detail::unified::nvfp4_int8_select_j(probe.tokens);
        const std::int32_t padded = ops::detail::unified::nvfp4_int8_padded_tokens(probe.tokens);
        if (j != probe.j || padded != probe.padded) {
            std::cerr << "NVFP4 token tile T=" << probe.tokens << ": got J=" << j << " padded="
                      << padded << ", expected J=" << probe.j << " padded=" << probe.padded << '\n';
            ++failures;
        }
        if (!ops::detail::unified::nvfp4_int8_j_admitted(j)) {
            std::cerr << "NVFP4 token tile T=" << probe.tokens << ": J=" << j
                      << " is not an admitted stage\n";
            ++failures;
        }
    }
    // Every admitted stage has to divide the padded width it produces -- the grid is (rows/128,
    // padded/J), so anything else is a partial tile column and a launch that drops columns.
    for (int t = 1; t <= 1024; ++t) {
        const int j = ops::detail::unified::nvfp4_int8_select_j(t);
        const std::int32_t padded = ops::detail::unified::nvfp4_int8_padded_tokens(t);
        if (padded % j != 0 || padded < t) {
            std::cerr << "NVFP4 token tile T=" << t << ": padded=" << padded << " is not whole "
                      << j << "-token tiles covering the width\n";
            ++failures;
            break;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_int8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_Int8Grouped Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_Int8Grouped Linear: " << error.what() << '\n';
        return 1;
    }
}
