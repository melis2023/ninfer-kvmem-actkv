// Round-6 fused-edge gate for [RMSNorm -> int8 quantise].
//
// Three layers, numerics before timing:
//   1. Plane level: the codes and scales the edge kernel writes must be the bytes the route writes
//      today through two launches (the model layer's standalone rmsnorm, then this route's own
//      per-32 quantiser). Compared raw, at every width the route addresses, padded tail included.
//   2. Route level: the attention outputs and the SwiGLU output, byte-compared between
//      [rmsnorm + the ordinary int8 launch] and the fused launch, at every width the route serves --
//      narrow band and above the crossover alike. The wide half was recorded as a dependency on
//      Round-6 retiring the 64-column activation group; that retirement is in this tree now, so the
//      widths above T=8 are compared rather than promised.
//   3. Negative controls, because a contract test that can only pass is worth nothing (Round-4's
//      lesson): (a) the plane bytes must differ from a reference whose first two groups were
//      exchanged, so the byte comparison demonstrably sees group order; (b) the fused outputs must
//      differ from a reference that never normalised, so "the norm ran inside the fused kernel" is
//      demonstrated rather than assumed.
//
// The route is pinned (unified table, grouped mainloop): a width that quietly resolved to A16 or to
// the own chain would answer twice with the same bytes and report a hollow green.

#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h"
#include "ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_route.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

namespace fused  = ops::detail::unified;
namespace narrow = ops::detail::narrow;

constexpr std::int32_t kHidden = 5120;
constexpr float        kEps    = 1e-6F;
constexpr std::int32_t kAttnRows = 14336;
constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kSwigluOut  = kGateUpRows / 2;

struct UnifiedInt8Pin {
    UnifiedInt8Pin()
        : previous(ops::detail::force_nvfp4_int8_mainloop(
              ops::detail::Nvfp4Int8Mainloop::Grouped)) {
        ops::detail::force_linear_route_table(ops::detail::LinearRouteTable::Unified);
    }
    ~UnifiedInt8Pin() {
        ops::detail::force_linear_route_table(std::nullopt);
        ops::detail::force_nvfp4_int8_mainloop(previous);
    }
    UnifiedInt8Pin(const UnifiedInt8Pin&)            = delete;
    UnifiedInt8Pin& operator=(const UnifiedInt8Pin&) = delete;
    std::optional<ops::detail::Nvfp4Int8Mainloop> previous;
};

std::uint16_t bf16_of(float value) {
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    std::uint16_t bits            = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

// A deterministic row with outliers: a flat fixture would let a dropped group pass on zeros.
std::vector<std::uint16_t> host_activation(std::int32_t tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> values(static_cast<std::size_t>(tokens) * kHidden);
    std::uint32_t x = seed;
    for (std::size_t i = 0; i < values.size(); ++i) {
        x = x * 1664525u + 1013904223u;
        float value = (static_cast<float>(x >> 8) / 8388608.0F) - 0.5F;
        if (i % 97u == 0u) { value *= 128.0F; }
        if (i % 89u == 0u) { value *= 0.001F; }
        values[i] = bf16_of(value);
    }
    return values;
}

std::vector<std::uint16_t> host_norm_weight(std::uint32_t seed) {
    std::vector<std::uint16_t> values(kHidden);
    std::uint32_t x = seed;
    for (std::int32_t i = 0; i < kHidden; ++i) {
        x         = x * 1664525u + 1013904223u;
        values[i] = bf16_of((static_cast<float>(x >> 8) / 8388608.0F) - 0.5F);
    }
    return values;
}

DeviceBuffer upload(const std::vector<std::uint16_t>& host) {
    DeviceBuffer buffer(host.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(buffer.p, host.data(), buffer.bytes, cudaMemcpyHostToDevice));
    return buffer;
}

std::vector<std::uint8_t> read_bytes(const void* device, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    return host;
}

int compare_bytes(const std::string& what, const void* lhs, const void* rhs, std::size_t bytes) {
    const std::vector<std::uint8_t> a = read_bytes(lhs, bytes);
    const std::vector<std::uint8_t> b = read_bytes(rhs, bytes);
    if (a == b) { return 0; }
    std::size_t first = 0;
    while (first < bytes && a[first] == b[first]) { ++first; }
    std::cerr << what << ": bytes differ at offset " << first << " of " << bytes << " ("
              << static_cast<int>(a[first]) << " vs " << static_cast<int>(b[first]) << ")\n";
    return 1;
}

struct Case {
    DeviceBuffer x;
    DeviceBuffer w;
    DeviceBuffer h;
    Tensor x_tensor;
    Tensor w_tensor;
    Tensor h_tensor;
    std::int32_t tokens = 0;
    std::int32_t padded = 0;
};

Case make_case(std::int32_t tokens, std::uint32_t seed) {
    Case c;
    c.tokens = tokens;
    c.padded = narrow::addressed_tokens(tokens, ops::detail::Nvfp4Int8Mainloop::Grouped);
    c.x      = upload(host_activation(tokens, seed));
    c.w      = upload(host_norm_weight(seed ^ 0x777u));
    c.h      = DeviceBuffer(static_cast<std::size_t>(tokens) * kHidden * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(c.h.p, 0x33, c.h.bytes));
    c.x_tensor = Tensor(c.x.p, DType::BF16, {kHidden, tokens});
    c.w_tensor = Tensor(c.w.p, DType::BF16, {kHidden, 1});
    c.h_tensor = Tensor(c.h.p, DType::BF16, {kHidden, tokens});
    ops::rmsnorm(c.x_tensor, c.w_tensor, kEps, true, c.h_tensor, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return c;
}

struct Planes {
    DeviceBuffer codes;
    DeviceBuffer scales;
};

Planes allocate_planes(std::int32_t padded, std::uint8_t poison) {
    Planes p;
    p.codes  = DeviceBuffer(static_cast<std::size_t>(padded) * kHidden);
    p.scales = DeviceBuffer(static_cast<std::size_t>(padded) * (kHidden / 32) * sizeof(float));
    CUDA_CHECK(cudaMemset(p.codes.p, poison, p.codes.bytes));
    CUDA_CHECK(cudaMemset(p.scales.p, poison, p.scales.bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    return p;
}

int run_plane_case(std::int32_t tokens) {
    Case c = make_case(tokens, 0x9e3779u + tokens);

    // Today's two launches: the norm above, then this route's own per-32 quantiser.
    Planes reference = allocate_planes(c.padded, 0x5a);
    fused::launch_nvfp4_int8_quantize32(c.h_tensor.data, c.tokens, kHidden,
                                        static_cast<std::int8_t*>(reference.codes.p),
                                        static_cast<float*>(reference.scales.p), c.padded, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // The fused one: no global BF16 row in between.
    Planes fused_planes = allocate_planes(c.padded, 0x5a);
    fused::launch_rmsnorm_int8_quantize(c.x_tensor, c.w_tensor, kEps,
                                        static_cast<std::int8_t*>(fused_planes.codes.p),
                                        static_cast<float*>(fused_planes.scales.p), c.padded,
                                        nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::string label = "T=" + std::to_string(tokens);
    int failures            = 0;
    failures += compare_bytes(label + " codes", reference.codes.p, fused_planes.codes.p,
                              reference.codes.bytes);
    failures += compare_bytes(label + " scales", reference.scales.p, fused_planes.scales.p,
                              reference.scales.bytes);

    // Negative control (a): exchange the first two groups of the reference codes. If the fused
    // bytes still matched that, the comparison would be blind to group order.
    std::vector<std::uint8_t> swapped = read_bytes(reference.codes.p, reference.codes.bytes);
    constexpr std::size_t kGroupBytes = 32;
    std::swap_ranges(swapped.begin(), swapped.begin() + kGroupBytes,
                     swapped.begin() + kGroupBytes * 2);
    if (read_bytes(fused_planes.codes.p, fused_planes.codes.bytes) == swapped) {
        std::cerr << label << ": negative control is vacuous -- group order is not observed\n";
        ++failures;
    }
    return failures;
}

// ---------------------------------------------------------------------------
// The GDN output projection's edge: a per-head SiLU-gated norm (48 rows of 128) plus the same
// per-32 quantise. Plane level, like the section above, because the claim is about bytes: the fused
// pass must write exactly the codes and scales that [ops::gated_rmsnorm -> this route's quantiser]
// writes. The norm half is not a free re-implementation -- one warp owns one head row, the four-trip
// window and the shuffle reduction fix the summation order, and the gate multiplies after the
// rounding of the product, so the reproduction here is against the launcher the model layer calls
// today, at the head geometry of the model this route serves.
// ---------------------------------------------------------------------------
constexpr std::int32_t kGdnK    = 6144;  // 48 value heads x 128
constexpr std::int32_t kGdnHead = 128;
constexpr std::int32_t kGdnHeads = kGdnK / kGdnHead;

int run_gated_plane_case(std::int32_t tokens) {
    const std::string label = "gated T=" + std::to_string(tokens);
    const std::int32_t padded = narrow::addressed_tokens(tokens, ops::detail::Nvfp4Int8Mainloop::Grouped);

    auto upload_row = [](std::int32_t elements, std::uint32_t seed, float scale) {
        std::vector<std::uint16_t> values(static_cast<std::size_t>(elements));
        std::uint32_t x = seed;
        for (std::size_t i = 0; i < values.size(); ++i) {
            x = x * 1664525u + 1013904223u;
            float value = (static_cast<float>(x >> 8) / 8388608.0F) - 0.5F;
            if (i % 61u == 0u) { value *= 64.0F; }    // outliers: zeros must not carry the case
            if (i % 53u == 0u) { value *= 0.002F; }
            values[i] = bf16_of(value * scale);
        }
        return upload(values);
    };

    // The norm weight is one head wide and shared by all 48 heads, exactly as the launcher indexes
    // it (`weight[pair]`, pair < d/2).
    auto upload_head_weight = []() {
        std::vector<std::uint16_t> values(kGdnHead);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = bf16_of(0.5F + static_cast<float>(i % 13u) * 0.25F);
        }
        return upload(values);
    };

    const auto source = upload_row(tokens * kGdnK, 0x5eed0001u + tokens, 1.0F);
    const auto gate   = upload_row(tokens * kGdnK, 0x0b17e7u + tokens, 2.0F);
    const auto weight = upload_head_weight();
    DeviceBuffer on(static_cast<std::size_t>(tokens) * kGdnK * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(on.p, 0x33, on.bytes));

    const Tensor x_t(source.p, DType::BF16, {kGdnHead, kGdnHeads, tokens});
    const Tensor z_t(gate.p, DType::BF16, {kGdnHead, kGdnHeads, tokens});
    const Tensor w_t(weight.p, DType::BF16, {kGdnHead, 1});
    Tensor on_t(on.p, DType::BF16, {kGdnHead, kGdnHeads, tokens});

    Planes reference;
    reference.codes  = DeviceBuffer(static_cast<std::size_t>(padded) * kGdnK);
    reference.scales = DeviceBuffer(static_cast<std::size_t>(padded) * (kGdnK / 32) * sizeof(float));
    CUDA_CHECK(cudaMemset(reference.codes.p, 0x5a, reference.codes.bytes));
    CUDA_CHECK(cudaMemset(reference.scales.p, 0x5a, reference.scales.bytes));
    ops::gated_rmsnorm(x_t, w_t, z_t, ops::GateActivation::Silu, kEps, on_t, nullptr);
    fused::launch_nvfp4_int8_quantize32(on.p, tokens, kGdnK,
                                        static_cast<std::int8_t*>(reference.codes.p),
                                        static_cast<float*>(reference.scales.p), padded, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    Planes fused_planes;
    fused_planes.codes  = DeviceBuffer(static_cast<std::size_t>(padded) * kGdnK);
    fused_planes.scales = DeviceBuffer(static_cast<std::size_t>(padded) * (kGdnK / 32) * sizeof(float));
    CUDA_CHECK(cudaMemset(fused_planes.codes.p, 0x5a, fused_planes.codes.bytes));
    CUDA_CHECK(cudaMemset(fused_planes.scales.p, 0x5a, fused_planes.scales.bytes));
    fused::launch_gated_rmsnorm_int8_quantize(x_t, w_t, z_t, kEps,
                                              static_cast<std::int8_t*>(fused_planes.codes.p),
                                              static_cast<float*>(fused_planes.scales.p), tokens,
                                              padded, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int failures = 0;
    failures += compare_bytes(label + " codes", reference.codes.p, fused_planes.codes.p,
                              reference.codes.bytes);
    failures += compare_bytes(label + " scales", reference.scales.p, fused_planes.scales.p,
                              reference.scales.bytes);

    // Control (a): group order must be observed -- same trick as the plain edge's case.
    std::vector<std::uint8_t> swapped = read_bytes(reference.codes.p, reference.codes.bytes);
    std::swap_ranges(swapped.begin(), swapped.begin() + 32, swapped.begin() + 64);
    const bool group_order_observed =
        read_bytes(fused_planes.codes.p, fused_planes.codes.bytes) != swapped;
    if (!group_order_observed) {
        std::cerr << label << ": control is vacuous -- group order is not observed\n";
        ++failures;
    }
    // Control (b): the gate must be observed. Build the reference the *ungated* way (Plain epilogue,
    // no z) and require that the fused planes do NOT match it. A fused pass that read no gate would
    // answer with exactly those bytes, and the equality test above could never see the difference.
    DeviceBuffer plain(static_cast<std::size_t>(tokens) * kGdnK * sizeof(std::uint16_t));
    Tensor plain_t(plain.p, DType::BF16, {kGdnHead, kGdnHeads, tokens});
    Planes ungated;
    ungated.codes  = DeviceBuffer(static_cast<std::size_t>(padded) * kGdnK);
    ungated.scales = DeviceBuffer(static_cast<std::size_t>(padded) * (kGdnK / 32) * sizeof(float));
    CUDA_CHECK(cudaMemset(ungated.codes.p, 0x5a, ungated.codes.bytes));
    CUDA_CHECK(cudaMemset(ungated.scales.p, 0x5a, ungated.scales.bytes));
    ops::rmsnorm(x_t, w_t, kEps, false, plain_t, nullptr);
    fused::launch_nvfp4_int8_quantize32(plain.p, tokens, kGdnK,
                                        static_cast<std::int8_t*>(ungated.codes.p),
                                        static_cast<float*>(ungated.scales.p), padded, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const bool gate_observed =
        read_bytes(fused_planes.codes.p, fused_planes.codes.bytes) !=
        read_bytes(ungated.codes.p, ungated.codes.bytes);
    if (!gate_observed) {
        std::cerr << label << ": control is vacuous -- the gate is not observed\n";
        ++failures;
    }
    const bool same_planes =
        read_bytes(reference.codes.p, reference.codes.bytes) ==
            read_bytes(fused_planes.codes.p, fused_planes.codes.bytes) &&
        read_bytes(reference.scales.p, reference.scales.bytes) ==
            read_bytes(fused_planes.scales.p, fused_planes.scales.bytes);
    // A silent pass would not say which widths were actually put through the kernel, and this
    // section is the only evidence the gated edge has until it is wired into a route.
    std::cout << label << ": planes " << (same_planes ? "identical" : "DIFFER") << " of "
              << reference.codes.bytes << " + " << reference.scales.bytes << " B, group-order control "
              << (group_order_observed ? "live" : "DEAD") << ", gate control "
              << (gate_observed ? "live" : "DEAD") << "\n";
    return failures;
}

// The narrow band only: above the crossover the grouped mainloop still reads per-64 groups.
int run_attn_route_case(std::int32_t tokens) {
    Case c = make_case(tokens, 0x1234u + tokens);
    const auto packed = linear::make_nvfp4_weight(kAttnRows, kHidden, 4242u);
    DeviceBuffer payload(packed.payload.size());
    CUDA_CHECK(cudaMemcpy(payload.p, packed.payload.data(), packed.payload.size(),
                          cudaMemcpyHostToDevice));
    const Weight weight = packed.device_weight(payload.p);

    const std::size_t qbytes = static_cast<std::size_t>(6144) * tokens * sizeof(std::uint16_t);
    const std::size_t kbytes = static_cast<std::size_t>(1024) * tokens * sizeof(std::uint16_t);
    DeviceBuffer q_ref(qbytes), g_ref(qbytes), k_ref(kbytes), v_ref(kbytes);
    DeviceBuffer q_fus(qbytes), g_fus(qbytes), k_fus(kbytes), v_fus(kbytes);
    for (DeviceBuffer* b : {&q_ref, &g_ref, &k_ref, &v_ref, &q_fus, &g_fus, &k_fus, &v_fus}) {
        CUDA_CHECK(cudaMemset(b->p, 0x11, b->bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor q_r(q_ref.p, DType::BF16, {6144, tokens});
    Tensor g_r(g_ref.p, DType::BF16, {6144, tokens});
    Tensor k_r(k_ref.p, DType::BF16, {1024, tokens});
    Tensor v_r(v_ref.p, DType::BF16, {1024, tokens});
    Tensor q_f(q_fus.p, DType::BF16, {6144, tokens});
    Tensor g_f(g_fus.p, DType::BF16, {6144, tokens});
    Tensor k_f(k_fus.p, DType::BF16, {1024, tokens});
    Tensor v_f(v_fus.p, DType::BF16, {1024, tokens});

    WorkspaceArena reference_workspace(512ULL << 20);
    WorkspaceArena fused_workspace(512ULL << 20);
    fused::nvfp4_attn_input_int8_launch(c.h_tensor, weight, q_r, g_r, k_r, v_r, reference_workspace,
                                        ops::detail::Nvfp4Int8Mainloop::Grouped, nullptr);
    fused::nvfp4_attn_input_int8_fused_norm_launch(c.x_tensor, c.w_tensor, kEps, weight, q_f, g_f,
                                                   k_f, v_f, fused_workspace,
                                                   ops::detail::Nvfp4Int8Mainloop::Grouped,
                                                   nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::string label = "attn T=" + std::to_string(tokens);
    int failures            = 0;
    failures += compare_bytes(label + " q", q_r.data, q_f.data, q_r.bytes());
    failures += compare_bytes(label + " gate", g_r.data, g_f.data, g_r.bytes());
    failures += compare_bytes(label + " k", k_r.data, k_f.data, k_r.bytes());
    failures += compare_bytes(label + " v", v_r.data, v_f.data, v_r.bytes());

    // Negative control (b): the same reference launcher fed the UN-normed row must not match the
    // fused result, otherwise the norm inside the fused kernel is unproven.
    Tensor x_as_h = c.x_tensor;
    fused::nvfp4_attn_input_int8_launch(x_as_h, weight, q_r, g_r, k_r, v_r, reference_workspace,
                                        ops::detail::Nvfp4Int8Mainloop::Grouped, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (read_bytes(q_r.data, q_r.bytes()) == read_bytes(q_f.data, q_f.bytes())) {
        std::cerr << label << ": negative control is vacuous -- the norm is not observed\n";
        ++failures;
    }
    return failures;
}

int run_swiglu_route_case(std::int32_t tokens) {
    Case c = make_case(tokens, 0x5E43u + tokens);
    const auto packed = linear::make_nvfp4_weight(kGateUpRows, kHidden, 9091u);
    DeviceBuffer payload(packed.payload.size());
    CUDA_CHECK(cudaMemcpy(payload.p, packed.payload.data(), packed.payload.size(),
                          cudaMemcpyHostToDevice));
    const Weight weight = packed.device_weight(payload.p);

    const std::size_t bytes = static_cast<std::size_t>(kSwigluOut) * tokens * sizeof(std::uint16_t);
    DeviceBuffer out_ref(bytes), out_fused(bytes);
    CUDA_CHECK(cudaMemset(out_ref.p, 0x11, out_ref.bytes));
    CUDA_CHECK(cudaMemset(out_fused.p, 0x11, out_fused.bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor r(out_ref.p, DType::BF16, {kSwigluOut, tokens});
    Tensor f(out_fused.p, DType::BF16, {kSwigluOut, tokens});
    WorkspaceArena reference_workspace(512ULL << 20);
    WorkspaceArena fused_workspace(512ULL << 20);
    fused::nvfp4_linear_swiglu_int8_grouped_launch(c.h_tensor, weight, r, reference_workspace,
                                                   ops::detail::Nvfp4Int8Mainloop::Grouped,
                                                   nullptr);
    fused::nvfp4_linear_swiglu_int8_fused_norm_launch(c.x_tensor, c.w_tensor, kEps, weight, f,
                                                      fused_workspace,
                                                      ops::detail::Nvfp4Int8Mainloop::Grouped,
                                                      nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return compare_bytes("swiglu T=" + std::to_string(tokens), r.data, f.data, r.bytes());
}

// The band the wiring may flip. This locks the *predicate* the model layer actually asks, so the
// predicate and the kernel's numeric cannot drift apart. Since Round-6 retired the 64-column
// activation group the edge serves every width the unified route answers -- and, symmetric to the
// old rule, it must now ACCEPT a wide width: a predicate that went stale in the other direction
// would silently leave prefill paying two launches for nothing.
int run_band_lock_case() {
    int failures = 0;
    // Printed so the log carries a positive witness that this block ran and with which answers,
    // rather than only an aggregate "OK" that would also be printed by a block that never executed.
    std::string attn_bits, swiglu_bits;
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 128, 1024}) {
        attn_bits += ops::detail::nvfp4_attn_input_int8_fused_norm_eligible(
                         ops::LinearPolicy::AllowNvfp4Int8, tokens)
                     ? '1'
                     : '0';
        swiglu_bits += ops::detail::nvfp4_linear_swiglu_int8_fused_norm_eligible(
                           ops::LinearPolicy::AllowNvfp4Int8, tokens)
                       ? '1'
                       : '0';
    }
    std::cout << "band lock T=1,2,4,8,16,128,1024  attn=" << attn_bits
              << " swiglu=" << swiglu_bits << '\n';
    // Every width is on the unified route under this pin, so all seven must be eligible.
    if (attn_bits != "1111111") {
        std::cerr << "band lock: attn predicate answered " << attn_bits
                  << ", expected 1111111 over T=1,2,4,8,16,128,1024\n";
        ++failures;
    }
    if (swiglu_bits != "1111111") {
        std::cerr << "band lock: swiglu predicate answered " << swiglu_bits
                  << ", expected 1111111 over T=1,2,4,8,16,128,1024\n";
        ++failures;
    }
    if (ops::detail::nvfp4_attn_input_int8_fused_norm_eligible(ops::LinearPolicy::A16Only, 1)) {
        std::cerr << "band lock: a non-int8 policy reached the fused edge\n";
        ++failures;
    }
    // And a wide call really runs instead of refusing: the byte comparison at these widths is layer 2.
    Case c          = make_case(16, 0xB4C0u);
    const auto packed = linear::make_nvfp4_weight(kAttnRows, kHidden, 4242u);
    DeviceBuffer payload(packed.payload.size());
    CUDA_CHECK(cudaMemcpy(payload.p, packed.payload.data(), packed.payload.size(),
                          cudaMemcpyHostToDevice));
    const Weight weight   = packed.device_weight(payload.p);
    const std::size_t qb  = static_cast<std::size_t>(6144) * 16 * sizeof(std::uint16_t);
    const std::size_t kb  = static_cast<std::size_t>(1024) * 16 * sizeof(std::uint16_t);
    DeviceBuffer q(qb), g(qb), k(kb), v(kb);
    Tensor q_t(q.p, DType::BF16, {6144, 16});
    Tensor g_t(g.p, DType::BF16, {6144, 16});
    Tensor k_t(k.p, DType::BF16, {1024, 16});
    Tensor v_t(v.p, DType::BF16, {1024, 16});
    WorkspaceArena workspace(512ULL << 20);
    bool threw = false;
    try {
        fused::nvfp4_attn_input_int8_fused_norm_launch(c.x_tensor, c.w_tensor, kEps, weight, q_t,
                                                      g_t, k_t, v_t, workspace,
                                                      ops::detail::Nvfp4Int8Mainloop::Grouped,
                                                      nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    std::cout << "band lock: attn launch at T=16 threw=" << (threw ? 1 : 0) << '\n';
    if (threw) {
        std::cerr << "band lock: the attn fused launch refused a width the unified route serves\n";
        ++failures;
    }
    // The same lock on the gate_up edge, which is wired by its own commit.
    {
        Case wide       = make_case(16, 0xB4C1u);
        const auto packed_up = linear::make_nvfp4_weight(kGateUpRows, kHidden, 9091u);
        DeviceBuffer payload_up(packed_up.payload.size());
        CUDA_CHECK(cudaMemcpy(payload_up.p, packed_up.payload.data(), packed_up.payload.size(),
                              cudaMemcpyHostToDevice));
        const Weight up_weight = packed_up.device_weight(payload_up.p);
        const std::size_t ob   = static_cast<std::size_t>(kSwigluOut) * 16 * sizeof(std::uint16_t);
        DeviceBuffer out_b(ob);
        Tensor out_t(out_b.p, DType::BF16, {kSwigluOut, 16});
        WorkspaceArena ws(512ULL << 20);
        bool threw_up = false;
        try {
            fused::nvfp4_linear_swiglu_int8_fused_norm_launch(
                wide.x_tensor, wide.w_tensor, kEps, up_weight, out_t, ws,
                ops::detail::Nvfp4Int8Mainloop::Grouped, nullptr);
        } catch (const std::invalid_argument&) {
            threw_up = true;
        }
        std::cout << "band lock: swiglu launch at T=16 threw=" << (threw_up ? 1 : 0) << '\n';
        if (threw_up) {
            std::cerr << "band lock: the swiglu fused launch refused a width the unified route serves\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    UnifiedInt8Pin pin;
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 16, 128, 129, 256, 1024}) {
        failures += run_plane_case(tokens);
    }
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 128, 1024}) {
        failures += run_gated_plane_case(tokens);
    }
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 16, 128, 129, 256, 1024}) {
        failures += run_attn_route_case(tokens);
        failures += run_swiglu_route_case(tokens);
    }
    failures += run_band_lock_case();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 int8 fused norm edge kernel\n";
    return failures == 0 ? 0 : 1;
}
