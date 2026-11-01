#include "ops/linear/linear_test_common.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/common/route_table.h"
#include "ops/op_tester.h"
#include <optional>
#include <string_view>

#include <utility>
#include <vector>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_nvfp4_a16() {
    std::vector<Invocation> invocations;
    for (int t = 1; t <= 33; ++t) invocations.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) invocations.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 27, 31, 33})
        invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    invocations.push_back({1, CallForm::A16Convenience});
    invocations.push_back({33, CallForm::Policy, ops::LinearPolicy::AllowA8});
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::SampledRows, true, invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::SampledRows, true, invocations});
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        failures += verify_workspace_envelopes(QType::NVFP4, n, k);
    }
    // DFlash2 feature, fused QKV, attention output, convolution projection and selector.
    const std::vector<Invocation> draft_calls{{1},{2},{3},{5},{6},{16},{32},{33},
        {6,CallForm::Policy,ops::LinearPolicy::A16Only,true}};
    for (auto [n,k] : {std::pair{25600,5120}, std::pair{5120,25600},
                       std::pair{6144,5120}, std::pair{5120,4096},
                       std::pair{1280,5120}, std::pair{256,5120}}) {
        failures += run_shape("NVFP4_DFLASH2",ActivationCompute::A16,make_nvfp4_weight,
            {n,k,739U,Comparison::SampledRows,true,draft_calls});
        failures += verify_workspace_envelopes(QType::NVFP4,n,k);
    }
    return failures;
}

// ---------------------------------------------------------------------------
// W4A16 across the two chains, byte for byte.
//
// Why this exists: the engine's route table steers the A16 NVFP4 parents to one of two chains
// (`unified/nvfp4_attn_input`, `unified/nvfp4_gdn_input`, `unified/nvfp4_linear_add`,
// `unified/nvfp4_linear_swiglu`), and every gate in this file so far scores the output against an
// independent reference with a tolerance. A tolerance gate cannot arbitrate "the two chains must
// compose the same arithmetic" -- the int8 family just proved why (its drift moved the reference's
// worst case by 1.3% against a 10x limit and stayed green), and a mixed-route run just produced a
// third output SHA, so the claim had to be measured instead of inferred.
//
// The switch is forced through the table directly rather than through a candidate name, because the
// A16 keys register exactly one candidate ("unified") and an unforced run means "whatever the
// installed profile says" -- which could silently be the same chain twice.
//
// Non-vacuity: the same bytes under both tables would be worth nothing if one arm had quietly
// answered twice, so each case also checks that a perturbation of the unified arm's activation
// changes its output. A comparison blind to one input element is not comparing kernels.
// That the two tables really are two different launches is settled by the code, not by the
// comparison: `nvfp4_dispatch` sends the unified table to `unified::nvfp4_dispatch` and every other
// width to this line's own `shape.a16`, and the fused projections branch the same way on
// `fused_route_table`. So "identical" here is two distinct kernels agreeing, not one kernel
// answering twice -- and the sign-bit control is what proves the byte comparison sees the input.
// ---------------------------------------------------------------------------
int run_a16_two_chain_bytes_data(bool odd) {
    int failures = 0;
    constexpr std::int32_t q_rows = 6144, kv_rows = 1024, qkv_rows = 10240, z_rows = 6144;

    const auto packed_attn = make_nvfp4_weight(14336, 5120, 7301U, odd ? 0.1F : 0.125F,
                                               /*wide_scale_range=*/true);
    const auto packed_gdn  = make_nvfp4_weight(16384, 5120, 7303U, odd ? 0.1F : 0.125F,
                                               /*wide_scale_range=*/true);
    const auto packed_gu   = make_nvfp4_weight(34816, 5120, 7307U, odd ? 0.1F : 0.125F,
                                               /*wide_scale_range=*/true);
    const auto packed_down = make_nvfp4_weight(5120, 17408, 7309U, odd ? 0.1F : 0.125F,
                                               /*wide_scale_range=*/true);
    // The 6144->5120 projection every layer runs after its attention/GGN mix. The first version of
    // this gate never built that shape, and it is the one the model exercises 64 times per token:
    // the e2e run under the A16 family disagreed between the two chains while this gate was green,
    // which is exactly the failure mode the width comment above warns about, transposed to shapes.
    const auto packed_o    = make_nvfp4_weight(5120, 6144, 7311U, odd ? 0.1F : 0.125F,
                                               /*wide_scale_range=*/true);
    const auto attn_blob = test::to_device(packed_attn.payload);
    const auto gdn_blob  = test::to_device(packed_gdn.payload);
    const auto gu_blob   = test::to_device(packed_gu.payload);
    const auto down_blob = test::to_device(packed_down.payload);
    const auto o_blob    = test::to_device(packed_o.payload);
    const Weight attn_weight = packed_attn.device_weight(attn_blob.p);
    const Weight gdn_weight  = packed_gdn.device_weight(gdn_blob.p);
    const Weight gu_weight   = packed_gu.device_weight(gu_blob.p);
    const Weight down_weight = packed_down.device_weight(down_blob.p);
    const Weight o_weight    = packed_o.device_weight(o_blob.p);

    // Deterministic sign-mixed row data, one pattern per (column, token): a wide call reads a
    // prefix of the same buffer, and element 0 is the one the control perturbs.
    const auto activation = [odd](std::int32_t k, std::int32_t tokens, std::uint32_t seed,
                                  bool perturb) {
        std::vector<std::uint16_t> bits(std::size_t(k) * tokens);
        for (std::size_t index = 0; index < bits.size(); ++index) {
            const std::uint32_t hash = std::uint32_t(index) * 2654435761U + seed;
            // Two regimes. The exact one keeps every partial sum representable (multiples of
            // 1/128, power-of-two divisor) so the comparison sees only wiring, never rounding. The
            // odd one scales by 1.7/1023, which is not representable, so FP32 rounding really
            // happens and any difference in reduction order -- or in where the global scale is
            // applied -- changes the bytes.
            const float value = odd ? float(int(hash & 0x3ffU) - 512) * (1.0f / 1023.0F) * 1.7F
                                    : float(int(hash & 0x3ffU) - 512) * (1.0f / 512.0f) * 4.0f;
            bits[index] = test::f32_to_bf16(value);
        }
        if (perturb) { bits[0] = std::uint16_t(bits[0] ^ 0x8000u); }
        return bits;
    };

    const auto one = [&](std::string_view label, std::int32_t k, std::int32_t tokens,
                         const auto& invoke) {
        const auto run = [&](ops::detail::LinearRouteTable table, bool perturb) {
            const auto plane = test::to_device(activation(k, tokens, 7411U, perturb));
            std::vector<std::uint8_t> out;
            ops::detail::force_linear_route_table(table);
            try {
                out = invoke(plane.p);
            } catch (const std::exception& error) {
                std::cerr << label << " T=" << tokens << ": " << error.what() << "\n";
                out.clear();
            }
            ops::detail::force_linear_route_table(std::nullopt);
            return out;
        };
        const auto legacy_bytes    = run(ops::detail::LinearRouteTable::Legacy, false);
        const auto unified_bytes   = run(ops::detail::LinearRouteTable::Unified, false);
        const auto perturbed_bytes = run(ops::detail::LinearRouteTable::Unified, true);
        if (legacy_bytes.empty() || unified_bytes.empty()) { return failures + 1; }
        if (legacy_bytes.size() != unified_bytes.size()) {
            std::cerr << label << " T=" << tokens << ": output size differs, "
                      << legacy_bytes.size() << " vs " << unified_bytes.size() << "\n";
            return failures + 1;
        }
        std::size_t differing = 0;
        for (std::size_t index = 0; index < legacy_bytes.size(); ++index) {
            if (legacy_bytes[index] == unified_bytes[index]) { continue; }
            if (differing < 3) {
                std::cerr << label << " T=" << tokens << ": A16 chains differ at byte " << index
                          << ", " << int(legacy_bytes[index]) << " vs "
                          << int(unified_bytes[index]) << "\n";
            }
            ++differing;
        }
        const bool controlled = perturbed_bytes != unified_bytes;
        std::cout << "a16 two-chain " << label << " T=" << tokens << ": "
                  << (differing == 0 ? "identical" : "DIFFERS") << " of "
                  << legacy_bytes.size() << " B, control=" << (controlled ? "live" : "DEAD")
                  << "\n";
        // This gate is deliberately stricter than upstream's contract: op-development.md §6.3 holds
        // matrix outputs to a normwise bound, not to bitwise equality. Under this routing the wide
        // cells compare one launch against itself (both tables delegate above the small-T ceiling),
        // so their equality is structural and carries no evidence; the cells that discriminate are
        // 3..32 (3..16 for swiglu), where the tables really do run different kernels. A red cell
        // therefore says "our own stricter gate was violated", never "upstream's contract was".
        if (differing != 0) { return failures + 1; }
        if (!controlled) { return failures + 1; }  // a blind comparison proves nothing
        return failures;
    };

    // Every width a speculative round can put in front of the mainloop: a verify pass is K+1 tokens
    // wide, so 3,5,6,7 and the 9..15 band are as much production surface as the powers of two. The
    // first version of this gate ran powers of two only, and the run that later disagreed end to end
    // under the A16 family did so at exactly those untested widths -- a gate that samples the
    // convenient widths is a gate that reports its own blind spot as a pass.
    //
    // The second lesson is the same one applied to the wide band: a 4202-token prompt under a
    // 1024-token prefill chunk ends with a 106-token forward, and a gate that only sampled 64, 128,
    // 512 and 1024 never launched that width. The ragged band below (96..130, around the 32-row
    // chunk the own chain splits wide calls into) is therefore as much production surface as the
    // powers of two are.
    for (const std::int32_t tokens :
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 24, 32, 48, 63, 64, 65, 96, 104, 105,
          106, 107, 108, 112, 120, 127, 128, 129, 130, 512, 1024}) {
        failures = one(odd ? "odd linear" : "linear", 5120, tokens, [&](void* x) {
            test::GuardedDeviceBuffer out(std::size_t(14336) * tokens * 2);
            out.fill(0);
            WorkspaceArena workspace(512ULL << 20);
            Tensor destination(out.data(), DType::BF16, {14336, tokens});
            ops::linear(Tensor(x, DType::BF16, {5120, tokens}), attn_weight, destination,
                        ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            return test::from_device<std::uint8_t>(out.data(), out.bytes());
        });
        failures = one(odd ? "odd linear_o6144" : "linear_o6144", 6144, tokens, [&](void* x) {
            test::GuardedDeviceBuffer out(std::size_t(5120) * tokens * 2);
            out.fill(0);
            WorkspaceArena workspace(512ULL << 20);
            Tensor destination(out.data(), DType::BF16, {5120, tokens});
            ops::linear(Tensor(x, DType::BF16, {6144, tokens}), o_weight, destination,
                        ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            return test::from_device<std::uint8_t>(out.data(), out.bytes());
        });
        // The FFN's gate_up projection is a plain linear on 34816x5120 -- the rotated dense path
        // calls ops::linear and then silu_mul separately, never the fused swiglu op. A gate that
        // only built the 14336 attention shape never launched the geometry the model runs 64 times
        // per token.
        failures = one(odd ? "odd linear_gu34816" : "linear_gu34816", 5120, tokens, [&](void* x) {
            test::GuardedDeviceBuffer out(std::size_t(34816) * tokens * 2);
            out.fill(0);
            WorkspaceArena workspace(512ULL << 20);
            Tensor destination(out.data(), DType::BF16, {34816, tokens});
            ops::linear(Tensor(x, DType::BF16, {5120, tokens}), gu_weight, destination,
                        ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            return test::from_device<std::uint8_t>(out.data(), out.bytes());
        });
        failures = one(odd ? "odd linear_swiglu" : "linear_swiglu", 5120, tokens, [&](void* x) {
            test::GuardedDeviceBuffer out(std::size_t(17408) * tokens * 2);
            out.fill(0);
            WorkspaceArena workspace(512ULL << 20);
            Tensor destination(out.data(), DType::BF16, {17408, tokens});
            ops::linear_swiglu(Tensor(x, DType::BF16, {5120, tokens}), gu_weight, destination,
                               ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            return test::from_device<std::uint8_t>(out.data(), out.bytes());
        });
        failures = one(odd ? "odd attn_input_proj" : "attn_input_proj", 5120, tokens, [&](void* x) {
            test::GuardedDeviceBuffer q(q_rows * tokens * 2), g(q_rows * tokens * 2),
                k(kv_rows * tokens * 2), v(kv_rows * tokens * 2);
            q.fill(0); g.fill(0); k.fill(0); v.fill(0);
            WorkspaceArena workspace(512ULL << 20);
            Tensor q_t(q.data(), DType::BF16, {q_rows, tokens}), g_t(g.data(), DType::BF16,
                                                                     {q_rows, tokens});
            Tensor k_t(k.data(), DType::BF16, {kv_rows, tokens}), v_t(v.data(), DType::BF16,
                                                                      {kv_rows, tokens});
            ops::attn_input_proj(Tensor(x, DType::BF16, {5120, tokens}), attn_weight, q_t, g_t, k_t,
                                 v_t, ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            std::vector<std::uint8_t> bytes;
            for (const auto* buffer : {&q, &g, &k, &v}) {
                failures += buffer->verify_guards("a16 attn");
                auto part = test::from_device<std::uint8_t>(buffer->data(), buffer->bytes());
                bytes.insert(bytes.end(), part.begin(), part.end());
            }
            return bytes;
        });
        failures = one(odd ? "odd gdn_input_proj" : "gdn_input_proj", 5120, tokens, [&](void* x) {
            test::GuardedDeviceBuffer qkv(qkv_rows * tokens * 2), z(z_rows * tokens * 2);
            qkv.fill(0); z.fill(0);
            WorkspaceArena workspace(512ULL << 20);
            Tensor qkv_t(qkv.data(), DType::BF16, {qkv_rows, tokens}),
                z_t(z.data(), DType::BF16, {z_rows, tokens});
            ops::gdn_input_proj(Tensor(x, DType::BF16, {5120, tokens}), gdn_weight, qkv_t, z_t,
                                ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            auto bytes = test::from_device<std::uint8_t>(qkv.data(), qkv.bytes());
            auto part  = test::from_device<std::uint8_t>(z.data(), z.bytes());
            bytes.insert(bytes.end(), part.begin(), part.end());
            return bytes;
        });
        failures = one(odd ? "odd linear_add" : "linear_add", 17408, tokens, [&](void* x) {
            const std::vector<std::uint16_t> residual(std::size_t(5120) * tokens,
                                                      std::uint16_t(0x3f80));
            test::GuardedDeviceBuffer out(std::size_t(5120) * tokens * 2);
            out.copy_from_host(residual.data(), out.bytes());
            WorkspaceArena workspace(512ULL << 20);
            Tensor destination(out.data(), DType::BF16, {5120, tokens});
            ops::linear_add(Tensor(x, DType::BF16, {17408, tokens}), down_weight, destination,
                            ops::LinearPolicy::A16Only, workspace, nullptr);
            test::cuda_synchronize();
            return test::from_device<std::uint8_t>(out.data(), out.bytes());
        });
        // The 6144->5120 attention output projection is a linear_add too, not a plain linear:
        // the model reaches it through the residual helper, so a plain-linear case would leave the
        // op's own dispatch untested. The first version of this gate covered only the 17408
        // down-projection shape.
        failures = one(odd ? "odd linear_add_o6144" : "linear_add_o6144", 6144, tokens,
                       [&](void* x) {
                           const std::vector<std::uint16_t> residual(std::size_t(5120) * tokens,
                                                                     std::uint16_t(0x3f80));
                           test::GuardedDeviceBuffer out(std::size_t(5120) * tokens * 2);
                           out.copy_from_host(residual.data(), out.bytes());
                           WorkspaceArena workspace(512ULL << 20);
                           Tensor destination(out.data(), DType::BF16, {5120, tokens});
                           ops::linear_add(Tensor(x, DType::BF16, {6144, tokens}), o_weight,
                                           destination, ops::LinearPolicy::A16Only, workspace,
                                           nullptr);
                           test::cuda_synchronize();
                           return test::from_device<std::uint8_t>(out.data(), out.bytes());
                       });
    }
    return failures;
}

int run_a16_two_chain_bytes() {
    // The exact regime first (it is what the file's other cases use), then the odd regime, which is
    // the one that can actually fail.
    return run_a16_two_chain_bytes_data(false) + run_a16_two_chain_bytes_data(true);
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = run_nvfp4_a16();
        failures += run_a16_two_chain_bytes();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
