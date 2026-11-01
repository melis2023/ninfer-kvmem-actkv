// Flint's branched passes on a real artifact (skipped without one):
//   NINFER_FLINT_ARTIFACT  a Flint .ninfer
// The prompt route's rounding depends on a pass's width (its rows, and how its tokens spread over
// the experts), so a branch is compared with a pass of the same shape, bit for bit: alone, it reads
// as a plain chunk of its tokens; beside another branch, it reads the same whatever that branch's
// tokens are (it sees none of them), in either place; after a prompt's last chunk in one pass, the
// sequence keeps the chunk's rows and nothing of the branches. It also prints how far the same
// logits move with the pass around them (the scale a read's labels move by against a chat's).

#include "artifact/reader.h"
#include "core/device.h"
#include "models/flint/executor.h"
#include "models/flint/model.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace flint = ninfer::models::flint;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::vector<float> host_logits(flint::Executor& executor, std::uint32_t rows) {
    const Tensor logits = executor.logits(rows);
    std::vector<__nv_bfloat16> raw(std::size_t(logits.ne[0]) * rows);
    CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
    CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) { out[i] = __bfloat162float(raw[i]); }
    return out;
}

// Row `row` of `rows` host logits.
std::vector<float> column(const std::vector<float>& logits, std::uint32_t rows, std::uint32_t row) {
    const std::size_t vocab = logits.size() / rows;
    return {logits.begin() + std::ptrdiff_t(row * vocab),
            logits.begin() + std::ptrdiff_t((row + 1) * vocab)};
}

double largest_difference(const std::vector<float>& a, const std::vector<float>& b,
                          std::int32_t domain) {
    double worst = 0;
    for (std::int32_t i = 0; i < domain; ++i) {
        worst = std::max(worst, double(std::fabs(a[std::size_t(i)] - b[std::size_t(i)])));
    }
    return worst;
}

// Deterministic ordinary token ids.
std::vector<std::int32_t> tokens(std::size_t n, std::uint32_t seed) {
    std::vector<std::int32_t> out(n);
    std::uint32_t state = seed;
    for (auto& token : out) {
        state = state * 1664525u + 1013904223u;
        token = std::int32_t(1000 + (state >> 8) % 60000);
    }
    return out;
}

std::vector<std::int32_t> joined(const std::vector<std::int32_t>& a,
                                 const std::vector<std::int32_t>& b) {
    std::vector<std::int32_t> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_FLINT_ARTIFACT");
    if (artifact_path == nullptr) {
        std::cout << "SKIP Flint branches: NINFER_FLINT_ARTIFACT unset\n";
        return 77;
    }
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    auto model = flint::load_model(reader, {.artifact = artifact_path}, device);
    const auto domain = std::int32_t(model->resources().public_token_count);
    flint::ExecutorOptions options;
    options.max_context   = 4096;
    options.prefill_chunk = 2048;
    options.sequences     = 1;
    flint::Executor executor(*model, device, options);
    executor.warm_up();

    // Branches A and B, and B2 and A2 of their lengths with other tokens.
    const std::vector<std::int32_t> prefix = tokens(300, 1), a = tokens(51, 2), b = tokens(45, 3),
                                    a2 = tokens(51, 4), b2 = tokens(45, 5);
    const std::uint32_t la = std::uint32_t(a.size()), lb = std::uint32_t(b.size());
    const std::vector<std::uint32_t> ab{la, lb}, ba{lb, la};

    executor.reset(0);
    executor.forward(0, prefix, 1);
    flint::SequenceSnapshot state;
    executor.snapshot(0, state);
    const auto from_state = [&] { executor.restore(0, state); };
    const auto branches = [&](const std::vector<std::int32_t>& run,
                              std::span<const std::uint32_t> lengths) {
        from_state();
        executor.forward_branches(0, run, lengths);
        return host_logits(executor, std::uint32_t(lengths.size()));
    };

    from_state();
    executor.forward(0, a, 1);
    const std::vector<float> a_plain = host_logits(executor, 1);
    const std::vector<float> a_alone = branches(a, std::vector<std::uint32_t>{la});
    expect(executor.position(0) == prefix.size(), "a branch pass leaves the sequence where it was");
    expect(largest_difference(a_alone, a_plain, domain) == 0, "a branch alone reads as a plain chunk");

    const std::vector<float> with_b = branches(joined(a, b), ab), with_b2 = branches(joined(a, b2), ab);
    const std::vector<float> after_b = branches(joined(b, a), ba), after_b2 = branches(joined(b2, a), ba);
    const std::vector<float> a2_then_b = branches(joined(a2, b), ab);
    const double first  = largest_difference(column(with_b, 2, 0), column(with_b2, 2, 0), domain);
    const double second = largest_difference(column(after_b, 2, 1), column(after_b2, 2, 1), domain);
    const double behind = largest_difference(column(with_b, 2, 1), column(a2_then_b, 2, 1), domain);
    std::cout << "a branch against another branch's tokens, max |logit difference|: first " << first
              << ", second " << second << ", second behind another first " << behind << '\n';
    expect(first == 0, "the first branch reads the same whatever the second holds");
    expect(second == 0, "the second branch reads the same whatever the first holds");
    expect(behind == 0, "the second branch reads the same behind another first branch");

    // The scale of the route's rounding: the same row in passes of other shapes, from the state
    // (before the one-pass case below rewrites the prefix's full-attention rows).
    from_state();
    executor.forward(0, joined(a, b), lb + 1);
    const double wider = largest_difference(column(host_logits(executor, lb + 1), lb + 1, 0), a_plain, domain);
    from_state();
    executor.forward(0, std::span(a).first(la - 3), 1);
    executor.forward(0, std::span(a).subspan(la - 3), 1);
    const double vector_rows = largest_difference(host_logits(executor, 1), a_plain, domain);
    std::cout << "A's last logits against its plain chunk: beside B's rows " << wider
              << ", last three as vector rows " << vector_rows << ", first of two branches "
              << largest_difference(column(with_b, 2, 0), a_plain, domain) << '\n';


    // The prompt's last chunk with both branches in one pass: the branches read as in a branch
    // pass of the same shape whatever the other holds, and the sequence ends at the prompt's end
    // with nothing of them: it continues the same after branches of other tokens.
    const std::size_t cut   = 250;
    const auto head         = std::span(prefix).first(cut);
    const auto tail         = std::span(prefix).subspan(cut);
    const auto one_pass_run = [&](const std::vector<std::int32_t>& run) {
        executor.reset(0);
        executor.forward(0, head, 1);
        executor.forward_prompt_branches(0, tail, run, ab);
        std::vector<float> read = host_logits(executor, 2);
        expect(executor.position(0) == prefix.size(),
               "a one-pass read leaves the sequence at its prompt's end");
        executor.forward(0, a, 1);
        return std::pair{read, host_logits(executor, 1)};
    };
    const auto [read_b, next_b]   = one_pass_run(joined(a, b));
    const auto [read_b2, next_b2] = one_pass_run(joined(a, b2));
    const double one_pass_first   = largest_difference(column(read_b, 2, 0), column(read_b2, 2, 0), domain);
    const double one_pass_next    = largest_difference(next_b, next_b2, domain);
    std::cout << "one pass against the other branch's tokens: first " << one_pass_first
              << ", the sequence after it " << one_pass_next << '\n';
    expect(one_pass_first == 0, "a one-pass branch reads the same whatever the other holds");
    expect(one_pass_next == 0, "a one-pass read leaves nothing of its branches in the sequence");

    std::cout << (failures == 0 ? "PASS" : "FAILED") << " Flint branches\n";
    return failures == 0 ? 0 : 1;
}
