// Gemma 4's assistant against a real artifact (skipped without one):
//   NINFER_GEMMA4_ARTIFACT  a .ninfer converted with --components text,mtp
//   NINFER_GEMMA4_TOKENS    a JSON list of prompt token ids, <bos> first
//   NINFER_GEMMA4_DRAFTS    optional: tools/reference/gemma4.py --assistant output for the same ids
//                           ({"anchor", "drafts", "margins"}), the FP64 oracle of the chain (F16 cache)
//   NINFER_GEMMA4_KV        optional: the cache's --kv-dtype (bf16 by default); a quantized cache
//                           skips the F16 oracle and keeps the verification checks
// It prefills the prompt, takes the greedy anchor, drafts three tokens and checks them against the
// oracle where the oracle's top-2 margin is clear of rounding (0.5 logits). It then verifies the
// anchor and the drafts in one pass on that sequence and decodes the same tokens one at a time on
// another, and checks that both give each position the same greedy token and logits within BF16
// rounding; and that commit() keeps the given prefix and leaves the sequence ready to draft.
// Then a snapshot's restore and a host image's round trip, and the prompt route against the
// decode route over 256 tokens past the sliding window, in the cache's format.

#include "artifact/reader.h"
#include "core/device.h"
#include "models/gemma4/executor.h"
#include "models/gemma4/model.h"

#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma4 = ninfer::models::gemma4;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

nlohmann::json read_json(const char* path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error(std::string("cannot read ") + path); }
    return nlohmann::json::parse(in);
}

std::vector<float> host_logits(gemma4::Executor& executor, std::uint32_t rows) {
    const Tensor logits = executor.logits(rows);
    std::vector<__nv_bfloat16> raw(std::size_t(logits.ne[0]) * rows);
    CUDA_CHECK(cudaStreamSynchronize(executor.head_stream()));
    CUDA_CHECK(cudaMemcpy(raw.data(), logits.data, raw.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) { out[i] = __bfloat162float(raw[i]); }
    return out;
}

KvCacheStorage kv_storage(const char* name) {
    const std::string n = name == nullptr ? "bf16" : name;
    if (n == "bf16") { return KvCacheStorage::BFloat16; }
    if (n == "int8") { return KvCacheStorage::Int8Group64; }
    if (n == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (n == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (n == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (n == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    if (n == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    if (n == "rk4v4-e8") { return KvCacheStorage::RotatedInt4KeyInt4ValueE8; }
    if (n == "rk2v4-e8") { return KvCacheStorage::RotatedE8RootKeyInt4Value; }
    throw std::runtime_error("NINFER_GEMMA4_KV: unknown format " + n);
}

std::int32_t argmax(const float* row, std::int32_t n) {
    std::int32_t best = 0;
    for (std::int32_t i = 1; i < n; ++i) {
        if (row[i] > row[best]) { best = i; }
    }
    return best;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_GEMMA4_ARTIFACT");
    const char* tokens_path   = std::getenv("NINFER_GEMMA4_TOKENS");
    if (artifact_path == nullptr || tokens_path == nullptr) {
        std::cout << "SKIP Gemma 4 assistant: NINFER_GEMMA4_ARTIFACT/NINFER_GEMMA4_TOKENS unset\n";
        return 77;
    }
    std::vector<std::int32_t> prompt = read_json(tokens_path).get<std::vector<std::int32_t>>();
    DeviceContext device(0);
    device.bind_to_current_thread();
    const artifact::Reader reader(artifact_path);
    auto model = gemma4::load_model(reader, {.artifact = artifact_path, .assistant = true}, device);
    const std::int32_t domain = std::int32_t(model->resources().public_token_count);
    constexpr std::uint32_t kSteps = 3;
    gemma4::ExecutorOptions options;
    options.max_context   = 4096;
    options.prefill_chunk = 2048;
    options.sequences     = 2;
    options.draft_tokens  = kSteps;
    options.kv_cache      = kv_storage(std::getenv("NINFER_GEMMA4_KV"));
    const bool quantized  = options.kv_cache != KvCacheStorage::BFloat16;
    gemma4::Executor executor(*model, device, options);
    executor.warm_up();

    const auto n = std::uint32_t(prompt.size());
    executor.forward(0, prompt, 1);
    const std::vector<float> first = host_logits(executor, 1);
    const std::int32_t anchor      = argmax(first.data(), domain);
    expect(executor.can_draft(0), "a prefilled sequence can draft");
    std::vector<std::int32_t> drafts(kSteps);
    const std::uint32_t s0 = 0, s1 = 1;
    executor.draft({&s0, 1}, {&anchor, 1}, drafts, kSteps);
    // Twice more: the eager run, the captured graph and its replay give the same chain.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<std::int32_t> again(kSteps);
        executor.draft({&s0, 1}, {&anchor, 1}, again, kSteps);
        expect(again == drafts, "a replayed draft chain gives the same drafts");
    }
    std::cout << nlohmann::json({{"anchor", anchor}, {"drafts", drafts}}).dump() << '\n';
    const char* reference = std::getenv("NINFER_GEMMA4_DRAFTS");
    if (reference != nullptr && quantized) {
        std::cout << "quantized cache: the F16 oracle's drafts are not compared\n";
    } else if (reference != nullptr) {
        const auto oracle = read_json(reference);
        expect(oracle.at("anchor").get<std::int32_t>() == anchor, "the target's greedy anchor");
        const auto expected = oracle.at("drafts").get<std::vector<std::int32_t>>();
        const auto margins  = oracle.at("margins").get<std::vector<double>>();
        // Each step feeds the last draft: compare until the first draft the oracle's margin leaves
        // to rounding.
        for (std::uint32_t i = 0; i < kSteps && i < expected.size(); ++i) {
            if (margins.at(i) < 0.5) { break; }
            expect(drafts[i] == expected[i],
                   "draft " + std::to_string(i) + ": " + std::to_string(drafts[i]) + " vs oracle " +
                       std::to_string(expected[i]));
        }
    }

    // One verification of the anchor and drafts against the same tokens decoded one by one.
    std::vector<std::int32_t> run{anchor};
    run.insert(run.end(), drafts.begin(), drafts.end());
    executor.verify({&s0, 1}, run);
    const std::vector<float> verified = host_logits(executor, kSteps + 1);
    executor.forward(1, prompt, 1);
    double worst = 0;
    std::size_t equal = 0;
    const std::size_t vocab = verified.size() / (kSteps + 1);
    for (std::uint32_t i = 0; i <= kSteps; ++i) {
        executor.decode({&s1, 1}, {&run[i], 1});
        const std::vector<float> decoded = host_logits(executor, 1);
        const float* v = verified.data() + i * vocab;
        for (std::size_t t = 0; t < std::size_t(domain); ++t) {
            worst = std::max(worst, double(std::fabs(v[t] - decoded[t])));
            equal += v[t] == decoded[t];
        }
        expect(argmax(v, domain) == argmax(decoded.data(), domain),
               "verified and decoded position " + std::to_string(i) + " agree on the greedy token");
    }
    std::cout << "verify vs decode: max |logit difference| " << worst << ", equal "
              << double(equal) / double(std::size_t(domain) * (kSteps + 1)) << '\n';
    expect(worst <= 0.5, "verified logits match decoded ones within BF16 rounding");

    // A snapshot of a sequence's rings resumes it: the same token after a restore gives the same
    // logits.
    {
        gemma4::SequenceSnapshot snap;
        executor.snapshot(1, snap);
        executor.decode({&s1, 1}, {&anchor, 1});
        const std::vector<float> before = host_logits(executor, 1);
        executor.restore(1, snap);
        executor.decode({&s1, 1}, {&anchor, 1});
        expect(host_logits(executor, 1) == before, "a restored sequence decodes as before");
    }

    const std::uint32_t kept = 2;
    executor.commit({&s0, 1}, {&kept, 1});
    expect(executor.position(0) == n + kept, "commit keeps the given prefix");
    expect(executor.can_draft(0), "a committed sequence can draft");
    std::vector<std::int32_t> next(kSteps);
    executor.draft({&s0, 1}, {&run[kept], 1}, next, kSteps);
    // A host image of a sequence moves it to another: the next token from the image decodes as
    // from the original, from its live state and from its checkpoint. The pages are an odd size,
    // so copies split inside rows, and the host waits for none of the copies.
    for (const bool from_checkpoint : {false, true}) {
        gemma4::SequenceSnapshot snap;
        executor.snapshot(1, snap);
        const std::uint32_t position = executor.position(1);
        constexpr std::size_t kPage  = 5'000'011;
        const std::size_t count      = (executor.image_bytes(position) + kPage - 1) / kPage;
        PinnedHostBuffer host(count * kPage);
        std::vector<std::byte*> pages(count);
        for (std::size_t i = 0; i < count; ++i) {
            // In reverse, so consecutive slices of the image are not consecutive in memory.
            pages[i] = static_cast<std::byte*>(host.data()) + (count - 1 - i) * kPage;
        }
        const auto image = executor.save_image(1, from_checkpoint ? &snap : nullptr, pages, kPage);
        expect(image.position == position, "an image keeps its position");
        executor.decode({&s1, 1}, {&anchor, 1});
        const std::vector<float> original = host_logits(executor, 1);
        executor.load_image(0, image, pages, kPage);
        executor.decode({&s0, 1}, {&anchor, 1});
        expect(host_logits(executor, 1) == original,
               std::string("a sequence loaded from a host image decodes as its source (") +
                   (from_checkpoint ? "checkpoint" : "live") + ")");
        executor.restore(1, snap);
    }

    // The prompt route against the decode route past the sliding window. The decode route writes
    // a text: the prompt, then greedy tokens to 2817. The same text prefilled at once leaves the
    // logits of its last 256 rows from the second chunk, which reads the window before it from the
    // stored cache and attends to its own rows as stored; their greedy tokens must be the decoded
    // ones but for near ties (the routes' matrix kernels and attention differ, and the model's 8-bit
    // activations make that difference about a logit on uncertain rows).
    {
        constexpr std::uint32_t kChunk = 2048, kPrefix = 2560, kTail = 256;
        std::vector<std::int32_t> text = prompt;
        executor.reset(1);
        executor.forward(1, text, 1);
        text.push_back(argmax(host_logits(executor, 1).data(), domain));
        while (text.size() < kPrefix + kTail + 1) {
            executor.decode({&s1, 1}, {&text.back(), 1});
            text.push_back(argmax(host_logits(executor, 1).data(), domain));
        }
        const std::span<const std::int32_t> all(text);
        executor.reset(0);
        executor.forward(0, all.first(kChunk), 1);
        executor.forward(0, all.subspan(kChunk, kPrefix + kTail - kChunk), kTail);
        const std::vector<float> prompted = host_logits(executor, kTail);
        const std::size_t vocab = prompted.size() / kTail;
        double nll = 0;
        std::uint32_t agree = 0;
        for (std::uint32_t i = 0; i < kTail; ++i) {
            const float* row = prompted.data() + i * vocab;
            const std::int32_t next = text[kPrefix + i + 1];
            double m = row[0], z = 0;
            for (std::int32_t t = 1; t < domain; ++t) { m = std::max(m, double(row[t])); }
            for (std::int32_t t = 0; t < domain; ++t) { z += std::exp(double(row[t]) - m); }
            nll += m + std::log(z) - double(row[next]);
            agree += argmax(row, domain) == next;
        }
        std::cout << "prompt route over the decode route's last " << kTail
                  << " greedy tokens: agreement " << double(agree) / kTail << ", NLL per token "
                  << nll / kTail << '\n';
        expect(double(agree) / kTail >= 0.85,
               "the prompt route predicts the decode route's greedy tokens but for near ties");
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 assistant\n";
    return failures == 0 ? 0 : 1;
}
