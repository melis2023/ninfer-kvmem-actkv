#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

// Opt-in public Engine regression: text + MTP, with an embedded or named companion ngram table.
// The full model needs two 24 GiB cards; NINFER_FLASH_NEXT_DEVICES defaults to 0,1.
namespace {
int failures = 0;
int target_differences = 0, batch_differences = 0;
void check(bool pass, const std::string& message) {
    if (!pass) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}

ninfer::EngineOptions options(const char* artifact, std::uint32_t drafts = 0) {
    ninfer::EngineOptions o;
    o.artifact_path = artifact;
    if (const char* table = std::getenv("NINFER_FLASH_NEXT_NGRAM_TABLE")) {
        o.ngram_table.path = table;
    }
    const char* devices = std::getenv("NINFER_FLASH_NEXT_DEVICES");
    std::string list = devices ? devices : "0,1";
    for (std::size_t begin = 0; begin < list.size();) {
        const auto end = list.find(',', begin);
        o.devices.push_back(std::stoi(list.substr(begin, end - begin)));
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    o.max_context = 4096;
    o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(12288);
    o.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    o.max_concurrency = 3;
    o.max_pending_requests = 3;
    o.prefill_chunk = 512;
    o.structured_output = true;
    o.context_cache.enabled = false;
    o.startup_observer.callback = [](const ninfer::StartupEvent& event) {
        if (event.status == ninfer::StartupStatus::Complete) {
            std::cerr << "startup phase=" << static_cast<int>(event.phase)
                      << " seconds=" << double(event.elapsed_ns) / 1e9 << '\n';
        }
    };
    if (drafts) {
        o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        o.speculative.draft_tokens = drafts;
    }
    return o;
}

ninfer::PromptInput prompt(bool thinking = false) {
    ninfer::PromptInput input;
    input.messages.push_back(ninfer::ChatMessage{
        .role = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                  .text = "Write a Python function that merges two sorted lists. Include code only."}}});
    input.options.enable_thinking = thinking;
    return input;
}

ninfer::RequestOptions request(std::uint32_t count = 48) {
    ninfer::RequestOptions r;
    r.execution.requested_output_tokens = count;
    r.execution.sampling.temperature = 0.0F;
    r.execution.allow_prefix_reuse = false;
    r.stop.include_model_defaults = false;
    return r;
}

std::vector<ninfer::RequestOptions> cases() {
    std::vector<ninfer::RequestOptions> out;
    for (const auto count : {1U, 3U, 9U, 48U}) { out.push_back(request(count)); }
    auto penalties = request();
    penalties.execution.sampling.presence_penalty = 0.6F;
    penalties.execution.sampling.frequency_penalty = 0.2F;
    out.push_back(penalties);
    auto sampled = penalties;
    sampled.execution.sampling.temperature = 0.8F;
    sampled.execution.sampling.top_k = 20;
    sampled.execution.sampling.top_p = 0.95F;
    sampled.execution.sampling.seed = 7;
    out.push_back(sampled);
    auto json = request();
    json.execution.structured_output.kind = ninfer::StructuredOutputKind::JsonObject;
    json.stop.include_model_defaults = true;
    out.push_back(json);
    auto eos = request(256);
    eos.stop.include_model_defaults = true;
    out.push_back(eos);
    return out;
}

std::vector<ninfer::TokenId> tokens(std::size_t count, std::uint32_t seed) {
    std::vector<ninfer::TokenId> out;
    while (out.size() < count) {
        seed = seed * 1664525U + 1013904223U;
        out.push_back(static_cast<ninfer::TokenId>(1000U + (seed >> 8U) % 30000U));
    }
    return out;
}

class CancelAfterTokens final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation event) override {
        generated = event.generated_tokens;
    }
    void publish(ninfer::OutputDelta) override {}
    std::uint32_t generated = 0;
};

void sampling_test(const char* artifact) {
    for (bool graphs : {true, false}) {
        auto o = options(artifact, 1);
        o.use_cuda_graph = graphs;
        ninfer::Engine engine(o);
        auto requests = cases();
        for (std::size_t c = 0; c < 5; ++c) {
            for (int repeat = 0; repeat < 2; ++repeat) {
                (void)engine.generate(engine.prepare(prompt()), requests[c]);
            }
        }
        requests[5].execution.logprobs = true;
        ninfer::GenerationResult reference;
        for (int repeat = 0; repeat < 24; ++repeat) {
            auto result = engine.generate(engine.prepare(prompt()), requests[5]);
            std::cout << "sampling graphs=" << graphs << " repeat=" << repeat << " tokens=";
            for (const auto id : result.generated_token_ids) { std::cout << id << ','; }
            std::cout << " content_bytes=" << result.content.size()
                      << " finish=" << static_cast<int>(result.finish_reason)
                      << " rounds=" << result.speculative.rounds
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens << '\n';
            if (repeat == 0) { reference = result; continue; }
            const auto mismatch = std::mismatch(
                reference.generated_token_ids.begin(), reference.generated_token_ids.end(),
                result.generated_token_ids.begin(), result.generated_token_ids.end());
            const auto at = std::size_t(mismatch.first - reference.generated_token_ids.begin());
            if (at < reference.content_logprobs.size() && at < result.content_logprobs.size()) {
                for (std::size_t j = 0; j < ninfer::kMaximumTokenLogprobs; ++j) {
                    std::cout << "sampling difference at=" << at << " rank=" << j
                              << " first=" << reference.content_logprobs[at].top_ids[j] << ':'
                              << reference.content_logprobs[at].top_values[j]
                              << " repeated=" << result.content_logprobs[at].top_ids[j] << ':'
                              << result.content_logprobs[at].top_values[j] << '\n';
                }
            }
            check(result.generated_token_ids == reference.generated_token_ids &&
                      result.content == reference.content &&
                      result.finish_reason == reference.finish_reason,
                  "seeded sampling repeats, graphs=" + std::to_string(graphs));
        }
    }
}

void cache_test(const char* artifact) {
    const bool compare_only = std::getenv("NINFER_FLASH_NEXT_CACHE_COMPARE_ONLY") != nullptr;
    const auto directory = std::filesystem::temp_directory_path() /
        ("ninfer-flash-cache-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    const auto first = tokens(700, 1);
    auto extended = first;
    const auto suffix = tokens(30, 2);
    extended.insert(extended.end(), suffix.begin(), suffix.end());
    const auto pressure = tokens(700, 3);
    auto r = request(8);
    r.execution.allow_prefix_reuse = true;
    std::vector<ninfer::TokenId> enabled_reference, disabled_reference, prefix_reference;
    for (bool disabled : {false, true}) {
        if (disabled && compare_only) { continue; }
        auto o = options(artifact, 4);
        o.max_concurrency = 1;
        o.ngram_table.disabled = disabled;
        if (disabled) { o.ngram_table.path.clear(); }
        ninfer::Engine engine(o);
        auto result = engine.generate(engine.prepare_tokens(extended), r);
        (disabled ? disabled_reference : enabled_reference) = result.generated_token_ids;
    }
    auto cached = options(artifact, 4);
    cached.max_concurrency = 1;
    cached.context_cache.enabled = true;
    // One ~130 MiB image fits, two do not: exercise real spill rather than disable all storage.
    cached.context_cache.host_cache_budget_bytes = 192ULL << 20;
    cached.context_cache.disk_kv_path = directory;
    cached.context_cache.disk_kv_restore = true;
    {
        ninfer::Engine engine(cached);
        (void)engine.generate(engine.prepare_tokens(first), r);
        const auto live = engine.generate(engine.prepare_tokens(extended), r);
        check(live.reused_prompt_tokens == first.size(), "live comparison uses the same 700-token prefix");
        prefix_reference = live.generated_token_ids;
        std::cout << "CACHE_LIVE_EQUALS_COLD " << (prefix_reference == enabled_reference) << '\n';
        (void)engine.generate(engine.prepare_tokens(pressure), r);
    }
    std::size_t files = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        files += entry.path().extension() == ".img" ? 1 : 0;
    }
    check(files > 0, "the host budget spills actual images to disk");
    {
        ninfer::Engine engine(cached);
        const auto restored = engine.generate(engine.prepare_tokens(extended), r);
        check(restored.reused_prompt_tokens == first.size(), "restart restores the same disk prefix");
        // Cold prefill uses 512+218 columns; prefix reuse uses 512+188, then 30.
        // Compare the same computation history across live and disk state. Record cold
        // differences separately under the numerical, fixed-mode repeatability contract.
        check(restored.generated_token_ids == prefix_reference,
              "disk-restored MTP continuation agrees with the live prefix state");
        std::cout << "CACHE_RESTORED_EQUALS_LIVE " << (restored.generated_token_ids == prefix_reference)
                  << " restored_tokens=";
        for (const auto token : restored.generated_token_ids) { std::cout << token << ','; }
        std::cout << " live_tokens=";
        for (const auto token : prefix_reference) { std::cout << token << ','; }
        std::cout << " cold_tokens=";
        for (const auto token : enabled_reference) { std::cout << token << ','; }
        std::cout << '\n';
    }
    if (compare_only) {
        std::filesystem::remove_all(directory);
        return;
    }
    {
        auto different_width = cached;
        different_width.speculative.draft_tokens = 1;
        ninfer::Engine engine(different_width);
        const auto isolated = engine.generate(engine.prepare_tokens(extended), r);
        check(isolated.reused_prompt_tokens == 0,
              "a different verify width cannot restore the old MTP image");
    }
    {
        cached.ngram_table.disabled = true;
        cached.ngram_table.path.clear();
        ninfer::Engine engine(cached);
        const auto isolated = engine.generate(engine.prepare_tokens(extended), r);
        check(isolated.reused_prompt_tokens == 0,
              "no-ngram run cannot restore table-enabled images");
        check(isolated.generated_token_ids == disabled_reference,
              "no-ngram continuation agrees with its own cold generation");
    }
    std::filesystem::remove_all(directory);
    std::cout << "disk cache: " << files << " images; failures=" << failures << '\n';
}

void ngram_cache_test(const char* artifact) {
    const auto input = tokens(700, 19); // two prefill chunks; identical shape in every mode
    for (const auto drafts : {0U, 4U}) {
        ninfer::GenerationResult reference;
        for (int mode = 0; mode < 3; ++mode) {
            auto o = options(artifact, drafts);
            o.max_concurrency = 1;
            o.ngram_table.ram_budget_bytes = mode == 0 ? 0 : (std::uint64_t{4} << 30U);
            o.ngram_table.io = mode == 2 ? ninfer::NgramIo::Direct : ninfer::NgramIo::Buffered;
            ninfer::Engine engine(o);
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto before = engine.runtime_stats().ngram_table;
                const auto start = std::chrono::steady_clock::now();
                const auto result = engine.generate(engine.prepare_tokens(input), request(24));
                const auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                const auto after = engine.runtime_stats().ngram_table;
                if (mode == 0 && repeat == 0) { reference = result; }
                check(result.generated_token_ids == reference.generated_token_ids &&
                          result.content == reference.content &&
                          result.finish_reason == reference.finish_reason,
                      "disk n-gram cache/I/O/lookahead preserve fixed-width output");
                const auto hits = after.resident_rows - before.resident_rows;
                if (mode == 0) { check(hits == 0, "zero disk budget serves no cached rows"); }
                if (mode != 0 && repeat == 1) {
                    check(hits == after.rows - before.rows,
                          "repeated request reuses every n-gram row without file reads");
                }
                std::cout << "NGRAM_CACHE K=" << drafts << " mode=" << mode << " repeat=" << repeat
                          << " seconds=" << elapsed << " rows=" << after.rows - before.rows
                          << " hits=" << hits << " read_seconds=" << after.read_seconds - before.read_seconds
                          << " stall_seconds=" << after.stall_seconds - before.stall_seconds << '\n';
            }
        }
    }
    std::cout << "NGRAM_CACHE_DONE failures=" << failures << '\n';
}

void layer_slice_test(const char* artifact) {
    auto o = options(artifact);
    o.devices = {0};
    o.max_concurrency = 1;
    const bool with_ple = !o.ngram_table.path.empty();
    {
        ninfer::Engine engine(o);
        for (const auto count : {32U, 700U}) {
            const auto input = tokens(count, 23);
            const auto first = engine.generate(engine.prepare_tokens(input), request(16));
            const auto repeat = engine.generate(engine.prepare_tokens(input), request(16));
            check(first.generated_token_ids.size() == 16, "slice reaches the output budget");
            check(first.generated_token_ids == repeat.generated_token_ids &&
                      first.content == repeat.content && first.finish_reason == repeat.finish_reason,
                  "slice repeats with fixed execution settings");
        }
        const auto rows = engine.runtime_stats().ngram_table.rows;
        check(with_ple ? rows > 0 : rows == 0, "slice reads n-gram rows exactly when it contains PLE");
    }
    o.ngram_table.path = with_ple ? "" : "unused-table.ninfer";
    bool refused = false;
    try {
        ninfer::Engine invalid(o);
    } catch (const std::exception& error) {
        refused = std::string_view(error.what()).find(with_ple ? "--ngram-table" : "without PLE layers") !=
                  std::string_view::npos;
    }
    check(refused, with_ple ? "PLE slice requires its companion table" :
                             "slice without PLE refuses unused table options");
    std::cout << "LAYER_SLICE_DONE failures=" << failures << '\n';
}

void hybrid_test(const char* artifact) {
    const auto input = tokens(128, 47);
    const char* direct_option = std::getenv("NINFER_FLASH_NEXT_HYBRID_DIRECT");
    const bool direct = direct_option && std::string_view(direct_option) == "1";
    for (const auto drafts : {0U, 4U}) {
        auto o = options(artifact, drafts);
        if (direct) { o.ngram_table.io = ninfer::NgramIo::Direct; }
        o.devices = {0};
        o.max_concurrency = 1;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        o.expert_residency = ninfer::ExpertResidency::Host;
        o.expert_cache_bytes = std::uint64_t{4} << 30;
        o.hybrid_experts.dma_share = 0.5F;
        o.hybrid_experts.cpu_threads = 8;
        const char* job = std::getenv("NINFER_JOB_DIR");
        o.hybrid_experts.record_profile = (job ? std::filesystem::path(job) :
            std::filesystem::current_path()) / ("hybrid-routing-k" + std::to_string(drafts) + ".json");
        ninfer::Engine engine(o);
        std::vector<ninfer::TokenId> expected;
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            const auto result = engine.generate(engine.prepare_tokens(input), request(16));
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            if (repeat == 0) { expected = result.generated_token_ids; }
            check(result.generated_token_ids == expected && expected.size() == 16 &&
                      result.finish_reason == ninfer::FinishReason::OutputLimit,
                  "hybrid fixed split repeats through the public Engine");
            if (drafts) { check(result.speculative.rounds > 0, "hybrid actually ran MTP"); }
            std::cout << "HYBRID_ENGINE K=" << drafts << " repeat=" << repeat
                      << " ngram_io=" << (direct ? "direct" : "buffered")
                      << " prompt=" << input.size() << " generated=" << result.generated_token_ids.size()
                      << " seconds=" << elapsed << '\n';
        }
        CancelAfterTokens observer;
        auto handle = engine.submit(engine.prepare_tokens(input), request(64),
            ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
        const auto cancelled = handle.wait(&observer,
            ninfer::CancellationView([&] { return observer.generated >= 3; }));
        check(cancelled.finish_reason == ninfer::FinishReason::Cancelled && observer.generated >= 3,
              "hybrid active generation cancels through the public Engine");
        check(engine.generate(engine.prepare_tokens(input), request(16)).generated_token_ids == expected,
              "hybrid Engine recovers after cancellation");
        check(std::filesystem::is_regular_file(o.hybrid_experts.record_profile) &&
                  std::filesystem::file_size(o.hybrid_experts.record_profile) > 0,
              "hybrid Engine records its routing profile");
        check(engine.runtime_stats().ngram_table.rows > 0, "hybrid Engine reads its n-gram table");
    }
    std::cout << "HYBRID_ENGINE_DONE failures=" << failures << '\n';
}

// At floor one, a wider draft chain must stop after, and verify exactly, the same first draft as
// K=1. Repeated requests exercise the per-step graphs' capture and replay, width-specific
// verification graphs, logprob strides and recovery. Without --spec, host experts draft
// kAutomaticDraftTokens a round with the floor kAutomaticDraftMinP.
void draft_confidence_test(const char* artifact) {
    const auto input = tokens(128, 47);
    std::vector<ninfer::TokenId> one_draft;
    struct Case {
        unsigned drafts;
        float floor;
        bool automatic;
    };
    for (const Case c : std::vector<Case>{{1, 0, false}, {4, 1, false}, {0, 0, true}, {4, 0, false}}) {
        auto o = options(artifact, c.drafts);
        o.devices = {0};
        o.max_concurrency = 1;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        o.expert_residency = ninfer::ExpertResidency::Host;
        o.expert_cache_bytes = std::uint64_t{4} << 30;
        o.speculative.draft_min_p = c.floor;
        o.speculative.automatic   = c.automatic;
        ninfer::Engine engine(o);
        const auto& resolved = engine.options().speculative;
        const unsigned drafts = resolved.draft_tokens;
        const float floor     = resolved.draft_min_p;
        if (c.automatic) {
            check(resolved.backend == ninfer::SpeculativeBackend::Mtp &&
                      drafts == ninfer::SpeculativeOptions::kAutomaticDraftTokens &&
                      floor == ninfer::SpeculativeOptions::kAutomaticDraftMinP,
                  "host experts draft with the automatic width and floor");
        }
        auto r = request(64);
        r.execution.logprobs = true;
        std::vector<ninfer::TokenId> expected;
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = engine.generate(engine.prepare_tokens(input), r);
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            if (repeat == 0) { expected = result.generated_token_ids; }
            check(result.generated_token_ids == expected && expected.size() == 64 &&
                      result.finish_reason == ninfer::FinishReason::OutputLimit,
                  "draft confidence repeats with width-specific graphs");
            check(result.speculative.rounds > 0, "confidence check actually ran MTP");
            if (floor == 1) {
                check(result.generated_token_ids == one_draft,
                      "floor one agrees with K=1 including committed recurrent/MTP state");
                check(result.speculative.accepted_per_position[1] == 0,
                      "floor one accepted a draft past its confidence prefix");
                check(result.speculative.drafted_tokens == result.speculative.rounds,
                      "floor one ends every draft chain after its first step");
            }
            std::cout << "DRAFT_CONFIDENCE K=" << drafts << " floor=" << floor
                      << " repeat=" << repeat << " seconds=" << elapsed
                      << " rounds=" << result.speculative.rounds
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens << '\n';
        }
        if (drafts == 1) { one_draft = expected; }
        if (c.automatic) {
            CancelAfterTokens observer;
            auto handle = engine.submit(engine.prepare_tokens(input), request(64),
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
            const auto cancelled = handle.wait(&observer,
                ninfer::CancellationView([&] { return observer.generated >= 3; }));
            check(cancelled.finish_reason == ninfer::FinishReason::Cancelled,
                  "variable-width generation cancels");
            check(engine.generate(engine.prepare_tokens(input), r).generated_token_ids == expected,
                  "variable-width generation recovers after cancellation");
        }
    }
    // Different confidence prefixes share a compact physical verification width. Exercise
    // its per-sequence offsets and live extents on the resident two-device path as well.
    const char* batched = std::getenv("NINFER_FLASH_NEXT_CONFIDENCE_BATCHED");
    if (batched && std::string_view(batched) == "1") {
        for (const float floor : {1.0F, 0.3F}) {
            auto o = options(artifact, 4);
            o.devices = {0, 1};
            o.speculative.draft_min_p = floor;
            ninfer::Engine engine(o);
            std::array<std::vector<ninfer::TokenId>, 3> expected;
            auto r = request(32);
            r.execution.logprobs = true;
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                std::vector<ninfer::GenerationHandle> pending;
                for (unsigned j = 0; j < 3; ++j) {
                    pending.push_back(engine.submit(engine.prepare_tokens(tokens(64 + 16 * j, 47 + j)), r));
                }
                for (unsigned j = 0; j < 3; ++j) {
                    const auto result = pending[j].wait();
                    if (repeat == 0) { expected[j] = result.generated_token_ids; }
                    check(result.generated_token_ids == expected[j] && expected[j].size() == 32,
                          "batched confidence prefixes preserve sequence state and repeat");
                    check(result.speculative.rounds > 0, "batched confidence check ran MTP");
                    if (floor == 1) {
                        check(result.speculative.accepted_per_position[1] == 0,
                              "batched floor one accepted past its confidence prefix");
                    }
                }
            }
        }
    }
    std::cout << "DRAFT_CONFIDENCE_DONE failures=" << failures << '\n';
}

int run_test() {
    std::cout << std::unitbuf;
    const char* artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (!artifact || !*artifact) { return 77; }
    if (const char* confidence = std::getenv("NINFER_FLASH_NEXT_CONFIDENCE_ONLY");
        confidence && std::string_view(confidence) == "1") {
        draft_confidence_test(artifact);
        return failures ? 1 : 0;
    }
    if (const char* hybrid_only = std::getenv("NINFER_FLASH_NEXT_HYBRID_ONLY");
        hybrid_only && std::string_view(hybrid_only) == "1") {
        hybrid_test(artifact);
        return failures ? 1 : 0;
    }
    if (const char* slice_only = std::getenv("NINFER_FLASH_NEXT_SLICE_ONLY");
        slice_only && std::string_view(slice_only) == "1") {
        layer_slice_test(artifact);
        return failures ? 1 : 0;
    }
    if (const char* ngram_only = std::getenv("NINFER_FLASH_NEXT_NGRAM_ONLY");
        ngram_only && std::string_view(ngram_only) == "1") {
        ngram_cache_test(artifact);
        return failures ? 1 : 0;
    }
    if (const char* sampling_only = std::getenv("NINFER_FLASH_NEXT_SAMPLING_ONLY");
        sampling_only && std::string_view(sampling_only) == "1") {
        sampling_test(artifact);
        return failures ? 1 : 0;
    }
    if (const char* cache_only = std::getenv("NINFER_FLASH_NEXT_CACHE_ONLY");
        cache_only && std::string_view(cache_only) == "1") {
        cache_test(artifact);
        return failures ? 1 : 0;
    }
    auto requests = cases();
    std::vector<ninfer::GenerationResult> reference;
    {
        ninfer::Engine plain(options(artifact));
        for (const auto& r : requests) {
            reference.push_back(plain.generate(plain.prepare(prompt()), r));
        }
        // Stop at a token drawn by the target, including a stop inside a verification round.
        auto stop = request();
        check(reference[3].generated_token_ids.size() > 5, "reference has a stop target");
        if (reference[3].generated_token_ids.size() > 5) {
            stop.stop.token_ids.push_back(reference[3].generated_token_ids[5]);
            requests.push_back(stop);
            reference.push_back(plain.generate(plain.prepare(prompt()), stop));
        }
    }
    check(reference[7].finish_reason == ninfer::FinishReason::StopToken,
          "the default EOS token actually terminates the reference");
    for (const auto k : {1U, 4U, 8U, 15U}) {
        ninfer::Engine engine(options(artifact, k));
        std::uint64_t rounds = 0;
        std::vector<ninfer::TokenId> sampled_reference;
        for (std::size_t c = 0; c < requests.size(); ++c) {
            std::cout << "run K=" << k << " case=" << c << '\n';
            const auto result = engine.generate(engine.prepare(prompt()), requests[c]);
            const auto label = "K=" + std::to_string(k) + " case=" + std::to_string(c);
            if (c == 5) {
                // Sampling and speculative rejection use different RNG purposes by contract.
                // The distribution oracle lives in the Op tests; exact seeded repetition here
                // compares the same backend, while greedy requests compare to the target.
                const auto repeated = engine.generate(engine.prepare(prompt()), requests[c]);
                sampled_reference = result.generated_token_ids;
                if (result.generated_token_ids != repeated.generated_token_ids ||
                    result.content != repeated.content || result.finish_reason != repeated.finish_reason) {
                    const auto mismatch = std::mismatch(
                        result.generated_token_ids.begin(), result.generated_token_ids.end(),
                        repeated.generated_token_ids.begin(), repeated.generated_token_ids.end());
                    std::cerr << "REPEAT_DIFFERENCE " << label << " at="
                              << (mismatch.first - result.generated_token_ids.begin())
                              << " content_bytes=" << result.content.size() << ',' << repeated.content.size()
                              << " finish=" << static_cast<int>(result.finish_reason) << ','
                              << static_cast<int>(repeated.finish_reason) << '\n';
                }
                check(repeated.generated_token_ids == result.generated_token_ids &&
                          repeated.content == result.content &&
                          repeated.finish_reason == result.finish_reason,
                      label + " repeats with the same seed and backend");
                check(result.generated_token_ids.size() == requests[c].execution.requested_output_tokens &&
                          result.finish_reason == ninfer::FinishReason::OutputLimit,
                      label + " respects the sampling output budget");
            } else {
                if (result.generated_token_ids != reference[c].generated_token_ids) {
                    ++target_differences;
                    const auto mismatch = std::mismatch(
                        result.generated_token_ids.begin(), result.generated_token_ids.end(),
                        reference[c].generated_token_ids.begin(), reference[c].generated_token_ids.end());
                    std::cerr << "TARGET_DIFFERENCE " << label << " at="
                              << (mismatch.first - result.generated_token_ids.begin())
                              << " plain_tokens=" << reference[c].generated_token_ids.size()
                              << " mtp_tokens=" << result.generated_token_ids.size() << '\n';
                }
                // The independent Op oracles qualify changed execution widths numerically;
                // exact repetition here holds the backend, width and request fixed.
                check(result.finish_reason == reference[c].finish_reason &&
                          result.generated_token_ids.size() <=
                              requests[c].execution.requested_output_tokens,
                      label + " respects the output policy");
                if (result.finish_reason == ninfer::FinishReason::OutputLimit) {
                    check(result.generated_token_ids.size() ==
                              requests[c].execution.requested_output_tokens,
                          label + " consumes exactly its output budget");
                }
                const auto repeated = engine.generate(engine.prepare(prompt()), requests[c]);
                check(repeated.generated_token_ids == result.generated_token_ids &&
                          repeated.content == result.content &&
                          repeated.finish_reason == result.finish_reason,
                      label + " repeats in the same backend");
            }
            rounds += result.speculative.rounds;
        }
        check(rounds > 0, "K=" + std::to_string(k) + " actually verifies drafts");
        // Report changed batch widths; enforce each row's own output policy.
        auto a = engine.submit(engine.prepare(prompt()), requests[3]);
        auto b = engine.submit(engine.prepare(prompt()), requests[4]);
        auto c = engine.submit(engine.prepare(prompt()), requests[5]);
        const std::array batch_results{a.wait(), b.wait(), c.wait()};
        const std::array serial_tokens{reference[3].generated_token_ids,
                                      reference[4].generated_token_ids, sampled_reference};
        for (std::size_t row = 0; row < batch_results.size(); ++row) {
            const auto& result = batch_results[row];
            if (result.generated_token_ids != serial_tokens[row]) {
                ++batch_differences;
                std::cout << "BATCH_DIFFERENCE K=" << k << " row=" << row << '\n';
            }
            check(result.generated_token_ids.size() == 48 &&
                      result.finish_reason == ninfer::FinishReason::OutputLimit,
                  "batch row=" + std::to_string(row) + " respects its output budget");
        }
        const auto cancelled = engine.generate(engine.prepare(prompt()), request(), nullptr,
                                 ninfer::CancellationView([] { return true; }));
        check(cancelled.finish_reason == ninfer::FinishReason::Cancelled,
              "cancellation reaches a terminal result");
        CancelAfterTokens observer;
        auto active = request(256);
        auto handle = engine.submit(engine.prepare(prompt()), active,
            ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
        const auto stopped = handle.wait(&observer,
            ninfer::CancellationView([&] { return observer.generated >= 3; }));
        check(stopped.finish_reason == ninfer::FinishReason::Cancelled && observer.generated >= 3,
              "an active decoding request cancels");
        check(engine.generate(engine.prepare(prompt()), requests[1]).generated_token_ids ==
                  reference[1].generated_token_ids, "Engine serves after cancellation");
        for (auto thinking_budget : {1U, 3U, 9U}) {
            auto limited = request(96);
            limited.execution.thinking.budget = thinking_budget;
            if (thinking_budget == 3) {
                limited.execution.post_thinking_sampling = ninfer::SamplingOverrides{
                    .temperature = 0.0F, .seed = 7};
            }
            const auto result = engine.generate(engine.prepare(prompt(true)), limited);
            check(result.thinking.effective_budget == thinking_budget &&
                      result.thinking.model_thinking_tokens <= thinking_budget &&
                      result.thinking.applied && result.thinking.injected_tokens > 0,
                  "K=" + std::to_string(k) + " thinking budget=" +
                      std::to_string(thinking_budget) + " applies its control suffix");
            check(result.generated_token_ids.size() == 96 &&
                      result.finish_reason == ninfer::FinishReason::OutputLimit,
                  "thinking control respects the total output budget");
            if (thinking_budget == 3) {
                check(result.thinking.post_thinking_sampling,
                      "sampling switches after the thinking control suffix");
            }
        }
        std::cout << "MTP K=" << k << " rounds=" << rounds << " failures=" << failures << '\n';
    }
    cache_test(artifact);
    std::cout << "ENGINE_QUALIFICATION target_differences=" << target_differences
              << " batch_differences=" << batch_differences << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}
} // namespace

NINFER_GUARDED_TEST_MAIN(run_test)
