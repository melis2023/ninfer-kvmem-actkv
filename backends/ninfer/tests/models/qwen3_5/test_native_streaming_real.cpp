#include "guarded_main.h"
#include "ninfer/engine.h"
#include "runtime/engine/worker_fault.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<TokenId> short_prompt() { return {248045, 846, 198, 5834, 248046, 198}; }
std::vector<TokenId> long_prompt() {
    auto out = short_prompt();
    const std::vector<TokenId> sentence{791, 5296, 315, 279, 1917, 374, 682, 198};
    while (out.size() < 2053) { out.insert(out.end(), sentence.begin(), sentence.end()); }
    out.resize(2053);
    return out;
}

EngineOptions options(const char* artifact, ExpertResidency residency, bool scoring = false) {
    EngineOptions out;
    out.artifact_path = artifact;
    out.purpose = scoring ? EnginePurpose::CausalScoring : EnginePurpose::Generation;
    out.max_context = scoring ? 512 : 4096;
    out.kv_capacity = KvCapacityPolicy::explicit_capacity(out.max_context);
    out.prefill_chunk = 1024;
    out.expert_residency = residency;
    out.expert_cache_bytes = 128ULL << 20;
    out.context_cache.enabled = false;
    out.enable_vision = false;
    out.max_concurrency = scoring ? 1 : 8;
    // A correctness gate: with the 128 MiB cache, requests queued behind a disk-streamed batch can
    // wait longer than the 30 s default admission timeout.
    out.pending_timeout_ms = 600000;
    out.suspend.enabled = !scoring && residency != ExpertResidency::Device;
    out.recover_invariant_failures = true;
    return out;
}

RequestOptions greedy(std::uint32_t outputs = 4) {
    RequestOptions out;
    out.execution.requested_output_tokens = outputs;
    out.execution.sampling.temperature = 0;
    out.execution.allow_prefix_reuse = false;
    out.stop.include_model_defaults = false;
    return out;
}

std::vector<float> score(const char* artifact, ExpertResidency residency, std::size_t targets = 256) {
    const auto start = std::chrono::steady_clock::now();
    Engine engine(options(artifact, residency, true));
    auto tokens = long_prompt();
    tokens.resize(targets + 1);
    auto result = engine.score_tokens(std::move(tokens), 1);
    require(result.size() == targets, "native streaming score target count differs");
    const auto stats = engine.runtime_stats().experts;
    std::cout << "score residency=" << static_cast<int>(residency) << " targets=" << targets
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " pairs=" << stats.routes << " hit_pairs=" << stats.hits
              << " copied=" << stats.transferred_bytes << std::endl;
    return result;
}

void compare_scores(const std::vector<float>& baseline, const std::vector<float>& candidate,
                      const char* label) {
    require(baseline.size() == candidate.size(), "native streaming score shape differs");
    double baseline_nll = 0, candidate_nll = 0, max_abs = 0;
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        require(std::isfinite(baseline[i]) && std::isfinite(candidate[i]),
                "native streaming score is nonfinite");
        baseline_nll -= baseline[i]; candidate_nll -= candidate[i];
        max_abs = std::max(max_abs, std::abs(double(baseline[i]) - candidate[i]));
    }
    const double mean_delta = (candidate_nll - baseline_nll) / baseline.size();
    std::cout << label << " score targets=" << baseline.size() << " baseline_nll="
              << baseline_nll << " candidate_nll=" << candidate_nll << " mean_nll_delta="
              << mean_delta << " max_logprob_delta=" << max_abs << '\n';
    require(std::abs(mean_delta) <= 0.01 && max_abs <= 0.15,
            "native streaming score differs beyond the fixed full-model comparison bound");
}

std::vector<std::vector<TokenId>> generation(const char* artifact, ExpertResidency residency,
                                 const std::vector<std::vector<TokenId>>& baseline) {
    auto configuration = options(artifact, residency);
    configuration.speculative.backend = SpeculativeBackend::Mtp;
    configuration.speculative.draft_tokens = 3;
    Engine engine(configuration);
    const auto memory = engine.memory_summary();
    std::cout << "residency=" << static_cast<int>(residency) << " weights="
              << memory.weights.capacity_bytes << " workspace=" << memory.workspace.capacity_bytes
              << " cache=" << memory.expert_cache_bytes << '\n';
    const auto solo = engine.generate(engine.prepare_tokens(short_prompt()), greedy(32));
    require(solo.generated_token_ids.size() == 32 && solo.speculative.rounds > 0,
            "native streamed single-request MTP did not finish");
    std::vector<std::vector<TokenId>> reference{solo.generated_token_ids};
    if (!baseline.empty()) { require(reference.front() == baseline.front(), "native single-request MTP differs from resident"); }
    std::vector<GenerationHandle> handles;
    for (unsigned i = 0; i < 8; ++i) {
        handles.push_back(engine.submit(engine.prepare_tokens(short_prompt()), greedy(32)));
    }
    const bool diagnose_batch = std::getenv("NINFER_NATIVE_BATCH_DIAG") != nullptr;
    unsigned lane = 0;
    for (auto& handle : handles) {
        const auto result = handle.wait();
        require(result.generated_token_ids.size() == 32 && result.speculative.rounds > 0,
                "native streamed MTP batch did not finish");
        if (!baseline.empty()) {
            require(result.generated_token_ids == baseline.at(lane + 1),
                    "native MTP batch lane differs from corresponding resident lane");
        }
        reference.push_back(result.generated_token_ids);
        if (diagnose_batch) {
            const auto mismatch = std::mismatch(reference[1].begin(), reference[1].end(), result.generated_token_ids.begin());
            std::cout << "batch residency=" << static_cast<int>(residency) << " lane=" << lane
                      << " first_difference=" << (mismatch.first - reference[1].begin()) << " tokens=";
            for (const auto token : result.generated_token_ids) { std::cout << token << ','; }
            std::cout << std::endl;
        }
        ++lane;
    }
    if (diagnose_batch) { return reference; }
    const auto batched = engine.runtime_stats();
    require(batched.decode_row_rounds > batched.decode_rounds,
            "native streamed requests did not execute a compact decode batch");

    if (residency == ExpertResidency::Device) { return reference; }

    // More than one prompt chunk: the first chunk has no logits and still must finish every DMA.
    const auto full = engine.generate(engine.prepare_tokens(long_prompt()), greedy(1));
    require(full.generated_token_ids.size() == 1 && full.prompt.prompt_tokens == 2053,
            "native streamed long prefill did not finish");

    const auto before_cancel = engine.runtime_stats().computed_prefill_tokens;
    const CancellationView cancel{[&] {
        return engine.runtime_stats().computed_prefill_tokens > before_cancel;
    }};
    const auto aborted = engine.generate(engine.prepare_tokens(long_prompt()), greedy(1), nullptr, cancel);
    require(aborted.finish_reason == FinishReason::Cancelled,
            "native streamed active prefill did not cancel");
    auto probe = engine.generate(engine.prepare_tokens(short_prompt()), greedy(1));
    require(probe.generated_token_ids.size() == 1, "native delivery did not survive abort");

    runtime::arm_worker_failures(1);
    bool failed = false;
    try { (void)engine.generate(engine.prepare_tokens(short_prompt()), greedy(1)); }
    catch (const std::exception&) { failed = true; }
    runtime::arm_worker_failures(0);
    require(failed, "native streamed worker failure was not observed");
    probe = engine.generate(engine.prepare_tokens(short_prompt()), greedy(1));
    require(engine.is_available() && probe.generated_token_ids.size() == 1 &&
                engine.runtime_stats().engine_recoveries > 0,
            "native delivery did not survive worker cleanup");

    const auto transfers = engine.runtime_stats().experts.transferred_bytes;
    const auto suspended = engine.suspend(false);
    require(suspended.state == ModelResidency::Suspended && suspended.releasable_device_bytes > 0,
            "native streamed suspend did not release backing");
    require(engine.resume().state == ModelResidency::Resident,
            "native streamed resume did not restore backing");
    const auto resumed = engine.generate(engine.prepare_tokens(short_prompt()), greedy());
    require(resumed.generated_token_ids.size() == 4 &&
                std::equal(resumed.generated_token_ids.begin(), resumed.generated_token_ids.end(),
                            reference.front().begin()) &&
                engine.runtime_stats().experts.transferred_bytes > transfers,
            "native Graph replay after restore did not refill the invalidated cache");
    const auto stats = engine.runtime_stats().experts;
    require(stats.routes > 0 && stats.hits <= stats.routes && stats.transferred_bytes > 0,
            "native streamed public counters are invalid");
    std::cout << "native residency=" << static_cast<int>(residency) << " routes=" << stats.routes
              << " hits=" << stats.hits << " copied=" << stats.transferred_bytes
              << " admitted=" << stats.admitted << " slots=" << stats.slots << '\n';
    return reference;
}

void profile_prefill(const char* artifact) {
    std::vector<TokenId> resident;
    for (const auto residency : {ExpertResidency::Device, ExpertResidency::Host, ExpertResidency::Disk}) {
        for (const std::uint64_t cache : {0ULL, 128ULL << 20}) {
            if (residency == ExpertResidency::Device && cache != 0) { continue; }
            auto configuration = options(artifact, residency);
            configuration.max_concurrency = 1;
            configuration.suspend.enabled = false;
            configuration.expert_cache_bytes = cache;
            Engine engine(configuration);
            const auto memory = engine.memory_summary();
            const auto load = engine.load_summary();
            std::cout << "PREFILL_MEMORY residency=" << static_cast<int>(residency)
                      << " requested_cache=" << cache << " weight_bytes=" << memory.weights.capacity_bytes
                      << " workspace_bytes=" << memory.workspace.capacity_bytes
                      << " expert_cache_bytes=" << memory.expert_cache_bytes
                      << " pinned_weight_bytes=" << load.pinned_weight_bytes << std::endl;
            for (unsigned trial = 0; trial < 2; ++trial) {
                auto prompt = engine.prepare_tokens(long_prompt());
                const auto before = engine.runtime_stats();
                const auto start = std::chrono::steady_clock::now();
                const auto result = engine.generate(std::move(prompt), greedy(1));
                const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const auto after = engine.runtime_stats();
                require(result.generated_token_ids.size() == 1, "profiled long prefill did not complete");
                if (resident.empty()) { resident = result.generated_token_ids; }
                require(result.generated_token_ids == resident, "profiled resident/host/disk first token differs");
                std::cout << "PREFILL_TIMING residency=" << static_cast<int>(residency)
                          << " requested_cache=" << cache << " trial=" << trial << " prompt_tokens=2053 outputs=1"
                          << " seconds=" << seconds
                          << " computed_prefill=" << after.computed_prefill_tokens - before.computed_prefill_tokens
                          << " route_pairs=" << after.experts.routes - before.experts.routes
                          << " hit_pairs=" << after.experts.hits - before.experts.hits
                          << " copied_bytes=" << after.experts.transferred_bytes - before.experts.transferred_bytes
                          << std::endl;
            }
        }
    }
}

void check_pipeline(const char* artifact) {
    const char* requested = std::getenv("NINFER_NATIVE_PIPELINE_DEVICES");
    const std::string_view selection = requested ? requested : "0,0";
    require(selection == "0,0" || selection == "0,1",
            "NINFER_NATIVE_PIPELINE_DEVICES must be 0,0 or 0,1");
    const bool physical = selection == "0,1";
    const std::vector<int> devices = physical ? std::vector<int>{0, 1} : std::vector<int>{0, 0};
    std::vector<TokenId> baseline;
    for (const auto residency : {ExpertResidency::Device, ExpertResidency::Host, ExpertResidency::Disk}) {
        auto configuration = options(artifact, residency);
        configuration.max_concurrency = 1;
        configuration.max_context = 512;
        configuration.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
        configuration.prefill_chunk = 128;
        configuration.devices = devices;
        configuration.use_cuda_graph = true;
        configuration.speculative = {};
        configuration.suspend.enabled = false;
        // This is a transport/ownership gate. Do not calibrate an unknown card during startup.
        if (physical) { configuration.device_profile = "off"; }
        Engine engine(configuration);
        const auto memory = engine.memory_summary();
        require(memory.devices.size() == 2, "native pipeline did not publish two ranks");
        std::uint64_t total_cache = 0;
        for (std::size_t rank = 0; rank < memory.devices.size(); ++rank) {
            const auto& m = memory.devices[rank];
            total_cache += m.expert_cache_bytes;
            require(m.device == devices[rank] && m.weights.capacity_bytes > 0 &&
                        m.workspace.capacity_bytes > 0,
                    "native pipeline rank device or allocation differs from the requested topology");
            std::cout << "PIPELINE_MEMORY residency=" << static_cast<int>(residency)
                      << " rank=" << rank << " device=" << m.device
                      << " weight_bytes=" << m.weights.capacity_bytes
                      << " workspace_bytes=" << m.workspace.capacity_bytes
                      << " expert_cache_bytes=" << m.expert_cache_bytes << std::endl;
        }
        require(total_cache == memory.expert_cache_bytes,
                "native pipeline total cache differs from its published rank capacities");
        auto previous = engine.runtime_stats().experts;
        require(previous.routes == 0 && previous.hits == 0 && previous.cpu_routes == 0 &&
                    previous.admitted == 0 && previous.transferred_bytes == 0,
                "native pipeline counters include startup work");
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            const auto result = engine.generate(engine.prepare_tokens(short_prompt()), greedy(4));
            require(result.generated_token_ids.size() == 4, "two-rank native request did not finish");
            if (baseline.empty()) { baseline = result.generated_token_ids; }
            require(result.generated_token_ids == baseline,
                    "two-rank native output differs from the matched resident control");
            const auto stats = engine.runtime_stats().experts;
            require(stats.routes >= previous.routes && stats.hits >= previous.hits &&
                        stats.admitted >= previous.admitted &&
                        stats.transferred_bytes >= previous.transferred_bytes &&
                        stats.hits <= stats.routes && stats.cpu_routes == 0,
                    "two-rank native delivery counters are inconsistent or decreased");
            if (residency == ExpertResidency::Device) {
                require(stats.routes == 0 && stats.transferred_bytes == 0,
                        "resident pipeline unexpectedly used expert delivery");
            } else {
                require(stats.routes > previous.routes &&
                            stats.transferred_bytes > previous.transferred_bytes,
                        "two-rank native request did not deliver expert bytes");
            }
            std::cout << "PIPELINE_TRIAL residency=" << static_cast<int>(residency)
                      << " repeat=" << repeat << " devices=" << selection << " tokens=4"
                      << " route_pairs=" << stats.routes - previous.routes
                      << " hit_pairs=" << stats.hits - previous.hits
                      << " copied_bytes=" << stats.transferred_bytes - previous.transferred_bytes
                      << std::endl;
            previous = stats;
        }
        const auto rounds = engine.runtime_stats();
        require(rounds.decode_rounds > 0, "two-rank native request did not execute decode Graphs");
        std::cout << "PIPELINE_PASS residency=" << static_cast<int>(residency)
                  << " devices=" << selection << " ranks=2 physical_gpus=" << (physical ? 2 : 1)
                  << " graphs=1 repeats=2 decode_rounds=" << rounds.decode_rounds
                  << " graph_measured_bytes=" << memory.cuda_graph_measured_bytes.value_or(0)
                  << " profile=" << (physical ? "off" : "auto")
                  << " route_pairs=" << previous.routes << " hit_pairs=" << previous.hits
                  << " copied_bytes=" << previous.transferred_bytes << std::endl;
    }
}

void check_metric_epoch(const char* artifact) {
    auto configuration = options(artifact, ExpertResidency::Host);
    configuration.max_concurrency = 1;
    configuration.expert_cache_bytes = 0;
    Engine engine(configuration);
    auto previous = engine.runtime_stats().experts;
    require(previous.routes == 0 && previous.transferred_bytes == 0,
            "startup Graph delivery leaked into runtime counters");
    std::uint64_t first_bytes = 0;
    for (unsigned trial = 0; trial < 2; ++trial) {
        const auto result = engine.generate(engine.prepare_tokens(long_prompt()), greedy(1));
        const auto stats = engine.runtime_stats().experts;
        const auto pairs = stats.routes - previous.routes;
        const auto bytes = stats.transferred_bytes - previous.transferred_bytes;
        require(result.generated_token_ids.size() == 1 && pairs == 2053ULL * 40 * 8 && bytes > 0,
                "native request delivery counters include work outside the request");
        if (trial == 0) { first_bytes = bytes; }
        // A layer the transport prefetched for the next call may land on either request.
        require(bytes * 10 >= first_bytes * 9 && bytes * 10 <= first_bytes * 11,
                "zero-cache identical requests copied byte counts more than a layer apart");
        std::cout << "METRIC_EPOCH_PASS trial=" << trial << " route_pairs=" << pairs
                  << " copied_bytes=" << bytes << std::endl;
        previous = stats;
    }
    runtime::arm_worker_failures(1);
    bool failed = false;
    try { (void)engine.generate(engine.prepare_tokens(short_prompt()), greedy(1)); }
    catch (const std::exception&) { failed = true; }
    runtime::arm_worker_failures(0);
    require(failed, "metric epoch check did not observe injected failure");
    (void)engine.generate(engine.prepare_tokens(short_prompt()), greedy(1));
    auto stats = engine.runtime_stats().experts;
    require(stats.routes >= previous.routes && stats.transferred_bytes >= previous.transferred_bytes,
            "native counters decreased after worker recovery");
    previous = stats;
    require(engine.suspend(false).state == ModelResidency::Suspended &&
                engine.resume().state == ModelResidency::Resident,
            "metric epoch suspend/restore did not finish");
    (void)engine.generate(engine.prepare_tokens(short_prompt()), greedy(1));
    stats = engine.runtime_stats().experts;
    require(stats.routes >= previous.routes && stats.transferred_bytes > previous.transferred_bytes,
            "native counters decreased after restore or failed to count delivery");
    std::cout << "METRIC_EPOCH_LIFECYCLE_PASS routes=" << stats.routes
              << " copied_bytes=" << stats.transferred_bytes << std::endl;
}

int run() {
    const char* artifact = std::getenv("NINFER_NATIVE_MOE_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: NINFER_NATIVE_MOE_ARTIFACT is not set\n";
        return 77;
    }
    if (std::getenv("NINFER_NATIVE_METRICS_CHECK")) {
        check_metric_epoch(artifact);
        return 0;
    }
    if (std::getenv("NINFER_NATIVE_PIPELINE_CHECK")) {
        check_pipeline(artifact);
        return 0;
    }
    if (std::getenv("NINFER_NATIVE_BATCH_DIAG")) {
        (void)generation(artifact, ExpertResidency::Device, {});
        (void)generation(artifact, ExpertResidency::Host, {});
        (void)generation(artifact, ExpertResidency::Disk, {});
        return 0;
    }
    if (std::getenv("NINFER_NATIVE_PREFILL_PROFILE")) {
        profile_prefill(artifact);
        return 0;
    }
    if (std::getenv("NINFER_NATIVE_STREAM_DIAG")) {
        for (const std::size_t targets : {1, 8, 64, 256}) {
            const auto resident = score(artifact, ExpertResidency::Device, targets);
            const auto host = score(artifact, ExpertResidency::Host, targets);
            const auto disk = score(artifact, ExpertResidency::Disk, targets);
            double max_delta = 0, sum_delta = 0;
            for (std::size_t i = 0; i < targets; ++i) {
                max_delta = std::max(max_delta, std::abs(double(resident[i]) - host[i]));
                sum_delta += double(resident[i]) - host[i];
                if (i < 8 || std::abs(double(resident[i]) - host[i]) > 1) {
                    std::cout << "score sample " << i << " resident=" << resident[i]
                              << " host=" << host[i] << " disk=" << disk[i] << std::endl;
                }
            }
            std::cout << "diagnostic targets=" << targets << " host_disk_exact=" << (host == disk)
                      << " mean_nll_delta=" << sum_delta / targets << " max_logprob_delta=" << max_delta << std::endl;
        }
        return 0;
    }
    const auto resident = score(artifact, ExpertResidency::Device);
    const auto host = score(artifact, ExpertResidency::Host);
    const auto disk = score(artifact, ExpertResidency::Disk);
    compare_scores(resident, host, "native host");
    compare_scores(resident, disk, "native disk");
    require(host == disk, "native host/disk scores differ for the same delivery bytes");
    const auto tokens = generation(artifact, ExpertResidency::Device, {});
    (void)generation(artifact, ExpertResidency::Host, tokens);
    (void)generation(artifact, ExpertResidency::Disk, tokens);
    std::cout << "native streamed Engine checks passed\n";
    return 0;
}
} // namespace

NINFER_GUARDED_TEST_MAIN(run)
