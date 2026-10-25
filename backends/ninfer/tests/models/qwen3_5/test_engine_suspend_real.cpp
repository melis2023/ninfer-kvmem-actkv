// A suspended model must come back exactly as it was.
//
// Greedy output is the oracle. A model with suspend enabled generates the same tokens as one
// without it (its memory only sits at fixed addresses); after a suspend and a resume, through an
// explicit call or a request that resumes the model on its own, it generates the same tokens again
// and still reuses the prefix of a conversation it retained before the suspend. While suspended the
// device memory is released, and a model held suspended refuses requests instead of running them.
//
// Every configuration runs the same requests: the plain path, a pinned snapshot, a host copy of the
// weights, MTP (when the artifact carries it), the hybrid prefix cache and a two-stage split (both
// stages on device 0 unless NINFER_TEST_DEVICE_IDS lists more).

#include "guarded_main.h"
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Configuration {
    std::string label;
    bool suspend          = true;
    bool pinned           = false;
    bool host_weights     = false;
    bool mtp              = false;
    bool hybrid_cache     = false;
    std::vector<int> devices;
};

std::vector<int> stage_devices(const std::vector<int>& requested) {
    const char* ids = std::getenv("NINFER_TEST_DEVICE_IDS");
    if (ids == nullptr || *ids == '\0' || requested.empty()) { return requested; }
    std::vector<int> available;
    std::istringstream list{std::string(ids)};
    for (std::string item; std::getline(list, item, ',');) { available.push_back(std::stoi(item)); }
    std::vector<int> out;
    for (std::size_t stage = 0; stage < requested.size(); ++stage) {
        out.push_back(available[stage % available.size()]);
    }
    return out;
}

ninfer::EngineOptions engine_options(const char* artifact, const Configuration& configuration) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk = 512;
    options.kv_cache      = ninfer::KvCacheStorage::Int8Group64;
    options.suspend.enabled         = configuration.suspend;
    options.suspend.snapshot_memory = configuration.pinned ? ninfer::SuspendSnapshotMemory::Pinned
                                                           : ninfer::SuspendSnapshotMemory::Pageable;
    options.suspend.weights = configuration.host_weights ? ninfer::SuspendWeightSource::Host
                                                         : ninfer::SuspendWeightSource::Artifact;
    if (configuration.mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    if (configuration.hybrid_cache) {
        options.context_cache.mode = ninfer::ContextCacheMode::Hybrid;
    }
    options.devices = stage_devices(configuration.devices);
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> long_prompt() {
    std::vector<ninfer::TokenId> prompt;
    prompt.push_back(248045);
    for (std::uint32_t index = 0; index < 1300; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + (index * 37U) % 5000U));
    }
    return prompt;
}

std::vector<ninfer::TokenId> short_prompt() { return {248045, 846, 198, 5834, 248046, 198}; }

struct Outputs {
    std::vector<ninfer::TokenId> short_run;
    std::vector<ninfer::TokenId> long_run;
    std::vector<ninfer::TokenId> continued_run;
    std::uint32_t reused_tokens = 0;

    bool operator==(const Outputs&) const = default;
};

// The short and long runs, then a continuation of the long one that the context cache serves in
// part from the long run's retained state.
Outputs run_requests(ninfer::Engine& engine, const std::vector<ninfer::TokenId>* long_output) {
    Outputs out;
    out.short_run =
        engine.generate(engine.prepare_tokens(short_prompt()), greedy(24)).generated_token_ids;
    if (long_output == nullptr) {
        out.long_run = engine.generate(engine.prepare_tokens(long_prompt()), greedy(24, true))
                           .generated_token_ids;
    } else {
        out.long_run = *long_output;
    }
    std::vector<ninfer::TokenId> continuation = long_prompt();
    continuation.insert(continuation.end(), out.long_run.begin(), out.long_run.end());
    continuation.push_back(198);
    const ninfer::GenerationResult continued =
        engine.generate(engine.prepare_tokens(std::move(continuation)), greedy(8, true));
    out.continued_run = continued.generated_token_ids;
    out.reused_tokens = continued.reused_prompt_tokens;
    return out;
}

// Free memory of the current device, or summed over the distinct devices a split runs its stages
// on: a suspend releases every stage's memory, not only the first device's.
std::size_t free_device_bytes(std::vector<int> devices) {
    std::sort(devices.begin(), devices.end());
    devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
    int current = 0;
    if (cudaGetDevice(&current) != cudaSuccess) { return 0; }
    if (devices.empty()) { devices.push_back(current); }
    std::size_t sum = 0;
    for (const int device : devices) {
        std::size_t free  = 0;
        std::size_t total = 0;
        if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&free, &total) != cudaSuccess) {
            sum = 0;
            break;
        }
        sum += free;
    }
    (void)cudaSetDevice(current);
    return sum;
}

int failures = 0;

std::string describe(const Outputs& got, const Outputs& want) {
    std::string out;
    out += got.short_run == want.short_run ? "short ok" : "short DIFFERS";
    out += got.long_run == want.long_run ? ", long ok" : ", long DIFFERS";
    out += got.continued_run == want.continued_run ? ", continued ok" : ", continued DIFFERS";
    out += ", reuse " + std::to_string(got.reused_tokens) + " vs " +
           std::to_string(want.reused_tokens);
    return out;
}

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "  FAILED: " << label << '\n';
    ++failures;
}

void run_configuration(const char* artifact, const Configuration& configuration,
                       const Outputs& expected) {
    std::cout << "running: " << configuration.label << std::endl;
    ninfer::Engine engine(engine_options(artifact, configuration));
    Outputs got;
    got.short_run =
        engine.generate(engine.prepare_tokens(short_prompt()), greedy(24)).generated_token_ids;
    got.long_run = engine.generate(engine.prepare_tokens(long_prompt()), greedy(24, true))
                       .generated_token_ids;

    // Suspend with automatic resume: the next request brings the model back on its own, and the
    // continuation reuses the long run's state retained before the suspend.
    const std::vector<int> devices = stage_devices(configuration.devices);
    const std::size_t free_resident = free_device_bytes(devices);
    const ninfer::ResidencyStatus suspended = engine.suspend();
    const std::size_t free_suspended = free_device_bytes(devices);
    check(suspended.state == ninfer::ModelResidency::Suspended, "state is suspended");
    check(suspended.host_snapshot_bytes > 0, "the retained state is in host memory");
    check(free_suspended > free_resident + suspended.releasable_device_bytes * 9 / 10,
          "the device memory came back");
    std::cout << "  suspend: released " << (free_suspended - free_resident) / (1 << 20)
              << " MiB of " << suspended.releasable_device_bytes / (1 << 20)
              << " MiB, snapshot " << suspended.host_snapshot_bytes / (1 << 20) << " MiB in "
              << suspended.last_suspend->seconds << " s (state "
              << suspended.last_suspend->state_seconds << " s)\n";
    std::vector<ninfer::TokenId> continuation = long_prompt();
    continuation.insert(continuation.end(), got.long_run.begin(), got.long_run.end());
    continuation.push_back(198);
    const ninfer::GenerationResult continued =
        engine.generate(engine.prepare_tokens(std::move(continuation)), greedy(8, true));
    got.continued_run = continued.generated_token_ids;
    got.reused_tokens = continued.reused_prompt_tokens;
    const ninfer::ResidencyStatus resumed = engine.residency();
    check(resumed.state == ninfer::ModelResidency::Resident && resumed.resume_count == 1,
          "a request resumed the model");
    check(got == expected, configuration.label + ": identical across an automatic resume (" +
                               describe(got, expected) + ")");
    std::cout << "  resume: " << resumed.last_resume->seconds << " s, weights "
              << resumed.last_resume->weight_bytes / (1 << 20) << " MiB in "
              << resumed.last_resume->weight_seconds << " s, state "
              << resumed.last_resume->state_bytes / (1 << 20) << " MiB in "
              << resumed.last_resume->state_seconds << " s; prefix reuse " << got.reused_tokens
              << " tokens\n";

    // Held suspended: requests are refused until an explicit resume.
    (void)engine.suspend(false);
    bool refused = false;
    try {
        (void)engine.generate(engine.prepare_tokens(short_prompt()), greedy(4));
    } catch (const ninfer::RequestError& error) {
        refused = error.kind() == ninfer::RequestErrorKind::Unavailable;
    }
    check(refused, "a model held suspended refuses requests");
    check(!engine.is_available(), "a model held suspended reports itself unavailable");
    (void)engine.resume();
    check(engine.is_available(), "available again after the resume");
    const auto short_again =
        engine.generate(engine.prepare_tokens(short_prompt()), greedy(24)).generated_token_ids;
    check(short_again == expected.short_run,
          configuration.label + ": identical after an explicit resume");
    check(engine.residency().resume_count == 2, "two resumes counted");
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const auto reference_of = [&](Configuration configuration) {
        configuration.suspend = false;
        ninfer::Engine engine(engine_options(artifact, configuration));
        return run_requests(engine, nullptr);
    };
    const Outputs reference = reference_of({.label = "reference"});
    if (reference.short_run.size() != 24 || reference.long_run.size() != 24 ||
        reference.continued_run.size() != 8 || reference.reused_tokens == 0) {
        std::cerr << "the reference did not generate its tokens or reuse its prefix\n";
        return 1;
    }
    run_configuration(artifact, {.label = "single device"}, reference);
    run_configuration(artifact, {.label = "pinned snapshot", .pinned = true}, reference);
    run_configuration(artifact, {.label = "host weight copy", .host_weights = true}, reference);
    run_configuration(artifact, {.label = "two stages", .devices = {0, 0}},
                      reference_of({.label = "two stages", .devices = {0, 0}}));
    try {
        const Outputs hybrid = reference_of({.label = "hybrid", .hybrid_cache = true});
        run_configuration(artifact, {.label = "hybrid prefix cache", .hybrid_cache = true}, hybrid);
    } catch (const std::exception& error) {
        std::cout << "hybrid rows skipped: " << error.what() << '\n';
    }
    std::optional<Outputs> mtp;
    try {
        mtp = reference_of({.label = "mtp", .mtp = true});
    } catch (const std::exception& error) {
        std::cout << "MTP rows skipped: " << error.what() << '\n';
    }
    if (mtp) { run_configuration(artifact, {.label = "MTP", .mtp = true}, *mtp); }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
