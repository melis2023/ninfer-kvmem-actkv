// Public Engine parity across Gemma's device, pinned host and disk expert delivery.
// Requires NINFER_GEMMA4_ARTIFACT and NINFER_GEMMA4_TOKENS (JSON token IDs with BOS).
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace ninfer;
namespace {
struct Scenario {
    ExpertResidency residency;
    std::optional<std::uint64_t> cache; // automatic: an elastic cache beside growing KV
    bool graphs, mtp, cancel_prefill;
    std::uint32_t concurrency;
    const char* name;
};
struct Result {
    std::vector<std::vector<TokenId>> tokens;
    std::vector<float> scores;
};

std::vector<TokenId> prompt(const std::vector<TokenId>& source, std::size_t n) {
    std::vector<TokenId> out{source.front()};
    for (std::size_t i = 1; i < n; ++i) { out.push_back(source[1 + (i - 1) % (source.size() - 1)]); }
    return out;
}

bool counters_zero(const ExpertResidencyStats& stats) {
    return !stats.routes && !stats.hits && !stats.cpu_routes && !stats.admitted && !stats.transferred_bytes;
}

void monotonic(ExpertResidencyStats& previous, const ExpertResidencyStats& current) {
    if (current.routes < previous.routes || current.hits < previous.hits ||
        current.cpu_routes < previous.cpu_routes || current.admitted < previous.admitted ||
        current.transferred_bytes < previous.transferred_bytes || current.hits > current.routes) {
        throw std::runtime_error("public expert counters are inconsistent or decreased");
    }
    previous = current;
}

struct PrefillObserver final : OutputSink {
    bool incomplete = false;
    std::uint32_t processed = 0, total = 0, cancel_at = 0;
    void start(GenerationStart) override {}
    void progress(PromptProgress p) override {
        processed = p.processed_prompt_tokens;
        total = p.total_prompt_tokens;
        incomplete |= processed > 0 && processed < total;
        if (processed > 0 && processed < total && cancel_at == 0) { cancel_at = processed; }
    }
    void timing(GenerationTimingObservation) override {}
    void publish(OutputDelta) override {}
};

void cancel_prefill(Engine& engine, const std::vector<TokenId>& source,
                    const RequestOptions& request, const char* scenario) {
    const auto reference = engine.generate(engine.prepare_tokens(prompt(source, 31), false), request);
    PrefillObserver observer;
    // A third prompt chunk leaves a cancellation boundary before the prompt completes.
    auto handle = engine.submit(engine.prepare_tokens(prompt(source, 4609), false), request,
                                 OutputConsumerMode::Streaming, {.prompt_progress = true});
    const auto cancelled = handle.wait(&observer, CancellationView([&] { return observer.incomplete; }));
    if (!observer.incomplete || cancelled.finish_reason != FinishReason::Cancelled) {
        throw std::runtime_error("long prefill did not cancel after an incomplete progress observation");
    }
    const auto after = engine.generate(engine.prepare_tokens(prompt(source, 31), false), request);
    if (after.generated_token_ids != reference.generated_token_ids || engine.has_failed()) {
        throw std::runtime_error("Engine did not serve the same request after prefill cancellation");
    }
    std::cout << nlohmann::json{{"kind", "cancellation"}, {"scenario", scenario},
        {"observed_processed", observer.cancel_at}, {"last_processed_observation", observer.processed}, {"prompt_tokens", observer.total},
        {"cancelled_generated_tokens", cancelled.generated_token_ids.size()},
        {"next_request_tokens_equal", true}}.dump() << std::endl;
}

Result run(const Scenario& scenario, const char* artifact, const std::vector<TokenId>& source) {
    Result result;
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 6144;
    options.max_concurrency = scenario.concurrency;
    options.expert_residency = scenario.residency;
    options.expert_cache_bytes = scenario.cache;
    options.use_cuda_graph = scenario.graphs;
    options.context_cache.enabled = false;
    if (scenario.mtp) {
        options.speculative.backend = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    {
        Engine engine(options);
        auto previous = engine.runtime_stats().experts;
        if (!counters_zero(previous)) {
            throw std::runtime_error("public expert counters include startup warmup");
        }
        const auto initial_slots = previous.slots;
        RequestOptions request;
        request.execution.requested_output_tokens = 16;
        request.execution.sampling.temperature = 0.0f;
        request.execution.allow_prefix_reuse = false;
        request.stop.include_model_defaults = false;
        request.output.raw = true;
        std::uint64_t speculative_rounds = 0, drafted_tokens = 0, accepted_tokens = 0, fallback_steps = 0;
        std::size_t completed_requests = 0;
        const auto record_generation = [&](GenerationResult generated, bool single_request) {
            if (generated.generated_token_ids.size() != request.execution.requested_output_tokens ||
                generated.finish_reason != FinishReason::OutputLimit) {
                throw std::runtime_error("generation did not complete its requested output workload");
            }
            if (scenario.mtp) {
                std::cout << nlohmann::json{{"kind", "speculation-observation"},
                    {"scenario", scenario.name}, {"case", completed_requests},
                    {"single_request", single_request}, {"enabled", generated.speculative.enabled},
                    {"backend", static_cast<int>(generated.speculative.backend)},
                    {"rounds", generated.speculative.rounds},
                    {"fallback_steps", generated.speculative.fallback_steps}}.dump() << std::endl;
            }
            // The MTP cohort uses a concurrency where the card permits assistant rounds.
            if (generated.speculative.enabled != scenario.mtp ||
                (scenario.mtp && (generated.speculative.backend != SpeculativeBackend::Mtp ||
                                  generated.speculative.rounds == 0))) {
                throw std::runtime_error("generation did not execute its selected speculative mode");
            }
            speculative_rounds += generated.speculative.rounds;
            drafted_tokens += generated.speculative.drafted_tokens;
            accepted_tokens += generated.speculative.accepted_tokens;
            fallback_steps += generated.speculative.fallback_steps;
            ++completed_requests;
            result.tokens.push_back(std::move(generated.generated_token_ids));
        };
        if (scenario.cancel_prefill) {
            cancel_prefill(engine, source, request, scenario.name);
            monotonic(previous, engine.runtime_stats().experts);
        }
        // The long prompt crosses the 2048-token chunk without intermediate logits. Repeats
        // cover captured decode/verification and changing batched route sets.
        for (int repeat = 0; repeat < 2; ++repeat) {
            record_generation(engine.generate(engine.prepare_tokens(prompt(source, 2305), false), request), true);
            monotonic(previous, engine.runtime_stats().experts);
            std::vector<GenerationHandle> handles;
            for (std::uint32_t row = 0; row < scenario.concurrency; ++row) {
                handles.push_back(engine.submit(engine.prepare_tokens(prompt(source, 17 + row), false), request));
            }
            for (auto& handle : handles) {
                record_generation(handle.wait(), false);
                monotonic(previous, engine.runtime_stats().experts);
            }
        }
        const auto memory = engine.memory_summary();
        const auto load = engine.load_summary();
        const auto stats = engine.runtime_stats();
        monotonic(previous, stats.experts);
        if (stats.decode_row_rounds <= stats.decode_rounds) {
            throw std::runtime_error("queued requests did not execute a multi-row decode batch");
        }
        if (scenario.residency != ExpertResidency::Device && !stats.experts.transferred_bytes) {
            throw std::runtime_error("residency flag did not deliver any expert bytes");
        }
        // A zero cache keeps the layer banks only; between prompts they serve as cache slots, so
        // decode may still hit experts there.
        if (scenario.residency != ExpertResidency::Device && scenario.cache == 0 &&
            memory.expert_cache_bytes) {
            throw std::runtime_error("zero-cache delivery reserved expert cache memory");
        }
        if ((scenario.residency == ExpertResidency::Host) != (load.pinned_weight_bytes != 0)) {
            throw std::runtime_error("pinned weight footprint differs from selected expert placement");
        }
        // Beside an automatic (elastic) cache the full-attention caches back only what the
        // sequences reached; otherwise all of them.
        const bool growing = scenario.residency != ExpertResidency::Device && !scenario.cache;
        if (growing != (memory.sequence.used_bytes < memory.sequence.capacity_bytes)) {
            throw std::runtime_error("the KV cache did not grow with its sequences as configured");
        }
        std::cout << nlohmann::json{{"kind", "metrics"}, {"scenario", scenario.name},
            {"graphs", scenario.graphs}, {"mtp", scenario.mtp},
            {"concurrency", scenario.concurrency},
            {"initial_counters_zero", true}, {"initial_cache_slots", initial_slots},
            {"counters_monotonic", true}, {"weights_bytes", memory.weights.capacity_bytes},
            {"pinned_weight_bytes", load.pinned_weight_bytes},
            {"cache_bytes", memory.expert_cache_bytes}, {"workspace_bytes", memory.workspace.capacity_bytes},
            {"sequence_bytes", memory.sequence.capacity_bytes},
            {"sequence_backed_bytes", memory.sequence.used_bytes},
            {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
            {"available_after_startup_bytes", memory.available_after_startup_bytes},
            {"routes", stats.experts.routes}, {"hits", stats.experts.hits},
            {"admissions", stats.experts.admitted}, {"transferred_bytes", stats.experts.transferred_bytes},
            {"decode_rounds", stats.decode_rounds}, {"decode_rows", stats.decode_row_rounds},
            {"completed_requests", completed_requests}, {"speculative_rounds", speculative_rounds},
            {"drafted_tokens", drafted_tokens}, {"accepted_tokens", accepted_tokens},
            {"fallback_steps", fallback_steps},
            {"prefill_seconds", stats.prefill_seconds_total},
            {"decode_seconds", stats.decode_seconds_total}}.dump() << std::endl;
    }
    options.purpose = EnginePurpose::CausalScoring;
    options.max_concurrency = 1;
    options.speculative = {};
    Engine scorer(options);
    auto score_counters = scorer.runtime_stats().experts;
    if (!counters_zero(score_counters)) {
        throw std::runtime_error("scoring expert counters include startup warmup");
    }
    result.scores = scorer.score_tokens(prompt(source, 257), 1);
    monotonic(score_counters, scorer.runtime_stats().experts);
    if (result.scores.size() != 256) {
        throw std::runtime_error("scoring did not return all 256 next-token log probabilities");
    }
    if (!std::all_of(result.scores.begin(), result.scores.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("scoring produced a non-finite log probability");
    }
    return result;
}
} // namespace

int main(int argc, char** argv) {
    bool mtp_only = false;
    std::uint32_t mtp_concurrency = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--mtp-only") { mtp_only = true; }
        else if (argument == "--mtp-concurrency" && i + 1 < argc &&
                 std::string_view(argv[i + 1]).size() == 1 &&
                 argv[i + 1][0] >= '1' && argv[i + 1][0] <= '8') {
            mtp_concurrency = static_cast<std::uint32_t>(argv[++i][0] - '0');
        } else {
            std::cerr << "usage: " << argv[0] << " [--mtp-only] [--mtp-concurrency 1..8]\n";
            return 2;
        }
    }
    const auto* artifact = std::getenv("NINFER_GEMMA4_ARTIFACT");
    const auto* tokens = std::getenv("NINFER_GEMMA4_TOKENS");
    if (!artifact || !tokens) { std::cout << "SKIP Gemma residency: artifact/tokens unset\n"; return 77; }
    try {
        std::ifstream file(tokens);
        const auto source = nlohmann::json::parse(file).get<std::vector<TokenId>>();
        if (source.size() < 2) { throw std::runtime_error("prompt needs BOS and one text token"); }
        const std::vector<Scenario> scenarios = {
            {ExpertResidency::Device, 0, true, false, true, 8, "device-graph"},
            {ExpertResidency::Device, 0, false, false, false, 8, "device-eager"},
            {ExpertResidency::Host, 0, false, false, true, 8, "host-zero-eager"},
            {ExpertResidency::Host, 1536ULL << 20, true, false, false, 8, "host-mixed-graph"},
            {ExpertResidency::Disk, 0, true, false, true, 8, "disk-zero-graph"},
            {ExpertResidency::Disk, 1536ULL << 20, true, false, false, 8, "disk-mixed-graph"},
            {ExpertResidency::Host, std::nullopt, true, false, false, 8, "host-auto-graph"},
            {ExpertResidency::Disk, std::nullopt, true, false, true, 8, "disk-auto-graph"},
            // The host and disk rounds run before any device-resident one: their verification graphs
            // wait on the device for the experts' service, so the Engine must have loaded every
            // round kernel at startup (a lazy load while such a graph waits never finishes).
            {ExpertResidency::Host, 1536ULL << 20, true, true, false, mtp_concurrency, "host-mtp-graph"},
            {ExpertResidency::Disk, 1536ULL << 20, true, true, false, mtp_concurrency, "disk-mtp-graph"},
            {ExpertResidency::Device, 0, true, false, false, mtp_concurrency, "device-mtp-plain-control"},
            {ExpertResidency::Device, 0, true, true, false, mtp_concurrency, "device-mtp-graph"},
        };
        std::map<std::tuple<bool, bool, std::uint32_t>, Result> controls;
        // Host and disk results, compared once their resident controls have run.
        std::vector<std::pair<const Scenario*, Result>> placements;
        int placement_failures = 0;
        for (const auto& scenario : scenarios) {
            if (std::string_view(scenario.name) == "device-mtp-plain-control" && mtp_concurrency == 8) { continue; }
            // Keep a plain control with the same startup concurrency for the MTP comparison.
            if (mtp_only && !scenario.mtp &&
                !(scenario.residency == ExpertResidency::Device && scenario.graphs &&
                  scenario.concurrency == mtp_concurrency)) { continue; }
            auto result = run(scenario, artifact, source);
            const auto key = std::tuple{scenario.graphs, scenario.mtp, scenario.concurrency};
            if (scenario.residency == ExpertResidency::Device) {
                if (scenario.mtp) {
                    const auto& plain = controls.at({scenario.graphs, false, scenario.concurrency});
                    auto differences = nlohmann::json::array();
                    for (std::size_t i = 0; i < result.tokens.size(); ++i) {
                        if (result.tokens[i] != plain.tokens[i]) {
                            differences.push_back({{"case", i}, {"plain", plain.tokens[i]}, {"mtp", result.tokens[i]}});
                        }
                    }
                    // This reports speculative arithmetic separately from placement parity.
                    std::cout << nlohmann::json{{"kind", "resident-control"},
                        {"comparison", "plain-vs-mtp"}, {"different_cases", differences.size()},
                        {"differences", differences},
                        {"total_cases", result.tokens.size()},
                        {"scores_equal", result.scores == plain.scores}}.dump() << std::endl;
                }
                controls.emplace(key, std::move(result));
                continue;
            }
            placements.emplace_back(&scenario, std::move(result));
        }
        for (const auto& [placed, result] : placements) {
            const Scenario& scenario = *placed;
            const auto& baseline =
                controls.at(std::tuple{scenario.graphs, scenario.mtp, scenario.concurrency});
            const bool tokens_equal = result.tokens == baseline.tokens;
            const bool scores_equal = result.scores == baseline.scores;
            auto token_differences = nlohmann::json::array();
            for (std::size_t i = 0; i < result.tokens.size(); ++i) {
                if (result.tokens[i] != baseline.tokens[i]) {
                    token_differences.push_back({{"case", i}, {"resident", baseline.tokens[i]},
                                                  {"placement", result.tokens[i]}});
                }
            }
            std::size_t score_differences = 0;
            double score_max_abs = 0;
            if (result.scores.size() != baseline.scores.size()) {
                throw std::runtime_error("score result length differs from its matched resident control");
            }
            for (std::size_t i = 0; i < result.scores.size(); ++i) {
                score_differences += result.scores[i] != baseline.scores[i];
                score_max_abs = std::max(score_max_abs, std::abs(double(result.scores[i]) - baseline.scores[i]));
            }
            std::cout << nlohmann::json{{"kind", "placement-parity"}, {"scenario", scenario.name},
                {"graphs", scenario.graphs}, {"mtp", scenario.mtp},
                {"concurrency", scenario.concurrency},
                {"tokens_equal", tokens_equal}, {"scores_equal", scores_equal},
                {"token_differences", token_differences}, {"score_differences", score_differences},
                {"score_max_abs", score_max_abs}}.dump() << std::endl;
            placement_failures += !tokens_equal || !scores_equal;
        }
        return placement_failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
