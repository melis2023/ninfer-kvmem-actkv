#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <algorithm>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Sink final : ninfer::OutputSink {
    ninfer::Engine& engine;
    std::uint64_t reservation = 0, payload = 0;
    std::uint32_t running = 0;
    explicit Sink(ninfer::Engine& value) : engine(value) {}
    void observe() {
        const auto stats = engine.runtime_stats();
        reservation = std::max(reservation, stats.kvmem.active_payload_reservation_bytes);
        payload = std::max(payload, stats.kvmem.host_payload_bytes);
        running = std::max(running, stats.running_requests);
        require(stats.kvmem.host_payload_bytes <= engine.options().kvmem.host_bytes,
                "RAM payload exceeded H");
        require(stats.kvmem.active_payload_reservation_bytes <= engine.options().kvmem.host_bytes,
                "RAM admission exceeded H");
    }
    void start(ninfer::GenerationStart) override { observe(); }
    void progress(ninfer::PromptProgress) override { observe(); }
    void timing(ninfer::GenerationTimingObservation) override { observe(); }
    void publish(ninfer::OutputDelta) override {}
};
struct Observed {
    ninfer::GenerationResult result;
    std::uint64_t reservation, payload;
    std::uint32_t running;
};
std::vector<Observed> batch(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& corpus,
                           unsigned frontier, unsigned output, unsigned lanes) {
    std::vector<std::future<Observed>> pending;
    for (unsigned lane = 0; lane < lanes; ++lane) {
        // The accepted KV frontier excludes the final emitted token.
        std::vector<ninfer::TokenId> tokens(corpus.begin(), corpus.begin() + frontier - output + 1);
        ninfer::RequestOptions request;
        request.execution.allow_prefix_reuse = false;
        request.execution.requested_output_tokens = output;
        request.execution.sampling.temperature = 0;
        request.stop.include_model_defaults = false;
        auto handle = engine.submit(engine.prepare_tokens(std::move(tokens), false), request,
            ninfer::OutputConsumerMode::Streaming, {.live_timings = true, .prompt_progress = true});
        pending.push_back(std::async(std::launch::async, [handle = std::move(handle), &engine]() mutable {
            Sink sink(engine);
            auto result = handle.wait(&sink);
            return Observed{std::move(result), sink.reservation, sink.payload, sink.running};
        }));
    }
    std::vector<Observed> results;
    for (auto& future : pending) {
        auto value = future.get();
        require(value.result.generated_token_ids.size() == output, "output limit was not reached");
        require(value.result.finish_reason == ninfer::FinishReason::OutputLimit, "wrong termination");
        require(value.result.reused_prompt_tokens == 0, "cold request reused history");
        results.push_back(std::move(value));
    }
    require(engine.is_available() && !engine.has_failed(), "RAM boundary failed the Engine");
    const auto memory = engine.memory_summary().kvmem;
    require(memory.host_payload_bytes == 0 && memory.active_payload_reservation_bytes == 0 &&
            memory.resident_pages == 0 && memory.mtp_resident_pages == 0,
            "completed batch leaked KV or reservations");
    if (lanes > 1)
        require(std::any_of(results.begin(), results.end(), [lanes](const auto& r) {
            return r.running == lanes;
        }), "budget test did not overlap active requests");
    return results;
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 6, "artifact drafts(0..15) kv-format eager|graph concurrency(1|2)");
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 4096;
        options.prefill_chunk = 128;
        options.max_concurrency = static_cast<unsigned>(std::stoul(argv[5]));
        require(options.max_concurrency == 1 || options.max_concurrency == 2, "invalid concurrency");
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512 * options.max_concurrency);
        options.kv_cache = ninfer::test::kvmem_test_storage(argv[3]);
        options.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[4]);
        options.context_cache.enabled = false;
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 1ULL << 30, .retained_sessions = 1, .verify_transfers = true};
        const auto drafts = static_cast<unsigned>(std::stoul(argv[2]));
        require(drafts <= 15, "invalid draft count");
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        struct Case { unsigned frontier, output; };
        // A one-token request finishes in prefill before another lane can be
        // admitted; the concurrent cases exercise decode across the page boundary.
        // MTP can split the last prompt chunk to leave proposal space in R.
        // Long outputs must still finish within the admitted service projection.
        const std::vector<Case> cases = options.max_concurrency == 1
            ? std::vector<Case>{{2047, 2}, {2048, 1}, {2048, 2}, {2048, 5}, {2049, 2},
                                {2051, 5}, {2052, 5}, {2047, 129}}
            : std::vector<Case>{{2047, 8}, {2048, 8}, {2049, 8},
                                {2047, 129}, {2048, 129}, {2048, 257}, {2049, 129}};
        std::vector<ninfer::TokenId> corpus;
        std::vector<std::vector<Observed>> reference;
        std::uint64_t page_bytes = 0;
        {
            ninfer::Engine roomy(options);
            std::string text;
            for (unsigned i = 0; i < 1024; ++i)
                text += "The observatory archives pulsars, cosmic rays and distant galaxies. ";
            corpus = roomy.tokenize_text(text);
            require(corpus.size() >= 2052, "prompt corpus is too short");
            // Observe one admitted request to size H for this artifact's native
            // main+draft KV representation, independently of the allocation path.
            const auto sizing = batch(roomy, corpus, 2048, 1, 1);
            require(sizing[0].reservation && sizing[0].reservation % 32 == 0,
                    "cannot size the native 64-token Host pages");
            page_bytes = sizing[0].reservation / 32;
            for (const auto& c : cases)
                reference.push_back(batch(roomy, corpus, c.frontier, c.output, options.max_concurrency));
        }
        // Use the exact admitted capacity, then repeat requests on the same Engine
        // so a boundary failure or leaked reservation cannot hide at destruction.
        for (unsigned pages : {32U, 33U}) {
            options.kvmem.host_bytes = page_bytes * pages * options.max_concurrency;
            ninfer::Engine tight(options);
            for (std::size_t i = 0; i < cases.size(); ++i) {
                const auto& c = cases[i];
                if ((c.frontier + 63U) / 64U != pages) continue;
                const auto actual = batch(tight, corpus, c.frontier, c.output, options.max_concurrency);
                for (std::size_t lane = 0; lane < actual.size(); ++lane)
                    require(actual[lane].result.generated_token_ids == reference[i][lane].result.generated_token_ids,
                            "tight Host budget changed greedy output");
                require(std::any_of(actual.begin(), actual.end(), [&](const auto& r) {
                    return r.reservation == options.kvmem.host_bytes;
                }), "test did not consume the entire Host reservation budget");
                std::cout << "RAM_BUDGET frontier=" << c.frontier << " output=" << c.output
                          << " lanes=" << options.max_concurrency << " H=" << options.kvmem.host_bytes
                          << " exact_output=1 resources_released=1" << std::endl;
            }
        }
        std::cout << "MEMORY_RAM_BUDGET PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
