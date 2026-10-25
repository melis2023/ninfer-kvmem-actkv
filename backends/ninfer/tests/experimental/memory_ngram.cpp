#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void add(ninfer::PromptInput& input, ninfer::ChatRole role, const std::string& text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = text});
    input.messages.push_back(std::move(message));
}
struct Sink final : ninfer::OutputSink {
    std::uint32_t stop_after = 0;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (stop_after && value.generated_tokens >= stop_after) cancelled = true;
    }
};
ninfer::OwnedMedia media(const char* path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "image fixture missing");
    ninfer::OwnedMedia result;
    result.bytes.assign(std::istreambuf_iterator<char>(file), {});
    result.media_type = "image/png";
    result.source_name = "image.png";
    return result;
}
std::string prefix(int seed) {
    return "def transform_" + std::to_string(seed) + "(value):\n    offset = " + std::to_string(seed + 17);
}
std::string source(int seed) {
    std::string result;
    for (int index = 0; index < 20; ++index) {
        const int value = seed + index;
        result += prefix(value) + "\n    return value * " + std::to_string(value + 2) + " + offset\n\n";
    }
    return result;
}
void concurrent(ninfer::Engine& engine, const ninfer::EngineOptions& options,
                const ninfer::OwnedMedia& red, const ninfer::OwnedMedia& blue) {
    const auto lanes = options.max_concurrency;
    std::vector<ninfer::PromptInput> prompts;
    std::vector<int> seeds;
    for (unsigned row = 0; row < lanes; ++row) {
        const int seed = 100 + 100 * row;
        seeds.push_back(seed);
        ninfer::PromptInput input;
        input.context_cache.session_key = "ngram-lane-" + std::to_string(row);
        input.options.enable_thinking = false;
        add(input, ninfer::ChatRole::System, "Copy complete files exactly when asked, preserving every function and every newline.");
        for (int entry = 0; entry < 8; ++entry) {
            std::string filler;
            for (int repeat = 0; repeat < 8; ++repeat)
                filler += "The observatory records pulsars, galaxies and meteor showers. ";
            add(input, ninfer::ChatRole::User, filler);
            add(input, ninfer::ChatRole::Assistant, "Recorded.");
        }
        add(input, ninfer::ChatRole::User, row == lanes - 1
            ? "Explain how hash tables resolve collisions in detail."
            : "Repeat the following Python file exactly. Output only the file, with no explanation or Markdown fences. "
              "Preserve every space and newline.\n\n" + source(seed));
        if (options.enable_vision && row % 3 != 2) {
            input.messages.back().parts.push_back({.kind = ninfer::MessagePartKind::Media,
                .media = row % 3 == 0 ? red : blue});
        }
        if (row != lanes - 1) {
            add(input, ninfer::ChatRole::Assistant, prefix(seed));
            input.options.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
        }
        prompts.push_back(std::move(input));
    }
    const auto batch = [&](const char* name, int cancel_row, bool restored, bool full_overlap) {
        std::vector<ninfer::PreparedPrompt> prepared;
        for (const auto& prompt : prompts) prepared.push_back(engine.prepare(prompt));
        std::atomic<unsigned> started{0};
        const unsigned all_started = (1U << lanes) - 1U;
        struct LaneSink final : ninfer::OutputSink {
            std::atomic<unsigned>& started;
            unsigned bit, all;
            bool cancel;
            std::atomic<bool> cancelled{false};
            LaneSink(std::atomic<unsigned>& value, unsigned row, unsigned mask, bool enabled)
                : started(value), bit(1U << row), all(mask), cancel(enabled) {}
            void start(ninfer::GenerationStart) override {}
            void progress(ninfer::PromptProgress) override {}
            void publish(ninfer::OutputDelta) override {}
            void timing(ninfer::GenerationTimingObservation value) override {
                if (value.generated_tokens) started.fetch_or(bit);
                if (cancel && value.generated_tokens >= 16 && (started.load() & all) == all)
                    cancelled = true;
            }
        };
        const auto before = engine.runtime_stats();
        std::vector<std::future<ninfer::GenerationResult>> futures;
        for (unsigned row = 0; row < lanes; ++row) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 256;
            request.execution.sampling.temperature = 0.0F;
            request.stop.include_model_defaults = true;
            auto handle = engine.submit(std::move(prepared[row]), request,
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
            futures.push_back(std::async(std::launch::async,
                [handle = std::move(handle), &started, row, all_started, cancel_row]() mutable {
                    LaneSink sink(started, row, all_started, static_cast<int>(row) == cancel_row);
                    return handle.wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
                }));
        }
        unsigned peak = 0;
        while (std::any_of(futures.begin(), futures.end(), [](auto& future) {
            return future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
        })) {
            const auto stats = engine.runtime_stats();
            peak = std::max(peak, stats.running_requests);
            require(stats.kvmem.host_payload_bytes <= options.kvmem.host_bytes &&
                    stats.kvmem.active_payload_reservation_bytes <= options.kvmem.host_bytes,
                    "concurrent ngram exceeded global H");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::uint64_t accepted = 0;
        std::vector<ninfer::GenerationResult> results;
        for (unsigned row = 0; row < lanes; ++row) {
            auto result = futures[row].get();
            const bool cancelled = static_cast<int>(row) == cancel_row;
            require((result.finish_reason == ninfer::FinishReason::Cancelled) == cancelled,
                    "ngram cancellation crossed lanes");
            require((result.reused_prompt_tokens > 0) == restored,
                    "concurrent ngram history reuse changed");
            require(result.prompt.has_media == (options.enable_vision && row % 3 != 2),
                    "concurrent ngram media crossed lanes");
            if (row != lanes - 1) {
                const auto produced = prefix(seeds[row]) + result.content;
                if (!source(seeds[row]).starts_with(produced))
                    throw std::runtime_error(std::string(name) + " copy lane " + std::to_string(row) +
                                             " diverged: " + produced.substr(0, 256));
                if (!cancelled) require(result.speculative.ngram_accepted_tokens > 0,
                                        "concurrent copy lane did not engage ngram");
                accepted += result.speculative.ngram_accepted_tokens;
            } else {
                const std::set<ninfer::TokenId> distinct(result.generated_token_ids.begin(), result.generated_token_ids.end());
                require(result.generated_token_ids.size() >= 32 && distinct.size() >= 16,
                        "mixed neural lane output degenerated");
            }
            if (options.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive)
                require(result.speculative.adaptive, "concurrent adaptive MTP silently disabled");
            std::cout << "MEMORY_NGRAM_LANE case=" << name << " row=" << row
                      << " reused=" << result.reused_prompt_tokens << " generated=" << result.generated_token_ids.size()
                      << " accepted=" << result.speculative.ngram_accepted_tokens << std::endl;
            results.push_back(std::move(result));
        }
        const auto after = engine.runtime_stats();
        const auto rounds = after.decode_rounds - before.decode_rounds;
        const auto rows = after.decode_row_rounds - before.decode_row_rounds;
        std::cout << "MEMORY_NGRAM_BATCH case=" << name << " peak=" << peak << " rounds=" << rounds
                  << " rows=" << rows << " accepted=" << accepted << std::endl;
        // Cold prefills are serialized. A fast copy lane can finish before the
        // next CPU-image lane reaches decode. Require actual full overlap in
        // each restored phase, independent of this cold-admission timing.
        require(rows >= rounds && peak >= 1 && (!full_overlap || (rows > rounds && peak == lanes)),
                "ngram did not exercise compact concurrent decode");
        const auto memory = engine.memory_summary().kvmem;
        require(engine.is_available() && memory.resident_pages == 0 && memory.mtp_resident_pages == 0 &&
                memory.active_payload_reservation_bytes == 0 && memory.host_payload_bytes <= options.kvmem.host_bytes,
                "concurrent ngram leaked resources");
        return results;
    };
    if (!std::getenv("NINFER_NGRAM_TEST_COLORS_ONLY")) {
    const auto cold = batch("cold-mixed", -1, false, false);
    const auto warm = batch("restored-mixed", -1, true, true);
    for (unsigned row = 0; row + 1 < lanes; ++row)
        require(cold[row].generated_token_ids == warm[row].generated_token_ids,
                "concurrent restore changed exact copy tokens");
    batch("cancel-one", 0, true, true);
    const auto recovered = batch("recover-peers", -1, true, true);
    for (unsigned row = 0; row + 1 < lanes; ++row)
        require(warm[row].generated_token_ids == recovered[row].generated_token_ids,
                "cancelled concurrent ngram polluted copy history");
    }
    if (options.enable_vision) {
        std::vector<ninfer::PreparedPrompt> prepared;
        for (unsigned row = 0; row < lanes; ++row) {
            ninfer::PromptInput input;
            input.options.enable_thinking = false;
            input.context_cache.session_key = "ngram-color-" + std::to_string(row);
            add(input, ninfer::ChatRole::System, "Answer questions about the supplied images in English.");
            add(input, ninfer::ChatRole::User, "Answer with exactly one word: what color fills this image?");
            input.messages.back().parts.push_back({.kind = ninfer::MessagePartKind::Media,
                .media = row % 2 == 0 ? red : blue});
            prepared.push_back(engine.prepare(input));
        }
        std::vector<ninfer::GenerationHandle> handles;
        for (auto& input : prepared) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 32;
            request.execution.sampling.temperature = 0.0F;
            handles.push_back(engine.submit(std::move(input), request));
        }
        for (unsigned row = 0; row < lanes; ++row) {
            auto result = handles[row].wait();
            std::transform(result.content.begin(), result.content.end(), result.content.begin(),
                [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
            if (result.content.find(row % 2 == 0 ? "red" : "blue") == std::string::npos)
                throw std::runtime_error("image color lane " + std::to_string(row) +
                    " expected " + (row % 2 == 0 ? "red" : "blue") + ": " + result.content);
            std::cout << "MEMORY_NGRAM_COLOR row=" << row << " correct=1" << std::endl;
        }
        const auto memory = engine.memory_summary().kvmem;
        require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0 &&
                memory.active_payload_reservation_bytes == 0, "ngram color requests leaked resources");
    }
    std::cout << (std::getenv("NINFER_NGRAM_TEST_COLORS_ONLY")
        ? "MEMORY_NGRAM_COLORS_ONLY PASS" : "MEMORY_NGRAM_CONCURRENT PASS") << std::endl;
}
}
int main(int argc, char** argv) {
    try {
        require((argc >= 5 && argc <= 8) || argc == 11,
                "artifact bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8 eager|graph ngram-width [mtp-drafts] [fixed|adaptive] [concurrency] [red.png blue.png resident|cpu]");
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off"; options.max_context = 8192; options.prefill_chunk = 128;
        options.context_cache.enabled = false;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 2, .verify_transfers = true};
        options.kv_cache = ninfer::test::kvmem_test_storage(argv[2]);
        options.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[3]);
        if (const auto* fast = std::getenv("NINFER_TEST_FAST_PREFILL")) {
            require(std::string(fast) == "0" || std::string(fast) == "1", "NINFER_TEST_FAST_PREFILL must be 0 or 1");
            options.fast_prefill_kernel = fast[0] == '1';
        }
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = argc >= 6 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 3;
        options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[4]));
        options.speculative.ngram_min_match = 4;
        if (argc >= 7) {
            require(std::string(argv[6]) == "adaptive" || std::string(argv[6]) == "fixed", "expected fixed|adaptive");
            if (std::string(argv[6]) == "adaptive") options.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
        }
        if (argc >= 8) {
            options.max_concurrency = static_cast<std::uint32_t>(std::stoul(argv[7]));
            require(options.max_concurrency >= 2 && options.max_concurrency <= 8, "expected concurrency 2..8");
            options.kvmem.host_bytes = 512ULL * options.max_concurrency << 20;
            options.kvmem.retained_sessions = options.max_concurrency;
            // Preserve the complete ~600-token copy query while still selecting its longer history.
            options.kvmem.selected_tokens = 768;
            options.pending_timeout_ms = 600000;
        }
        ninfer::OwnedMedia red, blue;
        if (argc == 11) {
            options.enable_vision = true;
            options.kvmem.selected_tokens = 1536;
            options.vision_max_merged_tokens = 1024;
            const std::string residency = argv[10];
            require(residency == "resident" || residency == "cpu", "expected resident|cpu");
            options.vision_residency = residency == "cpu" ? ninfer::VisionResidency::Cpu : ninfer::VisionResidency::Resident;
            red = media(argv[8]); blue = media(argv[9]);
        }
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.kvmem.device_tokens() * options.max_concurrency);
        std::cout << "MEMORY_FEATURE fast_prefill=" << options.fast_prefill_kernel << std::endl;
        ninfer::Engine engine(options);
        if (options.max_concurrency > 1) {
            concurrent(engine, options, red, blue);
            return 0;
        }
        ninfer::PromptInput prompt;
        prompt.options.enable_thinking = false;
        prompt.context_cache.session_key = "copy-history";
        add(prompt, ninfer::ChatRole::System, "When asked to copy text, reproduce it verbatim, without explanations or quotation marks.");
        for (int i = 0; i < 8; ++i) {
            std::string filler;
            for (int j = 0; j < 8; ++j) filler += "The observatory records pulsars, galaxies and meteor showers. ";
            add(prompt, ninfer::ChatRole::User, filler);
            add(prompt, ninfer::ChatRole::Assistant, "Recorded.");
        }
        std::string passage = "The violet observatory opens at sunrise. ";
        for (int i = 0; i < 12; ++i)
            passage += "Its telescope observes distant galaxies, while the astronomers record the changing light of stars. ";
        add(prompt, ninfer::ChatRole::User, "Store this passage for exact copying later:\n" + passage);
        add(prompt, ninfer::ChatRole::Assistant, "The passage is stored.");
        add(prompt, ninfer::ChatRole::User, "Copy the stored passage exactly. Start with: The violet observatory opens at sunrise.");
        const auto run = [&](const char* name, const ninfer::PromptInput& input, bool hit,
                             std::uint32_t count = 192, std::uint32_t cancel = 0, float temperature = 0.0F) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = count;
            request.execution.sampling.temperature = temperature;
            request.execution.sampling.top_k = 20; request.execution.sampling.top_p = 0.9F;
            request.execution.sampling.seed = 42;
            request.execution.allow_prefix_reuse = true;
            request.stop.include_model_defaults = false;
            Sink sink; sink.stop_after = cancel;
            const auto result = engine.submit(engine.prepare(input), request,
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
                .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
            const auto memory = engine.memory_summary().kvmem;
            require(engine.is_available(), "ngram made Engine unavailable");
            require((result.reused_prompt_tokens > 0) == hit, "ngram history reuse decision changed");
            require(memory.host_payload_bytes <= options.kvmem.host_bytes &&
                memory.resident_pages == 0 && memory.mtp_resident_pages == 0, "ngram leaked resources");
            require(cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                           : result.generated_token_ids.size() == count, "ngram output termination mismatch");
            std::cout << "MEMORY_NGRAM case=" << name << " reused=" << result.reused_prompt_tokens
                      << " generated=" << result.generated_token_ids.size()
                      << " rounds=" << result.speculative.ngram_rounds
                      << " drafted=" << result.speculative.ngram_drafted_tokens
                      << " accepted=" << result.speculative.ngram_accepted_tokens << std::endl;
            if (options.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive) {
                require(result.speculative.adaptive, "adaptive MTP was silently disabled");
                std::cout << "MEMORY_ADAPTIVE case=" << name
                          << " transitions=" << result.speculative.window_transitions << " windows=";
                for (auto count : result.speculative.rounds_per_window) std::cout << count << ',';
                std::cout << std::endl;
            }
            return result;
        };
        const auto cold = run("cold-copy", prompt, false);
        require(cold.content.find("violet observatory") != std::string::npos, "copy instruction failed");
        require(cold.speculative.ngram_rounds > 0 &&
                cold.speculative.ngram_accepted_tokens > 3 * cold.speculative.ngram_rounds,
                "wide ngram path did not accept more than neural draft width");
        const auto repeat = run("repeat-copy", prompt, true);
        require(cold.generated_token_ids == repeat.generated_token_ids, "restored ngram changed greedy output");
        run("cancel-copy", prompt, true, 192, 16);
        const auto recovered = run("recover-copy", prompt, true);
        require(cold.generated_token_ids == recovered.generated_token_ids, "cancelled ngram polluted history");
        for (auto count : {1U, 2U, 3U, 17U, 65U}) {
            const auto truncated = run("output-limit", prompt, true, count);
            require(truncated.generated_token_ids.front() == cold.generated_token_ids.front(),
                    "output truncation changed prompt's first token");
        }
        const auto sampled = run("sampling", prompt, true, 96, 0, 0.8F);
        const auto sampled_again = run("sampling-repeat", prompt, true, 96, 0, 0.8F);
        require(sampled.generated_token_ids == sampled_again.generated_token_ids,
                "ngram sampling did not restore logical RNG positions");
        auto edited = prompt;
        edited.messages[1].parts[0].text = "This archive was edited. The new observatory is amber.";
        run("edited-prefix", edited, false, 32);
        auto code = edited;
        code.messages.back().parts[0].text = "Copy the following Python code exactly, without markdown:\n";
        for (int i = 0; i < 12; ++i)
            code.messages.back().parts[0].text += "def square_" + std::to_string(i) + "(value):\n    return value * value\n\n";
        const auto copied_code = run("copy-code", code, true, 192);
        require(copied_code.speculative.ngram_accepted_tokens > 0, "code did not exercise ngram");
        std::cout << "MEMORY_NGRAM PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl; return 1;
    }
}
