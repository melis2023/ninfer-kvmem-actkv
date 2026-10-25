#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <fstream>
#include <future>
#include <filesystem>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ninfer::OwnedMedia media(const char* path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "image fixture is missing");
    ninfer::OwnedMedia image;
    image.bytes.assign(std::istreambuf_iterator<char>(file), {});
    image.media_type = "image/png";
    image.source_name = "image.png";
    return image;
}
ninfer::PromptInput history(const std::string& key, const std::string& code,
                           const ninfer::OwnedMedia* image = nullptr) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.context_cache.session_key = key;
    const auto add = [&](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
        input.messages.push_back(std::move(message));
    };
    add(ninfer::ChatRole::System, "Read archive " + key + ". Its access code is " + code + ".");
    if (image) {
        add(ninfer::ChatRole::User, "Keep this archive image for later questions.");
        input.messages.back().parts.push_back({.kind = ninfer::MessagePartKind::Media, .media = *image});
        add(ninfer::ChatRole::Assistant, "Recorded.");
    }
    for (int entry = 0; entry < 16; ++entry) {
        std::string text = "Entry " + std::to_string(entry) + ". ";
        for (int repeat = 0; repeat < 8; ++repeat)
            text += "The observatory files reports about pulsars, cosmic rays and distant galaxies. ";
        add(ninfer::ChatRole::User, text);
        add(ninfer::ChatRole::Assistant, "Recorded.");
    }
    add(ninfer::ChatRole::User, image
        ? "Start with the archive access code and the color of the supplied image in English, then explain what the observatory studies."
        : "Start with the archive access code, then explain what the observatory studies.");
    if (image) input.messages.back().parts.push_back({.kind = ninfer::MessagePartKind::Media, .media = *image});
    return input;
}
struct Sink final : ninfer::OutputSink {
    bool cancel_enabled = false;
    std::atomic<std::uint32_t>* started = nullptr;
    std::uint32_t row_bit = 0;
    std::uint32_t cancel_after_started = 0;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (value.generated_tokens && started) started->fetch_or(row_bit);
        if (cancel_enabled && value.generated_tokens >= 12 &&
            (!started || (started->load() & cancel_after_started) == cancel_after_started))
            cancelled.store(true);
    }
};
struct Work {
    ninfer::PromptInput prompt;
    std::uint32_t output = 96;
    bool cancel = false;
    std::string expected_code;
    std::string expected_color;
};
}

int main(int argc, char** argv) {
    try {
        require((argc >= 5 && argc <= 9) || argc == 12 || argc == 13,
                "artifact draft-count(0..15) concurrency(2..8) host-MiB [bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8] [eager|graph] [fixed|fast|adaptive|adaptive-fast] [disk-MiB | red.png blue.png resident|cpu vision-tokens [selected-tokens]]");
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        const auto concurrency = static_cast<std::uint32_t>(std::stoul(argv[3]));
        require(concurrency >= 2 && concurrency <= 8, "concurrency must be in [2,8]");
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 8192;
        options.max_concurrency = concurrency;
        // This correctness run synchronously verifies every transferred KV/state
        // byte; queued requests must survive that diagnostic cost.
        options.pending_timeout_ms = 600000;
        options.prefill_chunk = 128;
        options.kv_cache = argc >= 6 ? ninfer::test::kvmem_test_storage(argv[5])
                                    : ninfer::KvCacheStorage::Int8Group64;
        if (argc >= 7) options.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[6]);
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512 * concurrency);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = std::stoull(argv[4]) << 20, .retained_sessions = concurrency,
            .verify_transfers = true};
        options.context_cache.enabled = false;
        if (argc == 9) {
            options.kvmem.disk_bytes = std::stoull(argv[8]) << 20;
            options.kvmem.disk_path = std::filesystem::temp_directory_path() /
                ("ninfer-concurrent-ssd-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        }
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        if (argc >= 8) {
            const std::string feature = argv[7];
            require(feature == "fixed" || feature == "fast" || feature == "adaptive" || feature == "adaptive-fast", "invalid feature");
            options.fast_prefill_kernel = feature == "fast" || feature == "adaptive-fast";
            if (feature == "adaptive" || feature == "adaptive-fast")
                options.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
        }
        ninfer::OwnedMedia red, blue;
        if (argc >= 12) {
            options.enable_vision = true;
            options.kvmem.selected_tokens = argc == 13 ? static_cast<std::uint32_t>(std::stoul(argv[12])) : 1536;
            options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
                options.kvmem.device_tokens() * concurrency);
            const std::string residency = argv[10];
            require(residency == "resident" || residency == "cpu", "invalid Vision residency");
            options.vision_residency = residency == "cpu" ? ninfer::VisionResidency::Cpu
                                                         : ninfer::VisionResidency::Resident;
            options.vision_max_merged_tokens = static_cast<std::uint32_t>(std::stoul(argv[11]));
            red = media(argv[8]); blue = media(argv[9]);
        }
        std::cout << "MEMORY_OPTIONS concurrency=" << concurrency << " drafts=" << drafts
                  << " selected=" << options.kvmem.selected_tokens
                  << " reserve=" << options.kvmem.reserve_tokens
                  << " host=" << options.kvmem.host_bytes
                  << " vision_limit=" << options.vision_max_merged_tokens << std::endl;
        auto engine_owner = std::make_unique<ninfer::Engine>(options);
        auto& engine = *engine_owner;
        const bool pressure = !options.kvmem.disk_bytes && options.kvmem.host_bytes <
            ninfer::test::kvmem_test_host_bytes(256ULL << 20, options.kv_cache);
        const auto batch = [&](const char* label, std::vector<Work> jobs, bool expect_batch) {
            std::vector<ninfer::PreparedPrompt> prepared;
            for (auto& job : jobs) {
                prepared.push_back(engine.prepare(job.prompt));
                const auto& prompt = prepared.back();
                std::cout << "MEMORY_PREPARE " << label << " lane=" << prepared.size() - 1
                          << " tokens=" << prompt.summary().prompt_tokens
                          << " vision_tokens=" << prompt.preparation_stats().vision_tokens
                          << " media_items=" << prompt.preparation_stats().media_items << std::endl;
            }
            const auto before = engine.runtime_stats();
            std::vector<std::future<ninfer::GenerationResult>> futures;
            std::atomic<std::uint32_t> started{0};
            const auto cancel_after_started = expect_batch ? (1U << jobs.size()) - 1U : 0U;
            for (std::size_t row = 0; row < jobs.size(); ++row) {
                ninfer::RequestOptions request;
                request.execution.requested_output_tokens = jobs[row].output;
                request.execution.sampling.temperature = 0;
                request.stop.include_model_defaults = false;
                auto handle = engine.submit(std::move(prepared[row]), request,
                    ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
                futures.push_back(std::async(std::launch::async,
                    [handle = std::move(handle), cancel = jobs[row].cancel, row,
                     &started, cancel_after_started]() mutable {
                        Sink sink;
                        sink.cancel_enabled = cancel;
                        sink.started = &started;
                        sink.row_bit = 1U << row;
                        sink.cancel_after_started = cancel_after_started;
                        return handle.wait(&sink, ninfer::CancellationView([&] {
                            return sink.cancelled.load();
                        }));
                    }));
            }
            std::uint32_t peak_running = 0, peak_waiting = 0;
            while (std::any_of(futures.begin(), futures.end(), [](auto& future) {
                return future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
            })) {
                const auto stats = engine.runtime_stats();
                peak_running = std::max(peak_running, stats.running_requests);
                peak_waiting = std::max(peak_waiting, stats.waiting_requests);
                require(stats.kvmem.host_payload_bytes <= options.kvmem.host_bytes,
                        "concurrent archives exceeded global H");
                require(stats.kvmem.active_payload_reservation_bytes <= stats.kvmem.retained_payload_budget_bytes,
                        "admission overcommitted retained payload capacity");
                require(stats.kvmem.disk_payload_bytes <= options.kvmem.disk_bytes,
                        "concurrent archives exceeded global D");
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            std::vector<ninfer::GenerationResult> results;
            for (std::size_t row = 0; row < futures.size(); ++row) {
                auto result = futures[row].get();
                require(jobs[row].cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                                        : result.generated_token_ids.size() == jobs[row].output,
                        "one lane affected another lane's termination");
                if (!jobs[row].cancel && drafts)
                    require(result.speculative.accepted_tokens > 0, "MTP was silently disabled");
                if (options.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive) {
                    require(result.speculative.adaptive, "adaptive MTP was silently disabled");
                    std::cout << "MEMORY_ADAPTIVE " << label << " lane=" << row
                              << " transitions=" << result.speculative.window_transitions << " windows=";
                    for (auto count : result.speculative.rounds_per_window) std::cout << count << ',';
                    std::cout << std::endl;
                }
                if (!jobs[row].cancel && !jobs[row].expected_code.empty())
                    if (result.content.find(jobs[row].expected_code) == std::string::npos)
                        throw std::runtime_error(std::string(label) + " lane=" + std::to_string(row) +
                            " returned the wrong archive access code: " + result.content.substr(0, 256));
                if (!jobs[row].expected_color.empty()) {
                    require(result.prompt.has_media, "concurrent image request lost media");
                    auto content = result.content;
                    std::transform(content.begin(), content.end(), content.begin(),
                        [](unsigned char value) { return std::tolower(value); });
                    if (!jobs[row].cancel)
                        if (content.find(jobs[row].expected_color) == std::string::npos)
                            throw std::runtime_error(std::string(label) + " lane=" + std::to_string(row) +
                                " returned another lane's color: " + result.content.substr(0, 256));
                } else {
                    require(!result.prompt.has_media, "text request acquired another lane's image");
                }
                results.push_back(std::move(result));
            }
            require(engine.is_available(), "concurrent run made Engine unavailable");
            const auto after = engine.runtime_stats();
            const auto rounds = after.decode_rounds - before.decode_rounds;
            const auto rows = after.decode_row_rounds - before.decode_row_rounds;
            std::cout << "MEMORY_BATCH " << label << " rounds=" << rounds << " rows=" << rows
                      << " peak_running=" << peak_running << " peak_waiting=" << peak_waiting << std::endl;
            if (expect_batch) require(rows > rounds && peak_running >= 2,
                                      "requests did not execute in a compact decode batch");
            if (expect_batch && std::string(label) == "cancel-one")
                require(peak_running == concurrency, "configured lanes were not concurrently active");
            if (pressure && jobs.size() > 1)
                require(peak_running == 1 && peak_waiting > 0,
                        "Host pressure did not queue requests until a reservation was released");
            const auto memory = engine.memory_summary().kvmem;
            if (options.kvmem.disk_bytes) {
                std::cout << "MEMORY_SSD " << label << " H=" << memory.host_payload_bytes
                          << " D=" << memory.disk_payload_bytes << " reads=" << memory.disk_read_bytes
                          << " writes=" << memory.disk_written_bytes << " errors=" << memory.disk_errors << std::endl;
                require(memory.disk_written_bytes && !memory.disk_errors,
                        "concurrent SSD fixture did not perform successful writes");
                if (std::string(label) != "mixed-prefill")
                    require(memory.disk_read_bytes, "retained concurrent sessions did not read SSD");
            }
            require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0 &&
                    memory.active_payload_reservation_bytes == 0, "terminal lanes retained active resources");
            std::cout << "MEMORY_CONCURRENCY " << label << " rounds=" << rounds << " rows=" << rows
                      << " peak_running=" << peak_running << " peak_waiting=" << peak_waiting
                      << " host=" << memory.host_payload_bytes << std::endl;
            return results;
        };
        std::vector<Work> initial;
        for (std::uint32_t row = 0; row < concurrency; ++row) {
            const auto* image = !options.enable_vision || row % 3 == 2 ? nullptr
                                : row % 3 == 0 ? &red : &blue;
            initial.push_back({history("lane-" + std::to_string(row), "ORCHID-" + std::to_string(7100 + row), image),
                               192 - row * 16, false, "ORCHID-" + std::to_string(7100 + row),
                               image == &red ? "red" : image == &blue ? "blue" : ""});
        }
        // CPU image encoding and staged prefill are serialized. A cold short
        // reply may finish before another image lane becomes decode-ready.
        // Unchanged warm sessions below still must form a compact batch and fill all lanes.
        batch("mixed-prefill", initial, !pressure && !options.enable_vision);
        initial.front().cancel = true;
        batch("cancel-one", initial, !pressure);
        initial.front().cancel = false;
        for (auto& job : initial) job.output = 32;
        batch("resume-peers", initial, !pressure);
        if (options.enable_vision && !pressure) {
            initial[0].prompt = history("lane-0", "ORCHID-7100", &blue);
            initial[0].expected_color = "blue";
            initial[0].output = 96;
            // Changing the image invalidates its archived prefix and runs cold
            // image prefill again. The short unchanged peers need not overlap it.
            batch("replace-image", initial, false);
        }
        if (!pressure) {
            auto older = history("shared", "OLDER-8010");
            auto newer = history("shared", "NEWER-9020");
            batch("publication-order", {{older, 768, false, "OLDER-8010"},
                                        {newer, 24, false, "NEWER-9020"}}, true);
            const auto repeat = batch("newer-retained", {{newer, 24, false}}, false);
            require(repeat.front().reused_prompt_tokens > 0, "older late finish replaced newer session history");
        }
        engine_owner.reset();
        if (options.kvmem.disk_bytes) {
            require(std::filesystem::is_empty(options.kvmem.disk_path), "concurrent spill file leaked");
            std::filesystem::remove(options.kvmem.disk_path);
        }
        std::cout << "MEMORY_CONCURRENCY PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
