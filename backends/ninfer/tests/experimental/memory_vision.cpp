#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ninfer::OwnedMedia media(const char* path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "image fixture is missing");
    ninfer::OwnedMedia value;
    value.bytes.assign(std::istreambuf_iterator<char>(file), {});
    value.media_type = "image/png";
    value.source_name = "image.png"; // Content identity must not depend on the file name.
    return value;
}
void add(ninfer::PromptInput& input, ninfer::ChatRole role, std::string text,
         std::vector<ninfer::OwnedMedia> images = {}) {
    ninfer::ChatMessage message;
    message.role = role;
    for (auto& image : images)
        message.parts.push_back({.kind = ninfer::MessagePartKind::Media, .media = std::move(image)});
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
}
void filler(ninfer::PromptInput& input) {
    for (int index = 0; index < 12; ++index) {
        std::string text = "Archive entry " + std::to_string(index) + ". ";
        for (int count = 0; count < 8; ++count)
            text += "The observatory records pulsars, galaxies, stars and meteor showers. ";
        add(input, ninfer::ChatRole::User, text);
        add(input, ninfer::ChatRole::Assistant, "Recorded.");
    }
}
struct Sink final : ninfer::OutputSink {
    bool cancel_enabled = false;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (cancel_enabled && value.generated_tokens >= 12) cancelled.store(true);
    }
};
}

int main(int argc, char** argv) {
    try {
        require(argc >= 6 && argc <= 13,
                "artifact draft-count(0..15) native|kvmem red.png blue.png [bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8] [eager|graph] [resident|cpu] [all|query-only|native-boundary|copy-compare] [fixed|adaptive] [ngram-width] [ssd-host-MiB]");
        const auto red = media(argv[4]), blue = media(argv[5]);
        const bool memory = std::string(argv[3]) == "kvmem";
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        if (const auto* fast = std::getenv("NINFER_TEST_FAST_PREFILL")) {
            require(std::string(fast) == "0" || std::string(fast) == "1", "NINFER_TEST_FAST_PREFILL must be 0 or 1");
            options.fast_prefill_kernel = fast[0] == '1';
        }
        options.max_context = 8192;
        options.prefill_chunk = 128;
        options.enable_vision = true;
        options.vision_max_merged_tokens = 256;
        options.kv_cache = argc >= 7 ? ninfer::test::kvmem_test_storage(argv[6])
                                    : ninfer::KvCacheStorage::Int8Group64;
        if (argc >= 8) options.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[7]);
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(memory ? 896 : 8192);
        options.context_cache.enabled = false;
        if (memory) options.kvmem = {.selected_tokens = 768, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 4, .verify_transfers = true};
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        if (argc >= 9) {
            require(std::string(argv[8]) == "cpu" || std::string(argv[8]) == "resident", "expected resident or cpu");
            options.vision_residency = std::string(argv[8]) == "cpu" ? ninfer::VisionResidency::Cpu
                                                                   : ninfer::VisionResidency::Resident;
        }
        if (argc >= 11) {
            require(std::string(argv[10]) == "adaptive" || std::string(argv[10]) == "fixed", "expected fixed or adaptive");
            options.speculative.mtp_policy = std::string(argv[10]) == "adaptive"
                ? ninfer::MtpDraftPolicy::Adaptive : ninfer::MtpDraftPolicy::Fixed;
        }
        if (argc >= 12) options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[11]));
        if (argc == 13) {
            require(memory && std::string(argv[9]) == "all", "SSD fixture requires kvmem/all");
            options.kvmem.host_bytes = std::stoull(argv[12]) << 20;
            options.kvmem.disk_bytes = 4ULL << 30;
            options.kvmem.disk_path = std::filesystem::current_path() / "vision-ssd-spill";
        }
        std::cout << "MEMORY_FEATURE fast_prefill=" << options.fast_prefill_kernel << std::endl;
        ninfer::Engine engine(options);
        const auto input = [](const char* key) {
            ninfer::PromptInput result;
            result.options.enable_thinking = false;
            result.context_cache.session_key = key;
            add(result, ninfer::ChatRole::System, "Answer questions about the supplied images in English.");
            return result;
        };
        const auto run = [&](const char* label, ninfer::PromptInput prompt,
                             std::vector<std::string> expected, bool reused, bool cancel = false,
                             std::uint32_t count = 48, float temperature = 0.0F,
                             bool model_stops = false) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = cancel ? 160 : count;
            request.execution.sampling.temperature = temperature;
            request.execution.sampling.seed = 42;
            request.execution.sampling.top_k = 20;
            request.execution.sampling.top_p = 0.9F;
            request.stop.include_model_defaults = model_stops;
            Sink sink;
            sink.cancel_enabled = cancel;
            const auto result = engine.submit(engine.prepare(std::move(prompt)), request,
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
                .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
            require(engine.is_available(), "image request made Engine unavailable");
            require((result.reused_prompt_tokens != 0) == (memory && reused), "wrong image reuse decision");
            require(cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                           : (!result.generated_token_ids.empty() && result.generated_token_ids.size() <= count &&
                              (model_stops || result.generated_token_ids.size() == count)), "wrong image request termination");
            require(result.prompt.has_media, "vision fixture lost media");
            if (options.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive) {
                require(result.speculative.adaptive, "adaptive MTP was silently disabled");
                std::cout << "MEMORY_ADAPTIVE " << label
                          << " transitions=" << result.speculative.window_transitions << " windows=";
                for (auto count : result.speculative.rounds_per_window) std::cout << count << ',';
                std::cout << std::endl;
            }
            auto text = result.content;
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
            for (const auto& word : expected)
                require(text.find(word) != std::string::npos, "image answer did not match fixture");
            if (memory) {
                const auto usage = engine.memory_summary().kvmem;
                require(usage.resident_pages == 0 && usage.mtp_resident_pages == 0 &&
                    usage.host_payload_bytes <= options.kvmem.host_bytes, "image request leaked or exceeded resources");
                require(usage.disk_payload_bytes <= options.kvmem.disk_bytes && usage.disk_errors == 0,
                        "image request exceeded SSD quota or reported an I/O error");
            }
            // Stable token evidence also covers text omitted by presentation.
            std::cout << "MEMORY_VISION_TOKENS " << label << " ids=";
            for (auto token : result.generated_token_ids) std::cout << token << ',';
            std::cout << std::endl;
            std::cout << "MEMORY_VISION " << label << " reused=" << result.reused_prompt_tokens
                      << " output=" << result.generated_token_ids.size() << " accepted="
                      << result.speculative.accepted_tokens << " prompt=" << result.prompt.prompt_tokens
                      << " ng_rounds=" << result.speculative.ngram_rounds
                      << " ng_drafted=" << result.speculative.ngram_drafted_tokens
                      << " ng_accepted=" << result.speculative.ngram_accepted_tokens
                      << " text=" << result.content << std::endl;
            return result;
        };
        const bool copy_compare = argc >= 10 && std::string(argv[9]) == "copy-compare";
        if (options.speculative.ngram_draft_tokens || copy_compare) {
            auto copy = input("image-copy");
            filler(copy);
            std::string passage = "The violet observatory opens at sunrise. ";
            for (int i = 0; i < 12; ++i)
                passage += "Its telescope observes distant galaxies, while the astronomers record the changing light of stars. ";
            add(copy, ninfer::ChatRole::User,
                "First state the image's color in English. Then copy the following passage exactly, without explanations:\n" + passage, {red});
            const auto cold = run("image-copy-cold", copy, {"red", "violet observatory"}, false, false, 192, 0.0F, true);
            if (memory) require(cold.prompt.prompt_tokens > options.kvmem.device_tokens(), "copy did not cross the device window");
            if (options.speculative.ngram_draft_tokens) require(cold.speculative.ngram_rounds > 0 && cold.speculative.ngram_accepted_tokens >
                drafts * cold.speculative.ngram_rounds, "image copy did not exercise wide accepted ngram proposals");
            const auto repeat = run("image-copy-repeat", copy, {"red"}, true, false, 192, 0.0F, true);
            if (cold.generated_token_ids != repeat.generated_token_ids) {
                const auto diff = std::mismatch(cold.generated_token_ids.begin(), cold.generated_token_ids.end(),
                    repeat.generated_token_ids.begin(), repeat.generated_token_ids.end());
                std::cout << "MEMORY_VISION COPY_DIFFERENCE token=" << (diff.first - cold.generated_token_ids.begin())
                          << " cold_size=" << cold.generated_token_ids.size() << " repeat_size=" << repeat.generated_token_ids.size();
                if (diff.first != cold.generated_token_ids.end()) std::cout << " cold_token=" << *diff.first;
                if (diff.second != repeat.generated_token_ids.end()) std::cout << " repeat_token=" << *diff.second;
                std::cout << std::endl;
            }
            require(cold.generated_token_ids == repeat.generated_token_ids, "image ngram restore changed greedy output");
            if (copy_compare) { std::cout << "MEMORY_VISION COPY_COMPARE PASS" << std::endl; return 0; }
            run("image-copy-cancel", copy, {}, true, true, 192, 0.0F, true);
            const auto recovered = run("image-copy-recovery", copy, {"red"}, true, false, 192, 0.0F, true);
            require(cold.generated_token_ids == recovered.generated_token_ids, "image ngram cancellation polluted history");
            for (auto count : {1U, 2U, 65U}) {
                const auto limited = run("image-copy-limit", copy, {}, true, false, count, 0.0F, true);
                require(limited.generated_token_ids.front() == cold.generated_token_ids.front(), "image ngram output limit changed first token");
            }
            const auto sampled = run("image-copy-sampling", copy, {}, true, false, 96, 0.8F, true);
            const auto sampled_again = run("image-copy-sampling-repeat", copy, {}, true, false, 96, 0.8F, true);
            require(sampled.generated_token_ids == sampled_again.generated_token_ids, "image ngram restore changed seeded sampling");
            auto edited = copy;
            edited.messages[1].parts[0].text = "This archive entry was replaced.";
            run("image-copy-edited-prefix", edited, {"red", "violet observatory"}, false, false, 96, 0.0F, true);
        }
        if (argc >= 10 && std::string(argv[9]) != "all") {
            const bool boundary = std::string(argv[9]) == "native-boundary";
            require(boundary || std::string(argv[9]) == "query-only", "expected query-only or native-boundary");
            auto query = input("query-image");
            filler(query);
            add(query, ninfer::ChatRole::User, "What color fills this image?", {red});
            if (boundary) {
                // Align the first visual embedding with a 128-token unit boundary.
                // The preceding unit has no Main visual columns but one shifted MTP column.
                for (int i = 0; i < 36; ++i) query.messages[0].parts[0].text += " x";
            }
            run("query-image-replay", query, {"red"}, false);
            if (boundary) {
                std::cout << "MEMORY_VISION NATIVE_BOUNDARY PASS" << std::endl;
                return 0;
            }
            run("query-image-repeat", query, {"red"}, true);
            auto multiple = input("multiple");
            filler(multiple);
            add(multiple, ninfer::ChatRole::User,
                "Name the colors of the first and second images, in order.", {red, blue});
            run("multiple-query-images", multiple, {"red", "blue"}, false);
            run("query-cancel", multiple, {}, true, true);
            run("query-recovery", multiple, {"red", "blue"}, true);
            std::cout << "MEMORY_VISION QUERY PASS" << std::endl;
            return 0;
        }
        auto short_image = input("short");
        add(short_image, ninfer::ChatRole::User, "What color fills this image?", {red});
        run("single", short_image, {"red"}, false);

        auto old_image = input("old-image");
        add(old_image, ninfer::ChatRole::User, "Remember this image for a later question.", {red});
        add(old_image, ninfer::ChatRole::Assistant, "I will remember it.");
        filler(old_image);
        add(old_image, ninfer::ChatRole::User, "What color filled the image at the start?");
        run("old-image-cold", old_image, {"red"}, false);
        run("old-image-repeat", old_image, {"red"}, true);
        auto replacement = old_image;
        replacement.messages[1].parts[0].media = blue;
        run("same-placeholder-new-media", replacement, {"blue"}, false);
        run("replacement-repeat", replacement, {"blue"}, true);
        run("cancel", replacement, {}, true, true);
        run("cancel-recovery", replacement, {"blue"}, true);

        auto query_image = input("query-image");
        filler(query_image);
        add(query_image, ninfer::ChatRole::User, "What color fills this image?", {red});
        run("query-image-replay", query_image, {"red"}, false);
        run("query-image-repeat", query_image, {"red"}, true);
        auto multiple = input("multiple");
        filler(multiple);
        add(multiple, ninfer::ChatRole::User, "Name the colors of the first and second images, in order.", {red, blue});
        run("multiple-query-images", multiple, {"red", "blue"}, false);
        if (memory) {
            auto oversized = input("over-budget");
            add(oversized, ninfer::ChatRole::User, "Describe all three images.", {red, blue, red});
            bool rejected = false;
            try {
                ninfer::RequestOptions request;
                request.execution.requested_output_tokens = 16;
                (void)engine.generate(engine.prepare(std::move(oversized)), request);
            } catch (const ninfer::RequestError& error) {
                rejected = error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
            }
            require(rejected && engine.is_available(), "image budget failure was not rejected at admission");
            // Admission rejection must preserve the earlier short-input checkpoint.
            run("after-budget-error", short_image, {"red"}, true);
        }
        if (options.kvmem.disk_bytes) {
            const auto usage = engine.memory_summary().kvmem;
            require(usage.disk_read_bytes && usage.disk_written_bytes, "vision fixture did not exercise SSD");
            std::cout << "MEMORY_VISION_SSD reads=" << usage.disk_read_bytes
                      << " writes=" << usage.disk_written_bytes << std::endl;
        }
        std::cout << "MEMORY_VISION PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
