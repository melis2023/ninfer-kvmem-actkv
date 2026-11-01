#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <atomic>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ninfer::PromptInput history() {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.context_cache.session_key = "disk-history";
    const auto add = [&](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
        input.messages.push_back(std::move(message));
    };
    add(ninfer::ChatRole::System, "Read the archive and answer briefly.");
    for (int entry = 0; entry < 16; ++entry) {
        std::string text = "Archive entry " + std::to_string(entry) + ". ";
        for (int repeat = 0; repeat < 8; ++repeat)
            text += "The violet observatory studies pulsars, cosmic rays and distant galaxies. ";
        add(ninfer::ChatRole::User, std::move(text));
        add(ninfer::ChatRole::Assistant, "Recorded.");
    }
    add(ninfer::ChatRole::User, "Describe what the violet observatory studies.");
    return input;
}
struct Sink final : ninfer::OutputSink {
    bool cancel = false;
    std::atomic<bool> cancelled{false};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        if (cancel && value.generated_tokens >= 8) cancelled.store(true);
    }
};
void run(ninfer::Engine& engine, const ninfer::EngineOptions& options,
         const char* label, bool expected_reuse, bool allow_reuse = true, bool cancel = false) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = cancel ? 128 : 24;
    request.execution.sampling.temperature = 0;
    request.execution.allow_prefix_reuse = allow_reuse;
    request.stop.include_model_defaults = false;
    Sink sink;
    sink.cancel = cancel;
    const auto result = engine.submit(engine.prepare(history()), request,
        ninfer::OutputConsumerMode::Streaming, {.live_timings = true})
        .wait(&sink, ninfer::CancellationView([&] { return sink.cancelled.load(); }));
    const auto memory = engine.memory_summary().kvmem;
    require((result.reused_prompt_tokens != 0) == expected_reuse, "wrong process-local reuse decision");
    require(cancel ? result.finish_reason == ninfer::FinishReason::Cancelled
                   : result.generated_token_ids.size() == request.execution.requested_output_tokens,
            "wrong generation termination");
    require(engine.is_available(), "Engine became unavailable");
    if (options.speculative.backend == ninfer::SpeculativeBackend::Mtp)
        require(result.speculative.drafted_tokens > 0, "MTP fixture did not draft tokens");
    if (options.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive)
        require(result.speculative.adaptive, "adaptive MTP was silently disabled");
    require(memory.host_payload_bytes <= options.kvmem.host_bytes, "RAM quota exceeded");
    require(memory.disk_payload_bytes <= options.kvmem.disk_bytes, "SSD quota exceeded");
    require(memory.disk_errors == 0, "unexpected SSD I/O error");
    const char* checkpoint_policy = std::getenv("KVMEM_CHECKPOINT_POLICY");
    if (checkpoint_policy && std::string(checkpoint_policy) == "spread") {
        const char* checkpoint_mib = std::getenv("KVMEM_CHECKPOINT_HOST_MIB");
        const auto checkpoint_budget = (checkpoint_mib ? std::stoull(checkpoint_mib) : 1024ULL) << 20;
        require(memory.checkpoint_bytes <= checkpoint_budget * options.max_concurrency,
                "checkpoint Host quota exceeded");
    }
    require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0 &&
            memory.active_payload_reservation_bytes == 0, "terminal request leaked active resources");
    std::cout << "MEMORY_DISK case=" << label << " reused=" << result.reused_prompt_tokens
              << " H=" << memory.host_payload_bytes << " D=" << memory.disk_payload_bytes
              << " checkpoint_host=" << memory.checkpoint_bytes
              << " reads=" << memory.disk_read_bytes << " writes=" << memory.disk_written_bytes
              << " errors=" << memory.disk_errors << std::endl;
}
std::set<std::filesystem::path> files(const std::filesystem::path& directory) {
    std::set<std::filesystem::path> result;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        result.insert(entry.path().filename());
    return result;
}
}

int main(int argc, char** argv) {
    try {
        require(argc >= 6 && argc <= 8,
                "artifact drafts(0..15) kv-format cache-dir eager|graph [ngram-width] [fixed|adaptive]; "
                "runs process-local SSD lifecycle; obsolete cold/hit snapshot modes are unsupported");
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 16384;
        options.prefill_chunk = 128;
        options.kv_cache = ninfer::test::kvmem_test_storage(argv[3]);
        options.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[5]);
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 1ULL << 30, .retained_sessions = 1, .verify_transfers = true};
        options.context_cache.enabled = false;
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        require(drafts <= 15, "draft count must be in [0,15]");
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        if (argc >= 7) options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[6]));
        if (argc == 8) {
            const std::string policy = argv[7];
            require(policy == "fixed" || policy == "adaptive", "invalid MTP policy");
            if (policy == "adaptive") options.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
        }
        // Derive pressure from this artifact's native page size, as active_kv_ssd
        // does. A hard-coded H can accidentally turn the entire test into RAM-only.
        std::uint64_t history_bytes = 0, record_bytes = 0;
        {
            ninfer::Engine reference(options);
            run(reference, options, "ram-sizing", false);
            const auto memory = reference.memory_summary().kvmem;
            const auto pages = (memory.checkpoint_tokens + 63ULL) / 64;
            history_bytes = memory.host_payload_bytes;
            require(pages > 10 && history_bytes && history_bytes % pages == 0,
                    "cannot derive native record size from the RAM fixture");
            record_bytes = history_bytes / pages;
        }
        options.kvmem.host_bytes = ((10 * record_bytes + (1ULL << 20) - 1) >> 20) << 20;
        require(options.kvmem.host_bytes < history_bytes, "fixture does not force active SSD spill");
        options.kvmem.disk_bytes = 512ULL << 20;
        options.kvmem.disk_path = argv[4];
        std::filesystem::create_directories(options.kvmem.disk_path);
        const auto before_files = files(options.kvmem.disk_path);
        {
            ninfer::Engine engine(options);
            const auto initial = engine.memory_summary().kvmem;
            require(initial.retained_sessions == 0 && initial.history_hits == 0 &&
                    initial.disk_read_bytes == 0 && initial.disk_payload_bytes == 0,
                    "a new Engine imported history from an earlier process");
            run(engine, options, "cold", false);
            run(engine, options, "reuse", true);
            run(engine, options, "cancel", true, true, true);
            run(engine, options, "resume", true);
            run(engine, options, "clear", false, false);
            run(engine, options, "after-clear", false);
            const auto memory = engine.memory_summary().kvmem;
            require(memory.disk_read_bytes > 0 && memory.disk_written_bytes > 0 && memory.disk_payload_bytes > 0,
                    "fixture did not exercise live SSD reads and writes");
        }
        require(files(options.kvmem.disk_path) == before_files,
                "Engine shutdown leaked a spill file or removed an unrelated directory entry");
        std::cout << "MEMORY_DISK PASS process_local=1 cleanup=1" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
