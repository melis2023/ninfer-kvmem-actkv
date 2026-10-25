#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ninfer::PromptInput history() {
    ninfer::PromptInput input;
    input.context_cache.session_key = "active-ssd";
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
            text += entry < 4 ? "The violet observatory studies pulsars, cosmic rays and distant galaxies. "
                             : "The coastal office files invoices, transport schedules and warehouse inventories. ";
        add(ninfer::ChatRole::User, text);
        add(ninfer::ChatRole::Assistant, "Recorded.");
    }
    add(ninfer::ChatRole::User, "What does the violet observatory study?");
    return input;
}
struct Observation {
    std::vector<std::int32_t> tokens;
    std::uint32_t reused = 0;
    ninfer::KvmemMemorySummary memory;
};
std::vector<Observation> exercise(const ninfer::EngineOptions& options) {
    ninfer::Engine engine(options);
    std::vector<Observation> observations;
    auto input = history();
    const bool controls = !options.thinking_budget_message.empty();
    if (controls) input.options.enable_thinking = true;
    for (int step = 0; step < 3; ++step) {
        if (step == 2) input.messages.back().parts.front().text = "Which celestial objects appear in the early archive?";
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = controls ? 512 : 24;
        if (controls) request.execution.thinking.budget = 1;
        request.execution.sampling.temperature = 0.0f;
        request.stop.include_model_defaults = false;
        const auto result = engine.submit(engine.prepare(input), request).wait();
        require(engine.is_available(), "Engine became unavailable");
        require(result.generated_token_ids.size() == (controls ? 512 : 24), "wrong output length");
        if (controls) require(result.thinking.applied && result.thinking.injected_tokens > 128,
                              "control fixture did not cross a native reserve boundary");
        require((result.reused_prompt_tokens != 0) == (step != 0), "wrong session reuse");
        auto memory = engine.memory_summary().kvmem;
        if (options.kvmem.disk_bytes) {
            // Idle writes may outlive response publication. Observe their
            // eventual bounded placement, without treating this wait as latency.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            auto unchanged_since = std::chrono::steady_clock::now();
            while (memory.checkpoint_bytes && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                auto next = engine.memory_summary().kvmem;
                if (next.checkpoint_bytes != memory.checkpoint_bytes || next.disk_written_bytes != memory.disk_written_bytes)
                    unchanged_since = std::chrono::steady_clock::now();
                memory = next;
                if (std::chrono::steady_clock::now() - unchanged_since > std::chrono::seconds(1)) break;
            }
        }
        require(memory.host_payload_bytes <= options.kvmem.host_bytes, "RAM quota exceeded");
        require(memory.disk_payload_bytes <= options.kvmem.disk_bytes, "disk quota exceeded");
        require(memory.disk_errors == 0, "unexpected disk I/O error");
        require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0, "GPU pages leaked");
        observations.push_back({{result.generated_token_ids.begin(), result.generated_token_ids.end()},
            result.reused_prompt_tokens, memory});
        std::cout << "ACTIVE_SSD step=" << step << " H=" << memory.host_payload_bytes
                  << " D=" << memory.disk_payload_bytes << " reads=" << memory.disk_read_bytes
                  << " writes=" << memory.disk_written_bytes << " checkpoint=" << memory.checkpoint_tokens
                  << " state_ram=" << memory.checkpoint_bytes
                  << " reused=" << result.reused_prompt_tokens << std::endl;
    }
    return observations;
}
}

int main(int argc, char** argv) {
    try {
        require(argc >= 3 && argc <= 6, "artifact draft-count(0..4) [int8|rk8v4] [eager|graph] [control]");
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 16384;
        options.prefill_chunk = 128;
        options.kv_cache = argc >= 4 ? ninfer::test::kvmem_test_storage(argv[3]) : ninfer::KvCacheStorage::Int8Group64;
        if (argc >= 5) options.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[4]);
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 1ULL << 30, .retained_sessions = 1, .verify_transfers = true};
        options.context_cache.enabled = false;
        if (argc == 6) {
            require(std::string(argv[5]) == "control", "unknown SSD oracle mode");
            for (int i = 0; i < 64; ++i) options.thinking_budget_message += "Finish the reasoning. ";
        }
        const auto drafts = static_cast<std::uint32_t>(std::stoul(argv[2]));
        if (drafts) {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
        }
        const auto reference = exercise(options);
        const auto pages = (reference.front().memory.checkpoint_tokens + 63ULL) / 64;
        const auto history_bytes = reference.front().memory.host_payload_bytes;
        require(pages > 10 && history_bytes % pages == 0, "cannot derive native record size");
        const auto record_bytes = history_bytes / pages;
        options.kvmem.host_bytes = ((10 * record_bytes + (1ULL << 20) - 1) >> 20) << 20;
        require(options.kvmem.host_bytes < history_bytes, "fixture does not force active SSD spill");
        options.kvmem.disk_bytes = 512ULL << 20;
        options.kvmem.disk_path = std::filesystem::temp_directory_path() /
            ("ninfer-active-ssd-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        const auto actual = exercise(options);
        for (std::size_t i = 0; i < reference.size(); ++i) {
            require(actual[i].tokens == reference[i].tokens, "SSD changed generated token IDs");
            require(actual[i].reused == reference[i].reused, "SSD changed prefix reuse");
            require(actual[i].memory.disk_payload_bytes > 0, "no live SSD payload");
            require(actual[i].memory.disk_read_bytes > 0 && actual[i].memory.disk_written_bytes > 0,
                    "active inference did not read and write SSD");
            require(actual[i].memory.checkpoint_bytes < reference[i].memory.checkpoint_bytes,
                    "idle checkpoint did not share the SSD account");
            require(actual[i].memory.disk_payload_bytes >=
                        reference[i].memory.checkpoint_bytes - actual[i].memory.checkpoint_bytes,
                    "idle checkpoint bytes bypassed the disk quota");
        }
        require(std::filesystem::is_empty(options.kvmem.disk_path), "temporary spill file leaked");
        std::filesystem::remove(options.kvmem.disk_path);
        std::cout << "ACTIVE_KV_SSD PASS" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
