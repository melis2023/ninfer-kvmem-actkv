#include "guarded_main.h"
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {
std::vector<ninfer::TokenId> needle_prompt(const ninfer::Engine& engine, std::size_t total,
                                          std::size_t needle_at, const std::string& combination) {
    auto tokens = engine.tokenize_text("<|im_start|>user\nRead the notes and recall the vault combination.\n");
    const auto filler = engine.tokenize_text(
        "The library opens at nine. The garden contains apple trees. "
        "The river passes a stone bridge. The railway station has four platforms.\n");
    const auto needle = engine.tokenize_text("Remember this: the vault combination is " +
                                            combination + ".\n");
    const auto ending = engine.tokenize_text(
        "\nWhat is the vault combination? Reply with the four digits only.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    if (filler.empty() || tokens.size() >= needle_at ||
        needle_at + needle.size() + ending.size() >= total) {
        throw std::runtime_error("invalid long-context fixture geometry");
    }
    auto fill = [&](std::size_t end) {
        std::size_t at = 0;
        while (tokens.size() < end) { tokens.push_back(filler[at++ % filler.size()]); }
    };
    fill(needle_at);
    tokens.insert(tokens.end(), needle.begin(), needle.end());
    fill(total - ending.size());
    tokens.insert(tokens.end(), ending.begin(), ending.end());
    return tokens;
}

int run() {
    const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
    if (!path) { std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT required\n"; return 77; }
    ninfer::EngineOptions options;
    options.artifact_path = path;
    const char* device_env = std::getenv("NINFER_FLASH_NEXT_DEVICES");
    std::stringstream device_list(device_env ? device_env : "0,1");
    for (std::string value; std::getline(device_list, value, ',');) {
        options.devices.push_back(std::stoi(value));
    }
    options.max_context = 131072;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(131072);
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.max_concurrency = 1;
    options.max_pending_requests = 1;
    options.prefill_chunk = 512;
    options.context_cache.enabled = false;
    if (const char* table = std::getenv("NINFER_FLASH_NEXT_NGRAM_TABLE")) {
        options.ngram_table.path = table;
    }
    unsigned drafts = 0;
    if (const char* value = std::getenv("NINFER_FLASH_NEXT_DRAFTS")) { drafts = std::stoul(value); }
    if (drafts) {
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = drafts;
    }
    ninfer::Engine engine(options);
    const char* report_path = std::getenv("NINFER_FLASH_NEXT_PIPELINE_REPORT");
    nlohmann::json report = {{"schema", "flash-next-pipeline-1"}, {"devices", options.devices},
                             {"drafts", drafts}, {"kv", "int8_g64"},
                             {"prefill_chunk", options.prefill_chunk},
                             {"max_context", options.max_context},
                             {"cases", nlohmann::json::array()}};
    std::vector<std::pair<unsigned, unsigned>> contexts = {{33024U, 32768U}, {131008U, 98304U}};
    if (report_path) { contexts.insert(contexts.begin(), {4032U, 2048U}); }
    int failures = 0;
    for (const auto [count, position] : contexts) {
        const std::string combination = count < 65536 ? "9517" : "6823";
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = report_path ? 64 : 24;
        request.execution.sampling.temperature = 0.0F;
        request.execution.allow_prefix_reuse = false;
        request.stop.include_model_defaults = report_path == nullptr;
        const auto input = needle_prompt(engine, count, position, combination);
        report["cases"].push_back({{"prompt_tokens", count}, {"needle_position", position},
                                   {"expected", combination}, {"input_tokens", input},
                                   {"samples", nlohmann::json::array()}});
        ninfer::GenerationResult first;
        for (int repeat = 0; repeat < (report_path ? 3 : 1); ++repeat) {
            auto prompt = engine.prepare_tokens(input, false);
            const auto start = std::chrono::steady_clock::now();
            const auto result = engine.submit(std::move(prompt), request,
                ninfer::OutputConsumerMode::Aggregate, {.phase_timings = true}).wait();
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            bool pass = result.prompt.prompt_tokens == count &&
                        result.content.find(combination) != std::string::npos &&
                        result.reused_prompt_tokens == 0;
            if (report_path) {
                pass = pass && result.generated_token_ids.size() == 64 &&
                       result.finish_reason == ninfer::FinishReason::OutputLimit;
                if (repeat == 0) { first = result; }
                else {
                    pass = pass && result.generated_token_ids == first.generated_token_ids &&
                           result.content == first.content && result.reasoning == first.reasoning &&
                           result.finish_reason == first.finish_reason;
                }
                report["cases"].back()["samples"].push_back({
                    {"repeat", repeat}, {"pass", pass}, {"seconds", seconds},
                    {"output_tokens", result.generated_token_ids}, {"content", result.content},
                    {"reasoning", result.reasoning},
                    {"finish_reason", static_cast<int>(result.finish_reason)},
                    {"reused_prompt_tokens", result.reused_prompt_tokens},
                    {"prefill_seconds", result.timings.prefill_seconds},
                    {"decode_seconds", result.timings.decode_seconds},
                    {"prompt_wall_seconds", result.timings.prompt_wall_seconds},
                    {"generation_wall_seconds", result.timings.generation_wall_seconds}});
                std::ofstream output(report_path);
                if (!output) { throw std::runtime_error("cannot write pipeline report"); }
                output << report.dump(2) << '\n';
            }
            std::cout << "LONG_CONTEXT prompt_tokens=" << result.prompt.prompt_tokens
                      << " needle_position=" << position << " drafts=" << drafts
                      << " repeat=" << repeat << " expected=" << combination
                      << " seconds=" << seconds << " answer=" << result.content
                      << " pass=" << pass << std::endl;
            failures += pass ? 0 : 1;
        }
    }
    return failures ? 1 : 0;
}
} // namespace

NINFER_GUARDED_TEST_MAIN(run)
