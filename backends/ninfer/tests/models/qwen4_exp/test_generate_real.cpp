// Qwen3.8-Flash-Next end to end on a real artifact: load it (experts on the stage devices, in
// pinned host memory or on disk), feed chat prompts and greedy-decode short answers, checking that
// each answer names the expected fact and reporting prefill and decode throughput. Where decode
// replays CUDA graphs (experts not on disk), the prompts run again on an eager executor and every
// generated token must be the same.
//
//   NINFER_QWEN4_EXP_ARTIFACT  the model's .ninfer (required; skips without it)
//   NINFER_QWEN4_EXP_DEVICES   comma-separated device ids, one pipeline stage each (default 0)
//   NINFER_QWEN4_EXP_EXPERTS   device | host | disk (default device)
//   NINFER_QWEN4_EXP_NGRAM_TABLE  the n-gram table artifact of a model stored without its table
//   NINFER_QWEN4_EXP_NGRAM_RAM 1 loads the n-gram table into RAM
//   NINFER_QWEN4_EXP_PREFILL_CHUNK  tokens per prefill call (default 512)
//   NINFER_QWEN4_EXP_EXPERT_CACHE_MIB  host or disk experts: the device expert cache (default: what
//                                      the devices have free)
#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen4_exp/executor.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/ngram_component.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::vector<int> device_list() {
    const char* value = std::getenv("NINFER_QWEN4_EXP_DEVICES");
    std::vector<int> out;
    std::stringstream stream(value != nullptr ? value : "0");
    for (std::string item; std::getline(stream, item, ',');) { out.push_back(std::stoi(item)); }
    return out;
}

// The executor queues on its own (non-blocking) streams: read the logits after they are done.
void read_logits(const Executor& executor, std::vector<__nv_bfloat16>& out) {
    if (cudaStreamSynchronize(executor.head_stream()) != cudaSuccess ||
        cudaMemcpy(out.data(), executor.logits(1).data, out.size() * 2, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        throw std::runtime_error("reading the logits failed");
    }
}

int argmax(const std::vector<__nv_bfloat16>& logits, std::size_t domain) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < domain; ++i) {
        if (__bfloat162float(logits[i]) > __bfloat162float(logits[best])) { best = i; }
    }
    return static_cast<int>(best);
}

struct Prompt {
    std::string text;
    std::string expect; // empty: any answer, for a throughput sample
    int max_tokens = 24;
};

// Runs every prompt from an empty sequence of a fresh executor; returns each prompt's greedy tokens
// and counts the answers that miss their fact.
std::vector<std::vector<int>> generate(const Model& model, DeviceContext& device,
                                       const ExecutorOptions& options,
                                       const std::vector<Prompt>& prompts, int& failures) {
    Executor executor(model, device, options);
    device.synchronize();
    std::cout << "executor: state " << executor.memory().state_bytes / 1e6 << " MB, workspace "
              << executor.memory().workspace_bytes / 1e6 << " MB, expert cache "
              << executor.memory().expert_cache_bytes / 1e6 << " MB\n";
    const auto& tokenizer    = *model.resources().tokenizer;
    const std::size_t domain = model.resources().public_token_count;
    std::vector<std::vector<int>> out;
    std::vector<__nv_bfloat16> logits(model.config().vocab_size);
    for (const auto& prompt : prompts) {
        executor.reset(0);
        const auto ids = tokenizer.encode(prompt.text, {.parse_added_tokens = true});
        std::vector<std::int32_t> tokens(ids.begin(), ids.end());
        const auto prefill_start = Clock::now();
        for (std::size_t at = 0; at < tokens.size(); at += options.prefill_chunk) {
            const std::size_t n = std::min<std::size_t>(options.prefill_chunk, tokens.size() - at);
            executor.forward(0, std::span(tokens).subspan(at, n), 1);
        }
        read_logits(executor, logits);
        const auto prefill_end = Clock::now();
        std::vector<int> generated;
        int next = argmax(logits, domain);
        // Where a decode step's time goes: queueing its work on the host, then waiting for the
        // device to finish it.
        double enqueue = 0.0, wait = 0.0;
        for (int step = 0; step < prompt.max_tokens; ++step) {
            generated.push_back(next);
            if (next == 248046 || next == 248044) { break; }
            const std::int32_t token = next;
            const auto t0 = Clock::now();
            executor.forward(0, std::span(&token, 1), 1);
            const auto t1 = Clock::now();
            read_logits(executor, logits);
            enqueue += seconds(t0, t1);
            wait += seconds(t1, Clock::now());
            next = argmax(logits, domain);
        }
        const auto decode_end  = Clock::now();
        const std::string text = tokenizer.decode(generated);
        const bool ok          = text.find(prompt.expect) != std::string::npos;
        std::cout << (ok ? "OK   " : "FAIL ") << tokens.size() << " prompt tokens in "
                  << seconds(prefill_start, prefill_end) << " s, " << generated.size()
                  << " tokens at " << double(generated.size()) / seconds(prefill_end, decode_end)
                  << " tok/s (per step: enqueue " << 1e3 * enqueue / std::max<std::size_t>(1, generated.size() - 1)
                  << " ms, wait " << 1e3 * wait / std::max<std::size_t>(1, generated.size() - 1)
                  << " ms): \"" << text << "\"\n";
        failures += ok ? 0 : 1;
        const auto cache = executor.expert_cache_stats();
        if (cache.slots != 0) {
            std::cout << "     expert cache: " << cache.slots << " slots, hit rate "
                      << double(cache.hits) / double(std::max<std::uint64_t>(cache.routes, 1))
                      << " over " << cache.routes << " routes, " << cache.admitted << " admitted ("
                      << cache.copied_bytes / 1e9 << " GB)\n";
        }
        out.push_back(std::move(generated));
    }
    return out;
}

// Prefills `text` into `sequence` and returns its first greedy token.
int prefill(Executor& executor, const Model& model, std::uint32_t sequence, const std::string& text,
            std::uint32_t chunk, std::vector<__nv_bfloat16>& logits) {
    const auto ids = model.resources().tokenizer->encode(text, {.parse_added_tokens = true});
    const std::vector<std::int32_t> tokens(ids.begin(), ids.end());
    for (std::size_t at = 0; at < tokens.size(); at += chunk) {
        const std::size_t n = std::min<std::size_t>(chunk, tokens.size() - at);
        executor.forward(sequence, std::span(tokens).subspan(at, n), 1);
    }
    read_logits(executor, logits);
    return argmax(logits, model.resources().public_token_count);
}

// Greedy tokens of `steps` decode steps of one sequence, alone.
std::vector<int> decode_alone(Executor& executor, const Model& model, std::uint32_t sequence,
                              int first, int steps, std::vector<__nv_bfloat16>& logits) {
    std::vector<int> out{first};
    for (int step = 1; step < steps; ++step) {
        const std::int32_t token = out.back();
        executor.forward(sequence, std::span(&token, 1), 1);
        read_logits(executor, logits);
        out.push_back(argmax(logits, model.resources().public_token_count));
    }
    return out;
}

// Several sequences decoded together in one batched pass per step must produce the greedy tokens
// each produces alone (the experts read their weights once for the batch, each sequence's mixers
// run on its own state), and a sequence restored from a snapshot must continue as it did before.
int check_batch_and_snapshot(const Model& model, DeviceContext& device, ExecutorOptions options) {
    const std::vector<std::string> texts = {
        "<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n",
        "<|im_start|>user\nCount from one to ten in words.<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n",
        "<|im_start|>user\nWhat is the boiling point of water in Celsius?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"};
    constexpr int kSteps = 24;
    options.sequences    = static_cast<std::uint32_t>(texts.size());
    Executor executor(model, device, options);
    std::vector<__nv_bfloat16> logits(model.config().vocab_size);
    std::vector<std::vector<int>> alone;
    for (std::uint32_t s = 0; s < texts.size(); ++s) {
        executor.reset(s);
        alone.push_back(decode_alone(executor, model, s,
                                     prefill(executor, model, s, texts[s], options.prefill_chunk,
                                             logits),
                                     kSteps, logits));
    }
    std::vector<std::vector<int>> together(texts.size());
    for (std::uint32_t s = 0; s < texts.size(); ++s) {
        executor.reset(s);
        together[s].push_back(prefill(executor, model, s, texts[s], options.prefill_chunk, logits));
    }
    const std::size_t domain = model.resources().public_token_count;
    std::vector<__nv_bfloat16> batch_logits(logits.size() * texts.size());
    std::vector<std::uint32_t> sequences(texts.size());
    std::iota(sequences.begin(), sequences.end(), 0U);
    for (int step = 1; step < kSteps; ++step) {
        std::vector<std::int32_t> tokens;
        for (const auto& row : together) { tokens.push_back(row.back()); }
        executor.decode(sequences, tokens);
        if (cudaStreamSynchronize(executor.head_stream()) != cudaSuccess ||
            cudaMemcpy(batch_logits.data(), executor.logits(std::uint32_t(texts.size())).data,
                       batch_logits.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("reading the batch logits failed");
        }
        for (std::size_t s = 0; s < texts.size(); ++s) {
            const std::vector<__nv_bfloat16> column(
                batch_logits.begin() + std::ptrdiff_t(s * logits.size()),
                batch_logits.begin() + std::ptrdiff_t((s + 1) * logits.size()));
            together[s].push_back(argmax(column, domain));
        }
    }
    int failures = 0;
    for (std::size_t s = 0; s < texts.size(); ++s) {
        const bool same = together[s] == alone[s];
        std::cout << (same ? "OK   " : "FAIL ") << "batched decode of sequence " << s << ": \""
                  << model.resources().tokenizer->decode(together[s]) << "\"\n";
        failures += same ? 0 : 1;
    }
    // A snapshot after the prompt, a detour of decode steps, then the restored sequence.
    SequenceSnapshot snapshot;
    executor.reset(0);
    const int first = prefill(executor, model, 0, texts[0], options.prefill_chunk, logits);
    executor.snapshot(0, snapshot);
    const auto before = decode_alone(executor, model, 0, first, kSteps, logits);
    executor.restore(0, snapshot);
    const auto after = decode_alone(executor, model, 0, first, kSteps, logits);
    const bool same  = before == after && before == alone[0];
    std::cout << (same ? "OK   " : "FAIL ") << "restored sequence continues as before\n";
    failures += same ? 0 : 1;
    return failures;
}

int run(const char* artifact_path) {
    const auto start   = Clock::now();
    const auto devices = device_list();
    DeviceContext device{std::span<const int>(devices)};
    const artifact::Reader reader(artifact_path);
    LoadOptions load;
    load.artifact       = artifact_path;
    load.ranks          = devices.size();
    const char* experts = std::getenv("NINFER_QWEN4_EXP_EXPERTS");
    const std::string residency = experts != nullptr ? experts : "device";
    load.experts = residency == "host"   ? ExpertResidency::Host
                   : residency == "disk" ? ExpertResidency::Disk
                                         : ExpertResidency::Device;
    auto model   = load_model(reader, load, device);
    device.synchronize();
    const auto loaded = Clock::now();
    std::cout << "loaded in " << seconds(start, loaded) << " s: stages";
    for (std::size_t s = 0; s < model->stages().stages(); ++s) {
        std::cout << ' ' << model->stages().stage_layers(s);
    }
    std::cout << ", read " << model->storage_stats().read_bytes / 1e9 << " GB, pinned "
              << model->storage_stats().pinned_bytes / 1e9 << " GB\n";

    ExecutorOptions options;
    options.max_context   = 8192;
    const char* chunk     = std::getenv("NINFER_QWEN4_EXP_PREFILL_CHUNK");
    options.prefill_chunk = chunk != nullptr ? static_cast<std::uint32_t>(std::stoul(chunk)) : 512;
    if (const char* cache = std::getenv("NINFER_QWEN4_EXP_EXPERT_CACHE_MIB")) {
        options.expert_cache_bytes = std::uint64_t(std::stoull(cache)) << 20;
    }
    const char* table     = std::getenv("NINFER_QWEN4_EXP_NGRAM_TABLE");
    options.ngram         = ngram_table_source(reader, artifact_path, model->config(),
                                               table != nullptr ? table : "");
    const char* ram       = std::getenv("NINFER_QWEN4_EXP_NGRAM_RAM");
    options.ngram_residency =
        ram != nullptr && std::string(ram) == "1" ? NgramResidency::Ram : NgramResidency::Disk;
    std::cout << "n-gram table: " << options.ngram->layout.rows << " rows of "
              << options.ngram->layout.row_bytes << " bytes in "
              << options.ngram->layout.segments.size() << " file segment(s) of "
              << options.ngram->layout.segments.front().path.filename().string() << "\n";
    std::vector<Prompt> prompts = {
        {"<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
         "Paris"},
        {"<|im_start|>user\nWhat is 17 multiplied by 23? Reply with just the number.<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
         "391"},
        {"<|im_start|>user\nDescribe the water cycle in three sentences.<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n",
         "", 96},
    };
    // A fact deep inside a long context: several prefill chunks and the indexer's sparse
    // selection (past 2,051 positions) must still find it.
    std::string haystack;
    const char* filler[] = {
        "The river bends twice before it reaches the old mill, where the water slows. ",
        "Farmers in the valley rotate barley and clover to keep the soil rich. ",
        "A lighthouse keeper logs the passing ships and the weather every hour. ",
        "The library catalog lists maps, letters and ledgers from three centuries. ",
    };
    for (int i = 0; i < 300; ++i) {
        haystack += filler[i % 4];
        if (i == 97) { haystack += "Remember this: the vault combination is 4771. "; }
    }
    prompts.push_back(
        {"<|im_start|>user\n" + haystack +
             "\nWhat is the vault combination? Reply with the number only.<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
         "4771"});
    int failures = 0;
    const auto replayed = generate(*model, device, options, prompts, failures);
    failures += check_batch_and_snapshot(*model, device, options);
    if (load.experts != ExpertResidency::Disk) {
        std::cout << "eager decode (no CUDA graphs):\n";
        options.cuda_graphs = false;
        const auto eager    = generate(*model, device, options, prompts, failures);
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            if (eager[i] != replayed[i]) {
                const auto at = std::mismatch(eager[i].begin(), eager[i].end(), replayed[i].begin(),
                                              replayed[i].end());
                std::cout << "FAIL prompt " << i << ": graph decode diverges from eager decode at "
                          << "token " << (at.first - eager[i].begin()) << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN4_EXP_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "SKIP: NINFER_QWEN4_EXP_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const int failures = run(artifact);
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " qwen4_exp generate\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
