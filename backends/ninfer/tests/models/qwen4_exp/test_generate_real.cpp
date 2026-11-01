// Qwen3.8-Flash-Next end to end on a real artifact: load it (experts on the stage devices, in
// pinned host memory or on disk), feed chat prompts and greedy-decode short answers, checking that
// each answer names the expected fact and reporting prefill and decode throughput. Where decode
// replays CUDA graphs (experts not on disk), the prompts run again on an eager executor and every
// generated token must be the same. An artifact with its MTP block also decodes the prompts by
// speculative rounds (MTP drafts, one verification, the greedy run kept, the rest dropped by the
// commit), alone and two sequences at once: fixed-mode repeats must match; differences from plain
// decode or another batch composition are reported. The context cache's
// sequence images (live and at a snapshot) must continue as the sequence they were taken from.
//
//   NINFER_QWEN4_EXP_ARTIFACT  the model's .ninfer (required; skips without it)
//   NINFER_QWEN4_EXP_DRAFTS    MTP drafts a round (default 3; 0 skips the speculative check)
//   NINFER_QWEN4_EXP_DEVICES   comma-separated device ids, one pipeline stage each (default 0)
//   NINFER_QWEN4_EXP_EXPERTS   device | host | disk (default device)
//   NINFER_QWEN4_EXP_NGRAM_TABLE  the n-gram table artifact of a model stored without its table
//   NINFER_QWEN4_EXP_NGRAM_RESIDENCY  disk | ram | ram-hot (default disk); ram-hot takes the
//                                     profile NINFER_QWEN4_EXP_NGRAM_PROFILE and 4 GiB of RAM
//   NINFER_QWEN4_EXP_NGRAM_IO  buffered | direct | mmap (default buffered)
//   NINFER_QWEN4_EXP_PREFILL_CHUNK  tokens per prefill call (default 512)
//   NINFER_QWEN4_EXP_EXPERT_CACHE_MIB  host or disk experts: the device expert cache (default: what
//                                      the devices have free)
#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen4_exp/executor.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/ngram_component.h"
#include "models/qwen4_exp/ngram_profile.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::string env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value != nullptr ? value : fallback;
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
    // The n-gram rows: read per pass, and the time the PLE layer's device waited for them.
    const NgramTableStats rows = executor.ngram_stats();
    std::cout << "n-gram rows: " << rows.rows << " in " << rows.batches << " passes, "
              << rows.resident_rows << " from RAM, "
              << 1e6 * rows.read_seconds / double(std::max<std::uint64_t>(rows.batches, 1))
              << " us a pass; " << rows.stalls << " stalls, " << 1e3 * rows.stall_seconds
              << " ms waited\n";
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

// Each fixed batch must repeat exactly; record differences from solitary greedy decode because
// batch width can change floating-point reductions. A restored sequence must continue exactly.
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
    std::vector<std::vector<int>> together(texts.size()), first_batch;
    for (int repetition = 0; repetition < 2; ++repetition) {
        together.assign(texts.size(), {});
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
        if (repetition == 0) { first_batch = together; }
    }
    int failures = 0;
    for (std::size_t s = 0; s < texts.size(); ++s) {
        const bool same = together[s] == alone[s];
        std::cout << (same ? "SAME " : "DIFF ") << "batched versus alone, sequence " << s << ": \""
                  << model.resources().tokenizer->decode(together[s]) << "\"\n";
        const bool repeat = together[s] == first_batch[s];
        std::cout << (repeat ? "OK   " : "FAIL ") << "batched fixed-mode repeat " << s << '\n';
        failures += repeat ? 0 : 1;
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

// Greedy tokens of one speculative round per step for `sequences` (prefilled, their first tokens
// in `rows`), until each row holds `steps` tokens: the MTP block drafts from each row's last
// token, the target verifies the anchor and the drafts, the drafts up to the first that differs
// from the verification's argmax are accepted with the argmax after them, and the commit keeps the
// tokens fed up to there. Counts the drafts proposed and accepted.
// With `wrong`, one draft of each round (a different one each round) is replaced by another
// token, so rounds keep every number of tokens from one to all.
void decode_speculative(Executor& executor, const Model& model,
                        std::span<const std::uint32_t> sequences,
                        std::vector<std::vector<int>>& rows, int steps, std::uint64_t& drafted,
                        std::uint64_t& accepted, bool wrong = false) {
    const std::size_t k      = executor.draft_tokens();
    const std::size_t w      = k + 1;
    const std::size_t b      = sequences.size();
    const std::size_t vocab  = model.config().vocab_size;
    const std::size_t domain = model.resources().public_token_count;
    std::vector<__nv_bfloat16> logits(vocab * w * b);
    for (std::size_t round = 0;; ++round) {
        std::vector<std::uint32_t> live;
        std::vector<std::size_t> index;
        for (std::size_t j = 0; j < b; ++j) {
            if (int(rows[j].size()) < steps) {
                live.push_back(sequences[j]);
                index.push_back(j);
            }
        }
        if (live.empty()) { return; }
        std::vector<std::int32_t> anchors, proposed(live.size() * k), tokens;
        for (const std::size_t j : index) { anchors.push_back(rows[j].back()); }
        executor.draft(live, anchors, proposed);
        if (wrong && round % (k + 1) < k) {
            for (std::size_t r = 0; r < live.size(); ++r) {
                auto& draft = proposed[r * k + round % (k + 1)];
                draft       = (draft + 1) % static_cast<std::int32_t>(domain);
            }
        }
        for (std::size_t r = 0; r < live.size(); ++r) {
            tokens.push_back(anchors[r]);
            tokens.insert(tokens.end(), proposed.begin() + std::ptrdiff_t(r * k),
                          proposed.begin() + std::ptrdiff_t((r + 1) * k));
        }
        executor.verify(live, tokens);
        const std::size_t columns = live.size() * w;
        if (cudaStreamSynchronize(executor.head_stream()) != cudaSuccess ||
            cudaMemcpy(logits.data(), executor.logits(std::uint32_t(columns)).data,
                       columns * vocab * 2, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("reading the verification logits failed");
        }
        std::vector<std::uint32_t> kept;
        for (std::size_t r = 0; r < live.size(); ++r) {
            auto& row       = rows[index[r]];
            std::size_t col = 0;
            for (;; ++col) {
                const std::vector<__nv_bfloat16> column(
                    logits.begin() + std::ptrdiff_t((r * w + col) * vocab),
                    logits.begin() + std::ptrdiff_t((r * w + col + 1) * vocab));
                const int target = argmax(column, domain);
                row.push_back(target);
                if (col == k || target != proposed[r * k + col] || int(row.size()) >= steps) {
                    break;
                }
            }
            drafted += k;
            accepted += col < k ? col : k;
            kept.push_back(std::uint32_t(col + 1));
        }
        executor.commit(live, kept);
    }
}

// Check expected facts, fixed-profile repeats and snapshot continuation. Report plain/MTP and
// batch-composition differences without treating another arithmetic profile as an exact oracle.
int check_speculative(const Model& model, DeviceContext& device, ExecutorOptions options,
                      const std::vector<Prompt>& prompts,
                      const std::vector<std::vector<int>>& plain) {
    options.sequences = 2;
    Executor executor(model, device, options);
    std::vector<__nv_bfloat16> logits(model.config().vocab_size);
    int failures          = 0;
    std::uint64_t drafted = 0, accepted = 0;
    double elapsed = 0;
    std::size_t tokens = 0;
    std::vector<std::vector<int>> speculative;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const auto prompt_start = Clock::now();
        executor.reset(0);
        std::vector<std::vector<int>> rows{
            {prefill(executor, model, 0, prompts[i].text, options.prefill_chunk, logits)}};
        const std::uint32_t sequence = 0;
        std::uint64_t prompt_drafted = 0, prompt_accepted = 0;
        decode_speculative(executor, model, std::span(&sequence, 1), rows, int(plain[i].size()),
                           prompt_drafted, prompt_accepted);
        elapsed += seconds(prompt_start, Clock::now());
        drafted += prompt_drafted;
        accepted += prompt_accepted;
        tokens += rows[0].size();
        const bool same = rows[0] == plain[i];
        std::cout << (same ? "SAME " : "DIFF ") << "speculative versus plain, prompt " << i << " ("
                  << prompt_accepted << " of " << prompt_drafted << " drafts accepted): \""
                  << model.resources().tokenizer->decode(rows[0]) << "\"\n";
        const bool fact = model.resources().tokenizer->decode(rows[0]).find(prompts[i].expect) !=
                          std::string::npos;
        speculative.push_back(rows[0]);
        executor.reset(0);
        std::vector<std::vector<int>> repeated{{prefill(executor, model, 0, prompts[i].text,
                                                       options.prefill_chunk, logits)}};
        std::uint64_t ignored = 0, kept = 0;
        decode_speculative(executor, model, std::span(&sequence, 1), repeated, int(plain[i].size()),
                           ignored, kept);
        const bool repeat = repeated[0] == rows[0];
        std::cout << ((repeat && fact) ? "OK   " : "FAIL ")
                  << "speculative repeat and expected fact, prompt " << i << '\n';
        failures += repeat && fact ? 0 : 1;
    }
    std::cout << "     " << executor.draft_tokens() << " drafts a round: "
              << double(accepted) / double(std::max<std::uint64_t>(drafted, 1))
              << " of the drafts accepted, " << tokens / elapsed
              << " tok/s over the prompts (prefill included)\n";
    // Replay the same forced-wrong schedule exactly; restore the normal schedule after that detour.
    {
        std::uint64_t ignored = 0, kept = 0;
        executor.reset(0);
        const int first =
            prefill(executor, model, 0, prompts[0].text, options.prefill_chunk, logits);
        SequenceSnapshot snapshot;
        executor.snapshot(0, snapshot);
        const std::uint32_t sequence = 0;
        std::vector<std::vector<int>> wrong{{first}};
        decode_speculative(executor, model, std::span(&sequence, 1), wrong, int(plain[0].size()),
                           ignored, kept, true);
        std::cout << (wrong[0] == plain[0] ? "SAME " : "DIFF ")
                  << "forced-wrong drafts versus plain\n";
        executor.restore(0, snapshot);
        std::vector<std::vector<int>> wrong_repeat{{first}};
        decode_speculative(executor, model, std::span(&sequence, 1), wrong_repeat,
                           int(plain[0].size()), ignored, kept, true);
        const bool same = wrong_repeat[0] == wrong[0] &&
            model.resources().tokenizer->decode(wrong[0]).find(prompts[0].expect) != std::string::npos;
        std::cout << (same ? "OK   " : "FAIL ") << "forced-wrong repeat and expected fact\n";
        failures += same ? 0 : 1;
        executor.restore(0, snapshot);
        std::vector<std::vector<int>> restored{{first}};
        decode_speculative(executor, model, std::span(&sequence, 1), restored, int(plain[0].size()),
                           ignored, kept);
        const bool again = restored[0] == speculative[0];
        std::cout << (again ? "OK   " : "FAIL ") << "speculative decode from a snapshot\n";
        failures += again ? 0 : 1;
    }
    std::vector<std::vector<int>> rows;
    for (std::uint32_t s = 0; s < 2; ++s) {
        executor.reset(s);
        rows.push_back(
            {prefill(executor, model, s, prompts[s].text, options.prefill_chunk, logits)});
    }
    const std::vector<std::uint32_t> both{0, 1};
    std::array<SequenceSnapshot, 2> snapshots;
    executor.snapshot(0, snapshots[0]);
    executor.snapshot(1, snapshots[1]);
    auto repeated = rows;
    decode_speculative(executor, model, both, rows, int(std::max(plain[0].size(), plain[1].size())),
                       drafted, accepted);
    executor.restore(0, snapshots[0]);
    executor.restore(1, snapshots[1]);
    std::uint64_t ignored = 0, kept = 0;
    decode_speculative(executor, model, both, repeated,
                       int(std::max(plain[0].size(), plain[1].size())), ignored, kept);
    for (std::uint32_t s = 0; s < 2; ++s) {
        const bool repeat = rows[s] == repeated[s];
        rows[s].resize(std::min(rows[s].size(), plain[s].size()));
        const bool same = rows[s] == plain[s];
        std::cout << (same ? "SAME " : "DIFF ") << "batched speculative versus plain, sequence "
                  << s << "\n";
        const bool fact = model.resources().tokenizer->decode(rows[s]).find(prompts[s].expect) !=
                          std::string::npos;
        std::cout << ((repeat && fact) ? "OK   " : "FAIL ")
                  << "batched speculative repeat and expected fact " << s << '\n';
        failures += repeat && fact ? 0 : 1;
    }
    return failures;
}

// The context cache's images: a sequence's live state and the state at an earlier snapshot, saved
// to host memory and loaded into another sequence, must continue as the original did (greedy
// tokens identical to the plain decode), the snapshot's after the original moved on.
int check_images(const Model& model, DeviceContext& device, ExecutorOptions options,
                 const std::vector<Prompt>& prompts, const std::vector<std::vector<int>>& plain) {
    options.sequences = 2;
    Executor executor(model, device, options);
    std::vector<__nv_bfloat16> logits(model.config().vocab_size);
    int failures = 0;
    executor.reset(0);
    const int first = prefill(executor, model, 0, prompts[0].text, options.prefill_chunk, logits);
    SequenceSnapshot snapshot;
    executor.snapshot(0, snapshot);
    const std::uint32_t position = executor.position(0);
    // Pages of an odd size, in reverse order in memory, so copies split inside regions and
    // consecutive slices of an image are not consecutive in memory.
    constexpr std::size_t kPage = 5'000'011;
    const std::size_t count     = (executor.image_bytes(position) + kPage - 1) / kPage;
    std::vector<std::byte> live_bytes(count * kPage), snapshot_bytes(count * kPage);
    const auto pages_of = [&](std::vector<std::byte>& bytes) {
        std::vector<std::byte*> pages(count);
        for (std::size_t i = 0; i < count; ++i) { pages[i] = bytes.data() + (count - 1 - i) * kPage; }
        return pages;
    };
    const std::vector<std::byte*> live_pages = pages_of(live_bytes);
    const std::vector<std::byte*> snapshot_pages = pages_of(snapshot_bytes);
    const SequenceImage live = executor.save_image(0, nullptr, live_pages, kPage);
    // The original moves on; the snapshot's positions stay as they were.
    (void)decode_alone(executor, model, 0, first, 8, logits);
    const SequenceImage earlier = executor.save_image(0, &snapshot, snapshot_pages, kPage);
    const auto decoded = [&](const SequenceImage& image, const std::vector<std::byte*>& pages) {
        executor.reset(1);
        executor.load_image(1, image, pages, kPage);
        return decode_alone(executor, model, 1, first, int(plain[0].size()), logits);
    };
    for (const auto& [label, tokens] : {std::pair{"live", decoded(live, live_pages)},
                                        std::pair{"snapshot", decoded(earlier, snapshot_pages)}}) {
        const bool same = tokens == plain[0];
        std::cout << (same ? "OK   " : "FAIL ") << "sequence image (" << label
                  << ") continues as the original\n";
        failures += same ? 0 : 1;
    }
    std::cout << "     an image of " << position << " positions is "
              << double(executor.image_bytes(position)) / 1e6 << " MB\n";
    return failures;
}

// Full-vocabulary numerical evidence with teacher-forced inputs. Fixed-mode repetition is
// checked here; a separate comparison evaluates resident versus hybrid on identical records.
int record_logits(const Model& model, DeviceContext& device, ExecutorOptions options,
                  const std::filesystem::path& report_path, const std::string& residency) {
    options.max_context = 4096;
    options.sequences = 1;
    options.draft_tokens = 0;
    if (const char* share = std::getenv("NINFER_QWEN4_EXP_DMA_SHARE")) {
        options.hybrid_experts.dma_share = std::stof(share);
    }
    if (residency == "host") { options.hybrid_experts.cpu_threads = 8; }
    const auto domain = model.resources().public_token_count;
    const auto vocab = model.config().vocab_size;
    const auto seed = model.resources().tokenizer->encode(
        "The river reaches the old mill. Calculate 17 * 23. "
        "def square(x): return x * x\nВ библиотеке хранятся карты. 北京是中国的首都。\n");
    if (seed.empty()) { throw std::runtime_error("empty numerical fixture tokens"); }
    const auto payload_path = std::filesystem::path(report_path.string() + ".bf16");
    std::ofstream payload(payload_path, std::ios::binary);
    payload.exceptions(std::ios::badbit | std::ios::failbit);
    nlohmann::json report{{"vocab", vocab}, {"domain", domain}, {"dtype", "bf16-le"},
        {"residency", residency}, {"devices", device_list()},
        {"dma_share", options.hybrid_experts.dma_share ? nlohmann::json(*options.hybrid_experts.dma_share)
                                                       : nlohmann::json("auto")},
        {"prefill_chunk", options.prefill_chunk},
        {"kv", "bf16"}, {"prefixes", nlohmann::json::array()}, {"records", nlohmann::json::array()}};
    Executor executor(model, device, options);
    std::vector<__nv_bfloat16> logits(vocab);
    std::size_t offset = 0;
    for (const int prefix : {128, 2112}) {
        std::vector<std::int32_t> ids(prefix + 15);
        for (std::size_t i = 0; i < ids.size(); ++i) { ids[i] = seed[i % seed.size()]; }
        report["prefixes"].push_back({{"prefix", prefix}, {"tokens", ids}});
        std::vector<std::vector<__nv_bfloat16>> expected;
        for (int repeat = 0; repeat < 2; ++repeat) {
            executor.reset(0);
            for (std::size_t at = 0; at < std::size_t(prefix); at += options.prefill_chunk) {
                executor.forward(0, std::span(ids).subspan(at,
                    std::min<std::size_t>(options.prefill_chunk, prefix - at)), 1);
            }
            for (int step = 0; step < 16; ++step) {
                if (step) { executor.forward(0, std::span(ids).subspan(prefix + step - 1, 1), 1); }
                read_logits(executor, logits);
                for (std::size_t i = 0; i < domain; ++i) {
                    if (!std::isfinite(__bfloat162float(logits[i]))) {
                        throw std::runtime_error("non-finite numerical fixture logits");
                    }
                }
                if (repeat == 0) {
                    expected.push_back(logits);
                    payload.write(reinterpret_cast<const char*>(logits.data()), logits.size() * 2);
                    report["records"].push_back({{"prefix", prefix}, {"step", step}, {"offset", offset}});
                    offset += logits.size() * 2;
                } else if (std::memcmp(logits.data(), expected[step].data(), logits.size() * 2) != 0) {
                    throw std::runtime_error("fixed numerical fixture mode did not repeat");
                }
            }
        }
    }
    payload.close();
    report["payload"] = payload_path.filename().string();
    report["fixed_mode_repeats"] = true;
    std::ofstream out(report_path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << report.dump(2) << '\n';
    std::cout << "LOGITS_REPORT_READY records=" << report["records"].size()
              << " public_vocab=" << domain << " bytes=" << offset << '\n';
    return 0;
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
    const char* drafts_env      = std::getenv("NINFER_QWEN4_EXP_DRAFTS");
    const std::uint32_t drafts =
        drafts_env != nullptr ? static_cast<std::uint32_t>(std::stoul(drafts_env)) : 3;
    load.mtp   = drafts > 0 && reader.directory().components.contains("mtp");
    auto model = load_model(reader, load, device);
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
    const std::string rows_from  = env_or("NINFER_QWEN4_EXP_NGRAM_RESIDENCY", "disk");
    const std::string rows_io    = env_or("NINFER_QWEN4_EXP_NGRAM_IO", "buffered");
    options.ngram_read.residency = rows_from == "ram"       ? NgramResidency::Ram
                                   : rows_from == "ram-hot" ? NgramResidency::RamHot
                                                            : NgramResidency::Disk;
    options.ngram_read.io        = rows_io == "direct" ? NgramIo::Direct
                                   : rows_io == "mmap" ? NgramIo::Mapped
                                                       : NgramIo::Buffered;
    if (options.ngram_read.residency == NgramResidency::RamHot) {
        const auto profile = read_ngram_profile(env_or("NINFER_QWEN4_EXP_NGRAM_PROFILE", ""));
        options.ngram_read.hot_rows     = profile.rows;
        options.ngram_read.budget_bytes = std::uint64_t{4} << 30U;
    }
    std::cout << "n-gram table: " << options.ngram->layout.rows << " rows of "
              << options.ngram->layout.row_bytes << " bytes in "
              << options.ngram->layout.segments.size() << " file segment(s) of "
              << options.ngram->layout.segments.front().path.filename().string() << "\n";
    if (const char* output = std::getenv("NINFER_QWEN4_EXP_LOGITS_REPORT")) {
        return record_logits(*model, device, options, output, residency);
    }
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
    failures += check_images(*model, device, options, prompts, replayed);
    if (load.mtp) {
        ExecutorOptions speculative = options;
        speculative.draft_tokens    = drafts;
        failures += check_speculative(*model, device, speculative, prompts, replayed);
    } else {
        std::cout << "speculative decoding: not checked (no MTP block, or no drafts)\n";
    }
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
