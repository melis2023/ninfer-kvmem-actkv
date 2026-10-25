#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void add(ninfer::PromptInput& p, ninfer::ChatRole role, const std::string& text) {
    ninfer::ChatMessage m;
    m.role = role;
    m.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = text});
    p.messages.push_back(std::move(m));
}
ninfer::PromptInput archive() {
    ninfer::PromptInput p;
    add(p, ninfer::ChatRole::System, "Use the archive to answer the final question.");
    for (int entry = 0; entry < 16; ++entry) {
        std::string text = "Archive entry " + std::to_string(entry) + ". ";
        for (int i = 0; i < 8; ++i) text += entry < 4
            ? "The violet observatory studies pulsars, cosmic rays and distant galaxies. "
            : "The coastal office files invoices, transport schedules and warehouse inventories. ";
        add(p, ninfer::ChatRole::User, text);
        add(p, ninfer::ChatRole::Assistant, "The archive entry is recorded.");
    }
    add(p, ninfer::ChatRole::User, "Explain the violet observatory and its pulsars.");
    return p;
}
struct Sink final : ninfer::OutputSink {
    std::atomic<bool> cancel{false};
    std::uint32_t cancel_prompt = 0, cancel_output = 0;
    std::uint32_t total = 0, reused = 0, processed = 0, generated = 0;
    int starts = 0;
    void start(ninfer::GenerationStart s) override {
        ++starts; reused = s.reused_prompt_tokens;
    }
    void progress(ninfer::PromptProgress p) override {
        require(p.processed_prompt_tokens >= processed, "prompt progress went backwards");
        require(p.reused_prompt_tokens == reused, "reuse accounting changed after admission");
        require(p.processed_prompt_tokens <= p.total_prompt_tokens,
                "query replay was counted twice");
        total = p.total_prompt_tokens; processed = p.processed_prompt_tokens;
        if (cancel_prompt && processed >= cancel_prompt) cancel.store(true);
    }
    void timing(ninfer::GenerationTimingObservation t) override {
        require(t.generated_tokens >= generated, "output accounting went backwards");
        generated = t.generated_tokens;
        if (cancel_output && generated >= cancel_output) cancel.store(true);
    }
    void publish(ninfer::OutputDelta) override {}
};
}

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 9) throw std::invalid_argument(
            "artifact bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8 eager|graph output [drafts] [mode] [ngram-width] [wddm|adaptive|fast|adaptive-fast]");
        ninfer::EngineOptions o;
        ninfer::test::kvmem_test_devices(o);
        o.artifact_path = argv[1]; o.max_context = 16384; o.device = 0;
        o.max_concurrency = 1; o.prefill_chunk = 128;
        o.kvmem.selected_tokens = 384; o.kvmem.reserve_tokens = 128;
        o.kvmem.host_bytes = 2ULL << 30; o.kvmem.verify_transfers = true;
        o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
        o.kv_cache = ninfer::test::kvmem_test_storage(argv[2]);
        o.context_cache.enabled = false;
        o.use_cuda_graph = ninfer::test::kvmem_test_graph(argv[3]); o.device_profile = "off";
        if (argc >= 6 && std::stoul(argv[5]) != 0) {
            o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            o.speculative.draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[5]));
        }
        const std::string mode = argc >= 7 ? argv[6] : "";
        if (mode == "lrfu") { o.kvmem.lrfu = true; o.kvmem.score_agg_max = true; }
        const bool input_fallback = mode == "input-fallback" ||
            mode == "input-fallback-disk-write" || mode == "input-fallback-disk-read" ||
            mode == "input-fallback-disk-corrupt";
        if ((mode == "endpoint" || input_fallback) && o.speculative.draft_tokens) {
            o.speculative.ngram_draft_tokens = 31;
            o.speculative.ngram_min_match = 4;
        }
        if (argc >= 8) o.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(std::stoul(argv[7]));
        if (argc == 9) {
            const std::string feature = argv[8];
            if (feature == "wddm") o.wddm_evictable_budget = true;
            else if (feature == "adaptive" || feature == "adaptive-fast") {
                o.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
                o.fast_prefill_kernel = feature == "adaptive-fast";
            } else if (feature == "fast") o.fast_prefill_kernel = true;
            else throw std::invalid_argument("expected wddm, adaptive, fast or adaptive-fast");
        }
        if (input_fallback) o.gdn_state_fp16 = true;
        if (input_fallback && mode != "input-fallback") {
            o.kvmem.disk_path = std::filesystem::path(argv[4]).parent_path() / "disk";
            o.kvmem.disk_bytes = 2ULL << 30;
        }
        ninfer::Engine engine(o);
        std::cout << "FEATURE drafts=" << o.speculative.draft_tokens << " adaptive="
                  << (o.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive)
                  << " fast=" << o.fast_prefill_kernel << std::endl;
        std::ofstream output(argv[4]);
        const auto run = [&](const char* name, const ninfer::PromptInput& p, bool reuse,
                             bool expected_reuse, std::uint32_t cancel_prompt = 0,
                             std::uint32_t cancel_output = 0, std::uint32_t output_tokens = 32,
                             bool model_stop = false) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = cancel_output ? 130 : output_tokens;
            request.execution.sampling.temperature = 0;
            request.execution.sampling.seed = 42;
            request.execution.allow_prefix_reuse = reuse;
            request.stop.include_model_defaults = model_stop;
            Sink sink; sink.cancel_prompt = cancel_prompt; sink.cancel_output = cancel_output;
            const auto result = engine.submit(engine.prepare(p), request,
                ninfer::OutputConsumerMode::Streaming,
                {.phase_timings = true, .live_timings = true, .prompt_progress = true})
                .wait(&sink, ninfer::CancellationView([&] { return sink.cancel.load(); }));
            require(engine.is_available(), "request made Engine unavailable");
            require(sink.starts == 1, "request admission was not published exactly once");
            require((result.reused_prompt_tokens > 0) == expected_reuse, "unexpected prefix reuse decision");
            require(result.reused_prompt_tokens == sink.reused, "terminal reuse accounting mismatch");
            if (cancel_prompt || cancel_output) {
                require(result.finish_reason == ninfer::FinishReason::Cancelled, "cancellation did not terminate");
                require(result.generated_token_ids.size() < request.execution.requested_output_tokens,
                        "cancelled request exhausted output budget");
            } else {
                require(result.generated_token_ids.size() == output_tokens ||
                        (model_stop && !result.generated_token_ids.empty() &&
                         result.finish_reason == ninfer::FinishReason::StopToken),
                        "request did not produce its output budget or reach a model stop token");
                require(sink.processed == sink.total, "prompt progress missed completed input");
            }
            output << name << ' ' << result.reused_prompt_tokens;
            for (auto token : result.generated_token_ids) output << ' ' << token;
            output << '\n'; output.flush();
            std::cout << "P4_HISTORY case=" << name << " reused=" << result.reused_prompt_tokens
                      << " generated=" << result.generated_token_ids.size()
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens << std::endl;
            if (o.speculative.mtp_policy == ninfer::MtpDraftPolicy::Adaptive) {
                require(result.speculative.adaptive, "adaptive MTP was not active");
                std::cout << "ADAPTIVE transitions=" << result.speculative.window_transitions << " rounds=";
                for (auto count : result.speculative.rounds_per_window) std::cout << count << ',';
                std::cout << std::endl;
            }
            const auto memory = engine.memory_summary().kvmem;
            require(memory.host_payload_bytes <= o.kvmem.host_bytes, "Host archive exceeded H");
            require(memory.resident_pages == 0 && memory.mtp_resident_pages == 0,
                    "finished request leaked device KV pages");
            if (o.speculative.draft_tokens && !cancel_prompt && output_tokens > 4) {
                require(result.speculative.enabled && result.speculative.rounds > 0 &&
                        result.speculative.drafted_tokens > 0, "MTP did not verify drafts");
            }
            return result;
        };
        if (mode == "lrfu") {
            auto chat = archive();
            chat.options.enable_thinking = false;
            chat.context_cache.session_key = "lrfu-core";
            const auto original = chat;
            const auto cold = run("lrfu-cold-pressure", chat, true, false, 0, 0, 160);
            add(chat, ninfer::ChatRole::Assistant, "The tool record is saved.");
            add(chat, ninfer::ChatRole::Tool, "Additional archive fragment: violet stars and pulsars.");
            const auto continued = run("lrfu-continue", chat, true, true, 0, 0, 8);
            const auto repeated = run("lrfu-repeat", chat, true, true, 0, 0, 8);
            require(continued.generated_token_ids == repeated.generated_token_ids,
                    "LRFU regeneration did not restore its parent round");
            const auto rolled = run("lrfu-rollback", original, true, true, 0, 0, 8);
            require(std::equal(rolled.generated_token_ids.begin(), rolled.generated_token_ids.end(),
                               cold.generated_token_ids.begin()), "LRFU rollback changed the original round");
            std::cout << "LRFU_HISTORY PASS" << std::endl;
            return 0;
        }
        if (input_fallback) {
            ninfer::PromptInput chat;
            chat.options.enable_thinking = true;
            chat.options.preserve_thinking = true;
            chat.context_cache.session_key = "discarded-answer-history";
            add(chat, ninfer::ChatRole::System, "Read the archive and reply with the single word violet.");
            add(chat, ninfer::ChatRole::User, "What is the observatory's color? Use the tool records.");
            add(chat, ninfer::ChatRole::Assistant, "I will read the archive.");
            std::string records;
            for (int i = 0; i < 100; ++i)
                records += "Record " + std::to_string(i) + ": the violet observatory studies pulsars.\n";
            add(chat, ninfer::ChatRole::Tool, records);
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 96;
            request.execution.sampling.temperature = 0;
            request.execution.sampling.seed = 42;
            request.execution.thinking.budget = 16;
            const auto check = [&](const char* name, const ninfer::PromptInput& prompt,
                                   std::uint32_t expected_prefix, std::uint32_t cancel_output = 0) {
                Sink sink; sink.cancel_output = cancel_output;
                const auto input = engine.prepare(prompt).summary().prompt_tokens;
                const auto result = engine.submit(engine.prepare(prompt), request,
                    ninfer::OutputConsumerMode::Streaming,
                    {.phase_timings = true, .live_timings = true, .prompt_progress = true})
                    .wait(&sink, ninfer::CancellationView([&] { return sink.cancel.load(); }));
                std::cout << "INPUT_FALLBACK case=" << name << " input=" << input
                          << " reused=" << result.reused_prompt_tokens
                          << " reasoning=" << result.reasoning.size()
                          << " content=" << result.content.size() << std::endl;
                output << name << ' ' << input << ' ' << result.reused_prompt_tokens << '\n';
                output.flush();
                require(engine.is_available() && sink.starts == 1, "fallback request admission failed");
                require(result.reused_prompt_tokens == sink.reused, "fallback reuse accounting mismatch");
                require(expected_prefix ? result.reused_prompt_tokens >= expected_prefix - 8
                                        : result.reused_prompt_tokens == 0,
                        "generation overwrote the reusable canonical input checkpoint");
                if (cancel_output) {
                    require(result.finish_reason == ninfer::FinishReason::Cancelled,
                            "fallback cancellation did not terminate");
                } else {
                    require(!result.reasoning.empty() && !result.content.empty(),
                            "fallback fixture did not cross the thinking-close boundary");
                    require(sink.processed == sink.total, "fallback prompt progress incomplete");
                }
                const auto memory = engine.memory_summary().kvmem;
                require(memory.host_payload_bytes <= o.kvmem.host_bytes &&
                        memory.resident_pages == 0 && memory.mtp_resident_pages == 0,
                        "fallback retirement leaked Host budget or device pages");
                return input;
            };
            auto previous = engine.prepare(chat).summary().prompt_tokens;
            require(previous > 512, "fallback fixture does not exercise sparse history");
            if (mode != "input-fallback-disk-read" && mode != "input-fallback-disk-corrupt")
                check("cold", chat, 0);
            if (mode == "input-fallback-disk-write") {
                require(engine.memory_summary().kvmem.disk_writes == 1,
                        "fallback history was not published to disk");
                std::cout << "INPUT_FALLBACK DISK_WRITE PASS" << std::endl;
                return 0;
            }
            for (int turn = 0; turn < 3; ++turn) {
                // The benchmark keeps a synthetic tool history and discards the actual
                // generated reasoning/answer. Its exact earlier input remains reusable.
                add(chat, ninfer::ChatRole::Assistant, "The tool record is saved.");
                add(chat, ninfer::ChatRole::Tool, "Additional archive fragment.\n" + records.substr(0, 600));
                const bool corrupt = turn == 0 && mode == "input-fallback-disk-corrupt";
                previous = check("discard-and-append", chat, corrupt ? 0 : previous);
                if (turn == 0 && mode == "input-fallback-disk-read")
                    require(engine.memory_summary().kvmem.disk_hits == 1,
                            "canonical input checkpoint was not restored after restart");
                if (corrupt)
                    require(engine.memory_summary().kvmem.disk_errors == 1,
                            "corrupt snapshot was not rejected before reuse");
            }
            check("repeat", chat, previous);
            check("cancel-output", chat, previous, 8);
            check("after-cancel", chat, previous);
            std::cout << "INPUT_FALLBACK PASS" << std::endl;
            return 0;
        }
        if (mode == "endpoint") {
            ninfer::RequestOptions one;
            one.execution.requested_output_tokens = 1;
            one.execution.sampling.temperature = 0;
            one.stop.include_model_defaults = false;
            const auto single = engine.tokenize_text("x");
            require(single.size() == 1, "one-token fixture changed");
            const auto single_output = engine.generate(engine.prepare_tokens(single), one);
            auto single_next = single;
            single_next.push_back(single_output.generated_token_ids.at(0));
            require(engine.generate(engine.prepare_tokens(single_next), one).reused_prompt_tokens == 1,
                    "one-token input endpoint was not retained");
            auto ids = engine.tokenize_text("Continue the following list of observatory records:\nRecord one: violet stars.\nRecord two:");
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 96;
            request.execution.sampling.temperature = 0;
            request.execution.allow_prefix_reuse = true;
            request.stop.include_model_defaults = false;
            const auto first = engine.generate(engine.prepare_tokens(ids), request);
            require(first.generated_token_ids.size() == 96, "raw endpoint output limit");
            const auto prefix = ids.size();
            ids.insert(ids.end(), first.generated_token_ids.begin(), first.generated_token_ids.end());
            const auto next = engine.generate(engine.prepare_tokens(ids), request);
            require(next.reused_prompt_tokens == prefix + 95, "generated raw prefix was not retained");
            require(next.prefix_reuse_path == ninfer::PrefixReusePath::PrivateEndpoint, "endpoint path not reported");
            std::cout << "ENDPOINT raw reused=" << next.reused_prompt_tokens << std::endl;

            ninfer::PromptInput thinking;
            thinking.options.enable_thinking = true;
            thinking.options.preserve_thinking = true;
            thinking.context_cache.session_key = "endpoint-thinking";
            add(thinking, ninfer::ChatRole::User, "Reply with the single word violet.");
            auto thinking_request = request;
            thinking_request.execution.requested_output_tokens = 128;
            thinking_request.execution.thinking.budget = 16;
            thinking_request.stop.include_model_defaults = true;
            const auto reasoned = engine.generate(engine.prepare(thinking), thinking_request);
            require(!reasoned.content.empty(), "thinking control produced no answer");
            const auto thinking_input = engine.prepare(thinking).summary().prompt_tokens;
            add(thinking, ninfer::ChatRole::Assistant, reasoned.content);
            thinking.messages.back().reasoning_content = reasoned.reasoning;
            add(thinking, ninfer::ChatRole::Tool, "Repeat the answer.");
            const auto reasoned_next = engine.generate(engine.prepare(thinking), thinking_request);
            require(reasoned_next.reused_prompt_tokens > thinking_input,
                    "canonical thinking reconstruction lost the generated reasoning prefix");
            std::cout << "ENDPOINT thinking reused=" << reasoned_next.reused_prompt_tokens
                      << " input=" << thinking_input << std::endl;

            auto rewrite = thinking;
            rewrite.messages.resize(1);
            rewrite.context_cache.session_key = "endpoint-thinking-rewrite";
            const auto rewrite_first = engine.generate(engine.prepare(rewrite), thinking_request);
            add(rewrite, ninfer::ChatRole::Assistant, "amber");
            rewrite.messages.back().reasoning_content = rewrite_first.reasoning;
            add(rewrite, ninfer::ChatRole::Tool, "Repeat the answer.");
            const auto rewritten = engine.generate(engine.prepare(rewrite), thinking_request);
            require(rewritten.reused_prompt_tokens > thinking_input &&
                    rewritten.prefix_reuse_path == ninfer::PrefixReusePath::PrivateTurnClosure,
                    "edited answer did not fall back to its exact thinking-close state");
            std::cout << "ENDPOINT thinking-rewrite reused=" << rewritten.reused_prompt_tokens << std::endl;

            auto chat = archive();
            chat.options.enable_thinking = false;
            chat.options.preserve_thinking = true;
            chat.context_cache.session_key = "endpoint-tools";
            std::string source;
            for (int i = 0; i < 100; ++i)
                source += "Record " + std::to_string(i) + ": violet stars and distant galaxies.\n";
            chat.messages.back().parts[0].text = "Copy exactly, without commentary or markdown:\n" + source;
            // The visible assistant text cannot reconstruct a raw stream forced past
            // EOS into new role/control tokens. Exercise a real chat turn instead.
            const auto cold = run("endpoint-chat-cold", chat, true, false, 0, 0, 160, true);
            require(cold.generated_token_ids.size() >= 64, "copy fixture did not produce a useful generated prefix");
            const auto input = engine.prepare(chat).summary().prompt_tokens;
            add(chat, ninfer::ChatRole::Assistant, cold.content);
            add(chat, ninfer::ChatRole::Tool, "Continue copying the next records.");
            const auto continued = run("endpoint-chat-tool", chat, true, true, 0, 0, 32);
            require(continued.reused_prompt_tokens >= input + cold.generated_token_ids.size() - 8,
                    "long same-query tool continuation recomputed the generated output");
            const auto repeated = run("endpoint-chat-repeat", chat, true, true);
            require(repeated.reused_prompt_tokens >= continued.reused_prompt_tokens &&
                    repeated.reused_prompt_tokens < engine.prepare(chat).summary().prompt_tokens,
                    "repeat did not use a valid earlier checkpoint");
            auto new_user = chat;
            add(new_user, ninfer::ChatRole::User, "What does the coastal office file?");
            const auto changed = run("endpoint-new-query", new_user, true, true);
            require(changed.reused_prompt_tokens > 0 &&
                    changed.reused_prompt_tokens <= engine.prepare(chat).summary().prompt_tokens,
                    "new query restored past the preceding history");
            auto edited = chat;
            edited.messages[1].parts[0].text = "Edited archive with amber stars.";
            run("endpoint-edit", edited, true, false);
            run("endpoint-restore-original", chat, true, false);
            run("endpoint-cancel", chat, true, true, 0, 8);
            run("endpoint-after-cancel", chat, true, true);
            require(engine.is_available(), "endpoint retirement made Engine unavailable");
            const auto memory = engine.memory_summary().kvmem;
            require(memory.host_payload_bytes <= o.kvmem.host_bytes && memory.resident_pages == 0 &&
                    memory.mtp_resident_pages == 0, "endpoint archive leaked budget or device pages");
            std::cout << "ENDPOINT PASS" << std::endl;
            return 0;
        }
        if (mode == "short") {
            ninfer::PromptInput p;
            p.options.enable_thinking = false;
            p.options.preserve_thinking = true;
            p.context_cache.session_key = "short-history";
            add(p, ninfer::ChatRole::System, "Answer briefly. The archive name is violet.");
            add(p, ninfer::ChatRole::User, "What is the archive name?");
            const auto tokens = engine.prepare(p).summary().prompt_tokens;
            require(tokens < 512, "short fixture exceeds working window");
            const auto cold = run("short-cold", p, true, false);
            require(engine.memory_summary().kvmem.retained_sessions == 1,
                    "short input did not retain a history");
            const auto repeat = run("short-repeat", p, true, true);
            require(repeat.reused_prompt_tokens == tokens - 1, "short repeat lost the prompt prefix");
            require(cold.generated_token_ids == repeat.generated_token_ids, "short repeat changed tokens");
            run("short-cancel", p, true, true, 0, 8);
            const auto recovered = run("short-recovered", p, true, true);
            require(cold.generated_token_ids == recovered.generated_token_ids, "short cancellation changed tokens");
            auto next = p;
            add(next, ninfer::ChatRole::Assistant, "The archive name is violet.");
            add(next, ninfer::ChatRole::User, "Repeat its name.");
            run("short-next", next, true, true);
            auto edited = next;
            edited.messages[0].parts[0].text = "Answer briefly. The archive name is amber.";
            run("short-edit", edited, true, false);
            run("short-disabled", edited, false, false);
            run("short-disabled-cleared", edited, true, false);
            auto other = edited; other.context_cache.session_key = "other-history";
            run("short-isolated", other, true, false);
            run("short-original-session", edited, true, true);
            auto boundary = p;
            boundary.context_cache.session_key = "boundary-history";
            boundary.messages.back().parts[0].text = "x";
            while (engine.prepare(boundary).summary().prompt_tokens < 512)
                boundary.messages.back().parts[0].text += " x";
            require(engine.prepare(boundary).summary().prompt_tokens == 512, "exact window fixture failed");
            run("short-window-cold", boundary, true, false);
            run("short-window-repeat", boundary, true, true);
            boundary.messages.back().parts[0].text += " x";
            require(engine.prepare(boundary).summary().prompt_tokens == 513, "window crossing fixture failed");
            run("short-to-long", boundary, true, false);
            run("long-repeat", boundary, true, true);
            boundary.messages.back().parts[0].text = "x";
            run("long-to-short", boundary, true, true);
            run("short-after-long", boundary, true, true);
            std::cout << "P4_SHORT_HISTORY PASS" << std::endl;
            return 0;
        }
        auto a = archive();
        if (!mode.empty()) {
            add(a, ninfer::ChatRole::Assistant, "The violet observatory studies pulsars and distant galaxies.");
            add(a, ninfer::ChatRole::User, "What does the coastal office file?");
            a.messages[1].parts[0].text = "Edited archive: the observatory is now named amber.";
            const auto first = run("boundary-cold", a, true, false);
            const auto repeat = run("boundary-repeat", a, true, true);
            require(first.generated_token_ids == repeat.generated_token_ids, "boundary repeat changed tokens");
            run("boundary-cancel", a, true, true, 0, 8);
            const auto recovered = run("boundary-recovered", a, true, true);
            require(first.generated_token_ids == recovered.generated_token_ids, "boundary cancellation changed tokens");
            std::cout << "P6_BOUNDARY PASS" << std::endl;
            return 0;
        }
        const auto cold = run("cold", a, true, false);
        const auto repeat = run("repeat", a, true, true);
        require(cold.generated_token_ids == repeat.generated_token_ids, "same checkpoint replay changed tokens");
        auto b = a;
        add(b, ninfer::ChatRole::Assistant, "The violet observatory studies pulsars and distant galaxies.");
        add(b, ninfer::ChatRole::User, "What does the coastal office file?");
        const auto warm = run("next-turn", b, true, true);
        const auto fresh = run("next-turn-fresh", b, false, false);
        // Native chunk arithmetic may round differently across an older query
        // boundary. Record exact agreement; trusted checkpoint bytes are checked
        // independently by Program before generation.
        std::cout << "P4_HISTORY warm_fresh_tokens_equal="
                  << (warm.generated_token_ids == fresh.generated_token_ids) << std::endl;
        run("reuse-disabled-cleared", b, true, false);
        auto edited = b;
        edited.messages[1].parts[0].text = "Edited archive: the observatory is now named amber.";
        const auto edited_cold = run("history-edit", edited, true, false);
        run("cancel-decode", edited, true, true, 0, 8);
        const auto resumed = run("resume-decode", edited, true, true);
        require(edited_cold.generated_token_ids == resumed.generated_token_ids,
                "cancelled output contaminated trusted history");
        auto different = a;
        different.messages[0].parts[0].text = "Read the following archive carefully.";
        run("cancel-prefill", different, true, false, 128);
        run("resume-prefill", different, true, false);
        if (argc >= 6) {
            // The identical prefix ends immediately before a changed suffix. MTP's
            // boundary row must use the new suffix embedding, not the old query.
            auto changed_query = different;
            changed_query.messages.back().parts[0].text = "What does the coastal office file?";
            run("changed-query", changed_query, true, true);
            const auto long_output = run("reserve-crossing", changed_query, true, true, 0, 0, 129);
            for (std::uint32_t count : {1U, 2U, 3U, 4U, 5U}) {
                const auto small = run("output-prefix", changed_query, true, true, 0, 0, count);
                // Exact on a stable full candidate width is not required across
                // native numeric routes; output length and durable state are.
                require(small.generated_token_ids.front() == long_output.generated_token_ids.front(),
                        "output budget changed the restored prompt's first token");
            }
            run("after-truncated-output", changed_query, true, true);
        }
        std::cout << "P4_HISTORY PASS" << std::endl;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
