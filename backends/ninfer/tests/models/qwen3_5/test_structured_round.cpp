#include "models/qwen3_5/program/structured_round.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/request.h"
#include "ops/op_tester.h"
#include <array>
#include <iostream>
using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) { return 77; }
    const char* phase = "DFlash rounds";
    try {
        // Reserve a wider copy window than the actual one-token neural round.
        // Compact draft rows must not be read using the maximum reserved pitch.
        constexpr int vocab_size = 129, words = 5, width = 25, lanes = 8;
        DeviceContext device(0);
        DeviceArena arena(words * width * lanes * 4 + 2048);
        auto masks  = arena.alloc(DType::I32, {words, width, lanes});
        auto drafts = arena.alloc(DType::I32, {1, lanes});
        auto wide_drafts = arena.alloc(DType::I32, {width - 1, lanes});
        models::qwen3_5::StructuredRound round(masks, vocab_size, width, lanes, true);
        // Seed the maximum-width host frame deterministically. A compact copy must replace
        // each row, rather than leaving zero draft tokens in later reserved rows.
        CUDA_CHECK(cudaMemsetAsync(wide_drafts.data, 0, wide_drafts.bytes(), device.stream));
        round.begin();
        round.enqueue_dflash(wide_drafts, device.stream);
        round.wait(device.stream);
        device.synchronize();
        round.check();
        std::vector<std::string> vocab(vocab_size);
        for (int i = 0; i < 128; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
        text::StructuredCompiler compiler(vocab, {128});
        auto grammar    = compiler.compile({StructuredOutputKind::JsonObject, {}});
        auto set_drafts = [&](int token) {
            std::vector<int> ids(lanes, token);
            CUDA_CHECK(cudaMemcpyAsync(drafts.data, ids.data(), ids.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, device.stream));
            device.synchronize();
        };
        DecodeGraphDefinition definition;
        definition.capture(device.stream, [&] {
            round.enqueue_dflash(drafts, device.stream);
            round.wait(device.stream);
        });
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        // Changing compact rows and physical lanes must work without recapturing the graph.
        for (int pass = 0; pass < 3; ++pass) {
            round.begin();
            round.set_dflash_row(pass, 7 - pass, 1, grammar);
            set_drafts('{');
            executable.launch(device.stream);
            device.synchronize();
            round.check();
            const auto bits =
                from_device<std::uint32_t>(round.device_mask(7 - pass), words * width);
            std::vector<std::uint32_t> expected(words * 2);
            const TokenId expected_draft = '{';
            grammar->fill_masks(expected, {&expected_draft, 1});
            if (!std::equal(expected.begin(), expected.end(), bits.begin())) {
                throw std::runtime_error("compact draft row used reserved-width stride");
            }
            if (!(bits['{' / 32] & (1U << ('{' % 32))) || (bits[4] & 1U) ||
                (bits[words + 4] & 1U)) {
                std::cerr << "pass=" << pass << " bits=";
                for (auto b : bits) { std::cerr << std::hex << b << " "; }
                throw std::runtime_error("graph mask/EOS mismatch");
            }
        }
        // Alternate neural and maximum copy widths. Both graphs share the staging buffers,
        // but must retain their own transfer geometry across replay and lane remapping.
        for (const int draft_width : {7, width - 1}) {
            Tensor frame(wide_drafts.data, DType::I32, {draft_width, lanes});
            std::vector<TokenId> row_tokens(draft_width, ' ');
            const std::string prefix = "{\"a\":1}";
            std::copy(prefix.begin(), prefix.end(), row_tokens.begin());
            std::vector<TokenId> frame_tokens;
            for (int row = 0; row < lanes; ++row) {
                frame_tokens.insert(frame_tokens.end(), row_tokens.begin(), row_tokens.end());
            }
            CUDA_CHECK(cudaMemcpyAsync(frame.data, frame_tokens.data(), frame.bytes(),
                                       cudaMemcpyHostToDevice, device.stream));
            device.synchronize();
            DecodeGraphDefinition varying_definition;
            varying_definition.capture(device.stream, [&] {
                round.enqueue_dflash(frame, device.stream);
                round.wait(device.stream);
            });
            DecodeGraphExecutable varying_executable;
            varying_executable.instantiate(varying_definition);
            for (int pass = 0; pass < 3; ++pass) {
                round.begin();
                round.set_dflash_row(2 - pass, 5 + pass, draft_width, grammar);
                varying_executable.launch(device.stream);
                device.synchronize();
                round.check();
                const auto bits = from_device<std::uint32_t>(round.device_mask(5 + pass), words * width);
                std::vector<std::uint32_t> expected(words * (draft_width + 1));
                grammar->fill_masks(expected, row_tokens);
                if (!std::equal(expected.begin(), expected.end(), bits.begin())) {
                    throw std::runtime_error("variable-width draft graph mask mismatch");
                }
            }
        }
        round.begin();
        auto dead_vocab = std::vector<std::string>(vocab_size, "q");
        dead_vocab.back().clear();
        text::StructuredCompiler dead_compiler(dead_vocab, {128});
        auto dead_grammar = dead_compiler.compile({StructuredOutputKind::JsonObject, {}});
        round.set_dflash_row(0, 0, 1, dead_grammar);
        set_drafts(0); // no vocabulary token can start JSON: matcher must surface the dead end
        executable.launch(device.stream);
        device.synchronize();
        bool raised = false;
        try {
            round.check();
        } catch (const std::exception&) { raised = true; }
        if (!raised) { throw std::runtime_error("host callback exception was not surfaced"); }
        round.begin();
        set_drafts('{');
        executable.launch(device.stream);
        device.synchronize();
        round.check(); // reset clears both retained requests and the error

        // An ordinary or MTP round: the masks start on the host before the launch, and the
        // captured sampling point waits for their upload through an external event node.
        phase = "host-started rounds";
        DeviceArena host_arena(words * width * lanes * 4 * 2 + 2048);
        auto host_masks = host_arena.alloc(DType::I32, {words, width, lanes});
        auto observed   = host_arena.alloc(DType::I32, {words, width});
        models::qwen3_5::StructuredRound host_round(host_masks, vocab_size, width, lanes, false);
        const auto observe_lane = [&](std::uint32_t lane) {
            host_round.wait(device.stream);
            CUDA_CHECK(cudaMemcpyAsync(observed.data, host_round.device_mask(lane),
                                       observed.bytes(), cudaMemcpyDeviceToDevice, device.stream));
        };
        // The second lane's graph replaces the first by an executable update, as Program switches
        // graph profiles.
        std::array<DecodeGraphDefinition, 2> host_definitions;
        DecodeGraphExecutable host_executable;
        for (const std::uint32_t lane : {3U, 6U}) {
            DecodeGraphDefinition& host_definition = host_definitions[lane == 3U ? 0 : 1];
            host_definition.capture(device.stream, [&] { observe_lane(lane); });
            if (lane == 3U) {
                host_executable.instantiate(host_definition);
            } else {
                host_executable.update(host_definition);
            }
            const std::string tokens = "{\"a\":1}";
            for (std::size_t extent = 0; extent < 4; ++extent) {
                CUDA_CHECK(cudaMemsetAsync(host_masks.data, 0, host_masks.bytes(), device.stream));
                CUDA_CHECK(cudaMemsetAsync(observed.data, 0, observed.bytes(), device.stream));
                device.synchronize();
                std::vector<TokenId> lane_drafts(tokens.begin(), tokens.begin() + extent);
                host_round.begin();
                host_round.start(lane, *grammar, lane_drafts);
                host_round.publish();
                host_executable.launch(device.stream);
                device.synchronize();
                host_round.check();
                const auto bits = from_device<std::uint32_t>(observed.data, words * (extent + 1));
                std::vector<std::uint32_t> expected(words * (extent + 1));
                grammar->fill_masks(expected, lane_drafts);
                if (!std::equal(expected.begin(), expected.end(), bits.begin())) {
                    throw std::runtime_error("host-started masks were not uploaded before the "
                                             "captured sampling point");
                }
                if (!(host_round.seconds(lane) > 0.0)) {
                    throw std::runtime_error("host-started mask job time was not kept");
                }
            }
        }
        phase = "host-started round without a structured lane";
        // No structured lane: the published event is complete and the replay does not wait.
        host_round.begin();
        host_round.publish();
        {
            DecodeGraphDefinition idle_definition;
            idle_definition.capture(device.stream, [&] { observe_lane(0); });
            DecodeGraphExecutable idle_executable;
            idle_executable.instantiate(idle_definition);
            idle_executable.launch(device.stream);
            device.synchronize();
        }
        host_round.check();
        phase = "host-started mask error";
        host_round.begin();
        host_round.start(1, *dead_grammar, {});
        host_round.publish();
        raised = false;
        try {
            // The upload stream has collected the job once the event completes.
            host_round.wait(device.stream);
            device.synchronize();
            host_round.check();
        } catch (const std::exception&) { raised = true; }
        if (!raised) { throw std::runtime_error("host-started mask error was not surfaced"); }
        std::cout << "OK structured graph: lane remap, replay, exception transport, reset, "
                     "host-started masks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << phase << ": " << e.what() << '\n';
        return 1;
    }
}
