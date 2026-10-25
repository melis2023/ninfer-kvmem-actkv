#include "models/qwen3_5/program/structured_round.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include <iostream>
using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) { return 77; }
    try {
        // Reserve a wider copy window than the actual one-token neural round.
        // Compact draft rows must not be read using the maximum reserved pitch.
        constexpr int vocab_size = 129, words = 5, width = 25, lanes = 8;
        DeviceContext device(0);
        DeviceArena arena(words * width * lanes * 4 + 2048);
        auto masks  = arena.alloc(DType::I32, {words, width, lanes});
        auto drafts = arena.alloc(DType::I32, {1, lanes});
        auto wide_drafts = arena.alloc(DType::I32, {width - 1, lanes});
        models::qwen3_5::StructuredRound round(masks, vocab_size, width, lanes);
        // Seed the maximum-width host frame deterministically. A compact copy must replace
        // each row, rather than leaving zero draft tokens in later reserved rows.
        CUDA_CHECK(cudaMemsetAsync(wide_drafts.data, 0, wide_drafts.bytes(), device.stream));
        round.begin_dflash();
        round.enqueue_dflash(wide_drafts, device.stream);
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
        definition.capture(device.stream, [&] { round.enqueue_dflash(drafts, device.stream); });
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        // Changing compact rows and physical lanes must work without recapturing the graph.
        for (int pass = 0; pass < 3; ++pass) {
            round.begin_dflash();
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
            varying_definition.capture(device.stream, [&] { round.enqueue_dflash(frame, device.stream); });
            DecodeGraphExecutable varying_executable;
            varying_executable.instantiate(varying_definition);
            for (int pass = 0; pass < 3; ++pass) {
                round.begin_dflash();
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
        round.begin_dflash();
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
        round.begin_dflash();
        set_drafts('{');
        executable.launch(device.stream);
        device.synchronize();
        round.check(); // reset clears both retained requests and the error
        std::cout << "OK structured graph: lane remap, replay, exception transport, reset\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
