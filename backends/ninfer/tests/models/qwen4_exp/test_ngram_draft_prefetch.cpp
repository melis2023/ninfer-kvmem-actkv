// Captured per-step hints consume replay-time tokens, report host failures after the GPU join,
// and recover on the next window without changing committed contexts or demand-read counters.
#include "models/qwen4_exp/ngram_draft_prefetch.h"

#include "core/decode_graph.h"
#include "cuda_availability.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void run() {
    DeviceContext device;
    constexpr std::int32_t eos = 63;
    const auto hash = derive_ngram_hash_constants({
        .vocab_size = 64, .ngram_size = 3, .heads_per_ngram = 8,
        .vocab_base = 47, .divisible_by = 128, .seed = 1234});
    const auto path = std::filesystem::current_path() /
        ("ninfer_draft_prefetch_" + std::to_string(std::random_device{}()) + ".bin");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{path};
    std::vector<char> bytes(hash.rows * 162);
    {
        std::ofstream file(path, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        require(bool(file), "cannot write draft prefetch fixture");
    }
    for (const auto io : {NgramIo::Buffered, NgramIo::Mapped, NgramIo::Direct}) {
        NgramTableReader table({{{path, 0, bytes.size()}}, 162, hash.rows}, {.io = io});
        for (const std::size_t batch : {1U, 8U}) {
            constexpr std::size_t steps = 15;
            // Also exercise a partial batch in a component allocated for eight sequences.
            NgramDraftPrefetch prefetch(device.rank(0), table, hash, eos, 64, 8, steps);
            DeviceBuffer ids(2 * batch * steps * sizeof(std::int32_t));
            std::vector<std::int32_t> tokens(2 * batch * steps);
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                tokens[i] = i % 5 == 0 ? eos : static_cast<std::int32_t>(i % 63);
            }
            std::vector<NgramContext> contexts(batch, NgramContext::sequence_start(hash, eos));
            for (std::size_t i = 0; i < batch; ++i) {
                contexts[i].previous[0] = static_cast<std::int32_t>(i);
            }
            const auto original = contexts;
            std::vector<std::int32_t> anchors(batch, 3);
            const auto stream = device.rank(0).stream;
            const auto stage = [&] {
                CUDA_CHECK(cudaMemcpyAsync(ids.p, tokens.data(), ids.bytes,
                                           cudaMemcpyHostToDevice, stream));
                prefetch.begin(contexts, anchors);
            };
            const auto chain = [&] {
                for (std::size_t i = 0; i < steps; ++i) {
                    prefetch.enqueue(i, static_cast<const std::int32_t*>(ids.p) + i * batch,
                        static_cast<const std::int32_t*>(ids.p) + (steps + i) * batch, stream);
                }
                prefetch.join(stream);
            };
            stage();
            chain();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            prefetch.check();

            DecodeGraphDefinition definition;
            definition.capture(stream, chain);
            DecodeGraphExecutable graph;
            graph.instantiate(definition);
            // Every step must reach the host, including after the same graph saw a failed hint.
            for (std::size_t bad = 0; bad <= 2 * steps; ++bad) {
                const bool invalid = bad < 2 * steps;
                const std::size_t at = (invalid ? bad : 0) * batch + batch - 1;
                const auto saved = tokens[at];
                if (invalid) { tokens[at] = 64; }
                stage();
                graph.launch(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                bool rejected = false;
                try { prefetch.check(); } catch (const std::invalid_argument&) { rejected = true; }
                require(rejected == invalid, "captured hint did not consume this replay's tokens");
                tokens[at] = saved;
            }
            // Shrinking and growing in one allocation exercises the row and candidate strides.
            for (const std::size_t width : std::array<std::size_t, 4>{steps + 1, 2, 5, steps + 1}) {
            const auto count = batch * width;
            DeviceBuffer predictions(2 * count * sizeof(std::int32_t));
            std::vector<std::int32_t> candidates(2 * count), prefix(count);
            for (std::size_t i = 0; i < count; ++i) {
                prefix[i] = i % 3 == 0 ? eos : static_cast<std::int32_t>((7 * i) % 63);
                candidates[i] = static_cast<std::int32_t>((11 * i) % 63);
                candidates[count + i] = i % 5 == 0 ? eos : static_cast<std::int32_t>((13 * i) % 63);
            }
            const auto stage_verify = [&] {
                CUDA_CHECK(cudaMemcpyAsync(predictions.p, candidates.data(), predictions.bytes,
                                           cudaMemcpyHostToDevice, stream));
                prefetch.begin_verification(contexts, prefix);
            };
            const auto verify_chain = [&] {
                const auto* first = static_cast<const std::int32_t*>(predictions.p);
                prefetch.enqueue_verification(first, first + count, stream);
                prefetch.join(stream);
            };
            stage_verify();
            verify_chain();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            prefetch.check();
            DecodeGraphDefinition verify_definition;
            verify_definition.capture(stream, verify_chain);
            DecodeGraphExecutable verify_graph;
            verify_graph.instantiate(verify_definition);
            // Every column's prefix and both predictions must be consumed, including EOS and
            // the last sequence; recovery must replace all callback operands on graph replay.
            for (std::size_t bad = 0; bad <= 3 * width; ++bad) {
                const bool invalid = bad < 3 * width;
                const bool bad_prefix = invalid && bad >= 2 * width;
                const auto column = invalid ? bad % width : 0;
                const auto at = (batch - 1) * width + column;
                auto& value = bad_prefix ? prefix[at] :
                    candidates[at + (invalid && bad >= width ? count : 0)];
                const auto saved = value;
                if (invalid) { value = 64; }
                stage_verify();
                verify_graph.launch(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                bool rejected = false;
                try { prefetch.check(); } catch (const std::invalid_argument&) { rejected = true; }
                require(rejected == invalid, "verification hint did not consume this replay's inputs");
                value = saved;
            }
            }
            // Return to a draft window after the verification layout used the same slots.
            stage();
            graph.launch(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            prefetch.check();
            for (std::size_t i = 0; i < batch; ++i) {
                require(contexts[i].previous == original[i].previous,
                        "speculative hint changed the committed context");
            }
            require(table.counters().rows == 0 && table.counters().batches == 0,
                    "hints changed demand-read counters");
        }
    }
}
} // namespace

int main() {
    try {
        int count = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (ninfer::test::cuda_unavailable(result) || (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        run();
        std::cout << "NGRAM_DRAFT_PREFETCH_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
