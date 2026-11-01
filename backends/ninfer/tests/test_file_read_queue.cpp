#include "core/file_read_queue.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
template<class Error, class F> void refused(F&& call) {
    try { call(); } catch (const Error&) { return; }
    throw std::runtime_error("file read queue accepted an invalid range");
}
struct Fixture {
    std::array<std::filesystem::path, 2> paths;
    std::vector<std::uint8_t> bytes;
    Fixture() : bytes(128 * 1024 + 53) {
        std::mt19937 random(8173);
        for (auto& byte : bytes) { byte = static_cast<std::uint8_t>(random()); }
        const auto prefix = std::filesystem::current_path() /
            ("ninfer_read_queue_" + std::to_string(std::random_device{}()));
        for (std::size_t i = 0; i < paths.size(); ++i) {
            paths[i] = prefix.string() + "." + std::to_string(i);
            std::ofstream file(paths[i], std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            require(bool(file), "cannot write queue fixture");
        }
    }
    ~Fixture() {
        for (const auto& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }
};

int run() {
    Fixture fixture;
    refused<std::invalid_argument>([&] { (void)ninfer::FileReadQueue::open(fixture.paths, false, 0); });
    for (const bool direct : {false, true}) {
        for (const std::size_t depth : {1U, 5U, 64U}) {
            auto queue = ninfer::FileReadQueue::open(fixture.paths, direct, depth);
            if (!queue) {
                std::cout << "SKIP native file queue: unavailable on this OS or disabled by sandbox\n";
                return 77;
            }
            // More reads than slots, duplicate offsets with separate destinations, boundaries
            // inside a page, one range above the old 64 KiB bounce size, and a short final block.
            constexpr std::size_t count = 151;
            std::vector<std::vector<std::uint8_t>> outputs(count);
            std::vector<ninfer::QueuedFileRead> reads;
            for (std::size_t i = 0; i < count; ++i) {
                const auto offset = i == 0 ? 17U : (i * 997) % (fixture.bytes.size() - 8192);
                outputs[i].resize(i == 0 ? 70003 : 17 + i % 5000);
                reads.push_back({i % 2, offset, outputs[i]});
            }
            std::vector<std::uint8_t> tail(8003);
            reads.push_back({1, fixture.bytes.size() - tail.size(), tail});
            reads.push_back({0, fixture.bytes.size(), {}});
            for (int repeat = 0; repeat < 2; ++repeat) {
                for (auto& output : outputs) { std::fill(output.begin(), output.end(), 0); }
                std::fill(tail.begin(), tail.end(), 0);
                queue->read(reads);
                for (const auto& read : reads) {
                    require(std::equal(read.destination.begin(), read.destination.end(),
                                       fixture.bytes.begin() + read.offset), "queued bytes differ");
                }
            }
            // A failure among outstanding reads is drained before throwing. The same queue
            // must then accept another batch, without completions left over from the failure.
            std::array<std::array<std::uint8_t, 101>, 3> output{};
            const std::array<ninfer::QueuedFileRead, 3> invalid{{
                {0, 17, output[0]}, {1, fixture.bytes.size() - 5, output[1]}, {0, 70000, output[2]}}};
            refused<std::runtime_error>([&] { queue->read(invalid); });
            queue->read(reads);
            for (const auto& read : reads) {
                require(std::equal(read.destination.begin(), read.destination.end(),
                                   fixture.bytes.begin() + read.offset), "failed batch poisoned retry");
            }
            const ninfer::QueuedFileRead bad_file{2, 0, output[0]};
            refused<std::invalid_argument>([&] { queue->read(std::span(&bad_file, 1)); });
            queue->read({});
            std::cout << queue->backend() << " direct=" << direct << " depth=" << depth << " PASS\n";
        }
    }
    return 0;
}
} // namespace

int main() {
    try { return run(); }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
