#pragma once

// Only included by the explicitly enabled Query lifecycle test build. No
// request fields or production scheduling policy depend on this rendezvous.
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace ninfer::models::qwen3_5::detail {
inline void query_test_hash(std::uint64_t& hash, float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash ^= (bits >> shift) & 255U;
        hash *= 1099511628211ULL;
    }
}

inline void query_test_boundary(std::uint32_t lane, std::uint32_t accepted,
                                std::uint32_t total, bool replayed) {
    if (!accepted || accepted >= total || replayed) return;
    const char* directory = std::getenv("NINFER_QUERY_TEST_GATE");
    if (!directory || !*directory) return;
    const std::filesystem::path root(directory);
    if (!std::filesystem::exists(root / "arm") || std::filesystem::exists(root / "ready")) return;
    // The HTTP test queues another real request or disconnects the current
    // client here, then releases the worker to its normal scheduling boundary.
    {
        std::ofstream ready(root / "ready.tmp");
        ready << lane << ' ' << accepted << ' ' << total << '\n';
        if (!ready) throw std::runtime_error("Query test gate could not publish readiness");
    }
    std::filesystem::rename(root / "ready.tmp", root / "ready");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!std::filesystem::exists(root / "release")) {
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Query lifecycle test gate timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
} // namespace ninfer::models::qwen3_5::detail
