#include "models/qwen3_5/load/vision_cpu_cache.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <iostream>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using ninfer::models::qwen3_5::CpuVisionCache;
using ninfer::models::qwen3_5::CpuVisionWeights;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Predicate> void wait_until(Predicate ready) {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (!ready()) {
        require(std::chrono::steady_clock::now() < deadline, "idle eviction did not complete");
        std::this_thread::sleep_for(2ms);
    }
}
std::shared_ptr<const CpuVisionWeights> weights() {
    auto out = std::make_shared<CpuVisionWeights>();
    out->patch_embedding.assign(1024, 0.25F);
    return out;
}

void lazy_release_reload() {
    std::atomic<unsigned> loads{0};
    std::weak_ptr<const CpuVisionWeights> allocation;
    CpuVisionCache cache([&] { ++loads; auto w = weights(); allocation = w; return w; }, 40ms);
    std::this_thread::sleep_for(80ms);
    require(loads == 0 && cache.summary().bytes == 0, "construction loaded CPU weights");
    auto first = cache.acquire();
    std::weak_ptr<const CpuVisionWeights> weak = first;
    auto copy = first;
    first.reset();
    std::this_thread::sleep_for(80ms);
    require(cache.summary().unloads == 0 && !weak.expired(), "live reader lost its weights");
    require(copy->patch_embedding.front() == 0.25F, "live weights changed");
    copy.reset();
    wait_until([&] { return cache.summary().unloads == 1; });
    require(weak.expired() && allocation.expired() && cache.summary().bytes == 0, "idle eviction retained weight storage");
    auto again = cache.acquire();
    require(loads == 2 && again->patch_embedding.front() == 0.25F, "reload failed");
}

void concurrent_readers() {
    std::atomic<unsigned> loads{0};
    CpuVisionCache cache([&] { ++loads; std::this_thread::sleep_for(15ms); return weights(); }, 30ms);
    std::barrier start(8);
    std::vector<std::shared_ptr<const CpuVisionWeights>> leases(8);
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < leases.size(); ++i) {
        threads.emplace_back([&, i] { start.arrive_and_wait(); leases[i] = cache.acquire(); });
    }
    for (auto& thread : threads) thread.join();
    require(loads == 1 && cache.summary().users == 8, "concurrent first users duplicated weights");
    for (const auto& lease : leases) require(lease.get() == leases.front().get(), "readers differ");
    leases.clear();
    wait_until([&] { return cache.summary().unloads == 1; });
}

void failed_load_retry_and_owner_lifetime() {
    unsigned calls = 0;
    auto cache = std::make_unique<CpuVisionCache>([&] {
        if (++calls == 1) throw std::runtime_error("injected read failure");
        return weights();
    }, 20ms);
    bool failed = false;
    try { (void)cache->acquire(); } catch (const std::runtime_error&) { failed = true; }
    require(failed && cache->summary().bytes == 0 && cache->summary().loads == 0,
            "failed load published partial weights");
    auto lease = cache->acquire();
    cache.reset();
    require(lease->patch_embedding.front() == 0.25F, "owner destruction invalidated a reader");
    lease.reset();
}

void status_during_load() {
    std::atomic<bool> started{false}, proceed{false};
    CpuVisionCache cache([&] {
        started.store(true);
        while (!proceed.load()) std::this_thread::sleep_for(1ms);
        return weights();
    }, 0ms);
    auto loading = std::async(std::launch::async, [&] { return cache.acquire(); });
    wait_until([&] { return started.load(); });
    auto status = std::async(std::launch::async, [&] { return cache.summary(); });
    const bool ready = status.wait_for(1s) == std::future_status::ready;
    proceed.store(true); // always release the blocked loader before reporting a failed check
    const auto snapshot = status.get();
    auto lease = loading.get();
    require(ready && snapshot.loading && snapshot.bytes == 0,
            "status observation waited for the complete weight load");
}

void retention_and_reuse() {
    CpuVisionCache cache(weights, 0ms);
    auto first = cache.acquire();
    const auto* address = first.get();
    first.reset();
    std::this_thread::sleep_for(80ms);
    require(cache.summary().bytes == 4096 && cache.summary().unloads == 0, "zero timeout evicted weights");
    auto second = cache.acquire();
    require(cache.summary().loads == 1 && second.get() == address, "hot reuse loaded weights again");
}
} // namespace

int main() {
    try {
        lazy_release_reload();
        concurrent_readers();
        failed_load_retry_and_owner_lifetime();
        retention_and_reuse();
        status_during_load();
        std::cout << "CPU Vision cache lifetime/concurrency checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
