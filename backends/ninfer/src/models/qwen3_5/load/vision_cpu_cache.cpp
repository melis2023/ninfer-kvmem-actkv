#include "models/qwen3_5/load/vision_cpu_cache.h"

#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

struct CpuVisionCache::State {
    std::mutex mutex;
    std::condition_variable changed;
    Loader loader;
    std::shared_ptr<const CpuVisionWeights> weights;
    std::chrono::steady_clock::time_point idle_since;
    std::size_t users = 0;
    std::uint64_t loads = 0, unloads = 0;
    double last_load_seconds = 0;
    bool stopping = false;
    bool loading = false;
};

struct CpuVisionCache::Lease {
    std::shared_ptr<State> state;
    std::shared_ptr<const CpuVisionWeights> weights;
    ~Lease() {
        // Drop this lease's immutable owner before waking the reaper, so expiry frees the last
        // weight allocation rather than reporting an unload with a surviving lease-owned copy.
        weights.reset();
        std::lock_guard lock(state->mutex);
        if (--state->users == 0) state->idle_since = std::chrono::steady_clock::now();
        state->changed.notify_all();
    }
};

CpuVisionCache::CpuVisionCache(Loader loader, std::chrono::milliseconds idle_timeout)
    : state_(std::make_shared<State>()) {
    if (!loader || idle_timeout.count() < 0) {
        throw std::invalid_argument("invalid CPU Vision weight cache configuration");
    }
    state_->loader = std::move(loader);
    if (idle_timeout.count() == 0) return; // lazy load, retain until model unload
    reaper_ = std::thread([state = state_, idle_timeout] {
        std::unique_lock lock(state->mutex);
        for (;;) {
            state->changed.wait(lock, [&] {
                return state->stopping || (state->weights && state->users == 0);
            });
            if (state->stopping) return;
            const auto idle_since = state->idle_since;
            if (state->changed.wait_until(lock, idle_since + idle_timeout, [&] {
                    return state->stopping || state->users != 0 || state->idle_since != idle_since;
                })) continue;
            const auto bytes = state->weights->bytes();
            state->weights.reset();
            ++state->unloads;
            std::fprintf(stderr, "CPU_VISION unloaded bytes=%zu idle_ms=%lld\n", bytes,
                         static_cast<long long>(idle_timeout.count()));
        }
    });
}

CpuVisionCache::~CpuVisionCache() {
    {
        std::lock_guard lock(state_->mutex);
        state_->stopping = true;
    }
    state_->changed.notify_all();
    if (reaper_.joinable()) reaper_.join();
    // Any outstanding lease keeps its immutable bytes and State alive independently.
}

std::shared_ptr<const CpuVisionWeights> CpuVisionCache::acquire() {
    std::unique_lock lock(state_->mutex);
    state_->changed.wait(lock, [&] { return !state_->loading; });
    if (!state_->weights) {
        state_->loading = true;
        const auto started = std::chrono::steady_clock::now();
        // One loader runs at a time, outside the metadata lock. Memory/status observers remain
        // cheap while the image worker reads and decodes; other encoders wait for publication.
        lock.unlock();
        std::shared_ptr<const CpuVisionWeights> weights;
        try {
            weights = state_->loader();
            if (!weights) throw std::runtime_error("CPU Vision loader returned no weights");
        } catch (...) {
            lock.lock();
            state_->loading = false;
            state_->changed.notify_all();
            throw;
        }
        lock.lock();
        state_->loading = false;
        state_->last_load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        state_->weights = std::move(weights);
        ++state_->loads;
        state_->idle_since = std::chrono::steady_clock::now();
        state_->changed.notify_all();
        std::fprintf(stderr, "CPU_VISION loaded bytes=%zu seconds=%.3f\n",
                     state_->weights->bytes(), state_->last_load_seconds);
    }
    // Allocate the control block before accounting the lease. Aliasing preserves the lifetime
    // of the lease across arbitrary shared_ptr copies without retaining the Model itself.
    auto lease = std::make_shared<Lease>();
    lease->state = state_;
    lease->weights = state_->weights;
    ++state_->users;
    state_->changed.notify_all();
    return std::shared_ptr<const CpuVisionWeights>(std::move(lease), state_->weights.get());
}

CpuVisionCacheSummary CpuVisionCache::summary() const {
    std::lock_guard lock(state_->mutex);
    return {state_->weights ? state_->weights->bytes() : 0, state_->users,
            state_->loads, state_->unloads, state_->last_load_seconds, state_->loading};
}

} // namespace ninfer::models::qwen3_5
