#pragma once
#include "core/arena.h"
#include <memory>
#include <mutex>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// Only released StateImages may enter the cache. Live checkpoints keep exclusive
// buffer leases; the pool can outlive Program while a lease is still referenced.
class KvmemCheckpointBufferPool : public std::enable_shared_from_this<KvmemCheckpointBufferPool> {
public:
    void configure(std::size_t limit, std::size_t idle_count) {
        std::lock_guard lock(mutex_);
        limit_ = limit; idle_count_ = std::min(idle_count, idle_.size());
    }
    void trim() {
        std::lock_guard lock(mutex_);
        for (auto& idle : idle_) if (idle) { allocated_ -= idle->size(); idle.reset(); }
    }
    std::size_t allocated_bytes() const {
        std::lock_guard lock(mutex_); return allocated_;
    }
    std::shared_ptr<PinnedHostBuffer> acquire(std::size_t bytes) {
        auto pool = shared_from_this();
        std::unique_ptr<PinnedHostBuffer> buffer;
        {
            std::lock_guard lock(mutex_);
            for (auto& idle : idle_) if (idle && idle->size() == bytes) {
                buffer = std::move(idle); break;
            }
        }
        if (!buffer) {
            std::lock_guard lock(mutex_);
            if (allocated_ > limit_ || bytes > limit_ - allocated_)
                throw std::runtime_error("checkpoint pinned Host budget exhausted");
            buffer = std::make_unique<PinnedHostBuffer>(bytes);
            allocated_ += bytes;
        }
        return {buffer.release(), [pool = std::move(pool)](PinnedHostBuffer* value) noexcept {
            std::unique_ptr<PinnedHostBuffer> released(value);
            try {
                std::lock_guard lock(pool->mutex_);
                for (std::size_t i=0; i<pool->idle_count_; ++i) if (!pool->idle_[i]) {
                    pool->idle_[i] = std::move(released); break;
                }
                if (released) pool->allocated_ -= released->size();
            } catch (...) {} // Releasing a checkpoint must remain noexcept.
        }};
    }
    std::size_t idle_bytes() const {
        std::lock_guard lock(mutex_);
        std::size_t bytes = 0;
        for (const auto& idle : idle_) if (idle) bytes += idle->size();
        return bytes;
    }
private:
    mutable std::mutex mutex_;
    std::size_t allocated_ = 0, limit_ = std::numeric_limits<std::size_t>::max(), idle_count_ = 2;
    // A restored anchor can release both its input and endpoint snapshots.
    // Keep that pair; live snapshots can never enter these slots.
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> idle_;
};
} // namespace ninfer::models::qwen3_5::detail
