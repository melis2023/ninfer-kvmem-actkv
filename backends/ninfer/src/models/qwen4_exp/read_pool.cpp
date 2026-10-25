#include "models/qwen4_exp/read_pool.h"

#include <utility>

namespace ninfer::models::qwen4_exp {

ReadPool::ReadPool(std::size_t threads) {
    for (std::size_t i = 0; i < threads; ++i) {
        workers_.emplace_back([this] { work(); });
    }
}

ReadPool::~ReadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    ready_.notify_all();
    for (auto& worker : workers_) { worker.join(); }
}

void ReadPool::run(std::size_t count, const std::function<void(std::size_t)>& read) {
    if (count == 0) { return; }
    {
        std::lock_guard lock(mutex_);
        job_       = &read;
        count_     = count;
        next_      = 0;
        remaining_ = count;
        ++batch_;
    }
    ready_.notify_all();
    drain();
    std::unique_lock lock(mutex_);
    done_.wait(lock, [&] { return remaining_ == 0; });
    job_ = nullptr;
    if (failure_) { std::rethrow_exception(std::exchange(failure_, nullptr)); }
}

// Takes the batch's next index until none is left.
void ReadPool::drain() {
    for (;;) {
        std::size_t index                           = 0;
        const std::function<void(std::size_t)>* job = nullptr;
        {
            std::lock_guard lock(mutex_);
            if (job_ == nullptr || next_ >= count_) { return; }
            index = next_++;
            job   = job_;
        }
        std::exception_ptr failure;
        try {
            (*job)(index);
        } catch (...) { failure = std::current_exception(); }
        std::lock_guard lock(mutex_);
        if (failure && !failure_) { failure_ = failure; }
        if (--remaining_ == 0) { done_.notify_all(); }
    }
}

void ReadPool::work() {
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] {
                return stop_ || (batch_ != seen && job_ != nullptr && next_ < count_);
            });
            if (stop_) { return; }
            seen = batch_;
        }
        drain();
    }
}

} // namespace ninfer::models::qwen4_exp
