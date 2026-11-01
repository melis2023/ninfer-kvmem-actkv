#pragma once

// Blocking reads of local files reach a drive's bandwidth only with several requests in flight. A
// few threads run one batch of indexed reads at a time: start() hands the batch to them and
// returns, finish() runs what is left of it on the calling thread too, waits for the rest and
// rethrows the first failure.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ninfer::models::qwen4_exp {

class ReadPool {
public:
    explicit ReadPool(std::size_t threads);
    ~ReadPool();
    ReadPool(const ReadPool&)            = delete;
    ReadPool& operator=(const ReadPool&) = delete;

    // Starts read(i) for every i < count; the batch before must have finished.
    void start(std::size_t count, std::function<void(std::size_t)> read);
    void finish();
    // start() and finish().
    void run(std::size_t count, std::function<void(std::size_t)> read);

private:
    void work();
    void drain();

    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::function<void(std::size_t)> job_;
    std::size_t count_     = 0;
    std::size_t next_      = 0;
    std::size_t remaining_ = 0;
    std::uint64_t batch_   = 0;
    bool stop_             = false;
    std::exception_ptr failure_;
    std::vector<std::thread> workers_;
};

} // namespace ninfer::models::qwen4_exp
