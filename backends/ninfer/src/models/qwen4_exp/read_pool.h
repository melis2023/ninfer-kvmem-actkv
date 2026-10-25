#pragma once

// Blocking reads of local files reach a drive's bandwidth only with several requests in flight. A
// few threads run one batch of indexed reads at a time together with the calling thread; run()
// returns once every read of the batch has finished and rethrows the first failure.

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

    // read(i) for every i < count, one batch at a time.
    void run(std::size_t count, const std::function<void(std::size_t)>& read);

private:
    void work();
    void drain();

    std::mutex mutex_;
    std::condition_variable ready_, done_;
    const std::function<void(std::size_t)>* job_ = nullptr;
    std::size_t count_                           = 0;
    std::size_t next_                            = 0;
    std::size_t remaining_                       = 0;
    std::uint64_t batch_                         = 0;
    bool stop_                                   = false;
    std::exception_ptr failure_;
    std::vector<std::thread> workers_;
};

} // namespace ninfer::models::qwen4_exp
