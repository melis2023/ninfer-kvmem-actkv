#pragma once

#include "models/qwen3_5/load/vision_cpu.h"

#include <chrono>
#include <functional>
#include <memory>
#include <thread>

namespace ninfer::models::qwen3_5 {

struct CpuVisionCacheSummary {
    std::size_t bytes = 0;
    std::size_t users = 0;
    std::uint64_t loads = 0, unloads = 0;
    double last_load_seconds = 0;
    bool loading = false;
};

// Model-owned immutable weights with independent Host residency. An acquired shared pointer is
// a read lease: its last copy starts the idle timer. The reaper never touches CUDA or live weights.
class CpuVisionCache {
public:
    using Loader = std::function<std::shared_ptr<const CpuVisionWeights>()>;
    CpuVisionCache(Loader loader, std::chrono::milliseconds idle_timeout);
    ~CpuVisionCache();
    CpuVisionCache(const CpuVisionCache&) = delete;
    CpuVisionCache& operator=(const CpuVisionCache&) = delete;

    [[nodiscard]] std::shared_ptr<const CpuVisionWeights> acquire();
    [[nodiscard]] CpuVisionCacheSummary summary() const;

private:
    struct State;
    struct Lease;
    std::shared_ptr<State> state_;
    std::thread reaper_;
};

} // namespace ninfer::models::qwen3_5
