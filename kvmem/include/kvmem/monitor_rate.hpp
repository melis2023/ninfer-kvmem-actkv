#pragma once
#include <nlohmann/json.hpp>
#include <cstdint>

namespace kvmem {
// One request's native output-commit clock, never the browser polling clock.
class monitor_rate_window {
    bool initialized_ = false;
    uint64_t tokens_ = 0, elapsed_ns_ = 0;
    nlohmann::json live_{{"decode_tps_live", nullptr}, {"decode_tps_window_ms", nullptr},
                         {"decode_tps_sampled_at_ms", nullptr}};
public:
    nlohmann::json sample(uint64_t tokens, uint64_t elapsed_ns, int64_t sampled_at_ms) {
        if (!initialized_ || tokens < tokens_ || elapsed_ns < elapsed_ns_) {
            initialized_ = true; tokens_ = tokens; elapsed_ns_ = elapsed_ns;
            live_ = {{"decode_tps_live", nullptr}, {"decode_tps_window_ms", nullptr},
                     {"decode_tps_sampled_at_ms", nullptr}};
        } else if (elapsed_ns - elapsed_ns_ >= 5000000000ULL) {
            const auto duration = elapsed_ns - elapsed_ns_;
            live_ = {{"decode_tps_live", 1e9 * double(tokens - tokens_) / duration},
                     {"decode_tps_window_ms", duration / 1e6},
                     {"decode_tps_sampled_at_ms", sampled_at_ms}};
            tokens_ = tokens; elapsed_ns_ = elapsed_ns;
        }
        return live_;
    }
};
} // namespace kvmem
