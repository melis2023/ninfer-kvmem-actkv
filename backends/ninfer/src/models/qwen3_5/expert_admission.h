#pragma once

#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5 {

// With streamed experts the device cache has priority over automatic KV: automatic KV takes the
// minimum legal runtime (one request's full context) and the cache takes what remains, unless the
// cache is explicitly off. Every cached expert saves bus traffic on each token that routes to it;
// KV beyond the context in use saves nothing. Explicit KV keeps its requested shape.
[[nodiscard]] inline std::uint64_t native_expert_runtime_reservation(
    bool automatic_kv, std::optional<std::uint64_t> cache_bytes,
    std::uint64_t resolved_runtime, std::uint64_t minimum_runtime) noexcept {
    return automatic_kv && cache_bytes.value_or(1) != 0 ? minimum_runtime : resolved_runtime;
}

} // namespace ninfer::models::qwen3_5
