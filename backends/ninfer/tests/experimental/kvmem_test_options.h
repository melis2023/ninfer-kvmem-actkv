#pragma once

#include "core/paged_kv_storage.h"
#include "ninfer/types.h"

#include <cstdint>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test {
// Run the existing correctness fixtures on physical pipeline ranks without changing their CLI.
// CUDA_VISIBLE_DEVICES defines UUID order; these are the resulting local ordinals.
inline void kvmem_test_devices(EngineOptions& options) {
    const auto parse = [](const char* name) {
        std::vector<int> values;
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0') return values;
        std::string_view list(raw);
        while (!list.empty()) {
            const auto separator = list.find(',');
            const auto item = list.substr(0, separator);
            int value = 0;
            const auto result = std::from_chars(item.data(), item.data() + item.size(), value);
            if (result.ec != std::errc{} || result.ptr != item.data() + item.size() || value < 0)
                throw std::invalid_argument(std::string("invalid ") + name);
            values.push_back(value);
            if (separator == std::string_view::npos) break;
            list.remove_prefix(separator + 1);
            if (list.empty()) throw std::invalid_argument(std::string("invalid ") + name);
        }
        return values;
    };
    options.devices = parse("NINFER_TEST_DEVICE_IDS");
    const auto layers = parse("NINFER_TEST_STAGE_LAYERS");
    options.stage_layers.assign(layers.begin(), layers.end());
}

inline KvCacheStorage kvmem_test_storage(std::string_view name) {
    if (name == "bf16") return KvCacheStorage::BFloat16;
    if (name == "int8") return KvCacheStorage::Int8Group64;
    if (name == "nvfp4") return KvCacheStorage::Nvfp4Group16;
    if (name == "k8v4") return KvCacheStorage::Fp8KeyNvfp4Value;
    if (name == "rk8v4") return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    if (name == "fp8") return KvCacheStorage::Fp8E4M3Row256;
    if (name == "rk4v4") return KvCacheStorage::RotatedLloyd4KeyInt4Value;
    if (name == "rk4v4-e8") return KvCacheStorage::RotatedInt4KeyInt4ValueE8;
    if (name == "rk2v4-e8") return KvCacheStorage::RotatedE8RootKeyInt4Value;
    throw std::invalid_argument("unknown KVMem test KV format: " + std::string(name));
}

inline bool kvmem_test_graph(std::string_view name) {
    if (name == "graph") return true;
    if (name == "eager") return false;
    throw std::invalid_argument("unknown KVMem test execution mode: " + std::string(name));
}

// Keep the existing D256 INT8 Host-pressure workloads equally restrictive for
// other packed formats. Explicit command-line Host budgets still mean actual bytes.
inline std::uint64_t kvmem_test_host_bytes(std::uint64_t int8_bytes, KvCacheStorage storage) {
    const auto bytes = paged_kv_storage_layout(storage, 256).physical_bytes_per_token_head();
    const auto int8 = paged_kv_storage_layout(KvCacheStorage::Int8Group64, 256)
                          .physical_bytes_per_token_head();
    return int8_bytes * bytes / int8;
}
} // namespace ninfer::test
