#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace kvmem {

inline void report_token_adjustment(const char * name, uint32_t requested, uint32_t effective,
                                    const char * reason) {
    if (requested != effective)
        std::fprintf(stderr, "KVMEM_CONFIG %s: %u -> %u tokens (%s)\n", name, requested, effective, reason);
}

inline uint32_t align_token_option(const char * name, uint32_t value, uint32_t block,
                                   uint32_t minimum, bool minimum_block = false) {
    if (!block) throw std::invalid_argument("BlockTokens must be positive");
    const uint32_t aligned = value / block * block;
    const uint32_t effective = minimum_block ? std::max(minimum, aligned) : aligned;
    if (effective < minimum)
        throw std::invalid_argument(std::string(name) + " must retain at least " +
                                    std::to_string(minimum) + " tokens after alignment");
    report_token_adjustment(name, value, effective, "block alignment");
    return effective;
}

inline uint32_t normalize_prefill(uint32_t requested, uint32_t reserve, uint32_t alignment) {
    if (!requested) throw std::invalid_argument("Prefill must be positive");
    const uint32_t effective = std::min(requested, reserve) / alignment * alignment;
    if (!effective)
        throw std::invalid_argument("Prefill and Reserve must allow at least " +
                                    std::to_string(alignment) + " prefill tokens");
    report_token_adjustment("Prefill", requested, effective, "prefill alignment and Reserve limit");
    return effective;
}

inline void normalize_memory_tokens(uint32_t & budget, uint32_t & reserve,
                                    uint32_t & sink, uint32_t & recent,
                                    uint32_t block, bool llama, bool polystrata) {
    // Zero budget is the native disabled/automatic sentinel; callers resolve it first.
    if (!budget) return;
    budget = align_token_option("Budget", budget, block, llama ? block : 128, llama);
    reserve = align_token_option("Reserve", reserve, block, block, llama);
    sink = align_token_option("SinkTokens", sink, block, polystrata ? 0 : block, !polystrata);
    recent = align_token_option("RecentTokens", recent, block, llama ? 0 : block);
    if (uint64_t(sink) + recent > budget)
        throw std::invalid_argument("SinkTokens + RecentTokens must fit Budget after alignment");
}

} // namespace kvmem
