#include "calibration/route_catalog.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::calibration {
namespace {

std::vector<RouteCatalogEntry> build_catalog() {
    std::vector<RouteCatalogEntry> catalog;
    const auto add = [&](std::string key, std::vector<std::string> candidates) {
        catalog.push_back({std::move(key), std::move(candidates)});
    };

    // Ternary Bonsai 2 projections: the small-T kernel's row and column tiles per shape, and the
    // small-T kernel against the prefill GEMM per width.
    const std::vector<std::string> t2_small = {"r16c8",   "r16c16", "r32c32",  "r16c8w1",
                                               "r16c8s3", "r32c8",  "r32c8k4", "r16c16w1",
                                               "r32c16",  "r16c32", "r32c16k8"};
    for (const char* shape : {"4096+12288x5120", "7168+7168x5120", "5120x6144", "34816x5120",
                              "5120x17408", "248320x5120"}) {
        add(std::string("t2_i8_small/") + shape, t2_small);
    }
    add("t2_i8_route", {"small", "tile"});

    // Qwen3.6/3.8 groupwise-int fused projections, and the switch of each onto the unified
    // templates.
    add("q4_q5_attn_input/5120x6144x1024",
        {"parent_split_fixed", "small_t_mma", "grouped_r32_c32_s4", "grouped_r32_c64_s4",
         "mixed_r32_c64_s3", "pair_r32_c64_s3", "mixed_r64_c128_s2"});
    add("q4_q5_gdn_input/5120x4096x12288",
        {"independent_direct", "small_t_mma", "grouped_r64_c8", "grouped_r64_c16",
         "grouped_r64_c32", "grouped_r64_c64", "grouped_r64_c128", "grouped_r32_c32_s2",
         "grouped_r32_c64_s4"});
    const std::vector<std::string> q5_linear_add = {
        "split2_exact", "small_t_mma",    "mma_r64_c16",    "mma_r64_c24", "mma_r64_c32",
        "mma_r64_c64",  "mma_r64_c32_s3", "mma_r64_c32_s4", "mma_r64_c128"};
    add("q5_linear_add/5120x6144", q5_linear_add);
    add("q5_linear_add/5120x17408", q5_linear_add);
    // Q4 output and down projections (an imatrix-searched artifact stores them as Q4) and the
    // Q6 vocabulary head: the small-T kernels per column tile, and the head's per-row GEMV.
    const std::vector<std::string> q4_linear_add = {"small_t_c8", "small_t_c16", "small_t_c32"};
    add("q4_linear_add/5120x6144", q4_linear_add);
    add("q4_linear_add/5120x17408", q4_linear_add);
    add("q6_head/248320x5120", {"gemv", "small_t"});
    add("q4_linear_swiglu/34816x17408x5120",
        {"gemv_pair", "small_t_tiled", "split_half_pair_c40", "split_half_pair_c48",
         "split_half_pair_c128", "split_half_pair_c128_tail"});
    for (const char* key : {"unified/q4_q5_attn_input", "unified/q4_q5_gdn_input",
                            "unified/q5_linear_add/5120x6144", "unified/q5_linear_add/5120x17408",
                            "unified/q4_linear_swiglu", "unified/q4_linear_add"}) {
        add(key, {"unified"});
    }

    // The two-stage GDN prefill for 48 and 32 value heads.
    add("gdn_two_stage/h48", {"on"});
    add("gdn_two_stage/h32", {"on"});

    // Prompt attention switches, measured on rk8v4.
    add("attn_pv_f16", {"on"});
    add("attn_pack_gqa", {"on"});
    add("attn_prompt_fast", {"on"});
    // Parallel query tiles for single-row chunked small-T widths, measured on rk8v4.
    add("attn_parallel_tiles", {"on"});

    // Small-T attention launch tiers for 24 query heads over 4 KV heads, per KV coding and query
    // width: the tier list depends on how many 16-row tiles the width's query rows fill.
    constexpr int query_heads_per_kv_head = 24 / 4;
    for (const char* coding : {"rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8", "int8"}) {
        for (int width = 1; width <= 8; ++width) {
            const int row_tiles = (width * query_heads_per_kv_head + 15) / 16;
            std::vector<std::string> tiers;
            if (row_tiles == 1) {
                tiers = {"2x4x32",  "4x2x32",  "8x2x32",  "16x1x32",  "2x2x32q",
                         "4x2x32q", "4x2x32e", "8x2x32e", "16x1x32e", "4x2x32qe"};
            } else if (row_tiles == 2) {
                tiers = {"4x2x32",  "8x2x32",  "16x1x32",  "32x1x32",  "4x2x32q",  "8x2x32q",
                         "4x2x32e", "8x2x32e", "16x1x32e", "4x2x32qe", "8x2x32qe"};
            } else {
                tiers = {"6x2x32",   "12x1x32", "12x1x64d", "24x1x32",  "6x2x32q",
                         "12x1x32q", "6x2x32e", "12x1x32e", "6x2x32qe", "12x1x32qe"};
            }
            add(std::string("attn_i8_small/h24/") + coding + "/w" + std::to_string(width),
                std::move(tiers));
        }
    }

    std::sort(catalog.begin(), catalog.end(),
              [](const RouteCatalogEntry& a, const RouteCatalogEntry& b) { return a.key < b.key; });
    for (std::size_t index = 1; index < catalog.size(); ++index) {
        if (catalog[index].key == catalog[index - 1].key) {
            throw std::logic_error("route catalog lists " + catalog[index].key + " twice");
        }
    }
    return catalog;
}

} // namespace

const std::vector<RouteCatalogEntry>& calibration_route_catalog() {
    static const std::vector<RouteCatalogEntry> catalog = build_catalog();
    return catalog;
}

const std::vector<std::string>& calibration_candidates(std::string_view key) {
    const auto& catalog = calibration_route_catalog();
    const auto found    = std::lower_bound(
        catalog.begin(), catalog.end(), key,
        [](const RouteCatalogEntry& entry, std::string_view wanted) { return entry.key < wanted; });
    if (found == catalog.end() || found->key != key) {
        throw std::logic_error("route " + std::string(key) +
                               " is not in the calibration route catalog");
    }
    return found->candidates;
}

const std::string& calibration_route_catalog_digest() {
    static const std::string digest = [] {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        const auto feed    = [&](std::string_view text, char terminator) {
            for (const char c : text) {
                hash ^= static_cast<unsigned char>(c);
                hash *= 0x100000001b3ULL;
            }
            hash ^= static_cast<unsigned char>(terminator);
            hash *= 0x100000001b3ULL;
        };
        for (const RouteCatalogEntry& entry : calibration_route_catalog()) {
            feed(entry.key, '\n');
            for (const std::string& candidate : entry.candidates) { feed(candidate, '\0'); }
        }
        char text[17];
        std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
        return std::string(text);
    }();
    return digest;
}

} // namespace ninfer::calibration
