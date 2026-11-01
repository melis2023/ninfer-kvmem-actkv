#include "calibration/route_catalog.h"

#include "ops/linear/common/route_table.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::calibration {
namespace {

std::vector<RouteCatalogEntry> build_catalog(ops::DeviceTier tier) {
    std::vector<RouteCatalogEntry> catalog;
    const bool volta         = tier == ops::DeviceTier::Volta;
    const bool turing_shapes = ops::turing_shapes(tier);
    const auto add = [&](std::string key, std::vector<std::string> candidates) {
        catalog.push_back({std::move(key), std::move(candidates)});
    };

    if (!volta) {
        // A Volta card runs its own projection, decode-attention and single-row prompt routes
        // (ops/linear/sm70, small_t_sm70.cuh, prompt_sm70.cu), which consult none of the projection
        // and attention keys below; it calibrates the GDN prefill switch alone.

        // Ternary Bonsai 2 projections: the small-T kernel's row and column tiles per shape, and
        // the small-T kernel against the prefill GEMM per width.
        const std::vector<std::string> t2_small = {"r16c8",   "r16c16", "r32c32",  "r16c8w1",
                                                   "r16c8s3", "r32c8",  "r32c8k4", "r16c16w1",
                                                   "r32c16",  "r16c32", "r32c16k8"};
        for (const char* shape : {"4096+12288x5120", "7168+7168x5120", "5120x6144", "34816x5120",
                                  "5120x17408", "248320x5120"}) {
            add(std::string("t2_i8_small/") + shape, t2_small);
        }
        add("t2_i8_route", {"small", "tile"});
        for (const char* shape : {"4096x5120", "12288x5120", "7168x5120", "5120x6144",
                                  "34816x5120", "5120x17408", "248320x5120"}) {
            add(std::string("ptq1_i8_tile/") + shape, {"rows", "cols"});
        }

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
        // A card that cannot select the unified table (ops/linear/common/route_table.h) has no
        // switch onto it.
        if (ops::detail::unified_linear_routes(tier)) {
            for (const char* key :
                 {"unified/q4_q5_attn_input", "unified/q4_q5_gdn_input",
                  "unified/q5_linear_add/5120x6144", "unified/q5_linear_add/5120x17408",
                  "unified/q4_linear_swiglu", "unified/q4_linear_add"}) {
                add(key, {"unified"});
            }
            // QUASAR NVFP4 fused projections: the A16 chain choice at every width a fused launch
            // serves (the own chain's per-width kernels against the unified templates), and the
            // int8 family's own four keys (--nvfp4-w4a8-int), where the band picks which chain's
            // mainloop answers (own runs MMQ, unified runs the grouped ladder, and T <= 8 runs the
            // row-form kernel under either). The two families carry separate keys so that
            // calibrating one never overwrites the other.
            for (const char* key : {"unified/nvfp4_attn_input", "unified/nvfp4_gdn_input",
                                    "unified/nvfp4_linear_add", "unified/nvfp4_linear_swiglu",
                                    "nvfp4_int8/attn_input", "nvfp4_int8/gdn_input",
                                    "nvfp4_int8/linear_add", "nvfp4_int8/linear_swiglu"}) {
                add(key, {"unified"});
            }
        }
        // The cuBLAS prefill route's width gate per Op (--prefill-cublas), over the widths the
        // integer A8 mainloop cannot take and, under `/a8`, the multiples of its tile it can.
        for (const char* family : {"w4_cublas/q4_linear_swiglu", "w4_cublas/linear_add/5120x17408",
                                   "w4_cublas/linear_add/5120x6144", "w4_cublas/q4_q5_gdn_input",
                                   "w4_cublas/q4_q5_attn_input"}) {
            add(family, {"on", "off"});
            add(std::string(family) + "/a8", {"on", "off"});
        }
        // Q8 G32 K-split contractions (MTP heads, DFlash2 drafters, the official artifacts' heads):
        // the four-K-warp form of each 8-column tile (ops/linear/q8/q8_ksplit_config.h).
        for (const char* shape :
             {"6144x5120", "14336x5120", "17408x5120", "34816x5120", "248320x5120", "5120x6144",
              "5120x10240", "5120x17408", "5120x25600", "2048x4096", "2048x16384"}) {
            add(std::string("q8_ksplit/") + shape, {"kw4"});
        }
    }

    // The two-stage GDN prefill for 48 and 32 value heads.
    add("gdn_two_stage/h48", {"on"});
    add("gdn_two_stage/h32", {"on"});

    if (!volta) {
        // Prompt attention switches, measured on rk8v4. A card without the standard INT8 prompt
        // kernel (causal_prompt_i8_standard) has nothing to pack or to compare the fast one
        // against.
        add("attn_pv_f16", {"on"});
        if (ops::detail::causal_prompt_i8_standard(turing_shapes)) {
            add("attn_pack_gqa", {"on"});
            add("attn_prompt_fast", {"on"});
        }
        // Key splits of the fast prompt kernel, measured on rk8v4 where it serves prompts.
        add("attn_prompt_split", {"off"});
        // Parallel query tiles for single-row chunked small-T widths, measured on rk8v4.
        add("attn_parallel_tiles", {"on"});

        // Small-T attention launch tiers for 24 query heads over 4 KV heads, per KV coding and
        // query width: the tier list depends on how many 16-row tiles the width's query rows fill.
        constexpr int query_heads_per_kv_head = 24 / 4;
        for (const char* coding : {"rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8", "int8"}) {
            for (int width = 1; width <= 8; ++width) {
                const int row_tiles = (width * query_heads_per_kv_head + 15) / 16;
                std::vector<std::string> tiers;
                if (row_tiles == 1) {
                    tiers = {"2x4x32",  "4x2x32",  "8x2x32",  "16x1x32",  "2x2x32q",
                             "4x2x32q", "4x2x32e", "8x2x32e", "16x1x32e", "4x2x32qe"};
                } else if (row_tiles == 2) {
                    tiers = {"4x2x32",  "8x2x32",  "16x1x32",  "32x1x32",  "4x2x32q", "8x2x32q",
                             "4x2x32e", "8x2x32e", "16x1x32e", "4x2x32qe", "8x2x32qe"};
                } else {
                    tiers = {"6x2x32",   "12x1x32", "12x1x64d", "24x1x32",  "6x2x32q",
                             "12x1x32q", "6x2x32e", "12x1x32e", "6x2x32qe", "12x1x32qe"};
                    if (!ops::detail::small_t_i8_key_block64(turing_shapes)) {
                        std::erase(tiers, "12x1x64d");
                    }
                }
                // The pipelined kernel reads INT8 keys and values only.
                if (std::string_view(coding) == "int8") { tiers.emplace_back("8x2x32p"); }
                add(std::string("attn_i8_small/h24/") + coding + "/w" + std::to_string(width),
                    std::move(tiers));
            }
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

const std::vector<RouteCatalogEntry>& calibration_route_catalog(ops::DeviceTier tier) {
    static const std::array<std::vector<RouteCatalogEntry>, 4> catalogs = {
        build_catalog(ops::DeviceTier::Volta), build_catalog(ops::DeviceTier::Turing),
        build_catalog(ops::DeviceTier::Ampere), build_catalog(ops::DeviceTier::Blackwell)};
    return catalogs[static_cast<std::size_t>(tier)];
}

const std::vector<std::string>& calibration_candidates(std::string_view key, ops::DeviceTier tier) {
    const auto& catalog = calibration_route_catalog(tier);
    const auto found    = std::lower_bound(
        catalog.begin(), catalog.end(), key,
        [](const RouteCatalogEntry& entry, std::string_view wanted) { return entry.key < wanted; });
    if (found == catalog.end() || found->key != key) {
        throw std::logic_error("route " + std::string(key) +
                               " is not in the calibration route catalog");
    }
    return found->candidates;
}

namespace {

std::string catalog_digest(ops::DeviceTier tier) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    const auto feed    = [&](std::string_view text, char terminator) {
        for (const char c : text) {
            hash ^= static_cast<unsigned char>(c);
            hash *= 0x100000001b3ULL;
        }
        hash ^= static_cast<unsigned char>(terminator);
        hash *= 0x100000001b3ULL;
    };
    for (const RouteCatalogEntry& entry : calibration_route_catalog(tier)) {
        feed(entry.key, '\n');
        for (const std::string& candidate : entry.candidates) { feed(candidate, '\0'); }
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(text);
}

} // namespace

const std::string& calibration_route_catalog_digest(ops::DeviceTier tier) {
    static const std::array<std::string, 4> digests = {
        catalog_digest(ops::DeviceTier::Volta), catalog_digest(ops::DeviceTier::Turing),
        catalog_digest(ops::DeviceTier::Ampere), catalog_digest(ops::DeviceTier::Blackwell)};
    return digests[static_cast<std::size_t>(tier)];
}

} // namespace ninfer::calibration
