#pragma once

#include "ninfer/engine_options.h"

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

inline constexpr const char* kHybridExpertHelp =
    "  Host and disk expert options (--expert-residency host|disk):\n"
    "  --expert-dma-share F|auto     GPU share of a decode or verification call's cache\n"
    "                                misses, 0..1; the CPU computes the rest (GGUF and\n"
    "                                Qwen3.5/3.6 MoE experts: the GPU's results bit for bit).\n"
    "                                auto (default): those split each call by measured CPU\n"
    "                                and bus times, Flash-Next native experts copy every miss.\n"
    "                                Flash-Next, Gemma 4 and Flint calls of more than 8\n"
    "                                tokens, and Qwen3.5/3.6 calls routed to half the experts\n"
    "                                or more, always use the GPU\n"
    "  --expert-cpu-threads N        CPU workers, 1..256 (default: one per physical core, at\n"
    "                                most half the CPUs)\n"
    "  --expert-profile FILE         fill the expert cache from recorded counts (default: the\n"
    "                                built-in profile of the model family, where one exists)\n"
    "  --expert-profile-out FILE     record expert counts for this artifact after requests\n"
    "  Flash-Next-only expert options:\n"
    "  --expert-misses staged|mapped GGUF experts: copy misses into device staging while the\n"
    "                                cached experts run (default), or read host experts across\n"
    "                                the bus inside the expert kernels\n"
    "  --expert-tail-skip N          GGUF host experts: leave out of a decode or verification\n"
    "                                call each uncached expert that all its tokens rank past\n"
    "                                place N (1..9) of their top ten; changes results\n"
    "  --expert-prefill-buffers 1|2  GGUF host experts: device buffers for a long prompt's\n"
    "                                uncached experts; 2 copies the next layer's while this one\n"
    "                                runs (default 2 with the automatic cache, which holds them\n"
    "                                as expert cache slots between prompts, else 1)\n"
    "  --expert-cache-adaptive       replace cold cached experts between calls;\n"
    "                                changes the CPU/GPU arithmetic partition\n";

template<class Value>
bool parse_hybrid_expert_option(std::string_view option, HybridExpertOptions& out, Value&& value) {
    if (option == "--expert-dma-share") {
        const std::string text(value());
        if (text == "auto") {
            out.dma_share.reset();
            return true;
        }
        float share = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), share);
        if (error != std::errc{} || end != text.data() + text.size() ||
            !std::isfinite(share) || share < 0 || share > 1) {
            throw std::invalid_argument("--expert-dma-share takes a finite value in 0..1 or auto");
        }
        out.dma_share = share;
    } else if (option == "--expert-cpu-threads") {
        const std::string text(value());
        std::uint32_t threads = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), threads);
        if (error != std::errc{} || end != text.data() + text.size() || threads == 0 || threads > 256) {
            throw std::invalid_argument("--expert-cpu-threads takes an integer in 1..256");
        }
        out.cpu_threads = threads;
    } else if (option == "--expert-tail-skip") {
        const std::string text(value());
        std::uint32_t rank      = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), rank);
        if (error != std::errc{} || end != text.data() + text.size() || rank == 0 || rank > 9) {
            throw std::invalid_argument("--expert-tail-skip takes an integer in 1..9");
        }
        out.tail_skip_rank = rank;
    } else if (option == "--expert-prefill-buffers") {
        const std::string_view count(value());
        if (count != "1" && count != "2") {
            throw std::invalid_argument("--expert-prefill-buffers takes 1 or 2");
        }
        out.prefill_buffers = count == "2" ? 2 : 1;
    } else if (option == "--expert-misses") {
        const std::string_view mode(value());
        if (mode == "staged") {
            out.mapped_misses = false;
        } else if (mode == "mapped") {
            out.mapped_misses = true;
        } else {
            throw std::invalid_argument("--expert-misses takes staged or mapped");
        }
    } else if (option == "--expert-cache-adaptive") {
        out.adaptive_cache = true;
    } else if (option == "--expert-profile" || option == "--expert-profile-out") {
        const std::string path(value());
        if (path.empty()) { throw std::invalid_argument(std::string(option) + " takes a file path"); }
        (option == "--expert-profile" ? out.routing_profile : out.record_profile) = path;
    } else { return false; }
    return true;
}

} // namespace ninfer::product
