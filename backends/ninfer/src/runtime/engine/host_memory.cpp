#include "runtime/engine/host_memory.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ninfer::runtime {
namespace {

std::optional<std::uint64_t> read_u64_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::string text;
    if (!file || !(file >> text) || text == "max") { return std::nullopt; }
    try {
        return std::stoull(text);
    } catch (...) { return std::nullopt; }
}

// File-backed page cache charged to a cgroup (active and inactive lists) is reclaimable, so it does
// not count against what remains: pages read more than once, such as an artifact just loaded, sit
// on the active list.
std::uint64_t reclaimable_file_bytes(const std::filesystem::path& path, const char* active_key,
                                     const char* inactive_key) {
    std::ifstream file(path);
    std::string name;
    std::uint64_t value = 0;
    std::uint64_t total = 0;
    while (file >> name >> value) {
        if (name == active_key || name == inactive_key) { total += value; }
    }
    return total;
}

enum class CgroupMeasure { Remaining, Limit };

// One cgroup directory's limit, or its limit less its non-reclaimable use. Empty when the directory
// sets no limit (or does not exist), which leaves the decision to its ancestors.
std::optional<std::uint64_t> measure_under(const std::filesystem::path& dir, bool v2,
                                           CgroupMeasure measure) {
    const auto limit = read_u64_file(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
    // cgroup v1 reports "no limit" as a value near the top of the page-aligned 63-bit range.
    if (!limit || (!v2 && *limit >= (1ULL << 60))) { return std::nullopt; }
    if (measure == CgroupMeasure::Limit) { return limit; }
    const auto used = read_u64_file(dir / (v2 ? "memory.current" : "memory.usage_in_bytes"));
    if (!used) { return std::nullopt; }
    const std::uint64_t reclaimable = reclaimable_file_bytes(
        dir / "memory.stat", v2 ? "active_file" : "total_active_file",
        v2 ? "inactive_file" : "total_inactive_file");
    const std::uint64_t charged = *used > reclaimable ? *used - reclaimable : 0;
    return *limit > charged ? *limit - charged : 0;
}

std::vector<std::string> path_components(std::string path) {
    if (const auto deleted = path.find(" (deleted)"); deleted != std::string::npos) {
        path.erase(deleted);
    }
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(path);
    while (std::getline(stream, part, '/')) {
        if (!part.empty()) { parts.push_back(part); }
    }
    return parts;
}

// The tightest value over `base/<path>` and every ancestor up to `base`.
std::optional<std::uint64_t> tightest(const std::filesystem::path& base,
                                      const std::vector<std::string>& parts, bool v2,
                                      CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    for (std::size_t depth = parts.size() + 1; depth-- > 0;) {
        std::filesystem::path dir = base;
        for (std::size_t index = 0; index < depth; ++index) { dir /= parts[index]; }
        if (const auto value = measure_under(dir, v2, measure)) {
            best = best ? std::min(*best, *value) : *value;
        }
    }
    return best;
}

std::optional<std::uint64_t> cgroup_tightest(const std::filesystem::path& cgroup_root,
                                             std::string_view proc_self_cgroup,
                                             CgroupMeasure measure) {
    std::optional<std::uint64_t> best;
    const auto take = [&](std::optional<std::uint64_t> value) {
        if (value) { best = best ? std::min(*best, *value) : *value; }
    };
    std::istringstream lines{std::string(proc_self_cgroup)};
    std::string line;
    while (std::getline(lines, line)) {
        // "<hierarchy>:<controllers>:<path>"; the path may itself contain colons.
        const std::size_t first  = line.find(':');
        const std::size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (second == std::string::npos) { continue; }
        const std::string controllers        = line.substr(first + 1, second - first - 1);
        const std::vector<std::string> parts = path_components(line.substr(second + 1));
        if (line.compare(0, first, "0") == 0 && controllers.empty()) {
            take(tightest(cgroup_root, parts, true, measure));
        } else if ((',' + controllers + ',').find(",memory,") != std::string::npos) {
            take(tightest(cgroup_root / "memory", parts, false, measure));
        }
    }
    return best;
}

#if defined(__linux__)
std::optional<std::uint64_t> meminfo_bytes(std::string_view wanted) {
    std::ifstream file("/proc/meminfo");
    std::string key;
    std::uint64_t kib = 0;
    std::string unit;
    while (file >> key >> kib) {
        std::getline(file, unit);
        if (key == wanted) { return kib * 1024ULL; }
    }
    return std::nullopt;
}

std::string proc_self_cgroup() {
    std::ifstream self("/proc/self/cgroup");
    std::ostringstream text;
    text << self.rdbuf();
    return text.str();
}
#endif

} // namespace

std::optional<std::uint64_t> cgroup_remaining_bytes(const std::filesystem::path& cgroup_root,
                                                    std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Remaining);
}

std::optional<std::uint64_t> cgroup_limit_bytes(const std::filesystem::path& cgroup_root,
                                                std::string_view proc_self_cgroup) {
    return cgroup_tightest(cgroup_root, proc_self_cgroup, CgroupMeasure::Limit);
}

std::optional<std::uint64_t> available_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemAvailable:");
        const std::optional<std::uint64_t> cgroup =
            cgroup_remaining_bytes("/sys/fs/cgroup", proc_self_cgroup());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

std::optional<std::uint64_t> total_host_memory_bytes() noexcept {
    try {
#if defined(_WIN32)
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (!GlobalMemoryStatusEx(&status)) { return std::nullopt; }
        return static_cast<std::uint64_t>(status.ullTotalPhys);
#elif defined(__linux__)
        const std::optional<std::uint64_t> system = meminfo_bytes("MemTotal:");
        const std::optional<std::uint64_t> cgroup =
            cgroup_limit_bytes("/sys/fs/cgroup", proc_self_cgroup());
        if (system && cgroup) { return std::min(*system, *cgroup); }
        return system ? system : cgroup;
#else
        return std::nullopt;
#endif
    } catch (...) { return std::nullopt; }
}

ContextCacheOptions resolve_auto_host_cache(const ContextCacheOptions& requested,
                                            std::uint64_t available_bytes,
                                            std::optional<std::uint64_t> total_bytes,
                                            std::uint64_t extra_reserve_bytes) {
    if (!requested.host_cache_auto) {
        throw std::logic_error("host cache sizing was requested without host_cache_auto");
    }
    // Pinned pages cannot be reclaimed, so the reserve for what still grows (request buffers, the
    // response store, graph instantiation, the media caches) is a fixed amount, not a fraction.
    const std::uint64_t max  = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t base = requested.host_cache_reserve_bytes;
    const std::uint64_t reserve =
        base > max - extra_reserve_bytes ? max : base + extra_reserve_bytes;
    std::uint64_t budget = available_bytes > reserve ? available_bytes - reserve : 0;
    // The reserve leaves memory for what still grows; a cap leaves it for the machine's other
    // users, which the available memory cannot know will want it.
    if (requested.host_cache_max_bytes) {
        budget = std::min<std::uint64_t>(budget, *requested.host_cache_max_bytes);
    }
    if (requested.host_cache_percent) {
        const std::uint64_t percent = *requested.host_cache_percent;
        if (percent == 0 || percent > 100) {
            throw std::invalid_argument("host cache percent must be in [1,100]");
        }
        if (!total_bytes) {
            throw std::invalid_argument("host cache percent needs the machine's total memory");
        }
        const std::uint64_t total = *total_bytes;
        budget =
            std::min<std::uint64_t>(budget, total / 100 * percent + total % 100 * percent / 100);
    }
    budget = std::min<std::uint64_t>(budget, std::numeric_limits<std::size_t>::max());
    ContextCacheOptions resolved     = requested;
    resolved.host_cache_auto         = false;
    resolved.host_cache_budget_bytes = static_cast<std::size_t>(budget);
    return resolved;
}

ContextCacheOptions resolve_auto_host_cache_now(const ContextCacheOptions& requested,
                                                std::uint64_t extra_reserve_bytes) {
    if (!requested.host_cache_auto) { return requested; }
    const std::optional<std::uint64_t> available = available_host_memory_bytes();
    if (!available) {
        throw std::runtime_error(
            "--host-cache-mib auto needs the host's available memory, which this platform does "
            "not report; give the budget in MiB");
    }
    return resolve_auto_host_cache(requested, *available, total_host_memory_bytes(),
                                   extra_reserve_bytes);
}

} // namespace ninfer::runtime
