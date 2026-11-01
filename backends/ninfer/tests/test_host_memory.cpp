#include "runtime/engine/host_memory.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ninfer::ContextCacheOptions;
using ninfer::runtime::cgroup_limit_bytes;
using ninfer::runtime::cgroup_remaining_bytes;
using ninfer::runtime::resolve_auto_host_cache;

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kGiB = 1ULL << 30;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

void write(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

// A limit may sit on any ancestor of the process's cgroup, not the mount root, and the tightest
// level decides. Each case is a fabricated tree under a scratch directory.
int check_cgroup_probe() {
    int failures = 0;
    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "ninfer_host_memory_cgroup_test";
    std::filesystem::remove_all(scratch);

    // v2: an ancestor limited to 10 GiB with 2 GiB charged, of which 1 GiB inactive and 512 MiB
    // active file cache is reclaimable, so 9.5 GiB remain two levels below it.
    const std::filesystem::path v2 = scratch / "v2";
    write(v2 / "memory.max", "max\n");
    write(v2 / "app.slice" / "memory.max", "10737418240\n");
    write(v2 / "app.slice" / "memory.current", "2147483648\n");
    write(v2 / "app.slice" / "memory.stat",
          "anon 1\nactive_file 536870912\ninactive_file 1073741824\n");
    write(v2 / "app.slice" / "c1" / "memory.max", "max\n");
    write(v2 / "app.slice" / "c1" / "leaf" / "memory.max", "max\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c1/leaf\n") ==
                          9 * kGiB + 512 * kMiB,
                      "an ancestor's limit, less its file cache, was not applied");
    // A tighter child wins over the ancestor: 4 GiB limit, 1 GiB charged -> 3 GiB.
    write(v2 / "app.slice" / "c2" / "memory.max", "4294967296\n");
    write(v2 / "app.slice" / "c2" / "memory.current", "1073741824\n");
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c2\n") == 3 * kGiB,
                      "the tightest limit along the cgroup path did not win");
    failures += check(cgroup_remaining_bytes(v2, "0::/app.slice/c2 (deleted)\n") == 3 * kGiB,
                      "a deleted cgroup suffix broke path resolution");
    failures += check(!cgroup_remaining_bytes(v2, "0::/other/place\n").has_value(),
                      "an unlimited cgroup path reported a limit");
    // v1, with the memory controller listed among others: 8 GiB limit, 3 GiB charged, 1.5 GiB of
    // it reclaimable -> 6.5 GiB; v1's "no limit" value is no limit.
    const std::filesystem::path v1 = scratch / "v1";
    write(v1 / "memory" / "grp" / "memory.limit_in_bytes", "8589934592\n");
    write(v1 / "memory" / "grp" / "memory.usage_in_bytes", "3221225472\n");
    write(v1 / "memory" / "grp" / "memory.stat",
          "total_active_file 536870912\ntotal_inactive_file 1073741824\n");
    failures += check(cgroup_remaining_bytes(v1, "12:cpu,cpuacct:/x\n4:memory:/grp\n0::/\n") ==
                          6 * kGiB + 512 * kMiB,
                      "a cgroup v1 memory limit was not applied");
    write(v1 / "memory" / "free" / "memory.limit_in_bytes", "9223372036854771712\n");
    write(v1 / "memory" / "free" / "memory.usage_in_bytes", "1\n");
    failures += check(!cgroup_remaining_bytes(v1, "4:memory:/free\n").has_value(),
                      "a cgroup v1 'no limit' value was treated as a limit");
    // The total is the tightest limit itself, not what remains of it.
    failures += check(cgroup_limit_bytes(v2, "0::/app.slice/c1/leaf\n") == 10 * kGiB &&
                          cgroup_limit_bytes(v2, "0::/app.slice/c2\n") == 4 * kGiB &&
                          cgroup_limit_bytes(v1, "4:memory:/grp\n") == 8 * kGiB,
                      "the cgroup limit was not the tightest limit on the path");
    std::filesystem::remove_all(scratch);
    return failures;
}

int check_budget() {
    int failures = 0;
    ContextCacheOptions requested;
    requested.host_cache_auto = true;
    // Everything available but the 3 GiB reserve and the media caches.
    const ContextCacheOptions spent = resolve_auto_host_cache(requested, 40 * kGiB, 64 * kGiB,
                                                              3 * kGiB);
    failures += check(!spent.host_cache_auto && spent.host_cache_budget_bytes == 34 * kGiB,
                      "the automatic budget is not the available memory less the reserves");
    // Less available than reserved: no Host tier rather than an error.
    failures += check(resolve_auto_host_cache(requested, 2 * kGiB, 64 * kGiB, 0)
                              .host_cache_budget_bytes == 0,
                      "a machine below the reserve did not get an empty Host tier");
    // The cap and the percentage only lower it; the smallest bound wins.
    requested.host_cache_max_bytes = 20 * kGiB;
    failures += check(resolve_auto_host_cache(requested, 40 * kGiB, 64 * kGiB, 0)
                              .host_cache_budget_bytes == 20 * kGiB,
                      "--host-cache-max-mib did not cap the automatic budget");
    requested.host_cache_percent = 25;
    failures += check(resolve_auto_host_cache(requested, 40 * kGiB, 64 * kGiB, 0)
                              .host_cache_budget_bytes == 16 * kGiB,
                      "--host-cache-percent did not cap the automatic budget");
    bool refused = false;
    try {
        (void)resolve_auto_host_cache(requested, 40 * kGiB, std::nullopt, 0);
    } catch (const std::invalid_argument&) { refused = true; }
    failures += check(refused, "a percentage without the machine's total was accepted");
    return failures;
}

} // namespace

int main() {
    const int failures = check_cgroup_probe() + check_budget();
    if (failures == 0) { std::cout << "host memory tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
