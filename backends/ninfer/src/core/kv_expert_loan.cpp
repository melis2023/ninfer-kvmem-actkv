#include "core/kv_expert_loan.h"

#include "core/kv_loan.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace {

// Inserts a run into sorted, merged runs.
void add_run(std::vector<KVPageRun>& runs, KVPageRun run) {
    const auto at = std::lower_bound(runs.begin(), runs.end(), run.begin,
                                     [](const KVPageRun& r, std::int32_t page) { return r.begin < page; });
    const auto index = static_cast<std::size_t>(at - runs.begin());
    runs.insert(at, run);
    // Merge with the right neighbour, then the left one.
    if (index + 1 < runs.size() &&
        static_cast<std::int64_t>(runs[index].begin) + runs[index].count == runs[index + 1].begin) {
        runs[index].count += runs[index + 1].count;
        runs.erase(runs.begin() + static_cast<std::ptrdiff_t>(index + 1));
    }
    if (index > 0 &&
        static_cast<std::int64_t>(runs[index - 1].begin) + runs[index - 1].count == runs[index].begin) {
        runs[index - 1].count += runs[index].count;
        runs.erase(runs.begin() + static_cast<std::ptrdiff_t>(index));
    }
}

} // namespace

KvExpertLoan::KvExpertLoan(EvictableKVPool& arena, std::vector<DeviceKVPagePool*> pools,
                           std::vector<std::uint32_t> keep, Cache cache)
    : arena_(arena), cache_(std::move(cache)) {
    if (pools.size() != keep.size() || !cache_.grow || !cache_.shrink) {
        throw std::invalid_argument("KV expert loan needs a cache and a margin for every pool");
    }
    for (std::size_t index = 0; index < pools.size(); ++index) {
        if (pools[index] == nullptr) { throw std::invalid_argument("KV expert loan pool is null"); }
        pools_.push_back({pools[index], keep[index], {}});
    }
    for (std::size_t index = 0; index < pools_.size(); ++index) {
        pools_[index].pages->set_recall([this, index](std::uint32_t pages) { recall(index, pages); });
    }
}

KvExpertLoan::~KvExpertLoan() {
    for (const Pool& pool : pools_) { pool.pages->set_recall({}); }
}

std::uint64_t KvExpertLoan::lend(std::uint32_t minimum_pages) {
    const std::size_t granularity = arena_.granularity();
    bool loaned                   = false;
    std::vector<std::size_t> release;
    for (Pool& pool : pools_) {
        const std::uint32_t unit = kv_loan_unit_pages(*pool.pages, granularity);
        if (unit == 0) { continue; }
        const std::uint32_t free = pool.pages->unpromised_free_pages();
        std::uint32_t spare      = free > pool.keep ? (free - pool.keep) / unit * unit : 0;
        if (spare == 0 || spare < minimum_pages) { continue; }
        const std::span<const KVPageRun> runs = pool.pages->free_runs();
        const std::vector<KVPageRun> candidates(runs.begin(), runs.end());
        for (auto run = candidates.rbegin(); run != candidates.rend() && spare != 0; ++run) {
            const std::uint32_t length = std::min(run->count, spare) / unit * unit;
            if (length == 0) { continue; }
            const KVPageRun loan{.begin = run->begin + static_cast<std::int32_t>(run->count - length),
                                 .count = length};
            pool.pages->loan_pages(loan.begin, loan.count);
            add_run(pool.loaned, loan);
            stats_.loaned_pages += length;
            spare -= length;
            loaned = true;
        }
        for (const KVPageRun& run : pool.loaned) {
            std::vector<std::size_t> inside;
            collect_loan_granules(arena_, *pool.pages, run.begin, run.count, inside);
            for (const std::size_t granule : inside) {
                if (!arena_.granule_released(granule)) { release.push_back(granule); }
            }
        }
    }
    if (!loaned) { return 0; }
    std::sort(release.begin(), release.end());
    release.erase(std::unique(release.begin(), release.end()), release.end());
    if (!release.empty()) { arena_.release_granules(release); }
    stats_.released_bytes = arena_.released_granules() * granularity;
    ++stats_.lends;
    return cache_.grow();
}

std::vector<std::size_t> KvExpertLoan::touched(const Pool& pool,
                                               const std::vector<KVPageRun>& runs) const {
    const std::size_t granularity = arena_.granularity();
    const auto* const base        = static_cast<const std::byte*>(arena_.arena().data);
    std::vector<std::size_t> out;
    for (std::size_t plane = 0; plane < pool.pages->plane_count(); ++plane) {
        const auto stride = static_cast<std::size_t>(pool.pages->plane(plane).nb[3]);
        // The planes collect_loan_granules lends from; the others are never released.
        if (stride == 0 || granularity % stride != 0 || granularity / stride > kMaxLoanUnitPages) {
            continue;
        }
        for (const KVPageRun& run : runs) {
            const KVPlaneByteRange range = pool.pages->plane_page_range(plane, run.begin, run.count);
            const auto offset = static_cast<std::size_t>(static_cast<const std::byte*>(range.base) - base);
            for (std::size_t granule = offset / granularity;
                 granule <= (offset + range.bytes - 1) / granularity; ++granule) {
                if (arena_.granule_released(granule)) { out.push_back(granule); }
            }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void KvExpertLoan::recall(std::size_t index, std::uint32_t pages) {
    Pool& pool = pools_.at(index);
    const std::uint32_t unit = std::max<std::uint32_t>(
        1, kv_loan_unit_pages(*pool.pages, arena_.granularity()));
    // The lowest loaned pages, in whole units, until `pages` have returned.
    std::vector<KVPageRun> back;
    std::uint32_t total = 0;
    for (const KVPageRun& run : pool.loaned) {
        if (total >= pages) { break; }
        const std::uint32_t wanted = (pages - total + unit - 1) / unit * unit;
        const std::uint32_t length = std::min(run.count, wanted);
        back.push_back({.begin = run.begin, .count = length});
        total += length;
    }
    if (back.empty()) { return; }
    const std::vector<std::size_t> granules = touched(pool, back);
    if (!granules.empty()) {
        // The expert cache gives the device the granules' bytes before the arena backs them.
        cache_.shrink(granules.size() * arena_.granularity());
        arena_.restore_granules(granules);
    }
    for (const KVPageRun& run : back) {
        pool.pages->recall_pages(run.begin, run.count);
        KVPageRun& front = pool.loaned.front();
        front.begin += static_cast<std::int32_t>(run.count);
        front.count -= run.count;
        if (front.count == 0) { pool.loaned.erase(pool.loaned.begin()); }
        stats_.loaned_pages -= run.count;
    }
    stats_.released_bytes = arena_.released_granules() * arena_.granularity();
    ++stats_.recalls;
}

void KvExpertLoan::recall_all() {
    for (std::size_t index = 0; index < pools_.size(); ++index) {
        std::uint32_t loaned = 0;
        for (const KVPageRun& run : pools_[index].loaned) { loaned += run.count; }
        if (loaned != 0) { recall(index, loaned); }
    }
}

KvExpertLoan::Stats KvExpertLoan::stats() const noexcept { return stats_; }

} // namespace ninfer
