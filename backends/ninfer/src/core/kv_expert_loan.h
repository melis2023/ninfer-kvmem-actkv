#pragma once

#include "core/evictable_kv_pool.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ninfer {

// Free KV pages of a single-device arena as expert cache memory, for host or disk experts. lend()
// loans the pools' free pages that no reservation needs (DeviceKVPagePool::loan_pages), beyond
// `keep` pages per pool, from the highest free runs; gives the device the physical pieces of the
// arena granules that lie entirely inside loaned pages; and lets the elastic expert cache grow into
// them. A reservation the free runs cannot cover calls back: the lowest loaned pages return, the
// expert cache first giving back the bytes of the granules they need. Nothing quiesces a stream
// beyond what the expert cache does: loaned pages are free, so no work addresses them.
class KvExpertLoan {
public:
    // The elastic expert cache: grow takes free device memory and returns the bytes it took;
    // shrink gives back at least `bytes` and returns the bytes it freed.
    struct Cache {
        std::function<std::uint64_t()> grow;
        std::function<std::uint64_t(std::uint64_t bytes)> shrink;
    };
    struct Stats {
        std::uint32_t loaned_pages = 0;
        std::uint64_t released_bytes = 0;
        std::uint64_t lends = 0, recalls = 0;
    };

    // The pools' planes lie in `arena`; `keep[i]` free pages of pools[i] are never loaned.
    KvExpertLoan(EvictableKVPool& arena, std::vector<DeviceKVPagePool*> pools,
                 std::vector<std::uint32_t> keep, Cache cache);
    // Clears the recall hooks. Released granules stay released: the arena's destructor skips them.
    ~KvExpertLoan();
    KvExpertLoan(const KvExpertLoan&)            = delete;
    KvExpertLoan& operator=(const KvExpertLoan&) = delete;

    // Loans what the pools spare when that reaches `minimum_pages` in some pool, then grows the
    // cache. Returns the bytes the cache took.
    std::uint64_t lend(std::uint32_t minimum_pages);
    // Every loan returns (before a model suspend, which copies every page outside the free runs).
    void recall_all();
    [[nodiscard]] Stats stats() const noexcept;

private:
    struct Pool {
        DeviceKVPagePool* pages = nullptr;
        std::uint32_t keep      = 0;
        std::vector<KVPageRun> loaned; // sorted, merged
    };
    // Returns loaned pages of pool `index`, lowest first, until `pages` have returned (whole loan
    // units, at most the loaned pages).
    void recall(std::size_t index, std::uint32_t pages);
    // The released granules any of `runs` reaches in the pool's lendable planes.
    [[nodiscard]] std::vector<std::size_t> touched(const Pool& pool,
                                                   const std::vector<KVPageRun>& runs) const;

    EvictableKVPool& arena_;
    std::vector<Pool> pools_;
    Cache cache_;
    Stats stats_;
};

} // namespace ninfer
