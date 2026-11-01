// ninfer::ops - the residency map of a streamed sparse-attention layer (contract in
// include/ninfer/ops/kv_stream.h). One CTA of 1,024 threads resolves a call: it looks up every
// block the queries read, claims the missing ones, picks their slots with a clock sweep and
// re-points the table; a second launch of a fixed grid copies the claimed and rewritten blocks
// from the host copy into their slots. Both read the call's position from a device word and the
// number of copies from the map, so a captured graph replays them at any position.
#include "ninfer/ops/kv_stream.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kResolveThreads = 1024;
constexpr int kTopBlocks      = 512;
constexpr int kHostPage       = kPagedKVPageSize;
constexpr int kBlocksPerPage  = kHostPage / kKvStreamPageTokens; // 16
// Blocks one call writes: positions first .. first + 15 span at most five blocks.
constexpr int kWrittenBlocks = kKvStreamMaxTokens / kKvStreamPageTokens + 1;
constexpr int kCopyThreads   = 256;
constexpr int kCopyBlocks    = 64;

// control words: [0] epoch, [1] hand, [2] copies of the last call, [3] overflow, then u64 calls,
// lookups and copies.
constexpr int kControlWords = 16;

static_assert(kHostPage % kKvStreamPageTokens == 0);

// One plane of the storage: its row bytes (one position, one head) and its host and pool bases.
struct Plane {
    const unsigned char* host = nullptr;
    unsigned char* pool       = nullptr;
    int words                 = 0; // 8-byte words of one block's rows of one head
    int row_bytes             = 0;
};

struct Planes {
    Plane plane[4];
    int count = 0;
    int heads = 0;
};

// Block-wide exclusive prefix sum of one int per thread; `total` is the sum over the block.
__device__ int block_scan(int value, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, offset);
        if (lane >= offset) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int t = warp_sums[lane];
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, offset);
            if (lane >= offset) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total            = warp_sums[31];
    const int before = x - value + (warp > 0 ? warp_sums[warp - 1] : 0);
    __syncthreads();
    return before;
}

__device__ __forceinline__ unsigned long long* counters(const KvStreamMap& m) {
    return reinterpret_cast<unsigned long long*>(m.control + 4);
}

__global__ void __launch_bounds__(kResolveThreads)
    resolve_kernel(KvStreamMap m, const int* __restrict__ first_word,
                   const int* __restrict__ selected, const int* __restrict__ counts, int tokens) {
    __shared__ int s_misses, s_lookups, s_refreshes, s_cut;
    __shared__ int warp_sums[32];
    __shared__ int read_written[kWrittenBlocks];
    __shared__ int refresh_block[kWrittenBlocks], refresh_slot[kWrittenBlocks];
    const int first = *first_word;
    const int low   = first / kKvStreamPageTokens;
    const int high  = (first + tokens - 1) / kKvStreamPageTokens; // inclusive
    const int epoch = m.control[0] + 1;
    if (threadIdx.x == 0) {
        s_misses    = 0;
        s_lookups   = 0;
        s_refreshes = 0;
    }
    if (threadIdx.x < kWrittenBlocks) read_written[threadIdx.x] = 0;
    __syncthreads();

    // 1. Every block a query reads: a resident one takes this call's stamp and its reference
    //    bit, a missing one is claimed exactly once (-1 -> -2). The blocks this call wrote wait
    //    for step 2. A query's own incomplete block is always among them.
    int lookups = 0;
    for (int q = 0; q < tokens; ++q) {
        const int p      = first + q;
        const int count  = counts[q];
        const bool tail  = (p + 1) % kKvStreamPageTokens != 0;
        const int* list  = selected + static_cast<long long>(q) * kTopBlocks;
        const int blocks = count + (tail ? 1 : 0);
        for (int i = threadIdx.x; i < blocks; i += kResolveThreads) {
            const int b = i < count ? list[i] : p / kKvStreamPageTokens;
            if (b >= low && b <= high) {
                read_written[b - low] = 1;
                continue;
            }
            ++lookups;
            const int slot = m.table[b];
            if (slot >= 0) {
                m.slot_stamp[slot] = epoch;
                m.slot_ref[slot]   = 1;
            } else if (slot == -1 && atomicCAS(&m.table[b], -1, -2) == -1) {
                m.copy_block[atomicAdd(&s_misses, 1)] = b;
            }
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);
    __syncthreads();

    // 2. The blocks this call wrote: one the call reads is copied again (into its slot when it
    //    has one), one it does not read leaves the pool, so no slot keeps bytes the host copy
    //    has replaced.
    if (threadIdx.x <= high - low) {
        const int b    = low + threadIdx.x;
        const int slot = m.table[b];
        if (read_written[threadIdx.x] != 0) {
            atomicAdd(&s_lookups, 1);
            if (slot >= 0) {
                m.slot_stamp[slot] = epoch;
                m.slot_ref[slot]   = 1;
                const int r        = atomicAdd(&s_refreshes, 1);
                refresh_block[r]   = b;
                refresh_slot[r]    = slot;
            } else {
                m.table[b]                            = -2;
                m.copy_block[atomicAdd(&s_misses, 1)] = b;
            }
        } else if (slot >= 0) {
            m.table[b]         = -1;
            m.slot_block[slot] = -1;
        }
    }
    __syncthreads();

    // 3. One slot per missing block: a clock sweep from the hand, a window of at most one slot
    //    per thread (a pool smaller than the CTA must not show a slot to two threads). A slot this
    //    call uses (stamped with this epoch) is never taken; a referenced one loses its bit as the
    //    hand passes and is taken on the next pass.
    const int need = s_misses, n = m.slots;
    const int window = n < kResolveThreads ? n : kResolveThreads;
    const bool sweeps = static_cast<int>(threadIdx.x) < window;
    int hand = m.control[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += window) {
        const int j      = static_cast<int>((static_cast<long long>(hand) + threadIdx.x) % n);
        const bool mine  = sweeps && m.slot_stamp[j] == epoch;
        const bool taken = sweeps && !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total        = 0;
        const int rank   = block_scan(taken ? 1 : 0, warp_sums, total);
        const int want   = need - got;
        if (threadIdx.x == 0) s_cut = window;
        __syncthreads();
        // The hand stops just past the last slot taken.
        if (taken && rank == want - 1) s_cut = threadIdx.x + 1;
        __syncthreads();
        const int cut = s_cut;
        if (taken && rank < want) {
            m.copy_slot[got + rank] = j;
            m.slot_stamp[j]         = epoch; // a sweep that wraps around must not take it twice
        } else if (sweeps && static_cast<int>(threadIdx.x) < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = static_cast<int>((static_cast<long long>(hand) + cut) % n);
        __syncthreads();
    }

    // 4. Re-point the table; the rewritten blocks join the copies in their own slots.
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += kResolveThreads) {
        const int b = m.copy_block[k];
        if (k >= placed) {
            m.table[b] = -1; // overflow: a pool below kv_stream_minimum_slots
            continue;
        }
        const int slot = m.copy_slot[k];
        const int old  = m.slot_block[slot];
        if (old >= 0) m.table[old] = -1;
        m.slot_block[slot] = b;
        m.slot_ref[slot]   = 1;
        m.table[b]         = slot;
    }
    if (threadIdx.x < s_refreshes) {
        m.copy_block[placed + threadIdx.x] = refresh_block[threadIdx.x];
        m.copy_slot[placed + threadIdx.x]  = refresh_slot[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        const int copies = placed + s_refreshes;
        m.control[0]     = epoch;
        m.control[1]     = hand;
        m.control[2]     = copies;
        if (placed < need) m.control[3] = 1;
        unsigned long long* c = counters(m);
        c[0] += 1ull;
        c[1] += static_cast<unsigned long long>(s_lookups);
        c[2] += static_cast<unsigned long long>(copies);
    }
}

// Copy k of the last resolve: block b's rows of every head and plane, from its host page into
// its slot, in 8-byte words.
__global__ void __launch_bounds__(kCopyThreads) copy_kernel(KvStreamMap m, Planes planes) {
    const int copies = m.control[2];
    int per_block    = 0;
    for (int a = 0; a < planes.count; ++a) per_block += planes.plane[a].words * planes.heads;
    for (int k = blockIdx.x; k < copies; k += gridDim.x) {
        const long long b    = m.copy_block[k];
        const long long slot = m.copy_slot[k];
        const long long page = b / kBlocksPerPage;
        const int within     = static_cast<int>(b % kBlocksPerPage) * kKvStreamPageTokens;
        for (int i = threadIdx.x; i < per_block; i += blockDim.x) {
            int rest = i, a = 0;
            while (rest >= planes.plane[a].words * planes.heads) {
                rest -= planes.plane[a].words * planes.heads;
                ++a;
            }
            const Plane& plane = planes.plane[a];
            const int head     = rest / plane.words;
            const int word     = rest % plane.words;
            const long long source =
                ((page * planes.heads + head) * kHostPage + within) * plane.row_bytes;
            const long long target =
                ((slot * planes.heads + head) * kKvStreamPageTokens) * plane.row_bytes;
            reinterpret_cast<uint2*>(plane.pool + target)[word] =
                reinterpret_cast<const uint2*>(plane.host + source)[word];
        }
    }
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long start  = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (long long i = start; i < m.blocks; i += stride) m.table[i] = -1;
    for (long long i = start; i < m.slots; i += stride) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i]   = 0;
    }
    if (start < kControlWords) m.control[start] = 0;
}

__global__ void evict_kernel(KvStreamMap m, int first_block, int end_block) {
    for (int b = first_block + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         b < end_block; b += static_cast<int>(gridDim.x * blockDim.x)) {
        const int slot = m.table[b];
        if (slot >= 0) {
            m.table[b]         = -1;
            m.slot_block[slot] = -1;
        }
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("kv_stream: ") + message); }
}

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The planes of a storage that a block's copy moves, checked against the host copy's and the
// pool's tensors.
Planes planes_of(const KvStreamMap& map, const PagedKVLayerView& host,
                 const PagedKVLayerView& pool) {
    require(host.storage == pool.storage && host.head_dim == pool.head_dim &&
                host.num_kv_heads == pool.num_kv_heads && host.num_kv_heads > 0,
            "the host copy and the pool must share one storage and geometry");
    const PagedKVStorageLayout layout = paged_kv_storage_layout(host.storage, host.head_dim);
    const std::int32_t pages          = host.k_pages.ne[3];
    require(pages > 0 && static_cast<std::int64_t>(pages) * kBlocksPerPage >= map.blocks,
            "the host copy must hold every block of the map");
    Planes out;
    out.heads       = host.num_kv_heads;
    const auto plane = [&](const Tensor& h, const Tensor& p, std::int32_t leading, DType dtype) {
        require(h.data != nullptr && p.data != nullptr && h.dtype == dtype && p.dtype == dtype &&
                    h.is_contiguous() && p.is_contiguous() && h.ne[0] == leading &&
                    p.ne[0] == leading && h.ne[1] == kHostPage &&
                    p.ne[1] == kKvStreamPageTokens && h.ne[2] == out.heads &&
                    p.ne[2] == out.heads && h.ne[3] == pages && p.ne[3] == map.slots,
                "a plane does not match the storage, the host copy or the pool");
        const auto row_bytes = static_cast<int>(leading * static_cast<std::int32_t>(dtype_size(dtype)));
        require(row_bytes * kKvStreamPageTokens % 8 == 0, "a block's rows must be whole words");
        out.plane[out.count++] = Plane{static_cast<const unsigned char*>(h.data),
                                       static_cast<unsigned char*>(p.data),
                                       row_bytes * kKvStreamPageTokens / 8, row_bytes};
    };
    plane(host.k_pages, pool.k_pages, layout.key.data_leading_extent, layout.key.data_dtype);
    plane(host.v_pages, pool.v_pages, layout.value.data_leading_extent, layout.value.data_dtype);
    if (layout.key.has_scale()) {
        plane(host.k_scale_pages, pool.k_scale_pages, layout.key.scale_leading_extent,
              layout.key.scale_dtype);
    }
    if (layout.value.has_scale()) {
        plane(host.v_scale_pages, pool.v_scale_pages, layout.value.scale_leading_extent,
              layout.value.scale_dtype);
    }
    require(pool.block_table.data == map.table && pool.block_table.dtype == DType::I32 &&
                pool.block_table.ne[0] == map.blocks,
            "the pool's block table must be the map's table");
    return out;
}

} // namespace

std::size_t kv_stream_map_bytes(std::int32_t blocks, std::int32_t slots) {
    require(blocks > 0 && slots > 0, "a map needs blocks and slots");
    const auto words = [](std::int64_t n) { return aligned(std::size_t(n) * 4); };
    return words(blocks) + 3 * words(slots) + 2 * words(std::int64_t(slots) + kWrittenBlocks) +
           words(kControlWords);
}

KvStreamMap kv_stream_map(void* memory, std::int32_t blocks, std::int32_t slots) {
    require(memory != nullptr, "the map needs memory");
    auto* at       = static_cast<unsigned char*>(memory);
    const auto take = [&](std::int64_t n) {
        auto* p = reinterpret_cast<std::int32_t*>(at);
        at += aligned(std::size_t(n) * 4);
        return p;
    };
    KvStreamMap map;
    map.blocks     = blocks;
    map.slots      = slots;
    map.table      = take(blocks);
    map.slot_block = take(slots);
    map.slot_stamp = take(slots);
    map.slot_ref   = take(slots);
    map.copy_block = take(std::int64_t(slots) + kWrittenBlocks);
    map.copy_slot  = take(std::int64_t(slots) + kWrittenBlocks);
    map.control    = take(kControlWords);
    (void)kv_stream_map_bytes(blocks, slots);
    return map;
}

void kv_stream_reset(const KvStreamMap& map, cudaStream_t stream) {
    require(map.table != nullptr && map.blocks > 0 && map.slots > 0, "an empty map");
    const int grid = static_cast<int>(
        std::clamp<std::int64_t>((std::max(map.blocks, map.slots) + 255) / 256, 1, 1024));
    reset_kernel<<<grid, 256, 0, stream>>>(map);
    CUDA_CHECK(cudaGetLastError());
}

void kv_stream_resolve(const KvStreamMap& map, const Tensor& first_position,
                       const Tensor& selected, const Tensor& counts, const PagedKVLayerView& host,
                       const PagedKVLayerView& pool, cudaStream_t stream) {
    require(map.table != nullptr, "an empty map");
    require(selected.dtype == DType::I32 && selected.is_contiguous() && selected.data != nullptr &&
                selected.ne[0] == kTopBlocks && selected.ne[1] > 0,
            "selected must be contiguous I32 [512, tokens]");
    const std::int32_t tokens = selected.ne[1];
    require(tokens <= kKvStreamMaxTokens, "a resolved call has at most 16 queries");
    require(map.slots >= kv_stream_minimum_slots(tokens),
            "the pool cannot hold the blocks of one call");
    require(counts.dtype == DType::I32 && counts.is_contiguous() && counts.data != nullptr &&
                counts.ne[0] == tokens,
            "counts must be contiguous I32 [tokens]");
    require(first_position.dtype == DType::I32 && first_position.data != nullptr,
            "first_position must be a device I32 word");
    const Planes planes = planes_of(map, host, pool);
    resolve_kernel<<<1, kResolveThreads, 0, stream>>>(
        map, static_cast<const int*>(first_position.data), static_cast<const int*>(selected.data),
        static_cast<const int*>(counts.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    copy_kernel<<<kCopyBlocks, kCopyThreads, 0, stream>>>(map, planes);
    CUDA_CHECK(cudaGetLastError());
}

void kv_stream_evict(const KvStreamMap& map, std::int32_t first_block, std::int32_t end_block,
                     cudaStream_t stream) {
    require(map.table != nullptr, "an empty map");
    first_block = std::max(first_block, 0);
    end_block   = std::min(end_block, map.blocks);
    if (first_block >= end_block) { return; }
    const int grid = std::clamp((end_block - first_block + 255) / 256, 1, 1024);
    evict_kernel<<<grid, 256, 0, stream>>>(map, first_block, end_block);
    CUDA_CHECK(cudaGetLastError());
}

KvStreamCounters kv_stream_counters(const KvStreamMap& map, cudaStream_t stream) {
    require(map.control != nullptr, "an empty map");
    std::array<std::int32_t, kControlWords> control{};
    CUDA_CHECK(cudaMemcpyAsync(control.data(), map.control, sizeof(control),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto u64 = [&](int at) {
        return static_cast<std::uint64_t>(static_cast<std::uint32_t>(control[at])) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(control[at + 1])) << 32;
    };
    return {.calls = u64(4), .lookups = u64(6), .copies = u64(8), .overflow = control[3] != 0};
}

} // namespace ninfer::ops
