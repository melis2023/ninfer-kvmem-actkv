// Model suspend's memory primitives on a real device: a fixed-address region keeps its addresses
// and its captured graphs across a release and a restore; the eviction pools give their memory back
// and take it again; a device snapshot carries the live bytes of several ranges (including ranges
// longer than one staging slot) out to pageable or pinned host memory and back; a KV page pool
// reports exactly the bytes of its free pages.

#include "core/arena.h"
#include "core/device.h"
#include "core/device_snapshot.h"
#include "core/evictable_kv_pool.h"
#include "core/evictable_weight_pool.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/vmm.h"
#include "cuda_availability.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::test::cuda_unavailable;

int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

__global__ void stamp(std::uint32_t* words, std::size_t count, std::uint32_t seed) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        words[index] = seed ^ static_cast<std::uint32_t>(index * 2654435761U);
    }
}

void fill(void* destination, std::size_t bytes, std::uint32_t seed, cudaStream_t stream = nullptr) {
    stamp<<<256, 256, 0, stream>>>(static_cast<std::uint32_t*>(destination), bytes / 4, seed);
    CUDA_CHECK(cudaGetLastError());
}

bool holds(const void* source, std::size_t bytes, std::uint32_t seed) {
    std::vector<std::uint32_t> words(bytes / 4);
    CUDA_CHECK(cudaMemcpy(words.data(), source, bytes, cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < words.size(); ++index) {
        if (words[index] != (seed ^ static_cast<std::uint32_t>(index * 2654435761U))) {
            return false;
        }
    }
    return true;
}

void region_keeps_addresses_and_graphs() {
    constexpr std::size_t kBytes = 300ULL * 1024 * 1024; // spans two 256 MiB pieces
    ninfer::VmmRegion region(kBytes, ninfer::VmmRegion::Options{.device = 0});
    const ninfer::DeviceSpan before = region.span();
    expect(region.backed() && before.bytes == kBytes, "region starts backed");

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    fill(before.data, kBytes, 0xA5A5U, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    expect(holds(before.data, kBytes, 0xA5A5U), "graph wrote the region");

    region.release_backing();
    expect(!region.backed(), "release unbacks");
    region.release_backing(); // idempotent
    region.restore_backing();
    expect(region.backed() && region.span().data == before.data, "restore keeps the address");

    CUDA_CHECK(cudaMemset(before.data, 0, kBytes));
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    expect(holds(before.data, kBytes, 0xA5A5U), "graph captured before the release still runs");

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void pools_release_and_restore(ninfer::DeviceContext& device) {
    constexpr std::size_t kArena = 96ULL * 1024 * 1024;
    {
        ninfer::EvictableWeightPool pool(
            device, {.arena_bytes = kArena, .evictable_tail_bytes = 32ULL * 1024 * 1024});
        const auto arena = pool.arena();
        pool.release_backing();
        expect(!pool.backed(), "weight pool releases");
        pool.restore_backing();
        expect(pool.backed() && pool.arena().data == arena.data, "weight pool keeps its address");
        fill(arena.data, kArena, 7U);
        CUDA_CHECK(cudaDeviceSynchronize());
        expect(holds(arena.data, kArena, 7U), "restored weight pool is writable");
    }
    {
        const std::size_t granule = ninfer::vmm::granularity(0);
        ninfer::EvictableKVPool pool(device, {.arena_bytes           = kArena,
                                              .lendable_prefix_bytes = kArena / 2,
                                              .window_capacity_bytes = granule});
        const auto arena = pool.arena();
        pool.release_backing();
        expect(!pool.backed(), "KV pool releases");
        pool.restore_backing();
        expect(pool.backed() && pool.arena().data == arena.data, "KV pool keeps its address");
        fill(arena.data, kArena, 9U);
        CUDA_CHECK(cudaDeviceSynchronize());
        expect(holds(arena.data, kArena, 9U), "restored KV pool is writable");
        // A lease still works after a restore.
        const std::size_t first = 0;
        auto lease              = pool.lease(std::span<const std::size_t>(&first, 1), nullptr);
        expect(lease.open(), "KV pool lends after a restore");
        lease.close();
    }
}

void snapshot_round_trip(ninfer::DeviceContext& device, ninfer::DeviceSnapshot::Memory memory) {
    constexpr std::size_t kBytes = 200ULL * 1024 * 1024;
    ninfer::VmmRegion region(kBytes, ninfer::VmmRegion::Options{.device = 0});
    auto* base = static_cast<std::byte*>(region.span().data);
    fill(base, kBytes, 0x1234U);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Three live ranges, one longer than a staging slot; the holes between them are not kept.
    const std::vector<ninfer::DeviceSnapshot::Range> ranges{
        {0, base, 4096},
        {0, base + 1024 * 1024, 100ULL * 1024 * 1024},
        {0, base + 150ULL * 1024 * 1024, 50ULL * 1024 * 1024},
    };
    std::size_t live = 0;
    for (const auto& range : ranges) { live += range.bytes; }
    const std::size_t capacity = memory == ninfer::DeviceSnapshot::Memory::Pinned ? live : 0;
    ninfer::DeviceSnapshot snapshot(memory, capacity);
    const auto captured = snapshot.capture(device, ranges);
    expect(captured.bytes == live && snapshot.bytes() == live, "snapshot holds the live bytes");

    region.release_backing();
    region.restore_backing();
    CUDA_CHECK(cudaMemset(base, 0, kBytes));
    // The restore copies on the transfer stream, which does not wait for the legacy stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto restored = snapshot.restore(device);
    expect(restored.bytes == live, "restore copies every live byte back");
    std::cout << (memory == ninfer::DeviceSnapshot::Memory::Pinned ? "pinned" : "pageable")
              << ": " << live / (1 << 20) << " MiB captured in " << captured.seconds * 1e3
              << " ms, restored in " << restored.seconds * 1e3 << " ms\n";
    // The kept ranges hold their original words again; check them through a whole-region read.
    std::vector<std::uint32_t> words(kBytes / 4);
    CUDA_CHECK(cudaMemcpy(words.data(), base, kBytes, cudaMemcpyDeviceToHost));
    bool kept = true;
    bool holes_zero = true;
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::size_t offset = index * 4;
        bool live_word           = false;
        for (const auto& range : ranges) {
            const std::size_t begin = static_cast<std::byte*>(range.address) - base;
            live_word = live_word || (offset >= begin && offset < begin + range.bytes);
        }
        const std::uint32_t expected = 0x1234U ^ static_cast<std::uint32_t>(index * 2654435761U);
        if (live_word) {
            kept = kept && words[index] == expected;
        } else {
            holes_zero = holes_zero && words[index] == 0;
        }
    }
    expect(kept, "live ranges survive the release");
    expect(holes_zero, "holes are not restored");
    expect(!snapshot.empty(), "a restore keeps the host copy until cleared");
    snapshot.clear();
    expect(snapshot.empty() && snapshot.bytes() == 0, "clear empties the snapshot");
    if (memory == ninfer::DeviceSnapshot::Memory::Pinned) {
        bool refused = false;
        try {
            const std::vector<ninfer::DeviceSnapshot::Range> too_much{{0, base, live + 4096}};
            (void)snapshot.capture(device, too_much);
        } catch (const std::runtime_error&) {
            refused = true;
        }
        expect(refused && snapshot.empty(), "a pinned snapshot refuses more than it reserved");
    }
}

void free_extents_cover_free_pages() {
    const ninfer::DeviceKVPagePoolSpec spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 128, .head_extent = 2},
                           {.dtype = ninfer::DType::BF16, .leading_extent = 128, .head_extent = 2}},
            },
    };
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_device_kv_page_pool(builder, spec);
    ninfer::DeviceArena storage(builder.finish(256));
    ninfer::DeviceKVPagePool pool(ninfer::DeviceSpan{storage.base(), storage.capacity()}, layout);
    auto all = pool.free_extents();
    std::size_t free_bytes = 0;
    for (const auto& extent : all) { free_bytes += extent.bytes; }
    expect(all.size() == 2 && free_bytes == layout.payload_bytes(),
           "an empty pool's free extents are every plane whole");

    auto reservation = pool.reserve(3);
    expect(reservation.has_value(), "reserve three pages");
    std::vector<ninfer::DeviceKVPageLease> pages;
    pages.reserve(3); // materialize never allocates host memory
    pool.materialize(*reservation, 3, pages);
    const auto partial = pool.free_extents();
    std::size_t partial_bytes = 0;
    for (const auto& extent : partial) { partial_bytes += extent.bytes; }
    const std::size_t page_bytes = layout.payload_bytes() / 8;
    expect(partial_bytes == 5 * page_bytes, "free extents shrink by the materialized pages");
    pool.dematerialize(*reservation, 0, pages);
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (!ninfer::vmm::supported(0)) {
        std::cout << "SKIP: device 0 has no virtual memory management\n";
        return 77;
    }
    try {
        ninfer::DeviceContext device(0);
        region_keeps_addresses_and_graphs();
        pools_release_and_restore(device);
        snapshot_round_trip(device, ninfer::DeviceSnapshot::Memory::Pageable);
        snapshot_round_trip(device, ninfer::DeviceSnapshot::Memory::Pinned);
        free_extents_cover_free_pages();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "suspend memory primitives: ok\n";
    return 0;
}
