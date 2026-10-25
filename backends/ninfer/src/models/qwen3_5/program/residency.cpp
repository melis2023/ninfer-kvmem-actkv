// Model suspend: the Program's half. Persistent state (KV pages, block tables, state images, round
// state, replay records) survives a suspend in a host snapshot of its live bytes; workspace holds
// nothing between requests and is simply released. Every region keeps its addresses, so the
// tensors bound into it and the CUDA Graphs captured over it stay valid.

#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

struct ByteExtent {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
};

// The parts of `[base, base + bytes)` not covered by any of `holes`, as device ranges on `rank`.
void append_live(std::vector<DeviceSnapshot::Range>& out, std::size_t rank, void* base,
                 std::size_t bytes, std::vector<ByteExtent> holes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(base);
    const auto end   = begin + bytes;
    std::sort(holes.begin(), holes.end(),
              [](const ByteExtent& a, const ByteExtent& b) { return a.begin < b.begin; });
    std::uintptr_t cursor = begin;
    for (const ByteExtent& hole : holes) {
        const std::uintptr_t hole_begin = std::max(hole.begin, begin);
        const std::uintptr_t hole_end   = std::min(hole.end, end);
        if (hole_begin >= hole_end) { continue; }
        if (hole_begin > cursor) {
            out.push_back({rank, reinterpret_cast<void*>(cursor),
                           static_cast<std::size_t>(hole_begin - cursor)});
        }
        cursor = std::max(cursor, hole_end);
    }
    if (cursor < end) {
        out.push_back({rank, reinterpret_cast<void*>(cursor), static_cast<std::size_t>(end - cursor)});
    }
}

} // namespace

std::uint64_t ProgramImpl::device_state_backing_bytes() const noexcept {
    if (!suspendable.enabled) { return 0; }
    std::uint64_t bytes = kv_arena ? kv_arena->arena().bytes : 0;
    if (suspendable.persistent) { bytes += suspendable.persistent->reserved_bytes(); }
    if (suspendable.workspace) { bytes += suspendable.workspace->reserved_bytes(); }
    for (const VmmRegion& region : suspendable.persistent_by_rank) {
        bytes += region.reserved_bytes();
    }
    for (const VmmRegion& region : suspendable.workspace_by_rank) {
        bytes += region.reserved_bytes();
    }
    return bytes;
}

std::vector<DeviceSnapshot::Range> ProgramImpl::live_persistent_ranges() const {
    // Free KV pages are written before they are read, so a suspend skips them; on an idle server
    // they are most of the persistent arena.
    std::vector<std::vector<ByteExtent>> holes(device.size());
    const auto add_holes = [&](const DeviceKVPagePool& pool) {
        for (const auto& extent : pool.free_extents()) {
            const auto begin = reinterpret_cast<std::uintptr_t>(extent.base);
            holes.at(extent.rank).push_back({begin, begin + extent.bytes});
        }
    };
    add_holes(decoder->text_kv.page_pool());
    if (decoder->mtp_kv) { add_holes(decoder->mtp_kv->page_pool()); }

    std::vector<DeviceSnapshot::Range> out;
    append_live(out, 0, persistent.base(), persistent.capacity(), holes[0]);
    for (std::size_t index = 0; index < persistent_by_rank.size(); ++index) {
        const DeviceArena& arena = persistent_by_rank[index];
        append_live(out, index + 1, arena.base(), arena.capacity(), holes.at(index + 1));
    }
    return out;
}

void ProgramImpl::release_device_state_backing() {
    if (kv_arena) { kv_arena->release_backing(); }
    if (suspendable.persistent) { suspendable.persistent->release_backing(); }
    for (VmmRegion& region : suspendable.persistent_by_rank) { region.release_backing(); }
    if (suspendable.workspace) { suspendable.workspace->release_backing(); }
    for (VmmRegion& region : suspendable.workspace_by_rank) { region.release_backing(); }
}

void ProgramImpl::restore_device_state_backing() {
    try {
        if (kv_arena) {
            const RankBinding bound(device, 0);
            kv_arena->restore_backing();
        }
        if (suspendable.persistent) {
            const RankBinding bound(device, 0);
            suspendable.persistent->restore_backing();
        }
        for (std::size_t index = 0; index < suspendable.persistent_by_rank.size(); ++index) {
            const RankBinding bound(device, index + 1);
            suspendable.persistent_by_rank[index].restore_backing();
        }
        if (suspendable.workspace) {
            const RankBinding bound(device, 0);
            suspendable.workspace->restore_backing();
        }
        for (std::size_t index = 0; index < suspendable.workspace_by_rank.size(); ++index) {
            const RankBinding bound(device, index + 1);
            suspendable.workspace_by_rank[index].restore_backing();
        }
    } catch (...) {
        // A partial restore holds memory the device may need for the retry; give all of it back.
        try {
            release_device_state_backing();
        } catch (...) {}
        throw;
    }
}

DeviceSnapshot::Stats ProgramImpl::suspend_device_state(DeviceSnapshot& snapshot) {
    if (!suspendable.enabled) {
        throw std::logic_error("Qwen3.5 Program was not built with suspendable device memory");
    }
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty || request.prefill) {
            throw std::logic_error("model suspend requires an idle Program");
        }
    }
    if (has_context_transaction() || pressure_planning_active_ ||
        (vision_broker && vision_broker->window_open())) {
        throw std::logic_error("model suspend requires an idle Program");
    }
    if (disk_kv) { disk_kv->wait_idle(); }
    // Every stream of every rank, transfers and Vision included, not only the compute streams.
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        const RankBinding bound(device, rank);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const std::vector<DeviceSnapshot::Range> ranges = live_persistent_ranges();
    const DeviceSnapshot::Stats stats               = snapshot.capture(device, ranges);
    release_device_state_backing();
    return stats;
}

DeviceSnapshot::Stats ProgramImpl::resume_device_state(DeviceSnapshot& snapshot) {
    if (!suspendable.enabled) {
        throw std::logic_error("Qwen3.5 Program was not built with suspendable device memory");
    }
    restore_device_state_backing();
    DeviceSnapshot::Stats stats;
    try {
        stats = snapshot.restore(device);
    } catch (...) {
        try {
            release_device_state_backing();
        } catch (...) {}
        throw;
    }
    snapshot.clear();
    logprobs_armed.reset();
    return stats;
}

} // namespace ninfer::models::qwen3_5::detail
