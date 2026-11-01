#pragma once

// Memory and resource summaries: arenas, device and host memory, the media cache, host work, the
// n-gram table and expert residency.

#include "ninfer/kv_cache_storage.h"
#include "ninfer/model_enums.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer {

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Vision workspace extents and separate weight residency. Workspace extents describe regions
// within workspace.capacity_bytes; CPU/overlay Host weights are separate allocations.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
    VisionResidency residency             = VisionResidency::Resident;
    std::size_t window_capacity_bytes     = 0; // overlay: device bytes one window borrows
    std::size_t pinned_weight_bytes       = 0; // overlay: host-pinned tower bytes
    std::size_t mirror_bytes              = 0; // overlay: pinned mirror of the borrowable tail
    bool cpu_weights_lazy                = false;
    bool cpu_weights_loading             = false;
    std::size_t cpu_weight_bytes          = 0;
    std::size_t cpu_weight_users          = 0;
    std::uint64_t cpu_weight_loads        = 0, cpu_weight_unloads = 0;
    double cpu_weight_last_load_seconds   = 0;
    std::uint32_t cpu_weight_idle_seconds = 0;

};

// One device of a pipeline: what its weights, its layers' persistent state, its scratch and its
// expert cache hold there.
struct DeviceMemorySummary {
    int device = 0;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::size_t expert_cache_bytes = 0;
};

// Latest observed process residency, not a permanent physical-pinning guarantee. Intentional
// pinned Host allocations are established before shared_baseline_bytes; device-driven growth
// above that baseline invalidates a strict candidate. Zero counters while disabled are unknown.
struct CudaResidencySummary {
    bool enabled = false;
    bool verified = false;
    std::size_t cuda_free_bytes = 0;
    std::size_t cuda_total_bytes = 0;
    std::size_t dedicated_bytes = 0;
    std::size_t shared_bytes = 0;
    std::size_t shared_baseline_bytes = 0;
    std::size_t device_allocated_bytes = 0;
    std::size_t host_pool_bytes = 0;
    std::size_t host_used_bytes = 0;
    std::size_t verified_reserve_bytes = 0;
};

// Gauges of the current KVMem history (active or retained), not process-lifetime
// counters. Payload H excludes statistics, checkpoint images and native workspace.
struct KvmemMemorySummary {
    bool enabled = false;
    std::uint32_t selected_tokens = 0, reserve_tokens = 0;
    std::uint64_t host_payload_budget_bytes = 0, host_payload_bytes = 0;
    std::uint64_t disk_payload_budget_bytes = 0, disk_payload_bytes = 0;
    std::uint64_t retained_payload_budget_bytes = 0;
    std::uint64_t disk_read_bytes = 0, disk_written_bytes = 0;
    std::uint64_t statistics_bytes = 0, checkpoint_bytes = 0;
    std::uint64_t transfer_staging_bytes = 0;
    std::uint64_t statistics_staging_bytes = 0;
    std::uint32_t mtp_resident_pages = 0;
    std::uint32_t retained_sessions = 0, active_sessions = 0, retained_sessions_limit = 0;
    std::uint64_t history_hits = 0, history_misses = 0, history_evictions = 0;
    std::uint64_t disk_hits = 0, disk_writes = 0, disk_errors = 0;
    std::uint64_t active_payload_reservation_bytes = 0;
    std::uint32_t evaluated_tokens = 0, checkpoint_tokens = 0, resident_pages = 0;
    std::uint32_t history_swaps = 0;
    std::uint64_t history_spilled_bytes = 0, history_restored_bytes = 0;
};

struct MemorySummary {
    KvmemMemorySummary kvmem;
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Device memory CUDA Graph preparation actually took at startup (free memory before minus
    // after instantiating, uploading and launching every executable); 0 without CUDA Graphs.
    // Empty when the device had no free memory left after preparation (WDDM can hand out more
    // than the card holds), which leaves the difference only a lower bound.
    std::optional<std::size_t> cuda_graph_measured_bytes = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::uint32_t host_state_capacity_slots       = 0;
    std::uint32_t host_state_occupied_slots       = 0;
    std::size_t host_kv_capacity_bytes            = 0;
    std::size_t host_kv_occupied_bytes            = 0;
    // Host retention-tier unit costs. Every Host StateImage slot pins image_bytes regardless of
    // the prefix depth it resumes, and Host KV pins bytes per page group, so the split of a
    // budget between the two is the depth-versus-positions trade made visible.
    std::size_t host_state_image_bytes   = 0;
    std::size_t host_kv_page_group_bytes = 0;
    // Engaged only when the single host RAM budget mode is active.
    std::size_t host_cache_budget_bytes = 0;
    CudaResidencySummary cuda_residency;
    // Host or disk routed experts: persistent device cache bytes, over every device. Native
    // Qwen delivery windows are counted separately in workspace.
    std::size_t expert_cache_bytes = 0;
    // A pipeline over several devices: each device in stage order, the primary one (which the
    // single-device fields above describe) first. Empty on one device.
    std::vector<DeviceMemorySummary> devices;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t program_submit_ns       = 0;
    std::uint64_t program_post_ns         = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t device_wait_ns          = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t admission_policy_ns           = 0;
    std::uint64_t context_progress_ns           = 0;
    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t admission_policy_invocations  = 0;
    std::uint64_t context_progress_invocations  = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Qwen3.8-Flash-Next's n-gram table reads, monotonic. A batch is the rows of one pass; its latency
// runs from the pass's request to its last row, and latency_histogram[b] counts the batches that
// took [2^(b-1), 2^b) microseconds (b = 0: under one). A stall is a pass whose stage waited for
// the rows after the layers before the first PLE layer, and stall_seconds is the device time it
// waited.
struct NgramTableStats {
    static constexpr std::size_t kLatencyBuckets = 24;
    std::uint64_t rows                           = 0; // rows the passes addressed
    std::uint64_t resident_rows                  = 0; // of them, served from RAM
    std::uint64_t batches                        = 0;
    double read_seconds                          = 0.0; // latency summed over the batches
    std::array<std::uint64_t, kLatencyBuckets> latency_histogram{};
    std::uint64_t stalls = 0;
    double stall_seconds = 0.0;
};

// Routed experts with host or disk residency, monotonic. A route is one
// (token, expert) pair a pass took; a hit found its expert in a device slot. transferred_bytes
// counts the expert bytes copied to the devices (misses and cache admissions); cpu_routes the
// pairs computed by the CPU.
struct ExpertResidencyStats {
    std::uint64_t routes            = 0;
    std::uint64_t hits              = 0;
    std::uint64_t cpu_routes        = 0;
    std::uint64_t admitted          = 0;
    std::uint64_t transferred_bytes = 0;
    std::uint32_t slots             = 0;
};

} // namespace ninfer
