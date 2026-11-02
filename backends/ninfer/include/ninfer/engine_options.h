#pragma once

// Engine startup options: KV capacity, speculative decoding, diagnostics, the context cache,
// model suspension, expert and n-gram residency, and EngineOptions itself.

#include "ninfer/base.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/model_enums.h"
#include "ninfer/startup.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninfer {

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP, DFlash and DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    // Flash-Next MTP: end drafting and verification at the first draft whose absolute probability
    // is at or below this floor. Zero keeps the full window.
    float draft_min_p = 0;
    // An Engine's options() report the resolved head.
    ProposalHead proposal_head = ProposalHead::Auto;
    // Context-lookup drafting: match this many trailing tokens against the sequence so far and
    // propose whatever followed the last time they appeared. 0 disables it. It costs no device
    // work, it is exact (verify rejects a wrong guess), and it is strongest exactly where a draft
    // head is weakest -- output that repeats the input. Used as a draft source alongside the
    // configured backend, preferred whenever it finds a match.
    std::uint32_t lookup_ngram = 0;
    // MTP only; Adaptive verifies 1..draft_tokens drafts per round.
    MtpDraftPolicy mtp_policy = MtpDraftPolicy::Fixed;
    // Optional target-verified copy proposals with MTP, DFlash or DFlash2.
    // Engine clamps valid widths above 15 to 15 when max_concurrency > 1, at construction.
    // Zero disables the proposer; enabled draft width is 1..63 and minimum match 4..64.
    std::uint32_t ngram_draft_tokens = 0;
    std::uint32_t ngram_min_match    = 12;
    // CPU-only retention, separate from KV. Zero keeps request-local drafting.
    std::size_t ngram_archive_bytes = 0;
    std::size_t ngram_session_bytes = 128ULL << 20;
    // MTP only: the draft head attends to its first 64 keys and the newest this many keys before
    // its query instead of the whole history; target verification keeps full attention, so the
    // committed tokens do not depend on it. Zero reads the whole history.
    std::uint32_t mtp_attention_window = 0;
    // The backend was left to the Engine (no --spec): a MoE model with host or disk experts whose
    // artifact carries its MTP block or assistant (Qwen3.8-Flash-Next, Gemma 4) drafts with it,
    // kAutomaticDraftTokens a round, since a verification window's routes share the layer's
    // missing experts (Flash-Next ending a round's drafting at its first draft at or below
    // kAutomaticDraftMinP); others run without drafts. The Engine's options() report the resolved
    // backend.
    bool automatic = false;
    static constexpr std::uint32_t kAutomaticDraftTokens = 3;
    static constexpr float kAutomaticDraftMinP           = 0.5F;
};

enum class DiagnosticLevel : std::uint8_t {
    Debug,
    Info,
    Warning,
    Error,
};

// One runtime diagnostic from the Engine: the device route profile it installs, recovery from out
// of memory or a failed worker step, and context stores rebuilt after that recovery.
struct Diagnostic {
    DiagnosticLevel level = DiagnosticLevel::Info;
    std::string message;
};

struct DiagnosticObserver {
    // Receives every diagnostic, from the worker thread or the constructing one; the product
    // decides which levels to show. Without a callback, Info and above go to stderr. Callback
    // exceptions are ignored so a logging failure cannot disturb execution.
    std::function<void(const Diagnostic& diagnostic)> callback;
};

// Prefix-cache implementation selected at Engine construction. Legacy is the owner/checkpoint
// ResourceManager (docs/maintainer/resource-scheduling-and-context-cache.md). Hybrid is the
// content-addressed block tree with sparse state snapshots
// (docs/maintainer/hybrid-prefix-cache-spec.md).
enum class ContextCacheMode : std::uint8_t {
    Legacy,
    Hybrid,
};

// Hybrid-mode tuning. Every field is optional: Engine construction derives the unset ones from
// max_concurrency, prefill_chunk and whether a Host tier exists (host_cache_budget_bytes, default
// kDefaultHybridHostCacheBytes, 0 disables it), and Engine::options() reports the effective
// values. Legacy-only fields of ContextCacheOptions are rejected in Hybrid mode.
// Abandons the Host tier save of a stopping Engine (HybridPrefixCacheOptions::persistent_file)
// from any thread, also while the Engine is being destroyed. Copies share one state: the product
// keeps a copy from the options it passes, beyond the Engine's lifetime.
class PrefixCacheSaveControl {
public:
    enum class Abandon : std::uint8_t {
        // Nothing replaced the previous file; a save in progress deleted its unfinished file.
        Unsaved,
        // The save had already completed.
        Saved,
        // The save did not stop within the timeout; its unfinished file may remain.
        StillWriting,
    };

    PrefixCacheSaveControl();

    // Stops a save in progress, or one not begun yet, from replacing the previous file. Waits at
    // most `timeout` for a save in progress to stop and delete its unfinished file.
    Abandon abandon(std::chrono::milliseconds timeout) const noexcept;

    // The Engine's side. A save writes only after begin() returns true, polls abandoned() while
    // it writes, and then calls end() with whether it replaced the file.
    [[nodiscard]] bool begin() const noexcept;
    [[nodiscard]] bool abandoned() const noexcept;
    void end(bool saved) const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
};

struct HybridPrefixCacheOptions {
    // Device StateImage slots holding inactive snapshots (and tap/endpoint staging). Total Device
    // StateImage capacity is max_concurrency + device_snapshot_slots. Default: one per request
    // lane plus one staging slot with a Host tier, plus two without one (Device slots are then
    // the only snapshot storage).
    std::optional<std::uint32_t> device_snapshot_slots;
    // New prefill state snapshots one request may create. Default 8 with a Host tier, 2 without.
    std::optional<std::uint32_t> max_new_taps;
    // Geometric ladder base G: ladder taps target prompt_tokens - G * 2^k. Default
    // max(4096, 2 * prefill_chunk): ladder taps land on prefill chunk boundaries.
    std::optional<std::uint32_t> tap_ladder_tokens;
    // Ladder taps closer than this to another snapshot on the same path are skipped. Default
    // max(1024, prefill_chunk).
    std::optional<std::uint32_t> tap_min_gap_tokens;
    // Opt-in persistence of the Host tier: saved to this file when the Engine shuts down cleanly
    // and restored from it at startup, but only when it was written for the same artifact, KV and
    // state formats and `persistent_identity` (the product binary's build). Empty disables it.
    std::filesystem::path persistent_file;
    std::string persistent_identity;
    // Lets the product abandon the save at shutdown (ninfer-serve: Ctrl+C during the stop).
    PrefixCacheSaveControl persistent_save;
};

// One spill of an involuntarily evicted session to its slot file, reported from the writer thread.
struct SlotAutoSaveEvent {
    std::string path;
    std::uint32_t tokens = 0;
    std::uint64_t bytes  = 0;
    double seconds       = 0.0;
    // Empty on success.
    std::string error;
    // Set when the spill was skipped because the file already holds a deeper snapshot of the
    // session; the value is that depth.
    std::optional<std::uint32_t> skipped_behind_tokens;
    // Set when the spill was skipped because an explicit save, restore or erase of the path
    // happened after it was queued.
    bool superseded = false;
};

struct SlotAutoSaveOptions {
    // Before an involuntary eviction destroys a retained session that was last saved to or
    // restored from a slot file, snapshot it and write it back to that file off-thread.
    bool enabled = false;
    // Called on the writer thread after each spill. Exceptions are ignored.
    std::function<void(const SlotAutoSaveEvent& event)> listener;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=C, R=8, Host KV=8 GiB, P=2C, S=max(C,7) and L=2 (4 with automatic anchors);
    // Engine::options() returns those effective values.
    bool enabled = true;
    ContextCacheMode mode = ContextCacheMode::Legacy;
    HybridPrefixCacheOptions hybrid;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // Single host RAM ceiling for the whole retention tier. In Hybrid mode it is the pinned Host
    // slab pool that KV blocks and state snapshots share (default kDefaultHybridHostCacheBytes, 0
    // disables the Host tier). In Legacy mode, when engaged it is authoritative:
    // the plan sizes the Host state pool from the checkpoint inventory the capture path creates
    // (2 + anchors per private owner plus one per shared entry), spends the remaining state
    // headroom under the half-budget cap on extra long anchors per owner, gives Host KV the
    // remainder, and rejects a plan whose state footprint exceeds half the budget.
    // `host_state_slots` and `host_kv_capacity_bytes` are ignored in that mode.
    std::optional<std::size_t> host_cache_budget_bytes;
    // --host-cache-mib auto: host_cache_budget_bytes is sized at startup, once the weights are
    // loaded, from the host memory still available (the smaller of the system's available memory
    // and what the process's memory cgroup allows) less host_cache_reserve_bytes and, with Vision,
    // the media caches; never above host_cache_max_bytes nor host_cache_percent of the machine's
    // memory. For a machine that serves this one process: pinned pages cannot be reclaimed.
    bool host_cache_auto                  = false;
    std::size_t host_cache_reserve_bytes  = std::size_t{3} << 30;
    std::optional<std::size_t> host_cache_max_bytes;
    std::optional<std::uint32_t> host_cache_percent;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count. An engaged
    // host-cache budget raises the anchor count within the state inventory it funds.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
    // Input-complexity bound; this does not reserve checkpoint storage.
    std::optional<std::uint32_t> max_cache_markers_per_request;
    // Disk (L3) tier under the Host tier. An empty path disables it. With a path, an evicted
    // private continuation writes its KV prefix chain and its endpoint, rewrite and early anchor
    // StateImages to per-family files there, keyed by the prefix digest, and they persist across
    // restarts; the Engine keeps them apart per artifact and execution profile.
    std::filesystem::path disk_kv_path;
    std::uint64_t disk_kv_capacity_bytes = 0; // zero with a path selects 64 GiB
    // Read side: a request with no resident prefix that finds [0, E) restorable on disk seeds
    // those KV pages and the StateImage at E and prefills only [E, prompt). Off leaves the tier
    // write-only.
    bool disk_kv_restore = false;
    // Restore reads go through Microsoft DirectStorage (Windows builds with NINFER_DIRECTSTORAGE);
    // any other build refuses it at load. A failed batch falls back to mapped reads.
    bool disk_kv_directstorage = false;
    // Retention for one long-lived conversation whose prompt only grows: a capture that extends a
    // resident the request matched exactly at that resident's frontier inherits the resident's
    // demand, so rolling the frontier forward is not valued against its own history. Off by
    // default: when conversations share a prefix, one conversation's extension can evict the
    // prefix the others depend on.
    bool rolling_retention = false;
    // A private checkpoint of a conversation whose next prompt diverges from it at the checkpoint's
    // frontier keeps no retention value, so it is the first to go when the cache needs room. Off
    // by default: a client that switches back to an earlier branch of the conversation, or another
    // conversation that shares that prefix, could still have reused it.
    bool release_diverged_checkpoints = false;
    // Admission searches for a reuse plan for up to 250 ms at every boundary, and a request whose
    // plan is worth more earns a longer base grant, instead of 50 ms at an idle boundary and 10 ms
    // while other requests run. Off by default: a new request can then pause running decode for
    // up to that long, where a missed large prefix costs its whole re-prefill.
    bool thorough_admission_search = false;
    // Pressure ranks private conversations and shared prefixes in one least-recently-used order
    // (latest hit or publication). A request that does not fit gives up the fewest oldest owners
    // that make it fit and spares any of them it does not need, instead of clearing every owner
    // it cannot reuse; kept owners, shared prefixes included, are demoted to Host where it has
    // room, and incremental search may fully evict only inside that sacrificed tail and only an
    // owner Host cannot take. Off by default: the economic search alone chooses what goes.
    bool recency_eviction = false;
    // Pressure search charges evicting a private conversation its rank by rebuild cost among the
    // request's victims (0 for the cheapest), so among otherwise equal plans the search keeps the
    // conversations most expensive to rebuild, demoted to Host, and evicts the cheapest. Off by
    // default: it reorders the bounded search, which can then settle on a different plan.
    bool value_aware_demote = false;
    // An active request's Device KV entitlement is its prompt plus a bounded window of its output
    // (4096 tokens, or the prefill chunk when larger), extended at decode-round boundaries, rather
    // than prompt plus the whole max_tokens budget. When the pool cannot extend it, idle retained
    // owners are released least recently used first; if the smallest step still does not fit, the
    // answer ends with finish_reason length before max_tokens. Off by default for that reason.
    bool kv_lease_growth = false;
    // The engine anchors message boundaries itself: every request offers private long anchors at
    // up to L message boundaries, on the grid long_anchor_min_spacing_tokens sets, so a later
    // request that rewrites history there resumes from the anchor instead of root. L then defaults
    // to 4. Off by default: each anchor costs a prefill split and a StateImage whether or not the
    // client ever rewrites its history.
    bool automatic_long_anchors = false;
    // Minimum token gap between automatic long anchors, doubling per anchor walking back from the
    // prompt end (gap k >= spacing * 2^k), so the grid is sparse near the end and still reaches
    // deep history. Zero anchors every one of the last L message boundaries.
    std::uint32_t long_anchor_min_spacing_tokens = 1024;
    // Branch anchors: before a request is planned, the engine finds the deepest point at which its
    // prompt matches a retained continuation or shared prefix token for token, and when no
    // checkpoint of that owner lies at or near that depth, the request captures a private long
    // anchor there, so the next request that diverges at the same point resumes from it instead of
    // from an older checkpoint. Costs a scan of the catalog per planned request and, where it
    // fires, a prefill split and a StateImage. Off by default.
    bool branch_anchors = false;
    // Endpoint anchors: a request that resumes a retained conversation at its endpoint and
    // extends it keeps that endpoint as a private long anchor, taken from the resumed state
    // before the new suffix is prefilled. Resuming moves the endpoint with the conversation, so a
    // later request that diverges there -- another reply to the same answer, an edited last
    // message -- would otherwise find no checkpoint between root and the new turn. Fires where
    // the endpoint lies at least 1024 tokens above the conversation's deeper long anchors. Costs
    // a StateImage per firing, a Host copy when the Device slots are full, within the long-anchor
    // capacity; no prefill split. A request that diverges at any private long anchor branches off
    // the conversation where the cache holds the branch beside it as it stands, and otherwise
    // rewrites the conversation from the anchor, as a rewrite restore does from its checkpoint.
    bool endpoint_anchors = true;
    // Progress anchors: every prompt offers a private long anchor at each multiple of this many
    // tokens strictly inside it, so a prompt with no message boundary (one very long user message)
    // still holds checkpoints while it prefills. A prefill that is cancelled part way -- a client
    // that times out or disconnects -- keeps them when its live state cannot be kept as an
    // endpoint (under MTP until its last chunk), so the retry resumes from the deepest one instead
    // of token zero. They share the long-anchor capacity with every other long anchor; each costs
    // a StateImage and may split a prefill chunk. 0 disables; a nonzero value below
    // kMinimumProgressAnchorTokens is rejected.
    std::uint32_t progress_anchor_tokens = 16384;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

// A phantom-kv prompt graft: a hidden conversation prefix (e.g. a system turn plus an assistant
// acknowledgement) that sits in front of every request selecting it. `path` names the safetensors
// container; its metadata sidecar is the same path with a .json extension. The graft's tokens occupy
// positions [0, n) and the request's own rendered prompt starts at position n, exactly as if the
// hidden turn had been sent as text. They count toward max_context and the reported prompt tokens.
struct GraftSource {
    std::string name;
    std::filesystem::path path;
};

// Where a suspended model's retained device state waits in host memory.
enum class SuspendSnapshotMemory : std::uint8_t {
    // Ordinary memory allocated at suspend, sized to the live bytes, freed after the resume.
    Pageable,
    // One page-locked block reserved at startup for the largest snapshot: transfers at bus speed,
    // at the cost of that much host memory for the life of the Engine.
    Pinned,
};

// Where a resume takes the weights from.
enum class SuspendWeightSource : std::uint8_t {
    // Read again from the artifact the Engine loaded, whose files stay open while it runs.
    Artifact,
    // A host copy taken at suspend, kept in the snapshot memory: as much host memory as the device
    // weights while suspended, in exchange for a resume at bus speed.
    Host,
};

struct ModelSuspendOptions {
    // Engine::suspend and Engine::resume: every device allocation the model owns sits at fixed
    // addresses whose physical memory a suspend gives back to the device while captured graphs and
    // tensor views stay valid. Needs CUDA virtual memory management on every device.
    bool enabled                          = false;
    SuspendSnapshotMemory snapshot_memory = SuspendSnapshotMemory::Pageable;
    SuspendWeightSource weights           = SuspendWeightSource::Artifact;
    // Default for a suspend that does not say: a generation request arriving while suspended
    // resumes the model and then runs, instead of failing as Unavailable.
    bool auto_resume = true;
};

// Host and disk experts: how a decode or verification call gets the routed experts the device
// cache lacks. The GPU copies a share of them and the CPU computes the rest: for GGUF banks
// (Flash-Next, Gemma 4, Flint) and Qwen3.5/3.6 MoE experts the GPU's results bit for bit, for
// Flash-Next native banks different numerical results (and concurrency one).
struct HybridExpertOptions {
    // Host or disk experts: the share 0..1 of a decode or verification call's missing experts the
    // GPU copies; the CPU computes the rest. Unset (default): where the CPU's products are the
    // GPU's bit for bit (GGUF and Qwen3.5/3.6 MoE experts), each call splits its misses by
    // measured CPU and bus times; Flash-Next native experts copy every miss.
    std::optional<float> dma_share;
    std::uint32_t cpu_threads = 0; // automatic (one per physical core, half the CPUs), or 1..256
    // GGUF banks: copy missing experts into device staging while the cached ones run (staged), or
    // let the expert kernels read host experts across the bus (mapped; host experts only).
    bool mapped_misses = false;
    // GGUF host experts with staged misses: 1..9 leaves out of a decode or verification call each
    // uncached expert that all its tokens rank past this many places of their top ten, without
    // giving its router weight to the others (lossy); 0 (default) serves every miss.
    std::uint32_t tail_skip_rank = 0;
    // GGUF host experts: device buffers a wide prefill call copies its layer's uncached experts
    // into. 2 copies the next layer's while this layer's experts run, and takes the second
    // buffer's memory from the expert cache; 1 keeps one. 0 (default) takes two with the automatic
    // expert cache, which holds both between prompts, and one with an explicit cache size. Two
    // fall back to one when half the free memory does not hold them.
    std::uint32_t prefill_buffers = 0;
    bool adaptive_cache = false;
    std::filesystem::path routing_profile; // initial per-layer expert counts for this artifact
    std::filesystem::path record_profile; // write accumulated counts at request boundaries
};

// The n-gram table of Qwen3.8-Flash-Next's per-layer embedding: the model artifact's own rows, or
// a separate table artifact holding the table the model names by digest.
struct NgramTableOptions {
    // A table artifact to read the rows from; empty takes the model artifact's own.
    std::filesystem::path path;
    NgramResidency residency = NgramResidency::Disk;
    NgramIo io               = NgramIo::Buffered;
    // Disk, RamHot: cached/resident rows and their index take at most this RAM. Empty takes
    // zero for Disk and 4 GiB for RamHot. Ram always loads the whole table.
    std::optional<std::uint64_t> ram_budget_bytes;
    // RamHot: the hot-row profile (ninfer-ngram-profile) the rows are chosen by.
    std::filesystem::path hot_profile;
    // Ram, RamHot: lock the resident rows into physical memory (mlock, VirtualLock).
    bool lock = false;
    // Row reads in flight at once.
    std::uint32_t io_depth = 64;
    // Runs the model without its table: a non-standard, experimental mode that badly degrades the
    // output, since the model was trained with the table.
    bool disabled = false;
};

enum class CudaMemoryPolicy : std::uint8_t {
    DriverDefault,
    StrictVram,
    Mixed,
};

// Optional sparse KV working set. B and R are independent whole-page budgets;
// Host retains native page records, including quantization scale planes.
enum class KvmemQuerySource { User, Action, Auto };

struct KvmemOptions {
    std::uint32_t query_replay_max_tokens = 2048;
    std::uint32_t query_max_tokens = 512; // tail of the actual last-user span, matching llama.cpp
    bool lrfu = false;
    double lrfu_lambda = 0.5;
    double lrfu_top_fraction = 0.9;
    bool score_agg_max = false;
    // Structural retention: the system preamble, declared tool definitions and the
    // first user brief. The answer tail is not pinned after the dynamic recent
    // mechanism was removed.
    bool keep = false;
    // History query source: User keeps the newest user content, Action uses the
    // tool-call arguments of the action that opened a tool continuation, Auto
    // selects Action only when such a span exists (tool result as final message).
    // Default Auto: ordinary prompts have no action span and fall back to User.
    KvmemQuerySource query_source = KvmemQuerySource::Auto;
    // Draft ring KV storage, independent of the target's kv_cache. BF16 keeps proposal quality;
    // int8 trades some draft accuracy for capacity. There is no requirement to match the target.
    KvCacheStorage draft_kv_storage = KvCacheStorage::BFloat16;
    bool query_meanq_cosine = false;
    double graph_alpha = 0.2;
    std::uint32_t selected_tokens = 0; // B; zero disables KVMem
    std::uint32_t reserve_tokens = 0;  // R; also covers each prefill chunk
    // Whole-page rounding uses this backend's fixed 64-token page.
    // Sink 0 keeps one page; a positive value rounds down and stays at least one page.
    // Recent rounds down and must retain at least one page for the partial append tail.
    // Values below 64 are rejected when KVMem is enabled; the default keeps one page.
    std::uint32_t sink_tokens = 0;
    std::uint32_t recent_tokens = 64;
    std::uint64_t host_bytes = 0;      // H; hard bound on archived native payload
    [[nodiscard]] std::uint32_t sink_blocks() const noexcept {
        return sink_tokens < 64 ? 1U : sink_tokens / 64U;
    }
    [[nodiscard]] std::uint32_t recent_blocks() const noexcept { return recent_tokens / 64U; }
    std::uint32_t retained_sessions = 4; // bounded inactive Host histories, 1..16
    bool verify_transfers = false;    // correctness diagnostics, off in product runs
    // Process-local active/idle native KV tier. H bounds resident payload; D
    // bounds reserved disk extents. No restart persistence or DirectStorage.
    std::filesystem::path disk_path;
    std::uint64_t disk_bytes = 0;
    [[nodiscard]] std::uint64_t device_tokens() const noexcept {
        return static_cast<std::uint64_t>(selected_tokens) + reserve_tokens;
    }
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    // Message the model receives when it hits its thinking budget, before the canonical
    // </think> close the frontend appends when the message lacks it. Empty preserves the
    // model's built-in end-of-thinking control suffix.
    std::string thinking_budget_message;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    // Empty or one entry keeps the single-device route and `device` selects it. Several entries
    // split the model's layers into that many pipeline stages, one per device, in the given order:
    // the first holds the embedding, head and round state. Matching compute capability is required
    // at construction; peer access is only probed and recorded as a capability -- boundary
    // transfers stage through pinned host memory when it is unavailable.
    std::vector<int> devices;
    // Layers per stage, one count per entry of `devices`. Empty lets the engine choose from each
    // device's free memory.
    std::vector<std::uint32_t> stage_layers;
    // The devices share every layer instead of dealing layers out: tensor parallelism over the GDN
    // heads and the dense FFN rows, with attention, the KV cache, the embedding, the head and MTP
    // on the first device. Needs identical devices and a dense Qwen3.5-core model.
    bool tensor_parallel = false;
    // Native Qwen MoE, Qwen3.8-Flash-Next, Gemma 4 MoE and Flint.
    ExpertResidency expert_residency = ExpertResidency::Device;
    // Host/disk experts: device memory lent to the persistent cache, split
    // evenly over the devices. Automatic sizing follows each family's KV/workspace reservations.
    // Zero disables the cache for native Qwen and Gemma, which retain bounded transient slots;
    // Flash-Next disk residency requires a nonzero cache.
    std::optional<std::uint64_t> expert_cache_bytes;
    HybridExpertOptions hybrid_experts;
    // Qwen3.8-Flash-Next only.
    NgramTableOptions ngram_table;
    // Qwen3.8-Flash-Next only: positions of each sparse-attention layer's KV a sequence keeps in
    // device memory. A longer context streams those layers: their KV lives in pinned host memory
    // and the device keeps the blocks the indexer selected most recently, which leaves the rest
    // of the device memory to the expert cache. Unset: kDefaultKvResidentPositions with host or
    // disk experts, every position with device experts; 0 keeps every position.
    std::optional<std::uint32_t> kv_resident_positions;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    // Past the model's native window (max_position_embeddings), up to four times it: positions run
    // unscaled RoPE, or with rope_yarn the whole window takes Qwen's YaRN at factor
    // max_context / native (ops::RopeYarn).
    // Per-request bounded working sets; RoPE positions stay logical.
    KvmemOptions kvmem;
    bool rope_yarn = false;
    // A fixed YaRN factor in (1,4] for every position of the text and MTP layers, whatever
    // max_context is: Qwen documents one factor per deployment rather than one per window. 1 leaves
    // the choice to rope_yarn.
    float rope_yarn_factor = 1.0F;
    // Linear position interpolation instead of YaRN: for the text and MTP layers a position above
    // rope_scaling_original_context rotates at original + (p - original) / rope_scaling_factor,
    // while positions up to it keep their exact angles. 1 disables it; it excludes rope_yarn and
    // rope_yarn_factor and does not raise the window cap of four times the native window.
    float rope_scaling_factor = 1.0F;
    // The interpolation threshold; 0 is the model's native window.
    std::uint32_t rope_scaling_original_context = 0;
    // Enables JSON/JSON Schema request formats. Native tool constraints are available by default
    // and share the per-lane mask planes even when JSON request formats are disabled.
    bool structured_output = false;
    // Admit waiting requests to free lanes while other requests prefill, instead of holding
    // admission until the staged prefill finishes.
    bool concurrent_prefill = false;
    // A broken internal invariant (std::logic_error) in the worker fails the active and
    // materializing requests and keeps serving the queue, as an out-of-memory failure does, instead
    // of failing the Engine. Eight consecutive recoveries without a completed unit still fail it.
    bool recover_invariant_failures = false;
    // Windows builds only: budget the runtime against the adapter's dedicated memory less the
    // weights and a 512 MiB desktop floor, not against what is free now, on the grounds that WDDM
    // will page other allocations out, and commit every owning device arena at startup. Under
    // pressure from another process WDDM may page this process's memory out as well, which slows
    // generation until the memory comes back. Not for a GPU that drives the desktop, whose
    // allocations are often not evictable.
    bool wddm_evictable_budget         = false;
    // StrictVram is a Windows single-device residency guard and selects the Hybrid prefix cache.
    // Mixed is Windows single-device text generation using ordinary cudaMalloc: it attempts the
    // requested capacity even above reported CUDA free and accepts driver paging to Shared memory.
    // Mixed automatic capacity covers at most max_concurrency full max_context windows, without
    // expanding an idle prefix cache or probing system RAM. Normalization sets its automatic
    // headroom to zero; StrictVram sets it to cuda_vram_reserve_bytes. Only DriverDefault uses
    // the independently configured automatic KV headroom.
    // StrictVram probes may exceed reported free, but only a verified resident layout is admitted.
    CudaMemoryPolicy cuda_memory_policy = CudaMemoryPolicy::DriverDefault;
    std::size_t cuda_vram_reserve_bytes = 64ULL << 20;
    std::size_t cuda_memory_probe_step_bytes = 128ULL << 20;
    ModelSuspendOptions suspend;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    // Prompt tokens per prefill chunk; 0 is automatic: 1024, or 4096 for a native Qwen3.5-family MoE
    // whose routed experts live in host memory or files (each chunk carries the whole expert bank
    // across the bus once). Gemma 4 and Flint choose their own chunk.
    std::uint32_t prefill_chunk        = 0;
    // Decode rounds that run before a prefill chunk while other requests generate, proportionally
    // fewer (at least one) before a shorter prefill unit and more before a longer one. A decode
    // round takes tens of milliseconds and a chunk hundreds, so strict alternation (1) leaves
    // decoding streams a token per chunk; 0 is prefill_chunk / 64.
    std::uint32_t decode_rounds_per_prefill = 0;
    // Prefill with the fast INT8-KV prompt-attention kernel and round prefill_chunk down to whole
    // prompt-attention waves. Off keeps the default kernel and the requested chunk.
    bool fast_prefill_kernel           = false;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    VisionResidency vision_residency       = VisionResidency::Resident;
    std::uint32_t cpu_vision_idle_seconds = 300; // 0 retains lazily loaded Qwen CPU Vision weights
    // Speed-for-quality trades, opt-in and off by default. Measured in
    // docs/maintainer/quality-trade-experiments.md: lm_head_q4 costs +0.69% perplexity for a
    // C8 decode gain (~3%, real chat prompts); gdn_state_fp16 is free within measurement noise
    // (bit-identical greedy output, unchanged perplexity) for a ~2% C8 gain and a halved
    // per-slot host state image.
    bool lm_head_q4                        = false;
    bool gdn_state_fp16                    = false;
    // Device-memory trades that store a W8 vocabulary matrix at a narrower group-64 width while
    // the weights load (the artifact is unchanged). lm_head_q6 saves ~26% of the output head for
    // +0.01% perplexity; embedding_q4 halves the token embedding with no measurable perplexity
    // change. lm_head_q4 and lm_head_q6 are mutually exclusive.
    bool lm_head_q6                        = false;
    bool embedding_q4                      = false;
    // Q6G64 token embedding: the narrower-model option where Q4 costs measurable perplexity
    // (the 35B-A3B's 2048-wide table). Mutually exclusive with embedding_q4.
    bool embedding_q6                      = false;
    // Qwen3.6-35B-A3B only: the MTP draft layer's routed experts are stored W8 in the artifact;
    // transcode them to the text layers' formats (Q4 gate_up, Q6 down) at load. Drafts are
    // verified exactly, so this can change acceptance but never the output distribution.
    bool mtp_experts_q4                    = false;
    // Integer-activation MLP gate_up at decode and verify widths. Measured 4-6% off that Op from
    // sixteen columns up; see docs/maintainer/quality-trade-experiments.md.
    bool mlp_a8_decode                     = false;
    // Integer-activation routes at full prefill tiles, on by default where a route is registered.
    // Clearing it returns prefill to the A16 routes, which is the only way to measure what the
    // integer routes are worth on a whole request rather than per Op.
    bool prefill_a8                        = true;
    // Hand wide prefill GEMMs to cuBLAS instead of this fork's integer mainloop: the weight is
    // materialised as int8 with one scale per row and the activations quantised per token, which
    // runs about twice as fast (ops/linear_swiglu/q4cublas/w4_cublas_prefill.h) and is a further
    // quality trade on top of prefill_a8 -- hence off by default. Its dequantise pass is
    // weight-sized, so it wants a large prefill_chunk to amortise; the two belong together.
    bool prefill_cublas                    = false;
    // Extends prefill_cublas to the attention and GDN input projections, which hold about a fifth
    // of the linear parameters. Worth +13% prefill for +0.071% perplexity on top of what the route
    // already costs, the GDN half carrying nearly all of both. Separable because that is a
    // different trade from the MLP one and an owner may want only the cheaper half.
    bool prefill_cublas_projections        = true;
    // A native-FP4 card stores the A4 permission for an NVFP4 weight and runs W4A4 by default;
    // this asks for the dequantising W4A16 routes instead, the higher-precision of the two at the
    // cost of the FP4 tensor cores. On every other card the loader has already lowered the
    // permission to A16Only, so the flag is a no-op there: it only ever trades A4 away and never
    // introduces a compute domain the card lacks.
    bool nvfp4_w4a16                       = false;
    // NVFP4 weights on the integer tensor cores with s8-quantised activations (W4A8), prefill and
    // decode alike: the same activation trade the Q4/Q5 integer routes carry, opt-in per engine.
    // Which wide mainloop answers follows the route profile's nvfp4_int8/* bands; T <= 8 runs the
    // row-form kernel under either chain. On a native-FP4 card it replaces the A4 permission.
    // Where both this and nvfp4_w4a16 are set, nvfp4_w4a16 wins.
    bool nvfp4_w4a8_int                    = false;
    // Largest merged-token count one media item may occupy; larger media is downscaled at
    // preprocessing. Also bounds the overlay window.
    std::uint32_t vision_max_merged_tokens = 16384;
    bool use_cuda_graph                    = true;
    // Explicit total CUDA Graph driver-state allowance in bytes; zero keeps the
    // computed per-profile allowance.
    std::size_t cuda_graph_allowance_bytes = 0;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    // Per-GPU route profile (ops/common/device_route.h). "auto" installs the profile measured for
    // this device -- per route key, the entry at device_profile_path over the table compiled into
    // the binary; an entry measured against another route catalog is skipped -- and calibrates a
    // device that has neither once, storing the result at device_profile_path; "off" keeps the
    // compiled routes; "calibrate" measures anew. An empty path selects the user cache
    // ($NINFER_DEVICE_PROFILES, else $XDG_CACHE_HOME or ~/.cache, /ninfer/device-profiles.json).
    std::string device_profile = "auto";
    std::filesystem::path device_profile_path;
    // Lowest graphics clock, in MHz, the Engine's GPUs are held at while a generation or scoring
    // call runs (core/gpu_clock_floor.h): a layer pipeline's stages otherwise wait at idle clocks.
    // Zero leaves the clocks to the driver. Holding them needs administrator rights (root on
    // Linux); without them the Engine reports it and runs at the driver's clocks.
    std::uint32_t gpu_clock_floor_mhz = 0;
    // Prompt grafts a request may select by name through PromptOptions::graft. Each is loaded and
    // validated against the resident model at construction.
    std::vector<GraftSource> grafts;
    StartupObserver startup_observer;
    DiagnosticObserver diagnostic_observer;
    SlotAutoSaveOptions slot_auto_save;
};

} // namespace ninfer
