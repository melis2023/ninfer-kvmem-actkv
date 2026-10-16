#pragma once

// Block-sparse KV attention — block repository + working-set selection.
//
// A single long context (agent multi-turn, cumulative up to ~1M tokens) is
// split into fixed-size blocks (default 128 tokens, independent of the 16-token
// physical KV page). Each selection round an external algorithm (v1: built-in
// cumulative-attention top-k) picks a set of block IDs within the model's
// context budget (e.g. 128k). Only selected blocks are used for attention;
// the adapter builds its execution view from those IDs. Current native-payload
// execution keeps original RoPE positions; compact addresses do not rewrite K.
//
// This module is PURE HOST LOGIC: block metadata, selection, the diff that
// drives stage-in / stage-out, and the position remap plan. It owns no GPU
// memory and calls no kernels. The legacy set_selection path still computes
// re-RoPE bookkeeping for its existing callers/tests; it is not a native transfer
// contract. MemorySession uses pick_topk_blocks without mutating that bookkeeping.
// See memory_contract.hpp and tests/kvmem_store_test.cpp.
//
// Scope note: blocks/selection/tiering/remap apply ONLY to standard attention
// layers (1/4 of layers in Qwen3.6). DeltaNet recurrent layers store O(1)
// state and are untouched. Selection is GLOBAL (all attention layers share the
// same selected block IDs); the per-layer dimension is reserved for later.

#include <cstdint>
#include <string>
#include <vector>

namespace kvmem {

enum class KvTier : uint8_t { GPU = 0, CPU = 1, SSD = 2 };

// ActKV (arXiv:2609.31395, Sec. 5.1) AAV access pattern, read from
// KvMemBlock::access_ring. The paper separates entries that are hit every round
// (continuous), hit in some rounds but not others (intermittent), hit once
// (cold) and never hit since they appeared (silent). Intermittent is the class
// the paper's protection exists for.
enum class KvAavClass : uint8_t { Silent = 0, Cold = 1, Intermittent = 2, Continuous = 3 };

// Metadata for one context block, spanning all attention layers (a token range
// has KV in every attention layer). Physical storage handles per tier are
// added in the tiering task (#41); v1 tracks logical placement + remap state.
struct KvMemBlock {
    uint32_t block_id = 0;          // dense index in append order (0,1,2,...)
    uint32_t orig_pos_start = 0;    // first original token position in block
    uint32_t n_tokens = 0;          // tokens in this block (<= block_tokens)
    KvTier   tier = KvTier::GPU;    // active copy; stage-out may retain clean copies
    int32_t  gpu_slot = -1;         // future physical GPU block/page slot
    int32_t  cpu_slot = -1;         // CPU pinned tier slot, -1 if absent
    int32_t  nvme_slot = -1;        // NVMe tier slot, -1 if absent
    // Inclusive mode retains clean lower-tier copies while GPU is active:
    // tier names the active copy used for the next transition, while
    // cpu_slot/nvme_slot may describe additional clean copies. Exclusive mode
    // releases the source slot after promotion and keeps one authoritative copy.
    bool     ssd_clean = false;      // nvme_slot contains the current spill bytes
    bool     dirty_gpu = false;     // GPU copy newer than backing copy
    bool     in_flight = false;     // async copy/verification owns this block

    // Legacy re-RoPE bookkeeping, updated by set_selection. These fields are
    // NOT authoritative GPU K positions in the current packed-copy adapter.
    // Its original token positions live in orig_pos_start / engine row metadata.
    // New native-payload plans use ViewBlock::compact_start for addressing only.
    int64_t  baked_pos = -1;        // -1 until first registered (then orig_pos_start)
    uint32_t remap_count = 0;       // number of non-noop in-place re-RoPE moves
    uint64_t remap_abs_delta = 0;   // sum(abs(to_base-from_base)) since raw rebuild
    bool     in_working_set = false;

    // Cumulative attention quality (built-in top-k selection signal, #40). This
    // is the CONTENT signal: the one-shot retrieval score (Retrieval method) or
    // the window-local H2O heat. The ActKV keep score never writes it, so media
    // constraint (constrain_media) and diagnostics keep reading the content
    // signal even while the LRFU keep score drives the working set.
    double   attn_score = 0.0;
    double   profile_score = 0.0;    // window-local cumulative attention heat
    double   retrieval_score = 0.0;  // global retrieval score

    // ActKV (arXiv:2609.31395, Sec. 5.2 Alg. 1) action-oriented keep score:
    //   KS_t = lambda * KS_{t-1} + (top-p hit ? retrieval_t : 0)
    // Non-negative by construction. Committed only at commit_lrfu() in deferred
    // mode, so it is the persistent value and never the per-round ranking value.
    double   lrfu_score = 0.0;

    // The ranking signal pick_topk_blocks / the quota fill sort by. Exactly one
    // source writes it per round, so no writer can silently clobber another
    // (docs/a9-deferred-lrfu-regression.md, lesson 1: "the ranking signal and the
    // persistent score must be named and built separately"):
    //   Retrieval method -> retrieval_score
    //   H2O method       -> profile_score
    //   LRFU immediate   -> the new keep score KS
    //   LRFU deferred    -> the PROVISIONAL KS for this round; lrfu_score still
    //                       holds the committed value until commit_lrfu.
    double   select_score = 0.0;

    // ActKV (arXiv:2609.31395, Sec. 5.1) AAV access pattern: one hit bit per
    // compression round, most recent in bit 0, plus saturating counters. The
    // paper's finding is that continuous entries are the most common but carry
    // the least attention per access, while intermittent entries contribute
    // strongly and discontinuously -- so eviction driven by the current round's
    // preference can drop entries that matter again later.
    //
    // Advanced by record_access only for the diagnostic, so the default path
    // carries the fields untouched. Nothing in the selection path reads them:
    // aav_class_of and the trace turn them into a class, and that class has no
    // policy attached. The paper's AAV is an analysis (its Fig. 6) used to
    // motivate the single global lambda, not a switch -- treating a class as a
    // second decay factor freezes the working set and costs recall.
    uint16_t access_ring = 0;
    uint16_t access_rounds = 0;         // rounds observed since the block appeared

    uint32_t orig_pos_end() const { return orig_pos_start + n_tokens; }
};

// Legacy re-RoPE plan metadata. Existing selection tests exercise from_base /
// to_base / raw_refresh semantics. The llama adapter uses NativeResidencyPlan
// and copies native packed KV at original RoPE positions.
// New backends must consume WorkingSetPlan, not reinterpret this structure.
struct KvMemRemap {
    uint32_t block_id = 0;
    uint32_t n_tokens = 0;
    int32_t  from_base = 0;   // current bake position (de-rotate source)
    int32_t  to_base = 0;     // assigned in-window first-token position
    // True when the rotated working K survived on GPU at planning time.
    // This is deliberately distinct from KvMemPlan::stage_in, which tracks a
    // logical working-set transition and may include still-resident pages.
    bool     working_k_resident = false;
    bool     skip = false;    // resident valid K already baked at to_base
    // Rebuild this block from the unrotated CPU raw-K mirror instead of
    // applying another lossy in-place de-RoPE/re-RoPE pass. Always true for a
    // cold stage-in and when either refresh threshold is reached.
    bool     raw_refresh = false;
};

// One block removed by truncate_to, carrying the tier slots the executor must
// release. GPU pages are NOT freed here (restore_state already truncated the KV
// page table); only CPU/NVMe tier slots are the executor's to release.
struct KvMemDroppedBlock {
    uint32_t block_id = 0;
    KvTier   tier = KvTier::GPU;
    int32_t  gpu_slot = -1;
    int32_t  cpu_slot = -1;
    int32_t  nvme_slot = -1;
};

// Result of diffing a new selection against the current working set.
struct KvMemPlan {
    // Blocks newly entering the logical working set. The executor may avoid a
    // physical transfer when their GPU pages are still resident.
    std::vector<uint32_t> stage_in;
    std::vector<uint32_t> stage_out;  // resident blocks no longer selected
    // Legacy window-order bookkeeping. total_window_tokens is a count, not the
    // original query RoPE position. Native execution uses its engine row metadata.
    std::vector<KvMemRemap> remaps;
    uint32_t total_window_tokens = 0;  // sum of n_tokens over selected blocks
    // Natural overlap is reported even in the reuse-off ablation, where the
    // executor deliberately reloads blocks that it could otherwise retain.
    uint32_t selection_overlap_blocks = 0;
    uint32_t gpu_reused_blocks = 0;
    uint32_t retained_position_stable = 0;
    uint32_t retained_position_moved = 0;
};

// Transitional plan for llama's existing slot/tier executor. Planning is pure;
// membership is committed after admission. No re-RoPE bookkeeping is read or
// written. Unlike WorkingSetPlan this can include not-yet-evaluated prefill rows
// and does not certify completed payloads or a published attention view.
struct NativeResidencyPlan {
    std::vector<uint32_t> selected;
    std::vector<uint32_t> stage_in;
    std::vector<uint32_t> stage_out;
    uint32_t total_window_tokens = 0;
    uint32_t selection_overlap_blocks = 0;
    uint32_t gpu_reused_blocks = 0;
};

// Which signal ranks the middle (non-sink/recent) blocks each reselection.
// All three feed the SAME pick_topk_blocks (sink + recent windows are always
// kept); they differ only in what writes select_score:
//   Retrieval — global content-frame similarity to the current query; re-ranks
//               ALL historical blocks each interval and CAN resurrect a block
//               already dropped from the window (Quest/InfLLM-style). Default.
//   H2O       — window-local cumulative attention heat; a heavy-hitter
//               RETENTION signal that keeps attended blocks resident but cannot
//               resurrect a dropped one (H2O-style).
//   Recency   — no learned signal; pick_topk keeps only sink + recent blocks.
// Retrieval auto-falls-back to H2O when its index/query is not live (q8/fp8
// cache, or the first post-prefill reselect before any query exists), and H2O
// in turn degenerates to Recency when no heat has accumulated. Selecting H2O or
// Recency explicitly disables the more expensive upstream signal.
enum class KvMemMethod : uint8_t { Retrieval = 0, H2O = 1, Recency = 2 };
enum class KvMemSelectPolicy : uint8_t { TopK = 0, Quota = 1 };
// Query-conditioned scorer (--kvmem-retrieval-method): MeanK = softmax-over-pages
// on the cheap per-block mean key; PerToken = ExactMass over raw per-token keys;
// SubBlockMeanK = MeanK at sub-block granularity (n_subblocks means per block,
// softmax over all block*sub-block logits, mass summed back per block) so a
// concentrated relevant sub-region isn't diluted by a block's mean.
//   DeltaNet (EXPERIMENTAL, NOT RECOMMENDED) = rank blocks by each historical
//              block's NET EDIT to the DeltaNet
//              recurrent state (E_j = S_j - a_j S_{j-1}) read by the current
//              DeltaNet query, aggregated over query tokens / heads / layers with
//              a per-layer RMS normalization (see
//              docs/kvmem_deltanet_retrieval_design.md). Unlike
//              the mean-k / per-token scorers (which read standard-attention keys)
//              this reads the RECURRENT layers' state increments.
enum class KvMemRetrievalMethod : uint8_t { MeanK = 0, PerToken = 1, SubBlockMeanK = 2, DeltaNet = 3 };
// How the multiple Mean-K representatives inside one logical block are formed.
// Contiguous preserves the existing equal-width sub-block implementation.
// KeyDirectionFixed4 partitions a logical retrieval block into contiguous
// 32-token slices, then clusters each layer/KV-head slice's content-frame Keys
// into four non-contiguous direction groups. KeyDirectionAdaptive uses the same
// deterministic farthest-first candidates but keeps 1, 2, or 4 representatives
// according to normalized cosine-residual gains. Its persistent index is packed.
enum class KvMemPrototypeMode : uint8_t {
    Contiguous = 0,
    KeyDirectionFixed4 = 1,
    KeyDirectionAdaptive = 2,
};
// Sub-block score reduction (--kvmem-subblock-reduce; SubBlockMeanK only):
// Sum aggregates a block's sub-block softmax masses (average attention the block
// receives); Max scores a block by its single best sub-block (MaxSim-style late
// interaction: max over the document side, still summed over query tokens/heads/
// layers) so a concentrated relevant sub-region wins instead of being averaged
// down. At n_subblocks == 1 the two are identical (one sub-block per block).
enum class KvMemSubblockReduce : uint8_t { Sum = 0, Max = 1 };
// Logical semantic-group reduction. Max preserves the existing per-query
// MaxSim behavior. LengthNormalizedMass first sums the globally normalized
// 32-token attention mass inside the group, then divides the accumulated group
// score by group_length^group_length_norm_alpha on the host.
enum class KvMemGroupScoreReduce : uint8_t {
    Max = 0,
    LengthNormalizedMass = 1,
};
enum class KvMemSemanticExpansion : uint8_t {
    None = 0,
    Round = 1,
    Message = 2,
};
enum class KvMemUpdateMode : uint8_t { Interval = 0, Step = 1 };
enum class KvMemDeltaNetLayerPolicy : uint8_t { Even = 0, Late = 1 };

enum class KvMemKeepSource : uint8_t {
    Auto = 0,
    Tokens = 1,
    Blocks = 2,
};

struct KvMemKeepAllocation {
    uint32_t sink_target_tokens = 0;
    uint32_t recent_target_tokens = 0;
    uint32_t sink_blocks = 0;
    uint32_t recent_blocks = 0;
    uint32_t sink_effective_tokens = 0;
    uint32_t recent_effective_tokens = 0;
    KvMemKeepSource sink_source = KvMemKeepSource::Auto;
    KvMemKeepSource recent_source = KvMemKeepSource::Auto;
};

// Resolve token-based always-kept bands after both the selection budget and
// physical block size are known. A negative explicit value means "auto".
// Explicit token and block values are mutually exclusive for each band.
KvMemKeepAllocation resolve_kvmem_keep_allocation(
    uint32_t block_tokens,
    uint32_t select_budget,
    int64_t sink_blocks,
    int64_t recent_blocks,
    int64_t sink_tokens,
    int64_t recent_tokens);

enum class KvMemIndexPlacement : uint8_t {
    GPU = 0,
    CPU = 1,
};

enum class KvMemAdaptiveScoreMode : uint8_t {
    Auto = 0,
    LayerOnePass = 1,
    TiledTwoPass = 2,
    TiledOnePass = 3,
};

struct KvMemStoreConfig {
    uint32_t block_tokens = 128;     // --kvmem-block-tokens
    uint32_t select_budget = 131072; // --kvmem-budget (semantic window tokens)
    // Pressure-only prefill window. configure_kvmem normalizes zero to
    // select_budget, so old callers retain byte-identical behavior.
    uint32_t prefill_budget = 0;     // --kvmem-prefill-budget
    uint32_t gen_budget = 32768;     // --kvmem-gen-budget (GPU pool reserve for
                                     // one turn's newly generated tokens)
    uint32_t sink_blocks = 1;        // --kvmem-sink-blocks (always-kept prefix)
    uint32_t recent_blocks = 0;      // --kvmem-recent-blocks (always-kept suffix);
                                     // 0 => keep no suffix blocks unconditionally

    // ActKV (arXiv:2609.31395) action-oriented eviction. Opt-in; off keeps the
    // historical one-shot retrieval ranking byte-identical.
    //   KS_t(i) = lrfu_lambda * KS_{t-1}(i) + (hit at t ? retrieval_t(i) : 0)
    // A block is a hit when it carries a positive retrieval score AND ranks
    // inside the top lrfu_top_frac fraction of the positive scores (the paper's
    // p = 90%). A block with no positive evidence is never a hit, so a keep score
    // cannot grow from page age alone -- see accumulate_lrfu for why that matters
    // at large context.
    // Defaults follow the paper's tuned configuration: lambda = 0.5, p = 90%.
    bool   action_lrfu   = false;    // --kvmem-lrfu
    float  lrfu_lambda   = 0.50f;    // --kvmem-lrfu-lambda
    float  lrfu_top_frac = 0.90f;    // --kvmem-lrfu-top-frac
    // Restrict growth to the blocks the round actually kept (two-phase form).
    // Callers that select after scoring set this and call commit_lrfu; callers
    // that cannot report a selection leave it off and keep the score-only rule.
    // Not a CLI flag: the ninfer adapter turns it on unconditionally and commits
    // with the round's selection; the llama.cpp adapter leaves it off.
    bool   deferred_lrfu = false;

    // Advance the AAV access ring. This is the DIAGNOSTIC only: the serve sets it
    // from KVMEM_AAV_TRACE so the ring and the trace line exist for analysis, and
    // callers that want the ring without the environment variable (tests) set it
    // directly. It never changes a retention decision -- the paper's Sec. 5.1
    // classifies access patterns to justify one global lambda, and gives no
    // per-class decay.
    bool aav_trace = false;

    // Cross-page normalization of the one-shot retrieval score
    // (--kvmem-retrieval-temp). 0 = off, which keeps the historical raw-cosine
    // scale byte-identical to before this option existed.
    //
    // The retrieval_method note below calls MeanK a softmax over pages; with the
    // raw cosine that is not what runs, and the two terms the keep score folds
    // are not commensurable -- KS is an unbounded accumulator while the score is
    // in [-1, 1]. After a few rounds lambda * KS dominates, which is the failure
    // the provisional-KS comment in set_retrieval_scores describes: the round's
    // own evidence stops moving the ranking, and a block written this round
    // cannot outrank the incumbents. Softmax makes the score a mass (mean 1/N
    // over the positively scored blocks) and the temperature is then the
    // exchange rate between accumulated memory and this round's evidence.
    //
    // Only positively scored blocks carry mass, so the "no hit from page age
    // alone" rule is unchanged, and softmax is order-preserving, so the
    // within-round selection and the percentile hit mask are unchanged too.
    // What moves is only the weight the evidence carries inside KS_t.
    float  retrieval_temperature = 0.0f;  // --kvmem-retrieval-temp

    KvMemMethod select_method = KvMemMethod::Retrieval; // --kvmem-method
    KvMemSelectPolicy select_policy = KvMemSelectPolicy::TopK;
    KvMemRetrievalMethod retrieval_method = KvMemRetrievalMethod::MeanK;
    KvMemIndexPlacement index_placement = KvMemIndexPlacement::GPU;
    uint64_t index_staging_bytes = 64ull * 1024ull * 1024ull;
    std::string numa_policy = "auto";
    KvMemAdaptiveScoreMode adaptive_score_mode =
        KvMemAdaptiveScoreMode::Auto;
    uint32_t n_subblocks = 1;        // --kvmem-subblocks; sub-block means per block
                                     // (SubBlockMeanK only; 1 => plain mean-k)
    KvMemPrototypeMode prototype_mode = KvMemPrototypeMode::Contiguous;
    double adaptive_gain_1to2 = 0.10;
    double adaptive_gain_2to4 = 0.06;
    KvMemSubblockReduce subblock_reduce = KvMemSubblockReduce::Max; // --kvmem-subblock-reduce
    KvMemSemanticExpansion semantic_expansion =
        KvMemSemanticExpansion::None;
    KvMemGroupScoreReduce group_score_reduce =
        KvMemGroupScoreReduce::Max;
    double group_length_norm_alpha = 0.5;
    KvMemUpdateMode update_mode = KvMemUpdateMode::Interval;
    // Orthogonal performance controls. They must not alter retrieval,
    // precision, immutable-K, query replay, or any other correctness policy:
    //   stage-out: when completed GPU blocks are copied to a lower tier;
    //   stage-in:  which selected blocks must be loaded/rematerialized;
    //   pack:      how blocks are transferred (cross-block batches/pipelines).
    bool optimize_stage_out = true;
    bool optimize_stage_in = true;
    bool optimize_pack = true;

    // Legacy set_selection re-RoPE experiment controls. NativeResidencyPlan
    // and WorkingSetPlan ignore these: packed KV keeps original RoPE positions.
    bool immutable_source_k = false;
    // Engine-level MTP prefix/speculation is enabled for this executor. Kept
    // explicit so non-MTP KVMem runs do not allocate/tier an unused MTP cache
    // or reserve an unnecessary MTP raw-K mirror.
    bool mtp_enabled = false;
    uint32_t immutable_refresh_remaps = 8;
    uint64_t immutable_refresh_abs_delta_tokens = 262144;
    uint64_t immutable_max_baked_position = 262144;

    // Archived experimental DeltaNet retrieval. Not exposed by the CLI and not
    // recommended for production; retained for future research and reproducibility.
    // See docs/kvmem_deltanet_retrieval_experimental.md.
    // Scores blocks by their net edit to the DeltaNet recurrent state read by the
    // current DeltaNet query. The full per-block state edit E_j is a d_v x d_k
    // matrix per (layer, head, block) (~64 KB at state_size 128), so the number of
    // DeltaNet layers that feed the score is capped by a memory budget: the
    // executor picks evenly-spaced DeltaNet layers up to deltanet_layers, further
    // clamped so the E_j buffer fits deltanet_mem_budget_bytes.
    uint32_t deltanet_layers = 0;             // --kvmem-deltanet-layers (0 => derive from budget)
    KvMemDeltaNetLayerPolicy deltanet_layer_policy = KvMemDeltaNetLayerPolicy::Even;
    uint64_t deltanet_mem_budget_bytes = 0;   // --kvmem-deltanet-mem-budget-gb (0 => 32 GiB default)
    bool     deltanet_decay = true;           // --kvmem-deltanet-decay (apply exp(G_M - G_j))
    uint32_t deltanet_topk_q = 4;             // --kvmem-deltanet-topk-q (TopKMean over query tokens)
    uint32_t deltanet_topk_h = 4;             // --kvmem-deltanet-topk-h (TopKMean over heads)

    // Quota policy. Zero means derive from the remaining budget at selection
    // time. Quotas are measured in blocks and apply after sink/recent are kept.
    uint32_t retrieval_blocks = 0;
    uint32_t profile_blocks = 0;

    // GPU KV residency budget. The current implementation uses this to cap the
    // active working-set budget; CPU/NVMe tiering will use the same estimated
    // block capacity as the global GPU repository high-watermark.
    double gpu_memory_ratio = 0.50;
    double gpu_high_watermark = 0.95;
    double gpu_low_watermark = 0.85;
    uint32_t estimated_gpu_block_capacity = 0;
    // Complete GPU-resident K+V bytes for one logical block.  This differs
    // from estimated_block_bytes in immutable-source mode, where the lower
    // tiers store position-free K separately and each spill record contains
    // only V.  Resource reporting must use this value for the GPU page pool.
    uint64_t gpu_resident_block_bytes = 0;
    uint64_t estimated_block_bytes = 0;
    uint64_t cpu_tier_bytes = 0;
    uint64_t nvme_tier_bytes = 0;
    std::string nvme_tier_dir;
    // Reserve the first part of --kvmem-nvme-* for an immutable raw-K
    // authority. CPU raw-K chunks then become a bounded cache and cold selected
    // blocks stream directly from SSD into the existing packed staging slabs.
    bool raw_k_nvme = false;
};

class KvMemStore {
public:
    explicit KvMemStore(KvMemStoreConfig cfg) : cfg_(cfg) {}

    const KvMemStoreConfig &config() const { return cfg_; }
    uint32_t block_count() const { return static_cast<uint32_t>(blocks_.size()); }
    uint32_t total_tokens() const { return total_tokens_; }
    const std::vector<KvMemBlock> &blocks() const { return blocks_; }

    // AAV class of a resident block (ActKV Sec. 5.1). Diagnostics only: the
    // paper's classification describes access patterns and motivates the single
    // global lambda; no policy consumes it. Public so the trace and the tests
    // read the same rule.
    KvAavClass aav_class_of(uint32_t block_id) const;

    // Block that owns original token position `pos`, or -1 if none.
    int32_t block_id_containing(uint32_t pos) const;

    // Register newly-appended context tokens as blocks. Called as the context
    // grows. Partial trailing blocks are extended in place until full, then a
    // new block starts. Returns the number of blocks that became newly full
    // (the trailing partial block is always present once any token exists).
    void register_append(uint32_t n_new_tokens);

    // Drop trailing blocks so the store holds exactly `token_pos` tokens (the
    // inverse of register_append). Used by the kvmem prefix cache to rewind the
    // block table when resuming at a prompt-end checkpoint whose position is
    // BELOW the live end. A partially-covered trailing block is shrunk in place;
    // fully-past blocks are popped from the back (block_id == vector index, so
    // popping keeps indices dense). Returns the popped blocks + their tier slots
    // so the caller can release CPU/NVMe storage. `token_pos >= total_tokens_`
    // is a no-op (returns empty).
    std::vector<KvMemDroppedBlock> truncate_to(uint32_t token_pos);

    // Capacity of the semantic selection budget, in tokens. The ActKV Sec. 6 adaptive
    // budget used to narrow this per compression round through a runtime override; that
    // policy was removed (see serve_options.cpp), so the configured value is the answer.
    uint32_t select_budget_tokens() const { return cfg_.select_budget; }
    uint32_t budget_blocks() const {
        return select_budget_tokens() / cfg_.block_tokens;
    }
    uint32_t prefill_budget_blocks() const {
        const uint32_t tokens = cfg_.prefill_budget != 0
            ? cfg_.prefill_budget : cfg_.select_budget;
        return tokens / cfg_.block_tokens;
    }

    // Soft GPU-pool watermarks. `pool_tokens` is the physical slot-pool size
    // (budget + gen_reserve). A ratio <= 0 or > 1 is treated as 1 (hard cap).
    uint32_t gpu_high_watermark_tokens(uint32_t pool_tokens) const;
    uint32_t gpu_low_watermark_tokens(uint32_t pool_tokens) const;

    // True when the next prefill chunk would overflow the physical GPU pool
    // (budget + gen_reserve) or its high watermark. History longer than the
    // semantic prefill budget is not enough: unused gen_reserve slots are
    // slack, then pressure contracts back to prefill_budget in one reselect.
    bool prefill_needs_offload(uint32_t resident_tokens,
                               uint32_t incoming_tokens,
                               uint32_t pool_tokens) const;

    // Built-in v1 selection: top-k blocks by cumulative attention score, always
    // keeping the first `sink_blocks` and last `recent_blocks` blocks. Returns
    // selected block IDs in ascending (window) order. Honors the budget.
    std::vector<uint32_t> pick_topk_blocks() const;
    // Same selector with additional mandatory block IDs. Mandatory blocks
    // consume ordinary top-k slots; they are never appended beyond the budget.
    // This is used by query replay so its suffix stays present without growing
    // the active RoPE window past `select_budget`.
    std::vector<uint32_t> pick_topk_blocks(
        const std::vector<uint32_t> &mandatory_blocks) const;

    // Select complete variable-length logical groups under the ordinary
    // fixed-physical-block budget. `groups` are half-open token spans in the
    // original prompt and `group_scores` contains one GPU-reduced semantic score
    // per span. Sink/recent/mandatory blocks are charged first. A selected
    // group contributes every physical block it overlaps; adjacent groups may
    // share a boundary block, which is charged once. Groups that do not fit are
    // skipped rather than partially materialized.
    std::vector<uint32_t> pick_semantic_groups(
        const std::vector<std::pair<uint32_t, uint32_t>> &groups,
        const std::vector<double> &group_scores,
        const std::vector<uint32_t> &mandatory_blocks = {}) const;

    // Deterministic working set used only while a long prefill is under memory
    // pressure. Keep the configured sink prefix, then fill every remaining
    // budget slot with the newest blocks. This deliberately ignores all
    // retrieval/profile/attention scores and also ignores `recent_blocks`:
    // "recent" here means the full newest tail left after reserving the sink.
    std::vector<uint32_t> pick_prefill_pressure_blocks() const;
    // Same pressure selector with request-scoped mandatory blocks. Mandatory
    // blocks consume ordinary newest-tail slots and never expand the fixed
    // prefill budget.
    std::vector<uint32_t> pick_prefill_pressure_blocks(
        const std::vector<uint32_t> &mandatory_blocks) const;

    // Accumulate per-block attention quality (decode side-channel, #40). `scores`
    // is indexed by block_id; entries beyond block_count() are ignored.
    void accumulate_attn(const std::vector<double> &scores);

    // Overwrite per-block selection score (global KV retrieval). Unlike
    // accumulate_attn (a window-local running sum), retrieval re-scores every
    // historical block fresh each interval by query/mean-key similarity, so the
    // score is REPLACED, not added. `scores` is indexed by block_id; entries
    // beyond block_count() are ignored, blocks beyond `scores.size()` are reset
    // to 0 so a shrinking index never leaves stale heat behind.
    void set_attn_scores(const std::vector<double> &scores);

    // Explicit retrieval scores for selector plugins. This currently aliases
    // set_attn_scores for backwards compatibility, but also preserves the
    // retrieval/profile split used by the quota selector.
    //
    // ActKV: when KvMemStoreConfig::action_lrfu is on, this ALSO folds the
    // one-shot score into the temporal keep score (Alg. 1) and publishes KS as
    // select_score, so every backend that only calls this entry point (llama.cpp
    // and ninfer alike) gets the same action-oriented ranking. attn_score keeps
    // the one-shot content signal (constrain_media reads it, never KS).
    // `round_id`
    // identifies the compression round: a repeat with the same id is a no-op
    // that still republishes KS, so preview/apply passes stay consistent.
    //
    // round_id == 0 asks the store to derive the round from the block frontier.
    // That is what a caller with no ORA clock of its own passes -- the ninfer
    // backend scores from prepare_window() and cannot say which round it is in.
    // Deriving the stamp there means "context grew -> new round; same frontier ->
    // repeat", so the guard below still protects such a caller. Callers that do
    // have a round clock (the llama adapter) pass it explicitly.
    void set_retrieval_scores(const std::vector<double> &scores,
                              uint64_t round_id = 0);

    // Two-phase ActKV for callers that select AFTER scoring (the ninfer
    // backend does: score, pick_topk_blocks, then stage). set_retrieval_scores
    // with action_lrfu on would fold the score into KS immediately, before the
    // selection exists, so a block could grow on a score it never won a slot
    // for. With KvMemStoreConfig::deferred_lrfu the store instead holds the
    // round's scores and waits for commit_lrfu, which applies the update using
    // the caller's actual selection as the hit set (Alg. 1 over the action
    // region).
    //
    // Apply the deferred round: `selected` is the block set this round kept.
    // No-op unless action_lrfu && deferred_lrfu and a round is pending.
    void commit_lrfu(const std::vector<uint32_t> & selected);

    // ActKV Alg. 1 steps 3-4, applied directly. `scores` is the per-block
    // action-region retrieval score for this ORA round, indexed by block_id.
    //   KS[hit ] <- lambda * KS[hit ] + score[hit ]
    //   KS[miss] <- lambda * KS[miss]
    // KS is clamped at 0 (the paper's AttnScore is a non-negative attention
    // mass, so a negative keep score would permanently sink a block).
    // A zero `round_id` derives the round from the block frontier; see
    // set_retrieval_scores.
    //
    // `selected` is optional and narrows what counts as a hit: when it is
    // non-null, only a block that is BOTH selected this round and carrying a
    // positive score may grow. That is the paper's action region -- the entries
    // the model actually kept -- and it stops a block from building a keep score
    // out of a weak score it never won a slot for. A caller that does not know
    // its selection yet passes nullptr and keeps the score-only rule.
    // `selected` may be any order and may contain ids >= block_count().
    void accumulate_lrfu(const std::vector<double> &scores, uint64_t round_id = 0,
                         const std::vector<uint32_t> * selected = nullptr);

    // Drop every keep score (new session, prefix rewrite). Without this a
    // shrunk-then-regrown block keeps stale heat forever.
    void reset_lrfu();

    void set_block_tier(uint32_t block_id, KvTier tier,
                        int32_t cpu_slot = -1,
                        int32_t nvme_slot = -1);
    void set_block_gpu_slot(uint32_t block_id, int32_t gpu_slot);
    // Inclusive-copy metadata used by stage-out. These do not change which tier is
    // active, so evicting a CPU cache copy cannot accidentally mark a still-live
    // GPU block cold.
    void set_block_cpu_copy(uint32_t block_id, int32_t cpu_slot);
    void set_block_ssd_backing(uint32_t block_id, int32_t nvme_slot,
                               bool clean);
    void set_block_io_in_flight(uint32_t block_id, bool in_flight);
    void set_block_baked_pos(uint32_t block_id, int64_t baked_pos);
    void record_block_rerope(uint32_t block_id, int64_t baked_pos);

    // Diff `selected_ids` against current GPU residency / working-set state and produce the
    // stage-in/out lists + the window remap plan. Each remap starts from the
    // block's CURRENT GPU bake position. In immutable_source_k mode a cold
    // stage-in or a resident block exceeding the configured drift thresholds
    // is marked raw_refresh and rebuilt from the CPU raw-K mirror; otherwise a
    // resident move is an in-place delta rotation. `selected_ids` need not be sorted;
    // it is sorted ascending internally so window order is deterministic (sink
    // first ... recent last). Blocks are packed contiguously from window pos 0.
    KvMemPlan set_selection(std::vector<uint32_t> selected_ids,
                           bool force_raw_refresh = false);
    // Reject stale/duplicate IDs rather than silently shrinking the selection.
    NativeResidencyPlan plan_native_selection(std::vector<uint32_t> selected_ids) const;
    // Metadata only; every live GPU block must be selected and already valid.
    bool commit_resident_selection(const std::vector<uint32_t> & selected_ids);

    // Reset working-set membership (e.g. new session). Block table + baked_pos
    // are kept (the cache still holds K baked at whatever position it was left).
    void clear_working_set();

private:
    std::vector<uint32_t> constrain_media(std::vector<uint32_t> selected,
        const std::vector<uint32_t> & mandatory, uint32_t budget, bool recency) const;
    std::vector<uint32_t> pick_topk_ungrouped(const std::vector<uint32_t> & mandatory) const;
    std::vector<uint32_t> pick_prefill_ungrouped(const std::vector<uint32_t> & mandatory) const;
    // --kvmem-retrieval-temp: returns `scores` unchanged when the option is off,
    // otherwise a probability mass over the positively scored blocks.
    const std::vector<double> & normalized_retrieval_scores(const std::vector<double> & scores);
    // AAV bookkeeping: advances the per-block access ring from this round's
    // selection. A no-op unless the diagnostic is enabled (cfg.aav_trace or
    // KVMEM_AAV_TRACE).
    void record_access(const std::vector<uint32_t> & selected);
    std::vector<std::pair<uint32_t, uint32_t>> media_ranges_;
public:
    void set_media_ranges(std::vector<std::pair<uint32_t, uint32_t>> ranges) { media_ranges_ = std::move(ranges); }
    size_t allocated_bytes() const {
        return sizeof(*this) + blocks_.capacity()*sizeof(KvMemBlock) +
            media_ranges_.capacity()*sizeof(std::pair<uint32_t,uint32_t>);
    }
private:
    KvMemStoreConfig cfg_;
    std::vector<KvMemBlock> blocks_;
    uint32_t total_tokens_ = 0;  // total appended context length
    // Last compression round whose ActKV keep-score update was applied.
    uint64_t lrfu_last_round_ = 0;
    bool     lrfu_round_seen_ = false;
    // Scratch for the top-p threshold selection. Reused so the per-round LRFU
    // update does not heap-allocate: a caller may score once per prefill chunk.
    std::vector<double> lrfu_scratch_;
    // Round held between set_retrieval_scores and commit_lrfu in deferred mode.
    std::vector<double> lrfu_pending_scores_;
    uint64_t lrfu_pending_round_ = 0;
    bool     lrfu_pending_ = false;
    // --kvmem-retrieval-temp scratch. Reused so a scored chunk does not allocate.
    std::vector<double> retrieval_scratch_;
    // AAV diagnostic state. Only advanced while KVMEM_AAV_TRACE is set, so the
    // default path carries the fields but never touches them.
    uint64_t aav_last_round_ = 0;   // last token frontier seen (repeat guard)
    uint64_t aav_rows_ = 0;         // traced rows -- the monotone round number
    bool     aav_round_seen_ = false;
    // Last four selections as sorted id sets, for Jaccard. Sets, not bitmaps: the
    // block table grows every round, so equal-length comparisons are never equal.
    std::vector<uint32_t> aav_history_[4];
    uint32_t aav_history_next_ = 0;
    uint32_t aav_history_count_ = 0;
};

} // namespace kvmem
