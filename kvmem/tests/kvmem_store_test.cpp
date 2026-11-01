// KvMemStore host-logic unit test (block-sparse KV, Task #37).
//
// Pure host logic — no GPU. Covers: block registration (partial-block growth),
// the working-set diff (stage-in/out + window remap plan), and the built-in
// cumulative-attention top-k selection (sink/recent preservation + budget).

#include "kvmem/kvmem_store.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static void test_register_append() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    KvMemStore s(cfg);

    // 100 tokens -> 1 partial block of 100.
    s.register_append(100);
    CHECK(s.block_count() == 1);
    CHECK(s.blocks()[0].n_tokens == 100);
    CHECK(s.blocks()[0].orig_pos_start == 0);

    // +50 -> first block fills to 128, remainder 22 starts a new block.
    s.register_append(50);
    CHECK(s.block_count() == 2);
    CHECK(s.blocks()[0].n_tokens == 128);
    CHECK(s.blocks()[1].n_tokens == 22);
    CHECK(s.blocks()[1].orig_pos_start == 128);

    // +384 (3 full blocks) -> fills block1 to 128 (106 added), then more.
    s.register_append(384);
    // Total = 100+50+384 = 534 = 4 full blocks (512) + 22.
    CHECK(s.blocks()[1].n_tokens == 128);
    CHECK(s.block_count() == 5);
    CHECK(s.blocks()[2].n_tokens == 128);
    CHECK(s.blocks()[3].n_tokens == 128);
    CHECK(s.blocks()[4].n_tokens == 22);
    CHECK(s.blocks()[4].orig_pos_start == 128 * 4);
}

static void test_selection_diff_and_remap() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    KvMemStore s(cfg);
    s.register_append(128 * 5);  // 5 full blocks, ids 0..4
    CHECK(s.block_count() == 5);

    // Select blocks {0, 2, 4} (out of order on purpose).
    auto plan = s.set_selection({4, 0, 2});
    CHECK(plan.stage_in.size() == 3);       // all three newly resident
    CHECK(plan.stage_out.size() == 2);      // cold GPU blocks 1 and 3 leave
    CHECK(plan.stage_out[0] == 1);
    CHECK(plan.stage_out[1] == 3);
    CHECK(plan.remaps.size() == 3);
    CHECK(plan.total_window_tokens == 128 * 3);
    // Window order is ascending block id; packed contiguously from 0.
    // Each block's K is currently baked at its true position (orig_pos_start),
    // so from_base == orig position, to_base == window slot.
    CHECK(plan.remaps[0].block_id == 0);
    CHECK(plan.remaps[0].from_base == 0);
    CHECK(plan.remaps[0].to_base == 0);
    CHECK(plan.remaps[0].skip == true);     // block 0 already at window slot 0
    CHECK(plan.remaps[1].block_id == 2);
    CHECK(plan.remaps[1].from_base == 128 * 2);   // baked at true position
    CHECK(plan.remaps[1].to_base == 128);         // remapped into window slot 1
    CHECK(plan.remaps[1].skip == false);
    CHECK(plan.remaps[2].block_id == 4);
    CHECK(plan.remaps[2].from_base == 128 * 4);
    CHECK(plan.remaps[2].to_base == 128 * 2);
    CHECK(s.blocks()[0].remap_count == 0);
    CHECK(s.blocks()[2].remap_count == 1);
    CHECK(s.blocks()[4].remap_count == 1);
    for (uint32_t id : plan.stage_out) {
        s.set_block_tier(id, KvTier::CPU, static_cast<int32_t>(id), -1);
    }

    // Reselect {0, 2, 3}: block 4 leaves, block 3 enters, 0 and 2 stay.
    // Blocks 0 and 2 are now baked at their window slots (0 and 128), so on
    // reselection they map to the SAME slots and skip; block 3 enters from its
    // true position 128*3 -> window slot 2.
    auto plan2 = s.set_selection({0, 2, 3});
    CHECK(plan2.stage_in.size() == 1 && plan2.stage_in[0] == 3);
    CHECK(plan2.stage_out.size() == 1 && plan2.stage_out[0] == 4);
    CHECK(plan2.remaps.size() == 3);
    CHECK(plan2.remaps[0].block_id == 0 && plan2.remaps[0].skip == true);
    CHECK(plan2.remaps[1].block_id == 2 && plan2.remaps[1].skip == true);
    CHECK(plan2.remaps[2].block_id == 3);
    CHECK(plan2.remaps[2].from_base == 128 * 3);  // baked at true pos, first move
    CHECK(plan2.remaps[2].to_base == 128 * 2);
    CHECK(plan2.remaps[2].skip == false);
    CHECK(plan2.total_window_tokens == 128 * 3);
    CHECK(s.blocks()[2].remap_count == 1);  // stable slot: no new write
    CHECK(s.blocks()[3].remap_count == 1);

    // Tier canonicalization is another fp16 re-RoPE write and must contribute
    // to the diagnostic count even though it is not part of set_selection().
    s.record_block_rerope(3, s.blocks()[3].orig_pos_start);
    CHECK(s.blocks()[3].remap_count == 2);
}

static void test_immutable_source_selection_uses_bounded_delta_remaps() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    cfg.immutable_refresh_remaps = 2;
    cfg.immutable_refresh_abs_delta_tokens = 0;
    KvMemStore s(cfg);
    s.register_append(32 * 6);

    // A resident block uses one cheap in-place delta remap and records its new
    // active GPU frame.
    auto p1 = s.set_selection({0, 4});
    CHECK(p1.remaps.size() == 2);
    CHECK(p1.remaps[1].block_id == 4);
    CHECK(p1.remaps[1].from_base == 128);
    CHECK(p1.remaps[1].to_base == 32);
    CHECK(!p1.remaps[1].skip);
    CHECK(!p1.remaps[1].raw_refresh);
    CHECK(s.blocks()[4].baked_pos == 32);
    CHECK(s.blocks()[4].remap_count == 1);

    // A different selection starts at the current active frame.
    auto p2 = s.set_selection({0, 2, 4});
    CHECK(p2.remaps.size() == 3);
    CHECK(p2.remaps[2].block_id == 4);
    CHECK(p2.remaps[2].from_base == 32);
    CHECK(p2.remaps[2].to_base == 64);
    CHECK(s.blocks()[4].baked_pos == 64);
    CHECK(s.blocks()[4].remap_count == 2);

    // The next move crosses the configured two-write count limit and is
    // rebuilt from raw K, resetting the drift counters.
    auto p3 = s.set_selection({0, 4});
    CHECK(p3.remaps[1].from_base == 64);
    CHECK(p3.remaps[1].to_base == 32);
    CHECK(p3.remaps[1].raw_refresh);
    CHECK(s.blocks()[4].baked_pos == 32);
    CHECK(s.blocks()[4].remap_count == 0);

    // A block loaded from CPU/NVMe is always rebuilt because its GPU K slot is
    // newly allocated and has no valid active bake.
    s.set_block_tier(3, KvTier::CPU, 0, -1);
    auto p4 = s.set_selection({0, 3});
    CHECK(p4.remaps[1].raw_refresh);
    CHECK(!p4.remaps[1].skip);
}

static void test_cold_immutable_same_position_rebuilds_k() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    KvMemStore s(cfg);
    s.register_append(32 * 4);

    // Block 2 was previously compacted to slot 1, then evicted. Its recorded
    // bake equals the next target slot, but its working K no longer resides on
    // GPU. The plan must rebuild K from raw authority instead of skipping.
    (void)s.set_selection({0, 1});
    s.set_block_baked_pos(2, 32);
    s.set_block_tier(2, KvTier::CPU, 0, -1);
    const auto plan = s.set_selection({0, 2});
    CHECK(plan.stage_in.size() == 1);
    CHECK(plan.stage_in[0] == 2);
    CHECK(plan.remaps.size() == 2);
    CHECK(plan.remaps[1].from_base == 32);
    CHECK(plan.remaps[1].to_base == 32);
    CHECK(!plan.remaps[1].working_k_resident);
    CHECK(plan.remaps[1].raw_refresh);
    CHECK(!plan.remaps[1].skip);
}

static void test_default_profile_reuses_moved_resident_k() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    CHECK(cfg.optimize_stage_out);
    CHECK(cfg.optimize_stage_in);
    CHECK(cfg.optimize_pack);
    KvMemStore s(cfg);
    s.register_append(32 * 4);

    // Block 2 remains physically resident but moves from original position 64
    // to compact position 32. Default profiles use the bounded in-place
    // re-RoPE path and retain raw K as the periodic reset authority.
    const auto plan = s.set_selection({0, 2});
    CHECK(plan.remaps.size() == 2);
    CHECK(plan.remaps[1].working_k_resident);
    CHECK(plan.remaps[1].from_base == 64);
    CHECK(plan.remaps[1].to_base == 32);
    CHECK(!plan.remaps[1].raw_refresh);
    CHECK(!plan.remaps[1].skip);
    CHECK(s.blocks()[2].remap_count == 1);
}

static void test_explicit_force_raw_refresh_rebuilds_stable_selection() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    KvMemStore s(cfg);
    s.register_append(32 * 4);

    (void)s.set_selection({0, 2});
    const auto plan = s.set_selection({0, 2}, /*force_raw_refresh=*/true);
    CHECK(plan.remaps.size() == 2);
    CHECK(plan.stage_in.empty());
    for (const KvMemRemap &rm : plan.remaps) {
        CHECK(rm.working_k_resident);
        CHECK(rm.raw_refresh);
        CHECK(!rm.skip);
    }
}

static void test_stage_in_uses_tier_residency() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    KvMemStore s(cfg);
    s.register_append(128 * 4);

    auto plan = s.set_selection({0, 3});
    CHECK(plan.stage_out.size() == 2);
    s.set_block_tier(1, KvTier::CPU, 1, -1);
    s.set_block_tier(2, KvTier::CPU, 2, -1);

    auto plan2 = s.set_selection({0, 1, 3});
    CHECK(plan2.stage_in.size() == 1);
    CHECK(plan2.stage_in[0] == 1);
    CHECK(plan2.stage_out.empty());
}

static void test_stage_in_ablation_forces_full_reload() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    cfg.optimize_stage_in = false;
    KvMemStore s(cfg);
    s.register_append(32 * 5);

    // Even though blocks 0 and 2 remain selected, the ablation deliberately
    // stages every GPU block out and reloads the complete selected set.
    auto first = s.set_selection({0, 2, 4});
    CHECK(first.selection_overlap_blocks == 0);
    CHECK(first.gpu_reused_blocks == 0);
    CHECK(first.stage_out.size() == 5);
    CHECK(first.stage_in.size() == 3);
    for (const KvMemRemap &rm : first.remaps) {
        CHECK(!rm.working_k_resident);
        CHECK(rm.raw_refresh);
        CHECK(!rm.skip);
    }

    // Simulate the executor publishing the selected blocks back on GPU, then
    // reselect the same logical set. Natural overlap is observable, but actual
    // GPU reuse remains zero and all three blocks travel through the tiers.
    for (uint32_t id : first.stage_out) {
        s.set_block_tier(id, KvTier::CPU, static_cast<int32_t>(id), -1);
    }
    for (uint32_t id : std::vector<uint32_t>{0, 2, 4}) {
        s.set_block_tier(id, KvTier::GPU);
    }
    auto second = s.set_selection({0, 2, 4});
    CHECK(second.selection_overlap_blocks == 3);
    CHECK(second.gpu_reused_blocks == 0);
    CHECK(second.retained_position_stable == 3);
    CHECK(second.retained_position_moved == 0);
    CHECK(second.stage_out.size() == 3);
    CHECK(second.stage_in.size() == 3);
    for (const KvMemRemap &rm : second.remaps) {
        CHECK(!rm.working_k_resident);
        CHECK(rm.raw_refresh);
        CHECK(!rm.skip);
    }
}

static void test_topk_budget_sink_recent() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 4;   // budget = 4 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    KvMemStore s(cfg);
    s.register_append(128 * 10);   // 10 blocks, ids 0..9
    CHECK(s.budget_blocks() == 4);

    // Give middle blocks distinct attention scores; make block 5 and 6 hottest.
    std::vector<double> scores(10, 0.0);
    scores[5] = 100.0;
    scores[6] = 90.0;
    scores[3] = 10.0;
    s.accumulate_attn(scores);

    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 4);
    // Must contain sink (0) and recent (9), plus the two hottest middle (5,6).
    bool has0 = false, has9 = false, has5 = false, has6 = false;
    for (uint32_t id : sel) {
        if (id == 0) has0 = true;
        if (id == 9) has9 = true;
        if (id == 5) has5 = true;
        if (id == 6) has6 = true;
    }
    CHECK(has0 && has9 && has5 && has6);
    // Returned ascending.
    for (size_t i = 1; i < sel.size(); ++i) CHECK(sel[i] > sel[i - 1]);
}

static void test_topk_zero_recent_keeps_no_suffix() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 4;   // budget = 4 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;         // literal: no always-kept suffix
    KvMemStore s(cfg);
    s.register_append(128 * 10);   // ids 0..9

    std::vector<double> scores(10, 0.0);
    scores[3] = 80.0;
    scores[5] = 100.0;
    scores[6] = 90.0;
    scores[9] = -100.0;            // tail must not be pinned
    s.set_retrieval_scores(scores);

    const auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 4);
    CHECK(std::find(sel.begin(), sel.end(), 0) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 3) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 5) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 6) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 9) == sel.end());
}

static void test_budget_scaled_keep_allocation() {
    {
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/32,
            /*select_budget=*/32 * 1024,
            /*sink_blocks=*/-1,
            /*recent_blocks=*/-1,
            /*sink_tokens=*/-1,
            /*recent_tokens=*/-1);
        CHECK(keep.sink_target_tokens == 1024);
        CHECK(keep.recent_target_tokens == 4096);
        CHECK(keep.sink_blocks == 32);
        CHECK(keep.recent_blocks == 128);
        CHECK(keep.sink_effective_tokens == 1024);
        CHECK(keep.recent_effective_tokens == 4096);
        CHECK(keep.sink_source == KvMemKeepSource::Auto);
        CHECK(keep.recent_source == KvMemKeepSource::Auto);
    }
    {
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/512,
            /*select_budget=*/224 * 1024,
            -1, -1, -1, -1);
        CHECK(keep.sink_target_tokens == 2048);
        CHECK(keep.recent_target_tokens == 16384);
        CHECK(keep.sink_blocks == 4);
        CHECK(keep.recent_blocks == 32);
    }
    {
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/1024,
            /*select_budget=*/224 * 1024,
            -1, -1, -1, -1);
        CHECK(keep.sink_blocks == 2);
        CHECK(keep.recent_blocks == 16);
    }
    {
        // Explicit block values preserve historical experiments byte-for-byte.
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/512,
            /*select_budget=*/224 * 1024,
            /*sink_blocks=*/8,
            /*recent_blocks=*/0,
            /*sink_tokens=*/-1,
            /*recent_tokens=*/-1);
        CHECK(keep.sink_blocks == 8);
        CHECK(keep.recent_blocks == 0);
        CHECK(keep.sink_effective_tokens == 4096);
        CHECK(keep.recent_effective_tokens == 0);
        CHECK(keep.sink_source == KvMemKeepSource::Blocks);
        CHECK(keep.recent_source == KvMemKeepSource::Blocks);
    }
    {
        // Explicit token values are rounded up independently of block size.
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/512,
            /*select_budget=*/32 * 1024,
            /*sink_blocks=*/-1,
            /*recent_blocks=*/-1,
            /*sink_tokens=*/1025,
            /*recent_tokens=*/4097);
        CHECK(keep.sink_blocks == 3);
        CHECK(keep.recent_blocks == 9);
        CHECK(keep.sink_effective_tokens == 1536);
        CHECK(keep.recent_effective_tokens == 4608);
        CHECK(keep.sink_source == KvMemKeepSource::Tokens);
        CHECK(keep.recent_source == KvMemKeepSource::Tokens);
    }

    bool conflict_threw = false;
    try {
        (void)resolve_kvmem_keep_allocation(
            32, 32 * 1024,
            /*sink_blocks=*/1, /*recent_blocks=*/-1,
            /*sink_tokens=*/1024, /*recent_tokens=*/-1);
    } catch (const std::runtime_error &) {
        conflict_threw = true;
    }
    CHECK(conflict_threw);

    bool budget_threw = false;
    try {
        (void)resolve_kvmem_keep_allocation(
            1024, 32 * 1024,
            /*sink_blocks=*/20, /*recent_blocks=*/20,
            /*sink_tokens=*/-1, /*recent_tokens=*/-1);
    } catch (const std::runtime_error &) {
        budget_threw = true;
    }
    CHECK(budget_threw);
}

static void test_topk_mandatory_blocks_stay_inside_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;  // four blocks including mandatory suffix
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    std::vector<double> scores(10, 0.0);
    scores[2] = 100.0;
    scores[3] = 90.0;
    scores[4] = 80.0;
    // The low-scoring query suffix must replace ordinary top-k candidates,
    // never extend the compact window to six blocks.
    scores[8] = -10.0;
    scores[9] = -20.0;
    s.set_retrieval_scores(scores);
    const auto sel = s.pick_topk_blocks({8, 9});
    CHECK(sel.size() == 4);
    CHECK(std::find(sel.begin(), sel.end(), 0) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 2) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 8) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 9) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 3) == sel.end());
}

static void test_mandatory_overlap_deduplicates_and_recent_is_best_effort() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.sink_blocks = 2;
    cfg.recent_blocks = 3;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    // Mandatory block 0 overlaps the sink and is repeated. The hard union is
    // {0,1,5}, leaving exactly one best-effort recent slot for block 9.
    const auto sel = s.pick_topk_blocks({0, 0, 5});
    CHECK(sel.size() == 4);
    for (uint32_t id : {0u, 1u, 5u, 9u}) {
        CHECK(std::find(sel.begin(), sel.end(), id) != sel.end());
    }
    CHECK(std::find(sel.begin(), sel.end(), 8) == sel.end());
}

static void test_mandatory_hard_union_overflow_trims_newest() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 2;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    // sink {0} + mandatory {1,2,3,4} is 5 > budget 4. Keep the newest
    // mandatory ids: {0, 2, 3, 4}.
    const auto sel = s.pick_topk_blocks({1, 2, 3, 4});
    CHECK(sel.size() == 4);
    for (uint32_t id : {0u, 2u, 3u, 4u}) {
        CHECK(std::find(sel.begin(), sel.end(), id) != sel.end());
    }
    CHECK(std::find(sel.begin(), sel.end(), 1) == sel.end());

    const auto sem = s.pick_semantic_groups(
        {{32 * 5, 32 * 7}}, {1.0}, {1, 2, 3, 4});
    CHECK(sem.size() == 4);
    for (uint32_t id : {0u, 2u, 3u, 4u}) {
        CHECK(std::find(sem.begin(), sem.end(), id) != sem.end());
    }
    CHECK(std::find(sem.begin(), sem.end(), 1) == sem.end());
}

static void test_prefill_pressure_mandatory_blocks_stay_inside_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.prefill_budget = 32 * 5;
    cfg.sink_blocks = 1;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    const auto sel = s.pick_prefill_pressure_blocks({3, 4});
    CHECK(sel.size() == 5);
    CHECK(std::find(sel.begin(), sel.end(), 0) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 3) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 4) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 8) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 9) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 7) == sel.end());

    // sink {0} + mandatory {1..5} exceeds prefill budget 5; keep newest.
    const auto trimmed = s.pick_prefill_pressure_blocks({1, 2, 3, 4, 5});
    CHECK(trimmed.size() == 5);
    for (uint32_t id : {0u, 2u, 3u, 4u, 5u}) {
        CHECK(std::find(trimmed.begin(), trimmed.end(), id) != trimmed.end());
    }
    CHECK(std::find(trimmed.begin(), trimmed.end(), 1) == trimmed.end());
}

static void test_prefill_pressure_sink_full_recent_tail() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 5;  // budget = 5 blocks
    cfg.sink_blocks = 2;
    cfg.recent_blocks = 0;        // pressure policy deliberately ignores this
    KvMemStore s(cfg);
    s.register_append(128 * 10);  // ids 0..9

    // Make old middle blocks arbitrarily hot. Pressure selection must remain a
    // pure positional sink + newest-tail policy: [0,1] + [7,8,9].
    std::vector<double> scores(10, 0.0);
    scores[2] = 1000.0;
    scores[3] = 999.0;
    scores[4] = 998.0;
    s.set_retrieval_scores(scores);
    const std::vector<uint32_t> expected{0, 1, 7, 8, 9};
    CHECK(s.pick_prefill_pressure_blocks() == expected);

    // A prior semantic working set and a different score vector cannot affect
    // the next pressure choice.
    (void)s.set_selection({0, 1, 2, 3, 4});
    scores.assign(10, 0.0);
    scores[5] = 5000.0;
    scores[6] = 4000.0;
    s.set_retrieval_scores(scores);
    CHECK(s.pick_prefill_pressure_blocks() == expected);

    // `recent_blocks` is a semantic-selector pin, not the pressure tail size.
    cfg.recent_blocks = 4;
    KvMemStore with_semantic_recent(cfg);
    with_semantic_recent.register_append(128 * 10);
    CHECK(with_semantic_recent.pick_prefill_pressure_blocks() == expected);
}

static void test_prefill_pressure_edges() {
    // With no sink, the whole pressure budget is the newest tail.
    {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = 32 * 3;
        cfg.sink_blocks = 0;
        KvMemStore s(cfg);
        s.register_append(32 * 6);
        const std::vector<uint32_t> expected{3, 4, 5};
        CHECK(s.pick_prefill_pressure_blocks() == expected);
    }

    // Sink is clamped by the budget, leaving no tail when it consumes all slots.
    {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = 32 * 3;
        cfg.sink_blocks = 8;
        cfg.recent_blocks = 8;
        KvMemStore s(cfg);
        s.register_append(32 * 7 + 11);  // partial block id 7 included in n
        const std::vector<uint32_t> expected{0, 1, 2};
        CHECK(s.pick_prefill_pressure_blocks() == expected);
    }

    // Below budget remains identity, including the trailing partial block.
    {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = 32 * 8;
        cfg.sink_blocks = 2;
        KvMemStore s(cfg);
        s.register_append(32 * 3 + 7);
        const std::vector<uint32_t> expected{0, 1, 2, 3};
        CHECK(s.pick_prefill_pressure_blocks() == expected);
    }

    KvMemStoreConfig cfg;
    KvMemStore empty(cfg);
    CHECK(empty.pick_prefill_pressure_blocks().empty());
}

static void test_prefill_watermark_offload_predicate() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.sink_blocks = 1;
    cfg.gpu_high_watermark = 0.90;
    cfg.gpu_low_watermark = 0.80;
    KvMemStore s(cfg);
    s.register_append(32 * 10);  // 10 blocks > 4-block budget
    const uint32_t pool = 32 * 8;  // 256-token slot pool

    CHECK(s.gpu_high_watermark_tokens(pool) == 230);  // 256 * 0.90
    CHECK(s.gpu_low_watermark_tokens(pool) == 204);   // 256 * 0.80
    // History > semantic budget is not enough: GPU still has gen_reserve slack.
    CHECK(!s.prefill_needs_offload(0, 32, pool));
    CHECK(!s.prefill_needs_offload(32 * 4, 32, pool));  // 160 < 230
    CHECK(s.prefill_needs_offload(200, 40, pool));    // over high watermark
    CHECK(s.prefill_needs_offload(240, 32, pool));    // over hard pool
    CHECK(!s.prefill_needs_offload(0, 0, pool));

    KvMemStoreConfig under = cfg;
    under.select_budget = 32 * 16;
    KvMemStore fit(under);
    fit.register_append(32 * 4);
    CHECK(!fit.prefill_needs_offload(96, 32, pool));  // 128 < 230, under budget
}

static void test_prefill_pressure_budget_is_independent_from_semantic_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.prefill_budget = 32 * 8;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(32 * 12);

    // Pressure construction keeps a wide sink+tail context.
    const std::vector<uint32_t> pressure{0, 5, 6, 7, 8, 9, 10, 11};
    CHECK(s.prefill_budget_blocks() == 8);
    CHECK(s.pick_prefill_pressure_blocks() == pressure);

    // The final semantic selector still contracts to four blocks. The hotter
    // middle blocks win independently of the wider prefill construction frame.
    std::vector<double> scores(12, 0.0);
    scores[2] = 4.0;
    scores[3] = 3.0;
    scores[4] = 2.0;
    s.set_retrieval_scores(scores);
    const std::vector<uint32_t> semantic{0, 2, 3, 4};
    CHECK(s.budget_blocks() == 4);
    CHECK(s.pick_topk_blocks() == semantic);
}

static void test_request_semantic_budget_override_is_scoped_and_bounded() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 8;   // CLI/server maximum.
    cfg.prefill_budget = 32 * 12;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    KvMemStore s(cfg);
    s.register_append(32 * 16);

    std::vector<double> scores(16, 0.0);
    for (uint32_t id = 0; id < scores.size(); ++id) {
        scores[id] = static_cast<double>(id);
    }
    s.set_retrieval_scores(scores);

    CHECK(s.select_budget_tokens() == 32 * 8);
    CHECK(s.runtime_select_budget_tokens() == 0);
    CHECK(s.pick_topk_blocks().size() == 8);
    CHECK(s.pick_prefill_pressure_blocks().size() == 12);

    // A request may narrow semantic selection while pressure prefill remains
    // at the configured 12-block construction window.
    s.set_runtime_select_budget(32 * 4);
    CHECK(s.select_budget_tokens() == 32 * 4);
    CHECK(s.runtime_select_budget_tokens() == 32 * 4);
    CHECK(s.pick_topk_blocks().size() == 4);
    CHECK(s.pick_prefill_pressure_blocks().size() == 12);

    // Zero restores the CLI maximum for the next request.
    s.set_runtime_select_budget(0);
    CHECK(s.select_budget_tokens() == 32 * 8);
    CHECK(s.pick_topk_blocks().size() == 8);

    bool over_max_threw = false;
    try {
        s.set_runtime_select_budget(32 * 9);
    } catch (const std::invalid_argument &) {
        over_max_threw = true;
    }
    CHECK(over_max_threw);

    bool unaligned_threw = false;
    try {
        s.set_runtime_select_budget(32 * 4 + 1);
    } catch (const std::invalid_argument &) {
        unaligned_threw = true;
    }
    CHECK(unaligned_threw);

    bool keep_allocation_threw = false;
    try {
        s.set_runtime_select_budget(32);  // sink + recent need two blocks.
    } catch (const std::invalid_argument &) {
        keep_allocation_threw = true;
    }
    CHECK(keep_allocation_threw);
    CHECK(s.runtime_select_budget_tokens() == 0);
}

static void test_quota_policy_sink_recent_retrieval_profile() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 6;   // budget = 6 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.select_policy = KvMemSelectPolicy::Quota;
    cfg.retrieval_blocks = 2;
    cfg.profile_blocks = 2;
    KvMemStore s(cfg);
    s.register_append(128 * 12);

    std::vector<double> profile(12, 0.0);
    profile[3] = 80.0;
    profile[4] = 70.0;
    s.accumulate_attn(profile);

    std::vector<double> retrieval(12, 0.0);
    retrieval[7] = 100.0;
    retrieval[8] = 90.0;
    s.set_retrieval_scores(retrieval);

    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 6);
    bool has0 = false, has11 = false, has3 = false, has4 = false;
    bool has7 = false, has8 = false;
    for (uint32_t id : sel) {
        if (id == 0) has0 = true;
        if (id == 11) has11 = true;
        if (id == 3) has3 = true;
        if (id == 4) has4 = true;
        if (id == 7) has7 = true;
        if (id == 8) has8 = true;
    }
    CHECK(has0 && has11 && has3 && has4 && has7 && has8);
    for (size_t i = 1; i < sel.size(); ++i) CHECK(sel[i] > sel[i - 1]);
}

static void test_topk_all_fit() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 100;  // huge budget
    KvMemStore s(cfg);
    s.register_append(128 * 3);
    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 3);
    CHECK(sel[0] == 0 && sel[1] == 1 && sel[2] == 2);
}

static void test_topk_empty() {
    KvMemStoreConfig cfg;
    KvMemStore s(cfg);
    CHECK(s.pick_topk_blocks().empty());
    auto plan = s.set_selection({0, 1});  // stale ids on empty store
    CHECK(plan.remaps.empty());
    CHECK(plan.total_window_tokens == 0);
}

static void test_tier_metadata() {
    KvMemStoreConfig cfg;
    KvMemStore s(cfg);
    s.register_append(128 * 2);
    s.set_block_tier(1, KvTier::CPU, 7, -1);
    CHECK(s.blocks()[1].tier == KvTier::CPU);
    CHECK(s.blocks()[1].cpu_slot == 7);
    CHECK(s.blocks()[1].nvme_slot == -1);
    s.set_block_tier(1, KvTier::SSD, -1, 3);
    CHECK(s.blocks()[1].tier == KvTier::SSD);
    CHECK(s.blocks()[1].cpu_slot == -1);
    CHECK(s.blocks()[1].nvme_slot == 3);
    s.set_block_baked_pos(1, 128);
    CHECK(s.blocks()[1].baked_pos == 128);
    s.set_block_baked_pos(1, 0);
    CHECK(s.blocks()[1].baked_pos == 0);
}

static void test_inclusive_tier_metadata() {
    KvMemStoreConfig cfg;
    cfg.optimize_stage_out = true;
    KvMemStore s(cfg);
    s.register_append(128 * 2);

    // First spill leaves both a CPU cache copy and clean SSD backing.
    s.set_block_ssd_backing(1, 3, true);
    s.set_block_tier(1, KvTier::CPU, 7, 3);
    CHECK(s.blocks()[1].tier == KvTier::CPU);
    CHECK(s.blocks()[1].cpu_slot == 7);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);

    // Promotion makes GPU active without discarding either clean copy.
    s.set_block_tier(1, KvTier::GPU);
    CHECK(s.blocks()[1].tier == KvTier::GPU);
    CHECK(s.blocks()[1].cpu_slot == 7);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);

    // A CPU-cache eviction must not change active GPU residency.
    s.set_block_cpu_copy(1, -1);
    CHECK(s.blocks()[1].tier == KvTier::GPU);
    CHECK(s.blocks()[1].cpu_slot == -1);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);
}

static void test_stage_out_off_exclusive_tier_metadata() {
    KvMemStoreConfig cfg;
    cfg.optimize_stage_out = false;
    KvMemStore s(cfg);
    s.register_append(128 * 2);

    // Stage-out-off uses exclusive move semantics. Promotion must drop every
    // lower-tier handle.
    s.set_block_tier(1, KvTier::SSD, -1, 3);
    CHECK(s.blocks()[1].tier == KvTier::SSD);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);
    s.set_block_tier(1, KvTier::GPU);
    CHECK(s.blocks()[1].tier == KvTier::GPU);
    CHECK(s.blocks()[1].cpu_slot == -1);
    CHECK(s.blocks()[1].nvme_slot == -1);
    CHECK(!s.blocks()[1].ssd_clean);
}

static void test_factorial_optimization_controls_are_orthogonal() {
    for (bool stage_out : {false, true}) {
        for (bool stage_in : {false, true}) {
            for (bool pack : {false, true}) {
                KvMemStoreConfig cfg;
                cfg.block_tokens = 32;
                cfg.immutable_source_k = true;
                cfg.optimize_stage_out = stage_out;
                cfg.optimize_stage_in = stage_in;
                cfg.optimize_pack = pack;

                KvMemStore selection_store(cfg);
                selection_store.register_append(32 * 3);
                (void) selection_store.set_selection({0, 2});
                const KvMemPlan plan =
                    selection_store.set_selection({0, 2});
                CHECK(plan.gpu_reused_blocks ==
                      (stage_in ? 2u : 0u));
                CHECK(plan.stage_in.size() ==
                      (stage_in ? 0u : 2u));
                CHECK(plan.stage_out.size() ==
                      (stage_in ? 1u : 3u));

                KvMemStore tier_store(cfg);
                tier_store.register_append(32);
                tier_store.set_block_ssd_backing(0, 9, true);
                tier_store.set_block_tier(
                    0, KvTier::SSD, -1, 9);
                tier_store.set_block_tier(
                    0, KvTier::GPU);
                CHECK((tier_store.blocks()[0].nvme_slot >= 0) ==
                      stage_out);
                CHECK(tier_store.blocks()[0].ssd_clean ==
                      stage_out);
            }
        }
    }
}

static void test_truncate_to() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;

    // (1) truncate on a block boundary: 5 full blocks -> keep exactly 3.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 5);           // ids 0..4, 640 tokens
        auto dropped = s.truncate_to(128 * 3);
        CHECK(s.block_count() == 3);
        CHECK(dropped.size() == 2);           // blocks 4 then 3 popped (back-first)
        CHECK(dropped[0].block_id == 4);
        CHECK(dropped[1].block_id == 3);
        CHECK(s.blocks()[2].n_tokens == 128);
        // Re-registering the suffix restores the original layout.
        s.register_append(128 * 2);
        CHECK(s.block_count() == 5);
        CHECK(s.blocks()[3].orig_pos_start == 128 * 3);
        CHECK(s.blocks()[4].n_tokens == 128);
    }

    // (2) truncate mid-block: shrink the trailing partial block in place.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 3);           // ids 0..2
        auto dropped = s.truncate_to(128 * 2 + 40);
        CHECK(s.block_count() == 3);          // block 2 shrunk, not popped
        CHECK(dropped.empty());
        CHECK(s.blocks()[2].n_tokens == 40);
        CHECK(s.blocks()[2].orig_pos_start == 128 * 2);
        // Growth continues from the shrunk boundary, not the old end.
        s.register_append(10);
        CHECK(s.block_count() == 3);
        CHECK(s.blocks()[2].n_tokens == 50);
    }

    // (3) truncate straddling a partial trailing block: pop the partial, shrink
    // the now-trailing full block.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 2 + 30);      // ids 0,1 full; id 2 partial (30)
        CHECK(s.block_count() == 3);
        auto dropped = s.truncate_to(128 + 50); // keep block0 full, block1 -> 50
        CHECK(s.block_count() == 2);
        CHECK(dropped.size() == 1);
        CHECK(dropped[0].block_id == 2);
        CHECK(s.blocks()[1].n_tokens == 50);
    }

    // (4) no-op cases: token_pos == total (the ckpt_M case) and > total.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 2 + 10);      // 266 tokens
        CHECK(s.truncate_to(266).empty());
        CHECK(s.block_count() == 3);
        CHECK(s.truncate_to(9999).empty());
        CHECK(s.block_count() == 3);
    }

    // (5) truncate to 0: everything dropped, dense indices intact on regrow.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 2);
        auto dropped = s.truncate_to(0);
        CHECK(s.block_count() == 0);
        CHECK(dropped.size() == 2);
        s.register_append(128);
        CHECK(s.block_count() == 1);
        CHECK(s.blocks()[0].block_id == 0);
        CHECK(s.blocks()[0].orig_pos_start == 0);
    }

    // (6) dropped list carries tier slots so the executor can release them.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 3);
        s.set_block_tier(2, KvTier::CPU, 7, -1);
        auto dropped = s.truncate_to(128 * 2);
        CHECK(dropped.size() == 1);
        CHECK(dropped[0].block_id == 2);
        CHECK(dropped[0].tier == KvTier::CPU);
        CHECK(dropped[0].cpu_slot == 7);
    }
}

// DeltaNet-retrieval config surfaces through KvMemStoreConfig with sane defaults
// and that retrieval scores (which the DeltaNet scorer writes via
// set_retrieval_scores) rank blocks exactly like the mean-k path — the store is
// method-agnostic, so a DeltaNet score set feeds pick_topk identically.
static void test_deltanet_config_and_scores() {
    KvMemStoreConfig cfg;
    // Defaults: DeltaNet retrieval off by default; decay on; topk = 4.
    CHECK(cfg.retrieval_method == KvMemRetrievalMethod::MeanK);
    CHECK(cfg.deltanet_decay == true);
    CHECK(cfg.deltanet_topk_q == 4);
    CHECK(cfg.deltanet_topk_h == 4);
    CHECK(cfg.deltanet_layers == 0);
    CHECK(cfg.deltanet_layer_policy == KvMemDeltaNetLayerPolicy::Even);
    CHECK(cfg.deltanet_mem_budget_bytes == 0);

    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 4;   // budget = 4 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.retrieval_method = KvMemRetrievalMethod::DeltaNet;
    KvMemStore s(cfg);
    s.register_append(128 * 10);   // 10 blocks, ids 0..9

    // A DeltaNet score vector (what kvmem_retrieval_score_deltanet produces):
    // blocks 4 and 7 rank highest in the middle.
    std::vector<double> dn_scores(10, 0.0);
    dn_scores[4] = 5.0;
    dn_scores[7] = 4.0;
    dn_scores[2] = 1.0;
    s.set_retrieval_scores(dn_scores);

    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 4);
    bool has0 = false, has9 = false, has4 = false, has7 = false;
    for (uint32_t id : sel) {
        if (id == 0) has0 = true;
        if (id == 9) has9 = true;
        if (id == 4) has4 = true;
        if (id == 7) has7 = true;
    }
    // Sink (0) + recent (9) kept unconditionally; the two hottest middle (4,7) win.
    CHECK(has0 && has9 && has4 && has7);
    for (size_t i = 1; i < sel.size(); ++i) CHECK(sel[i] > sel[i - 1]);
}

static void test_round_groups_are_selected_whole() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 8;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.semantic_expansion = KvMemSemanticExpansion::Round;
    cfg.group_score_reduce =
        KvMemGroupScoreReduce::LengthNormalizedMass;
    cfg.group_length_norm_alpha = 0.5;
    KvMemStore s(cfg);
    s.register_append(32 * 20);

    // Four variable-length groups. Group 1 is hottest and overlaps blocks
    // [4,7]; group 2 is next and overlaps [8,10]. Sink block 0 and recent block
    // 19 leave six slots, so group 1 must be admitted whole and group 2 skipped
    // rather than partially selected.
    const std::vector<std::pair<uint32_t, uint32_t>> groups{
        {32, 32 * 4},
        {32 * 4 + 3, 32 * 8 - 5},
        {32 * 8, 32 * 11},
        {32 * 12, 32 * 14},
    };
    const std::vector<double> scores{1.0, 10.0, 9.0, 2.0};
    const auto selected = s.pick_semantic_groups(groups, scores);
    CHECK(selected.size() <= 8);
    CHECK(std::find(selected.begin(), selected.end(), 0) != selected.end());
    CHECK(std::find(selected.begin(), selected.end(), 19) != selected.end());
    for (uint32_t id = 4; id <= 7; ++id) {
        CHECK(std::find(selected.begin(), selected.end(), id) != selected.end());
    }
    // The second-ranked group cannot fit after the first: none of its unique
    // blocks may leak in through a partial admission.
    for (uint32_t id = 8; id <= 10; ++id) {
        CHECK(std::find(selected.begin(), selected.end(), id) == selected.end());
    }
}

static void test_round_groups_charge_shared_boundary_once() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 64;
    cfg.select_budget = 64 * 4;
    cfg.sink_blocks = 0;
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(64 * 8);

    // Both groups overlap block 2. Their union is blocks [1,3], not four
    // blocks, so both fit and the shared physical transfer is charged once.
    const std::vector<std::pair<uint32_t, uint32_t>> groups{
        {64, 64 * 2 + 10},
        {64 * 2 + 10, 64 * 4},
    };
    const auto selected =
        s.pick_semantic_groups(groups, std::vector<double>{2.0, 1.0}, {7});
    CHECK(selected.size() == 4);
    for (uint32_t id : {1u, 2u, 3u, 7u}) {
        CHECK(std::find(selected.begin(), selected.end(), id) != selected.end());
    }
}

static void test_media_groups_with_shared_boundary() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = cfg.prefill_budget = 6*32;
    cfg.sink_blocks = 1;
    KvMemStore store(cfg);
    store.register_append(12*32);
    // Two adjacent images share block 6, so blocks 4..8 form one group.
    store.set_media_ranges({{4*32+3, 6*32+9}, {6*32+10, 9*32}});
    for (bool prefill : {false, true}) {
        auto selected = prefill ? store.pick_prefill_pressure_blocks({}) : store.pick_topk_blocks({});
        CHECK((selected == std::vector<uint32_t>{0, 4, 5, 6, 7, 8}));
        bool threw = false;
        try {
            selected = prefill ? store.pick_prefill_pressure_blocks({11}) : store.pick_topk_blocks({11});
        } catch (const std::runtime_error &) { threw = true; }
        // Incoming prefill rows are hard requirements. Retrieval can drop the
        // suffix and let the server continue from the completed first pass.
        CHECK(threw == prefill);
        if (!prefill) CHECK((selected == std::vector<uint32_t>{0, 4, 5, 6, 7, 8}));
    }
    // A historical image is either entirely selected or absent.
    store.set_media_ranges({{2*32+1, 4*32}, {9*32, 11*32}});
    auto selected = store.pick_topk_blocks({11});
    CHECK(std::binary_search(selected.begin(), selected.end(), 2) == std::binary_search(selected.begin(), selected.end(), 3));
    CHECK(std::binary_search(selected.begin(), selected.end(), 9));
    CHECK(std::binary_search(selected.begin(), selected.end(), 10));
}

static void test_media_suffix_over_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = cfg.prefill_budget = 6*32;
    cfg.sink_blocks = 1;
    KvMemStore store(cfg);
    store.register_append(12*32);
    store.set_media_ranges({{2*32+1, 4*32}, {6*32+1, 8*32}});
    // Text alone fits, but sink + image + suffix do not. Keep newest text.
    CHECK((store.pick_topk_blocks({8, 9, 10, 11}) == std::vector<uint32_t>{0, 6, 7, 9, 10, 11}));
    // An unsorted/duplicated mandatory list cannot change recency priority.
    CHECK((store.pick_topk_blocks({11, 8, 10, 9, 8}) == std::vector<uint32_t>{0, 6, 7, 9, 10, 11}));
    // Exact fit including a shared image/text boundary block.
    CHECK((store.pick_topk_blocks({7, 9, 10, 11}) == std::vector<uint32_t>{0, 6, 7, 9, 10, 11}));
    // Older image groups remain atomic even when the suffix intersects them.
    CHECK((store.pick_topk_blocks({3, 8, 9, 10, 11}) == std::vector<uint32_t>{0, 6, 7, 9, 10, 11}));
    // A single image that cannot fit intact is still rejected.
    store.set_media_ranges({{2*32, 8*32}});
    bool threw = false;
    try { store.pick_topk_blocks({11}); }
    catch (const std::runtime_error &) { threw = true; }
    CHECK(threw);
}

static void test_lrfu_round_and_restore() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 64; cfg.select_budget = 192; cfg.sink_blocks = 1; cfg.recent_blocks = 1;
    cfg.lrfu = true; cfg.lrfu_top_fraction = 1;
    KvMemStore store(cfg); store.register_append(384);
    store.begin_retrieval_round();
    std::vector<double> scores(6, -0.2); scores[2] = 0.8; scores[3] = 0.6;
    store.set_retrieval_scores(scores);
    const auto chosen = store.pick_topk_blocks();
    CHECK(chosen == std::vector<uint32_t>({0, 2, 5}));
    CHECK(store.retrieval_checkpoint().scores->empty()); // Preview is not a committed update.
    CHECK(store.commit_retrieval_round());
    const auto first = store.retrieval_checkpoint();
    store.begin_retrieval_round(); scores[2] = 0.1; scores[3] = 0.3;
    store.set_retrieval_scores(scores); store.set_retrieval_scores(scores);
    CHECK(std::abs(store.selection_score(2) - 0.5) < 1e-12);
    CHECK(std::abs(store.selection_score(3) - 0.6) < 1e-12);
    CHECK(store.commit_retrieval_round());
    scores[2] = 100; store.set_retrieval_scores(scores);
    const auto second = store.retrieval_checkpoint();
    CHECK(store.pick_topk_blocks() == std::vector<uint32_t>({0, 3, 5}));
    CHECK(!store.commit_retrieval_round());
    store.restore_retrieval(first);
    CHECK(store.pick_topk_blocks() == chosen); // Rollback removes rewards on surviving blocks too.
    std::vector<uint8_t> bytes;
    kvmem::SnapshotWriter out([&](const void* p, size_t n) {
        const auto* first = static_cast<const uint8_t*>(p); bytes.insert(bytes.end(), first, first + n);
    });
    kvmem::write_lrfu(out, first);
    const std::vector<kvmem::LrfuPoint> points{{384, first}, {512, second}};
    CHECK(kvmem::lrfu_input_point(128, 64, first).state.scores->size() == 2);
    CHECK(kvmem::lrfu_prefix_points(points, 384, 384) == 0);
    CHECK(kvmem::lrfu_prefix_points(points, 512, 512) == 1);
    CHECK(kvmem::lrfu_prefix_points(points, 500, 600) == 1);
    CHECK(kvmem::lrfu_prefix_points(points, 600, 600) == 2);
    kvmem::write_lrfu_points(out, points);
    size_t offset = 0;
    kvmem::SnapshotReader in([&](void* p, size_t n) { std::memcpy(p, bytes.data() + offset, n); offset += n; }, bytes.size());
    store.restore_retrieval(kvmem::read_lrfu(in, 6));
    const auto restored_points = kvmem::read_lrfu_points(in, 512, 64);
    CHECK(restored_points.size() == 2 && restored_points.back().state.round == 2);
    CHECK(*restored_points.front().state.scores == *first.scores);
    scores[3] = 100; store.set_retrieval_scores(scores);
    CHECK(store.pick_topk_blocks() == chosen); // Restored committed round cannot be rewarded twice.
    store.register_append(64); CHECK(store.selection_score(6) == 0);
    kvmem::LrfuHistory half(0, 0.5); half.begin_round(); half.preview({-.8, -.6, -.2, -.2}); half.commit();
    CHECK(half.score(0) == -.8 && half.score(1) == -.6 && half.score(2) == -.2 && half.score(3) == -.2);
}

static void test_lrfu_signed_hit_order() {
    // Equal prior history must never prefer a lower current relevance solely
    // because top-fraction omitted its update. Cover negative, mixed and zero ties.
    for (const auto& input : {std::vector<double>{-.8,-.6,-.2,-.2},
                              std::vector<double>{-.8,-.2,0,.2,.6,.8},
                              std::vector<double>{-.8,0,0,0},
                              std::vector<double>{0,.2,.6,.8}}) {
        for (double fraction : {.1,.5,.9,1.}) {
            kvmem::LrfuHistory history(0,fraction);history.begin_round();history.preview(input);
            for(size_t i=0;i<input.size();++i) {
                if(input[i]<0 || fraction==1) CHECK(history.score(i)==input[i]);
                if(i) CHECK(history.score(i)>=history.score(i-1));
            }
        }
    }
    // Existing nonnegative hit behavior and decay are unchanged; negative
    // evidence applies even outside the positive hit set.
    kvmem::LrfuHistory history(.5,.5);history.begin_round();history.preview({.1,.2,.6,.8});
    CHECK(history.score(0)==0 && history.score(1)==0 && history.score(2)==.6 && history.score(3)==.8);
    CHECK(history.commit());const auto checkpoint=history.checkpoint();
    history.begin_round();history.preview({-.8,-.2,.4,.8});history.preview({-.8,-.2,.4,.8});
    CHECK(std::abs(history.score(0)+.8)<1e-12 && std::abs(history.score(1)+.2)<1e-12);
    CHECK(std::abs(history.score(2)-.7)<1e-12 && std::abs(history.score(3)-1.2)<1e-12);
    CHECK(history.commit() && !history.commit());history.restore(checkpoint);
    CHECK(history.score(0)==0 && history.score(2)==.6);

    KvMemStoreConfig cfg;cfg.block_tokens=64;cfg.select_budget=128;cfg.sink_blocks=0;cfg.recent_blocks=0;
    cfg.lrfu=true;cfg.lrfu_lambda=0;cfg.lrfu_top_fraction=.5;
    KvMemStore store(cfg);store.register_append(256);store.begin_retrieval_round();
    store.set_retrieval_scores({-.8,-.6,-.2,-.1});
    CHECK(store.pick_topk_blocks()==std::vector<uint32_t>({2,3}));
}

int main() {
    test_lrfu_signed_hit_order();
    test_lrfu_round_and_restore();
    test_media_suffix_over_budget();
    test_media_groups_with_shared_boundary();
    test_register_append();
    test_selection_diff_and_remap();
    test_immutable_source_selection_uses_bounded_delta_remaps();
    test_cold_immutable_same_position_rebuilds_k();
    test_default_profile_reuses_moved_resident_k();
    test_explicit_force_raw_refresh_rebuilds_stable_selection();
    test_stage_in_uses_tier_residency();
    test_stage_in_ablation_forces_full_reload();
    test_topk_budget_sink_recent();
    test_topk_zero_recent_keeps_no_suffix();
    test_budget_scaled_keep_allocation();
    test_topk_mandatory_blocks_stay_inside_budget();
    test_mandatory_overlap_deduplicates_and_recent_is_best_effort();
    test_mandatory_hard_union_overflow_trims_newest();
    test_prefill_pressure_mandatory_blocks_stay_inside_budget();
    test_prefill_pressure_sink_full_recent_tail();
    test_prefill_pressure_edges();
    test_prefill_watermark_offload_predicate();
    test_prefill_pressure_budget_is_independent_from_semantic_budget();
    test_request_semantic_budget_override_is_scoped_and_bounded();
    test_quota_policy_sink_recent_retrieval_profile();
    test_topk_all_fit();
    test_topk_empty();
    test_tier_metadata();
    test_inclusive_tier_metadata();
    test_stage_out_off_exclusive_tier_metadata();
    test_factorial_optimization_controls_are_orthogonal();
    test_truncate_to();
    test_deltanet_config_and_scores();
    test_round_groups_are_selected_whole();
    test_round_groups_charge_shared_boundary_once();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
