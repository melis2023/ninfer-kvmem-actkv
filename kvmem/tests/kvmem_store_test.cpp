// KvMemStore host-logic unit test (block-sparse KV, Task #37).
//
// Pure host logic — no GPU. Covers: block registration (partial-block growth),
// the working-set diff (stage-in/out + window remap plan), and the built-in
// cumulative-attention top-k selection (sink/recent preservation + budget).

#include "kvmem/kvmem_store.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

    // The configured budget is the answer: the ActKV Sec. 6 adaptive budget used to
    // narrow this through a runtime override, and that policy (plus its setter) was
    // removed after it cost 23 needles of recall on the 40-needle / 256k test.
    CHECK(s.select_budget_tokens() == 32 * 8);
    CHECK(s.budget_blocks() == 8);
    CHECK(s.pick_topk_blocks().size() == 8);
    CHECK(s.pick_prefill_pressure_blocks().size() == 12);
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

// --- ActKV (arXiv:2609.31395 Sec. 5.2 Alg. 1): action-oriented LRFU ---------
//   KS_t(i) = lambda * KS_{t-1}(i) + (hit at t ? retrieval_t(i) : 0)

static void test_lrfu_defaults_match_actkv() {
    KvMemStoreConfig cfg;
    CHECK(cfg.action_lrfu == false);
    CHECK(cfg.lrfu_lambda > 0.49f && cfg.lrfu_lambda < 0.51f);
    CHECK(cfg.lrfu_top_frac > 0.89f && cfg.lrfu_top_frac < 0.91f);
}

static void test_lrfu_off_keeps_one_shot_ranking() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;   // action_lrfu defaults off
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 2.0, 3.0, 4.0});
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(s.blocks()[i].retrieval_score == double(i + 1));
        CHECK(s.blocks()[i].attn_score == double(i + 1));
        CHECK(s.blocks()[i].select_score == double(i + 1));
        CHECK(s.blocks()[i].lrfu_score == 0.0);
    }
}

static void test_lrfu_accumulates_and_decays() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 1);
    CHECK(s.blocks()[0].lrfu_score == 1.0);
    CHECK(s.blocks()[0].select_score == 1.0);   // KS is the ranking signal
    CHECK(s.blocks()[0].retrieval_score == 1.0);
    CHECK(s.blocks()[0].attn_score == 1.0);     // content signal, never KS
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 2);
    CHECK(s.blocks()[0].lrfu_score == 1.5);   // 0.5*1.0 + 1.0
    CHECK(s.blocks()[1].lrfu_score == 0.0);
}

static void test_lrfu_round_id_is_idempotent() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 7);
    CHECK(s.blocks()[0].lrfu_score == 1.0);
    // A repeat inside the same round must not decay or re-add; it only keeps
    // select_score (the ranking signal) equal to KS.
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 7);
    CHECK(s.blocks()[0].lrfu_score == 1.0);
    CHECK(s.blocks()[0].select_score == 1.0);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 8);
    CHECK(s.blocks()[0].lrfu_score == 1.5);
}

static void test_lrfu_top_frac_masks_low_scores() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true;
    cfg.lrfu_lambda = 0.0f;      // no carry-over: isolate the hit mask
    cfg.lrfu_top_frac = 0.5f;    // top 50% of 10 blocks = the 5 highest
    KvMemStore s(cfg);
    s.register_append(128 * 10);
    std::vector<double> scores(10);
    for (int i = 0; i < 10; ++i) scores[i] = double(i);
    s.set_retrieval_scores(scores);
    for (uint32_t i = 0; i < 10; ++i) {
        const double expect = i >= 5 ? double(i) : 0.0;
        CHECK(s.blocks()[i].lrfu_score == expect);
    }
}

static void test_lrfu_never_goes_negative_and_can_recover() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({-1.0, -2.0, 0.5, 0.0}, 1);
    for (uint32_t i = 0; i < 4; ++i) CHECK(s.blocks()[i].lrfu_score >= 0.0);
    CHECK(s.blocks()[2].lrfu_score == 0.5);
    // A block that scored badly is not sunk forever.
    s.set_retrieval_scores({2.0, 0.0, 0.0, 0.0}, 2);
    CHECK(s.blocks()[0].lrfu_score == 2.0);
}

// A caller with no round clock of its own (the ninfer backend scores out of
// prepare_window) passes round_id == 0. The store then derives the round from
// the block frontier, so re-scoring the SAME frontier must not decay the pool
// twice, while the next append starts a new round.
static void test_lrfu_derived_round_is_idempotent() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0});   // derived round = frontier 512
    CHECK(s.blocks()[0].lrfu_score == 1.0);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0});   // same frontier: a repeat
    CHECK(s.blocks()[0].lrfu_score == 1.0);
    CHECK(s.blocks()[0].select_score == 1.0);
    s.register_append(128);                          // frontier grew: new round
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0});
    CHECK(s.blocks()[0].lrfu_score == 1.5);
}

static void test_lrfu_reset_clears_keep_scores() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 1.0, 0.0, 0.0});
    s.reset_lrfu();
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(s.blocks()[i].lrfu_score == 0.0);
        CHECK(s.blocks()[i].select_score == 0.0);
        CHECK(s.blocks()[i].attn_score == 0.0);
    }
}

// A block with no positive evidence must not build a keep score across rounds.
// KVMem re-scores the whole resident pool every round, so if zero-score blocks
// counted as hits their KS would grow with page AGE and outrank the blocks the
// query actually matched -- and at large context the oldest blocks are the ones
// whose host records the tier has already dropped.
static void test_lrfu_zero_scores_do_not_accumulate() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 0.90f;
    KvMemStore s(cfg);
    s.register_append(128 * 8);
    std::vector<double> scores(8, 0.0);
    scores[7] = 1.0;                     // only block 7 has evidence
    s.set_retrieval_scores(scores, 1);
    s.set_retrieval_scores(scores, 2);
    s.set_retrieval_scores(scores, 3);
    CHECK(s.blocks()[7].lrfu_score == 1.75);   // 0.5*(0.5*1+1)+1
    for (uint32_t i = 0; i < 7; ++i) {
        CHECK(s.blocks()[i].lrfu_score == 0.0);
        CHECK(s.blocks()[i].select_score == 0.0);
        CHECK(s.blocks()[i].attn_score == 0.0);
    }
}

// A round where nothing matches must be a pure decay, not a rescaling of the
// whole pool.
static void test_lrfu_round_without_evidence_only_decays() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 0.90f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({0.0, 2.0, 0.0, 0.0}, 1);
    CHECK(s.blocks()[1].lrfu_score == 2.0);
    s.set_retrieval_scores({0.0, 0.0, 0.0, 0.0}, 2);
    CHECK(s.blocks()[1].lrfu_score == 1.0);    // decayed, nothing added
    CHECK(s.blocks()[0].lrfu_score == 0.0);
}

// Action mode must change WHICH blocks survive a tight budget: a block with a
// weaker current score but a strong recent history outranks a fresher one.
static void test_lrfu_changes_selection_order() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.select_budget = 128 * 2;   // sink(1) + one retrieved block
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 8);
    s.set_retrieval_scores({0, 0, 0, 0, 0, 1.0, 0.0, 0}, 1);   // block 5 strong
    s.set_retrieval_scores({0, 0, 0, 0, 0, 0.0, 0.4, 0}, 2);   // block 6 now best
    // KS: block5 = 0.5, block6 = 0.4 -> the history keeps block 5 selected.
    auto picked = s.pick_topk_blocks();
    CHECK(std::find(picked.begin(), picked.end(), 0u) != picked.end());
    CHECK(std::find(picked.begin(), picked.end(), 5u) != picked.end());
    CHECK(std::find(picked.begin(), picked.end(), 6u) == picked.end());

    // Same scores with ActKV off: the one-shot winner (block 6) is selected.
    KvMemStoreConfig one_shot = cfg; one_shot.action_lrfu = false;
    KvMemStore t(one_shot);
    t.register_append(128 * 8);
    t.set_retrieval_scores({0, 0, 0, 0, 0, 0.0, 0.4, 0});
    auto picked2 = t.pick_topk_blocks();
    CHECK(std::find(picked2.begin(), picked2.end(), 6u) != picked2.end());
    CHECK(std::find(picked2.begin(), picked2.end(), 5u) == picked2.end());
}

// T1 (docs/dev-actkv-swap-redesign.md): the ActKV keep score and the ranking
// signal are separate fields. The keep score is never written into attn_score,
// so the content signal constrain_media reads stays a function of the current
// query, not of how many rounds LRFU has rewarded a block.
static void test_lrfu_keeps_content_signal_out_of_the_ranking_field() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 1);
    s.set_retrieval_scores({1.0, 0.0, 0.0, 0.0}, 2);   // KS = 1.5, one-shot = 1.0
    CHECK(s.blocks()[0].lrfu_score == 1.5);            // persistent keep score
    CHECK(s.blocks()[0].select_score == 1.5);          // ranking follows KS
    CHECK(s.blocks()[0].retrieval_score == 1.0);
    CHECK(s.blocks()[0].attn_score == 1.0);            // content signal, not KS
}

// T1 acceptance: constrain_media ranks image groups by attn_score, so a block
// with a large keep score but no current query evidence must not win a
// media-constrained slot. With KS leaking into attn_score (the pre-T1 bug) the
// stale block 2 would be kept and the fresh block 10 dropped.
static void test_media_constraint_ignores_actkv_keep_score() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = cfg.prefill_budget = 6 * 32;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    cfg.action_lrfu = true;
    cfg.lrfu_lambda = 1.0f;      // no decay: KS diverges from the one-shot score
    cfg.lrfu_top_frac = 1.0f;
    KvMemStore store(cfg);
    store.register_append(12 * 32);
    // Round 1 rewards block 2; round 2 has evidence only for block 9.
    store.set_retrieval_scores({0, 0, 8.0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 1);
    store.set_retrieval_scores({0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0}, 2);
    CHECK(store.blocks()[2].lrfu_score == 8.0);
    CHECK(store.blocks()[2].attn_score == 0.0);   // no current evidence
    CHECK(store.blocks()[9].attn_score == 1.0);
    store.set_media_ranges({{6 * 32, 8 * 32}});   // image group = blocks 6..7
    const auto selected = store.pick_topk_blocks({});
    // sink(0) + latest image(6,7) + the three highest CONTENT blocks (9,10,11).
    CHECK((selected == std::vector<uint32_t>{0, 6, 7, 9, 10, 11}));
}

// Deferred LRFU (two-phase): scoring happens before the selection exists, so the
// round is held and applied at commit_lrfu. The selector must still rank by the
// PROVISIONAL keep score that includes this round, or a freshly written block
// (KS = 0) can never outrank the incumbents and is evicted immediately
// (docs/a9-deferred-lrfu-regression.md).
static void test_lrfu_deferred_ranking_sees_current_query() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.select_budget = 128 * 3;   // sink(1) + two retrieved
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    cfg.action_lrfu = true; cfg.deferred_lrfu = true;
    cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 8);
    s.set_retrieval_scores({0, 0, 0, 0, 0, 10.0, 0.0, 0}, 1);
    s.commit_lrfu({0, 5, 7});
    CHECK(s.blocks()[5].lrfu_score == 10.0);
    // Round 2: block 6 has the only evidence. Its committed KS is still 0, but
    // the provisional ranking must see this round's 8.0 and select it.
    s.set_retrieval_scores({0, 0, 0, 0, 0, 0.0, 8.0, 0}, 2);
    CHECK(s.blocks()[6].lrfu_score == 0.0);    // not committed yet
    CHECK(s.blocks()[6].select_score == 8.0);  // provisional ranking value
    const auto picked = s.pick_topk_blocks();
    CHECK(std::find(picked.begin(), picked.end(), 6u) != picked.end());
    s.commit_lrfu(picked);
    CHECK(s.blocks()[6].lrfu_score == 8.0);    // committed after the selection
}

// Deferred commit narrows growth to the blocks this round actually kept; a
// positive score alone is not enough.
static void test_lrfu_deferred_commit_narrows_hits() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.deferred_lrfu = true;
    cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 2.0, 0.0, 0.0}, 1);
    CHECK(s.blocks()[0].select_score == 1.0);   // provisional, pre-commit
    CHECK(s.blocks()[0].lrfu_score == 0.0);
    s.commit_lrfu({1});
    CHECK(s.blocks()[1].lrfu_score == 2.0);     // selected + positive -> grew
    CHECK(s.blocks()[0].lrfu_score == 0.0);     // positive but not selected -> no growth
}

// Deferred off keeps the score-only rule: no selection gate, every positive
// block grows immediately.
static void test_lrfu_deferred_off_keeps_score_only_rule() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.deferred_lrfu = false;
    cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({1.0, 2.0, 0.0, 0.0}, 1);
    CHECK(s.blocks()[0].lrfu_score == 1.0);
    CHECK(s.blocks()[1].lrfu_score == 2.0);
}

// ActKV Sec. 5.1 AAV: record_access turns each round's selection into one hit
// bit, and aav_class_of reads the ring back. Only the observed prefix of the
// 16-bit ring counts, so a block that appeared this round is not "silent".
static void test_aav_ring_classification() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.aav_trace = true;        // advances the ring without the trace env var
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.commit_lrfu({0});          // r1: 0 hit, 1..3 miss
    s.register_append(128);
    s.commit_lrfu({0, 1});       // r2: 0 hit, 1 hit
    s.register_append(128);
    s.commit_lrfu({0});          // r3: 0 hit, 1 miss
    s.register_append(128);
    s.commit_lrfu({0, 1});       // r4: 0 hit, 1 hit
    // Hit in every observed round.
    CHECK(s.aav_class_of(0) == KvAavClass::Continuous);
    // Hit twice, missed twice: repeated but discontinuous -- the paper's
    // "intermittent" access pattern.
    CHECK(s.aav_class_of(1) == KvAavClass::Intermittent);
    // Never hit, but old enough to have been observed.
    CHECK(s.aav_class_of(2) == KvAavClass::Silent);
    CHECK(s.aav_class_of(3) == KvAavClass::Silent);
    // Out of range is not a crash.
    CHECK(s.aav_class_of(9999) == KvAavClass::Cold);
}

// The classification is computed over the observed 16-round window, so a single
// missed round keeps a block intermittent only while that miss is still inside
// it. This is the boundary that decides how long one unlucky round keeps paying.
static void test_aav_intermittent_forgets_an_old_miss() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    cfg.action_lrfu = true; cfg.lrfu_lambda = 0.5f; cfg.lrfu_top_frac = 1.0f;
    cfg.aav_trace = true;
    KvMemStore s(cfg);
    s.register_append(128 * 4);
    s.set_retrieval_scores({0.0, 1.0, 0.0, 0.0}, 1);
    s.commit_lrfu({1});                       // r1 hit
    s.register_append(128);
    s.set_retrieval_scores({}, 2);
    s.commit_lrfu({0});                       // r2 miss
    s.register_append(128);
    s.set_retrieval_scores({0.0, 1.0, 0.0, 0.0}, 3);
    s.commit_lrfu({1});                       // r3 hit
    CHECK(s.aav_class_of(1) == KvAavClass::Intermittent);
    // Hit from r4 through r18. At r17 the window still holds the r2 miss; by r18
    // the last 16 rounds are all hits, so the class clears.
    for (uint32_t round = 4; round <= 18; ++round) {
        s.register_append(128);
        s.set_retrieval_scores({0.0, 1.0, 0.0, 0.0}, round);
        s.commit_lrfu({1});
        if (round == 17) { CHECK(s.aav_class_of(1) == KvAavClass::Intermittent); }
        if (round == 18) { CHECK(s.aav_class_of(1) == KvAavClass::Continuous); }
    }
}

int main() {
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
    test_lrfu_defaults_match_actkv();
    test_lrfu_off_keeps_one_shot_ranking();
    test_lrfu_accumulates_and_decays();
    test_lrfu_round_id_is_idempotent();
    test_lrfu_derived_round_is_idempotent();
    test_lrfu_top_frac_masks_low_scores();
    test_lrfu_never_goes_negative_and_can_recover();
    test_lrfu_zero_scores_do_not_accumulate();
    test_lrfu_round_without_evidence_only_decays();
    test_lrfu_reset_clears_keep_scores();
    test_lrfu_changes_selection_order();
    test_lrfu_keeps_content_signal_out_of_the_ranking_field();
    test_media_constraint_ignores_actkv_keep_score();
    test_lrfu_deferred_ranking_sees_current_query();
    test_lrfu_deferred_commit_narrows_hits();
    test_lrfu_deferred_off_keeps_score_only_rule();
    test_aav_ring_classification();
    test_aav_intermittent_forgets_an_old_miss();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
