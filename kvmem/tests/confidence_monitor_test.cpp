#include "kvmem/confidence_monitor.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using kvmem::KvMemConfidenceConfig;
using kvmem::KvMemConfidenceMonitor;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static KvMemConfidenceConfig test_cfg() {
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 64;
    cfg.stride = 1;          // pool after every token: one R' entry per token
    cfg.bottom_frac = 0.5f;
    cfg.min_pool = 8;
    return cfg;
}

static void test_decreasing_tail_grows_budget() {
    KvMemConfidenceMonitor m(test_cfg());
    CHECK(!m.ready());
    CHECK(m.slope() == 0.0);
    CHECK(!m.should_increase_budget());
    // Confidence falls steadily: the model is losing certainty, so the budget
    // should grow once enough pooled minima exist.
    for (int i = 0; i < 16; ++i) m.observe(static_cast<double>(10 - i));
    CHECK(m.ready());
    CHECK(m.slope() < 0.0);
    CHECK(m.should_increase_budget());
}

static void test_increasing_tail_keeps_budget() {
    // The pooled signal is a window MINIMUM, so it only rises once the window
    // actually slides past the low tokens. Feed whole groups: group g is
    // uniformly (100 + 10g), so R'_g = 100 + 10g.
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 4;
    cfg.stride = 4;
    cfg.bottom_frac = 1.0f;
    cfg.min_pool = 4;
    KvMemConfidenceMonitor m(cfg);
    for (int g = 0; g < 8; ++g) {
        for (int k = 0; k < 4; ++k) m.observe(100.0 + 10.0 * g);
    }
    CHECK(m.pooled() == 8);
    CHECK(m.slope() > 0.0);
    CHECK(!m.should_increase_budget());
}

// While the window is still filling, the running minimum cannot rise: a rising
// token sequence inside one window still pools to its lowest token. Pin this so
// a future "fix" does not quietly switch the aggregation back to a mean.
static void test_min_pooling_is_flat_until_window_slides() {
    KvMemConfidenceMonitor m(test_cfg());  // stride 1, pool_size 64
    for (int i = 0; i < 16; ++i) m.observe(static_cast<double>(i));
    CHECK(m.slope() == 0.0);
}

static void test_flat_tail_is_stable() {
    KvMemConfidenceMonitor m(test_cfg());
    for (int i = 0; i < 16; ++i) m.observe(0.0);
    CHECK(m.slope() == 0.0);
    CHECK(!m.should_increase_budget());
}

static void test_ring_wraps_and_stays_finite() {
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 8;
    cfg.stride = 1;
    cfg.bottom_frac = 0.5f;
    cfg.min_pool = 4;
    KvMemConfidenceMonitor m(cfg);
    // Far more observations than the window; the ring must keep dropping the
    // oldest tokens and still report a finite, negative slope on a decay.
    for (int i = 0; i < 200; ++i) m.observe(0.0 - i);
    CHECK(m.size() == cfg.pool_size);
    CHECK(m.slope() < 0.0);
    CHECK(m.slope() == m.slope());  // a real number, not NaN
    CHECK(m.should_increase_budget());
}

// ActKV Sec. 6.1: the aggregate is the window MINIMUM, not the mean. A window
// whose other tokens keep getting more confident must not raise the trend
// while a constant low-confidence token is still inside it.
static void test_min_pooling_not_mean() {
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 4;
    cfg.stride = 4;          // one pooled minimum per group of 4 tokens
    cfg.bottom_frac = 1.0f;  // keep all pooled minima
    cfg.min_pool = 4;
    KvMemConfidenceMonitor m(cfg);
    for (int g = 0; g < 8; ++g) {
        m.observe(5.0);                              // constant low token
        for (int k = 0; k < 3; ++k) m.observe(100.0 + 10.0 * g);
    }
    CHECK(m.pooled() == 8);
    // Min is constant at 5.0 -> flat trend. A mean-based pool would rise with
    // the 100+10g tokens and report a positive slope.
    CHECK(m.slope() == 0.0);
    CHECK(!m.should_increase_budget());
}

// Stride s > 1 must emit one pooled point per stride, not per token.
static void test_stride_controls_pooling_rate() {
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 16;
    cfg.stride = 8;
    cfg.bottom_frac = 1.0f;
    cfg.min_pool = 2;
    KvMemConfidenceMonitor m(cfg);
    for (int i = 0; i < 80; ++i) m.observe(static_cast<double>(i));
    CHECK(m.pooled() == 10);
    CHECK(m.ready());
}

// Bottom-b% keeps only the worst segments, so a rising trace with a few
// improving-but-still-worst segments is judged on those segments alone.
static void test_bottom_frac_selects_worst_segments() {
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 4;
    cfg.stride = 4;
    cfg.bottom_frac = 0.25f;
    cfg.min_pool = 4;
    KvMemConfidenceMonitor m(cfg);
    // 8 pooled minima: six high (100) and two low (1 then 2) at the end.
    for (int g = 0; g < 6; ++g) {
        for (int k = 0; k < 4; ++k) m.observe(100.0);
    }
    for (int k = 0; k < 4; ++k) m.observe(1.0);
    for (int k = 0; k < 4; ++k) m.observe(2.0);
    CHECK(m.pooled() == 8);
    // bottom 25% = 2 points = the two low segments, in arrival order: 1 then 2
    CHECK(m.slope() > 0.0);
    CHECK(!m.should_increase_budget());
}

static void test_reset_clears_state() {
    KvMemConfidenceMonitor m(test_cfg());
    for (int i = 0; i < 16; ++i) m.observe(static_cast<double>(10 - i));
    CHECK(m.ready());
    m.reset();
    CHECK(!m.ready());
    CHECK(m.size() == 0);
    CHECK(m.pooled() == 0);
    CHECK(m.slope() == 0.0);
}

// The pooled ring must stay bounded no matter how long the trace runs.
static void test_pooled_ring_is_bounded() {
    KvMemConfidenceConfig cfg;
    cfg.pool_size = 8;
    cfg.stride = 1;
    cfg.bottom_frac = 0.5f;
    cfg.min_pool = 4;
    KvMemConfidenceMonitor m(cfg);
    for (int i = 0; i < 5000; ++i) m.observe(static_cast<double>(i % 37));
    CHECK(m.pooled() <= 512);
    CHECK(m.size() == 8);
}

// --- token_confidence(): T_i = -(1/k) * sum_{j<=k} log P_i(j) --------------

// A one-hot distribution puts all mass on the argmax. With k = 1 the sum is a
// single term, -log P(argmax) = -log 1 = 0, i.e. the confident end of the range.
// (With k > 1 the near-zero tail dominates the average and T is large -- the
// mean is over the top-k, not just the winner.)
static void test_token_confidence_one_hot_is_zero() {
    const float logits[4] = {0.0f, -50.0f, -50.0f, -50.0f};
    const float t = kvmem::token_confidence(logits, 4, 1);
    CHECK(t >= 0.0f);
    CHECK(t < 1e-3f);
}

// A uniform distribution over n tokens has P = 1/n everywhere, so T = log n.
// This is the flat end of the range and the case the monitor must catch.
static void test_token_confidence_uniform_is_log_n() {
    const int n = 64;
    std::vector<float> logits(static_cast<size_t>(n), 0.0f);
    const float t = kvmem::token_confidence(logits.data(), n, 20);
    CHECK(std::fabs(t - static_cast<float>(std::log(static_cast<double>(n)))) < 1e-3f);
}

// Peakedness must move T in the right direction: sharpening raises it.
static void test_token_confidence_rises_with_peakedness() {
    const float flat[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float sharp[4] = {6.0f, 0.0f, 0.0f, 0.0f};
    const float t_flat = kvmem::token_confidence(flat, 4, 4);
    const float t_sharp = kvmem::token_confidence(sharp, 4, 4);
    CHECK(t_sharp > t_flat);
}

// k caps how many log-probabilities are averaged. With k=1 only the argmax is
// used, which is strictly more confident than averaging in the tail.
static void test_token_confidence_respects_k() {
    const float logits[4] = {1.0f, -10.0f, -10.0f, -10.0f};
    const float t1 = kvmem::token_confidence(logits, 4, 1);
    const float t4 = kvmem::token_confidence(logits, 4, 4);
    CHECK(t1 < t4);  // more tokens averaged in -> more negative log P -> larger T
    CHECK(kvmem::token_confidence(logits, 4, 99) == t4);  // k > n clamps to n
}

// Shift invariance: softmax is invariant to a constant offset, so T must be too.
// This is what the max-subtraction in pass 2 is for.
static void test_token_confidence_is_shift_invariant() {
    const float a[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float b[4] = {1001.0f, 1000.0f, 1000.0f, 1000.0f};
    const float ta = kvmem::token_confidence(a, 4, 4);
    const float tb = kvmem::token_confidence(b, 4, 4);
    CHECK(std::fabs(ta - tb) < 1e-3f);
}

// Unusable input must be NaN, not a plausible-looking number: a NaN fed to the
// monitor poisons every pooled minimum containing it and silently disables
// budget growth for the rest of the trace.
static void test_token_confidence_rejects_bad_input() {
    const float ok[2] = {0.0f, 0.0f};
    CHECK(std::isnan(kvmem::token_confidence(nullptr, 4, 4)));
    CHECK(std::isnan(kvmem::token_confidence(ok, 0, 4)));
    CHECK(std::isnan(kvmem::token_confidence(ok, 2, 0)));
    const float all_nan[2] = {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN()};
    CHECK(std::isnan(kvmem::token_confidence(all_nan, 2, 2)));
}

// Grammar masking writes -inf into the candidate row. Those tokens are excluded
// from the distribution, so the result must equal the unmasked one, not NaN.
static void test_token_confidence_skips_masked_tokens() {
    const std::vector<float> masked = {1.0f, 0.0f, -std::numeric_limits<float>::infinity()};
    const std::vector<float> plain = {1.0f, 0.0f};
    const float tm = kvmem::token_confidence(masked.data(), 3, 2);
    const float tp = kvmem::token_confidence(plain.data(), 2, 2);
    CHECK(std::isfinite(tm));
    CHECK(std::fabs(tm - tp) < 1e-3f);
}

// --- KvMemConfidenceBudget (Sec. 6.2, round-gated) -------------------------

using kvmem::KvMemConfidenceBudget;
using kvmem::KvMemConfidenceBudgetConfig;

// The budget must be inert until the confidence trend is actually negative.
static void test_budget_grows_only_on_negative_slope() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4096;
    cfg.upper_tokens = 36864;
    cfg.scale_up = 1.75f;
    cfg.cooldown_rounds = 1;
    KvMemConfidenceConfig mon;
    mon.pool_size = 8;
    mon.stride = 1;
    mon.bottom_frac = 1.0f;
    mon.min_pool = 4;
    KvMemConfidenceBudget b(cfg, mon);
    CHECK(b.tokens() == 4096);
    // Confidence RISES (a good allocation): every round keeps the budget.
    for (int r = 0; r < 20; ++r) {
        for (int i = 0; i < 8; ++i) b.observe(10.0 + r);
        CHECK(b.advance_round() == 4096);
    }
    CHECK(b.growths() == 0);
}

// The regression this class exists for: a persistent negative slope must NOT grow
// the budget on every round. Upstream, per-token scaling reached the ceiling in a
// handful of tokens.
static void test_budget_is_rate_limited_per_round() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4096;
    cfg.upper_tokens = 36864;
    cfg.scale_up = 1.75f;
    cfg.cooldown_rounds = 4;
    KvMemConfidenceConfig mon;
    mon.pool_size = 8;
    mon.stride = 1;
    mon.bottom_frac = 1.0f;
    mon.min_pool = 4;
    KvMemConfidenceBudget b(cfg, mon);
    // Confidence decays hard and forever; the slope stays negative throughout.
    for (int r = 0; r < 40; ++r) {
        for (int i = 0; i < 8; ++i) b.observe(100.0 - 20.0 * r - i);
        b.advance_round();
    }
    // One growth per cooldown window bounds the total number of growths. An
    // ungated policy would grow every round (4 steps of 1.75x reach the ceiling
    // from 4096), so 40 rounds cannot produce more than ~10 growths here.
    CHECK(b.growths() <= 40 / cfg.cooldown_rounds + 1);

    // The gate must actually bind: within one cooldown window a persistently
    // negative slope produces exactly one growth, not one per round.
    KvMemConfidenceBudget c(cfg, mon);
    for (int i = 0; i < 8; ++i) c.observe(100.0 - i);
    c.advance_round();                 // first growth
    const uint32_t after_first = c.tokens();
    for (int r = 0; r < cfg.cooldown_rounds - 1; ++r) {
        for (int i = 0; i < 8; ++i) c.observe(-100.0);   // slope stays negative
        CHECK(c.advance_round() == after_first);          // held by the cooldown
    }
    CHECK(c.growths() == 1);
    // An ungated policy would already have saturated the ceiling here.
    CHECK(after_first < cfg.upper_tokens);
}

static void test_budget_never_exceeds_upper() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4096;
    cfg.upper_tokens = 8192;
    cfg.scale_up = 10.0f;   // a single step would overshoot
    cfg.cooldown_rounds = 1;
    KvMemConfidenceConfig mon;
    mon.pool_size = 8;
    mon.stride = 1;
    mon.bottom_frac = 1.0f;
    mon.min_pool = 4;
    KvMemConfidenceBudget b(cfg, mon);
    for (int r = 0; r < 50; ++r) {
        for (int i = 0; i < 8; ++i) b.observe(100.0 - 20.0 * r - i);
        const uint32_t t = b.advance_round();
        CHECK(t <= cfg.upper_tokens);
    }
    CHECK(b.tokens() == 8192);
}

// Sec. 6.2: recovery retains the current budget -- it never shrinks.
static void test_budget_never_shrinks() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4096;
    cfg.upper_tokens = 36864;
    cfg.cooldown_rounds = 1;
    KvMemConfidenceConfig mon;
    mon.pool_size = 8;
    mon.stride = 1;
    mon.bottom_frac = 1.0f;
    mon.min_pool = 4;
    KvMemConfidenceBudget b(cfg, mon);
    for (int i = 0; i < 8; ++i) b.observe(100.0 - i);
    const uint32_t grew = b.advance_round();
    CHECK(grew > 4096);
    uint32_t previous = grew;
    // Confidence recovers strongly; the budget must hold, not fall back.
    for (int r = 0; r < 20; ++r) {
        for (int i = 0; i < 8; ++i) b.observe(500.0 + r);
        const uint32_t t = b.advance_round();
        CHECK(t >= previous);
        previous = t;
    }
    CHECK(b.tokens() >= grew);
}

static void test_budget_disabled_when_start_zero() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 0;
    cfg.upper_tokens = 36864;
    KvMemConfidenceBudget b(cfg);
    for (int r = 0; r < 10; ++r) {
        for (int i = 0; i < 8; ++i) b.observe(-100.0);
        CHECK(b.advance_round() == 0);
    }
    CHECK(b.growths() == 0);
}

static void test_budget_rounds_to_granularity() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4097;   // not a multiple of 64
    cfg.upper_tokens = 36864;
    cfg.round_tokens = 64;
    KvMemConfidenceBudget b(cfg);
    CHECK(b.tokens() == 4160);  // rounded up to a whole page
}

// A NaN would poison the pooled minimum and make the slope NaN, which compares
// false against < 0 and silently disables growth for the rest of the trace.
static void test_budget_ignores_nan() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4096;
    cfg.upper_tokens = 36864;
    cfg.cooldown_rounds = 1;
    KvMemConfidenceConfig mon;
    mon.pool_size = 8;
    mon.stride = 1;
    mon.bottom_frac = 1.0f;
    mon.min_pool = 4;
    KvMemConfidenceBudget b(cfg, mon);
    for (int i = 0; i < 8; ++i) b.observe(100.0 - i);
    b.observe(std::numeric_limits<double>::quiet_NaN());
    const uint32_t t = b.advance_round();
    CHECK(t > 4096);  // the good samples still drove a growth
}

// --- token_confidence_from_logprobs() -------------------------------------

// The ninfer gather emits top_values = s[v] - lse = log P(v). This entry point
// must agree with the raw-logit path on the same distribution.
static void test_logprobs_match_token_confidence() {
    const float logits[4] = {1.0f, 0.5f, 0.0f, -1.0f};
    const float t_logits = kvmem::token_confidence(logits, 4, 3);
    // log P for the top 3, computed independently.
    const double mx = 1.0;
    double z = 0.0;
    for (int i = 0; i < 4; ++i) z += std::exp(static_cast<double>(logits[i]) - mx);
    const double lse = mx + std::log(z);
    const float logp[3] = {
        static_cast<float>(1.0 - lse), static_cast<float>(0.5 - lse),
        static_cast<float>(0.0 - lse)};
    const float t_logp = kvmem::token_confidence_from_logprobs(logp, 3);
    CHECK(std::fabs(t_logits - t_logp) < 1e-3f);
}

static void test_logprobs_rejects_bad_input() {
    CHECK(std::isnan(kvmem::token_confidence_from_logprobs(nullptr, 4)));
    CHECK(std::isnan(kvmem::token_confidence_from_logprobs(nullptr, 0)));
    // The gather fills unused slots with -inf when fewer than k tokens remain.
    const float mostly_unused[4] = {-0.5f, -std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
    const float t = kvmem::token_confidence_from_logprobs(mostly_unused, 4);
    CHECK(std::isfinite(t));
    CHECK(std::fabs(t - 0.5f) < 1e-4f);  // only the one finite slot is averaged
    const float all_inf[2] = {-std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};
    CHECK(std::isnan(kvmem::token_confidence_from_logprobs(all_inf, 2)));
}

// A re-entered boundary (async prepare publishing and retrying, or a
// preview/apply pair) must not consume two rounds or grow twice from the same
// evidence.
static void test_budget_round_key_is_idempotent() {
    KvMemConfidenceBudgetConfig cfg;
    cfg.start_tokens = 4096;
    cfg.upper_tokens = 36864;
    cfg.cooldown_rounds = 1;
    KvMemConfidenceConfig mon;
    mon.pool_size = 8;
    mon.stride = 1;
    mon.bottom_frac = 1.0f;
    mon.min_pool = 4;
    KvMemConfidenceBudget b(cfg, mon);
    for (int i = 0; i < 8; ++i) b.observe(100.0 - i);
    const uint32_t first = b.advance_round(4096);
    CHECK(first > 4096);
    CHECK(b.rounds() == 1);
    CHECK(b.growths() == 1);
    // Same key: no new round, no new growth.
    for (int i = 0; i < 3; ++i) {
        CHECK(b.advance_round(4096) == first);
        CHECK(b.rounds() == 1);
        CHECK(b.growths() == 1);
    }
    // A new key is a new round.
    b.advance_round(8192);
    CHECK(b.rounds() == 2);
}

int main() {
    test_token_confidence_one_hot_is_zero();
    test_token_confidence_uniform_is_log_n();
    test_token_confidence_rises_with_peakedness();
    test_token_confidence_respects_k();
    test_token_confidence_is_shift_invariant();
    test_token_confidence_rejects_bad_input();
    test_token_confidence_skips_masked_tokens();
    test_logprobs_match_token_confidence();
    test_logprobs_rejects_bad_input();
    test_budget_grows_only_on_negative_slope();
    test_budget_is_rate_limited_per_round();
    test_budget_never_exceeds_upper();
    test_budget_never_shrinks();
    test_budget_disabled_when_start_zero();
    test_budget_rounds_to_granularity();
    test_budget_ignores_nan();
    test_budget_round_key_is_idempotent();
    test_decreasing_tail_grows_budget();
    test_increasing_tail_keeps_budget();
    test_min_pooling_is_flat_until_window_slides();
    test_flat_tail_is_stable();
    test_ring_wraps_and_stays_finite();
    test_min_pooling_not_mean();
    test_stride_controls_pooling_rate();
    test_bottom_frac_selects_worst_segments();
    test_reset_clears_state();
    test_pooled_ring_is_bounded();
    if (g_fail == 0) {
        std::printf("OK\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", g_fail);
    return 1;
}
