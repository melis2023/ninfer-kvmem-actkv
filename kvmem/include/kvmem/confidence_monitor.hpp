#pragma once
#include <cstddef>
#include <vector>

namespace kvmem {

// Default top-k for T_i (ActKV Sec. 6.1 / Sec. 8: k = 20).
constexpr int kDefaultConfTopK = 20;

// ActKV token-level confidence (arXiv:2609.31395, Sec. 6.1):
//   T_i = -(1/k) * sum_{j=1..k} log P_i(j)
// with P_i the softmax of `logits` and the sum over the k LARGEST logits.
// T is large for a peaked distribution (every log P close to 0) and small for
// a flat one, so HIGHER T = MORE confident.
//
// `logits` must be the RAW model logits. Temperature, repeat penalties and
// grammar masking all reshape the distribution and would turn the signal into
// a function of the sampling configuration instead of the model -- temperature
// < 1 in particular inflates T and hides exactly the degradation this is
// meant to detect.
//
// Returns a quiet NaN when the input is unusable (null, n <= 0, k <= 0, or a
// degenerate distribution). Callers must drop NaN instead of feeding it to
// KvMemConfidenceMonitor: one NaN poisons every pooled minimum that contains
// it and the slope becomes NaN, which compares false against < 0 and silently
// disables budget growth.
float token_confidence(const float * logits, int n, int k = kDefaultConfTopK);

// ActKV Sec. 6.1 token confidence from an ALREADY-NORMALIZED top-k log-probability
// vector, i.e. values that are log P(j) rather than raw logits. The ninfer engine's
// logprob_topk Op emits exactly that (top_values = s[v] - lse), so this entry point
// lets a backend reuse the gather it already performs instead of re-reducing the
// vocabulary. Same math as token_confidence's second half:
//   T_i = -(1/k) * sum_j log P(j)
// Non-finite entries are skipped (the gather writes -inf into the unused slots when
// fewer than k tokens remain). Returns quiet NaN when no finite entry exists.
float token_confidence_from_logprobs(const float * log_probs, int n);

// ActKV confidence-driven adaptive budget (arXiv:2609.31395, Sec. 6.1 Alg. 2).
//
// Token-level confidence is derived from the predicted distribution:
//   T_i = -(1/k) * sum_{j=1..k} log P_i(j)      (k = 20 in the paper)
// i.e. the negative mean log-probability of the top-k tokens. A peaked
// distribution makes every log P close to 0, so T is LARGE for a confident
// token; a flat distribution makes them strongly negative, so T is SMALL.
// Higher T = more confident. (The caller computes T; this class consumes it.)
//
// Raw T is noisy, so it is aggregated in three steps:
//   1. R'_m = min over a sliding window of w tokens, stride s  (min pooling)
//   2. R    = the bottom b% of R'                              (worst segments)
//   3. alpha = first-order least-squares slope of R            (trend)
// Minimum pooling and bottom-b% selection jointly emphasise localised
// degradation -- the model restating missing context ("wait", "I guess",
// "let me check") when the compressed KV cache lost action-critical entries.
//
// Sec. 6.2: the budget is raised when alpha < 0 and NEVER lowered on recovery.
// "When confidence stabilizes or recovers, we retain the current budget, as
// this improvement suggests that the allocation is sufficient but does not
// establish that a smaller budget would suffice." Shrinking re-introduces the
// information loss that caused the confidence drop in the first place.
struct KvMemConfidenceConfig {
    uint32_t pool_size = 64;      // w: sliding window over token confidences
    uint32_t stride = 8;          // s: window stride between pooled minima
    float    bottom_frac = 0.10f; // b: bottom fraction of R' kept as R
    uint32_t min_pool = 8;        // min pooled points before a slope is trusted
};

class KvMemConfidenceMonitor {
public:
    explicit KvMemConfidenceMonitor(KvMemConfidenceConfig cfg = {});

    // Feed one token-level confidence T_i (higher = more confident).
    void observe(double token_conf);
    void reset();

    // Token confidences currently buffered (capped at cfg.pool_size).
    uint32_t size() const;
    // Pooled minima available for the fit (R').
    uint32_t pooled() const;
    uint32_t tokens() const;
    bool ready() const;

    // Least-squares slope alpha of R ordered by arrival time. Only valid when
    // ready(); returns 0.0 otherwise. alpha < 0 means confidence is decaying.
    double slope() const;
    // True when the confidence trend is negative: the KV budget is too small.
    bool should_increase_budget() const;

private:
    void pool_one();
    double min_window() const;

    KvMemConfidenceConfig cfg_;
    std::vector<double> tokens_;   // ring of the last w token confidences
    std::vector<double> pooled_;   // ring of the last N pooled minima (R')
    size_t token_head_ = 0;        // write position in tokens_ once full
    size_t pooled_head_ = 0;       // write position in pooled_ once full
    uint64_t n_tokens_ = 0;        // total token confidences observed
    uint64_t n_pooled_ = 0;        // total pooled minima produced
};

// ActKV Sec. 6.2 confidence-driven adaptive budget, with the round gating the paper
// implies but does not spell out.
//
// The monitor's should_increase_budget() is a PERSISTENT property of the pooled ring,
// not a one-shot event: while the ring still holds a negative slope, every query
// answers "yes". Scaling on every call is the failure mode the upstream ActKV tree
// had -- fed per token, 512 -> 896 -> 1568 ... reaches the ceiling within a handful of
// tokens, so the budget is effectively binary and the memory saving disappears.
//
// This class makes growth an explicit, rate-limited event:
//   * growth is only considered at a compression boundary (advance_round), never per
//     token, so the budget can change at most once per ORA round;
//   * a cooldown of `cooldown_rounds` rounds must elapse between two growths;
//   * the monitor is reset after a growth, so the stale negative slope that triggered
//     it cannot immediately trigger the next one;
//   * the budget never shrinks (Sec. 6.2: a recovery shows the current allocation is
//     sufficient but does not establish that a smaller one would be).
//
// `upper_tokens` is the hard ceiling. The device KV pool is sized from B + R at
// startup, so keeping upper_tokens <= the configured select budget is what lets this
// policy run without touching the physical layout.
struct KvMemConfidenceBudgetConfig {
    uint32_t start_tokens = 0;      // B at session start; 0 disables the policy
    uint32_t upper_tokens = 0;      // B_upper; growth stops here (0 => no growth)
    float    scale_up = 1.75f;      // the paper's Sec. 8.1 value
    uint32_t cooldown_rounds = 4;   // rounds that must elapse between two growths
    uint32_t round_tokens = 64;     // granularity the result is rounded to
};

class KvMemConfidenceBudget {
public:
    KvMemConfidenceBudget(KvMemConfidenceBudgetConfig cfg = {},
                          KvMemConfidenceConfig monitor_cfg = {});

    // Current budget in tokens. Never decreases; never exceeds upper_tokens.
    uint32_t tokens() const { return tokens_; }
    // How many times the budget has been raised. Diagnostics/tests.
    uint32_t growths() const { return growths_; }
    uint32_t rounds() const { return rounds_; }

    // Feed one token-level confidence T_i (higher = more confident).
    void observe(double token_conf);

    // End of a compression round: the only point at which growth is considered.
    // Returns the budget to use for the coming round.
    //
    // `round_key` identifies the round (the caller passes the logical frontier).
    // A repeat of the same key is a no-op that returns the current budget, so a
    // reselect path that re-enters its own boundary -- an async prepare that
    // publishes and retries, or a preview/apply pair -- cannot consume two rounds
    // or grow twice from the same evidence. This mirrors the LRFU round stamp.
    uint32_t advance_round(uint64_t round_key = 0);

private:
    uint32_t round_up(uint32_t tokens) const;

    KvMemConfidenceBudgetConfig cfg_;
    KvMemConfidenceMonitor monitor_;
    uint32_t tokens_ = 0;
    uint32_t growths_ = 0;
    uint32_t rounds_ = 0;
    uint32_t last_growth_round_ = 0;
    uint64_t last_round_key_ = 0;
    bool     round_key_seen_ = false;
    bool     grew_ = false;
};

} // namespace kvmem
