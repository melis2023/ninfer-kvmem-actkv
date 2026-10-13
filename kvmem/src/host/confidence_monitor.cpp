#include "kvmem/confidence_monitor.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace kvmem {

namespace {
constexpr size_t kPooledCap = 512;  // R' ring: ~4k tokens of trace history
}

float token_confidence(const float * logits, int n, int k) {
    if (logits == nullptr || n <= 0 || k <= 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int kk = k < n ? k : n;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Pass 1: max logit (numerical stability for the softmax) plus the k
    // largest logits, held descending. The common case rejects against
    // top[kk-1] on the first compare, so this stays close to one linear scan
    // and never copies the vocabulary.
    float mx = neg_inf;
    std::vector<float> top(static_cast<size_t>(kk), neg_inf);
    int ntop = 0;
    for (int i = 0; i < n; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            // -inf is a masked token; NaN would poison the whole reduction.
            continue;
        }
        if (v > mx) mx = v;
        if (ntop < kk || v > top[static_cast<size_t>(kk) - 1]) {
            int j = ntop < kk ? ntop : kk - 1;
            while (j > 0 && top[static_cast<size_t>(j) - 1] < v) {
                top[static_cast<size_t>(j)] = top[static_cast<size_t>(j) - 1];
                --j;
            }
            top[static_cast<size_t>(j)] = v;
            if (ntop < kk) ++ntop;
        }
    }
    if (ntop == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // Pass 2: log Z = max + log sum exp(logit - max), then T.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) continue;
        sum += std::exp(static_cast<double>(v) - static_cast<double>(mx));
    }
    if (!(sum > 0.0)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const double logz = static_cast<double>(mx) + std::log(sum);

    double acc = 0.0;
    for (int j = 0; j < ntop; ++j) {
        acc += static_cast<double>(top[static_cast<size_t>(j)]) - logz;
    }
    // log P <= 0, so -mean(log P) >= 0: T is non-negative by construction.
    return static_cast<float>(-acc / static_cast<double>(ntop));
}

float token_confidence_from_logprobs(const float * log_probs, int n) {
    if (log_probs == nullptr || n <= 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    double acc = 0.0;
    int used = 0;
    for (int j = 0; j < n; ++j) {
        const float v = log_probs[j];
        // The gather fills unused slots with -inf; a NaN would poison the mean.
        if (!std::isfinite(v)) continue;
        acc += static_cast<double>(v);
        ++used;
    }
    if (used == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(-acc / static_cast<double>(used));
}

KvMemConfidenceMonitor::KvMemConfidenceMonitor(KvMemConfidenceConfig cfg)
    : cfg_(cfg) {
    if (cfg_.pool_size == 0) cfg_.pool_size = 1;
    if (cfg_.stride == 0) cfg_.stride = 1;
    if (cfg_.bottom_frac <= 0.0f) cfg_.bottom_frac = 1.0f;
    if (cfg_.bottom_frac > 1.0f) cfg_.bottom_frac = 1.0f;
    if (cfg_.min_pool < 2) cfg_.min_pool = 2;
    tokens_.reserve(cfg_.pool_size);
    pooled_.reserve(kPooledCap);
}

void KvMemConfidenceMonitor::reset() {
    tokens_.clear();
    pooled_.clear();
    token_head_ = 0;
    pooled_head_ = 0;
    n_tokens_ = 0;
    n_pooled_ = 0;
}

void KvMemConfidenceMonitor::observe(double token_conf) {
    if (tokens_.size() < cfg_.pool_size) {
        tokens_.push_back(token_conf);
        if (tokens_.size() == cfg_.pool_size) token_head_ = 0;
    } else {
        tokens_[token_head_] = token_conf;
        token_head_ = (token_head_ + 1) % cfg_.pool_size;
    }
    ++n_tokens_;
    if (n_tokens_ % cfg_.stride == 0) {
        pool_one();
    }
}

void KvMemConfidenceMonitor::pool_one() {
    const double m = min_window();
    if (pooled_.size() < kPooledCap) {
        pooled_.push_back(m);
        if (pooled_.size() == kPooledCap) pooled_head_ = 0;
    } else {
        pooled_[pooled_head_] = m;
        pooled_head_ = (pooled_head_ + 1) % kPooledCap;
    }
    ++n_pooled_;
}

double KvMemConfidenceMonitor::min_window() const {
    if (tokens_.empty()) return 0.0;
    return *std::min_element(tokens_.begin(), tokens_.end());
}

uint32_t KvMemConfidenceMonitor::size() const {
    return static_cast<uint32_t>(tokens_.size());
}

uint32_t KvMemConfidenceMonitor::pooled() const {
    return static_cast<uint32_t>(pooled_.size());
}

uint32_t KvMemConfidenceMonitor::tokens() const {
    return static_cast<uint32_t>(std::min<uint64_t>(n_tokens_, 0xFFFFFFFFu));
}

bool KvMemConfidenceMonitor::ready() const {
    return pooled_.size() >= cfg_.min_pool;
}

double KvMemConfidenceMonitor::slope() const {
    if (!ready()) return 0.0;
    const size_t n = pooled_.size();
    size_t k = static_cast<size_t>(std::lround(cfg_.bottom_frac * static_cast<double>(n)));
    if (k > n) k = n;
    if (k < 2) return 0.0;  // need at least two points to fit a slope
    // Arrival-time offset of each ring slot (oldest -> newest), ring-aware.
    const auto t = [n, head = pooled_head_](size_t i) -> int {
        return static_cast<int>((i - head + n) % n);
    };
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    // Bottom-b% by VALUE (the least confident segments)...
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](size_t a, size_t b) { return pooled_[a] < pooled_[b]; });
    // ...then ordered by ARRIVAL so the fit is a time trend.
    std::sort(idx.begin(), idx.begin() + k,
              [&](size_t a, size_t b) { return t(a) < t(b); });
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < k; ++i) {
        const double x = static_cast<double>(t(idx[i]));
        const double y = pooled_[idx[i]];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double denom = static_cast<double>(k) * sxx - sx * sx;
    if (denom == 0.0) return 0.0;
    return (static_cast<double>(k) * sxy - sx * sy) / denom;
}

bool KvMemConfidenceMonitor::should_increase_budget() const {
    return ready() && slope() < 0.0;
}

KvMemConfidenceBudget::KvMemConfidenceBudget(KvMemConfidenceBudgetConfig cfg,
                                             KvMemConfidenceConfig monitor_cfg)
    : cfg_(cfg), monitor_(monitor_cfg) {
    if (cfg_.round_tokens == 0) cfg_.round_tokens = 1;
    if (cfg_.upper_tokens < cfg_.start_tokens) cfg_.upper_tokens = cfg_.start_tokens;
    if (!(cfg_.scale_up > 1.0f)) cfg_.scale_up = 1.0f;
    // Round the start up so the very first round is already a legal page multiple.
    tokens_ = round_up(cfg_.start_tokens);
    if (tokens_ > cfg_.upper_tokens) tokens_ = cfg_.upper_tokens;
}

uint32_t KvMemConfidenceBudget::round_up(uint32_t tokens) const {
    const uint32_t g = cfg_.round_tokens;
    if (tokens == 0) return 0;
    return ((tokens + g - 1) / g) * g;
}

void KvMemConfidenceBudget::observe(double token_conf) {
    // A NaN would poison every pooled minimum containing it and make the slope NaN,
    // which compares false against < 0 and would silently disable growth.
    if (!(token_conf == token_conf)) return;
    monitor_.observe(token_conf);
}

uint32_t KvMemConfidenceBudget::advance_round(uint64_t round_key) {
    // Idempotent per round: a re-entered boundary must not consume two rounds.
    if (round_key != 0) {
        if (round_key_seen_ && last_round_key_ == round_key) {
            return tokens_;
        }
        last_round_key_ = round_key;
        round_key_seen_ = true;
    }
    ++rounds_;
    if (tokens_ == 0 || tokens_ >= cfg_.upper_tokens) {
        return tokens_;  // policy disabled, or already at the ceiling
    }
    if (!monitor_.should_increase_budget()) {
        return tokens_;
    }
    // Rate limit: one growth per cooldown window. Without this a persistent negative
    // slope grows the budget on every consecutive round and races to the ceiling.
    if (grew_ && rounds_ - last_growth_round_ < cfg_.cooldown_rounds) {
        return tokens_;
    }
    const double scaled = static_cast<double>(tokens_) * static_cast<double>(cfg_.scale_up);
    uint32_t next = scaled >= static_cast<double>(cfg_.upper_tokens)
        ? cfg_.upper_tokens
        : round_up(static_cast<uint32_t>(scaled));
    if (next <= tokens_) {
        // scale_up is close to 1 or the rounding was flat: take one step so a
        // non-trivial scale_up still makes progress.
        next = std::min(cfg_.upper_tokens, round_up(tokens_ + cfg_.round_tokens));
    }
    if (next <= tokens_) {
        return tokens_;
    }
    tokens_ = next;
    ++growths_;
    last_growth_round_ = rounds_;
    grew_ = true;
    // The negative slope that justified this growth is now stale; clear the window so
    // the same evidence cannot immediately justify the next step.
    monitor_.reset();
    return tokens_;
}

} // namespace kvmem
