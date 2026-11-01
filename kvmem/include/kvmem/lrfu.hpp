#pragma once

#include "snapshot.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace kvmem {
// Immutable committed scores can be shared by checkpoints of the same round.
struct LrfuState {
    std::shared_ptr<const std::vector<double>> scores = std::make_shared<const std::vector<double>>();
    uint64_t round = 0;
    bool committed = false;
};

// Logical input boundaries are independent of physical replay checkpoints.
struct LrfuPoint {
    uint32_t input_end = 0;
    LrfuState state;
};

inline LrfuPoint lrfu_input_point(uint32_t end, uint32_t block_tokens, LrfuState state) {
    if (!end || !block_tokens) throw std::invalid_argument("invalid LRFU input prefix");
    const auto blocks = (uint64_t(end) + block_tokens - 1) / block_tokens;
    if (state.scores->size() > blocks) state.scores = std::make_shared<const std::vector<double>>(
        state.scores->begin(), state.scores->begin() + blocks);
    return {end, std::move(state)};
}

inline size_t lrfu_prefix_points(const std::vector<LrfuPoint>& points,
                               size_t verified_prefix, size_t input_size) {
    size_t count = 0;
    for (const auto& point : points) {
        // Repeating the same input starts from its parent round.
        if (point.input_end > verified_prefix || point.input_end >= input_size) break;
        ++count;
    }
    return count;
}

class LrfuHistory {
public:
    LrfuHistory(double lambda = 0.5, double top_fraction = 0.9)
        : lambda_(lambda), fraction_(top_fraction) {
        if (!std::isfinite(lambda) || lambda < 0 || lambda > 1 ||
            !std::isfinite(top_fraction) || top_fraction <= 0 || top_fraction > 1)
            throw std::invalid_argument("invalid LRFU decay or hit fraction");
    }
    void begin_round() {
        if (++state_.round == 0) throw std::overflow_error("LRFU round overflow");
        state_.committed = false;
        candidate_.reset();
    }
    // Repeated probes always derive from committed history, never another probe.
    void preview(const std::vector<double>& scores) {
        if (state_.committed) return;
        if (!state_.round) throw std::logic_error("LRFU probe outside a round");
        for (double value : scores) if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite LRFU retrieval score");
        if (scores.empty()) { candidate_.reset(); return; }
        auto next = std::make_shared<std::vector<double>>(scores.size());
        double threshold = -INFINITY;
        if (fraction_ < 1) {
            auto ranked = scores;
            const auto rank = std::min(scores.size() - 1,
                static_cast<size_t>((1 - fraction_) * scores.size()));
            std::nth_element(ranked.begin(), ranked.begin() + rank, ranked.end());
            threshold = ranked[rank];
        }
        // The hit fraction gates positive reinforcement only. Dropping an
        // unselected negative score to zero would reward it relative to a hit
        // with a less-negative score, reversing relevance with no prior history.
        for (size_t i = 0; i < scores.size(); ++i)
            (*next)[i] = lambda_ * (i < state_.scores->size() ? (*state_.scores)[i] : 0) +
                (scores[i] >= threshold ? scores[i] : std::min(scores[i], 0.));
        candidate_ = std::move(next);
    }
    bool commit() noexcept {
        if (state_.committed || !candidate_) return false;
        state_.scores = std::move(candidate_);
        state_.committed = true;
        return true;
    }
    double score(size_t block) const noexcept {
        const auto& scores = candidate_ ? *candidate_ : *state_.scores;
        return block < scores.size() ? scores[block] : 0;
    }
    LrfuState checkpoint() const { return state_; }
    void restore(const LrfuState& state) {
        if (!state.scores || (!state.round && (state.committed || !state.scores->empty())))
            throw std::invalid_argument("invalid LRFU checkpoint");
        for (double value : *state.scores) if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite LRFU checkpoint");
        state_ = state;
        candidate_.reset();
    }
    void discard_candidate() noexcept { candidate_.reset(); }
    size_t allocated_bytes() const noexcept {
        return state_.scores->capacity() * sizeof(double) +
            (candidate_ ? candidate_->capacity() * sizeof(double) : 0);
    }
private:
    double lambda_, fraction_;
    LrfuState state_;
    std::shared_ptr<const std::vector<double>> candidate_;
};

inline void write_lrfu(SnapshotWriter& out, const LrfuState& state) {
    out.scalar(state.round); out.scalar(uint8_t(state.committed)); out.vector(*state.scores);
}
inline LrfuState read_lrfu(SnapshotReader& in, uint32_t blocks) {
    LrfuState state;
    state.round = in.scalar<uint64_t>();
    const auto committed = in.scalar<uint8_t>();
    if (committed > 1) throw SnapshotCorrupt("invalid LRFU commit flag");
    state.committed = committed != 0;
    state.scores = std::make_shared<const std::vector<double>>(in.vector<double>(blocks));
    LrfuHistory validator;
    validator.restore(state);
    return state;
}

inline void write_lrfu_points(SnapshotWriter& out, const std::vector<LrfuPoint>& points) {
    out.scalar(uint32_t(points.size()));
    for (const auto& point : points) {
        out.scalar(point.input_end);
        write_lrfu(out, point.state);
    }
}

inline std::vector<LrfuPoint> read_lrfu_points(SnapshotReader& in, uint32_t input_size,
                                            uint32_t block_tokens) {
    const auto count = in.scalar<uint32_t>();
    if (!block_tokens || count > input_size) throw SnapshotCorrupt("invalid LRFU timeline size");
    std::vector<LrfuPoint> points;
    uint32_t previous = 0;
    uint64_t round = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const auto end = in.scalar<uint32_t>();
        if (end <= previous || end > input_size) throw SnapshotCorrupt("invalid LRFU input boundary");
        auto state = read_lrfu(in, (end + block_tokens - 1) / block_tokens);
        if (!state.committed || state.round <= round) throw SnapshotCorrupt("invalid LRFU timeline round");
        previous = end; round = state.round;
        points.push_back({end, std::move(state)});
    }
    return points;
}
} // namespace kvmem
