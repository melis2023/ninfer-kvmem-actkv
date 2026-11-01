#pragma once

// Structured-output masks computed off the engine thread.
//
// A structured request samples under its grammar's token mask. Computing a mask can cost from
// microseconds to many milliseconds with a large vocabulary, so the cores compute it on a thread of
// their own (a GrammarMaskJob, from a fork of the grammar) while the GPU runs, instead of holding
// every request's round for it. A speculative round needs one mask per draft and the bonus; their
// columns fill in parallel (GrammarState::fill_masks) while the verification runs, which hides a few
// milliseconds per column. The adapted Qwen cores do not gate drafting on mask duration.

#include "ninfer/base.h"
#include "text/structured_output.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// One mask over this budget fails its request: the grammar is too costly to serve.
inline constexpr double kGrammarMaskBudgetSeconds = 5.0;

// Retained for the upstream Flint core, which is outside the Qwen integration scope.
// The adapted Qwen cores do not use this cost-based draft gate.
inline constexpr double kSpeculativeMaskSeconds = 0.004;
class GrammarMaskCost final {
public:
    void observe(double seconds) noexcept {
        seconds_ = seen_ ? seconds_ + kAlpha * (seconds - seconds_) : seconds;
        seen_    = true;
    }
    [[nodiscard]] bool speculates() const noexcept {
        return !seen_ || seconds_ <= kSpeculativeMaskSeconds;
    }
    [[nodiscard]] double seconds() const noexcept { return seconds_; }

private:
    static constexpr double kAlpha = 0.25;
    double seconds_ = 0.0;
    bool seen_      = false;
};

// Masks of one structured request computed on a thread of its own from a fork of its grammar taken
// at start: the next token's (no drafts), or a speculative round's columns after its drafts. The
// job owns its result; a request that goes away leaves a running job to finish and be discarded.
class GrammarMaskJob final {
public:
    [[nodiscard]] static std::shared_ptr<GrammarMaskJob> start(const text::GrammarState& grammar,
                                                               std::size_t words,
                                                               std::vector<TokenId> drafts = {}) {
        auto job = std::make_shared<GrammarMaskJob>();
        job->columns_ = drafts.size() + 1;
        job->words_.resize(words * job->columns_);
        std::thread([job, state = std::shared_ptr<text::GrammarState>(grammar.fork()),
                     drafts = std::move(drafts)] {
            const auto begin = std::chrono::steady_clock::now();
            try {
                state->fill_masks(job->words_, drafts);
            } catch (...) { job->error_ = std::current_exception(); }
            job->seconds_ =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            job->done_.store(true, std::memory_order_release);
        }).detach();
        return job;
    }

    [[nodiscard]] bool ready() const noexcept { return done_.load(std::memory_order_acquire); }
    void wait() const noexcept {
        while (!ready()) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    }
    // Valid once ready: `columns()` masks of `words` words each.
    [[nodiscard]] const std::uint32_t* masks() const noexcept { return words_.data(); }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] double seconds() const noexcept { return seconds_; }
    [[nodiscard]] std::exception_ptr error() const noexcept { return error_; }

private:
    std::vector<std::uint32_t> words_;
    std::size_t columns_ = 1;
    double seconds_      = 0.0;
    std::exception_ptr error_;
    std::atomic<bool> done_{false};
};

// Why a finished mask job cannot serve: its grammar had no admissible token, or it took over the
// budget. Empty when it can.
[[nodiscard]] inline std::string grammar_mask_problem(const GrammarMaskJob& job) {
    if (job.error()) {
        try {
            std::rethrow_exception(job.error());
        } catch (const std::exception& e) {
            return e.what();
        } catch (...) { return "the grammar mask failed"; }
    }
    if (job.seconds() > kGrammarMaskBudgetSeconds) {
        return "the grammar mask took " + std::to_string(job.seconds()) + " s, over the " +
               std::to_string(kGrammarMaskBudgetSeconds) + " s budget";
    }
    return {};
}

} // namespace ninfer::runtime
