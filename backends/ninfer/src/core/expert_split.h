#pragma once

// How a decode or verification call divides the routed experts its device cache lacks between
// copies to the GPU and products on the CPU, for every MoE family with host or disk experts: the
// split whose copies and CPU products end soonest by two straight lines fitted to earlier calls'
// copy and CPU times. The GPU takes the experts with the most pairs (a copy serves all of them at
// once; the CPU's work grows with each pair), the CPU the rest. A side left unused for eight calls
// takes one expert, so both lines follow the machine.

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer {

// t = a + b x, fitted by least squares with exponential forgetting and the intercept drawn toward
// its prior: how long a call's copies take for the bytes they move, or its CPU products for the
// work they do (x in megabytes, t in microseconds). The prior line seeds the fit. A sample counts at
// most four times the line's value (or a quarter of it), so a copy queued behind other transfers or
// a descheduled CPU request moves the line gradually; and a negative intercept is refitted as zero,
// the line through the origin, since samples of one size (one expert while a side only probes)
// leave the slope to the prior's point.
class TimeFit {
public:
    TimeFit(double a, double b);
    void add(double x, double t, double weight = 1.0);
    [[nodiscard]] double at(double x) const { return a_ + b_ * x; }

private:
    double prior_, a_, b_;
    double s0_ = 0, s1_ = 0, s2_ = 0, t0_ = 0, t1_ = 0;
};

class ExpertSplit {
public:
    ExpertSplit();
    // The CPU's work on one expert of `megabytes` stored bytes with `pairs` pairs: the decode of
    // each stored block serves every token, a token's products add about a quarter.
    [[nodiscard]] static double cpu_work(double megabytes, int pairs);
    // How many of a call's missing experts the CPU computes: the last ones of `pairs` (each
    // missing expert's pairs in the call, most first), all experts `megabytes` large.
    [[nodiscard]] std::size_t cpu_count(std::span<const int> pairs, double megabytes);
    // A call's copies of `megabytes` took `microseconds`; its CPU products of `work` took theirs.
    void copied(double megabytes, double microseconds) { copy_.add(megabytes, microseconds); }
    void computed(double work, double microseconds) { cpu_.add(work, microseconds); }

private:
    TimeFit copy_{10.0, 50.0}; // about 20 GB/s
    TimeFit cpu_{20.0, 100.0}; // about 10 GB/s, refitted by the first CPU calls
    std::uint32_t without_copies_ = 0, without_cpu_ = 0;
};

} // namespace ninfer
