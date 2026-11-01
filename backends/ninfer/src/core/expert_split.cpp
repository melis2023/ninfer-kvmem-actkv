#include "core/expert_split.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ninfer {

TimeFit::TimeFit(double a, double b) : prior_(a), a_(a), b_(b) { add(2.0, a + 2.0 * b, 4.0); }

void TimeFit::add(double x, double t, double weight) {
    constexpr double kDecay = 0.99, kPull = 1.0, kReach = 4.0;
    const double line = at(x);
    t   = std::clamp(t, line / kReach, line * kReach);
    s0_ = kDecay * s0_ + weight;
    s1_ = kDecay * s1_ + weight * x;
    s2_ = kDecay * s2_ + weight * x * x;
    t0_ = kDecay * t0_ + weight * t;
    t1_ = kDecay * t1_ + weight * x * t;
    const double m00 = s0_ + kPull, det = m00 * s2_ - s1_ * s1_;
    if (!(det > 1e-12)) { return; }
    double a = ((t0_ + kPull * prior_) * s2_ - s1_ * t1_) / det;
    double b = (m00 * t1_ - s1_ * (t0_ + kPull * prior_)) / det;
    if (a < 0) {
        a = 0;
        b = t1_ / s2_;
    }
    if (std::isfinite(a) && std::isfinite(b) && b > 0) {
        a_ = a;
        b_ = b;
    }
}

ExpertSplit::ExpertSplit() = default;

double ExpertSplit::cpu_work(double megabytes, int pairs) {
    constexpr double kTokenWork = 0.25;
    return megabytes * (1.0 + kTokenWork * double(std::max(pairs, 1) - 1));
}

std::size_t ExpertSplit::cpu_count(std::span<const int> pairs, double megabytes) {
    constexpr std::uint32_t kProbe = 8;
    const std::size_t n = pairs.size();
    if (n == 0) { return 0; }
    std::vector<double> work(n + 1, 0.0); // the CPU's work on experts [g, n)
    for (std::size_t i = n; i-- > 0;) { work[i] = work[i + 1] + cpu_work(megabytes, pairs[i]); }
    std::size_t gpu = n;
    double soonest  = copy_.at(double(n) * megabytes);
    for (std::size_t g = n; g-- > 0;) {
        const double copies = g == 0 ? 0.0 : copy_.at(double(g) * megabytes);
        const double end    = std::max(copies, cpu_.at(work[g]));
        if (end < soonest) {
            soonest = end;
            gpu     = g;
        }
    }
    if (gpu == n && without_cpu_ >= kProbe) {
        gpu = n - 1;
    } else if (gpu == 0 && without_copies_ >= kProbe) {
        gpu = 1;
    }
    without_cpu_    = gpu < n ? 0 : without_cpu_ + 1;
    without_copies_ = gpu > 0 ? 0 : without_copies_ + 1;
    return n - gpu;
}

} // namespace ninfer
