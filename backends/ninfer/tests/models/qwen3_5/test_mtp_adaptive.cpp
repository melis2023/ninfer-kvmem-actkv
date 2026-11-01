// Adaptive MTP only chooses how many drafts a round verifies; greedy output is the same at every
// width. What these checks defend is the choice: the survival estimate, the measured cost model,
// and that the controller widens under full acceptance and settles low under none.

#include "runtime/contract/mtp_adaptive.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace {

using ninfer::runtime::MtpAdaptiveBatchController;
using ninfer::runtime::MtpAdaptiveSignal;
using ninfer::runtime::MtpRoundCostModel;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void survival_follows_acceptance() {
    MtpAdaptiveSignal fresh;
    const float prior = fresh.expected_tokens(3, 3);
    require(prior > 1.5F && prior < 2.5F, "a fresh signal expects the prior survival");
    require(fresh.expected_tokens(3, 0) == 1.0F, "no ready draft expects only the bonus token");

    MtpAdaptiveSignal accepting;
    for (int round = 0; round < 40; ++round) { accepting.observe(5, 5); }
    require(accepting.expected_tokens(5, 5) > 5.5F, "full acceptance expects every draft");
    require(accepting.confident_tail(5), "a long success streak is a confident tail");

    MtpAdaptiveSignal rejecting;
    for (int round = 0; round < 40; ++round) { rejecting.observe(5, 0); }
    require(rejecting.expected_tokens(5, 5) < 1.1F, "no acceptance expects only the bonus token");
}

void costs_are_measured_then_scaled() {
    MtpRoundCostModel costs;
    const float shape_one   = costs.cost(1, 1);
    const float shape_three = costs.cost(1, 3);
    require(shape_three > shape_one, "the relative shape grows with the width");
    for (int round = 0; round < 8; ++round) { costs.observe(1, 3, 0.030); }
    require(std::abs(costs.cost(1, 3) - 0.030F) < 1e-6F, "a measured width reports its seconds");
    require(std::abs(costs.cost(1, 1) - 0.030F * shape_one / shape_three) < 1e-6F,
            "an unmeasured width scales from the measured one");
    require(costs.cost(2, 3) == shape_three, "batch sizes are measured separately");
}

struct Simulation {
    std::uint32_t final_window = 0;
    std::uint32_t widest       = 0;
};

// Rounds of one row with `maximum` drafts always ready; `accept` gives the accepted drafts of a
// round at a width, `seconds` its cost.
template <typename Accept, typename Seconds>
Simulation simulate(std::uint32_t maximum, Accept accept, Seconds seconds,
                    std::uint32_t floor = 3) {
    MtpAdaptiveBatchController controller;
    controller.reset(maximum, floor);
    MtpAdaptiveSignal signal;
    Simulation out;
    const std::array<const MtpAdaptiveSignal*, 1> signals{&signal};
    const std::array<std::uint32_t, 1> available{maximum};
    const std::array<std::uint32_t, 1> room{4096};
    for (int round = 0; round < 300; ++round) {
        const std::uint32_t window = controller.select(signals, available, room, 7);
        require(window >= 1 && window <= maximum, "the selected width is outside the range");
        controller.observe_execution(1, window, seconds(window));
        signal.observe(window, accept(window));
        out.widest       = std::max(out.widest, window);
        out.final_window = window;
    }
    return out;
}

void the_controller_widens_and_narrows() {
    const auto seconds         = [](std::uint32_t window) { return 0.020 + 0.001 * window; };
    const Simulation accepting = simulate(7, [](std::uint32_t window) { return window; }, seconds);
    require(accepting.final_window >= 5, "full acceptance did not widen the window");
    const Simulation rejecting = simulate(7, [](std::uint32_t) { return 0U; }, seconds);
    require(rejecting.final_window == 3, "no acceptance did not settle at the narrowest width");
    const Simulation fixed_short =
        simulate(2, [](std::uint32_t window) { return window; }, seconds);
    require(fixed_short.widest <= 2, "the window exceeded its maximum");
    // With a floor of one (each draft a step of its own, as in Qwen3.8-Flash-Next) rejected drafts
    // settle at a single draft, and drafts that keep surviving still widen the window.
    const Simulation rejecting_floor_one =
        simulate(7, [](std::uint32_t) { return 0U; }, seconds, 1);
    require(rejecting_floor_one.final_window == 1, "a floor of one did not settle at one draft");
    const Simulation accepting_floor_one =
        simulate(7, [](std::uint32_t window) { return window; }, seconds, 1);
    require(accepting_floor_one.final_window >= 5, "a floor of one did not widen under acceptance");
    // Drafts that survive about two positions settle between the extremes when every step costs.
    const auto stepped = [](std::uint32_t window) { return 0.012 + 0.004 * window; };
    const Simulation partial = simulate(
        7, [](std::uint32_t window) { return window < 2 ? window : 2U; }, stepped, 1);
    require(partial.final_window >= 2 && partial.final_window <= 3,
            "two surviving drafts did not settle at two or three");
}

} // namespace

int main() {
    try {
        survival_follows_acceptance();
        costs_are_measured_then_scaled();
        the_controller_widens_and_narrows();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("adaptive MTP: survival, costs and width selection ok");
    return 0;
}
