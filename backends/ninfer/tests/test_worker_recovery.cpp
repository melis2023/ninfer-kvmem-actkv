#include "runtime/engine/worker_recovery.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::runtime::exception_text;
using ninfer::runtime::RecoveryStreak;
using Clock = RecoveryStreak::Clock;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using std::chrono::seconds;
    const Clock::time_point t0{seconds(1000)};
    int failures = 0;

    // A tight loop: the failure after `maximum` recoveries fails every request.
    RecoveryStreak loop(3, seconds(30));
    failures +=
        check(loop.record_failure(t0) && loop.record_failure(t0 + seconds(1)) &&
                  loop.record_failure(t0 + seconds(2)) && !loop.record_failure(t0 + seconds(3)),
              "the fourth failure in a tight loop was recovered from");
    failures += check(loop.count() == 4, "the refused failure is not reported in the streak");

    // A completed work unit ends the streak.
    RecoveryStreak served(3, seconds(30));
    for (int i = 0; i < 3; ++i) { (void)served.record_failure(t0 + seconds(i)); }
    served.record_success();
    failures += check(served.count() == 0 && served.record_failure(t0 + seconds(3)),
                      "a completed work unit did not end the streak");

    // One client resending a request that fails every two minutes, with no work unit completing
    // in between, never exhausts the streak.
    RecoveryStreak repeating(3, seconds(30));
    bool refused = false;
    for (int i = 0; i < 20; ++i) {
        refused = refused || !repeating.record_failure(t0 + seconds(120 * i));
    }
    failures += check(!refused && repeating.count() == 1,
                      "failures further apart than the window exhausted the streak");

    // The window runs from the last failure: one just inside it keeps the streak and extends it.
    RecoveryStreak spaced(2, seconds(30));
    (void)spaced.record_failure(t0);
    (void)spaced.record_failure(t0 + seconds(29));
    failures += check(!spaced.record_failure(t0 + seconds(58)),
                      "failures inside the window did not exhaust the streak");
    RecoveryStreak boundary(2, seconds(30));
    (void)boundary.record_failure(t0);
    (void)boundary.record_failure(t0 + seconds(29));
    failures += check(boundary.record_failure(t0 + seconds(59)) && boundary.count() == 1,
                      "a gap of the whole window did not end the streak");

    failures += check(exception_text(std::make_exception_ptr(std::runtime_error("boom"))) == "boom",
                      "std::exception text missing");
    failures += check(exception_text(std::make_exception_ptr(42)) == "unknown exception",
                      "a non-std exception is not reported as unknown");
    failures += check(exception_text(nullptr) == "unknown exception",
                      "a null exception is not reported as unknown");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
