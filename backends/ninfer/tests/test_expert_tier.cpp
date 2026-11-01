// The host side of every MoE family's host and disk experts: the expert tier must fill free slots
// first, swap a candidate in only when its count beats the victim's by the gain, pair the hottest
// candidates with the coldest victims of their own pool, order a round by gain over every pool,
// cap it, decay the counts, and admit a candidate over a held expert its count beats, never one
// its call routes. The split must give the CPU more of the misses when copies turn out slow and
// fewer when its products do, and probe a side it left unused. The pinned pool must run every index
// of a job exactly once, run after run, rethrow a job's failure, run a side once on the calling
// thread, finish every job before a failing side's error and stamp its last job's end before it
// returns; row tasks must cover a phase's rows once in about three tasks per thread.
#include "core/cpu_spin_pool.h"
#include "core/expert_split.h"
#include "core/expert_tier.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace ninfer;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool near(float a, float b) { return std::fabs(a - b) < 1e-4F; }

void tier_rounds() {
    using State = ExpertTier::State;
    // Two layers of eight experts, each with its own pool; a round every second pass.
    ExpertTier tier({8, 8}, {0, 1}, {.every = 2});
    tier.set_free(0, 2);
    for (std::int32_t e = 0; e < 4; ++e) { tier.set(1, e, State::Held); }
    tier.route(0, std::vector<std::int32_t>{1, 1, 1, 2, 2, 3, -1, 8});
    tier.route(1, std::vector<std::int32_t>{5, 5, 5, 5, 6, 6, 6, 0});
    require(near(tier.count(0, 1), 3) && near(tier.count(0, 3), 1), "routes count once per pair");
    require(tier.end_pass().empty(), "no round before `every` passes");
    const auto swaps = tier.end_pass();
    // Layer 0 fills its free slots with experts 1 and 2 (3 is below min_count); layer 1 swaps 5
    // and 6 for its coldest held experts 1 and 2 (0 was routed once), best gain first.
    require(swaps.size() == 4, "a round's swaps: " + std::to_string(swaps.size()));
    require(swaps[0].layer == 1 && swaps[0].in == 5 && swaps[0].out_layer == 1 && swaps[0].out == 1 &&
                near(swaps[0].gain, 4),
            "the best gain comes first");
    require(swaps[1].layer == 0 && swaps[1].in == 1 && swaps[1].out == -1, "a free slot needs no victim");
    require(swaps[2].layer == 1 && swaps[2].in == 6 && swaps[2].out == 2, "the next coldest leaves next");
    require(swaps[3].layer == 0 && swaps[3].in == 2 && swaps[3].out == -1, "the second free slot");
    require(near(tier.count(1, 5), 4 * 0.7F), "a round decays every count");

    // A gain below min_gain swaps nothing; the cap keeps the best.
    ExpertTier close({4}, {0}, {.every = 1});
    close.set(0, 0, State::Held);
    close.route(0, std::vector<std::int32_t>{0, 0, 1, 1, 1});
    require(close.end_pass().empty(), "a candidate one route ahead does not swap");
    ExpertTier capped({4, 4}, {0, 1}, {.every = 1, .swaps = 1});
    capped.set(0, 0, State::Held);
    capped.set(1, 0, State::Held);
    capped.route(0, std::vector<std::int32_t>{1, 1, 1});
    capped.route(1, std::vector<std::int32_t>{1, 1, 1, 1, 1});
    const auto best = capped.end_pass();
    require(best.size() == 1 && best[0].layer == 1 && best[0].in == 1, "the cap keeps the best swap");
    // Pending experts are neither candidates nor victims.
    ExpertTier pending({4}, {0}, {.every = 1});
    pending.set(0, 0, State::Pending);
    pending.set(0, 1, State::Held);
    pending.route(0, std::vector<std::int32_t>{0, 0, 0, 0, 0});
    require(pending.end_pass().empty(), "a pending expert is no candidate");
}

void tier_admission() {
    using State = ExpertTier::State;
    ExpertTier tier({8}, {0}, {.every = 0});
    for (std::int32_t e = 0; e < 4; ++e) { tier.set(0, e, State::Held); }
    tier.route(0, std::vector<std::int32_t>{0, 0, 0, 1, 2, 2});
    require(tier.end_pass().empty(), "every = 0 runs no rounds");
    std::vector<unsigned char> busy(8, 0);
    // Coldest first: 3 (0 routes), 1 (1), 2 (2), 0 (3).
    require(tier.admission_victim(0, 7, 1, busy).expert < 0, "one pair does not beat 0 + 1.5");
    busy[3] = 1;
    const auto first = tier.admission_victim(0, 7, 3, busy);
    require(first.expert == 1, "the coldest expert the call does not route leaves");
    tier.set(0, 1, State::Out);
    const auto second = tier.admission_victim(0, 6, 5, busy);
    require(second.expert == 2, "an expert that left is not offered again");
    tier.set(0, 2, State::Out);
    require(tier.admission_victim(0, 5, 2, busy).expert < 0, "a cold candidate beats no warmer expert");
    require(tier.end_pass().empty(), "a pass ends");
    std::fill(busy.begin(), busy.end(), 0);
    // Routed since the pass began, expert 3 is no longer the one to beat.
    tier.route(0, std::vector<std::int32_t>(6, 3));
    require(tier.admission_victim(0, 7, 4, busy).expert < 0, "no held expert is cold enough");
    require(tier.admission_victim(0, 7, 5, busy).expert == 0, "the scan passes the warmed expert");

    // One pool for two layers: the victim may be another layer's.
    ExpertTier shared({4, 4}, {0, 0}, {.every = 0});
    shared.set(1, 2, State::Held);
    shared.set(0, 1, State::Held);
    shared.route(0, std::vector<std::int32_t>{1, 1, 1});
    require(shared.end_pass().empty(), "a pass ends");
    const auto other = shared.admission_victim(0, 3, 2, std::vector<unsigned char>(4, 0));
    require(other.layer == 1 && other.expert == 2, "a shared pool's coldest expert may be another layer's");
}

// The tier as a plain whole sort states it: held experts coldest first as the pass ended (lower
// entry first among equals), candidates hottest first, every pool's swaps until the first one that
// does not gain, the best `swaps` of them over all pools.
struct WholeSortTier {
    std::vector<std::uint32_t> pool_of; // by entry
    std::vector<float> counts;
    std::vector<ExpertTier::State> states;
    std::vector<std::uint32_t> free;
    std::vector<std::vector<std::size_t>> cold;
    std::vector<std::size_t> next;
    ExpertTierOptions options;
    std::uint32_t passes = 0;

    void refresh() {
        for (auto& list : cold) { list.clear(); }
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (states[i] == ExpertTier::State::Held) { cold[pool_of[i]].push_back(i); }
        }
        for (auto& list : cold) {
            std::stable_sort(list.begin(), list.end(),
                             [&](std::size_t a, std::size_t b) { return counts[a] < counts[b]; });
        }
        std::fill(next.begin(), next.end(), 0);
    }

    std::int64_t victim(std::size_t entry, float pairs, const std::vector<unsigned char>& busy) {
        const float candidate = counts[entry] + pairs;
        auto& list            = cold[pool_of[entry]];
        auto& first           = next[pool_of[entry]];
        while (first < list.size() && states[list[first]] != ExpertTier::State::Held) { ++first; }
        std::size_t held = 0;
        for (std::size_t at = first; at < list.size() && held < 32; ++at) {
            const std::size_t i = list[at];
            if (states[i] != ExpertTier::State::Held) { continue; }
            ++held;
            if (busy[i] != 0) { continue; }
            if (candidate >= counts[i] + options.min_gain) { return std::int64_t(i); }
        }
        return -1;
    }

    std::vector<std::tuple<std::size_t, std::int64_t, float>> end_pass() {
        std::vector<std::tuple<std::size_t, std::int64_t, float>> swaps;
        if (++passes >= options.every) {
            passes = 0;
            for (std::uint32_t pool = 0; pool < cold.size(); ++pool) {
                std::vector<std::size_t> in, out;
                for (std::size_t i = 0; i < counts.size(); ++i) {
                    if (pool_of[i] != pool) { continue; }
                    if (states[i] == ExpertTier::State::Held) {
                        out.push_back(i);
                    } else if (states[i] == ExpertTier::State::Out &&
                               counts[i] >= options.min_count) {
                        in.push_back(i);
                    }
                }
                std::stable_sort(in.begin(), in.end(), [&](std::size_t a, std::size_t b) {
                    return counts[a] > counts[b];
                });
                std::stable_sort(out.begin(), out.end(), [&](std::size_t a, std::size_t b) {
                    return counts[a] < counts[b];
                });
                std::size_t n = 0;
                for (std::size_t c = 0; c < in.size(); ++c) {
                    if (c < free[pool]) {
                        swaps.emplace_back(in[c], -1, counts[in[c]]);
                        continue;
                    }
                    if (n == out.size()) { break; }
                    const float gain = counts[in[c]] - counts[out[n]];
                    if (gain < options.min_gain) { break; }
                    swaps.emplace_back(in[c], std::int64_t(out[n++]), gain);
                }
            }
            std::stable_sort(swaps.begin(), swaps.end(), [](const auto& a, const auto& b) {
                return std::get<2>(a) > std::get<2>(b);
            });
            if (swaps.size() > options.swaps) { swaps.resize(options.swaps); }
            for (auto& count : counts) { count *= options.decay; }
        }
        refresh();
        return swaps;
    }
};

// Thousands of held experts: a pass orders only the coldest few hundred, the scans of a pass may
// pass all of them, and a round pairs a few hundred candidates with victims. Every victim and
// every round must be those of the whole sort, many passes in a row, with counts that tie.
void tier_matches_whole_sort() {
    using State                     = ExpertTier::State;
    constexpr std::uint32_t kLayers = 3, kExperts = 700;
    // Layers 0 and 1 share pool 0 (1,400 entries), layer 2 owns pool 1.
    const ExpertTierOptions options{.every = 2, .swaps = 96};
    ExpertTier tier(std::vector<std::uint32_t>(kLayers, kExperts), {0, 0, 1}, options);
    WholeSortTier whole;
    whole.options = options;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        for (std::uint32_t e = 0; e < kExperts; ++e) { whole.pool_of.push_back(l < 2 ? 0 : 1); }
    }
    const std::size_t entries = whole.pool_of.size();
    whole.counts.assign(entries, 0);
    whole.states.assign(entries, State::Out);
    whole.free.assign(2, 0);
    whole.cold.resize(2);
    whole.next.assign(2, 0);
    const auto at = [](std::size_t i) {
        return std::pair{std::uint32_t(i / kExperts), std::int32_t(i % kExperts)};
    };
    const auto set = [&](std::size_t i, State state) {
        const auto [l, e] = at(i);
        tier.set(l, e, state);
        whole.states[i] = state;
    };
    std::uint64_t seed = 12345;
    const auto random  = [&](std::uint64_t n) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return (seed >> 33) % n;
    };
    for (std::size_t i = 0; i < entries; ++i) {
        if (random(10) < 8) { set(i, State::Held); }
    }
    std::size_t victims = 0, swapped = 0;
    for (int pass = 0; pass < 40; ++pass) {
        // Routes: small integer counts, so many experts tie.
        for (int r = 0; r < 3000; ++r) {
            const std::size_t i = random(4) == 0 ? random(60) : random(entries);
            const auto [l, e]   = at(i);
            const float pairs   = float(1 + random(3));
            tier.route(l, e, pairs);
            whole.counts[i] += pairs;
        }
        // Misses: admissions take victims; on some passes the 600 coldest of pool 0 leave first,
        // so its scans reach past the experts a pass orders.
        if (pass % 3 == 1) {
            const auto coldest = whole.cold[0];
            for (std::size_t c = 0; c < coldest.size() && c < 600; ++c) {
                set(coldest[c], State::Out);
            }
        }
        for (int m = 0; m < 40; ++m) {
            const std::size_t miss = random(entries);
            if (whole.states[miss] != State::Out) { continue; }
            const auto [l, e] = at(miss);
            std::vector<unsigned char> busy(entries, 0), layer_busy(kExperts, 0);
            for (int b = 0; b < 6; ++b) {
                const std::size_t routed      = l * kExperts + random(kExperts);
                busy[routed]                  = 1;
                layer_busy[routed % kExperts] = 1;
            }
            const float pairs   = float(random(40));
            const auto expected = whole.victim(miss, pairs, busy);
            const auto got      = tier.admission_victim(l, e, pairs, layer_busy);
            const std::int64_t got_entry =
                got.expert < 0 ? -1 : std::int64_t(got.layer) * kExperts + got.expert;
            require(got_entry == expected, "pass " + std::to_string(pass) + ": admission victim " +
                                               std::to_string(got_entry) + " instead of " +
                                               std::to_string(expected));
            if (expected >= 0) {
                set(std::size_t(expected), State::Out);
                set(miss, State::Held);
                ++victims;
            }
        }
        const std::uint32_t free0 = std::uint32_t(random(150)), free1 = std::uint32_t(random(3));
        tier.set_free(0, free0);
        tier.set_free(1, free1);
        whole.free          = {free0, free1};
        const auto swaps    = tier.end_pass();
        const auto expected = whole.end_pass();
        require(swaps.size() == expected.size(),
                "pass " + std::to_string(pass) + ": " + std::to_string(swaps.size()) +
                    " swaps instead of " + std::to_string(expected.size()));
        for (std::size_t s = 0; s < swaps.size(); ++s) {
            const auto& [in, out, gain] = expected[s];
            const bool same =
                std::size_t(swaps[s].layer) * kExperts + std::size_t(swaps[s].in) == in &&
                (out < 0 ? swaps[s].out < 0
                         : std::size_t(swaps[s].out_layer) * kExperts + std::size_t(swaps[s].out) ==
                               std::size_t(out)) &&
                swaps[s].gain == gain;
            require(same,
                    "pass " + std::to_string(pass) + ": swap " + std::to_string(s) + " differs");
        }
        for (const auto& [in, out, gain] : expected) {
            if (out >= 0) { set(std::size_t(out), State::Out); }
            set(in, State::Held);
            ++swapped;
        }
        require(std::fabs(tier.count(0, 0) - whole.counts[0]) <= 1e-6F * (1 + whole.counts[0]),
                "counts decay alike");
    }
    require(victims > 50 && swapped > 100,
            "the scenario admits and swaps: " + std::to_string(victims) + " victims, " +
                std::to_string(swapped) + " swaps");
}

void split() {
    // Ten missing experts of 2 MB, one pair each.
    const std::vector<int> pairs(10, 1);
    ExpertSplit slow_copies;
    for (int i = 0; i < 400; ++i) {
        slow_copies.copied(10.0, 10000.0);    // 1 GB/s
        slow_copies.computed(10.0, 500.0);    // 20 GB/s of work
    }
    ExpertSplit slow_cpu;
    for (int i = 0; i < 400; ++i) {
        slow_cpu.copied(10.0, 400.0);         // 25 GB/s
        slow_cpu.computed(10.0, 20000.0);     // 0.5 GB/s of work
    }
    const auto fast_cpu_share = slow_copies.cpu_count(pairs, 2.0);
    const auto slow_cpu_share = slow_cpu.cpu_count(pairs, 2.0);
    require(fast_cpu_share > slow_cpu_share,
            "slow copies give the CPU more: " + std::to_string(fast_cpu_share) + " vs " +
                std::to_string(slow_cpu_share));
    require(fast_cpu_share >= 8 && slow_cpu_share == 0, "the faster side takes nearly all");
    // A side left out for eight calls takes one expert.
    std::size_t probes = 0;
    for (int call = 0; call < 9; ++call) { probes += slow_cpu.cpu_count(pairs, 2.0); }
    require(probes == 1, "the CPU is probed once in nine calls: " + std::to_string(probes));
    require(ExpertSplit::cpu_work(2.0, 5) > ExpertSplit::cpu_work(2.0, 1), "more pairs, more work");
}

void pool() {
    for (const bool pin : {false, true}) {
        CpuSpinPool workers(4, pin);
        require(workers.threads() == 4 && workers.pinned() == pin, "the pool's threads");
        for (int run = 0; run < 200; ++run) {
            const std::size_t count = std::size_t(1 + run * 7 % 97);
            std::vector<std::atomic<int>> hits(count);
            if (run == 50) { workers.spin(true); }
            if (run == 150) { workers.spin(false); }
            workers.run(count, [&](std::size_t i) { hits[i].fetch_add(1); });
            for (std::size_t i = 0; i < count; ++i) {
                require(hits[i].load() == 1, "every index runs once");
            }
        }
        bool thrown = false;
        try {
            workers.run(16, [](std::size_t i) {
                if (i == 5) { throw std::runtime_error("job failure"); }
            });
        } catch (const std::runtime_error&) { thrown = true; }
        require(thrown, "a job's failure reaches the caller");
        std::atomic<int> after{0};
        workers.run(8, [&](std::size_t) { after.fetch_add(1); });
        require(after.load() == 8, "the pool runs after a failure");
        // A side runs once on the calling thread beside the jobs; a failing side still waits
        // for every job before its failure reaches the caller.
        std::vector<std::atomic<int>> hits(64);
        int sides         = 0;
        const auto caller = std::this_thread::get_id();
        workers.run(
            hits.size(), [&](std::size_t i) { hits[i].fetch_add(1); },
            [&] { sides += std::this_thread::get_id() == caller ? 1 : 100; });
        require(sides == 1, "the side runs once on the caller");
        for (const auto& hit : hits) {
            require(hit.load() == 1, "every index runs once beside a side");
        }
        require(workers.finished().time_since_epoch().count() > 0, "the last job's end is stamped");
        // finished() is the end of this run's last job once run() returns, never an earlier
        // run's: the CPU-time fits of the expert split read it.
        workers.spin(true);
        for (int run = 0; run < 20000; ++run) {
            std::atomic<std::chrono::steady_clock::rep> last{0};
            workers.run(std::size_t(1 + run % 6), [&](std::size_t) {
                const auto end = std::chrono::steady_clock::now().time_since_epoch().count();
                auto seen      = last.load();
                while (seen < end && !last.compare_exchange_weak(seen, end)) {}
            });
            require(workers.finished().time_since_epoch().count() >= last.load(),
                    "finished() is this run's last job's end");
        }
        workers.spin(false);
        std::atomic<int> late{0};
        thrown = false;
        try {
            workers.run(
                32,
                [&](std::size_t) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                    late.fetch_add(1);
                },
                [] { throw std::runtime_error("side failure"); });
        } catch (const std::runtime_error&) { thrown = true; }
        require(thrown && late.load() == 32, "a failing side waits for every job");
    }
    require(pool_cpu_count() >= 1 && pool_cpu_count() <= std::max(1U, usable_cpus() / 2),
            "the default pool leaves half the CPUs to the rest of the process");
    require(!physical_core_cpus().empty(), "every machine has a core");
}

// Row tasks cover every row of every part once, stay in their part, start at a multiple of the
// quantum and give a phase of one or two experts about three tasks per thread.
void rows() {
    for (const std::uint32_t threads : {1U, 4U, 15U, 29U}) {
        for (const int experts : {1, 2, 5, 12}) {
            std::vector<RowPart> parts;
            for (int e = 0; e < experts; ++e) {
                parts.push_back({640, 1.0 + 0.25 * double(e % 3)});
                parts.push_back({640, 1.0 + 0.25 * double(e % 3)});
            }
            parts.push_back({0, 1.0});
            parts.push_back({40, 1.0});
            const auto tasks = row_tasks(parts, threads, 16);
            std::vector<std::vector<int>> covered(parts.size());
            for (std::size_t p = 0; p < parts.size(); ++p) {
                covered[p].assign(std::size_t(parts[p].rows), 0);
            }
            for (const auto& task : tasks) {
                require(task.part < parts.size() && task.begin < task.end &&
                            task.end <= parts[task.part].rows && task.begin % 16 == 0,
                        "a task stays in its part on a quantum boundary");
                for (int r = task.begin; r < task.end; ++r) {
                    ++covered[task.part][std::size_t(r)];
                }
            }
            for (const auto& part : covered) {
                for (const int count : part) { require(count == 1, "every row once"); }
            }
            const double per_thread = double(tasks.size()) / double(threads);
            require(experts > 2 || threads == 1 || (per_thread >= 2.0 && per_thread <= 4.5),
                    "about three tasks per thread: " + std::to_string(tasks.size()) + " for " +
                        std::to_string(threads));
        }
    }
    require(row_tasks({}, 8, 16).empty(), "no rows, no tasks");
}

} // namespace

int main() {
    try {
        tier_rounds();
        tier_admission();
        tier_matches_whole_sort();
        split();
        pool();
        rows();
        std::cout << "PASS expert tier, split and pool\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
