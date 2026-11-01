// The expert cache must never change what a kernel reads: after any sequence of routes, rounds,
// admissions and growth, every table entry points either at the expert's bytes in the pinned host
// block or at a device slot holding the same bytes, a cached down matrix followed by zeros (the
// expert matrix kernel reads past a down row's end). Its rounds must fill free slots with the
// experts decode routes most and swap a much more routed expert for a cold one, with the victim's
// entries back on the host at once and the candidate's moved only once its copy has landed; prompt
// routes must not steer them. A routing profile must size each layer's share of the slots and fill
// them. An admission takes a free slot, else the coldest expert its call does not route when the
// newcomer's count beats it, and counts the newcomer's routes in that call as misses. Lent buffers
// are free slots until a wide pass reclaims them: their most routed experts move into colder slots
// of the cache's own, no entry points inside them while the pass writes them, and they come back
// zeroed, refilled with the most routed experts outside the cache and free for admissions past
// those. A round's swaps start over the passes until the next round, a share of them a pass.
#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

constexpr int kExperts = 64;

struct Bank {
    std::int64_t bytes;
    std::unique_ptr<PinnedHostBuffer> host;
    DeviceBuffer table;
    std::vector<const void*> pointers;

    Bank(std::int64_t expert_bytes, std::mt19937& random)
        : bytes(expert_bytes), host(std::make_unique<PinnedHostBuffer>(expert_bytes * kExperts)),
          table(kExperts * sizeof(void*)) {
        auto* data = static_cast<std::uint8_t*>(host->data());
        for (std::int64_t i = 0; i < expert_bytes * kExperts; ++i) {
            data[i] = std::uint8_t(random());
        }
        for (int e = 0; e < kExperts; ++e) { pointers.push_back(data + e * expert_bytes); }
        table.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
    }

    [[nodiscard]] ExpertBank view() const {
        return {.expert_bytes = bytes, .host = pointers, .table = table.p};
    }

    // Every entry's bytes equal the expert's, and `tail` zero bytes follow a cached one; returns
    // the experts whose entries point at device memory.
    std::vector<int> verify(const std::string& label, std::size_t tail = 0) const {
        require(cudaDeviceSynchronize() == cudaSuccess, label + ": the device failed");
        std::vector<const void*> entries(kExperts);
        table.copy_to_host(entries.data(), entries.size() * sizeof(void*));
        std::vector<int> cached;
        std::vector<std::uint8_t> bytes_read(static_cast<std::size_t>(bytes) + tail);
        for (int e = 0; e < kExperts; ++e) {
            if (entries[e] == pointers[e]) { continue; }
            cached.push_back(e);
            require(cudaMemcpy(bytes_read.data(), entries[e], bytes_read.size(),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                    label + ": a cached entry is not readable");
            require(std::memcmp(bytes_read.data(), pointers[e], std::size_t(bytes)) == 0,
                    label + ": a slot differs from its expert's bytes");
            require(std::all_of(bytes_read.begin() + bytes, bytes_read.end(),
                                [](std::uint8_t b) { return b == 0; }),
                    label + ": a cached down matrix is not followed by zeros");
        }
        return cached;
    }
};

// Two layers whose banks have different expert sizes, as mixed block types do: slots of 4096 +
// 4096 + 2304 bytes for layer 0 and 1536 + 3072 + 1280 for layer 1 (each down followed by its 256
// zero bytes, aligned), 16,384 bytes a slot pair.
struct Model {
    std::mt19937 random{9100u};
    Bank g0{4096, random}, u0{4096, random}, d0{2048, random};
    Bank g1{1536, random}, u1{3072, random}, d1{1024, random};

    std::vector<ExpertCacheLayer> layers(const DeviceContext& device) const {
        return {
            {.rank = 0, .stream = device.stream, .gate = g0.view(), .up = u0.view(), .down = d0.view()},
            {.rank = 0, .stream = device.stream, .gate = g1.view(), .up = u1.view(), .down = d1.view()},
        };
    }

    // The experts layer `l` caches, after checking that its three banks agree and hold their bytes.
    std::vector<int> cached(int l, const std::string& label) const {
        const Bank* banks[3] = {l == 0 ? &g0 : &g1, l == 0 ? &u0 : &u1, l == 0 ? &d0 : &d1};
        const auto gate = banks[0]->verify(label + " gate");
        require(banks[1]->verify(label + " up") == gate && banks[2]->verify(label + " down", 256) == gate,
                label + ": a layer's three banks cache the same experts");
        return gate;
    }
};

bool holds(const std::vector<int>& set, int expert) {
    return std::find(set.begin(), set.end(), expert) != set.end();
}

// Rounds every four decode passes (the default tier).
void rounds(DeviceContext& device) {
    Model model;
    ExpertCache cache(device, model.layers(device), std::vector<std::uint64_t>{2 * 102400});
    require(cache.stats().slots == 12 + 12,
            "without a profile every layer gets the same slots: " + std::to_string(cache.stats().slots));
    const auto pass = [&](const std::vector<std::int32_t>& ids, bool learn) {
        cache.observe(0, ids, learn);
        cache.adapt();
    };
    // Layer 0 routes experts 5 and 7 forty times a pass and 0..39 once each.
    std::vector<std::int32_t> hot;
    for (int t = 0; t < 40; ++t) {
        hot.push_back(5);
        hot.push_back(7);
        hot.push_back(t);
    }
    for (int p = 0; p < 3; ++p) { pass(hot, true); }
    require(model.cached(0, "before a round").empty(), "nothing enters before the first round");
    pass(hot, true);
    require(model.cached(0, "copies in flight").empty() && cache.cached(0, 0, 5) == nullptr,
            "a candidate's entries stay on the host until its copy has landed");
    require(cudaDeviceSynchronize() == cudaSuccess, "the round's copies");
    pass({}, true);
    const auto first = model.cached(0, "first round");
    require(first.size() == 12 && holds(first, 5) && holds(first, 7),
            "the round fills the free slots, the most routed experts first");
    require(cache.cached(0, 0, 5) != nullptr, "a landed candidate is cached");
    require(model.cached(1, "an idle layer").empty(), "a layer without routes takes nothing");

    // Experts 50..53 now route forty times a pass: the next round swaps them for four of the
    // coldest held experts (routed four times), never for 5 or 7.
    const std::vector<std::int32_t> fresh = [] {
        std::vector<std::int32_t> ids;
        for (int t = 0; t < 40; ++t) {
            for (int e = 50; e < 54; ++e) { ids.push_back(e); }
        }
        return ids;
    }();
    for (int p = 0; p < 3; ++p) { pass(fresh, true); }
    const auto swapped = model.cached(0, "victims left");
    require(swapped.size() == 8 && holds(swapped, 5) && holds(swapped, 7),
            "the victims' entries return to the host at once: " + std::to_string(swapped.size()));
    require(cudaDeviceSynchronize() == cudaSuccess, "the round's copies");
    pass({}, true);
    const auto second = model.cached(0, "second round");
    require(second.size() == 12 && holds(second, 50) && holds(second, 53) && holds(second, 5),
            "the candidates take the victims' slots");

    // Prompt routes do not steer the rounds: expert 60, routed only by prompts, never enters.
    const std::vector<std::int32_t> prompt(400, 60);
    for (int p = 0; p < 8; ++p) { pass(prompt, false); }
    require(cudaDeviceSynchronize() == cudaSuccess, "the rounds' copies");
    pass({}, true);
    require(!holds(model.cached(0, "prompt routes"), 60), "prompt routes are not counted");
    const auto stats = cache.stats();
    require(stats.admitted == 16 && stats.routes == 4 * 120 + 3 * 160 + 8 * 400,
            "the cache counts routes and admissions");
    cache.observe(0, std::vector<std::int32_t>{5, 7, 50, 60}, false);
    require(cache.stats().hits == stats.hits + 3 && cache.stats().routes == stats.routes + 4,
            "a route to a cached expert is a hit");
}

// A round's swaps start over the passes until the next round, an even share of the tier's swaps
// a pass (rounds every four passes of up to eight swaps: two a pass), best gain first, so one pass
// never copies a whole round.
void paced(DeviceContext& device) {
    Model model;
    ExpertCache cache(device, model.layers(device), std::vector<std::uint64_t>{2 * 102400}, nullptr,
                      false, ExpertTierOptions{.every = 4, .swaps = 8});
    // Layer 0 routes experts 0..9, expert e 10 + e times a pass.
    std::vector<std::int32_t> ids;
    for (int e = 0; e < 10; ++e) { ids.insert(ids.end(), std::size_t(10 + e), e); }
    const auto pass = [&] {
        cache.observe(0, ids, true);
        cache.adapt();
        require(cudaDeviceSynchronize() == cudaSuccess, "a pass's copies");
    };
    for (int p = 0; p < 4; ++p) { pass(); }
    require(model.cached(0, "round started").empty(), "the round's first share is in flight");
    // Each later pass lands the share before it and starts the next: 9 8, then 7 6, 5 4, 3 2.
    const std::vector<std::vector<int>> landed = {{8, 9}, {6, 7, 8, 9}, {4, 5, 6, 7, 8, 9}};
    for (const auto& expected : landed) {
        pass();
        auto cached = model.cached(0, "paced");
        std::sort(cached.begin(), cached.end());
        require(cached == expected,
                "a pass starts two swaps, the most routed first: " + std::to_string(cached.size()));
    }
    // The fifth pass after the round lands its last share; the next round takes 1 and 0.
    pass();
    require(model.cached(0, "round done").size() == 8, "the round's eight swaps have landed");
    pass();
    require(model.cached(0, "next round").size() == 10, "the next round fills the rest");
    require(cache.stats().admitted == 10, "every swap started once");
}

// Admits `expert` into layer `l` and copies it in and patches its entries, as the expert misses
// do; the result's slot is null when the cache refused it.
ExpertCache::Admitted admit(ExpertCache& cache, const Model& model, std::size_t l, std::int32_t expert,
                            int pairs, const std::vector<unsigned char>& in_use) {
    ExpertCache::Admitted out;
    if (!cache.admit(l, expert, pairs, in_use, out)) { return out; }
    const Bank* banks[3] = {l == 0 ? &model.g0 : &model.g1, l == 0 ? &model.u0 : &model.u1,
                            l == 0 ? &model.d0 : &model.d1};
    for (int k = 0; k < 3; ++k) {
        require(cudaMemcpy(out.slot[k], banks[k]->pointers[expert], banks[k]->bytes,
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "copy an admitted expert");
        auto* table = static_cast<const void**>(banks[k]->table.p);
        require(cudaMemcpy(table + expert, &out.slot[k], sizeof(void*), cudaMemcpyHostToDevice) ==
                    cudaSuccess,
                "patch an admitted entry");
        if (out.victim >= 0) {
            require(cudaMemcpy(table + out.victim, &out.victim_host[k], sizeof(void*),
                               cudaMemcpyHostToDevice) == cudaSuccess,
                    "patch a victim entry");
        }
    }
    return out;
}

void loans(DeviceContext& device) {
    Model model;
    // Four slots a layer of the cache's own, and a buffer of four more a layer (and slack).
    ExpertCache cache(device, model.layers(device), std::vector<std::uint64_t>{4 * 16384});
    DeviceBuffer buffer(4 * 16384 + 1000);
    cudaStream_t transfer = nullptr;
    require(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking) == cudaSuccess, "a stream");
    const auto lent = cache.lend(std::vector<ExpertCache::Loan>{{.rank = 0, .base = buffer.p, .bytes = buffer.bytes}});
    require(lent.size() == 1 && lent[0] == 4 * 16384 && cache.stats().slots == 16,
            "a lent buffer gives each layer the same free slots: " + std::to_string(cache.stats().slots));
    const auto ranges = cache.storage(0);
    const auto inside = [&](const void* at) {
        const auto a = reinterpret_cast<std::uintptr_t>(at);
        return a >= reinterpret_cast<std::uintptr_t>(buffer.p) &&
               a < reinterpret_cast<std::uintptr_t>(buffer.p) + buffer.bytes;
    };
    require(ranges[2].second - ranges[2].first == 4 * 10496 && inside(reinterpret_cast<void*>(ranges[2].first)),
            "layer 0's lent block lies in the buffer");
    cache.hand_over(0);
    std::vector<unsigned char> in_use(kExperts, 0);
    // Experts 0..3 take the cache's own slots, 4..7 the lent ones; expert e routes 5(e + 1) times.
    for (std::int32_t e = 0; e < 8; ++e) {
        const auto out = admit(cache, model, 0, e, 1, in_use);
        require(out.slot[0] != nullptr && inside(out.slot[0]) == (e >= 4),
                "the cache's own free slots come before the lent ones");
    }
    std::vector<std::int32_t> routes;
    for (int e = 0; e < 8; ++e) { routes.insert(routes.end(), std::size_t(5 * (e + 1)), e); }
    cache.observe(0, routes, true);
    cache.adapt();
    require(model.cached(0, "lent and filled").size() == 8, "eight slots hold their experts");

    // A wide pass takes the buffer back: 4..7 move into the slots of 0..3, which leave.
    cache.reclaim(0, transfer);
    require(cudaStreamSynchronize(transfer) == cudaSuccess, "the handover");
    const auto kept = model.cached(0, "reclaimed");
    require(kept == std::vector<int>({4, 5, 6, 7}), "the most routed experts stay in the cache's own slots");
    for (std::int32_t e = 4; e < 8; ++e) {
        require(cache.cached(0, 0, e) != nullptr && !inside(cache.cached(0, 0, e)), "kept outside the buffer");
    }
    require(cudaMemsetAsync(buffer.p, 0xab, buffer.bytes, transfer) == cudaSuccess, "a wide pass writes it");
    require(cudaStreamSynchronize(transfer) == cudaSuccess, "the wide pass");
    require(model.cached(0, "written").size() == 4, "the buffer's writes reach no cached expert");
    const auto refused = admit(cache, model, 0, 20, 1, in_use);
    require(refused.slot[0] == nullptr || !inside(refused.slot[0]), "no admission into a reclaimed buffer");

    // The next decode pass returns it zeroed. Layer 0's returned slots take the experts that left
    // (0..3, the most routed outside the cache), which serve once their copies have landed.
    const auto admitted = cache.stats().admitted;
    cache.give_back(0, transfer);
    require(cudaDeviceSynchronize() == cudaSuccess, "the zeroing and the copies");
    require(model.cached(0, "refilling") == std::vector<int>({4, 5, 6, 7}),
            "a refilled expert's entries stay on the host until its copy has landed");
    cache.adapt();
    const auto refilled = model.cached(0, "refilled");
    require(refilled.size() == 8 && holds(refilled, 0) && holds(refilled, 3), "the returned slots are refilled");
    for (std::int32_t e = 0; e < 4; ++e) {
        require(inside(cache.cached(0, 0, e)), "a refilled expert lies in a returned slot");
    }
    require(cache.stats().admitted == admitted + 4, "the refill copies four experts");
    // Layer 1 routed nothing: its returned slots stay free for the admissions past its own.
    cache.hand_over(1);
    for (std::int32_t e = 0; e < 5; ++e) {
        const auto out = admit(cache, model, 1, 9 + e, 1, in_use);
        require(out.slot[0] != nullptr && inside(out.slot[0]) == (e == 4),
                "a returned slot takes a miss after the cache's own free slots");
    }
    require(model.cached(1, "returned").size() == 5,
            "returned slots hold their experts, followed by zeros");
    cache.give_back(0, transfer);
    require(cache.stats().slots == 16, "a second return changes nothing");
    cudaStreamDestroy(transfer);
}

void admissions(DeviceContext& device) {
    Model model;
    ExpertCache cache(device, model.layers(device), std::vector<std::uint64_t>{2 * 102400});
    // Layer 1 admits from now on: the caller copies the expert in and patches the tables, as the
    // expert misses do.
    cache.hand_over(1);
    const auto admit = [&](std::int32_t expert, int pairs, const std::vector<unsigned char>& in_use) {
        return ::admit(cache, model, 1, expert, pairs, in_use);
    };
    std::vector<unsigned char> in_use(kExperts, 0);
    // Free slots take any miss.
    for (std::int32_t e = 0; e < 12; ++e) {
        const auto out = admit(e, 1, in_use);
        require(out.slot[0] != nullptr && out.victim < 0, "a free slot takes a miss");
    }
    require(model.cached(1, "filled").size() == 12, "every free slot holds its expert");
    const std::vector<std::int32_t> zeros(10, 0);
    const auto before = cache.stats();
    cache.observe(1, zeros, true);
    require(cache.stats().hits == before.hits, "an admitted expert's routes in its call are misses");
    // Expert e of 1..11 routed e times, expert 0 ten more: coldest first 1, 2, 3, ...
    std::vector<std::int32_t> spread(zeros);
    for (int e = 1; e < 12; ++e) { spread.insert(spread.end(), std::size_t(e), e); }
    cache.observe(1, spread, true);
    require(cache.stats().hits == before.hits + spread.size(), "afterwards they are hits");
    cache.adapt(); // a pass ends: the tier ranks the held experts

    const auto refused = admit(40, 2, in_use);
    require(refused.slot[0] == nullptr && cache.cached(1, 0, 40) == nullptr,
            "a newcomer that does not beat the coldest expert by the gain stays out");
    in_use[1] = 1;
    const auto taken = admit(41, 4, in_use);
    require(taken.slot[0] != nullptr && taken.victim == 2,
            "the coldest expert the call does not route leaves");
    require(cache.cached(1, 0, 41) == taken.slot[0] && cache.cached(1, 0, 2) == nullptr,
            "the newcomer holds the victim's slot");
    const auto next = admit(42, 10, in_use);
    require(next.victim == 3, "each victim leaves once; the next coldest follows");
    const auto cached = model.cached(1, "admitted");
    require(cached.size() == 12 && holds(cached, 41) && holds(cached, 42) && !holds(cached, 2) &&
                !holds(cached, 3),
            "admitted slots hold their experts");

    // Growth after startup: both layers gain the same number of free slots in a second block,
    // which the next admissions fill.
    const auto ranges_before = cache.storage(1);
    require(ranges_before[1].first == ranges_before[1].second, "no second block before growth");
    const auto added = cache.grow(std::vector<std::uint64_t>{4 * 16384});
    require(added.size() == 1 && added[0] == 4 * 16384 && cache.stats().slots == 24 + 8,
            "growth gives each growing layer the same slots: " + std::to_string(cache.stats().slots));
    const auto ranges_after = cache.storage(1);
    require(ranges_after[1].second - ranges_after[1].first == 4 * 5888, "layer 1's second block");
    std::fill(in_use.begin(), in_use.end(), 0);
    const auto grown = admit(43, 1, in_use);
    const auto at    = reinterpret_cast<std::uintptr_t>(grown.slot[0]);
    require(grown.victim < 0 && at >= ranges_after[1].first && at < ranges_after[1].second,
            "a grown layer admits into a free slot of its second block");
    require(model.cached(1, "grown").size() == 13, "the grown layer stays consistent");
}

void profile(DeviceContext& device) {
    Model model;
    // Layer 0 routed experts 0..7 a hundred times each, layer 1 experts 0..31 fifty times: the
    // ranking gives layer 0 its eight slots (83,968 bytes) and layer 1 twenty of the rest.
    std::vector<std::vector<std::uint64_t>> counts(2, std::vector<std::uint64_t>(kExperts, 0));
    for (int e = 0; e < 8; ++e) { counts[0][std::size_t(e)] = 100; }
    for (int e = 0; e < 32; ++e) { counts[1][std::size_t(e)] = 50; }
    ExpertCache cache(device, model.layers(device), std::vector<std::uint64_t>{2 * 102400}, &counts);
    require(cache.stats().slots == 8 + 20,
            "a profile sizes each layer's share: " + std::to_string(cache.stats().slots));
    cache.seed(counts);
    const auto zero = model.cached(0, "seeded 0");
    const auto one  = model.cached(1, "seeded 1");
    require(zero.size() == 8 && holds(zero, 0) && holds(zero, 7), "seeding fills layer 0 by its counts");
    require(one.size() == 20 && holds(one, 0) && holds(one, 19) && !holds(one, 20),
            "seeding fills layer 1 with its most routed experts");
    const auto stats = cache.stats();
    require(stats.admitted == 28 && stats.copied_bytes == 8 * 10240 + 20 * 5632,
            "seeding counts its copies");
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        DeviceContext device;
        rounds(device);
        paced(device);
        admissions(device);
        profile(device);
        loans(device);
        std::cout << "PASS qwen4_exp expert cache\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
