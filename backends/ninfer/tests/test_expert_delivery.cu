#include "core/arena.h"
#include "core/device.h"
#include "core/expert_delivery.h"
#include "core/vmm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
namespace {
constexpr int kLayers = 4, kExperts = 16, kPairs = 2048, kTail = 512;
constexpr std::uint64_t kGu = 131072, kDown = 98304;

__host__ __device__ unsigned char signature(int layer, int expert, int blob, std::uint64_t at) {
    auto x = std::uint32_t(at) * 0x27d4eb2du ^ std::uint32_t(layer + 1) * 0x9e3779b1u ^
             std::uint32_t(expert + 1) * 0x85ebca77u ^ std::uint32_t(blob + 1) * 0xc2b2ae3du;
    x ^= x >> 15;
    return static_cast<unsigned char>(x);
}

// Layer l's down blobs are `step` bytes shorter than layer l - 1's (Fixture).
__global__ void inspect(const void* const* table, const std::int32_t* ids, int pairs, int layer,
                        unsigned long long* failures, unsigned long long* visited,
                        std::uint64_t step = 256) {
    const int p = blockIdx.x;
    if (p >= pairs || ids[p] < 0) { return; }
    const int expert = ids[p];
    if (threadIdx.x == 0) { atomicAdd(visited, 1ull); }
    for (int b = 0; b < 2; ++b) {
        const auto* data = static_cast<const unsigned char*>(table[(b ? 3 : 2) * kExperts + expert]);
        if (!data) { if (!threadIdx.x) { atomicAdd(failures, 1ull); } continue; }
        const auto bytes = b ? kDown - std::uint64_t(layer) * step : kGu;
        // Every represented byte, its last byte, and every tail byte.
        for (std::uint64_t at = threadIdx.x; at < bytes; at += blockDim.x) {
            if (data[at] != signature(layer, expert, b, at)) { atomicAdd(failures, 1ull); }
        }
        if (data[bytes - 1] != signature(layer, expert, b, bytes - 1)) { atomicAdd(failures, 1ull); }
        // A kernel may read kTail bytes past a blob: zeros, or in a layer bank the next expert of
        // the same layer and blob, laid out as in a resident bank.
        for (int at = threadIdx.x; at < kTail; at += blockDim.x) {
            const unsigned char value = data[bytes + at];
            if (value && (expert + 1 == kExperts || value != signature(layer, expert + 1, b, at))) {
                atomicAdd(failures, 1ull);
            }
        }
        if (!b && !threadIdx.x) {
            if (table[expert] != data || table[kExperts + expert] != data + kGu / 2) {
                atomicAdd(failures, 1ull);
            }
            if (reinterpret_cast<std::uintptr_t>(data) % 256) { atomicAdd(failures, 1ull); }
        }
    }
}

// A set's list (its count, then its experts ascending) against its pairs: exactly the experts some
// pair of the set routes to.
__global__ void listed(const std::int32_t* set, int pairs, const std::int32_t* list,
                       unsigned long long* failures) {
    __shared__ int routed[kExperts];
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { routed[e] = 0; }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        if (set[p] >= 0 && set[p] < kExperts) { routed[set[p]] = 1; }
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }
    int count = 0;
    for (int e = 0; e < kExperts; ++e) {
        if (!routed[e]) { continue; }
        if (count >= list[0] || list[1 + count] != e) {
            atomicAdd(failures, 1ull);
            return;
        }
        ++count;
    }
    if (count != list[0]) { atomicAdd(failures, 1ull); }
}

// Four layers of sixteen experts, each a [gate; up] blob and a down blob whose layer l is `step`
// bytes shorter than layer l - 1's: with the default step every layer's slots need their own room.
struct Fixture {
    std::vector<std::unique_ptr<PinnedHostBuffer>> pinned;
    std::vector<ExpertDeliveryLayer> layers;
    std::vector<std::filesystem::path> paths;

    explicit Fixture(bool disk, const std::filesystem::path& directory, std::uint64_t step = 256) {
        std::array<std::ofstream, 2> output;
        if (disk) {
            for (int i = 0; i < 2; ++i) {
                paths.push_back(directory / ("part-" + std::to_string(i)));
                output[i].open(paths[i], std::ios::binary);
                if (!output[i]) { throw std::runtime_error("fixture open failed"); }
            }
        }
        std::uint64_t file_offsets[2]{};
        for (int l = 0; l < kLayers; ++l) {
            ExpertDeliveryLayer layer;
            layer.aliases = {{0, 0}, {0, kGu / 2}, {0, 0}, {1, 0}};
            // Host blobs of one layer and blob index share an allocation, experts in order, as a
            // model's pinned parent does: runs of them copy as one.
            std::array<std::byte*, 2> host{};
            if (!disk) {
                for (int b = 0; b < 2; ++b) {
                    pinned.push_back(std::make_unique<PinnedHostBuffer>(
                        kExperts * (b ? kDown - std::uint64_t(l) * step : kGu)));
                    host[b] = static_cast<std::byte*>(pinned.back()->data());
                }
            }
            for (int e = 0; e < kExperts; ++e) {
                std::vector<ExpertBlob> expert;
                for (int b = 0; b < 2; ++b) {
                    ExpertBlob blob;
                    blob.bytes     = b ? kDown - std::uint64_t(l) * step : kGu;
                    blob.zero_tail = kTail;
                    std::vector<unsigned char> data(blob.bytes);
                    for (std::uint64_t i = 0; i < blob.bytes; ++i) { data[i] = signature(l, e, b, i); }
                    if (disk) {
                        const auto split = blob.bytes / 3;
                        output[0].write(reinterpret_cast<const char*>(data.data()), split);
                        output[1].write(reinterpret_cast<const char*>(data.data() + split), blob.bytes - split);
                        blob.extents = {{0, file_offsets[0], split},
                                        {1, file_offsets[1], blob.bytes - split}};
                        file_offsets[0] += split;
                        file_offsets[1] += blob.bytes - split;
                    } else {
                        auto* target = host[b] + std::uint64_t(e) * blob.bytes;
                        std::memcpy(target, data.data(), data.size());
                        blob.pinned = target;
                        blob.pinned_region = reinterpret_cast<std::uintptr_t>(host[b]);
                    }
                    expert.push_back(std::move(blob));
                }
                layer.experts.push_back(std::move(expert));
            }
            layers.push_back(std::move(layer));
        }
    }
};

int run(DeviceContext& device, bool disk, std::uint64_t cache, const std::filesystem::path& path,
        bool shifting = false) {
    Fixture fixture(disk, path);
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes = cache;
    options.staging_bytes = 4096; // exercise repeated DMA half reuse inside one blob
    options.suspendable = true;
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8);
    failures.fill();
    visited.fill();
    const auto pass = [&] {
        for (int l = 0; l < kLayers; ++l) {
            const auto call = delivery.begin(l, static_cast<const std::int32_t*>(ids.p), kPairs, device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.cached_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            listed<<<1, 256, 0, device.stream>>>(call.cached_ids, kPairs, call.cached_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            delivery.await(device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.missing_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            listed<<<1, 256, 0, device.stream>>>(call.missing_ids, kPairs, call.missing_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            CUDA_CHECK(cudaGetLastError());
        }
    };
    std::vector<std::int32_t> routes(kPairs);
    for (int i = 0; i < kPairs; ++i) { routes[i] = i % 5; }
    ids.copy_from_host(routes.data(), ids.bytes);
    pass(); // load all kernels before a captured GPU wait can run
    delivery.drain();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
    pass();
    CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    constexpr int repeats = 12;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        for (int p = 0; p < kPairs; ++p) {
            if (shifting) {
                // Narrow calls whose five experts move by three every replay, and a wide call every
                // fourth: once the cache is full, admissions and the tier's rounds evict experts of
                // every layer while the GPU reads the next call's cached pairs.
                routes[p] = iteration % 4 == 3 ? p % kExperts : (3 * iteration + p % 5) % kExperts;
            } else {
                routes[p] = iteration % 3 == 0 ? p % 5 : (p + iteration) % kExperts;
            }
        }
        ids.copy_from_host(routes.data(), ids.bytes);
        CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
        delivery.drain();
        if (iteration == 5) {
            const auto* addresses = delivery.tables(0);
            const auto bytes = delivery.state_backing_bytes();
            if (!bytes) { throw std::runtime_error("suspend did not account device backing"); }
            delivery.release_device_backing();
            delivery.restore_device_backing();
            if (delivery.tables(0) != addresses) {
                throw std::runtime_error("suspend changed graph table address");
            }
        }
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    unsigned long long bad = 0, count = 0;
    failures.copy_to_host(&bad, 8);
    visited.copy_to_host(&count, 8);
    const auto stats = delivery.stats();
    const auto memory = delivery.memory();
    // Two of three replays route all sixteen experts of every layer (wide calls): without a cache
    // the next layer's bank is prefetched and reused; with one, wide calls promote by device copies.
    const bool ok = !bad && count == std::uint64_t(repeats + 1) * kLayers * kPairs &&
                    stats.routes == stats.hits + stats.misses && stats.misses &&
                    (cache ? stats.hits != 0 : stats.hits == 0) &&
                    (disk ? stats.read_bytes == stats.copied_bytes : stats.read_bytes == 0) &&
                    (cache ? memory.cache_bytes != 0 : memory.cache_bytes == 0) &&
                    memory.banks == 2 &&
                    (cache ? shifting || stats.promoted_bytes != 0
                           : stats.prefetched_layers != 0 && stats.bank_reuses != 0) &&
                    (!shifting || stats.evictions != 0);
    std::printf("%s%s cache=%llu bad=%llu pairs=%llu hits=%llu misses=%llu evictions=%llu "
                "admissions=%llu bank_reuses=%llu prefetched=%llu read=%llu copied=%llu "
                "promoted=%llu banks=%u %s\n", disk ? "disk" : "host", shifting ? " shifting" : "",
                (unsigned long long)cache, bad, count, (unsigned long long)stats.hits,
                (unsigned long long)stats.misses, (unsigned long long)stats.evictions,
                (unsigned long long)stats.admissions, (unsigned long long)stats.bank_reuses,
                (unsigned long long)stats.prefetched_layers, (unsigned long long)stats.read_bytes,
                (unsigned long long)stats.copied_bytes, (unsigned long long)stats.promoted_bytes,
                memory.banks, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int failure_control(DeviceContext& device, const std::filesystem::path& path, bool omit_wait) {
    Fixture fixture(true, path);
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes = 0;
    options.staging_bytes = 4096;
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8);
    failures.fill();
    visited.fill();
    std::vector<std::int32_t> routes(kPairs, -1);
    ids.copy_from_host(routes.data(), ids.bytes);
    // Load inspect before an asynchronous transfer can start.
    inspect<<<1, 32, 0, device.stream>>>(delivery.tables(0), static_cast<const std::int32_t*>(ids.p),
        1, 0, static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::fill(routes.begin(), routes.end(), 0);
    ids.copy_from_host(routes.data(), ids.bytes);
    if (!omit_wait) { std::filesystem::resize_file(fixture.paths.front(), 0); }
    const auto call = delivery.begin(0, static_cast<const std::int32_t*>(ids.p), kPairs, device.stream);
    if (omit_wait) {
        inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(0), call.missing_ids, kPairs, 0,
            static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
        delivery.await(device.stream);
        delivery.drain();
        unsigned long long bad = 0;
        failures.copy_to_host(&bad, 8);
        std::printf("positive control without ready dependency: %llu mismatches\n", bad);
        return bad ? 1 : 0; // this intentionally broken execution must fail
    }
    try { delivery.await(device.stream); }
    catch (const std::exception&) {
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        std::puts("truncated source: error propagated and GPU wait released PASS");
        return 0;
    }
    std::puts("FAIL truncated source was accepted");
    return 1;
}

int admission_boundaries(DeviceContext& device) {
    PinnedHostBuffer source(256);
    std::memset(source.data(), 0, source.size());
    ExpertDeliveryLayer layer;
    layer.experts = {{ExpertBlob{.pinned = source.data(), .bytes = 256}}};
    layer.aliases = {{0, 0}};
    for (const bool suspendable : {false, true}) {
        ExpertDeliveryOptions options;
        options.pair_capacity = 1;
        options.suspendable = suspendable;
        options.banks = 1;
        // Every miss needs a layer bank: with nothing available it is refused before allocation.
        options.keep_free_bytes = std::numeric_limits<std::uint64_t>::max();
        options.cache_bytes = 0;
        bool rejected = false;
        try { ExpertDelivery delivery(device.rank(0), {}, {layer}, options); }
        catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()).find("expert delivery: a layer bank needs ") == 0;
        }
        if (!rejected) { throw std::runtime_error("a layer bank consumed the reserved budget"); }
        // The exact zero-cache boundary remains valid, both explicitly and automatically.
        options.keep_free_bytes = 0;
        std::uint64_t mandatory_backing = 0;
        {
            ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
            if (delivery.memory().cache_bytes || delivery.memory().banks != 1) {
                throw std::runtime_error("zero cache allocated backing");
            }
            mandatory_backing = delivery.state_backing_bytes();
        }
        // With available room, report the physical size of a sub-granule slot, not its 256-byte
        // logical size.
        options.cache_bytes = 256;
        {
            ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
            const auto granule = suspendable ? vmm::granularity(device.device) : 1;
            const auto expected = (256 + granule - 1) / granule * granule;
            if (delivery.memory().cache_bytes != expected || delivery.memory().cache_slots != 1) {
                throw std::runtime_error("sub-granule cache did not report its physical backing");
            }
        }
#ifndef _WIN32
        // Linux pinned buffers do not consume device backing. Use the observed mandatory
        // footprint, without assuming its allocation layout, to leave 256 bytes for the cache.
        // The logical slot fits; its VMM granule must be refused, and auto must choose zero.
        if (suspendable) {
            const auto granule = vmm::granularity(device.device);
            if (granule > 256) {
                std::size_t free = 0, total = 0;
                CUDA_CHECK(cudaMemGetInfo(&free, &total));
                if (free <= mandatory_backing + granule) {
                    throw std::runtime_error("insufficient free memory for cache admission boundary test");
                }
                options.keep_free_bytes = free - mandatory_backing - 256;
                rejected = false;
                try { ExpertDelivery delivery(device.rank(0), {}, {layer}, options); }
                catch (const std::invalid_argument& error) {
                    rejected = std::string(error.what()).find("but only 256 remain") != std::string::npos;
                    if (!rejected) { throw std::runtime_error(error.what()); }
                }
                if (!rejected) { throw std::runtime_error("VMM cache rounding consumed the reserve"); }
                CUDA_CHECK(cudaMemGetInfo(&free, &total));
                options.keep_free_bytes = free - mandatory_backing - (granule - 1);
                options.cache_bytes.reset();
                ExpertDelivery delivery(device.rank(0), {}, {layer}, options);
                if (delivery.memory().cache_bytes) {
                    throw std::runtime_error("auto cache rounded past its physical remainder");
                }
            }
        }
#endif
        std::printf("cache admission suspendable=%d bank reject, zero/auto accept, "
                    "physical rounding PASS\n", int(suspendable));
    }
    return 0;
}

// Two layers of twelve experts with a four-slot cache, every call narrow: misses fill the free
// slots; once they are full, a miss enters only when its routes beat a cold expert its call does
// not route by the tier's gain.
int tier_policy(DeviceContext& device) {
    constexpr int experts = 12, slots = 4;
    std::vector<std::unique_ptr<PinnedHostBuffer>> sources;
    std::vector<ExpertDeliveryLayer> layers(2);
    for (auto& layer : layers) {
        layer.aliases = {{0, 0}};
        sources.push_back(std::make_unique<PinnedHostBuffer>(std::size_t(experts) * 256));
        std::memset(sources.back()->data(), 0, sources.back()->size());
        for (int e = 0; e < experts; ++e) {
            auto* data = static_cast<std::byte*>(sources.back()->data()) + std::size_t(e) * 256;
            layer.experts.push_back({ExpertBlob{.pinned = data,
                .pinned_region = reinterpret_cast<std::uintptr_t>(sources.back()->data()), .bytes = 256}});
        }
    }
    ExpertDeliveryOptions options;
    options.pair_capacity = 64;
    options.cache_bytes = slots * 256;
    options.tier.every = 1; // a round at every pass
    ExpertDelivery delivery(device.rank(0), {}, layers, options);
    if (delivery.memory().cache_slots != slots) { throw std::runtime_error("tier: slot count"); }
    DeviceBuffer ids(64 * 4);
    const auto call = [&](int layer, const std::vector<std::int32_t>& routed) {
        ids.copy_from_host(routed.data(), routed.size() * 4);
        (void)delivery.begin(layer, static_cast<const std::int32_t*>(ids.p), std::uint32_t(routed.size()),
                             device.stream);
        delivery.await(device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        delivery.drain();
        return delivery.stats();
    };
    int failures = 0;
    const auto expect = [&](bool condition, const char* what) {
        if (!condition) {
            std::printf("tier policy: %s\n", what);
            ++failures;
        }
    };
    // Pass 1 fills the four slots.
    call(0, {0, 1});
    auto stats = call(1, {0, 1});
    expect(stats.admissions == 4 && stats.evictions == 0, "free slots take every miss");
    // Pass 2: one pair of a new expert does not beat a held expert's route by 1.5.
    call(0, {0, 1});
    stats = call(0, {2});
    expect(stats.evictions == 0 && stats.admissions == 4, "a weak miss stays out of a full cache");
    stats = call(1, {0, 1});
    // Pass 3: twenty pairs of expert 3 beat the coldest expert its call does not route.
    stats = call(0, std::vector<std::int32_t>(20, 3));
    expect(stats.evictions == 1 && stats.admissions == 5, "a strong miss takes the coldest slot");
    const auto after = call(0, std::vector<std::int32_t>(20, 3));
    expect(after.pair_hits - stats.pair_hits == 20, "the admitted expert hits");
    // The slots serve every layer: a strong miss of layer 1, whose call routes its own held
    // experts, takes the slot of layer 0's cold expert 1.
    stats = call(1, {0, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
    expect(stats.evictions == 2 && call(0, {1}).misses == stats.misses + 1,
           "a strong miss takes another layer's cold expert");
    std::printf("tier policy admissions=%llu evictions=%llu hits=%llu %s\n",
                (unsigned long long)stats.admissions, (unsigned long long)stats.evictions,
                (unsigned long long)stats.pair_hits, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

// The CPU's products partition a narrow call with the cache and the copies: every pair is cached,
// missing or computed on the CPU, which reads each expert's stored bytes (pinned or from the
// files); and a routing profile seeds the cache, so the first call of its experts hits.
class PairEcho final : public ExpertCpuProducts {
public:
    bool supports(std::size_t) const override { return true; }

    Done compute(const Work& work) override {
        if (work.side) { work.side(); }
        auto* out = reinterpret_cast<std::int32_t*>(work.products);
        const auto* inputs = reinterpret_cast<const std::int32_t*>(work.inputs);
        for (std::size_t p = 0; p < work.ids.size(); ++p) {
            out[p] = -1;
            if (inputs[p] != work.ids[p]) { ++bad; }
            for (std::size_t i = 0; i < work.experts.size(); ++i) {
                if (work.experts[i] == work.ids[p]) { out[p] = work.ids[p]; }
            }
        }
        for (std::size_t i = 0; i < work.experts.size(); ++i) {
            for (int b = 0; b < 2; ++b) {
                const auto bytes = b ? kDown - std::uint64_t(work.layer) * 256 : kGu;
                const auto* data = reinterpret_cast<const unsigned char*>(work.blobs[i][std::size_t(b)]);
                for (const std::uint64_t at : {std::uint64_t(0), bytes / 2, bytes - 1}) {
                    if (data[at] != signature(int(work.layer), work.experts[i], b, at)) { ++bad; }
                }
            }
        }
        ++calls;
        return {work.ids.size() * 4, 1.0};
    }
    std::uint64_t bad = 0, calls = 0;
};

__global__ void partition(const std::int32_t* ids, const std::int32_t* cached, const std::int32_t* missing,
                          const std::int32_t* products, const std::int32_t* cpu, int pairs,
                          unsigned long long* failures, unsigned long long* cpu_pairs) {
    const int p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= pairs) { return; }
    const int e = ids[p];
    const bool c = cached[p] >= 0, m = missing[p] >= 0, u = *cpu != 0 && products[p] >= 0;
    if (int(c) + int(m) + int(u) != 1 || (c && cached[p] != e) || (m && missing[p] != e) ||
        (u && products[p] != e)) {
        atomicAdd(failures, 1ull);
    }
    if (u) { atomicAdd(cpu_pairs, 1ull); }
}

// With every miss on the CPU, only the profile and the tier's rounds fill the cache.
int cpu_share(DeviceContext& device, bool disk, double share, const std::filesystem::path& path) {
    Fixture fixture(disk, path);
    PairEcho echo;
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes = 6ULL << 20;
    options.cpu.products = &echo;
    options.cpu.input_bytes = kPairs * 4;
    options.cpu.product_bytes = kPairs * 4;
    options.cpu.share = share;
    options.cpu.threads = 2;
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    // A profile of layer 0's experts 0..2 seeds the cache.
    std::vector<std::vector<std::uint64_t>> profile(kLayers, std::vector<std::uint64_t>(kExperts, 0));
    profile[0] = {9, 8, 7};
    profile[0].resize(kExperts, 0);
    delivery.prime(&profile);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8), cpu_pairs(8);
    failures.fill();
    visited.fill();
    cpu_pairs.fill();
    std::vector<std::int32_t> routes(kPairs);
    std::uint64_t seeded_hits = 0;
    for (int iteration = 0; iteration < 8; ++iteration) {
        // Narrow calls: six experts, one shifted each time.
        for (int p = 0; p < kPairs; ++p) { routes[p] = (iteration + p % 6) % kExperts; }
        ids.copy_from_host(routes.data(), ids.bytes);
        for (int l = 0; l < kLayers; ++l) {
            const auto before = delivery.stats();
            const ExpertDelivery::Input input{ids.p, std::size_t(kPairs) * 4};
            const auto call = delivery.begin(l, static_cast<const std::int32_t*>(ids.p), kPairs,
                                             device.stream, {&input, 1});
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.cached_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            delivery.await(device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.missing_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            // Without the CPU's experts: the list the host rewrote.
            listed<<<1, 256, 0, device.stream>>>(call.missing_ids, kPairs, call.missing_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            partition<<<kPairs / 256, 256, 0, device.stream>>>(
                static_cast<const std::int32_t*>(ids.p), call.cached_ids, call.missing_ids,
                static_cast<const std::int32_t*>(call.products), call.cpu, kPairs,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(cpu_pairs.p));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            delivery.drain();
            if (iteration == 0 && l == 0) { seeded_hits = delivery.stats().hits - before.hits; }
        }
    }
    unsigned long long bad = 0, computed = 0;
    failures.copy_to_host(&bad, 8);
    cpu_pairs.copy_to_host(&computed, 8);
    const auto stats = delivery.stats();
    // Seeded experts 0..2 of layer 0 serve six routed experts' calls; with every miss on the CPU,
    // more hits come only from slots a round filled.
    const bool ok = !bad && !echo.bad && echo.calls != 0 && computed == stats.cpu_pairs &&
                    stats.cpu_experts != 0 && seeded_hits == 3 &&
                    (share < 1 || stats.hits > 6) &&
                    delivery.cpu_description().find("2 ") == 0;
    std::printf("%s cpu share %.1f bad=%llu echo_bad=%llu cpu_experts=%llu cpu_pairs=%llu hits=%llu "
                "seeded_hits=%llu %s\n",
                disk ? "disk" : "host", share, bad, (unsigned long long)echo.bad,
                (unsigned long long)stats.cpu_experts, computed, (unsigned long long)stats.hits,
                (unsigned long long)seeded_hits,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// An elastic cache of one pool under a graph captured before it changes: narrow passes over
// experts 0..4 of every layer (four times) and 5..7 (once) fill it; giving back all but two
// granules keeps the most routed experts that fit; giving back everything leaves every expert to
// the banks; growing takes the slots back. Every replay reads every pair's bytes.
int elastic_cache(DeviceContext& device, bool disk, const std::filesystem::path& path) {
    // Every layer's slots alike, so the cache is one pool of interchangeable slots.
    Fixture fixture(disk, path, 0);
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes   = 64ULL << 20; // every expert
    options.staging_bytes = 4096;
    options.suspendable   = true;
    options.elastic       = true;
    options.lend_banks    = false; // the banks serve every miss the cache's slots do not hold
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    constexpr std::uint64_t slot = 131584 + 98816; // each blob and its tail, 256-byte aligned
    const std::uint64_t granule  = vmm::granularity(device.device);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8);
    failures.fill();
    visited.fill();
    const auto pass = [&] {
        for (int l = 0; l < kLayers; ++l) {
            const auto call = delivery.begin(l, static_cast<const std::int32_t*>(ids.p), kPairs, device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(
                delivery.tables(l), call.cached_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p),
                static_cast<unsigned long long*>(visited.p), 0);
            listed<<<1, 256, 0, device.stream>>>(call.cached_ids, kPairs, call.cached_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            delivery.await(device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(
                delivery.tables(l), call.missing_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p),
                static_cast<unsigned long long*>(visited.p), 0);
            listed<<<1, 256, 0, device.stream>>>(call.missing_ids, kPairs, call.missing_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            CUDA_CHECK(cudaGetLastError());
        }
    };
    std::vector<std::int32_t> routes(kPairs);
    const auto route = [&](int first, int count) {
        for (int p = 0; p < kPairs; ++p) { routes[p] = first + p % count; }
        ids.copy_from_host(routes.data(), ids.bytes);
    };
    route(0, 5);
    pass();
    delivery.drain();
    delivery.prime(nullptr);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
    pass();
    CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    int replays = 0;
    const auto replay = [&](int first, int count) {
        route(first, count);
        const auto before = delivery.stats();
        CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
        delivery.drain();
        ++replays;
        return delivery.stats() - before;
    };
    int bad_steps = 0;
    const auto expect = [&](bool condition, const char* what) {
        if (!condition) {
            std::printf("elastic cache: %s\n", what);
            ++bad_steps;
        }
    };
    for (int i = 0; i < 4; ++i) { replay(0, 5); }
    replay(5, 3);
    const auto full = delivery.memory();
    expect(full.cache_slots == kLayers * kExperts, "the cache holds every expert");
    // All but two granules go back.
    const auto freed = delivery.shrink_cache(full.cache_bytes - 2 * granule);
    const auto kept = delivery.memory();
    const auto keep = std::min<std::uint64_t>(full.cache_slots, 2 * granule / slot);
    expect(freed == full.cache_bytes - 2 * granule && kept.cache_bytes == 2 * granule &&
           kept.cache_slots == keep, "shrinking keeps the slots of two granules");
    // The hottest experts stayed: experts 0..4 of the four layers, as many as fit.
    const auto hot = replay(0, 5);
    expect(hot.hits == std::min<std::uint64_t>(keep, kLayers * 5), "the most routed experts stay");
    delivery.shrink_cache(std::uint64_t(1) << 40);
    expect(delivery.memory().cache_slots == 0 && delivery.memory().cache_bytes == 0,
           "everything goes back");
    const auto empty = replay(0, 5);
    expect(empty.hits == 0 && empty.misses == kLayers * 5, "an empty cache serves from the banks");
    const auto grown = delivery.grow_cache();
    expect(grown != 0 && delivery.memory().cache_slots == kLayers * kExperts, "growing takes it back");
    replay(0, 5);
    const auto again = replay(0, 5);
    expect(again.hits == kLayers * 5, "the grown slots fill and hit");
    const auto* addresses = delivery.tables(0);
    delivery.release_device_backing();
    delivery.restore_device_backing();
    expect(delivery.tables(0) == addresses && delivery.memory().cache_slots == kLayers * kExperts,
           "suspend keeps the grown cache");
    replay(5, 3);
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    unsigned long long bad = 0, count = 0;
    failures.copy_to_host(&bad, 8);
    visited.copy_to_host(&count, 8);
    const bool ok = !bad_steps && !bad && count == std::uint64_t(replays + 1) * kLayers * kPairs;
    std::printf("%s elastic cache keep=%llu hot_hits=%llu bad=%llu pairs=%llu %s\n",
                disk ? "disk" : "host", (unsigned long long)keep, (unsigned long long)hot.hits, bad,
                count, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
// Lent banks under a captured graph, with no cache: narrow passes over five experts of every layer
// lend both banks after two passes, and their misses then stay in lent slots and hit; a wide pass
// takes the banks back, reading every expert's bytes; narrow passes lend them again. Every replay
// reads every pair's bytes.
int lent_banks(DeviceContext& device, bool disk, const std::filesystem::path& path) {
    Fixture fixture(disk, path);
    ExpertDeliveryOptions options;
    options.pair_capacity = kPairs;
    options.cache_bytes   = 0;
    options.staging_bytes = 4096;
    options.suspendable   = true;
    ExpertDelivery delivery(device.rank(0), fixture.paths, std::move(fixture.layers), options);
    DeviceBuffer ids(kPairs * 4), failures(8), visited(8);
    failures.fill();
    visited.fill();
    const auto pass = [&] {
        for (int l = 0; l < kLayers; ++l) {
            const auto call = delivery.begin(l, static_cast<const std::int32_t*>(ids.p), kPairs, device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.cached_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            listed<<<1, 256, 0, device.stream>>>(call.cached_ids, kPairs, call.cached_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            delivery.await(device.stream);
            inspect<<<kPairs, 32, 0, device.stream>>>(delivery.tables(l), call.missing_ids, kPairs, l,
                static_cast<unsigned long long*>(failures.p), static_cast<unsigned long long*>(visited.p));
            listed<<<1, 256, 0, device.stream>>>(call.missing_ids, kPairs, call.missing_experts,
                                                 static_cast<unsigned long long*>(failures.p));
            CUDA_CHECK(cudaGetLastError());
        }
    };
    std::vector<std::int32_t> routes(kPairs);
    const auto route = [&](int first, int count) {
        for (int p = 0; p < kPairs; ++p) { routes[p] = (first + p % count) % kExperts; }
        ids.copy_from_host(routes.data(), ids.bytes);
    };
    route(0, 5);
    pass();
    delivery.drain();
    delivery.prime(nullptr);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
    pass();
    CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    int replays = 0;
    const auto replay = [&](int first, int count) {
        route(first, count);
        const auto before = delivery.stats();
        CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
        delivery.drain();
        ++replays;
        return delivery.stats() - before;
    };
    int bad_steps = 0;
    const auto expect = [&](bool condition, const char* what) {
        if (!condition) {
            std::printf("lent banks: %s\n", what);
            ++bad_steps;
        }
    };
    replay(0, 5);
    replay(0, 5);
    expect(delivery.memory().lent_slots == 0, "two narrow passes lend nothing");
    const auto lending = replay(0, 5);
    const auto lent = delivery.memory().lent_slots;
    expect(lent >= kExperts && lending.lent_admissions == kLayers * 5,
           "the third narrow pass lends both banks and its misses enter them");
    const auto held = replay(0, 5);
    expect(held.hits == kLayers * 5 && held.misses == 0, "lent slots hit");
    const auto moved = replay(3, 5);
    expect(moved.misses == kLayers * 3 && moved.lent_admissions == kLayers * 3,
           "new experts enter free lent slots");
    const auto wide = replay(0, kExperts);
    expect(wide.hits == 0 && delivery.memory().lent_slots == 0 && wide.lent_evictions != 0,
           "a wide pass reads lent slots as misses and takes the banks back");
    replay(0, 5);
    replay(0, 5);
    const auto again = replay(0, 5);
    expect(delivery.memory().lent_slots == lent && again.hits == kLayers * 5,
           "narrow passes lend the banks again");
    const auto* addresses = delivery.tables(0);
    delivery.release_device_backing();
    delivery.restore_device_backing();
    expect(delivery.tables(0) == addresses && delivery.memory().lent_slots == 0,
           "suspend returns the banks");
    replay(5, 5);
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    unsigned long long bad = 0, count = 0;
    failures.copy_to_host(&bad, 8);
    visited.copy_to_host(&count, 8);
    const auto stats = delivery.stats();
    const bool ok = !bad_steps && !bad && count == std::uint64_t(replays + 1) * kLayers * kPairs;
    std::printf("%s lent banks slots=%u lent_admissions=%llu lent_evictions=%llu bad=%llu pairs=%llu %s\n",
                disk ? "disk" : "host", lent, (unsigned long long)stats.lent_admissions,
                (unsigned long long)stats.lent_evictions, bad, count, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Every pair's expert at its table entry: its first, middle and last byte of blob 0.
__global__ void sampled(const void* const* table, const std::int32_t* ids, int pairs, int layer,
                        std::uint64_t bytes, unsigned long long* failures) {
    const int p = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= pairs || ids[p] < 0) { return; }
    const auto* data = static_cast<const unsigned char*>(table[ids[p]]);
    if (!data) {
        atomicAdd(failures, 1ull);
        return;
    }
    const std::uint64_t at[3]{0, bytes / 2, bytes - 1};
    for (const auto a : at) {
        if (data[a] != signature(layer, ids[p], 0, a)) { atomicAdd(failures, 1ull); }
    }
}

// Layers of two sizes, 64 KiB experts in layers 0 and 1 and 32 KiB ones in layers 2 and 3, 64
// each: the cache gives every layer as many slots, each its own size, and an expert enters only a
// slot of its size. An explicit cache of eight slots per layer holds 32 slots where slots of the
// largest expert would hold 24; once every layer's eight experts fill them, a strong miss of layer
// 2 takes a slot of its own pool and layers 0 and 1 keep theirs. An elastic cache of every expert
// gives back at least what is asked and keeps each layer's routed experts; it gives back
// everything and takes it again. Every pair reads its expert's bytes.
int slot_pools(DeviceContext& device) {
    constexpr int layer_count = 4, experts = 64;
    const auto size = [](int layer) { return std::uint64_t(layer < 2 ? 65536 : 32768); };
    std::vector<std::unique_ptr<PinnedHostBuffer>> sources;
    std::vector<ExpertDeliveryLayer> layers(layer_count);
    for (int l = 0; l < layer_count; ++l) {
        auto& layer   = layers[std::size_t(l)];
        layer.aliases = {{0, 0}};
        sources.push_back(std::make_unique<PinnedHostBuffer>(std::size_t(experts * size(l))));
        auto* base = static_cast<std::byte*>(sources.back()->data());
        for (int e = 0; e < experts; ++e) {
            auto* data = base + std::uint64_t(e) * size(l);
            for (std::uint64_t at = 0; at < size(l); ++at) {
                data[at] = std::byte(signature(l, e, 0, at));
            }
            layer.experts.push_back(
                {ExpertBlob{.pinned        = data,
                            .pinned_region = reinterpret_cast<std::uintptr_t>(base),
                            .bytes         = size(l)}});
        }
    }
    int failures      = 0;
    const auto expect = [&](bool condition, const char* what) {
        if (!condition) {
            std::printf("slot pools: %s\n", what);
            ++failures;
        }
    };
    DeviceBuffer ids(64 * 4), bad(8);
    bad.fill();
    const auto run_call = [&](ExpertDelivery& delivery, int layer,
                              const std::vector<std::int32_t>& routed) {
        ids.copy_from_host(routed.data(), routed.size() * 4);
        const auto before = delivery.stats();
        const auto call =
            delivery.begin(std::size_t(layer), static_cast<const std::int32_t*>(ids.p),
                           std::uint32_t(routed.size()), device.stream);
        sampled<<<1, 64, 0, device.stream>>>(delivery.tables(std::size_t(layer)), call.cached_ids,
                                             int(routed.size()), layer, size(layer),
                                             static_cast<unsigned long long*>(bad.p));
        delivery.await(device.stream);
        sampled<<<1, 64, 0, device.stream>>>(delivery.tables(std::size_t(layer)), call.missing_ids,
                                             int(routed.size()), layer, size(layer),
                                             static_cast<unsigned long long*>(bad.p));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        delivery.drain();
        return delivery.stats() - before;
    };
    const std::vector<std::int32_t> eight{0, 1, 2, 3, 4, 5, 6, 7};
    const auto pass = [&](ExpertDelivery& delivery) {
        ExpertDeliveryStats total;
        for (int l = 0; l < layer_count; ++l) { total = total + run_call(delivery, l, eight); }
        return total;
    };
    {
        ExpertDeliveryOptions options;
        options.pair_capacity = 64;
        options.cache_bytes   = 2 * 8 * size(0) + 2 * 8 * size(2);
        options.lend_banks    = false;
        options.tier.every    = 0; // admissions only
        ExpertDelivery delivery(device.rank(0), {}, layers, options);
        expect(delivery.memory().cache_slots == 32, "eight slots of its own size for every layer");
        expect(pass(delivery).admissions == 32, "every layer's eight experts enter free slots");
        expect(pass(delivery).hits == 32, "and hit");
        const auto strong = run_call(delivery, 2, std::vector<std::int32_t>(20, 9));
        expect(strong.admissions == 1 && strong.evictions == 1, "a strong miss takes a slot");
        const auto rest = pass(delivery);
        expect(rest.hits == 31 && rest.misses == 1, "it took one of its own size");
    }
    {
        ExpertDeliveryOptions options;
        options.pair_capacity = 64;
        options.cache_bytes   = 64ULL << 20; // every expert
        options.suspendable   = true;
        options.elastic       = true;
        options.lend_banks    = false;
        ExpertDelivery delivery(device.rank(0), {}, layers, options);
        delivery.prime(nullptr);
        const auto full = delivery.memory();
        expect(full.cache_slots == layer_count * experts, "the elastic cache holds every expert");
        pass(delivery);
        expect(pass(delivery).hits == 32, "the routed experts hit");
        const std::uint64_t ask = 4ULL << 20;
        const auto freed        = delivery.shrink_cache(ask);
        const auto kept         = delivery.memory();
        expect(freed >= ask && kept.cache_bytes == full.cache_bytes - freed &&
                   kept.cache_slots < full.cache_slots,
               "shrinking gives back what is asked");
        expect(pass(delivery).hits == 32, "every layer keeps its routed experts");
        delivery.shrink_cache(std::uint64_t(1) << 40);
        expect(delivery.memory().cache_slots == 0 && delivery.memory().cache_bytes == 0,
               "everything goes back");
        expect(pass(delivery).hits == 0, "an empty cache serves from the banks");
        expect(delivery.grow_cache() == full.cache_bytes &&
                   delivery.memory().cache_slots == full.cache_slots,
               "growing takes it back");
        pass(delivery);
        expect(pass(delivery).hits == 32, "the grown slots fill and hit");
    }
    unsigned long long wrong = 0;
    bad.copy_to_host(&wrong, 8);
    expect(wrong == 0, "every pair reads its expert");
    std::printf("slot pools bad=%llu %s\n", wrong, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        DeviceContext device(0);
        if (argc == 2 && std::string(argv[1]) == "--admission-only") {
            return admission_boundaries(device);
        }
        const auto path = std::filesystem::temp_directory_path() /
            ("ninfer-expert-delivery-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
        struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{path};
        if (argc == 2 && std::string(argv[1]) == "--no-ready-wait") {
            return failure_control(device, path, true);
        }
        int failures = admission_boundaries(device);
        failures += tier_policy(device);
        failures += slot_pools(device);
        for (bool disk : {false, true}) {
            for (auto cache : {0ULL, 6ULL << 20, 64ULL << 20}) {
                failures += run(device, disk, cache, path);
            }
            failures += run(device, disk, 6ULL << 20, path, true);
            for (const double share : {0.5, 1.0}) { failures += cpu_share(device, disk, share, path); }
            failures += elastic_cache(device, disk, path);
            failures += lent_banks(device, disk, path);
        }
        failures += failure_control(device, path, false);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
