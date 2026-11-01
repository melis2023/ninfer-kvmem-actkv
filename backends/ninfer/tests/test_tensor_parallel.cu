// Qualification of `TensorParallelExchange`, the all-reduce of a tensor-parallel group.
//
// The oracle is the reduction itself evaluated on the host in the same order: each rank's FP32
// partials summed in rank order, added to the BF16 residual in FP32, rounded once. Only adds
// appear, so the host evaluates the same IEEE operations and the comparison is exact. Every rank's
// replica must also equal every other rank's. Ranks share device 0, as in the stage-link test:
// `force_staged` runs the real mapped-host path, and without it ranks on one card take the peer
// path. With two or more devices the ranks are also dealt over them, so the staged path crosses
// cards. A captured sequence of exchanges with changed inputs between replays checks that the
// protocol holds inside one graph spanning every rank's stream.
//
// The host writes the inputs with blocking copies from pageable memory, which may return before
// the data reaches the device; the rank streams do not wait for them (they are non-blocking), so
// every device is synchronized before the exchange runs.

#include "core/device.h"
#include "core/tensor_parallel.h"
#include "cuda_availability.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

using ninfer::test::cuda_unavailable;

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

std::uint16_t bits(__nv_bfloat16 value) {
    std::uint16_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

struct Case {
    std::size_t ranks;
    std::int64_t values;
    bool force_staged;
    int devices; // ranks are dealt over devices [0, devices)
};

// The replica every rank must hold: the residual plus the rank-order sum of the partials.
std::vector<std::uint16_t> oracle(const std::vector<__nv_bfloat16>& residual,
                                  const std::vector<std::vector<float>>& partials) {
    std::vector<std::uint16_t> out(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        float sum = partials[0][i];
        for (std::size_t r = 1; r < partials.size(); ++r) { sum += partials[r][i]; }
        out[i] = bits(__float2bfloat16(__bfloat162float(residual[i]) + sum));
    }
    return out;
}

int compare_replicas(const std::vector<__nv_bfloat16*>& replica, std::int64_t values,
                     const std::vector<std::uint16_t>& want, const char* label) {
    int failures = 0;
    for (std::size_t r = 0; r < replica.size(); ++r) {
        std::vector<std::uint16_t> got(want.size());
        CUDA_CHECK(cudaMemcpy(got.data(), replica[r], values * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < want.size(); ++i) { wrong += got[i] != want[i]; }
        if (wrong != 0) {
            std::cerr << "rank " << r << " of " << replica.size() << ", " << values << " values: "
                      << wrong << " differ\n";
        }
        failures += expect(wrong == 0, label);
    }
    return failures;
}

int run_case(const Case& c, std::mt19937& rng) {
    std::vector<int> ids(c.ranks);
    for (std::size_t r = 0; r < c.ranks; ++r) { ids[r] = static_cast<int>(r) % c.devices; }
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    const auto settle = [&] {
        for (int device = 0; device < c.devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(0));
    };
    const std::size_t bytes = static_cast<std::size_t>(c.values) * sizeof(float);
    ninfer::TensorParallelExchange exchange(ctx,
                                            {.slot_bytes = bytes, .force_staged = c.force_staged});
    // Across cards the transport follows their peer access; on one card it is the override's.
    int failures = expect((c.devices > 1 && !c.force_staged) ||
                              exchange.transport() == (c.force_staged
                                                           ? ninfer::TensorParallelTransport::StagedHost
                                                           : ninfer::TensorParallelTransport::Peer),
                          "transport follows the topology and the override");

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16*> replica(c.ranks);
    for (std::size_t r = 0; r < c.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replica[r]),
                              c.values * sizeof(__nv_bfloat16)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    const auto residual_of = [&](float scale) {
        std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(c.values));
        for (auto& v : residual) { v = __float2bfloat16(scale * normal(rng)); }
        return residual;
    };
    const auto partials_of = [&](float scale) {
        std::vector<std::vector<float>> partials(c.ranks,
                                                 std::vector<float>(static_cast<std::size_t>(c.values)));
        for (auto& p : partials) {
            for (auto& v : p) { v = scale * normal(rng); }
        }
        return partials;
    };
    const auto load = [&](const std::vector<__nv_bfloat16>& residual,
                          const std::vector<std::vector<float>>& partials, std::size_t slot) {
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaMemcpy(replica[r], residual.data(), c.values * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(), bytes,
                                  cudaMemcpyHostToDevice));
        }
        settle();
    };

    // Three exchanges in a row, so both slots are used and the first is reused.
    for (int round = 0; round < 3; ++round) {
        const std::size_t slot = static_cast<std::size_t>(round) % 2;
        const auto residual    = residual_of(4.0f);
        const auto partials    = partials_of(1.0f + static_cast<float>(round));
        load(residual, partials, slot);
        for (std::size_t r = 0; r < c.ranks; ++r) {
            exchange.publish(r, slot, bytes, ctx.rank(r).stream);
        }
        for (std::size_t r = 0; r < c.ranks; ++r) {
            exchange.reduce(r, slot, replica[r], c.values, c.ranks, ctx.rank(r).stream);
        }
        ctx.synchronize();
        failures += compare_replicas(replica, c.values, oracle(residual, partials),
                                     "every replica equals the rank-order sum");
    }

    // A projection only rank 0 runs: the others publish nothing and every replica adds rank 0's.
    {
        const auto residual = residual_of(4.0f);
        const auto partials = partials_of(1.0f);
        load(residual, partials, 1);
        for (std::size_t r = 0; r < c.ranks; ++r) {
            exchange.publish(r, 1, r == 0 ? bytes : 0, ctx.rank(r).stream);
        }
        for (std::size_t r = 0; r < c.ranks; ++r) {
            exchange.reduce(r, 1, replica[r], c.values, 1, ctx.rank(r).stream);
        }
        ctx.synchronize();
        const std::vector<std::vector<float>> first(partials.begin(), partials.begin() + 1);
        failures += compare_replicas(replica, c.values, oracle(residual, first),
                                     "one contributor reaches every replica");
    }

    // A captured pass: two exchanges back to back on every rank's stream, joined into rank 0's.
    {
        cudaStream_t origin = ctx.rank(0).stream;
        std::vector<cudaEvent_t> fork(c.ranks), join(c.ranks);
        // An event is recorded on a stream of its own device.
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaEventCreateWithFlags(&fork[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&join[r], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaSetDevice(0));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
        CUDA_CHECK(cudaEventRecord(fork[0], origin));
        for (std::size_t r = 1; r < c.ranks; ++r) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.rank(r).stream, fork[0], 0));
        }
        for (std::size_t slot = 0; slot < 2; ++slot) {
            for (std::size_t r = 0; r < c.ranks; ++r) {
                exchange.publish(r, slot, bytes, ctx.rank(r).stream);
            }
            for (std::size_t r = 0; r < c.ranks; ++r) {
                exchange.reduce(r, slot, replica[r], c.values, c.ranks, ctx.rank(r).stream);
            }
        }
        for (std::size_t r = 1; r < c.ranks; ++r) {
            CUDA_CHECK(cudaEventRecord(join[r], ctx.rank(r).stream));
            CUDA_CHECK(cudaStreamWaitEvent(origin, join[r], 0));
        }
        CUDA_CHECK(cudaStreamEndCapture(origin, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            const auto residual = residual_of(1.0f);
            const auto first    = partials_of(1.0f);
            const auto second   = partials_of(1.0f);
            load(residual, first, 0);
            for (std::size_t r = 0; r < c.ranks; ++r) {
                CUDA_CHECK(cudaMemcpy(exchange.partial(r, 1), second[r].data(), bytes,
                                      cudaMemcpyHostToDevice));
            }
            settle();
            CUDA_CHECK(cudaGraphLaunch(executable, origin));
            CUDA_CHECK(cudaStreamSynchronize(origin));
            const auto once = oracle(residual, first);
            std::vector<__nv_bfloat16> after(once.size());
            std::memcpy(after.data(), once.data(), once.size() * sizeof(std::uint16_t));
            failures += compare_replicas(replica, c.values, oracle(after, second),
                                         "a captured pass replays both exchanges");
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaEventDestroy(fork[r]));
            CUDA_CHECK(cudaEventDestroy(join[r]));
        }
    }

    try {
        exchange.publish(0, 0, bytes + sizeof(float), ctx.rank(0).stream);
        failures += expect(false, "an oversized partial is refused");
    } catch (const std::invalid_argument&) {}
    for (std::size_t r = 0; r < c.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaFree(replica[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

int run() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));
    std::mt19937 rng(20261025);
    int failures = 0;
    // One decode column of the 27B residual, a verify window, an odd size and a prompt chunk.
    for (const int devices : {1, std::min(count, 2)}) {
        for (const bool staged : {true, false}) {
            for (const std::size_t ranks : {2U, 3U, 4U}) {
                for (const std::int64_t values : {5120LL, 5120LL * 4, 1031LL, 5120LL * 512}) {
                    failures += run_case({ranks, values, staged, devices}, rng);
                }
            }
        }
        if (count < 2) { break; }
    }
    if (failures != 0) { return 1; }
    std::cout << "tensor-parallel exchange holds\n";
    return 0;
}

} // namespace

int run_tensor_parallel_test() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
