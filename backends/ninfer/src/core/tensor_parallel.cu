#include "core/tensor_parallel.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string("TensorParallelExchange ") + what + ": " +
                             cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

struct PartialSet {
    const float* part[kMaxRanks];
    int count;
};

// Four values per thread where the residual allows it; the adds run in rank order either way.
__global__ void tensor_parallel_reduce_kernel(__nv_bfloat16* __restrict__ residual,
                                              PartialSet parts, std::int64_t values) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < values; i += stride) {
        float sum = parts.part[0][i];
        for (int r = 1; r < parts.count; ++r) { sum += parts.part[r][i]; }
        residual[i] = __float2bfloat16(__bfloat162float(residual[i]) + sum);
    }
}

} // namespace

namespace detail {

void tensor_parallel_reduce(__nv_bfloat16* residual, const float* const* parts, std::size_t count,
                            std::int64_t values, cudaStream_t stream) {
    if (count == 0 || count > kMaxRanks || values < 0) {
        throw std::invalid_argument("tensor_parallel_reduce: invalid rank or value count");
    }
    if (values == 0) { return; }
    PartialSet set{};
    for (std::size_t r = 0; r < count; ++r) { set.part[r] = parts[r]; }
    set.count             = static_cast<int>(count);
    constexpr int kBlock  = 256;
    const std::int64_t blocks = std::min<std::int64_t>((values + kBlock - 1) / kBlock, 4096);
    tensor_parallel_reduce_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
        residual, set, values);
    check(cudaGetLastError(), "reduce launch");
}

} // namespace detail

TensorParallelExchange::TensorParallelExchange(const DeviceContext& context,
                                               TensorParallelExchangeOptions options)
    : slot_bytes_(options.slot_bytes) {
    if (context.size() < 2) {
        throw std::invalid_argument("a tensor-parallel exchange needs at least two ranks");
    }
    if (slot_bytes_ == 0 || slot_bytes_ % sizeof(float) != 0) {
        throw std::invalid_argument("a tensor-parallel slot holds a positive number of floats");
    }
    bool peer = !options.force_staged;
    for (std::size_t a = 0; a < context.size() && peer; ++a) {
        for (std::size_t b = 0; b < context.size(); ++b) {
            if (a != b && !context.same_physical_device(a, b) && !context.peer_access(b, a)) {
                peer = false;
                break;
            }
        }
    }
    transport_ = peer ? TensorParallelTransport::Peer : TensorParallelTransport::StagedHost;
    ranks_.resize(context.size());
    try {
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            Rank& rank  = ranks_[r];
            rank.device = context.rank(r).device;
            DeviceBinding bind(rank.device);
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                check(cudaMalloc(reinterpret_cast<void**>(&rank.device_slot[slot]), slot_bytes_),
                      "cudaMalloc");
                if (transport_ == TensorParallelTransport::StagedHost) {
                    // Mapped, so a reduce on any rank reads it in place; portable, so every
                    // rank's context treats it as pinned.
                    check(cudaHostAlloc(reinterpret_cast<void**>(&rank.host_slot[slot]),
                                        slot_bytes_, cudaHostAllocMapped | cudaHostAllocPortable),
                          "cudaHostAlloc");
                }
                // Events cannot be created while a stream captures, so all exist up front. A
                // rank records its own events.
                check(cudaEventCreateWithFlags(&rank.published[slot], cudaEventDisableTiming),
                      "cudaEventCreate");
            }
        }
    } catch (...) {
        release();
        throw;
    }
}

TensorParallelExchange::~TensorParallelExchange() { release(); }

void TensorParallelExchange::release() noexcept {
    for (Rank& rank : ranks_) {
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            if (rank.published[slot] != nullptr) { (void)cudaEventDestroy(rank.published[slot]); }
            if (rank.device_slot[slot] != nullptr) { (void)cudaFree(rank.device_slot[slot]); }
            if (rank.host_slot[slot] != nullptr) { (void)cudaFreeHost(rank.host_slot[slot]); }
            rank.published[slot]   = nullptr;
            rank.device_slot[slot] = nullptr;
            rank.host_slot[slot]   = nullptr;
        }
    }
    ranks_.clear();
}

void TensorParallelExchange::check_slot(std::size_t rank, std::size_t slot,
                                        std::size_t bytes) const {
    if (rank >= ranks_.size() || slot >= kSlots) {
        throw std::out_of_range("tensor-parallel rank or slot is out of range");
    }
    if (bytes > slot_bytes_) {
        throw std::invalid_argument("tensor-parallel partial exceeds its slot");
    }
}

float* TensorParallelExchange::partial(std::size_t rank, std::size_t slot) const {
    check_slot(rank, slot, 0);
    return ranks_[rank].device_slot[slot];
}

void TensorParallelExchange::publish(std::size_t rank, std::size_t slot, std::size_t bytes,
                                     cudaStream_t stream) {
    check_slot(rank, slot, bytes);
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    if (transport_ == TensorParallelTransport::StagedHost && bytes != 0) {
        check(cudaMemcpyAsync(self.host_slot[slot], self.device_slot[slot], bytes,
                              cudaMemcpyDeviceToHost, stream),
              "publish copy");
    }
    check(cudaEventRecord(self.published[slot], stream), "publish fence");
}

void TensorParallelExchange::reduce(std::size_t rank, std::size_t slot, __nv_bfloat16* residual,
                                    std::int64_t count, std::size_t contributors,
                                    cudaStream_t stream) {
    check_slot(rank, slot, static_cast<std::size_t>(count) * sizeof(float));
    if (contributors == 0 || contributors > ranks_.size()) {
        throw std::invalid_argument("tensor-parallel reduce needs one to all ranks as contributors");
    }
    std::array<const float*, kMaxRanks> parts{};
    DeviceBinding bind(ranks_[rank].device);
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = ranks_[r].device_slot[slot];
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot], 0), "reduce wait");
        parts[r] = transport_ == TensorParallelTransport::Peer ? ranks_[r].device_slot[slot]
                                                               : ranks_[r].host_slot[slot];
    }
    detail::tensor_parallel_reduce(residual, parts.data(), contributors, count, stream);
}

} // namespace ninfer
