#pragma once

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

// The all-reduce of a tensor-parallel group. Every rank of a DeviceContext holds a replica of the
// BF16 residual and an FP32 partial of one row-split projection; each rank adds the sum of all
// partials, taken in rank order, to its replica and rounds once:
//
//   x_r <- bf16(float(x_r) + ((p_0 + p_1) + ... + p_{N-1}))
//
// Every rank evaluates the same expression on the same values, so the replicas stay identical, and
// the one rounding is the one a single device's residual-adding projection makes. A projection only
// rank 0 runs is an exchange with one contributor: every replica adds rank 0's output.
//
// A rank writes its partial into its slot (`partial`), publishes it (`publish`) and, once every
// rank has published, reduces (`reduce`). Every rank publishes in every exchange, one outside the
// contributors with zero bytes. Two slots alternate between exchanges. A rank writes a
// slot again two exchanges later, after its reduce of the exchange in between, which waits for
// every other rank's publish of that exchange, which each rank issues after its own reduce of the
// earlier one: so no rank overwrites a slot another is still reading. Consecutive passes are
// separate in the same way as long as a pass ends in a synchronize, as every round does.
//
// Records and waits pair within one exchange, so the protocol is legal eagerly and inside one
// capture that spans every rank's stream.
//
// Transports, chosen once from the topology:
//   Peer        every pair of ranks has peer access or shares a device: a reduce reads the other
//               ranks' device slots directly.
//   StagedHost  a publish copies the slot into a pinned, mapped host slot, and a reduce reads the
//               other ranks' host slots across the bus. Works between any two cards.
enum class TensorParallelTransport { Peer, StagedHost };

struct TensorParallelExchangeOptions {
    // Largest partial one slot holds.
    std::size_t slot_bytes = 0;
    // Use the staged protocol even where peer access exists (tests).
    bool force_staged = false;
};

class TensorParallelExchange {
public:
    static constexpr std::size_t kSlots = 2;

    TensorParallelExchange(const DeviceContext& context, TensorParallelExchangeOptions options);
    ~TensorParallelExchange();

    TensorParallelExchange(const TensorParallelExchange&)            = delete;
    TensorParallelExchange& operator=(const TensorParallelExchange&) = delete;
    TensorParallelExchange(TensorParallelExchange&&)                 = delete;
    TensorParallelExchange& operator=(TensorParallelExchange&&)      = delete;

    [[nodiscard]] std::size_t ranks() const noexcept { return ranks_.size(); }
    [[nodiscard]] TensorParallelTransport transport() const noexcept { return transport_; }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }

    // Where `rank` writes its FP32 partial for an exchange in `slot`: device memory of that rank.
    [[nodiscard]] float* partial(std::size_t rank, std::size_t slot) const;
    // Makes `rank`'s partial of `bytes` in `slot` readable by the other ranks.
    void publish(std::size_t rank, std::size_t slot, std::size_t bytes, cudaStream_t stream);
    // Adds the partials ranks [0, contributors) published in `slot` to `rank`'s BF16 residual of
    // `count` values.
    void reduce(std::size_t rank, std::size_t slot, __nv_bfloat16* residual, std::int64_t count,
                std::size_t contributors, cudaStream_t stream);

private:
    struct Rank {
        int device = 0;
        std::array<float*, kSlots> device_slot{};
        std::array<float*, kSlots> host_slot{};
        std::array<cudaEvent_t, kSlots> published{};
    };

    void release() noexcept;
    void check_slot(std::size_t rank, std::size_t slot, std::size_t bytes) const;

    std::vector<Rank> ranks_;
    TensorParallelTransport transport_ = TensorParallelTransport::StagedHost;
    std::size_t slot_bytes_            = 0;
};

namespace detail {

// One rank's reduction: residual[i] <- bf16(float(residual[i]) + ((part[0][i] + part[1][i]) + ...)).
void tensor_parallel_reduce(__nv_bfloat16* residual, const float* const* parts, std::size_t count,
                            std::int64_t values, cudaStream_t stream);

} // namespace detail

} // namespace ninfer
