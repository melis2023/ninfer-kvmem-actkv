#pragma once

#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/stage_link.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// A non-blocking stream of one device, destroyed with it.
class SideStream {
public:
    explicit SideStream(int device) : device_(device) {
        DeviceBinding bind(device_);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~SideStream() {
        if (stream_ == nullptr) { return; }
        DeviceBinding bind(device_);
        (void)cudaStreamDestroy(stream_);
    }

    SideStream(SideStream&& other) noexcept
        : device_(other.device_), stream_(std::exchange(other.stream_, nullptr)) {}

    SideStream(const SideStream&)            = delete;
    SideStream& operator=(const SideStream&) = delete;
    SideStream& operator=(SideStream&&)      = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    int device_          = 0;
    cudaStream_t stream_ = nullptr;
};

// What a forward pass across pipeline stages needs beyond one device's state. The Program owns one
// when the model is split over several devices and hands a pointer to each TextContext; it is null on
// one device, where none of this exists.
//
// A stage owns its layers whole, so each stage reads its own device's KV planes and Linear Attention
// state. The embedding, head and round state stay on rank 0: the residual stream leaves rank 0 after
// its layers, crosses each later stage in turn, and the last stage sends it back.
struct StageRuntime {
    // The Linear Attention state pool of each shard, and the global index of the first layer in it.
    std::vector<LinearAttentionStatePool*> state;
    std::vector<std::uint32_t> state_first_layer;
    // The ReplaySSM record storage of each shard, in the same order; empty without speculative
    // decoding. A stage records only its own layers, on its own device.
    std::vector<const GdnReplayRecords*> replay;

    // forward[s] carries the residual from stage s to stage s+1; `back` from the last stage to rank
    // 0. `control[s-1]` carries the small position, row and slot tensors the layers read from rank 0
    // to stage s, whose device cannot follow a pointer into rank 0's memory.
    std::vector<StageLink> forward;
    std::optional<StageLink> back;
    std::vector<StageLink> control;

    // The DFlash feature taps among one later stage's layers, in the drafter's order, each with the
    // link that carries its output, a [hidden, columns] slab, to rank 0, where the drafter's
    // buffers live. A tapped layer's output leaves on `stream` while the stage's later layers run:
    // `ready[j]` hands its staged copy from the stage's stream to `stream`, and `sent` joins
    // `stream` back into the stage's stream once its layers are done.
    struct FeatureLink {
        std::vector<std::uint32_t> layers;
        std::vector<StageLink> links;
        std::vector<CudaCompletionEvent> ready;
        CudaCompletionEvent sent;
        SideStream stream;
    };

    // features[s-1] serves stage s: present when that stage holds a tapped layer, absent otherwise.
    // Empty without a masked drafter.
    std::vector<std::optional<FeatureLink>> features;

    // Alternates per forward pass so consecutive passes use different slots of every link.
    std::uint32_t next_slot = 0;
};

} // namespace ninfer::models::qwen3_5::execution
