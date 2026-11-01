#pragma once

#include "core/device.h"

#include <cstdint>
#include <cstddef>
#include <optional>

namespace ninfer {
namespace handshake {
struct DeviceMailbox;
}

// One in-flight exchange between a CUDA stream and a host worker. Every launch is capturable;
// eager waits block the calling host thread to allow first-use CUDA kernel loading, while a
// captured graph waits on the GPU after its kernel set has loaded.
// the worker owns its scheduling and payload buffers. The stream publishes after its D2H copies,
// waits for preparation before consuming host operands, and retires after consuming the result.
// Retirement acknowledgement prevents a later graph node from reusing the shared host buffers.
class CudaHostHandshake {
public:
    struct Request {
        std::uint32_t sequence, tag, width, flags;
    };

    // `sequence`: a zeroed device word the owner keeps for the handshake's lifetime, the last
    // request's sequence. Only the device writes it; resume() restores it after its backing was
    // replaced.
    CudaHostHandshake(const RankContext& rank, std::uint32_t* sequence);
    ~CudaHostHandshake();
    CudaHostHandshake(const CudaHostHandshake&) = delete;
    CudaHostHandshake& operator=(const CudaHostHandshake&) = delete;

    void publish(std::uint32_t tag, std::uint32_t width, std::uint32_t flags, cudaStream_t stream);
    // For a kernel that publishes the request itself (handshake::publish, cuda_host_handshake.cuh).
    [[nodiscard]] handshake::DeviceMailbox device() const noexcept;
    void resume();
    void wait_prepared(cudaStream_t stream);
    void wait_completed(cudaStream_t stream);
    void retire(cudaStream_t stream);

    [[nodiscard]] std::optional<Request> pending(std::uint32_t last_sequence) const noexcept;
    void prepared(std::uint32_t sequence) noexcept;
    void completed(std::uint32_t sequence) noexcept;
    [[nodiscard]] bool retired(std::uint32_t sequence) const noexcept;
    void release(std::uint32_t sequence) noexcept;
    [[nodiscard]] std::size_t pinned_bytes() const noexcept;

private:
    void* host_ = nullptr;
    void* device_ = nullptr;
    std::uint32_t* sequence_ = nullptr;
};

} // namespace ninfer
