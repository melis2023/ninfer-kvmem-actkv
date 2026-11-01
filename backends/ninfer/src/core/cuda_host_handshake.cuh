#pragma once

// Device side of CudaHostHandshake, for a kernel that publishes its own request after writing the
// worker's operands into mapped host memory: the stream then needs no publish() or D2H copy.

#include <cuda/atomic>

#include <cstdint>

namespace ninfer::handshake {

struct Mailbox {
    alignas(64) std::uint32_t requested = 0;
    std::uint32_t tag = 0, width = 0, flags = 0;
    alignas(64) std::uint32_t prepared = 0;
    alignas(64) std::uint32_t completed = 0;
    alignas(64) std::uint32_t retired = 0;
    alignas(64) std::uint32_t released = 0;
};

// Naturally aligned 32-bit loads/stores to mapped host memory are system-atomic even on PCIe
// devices without hostNativeAtomicSupported. Do not replace these with RMW operations.
// https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html
using Atomic = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;

// The mapped mailbox and the device's copy of its request sequence, which the device alone writes:
// device code reads it without a round trip over the bus.
struct DeviceMailbox {
    Mailbox* box             = nullptr;
    std::uint32_t* sequence  = nullptr;
};

// One thread, once every thread whose writes the worker reads has fenced them at system scope and
// the block has synchronized.
__device__ inline void publish(DeviceMailbox mailbox, std::uint32_t tag, std::uint32_t width,
                               std::uint32_t flags) {
    const auto sequence = *mailbox.sequence + 1;
    *mailbox.sequence   = sequence;
    mailbox.box->tag    = tag;
    mailbox.box->width  = width;
    mailbox.box->flags  = flags;
    Atomic(mailbox.box->requested).store(sequence, cuda::memory_order_release);
}

} // namespace ninfer::handshake
