#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace ninfer {

struct QueuedFileRead {
    std::size_t file = 0;
    std::uint64_t offset = 0;
    std::span<std::uint8_t> destination;
};

// A bounded OS queue for positioned reads into host memory. Call read() from the owning
// worker to overlap a batch with device execution. Output ranges must not overlap and stay
// alive until read() returns. Even a failed read drains every submitted operation first.
class FileReadQueue {
public:
    // Linux: io_uring; Windows: overlapped reads and IOCP. Returns null when the OS or its
    // sandbox disables the backend. File/open/allocation failures throw. Direct mode accepts
    // unaligned logical ranges; the queue owns at most depth aligned bounce buffers.
    [[nodiscard]] static std::unique_ptr<FileReadQueue>
    open(std::span<const std::filesystem::path> paths, bool direct, std::size_t depth);
    ~FileReadQueue();
    FileReadQueue(const FileReadQueue&) = delete;
    FileReadQueue& operator=(const FileReadQueue&) = delete;

    void read(std::span<const QueuedFileRead> reads);
    [[nodiscard]] const char* backend() const noexcept;

private:
    struct Impl;
    explicit FileReadQueue(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
