#include "core/file_read_queue.h"
#include "core/file_read_error.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__linux__)
#    include <cerrno>
#    include <fcntl.h>
#    include <linux/io_uring.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <sys/uio.h>
#    include <unistd.h>
#endif

namespace ninfer {
namespace {
constexpr std::size_t kAlignment = 4096;

struct AlignedFree {
    void operator()(std::uint8_t* p) const noexcept {
        ::operator delete(p, std::align_val_t{kAlignment});
    }
};

struct ReadSlot {
    std::unique_ptr<std::uint8_t, AlignedFree> bounce;
    std::size_t capacity = 0;
    std::uint8_t* data = nullptr;
    std::uint64_t offset = 0;
    std::size_t requested = 0, needed = 0, have = 0, prefix = 0;
    int error = 0;
    bool done = false;
};

#if defined(__linux__)
unsigned acquire(unsigned* p) { return std::atomic_ref<unsigned>(*p).load(std::memory_order_acquire); }
void release(unsigned* p, unsigned value) {
    std::atomic_ref<unsigned>(*p).store(value, std::memory_order_release);
}
#endif
} // namespace

struct FileReadQueue::Impl {
    bool direct;
    std::vector<ReadSlot> slots;
    std::vector<std::filesystem::path> paths;
#ifdef _WIN32
    HANDLE port = nullptr;
    std::vector<HANDLE> files;
    std::vector<OVERLAPPED> positions;
#elif defined(__linux__)
    int ring = -1;
    std::vector<int> files;
    std::vector<iovec> vectors;
    void* sq_map = MAP_FAILED;
    void* cq_map = MAP_FAILED;
    void* entries_map = MAP_FAILED;
    std::size_t sq_bytes = 0, cq_bytes = 0, entries_bytes = 0;
    unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
    unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_sqe* entries = nullptr;
    io_uring_cqe* completions = nullptr;
#endif

    Impl(bool direct_io, std::size_t depth) : direct(direct_io), slots(depth) {}

    ~Impl() {
#ifdef _WIN32
        for (const auto file : files) { CloseHandle(file); }
        if (port != nullptr) { CloseHandle(port); }
#elif defined(__linux__)
        if (entries_map != MAP_FAILED) { ::munmap(entries_map, entries_bytes); }
        if (cq_map != MAP_FAILED && cq_map != sq_map) { ::munmap(cq_map, cq_bytes); }
        if (sq_map != MAP_FAILED) { ::munmap(sq_map, sq_bytes); }
        if (ring >= 0) { ::close(ring); }
        for (const int file : files) { ::close(file); }
#endif
    }

    bool initialize(std::span<const std::filesystem::path> input) {
        paths.assign(input.begin(), input.end());
#ifdef _WIN32
        positions.resize(slots.size());
        files.reserve(paths.size());
        port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
        if (port == nullptr) { throw std::runtime_error("file read queue: cannot create IOCP"); }
        for (const auto& path : paths) {
            DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS;
            if (direct) { flags |= FILE_FLAG_NO_BUFFERING; }
            const HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                            OPEN_EXISTING, flags, nullptr);
            if (file == INVALID_HANDLE_VALUE) {
                throw std::runtime_error("file read queue: cannot open " + path.string() +
                                         " (error " + std::to_string(GetLastError()) + ")");
            }
            files.push_back(file);
            if (CreateIoCompletionPort(file, port, 0, 0) == nullptr) {
                throw std::runtime_error("file read queue: cannot attach " + path.string() + " to IOCP");
            }
        }
        return true;
#elif defined(__linux__)
        io_uring_params params{};
        ring = static_cast<int>(::syscall(__NR_io_uring_setup, slots.size(), &params));
        if (ring < 0) {
            if (errno == ENOSYS || errno == EPERM || errno == EACCES || errno == EOPNOTSUPP) {
                return false;
            }
            throw std::runtime_error("file read queue: io_uring_setup: " + std::string(std::strerror(errno)));
        }
        sq_bytes = params.sq_off.array + params.sq_entries * sizeof(unsigned);
        cq_bytes = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
        const bool shared = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (shared) { sq_bytes = std::max(sq_bytes, cq_bytes); }
        const auto map = [&](std::size_t bytes, std::uint64_t offset) {
            void* memory = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, ring, offset);
            if (memory == MAP_FAILED) {
                throw std::runtime_error("file read queue: mmap: " + std::string(std::strerror(errno)));
            }
            return memory;
        };
        sq_map = map(sq_bytes, IORING_OFF_SQ_RING);
        cq_map = shared ? sq_map : map(cq_bytes, IORING_OFF_CQ_RING);
        entries_bytes = params.sq_entries * sizeof(io_uring_sqe);
        entries_map = map(entries_bytes, IORING_OFF_SQES);
        const auto field = [](void* base, unsigned offset) {
            return reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(base) + offset);
        };
        sq_head = field(sq_map, params.sq_off.head);
        sq_tail = field(sq_map, params.sq_off.tail);
        sq_mask = field(sq_map, params.sq_off.ring_mask);
        sq_array = field(sq_map, params.sq_off.array);
        cq_head = field(cq_map, params.cq_off.head);
        cq_tail = field(cq_map, params.cq_off.tail);
        cq_mask = field(cq_map, params.cq_off.ring_mask);
        entries = static_cast<io_uring_sqe*>(entries_map);
        completions = reinterpret_cast<io_uring_cqe*>(static_cast<std::uint8_t*>(cq_map) + params.cq_off.cqes);
        vectors.resize(slots.size());
        files.reserve(paths.size());
        for (const auto& path : paths) {
            const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
            if (file < 0) {
                throw std::runtime_error("file read queue: cannot open " + path.string() + ": " +
                                         std::strerror(errno));
            }
            files.push_back(file);
        }
        return true;
#else
        return false;
#endif
    }

    // All allocation and validation precede submission. Completion handling cannot throw while
    // another operation can still write to a caller's output or this queue's bounce buffers.
    void prepare(std::span<const QueuedFileRead> group) {
        for (std::size_t i = 0; i < group.size(); ++i) {
            const auto& read = group[i];
            auto& slot = slots[i];
            slot.prefix = direct ? read.offset % kAlignment : 0;
            slot.offset = read.offset - slot.prefix;
            slot.needed = slot.prefix + read.destination.size();
            slot.requested = direct ? (slot.needed + kAlignment - 1) & ~(kAlignment - 1) : slot.needed;
            if (direct && slot.requested > slot.capacity) {
                slot.bounce.reset(static_cast<std::uint8_t*>(
                    ::operator new(slot.requested, std::align_val_t{kAlignment})));
                slot.capacity = slot.requested;
            }
            slot.data = direct ? slot.bounce.get() : read.destination.data();
            slot.have = 0;
            slot.error = 0;
            slot.done = read.destination.empty();
        }
    }

    void enqueue(std::size_t index, std::size_t file) noexcept {
        auto& slot = slots[index];
        const auto offset = slot.offset + slot.have;
#ifdef _WIN32
        auto& position = positions[index];
        position = {};
        position.Offset = static_cast<DWORD>(offset);
        position.OffsetHigh = static_cast<DWORD>(offset >> 32);
        const BOOL ok = ReadFile(files[file], slot.data + slot.have,
                                 static_cast<DWORD>(slot.requested - slot.have), nullptr, &position);
        if (!ok && GetLastError() != ERROR_IO_PENDING) {
            slot.error = static_cast<int>(GetLastError());
            slot.done = true;
        }
#elif defined(__linux__)
        const unsigned tail = acquire(sq_tail);
        const unsigned entry = tail & *sq_mask;
        vectors[index] = {slot.data + slot.have, slot.requested - slot.have};
        auto& sqe = entries[entry];
        sqe = {};
        sqe.opcode = IORING_OP_READV;
        sqe.fd = files[file];
        sqe.off = offset;
        sqe.addr = reinterpret_cast<std::uintptr_t>(&vectors[index]);
        sqe.len = 1;
        sqe.user_data = index;
        sq_array[entry] = entry;
        release(sq_tail, tail + 1);
#else
        (void)file;
        (void)offset;
#endif
    }

    void flush() noexcept {
#if defined(__linux__)
        for (;;) {
            const unsigned head = acquire(sq_head), tail = acquire(sq_tail);
            if (head == tail) { return; }
            const long result = ::syscall(__NR_io_uring_enter, ring, tail - head, 0, 0, nullptr, 0);
            if (result >= 0 || errno == EINTR) { continue; }
            const int error = errno;
            // No SQPOLL: after enter returns, only another enter can consume pending SQEs.
            // Withdraw unsubmitted reads; completions of consumed ones are still drained.
            const unsigned consumed = acquire(sq_head);
            for (unsigned item = consumed; item != tail; ++item) {
                auto& slot = slots[entries[sq_array[item & *sq_mask]].user_data];
                slot.error = error;
                slot.done = true;
            }
            release(sq_tail, consumed);
            return;
        }
#endif
    }

    std::pair<std::size_t, std::int64_t> complete() noexcept {
#ifdef _WIN32
        for (;;) {
            DWORD bytes = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* position = nullptr;
            const BOOL ok = GetQueuedCompletionStatus(port, &bytes, &key, &position, INFINITE);
            if (position != nullptr) {
                return {static_cast<std::size_t>(position - positions.data()),
                        ok ? std::int64_t(bytes) : -std::int64_t(GetLastError())};
            }
            // A port-level error must not free buffers with outstanding reads. Request
            // cancellation and drain each corresponding completion before returning failure.
            for (const auto file : files) { CancelIoEx(file, nullptr); }
            std::this_thread::yield();
        }
#elif defined(__linux__)
        for (;;) {
            const unsigned head = acquire(cq_head);
            if (head != acquire(cq_tail)) {
                const auto cqe = completions[head & *cq_mask];
                release(cq_head, head + 1);
                return {static_cast<std::size_t>(cqe.user_data), cqe.res};
            }
            if (::syscall(__NR_io_uring_enter, ring, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0) < 0 &&
                errno != EINTR) {
                std::this_thread::yield();
            }
        }
#else
        return {};
#endif
    }

    void read(std::span<const QueuedFileRead> reads) {
        for (const auto& read : reads) {
            if (read.file >= paths.size() || read.destination.size() > (1U << 30) ||
                read.offset > std::uint64_t(std::numeric_limits<std::int64_t>::max()) -
                                  read.destination.size() - kAlignment) {
                throw std::invalid_argument("file read queue: invalid file, offset or range size");
            }
        }
        for (std::size_t first = 0; first < reads.size(); first += slots.size()) {
            const auto group = reads.subspan(first, std::min(slots.size(), reads.size() - first));
            prepare(group);
            for (std::size_t i = 0; i < group.size(); ++i) {
                if (!slots[i].done) { enqueue(i, group[i].file); }
            }
            for (;;) {
                flush();
                if (std::all_of(slots.begin(), slots.begin() + group.size(),
                                [](const auto& slot) { return slot.done; })) { break; }
                const auto [index, result] = complete();
                auto& slot = slots[index];
                if (result <= 0) {
                    // Zero means EOF. Use a platform-independent sentinel for a short range.
                    slot.error = result < 0 ? static_cast<int>(-result) : -1;
                    slot.done = true;
                } else {
                    slot.have += static_cast<std::size_t>(result);
                    if (slot.have >= slot.needed) {
                        slot.done = true;
                    } else if (direct && slot.have % kAlignment != 0) {
                        slot.error = -1;
                        slot.done = true;
                    } else {
                        enqueue(index, group[index].file);
                    }
                }
            }
            for (std::size_t i = 0; i < group.size(); ++i) {
                const auto& slot = slots[i];
                if (slot.error != 0) {
                    throw FileReadError("file read queue: " + paths[group[i].file].string() +
                                        (slot.error == -1 ? " ends early"
                                                          : " read failed (error " +
                                                                std::to_string(slot.error) + ")"));
                }
            }
            if (direct) {
                for (std::size_t i = 0; i < group.size(); ++i) {
                    if (!group[i].destination.empty()) {
                        std::memcpy(group[i].destination.data(), slots[i].data + slots[i].prefix,
                                    group[i].destination.size());
                    }
                }
            }
        }
    }
};

std::unique_ptr<FileReadQueue> FileReadQueue::open(
    std::span<const std::filesystem::path> paths, bool direct, std::size_t depth) {
    if (depth == 0 || depth > 1024) { throw std::invalid_argument("file read queue: depth must be 1..1024"); }
    auto impl = std::make_unique<Impl>(direct, depth);
    if (!impl->initialize(paths)) { return nullptr; }
    return std::unique_ptr<FileReadQueue>(new FileReadQueue(std::move(impl)));
}

FileReadQueue::FileReadQueue(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FileReadQueue::~FileReadQueue() = default;
void FileReadQueue::read(std::span<const QueuedFileRead> reads) { impl_->read(reads); }
const char* FileReadQueue::backend() const noexcept {
#ifdef _WIN32
    return "iocp";
#elif defined(__linux__)
    return "io_uring";
#else
    return "unavailable";
#endif
}
} // namespace ninfer
