#include "models/qwen4_exp/ngram_table.h"
#include "core/file_read_error.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#    ifndef _WIN32_WINNT
#        define _WIN32_WINNT 0x0602
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/uio.h>
#    include <unistd.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

// Direct reads cover whole 4 KiB blocks at 4 KiB-aligned file offsets into 4 KiB-aligned memory.
constexpr std::uint64_t kAlign = 4096;
// A residency's load reads the table in pieces of about this size, several at once.
constexpr std::uint64_t kLoadChunk = 4ULL << 20;
// A load piece holding fewer hot rows than one per this many bytes reads them one by one.
constexpr std::uint64_t kSparseLoad = 8192;
// Direct row reads up to this size use a buffer each thread keeps.
constexpr std::uint64_t kThreadBounce = 64 * 1024;
// A batch of at most this many rows reads those the page cache holds on the submitting thread
// (about a microsecond each); a prompt chunk's thousands go to the threads.
constexpr std::size_t kInlineRows = 1024;

std::string mib(std::uint64_t bytes) {
    return std::to_string((bytes + (1ULL << 19)) >> 20) + " MiB";
}

#ifndef _WIN32
std::string error_text(int error) { return std::strerror(error); }
#endif

struct AlignedFree {
    void operator()(std::uint8_t* p) const noexcept {
        ::operator delete(p, std::align_val_t{kAlign});
    }
};

using AlignedBytes = std::unique_ptr<std::uint8_t, AlignedFree>;

AlignedBytes aligned_bytes(std::uint64_t bytes) {
    return AlignedBytes(
        static_cast<std::uint8_t*>(::operator new(bytes, std::align_val_t{kAlign})));
}

// The bounce buffer of a direct row read on this thread.
std::uint8_t* thread_bounce() {
    thread_local AlignedBytes buffer = aligned_bytes(kThreadBounce);
    return buffer.get();
}

std::size_t latency_bucket(std::uint64_t nanoseconds) {
    const std::uint64_t micro = nanoseconds / 1000;
    return std::min<std::size_t>(std::bit_width(micro), NgramTableStats::kLatencyBuckets - 1);
}

} // namespace

struct NgramTableReader::File {
    NgramIo io;
    std::filesystem::path path;
    std::uint64_t bytes      = 0;       // the file's size
    const std::uint8_t* view = nullptr; // Mapped: the whole file
    // Linux Buffered: a mapping of the whole file, never read, for mincore() to tell which of its
    // pages the page cache holds where RWF_NOWAIT is refused (overlayfs).
    const std::uint8_t* probe = nullptr;
    std::uint64_t page        = 4096;
    mutable std::atomic<bool> nowait_unsupported{false};
#ifdef _WIN32
    HANDLE handle  = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int descriptor = -1;
#endif

    File(std::filesystem::path file, NgramIo mode) : io(mode), path(std::move(file)) {
#ifdef _WIN32
        // Random access: the cache manager does not read ahead past each row.
        DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS;
        if (io == NgramIo::Direct) { flags |= FILE_FLAG_NO_BUFFERING; }
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             flags, nullptr);
        if (handle == INVALID_HANDLE_VALUE) { fail("cannot open"); }
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(handle, &size)) { fail("cannot size"); }
        bytes = static_cast<std::uint64_t>(size.QuadPart);
        if (io == NgramIo::Mapped) {
            mapping = CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping == nullptr) { fail("cannot map"); }
            view = static_cast<const std::uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
            if (view == nullptr) { fail("cannot map"); }
        }
#else
        int flags = O_RDONLY | O_CLOEXEC;
        if (io == NgramIo::Direct) { flags |= O_DIRECT; }
        descriptor = ::open(path.c_str(), flags);
        if (descriptor < 0) {
            const int error = errno;
            if (io == NgramIo::Direct && error == EINVAL) {
                throw std::runtime_error("n-gram table: " + path.string() +
                                         " is on a file system without direct I/O; use "
                                         "--ngram-io buffered or mapped");
            }
            throw std::runtime_error("n-gram table: cannot open " + path.string() + ": " +
                                     error_text(error));
        }
        struct stat status{};
        if (::fstat(descriptor, &status) != 0) { fail("cannot size"); }
        bytes = static_cast<std::uint64_t>(status.st_size);
#    if defined(POSIX_FADV_RANDOM)
        if (io != NgramIo::Direct) { (void)::posix_fadvise(descriptor, 0, 0, POSIX_FADV_RANDOM); }
#    endif
        if (io == NgramIo::Mapped && bytes > 0) {
            void* p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, descriptor, 0);
            if (p == MAP_FAILED) { fail("cannot map"); }
            (void)::madvise(p, bytes, MADV_RANDOM);
            view = static_cast<const std::uint8_t*>(p);
        }
#    ifdef __linux__
        if (io == NgramIo::Buffered && bytes > 0) {
            void* p = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, descriptor, 0);
            if (p != MAP_FAILED) {
                (void)::madvise(p, bytes, MADV_RANDOM);
                probe = static_cast<const std::uint8_t*>(p);
            }
            page = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
        }
#    endif
#endif
    }

    ~File() {
#ifdef _WIN32
        if (view != nullptr) UnmapViewOfFile(view);
        if (mapping != nullptr) CloseHandle(mapping);
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (view != nullptr) ::munmap(const_cast<std::uint8_t*>(view), bytes);
        if (probe != nullptr) ::munmap(const_cast<std::uint8_t*>(probe), bytes);
        if (descriptor >= 0) ::close(descriptor);
#endif
    }

    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    [[noreturn]] void fail(const char* what) const {
#ifdef _WIN32
        throw std::runtime_error("n-gram table: " + std::string(what) + " " + path.string() +
                                 " (error " + std::to_string(GetLastError()) + ")");
#else
        throw std::runtime_error("n-gram table: " + std::string(what) + " " + path.string() + ": " +
                                 error_text(errno));
#endif
    }

    // A row read that failed while the Engine ran; the reader caches no row of a failed batch.
    [[noreturn]] void fail_read() const {
#ifdef _WIN32
        throw FileReadError("n-gram table: cannot read " + path.string() + " (error " +
                            std::to_string(GetLastError()) + ")");
#else
        throw FileReadError("n-gram table: cannot read " + path.string() + ": " +
                            error_text(errno));
#endif
    }

    // One positioned read of up to `bytes` at `offset`; 0 at the end of the file.
    std::size_t read_some(std::uint64_t offset, std::uint8_t* destination,
                          std::size_t bytes) const {
        bytes = std::min<std::size_t>(bytes, 1U << 30U);
        for (;;) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            DWORD done          = 0;
            if (!ReadFile(handle, destination, static_cast<DWORD>(bytes), &done, &position)) {
                if (GetLastError() == ERROR_HANDLE_EOF) { return 0; }
                fail_read();
            }
            return done;
#else
            const ssize_t done =
                ::pread(descriptor, destination, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) { continue; }
            if (done < 0) { fail_read(); }
            return static_cast<std::size_t>(done);
#endif
        }
    }

    // Reads exactly `bytes` at `offset`, through the page cache or out of the mapping.
    void read_cached(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        if (view != nullptr) {
            std::memcpy(destination, view + offset, bytes);
            return;
        }
        while (bytes > 0) {
            const std::size_t done = read_some(offset, destination, bytes);
            if (done == 0) {
                throw FileReadError("n-gram table: " + path.string() + " ends early");
            }
            destination += done;
            offset += done;
            bytes -= done;
        }
    }

    // Reads exactly `bytes` at `offset` past the page cache: the 4 KiB blocks around them into an
    // aligned buffer, then the bytes out of it.
    void read_direct(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        const std::uint64_t begin = offset & ~(kAlign - 1);
        const std::uint64_t end   = (offset + bytes + kAlign - 1) & ~(kAlign - 1);
        AlignedBytes owned;
        std::uint8_t* bounce = nullptr;
        if (end - begin <= kThreadBounce) {
            bounce = thread_bounce();
        } else {
            owned  = aligned_bytes(end - begin);
            bounce = owned.get();
        }
        const std::uint64_t need = offset + bytes - begin;
        std::uint64_t have       = 0;
        while (have < need) {
            const std::size_t done = read_some(begin + have, bounce + have, end - begin - have);
            if (done == 0) {
                throw FileReadError("n-gram table: " + path.string() + " ends early");
            }
            have += done;
        }
        std::memcpy(destination, bounce + (offset - begin), bytes);
    }

    void read(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        if (io == NgramIo::Direct) {
            read_direct(offset, destination, bytes);
        } else {
            read_cached(offset, destination, bytes);
        }
    }

    // Reads exactly `bytes` at `offset` when the page cache holds them all, without waiting for
    // the device; false otherwise. RWF_NOWAIT asks the kernel; where the file system refuses it,
    // mincore() over `probe` tells, which reports the page cache for a file this process owns or
    // may write and no page resident otherwise.
    bool read_resident(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
#ifdef __linux__
        if (io != NgramIo::Buffered) { return false; }
#    ifdef RWF_NOWAIT
        if (!nowait_unsupported.load(std::memory_order_relaxed)) {
            iovec vector{destination, bytes};
            for (;;) {
                const ssize_t done =
                    ::preadv2(descriptor, &vector, 1, static_cast<off_t>(offset), RWF_NOWAIT);
                if (done == static_cast<ssize_t>(bytes)) { return true; }
                if (done < 0 && errno == EINTR) { continue; }
                if (done >= 0 || errno == EAGAIN) { return false; }
                break;
            }
            nowait_unsupported.store(true, std::memory_order_relaxed);
        }
#    endif
        const std::uint64_t first = offset / page * page;
        const std::uint64_t end   = (offset + bytes + page - 1) / page * page;
        unsigned char resident[4] = {};
        if (probe == nullptr || (end - first) / page > std::size(resident) ||
            ::mincore(const_cast<std::uint8_t*>(probe) + first, end - first, resident) != 0) {
            return false;
        }
        for (std::uint64_t i = 0; i < (end - first) / page; ++i) {
            if ((resident[i] & 1) == 0) { return false; }
        }
        read_cached(offset, destination, bytes);
        return true;
#else
        (void)offset;
        (void)destination;
        (void)bytes;
        return false;
#endif
    }

    void prefetch(std::uint64_t offset, std::uint64_t count) const noexcept {
        if (io == NgramIo::Direct) { return; }
#ifdef _WIN32
        if (view != nullptr) {
            WIN32_MEMORY_RANGE_ENTRY range{const_cast<std::uint8_t*>(view + offset),
                                          static_cast<SIZE_T>(count)};
            (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
        }
#elif defined(POSIX_FADV_WILLNEED)
        (void)::posix_fadvise(descriptor, static_cast<off_t>(offset),
                             static_cast<off_t>(count), POSIX_FADV_WILLNEED);
#endif
    }
};

// Anonymous memory for the resident rows: 2 MiB pages where Linux offers them, and lockable.
class NgramTableReader::Memory {
public:
    explicit Memory(std::uint64_t bytes, bool huge_pages = true) : bytes_(bytes) {
        if (bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("n-gram table: " + mib(bytes) + " exceed the address space");
        }
#ifdef _WIN32
        (void)huge_pages;
        data_ = static_cast<std::uint8_t*>(
            VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (data_ == nullptr) {
            throw std::runtime_error("n-gram table: cannot allocate " + mib(bytes) +
                                     " of RAM (error " + std::to_string(GetLastError()) + ")");
        }
#else
        void* p =
            ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error("n-gram table: cannot allocate " + mib(bytes) +
                                     " of RAM: " + error_text(errno));
        }
#    if defined(MADV_HUGEPAGE)
        (void)::madvise(p, bytes, huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
#    endif
        data_ = static_cast<std::uint8_t*>(p);
#endif
    }

    ~Memory() {
#ifdef _WIN32
        if (locked_) VirtualUnlock(data_, bytes_);
        VirtualFree(data_, 0, MEM_RELEASE);
#else
        if (locked_) ::munlock(data_, bytes_);
        ::munmap(data_, bytes_);
#endif
    }

    Memory(const Memory&)            = delete;
    Memory& operator=(const Memory&) = delete;

    [[nodiscard]] std::uint8_t* data() const noexcept { return data_; }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void lock() {
#ifdef _WIN32
        // VirtualLock keeps pages in the working set, whose minimum must make room for them.
        const HANDLE process = GetCurrentProcess();
        SIZE_T minimum = 0, maximum = 0;
        if (!GetProcessWorkingSetSize(process, &minimum, &maximum) ||
            !SetProcessWorkingSetSize(process, minimum + bytes_, maximum + bytes_) ||
            !VirtualLock(data_, bytes_)) {
            throw std::runtime_error("n-gram table: cannot lock " + mib(bytes_) +
                                     " in RAM (error " + std::to_string(GetLastError()) + ")");
        }
#else
        if (::mlock(data_, bytes_) != 0) {
            throw std::runtime_error("n-gram table: cannot lock " + mib(bytes_) +
                                     " in RAM: " + error_text(errno) +
                                     " (the memlock limit, ulimit -l, or CAP_IPC_LOCK)");
        }
#endif
        locked_ = true;
    }

private:
    std::uint8_t* data_ = nullptr;
    std::uint64_t bytes_;
    bool locked_ = false;
};

// Four-way CLOCK sets. The allocation includes tags, reference bits, hands, and row bytes;
// unlike a table-sized index its size depends only on the budget. Only the owner accesses it.
class NgramTableReader::RowCache {
public:
    RowCache(std::uint64_t rows, std::uint32_t row_bytes, std::uint64_t budget)
        : row_bytes_(row_bytes) {
        constexpr std::uint64_t ways = 4;
        const std::uint64_t per_set = ways * (std::uint64_t(row_bytes) + sizeof(std::uint64_t)) + 1;
        sets_ = std::min(budget / per_set, (rows - 1) / ways + 1);
        if (sets_ == 0) { return; }
        // Sparse row admission should not fault a 2 MiB page for one short row.
        memory_ = std::make_unique<Memory>(sets_ * per_set, false);
        tags_ = reinterpret_cast<std::uint64_t*>(memory_->data());
        clocks_ = memory_->data() + sets_ * ways * sizeof(std::uint64_t);
        payload_ = clocks_ + sets_;
        // Tags hold row + 1, so the anonymous memory's initial zeros mean empty. Neither the
        // index nor the payload needs touching before a lookup/admission reaches its page.
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return memory_ ? memory_->bytes() : 0; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }

    [[nodiscard]] bool contains(std::uint64_t row) const noexcept {
        if (sets_ == 0) { return false; }
        const std::uint64_t first = set(row) * 4;
        for (std::uint64_t i = 0; i < 4; ++i) {
            if (tags_[first + i] == row + 1) { return true; }
        }
        return false;
    }

    bool copy(std::uint64_t row, std::uint8_t* out) noexcept {
        if (sets_ == 0) { return false; }
        const std::uint64_t group = set(row);
        for (unsigned i = 0; i < 4; ++i) {
            const std::uint64_t slot = group * 4 + i;
            if (tags_[slot] != row + 1) { continue; }
            clocks_[group] |= static_cast<std::uint8_t>(1U << i);
            std::memcpy(out, payload_ + slot * row_bytes_, row_bytes_);
            return true;
        }
        return false;
    }

    void insert(std::uint64_t row, const std::uint8_t* data) noexcept {
        if (sets_ == 0) { return; }
        const std::uint64_t group = set(row);
        auto& clock = clocks_[group];
        for (unsigned i = 0; i < 4; ++i) {
            if (tags_[group * 4 + i] == row + 1) {
                clock |= static_cast<std::uint8_t>(1U << i);
                return;
            }
        }
        unsigned hand = (clock >> 4U) & 3U;
        while (tags_[group * 4 + hand] != kEmpty && (clock & (1U << hand)) != 0) {
            clock &= static_cast<std::uint8_t>(~(1U << hand));
            hand = (hand + 1) & 3U;
        }
        const std::uint64_t slot = group * 4 + hand;
        size_ += tags_[slot] == kEmpty;
        std::memcpy(payload_ + slot * row_bytes_, data, row_bytes_);
        tags_[slot] = row + 1;
        clock = static_cast<std::uint8_t>((clock & 15U) | (1U << hand) | (((hand + 1) & 3U) << 4U));
    }

private:
    [[nodiscard]] std::uint64_t set(std::uint64_t row) const noexcept {
        row ^= row >> 30U;
        row *= 0xbf58476d1ce4e5b9ULL;
        row ^= row >> 27U;
        row *= 0x94d049bb133111ebULL;
        return (row ^ (row >> 31U)) % sets_;
    }
    static constexpr std::uint64_t kEmpty = 0;
    std::uint64_t sets_ = 0, size_ = 0;
    std::uint32_t row_bytes_;
    std::unique_ptr<Memory> memory_;
    std::uint64_t* tags_ = nullptr;
    std::uint8_t* clocks_ = nullptr;
    std::uint8_t* payload_ = nullptr;
};

// Direct I/O cannot warm the OS page cache; Windows buffered reads have no file-offset hint.
// Keep one bounded speculative batch separately from
// demand reads; only a completely successful batch may supply bytes to a later request. The
// owner never waits for speculative work in prefetch() or submit(). Failed hints are discarded:
// a demand read still reports the underlying failure through its ordinary path.
class NgramTableReader::Lookahead {
public:
    explicit Lookahead(NgramTableReader& owner)
        : owner_(owner), payload_(std::size_t(owner.options_.depth) * owner.layout_.row_bytes),
          pool_(1) {
        std::vector<std::filesystem::path> paths;
        for (const auto& segment : owner.layout_.segments) { paths.push_back(segment.path); }
        queue_ = FileReadQueue::open(paths, owner.options_.io == NgramIo::Direct,
                                    owner.options_.depth);
        ids_.reserve(owner.options_.depth);
    }

    ~Lookahead() {
        try { pool_.finish(); } catch (...) {}
    }

    void discard() {
        if (active_) { pool_.finish(); }
        active_ = false;
    }

    void submit(std::span<const std::uint64_t> rows) {
        if (rows.empty() || (active_ && !complete_.load(std::memory_order_acquire))) { return; }
        if (active_) { pool_.finish(); }
        const auto count = std::min<std::size_t>(rows.size(), owner_.options_.depth);
        ids_.assign(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(count));
        reads_.clear();
        const auto rb = owner_.layout_.row_bytes;
        if (queue_) {
            for (std::size_t i = 0; i < count; ++i) {
                std::uint64_t offset = ids_[i] * rb;
                std::size_t remaining = rb;
                auto* destination = payload_.data() + i * rb;
                auto segment = static_cast<std::size_t>(
                    std::upper_bound(owner_.starts_.begin(), owner_.starts_.end(), offset) -
                    owner_.starts_.begin() - 1);
                while (remaining != 0) {
                    const auto& part = owner_.layout_.segments[segment];
                    const auto local = offset - owner_.starts_[segment];
                    const auto bytes = static_cast<std::size_t>(
                        std::min<std::uint64_t>(remaining, part.bytes - local));
                    reads_.push_back({segment, part.file_offset + local, {destination, bytes}});
                    offset += bytes;
                    destination += bytes;
                    remaining -= bytes;
                    ++segment;
                }
            }
        }
        succeeded_ = false;
        complete_.store(false, std::memory_order_relaxed);
        active_ = true;
        pool_.start(1, [this, rb](std::size_t) {
            try {
                if (queue_) { queue_->read(reads_); }
                else {
                    for (std::size_t i = 0; i < ids_.size(); ++i) {
                        owner_.read(ids_[i] * rb, payload_.data() + i * rb, rb);
                    }
                }
                succeeded_ = true;
            } catch (...) {}
            complete_.store(true, std::memory_order_release);
        });
    }

    bool copy(std::uint64_t row, std::uint8_t* destination) const {
        if (!active_ || !complete_.load(std::memory_order_acquire) || !succeeded_) { return false; }
        const auto found = std::lower_bound(ids_.begin(), ids_.end(), row);
        if (found == ids_.end() || *found != row) { return false; }
        const auto rb = owner_.layout_.row_bytes;
        std::memcpy(destination, payload_.data() + std::size_t(found - ids_.begin()) * rb, rb);
        return true;
    }

private:
    NgramTableReader& owner_;
    std::vector<std::uint8_t> payload_;
    std::vector<std::uint64_t> ids_;
    std::vector<QueuedFileRead> reads_;
    std::unique_ptr<FileReadQueue> queue_;
    bool active_ = false, succeeded_ = false;
    std::atomic<bool> complete_{false};
    ReadPool pool_;
};

// OS hints of many page ranges, one system call each, on a worker thread: a prompt step's rows
// take over a hundred thousand. A batch that arrives while the worker still hints the previous
// one is dropped; hints are advisory.
class NgramTableReader::Hints {
public:
    struct Range {
        std::size_t segment;
        std::uint64_t begin, bytes;
    };

    explicit Hints(const NgramTableReader& owner) : owner_(owner), pool_(1) {}

    ~Hints() {
        try { pool_.finish(); } catch (...) {}
    }

    void submit(std::vector<Range> ranges) {
        std::lock_guard lock(mutex_);
        if (active_ && !complete_.load(std::memory_order_acquire)) { return; }
        if (active_) { pool_.finish(); }
        ranges_ = std::move(ranges);
        complete_.store(false, std::memory_order_relaxed);
        active_ = true;
        pool_.start(1, [this](std::size_t) {
            for (const Range& range : ranges_) {
                owner_.files_[range.segment]->prefetch(range.begin, range.bytes);
            }
            complete_.store(true, std::memory_order_release);
        });
    }

private:
    const NgramTableReader& owner_;
    std::mutex mutex_;
    std::vector<Range> ranges_;
    bool active_ = false;
    std::atomic<bool> complete_{false};
    ReadPool pool_;
};

NgramTableReader::NgramTableReader(NgramTableLayout layout, const NgramReadOptions& options)
    : layout_(std::move(layout)), options_(options) {
    if (layout_.row_bytes == 0 || layout_.rows == 0 || layout_.segments.empty()) {
        throw std::invalid_argument("n-gram table: empty layout");
    }
    if (layout_.rows > std::numeric_limits<std::uint64_t>::max() / layout_.row_bytes) {
        throw std::invalid_argument("n-gram table: payload size overflows");
    }
    if (options_.depth == 0 || options_.depth > 1024) {
        throw std::invalid_argument("n-gram table: 1..1024 reads in flight");
    }
    if (options_.lock && options_.residency == NgramResidency::Disk) {
        throw std::invalid_argument("n-gram table: locking needs ram or ram-hot residency");
    }
    const std::uint64_t table_bytes = layout_.rows * layout_.row_bytes;
    std::uint64_t covered           = 0;
    for (const auto& segment : layout_.segments) {
        files_.push_back(std::make_unique<File>(segment.path, options_.io));
        if (segment.bytes == 0 ||
            segment.file_offset > std::numeric_limits<std::uint64_t>::max() - segment.bytes ||
            files_.back()->bytes < segment.file_offset + segment.bytes) {
            throw std::runtime_error("n-gram table: " + segment.path.string() + " holds " +
                                     std::to_string(files_.back()->bytes) +
                                     " bytes, short of its table segment");
        }
        starts_.push_back(covered);
        covered += segment.bytes;
    }
    if (covered < table_bytes) {
        throw std::runtime_error("n-gram table: the files hold " + std::to_string(covered) +
                                 " table bytes, the table needs " + std::to_string(table_bytes));
    }
    pool_ = std::make_unique<ReadPool>(options_.depth);
    if (options_.residency == NgramResidency::Ram) {
        load_resident();
    } else if (options_.residency == NgramResidency::RamHot) {
        load_hot();
    } else if (options_.budget_bytes != 0) {
        cache_ = std::make_unique<RowCache>(layout_.rows, layout_.row_bytes, options_.budget_bytes);
    }
    if (options_.residency != NgramResidency::Ram && options_.io != NgramIo::Mapped) {
        std::vector<std::filesystem::path> paths;
        for (const auto& segment : layout_.segments) { paths.push_back(segment.path); }
        queue_ = FileReadQueue::open(paths, options_.io == NgramIo::Direct, options_.depth);
        if (queue_) {
            // Startup's parallel bulk loads are finished. One worker now drives the OS queue;
            // its depth bounds native reads instead of allocating a thread for every read.
            pool_ = std::make_unique<ReadPool>(1);
        }
    }
    const bool staged_hints = options_.io == NgramIo::Direct
#ifdef _WIN32
                              || options_.io == NgramIo::Buffered
#endif
        ;
    if (options_.residency != NgramResidency::Ram && staged_hints) {
        lookahead_ = std::make_unique<Lookahead>(*this);
    } else if (options_.residency != NgramResidency::Ram) {
        hints_ = std::make_unique<Hints>(*this);
    }
    // The profile has served its purpose.
    options_.hot_rows = {};
    options_.hot_rows.shrink_to_fit();
}

NgramTableReader::~NgramTableReader() {
    try {
        wait();
    } catch (...) {}
}

std::uint64_t NgramTableReader::resident_bytes() const noexcept {
    return (resident_ ? resident_->bytes() : 0) + hot_bits_.size() * 8 + hot_blocks_.size() * 4 +
           (cache_ ? cache_->bytes() : 0);
}

const char* NgramTableReader::io_backend() const noexcept {
    if (options_.residency == NgramResidency::Ram) { return "ram"; }
    if (options_.io == NgramIo::Mapped) { return "mapped"; }
    return queue_ ? queue_->backend() : "positioned-threads";
}

void NgramTableReader::read(std::uint64_t offset, std::uint8_t* destination,
                            std::size_t bytes) const {
    auto segment = static_cast<std::size_t>(
        std::upper_bound(starts_.begin(), starts_.end(), offset) - starts_.begin() - 1);
    while (bytes > 0) {
        const auto& layout        = layout_.segments[segment];
        const std::uint64_t local = offset - starts_[segment];
        const std::size_t count =
            static_cast<std::size_t>(std::min<std::uint64_t>(bytes, layout.bytes - local));
        files_[segment]->read(layout.file_offset + local, destination, count);
        destination += count;
        offset += count;
        bytes -= count;
        ++segment;
    }
}

void NgramTableReader::load(std::uint64_t first, std::uint64_t count,
                            std::uint8_t* destination) const {
    const std::uint64_t rb    = layout_.row_bytes;
    const std::uint64_t piece = std::max<std::uint64_t>(1, kLoadChunk / rb);
    pool_->run(static_cast<std::size_t>((count + piece - 1) / piece), [&](std::size_t i) {
        const std::uint64_t row = first + i * piece;
        const std::uint64_t n   = std::min(piece, first + count - row);
        read(row * rb, destination + (row - first) * rb, static_cast<std::size_t>(n * rb));
    });
}

void NgramTableReader::load_resident() {
    resident_ = std::make_unique<Memory>(layout_.rows * layout_.row_bytes);
    load(0, layout_.rows, resident_->data());
    if (options_.lock) { resident_->lock(); }
    resident_rows_ = layout_.rows;
}

void NgramTableReader::load_hot() {
    const std::uint64_t rows   = layout_.rows;
    const std::uint64_t rb     = layout_.row_bytes;
    const std::uint64_t words  = (rows + 63) / 64;
    const std::uint64_t blocks = (rows + 511) / 512;
    const std::uint64_t index  = words * 8 + blocks * 4;
    if (options_.budget_bytes < index + rb) {
        throw std::invalid_argument("n-gram table: a RAM budget of " + mib(options_.budget_bytes) +
                                    " does not cover the hot rows' index of " + mib(index));
    }
    if (rows > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("n-gram table: a hot-row profile addresses 2^32 rows");
    }
    const std::uint64_t fit = (options_.budget_bytes - index) / rb;
    const std::size_t take =
        static_cast<std::size_t>(std::min<std::uint64_t>(options_.hot_rows.size(), fit));
    hot_bits_.assign(static_cast<std::size_t>(words), 0);
    for (std::size_t i = 0; i < take; ++i) {
        const std::uint64_t row = options_.hot_rows[i];
        if (row >= rows) {
            throw std::invalid_argument("n-gram table: the hot-row profile names row " +
                                        std::to_string(row) + " past the table");
        }
        std::uint64_t& word     = hot_bits_[row >> 6U];
        const std::uint64_t bit = 1ULL << (row & 63U);
        if ((word & bit) != 0) {
            throw std::invalid_argument("n-gram table: the hot-row profile names row " +
                                        std::to_string(row) + " twice");
        }
        word |= bit;
    }
    hot_blocks_.resize(static_cast<std::size_t>(blocks));
    std::uint64_t running = 0;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        hot_blocks_[b] = static_cast<std::uint32_t>(running);
        for (std::uint64_t w = b * 8; w < std::min(words, b * 8 + 8); ++w) {
            running += static_cast<std::uint64_t>(std::popcount(hot_bits_[w]));
        }
    }
    resident_rows_ = running;
    if (running == 0) { return; }
    resident_ = std::make_unique<Memory>(running * rb);
    // The hot rows in row order, which is their order in RAM.
    std::vector<std::uint32_t> by_slot;
    by_slot.reserve(static_cast<std::size_t>(running));
    for (std::uint64_t w = 0; w < words; ++w) {
        for (std::uint64_t bits = hot_bits_[w]; bits != 0; bits &= bits - 1) {
            by_slot.push_back(static_cast<std::uint32_t>(w * 64 + std::countr_zero(bits)));
        }
    }
    // Pieces of the table that hold hot rows: a dense piece is read whole and its hot rows copied
    // out, a sparse one reads its hot rows one by one.
    const std::uint64_t piece = std::max<std::uint64_t>(1, kLoadChunk / rb);
    std::uint8_t* const out   = resident_->data();
    pool_->run(static_cast<std::size_t>((rows + piece - 1) / piece), [&](std::size_t i) {
        const std::uint64_t first = i * piece;
        const std::uint64_t end   = std::min(rows, first + piece);
        const auto lo             = std::lower_bound(by_slot.begin(), by_slot.end(), first);
        const auto hi             = std::lower_bound(lo, by_slot.end(), end);
        if (lo == hi) { return; }
        if (std::uint64_t(hi - lo) * kSparseLoad < (end - first) * rb) {
            for (auto it = lo; it != hi; ++it) {
                read(*it * rb, out + std::uint64_t(it - by_slot.begin()) * rb,
                     static_cast<std::size_t>(rb));
            }
            return;
        }
        std::vector<std::uint8_t> rows_read(static_cast<std::size_t>((end - first) * rb));
        read(first * rb, rows_read.data(), rows_read.size());
        for (auto it = lo; it != hi; ++it) {
            std::memcpy(out + std::uint64_t(it - by_slot.begin()) * rb,
                        rows_read.data() + (*it - first) * rb, static_cast<std::size_t>(rb));
        }
    });
    if (options_.lock) { resident_->lock(); }
}

// The row's place among the resident rows, or -1 when it is not hot.
std::int64_t NgramTableReader::hot_slot(std::uint64_t row) const noexcept {
    const std::uint64_t word  = row >> 6U;
    const std::uint64_t bits  = hot_bits_[word];
    const std::uint64_t below = (1ULL << (row & 63U)) - 1;
    if (((bits >> (row & 63U)) & 1U) == 0) { return -1; }
    std::uint64_t slot = hot_blocks_[row >> 9U];
    for (std::uint64_t w = (row >> 9U) << 3U; w < word; ++w) {
        slot += static_cast<std::uint64_t>(std::popcount(hot_bits_[w]));
    }
    return static_cast<std::int64_t>(slot) + std::popcount(bits & below);
}

void NgramTableReader::submit(std::span<const std::uint64_t> row_ids, std::span<std::uint8_t> out) {
    if (pending_) { throw std::logic_error("n-gram table: a batch is in flight"); }
    const std::size_t rb = layout_.row_bytes;
    if (out.size() != row_ids.size() * rb) {
        throw std::invalid_argument("n-gram table: output size mismatch");
    }
    for (const std::uint64_t row : row_ids) {
        if (row >= layout_.rows) {
            throw std::out_of_range("n-gram table: row " + std::to_string(row) + " past " +
                                    std::to_string(layout_.rows));
        }
    }
    batch_start_ = std::chrono::steady_clock::now();
    misses_.clear();
    admissions_.clear();
    std::uint64_t hits = 0;
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        if (cache_ && cache_->copy(row_ids[i], out.data() + i * rb)) {
            ++hits;
            continue;
        }
        if (lookahead_ && lookahead_->copy(row_ids[i], out.data() + i * rb)) {
            ++hits;
            if (cache_) { admissions_.push_back(static_cast<std::uint32_t>(i)); }
            continue;
        }
        std::int64_t slot = -1;
        if (options_.residency == NgramResidency::Ram) {
            slot = static_cast<std::int64_t>(row_ids[i]);
        } else if (options_.residency == NgramResidency::RamHot && resident_) {
            slot = hot_slot(row_ids[i]);
        }
        if (slot < 0) {
            misses_.push_back(static_cast<std::uint32_t>(i));
            if (cache_) { admissions_.push_back(static_cast<std::uint32_t>(i)); }
            continue;
        }
        std::memcpy(out.data() + i * rb, resident_->data() + std::uint64_t(slot) * rb, rb);
        ++hits;
    }
    if (queue_) {
        reads_.clear();
        for (const auto i : misses_) {
            std::uint64_t offset = row_ids[i] * rb;
            std::size_t remaining = rb;
            auto* destination = out.data() + i * rb;
            auto segment = static_cast<std::size_t>(
                std::upper_bound(starts_.begin(), starts_.end(), offset) - starts_.begin() - 1);
            while (remaining != 0) {
                const auto& part = layout_.segments[segment];
                const auto local = offset - starts_[segment];
                const auto count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(remaining, part.bytes - local));
                reads_.push_back({segment, part.file_offset + local, {destination, count}});
                destination += count;
                offset += count;
                remaining -= count;
                ++segment;
            }
        }
    }
    rows_.fetch_add(row_ids.size(), std::memory_order_relaxed);
    resident_hits_.fetch_add(hits, std::memory_order_relaxed);
    batch_ids_ = row_ids;
    batch_out_ = out;
    pending_   = true;
    if (!queue_ && options_.io == NgramIo::Buffered && misses_.size() <= kInlineRows) {
        // Rows the page cache holds are read here at once: a decode pass's few dozen rows then
        // need no hand-off to the threads, whose wake-ups would cost more than the reads.
        std::size_t kept = 0;
        for (const auto i : misses_) {
            const std::uint64_t offset = row_ids[i] * rb;
            const auto segment         = static_cast<std::size_t>(
                std::upper_bound(starts_.begin(), starts_.end(), offset) - starts_.begin() - 1);
            const auto& part           = layout_.segments[segment];
            const std::uint64_t local  = offset - starts_[segment];
            if (local + rb > part.bytes ||
                !files_[segment]->read_resident(part.file_offset + local, out.data() + i * rb, rb)) {
                misses_[kept++] = i;
            }
        }
        misses_.resize(kept);
    }
    if (misses_.empty()) { return; }
    if (queue_) {
        pool_->start(1, [this](std::size_t) { queue_->read(reads_); });
        return;
    }
    // Groups of rows, a few per thread, so a prompt chunk's thousands of rows share the threads
    // without a hand-off per row.
    const std::size_t groups = std::min<std::size_t>(misses_.size(), 4 * options_.depth);
    const std::size_t per    = (misses_.size() + groups - 1) / groups;
    pool_->start(groups, [this, per](std::size_t group) {
        const std::size_t row_bytes = layout_.row_bytes;
        const std::size_t end       = std::min(misses_.size(), (group + 1) * per);
        for (std::size_t m = group * per; m < end; ++m) {
            const std::size_t i = misses_[m];
            read(batch_ids_[i] * row_bytes, batch_out_.data() + i * row_bytes, row_bytes);
        }
    });
}

void NgramTableReader::wait() {
    if (!pending_) { return; }
    pending_ = false;
    std::exception_ptr failure;
    try {
        pool_->finish();
    } catch (...) { failure = std::current_exception(); }
    if (!failure && cache_) {
        for (const std::size_t i : admissions_) {
            cache_->insert(batch_ids_[i], batch_out_.data() + i * layout_.row_bytes);
        }
        resident_rows_.store(cache_->size(), std::memory_order_relaxed);
    }
    const auto nanoseconds =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - batch_start_)
                                       .count());
    batches_.fetch_add(1, std::memory_order_relaxed);
    latency_ns_.fetch_add(nanoseconds, std::memory_order_relaxed);
    histogram_[latency_bucket(nanoseconds)].fetch_add(1, std::memory_order_relaxed);
    if (failure) {
        if (lookahead_) { lookahead_->discard(); }
        std::rethrow_exception(failure);
    }
}

void NgramTableReader::read_rows(std::span<const std::uint64_t> row_ids,
                                 std::span<std::uint8_t> out) {
    submit(row_ids, out);
    wait();
}

void NgramTableReader::prefetch(std::span<const std::uint64_t> row_ids) {
    for (const auto row : row_ids) {
        if (row >= layout_.rows) { throw std::out_of_range("n-gram prefetch: row past the table"); }
    }
    if (options_.residency == NgramResidency::Ram) { return; }
    std::vector<std::uint64_t> rows;
    rows.reserve(row_ids.size());
    for (const auto row : row_ids) {
        if (cache_ && cache_->contains(row)) { continue; }
        if (options_.residency == NgramResidency::RamHot && hot_slot(row) >= 0) { continue; }
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    if (lookahead_) {
        lookahead_->submit(rows);
        return;
    }
    // Coalesce ranges that touch the same page, including rows split between artifact parts.
    std::vector<Hints::Range> ranges;
    std::size_t previous = files_.size();
    std::uint64_t begin = 0, end = 0;
    for (const auto row : rows) {
        std::uint64_t offset = row * layout_.row_bytes;
        std::uint64_t remaining = layout_.row_bytes;
        auto segment = static_cast<std::size_t>(
            std::upper_bound(starts_.begin(), starts_.end(), offset) - starts_.begin() - 1);
        while (remaining != 0) {
            const auto& part = layout_.segments[segment];
            const std::uint64_t local = offset - starts_[segment];
            const std::uint64_t count = std::min(remaining, part.bytes - local);
            const std::uint64_t first = part.file_offset + local;
            if (segment == previous && first / kAlign <= end / kAlign) {
                end = std::max(end, first + count);
            } else {
                if (previous != files_.size()) { ranges.push_back({previous, begin, end - begin}); }
                previous = segment;
                begin = first;
                end = first + count;
            }
            offset += count;
            remaining -= count;
            ++segment;
        }
    }
    if (previous != files_.size()) { ranges.push_back({previous, begin, end - begin}); }
    if (ranges.size() > kInlineHints) {
        hints_->submit(std::move(ranges));
        return;
    }
    for (const auto& range : ranges) { files_[range.segment]->prefetch(range.begin, range.bytes); }
}

NgramTableStats NgramTableReader::counters() const {
    NgramTableStats out;
    out.rows          = rows_.load(std::memory_order_relaxed);
    out.resident_rows = resident_hits_.load(std::memory_order_relaxed);
    out.batches       = batches_.load(std::memory_order_relaxed);
    out.read_seconds  = static_cast<double>(latency_ns_.load(std::memory_order_relaxed)) * 1e-9;
    for (std::size_t b = 0; b < out.latency_histogram.size(); ++b) {
        out.latency_histogram[b] = histogram_[b].load(std::memory_order_relaxed);
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
