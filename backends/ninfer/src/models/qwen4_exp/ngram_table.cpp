#include "models/qwen4_exp/ngram_table.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

// Rows in flight at once on disk residency.
constexpr std::size_t kReadThreads = 8;

} // namespace

struct NgramTableReader::File {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int descriptor = -1;
#endif

    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        // Random access: the cache manager does not read ahead past each row.
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("n-gram table: cannot open " + path.string());
        }
#else
        descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw std::runtime_error("n-gram table: cannot open " + path.string() + ": " +
                                     std::strerror(errno));
        }
#    if defined(POSIX_FADV_RANDOM)
        (void)::posix_fadvise(descriptor, 0, 0, POSIX_FADV_RANDOM);
#    endif
#endif
    }

    ~File() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (descriptor >= 0) ::close(descriptor);
#endif
    }

    [[nodiscard]] std::uint64_t size() const {
#ifdef _WIN32
        LARGE_INTEGER bytes{};
        if (!GetFileSizeEx(handle, &bytes)) throw std::runtime_error("n-gram table: cannot size");
        return static_cast<std::uint64_t>(bytes.QuadPart);
#else
        struct stat status {};
        if (::fstat(descriptor, &status) != 0) throw std::runtime_error("n-gram table: cannot size");
        return static_cast<std::uint64_t>(status.st_size);
#endif
    }

    // Reads exactly `bytes` at `offset`.
    void read(std::uint64_t offset, std::uint8_t* destination, std::size_t bytes) const {
        while (bytes > 0) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            const DWORD request = static_cast<DWORD>(
                std::min<std::size_t>(bytes, std::numeric_limits<DWORD>::max() / 2));
            DWORD done = 0;
            if (!ReadFile(handle, destination, request, &done, &position) || done == 0) {
                throw std::runtime_error("n-gram table: read failed");
            }
#else
            const ssize_t done = ::pread(descriptor, destination, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) continue;
            if (done <= 0) {
                throw std::runtime_error(std::string("n-gram table: read failed: ") +
                                         (done < 0 ? std::strerror(errno) : "end of file"));
            }
#endif
            destination += done;
            offset += static_cast<std::uint64_t>(done);
            bytes -= static_cast<std::size_t>(done);
        }
    }
};

NgramTableReader::NgramTableReader(NgramTableLayout layout, NgramResidency residency)
    : layout_(std::move(layout)), residency_(residency) {
    if (layout_.row_bytes == 0 || layout_.rows == 0 || layout_.segments.empty()) {
        throw std::invalid_argument("n-gram table: empty layout");
    }
    if (layout_.rows > std::numeric_limits<std::uint64_t>::max() / layout_.row_bytes) {
        throw std::invalid_argument("n-gram table: payload size overflows");
    }
    const std::uint64_t table_bytes = layout_.rows * layout_.row_bytes;
    std::uint64_t covered           = 0;
    for (const auto& segment : layout_.segments) {
        files_.push_back(std::make_unique<File>(segment.path));
        if (segment.bytes == 0 ||
            segment.file_offset > std::numeric_limits<std::uint64_t>::max() - segment.bytes ||
            files_.back()->size() < segment.file_offset + segment.bytes) {
            throw std::runtime_error("n-gram table: " + segment.path.string() + " holds " +
                                     std::to_string(files_.back()->size()) +
                                     " bytes, short of its table segment");
        }
        starts_.push_back(covered);
        covered += segment.bytes;
    }
    if (covered < table_bytes) {
        throw std::runtime_error("n-gram table: the files hold " + std::to_string(covered) +
                                 " table bytes, the table needs " + std::to_string(table_bytes));
    }
    if (residency_ == NgramResidency::Ram) {
        if (table_bytes > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("n-gram table: payload exceeds the address space");
        }
        resident_.resize(static_cast<std::size_t>(table_bytes));
        read(0, resident_.data(), resident_.size());
    } else {
        pool_ = std::make_unique<ReadPool>(kReadThreads - 1); // the caller reads too
    }
}

NgramTableReader::~NgramTableReader() = default;

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

void NgramTableReader::read_rows(std::span<const std::uint64_t> row_ids,
                                 std::span<std::uint8_t> out) const {
    const std::size_t row_bytes = layout_.row_bytes;
    if (out.size() != row_ids.size() * row_bytes) {
        throw std::invalid_argument("n-gram table: output size mismatch");
    }
    for (const std::uint64_t row : row_ids) {
        if (row >= layout_.rows) {
            throw std::out_of_range("n-gram table: row " + std::to_string(row) + " past " +
                                    std::to_string(layout_.rows));
        }
    }
    if (residency_ == NgramResidency::Ram) {
        for (std::size_t i = 0; i < row_ids.size(); ++i) {
            std::memcpy(out.data() + i * row_bytes, resident_.data() + row_ids[i] * row_bytes,
                        row_bytes);
        }
        return;
    }
    // Groups of rows, a few per thread, so a prompt chunk's thousands of rows share the threads
    // without a hand-off per row.
    const std::size_t groups = std::min<std::size_t>(row_ids.size(), 4 * kReadThreads);
    const std::size_t per    = (row_ids.size() + groups - 1) / groups;
    pool_->run(groups, [&](std::size_t group) {
        const std::size_t end = std::min(row_ids.size(), (group + 1) * per);
        for (std::size_t i = group * per; i < end; ++i) {
            read(row_ids[i] * row_bytes, out.data() + i * row_bytes, row_bytes);
        }
    });
}

} // namespace ninfer::models::qwen4_exp
