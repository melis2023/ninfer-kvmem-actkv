#include "models/qwen4_exp/expert_stream.h"

#include "models/qwen4_exp/read_pool.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

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

constexpr std::uint64_t kAlign = 256;
// Two halves of the page-locked ring: one fills from the files while the other's copies run.
constexpr std::size_t kStagingHalf = 128ULL << 20;
// Reads in flight at once: an NVMe drive reaches its bandwidth only with several requests queued.
constexpr std::size_t kReadThreads = 8;
// Zero bytes after every slot's down matrix. The expert matrix kernel reads whole 256-value K steps
// past a 640-value down row's end and decodes what follows as the down's blocks: another format's
// bytes there (the next slot's gate, or a larger down an earlier layer left) can hold a non-finite
// scale, which times the activation's zero padding is NaN.
constexpr std::uint64_t kTail = 256;

std::uint64_t aligned(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// A read-only file read at explicit offsets, through the page cache.
class File {
public:
    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("expert stream: cannot open " + path.string());
        }
#else
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) {
            throw std::runtime_error("expert stream: cannot open " + path.string() + ": " +
                                     std::strerror(errno));
        }
#    if defined(POSIX_FADV_RANDOM)
        (void)::posix_fadvise(descriptor_, 0, 0, POSIX_FADV_RANDOM);
#    endif
#endif
    }

    ~File() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if (descriptor_ >= 0) ::close(descriptor_);
#endif
    }

    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    void read(std::uint64_t offset, std::byte* destination, std::size_t bytes) const {
        while (bytes > 0) {
#ifdef _WIN32
            OVERLAPPED position{};
            position.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            const DWORD request = static_cast<DWORD>(
                std::min<std::size_t>(bytes, std::numeric_limits<DWORD>::max() / 2));
            DWORD done = 0;
            if (!ReadFile(handle_, destination, request, &done, &position) || done == 0) {
                throw std::runtime_error("expert stream: read failed");
            }
#else
            const ssize_t done =
                ::pread(descriptor_, destination, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) continue;
            if (done <= 0) {
                throw std::runtime_error(std::string("expert stream: read failed: ") +
                                         (done < 0 ? std::strerror(errno) : "end of file"));
            }
#endif
            destination += done;
            offset += static_cast<std::uint64_t>(done);
            bytes -= static_cast<std::size_t>(done);
        }
    }

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};

// One read of a staged batch.
struct Read {
    const File* file       = nullptr;
    std::uint64_t offset   = 0;
    std::byte* destination = nullptr;
    std::size_t bytes      = 0;
};

struct Slot {
    std::int32_t layer  = -1;
    std::int32_t expert = -1;
    std::uint64_t call  = 0; // the last prepare() that needed it
    bool referenced     = false;
    // Bytes from the start of the down region that a down has written: past a smaller down, the
    // tail must be zeroed again.
    std::uint64_t down_written = 0;
};

struct Pool {
    DeviceBuffer storage;
    std::uint64_t slot_bytes = 0;
    std::array<std::uint64_t, 3> offset{};
    std::vector<Slot> slots;
    std::size_t hand = 0; // CLOCK
};

} // namespace

struct ExpertStream::Impl {
    struct Layer {
        ExpertStreamLayer spec;
        std::vector<std::int32_t> slot_of;
        std::array<std::vector<const void*>, 3> mirror;
        std::unique_ptr<PinnedHostBuffer> table_staging;
    };

    DeviceContext& device;
    std::vector<std::unique_ptr<File>> files;
    std::vector<Layer> layers;
    std::map<std::size_t, Pool> pools;
    std::unique_ptr<PinnedHostBuffer> staging;
    std::array<cudaEvent_t, 2> half_done{};
    std::array<std::size_t, 3> half_rank{};
    std::size_t half         = 0; // the half being filled
    std::size_t cursor       = 0; // bytes used in it
    cudaStream_t half_stream = nullptr;
    std::uint64_t call       = 0;
    ExpertStreamStats stats;
    ReadPool reads{kReadThreads - 1}; // the caller reads too

    // The batch being staged in the current half: its reads, then the copies out of the half (a
    // copy without a source zeroes its target).
    struct Copy {
        void* target            = nullptr;
        const std::byte* source = nullptr;
        std::size_t bytes       = 0;
    };

    std::vector<Read> batch_reads;
    std::vector<Copy> batch_copies;

    Impl(DeviceContext& d, const std::vector<std::filesystem::path>& paths,
         std::vector<ExpertStreamLayer> specs, std::span<const std::uint64_t> bytes_by_rank)
        : device(d) {
        for (const auto& path : paths) { files.push_back(std::make_unique<File>(path)); }
        std::map<std::size_t, std::array<std::uint64_t, 3>> widest;
        for (const auto& spec : specs) {
            auto& sizes = widest[spec.rank];
            for (int k = 0; k < 3; ++k) {
                for (const auto& location : spec.experts[k]) {
                    sizes[k] = std::max(sizes[k], location.bytes);
                }
            }
        }
        for (const auto& [rank, sizes] : widest) {
            Pool pool;
            pool.offset     = {0, aligned(sizes[0]), aligned(sizes[0]) + aligned(sizes[1])};
            pool.slot_bytes = aligned(sizes[0]) + aligned(sizes[1]) + aligned(sizes[2] + kTail);
            const std::uint64_t budget = rank < bytes_by_rank.size() ? bytes_by_rank[rank] : 0;
            pool.slots.resize(static_cast<std::size_t>(budget / pool.slot_bytes));
            if (!pool.slots.empty()) {
                // Zeroed, so each slot's tail stays zero: copies write the matrices only.
                RankBinding bind(device, rank);
                pool.storage = DeviceBuffer(pool.slots.size() * pool.slot_bytes);
                CUDA_CHECK(cudaMemset(pool.storage.p, 0, pool.storage.bytes));
            }
            stats.slots += static_cast<std::uint32_t>(pool.slots.size());
            pools.emplace(rank, std::move(pool));
        }
        for (auto& spec : specs) {
            Layer layer;
            const std::size_t experts = spec.experts[0].size();
            layer.slot_of.assign(experts, -1);
            for (auto& mirror : layer.mirror) { mirror.assign(experts, nullptr); }
            layer.table_staging = std::make_unique<PinnedHostBuffer>(3 * experts * sizeof(void*));
            layer.spec          = std::move(spec);
            layers.push_back(std::move(layer));
        }
        staging = std::make_unique<PinnedHostBuffer>(2 * kStagingHalf);
        for (auto& event : half_done) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
    }

    ~Impl() {
        for (auto& event : half_done) {
            if (event != nullptr) { cudaEventDestroy(event); }
        }
    }

    // Closes the half being filled behind the copies queued from it and starts the other one,
    // once the copies that last read it have finished.
    void switch_half() {
        if (half_stream != nullptr) {
            RankBinding bind(device, half_rank[half]);
            CUDA_CHECK(cudaEventRecord(half_done[half], half_stream));
        }
        half   = 1 - half;
        cursor = 0;
        CUDA_CHECK(cudaEventSynchronize(half_done[half]));
    }

    // Reads the staged batch in parallel and queues its copies on `stream`.
    void flush(cudaStream_t stream) {
        reads.run(batch_reads.size(), [&](std::size_t i) {
            const Read& read = batch_reads[i];
            read.file->read(read.offset, read.destination, read.bytes);
        });
        for (const Copy& copy : batch_copies) {
            if (copy.source == nullptr) {
                CUDA_CHECK(cudaMemsetAsync(copy.target, 0, copy.bytes, stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(copy.target, copy.source, copy.bytes,
                                           cudaMemcpyHostToDevice, stream));
            }
        }
        batch_reads.clear();
        batch_copies.clear();
    }

    std::byte* stage(std::size_t bytes, std::size_t rank, cudaStream_t stream) {
        if (bytes > kStagingHalf) {
            throw std::logic_error("expert stream: an expert exceeds the staging half");
        }
        if (cursor + bytes > kStagingHalf || (half_stream != nullptr && half_stream != stream)) {
            // The half closes behind the batch staged in it.
            flush(half_stream != nullptr ? half_stream : stream);
            switch_half();
        }
        half_stream     = stream;
        half_rank[half] = rank;
        std::byte* out  = static_cast<std::byte*>(staging->data()) + half * kStagingHalf + cursor;
        cursor += aligned(bytes);
        return out;
    }

    std::size_t victim(Pool& pool) {
        // CLOCK: a slot used since the hand last passed gets another round, a slot this call
        // needs is never taken. The caller has checked that the call fits the pool.
        for (;;) {
            Slot& slot              = pool.slots[pool.hand];
            const std::size_t index = pool.hand;
            pool.hand               = (pool.hand + 1) % pool.slots.size();
            if (slot.call == call && slot.expert >= 0) { continue; }
            if (slot.referenced) {
                slot.referenced = false;
                continue;
            }
            return index;
        }
    }

    void prepare(std::size_t index, std::span<const std::int32_t> ids) {
        Layer& layer = layers.at(index);
        Pool& pool   = pools.at(layer.spec.rank);
        ++call;
        std::vector<std::int32_t> needed;
        needed.reserve(ids.size());
        const auto experts = static_cast<std::int32_t>(layer.slot_of.size());
        for (const std::int32_t id : ids) {
            if (id >= 0 && id < experts) { needed.push_back(id); }
        }
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
        if (needed.size() > pool.slots.size()) {
            throw std::runtime_error("expert stream: one layer routes to " +
                                     std::to_string(needed.size()) +
                                     " experts but the device "
                                     "cache holds " +
                                     std::to_string(pool.slots.size()));
        }
        std::vector<std::int32_t> missing;
        for (const std::int32_t expert : needed) {
            ++stats.routes;
            const std::int32_t slot = layer.slot_of[expert];
            if (slot >= 0) {
                ++stats.hits;
                pool.slots[slot].call       = call;
                pool.slots[slot].referenced = true;
            } else {
                missing.push_back(expert);
            }
        }
        if (missing.empty()) { return; }
        RankBinding bind(device, layer.spec.rank);
        const cudaStream_t stream = layer.spec.stream;
        const auto start          = std::chrono::steady_clock::now();
        for (const std::int32_t expert : missing) {
            const std::size_t s = victim(pool);
            Slot& slot          = pool.slots[s];
            if (slot.expert >= 0) { layers[slot.layer].slot_of[slot.expert] = -1; }
            auto* base = static_cast<std::byte*>(pool.storage.p) + s * pool.slot_bytes;
            for (int k = 0; k < 3; ++k) {
                const ExpertLocation& location = layer.spec.experts[k][expert];
                std::byte* buffer              = stage(location.bytes, layer.spec.rank, stream);
                std::uint64_t at               = 0;
                for (const auto& run : location.runs) {
                    if (run.bytes == 0) { continue; }
                    batch_reads.push_back({files.at(run.file).get(), run.offset, buffer + at,
                                           static_cast<std::size_t>(run.bytes)});
                    at += run.bytes;
                }
                std::byte* target = base + pool.offset[k];
                batch_copies.push_back({target, buffer, static_cast<std::size_t>(location.bytes)});
                if (k == 2) {
                    if (slot.down_written > location.bytes) {
                        batch_copies.push_back({target + location.bytes, nullptr,
                                                static_cast<std::size_t>(std::min(
                                                    slot.down_written - location.bytes, kTail))});
                    }
                    slot.down_written = std::max(slot.down_written, location.bytes);
                }
                layer.mirror[k][expert] = target;
                stats.read_bytes += location.bytes;
            }
            slot.layer            = static_cast<std::int32_t>(index);
            slot.expert           = expert;
            slot.call             = call;
            slot.referenced       = true;
            layer.slot_of[expert] = static_cast<std::int32_t>(s);
        }
        flush(stream);
        auto* tables = static_cast<std::byte*>(layer.table_staging->data());
        for (int k = 0; k < 3; ++k) {
            const std::size_t bytes = layer.mirror[k].size() * sizeof(void*);
            std::memcpy(tables + k * bytes, layer.mirror[k].data(), bytes);
            CUDA_CHECK(cudaMemcpyAsync(layer.spec.tables[k], tables + k * bytes, bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        stats.read_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

ExpertStream::ExpertStream(DeviceContext& device, const std::vector<std::filesystem::path>& files,
                           std::vector<ExpertStreamLayer> layers,
                           std::span<const std::uint64_t> bytes_by_rank)
    : impl_(std::make_unique<Impl>(device, files, std::move(layers), bytes_by_rank)) {}

ExpertStream::~ExpertStream() = default;

void ExpertStream::prepare(std::size_t layer, std::span<const std::int32_t> ids) {
    impl_->prepare(layer, ids);
}

ExpertStreamStats ExpertStream::stats() const noexcept { return impl_->stats; }

} // namespace ninfer::models::qwen4_exp
