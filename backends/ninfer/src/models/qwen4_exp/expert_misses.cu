#include "models/qwen4_exp/expert_misses.h"

#include "core/cpu_spin_pool.h"
#include "core/expert_split.h"
#include "core/file_read_error.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cuda/atomic>
#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <unistd.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

constexpr int kTop            = 10;
constexpr int kHidden         = 2560;
constexpr int kWidth          = 640;
constexpr int kMaxExperts     = 512;
constexpr int kSplitThreads   = kMaxExperts / 2; // two expert flags a thread
constexpr std::uint64_t kTail = 256;
constexpr std::uint64_t kAlign = 256;
constexpr int kRanges          = int(std::tuple_size_v<ExpertCache::Ranges>);

std::uint64_t aligned(std::uint64_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

using Atomic = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>;

// One device's doorbell in pinned host memory, read and written by both sides.
struct Mailbox {
    alignas(64) std::uint32_t requested = 0; // device: the last posted request
    std::uint32_t layer = 0, tokens = 0, missing = 0;
    alignas(64) std::uint32_t prepared = 0; // host: the last request ready
};

// [ranges[2i], ranges[2i + 1]) for i < kRanges: a layer's cached blocks.
__device__ __forceinline__ bool in_ranges(std::uintptr_t at, const std::uintptr_t* ranges) {
    bool in = false;
#pragma unroll
    for (int i = 0; i < kRanges; ++i) { in = in || (at >= ranges[2 * i] && at < ranges[2 * i + 1]); }
    return in;
}

// Splits a call's pairs: a pair whose expert's gate entry lies in the layer's cached storage stays
// in `cached` (else -1); every other expert is listed once in `missing`. With `keep_ranks` above
// zero, an uncached expert that every token of the call ranks at or past that position of its top
// ten is left out instead: its pairs add nothing and the others keep their router weights. With
// missing experts the request is posted, after the pairs (and, for the CPU, the input) reach the
// host.
__global__ void __launch_bounds__(kSplitThreads)
    split_kernel(const std::int32_t* __restrict__ ids, int pairs, int experts,
                 const void* const* __restrict__ gate, const std::uintptr_t* __restrict__ ranges,
                 std::int32_t* __restrict__ cached, std::int32_t* __restrict__ host_ids,
                 std::int32_t* __restrict__ host_missing, std::int32_t* __restrict__ missing_ids,
                 std::int32_t* product_count, const __nv_bfloat16* __restrict__ m,
                 __nv_bfloat16* __restrict__ host_m, int m_values, Mailbox* box,
                 std::uint32_t layer, int tokens, int keep_ranks, std::uint32_t* need) {
    __shared__ unsigned char missing[kMaxExperts];
    __shared__ int warp_total[kSplitThreads / 32];
    __shared__ int count;
    for (int e = threadIdx.x; e < kMaxExperts; e += blockDim.x) { missing[e] = 0; }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        bool resident = false;
        if (e >= 0 && e < experts) {
            resident = in_ranges(reinterpret_cast<std::uintptr_t>(gate[e]), ranges);
            // Pairs come in rank order, ten a token (the route's top ten, largest weight first).
            if (!resident && (keep_ranks == 0 || p % kTop < keep_ranks)) { missing[e] = 1; }
        }
        cached[p]      = resident ? e : -1;
        missing_ids[p] = -1; // the service rewrites the missing pairs' entries of a request
    }
    __syncthreads();
    // The missing experts in ascending order: thread t holds experts 2t and 2t + 1, a block scan
    // places them.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool first = missing[2 * threadIdx.x] != 0, second = missing[2 * threadIdx.x + 1] != 0;
    const int own    = int(first) + int(second);
    int scan         = own;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFFFFFFu, scan, offset);
        if (lane >= offset) { scan += other; }
    }
    if (lane == 31) { warp_total[warp] = scan; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < kSplitThreads / 32; ++w) {
            const int here = warp_total[w];
            warp_total[w]  = total;
            total += here;
        }
        count = total;
    }
    __syncthreads();
    if (count == 0) {
        // No request: stage two adds nothing.
        if (threadIdx.x == 0) {
            *product_count = 0;
            *need          = 0;
        }
        return;
    }
    const int at = warp_total[warp] + scan - own;
    if (first) { host_missing[at] = 2 * threadIdx.x; }
    if (second) { host_missing[at + int(first)] = 2 * threadIdx.x + 1; }
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) { host_ids[p] = ids[p]; }
    if (host_m != nullptr) {
        for (int i = threadIdx.x; i < m_values; i += blockDim.x) { host_m[i] = m[i]; }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint32_t sequence = Atomic(box->requested).load(cuda::memory_order_relaxed) + 1;
        box->layer   = layer;
        box->tokens  = std::uint32_t(tokens);
        box->missing = std::uint32_t(count);
        *need        = sequence;
        Atomic(box->requested).store(sequence, cuda::memory_order_release);
    }
}

__global__ void wait_kernel(Mailbox* box, const std::uint32_t* need) {
    const std::uint32_t sequence = *need;
    if (sequence == 0) { return; }
    while (Atomic(box->prepared).load(cuda::memory_order_acquire) != sequence) { __nanosleep(128); }
}

// One thread per token: its pairs whose expert is cached keep their id, with the token's weights
// renormalized to its whole routed sum; the others become -1 with weight 0.
__global__ void drop_missing_kernel(const std::int32_t* __restrict__ ids,
                                    const float* __restrict__ weights, int tokens, int experts,
                                    const void* const* __restrict__ gate,
                                    const std::uintptr_t* __restrict__ ranges,
                                    std::int32_t* __restrict__ kept_ids,
                                    float* __restrict__ kept_weights) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) { return; }
    bool resident[kTop];
    float all = 0.0f, kept = 0.0f;
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
        const int e = ids[t * kTop + j];
        resident[j] = false;
        if (e >= 0 && e < experts) {
            resident[j] = in_ranges(reinterpret_cast<std::uintptr_t>(gate[e]), ranges);
        }
        all += weights[t * kTop + j];
        kept += resident[j] ? weights[t * kTop + j] : 0.0f;
    }
    const float scale = kept > 0.0f ? all / kept : 0.0f;
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
        kept_ids[t * kTop + j]     = resident[j] ? ids[t * kTop + j] : -1;
        kept_weights[t * kTop + j] = resident[j] ? weights[t * kTop + j] * scale : 0.0f;
    }
}

// A device table entry change of an admission: `expert` now reads `at`.
struct TablePatch {
    std::int32_t expert = -1;
    const void* at[3]   = {};
};

__global__ void patch_kernel(const void** gate, const void** up, const void** down,
                             const TablePatch* patches, int count) {
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const TablePatch patch = patches[i];
        gate[patch.expert]     = patch.at[0];
        up[patch.expert]       = patch.at[1];
        down[patch.expert]     = patch.at[2];
    }
}

bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

class File {
public:
    explicit File(const std::filesystem::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("expert misses: cannot open " + path.string());
        }
#else
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) {
            throw std::runtime_error("expert misses: cannot open " + path.string() + ": " +
                                     std::strerror(errno));
        }
#endif
    }
    ~File() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) { CloseHandle(handle_); }
#else
        if (descriptor_ >= 0) { ::close(descriptor_); }
#endif
    }
    File(const File&)            = delete;
    File& operator=(const File&) = delete;

    void read(std::uint64_t offset, std::byte* out, std::size_t bytes) const {
        while (bytes > 0) {
#ifdef _WIN32
            OVERLAPPED at{};
            at.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            at.OffsetHigh = static_cast<DWORD>(offset >> 32U);
            DWORD done    = 0;
            const DWORD request =
                static_cast<DWORD>(std::min<std::size_t>(bytes, std::size_t(1) << 30));
            if (!ReadFile(handle_, out, request, &done, &at) || done == 0) {
                throw FileReadError("expert misses: read failed");
            }
#else
            const ssize_t done = ::pread(descriptor_, out, bytes, static_cast<off_t>(offset));
            if (done < 0 && errno == EINTR) { continue; }
            if (done <= 0) {
                throw FileReadError(std::string("expert misses: read failed: ") +
                                    (done < 0 ? std::strerror(errno) : "end of file"));
            }
#endif
            out += done;
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

} // namespace

struct ExpertMisses::Impl {
    struct Rank {
        bool used = false;
        std::unique_ptr<PinnedHostBuffer> box_memory;
        Mailbox* box = nullptr;
        DeviceBuffer cached_ids, missing_device, staging;
        std::unique_ptr<PinnedHostBuffer> need; // the call's request, 0 without one
        // Host-written stage-two inputs, read by the GPU across the bus.
        std::unique_ptr<PinnedHostBuffer> tables;   // [3][512] expert pointers
        std::unique_ptr<PinnedHostBuffer> missing_ids, products, product_pairs, product_count;
        std::unique_ptr<PinnedHostBuffer> host_ids, host_missing, host_m;
        std::unique_ptr<PinnedHostBuffer> file_staging; // disk experts, two halves
        std::unique_ptr<PinnedHostBuffer> patches;      // one request's table changes
        cudaStream_t copies    = nullptr;
        cudaEvent_t copying    = nullptr; // balance only: before a request's copies
        cudaEvent_t copied     = nullptr; // after them (timed for the balance)
        std::uint32_t served   = 0;
        // Balance: this device's copy and CPU times.
        ExpertSplit split;
    };

    DeviceContext& device;
    std::vector<MissLayer> layers;
    ExpertCache* cache = nullptr;
    // Per layer: its cached blocks as four device words, read by the kernels (captured graphs
    // see a change).
    std::vector<DeviceBuffer> ranges;
    std::vector<std::unique_ptr<File>> files;
    MissOptions options;
    std::vector<Rank> ranks;
    std::uint32_t pair_capacity = 0;
    std::uint64_t half_bytes = 0, file_half_bytes = 0;
    std::unique_ptr<CpuSpinPool> pool;
    ops::GgufCpuBackend backend = ops::GgufCpuBackend::Automatic;
    std::uint32_t cpu_threads   = 0; // the pool's workers and the service thread
    bool disk = false;

    // CPU work of one request.
    struct CpuExpert {
        std::int32_t expert = -1;
        std::vector<std::int32_t> pairs, tokens; // pair index and its token, per column
        std::array<const std::uint8_t*, 3> bytes{};
        ops::GgufCpuActivation input, input_up, middle; // input_up: an up of another format
        std::vector<float> gate, up;
        std::vector<std::uint16_t> middle_bf16;
        std::int32_t first = 0; // its first pair's product row
        bool every_token   = false; // routed by every token: reads the shared inputs
    };

    // A 512-value piece of a token's input (`up`: for the up format), or a pair's whole middle.
    struct Piece {
        std::uint32_t job = 0;
        int token = 0, part = 0;
        bool up = false;
    };

    // Rows per task are a multiple of 16: 64 bytes of a task's FP32 outputs.
    static constexpr int kRowQuantum = 16;
    std::vector<CpuExpert> cpu_experts;
    ops::GgufCpuActivation shared, shared_up; // the request's tokens, quantized once per format
    std::vector<Piece> pieces;
    std::vector<RowPart> parts;
    std::vector<RowTask> tasks;

    mutable std::mutex stats_mutex;
    MissStats counters;
    std::atomic<std::int64_t> deadline{0};
    std::atomic<bool> sleeping{false}, stopping{false};
    std::mutex wake_mutex;
    std::condition_variable wake;
    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::thread service;

    Impl(DeviceContext& d, std::vector<MissLayer> l, std::vector<std::filesystem::path> paths,
         MissOptions o, ExpertCache* c)
        : device(d), layers(std::move(l)), cache(c), options(o), ranks(d.size()) {
        if (options.token_capacity == 0 || options.token_capacity > 64 || layers.empty() ||
            !std::isfinite(options.cpu_share) || options.cpu_share < 0 || options.cpu_share > 1 ||
            (options.cpu_balance && options.cpu_share != 0) ||
            options.tail_skip_rank >= std::uint32_t(kTop)) {
            throw std::invalid_argument("expert misses: invalid capacity, CPU share or tail skip");
        }
        pair_capacity = options.token_capacity * kTop;
        std::uint64_t largest = 0;
        for (const auto& layer : layers) {
            if (layer.rank >= ranks.size() || layer.experts < kTop || layer.experts > kMaxExperts ||
                !layer.gate_table) {
                throw std::invalid_argument("expert misses: invalid layer");
            }
            std::uint64_t bytes = 0;
            for (const auto& p : layer.projections) { bytes += aligned(std::uint64_t(p.expert_bytes)); }
            largest = std::max(largest, bytes + kTail);
            disk    = disk || !layer.projections[0].located.empty();
            ranks[layer.rank].used = true;
            if (layer.admit && (cache == nullptr || !layer.projections[0].located.empty())) {
                throw std::invalid_argument("expert misses: admission needs a cache of host experts");
            }
        }
        for (const auto& layer : layers) {
            RankBinding bind(device, layer.rank);
            ranges.emplace_back(2 * kRanges * sizeof(std::uintptr_t));
            write_ranges(ranges.back(), layer.resident);
        }
        for (const auto& path : paths) { files.push_back(std::make_unique<File>(path)); }
        half_bytes = std::max<std::uint64_t>(options.staging_bytes / 2, largest * 2);
        half_bytes = half_bytes / kAlign * kAlign;
        file_half_bytes = disk ? half_bytes : 0;
        // A CPU share needs a layer whose formats have CPU products.
        const bool cpu = (options.cpu_share > 0 || options.cpu_balance) &&
                         std::any_of(layers.begin(), layers.end(), [](const MissLayer& layer) {
                             return std::all_of(layer.projections.begin(), layer.projections.end(),
                                                [](const auto& p) { return ops::gguf_cpu_supported(p.format); });
                         });
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            Rank& rank = ranks[r];
            if (!rank.used) { continue; }
            RankBinding bind(device, r);
            rank.box_memory = std::make_unique<PinnedHostBuffer>(sizeof(Mailbox));
            rank.box        = ::new (rank.box_memory->data()) Mailbox();
            rank.cached_ids     = DeviceBuffer(std::size_t(pair_capacity) * 4);
            rank.missing_device = DeviceBuffer(std::size_t(pair_capacity) * 4);
            rank.staging    = DeviceBuffer(2 * half_bytes);
            rank.staging.fill(0);
            rank.need          = std::make_unique<PinnedHostBuffer>(64);
            *static_cast<std::uint32_t*>(rank.need->data()) = 0;
            rank.tables        = std::make_unique<PinnedHostBuffer>(3 * kMaxExperts * sizeof(void*));
            std::memset(rank.tables->data(), 0, rank.tables->size());
            rank.missing_ids   = std::make_unique<PinnedHostBuffer>(std::size_t(pair_capacity) * 4);
            rank.product_pairs = std::make_unique<PinnedHostBuffer>(std::size_t(pair_capacity) * 4);
            rank.product_count = std::make_unique<PinnedHostBuffer>(64);
            rank.products      = std::make_unique<PinnedHostBuffer>(
                cpu ? std::size_t(pair_capacity) * kHidden * 4 : 64);
            rank.host_ids      = std::make_unique<PinnedHostBuffer>(std::size_t(pair_capacity) * 4);
            rank.host_missing  = std::make_unique<PinnedHostBuffer>(kMaxExperts * 4);
            rank.host_m        = cpu ? std::make_unique<PinnedHostBuffer>(
                                    std::size_t(options.token_capacity) * kHidden * 2)
                                     : nullptr;
            if (disk) { rank.file_staging = std::make_unique<PinnedHostBuffer>(2 * file_half_bytes); }
            rank.patches = std::make_unique<PinnedHostBuffer>(2 * kMaxExperts * sizeof(TablePatch));
            CUDA_CHECK(cudaStreamCreateWithFlags(&rank.copies, cudaStreamNonBlocking));
            // Timed only for the balance, which reads the copies' durations.
            const unsigned timing = options.cpu_balance ? cudaEventDefault : cudaEventDisableTiming;
            if (options.cpu_balance) { CUDA_CHECK(cudaEventCreate(&rank.copying)); }
            CUDA_CHECK(cudaEventCreateWithFlags(&rank.copied, timing));
            // Load every kernel the misses launch now: a lazy load while a layer waits on the
            // device for this service could wait for that layer (see the executor's pass()).
            patch_kernel<<<1, 32, 0, rank.copies>>>(nullptr, nullptr, nullptr, nullptr, 0);
            wait_kernel<<<1, 1, 0, rank.copies>>>(rank.box, static_cast<const std::uint32_t*>(rank.need->data()));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(rank.copies));
        }
        if (cpu) {
            // By default one thread per physical core, this service's included; pinned, since the
            // products read the experts at memory speed and a core's two threads share its loads.
            const std::uint32_t threads =
                options.cpu_threads != 0 ? options.cpu_threads : pool_cpu_count();
            pool        = std::make_unique<CpuSpinPool>(threads, true);
            backend     = ops::gguf_cpu_backend_selected(ops::GgufCpuBackend::Automatic);
            cpu_threads = threads;
        }
        service = std::thread([this] { serve(); });
    }

    ~Impl() {
        try {
            device.synchronize();
        } catch (...) {}
        {
            std::lock_guard lock(wake_mutex);
            stopping.store(true);
        }
        wake.notify_all();
        if (service.joinable()) { service.join(); }
        for (auto& rank : ranks) {
            if (rank.copies) { cudaStreamDestroy(rank.copies); }
            if (rank.copying) { cudaEventDestroy(rank.copying); }
            if (rank.copied) { cudaEventDestroy(rank.copied); }
        }
    }

    static void write_ranges(DeviceBuffer& buffer, const ExpertCache::Ranges& resident) {
        std::uintptr_t words[2 * kRanges];
        for (int i = 0; i < kRanges; ++i) {
            words[2 * i]     = resident[std::size_t(i)].first;
            words[2 * i + 1] = resident[std::size_t(i)].second;
        }
        CUDA_CHECK(cudaMemcpy(buffer.p, words, sizeof(words), cudaMemcpyHostToDevice));
    }

    void remember_failure() noexcept {
        std::lock_guard lock(failure_mutex);
        if (!failure) { failure = std::current_exception(); }
    }

    // Reads one projection of a disk expert into `host_out`.
    void fetch(const MissLayer::Projection& projection, std::int32_t expert, std::byte* host_out) {
        const ExpertLocation& location = projection.located[std::size_t(expert)];
        std::uint64_t at = 0;
        for (const auto& run : location.runs) {
            if (run.bytes == 0) { continue; }
            files.at(run.file)->read(run.offset, host_out + at, std::size_t(run.bytes));
            at += run.bytes;
        }
    }

    bool cpu_capable(const MissLayer& layer) const {
        for (const auto& p : layer.projections) {
            if (!ops::gguf_cpu_supported(p.format)) { return false; }
        }
        return pool != nullptr;
    }

    // The CPU's share of a request in four pool phases of about three tasks per thread: the
    // request's tokens quantized once for the gate (and up) format, every expert's gate and up rows
    // (while the calling thread runs `side`, the request's copies), the BF16 middle quantized for
    // down, and the down rows straight into the products, the experts' pairs in turn. An expert
    // routed by every token reads the shared inputs; any other copies its tokens' columns. Returns
    // the microseconds the phases took, `side` left out.
    template <class Side>
    double compute_cpu(const MissLayer& layer, Rank& rank, Side&& side) {
        using Clock                   = std::chrono::steady_clock;
        const auto started            = Clock::now();
        const auto* m                 = static_cast<const std::uint16_t*>(rank.host_m->data());
        auto* products                = static_cast<float*>(rank.products->data());
        auto* product_pairs           = static_cast<std::int32_t*>(rank.product_pairs->data());
        const auto& gate_p            = layer.projections[0];
        const auto& up_p              = layer.projections[1];
        const auto& down_p            = layer.projections[2];
        const bool own_up             = up_p.format != gate_p.format;
        const int capacity            = int(options.token_capacity);
        constexpr int kGroupsPerPiece = 16; // 512 values
        constexpr int kHiddenPieces   = kHidden / 32 / kGroupsPerPiece;
        int token_count               = 0;
        for (const auto& job : cpu_experts) {
            for (const auto t : job.tokens) { token_count = std::max(token_count, int(t) + 1); }
        }
        if (shared.k() != kHidden) { shared = ops::GgufCpuActivation(capacity, kHidden); }
        shared.reset(gate_p.format, token_count);
        if (own_up) {
            if (shared_up.k() != kHidden) { shared_up = ops::GgufCpuActivation(capacity, kHidden); }
            shared_up.reset(up_p.format, token_count);
        }
        pieces.clear();
        for (int t = 0; t < token_count; ++t) {
            for (int piece = 0; piece < kHiddenPieces; ++piece) {
                pieces.push_back({0, t, piece, false});
                if (own_up) { pieces.push_back({0, t, piece, true}); }
            }
        }
        pool->run(pieces.size(), [&](std::size_t index) {
            const Piece& piece = pieces[index];
            (piece.up ? shared_up : shared)
                .quantize_bf16(piece.token, m + std::size_t(piece.token) * kHidden,
                               piece.part * kGroupsPerPiece, (piece.part + 1) * kGroupsPerPiece);
        });
        std::int32_t first = 0;
        for (auto& job : cpu_experts) {
            const int n     = static_cast<int>(job.tokens.size());
            job.every_token = n == token_count;
            for (int j = 0; j < n && job.every_token; ++j) {
                job.every_token = job.tokens[std::size_t(j)] == j;
            }
            if (!job.every_token) {
                if (job.input.k() != kHidden) {
                    job.input = ops::GgufCpuActivation(capacity, kHidden);
                }
                job.input.reset(gate_p.format, n);
                if (own_up) {
                    if (job.input_up.k() != kHidden) {
                        job.input_up = ops::GgufCpuActivation(capacity, kHidden);
                    }
                    job.input_up.reset(up_p.format, n);
                }
                for (int j = 0; j < n; ++j) {
                    job.input.copy_token(j, shared, job.tokens[std::size_t(j)]);
                    if (own_up) {
                        job.input_up.copy_token(j, shared_up, job.tokens[std::size_t(j)]);
                    }
                }
            }
            if (job.middle.k() != kWidth) { job.middle = ops::GgufCpuActivation(capacity, kWidth); }
            job.middle.reset(down_p.format, n);
            job.gate.resize(std::size_t(n) * kWidth);
            job.up.resize(std::size_t(n) * kWidth);
            job.middle_bf16.resize(std::size_t(n) * kWidth);
            job.first = first;
            for (int j = 0; j < n; ++j) { product_pairs[first + j] = job.pairs[std::size_t(j)]; }
            first += n;
        }
        const auto quantized = Clock::now();
        parts.clear();
        for (const auto& job : cpu_experts) {
            const double cost = ExpertSplit::cpu_work(1.0, int(job.tokens.size()));
            parts.push_back({kWidth, cost});
            parts.push_back({kWidth, cost});
        }
        tasks = row_tasks(parts, pool->threads(), kRowQuantum);
        pool->run(
            tasks.size(),
            [&](std::size_t index) {
                const RowTask& task = tasks[index];
                auto& job           = cpu_experts[task.part / 2];
                const bool is_up    = (task.part & 1U) != 0;
                const auto& p       = is_up ? up_p : gate_p;
                const ops::GgufCpuMatrix w{p.format, job.bytes[is_up ? 1 : 0], p.row_bytes, p.rows,
                                           p.k};
                const auto& x = is_up && own_up ? (job.every_token ? shared_up : job.input_up)
                                                : (job.every_token ? shared : job.input);
                ops::gguf_cpu_rows(w, x, task.begin, task.end, (is_up ? job.up : job.gate).data(),
                                   kWidth, backend);
            },
            side);
        const double rows_us =
            std::chrono::duration<double, std::micro>(pool->finished() - quantized).count();
        const auto middle = Clock::now();
        // The middle in BF16, as the GPU's up kernel stores it before quantizing it.
        pieces.clear();
        for (std::uint32_t i = 0; i < cpu_experts.size(); ++i) {
            for (int j = 0; j < int(cpu_experts[i].tokens.size()); ++j) {
                pieces.push_back({i, j, 0, false});
            }
        }
        pool->run(pieces.size(), [&](std::size_t index) {
            const Piece& piece   = pieces[index];
            auto& job            = cpu_experts[piece.job];
            const std::size_t at = std::size_t(piece.token) * kWidth;
            ops::gguf_cpu_swiglu_bf16(job.gate.data() + at, job.up.data() + at, kWidth,
                                      job.middle_bf16.data() + at, backend);
            job.middle.quantize_bf16(piece.token, job.middle_bf16.data() + at, 0, kWidth / 32);
        });
        parts.clear();
        for (const auto& job : cpu_experts) {
            parts.push_back({kHidden, ExpertSplit::cpu_work(1.0, int(job.tokens.size()))});
        }
        tasks = row_tasks(parts, pool->threads(), kRowQuantum);
        pool->run(tasks.size(), [&](std::size_t index) {
            const RowTask& task = tasks[index];
            auto& job           = cpu_experts[task.part];
            const ops::GgufCpuMatrix w{down_p.format, job.bytes[2], down_p.row_bytes, down_p.rows,
                                       down_p.k};
            ops::gguf_cpu_rows(w, job.middle, task.begin, task.end,
                               products + std::size_t(job.first) * kHidden, kHidden, backend);
        });
        *static_cast<std::int32_t*>(rank.product_count->data()) = first;

        const auto done = Clock::now();
        return std::chrono::duration<double, std::micro>(quantized - started).count() + rows_us +
               std::chrono::duration<double, std::micro>(done - middle).count();
    }

    static double expert_megabytes(const MissLayer& layer) {
        double bytes = 0;
        for (const auto& p : layer.projections) { bytes += double(p.expert_bytes); }
        return bytes / 1e6;
    }

    // The CPU's count of a request's missing experts, the last of `order` (most pairs first).
    static std::size_t balance(Rank& rank, const MissLayer& layer,
                               const std::vector<std::pair<int, std::int32_t>>& order) {
        std::vector<int> pairs;
        pairs.reserve(order.size());
        for (const auto& [uses, expert] : order) { pairs.push_back(uses); }
        return rank.split.cpu_count(pairs, expert_megabytes(layer));
    }

    void serve_request(Rank& rank, std::uint32_t sequence) {
        const auto started = std::chrono::steady_clock::now();
        Mailbox& box       = *rank.box;
        const std::uint32_t layer_index = box.layer;
        const std::uint32_t tokens      = box.tokens;
        const std::uint32_t missing     = box.missing;
        const MissLayer& layer          = layers.at(layer_index);
        RankBinding bind(device, layer.rank);
        const auto* ids        = static_cast<const std::int32_t*>(rank.host_ids->data());
        const auto* list       = static_cast<const std::int32_t*>(rank.host_missing->data());
        auto* tables           = static_cast<const void**>(rank.tables->data());
        auto* missing_ids      = static_cast<std::int32_t*>(rank.missing_ids->data());
        const std::uint32_t pairs = tokens * kTop;
        // Requests of a rank come one at a time: a layer's request is posted only after the stage
        // two of the one before has run on the same stream, so both staging halves are free.
        const std::uint32_t half = sequence & 1U;
        auto* staging = static_cast<std::byte*>(rank.staging.p) + half * half_bytes;
        auto* file_staging =
            disk ? static_cast<std::byte*>(rank.file_staging->data()) + half * file_half_bytes
                 : nullptr;
        // Experts with the most pairs go to the GPU; the CPU takes its share from the rest.
        std::vector<std::pair<int, std::int32_t>> order;
        for (std::uint32_t i = 0; i < missing; ++i) {
            const std::int32_t e = list[i];
            int uses             = 0;
            for (std::uint32_t p = 0; p < pairs; ++p) { uses += ids[p] == e; }
            order.emplace_back(uses, e);
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t cpu_count = 0;
        if (cpu_capable(layer)) {
            cpu_count = options.cpu_balance
                            ? balance(rank, layer, order)
                            : std::min(order.size(), std::size_t(std::lround(
                                                         double(order.size()) * options.cpu_share)));
        }
        const std::size_t gpu_count = order.size() - cpu_count;
        std::array<bool, kMaxExperts> on_gpu{};
        std::uint64_t used = 0, file_used = 0, staged_bytes = 0;
        std::uint64_t staged = 0, mapped = 0, admitted_count = 0;
        bool copies = false;
        auto* patches       = static_cast<TablePatch*>(rank.patches->data());
        int patch_count     = 0;
        std::array<unsigned char, kMaxExperts> in_use{};
        for (std::uint32_t p = 0; p < pairs; ++p) {
            if (ids[p] >= 0 && ids[p] < kMaxExperts) { in_use[std::size_t(ids[p])] = 1; }
        }
        // The GPU's share: admissions, staged copies and the second stage's missing pairs.
        const auto issue = [&] {
            if (rank.copying && gpu_count > 0) {
                CUDA_CHECK(cudaEventRecord(rank.copying, rank.copies));
            }
            for (std::size_t i = 0; i < gpu_count; ++i) {
                const std::int32_t e   = order[i].second;
                on_gpu[std::size_t(e)] = true;
                ExpertCache::Admitted admitted;
                if (layer.admit && cache->admit(layer_index, e, order[i].first,
                                                {in_use.data(), in_use.size()}, admitted)) {
                    for (int k = 0; k < 3; ++k) {
                        const auto& p = layer.projections[std::size_t(k)];
                        CUDA_CHECK(cudaMemcpyAsync(
                            admitted.slot[std::size_t(k)], p.host[std::size_t(e)],
                            std::size_t(p.expert_bytes), cudaMemcpyHostToDevice, rank.copies));
                        tables[std::size_t(k) * kMaxExperts + std::size_t(e)] =
                            admitted.slot[std::size_t(k)];
                        staged_bytes += std::uint64_t(p.expert_bytes);
                    }
                    patches[patch_count++] =
                        TablePatch{e, {admitted.slot[0], admitted.slot[1], admitted.slot[2]}};
                    if (admitted.victim >= 0) {
                        patches[patch_count++] =
                            TablePatch{admitted.victim,
                                       {admitted.victim_host[0], admitted.victim_host[1],
                                        admitted.victim_host[2]}};
                    }
                    copies = true;
                    ++staged;
                    ++admitted_count;
                    continue;
                }
                std::uint64_t need = kTail;
                for (const auto& p : layer.projections) {
                    need += aligned(std::uint64_t(p.expert_bytes));
                }
                if (used + need <= half_bytes) {
                    std::uint64_t offset = 0;
                    for (int k = 0; k < 3; ++k) {
                        const auto& p     = layer.projections[std::size_t(k)];
                        std::byte* target = staging + used + offset;
                        const void* source;
                        if (disk) {
                            if (file_used + std::uint64_t(p.expert_bytes) > file_half_bytes) {
                                throw std::runtime_error("expert misses: file staging exhausted");
                            }
                            fetch(p, e, file_staging + file_used);
                            source = file_staging + file_used;
                            file_used += aligned(std::uint64_t(p.expert_bytes));
                        } else {
                            source = p.host[std::size_t(e)];
                        }
                        CUDA_CHECK(cudaMemcpyAsync(target, source, std::size_t(p.expert_bytes),
                                                   cudaMemcpyHostToDevice, rank.copies));
                        if (k == 2) {
                            CUDA_CHECK(
                                cudaMemsetAsync(target + p.expert_bytes, 0, kTail, rank.copies));
                        }
                        tables[std::size_t(k) * kMaxExperts + std::size_t(e)] = target;
                        offset += aligned(std::uint64_t(p.expert_bytes));
                        staged_bytes += std::uint64_t(p.expert_bytes);
                    }
                    used += need;
                    copies = true;
                    ++staged;
                } else {
                    // The staging half is full: the GPU reads this expert where it is.
                    for (int k = 0; k < 3; ++k) {
                        const auto& p = layer.projections[std::size_t(k)];
                        const void* where;
                        if (disk) {
                            if (file_used + std::uint64_t(p.expert_bytes) > file_half_bytes) {
                                throw std::runtime_error("expert misses: file staging exhausted");
                            }
                            fetch(p, e, file_staging + file_used);
                            where = file_staging + file_used;
                            file_used += aligned(std::uint64_t(p.expert_bytes));
                        } else {
                            where = p.host[std::size_t(e)];
                        }
                        tables[std::size_t(k) * kMaxExperts + std::size_t(e)] = where;
                    }
                    ++mapped;
                }
            }
            if (patch_count > 0) {
                const auto tables_of = cache->tables(layer_index);
                patch_kernel<<<1, 128, 0, rank.copies>>>(static_cast<const void**>(tables_of[0]),
                                                         static_cast<const void**>(tables_of[1]),
                                                         static_cast<const void**>(tables_of[2]),
                                                         patches, patch_count);
                CUDA_CHECK(cudaGetLastError());
            }
            for (std::uint32_t p = 0; p < pairs; ++p) {
                const std::int32_t e = ids[p];
                missing_ids[p] = (e >= 0 && e < kMaxExperts && on_gpu[std::size_t(e)]) ? e : -1;
            }
            // The second stage reads the missing pairs from device memory, which the split kernel
            // filled with -1.
            if (gpu_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(rank.missing_device.p, missing_ids,
                                           std::size_t(pairs) * 4, cudaMemcpyHostToDevice,
                                           rank.copies));
                copies = true;
            }
            if (copies) { CUDA_CHECK(cudaEventRecord(rank.copied, rank.copies)); }
        };
        // The CPU's share, while this thread issues the copies (disk experts read the GPU's share
        // first: both share the file staging).
        std::uint64_t cpu_pairs = 0;
        double cpu_work_done    = 0;
        if (cpu_count > 0) {
            if (disk) { issue(); }
            cpu_experts.resize(cpu_count);
            for (std::size_t i = 0; i < cpu_count; ++i) {
                CpuExpert& job = cpu_experts[i];
                job.expert     = order[gpu_count + i].second;
                job.pairs.clear();
                job.tokens.clear();
                for (std::uint32_t p = 0; p < pairs; ++p) {
                    if (ids[p] == job.expert) {
                        job.pairs.push_back(std::int32_t(p));
                        job.tokens.push_back(std::int32_t(p / kTop));
                    }
                }
                cpu_pairs += job.pairs.size();
                cpu_work_done += ExpertSplit::cpu_work(expert_megabytes(layer), int(job.pairs.size()));
                for (int k = 0; k < 3; ++k) {
                    const auto& proj = layer.projections[std::size_t(k)];
                    if (disk) {
                        if (file_used + std::uint64_t(proj.expert_bytes) > file_half_bytes) {
                            throw std::runtime_error("expert misses: file staging exhausted");
                        }
                        fetch(proj, job.expert, file_staging + file_used);
                        job.bytes[std::size_t(k)] =
                            reinterpret_cast<const std::uint8_t*>(file_staging + file_used);
                        file_used += aligned(std::uint64_t(proj.expert_bytes));
                    } else {
                        job.bytes[std::size_t(k)] =
                            static_cast<const std::uint8_t*>(proj.host[std::size_t(job.expert)]);
                    }
                }
            }
            const double microseconds =
                disk ? compute_cpu(layer, rank, [] {}) : compute_cpu(layer, rank, issue);
            if (options.cpu_balance) { rank.split.computed(cpu_work_done, microseconds); }
        } else {
            issue();
            if (rank.product_count) { *static_cast<std::int32_t*>(rank.product_count->data()) = 0; }
        }
        if (copies) { CUDA_CHECK(cudaEventSynchronize(rank.copied)); }
        if (options.cpu_balance && gpu_count > 0 && staged_bytes > 0 && mapped == 0) {
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, rank.copying, rank.copied));
            rank.split.copied(double(staged_bytes) / 1e6, 1e3 * double(milliseconds));
        }
        Atomic(box.prepared).store(sequence, cuda::memory_order_release);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::lock_guard lock(stats_mutex);
        ++counters.requests;
        counters.staged += staged;
        counters.mapped += mapped;
        counters.admitted += admitted_count;
        counters.computed += cpu_count;
        counters.cpu_pairs += cpu_pairs;
        counters.staged_bytes += staged_bytes;
        counters.service_seconds += seconds;
    }

    static std::int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void serve() {
        bool awake = false;
        while (!stopping.load(std::memory_order_acquire)) {
            if (now_ns() > deadline.load(std::memory_order_acquire)) {
                if (awake && pool) { pool->spin(false); }
                awake = false;
                std::unique_lock lock(wake_mutex);
                sleeping.store(true, std::memory_order_release);
                wake.wait(lock, [this] {
                    return now_ns() <= deadline.load(std::memory_order_acquire) ||
                           stopping.load(std::memory_order_acquire);
                });
                sleeping.store(false, std::memory_order_release);
                continue;
            }
            if (!awake && pool) { pool->spin(true); }
            awake = true;
            bool worked = false;
            for (auto& rank : ranks) {
                if (!rank.used) { continue; }
                const std::uint32_t sequence =
                    Atomic(rank.box->requested).load(cuda::memory_order_acquire);
                if (sequence == rank.served) { continue; }
                try {
                    serve_request(rank, sequence);
                } catch (...) {
                    remember_failure();
                    // Release the GPU with an empty stage two; the failure surfaces at the
                    // executor's next check.
                    std::memset(rank.missing_ids->data(), 0xff, rank.missing_ids->size());
                    if (rank.product_count) {
                        *static_cast<std::int32_t*>(rank.product_count->data()) = 0;
                    }
                    Atomic(rank.box->prepared).store(sequence, cuda::memory_order_release);
                }
                rank.served = sequence;
                worked      = true;
                // Activity keeps the service awake: a pass is mid-flight.
                deadline.store(std::max(deadline.load(std::memory_order_relaxed),
                                        now_ns() + 500'000'000),
                               std::memory_order_release);
            }
            if (!worked) { spin_hint(); }
        }
    }

    void drop_missing(std::size_t index, const Tensor& ids, const Tensor& weights,
                      Tensor& kept_ids, Tensor& kept_weights, cudaStream_t stream) {
        const MissLayer& layer = layers.at(index);
        const int tokens       = ids.ne[1];
        drop_missing_kernel<<<(tokens + 127) / 128, 128, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), static_cast<const float*>(weights.data),
            tokens, int(layer.experts), layer.gate_table,
            static_cast<const std::uintptr_t*>(ranges.at(index).p),
            static_cast<std::int32_t*>(kept_ids.data), static_cast<float*>(kept_weights.data));
        CUDA_CHECK(cudaGetLastError());
    }

    Call begin(std::size_t index, const Tensor& ids, const Tensor& m, std::int32_t tokens,
               const ops::GgufMoeWeights& banks, cudaStream_t stream) {
        keep_alive();
        const MissLayer& layer = layers.at(index);
        Rank& rank             = ranks.at(layer.rank);
        if (tokens <= 0 || std::uint32_t(tokens) > options.token_capacity) {
            throw std::invalid_argument("expert misses: tokens exceed the call capacity");
        }
        const int pairs = tokens * kTop;
        split_kernel<<<1, kSplitThreads, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), pairs, int(layer.experts), layer.gate_table,
            static_cast<const std::uintptr_t*>(ranges.at(index).p),
            static_cast<std::int32_t*>(rank.cached_ids.p),
            static_cast<std::int32_t*>(rank.host_ids->data()),
            static_cast<std::int32_t*>(rank.host_missing->data()),
            static_cast<std::int32_t*>(rank.missing_device.p),
            static_cast<std::int32_t*>(rank.product_count->data()),
            static_cast<const __nv_bfloat16*>(m.data),
            rank.host_m ? static_cast<__nv_bfloat16*>(rank.host_m->data()) : nullptr,
            tokens * kHidden, rank.box, std::uint32_t(index), tokens, int(options.tail_skip_rank),
            static_cast<std::uint32_t*>(rank.need->data()));
        CUDA_CHECK(cudaGetLastError());
        Call call;
        call.cached_ids  = Tensor(rank.cached_ids.p, DType::I32, {kTop, tokens});
        call.missing_ids = Tensor(rank.missing_device.p, DType::I32, {kTop, tokens});
        call.missing_banks = banks;
        auto* tables       = static_cast<const void* const*>(rank.tables->data());
        call.missing_banks.gate.experts = tables;
        call.missing_banks.up.experts   = tables + kMaxExperts;
        call.missing_banks.down.experts = tables + 2 * kMaxExperts;
        // Staged experts sit in device memory with zeros after each down, as the matrix kernel
        // wants; experts the GPU reads across the bus keep the call on the vector products.
        call.missing_banks.device_resident = false;
        call.products      = rank.host_m ? static_cast<const float*>(rank.products->data()) : nullptr;
        call.pairs         = static_cast<const std::int32_t*>(rank.product_pairs->data());
        call.count         = static_cast<const std::int32_t*>(rank.product_count->data());
        call.max_products  = rank.host_m ? pairs : 0;
        return call;
    }

    void await(std::size_t index, cudaStream_t stream) {
        Rank& rank = ranks.at(layers.at(index).rank);
        if (!capturing(stream)) {
            // Eager: wait on the host, so a lazily loaded kernel never waits behind a GPU spin.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::uint32_t sequence = *static_cast<volatile std::uint32_t*>(rank.need->data());
            if (sequence != 0) {
                while (Atomic(rank.box->prepared).load(cuda::memory_order_acquire) != sequence) {
                    std::this_thread::yield();
                    std::lock_guard lock(failure_mutex);
                    if (failure) { std::rethrow_exception(failure); }
                }
            }
            return;
        }
        wait_kernel<<<1, 1, 0, stream>>>(rank.box, static_cast<const std::uint32_t*>(rank.need->data()));
        CUDA_CHECK(cudaGetLastError());
    }

    void raise_failure() {
        std::lock_guard lock(failure_mutex);
        if (failure) { std::rethrow_exception(std::exchange(failure, nullptr)); }
    }

    void keep_alive() {
        raise_failure();
        deadline.store(now_ns() + 500'000'000, std::memory_order_release);
        if (sleeping.load(std::memory_order_acquire)) {
            std::lock_guard lock(wake_mutex);
            wake.notify_all();
        }
    }
};

ExpertMisses::ExpertMisses(DeviceContext& device, std::vector<MissLayer> layers,
                           std::vector<std::filesystem::path> files, MissOptions options,
                           ExpertCache* cache)
    : impl_(std::make_unique<Impl>(device, std::move(layers), std::move(files), options, cache)) {}
ExpertMisses::~ExpertMisses() = default;

ExpertMisses::Call ExpertMisses::begin(std::size_t layer, const Tensor& ids, const Tensor& m,
                                       std::int32_t tokens, const ops::GgufMoeWeights& banks,
                                       cudaStream_t stream) {
    return impl_->begin(layer, ids, m, tokens, banks, stream);
}
void ExpertMisses::drop_missing(std::size_t layer, const Tensor& ids, const Tensor& weights,
                                Tensor& kept_ids, Tensor& kept_weights, cudaStream_t stream) {
    impl_->drop_missing(layer, ids, weights, kept_ids, kept_weights, stream);
}
void ExpertMisses::await(std::size_t layer, cudaStream_t stream) { impl_->await(layer, stream); }
void ExpertMisses::keep_alive() { impl_->keep_alive(); }

void ExpertMisses::raise_failure() { impl_->raise_failure(); }
void ExpertMisses::set_resident(std::size_t layer, const ExpertCache::Ranges& resident) {
    auto& target = impl_->layers.at(layer);
    target.resident = resident;
    RankBinding bind(impl_->device, target.rank);
    Impl::write_ranges(impl_->ranges.at(layer), resident);
}
MissStats ExpertMisses::stats() const {
    std::lock_guard lock(impl_->stats_mutex);
    return impl_->counters;
}
bool ExpertMisses::cpu_enabled() const noexcept { return impl_->pool != nullptr; }
std::string ExpertMisses::cpu_description() const {
    if (!impl_->pool) { return {}; }
    return std::to_string(impl_->cpu_threads) +
           (impl_->pool->pinned() ? " pinned threads, " : " threads, ") +
           ops::gguf_cpu_backend_name(impl_->backend);
}

} // namespace ninfer::models::qwen4_exp
