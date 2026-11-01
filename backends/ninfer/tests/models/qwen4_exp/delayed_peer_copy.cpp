#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

// Exercise reuse of the first stage's embedding buffer during MTP catch-up. This fixture's
// first prompt chunk has eight tokens, hence seven MTP cells. Delay alternate transfers so
// an omitted producer wait cannot hide behind a consistently fast peer copy.
extern "C" cudaError_t cudaMemcpyPeerAsync(void* destination, int destination_device,
                                           const void* source, int source_device,
                                           std::size_t bytes, cudaStream_t stream) {
    using Copy = cudaError_t (*)(void*, int, const void*, int, std::size_t, cudaStream_t);
    static auto real_copy = reinterpret_cast<Copy>(dlsym(RTLD_NEXT, "cudaMemcpyPeerAsync"));
    static std::atomic<unsigned> prompt_copies{0};
    if (!real_copy) { return cudaErrorUnknown; }
    if (source_device == 0 && destination_device == 1 && bytes == 7 * 2560 * 2 &&
        prompt_copies.fetch_add(1) % 2 == 0) {
        const auto delayed = cudaLaunchHostFunc(stream, [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }, nullptr);
        if (delayed != cudaSuccess) { return delayed; }
        std::fprintf(stderr, "injected delayed MTP embedding copy\n");
    }
    return real_copy(destination, destination_device, source, source_device, bytes, stream);
}
