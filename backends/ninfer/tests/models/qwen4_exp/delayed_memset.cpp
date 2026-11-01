#include <cuda_runtime.h>
#include <dlfcn.h>
#include <chrono>
#include <thread>

// Characterize setup visibility: delay the legacy stream without changing the bytes written or
// cudaMemset's documented asynchronous semantics. Only the expert-cache regression preloads this.
extern "C" cudaError_t cudaMemset(void* destination, int value, std::size_t bytes) {
    using Memset = cudaError_t (*)(void*, int, std::size_t);
    static auto real_memset = reinterpret_cast<Memset>(dlsym(RTLD_NEXT, "cudaMemset"));
    if (!real_memset) { return cudaErrorUnknown; }
    const auto delayed = cudaLaunchHostFunc(nullptr, [](void*) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }, nullptr);
    if (delayed != cudaSuccess) { return delayed; }
    return real_memset(destination, value, bytes);
}
