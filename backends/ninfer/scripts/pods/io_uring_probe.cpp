#include <cerrno>
#include <cstdio>

#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

int main() {
#ifdef __linux__
    io_uring_params parameters{};
    const int descriptor = static_cast<int>(syscall(__NR_io_uring_setup, 64, &parameters));
    if (descriptor < 0) {
        std::printf("IO_URING_SETUP_UNAVAILABLE errno=%d\n", errno);
        return 77;
    }
    std::printf("IO_URING_SETUP_OK sq=%u cq=%u features=%u\n",
                parameters.sq_entries, parameters.cq_entries, parameters.features);
    close(descriptor);
    return 0;
#else
    std::puts("IO_URING_SETUP_UNAVAILABLE platform");
    return 77;
#endif
}
