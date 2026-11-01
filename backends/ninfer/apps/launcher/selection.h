#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::launcher {
struct Backend {
    int major;
    int minimum_driver_api;
    std::vector<int> architectures;
};
inline const Backend cuda12{12, 12000, {70, 75, 80, 86, 89}};
inline const Backend cuda13{13, 13000, {75, 80, 86, 89, 120}};

inline bool supports(const Backend& backend, int driver, const std::vector<int>& devices) {
    return driver >= backend.minimum_driver_api && !devices.empty() &&
           std::all_of(devices.begin(), devices.end(), [&](int sm) {
               return std::find(backend.architectures.begin(), backend.architectures.end(), sm) !=
                      backend.architectures.end();
           });
}
inline int select(int requested, int driver, const std::vector<int>& devices,
                  bool has12, bool has13,
                  const Backend& built12 = cuda12, const Backend& built13 = cuda13) {
    for (const auto& backend : {built13, built12}) {
        if (requested != 0 && requested != backend.major) continue;
        if ((backend.major == 12 ? has12 : has13) && supports(backend, driver, devices))
            return backend.major;
    }
    throw std::runtime_error("No packaged CUDA backend supports the driver and all selected GPUs. "
                             "CUDA 12 supports sm_70/75/80/86/89; CUDA 13 supports sm_75/80/86/89/120a. "
                             "Check --device/--devices, --cuda-backend and the NVIDIA driver.");
}
}  // namespace ninfer::launcher
