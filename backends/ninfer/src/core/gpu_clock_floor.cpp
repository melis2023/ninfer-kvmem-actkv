#include "core/gpu_clock_floor.h"

#include <cuda_runtime_api.h>

#include <algorithm>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace ninfer {
namespace {

// The NVML entry points this needs, declared from its stable C ABI: the library is the driver's,
// loaded at run time, so a build needs no NVML headers and a host without NVML still starts.
using NvmlReturn   = int;
using NvmlDevice   = void*;
using InitFn       = NvmlReturn (*)();
using ShutdownFn   = NvmlReturn (*)();
using ErrorFn      = const char* (*)(NvmlReturn);
using ByPciFn      = NvmlReturn (*)(const char*, NvmlDevice*);
using MaxClockFn   = NvmlReturn (*)(NvmlDevice, int, unsigned int*);
using LockFn       = NvmlReturn (*)(NvmlDevice, unsigned int, unsigned int);
using UnlockFn     = NvmlReturn (*)(NvmlDevice);
constexpr NvmlReturn kNvmlSuccess      = 0;
constexpr NvmlReturn kNvmlNoPermission = 4;
constexpr int kNvmlClockGraphics       = 0;

} // namespace

struct GpuClockFloor::Nvml {
    ~Nvml() {
        if (initialized) { shutdown(); }
#if defined(_WIN32)
        if (library != nullptr) { FreeLibrary(static_cast<HMODULE>(library)); }
#else
        if (library != nullptr) { dlclose(library); }
#endif
    }

    [[nodiscard]] std::string error(NvmlReturn result) const {
        const char* text = error_string != nullptr ? error_string(result) : nullptr;
        return text != nullptr ? text : "NVML error " + std::to_string(result);
    }

    void* library        = nullptr;
    bool initialized     = false;
    InitFn init          = nullptr;
    ShutdownFn shutdown  = nullptr;
    ErrorFn error_string = nullptr;
    ByPciFn by_pci       = nullptr;
    MaxClockFn max_clock = nullptr;
    LockFn lock          = nullptr;
    UnlockFn unlock      = nullptr;
};

std::shared_ptr<GpuClockFloor::Nvml> GpuClockFloor::load_nvml(std::string& problem) {
    auto nvml = std::make_shared<Nvml>();
#if defined(_WIN32)
    // The driver installs nvml.dll in System32; drivers before 2019 kept it in NVSMI.
    HMODULE module = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module == nullptr) {
        wchar_t root[MAX_PATH];
        const DWORD length = GetEnvironmentVariableW(L"ProgramW6432", root, MAX_PATH);
        if (length > 0 && length < MAX_PATH) {
            const std::wstring path =
                std::wstring(root) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
            module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    nvml->library      = module;
    const auto symbol  = [&](const char* name) {
        return module != nullptr ? reinterpret_cast<void*>(GetProcAddress(module, name)) : nullptr;
    };
#else
    nvml->library     = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    const auto symbol = [&](const char* name) {
        return nvml->library != nullptr ? dlsym(nvml->library, name) : nullptr;
    };
#endif
    if (nvml->library == nullptr) {
        problem = "the NVIDIA management library (NVML) is not available";
        return nullptr;
    }
    nvml->init         = reinterpret_cast<InitFn>(symbol("nvmlInit_v2"));
    nvml->shutdown     = reinterpret_cast<ShutdownFn>(symbol("nvmlShutdown"));
    nvml->error_string = reinterpret_cast<ErrorFn>(symbol("nvmlErrorString"));
    nvml->by_pci       = reinterpret_cast<ByPciFn>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
    nvml->max_clock    = reinterpret_cast<MaxClockFn>(symbol("nvmlDeviceGetMaxClockInfo"));
    nvml->lock         = reinterpret_cast<LockFn>(symbol("nvmlDeviceSetGpuLockedClocks"));
    nvml->unlock       = reinterpret_cast<UnlockFn>(symbol("nvmlDeviceResetGpuLockedClocks"));
    if (nvml->init == nullptr || nvml->shutdown == nullptr || nvml->by_pci == nullptr ||
        nvml->max_clock == nullptr || nvml->lock == nullptr || nvml->unlock == nullptr) {
        problem = "the NVML library lacks the locked-clock entry points";
        return nullptr;
    }
    const NvmlReturn started = nvml->init();
    if (started != kNvmlSuccess) {
        problem = "NVML did not start: " + nvml->error(started);
        return nullptr;
    }
    nvml->initialized = true;
    return nvml;
}

GpuClockFloor::GpuClockFloor(std::span<const int> devices, std::uint32_t floor_mhz) {
    if (floor_mhz == 0) { return; }
    std::vector<int> unique;
    for (const int device : devices) {
        if (std::find(unique.begin(), unique.end(), device) == unique.end()) {
            unique.push_back(device);
        }
    }
    std::string problem;
    nvml_ = load_nvml(problem);
    if (nvml_ == nullptr) {
        report_.push_back(problem);
        return;
    }
    complete_ = true;
    for (const int device : unique) {
        char bus[32] = {};
        NvmlDevice handle = nullptr;
        unsigned int max_mhz = 0;
        std::string line = "device " + std::to_string(device) + ": ";
        if (cudaDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), device) != cudaSuccess) {
            (void)cudaGetLastError();
            report_.push_back(line + "no PCI bus id, clocks left alone");
            complete_ = false;
            continue;
        }
        NvmlReturn result = nvml_->by_pci(bus, &handle);
        if (result == kNvmlSuccess) {
            result = nvml_->max_clock(handle, kNvmlClockGraphics, &max_mhz);
        }
        const unsigned int low = std::min<unsigned int>(floor_mhz, max_mhz);
        // A trial lock shows now, rather than at the first request, whether the rights suffice.
        if (result == kNvmlSuccess) { result = nvml_->lock(handle, low, max_mhz); }
        if (result == kNvmlSuccess) { result = nvml_->unlock(handle); }
        if (result != kNvmlSuccess) {
            report_.push_back(line + "clocks left alone: " + nvml_->error(result) +
                              (result == kNvmlNoPermission
                                   ? " (holding a clock floor needs administrator or root rights)"
                                   : ""));
            complete_ = false;
            continue;
        }
        devices_.push_back(Device{.handle = handle, .low_mhz = low, .high_mhz = max_mhz});
        report_.push_back(line + "graphics clock held at " + std::to_string(low) + "-" +
                          std::to_string(max_mhz) + " MHz while requests run");
    }
    if (devices_.empty()) {
        nvml_.reset();
        return;
    }
    governor_ = std::thread([this] { govern(); });
}

GpuClockFloor::~GpuClockFloor() {
    if (!governor_.joinable()) { return; }
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_one();
    governor_.join();
}

std::shared_ptr<void> GpuClockFloor::hold() {
    if (devices_.empty()) { return nullptr; }
    {
        std::lock_guard lock(mutex_);
        ++holders_;
    }
    changed_.notify_one();
    return std::shared_ptr<void>(this, [](void* floor) {
        static_cast<GpuClockFloor*>(floor)->leave();
    });
}

void GpuClockFloor::leave() noexcept {
    {
        std::lock_guard lock(mutex_);
        --holders_;
    }
    changed_.notify_one();
}

// Locks the devices when the first holder arrives and resets them once none has been left for
// kRelease, or when this object ends.
void GpuClockFloor::govern() {
    std::unique_lock lock(mutex_);
    while (true) {
        changed_.wait(lock, [this] { return stopping_ || holders_ > 0; });
        if (stopping_) { return; }
        lock.unlock();
        for (const Device& device : devices_) {
            (void)nvml_->lock(device.handle, device.low_mhz, device.high_mhz);
        }
        lock.lock();
        while (!stopping_) {
            changed_.wait(lock, [this] { return stopping_ || holders_ == 0; });
            if (stopping_ ||
                !changed_.wait_for(lock, kRelease, [this] { return stopping_ || holders_ > 0; })) {
                break;
            }
        }
        lock.unlock();
        for (const Device& device : devices_) { (void)nvml_->unlock(device.handle); }
        lock.lock();
    }
}

} // namespace ninfer
