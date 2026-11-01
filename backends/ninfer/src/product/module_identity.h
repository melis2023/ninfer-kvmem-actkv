#pragma once
#include <filesystem>
#include <string>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ninfer::product {
// Identify the module that owns the code, whether it is an EXE or a private runtime.
// Rebuilding a dirty source tree can retain its build ID, so include module size and mtime.
inline std::string module_identity(const void* address, std::string build_id) {
    std::filesystem::path path;
#ifdef _WIN32
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(address), &owner))
        throw std::runtime_error("Cannot locate the executing NInfer module");
    std::wstring buffer(32768, L'\0');
    const auto length = GetModuleFileNameW(owner, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!length || length == buffer.size()) throw std::runtime_error("Cannot locate the executing NInfer module");
    buffer.resize(length);
    path = buffer;
#else
    Dl_info module{};
    if (!dladdr(address, &module) || !module.dli_fname)
        throw std::runtime_error("Cannot locate the executing NInfer module");
    path = module.dli_fname;
#ifndef __APPLE__
    // A PATH-launched executable can have a relative link-map name. Runtime DLLs
    // are loaded by absolute path, so this branch identifies only the executable.
    if (!path.is_absolute()) path = std::filesystem::canonical("/proc/self/exe");
#endif
#endif
    // Errors must not turn this into a build-ID-only cache identity.
    const auto size = std::filesystem::file_size(path);
    const auto time = std::filesystem::last_write_time(path).time_since_epoch().count();
    return build_id + ";module=" + (path.parent_path().filename() / path.filename()).string() + ";size=" + std::to_string(size) +
           ";mtime=" + std::to_string(static_cast<long long>(time));
}
}  // namespace ninfer::product
