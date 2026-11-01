// Toolkit-independent launcher. One fixed backend remains loaded until process exit.
#include "selection.h"
#include "ninfer_build_id.h"
#ifdef NINFER_ROUTER_LAUNCHER
#include "serve/model_catalog.h"
#include <optional>
#endif
#include <charconv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace {
namespace fs = std::filesystem;
#ifdef _WIN32
using Module = HMODULE;
#else
using Module = void*;
#endif

fs::path executable() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!length || length == buffer.size()) throw std::runtime_error("Cannot locate the executable");
    return fs::path(std::wstring(buffer.data(), length));
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size)) throw std::runtime_error("Cannot locate the executable");
    return fs::canonical(buffer.data());
#else
    return fs::canonical("/proc/self/exe");
#endif
}

Module load(const fs::path& path, bool driver = false) {
#ifdef _WIN32
    auto module = LoadLibraryExW(path.c_str(), nullptr, driver ? LOAD_LIBRARY_SEARCH_SYSTEM32 :
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) throw std::runtime_error("Cannot load " + path.string() +
                                         " (Windows error " + std::to_string(GetLastError()) + ")");
#else
    (void)driver;
    auto module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) throw std::runtime_error("Cannot load " + path.string() + ": " + dlerror());
#endif
    return module;
}

template<class Function> Function symbol(Module module, const char* name) {
#ifdef _WIN32
    const auto address = GetProcAddress(module, name);
#else
    const auto address = dlsym(module, name);
#endif
    if (!address) throw std::runtime_error(std::string("Missing entry point: ") + name);
    return reinterpret_cast<Function>(address);
}

int number(std::string_view text) {
    int value = -1;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < 0)
        throw std::runtime_error("Invalid CUDA device index: " + std::string(text));
    return value;
}

struct Options {
    int requested = 0;
    std::vector<int> devices{0};
    std::vector<char*> arguments;
#ifdef NINFER_ROUTER_LAUNCHER
    std::optional<fs::path> models_directory;
    std::optional<fs::path> models_preset;
#endif
};
Options parse(int argc, char** argv) {
    Options options;
    bool backend_seen = false;
    int device_kind = 0;
    options.arguments.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--cuda-backend" || arg.starts_with("--cuda-backend=")) {
            if (backend_seen) throw std::runtime_error("Specify --cuda-backend once");
            backend_seen = true;
            std::string_view value;
            if (arg == "--cuda-backend") {
                if (++i == argc) throw std::runtime_error("--cuda-backend requires auto, 12 or 13");
                value = argv[i];
            } else value = arg.substr(std::strlen("--cuda-backend="));
            if (value == "12") options.requested = 12;
            else if (value == "13") options.requested = 13;
            else if (value != "auto") throw std::runtime_error("--cuda-backend requires auto, 12 or 13");
            continue;
        }
        options.arguments.push_back(argv[i]);
        if (arg == "--device" || arg == "--devices") {
            const int kind = arg == "--device" ? 1 : 2;
            if (device_kind && device_kind != kind)
                throw std::runtime_error("--device and --devices are mutually exclusive");
            device_kind = kind;
            if (++i == argc) throw std::runtime_error(std::string(arg) + " requires a value");
            options.arguments.push_back(argv[i]);
            options.devices.clear();
            std::string_view list(argv[i]);
            for (;;) {
                const auto comma = list.find(',');
                options.devices.push_back(number(list.substr(0, comma)));
                if (comma == std::string_view::npos) break;
                if (arg == "--device") throw std::runtime_error("--device requires one index");
                list.remove_prefix(comma + 1);
            }
            if (options.devices.size() > 8) throw std::runtime_error("--devices supports at most eight GPUs");
        }
        // Quoted prompt/template strings must not become launcher options. These
        // product flags consume text even when that text starts with '--'.
        else if (arg == "--prompt" || arg == "--system" || arg == "--chat-template" ||
                 arg == "--prompt-file" || arg == "--system-file" || arg == "--messages" ||
                 arg == "--stop" || arg == "--reasoning-stop" || arg == "--model-id" ||
                 arg == "--api-key" || arg == "--host" || arg == "--device-profile-path" ||
                 arg == "--context-cost-presets" || arg == "--prefix-cache-file" ||
                 arg == "--disk-kv-path" || arg == "--request-log-jsonl" ||
                 arg == "--thinking-budget-message" || arg == "--slot-save-path" ||
                 arg == "--models-dir" || arg == "--models-preset" || arg == "--graft" ||
                 arg == "--default-graft" || arg == "--ngram-table" || arg == "--ngram-profile" ||
                 arg == "--corpus" || arg == "--text" || arg == "--output" ||
                 arg == "--weights" || arg == "-m" || arg == "-f" || arg == "--output-file" ||
                 arg == "--out" || arg == "--evaluate") {
            if (i + 1 < argc) {
#ifdef NINFER_ROUTER_LAUNCHER
                if (arg == "--models-dir") options.models_directory = argv[i + 1];
                if (arg == "--models-preset") options.models_preset = argv[i + 1];
#endif
                options.arguments.push_back(argv[++i]);
            }
        }
    }
    return options;
}

#ifdef NINFER_ROUTER_LAUNCHER
std::vector<int> catalog_devices(const Options& options) {
    if (!options.models_directory && !options.models_preset) return options.devices;
    std::optional<ninfer::serve::PresetFile> preset;
    if (options.models_preset) {
        std::ifstream file(*options.models_preset);
        if (!file) throw std::runtime_error("Cannot read --models-preset");
        std::stringstream text;
        text << file.rdbuf();
        preset = ninfer::serve::parse_preset_file(text.str());
    }
    std::vector<std::string> shared;
    for (std::size_t i = 1; i < options.arguments.size(); ++i) shared.emplace_back(options.arguments[i]);
    std::vector<int> devices;
    for (const auto& model : ninfer::serve::build_catalog(options.models_directory, preset, shared)) {
        std::vector<std::string> words{NINFER_COMMAND, model.artifact.string()};
        words.insert(words.end(), model.arguments.begin(), model.arguments.end());
        std::vector<char*> args;
        for (auto& word : words) args.push_back(word.data());
        const auto selected = parse(static_cast<int>(args.size()), args.data());
        devices.insert(devices.end(), selected.devices.begin(), selected.devices.end());
    }
    std::sort(devices.begin(), devices.end());
    devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
    return devices;
}
#endif

struct Hardware { int driver; std::vector<int> architectures; };
Hardware inspect(const std::vector<int>& ordinals) {
#ifdef _WIN32
    auto module = load("nvcuda.dll", true);
#define NINFER_DRIVER_CALL __stdcall
#else
    auto module = load("libcuda.so.1", true);
#define NINFER_DRIVER_CALL
#endif
    using Init = int(NINFER_DRIVER_CALL*)(unsigned);
    using GetInt = int(NINFER_DRIVER_CALL*)(int*);
    using GetDevice = int(NINFER_DRIVER_CALL*)(int*, int);
    using Attribute = int(NINFER_DRIVER_CALL*)(int*, int, int);
    auto check = [](int result) {
        if (result) throw std::runtime_error("NVIDIA Driver API failed (code " + std::to_string(result) + ")");
    };
    check(symbol<Init>(module, "cuInit")(0));
    Hardware hardware{};
    check(symbol<GetInt>(module, "cuDriverGetVersion")(&hardware.driver));
    int count = 0;
    check(symbol<GetInt>(module, "cuDeviceGetCount")(&count));
    const auto get = symbol<GetDevice>(module, "cuDeviceGet");
    const auto attribute = symbol<Attribute>(module, "cuDeviceGetAttribute");
    for (int ordinal : ordinals) {
        if (ordinal >= count) throw std::runtime_error("Selected CUDA device does not exist: " + std::to_string(ordinal));
        int device = 0, major = 0, minor = 0;
        check(get(&device, ordinal));
        // Stable Driver API enum values: COMPUTE_CAPABILITY_MAJOR and MINOR.
        check(attribute(&major, 75, device));
        check(attribute(&minor, 76, device));
        hardware.architectures.push_back(major * 10 + minor);
    }
    return hardware;
}

fs::path runtime(const fs::path& root, int major) {
    const auto directory = root / "backends" / ("cuda" + std::to_string(major));
#ifdef _WIN32
    return directory / "ninfer-runtime.dll";
#elif defined(__APPLE__)
    return directory / "libninfer-runtime.dylib";
#else
    return directory / "libninfer-runtime.so";
#endif
}
ninfer::launcher::Backend configuration(const fs::path& root, int major, bool present) {
    auto backend = major == 12 ? ninfer::launcher::cuda12 : ninfer::launcher::cuda13;
    if (!present) return backend;
    std::ifstream file(runtime(root, major).parent_path() / "architectures.txt");
    if (!file) throw std::runtime_error("Missing backend architectures.txt");
    backend.architectures.clear();
    std::string arch;
    while (file >> arch) {
        if (arch != "70" && arch != "75" && arch != "80" && arch != "86" &&
            arch != "89" && arch != "120a") throw std::runtime_error("Invalid backend architecture: " + arch);
        backend.architectures.push_back(arch == "120a" ? 120 : number(arch));
    }
    if (backend.architectures.empty()) throw std::runtime_error("Empty backend architecture set");
    return backend;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        auto options = parse(argc, argv);
        const auto root = executable().parent_path();
        const bool help = options.arguments.size() >= 2 &&
            (std::string_view(options.arguments[1]) == "--help" ||
             std::string_view(options.arguments[1]) == "-h");
        if (options.arguments.size() >= 2 && std::string_view(options.arguments[1]) == "--version") {
            std::cout << NINFER_COMMAND << ' ' << NINFER_VERSION << '+' << NINFER_BUILD_ID << '\n';
            return 0;
        }
        if (help) {
            std::ifstream text(root / "help" / (std::string(NINFER_COMMAND) + ".txt"));
            if (text) {
                std::cout << text.rdbuf();
                std::cout << "\n  --cuda-backend auto|12|13     select a packaged CUDA backend (default auto)\n";
                return 0;
            }
        }
#ifdef NINFER_ROUTER_LAUNCHER
        if (!help) options.devices = catalog_devices(options);
#endif
        const bool has12 = fs::is_regular_file(runtime(root, 12));
        const bool has13 = fs::is_regular_file(runtime(root, 13));
        int selected = options.requested;
        if (help || options.devices.empty() || std::string_view(NINFER_COMMAND) == "ninfer-ngram-profile") {
            // The n-gram profiler is a CPU command. It needs no device selection.
            // The packager captures each command's real help before shipping.
            // Linux uses a temporary driver stub for this build-tree operation.
            if (!selected) selected = has13 ? 13 : 12;
        } else {
            const auto hardware = inspect(options.devices);
            selected = ninfer::launcher::select(options.requested, hardware.driver,
                                                hardware.architectures, has12, has13,
                                                configuration(root, 12, has12), configuration(root, 13, has13));
        }
        auto module = load(runtime(root, selected));
        using BuildId = const char*(*)();
        if (std::strcmp(symbol<BuildId>(module, "ninfer_backend_build_id_v1")(), NINFER_BUILD_ID))
            throw std::runtime_error("Launcher and CUDA backend come from different builds");
        using Entry = int(*)(const char*, int, char**);
        const auto entry = symbol<Entry>(module, "ninfer_backend_main_v1");
        if (!help) std::fprintf(stderr, "%s: CUDA %d backend\n", NINFER_COMMAND, selected);
        std::string help_command = NINFER_COMMAND;
        if (help) options.arguments[0] = help_command.data();
        const int count = static_cast<int>(options.arguments.size());
        options.arguments.push_back(nullptr);
        return entry(NINFER_COMMAND, count, options.arguments.data());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", NINFER_COMMAND, error.what());
        return 1;
    }
}
