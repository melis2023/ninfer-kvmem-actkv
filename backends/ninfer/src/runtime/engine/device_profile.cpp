#include "runtime/engine/device_profile.h"

#include "runtime/engine/device_profiles_builtin.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer::runtime {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::string_view kSchema = "ninfer.device-route-profiles";
// 2: per-key layering over the compiled table and the "calibration" provenance block.
constexpr int kSchemaVersion = 2;

int process_id() noexcept {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

[[noreturn]] void fail(std::string_view source, const std::string& message) {
    throw std::invalid_argument(std::string(source) + ": " + message);
}

DeviceProfileEntry parse_device(const Json& entry, std::string_view source) {
    if (!entry.is_object() || !entry.contains("hardware_class") || !entry.contains("routes")) {
        fail(source, "a device entry needs hardware_class and routes");
    }
    DeviceProfileEntry parsed_entry;
    ops::DeviceRouteProfile& profile = parsed_entry.profile;
    profile.hardware_class = entry.at("hardware_class").get<std::string>();
    if (profile.hardware_class.empty()) { fail(source, "hardware_class is empty"); }
    profile.multiprocessors = entry.value("multiprocessors", 0);
    profile.origin          = entry.value("origin", std::string(source));
    if (entry.contains("calibration")) {
        const Json& block = entry.at("calibration");
        if (!block.is_object() || !block.contains("route_catalog") ||
            !block.at("route_catalog").is_string() ||
            block.at("route_catalog").get<std::string>().empty()) {
            fail(source, profile.hardware_class + ": calibration needs a route_catalog");
        }
        DeviceProfileCalibration calibration;
        calibration.route_catalog = block.at("route_catalog").get<std::string>();
        calibration.ninfer_build  = block.value("ninfer_build", std::string());
        calibration.cuda_driver   = block.value("cuda_driver", 0);
        calibration.cuda_runtime  = block.value("cuda_runtime", 0);
        calibration.date          = block.value("date", std::string());
        parsed_entry.calibration  = std::move(calibration);
    }
    const Json& routes      = entry.at("routes");
    if (!routes.is_object()) { fail(source, "routes must be an object"); }
    for (const auto& [key, bands] : routes.items()) {
        if (!bands.is_array() || bands.empty()) { fail(source, key + ": bands must be a nonempty array"); }
        std::vector<ops::DeviceRouteBand> parsed;
        std::int64_t previous = 0;
        for (const Json& band : bands) {
            if (!band.is_array() || band.size() != 2 || !band[0].is_number_integer() ||
                !band[1].is_string()) {
                fail(source, key + ": a band is [last, schedule]");
            }
            const std::int64_t last = band[0].get<std::int64_t>();
            if (last <= previous || last > std::numeric_limits<std::int32_t>::max()) {
                fail(source, key + ": band bounds must increase");
            }
            previous = last;
            parsed.push_back({static_cast<std::int32_t>(last), band[1].get<std::string>()});
        }
        profile.routes.emplace(key, std::move(parsed));
    }
    return parsed_entry;
}

Json device_json(const DeviceProfileEntry& entry) {
    const ops::DeviceRouteProfile& profile = entry.profile;
    Json routes = Json::object();
    for (const auto& [key, bands] : profile.routes) {
        Json list = Json::array();
        for (const ops::DeviceRouteBand& band : bands) { list.push_back(Json::array({band.last, band.schedule})); }
        routes[key] = std::move(list);
    }
    Json device{{"hardware_class", profile.hardware_class},
                {"multiprocessors", profile.multiprocessors},
                {"origin", profile.origin}};
    if (entry.calibration) {
        device["calibration"] = Json{{"route_catalog", entry.calibration->route_catalog},
                                     {"ninfer_build", entry.calibration->ninfer_build},
                                     {"cuda_driver", entry.calibration->cuda_driver},
                                     {"cuda_runtime", entry.calibration->cuda_runtime},
                                     {"date", entry.calibration->date}};
    }
    device["routes"] = std::move(routes);
    return device;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("failed to open device profiles: " + path.string()); }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

const DeviceProfileEntry* find_in(const std::vector<DeviceProfileEntry>& entries,
                                  std::string_view hardware_class, int multiprocessors) {
    for (const DeviceProfileEntry& entry : entries) {
        if (entry.profile.hardware_class == hardware_class &&
            (entry.profile.multiprocessors == 0 || entry.profile.multiprocessors == multiprocessors)) {
            return &entry;
        }
    }
    return nullptr;
}

// A ninfer.device-route-profiles document of another schema_version than this build's.
class OtherSchemaVersion : public std::invalid_argument {
public:
    OtherSchemaVersion(const std::string& message, bool older)
        : std::invalid_argument(message), older(older) {}
    bool older; // written by an older build (or with no version at all)
};

std::string utc_now_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

} // namespace

std::vector<DeviceProfileEntry> parse_device_route_profiles(std::string_view json,
                                                            std::string_view source_name) {
    Json document;
    try {
        document = Json::parse(json.begin(), json.end());
    } catch (const nlohmann::json::exception& error) {
        fail(source_name, std::string("invalid JSON: ") + error.what());
    }
    if (!document.is_object() || document.value("schema", std::string()) != kSchema) {
        fail(source_name, "not a ninfer.device-route-profiles document");
    }
    const Json version =
        document.contains("schema_version") ? document.at("schema_version") : Json();
    if (!version.is_number_integer() || version.get<int>() != kSchemaVersion) {
        throw OtherSchemaVersion(
            std::string(source_name) + ": device profiles of schema_version " + version.dump() +
            ", this build reads only version " + std::to_string(kSchemaVersion) +
            "; recalibrate (ninfer-calibrate, or --device-profile calibrate), which replaces the file",
            !version.is_number_integer() || version.get<int>() < kSchemaVersion);
    }
    if (!document.contains("devices") || !document.at("devices").is_array()) {
        fail(source_name, "devices must be an array");
    }
    std::vector<DeviceProfileEntry> entries;
    for (const Json& entry : document.at("devices")) {
        entries.push_back(parse_device(entry, source_name));
    }
    return entries;
}

std::string serialize_device_route_profiles(const std::vector<DeviceProfileEntry>& entries) {
    Json devices = Json::array();
    for (const DeviceProfileEntry& entry : entries) { devices.push_back(device_json(entry)); }
    const Json document{{"schema", kSchema}, {"schema_version", kSchemaVersion}, {"devices", std::move(devices)}};
    return document.dump(1) + '\n';
}

std::filesystem::path default_device_profile_path() {
    if (const char* explicit_path = std::getenv("NINFER_DEVICE_PROFILES");
        explicit_path != nullptr && explicit_path[0] != '\0') {
        return explicit_path;
    }
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && local[0] != '\0') {
        return std::filesystem::path(local) / "ninfer" / "device-profiles.json";
    }
#endif
    if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache != nullptr && cache[0] != '\0') {
        return std::filesystem::path(cache) / "ninfer" / "device-profiles.json";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "ninfer" / "device-profiles.json";
    }
    return std::filesystem::path("device-profiles.json");
}

DeviceProfileEntry calibrated_device_profile_entry(ops::DeviceRouteProfile measured,
                                                   std::string route_catalog) {
    DeviceProfileEntry entry;
    entry.profile = std::move(measured);
    DeviceProfileCalibration calibration;
    calibration.route_catalog = std::move(route_catalog);
    calibration.ninfer_build  = std::string(ninfer_build_id());
    if (cudaDriverGetVersion(&calibration.cuda_driver) != cudaSuccess) {
        calibration.cuda_driver = 0;
    }
    if (cudaRuntimeGetVersion(&calibration.cuda_runtime) != cudaSuccess) {
        calibration.cuda_runtime = 0;
    }
    calibration.date  = utc_now_iso8601();
    entry.calibration = std::move(calibration);
    return entry;
}

ops::DeviceRouteProfile layer_device_route_profiles(const ops::DeviceRouteProfile& upper,
                                                    const ops::DeviceRouteProfile& lower) {
    ops::DeviceRouteProfile layered = lower;
    layered.hardware_class          = upper.hardware_class;
    layered.multiprocessors         = upper.multiprocessors;
    layered.origin                  = upper.origin + " over " + lower.origin;
    for (const auto& [key, bands] : upper.routes) {
        const auto found = layered.routes.find(key);
        if (found == layered.routes.end() || bands.empty()) {
            layered.routes.insert_or_assign(key, bands);
            continue;
        }
        // Past the upper layer's last band, the lower layer's bands answer; the first one kept is
        // clipped to start after that band, as bands start after their predecessor.
        std::vector<ops::DeviceRouteBand> merged = bands;
        for (const ops::DeviceRouteBand& band : found->second) {
            if (band.last <= merged.back().last) { continue; }
            if (band.schedule == merged.back().schedule) {
                merged.back().last = band.last;
            } else {
                merged.push_back(band);
            }
        }
        found->second = std::move(merged);
    }
    return layered;
}

std::optional<ops::DeviceRouteProfile> compiled_device_route_profile(std::string_view hardware_class,
                                                                     int multiprocessors) {
    static const std::vector<DeviceProfileEntry> compiled =
        parse_device_route_profiles(compiled_device_route_profiles_json(), "compiled device profiles");
    if (const DeviceProfileEntry* found = find_in(compiled, hardware_class, multiprocessors)) {
        return found->profile;
    }
    return std::nullopt;
}

std::optional<ops::DeviceRouteProfile> find_device_route_profile(
    std::string_view hardware_class, int multiprocessors, const std::filesystem::path& path,
    std::string_view route_catalog, const std::function<void(const std::string&)>& log) {
    const auto note = [&](const std::string& line) {
        if (log) { log(line); }
    };
    std::optional<ops::DeviceRouteProfile> measured;
    std::error_code error;
    if (!path.empty() && std::filesystem::exists(path, error)) {
        std::vector<DeviceProfileEntry> entries;
        try {
            entries = parse_device_route_profiles(read_file(path), path.string());
        } catch (const std::exception& failure) {
            note(std::string("device profile file skipped: ") + failure.what());
        }
        if (const DeviceProfileEntry* found = find_in(entries, hardware_class, multiprocessors)) {
            if (!found->calibration) {
                note(path.string() + ": the entry for " + std::string(hardware_class) +
                     " records no route catalog and is ignored; recalibrate (ninfer-calibrate, or "
                     "--device-profile calibrate)");
            } else if (found->calibration->route_catalog != route_catalog) {
                note(path.string() + ": the entry for " + std::string(hardware_class) +
                     " was calibrated against route catalog " + found->calibration->route_catalog +
                     " (build " + found->calibration->ninfer_build + "), this build's is " +
                     std::string(route_catalog) +
                     "; it is ignored. Recalibrate (ninfer-calibrate, or --device-profile calibrate)");
            } else {
                measured = found->profile;
            }
        }
    }
    std::optional<ops::DeviceRouteProfile> compiled =
        compiled_device_route_profile(hardware_class, multiprocessors);
    if (measured && compiled) { return layer_device_route_profiles(*measured, *compiled); }
    return measured ? measured : compiled;
}

DeviceProfileEntry upsert_device_route_profile_atomic(const std::filesystem::path& path,
                                                      const DeviceProfileEntry& entry) {
    std::vector<DeviceProfileEntry> entries;
    std::error_code error;
    if (std::filesystem::exists(path, error)) {
        try {
            entries = parse_device_route_profiles(read_file(path), path.string());
        } catch (const OtherSchemaVersion& outdated) {
            if (!outdated.older) { throw; }
            entries.clear(); // unusable by this build; replaced by this entry
        }
    }
    const ops::DeviceRouteProfile& profile = entry.profile;
    DeviceProfileEntry stored              = entry;
    bool replaced                          = false;
    for (DeviceProfileEntry& existing : entries) {
        if (existing.profile.hardware_class != profile.hardware_class ||
            existing.profile.multiprocessors != profile.multiprocessors) {
            continue;
        }
        if (!replaced && existing.calibration && entry.calibration &&
            existing.calibration->route_catalog == entry.calibration->route_catalog) {
            for (const auto& [key, bands] : existing.profile.routes) {
                stored.profile.routes.try_emplace(key, bands);
            }
        }
        existing = stored;
        replaced = true;
    }
    if (!replaced) { entries.push_back(stored); }
    const std::string serialized = serialize_device_route_profiles(entries);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(process_id()) + "." +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("failed to write device profiles: " + temporary.string()); }
        output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
        if (!output) { throw std::runtime_error("failed to write device profiles: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, path);
    return stored;
}

} // namespace ninfer::runtime
