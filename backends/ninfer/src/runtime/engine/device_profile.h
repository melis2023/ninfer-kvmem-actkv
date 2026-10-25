#pragma once

// Device route profiles on disk and in the binary, and their installation at Engine startup.
//
// A profile file holds one entry per hardware class (context_cost_hardware_class of the GPU):
//
//   {"schema": "ninfer.device-route-profiles", "schema_version": 2,
//    "devices": [{"hardware_class": "nvidia-geforce-rtx-3090-sm86", "multiprocessors": 82,
//                 "origin": "...",
//                 "calibration": {"route_catalog": "<digest>", "ninfer_build": "...",
//                                 "cuda_driver": 12080, "cuda_runtime": 12080,
//                                 "date": "2026-10-02T09:30:00Z"},
//                 "routes": {"<key>": [[last, "<schedule>"], ...], ...}}]}
//
// An empty schedule string keeps the compiled route for its band. Entries whose multiprocessor
// count differs from the device's are not applied: a laptop part can share a desktop part's name
// and not its SM count. "calibration" is the provenance of a measured entry; an entry in the user's
// file is applied only when its route_catalog equals the running binary's
// (calibration::calibration_route_catalog_digest()). Another schema_version is refused whole.
//
// Profiles layer per route key: the user file's entry over the compiled table's entry over the
// Op's compiled route. For a key both name, the user entry's bands answer up to its last band and
// the compiled table's bands past it.

#include "ops/common/device_route.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::runtime {

// Which binary measured an entry, against which route catalog, on which driver, and when. The SM
// count is the entry's multiprocessors.
struct DeviceProfileCalibration {
    std::string route_catalog; // calibration::calibration_route_catalog_digest() of the binary
    std::string ninfer_build;  // NINFER_BUILD_ID
    int cuda_driver  = 0;      // cudaDriverGetVersion, e.g. 12080
    int cuda_runtime = 0;      // cudaRuntimeGetVersion
    std::string date;          // UTC, ISO 8601
};

struct DeviceProfileEntry {
    ops::DeviceRouteProfile profile;
    std::optional<DeviceProfileCalibration> calibration; // set on calibrated entries
};

[[nodiscard]] std::vector<DeviceProfileEntry>
parse_device_route_profiles(std::string_view json, std::string_view source_name);
[[nodiscard]] std::string
serialize_device_route_profiles(const std::vector<DeviceProfileEntry>& entries);

// $NINFER_DEVICE_PROFILES, else $XDG_CACHE_HOME/ninfer/device-profiles.json, else
// ~/.cache/ninfer/device-profiles.json (%LOCALAPPDATA% on Windows).
[[nodiscard]] std::filesystem::path default_device_profile_path();

// The build id this binary was stamped with (cmake/GenerateBuildId.cmake).
[[nodiscard]] std::string_view ninfer_build_id();

// A calibrated entry for the routes calibration just measured on the current device: their
// provenance is this binary's build id, the CUDA driver and runtime versions, and the current date.
[[nodiscard]] DeviceProfileEntry calibrated_device_profile_entry(ops::DeviceRouteProfile measured,
                                                                 std::string route_catalog);

// `upper` over `lower`, per route key: a key only one names keeps its bands; for a key both name,
// `upper`'s bands answer up to their last width and `lower`'s past it. The hardware class and SM
// count are `upper`'s.
[[nodiscard]] ops::DeviceRouteProfile
layer_device_route_profiles(const ops::DeviceRouteProfile& upper,
                            const ops::DeviceRouteProfile& lower);

// The compiled table's entry for this hardware class and SM count.
[[nodiscard]] std::optional<ops::DeviceRouteProfile>
compiled_device_route_profile(std::string_view hardware_class, int multiprocessors);

// The profile for this hardware class and SM count: the file's calibrated entry layered over the
// compiled table's, or whichever of the two exists. A file entry measured against another route
// catalog than `route_catalog`, and a file this build cannot read, are skipped with a line to
// `log`; the compiled table's entry still applies.
[[nodiscard]] std::optional<ops::DeviceRouteProfile>
find_device_route_profile(std::string_view hardware_class, int multiprocessors,
                          const std::filesystem::path& path, std::string_view route_catalog,
                          const std::function<void(const std::string&)>& log = {});

// Stores the entry in the file, atomically, and returns the entry as stored. An existing entry of
// the same hardware class and SM count measured against the same route catalog keeps the keys the
// new one did not measure (a calibration limited by --only); any other existing entry for it is
// replaced. A file of an older schema_version is replaced as a whole: this build cannot use it.
DeviceProfileEntry upsert_device_route_profile_atomic(const std::filesystem::path& path,
                                                      const DeviceProfileEntry& entry);

} // namespace ninfer::runtime
