#include "calibration/route_catalog.h"
#include "ops/common/device_route.h"
#include "ops/linear/common/route_table.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"
#include "runtime/engine/device_profiles_builtin.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

constexpr std::string_view kCatalog = "0123456789abcdef";

ninfer::ops::DeviceRouteProfile sample_profile() {
    ninfer::ops::DeviceRouteProfile profile;
    profile.hardware_class  = "test-gpu-sm86";
    profile.multiprocessors = 82;
    profile.origin          = "unit test";
    profile.routes["attn_i8_small/h24/rk8v4/w2"] = {{65536, ""}, {1048576, "4x2x32e"}};
    profile.routes["attn_pv_f16"]                = {{1, "on"}};
    return profile;
}

ninfer::runtime::DeviceProfileEntry sample_entry(std::string_view catalog = kCatalog) {
    ninfer::runtime::DeviceProfileEntry entry;
    entry.profile     = sample_profile();
    entry.calibration = ninfer::runtime::DeviceProfileCalibration{
        std::string(catalog), "v0.0.0-test", 12080, 12080, "2026-10-02T00:00:00Z"};
    return entry;
}

bool same_bands(const std::vector<ninfer::ops::DeviceRouteBand>& a,
                const std::vector<ninfer::ops::DeviceRouteBand>& b) {
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].last != b[i].last || a[i].schedule != b[i].schedule) { return false; }
    }
    return true;
}

std::filesystem::path temporary_profile_path(const char* tag) {
    return std::filesystem::temp_directory_path() /
           (std::string("ninfer-device-profile-") + tag + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
}

void write_text(const std::filesystem::path& path, std::string_view text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void test_round_trip() {
    const ninfer::runtime::DeviceProfileEntry entry = sample_entry();
    const std::string text = ninfer::runtime::serialize_device_route_profiles({entry});
    const auto parsed      = ninfer::runtime::parse_device_route_profiles(text, "round trip");
    expect(parsed.size() == 1, "one device parses back");
    expect(parsed[0].profile.hardware_class == entry.profile.hardware_class, "hardware class survives");
    expect(parsed[0].profile.multiprocessors == 82, "SM count survives");
    bool same = parsed[0].profile.routes.size() == entry.profile.routes.size();
    for (const auto& [key, bands] : entry.profile.routes) {
        const auto found = parsed[0].profile.routes.find(key);
        same = same && found != parsed[0].profile.routes.end() && same_bands(found->second, bands);
    }
    expect(same, "route bands survive");
    expect(parsed[0].calibration && parsed[0].calibration->route_catalog == kCatalog &&
               parsed[0].calibration->ninfer_build == "v0.0.0-test" &&
               parsed[0].calibration->cuda_driver == 12080 &&
               parsed[0].calibration->cuda_runtime == 12080 &&
               parsed[0].calibration->date == "2026-10-02T00:00:00Z",
           "the calibration provenance survives");
}

void test_rejects_other_documents() {
    const auto refused = [](std::string_view text, std::string_view needle) {
        try {
            (void)ninfer::runtime::parse_device_route_profiles(text, "document");
        } catch (const std::exception& error) {
            return std::string_view(error.what()).find(needle) != std::string_view::npos;
        }
        return false;
    };
    expect(refused(R"({"schema": "other", "devices": []})", "not a ninfer.device-route-profiles"),
           "a document of another schema is refused");
    expect(refused(R"({"schema": "ninfer.device-route-profiles", "schema_version": 1, "devices": []})",
                   "recalibrate"),
           "a schema_version 1 document is refused with a message to recalibrate");
    expect(refused(R"({"schema": "ninfer.device-route-profiles", "schema_version": 2, "devices": [
                      {"hardware_class": "x", "calibration": {"ninfer_build": "b"}, "routes": {}}]})",
                   "route_catalog"),
           "a calibration block without a route catalog is refused");
}

void test_compiled_table_parses() {
    const auto compiled = ninfer::runtime::parse_device_route_profiles(
        ninfer::runtime::compiled_device_route_profiles_json(), "compiled");
    for (const auto& entry : compiled) {
        expect(!entry.profile.hardware_class.empty(), "every compiled entry names its hardware class");
        expect(entry.profile.multiprocessors > 0, "every compiled entry names its SM count");
    }
}

// The code tier a compiled entry's card runs (its hardware class ends in its compute capability).
ninfer::ops::DeviceTier compiled_entry_tier(const std::string& hardware_class) {
    using ninfer::ops::DeviceTier;
    if (hardware_class.ends_with("-sm70")) { return DeviceTier::Volta; }
    if (hardware_class.ends_with("-sm75")) { return DeviceTier::Turing; }
    if (hardware_class.ends_with("-sm120")) { return DeviceTier::Blackwell; }
    return DeviceTier::Ampere;
}

// The compiled table may only name, for a key calibration measures, schedules calibration knows on
// the entry's card: a schedule the catalog dropped or renamed would silently keep the compiled
// route.
void test_compiled_table_names_catalog_schedules() {
    const auto compiled = ninfer::runtime::parse_device_route_profiles(
        ninfer::runtime::compiled_device_route_profiles_json(), "compiled");
    for (const auto& entry : compiled) {
        const auto& catalog = ninfer::calibration::calibration_route_catalog(
            compiled_entry_tier(entry.profile.hardware_class));
        for (const auto& [key, bands] : entry.profile.routes) {
            const auto found = std::find_if(catalog.begin(), catalog.end(),
                                            [&](const auto& item) { return item.key == key; });
            if (found == catalog.end()) { continue; }
            for (const auto& band : bands) {
                const bool known =
                    band.schedule.empty() ||
                    std::find(found->candidates.begin(), found->candidates.end(), band.schedule) !=
                        found->candidates.end();
                if (!known) {
                    std::cerr << entry.profile.hardware_class << ' ' << key << ' ' << band.schedule << '\n';
                }
                expect(known, "a compiled route names a catalog candidate");
            }
        }
    }
    using ninfer::ops::DeviceTier;
    for (const DeviceTier tier :
         {DeviceTier::Volta, DeviceTier::Turing, DeviceTier::Ampere, DeviceTier::Blackwell}) {
        const std::string& digest = ninfer::calibration::calibration_route_catalog_digest(tier);
        expect(digest.size() == 16 &&
                   digest.find_first_not_of("0123456789abcdef") == std::string::npos,
               "the route catalog digest is 16 hex digits");
    }
    // A 12.0 card runs Blackwell's code from a build with 120a and Ampere's from one without; a
    // profile it calibrated under either applies under the other.
    expect(ninfer::calibration::calibration_route_catalog_digest(DeviceTier::Ampere) ==
               ninfer::calibration::calibration_route_catalog_digest(DeviceTier::Blackwell),
           "a 12.0 card has one route catalog whichever tier runs it");
    expect(ninfer::calibration::calibration_route_catalog_digest(DeviceTier::Turing) !=
                   ninfer::calibration::calibration_route_catalog_digest(DeviceTier::Ampere) &&
               ninfer::calibration::calibration_route_catalog_digest(DeviceTier::Volta) !=
                   ninfer::calibration::calibration_route_catalog_digest(DeviceTier::Turing),
           "the Volta, Turing and Ampere catalogs differ");
}

// The cards that ship a measured profile, by the name and compute capability the driver reports, so
// a change to the hardware-class slug cannot silently orphan their entries.
void test_builtin_parts() {
    struct Part {
        const char* name;
        int major;
        int minor;
        int multiprocessors;
    };

    const Part parts[] = {
        {"NVIDIA GeForce RTX 3090", 8, 6, 82},
        {"NVIDIA GeForce RTX 4090", 8, 9, 128},
        {"NVIDIA GeForce RTX 5090", 12, 0, 170},
        {"NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 12, 0, 188},
        {"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", 12, 0, 188},
        {"NVIDIA RTX PRO 6000 Blackwell Server Edition", 12, 0, 188},
        {"Tesla V100-SXM2-16GB", 7, 0, 80},
        {"Tesla V100-SXM2-32GB", 7, 0, 80},
        {"NVIDIA GeForce RTX 5060 Ti", 12, 0, 36},
        {"NVIDIA GeForce RTX 5070", 12, 0, 48},
        {"NVIDIA GeForce RTX 5070 Ti", 12, 0, 70},
        {"NVIDIA GeForce RTX 5080", 12, 0, 84},
        {"NVIDIA GeForce RTX 5090 D v2", 12, 0, 170},
        {"NVIDIA RTX PRO 4000 Blackwell", 12, 0, 70},
        {"NVIDIA RTX PRO 4500 Blackwell", 12, 0, 82},
        {"NVIDIA RTX PRO 4500 Blackwell Server Edition", 12, 0, 82},
        {"NVIDIA RTX PRO 5000 Blackwell", 12, 0, 110},
        {"NVIDIA GeForce RTX 4060 Ti", 8, 9, 34},
        {"NVIDIA GeForce RTX 4070", 8, 9, 46},
        {"NVIDIA GeForce RTX 4070 Ti", 8, 9, 60},
        {"NVIDIA GeForce RTX 4070 Ti SUPER", 8, 9, 66},
        {"NVIDIA GeForce RTX 4080", 8, 9, 76},
        {"NVIDIA GeForce RTX 4080 SUPER", 8, 9, 80},
        {"NVIDIA L4", 8, 9, 58},
        {"NVIDIA L40S", 8, 9, 142},
        {"NVIDIA RTX 4000 Ada Generation", 8, 9, 48},
        {"NVIDIA RTX 5000 Ada Generation", 8, 9, 100},
        {"NVIDIA RTX 6000 Ada Generation", 8, 9, 142},
        {"NVIDIA A10", 8, 6, 72},
        {"NVIDIA A40", 8, 6, 84},
        {"NVIDIA GeForce RTX 3060", 8, 6, 28},
        {"NVIDIA GeForce RTX 3060 Ti", 8, 6, 38},
        {"NVIDIA GeForce RTX 3070", 8, 6, 46},
        {"NVIDIA GeForce RTX 3080", 8, 6, 68},
        {"NVIDIA GeForce RTX 3080 Ti", 8, 6, 80},
        {"NVIDIA RTX A5000", 8, 6, 64},
        {"NVIDIA A100 80GB PCIe", 8, 0, 108},
        {"NVIDIA A100-PCIE-40GB", 8, 0, 108},
        {"NVIDIA A100-SXM4-40GB", 8, 0, 108},
        {"NVIDIA A100-SXM4-80GB", 8, 0, 108},
        {"NVIDIA GeForce RTX 2060 SUPER", 7, 5, 34},
        {"NVIDIA GeForce RTX 2070", 7, 5, 36},
        {"NVIDIA GeForce RTX 2070 SUPER", 7, 5, 40},
        {"NVIDIA GeForce RTX 2080 Ti", 7, 5, 68},
        {"Quadro RTX 6000", 7, 5, 72},
        {"Quadro RTX 8000", 7, 5, 72},
        {"Tesla T4", 7, 5, 40},
    };
    for (const Part& part : parts) {
        const auto found = ninfer::runtime::find_device_route_profile(
            ninfer::runtime::context_cost_hardware_class(part.name, part.major, part.minor),
            part.multiprocessors, {}, kCatalog);
        expect(found.has_value() && !found->routes.empty(), part.name);
    }
}

void test_layering() {
    ninfer::ops::DeviceRouteProfile lower;
    lower.hardware_class = "test-gpu-sm86";
    lower.origin         = "built in";
    lower.routes["only_lower"] = {{64, "a"}};
    lower.routes["both"]       = {{8, "x"}, {64, "y"}, {1024, "z"}};
    lower.routes["covered"]    = {{16, "x"}};
    ninfer::ops::DeviceRouteProfile upper;
    upper.hardware_class  = "test-gpu-sm86";
    upper.multiprocessors = 82;
    upper.origin          = "calibration";
    upper.routes["only_upper"] = {{4, "b"}};
    upper.routes["both"]       = {{16, ""}, {32, "y"}};
    upper.routes["covered"]    = {{32, ""}};
    const auto layered = ninfer::runtime::layer_device_route_profiles(upper, lower);
    expect(layered.multiprocessors == 82, "the layered profile takes the upper SM count");
    expect(same_bands(layered.routes.at("only_lower"), {{64, "a"}}), "a lower-only key survives");
    expect(same_bands(layered.routes.at("only_upper"), {{4, "b"}}), "an upper-only key applies");
    expect(same_bands(layered.routes.at("both"), {{16, ""}, {64, "y"}, {1024, "z"}}),
           "the upper bands answer to their last width, the lower ones past it");
    expect(same_bands(layered.routes.at("covered"), {{32, ""}}),
           "a measured compiled route overrides the lower layer's choice");
}

void test_file_lookup() {
    const std::filesystem::path path = temporary_profile_path("lookup");
    (void)ninfer::runtime::upsert_device_route_profile_atomic(path, sample_entry());
    auto found = ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path, kCatalog);
    expect(found.has_value(), "the stored profile is found");
    expect(found && found->routes.count("attn_pv_f16") == 1, "its routes come back");
    expect(!ninfer::runtime::find_device_route_profile("test-gpu-sm86", 84, path, kCatalog).has_value(),
           "a part with another SM count does not take the entry");

    std::vector<std::string> lines;
    const auto log = [&](const std::string& line) { lines.push_back(line); };
    expect(!ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path, "fedcba9876543210", log)
                .has_value(),
           "an entry of another route catalog is not applied");
    expect(lines.size() == 1 && lines[0].find("Recalibrate") != std::string::npos,
           "an ignored entry is logged with the remedy");

    // A calibration over the same catalog that measured only some keys keeps the others.
    ninfer::runtime::DeviceProfileEntry partial = sample_entry();
    partial.profile.routes.erase("attn_pv_f16");
    partial.profile.routes["attn_i8_small/h24/rk8v4/w2"] = {{1048576, ""}};
    const auto stored = ninfer::runtime::upsert_device_route_profile_atomic(path, partial);
    found = ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path, kCatalog);
    expect(stored.profile.routes.count("attn_pv_f16") == 1 && found &&
               found->routes.count("attn_pv_f16") == 1 &&
               same_bands(found->routes.at("attn_i8_small/h24/rk8v4/w2"), {{1048576, ""}}),
           "an upsert over the same catalog replaces the measured keys and keeps the rest");

    // A calibration over another catalog replaces the entry.
    ninfer::runtime::DeviceProfileEntry other = sample_entry("fedcba9876543210");
    other.profile.routes.erase("attn_pv_f16");
    (void)ninfer::runtime::upsert_device_route_profile_atomic(path, other);
    found = ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path, "fedcba9876543210");
    expect(found && found->routes.count("attn_pv_f16") == 0,
           "an upsert over another catalog replaces the entry");
    std::filesystem::remove(path);
}

// A profile file never shadows a built-in key it does not name, so calibrating a part with a
// built-in profile keeps every key calibration does not measure.
void test_file_layers_over_builtin() {
    const std::string rtx3090 =
        ninfer::runtime::context_cost_hardware_class("NVIDIA GeForce RTX 3090", 8, 6);
    const auto builtin = ninfer::runtime::compiled_device_route_profile(rtx3090, 82);
    expect(builtin.has_value(), "the RTX 3090 has a built-in profile");
    if (!builtin) { return; }
    ninfer::runtime::DeviceProfileEntry entry = sample_entry();
    entry.profile.hardware_class = rtx3090;
    entry.profile.routes.clear();
    entry.profile.routes["attn_pv_f16"] = {{1, ""}};
    const std::filesystem::path path = temporary_profile_path("layer");
    (void)ninfer::runtime::upsert_device_route_profile_atomic(path, entry);
    const auto found = ninfer::runtime::find_device_route_profile(rtx3090, 82, path, kCatalog);
    bool kept = found.has_value();
    for (const auto& [key, bands] : builtin->routes) {
        if (key == "attn_pv_f16") { continue; }
        kept = kept && found->routes.count(key) == 1 && same_bands(found->routes.at(key), bands);
    }
    expect(kept, "every built-in key the file does not name keeps its bands");
    expect(found && same_bands(found->routes.at("attn_pv_f16"), {{1, ""}}), "the file's key wins");

    // A stale entry leaves the built-in profile in force.
    const auto stale = ninfer::runtime::find_device_route_profile(rtx3090, 82, path, "fedcba9876543210");
    expect(stale && stale->routes.size() == builtin->routes.size(),
           "an entry of another catalog falls back to the built-in profile");
    std::filesystem::remove(path);
}

void test_schema_versions_on_disk() {
    const std::filesystem::path path = temporary_profile_path("schema");
    write_text(path, R"({"schema": "ninfer.device-route-profiles", "schema_version": 1, "devices": [
        {"hardware_class": "test-gpu-sm86", "multiprocessors": 82, "routes": {"attn_pv_f16": [[1, "on"]]}}]})");
    std::vector<std::string> lines;
    const auto log = [&](const std::string& line) { lines.push_back(line); };
    expect(!ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path, kCatalog, log).has_value(),
           "a schema_version 1 file is not applied");
    expect(lines.size() == 1 && lines[0].find("recalibrate") != std::string::npos,
           "a schema_version 1 file is reported with the remedy");
    (void)ninfer::runtime::upsert_device_route_profile_atomic(path, sample_entry());
    expect(ninfer::runtime::find_device_route_profile("test-gpu-sm86", 82, path, kCatalog).has_value(),
           "calibrating replaces a schema_version 1 file");

    write_text(path, R"({"schema": "ninfer.device-route-profiles", "schema_version": 3, "devices": []})");
    bool threw = false;
    try {
        (void)ninfer::runtime::upsert_device_route_profile_atomic(path, sample_entry());
    } catch (const std::exception&) { threw = true; }
    expect(threw, "a file of a newer schema_version is not overwritten");
    std::filesystem::remove(path);
}

void test_bands_and_force() {
    using ninfer::ops::device_route_schedule;
    ninfer::ops::install_device_route_profile(
        std::make_shared<const ninfer::ops::DeviceRouteProfile>(sample_profile()));
    expect(device_route_schedule("attn_i8_small/h24/rk8v4/w2", 8192).empty(),
           "a width in a compiled band keeps the compiled route");
    expect(device_route_schedule("attn_i8_small/h24/rk8v4/w2", 262144) == "4x2x32e",
           "a width in a routed band takes its schedule");
    expect(device_route_schedule("attn_i8_small/h24/rk8v4/w2", 2000000).empty(),
           "a width past the last band keeps the compiled route");
    expect(device_route_schedule("unknown/key", 1).empty(), "an unknown key keeps the compiled route");
    {
        const ninfer::ops::DeviceRouteForce force("attn_pv_f16", "");
        expect(device_route_schedule("attn_pv_f16", 1).empty(), "a forced route wins over the profile");
    }
    expect(device_route_schedule("attn_pv_f16", 1) == "on", "the profile returns after the force ends");
    ninfer::ops::install_device_route_profile(nullptr);
    expect(device_route_schedule("attn_pv_f16", 1).empty(), "removing the profile restores the tables");
}

// Past the widths the route bands were measured at, a call takes the table its Op names for wide
// calls unless a profile band reaches the width; FP8 A16 calls name the unified one.
void test_wide_tables() {
    using ninfer::ops::detail::fused_route_table;
    using ninfer::ops::detail::kRouteBandsMeasuredTo;
    using ninfer::ops::detail::LinearRouteFamily;
    using ninfer::ops::detail::linear_route_table;
    using ninfer::ops::detail::LinearRouteTable;
    using ninfer::ops::detail::fp8_wide_route_table;
    using ninfer::ops::LinearPolicy;
    expect(fp8_wide_route_table(LinearPolicy::A16Only) == LinearRouteTable::Unified &&
               fp8_wide_route_table(LinearPolicy::AllowA8Int) == LinearRouteTable::Unified &&
               fp8_wide_route_table(LinearPolicy::AllowA8) == LinearRouteTable::Legacy,
           "an FP8 A16 call names the unified table for wide calls, an A8 call the legacy one");
    if (!ninfer::ops::detail::unified_linear_routes_here()) {
        expect(linear_route_table(LinearRouteFamily::Fp8, 4096, LinearRouteTable::Unified) ==
                   LinearRouteTable::Legacy,
               "a card without the unified table takes the legacy one at every width");
        return;
    }
    constexpr auto kUnified = LinearRouteTable::Unified;
    constexpr auto kLegacy  = LinearRouteTable::Legacy;
    const std::int32_t past = kRouteBandsMeasuredTo + 1;
    expect(linear_route_table(LinearRouteFamily::Fp8, past, kUnified) == kUnified &&
               linear_route_table(LinearRouteFamily::Fp8, 1 << 20, kUnified) == kUnified,
           "a wide FP8 width takes the table named for wide calls");
    expect(linear_route_table(LinearRouteFamily::Fp8, past) == kLegacy &&
               linear_route_table(LinearRouteFamily::Q4, 4096) == kLegacy,
           "a wide width takes the legacy table by default");

    ninfer::ops::DeviceRouteProfile profile = sample_profile();
    profile.routes["unified/fp8_linear_add"] = {{4, ""}, {1024, "unified"}};
    profile.routes["unified/measured_wide"]  = {{8192, ""}};
    ninfer::ops::install_device_route_profile(
        std::make_shared<const ninfer::ops::DeviceRouteProfile>(profile));
    expect(fused_route_table("unified/fp8_linear_add", 2, kUnified) == kLegacy &&
               fused_route_table("unified/fp8_linear_add", 1024, kUnified) == kUnified,
           "a width in a profile band takes the band's table");
    expect(fused_route_table("unified/fp8_linear_add", past) == kLegacy &&
               fused_route_table("unified/fp8_linear_add", past, kUnified) == kUnified,
           "a width past the profile's bands takes the table named for wide calls");
    expect(fused_route_table("unified/missing", 512, kUnified) == kLegacy &&
               fused_route_table("unified/missing", 4096, kUnified) == kUnified,
           "a key the profile lacks keeps the legacy table up to the measured widths only");
    expect(fused_route_table("unified/measured_wide", 4096, kUnified) == kLegacy,
           "a profile band that reaches a wide width answers for it");
    {
        const ninfer::ops::DeviceRouteForce force("unified/missing", "");
        expect(fused_route_table("unified/missing", 4096, kUnified) == kLegacy,
               "a forced route answers at any width");
    }
    ninfer::ops::install_device_route_profile(nullptr);
}

} // namespace

int main() {
    test_round_trip();
    test_rejects_other_documents();
    test_compiled_table_parses();
    test_compiled_table_names_catalog_schedules();
    test_builtin_parts();
    test_layering();
    test_file_lookup();
    test_file_layers_over_builtin();
    test_schema_versions_on_disk();
    test_bands_and_force();
    test_wide_tables();
    if (failures != 0) {
        std::cerr << failures << " device profile check(s) failed\n";
        return 1;
    }
    std::cout << "device profile tests passed\n";
    return 0;
}
