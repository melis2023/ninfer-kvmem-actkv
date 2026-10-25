#pragma once

// The route catalog: every device-profile route key calibration measures, with the candidate
// schedules it times for that key. device_calibration.cu takes its candidates from here, so the
// catalog is exactly what this binary can calibrate.
//
// Its digest is stamped into every calibrated profile entry. A stored entry whose digest differs
// from the running binary's was measured over another set of keys or schedules (a schedule renamed,
// added or dropped, a key added), so it is not applied: it would shadow the built-in table with
// routes this binary may no longer have, or miss keys it now calibrates.

#include <string>
#include <string_view>
#include <vector>

namespace ninfer::calibration {

struct RouteCatalogEntry {
    std::string key;
    std::vector<std::string> candidates;
};

// In key order; built once.
[[nodiscard]] const std::vector<RouteCatalogEntry>& calibration_route_catalog();

// The candidate schedules calibration times for `key`; throws std::logic_error for a key outside
// the catalog, so a family the catalog does not list cannot be calibrated.
[[nodiscard]] const std::vector<std::string>& calibration_candidates(std::string_view key);

// 64-bit FNV-1a over the catalog's keys and candidate names, as 16 lowercase hex digits.
[[nodiscard]] const std::string& calibration_route_catalog_digest();

} // namespace ninfer::calibration
