#pragma once

// The route catalog: every device-profile route key calibration measures, with the candidate
// schedules it times for that key, for a card of one code tier (ops/common/device_tier.h): the
// routes a card can take follow its tier, whatever other architectures the binary carries.
// device_calibration.cu takes its candidates from here, so the catalog is exactly what this binary
// can calibrate on the card.
//
// Its digest is stamped into every calibrated profile entry. A stored entry whose digest differs
// from the running binary's for the card's tier was measured over another set of keys or schedules
// (a schedule renamed, added or dropped, a key added), so it is not applied: it would shadow the
// built-in table with routes this binary may no longer have, or miss keys it now calibrates.

#include "ops/common/device_tier.h"

#include <string>
#include <string_view>
#include <vector>

namespace ninfer::calibration {

struct RouteCatalogEntry {
    std::string key;
    std::vector<std::string> candidates;
};

// In key order; built once per tier.
[[nodiscard]] const std::vector<RouteCatalogEntry>& calibration_route_catalog(ops::DeviceTier tier);

// The candidate schedules calibration times for `key` on a card of `tier`; throws
// std::logic_error for a key outside that catalog, so a family the catalog does not list cannot be
// calibrated.
[[nodiscard]] const std::vector<std::string>& calibration_candidates(std::string_view key,
                                                                     ops::DeviceTier tier);

// 64-bit FNV-1a over the tier's catalog keys and candidate names, as 16 lowercase hex digits.
[[nodiscard]] const std::string& calibration_route_catalog_digest(ops::DeviceTier tier);

} // namespace ninfer::calibration
