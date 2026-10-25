#include "runtime/engine/device_profile.h"

#include "ninfer_build_id.h"

namespace ninfer::runtime {

// Its own translation unit: only this file recompiles when the build id changes.
std::string_view ninfer_build_id() { return NINFER_BUILD_ID; }

} // namespace ninfer::runtime
