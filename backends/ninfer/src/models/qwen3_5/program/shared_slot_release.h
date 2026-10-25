#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

enum class SharedPrefixSlotRole : std::uint8_t {
    Free,
    ReservedCapture,
    ReservedReplacement,
    Catalogued,
    Pinned,
};

enum class SharedSlotReleaseAction : std::uint8_t {
    // The slot is not reserved: leave it exactly as it is.
    Leave,
    // Return it to the pool.
    Free,
    // The replacement it stood for is still installed, so it returns to the catalog.
    Catalogue,
};

// What happens to a shared-prefix slot when the capture transaction that reserved it is aborted.
// Fail-all cleanup releases only live (Catalogued or Pinned) slots, so a slot left in a reserved
// role would keep its KV addresses and state checkpoint reference with no owner that can free them.
// A reserved role therefore never resolves to Leave. `replacement_removed` says whether the
// aborted transaction had already released the replacement it reserved the slot for; `role` is
// the slot's role as observed.
[[nodiscard]] constexpr SharedSlotReleaseAction
resolve_shared_slot_release(bool replacement_removed, SharedPrefixSlotRole role) noexcept {
    if (role == SharedPrefixSlotRole::ReservedReplacement) {
        return replacement_removed ? SharedSlotReleaseAction::Free
                                   : SharedSlotReleaseAction::Catalogue;
    }
    if (role == SharedPrefixSlotRole::ReservedCapture) { return SharedSlotReleaseAction::Free; }
    return SharedSlotReleaseAction::Leave;
}

} // namespace ninfer::models::qwen3_5::detail
