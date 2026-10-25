// The disposition of a shared-prefix slot when the capture transaction that reserved it aborts.
// The combinations are enumerated rather than sampled: the defect this protects against was a gap
// in an if/else chain that left two reserved combinations untouched, so a test of the handled
// cases alone would pass against it. The control is the invariant that no reserved role resolves
// to Leave.

#include "models/qwen3_5/program/shared_slot_release.h"

#include <iostream>
#include <string_view>

namespace {

using ninfer::models::qwen3_5::detail::resolve_shared_slot_release;
using Action = ninfer::models::qwen3_5::detail::SharedSlotReleaseAction;
using Role   = ninfer::models::qwen3_5::detail::SharedPrefixSlotRole;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_no_reserved_role_is_left_stranded() {
    for (const bool removed : {false, true}) {
        for (const Role role : {Role::ReservedCapture, Role::ReservedReplacement}) {
            expect(resolve_shared_slot_release(removed, role) != Action::Leave,
                   "a reserved role resolved to Leave: its KV and state reference would be "
                   "stranded with no owner");
        }
    }
}

void test_replacement_flow() {
    // Reserve marks the catalogued replacement ReservedReplacement; preparation releases it and
    // turns the slot into ReservedCapture with replacement_removed set.
    expect(resolve_shared_slot_release(false, Role::ReservedReplacement) == Action::Catalogue,
           "an abort before preparation did not return the intact replacement to the catalog");
    expect(resolve_shared_slot_release(true, Role::ReservedCapture) == Action::Free,
           "an abort after the replacement was released did not free the slot");
    expect(resolve_shared_slot_release(true, Role::ReservedReplacement) == Action::Free,
           "a reserved replacement whose replacement was released stayed reserved");
}

void test_vacant_capture_flow() {
    expect(resolve_shared_slot_release(false, Role::ReservedCapture) == Action::Free,
           "a vacant capture reservation did not return to the pool");
}

void test_unreserved_roles_are_left_alone() {
    for (const bool removed : {false, true}) {
        for (const Role role : {Role::Free, Role::Catalogued, Role::Pinned}) {
            expect(resolve_shared_slot_release(removed, role) == Action::Leave,
                   "an unreserved slot was touched by the release decision");
        }
    }
}

} // namespace

int main() {
    test_no_reserved_role_is_left_stranded();
    test_replacement_flow();
    test_vacant_capture_flow();
    test_unreserved_roles_are_left_alone();
    if (failures != 0) {
        std::cerr << failures << " shared-slot release checks failed\n";
        return 1;
    }
    std::cout << "shared-slot release checks passed\n";
    return 0;
}
