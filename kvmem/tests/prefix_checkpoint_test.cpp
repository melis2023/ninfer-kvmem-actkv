#include "kvmem/prefix_checkpoint.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace kvmem;
static void check(bool value) { if (!value) throw std::runtime_error("prefix checkpoint contract failed"); }
struct Record {
    uint64_t row = 0;
    CheckpointPurpose purposes = CheckpointPurpose::Endpoint;
    std::shared_ptr<int> state;
    uint64_t size = 8;
    CheckpointInfo checkpoint_info() const { return {row, purposes, state.get(), size}; }
};
int main() {
    // A missing state does not discard the observed equal prefix. The next
    // request must recover after one cold reconstruction, including pressure.
    CheckpointCatalog<Record> catalog({4, 32});
    catalog.remember({3465, CheckpointPurpose::QueryReplay, std::make_shared<int>(1)});
    check(!catalog.select(prefix_restore_limit(3464, 3550, 3800, true), [](const auto&) { return true; }));
    const auto learned = prefix_anchor(0, 3464, 3465);
    catalog.remember({learned, CheckpointPurpose::ReuseAnchor, std::make_shared<int>(2)});
    for (uint64_t row = 3500; row < 3512; ++row)
        catalog.remember({row, CheckpointPurpose::Endpoint, std::make_shared<int>(3)});
    const auto base = catalog.select(3464, [](const auto&) { return true; });
    check(base && base->row == 3464 && catalog.size() == 4 && catalog.bytes() == 32);
    check(!catalog.select(3464, [](const auto&) { return false; }));

    // Equal-boundary roles share one state. Rewind invalidates all later marks;
    // external active-request leases survive catalog eviction and cancellation.
    auto lease = base->state;
    catalog.truncate(3464);
    catalog.remember({3464, CheckpointPurpose::QueryReplay, lease});
    check(catalog.size() == 1 && catalog.bytes() == 8);
    check(has_purpose(catalog.front().purposes, CheckpointPurpose::ReuseAnchor));
    catalog.clear();
    check(*lease == 2);

    // Native groups are indivisible; exact endpoints remain legal. Ordinary
    // decoding leaves a row for logits and an empty input never underflows.
    check(prefix_anchor(0, 150, 300, {{100, 200}}) == 100);
    check(prefix_anchor(0, 200, 300, {{100, 200}}) == 200);
    check(prefix_restore_limit(100, 100, 100, true) == 99);
    check(prefix_restore_limit(0, 0, 0, true) == 0);
    check(prefix_anchor(200, 200, 400) == 200);
    CheckpointCatalog<Record> native_slots({3, 32});
    native_slots.slot(CheckpointPurpose::QueryReplay) = {200, CheckpointPurpose::QueryReplay, std::make_shared<int>(5)};
    native_slots.slot(CheckpointPurpose::Endpoint) = {200, CheckpointPurpose::Endpoint, std::make_shared<int>(6)};
    const auto prefer_endpoint = [](const Record & a, const Record &) {
        return has_purpose(a.purposes, CheckpointPurpose::Endpoint);
    };
    check(*native_slots.select(200, [](const auto&) { return true; }, prefer_endpoint)->state == 6);
    check(*native_slots.select(200, [](const auto & r) { return *r.state == 5; }, prefer_endpoint)->state == 5);
    catalog.remember({10, CheckpointPurpose::ReuseAnchor, std::make_shared<int>(4), 64});
    check(catalog.empty());
    std::cout << "prefix checkpoint: reconstruction, retention, leases and boundaries passed\n";
}
