#include "kvmem/checkpoint_policy.hpp"
#include <iostream>
#include <set>
#include <stdexcept>
using namespace kvmem;
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
struct Record {
    uint64_t row = 0;
    std::shared_ptr<int> data;
    CheckpointInfo checkpoint_info() const { return {row, CheckpointPurpose::None, data.get(), data ? 8U : 0U}; }
};
int main() {
    require(!checkpoint_anonymous_continuation(26,7103,7110), "shared root cannot consume an anonymous owner");
    require(!checkpoint_anonymous_continuation(3500,7103,7110), "anonymous middle edit is a branch");
    require(checkpoint_anonymous_continuation(7078,7103,7110), "ordinary tail edit can continue");
    require(checkpoint_anonymous_continuation(7103,7103,8000), "generated output can be dropped");
    require(checkpoint_anonymous_continuation(1902,7103,1902), "cancelled prefill keeps its accepted frontier");
    CheckpointPolicy p; p.enabled = true; p.count = 8; p.bytes = 64; p.interval = 16; p.min_gap = 0;
    CheckpointArchive<Record> a(p);
    auto root = std::make_shared<int>(1);
    a.remember({2, root}, CoveragePurpose::Root);
    for (uint64_t row = 16; row <= 512; row += 16)
        a.remember({row, std::make_shared<int>(int(row))}, CoveragePurpose::Periodic);
    require(a.size() == 8 && a.bytes() == 64, "bounded retained records");
    auto valid = [](const Record &) { return true; };
    require(a.select(2, valid)->data == root, "root survived long prefill");
    require(a.select(512, valid)->row == 512, "newest survived");
    const auto middle = a.select(256, valid);
    require(middle && middle->row > 2, "middle edit has coverage instead of only the root");
    auto retained = middle->data;
    const auto boundary = middle->row;
    a.truncate(boundary);
    require(a.select(512, valid)->row == boundary, "rewind removed later states");
    auto replacement = std::make_shared<int>(-1);
    a.remember({boundary, replacement}, CoveragePurpose::Tail);
    require(a.select(boundary, valid)->data == replacement && *retained == int(boundary), "same position replaces version without changing leased state");
    a.clear(); require(*root == 1, "external lease survives archive clear");
    a.remember({10, replacement}, CoveragePurpose::Root);
    a.remember({10, replacement}, CoveragePurpose::Anchor);
    require(a.size() == 1 && a.bytes() == 8, "same payload not copied or charged twice");
    a.erase_invalid([](const Record&) { return false; });
    require(!a.select(100, valid), "revoked prefix unavailable");
    CheckpointPolicy small=p; small.count=3; small.bytes=24;
    CheckpointArchive<Record> shared(small);
    auto fixed=std::make_shared<int>(11), external=std::make_shared<int>(12);
    shared.remember({10,fixed},CoveragePurpose::Root);
    shared.remember({20,std::make_shared<int>(20)},CoveragePurpose::Periodic);
    shared.remember({30,std::make_shared<int>(30)},CoveragePurpose::Endpoint);
    shared.prune({{10,CheckpointPurpose::Endpoint,fixed.get(),8},
                  {15,CheckpointPurpose::ReuseAnchor,external.get(),8}});
    require(shared.size()==2 && shared.bytes()==16,"fixed slots share the combined directory budget");
    require(shared.select(10,valid)->data==fixed && shared.select(100,valid)->row==30,
            "aliases charged once and optional newest retained");
    for (size_t cap = 0; cap < 8; ++cap) for (uint64_t budget = 0; budget < 80; budget += 7) {
        std::vector<CoverageInfo> infos;
        std::vector<int> identities(20);
        for (size_t i = 0; i < 20; ++i)
            infos.push_back({i * 16, 8, i, &identities[i], i == 0 ? CoveragePurpose::Root : i == 18 ? CoveragePurpose::Anchor : CoveragePurpose::Periodic});
        const auto keep = retain_checkpoint_coverage(infos, cap, budget, 16);
        require(keep.size() <= cap && keep.size()*8 <= budget, "all tiny budgets obey both limits");
        if (!keep.empty()) require(keep.back() == 19, "small optional budgets retain newest");
    }
    CheckpointCapturePlan plan(0, 16, 3, 50);
    require(plan.cut(0, 8) == 3 && plan.due(3) == CoveragePurpose::Root, "semantic cut");
    plan.saved(3);
    require(!plan.due(18) && plan.due(24) == CoveragePurpose::Periodic, "periodic capture waits for actual chunk boundary");
    plan.saved(24);
    require(plan.cut(48, 56) == 50 && plan.due(50) == CoveragePurpose::Tail, "tail cut");
    plan.saved(50);
    require(!plan.due(50), "no duplicate capture");
    CheckpointPolicy spaced=p; spaced.min_gap=64;
    CheckpointArchive<Record> nearby(spaced);
    auto first=std::make_shared<int>(100);
    nearby.remember({100,first},CoveragePurpose::Root);
    nearby.remember({104,std::make_shared<int>(104)},CoveragePurpose::Periodic);
    nearby.remember({107,std::make_shared<int>(107)},CoveragePurpose::Anchor);
    nearby.remember({118,std::make_shared<int>(118)},CoveragePurpose::Endpoint);
    require(nearby.size()==2 && nearby.select(107,valid)->row==100,
        "close history points collapse before budgets fill, latest endpoint survives");
    require(nearby.nearby(117,valid)->row==100 && !nearby.nearby(100,valid),
        "pre-capture gate requires an earlier state, never same-frontier aliasing");
    require(!nearby.nearby(117,[](const Record&) {return false;}),"invalid native state cannot suppress capture");
    nearby.remember({190,std::make_shared<int>(190)},CoveragePurpose::Endpoint);
    require(nearby.size()==2 && nearby.select(118,valid)->row==100,
        "old endpoint obeys ordinary spacing after endpoint advances");
    nearby.remember({140,std::make_shared<int>(140)},CoveragePurpose::Periodic);
    nearby.remember({180,std::make_shared<int>(180)},CoveragePurpose::Periodic);
    require(nearby.select(180,valid)->row==180,"spacing uses retained representative, not skipped neighbor");
    require(!nearby.nearby(164,valid),"exact gap boundary allows a new coverage point");
    auto new_state=std::make_shared<int>(-100);
    nearby.remember({100,new_state},CoveragePurpose::Root);
    require(nearby.select(100,valid)->data==new_state && *first==100,
        "same-frontier replacement preserves the leased old payload");
    nearby.truncate(100);
    require(!nearby.select(99,valid) && nearby.nearby(130,valid)->data==new_state,
        "rewind removes future coverage while keeping valid earlier replay");
    CheckpointArchive<Record> unspaced(p);
    unspaced.remember({100,first},CoveragePurpose::Periodic);
    unspaced.remember({104,new_state},CoveragePurpose::Periodic);
    require(unspaced.size()==2 && !unspaced.nearby(110,valid),"zero gap disables both thinning and capture gate");
    std::cout << "checkpoint policy: coverage, tiny budgets, identity, leases, rewind and capture boundaries passed\n";
}
