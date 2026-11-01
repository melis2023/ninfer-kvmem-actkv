#include "memory_test_support.hpp"
#include "kvmem/host_kv_storage.hpp"
#include <array>
#include <cstring>
#include <thread>

int main() {
    HostKvStorage storage(16);
    std::array<uint8_t, 8> bytes{{1, 2, 3, 4, 5, 6, 7, 8}}, copy{};
    {
        auto reserved = storage.reserve(16);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { storage.reserve(1); });
        auto first = reserved.allocate(8, bytes.data());
        CHECK(reserved.remaining() == 8 && storage.charged_bytes() == 16);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { reserved.allocate(9); });
        CHECK(reserved.remaining() == 8);
        auto second = reserved.allocate(8);
        second.read(0, copy.data(), copy.size());
        CHECK((copy == std::array<uint8_t, 8>{}));
        {
            auto lease = first.lease();
            rejects(MemoryErrorCode::Pending, [&] { first.write(0, bytes.data(), 1); });
            auto moved = std::move(first);
            CHECK(first.empty());
            moved.reset();
            CHECK(storage.charged_bytes() == 16);
            CHECK(lease.size() == 8 && std::memcmp(lease.data(), bytes.data(), 8) == 0);
        }
        CHECK(storage.charged_bytes() == 8);
        second.write(2, bytes.data(), 4);
        second.read(0, copy.data(), 8);
        CHECK((copy == std::array<uint8_t, 8>{{0, 0, 1, 2, 3, 4, 0, 0}}));
        rejects(MemoryErrorCode::InvalidPlan, [&] { second.read(SIZE_MAX, copy.data(), 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { second.write(7, bytes.data(), 2); });
    }
    CHECK(storage.charged_bytes() == 0);
    {
        HostKvStorage branches(24);
        auto parent=branches.allocate(8,bytes.data());
        rejects(MemoryErrorCode::Pending,[&] { parent.fork(); });
        parent.seal();
        parent.write(0,bytes.data(),1);
        rejects(MemoryErrorCode::Pending,[&] { parent.fork(); });
        parent.seal();
        auto child=parent.fork();
        CHECK(parent.storage_identity()==child.storage_identity());
        CHECK(branches.charged_bytes()==8);
        uint8_t value=99;
        child.write(0,&value,1);
        CHECK(branches.charged_bytes()==16);
        CHECK(parent.data()[0]==1 && child.data()[0]==99);
        value=77; parent.write(1,&value,1);
        CHECK(parent.data()[1]==77 && child.data()[1]==2);
        parent.reset();
        CHECK(child.data()[0]==99 && branches.charged_bytes()==8);
    }
    {
        HostKvStorage tight(8);
        auto parent=tight.allocate(8,bytes.data()); parent.seal();
        auto child=parent.fork();
        rejects(MemoryErrorCode::BudgetExceeded,[&] { child.write(0,bytes.data(),1); });
        CHECK(parent.storage_identity()==child.storage_identity() && tight.charged_bytes()==8);
        CHECK(std::memcmp(parent.data(),bytes.data(),8)==0);
    }
    {
        // Once the other owner is gone, writing the survivor must not need a
        // second page: the entire hard RAM budget is already this one page.
        HostKvStorage tight(8);
        auto parent=tight.allocate(8,bytes.data()); parent.seal();
        auto survivor=parent.fork(); parent.reset();
        CHECK(survivor.mutable_resident());
        uint8_t changed=42;
        try { survivor.write(0,&changed,1); }
        catch(const MemoryError& error) {
            std::fprintf(stderr,"surviving owner cannot reuse its page: %s\n",error.what());
            return 1;
        }
        CHECK(survivor.data()[0]==42 && tight.charged_bytes()==8);
    }
    {
        HostKvStorage tight(8);
        auto parent=tight.allocate(8,bytes.data()); parent.seal();
        auto survivor=parent.fork(); auto reader=parent.source(); parent.reset();
        uint8_t changed=42;
        // Read sources count too: do not mutate data held by an in-flight read
        // just because the other branch's record was deleted.
        rejects(MemoryErrorCode::BudgetExceeded,[&] { survivor.write(0,&changed,1); });
        reader.read(0,copy.data(),copy.size()); CHECK(copy==bytes);
        auto lease=survivor.lease(); reader={};
        rejects(MemoryErrorCode::BudgetExceeded,[&] { survivor.write(0,&changed,1); });
        lease.read(0,copy.data(),copy.size()); CHECK(copy==bytes);
        lease={}; survivor.write(0,&changed,1);
        CHECK(survivor.data()[0]==42 && tight.charged_bytes()==8);
    }
    {
        auto first = storage.reserve(8), second = storage.reserve(8);
        first = std::move(second);
        CHECK(storage.charged_bytes() == 8 && second.remaining() == 0);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { second.allocate(1); });
    }
    CHECK(storage.charged_bytes() == 0);
    // Separate engine owners share one hard bound, while record contents are private.
    HostKvStorage shared(8 * 4);
    std::vector<std::thread> workers;
    for (int i = 0; i < 4; ++i) workers.emplace_back([&] {
        for (int j = 0; j < 1000; ++j) {
            auto record = shared.allocate(8, bytes.data());
            CHECK(record.lease().size() == 8);
        }
    });
    for (auto & worker : workers) worker.join();
    CHECK(shared.charged_bytes() == 0);
    // Leases also outlive the storage facade, as required by cancellation cleanup.
    auto survivor = [] {
        HostKvStorage temporary(8);
        auto record = temporary.allocate(8);
        return record.lease();
    }();
    CHECK(survivor.size() == 8 && survivor.data()[0] == 0);
    std::puts("host KV storage: byte preservation, reservations, leases and shared limits passed");
}
