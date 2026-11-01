#include "models/qwen3_5/program/storage/checkpoint_buffer_pool.h"
#include "models/qwen3_5/program/program_impl.h"
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
int main(int argc, char** argv) {
    using namespace ninfer::models::qwen3_5::detail;
    {
        ResidentPrefixIdentity source;
        source.append_generated(53000, 0);
        KvmemPrefixCheckpoint checkpoint;
        const auto empty = checkpoint.metadata_bytes();
        checkpoint.prefix_identity = source;
        checkpoint.prefix_identity.truncate(26);
        CHECK(checkpoint.prefix_identity.size() == 26);
        CHECK(checkpoint.prefix_identity.prefix_equals(source, 26));
        CHECK(checkpoint.metadata_bytes() >= empty + 53000*13);
        CHECK(checkpoint.metadata_bytes() == empty + checkpoint.prefix_identity.allocated_bytes());
        const auto before_pages = checkpoint.metadata_bytes();
        checkpoint.resident_pages.reserve(32);
        CHECK(checkpoint.metadata_bytes() == before_pages + checkpoint.resident_pages.capacity()*sizeof(uint64_t));
        checkpoint.resident_pages.push_back(0);
        CHECK(checkpoint.metadata_bytes() == before_pages + checkpoint.resident_pages.capacity()*sizeof(uint64_t));

        ninfer::models::qwen3_5::VisionItem image;
        image.token_spans = {{1, 2}};
        image.timestamps = {0.0, 1.0};
        std::vector<ninfer::models::qwen3_5::VisionItem> images{image};
        std::array<std::vector<int32_t>, 3> axes;
        for (auto& axis : axes) axis.resize(32);
        ResidentPrefixIdentity media;
        media.restore(std::vector<uint8_t>(32), std::move(axes), std::move(images), {4, 16});
        size_t owned = media.token_types().capacity() + media.vision_items().capacity()*sizeof(image) +
            media.rewrite_execution_frontiers().capacity()*sizeof(uint32_t);
        for (size_t i = 0; i < 3; ++i) owned += media.position_axis(i).capacity()*sizeof(int32_t);
        for (const auto& item : media.vision_items())
            owned += item.timestamps.capacity()*sizeof(double) + item.token_spans.capacity()*sizeof(item.token_spans[0]);
        CHECK(media.allocated_bytes() == owned);
        auto copy = media;
        copy.truncate(4);
        CHECK(copy.prefix_equals(media, 4));
        CHECK(copy.allocated_bytes() >= 32*13 + sizeof(image) + 2*sizeof(double) + sizeof(image.token_spans[0]) + 2*sizeof(uint32_t));
        checkpoint.prefix.resize(26);
        auto retained = std::make_shared<KvmemPrefixCheckpoint>(std::move(checkpoint));
        KvmemCheckpointRecord record{kvmem::CheckpointPurpose::None, retained};
        kvmem::CheckpointPolicy policy;
        policy.enabled = true;
        policy.bytes = record.checkpoint_info().bytes - 1;
        kvmem::CheckpointArchive<KvmemCheckpointRecord> short_budget(policy);
        CHECK(!short_budget.remember(record, kvmem::CoveragePurpose::Root));
        ++policy.bytes;
        kvmem::CheckpointArchive<KvmemCheckpointRecord> exact_budget(policy);
        CHECK(exact_budget.remember(record, kvmem::CoveragePurpose::Root));
        CHECK(exact_budget.bytes() == policy.bytes);
    }
    if (argc > 1 && std::strcmp(argv[1], "--metadata-only") == 0) return 0;
    auto pool = std::make_shared<KvmemCheckpointBufferPool>();
    auto first = pool->acquire(4096);
    auto other = pool->acquire(4096);
    CHECK(first->data() != other->data());
    std::memset(first->data(), 17, first->size());
    std::memset(other->data(), 23, other->size());
    auto alias = first;
    const auto address = first->data();
    first.reset();
    CHECK(pool->idle_bytes() == 0); // An aliased checkpoint is still live.
    alias.reset();
    CHECK(pool->idle_bytes() == 4096);
    first = pool->acquire(4096);
    CHECK(first->data() == address && pool->idle_bytes() == 0);
    std::memset(first->data(), 29, first->size());
    for (std::size_t i = 0; i < other->size(); ++i)
        CHECK(static_cast<unsigned char*>(other->data())[i] == 23);
    first.reset(); other.reset();
    CHECK(pool->idle_bytes() == 8192);
    auto extra = pool->acquire(8192);
    extra.reset();
    CHECK(pool->idle_bytes() == 8192); // A third release cannot grow the cache.
    auto last = pool->acquire(4096);
    pool.reset(); // Outstanding leases own the recycler's lifetime.
    std::memset(last->data(), 31, last->size());
    last.reset();
    auto bounded = std::make_shared<KvmemCheckpointBufferPool>();
    bounded->configure(8192,1);
    auto live = bounded->acquire(4096), queued = bounded->acquire(4096);
    CHECK(bounded->allocated_bytes()==8192);
    bool rejected=false;
    try { (void)bounded->acquire(1); } catch(const std::exception&) { rejected=true; }
    CHECK(rejected);
    live.reset(); queued.reset();
    CHECK(bounded->allocated_bytes()==4096 && bounded->idle_bytes()==4096);
    bounded->trim();
    CHECK(bounded->allocated_bytes()==0);
    return 0;
}
