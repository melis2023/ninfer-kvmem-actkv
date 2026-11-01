// A streamed sparse-attention layer (kv_stream.h) against the same layer kept resident: one
// sequence's K and V go through kv_cache_append into a resident identity-layout cache and into the
// streamed layer's host copy; before every narrow call's attention kv_stream_resolve makes the
// call's blocks resident in a small device pool, and sparse_softmax_attention over the pool must
// give the resident attention's output bit for bit, since only the addresses differ. After every
// call the map must be consistent (each slot names the block whose table entry names it) and every
// resident block's bytes must equal the host copy's in every plane: no slot is ever stale.
//
// The calls cover single decode steps against a pool of one query's minimum (every step evicts),
// verification widths up to 16 with rejected positions rewritten by the next call, wide prompt
// calls followed by kv_stream_evict, and single steps replayed from a captured graph whose
// position, inputs and selections change between replays.
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/kv_stream.h"
#include "ninfer/ops/sparse_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim = 256, kQHeads = 24, kKvHeads = 2, kTop = 512, kPage = 64, kBlock = 4;

const char* storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rk8v4";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    default: return "?";
    }
}

// The planes of one storage: data and optional scales of K and V.
struct PlaneSpec {
    std::int32_t leading = 0;
    DType dtype          = DType::U8;
};

std::vector<PlaneSpec> plane_specs(KvCacheStorage storage) {
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kDim);
    std::vector<PlaneSpec> out{{layout.key.data_leading_extent, layout.key.data_dtype},
                               {layout.value.data_leading_extent, layout.value.data_dtype}};
    if (layout.key.has_scale()) {
        out.push_back({layout.key.scale_leading_extent, layout.key.scale_dtype});
    }
    if (layout.value.has_scale()) {
        out.push_back({layout.value.scale_leading_extent, layout.value.scale_dtype});
    }
    return out;
}

std::size_t row_bytes(const PlaneSpec& plane) {
    return std::size_t(plane.leading) * dtype_size(plane.dtype);
}

// One streamed layer beside its resident twin.
struct Layer {
    KvCacheStorage storage;
    int pages = 0, slots = 0, blocks = 0;
    std::vector<PlaneSpec> specs;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> resident, pool;
    std::vector<std::unique_ptr<PinnedHostBuffer>> host;
    DeviceBuffer identity, map_memory;
    ops::KvStreamMap map;
    PagedKVLayerView resident_view{}, host_view{}, pool_view{};

    Layer(KvCacheStorage s, int length, int pool_slots)
        : storage(s), pages((length + kPage - 1) / kPage), slots(pool_slots),
          blocks(pages * (kPage / kBlock)), specs(plane_specs(s)) {
        for (const PlaneSpec& plane : specs) {
            const std::size_t page_bytes = row_bytes(plane) * kPage * kKvHeads;
            resident.push_back(std::make_unique<GuardedDeviceBuffer>(page_bytes * pages));
            resident.back()->fill(0);
            host.push_back(std::make_unique<PinnedHostBuffer>(page_bytes * pages));
            std::memset(host.back()->data(), 0, page_bytes * pages);
            pool.push_back(
                std::make_unique<GuardedDeviceBuffer>(row_bytes(plane) * kBlock * kKvHeads * slots));
            pool.back()->fill(0);
        }
        std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
        std::iota(table.begin(), table.end(), 0);
        identity = to_device(table);
        map_memory = DeviceBuffer(ops::kv_stream_map_bytes(blocks, slots));
        map        = ops::kv_stream_map(map_memory.p, blocks, slots);
        ops::kv_stream_reset(map, nullptr);
        const auto view = [&](auto data_of, std::int32_t page_tokens, std::int32_t count,
                              Tensor table_tensor) {
            PagedKVLayerView out{};
            Tensor* targets[4] = {&out.k_pages, &out.v_pages, &out.k_scale_pages,
                                  &out.v_scale_pages};
            const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kDim);
            std::size_t next                  = 0;
            for (int i = 0; i < 4; ++i) {
                const bool present = i < 2 || (i == 2 && layout.key.has_scale()) ||
                                     (i == 3 && layout.value.has_scale());
                if (!present) { continue; }
                *targets[i] = Tensor(data_of(next), specs[next].dtype,
                                     {specs[next].leading, page_tokens, kKvHeads, count});
                ++next;
            }
            out.block_table  = table_tensor;
            out.head_dim     = kDim;
            out.num_kv_heads = kKvHeads;
            out.storage      = storage;
            return out;
        };
        const Tensor identity_table(identity.p, DType::I32, {pages});
        resident_view = view([&](std::size_t i) { return resident[i]->data(); }, kPage, pages,
                             identity_table);
        host_view     = view([&](std::size_t i) { return host[i]->data(); }, kPage, pages,
                             identity_table);
        pool_view     = view([&](std::size_t i) { return pool[i]->data(); }, kBlock, slots,
                             Tensor(map.table, DType::I32, {blocks}));
    }

    // Every resident block's bytes against the host copy, and the map's consistency.
    int check(const std::string& label, cudaStream_t stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto table      = from_device<std::int32_t>(map.table, std::size_t(blocks));
        const auto slot_block = from_device<std::int32_t>(map.slot_block, std::size_t(slots));
        std::vector<std::vector<std::uint8_t>> slots_bytes;
        for (std::size_t i = 0; i < specs.size(); ++i) {
            slots_bytes.push_back(from_device<std::uint8_t>(pool[i]->data(), pool[i]->bytes()));
        }
        for (int b = 0; b < blocks; ++b) {
            const int slot = table[std::size_t(b)];
            if (slot == -1) { continue; }
            if (slot < 0 || slot >= slots || slot_block[std::size_t(slot)] != b) {
                std::cerr << label << ": block " << b << " maps to slot " << slot
                          << " which does not name it\n";
                return 1;
            }
            for (std::size_t i = 0; i < specs.size(); ++i) {
                const std::size_t row = row_bytes(specs[i]);
                const auto* h         = static_cast<const std::uint8_t*>(host[i]->data());
                for (int head = 0; head < kKvHeads; ++head) {
                    const std::size_t source =
                        ((std::size_t(b / 16) * kKvHeads + head) * kPage + (b % 16) * kBlock) * row;
                    const std::size_t target = (std::size_t(slot) * kKvHeads + head) * kBlock * row;
                    if (std::memcmp(h + source, slots_bytes[i].data() + target, kBlock * row) != 0) {
                        std::cerr << label << ": resident block " << b << " (slot " << slot
                                  << ", plane " << i << ") differs from the host copy\n";
                        return 1;
                    }
                }
            }
        }
        for (int slot = 0; slot < slots; ++slot) {
            const int b = slot_block[std::size_t(slot)];
            if (b >= 0 && table[std::size_t(b)] != slot) {
                std::cerr << label << ": slot " << slot << " names block " << b
                          << " whose table entry is " << table[std::size_t(b)] << '\n';
                return 1;
            }
        }
        int failures = 0;
        for (auto& buffer : pool) failures += buffer->verify_guards(label);
        for (auto& buffer : resident) failures += buffer->verify_guards(label);
        return failures;
    }
};

// Device inputs of one call of up to 16 (narrow) or `wide` tokens at fixed addresses.
struct Inputs {
    DeviceBuffer k, v, positions, first, selected, counts, q;
    explicit Inputs(int capacity)
        : k(std::size_t(capacity) * kKvHeads * kDim * 2), v(k.bytes),
          positions(std::size_t(capacity) * 4), first(4),
          selected(std::size_t(kTop) * ops::kKvStreamMaxTokens * 4),
          counts(std::size_t(ops::kKvStreamMaxTokens) * 4),
          q(std::size_t(ops::kKvStreamMaxTokens) * kQHeads * kDim * 2) {}
};

class Sequence {
public:
    Sequence(Layer& layer, std::uint32_t seed, int wide)
        : layer_(layer), random_(seed), in_(std::max(wide, ops::kKvStreamMaxTokens)),
          out_stream_(std::size_t(ops::kKvStreamMaxTokens) * kQHeads * kDim * 2),
          out_resident_(out_stream_.bytes),
          workspace_(ops::sparse_softmax_attention_workspace_bytes(ops::kKvStreamMaxTokens)) {}

    int position() const { return position_; }
    void rewind(int to) { position_ = to; }

    // Appends `t` random positions at the current position to both caches.
    void append(int t, cudaStream_t stream) {
        std::vector<float> k(std::size_t(t) * kKvHeads * kDim), v(k.size());
        fill_uniform(k, std::uint32_t(random_()), -1.0f, 1.0f);
        fill_uniform(v, std::uint32_t(random_()), -2.0f, 2.0f);
        std::vector<std::uint16_t> kb(k.size()), vb(v.size());
        for (std::size_t i = 0; i < k.size(); ++i) {
            kb[i] = f32_to_bf16(k[i]);
            vb[i] = f32_to_bf16(v[i]);
        }
        std::vector<std::int32_t> positions(static_cast<std::size_t>(t));
        std::iota(positions.begin(), positions.end(), position_);
        CUDA_CHECK(cudaMemcpyAsync(in_.k.p, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(in_.v.p, vb.data(), vb.size() * 2, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(in_.positions.p, positions.data(), positions.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(in_.first.p, &position_, 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void append_kernels(int t, const PagedKVLayerView& cache, cudaStream_t stream) {
        const Tensor k(in_.k.p, DType::BF16, {kDim, kKvHeads, t});
        const Tensor v(in_.v.p, DType::BF16, {kDim, kKvHeads, t});
        const Tensor positions(in_.positions.p, DType::I32, {t});
        ops::kv_cache_append(k, v, positions, cache, stream);
    }

    // Random selections and queries for `t` queries at the current position: recent blocks and
    // far ones, ascending and distinct, at most 512.
    void select(int t, cudaStream_t stream) {
        std::vector<std::int32_t> selected(std::size_t(kTop) * t, -1);
        std::vector<std::int32_t> counts(static_cast<std::size_t>(t));
        for (int q = 0; q < t; ++q) {
            const int complete = (position_ + q + 1) / kBlock;
            std::set<int> chosen;
            if (complete <= kTop) {
                for (int b = 0; b < complete; ++b) chosen.insert(b);
            } else {
                const int recent = std::min(complete, 96);
                for (int b = complete - recent; b < complete; ++b) chosen.insert(b);
                std::uniform_int_distribution<int> any(0, complete - 1);
                while (int(chosen.size()) < kTop) chosen.insert(any(random_));
            }
            counts[std::size_t(q)] = int(chosen.size());
            std::copy(chosen.begin(), chosen.end(), selected.begin() + std::ptrdiff_t(q) * kTop);
        }
        std::vector<float> q(std::size_t(t) * kQHeads * kDim);
        fill_uniform(q, std::uint32_t(random_()), -1.0f, 1.0f);
        std::vector<std::uint16_t> qb(q.size());
        for (std::size_t i = 0; i < q.size(); ++i) qb[i] = f32_to_bf16(q[i]);
        CUDA_CHECK(cudaMemcpyAsync(in_.selected.p, selected.data(), selected.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(in_.counts.p, counts.data(), counts.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(in_.q.p, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // The streamed call: append into the host copy, resolve, attend over the pool.
    void streamed(int t, cudaStream_t stream) {
        append_kernels(t, layer_.host_view, stream);
        const Tensor first(in_.first.p, DType::I32, {1});
        const Tensor selected(in_.selected.p, DType::I32, {kTop, t});
        const Tensor counts(in_.counts.p, DType::I32, {t});
        ops::kv_stream_resolve(layer_.map, first, selected, counts, layer_.host_view,
                               layer_.pool_view, stream);
        attend(t, layer_.pool_view, out_stream_.p, stream);
    }

    void attend(int t, const PagedKVLayerView& cache, void* out, cudaStream_t stream) {
        const Tensor q(in_.q.p, DType::BF16, {kDim, kQHeads, t});
        const Tensor first(in_.first.p, DType::I32, {1});
        const Tensor selected(in_.selected.p, DType::I32, {kTop, t});
        const Tensor counts(in_.counts.p, DType::I32, {t});
        Tensor result(out, DType::BF16, {kDim, kQHeads, t});
        ops::sparse_softmax_attention(q, first, selected, counts, cache, 1.0f / 16.0f, workspace_,
                                      result, stream);
    }

    // The resident twin of the call, then the comparison.
    int compare(int t, const std::string& label, cudaStream_t stream) {
        append_kernels(t, layer_.resident_view, stream);
        attend(t, layer_.resident_view, out_resident_.p, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t n = std::size_t(t) * kQHeads * kDim;
        return verify_exact(label.c_str(), from_device<std::uint16_t>(out_stream_.p, n),
                            from_device<std::uint16_t>(out_resident_.p, n));
    }

    // A wide call: both caches take its positions; the pool drops the blocks it wrote.
    void wide(int t, cudaStream_t stream) {
        append(t, stream);
        append_kernels(t, layer_.host_view, stream);
        append_kernels(t, layer_.resident_view, stream);
        ops::kv_stream_evict(layer_.map, position_ / kBlock, (position_ + t - 1) / kBlock + 1,
                             stream);
        position_ += t;
    }

    void advance(int t) { position_ += t; }

private:
    Layer& layer_;
    std::mt19937 random_;
    Inputs in_;
    DeviceBuffer out_stream_, out_resident_;
    WorkspaceArena workspace_;
    int position_ = 0;
};

struct Plan {
    const char* name;
    int length, slots, prompt, steps;
    std::vector<int> widths;
    bool rewrites = false, wide_calls = false, graph = false;
};

int run(KvCacheStorage storage, const Plan& plan, std::uint32_t seed) {
    Layer layer(storage, plan.length, plan.slots);
    Sequence sequence(layer, seed, 64);
    std::mt19937 random(seed ^ 0x5bd1e995u);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int failures = 0;
    const std::string base = std::string("kv_stream ") + storage_name(storage) + " " + plan.name;
    for (int done = 0; done < plan.prompt;) {
        const int t = std::min(64, plan.prompt - done);
        sequence.wide(t, stream);
        done += t;
    }
    failures += layer.check(base + " after the prompt", stream);

    cudaGraphExec_t graph = nullptr;
    int calls             = 0;
    for (int step = 0; step < plan.steps && failures == 0; ++step) {
        if (plan.wide_calls && step % 17 == 16) {
            sequence.wide(40, stream);
            failures += layer.check(base + " after a wide call", stream);
            continue;
        }
        const int t = plan.widths[std::size_t(random()) % plan.widths.size()];
        if (sequence.position() + t > plan.length) { break; }
        sequence.append(t, stream);
        sequence.select(t, stream);
        if (plan.graph) {
            if (graph == nullptr) {
                cudaGraph_t captured = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                sequence.streamed(t, stream);
                CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
                CUDA_CHECK(cudaGraphInstantiate(&graph, captured, 0));
                CUDA_CHECK(cudaGraphDestroy(captured));
            }
            CUDA_CHECK(cudaGraphLaunch(graph, stream));
        } else {
            sequence.streamed(t, stream);
        }
        ++calls;
        const std::string label = base + " step " + std::to_string(step) + " p=" +
                                  std::to_string(sequence.position()) + " T=" + std::to_string(t);
        failures += sequence.compare(t, label, stream);
        if (step % 7 == 0) { failures += layer.check(label, stream); }
        // A verification keeps a prefix of its positions; the next call rewrites the rest.
        const int kept = plan.rewrites && t > 1 ? 1 + int(random() % std::uint32_t(t)) : t;
        sequence.advance(kept);
    }
    if (graph != nullptr) { CUDA_CHECK(cudaGraphExecDestroy(graph)); }
    failures += layer.check(base + " at the end", stream);
    const ops::KvStreamCounters counters = ops::kv_stream_counters(layer.map, stream);
    if (counters.overflow || counters.calls != std::uint64_t(calls) || counters.copies == 0 ||
        counters.lookups < counters.copies) {
        std::cerr << base << ": counters calls=" << counters.calls << " (expected " << calls
                  << ") lookups=" << counters.lookups << " copies=" << counters.copies
                  << " overflow=" << counters.overflow << '\n';
        ++failures;
    }
    std::cout << base << ": " << calls << " calls, " << counters.lookups << " lookups, "
              << counters.copies << " copies\n";
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const std::vector<Plan> plans{
        // One query's minimum pool against a long context: nearly every step evicts.
        {"single", 40000, ops::kv_stream_minimum_slots(1), 30000, 160, {1}},
        // Verification widths with rejected positions rewritten.
        {"verify", 12000, ops::kv_stream_minimum_slots(4), 2600, 160, {1, 2, 3, 4}, true},
        // Every narrow width, wide calls between them.
        {"wide", 26000, ops::kv_stream_minimum_slots(16), 21000, 120, {1, 5, 9, 16}, true, true},
        // Single steps replayed from one captured graph.
        {"graph", 20000, ops::kv_stream_minimum_slots(1), 9000, 120, {1}, false, false, true},
    };
    int failures = 0;
    std::uint32_t seed = 7300;
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, KvCacheStorage::Nvfp4Group16}) {
        for (const Plan& plan : plans) { failures += run(storage, plan, seed++); }
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " kv_stream\n";
    return failures == 0 ? 0 : 1;
}
