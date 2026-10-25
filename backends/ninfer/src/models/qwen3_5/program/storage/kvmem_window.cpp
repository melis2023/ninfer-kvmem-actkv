#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/context_work.h"

#include "core/paged_kv_cache.h"
#include "kvmem/prefix_checkpoint.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <utility>

namespace ninfer::models::qwen3_5::detail {
namespace {
constexpr std::uint32_t page_tokens = static_cast<std::uint32_t>(kPagedKVPageSize);
std::atomic<std::uint64_t> next_window_session{1};

std::uint32_t valid_columns(std::uint32_t logical_page, std::uint32_t frontier) {
    return std::min(page_tokens, frontier - logical_page * page_tokens);
}
} // namespace

// This bridge borrows Program's existing native pool, address space, stream and
// reservation. It owns no device allocator. Its lifetime is one safe boundary;
// durable history and versions travel with SequenceState.
class WindowMemoryBackend final : public kvmem::MemoryBackend {
public:
    WindowMemoryBackend(ProgramImpl& program, SequenceState& sequence, std::uint32_t end, bool resident_native = false)
        : program(program), sequence(sequence), end(end), resident_native(resident_native),
          layout(plan_host_kv_page_layout(program.text_kv_pages->physical_pool().geometry())) {
        if (program.backend_kv_pages) {
            // kvmem_backend_frontier(), not mtp_kv_valid and not backend_kv_valid(): a masked
            // draft backend (DFlash or DFlash2) keeps its conditioning frontier in
            // dflash_context_frontier, mtp_kv_valid is never written for it, and
            // backend_kv_valid() narrows DFlash2 to 0 on purpose for the standard path.
            if (program.kvmem_backend_frontier(sequence) != sequence.text_kv_valid) {
                throw std::logic_error("KVMem dual-pool transfer requires a committed safe boundary");
            }
            mtp_layout = plan_host_kv_page_layout(program.backend_kv_pages->physical_pool().geometry());
        }
        page_bytes = layout.page_stride + (mtp_layout ? mtp_layout->page_stride : 0);
        auto& stamp = sequence.window.stamp;
        if (!stamp.session.id) {
            stamp = {true, {next_window_session.fetch_add(1), 1}, 1, 0, 0, 1};
        }
        if (!stamp.valid) { throw std::logic_error("KVMem sequence requires fresh prefill"); }
        if (stamp.frontier != sequence.text_kv_valid) {
            stamp.frontier = sequence.text_kv_valid;
            ++stamp.history_revision;
        }
        const auto count = kv_pages_for_tokens(sequence.text_kv_valid);
        // Append preparation may already own buffers for the next GPU chunk.
        // Keep them until publication or explicit history truncation.
        if (sequence.window.archive.size() < count) sequence.window.archive.resize(count);
        if (sequence.window.host_versions.size() < count) sequence.window.host_versions.resize(count);
    }

    kvmem::MemoryDescriptor descriptor() const override {
        kvmem::MemoryDescriptor d;
        std::string model_id;
        constexpr char hex[] = "0123456789abcdef";
        for (const auto value : program.parameters.model.info().artifact_id) {
            const auto byte = std::to_integer<unsigned char>(value);
            model_id += hex[byte >> 4]; model_id += hex[byte & 15];
        }
        d.identity = {"ninfer", program.backend_kv_pages ? "kvmem-main-mtp-pages-v2" : "kvmem-native-pages-v1",
            model_id, "qwen3.5-kvmem-state-image-v3"};
        const char* layout_name = nullptr;
        switch (program.kv_storage) {
        case KvCacheStorage::BFloat16: layout_name = "ninfer.bf16.page.v1"; break;
        case KvCacheStorage::Int8Group64: layout_name = "ninfer.int8-g64.page.v1"; break;
        case KvCacheStorage::Fp8E4M3Row256: layout_name = "ninfer.fp8-e4m3-row256.page.v1"; break;
        case KvCacheStorage::Nvfp4Group16: layout_name = "ninfer.nvfp4-g16.page.v1"; break;
        case KvCacheStorage::Fp8KeyNvfp4Value: layout_name = "ninfer.k8v4-fp8-nvfp4.page.v1"; break;
        case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
            layout_name = "ninfer.rk8v4-k64-v32.page.v1";
            break;
        case KvCacheStorage::RotatedLloyd4KeyInt4Value:
            layout_name = "ninfer.rk4v4-lloyd-k64-v32.page.v1";
            break;
        case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
            layout_name = "ninfer.rk4v4-e8-k64-v32.page.v1";
            break;
        case KvCacheStorage::RotatedE8RootKeyInt4Value:
            layout_name = "ninfer.rk2v4-e8-root-k64-v32.page.v1";
            break;
        default: throw std::logic_error("Unsupported KVMem native KV storage");
        }
        d.layout = {layout_name, 1, page_tokens, page_bytes, 256, {}};
        const auto& config = program.parameters.model.config().text;
        const std::uint32_t planes = program.kv_storage == KvCacheStorage::BFloat16 ? 2 : 4;
        const kvmem::PayloadPlaneKind kinds[] = {kvmem::PayloadPlaneKind::Key,
            kvmem::PayloadPlaneKind::Value, kvmem::PayloadPlaneKind::KeyScale,
            kvmem::PayloadPlaneKind::ValueScale};
        std::size_t ordinal = 0;
        for (std::uint32_t layer = 0; layer < config.layer_types.size(); ++layer) {
            if (config.layer_types[layer] != MixerKind::FullAttention) { continue; }
            for (std::uint32_t kind = 0; kind < planes; ++kind) {
                const auto& plane = layout.planes.at(ordinal++);
                d.layout.planes.push_back({layer, kinds[kind], plane.offset,
                    plane.page_payload_bytes, 1, 0});
            }
        }
        if (ordinal != layout.planes.size()) { throw std::logic_error("KVMem plane inventory mismatch"); }
        if (mtp_layout) {
            for (std::uint32_t kind = 0; kind < planes; ++kind) {
                const auto& plane = mtp_layout->planes.at(kind);
                d.layout.planes.push_back({static_cast<std::uint32_t>(config.layer_types.size()),
                    kinds[kind], layout.page_stride + plane.offset, plane.page_payload_bytes, 1, 0});
            }
        }
        d.capabilities = {page_tokens, program.capacity, true, true, true, false, false, false};
        return d;
    }

    kvmem::MemorySnapshot snapshot() const override {
        kvmem::MemorySnapshot s;
        s.stamp = stamp();
        const auto count = kv_pages_for_tokens(sequence.text_kv_valid);
        for (std::uint32_t id = 0; id < count; ++id) {
            const auto columns = valid_columns(id, sequence.text_kv_valid);
            s.blocks.push_back({{s.stamp.session, id, id * page_tokens, columns, columns, 0},
                sequence.window.host_versions.at(id)});
        }
        for (auto id : sequence.window.pages) {
            if (id < count) { s.resident.push_back(id); }
        }
        return s;
    }
    kvmem::MemoryStamp stamp() const noexcept override { return sequence.window.stamp; }
    std::unique_ptr<kvmem::MemoryTransfer> prepare(const kvmem::WorkingSetPlan&) override;
    void invalidate() noexcept override { sequence.window.stamp.valid = false; }

    ProgramImpl& program;
    SequenceState& sequence;
    std::uint32_t end;
    bool resident_native = false; // materialization owns RAM leases through restore
    HostKVPageLayout layout;
    std::optional<HostKVPageLayout> mtp_layout;
    std::size_t page_bytes = 0;
};

class WindowMemoryTransfer final : public kvmem::MemoryTransfer {
public:
    WindowMemoryTransfer(WindowMemoryBackend& backend, const kvmem::WorkingSetPlan& plan,
                         bool verify_restore = true)
        : backend(backend), plan(plan), asynchronous(backend.program.memory_io && verify_restore && !backend.resident_native) {
        auto& seq = backend.sequence;
        auto& p = backend.program;
        const auto count = kv_pages_for_tokens(seq.text_kv_valid);
        pending.resize(count);
        // Rebuilding a native compact address space also moves retained pages.
        // Reserve their host records up front and include them in physical H and logical history admission.
        // Complete host copies are reused; only the changed tail is copied again.
        std::uint64_t bytes = seq.window.payload_bytes;
        for (std::uint32_t slot = 0; slot < seq.window.pages.size(); ++slot) {
            const auto id = seq.window.pages[slot];
            if (id >= count) { continue; }
            if (!seq.window.archive[id]) {
                if (bytes > p.memory_retained_budget ||
                    backend.page_bytes > p.memory_retained_budget - bytes) {
                    throw kvmem::MemoryError(kvmem::MemoryErrorCode::BudgetExceeded,
                        "KVMem retained payload budget exhausted before transfer");
                }
                bytes += backend.page_bytes;
                if (!asynchronous) pending[id] = p.memory_payload_storage.allocate(backend.page_bytes);
            }
            if (seq.window.host_versions[id] != valid_columns(id, seq.text_kv_valid)) {
                spill_ids.push_back(id);
                spill_slots.push_back(slot);
                sources.push_back(p.text_kv_addresses->physical_page(seq.kv->text, slot));
                source_epochs.push_back(p.text_kv_addresses->content_epoch(seq.kv->text, slot));
                if (backend.mtp_layout) {
                    mtp_sources.push_back(p.backend_kv_addresses->physical_page(*seq.kv->backend, slot));
                    mtp_source_epochs.push_back(p.backend_kv_addresses->content_epoch(*seq.kv->backend, slot));
                }
            }
        }
        prepared_payload_bytes = bytes;
        for (const auto& block : plan.next.blocks) {
            const auto id = static_cast<std::uint32_t>(block.block.id);
            if (!record(id) && (!asynchronous || std::find(spill_ids.begin(), spill_ids.end(), id) == spill_ids.end()))
                throw std::logic_error("KVMem selected page has no native payload");
            selected.push_back(id);
        }
        // Stable page order is what makes a slot-by-slot diff meaningful: the same logical
        // page keeps the same physical slot across reselections, so the next restore can
        // skip it instead of copying it back from the host.
        std::sort(selected.begin(), selected.end());
        if (seq.text_kv_valid % page_tokens &&
            (selected.empty() || selected.back() != seq.text_kv_valid / page_tokens)) {
            throw std::logic_error("KVMem partial append page must remain the last selected page");
        }
        const auto rows = std::max(spill_ids.size(), selected.size());
        if (rows > p.kvmem_window_tokens / page_tokens) {
            throw std::logic_error("KVMem staging exceeds its physical window");
        }
        const auto staging_bytes = rows * backend.page_bytes;
        if (staging_bytes && (!p.memory_kv_staging || p.memory_kv_staging->size() < staging_bytes)) {
            p.memory_kv_staging = std::make_unique<PinnedHostBuffer>(staging_bytes);
        }
        // Only this bounded buffer participates in CUDA host transfers. Archive
        // records are byte-preserving pageable RAM and are never CUDA-registered.
        auto* staging = p.memory_kv_staging
            ? static_cast<std::byte*>(p.memory_kv_staging->data()) : nullptr;
        for (std::size_t i = 0; i < spill_ids.size(); ++i)
            spill_records.push_back(staging + i * backend.page_bytes);
        for (std::size_t i = 0; i < selected.size(); ++i)
            restore_records.push_back(staging + i * backend.page_bytes);
        if (verify_restore && p.kvmem_options.verify_transfers) {
            const auto bytes_state = p.state_images->host_layout().image_bytes;
            before = std::make_unique<PinnedHostBuffer>(bytes_state);
            after = std::make_unique<PinnedHostBuffer>(bytes_state);
            checked = std::make_unique<PinnedHostBuffer>(selected.size() * backend.page_bytes);
            std::memset(before->data(), 0, bytes_state);
            std::memset(after->data(), 0, bytes_state);
            std::memset(checked->data(), 0, selected.size() * backend.page_bytes);
        }
    }

    void start_spill() override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        const auto spill_t0 = std::chrono::steady_clock::now();
        bind_staging();
        if (!spill_ids.empty()) {
            // Only a partial tail page has columns outside the valid frontier;
            // full pages are overwritten entirely by copy_to_host_records.
            for (std::size_t i = 0; i < spill_ids.size(); ++i) {
                if (valid_columns(spill_ids[i], seq.text_kv_valid) != page_tokens) {
                    std::memset(spill_records[i], 0, backend.page_bytes);
                }
            }
        }
        const auto spill_t_memset = std::chrono::steady_clock::now();
        if (before) {
            std::array<std::byte*, 1> rows{static_cast<std::byte*>(before->data())};
            p.state_images->copy_to_host_segments(p.state_store->physical_slot(seq.state.write),
                rows, p.state_images->host_layout().image_bytes, p.compute_streams);
        }
        p.text_kv_pages->physical_pool().copy_to_host_records(sources, spill_records,
            {}, backend.layout, p.compute_streams);
        if (backend.mtp_layout) {
            std::vector<std::byte*> mtp_records;
            for (auto* record : spill_records) mtp_records.push_back(record + backend.layout.page_stride);
            p.backend_kv_pages->physical_pool().copy_to_host_records(mtp_sources, mtp_records,
                {}, *backend.mtp_layout, p.compute_streams);
        }
        const auto spill_t_copy = std::chrono::steady_clock::now();
        p.device.synchronize();
        const auto spill_t_sync = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < spill_ids.size(); ++i) {
            if (source_epochs[i] != p.text_kv_addresses->content_epoch(seq.kv->text, spill_slots[i])) {
                throw std::logic_error("KVMem source epoch changed during spill");
            }
            if (backend.mtp_layout && mtp_source_epochs[i] !=
                p.backend_kv_addresses->content_epoch(*seq.kv->backend, spill_slots[i])) {
                throw std::logic_error("KVMem MTP source epoch changed during spill");
            }
            if (!asynchronous) {
                auto* destination = record(spill_ids[i]);
                destination->write(0, spill_records[i], backend.page_bytes);
                if (valid_columns(spill_ids[i], seq.text_kv_valid) == page_tokens) destination->seal();
            }
        }
        const auto spill_t_write = std::chrono::steady_clock::now();
        if (asynchronous) {
            storage_work = std::make_shared<StorageWork>(p.memory_payload_storage, backend.page_bytes);
            auto& work = *storage_work;
            work.records.resize(pending.size());
            work.spill_ids = spill_ids;
            work.selected = selected;
            work.sources.resize(selected.size());
            auto scratch = p.memory_transfer_storage.reserve((spill_ids.size() + selected.size()) * backend.page_bytes);
            work.captured = scratch.allocate(spill_ids.size() * backend.page_bytes);
            work.restored = scratch.allocate(selected.size() * backend.page_bytes);
            for (std::size_t i = 0; i < spill_ids.size(); ++i) {
                std::memcpy(work.captured.writable_data() + i * backend.page_bytes, spill_records[i], backend.page_bytes);
                // Old partial pages remain recoverable until publication, but
                // their storage tier may change while the new copy is prepared.
                if (auto* old = record(spill_ids[i])) old->seal();
            }
            for (std::size_t i = 0; i < selected.size(); ++i) {
                if (std::find(spill_ids.begin(), spill_ids.end(), selected[i]) != spill_ids.end()) continue;
                auto* source = record(selected[i]);
                source->seal();
                work.sources[i] = source->source();
            }
            ticket = p.memory_io->submit(work.captured.size() + work.restored.size(), [work = storage_work] {
                for (std::size_t i = 0; i < work->spill_ids.size(); ++i) {
                    auto& destination = work->records[work->spill_ids[i]];
                    destination = work->storage.allocate(work->stride, work->captured.data() + i * work->stride);
                    destination.seal();
                }
                for (std::size_t i = 0; i < work->selected.size(); ++i) {
                    auto* destination = work->restored.writable_data() + i * work->stride;
                    if (const auto& record = work->records[work->selected[i]]; record)
                        record.read(0, destination, work->stride);
                    else work->sources[i].read(0, destination, work->stride);
                }
            });
            seq.window.pending_window_io = ticket;
            return;
        }
        publish_host();
        {
            const auto us = [](auto a, auto b) {
                return static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
            };
            std::fprintf(stderr,
                "KVMEM_SPILL pages=%zu page_bytes=%zu async=%d memset_us=%lld copy_us=%lld sync_us=%lld "
                "write_us=%lld total_us=%lld\n",
                spill_ids.size(), backend.page_bytes, int(asynchronous),
                us(spill_t0, spill_t_memset), us(spill_t_memset, spill_t_copy),
                us(spill_t_copy, spill_t_sync), us(spill_t_sync, spill_t_write),
                us(spill_t0, spill_t_write));
        }
    }

    kvmem::TransferPoll poll() override {
        if (asynchronous && !host_published) {
            if (!ticket.ready()) return {kvmem::TransferStatus::Pending, {}};
            try {
                ticket.wait();
                pending = std::move(storage_work->records);
                publish_host();
            } catch (const std::exception& error) {
                return {kvmem::TransferStatus::Failed, error.what()};
            }
        }
        return {kvmem::TransferStatus::Complete, {}};
    }

    void publish_host() {
        auto& seq = backend.sequence;
        for (std::size_t id = 0; id < pending.size(); ++id) {
            if (pending[id]) { seq.window.archive[id] = std::move(pending[id]); }
        }
        for (auto id : spill_ids) { seq.window.host_versions[id] = valid_columns(id, seq.text_kv_valid); }
        seq.window.payload_bytes = prepared_payload_bytes;
        seq.window.spilled_bytes += spill_ids.size() * backend.page_bytes;
        host_published = true;
    }

    void start_restore() override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        const auto restore_t0 = std::chrono::steady_clock::now();
        bind_staging();
        // Decide reuse before reading the host records: an unchanged slot already has the
        // right bytes in its device replica, so it does not need to cross the staging buffer.
        const auto& previous = seq.window.pages;
        // The window may hold more pages than the selection budget: the reserve tail is
        // appended to the address space but not kept by the selector. Compare the prefix
        // only, so an unchanged selection still counts as reuse.
        const bool enough = previous.size() >= selected.size();
        std::vector<std::uint32_t> changed;
        if (enough) {
            changed.reserve(selected.size());
            for (std::uint32_t i = 0; i < selected.size(); ++i) {
                if (previous[i] != selected[i]) changed.push_back(i);
            }
        }
        const std::size_t selection_pages =
            static_cast<std::size_t>(backend.program.kvmem_options.selected_tokens / page_tokens);
        const bool incremental = !checked && enough &&
            changed.size() * 2 <= selected.size() &&
            selected.size() >= std::max<std::size_t>(1, selection_pages / 2);
        std::fprintf(stderr, "KVMEM_RESTORE_DIFF prev=%zu sel=%zu changed=%zu same=%d inc=%d\n",
            previous.size(), selected.size(), changed.size(), int(enough), int(incremental));
        if (asynchronous) {
            if (!host_published) throw std::logic_error("KVMem restored before storage completion");
            if (storage_work->restored.size())
                std::memcpy(p.memory_kv_staging->data(), storage_work->restored.data(), storage_work->restored.size());
        } else if (incremental) {
            for (const std::uint32_t i : changed) {
                record(selected[i])->read(0,
                    static_cast<std::byte*>(p.memory_kv_staging->data()) + i * backend.page_bytes, backend.page_bytes);
            }
        } else {
            for (std::size_t i = 0; i < selected.size(); ++i) {
                record(selected[i])->read(0,
                    static_cast<std::byte*>(p.memory_kv_staging->data()) + i * backend.page_bytes, backend.page_bytes);
            }
        }
        const auto restore_t_read = std::chrono::steady_clock::now();
        destructive = !incremental;
        if (!incremental) {
            p.text_kv_addresses->destructive_truncate(seq.kv->text, 0);
        } else if (previous.size() > selected.size()) {
            // Trim the reserve tail that the selector did not keep, while preserving the
            // mapping (and therefore the device replicas) of the retained prefix. The
            // frontier is the visible token count, not selected.size() * page_tokens: the
            // last page may be partial, and commit_frontier() rejects a smaller value.
            const auto frontier = static_cast<std::uint32_t>(plan.next.visible_tokens);
            p.text_kv_addresses->destructive_truncate(seq.kv->text, frontier);
        }
        const auto compact_end = static_cast<std::uint32_t>(plan.next.visible_tokens) + backend.end - seq.text_kv_valid;
        p.text_kv_addresses->ensure_mapped_to_tokens(seq.kv->text, compact_end, p.compute_streams);
        if (backend.mtp_layout) {
            if (!incremental) {
                p.backend_kv_addresses->destructive_truncate(*seq.kv->backend, 0);
            } else if (previous.size() > selected.size()) {
                const auto frontier = static_cast<std::uint32_t>(plan.next.visible_tokens);
                p.backend_kv_addresses->destructive_truncate(*seq.kv->backend, frontier);
            }
            p.backend_kv_addresses->ensure_mapped_to_tokens(*seq.kv->backend, compact_end, p.compute_streams);
        }
        const auto restore_t_remap = std::chrono::steady_clock::now();
        std::vector<std::uint32_t> slots;
        if (incremental) {
            slots = std::move(changed);
        } else {
            slots.resize(selected.size());
            for (std::uint32_t i = 0; i < selected.size(); ++i) slots[i] = i;
        }
        std::vector<DeviceKVPageHandle> destinations;
        std::vector<const std::byte*> records;
        destinations.reserve(slots.size());
        records.reserve(slots.size());
        for (const std::uint32_t i : slots) {
            destinations.push_back(p.text_kv_addresses->physical_page(seq.kv->text, i));
            records.push_back(restore_records[i]);
        }
        auto& pool = p.text_kv_pages->physical_pool();
        // All records are page slots of one pinned staging buffer, so every record has the
        // same allocation group; feeding block ids here would forbid run merging and turn
        // each plane copy into one transfer per page.
        if (!records.empty()) {
            pool.copy_from_host_records(records, {}, destinations, backend.layout, p.compute_streams);
        }
        std::vector<DeviceKVPageHandle> mtp_destinations;
        if (backend.mtp_layout) {
            std::vector<const std::byte*> mtp_records;
            mtp_destinations.reserve(slots.size());
            mtp_records.reserve(slots.size());
            for (const std::uint32_t i : slots) {
                mtp_destinations.push_back(p.backend_kv_addresses->physical_page(*seq.kv->backend, i));
                mtp_records.push_back(restore_records[i] + backend.layout.page_stride);
            }
            if (!mtp_records.empty()) {
                p.backend_kv_pages->physical_pool().copy_from_host_records(mtp_records, {},
                    mtp_destinations, *backend.mtp_layout, p.compute_streams);
            }
        }
        const auto restore_t_submit = std::chrono::steady_clock::now();
        p.device.synchronize();
        const auto restore_t_sync = std::chrono::steady_clock::now();
        {
            std::size_t runs = pool.contiguous_runs(destinations);
            const auto us = [](auto a, auto b) {
                return static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
            };
            std::fprintf(stderr,
                "KVMEM_RESTORE pages=%zu runs=%zu page_bytes=%zu async=%d read_us=%lld remap_us=%lld "
                "submit_us=%lld sync_us=%lld\n",
                destinations.size(), runs, backend.page_bytes, int(asynchronous),
                us(restore_t0, restore_t_read), us(restore_t_read, restore_t_remap),
                us(restore_t_remap, restore_t_submit), us(restore_t_submit, restore_t_sync));
            if (!destinations.empty()) {
                const auto& g = pool.geometry();
                std::fprintf(stderr,
                    "KVMEM_GEOM order=%d planes=%zu page_payload=%zu head_payload=%zu "
                    "page_stride=%zu planes_per_rank=%d\n",
                    int(g.device_plane_order), backend.layout.planes.size(),
                    backend.layout.planes.front().page_payload_bytes,
                    backend.layout.planes.front().head_payload_bytes, backend.layout.page_stride,
                    g.planes.empty() ? -1 : int(g.planes.size()));
            }
        }
        if (checked) {
            std::vector<std::byte*> rows;
            for (std::size_t i = 0; i < selected.size(); ++i) {
                rows.push_back(static_cast<std::byte*>(checked->data()) + i * backend.page_bytes);
            }
            pool.copy_to_host_records(destinations, rows, {}, backend.layout, p.compute_streams);
            if (backend.mtp_layout) {
                std::vector<std::byte*> mtp_rows;
                for (auto* record : rows) mtp_rows.push_back(record + backend.layout.page_stride);
                p.backend_kv_pages->physical_pool().copy_to_host_records(mtp_destinations, mtp_rows,
                    {}, *backend.mtp_layout, p.compute_streams);
            }
            std::array<std::byte*, 1> state{static_cast<std::byte*>(after->data())};
            p.state_images->copy_to_host_segments(p.state_store->physical_slot(seq.state.write),
                state, p.state_images->host_layout().image_bytes, p.compute_streams);
            p.device.synchronize();
            for (std::size_t i = 0; i < rows.size(); ++i) {
                if (std::memcmp(rows[i], restore_records[i], backend.page_bytes)) {
                    throw std::logic_error("KVMem native payload changed during restore");
                }
            }
            if (std::memcmp(before->data(), after->data(), p.state_images->host_layout().image_bytes)) {
                throw std::logic_error("KVMem page transfer changed recurrent state");
            }
        }
        seq.window.restored_bytes += slots.size() * backend.page_bytes;
    }

    void publish(const kvmem::ExecutionView& view) override {
        auto& p = backend.program;
        auto& seq = backend.sequence;
        p.text_kv_addresses->commit_frontier(seq.kv->text, static_cast<std::uint32_t>(view.visible_tokens));
        if (backend.mtp_layout) {
            p.backend_kv_addresses->commit_frontier(*seq.kv->backend,
                static_cast<std::uint32_t>(view.visible_tokens));
            // The earlier proposal was generated against the previous view. Rebuild
            // it from the next verified anchor instead of publishing stale drafts.
            seq.mtp_draft_count = 0;
        }
        seq.window.removed_tokens = seq.text_kv_valid - static_cast<std::uint32_t>(view.visible_tokens);
        seq.window.pages = std::move(selected);
        seq.window.stamp = view.stamp;
        ++seq.window.swaps;
        published = true;
        if (p.kvmem_options.verify_transfers) {
            std::fprintf(stderr, "KVMEM_NATIVE_VIEW logical=%u compact=%llu physical_pages=%u host_bytes=%llu swap=%u payload_exact=1 state_exact=1\n",
                seq.text_kv_valid, static_cast<unsigned long long>(view.visible_tokens),
                p.text_kv_pages->physical_pool().allocated_pages(),
                static_cast<unsigned long long>(seq.window.payload_bytes), seq.window.swaps);
        }
    }

    kvmem::AbortResult abort() noexcept override {
        // Cancellation never publishes a partial restore. Drain a started
        // CPU task before the caller may reuse this sequence's old records.
        if (asynchronous) {
            ticket.cancel();
            try { ticket.wait(); } catch (...) {}
        }
        try { backend.program.device.synchronize(); }
        catch (...) { backend.invalidate(); return kvmem::AbortResult::SessionInvalidated; }
        if (destructive && !published) {
            backend.invalidate();
            return kvmem::AbortResult::SessionInvalidated;
        }
        return kvmem::AbortResult::PreviousViewPreserved;
    }

private:
    struct StorageWork {
        StorageWork(const kvmem::HostKvStorage& storage, std::size_t stride) : storage(storage), stride(stride) {}
        kvmem::HostKvStorage storage;
        std::size_t stride;
        std::vector<std::uint32_t> spill_ids, selected;
        std::vector<KvmemHostRecord> records;
        std::vector<kvmem::HostKvSource> sources;
        KvmemHostRecord captured, restored;
    };
    void bind_staging() {
        auto* staging = backend.program.memory_kv_staging
            ? static_cast<std::byte*>(backend.program.memory_kv_staging->data()) : nullptr;
        spill_records.clear(); restore_records.clear();
        for (std::size_t i = 0; i < spill_ids.size(); ++i) spill_records.push_back(staging + i * backend.page_bytes);
        for (std::size_t i = 0; i < selected.size(); ++i) restore_records.push_back(staging + i * backend.page_bytes);
    }
    KvmemHostRecord* record(std::uint32_t id) {
        if (pending.at(id)) return &pending[id];
        auto & archived = backend.sequence.window.archive.at(id);
        return archived ? &archived : nullptr;
    }
    WindowMemoryBackend& backend;
    kvmem::WorkingSetPlan plan;
    std::vector<KvmemHostRecord> pending;
    std::vector<std::uint32_t> spill_ids, spill_slots, selected;
    std::vector<DeviceKVPageHandle> sources, mtp_sources;
    std::vector<std::uint64_t> mtp_source_epochs;
    std::vector<std::uint64_t> source_epochs;
    std::vector<std::byte*> spill_records;
    std::vector<const std::byte*> restore_records;
    std::unique_ptr<PinnedHostBuffer> before, after, checked;
    std::uint64_t prepared_payload_bytes = 0;
    const bool asynchronous;
    bool host_published = false;
    std::shared_ptr<StorageWork> storage_work;
    kvmem::StorageTicket ticket;
    bool destructive = false, published = false;
};

std::unique_ptr<kvmem::MemoryTransfer> WindowMemoryBackend::prepare(const kvmem::WorkingSetPlan& plan) {
    if (!(plan.expected == stamp())) { throw std::logic_error("KVMem plan is stale"); }
    return std::make_unique<WindowMemoryTransfer>(*this, plan);
}

std::vector<std::uint32_t> ProgramImpl::memory_image_pages(const PreparedPromptData& prompt) const {
    std::vector<std::uint32_t> pages;
    for (const auto& item : prompt.vision_items) {
        if (item.modality != PromptModality::Image) {
            throw RequestError(RequestErrorKind::InvalidMedia, "KVMem currently supports images, not video");
        }
        for (const auto& span : item.token_spans) {
            if (!span.count || span.begin > prompt.token_ids.size() ||
                span.count > prompt.token_ids.size() - span.begin) {
                throw std::logic_error("KVMem received an invalid image token span");
            }
            const auto last = (span.begin + span.count - 1) / page_tokens;
            for (auto id = span.begin / page_tokens; id <= last; ++id)
                pages.push_back(static_cast<std::uint32_t>(id));
        }
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    if (!pages.empty()) {
        std::uint32_t extra_sink = 0;
        for (std::uint32_t id = 0; id < kvmem_options.sink_blocks(); ++id) {
            if (!std::binary_search(pages.begin(), pages.end(), id)) ++extra_sink;
        }
        // Recent pages include the mandatory partial append tail; startup requires at least one.
        const auto tail_pages = kvmem_options.recent_blocks();
        if (pages.size() + extra_sink + tail_pages > kvmem_options.selected_tokens / page_tokens) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                "KVMem selection budget cannot retain complete image regions; increase the budget or reduce image resolution");
        }
    }
    return pages;
}

bool ProgramImpl::prepare_memory_append(SequenceState& sequence, std::uint32_t begin,
                                       std::uint32_t end, bool allow_pending) {
    if (!memory_io || begin == end) return true;
    auto& window = sequence.window;
    if (window.pending_append) {
        auto& preparation = *window.pending_append;
        if (!preparation.ticket.ready()) {
            if (allow_pending) return false;
            preparation.ticket.wait();
        }
        preparation.ticket.wait();
        auto& work = *preparation.work;
        for (std::size_t i = 0; i < work.ids.size(); ++i) {
            auto& destination = window.archive.at(work.ids[i]);
            if (!destination) window.payload_bytes += work.records[i].size();
            destination = std::move(work.records[i]);
        }
        window.pending_append.reset();
    }
    const auto count = kv_pages_for_tokens(end);
    if (window.archive.size() < count) window.archive.resize(count);
    if (window.host_versions.size() < count) window.host_versions.resize(count);
    std::shared_ptr<KvmemAppendPreparation::Work> work;
    for (auto id = begin / page_tokens; id < count; ++id) {
        const auto& record = window.archive[id];
        if (record.mutable_resident()) continue;
        if (!work) work = std::make_shared<KvmemAppendPreparation::Work>();
        work->ids.push_back(id);
        work->sources.push_back(record ? record.source() : kvmem::HostKvSource{});
    }
    if (!work) return true;
    const auto layout = plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry());
    const auto stride = layout.page_stride + (backend_kv_pages
        ? plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry()).page_stride : 0);
    work->records.resize(work->ids.size());
    auto preparation = std::make_unique<KvmemAppendPreparation>();
    preparation->work = work;
    preparation->ticket = memory_io->submit(work->ids.size() * stride,
        [work, storage = memory_payload_storage, stride]() mutable {
            for (std::size_t i = 0; i < work->ids.size(); ++i) {
                auto& destination = work->records[i];
                destination = storage.allocate(stride);
                if (work->sources[i]) work->sources[i].read(0, destination.writable_data(), stride);
            }
            work->sources.clear();
        });
    window.pending_append = std::move(preparation);
    if (allow_pending) return false;
    return prepare_memory_append(sequence, begin, end, false);
}

bool ProgramImpl::prepare_window(SequenceState& sequence, std::uint32_t begin, std::uint32_t end,
                                  bool force_selection, bool allow_pending) {
    if (!kvmem_window_tokens) { return true; }
    if (!sequence.kv || begin != sequence.text_kv_valid || end < begin || end > capacity ||
        end - begin > kvmem_options.reserve_tokens) {
        throw std::logic_error("KVMem append exceeds logical frontier or reserve");
    }
    auto& pending = sequence.window.pending_window;
    if (pending) {
        if (sequence.window.pending_window_begin != begin ||
            (end > sequence.window.pending_window_end && sequence.window.pending_window_end != begin))
            throw std::logic_error("KVMem pending window no longer matches its request boundary");
        for (;;) {
            const auto epoch = memory_io->completion_epoch();
            const auto phase = pending->advance();
            if (phase == kvmem::MemoryPhase::Ready) break;
            if (phase == kvmem::MemoryPhase::Spilling) {
                if (allow_pending) { sequence.window.pending_io_epoch = epoch; return false; }
                memory_io->wait_for_completion(epoch, std::chrono::milliseconds(1));
            }
        }
        pending->publish();
        pending.reset();
        force_selection = false;
    }
    if (end == begin && !force_selection) { return true; }
    // Begin the swap at 80% of the window so the transfer overlaps the decode that
    // follows instead of stalling the request at 100%. Selection is unchanged; only
    // the trigger moves earlier. A completed swap compacts the view back to
    // selected_tokens, which leaves the reserve free for the next segment.
    const std::uint32_t reselect_high_water =
        kvmem_window_tokens - kvmem_window_tokens / 5;
    const bool reselecting =
        force_selection || compact_position(sequence, end) > reselect_high_water;
    long long reselect_score_us = 0;
    long long reselect_select_us = 0;
    if (reselecting) {
        // A reselection replaces the selected blocks, but the masked-draft conditioning ring is
        // its own device allocation that every round appends to unconditionally, so paged-block
        // eviction cannot stale a ring row. The floor only has to stay within the ring: a floor at
        // `end - ring` leaves the whole ring readable (the kernel clamps the span to the ring
        // width), while the KVMem recent-page span silently hides most of it and drags acceptance
        // down with it.
        const std::uint64_t recent_tokens = std::max<std::uint64_t>(
            static_cast<std::uint64_t>(kvmem_options.recent_blocks()) * page_tokens,
            masked_draft_ring_capacity());
        sequence.dflash_context_floor =
            end > recent_tokens ? static_cast<std::uint32_t>(end - recent_tokens) : 0U;
        clamp_masked_draft_floor(sequence);
        auto backend = std::make_shared<WindowMemoryBackend>(*this, sequence, end);
        const auto count = kv_pages_for_tokens(begin);
        const auto keep = std::min(count, kvmem_options.selected_tokens / page_tokens);
        auto& state = sequence.window;
        std::vector<std::uint64_t> mandatory;
        if (count == 0) {
            mandatory.push_back(0);
        } else {
            const auto sink_keep = std::min(kvmem_options.sink_blocks(), count);
            for (std::uint32_t id = 0; id < sink_keep; ++id) mandatory.push_back(id);
            for (const auto id : state.media_pages) if (id < count) mandatory.push_back(id);
            const auto recent_keep = std::min(kvmem_options.recent_blocks(), count);
            for (std::uint32_t i = 0; i < recent_keep; ++i) mandatory.push_back(count - 1 - i);
        }
        std::sort(mandatory.begin(), mandatory.end());
        mandatory.erase(std::unique(mandatory.begin(), mandatory.end()), mandatory.end());
        std::vector<std::uint64_t> selected;
        const bool query_complete = state.query_tokens != 0 &&
            (state.query_replayed || begin >= state.query_end) &&
            state.query_tokens == state.query_end - state.query_begin;
        if (query_complete) {
            if (!state.selector || state.statistics_frontier != begin ||
                state.selector->total_tokens() != begin) {
                throw std::logic_error("KVMem retrieval index does not match the evaluated prefix");
            }
            const auto& config = parameters.model.config().text;
            std::vector<std::uint32_t> counts(config.full_attention_layers, state.query_tokens);
            const auto score_started = std::chrono::steady_clock::now();
            auto scores = kvmem::mean_key_retrieval_scores(
                *state.statistics, count, {state.query_sum.data(), counts.data(),
                    config.full_attention_layers, config.attention->head_dim,
                    config.attention->num_attention_heads, config.attention->num_key_value_heads,
                    kvmem_options.score_agg_max ? kvmem::GqaScoreAggregation::Max : kvmem::GqaScoreAggregation::Mean});
            const auto score_ended = std::chrono::steady_clock::now();
            state.selector->set_retrieval_scores(scores);
            const std::vector<std::uint32_t> required(mandatory.begin(), mandatory.end());
            const auto ids = state.selector->pick_topk_blocks(required);
            const auto select_ended = std::chrono::steady_clock::now();
            reselect_score_us = std::chrono::duration_cast<std::chrono::microseconds>(
                score_ended - score_started).count();
            reselect_select_us = std::chrono::duration_cast<std::chrono::microseconds>(
                select_ended - score_ended).count();
            selected.assign(ids.begin(), ids.end());
            if (kvmem_options.verify_transfers) {
                std::size_t cold = 0;
                for (auto id : ids) {
                    if (std::find(state.pages.begin(), state.pages.end(), id) == state.pages.end()) ++cold;
                }
                std::fprintf(stderr, "KVMEM_RETRIEVAL logical=%u query_tokens=%u cold=%zu selected=",
                    begin, state.query_tokens, cold);
                for (std::size_t i = 0; i < ids.size(); ++i) std::fprintf(stderr, "%s%u", i ? "," : "", ids[i]);
                std::fprintf(stderr, "\n");
            }
        } else {
            selected = mandatory;
            for (auto id = count; id-- > 0 && selected.size() < keep;) {
                if (std::find(selected.begin(), selected.end(), id) == selected.end()) selected.push_back(id);
            }
            std::sort(selected.begin(), selected.end());
        }
        const kvmem::MemoryBudget budget{kvmem_options.selected_tokens, kvmem_options.reserve_tokens,
            kvmem_window_tokens, memory_retained_budget};
        const auto plan_started = std::chrono::steady_clock::now();
        if (memory_io) {
            auto session = std::make_unique<kvmem::MemorySession>(backend);
            session->prepare(backend->stamp(), selected, mandatory, budget);
            sequence.window.pending_window_begin = begin;
            sequence.window.pending_window_end = end;
            pending = std::move(session);
            std::fprintf(stderr,
                "KVMEM_RESELECT logical=%u blocks=%llu score_us=%lld select_us=%lld plan_us=%lld "
                "transfer=async selected=%zu\n",
                begin, static_cast<unsigned long long>(count), reselect_score_us,
                reselect_select_us,
                static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - plan_started).count()),
                selected.size());
            return prepare_window(sequence, begin, end, false, allow_pending);
        }
        kvmem::MemorySession session(backend);
        session.prepare(backend->stamp(), selected, mandatory, budget);
        while (session.advance() != kvmem::MemoryPhase::Ready) {}
        session.publish();
        std::fprintf(stderr,
            "KVMEM_RESELECT logical=%u blocks=%llu score_us=%lld select_us=%lld transfer_us=%lld "
            "selected=%zu\n",
            begin, static_cast<unsigned long long>(count), reselect_score_us,
            reselect_select_us,
            static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - plan_started).count()),
            selected.size());
    }
    // Any host allocation/promotion that might spill belongs to the I/O job.
    // Checkpoint capture after this chunk only writes already-resident buffers.
    if (!prepare_memory_append(sequence, begin, end, allow_pending)) return false;
    ensure_sequence_kv_mapped(sequence, end, backend_kv_pages ? end : 0);
    const auto mapped = kv_pages_for_tokens(compact_position(sequence, end));
    while (sequence.window.pages.size() < mapped) {
        const auto slot = static_cast<std::uint32_t>(sequence.window.pages.size());
        sequence.window.pages.push_back(slot + sequence.window.removed_tokens / page_tokens);
    }
    if (text_kv_pages->physical_pool().allocated_pages() >
        static_cast<std::uint64_t>(max_concurrency) * kvmem_window_tokens / page_tokens) {
        throw std::logic_error("KVMem exceeded its native physical page budget");
    }
    return true;
}
bool ProgramImpl::memory_pending(SequenceHandle handle) const noexcept {
    if (!memory_io || !valid_sequence(handle)) return false;
    const auto& window = active_sequence(ContractAccess::lane(handle).value).window;
    if (window.pending_append && !window.pending_append->ticket.ready()) return true;
    return window.pending_window && window.pending_io_epoch == memory_io->completion_epoch() &&
        memory_io->pending_jobs() != 0;
}

bool ProgramImpl::prepare_decode_memory(SequenceHandle handle) {
    if (!memory_io) return true;
    if (!valid_sequence(handle)) throw std::logic_error("KVMem decode preparation has a stale sequence");
    if (memory_pending(handle)) return false;
    auto& sequence = active_sequence(ContractAccess::lane(handle).value);
    const auto append = backend_kv_pages ? std::max(draft_window, ngram_draft_window) + draft_window : 1U;
    return prepare_window(sequence, sequence.execution_frontier,
        std::min(capacity, sequence.execution_frontier + append), false, true);
}
std::uint32_t ProgramImpl::memory_control_chunk() const noexcept {
    return memory_io ? std::min(prefill_chunk, kvmem_options.reserve_tokens) : UINT32_MAX;
}
bool ProgramImpl::prepare_control_memory(SequenceHandle handle, std::uint32_t tokens) {
    if (!memory_io) return true;
    if (!valid_sequence(handle)) throw std::logic_error("KVMem control preparation has a stale sequence");
    if (memory_pending(handle)) return false;
    auto& sequence = active_sequence(ContractAccess::lane(handle).value);
    return prepare_window(sequence, sequence.execution_frontier,
                          sequence.execution_frontier + tokens, false, true);
}
bool ProgramImpl::cancel_memory_ready(SequenceHandle handle) noexcept {
    if (!memory_io || !valid_sequence(handle)) return true;
    auto& window = active_sequence(ContractAccess::lane(handle).value).window;
    if (window.pending_append) window.pending_append->ticket.cancel();
    window.pending_window_io.cancel();
    // Keep the sequence parked while an already-started CPU task drains.
    // Other lanes remain runnable; abort will only encounter ready tickets.
    return (!window.pending_append || window.pending_append->ticket.ready()) &&
           window.pending_window_io.ready();
}
// Capture uses the same complete-before-publish Host spill as a working-set
// transfer, but stops before any destructive restore. Native device ownership
// and attention view remain unchanged.
void ProgramImpl::archive_memory_window(SequenceState& sequence) {
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto snapshot = backend.snapshot();
    kvmem::WorkingSetPlan plan;
    plan.expected = snapshot.stamp;
    for (auto id : snapshot.resident) {
        const auto& block = snapshot.blocks.at(static_cast<std::size_t>(id));
        plan.next.blocks.push_back({block.block, plan.next.visible_tokens});
        plan.next.visible_tokens += block.block.valid_tokens;
    }
    WindowMemoryTransfer spill(backend, plan, false);
    try {
        spill.start_spill();
    } catch (...) {
        (void)spill.abort();
        throw;
    }
}

bool ProgramImpl::memory_query_probe(const SequenceState& sequence, std::uint32_t prompt_tokens) const {
    const auto& window = sequence.window;
    return kvmem_window_tokens && prompt_tokens > kvmem_window_tokens &&
        window.query_begin > 0 && window.query_end > window.query_begin && !window.query_replayed;
}

std::uint32_t ProgramImpl::memory_checkpoint_frontier(const SequenceState& sequence,
    std::uint32_t prompt_tokens, bool allow_reuse) const {
    if (memory_query_probe(sequence, prompt_tokens)) return sequence.window.query_begin;
    // Leave one suffix token to evaluate and sample normally, including the MTP
    // boundary row rebuilt on restore. No query probe/replay is needed in this path.
    if (kvmem_window_tokens && allow_reuse && prompt_tokens > 1 &&
        prompt_tokens <= kvmem_window_tokens) return prompt_tokens - 1;
    return 0;
}

void ProgramImpl::capture_memory_prefix(SequenceState& sequence, std::uint32_t chunk_tokens) {
    auto& window = sequence.window;
    if (!chunk_tokens || (window.prefix_checkpoint &&
        window.prefix_checkpoint->prefix.size() >= sequence.text_kv_valid)) {
        throw std::logic_error("KVMem prefix checkpoint is not at an uncaptured evaluated boundary");
    }
    copy_tail(sequence, prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1));
    window.prefix_checkpoint = capture_memory_checkpoint(sequence);
}

std::unique_ptr<KvmemPrefixCheckpoint> ProgramImpl::capture_memory_checkpoint(SequenceState& sequence) {
    auto& window = sequence.window;
    auto checkpoint = std::make_unique<KvmemPrefixCheckpoint>();
    const auto frontier = sequence.text_kv_valid;
    if (!frontier || frontier > sequence.ledger.size() || frontier != window.statistics_frontier ||
        !sequence.tail_hidden_valid || sequence.state.fork_pending ||
        (backend_kv_pages && kvmem_backend_frontier(sequence) != frontier)) {
        throw std::logic_error("KVMem checkpoint is not a complete committed boundary");
    }
    checkpoint->prefix.assign(sequence.ledger.begin(), sequence.ledger.begin() + frontier);
    checkpoint->prefix_identity = sequence.prefix_identity;
    checkpoint->prefix_identity.truncate(frontier);
    checkpoint->mean_tail = window.statistics->mean_checkpoint(frontier);
    checkpoint->history = window.selector->retrieval_checkpoint();
    const auto bytes = state_images->host_layout().image_bytes;
    checkpoint->state = std::make_unique<PinnedHostBuffer>(bytes);
    std::memset(checkpoint->state->data(), 0, bytes);
    archive_memory_window(sequence);
    WindowMemoryBackend backend(*this, sequence, frontier);
    const auto descriptor = backend.descriptor();
    checkpoint->identity = {descriptor.identity, descriptor.layout, backend.stamp()};
    checkpoint->resident_pages = backend.snapshot().resident;
    std::array<std::byte*, 1> host{static_cast<std::byte*>(checkpoint->state->data())};
    state_images->copy_to_host_segments(state_store->physical_slot(sequence.state.write),
        host, bytes, compute_streams);
    device.synchronize();
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_PREFIX_CHECKPOINT frontier=%u state_bytes=%zu\n", frontier, bytes);
    }
    return checkpoint;
}

std::unique_ptr<KvmemPrefixCheckpoint>& ProgramImpl::memory_checkpoint_slot(KvmemWindowState& window,
                                                                          KvmemCheckpointKind kind) {
    switch (kind) {
    case KvmemCheckpointKind::Base: return window.prefix_checkpoint;
    case KvmemCheckpointKind::Endpoint: return window.endpoint_checkpoint;
    case KvmemCheckpointKind::InputFallback: return window.input_checkpoint;
    case KvmemCheckpointKind::Reconstruction: return window.reconstruction_checkpoint;
    }
    throw std::logic_error("invalid KVMem checkpoint kind");
}

const KvmemPrefixCheckpoint* ProgramImpl::memory_checkpoint(const KvmemWindowState& window,
                                                           KvmemCheckpointKind kind) const {
    switch (kind) {
    case KvmemCheckpointKind::Base: return window.prefix_checkpoint.get();
    case KvmemCheckpointKind::Endpoint: return window.endpoint_checkpoint.get();
    case KvmemCheckpointKind::InputFallback: return window.input_checkpoint.get();
    case KvmemCheckpointKind::Reconstruction: return window.reconstruction_checkpoint.get();
    }
    return nullptr;
}

void ProgramImpl::capture_memory_endpoint(SequenceState& sequence, const RequestControl& request) noexcept {
    if (!kvmem_window_tokens || !request.allow_memory_reuse || !sequence.window.statistics ||
        request.lifecycle != Lifecycle::Finishable) return;
    const auto frontier = sequence.execution_frontier;
    if (sequence.text_kv_valid != frontier || !sequence.tail_hidden_valid ||
        sequence.state.fork_pending || (backend_kv_pages && kvmem_backend_frontier(sequence) != frontier)) return;
    for (const auto kind : kKvmemCheckpointKinds)
        if (const auto* checkpoint = memory_checkpoint(sequence.window, kind);
            checkpoint && checkpoint->prefix.size() == frontier) return;
    try {
        const auto started = Clock::now();
        sequence.window.endpoint_checkpoint = capture_memory_checkpoint(sequence);
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_ENDPOINT frontier=%u committed=1 capture_ms=%.3f\n", frontier,
                std::chrono::duration<double, std::milli>(Clock::now() - started).count());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_ENDPOINT capture skipped: %s\n", error.what());
    }
}

bool ProgramImpl::memory_same_query(const KvmemWindowState& window, const PreparedPromptData& prompt,
                                    std::uint32_t frontier) const {
    if (!prompt.memory_query || !window.query_tokens ||
        window.query_tokens != window.query_end - window.query_begin) return false;
    const auto& query = *prompt.memory_query;
    // The caller's exact checkpoint match proves all tokens/positions through this range.
    if (query.begin != window.query_begin || query.count != window.query_tokens) return false;
    // LCP boundary: a reuse may not restore past the evaluated frontier. The replay itself
    // keeps the committed query sum, so no logits are needed at the boundary.
    const auto limit = kvmem::prefix_restore_limit(
        window.query_end, window.query_end, frontier, /*needs_logits=*/false);
    return limit >= window.query_end;
}

void ProgramImpl::capture_memory_rewrite(SequenceState& sequence, KvmemCheckpointKind kind) noexcept {
    auto& window = sequence.window;
    const auto* purpose = kind == KvmemCheckpointKind::InputFallback ? "input" : "reconstruction";
    try {
        if (kind != KvmemCheckpointKind::InputFallback && kind != KvmemCheckpointKind::Reconstruction)
            throw std::logic_error("invalid KVMem rewrite checkpoint purpose");
        if (!window.prefix_checkpoint || window.prefix_checkpoint->prefix.size() != sequence.text_kv_valid)
            memory_checkpoint_slot(window, kind) = capture_memory_checkpoint(sequence);
        if (kvmem_options.verify_transfers)
            std::fprintf(stderr, "KVMEM_REWRITE purpose=%s frontier=%u canonical=1\n",
                         purpose, sequence.text_kv_valid);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "KVMEM_REWRITE purpose=%s capture skipped: %s\n", purpose, error.what());
    }
    if (kind == KvmemCheckpointKind::Reconstruction) window.reconstruction_frontier = 0;
}

void ProgramImpl::capture_memory_reconstruction_if_ready(SequenceState& sequence) {
    auto& window = sequence.window;
    if (window.reconstruction_frontier < sequence.text_kv_valid) window.reconstruction_frontier = 0;
    if (!kvmem_window_tokens || !requests[sequence.lane].allow_memory_reuse ||
        !window.reconstruction_frontier || window.reconstruction_frontier != sequence.text_kv_valid ||
        sequence.execution_frontier != sequence.text_kv_valid ||
        (backend_kv_pages && kvmem_backend_frontier(sequence) != sequence.text_kv_valid)) return;
    capture_memory_rewrite(sequence, KvmemCheckpointKind::Reconstruction);
}

void ProgramImpl::rewind_memory_query(SequenceState& sequence) {
    auto& window = sequence.window;
    if (!window.prefix_checkpoint || window.query_replayed ||
        window.query_tokens != window.query_end - window.query_begin ||
        sequence.state.read != sequence.state.write || sequence.state.fork_pending) {
        throw std::logic_error("KVMem query replay has no complete exclusive checkpoint");
    }
    const auto& checkpoint = *window.prefix_checkpoint;
    if (checkpoint.prefix.size() != window.query_begin) {
        throw std::logic_error("KVMem query checkpoint is not the current query boundary");
    }
    const auto previous_frontier = sequence.text_kv_valid;
    window.query_replayed = true; // Probe Q stays frozen while replay replaces K once.
    restore_memory_prefix(sequence, checkpoint, false, true);
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_QUERY_REPLAY from=%u to=%u query_tokens=%u prefix_match=1 state_exact=1\n",
            previous_frontier, sequence.text_kv_valid, window.query_tokens);
    }
}

void ProgramImpl::restore_memory_prefix(SequenceState& sequence, const KvmemPrefixCheckpoint& checkpoint,
                                        bool preserve_view, bool allow_pending, bool resident_native) {
    auto& window = sequence.window;
    const auto frontier = static_cast<std::uint32_t>(checkpoint.prefix.size());
    WindowMemoryBackend backend(*this, sequence, sequence.text_kv_valid);
    const auto descriptor = backend.descriptor();
    const kvmem::CheckpointIdentity expected{descriptor.identity, descriptor.layout, checkpoint.identity.stamp};
    if (!checkpoint.identity.compatible_with(expected) ||
        !(checkpoint.identity.stamp.session == window.stamp.session) ||
        frontier != checkpoint.identity.stamp.frontier || frontier > sequence.ledger.size() ||
        !std::equal(checkpoint.prefix.begin(), checkpoint.prefix.end(), sequence.ledger.begin()) ||
        !checkpoint.prefix_identity.prefix_equals(sequence.prefix_identity, frontier)) {
        throw std::logic_error("KVMem rejected a stale query checkpoint before mutation");
    }
    const auto bytes = state_images->host_layout().image_bytes;
    // All prefix records were made durable at capture. Query appends cannot alter
    // native values/scales in earlier rows; the partial page is explicitly clipped.
    const auto pages = kv_pages_for_tokens(frontier);
    if (window.archive.size() < pages) throw std::logic_error("KVMem checkpoint lost Host prefix");
    for (std::uint32_t id = 0; id < pages; ++id) {
        if (!window.archive[id] || window.host_versions[id] < valid_columns(id, frontier)) {
            throw std::logic_error("KVMem checkpoint has an incomplete native prefix");
        }
    }
    window.statistics->truncate_to(frontier);
    window.statistics->restore_mean_checkpoint(frontier, checkpoint.mean_tail);
    window.selector->truncate_to(frontier);
    if (preserve_view) window.selector->restore_retrieval(checkpoint.history);
    window.statistics_frontier = frontier;
    window.archive.resize(pages);
    window.host_versions.resize(pages);
    window.payload_bytes = pages * backend.page_bytes;
    if (frontier % page_tokens) window.host_versions.back() = frontier % page_tokens;
    // A consumed history has one archive. Rewinding explicitly revokes later states
    // before their payload/statistics can be overwritten; no full-history COW is needed.
    for (const auto kind : kKvmemCheckpointKinds) {
        auto& slot = memory_checkpoint_slot(window, kind);
        if (slot && slot->prefix.size() > frontier) slot.reset();
    }
    sequence.text_kv_valid = frontier;
    if (backend_kv_pages) {
        // The captured backend page covers [0, frontier) for every drafter, so the restored
        // backend frontier is the same value. MTP additionally forgets its draft count.
        kvmem_set_backend_frontier(sequence, frontier);
        if (speculative_backend == SpeculativeBackend::Mtp) sequence.mtp_draft_count = 0;
    }
    ++window.stamp.execution_history;
    state_store->begin_active_overwrite(sequence.state.write);
    std::array<const std::byte*, 1> host{static_cast<const std::byte*>(checkpoint.state->data())};
    state_images->copy_from_host_segments(host, bytes,
        state_store->physical_slot(sequence.state.write), compute_streams);
    device.synchronize();
    if (kvmem_options.verify_transfers) {
        PinnedHostBuffer verify(bytes);
        std::memset(verify.data(), 0, bytes);
        std::array<std::byte*, 1> destination{static_cast<std::byte*>(verify.data())};
        state_images->copy_to_host_segments(state_store->physical_slot(sequence.state.write),
            destination, bytes, compute_streams);
        device.synchronize();
        if (std::memcmp(verify.data(), checkpoint.state->data(), bytes)) {
            throw std::logic_error("KVMem query StateImage restore changed native bytes");
        }
    }
    if (preserve_view) {
        auto restored = std::make_shared<WindowMemoryBackend>(*this, sequence, frontier, resident_native);
        kvmem::MemorySession session(restored);
        // A checkpoint includes already appended reserve rows. Restore its exact
        // view within B+R, then ordinary append pressure restores the B/R split.
        session.prepare(restored->stamp(), checkpoint.resident_pages, checkpoint.resident_pages,
            {kvmem_window_tokens, 0, kvmem_window_tokens, memory_retained_budget});
        for (;;) {
            const auto epoch = memory_io ? memory_io->completion_epoch() : 0;
            if (session.advance() == kvmem::MemoryPhase::Ready) break;
            if (memory_io && session.phase() == kvmem::MemoryPhase::Spilling)
                memory_io->wait_for_completion(epoch, std::chrono::milliseconds(1));
        }
        session.publish();
    } else {
        prepare_window(sequence, frontier, frontier, true, allow_pending);
    }
    sequence.tail_hidden_valid = true;
}

std::optional<KvmemHistoryMatch> ProgramImpl::memory_restorable_history(const PreparedPromptData& prompt) const {
    if (!kvmem_window_tokens || !prompt.identity.reusable || (prompt.context_cache.session_key &&
        !prompt.context_cache.update_session_index)) return std::nullopt;
    std::optional<KvmemHistoryMatch> best;
    std::size_t best_frontier = 0;
    unsigned prefix_rejected = 0, identity_rejected = 0, query_rejected = 0, incomplete = 0;
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        const auto& history = memory_histories[index];
        if (!history || !history->stamp.valid ||
            history->session_key != prompt.context_cache.session_key) continue;
        for (const auto kind : kKvmemCheckpointKinds) {
            const auto* candidate = memory_checkpoint(*history, kind);
            if (!candidate) continue;
            const auto& checkpoint = *candidate;
            const auto frontier = static_cast<std::uint32_t>(checkpoint.prefix.size());
            const auto pages = kv_pages_for_tokens(frontier);
            bool complete = frontier && checkpoint.state_bytes() &&
                checkpoint.state_bytes() == state_images->host_layout().image_bytes &&
                frontier == checkpoint.identity.stamp.frontier && history->statistics && history->selector &&
                history->statistics_frontier >= frontier && history->archive.size() >= pages &&
                history->host_versions.size() >= pages;
            for (std::uint32_t page = 0; complete && page < pages; ++page)
                complete = history->archive[page] && history->host_versions[page] >= valid_columns(page, frontier);
            if (!complete) { ++incomplete; continue; }
            if (frontier >= prompt.token_ids.size() ||
                !std::equal(checkpoint.prefix.begin(), checkpoint.prefix.end(), prompt.token_ids.begin())) {
                ++prefix_rejected; continue;
            }
            if (!checkpoint.prefix_identity.matches(prompt, frontier)) { ++identity_rejected; continue; }
            if (prompt.token_ids.size() > kvmem_window_tokens && prompt.memory_query &&
                frontier > prompt.memory_query->begin && !memory_same_query(*history, prompt, frontier)) {
                ++query_rejected; continue;
            }
            if (frontier > best_frontier || (frontier == best_frontier && best &&
                    history->publication_order > memory_histories[best->history]->publication_order)) {
                best = KvmemHistoryMatch{index, kind, frontier};
                best_frontier = frontier;
            }
        }
    }
    if (kvmem_options.verify_transfers)
        std::fprintf(stderr, "KVMEM_HISTORY_LOOKUP frontier=%zu prefix_rejected=%u identity_rejected=%u query_rejected=%u incomplete=%u\n",
                     best_frontier, prefix_rejected, identity_rejected, query_rejected, incomplete);
    return best;
}

bool ProgramImpl::prepare_memory_materialization(MaterializationTransaction& transaction) {
    if (!memory_io) return true;
    auto& staged = *requests[transaction.destination.value].prefill;
    if (!staged.memory_restore_frontier || !staged.memory_restore_generation) return true;
    if (transaction.memory_history && !transaction.memory_history->ticket.ready()) return false;
    const auto match = memory_restorable_history(staged.prompt);
    if (!match || match->frontier != staged.memory_restore_frontier ||
        memory_histories[match->history]->stamp.session.id != staged.memory_restore_generation) {
        transaction.memory_history.reset();
        return true; // ordinary quoted-prefix replay remains the fallback
    }
    auto& history = *memory_histories[match->history];
    auto& pending = transaction.memory_history;
    try {
        if (pending) {
            if (pending->history != match->history || pending->generation != history.stamp.session.id)
                throw std::logic_error("prepared SSD history changed owner");
            if (pending->native_ready) return true;
            if (!pending->ticket.ready()) return false;
            pending->ticket.wait();
            auto& work = *pending->work;
            if (pending->checkpoint) {
                auto& checkpoint = memory_checkpoint_slot(history, *pending->checkpoint);
                if (!checkpoint || checkpoint->state || checkpoint->cold_state.size() != work.state.size())
                    throw std::logic_error("prepared state image changed before publication");
                auto state = std::make_unique<PinnedHostBuffer>(work.state.size());
                std::memcpy(state->data(), work.state.data(), work.state.size());
                checkpoint->state = std::move(state);
                checkpoint->cold_state = {};
                pending.reset();
            } else {
                for (std::size_t i = 0; i < work.ids.size(); ++i)
                    if (work.records[i]) history.archive.at(work.ids[i]) = std::move(work.records[i]);
                pending->native_ready = true;
                return true; // leases retain prepared RAM until GPU restoration
            }
        }
        for (const auto kind : kKvmemCheckpointKinds) {
            const auto* checkpoint = memory_checkpoint(history, kind);
            if (!checkpoint || checkpoint->state || !checkpoint->cold_state.size()) continue;
            // A cancelled read may still own this bounded CPU buffer. Let it
            // drain without turning the next cache hit into an allocation error.
            if (memory_state_staging.charged_bytes()) return false;
            auto prepared = std::make_unique<KvmemHistoryPreparation>();
            prepared->history = match->history;
            prepared->generation = history.stamp.session.id;
            prepared->checkpoint = kind;
            prepared->work = std::make_shared<KvmemHistoryPreparation::Work>();
            prepared->work->state_source = checkpoint->cold_state;
            prepared->ticket = memory_io->submit(checkpoint->cold_state.size(),
                [work = prepared->work, storage = memory_state_staging]() mutable {
                    work->state = storage.allocate(work->state_source.size());
                    work->state_source.read(0, work->state.writable_data(), work->state.size());
                });
            pending = std::move(prepared);
            return false;
        }
        auto prepared = std::make_unique<KvmemHistoryPreparation>();
        prepared->history = match->history;
        prepared->generation = history.stamp.session.id;
        prepared->work = std::make_shared<KvmemHistoryPreparation::Work>();
        auto& work = *prepared->work;
        const auto& checkpoint = *memory_checkpoint(history, match->kind);
        for (const auto id : checkpoint.resident_pages) {
            work.ids.push_back(static_cast<std::uint32_t>(id));
            work.sources.push_back(history.archive.at(id).source());
        }
        work.records.resize(work.ids.size());
        work.leases.resize(work.ids.size());
        const auto stride = text_host_kv_page_stride + backend_host_kv_page_stride;
        const auto tail = (match->frontier - 1) / page_tokens;
        prepared->ticket = memory_io->submit(work.ids.size() * stride,
            [work = prepared->work, storage = memory_payload_storage, stride, tail]() mutable {
                for (std::size_t i = 0; i < work->ids.size(); ++i) {
                    auto source = work->sources[i].lease();
                    if (source.data() && work->ids[i] != tail) {
                        work->leases[i] = std::move(source);
                    } else {
                        auto& record = work->records[i];
                        record = storage.allocate(stride);
                        source.read(0, record.writable_data(), stride);
                        work->leases[i] = record.lease();
                    }
                }
                work->sources.clear();
            });
        pending = std::move(prepared);
        return false;
    } catch (const kvmem::MemoryError& error) {
        // Keep the original history valid. Recompute the already quoted prefix
        // using its reserved service budget if an optional cache read fails.
        std::fprintf(stderr, "KVMEM_HISTORY SSD preparation failed; replaying prefix: %s\n", error.what());
        pending.reset();
        staged.memory_restore_generation = 0;
        return true;
    }
}

bool ProgramImpl::restore_memory_history(SequenceState& sequence, RequestControl::Prefill& staged) {
    const auto started = Clock::now();
    const auto match = memory_restorable_history(staged.prompt);
    if (!match) return false;
    auto& history = memory_histories[match->history];
    const auto frontier = std::optional<std::uint32_t>(match->frontier);
    if (*frontier != staged.memory_restore_frontier ||
        staged.memory_restore_generation != history->stamp.session.id) return false;
    // Materialize attachments before consuming the history or mutating native
    // state. A failed read leaves every cold source valid and retryable.
    for (const auto kind : kKvmemCheckpointKinds) {
        auto& slot = memory_checkpoint_slot(*history, kind);
        if (!slot || slot->state || !slot->cold_state.size()) continue;
        auto state = std::make_unique<PinnedHostBuffer>(slot->cold_state.size());
        slot->cold_state.read(0, state->data(), state->size());
        slot->state = std::move(state);
        slot->cold_state = {};
    }
    sequence.window = std::move(*history);
    history.reset();
    auto& window = sequence.window;
    const bool same_query = memory_same_query(window, staged.prompt, *frontier);
    auto checkpoint = std::move(memory_checkpoint_slot(window, match->kind));
    // The previous Program address-space lease has been released. These IDs
    // must never be treated as resident in the new native address space.
    window.pages.clear();
    window.removed_tokens = 0;
    window.reconstruction_frontier = 0;
    window.query_replayed = same_query;
    const auto query = staged.prompt.memory_query.value_or(qwen3_5::TokenSpan{});
    window.query_begin = static_cast<std::uint32_t>(query.begin);
    window.query_end = static_cast<std::uint32_t>(query.begin + query.count);
    if (!same_query) {
        window.query_tokens = 0;
        std::fill(window.query_sum.begin(), window.query_sum.end(), 0.0F);
    }
    // The durable archive already covers the quoted prefix. This is not a
    // native allocation frontier; force-selection creates only B resident rows.
    sequence.text_kv_valid = *frontier;
    if (backend_kv_pages) kvmem_set_backend_frontier(sequence, *frontier);
    auto* materialization = std::get_if<MaterializationTransaction>(&context_transaction_);
    const bool resident_native = materialization && materialization->memory_history &&
        materialization->memory_history->native_ready;
    restore_memory_prefix(sequence, *checkpoint, true, false, resident_native);
    if (resident_native) materialization->memory_history.reset();
    if (kvmem_options.lrfu) {
        const auto limit = std::min(window.history_input.size(), staged.prompt.token_ids.size());
        std::size_t prefix = 0;
        while (prefix < limit && window.history_input[prefix] == staged.prompt.token_ids[prefix]) ++prefix;
        const auto input_end = staged.prompt.memory_input_end ? staged.prompt.memory_input_end : staged.prompt.token_ids.size();
        auto count = kvmem::lrfu_prefix_points(window.history_points, prefix, input_end);
        while (count && !window.history_input_identity.matches(staged.prompt, window.history_points[count - 1].input_end))
            --count;
        window.history_points.resize(count);
        window.selector->restore_retrieval(count ? window.history_points.back().state : kvmem::LrfuState{});
        if (kvmem_options.verify_transfers) std::fprintf(stderr,
            "KVMEM_LRFU_BEGIN parent=%llu inherited_inputs=%zu\n",
            static_cast<unsigned long long>(window.selector->retrieval_checkpoint().round), count);
    }
    window.selector->begin_retrieval_round();
    if (backend_kv_pages) {
        // A masked draft (DFlash, DFlash2) keeps a plain one-row-per-token conditioning cache,
        // already restored whole by the checkpoint image, so it only has to publish the restored
        // frontier. MTP instead pairs target hidden at i with the embedding of token i+1, so its
        // last row is half-built after a restore and must be rebuilt from this request's first
        // suffix token before it is trusted.
        if (speculative_backend != SpeculativeBackend::Mtp) {
            kvmem_set_backend_frontier(sequence, *frontier);
            // The ring snapshot holds the trailing window; anything earlier may be reselected
            // history and must not feed the draft.
            const std::uint64_t recent_tokens =
                static_cast<std::uint64_t>(kvmem_options.recent_blocks()) * page_tokens;
            sequence.dflash_context_floor = *frontier > recent_tokens
                ? static_cast<std::uint32_t>(*frontier - recent_tokens) : 0U;
            // Re-anchor the restored ring to the sequence's current write slot.
            sequence.dflash_state_slot = state_selectors(sequence).destination;
            ensure_sequence_kv_mapped(sequence, *frontier, *frontier);
            commit_sequence_kv(sequence, *frontier, *frontier);
            window.host_versions.at((*frontier - 1U) / page_tokens) = 0;
            archive_memory_window(sequence);
        } else {
            // An MTP row combines target hidden at i with the embedding of token i+1.
            // Prefix identity proves only [0,frontier), so its last MTP row must be
            // rebuilt using this request's first suffix token before it is trusted.
            const auto last = *frontier - 1U;
            sequence.mtp_kv_valid = last;
            trim_sequence_kv(sequence, *frontier, last);
            ensure_sequence_kv_mapped(sequence, *frontier, *frontier);
            const auto selectors = state_selectors(sequence);
            execution::PrefillContext bridge{
                {device, parameters, work, state_images->linear(0),
                 replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
                 proposal_head, stage_runtime.get(), rope_yarn, fast_prefill_kernel,
                 mtp_attention_window},
                text_kv_view(sequence), mtp_kv_view(sequence), decoder->text_kv,
                decoder->mtp_cache(), nullptr, *frontier, nullptr, nullptr,
                selectors.source, selectors.destination, 0, 0};
            bridge.cache_position_shift = window.removed_tokens;
            set_device_i32(io.rope_delta, sequence.rope_delta +
                checked_i32(window.removed_tokens, "restored MTP cache shift"));
            auto token = io.mtp->target_input_ids.slice(0, 0, 1);
            set_device_i32(token, staged.prompt.token_ids[*frontier]);
            const auto tokens = staged.prompt.token_ids.size();
            const std::array<std::int32_t, 3> rope{
                staged.prompt.positions.at(last), staged.prompt.positions.at(tokens + last),
                staged.prompt.positions.at(2 * tokens + last)};
            mark_workspace_usage(workspace_plan.mtp_prefill);
            if (staged.vision) {
                execution::mtp_bridge_multimodal(bridge, staged.prompt, *staged.vision,
                    {.previous_hidden = &sequence.tail_hidden,
                     .position = checked_i32(last, "restored MTP tail"), .rope_position = rope});
            } else {
                execution::mtp_bridge_and_propose(bridge, token, sequence.tail_hidden,
                    checked_i32(last, "restored MTP tail"), rope, false);
            }
            device.synchronize();
            work.reset();
            sequence.mtp_kv_valid = *frontier;
            commit_sequence_kv(sequence, *frontier, *frontier);
            window.host_versions.at(last / page_tokens) = 0;
            // Same-boundary query replay can restore before another view switch.
            archive_memory_window(sequence);
        }
    }
    if (*frontier == memory_checkpoint_frontier(sequence, staged.prompt_tokens, true))
        window.prefix_checkpoint = std::move(checkpoint);
    else
        // An exact replay can start at the input capture boundary, so prefill
        // will not cross it again. Keep the restored complete state for repeats
        // and cancellation recovery; restoring does not consume its payload.
        memory_checkpoint_slot(window, match->kind) = std::move(checkpoint);
    staged.base = *frontier;
    staged.cursor = *frontier;
    ++memory_history_hits;
    if (kvmem_options.verify_transfers) {
        std::fprintf(stderr, "KVMEM_HISTORY_RESTORE frontier=%u kind=%u same_query=%u prefix_match=1 state_exact=1 restore_ms=%.3f\n",
            *frontier, static_cast<unsigned>(match->kind), same_query ? 1U : 0U,
            std::chrono::duration<double, std::milli>(Clock::now() - started).count());
    }
    return true;
}

void ProgramImpl::reserve_memory_history(SequenceState& sequence, RequestControl& request,
                                         const PreparedPromptData& prompt) {
    const auto& key = prompt.context_cache.session_key;
    if (key && prompt.context_cache.update_session_index) {
        MemoryPublication* target = nullptr;
        for (auto& publication : memory_publications) {
            if (publication.key == key) { target = &publication; break; }
            if (!publication.key) { target = &publication; }
        }
        if (!target) {
            for (auto& publication : memory_publications) {
                const bool active = std::any_of(requests.begin(), requests.end(), [&](const auto& value) {
                    return value.memory_session_key == publication.key;
                });
                const bool retained = std::any_of(memory_histories.begin(), memory_histories.end(), [&](const auto& value) {
                    return value && value->session_key == publication.key;
                });
                if (!active && !retained && (!target || publication.order < target->order)) target = &publication;
            }
        }
        if (!target) throw std::logic_error("KVMem publication table has no reclaimable entry");
        if (target->key != key) { target->key = key; target->order = 0; }
        target->order = std::max(target->order, sequence.window.publication_order);
    }
    if (!sequence.window.statistics) ++memory_history_misses;
    // An explicit fresh request invalidates only its own lineage. A caller that
    // forbids index updates cannot consume or erase the named cached history.
    if (!request.allow_memory_reuse && prompt.context_cache.update_session_index) {
        for (auto& history : memory_histories) {
            if (history && history->session_key == prompt.context_cache.session_key &&
                history->publication_order <= sequence.window.publication_order) history.reset();
        }
    }
    std::uint64_t reserved = 0;
    for (const auto& active : requests) reserved += active.memory_payload_reservation;
    if (reserved > memory_retained_budget) {
        throw std::logic_error("KVMem active payload reservations exceed admission capacity");
    }
    auto available = memory_retained_budget - reserved;
    for (;;) {
        std::uint64_t cached_bytes = memory_idle_transfer ? memory_idle_transfer->work->destination.size() : 0;
        std::optional<std::size_t> victim;
        for (std::size_t index = 0; index < memory_histories.size(); ++index) {
            const auto& history = memory_histories[index];
            if (!history) continue;
            cached_bytes += history->payload_bytes;
            for (const auto kind : kKvmemCheckpointKinds)
                if (const auto* checkpoint = memory_checkpoint(*history, kind))
                    cached_bytes += checkpoint->cold_state.size();
            if (!victim || history->publication_order < memory_histories[*victim]->publication_order) victim = index;
        }
        if (cached_bytes <= available) break;
        if (!victim) throw std::logic_error("KVMem retained Host accounting has no victim");
        memory_histories[*victim].reset();
        ++memory_history_evictions;
    }
}

void ProgramImpl::retain_memory_history(SequenceState& sequence, const RequestControl& request) noexcept {
    // A cancelled request can retain an earlier complete checkpoint, but cannot
    // move a transaction whose backend still addresses the active sequence.
    sequence.window.pending_window.reset();
    sequence.window.pending_append.reset();
    if (!kvmem_window_tokens || !request.allow_memory_reuse || !sequence.window.stamp.valid ||
        (!sequence.window.prefix_checkpoint && !sequence.window.endpoint_checkpoint &&
         !sequence.window.input_checkpoint && !sequence.window.reconstruction_checkpoint)) return;
    auto& window = sequence.window;
    if (window.session_key && !window.update_session_index) return;
    if (window.session_key) {
        for (const auto& publication : memory_publications) {
            if (publication.key == window.session_key && publication.order > window.publication_order) return;
        }
    }
    std::uint32_t frontier = 0;
    for (const auto kind : kKvmemCheckpointKinds)
        if (const auto* checkpoint = memory_checkpoint(window, kind))
            frontier = std::max(frontier, static_cast<std::uint32_t>(checkpoint->prefix.size()));
    const auto pages = kv_pages_for_tokens(frontier);
    if (!pages || pages > window.archive.size() || pages > window.host_versions.size() ||
        !window.archive.front()) return;
    const auto page_bytes = window.archive.front().size();
    // All slots share one archive through the latest complete checkpoint.
    // Uncommitted/cancelled rows past that frontier are never retained.
    window.archive.resize(pages);
    window.host_versions.resize(pages);
    window.payload_bytes = pages * page_bytes;
    std::optional<std::size_t> destination;
    std::optional<std::size_t> oldest;
    std::size_t count = 0;
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        auto& history = memory_histories[index];
        if (!history) { if (!destination) destination = index; continue; }
        ++count;
        if (window.session_key && history->session_key == window.session_key) {
            if (history->publication_order > window.publication_order) return;
            destination = index;
            break;
        }
        if (!oldest || history->publication_order < memory_histories[*oldest]->publication_order) oldest = index;
    }
    if (count >= kvmem_options.retained_sessions && (!destination || !memory_histories[*destination])) {
        destination = oldest;
        ++memory_history_evictions;
    }
    if (!destination) return;
    // Capture already completed every Host copy. The active state's later
    // writes and unpublished output never become this checkpoint's contents.
    for (auto& record : window.archive) record.seal();
    memory_histories[*destination].emplace(std::move(window));
    if (memory_io && memory_payload_storage.disk_bytes()) memory_idle_requested = true;

}

bool ProgramImpl::memory_maintenance_pending() const noexcept {
    return memory_io && (memory_idle_requested || memory_idle_transfer);
}

bool ProgramImpl::progress_memory_storage() {
    if (!memory_maintenance_pending()) return false;
    bool changed = false;
    const auto being_restored = [&](std::uint64_t generation) {
        const auto* transaction = std::get_if<MaterializationTransaction>(&context_transaction_);
        if (!transaction) return false;
        const auto& prefill = requests[transaction->destination.value].prefill;
        return prefill && prefill->memory_restore_frontier && prefill->memory_restore_generation == generation;
    };
    if (memory_idle_transfer) {
        auto& pending = *memory_idle_transfer;
        if (!pending.ticket.ready()) return false;
        try {
            pending.ticket.wait();
            auto& history = memory_histories.at(pending.history);
            if (history && history->stamp.session.id == pending.generation && !being_restored(pending.generation)) {
                auto& checkpoint = memory_checkpoint_slot(*history, pending.kind);
                if (checkpoint && checkpoint->state == pending.source && !checkpoint->cold_state.size()) {
                    checkpoint->cold_state = std::move(pending.work->destination);
                    checkpoint->state.reset();
                }
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "KVMEM_IDLE state spill retained RAM: %s\n", error.what());
            memory_idle_requested = false; // retry on a subsequent retention boundary
        }
        memory_idle_transfer.reset(); // pinned owner is released on this engine thread
        changed = true;
    }
    if (!memory_idle_requested || has_context_transaction()) return changed;
    memory_idle_requested = false;
    const auto disk = memory_payload_storage.spill_file();
    std::uint64_t used = 0;
    for (const auto& active : requests) used += active.memory_payload_reservation;
    for (const auto& history : memory_histories) {
        if (!history) continue;
        used += history->payload_bytes;
        for (const auto kind : kKvmemCheckpointKinds)
            if (const auto* checkpoint = memory_checkpoint(*history, kind)) used += checkpoint->cold_state.size();
    }
    // One background image at a time. Its original pinned owner is retained,
    // so there is no second full state-image allocation and no worker CUDA call.
    for (std::size_t index = 0; index < memory_histories.size(); ++index) {
        const auto& history = memory_histories[index];
        if (!history) continue;
        for (const auto kind : kKvmemCheckpointKinds) {
            const auto* checkpoint = memory_checkpoint(*history, kind);
            if (!checkpoint || !checkpoint->state || checkpoint->cold_state.size()) continue;
            const auto bytes = checkpoint->state->size();
            if (used > memory_retained_budget || bytes > memory_retained_budget - used ||
                bytes > disk->capacity_bytes() - disk->charged_bytes()) continue;
            try {
                auto pending = std::make_unique<KvmemIdleTransfer>();
                pending->history = index;
                pending->generation = history->stamp.session.id;
                pending->kind = kind;
                pending->source = checkpoint->state;
                pending->work = std::make_shared<KvmemIdleTransfer::Work>();
                pending->work->destination = disk->reserve(bytes);
                pending->work->data = pending->source->data();
                pending->work->bytes = bytes;
                pending->ticket = memory_io->submit(bytes, [work = pending->work] {
                    work->destination.write(0, work->data, work->bytes);
                });
                memory_idle_transfer = std::move(pending);
                memory_idle_requested = true;
                return true;
            } catch (const std::exception& error) {
                std::fprintf(stderr, "KVMEM_IDLE state spill retained RAM: %s\n", error.what());
                return changed;
            }
        }
    }
    return changed;
}

} // namespace ninfer::models::qwen3_5::detail
