#include "ninfer/engine.h"
#include "kvmem_test_options.h"
#include "kvmem/host_kv_storage.hpp"
#include "kvmem/spill_io.hpp"
#include "kvmem/memory_contract.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Replace only this test executable's ReadFile/WriteFile import. No production hook,
// environment variable, filesystem modification or other process is involved.
// One KV write receives the same error as a full Windows volume would return.
template<bool Reading = false> class FileFailure {
    using Pointer = std::conditional_t<Reading, LPVOID, LPCVOID>;
    using Write = BOOL(WINAPI*)(HANDLE, Pointer, DWORD, LPDWORD, LPOVERLAPPED);
    inline static Write original = nullptr;
    inline static std::atomic<bool> armed{false};
    inline static std::atomic<unsigned> failures{0};
    inline static std::mutex gate_mutex;
    inline static std::condition_variable gate_changed;
    inline static bool hold = false, entered = false, released = false;
    ULONG_PTR* slot = nullptr;
    static BOOL WINAPI write(HANDLE file, Pointer data, DWORD size, LPDWORD done, LPOVERLAPPED op) {
        if (armed.load()) {
            wchar_t path[2048]{};
            const auto count = GetFinalPathNameByHandleW(file, path, 2048, FILE_NAME_NORMALIZED);
            if (count && count < 2048 && std::wcsstr(path, L"kvmem-spill-") && armed.exchange(false)) {
                std::unique_lock<std::mutex> lock(gate_mutex);
                if (hold) {
                    entered = true;
                    gate_changed.notify_all();
                    gate_changed.wait(lock, [] { return released; });
                    lock.unlock();
                    return original(file, data, size, done, op);
                }
                ++failures;
                SetLastError(Reading ? ERROR_READ_FAULT : ERROR_DISK_FULL);
                return FALSE;
            }
        }
        return original(file, data, size, done, op);
    }
    void replace(ULONG_PTR value) {
        DWORD previous = 0;
        require(VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &previous) != 0, "cannot change test import");
        *slot = value;
        DWORD ignored = 0;
        require(VirtualProtect(slot, sizeof(*slot), previous, &ignored) != 0, "cannot restore test import protection");
    }
public:
    FileFailure() {
        auto* base = reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        auto* imports = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base +
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress);
        for (; imports->Name && !slot; ++imports) {
            if (!imports->OriginalFirstThunk) continue;
            auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->OriginalFirstThunk);
            auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->FirstThunk);
            for (; names->u1.AddressOfData; ++names, ++addresses) {
                if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
                auto* name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
                if (std::strcmp(reinterpret_cast<const char*>(name->Name), Reading ? "ReadFile" : "WriteFile") == 0) {
                    slot = &addresses->u1.Function;
                    original = reinterpret_cast<Write>(*slot);
                    break;
                }
            }
        }
        require(slot, "test executable does not import required file operation");
        replace(reinterpret_cast<ULONG_PTR>(&write));
    }
    ~FileFailure() { armed = false; release(); if (slot) replace(reinterpret_cast<ULONG_PTR>(original)); }
    void arm(bool block = false) {
        std::lock_guard<std::mutex> lock(gate_mutex);
        hold = block; entered = released = false; armed = true;
    }
    bool wait_entered() {
        std::unique_lock<std::mutex> lock(gate_mutex);
        return gate_changed.wait_for(lock, std::chrono::seconds(30), [] { return entered; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(gate_mutex);
        released = true; gate_changed.notify_all();
    }
    unsigned count() const { return failures.load(); }
};
using WriteFailure = FileFailure<false>;

void cold_rewrite_peer_progress() {
    kvmem::HostKvStorage storage(32, std::make_shared<kvmem::SpillFile>(
        std::filesystem::current_path() / "promotion-spill", 128));
    const unsigned char expected[8] = {1,2,3,4,5,6,7,8};
    auto cold = storage.allocate(8, expected);
    cold.seal(); storage.trim(0);
    FileFailure<true> gate;
    gate.arm(true);
    auto rewrite = std::async(std::launch::async, [&] { cold.writable_data()[0] = 9; });
    const bool entered = gate.wait_entered();
    auto peer = std::async(std::launch::async, [&] { return storage.allocate(8); });
    const bool progressed = peer.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    gate.release(); // Release before assertions/future destruction even on failure.
    rewrite.get(); auto resident = peer.get();
    require(entered && progressed, "cold rewrite blocked an independent RAM reservation");
    cold.seal(); storage.trim(0);
    gate.arm();
    bool failed = false;
    try { (void)cold.writable_data(); }
    catch (const kvmem::MemoryError& error) { failed = error.code() == kvmem::MemoryErrorCode::TransferFailed; }
    require(failed && cold.resident_bytes() == 0 && storage.charged_bytes() == 8,
            "failed promotion discarded disk or leaked reserved RAM");
    unsigned char restored[8]{};
    cold.read(0, restored, sizeof(restored));
    require(restored[0] == 9 && std::memcmp(restored+1, expected+1, 7) == 0,
            "failed promotion corrupted the only valid copy");
    std::cout << "COLD_REWRITE peer progress and failed-read rollback PASS" << std::endl;
}
struct Sink final : ninfer::OutputSink {
    std::promise<void> first;
    bool notified = false;
    std::atomic<unsigned> generated{0};
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void publish(ninfer::OutputDelta) override {}
    void timing(ninfer::GenerationTimingObservation value) override {
        generated = value.generated_tokens;
        if (value.generated_tokens && !notified) { notified = true; first.set_value(); }
    }
};
ninfer::PromptInput prompt(bool cold) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.context_cache.session_key = cold ? "cold-failure" : "hot-peer";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    std::string text;
    if (cold) for (int i = 0; i < 160; ++i)
        text += "The observatory archives pulsars, cosmic rays and distant galaxies. ";
    text += "List 100 numbered facts about astronomy.";
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
    return input;
}
ninfer::RequestOptions request(unsigned output) {
    ninfer::RequestOptions value;
    value.execution.requested_output_tokens = output;
    value.execution.sampling.temperature = 0;
    value.stop.include_model_defaults = false;
    return value;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "artifact");
        cold_rewrite_peer_progress();
        const auto promotion_path = std::filesystem::current_path() / "promotion-spill";
        require(std::filesystem::is_empty(promotion_path), "promotion leaked its spill file");
        std::filesystem::remove(promotion_path);
        ninfer::EngineOptions options;
        ninfer::test::kvmem_test_devices(options);
        options.artifact_path = argv[1];
        options.device_profile = "off";
        options.max_context = 8192;
        options.max_concurrency = 2;
        options.prefill_chunk = 128;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
        options.kvmem = {.selected_tokens = 384, .reserve_tokens = 128,
            .host_bytes = 512ULL << 20, .retained_sessions = 2, .verify_transfers = true};
        options.context_cache.enabled = false;
        std::vector<std::int32_t> reference;
        {
            ninfer::Engine engine(options);
            const auto result = engine.submit(engine.prepare(prompt(false)), request(256)).wait();
            reference.assign(result.generated_token_ids.begin(), result.generated_token_ids.end());
        }
        options.kvmem.host_bytes = 48ULL << 20;
        options.kvmem.disk_bytes = 1024ULL << 20;
        options.kvmem.disk_path = std::filesystem::temp_directory_path() /
            ("ninfer-ssd-failure-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        WriteFailure failure;
        {
            ninfer::Engine engine(options);
            Sink sink;
            auto first = sink.first.get_future();
            auto hot = engine.submit(engine.prepare(prompt(false)), request(256),
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
            auto peer = std::async(std::launch::async, [hot = std::move(hot), &sink]() mutable { return hot.wait(&sink); });
            require(first.wait_for(std::chrono::seconds(60)) == std::future_status::ready, "hot peer never started");
            failure.arm();
            bool failed = false;
            try { (void)engine.submit(engine.prepare(prompt(true)), request(32)).wait(); }
            catch (const std::exception& error) {
                failed = true;
                std::cout << "EXPECTED_STORAGE_FAILURE " << error.what() << std::endl;
            }
            require(failed && failure.count() == 1, "cold lane did not receive the injected write failure");
            const auto survived = peer.get();
            require(std::vector<std::int32_t>(survived.generated_token_ids.begin(), survived.generated_token_ids.end()) == reference,
                    "storage failure changed the hot peer output");
            require(engine.is_available(), "one storage failure disabled the Engine");
            const auto recovered = engine.submit(engine.prepare(prompt(true)), request(32)).wait();
            require(recovered.generated_token_ids.size() == 32 && engine.is_available(), "retry after storage failure did not complete");
            const auto memory = engine.memory_summary().kvmem;
            require(memory.disk_errors == 1 && memory.disk_written_bytes && memory.host_payload_bytes <= options.kvmem.host_bytes &&
                    memory.disk_payload_bytes <= options.kvmem.disk_bytes && memory.resident_pages == 0,
                    "failure recovery leaked quota/pages or lost I/O accounting");
            std::cout << "SSD_FAILURE errors=" << memory.disk_errors << " peer_exact=1 retry=1" << std::endl;
        }
        {
            ninfer::Engine engine(options);
            Sink sink;
            auto first = sink.first.get_future();
            auto hot = engine.submit(engine.prepare(prompt(false)), request(512),
                ninfer::OutputConsumerMode::Streaming, {.live_timings = true});
            auto peer = std::async(std::launch::async, [hot = std::move(hot), &sink]() mutable { return hot.wait(&sink); });
            require(first.wait_for(std::chrono::seconds(60)) == std::future_status::ready, "cancellation peer never started");
            failure.arm(true);
            std::atomic<bool> cancel{false};
            auto cold = engine.submit(engine.prepare(prompt(true)), request(32));
            auto cancelled = std::async(std::launch::async, [cold = std::move(cold), &cancel]() mutable {
                return cold.wait(nullptr, ninfer::CancellationView([&] { return cancel.load(); }));
            });
            const bool blocked = failure.wait_entered();
            const auto before = sink.generated.load();
            cancel = true;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (sink.generated.load() < before + 8 && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const auto advanced = sink.generated.load() - before;
            failure.release(); // always unblock the worker before any assertion/cleanup
            require(blocked && advanced >= 8, "cancelled slow storage job blocked the hot peer");
            require(cancelled.get().finish_reason == ninfer::FinishReason::Cancelled,
                    "cancelled storage lane did not settle as cancelled");
            require(peer.get().generated_token_ids.size() == 512 && engine.is_available(),
                    "storage cancellation damaged the peer or Engine");
            std::cout << "SSD_CANCEL peer_tokens_while_io_blocked=" << advanced << " PASS" << std::endl;
        }
        require(std::filesystem::is_empty(options.kvmem.disk_path), "failure path leaked its spill file");
        std::filesystem::remove(options.kvmem.disk_path);
        std::cout << "SSD_FAILURE PASS" << std::endl;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
#else
int main() { std::cout << "Windows import fault injection requires Windows; Linux OS failure coverage is in spill_io_test\n"; return 77; }
#endif
