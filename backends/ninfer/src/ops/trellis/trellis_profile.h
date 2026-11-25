#pragma once

// Opt-in attribution for the coded path. NINFER_PROFILE=1 records, per named scope, the device
// time between two events on the calling stream and the host time the scope's caller spent
// issuing it, and prints the table when the process exits. Unset, a scope costs one relaxed load.
//
// Device times are read back lazily: an event pair is reused once an older pair has certainly
// completed, so the profiled stream is never synchronised and nothing is created or waited on
// while a CUDA graph is being captured (those scopes are counted on the host only).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::ops::detail::trellis::profile {

inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

struct Row {
    double device_ms = 0;
    double host_ms   = 0;
    std::uint64_t calls = 0;
};

struct Table {
    std::mutex mutex;
    std::map<std::string, Row> rows;

    struct Pending {
        cudaEvent_t begin;
        cudaEvent_t end;
        std::string key;
        double host_ms;
    };
    std::deque<Pending> pending;
    std::vector<cudaEvent_t> free_events;
    std::uint64_t unread = 0;  // scopes dropped because their events could not be created

    static Table& instance() {
        static Table table;
        return table;
    }

    cudaEvent_t acquire(cudaStream_t stream) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
            status != cudaStreamCaptureStatusNone) {
            return nullptr;
        }
        if (!free_events.empty()) {
            const cudaEvent_t event = free_events.back();
            free_events.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        if (cudaEventCreate(&event) != cudaSuccess) { return nullptr; }
        return event;
    }

    void add_host(const std::string& key, double host_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        Row& row        = rows[key];
        row.host_ms    += host_ms;
        row.calls      += 1;
    }

    void submit(const std::string& key, cudaEvent_t begin, cudaEvent_t end, double host_ms) {
        std::lock_guard<std::mutex> lock(mutex);
        pending.push_back(Pending{begin, end, key, host_ms});
        // Older pairs have long completed by the time the queue is this deep; reading them here
        // keeps the queue bounded without a stream synchronisation.
        if (pending.size() > 256) { drain(128); }
    }

    void drain(std::size_t count) {
        std::size_t done = 0;
        while (!pending.empty() && done < count) {
            const Pending item = pending.front();
            pending.pop_front();
            float ms = 0.0F;
            if (cudaEventSynchronize(item.end) == cudaSuccess &&
                cudaEventElapsedTime(&ms, item.begin, item.end) == cudaSuccess) {
                Row& row     = rows[item.key];
                row.device_ms += static_cast<double>(ms);
                row.host_ms   += item.host_ms;
                row.calls     += 1;
            }
            free_events.push_back(item.begin);
            free_events.push_back(item.end);
            ++done;
        }
    }

    ~Table() {
        drain(pending.size());
        std::fprintf(stderr,
                     "\nprofile | scope | calls | device ms | host ms | device/call us | "
                     "host/call us | GB/s-equiv\n");
        std::vector<const std::pair<const std::string, Row>*> order;
        for (const auto& item : rows) { order.push_back(&item); }
        std::sort(order.begin(), order.end(), [](const auto* a, const auto* b) {
            return a->second.device_ms > b->second.device_ms;
        });
        double device_total = 0, host_total = 0;
        for (const auto* item : order) {
            const Row& row = item->second;
            device_total += row.device_ms;
            host_total += row.host_ms;
            if (row.calls == 0) { continue; }
            std::fprintf(stderr, "profile | %s | %llu | %.3f | %.3f | %.1f | %.1f\n",
                         item->first.c_str(), static_cast<unsigned long long>(row.calls),
                         row.device_ms, row.host_ms, row.device_ms * 1000.0 / double(row.calls),
                         row.host_ms * 1000.0 / double(row.calls));
        }
        std::fprintf(stderr, "profile | TOTAL | %llu | %.3f | %.3f\n",
                     static_cast<unsigned long long>(unread), device_total, host_total);
    }
};

class Scope {
public:
    Scope(const std::string& key, cudaStream_t stream) : stream_(stream), key_(key), start_() {
        if (!enabled()) { return; }
        start_ = std::chrono::steady_clock::now();
        begin_ = Table::instance().acquire(stream);
        if (begin_ == nullptr) { return; }
        end_ = Table::instance().acquire(stream);
        if (end_ == nullptr) {
            Table::instance().free_events.push_back(begin_);
            begin_ = nullptr;
            return;
        }
        cudaEventRecord(begin_, stream_);
    }

    ~Scope() {
        if (!enabled()) { return; }
        const double host =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                .count();
        if (begin_ == nullptr) {
            Table::instance().add_host(key_, host);
            return;
        }
        cudaEventRecord(end_, stream_);
        Table::instance().submit(key_, begin_, end_, host);
    }

    Scope(const Scope&)            = delete;
    Scope& operator=(const Scope&) = delete;

private:
    cudaStream_t stream_ = nullptr;
    std::string key_;
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t end_   = nullptr;
    std::chrono::steady_clock::time_point start_;
};

} // namespace ninfer::ops::detail::trellis::profile
