#pragma once

// Serving-only, bounded metadata. Never stores prompts, output text or credentials.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <string>

namespace kvmem {
class monitor {
    using clock = std::chrono::steady_clock;
    struct entry { nlohmann::json data; clock::time_point started; };
    mutable std::mutex mutex_;
    std::map<std::string, entry> active_;
    std::deque<entry> recent_;
    uint64_t sequence_ = 0, dropped_ = 0;
    const std::string instance_ = std::to_string(std::random_device{}()) + "-" + std::to_string(now_ms());
public:
    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    std::string begin(const std::string & endpoint, const std::string & key = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto id = key.empty() ? std::to_string(++sequence_) : key;
        // Preparation can register a keyed request before execution starts.
        // Keep its original clock and input metadata when execution adopts it.
        if (!key.empty() && active_.count(id)) return id;
        if (active_.size() >= 512) { ++dropped_; return ""; }
        active_[id] = {{{"id", id}, {"endpoint", endpoint}, {"started_at_ms", now_ms()},
                        {"status", "processing"}, {"prompt_tokens", nullptr}, {"cached_tokens", nullptr},
                        {"generated_tokens", nullptr}, {"limit", nullptr}, {"prefill_tps", nullptr},
                        {"decode_tps", nullptr}, {"elapsed_ms", 0}}, clock::now()};
        return id;
    }
    void update(const std::string & id, const nlohmann::json & values) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_.find(id);
        if (it != active_.end()) it->second.data.update(values);
    }
    // Also supports transport failure after generation completed, without duplicating history.
    void finish(const std::string & id, const std::string & status) {
        if (id.empty()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto set_status = [&](nlohmann::json& data) {
            data["status"] = status;
            if (status == "error") data["finish_reason"] = "error";
            else if (status == "disconnected") data["finish_reason"] = "disconnect";
        };
        auto it = active_.find(id);
        if (it == active_.end()) {
            if (status != "completed") for (auto & e : recent_)
                if (e.data["id"] == id) { set_status(e.data); break; }
            return;
        }
        auto e = std::move(it->second);
        active_.erase(it);
        set_status(e.data);
        e.data["elapsed_ms"] = std::chrono::duration<double, std::milli>(clock::now() - e.started).count();
        recent_.push_front(std::move(e));
        if (recent_.size() > 100) recent_.pop_back();
    }
    nlohmann::json snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto running = nlohmann::json::array(), recent = nlohmann::json::array();
        for (const auto & pair : active_) {
            auto data = pair.second.data;
            data["elapsed_ms"] = std::chrono::duration<double, std::milli>(clock::now() - pair.second.started).count();
            running.push_back(std::move(data));
        }
        for (const auto & e : recent_) recent.push_back(e.data);
        return {{"schema_version", 1}, {"server_instance", instance_}, {"sampled_at_ms", now_ms()},
                {"active", running}, {"recent", recent}, {"dropped_records", dropped_}};
    }
};
} // namespace kvmem
