#pragma once

// NINFER_TOKEN_DUMP=<file>: append every proposal argmax the MTP head emits, so a
// converter can build the proposal shortlist from the tokens the drafter actually
// reaches for. The file opens on the first dump and flushes on every add; unset,
// it costs one relaxed load.

#include <cstdio>
#include <cstdlib>

namespace ninfer::models::qwen3_5 {

inline bool token_dump_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TOKEN_DUMP");
        return v != nullptr && v[0] != '\0';
    }();
    return on;
}

class TokenDump {
public:
    static void add(const std::int32_t* ids, std::size_t count) {
        if (!token_dump_enabled()) { return; }
        TokenDump& dump = instance();
        dump.append(ids, count);
    }

private:
    static TokenDump& instance() {
        static TokenDump dump;
        return dump;
    }

    void append(const std::int32_t* ids, std::size_t count) {
        if (file_ == nullptr) {
            const char* path = std::getenv("NINFER_TOKEN_DUMP");
            if (path == nullptr) { return; }
            file_ = std::fopen(path, "ab");
            if (file_ == nullptr) { return; }
        }
        std::fwrite(ids, sizeof(std::int32_t), count, file_);
        std::fflush(file_);
    }

    std::FILE* file_ = nullptr;

public:
    ~TokenDump() {
        if (file_ != nullptr) { std::fclose(file_); }
    }
};

} // namespace ninfer::models::qwen3_5
