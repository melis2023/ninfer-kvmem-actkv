#pragma once

#include "artifact/framing.h"
#include "artifact/schema.h"
#include "core/weight_view.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::artifact {

struct ReadSegment {
    std::size_t file_index           = 0;
    std::uint64_t file_offset        = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

// A directory another container has already laid out. `directory` addresses bytes that begin at
// `data_offset` in `path`, and `derived` holds everything the directory names past the file's own
// payload, which the reader serves from Host memory instead of from a file. A drafter GGUF opens
// this way (models/draft_gguf.cpp): its tensors are the file's own bytes, and the tensors its
// consumers read in a representation a GGUF cannot store are derived once, at open, into
// `derived`. `source_alignment` is the container's object alignment, which a format's own
// alignment may not divide.
struct PreparedPayload {
    Directory directory;
    std::uint64_t data_offset      = 0;
    std::uint64_t source_alignment = 32;
    ArtifactId artifact_id{};
    std::vector<std::byte> derived;
};

// Owns the cold directory and lazily opened files. Runtime views borrow materialized storage,
// never this Reader. Encodings are interpreted only for requested objects.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path);
    Reader(const std::filesystem::path& path, PreparedPayload payload);
    ~Reader();
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    // Whether the whole range belongs to the derived block rather than to a file. A caller that
    // transfers payload directly (artifact/materializer.cpp) reads those through the Host instead.
    [[nodiscard]] bool derived(std::uint64_t offset, std::uint64_t bytes) const;

    [[nodiscard]] const Directory& directory() const noexcept;
    [[nodiscard]] const ArtifactId& artifact_id() const noexcept;
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;
    [[nodiscard]] ObjectHandle find(std::string_view id) const;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const;
    void validate_object(ObjectHandle handle) const;

    [[nodiscard]] std::vector<ReadSegment> segments(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read_range(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    [[nodiscard]] std::vector<std::byte> read_object(ObjectHandle handle) const;
    [[nodiscard]] std::size_t read_direct(std::size_t file_index, std::uint64_t file_offset,
                                          std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
