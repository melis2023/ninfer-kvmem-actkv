#include "artifact/reader.h"

#include "artifact/file_io.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace ninfer::artifact {

struct Reader::Impl {
    std::filesystem::path entry;
    Directory directory;
    ArtifactId id{};
    std::uint64_t entry_payload_start = 0;
    std::uint64_t file_bytes          = 0;
    // Payload the files hold. Everything past it is `derived`, which no file describes.
    std::uint64_t file_payload_bytes = 0;
    std::uint64_t source_alignment   = 256;
    std::vector<std::byte> derived;
    mutable std::vector<std::unique_ptr<InputFile>> files;
    mutable std::vector<std::optional<WeightGeometry>> geometries;

    explicit Impl(const std::filesystem::path& path) : entry(path) {
        auto file = std::make_unique<InputFile>(entry);
        std::array<std::byte, kHeaderBytes> header{};
        file->read_exact(0, header);
        if (!std::equal(kEntryMagic.begin(), kEntryMagic.end(), header.begin())) {
            if (std::equal(kEntryMagic.begin(), kEntryMagic.end() - 1, header.begin()) &&
                header[kEntryMagic.size() - 1] == std::byte{2}) {
                throw ArtifactError(
                    entry.string() +
                    ": NInfer v2 artifact is not supported. Upgrade to v3 with: "
                    "python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer");
            }
            throw ArtifactError(entry.string() + ": expected NInfer v3 entry magic");
        }
        const auto json_bytes = read_u64_le(header.data() + 8);
        if (!json_bytes || json_bytes > file->bytes() - kHeaderBytes ||
            json_bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError(entry.string() + ": invalid directory length");
        }
        std::copy_n(header.begin() + 16, id.size(), id.begin());
        std::string text(static_cast<std::size_t>(json_bytes), '\0');
        file->read_exact(kHeaderBytes, std::as_writable_bytes(std::span(text.data(), text.size())));
        directory =
            parse_directory(parse_json(text, "artifact directory"), entry.filename().string());
        entry_payload_start = align_up(checked_add(kHeaderBytes, json_bytes, "metadata end"),
                                       kPayloadAlignment, "entry payload start");
        const auto expected =
            checked_add(entry_payload_start, directory.files[0].payload_bytes, "entry length");
        if (expected != file->bytes()) {
            throw ArtifactError(entry.string() + ": entry length differs from directory");
        }
        file_bytes = checked_add(entry_payload_start, directory.payload_bytes, "file bytes");
        file_bytes = checked_add(
            file_bytes,
            checked_mul(directory.files.size() - 1, kPayloadAlignment, "continuation headers"),
            "file bytes");
        files.resize(directory.files.size());
        files[0] = std::move(file);
        geometries.resize(directory.objects.size());
        file_payload_bytes = directory.payload_bytes;
    }

    Impl(const std::filesystem::path& path, PreparedPayload payload)
        : entry(path), directory(std::move(payload.directory)), id(payload.artifact_id),
          entry_payload_start(payload.data_offset), source_alignment(payload.source_alignment),
          derived(std::move(payload.derived)) {
        if (!payload.source_alignment) { throw ArtifactError(entry.string() + ": zero alignment"); }
        if (directory.files.size() != 1 || directory.files[0].path) {
            throw ArtifactError(entry.string() + ": a prepared payload is one file");
        }
        auto file        = std::make_unique<InputFile>(entry);
        file_bytes       = file->bytes();
        file_payload_bytes = directory.files[0].payload_bytes;
        const auto held    = checked_add(entry_payload_start, file_payload_bytes, "file payload");
        if (file_payload_bytes == 0 || held > file_bytes) {
            throw ArtifactError(entry.string() + ": prepared payload exceeds the file");
        }
        if (checked_add(file_payload_bytes, derived.size(), "payload bytes") !=
            directory.payload_bytes) {
            throw ArtifactError(entry.string() + ": prepared payload length differs from directory");
        }
        files.resize(1);
        files[0] = std::move(file);
        geometries.resize(directory.objects.size());
    }

    InputFile& file(std::size_t index) const {
        if (index >= files.size()) { throw ArtifactError("invalid continuation index"); }
        if (!files[index]) {
            const auto& record = directory.files[index];
            auto input         = std::make_unique<InputFile>(entry.parent_path() / *record.path);
            std::array<std::byte, kHeaderBytes> header{};
            input->read_exact(0, header);
            if (!std::equal(kPartMagic.begin(), kPartMagic.end(), header.begin()) ||
                read_u64_le(header.data() + 8) != index ||
                !std::equal(id.begin(), id.end(), header.begin() + 16)) {
                throw ArtifactError(*record.path +
                                    ": continuation header does not belong to this entry");
            }
            if (input->bytes() !=
                checked_add(kPayloadAlignment, record.payload_bytes, "part length")) {
                throw ArtifactError(*record.path + ": continuation length differs from directory");
            }
            files[index] = std::move(input);
        }
        return *files[index];
    }
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::Reader(const std::filesystem::path& path, PreparedPayload payload)
    : impl_(std::make_unique<Impl>(path, std::move(payload))) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const Directory& Reader::directory() const noexcept { return impl_->directory; }

const ArtifactId& Reader::artifact_id() const noexcept { return impl_->id; }

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file_bytes; }

bool Reader::derived(std::uint64_t offset, std::uint64_t bytes) const {
    return bytes != 0 && offset >= impl_->file_payload_bytes;
}

ObjectHandle Reader::find(std::string_view id) const {
    const auto found = directory().object_index.find(id);
    if (found == directory().object_index.end()) {
        throw ArtifactError("missing object " + std::string(id));
    }
    return found->second;
}

const WeightGeometry& Reader::geometry(ObjectHandle handle) const {
    const auto& tensor = directory().tensor(handle);
    auto& result       = impl_->geometries[handle.index];
    if (!result) { result = describe_tensor(tensor, impl_->source_alignment); }
    return *result;
}

void Reader::validate_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (std::holds_alternative<TensorObject>(object)) {
        (void)geometry(handle);
    } else if (std::get<ResourceObject>(object).encoding != kRawBytesEncoding) {
        throw ArtifactError(object_id(object) + ": unsupported resource encoding");
    }
}

std::vector<ReadSegment> Reader::segments(std::uint64_t offset, std::uint64_t bytes) const {
    const auto& data = directory();
    if (offset > impl_->file_payload_bytes || bytes > impl_->file_payload_bytes - offset) {
        throw ArtifactError("read range exceeds the payload the files hold");
    }
    std::vector<ReadSegment> result;
    if (!bytes) { return result; }
    const auto found     = std::upper_bound(data.files.begin(), data.files.end(), offset,
                                            [](std::uint64_t position, const FileRecord& file) {
                                            return position < file.logical_begin;
                                        });
    std::size_t index    = static_cast<std::size_t>(found - data.files.begin() - 1);
    std::uint64_t copied = 0;
    while (copied < bytes) {
        const auto& file = data.files[index];
        const auto local = offset + copied - file.logical_begin;
        const auto count = std::min(bytes - copied, file.payload_bytes - local);
        const auto start = index == 0 ? impl_->entry_payload_start : kPayloadAlignment;
        result.push_back({index, checked_add(start, local, "file offset"), copied, count});
        copied += count;
        ++index;
    }
    return result;
}

void Reader::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    if (destination.empty()) { return; }
    const auto derived_at = impl_->derived.data();
    // The derived block follows the payload the files hold, so a range either sits wholly in one
    // of them or straddles the boundary between the two.
    if (offset + destination.size() > impl_->file_payload_bytes) {
        const std::uint64_t local = offset > impl_->file_payload_bytes
                                        ? offset - impl_->file_payload_bytes
                                        : 0;
        const std::size_t from_file =
            offset >= impl_->file_payload_bytes
                ? 0
                : static_cast<std::size_t>(impl_->file_payload_bytes - offset);
        if (from_file) {
            for (const auto& segment : segments(offset, from_file)) {
                impl_->file(segment.file_index)
                    .read_exact(segment.file_offset,
                                destination.subspan(
                                    static_cast<std::size_t>(segment.destination_offset),
                                    static_cast<std::size_t>(segment.bytes)));
            }
        }
        if (local > impl_->derived.size() ||
            destination.size() - from_file > impl_->derived.size() - local) {
            throw ArtifactError("read range exceeds the derived payload");
        }
        std::memcpy(destination.data() + from_file, derived_at + local,
                    destination.size() - from_file);
        return;
    }
    for (const auto& segment : segments(offset, destination.size())) {
        impl_->file(segment.file_index)
            .read_exact(segment.file_offset,
                        destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                            static_cast<std::size_t>(segment.bytes)));
    }
}

std::vector<std::byte> Reader::read_range(std::uint64_t offset, std::uint64_t bytes) const {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host read size exceeds size_t");
    }
    const auto& data = directory();
    if (offset > data.payload_bytes || bytes > data.payload_bytes - offset) {
        throw ArtifactError("read range exceeds logical payload");
    }
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    read_into(offset, result);
    return result;
}

std::vector<std::byte> Reader::read_object(ObjectHandle handle) const {
    validate_object(handle);
    const auto& object = directory().object(handle);
    return read_range(object_offset(object), object_bytes(object));
}

std::size_t Reader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                std::span<std::byte> destination) const {
    return impl_->file(file_index).read_direct(file_offset, destination);
}

} // namespace ninfer::artifact
