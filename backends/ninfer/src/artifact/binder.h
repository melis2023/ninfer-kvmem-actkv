#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::artifact {

// Pinned: the object lives only in one page-locked Host block laid out by the plan (for weights
// streamed to the device on demand); it never receives a device placement.
enum class Residency { Device, Host, Values, Pinned };

struct ParameterReference {
    std::string name;
    Shape shape;
    Binding binding;
    Residency residency = Residency::Device;
    // Explicit replica selected by a stage-local auxiliary binding; ordinary weights use their
    // sole placement as before.
    std::optional<std::size_t> device_replica;
};

struct HostValues {
    QType format           = QType::FP32;
    std::uint64_t elements = 0;
    std::vector<std::byte> data;

    [[nodiscard]] float scalar_f32() const;
    [[nodiscard]] std::vector<std::int32_t> integers() const;
};

// Collects selected logical demands; neither physical object IDs nor whole-artifact profiles
// determine parameter shapes or native operation support.
class Binder {
public:
    explicit Binder(const Reader& reader);

    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               Residency residency = Residency::Device,
                                               std::optional<QType> exact_format = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             Residency residency,
                                             std::optional<QType> exact_format = {});
    [[nodiscard]] const Use& use(std::string_view parameter, std::string_view input) const;
    [[nodiscard]] bool contains(std::string_view parameter) const;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    void require_device(ObjectHandle object, std::uint64_t alignment = 256);
    // Reserves `bytes` of zeros right after a device object, which the load writes: a kernel that
    // reads whole K steps may read past the end of a matrix whose rows are not a whole number of
    // them (ggml's matrix kernel reads up to one 256-value step past such a row). The object must
    // already have a device demand; a repeated request keeps the largest.
    void device_tail(ObjectHandle object, std::uint64_t bytes);
    // Materialize a device object in a narrower row-split format than the artifact stores. The
    // object must already have a device demand and be a row-split Q8_G32_FP16 tensor, and every
    // consumer must accept the target format; a repeated request must name the same target.
    void transcode_device(ObjectHandle object, QType target);
    // Moves a device object into the arena's evictable tail. Ranked objects follow every resident
    // object; ascending rank places the highest rank at the arena end, which an eviction pool
    // borrows first. A repeated request keeps the highest rank.
    void evict_device(ObjectHandle object, std::uint32_t rank);
    // Materializes a device object in another pipeline rank's arena, on that rank's device. The
    // object must already have a device demand, and a repeated request must name the same rank.
    //
    // This is what a `--devices` expert offload is made of: the placement mechanism is the plan's,
    // so the artifact is still validated in full and every consumer keeps reading one
    // MaterializedArtifact. Rank 0 is the default and the only rank a single-device load uses.
    void device_rank(ObjectHandle object, std::size_t rank);
    // Shared immutable auxiliaries may need a local copy on multiple pipeline devices.
    void replicate_device(ObjectHandle object, std::size_t rank);
    // Places an object in the pinned Host block, in first-request order so a caller's logical
    // groups stay contiguous. It excludes device and Host placements of the same object.
    void require_pinned(ObjectHandle object);
    [[nodiscard]] std::span<const std::byte> host_object(ObjectHandle object);
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    // evictable_alignment aligns the start of the evictable tail (an eviction pool's chunk size)
    // so borrowing whole chunks never touches a resident object.
    [[nodiscard]] MaterializationPlan finish(std::uint64_t evictable_alignment = 1) &&;

private:
    struct Demand {
        bool device             = false;
        bool host               = false;
        std::uint64_t alignment = 256;
        std::optional<QType> transcode;
        std::uint32_t evict_rank = 0;
        std::uint64_t tail       = 0;
        std::optional<std::size_t> device_rank;
        std::vector<std::size_t> device_replicas;
        std::optional<std::uint64_t> pinned_order;
        std::vector<std::byte> host_data;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
    std::uint64_t next_pinned_order_ = 0;
};

} // namespace ninfer::artifact
