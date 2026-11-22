#include "models/draft_gguf.h"

#include "artifact/file_io.h"
#include "artifact/formats.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/draft_artifact.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::models {
namespace {

using artifact::ArtifactError;
using artifact::Binding;
using artifact::Directory;
using artifact::Json;
using artifact::ObjectHandle;
using artifact::Shape;
using artifact::Use;

constexpr std::uint32_t kGgufVersion = 3;
constexpr std::array<std::byte, 4> kGgufMagic = {
    std::byte{'G'}, std::byte{'G'}, std::byte{'U'}, std::byte{'F'}};

// ggml's value type ids (ggml.h).
enum : std::uint32_t {
    kU8     = 0,
    kI8     = 1,
    kU16    = 2,
    kI16    = 3,
    kU32    = 4,
    kI32    = 5,
    kF32    = 6,
    kBool   = 7,
    kString = 8,
    kArray  = 9,
    kU64    = 10,
    kI64    = 11,
    kF64    = 12,
};

// ggml's tensor type ids (ggml.h) and the artifact format that stores the same blocks. F16 is
// absent because the artifact has no unquantised half format, and nothing a drafter stores whole
// is written that way.
struct Format {
    std::uint32_t ggml = 0;
    QType type         = QType::BF16;
};

constexpr Format kFormats[] = {
    {0, QType::FP32},          {30, QType::BF16},      {2, QType::GGUF_Q4_0},
    {6, QType::GGUF_Q5_0},     {8, QType::GGUF_Q8_0},  {10, QType::GGUF_Q2_K},
    {11, QType::GGUF_Q3_K},    {12, QType::GGUF_Q4_K}, {13, QType::GGUF_Q5_K},
    {14, QType::GGUF_Q6_K},    {16, QType::GGUF_IQ2_XXS}, {17, QType::GGUF_IQ2_XS},
    {18, QType::GGUF_IQ3_XXS}, {19, QType::GGUF_IQ1_S},    {20, QType::GGUF_IQ4_NL},
    {21, QType::GGUF_IQ3_S},   {22, QType::GGUF_IQ2_S},    {23, QType::GGUF_IQ4_XS},
    {29, QType::GGUF_IQ1_M},   {42, QType::GGUF_Q2_0},
};

QType format_of(std::uint32_t ggml, const std::filesystem::path& path, std::string_view name) {
    for (const auto& format : kFormats) {
        if (format.ggml == ggml) { return format.type; }
    }
    throw ArtifactError(path.string() + ": " + std::string(name) + " is stored as ggml type " +
                        std::to_string(ggml) + ", which no artifact format carries");
}

constexpr QuantLayout layout_of(QType type) {
    return is_gguf(type) ? QuantLayout::GgufBlocks : QuantLayout::Contiguous;
}

// The file's prefix, grown on demand. A GGUF's key/value table and tensor directory precede its
// data section, and a tokenizer's arrays are megabytes, so the window doubles rather than reading
// the header twice.
class Prefix {
public:
    explicit Prefix(const artifact::InputFile& file) : file_(file) {}

    [[nodiscard]] std::span<const std::byte> at(std::uint64_t offset, std::uint64_t size) {
        if (size == 0) { return {}; }
        if (offset <= bytes_.size() && size <= bytes_.size() - offset) {
            return {bytes_.data() + offset, static_cast<std::size_t>(size)};
        }
        const auto needed = artifact::checked_add(offset, size, "GGUF header window");
        if (needed > file_.bytes()) { throw ArtifactError("GGUF header runs past the file's end"); }
        const auto target =
            std::min<std::uint64_t>(file_.bytes(),
                                    std::max<uint64_t>({needed, bytes_.size() * 2, 1U << 20}));
        bytes_.resize(static_cast<std::size_t>(target));
        file_.read_exact(0, bytes_);
        return {bytes_.data() + offset, static_cast<std::size_t>(size)};
    }

private:
    const artifact::InputFile& file_;
    std::vector<std::byte> bytes_;
};

// Little-endian reads over `Prefix`, which is how every GGUF field is written.
class Walk {
public:
    Walk(Prefix& prefix, std::uint64_t at) : prefix_(prefix), at_(at) {}

    [[nodiscard]] std::uint64_t position() const noexcept { return at_; }

    std::uint8_t u8() { return static_cast<std::uint8_t>(take(1)); }
    std::uint16_t u16() { return static_cast<std::uint16_t>(take(2)); }
    std::uint32_t u32() { return static_cast<std::uint32_t>(take(4)); }
    std::uint64_t u64() { return take(8); }

    std::string text() {
        const auto size = u64();
        if (size > (1U << 20)) { throw ArtifactError("a GGUF string is implausibly long"); }
        const auto bytes = prefix_.at(at_, size);
        at_ += size;
        return std::string(reinterpret_cast<const char*>(bytes.data()),
                           static_cast<std::size_t>(size));
    }

    [[nodiscard]] double real(std::uint32_t type) {
        if (type == kF32) { return static_cast<double>(std::bit_cast<float>(u32())); }
        return std::bit_cast<double>(u64());
    }

private:
    std::uint64_t take(std::uint64_t size) {
        const auto bytes = prefix_.at(at_, size);
        at_ += size;
        std::uint64_t out = 0;
        for (std::uint64_t i = 0; i < size; ++i) {
            out |= std::uint64_t(std::to_integer<unsigned>(bytes[static_cast<std::size_t>(i)]))
                   << (8 * i);
        }
        return out;
    }

    Prefix& prefix_;
    std::uint64_t at_ = 0;
};

// One key/value entry. An array of strings keeps only its length: the tokenizer's vocabulary is the
// only one a drafter carries, and nothing here reads its text.
struct Entry {
    enum class Kind : std::uint8_t { Integer, Real, Boolean, Text, Integers, Reals, Booleans, Texts };

    Kind kind            = Kind::Integer;
    std::int64_t integer = 0;
    double real          = 0.0;
    bool boolean         = false;
    std::string text;
    std::vector<std::int64_t> integers;
    std::vector<double> reals;
    std::vector<bool> booleans;
    std::uint64_t texts = 0;
};

using Table = std::map<std::string, Entry, std::less<>>;

// A GGUF writes each scalar in its own width, so a value's type decides how many bytes it takes.
std::uint64_t scalar_bits(std::uint32_t type, Walk& walk) {
    switch (type) {
    case kU8:
    case kI8: return walk.u8();
    case kU16:
    case kI16: return walk.u16();
    case kU32:
    case kI32: return walk.u32();
    case kBool: return walk.u8();
    case kU64:
    case kI64: return walk.u64();
    default: break;
    }
    throw ArtifactError("unsupported GGUF value type " + std::to_string(type));
}

std::int64_t as_signed(std::uint64_t bits, std::uint32_t type) {
    switch (type) {
    case kI8: return static_cast<std::int8_t>(bits);
    case kI16: return static_cast<std::int16_t>(bits);
    case kI32: return static_cast<std::int32_t>(bits);
    case kI64: return static_cast<std::int64_t>(bits);
    default: return static_cast<std::int64_t>(bits);
    }
}

Entry read_entry(std::uint32_t type, Walk& walk) {
    Entry out;
    switch (type) {
    case kString:
        out.kind = Entry::Kind::Text;
        out.text = walk.text();
        return out;
    case kBool:
        out.kind    = Entry::Kind::Boolean;
        out.boolean = walk.u8() != 0;
        return out;
    case kF32:
    case kF64:
        out.kind = Entry::Kind::Real;
        out.real = walk.real(type);
        return out;
    case kArray: {
        const auto element = walk.u32();
        const auto count   = walk.u64();
        if (element == kString) {
            out.kind  = Entry::Kind::Texts;
            out.texts = count;
            for (std::uint64_t i = 0; i < count; ++i) { (void)walk.text(); }
            return out;
        }
        if (element == kArray) {
            throw ArtifactError("nested GGUF arrays are unsupported");
        }
        if (element == kF32 || element == kF64) {
            out.kind = Entry::Kind::Reals;
            for (std::uint64_t i = 0; i < count; ++i) { out.reals.push_back(walk.real(element)); }
            return out;
        }
        if (element == kBool) {
            out.kind = Entry::Kind::Booleans;
            for (std::uint64_t i = 0; i < count; ++i) {
                out.booleans.push_back((scalar_bits(element, walk) & 0xFFU) != 0);
            }
            return out;
        }
        out.kind = Entry::Kind::Integers;
        for (std::uint64_t i = 0; i < count; ++i) {
            out.integers.push_back(as_signed(scalar_bits(element, walk), element));
        }
        return out;
    }
    default: break;
    }
    out.kind    = Entry::Kind::Integer;
    out.integer = as_signed(scalar_bits(type, walk), type);
    return out;
}

[[nodiscard]] const Entry& entry(const Table& kv, std::string_view key,
                                 const std::filesystem::path& path) {
    const auto found = kv.find(key);
    if (found == kv.end()) {
        throw ArtifactError(path.string() + ": missing GGUF key " + std::string(key));
    }
    return found->second;
}

[[nodiscard]] std::uint64_t integer(const Table& kv, std::string_view key,
                                    const std::filesystem::path& path) {
    const auto& value = entry(kv, key, path);
    if (value.kind != Entry::Kind::Integer || value.integer < 0) {
        throw ArtifactError(path.string() + ": " + std::string(key) +
                            " must be a nonnegative integer");
    }
    return static_cast<std::uint64_t>(value.integer);
}

[[nodiscard]] double real(const Table& kv, std::string_view key,
                          const std::filesystem::path& path) {
    const auto& value = entry(kv, key, path);
    if (value.kind != Entry::Kind::Real || !std::isfinite(value.real) || value.real <= 0) {
        throw ArtifactError(path.string() + ": " + std::string(key) + " must be a positive real");
    }
    return value.real;
}

[[nodiscard]] std::vector<std::uint64_t> taps_of(const Table& kv, std::string_view key,
                                                 const std::filesystem::path& path) {
    const auto& value = entry(kv, key, path);
    if (value.kind != Entry::Kind::Integers || value.integers.empty()) {
        throw ArtifactError(path.string() + ": " + std::string(key) +
                            " must be a nonempty array of integers");
    }
    std::vector<std::uint64_t> out;
    for (const auto tap : value.integers) {
        if (tap < 0) {
            throw ArtifactError(path.string() + ": " + std::string(key) +
                                " holds a negative index");
        }
        out.push_back(static_cast<std::uint64_t>(tap));
    }
    return out;
}

// One tensor of the file's directory.
struct Tensor {
    std::string name;
    // The file lists its fastest dimension first; an artifact shape is row-major, so this is the
    // file's dims reversed.
    Shape shape;
    QType type           = QType::BF16;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

using Tensors = std::map<std::string, Tensor, std::less<>>;

std::uint64_t block_row_bytes(QType format, std::uint64_t k) {
    const auto block = gguf_block_shape(format);
    return artifact::checked_mul(k / std::uint64_t(block.elements), std::uint64_t(block.bytes),
                                 "GGUF block row bytes");
}

// The file's data section: a tensor's bytes, addressed by the directory's offsets.
class Source {
public:
    Source(const artifact::InputFile& file, std::uint64_t data_offset)
        : file_(file), data_offset_(data_offset) {}

    [[nodiscard]] std::vector<std::byte> bytes(const Tensor& tensor) const {
        if (tensor.bytes > (std::uint64_t(1) << 33)) {
            throw ArtifactError(tensor.name + ": tensor is too large to hold whole");
        }
        std::vector<std::byte> out(static_cast<std::size_t>(tensor.bytes));
        if (!out.empty()) { file_.read_exact(data_offset_ + tensor.offset, out); }
        return out;
    }

    // Rows `[first, last)` of a matrix whose stored rows are `stride` bytes.
    [[nodiscard]] std::vector<std::byte> rows(const Tensor& tensor, std::uint64_t first,
                                              std::uint64_t last, std::uint64_t stride) const {
        const auto count = artifact::checked_mul(last - first, stride, tensor.name);
        std::vector<std::byte> out(static_cast<std::size_t>(count));
        if (!out.empty()) {
            file_.read_exact(data_offset_ + tensor.offset + first * stride, out);
        }
        return out;
    }

private:
    const artifact::InputFile& file_;
    std::uint64_t data_offset_ = 0;
};

// The payload a drafter's tensors live in, as the artifact describes it.
struct Manifest {
    Manifest(const Tensors& source, Directory directory)
        : tensors(source), directory(std::move(directory)) {}

    const Tensors& tensors;
    Directory directory;
    std::map<std::string, ObjectHandle, std::less<>> objects;
    std::vector<std::byte> derived;
    std::uint64_t file_payload_bytes = 0;
    std::uint64_t next_derived       = 0;

    [[nodiscard]] const Tensor& stored(std::string_view name) const {
        const auto found = tensors.find(name);
        if (found == tensors.end()) {
            throw ArtifactError("GGUF has no tensor " + std::string(name));
        }
        return found->second;
    }

    [[nodiscard]] ObjectHandle handle(std::string_view name) const {
        const auto found = objects.find(name);
        if (found == objects.end()) {
            throw ArtifactError("GGUF has no tensor " + std::string(name));
        }
        return found->second;
    }

    // A tensor the file does not hold, in Host memory beside its payload. `fill` writes its encoded
    // bytes in place, so a vocabulary-sized table is never held twice.
    template <typename Fill>
    ObjectHandle derive(const std::string& id, Shape shape, QType format, QuantLayout layout,
                        Fill&& fill) {
        const std::uint64_t bytes = weight_geometry(format, layout, shape).bytes;
        if (!directory.object_index.emplace(id, ObjectHandle{directory.objects.size()}).second) {
            throw ArtifactError(id + ": duplicate derived object");
        }
        const auto absolute =
            artifact::align_up(file_payload_bytes + next_derived, 256, "derived object offset");
        next_derived = absolute - file_payload_bytes;
        if (derived.size() < next_derived + bytes) {
            derived.resize(static_cast<std::size_t>(next_derived + bytes));
        }
        fill(std::span<std::byte>(derived.data() + next_derived, static_cast<std::size_t>(bytes)));
        directory.objects.emplace_back(artifact::TensorObject{
            id, std::move(shape), std::string(artifact::format_name(format)),
            std::string(artifact::layout_name(layout)), absolute, bytes, 1});
        next_derived += bytes;
        return ObjectHandle{directory.objects.size() - 1};
    }

    // A parameter the loader multiplies by `input`: one whole tensor, since a GGUF stores each
    // projection on its own.
    void parameter(const std::string& name, ObjectHandle object, const Shape& shape,
                   const std::string& input) {
        bind(name, whole_object(name, object, shape));
        use(name, input);
    }

    // A parameter that is some of a shared object's rows, which is how a fused projection is
    // addressed: the object holds more rows than the parameter reads.
    void rows_of(const std::string& name, ObjectHandle object, const Shape& shape,
                 std::uint64_t first_row, const std::string& input) {
        if (shape.size() != 2) { throw ArtifactError(name + ": a row range is a matrix"); }
        Binding binding;
        binding.elements = weight_element_count(shape);
        const auto begin = first_row * shape[1];
        binding.parts.push_back({object, begin, begin + binding.elements});
        bind(name, std::move(binding));
        use(name, input);
    }

    // A tensor its consumer reads whole, which every loader form declares BF16.
    void direct(const std::string& name, ObjectHandle object, const Shape& shape) {
        const auto& stored_object = directory.tensor(object);
        if (stored_object.format != artifact::format_name(QType::BF16)) {
            throw ArtifactError(name + ": a direct parameter must be BF16, the GGUF stores " +
                                stored_object.format);
        }
        bind(name, whole_object(name, object, shape));
    }

    Binding whole_object(const std::string& name, ObjectHandle object, const Shape& shape) const {
        const auto& stored_object = directory.tensor(object);
        if (stored_object.shape != shape) {
            throw ArtifactError(name + ": the GGUF tensor is " + shape_text(stored_object.shape) +
                                ", the parameter " + shape_text(shape));
        }
        Binding binding;
        binding.whole_object = true;
        binding.elements     = weight_element_count(shape);
        binding.parts.push_back({object, 0, binding.elements});
        return binding;
    }

    void bind(const std::string& name, Binding binding) {
        if (!directory.bindings.emplace(name, std::move(binding)).second) {
            throw ArtifactError(name + ": duplicate parameter");
        }
    }

    // Every stored block type takes the A16 route: no GGUF block format carries the FP8 or NVFP4
    // permission a narrower activation would need.
    void use(const std::string& parameter, const std::string& input) {
        Use out;
        out.parameter         = parameter;
        out.input             = input;
        out.activation_policy = artifact::ActivationPolicy::A16Only;
        if (!directory.uses.emplace(std::pair{parameter, input}, std::move(out)).second) {
            throw ArtifactError(parameter + "@" + input + ": duplicate Use");
        }
    }

    static std::string shape_text(const Shape& shape) {
        std::string out = "[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) { out += ","; }
            out += std::to_string(shape[i]);
        }
        return out + "]";
    }
};

// The drafter's own geometry, from the file's `dflash.*` keys and its tensors.
struct Geometry {
    std::uint64_t hidden       = 0;
    std::uint64_t vocab        = 0;
    std::uint64_t intermediate = 0;
    std::uint64_t heads        = 0;
    std::uint64_t kv_heads     = 0;
    std::uint64_t head_dim     = 0;
    std::uint64_t layers       = 0;
    std::uint64_t context      = 0;
    std::uint64_t window       = 0;
    std::uint64_t block_size   = 0;
    std::uint64_t conv_kernel  = 0;
    std::uint64_t conv_group   = 0;
    std::uint64_t selector     = 0;
    std::uint64_t selector_top = 0;
    std::uint64_t mask_token   = 0;
    std::uint64_t code_size    = 0;
    std::uint64_t rank         = 0;
    double norm_eps            = 0.0;
    double rope_theta          = 0.0;
    bool causal                = false;
    std::vector<std::uint64_t> taps;
    std::vector<bool> sliding;

    [[nodiscard]] std::uint64_t query_width() const { return heads * head_dim; }
    [[nodiscard]] std::uint64_t key_width() const { return kv_heads * head_dim; }
    [[nodiscard]] bool any_sliding() const {
        return std::find(sliding.begin(), sliding.end(), true) != sliding.end();
    }
};

DraftArtifactKind read_kind(const Table& kv, const Tensors& tensors,
                            const std::filesystem::path& path) {
    const auto architecture = kv.find("general.architecture");
    if (architecture == kv.end() || architecture->second.text != "dflash") {
        throw ArtifactError(path.string() +
                            ": general.architecture is not 'dflash'; this is not a drafter GGUF");
    }
    // Every release declares the `dflash` architecture; the finetune name and the tensors tell the
    // three apart, and a file that states neither is a plain DFlash.
    const auto finetune    = kv.find("general.finetune");
    const std::string name = finetune == kv.end() ? std::string() : finetune->second.text;
    if (name.find("DSpine") != std::string::npos) { return DraftArtifactKind::DSpine; }
    if (name.find("DFlash2") != std::string::npos) { return DraftArtifactKind::DFlash2; }
    if (tensors.contains("dspine_codebook.weight")) { return DraftArtifactKind::DSpine; }
    if (tensors.contains("selector_hidden.weight")) { return DraftArtifactKind::DFlash2; }
    return DraftArtifactKind::DFlash;
}

Geometry read_geometry(const Table& kv, const Tensors& tensors, DraftArtifactKind kind,
                       const std::filesystem::path& path) {
    Geometry out;
    out.hidden       = integer(kv, "dflash.embedding_length", path);
    out.intermediate = integer(kv, "dflash.feed_forward_length", path);
    out.heads        = integer(kv, "dflash.attention.head_count", path);
    out.kv_heads     = integer(kv, "dflash.attention.head_count_kv", path);
    out.head_dim     = integer(kv, "dflash.attention.key_length", path);
    out.layers       = integer(kv, "dflash.block_count", path);
    out.context      = integer(kv, "dflash.context_length", path);
    out.window       = integer(kv, "dflash.attention.sliding_window", path);
    out.block_size   = integer(kv, "dflash.block_size", path);
    out.mask_token   = integer(kv, "tokenizer.ggml.mask_token_id", path);
    // A DFlash2 drafter states how many candidates its selector keeps; a DFlash one has none, and a
    // DSpine one reads its own `head_top_k` from the same number.
    out.selector_top = kv.contains("dflash.selector_top_k")
                           ? integer(kv, "dflash.selector_top_k", path)
                           : 0;
    out.norm_eps     = real(kv, "dflash.attention.layer_norm_rms_epsilon", path);
    out.rope_theta   = real(kv, "dflash.rope.freq_base", path);
    out.taps         = taps_of(kv, "dflash.target_layers", path);
    // A drafter keeps its key and value the same width; a file that states only the key's says so.
    if (kv.contains("dflash.attention.value_length") &&
        integer(kv, "dflash.attention.value_length", path) != out.head_dim) {
        throw ArtifactError(path.string() + ": the drafter's key and value widths differ");
    }
    if (kv.contains("dflash.attention.causal")) {
        const auto causal = entry(kv, "dflash.attention.causal", path);
        if (causal.kind != Entry::Kind::Boolean) {
            throw ArtifactError(path.string() + ": dflash.attention.causal must be a boolean");
        }
        out.causal = causal.boolean;
    }
    // An earlier drafter states no per-layer pattern: its window, when it has one, is every layer's.
    if (!kv.contains("dflash.attention.sliding_window_pattern")) {
        out.sliding.assign(out.layers, out.window != 0);
    } else {
        const auto pattern = entry(kv, "dflash.attention.sliding_window_pattern", path);
        if (pattern.kind != Entry::Kind::Booleans || pattern.booleans.size() != out.layers) {
            throw ArtifactError(path.string() +
                                ": dflash.attention.sliding_window_pattern must have one entry per "
                                "layer");
        }
        out.sliding = pattern.booleans;
    }
    const auto tokens = entry(kv, "tokenizer.ggml.tokens", path);
    if (tokens.kind != Entry::Kind::Texts || tokens.texts == 0) {
        throw ArtifactError(path.string() + ": tokenizer.ggml.tokens is absent");
    }
    out.vocab = tokens.texts;
    if (out.mask_token >= out.vocab) {
        throw ArtifactError(path.string() + ": the mask token lies outside the vocabulary");
    }
    if (kind == DraftArtifactKind::DFlash2) {
        out.conv_kernel = integer(kv, "dflash.conv_kernel_size", path);
        out.conv_group  = integer(kv, "dflash.conv_group_size", path);
        out.selector    = integer(kv, "dflash.selector_rank", path);
        if (!out.conv_kernel || !out.conv_group || out.hidden % out.conv_group) {
            throw ArtifactError(path.string() + ": the DFlash2 convolution geometry is invalid");
        }
        if (!out.selector_top) {
            throw ArtifactError(path.string() + ": dflash.selector_top_k is absent");
        }
    }
    if (kind == DraftArtifactKind::DSpine) {
        // The bridge's widths come from the tensors themselves: the codebook's rows are the
        // vocabulary, its columns the code width, and the gate's rows the bridge rank.
        out.code_size = tensors.at("dspine_codebook.weight").shape[1];
        out.rank      = tensors.at("dspine_gate.weight").shape[0];
        if (out.code_size % 8 || out.rank % 32 || out.rank > 512) {
            throw ArtifactError(path.string() + ": the DSpine bridge geometry is invalid");
        }
        if (std::any_of(out.sliding.begin(), out.sliding.end(), [](bool on) { return !on; })) {
            throw ArtifactError(path.string() + ": every DSpine layer must be sliding-attention");
        }
    }
    if (out.head_dim % 2 || !out.kv_heads || out.heads % out.kv_heads) {
        throw ArtifactError(path.string() + ": the drafter's attention geometry is invalid");
    }
    return out;
}

// The drafter's config: exactly the members models/qwen3_5/config.cpp (DFlash, DFlash2) or
// models/dspine/config.cpp (DSpine) parse, since both refuse a member they do not know.
Json draft_config(const Geometry& g, DraftArtifactKind kind) {
    const bool dspine = kind == DraftArtifactKind::DSpine;
    Json inner;
    inner["target_layer_ids"] = Json::array();
    // The GGUF numbers the target's layers from one. A DFlash drafter taps the input of a layer and
    // restates the tap zero-based; a DSpine drafter keeps the file's numbering because the engine
    // reads a layer's input.
    for (const auto tap : g.taps) {
        inner["target_layer_ids"].push_back(dspine ? tap : tap - 1);
    }
    inner["mask_token_id"] = g.mask_token;
    if (kind == DraftArtifactKind::DFlash2) {
        inner["conv_kernel_size"] = g.conv_kernel;
        inner["conv_group_size"]  = g.conv_group;
        inner["selector_rank"]    = g.selector;
        inner["selector_top_k"]   = g.selector_top;
    }
    if (dspine) {
        inner["code_size"]  = g.code_size;
        inner["rank"]       = g.rank;
        inner["head_top_k"] = g.selector_top;
        inner["bridge_eps"] = 1e-6;
    }
    Json out;
    out["architectures"] = Json::array({dspine ? "DSpineDraftModel"
                                               : (kind == DraftArtifactKind::DFlash2
                                                      ? "DFlash2DraftModel"
                                                      : "DFlashDraftModel")});
    out["model_type"]              = "qwen3";
    out["hidden_size"]             = g.hidden;
    out["vocab_size"]              = g.vocab;
    out["intermediate_size"]       = g.intermediate;
    out["num_attention_heads"]     = g.heads;
    out["num_key_value_heads"]     = g.kv_heads;
    out["head_dim"]                = g.head_dim;
    out["num_hidden_layers"]       = g.layers;
    out["max_position_embeddings"] = g.context;
    out["rms_norm_eps"]            = g.norm_eps;
    out["rope_parameters"]         = Json{{"rope_theta", g.rope_theta}};
    out["layer_types"]             = Json::array();
    for (const bool sliding : g.sliding) {
        out["layer_types"].push_back(sliding ? "sliding_attention" : "full_attention");
    }
    out["is_causal"] = g.causal;
    if (dspine) {
        // The bridge's block and window are the DSpine config's own, not a DFlash one's.
        out["block_size"]     = g.block_size;
        out["sliding_window"] = g.window;
        out["dspine_config"]  = std::move(inner);
        return out;
    }
    inner["block_size"]  = g.block_size;
    out["dflash_config"] = std::move(inner);
    if (g.any_sliding()) { out["sliding_window"] = g.window; }
    return out;
}

// ---------------------------------------------------------------- derived tensors

float half_to_float(std::uint16_t half);

std::uint16_t load16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0] | (unsigned(bytes[1]) << 8U));
}

// Q2_K and Q3_K as ggml dequantises them. The engine's slice decoder (`decode_slice`) carries the
// types its CPU experts need and neither of these, and both change scale every 16 values, which its
// two factors cannot express.
bool decode_k_extra(QType format, const std::byte* row, std::uint64_t k, float* out) {
    if (format != QType::GGUF_Q2_K && format != QType::GGUF_Q3_K) { return false; }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(row);
    const auto block  = gguf_block_shape(format);
    for (std::uint64_t index = 0; index < k / 256; ++index) {
        const std::uint8_t* at = bytes + index * std::uint64_t(block.bytes);
        float* values          = out + index * 256;
        if (format == QType::GGUF_Q2_K) {
            const float d    = half_to_float(load16(at + 80));
            const float dmin = half_to_float(load16(at + 82));
            const std::uint8_t* codes = at + 16;
            for (int half = 0; half < 2; ++half) {
                for (int j = 0; j < 4; ++j) {
                    const int shift = 2 * j;
                    for (int g = 0; g < 2; ++g) {
                        const std::uint8_t scale = at[8 * half + 2 * j + g];
                        const float dl           = d * float(scale & 15);
                        const float ml           = dmin * float(scale >> 4);
                        const std::uint8_t* row16 = codes + 32 * half + 16 * g;
                        for (int l = 0; l < 16; ++l) {
                            values[128 * half + 32 * j + 16 * g + l] =
                                dl * float((row16[l] >> shift) & 3) - ml;
                        }
                    }
                }
            }
            continue;
        }
        // Q3_K: 32 bytes of high bits, 64 of 2-bit codes, 12 of 6-bit scales, then d.
        const float d = half_to_float(load16(at + 108));
        constexpr std::uint32_t kLow   = 0x0f0f0f0fU;
        constexpr std::uint32_t kHigh  = 0x03030303U;
        std::uint32_t words[4]{};
        std::memcpy(words, at + 96, 12);
        const std::uint32_t packed = words[2];
        words[2] = ((words[0] >> 4) & kLow) | (((packed >> 4) & kHigh) << 4);
        words[3] = ((words[1] >> 4) & kLow) | (((packed >> 6) & kHigh) << 4);
        words[0] = (words[0] & kLow) | (((packed >> 0) & kHigh) << 4);
        words[1] = (words[1] & kLow) | (((packed >> 2) & kHigh) << 4);
        const auto* scales        = reinterpret_cast<const std::int8_t*>(words);
        const std::uint8_t* codes = at + 32;
        for (int half = 0; half < 2; ++half) {
            for (int j = 0; j < 4; ++j) {
                const int shift       = 2 * j;
                const std::uint8_t on = static_cast<std::uint8_t>(1U << (4 * half + j));
                const float low       = d * float(int(scales[8 * half + 2 * j]) - 32);
                const float high      = d * float(int(scales[8 * half + 2 * j + 1]) - 32);
                const std::uint8_t* row32 = codes + 32 * half;
                for (int l = 0; l < 16; ++l) {
                    const int code = (row32[l] >> shift) & 3;
                    const int sign = (at[l] & on) ? 0 : 4;
                    values[128 * half + 32 * j + l] = low * float(code - sign);
                }
                for (int l = 0; l < 16; ++l) {
                    const int code = (row32[16 + l] >> shift) & 3;
                    const int sign = (at[16 + l] & on) ? 0 : 4;
                    values[128 * half + 32 * j + 16 + l] = high * float(code - sign);
                }
            }
        }
    }
    return true;
}

// The block types this translation unit can read back: `decode_slice`'s set plus Q2_K and Q3_K.
bool decodable(QType format) {
    switch (format) {
    case QType::GGUF_Q2_K:
    case QType::GGUF_Q3_K:
    case QType::GGUF_Q2_0:
    case QType::GGUF_Q4_0:
    case QType::GGUF_Q5_0:
    case QType::GGUF_Q8_0:
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K:
    case QType::GGUF_Q6_K:
    case QType::GGUF_IQ2_S:
    case QType::GGUF_IQ3_XXS:
    case QType::GGUF_IQ3_S:
    case QType::GGUF_IQ4_NL:
    case QType::GGUF_IQ4_XS: return true;
    default: return false;
    }
}

// One row of decoded values from `row`, which is `k` wide. `decode_slice` is the engine's own Host
// decoder (ops/moe_experts), so these are the values its GGUF products read.
void decode_row(QType format, const std::byte* row, std::uint64_t k, float* out) {
    if (decode_k_extra(format, row, k, out)) { return; }
    if (!decodable(format)) {
        throw ArtifactError(std::string(artifact::format_name(format)) +
                            ": a drafter GGUF of this block type cannot be read on the Host");
    }
    const bool paired = format == QType::GGUF_Q6_K || format == QType::GGUF_IQ2_S;
    const bool offset = format == QType::GGUF_Q4_K || format == QType::GGUF_Q5_K;
    ops::gguf_cpu_detail::GgufCpuSlice slice;
    for (std::uint64_t s = 0; s < k / 32; ++s) {
        ops::gguf_cpu_detail::decode_slice(
            format, reinterpret_cast<const std::uint8_t*>(row), static_cast<int>(s), slice);
        for (int j = 0; j < 32; ++j) {
            const float factor = paired ? (j < 16 ? slice.factor[0] : slice.factor[1])
                                        : slice.factor[0];
            const float value  = static_cast<float>(slice.weights[j]) * factor;
            out[s * 32 + std::size_t(j)] = offset ? value - slice.factor[1] : value;
        }
    }
}

constexpr float bf16_to_float(std::uint16_t word) {
    return std::bit_cast<float>(std::uint32_t(word) << 16);
}

// `shape`'s values in BF16: a stored F32 rounds, a stored BF16 is copied, and a block matrix
// decodes a row at a time, its k being the shape's last axis.
void write_bf16(const Source& source, const Tensor& tensor, const Shape& shape,
                std::span<std::byte> out) {
    auto* words                  = reinterpret_cast<std::uint16_t*>(out.data());
    const std::uint64_t elements = weight_element_count(shape);
    if (tensor.type == QType::BF16) {
        const auto bytes = source.bytes(tensor);
        if (bytes.size() != out.size()) { throw ArtifactError(tensor.name + ": size differs"); }
        std::memcpy(out.data(), bytes.data(), out.size());
        return;
    }
    if (tensor.type == QType::FP32) {
        const auto bytes   = source.bytes(tensor);
        const auto* values = reinterpret_cast<const float*>(bytes.data());
        for (std::uint64_t i = 0; i < elements; ++i) {
            words[i] = ops::gguf_cpu_detail::float_to_bf16(values[i]);
        }
        return;
    }
    const std::uint64_t k      = shape.back();
    const std::uint64_t rows   = elements / k;
    const std::uint64_t stride = block_row_bytes(tensor.type, k);
    const auto bytes           = source.rows(tensor, 0, rows, stride);
    std::vector<float> values(static_cast<std::size_t>(k));
    for (std::uint64_t row = 0; row < rows; ++row) {
        decode_row(tensor.type, bytes.data() + row * stride, k, values.data());
        for (std::uint64_t c = 0; c < k; ++c) {
            words[row * k + c] = ops::gguf_cpu_detail::float_to_bf16(
                values[static_cast<std::size_t>(c)]);
        }
    }
}

// The stored tensor's values as a Host BF16 tensor of `shape`.
ObjectHandle bf16_of(Manifest& manifest, const Source& source, const std::string& name,
                     const Shape& shape) {
    const Tensor tensor = manifest.stored(name);
    return manifest.derive(name + ".bf16", shape, QType::BF16, QuantLayout::Contiguous,
                           [&](std::span<std::byte> out) {
                               write_bf16(source, tensor, shape, out);
                           });
}

// The refinement's sender table: every token's [predecessor gate; value] rows, which the reference
// builds as `(sqrt(code) * codebook).bf16() @ [predecessor_gate; prev].T` and rounds to BF16. The
// GGUF holds its ingredients rather than the product, and the bridge Op reads the product by token,
// so it is computed here once.
void write_sender_table(const Source& source, const Tensor& codebook, const Tensor& predecessor,
                        const Tensor& previous, std::uint64_t vocab, std::uint64_t code,
                        std::uint64_t rank, std::span<std::byte> out) {
    const std::uint64_t width = 2 * rank;
    std::vector<float> sender(static_cast<std::size_t>(width * code));
    for (std::uint64_t half = 0; half < 2; ++half) {
        const Tensor& tensor = half == 0 ? predecessor : previous;
        const auto bytes     = source.bytes(tensor);
        const auto* words    = reinterpret_cast<const std::uint16_t*>(bytes.data());
        for (std::uint64_t i = 0; i < rank * code; ++i) {
            sender[static_cast<std::size_t>(half * rank * code + i)] = bf16_to_float(words[i]);
        }
    }
    const std::uint64_t stride = block_row_bytes(codebook.type, code);
    const float scale          = std::sqrt(static_cast<float>(code));
    const unsigned threads     = std::clamp(std::thread::hardware_concurrency(), 1U, 32U);
    const std::uint64_t chunk  = (vocab + threads - 1) / threads;
    std::vector<std::exception_ptr> failures(threads);
    const auto work = [&](unsigned index) {
        try {
            const std::uint64_t first = index * chunk;
            const std::uint64_t last  = std::min(vocab, first + chunk);
            if (first >= last) { return; }
            const auto bytes = source.rows(codebook, first, last, stride);
            std::vector<float> codes(static_cast<std::size_t>(code));
            auto* words = reinterpret_cast<std::uint16_t*>(out.data()) + first * width;
            for (std::uint64_t row = first; row < last; ++row) {
                decode_row(codebook.type, bytes.data() + (row - first) * stride, code,
                           codes.data());
                // The reference scales the codebook row and rounds it to BF16 before the product.
                for (std::uint64_t c = 0; c < code; ++c) {
                    const auto& value = codes[static_cast<std::size_t>(c)];
                    codes[static_cast<std::size_t>(c)] = bf16_to_float(
                        ops::gguf_cpu_detail::float_to_bf16(scale * value));
                }
                for (std::uint64_t j = 0; j < width; ++j) {
                    const float* line = sender.data() + j * code;
                    float sum         = 0.0F;
                    for (std::uint64_t c = 0; c < code; ++c) {
                        sum += codes[static_cast<std::size_t>(c)] * line[c];
                    }
                    words[j] = ops::gguf_cpu_detail::float_to_bf16(sum);
                }
                words += width;
            }
        } catch (...) { failures[index] = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (unsigned index = 1; index < threads; ++index) { pool.emplace_back(work, index); }
    work(0);
    for (auto& thread : pool) { thread.join(); }
    for (const auto& failure : failures) {
        if (failure) { std::rethrow_exception(failure); }
    }
}

// ---------------------------------------------------------------- the fused query/key/value

// IEEE binary16 from binary32, round to nearest even. ggml writes a q8 block's multiplier this way,
// and the kernel reads it back as the scale it quantised against.
std::uint16_t float_to_half(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t rest = bits & 0x7fffffffU;
    if (rest >= 0x7f800000U) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | ((rest & 0x007fffffU) ? 0x0200U : 0U));
    }
    int exponent            = static_cast<int>(rest >> 23U) - 127 + 15;
    std::uint32_t mantissa  = rest & 0x007fffffU;
    const auto round_even   = [](std::uint32_t v, int shift) {
        const std::uint32_t mask = (1U << shift) - 1U;
        const std::uint32_t half = 1U << (shift - 1);
        const std::uint32_t base = v >> shift;
        const std::uint32_t rem  = v & mask;
        return base + ((rem > half || (rem == half && (base & 1U))) ? 1U : 0U);
    };
    if (exponent <= 0) {
        if (exponent < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000U;
        return static_cast<std::uint16_t>(sign | round_even(mantissa, 14 - exponent));
    }
    if (exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    std::uint32_t rounded = round_even(mantissa, 13);
    if (rounded == 0x0400U) {
        rounded = 0;
        if (++exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10U) | rounded);
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign     = (static_cast<std::uint32_t>(half) & 0x8000U) << 16U;
    std::uint32_t exponent       = (static_cast<std::uint32_t>(half) >> 10U) & 0x1fU;
    std::uint32_t mantissa       = static_cast<std::uint32_t>(half) & 0x03ffU;
    if (exponent == 0) {
        if (!mantissa) { return std::bit_cast<float>(sign); }
        int shift = -14;
        while (!(mantissa & 0x0400U)) {
            mantissa <<= 1U;
            --shift;
        }
        mantissa &= 0x03ffU;
        return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(shift + 127) << 23U) |
                                    (mantissa << 13U));
    }
    if (exponent == 31) {
        return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    }
    return std::bit_cast<float>(sign | ((exponent - 15 + 127) << 23U) | (mantissa << 13U));
}

// One row in Q8_G32_FP16: per 32 values the largest magnitude over 127 as a binary16 multiplier,
// codes rounded to nearest against it. The multipliers live in their own plane.
void quantize_q8_row(const float* values, std::uint64_t k, std::byte* codes, std::byte* scales) {
    const std::uint64_t groups = k / 32;
    for (std::uint64_t group = 0; group < groups; ++group) {
        const float* source = values + group * 32;
        float largest       = 0.0F;
        for (int j = 0; j < 32; ++j) { largest = std::max(largest, std::fabs(source[j])); }
        const std::uint16_t word = float_to_half(largest / 127.0F);
        std::byte* scale         = scales + group * 2;
        scale[0]                 = static_cast<std::byte>(word & 0xffU);
        scale[1]                 = static_cast<std::byte>(word >> 8U);
        const float divisor  = half_to_float(word);
        const float inverse  = divisor > 0.0F ? 1.0F / divisor : 0.0F;
        std::int8_t* out     = reinterpret_cast<std::int8_t*>(codes + group * 32);
        for (int j = 0; j < 32; ++j) {
            const float scaled = std::nearbyint(source[j] * inverse);
            out[j]             = static_cast<std::int8_t>(
                std::clamp(scaled, -127.0F, 127.0F));
        }
    }
}

// The fused query/key/value matrix a drafter's attention reads in one pass. Its kernel takes one
// row-split Q8 parent (ops/wrapper/attn_input_proj.cpp), which is what tools/convert writes into a
// .ninfer drafter; a GGUF stores the three projections apart, so they are quantised together here.
void write_qkv_q8(Manifest& m, const Source& source, const Geometry& g,
                  const std::array<std::string, 3>& names, std::span<std::byte> out) {
    const std::uint64_t k = g.hidden;
    const std::array<std::uint64_t, 3> counts{g.query_width(), g.key_width(), g.key_width()};
    std::uint64_t total = 0;
    for (const auto count : counts) { total += count; }
    const auto geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, Shape{total, k});
    if (geometry.bytes != out.size()) {
        throw ArtifactError("the fused query/key/value matrix is the wrong size");
    }
    std::array<Tensor, 3> stored{};
    for (std::size_t i = 0; i < names.size(); ++i) { stored[i] = m.stored(names[i]); }
    std::byte* const codes  = out.data();
    std::byte* const scales = out.data() + geometry.scale_offset;
    const std::uint64_t scale_row_bytes = geometry.scale_bytes_per_row;
    const unsigned threads = std::clamp(std::thread::hardware_concurrency(), 1U, 32U);
    const std::uint64_t chunk = (total + threads - 1) / threads;
    std::vector<std::exception_ptr> failures(threads);
    const auto work = [&](unsigned index) {
        try {
            const std::uint64_t first = index * chunk;
            const std::uint64_t last  = std::min(total, first + chunk);
            if (first >= last) { return; }
            std::vector<float> values(static_cast<std::size_t>(k));
            std::uint64_t base = 0;
            for (std::size_t part = 0; part < stored.size(); ++part) {
                const std::uint64_t begin = base;
                const std::uint64_t end   = base + counts[part];
                base                      = end;
                const std::uint64_t from  = std::max(first, begin);
                const std::uint64_t to    = std::min(last, end);
                if (from >= to) { continue; }
                const auto& tensor = stored[part];
                const auto stride  = block_row_bytes(tensor.type, k);
                const auto bytes   = source.rows(tensor, from - begin, to - begin, stride);
                for (std::uint64_t row = from; row < to; ++row) {
                    decode_row(tensor.type, bytes.data() + (row - from) * stride, k, values.data());
                    quantize_q8_row(values.data(), k, codes + row * k,
                                    scales + row * scale_row_bytes);
                }
            }
        } catch (...) { failures[index] = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    for (unsigned index = 1; index < threads; ++index) { pool.emplace_back(work, index); }
    work(0);
    for (auto& thread : pool) { thread.join(); }
    for (const auto& failure : failures) {
        if (failure) { std::rethrow_exception(failure); }
    }
}

// The fused gate and up projection a SwiGLU product reads as one parent. A GGUF stores the two as
// row blocks either way, so fusing is copying: the object holds the file's bytes end to end and no
// value is re-encoded.
void write_fused_blocks(Manifest& m, const Source& source,
                        std::initializer_list<std::string> names, QType format,
                        std::span<std::byte> out) {
    std::uint64_t at = 0;
    for (const auto& name : names) {
        const auto& tensor = m.stored(name);
        if (tensor.type != format) {
            throw ArtifactError(name + ": a fused projection's rows differ in format");
        }
        const auto bytes = source.bytes(tensor);
        if (bytes.size() > out.size() - at) {
            throw ArtifactError(name + ": the fused projection overflows its object");
        }
        std::memcpy(out.data() + at, bytes.data(), bytes.size());
        at += bytes.size();
    }
    if (at != out.size()) {
        throw ArtifactError("the fused projection does not fill its object");
    }
}

// One layer's SwiGLU bank: the fused object plus the gate and up row ranges over it.
void add_swiglu(Manifest& m, const Source& source, const std::string& prefix,
                const std::string& gate_name, const std::string& up_name, std::uint64_t h,
                std::uint64_t inner, const std::string& input) {
    const auto& gate   = m.stored(gate_name);
    const auto object  = m.derive(prefix + "mlp/gate_up", {2 * inner, h}, gate.type,
                                  layout_of(gate.type), [&](std::span<std::byte> out) {
                                      write_fused_blocks(m, source, {gate_name, up_name},
                                                         gate.type, out);
                                  });
    m.rows_of(prefix + "mlp/gate", object, {inner, h}, 0, input);
    m.rows_of(prefix + "mlp/up", object, {inner, h}, inner, input);
}

// ---------------------------------------------------------------- the parameter sets

// models/qwen3_5/load/dflash.cpp and dflash2.cpp: what a DFlash or DFlash2 drafter binds.
void add_dflash(Manifest& m, const Source& source, const Geometry& g, DraftArtifactKind kind) {
    const bool second = kind == DraftArtifactKind::DFlash2;
    const std::string component(draft_artifact_name(kind));
    const auto h     = g.hidden;
    const auto inner = g.intermediate;
    const auto q     = g.query_width();
    const auto k     = g.key_width();
    const auto d     = g.head_dim;
    const auto taps  = std::uint64_t(g.taps.size());

    m.parameter(component + "/feature_projection", m.handle("fc.weight"), {h, taps * h},
                component + "/target_features");
    m.direct(component + "/context_norm", bf16_of(m, source, "enc.output_norm.weight", {h}), {h});
    m.direct(component + "/final_norm", bf16_of(m, source, "output_norm.weight", {h}), {h});
    for (std::uint64_t layer = 0; layer < g.layers; ++layer) {
        const std::string p   = component + "/layers/" + std::to_string(layer) + "/";
        const std::string blk = "blk." + std::to_string(layer) + ".";
        m.direct(p + "input_norm", bf16_of(m, source, blk + "attn_norm.weight", {h}), {h});
        m.direct(p + "post_attention_norm", bf16_of(m, source, blk + "ffn_norm.weight", {h}), {h});
        // The attention reads query, key and value as one row-split Q8 matrix, so the file's three
        // projections become one object here and the parameters address its row ranges.
        const auto qkv = m.derive(p + "attention/query_key_value", {q + 2 * k, h},
                                  QType::Q8_G32_FP16, QuantLayout::RowSplit,
                                  [&](std::span<std::byte> out) {
                                      write_qkv_q8(m, source, g,
                                                   {blk + "attn_q.weight", blk + "attn_k.weight",
                                                    blk + "attn_v.weight"},
                                                   out);
                                  });
        m.rows_of(p + "attention/query", qkv, {q, h}, 0, p + "query_projection_input");
        m.rows_of(p + "attention/key", qkv, {k, h}, q, p + "query_projection_input");
        m.rows_of(p + "attention/value", qkv, {k, h}, q + k, p + "query_projection_input");
        // The context projections are the key and value rows again: the adapter shares them, and a
        // GGUF stores each of those once.
        m.rows_of(p + "attention/context_key", qkv, {k, h}, q, component + "/context_input");
        m.rows_of(p + "attention/context_value", qkv, {k, h}, q + k, component + "/context_input");
        m.direct(p + "attention/query_norm", bf16_of(m, source, blk + "attn_q_norm.weight", {d}),
                 {d});
        m.direct(p + "attention/key_norm", bf16_of(m, source, blk + "attn_k_norm.weight", {d}), {d});
        m.parameter(p + "attention/output", m.handle(blk + "attn_output.weight"), {h, q},
                    p + "attention_output");
        add_swiglu(m, source, p, blk + "ffn_gate.weight", blk + "ffn_up.weight", h, inner,
                   p + "mlp_input");
        m.parameter(p + "mlp/down", m.handle(blk + "ffn_down.weight"), {h, inner},
                    p + "mlp_product");
        if (!second) { continue; }
        // The dynamic convolutions: a base kernel the consumer reads whole and a projection the
        // convolution reshapes into 2 * kernel groups of h / group rows.
        const auto rows = 2 * g.conv_kernel * (h / g.conv_group);
        for (const auto& [conv, gguf] :
             {std::pair<std::string, std::string>{"attention_conv", "attn_conv"},
              std::pair<std::string, std::string>{"mlp_conv", "ffn_conv"}}) {
            const std::string q_conv = p + conv + "/";
            m.direct(q_conv + "base_kernel",
                     bf16_of(m, source, blk + gguf + "_base", {2, g.conv_kernel, h}),
                     {2, g.conv_kernel, h});
            // The projection is read as a BF16 matrix of conv coefficients, whichever representation
            // the file stores it in.
            m.parameter(q_conv + "kernel_projection",
                        bf16_of(m, source, blk + gguf + "_proj.weight", {rows, h}), {rows, h},
                        p + conv + "_input");
        }
    }
    if (!second) { return; }
    m.parameter(component + "/candidate_selector/hidden_projection",
                m.handle("selector_hidden.weight"), {g.selector, h}, component + "/final_hidden");
    for (const auto& [role, gguf] :
         {std::pair<std::string, std::string>{"predecessor_codebook", "selector_predecessor.weight"},
          std::pair<std::string, std::string>{"successor_codebook", "selector_successor.weight"}}) {
        const auto& tensor = m.stored(gguf);
        m.direct(component + "/candidate_selector/" + role,
                 bf16_of(m, source, gguf, tensor.shape), tensor.shape);
    }
}

// models/dspine/model.cpp: what a standalone DSpine drafter binds. Its bridge reads three tables the
// GGUF does not store: the three gate blocks as one projection, the last layer's write transposed,
// and the sender table the refinement chains.
void add_dspine(Manifest& m, const Source& source, const Geometry& g) {
    const std::string c("dspine");
    const auto h     = g.hidden;
    const auto inner = g.intermediate;
    const auto q     = g.query_width();
    const auto k     = g.key_width();
    const auto d     = g.head_dim;
    const auto code  = g.code_size;
    const auto rank  = g.rank;
    const auto taps  = std::uint64_t(g.taps.size());

    m.parameter(c + "/feature_projection", m.handle("fc.weight"), {h, taps * h},
                c + "/target_features");
    m.direct(c + "/context_norm", bf16_of(m, source, "enc.output_norm.weight", {h}), {h});
    m.direct(c + "/final_norm", bf16_of(m, source, "output_norm.weight", {h}), {h});
    for (std::uint64_t layer = 0; layer < g.layers; ++layer) {
        const std::string p      = c + "/layers/" + std::to_string(layer) + "/";
        const std::string bridge = c + "/bridge/layers/" + std::to_string(layer) + "/";
        const std::string blk    = "blk." + std::to_string(layer) + ".";
        m.direct(p + "input_norm", bf16_of(m, source, blk + "attn_norm.weight", {h}), {h});
        m.direct(p + "post_attention_norm", bf16_of(m, source, blk + "ffn_norm.weight", {h}), {h});
        const auto key   = m.handle(blk + "attn_k.weight");
        const auto value = m.handle(blk + "attn_v.weight");
        m.parameter(p + "attention/query", m.handle(blk + "attn_q.weight"), {q, h},
                    p + "query_projection_input");
        m.parameter(p + "attention/key", key, {k, h}, p + "query_projection_input");
        m.parameter(p + "attention/value", value, {k, h}, p + "query_projection_input");
        m.parameter(p + "attention/context_key", key, {k, h}, c + "/context_input");
        m.parameter(p + "attention/context_value", value, {k, h}, c + "/context_input");
        m.direct(p + "attention/query_norm", bf16_of(m, source, blk + "attn_q_norm.weight", {d}),
                 {d});
        m.direct(p + "attention/key_norm", bf16_of(m, source, blk + "attn_k_norm.weight", {d}), {d});
        m.parameter(p + "attention/output", m.handle(blk + "attn_output.weight"), {h, q},
                    p + "attention_output");
        add_swiglu(m, source, p, blk + "ffn_gate.weight", blk + "ffn_up.weight", h, inner,
                   p + "mlp_input");
        m.parameter(p + "mlp/down", m.handle(blk + "ffn_down.weight"), {h, inner},
                    p + "mlp_product");
        m.parameter(bridge + "read", m.handle(blk + "dspine_read.weight"), {code, h},
                    bridge + "read_input");
        m.parameter(bridge + "write", m.handle(blk + "dspine_write.weight"), {h, rank},
                    bridge + "message");
    }
    // The bridge's projection is its three row blocks stacked: the gate, the predecessor gate and
    // the value write. They are one matrix to the Op, so they are stored as one.
    m.parameter(c + "/bridge/projection",
                m.derive("dspine/bridge/projection", {3 * rank, code}, QType::BF16,
                         QuantLayout::Contiguous,
                         [&](std::span<std::byte> out) {
                             const std::uint64_t bytes = rank * code * 2;
                             std::uint64_t at          = 0;
                             for (const auto* name : {"dspine_gate.weight",
                                                      "dspine_pred_gate.weight",
                                                      "dspine_prev.weight"}) {
                                 const auto block = source.bytes(m.stored(name));
                                 if (block.size() != bytes) {
                                     throw ArtifactError(std::string(name) +
                                                         ": the bridge blocks disagree");
                                 }
                                 std::memcpy(out.data() + at, block.data(), block.size());
                                 at += block.size();
                             }
                         }),
                {3 * rank, code}, c + "/bridge/code_input");
    m.direct(c + "/bridge/gate_bias", bf16_of(m, source, "dspine_gate.bias", {rank}), {rank});
    // The head's write is the last layer's write transposed: column r of the stored matrix is the
    // write of rank r, and the consumer reads it as a [h, rank] activation.
    m.direct(c + "/bridge/head_write",
             m.derive("dspine/bridge/head_write", {rank, h}, QType::BF16,
                      QuantLayout::Contiguous,
                      [&](std::span<std::byte> out) {
                          const auto& tensor = m.stored("blk." + std::to_string(g.layers - 1) +
                                                        ".dspine_write.weight");
                          if (tensor.type != QType::BF16) {
                              throw ArtifactError(tensor.name + ": the bridge write must be BF16");
                          }
                          const auto bytes  = source.bytes(tensor);
                          const auto* words = reinterpret_cast<const std::uint16_t*>(bytes.data());
                          auto* out_words   = reinterpret_cast<std::uint16_t*>(out.data());
                          for (std::uint64_t r = 0; r < rank; ++r) {
                              for (std::uint64_t i = 0; i < h; ++i) {
                                  out_words[r * h + i] = words[i * rank + r];
                              }
                          }
                      }),
             {rank, h});
    m.direct(c + "/bridge/sender_table",
             m.derive("dspine/bridge/sender_table", {g.vocab, 2 * rank}, QType::BF16,
                      QuantLayout::Contiguous,
                      [&](std::span<std::byte> out) {
                          write_sender_table(source, m.stored("dspine_codebook.weight"),
                                             m.stored("dspine_pred_gate.weight"),
                                             m.stored("dspine_prev.weight"), g.vocab, code, rank,
                                             out);
                      }),
             {g.vocab, 2 * rank});
}

// The file's directory in the order an artifact lists its objects: ascending offset, which is also
// the order a device upload reads them in.
std::vector<const Tensor*> directory_order(const Tensors& tensors) {
    std::vector<const Tensor*> out;
    out.reserve(tensors.size());
    for (const auto& [name, tensor] : tensors) { out.push_back(&tensor); }
    std::sort(out.begin(), out.end(), [](const Tensor* a, const Tensor* b) {
        return std::tie(a->offset, a->name) < std::tie(b->offset, b->name);
    });
    std::uint64_t end = 0;
    for (const auto* tensor : out) {
        if (tensor->offset < end) {
            throw ArtifactError(tensor->name + ": GGUF tensors overlap; the directory is not an "
                                               "artifact's");
        }
        end = tensor->offset + tensor->bytes;
    }
    return out;
}

} // namespace

bool is_gguf(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { return false; }
    std::array<char, 4> magic{};
    file.read(magic.data(), magic.size());
    return file.gcount() == static_cast<std::streamsize>(magic.size()) &&
           std::equal(magic.begin(), magic.end(), kGgufMagic.begin(),
                      [](char a, std::byte b) { return a == static_cast<char>(b); });
}

artifact::PreparedPayload prepare_draft_gguf(const std::filesystem::path& path) {
    const artifact::InputFile file(path);
    Prefix prefix(file);
    Walk walk(prefix, 0);
    for (const auto expected : kGgufMagic) {
        if (walk.u8() != std::to_integer<std::uint8_t>(expected)) {
            throw ArtifactError(path.string() + ": not a GGUF file");
        }
    }
    const auto version = walk.u32();
    if (version != kGgufVersion) {
        throw ArtifactError(path.string() + ": GGUF version " + std::to_string(version) +
                            " is not supported (3 is)");
    }
    const auto tensor_count = walk.u64();
    const auto entry_count  = walk.u64();
    Table kv;
    for (std::uint64_t i = 0; i < entry_count; ++i) {
        auto key = walk.text();
        const auto type = walk.u32();
        kv.emplace(std::move(key), read_entry(type, walk));
    }
    Tensors tensors;
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        Tensor tensor;
        tensor.name      = walk.text();
        const auto rank  = walk.u32();
        if (!rank || rank > 4) {
            throw ArtifactError(tensor.name + ": GGUF tensor rank " + std::to_string(rank) +
                                " is not supported");
        }
        Shape shape(rank);
        for (std::uint32_t axis = 0; axis < rank; ++axis) { shape[rank - 1 - axis] = walk.u64(); }
        const auto type_id = walk.u32();
        tensor.offset      = walk.u64();
        tensor.type        = format_of(type_id, path, tensor.name);
        tensor.shape       = std::move(shape);
        try {
            tensor.bytes =
                weight_geometry(tensor.type, layout_of(tensor.type), tensor.shape).bytes;
        } catch (const std::exception& error) {
            throw ArtifactError(tensor.name + ": " + error.what());
        }
        const auto name    = tensor.name;
        if (!tensors.emplace(name, std::move(tensor)).second) {
            throw ArtifactError(name + ": duplicate GGUF tensor");
        }
    }
    const auto alignment_entry = kv.find("general.alignment");
    const auto alignment = alignment_entry == kv.end()
                               ? std::uint64_t(32)
                               : static_cast<std::uint64_t>(alignment_entry->second.integer);
    if (!alignment || (alignment & (alignment - 1))) {
        throw ArtifactError(path.string() + ": general.alignment must be a power of two");
    }
    const auto data_offset = artifact::align_up(walk.position(), alignment, "GGUF data section");
    if (data_offset >= file.bytes()) {
        throw ArtifactError(path.string() + ": the GGUF has no data section");
    }
    const auto file_payload_bytes = file.bytes() - data_offset;

    const auto kind     = read_kind(kv, tensors, path);
    const auto geometry = read_geometry(kv, tensors, kind, path);

    Directory directory;
    directory.files.push_back(artifact::FileRecord{std::nullopt, file_payload_bytes, 0});
    Manifest manifest(tensors, std::move(directory));
    manifest.file_payload_bytes = file_payload_bytes;
    for (const auto* tensor : directory_order(tensors)) {
        manifest.objects.emplace(tensor->name, ObjectHandle{manifest.directory.objects.size()});
        manifest.directory.objects.emplace_back(artifact::TensorObject{
            tensor->name, tensor->shape,
            std::string(artifact::format_name(tensor->type)),
            std::string(artifact::layout_name(layout_of(tensor->type))), tensor->offset,
            tensor->bytes, 1});
        manifest.directory.object_index.emplace(tensor->name,
                                               ObjectHandle{manifest.directory.objects.size() - 1});
    }
    if (kind == DraftArtifactKind::DSpine) {
        add_dspine(manifest, Source(file, data_offset), geometry);
    } else {
        add_dflash(manifest, Source(file, data_offset), geometry, kind);
    }

    artifact::Directory built = std::move(manifest.directory);
    built.payload_bytes = artifact::checked_add(file_payload_bytes, manifest.derived.size(),
                                               "GGUF payload");
    built.components.emplace(std::string(draft_artifact_name(kind)),
                             artifact::Component{draft_config(geometry, kind), std::nullopt, {}, {}});
    const auto name = kv.find("general.name");
    built.metadata   = Json{{"name", name == kv.end() ? path.stem().string() : name->second.text}};
    built.provenance = Json{{"source", path.string()},
                            {"container", "gguf"},
                            {"finetune", kv.contains("general.finetune")
                                             ? kv.at("general.finetune").text
                                             : std::string()}};

    artifact::PreparedPayload out;
    out.directory       = std::move(built);
    out.data_offset     = data_offset;
    out.source_alignment = std::min<std::uint64_t>(alignment, 256);
    // A prepared payload has no container id; the file's shape stands in, so a diagnostic can tell
    // two artifacts apart and a reload can compare them.
    std::uint64_t fingerprint = file.bytes();
    for (const auto& [tensor_name, tensor] : tensors) {
        for (const char c : tensor_name) {
            fingerprint = (fingerprint ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
        }
        fingerprint = (fingerprint ^ tensor.offset) * 1099511628211ULL;
    }
    std::memcpy(out.artifact_id.data(), &fingerprint, sizeof(fingerprint));
    std::memcpy(out.artifact_id.data() + sizeof(fingerprint), &data_offset, sizeof(data_offset));
    out.derived = std::move(manifest.derived);
    return out;
}

} // namespace ninfer::models
