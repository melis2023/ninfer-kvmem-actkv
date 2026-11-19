// Reads a llama.cpp-compatible GGUF LoRA adapter (llama.cpp convert_lora_to_gguf.py layout):
// `blk.<n>.<projection>.weight.lora_a` / `.lora_b`, plus the PEFT spellings as a fallback.
// The file is streamed, never mapped: only the factors of the text blocks are kept.
#include "models/qwen3_5/lora_adapter.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <cuda_fp16.h>

namespace ninfer::models::qwen3_5 {
namespace {

constexpr std::uint32_t kGgufMagic = 0x46554747U; // "GGUF"
constexpr std::uint32_t kLoRaLzma      = 0;
constexpr std::size_t kDefaultAlignment = 32;

enum class GgufType : std::uint32_t {
    Uint8 = 0,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
};

enum class GgmlType : std::uint32_t {
    F32  = 0,
    F16  = 1,
    BF16 = 30,
};

class Stream {
public:
    explicit Stream(const std::filesystem::path& path) {
        file_.open(path, std::ios::binary);
        if (!file_) { throw std::runtime_error("lora adapter: cannot open " + path.string()); }
        file_.seekg(0, std::ios::end);
        size_ = static_cast<std::uint64_t>(file_.tellg());
        file_.seekg(0, std::ios::beg);
    }

    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] std::uint64_t position() const noexcept { return position_; }

    void seek(std::uint64_t offset) {
        if (offset > size_) { throw std::runtime_error("lora adapter: truncated file"); }
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        position_ = offset;
    }

    void read(void* destination, std::size_t count) {
        if (count > size_ - position_) { throw std::runtime_error("lora adapter: truncated file"); }
        file_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (!file_) { throw std::runtime_error("lora adapter: read failed"); }
        position_ += count;
    }

    void skip(std::uint64_t count) {
        if (count > size_ - position_) { throw std::runtime_error("lora adapter: truncated file"); }
        position_ += count;
        file_.seekg(static_cast<std::streamoff>(position_), std::ios::beg);
    }

    [[nodiscard]] std::uint32_t u32() {
        std::uint32_t value = 0;
        read(&value, sizeof(value));
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0;
        read(&value, sizeof(value));
        return value;
    }

    [[nodiscard]] std::string string() {
        const std::uint64_t length = u64();
        if (length > size_ - position_) { throw std::runtime_error("lora adapter: bad string"); }
        std::string value(static_cast<std::size_t>(length), '\0');
        if (length != 0) { read(value.data(), static_cast<std::size_t>(length)); }
        return value;
    }

private:
    std::ifstream file_;
    std::uint64_t size_     = 0;
    std::uint64_t position_ = 0;
};

std::uint64_t type_size(GgufType type) noexcept {
    switch (type) {
    case GgufType::Uint8:
    case GgufType::Int8:
    case GgufType::Bool: return 1;
    case GgufType::Uint16:
    case GgufType::Int16: return 2;
    case GgufType::Uint32:
    case GgufType::Int32:
    case GgufType::Float32: return 4;
    case GgufType::Uint64:
    case GgufType::Int64:
    case GgufType::Float64: return 8;
    default: throw std::runtime_error("lora adapter: unsupported metadata value type");
    }
}

void skip_value(Stream& stream, GgufType type, int depth) {
    if (depth > 1) { throw std::runtime_error("lora adapter: nested array metadata"); }
    switch (type) {
    case GgufType::String: (void)stream.string(); return;
    case GgufType::Array: {
        const auto element = static_cast<GgufType>(stream.u32());
        const std::uint64_t count = stream.u64();
        if (element == GgufType::String) {
            for (std::uint64_t index = 0; index < count; ++index) { (void)stream.string(); }
            return;
        }
        const std::uint64_t element_size = type_size(element);
        stream.skip(element_size * count);
        return;
    }
    default: stream.skip(type_size(type)); return;
    }
}

struct TensorInfo {
    std::string name;
    std::uint64_t ne0 = 0;
    std::uint64_t ne1 = 1;
    GgmlType type     = GgmlType::F16;
    std::uint64_t offset = 0;
};

bool ends_with(std::string_view text, std::string_view suffix) noexcept {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(std::string_view text, std::string_view needle) noexcept {
    return text.find(needle) != std::string_view::npos;
}

// `blk.12.` or `layers.12.`; -1 when the name carries no block index.
int layer_index(std::string_view name) noexcept {
    for (const std::string_view marker : {std::string_view{"blk."}, std::string_view{"layers."}}) {
        const std::size_t at = name.find(marker);
        if (at == std::string_view::npos) { continue; }
        std::size_t index = at + marker.size();
        int value         = 0;
        bool digits       = false;
        while (index < name.size() && name[index] >= '0' && name[index] <= '9') {
            value  = value * 10 + (name[index] - '0');
            digits = true;
            ++index;
        }
        if (digits && index < name.size() && name[index] == '.') { return value; }
    }
    return -1;
}

// Returns false for a factor this build ignores (embedding, head, MoE router, attention conv).
bool projection_of(std::string_view base, LoraProjection& projection) noexcept {
    struct Entry {
        std::string_view token;
        LoraProjection projection;
    };
    // MoE routers and the linear-attention path have no text-block slot in this engine.
    for (const std::string_view skip : {std::string_view{"attn_qkv"}, std::string_view{"ffn_gate_inp"},
                                        std::string_view{"ffn_gate_exps"}, std::string_view{"ffn_up_exps"},
                                        std::string_view{"ffn_down_exps"}, std::string_view{"attn_gate_exps"},
                                        std::string_view{"ssm_"}, std::string_view{"conv1d"},
                                        std::string_view{"token_embd"}}) {
        if (contains(base, skip)) { return false; }
    }
    static constexpr Entry entries[] = {
        {"attn_q", LoraProjection::Query},        {"q_proj", LoraProjection::Query},
        {"attn_k", LoraProjection::Key},          {"k_proj", LoraProjection::Key},
        {"attn_v", LoraProjection::Value},        {"v_proj", LoraProjection::Value},
        {"attn_output", LoraProjection::Output},  {"o_proj", LoraProjection::Output},
        {"attn_gate", LoraProjection::AttentionGate},
        {"ffn_gate", LoraProjection::FfnGate},    {"gate_proj", LoraProjection::FfnGate},
        {"ffn_up", LoraProjection::FfnUp},        {"up_proj", LoraProjection::FfnUp},
        {"ffn_down", LoraProjection::FfnDown},    {"down_proj", LoraProjection::FfnDown},
    };
    for (const Entry& entry : entries) {
        if (contains(base, entry.token)) {
            projection = entry.projection;
            return true;
        }
    }
    return false;
}

// Reads one factor and narrows F32/BF16 to the FP16 the Op multiplies with.
void read_matrix(Stream& stream, const TensorInfo& info, std::uint64_t data_start,
                 LoraMatrixHost& matrix) {
    const std::uint64_t elements = info.ne0 * info.ne1;
    if (elements == 0 || elements > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("lora adapter: factor " + info.name + " has an impossible shape");
    }
    if (info.ne0 > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        info.ne1 > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("lora adapter: factor " + info.name + " exceeds the integer domain");
    }
    matrix.ne0 = static_cast<std::int32_t>(info.ne0);
    matrix.ne1 = static_cast<std::int32_t>(info.ne1);
    matrix.data.resize(static_cast<std::size_t>(elements));
    stream.seek(data_start + info.offset);
    if (info.type == GgmlType::F16) {
        stream.read(matrix.data.data(), matrix.data.size() * sizeof(std::uint16_t));
        return;
    }
    if (info.type == GgmlType::F32) {
        std::vector<float> source(static_cast<std::size_t>(elements));
        stream.read(source.data(), source.size() * sizeof(float));
        for (std::size_t index = 0; index < source.size(); ++index) {
            const __half value = __float2half_rn(source[index]);
            std::memcpy(&matrix.data[index], &value, sizeof(std::uint16_t));
        }
        return;
    }
    if (info.type == GgmlType::BF16) {
        std::vector<std::uint16_t> source(static_cast<std::size_t>(elements));
        stream.read(source.data(), source.size() * sizeof(std::uint16_t));
        for (std::size_t index = 0; index < source.size(); ++index) {
            std::uint32_t wide = static_cast<std::uint32_t>(source[index]) << 16U;
            float value        = 0.0F;
            std::memcpy(&value, &wide, sizeof(float));
            const __half half  = __float2half_rn(value);
            std::memcpy(&matrix.data[index], &half, sizeof(std::uint16_t));
        }
        return;
    }
    throw std::runtime_error("lora adapter: factor " + info.name +
                             " is not F32, F16 or BF16 (only dense factors are supported)");
}

} // namespace

const char* lora_projection_name(LoraProjection projection) noexcept {
    switch (projection) {
    case LoraProjection::Query: return "attn_q";
    case LoraProjection::Key: return "attn_k";
    case LoraProjection::Value: return "attn_v";
    case LoraProjection::Output: return "attn_output";
    case LoraProjection::AttentionGate: return "attn_gate";
    case LoraProjection::FfnGate: return "ffn_gate";
    case LoraProjection::FfnUp: return "ffn_up";
    case LoraProjection::FfnDown: return "ffn_down";
    }
    return "unknown";
}

std::size_t LoraAdapter::bytes() const {
    std::size_t total = 0;
    for (const LoraLayerHost& layer : layers) {
        for (const LoraPairHost& pair : layer.projections) {
            total += pair.a.bytes() + pair.b.bytes();
        }
    }
    return total;
}

std::size_t LoraAdapter::projection_count() const {
    std::size_t total = 0;
    for (const LoraLayerHost& layer : layers) {
        for (const LoraPairHost& pair : layer.projections) {
            total += pair.present() ? 1U : 0U;
        }
    }
    return total;
}

LoraAdapter read_lora_adapter(const std::filesystem::path& path, std::size_t layer_count) {
    Stream stream(path);
    if (stream.u32() != kGgufMagic) {
        throw std::runtime_error("lora adapter: " + path.string() + " is not a GGUF file");
    }
    const std::uint32_t version = stream.u32();
    if (version < 2 || version > 3) {
        throw std::runtime_error("lora adapter: unsupported GGUF version " + std::to_string(version));
    }
    const std::uint64_t tensor_count = stream.u64();
    const std::uint64_t field_count  = stream.u64();
    // A GGUF with an implausible header is a mis-typed artifact, not an adapter.
    if (tensor_count > (1U << 20) || field_count > (1U << 20)) {
        throw std::runtime_error("lora adapter: implausible GGUF header");
    }

    LoraAdapter adapter;
    adapter.path = path.string();
    std::uint64_t alignment = kDefaultAlignment;
    for (std::uint64_t field = 0; field < field_count; ++field) {
        const std::string key   = stream.string();
        const auto type         = static_cast<GgufType>(stream.u32());
        if (key == "adapter.lora.alpha" && type == GgufType::Float32) {
            float alpha = 0.0F;
            stream.read(&alpha, sizeof(alpha));
            adapter.alpha = alpha;
            continue;
        }
        if (key == "general.alignment" && type == GgufType::Uint32) {
            alignment = stream.u32();
            if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
                throw std::runtime_error("lora adapter: general.alignment is not a power of two");
            }
            continue;
        }
        skip_value(stream, type, 0);
    }

    std::vector<TensorInfo> tensors;
    tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        TensorInfo info;
        info.name              = stream.string();
        const std::uint32_t rank_count = stream.u32();
        if (rank_count == 0 || rank_count > 4) {
            throw std::runtime_error("lora adapter: factor " + info.name + " has an invalid rank");
        }
        std::uint64_t dims[4] = {1, 1, 1, 1};
        for (std::uint32_t dim = 0; dim < rank_count; ++dim) { dims[dim] = stream.u64(); }
        info.ne0    = dims[0];
        info.ne1    = rank_count > 1 ? dims[1] : 1;
        info.type   = static_cast<GgmlType>(stream.u32());
        info.offset = stream.u64();
        tensors.push_back(std::move(info));
    }

    const std::uint64_t data_start = (stream.position() + alignment - 1) / alignment * alignment;
    adapter.layers.resize(layer_count);

    for (const TensorInfo& info : tensors) {
        const bool is_a = ends_with(info.name, ".lora_a");
        const bool is_b = ends_with(info.name, ".lora_b");
        if (!is_a && !is_b) {
            adapter.ignored_tensors.push_back(info.name);
            continue;
        }
        const std::string_view base(info.name.data(), info.name.size() - std::string_view(".lora_a").size());
        const int layer = layer_index(base);
        if (layer < 0 || static_cast<std::size_t>(layer) >= layer_count) {
            adapter.ignored_tensors.push_back(info.name);
            continue;
        }
        LoraProjection projection{};
        if (!projection_of(base, projection)) {
            adapter.ignored_tensors.push_back(info.name);
            continue;
        }
        LoraPairHost& pair = adapter.layers[static_cast<std::size_t>(layer)]
                                 .projections[static_cast<std::size_t>(projection)];
        LoraMatrixHost& matrix = is_a ? pair.a : pair.b;
        if (matrix.present()) {
            throw std::runtime_error("lora adapter: duplicate factor " + info.name);
        }
        read_matrix(stream, info, data_start, matrix);
    }

    if (adapter.projection_count() == 0) {
        throw std::runtime_error("lora adapter: " + path.string() +
                                 " carries no text-block projection this engine can attach");
    }
    return adapter;
}

} // namespace ninfer::models::qwen3_5
