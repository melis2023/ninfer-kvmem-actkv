#pragma once

// Attached LoRA adapters: a llama.cpp-compatible GGUF adapter file read beside the artifact.
// The adapter is a runtime overlay, never a weight of the artifact: `attach_lora` uploads each
// projection's FP16 factors once, and every forward pass that uses the projection adds
// `scale * B(A x)` to its result.

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5 {

// The projections a text-block adapter may carry. The order is the load order. `AttentionGate` is
// the gated-attention query gate, absent from plain Qwen attention.
enum class LoraProjection : std::uint8_t {
    Query = 0,
    Key,
    Value,
    Output,
    AttentionGate,
    FfnGate,
    FfnUp,
    FfnDown,
};

inline constexpr std::size_t kLoraProjectionCount = 8;

[[nodiscard]] const char* lora_projection_name(LoraProjection projection) noexcept;

// One factor exactly as the file stores it: `ne0`/`ne1` are the ggml dimensions (ne0 varies
// fastest) and `data` holds `ne0 * ne1` FP16 elements in that order.
struct LoraMatrixHost {
    std::int32_t ne0 = 0;
    std::int32_t ne1 = 0;
    std::vector<std::uint16_t> data;

    [[nodiscard]] bool present() const noexcept { return ne0 > 0 && ne1 > 0; }
    [[nodiscard]] std::size_t bytes() const noexcept { return data.size() * sizeof(std::uint16_t); }
};

struct LoraPairHost {
    LoraMatrixHost a;
    LoraMatrixHost b;
    [[nodiscard]] bool present() const noexcept { return a.present() && b.present(); }
};

struct LoraLayerHost {
    std::array<LoraPairHost, kLoraProjectionCount> projections;
};

struct LoraAdapter {
    std::string path;
    // `adapter.lora.alpha` from the file; 0 when the file does not carry one, which leaves the
    // effective scale at `lora_scale` alone (llama.cpp's `get_scale`).
    float alpha = 0.0F;
    std::vector<LoraLayerHost> layers;
    std::vector<std::string> ignored_tensors;

    [[nodiscard]] std::size_t bytes() const;
    [[nodiscard]] std::size_t projection_count() const;
};

// Reads a GGUF LoRA adapter. Throws when the file is not a GGUF adapter, when a factor uses a
// tensor type other than F32/F16/BF16, or when the file holds no text-block projection.
[[nodiscard]] LoraAdapter read_lora_adapter(const std::filesystem::path& path,
                                            std::size_t layer_count);

} // namespace ninfer::models::qwen3_5
