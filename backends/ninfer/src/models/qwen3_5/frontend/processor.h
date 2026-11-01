#pragma once

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

class MediaPreprocessCache;

enum class ProcessorErrorKind {
    BudgetExceeded,
    ContextLengthExceeded,
    InvalidMedia,
};

class ProcessorError final : public std::runtime_error {
public:
    ProcessorError(ProcessorErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ProcessorErrorKind kind() const noexcept { return kind_; }

private:
    ProcessorErrorKind kind_;
};

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PreprocessStats {
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::uint64_t attention_pairs        = 0;
    std::size_t prompt_tokens            = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;

    [[nodiscard]] std::string summary() const;
};

// Gemma 4's media geometry (the release's processor_config.json): patches of `patch` pixels, an
// average over pooling x pooling patches per soft token, and a soft-token budget per image and per
// video frame out of kGemmaSoftTokenBudgets. A medium is resized, aspect kept, to the largest size
// whose sides are multiples of pooling * patch within budget * pooling^2 patches.
inline constexpr std::array<std::uint32_t, 5> kGemmaSoftTokenBudgets{70, 140, 280, 560, 1120};

struct GemmaMediaOptions {
    int patch                  = 16;
    int pooling                = 3;
    std::uint32_t image_tokens = 280;  // an image's budget at auto detail
    std::uint32_t video_tokens = 70;   // a video frame's budget
    std::uint32_t max_tokens   = 1120; // the server's bound on any budget (--vision-max-merged)
    int frames                 = 32;   // video frames sampled, i * frames_in_video / 32
};

// The largest budget of kGemmaSoftTokenBudgets at most `bound`; throws when every one exceeds it.
[[nodiscard]] std::uint32_t gemma_budget_at_most(std::uint64_t bound);

struct ProcessorOptions {
    // Present for Gemma 4, whose media take its own geometry and placeholders.
    std::optional<GemmaMediaOptions> gemma;
    std::uint64_t image_min_pixels = 32ULL * 32ULL;
    std::uint64_t image_max_pixels = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels = 4ULL * 1024ULL * 1024ULL;
    // Encoded bytes are aggregate per prompt. Decode limits are per item; the fixed worker pool
    // bounds concurrently decoded media.
    std::size_t max_encoded_media_bytes    = kMaximumPromptMediaBytes;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::uint64_t max_raw_patches          = kMaximumPromptVisionRawPatches;
    std::uint64_t max_vision_tokens        = kMaximumPromptVisionTokens;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
};

// Clamps the smart-resize pixel ceilings so one image, or one two-frame video group, never
// exceeds merged_tokens after resizing; oversized media downscales instead of being rejected.
void bound_merged_tokens(ProcessorOptions& options, std::uint64_t merged_tokens);

struct ProcessedInput {
    bool starts_in_reasoning = false;
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<VisionItem> vision_items;
    // One immutable row-major [raw_patches, 1536] payload per Vision item.
    std::vector<std::shared_ptr<const qwen3_5::PreparedMediaPayload>> media_payloads;
    std::optional<TokenSpan> memory_query;
    std::optional<TokenSpan> first_user_region;
    std::optional<TokenSpan> tools_region;
    std::optional<TokenSpan> action_region;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    PreprocessStats stats;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;
};

struct EncodedChat {
    std::vector<int> input_ids;

    struct MediaTokenRun {
        TokenSpan tokens;
        Modality modality       = Modality::Image;
        std::size_t item_index  = 0;
        std::size_t frame_index = 0;
    };

    std::vector<MediaTokenRun> media_token_runs;
    std::optional<TokenSpan> memory_query;
    std::optional<TokenSpan> first_user_region;
    std::optional<TokenSpan> tools_region;
    std::optional<TokenSpan> action_region;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
};

// The placeholder token an image and a video soft token is written as.
struct MediaPads {
    std::string_view image;
    std::string_view video;
};
inline constexpr MediaPads kQwenMediaPads{"<|image_pad|>", "<|video_pad|>"};
inline constexpr MediaPads kGemmaMediaPads{"<|image|>", "<|video|>"};

EncodedChat
encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered,
                     std::size_t maximum_tokens = std::numeric_limits<std::size_t>::max(),
                     MediaPads pads             = kQwenMediaPads);

class Processor {
public:
    Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              ProcessorOptions options, std::shared_ptr<MediaPreprocessCache> media_cache);

    [[nodiscard]] std::size_t count_tokens(std::vector<ChatMessage> messages,
                                           ChatRenderOptions render_options  = {},
                                           const PreparationControl& control = {}) const;

    ProcessedInput
    process(std::vector<ChatMessage> messages, ChatRenderOptions render_options = {},
            const PreparationControl& control = {},
            std::size_t maximum_prompt_tokens = std::numeric_limits<std::size_t>::max()) const;

private:
    const Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    ProcessorOptions options_;
    std::shared_ptr<MediaPreprocessCache> media_cache_;
};

} // namespace ninfer::models::qwen3_5::frontend
