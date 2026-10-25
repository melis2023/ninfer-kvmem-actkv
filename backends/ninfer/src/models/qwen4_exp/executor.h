#pragma once

// The forward pass of Qwen3.8-Flash-Next over a loaded Model: the four-stream residual stack in
// FP32, Gated DeltaNet and sparse-attention mixers, the PLE n-gram injection and the 512-expert
// MoE, layer by layer on each layer's stage device, the stack crossing to the next device at a
// stage boundary. The executor owns every sequence's mutable state (recurrent and convolution
// states, paged KV, the indexer's pooled keys, the PLE history and n-gram context) and the
// workspace; the model stays immutable.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/types.h"
#include "models/qwen4_exp/expert_cache.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/ngram_component.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen3_5/program/vision_control.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExecutorOptions {
    std::uint32_t max_context = 32768;
    // Tokens per forward call: a prompt is fed in calls of at most this many.
    std::uint32_t prefill_chunk = 2048;
    std::uint32_t sequences     = 1;
    // The n-gram table of the PLE layer. Empty runs the model without it: the PLE injection is
    // skipped, which is what an all-zero table would give.
    std::optional<NgramTableSource> ngram;
    NgramResidency ngram_residency = NgramResidency::Disk;
    // Host-resident experts only: device memory lent to the expert cache, split evenly over the
    // ranks; kAutoExpertCache takes what each device has free less a margin, 0 none.
    static constexpr std::uint64_t kAutoExpertCache = ~std::uint64_t{0};
    std::uint64_t expert_cache_bytes                = kAutoExpertCache;
    // Decode steps (one token) of device- and host-resident experts replay a CUDA graph per
    // segment of consecutive layers on one device.
    bool cuda_graphs = true;
    // A model loaded with its Vision tower: the most merged tokens the media of one prompt may
    // hold, which sizes the encoder's workspace and the embeddings a prefill reads.
    std::uint32_t vision_max_merged_tokens = 16384;
    // How the sparse-attention layers store their paged KV: any storage kv_cache_append writes
    // for two heads of 256.
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
};

// One media item of a prompt for the Vision tower: its patches and the frontend's control.
struct MediaItem {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

struct ExecutorMemory {
    struct Rank {
        std::uint64_t state_bytes        = 0; // the sequences' state of the rank's layers
        std::uint64_t workspace_bytes    = 0;
        std::uint64_t expert_cache_bytes = 0;
    };
    // Over every device.
    std::uint64_t state_bytes        = 0;
    std::uint64_t workspace_bytes    = 0;
    std::uint64_t expert_cache_bytes = 0;
    std::uint64_t kv_bytes = 0; // the sparse-attention layers' KV and scale pages, in state_bytes
    std::vector<Rank> ranks; // by device rank
};

// One sequence's recurrent state at its position: the Gated DeltaNet and convolution states, the
// indexer's partial block, the PLE history and the n-gram context. Executor::restore continues
// the sequence from that position after it moved on; the paged KV and the indexer's pooled keys
// of earlier positions are not copied, since a sequence only writes positions at or past its own.
// Device buffers on each layer's device, sized at the first snapshot and reused after.
struct SequenceSnapshot {
    struct Layer {
        std::vector<DeviceBuffer> buffers; // ssm, conv, tail, history; empty where absent
    };
    std::vector<Layer> layers;
    std::uint32_t position = 0;
    NgramContext context;
};

class Executor {
public:
    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;

    // Empties the sequence: position zero, zero states, a fresh n-gram context.
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;

    // Runs `tokens` (at most prefill_chunk) at the sequence's next positions and leaves the logits
    // of the last `logit_rows` of them in logits(), BF16 [vocab, logit_rows] on the head device.
    // Work is queued on the device streams; logits() is ready on head_stream().
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows);
    // Runs one token of each of `sequences` (distinct) at its next position in one pass, and
    // leaves their logits in logits(), one column per sequence in that order. The experts read
    // their weights once for the whole batch; each sequence's mixers run on its own state.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // Copies the sequence's recurrent state into `out`, and back; both wait for the copies.
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // Vision: encodes the media of the prompt the (just reset) sequence prefills next, whose image
    // and video tokens then take the items' merged embeddings in place of the token embedding.
    // `rope_positions` holds the prompt's three RoPE position axes, axis-major [3, tokens]; every
    // later position rotates at its index plus `rope_delta`. The embeddings are kept until another
    // sequence's media replace them, so the prompt must be prefilled before that. Returns once the
    // tower has run.
    void set_media(std::uint32_t sequence, std::span<const MediaItem> items,
                   std::vector<std::int32_t> rope_positions, std::int32_t rope_delta);
    [[nodiscard]] bool vision() const noexcept;
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;
    [[nodiscard]] std::size_t head_rank() const noexcept;
    [[nodiscard]] ExpertCacheStats expert_cache_stats() const noexcept;
    [[nodiscard]] cudaStream_t head_stream() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
