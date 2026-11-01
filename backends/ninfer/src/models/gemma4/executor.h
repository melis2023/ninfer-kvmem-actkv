#pragma once

// The forward pass of Gemma 4 over a loaded Model. Each sequence owns a cache per block, F16 or a
// quantized format (kv_format.h): keys and values in a ring of the last window (and a round's
// drafts) for the sliding-window blocks, the values of every position for the full-attention blocks,
// whose keys (K = V) are derived from them where they are read. Prompt chunks run llama.cpp's tensor-core flash attention; decode steps run
// one token of each of several sequences at once, replayed from a CUDA graph per batch size.
//
// With draft_tokens, the model's assistant drafts: from a sequence's next token (its anchor) and the
// target's final hidden state before it, a chain of steps at the anchor's position, each attending
// over the target's caches, proposes the tokens after it; verify() runs the anchor and the drafts
// of each sequence as decode rows, and commit() keeps the first ones.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/expert_profile.h"
#include "models/gemma4/kv_format.h"
#include "models/gemma4/model.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::gemma4 {

// The device lacks the memory the options need (the caches, or the workspace and CUDA graphs
// beside them); the message names the amounts.
struct InsufficientMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ExecutorOptions {
    std::uint32_t max_context   = 32768;
    std::uint32_t prefill_chunk = 1024; // prompt tokens per forward()
    std::uint32_t sequences     = 1;
    bool cuda_graphs            = true;
    // Drafts a speculative round may propose per sequence (0: no assistant); a verification's rows,
    // the anchors and drafts of all its sequences, are at most 8.
    std::uint32_t draft_tokens = 0;
    // Prompt chunks may carry media (a model loaded with its vision tower): the most patches one
    // media unit (an image or a video frame) has, which also bounds a batch of units the device
    // tower encodes at once. 0: text only.
    std::uint32_t vision_max_patches = 0;
    // The caches' storage (kv_format.h); BFloat16 selects the F16 cache.
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    // One checkpoint of every sequence's rings (snapshot/restore), allocated with the caches.
    bool checkpoints = true;
    std::optional<std::uint64_t> expert_cache_bytes;
    // Host or disk experts: the cache's first contents, the most routed of these counts
    // ([expert layer][expert]) once warm-up is done; none leaves the cache empty then.
    std::optional<ExpertRouteCounts> expert_profile;
    // The share 0..1 of a decode or verification call's missing experts the GPU copies; the CPU
    // computes the rest (cpu_experts.h). Unset: each call splits its misses by measured CPU and
    // bus times. CPU threads: 0 for one per physical core.
    std::optional<float> expert_dma_share;
    std::uint32_t expert_cpu_threads = 0;
    // YaRN past the model's native window (a factor above one) for the full-attention blocks'
    // rotation, the assistant's included: their proportional schedule takes YaRN's frequencies,
    // and the queries the square of its attention factor. The sliding-window blocks never see a
    // distance past their window and keep their rotation. Inactive keeps every rotation as is.
    ops::RopeYarn rope_yarn;
};

// One media item's units in a prompt chunk: consecutive units (an image, or frames of a video) of
// grid_w x grid_h patches, and the chunk row of each unit's first soft token; a unit's
// (grid_w / pooling) x (grid_h / pooling) soft tokens fill the rows from there. The units come as
// their patches, BF16 [units][grid_h * grid_w][patch features] in host memory, which the device
// tower encodes, or as soft-token embeddings already computed (the CPU tower), BF16
// [units][tokens][hidden] in host memory.
struct ChunkMedia {
    std::int32_t grid_w = 0, grid_h = 0;
    std::span<const std::uint16_t> patches;
    std::span<const std::uint16_t> embeddings;
    std::vector<std::uint32_t> rows;
};

struct ExecutorMemory {
    std::uint64_t kv_bytes        = 0;
    // Beside an elastic expert cache, the full-attention caches grow with their sequences: the
    // bytes backed now, of kv_bytes at most.
    std::uint64_t kv_backed_bytes = 0;
    bool kv_grows                 = false;
    std::uint64_t workspace_bytes = 0;
    // A device vision tower: the scratch of one encode within workspace_bytes, and the device bytes
    // the tower owns beside the model's weights (the streamed window and its resident copies).
    std::uint64_t vision_encode_bytes = 0;
    std::uint64_t vision_owned_bytes  = 0;
    ExpertDeliveryMemory experts;
};

// A sequence's checkpoint: its sliding-window rings, kept in the executor's checkpoint storage of
// that sequence (one per sequence), and its position, which Executor::restore resumes from. The
// full-attention caches of earlier positions are not copied, since a sequence only writes positions
// at or past its own.
struct SequenceSnapshot {
    std::uint32_t position = 0;
};

// A sequence's state in host memory, laid out as image_bytes() sizes it: its sliding-window rings
// and its full-attention rows below `position`. The context cache keeps prefixes this way beyond
// the device's sequences; the header travels beside the bytes.
struct SequenceImage {
    std::uint32_t position = 0;
};

class Executor {
public:
    // Logit rows forward() may leave at once.
    static constexpr std::uint32_t kMaxLogitRows = 256;

    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;
    // Position 0; growing caches give the sequence's rows back (ExecutorMemory::kv_grows).
    void reset(std::uint32_t sequence);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;
    // Loads the kernels every route reaches, and captures the decode graphs.
    void warm_up();
    // Check after a stream completion and before output publication.
    void check_experts() const;
    [[nodiscard]] ExpertDeliveryStats expert_stats() const;
    // Every route the decode and verification calls took through each expert layer's experts
    // since warm-up, the form a recorded routing profile takes.
    [[nodiscard]] ExpertRouteCounts expert_totals() const;
    // The CPU's share of missing experts: its threads and backend, or empty without one.
    [[nodiscard]] std::string expert_cpu_description() const;
    // Runs up to prefill_chunk tokens at the sequence's next positions and leaves the logits of the
    // last `logit_rows` (at most kMaxLogitRows) in logits(). The soft tokens of `media` take the
    // place of their rows' placeholder embeddings, and with the vision config's bidirectional
    // attention a unit's soft tokens see each other in the sliding-window blocks; a unit must lie
    // whole in one chunk.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const ChunkMedia> media = {});
    // Branches of a text prompt in one pass, each continuing the sequence from its position:
    // branch b runs `lengths[b]` of `tokens` (in branch order, at most prefill_chunk in all, at
    // most kMaxLogitRows branches) at the positions after the sequence's, seeing the sequence's
    // keys and its own, no other branch's. Leaves the logits of each branch's last token in
    // logits(), one column per branch in order. The sequence keeps its position and window.
    void forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths);
    // Whether prompt chunks may carry media.
    [[nodiscard]] bool vision() const noexcept;
    // The device time the last forward() spent on its media (encoding and placing its soft
    // tokens); waits for it.
    [[nodiscard]] double last_media_seconds() const;
    // One token of each of `sequences` (distinct) at its next position; logits one column each.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // BF16 [vocab, rows], ready on head_stream().
    [[nodiscard]] Tensor logits(std::uint32_t rows) const;

    // The assistant (draft_tokens > 0). A sequence can draft once a prompt chunk, a decode step or
    // a commit has left the target's hidden state after its last fed token.
    [[nodiscard]] std::uint32_t draft_tokens() const noexcept;
    [[nodiscard]] bool can_draft(std::uint32_t sequence) const;
    // `steps` (1..draft_tokens) greedy drafts after each sequence's anchor, the token its next
    // position takes: drafts[j * steps + i] is sequence j's i-th draft.
    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::span<std::int32_t> drafts, std::uint32_t steps);
    // tokens.size() / sequences.size() tokens of each sequence (its anchor, then its drafts) at its
    // next positions, as decode rows: logits one column each, sequence by sequence. The positions
    // advance only with commit().
    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // Keeps the first kept[j] (1..width) of sequence j's verified tokens.
    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept);
    [[nodiscard]] cudaStream_t head_stream() const noexcept;
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // Bytes of an image at `position`.
    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const;
    // Copies the sequence's state into `pages`, pinned host memory that holds image_bytes() of its
    // position in consecutive slices of `page_bytes` (the last one may be partly used): its live
    // state, or the state at `at`, its checkpoint, whose positions it has not written since.
    // Snapshots, restores and image copies are enqueued on the executor's stream, ordered with
    // the work around them, and the host does not wait for them: device work enqueued later sees
    // them done, and pages are only to be rewritten by later image copies.
    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes);
    // Makes the sequence hold the image's state.
    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::gemma4
