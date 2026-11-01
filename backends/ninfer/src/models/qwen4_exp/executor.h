#pragma once

// The forward pass of Qwen3.8-Flash-Next over a loaded Model: the four-stream residual stack in
// FP32, Gated DeltaNet and sparse-attention mixers, the PLE n-gram injection and the 512-expert
// MoE, layer by layer on each layer's stage device, the stack crossing to the next device at a
// stage boundary. The executor owns every sequence's mutable state (recurrent and convolution
// states, paged KV, the indexer's pooled keys, the PLE history and n-gram context) and the
// workspace; the model stays immutable.
//
// MTP speculative decoding (a model loaded with its MTP block and ExecutorOptions::draft_tokens):
// the MTP block drafts from the stack the target left at a sequence's last position and the token
// that follows (vLLM's cells: cell i pairs the target's pre-mixer stack of position i with token
// i + 1 and rotates at i), verify() runs a sequence's anchor and drafts without committing them,
// and commit() keeps a prefix: the Gated DeltaNet layers replay their recorded transitions over it
// (gdn_replay_fold), the sparse-attention indexer tails and the PLE history advance over it from
// what the verification recorded, and the MTP block catches up over the committed cells. Every
// other pass (a prompt chunk, a plain decode step) advances the MTP block too, so a sequence can
// draft whenever it decodes.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/rope.h"
#include "ninfer/engine_options.h"
#include "ninfer/kv_cache_storage.h"
#include "ninfer/observability/resources.h"
#include "models/qwen4_exp/expert_cache.h"
#include "models/qwen4_exp/hybrid_experts.h"
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
    NgramReadOptions ngram_read;
    // Host-resident experts only: device memory lent to the expert cache, split evenly over the
    // ranks; kAutoExpertCache takes what each device has free less a margin, 0 none.
    static constexpr std::uint64_t kAutoExpertCache = ~std::uint64_t{0};
    std::uint64_t expert_cache_bytes                = kAutoExpertCache;
    HybridExpertOptions hybrid_experts;
    // Decode steps (one token) of device- and host-resident experts replay a CUDA graph per
    // segment of consecutive layers on one device.
    bool cuda_graphs = true;
    // A model loaded with its Vision tower: the most merged tokens the media of one prompt may
    // hold, which sizes the encoder's workspace and the embeddings a prefill reads.
    std::uint32_t vision_max_merged_tokens = 16384;
    // How the sparse-attention layers store their paged KV: any storage kv_cache_append writes
    // for two heads of 256.
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    // Positions of each sparse-attention layer's KV a sequence keeps in device memory; 0 keeps
    // all of them. Fewer than max_context streams those layers (ops/kv_stream.h): their paged KV
    // lives in pinned host memory, and the device keeps that many positions' blocks (at least
    // what one decode or verification call reads), the ones the indexer selected most recently.
    // A prompt chunk reads one layer's KV from a staging copy on the device.
    std::uint32_t kv_resident = 0;
    // MTP drafts per speculative round (1..15), for a model loaded with its MTP block; 0 runs
    // without speculation.
    std::uint32_t draft_tokens = 0;
    // Absolute greedy-draft probability floor; zero disables confidence computation.
    float draft_min_p = 0;
    // YaRN past the model's native window (a factor above one), for the sparse-attention layers'
    // RoPE, their indexer's and the MTP block's alike; inactive keeps the unscaled rotation.
    ops::RopeYarn rope_yarn;
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
    // The sparse-attention layers' KV and scale pages on the device (a streamed layer's pool), in
    // state_bytes, and a streamed layer's pinned host copies.
    std::uint64_t kv_bytes      = 0;
    std::uint64_t host_kv_bytes = 0;
    bool streamed_kv            = false;
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
    // The MTP block: its indexer tail and the stack its next cell starts from.
    std::vector<DeviceBuffer> mtp;
    bool mtp_follows       = false;
    std::uint32_t position = 0;
    NgramContext context;
};

// A sequence's whole state at a position, in host memory: the paged KV and the indexer's pooled
// keys of the positions before it, the recurrent state there, and the MTP block's, in the byte
// layout image_bytes() sizes. The context cache keeps prefixes this way beyond the device's
// sequences. The header travels beside the bytes.
struct SequenceImage {
    std::uint32_t position = 0;
    NgramContext context;
    bool mtp_follows = false;
};

class Executor {
public:
    Executor(const Model& model, DeviceContext& device, ExecutorOptions options);
    ~Executor();
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    [[nodiscard]] const ExecutorOptions& options() const noexcept;
    [[nodiscard]] ExecutorMemory memory() const noexcept;
    // The n-gram table's reads since startup, and the RAM its resident rows take.
    [[nodiscard]] NgramTableStats ngram_stats();
    [[nodiscard]] std::uint64_t ngram_resident_bytes() const noexcept;

    // Empties the sequence: position zero, zero states, a fresh n-gram context.
    void reset(std::uint32_t sequence);
    // Discards a pass interrupted inside a layer, draining its row reads and device work before
    // releasing provisional state. The caller must discard this sequence's reusable prefix.
    void abort(std::uint32_t sequence);
    // Rethrows a failure the staged misses' service recorded while serving a pass's layers (a
    // failed expert read among them). Such a pass ran without some missing experts, so call this
    // once its results are on the host and before they are used. No effect without staged misses.
    void raise_service_failure();
    // Runs the routes requests take on sequence 0 once and resets it, so the first request does
    // not pay for loading their kernels (CUDA loads each at its first launch): one token, a
    // verification's width, a full prefill chunk (16 tokens with host or disk experts, which a
    // chunk would copy or read whole), a decode step and, with drafts, a draft and a commit.
    void warm_up();
    // After warm_up: gives the host expert cache what each device still has free beyond
    // `keep_free_bytes`, as more slots. No effect without a host expert cache.
    void grow_expert_cache(std::uint64_t keep_free_bytes);
    [[nodiscard]] std::uint32_t position(std::uint32_t sequence) const;

    // Prompt tokens one forward() may take: prefill_chunk, or with layer-major spans (one stage,
    // GGUF host experts) several chunks, which forward() runs layer by layer so that each layer's
    // uncached experts cross the bus once for all of them.
    [[nodiscard]] std::uint32_t prompt_step() const noexcept;
    // Runs `tokens` (at most prompt_step(); beyond prefill_chunk not a media prompt's) at the
    // sequence's next positions and leaves the logits of the last `logit_rows` of them in
    // logits(), BF16 [vocab, logit_rows] on the head device. Work is queued on the device
    // streams; logits() is ready on head_stream(). Once the table rows of `tokens` are read, the
    // rows of `next` (the known tokens that follow, at most prompt_step()) are hinted, to be
    // read while the device runs `tokens`.
    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const std::int32_t> next = {});
    // Branches of a sequence in one pass, each from the state `at` (a snapshot of the sequence):
    // branch b runs `lengths[b]` of `tokens` (in branch order, at most prefill_chunk in all) at the
    // positions after `at`, its mixers from `at`'s state, while every layer's experts run over all
    // branches' tokens at once. Leaves the logits of each branch's last token in logits(), one
    // column per branch in order, at most max_logit_rows() branches. The sequence then holds the
    // last branch's state, which its MTP block does not follow; restore it before it runs again.
    // Needs runs_branches(): no streamed KV layer.
    void forward_branches(std::uint32_t sequence, const SequenceSnapshot& at,
                          std::span<const std::int32_t> tokens,
                          std::span<const std::uint32_t> lengths);
    // A prompt's last tokens and the first branches of a branched read in one pass, so their
    // experts cross the bus once: the first `prompt` of `tokens` continue the sequence, `at` takes
    // its state at their end (as snapshot() would; the MTP block follows none of the pass), and
    // the branches (`lengths`, the rest of `tokens`) run from there as forward_branches() runs
    // them, with the same logits. Needs runs_branches().
    void forward_prompt_branches(std::uint32_t sequence, std::uint32_t prompt, SequenceSnapshot& at,
                                 std::span<const std::int32_t> tokens,
                                 std::span<const std::uint32_t> lengths);
    [[nodiscard]] bool runs_branches() const noexcept;
    // The most logit rows one pass leaves.
    [[nodiscard]] std::uint32_t max_logit_rows() const noexcept;
    // Runs one token of each of `sequences` (distinct) at its next position in one pass, and
    // leaves their logits in logits(), one column per sequence in that order. The experts read
    // their weights once for the whole batch; each sequence's mixers run on its own state.
    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // Copies the sequence's recurrent state into `out`, and back; both wait for the copies.
    void snapshot(std::uint32_t sequence, SequenceSnapshot& out);
    // Device bytes one SequenceSnapshot's buffers take once its first snapshot sizes them.
    [[nodiscard]] std::uint64_t snapshot_bytes() const;
    void restore(std::uint32_t sequence, const SequenceSnapshot& from);
    // Bytes of an image at `position`.
    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const;
    // Copies the sequence's state into `pages`, host memory that holds image_bytes() of its
    // position in consecutive slices of `page_bytes` (the last one may be partly used): its live
    // state, or the state at `at`, a snapshot it took whose positions it has not written since.
    // Returns the image's header once the copies are done.
    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes);
    // Makes the sequence hold the image's state; returns once the copies are done.
    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes);
    // Vision: encodes the media of the prompt the (just reset) sequence prefills next, whose image
    // and video tokens then take the items' merged embeddings in place of the token embedding.
    // `rope_positions` holds the prompt's three RoPE position axes, axis-major [3, tokens]; every
    // later position rotates at its index plus `rope_delta`. The embeddings are kept until another
    // sequence's media replace them, so the prompt must be prefilled before that. Returns once the
    // tower has run.
    void set_media(std::uint32_t sequence, std::span<const MediaItem> items,
                   std::vector<std::int32_t> rope_positions, std::int32_t rope_delta);
    [[nodiscard]] bool vision() const noexcept;

    // MTP speculative decoding; every call needs an executor with drafts.
    [[nodiscard]] std::uint32_t draft_tokens() const noexcept;
    // Whether the sequence's MTP state follows its tokens, which a media prompt breaks until the
    // sequence is reset; a sequence drafts only from a position past its first.
    [[nodiscard]] bool can_draft(std::uint32_t sequence) const;
    // The sequence's MTP state stops following its tokens until a reset or a restore of a state
    // that follows them: its passes skip the MTP block's catch-up, for a request that never drafts.
    void stop_drafting(std::uint32_t sequence);
    // Drafts `steps` tokens (1..draft_tokens(); 0 means draft_tokens()) for each of `sequences`
    // (distinct) from its next position: the MTP block runs its anchor (the token it sampled last
    // and has not fed) and then its own drafts, greedily. `out` receives them sequence-major, each
    // sequence's row draft_tokens() long; returns once they are on the host, with the number of
    // steps run. Optional `extents` receives one length per sequence, up to the first draft at or
    // below the probability floor. Without a floor or `extents` the whole chain of `steps` runs,
    // captured once per step count; with both, each step runs from its own graph and the chain
    // ends at the first step after which every sequence has drafted at or below the floor (the
    // row's entries past the steps run are -1).
    std::uint32_t draft(std::span<const std::uint32_t> sequences,
                        std::span<const std::int32_t> anchors, std::span<std::int32_t> out,
                        std::span<std::uint32_t> extents = {}, std::uint32_t steps = 0);
    // Runs 2..draft_tokens() + 1 tokens of each sequence (its anchor, then its drafts) at its next
    // positions without committing them, and leaves the logits of every token in logits(), one
    // column per token, sequence after sequence. The sequences' positions and states stay where
    // they were until commit(). Every sequence has the same width within this call.
    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens);
    // Keeps the first `columns[i]` (1..draft_tokens() + 1) tokens of the last verify() of
    // `sequences[i]`, which lists the verified sequences in their order: each sequence's state is
    // then what feeding those tokens would have left, and its MTP block has run their cells.
    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> columns);

    [[nodiscard]] Tensor logits(std::uint32_t rows) const;
    [[nodiscard]] std::size_t head_rank() const noexcept;
    [[nodiscard]] ExpertCacheStats expert_cache_stats() const noexcept;
    [[nodiscard]] std::string expert_execution_profile() const;
    // GGUF host or disk experts with a CPU share: its threads and vector backend; else empty.
    [[nodiscard]] std::string expert_cpu_description() const;
    [[nodiscard]] bool hybrid_experts() const noexcept;
    void save_expert_profile() const;
    // Worker-thread scope only. The owner retains the flag until the eager call returns, then
    // clears the borrow. No consumer thread mutates Program state through this method.
    void bind_cancellation(const std::atomic<bool>* cancelled) noexcept;
    [[nodiscard]] cudaStream_t head_stream() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
