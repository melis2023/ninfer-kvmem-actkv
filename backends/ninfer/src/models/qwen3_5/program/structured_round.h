#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include "runtime/contract/grammar_masks.h"
#include "text/structured_output.h"
#include <array>
#include <exception>
#include <memory>
#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5 {
// Program-owned addresses survive CUDA Graph capture/replay. Grammar state belongs to the
// request, never to a reusable KV checkpoint. Only Engine's output commit advances it.
//
// A round's masks are computed on host threads while the round's forward pass runs, and the
// round's sampling waits for them: each lane's mask job (its columns in parallel) starts before
// the launch, from the drafts the round verifies, or for DFlash from a host node once the round's
// drafter has proposed; the sampling waits on an event recorded after the masks' upload, or for
// DFlash on a host node that collects the jobs.
class StructuredRound {
public:
    static constexpr std::uint32_t kMaximumLanes = 8;

    StructuredRound(Tensor masks, std::uint32_t vocab, std::uint32_t width, std::uint32_t lanes,
                    bool dflash);
    ~StructuredRound();
    StructuredRound(const StructuredRound&)            = delete;
    StructuredRound& operator=(const StructuredRound&) = delete;

    const std::uint32_t* device_mask(std::uint32_t lane) const;

    int stride() const { return words_; }

    // The lane's first mask, synchronously, when its request installs its sampling.
    void fill(std::uint32_t lane, const text::GrammarState& grammar, cudaStream_t stream);

    // Every round begins here.
    void begin();

    // An ordinary or MTP round: start() each structured lane's masks before the launch, then
    // publish() their upload; the round's sampling waits for it (wait()).
    void start(std::uint32_t lane, const text::GrammarState& grammar,
               std::span<const TokenId> drafts);
    // A lane's mask computed ahead of the round (one column), used as the lane's job.
    void adopt(std::uint32_t lane, std::shared_ptr<runtime::GrammarMaskJob> job);
    void publish();

    // A DFlash round: the rows' lanes, draft extents and grammars, then from inside the round the
    // drafts (enqueue_dflash) and the wait before sampling.
    void set_dflash_row(std::uint32_t row, std::uint32_t lane, std::uint32_t extent,
                        std::shared_ptr<text::GrammarState> grammar);
    void enqueue_dflash(const Tensor& drafts, cudaStream_t stream);

    // Captured before the round's sampling.
    void wait(cudaStream_t stream);

    // After the round: rethrows a job's error, and keeps each job's time for seconds().
    void check();
    // How long the lane's masks took in the last round that computed them.
    [[nodiscard]] double seconds(std::uint32_t lane) const { return seconds_.at(lane); }

private:
    static void CUDART_CB start_dflash(void* self) noexcept;
    static void CUDART_CB collect(void* self) noexcept;
    Tensor masks_;
    int words_;
    std::uint32_t width_, lanes_;
    bool dflash_;
    PinnedHostBuffer host_masks_, host_drafts_;
    cudaStream_t upload_  = nullptr;
    cudaEvent_t uploaded_ = nullptr;

    struct Row {
        std::uint32_t lane = 0, extent = 0;
        std::shared_ptr<text::GrammarState> grammar;
    };

    std::array<Row, kMaximumLanes> rows_{};
    // By lane: the round's mask jobs, and how long the last ones took.
    std::array<std::shared_ptr<runtime::GrammarMaskJob>, kMaximumLanes> jobs_{};
    std::array<double, kMaximumLanes> seconds_{};
    std::exception_ptr error_;
};
} // namespace ninfer::models::qwen3_5
