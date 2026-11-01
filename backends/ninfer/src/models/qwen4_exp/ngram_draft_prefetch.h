#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/ngram_hash.h"
#include "models/qwen4_exp/ngram_table.h"

#include <exception>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

// Draft tokens have distinct storage for the whole window. An auxiliary stream copies each
// completed step and issues its row hints while the compute stream runs subsequent steps.
// begin() takes private hash contexts; no speculative token changes committed sequence state.
// Verification hints use the actual verified prefix at each column, not preceding predictions.
class NgramDraftPrefetch {
public:
    NgramDraftPrefetch(const RankContext& rank, NgramTableReader& table,
                       const NgramHashConstants& hash, std::int32_t eos,
                       std::uint32_t vocab, std::size_t sequences, std::size_t steps);
    ~NgramDraftPrefetch();
    NgramDraftPrefetch(const NgramDraftPrefetch&) = delete;
    NgramDraftPrefetch& operator=(const NgramDraftPrefetch&) = delete;

    // The preceding compute stream must have completed before these host operands are reused.
    void begin(std::vector<NgramContext> contexts, std::span<const std::int32_t> anchors);
    // Capturable: both candidates are device I32 [batch], immutable until join() completes.
    void enqueue(std::size_t step, const std::int32_t* first, const std::int32_t* second,
                  cudaStream_t compute);
    // Prefix and candidates are sequence-major [batch, width], width in [2, steps + 1]. Each column's candidates
    // follow that prefix column; neither candidate changes the following column's context.
    void begin_verification(std::vector<NgramContext> contexts,
                             std::span<const std::int32_t> prefix);
    void enqueue_verification(const std::int32_t* first, const std::int32_t* second,
                               cudaStream_t compute);
    // Capturable join, required before ending a multi-stream capture or reusing the operands.
    void join(cudaStream_t compute);
    // After the joined compute stream completes, propagate a failed hash or hint operation.
    void check() const;

private:
    struct Step {
        NgramDraftPrefetch* owner;
        std::size_t index;
        CudaCompletionEvent ready;
    };
    static void CUDART_CB prefetch(void* data) noexcept;
    static void CUDART_CB prefetch_verification(void* data) noexcept;
    void issue(std::span<const std::int32_t> tokens);
    void issue_candidates(const std::int32_t* first, const std::int32_t* second);

    int device_;
    NgramTableReader& table_;
    const NgramHashConstants& hash_;
    std::int32_t eos_;
    std::uint32_t vocab_;
    std::size_t capacity_, batch_ = 0;
    std::size_t verification_width_ = 0;
    PinnedHostBuffer tokens_;
    std::vector<Step> steps_;
    CudaCompletionEvent done_;
    cudaStream_t stream_ = nullptr;
    std::vector<NgramContext> contexts_;
    std::vector<std::int32_t> verification_prefix_;
    std::vector<std::uint64_t> rows_;
    std::exception_ptr failure_;
};

} // namespace ninfer::models::qwen4_exp
