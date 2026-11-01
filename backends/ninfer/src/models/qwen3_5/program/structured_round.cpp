#include "models/qwen3_5/program/structured_round.h"
#include "core/device.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {
StructuredRound::StructuredRound(Tensor masks, std::uint32_t vocab, std::uint32_t width,
                                 std::uint32_t lanes, bool dflash)
    : masks_(masks), words_((vocab + 31) / 32), width_(width), lanes_(lanes), dflash_(dflash),
      host_masks_(masks.bytes()), host_drafts_(std::max(1U, width - 1) * lanes * sizeof(TokenId)) {
    if (lanes > kMaximumLanes) { throw std::logic_error("structured round exceeds its lanes"); }
    std::fill_n(static_cast<std::uint32_t*>(host_masks_.data()), masks.bytes() / 4,
                ~std::uint32_t{0});
    CUDA_CHECK(cudaStreamCreateWithFlags(&upload_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&uploaded_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(uploaded_, upload_));
}

StructuredRound::~StructuredRound() {
    if (upload_ != nullptr) { cudaStreamSynchronize(upload_); }
    if (uploaded_ != nullptr) { cudaEventDestroy(uploaded_); }
    if (upload_ != nullptr) { cudaStreamDestroy(upload_); }
}

const std::uint32_t* StructuredRound::device_mask(std::uint32_t lane) const {
    return static_cast<const std::uint32_t*>(masks_.data) + lane * width_ * words_;
}

void StructuredRound::fill(std::uint32_t lane, const text::GrammarState& grammar,
                           cudaStream_t stream) {
    auto* host = static_cast<std::uint32_t*>(host_masks_.data()) + lane * width_ * words_;
    grammar.fill_masks({host, static_cast<std::size_t>(words_)}, {});
    CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint32_t*>(device_mask(lane)), host,
                               words_ * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream));
}

void StructuredRound::begin() {
    rows_  = {};
    jobs_  = {};
    error_ = nullptr;
}

void StructuredRound::start(std::uint32_t lane, const text::GrammarState& grammar,
                            std::span<const TokenId> drafts) {
    jobs_.at(lane) = runtime::GrammarMaskJob::start(
        grammar, static_cast<std::size_t>(words_), std::vector<TokenId>(drafts.begin(), drafts.end()));
}

void StructuredRound::adopt(std::uint32_t lane, std::shared_ptr<runtime::GrammarMaskJob> job) {
    jobs_.at(lane) = std::move(job);
}

void StructuredRound::publish() {
    // The upload stream collects the jobs, uploads their columns and records the event the round's
    // sampling waits on; with no structured lane the event is recorded at once.
    if (std::any_of(jobs_.begin(), jobs_.end(), [](const auto& job) { return job != nullptr; })) {
        CUDA_CHECK(cudaLaunchHostFunc(upload_, collect, this));
        for (std::uint32_t lane = 0; lane < lanes_; ++lane) {
            if (!jobs_[lane]) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint32_t*>(device_mask(lane)),
                                       static_cast<std::uint32_t*>(host_masks_.data()) +
                                           lane * width_ * words_,
                                       jobs_[lane]->columns() * words_ * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, upload_));
        }
    }
    CUDA_CHECK(cudaEventRecord(uploaded_, upload_));
}

void StructuredRound::set_dflash_row(std::uint32_t row, std::uint32_t lane, std::uint32_t extent,
                                     std::shared_ptr<text::GrammarState> grammar) {
    rows_.at(row) = {lane, extent, std::move(grammar)};
}

void CUDART_CB StructuredRound::start_dflash(void* opaque) noexcept {
    auto& self = *static_cast<StructuredRound*>(opaque);
    try {
        for (std::size_t row = 0; row < self.rows_.size(); ++row) {
            const auto& entry = self.rows_[row];
            if (!entry.grammar) { continue; }
            const auto* drafts =
                static_cast<const TokenId*>(self.host_drafts_.data()) + row * (self.width_ - 1);
            self.jobs_.at(entry.lane) = runtime::GrammarMaskJob::start(
                *entry.grammar, static_cast<std::size_t>(self.words_),
                std::vector<TokenId>(drafts, drafts + entry.extent));
        }
    } catch (...) { self.error_ = std::current_exception(); }
}

void CUDART_CB StructuredRound::collect(void* opaque) noexcept {
    auto& self = *static_cast<StructuredRound*>(opaque);
    try {
        for (std::uint32_t lane = 0; lane < self.lanes_; ++lane) {
            const auto& job = self.jobs_[lane];
            if (!job) { continue; }
            job->wait();
            if (job->error()) {
                if (!self.error_) { self.error_ = job->error(); }
                continue;
            }
            if (const auto problem = runtime::grammar_mask_problem(*job); !problem.empty()) {
                if (!self.error_) {
                    self.error_ = std::make_exception_ptr(std::runtime_error(problem));
                }
                continue;
            }
            std::copy_n(job->masks(), job->columns() * self.words_,
                        static_cast<std::uint32_t*>(self.host_masks_.data()) +
                            lane * self.width_ * self.words_);
        }
    } catch (...) { self.error_ = std::current_exception(); }
}

void StructuredRound::enqueue_dflash(const Tensor& drafts, cudaStream_t stream) {
    if (drafts.dtype != DType::I32 || drafts.ne[0] <= 0 ||
        static_cast<std::uint32_t>(drafts.ne[0]) >= width_ || drafts.ne[1] <= 0 ||
        static_cast<std::uint32_t>(drafts.ne[1]) > lanes_ || drafts.ne[2] != 1 ||
        drafts.ne[3] != 1 || drafts.nb[0] != sizeof(TokenId)) {
        throw std::logic_error("structured draft tensor exceeds its reserved frame");
    }
    // No CUDA calls or exceptions escape the host functions. The drafts' transfer and the host node
    // that starts the masks from them are captured together; the masks compute while the target
    // verifies, and wait() collects them before the acceptance samples.
    // The device frame is compact at this round's width (neural/copy widths can differ).
    // The host node addresses fixed maximum-width host rows; preserve that pitch on every replay.
    CUDA_CHECK(cudaMemcpy2DAsync(host_drafts_.data(), (width_ - 1U) * sizeof(TokenId),
                                 drafts.data, drafts.nb[1], drafts.ne[0] * sizeof(TokenId),
                                 drafts.ne[1], cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaLaunchHostFunc(stream, start_dflash, this));
}

void StructuredRound::wait(cudaStream_t stream) {
    if (dflash_) {
        CUDA_CHECK(cudaLaunchHostFunc(stream, collect, this));
        CUDA_CHECK(cudaMemcpyAsync(masks_.data, host_masks_.data(), masks_.bytes(),
                                   cudaMemcpyHostToDevice, stream));
        return;
    }
    // Captured as an external event node, so each replay waits for the upload published before
    // it; outside a capture (a warm-up or ungraphed round) an ordinary wait.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    CUDA_CHECK(cudaStreamWaitEvent(stream, uploaded_,
                                   capture == cudaStreamCaptureStatusActive ? cudaEventWaitExternal
                                                                            : cudaEventWaitDefault));
}

void StructuredRound::check() {
    for (std::uint32_t lane = 0; lane < lanes_; ++lane) {
        if (jobs_[lane] && jobs_[lane]->ready()) { seconds_[lane] = jobs_[lane]->seconds(); }
    }
    jobs_ = {};
    if (error_) { std::rethrow_exception(std::exchange(error_, nullptr)); }
}
} // namespace ninfer::models::qwen3_5
