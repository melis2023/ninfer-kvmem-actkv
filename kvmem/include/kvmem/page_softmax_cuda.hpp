#pragma once
#include "graph_recall.hpp"
#include <memory>

namespace kvmem {
// Owned by one adapter/Program, used at the quiescent Query selection boundary.
// Device storage is bounded to one layer and <=64 MiB of logits, not all L*T*H*B.
// No process-global stream, mutable buffer, model state or history is shared.
class PageSoftmaxCuda {
public:
    explicit PageSoftmaxCuda(int device);
    // Borrow a startup-planned workspace. score() never allocates device memory
    // in this mode; exceeding the declared geometry fails before a launch.
    static size_t workspace_bytes(uint32_t max_blocks,uint32_t query_heads,uint32_t kv_heads,uint32_t head_dim);
    PageSoftmaxCuda(int device,void* workspace,size_t bytes,uint32_t max_blocks,
        uint32_t query_heads,uint32_t kv_heads,uint32_t head_dim);
    ~PageSoftmaxCuda();
    PageSoftmaxCuda(const PageSoftmaxCuda&) = delete;
    PageSoftmaxCuda& operator=(const PageSoftmaxCuda&) = delete;
    std::vector<double> score(const RawKvStore&, PageQueryView, const QueryScoreFrame&, GqaScoreAggregation = GqaScoreAggregation::Mean);
    size_t device_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kvmem
