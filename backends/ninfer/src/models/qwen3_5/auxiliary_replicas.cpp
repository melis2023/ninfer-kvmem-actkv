#include "models/qwen3_5/auxiliary_replicas.h"

#include <cstdint>

namespace ninfer::models::qwen3_5 {
namespace {

Tensor auxiliary_tensor(const BoundWeight& weight) {
    return weight_tensor(weight.view, {static_cast<std::int32_t>(weight.view.shape[0])});
}

} // namespace

AuxiliaryReplicas::AuxiliaryReplicas(std::span<const BoundWeight> bound, DeviceContext& device) {
    for (const auto& weight : bound) {
        for (const auto& use : weight.uses) {
            for (const auto& auxiliary : {use.hadamard_signs, use.input_columns}) {
                if (!auxiliary) { continue; }
                const BoundWeight& source = bound[auxiliary->index];
                if (source.rank == weight.rank ||
                    copies_.contains({auxiliary->index, weight.rank})) {
                    continue;
                }
                const Tensor tensor = auxiliary_tensor(source);
                RankBinding bind(device, weight.rank);
                DeviceBuffer copy(tensor.bytes());
                CUDA_CHECK(cudaMemcpyPeer(copy.p, device.rank(weight.rank).device, tensor.data,
                                          device.rank(source.rank).device, tensor.bytes()));
                copies_.emplace(std::pair{auxiliary->index, weight.rank}, std::move(copy));
            }
        }
    }
}

Tensor AuxiliaryReplicas::on_rank(std::span<const BoundWeight> bound, WeightId id,
                                  std::size_t rank) const {
    Tensor tensor      = auxiliary_tensor(bound[id.index]);
    const auto replica = copies_.find({id.index, rank});
    if (replica != copies_.end()) { tensor.data = replica->second.p; }
    return tensor;
}

} // namespace ninfer::models::qwen3_5
