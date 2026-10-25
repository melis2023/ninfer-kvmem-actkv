// ninfer::ops::detail - launch of the fast NVFP4 prompt kernel (block-scaled FP4 QK): its split
// partials, the partial kernel and the split merge. Blackwell builds only; see launch.h.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/plane_types.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_fast.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_fast_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// causal_softmax_attention_prompt_wave_tokens() counts waves of these rows.
static_assert(CausalPromptRotatedShape<8>::Br == 128);

template <class G>
void launch_fast_for(const Tensor& q, const Tensor& positions, const Tensor* valid_columns,
                     const Tensor* table_rows, float scale, const PagedKVBatchLayerView& cache,
                     std::uint32_t max_visible_keys, WorkspaceArena& workspace, Tensor& out,
                     cudaStream_t stream) {
    if (q.ne[3] != 1)
        throw std::invalid_argument("fast prompt attention requires a complete single query row");
    constexpr auto kStorage = KvCacheStorage::Nvfp4Group16;
    const auto* keys        = static_cast<const KvKeyCodeT<kStorage>*>(cache.k_pages.data);
    const auto* values      = static_cast<const KvValueCodeT<kStorage>*>(cache.v_pages.data);
    const auto* key_scales  = static_cast<const KvKeyScaleT<kStorage>*>(cache.k_scale_pages.data);
    const auto* value_scales =
        static_cast<const KvValueScaleT<kStorage>*>(cache.v_scale_pages.data);
    const auto width = static_cast<std::int32_t>(q.ne[2]);
    const RotatedFastPromptPlan plan = rotated_fast_prompt_plan(G::QHeads, width, max_visible_keys);
    auto scope                       = workspace.scope();
    RotatedFastPromptPartials partials{};
    if (plan.splits > 1)
        partials = allocate_rotated_fast_prompt_partials(workspace, G::QHeads, width, plan.splits);
    const auto* q_data  = static_cast<const __nv_bfloat16*>(q.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    auto* out_data      = static_cast<__nv_bfloat16*>(out.data);
    const auto invoke = [&]<class Metadata>(Metadata metadata, const std::int32_t* valid) {
        const auto launch = [&]<int Warps, bool Split>() {
            using Shape = CausalPromptRotatedShape<Warps>;
            configure_cuda_device_once([] {
                return cudaFuncSetAttribute(
                    causal_attention_prompt_rotated_fast_kernel<G, Metadata, Warps, Split>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
            });
            const dim3 grid(div_up(width, Shape::Br), G::QHeads, plan.splits);
            causal_attention_prompt_rotated_fast_kernel<G, Metadata, Warps, Split>
                <<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                q_data, keys, values, key_scales, value_scales, metadata, pos, scale, out_data,
                width, static_cast<float*>(partials.rows.data),
                static_cast<float2*>(partials.stats.data));
            CUDA_CHECK(cudaGetLastError());
        };
        const auto dispatch = [&]<bool Split>() {
            if (plan.warps == 4)
                launch.template operator()<4, Split>();
            else
                launch.template operator()<8, Split>();
        };
        if (plan.splits == 1) {
            dispatch.template operator()<false>();
            return;
        }
        dispatch.template operator()<true>();
        constexpr float Log2E = 1.4426950408889634074f;
        causal_attention_prompt_rotated_fast_merge_kernel<G>
            <<<dim3(width, G::QHeads), kCausalPromptHeadDim, 0, stream>>>(
                static_cast<const float*>(partials.rows.data),
                static_cast<const float2*>(partials.stats.data), valid, width, plan.splits,
                scale * Log2E, out_data);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto* tables = static_cast<const std::int32_t*>(cache.block_tables.data);
    if (table_rows == nullptr) {
        invoke(PagedKVDirectMetadata{tables}, nullptr);
        return;
    }
    const auto* rows = static_cast<const std::int32_t*>(table_rows->data);
    if (valid_columns != nullptr) {
        const auto* valid = static_cast<const std::int32_t*>(valid_columns->data);
        invoke(PagedKVBatchMetadata<true>{tables, valid, rows, cache.block_tables.ne[0]}, valid);
    } else {
        invoke(PagedKVBatchMetadata<false>{tables, nullptr, rows, cache.block_tables.ne[0]},
               nullptr);
    }
}

} // namespace

void causal_attention_prompt_nvfp4_fast_launch(const Tensor& q, const Tensor& positions,
                                               const Tensor* valid_columns,
                                               const Tensor* table_rows, float scale,
                                               PagedKVBatchLayerView cache,
                                               std::uint32_t max_visible_keys,
                                               WorkspaceArena& workspace, Tensor& out,
                                               cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_fast_for<CausalD256H24Kv4>(q, positions, valid_columns, table_rows, scale, cache,
                                          max_visible_keys, workspace, out, stream);
        return;
    }
    launch_fast_for<CausalD256H16Kv2>(q, positions, valid_columns, table_rows, scale, cache,
                                      max_visible_keys, workspace, out, stream);
}

} // namespace ninfer::ops::detail
