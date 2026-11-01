// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/device_multiprocessors.h"
#include "ops/common/device_route.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fast_plan.h"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
static_assert(kPromptWaveRows == CausalPromptI8FastShape<8>::Br);
static_assert(kPromptWaveRows % CausalPromptI8FastShape<4>::Br == 0);
static_assert(kPromptWaveRows % kCausalPromptI8Br == 0);
static_assert(kPromptWaveRows % kCausalPromptBr == 0);

namespace {

// Sums each key tile's PV products in FP16 on the packed-value INT8 prompt kernels where the
// device profile's "attn_pv_f16" says "on"; NINFER_PROMPT_PV_F16=0|1 overrides the profile.
[[maybe_unused]] bool prompt_pv_f16() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_PV_F16");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pv_f16", 1) == "on";
}

// Packs one KV head's query heads into each INT8 prompt tile (causal_attention_prompt_i8_kernel's
// PackGqa) where the device profile's "attn_pack_gqa" says "on"; NINFER_PROMPT_PACK_GQA=0|1
// overrides the profile. The fast kernel already issues a row block's heads together.
bool prompt_pack_gqa() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_PACK_GQA");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_pack_gqa", 1) == "on";
}

// The int8 cache and rk8v4 share the fast kernel: their keys are the same INT8 G64 codes, and
// PackedValues decodes rk8v4's packed int4 value plane; Keys expands the packed key codings. The
// plan (prompt_i8_fast_plan.h) picks the CTA shape and, past the envelope's key count, may split
// every row block's keys over several CTAs whose partial rows the merge kernel combines.
template <typename Geometry, bool PackedValues, KvKeyCoding Keys = KvKeyCoding::Int8,
          typename CacheView, typename Metadata>
void causal_attention_prompt_i8_fast_launch_for(const Tensor& q, const Tensor& positions,
                                                float scale, const CacheView& cache,
                                                Metadata metadata, const std::int32_t* valid,
                                                std::uint32_t max_visible_keys,
                                                WorkspaceArena& workspace, Tensor& out,
                                                cudaStream_t stream) {
    with_shape_set([&]<bool TuringShapes>() {
        // Eight warps with INT8 values do not fit Turing's shared memory; those launches take four.
        constexpr bool kWide = CausalPromptI8FastShape<8, PackedValues, TuringShapes>::Launchable;
        const auto tokens    = static_cast<std::int32_t>(q.ne[2]);
        const CausalPromptI8FastPlan plan = causal_prompt_i8_fast_plan(
            Geometry::QHeads, tokens, max_visible_keys, current_device_multiprocessors(170), kWide,
            causal_attention_prompt_fast_split_policy());
        auto scope = workspace.scope();
        CausalPromptI8FastPartials partials{};
        if (plan.splits > 1) {
            partials = allocate_causal_prompt_i8_fast_partials(workspace, Geometry::QHeads, tokens,
                                                               plan.splits);
        }
        const auto launch = [&]<int Warps, bool Split>() {
            using Shape = CausalPromptI8FastShape<Warps, PackedValues, TuringShapes>;
            constexpr auto kernel =
                causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps, PackedValues,
                                                       Keys, TuringShapes, Split>;
            raise_dynamic_shared_memory(
                causal_attention_prompt_i8_fast_kernel<Geometry, Metadata, Warps, PackedValues,
                                                       Keys, TuringShapes, Split>,
                Shape::SmemBytes);
            const dim3 grid(static_cast<unsigned>(div_up(tokens, Shape::Br)),
                            static_cast<unsigned>(Geometry::QHeads),
                            static_cast<unsigned>(plan.splits));
            kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache.k_pages.data),
                static_cast<const std::int8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens,
                static_cast<float*>(partials.rows.data), static_cast<float2*>(partials.stats.data));
        };
        if (plan.splits == 1) {
            if constexpr (kWide) {
                if (plan.warps == 8) {
                    launch.template operator()<8, false>();
                    return;
                }
            }
            launch.template operator()<4, false>();
            return;
        }
        // A split launch takes the widest CTA the shape set launches (causal_prompt_i8_fast_plan).
        launch.template operator()<kWide ? 8 : 4, true>();
        CUDA_CHECK(cudaGetLastError());
        constexpr float Log2E = 1.4426950408889634074f;
        causal_attention_prompt_i8_fast_merge_kernel<Geometry>
            <<<dim3(static_cast<unsigned>(tokens), static_cast<unsigned>(Geometry::QHeads)),
               kCausalPromptHeadDim, 0, stream>>>(static_cast<const float*>(partials.rows.data),
                                                  static_cast<const float2*>(partials.stats.data),
                                                  valid, tokens, plan.splits, scale * Log2E,
                                                  static_cast<__nv_bfloat16*>(out.data));
    });
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, const std::int32_t* valid,
                                                  std::uint32_t max_visible_keys,
                                                  WorkspaceArena& workspace, Tensor& out, bool fast,
                                                  cudaStream_t stream) {
    fast = causal_attention_prompt_fast_kernel(cache.storage, fast);
    // The fast kernels hold 100 KiB of shared memory, past sm_70's 96 KiB: a Volta card never takes
    // them (causal_attention_prompt_fast_kernel), and a build for Volta alone does not compile
    // them.
    if constexpr (kBuildTuring || kBuildAmpere || kBuildBlackwell) {
        if (fast && cache.storage == KvCacheStorage::Int8Group64) {
            causal_attention_prompt_i8_fast_launch_for<Geometry, false>(
                q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out,
                stream);
            return;
        }
        if (fast && cache.storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
            causal_attention_prompt_i8_fast_launch_for<Geometry, true>(
                q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out,
                stream);
            return;
        }
        if (fast && cache.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
            causal_attention_prompt_i8_fast_launch_for<Geometry, true, KvKeyCoding::Lloyd4>(
                q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out,
                stream);
            return;
        }
        if (fast && cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
            causal_attention_prompt_i8_fast_launch_for<Geometry, true, KvKeyCoding::Int4E8>(
                q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out,
                stream);
            return;
        }
        if (fast && cache.storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
            causal_attention_prompt_i8_fast_launch_for<Geometry, true, KvKeyCoding::RootE8>(
                q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out,
                stream);
            return;
        }
    }
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    const auto tokens     = static_cast<std::int32_t>(q.ne[2]);
    if (kv_cache_is_int8_family(cache.storage)) {
        if (!causal_prompt_i8_standard_here()) {
            // Every INT8-family coding took the fast kernel above
            // (causal_attention_prompt_fast_kernel).
            throw std::logic_error(
                "INT8-family prompt attention reached the standard kernel, which this card's shape "
                "set lacks");
        }
        // A build of Turing's shape set alone does not compile the standard kernel.
        if constexpr (kBuildStandardShapes) {
            // The INT8 family is the one place that deliberately does not take its plane types from
            // ops/kv_cache/plane_types.h. One kernel serves both codings, so its value parameter is
            // std::int8_t* for int8-g64 and for rk8v4 alike, and the packed-int4 path re-casts to
            // std::uint8_t where it unpacks (causal_prompt_i4_dequant_f16x8). Substituting
            // KvValueCodeT<RotatedInt8KeyInt4ValueGroup64>, which is U8 because the profile
            // describes the plane's storage rather than this kernel's signature, would not be a
            // cleanup. Leave it; the dtype check below is the guard that matters here.
            const bool pack_gqa  = Geometry::GroupSize > 1 && prompt_pack_gqa();
            const auto launch_i8 = [&]<bool PackedValues, KvKeyCoding Keys, bool PvF16>() {
                const auto launch = [&]<bool PackGqa>() {
                    const auto kernel =
                        causal_attention_prompt_i8_kernel<Geometry, Metadata, PackedValues, Keys,
                                                          PvF16, PackGqa>;
                    // The kernel exceeds the default 48 KiB dynamic-smem ceiling.
                    raise_dynamic_shared_memory(kernel, kCausalPromptI8SmemBytes);
                    const dim3 grid =
                        PackGqa ? dim3(static_cast<unsigned>(
                                           div_up(tokens * Geometry::GroupSize, kCausalPromptI8Br)),
                                       static_cast<unsigned>(Geometry::KVHeads), 1u)
                                : dim3(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                       static_cast<unsigned>(Geometry::QHeads), 1u);
                    kernel<<<grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data),
                        static_cast<const std::int8_t*>(cache_k.data),
                        static_cast<const std::int8_t*>(cache_v.data),
                        static_cast<const __half*>(cache.k_scale_pages.data),
                        static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                        static_cast<const std::int32_t*>(positions.data), scale,
                        static_cast<__nv_bfloat16*>(out.data), tokens);
                };
                // A Volta card emulates these tiles: one variant per coding keeps ptxas within
                // bounds, and a build for Volta alone compiles no other.
                if constexpr (kBuildTuring || kBuildAmpere || kBuildBlackwell) {
                    if (pack_gqa && !current_device_volta()) {
                        launch.template operator()<true>();
                        return;
                    }
                }
                launch.template operator()<false>();
            };
            // A U8 key plane is a packed key coding (rk4v4, rk4v4-e8 or rk2v4-e8), told apart by
            // storage; a U8 value plane is the packed signed int4 coding they share with rk8v4.
            // FP16 PV sums where the profile says so; a Volta card takes none (one variant per
            // coding, as above).
            const auto with_pv = [&]<bool PackedValues, KvKeyCoding Keys>() {
                if constexpr (kBuildTuring || kBuildAmpere || kBuildBlackwell) {
                    if (!current_device_volta() && prompt_pv_f16()) {
                        launch_i8.template operator()<PackedValues, Keys, true>();
                        return;
                    }
                }
                launch_i8.template operator()<PackedValues, Keys, false>();
            };
            if (cache.storage == KvCacheStorage::RotatedLloyd4KeyInt4Value) {
                with_pv.template operator()<true, KvKeyCoding::Lloyd4>();
            } else if (cache.storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
                with_pv.template operator()<true, KvKeyCoding::Int4E8>();
            } else if (cache.storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
                launch_i8.template operator()<true, KvKeyCoding::RootE8, false>();
            } else if (cache_v.dtype == DType::U8) {
                with_pv.template operator()<true, KvKeyCoding::Int8>();
            } else {
                launch_i8.template operator()<false, KvKeyCoding::Int8, false>();
            }
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        // The kernel of the current device's shape set, past the default 48 KiB dynamic-smem
        // ceiling in both.
        const auto launch = [&](auto kernel, int smem) {
            raise_dynamic_shared_memory(kernel, smem);
            kernel<<<attention_grid, kCausalPromptThreads, smem, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
        };
        with_shape_set([&]<bool TuringShapes>() {
            if constexpr (TuringShapes) {
                launch(causal_attention_prompt_bf16_turing_kernel<Geometry, Metadata>,
                       kCausalPromptBf16TuringSmemBytes);
            } else {
                launch(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                       kCausalPromptBf16SmemBytes);
            }
        });
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool causal_attention_prompt_fast_kernel(KvCacheStorage storage, bool requested) {
    // The fast kernels hold 100 KiB of shared memory, past a Volta card's 96 KiB.
    if (current_device_volta()) { return false; }
    const bool supported = storage == KvCacheStorage::Int8Group64 ||
                           storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                           storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                           storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
                           storage == KvCacheStorage::RotatedE8RootKeyInt4Value ||
                           storage == KvCacheStorage::Nvfp4Group16;
    if (!supported) { return false; }
    // A card without the standard INT8-family prompt kernel (causal_prompt_i8_standard) takes the
    // fast one for every INT8-family coding.
    if (!causal_prompt_i8_standard_here() && storage != KvCacheStorage::Nvfp4Group16) {
        return true;
    }
    if (requested) { return true; }
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_FAST");
        return value == nullptr ? -1 : (value[0] == '1' ? 1 : 0);
    }();
    if (forced >= 0) { return forced == 1; }
    return device_route_schedule("attn_prompt_fast", 1) == "on";
}

bool causal_prompt_i8_fast_wide(KvCacheStorage storage) {
    const bool packed = storage != KvCacheStorage::Int8Group64;
    if (turing_shapes_here()) {
        return packed ? CausalPromptI8FastShape<8, true, true>::Launchable
                      : CausalPromptI8FastShape<8, false, true>::Launchable;
    }
    return packed ? CausalPromptI8FastShape<8, true, false>::Launchable
                  : CausalPromptI8FastShape<8, false, false>::Launchable;
}

CausalPromptI8FastSplitPolicy causal_attention_prompt_fast_split_policy() {
    static const int forced = [] {
        const char* value = std::getenv("NINFER_PROMPT_SPLITS");
        return value == nullptr ? -1 : std::max(0, std::atoi(value));
    }();
    if (forced == 0) { return {.maximum = 1}; }
    if (forced > 0) { return {.maximum = kCausalPromptI8FastMaxSplits, .forced = forced}; }
    if (device_route_schedule("attn_prompt_split", 1) == "off") { return {.maximum = 1}; }
    return {};
}

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache,
                                              std::uint32_t max_visible_keys,
                                              WorkspaceArena& workspace, Tensor& out, bool fast,
                                              cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
            q, positions, scale, cache, metadata, nullptr, max_visible_keys, workspace, out, fast,
            stream);
        return;
    }
    if (q.ne[1] == 8) {
        causal_attention_prompt_attention_launch_for<CausalD256H8Kv2>(q, positions, scale, cache,
                                                                   metadata, nullptr, max_visible_keys, workspace, out, fast, stream);
        return;
    }
    if (cache.num_kv_heads == 4) {
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv4>(q, positions, scale, cache,
                                                                   metadata, nullptr, max_visible_keys, workspace, out, fast, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, nullptr, max_visible_keys, workspace, out, fast, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, std::uint32_t max_visible_keys,
                                    WorkspaceArena& workspace, Tensor& out, bool fast,
                                    cudaStream_t stream) {
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        causal_attention_prompt_k8v4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                            cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        causal_attention_prompt_nvfp4_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, out, stream);
        return;
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        const std::int32_t* valid = metadata.valid_columns;
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, valid, max_visible_keys, workspace, out, fast,
                stream);
            return;
        }
        if (q.ne[1] == 8) {
            causal_attention_prompt_attention_launch_for<CausalD256H8Kv2>(q, positions, scale, cache,
                                                                       metadata, valid, max_visible_keys, workspace, out, fast, stream);
            return;
        }
        if (cache.num_kv_heads == 4) {
            causal_attention_prompt_attention_launch_for<CausalD256H16Kv4>(q, positions, scale, cache,
                                                                       metadata, valid, max_visible_keys, workspace, out, fast, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, valid, max_visible_keys, workspace, out, fast, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
