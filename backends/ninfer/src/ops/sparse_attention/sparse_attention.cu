// ninfer::ops - Qwen3.8-Flash-Next sparse attention over selected blocks and the query's own
// incomplete block (contract in include/ninfer/ops/sparse_attention.h). A CTA takes one (query,
// KV head): its twelve query heads, one warp each, stream positions in 32-key tiles staged in
// shared memory, each lane scoring one key and accumulating eight output dims online. Up to eight
// queries split their positions over CTAs of 64 positions each (a grid fixed for any position, so
// a graph replays it), and a second launch merges the partial softmaxes; wider calls run the
// whole list in one CTA per (query, KV head).
//
// A quantized cache is decoded while a tile is staged, sixteen dimensions per thread, to FP16 in
// shared memory: keys in the rotated basis the append stored them in, against a query rotated by
// the same normalized Hadamard transform, and values as stored -- rotated too for NVFP4 and K8V4,
// whose reduction then takes the transform back.
#include "ninfer/ops/sparse_attention.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {

constexpr int kHeadDim     = 256;
constexpr int kQueryHeads  = 24;
constexpr int kKvHeads     = 2;
constexpr int kGroup       = kQueryHeads / kKvHeads; // 12
constexpr int kTopBlocks   = 512;
constexpr int kTile        = 32;
constexpr int kThreads     = kGroup * 32;
constexpr int kRowStride   = kHeadDim + 2; // BF16 elements; 129 words per row spreads the banks
constexpr int kSplitTokens    = 8;
constexpr int kSplitPositions = 2 * kTile;
constexpr int kMaxPositions   = 4 * kTopBlocks + 3; // the selected blocks and the query's own
constexpr int kSplits         = (kMaxPositions + kSplitPositions - 1) / kSplitPositions;
constexpr int kPartialFloats  = kHeadDim + 2; // unnormalised output, running max, running sum

constexpr int kChunk         = 16; // dimensions a thread decodes at once from a quantized row
constexpr int kChunks        = kHeadDim / kChunk;
constexpr int kWordsPerChunk = kChunk / 2;

__device__ __forceinline__ int position_at(const int* blocks, int count, int tail_begin, int i) {
    return i < 4 * count ? blocks[i >> 2] * 4 + (i & 3) : tail_begin + (i - 4 * count);
}

// The four planes of one paged KV layer; the scale planes are null where the storage has none.
struct KvPlanes {
    const void* k       = nullptr;
    const void* v       = nullptr;
    const void* k_scale = nullptr;
    const void* v_scale = nullptr;
};

// Element offset of `leading` in the row of position `p` and KV head `head`, for a plane whose
// rows are `Leading` elements wide.
template <int Leading>
__device__ __forceinline__ std::int64_t plane_offset(const std::int32_t* table, int head, int p,
                                                     int leading) {
    return paged_kv_element_offset<Leading, kKvHeads>(table, head, p, leading);
}

__device__ __forceinline__ float scale_at(const void* plane, std::int64_t offset) {
    return __half2float(static_cast<const __half*>(plane)[offset]);
}

// Sixteen int8 codes in dimension order (byte 0 of x is the first) times `s`, as eight FP16 pairs.
__device__ __forceinline__ void codes_to_half(int4 codes, float s,
                                              unsigned (&out)[kWordsPerChunk]) {
    const unsigned words[4] = {static_cast<unsigned>(codes.x), static_cast<unsigned>(codes.y),
                               static_cast<unsigned>(codes.z), static_cast<unsigned>(codes.w)};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const auto code = [&](int byte) {
            return static_cast<float>(static_cast<std::int8_t>((words[w] >> (8 * byte)) & 0xffu));
        };
        const __half2 low  = __floats2half2_rn(code(0) * s, code(1) * s);
        const __half2 high = __floats2half2_rn(code(2) * s, code(3) * s);
        out[2 * w]         = *reinterpret_cast<const unsigned*>(&low);
        out[2 * w + 1]     = *reinterpret_cast<const unsigned*>(&high);
    }
}

__device__ __forceinline__ void halves_to_words(int4 low, int4 high,
                                                unsigned (&out)[kWordsPerChunk]) {
    out[0] = static_cast<unsigned>(low.x);
    out[1] = static_cast<unsigned>(low.y);
    out[2] = static_cast<unsigned>(low.z);
    out[3] = static_cast<unsigned>(low.w);
    out[4] = static_cast<unsigned>(high.x);
    out[5] = static_cast<unsigned>(high.y);
    out[6] = static_cast<unsigned>(high.z);
    out[7] = static_cast<unsigned>(high.w);
}

// Key dimensions [16 c, 16 c + 16) of one row, in the rotated basis, as FP16 pairs.
template <KvCacheStorage Storage>
__device__ __forceinline__ void decode_key(const KvPlanes& planes, const std::int32_t* table,
                                           int head, int p, int c,
                                           unsigned (&out)[kWordsPerChunk]) {
    const int d = c * kChunk;
    if constexpr (Storage == KvCacheStorage::Int8Group64 ||
                  Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64) {
        const auto* codes =
            static_cast<const std::int8_t*>(planes.k) + plane_offset<256>(table, head, p, d);
        codes_to_half(load_vec<int4>(codes),
                      scale_at(planes.k_scale, plane_offset<4>(table, head, p, d / 64)), out);
    } else if constexpr (Storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                         Storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<128>(table, head, p, d / 2);
        const uint2 packed = load_vec<uint2>(bytes);
        const int4 codes   = Storage == KvCacheStorage::RotatedLloyd4KeyInt4Value
                                 ? kv_cache_lloyd4_expand16(packed)
                                 : kv_cache_int4_unpack_i8x16(packed);
        codes_to_half(codes, scale_at(planes.k_scale, plane_offset<4>(table, head, p, d / 64)),
                      out);
    } else if constexpr (Storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<64>(table, head, p, d / 4);
        codes_to_half(kv_cache_e8_root_unpack_i8x16(load_vec<std::uint32_t>(bytes)),
                      scale_at(planes.k_scale, plane_offset<4>(table, head, p, d / 64)), out);
    } else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256 ||
                         Storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<256>(table, head, p, d);
        const __half s =
            static_cast<const __half*>(planes.k_scale)[plane_offset<1>(table, head, p, 0)];
        halves_to_words(kv_cache_fp8_dequant_f16x8(codes, s),
                        kv_cache_fp8_dequant_f16x8(codes + 8, s), out);
    } else {
        static_assert(Storage == KvCacheStorage::Nvfp4Group16);
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.k) + plane_offset<128>(table, head, p, d / 2);
        const std::uint8_t s =
            static_cast<const std::uint8_t*>(planes.k_scale)[plane_offset<16>(table, head, p, c)];
        halves_to_words(kv_cache_nvfp4_dequant_f16x8(codes, s),
                        kv_cache_nvfp4_dequant_f16x8(codes + 4, s), out);
    }
}

// Value dimensions [16 c, 16 c + 16) of one row as FP16 pairs.
template <KvCacheStorage Storage>
__device__ __forceinline__ void decode_value(const KvPlanes& planes, const std::int32_t* table,
                                             int head, int p, int c,
                                             unsigned (&out)[kWordsPerChunk]) {
    const int d = c * kChunk;
    if constexpr (Storage == KvCacheStorage::Int8Group64) {
        const auto* codes =
            static_cast<const std::int8_t*>(planes.v) + plane_offset<256>(table, head, p, d);
        codes_to_half(load_vec<int4>(codes),
                      scale_at(planes.v_scale, plane_offset<4>(table, head, p, d / 64)), out);
    } else if constexpr (Storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 ||
                         Storage == KvCacheStorage::RotatedLloyd4KeyInt4Value ||
                         Storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
                         Storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
        const auto* bytes =
            static_cast<const std::uint8_t*>(planes.v) + plane_offset<128>(table, head, p, d / 2);
        codes_to_half(kv_cache_int4_unpack_i8x16(load_vec<uint2>(bytes)),
                      scale_at(planes.v_scale, plane_offset<8>(table, head, p, d / 32)), out);
    } else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256) {
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.v) + plane_offset<256>(table, head, p, d);
        const __half s =
            static_cast<const __half*>(planes.v_scale)[plane_offset<1>(table, head, p, 0)];
        halves_to_words(kv_cache_fp8_dequant_f16x8(codes, s),
                        kv_cache_fp8_dequant_f16x8(codes + 8, s), out);
    } else {
        static_assert(Storage == KvCacheStorage::Nvfp4Group16 ||
                      Storage == KvCacheStorage::Fp8KeyNvfp4Value);
        const auto* codes =
            static_cast<const std::uint8_t*>(planes.v) + plane_offset<128>(table, head, p, d / 2);
        const std::uint8_t s =
            static_cast<const std::uint8_t*>(planes.v_scale)[plane_offset<16>(table, head, p, c)];
        halves_to_words(kv_cache_nvfp4_dequant_f16x8(codes, s),
                        kv_cache_nvfp4_dequant_f16x8(codes + 4, s), out);
    }
}

// The staged tiles hold the cache's own BF16 rows, or a quantized cache's rows decoded to FP16.
template <KvCacheStorage Storage>
using TileElement = std::conditional_t<Storage == KvCacheStorage::BFloat16, __nv_bfloat16, __half>;

__device__ __forceinline__ float2 to_float2(const __nv_bfloat162& pair) {
    return __bfloat1622float2(pair);
}

__device__ __forceinline__ float2 to_float2(const __half2& pair) { return __half22float2(pair); }

// Split: this CTA takes positions [64 z, 64 z + 64) of the list and leaves its partial softmax in
// `partial`; otherwise the whole list, normalised into `out`.
template <bool Split, KvCacheStorage Storage>
__global__ void __launch_bounds__(kThreads)
    sparse_attention_kernel(const __nv_bfloat16* __restrict__ q, const int* __restrict__ first,
                            const int* __restrict__ selected, const int* __restrict__ counts,
                            KvPlanes planes, const std::int32_t* __restrict__ block_table,
                            float scale, __nv_bfloat16* __restrict__ out,
                            float* __restrict__ partial) {
    using Element = TileElement<Storage>;
    using Pair    = std::conditional_t<std::is_same_v<Element, __half>, __half2, __nv_bfloat162>;
    constexpr bool kQuantized = Storage != KvCacheStorage::BFloat16;
    // NVFP4 and K8V4 store their values rotated as well as their keys.
    constexpr bool kRotatedValues =
        Storage == KvCacheStorage::Nvfp4Group16 || Storage == KvCacheStorage::Fp8KeyNvfp4Value;
    __shared__ float query[kGroup][kHeadDim];
    __shared__ Element keys[kTile * kRowStride];
    __shared__ Element values[kTile * kRowStride];

    const int t        = blockIdx.x;
    const int kv_head  = blockIdx.y;
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const int q_head   = kv_head * kGroup + warp;
    const int position   = *first + t;
    const int count    = counts[t];
    const int tail_begin = (position + 1) / 4 * 4;
    const int total      = 4 * count + (position + 1 - tail_begin);
    const int* blocks    = selected + static_cast<std::int64_t>(t) * kTopBlocks;
    const int begin      = Split ? static_cast<int>(blockIdx.z) * kSplitPositions : 0;
    const int end        = Split ? min(total, begin + kSplitPositions) : total;

    {
        // Lane l holds dimensions l + 32 r, the layout the key rotation used.
        float row[kHeadDim / 32];
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) {
            row[r] = __bfloat162float(
                q[(static_cast<std::int64_t>(t) * kQueryHeads + q_head) * kHeadDim + lane +
                  32 * r]);
        }
        if constexpr (kQuantized) { normalized_hadamard_d256_inplace(row, lane); }
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) { query[warp][lane + 32 * r] = row[r]; }
    }

    float running_max = -INFINITY, running_sum = 0.0f;
    float acc[8]      = {};
    for (int base = begin; base < end; base += kTile) {
        __syncthreads();
        // Stage the tile: 32 rows of K and V, 4-byte words (the padded rows are not 16B aligned).
        if constexpr (kQuantized) {
            for (int i = threadIdx.x; i < kTile * kChunks; i += kThreads) {
                const int row = i / kChunks, c = i % kChunks;
                const int index             = base + row;
                unsigned kw[kWordsPerChunk] = {}, vw[kWordsPerChunk] = {};
                if (index < end) {
                    const int p = position_at(blocks, count, tail_begin, index);
                    decode_key<Storage>(planes, block_table, kv_head, p, c, kw);
                    decode_value<Storage>(planes, block_table, kv_head, p, c, vw);
                }
                auto* key_words = reinterpret_cast<unsigned*>(&keys[row * kRowStride + c * kChunk]);
                auto* value_words =
                    reinterpret_cast<unsigned*>(&values[row * kRowStride + c * kChunk]);
#pragma unroll
                for (int w = 0; w < kWordsPerChunk; ++w) {
                    key_words[w]   = kw[w];
                    value_words[w] = vw[w];
                }
            }
        } else {
            const auto* k_pages = static_cast<const __nv_bfloat16*>(planes.k);
            const auto* v_pages = static_cast<const __nv_bfloat16*>(planes.v);
            for (int i = threadIdx.x; i < kTile * (kHeadDim / 2); i += kThreads) {
                const int row = i / (kHeadDim / 2), pair = i % (kHeadDim / 2);
                const int index = base + row;
                unsigned kw = 0, vw = 0;
                if (index < end) {
                    const int p               = position_at(blocks, count, tail_begin, index);
                    const std::int64_t offset = paged_kv_element_offset<kHeadDim, kKvHeads>(
                        block_table, kv_head, p, 2 * pair);
                    kw = *reinterpret_cast<const unsigned*>(k_pages + offset);
                    vw = *reinterpret_cast<const unsigned*>(v_pages + offset);
                }
                *reinterpret_cast<unsigned*>(&keys[row * kRowStride + 2 * pair])   = kw;
                *reinterpret_cast<unsigned*>(&values[row * kRowStride + 2 * pair]) = vw;
            }
        }
        __syncthreads();
        // Lane j scores key base + j for this warp's head.
        float score = -INFINITY;
        if (base + lane < end) {
            float dot       = 0.0f;
            const Pair* row = reinterpret_cast<const Pair*>(&keys[lane * kRowStride]);
#pragma unroll 8
            for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                const float2 k = to_float2(row[pair]);
                dot = fmaf(query[warp][2 * pair], k.x, fmaf(query[warp][2 * pair + 1], k.y, dot));
            }
            score = dot * scale;
        }
        float tile_max = score;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
        }
        const float new_max    = fmaxf(running_max, tile_max);
        const float correction = __expf(running_max - new_max);
        const float weight     = base + lane < end ? __expf(score - new_max) : 0.0f;
        float weight_sum       = weight;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            weight_sum += __shfl_xor_sync(0xffffffffu, weight_sum, offset);
        }
        running_sum = running_sum * correction + weight_sum;
        running_max = new_max;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= correction;
        const int keys_in_tile = min(kTile, end - base);
        for (int j = 0; j < keys_in_tile; ++j) {
            const float p   = __shfl_sync(0xffffffffu, weight, j);
            const Pair* row = reinterpret_cast<const Pair*>(&values[j * kRowStride + lane * 8]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 v = to_float2(row[i]);
                acc[2 * i]     = fmaf(p, v.x, acc[2 * i]);
                acc[2 * i + 1] = fmaf(p, v.y, acc[2 * i + 1]);
            }
        }
    }
    if constexpr (kRotatedValues) {
        // The values were stored rotated, so the reduction is too: the same transform takes it
        // back, through this warp's query row, which scoring no longer needs.
        float* scratch = query[warp];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) scratch[lane * 8 + i] = acc[i];
        __syncwarp();
        float row[kHeadDim / 32];
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) row[r] = scratch[lane + 32 * r];
        normalized_hadamard_d256_inplace(row, lane);
#pragma unroll
        for (int r = 0; r < kHeadDim / 32; ++r) scratch[lane + 32 * r] = row[r];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = scratch[lane * 8 + i];
    }
    if constexpr (Split) {
        float* slot =
            partial +
            ((static_cast<std::int64_t>(t) * kSplits + blockIdx.z) * kQueryHeads + q_head) *
                kPartialFloats;
#pragma unroll
        for (int i = 0; i < 8; ++i) slot[lane * 8 + i] = acc[i];
        if (lane == 0) {
            slot[kHeadDim]     = running_max;
            slot[kHeadDim + 1] = running_sum;
        }
    } else {
        const float inverse = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
        __nv_bfloat16* destination =
            out + (static_cast<std::int64_t>(t) * kQueryHeads + q_head) * kHeadDim + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) destination[i] = __float2bfloat16_rn(acc[i] * inverse);
    }
}

// One CTA per (query, query head), one thread per output dim: the splits' partial softmaxes
// rescaled to the common maximum. A split with no positions has a zero sum and adds nothing.
__global__ void __launch_bounds__(kHeadDim)
    sparse_attention_merge_kernel(const float* __restrict__ partial,
                                  __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const float* first_split =
        partial + (static_cast<std::int64_t>(t) * kSplits * kQueryHeads + head) * kPartialFloats;
    constexpr std::int64_t kStride = std::int64_t(kQueryHeads) * kPartialFloats;
    float maximum                  = -INFINITY;
    for (int s = 0; s < kSplits; ++s) {
        const float* slot = first_split + s * kStride;
        if (slot[kHeadDim + 1] > 0.0f) maximum = fmaxf(maximum, slot[kHeadDim]);
    }
    float sum = 0.0f, value = 0.0f;
    for (int s = 0; s < kSplits; ++s) {
        const float* slot = first_split + s * kStride;
        if (slot[kHeadDim + 1] > 0.0f) {
            const float weight = __expf(slot[kHeadDim] - maximum);
            sum += slot[kHeadDim + 1] * weight;
            value += slot[d] * weight;
        }
    }
    out[(static_cast<std::int64_t>(t) * kQueryHeads + head) * kHeadDim + d] =
        __float2bfloat16_rn(sum > 0.0f ? value / sum : 0.0f);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(std::string("sparse_softmax_attention: ") + message);
    }
}

// One plane of the cache against the storage's layout: its dtype and row width.
void require_plane(const Tensor& plane, DType dtype, std::int32_t leading, const char* message) {
    require(plane.data != nullptr && plane.dtype == dtype && plane.is_contiguous() &&
                plane.ne[0] == leading && plane.ne[1] == kPagedKVPageSize &&
                plane.ne[2] == kKvHeads,
            message);
}

template <KvCacheStorage Storage>
void launch(const __nv_bfloat16* q, const int* first, const int* selected, const int* counts,
            const KvPlanes& planes, const std::int32_t* table, float scale, std::int32_t tokens,
            float* partial, __nv_bfloat16* out, cudaStream_t stream) {
    if (partial == nullptr) {
        sparse_attention_kernel<false, Storage><<<dim3(tokens, kKvHeads), kThreads, 0, stream>>>(
            q, first, selected, counts, planes, table, scale, out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    sparse_attention_kernel<true, Storage>
        <<<dim3(tokens, kKvHeads, kSplits), kThreads, 0, stream>>>(
            q, first, selected, counts, planes, table, scale, out, partial);
    CUDA_CHECK(cudaGetLastError());
    sparse_attention_merge_kernel<<<dim3(tokens, kQueryHeads), kHeadDim, 0, stream>>>(partial, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t sparse_softmax_attention_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32,
                       {kPartialFloats, kQueryHeads, kSplits, std::min(tokens, kSplitTokens)});
    return layout.peak_bytes(1);
}

void sparse_softmax_attention(const Tensor& q, const Tensor& first_position, const Tensor& selected,
                              const Tensor& counts, const PagedKVLayerView& cache, float scale,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    require(q.dtype == DType::BF16 && q.is_contiguous() && q.data != nullptr &&
                q.ne[0] == kHeadDim && q.ne[1] == kQueryHeads && q.ne[2] > 0 && q.ne[3] == 1,
            "q must be contiguous BF16 [256, 24, tokens]");
    const std::int32_t tokens = q.ne[2];
    require(first_position.dtype == DType::I32 && first_position.data != nullptr &&
                first_position.numel() >= 1,
            "first_position must be a device I32 word");
    require(selected.dtype == DType::I32 && selected.is_contiguous() && selected.data != nullptr &&
                selected.ne[0] == kTopBlocks && selected.ne[1] == tokens,
            "selected must be contiguous I32 [512, tokens]");
    require(counts.dtype == DType::I32 && counts.is_contiguous() && counts.data != nullptr &&
                counts.ne[0] == tokens,
            "counts must be contiguous I32 [tokens]");
    require(out.dtype == DType::BF16 && out.is_contiguous() && out.data != nullptr &&
                out.ne[0] == kHeadDim && out.ne[1] == kQueryHeads && out.ne[2] == tokens,
            "out must be contiguous BF16 [256, 24, tokens]");
    require(cache.head_dim == kHeadDim && cache.num_kv_heads == kKvHeads &&
                cache.block_table.dtype == DType::I32 && cache.block_table.data != nullptr,
            "cache must be a paged layer of 2 heads of 256");
    const PagedKVStorageLayout layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    require_plane(cache.k_pages, layout.key.data_dtype, layout.key.data_leading_extent,
                  "the key plane does not match the cache storage");
    require_plane(cache.v_pages, layout.value.data_dtype, layout.value.data_leading_extent,
                  "the value plane does not match the cache storage");
    if (layout.key.has_scale()) {
        require_plane(cache.k_scale_pages, layout.key.scale_dtype, layout.key.scale_leading_extent,
                      "the key scale plane does not match the cache storage");
    }
    if (layout.value.has_scale()) {
        require_plane(cache.v_scale_pages, layout.value.scale_dtype,
                      layout.value.scale_leading_extent,
                      "the value scale plane does not match the cache storage");
    }
    require(std::isfinite(scale) && scale > 0.0f, "scale must be positive and finite");
    const auto* q_p      = static_cast<const __nv_bfloat16*>(q.data);
    const auto* first_p  = static_cast<const int*>(first_position.data);
    const auto* select_p = static_cast<const int*>(selected.data);
    const auto* counts_p = static_cast<const int*>(counts.data);
    const auto* table_p  = static_cast<const std::int32_t*>(cache.block_table.data);
    auto* out_p          = static_cast<__nv_bfloat16*>(out.data);
    const KvPlanes planes{.k       = cache.k_pages.data,
                          .v       = cache.v_pages.data,
                          .k_scale = layout.key.has_scale() ? cache.k_scale_pages.data : nullptr,
                          .v_scale = layout.value.has_scale() ? cache.v_scale_pages.data : nullptr};
    auto scope       = workspace.scope();
    float* partial_p = nullptr;
    if (tokens <= kSplitTokens) {
        Tensor partial =
            workspace.alloc(DType::FP32, {kPartialFloats, kQueryHeads, kSplits, tokens});
        partial_p = static_cast<float*>(partial.data);
    }
    switch (cache.storage) {
#define NINFER_SPARSE_ATTENTION_CASE(STORAGE)                                                      \
    case KvCacheStorage::STORAGE:                                                                  \
        launch<KvCacheStorage::STORAGE>(q_p, first_p, select_p, counts_p, planes, table_p, scale,  \
                                        tokens, partial_p, out_p, stream);                         \
        return;
        NINFER_SPARSE_ATTENTION_CASE(BFloat16)
        NINFER_SPARSE_ATTENTION_CASE(Int8Group64)
        NINFER_SPARSE_ATTENTION_CASE(Fp8E4M3Row256)
        NINFER_SPARSE_ATTENTION_CASE(RotatedInt8KeyInt4ValueGroup64)
        NINFER_SPARSE_ATTENTION_CASE(Nvfp4Group16)
        NINFER_SPARSE_ATTENTION_CASE(Fp8KeyNvfp4Value)
        NINFER_SPARSE_ATTENTION_CASE(RotatedLloyd4KeyInt4Value)
        NINFER_SPARSE_ATTENTION_CASE(RotatedInt4KeyInt4ValueE8)
        NINFER_SPARSE_ATTENTION_CASE(RotatedE8RootKeyInt4Value)
#undef NINFER_SPARSE_ATTENTION_CASE
    }
    require(false, "unsupported cache storage");
}

} // namespace ninfer::ops
