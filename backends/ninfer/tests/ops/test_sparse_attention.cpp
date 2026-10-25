// sparse_softmax_attention against an FP64 oracle over the selected blocks and the query's own
// incomplete block, through a fragmented paged cache: a full 512-block selection at long context,
// short contexts where every block is selected, and every tail length 0..3, both for up to eight
// queries (positions split over CTAs, partial softmaxes merged) and for wider calls.
//
// A BF16 cache is written directly. Every quantized storage is written by kv_cache_append and its
// planes are decoded back on the host, independently of the kernel, to the represented keys and
// values: keys in the rotated basis the append stores them in, scored against the query under the
// same normalized Hadamard transform, and values as stored, which for NVFP4 and K8V4 are rotated
// too, so the oracle takes the transform back from the result.
#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sparse_attention.h"
#include "ops/kv_cache_e8_root_host.h"
#include "ops/kv_cache_lloyd4_oracle.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim = 256, kQHeads = 24, kKvHeads = 2, kTop = 512, kPage = 64;

namespace codec = ninfer::test::quantized_weight::detail;

// The normalized Sylvester transform H256/16, in FP64; it is its own inverse.
void hadamard(double* x) {
    for (int stride = 1; stride < kDim; stride *= 2) {
        for (int base = 0; base < kDim; base += 2 * stride) {
            for (int offset = 0; offset < stride; ++offset) {
                const double low  = x[base + offset];
                const double high = x[base + offset + stride];
                x[base + offset]          = low + high;
                x[base + offset + stride] = low - high;
            }
        }
    }
    for (int d = 0; d < kDim; ++d) x[d] *= 0x1p-4;
}

const char* storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rk8v4";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value: return "rk4v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8: return "rk4v4-e8";
    case KvCacheStorage::RotatedE8RootKeyInt4Value: return "rk2v4-e8";
    }
    return "?";
}

std::int8_t signed_nibble(std::uint8_t byte, int d) {
    const unsigned nibble = (d & 1) != 0 ? (byte >> 4) : (byte & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

std::uint8_t nibble(std::uint8_t byte, int d) {
    return static_cast<std::uint8_t>((d & 1) != 0 ? (byte >> 4) : (byte & 0x0fu));
}

// One plane read back from the device, addressed like the kernels address it.
struct HostPlane {
    std::vector<std::uint8_t> bytes;
    std::int32_t leading = 0;
    std::size_t element  = 1;

    template <class T>
    T at(int page, int head, int slot, int index) const {
        const std::size_t offset =
            ((static_cast<std::size_t>(page) * kKvHeads + head) * kPage + slot) * leading + index;
        T value;
        std::memcpy(&value, bytes.data() + offset * element, sizeof(T));
        return value;
    }
};

// Decodes the represented key (rotated basis) and value of one row.
void decode_row(KvCacheStorage storage, const HostPlane& k, const HostPlane& v,
                const HostPlane& ks, const HostPlane& vs, int page, int head, int slot,
                double* key, double* value) {
    using S = KvCacheStorage;
    const auto f16 = [](std::uint16_t bits) { return double(codec::f16_to_f32(bits)); };
    for (int d = 0; d < kDim; ++d) {
        switch (storage) {
        case S::Int8Group64:
        case S::RotatedInt8KeyInt4ValueGroup64:
            key[d] = double(k.at<std::int8_t>(page, head, slot, d)) *
                     f16(ks.at<std::uint16_t>(page, head, slot, d / 64));
            break;
        case S::RotatedLloyd4KeyInt4Value:
            key[d] = double(lloyd4_code(nibble(k.at<std::uint8_t>(page, head, slot, d / 2), d))) *
                     f16(ks.at<std::uint16_t>(page, head, slot, d / 64));
            break;
        case S::RotatedInt4KeyInt4ValueE8:
            key[d] = double(signed_nibble(k.at<std::uint8_t>(page, head, slot, d / 2), d)) *
                     f16(ks.at<std::uint16_t>(page, head, slot, d / 64));
            break;
        case S::RotatedE8RootKeyInt4Value: {
            const e8_root::Code code{k.at<std::uint8_t>(page, head, slot, 2 * (d / 8)),
                                     k.at<std::uint8_t>(page, head, slot, 2 * (d / 8) + 1)};
            key[d] = double(e8_root::decode(code)[static_cast<std::size_t>(d % 8)]) *
                     f16(ks.at<std::uint16_t>(page, head, slot, d / 64));
            break;
        }
        case S::Fp8E4M3Row256:
        case S::Fp8KeyNvfp4Value:
            key[d] = codec::decode_e4m3fn(k.at<std::uint8_t>(page, head, slot, d)) *
                     f16(ks.at<std::uint16_t>(page, head, slot, 0));
            break;
        case S::Nvfp4Group16:
            key[d] = codec::decode_e2m1(nibble(k.at<std::uint8_t>(page, head, slot, d / 2), d)) *
                     codec::decode_e4m3fn(ks.at<std::uint8_t>(page, head, slot, d / 16));
            break;
        case S::BFloat16: break;
        }
        switch (storage) {
        case S::Int8Group64:
            value[d] = double(v.at<std::int8_t>(page, head, slot, d)) *
                       f16(vs.at<std::uint16_t>(page, head, slot, d / 64));
            break;
        case S::RotatedInt8KeyInt4ValueGroup64:
        case S::RotatedLloyd4KeyInt4Value:
        case S::RotatedInt4KeyInt4ValueE8:
        case S::RotatedE8RootKeyInt4Value:
            value[d] = double(signed_nibble(v.at<std::uint8_t>(page, head, slot, d / 2), d)) *
                       f16(vs.at<std::uint16_t>(page, head, slot, d / 32));
            break;
        case S::Fp8E4M3Row256:
            value[d] = codec::decode_e4m3fn(v.at<std::uint8_t>(page, head, slot, d)) *
                       f16(vs.at<std::uint16_t>(page, head, slot, 0));
            break;
        case S::Nvfp4Group16:
        case S::Fp8KeyNvfp4Value:
            value[d] = codec::decode_e2m1(nibble(v.at<std::uint8_t>(page, head, slot, d / 2), d)) *
                       codec::decode_e4m3fn(vs.at<std::uint8_t>(page, head, slot, d / 16));
            break;
        case S::BFloat16: break;
        }
    }
}

int run(KvCacheStorage storage, int first_position, int tokens, std::uint32_t seed) {
    const bool quantized      = storage != KvCacheStorage::BFloat16;
    const bool rotated_values = storage == KvCacheStorage::Nvfp4Group16 ||
                                storage == KvCacheStorage::Fp8KeyNvfp4Value;
    const int length = first_position + tokens;
    const int pages  = (length + kPage - 1) / kPage;
    // Fragmented table: logical page i lives at physical page perm[i].
    std::vector<int> table(pages);
    std::iota(table.begin(), table.end(), 0);
    std::mt19937 random(seed);
    std::shuffle(table.begin(), table.end(), random);
    // Logical K and V, [position][head][dim], and the queries.
    std::vector<float> k(static_cast<std::size_t>(length) * kKvHeads * kDim), v(k.size()),
        q(static_cast<std::size_t>(tokens) * kQHeads * kDim);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -2.0f, 2.0f);
    fill_uniform(q, seed + 3, -1.0f, 1.0f);
    round_to_bf16(k);
    round_to_bf16(v);
    round_to_bf16(q);
    const auto logical = [](int position, int head) {
        return (static_cast<std::size_t>(position) * kKvHeads + head) * kDim;
    };
    const auto encode = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> bits(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
        return bits;
    };

    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, kDim);
    const auto plane_elements = [&](std::int32_t leading) {
        return static_cast<std::size_t>(leading) * kPage * kKvHeads * pages;
    };
    const auto buffer = [&](std::int32_t leading, DType dtype) {
        return std::make_unique<GuardedDeviceBuffer>(
            std::max<std::size_t>(plane_elements(leading) * dtype_size(dtype), 1));
    };
    auto d_k = buffer(layout.key.data_leading_extent, layout.key.data_dtype);
    auto d_v = buffer(layout.value.data_leading_extent, layout.value.data_dtype);
    auto d_ks = buffer(layout.key.has_scale() ? layout.key.scale_leading_extent : 1,
                       layout.key.scale_dtype);
    auto d_vs = buffer(layout.value.has_scale() ? layout.value.scale_leading_extent : 1,
                       layout.value.scale_dtype);
    GuardedDeviceBuffer d_q(q.size() * 2), d_out(q.size() * 2), d_table(table.size() * 4),
        d_first(4);
    d_table.copy_from_host(table.data(), d_table.bytes());
    d_first.copy_from_host(&first_position, 4);
    const auto plane = [&](GuardedDeviceBuffer& memory, std::int32_t leading, DType dtype) {
        return Tensor(memory.data(), dtype, {leading, kPage, kKvHeads, pages});
    };
    PagedKVLayerView cache{};
    cache.k_pages = plane(*d_k, layout.key.data_leading_extent, layout.key.data_dtype);
    cache.v_pages = plane(*d_v, layout.value.data_leading_extent, layout.value.data_dtype);
    if (layout.key.has_scale()) {
        cache.k_scale_pages = plane(*d_ks, layout.key.scale_leading_extent, layout.key.scale_dtype);
    }
    if (layout.value.has_scale()) {
        cache.v_scale_pages =
            plane(*d_vs, layout.value.scale_leading_extent, layout.value.scale_dtype);
    }
    cache.block_table  = Tensor(d_table.data(), DType::I32, {pages});
    cache.head_dim     = kDim;
    cache.num_kv_heads = kKvHeads;
    cache.storage      = storage;

    // The represented cache the oracle reads, [position][head][dim].
    std::vector<double> key(k.size()), value(v.size());
    if (!quantized) {
        std::vector<std::uint16_t> paged_k(plane_elements(kDim)), paged_v(paged_k.size());
        for (int p = 0; p < length; ++p) {
            for (int h = 0; h < kKvHeads; ++h) {
                for (int d = 0; d < kDim; ++d) {
                    const std::size_t paged =
                        ((static_cast<std::size_t>(table[p / kPage]) * kKvHeads + h) * kPage +
                         p % kPage) * kDim + d;
                    paged_k[paged] = f32_to_bf16(k[logical(p, h) + d]);
                    paged_v[paged] = f32_to_bf16(v[logical(p, h) + d]);
                    key[logical(p, h) + d]   = k[logical(p, h) + d];
                    value[logical(p, h) + d] = v[logical(p, h) + d];
                }
            }
        }
        d_k->copy_from_host(paged_k.data(), paged_k.size() * 2);
        d_v->copy_from_host(paged_v.data(), paged_v.size() * 2);
    } else {
        // The append writes the cache in calls of at most 1,024 positions, as a prefill would.
        const auto kb = encode(k), vb = encode(v);
        GuardedDeviceBuffer d_kin(kb.size() * 2), d_vin(vb.size() * 2), d_positions(length * 4);
        d_kin.copy_from_host(kb.data(), d_kin.bytes());
        d_vin.copy_from_host(vb.data(), d_vin.bytes());
        std::vector<std::int32_t> positions(static_cast<std::size_t>(length));
        std::iota(positions.begin(), positions.end(), 0);
        d_positions.copy_from_host(positions.data(), d_positions.bytes());
        for (int begin = 0; begin < length; begin += 1024) {
            const int count = std::min(1024, length - begin);
            const Tensor kin(static_cast<std::uint16_t*>(d_kin.data()) + logical(begin, 0),
                             DType::BF16, {kDim, kKvHeads, count});
            const Tensor vin(static_cast<std::uint16_t*>(d_vin.data()) + logical(begin, 0),
                             DType::BF16, {kDim, kKvHeads, count});
            const Tensor at(static_cast<std::int32_t*>(d_positions.data()) + begin, DType::I32,
                            {count});
            ops::kv_cache_append(kin, vin, at, cache, nullptr);
        }
        cuda_synchronize();
        const auto read = [&](GuardedDeviceBuffer& memory, std::int32_t leading, DType dtype) {
            HostPlane out;
            out.leading = leading;
            out.element = dtype_size(dtype);
            out.bytes   = from_device<std::uint8_t>(memory.data(),
                                                    plane_elements(leading) * out.element);
            return out;
        };
        const HostPlane hk = read(*d_k, layout.key.data_leading_extent, layout.key.data_dtype);
        const HostPlane hv =
            read(*d_v, layout.value.data_leading_extent, layout.value.data_dtype);
        const HostPlane hks = read(*d_ks, layout.key.scale_leading_extent, layout.key.scale_dtype);
        const HostPlane hvs =
            read(*d_vs, layout.value.scale_leading_extent, layout.value.scale_dtype);
        for (int p = 0; p < length; ++p) {
            for (int h = 0; h < kKvHeads; ++h) {
                decode_row(storage, hk, hv, hks, hvs, table[p / kPage], h, p % kPage,
                           &key[logical(p, h)], &value[logical(p, h)]);
            }
        }
    }

    // Selections: every block when there are at most 512, else 512 distinct random ones.
    std::vector<int> selected(static_cast<std::size_t>(kTop) * tokens, -1), counts(tokens);
    for (int t = 0; t < tokens; ++t) {
        const int blocks = (first_position + t + 1) / 4;
        std::vector<int> all(blocks);
        std::iota(all.begin(), all.end(), 0);
        if (blocks > kTop) {
            std::shuffle(all.begin(), all.end(), random);
            all.resize(kTop);
            std::sort(all.begin(), all.end());
        }
        counts[t] = static_cast<int>(all.size());
        std::copy(all.begin(), all.end(), selected.begin() + static_cast<std::ptrdiff_t>(t) * kTop);
    }
    const float scale = 1.0f / 16.0f;
    std::vector<double> expected(q.size());
    for (int t = 0; t < tokens; ++t) {
        const int p = first_position + t, tail = (p + 1) / 4 * 4;
        std::vector<int> positions;
        for (int i = 0; i < counts[t]; ++i)
            for (int j = 0; j < 4; ++j) positions.push_back(selected[static_cast<std::size_t>(t) * kTop + i] * 4 + j);
        for (int j = tail; j <= p; ++j) positions.push_back(j);
        for (int h = 0; h < kQHeads; ++h) {
            const int kv = h / 12;
            std::array<double, kDim> query{};
            for (int d = 0; d < kDim; ++d) query[d] = q[(static_cast<std::size_t>(t) * kQHeads + h) * kDim + d];
            if (quantized) hadamard(query.data());
            std::vector<double> logits(positions.size());
            double maximum = -INFINITY;
            for (std::size_t i = 0; i < positions.size(); ++i) {
                double dot = 0;
                for (int d = 0; d < kDim; ++d) dot += query[d] * key[logical(positions[i], kv) + d];
                logits[i] = dot * scale;
                maximum   = std::max(maximum, logits[i]);
            }
            double sum = 0;
            for (double& l : logits) sum += (l = std::exp(l - maximum));
            std::array<double, kDim> output{};
            for (int d = 0; d < kDim; ++d) {
                double o = 0;
                for (std::size_t i = 0; i < positions.size(); ++i) o += logits[i] * value[logical(positions[i], kv) + d];
                output[d] = o / sum;
            }
            if (rotated_values) hadamard(output.data());
            for (int d = 0; d < kDim; ++d) expected[(static_cast<std::size_t>(t) * kQHeads + h) * kDim + d] = output[d];
        }
    }
    GuardedDeviceBuffer d_selected(selected.size() * 4), d_counts(counts.size() * 4);
    const auto qb = encode(q);
    d_q.copy_from_host(qb.data(), d_q.bytes());
    d_selected.copy_from_host(selected.data(), d_selected.bytes());
    d_counts.copy_from_host(counts.data(), d_counts.bytes());
    Tensor t_q(d_q.data(), DType::BF16, {kDim, kQHeads, tokens});
    Tensor t_out(d_out.data(), DType::BF16, {kDim, kQHeads, tokens});
    Tensor t_selected(d_selected.data(), DType::I32, {kTop, tokens});
    Tensor t_counts(d_counts.data(), DType::I32, {tokens});
    const Tensor t_first(d_first.data(), DType::I32, {1});
    WorkspaceArena workspace(ops::sparse_softmax_attention_workspace_bytes(tokens));
    ops::sparse_softmax_attention(t_q, t_first, t_selected, t_counts, cache, scale, workspace, t_out, nullptr);
    cuda_synchronize();
    const std::string label = std::string("sparse attention ") + storage_name(storage) +
                              " p=" + std::to_string(first_position) + " T=" + std::to_string(tokens);
    int failures = verify_reduction(label, from_device_bf16(d_out.data(), q.size()), expected,
                                    {4.0e-3, 1.0e-4, 2.0 * 3.90625e-3});
    for (auto* memory : {d_k.get(), d_v.get(), d_ks.get(), d_vs.get(), &d_q, &d_out, &d_selected,
                         &d_counts, &d_table, &d_first}) {
        failures += memory->verify_guards(label.c_str());
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const KvCacheStorage storage :
         {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
          KvCacheStorage::RotatedInt8KeyInt4ValueGroup64, KvCacheStorage::Nvfp4Group16,
          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::RotatedLloyd4KeyInt4Value,
          KvCacheStorage::RotatedInt4KeyInt4ValueE8, KvCacheStorage::RotatedE8RootKeyInt4Value}) {
        failures += run(storage, 0, 1, 9100u);    // one key: the query itself
        failures += run(storage, 5, 4, 9101u);    // short context, tails 2,3,0,1
        failures += run(storage, 2047, 9, 9102u); // across the 512-block edge
        failures += run(storage, 4096, 3, 9103u); // a full selection
        failures += run(storage, 9000, 1, 9104u);
        failures += run(storage, 6001, 8, 9105u); // the widest split call, every split busy
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " sparse_softmax_attention\n";
    return failures == 0 ? 0 : 1;
}
