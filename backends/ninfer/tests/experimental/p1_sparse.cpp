// Reuse the maintained represented-value codecs and guarded buffers, but select
// original logical keys independently of the production compact page table.
#include "ops/softmax_attention/causal_cache.cpp"
#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "kvmem_test_options.h"

int sparse_case(KvCacheStorage storage, const std::vector<int>& prefix, int offset,
                int width, bool graph, bool fast = false) {
    const Geometry geometry = kGeometries[0]; // Qwen3.8-27B D256/H24/KV4
    const int original_base = 6 * 64 + offset;
    const int compact_base = static_cast<int>(prefix.size()) * 64 + offset;
    std::vector<int> selected = prefix;
    selected.insert(selected.end(), {6, 7, 8});
    auto initial = storage == KvCacheStorage::RotatedLloyd4KeyInt4Value
        ? make_cache(geometry, kPlanRk4v4, 640, 7103)
        : storage == KvCacheStorage::RotatedInt8KeyInt4ValueGroup64
        ? make_cache(geometry, kPlanRk8v4, 640, 7103)
        : make_cache(geometry, storage, 640, 7103);
    if (storage == KvCacheStorage::Nvfp4Group16 || storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        // Unwritten future cache bytes have no value contract. Poison their
        // E4M3 scales with NaNs: causal/tile masks must keep them out of both
        // QK and PV, even when a zero probability would otherwise multiply NaN.
        for (int position = original_base + width; position < initial.logical_capacity; ++position) {
            for (int head = 0; head < geometry.kv_heads; ++head) {
                if (storage == KvCacheStorage::Fp8KeyNvfp4Value) {
                    initial.k_scale.at(fp8_scale_index(geometry, initial.logical_capacity, head, position)) = 0x7e00;
                }
                for (int group = 0; group < kNvfp4QuantGroups; ++group) {
                    const auto index = logical_plane_index(kNvfp4QuantGroups, geometry,
                        initial.logical_capacity, head, position, group);
                    if (storage == KvCacheStorage::Nvfp4Group16) initial.k_nvfp4_scale.at(index) = 0x7f;
                    initial.v_nvfp4_scale.at(index) = 0xff;
                }
            }
        }
    }
    auto expected = initial;
    auto q = make_bf16_values(256ULL * geometry.q_heads * width, 7111, -0.25f, 0.25f);
    auto k = make_bf16_values(256ULL * geometry.kv_heads * width, 7112, -0.25f, 0.25f);
    auto v = make_bf16_values(256ULL * geometry.kv_heads * width, 7113, -1.0f, 1.0f);
    std::vector<std::int32_t> original_positions, compact_positions;
    for (int i = 0; i < width; ++i) {
        original_positions.push_back(original_base + i);
        compact_positions.push_back(compact_base + i);
    }
    append_cache(expected, k, v, original_positions);
    std::vector<double> reference(q.size());
    const int keys = compact_base + width;
    const auto logical_key = [&](int key) { return selected.at(key / 64) * 64 + key % 64; };
    const bool nvfp4 = storage == KvCacheStorage::Nvfp4Group16;
    const bool k8v4 = storage == KvCacheStorage::Fp8KeyNvfp4Value;
    const bool rotated_nvfp4 = nvfp4 || k8v4;
    const auto represented_q = k8v4 ? quantize_fp8_rotated_queries(q, geometry, width)
        : nvfp4 ? represent_f16_rotated_queries(q, geometry, width) : q;
    const auto represented_cache = [&](bool key, int d, int h, int j) {
        if (!rotated_nvfp4) return cache_value(expected, key, h, logical_key(j), d);
        const auto index = cache_index(geometry, expected.logical_capacity, h, logical_key(j), d);
        const auto represented = key ? expected.rotated_k_quantized.at(index)
                                     : expected.rotated_v_quantized.at(index);
        return (key && k8v4) ? double(represented) : double(round_to_f16(represented));
    };
    naive_dense_softmax_attention(op_geometry(geometry), width, keys, 0.0625,
        [&](int d, int h, int t) { return double(represented_q.at(q_index(geometry, h, d, t))); },
        [&](int d, int h, int j) { return represented_cache(true, d, h, j); },
        [&](int d, int h, int j) { return represented_cache(false, d, h, j); },
        [&](int t, int j) { return logical_key(j) <= original_positions[t]; },
        [&](int d, int h, int t, double x) { reference[q_index(geometry, h, d, t)] = x; });
    if (rotated_nvfp4) {
        for (int t = 0; t < width; ++t) {
            for (int h = 0; h < geometry.q_heads; ++h) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d) row[d] = reference[q_index(geometry, h, d, t)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d) reference[q_index(geometry, h, d, t)] = row[d];
            }
        }
    }

    DeviceCache cache(initial, MappingPattern::Fragmented);
    auto view = cache.batch_view();
    std::vector<std::int32_t> native(view.block_tables.ne[0]);
    cuda_check(cudaMemcpy(native.data(), view.block_tables.data, native.size() * 4,
                          cudaMemcpyDeviceToHost), "read physical table");
    std::vector<std::int32_t> mapping;
    for (auto page : selected) { mapping.push_back(native.at(page)); }
    GuardedDeviceBuffer table(mapping.size() * 4), positions(width * 4), rows(4);
    table.copy_from_host(mapping.data(), mapping.size() * 4);
    positions.copy_from_host(compact_positions.data(), width * 4);
    const std::int32_t row = 0;
    rows.copy_from_host(&row, 4);
    view.block_tables = Tensor(table.data(), DType::I32, {static_cast<int>(mapping.size()), 1});
    auto qb = to_bf16_bits(q), kb = to_bf16_bits(k), vb = to_bf16_bits(v);
    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2), dout(qb.size() * 2);
    dq.copy_from_host(qb.data(), qb.size() * 2);
    dk.copy_from_host(kb.data(), kb.size() * 2);
    dv.copy_from_host(vb.data(), vb.size() * 2);
    Tensor tq(dq.data(), DType::BF16, {256, geometry.q_heads, width});
    Tensor tk(dk.data(), DType::BF16, {256, geometry.kv_heads, width});
    Tensor tv(dv.data(), DType::BF16, {256, geometry.kv_heads, width});
    Tensor tp(positions.data(), DType::I32, {width});
    Tensor tr(rows.data(), DType::I32, {1});
    Tensor output(dout.data(), DType::BF16, {256, geometry.q_heads, width});
    const ops::CausalAttentionExecutionEnvelope envelope{
        .min_visible_keys = static_cast<unsigned>(keys),
        .max_visible_keys = static_cast<unsigned>(keys), .fast_prompt_kernel = fast, .small_prefill = true};
    if (fast && (ops::detail::causal_attention_resolve_route(geometry.q_heads, width, 1, storage, envelope) !=
                     ops::detail::CausalAttentionRoute::Prompt ||
                 !ops::detail::causal_attention_prompt_fast_kernel(storage, true)))
        throw std::runtime_error("Fast sparse fixture did not select the fast prompt route");
    const auto bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), storage, envelope, 1, width, width);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    launch_attention_case([&](cudaStream_t stream) {
        ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, tr, op_geometry(geometry),
            0.0625F, view, envelope, workspace, output, stream);
    }, graph);
    const auto actual = bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, qb.size()));
    const auto label = std::string("P1 sparse ") + cache_name(storage) +
        " prefix=" + std::to_string(prefix.size()) + " offset=" + std::to_string(offset) +
        " width=" + std::to_string(width) + " graph=" + std::to_string(graph) + (fast ? " fast=1" : "");
    int failures = verify_attention(label, actual, reference, attention_criterion(storage));
    failures += verify_cache(label, cache.snapshot(), expected, storage == KvCacheStorage::BFloat16 || rotated_nvfp4);
    if (storage == KvCacheStorage::RotatedInt4KeyInt4ValueE8 ||
        storage == KvCacheStorage::RotatedE8RootKeyInt4Value) {
        // These codecs permit different Host/device rotation rounding. Keep the
        // independent attention oracle and also require exact fused/append parity.
        DeviceCache standalone(initial, MappingPattern::Fragmented);
        GuardedDeviceBuffer original_position_buffer(original_positions.size() * 4);
        original_position_buffer.copy_from_host(original_positions.data(), original_positions.size() * 4);
        Tensor original_position_tensor(original_position_buffer.data(), DType::I32, {width});
        ops::kv_cache_append(tk, tv, original_position_tensor, standalone.view(), nullptr);
        cuda_synchronize();
        failures += verify_cache(label + " fused/standalone bytes", cache.snapshot(), standalone.snapshot(), true);
        failures += standalone.verify_guards(label + " standalone append");
    }
    failures += cache.verify_guards(label);
    failures += table.verify_guards("compact table");
    failures += scratch.verify_guards("workspace");
    failures += dout.verify_guards("output");
    std::cout << label << (failures ? " FAIL" : " PASS") << std::endl;
    return failures;
}

int main(int argc, char** argv) {
    try {
        int failures = 0;
        if (argc > 3 || (argc == 3 && std::string(argv[2]) != "fast"))
            throw std::invalid_argument("[bf16|int8|fp8|nvfp4|k8v4|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8] [fast]");
        const bool fast = argc == 3;
        const std::vector<int> widths = fast ? std::vector<int>{65, 127, 128, 129}
            : std::vector<int>{1, 2, 3, 4, 5, 13, 53, 63, 64, 65};
        const auto storages = argc >= 2
            ? std::vector{ninfer::test::kvmem_test_storage(argv[1])}
            : std::vector{KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value};
        for (const auto storage : storages) {
            for (const std::vector<int> pages : {std::vector<int>{0,1,2,3,4,5},
                                               std::vector<int>{0,2,4},
                                               std::vector<int>{4,0,2}}) {
                for (int offset : {0, 63}) {
                    for (int width : widths) {
                        failures += sparse_case(storage, pages, offset, width, false, fast);
                    }
                }
                for (int width : widths) {
                    failures += sparse_case(storage, pages, 63, width, true, fast);
                }
            }
        }
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_SPARSE FAIL " << e.what() << '\n';
        return 1;
    }
}
