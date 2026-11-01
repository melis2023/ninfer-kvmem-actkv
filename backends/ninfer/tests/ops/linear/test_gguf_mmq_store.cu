#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ninfer::ops::gguf;
namespace detail = ninfer::ops::detail;

struct TypeCase {
    QType qtype;
    gguf::GgmlType type;
    const char* name;
    std::vector<int> half_fields;
};

const TypeCase kCases[] = {
    {QType::GGUF_Q8_0, gguf::GgmlType::Q8_0, "q8_0", {0}},
    {QType::GGUF_Q2_K, gguf::GgmlType::Q2_K, "q2_k", {80, 82}},
    {QType::GGUF_Q3_K, gguf::GgmlType::Q3_K, "q3_k", {108}},
    {QType::GGUF_Q4_K, gguf::GgmlType::Q4_K, "q4_k", {0, 2}},
    {QType::GGUF_Q5_K, gguf::GgmlType::Q5_K, "q5_k", {0, 2}},
    {QType::GGUF_Q6_K, gguf::GgmlType::Q6_K, "q6_k", {208}},
    {QType::GGUF_IQ2_XXS, gguf::GgmlType::IQ2_XXS, "iq2_xxs", {0}},
    {QType::GGUF_IQ2_XS, gguf::GgmlType::IQ2_XS, "iq2_xs", {0}},
    {QType::GGUF_IQ2_S, gguf::GgmlType::IQ2_S, "iq2_s", {0}},
    {QType::GGUF_IQ3_XXS, gguf::GgmlType::IQ3_XXS, "iq3_xxs", {0}},
    {QType::GGUF_IQ3_S, gguf::GgmlType::IQ3_S, "iq3_s", {0}},
    {QType::GGUF_IQ1_S, gguf::GgmlType::IQ1_S, "iq1_s", {0}},
    {QType::GGUF_IQ4_NL, gguf::GgmlType::IQ4_NL, "iq4_nl", {0}},
    {QType::GGUF_IQ4_XS, gguf::GgmlType::IQ4_XS, "iq4_xs", {0}},
    {QType::GGUF_Q4_0, gguf::GgmlType::Q4_0, "q4_0", {0}},
    {QType::GGUF_Q5_0, gguf::GgmlType::Q5_0, "q5_0", {0}},
    {QType::GGUF_Q2_0, gguf::GgmlType::Q2_0, "q2_0", {0}},
};

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half converted = __float2half(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

std::vector<std::uint8_t> make_weight(const TypeCase& c, int rows, int k, std::uint32_t seed) {
    const gguf::BlockShape block = gguf::block_shape(c.type);
    const std::size_t count = std::size_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& value : bytes) {
        value = static_cast<std::uint8_t>(byte(rng));
    }
    for (std::size_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
    }
    return bytes;
}

int rows_for_fixup(const TypeCase& c, int columns, bool wanted) {
    const int sm_count = gguf::multiprocessor_count();
    const int max_rows = 128 * sm_count * 4;
    for (int rows = 128; rows <= max_rows; rows += 128) {
        const bool has_fixup = gguf::matrix_fixup_bytes(c.type, rows, columns) != 0;
        if (has_fixup == wanted) {
            return rows;
        }
    }
    return 0;
}

int padded_tail_k(const TypeCase& c) {
    const int qk = gguf::block_shape(c.type).elements;
    for (int k = qk * 9; k <= qk * 16; k += qk) {
        if (k % 256 != 0) {
            return k;
        }
    }
    return 0;
}

void check_rejected_geometry(const TypeCase& c) {
    const auto reject = [](auto call, const char* message) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            if (std::string(error.what()).find(message) != std::string::npos) { return; }
            throw;
        }
        throw std::runtime_error("GGUF MMQ accepted unsupported geometry");
    };
    for (const int value : {0, -128}) {
        reject([&] { (void)gguf::matrix_fixup_bytes(c.type, value, 16); }, "nonpositive geometry");
        reject([&] { (void)gguf::matrix_fixup_bytes(c.type, 128, value); }, "nonpositive geometry");
    }
    reject([&] { (void)gguf::matrix_fixup_bytes(c.type, INT_MAX, INT_MAX); }, "index range exceeded");
    (void)gguf::matrix_fixup_bytes(c.type, 128, INT_MAX);
    const auto block = gguf::block_shape(c.type);
    const int k = INT_MAX - 255;
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    reject([&] {
        gguf::matrix_product(c.type, nullptr, row_bytes, 128, k, nullptr, 65536,
                             nullptr, 128, nullptr, nullptr);
    }, "index range exceeded");
    for (const bool accumulate : {false, true}) {
        reject([&] {
            gguf::matrix_product_bf16(c.type, nullptr, row_bytes, 128, k, nullptr, 65536,
                                      nullptr, 128, nullptr, nullptr, nullptr, accumulate);
        }, "index range exceeded");
    }
    for (const int invalid_k : {0, -256}) {
        reject([&] {
            gguf::matrix_product(c.type, nullptr, 0, 128, invalid_k, nullptr, 16,
                                 nullptr, 128, nullptr, nullptr);
        }, "unsupported geometry");
    }
}

void compare_outputs(const TypeCase& c, int k, int columns, int rows, int padding,
                     bool accumulate, const std::vector<__nv_bfloat16>& initial,
                     const std::vector<__nv_bfloat16>& expected,
                     const std::vector<__nv_bfloat16>& actual) {
    const int output_rows = rows + 2 * padding;
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < output_rows; ++row) {
            const std::size_t at = std::size_t(column) * output_rows + row;
            std::uint16_t expected_bits = 0;
            std::uint16_t actual_bits = 0;
            std::memcpy(&expected_bits, &expected[at], sizeof(expected_bits));
            std::memcpy(&actual_bits, &actual[at], sizeof(actual_bits));
            if (expected_bits != actual_bits) {
                throw std::runtime_error(std::string(c.name) + " K=" + std::to_string(k) +
                    " T=" + std::to_string(columns) + " row=" + std::to_string(row) +
                    " accumulate=" + std::to_string(accumulate) + " differs from native FP32 route");
            }
            if ((row < padding || row >= padding + rows) &&
                std::memcmp(&actual[at], &initial[at], sizeof(std::uint16_t)) != 0) {
                throw std::runtime_error(std::string(c.name) + " modified output guard rows");
            }
        }
    }
}

void run_direct_case(const TypeCase& c, int k, int columns, int rows, int padding,
                     bool accumulate, bool require_fixup, bool tail = false) {
    const gguf::BlockShape block = gguf::block_shape(c.type);
    if (rows % 128 != 0 || k % block.elements != 0) {
        throw std::runtime_error(std::string(c.name) + " has invalid test geometry");
    }
    const std::size_t fixup_bytes = gguf::matrix_fixup_bytes(c.type, rows, columns);
    if ((fixup_bytes != 0) != require_fixup) {
        throw std::runtime_error(std::string(c.name) + " did not select the requested fixup geometry");
    }

    const int output_rows = rows + 2 * padding;
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const auto host_weight = make_weight(c, rows, k, 0xE190u + static_cast<unsigned>(c.qtype));
    DeviceBuffer weight(host_weight.size() + (tail ? 4096 : 0));
    weight.copy_from_host(host_weight.data(), host_weight.size());
    if (tail) {
        check(cudaMemset(static_cast<char*>(weight.p) + host_weight.size(), 0, 4096),
              "padded matrix weight tail");
    }

    std::mt19937 rng(0xA119u + static_cast<unsigned>(c.qtype) +
                     static_cast<unsigned>(k + columns + rows));
    std::uniform_real_distribution<float> sample(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> host_input(std::size_t(k) * columns);
    std::vector<__nv_bfloat16> initial(std::size_t(output_rows) * columns);
    for (auto& value : host_input) { value = __float2bfloat16(sample(rng)); }
    for (auto& value : initial) { value = __float2bfloat16(sample(rng)); }

    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    const int activation_k = tail ? gguf::matrix_padded_k(k) : k;
    DeviceBuffer activation(gguf::matrix_activation_bytes(activation_k, columns));
    if (tail) {
        gguf::quantize_matrix_activation_padded(c.type,
            static_cast<const __nv_bfloat16*>(input.p), k, columns, nullptr, activation.p, nullptr);
    } else {
        gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                         columns, nullptr, activation.p, nullptr);
    }

    DeviceBuffer native_plane(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer partial(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup(std::max<std::size_t>(fixup_bytes, 1));
    DeviceBuffer native_output(initial.size() * sizeof(__nv_bfloat16));
    DeviceBuffer bf16_output(initial.size() * sizeof(__nv_bfloat16));
    native_output.copy_from_host(initial.data(), native_output.bytes);
    bf16_output.copy_from_host(initial.data(), bf16_output.bytes);

    gguf::matrix_product(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                         static_cast<float*>(native_plane.p), rows, fixup.p, nullptr, tail);
    gguf::VectorOutput destination;
    destination.bf16 = static_cast<__nv_bfloat16*>(native_output.p) + padding;
    destination.column_stride = output_rows;
    destination.epilogue = accumulate ? gguf::Epilogue::Accumulate : gguf::Epilogue::Store;
    gguf::store_plane(static_cast<const float*>(native_plane.p), rows, rows, columns,
                      destination, nullptr);

    gguf::matrix_product_bf16(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                              static_cast<__nv_bfloat16*>(bf16_output.p) + padding,
                              output_rows, static_cast<float*>(partial.p), fixup.p, nullptr,
                              accumulate, tail);
    check(cudaDeviceSynchronize(), "GGUF MMQ BF16 store synchronization");

    std::vector<__nv_bfloat16> expected(initial.size());
    std::vector<__nv_bfloat16> actual(initial.size());
    native_output.copy_to_host(expected.data(), native_output.bytes);
    bf16_output.copy_to_host(actual.data(), bf16_output.bytes);
    compare_outputs(c, k, columns, rows, padding, accumulate, initial, expected, actual);
}

void run_public_gather_case(const TypeCase& c, bool accumulate) {
    constexpr int k = 512;
    constexpr int rows = 128;
    constexpr int columns = 17;
    constexpr int padding = 7;
    const gguf::BlockShape block = gguf::block_shape(c.type);
    const std::int64_t row_bytes = std::int64_t(k / block.elements) * block.bytes;
    const int output_rows = rows + 2 * padding;
    const auto host_weight = make_weight(c, rows, k, 0xB410u + static_cast<unsigned>(c.qtype));
    DeviceBuffer weight(host_weight.size());
    weight.copy_from_host(host_weight.data(), host_weight.size());

    std::mt19937 rng(0xC318u + static_cast<unsigned>(c.qtype));
    std::uniform_real_distribution<float> sample(-0.25f, 0.25f);
    std::vector<__nv_bfloat16> host_input(std::size_t(k) * columns);
    std::vector<__nv_bfloat16> initial(std::size_t(output_rows) * columns);
    std::vector<std::int32_t> host_columns(k);
    for (int i = 0; i < k; ++i) { host_columns[i] = (i * 7) % k; }
    for (auto& value : host_input) { value = __float2bfloat16(sample(rng)); }
    for (auto& value : initial) { value = __float2bfloat16(sample(rng)); }

    DeviceBuffer input(host_input.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(host_input.data(), input.bytes);
    DeviceBuffer columns_device(host_columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(host_columns.data(), columns_device.bytes);
    DeviceBuffer activation(gguf::matrix_activation_bytes(k, columns));
    gguf::quantize_matrix_activation(c.type, static_cast<const __nv_bfloat16*>(input.p), k,
                                     columns, static_cast<const std::int32_t*>(columns_device.p),
                                     activation.p, nullptr);

    DeviceBuffer native_plane(std::size_t(rows) * columns * sizeof(float));
    DeviceBuffer fixup(std::max<std::size_t>(
        gguf::matrix_fixup_bytes(c.type, rows, columns), 1));
    DeviceBuffer native_output(initial.size() * sizeof(__nv_bfloat16));
    DeviceBuffer public_output(initial.size() * sizeof(__nv_bfloat16));
    native_output.copy_from_host(initial.data(), native_output.bytes);
    public_output.copy_from_host(initial.data(), public_output.bytes);
    gguf::matrix_product(c.type, weight.p, row_bytes, rows, k, activation.p, columns,
                         static_cast<float*>(native_plane.p), rows, fixup.p, nullptr);
    gguf::VectorOutput destination;
    destination.bf16 = static_cast<__nv_bfloat16*>(native_output.p) + padding;
    destination.column_stride = output_rows;
    destination.epilogue = accumulate ? gguf::Epilogue::Accumulate : gguf::Epilogue::Store;
    gguf::store_plane(static_cast<const float*>(native_plane.p), rows, rows, columns,
                      destination, nullptr);

    Tensor x(input.p, DType::BF16, {k, columns});
    Tensor y(public_output.p, DType::BF16, {output_rows, columns});
    Weight w;
    w.qtype = c.qtype;
    w.layout = QuantLayout::GgufBlocks;
    w.qdata = weight.p;
    w.payload = weight.p;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim = 2;
    w.group_size = static_cast<std::uint32_t>(block.elements);
    w.group = block.elements;
    w.input_columns = static_cast<const std::int32_t*>(columns_device.p);
    const detail::GgufProduct product{
        &w, &y, padding,
        accumulate ? detail::GgufEpilogue::Accumulate : detail::GgufEpilogue::Store};
    const detail::GgufShape shape{c.qtype, rows, k};
    WorkspaceArena workspace(detail::gguf_project_workspace_bytes({&shape, 1}, columns, columns) +
                             (1U << 20));
    detail::gguf_project(x, {&product, 1}, workspace, nullptr);
    check(cudaDeviceSynchronize(), "public GGUF project synchronization");

    std::vector<__nv_bfloat16> expected(initial.size());
    std::vector<__nv_bfloat16> actual(initial.size());
    native_output.copy_to_host(expected.data(), native_output.bytes);
    public_output.copy_to_host(actual.data(), public_output.bytes);
    compare_outputs(c, k, columns, rows, padding, accumulate, initial, expected, actual);
}

}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int sm_count = gguf::multiprocessor_count();
        for (const TypeCase& c : kCases) {
            check_rejected_geometry(c);
            const int nofix_rows = rows_for_fixup(c, 9, false);
            if (nofix_rows == 0) {
                throw std::runtime_error(std::string(c.name) + " has no non-fixup geometry");
            }
            run_direct_case(c, 512, 9, nofix_rows, 7, false, false);
            run_direct_case(c, 512, 9, nofix_rows, 7, true, false);

            const int split_rows = rows_for_fixup(c, 9, true);
            if (split_rows == 0) {
                throw std::runtime_error(std::string(c.name) + " did not expose a stream-K fixup shape");
            }
            run_direct_case(c, 512, 9, split_rows, 7, false, true);
            run_direct_case(c, 512, 9, split_rows, 7, true, true);
            run_direct_case(c, 256, 9, split_rows, 7, false, true, true);
            run_direct_case(c, 256, 9, split_rows, 7, true, true, true);
            const int tail_k = padded_tail_k(c);
            if (tail_k != 0) {
                run_direct_case(c, tail_k, 9, nofix_rows, 7, false, false, true);
                run_direct_case(c, tail_k, 9, nofix_rows, 7, true, false, true);
            }
            std::cout << "OK GGUF MMQ BF16 " << c.name << " nonfixup/split="
                      << (split_rows != 0) << '\n';
        }

        const TypeCase& q4 = kCases[3];
        const TypeCase& q6 = kCases[5];
        const bool q6_t16_fixup = gguf::matrix_fixup_bytes(q6.type, 5120, 16) != 0;
        for (const bool accumulate : {false, true}) {
            run_direct_case(q6, 17408, 16, 5120, 7, accumulate, q6_t16_fixup);
        }
        for (const TypeCase* c : {&q4, &q6}) {
            if (gguf::matrix_fixup_bytes(c->type, 128, 9) == 0) {
                throw std::runtime_error(std::string(c->name) + " has no multi-contributor fixup geometry");
            }
            for (const int columns : {9, 17, 65, 128, 129, 512, 1024}) {
                const int rows = 128;
                const bool fixup = gguf::matrix_fixup_bytes(c->type, rows, columns) != 0;
                if (columns == 9 && !fixup) {
                    throw std::runtime_error(std::string(c->name) + " lost the stream-K fixup route");
                }
                for (const bool accumulate : {false, true}) {
                    run_direct_case(*c, 5120, columns, rows, 7, accumulate, fixup);
                }
            }
            for (const bool accumulate : {false, true}) {
                run_public_gather_case(*c, accumulate);
            }
        }
        std::cout << "OK GGUF MMQ BF16 store, sm=" << sm_count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GGUF MMQ BF16 store: " << error.what() << '\n';
        return 1;
    }
}
