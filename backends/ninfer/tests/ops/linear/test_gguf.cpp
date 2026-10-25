// GGUF block matrices through the public linear, linear_add and linear_swiglu Ops and the embedding
// gather, against an FP64 product of the exactly dequantized weights. The products quantize the
// activation to q8_1 as llama.cpp does, so they are held to that arithmetic's error, not to A16's.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <charconv>
#include <vector>

namespace {

using namespace ninfer;

struct TypeCase {
    QType qtype;
    const char* name;
    // Byte offsets of the binary16 multipliers inside one block, set to finite values.
    std::vector<int> half_fields;
};

std::vector<TypeCase> cases() {
    return {
        {QType::GGUF_Q8_0, "q8_0", {0}},       {QType::GGUF_Q2_K, "q2_k", {80, 82}},
        {QType::GGUF_Q3_K, "q3_k", {108}},     {QType::GGUF_Q4_K, "q4_k", {0, 2}},
        {QType::GGUF_Q5_K, "q5_k", {0, 2}},    {QType::GGUF_Q6_K, "q6_k", {208}},
        {QType::GGUF_IQ2_XXS, "iq2_xxs", {0}}, {QType::GGUF_IQ2_XS, "iq2_xs", {0}},
        {QType::GGUF_IQ2_S, "iq2_s", {0}},     {QType::GGUF_IQ3_XXS, "iq3_xxs", {0}},
        {QType::GGUF_IQ3_S, "iq3_s", {0}},     {QType::GGUF_IQ1_S, "iq1_s", {0}},
        {QType::GGUF_IQ1_M, "iq1_m", {}},      {QType::GGUF_IQ4_NL, "iq4_nl", {0}},
        {QType::GGUF_IQ4_XS, "iq4_xs", {0}},   {QType::GGUF_Q4_0, "q4_0", {0}},
        {QType::GGUF_Q5_0, "q5_0", {0}},       {QType::GGUF_Q2_0, "q2_0", {0}},
    };
}

float half_value(const std::uint8_t* p) {
    __half h;
    std::memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

// An independent host decode of the formats simple enough to state here, each value d * (code -
// offset): Q4_0 (low nibbles first, then high), Q5_0 (Q4_0 plus bit j of qh as bit 4 of value j)
// and Q2_0 (two bits per value, low bits first). Empty for every other type.
std::vector<float> host_decode(QType qtype, const std::vector<std::uint8_t>& bytes) {
    std::vector<float> out;
    const auto block = gguf_block_shape(qtype);
    if (qtype != QType::GGUF_Q4_0 && qtype != QType::GGUF_Q5_0 && qtype != QType::GGUF_Q2_0) {
        return out;
    }
    const std::size_t blocks = bytes.size() / block.bytes;
    out.reserve(blocks * block.elements);
    for (std::size_t i = 0; i < blocks; ++i) {
        const std::uint8_t* b = bytes.data() + i * block.bytes;
        const float d         = half_value(b);
        for (int j = 0; j < block.elements; ++j) {
            int code = 0;
            if (qtype == QType::GGUF_Q2_0) {
                code = ((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1;
            } else {
                const std::uint8_t* qs = b + (qtype == QType::GGUF_Q5_0 ? 6 : 2);
                code                   = j < 16 ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
                if (qtype == QType::GGUF_Q5_0) {
                    std::uint32_t qh;
                    std::memcpy(&qh, b + 2, sizeof(qh));
                    code = (code | int(((qh >> j) & 1U) << 4)) - 16;
                } else {
                    code -= 8;
                }
            }
            out.push_back(d * float(code));
        }
    }
    return out;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint16_t half_bits(float value) {
    const __half h = __float2half(value);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// Random blocks with every binary16 multiplier finite; IQ1_M spreads its multiplier over the top
// nibbles of its four scale words.
std::vector<std::uint8_t> random_blocks(const TypeCase& c, std::int32_t rows, std::int32_t k,
                                        std::mt19937& rng) {
    const auto block = gguf_block_shape(c.qtype);
    const std::int64_t count = std::int64_t(rows) * (k / block.elements);
    std::vector<std::uint8_t> bytes(count * block.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto& b : bytes) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::int64_t i = 0; i < count; ++i) {
        std::uint8_t* base = bytes.data() + i * block.bytes;
        for (const int offset : c.half_fields) {
            const std::uint16_t bits = half_bits(scale(rng));
            std::memcpy(base + offset, &bits, sizeof(bits));
        }
        if (c.qtype == QType::GGUF_IQ1_M) {
            const std::uint16_t d = half_bits(scale(rng));
            auto* sc              = reinterpret_cast<std::uint16_t*>(base + 48);
            for (int j = 0; j < 4; ++j) {
                sc[j] = static_cast<std::uint16_t>((sc[j] & 0x0fff) | (((d >> (4 * j)) & 0xf) << 12));
            }
        }
    }
    return bytes;
}

Weight gguf_weight(QType qtype, const void* data, std::int32_t rows, std::int32_t k) {
    Weight w;
    w.qtype   = qtype;
    w.layout  = QuantLayout::GgufBlocks;
    w.qdata   = data;
    w.payload = data;
    w.n = w.shape[0] = w.padded_shape[0] = rows;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.ndim       = 2;
    w.group_size = static_cast<std::uint32_t>(gguf_block_shape(qtype).elements);
    w.group      = gguf_block_shape(qtype).elements;
    return w;
}

std::vector<float> to_float(const std::vector<__nv_bfloat16>& values) {
    std::vector<float> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = __bfloat162float(values[i]); }
    return out;
}

struct Error {
    double relative_rms = 0;
    double relative_max = 0;
};

Error compare(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0, den = 0, worst = 0, scale = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = double(got[i]) - want[i];
        num += d * d;
        den += want[i] * want[i];
        worst = std::max(worst, std::abs(d));
        scale = std::max(scale, std::abs(want[i]));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), worst / std::max(scale, 1e-30)};
}

// K = 1536 and K = 2560, Qwen3.8-Flash-Next's hidden width, whose 80 slices leave the vector
// products' last pass over a row with half of a warp's lanes idle.
int run_type(const TypeCase& c, std::uint32_t seed, std::int32_t k,
             const std::vector<std::int32_t>& widths) {
    constexpr std::int32_t kRows = 256;
    const std::int32_t kK        = k;
    const std::int32_t max_t     = *std::max_element(widths.begin(), widths.end());
    std::mt19937 rng(seed);
    int failures = 0;

    const auto host_blocks = random_blocks(c, 2 * kRows, kK, rng);
    DeviceBuffer blocks(host_blocks.size());
    blocks.copy_from_host(host_blocks.data(), host_blocks.size());
    const auto block        = gguf_block_shape(c.qtype);
    const std::int64_t row_bytes = std::int64_t(kK / block.elements) * block.bytes;
    const Weight gate = gguf_weight(c.qtype, blocks.p, kRows, kK);
    Weight up         = gate;
    up.qdata          = static_cast<const std::byte*>(blocks.p) + kRows * row_bytes;
    const Weight both = gguf_weight(c.qtype, blocks.p, 2 * kRows, kK);

    // Exact weights.
    DeviceBuffer exact_device(std::size_t(2) * kRows * kK * sizeof(float));
    ops::gguf::dequantize_rows(ops::detail::gguf_type(c.qtype), blocks.p, row_bytes, kK,
                               nullptr, 2 * kRows, static_cast<float*>(exact_device.p), kK,
                               nullptr);
    check(cudaDeviceSynchronize(), "dequantize");
    std::vector<float> exact(std::size_t(2) * kRows * kK);
    exact_device.copy_to_host(exact.data(), exact.size() * sizeof(float));
    for (const float v : exact) {
        if (!std::isfinite(v)) {
            std::cerr << c.name << ": non-finite dequantized weight\n";
            return 1;
        }
    }
    if (const auto host = host_decode(c.qtype, host_blocks); !host.empty() && host != exact) {
        std::cerr << "FAIL " << c.name << ": dequantization differs from the host decode\n";
        return 1;
    }

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x_host(std::size_t(kK) * max_t);
    for (auto& v : x_host) { v = __float2bfloat16(normal(rng)); }
    const auto x_float = to_float(x_host);
    DeviceBuffer x_device(x_host.size() * sizeof(__nv_bfloat16));
    x_device.copy_from_host(x_host.data(), x_host.size() * sizeof(__nv_bfloat16));

    std::vector<std::int32_t> columns(kK);
    for (std::int32_t i = 0; i < kK; ++i) { columns[i] = i; }
    std::shuffle(columns.begin(), columns.end(), rng);
    DeviceBuffer columns_device(columns.size() * sizeof(std::int32_t));
    columns_device.copy_from_host(columns.data(), columns.size() * sizeof(std::int32_t));
    Weight gathered        = gate;
    gathered.input_columns = static_cast<const std::int32_t*>(columns_device.p);

    const auto product = [&](std::int32_t row0, std::int32_t t, bool permuted) {
        std::vector<double> out(std::size_t(kRows) * t);
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < kRows; ++r) {
                double acc = 0;
                const float* w = exact.data() + std::size_t(row0 + r) * kK;
                const float* x = x_float.data() + std::size_t(col) * kK;
                for (std::int32_t i = 0; i < kK; ++i) {
                    acc += double(w[i]) * x[permuted ? columns[i] : i];
                }
                out[std::size_t(col) * kRows + r] = acc;
            }
        }
        return out;
    };

    const std::size_t capacity = std::max(
        {ops::linear_workspace_capacity_bytes(c.qtype, kRows, kK, ops::LinearPolicy::A16Only, 1,
                                              max_t),
         ops::linear_swiglu_workspace_capacity_bytes(c.qtype, 2 * kRows, kK,
                                                     ops::LinearPolicy::A16Only, 1, max_t),
         ops::linear_swiglu_pair_workspace_capacity_bytes(c.qtype, c.qtype, kRows, kK,
                                                          ops::LinearPolicy::A16Only, 1, max_t)});
    WorkspaceArena workspace(capacity + (1U << 20));

    // q8_1 activations cost one product about 1 % RMS against the exact weights; a SwiGLU output
    // carries two such products, so it gets twice the budget.
    const auto run_check = [&](const std::string& label, const std::vector<float>& got,
                               const std::vector<double>& want) {
        const Error e     = compare(got, want);
        const double gain = label.starts_with("swiglu") ? 2.0 : 1.0;
        const bool ok     = e.relative_rms < 2e-2 * gain && e.relative_max < 8e-2 * gain;
        if (!ok) {
            std::cerr << "FAIL " << c.name << " " << label << " rms=" << e.relative_rms
                      << " max=" << e.relative_max << '\n';
        }
        return ok ? 0 : 1;
    };

    for (const std::int32_t t : widths) {
        Tensor x(x_device.p, DType::BF16, {kK, t});
        DeviceBuffer out_device(std::size_t(kRows) * t * sizeof(__nv_bfloat16));
        Tensor out(out_device.p, DType::BF16, {kRows, t});
        std::vector<__nv_bfloat16> host(std::size_t(kRows) * t);

        ops::linear(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        const auto plain = product(0, t, false);
        failures += run_check("linear T=" + std::to_string(t), to_float(host), plain);

        ops::linear(x, gathered, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "gathered linear");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("gathered T=" + std::to_string(t), to_float(host),
                              product(0, t, true));

        std::vector<__nv_bfloat16> residual(std::size_t(kRows) * t);
        for (auto& v : residual) { v = __float2bfloat16(normal(rng)); }
        out_device.copy_from_host(residual.data(), residual.size() * sizeof(__nv_bfloat16));
        ops::linear_add(x, gate, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_add");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        std::vector<double> sum = plain;
        for (std::size_t i = 0; i < sum.size(); ++i) { sum[i] += __bfloat162float(residual[i]); }
        failures += run_check("linear_add T=" + std::to_string(t), to_float(host), sum);

        const auto upper = product(kRows, t, false);
        std::vector<double> swiglu(plain.size());
        for (std::size_t i = 0; i < swiglu.size(); ++i) {
            swiglu[i] = plain[i] / (1.0 + std::exp(-plain[i])) * upper[i];
        }
        ops::linear_swiglu(x, both, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu T=" + std::to_string(t), to_float(host), swiglu);

        ops::linear_swiglu(x, gate, up, out, ops::LinearPolicy::A16Only, workspace, nullptr);
        check(cudaDeviceSynchronize(), "linear_swiglu pair");
        out_device.copy_to_host(host.data(), host.size() * sizeof(__nv_bfloat16));
        failures += run_check("swiglu pair T=" + std::to_string(t), to_float(host), swiglu);
    }

    // The token-table gather is the exact weights rounded to BF16.
    const std::vector<std::int32_t> ids{3, 0, 255, 17, 3};
    DeviceBuffer ids_device(ids.size() * sizeof(std::int32_t));
    ids_device.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    DeviceBuffer rows_device(ids.size() * kK * sizeof(__nv_bfloat16));
    Tensor ids_tensor(ids_device.p, DType::I32, {static_cast<std::int32_t>(ids.size())});
    Tensor rows(rows_device.p, DType::BF16, {kK, static_cast<std::int32_t>(ids.size())});
    ops::embedding(ids_tensor, gate, rows, nullptr);
    check(cudaDeviceSynchronize(), "embedding");
    std::vector<__nv_bfloat16> gathered_rows(ids.size() * kK);
    rows_device.copy_to_host(gathered_rows.data(), gathered_rows.size() * sizeof(__nv_bfloat16));
    for (std::size_t i = 0; i < ids.size(); ++i) {
        for (std::int32_t j = 0; j < kK; ++j) {
            const float want = __bfloat162float(__float2bfloat16(exact[std::size_t(ids[i]) * kK + j]));
            if (__bfloat162float(gathered_rows[i * kK + j]) != want) {
                std::cerr << "FAIL " << c.name << " embedding row " << ids[i] << " column " << j
                          << '\n';
                ++failures;
                i = ids.size();
                break;
            }
        }
    }
    return failures;
}

int cross_device_matrix() {
    // Stream-K partitions depend on SM count. Measure the same encoded matrix and activation on
    // both devices before attributing cross-card greedy differences to pipeline state handling.
    constexpr int rows = 5120, k = 6144, columns = 128;
    const TypeCase type{QType::GGUF_IQ3_S, "iq3_s", {0}};
    const auto native = ops::detail::gguf_type(type.qtype);
    std::mt19937 rng(314159);
    const auto bytes = random_blocks(type, rows, k, rng);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16> input(std::size_t(k) * columns);
    for (auto& value : input) { value = __float2bfloat16(normal(rng)); }
    const auto block = gguf_block_shape(type.qtype);
    const auto stride = std::int64_t(k / block.elements) * block.bytes;
    std::array<std::vector<float>, 2> results;
    for (int device = 0; device < 2; ++device) {
        check(cudaSetDevice(device), "select matrix device");
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device), "matrix device properties");
        DeviceBuffer weights(bytes.size()), x(input.size() * sizeof(__nv_bfloat16));
        weights.copy_from_host(bytes.data(), bytes.size());
        x.copy_from_host(input.data(), input.size() * sizeof(__nv_bfloat16));
        DeviceBuffer activation(ops::gguf::matrix_activation_bytes(k, columns));
        DeviceBuffer out(std::size_t(rows) * columns * sizeof(float));
        DeviceBuffer fixup(std::max<std::size_t>(ops::gguf::matrix_fixup_bytes(native, rows, columns), 1));
        ops::gguf::quantize_matrix_activation(native, static_cast<const __nv_bfloat16*>(x.p),
                                               k, columns, nullptr, activation.p, nullptr);
        ops::gguf::matrix_product(native, weights.p, stride, rows, k, activation.p, columns,
                                   static_cast<float*>(out.p), rows, fixup.p, nullptr);
        check(cudaDeviceSynchronize(), "cross-device matrix");
        results[device].resize(std::size_t(rows) * columns);
        out.copy_to_host(results[device].data(), results[device].size() * sizeof(float));
        std::cout << "GGUF_DEVICE " << device << " " << props.name
                  << " SMs=" << props.multiProcessorCount << '\n';
    }
    std::size_t different = 0, bf16_different = 0;
    double squared = 0, reference_squared = 0, max_error = 0, max_reference = 0;
    for (std::size_t i = 0; i < results[0].size(); ++i) {
        const auto a = results[0][i], b = results[1][i];
        if (!std::isfinite(a) || !std::isfinite(b)) { throw std::runtime_error("non-finite matrix output"); }
        different += a != b;
        bf16_different += __bfloat162float(__float2bfloat16(a)) != __bfloat162float(__float2bfloat16(b));
        const double delta = double(a) - b;
        squared += delta * delta;
        reference_squared += double(a) * a;
        max_error = std::max(max_error, std::abs(delta));
        max_reference = std::max(max_reference, std::abs(double(a)));
    }
    const auto rms = std::sqrt(squared / std::max(reference_squared, 1e-30));
    const auto relative_max = max_error / std::max(max_reference, 1e-30);
    std::cout << "GGUF_CROSS_DEVICE total=" << results[0].size() << " fp32_different=" << different
              << " bf16_different=" << bf16_different << " relative_rms=" << rms
              << " relative_max=" << relative_max << '\n';
    return rms < 1e-5 && relative_max < 1e-5 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (std::getenv("NINFER_TEST_GGUF_CROSS_DEVICE")) {
            return devices >= 2 ? cross_device_matrix() : 77;
        }
        int failures      = 0;
        std::uint32_t seed = 9001;
        std::vector<std::string> formats;
        std::vector<std::int32_t> dimensions{1536, 2560};
        std::vector<std::int32_t> widths{1, 2, 3, 4, 5, 8, 9, 17, 40, 70, 130};
        const auto split = [](std::string_view value) {
            std::vector<std::string> out;
            while (!value.empty()) {
                const auto end = value.find(',');
                const auto item = value.substr(0, end);
                if (item.empty()) throw std::invalid_argument("empty filter value");
                out.emplace_back(item);
                if (end == std::string_view::npos) return out;
                value.remove_prefix(end + 1);
            }
            throw std::invalid_argument("empty filter value");
        };
        const auto numbers = [&](std::string_view value) {
            std::vector<std::int32_t> out;
            for (const auto& item : split(value)) {
                std::int32_t n = 0;
                const auto parsed = std::from_chars(item.data(), item.data() + item.size(), n);
                if (parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() || n <= 0)
                    throw std::invalid_argument("filter requires positive integers");
                out.push_back(n);
            }
            return out;
        };
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (++i >= argc) throw std::invalid_argument("GGUF filter needs a value");
            if (flag == "--formats") formats = split(argv[i]);
            else if (flag == "--k") dimensions = numbers(argv[i]);
            else if (flag == "--widths") widths = numbers(argv[i]);
            else throw std::invalid_argument("unknown GGUF filter");
        }
        for (const auto& name : formats) {
            const auto inventory = cases();
            if (std::none_of(inventory.begin(), inventory.end(), [&](const auto& c) { return name == c.name; }))
                throw std::invalid_argument("unknown GGUF format: " + name);
        }
        for (const auto& c : cases()) {
            if (!formats.empty() && std::find(formats.begin(), formats.end(), c.name) == formats.end()) continue;
            for (const std::int32_t k : dimensions) {
                if (k % gguf_block_shape(c.qtype).elements != 0)
                    throw std::invalid_argument("K must contain whole GGUF blocks");
                const int f = run_type(c, seed++, k, widths);
                std::cout << (f == 0 ? "OK   " : "FAIL ") << "gguf " << c.name << " K=" << k
                          << '\n';
                failures += f;
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF Linear: " << error.what() << '\n';
        return 1;
    }
}
