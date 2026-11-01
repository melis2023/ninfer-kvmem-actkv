#include "ninfer/ops/readout_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kReadoutLogprobsFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

// `count` ids: the first and last valid rows, then scattered rows.
std::vector<std::int32_t> make_ids(std::int32_t valid_rows, std::int32_t count) {
    std::vector<std::int32_t> ids(static_cast<std::size_t>(count));
    for (std::int32_t k = 0; k < count; ++k) {
        if (k == 0) {
            ids[static_cast<std::size_t>(k)] = 0;
        } else if (k == 1) {
            ids[static_cast<std::size_t>(k)] = valid_rows - 1;
        } else {
            ids[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(
                (static_cast<std::uint64_t>(k + 1) * 7919u) % static_cast<std::uint32_t>(valid_rows));
        }
    }
    return ids;
}

std::vector<std::uint16_t> make_random_logits(std::int32_t physical_rows, std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows));
    for (std::int32_t row = 0; row < valid_rows; ++row) {
        const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u + 1013904223u;
        const float value         = -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
        logits[static_cast<std::size_t>(row)] = f32_to_bf16(value);
    }
    for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
        logits[static_cast<std::size_t>(row)] = f32_to_bf16(96.0f);
    }
    return logits;
}

std::vector<std::uint16_t> make_uniform_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               float value) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows), f32_to_bf16(value));
    for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
        logits[static_cast<std::size_t>(row)] = f32_to_bf16(112.0f);
    }
    return logits;
}

std::vector<std::uint16_t> make_extreme_logits(std::int32_t physical_rows, std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows));
    constexpr float values[] = {-80.0f, -32.0f, -1.0f, 0.0f, 1.0f, 32.0f, 80.0f};
    for (std::int32_t row = 0; row < valid_rows; ++row) {
        logits[static_cast<std::size_t>(row)] =
            f32_to_bf16(values[static_cast<std::size_t>(row) % std::size(values)]);
    }
    for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
        logits[static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
    }
    return logits;
}

std::vector<double> readout_oracle(const std::vector<std::uint16_t>& logits,
                                   const std::vector<std::int32_t>& ids, std::int32_t valid_rows) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int32_t row = 0; row < valid_rows; ++row) {
        maximum = std::max(maximum, static_cast<double>(bf16_to_f32(logits[row])));
    }
    double sum = 0.0;
    for (std::int32_t row = 0; row < valid_rows; ++row) {
        sum += std::exp(static_cast<double>(bf16_to_f32(logits[row])) - maximum);
    }
    std::vector<double> expected(ids.size());
    for (std::size_t k = 0; k < ids.size(); ++k) {
        expected[k] = static_cast<double>(bf16_to_f32(logits[ids[k]])) - maximum - std::log(sum);
    }
    return expected;
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t count, std::int32_t capacity, const std::vector<std::uint16_t>& logits) {
    const auto ids      = make_ids(valid_rows, count);
    const auto expected = readout_oracle(logits, ids, valid_rows);
    std::vector<std::int32_t> stored_ids(static_cast<std::size_t>(capacity), -7);
    std::copy(ids.begin(), ids.end(), stored_ids.begin());

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(stored_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(capacity) * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_ids.copy_from_host(stored_ids.data(), device_ids.bytes());
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows});
    Tensor ids_tensor(device_ids.data(), DType::I32, {capacity});
    Tensor output_tensor(device_output.data(), DType::FP32, {capacity});
    ops::readout_logprobs(logits_tensor, ids_tensor, count, valid_rows, output_tensor, nullptr);
    cuda_synchronize();

    const auto written = from_device<float>(device_output.data(), static_cast<std::size_t>(count));
    int failures = verify_reduction(label, std::vector<double>(written.begin(), written.end()),
                                    expected, kReadoutLogprobsFp32Criterion);
    // Slots past `count` keep their fill.
    const auto raw = from_device<std::uint32_t>(device_output.data(), static_cast<std::size_t>(capacity));
    for (std::size_t k = static_cast<std::size_t>(count); k < raw.size(); ++k) {
        if (raw[k] != 0xcdcdcdcdu) {
            std::cerr << label << ": output slot " << k << " past count was written\n";
            ++failures;
            break;
        }
    }
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures += verify_exact((label + " preserves ids").c_str(),
                             from_device<std::int32_t>(device_ids.data(), stored_ids.size()),
                             stored_ids);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_ids.verify_guards(label + " id guards");
    failures += device_output.verify_guards(label + " output guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(16 * sizeof(std::uint16_t));
    DeviceBuffer ids_data(8 * sizeof(std::int32_t));
    DeviceBuffer output_data(8 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {16});
    Tensor ids(ids_data.p, DType::I32, {8});
    Tensor output(output_data.p, DType::FP32, {8});

    int failures = 0;
    failures += expect_invalid("readout_logprobs rejects count=0",
                               [&] { ops::readout_logprobs(logits, ids, 0, 16, output, nullptr); });
    failures += expect_invalid("readout_logprobs rejects count>capacity",
                               [&] { ops::readout_logprobs(logits, ids, 9, 16, output, nullptr); });
    failures += expect_invalid("readout_logprobs rejects count>maximum", [&] {
        DeviceBuffer wide_ids(64 * sizeof(std::int32_t));
        DeviceBuffer wide_output(64 * sizeof(float));
        Tensor many_ids(wide_ids.p, DType::I32, {64});
        Tensor many_output(wide_output.p, DType::FP32, {64});
        ops::readout_logprobs(logits, many_ids, 33, 16, many_output, nullptr);
    });
    failures += expect_invalid("readout_logprobs rejects valid_rows=0",
                               [&] { ops::readout_logprobs(logits, ids, 4, 0, output, nullptr); });
    failures += expect_invalid("readout_logprobs rejects valid_rows>physical_rows",
                               [&] { ops::readout_logprobs(logits, ids, 4, 17, output, nullptr); });
    failures += expect_invalid("readout_logprobs rejects output dtype", [&] {
        Tensor wrong_output(output_data.p, DType::BF16, {8});
        ops::readout_logprobs(logits, ids, 4, 16, wrong_output, nullptr);
    });
    failures += expect_invalid("readout_logprobs rejects id dtype", [&] {
        Tensor wrong_ids(ids_data.p, DType::FP32, {8});
        ops::readout_logprobs(logits, wrong_ids, 4, 16, output, nullptr);
    });
    failures += expect_invalid("readout_logprobs rejects a logits matrix", [&] {
        Tensor matrix(logits_data.p, DType::BF16, {8, 2});
        ops::readout_logprobs(matrix, ids, 4, 8, output, nullptr);
    });
    failures += expect_invalid("readout_logprobs rejects non-contiguous logits", [&] {
        Tensor strided_logits = logits;
        strided_logits.nb[0] += 2;
        ops::readout_logprobs(strided_logits, ids, 4, 16, output, nullptr);
    });
    failures += expect_invalid("readout_logprobs rejects null output", [&] {
        Tensor null_output(nullptr, DType::FP32, {8});
        ops::readout_logprobs(logits, ids, 4, 16, null_output, nullptr);
    });
    failures += expect_invalid("readout_logprobs rejects output alias", [&] {
        Tensor alias_output(logits_data.p, DType::FP32, {8});
        ops::readout_logprobs(logits, ids, 4, 16, alias_output, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("readout_logprobs full vocabulary 32 ids", 248320, 248077, 32, 32,
                         make_random_logits(248320, 248077));
    failures += run_case("readout_logprobs Qwen vocabulary 26 ids", 151936, 151936, 26, 32,
                         make_random_logits(151936, 151936));
    failures += run_case("readout_logprobs non-aligned rows, one id", 523, 509, 1, 32,
                         make_random_logits(523, 509));
    failures += run_case("readout_logprobs fewer rows than threads", 263, 257, 10, 10,
                         make_random_logits(263, 257));
    failures += run_case("readout_logprobs uniform logits", 263, 257, 12, 32,
                         make_uniform_logits(263, 257, 3.5f));
    failures += run_case("readout_logprobs one valid row", 13, 1, 1, 4,
                         make_uniform_logits(13, 1, -7.0f));
    failures += run_case("readout_logprobs extreme finite logits", 4099, 4093, 32, 32,
                         make_extreme_logits(4099, 4093));
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " readout_logprobs\n";
    return failures ? 1 : 0;
}
