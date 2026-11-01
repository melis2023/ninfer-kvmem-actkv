#include "ninfer/ops/residual_add.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion residual_add_bf16_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> residual_add_oracle(const std::vector<float>& y, const std::vector<float>& x) {
    std::vector<double> expected(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        expected[i] = static_cast<double>(x[i]) + static_cast<double>(y[i]);
    }
    return expected;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> y(count), x(count);
    fill_uniform(y, seed, -8.0f, 8.0f);
    fill_uniform(x, seed + 1, -8.0f, 8.0f);
    round_to_bf16(y);
    round_to_bf16(x);

    const auto expected = residual_add_oracle(y, x);
    const auto y_bits   = encode_bf16(y);
    const auto x_bits   = encode_bf16(x);
    GuardedDeviceBuffer device_y(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_y.copy_from_host(y_bits.data(), device_y.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor y_tensor(device_y.data(), DType::BF16, {rows, columns});
    Tensor x_tensor(device_x.data(), DType::BF16, {rows, columns});
    ops::residual_add(y_tensor, x_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                    residual_add_bf16_criterion());
    failures += verify_exact("residual_add y unchanged",
                             from_device<std::uint16_t>(device_y.data(), count), y_bits);
    failures += device_y.verify_guards("residual_add y");
    failures += device_x.verify_guards("residual_add x");
    return failures;
}

int run_cancellation_case() {
    std::vector<float> y{8.0f, -8.0f, 1.0f, -1.0f, 0.0f, 3.5f, -3.5f};
    std::vector<float> x{-8.0f, 8.0f, -1.0f, 1.0f, -0.0f, -3.5f, 3.5f};
    round_to_bf16(y);
    round_to_bf16(x);

    const auto expected = residual_add_oracle(y, x);
    const auto y_bits   = encode_bf16(y);
    const auto x_bits   = encode_bf16(x);
    GuardedDeviceBuffer device_y(y_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_y.copy_from_host(y_bits.data(), device_y.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor y_tensor(device_y.data(), DType::BF16, {static_cast<int>(y.size())});
    Tensor x_tensor(device_x.data(), DType::BF16, {static_cast<int>(x.size())});
    ops::residual_add(y_tensor, x_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("residual_add cancellation",
                                    from_device_bf16(device_x.data(), expected.size()), expected,
                                    residual_add_bf16_criterion());
    failures += verify_exact("residual_add cancellation y unchanged",
                             from_device<std::uint16_t>(device_y.data(), y_bits.size()), y_bits);
    failures += device_y.verify_guards("residual_add cancellation y");
    failures += device_x.verify_guards("residual_add cancellation x");
    return failures;
}

int run_fp32(std::int32_t rows, std::int32_t columns) {
    const auto count = std::size_t(rows) * columns;
    std::vector<float> x(count), y(count);
    fill_uniform(x, 4101, -8.0F, 8.0F);
    fill_uniform(y, 4102, -8.0F, 8.0F);
    for (std::size_t i = 0; i < count; i += 13) { y[i] = -x[i]; }
    const auto expected = residual_add_oracle(y, x);
    GuardedDeviceBuffer dx(count * sizeof(float)), dy(count * sizeof(float));
    dx.copy_from_host(x.data(), count * sizeof(float));
    dy.copy_from_host(y.data(), count * sizeof(float));
    Tensor xt(dx.data(), DType::FP32, {rows, columns});
    const Tensor yt(dy.data(), DType::FP32, {rows, columns});
    ops::residual_add(yt, xt, nullptr);
    cuda_synchronize();
    const auto actual = from_device<float>(dx.data(), count);
    int failures = verify_pointwise("residual_add FP32", std::vector<double>(actual.begin(), actual.end()),
                                    expected, {/*absolute*/ 0.0, /*relative*/ 6e-8});
    failures += verify_exact("residual_add FP32 input unchanged", from_device<float>(dy.data(), count), y);
    failures += dx.verify_guards("residual_add FP32 x") + dy.verify_guards("residual_add FP32 y");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("residual_add [5120,1]", 5120, 1, 101u);
    failures += run_case("residual_add [5120,48]", 5120, 48, 102u);
    failures += run_case("residual_add [2048,17]", 2048, 17, 201u);
    failures += run_case("residual_add [2048,48]", 2048, 48, 202u);
    failures += run_case("residual_add [1152,128]", 1152, 128, 301u);
    failures += run_cancellation_case();
    failures += run_fp32(2560, 1);
    failures += run_fp32(2560, 16);
    failures += run_fp32(17, 3);
    std::cout << (failures ? "FAIL" : "OK") << " residual_add\n";
    return failures ? 1 : 0;
}
