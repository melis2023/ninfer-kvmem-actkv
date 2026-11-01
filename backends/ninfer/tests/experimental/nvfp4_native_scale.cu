#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include <cuda_runtime.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr unsigned cases = 256 * 128;
__global__ void decode(std::uint32_t* output) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cases) return;
    const auto code = static_cast<std::uint8_t>(index & 255);
    const auto scale = static_cast<std::uint8_t>(index >> 8);
    output[5 * index] = ninfer::ops::detail::unified::nvfp4_scaled_pair_bf16(code, scale);
    unsigned word = 0;
    for (unsigned i = 0; i < 4; ++i) word |= unsigned(code ^ (i * 0x55u)) << (8 * i);
    ninfer::ops::detail::unified::nvfp4_scaled_quad_bf16(word, scale, output + 5 * index + 1);
}
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::uint16_t expected(unsigned code, unsigned scale) {
    constexpr double magnitudes[]{0, 0.5, 1, 1.5, 2, 3, 4, 6};
    const double value = (code & 8) ? -magnitudes[code & 7] : magnitudes[code & 7];
    const unsigned exponent = scale >> 3, mantissa = scale & 7;
    const double multiplier = scale == 127 ? 0 : exponent == 0
        ? std::ldexp(double(mantissa), -9)
        : std::ldexp(1 + double(mantissa) / 8, int(exponent) - 7);
    // These represented products need at most six significand bits and fit BF16 exactly.
    const float represented = static_cast<float>(value * multiplier);
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(represented) >> 16);
}
std::uint32_t expected_pair(unsigned code, unsigned scale) {
    return unsigned(expected(code & 15, scale)) | (unsigned(expected(code >> 4, scale)) << 16);
}
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    std::uint32_t* device = nullptr;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&device), cases * 5 * sizeof(std::uint32_t)));
        decode<<<(cases + 255) / 256, 256>>>(device);
        check(cudaGetLastError());
        std::vector<std::uint32_t> values(cases * 5);
        check(cudaMemcpy(values.data(), device, values.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        check(cudaFree(device)); device = nullptr;
        for (unsigned index = 0; index < cases; ++index) {
            const unsigned code = index & 255, scale = index >> 8;
            for (unsigned part = 0; part < 5; ++part) {
                const unsigned selected = part ? code ^ ((part - 1) * 0x55u) : code;
                if (values[5 * index + part] != expected_pair(selected, scale)) {
                    std::cerr << "NVFP4 scale decode differs: code=" << selected << " scale="
                              << scale << " part=" << part << '\n';
                    return 1;
                }
            }
        }
        std::cout << "NVFP4 pair/quad decode: all 32768 code/scale pairs match independent FP64 oracle\n";
        return 0;
    } catch (const std::exception& error) {
        if (device) cudaFree(device);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
