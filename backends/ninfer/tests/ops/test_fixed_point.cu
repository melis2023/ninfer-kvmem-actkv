// The integer 2^-32 fixed-point rounding against its exact definition, the FP64 form it replaces:
// random values over every exponent, subnormals, signed zeros, infinities, NaNs, the clamp and
// exact half-way products, on the device and on the host.
#include "ops/common/fixed_point.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace ninfer::test;

__global__ void products(const float* a, const float* b, long long* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = ninfer::ops::fixed_product(a[i], b[i]); }
}

long long reference(float a, float b) {
    const double scaled = std::fmin(std::fmax(double(a) * double(b) * 4294967296.0, -4.0e18), 4.0e18);
    return std::llrint(scaled); // the default rounding: to nearest, ties to even
}

float any_float(std::mt19937& rng) {
    // Every finite exponent equally often, plus the special values.
    const std::uint32_t pick = rng() % 64;
    if (pick == 0) { return std::numeric_limits<float>::infinity(); }
    if (pick == 1) { return -std::numeric_limits<float>::infinity(); }
    if (pick == 2) { return std::numeric_limits<float>::quiet_NaN(); }
    if (pick == 3) { return 0.0F; }
    if (pick == 4) { return -0.0F; }
    const std::uint32_t exponent = rng() % 255; // 0 (subnormal) .. 254
    const std::uint32_t bits = (rng() & 0x807FFFFFu) | (exponent << 23);
    return std::bit_cast<float>(bits);
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        std::fesetround(FE_TONEAREST);
        std::mt19937 rng(20261009);
        std::vector<float> a, b;
        constexpr int kRandom = 1 << 20;
        for (int i = 0; i < kRandom; ++i) {
            a.push_back(any_float(rng));
            b.push_back(any_float(rng));
        }
        // Realistic magnitudes: activations times routing weights.
        std::normal_distribution<float> normal(0.0F, 4.0F);
        std::uniform_real_distribution<float> weight(0.0F, 1.0F);
        for (int i = 0; i < kRandom; ++i) {
            a.push_back(normal(rng));
            b.push_back(weight(rng));
        }
        // Exact half-way products at the rounding bit, odd and even, both signs; the clamp edge.
        for (int shift = 33; shift <= 60; ++shift) {
            for (const float m : {1.0F, 3.0F, 5.0F, 7.0F, -1.0F, -3.0F}) {
                a.push_back(std::ldexp(m, -shift));
                b.push_back(1.0F);
                a.push_back(std::ldexp(m, -shift / 2));
                b.push_back(std::ldexp(1.0F, -(shift - shift / 2)));
            }
        }
        for (const float v : {4.0e18F / 4294967296.0F, 9.3132257e8F, 9.3132258e8F, 1.0e9F, -1.0e9F}) {
            a.push_back(v);
            b.push_back(1.0F);
        }
        const int n = static_cast<int>(a.size());
        float *da = nullptr, *db = nullptr;
        long long* dout = nullptr;
        cuda_check(cudaMalloc(&da, n * sizeof(float)), "cudaMalloc a");
        cuda_check(cudaMalloc(&db, n * sizeof(float)), "cudaMalloc b");
        cuda_check(cudaMalloc(&dout, n * sizeof(long long)), "cudaMalloc out");
        cuda_check(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy a");
        cuda_check(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy b");
        products<<<(n + 255) / 256, 256>>>(da, db, dout, n);
        cuda_check_last_launch("products");
        std::vector<long long> out(n);
        cuda_check(cudaMemcpy(out.data(), dout, n * sizeof(long long), cudaMemcpyDeviceToHost),
                   "copy out");
        cuda_check(cudaFree(da), "free a");
        cuda_check(cudaFree(db), "free b");
        cuda_check(cudaFree(dout), "free out");
        std::size_t failures = 0;
        for (int i = 0; i < n; ++i) {
            const long long expected = reference(a[i], b[i]);
            const long long host     = ninfer::ops::fixed_product(a[i], b[i]);
            if (out[i] == expected && host == expected) { continue; }
            if (++failures <= 8) {
                std::cerr << "fixed product of " << a[i] << " and " << b[i] << ": device " << out[i]
                          << ", host " << host << ", expected " << expected << '\n';
            }
        }
        if (failures != 0) {
            std::cerr << failures << " of " << n << " fixed-point products differ\n";
            return 1;
        }
        std::cout << "fixed_product matches the FP64 form for all " << n << " inputs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixed point test: " << error.what() << '\n';
        return 1;
    }
}
