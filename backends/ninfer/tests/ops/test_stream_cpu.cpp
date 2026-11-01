// The streamed native experts' CPU rows (ops/sparse_moe/streamed_cpu.h) against an FP64 oracle of
// independently decoded stored bytes, and every vector backend against the scalar one bit for bit.
// The GPU's bits are the streaming SparseMoe test's (test_sparse_moe --streaming).
#include "ops/sparse_moe/streamed_cpu.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

double exact_half(unsigned bits) {
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    const double value = exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

double exact_e4m3(unsigned bits) {
    const unsigned exponent = (bits >> 3) & 15, mantissa = bits & 7;
    const double value = exponent ? std::ldexp(8.0 + mantissa, int(exponent) - 10)
                                  : std::ldexp(double(mantissa), -9);
    return bits & 0x80 ? -value : value;
}

std::uint16_t bf16(float value) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    return static_cast<std::uint16_t>(word >> 16);
}

// One plane with random stored bytes and its weights decoded in natural order.
struct Plane {
    StreamCpuPlane view;
    std::vector<std::uint8_t> codes, high, scales;
    std::vector<float> divisors;
    std::vector<double> exact; // [row][k]

    Plane(QType format, int rows, int k, unsigned seed) : exact(std::size_t(rows) * k) {
        std::mt19937 random(seed);
        const auto byte = [&] { return std::uint8_t(random() & 255); };
        const auto half = [&] {
            // Positive and negative binary16 scales between 2^-12 and 2^-4.
            const unsigned exponent = 3 + random() % 9, fraction = random() & 1023;
            return std::uint16_t(((random() & 1) << 15) | (exponent << 10) | fraction);
        };
        const int group = format == QType::Q8_G32_FP16 ? 32 : 64;
        const std::size_t groups = std::size_t(rows) * (k / group);
        if (format == QType::NVFP4) {
            codes.resize(std::size_t(rows) * k / 2);
            for (auto& b : codes) { b = byte(); }
            scales.resize(std::size_t(rows) * k / 16);
            for (auto& b : scales) {
                do { b = byte(); } while ((b & 0x7F) == 0x7F || ((b >> 3) & 15) > 9);
            }
            divisors.resize(std::size_t(rows / 128));
            for (auto& d : divisors) { d = 0.5F + float(random() % 1000) / 500.0F; }
        } else {
            codes.resize(groups * 32); // 32 int8 codes or 64 nibbles per group
            for (auto& b : codes) { b = byte(); }
            if (format == QType::Q5_G64_FP16 || format == QType::Q6_G64_FP16) {
                high.resize(groups * (format == QType::Q5_G64_FP16 ? 8 : 16));
                for (auto& b : high) { b = byte(); }
            }
            scales.resize(groups * 2);
            for (std::size_t g = 0; g < groups; ++g) {
                const auto h = half();
                scales[2 * g] = h & 255;
                scales[2 * g + 1] = h >> 8;
            }
        }
        view.format = format;
        view.codes = codes.data();
        view.high = high.empty() ? nullptr : high.data();
        view.scales = scales.data();
        view.divisors = divisors.empty() ? nullptr : divisors.data();
        view.divisor_rows = 128;
        view.rows = rows;
        view.k = k;
        for (int r = 0; r < rows; ++r) {
            for (int v = 0; v < k; ++v) { exact[std::size_t(r) * k + v] = decode(r, v); }
        }
    }

    // Value v of row r from the storage formats' definitions, in natural order.
    double decode(int r, int v) const {
        const int k = view.k;
        switch (view.format) {
        case QType::Q8_G32_FP16: {
            const std::size_t g = std::size_t(r) * (k / 32) + v / 32;
            const double scale = exact_half(scales[2 * g] | (scales[2 * g + 1] << 8));
            return double(std::int8_t(codes[g * 32 + v % 32])) * scale;
        }
        case QType::NVFP4: {
            const unsigned code = (codes[std::size_t(r) * k / 2 + v / 2] >> (4 * (v % 2))) & 15;
            static constexpr double kValues[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
            const double value = code & 8 ? -kValues[code & 7] : kValues[code & 7];
            // The 128x4 tiles: four rows 32 apart share a 16-byte line of four 16-value blocks.
            const int block = v / 16, tile = r / 128, inner = r % 128;
            const std::size_t at = (std::size_t(tile) * (k / 64) + block / 4) * 512 +
                                   (inner % 32) * 16 + (inner / 32) * 4 + block % 4;
            const double coefficient =
                exact_e4m3(scales[at]) * double(1.0F / divisors[std::size_t(r / 128)]);
            return value * coefficient;
        }
        default: {
            const std::size_t g = std::size_t(r) * (k / 64) + v / 64;
            const int at = v % 64;
            unsigned code = (codes[g * 32 + at / 2] >> (4 * (at % 2))) & 15;
            int bits = 4;
            if (view.format == QType::Q5_G64_FP16) {
                code |= ((high[g * 8 + at / 8] >> (at % 8)) & 1u) << 4;
                bits = 5;
            } else if (view.format == QType::Q6_G64_FP16) {
                code |= ((high[g * 16 + at / 4] >> (2 * (at % 4))) & 3u) << 4;
                bits = 6;
            }
            const int value = int(code) - (code & (1u << (bits - 1)) ? (1 << bits) : 0);
            return double(value) * exact_half(scales[2 * g] | (scales[2 * g + 1] << 8));
        }
        }
    }
};

int check(QType format, int k, const char* name) {
    constexpr int kRows = 256, kTokens = 3;
    const Plane plane(format, kRows, k, unsigned(k) * 31u + unsigned(format));
    std::mt19937 random(7);
    std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
    std::vector<std::vector<double>> x(kTokens, std::vector<double>(std::size_t(k)));
    StreamCpuInput input(kTokens, k);
    for (int t = 0; t < kTokens; ++t) {
        if (k == 2048) {
            std::vector<std::uint16_t> words(static_cast<std::size_t>(k));
            for (int v = 0; v < k; ++v) {
                words[std::size_t(v)] = bf16(uniform(random));
                x[std::size_t(t)][std::size_t(v)] =
                    std::bit_cast<float>(std::uint32_t(words[std::size_t(v)]) << 16);
            }
            input.load_bf16(t, words.data());
        } else {
            std::vector<float> values(static_cast<std::size_t>(k));
            for (int v = 0; v < k; ++v) {
                values[std::size_t(v)] = uniform(random);
                x[std::size_t(t)][std::size_t(v)] = values[std::size_t(v)];
            }
            input.load_fp32(t, values.data());
        }
    }
    const float coefficients[kTokens] = {0.25F, -1.5F, 0.0625F};
    int failures = 0;
    std::vector<float> scalar;
    std::string backends;
    for (const GgufCpuBackend backend : {GgufCpuBackend::Scalar, GgufCpuBackend::Avx2}) {
        if (!gguf_cpu_backend_available(backend)) { continue; }
        backends += std::string(backends.empty() ? "" : ", ") + gguf_cpu_backend_name(backend);
        std::vector<float> out(std::size_t(kTokens) * kRows, std::nanf(""));
        std::vector<float*> rows;
        for (int t = 0; t < kTokens; ++t) { rows.push_back(out.data() + std::size_t(t) * kRows); }
        // Two calls over complementary row ranges, one with route weights.
        stream_cpu_rows(plane.view, input, 0, 100, coefficients, rows, backend);
        stream_cpu_rows(plane.view, input, 100, kRows, coefficients, rows, backend);
        for (int t = 0; t < kTokens; ++t) {
            for (int r = 0; r < kRows; ++r) {
                double dot = 0, magnitude = 0;
                for (int v = 0; v < k; ++v) {
                    const double product = plane.exact[std::size_t(r) * k + v] * x[std::size_t(t)][std::size_t(v)];
                    dot += product;
                    magnitude += std::fabs(product);
                }
                const double expected = double(coefficients[t]) * dot;
                const double actual = out[std::size_t(t) * kRows + r];
                const double bound = std::fabs(double(coefficients[t])) * (magnitude * 1e-5 + 1e-30);
                if (!(std::fabs(actual - expected) <= bound)) {
                    if (failures < 5) {
                        std::cerr << name << ' ' << gguf_cpu_backend_name(backend) << " row " << r
                                  << " token " << t << ": " << actual << " vs " << expected << '\n';
                    }
                    ++failures;
                }
            }
        }
        if (scalar.empty()) {
            scalar = out;
        } else if (std::memcmp(scalar.data(), out.data(), out.size() * sizeof(float)) != 0) {
            std::cerr << name << ": " << gguf_cpu_backend_name(backend) << " differs from scalar\n";
            ++failures;
        }
    }
    std::cout << name << (failures ? " FAIL" : " PASS") << " (" << backends << ")\n";
    return failures;
}

int check_refusals() {
    int failures = 0;
    const Plane plane(QType::Q4_G64_FP16, 128, 2048, 1);
    StreamCpuInput input(1, 512);
    std::vector<float> out(128);
    std::vector<float*> rows{out.data()};
    try {
        stream_cpu_rows(plane.view, input, 0, 128, nullptr, rows, GgufCpuBackend::Scalar);
        std::cerr << "a K mismatch was accepted\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)StreamCpuInput(1, 300);
        std::cerr << "k 300 was accepted\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    const bool supported = stream_cpu_supported(QType::Q4_G64_FP16, QType::Q6_G64_FP16) &&
                           stream_cpu_supported(QType::NVFP4, QType::NVFP4) &&
                           !stream_cpu_supported(QType::Q4_G64_FP16, QType::Q8_G32_FP16);
    if (!supported) {
        std::cerr << "unexpected supported format pairs\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    for (const int k : {2048, 512}) {
        const std::string suffix = " k=" + std::to_string(k);
        failures += check(QType::Q4_G64_FP16, k, ("q4" + suffix).c_str());
        failures += check(QType::Q5_G64_FP16, k, ("q5" + suffix).c_str());
        failures += check(QType::Q6_G64_FP16, k, ("q6" + suffix).c_str());
        failures += check(QType::Q8_G32_FP16, k, ("q8" + suffix).c_str());
        failures += check(QType::NVFP4, k, ("nvfp4" + suffix).c_str());
    }
    failures += check_refusals();
    std::cout << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
}
