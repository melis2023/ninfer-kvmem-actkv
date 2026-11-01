// ngram_embed_rows against independently decoded rows: every FP8 E4M3 code (NaN excluded) under
// FP16 scales across the exponent range, every IQ4_NL code under random FP16 scales, and BF16 rows
// bit-exactly.
#include "core/device.h"
#include "ninfer/ops/ngram_rows.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kWidth = 160;

double decode_e4m3(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign ? -magnitude : magnitude;
}

double decode_fp16(std::uint16_t bits) {
    const int sign     = bits >> 15;
    const int exponent = (bits >> 10) & 0x1f;
    const int mantissa = bits & 0x3ff;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 1024.0, -14)
                                           : std::ldexp(1.0 + mantissa / 1024.0, exponent - 15);
    return sign ? -magnitude : magnitude;
}

int run_fp8(int heads, int tokens, std::uint32_t seed) {
    const int rows = heads * tokens;
    const int row_bytes = static_cast<int>(ops::ngram_row_bytes(ops::NgramRowFormat::Fp8E4M3RowScale));
    std::vector<std::uint8_t> staged(static_cast<std::size_t>(rows) * row_bytes);
    std::vector<double> expected(static_cast<std::size_t>(rows) * kWidth);
    std::mt19937 random(seed);
    for (int r = 0; r < rows; ++r) {
        // Zero, subnormals, the normal boundary and largest finite FP16, plus normal scales.
        constexpr std::uint16_t boundaries[] = {0, 1, 0x3ff, 0x400, 0x7bff};
        const std::uint16_t scale = r % 37 < 5 ? boundaries[r % 37] :
            static_cast<std::uint16_t>(((3 + random() % 17) << 10) | (random() & 0x3ff));
        for (int j = 0; j < kWidth; ++j) {
            std::uint8_t code = static_cast<std::uint8_t>((r * kWidth + j) & 0xff);
            if ((code & 0x7f) == 0x7f) code ^= 0x01; // E4M3FN NaN
            if (scale == 0) code = 0;
            staged[static_cast<std::size_t>(r) * row_bytes + j] = code;
            expected[static_cast<std::size_t>(r) * kWidth + j] = decode_e4m3(code) * decode_fp16(scale);
        }
        staged[static_cast<std::size_t>(r) * row_bytes + kWidth]     = static_cast<std::uint8_t>(scale & 0xff);
        staged[static_cast<std::size_t>(r) * row_bytes + kWidth + 1] = static_cast<std::uint8_t>(scale >> 8);
    }
    GuardedDeviceBuffer d_rows(staged.size()), d_out(expected.size() * 2);
    d_rows.copy_from_host(staged.data(), staged.size());
    Tensor t_rows(d_rows.data(), DType::U8, {row_bytes, rows});
    Tensor t_out(d_out.data(), DType::BF16, {heads * kWidth, tokens});
    ops::ngram_embed_rows(t_rows, ops::NgramRowFormat::Fp8E4M3RowScale, heads, t_out, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("ngram fp8 rows", from_device_bf16(d_out.data(), expected.size()),
                                    expected, {1.0e-30, 4.0e-3});
    failures += d_rows.verify_guards("ngram fp8 rows input");
    failures += d_out.verify_guards("ngram fp8 rows output");
    return failures;
}

// ggml's IQ4_NL codebook, restated for the oracle.
constexpr int kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                            1,    13,   25,  38,  53,  69,  89,  113};

int run_iq4nl(int heads, int tokens, std::uint32_t seed) {
    const int rows      = heads * tokens;
    const int row_bytes = static_cast<int>(ops::ngram_row_bytes(ops::NgramRowFormat::Iq4Nl));
    std::vector<std::uint8_t> staged(static_cast<std::size_t>(rows) * row_bytes);
    std::vector<double> expected(static_cast<std::size_t>(rows) * kWidth);
    std::mt19937 random(seed);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < kWidth / 32; ++b) {
            std::uint8_t* block = staged.data() + static_cast<std::size_t>(r) * row_bytes + b * 18;
            // Positive and negative normal FP16 scales from 2^-14 to 2^0.
            const std::uint16_t scale = static_cast<std::uint16_t>(
                ((random() & 1) << 15) | ((1 + random() % 15) << 10) | (random() & 0x3ff));
            block[0] = static_cast<std::uint8_t>(scale & 0xff);
            block[1] = static_cast<std::uint8_t>(scale >> 8);
            for (int i = 0; i < 16; ++i) { block[2 + i] = static_cast<std::uint8_t>(random()); }
            for (int i = 0; i < 32; ++i) {
                const int code = i < 16 ? (block[2 + i] & 0xf) : (block[2 + i - 16] >> 4);
                expected[static_cast<std::size_t>(r) * kWidth + b * 32 + i] =
                    decode_fp16(scale) * kIq4Nl[code];
            }
        }
    }
    GuardedDeviceBuffer d_rows(staged.size()), d_out(expected.size() * 2);
    d_rows.copy_from_host(staged.data(), staged.size());
    Tensor t_rows(d_rows.data(), DType::U8, {row_bytes, rows});
    Tensor t_out(d_out.data(), DType::BF16, {heads * kWidth, tokens});
    ops::ngram_embed_rows(t_rows, ops::NgramRowFormat::Iq4Nl, heads, t_out, nullptr);
    cuda_synchronize();
    int failures =
        verify_pointwise("ngram iq4_nl rows", from_device_bf16(d_out.data(), expected.size()),
                         expected, {1.0e-30, 4.0e-3});
    failures += d_rows.verify_guards("ngram iq4_nl rows input");
    failures += d_out.verify_guards("ngram iq4_nl rows output");
    return failures;
}

int run_bf16(int heads, int tokens, std::uint32_t seed) {
    const int rows = heads * tokens;
    std::vector<std::uint16_t> staged(static_cast<std::size_t>(rows) * kWidth);
    std::mt19937 random(seed);
    for (auto& v : staged) {
        v = static_cast<std::uint16_t>(random());
        if ((v & 0x7f80) == 0x7f80) v &= 0xbfff; // keep finite
    }
    GuardedDeviceBuffer d_rows(staged.size() * 2), d_out(staged.size() * 2);
    d_rows.copy_from_host(staged.data(), d_rows.bytes());
    Tensor t_rows(d_rows.data(), DType::U8, {kWidth * 2, rows});
    Tensor t_out(d_out.data(), DType::BF16, {heads * kWidth, tokens});
    ops::ngram_embed_rows(t_rows, ops::NgramRowFormat::Bf16, heads, t_out, nullptr);
    cuda_synchronize();
    int failures = verify_exact("ngram bf16 rows", from_device<std::uint16_t>(d_out.data(), staged.size()), staged);
    failures += d_out.verify_guards("ngram bf16 rows output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_fp8(16, 1, 6100u);
    failures += run_fp8(16, 33, 6101u);
    failures += run_iq4nl(16, 1, 6300u);
    failures += run_iq4nl(16, 29, 6301u);
    failures += run_bf16(16, 1, 6200u);
    failures += run_bf16(16, 17, 6201u);
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " ngram_embed_rows\n";
    return failures == 0 ? 0 : 1;
}
