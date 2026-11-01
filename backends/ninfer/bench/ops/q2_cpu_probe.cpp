// Experimental routed-expert CPU probe, not a product execution route. Build on x86 with
// -O3 -mavx2 -mf16c -fopenmp. Public inputs are stored GGUF Q2_0 blocks and BF16 activations;
// q8_1 and the BF16 middle are implementation arithmetic, not the full FP64 oracle's inputs.
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int H = 2560, W = 640, TOP = 10;
struct A8 { std::array<std::int8_t, 32> q; float d; };
float bf16(float v) {
    auto bits = std::bit_cast<std::uint32_t>(v);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}
A8 quantize(const float* x) {
    float maximum = 0;
    for (int i = 0; i < 32; ++i) { maximum = std::max(maximum, std::abs(x[i])); }
    A8 out{};
    if (maximum == 0) { return out; }
    const auto bits = _cvtss_sh(maximum / 127.0F, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    out.d = _cvtsh_ss(bits);
    for (int i = 0; i < 32; ++i) {
        out.q[i] = static_cast<std::int8_t>(std::round(x[i] / (maximum / 127.0F)));
    }
    return out;
}
std::uint16_t word(const std::uint8_t* p) { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
// Independent exact IEEE binary16 decoder for the FP64 oracle.
double exact_half(std::uint16_t v) {
    const int exponent = (v >> 10) & 31, fraction = v & 1023;
    if (exponent == 31) { throw std::runtime_error("nonfinite weight scale"); }
    const double magnitude = exponent ? std::ldexp(1024.0 + fraction, exponent - 25)
                                      : std::ldexp(double(fraction), -24);
    return v & 0x8000 ? -magnitude : magnitude;
}
std::uint32_t spread(std::uint32_t q) {
    return (q | (q << 6) | (q << 12) | (q << 18)) & 0x03030303U;
}
__m256i codes(const std::uint8_t* q) {
    return _mm256_set_epi32(spread(q[7]), spread(q[6]), spread(q[5]), spread(q[4]),
                            spread(q[3]), spread(q[2]), spread(q[1]), spread(q[0]));
}
int sum32(__m256i v) {
    auto s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
float dot(const std::uint8_t* row, const A8* x, int k) {
    float out = 0;
    const auto ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    for (int b = 0; b < k / 64; ++b, row += 18) {
        const float scale = _cvtsh_ss(word(row));
        for (int half = 0; half < 2; ++half) {
            const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[2*b+half].q.data()));
            const auto q = codes(row + 2 + half * 8);
            const auto product = _mm256_madd_epi16(_mm256_maddubs_epi16(q, a), ones16);
            const auto offset = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, a), ones16);
            out += float(sum32(_mm256_sub_epi32(product, offset))) * scale * x[2*b+half].d;
        }
    }
    return out;
}
double oracle_a8_dot(const std::uint8_t* row, const A8* x, int k) {
    double out = 0;
    for (int i = 0; i < k; ++i) {
        const auto* block = row + (i / 64) * 18;
        const int code = (block[2 + (i % 64) / 4] >> (2 * (i % 4))) & 3;
        out += exact_half(word(block)) * double(code - 1) * x[i / 32].q[i % 32] * x[i / 32].d;
    }
    return out;
}
double oracle_dot(const std::uint8_t* row, const double* x, int k) {
    double out = 0;
    for (int i = 0; i < k; ++i) {
        const auto* block = row + (i / 64) * 18;
        const int code = (block[2 + (i % 64) / 4] >> (2 * (i % 4))) & 3;
        out += exact_half(word(block)) * double(code - 1) * x[i];
    }
    return out;
}
struct Probe {
    const std::vector<std::uint8_t> &gate, &up, &down;
    int columns, experts, copies;
    bool reuse;
    std::vector<A8> x, middle;
    std::vector<double> raw_x;
    std::vector<float> gu, values, result;
    Probe(const std::vector<std::uint8_t>& g, const std::vector<std::uint8_t>& u,
          const std::vector<std::uint8_t>& d, int t, bool shared)
        : gate(g), up(u), down(d), columns(t), experts(shared ? TOP : TOP*t),
          copies(shared ? t : 1), reuse(shared), x(t * H/32), middle(experts * copies * W/32),
          raw_x(t * H), gu(experts * copies * 2*W), values(experts * copies * H), result(t * H) {
        std::array<float, H> input;
        for (int c = 0; c < t; ++c) {
            for (int i = 0; i < H; ++i) {
                input[i] = bf16(std::sin(float(i*13+c*7)) * 1.3F);
                raw_x[c*H+i] = input[i];
            }
            for (int b = 0; b < H/32; ++b) { x[c*H/32+b] = quantize(input.data()+b*32); }
        }
    }
    std::vector<double> oracle() const {
        std::vector<double> out(columns * H);
        #pragma omp parallel for schedule(static)
        for (int token = 0; token < columns; ++token) {
            const double* input = raw_x.data() + token*H;
            for (int routed = 0; routed < TOP; ++routed) {
                const int expert = reuse ? routed : token*TOP+routed;
                std::array<double, W> activation;
                for (int r = 0; r < W; ++r) {
                    const auto offset = (std::size_t(expert)*W+r)*H/64*18;
                    const double g = oracle_dot(gate.data()+offset, input, H);
                    const double u = oracle_dot(up.data()+offset, input, H);
                    activation[r] = g / (1.0 + std::exp(-g)) * u;
                }
                for (int r = 0; r < H; ++r) {
                    const auto* d = down.data()+(std::size_t(expert)*H+r)*W/64*18;
                    out[token*H+r] += double(0.1F) * oracle_dot(d, activation.data(), W);
                }
            }
        }
        return out;
    }
    void execute() {
        #pragma omp parallel for schedule(static)
        for (int item = 0; item < experts * W; ++item) {
            const int e = item / W, r = item % W;
            for (int c = 0; c < copies; ++c) {
                const int token = reuse ? c : e / TOP, p = e*copies+c;
                const auto* g = gate.data()+(std::size_t(e)*W+r)*H/64*18;
                const auto* u = up.data()+(std::size_t(e)*W+r)*H/64*18;
                const auto* a = x.data()+token*H/32;
                gu[p*2*W+r] = dot(g,a,H); gu[p*2*W+W+r] = dot(u,a,H);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < experts*copies; ++p) {
            std::array<float, W> activation;
            for (int r = 0; r < W; ++r) {
                const float g = gu[p*2*W+r], u = gu[p*2*W+W+r];
                activation[r] = bf16(g / (1.0F + std::exp(-g)) * u);
            }
            for (int b = 0; b < W/32; ++b) {
                middle[p*W/32+b] = quantize(activation.data()+b*32);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int item = 0; item < experts * H; ++item) {
            const int e = item / H, r = item % H;
            for (int c = 0; c < copies; ++c) {
                const int p = e*copies+c;
                const auto* d = down.data()+(std::size_t(e)*H+r)*W/64*18;
                const auto* a = middle.data()+p*W/32;
                values[p*H+r] = dot(d,a,W);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < columns * H; ++i) {
            const int c = i/H, r = i%H;
            std::int64_t fixed = 0;
            for (int e = 0; e < TOP; ++e) {
                const int p = reuse ? e*copies+c : c*TOP+e;
                const double value = values[p*H+r];
                fixed += std::llrint(value * double(0.1F) * 4294967296.0);
            }
            result[i] = float(double(fixed) / 4294967296.0);
        }
    }
};
} // namespace

int main(int argc, char** argv) {
    if ((argc != 2 && argc != 4) || (argc == 4 && std::string(argv[2]) != "--oracle-output") ||
        !__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("f16c")) { return 1; }
    std::ofstream oracle_output;
    if (argc == 4) { oracle_output.open(argv[3], std::ios::binary); }
    std::ifstream input(argv[1], std::ios::binary);
    std::array<std::uint32_t,3> header{};
    input.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (header[0] != H || header[1] != W || header[2] < 80) { return 1; }
    std::vector<std::uint8_t> gate(header[2]*std::size_t(W)*H/64*18), up(gate.size());
    std::vector<std::uint8_t> down(header[2]*std::size_t(H)*W/64*18);
    for (auto* bank : {&gate, &up, &down}) {
        input.read(reinterpret_cast<char*>(bank->data()), bank->size());
    }
    if (!input) { return 1; }
    // Exact zero in the packed dot; the full expert oracle uses the original BF16 inputs and
    // independently decoded stored scales, with no internal activation quantization or casts.
    std::array<A8,H/32> zero{};
    if (dot(gate.data(), zero.data(), H) != 0 || oracle_a8_dot(gate.data(), zero.data(), H) != 0) { return 1; }
    std::vector<float> flush(64*1024*1024, 1.0F);
    double checksum = 0;
    for (int t : {1,2,5,8}) for (bool reuse : {true,false}) {
        Probe probe(gate,up,down,t,reuse);
        omp_set_num_threads(8);
        const auto expected = probe.oracle();
        if (argc == 4) {
            const std::array<std::uint32_t, 3> record{std::uint32_t(t), std::uint32_t(reuse),
                                                     std::uint32_t(expected.size())};
            oracle_output.write(reinterpret_cast<const char*>(record.data()), sizeof(record));
            oracle_output.write(reinterpret_cast<const char*>(expected.data()), expected.size() * 8);
            if (!oracle_output) { return 2; }
            continue;
        }
        for (int threads : {1,8,26}) {
            omp_set_num_threads(threads);
            probe.execute();
            double error = 0, norm = 0, largest = 0, peak = 0;
            for (std::size_t i=0; i<expected.size(); ++i) {
                if (!std::isfinite(expected[i]) || !std::isfinite(probe.result[i])) {
                    std::cerr << "FAIL: nonfinite expert output\n";
                    return 2;
                }
                const double diff = double(probe.result[i])-expected[i];
                error += diff*diff; norm += double(expected[i])*expected[i];
                largest = std::max(largest,std::abs(diff));
                peak = std::max(peak, std::abs(expected[i]));
            }
            const double relative = std::sqrt(error/std::max(norm,1e-30));
            const double gross = largest/std::max(peak,1e-15);
            std::cout << "oracle T="<<t<<" reuse="<<reuse<<" threads="<<threads
                      <<" relative_l2="<<relative<<" max_over_peak="<<gross<<'\n';
            // Same full-mathematics criteria as the production GGUF MoE Op qualification.
            if (!std::isfinite(relative) || !std::isfinite(gross) ||
                relative >= 0.04 || gross >= 0.16) { return 2; }
            for (bool cold : {false,true}) {
                std::vector<double> samples;
                for (int rep=0; rep<9; ++rep) {
                    if (cold) {
                        for (std::size_t i=0;i<flush.size();i+=16) { flush[i]+=1; checksum+=flush[i]; }
                    }
                    const auto start=std::chrono::steady_clock::now();
                    probe.execute();
                    samples.push_back(std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count());
                    checksum+=probe.result[rep];
                }
                std::sort(samples.begin(),samples.end());
                std::cout<<"cpu T="<<t<<" reuse="<<reuse<<" threads="<<threads<<" cold="<<cold
                         <<" median_ms="<<samples[4]<<" min_ms="<<samples.front()
                         <<" max_ms="<<samples.back()<<'\n';
            }
        }
    }
    if (argc == 4) { std::cout << "CPU_ORACLE_WRITTEN records=8\n"; }
    else { std::cout << "CPU_PROBE_PASS checksum=" << checksum << '\n'; }
}
