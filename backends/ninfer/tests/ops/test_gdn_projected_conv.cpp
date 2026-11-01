// The GDN convolution over projected rows (gdn_projected_conv.h), the step the fused input
// projections run between their GEMM and the recurrence, at every registered channel profile:
// the 27B and 35B layers and a 27B layer's share on each of two and four tensor-parallel ranks.
//
// The oracle is the definition in FP64 on the BF16 inputs: out = silu(w0*s0 + w1*s1 + w2*s2 + w3*p)
// per channel and column, from the window the row's initial slot names, rounded once to BF16 and
// split into query, key and value rows. The kernel accumulates in FP32, so an output may differ by
// one BF16 step where the FP32 sum lands on the other side of a rounding boundary. Invalid tail
// columns must be exact zeros. The snapshot form must publish after valid column j the window
// (s1, s2, p) to slot base+j exactly, as BF16 copies; the record form must leave the state alone.

#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

bool cuda_unavailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) != cudaSuccess || count == 0;
}

std::uint16_t bits(float value) {
    const __nv_bfloat16 h = __float2bfloat16(value);
    std::uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

float value(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &wide, sizeof(out));
    return out;
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Device() { (void)cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

template <class T>
void upload(Device& device, const std::vector<T>& host) {
    check(cudaMemcpy(device.p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
          "upload");
}

template <class T>
std::vector<T> download(const Device& device, std::size_t count) {
    std::vector<T> host(count);
    check(cudaMemcpy(host.data(), device.p, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return host;
}

struct Profile {
    std::int32_t query;
    std::int32_t key;
    std::int32_t value;
    std::int32_t channels() const { return query + key + value; }
};

int run_case(const Profile& profile, std::int32_t width, std::int32_t batch, bool masked,
             bool snapshot, std::uint32_t seed) {
    const std::int32_t channels = profile.channels();
    const std::int32_t slots    = batch * (width + 1);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const auto random_bits = [&](std::size_t count, float scale) {
        std::vector<std::uint16_t> out(count);
        for (auto& b : out) { b = bits(scale * normal(rng)); }
        return out;
    };
    const auto projected = random_bits(static_cast<std::size_t>(channels) * width * batch, 1.0F);
    const auto weight    = random_bits(static_cast<std::size_t>(channels) * 4, 0.5F);
    const auto states    = random_bits(static_cast<std::size_t>(channels) * 3 * slots, 1.0F);
    // Row b starts from slot b and publishes into [batch + b * width, ...).
    std::vector<std::int32_t> valid(static_cast<std::size_t>(batch), width);
    if (masked) {
        for (std::int32_t b = 0; b < batch; ++b) { valid[b] = 1 + (b * 5 + 3) % width; }
    }
    std::vector<std::int32_t> initial(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> base(static_cast<std::size_t>(batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        initial[b] = b;
        base[b]    = batch + b * width;
    }

    Device d_projected(projected.size() * 2), d_weight(weight.size() * 2),
        d_states(states.size() * 2), d_valid(valid.size() * 4), d_initial(initial.size() * 4),
        d_base(base.size() * 4);
    const std::size_t columns = static_cast<std::size_t>(width) * batch;
    Device d_query(columns * profile.query * 2), d_key(columns * profile.key * 2),
        d_value(columns * profile.value * 2);
    upload(d_projected, projected);
    upload(d_weight, weight);
    upload(d_states, states);
    upload(d_valid, valid);
    upload(d_initial, initial);
    upload(d_base, base);

    const Tensor projected_t(d_projected.p, DType::BF16, {channels, width, batch});
    const Tensor weight_t(d_weight.p, DType::BF16, {channels, 4});
    Tensor states_t(d_states.p, DType::BF16, {channels, 3, slots});
    const Tensor valid_t = masked ? Tensor(d_valid.p, DType::I32, {batch}) : Tensor{};
    const Tensor initial_t(d_initial.p, DType::I32, {batch});
    const Tensor base_t(d_base.p, DType::I32, {batch});
    Tensor query_t(d_query.p, DType::BF16, {profile.query, width, batch});
    Tensor key_t(d_key.p, DType::BF16, {profile.key, width, batch});
    Tensor value_t(d_value.p, DType::BF16, {profile.value, width, batch});
    if (snapshot) {
        ops::detail::gdn_projected_conv_snapshot_launch(projected_t, weight_t, states_t, valid_t,
                                                        initial_t, base_t, query_t, key_t,
                                                        value_t, nullptr);
    } else {
        ops::detail::gdn_projected_conv_record_launch(projected_t, weight_t, states_t, valid_t,
                                                      initial_t, query_t, key_t, value_t, nullptr);
    }
    check(cudaDeviceSynchronize(), "launch");

    const auto query  = download<std::uint16_t>(d_query, columns * profile.query);
    const auto key    = download<std::uint16_t>(d_key, columns * profile.key);
    const auto values = download<std::uint16_t>(d_value, columns * profile.value);
    const auto after  = download<std::uint16_t>(d_states, states.size());
    const auto output = [&](std::int64_t column, std::int32_t row) {
        if (row < profile.query) { return query[column * profile.query + row]; }
        if (row < profile.query + profile.key) {
            return key[column * profile.key + row - profile.query];
        }
        return values[column * profile.value + row - profile.query - profile.key];
    };
    std::vector<std::uint16_t> want_states = states;
    int far = 0, exact_misses = 0, state_misses = 0;
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int64_t slot = static_cast<std::int64_t>(initial[b]) * channels * 3;
        for (std::int32_t row = 0; row < channels; ++row) {
            double s[3] = {value(states[slot + row]), value(states[slot + channels + row]),
                           value(states[slot + 2LL * channels + row])};
            const double w[4] = {value(weight[row]), value(weight[channels + row]),
                                 value(weight[2LL * channels + row]),
                                 value(weight[3LL * channels + row])};
            for (std::int32_t token = 0; token < width; ++token) {
                const std::int64_t column = static_cast<std::int64_t>(b) * width + token;
                const std::uint16_t got   = output(column, row);
                if (token >= valid[b]) {
                    exact_misses += got != bits(0.0F);
                    continue;
                }
                const double p    = value(projected[column * channels + row]);
                const double conv = w[0] * s[0] + w[1] * s[1] + w[2] * s[2] + w[3] * p;
                const double silu = conv / (1.0 + std::exp(-conv));
                const double step = std::ldexp(1.0, std::ilogb(std::fabs(silu) + 1e-30) - 7);
                far += std::fabs(static_cast<double>(value(got)) - silu) > step;
                s[0] = s[1];
                s[1] = s[2];
                s[2] = p;
                if (snapshot) {
                    const std::int64_t out = static_cast<std::int64_t>(base[b] + token) * channels * 3;
                    want_states[out + row]                 = bits(static_cast<float>(s[0]));
                    want_states[out + channels + row]      = bits(static_cast<float>(s[1]));
                    want_states[out + 2LL * channels + row] = bits(static_cast<float>(s[2]));
                }
            }
        }
    }
    for (std::size_t i = 0; i < after.size(); ++i) { state_misses += after[i] != want_states[i]; }
    if (far != 0 || exact_misses != 0 || state_misses != 0) {
        std::cerr << "channels " << channels << " width " << width << " batch " << batch
                  << (masked ? " masked" : "") << (snapshot ? " snapshot" : " record") << ": "
                  << far << " outputs off by more than one BF16 step, " << exact_misses
                  << " nonzero tail outputs, " << state_misses << " wrong state values\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        const Profile profiles[] = {
            {2048, 2048, 6144}, {2048, 2048, 4096}, {1024, 1024, 3072}, {512, 512, 1536}};
        std::uint32_t seed = 3100;
        for (const Profile& profile : profiles) {
            for (const bool snapshot : {true, false}) {
                // The four-column single-row route, single rows at other widths, and batches.
                failures += run_case(profile, 4, 1, false, snapshot, ++seed);
                failures += run_case(profile, 1, 1, false, snapshot, ++seed);
                failures += run_case(profile, 7, 1, true, snapshot, ++seed);
                failures += run_case(profile, 16, 1, false, snapshot, ++seed);
                failures += run_case(profile, 4, 3, true, snapshot, ++seed);
                failures += run_case(profile, 2, 8, false, snapshot, ++seed);
            }
        }
        try {
            failures += run_case({1024, 1024, 2048}, 2, 1, false, true, ++seed) == 0;
            std::cerr << "an unregistered profile was accepted\n";
        } catch (const std::invalid_argument&) {}
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_projected_conv\n";
    return failures == 0 ? 0 : 1;
}
