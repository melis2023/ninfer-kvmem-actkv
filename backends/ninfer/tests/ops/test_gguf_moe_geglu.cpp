// Gemma 4's routed experts over Q4_0 banks through the GGUF bridge, driven as its executor drives
// them: a [gate; up] parent of 2 x 704 rows over 2816 inputs with a GeLU-tanh gate, and a down
// projection of 2816 rows over 704 inputs, a K that is no whole number of the matrix kernel's
// 256-value steps (its last step reads the next row, then the zero tail after the bank). Up to
// eight tokens take the vector kernels, wider calls the matrix kernel. Each stage is held against
// an FP64 product of the weights decoded on the host: the [gate; up] rows, the down rows of the
// middle the products themselves rounded to BF16, and the routed weighted sum.

#include "core/arena.h"
#include "core/expert_delivery.h"
#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gguf = ops::gguf;

constexpr int kHidden = 2816, kWidth = 704, kExperts = 16, kTopK = 8;
constexpr int kBlock = 18, kBlockValues = 32; // Q4_0
constexpr std::size_t kTail = 512;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// Q4_0 blocks with finite binary16 scales; their exact values decoded on the host.
struct Bank {
    int rows = 0, k = 0;
    std::int64_t row_bytes = 0, expert_bytes = 0;
    DeviceBuffer bytes;
    std::unique_ptr<PinnedHostBuffer> pinned;
    std::vector<float> exact; // [experts][rows][k]

    Bank(int rows_, int k_, std::mt19937& rng) : rows(rows_), k(k_) {
        row_bytes    = std::int64_t(k / kBlockValues) * kBlock;
        expert_bytes = row_bytes * rows;
        std::vector<std::uint8_t> host(std::size_t(expert_bytes) * kExperts);
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_real_distribution<float> scale(0.002f, 0.02f);
        exact.resize(std::size_t(kExperts) * rows * k);
        for (std::size_t b = 0; b < host.size() / kBlock; ++b) {
            std::uint8_t* block = host.data() + b * kBlock;
            const __half d      = __float2half(scale(rng));
            std::memcpy(block, &d, 2);
            for (int i = 2; i < kBlock; ++i) { block[i] = std::uint8_t(byte(rng)); }
            const float dv = __half2float(d);
            for (int j = 0; j < 16; ++j) {
                exact[b * kBlockValues + j]      = dv * float((block[2 + j] & 15) - 8);
                exact[b * kBlockValues + j + 16] = dv * float((block[2 + j] >> 4) - 8);
            }
        }
        pinned = std::make_unique<PinnedHostBuffer>(host.size());
        std::memcpy(pinned->data(), host.data(), host.size());
        bytes = DeviceBuffer(host.size() + kTail);
        check(cudaMemset(bytes.p, 0, host.size() + kTail), "zero a bank");
        bytes.copy_from_host(host.data(), host.size());
    }

    [[nodiscard]] const float* row(int expert, int r) const {
        return exact.data() + (std::size_t(expert) * rows + r) * k;
    }
    // Device table of expert bases, offset by `first_row` rows.
    [[nodiscard]] DeviceBuffer table(int first_row) const {
        std::vector<const void*> pointers(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            pointers[e] = static_cast<const std::uint8_t*>(bytes.p) + e * expert_bytes +
                          first_row * row_bytes;
        }
        DeviceBuffer out(pointers.size() * sizeof(void*));
        out.copy_from_host(pointers.data(), pointers.size() * sizeof(void*));
        return out;
    }
};

double gelu_tanh(double x) {
    return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

struct Error {
    double rms = 0, max = 0;
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

int report(const char* what, int tokens, const Error& e, double rms_bound, double max_bound) {
    const bool ok = std::isfinite(e.rms) && e.rms < rms_bound && e.max < max_bound;
    std::cout << (ok ? "OK   " : "FAIL ") << what << " T=" << tokens << " rms=" << e.rms
              << " max=" << e.max << '\n';
    return ok ? 0 : 1;
}

int run(int tokens, const Bank& gate_up, const Bank& down, std::mt19937& rng,
        ExpertDelivery* delivery = nullptr) {
    const int pairs = tokens * kTopK;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> unit(0.05f, 1.0f);
    std::vector<__nv_bfloat16> x(std::size_t(tokens) * kHidden);
    for (auto& v : x) { v = __float2bfloat16(normal(rng)); }
    std::vector<std::int32_t> ids(pairs);
    std::vector<float> weights(pairs);
    std::vector<int> pool(kExperts);
    std::iota(pool.begin(), pool.end(), 0);
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < kTopK; ++j) {
            ids[t * kTopK + j]     = pool[j];
            weights[t * kTopK + j] = unit(rng);
        }
    }

    // FP64 [gate; up] rows of every pair.
    std::vector<double> want_gu(std::size_t(pairs) * 2 * kWidth);
    for (int p = 0; p < pairs; ++p) {
        const __nv_bfloat16* xt = x.data() + std::size_t(p / kTopK) * kHidden;
        for (int r = 0; r < 2 * kWidth; ++r) {
            const float* w = gate_up.row(ids[p], r);
            double acc     = 0;
            for (int i = 0; i < kHidden; ++i) { acc += double(w[i]) * __bfloat162float(xt[i]); }
            want_gu[std::size_t(p) * 2 * kWidth + r] = acc;
        }
    }
    std::vector<double> want_middle(std::size_t(pairs) * kWidth);
    for (int p = 0; p < pairs; ++p) {
        for (int r = 0; r < kWidth; ++r) {
            const double* row = want_gu.data() + std::size_t(p) * 2 * kWidth;
            want_middle[std::size_t(p) * kWidth + r] = gelu_tanh(row[r]) * row[kWidth + r];
        }
    }

    DeviceBuffer d_x(x.size() * 2), d_ids(pairs * 4), d_w(pairs * 4);
    d_x.copy_from_host(x.data(), x.size() * 2);
    d_ids.copy_from_host(ids.data(), pairs * 4);
    d_w.copy_from_host(weights.data(), pairs * 4);
    DeviceBuffer routing_space(gguf::moe_routing_bytes(kExperts, pairs));
    const gguf::MoeRouting routing = gguf::moe_sort_routes(
        static_cast<const std::int32_t*>(d_ids.p), pairs, kExperts, routing_space.p, nullptr);
    const int max_active = std::min(pairs, kExperts);
    const DeviceBuffer gu_table = gate_up.table(0), gate_table = gate_up.table(0),
                       up_table = gate_up.table(kWidth), down_table = down.table(0);
    if (delivery) {
        (void)delivery->begin(0, static_cast<const std::int32_t*>(d_ids.p), pairs, nullptr);
        delivery->await(nullptr);
    }
    const auto* delivered = delivery ? delivery->tables(0) : nullptr;
    const gguf::MoeTable t_gu{delivered ? delivered + 2 * kExperts : static_cast<const void* const*>(gu_table.p), gate_up.row_bytes,
                              2 * kWidth, kHidden};
    const gguf::MoeTable t_gate{delivered ? delivered : static_cast<const void* const*>(gate_table.p), gate_up.row_bytes,
                                kWidth, kHidden};
    const gguf::MoeTable t_up{delivered ? delivered + kExperts : static_cast<const void* const*>(up_table.p), gate_up.row_bytes,
                              kWidth, kHidden};
    const gguf::MoeTable t_down{delivered ? delivered + 3 * kExperts : static_cast<const void* const*>(down_table.p), down.row_bytes,
                                kHidden, kWidth};
    DeviceBuffer d_middle(std::size_t(pairs) * kWidth * 2);
    std::vector<__nv_bfloat16> middle(std::size_t(pairs) * kWidth);
    std::vector<float> got_rows(std::size_t(pairs) * kHidden);
    std::vector<float> got_sum(std::size_t(tokens) * kHidden);
    int failures = 0;

    if (tokens <= gguf::kMaxVectorColumns) {
        const int chunk = tokens <= 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8;
        DeviceBuffer act(gguf::vector_activation_bytes(kHidden, tokens));
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(d_x.p), kHidden, tokens,
                                         nullptr, act.p, nullptr);
        gguf::moe_vector_swiglu(gguf::GgmlType::Q4_0, t_gate, t_up, routing, max_active, kTopK,
                                act.p, tokens, static_cast<__nv_bfloat16*>(d_middle.p), chunk,
                                nullptr, gguf::GateActivation::GeluTanh);
        check(cudaDeviceSynchronize(), "vector geglu");
        d_middle.copy_to_host(middle.data(), middle.size() * 2);
        DeviceBuffer act2(gguf::vector_activation_bytes(kWidth, pairs));
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(d_middle.p), kWidth,
                                         pairs, nullptr, act2.p, nullptr);
        DeviceBuffer d_rows(got_rows.size() * 4), d_fixed(got_sum.size() * 8);
        gguf::MoeOutput rows_out;
        rows_out.f32 = static_cast<float*>(d_rows.p);
        gguf::moe_vector_product(gguf::GgmlType::Q4_0, t_down, routing, max_active, 1, kTopK,
                                 act2.p, pairs, rows_out, chunk, nullptr);
        check(cudaMemset(d_fixed.p, 0, got_sum.size() * 8), "zero sum");
        gguf::MoeOutput sum_out;
        sum_out.weighted = static_cast<unsigned long long*>(d_fixed.p);
        sum_out.weights  = static_cast<const float*>(d_w.p);
        gguf::moe_vector_product(gguf::GgmlType::Q4_0, t_down, routing, max_active, 1, kTopK,
                                 act2.p, pairs, sum_out, chunk, nullptr);
        check(cudaDeviceSynchronize(), "vector down");
        d_rows.copy_to_host(got_rows.data(), got_rows.size() * 4);
        std::vector<std::int64_t> fixed(got_sum.size());
        d_fixed.copy_to_host(fixed.data(), fixed.size() * 8);
        for (std::size_t i = 0; i < fixed.size(); ++i) {
            got_sum[i] = float(double(fixed[i]) / 4294967296.0);
        }
        std::vector<float> got_middle(middle.size());
        for (std::size_t i = 0; i < middle.size(); ++i) {
            got_middle[i] = __bfloat162float(middle[i]);
        }
        failures += report("vector geglu", tokens, compare(got_middle, want_middle), 4e-2, 1.6e-1);
    } else {
        DeviceBuffer act(gguf::moe_matrix_activation_bytes(kHidden, pairs));
        gguf::quantize_moe_matrix_activation(gguf::GgmlType::Q4_0,
                                             static_cast<const __nv_bfloat16*>(d_x.p), kHidden,
                                             routing, pairs, kTopK, act.p, nullptr);
        DeviceBuffer d_gu(std::size_t(pairs) * 2 * kWidth * 4);
        gguf::moe_matrix_product(gguf::GgmlType::Q4_0, t_gu, routing, max_active, pairs, true,
                                 act.p, static_cast<float*>(d_gu.p), nullptr);
        check(cudaDeviceSynchronize(), "matrix gate_up");
        std::vector<float> gu(std::size_t(pairs) * 2 * kWidth);
        d_gu.copy_to_host(gu.data(), gu.size() * 4);
        failures += report("matrix gate_up", tokens, compare(gu, want_gu), 2e-2, 8e-2);
        for (int p = 0; p < pairs; ++p) {
            for (int r = 0; r < kWidth; ++r) {
                const float* row = gu.data() + std::size_t(p) * 2 * kWidth;
                middle[std::size_t(p) * kWidth + r] =
                    __float2bfloat16(float(gelu_tanh(row[r]) * row[kWidth + r]));
            }
        }
        d_middle.copy_from_host(middle.data(), middle.size() * 2);
        DeviceBuffer act2(gguf::moe_matrix_activation_bytes(kWidth, pairs));
        gguf::quantize_moe_matrix_activation(gguf::GgmlType::Q4_0,
                                             static_cast<const __nv_bfloat16*>(d_middle.p), kWidth,
                                             routing, pairs, 1, act2.p, nullptr);
        DeviceBuffer d_rows(got_rows.size() * 4);
        gguf::moe_matrix_product(gguf::GgmlType::Q4_0, t_down, routing, max_active, pairs, true,
                                 act2.p, static_cast<float*>(d_rows.p), nullptr);
        check(cudaDeviceSynchronize(), "matrix down");
        d_rows.copy_to_host(got_rows.data(), got_rows.size() * 4);
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < kHidden; ++r) {
                double s = 0;
                for (int j = 0; j < kTopK; ++j) {
                    const int p = t * kTopK + j;
                    s += double(weights[p]) * got_rows[std::size_t(p) * kHidden + r];
                }
                got_sum[std::size_t(t) * kHidden + r] = float(s);
            }
        }
    }

    // The down rows of the BF16 middle the products produced, and the routed sum.
    std::vector<double> want_rows(std::size_t(pairs) * kHidden);
    std::vector<double> want_sum(std::size_t(tokens) * kHidden, 0.0);
    for (int p = 0; p < pairs; ++p) {
        const __nv_bfloat16* m = middle.data() + std::size_t(p) * kWidth;
        for (int r = 0; r < kHidden; ++r) {
            const float* w = down.row(ids[p], r);
            double acc     = 0;
            for (int i = 0; i < kWidth; ++i) { acc += double(w[i]) * __bfloat162float(m[i]); }
            want_rows[std::size_t(p) * kHidden + r] = acc;
            want_sum[std::size_t(p / kTopK) * kHidden + r] += weights[p] * acc;
        }
    }
    const char* path = tokens <= gguf::kMaxVectorColumns ? "vector" : "matrix";
    failures += report((std::string(path) + " down").c_str(), tokens,
                       compare(got_rows, want_rows), 2e-2, 8e-2);
    failures += report((std::string(path) + " weighted sum").c_str(), tokens,
                       compare(got_sum, want_sum), 2e-2, 8e-2);
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(4242);
        const Bank gate_up(2 * kWidth, kHidden, rng);
        const Bank down(kHidden, kWidth, rng);
        int failures = 0;
        for (const int tokens : {1, 3, 8, 9, 26, 130}) { failures += run(tokens, gate_up, down, rng); }
        DeviceContext device(0);
        RankContext rank = device.rank(0);
        rank.stream = nullptr; // the oracle harness deliberately uses the legacy stream
        const auto path = std::filesystem::temp_directory_path() /
            ("ninfer-gemma-adapter-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
        struct Cleanup { std::filesystem::path p; ~Cleanup() { std::filesystem::remove_all(p); } } cleanup{path};
        for (bool disk : {false, true}) {
            ExpertDeliveryLayer layer;
            layer.aliases = {{0, 0}, {0, std::uint64_t(kWidth * gate_up.row_bytes)}, {0, 0}, {1, 0}};
            std::vector<std::filesystem::path> files;
            for (const Bank* bank : {&gate_up, &down}) {
                if (disk) {
                    files.push_back(path / std::to_string(files.size()));
                    std::ofstream file(files.back(), std::ios::binary);
                    file.write(static_cast<const char*>(bank->pinned->data()), bank->pinned->size());
                    if (!file) { throw std::runtime_error("adapter fixture write failed"); }
                }
            }
            for (int e = 0; e < kExperts; ++e) {
                std::vector<ExpertBlob> blobs;
                for (const Bank* bank : {&gate_up, &down}) {
                    ExpertBlob blob;
                    blob.bytes = bank->expert_bytes;
                    blob.zero_tail = kTail;
                    if (disk) {
                        blob.extents = {{blobs.size(), std::uint64_t(e * bank->expert_bytes), blob.bytes}};
                    } else {
                        blob.pinned = static_cast<const std::byte*>(bank->pinned->data()) + e * bank->expert_bytes;
                        blob.pinned_region = reinterpret_cast<std::uintptr_t>(bank->pinned->data());
                    }
                    blobs.push_back(std::move(blob));
                }
                layer.experts.push_back(std::move(blobs));
            }
            ExpertDeliveryOptions options;
            options.pair_capacity = 9 * kTopK;
            options.cache_bytes = 8ULL << 20;
            ExpertDelivery delivery(rank, files, {std::move(layer)}, options);
            std::cout << (disk ? "disk" : "host") << " delivered Gemma FP64 oracle\n";
            for (int tokens : {1, 8, 9}) { failures += run(tokens, gate_up, down, rng, &delivery); }
            delivery.drain();
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGUF MoE GeGLU\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGUF MoE GeGLU: " << error.what() << '\n';
        return 1;
    }
}
