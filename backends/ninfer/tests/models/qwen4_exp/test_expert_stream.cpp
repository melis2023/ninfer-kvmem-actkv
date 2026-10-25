// Disk-resident experts: after prepare(), every table entry of an expert the call routed to points
// at a device slot holding exactly that expert's bytes from the files, including experts that
// straddle the boundary between two files, whatever was evicted to make room, and every down matrix
// is followed by zero bytes even where a larger down of another layer was before (the expert matrix
// kernel reads past a down row's end); a call that routes to more experts than the cache holds is
// refused; and experts already resident are not read again.
#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/expert_stream.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

constexpr int kExperts = 32;

int run() {
    DeviceContext device;
    std::mt19937 random(9300u);
    const std::string stem = "ninfer_expert_stream_" + std::to_string(std::random_device{}());
    // Two layers; layer 1's experts are smaller, as mixed block types are. Each projection of each
    // expert is a distinct random byte run at its own offset of one logical byte stream.
    const std::int64_t sizes[2][3] = {{3000, 3000, 1500}, {1200, 2400, 900}};
    std::vector<std::uint8_t> file_bytes;
    std::vector<std::vector<std::uint64_t>> logical(6);
    for (int layer = 0; layer < 2; ++layer) {
        for (int k = 0; k < 3; ++k) {
            for (int e = 0; e < kExperts; ++e) {
                const std::uint64_t offset = file_bytes.size() + 17; // unaligned on purpose
                file_bytes.resize(offset + sizes[layer][k]);
                for (std::int64_t i = 0; i < sizes[layer][k]; ++i) {
                    file_bytes[offset + i] = std::uint8_t(random());
                }
                logical[layer * 3 + k].push_back(offset);
            }
        }
    }
    // The stream is stored as three files cut inside two experts (layer 0's up of expert 5 and
    // layer 1's down of expert 20), which then straddle a file boundary.
    const std::vector<std::uint64_t> cuts = {logical[1][5] + 1234, logical[5][20] + 450,
                                             file_bytes.size()};
    std::vector<std::filesystem::path> paths;
    for (std::size_t f = 0; f < cuts.size(); ++f) {
        const std::uint64_t begin = f == 0 ? 0 : cuts[f - 1];
        paths.push_back(std::filesystem::temp_directory_path() /
                        (stem + "." + std::to_string(f) + ".bin"));
        std::ofstream out(paths.back(), std::ios::binary);
        out.write(reinterpret_cast<const char*>(file_bytes.data() + begin),
                  static_cast<std::streamsize>(cuts[f] - begin));
    }
    std::vector<std::vector<ExpertLocation>> locations(6);
    int straddling = 0;
    for (int i = 0; i < 6; ++i) {
        const std::int64_t bytes = sizes[i / 3][i % 3];
        for (const std::uint64_t offset : logical[i]) {
            ExpertLocation location{.runs      = {},
                                    .bytes     = std::uint64_t(bytes),
                                    .format    = QType::GGUF_Q8_0,
                                    .row_bytes = 34,
                                    .rows      = 1};
            std::size_t run = 0;
            for (std::size_t f = 0; f < cuts.size(); ++f) {
                const std::uint64_t begin = f == 0 ? 0 : cuts[f - 1];
                const std::uint64_t lo    = std::max(begin, offset);
                const std::uint64_t hi    = std::min(cuts[f], offset + bytes);
                if (lo < hi) { location.runs.at(run++) = {f, lo - begin, hi - lo}; }
            }
            straddling += run == 2 ? 1 : 0;
            locations[i].push_back(location);
        }
    }
    require(straddling == 2, "two experts straddle a file boundary");
    std::vector<DeviceBuffer> tables;
    for (int i = 0; i < 6; ++i) { tables.emplace_back(kExperts * sizeof(void*)); }
    std::vector<ExpertStreamLayer> layers;
    for (int layer = 0; layer < 2; ++layer) {
        layers.push_back(ExpertStreamLayer{
            .rank    = 0,
            .stream  = device.stream,
            .experts = {locations[layer * 3], locations[layer * 3 + 1], locations[layer * 3 + 2]},
            .tables  = {tables[layer * 3].p, tables[layer * 3 + 1].p, tables[layer * 3 + 2].p}});
    }
    // Slots are sized for the widest projections and the zeros after the down: 3072 + 3072 + 1792
    // bytes, ten of them.
    const std::vector<std::uint64_t> budget = {10 * (3072 + 3072 + 1792)};
    ExpertStream stream(device, paths, std::move(layers), budget);
    require(stream.stats().slots == 10, "ten slots");

    const auto check = [&](int layer, const std::vector<std::int32_t>& ids, const std::string& label) {
        stream.prepare(layer, ids);
        device.synchronize();
        for (int k = 0; k < 3; ++k) {
            std::vector<const void*> entries(kExperts);
            tables[layer * 3 + k].copy_to_host(entries.data(), entries.size() * sizeof(void*));
            for (const std::int32_t e : ids) {
                const auto& location = locations[layer * 3 + k][e];
                const std::uint64_t offset = logical[layer * 3 + k][e];
                const std::size_t tail     = k == 2 ? 256 : 0;
                std::vector<std::uint8_t> got(location.bytes + tail);
                require(entries[e] != nullptr, label + ": a routed expert has no slot");
                require(cudaMemcpy(got.data(), entries[e], got.size(), cudaMemcpyDeviceToHost) ==
                            cudaSuccess,
                        label + ": a slot is not readable");
                require(std::memcmp(got.data(), file_bytes.data() + offset, location.bytes) == 0,
                        label + ": a slot differs from the expert's bytes");
                require(std::all_of(got.begin() + location.bytes, got.end(),
                                    [](std::uint8_t b) { return b == 0; }),
                        label + ": a down matrix is not followed by zeros");
            }
        }
    };
    check(0, {3, 7, 3, 12}, "first call");
    check(1, {20, 5}, "straddling experts");
    check(0, {5}, "a straddling expert");
    const auto after_first = stream.stats();
    require(after_first.routes == 6 && after_first.hits == 0, "six distinct experts read");
    check(0, {7, 12, 3}, "resident again");
    require(stream.stats().hits == 3 && stream.stats().read_bytes == after_first.read_bytes,
            "resident experts are not read again");
    // Layer 1 and more layer-0 calls churn through the ten slots.
    for (int round = 0; round < 12; ++round) {
        std::vector<std::int32_t> ids;
        for (int i = 0; i < 6; ++i) { ids.push_back(std::int32_t(random() % kExperts)); }
        check(round % 2, ids, "churn " + std::to_string(round));
    }
    bool refused = false;
    try {
        std::vector<std::int32_t> too_many;
        for (int e = 0; e < 11; ++e) { too_many.push_back(e); }
        stream.prepare(1, too_many);
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a call that needs more experts than slots is refused");
    for (const auto& path : paths) { std::filesystem::remove(path); }
    return 0;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        run();
        std::cout << "PASS qwen4_exp expert stream\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
