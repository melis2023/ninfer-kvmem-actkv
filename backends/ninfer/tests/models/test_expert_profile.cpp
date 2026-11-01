#include "models/expert_profile.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace ninfer::models;
    const auto directory = std::filesystem::temp_directory_path() /
        ("ninfer-expert-profile-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
    try {
        const auto path = directory / "profile.json";
        ninfer::artifact::ArtifactId artifact{};
        const std::array<std::size_t, 2> widths{3, 2};
        const ExpertRouteCounts expected{{0, 17, std::numeric_limits<std::uint64_t>::max()}, {3, 4}};
        write_expert_profile(path, artifact, expected);
        if (read_expert_profile(path, artifact, widths) != expected) {
            throw std::runtime_error("expert profile did not preserve exact counts");
        }
        const auto rejects = [&](auto id, auto geometry) {
            try { (void)read_expert_profile(path, id, geometry); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("expert profile accepted incompatible input");
        };
        auto wrong = artifact; wrong[0] = std::byte{1};
        rejects(wrong, widths);
        rejects(artifact, std::array<std::size_t, 1>{3});
        rejects(artifact, std::array<std::size_t, 2>{2, 3});
        for (const auto* counts : {"[[1,-2,3],[4,5]]", "[[1,2.5,3],[4,5]]", "[[1,2,3],[4]]"}) {
            std::ofstream(path) << "{\"version\":1,\"artifact_id\":\"00000000000000000000000000000000\",\"counts\":" << counts << '}';
            rejects(artifact, widths);
        }
        std::ofstream(path) << "{\"version\":1,";
        rejects(artifact, widths);
        std::cout << "EXPERT_PROFILE_CONTRACT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
