// Gemma 4's vision path against a real artifact (skipped without one):
//   NINFER_GEMMA4_VISION_ARTIFACT  a .ninfer converted with --components text,vision (or with mtp)
//   NINFER_GEMMA4_VISION_IMAGE     an image file
//   NINFER_GEMMA4_VISION_DIR       a directory: the test writes the frontend's patches there
//                                  (patches.bin, BF16 words, and grid.json) and each tower's soft
//                                  tokens (device.bin, overlay.bin, cpu.bin: FP32 [tokens][text
//                                  hidden]); when reference.bin is there too
//                                  (tools/reference/gemma4_vision.py --out, FP64 soft tokens, saved
//                                  raw), the towers are checked against it
//   NINFER_GEMMA4_VISION_CPU       optional, 1: also the CPU tower (--vision-residency cpu)
// The frontend prepares a user turn holding the image: its soft tokens sit between <|image> and
// <image|>, (W / 3) x (H / 3) of them for a W x H patch grid within the default 280-token budget,
// and the item's patches are the grid's. The resident tower then encodes them, and the tower whose
// blocks stream from pinned memory gives the same soft tokens bit for bit. Against the FP64 oracle
// the device towers, which round their matrices' inputs, Q, K and the attention probabilities to
// BF16 (their results stay FP32), keep every soft token's cosine at 0.98 or more and the relative
// RMS error within 3% (transformers' own tower in BF16: 0.889 and 6.9% on the test image); the
// FP32 CPU tower, which rounds only its output to BF16, 0.9999 and 0.5%.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "models/gemma4/model.h"
#include "models/gemma4/vision.h"
#include "models/gemma4/vision_cpu.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace gemma4 = ninfer::models::gemma4;
namespace qwen   = ninfer::models::qwen3_5;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Against the FP64 oracle: the smallest per-token cosine and the relative RMS error.
void compare(const std::vector<float>& got, const std::vector<double>& want, int width,
             const std::string& label, double min_cosine, double max_relative) {
    if (got.size() != want.size()) {
        expect(false, label + ": soft-token count differs from the reference");
        return;
    }
    double err = 0.0, norm = 0.0, worst = 1.0;
    for (std::size_t t = 0; t < got.size() / width; ++t) {
        double dot = 0.0, a = 0.0, b = 0.0;
        for (int c = 0; c < width; ++c) {
            const double x = got[t * width + c], y = want[t * width + c];
            dot += x * y;
            a += x * x;
            b += y * y;
            err += (x - y) * (x - y);
            norm += y * y;
        }
        worst = std::min(worst, dot / std::sqrt(a * b));
    }
    const double relative = std::sqrt(err / norm);
    std::cout << label << ": worst token cosine " << worst << ", relative RMS error " << relative
              << '\n';
    expect(worst >= min_cosine, label + ": a soft token's cosine is below " +
                                    std::to_string(min_cosine));
    expect(relative <= max_relative, label + ": relative RMS error above " +
                                         std::to_string(max_relative));
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_GEMMA4_VISION_ARTIFACT");
    const char* image_path    = std::getenv("NINFER_GEMMA4_VISION_IMAGE");
    const char* dir_value     = std::getenv("NINFER_GEMMA4_VISION_DIR");
    if (artifact_path == nullptr || image_path == nullptr || dir_value == nullptr) {
        std::cout << "SKIP: set NINFER_GEMMA4_VISION_ARTIFACT, _IMAGE and _DIR\n";
        return 77;
    }
    const std::filesystem::path dir(dir_value);
    DeviceContext device(0);
    const artifact::Reader reader(artifact_path);
    auto model = gemma4::load_model(
        reader, {.artifact = artifact_path, .vision = gemma4::VisionPlacement::Device}, device);
    const gemma4::VisionConfig vc = *model->vision_config();

    auto frontend = qwen::make_frontend(model->resources(),
                                        {.architecture   = models::Architecture::Gemma4,
                                         .vision_enabled = true,
                                         .max_context    = 8192});
    PromptInput input;
    ChatMessage user;
    user.role = ChatRole::User;
    MessagePart image;
    image.kind        = MessagePartKind::Media;
    image.media.kind  = MediaKind::Image;
    image.media.bytes = read_bytes(image_path);
    user.parts.push_back(std::move(image));
    MessagePart text;
    text.text = "Describe this image.";
    user.parts.push_back(std::move(text));
    input.messages.push_back(std::move(user));
    const auto prompt = frontend.prepare(std::move(input));
    const auto& data  = qwen::PreparedPromptAccess::view(prompt);

    expect(data.vision_items.size() == 1, "one vision item");
    if (data.vision_items.size() != 1) { return 1; }
    const auto& item = data.vision_items.front();
    const int gw = item.grid.width, gh = item.grid.height;
    const int k  = int(vc.pooling_kernel_size);
    const int tokens = (gw / k) * (gh / k);
    std::cout << "image: patch grid " << gw << " x " << gh << ", " << tokens << " soft tokens\n";
    expect(item.grid.temporal == 1 && gw % k == 0 && gh % k == 0, "a whole pooled grid");
    expect(tokens <= int(vc.default_output_length), "within the default budget");
    expect(item.token_spans.size() == 1 && int(item.token_spans[0].count) == tokens,
           "one run of soft tokens");
    const auto ids  = std::span<const TokenId>(data.token_ids);
    const auto span = item.token_spans.front();
    expect(span.begin > 0 && ids[span.begin - 1] == vc.boi_token_id, "<|image> opens the run");
    expect(span.begin + span.count < ids.size() && ids[span.begin + span.count] == vc.eoi_token_id,
           "<image|> closes the run");
    bool placeholders = true;
    for (std::size_t i = 0; i < span.count; ++i) {
        placeholders &= ids[span.begin + i] == vc.image_token_id &&
                        data.token_types[span.begin + i] == 1;
    }
    expect(placeholders, "the run is image placeholders");

    const auto patches = data.media_payloads.front()->span();
    expect(patches.size() == std::size_t(gw) * gh * vc.patch_features(), "the item's patches");
    std::filesystem::create_directories(dir);
    {
        std::ofstream out(dir / "patches.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(patches.data()), std::streamsize(patches.size() * 2));
        std::ofstream grid(dir / "grid.json");
        grid << nlohmann::json{{"w", gw}, {"h", gh}, {"tokens", tokens}}.dump() << '\n';
    }

    std::vector<std::uint16_t> host_patches(patches.begin(), patches.end());
    model.reset();

    // One tower over a fresh load of the artifact with the given placement (each load holds the
    // text weights too, so one at a time): the soft tokens as FP32 values, saved as <name>.bin.
    const std::size_t width = vc.projection_dim;
    const auto device_encode = [&](gemma4::VisionPlacement placement, const char* name) {
        auto loaded = gemma4::load_model(reader, {.artifact = artifact_path, .vision = placement},
                                         device);
        gemma4::VisionTower tower(*loaded, std::uint32_t(gw * gh));
        expect(tower.streamed() == (placement == gemma4::VisionPlacement::Pinned),
               std::string(name) + ": streamed only from pinned blocks");
        WorkspaceArena workspace(tower.workspace_bytes());
        DeviceBuffer d_patches(host_patches.size() * 2), d_out(std::size_t(tokens) * width * 2);
        d_patches.copy_from_host(host_patches.data(), host_patches.size() * 2);
        tower.encode(static_cast<const __nv_bfloat16*>(d_patches.p), 1, gw, gh,
                     static_cast<__nv_bfloat16*>(d_out.p), workspace, device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        std::vector<__nv_bfloat16> raw(std::size_t(tokens) * width);
        d_out.copy_to_host(raw.data(), raw.size() * 2);
        std::vector<float> soft(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) { soft[i] = __bfloat162float(raw[i]); }
        std::ofstream(dir / (std::string(name) + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char*>(soft.data()), std::streamsize(soft.size() * 4));
        return soft;
    };
    const auto resident = device_encode(gemma4::VisionPlacement::Device, "device");
    // The streamed tower runs the same kernels over the same bytes.
    const auto overlay = device_encode(gemma4::VisionPlacement::Pinned, "overlay");
    expect(overlay == resident, "the streamed tower's soft tokens equal the resident tower's");

    std::vector<float> cpu_soft;
    const char* cpu = std::getenv("NINFER_GEMMA4_VISION_CPU");
    if (cpu != nullptr && std::string(cpu) == "1") {
        auto host_model = gemma4::load_model(
            reader, {.artifact = artifact_path, .vision = gemma4::VisionPlacement::Host}, device);
        const gemma4::CpuVisionTower cpu_tower(*host_model);
        const auto start = std::chrono::steady_clock::now();
        const auto words = cpu_tower.encode(host_patches, 1, gw, gh);
        std::cout << "CPU tower: " << std::chrono::duration<double>(
                                          std::chrono::steady_clock::now() - start).count()
                  << " s for " << gw * gh << " patches\n";
        cpu_soft.resize(words.size());
        for (std::size_t i = 0; i < words.size(); ++i) {
            cpu_soft[i] = std::bit_cast<float>(std::uint32_t(words[i]) << 16U);
        }
        std::ofstream(dir / "cpu.bin", std::ios::binary)
            .write(reinterpret_cast<const char*>(cpu_soft.data()),
                   std::streamsize(cpu_soft.size() * 4));
    }

    const auto reference_path = dir / "reference.bin";
    if (!std::filesystem::exists(reference_path)) {
        std::cout << "patches written to " << dir << "; run tools/reference/gemma4_vision.py and "
                     "save reference.bin to compare\n";
    } else {
        const auto bytes = read_bytes(reference_path);
        std::vector<double> reference(bytes.size() / sizeof(double));
        std::memcpy(reference.data(), bytes.data(), reference.size() * sizeof(double));
        compare(resident, reference, int(width), "device tower", 0.98, 0.03);
        if (!cpu_soft.empty()) {
            compare(cpu_soft, reference, int(width), "CPU tower", 0.9999, 0.005);
        }
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " gemma4 vision real\n";
    return failures == 0 ? 0 : 1;
}
