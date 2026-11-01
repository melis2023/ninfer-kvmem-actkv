// ngram_table_source finds a Qwen3.8-Flash-Next model's n-gram table: the rows its own artifact
// stores, or those of a separate table artifact (an `ngram` component alone) that names the same
// digest and format. It refuses a model stored without its rows when no table artifact is given, a
// table artifact holding another table or no rows, and a descriptor whose constants or format
// differ from the model's; is_ngram_table_artifact tells a table artifact from a model.
#include "artifact/framing.h"
#include "artifact/reader.h"
#include "models/qwen4_exp/ngram_component.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <span>

using namespace ninfer::models::qwen4_exp;
using ninfer::artifact::ArtifactError;
using ninfer::artifact::Json;

namespace {

constexpr std::uint64_t kWidth = 160;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void refuses(const std::function<void()>& fn, const std::string& needle,
             const std::string& label) {
    try {
        fn();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(needle) != std::string::npos,
                label + ": unexpected message: " + error.what());
        return;
    }
    throw std::runtime_error("not refused: " + label);
}

TextConfig text_config() {
    TextConfig config;
    config.vocab_size    = 1000;
    config.eos_token_id  = 7;
    config.ngram         = {.vocab_size      = 1000,
                            .ngram_size      = 3,
                            .heads_per_ngram = 2,
                            .ple_layer_index = 0,
                            .vocab_base      = 100,
                            .divisible_by    = 8,
                            .seed            = 1234};
    config.ple_embed_dim = static_cast<std::uint32_t>(config.ngram_heads() * kWidth);
    return config;
}

Json descriptor(const TextConfig& config, const std::string& format, const std::string& digest) {
    const NgramHashConstants constants = derive_ngram_hash_constants(config.ngram);
    return {{"architectures", {"Qwen4ExpNgramTable"}},
            {"model_type", "qwen4_exp_ngram"},
            {"vocab_size", config.vocab_size},
            {"eos_token_id", config.eos_token_id},
            {"ngram_size", config.ngram.ngram_size},
            {"heads_per_ngram", config.ngram.heads_per_ngram},
            {"row_width", kWidth},
            {"rows", constants.rows},
            {"multipliers", constants.multipliers},
            {"head_vocab", constants.head_vocab},
            {"head_offset", constants.head_offset},
            {"format", format},
            {"table_sha256", digest}};
}

struct Table {
    std::string format = "gguf_iq4_nl";
    std::uint64_t row_bytes = 90;
    std::string layout = "gguf_blocks_v1";
};

class Directory {
  public:
    Directory() {
        std::random_device entropy;
        root_ = std::filesystem::temp_directory_path() /
                ("ninfer-ngram-component-" + std::to_string(entropy()));
        std::filesystem::create_directories(root_);
    }
    ~Directory() {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }
    Directory(const Directory&)            = delete;
    Directory& operator=(const Directory&) = delete;

    // A one-file artifact with the given components; `table` stores rows bound as ngram/table
    // after a small model tensor (absent from a table artifact).
    std::filesystem::path write(const std::string& name, const Json& components,
                                const Table* table, std::uint64_t rows, bool model,
                                std::span<const std::byte> profile = {}) const {
        auto stored_components = components;
        Json objects  = Json::array();
        Json bindings = Json::object();
        std::uint64_t payload = 0;
        if (model) {
            objects.push_back({{"id", "w"},
                               {"kind", "tensor"},
                               {"shape", {4}},
                               {"format", "fp32"},
                               {"layout", "contiguous_le_v1"},
                               {"offset", 0},
                               {"bytes", 16}});
            bindings["text/weight"] = {{"object", "w"}};
            payload                 = 256;
        }
        if (table != nullptr) {
            const std::uint64_t bytes = rows * table->row_bytes;
            objects.push_back({{"id", "table"},
                               {"kind", "tensor"},
                               {"shape", {rows, kWidth}},
                               {"format", table->format},
                               {"layout", table->layout},
                               {"offset", payload},
                               {"bytes", bytes}});
            bindings["ngram/table"] = {{"object", "table"}};
            payload += bytes;
        }
        const auto profile_offset = payload;
        if (!profile.empty()) {
            objects.push_back({{"id", "hot"}, {"kind", "resource"},
                               {"encoding", "raw_bytes_v1"}, {"offset", payload},
                               {"bytes", profile.size()}});
            stored_components["ngram"]["resources"] = {{"hot_profile", "hot"}};
            payload += profile.size();
        }
        const Json root = {{"components", stored_components},
                           {"objects", objects},
                           {"bindings", bindings},
                           {"uses", Json::array()},
                           {"files", Json::array({{{"path", nullptr}, {"payload_bytes", payload}}})}};
        const std::string text = root.dump();
        std::vector<char> header(ninfer::artifact::kHeaderBytes, 0);
        for (std::size_t i = 0; i < 8; ++i) {
            header[i] = static_cast<char>(ninfer::artifact::kEntryMagic[i]);
        }
        for (unsigned i = 0; i < 8; ++i) {
            header[8 + i] = static_cast<char>((text.size() >> (8 * i)) & 255);
        }
        header[16]          = 0x5a;
        const auto path     = root_ / name;
        const auto position = header.size() + text.size();
        const auto aligned  = (position + ninfer::artifact::kPayloadAlignment - 1) /
                             ninfer::artifact::kPayloadAlignment *
                             ninfer::artifact::kPayloadAlignment;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        const std::vector<char> zeros(aligned - position + payload, 0);
        file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
        if (!profile.empty()) {
            file.seekp(static_cast<std::streamoff>(aligned + profile_offset));
            file.write(reinterpret_cast<const char*>(profile.data()),
                       static_cast<std::streamsize>(profile.size()));
        }
        return path;
    }

  private:
    std::filesystem::path root_;
};

int run() {
    const TextConfig config = text_config();
    const std::uint64_t rows = derive_ngram_hash_constants(config.ngram).rows;
    require(rows == 424, "four heads of 101..109 rows round up to 424");
    const std::string digest(64, 'a');
    const std::string other(64, 'b');
    const Table iq4;
    const Table bf16{"bf16", kWidth * 2, "contiguous_le_v1"};
    const Table fp8{"fp8_e4m3fn_row_fp16", kWidth + 2, "row_interleaved_v1"};
    const Directory dir;
    const Json text = {{"config", Json::object()}};

    const auto embedded = dir.write(
        "embedded.ninfer",
        {{"text", text}, {"ngram", {{"config", descriptor(config, iq4.format, digest)}}}}, &iq4,
        rows, true);
    const auto bare = dir.write(
        "bare.ninfer",
        {{"text", text}, {"ngram", {{"config", descriptor(config, iq4.format, digest)}}}},
        nullptr, rows, true);
    const auto table = dir.write("table.ninfer",
                                 {{"ngram", {{"config", descriptor(config, iq4.format, digest)}}}},
                                 &iq4, rows, false);
    const auto foreign = dir.write("foreign.ninfer",
                                   {{"ngram", {{"config", descriptor(config, iq4.format, other)}}}},
                                   &iq4, rows, false);
    const auto widened = dir.write("bf16.ninfer",
                                   {{"ngram", {{"config", descriptor(config, bf16.format, digest)}}}},
                                   &bf16, rows, false);
    const auto mislabeled = dir.write(
        "mislabeled.ninfer", {{"ngram", {{"config", descriptor(config, iq4.format, digest)}}}},
        &bf16, rows, false);

    const ninfer::artifact::Reader embedded_reader(embedded);
    const ninfer::artifact::Reader bare_reader(bare);

    const auto fp8_path = dir.write(
        "fp8.ninfer", {{"ngram", {{"config", descriptor(config, fp8.format, digest)}}}},
        &fp8, rows, false);
    const auto fp8_source = ngram_table_source(ninfer::artifact::Reader(fp8_path), fp8_path, config);
    require(fp8_source.format == ninfer::ops::NgramRowFormat::Fp8E4M3RowScale &&
                fp8_source.layout.row_bytes == 162 && fp8_source.layout.rows == rows &&
                fp8_source.layout.segments[0].bytes == rows * 162,
            "FP8 table did not select interleaved FP16 row decoding");
    const Table wrong_layout{fp8.format, fp8.row_bytes, "row_scale_v1"};
    const auto wrong_path = dir.write(
        "wrong-layout.ninfer", {{"ngram", {{"config", descriptor(config, fp8.format, digest)}}}},
        &wrong_layout, rows, false);
    refuses([&] { (void)ngram_table_source(ninfer::artifact::Reader(wrong_path), wrong_path, config); },
            "RowScale", "FP16 n-gram rows declared as BF16 scale planes");

    // The model's own rows: one segment of its file, where the payload stores them.
    const NgramTableSource own = ngram_table_source(embedded_reader, embedded, config);
    require(own.format == ninfer::ops::NgramRowFormat::Iq4Nl && own.layout.row_bytes == 90 &&
                own.layout.rows == rows,
            "embedded table geometry");
    require(own.layout.segments.size() == 1 && own.layout.segments[0].path == embedded &&
                own.layout.segments[0].bytes == rows * 90 &&
                own.layout.segments[0].file_offset % ninfer::artifact::kPayloadAlignment == 256,
            "embedded table segment");

    // A model stored without its rows takes them from the table artifact naming its digest.
    refuses([&] { (void)ngram_table_source(bare_reader, bare, config); }, "--ngram-table",
            "a model without its rows and no table artifact");
    const NgramTableSource external = ngram_table_source(bare_reader, bare, config, table);
    require(external.layout.segments.size() == 1 && external.layout.segments[0].path == table &&
                external.layout.segments[0].bytes == rows * 90 && external.layout.rows == rows,
            "external table segment");
    // A table artifact also stands in for a model's own rows.
    const NgramTableSource replaced = ngram_table_source(embedded_reader, embedded, config, table);
    require(replaced.layout.segments[0].path == table, "an explicit table artifact wins");

    // A profile belongs to the selected table, remains cold unless requested, and preserves rank.
    std::vector<std::byte> profile;
    for (const char c : std::string("NFNGHOT1")) { profile.push_back(static_cast<std::byte>(c)); }
    const auto append = [&](std::uint64_t value, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) {
            profile.push_back(static_cast<std::byte>((value >> (8U * i)) & 255U));
        }
    };
    append(rows, 8);
    append(ngram_hash_fingerprint(derive_ngram_hash_constants(config.ngram)), 8);
    append(12345, 8);
    append(3, 8);
    for (auto row : {17U, 3U, 99U}) { append(row, 4); }
    const auto hot = dir.write("hot.ninfer",
        {{"ngram", {{"config", descriptor(config, iq4.format, digest)}}}}, &iq4, rows, false, profile);
    require(!ngram_table_source(bare_reader, bare, config, hot).hot_profile,
            "disk mode leaves the embedded profile unread");
    const auto selected_hot = ngram_table_source(bare_reader, bare, config, hot, true);
    require(selected_hot.hot_profile && selected_hot.hot_profile->tokens == 12345 &&
                selected_hot.hot_profile->rows == std::vector<std::uint32_t>{17, 3, 99},
            "an external table supplies its profile in frequency order");
    const auto own_hot = ngram_table_source(ninfer::artifact::Reader(hot), hot, config, {}, true);
    require(own_hot.hot_profile && own_hot.hot_profile->rows == selected_hot.hot_profile->rows,
            "an artifact's own table supplies its profile");
    require(!ngram_table_source(bare_reader, bare, config, table, true).hot_profile,
            "a table without a profile stays distinguishable from an empty profile");
    profile[16] ^= std::byte{1};
    const auto bad_hot = dir.write("bad-hot.ninfer",
        {{"ngram", {{"config", descriptor(config, iq4.format, digest)}}}}, &iq4, rows, false, profile);
    (void)ngram_table_source(bare_reader, bare, config, bad_hot);
    refuses([&] { (void)ngram_table_source(bare_reader, bare, config, bad_hot, true); },
            "another n-gram hash", "an embedded profile for a different hash");

    refuses([&] { (void)ngram_table_source(bare_reader, bare, config, foreign); },
            "different n-gram table", "another digest");
    refuses([&] { (void)ngram_table_source(bare_reader, bare, config, widened); },
            "different n-gram table", "another format");
    refuses([&] { (void)ngram_table_source(bare_reader, bare, config, bare); },
            "stores no n-gram table", "a table artifact without rows");
    refuses([&] { (void)ngram_table_source(bare_reader, bare, config, mislabeled); },
            "described as", "rows stored in another format than described");

    // The descriptor must carry the constants the model derives.
    TextConfig shifted = config;
    shifted.ngram.seed = 99;
    refuses([&] { (void)ngram_table_source(embedded_reader, embedded, shifted); },
            "hash constants", "constants of another seed");
    TextConfig larger         = config;
    larger.vocab_size         = 1001;
    larger.ngram.vocab_size   = 1001;
    refuses([&] { (void)ngram_table_source(embedded_reader, embedded, larger); }, "vocab_size",
            "another vocabulary");
    Json unlabeled = descriptor(config, iq4.format, digest);
    unlabeled.erase("table_sha256");
    const auto old = dir.write("old.ninfer", {{"text", text}, {"ngram", {{"config", unlabeled}}}},
                               &iq4, rows, true);
    refuses([&] { (void)ngram_table_source(ninfer::artifact::Reader(old), old, config); },
            "table_sha256", "a descriptor without its digest");

    require(is_ngram_table_artifact(ninfer::artifact::Reader(table)), "a table artifact");
    require(!is_ngram_table_artifact(embedded_reader) && !is_ngram_table_artifact(bare_reader),
            "models are not table artifacts");
    return 0;
}

int writer_fixture(const std::filesystem::path& path, const std::filesystem::path& oracle) {
    using namespace ninfer;
    using namespace ninfer::test;
    if (cuda_unavailable()) { return 77; }
    const artifact::Reader artifact(path);
    const auto& descriptor = artifact.directory().component("ngram").config;
    TextConfig config = text_config();
    config.vocab_size = config.ngram.vocab_size = descriptor.at("vocab_size").get<std::uint32_t>();
    config.eos_token_id = descriptor.at("eos_token_id").get<std::int32_t>();
    const auto source = ngram_table_source(artifact, path, config, {}, true);
    require(source.hot_profile && source.hot_profile->tokens == 12345 &&
                source.hot_profile->rows == std::vector<std::uint32_t>{17, 3, 99},
            "Python writer's embedded profile differs from the C++ hash or row order");
    require(source.format == ops::NgramRowFormat::Fp8E4M3RowScale && source.layout.rows == 424,
            "Python writer selected the wrong n-gram encoding");
    std::vector<std::uint16_t> decoded(424 * kWidth);
    require(std::filesystem::file_size(oracle) == decoded.size() * 2, "oracle size mismatch");
    std::ifstream input(oracle, std::ios::binary);
    input.exceptions(std::ios::badbit | std::ios::failbit);
    input.read(reinterpret_cast<char*>(decoded.data()), static_cast<std::streamsize>(decoded.size() * 2));
    std::vector<std::uint64_t> ids;
    for (std::uint64_t row = 0; row < 424; ++row) { ids.push_back(row * 73 % 424); }
    ids.insert(ids.end(), {0, 423, 3, 131}); // repeated rows at source boundaries
    std::vector<std::uint8_t> staged(ids.size() * source.layout.row_bytes);
    NgramTableReader reader(source.layout, {});
    reader.read_rows(ids, staged);
    std::vector<std::uint16_t> expected;
    for (const auto row : ids) {
        expected.insert(expected.end(), decoded.begin() + row * kWidth,
                         decoded.begin() + (row + 1) * kWidth);
    }
    GuardedDeviceBuffer d_rows(staged.size()), d_out(expected.size() * 2);
    d_rows.copy_from_host(staged.data(), staged.size());
    Tensor rows(d_rows.data(), DType::U8, {162, static_cast<std::int32_t>(ids.size())});
    Tensor embedding(d_out.data(), DType::BF16, {640, static_cast<std::int32_t>(ids.size() / 4)});
    int failures = 0;
    for (int repeat = 0; repeat < 2; ++repeat) {
        ops::ngram_embed_rows(rows, source.format, 4, embedding, nullptr);
        cuda_synchronize();
        failures += verify_exact("HF n-gram writer/reader/GPU",
                                  from_device<std::uint16_t>(d_out.data(), expected.size()), expected);
    }
    failures += d_rows.verify_guards("HF n-gram rows");
    failures += d_out.verify_guards("HF n-gram decoded rows");
    std::cout << (failures ? "FAIL" : "PASS") << " FP8 writer interop, "
              << source.layout.segments.size() << " file segments\n";
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-cuda") {
            return ninfer::test::cuda_unavailable() ? 77 : 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--writer-fixture") {
            return writer_fixture(argv[2], argv[3]);
        }
        const int result = run();
        std::cout << "PASS qwen4_exp n-gram table source\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
