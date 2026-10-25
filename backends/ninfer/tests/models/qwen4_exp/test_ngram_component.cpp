// ngram_table_source finds a Qwen3.8-Flash-Next model's n-gram table: the rows its own artifact
// stores, or those of a separate table artifact (an `ngram` component alone) that names the same
// digest and format. It refuses a model stored without its rows when no table artifact is given, a
// table artifact holding another table or no rows, and a descriptor whose constants or format
// differ from the model's; is_ngram_table_artifact tells a table artifact from a model.
#include "artifact/framing.h"
#include "artifact/reader.h"
#include "models/qwen4_exp/ngram_component.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
    } catch (const ArtifactError& error) {
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
                                const Table* table, std::uint64_t rows, bool model) const {
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
                               {"layout", "row_major_v1"},
                               {"offset", payload},
                               {"bytes", bytes}});
            bindings["ngram/table"] = {{"object", "table"}};
            payload += bytes;
        }
        const Json root = {{"components", components},
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
    const Table bf16{"bf16", kWidth * 2};
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

} // namespace

int main() {
    try {
        const int result = run();
        std::cout << "PASS qwen4_exp n-gram table source\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
