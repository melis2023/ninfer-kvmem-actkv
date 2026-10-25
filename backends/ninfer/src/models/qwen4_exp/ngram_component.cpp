#include "models/qwen4_exp/ngram_component.h"

#include "models/qwen4_exp/ngram_hash.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Json;

constexpr const char* kTableBinding = "ngram/table";

std::vector<std::uint64_t> u64_array(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_array()) { throw ArtifactError(std::string(name) + " must be an array"); }
    std::vector<std::uint64_t> out;
    for (const auto& item : value) { out.push_back(artifact::require_u64(item, name)); }
    return out;
}

void require_equal(const std::filesystem::path& artifact, std::uint64_t actual,
                   std::uint64_t expected, const char* name) {
    if (actual != expected) {
        throw ArtifactError(artifact.string() + ": the n-gram table's " + name + " is " +
                            std::to_string(actual) + "; the model needs " +
                            std::to_string(expected));
    }
}

ops::NgramRowFormat row_format(const std::filesystem::path& artifact, const std::string& format) {
    if (format == "gguf_iq4_nl") { return ops::NgramRowFormat::Iq4Nl; }
    if (format == "bf16") { return ops::NgramRowFormat::Bf16; }
    throw ArtifactError(artifact.string() + ": the n-gram table is stored as " + format +
                        "; the runtime decodes gguf_iq4_nl and bf16 rows");
}

// What a `ngram` component says about its table, checked against the model's text config.
struct Descriptor {
    std::string format;
    std::string digest; // SHA-256 of the table's bytes, lowercase hex
    std::uint64_t rows  = 0;
    std::uint64_t width = 0;
};

Descriptor descriptor(const artifact::Reader& reader, const std::filesystem::path& artifact,
                      const TextConfig& config) {
    const auto& directory = reader.directory();
    if (!directory.components.contains("ngram")) {
        throw ArtifactError(artifact.string() +
                            " describes no n-gram table (its `ngram` component); convert it "
                            "with its table source, --source ngram=PATH");
    }
    const Json& c = directory.component("ngram").config;
    artifact::require_members(c,
                              {"architectures", "model_type", "vocab_size", "eos_token_id",
                               "ngram_size", "heads_per_ngram", "row_width", "rows", "multipliers",
                               "head_vocab", "head_offset", "format", "table_sha256"},
                              {}, "n-gram table config");
    if (!c.at("architectures").is_array() || c.at("architectures").size() != 1 ||
        c.at("architectures")[0] != "Qwen4ExpNgramTable") {
        throw ArtifactError(artifact.string() +
                            ": the `ngram` component is not a Qwen3.8-Flash-Next n-gram table");
    }
    const NgramHashConstants expected = derive_ngram_hash_constants(config.ngram);
    require_equal(artifact, artifact::require_u64(c.at("vocab_size"), "vocab_size"),
                  config.vocab_size, "vocab_size");
    require_equal(artifact, artifact::require_u64(c.at("eos_token_id"), "eos_token_id"),
                  static_cast<std::uint64_t>(config.eos_token_id), "eos_token_id");
    require_equal(artifact, artifact::require_u64(c.at("ngram_size"), "ngram_size"),
                  config.ngram.ngram_size, "ngram_size");
    require_equal(artifact, artifact::require_u64(c.at("heads_per_ngram"), "heads_per_ngram"),
                  config.ngram.heads_per_ngram, "heads_per_ngram");
    Descriptor out;
    out.width = config.ple_embed_dim / config.ngram_heads();
    out.rows  = expected.rows;
    require_equal(artifact, artifact::require_u64(c.at("row_width"), "row_width"), out.width,
                  "row_width");
    require_equal(artifact, artifact::require_u64(c.at("rows"), "rows"), out.rows, "rows");
    if (u64_array(c, "multipliers") != expected.multipliers ||
        u64_array(c, "head_vocab") != expected.head_vocab ||
        u64_array(c, "head_offset") != expected.head_offset) {
        throw ArtifactError(artifact.string() +
                            ": the n-gram table's hash constants differ from the model's");
    }
    out.format = artifact::require_id(c.at("format"), "n-gram table format");
    if (!c.at("table_sha256").is_string()) {
        throw ArtifactError(artifact.string() + ": table_sha256 must be a string");
    }
    out.digest = c.at("table_sha256").get<std::string>();
    if (out.digest.size() != 64 ||
        out.digest.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw ArtifactError(artifact.string() +
                            ": table_sha256 must be 64 lowercase hexadecimal digits");
    }
    return out;
}

// The rows `reader` stores for the table `d` describes.
NgramTableSource stored_rows(const artifact::Reader& reader, const std::filesystem::path& artifact,
                             const Descriptor& d) {
    const auto& directory = reader.directory();
    const auto found      = directory.bindings.find(kTableBinding);
    if (found == directory.bindings.end() || found->second.parts.size() != 1) {
        throw ArtifactError(artifact.string() + ": the n-gram table must be one stored object");
    }
    const auto& part   = found->second.parts.front();
    const auto& tensor = directory.tensor(part.object);
    if (tensor.format != d.format) {
        throw ArtifactError(artifact.string() + ": the n-gram table is stored as " +
                            tensor.format + " but described as " + d.format);
    }
    NgramTableSource out;
    out.format                      = row_format(artifact, tensor.format);
    out.layout.row_bytes            = ops::ngram_row_bytes(out.format);
    out.layout.rows                 = d.rows;
    const std::uint64_t table_bytes = out.layout.rows * out.layout.row_bytes;
    if (tensor.shape != artifact::Shape{d.rows, d.width} || part.begin != 0 ||
        tensor.bytes != table_bytes) {
        throw ArtifactError(artifact.string() + ": the n-gram table has an unexpected geometry");
    }
    for (const auto& segment : reader.segments(tensor.offset, table_bytes)) {
        const auto& record = directory.files.at(segment.file_index);
        const std::filesystem::path file =
            segment.file_index == 0 ? artifact : artifact.parent_path() / record.path.value();
        out.layout.segments.push_back({file, segment.file_offset, segment.bytes});
    }
    return out;
}

} // namespace

bool is_ngram_table_artifact(const artifact::Reader& reader) {
    const auto& directory = reader.directory();
    return !directory.components.contains("text") && directory.components.contains("ngram") &&
           directory.bindings.contains(kTableBinding);
}

NgramTableSource ngram_table_source(const artifact::Reader& reader,
                                    const std::filesystem::path& artifact,
                                    const TextConfig& config,
                                    const std::filesystem::path& table) {
    const Descriptor model = descriptor(reader, artifact, config);
    if (table.empty()) {
        if (!reader.directory().bindings.contains(kTableBinding)) {
            throw ArtifactError(
                artifact.string() + ": the model's n-gram table (sha256 " +
                model.digest.substr(0, 16) +
                "...) is stored separately; pass its table artifact with --ngram-table PATH. "
                "--no-ngram-table runs the model without it, a non-standard experimental mode "
                "that badly degrades the output");
        }
        return stored_rows(reader, artifact, model);
    }
    const artifact::Reader other(table);
    if (!other.directory().bindings.contains(kTableBinding)) {
        throw ArtifactError(table.string() + " stores no n-gram table");
    }
    const Descriptor theirs = descriptor(other, table, config);
    if (theirs.digest != model.digest || theirs.format != model.format) {
        throw ArtifactError(table.string() + " holds a different n-gram table (" + theirs.format +
                            ", sha256 " + theirs.digest.substr(0, 16) + "...) than the model's (" +
                            model.format + ", sha256 " + model.digest.substr(0, 16) + "...)");
    }
    return stored_rows(other, table, theirs);
}

} // namespace ninfer::models::qwen4_exp
