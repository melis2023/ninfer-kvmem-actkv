#pragma once

// The models a router-mode server can serve, read from a models directory and a preset file in
// llama.cpp's conventions. Each model carries the serve arguments it is started with: its preset's
// global section, then its own section, then the command line, the later winning where they repeat
// an option -- the command line's precedence over presets is llama.cpp's.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct CatalogModel {
    // Public id: the preset section name, or the artifact's stem (or its directory's name) in a
    // models directory.
    std::string id;
    std::vector<std::string> aliases;
    std::filesystem::path artifact;
    // Serve options for this model, without the program name and artifact path.
    std::vector<std::string> arguments;
    bool load_on_startup = false;
};

// `[*]` holds defaults for every model. Any other section is a model: `model` (or `artifact`) is
// its artifact path, `alias` a comma-separated list of further ids, `load-on-startup` a boolean;
// every other key is a serve option without its leading dashes (`kv-capacity = 65536`), a boolean
// key a flag (`model-suspend = true`; `false` leaves it out). A section whose name matches a model
// found in the models directory configures that model and needs no path. Comments start with ';' or
// '#'.
struct PresetFile {
    struct Section {
        std::string name;
        std::vector<std::pair<std::string, std::string>> entries;
    };

    std::vector<Section> sections;
};

[[nodiscard]] PresetFile parse_preset_file(const std::string& text);

// The .ninfer artifacts directly in `directory`, and each subdirectory holding exactly one.
[[nodiscard]] std::vector<CatalogModel> discover_models(const std::filesystem::path& directory);

// Merges the directory's models with the preset's sections and appends `command_line` (the serve
// arguments shared by every model) to each. Throws std::invalid_argument for an unknown or
// duplicate id, a section without an artifact, or a malformed entry.
[[nodiscard]] std::vector<CatalogModel>
build_catalog(const std::optional<std::filesystem::path>& models_dir,
              const std::optional<PresetFile>& preset,
              const std::vector<std::string>& command_line);

} // namespace ninfer::serve
