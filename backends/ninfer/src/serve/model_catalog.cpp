#include "serve/model_catalog.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ninfer::serve {
namespace {

std::string trim(std::string text) {
    const auto first = std::find_if_not(text.begin(), text.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto last  = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char c) {
                          return std::isspace(c) != 0;
                       }).base();
    return first < last ? std::string(first, last) : std::string();
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string unquote(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

bool is_artifact(const std::filesystem::path& path) {
    return path.extension() == ".ninfer" && std::filesystem::is_regular_file(path);
}

std::vector<std::string> split_aliases(const std::string& value) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    for (std::string item; std::getline(stream, item, ',');) {
        item = trim(item);
        if (!item.empty()) { out.push_back(std::move(item)); }
    }
    return out;
}

// Keys a preset uses that are not serve options.
bool catalog_key(const std::string& key) {
    return key == "model" || key == "artifact" || key == "alias" || key == "load-on-startup" ||
           key == "stop-timeout" || key == "version";
}

void append_option(std::vector<std::string>& arguments, const std::string& section,
                   const std::string& key, const std::string& value) {
    if (key.empty() || key.front() == '-') {
        throw std::invalid_argument("preset [" + section + "]: option '" + key +
                                    "' must be written without leading dashes");
    }
    const std::string flag = "--" + key;
    const std::string word = lower(value);
    if (word == "true" || word == "on" || word == "yes") {
        arguments.push_back(flag);
    } else if (word == "false" || word == "off" || word == "no") {
        // A false flag leaves the option out; an explicit negation is its own key (no-...).
    } else {
        arguments.push_back(flag);
        arguments.push_back(value);
    }
}

bool parse_bool(const std::string& section, const std::string& key, const std::string& value) {
    const std::string word = lower(value);
    if (word == "true" || word == "on" || word == "yes" || word == "1") { return true; }
    if (word == "false" || word == "off" || word == "no" || word == "0") { return false; }
    throw std::invalid_argument("preset [" + section + "]: " + key + " must be a boolean");
}

} // namespace

PresetFile parse_preset_file(const std::string& text) {
    PresetFile out;
    std::stringstream stream(text);
    std::size_t line_number      = 0;
    PresetFile::Section* current = nullptr;
    PresetFile::Section top{.name = "*", .entries = {}};
    for (std::string line; std::getline(stream, line);) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line.front() == ';' || line.front() == '#') { continue; }
        if (line.front() == '[') {
            if (line.back() != ']' || line.size() < 3) {
                throw std::invalid_argument("preset line " + std::to_string(line_number) +
                                            ": malformed section header");
            }
            const std::string name = trim(line.substr(1, line.size() - 2));
            auto existing = std::find_if(out.sections.begin(), out.sections.end(),
                                         [&](const auto& section) { return section.name == name; });
            if (existing != out.sections.end()) {
                throw std::invalid_argument("preset line " + std::to_string(line_number) +
                                            ": duplicate section [" + name + "]");
            }
            out.sections.push_back({.name = name, .entries = {}});
            current = &out.sections.back();
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::invalid_argument("preset line " + std::to_string(line_number) +
                                        ": expected key = value");
        }
        const std::string key   = lower(trim(line.substr(0, equals)));
        const std::string value = unquote(trim(line.substr(equals + 1)));
        (current != nullptr ? *current : top).entries.emplace_back(key, value);
    }
    // Entries before any section (llama.cpp's `version = 1`) belong to the global section.
    if (!top.entries.empty()) {
        auto global = std::find_if(out.sections.begin(), out.sections.end(),
                                   [](const auto& section) { return section.name == "*"; });
        if (global == out.sections.end()) {
            out.sections.insert(out.sections.begin(), std::move(top));
        } else {
            global->entries.insert(global->entries.begin(), top.entries.begin(), top.entries.end());
        }
    }
    return out;
}

std::vector<CatalogModel> discover_models(const std::filesystem::path& directory) {
    if (!std::filesystem::is_directory(directory)) {
        throw std::invalid_argument("--models-dir is not a directory: " + directory.string());
    }
    std::vector<CatalogModel> out;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (is_artifact(entry.path())) {
            out.push_back({.id = entry.path().stem().string(), .artifact = entry.path()});
            continue;
        }
        if (!entry.is_directory()) { continue; }
        std::vector<std::filesystem::path> artifacts;
        for (const auto& inner : std::filesystem::directory_iterator(entry.path())) {
            if (is_artifact(inner.path())) { artifacts.push_back(inner.path()); }
        }
        if (artifacts.size() == 1) {
            out.push_back({.id = entry.path().filename().string(), .artifact = artifacts.front()});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const CatalogModel& a, const CatalogModel& b) { return a.id < b.id; });
    return out;
}

std::vector<CatalogModel> build_catalog(const std::optional<std::filesystem::path>& models_dir,
                                        const std::optional<PresetFile>& preset,
                                        const std::vector<std::string>& command_line) {
    std::vector<CatalogModel> models;
    if (models_dir) { models = discover_models(*models_dir); }
    std::vector<std::string> global_arguments;
    if (preset) {
        for (const auto& section : preset->sections) {
            if (section.name != "*") { continue; }
            for (const auto& [key, value] : section.entries) {
                if (catalog_key(key)) { continue; }
                append_option(global_arguments, section.name, key, value);
            }
        }
        for (const auto& section : preset->sections) {
            if (section.name == "*") { continue; }
            auto model = std::find_if(models.begin(), models.end(),
                                      [&](const CatalogModel& m) { return m.id == section.name; });
            if (model == models.end()) {
                models.push_back({.id = section.name});
                model = std::prev(models.end());
            }
            for (const auto& [key, value] : section.entries) {
                if (key == "model" || key == "artifact") {
                    model->artifact = value;
                } else if (key == "alias") {
                    const auto aliases = split_aliases(value);
                    model->aliases.insert(model->aliases.end(), aliases.begin(), aliases.end());
                } else if (key == "load-on-startup") {
                    model->load_on_startup = parse_bool(section.name, key, value);
                } else if (!catalog_key(key)) {
                    append_option(model->arguments, section.name, key, value);
                }
            }
            if (model->artifact.empty()) {
                throw std::invalid_argument("preset [" + section.name +
                                            "] names no artifact (model = PATH) and no such model "
                                            "is in --models-dir");
            }
        }
    }
    std::set<std::string> seen;
    for (CatalogModel& model : models) {
        for (const std::string& name : [&] {
                 std::vector<std::string> names{model.id};
                 names.insert(names.end(), model.aliases.begin(), model.aliases.end());
                 return names;
             }()) {
            if (!seen.insert(name).second) {
                throw std::invalid_argument("model id or alias '" + name + "' is used twice");
            }
        }
        // Precedence, lowest first: the preset's global section, the model's own section, then the
        // command line. The option parser keeps the last value an option is given.
        std::vector<std::string> arguments = global_arguments;
        arguments.insert(arguments.end(), model.arguments.begin(), model.arguments.end());
        arguments.insert(arguments.end(), command_line.begin(), command_line.end());
        model.arguments = std::move(arguments);
    }
    if (models.empty()) {
        throw std::invalid_argument(
            "router mode found no models in --models-dir or --models-preset");
    }
    return models;
}

} // namespace ninfer::serve
