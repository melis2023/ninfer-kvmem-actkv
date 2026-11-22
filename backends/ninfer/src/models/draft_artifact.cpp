#include "models/draft_artifact.h"

#include "artifact/schema.h"
#include "models/draft_reader.h"

#include <string>

namespace ninfer::models {
namespace {

struct Kind {
    DraftArtifactKind kind;
    std::string_view component;
    std::string_view architecture;
};

// One row per standalone drafter: the component name the converter writes and the architecture its
// config declares. Both are checked, so a renamed or relabelled component cannot pass for another
// backend's drafter.
constexpr Kind kKinds[] = {
    {DraftArtifactKind::DSpine, "dspine", "DSpineDraftModel"},
    {DraftArtifactKind::DFlash, "dflash", "DFlashDraftModel"},
    {DraftArtifactKind::DFlash2, "dflash2", "DFlash2DraftModel"},
};

} // namespace

DraftArtifactKind draft_artifact_kind(const artifact::Directory& directory) noexcept {
    if (directory.components.size() != 1) { return DraftArtifactKind::Unknown; }
    const auto& name   = directory.components.begin()->first;
    const auto& config = directory.components.begin()->second.config;
    if (!config.is_object() || !config.contains("architectures")) {
        return DraftArtifactKind::Unknown;
    }
    const auto& architectures = config.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        !architectures[0].is_string()) {
        return DraftArtifactKind::Unknown;
    }
    const auto architecture = architectures[0].get<std::string>();
    for (const auto& entry : kKinds) {
        if (name == entry.component && architecture == entry.architecture) { return entry.kind; }
    }
    return DraftArtifactKind::Unknown;
}

DraftArtifactKind draft_artifact_kind(const std::filesystem::path& path) {
    return draft_artifact_kind(open_drafter(path)->directory());
}

std::string_view draft_artifact_name(DraftArtifactKind kind) noexcept {
    for (const auto& entry : kKinds) {
        if (entry.kind == kind) { return entry.component; }
    }
    return {};
}

} // namespace ninfer::models
