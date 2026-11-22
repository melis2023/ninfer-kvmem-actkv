#pragma once

#include "artifact/reader.h"

#include <cstdint>
#include <filesystem>
#include <string_view>

namespace ninfer::models {

// The drafter a standalone `--draft-model` artifact carries. Such an artifact holds the drafter
// alone: one component, named for the backend, that targets nothing (the target model, its
// embedding and its head stay in the target artifact).
enum class DraftArtifactKind : std::uint8_t { Unknown, DSpine, DFlash, DFlash2 };

// Reads the kind from the artifact's single component: its name and its architecture. Returns
// Unknown when the directory is not a drafter artifact at all, so a caller deciding a backend can
// report its own context.
[[nodiscard]] DraftArtifactKind draft_artifact_kind(const artifact::Directory& directory) noexcept;

// The same, opening `path`; an unreadable artifact throws `artifact::ArtifactError` naming it.
[[nodiscard]] DraftArtifactKind draft_artifact_kind(const std::filesystem::path& path);

// The component name and weight prefix the kind binds under ("dspine"/"dflash"/"dflash2"), or
// empty for Unknown.
[[nodiscard]] std::string_view draft_artifact_name(DraftArtifactKind kind) noexcept;

} // namespace ninfer::models
