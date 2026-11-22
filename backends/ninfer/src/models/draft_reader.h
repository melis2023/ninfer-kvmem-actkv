#pragma once

#include "artifact/reader.h"

#include <filesystem>
#include <memory>

namespace ninfer::models {

// The reader a drafter path needs. A `.ninfer` artifact opens as itself; a drafter GGUF opens over
// the payload models/draft_gguf.cpp prepares for it, so `--draft-model` takes either container. Any
// other container is refused here, where the path can be named.
[[nodiscard]] std::shared_ptr<artifact::Reader> open_drafter(const std::filesystem::path& path);

} // namespace ninfer::models
