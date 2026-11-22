#include "models/draft_reader.h"

#include "models/draft_gguf.h"

namespace ninfer::models {

std::shared_ptr<artifact::Reader> open_drafter(const std::filesystem::path& path) {
    if (!is_gguf(path)) { return std::make_shared<artifact::Reader>(path); }
    return std::make_shared<artifact::Reader>(path, prepare_draft_gguf(path));
}

} // namespace ninfer::models
