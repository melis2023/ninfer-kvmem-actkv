#pragma once

#include "artifact/reader.h"

#include <filesystem>

namespace ninfer::models {

// Whether `path` begins with GGUF's magic. A `--draft-model` path may be either container, and this
// is how the loader tells them apart before opening a Reader over it.
[[nodiscard]] bool is_gguf(const std::filesystem::path& path);

// A drafter GGUF presented as the payload a Reader serves (artifact::PreparedPayload).
//
// llama.cpp's drafter releases ship as GGUFs, and a GGUF is not an artifact directory: its shapes
// are the reverse of the artifact's, its metadata is `dflash.*` keys rather than the drafter's
// config, and it stores the tensors the engine reads whole (norms, convolution bases, codebooks)
// as F32 or in a ggml block type. This synthesises the directory those consumers bind: every
// tensor it can carry as its own blocks becomes a `gguf_blocks_v1` object over the file's bytes,
// every tensor a consumer requires in BF16 is derived once into the payload's Host block, and the
// parameters, Uses and config are the ones models/qwen3_5 and models/dspine declare.
//
// Only a drafter GGUF is accepted: a target checkpoint has no single component to present.
[[nodiscard]] artifact::PreparedPayload prepare_draft_gguf(const std::filesystem::path& path);

} // namespace ninfer::models
