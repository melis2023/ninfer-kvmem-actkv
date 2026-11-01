#pragma once

// Token ids and the limits and defaults the Engine options, requests and statistics share.

#include <cstddef>
#include <cstdint>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;

inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;

inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;

// ContextCacheHints::allow_engine_prefix_grid proposes at most this many shared candidates, at
// multiples of a stride that starts at one grid page and doubles until the count fits.
inline constexpr std::uint32_t kPrefixGridCandidates = 8;

inline constexpr std::uint32_t kPrefixGridPageTokens = 256;

// Explicit markers plus the engine's automatic tool/leading-instruction/full-prompt candidates:
// the shared-prefix opportunities one request can materialize on its own. Prefix-grid points are
// excluded because they carry only observed evidence and need a second reuse domain.
inline constexpr std::size_t kMaximumPreparedPromptCacheCandidatesPerRequest = 7;

// Smallest nonzero ContextCacheOptions::progress_anchor_tokens. Preparation proposes one candidate
// per multiple, so a finer stride would make a long prompt's candidate list, and the duplicate
// scan over it, grow without bound.
inline constexpr std::uint32_t kMinimumProgressAnchorTokens = 256;

// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;

inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;

inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;

inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;

inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;

// Largest text hidden size among shipped targets (27B). Used only to conservatively size the
// pinned cross-rank staging buffer for a model-parallel `DeviceContext` from the configured
// prefill chunk, before the target's actual hidden size is known; a wider future target just
// widens this.
inline constexpr std::size_t kMaxSupportedResidualHiddenSize = 5120;

// BF16: the residual stream's dtype for every shipped target, independent of KV cache storage.
inline constexpr std::size_t kResidualStreamBytesPerElement = 2;

// Pinned Host tier of the hybrid prefix cache when --host-cache-mib is not given.
inline constexpr std::size_t kDefaultHybridHostCacheBytes = 8ULL << 30;

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

// Flash-Next's default for EngineOptions::kv_resident_positions with host or disk experts.
inline constexpr std::uint32_t kDefaultKvResidentPositions = 32768;

// The most token ids one request may name for the typed readout (ExecutionOptions::readout_tokens).
inline constexpr std::size_t kMaximumReadoutTokens = 32;

// Log probabilities describe the distribution the sampler draws from, before truncation: the
// logits after the structured-output mask and the presence and frequency penalties, divided by the
// request's temperature (1 for greedy decoding) and normalized over the whole token domain. top_k,
// top_p and min_p do not change them.
inline constexpr std::size_t kMaximumTokenLogprobs = 20;

// OpenAI's sentinel log probability for a generated token outside its reported top set.
inline constexpr float kLogprobSentinel = -9999.0f;

} // namespace ninfer
