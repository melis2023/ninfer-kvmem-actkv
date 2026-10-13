#pragma once

#include <cstdint>
#include <vector>

namespace kvmem {
class RawKvStore;

// Borrowed post-RMSNorm/pre-RoPE sums. Layers match the statistics store's
// ordinals; each row contains query_heads * head_dim FP32 elements. A layer
// with no query tokens contributes no score. The caller owns prefix identity
// and commits only evaluated/accepted rows to the supplied statistics.
struct MeanKeyQueryView {
    const float* sums = nullptr;
    const uint32_t* token_counts = nullptr;
    uint32_t layers = 0;
    uint32_t head_dim = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
};

// Mean Q over query rows and each GQA group, cosine with each block's mean K,
// averaged across KV heads and participating layers. FP32 dot/norm operations
// and epsilon 1e-6 preserve the existing llama product adapter's host policy.
// This is selection math only: it mutates no payload, frontier or residency.
//
// GQA aggregation is the folded mean, and only that. ActKV Alg. 3 step 8 takes
// the MAX across a group's query heads; that variant was implemented and
// measured on the 40-needle / 256k test, where it REGRESSED recall from 39/40
// to 28/40 (same-turn) and decayed progressively from round 6 instead of
// failing randomly. The per-head max is dominated by whichever head happens to
// peak, which makes the score noisier; fed into the LRFU keep score
// (KS_t = lambda*KS_{t-1} + hit*score) that noise compounds across rounds and
// evicts action-critical blocks. Mean aggregation is both faster (1x rather
// than g x dot products) and measurably more accurate here, so the option is
// gone rather than left as a footgun. See
// docs/actkv-section56-wiring-record.md section 5.
std::vector<double> mean_key_retrieval_scores(const RawKvStore& keys, uint32_t blocks,
                                               MeanKeyQueryView query);
} // namespace kvmem
