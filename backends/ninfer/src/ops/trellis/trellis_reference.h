#pragma once

// Host-side reference for the NariQ coded-linear format (nariq/FORMAT.md version 1).
// Adapted from nari-llama.cpp `ggml/src/ggml-trellis.cpp` (MIT), which is the normative host
// implementation of the tile decode, descriptor checks and side transforms. The device path
// (trellis_kernels.cu) must agree with this file element for element; trellis_reference_test.cpp
// is the gate.

#include "core/weight.h"

#include <cstdint>
#include <vector>

namespace ninfer::ops::detail::trellis {

inline constexpr int kMaxParts = 8;

struct Desc {
    std::int32_t rows_kind = 0;
    std::int32_t rows_block = 0;
    std::int32_t rows_shards = 1;
    std::int32_t cols_kind = 0;
    std::int32_t cols_block = 0;
    std::int32_t cols_shards = 1;
    std::int32_t nparts = 0;
    std::int32_t parts[kMaxParts] = {};
};

[[nodiscard]] bool is_trellis(QType qtype) noexcept;
[[nodiscard]] int bits_of(QType qtype) noexcept;

// Reads w.trellis_desc (INT32, `trellis.desc.<w>`) and validates it against the geometry. The
// check is ggml_trellis_desc_supported's, plus the permutation admission of the gathers.
[[nodiscard]] bool parse_desc(const Weight& w, std::int64_t K, std::int64_t N, Desc& out);

// The Paley matrix H_m (m in {1,12,20,68}), row major, m*m signed bytes: H = H^T is orthogonal
// up to 1/sqrt(m), which is the factor the transforms apply.
void paley_table(int m, signed char* out);

// The Paley factor of a whole-slice side: m in {1,12,20,68} with n/shards = m * 2^j.
[[nodiscard]] int hadfull_m(std::int64_t len) noexcept;

// Normalised Sylvester butterflies on `block` entries, scaled once by 1/sqrt(block).
void fwht(float* v, std::int64_t n, std::int64_t block);

// One shard of a kind-2 side. `tmp` holds 2n floats.
bool hadfull_slice(float* v, std::int64_t n, const std::int32_t* parts, int nparts, int shards,
                   int r, bool inverse, float* tmp);

// One side of a coded linear (kind 0/1/2), forward or inverse.
bool side_apply(float* v, std::int64_t n, int kind, int block, int shards, const std::int32_t* parts,
                int nparts, bool inverse, float* tmp);

// Decodes the 16-row groups [n0, n0+n_rows) of a coded matrix into `out` [n_rows,K] (row major),
// applying s16 only: W_hat, not W.
void decode_rows(QType qtype, const void* data, std::int64_t K, std::int64_t n0,
                 std::int64_t n_rows, const void* s16_f16, float* out);

// Full W in the primal basis (P_out diag(post) T_rows^T W_hat T_cols diag(pre) P_in), [N,K].
bool dequantize(const Weight& w, float* out);

// y[n + N*t] = sum_k W[n,k] x[k + K*t], with x/y in FP32. Verification path only.
bool project_host(const float* x, int tokens, const Weight& w, float* y);

// rows[k + K*t] = E[id[t], k] in the primal basis. Verification path only.
bool embedding_host(const std::int32_t* ids, int tokens, const Weight& w, float* rows);

// A device-preparable side plan: the stored-index permutation of a kind-2 side, or the block size
// of a kind-1 side. Built once per weight and cached by the dispatcher.
struct SidePlan {
    int kind = 0;
    int block = 0;
    int shards = 1;
    int nparts = 0;
    int parts[kMaxParts] = {};
    std::int64_t n = 0;
    std::int64_t shard_len = 0;  // n/shards
    int m = 1;                   // Paley factor of a kind-2 side
    std::int64_t run = 0;        // shard_len/m
    std::vector<std::int32_t> perm;  // perm[d] = stored index of logical position d
};

bool build_side_plan(const Weight& w, std::int64_t n, bool rows_side, SidePlan& out);

} // namespace ninfer::ops::detail::trellis
