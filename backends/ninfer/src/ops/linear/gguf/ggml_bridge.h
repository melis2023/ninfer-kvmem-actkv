#pragma once

// Plain entry points into the vendored ggml CUDA kernels (third_party/ggml-quants). The ggml headers
// define their own CUDA_CHECK and friends, so only the bridge's own translation unit includes them;
// everything here is raw pointers and cudaStream_t.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gguf {

// The block type's id in a GGUF tensor directory (ggml_type).
enum class GgmlType : int {
    Q4_0    = 2,
    Q5_0    = 6,
    Q8_0    = 8,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    IQ1_M   = 29,
    Q2_0    = 42,
};

struct BlockShape {
    int elements = 0;
    int bytes    = 0;
};

[[nodiscard]] BlockShape block_shape(GgmlType type);

// Streaming multiprocessors of the current device (cached per device).
[[nodiscard]] int multiprocessor_count();

// Whether the integer tensor-core prefill kernel exists for the type. IQ1_M has none (nor does it in
// llama.cpp); its wide products dequantize and take a BF16 GEMM.
[[nodiscard]] bool has_matrix_kernel(GgmlType type);

// Columns the vector kernel takes in one launch; wider products take the matrix kernel.
inline constexpr int kMaxVectorColumns = 8;

// Activation for the vector kernel: q8_1 blocks of 32 values, [columns][k / 32].
[[nodiscard]] std::size_t vector_activation_bytes(int k, int columns);
// Its layout: int8 values [columns][k] from byte 0, then half2 (d, sum) pairs [columns][k / 32]
// from this byte offset; d = max|x| / 127 over the 32 values, sum their (unquantized) total.
[[nodiscard]] std::size_t vector_activation_scales_offset(int k, int columns);
// `input_columns` (optional, [k]) makes element c of each column x[input_columns[c]].
void quantize_vector_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);

enum class Epilogue : int {
    Store,       // out = y
    Accumulate,  // out = out + y
    GateProduct, // out = act(gate) * y, gate read from a float plane
};

// The activation a gated product applies to its gate: SiLU (SwiGLU) or tanh-approximated GELU
// (GeGLU, Gemma 4).
enum class GateActivation : int {
    Silu     = 0,
    GeluTanh = 1,
};

struct VectorOutput {
    __nv_bfloat16* bf16 = nullptr; // exactly one of bf16 / f32
    float* f32          = nullptr;
    std::int64_t column_stride = 0; // elements between consecutive columns
    Epilogue epilogue          = Epilogue::Store;
    const float* gate          = nullptr; // GateProduct only
    std::int64_t gate_column_stride = 0;
    GateActivation activation       = GateActivation::Silu; // GateProduct only
};

// out[r, c] <- epilogue(sum_k W[r, k] x[k, c]) for rows [0, rows) and columns [0, columns),
// columns <= kMaxVectorColumns. `weight` points at row 0; rows are `row_bytes` apart.
void vector_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream);

// out[r, c] = act(gate_r . x_c) * (up_r . x_c) where gate row r and up row r are rows r and
// rows + r of one parent of this type.
void vector_swiglu(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, __nv_bfloat16* out,
                   std::int64_t out_column_stride, cudaStream_t stream,
                   GateActivation gate_activation = GateActivation::Silu);

// Activation for the matrix kernel, laid out for the type's scale/sum schedule. Types that share a
// schedule share an activation, see matrix_activation_layout().
[[nodiscard]] int matrix_activation_layout(GgmlType type);
// Whether the type's matrix activation keeps each 32 values' sum in F16 (Q4_0, Q4_1, Q5_1, Q4_K,
// Q5_K, IQ1_S): a sum past 65504 is infinite there, so callers whose rows can reach it scale them.
[[nodiscard]] bool matrix_activation_f16_sums(GgmlType type);
// Whether the type's vector dot reads each 32 values' F16 sum of its vector activation (Q4_K, Q5_K,
// IQ1_S): the same limit holds there.
[[nodiscard]] bool vector_activation_f16_sums(GgmlType type);
[[nodiscard]] std::size_t matrix_activation_bytes(int k, int columns);
void quantize_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream);

// Bytes of the stream-k fixup plane the matrix kernel may need for this product on the current
// device (zero when the tiles divide evenly across the machine).
[[nodiscard]] std::size_t matrix_fixup_bytes(GgmlType type, int rows, int columns);

// out[c * out_column_stride + r] = sum_k W[r, k] x[k, c] in FP32. `k` is a whole number of the
// kernel's 256-value steps. A matrix whose rows are shorter (row_bytes holds fewer than k values) is
// read past each row's end into the next row, and past the last row into zeros the caller promises
// (`tail`); its activation must be zero from the row width up to k (quantize_matrix_activation_padded).
void matrix_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, float* out,
                    std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                    bool tail = false);
// The 256-value step a padded matrix product covers for rows of `k` values.
[[nodiscard]] int matrix_padded_k(int k);
// The activation of a padded product: x BF16 [k, columns] quantized with zeros from k up to
// matrix_padded_k(k), into matrix_activation_bytes(matrix_padded_k(k), columns).
void quantize_matrix_activation_padded(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                       const std::int32_t* input_columns, void* out,
                                       cudaStream_t stream);

void matrix_product_bf16(GgmlType type, const void* weight, std::int64_t row_bytes, int rows,
                         int k, const void* activation, int columns, __nv_bfloat16* out,
                         std::int64_t out_column_stride, float* partial, void* fixup,
                         cudaStream_t stream, bool accumulate = false, bool tail = false);

// out <- epilogue(in) for an FP32 [rows, columns] plane whose columns are `in_column_stride` apart.
void store_plane(const float* in, std::int64_t in_column_stride, int rows, int columns,
                 const VectorOutput& out, cudaStream_t stream);

// out[c * k + i] = x[c * k + input_columns[i]] for each of `columns` BF16 columns of width k.
void gather_columns(const __nv_bfloat16* x, int k, int columns, const std::int32_t* input_columns,
                    __nv_bfloat16* out, cudaStream_t stream);

// Rows per pass of dequantized_product for a scratch of `scratch_bytes`.
[[nodiscard]] int dequantized_rows_per_pass(int k, std::size_t scratch_bytes);

// The same product as matrix_product for a type without an integer kernel: each pass dequantizes
// up to dequantized_rows_per_pass rows into `scratch` (BF16) and runs a cuBLAS BF16 GEMM with an
// FP32 accumulator. x is BF16 [k, columns].
void dequantized_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const __nv_bfloat16* x, int columns, float* out,
                         std::int64_t out_column_stride, void* scratch, std::size_t scratch_bytes,
                         cudaStream_t stream);

// --- Mixture of experts -------------------------------------------------------------------------
// A product over the experts a router selected: pair p (token p / per_token, its slot p %
// per_token) multiplies the rows of expert ids[p]. Experts are reached through a device table of
// base pointers, so an expert may live in a device bank, a cache slot or mapped host memory.

// The pairs grouped by expert: `sorted` lists the pairs of expert e at [bounds[e], bounds[e + 1]),
// and `active` the `*active_count` experts that have any, ascending. Device arrays.
// moe_sort_routes also lists the matrix kernel's work for each of its column tile widths
// (16, 32, 64 and 128 pairs, list w for width 16 << w): every (expert, column tile) that holds a
// pair, expert | tile << 16, ascending, `tile_count[w]` of them at tiles + w * tile_capacity.
struct MoeRouting {
    const std::int32_t* bounds       = nullptr; // [experts + 1]
    const std::int32_t* sorted       = nullptr; // [pairs]
    const std::int32_t* active       = nullptr; // [experts]
    const std::int32_t* active_count = nullptr; // [1]
    const std::int32_t* tile_count   = nullptr; // [4]
    const std::int32_t* tiles        = nullptr; // [4][tile_capacity]
    int tile_capacity                = 0;
};

[[nodiscard]] std::size_t moe_routing_bytes(int experts, int pairs);
// Groups the pairs of `ids` ([pairs], each in [0, experts)) into `workspace` (moe_routing_bytes).
// Within one expert the pairs keep no particular order; every product below writes each pair's
// result to its own place, so the order changes no value.
[[nodiscard]] MoeRouting moe_sort_routes(const std::int32_t* ids, int pairs, int experts,
                                         void* workspace, cudaStream_t stream);

// One projection of every expert: expert e's `rows` rows of `k` values start at experts[e].
struct MoeTable {
    const void* const* experts = nullptr; // device array [experts]
    std::int64_t row_bytes     = 0;
    int rows                   = 0;
    int k                      = 0;
};

// Where a product over experts puts pair p's row r.
struct MoeOutput {
    __nv_bfloat16* bf16 = nullptr; // [pairs][rows], or
    float* f32          = nullptr; // [pairs][rows], or
    // sum over a token's pairs of weights[p] * value, as 2^-32 fixed point in two's complement:
    // [tokens][rows] where token = p / per_token. Integer addition is associative, so the sum does
    // not depend on the order the experts finish in. The caller zeroes it first.
    unsigned long long* weighted = nullptr;
    const float* weights         = nullptr; // [pairs], with `weighted`
    // Multiplies the value by act(gate[p][r]) first ([pairs][rows]).
    const float* gate = nullptr;
    GateActivation activation = GateActivation::Silu;
};

// Activation column of pair p is p / column_group: the token's (column_group = per_token) for the
// first projection, the pair's own middle (column_group = 1) for the second. `max_active` bounds
// *routing.active_count (min(pairs, experts)); `chunk` is how many of an expert's columns share one
// pass over its rows (1, 2, 4 or 8).
void moe_vector_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int column_group, int per_token, const void* activation,
                        int columns, const MoeOutput& out, int chunk, cudaStream_t stream);
// silu(gate . x) * (up . x) of each pair into out [pairs][rows] (BF16): gate and up are two tables
// of one block type.
void moe_vector_swiglu(GgmlType type, const MoeTable& gate, const MoeTable& up,
                       const MoeRouting& routing, int max_active, int column_group,
                       const void* activation, int columns, __nv_bfloat16* out, int chunk,
                       cudaStream_t stream, GateActivation gate_activation = GateActivation::Silu);

// One decode or verification call's experts (up to 8 tokens) in few launches: moe_decode_up writes
// silu(gate . x) * (up . x) of every listed pair to `middle`, quantizing its tokens' BF16 inputs
// itself; moe_decode_down runs the down rows of every listed pair into the 2^-32 fixed-point sum
// and, when `y` is given, its last block adds the external products and stores y. Each block row
// serves one expert's pairs, found by scanning the ids (no sort). The arithmetic of every row is
// moe_vector_swiglu's and moe_vector_product's, so the sums are theirs bit for bit. `chunk` is how
// many of an expert's pairs share a pass over its rows (1, 2, 4 or 8).
//
// A call can be prepared once instead of in every block: moe_decode_prep lists each slot's pairs
// (records) and quantizes the tokens' inputs, and the up launches quantize the pairs' middles for
// the down kernel (and list the second stage's slots, whose ids are final only then).
struct MoeDecodeRecord {
    // The slot's expert and the id set it came from (0: ids_a, 1: ids_b), with that expert's
    // pairs in ascending order and their tokens; n == 0 when the slot is negative in its set or
    // repeats an earlier slot's expert.
    std::int32_t expert = -1, n = 0, source = 0, reserved = 0;
    std::int32_t pairs[8] = {}, columns[8] = {};
};
struct MoeDecodePrepArgs {
    const __nv_bfloat16* m = nullptr; // BF16 [tokens][2560]
    int tokens             = 0;
    const std::int32_t* ids = nullptr; // [slots], the first stage's set (ids_a)
    int slots = 0, per_token = 10;
    MoeDecodeRecord* records = nullptr; // [slots]
    // The inputs as `type`'s up kernel reads them: q8_1 values [tokens][2560], (d, sum) and the
    // integer sum of every 32 values [tokens][80].
    std::int8_t* qs          = nullptr;
    half2* ds                = nullptr;
    int* qsum                = nullptr;
    unsigned long long* zero = nullptr; // zeroed when given
    int zero_count           = 0;
};
void moe_decode_prep(GgmlType type, const MoeDecodePrepArgs& args, cudaStream_t stream);
// Whether `type`'s decode kernels read transposed activations (MoeDecodeUpArgs::middle_transposed).
[[nodiscard]] bool moe_decode_transposed(GgmlType type);

struct MoeDecodeUpArgs {
    const __nv_bfloat16* m = nullptr; // BF16 [tokens][k]
    int k = 2560, rows = 640;
    const std::int32_t* ids = nullptr; // [slots]: negative entries skipped; null: all expert 0
    int slots = 0, per_token = 10;
    const std::uint8_t* const* gate = nullptr;
    const std::uint8_t* const* up   = nullptr;
    std::int64_t row_bytes          = 0;
    __nv_bfloat16* middle           = nullptr; // [slots][rows]
    unsigned long long* zero        = nullptr; // zeroed first when given
    int zero_count                  = 0;
    // Prepared: the slots' records (instead of scanning ids) and the quantized inputs (instead of m).
    const MoeDecodeRecord* records = nullptr;
    const std::int8_t* qs          = nullptr;
    const half2* ds                = nullptr;
    const int* qsum                = nullptr;
    // When given, every block quantizes its 32 middle values of every pair for the down kernel
    // (transposed when its decoder reads them so) instead of storing them: [slots][640], and
    // [slots][20] (d, sum) and integer sums. Requires rows to be whole groups of 32.
    std::int8_t* middle_qs         = nullptr;
    half2* middle_ds               = nullptr;
    int* middle_qsum               = nullptr;
    bool middle_transposed         = false;
    // Scanning `ids`, a block row also lists its slot in `write_records` as `record_source`: the
    // first stage (0) every slot, the second (1) the slots it serves.
    MoeDecodeRecord* write_records = nullptr;
    int record_source              = 1;
};
struct MoeDecodeDownArgs {
    int rows = 2560; // the down projection's outputs; its k is 640
    // A slot is served from table_a when ids_a[slot] >= 0, else from table_b when ids_b[slot] >= 0.
    const std::int32_t* ids_a = nullptr; // null: all expert 0
    const std::int32_t* ids_b = nullptr;
    int slots = 0, per_token = 10;
    const std::uint8_t* const* table_a = nullptr;
    const std::uint8_t* const* table_b = nullptr;
    std::int64_t row_bytes             = 0;
    const __nv_bfloat16* middle        = nullptr; // [slots][640]
    const float* weights               = nullptr; // [slots]
    unsigned long long* fixed          = nullptr; // [tokens][rows]
    int tokens                         = 0;
    // The call's last launch: y [tokens][rows], external products (as add_products), and a
    // completion counter that starts at zero and is left at zero.
    float* y                         = nullptr;
    unsigned int* counter            = nullptr;
    const float* products            = nullptr;
    const std::int32_t* product_pairs = nullptr;
    const std::int32_t* product_count = nullptr;
    // Prepared: the slots' records and the quantized middles (MoeDecodeUpArgs::middle_qs).
    const MoeDecodeRecord* records = nullptr;
    const std::int8_t* qs          = nullptr;
    const half2* ds                = nullptr;
    const int* qsum                = nullptr;
};
void moe_decode_up(GgmlType type, const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream);
void moe_decode_down(GgmlType type, const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream);

// The matrix kernel over the experts a router selected, for products wide enough for its tiles:
// pair p's FP32 row lands at out[p * table.rows]. The activation holds the pairs' columns in
// routing order (quantize_moe_matrix_activation into moe_matrix_activation_bytes); `routing` comes
// from moe_sort_routes over the same pairs, and `max_active` bounds its active experts. One CTA
// runs per row tile and listed column tile. The kernel reads k in whole 256-value steps: a
// geometry whose k is not a whole number of them needs `tail`, the caller's promise that zeros
// follow each bank's last expert (Binder::device_tail, a zeroed slot pool), since a row's last
// step reads into the next row. A type without an integer kernel, or a geometry no tile fits
// (rows a whole number of row tiles), is not supported and stays on moe_vector_product.
[[nodiscard]] bool moe_matrix_supported(GgmlType type, int rows, int k, bool tail);
void moe_matrix_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int pairs, bool tail, const void* activation, float* out,
                        cudaStream_t stream);
// Column c of the activation is x column routing.sorted[c] / column_group, in the layout of the
// type's scale/sum schedule (matrix_activation_layout), zero past k up to a whole 256-value step.
[[nodiscard]] std::size_t moe_matrix_activation_bytes(int k, int pairs);
void quantize_moe_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k,
                                    const MoeRouting& routing, int pairs, int column_group,
                                    void* out, cudaStream_t stream);

// out[i, :] = dequantize(W[row_ids ? row_ids[i] : i, :]) as BF16, rows `out_row_stride` apart.
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __nv_bfloat16* out,
                     std::int64_t out_row_stride, cudaStream_t stream);
// The same rows in FP32, where every value is exact.
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, float* out,
                     std::int64_t out_row_stride, cudaStream_t stream);
// The same rows in FP16, the operand type of sm_70's tensor cores (ops/linear/sm70).
void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __half* out,
                     std::int64_t out_row_stride, cudaStream_t stream);

} // namespace ninfer::ops::gguf
