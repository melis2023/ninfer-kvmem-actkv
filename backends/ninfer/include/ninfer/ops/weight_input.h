#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sparse_moe.h"

#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
    // BF16 [K] signs of a Hadamard-rotated matrix: the Use multiplies the stored rows with
    // hadamard_transform(input, signs) instead of the input. Empty for an ordinary matrix. No Op
    // reads it; the caller rotates the activation (see the model's execution layer).
    Tensor hadamard_signs{};
    // INT32 [K] input gather of a GGUF matrix stored over permuted input columns (see
    // Weight::input_columns). Empty for every other matrix.
    Tensor input_columns{};
    // NariQ coded weight (Weight::trellis_*): the per-tile scale, the two side diagonals, the
    // optional input/output gathers and the host-resident descriptor. All empty for a plain matrix.
    Tensor trellis_s16{}, trellis_pre{}, trellis_post{};
    Tensor trellis_pin{}, trellis_pout{};
    Tensor trellis_desc{};
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    Tensor hadamard_signs{};
};

struct PairedProjectionWeights {
    Weight first, second;
    // Split parents reach their Op without a SingleProjectionWeight to carry the decision, so the
    // pair carries it. A16Only unless a registered route exists for these exact shapes.
    LinearPolicy policy = LinearPolicy::A16Only;
    Tensor hadamard_signs{};
};

// A fused input projection over GGUF matrices, whose logical parts may sit in different parents
// of different block types. Each part is rows of one parent landing at `row` of output `output`:
// for a GDN projection 0 = q/k/v and 1 = z; for an attention projection 0 = query, 1 = gate,
// 2 = key and 3 = value.
struct GgufProjectionPart {
    Weight weight;
    std::int32_t output = 0;
    std::int32_t row    = 0;
};

struct GgufProjectionWeights {
    std::vector<GgufProjectionPart> parts;
    LinearPolicy policy = LinearPolicy::A16Only;
    Tensor hadamard_signs{};
};

// Where the rows of one coded parent land in the caller's output bank. A coded matrix cannot be
// cut into row slices the way a dense one can - its output-side Hadamard spans whole row blocks of
// the stored matrix - so the parent is projected once and its already logical rows are copied to
// the destinations by an exact BF16 row copy.
struct CodedProjectionReach {
    std::int32_t parent = 0;  // index into CodedProjectionWeights::parents
    std::int32_t output = 0;  // index into the caller's output array
    std::int32_t dst_row = 0;
    std::int32_t rows = 0;
    // The parent rows this reach reads when they are a stride pattern rather than the identity:
    // parent[(i / src_group) * src_stride + src_origin + i % src_group]. Zero leaves it the identity.
    std::int32_t src_group  = 0;
    std::int32_t src_stride = 0;
    std::int32_t src_origin = 0;
};

struct CodedProjectionWeights {
    std::vector<SingleProjectionWeight> parents;
    std::vector<CodedProjectionReach> reaches;
    Tensor hadamard_signs{};

    [[nodiscard]] const Weight& leader() const { return parents.front().weight; }
};

using ProjectionWeights =
    std::variant<SingleProjectionWeight, PairedProjectionWeights, GgufProjectionWeights,
                 CodedProjectionWeights>;

// Whether the rows of `inputs`, in order, are one contiguous region of one parent.
[[nodiscard]] bool joins(std::span<const WeightInput> inputs);

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
// Coded (NariQ TRL) forms. A coded parent keeps `query|key|value` (GDN, 2*key_rows + value_rows)
// in already logical order, so each logical block becomes one Reach into the caller's outputs
// instead of a row slice of its own matrix. The full-attention parent holds `query|gate` fused
// head by head - the GGUF's `attn_q` is 2*head_width rows per head in qwen3.5 turn order, exactly
// as `tools/convert/qwen3_5.py` slices it - so that parent needs `head_width` to read its query and
// gate reaches through the stride pattern instead of as two contiguous blocks.
[[nodiscard]] ProjectionWeights prepare_coded_attn_projection(const WeightInput& packed,
                                                              const WeightInput& key,
                                                              const WeightInput& value,
                                                              std::uint64_t query_rows,
                                                              std::uint64_t key_rows,
                                                              std::uint64_t head_width);
[[nodiscard]] ProjectionWeights prepare_coded_gdn_projection(const WeightInput& packed,
                                                            const WeightInput& z,
                                                            std::uint64_t key_rows,
                                                            std::uint64_t value_rows);
[[nodiscard]] ProjectionWeights prepare_coded_gating_projection(const WeightInput& a,
                                                                const WeightInput& b);
// Runs one trellis_linear per coded parent and copies its rows into `outputs` at their Reach.
[[nodiscard]] std::size_t coded_projection_workspace_bytes(const CodedProjectionWeights& weights,
                                                           std::int32_t max_tokens);
void coded_projection(const Tensor& x, const CodedProjectionWeights& weights,
                      Tensor* const* outputs, std::size_t output_count, std::int32_t max_tokens,
                      WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);

} // namespace ninfer::ops
