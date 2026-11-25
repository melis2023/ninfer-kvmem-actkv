#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/trellis.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// Whether a proposal head's projection can store FP32 logits. Its top-1 then ranks the unrounded
// scores, so two candidates BF16 would round to one value are told apart.
[[nodiscard]] inline bool fp32_proposal_logits(const LinearParameters& head) {
    return head.weight.qtype == QType::BF16 || is_gguf(head.weight.qtype);
}

// The dtype the target output head writes logits in; the DSpine drafter reads them through
// the same head and must be told which type its proposal operand is.
[[nodiscard]] inline DType target_logits_dtype(const LinearParameters& head) {
    return fp32_proposal_logits(head) ? DType::FP32 : DType::BF16;
}

// A banded coded head writes BF16 logits: each band's rows are one BF16 rounding.
[[nodiscard]] inline DType target_logits_dtype(const HeadParameters& head) {
    return head.banded() ? DType::BF16 : target_logits_dtype(head.single);
}

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear(x, p.weight, output, p.policy, workspace, stream);
}

// NINFER_TRACE_XFILE=<path>: replace a banded head's input with a column read from disk (K FP32
// values), so the head Op can be probed with a vector whose expected result is computable off-line.
// A no-op unless the variable is set; the file is read once per process.
inline Tensor probe_head_input(const Tensor& input, WorkspaceArena& workspace, cudaStream_t stream) {
    const char* path = std::getenv("NINFER_TRACE_XFILE");
    if (path == nullptr || path[0] == '\0') {
        return input;
    }
    static std::vector<float> host;
    if (host.empty()) {
        std::ifstream file(path, std::ios::binary);
        if (!file) { throw std::runtime_error("NINFER_TRACE_XFILE: cannot open the probe vector"); }
        const std::vector<char> raw{std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>()};
        std::vector<float> values(raw.size() / sizeof(float));
        std::memcpy(values.data(), raw.data(), values.size() * sizeof(float));
        host = std::move(values);
    }
    const std::int64_t k = input.ne[0];
    if (static_cast<std::int64_t>(host.size()) != k) {
        throw std::runtime_error("NINFER_TRACE_XFILE: the probe vector is not K wide");
    }
    Tensor out = workspace.alloc(DType::BF16, {dimension(k), input.ne[1]});
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < bits.size(); ++i) {
        std::uint32_t word;
        std::memcpy(&word, &host[i], sizeof(word));
        bits[i] = static_cast<std::uint16_t>(word >> 16);
    }
    for (std::int32_t column = 0; column < input.ne[1]; ++column) {
        void* destination = static_cast<std::uint8_t*>(out.data) +
                            static_cast<std::size_t>(column) * bits.size() * sizeof(std::uint16_t);
        cudaMemcpyAsync(destination, bits.data(), bits.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice, stream);
    }
    static bool reported = false;
    if (!reported) {
        reported = true;
        std::uint16_t check[4] = {};
        cudaStreamSynchronize(stream);
        cudaMemcpy(check, out.data, sizeof(check), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "probe | %s | host %.4f %.4f %.4f | bits %04x %04x %04x %04x\n", path,
                     static_cast<double>(host[0]), static_cast<double>(host[1]),
                     static_cast<double>(host[2]), bits[0], bits[1], bits[2], bits[3]);
        (void)check;
    }
    return out;
}

// The NariQ banded head: one coded linear per band into the concatenated plane, then the
// vocabulary gather, which is `ops::trellis_banded_head`.
inline void project(const Tensor& input, const HeadParameters& head, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    InputBasis basis = InputBasis::Primal) {
    if (!head.banded()) {
        project(input, head.single, output, workspace, stream, basis);
        return;
    }
    auto scope     = workspace.scope();
    const Tensor probe = probe_head_input(input, workspace, stream);
    static const bool trace_probe = [] {
        const char* value = std::getenv("NINFER_TRACE_XFILE");
        return value != nullptr && value[0] != '\0';
    }();
    if (trace_probe) {
        std::fprintf(stderr, "probe | banded head | input %lld x %lld | replaced %d\n",
                     static_cast<long long>(input.ne[0]), static_cast<long long>(input.ne[1]),
                     probe.data != nullptr && probe.data != input.data);
    }
    const Tensor x = probe.data == input.data
                         ? rotated_input(input, head.single.hadamard_signs, workspace, stream, basis)
                         : probe;
    std::vector<const Weight*> bands;
    bands.reserve(head.bands.size());
    for (const auto& band : head.bands) {
        if (!is_coded(band.weight.qtype)) {
            throw std::logic_error("a banded output head must store coded bands");
        }
        bands.push_back(&band.weight);
    }
    ops::trellis_banded_head(x, bands.data(), bands.size(), head.gather, output, x.ne[1], workspace,
                             stream);
}

// `wide_verification` keeps the neural path's residual activation precision in a wide
// single-row copy verification (see residual_projection_policy). `accum`, when given, is an FP32
// residual stream that receives the same delta: the coded route already forms one, so it can hand
// it to the wide accumulator without a second projection.
inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream,
                        InputBasis basis = InputBasis::Primal, bool wide_verification = false,
                        Tensor* accum = nullptr) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    if (is_coded(p.weight.qtype)) {
        // The fused residual add owns an integer route per format; a coded weight has only the
        // trellis linear, so its residual is formed separately.
        Tensor delta = workspace.alloc(DType::BF16, {p.weight.n, x.ne[1]});
        {
            auto call = workspace.scope();
            ops::linear(x, p.weight, delta, p.policy, workspace, stream);
        }
        ops::residual_add(delta, residual, stream);
        if (accum != nullptr) { ops::residual_add(delta, *accum, stream); }
        return;
    }
    if (accum != nullptr) {
        throw std::logic_error("project_add: a wide residual accumulator needs a coded projection");
    }
    ops::linear_add(x, p.weight, residual, residual_projection_policy(p, wide_verification),
                    workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear_swiglu(x, p.weight, output, p.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
