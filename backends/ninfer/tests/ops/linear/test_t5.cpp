// T5_G128_FP16 (base-3) projections and embedding against an FP64 oracle from the format
// weight (n, k) = (code - 1) * scale[n][k / 128],
// code = (codes[n][k / 4] >> 2 * (k % 4)) & 3. The weight is built through the production
// geometry and native Weight view, including a row view of a fused parent.
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ops/linear/t5/t5_project.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// One BF16 unit roundoff in relative L2; gross error covers final BF16 storage (A16 criterion).
constexpr ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
// The shared Linear A8 criterion: activation quantization allowance plus BF16 storage.
constexpr ReductionCriterion kA8{0.04, 1.0 / 256.0, 0.06};

ReductionCriterion criterion(ops::LinearPolicy policy) {
    return ops::allows_a8(policy) ? kA8 : kA16;
}

const char* policy_name(ops::LinearPolicy policy) {
    return ops::allows_a8(policy) ? " A8" : " A16";
}

struct Ternary {
    std::int32_t n = 0, k = 0;
    std::vector<std::uint8_t> code;  // [n][k] in {0, 1, 2}
    std::vector<float> scale;        // [n][k / 128], FP16-representable
    std::vector<std::byte> payload;  // RowSplit, base-3 codes
    WeightGeometry geometry;
    DeviceBuffer device;
    WeightParent parent;

    Ternary(std::int32_t rows, std::int32_t columns, std::uint32_t seed) : n(rows), k(columns) {
        std::mt19937 rng(seed);
        code.resize(std::size_t(n) * k);
        scale.resize(std::size_t(n) * (k / 128));
        for (auto& c : code) c = static_cast<std::uint8_t>(rng() % 3);
        std::uniform_real_distribution<float> magnitude(0.002f, 0.05f);
        for (auto& s : scale) s = __half2float(__float2half(magnitude(rng)));
        if (n == 248320) {
            // Exactly represented integer logits, with deliberate ties and ample score gaps.
            // This makes top-k ordering an exact oracle instead of treating near-tie floating
            // differences in a random full-vocabulary dot product as a ranking defect.
            std::fill(scale.begin(), scale.end(), 0.0f);
            for (int j = 0; j < 32; ++j) {
                const int row = 37 + 7919 * j;
                std::fill_n(code.begin() + std::size_t(row) * k, 128, std::uint8_t{2});
                scale[std::size_t(row) * (k / 128)] = float(j / 2 + 1) / 128.0f;
            }
            // A deliberately dominant padding row must never enter the public vocabulary.
            std::fill(code.end() - k, code.end(), std::uint8_t{2});
            scale[std::size_t(n - 1) * (k / 128)] = 8.0f;
        }
        const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
        geometry = weight_geometry(QType::T5_G128_FP16, QuantLayout::RowSplit, shape);
        payload.assign(geometry.bytes, std::byte{0});
        for (std::int32_t row = 0; row < n; ++row) {
            // Base-3 units (design doc 9.1): byte i < 12 of unit u, g = i / 4, j = i % 4, holds
            // t_m = c[64 u + 20 g + 4 m + j]; byte 12 holds c[64 u + 60 + m] (m < 4) and t_4 = 0.
            // q = ceil(256 v / 243), v = sum_m t_m 3^(4 - m).
            const std::uint8_t* c = &code[std::size_t(row) * k];
            for (std::int32_t unit = 0; unit < k / 64; ++unit) {
                for (int i = 0; i < 13; ++i) {
                    int v = 0;
                    for (int m = 0; m < 5; ++m) {
                        const int column = i < 12 ? 20 * (i / 4) + 4 * m + i % 4 : 60 + m;
                        v = 3 * v + (i < 12 || m < 4 ? c[64 * unit + column] : 0);
                    }
                    payload[std::size_t(row) * (k / 64 * 13) + unit * 13 + i] =
                        std::byte((256 * v + 242) / 243);
                }
            }
            for (std::int32_t group = 0; group < k / 128; ++group) {
                const __half h = __float2half(scale[std::size_t(row) * (k / 128) + group]);
                std::memcpy(&payload[geometry.scale_offset + (std::size_t(row) * (k / 128) + group) * 2],
                            &h, 2);
            }
        }
        device = DeviceBuffer(payload.size());
        device.copy_from_host(payload.data(), payload.size());
        parent = {geometry, static_cast<const std::byte*>(device.p)};
    }

    // Prism rotation: the weight multiplies (1/32) H (signs * x) per 1024-column block.
    std::vector<float> signs;
    DeviceBuffer device_signs;

    void rotate(std::uint32_t seed) {
        std::mt19937 rng(seed);
        signs.resize(std::size_t(k));
        for (auto& sign : signs) sign = rng() & 1 ? 1.0f : -1.0f;
        device_signs = to_device_bf16(signs);
    }

    Weight rows(std::int32_t first, std::int32_t count) const {
        Weight view = native_weight(WeightView{{std::uint64_t(count), std::uint64_t(k)},
                                               {{&parent, std::uint64_t(first) * k,
                                                 std::uint64_t(first + count) * k}}});
        return view;
    }

    double weight(std::int32_t row, std::int32_t column) const {
        return (double(code[std::size_t(row) * k + column]) - 1.0) *
               scale[std::size_t(row) * (k / 128) + column / 128];
    }
};

std::vector<float> activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<float> x(std::size_t(k) * t);
    fill_uniform(x, seed, -4.0f, 4.0f);
    round_to_bf16(x);
    return x;
}

// The weight's input: the logical input itself, or its Prism rotation (1/32) H (signs * input)
// with the Sylvester Walsh-Hadamard matrix H[r][c] = (-1)^popcount(r & c) of every 1024-column
// block, FP64.
std::vector<double> weight_input(const Ternary& w, std::vector<double> input, std::int32_t t) {
    if (w.signs.empty()) return input;
    std::vector<double> rotated(input.size());
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t block = 0; block < w.k; block += 1024) {
            const std::size_t base = std::size_t(token) * w.k + block;
            for (int r = 0; r < 1024; ++r) {
                double sum = 0;
                for (int c = 0; c < 1024; ++c) {
                    const double term = input[base + c] * w.signs[block + c];
                    sum += std::popcount(unsigned(r & c)) & 1 ? -term : term;
                }
                rotated[base + r] = sum / 32.0;
            }
        }
    }
    return rotated;
}

// out[t][r] = sum_c W[first + r][c] input[t][c] for the weight input [T][K], FP64.
std::vector<double> project(const Ternary& w, std::int32_t first, std::int32_t rows,
                            const std::vector<double>& input, std::int32_t t) {
    std::vector<double> out(std::size_t(rows) * t), row(std::size_t(w.k));
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t c = 0; c < w.k; ++c) row[c] = w.weight(first + r, c);
        for (std::int32_t token = 0; token < t; ++token) {
            const double* x = &input[std::size_t(token) * w.k];
            double sum      = 0;
            for (std::int32_t c = 0; c < w.k; ++c) sum += row[c] * x[c];
            out[std::size_t(token) * rows + r] = sum;
        }
    }
    return out;
}

std::vector<double> oracle(const Ternary& w, std::int32_t first, std::int32_t rows,
                           const std::vector<float>& x, std::int32_t t) {
    return project(w, first, rows, weight_input(w, std::vector<double>(x.begin(), x.end()), t), t);
}

int linear_case(const Ternary& w, std::int32_t first, std::int32_t rows, std::int32_t t,
                bool graph, ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
    auto x = activation(w.k, t, 17u * t + rows);
    auto device_x = to_device_bf16(x);
    GuardedDeviceBuffer out(std::size_t(rows) * t * 2);
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    Tensor to(out.data(), DType::BF16, {rows, t});
    const Weight view = w.rows(first, rows);
    DeviceArena workspace(
        ops::linear_workspace_capacity_bytes(QType::T5_G128_FP16, rows, w.k, policy, 1, t));
    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::linear(tx, view, to, policy, workspace, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        x = activation(w.k, t, 29u * t + rows);
        std::vector<std::uint16_t> updated(x.size());
        for (std::size_t i=0; i<x.size(); ++i) updated[i]=f32_to_bf16(x[i]);
        device_x.copy_from_host(updated.data(), updated.size() * 2);
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        ops::linear(tx, view, to, policy, workspace, nullptr);
        cuda_synchronize();
    }
    if (workspace.used() != 0) throw std::runtime_error("workspace leak");
    const std::string label = "t5 linear [" + std::to_string(rows) + "," + std::to_string(w.k) +
                              "] rows+" + std::to_string(first) + " T=" + std::to_string(t) +
                              policy_name(policy);
    return verify_reduction(label, from_device_bf16(out.data(), std::size_t(rows) * t),
                            oracle(w, first, rows, x, t), criterion(policy)) +
           out.verify_guards(label.c_str());
}

int linear_add_case(const Ternary& w, std::int32_t t,
                    ops::LinearPolicy policy = ops::LinearPolicy::AllowA8) {
    const auto x        = activation(w.k, t, 91u + t);
    auto residual_value = activation(w.n, t, 92u + t);
    auto expected       = oracle(w, 0, w.n, x, t);
    for (std::size_t i = 0; i < expected.size(); ++i) expected[i] += residual_value[i];
    auto device_x        = to_device_bf16(x);
    auto device_residual = to_device_bf16(residual_value);
    Tensor tx(device_x.p, DType::BF16, {w.k, t});
    Tensor tr(device_residual.p, DType::BF16, {w.n, t});
    DeviceArena workspace(
        ops::linear_add_workspace_capacity_bytes(QType::T5_G128_FP16, w.n, w.k, policy, 1, t));
    ops::linear_add(tx, w.rows(0, w.n), tr, policy, workspace, nullptr);
    cuda_synchronize();
    return verify_reduction("t5 linear_add [" + std::to_string(w.n) + "," + std::to_string(w.k) +
                                "] T=" + std::to_string(t) + policy_name(policy),
                            from_device_bf16(device_residual.p, expected.size()), expected,
                            criterion(policy));
}

// Embedding gather: the logical row ids[t] of the table, i.e. the decoded stored row or, for a
// rotated table, signs * H(z') / 32 per 1024-column block (FP64 Sylvester oracle).
int embedding_case(const Ternary& w, const std::vector<std::int32_t>& ids) {
    // A row of the logical table W' H S is (z' H) S: the signs follow the butterfly.
    const auto t = static_cast<std::int32_t>(ids.size());
    std::vector<double> expected(std::size_t(w.k) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t block = 0; block < w.k; block += 1024) {
            for (int r = 0; r < 1024; ++r) {
                double value = w.weight(ids[token], block + r);
                if (!w.signs.empty()) {
                    double sum = 0;
                    for (int c = 0; c < 1024; ++c) {
                        const double term = w.weight(ids[token], block + c);
                        sum += std::popcount(unsigned(r & c)) & 1 ? -term : term;
                    }
                    value = sum / 32.0 * w.signs[block + r];
                }
                expected[std::size_t(token) * w.k + block + r] = value;
            }
        }
    }
    DeviceBuffer device_ids(ids.size() * 4);
    device_ids.copy_from_host(ids.data(), ids.size() * 4);
    GuardedDeviceBuffer out(std::size_t(w.k) * t * 2);
    Tensor tids(device_ids.p, DType::I32, {t});
    Tensor tout(out.data(), DType::BF16, {w.k, t});
    Tensor signs(w.device_signs.p, DType::BF16, {w.k});
    if (w.signs.empty()) {
        ops::embedding(tids, w.rows(0, w.n), tout, nullptr);
    } else {
        ops::embedding_rotated(tids, w.rows(0, w.n), signs, tout, nullptr);
    }
    cuda_synchronize();
    const std::string label = std::string("t5 embedding") + (w.signs.empty() ? "" : " rotated") +
                              " T=" + std::to_string(t);
    return verify_reduction(label, from_device_bf16(out.data(), expected.size()), expected, kA16) +
           out.verify_guards(label.c_str());
}

} // namespace


int main() {
    if (cuda_unavailable()) return 77;
    int failures = 0;
    const Ternary small(301, 3072, 15);
    for (int t : {1,2,3,4,5,8,9,16,17,24,25,32,33,65})
        failures += linear_case(small,0,301,t,t==3 || t==9);
    failures += linear_case(small,44,200,6,false);
    const Ternary tile(384, 5120, 18);
    for(int t : {32,33,64,65,127,128,129,193,256})
        failures += linear_case(tile,0,384,t,t==65 || t==129);
    // Real reduction widths and row views, including the tallest gate/up parent.
    for (const auto [n,k] : std::array<std::pair<int,int>,4>{{
        {5120,6144},{5120,17408},{16384,5120},{34816,5120}}}) {
        const Ternary w(n,k,3000u+n+k);
        for(int t : {1,4,9}) failures += linear_case(w,0,n,t,t==4);
        if(n==5120) failures += linear_add_case(w,3);
    }
    // Same split-parent layout as production, including GDN's strided qkv destinations.
    for(int t : {1,4,9,65}) {
        const Ternary first(4096,5120,71), second(12288,5120,72);
        auto x=activation(5120,t,73+t); auto dx=to_device_bf16(x);
        GuardedDeviceBuffer dqkv(std::size_t(10240)*t*2), dz(std::size_t(6144)*t*2);
        Tensor tx(dx.p,DType::BF16,{5120,t}), qkv(dqkv.data(),DType::BF16,{10240,t}),
               z(dz.data(),DType::BF16,{6144,t});
        auto policy=ops::LinearPolicy::AllowA8;
        const auto bytes=ops::gdn_input_proj_split_workspace_capacity_bytes(
            QType::T5_G128_FP16,4096,QType::T5_G128_FP16,12288,5120,policy,t,t);
        DeviceArena ws(bytes);
        ops::gdn_input_proj(tx,first.rows(0,4096),second.rows(0,12288),qkv,z,policy,ws,nullptr);
        cuda_synchronize();
        const auto a=oracle(first,0,4096,x,t), b=oracle(second,0,6144,x,t);
        std::vector<double> expected(std::size_t(10240)*t);
        for(int c=0;c<t;++c) {
            std::copy_n(a.data()+c*4096,4096,expected.data()+c*10240);
            std::copy_n(b.data()+c*6144,6144,expected.data()+c*10240+4096);
        }
        failures += verify_reduction("T5 GDN qkv",from_device_bf16(dqkv.data(),expected.size()),expected,kA8);
        failures += verify_reduction("T5 GDN z",from_device_bf16(dz.data(),std::size_t(6144)*t),oracle(second,6144,6144,x,t),kA8);
        failures += dqkv.verify_guards("T5 GDN qkv") + dz.verify_guards("T5 GDN z");
        if(ws.used()!=0 || ws.peak_used()>bytes) ++failures;
    }
    // Split attention output order and composed SwiGLU, independent FP64 references.
    for (int t : {1, 5, 33, 65}) {
        const Ternary qk(7168,5120,81), gv(7168,5120,82);
        auto x=activation(5120,t,83+t); auto dx=to_device_bf16(x);
        Tensor tx(dx.p,DType::BF16,{5120,t});
        std::array<GuardedDeviceBuffer,4> storage{
            GuardedDeviceBuffer(std::size_t(6144)*t*2), GuardedDeviceBuffer(std::size_t(6144)*t*2),
            GuardedDeviceBuffer(std::size_t(1024)*t*2), GuardedDeviceBuffer(std::size_t(1024)*t*2)};
        Tensor q(storage[0].data(),DType::BF16,{6144,t}), gate(storage[1].data(),DType::BF16,{6144,t}),
               k(storage[2].data(),DType::BF16,{1024,t}), v(storage[3].data(),DType::BF16,{1024,t});
        auto policy=ops::LinearPolicy::AllowA8;
        auto bytes=ops::attn_input_proj_split_workspace_capacity_bytes(
            QType::T5_G128_FP16,7168,QType::T5_G128_FP16,7168,5120,policy,t,t);
        DeviceArena ws(bytes);
        ops::attn_input_proj(tx,qk.rows(0,7168),gv.rows(0,7168),q,gate,k,v,policy,ws,nullptr);
        cuda_synchronize();
        const Ternary* weights[]={&qk,&gv,&qk,&gv};
        for(int i=0;i<4;++i) {
            const int count=i<2?6144:1024, first=i<2?0:6144;
            failures+=verify_reduction("T5 attention output "+std::to_string(i),
                from_device_bf16(storage[i].data(),std::size_t(count)*t),
                oracle(*weights[i],first,count,x,t),kA8);
            failures+=storage[i].verify_guards("T5 attention");
        }
        if(ws.used()!=0 || ws.peak_used()>bytes) ++failures;
    }
    {
        const Ternary w(34816,5120,91); const int t=5;
        auto x=activation(5120,t,92); auto dx=to_device_bf16(x);
        GuardedDeviceBuffer dest(std::size_t(17408)*t*2);
        Tensor tx(dx.p,DType::BF16,{5120,t}), out(dest.data(),DType::BF16,{17408,t});
        auto policy=ops::LinearPolicy::AllowA8;
        auto bytes=ops::linear_swiglu_workspace_capacity_bytes(QType::T5_G128_FP16,34816,5120,policy,t,t);
        DeviceArena ws(bytes);
        ops::linear_swiglu(tx,w.rows(0,34816),out,policy,ws,nullptr);
        cuda_synchronize();
        auto gate=oracle(w,0,17408,x,t), up=oracle(w,17408,17408,x,t);
        for(std::size_t i=0;i<gate.size();++i) gate[i]=gate[i]/(1+std::exp(-gate[i]))*up[i];
        failures+=verify_reduction("T5 SwiGLU",from_device_bf16(dest.data(),gate.size()),gate,kA8);
        failures+=dest.verify_guards("T5 SwiGLU");
        if(ws.used()!=0 || ws.peak_used()>bytes) ++failures;
    }
    Ternary table(301,5120,33);
    failures += embedding_case(table,{0,300,7,7,150});
    table.rotate(34);
    failures += embedding_case(table,{0,300,7,7,150});
    {
        // Constant represented inputs are exact under s8 quantization. Thus the full-vocabulary
        // FP64 score oracle tests mapping, valid-row exclusion and stable BF16 ties independently.
        const Ternary head(248320,5120,111); const int valid=248077, tokens=9;
        std::vector<float> ones(5120,1.0f);
        auto ideal=oracle(head,0,valid,ones,1);
        std::vector<std::pair<float,int>> ranked;
        for(int i=0;i<valid;++i) ranked.emplace_back(bf16_to_f32(f32_to_bf16(float(ideal[i]))),i);
        std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
        auto x=to_device_bf16(std::vector<float>(std::size_t(5120)*tokens,1.0f));
        GuardedDeviceBuffer ids(std::size_t(16)*tokens*4),scores(std::size_t(16)*tokens*4);
        Tensor tx(x.p,DType::BF16,{5120,tokens}), ti(ids.data(),DType::I32,{16,tokens}),
               ts(scores.data(),DType::FP32,{16,tokens});
        auto bytes=ops::linear_topk_workspace_capacity_bytes(QType::T5_G128_FP16,248320,5120,1,tokens);
        DeviceArena ws(bytes);
        ops::linear_topk(tx,head.rows(0,head.n),valid,ti,ts,ws,nullptr); cuda_synchronize();
        auto actual=from_device<std::int32_t>(ids.data(),16*tokens);
        auto values=from_device<float>(scores.data(),16*tokens);
        for(int t=0;t<tokens;++t) for(int r=0;r<16;++r) {
            if(actual[t*16+r]!=ranked[r].second || values[t*16+r]!=ranked[r].first) {
                if(t==0) std::cerr << "T5 topk rank " << r << ": " << actual[r] << "/" << values[r]
                                  << " expected " << ranked[r].second << "/" << ranked[r].first << '\n';
                ++failures;
            }
        }
        CUDA_CHECK(cudaMemset(x.p,0,std::size_t(5120)*tokens*2));
        ops::linear_topk(tx,head.rows(0,head.n),valid,ti,ts,ws,nullptr); cuda_synchronize();
        actual=from_device<std::int32_t>(ids.data(),16*tokens);
        values=from_device<float>(scores.data(),16*tokens);
        for(int t=0;t<tokens;++t) for(int r=0;r<16;++r)
            if(actual[t*16+r]!=r || values[t*16+r]!=0) ++failures;
        failures+=ids.verify_guards("T5 topk ids")+scores.verify_guards("T5 topk scores");
        if(ws.used()!=0 || ws.peak_used()>bytes) ++failures;
        std::cout << "T5 full-vocabulary topk checked\n";
    }
    bool rejected=false;
    try { (void)ops::linear_workspace_capacity_bytes(QType::T5_G128_FP16,128,5120,ops::LinearPolicy::A16Only,1,1); }
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected) ++failures;
    for (const auto [n,k] : std::array<std::pair<int,int>,3>{{{0,5120},{128,0},{128,5121}}}) {
        bool invalid=false;
        try { (void)ops::linear_workspace_capacity_bytes(QType::T5_G128_FP16,n,k,ops::LinearPolicy::AllowA8,1,1); }
        catch(const std::invalid_argument&) {invalid=true;}
        if(!invalid) ++failures;
    }
    std::cout << (failures ? "FAIL" : "OK") << " T5 A8\n";
    return failures ? 1 : 0;
}
