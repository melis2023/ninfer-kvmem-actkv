#include "ninfer/ops/linear_add.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

int main() {
    using namespace ninfer;
    using namespace ninfer::test;
    if (cuda_unavailable()) return 77;
    int failures = 0;
    for (int k : {6144,17408}) {
        constexpr int n = 5120;
        auto host = quantized_weight::make_ptq1_patterned_weight(n, k, 1801U + k);
        auto payload = to_device(host.payload);
        auto weight = host.device_weight(payload.p);
        const std::array<int,7> rows{0, 31, 64, 127, 1023, 4096, 5119};
        auto oracle_weights = quantized_weight::materialize_rows_fp32(host, rows);
        for (auto policy : {ops::LinearPolicy::A16Only,ops::LinearPolicy::AllowA8Int}) {
            for (int t : {1,3,16,17,32,33,64,65,128,512}) {
                std::vector<float> x(std::size_t(k)*t), residual(std::size_t(n)*t);
                fill_uniform(x, 1807U+t, -0.5F,0.5F); round_to_bf16(x);
                fill_uniform(residual, 1811U+t,-0.1F,0.1F); round_to_bf16(residual);
                auto input = to_device_bf16(x), output = to_device_bf16(residual);
                Tensor tx(input.p,DType::BF16,{k,t}), tr(output.p,DType::BF16,{n,t});
                const auto capacity = ops::linear_add_workspace_capacity_bytes(weight.qtype,n,k,policy,t,t);
                DeviceArena arena(std::max<std::size_t>(256,capacity));
                ops::linear_add(tx,weight,tr,policy,arena,nullptr); cuda_synchronize();
                const auto actual_all = from_device_bf16(output, residual.size());
                std::vector<double> actual, expected;
                for (int token=0;token<t;++token) for (std::size_t r=0;r<rows.size();++r) {
                    double sum=0;
                    for(int c=0;c<k;++c) sum += double(oracle_weights[r*k+c])*x[std::size_t(token)*k+c];
                    expected.push_back(sum+residual[std::size_t(token)*n+rows[r]]);
                    actual.push_back(actual_all[std::size_t(token)*n+rows[r]]);
                }
                const ReductionCriterion criterion = policy==ops::LinearPolicy::A16Only
                    ? ReductionCriterion{0.006, 1e-4, 0.012}
                    : ReductionCriterion{0.04, 1e-4, 0.06};
                failures += verify_reduction("PTQ1 LinearAdd",actual,expected,criterion);
                if (arena.used()!=0 || arena.peak_used()>capacity) ++failures;
            }
        }
    }
    std::cout << (failures?"FAIL":"OK") << " PTQ1 LinearAdd\n";
    return failures?1:0;
}
