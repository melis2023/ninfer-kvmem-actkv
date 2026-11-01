#include "kvmem/page_softmax_cuda.hpp"
#include "kvmem/raw_kv_store.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main() {
    try {
        cudaDeviceProp props{};
        if(cudaGetDeviceProperties(&props,0)!=cudaSuccess) throw std::runtime_error("qualification GPU unavailable");
        std::cout<<"functional qualification: "<<props.name<<'\n';
        kvmem::PageSoftmaxCuda scorer(0);
        struct Shape { uint32_t blocks,rows,dim,qheads,kvheads; };
        for(const auto aggregation : {kvmem::GqaScoreAggregation::Mean,kvmem::GqaScoreAggregation::Max})
        for(const auto shape:{Shape{7,129,256,8,2},Shape{8192,2,64,4,1},
            Shape{8193,2,64,4,1},Shape{32769,1,64,2,1}}) {
            constexpr uint32_t layers=2;
            const auto kw=shape.kvheads*shape.dim,qw=shape.qheads*shape.dim;
            kvmem::RawKvStore keys({layers,kw,0,4});
            std::vector<float> key(kw);
            for(uint32_t l=0;l<layers;++l) for(uint32_t b=0;b<shape.blocks;++b) {
                for(uint32_t d=0;d<kw;++d) key[d]=float(int((b*13+d*7+l*11)%101)-50)/16;
                keys.write_layer_mean_sum(b*4,4,l,key.data());
            }
            std::vector<float> query(size_t(layers)*shape.rows*qw);
            for(size_t d=0;d<query.size();++d) query[d]=float(int((d*17)%127)-63)/32;
            const uint32_t counts[]{shape.rows,shape.rows};
            auto frame=kvmem::query_score_frame(uint64_t(shape.blocks)*4,4,3,1,1,{0,shape.blocks-1});
            kvmem::PageQueryView view{query.data(),counts,layers,shape.rows,shape.qheads,shape.kvheads,shape.dim};
            const auto expected=kvmem::page_softmax_scores(keys,view,frame,aggregation);
            const auto actual=scorer.score(keys,view,frame,aggregation);
            if(shape.blocks==7) {
                const auto bytes=kvmem::PageSoftmaxCuda::workspace_bytes(shape.blocks,shape.qheads,shape.kvheads,shape.dim);
                void* workspace=nullptr;
                if(cudaMalloc(&workspace,bytes)!=cudaSuccess) throw std::runtime_error("planned test workspace unavailable");
                {
                    kvmem::PageSoftmaxCuda borrowed(0,workspace,bytes,shape.blocks,shape.qheads,shape.kvheads,shape.dim);
                    const auto planned=borrowed.score(keys,view,frame,aggregation);
                    for(size_t i=0;i<planned.size();++i)
                        if(std::abs(planned[i]-expected[i])>2e-6+expected[i]*3e-5)
                            throw std::runtime_error("planned workspace differs from FP64 oracle");
                }
                cudaFree(workspace);
            }
            double worst=0;
            for(size_t b=0;b<actual.size();++b) {
                const auto error=std::abs(actual[b]-expected[b]);
                worst=std::max(worst,error);
                if(!std::isfinite(actual[b]) || error>2e-6+expected[b]*3e-5)
                    throw std::runtime_error("CUDA page softmax differs from represented-input FP64 oracle");
            }
            const auto mass=std::accumulate(actual.begin(),actual.end(),0.);
            if(std::abs(mass-shape.rows)>shape.rows*2e-5) throw std::runtime_error("CUDA softmax mass was lost");
            std::cout<<"agg="<<int(aggregation)<<" pages="<<shape.blocks<<" rows="<<shape.rows<<" D="<<shape.dim
                <<" worst_abs="<<worst<<" workspace="<<scorer.device_bytes()<<'\n';
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
