#include "ninfer/engine.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
ninfer::EngineOptions options(const char* artifact) {
    ninfer::EngineOptions out;
    out.artifact_path=artifact;out.device_profile="off";out.max_context=1024;
    out.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(1024);out.prefill_chunk=128;
    out.use_cuda_graph=false;out.speculative.backend=ninfer::SpeculativeBackend::None;
    out.speculative.ngram_draft_tokens=0;out.max_concurrency=1;
    out.context_cache.device_state_slots=1;out.context_cache.host_state_slots=0;
    out.context_cache.host_kv_capacity_bytes=0;out.context_cache.max_private_continuations=1;
    out.context_cache.max_shared_prefixes=0;
    return out;
}
void save(ninfer::Engine& engine,const std::string& path) {
    ninfer::PromptInput prompt;prompt.options.enable_thinking=false;
    ninfer::ChatMessage message;message.role=ninfer::ChatRole::User;
    message.parts.push_back({.kind=ninfer::MessagePartKind::Text,.text="Return one word."});
    prompt.messages.push_back(std::move(message));
    ninfer::RequestOptions request;request.execution.requested_output_tokens=2;
    request.execution.sampling.temperature=0;request.stop.include_model_defaults=false;
    const auto result=engine.generate(engine.prepare(prompt),request);
    require(result.slot>=0,"fixture did not retain a native slot");
    require(engine.save_slot(static_cast<std::uint32_t>(result.slot),path).bytes>0,"empty snapshot");
}
void reject(ninfer::Engine& engine,const std::string& path) {
    bool refused=false;
    try {(void)engine.restore_slot(0,path);} catch(const std::invalid_argument& error) {
        refused=std::string(error.what()).find("different model")!=std::string::npos;
    }
    require(refused,"snapshot from another weight mode was accepted");
}
}
int main(int argc,char** argv) {
    if (argc!=3) {
        std::cout<<"SKIP: provide artifact and output-directory for native snapshot mode binding\n";
        return 77;
    }
    try {
        const auto directory=std::filesystem::absolute(argv[2]);std::filesystem::create_directories(directory);
        const auto base_file=(directory/"default.bin").string(),a16_file=(directory/"a16.bin").string();
        std::string base,a16;
        {ninfer::Engine engine(options(argv[1]));base=engine.load_summary().prefill_signature;save(engine,base_file);}
        {auto o=options(argv[1]);o.nvfp4_w4a16=true;ninfer::Engine engine(o);
         a16=engine.load_summary().prefill_signature;require(a16!=base,"W4A16 signature equals default");
         reject(engine,base_file);save(engine,a16_file);}
        {auto o=options(argv[1]);o.nvfp4_w4a8_int=true;ninfer::Engine engine(o);
         const auto signature=engine.load_summary().prefill_signature;
         require(signature!=base && signature!=a16,"W4A8 signature collides with another mode");
         reject(engine,base_file);reject(engine,a16_file);}
        {auto o=options(argv[1]);o.nvfp4_w4a16=true;o.nvfp4_w4a8_int=true;ninfer::Engine engine(o);
         require(engine.load_summary().prefill_signature==a16,"W4A16 precedence changes signature");
         require(engine.restore_slot(0,a16_file).tokens>0,"same effective mode restore failed");}
        std::cout<<"NVFP4 mode signatures and native snapshot isolation passed; non-NVFP4 fixture qualifies binding only\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
