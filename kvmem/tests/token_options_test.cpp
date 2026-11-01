#include "kvmem/token_options.hpp"
#include <iostream>

int main() {
    int failures = 0;
    const auto check = [&](bool ok) { if (!ok) ++failures; };
    for (const uint32_t block : {32u, 64u, 128u}) {
        uint32_t b=10001, r=1001, sink=101, recent=101;
        kvmem::normalize_memory_tokens(b,r,sink,recent,block,true,false);
        check(b==9984 && r==(block==32 ? 992 : block==64 ? 960 : 896));
        check(sink==(block==32 ? 96 : block==64 ? 64 : 128));
        check(recent==(block==32 ? 96 : block==64 ? 64 : 0));
        const auto old_b=b,old_r=r,old_sink=sink,old_recent=recent;
        kvmem::normalize_memory_tokens(b,r,sink,recent,block,true,false);
        check(b==old_b && r==old_r && sink==old_sink && recent==old_recent);
    }
    uint32_t b=10001,r=1001,sink=101,recent=101;
    kvmem::normalize_memory_tokens(b,r,sink,recent,64,false,false);
    check(b==9984 && r==960 && sink==64 && recent==64);
    check(kvmem::normalize_prefill(999,r,128)==896);
    b=129; r=129; sink=0; recent=64;
    kvmem::normalize_memory_tokens(b,r,sink,recent,64,false,false);
    check(b+r==256 && sink==64);
    b=160; r=33; sink=0; recent=33;
    kvmem::normalize_memory_tokens(b,r,sink,recent,32,false,true);
    check(b==160 && r==32 && sink==0 && recent==32);
    check(kvmem::normalize_prefill(999,r,1)==32);
    for (int invalid=0;invalid<4;++invalid) {
        b=128;r=128;sink=64;recent=64;
        if(invalid==0)b=127;
        if(invalid==1)r=63;
        if(invalid==2)recent=63;
        if(invalid==3)sink=128;
        bool rejected=false;
        try { kvmem::normalize_memory_tokens(b,r,sink,recent,64,false,false); }
        catch(const std::invalid_argument&) { rejected=true; }
        check(rejected);
    }
    bool rejected=false;
    try { kvmem::normalize_prefill(1000,64,128); }
    catch(const std::invalid_argument&) { rejected=true; }
    check(rejected);
    check(kvmem::align_token_option("Budget",2147483647,64,128)==2147483584);
    std::cout << "token_options_test failures=" << failures << '\n';
    return failures ? 1 : 0;
}
