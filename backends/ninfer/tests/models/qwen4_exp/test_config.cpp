// parse_text_config on the text config the converter writes for the Qwen3.8-Flash-Next
// checkpoint (tests/fixtures/qwen4_exp/text_config.json, which tests/convert/test_qwen4_exp.py
// holds to tools/convert/qwen4_exp.py), and its refusals.
#include "models/qwen4_exp/config.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace ninfer::models::qwen4_exp;
using ninfer::artifact::Json;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

Json fixture() {
    std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/text_config.json");
    require(file.good(), "fixture missing");
    return Json::parse(file);
}

void expect_refusal(const std::function<void(Json&)>& edit, const std::string& label) {
    Json config = fixture();
    edit(config);
    bool refused = false;
    try {
        (void)parse_text_config(config);
    } catch (const ninfer::artifact::ArtifactError&) { refused = true; }
    require(refused, "not refused: " + label);
}

int run() {
    const TextConfig config = parse_text_config(fixture());
    require(config.num_hidden_layers == 48 && config.layer_types.size() == 48, "48 blocks");
    std::size_t attention = 0;
    for (std::size_t i = 0; i < config.layer_types.size(); ++i) {
        const bool qsa = config.layer_types[i] == MixerKind::SparseAttention;
        require(qsa == (i % 4 == 3), "QSA every fourth block");
        attention += qsa;
    }
    require(attention == 12, "12 QSA blocks");
    require(config.ple_layers == std::vector<std::uint32_t>{1}, "PLE before block 1");
    require(config.eos_token_id == 248044 && !config.tie_word_embeddings, "EOS, untied head");
    require(config.indexer_block_budget() == 512, "512 selected blocks");
    require(config.ngram_heads() == 16, "16 n-gram heads");
    const NgramHashConstants constants = derive_ngram_hash_constants(config.ngram);
    require(constants.rows == 320001536ULL, "n-gram table rows");

    expect_refusal([](Json& c) { c["hidden_size"] = 2048; }, "another hidden width");
    // An expert-pruned release keeps 256 of the 512 experts; fewer than the ten a token selects,
    // or more than 512, are refused.
    Json pruned           = fixture();
    pruned["num_experts"] = 256;
    require(parse_text_config(pruned).num_experts == 256, "an expert-pruned release parses");
    expect_refusal([](Json& c) { c["num_experts"] = 9; }, "fewer experts than a token selects");
    expect_refusal([](Json& c) { c["num_experts"] = 1024; }, "more experts than implemented");
    expect_refusal([](Json& c) { c["ple_layers"] = Json::array({3}); }, "PLE on a QSA block");
    expect_refusal([](Json& c) { c.erase("hc_lowrank"); }, "a missing field");
    expect_refusal([](Json& c) { c["architectures"] = Json::array({"Qwen3_5ForCausalLM"}); },
                   "another architecture");
    expect_refusal([](Json& c) { c["eos_token_id"] = 248320; }, "EOS past the vocabulary");
    expect_refusal([](Json& c) { c["rope_parameters"]["mrope_section"] = Json::array({11, 11, 11}); },
                   "MRoPE sections past the rotary width");
    return 0;
}

} // namespace

int main() {
    try {
        const int result = run();
        std::cout << "PASS qwen4_exp config\n";
        return result;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
