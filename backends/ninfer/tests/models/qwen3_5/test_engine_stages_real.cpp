// A model split into pipeline stages must generate exactly what the same model does on one device.
//
// The layers run the same kernels on the same data whichever device holds them, so greedy output is
// an exact oracle: any difference is a wrong-stage weight, a stale control tensor, a KV plane read
// from the wrong copy of the block table, a boundary transfer that lost or reordered bytes, or a
// forward pass that raced its own staging. Each configuration below is compared byte for byte with
// the single-device run.
//
// By default every stage shares device 0, so a pointer into "another stage's" memory still works and
// that cannot see a wrong-device access. What it does cover is everything else about the stage path,
// including the pinned-host protocol (forced, since same-device stages would otherwise take a
// device-to-device shortcut), CUDA graph capture across stages, and prefill that spans several
// chunks.
//
// On a machine with several GPUs, NINFER_TEST_DEVICE_IDS=0,1 puts stage i on the i-th listed device
// (wrapping around), which is the check the aliased run cannot make. Two settings adapt it to a
// pair of cards that cannot hold the model alone:
//   NINFER_TEST_SPLIT_INVARIANCE=1  compare every row with the first row instead of one device
//   NINFER_TEST_MAX_STAGES=2        skip rows with more stages than that
//
// NINFER_TEST_GRAFT=<container.bin> also loads that prompt graft and compares a grafted chat request
// across the rows. A direct_kv graft's K/V and Gated DeltaNet state are written at startup onto the
// rank that owns each layer, so a wrong shard, local layer index or block-table replica shows here.
//
// Speculative rows run for each backend the artifact carries: MTP, and the DFlash or DFlash2
// drafter, whose feature taps sit on several stages and cross to rank 0 after each pass. The uneven
// split is NINFER_TEST_STAGE_LAYERS (20,44 by default, for a 64-layer model; 8,32 suits the
// 40-layer 35B-A3B).

#include "guarded_main.h"
#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Configuration {
    std::string label;
    std::vector<int> devices;
    std::vector<std::uint32_t> stage_layers;
    bool cuda_graph   = true;
    bool force_staged = false;
    // Speculative decoding. Its output is not the same as ordinary decoding's (verification
    // evaluates several columns at once), so it is compared with the same backend on one device,
    // not with the plain reference.
    ninfer::SpeculativeBackend speculative = ninfer::SpeculativeBackend::None;
};

const char* backend_name(ninfer::SpeculativeBackend backend) {
    switch (backend) {
    case ninfer::SpeculativeBackend::None:
        return "plain";
    case ninfer::SpeculativeBackend::Mtp:
        return "MTP";
    case ninfer::SpeculativeBackend::DFlash:
        return "DFlash";
    case ninfer::SpeculativeBackend::DFlash2:
        return "DFlash2";
    }
    return "unknown";
}

// A comma-separated list of integers.
std::vector<std::uint32_t> parse_list(const std::string& text) {
    std::vector<std::uint32_t> out;
    std::istringstream list{text};
    for (std::string item; std::getline(list, item, ',');) {
        out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
    return out;
}

// Stage i of a configuration sits on the i-th id of NINFER_TEST_DEVICE_IDS, wrapping around; without
// the variable every stage is on device 0.
std::vector<int> stage_devices(const std::vector<int>& requested) {
    const char* ids = std::getenv("NINFER_TEST_DEVICE_IDS");
    if (ids == nullptr || *ids == '\0' || requested.empty()) { return requested; }
    std::vector<int> available;
    for (const std::uint32_t id : parse_list(ids)) { available.push_back(static_cast<int>(id)); }
    if (available.empty()) { return requested; }
    std::vector<int> out;
    for (std::size_t stage = 0; stage < requested.size(); ++stage) {
        out.push_back(available[stage % available.size()]);
    }
    return out;
}

bool flag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

ninfer::EngineOptions engine_options(const char* artifact, const Configuration& configuration) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.device_profile = "off";
    options.max_context   = 2048;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk = 512;
    options.kv_cache      = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = configuration.cuda_graph;
    options.speculative.backend = configuration.speculative;
    if (configuration.speculative == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens = 3;
    } else if (configuration.speculative != ninfer::SpeculativeBackend::None) {
        options.speculative.draft_tokens = 7;
    }
    options.devices               = stage_devices(configuration.devices);
    options.stage_layers          = configuration.stage_layers;
    if (const char* graft = std::getenv("NINFER_TEST_GRAFT"); graft != nullptr && *graft != '\0') {
        options.grafts.push_back(ninfer::GraftSource{.name = "g", .path = graft});
    }
    // Heterogeneous memory sizes may need a different uneven split than the default 20/44 row.
    if (!options.stage_layers.empty()) {
        if (const auto* split = std::getenv("NINFER_TEST_STAGE_LAYERS")) {
            options.stage_layers.clear();
            std::istringstream list{std::string(split)};
            for (std::string item; std::getline(list, item, ',');)
                options.stage_layers.push_back(static_cast<std::uint32_t>(std::stoul(item)));
        }
    }
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

// Token ids inside the vocabulary and not special. 1,300 of them is three prefill chunks of 512
// with a partial last one, so the stage loop runs for every chunk shape and the KV grows across them.
std::vector<ninfer::TokenId> long_prompt() {
    std::vector<ninfer::TokenId> prompt;
    prompt.push_back(248045);
    for (std::uint32_t index = 0; index < 1300; ++index) {
        prompt.push_back(static_cast<ninfer::TokenId>(1000 + (index * 37U) % 5000U));
    }
    return prompt;
}

std::vector<ninfer::TokenId> short_prompt() { return {248045, 846, 198, 5834, 248046, 198}; }

struct Outputs {
    std::vector<ninfer::TokenId> short_run;
    std::vector<ninfer::TokenId> long_run;
    // A continuation of the long prompt, which the context cache serves in part from the first run:
    // its state and KV come back from the cache, on whichever devices hold them.
    std::vector<ninfer::TokenId> continued_run;
    std::uint32_t reused_tokens = 0;
    // A chat request selecting NINFER_TEST_GRAFT; empty without it.
    std::vector<ninfer::TokenId> graft_run;
};

Outputs generate(const char* artifact, const Configuration& configuration) {
#ifdef _WIN32
    _putenv_s("NINFER_FORCE_STAGED_LINKS", configuration.force_staged ? "1" : "0");
#else
    setenv("NINFER_FORCE_STAGED_LINKS", configuration.force_staged ? "1" : "0", 1);
#endif
    std::cout << "running: " << configuration.label << std::endl;
    ninfer::Engine engine(engine_options(artifact, configuration));
    Outputs out;
    out.short_run = engine.generate(engine.prepare_tokens(short_prompt()), greedy(24)).generated_token_ids;
    out.long_run  = engine.generate(engine.prepare_tokens(long_prompt()), greedy(24, true)).generated_token_ids;

    std::vector<ninfer::TokenId> continuation = long_prompt();
    continuation.insert(continuation.end(), out.long_run.begin(), out.long_run.end());
    continuation.push_back(198);
    const ninfer::GenerationResult continued =
        engine.generate(engine.prepare_tokens(std::move(continuation)), greedy(8, true));
    out.continued_run = continued.generated_token_ids;
    out.reused_tokens = continued.reused_prompt_tokens;

    const char* graft = std::getenv("NINFER_TEST_GRAFT");
    if (graft != nullptr && *graft != '\0') {
        ninfer::PromptInput input;
        input.messages.push_back(ninfer::ChatMessage{
            .role = ninfer::ChatRole::User, .parts = {ninfer::MessagePart{.text = "Who are you?"}}});
        input.options.graft = "g";
        out.graft_run =
            engine.generate(engine.prepare(std::move(input)), greedy(24)).generated_token_ids;
    }
    std::cout << "continuation: reused=" << out.reused_tokens << " tokens=";
    for (const auto token : out.continued_run) { std::cout << token << ','; }
    std::cout << std::endl;
    return out;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    const bool split_invariance = flag("NINFER_TEST_SPLIT_INVARIANCE");
    const bool same_placement = flag("NINFER_TEST_SAME_PLACEMENT_REFERENCE");
    const char* max_stages_text = std::getenv("NINFER_TEST_MAX_STAGES");
    const std::size_t max_stages =
        max_stages_text != nullptr ? static_cast<std::size_t>(std::atoi(max_stages_text)) : 64;
    // One card cannot always hold the model. Then the first row is the reference, and what is
    // checked is that where the layers are cut does not change the output.
    const Outputs reference =
        generate(artifact, split_invariance
                               ? Configuration{.label = "two stages, graphs", .devices = {0, 0}}
                               : Configuration{.label = "single device", .devices = {}});
    if (reference.short_run.size() != 24 || reference.long_run.size() != 24 ||
        reference.continued_run.size() != 8 || reference.reused_tokens == 0) {
        std::cerr << "the single-device reference did not generate its tokens or reuse its prefix\n";
        return 1;
    }
    const char* graft = std::getenv("NINFER_TEST_GRAFT");
    if (graft != nullptr && *graft != '\0' && reference.graft_run.size() != 24) {
        std::cerr << "the reference did not generate the grafted request's tokens\n";
        return 1;
    }
    // A speculative backend needs a model that carries its weights; the rows of a backend the
    // artifact lacks are skipped.
    using ninfer::SpeculativeBackend;
    std::map<SpeculativeBackend, Outputs> references;
    references.emplace(SpeculativeBackend::None, reference);
    for (const SpeculativeBackend backend :
         {SpeculativeBackend::Mtp, SpeculativeBackend::DFlash, SpeculativeBackend::DFlash2}) {
        const std::string name = backend_name(backend);
        try {
            references.emplace(
                backend, generate(artifact, split_invariance
                                                ? Configuration{.label   = "two stages, " + name,
                                                                .devices = {0, 0},
                                                                .speculative = backend}
                                                : Configuration{.label   = "single device, " + name,
                                                                .devices = {},
                                                                .speculative = backend}));
        } catch (const std::exception& error) {
            std::cout << name << " rows skipped: " << error.what() << '\n';
        }
    }

    const char* uneven_text = std::getenv("NINFER_TEST_STAGE_LAYERS");
    const std::vector<std::uint32_t> uneven =
        parse_list(uneven_text != nullptr && *uneven_text != '\0' ? uneven_text : "20,44");
    std::vector<Configuration> configurations = {
        {.label = "two stages, graphs", .devices = {0, 0}},
        {.label = "two stages, eager", .devices = {0, 0}, .cuda_graph = false},
        {.label = "two stages, staged transport", .devices = {0, 0}, .force_staged = true},
        {.label = "three stages", .devices = {0, 0, 0}},
        // Uneven counts: the split must not change what the model computes.
        {.label = "two stages, uneven layers", .devices = {0, 0}, .stage_layers = uneven},
        {.label = "two stages, MTP", .devices = {0, 0}, .speculative = SpeculativeBackend::Mtp},
        {.label       = "three stages, MTP, eager",
         .devices     = {0, 0, 0},
         .cuda_graph  = false,
         .speculative = SpeculativeBackend::Mtp},
    };
    // The drafter's taps fall on more than one stage of these splits, so a pass sends features from
    // the later stages while stage 0's are captured on rank 0 directly.
    for (const SpeculativeBackend backend :
         {SpeculativeBackend::DFlash, SpeculativeBackend::DFlash2}) {
        const std::string name = backend_name(backend);
        configurations.push_back(
            {.label = "two stages, " + name, .devices = {0, 0}, .speculative = backend});
        configurations.push_back({.label        = "two stages, " + name + ", staged transport",
                                  .devices      = {0, 0},
                                  .force_staged = true,
                                  .speculative  = backend});
        configurations.push_back({.label       = "three stages, " + name + ", eager",
                                  .devices     = {0, 0, 0},
                                  .cuda_graph  = false,
                                  .speculative = backend});
        configurations.push_back({.label        = "two stages, " + name + ", uneven layers",
                                  .devices      = {0, 0},
                                  .stage_layers = uneven,
                                  .speculative  = backend});
    }

    int failures = 0;
    for (const Configuration& configuration : configurations) {
        const auto found = references.find(configuration.speculative);
        if (found == references.end()) { continue; }
        if (configuration.devices.size() > max_stages) { continue; }
        const Outputs& expected = found->second;
        const Outputs outputs   = generate(artifact, configuration);
        const bool short_ok     = outputs.short_run == expected.short_run;
        const bool long_ok      = outputs.long_run == expected.long_run;
        const bool cache_ok     = outputs.continued_run == expected.continued_run &&
                              outputs.reused_tokens == expected.reused_tokens;
        const bool graft_ok     = outputs.graft_run == expected.graft_run;
        std::cout << configuration.label << ": short " << (short_ok ? "identical" : "DIFFERS")
                  << ", long " << (long_ok ? "identical" : "DIFFERS") << ", prefix reuse ("
                  << outputs.reused_tokens << " tokens) " << (cache_ok ? "identical" : "DIFFERS");
        if (!expected.graft_run.empty()) {
            std::cout << ", graft " << (graft_ok ? "identical" : "DIFFERS");
        }
        std::cout << '\n';
        if (!graft_ok && short_ok && long_ok && cache_ok) {
            ++failures;
            std::size_t first = 0;
            while (first < expected.graft_run.size() && first < outputs.graft_run.size() &&
                   expected.graft_run[first] == outputs.graft_run[first]) {
                ++first;
            }
            std::cerr << "  graft: first difference at token " << first << '\n';
        } else if (!short_ok || !long_ok || !cache_ok) {
            ++failures;
            const auto& want = !short_ok ? expected.short_run : !long_ok ? expected.long_run : expected.continued_run;
            const auto& got  = !short_ok ? outputs.short_run : !long_ok ? outputs.long_run : outputs.continued_run;
            std::size_t first = 0;
            while (first < want.size() && first < got.size() && want[first] == got[first]) { ++first; }
            std::cerr << "  first difference at token " << first << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
