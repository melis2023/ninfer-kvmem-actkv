// The router's model lifecycle and catalog, without an Engine: a fake service records what the
// registry asks of it. Covered: resolving ids and aliases, autoload, making room under models_max
// by putting the least recently used idle model to sleep (or unloading it when it cannot sleep),
// waking a sleeping model, leases holding a model in place, idle sleep, explicit load/unload/sleep,
// the event stream, and the preset/models-directory catalog with its option precedence.

#include "serve/model_catalog.h"
#include "serve/model_registry.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer::serve;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

struct Journal {
    std::vector<std::string> calls;
    std::map<std::string, int> alive;
    // Read by the registry's idle thread: entries exist before a service is loaded.
    std::map<std::string, std::atomic<bool>> failing;
};

class FakeService final : public ModelService {
public:
    FakeService(std::string id, bool suspendable, Journal& journal)
        : id_(std::move(id)), suspendable_(suspendable), journal_(journal) {
        journal_.calls.push_back("load " + id_);
        ++journal_.alive[id_];
    }

    ~FakeService() override {
        journal_.calls.push_back("unload " + id_);
        --journal_.alive[id_];
    }

    GenerationService& service() override { throw std::logic_error("fake service"); }

    bool suspendable() const override { return suspendable_; }

    void suspend() override { journal_.calls.push_back("sleep " + id_); }

    void resume() override { journal_.calls.push_back("wake " + id_); }

    bool failed() const override {
        const auto found = journal_.failing.find(id_);
        return found != journal_.failing.end() && found->second.load();
    }

private:
    std::string id_;
    bool suspendable_;
    Journal& journal_;
};

ModelDefinition model(std::string id, std::vector<std::string> aliases = {}) {
    ModelDefinition out;
    out.id      = std::move(id);
    out.aliases = std::move(aliases);
    return out;
}

ModelServiceFactory factory(Journal& journal, std::vector<std::string> non_suspendable = {}) {
    return [&journal, non_suspendable](const ModelDefinition& definition) {
        const bool can_sleep = std::find(non_suspendable.begin(), non_suspendable.end(),
                                         definition.id) == non_suspendable.end();
        return std::make_unique<FakeService>(definition.id, can_sleep, journal);
    };
}

ModelState state_of(const ModelRegistry& registry, const std::string& id) {
    return registry.status(id)->state;
}

void lifecycle_under_one_slot() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max = 1;
    ModelRegistry registry({model("a", {"alpha"}), model("b"), model("c")}, policy,
                           factory(journal, {"c"}), true);
    expect(state_of(registry, "a") == ModelState::Unloaded, "nothing loads before it is asked");
    {
        auto lease = registry.acquire("alpha");
        expect(lease.model() == "a", "an alias resolves to its model");
        expect(state_of(registry, "a") == ModelState::Loaded, "acquire loads");
    }
    {
        auto lease = registry.acquire("b");
        expect(state_of(registry, "a") == ModelState::Sleeping, "the idle model sleeps for room");
        expect(state_of(registry, "b") == ModelState::Loaded, "the new model loads");
    }
    {
        auto lease = registry.acquire("a");
        expect(state_of(registry, "a") == ModelState::Loaded, "a sleeping model wakes");
        expect(state_of(registry, "b") == ModelState::Sleeping, "the other one sleeps");
    }
    {
        auto lease = registry.acquire("c");
        expect(state_of(registry, "a") == ModelState::Sleeping, "room again by sleep");
    }
    {
        auto lease = registry.acquire("b");
        expect(state_of(registry, "c") == ModelState::Unloaded,
               "a model that cannot sleep is unloaded for room");
        expect(journal.alive["c"] == 0, "its service is gone");
    }
    const std::vector<std::string> expected{"load a",  "sleep a", "load b",   "sleep b", "wake a",
                                            "sleep a", "load c",  "unload c", "wake b"};
    expect(journal.calls == expected, "the registry asked for exactly these transitions");
    bool unknown = false;
    try {
        (void)registry.acquire("nope");
    } catch (const ModelUnknown&) { unknown = true; }
    expect(unknown, "an unknown model is refused");
    bool not_loaded = false;
    try {
        (void)registry.acquire("c", false);
    } catch (const ModelNotLoaded&) { not_loaded = true; }
    expect(not_loaded, "autoload off refuses a model that is not loaded");
}

void leases_hold_models() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max   = 1;
    policy.room_timeout = std::chrono::seconds(5);
    ModelRegistry registry({model("a"), model("b")}, policy, factory(journal), true);
    auto held = std::make_unique<ModelRegistry::Lease>(registry.acquire("a"));
    std::atomic<bool> loaded_b{false};
    std::thread other([&] {
        auto lease = registry.acquire("b");
        loaded_b   = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    expect(!loaded_b, "a leased model is not displaced");
    held.reset();
    other.join();
    expect(loaded_b && state_of(registry, "a") == ModelState::Sleeping,
           "the waiting request proceeds once the lease ends");
}

void idle_sleep_and_events() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max = 2;
    policy.sleep_idle = std::chrono::seconds(1);
    ModelRegistry registry({model("a"), model("b")}, policy, factory(journal), true);
    std::vector<ModelEvent> events;
    std::uint64_t cursor = registry.events(0, std::chrono::milliseconds(0), events);
    { auto lease = registry.acquire("a"); }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state_of(registry, "a") != ModelState::Sleeping &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    expect(state_of(registry, "a") == ModelState::Sleeping, "an idle model goes to sleep");
    cursor = registry.events(cursor, std::chrono::milliseconds(0), events);
    std::vector<std::string> states;
    for (const ModelEvent& event : events) { states.push_back(model_state_name(event.state)); }
    const std::vector<std::string> expected{"loading", "loaded", "unloading", "sleeping"};
    expect(states == expected, "the event stream reports every transition in order");
    registry.unload("a");
    expect(state_of(registry, "a") == ModelState::Unloaded && journal.alive["a"] == 0,
           "unload destroys a sleeping model");
    registry.load("b");
    registry.sleep("b");
    expect(state_of(registry, "b") == ModelState::Sleeping, "an explicit sleep");
    registry.load("b");
    expect(state_of(registry, "b") == ModelState::Loaded, "an explicit load wakes it");
}

void observing_is_not_use() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max = 1;
    policy.sleep_idle = std::chrono::seconds(1);
    ModelRegistry registry({model("a")}, policy, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    // A monitor polling the model (GET /props, /metrics) must not keep it awake.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state_of(registry, "a") != ModelState::Sleeping &&
           std::chrono::steady_clock::now() < deadline) {
        { auto observed = registry.observe("a"); }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    expect(state_of(registry, "a") == ModelState::Sleeping,
           "an observed but otherwise idle model goes to sleep");
}

void failed_model_reloads() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max    = 1;
    journal.failing["a"] = false;
    ModelRegistry registry({model("a")}, policy, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    journal.failing["a"] = true;
    const auto deadline  = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state_of(registry, "a") != ModelState::Unloaded &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const auto status = registry.status("a");
    expect(status->state == ModelState::Unloaded && status->failed && journal.alive["a"] == 0,
           "a router unloads a model whose Engine failed and reports the failure");
    journal.failing["a"] = false;
    { auto lease = registry.acquire("a"); }
    expect(state_of(registry, "a") == ModelState::Loaded && !registry.status("a")->failed,
           "the next request loads the failed model again");
}

void single_model_mode() {
    Journal journal;
    ModelRegistry registry({model("only")}, RegistryPolicy{}, factory(journal), false);
    registry.load_startup_models();
    expect(state_of(registry, "only") == ModelState::Loaded, "single mode loads its model");
    auto lease = registry.acquire("");
    expect(lease.model() == "only", "a request naming no model gets the one model");
}

void catalog() {
    const auto preset = parse_preset_file(R"(
version = 1
; shared by every model
[*]
max-context = 8192
model-suspend = true

[small]
model = /models/small.ninfer
alias = s, tiny
kv-capacity = 4096
load-on-startup = true

[big]
artifact = "/models/big.ninfer"
model-suspend = false
)");
    const auto models = build_catalog(std::nullopt, preset, {"--kv-capacity", "16384"});
    expect(models.size() == 2, "two models from the preset");
    const CatalogModel& small = models[0].id == "small" ? models[0] : models[1];
    const CatalogModel& big   = models[0].id == "small" ? models[1] : models[0];
    expect(small.artifact == "/models/small.ninfer" && small.load_on_startup,
           "path and startup flag");
    expect(small.aliases == std::vector<std::string>({"s", "tiny"}), "aliases");
    const std::vector<std::string> small_arguments{"--max-context", "8192", "--model-suspend",
                                                   "--kv-capacity", "4096", "--kv-capacity",
                                                   "16384"};
    expect(small.arguments == small_arguments,
           "global, then model, then command line, so the command line wins");
    expect(big.artifact == "/models/big.ninfer", "quoted path");
    bool duplicate = false;
    try {
        (void)build_catalog(
            std::nullopt,
            parse_preset_file("[a]\nmodel=/x.ninfer\nalias=b\n[b]\nmodel=/y.ninfer\n"), {});
    } catch (const std::invalid_argument&) { duplicate = true; }
    expect(duplicate, "an alias may not repeat another model's id");

    const auto directory = std::filesystem::temp_directory_path() / "ninfer_catalog_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory / "nested");
    std::ofstream(directory / "flat.ninfer") << "x";
    std::ofstream(directory / "nested" / "weights.ninfer") << "x";
    std::ofstream(directory / "readme.txt") << "x";
    const auto found = discover_models(directory);
    expect(found.size() == 2 && found[0].id == "flat" && found[1].id == "nested",
           "a models directory yields its artifacts and single-artifact subdirectories");
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    try {
        lifecycle_under_one_slot();
        leases_hold_models();
        idle_sleep_and_events();
        observing_is_not_use();
        failed_model_reloads();
        single_model_mode();
        catalog();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "model registry: ok\n";
    return 0;
}
