#pragma once

// The models one server serves and their lifecycle: unloaded, loading, loaded (resident), sleeping
// (suspended: the Engine and its host caches are kept, its device memory is given back), and
// unloading. A request leases a loaded model for as long as it runs; a model with leases is never
// put to sleep or unloaded under it.
//
// A request for a model that is not loaded loads it, or wakes it, when autoload allows. Making room
// follows the configured bounds: at most `models_max` models are loaded at once, and the least
// recently used loaded model without leases makes way -- put to sleep when it was started with
// model suspend (it then wakes in seconds, with its retained conversations), unloaded otherwise.
// Idle models go to sleep after `sleep_idle` and are unloaded after `unload_idle`. In router mode a
// model whose Engine failed Engine-wide is unloaded once its requests are gone, so the next request
// for it loads it again.

#include "serve/generation_service.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace ninfer::serve {

enum class ModelState : std::uint8_t {
    Unloaded,
    Loading,
    Loaded,
    Sleeping,
    Unloading,
};

[[nodiscard]] const char* model_state_name(ModelState state) noexcept;

struct ModelDefinition {
    std::string id;
    std::vector<std::string> aliases;
    ServeOptions options;
    // The serve arguments the model was configured with, reported by GET /models.
    std::vector<std::string> arguments;
    bool load_on_startup = false;
};

struct ModelStatusSnapshot {
    std::string id;
    std::vector<std::string> aliases;
    ModelState state = ModelState::Unloaded;
    std::vector<std::string> arguments;
    // The last load or wake failed; cleared by the next success.
    bool failed = false;
    std::string last_error;
    std::uint32_t leases = 0;
    std::chrono::steady_clock::time_point last_used{};
    bool held = false;
};

struct RegistryPolicy {
    // Loaded models at once; zero means unbounded.
    std::uint32_t models_max = 1;
    bool autoload            = true;
    // Zero disables each idle transition.
    std::chrono::seconds sleep_idle{0};
    std::chrono::seconds unload_idle{0};
    // How long a request waits for a busy model to make room before it fails.
    std::chrono::seconds room_timeout{600};
};

// A service the registry starts for a model, and how it is driven. Production builds a
// GenerationService and warms it up; tests substitute a fake.
class ModelService {
public:
    virtual ~ModelService()                            = default;
    [[nodiscard]] virtual GenerationService& service() = 0;
    [[nodiscard]] virtual bool suspendable() const     = 0;
    virtual void suspend()                             = 0;
    virtual void resume()                              = 0;

    // The model's Engine failed Engine-wide and never recovers: only a reload serves it again.
    [[nodiscard]] virtual bool failed() const { return false; }
};

using ModelServiceFactory =
    std::function<std::unique_ptr<ModelService>(const ModelDefinition& definition)>;

// The production service: one Engine behind a GenerationService. Sleep is the Engine's suspend,
// which needs the model started with --model-suspend.
class EngineModelService final : public ModelService {
public:
    explicit EngineModelService(std::unique_ptr<GenerationService> service)
        : service_(std::move(service)) {}

    [[nodiscard]] GenerationService& service() override { return *service_; }

    [[nodiscard]] bool suspendable() const override {
        return service_->engine_options().suspend.enabled;
    }

    void suspend() override { (void)service_->suspend(true); }

    void resume() override { (void)service_->resume(); }

    [[nodiscard]] bool failed() const override { return service_->has_failed(); }

private:
    std::unique_ptr<GenerationService> service_;
};

// One status change, for GET /models/sse.
struct ModelEvent {
    std::string model;
    ModelState state = ModelState::Unloaded;
    bool failed      = false;
    std::string error;
};

class ModelRegistry {
public:
    // `router` is router mode (several models, GET /models lists them all); single-model mode has
    // exactly one model, loaded at startup and never unloaded by policy.
    ModelRegistry(std::vector<ModelDefinition> models, RegistryPolicy policy,
                  ModelServiceFactory factory, bool router);
    ~ModelRegistry();

    ModelRegistry(const ModelRegistry&)            = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    class Lease {
    public:
        Lease() noexcept = default;
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;

        [[nodiscard]] GenerationService& service() const;
        [[nodiscard]] const std::string& model() const;

        [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

    private:
        friend class ModelRegistry;
        Lease(ModelRegistry& owner, std::size_t index, bool use) noexcept;
        ModelRegistry* owner_ = nullptr;
        std::size_t index_    = 0;
        // A request's lease counts as use of the model; an observer's does not, so polling a
        // model's state never keeps it from going idle.
        bool use_ = true;
    };

    // Resolves `model` (an id or alias; empty names the default model) and returns it loaded and
    // leased, loading or waking it -- and making room -- when `autoload` allows. Throws
    // ModelUnknown for a name that matches nothing, ModelNotLoaded when it is not loaded and may
    // not be, and ModelLoadFailed when loading or waking fails.
    Lease acquire(std::string_view model, std::optional<bool> autoload = std::nullopt);

    // A loaded or sleeping model leased without waking it, to read its state (statistics, slots,
    // residency); nullopt when it is not loaded. Throws ModelUnknown for an unknown name.
    [[nodiscard]] std::optional<Lease> observe(std::string_view model);

    // Explicit lifecycle (POST /models/load, /models/unload). Load wakes a sleeping model. Unload
    // waits for the model's leases to end.
    void load(std::string_view model);
    void unload(std::string_view model);
    // Puts a loaded model to sleep (requires model suspend); a sleeping model stays as it is. A
    // held model is not woken by requests, which fail until load() wakes it. Without `wait` a
    // model in use refuses with ModelBusy instead of waiting for its requests to end.
    void sleep(std::string_view model, bool hold = false, bool wait = true);

    [[nodiscard]] std::vector<ModelStatusSnapshot> status() const;
    [[nodiscard]] std::optional<ModelStatusSnapshot> status(std::string_view model) const;
    // The id `model` resolves to, or nullopt.
    [[nodiscard]] std::optional<std::string> resolve(std::string_view model) const;

    [[nodiscard]] std::size_t size() const noexcept { return models_.size(); }

    [[nodiscard]] bool router() const noexcept { return router_; }

    [[nodiscard]] const RegistryPolicy& policy() const noexcept { return policy_; }

    // Events since `cursor` (an opaque position; 0 starts from the oldest kept), waiting up to
    // `wait` for one when there is none. Returns the new cursor.
    std::uint64_t events(std::uint64_t cursor, std::chrono::milliseconds wait,
                         std::vector<ModelEvent>& out);

    // Every loaded or sleeping model's service, for operations that address all of them (stop).
    void for_each_service(const std::function<void(const std::string&, GenerationService&)>& visit);

    // Loads every model marked load_on_startup (in single-model mode, the model).
    void load_startup_models();
    void stop() noexcept;

private:
    struct Slot {
        ModelDefinition definition;
        ModelState state = ModelState::Unloaded;
        std::unique_ptr<ModelService> service;
        std::uint32_t leases = 0;
        std::chrono::steady_clock::time_point last_used{};
        bool failed = false;
        std::string last_error;
        bool held = false;
    };

    [[nodiscard]] ModelStatusSnapshot snapshot_locked(const Slot& slot) const;
    [[nodiscard]] std::optional<std::size_t> find_locked(std::string_view model) const;
    [[nodiscard]] std::size_t require_locked(std::string_view model) const;
    // Brings `index` to Loaded, releasing the lock while the service loads or wakes.
    void make_loaded_locked(std::unique_lock<std::mutex>& lock, std::size_t index);
    // Frees one loaded slot other than `keep`, waiting for one to become idle; false on timeout.
    bool make_room_locked(std::unique_lock<std::mutex>& lock, std::size_t keep,
                          std::chrono::steady_clock::time_point deadline);
    void put_to_sleep_locked(std::unique_lock<std::mutex>& lock, std::size_t index);
    void unload_locked(std::unique_lock<std::mutex>& lock, std::size_t index);
    void publish_locked(std::size_t index);
    void release(std::size_t index, bool use) noexcept;
    void idle_loop();
    [[nodiscard]] std::size_t loaded_count_locked() const;

    const RegistryPolicy policy_;
    const ModelServiceFactory factory_;
    const bool router_;
    std::vector<Slot> models_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<ModelEvent> events_;
    std::uint64_t first_event_ = 1;
    bool stopping_             = false;
    std::thread idle_thread_;
};

struct ModelUnknown : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

struct ModelNotLoaded : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ModelLoadFailed : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ModelBusy : std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace ninfer::serve
