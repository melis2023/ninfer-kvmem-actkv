#include "ninfer_build_id.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/model_catalog.h"
#include "serve/model_registry.h"
#include "serve/operational_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"
#include "serve/stop_control.h"

#include <spdlog/logger.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <typeinfo>
#include <utility>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <pthread.h>
#endif

namespace {

using ninfer::serve::StopControl;
using ninfer::serve::StopEvent;

// Routes console and signal events to the StopControl and withdraws an unconfirmed Ctrl+C once
// its window passes. Events arrive on a thread of their own (the Windows console-control thread or
// the POSIX signal thread), never inside a signal handler, so handling may lock and log.
class StopEvents {
public:
    explicit StopEvents(StopControl& control) : control_(control) {
        timer_ = std::thread([this] { run_timer(); });
        std::lock_guard lock(target_mutex());
        target() = this;
    }

    ~StopEvents() {
        {
            std::lock_guard lock(target_mutex());
            target() = nullptr;
        }
        {
            std::lock_guard lock(mutex_);
            done_ = true;
        }
        wake_.notify_all();
        timer_.join();
        control_.finish();
    }

    StopEvents(const StopEvents&)            = delete;
    StopEvents& operator=(const StopEvents&) = delete;

    // Returns false when nothing handles `event`: the caller then applies its default action.
    static bool dispatch(StopEvent event) {
        std::lock_guard lock(target_mutex());
        StopEvents* const events = target();
        if (events == nullptr) { return false; }
        const bool handled = events->control_.handle(event, StopControl::Clock::now());
        {
            std::lock_guard timer_lock(events->mutex_);
            events->changed_ = true;
        }
        events->wake_.notify_all();
        return handled;
    }

private:
    // Never destroyed: an event may still arrive while the process exits.
    static std::mutex& target_mutex() {
        static auto* const mutex = new std::mutex();
        return *mutex;
    }

    static StopEvents*& target() {
        static StopEvents* events = nullptr;
        return events;
    }

    void run_timer() {
        std::unique_lock lock(mutex_);
        while (!done_) {
            changed_ = false;
            lock.unlock();
            const std::optional<StopControl::Clock::time_point> deadline =
                control_.expire(StopControl::Clock::now());
            lock.lock();
            const auto woken = [this] { return done_ || changed_; };
            if (deadline) {
                // expire() withdraws only after the deadline, so wake just past it.
                wake_.wait_until(lock, *deadline + std::chrono::milliseconds(1), woken);
            } else {
                wake_.wait(lock, woken);
            }
        }
    }

    StopControl& control_;
    std::mutex mutex_;
    std::condition_variable wake_;
    bool done_    = false;
    bool changed_ = false;
    std::thread timer_;
};

class ServingScope {
public:
    ServingScope(StopControl& control, ninfer::serve::HttpServer& server) : control_(control) {
        control_.serve([&server] { server.stop(); });
    }
    ~ServingScope() { control_.end_serving(); }
    ServingScope(const ServingScope&)            = delete;
    ServingScope& operator=(const ServingScope&) = delete;

private:
    StopControl& control_;
};

// An exception that escapes a request boundary ends the process through std::terminate, and the
// default handler's message is the only record of which exception it was. Under a container this
// process is pid 1: the kernel discards the SIGABRT that abort() raises against itself, glibc falls
// through to its abort instruction, and all the kernel reports is a protection fault inside libc.
[[noreturn]] void log_terminate() {
    std::string detail = "terminate called with no active exception";
    if (std::current_exception() != nullptr) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& error) {
            detail = std::string("terminate called after throwing ") + typeid(error).name() + ": " +
                     error.what();
        } catch (...) { detail = "terminate called after throwing a non-std exception"; }
    }
    std::fprintf(stderr, "ninfer-serve: %s\n", detail.c_str());
    std::fflush(stderr);
    std::abort();
}

#ifdef _WIN32
// A Ctrl+C with console text selected copies it and never reaches the process. Ctrl+Break is also
// the one graceful stop a parent process can request (GenerateConsoleCtrlEvent), since Windows
// cannot deliver SIGTERM. Closing the console window terminates the process as soon as this
// handler returns, so the close blocks here while the stop runs; Windows ends the process anyway
// about 5 seconds after the close, and an unfinished prefix-cache save leaves the previous file in
// place.
BOOL WINAPI handle_console_event(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
        return StopEvents::dispatch(StopEvent::Interrupt) ? TRUE : FALSE;
    case CTRL_BREAK_EVENT:
        return StopEvents::dispatch(StopEvent::Terminate) ? TRUE : FALSE;
    case CTRL_CLOSE_EVENT:
        if (!StopEvents::dispatch(StopEvent::Terminate)) { return FALSE; }
        Sleep(INFINITE);
        return TRUE;
    default:
        return FALSE;
    }
}

void install_stop_handlers() { SetConsoleCtrlHandler(handle_console_event, TRUE); }
#else
// SIGINT and SIGTERM are blocked before any other thread starts, so every thread inherits the
// mask and only this thread receives them, outside signal-handler context.
void install_stop_handlers() {
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);
    std::thread([signals] {
        for (;;) {
            int signal = 0;
            if (sigwait(&signals, &signal) != 0) { continue; }
            const StopEvent event =
                signal == SIGINT ? StopEvent::Interrupt : StopEvent::Terminate;
            if (StopEvents::dispatch(event)) { continue; }
            // Nothing is served yet or any more: the default action ends the process.
            std::signal(signal, SIG_DFL);
            sigset_t only;
            sigemptyset(&only);
            sigaddset(&only, signal);
            pthread_sigmask(SIG_UNBLOCK, &only, nullptr);
            std::raise(signal);
        }
    }).detach();
}
#endif

// Identity of this exact binary for the persisted prefix cache: the build id alone repeats for
// every uncommitted build, so the executable's size and modification time are included and any
// rebuild invalidates a saved cache whose bytes it may compute differently.
std::string binary_identity(const char* argv0) {
    std::string out;
#ifdef NINFER_BUILD_ID
    out = NINFER_BUILD_ID;
#endif
    std::error_code error;
#ifdef _WIN32
    // argv[0] is whatever the shell typed, which for a PATH launch is not a path to this file.
    (void)argv0;
    std::wstring module(MAX_PATH, L'\0');
    DWORD length = 0;
    while ((length = GetModuleFileNameW(nullptr, module.data(),
                                        static_cast<DWORD>(module.size()))) == module.size()) {
        module.resize(module.size() * 2);
    }
    module.resize(length);
    const std::filesystem::path self(module);
#else
    std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) { self = std::filesystem::absolute(argv0, error); }
#endif
    const auto size = std::filesystem::file_size(self, error);
    if (!error) { out += ";size=" + std::to_string(size); }
    const auto time = std::filesystem::last_write_time(self, error);
    if (!error) {
        out += ";mtime=" + std::to_string(static_cast<long long>(time.time_since_epoch().count()));
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::set_terminate(log_terminate);
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }
    // --log-colours on colours the statistics tokens as well as the levels; off keeps the log plain.
    ninfer::serve::set_operational_log_colours(options.log_colours.value_or(false));
    if (!options.context_cache.hybrid.persistent_file.empty()) {
        options.context_cache.hybrid.persistent_identity = binary_identity(argv[0]);
    }
    install_stop_handlers();

    ninfer::product::LoggingOptions logging_options;
    logging_options.logger_name  = "ninfer-serve";
    logging_options.level        = options.log_level;
    logging_options.presentation = ninfer::product::LogPresentation::Service;
    if (options.log_colours) {
        logging_options.color = *options.log_colours ? ninfer::product::LogColorMode::Always
                                                     : ninfer::product::LogColorMode::Never;
    }
    ninfer::product::LoggingRuntime logging(logging_options);
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    if (options.requested_ngram_draft_tokens) {
        logger->warn("ngram draft tokens reduced from {} to {} for max concurrency {}",
                     *options.requested_ngram_draft_tokens, options.speculative.ngram_draft_tokens,
                     options.max_concurrency);
    }
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    // Stop prompts use the transient bottom line beneath the statistics panel; output without one
    // (redirected, or not a terminal) logs them instead.
    const std::shared_ptr<ninfer::product::TerminalProgress> console_line =
        logging.terminal_progress();
    // Shares its state with the Engine's copy of the options, and outlives the Engine.
    const ninfer::PrefixCacheSaveControl save_control =
        options.context_cache.hybrid.persistent_save;
    StopControl stop_control(
        !options.context_cache.hybrid.persistent_file.empty(),
        {.show =
             [&](const ninfer::serve::StopConsoleLine& line) {
                 if (console_line->enabled()) {
                     if (line.text.empty()) {
                         console_line->clear();
                     } else {
                         console_line->update(line.text);
                     }
                 } else if (line.prompt) {
                     operational_log.write({.severity = ninfer::serve::OperationalSeverity::Warning,
                                            .message  = line.text});
                 }
             },
         .record = [&](const ninfer::serve::OperationalRecord& record) {
             operational_log.write(record);
         },
         // The writer checks between slabs of a few MiB, so it lets go within milliseconds unless
         // the disk stalls.
         .abandon_save = [save_control] { return save_control.abandon(std::chrono::seconds(2)); },
         .exit_now =
             [&] {
                 logging.flush();
                 std::_Exit(130);
             }});
    const StopEvents stop_events(stop_control);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger, logging.terminal_panel());
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        // Answer 503 from here on rather than leaving the accepted connection silent. The socket
        // has been listenable since bind() either way; the difference is whether a caller arriving
        // during the ten seconds of weight loading gets a documented "still loading" or a hang.
        server.start_serving_during_startup();
        logger->info("build {}", NINFER_BUILD_ID);

        if (!options.slot_save_path.empty()) {
            std::error_code directory_error;
            std::filesystem::create_directories(options.slot_save_path, directory_error);
            if (!std::filesystem::is_directory(options.slot_save_path)) {
                operational_log.server_failure(
                    false, "--slot-save-path is not a usable directory: " +
                               options.slot_save_path.string());
                return 1;
            }
        }
        // One model's service: its Engine loaded and warmed up. Single-model mode builds it before
        // the registry exists, to learn the model's public id; router mode builds each on demand.
        const auto make_service = [&](const ninfer::serve::ServeOptions& model_options) {
            if (options.router() && model_options.kvmem.selected_tokens != 0) {
                throw std::invalid_argument("KVMem is not supported in model router mode");
            }
            auto service = std::make_unique<ninfer::serve::GenerationService>(
                model_options, startup_log.observer(),
                ninfer::product::engine_diagnostic_observer(logger),
                [&operational_log](const ninfer::SlotAutoSaveEvent& event) {
                    operational_log.slot_auto_save(event);
                });
            startup_log.engine_ready(service->load_summary());
            operational_log.engine_capacity(*service);
            using Clock                            = std::chrono::steady_clock;
            const Clock::time_point warmup_started = Clock::now();
            operational_log.warmup_started();
            try {
                service->warmup();
            } catch (const std::exception& exception) {
                const double seconds =
                    std::chrono::duration<double>(Clock::now() - warmup_started).count();
                operational_log.warmup_failure(seconds, exception.what());
                throw;
            }
            operational_log.warmup_complete(
                std::chrono::duration<double>(Clock::now() - warmup_started).count());
            return service;
        };

        std::vector<ninfer::serve::ModelDefinition> definitions;
        std::unique_ptr<ninfer::serve::GenerationService> first;
        if (!options.router()) {
            try {
                first = make_service(options);
            } catch (const std::exception& exception) {
                operational_log.server_failure(false, exception.what());
                return 1;
            }
            definitions.push_back(
                ninfer::serve::ModelDefinition{.id = ninfer::serve::resolve_public_model_id(
                                                   options, first->load_summary().model_name),
                                               .options         = options,
                                               .arguments       = options.model_arguments,
                                               .load_on_startup = true});
        } else {
            std::optional<ninfer::serve::PresetFile> preset;
            if (options.models_preset) {
                std::ifstream file(*options.models_preset);
                if (!file) {
                    operational_log.server_failure(false, "cannot read --models-preset " +
                                                              options.models_preset->string());
                    return 1;
                }
                std::stringstream text;
                text << file.rdbuf();
                preset = ninfer::serve::parse_preset_file(text.str());
            }
            for (ninfer::serve::CatalogModel& model : ninfer::serve::build_catalog(
                     options.models_dir, preset, options.model_arguments)) {
                std::vector<std::string> words{argv[0], model.artifact.string()};
                words.insert(words.end(), model.arguments.begin(), model.arguments.end());
                std::vector<char*> model_argv;
                for (std::string& word : words) { model_argv.push_back(word.data()); }
                ninfer::serve::ServeOptions model_options = ninfer::serve::parse_serve_options(
                    static_cast<int>(model_argv.size()), model_argv.data());
                // The router owns the address, the key and the logs; a model never binds its own.
                model_options.host              = options.host;
                model_options.port              = options.port;
                model_options.api_key           = options.api_key;
                model_options.model_id_override = model.id;
                definitions.push_back(
                    ninfer::serve::ModelDefinition{.id              = model.id,
                                                   .aliases         = model.aliases,
                                                   .options         = std::move(model_options),
                                                   .arguments       = model.arguments,
                                                   .load_on_startup = model.load_on_startup});
            }
        }
        ninfer::serve::ModelRegistry registry(
            std::move(definitions),
            ninfer::serve::RegistryPolicy{
                .models_max  = options.models_max,
                .autoload    = options.models_autoload,
                .sleep_idle  = std::chrono::seconds(options.sleep_idle_seconds),
                .unload_idle = std::chrono::seconds(options.unload_idle_seconds)},
            [&](const ninfer::serve::ModelDefinition& definition)
                -> std::unique_ptr<ninfer::serve::ModelService> {
                if (first) {
                    return std::make_unique<ninfer::serve::EngineModelService>(std::move(first));
                }
                return std::make_unique<ninfer::serve::EngineModelService>(
                    make_service(definition.options));
            },
            options.router());
        try {
            registry.load_startup_models();
        } catch (const std::exception& exception) {
            operational_log.server_failure(false, exception.what());
            return 1;
        }
        server.attach(registry);
        const ServingScope serving_scope(stop_control, server);

        serving = true;
        operational_log.server_ready(options.host, options.port,
                                     options.router() ? std::string("router")
                                                      : server.public_model_id(),
                                     !options.api_key.empty());
        operational_log.server_urls(options.host, options.port, server.webui_enabled());

        const bool ok = server.listen();
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        // The Engine never recovers from an Engine-wide failure: exit with status 2, distinct
        // from the startup failures, so a supervisor reloads the model instead of keeping a dead
        // endpoint up. The engine watch already logged the failure.
        if (server.engine_failed()) { return 2; }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
