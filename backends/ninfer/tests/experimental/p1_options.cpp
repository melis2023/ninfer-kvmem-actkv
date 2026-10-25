#include "runtime/engine/model_instance.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    ninfer::EngineOptions base;
    base.max_context = 2048;
    base.kvmem = {.selected_tokens = 128, .reserve_tokens = 128,
                 .host_bytes = 1024ULL * 1024ULL * 1024ULL};
    base.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
    base.prefill_chunk = 128;
    base.context_cache.enabled = false;
    try {
        const auto valid = ninfer::runtime::normalize_engine_options(base);
        for (const bool suspend : {false, true}) {
            auto bad = base;
            if (suspend) bad.suspend.enabled = true;
            else bad.grafts.push_back({.name = "unsupported", .path = "graft.bin"});
            bool rejected = false;
            try { (void)ninfer::runtime::normalize_engine_options(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("KVMem allowed an unsupported new state owner");
        }
        if (valid.kvmem.device_tokens() != 256 || valid.max_context != 2048 ||
            valid.kvmem.host_bytes != (1ULL << 30) || valid.kvmem.sink_blocks() != 1 ||
            valid.kvmem.recent_blocks() != 1) {
            throw std::runtime_error("normalization changed logical/physical limits");
        }
        for (const auto recent : {0U, 1U, 63U}) {
            auto bad = base;
            bad.kvmem.recent_tokens = recent;
            bool rejected = false;
            try { (void)ninfer::runtime::normalize_engine_options(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Engine allowed an unprotected partial append page");
        }
        for (const auto recent : {64U, 65U, 127U}) {
            auto kept = base;
            kept.kvmem.recent_tokens = recent;
            if (ninfer::runtime::normalize_engine_options(kept).kvmem.recent_blocks() != 1)
                throw std::runtime_error("Engine changed the rounded recent page allocation");
        }
        for (const bool all_sink : {false, true}) {
            auto bad = base;
            if (all_sink) bad.kvmem.sink_tokens = 128;
            else bad.kvmem.recent_tokens = 128;
            bool rejected = false;
            try { (void)ninfer::runtime::normalize_engine_options(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Engine allowed sink plus recent larger than B");
        }
        for (const bool mtp_enabled : {false, true}) {
            for (const bool image_enabled : {false, true}) {
                for (const auto lanes : {1U, 8U}) {
                    for (const auto lookup : {1U, 4U, 64U}) {
                        auto legacy = base;
                        legacy.enable_vision = image_enabled;
                        legacy.vision_residency = ninfer::VisionResidency::Cpu;
                        legacy.max_concurrency = lanes;
                        legacy.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
                        legacy.speculative.lookup_ngram = lookup;
                        if (mtp_enabled) {
                            legacy.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                            legacy.speculative.draft_tokens = 3;
                            legacy.speculative.ngram_draft_tokens = 15;
                        }
                        bool disabled = false;
                        try { (void)ninfer::runtime::normalize_engine_options(legacy); }
                        catch (const std::invalid_argument& error) {
                            const std::string message = error.what();
                            disabled = message.find("disables legacy --lookup-ngram") != std::string::npos &&
                                message.find("--ngram-draft-tokens") != std::string::npos;
                        }
                        if (!disabled) throw std::runtime_error("Engine allowed disabled legacy lookup drafting");
                    }
                }
            }
        }
        for (const auto residency : {ninfer::VisionResidency::Resident, ninfer::VisionResidency::Cpu}) {
            auto pipeline = base;
            pipeline.devices = {1, 0};
            pipeline.stage_layers = {40, 24};
            pipeline.enable_vision = true;
            pipeline.vision_residency = residency;
            pipeline.max_concurrency = 2;
            pipeline.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512);
            pipeline.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            pipeline.speculative.draft_tokens = 3;
            const auto accepted = ninfer::runtime::normalize_engine_options(pipeline);
            if (accepted.devices != pipeline.devices || accepted.stage_layers != pipeline.stage_layers ||
                accepted.kvmem.host_bytes != base.kvmem.host_bytes ||
                accepted.kv_capacity.explicit_tokens != 512 || accepted.vision_residency != residency)
                throw std::runtime_error("pipeline changed rank order, shared Host budget or lane windows");
        }
        for (const auto drafts : {1U, 4U, 5U, 15U}) {
            auto mtp = base;
            mtp.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            mtp.speculative.draft_tokens = drafts;
            const auto accepted = ninfer::runtime::normalize_engine_options(mtp);
            if (accepted.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
                accepted.speculative.draft_tokens != drafts) {
                throw std::runtime_error("KVMem normalization silently disabled MTP");
            }
        }
        for (const auto width : {15U, 31U, 63U}) {
            auto copy = base;
            copy.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            copy.speculative.draft_tokens = 3;
            copy.speculative.ngram_draft_tokens = width;
            const auto accepted = ninfer::runtime::normalize_engine_options(copy);
            if (accepted.speculative.ngram_draft_tokens != width)
                throw std::runtime_error("KVMem silently disabled ngram");
        }
        for (const auto concurrency : {2U, 4U, 5U, 8U}) {
            auto concurrent = base;
            concurrent.max_concurrency = concurrency;
            concurrent.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * concurrency);
            const auto accepted = ninfer::runtime::normalize_engine_options(concurrent);
            if (accepted.max_concurrency != concurrency || accepted.kvmem.device_tokens() != 256) {
                throw std::runtime_error("KVMem normalization changed per-request working capacity");
            }
        }
        auto vision = base;
        vision.enable_vision = true;
        vision.vision_residency = ninfer::VisionResidency::Resident;
        (void)ninfer::runtime::normalize_engine_options(vision);
        auto rotated = base;
        rotated.kv_cache = ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
        (void)ninfer::runtime::normalize_engine_options(rotated);
        for (const bool memory_enabled : {false, true}) {
            for (const unsigned disk_config : {1U, 2U, 3U}) {
                auto disk = memory_enabled ? base : ninfer::EngineOptions{};
                disk.device_profile = "off";
                if (disk_config & 1U) disk.kvmem.disk_path = "snapshot-test";
                if (disk_config & 2U) disk.kvmem.disk_bytes = 1ULL << 30;
                bool rejected = false;
                try { (void)ninfer::runtime::normalize_engine_options(disk); }
                catch (const std::invalid_argument&) {
                    rejected = true;
                }
                const bool expected = memory_enabled && disk_config == 3U;
                if (rejected == expected)
                    throw std::runtime_error("Engine accepted an incomplete tiering configuration or rejected paired H/D storage");
            }
        }
        for (const auto lanes : {1U, 2U, 8U}) {
            for (const bool fast : {false, true}) {
                auto tiered = base;
                tiered.device_profile = fast ? "off" : "auto";
                tiered.kvmem.disk_path = "paired-tiering";
                tiered.kvmem.disk_bytes = 1ULL << 30;
                tiered.max_concurrency = lanes;
                tiered.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
                tiered.kv_cache = ninfer::KvCacheStorage::Int8Group64;
                tiered.fast_prefill_kernel = fast;
                const auto accepted = ninfer::runtime::normalize_engine_options(tiered);
                if (accepted.max_concurrency != lanes || accepted.fast_prefill_kernel != fast ||
                    accepted.kvmem.disk_path != tiered.kvmem.disk_path ||
                    accepted.kvmem.disk_bytes != tiered.kvmem.disk_bytes)
                    throw std::runtime_error("Tiering changed concurrency, fast prefill or shared disk quota");
            }
        }
        for (const auto storage : {ninfer::KvCacheStorage::Nvfp4Group16,
                                   ninfer::KvCacheStorage::Fp8KeyNvfp4Value}) {
            for (const auto drafts : {0U, 1U, 2U, 3U, 4U}) {
                auto quantized = base;
                quantized.kv_cache = storage;
                if (drafts) {
                    quantized.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                    quantized.speculative.draft_tokens = drafts;
                }
                for (const auto lanes : {1U, 2U, 3U, 4U}) {
                    auto concurrent = quantized;
                    concurrent.max_concurrency = lanes;
                    concurrent.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
                    const auto normalized = ninfer::runtime::normalize_engine_options(concurrent);
                    if (normalized.kv_cache != storage ||
                        normalized.max_concurrency != lanes || normalized.kvmem.device_tokens() != 256 ||
                        normalized.speculative.backend != quantized.speculative.backend ||
                        (drafts && normalized.speculative.draft_tokens != drafts) ||
                        normalized.speculative.ngram_draft_tokens != 0 || normalized.speculative.lookup_ngram != 0) {
                        throw std::runtime_error("KVMem NVFP4/K8V4 normalization changed the requested execution");
                    }
                }
                auto image = quantized;
                image.enable_vision = true;
                image.vision_residency = ninfer::VisionResidency::Resident;
                (void)ninfer::runtime::normalize_engine_options(image);
            }
        }
        for (const auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
                ninfer::KvCacheStorage::Nvfp4Group16, ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                ninfer::KvCacheStorage::Fp8E4M3Row256, ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value,
                ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8, ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value}) {
            auto extended = base;
            extended.kv_cache = storage;
            extended.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            extended.speculative.draft_tokens = 15;
            extended.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
            (void)ninfer::runtime::normalize_engine_options(extended);
            extended.speculative.mtp_policy = ninfer::MtpDraftPolicy::Fixed;
            extended.max_concurrency = 4;
            extended.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
            (void)ninfer::runtime::normalize_engine_options(extended);
            extended.max_concurrency = 1;
            extended.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
            extended.enable_vision = true;
            extended.vision_residency = ninfer::VisionResidency::Cpu;
            (void)ninfer::runtime::normalize_engine_options(extended);
            extended.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
            (void)ninfer::runtime::normalize_engine_options(extended);
            extended.vision_residency = ninfer::VisionResidency::Resident;
            (void)ninfer::runtime::normalize_engine_options(extended);
            for (const auto residency : {ninfer::VisionResidency::Resident, ninfer::VisionResidency::Cpu}) {
                extended.vision_residency = residency;
                for (const auto policy : {ninfer::MtpDraftPolicy::Fixed, ninfer::MtpDraftPolicy::Adaptive}) {
                    extended.speculative.mtp_policy = policy;
                    for (const auto width : {15U, 31U, 63U}) {
                        extended.speculative.ngram_draft_tokens = width;
                        const auto accepted = ninfer::runtime::normalize_engine_options(extended);
                        if (!accepted.enable_vision || accepted.vision_residency != residency ||
                            accepted.speculative.ngram_draft_tokens != width || accepted.speculative.mtp_policy != policy)
                            throw std::runtime_error("KVMem changed the image/ngram combination");
                    }
                }
            }
            extended.speculative.ngram_draft_tokens = 0;
            extended.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
            extended.enable_vision = false;
            extended.max_concurrency = 4;
            extended.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(1024);
            (void)ninfer::runtime::normalize_engine_options(extended);
            extended.max_concurrency = 1;
            extended.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
            extended.speculative.ngram_draft_tokens = 31;
            (void)ninfer::runtime::normalize_engine_options(extended);
        }
        for (const auto storage : {ninfer::KvCacheStorage::Int8Group64,
                ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value,
                ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8,
                ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value}) {
            for (unsigned lanes = 1; lanes <= 8; ++lanes) {
                auto fast = base; fast.kv_cache = storage; fast.fast_prefill_kernel = true;
                fast.max_concurrency = lanes;
                fast.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
                (void)ninfer::runtime::normalize_engine_options(fast);
                for (const auto residency : {ninfer::VisionResidency::Resident, ninfer::VisionResidency::Cpu}) {
                    fast.enable_vision = true; fast.vision_residency = residency;
                    fast.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                    fast.speculative.draft_tokens = 3; fast.speculative.ngram_draft_tokens = 31;
                    fast.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
                    fast.diagnostic_observer.callback = [](const ninfer::Diagnostic&) {};
                    const auto image = ninfer::runtime::normalize_engine_options(fast);
                    if (!image.fast_prefill_kernel || !image.enable_vision || image.max_concurrency != lanes ||
                        image.kv_cache != storage || image.vision_residency != residency ||
                        image.speculative.ngram_draft_tokens != (lanes > 1 ? 15U : 31U))
                        throw std::runtime_error("Fast image/ngram normalization changed execution settings");
                }
            }
        }
        for (const auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
                ninfer::KvCacheStorage::Nvfp4Group16, ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                ninfer::KvCacheStorage::Fp8E4M3Row256, ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value,
                ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8, ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value}) {
            for (const auto residency : {ninfer::VisionResidency::Resident, ninfer::VisionResidency::Cpu}) {
                for (std::uint32_t lanes = 1; lanes <= 8; ++lanes) {
                    for (const auto drafts : {0U, 3U, 15U}) {
                        auto image = base;
                        image.kv_cache = storage;
                        image.enable_vision = true;
                        image.vision_residency = residency;
                        image.vision_max_merged_tokens = 1024;
                        image.max_concurrency = lanes;
                        image.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
                        if (drafts) {
                            image.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                            image.speculative.draft_tokens = drafts;
                            image.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
                        }
                        const auto accepted = ninfer::runtime::normalize_engine_options(image);
                        if (accepted.max_concurrency != lanes || accepted.kv_capacity.explicit_tokens != 256 * lanes ||
                            accepted.vision_residency != residency || accepted.vision_max_merged_tokens != 1024)
                            throw std::runtime_error("Concurrent Vision changed lane capacity or preprocessing");
                    }
                }
            }
        }
        for (const auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
                ninfer::KvCacheStorage::Nvfp4Group16, ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                ninfer::KvCacheStorage::Fp8E4M3Row256, ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value,
                ninfer::KvCacheStorage::RotatedInt4KeyInt4ValueE8, ninfer::KvCacheStorage::RotatedE8RootKeyInt4Value}) {
            for (std::uint32_t lanes = 2; lanes <= 8; ++lanes) {
                for (const auto width : {0U, 1U, 4U, 15U, 16U, 31U, 63U}) {
                    for (const auto policy : {ninfer::MtpDraftPolicy::Fixed, ninfer::MtpDraftPolicy::Adaptive}) {
                        auto copy = base;
                        copy.kv_cache = storage;
                        copy.max_concurrency = lanes;
                        copy.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256 * lanes);
                        copy.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                        copy.speculative.draft_tokens = 3;
                        copy.speculative.mtp_policy = policy;
                        copy.speculative.ngram_draft_tokens = width;
                        unsigned warnings = 0;
                        copy.diagnostic_observer.callback = [&](const ninfer::Diagnostic& diagnostic) {
                            if (diagnostic.level == ninfer::DiagnosticLevel::Warning) ++warnings;
                        };
                        const auto accepted = ninfer::runtime::normalize_engine_options(copy);
                        if (accepted.max_concurrency != lanes || accepted.speculative.ngram_draft_tokens != std::min(width, 15U) ||
                            accepted.speculative.mtp_policy != policy || warnings != (width > 15 ? 1U : 0U))
                            throw std::runtime_error("Concurrent ngram changed the requested execution");
                        for (const auto residency : {ninfer::VisionResidency::Resident, ninfer::VisionResidency::Cpu}) {
                            copy.enable_vision = true;
                            copy.vision_residency = residency;
                            const auto image = ninfer::runtime::normalize_engine_options(copy);
                            if (image.kv_cache != storage || image.max_concurrency != lanes ||
                                image.vision_residency != residency ||
                                image.speculative.ngram_draft_tokens != std::min(width, 15U) ||
                                image.speculative.mtp_policy != policy)
                                throw std::runtime_error("Concurrent image/ngram settings changed");
                        }
                    }
                }
            }
        }
        for (const auto backend : {ninfer::SpeculativeBackend::Mtp, ninfer::SpeculativeBackend::DFlash,
                                   ninfer::SpeculativeBackend::DFlash2}) {
            for (const auto lanes : {1U, 2U, 8U}) {
                for (const auto width : {0U, 1U, 15U, 16U, 31U, 63U, 64U}) {
                    ninfer::EngineOptions native;
                    native.max_concurrency = lanes;
                    native.speculative.backend = backend;
                    native.speculative.draft_tokens = 3;
                    native.speculative.ngram_draft_tokens = width;
                    unsigned warnings = 0;
                    bool warning_has_settings = true;
                    native.diagnostic_observer.callback = [&](const ninfer::Diagnostic& diagnostic) {
                        if (diagnostic.level != ninfer::DiagnosticLevel::Warning) return;
                        ++warnings;
                        warning_has_settings = warning_has_settings &&
                            diagnostic.message.find("from " + std::to_string(width) + " to 15") != std::string::npos &&
                            diagnostic.message.find("concurrency " + std::to_string(lanes)) != std::string::npos;
                    };
                    bool rejected = false;
                    try {
                        const auto normalized = ninfer::runtime::normalize_engine_options(native);
                        const auto expected = lanes == 1 ? width : std::min(width, 15U);
                        if (normalized.speculative.ngram_draft_tokens != expected)
                            throw std::runtime_error("Native ngram normalization changed supported width");
                    } catch (const std::invalid_argument&) { rejected = true; }
                    if (rejected != (width == 64) || !warning_has_settings ||
                        warnings != (lanes > 1 && width > 15 && width <= 63 ? 1U : 0U))
                        throw std::runtime_error("Native ngram clamp or invalid-width validation failed");
                }
            }
        }
        const std::vector<std::function<void(ninfer::EngineOptions&)>> invalid{
            [](auto& o) { o.max_concurrency = 2; },
            [](auto& o) { o.max_concurrency = 9;
                         o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2304); },
            [](auto& o) { o.context_cache.enabled = true; },
            [](auto& o) { o.kvmem.disk_path = "missing-quota"; },
            [](auto& o) { o.kvmem.disk_bytes = 1ULL << 30; },
            [](auto& o) { o.enable_vision = true; o.vision_residency = ninfer::VisionResidency::Overlay; },
            [](auto& o) { o.enable_vision = true; o.max_concurrency = 8;
                         o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256); },
            [](auto& o) { o.concurrent_prefill = true; },
            [](auto& o) { o.fast_prefill_kernel = true; },
            [](auto& o) { o.fast_prefill_kernel = true; o.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256; },
            [](auto& o) { o.fast_prefill_kernel = true; o.kv_cache = ninfer::KvCacheStorage::Nvfp4Group16; },
            [](auto& o) { o.fast_prefill_kernel = true; o.kv_cache = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
                         o.enable_vision = true; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp; },
            [](auto& o) { o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.kvmem.selected_tokens = 127; },
            [](auto& o) { o.prefill_chunk = 256; },
            [](auto& o) { o.purpose = ninfer::EnginePurpose::CausalScoring; },
            [](auto& o) { o.kv_cache = static_cast<ninfer::KvCacheStorage>(255); },
            [](auto& o) { o.cuda_memory_policy = ninfer::CudaMemoryPolicy::Mixed; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 16; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3;
                         o.speculative.mtp_policy = ninfer::MtpDraftPolicy::Adaptive;
                         o.speculative.ngram_draft_tokens = 64;
                         o.max_concurrency = 2; o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 64; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 64;
                         o.max_concurrency = 2; o.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(512); },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.ngram_draft_tokens = 63;
                         o.kvmem.selected_tokens = 192; o.kvmem.reserve_tokens = 64; o.prefill_chunk = 64; },
            [](auto& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                         o.speculative.draft_tokens = 3; o.speculative.mtp_attention_window = 128; },
        };
        for (std::size_t case_index = 0; case_index < invalid.size(); ++case_index) {
            const auto& change = invalid[case_index];
            for (const auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Nvfp4Group16,
                                       ninfer::KvCacheStorage::Fp8KeyNvfp4Value}) {
                auto bad = base;
                bad.kv_cache = storage;
                change(bad);
                bool rejected = false;
                try { (void)ninfer::runtime::normalize_engine_options(bad); }
                catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) {
                    throw std::runtime_error("unsupported KVMem combination accepted: case=" +
                        std::to_string(case_index) + " storage=" + std::to_string(static_cast<unsigned>(storage)));
                }
            }
        }
        std::cout << "P1_OPTIONS PASS: NVFP4/K8V4 ordinary/fixed MTP1..15, adaptive lanes/images/ngram, CPU Vision, fast prefill with shared tiering, paired H/D validation and " << invalid.size() * 3
                  << " rejected combinations\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "P1_OPTIONS FAIL " << e.what() << '\n';
        return 1;
    }
}
