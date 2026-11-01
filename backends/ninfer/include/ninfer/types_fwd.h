#pragma once

// Declarations of the value types of the in-tree interface. A header that names a type only by
// reference or pointer, or an enum without its enumerators, includes this header instead of
// the defining one, so a change to the definition does not recompile its users.
// ninfer/types.h includes both, which keeps the declarations and the definitions consistent;
// scripts/check-include-boundaries.py checks that every type is declared here.

#include <cstdint>

namespace ninfer {

// ninfer/model_enums.h
enum class EnginePurpose : std::uint8_t;
enum class KvCapacityMode : std::uint8_t;
enum class ProposalHead : std::uint8_t;
enum class SpeculativeBackend : std::uint8_t;
enum class MtpDraftPolicy : std::uint8_t;
enum class VisionResidency : std::uint8_t;
enum class ExpertResidency : std::uint8_t;
enum class NgramResidency : std::uint8_t;
enum class NgramIo : std::uint8_t;

// ninfer/kv_cache_storage.h
enum class KvCacheStorage : std::uint8_t;

// ninfer/startup.h
enum class StartupPhase : std::uint8_t;
enum class StartupStatus : std::uint8_t;
enum class StartupProgressUnit : std::uint8_t;
struct StartupEvent;
struct StartupObserver;

// ninfer/engine_options.h
struct KvCapacityPolicy;
struct SpeculativeOptions;
enum class DiagnosticLevel : std::uint8_t;
struct Diagnostic;
struct DiagnosticObserver;
enum class ContextCacheMode : std::uint8_t;
class PrefixCacheSaveControl;
struct HybridPrefixCacheOptions;
struct SlotAutoSaveEvent;
struct SlotAutoSaveOptions;
struct ContextCacheOptions;
struct ContextCostOptions;
struct GraftSource;
enum class SuspendSnapshotMemory : std::uint8_t;
enum class SuspendWeightSource : std::uint8_t;
struct ModelSuspendOptions;
struct HybridExpertOptions;
struct NgramTableOptions;
enum class CudaMemoryPolicy : std::uint8_t;
struct KvmemOptions;
struct EngineOptions;

// ninfer/request.h
enum class SamplingMode : std::uint8_t;
struct SamplingPreset;
struct ModelSamplingDefaults;
struct SamplingOverrides;
struct ResolvedSamplingParameters;
enum class OutputChannel : std::uint8_t;
struct StopString;
struct StopPolicy;
enum class ReasoningLoopAction : std::uint8_t;
struct ThinkingControlOptions;
enum class StructuredOutputKind : std::uint8_t;
struct StructuredOutputOptions;
struct ExecutionOptions;
struct OutputOptions;
struct NgramSessionHints;
enum class ToolChoiceMode : std::uint8_t;
enum class ToolConstraintMode : std::uint8_t;
struct ToolChoice;
struct RequestOptions;
enum class MediaKind : std::uint8_t;
enum class ImageResizePolicy : std::uint8_t;
enum class ImageDetail : std::uint8_t;
struct OwnedMedia;
struct ToolCall;
struct GeneratedToolCall;
enum class ToolCallParseFallbackReason : std::uint8_t;
struct ToolCallParseDiagnostics;
enum class ChatRole : std::uint8_t;
enum class MessagePartKind : std::uint8_t;
struct MessagePart;
struct ChatMessage;
enum class ReasoningEffort : std::uint8_t;
enum class PromptContinuationMode : std::uint8_t;
struct PromptOptions;
enum class CacheRetentionHint : std::uint8_t;
enum class PromptCacheMarkerKind : std::uint8_t;
enum class SharedCandidateEvidence : std::uint8_t;
enum class PromptCacheMarkerLocation : std::uint8_t;
struct PromptCacheMarker;
struct ContextCacheHints;
struct PromptInput;
class ContextCacheExhausted;
enum class RequestErrorKind : std::uint8_t;
class RequestError;
struct PromptSummary;
struct PromptPreparationStats;
enum class FinishReason : std::uint8_t;
struct TokenLogprob;
struct BranchResult;
struct OutputDelta;
struct GenerationStart;
struct PromptProgress;
struct GenerationTimingObservation;
class OutputSink;
enum class OutputConsumerMode : std::uint8_t;
struct GenerationObservationOptions;
class CancellationView;
struct PreparationControl;
struct GenerationResult;

// ninfer/observability/execution.h
struct GenerationTimings;
struct GenerationEngineTiming;
struct SpeculativeStats;
struct ThinkingBudgetStats;
enum class PrefixReusePath : std::uint8_t;
enum class MaterializationStopReason : std::uint8_t;
enum class MaterializationSearchPhase : std::uint8_t;
struct MaterializationDiagnostics;
struct NgramArchiveStats;

// ninfer/observability/resources.h
struct MediaCacheSummary;
struct ArenaMemorySummary;
struct VisionWorkspaceMemorySummary;
struct DeviceMemorySummary;
struct KvmemMemorySummary;
struct CudaResidencySummary;
struct MemorySummary;
struct RuntimeHostWorkStats;
struct NgramTableStats;
struct ExpertResidencyStats;

// ninfer/observability/snapshot.h
enum class ModelResidency : std::uint8_t;
struct ResidencyTransition;
struct ResidencyStatus;
struct RuntimeStats;
enum class ContextCostPresetSource : std::uint8_t;
struct ContextCostSummary;
struct ModelMetadata;
struct SlotCheckpoint;
struct SlotState;
struct SlotSaveResult;
struct SlotRestoreResult;
class SlotSessionMismatch;
enum class RopeScalingType : std::uint8_t;
struct RopeScalingSummary;
struct LoadSummary;

} // namespace ninfer
