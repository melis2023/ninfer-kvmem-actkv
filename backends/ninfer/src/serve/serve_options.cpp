#include "serve/serve_options.h"
#include "kvmem/retrieval_options.hpp"
#include "runtime/speculative_limits.h"
#include "product/cuda_memory_options.h"
#include "product/post_thinking_options.h"
#include "product/rope_yarn_options.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    // RotorQuant rk8v4: rotated INT8 keys with a packed signed int4 value plane. Opt-in.
    if (value == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    // rk4v4: rotated 4-bit Lloyd-Max keys with rk8v4's packed int4 value plane. Opt-in.
    if (value == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }
    // rk4v4-e8: rotated E8-snapped int4 keys with the rk8v4 value plane. Opt-in.
    if (value == "rk4v4-e8") { return KvCacheStorage::RotatedInt4KeyInt4ValueE8; }
    // rk2v4-e8: rotated keys as E8 root codes, two bytes per eight dimensions. Opt-in.
    if (value == "rk2v4-e8") { return KvCacheStorage::RotatedE8RootKeyInt4Value; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> [options]\n"
           "\n"
           "Serves the OpenAI Responses/Chat Completions and Anthropic Messages APIs.\n"
           "  --help, -h                    show this help and exit\n"
           "\n"
           "MODEL & CONTEXT\n"
           "  --kvmem-budget N              selected history B, tokens aligned to 64\n"
            "  --kvmem-lrfu               rank by decayed retrieval history\n"
            "  --kvmem-lrfu-lambda X      history decay [0,1] (default 0.5)\n"
            "  --kvmem-lrfu-top-frac X    hit fraction (0,1] (default 0.9)\n"
            "  --kvmem-score-agg MODE     mean | max across GQA heads (default mean)\n"
           "  --kvmem-gen-reserve N         append workspace R, tokens aligned to 64\n"
           "  --kvmem-host-mib N            Host native KV payload budget H, MiB\n"
           "  --kvmem-sink-tokens N         always-kept prefix; default 0 = one 64-token\n"
           "                                page; positive values round down, minimum one page\n"
           "  --kvmem-recent-tokens N       always-kept newest suffix; default 64 (one page);\n"
           "                                minimum 64; values round down to whole pages\n"
           "  --kvmem-sessions N            retained Host histories, 1..16 (default 4)\n"
           "  --kvmem-disk-path PATH        Ephemeral active/idle KV spill directory\n"
           "  --kvmem-disk-mib N            Shared SSD payload quota D, MiB\n"
           "                                KVMem uses B+R device tokens and its own\n"
           "                                verified history reuse; BF16/INT8/FP8/NVFP4/K8V4,\n"
           "                                RK8V4/RK4V4/RK4V4-E8/RK2V4-E8,\n"
           "                                1..8 text or resident/CPU-vision lanes,\n"
           "                                ordinary/fixed/adaptive MTP1..15; ngram1..63,\n"
           "                                C2..8 automatically cap ngram at 15;\n"
           "                                ngram text/images use all nine KV formats,\n"
           "                                SSD: process-local active/idle tiering; fast prefill supports\n"
           "                                INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 text/images\n"
           "  --kvmem-verify-transfers      check restored KV/GDN bytes (diagnostic)\n"
           "  --max-context N               logical context ceiling of each request (default\n"
           "                                8192)\n"
           "  --max-concurrency N           requests decoded together, 1..8 (default 1)\n"
           "  --prefill-chunk N             prefill chunk in tokens, a multiple of 128\n"
           "                                (default 1024)\n"
           "  --default-max-tokens N        output limit of a request that sets none\n"
           "                                (default: the largest budget that still lets\n"
           "                                every lane be admitted at once, the remaining\n"
           "                                context with one lane)\n"
           "  --model-id ID                 public model name instead of the artifact's\n"
           "                                metadata.name\n"
           "  --chat-template FILE          replace the artifact's chat template at startup\n"
           "  --device N                    CUDA device index (default 0)\n"
           "  --devices A,B,...             one pipeline stage per listed device, 2..8\n"
           "                                (Linux/Windows); excludes --device\n"
           "  --stage-layers A,B,...        layers per stage in --devices order (default:\n"
           "                                split by free memory)\n"
           "  --expert-residency device|host|disk  Qwen3.8-Flash-Next: expert banks in GPU\n"
           "                                memory (default), in pinned host memory read\n"
           "                                across the bus, or left in the artifact's files\n"
           "                                and streamed into the device expert cache\n"
           "  --expert-cache-mib N|auto     with host or disk experts: device memory for the\n"
           "                                most used experts (default auto: what is free;\n"
           "                                0 turns the host cache off; disk needs one)\n"
           "  --ngram-table PATH            Qwen3.8-Flash-Next: the n-gram table artifact,\n"
           "                                for a model published without its table\n"
           "  --ngram-ram                   Qwen3.8-Flash-Next: load the n-gram table into\n"
           "                                RAM instead of reading its rows from the file\n"
           "  --no-ngram-table              Qwen3.8-Flash-Next: run without the n-gram table.\n"
           "                                Non-standard experimental mode: the model was\n"
           "                                trained with the table and degrades badly\n"
           "                                without it (WikiText-2 perplexity 2.66 -> 5.01)\n"
           "  --no-cuda-graph               decode without CUDA Graphs (on by default)\n"
           "  --cuda-graph-allowance-mib N  CUDA Graph driver-state allowance taken from the\n"
           "                                KV sizing budget (default 0: computed per\n"
           "                                profile)\n"
           "  --device-profile auto|off|calibrate\n"
           "                                per-GPU route profile: auto uses the stored or\n"
           "                                built-in one and calibrates a device that has\n"
           "                                none; off keeps the compiled routes; calibrate\n"
           "                                measures anew (default auto)\n"
           "  --device-profile-path FILE    profile file instead of the user cache\n"
           "  --context-cost-presets FILE   runtime context-cost presets over the\n"
           "                                compiled-in values\n"
           "\n"
           "PRECISION & KERNELS\n"
           "  --fast-prefill-kernel         prefill an int8 or rk* KV cache with the fast\n"
           "                                prompt-attention kernel (FP16 PV per 64-key\n"
           "                                tile) and round --prefill-chunk down to whole\n"
           "                                attention waves; an nvfp4 cache past 2048 keys\n"
           "                                with its FP4 Tensor Core QK kernel (Blackwell)\n"
           "  --prefill-cublas              hand wide prefill GEMMs to cuBLAS: a large\n"
           "                                prefill speedup for a small perplexity cost;\n"
           "                                wants a larger --prefill-chunk\n"
           "  --no-prefill-cublas-projections\n"
           "                                with --prefill-cublas, keep the attention and\n"
           "                                GDN input projections off that route\n"
           "  --no-prefill-a8               prefill every projection on the A16 routes\n"
           "                                instead of the integer-activation ones\n"
           "  --mlp-a8-decode               integer-activation MLP gate_up at decode and\n"
           "                                verify widths\n"
           "  --lm-head-q4 | --lm-head-q6   store the output head as Q4 or Q6 while loading\n"
           "                                (Q4 costs +0.69% perplexity, Q6 +0.01%)\n"
           "  --embedding-q4 | --embedding-q6\n"
           "                                store the token embedding as Q4 or Q6 while\n"
           "                                loading\n"
           "  --mtp-experts-q4              Qwen3.6-35B-A3B: store the MTP layer's routed\n"
           "                                experts in the text layers' formats\n"
           "  --gdn-state-fp16              keep the GDN recurrent state in FP16 (halves\n"
           "                                each host state image)\n"
           "  --rope-yarn                   past the model's native window, apply Qwen's\n"
           "                                YaRN at factor max-context / native instead of\n"
           "                                unscaled RoPE\n"
           "  --rope-yarn-factor F          apply YaRN at this fixed factor, 1..4, to every\n"
           "                                position whatever --max-context is (default 1:\n"
           "                                as --rope-yarn decides)\n"
           "  --rope-scaling-factor F       instead of YaRN, interpolate positions past\n"
           "                                --rope-scaling-original-context linearly by F,\n"
           "                                1..32 (default 1: off)\n"
           "  --rope-scaling-original-context N\n"
           "                                the interpolation threshold (default: the\n"
           "                                model's native window)\n"
           "  --cuda-memory-policy MODE    behavior when dedicated GPU memory is insufficient:\n"
           "    default                     plan against CUDA-reported free memory; ordinary\n"
           "                                driver behavior, no dedicated-residency guarantee\n"
           "    mixed                       attempt requested allocations beyond CUDA free;\n"
           "                                allow Windows to borrow system RAM for device data.\n"
           "                                Placement is driver-managed, not forced to Shared;\n"
           "                                inference may become much slower. Auto KV is capped\n"
           "                                at one page-rounded context window per concurrent\n"
           "                                request; allocation failure is reported, not retried\n"
           "    strict                      require observed dedicated-VRAM residency; equals\n"
           "                                strict-64-128 (64 MiB reserve, 128 MiB probe step)\n"
           "    strict-RESERVE-STEP          customize both values in MiB; RESERVE >= 0,\n"
           "                                STEP 1..16384. Example: strict-128-64. Reserve is\n"
           "                                verified at startup, not permanently locked; step\n"
           "                                controls probes, final device arenas stay contiguous\n"
           "                                strict selects its required cache automatically.\n"
           "                                --host-cache-mib independently sizes CPU context\n"
           "                                retention, not device-data system-RAM fallback.\n"
           "                                mixed/strict: Windows single-GPU text, no WDDM mode;\n"
           "                                no model suspend, router, grafts or qwen4_exp.\n"
           "  --wddm-evictable-budget       Windows D3D12 builds: budget against dedicated\n"
           "                                memory, holding arenas resident\n"
           "\n"
           "ROUTER (several models, llama.cpp-compatible; give no artifact path)\n"
           "  --models-dir DIR              serve every .ninfer in DIR (and each subdirectory\n"
           "                                holding one) by its file or directory name\n"
           "  --models-preset FILE          INI presets: [id] sections with model = PATH and\n"
           "                                options without dashes; [*] applies to all\n"
           "  --models-max N                models loaded at once (default 1, 0 = no limit);\n"
           "                                the least recently used idle one sleeps (with\n"
           "                                --model-suspend) or unloads to make room\n"
           "  --no-models-autoload          a request for a model that is not loaded fails\n"
           "                                instead of loading it (?autoload=1 overrides)\n"
           "  --sleep-idle-seconds N        put a model idle for N s to sleep (needs\n"
           "                                --model-suspend); also without a router\n"
           "  --unload-idle-seconds N       unload a model idle for N s\n"
           "\n"
           "MODEL SUSPEND\n"
           "  --model-suspend               enable POST /v1/models/{id}/suspend and /resume\n"
           "                                (and /models/unload, /models/load): device memory\n"
           "                                at fixed addresses, released while suspended;\n"
           "                                needs CUDA virtual memory management\n"
           "  --suspend-snapshot M          where suspended state waits: pageable (default;\n"
           "                                host memory only while suspended) or pinned\n"
           "                                (reserved at startup, faster transfers)\n"
           "  --suspend-weights S           resume reads the weights from: artifact\n"
           "                                (default) or host (a host copy taken at suspend:\n"
           "                                weight-sized host memory, resume at bus speed)\n"
           "  --no-auto-resume              a request while suspended fails (503) instead of\n"
           "                                resuming the model (also per suspend request)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto          shared KV capacity in tokens (default\n"
           "                                --max-context, or auto with\n"
           "                                Hybrid/strict); auto follows --cuda-memory-policy\n"
           "  --kv-headroom-mib N           default policy only: auto KV leaves free (default\n"
           "                                " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           "); alias --vram-headroom-mib\n"
           "  --kv-dtype T                  KV storage: bf16 (default), int8, fp8, rk8v4,\n"
           "                                rk4v4, rk4v4-e8, rk2v4-e8, nvfp4 or k8v4\n"
           "\n"
           "CONTEXT CACHE\n"
           "  --no-prefix-reuse             disable compatible-prefix caching (on by\n"
           "                                default); excludes the options below\n"
           "  --device-state-slots N        checkpoint states on the device beyond the\n"
           "                                active lanes (default max-concurrency)\n"
           "  --host-state-slots N          pinned host checkpoint states (default 8)\n"
           "  --host-kv-mib N               pinned host KV in MiB (default 8192)\n"
           "  --host-cache-mib N            one pinned host RAM ceiling instead of the two\n"
           "                                above: host states for every checkpoint the\n"
           "                                capture path creates (at most half), more\n"
           "                                automatic anchors with the headroom, host KV the\n"
           "                                rest; with --use-alt-prefix-caching the pool KV\n"
           "                                blocks and state snapshots share (default 8192,\n"
           "                                0 keeps the cache on the device)\n"
           "  --max-private-continuations N private continuation catalog (default 2x\n"
           "                                max-concurrency)\n"
           "  --max-shared-prefixes N       shared stable-prefix catalog (default\n"
           "                                max(max-concurrency,7))\n"
           "  --max-long-anchors-per-continuation N\n"
           "                                long anchors kept per continuation (default 2, 4\n"
           "                                with --auto-long-anchors)\n"
           "  --auto-long-anchors           anchor up to that many message boundaries of\n"
           "                                every request, so a request that rewrites\n"
           "                                earlier history resumes from the nearest anchor\n"
           "                                instead of root (off)\n"
           "  --branch-anchors              anchor a request where its prompt stops matching\n"
           "                                a retained conversation, when no checkpoint lies\n"
           "                                near that depth, so the next request diverging\n"
           "                                there resumes from it (off)\n"
           "  --no-endpoint-anchors         do not keep the endpoint a request resumes and\n"
           "                                extends as a long anchor; a later request that\n"
           "                                diverges there (another reply, an edited last\n"
           "                                message) then re-prefills the conversation (kept)\n"
           "  --long-anchor-spacing N       with --auto-long-anchors, minimum tokens between\n"
           "                                anchors, doubling per anchor back from the\n"
           "                                prompt end (default 1024; 0 anchors every\n"
           "                                boundary)\n"
           "  --max-cache-markers-per-request N\n"
           "                                client cache markers one request may carry\n"
           "                                (default 4)\n"
           "  --auto-prefix-grid            offer shared candidates on a token grid, so\n"
           "                                callers whose prompts start alike share a prefix\n"
           "                                without a client hint; a grid frontier is\n"
           "                                published once two callers ask for it\n"
           "  --derive-session-keys         give a request without a session key one derived\n"
           "                                from its instructions and first user message,\n"
           "                                so the conversation keeps a session lineage\n"
           "                                and live-session retention (default off)\n"
           "  --context-cache-policy default|rolling\n"
           "                                rolling: a capture that extends a resident\n"
           "                                checkpoint inherits its demand, for one\n"
           "                                conversation whose prompt only grows\n"
           "  --release-diverged-checkpoints\n"
           "                                a conversation's private checkpoint that its\n"
           "                                next prompt diverges from keeps no value and\n"
           "                                goes first\n"
           "  --thorough-admission-search   search up to 250 ms for a new request's reuse\n"
           "                                plan even while others decode (otherwise 10 ms),\n"
           "                                longer for a costly request, exploring every\n"
           "                                eligible option\n"
           "  --recency-eviction            under pressure give up the least recently used\n"
           "                                owners first, only as many as needed, and demote\n"
           "                                kept ones to Host\n"
           "  --value-aware-demote          under pressure prefer evicting the conversations\n"
           "                                cheapest to rebuild and demoting the costliest\n"
           "  --kv-lease-growth             reserve a 4096-token output window and extend it\n"
           "                                instead of the whole max_tokens budget; an answer\n"
           "                                the pool cannot extend ends with length\n"
           "  --concurrent-prefill          admit waiting requests to free lanes while other\n"
           "                                requests prefill\n"
           "  --disk-kv-path DIR            disk tier: evicted continuations write their KV\n"
           "                                and states there, per artifact and profile, and\n"
           "                                survive restarts\n"
           "  --disk-kv-gib N               disk tier budget (default 64)\n"
           "  --disk-kv-restore             seed a new request's matching prefix from the\n"
           "                                disk tier\n"
           "  --disk-kv-directstorage       read restores through DirectStorage (Windows\n"
           "                                builds with NINFER_DIRECTSTORAGE)\n"
           "  --slot-save-path DIR          enable POST /slots/{id}?action=save|restore|erase,\n"
           "                                which writes a retained session to a file in DIR\n"
           "                                or restores one from it\n"
           "  --auto-save-evicted           write a retained session back to the slot file it\n"
           "                                was last saved to or restored from before an\n"
           "                                involuntary eviction destroys it (needs\n"
           "                                --slot-save-path)\n"
           "  --use-alt-prefix-caching      the hybrid prefix cache instead of the checkpoint\n"
           "                                catalog: content-addressed KV blocks and sparse\n"
           "                                state snapshots; free VRAM becomes block cache\n"
           "                                and --host-cache-mib sizes the Host tier;\n"
           "                                excludes the catalog options above\n"
           "  --use-original-prefix-caching the checkpoint catalog, which is the default;\n"
           "                                accepted for command lines that name it\n"
           "  --device-snapshot-slots N     hybrid: device state snapshot slots (default\n"
           "                                concurrency + 1; + 2 without a Host tier)\n"
           "  --cache-taps-per-request N    hybrid: new prefill snapshots per request\n"
           "                                (default 8; 2 without a Host tier)\n"
           "  --cache-tap-ladder N          hybrid: ladder base in tokens for history\n"
           "                                snapshots (default max(4096, 2x prefill chunk))\n"
           "  --cache-tap-min-gap N         hybrid: minimum tokens between ladder snapshots\n"
           "                                (default max(1024, prefill chunk))\n"
           "  --prefix-cache-file PATH      hybrid: restore the Host tier from PATH at\n"
           "                                startup when this binary wrote it for the same\n"
           "                                artifact and KV format, and save it there when\n"
           "                                the server stops (Ctrl+C twice) (default off)\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2     speculative decoding backend\n"
           "  --draft-tokens N              drafts per round, 1..15\n"
           "  --lm-head-draft               draft with the optimized proposal head, which\n"
           "                                the artifact must store (the default when it does)\n"
           "  --full-draft-head             draft with the full output head even when the\n"
           "                                artifact stores a proposal head\n"
           "  --adaptive-mtp                each MTP round verifies 3..--draft-tokens\n"
           "                                drafts, the width the drafts' measured survival\n"
           "                                and round cost favor; greedy output is unchanged\n"
           "  --mtp-attention-window N      the MTP draft head attends to its first 64 keys\n"
           "                                and the newest N before its query, not the\n"
           "                                whole history; verification is unchanged\n"
           "                                (default 0: whole history)\n"
           "  --ngram-draft-tokens N        copy up to N tokens (1..63) per round from\n"
           "                                earlier prompt, tool-result or output text,\n"
           "                                verified alongside --spec; on with 15 whenever\n"
           "                                --spec is set, 0 disables it; widths above 15\n"
           "                                automatically reduce to 15 with concurrency >1\n"
           "  --ngram-min-match N           shortest match a copy is drawn from, 4..64\n"
           "                                (default 12)\n"
           "  --ngram-archive-mib N         keep finished requests' copy sources in RAM for\n"
           "                                later requests naming the same\n"
           "                                X-NInfer-Draft-Session (default 0: off)\n"
           "  --ngram-session-mib N         per-session share of that archive (default 128)\n"
           "  --ngram-native-sessions       also recognize the session identities Kilo,\n"
           "                                Codex and Claude send\n"
           "  --lookup-ngram N              context-lookup drafting alongside --spec: the\n"
           "                                last N tokens are matched against the sequence\n"
           "                                so far and what followed is proposed (default 0:\n"
           "                                off; disabled with KVMem)\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                      accept images and video and load the Vision GPU\n"
           "                                allocations\n"
           "  --vision-residency resident|overlay|cpu\n"
           "                                overlay keeps the Vision tower in host memory\n"
           "                                and borrows device memory per image (alias\n"
           "                                --vision-offload); cpu runs it on CPU threads\n"
           "                                with no device Vision memory, and caps\n"
           "                                --vision-max-merged at 256 unless given\n"
           "  --vision-cpu                  --vision with --vision-residency cpu\n"
           "  --vision-max-merged N         merged tokens of one media item, 64..16384\n"
           "                                (default 16384); larger media is downscaled\n"
           "  --media-cache-mib N           retained prepared media (default 1024; 0\n"
           "                                disables)\n"
           "  --media-live-mib N            all live prepared media payloads (default 2048)\n"
           "  --media-preprocess-threads N  media preprocessing workers (default 0: auto, at\n"
           "                                most 16)\n"
           "\n"
           "SAMPLING & THINKING\n"
           "  --temperature F               0..2\n"
           "  --top-p F                     0..1\n"
           "  --top-k N                     0..20\n"
           "  --min-p F                     0..1\n"
           "  --presence-penalty F          -2..2\n"
           "  --frequency-penalty F         -2..2\n"
           "  --seed N                      seed of a request that sets none (default: fresh\n"
           "                                per request)\n"
           "  --greedy                      force temperature 0 (exact argmax)\n"
           "  --post-thinking               sample a thinking request's answer with the\n"
           "                                post-thinking preset (temperature 0.2) from the\n"
           "                                token after its reasoning closes (default off;\n"
           "                                a request opts in with a post_thinking object)\n"
           "  --post-thinking-temperature F 0..2; implies --post-thinking\n"
           "  --post-thinking-top-p F       0..1; implies --post-thinking\n"
           "  --post-thinking-top-k N       0..20; implies --post-thinking\n"
           "  --post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]\n"
           "                                the same fields in one flag\n"
           "  --no-thinking                 disable thinking by default\n"
           "  --preserve-thinking           retain closed-turn assistant reasoning in later\n"
           "                                prompts\n"
           "  --default-thinking-budget N   cap model-origin thinking of thinking requests;\n"
           "                                control tokens count toward the output limit\n"
           "  --thinking-budget-message TEXT\n"
           "                                notice a request gets at its thinking budget;\n"
           "                                the canonical </think> close is appended when\n"
           "                                missing\n"
           "  --default-reasoning-effort E  none, minimal, low, medium, high, xhigh or max\n"
           "                                for requests that set no effort and keep\n"
           "                                thinking on\n"
           "\n"
           "PROMPT GRAFTS\n"
           "  --graft NAME=PATH             load a phantom-kv graft (a safetensors container\n"
           "                                with a .json sidecar beside it); a request\n"
           "                                selecting it with \"graft\": \"NAME\" runs as if\n"
           "                                the graft's hidden turn preceded its own\n"
           "                                messages; repeatable\n"
           "  --default-graft NAME          apply a loaded graft to every request that names\n"
           "                                none; a request opts out with \"graft\": \"\"\n"
           "\n"
           "API BEHAVIOR\n"
           "  --structured-output           accept JSON and JSON Schema response formats;\n"
           "                                native tool constraints are enabled by default\n"
           "  --unconstrained-response-format\n"
           "                                without --structured-output, generate such a\n"
           "                                request unconstrained instead of refusing it\n"
           "  --assistant-prefill           continue a Chat Completions request's trailing\n"
           "                                assistant message in place, as /v1/messages\n"
           "                                does; needs thinking disabled\n"
           "  --lenient-assistant-history   accept Responses input whose assistant text or\n"
           "                                reasoning follows function_call Items; it joins\n"
           "                                that turn, rendered before its calls\n"
           "  --usage-chunk-choice          give the streamed usage chunk a zero-delta\n"
           "                                choice, for strict parsers that reject\n"
           "                                choices:[]\n"
           "\n"
           "NETWORK & LIMITS\n"
           "  --host H                      listen address (default 127.0.0.1)\n"
           "  --port N                      listen port (default 8080)\n"
           "  --stats-port N                also serve /health, /stats, /v1/load and /metrics\n"
           "                                on port N with a worker of their own (default off)\n"
           "  --api-key KEY                 require this bearer or x-api-key value (default:\n"
           "                                none)\n"
           "  --cors                        send permissive CORS headers for browser clients\n"
           "  --no-webui                    do not serve a WebUI built in with\n"
           "                                NINFER_WEBUI_DIR on GET /\n"
           "  --webui-mcp-proxy             relay the WebUI's MCP traffic at /cors-proxy;\n"
           "                                http targets only, reaches any host and carries\n"
           "                                no API key, so only on a trusted bind\n"
           "  --max-request-mib N           request body limit, enforced before parsing\n"
           "                                (default 384)\n"
           "  --max-pending-requests N      requests waiting for admission (default 16)\n"
           "  --pending-timeout-ms N        preparation-plus-admission wait (default 600000)\n"
           "  --recover-invariant-failures  on a broken internal invariant, fail the active\n"
           "                                requests and keep serving the queue instead of\n"
           "                                failing the engine\n"
           "  --response-store-max-records N\n"
           "                                retained Responses objects (default 1024)\n"
           "  --response-store-max-mib N    Responses state budget (default 256)\n"
           "\n"
           "LOGGING\n"
           "  --request-log-jsonl FILE      append full-precision server and request records\n"
           "  --request-log-max-mib N       rotate the request log at N MiB (default 0: one\n"
           "                                unbounded file)\n"
           "  --request-log-keep N          rotated request logs kept (default 4)\n"
           "  --log-stats-interval-ms N     throughput report interval (default 5000; 0\n"
           "                                disables)\n"
           "  --log-level L                 trace, debug, info (default), warning, error,\n"
           "                                critical or off\n"
           "  --log-colours on|off          colour the console log's levels and statistics\n"
           "                                (default: levels on a console, statistics off)\n"
           "  --log-stats-panel on|off      pin session statistics beneath the console log on\n"
           "                                an interactive terminal (default off)\n"
           "\n"
           "NOTES\n"
           "  Sampler defaults come from the loaded model and the resolved thinking mode;\n"
           "  server flags and request fields override individual values.\n"
           "  context cache defaults: device-state=max-concurrency, private=2x concurrency,\n"
           "  shared=max(max-concurrency,7), anchors=2 (4 with --auto-long-anchors),\n"
           "  Host state=8 slots, Host KV=8192 MiB.\n";
}

// "1,2,3" selects the ordered devices the model's pipeline stages run on; the first also holds the
// embedding, head and round state. One entry is accepted and is equivalent to --device. The engine
// validates matching compute capability at startup; this only parses the shape.
constexpr std::size_t kMaximumDevices = 8;
std::vector<int> parse_device_list(std::string_view value) {
    std::vector<int> devices;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        const std::string_view piece =
            value.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                                : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--devices entries must not be empty"); }
        const std::string entry(piece);
        devices.push_back(parse_nonnegative_int(entry.c_str(), "devices"));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    if (devices.empty() || devices.size() > kMaximumDevices) {
        throw std::invalid_argument("--devices takes between one and " +
                                    std::to_string(kMaximumDevices) + " CUDA device ids");
    }
    // Repeated ids are deliberately permitted: they put several stages on one card, which saves no
    // memory but exercises the whole split path on a single-GPU machine.
    return devices;
}

// "30,34": the layers each pipeline stage owns, one count per device in --devices. Whether they add
// up to the model's layers is checked when the model loads; this only parses the shape.
std::vector<std::uint32_t> parse_stage_layers(std::string_view text) {
    std::vector<std::uint32_t> counts;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string_view piece =
            text.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                               : comma - start);
        if (piece.empty()) { throw std::invalid_argument("--stage-layers entries must not be empty"); }
        const std::string entry(piece);
        const int count = parse_nonnegative_int(entry.c_str(), "stage-layers");
        if (count == 0) { throw std::invalid_argument("--stage-layers counts must be positive"); }
        counts.push_back(static_cast<std::uint32_t>(count));
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return counts;
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool kv_capacity_explicit         = false;
    bool device_explicit              = false;
    bool context_capacity_explicit    = false;
    bool host_state_slots_explicit    = false;
    bool host_kv_mib_explicit         = false;
    bool host_cache_budget_explicit   = false;
    bool long_anchor_spacing_explicit = false;
    bool ngram_width_explicit         = false;
    bool vision_max_merged_explicit   = false;
    bool sink_tokens_explicit         = false;
    bool recent_tokens_explicit       = false;
    std::optional<std::size_t> kv_headroom_mib;
    // Last flag seen that belongs to only one prefix-cache mode, for the cross-mode error.
    const char* legacy_cache_flag  = nullptr;
    const char* hybrid_option_flag = nullptr;
    bool original_cache_selected   = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    // Router mode starts with an option: the models come from --models-dir or --models-preset.
    const bool artifact_given = std::string_view(argv[1]).substr(0, 2) != "--";
    if (artifact_given) { options.artifact_path = argv[1]; }
    // Router options apply to the router; every other argument is passed on to each model.
    const auto router_option = [](std::string_view arg) {
        return arg == "--models-dir" || arg == "--models-preset" || arg == "--models-max" ||
               arg == "--sleep-idle-seconds" || arg == "--unload-idle-seconds";
    };
    for (int i = artifact_given ? 2 : 1; i < argc; ++i) {
        const std::string_view current = argv[i];
        if (router_option(current)) {
            ++i;
            continue;
        }
        if (current == "--models-autoload" || current == "--no-models-autoload") { continue; }
        options.model_arguments.emplace_back(current);
    }
    for (int i = artifact_given ? 2 : 1; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        // One `if` per option, each ending in `continue`, not an `else if` chain: MSVC nests each
        // `else if` one block deeper and stops compiling at 128 (C1061).
        if (arg == "--host") {
            options.host = require_value("--host");
            continue;
        }
        if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
            continue;
        }
        if (arg == "--stats-port") {
            options.stats_port = parse_nonnegative_int(require_value("--stats-port"), "stats-port");
            continue;
        }
        if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
            continue;
        }
        if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
            continue;
        }
        if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
            continue;
        }
        if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
            continue;
            continue;
        }
        if (arg == "--kvmem-budget") {
            options.kvmem.selected_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-budget"), "kvmem-budget"));
            continue;
        }
        if (arg == "--kvmem-lrfu") { options.kvmem.lrfu = true; continue; }
        if (arg == "--kvmem-lrfu-lambda") {
            options.kvmem.lrfu_lambda = kvmem::parse_lrfu_real(require_value("--kvmem-lrfu-lambda"), true);
            continue;
        }
        if (arg == "--kvmem-lrfu-top-frac") {
            options.kvmem.lrfu_top_fraction = kvmem::parse_lrfu_real(require_value("--kvmem-lrfu-top-frac"), false);
            continue;
        }
        if (arg == "--kvmem-score-agg") {
            options.kvmem.score_agg_max = kvmem::parse_score_aggregation(require_value("--kvmem-score-agg")) == 1;
            continue;
        }
        if (arg == "--kvmem-gen-reserve") {
            options.kvmem.reserve_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-gen-reserve"), "kvmem-gen-reserve"));
            continue;
        }
        if (arg == "--kvmem-sink-tokens") {
            options.kvmem.sink_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-sink-tokens"), "kvmem-sink-tokens"));
            sink_tokens_explicit = true;
            continue;
        }
        if (arg == "--kvmem-recent-tokens") {
            options.kvmem.recent_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-recent-tokens"), "kvmem-recent-tokens"));
            recent_tokens_explicit = true;
            continue;
        }
        if (arg == "--kvmem-sessions") {
            options.kvmem.retained_sessions = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--kvmem-sessions"), "kvmem-sessions"));
            continue;
        }
        if (arg == "--kvmem-disk-path") {
            options.kvmem.disk_path = std::filesystem::u8path(require_value("--kvmem-disk-path"));
            if (options.kvmem.disk_path.empty())
                throw std::invalid_argument("--kvmem-disk-path must not be empty");
            continue;
        }
        if (arg == "--kvmem-disk-mib") {
            const auto mib = parse_u64(require_value("--kvmem-disk-mib"), "kvmem-disk-mib");
            if (!mib || mib > std::uint64_t(INT64_MAX) / (1ULL << 20))
                throw std::invalid_argument("--kvmem-disk-mib is out of range");
            options.kvmem.disk_bytes = mib * (1ULL << 20);
            continue;
        }
        if (arg == "--kvmem-host-mib") {
            const auto mib = parse_u64(require_value("--kvmem-host-mib"), "kvmem-host-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--kvmem-host-mib is out of range");
            }
            options.kvmem.host_bytes = static_cast<std::size_t>(mib * (1ULL << 20));
            continue;
        }
        if (arg == "--kvmem-verify-transfers") {
            options.kvmem.verify_transfers = true;
            continue;
        }
        if (arg == "--kv-headroom-mib" || arg == "--vram-headroom-mib") {
            // --vram-headroom-mib is the Wallawalla47 fork's name for the same headroom.
            const std::uint64_t mib = parse_u64(require_value(arg.c_str()), arg.c_str() + 2);
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument(arg + " is out of range");
            }
            kv_headroom_mib = static_cast<std::size_t>(mib);
            continue;
        }
        if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
            continue;
        }
        if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
            continue;
        }
        if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
            continue;
        }
        if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
            continue;
        }
        if (arg == "--fast-prefill-kernel") {
            options.fast_prefill_kernel = true;
            continue;
        }
        if (arg == "--device-profile") {
            options.device_profile = require_value("--device-profile");
            if (options.device_profile != "auto" && options.device_profile != "off" &&
                options.device_profile != "calibrate") {
                throw std::invalid_argument("--device-profile must be auto, off or calibrate");
            }
            continue;
        }
        if (arg == "--device-profile-path") {
            options.device_profile_path = require_value("--device-profile-path");
            if (options.device_profile_path.empty()) {
                throw std::invalid_argument("--device-profile-path must not be empty");
            }
            continue;
        }
        if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
            continue;
        }
        if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
            continue;
        }
        if (arg == "--log-colours") {
            const std::string_view value = require_value("--log-colours");
            if (value == "on") {
                options.log_colours = true;
            } else if (value == "off") {
                options.log_colours = false;
            } else {
                throw std::invalid_argument("--log-colours accepts on or off");
            }
            continue;
        }
        if (arg == "--log-stats-panel") {
            const std::string_view value = require_value("--log-stats-panel");
            if (value == "on") {
                options.log_stats_panel = true;
            } else if (value == "off") {
                options.log_stats_panel = false;
            } else {
                throw std::invalid_argument("--log-stats-panel accepts on or off");
            }
            continue;
        }
        if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
            continue;
        }
        if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
            continue;
        }
        if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
            continue;
        }
        if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
            continue;
        }
        if (arg == "--use-alt-prefix-caching") {
            options.context_cache.mode = ContextCacheMode::Hybrid;
            continue;
        }
        if (arg == "--use-original-prefix-caching") {
            original_cache_selected = true;
            continue;
        }
        if (arg == "--device-snapshot-slots") {
            options.context_cache.hybrid.device_snapshot_slots =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--device-snapshot-slots"), "device-snapshot-slots"));
            hybrid_option_flag = "--device-snapshot-slots";
            continue;
        }
        if (arg == "--cache-taps-per-request") {
            options.context_cache.hybrid.max_new_taps =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--cache-taps-per-request"), "cache-taps-per-request"));
            hybrid_option_flag = "--cache-taps-per-request";
            continue;
        }
        if (arg == "--cache-tap-ladder") {
            options.context_cache.hybrid.tap_ladder_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-ladder"), "cache-tap-ladder"));
            hybrid_option_flag = "--cache-tap-ladder";
            continue;
        }
        if (arg == "--prefix-cache-file") {
            options.context_cache.hybrid.persistent_file = require_value("--prefix-cache-file");
            if (options.context_cache.hybrid.persistent_file.empty()) {
                throw std::invalid_argument("--prefix-cache-file must not be empty");
            }
            hybrid_option_flag = "--prefix-cache-file";
            continue;
        }
        if (arg == "--cache-tap-min-gap") {
            options.context_cache.hybrid.tap_min_gap_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-min-gap"), "cache-tap-min-gap"));
            hybrid_option_flag = "--cache-tap-min-gap";
            continue;
        }
        if (arg == "--device-state-slots") {
            legacy_cache_flag                        = "--device-state-slots";
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
            continue;
        }
        if (arg == "--host-state-slots") {
            legacy_cache_flag                      = "--host-state-slots";
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
            host_state_slots_explicit = true;
            continue;
        }
        if (arg == "--host-kv-mib") {
            legacy_cache_flag       = "--host-kv-mib";
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
            host_kv_mib_explicit                         = true;
            continue;
        }
        if (arg == "--host-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--host-cache-mib"), "host-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-cache-mib is out of range");
            }
            options.context_cache.host_cache_budget_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                     = true;
            host_cache_budget_explicit                    = true;
            continue;
        }
        if (arg == "--disk-kv-path") {
            legacy_cache_flag                  = "--disk-kv-path";
            options.context_cache.disk_kv_path = require_value("--disk-kv-path");
            if (options.context_cache.disk_kv_path.empty()) {
                throw std::invalid_argument("--disk-kv-path must not be empty");
            }
            continue;
        }
        if (arg == "--disk-kv-gib") {
            const std::uint64_t gib = parse_u64(require_value("--disk-kv-gib"), "disk-kv-gib");
            if (gib == 0 || gib > (std::numeric_limits<std::uint64_t>::max() >> 30U)) {
                throw std::invalid_argument("--disk-kv-gib must be positive and in range");
            }
            options.context_cache.disk_kv_capacity_bytes = gib << 30U;
            continue;
        }
        if (arg == "--disk-kv-restore") {
            options.context_cache.disk_kv_restore = true;
            continue;
        }
        if (arg == "--disk-kv-directstorage") {
            options.context_cache.disk_kv_directstorage = true;
            continue;
        }
        if (arg == "--context-cache-policy") {
            legacy_cache_flag        = "--context-cache-policy";
            const std::string policy = require_value("--context-cache-policy");
            if (policy != "default" && policy != "rolling") {
                throw std::invalid_argument("--context-cache-policy must be default or rolling");
            }
            options.context_cache.rolling_retention = policy == "rolling";
            context_capacity_explicit               = true;
            continue;
        }
        if (arg == "--release-diverged-checkpoints") {
            legacy_cache_flag                                  = "--release-diverged-checkpoints";
            options.context_cache.release_diverged_checkpoints = true;
            context_capacity_explicit                          = true;
            continue;
        }
        if (arg == "--concurrent-prefill") {
            options.concurrent_prefill = true;
            continue;
        }
        if (arg == "--recover-invariant-failures") {
            options.recover_invariant_failures = true;
            continue;
        }
        if (arg == "--thorough-admission-search") {
            legacy_cache_flag                               = "--thorough-admission-search";
            options.context_cache.thorough_admission_search = true;
            context_capacity_explicit                       = true;
            continue;
        }
        if (arg == "--recency-eviction") {
            legacy_cache_flag                      = "--recency-eviction";
            options.context_cache.recency_eviction = true;
            context_capacity_explicit              = true;
            continue;
        }
        if (arg == "--value-aware-demote") {
            legacy_cache_flag                        = "--value-aware-demote";
            options.context_cache.value_aware_demote = true;
            context_capacity_explicit                = true;
            continue;
        }
        if (arg == "--kv-lease-growth") {
            options.context_cache.kv_lease_growth = true;
            context_capacity_explicit             = true;
            continue;
        }
        if (arg == "--max-private-continuations") {
            legacy_cache_flag = "--max-private-continuations";
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
            continue;
        }
        if (arg == "--max-shared-prefixes") {
            legacy_cache_flag = "--max-shared-prefixes";
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
            continue;
        }
        if (arg == "--max-long-anchors-per-continuation") {
            legacy_cache_flag = "--max-long-anchors-per-continuation";
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
            continue;
        }
        if (arg == "--auto-long-anchors") {
            legacy_cache_flag                            = "--auto-long-anchors";
            options.context_cache.automatic_long_anchors = true;
            context_capacity_explicit                    = true;
            continue;
        }
        if (arg == "--branch-anchors") {
            legacy_cache_flag                    = "--branch-anchors";
            options.context_cache.branch_anchors = true;
            continue;
        }
        if (arg == "--no-endpoint-anchors") {
            legacy_cache_flag                      = "--no-endpoint-anchors";
            options.context_cache.endpoint_anchors = false;
            continue;
        }
        if (arg == "--long-anchor-spacing") {
            legacy_cache_flag                                    = "--long-anchor-spacing";
            options.context_cache.long_anchor_min_spacing_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--long-anchor-spacing"),
                                      "long-anchor-spacing"));
            long_anchor_spacing_explicit = true;
            continue;
        }
        if (arg == "--max-cache-markers-per-request") {
            options.context_cache.max_cache_markers_per_request = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-cache-markers-per-request"),
                                      "max-cache-markers-per-request"));
            context_capacity_explicit = true;
            continue;
        }
        if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
            continue;
        }
        if (arg == "--request-log-max-mib") {
            options.request_log_max_mib = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--request-log-max-mib"), "request-log-max-mib"));
            continue;
        }
        if (arg == "--request-log-keep") {
            options.request_log_keep = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--request-log-keep"), "request-log-keep"));
            continue;
        }
        if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
            continue;
        }
        if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
            continue;
        }
        if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
            device_explicit = true;
            continue;
        }
        if (arg == "--devices") {
            options.devices = parse_device_list(require_value("--devices"));
            continue;
        }
        if (arg == "--expert-residency") {
            const std::string residency = require_value("--expert-residency");
            if (residency == "device") {
                options.expert_residency = ExpertResidency::Device;
            } else if (residency == "host") {
                options.expert_residency = ExpertResidency::Host;
            } else if (residency == "disk") {
                options.expert_residency = ExpertResidency::Disk;
            } else {
                throw std::invalid_argument("--expert-residency must be device, host or disk");
            }
            continue;
        }
        if (arg == "--expert-cache-mib") {
            const std::string mib = require_value("--expert-cache-mib");
            if (mib == "auto") {
                options.expert_cache_bytes.reset();
            } else {
                options.expert_cache_bytes =
                    std::uint64_t(parse_nonnegative_int(mib.c_str(), "expert-cache-mib")) << 20;
            }
            continue;
        }
        if (arg == "--ngram-table") {
            options.ngram_table.path = require_value("--ngram-table");
            continue;
        }
        if (arg == "--ngram-ram") {
            options.ngram_table.ram = true;
            continue;
        }
        if (arg == "--no-ngram-table") {
            options.ngram_table.disabled = true;
            continue;
        }
        if (arg == "--stage-layers") {
            options.stage_layers = parse_stage_layers(require_value("--stage-layers"));
            continue;
        }
        if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
            continue;
        }
        if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
            continue;
        }
        if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
            continue;
        }
        if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-draft-tokens"), "ngram-draft-tokens"));
            ngram_width_explicit = true;
            continue;
        }
        if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-min-match"), "ngram-min-match"));
            continue;
        }
        if (arg == "--ngram-archive-mib" || arg == "--ngram-session-mib") {
            const auto mib = parse_u64(require_value(arg.c_str()), arg.c_str());
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("ngram archive capacity is out of range");
            }
            auto& bytes = arg == "--ngram-archive-mib" ? options.speculative.ngram_archive_bytes
                                                       : options.speculative.ngram_session_bytes;
            bytes       = static_cast<std::size_t>(mib << 20);
            continue;
        }
        if (arg == "--ngram-native-sessions") {
            options.ngram_native_sessions = true;
            continue;
        }
        if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            continue;
        }
        if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
            continue;
        }
        if (arg == "--default-reasoning-effort") {
            const std::string value = require_value("--default-reasoning-effort");
            const auto effort       = parse_requested_reasoning_effort(value);
            if (!effort) {
                throw std::invalid_argument("--default-reasoning-effort must be none, minimal, low, "
                                            "medium, high, xhigh, or max");
            }
            options.default_reasoning_effort = *effort;
            continue;
        }
        if (arg == "--thinking-budget-message") {
            options.thinking_budget_message = require_value("--thinking-budget-message");
            if (options.thinking_budget_message.empty()) {
                throw std::invalid_argument("--thinking-budget-message must not be empty");
            }
            continue;
        }
        if (arg == "--vision") {
            options.enable_vision = true;
            continue;
        }
        if (arg == "--vision-residency") {
            const std::string_view mode = require_value("--vision-residency");
            if (mode == "resident") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "overlay") {
                options.vision_residency = VisionResidency::Overlay;
            } else if (mode == "cpu") {
                options.vision_residency = VisionResidency::Cpu;
            } else {
                throw std::invalid_argument("--vision-residency must be resident, overlay or cpu");
            }
            continue;
        }
        if (arg == "--vision-cpu") {
            // The gzenz fork's switch for Vision on CPU threads.
            options.enable_vision    = true;
            options.vision_residency = VisionResidency::Cpu;
            continue;
        }
        if (arg == "--vision-offload") {
            // The Wallawalla47 fork's switch for the same overlay residency.
            const std::string_view mode = require_value("--vision-offload");
            if (mode == "off") {
                options.vision_residency = VisionResidency::Resident;
            } else if (mode == "on") {
                options.vision_residency = VisionResidency::Overlay;
            } else {
                throw std::invalid_argument("--vision-offload must be on or off");
            }
            continue;
        }
        if (arg == "--vision-max-merged") {
            const std::uint64_t merged =
                parse_u64(require_value("--vision-max-merged"), "vision-max-merged");
            if (merged < 64 || merged > 16384) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 16384]");
            }
            options.vision_max_merged_tokens = static_cast<std::uint32_t>(merged);
            vision_max_merged_explicit       = true;
            continue;
        }
        if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
            continue;
        }
        if (arg == "--cuda-graph-allowance-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--cuda-graph-allowance-mib"), "cuda-graph-allowance-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--cuda-graph-allowance-mib is out of range");
            }
            options.cuda_graph_allowance_mib = mib;
            continue;
        }
        if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
            continue;
        }
        if (arg == "--lenient-assistant-history") {
            options.lenient_assistant_history = true;
            continue;
        }
        if (arg == "--derive-session-keys") {
            // The lineage hint also feeds KVMem histories.
            options.derive_session_keys = true;
            continue;
        }
        if (arg == "--slot-save-path") {
            // Slots are the checkpoint catalog's private continuation cells.
            legacy_cache_flag      = "--slot-save-path";
            options.slot_save_path = require_value("--slot-save-path");
            if (options.slot_save_path.empty()) {
                throw std::invalid_argument("--slot-save-path must not be empty");
            }
            continue;
        }
        if (arg == "--auto-save-evicted") {
            legacy_cache_flag         = "--auto-save-evicted";
            options.auto_save_evicted = true;
            continue;
        }
        if (arg == "--auto-prefix-grid") {
            legacy_cache_flag        = "--auto-prefix-grid";
            options.auto_prefix_grid = true;
            continue;
        }
        if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
            continue;
        }
        if (arg == "--full-draft-head") {
            options.speculative.proposal_head = ProposalHead::Full;
            continue;
        }
        if (arg == "--adaptive-mtp") {
            options.speculative.mtp_policy = MtpDraftPolicy::Adaptive;
            continue;
        }
        if (arg == "--mtp-attention-window") {
            options.speculative.mtp_attention_window =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--mtp-attention-window"), "mtp-attention-window"));
            continue;
        }
        if (arg == "--lm-head-q4") {
            options.lm_head_q4 = true;
            continue;
        }
        if (arg == "--lm-head-q6") {
            options.lm_head_q6 = true;
            continue;
        }
        if (arg == "--embedding-q4") {
            options.embedding_q4 = true;
            continue;
        }
        if (arg == "--embedding-q6") {
            options.embedding_q6 = true;
            continue;
        }
        if (arg == "--mtp-experts-q4") {
            options.mtp_experts_q4 = true;
            continue;
        }
        if (arg == "--gdn-state-fp16") {
            options.gdn_state_fp16 = true;
            continue;
        }
        if (arg == "--rope-yarn") {
            options.rope_yarn = true;
            continue;
        }
        if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor =
                product::parse_rope_yarn_factor(require_value("--rope-yarn-factor"));
            continue;
        }
        if (arg == "--rope-scaling-factor") {
            options.rope_scaling_factor =
                product::parse_rope_scaling_factor(require_value("--rope-scaling-factor"));
            continue;
        }
        if (arg == "--rope-scaling-original-context") {
            options.rope_scaling_original_context = product::parse_rope_scaling_original_context(
                require_value("--rope-scaling-original-context"));
            continue;
        }
        if (arg == "--cuda-memory-policy") {
            const auto policy = product::parse_cuda_memory_options(require_value(arg.c_str()));
            options.cuda_memory_policy = policy.policy;
            options.cuda_vram_reserve_bytes = policy.reserve_bytes;
            options.cuda_memory_probe_step_bytes = policy.probe_step_bytes;
            continue;
        }
        if (arg == "--wddm-evictable-budget") {
            options.wddm_evictable_budget = true;
            continue;
        }
        if (arg == "--model-suspend") {
            options.suspend.enabled = true;
            continue;
        }
        if (arg == "--models-dir") {
            options.models_dir = require_value("--models-dir");
            continue;
        }
        if (arg == "--models-preset") {
            options.models_preset = require_value("--models-preset");
            continue;
        }
        if (arg == "--models-max") {
            options.models_max = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--models-max"), "models-max"));
            continue;
        }
        if (arg == "--models-autoload") {
            options.models_autoload = true;
            continue;
        }
        if (arg == "--no-models-autoload") {
            options.models_autoload = false;
            continue;
        }
        if (arg == "--sleep-idle-seconds") {
            options.sleep_idle_seconds = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--sleep-idle-seconds"), "sleep-idle-seconds"));
            continue;
        }
        if (arg == "--unload-idle-seconds") {
            options.unload_idle_seconds = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--unload-idle-seconds"), "unload-idle-seconds"));
            continue;
        }
        if (arg == "--suspend-snapshot") {
            const std::string value = require_value("--suspend-snapshot");
            if (value == "pageable") {
                options.suspend.snapshot_memory = SuspendSnapshotMemory::Pageable;
            } else if (value == "pinned") {
                options.suspend.snapshot_memory = SuspendSnapshotMemory::Pinned;
            } else {
                throw std::invalid_argument("--suspend-snapshot must be pageable or pinned");
            }
            continue;
        }
        if (arg == "--suspend-weights") {
            const std::string value = require_value("--suspend-weights");
            if (value == "artifact") {
                options.suspend.weights = SuspendWeightSource::Artifact;
            } else if (value == "host") {
                options.suspend.weights = SuspendWeightSource::Host;
            } else {
                throw std::invalid_argument("--suspend-weights must be artifact or host");
            }
            continue;
        }
        if (arg == "--no-auto-resume") {
            options.suspend.auto_resume = false;
            continue;
        }
        if (arg == "--mlp-a8-decode") {
            options.mlp_a8_decode = true;
            continue;
        }
        if (arg == "--no-prefill-a8") {
            options.prefill_a8 = false;
            continue;
        }
        if (arg == "--lookup-ngram") {
            options.speculative.lookup_ngram = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--lookup-ngram"), "lookup-ngram"));
            continue;
        }
        if (arg == "--prefill-cublas") {
            options.prefill_cublas = true;
            continue;
        }
        if (arg == "--no-prefill-cublas-projections") {
            options.prefill_cublas_projections = false;
            continue;
        }
        if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
            if (options.chat_template_path.empty()) {
                throw std::invalid_argument("--chat-template must not be empty");
            }
            continue;
        }
        if (arg == "--no-thinking") {
            options.enable_thinking = false;
            continue;
        }
        if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
            continue;
        }
        if (arg == "--graft") {
            const std::string_view spec = require_value("--graft");
            const std::size_t equals    = spec.find('=');
            if (equals == 0 || equals == std::string_view::npos || equals + 1 == spec.size()) {
                throw std::invalid_argument("--graft must be NAME=PATH");
            }
            options.grafts.push_back(GraftSource{.name = std::string(spec.substr(0, equals)),
                                                 .path = std::string(spec.substr(equals + 1))});
            continue;
        }
        if (arg == "--default-graft") {
            options.default_graft = require_value("--default-graft");
            if (options.default_graft.empty()) {
                throw std::invalid_argument("--default-graft needs a graft name");
            }
            continue;
        }
        if (arg == "--cors") {
            options.enable_cors = true;
            continue;
        }
        if (arg == "--no-webui") {
            options.enable_webui = false;
            continue;
        }
        if (arg == "--webui-mcp-proxy") {
            options.webui_mcp_proxy = true;
            continue;
        }
        if (arg == "--structured-output") {
            options.structured_output = true;
            continue;
        }
        if (arg == "--unconstrained-response-format") {
            options.unconstrained_response_format = true;
            continue;
        }
        if (arg == "--assistant-prefill") {
            options.assistant_prefill = true;
            continue;
        }
        if (arg == "--usage-chunk-choice") {
            options.usage_chunk_choice = true;
            continue;
        }
        if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
            continue;
        }
        if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
            continue;
        }
        if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
            continue;
        }
        if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
            continue;
        }
        if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
            continue;
        }
        if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
            continue;
        }
        if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
            continue;
        }
        if (arg == "--greedy") {
            options.greedy = true;
            continue;
        }
        if (arg == "--post-thinking") {
            if (!options.post_thinking_overrides) { options.post_thinking_overrides.emplace(); }
            continue;
        }
        if (arg == "--post-thinking-temperature" || arg == "--post-thinking-top-p" ||
            arg == "--post-thinking-top-k") {
            const std::string_view key = arg == "--post-thinking-temperature" ? "temp"
                                         : arg == "--post-thinking-top-p"     ? "top_p"
                                                                              : "top_k";
            if (!options.post_thinking_overrides) { options.post_thinking_overrides.emplace(); }
            product::set_post_thinking_field(*options.post_thinking_overrides, key,
                                             require_value(arg.c_str()));
            continue;
        }
        if (arg == "--post-thinking-sampler") {
            if (!options.post_thinking_overrides) { options.post_thinking_overrides.emplace(); }
            product::apply_post_thinking_sampler(require_value("--post-thinking-sampler"),
                                                 *options.post_thinking_overrides);
            continue;
        }
        if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
            continue;
        }
        throw std::invalid_argument("unknown argument: " + arg);
    }
    if (!options.default_graft.empty() &&
        std::none_of(options.grafts.begin(), options.grafts.end(), [&](const GraftSource& source) {
            return source.name == options.default_graft;
        })) {
        throw std::invalid_argument("--default-graft '" + options.default_graft +
                                    "' does not name a --graft");
    }
    if (options.kvmem.selected_tokens != 0 &&
        (options.suspend.enabled || !options.grafts.empty() || options.router())) {
        throw std::invalid_argument("KVMem does not support model suspend, prompt grafts or model router mode");
    }
    // Resolve policy-implied cache mode before choosing the KV default or validating cache flags.
    if (options.cuda_memory_policy == CudaMemoryPolicy::StrictVram) {
        if (original_cache_selected || !options.allow_prefix_reuse) {
            throw std::invalid_argument(
                "--cuda-memory-policy strict enables hybrid prefix caching and conflicts with "
                "--use-original-prefix-caching and --no-prefix-reuse");
        }
        options.context_cache.mode = ContextCacheMode::Hybrid;
    } else {
        options.cuda_vram_reserve_bytes = 0;
        options.cuda_memory_probe_step_bytes = 0;
    }
    if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault &&
        (options.wddm_evictable_budget || options.enable_vision || options.devices.size() > 1)) {
        throw std::invalid_argument(
            "--cuda-memory-policy mixed/strict requires one GPU, text-only inference and "
            "ordinary CUDA allocation (no --wddm-evictable-budget)");
    }
    if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault &&
        (options.suspend.enabled || !options.grafts.empty() || options.router())) {
        throw std::invalid_argument("--cuda-memory-policy mixed/strict does not support model suspend, prompt grafts or the model router");
    }
    if (!kv_capacity_explicit) {
        // The hybrid cache turns every Device page no active request holds into block cache, so
        // it sizes the KV pool to free VRAM unless a capacity is given.
        options.kv_capacity = options.context_cache.mode == ContextCacheMode::Hybrid
                                  ? KvCapacityPolicy::automatic()
                                  : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (original_cache_selected && options.context_cache.mode == ContextCacheMode::Hybrid) {
        throw std::invalid_argument(
            "--use-original-prefix-caching and --use-alt-prefix-caching select different prefix "
            "caching systems");
    }
    if (options.context_cache.mode == ContextCacheMode::Hybrid) {
        if (legacy_cache_flag != nullptr || options.derive_session_keys) {
            throw std::invalid_argument(std::string(legacy_cache_flag ? legacy_cache_flag : "--derive-session-keys") +
                                        " configures the default prefix cache and cannot be "
                                        "combined with --use-alt-prefix-caching");
        }
        if (!options.allow_prefix_reuse) {
            throw std::invalid_argument(
                "--use-alt-prefix-caching cannot be combined with --no-prefix-reuse");
        }
        if (options.devices.size() > 1) {
            throw std::invalid_argument(
                "--use-alt-prefix-caching runs on one device and cannot be combined with "
                "pipeline --devices");
        }
        std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        if (!file.empty()) {
            if (host_cache_budget_explicit && options.context_cache.host_cache_budget_bytes == 0) {
                throw std::invalid_argument(
                    "--prefix-cache-file saves the Host tier, which --host-cache-mib 0 removes");
            }
            // Resolved now, so the save at shutdown writes where startup read, and checked now,
            // so an unusable location fails at launch rather than after a session of caching.
            file = std::filesystem::absolute(file).lexically_normal();
            std::error_code error;
            if (std::filesystem::is_directory(file, error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            " is a directory; name a file in it");
            }
            if (!std::filesystem::is_directory(file.parent_path(), error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            ": the directory " + file.parent_path().string() +
                                            " does not exist");
            }
        }
    } else if (hybrid_option_flag != nullptr) {
        throw std::invalid_argument(std::string(hybrid_option_flag) +
                                    " requires --use-alt-prefix-caching");
    }
    if (kv_headroom_mib.has_value()) {
        if (options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault) {
            throw std::invalid_argument("--kv-headroom-mib is available only with --cuda-memory-policy default");
        }
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument("--kv-headroom-mib requires --kv-capacity auto");
        }
        options.kv_capacity = KvCapacityPolicy::automatic(*kv_headroom_mib << 20);
    } else if (options.kv_capacity.mode == KvCapacityMode::Automatic &&
               options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault) {
        options.kv_capacity = KvCapacityPolicy::automatic(options.cuda_vram_reserve_bytes);
    }
    if (long_anchor_spacing_explicit && !options.context_cache.automatic_long_anchors) {
        throw std::invalid_argument("--long-anchor-spacing requires --auto-long-anchors");
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        if (options.auto_prefix_grid) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --auto-prefix-grid");
        }
        if (options.derive_session_keys) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --derive-session-keys");
        }
        if (!options.slot_save_path.empty()) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with --slot-save-path");
        }
        if (!options.context_cache.disk_kv_path.empty()) {
            throw std::invalid_argument("--no-prefix-reuse cannot be combined with --disk-kv-path");
        }
        options.context_cache.enabled                = false;
        options.context_cache.host_state_slots       = 0;
        options.context_cache.host_kv_capacity_bytes = 0;
    }
    if (options.context_cache.disk_kv_path.empty() &&
        (options.context_cache.disk_kv_restore || options.context_cache.disk_kv_directstorage ||
         options.context_cache.disk_kv_capacity_bytes != 0)) {
        throw std::invalid_argument(
            "--disk-kv-restore, --disk-kv-directstorage and --disk-kv-gib need --disk-kv-path");
    }
    if (!options.devices.empty() && device_explicit) {
        throw std::invalid_argument("--device and --devices are mutually exclusive");
    }
    if (!options.stage_layers.empty() && options.devices.size() < 2) {
        throw std::invalid_argument("--stage-layers needs --devices naming more than one device");
    }
    if (options.request_log_max_mib != 0 && options.request_log_jsonl.empty()) {
        throw std::invalid_argument("--request-log-max-mib requires --request-log-jsonl");
    }
    if (options.auto_save_evicted && options.slot_save_path.empty()) {
        throw std::invalid_argument("--auto-save-evicted requires --slot-save-path");
    }
    if (host_cache_budget_explicit) {
        // The budget is the one host RAM ceiling; the two component flags would silently
        // fight it, and their independent-allocation semantics are exactly what the budget
        // exists to replace.
        if (host_state_slots_explicit || host_kv_mib_explicit) {
            throw std::invalid_argument(
                "--host-cache-mib cannot be combined with --host-state-slots or --host-kv-mib: "
                "the budget derives both Host state slots and Host KV bytes");
        }
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.stats_port != 0 &&
        (options.stats_port > 65535 || options.stats_port == options.port)) {
        throw std::invalid_argument("--stats-port must be in [1,65535] and differ from --port");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (!options.kvmem.selected_tokens && options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (options.unconstrained_response_format && options.structured_output) {
        throw std::invalid_argument(
            "--unconstrained-response-format conflicts with --structured-output");
    }
    product::apply_default_ngram_draft_tokens(options.speculative, ngram_width_explicit);
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_residency != VisionResidency::Resident && !options.enable_vision) {
        throw std::invalid_argument("--vision-residency overlay or cpu requires --vision");
    }
    // A CPU encode grows with the square of an item's patches: 256 merged tokens (about 512x512
    // pixels) keeps one to a few seconds.
    if (options.vision_residency == VisionResidency::Cpu && !vision_max_merged_explicit) {
        options.vision_max_merged_tokens = 256;
    }
    if (options.enable_thinking == false && options.default_reasoning_effort &&
        *options.default_reasoning_effort != RequestedReasoningEffort::None) {
        throw std::invalid_argument("--default-reasoning-effort conflicts with --no-thinking");
    }
    if (options.cuda_graph_allowance_mib != 0 && !options.use_cuda_graph) {
        throw std::invalid_argument(
            "--cuda-graph-allowance-mib requires CUDA graphs (omit --no-cuda-graph)");
    }
    const auto effective_ngram = runtime::concurrent_ngram_draft_tokens(
        options.speculative.ngram_draft_tokens, options.max_concurrency);
    if (effective_ngram != options.speculative.ngram_draft_tokens) {
        options.requested_ngram_draft_tokens = options.speculative.ngram_draft_tokens;
        options.speculative.ngram_draft_tokens = effective_ngram;
    }
    if (options.ngram_native_sessions && options.speculative.ngram_archive_bytes == 0) {
        throw std::invalid_argument("--ngram-native-sessions requires --ngram-archive-mib");
    }
    if (options.default_max_tokens && *options.default_max_tokens <= 0) {
        throw std::invalid_argument("--default-max-tokens must be positive");
    }
    if (!artifact_given && !options.router()) {
        throw std::invalid_argument(
            "artifact path is required (or --models-dir / --models-preset for router mode)");
    }
    if (artifact_given && options.router()) {
        throw std::invalid_argument(
            "router mode takes its models from --models-dir / --models-preset, not an artifact "
            "path");
    }
    if (options.sleep_idle_seconds != 0 && !options.suspend.enabled && !options.router()) {
        throw std::invalid_argument("--sleep-idle-seconds requires --model-suspend");
    }
    if (options.kvmem.selected_tokens) {
        if (options.speculative.lookup_ngram != 0) {
            throw std::invalid_argument(
                "KVMem disables legacy --lookup-ngram; use --ngram-draft-tokens and --ngram-min-match");
        }
        const auto window = options.kvmem.device_tokens();
        const bool ordinary = options.speculative.backend == SpeculativeBackend::None &&
            options.speculative.draft_tokens == 0;
        const bool memory_mtp = options.speculative.backend == SpeculativeBackend::Mtp &&
            (options.speculative.draft_tokens >= 1 && options.speculative.draft_tokens <= 15);
        const bool fixed_mtp = memory_mtp && options.speculative.mtp_policy == MtpDraftPolicy::Fixed;
        const bool adaptive_mtp = memory_mtp &&
            options.speculative.mtp_policy == MtpDraftPolicy::Adaptive;
        // DFlash and DFlash2 keep their conditioning KV in the same backend page store MTP
        // uses (ProgramImpl::backend_kv_cache() returns dflash->full), so the KVMem window can
        // carry it through a checkpoint. Note that backend_kv_valid() deliberately reports 0
        // for DFlash2 -- the standard context cache does not reuse a DFlash2 backend frontier
        // -- so the window tracks it through its own accessor rather than changing that
        // contract. Opening this gate is EXPERIMENTAL: block-sparse selection against a masked
        // draft computation has not been measured.
        const bool memory_dflash =
            (options.speculative.backend == SpeculativeBackend::DFlash ||
             options.speculative.backend == SpeculativeBackend::DFlash2) &&
            (options.speculative.draft_tokens >= 1 && options.speculative.draft_tokens <= 15);
        // Any drafter can carry ngram copy proposals; only the draft window bounds them.
        const bool memory_drafter = memory_mtp || memory_dflash;
        const bool memory_ngram_supported = options.speculative.ngram_draft_tokens == 0 ||
            (memory_drafter && (options.max_concurrency == 1 || options.speculative.ngram_draft_tokens <= 15) &&
             options.speculative.ngram_draft_tokens <= 63 &&
             options.speculative.ngram_min_match >= 4 && options.speculative.ngram_min_match <= 64 &&
             std::max(options.speculative.draft_tokens, options.speculative.ngram_draft_tokens) +
                 options.speculative.draft_tokens <= options.kvmem.reserve_tokens);
        const bool memory_spec_supported =
            (ordinary || fixed_mtp || adaptive_mtp || memory_dflash) && memory_ngram_supported &&
            options.speculative.mtp_attention_window == 0;

        if (legacy_cache_flag || hybrid_option_flag || original_cache_selected ||
            options.context_cache.mode != ContextCacheMode::Legacy ||
            options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency ||
            (options.enable_vision && options.vision_residency != VisionResidency::Resident &&
                options.vision_residency != VisionResidency::Cpu) ||
            !memory_spec_supported || options.concurrent_prefill ||
            (options.fast_prefill_kernel &&
                (options.kv_cache != KvCacheStorage::Int8Group64 &&
                 options.kv_cache != KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 &&
                 options.kv_cache != KvCacheStorage::RotatedLloyd4KeyInt4Value &&
                 options.kv_cache != KvCacheStorage::RotatedInt4KeyInt4ValueE8 &&
                 options.kv_cache != KvCacheStorage::RotatedE8RootKeyInt4Value)) || options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault ||
            (options.kv_cache != KvCacheStorage::BFloat16 && options.kv_cache != KvCacheStorage::Int8Group64 &&
             options.kv_cache != KvCacheStorage::Nvfp4Group16 &&
             options.kv_cache != KvCacheStorage::Fp8KeyNvfp4Value &&
             options.kv_cache != KvCacheStorage::RotatedInt8KeyInt4ValueGroup64 &&
             options.kv_cache != KvCacheStorage::Fp8E4M3Row256 &&
             options.kv_cache != KvCacheStorage::RotatedLloyd4KeyInt4Value &&
             options.kv_cache != KvCacheStorage::RotatedInt4KeyInt4ValueE8 &&
             options.kv_cache != KvCacheStorage::RotatedE8RootKeyInt4Value)) {
            throw std::invalid_argument("KVMem requires ordinary/fixed/adaptive MTP1..15, ngram1..63 (automatically capped at 15 with 2..8 lanes), single-GPU or layer-pipeline BF16/INT8/FP8/NVFP4/K8V4/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 generation, 1..8 text or resident/CPU-vision lanes, INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 fast prefill, "
                "and exclusive ownership of history caching");
        }
        if (options.kvmem.recent_blocks() == 0) {
            throw std::invalid_argument(
                "KVMem --kvmem-recent-tokens must be at least 64 to retain the partial append page");
        }
        if (static_cast<std::uint64_t>(options.kvmem.sink_blocks()) + options.kvmem.recent_blocks() >
            options.kvmem.selected_tokens / 64) {
            throw std::invalid_argument(
                "KVMem sink and recent tokens exceed --kvmem-budget after rounding to 64-token pages");
        }
        if (options.kvmem.selected_tokens < 128 || options.kvmem.selected_tokens % 64 ||
            options.kvmem.reserve_tokens < 64 || options.kvmem.reserve_tokens % 64 ||
            window < 256 || window > options.max_context || !options.kvmem.host_bytes ||
            options.kvmem.retained_sessions == 0 || options.kvmem.retained_sessions > 16 ||
            options.prefill_chunk > options.kvmem.reserve_tokens ||
            (kv_capacity_explicit && (options.kv_capacity.mode != KvCapacityMode::Explicit ||
                                     options.kv_capacity.explicit_tokens != window * options.max_concurrency))) {
            throw std::invalid_argument("KVMem requires aligned B/R, nonzero H, prefill <= R, "
                                        "and device capacity B+R <= max-context");
        }
        if (window * options.max_concurrency > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("KVMem global device capacity exceeds uint32");
        }
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(window * options.max_concurrency));
        // --no-prefix-reuse still controls request reuse. Only the native catalog
        // is disabled here; KVMem owns the current session's trusted checkpoint.
        options.context_cache.enabled = false;
    } else if (options.kvmem.reserve_tokens || options.kvmem.host_bytes || options.kvmem.verify_transfers ||
         !options.kvmem.disk_path.empty() || options.kvmem.disk_bytes ||
         sink_tokens_explicit || recent_tokens_explicit) {
        throw std::invalid_argument("KVMem auxiliary options require --kvmem-budget");
    }
    if (options.kvmem.disk_path.empty() != (options.kvmem.disk_bytes == 0) ||
        options.kvmem.host_bytes > UINT64_MAX - options.kvmem.disk_bytes)
        throw std::invalid_argument("KVMem SSD requires both --kvmem-disk-path and --kvmem-disk-mib");
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace ninfer::serve
