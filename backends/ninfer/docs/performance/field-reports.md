# Field reports

Measurements that users took on their own machines and sent as GitHub issues. Each report keeps
its reporter's setup, method and caveats. The methods differ from the
[project's methods](../performance.md), so these numbers are not read against the project's tables
or against each other as a benchmark. [Send a report](#send-a-report) to add a card or a workload.

| Card | OS | Model | Workload | Report |
|---|---|---|---|---|
| RTX 5070 Ti, 16 GB | Windows 11 | Qwen3.8-27B GSQ-RCO IQ3_S, Swift 1.5 IQ3_S | single requests to 104K tokens, against llama.cpp | [RTX 5070 Ti](#rtx-5070-ti-windows-11) |
| RTX 4090, 24 GB | Windows | Qwen3.8-27B GSQ-RCO IQ3_S, Swift 1.5 IQ3_S | Qwen Code sessions with DFlash2 5 and 7 | [RTX 4090: DFlash2](#rtx-4090-qwen-code-with-dflash2) |
| RTX 4090, 24 GB | Windows | Swift Qwen3.8 GSQ-RCO IQ3_S | agent traffic with MTP 3, four lanes and the hybrid prefix cache | [RTX 4090: MTP 3](#rtx-4090-agent-traffic-with-mtp-3-and-the-hybrid-prefix-cache) |
| RTX 4080 and RTX 4090 | Windows 11 | Qwen3.8-27B GSQ-RCO IQ3_S | Hermes Desktop | [RTX 4080 and 4090](#rtx-4080-and-rtx-4090-windows-11) |

## RTX 5070 Ti, Windows 11

Reported by @NickWalters in
[iamwavecut/ninfer-all#111](https://github.com/iamwavecut/ninfer-all/issues/111), 20 October 2026.

- RTX 5070 Ti 16 GB at its default 300 W limit, driver 610.47. The card also drives one 3440x1440
  desktop, which held about 1.3 GB of its memory at idle.
- Windows 11 Pro (build 26200), Ryzen 7 9800X3D, 32 GB of RAM.
- `master` at `c04ea31`, a `120a` build with CUDA 13.4 and Visual Studio 2026 Build Tools from
  `scripts\build-native.bat`.
- ISTA-DASLab `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, converted with
  `--recipe qwen3_8_27b_gguf --components text,mtp`.

The MTP rows:

```powershell
ninfer-serve.exe qwen3_8_27b_gsq_rco_iq3_s.noprop.ninfer `
  --model-id qwen3.8-27b --max-concurrency 1 `
  --gdn-state-fp16 --kv-dtype rk4v4-e8 --host-cache-mib 0 `
  --max-context 98304 --kv-capacity auto --kv-headroom-mib 192 `
  --spec mtp --draft-tokens 3
```

The rows without a drafter replace the last two lines with
`--max-context 122880 --kv-capacity 122880 --spec none`. llama.cpp is the b10930 CUDA 13.3 release
build with the same GGUF, `-np 1`, a 120K window, a `q4_0` KV cache and no MTP.

One request at a time, 512 output tokens, temperature 0. The prompt is source code from this
repository followed by "Summarize what this code does in detail." Prompt rate is prompt tokens over
the time to the first token; decode excludes the first token. Each cell is one run, prompt / decode
in tok/s:

| Prompt tokens | NInfer, MTP 3, 96K window | NInfer, no drafter, 120K window | llama.cpp, no MTP, 120K window |
|---:|---:|---:|---:|
| 1.2K | 1,908 / 133.1 | 1,977 / 57.7 | 1,288 / 51.3 |
| 16.8K | 2,037 / 113.9 | 2,058 / 56.4 | 1,581 / 46.1 |
| 33.4K | 1,924 / 112.6 | 1,945 / 55.2 | 1,387 / 42.2 |
| 56.1K | 1,798 / 110.7 | 1,805 / 53.3 | 1,172 / 38.2 |
| 86.0K | 1,650 / 102.8 | | |
| 104.2K | | 1,584 / 50.7 | 899 / 31.0 |

**Memory.** Peak use in `nvidia-smi`, desktop included, was 15,600 MiB with MTP and 15,443 to
15,463 MiB without. With `rk4v4-e8` the largest windows that started were 98,304 tokens with MTP 3
and 122,880 without a drafter.

**Swift 1.5.** UkisAI's `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` is a Qwen3.8-27B fine-tune
whose config, tokenizer and chat template are byte-identical to Qwen's. It converts with the same
recipe and the Qwen base folder as `--model` (846 objects, 39 s on the CPU) and serves with the same
flags. In the reporter's eval of 10 API tasks and 2 agent tasks, 3 runs each, both models passed 34
of 36. Swift used 17% fewer tokens, and on the API tasks it averaged 125 output tok/s against 114
for the stock model (completion tokens over wall time).

**Conversion without CUDA.** A PyTorch without CUDA converts with `--device cpu`; the converter's
error message names the option.

## RTX 4090: Qwen Code with DFlash2

Reported by @spk-c in [iamwavecut/ninfer-all#96](https://github.com/iamwavecut/ninfer-all/issues/96)
and its [follow-up](https://github.com/iamwavecut/ninfer-all/issues/96#issuecomment-6297428000),
17 to 20 October 2026: Qwen Code sessions with tool calls on an RTX 4090 (24 GB) under Windows.
Both setups below name `--derive-session-keys` and `--auto-long-anchors`, which select the
checkpoint catalog instead of the default hybrid prefix cache
([`--use-alt-prefix-caching`](../serving.md#server-options)).

### DFlash2 5, Qwen3.8-27B IQ3_S

The first session served Qwen3.8-27B GSQ-RCO IQ3_S with:

```powershell
.\build-ninja\apps\ninfer-serve.exe `
  ".\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer" `
  --model-id qwen3.8-27b --host 127.0.0.1 --port 8080 `
  --max-context 131072 --kv-capacity 131072 --kv-dtype rk8v4 `
  --gdn-state-fp16 --max-concurrency 1 `
  --spec dflash2 --draft-tokens 5 `
  --kv-lease-growth --derive-session-keys --auto-long-anchors `
  --vision --vision-residency overlay
```

The server log of 221 completed requests (prompts up to 101,389 tokens) gives:

| Measure | Value |
|---|---:|
| decode, median of the requests | 168 tok/s |
| decode, mean of the requests | 180.9 tok/s |
| decode, highest request (very high draft acceptance) | 484.4 tok/s |
| prompt rate, median | about 2,070 tok/s |
| prompt-cache hits, median | 94.6% |
| time to the first token, median | about 1.9 s |

Requests with more than 95% of the prompt in the cache began to answer in about a second; long
prompts with almost nothing reused sometimes waited more than 30 s, about what a cold prefill of
that length takes at these rates. The log held no `WARN`, `ERROR` or `FATAL` entry.

Against llama.cpp with a Q4_K_M quant, in separate tests: a short decode ran at 70.57 tok/s on
llama.cpp and 193.77 on NInfer, and a prompt at 2,347 and 2,828 tok/s. In long sessions llama.cpp
decoded about 40 tok/s, at most about 100, by the reporter's recollection. The quants, windows and
speculation differ, so this is no controlled comparison.

### DFlash2 7, Swift 1.5 IQ3_S

The later sessions used build `5ed86dff` (`89`, CUDA 13.4) with a locally calibrated RTX 4090
profile, Huihui's Qwen3.8-27B Swift 1.5 GSQ-RCO IQ3_S artifact with its DFlash2 drafter, Qwen
Code 0.25.3, a 176,128-token window, `rk4v4` with `--kv-capacity auto`, `--host-cache-mib 16384`,
`--max-concurrency 2`, `--derive-session-keys`, `--auto-long-anchors`, `--kv-lease-growth`,
`--gdn-state-fp16` and Vision overlay. Each column is one long session with its own task mix:

| Measure | DFlash2 5 | DFlash2 7 | DFlash2 7, cuBLAS prefill 2,048 |
|---|---:|---:|---:|
| coding requests | 387 | 309 | 226 |
| decode, median | 154.3 tok/s | 162.1 tok/s | 156.2 tok/s |
| decode, weighted by output | 146.6 tok/s | 174.8 tok/s | 148.4 tok/s |
| time to the first token, median | 1.50 s | 2.17 s | 1.37 s |
| prompt-cache hits, weighted | 93.78% | 91.96% | 93.39% |
| Engine recoveries | 0 | 0 | 0 |

- **Draft count.** In the reporter's long-output decode tests DFlash2 7 led DFlash2 5 throughout,
  210.3 against 184.3 tok/s at a 160K context.
- **cuBLAS prefill.** `--prefill-cublas --no-prefill-cublas-projections --prefill-chunk 2048`
  raised a synthetic 160K cold prefill with DFlash2 5 from 2,391.7 to 2,470.2 tok/s (+3.3%). The
  session with it showed no overall gain: its compaction prompts ran at about 2,320 tok/s against
  2,398 without it, on different prompts. The reporter keeps cuBLAS prefill off.
- **Compaction.** The DFlash2 7 session without cuBLAS ran 14 context compactions, about 23.7
  minutes in total, with a median time to the first token of about 54.4 s. Qwen Code's compaction
  rewrites most of the conversation, so the earlier prefix checkpoints no longer match it. Cache
  reuse otherwise stayed high above 100K prompt tokens.
- **MTP 3 with the hybrid prefix cache.** On the same card, a synthetic 100K test gave about 2,759
  tok/s cold prefill, 140.4 tok/s decode and a 0.088 s time to the first token on an exact repeat
  (99.99% cache hits). One real session gave a median decode of about 135.5 tok/s and 49.5%
  weighted cache reuse, with drops on growing or rewritten conversations. The drafter, model
  variant, settings and workloads differed from the DFlash2 sessions, so this does not isolate the
  cache.
- **Reasoning effort.** Qwen Code 0.25.3 with `"model": {"reasoningEffort": "medium"}` sent no
  effort: `requested_reasoning_effort` was `null` on all 561 requests of the two DFlash2 7
  sessions, and the server ran without `--default-reasoning-effort`. Such a request renders with
  the template's default effort, `xhigh` for the Qwen3.8 template. In one session of 4 h 42 min,
  1,416,104 of 1,712,734 output tokens (82.7%) were thinking, with 23 compactions.
  `--default-reasoning-effort medium` sets the effort of every request that names none
  ([reasoning effort](../serving.md#openai-chat-completions)).

## RTX 4090: agent traffic with MTP 3 and the hybrid prefix cache

Reported by @foxhoundhcm in
[iamwavecut/ninfer-all#96](https://github.com/iamwavecut/ninfer-all/issues/96#issuecomment-6251383234),
18 October 2026: an RTX 4090 (24 GB) under Windows, an `89` build of `4027aa63` with CUDA 13.4.3
(the cold-prefix table below ran on `bda41158`), a Swift Qwen3.8-27B GSQ-RCO IQ3_S artifact:

```powershell
ninfer-serve.exe swift-qwen38-rco-iq3s.ninfer `
  --host 127.0.0.1 --port 18006 --model-id qwen3.8-27b `
  --kv-dtype rk4v4 --spec mtp --draft-tokens 3 --mtp-attention-window 4096 --lm-head-draft `
  --max-context 262144 --default-max-tokens 32768 --kv-capacity auto --max-concurrency 4 `
  --use-alt-prefix-caching --host-cache-mib 6144 --concurrent-prefill --structured-output `
  --gdn-state-fp16 --vision --vision-residency overlay `
  --model-suspend --sleep-idle-seconds 900 --recover-invariant-failures --stats-port 18007 `
  --request-log-jsonl req.jsonl --request-log-max-mib 200 --request-log-keep 3
```

Cold prefixes, 256 output tokens, 3 runs per depth:

| Depth | Prompt tokens | Decode, tok/s | Prompt, tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| short | 145 | 187.5 | 1,423 | 0.724 |
| middle | 44,006 | 151.9 | 3,194 | 0.646 |
| deep | 133,007 | 135.8 | 2,533 | 0.603 |

From the request log of 7,127 agent requests (median prompt 87K tokens, largest 231K):

| Measure | Median | 90th percentile | Highest |
|---|---:|---:|---:|
| decode, tok/s | 123.6 | 176.0 | 482.9 |
| prompt rate of requests that computed 1K tokens or more, tok/s | 2,314 | 3,532 | 3,738 |
| time to the first token, s | 0.56 | 11.1 | 226.7 |

Time to the first token of the prompts longer than 40K tokens, by the share served from the cache:

| Cache hits | Requests | Median | 90th percentile |
|---:|---:|---:|---:|
| 0 to 50% | 225 | 24.1 s | 72.6 s |
| 50 to 90% | 216 | 9.3 s | 33.3 s |
| 90 to 99% | 1,618 | 1.38 s | 4.3 s |
| above 99% | 2,932 | 0.38 s | 0.8 s |

A prefix the cache has never seen costs a full prefill, about 2.9K tok/s on this card: about 35 s
for 100K tokens. Against the checkpoint catalog on the same card:

- three concurrent requests that share a new 73K-token prefix computed 73,085 prompt tokens with the
  hybrid prefix cache and 218,877 with the catalog; the burst took 24.7 s against 75.6 s;
- the same prompt sent again started in 0.10 s instead of 25.6 s, reusing `shared_stable_prefix`
  instead of starting from `root`;
- in real traffic, `shared_stable_prefix` reuse rose from 612 of 6,575 requests on the catalog
  (before 11 October) to 1,756 of 7,127 on the hybrid prefix cache, and requests below 50% cache
  hits fell from 362 to 225.

A single cold prefill takes the same time with either cache.

- **KV format.** `rk4v4` instead of `rk8v4` took the runtime's memory from 9.66 to 7.54 GiB, which
  pays for the Host prefix cache.
- **Host cache.** `--host-cache-mib 6144` runs on this card and 12288 did not start.
- **Draft count.** MTP 5 committed 17 to 22% more tokens per round than MTP 3, while its
  acceptance at 120K fell from 0.607 to 0.482.
- **Concurrency.** Four lanes keep the same window and KV pool as one, and 2 to 4 of them were in
  use. Without `--default-max-tokens`, a request's output limit follows its lane's share of the
  pool ([default output limit](../serving.md#default-output-limit)): a 60K prompt received 5,983
  tokens at four lanes and 71,519 at two, and long agent turns ended with
  `finish_reason=length`.
- **Slow turns.** The request log attributes a slow turn: `prefix_reuse_path`,
  `prefix_cache_hit_tokens`, `computed_prefill_tokens` and the prefill, decode and first-token
  times ([request log](../serving.md#structured-request-log)). `root` on an otherwise unchanged
  long conversation marks a rewritten prefix: a compaction or summary, a changed system prompt or
  tool schema, or a client that serializes the history differently.

## RTX 4080 and RTX 4090, Windows 11

Reported by @jabbor in [iamwavecut/ninfer-all#70](https://github.com/iamwavecut/ninfer-all/issues/70),
13 October 2026: `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer` on an RTX 4080 and on an RTX 4090,
each machine with 64 GB of RAM, served to Hermes Desktop. The reporter found it much faster than
llama.cpp, with larger contexts. The report gives no rates.

## Send a report

Open an [issue](https://github.com/iamwavecut/ninfer-all/issues) with:

- the card, its memory, power limit and driver, and whether it also drives a display;
- the OS, the CPU and its RAM;
- the build: commit, architecture (`120a`, `89`, `86`, ...), CUDA and compiler;
- the artifact and how it was converted;
- the full `ninfer-serve` or `ninfer` command;
- the method: prompt and output lengths, sampling, how many runs, and how the rates were computed;
- the results, with anything that failed or did not fit.
