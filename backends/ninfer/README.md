# NInfer-all

One line of [NInfer](https://github.com/Neroued/ninfer) for the RTX 3090, RTX 4090, RTX 5090 and RTX
PRO 6000 Blackwell: the forks that carry it, consolidated into one tree, plus this repository's own
work. Every change keeps its author; the [maintainer map](docs/maintainer/consolidated-line.md) lists
them with the files they touch.

<details>
<summary>Where the code comes from</summary>

The base is the `master` of [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090):
v0.12.0 (prompt grafts, `/slots` session persistence, the effective thinking budget, worker
recovery) and the multi-GPU pipeline stages, most of both by [Warlax](https://github.com/WarlaxZ),
on the line [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) started from Neroued's
NInfer. On top of it come:

- patches from [TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090);
- ideas from [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) and its
  contributors;
- open pull requests to [Neroued/ninfer](https://github.com/Neroued/ninfer);
- work by [IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv),
  [Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile), Ian Ranson
  ([Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)),
  [tmark00](https://github.com/tmark00/ninfer) and David Oelfke ([gzenz/ninfer](https://github.com/gzenz/ninfer)).

</details>

What the engine does beyond this page (packages, serving APIs, supported models, flags) is in the
READMEs of [NInfer-3090](https://github.com/ashalliants/ninfer-3090#readme) and
[NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090#readme). New features that change
numbers or serving behaviour are opt-in, except four defaults:

- the routes the card's measured device profile picks (`--device-profile off` keeps the compiled
  tables);
- prefill chunks rounded to whole waves of the card's SMs (`NINFER_PREFILL_ALIGN=0` keeps the
  requested chunk);
- TertiumOrganum1's ternary prefill tile (`NINFER_T2_A8_TILE=off`);
- endpoint anchors (`--no-endpoint-anchors`).

## Highlights

### Ternary Bonsai 2 27B and Qwen3.8-27B

Measured in September 2026, one card each, greedy, one request unless the row says otherwise.
Setups and full tables: [reference measurements](docs/performance/reference-2026-09.md).

| | RTX 3090 | RTX 4090 | RTX 5090 | RTX PRO 6000 |
|---|---:|---:|---:|---:|
| **Ternary Bonsai 2 27B**, short chat (DFlash2, 7 drafts) | 202 tok/s | 256 tok/s | 397 tok/s | 381 tok/s |
| decode after a 261K-token document (fastest drafter) | 90 tok/s | 123 tok/s | 218 tok/s | 218 tok/s |
| time to first token for a 261K-token prompt | 215 s | 102 s | 82 s | 78 s |
| largest context, filled and all three needles found | 970,752 | 958,464 | 978,944 | 1,048,576\* |
| eight requests at once (MTP, 3 drafts), total | 551 tok/s | 824 tok/s | 1,063 tok/s | 1,155 tok/s |
| **Qwen3.8-27B**, short chat (DFlash2, 7 drafts) | 118 tok/s | 149 tok/s | 236 tok/s | 237 tok/s |
| largest context, filled and all three needles found | 417,792 | 405,504 | 872,448 | 1,048,576\* |
| eight requests at once (MTP, 3 drafts), total | 329 tok/s | 442 tok/s | 690 tok/s | 739 tok/s |

\* The engine's ceiling. The RTX PRO 6000 (96 GB) starts there with every KV format and drafter;
filled to it, both models find two of the three needles.

- **RTX PRO 6000 against RTX 5090.** Re-measured in October beside an RTX 5090, both at 600 W, the
  PRO 6000 came back within 2% of September wherever the drafts accepted the same share. One request's decode is
  bound by memory bandwidth, and both cards have 1.79 TB/s of GDDR7 (Bonsai 2 without speculation:
  167.7 against 170.4 tok/s). The PRO 6000's 188 SMs against 170 show where compute decides: the
  261K prompt is 2% faster and eight requests at once 6 to 7% faster
  ([re-check](docs/performance/reference-2026-09.md#re-check-october-2026)).
- **Against the previous `master` on the same card.** A 261K-token Bonsai prompt takes 215 s
  instead of 315 s on the RTX 3090, 102 s instead of 138 s on the RTX 4090 and 82 s instead of
  115 s on the RTX 5090; `rk4v4` decode after it is 11 to 13% faster on the 24 GB cards. Decode at
  short context is unchanged, and Qwen3.8's 8K to 32K prompts on the RTX 5090 take 8 to 10% longer.
- **Draft length.** DFlash2 with seven drafts is fastest on short answers; after long documents the
  best count is three to seven. MTP runs up to fifteen drafts and is fastest at three to five.
- **Past the native window.** Filled to about 880K tokens, Bonsai 2 returned all three planted codes
  on every card. At 1,048,576 tokens (RTX 5090 and PRO 6000 only) it misses the one at 943K.

### Qwen3.8-Flash-Next

One request, greedy, October 2026. Each cell is **decode of a short answer · prefill of a
4,463-token prompt**, in tok/s. Decode after the long prompt, memory, host links and power limits:
[Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md#measurements).

**Q2_0** (35.9 GiB model + 26.8 GiB n-gram table):

| Experts | RTX PRO 6000 | RTX 5090 | RTX 4090 | RTX 3090 |
|---|---:|---:|---:|---:|
| on the GPU (two cards, except the PRO 6000) | 138 · 2,939 | 136 · 3,809 | 97 · 3,608 | 90 · 1,504 † |
| in pinned host memory | 55 · 1,297 | 74 · 1,677 | 53 · 1,138 | 49 · 842 |
| on disk, the files in the page cache | 76 · 2,024 | 68 · 1,739 | 42 · 723 | 47 · 630 |
| on disk, cold (pages evicted every second) | 35 · 1,000 | 25 · 694 | 18 · 160 | 17 · 195 |

† Two RTX 3090 Ti.

**IQ3_S** (51.9 GiB model + the same table):

| Experts | RTX PRO 6000 | RTX 5090 | RTX 3090 |
|---|---:|---:|---:|
| on the GPU (two cards for the RTX 5090) | 125 · 2,541 | 122 · 3,326 | — |
| in pinned host memory | 29 · 803 | 41 · 1,100 | 34 · 533 |
| on disk, the files in the page cache | 66 · 1,661 | 55 · 439 | 19 · 209 ‡ |
| on disk, cold | 29 · 773 | 19 · 171 | 11 · 47 |

— Not measured: every expert needs three 24 GB cards. ‡ The host's 62 GB of RAM cached only part
of the file.

**Coder IQ1_M** (28.4 GiB model + the same table), one NVIDIA L40S (48 GB):

| Experts | NVIDIA L40S |
|---|---:|
| in pinned host memory | 35 · 1,648 |
| on disk, the files in the page cache | 42 · 1,938 |

- **The host matters.** Host and disk rows depend on the host as much as on the card. The RTX 5090
  host had PCIe 5.0 x16; the others had PCIe 4.0 x16.
- **The PRO 6000 caches almost everything.** Its 96 GB device expert cache ends up holding nearly
  every expert.
- **RTX 4090 host pinning.** The RTX 4090 rows ran pinned to the GPUs' NUMA node in a two-socket VM.
  Unpinned, host experts decode there at 40 tok/s and disk experts at 32 to 33.
- **Many requests at once.** Six reasoning requests at once on two RTX 5090s produced 193 tok/s of
  output with Q2_0.

## What this line adds

- **Model suspend.** `--model-suspend` lets an idle server give its device memory back without
  exiting (`POST /v1/models/{id}/suspend`) and take it again on the next request or on `/resume`,
  retained conversations included. Ternary Bonsai 2 27B on an RTX 3090: 7.9 GiB down to 0.3 GiB in
  0.33 s, back in 1.1 s. Output and prefix reuse are identical afterwards, on one device or across
  pipeline stages. [Model suspend](docs/serving.md#model-suspend).
- **Several models behind one server.** With `--models-dir` or a llama.cpp-style `--models-preset`,
  `ninfer-serve` is a router with llama.cpp's model API (`/models`, `/models/load`,
  `/models/unload`, `/models/sse`, `--models-max`). With `--model-suspend` a model sleeps instead of
  unloading: two 27B models on one 24 GB RTX 3090 swap in 1.8 s instead of an 11.6 s cold load.
  [Several models](docs/serving.md#several-models-router).
- **llama.cpp's native endpoints.** `POST /completion` and `/v1/completions` continue a raw prompt
  (text, token ids or both). `/tokenize`, `/detokenize` and `/apply-template` expose the tokenizer
  and the chat template. [Raw-prompt completion](docs/serving.md#raw-prompt-completion-and-the-tokenizer).
- **Rerank.** `POST /v1/rerank` (Jina, llama.cpp and TEI shapes) ranks documents with the served
  model as the judge, scored Qwen3-Reranker's way: P(yes) / (P(yes) + P(no)). [Rerank](docs/serving.md#rerank).
- **GGUF block formats.** Qwen3.8-27B GGUF releases that pick a ggml type per tensor, such as
  ISTA-DASLab's GSQ-RCO, convert without requantization (`qwen3_8_27b_gguf`), with MTP, DFlash2 and
  Vision. The 3.5-bit IQ3_S release:
  - 10.95 GiB of weights instead of 15.9;
  - WikiText-2 perplexity 7.071 (its card: 7.07; the official artifact: 7.286);
  - 80.3% on IFBench, 100% on AIME 2025 and 2026, 88.4% on GPQA-Diamond (the official artifact:
    77.7, 96.7, 96.7 and 87.4);
  - 59.9 tok/s against the official artifact's 40.3 on an RTX 3090.

  [GGUF block formats](docs/gguf.md).
- **Device route profiles.** Each card's measured profile picks the kernel route for every
  operation and width before the compiled tables do. Profiles are built in for the RTX 3090, 4090,
  5090 and the three RTX PRO 6000 editions; any other GPU is measured once at first start (20 to
  40 s), and `ninfer-calibrate` re-measures. On the RTX 3090 the `rk4v4` verify attention at 262K
  runs 3.2 times faster, and FP16 P·V with the fast prompt kernel cuts prompt attention by 19 to
  30%. A greedy answer served in a batch now parts more often from the same request served alone,
  at near-tied tokens. `--device-profile off` keeps the compiled routes.
  [Device profiles](docs/device-profiles.md).
- **FP8 and NVFP4 on the default Blackwell build.** Every `120a` build carries the FP8 A8 and NVFP4
  W4A4 tensor-core units. On an RTX PRO 6000, Qwen3.8-27B NVFP4/FP8 prefills 4,096 tokens at
  11,822 tok/s, within 1.4% of a native build; Qwen3.6-35B-A3B NVFP4 prefills at 30,938 tok/s.
- **Faster attention at long context.** Three changes cut a 131K `rk8v4` prompt on an RTX 3090 from
  101 s to 76 s:
  - new small-T tiers for the INT8-family caches;
  - the fast prompt kernel for `rk8v4`, `rk4v4`, `rk4v4-e8` and `rk2v4-e8`;
  - prefill chunks sized to whole SM waves.
- **MTP up to fifteen drafts.** `--draft-tokens 10..15` starts; it used to fail at graph update.
- **BF16 KV with graphs.** MTP with the default BF16 KV cache, and Qwen3.8 without speculation at
  512 and 1,024 tokens of context, no longer fail at startup.
- **Parallel query tiles (opt-in).** A verify step or a short prefill over an INT8-family cache runs
  its 9 to 64 columns as tiles of one split-KV launch (`attn_parallel_tiles` in the profile, or
  `NINFER_ATTN_PARALLEL_TILES=1`).
- **Branch and endpoint anchors.**
  - `--branch-anchors` (opt-in) captures a request where its prompt stops matching a retained
    conversation.
  - Endpoint anchors (on by default; `--no-endpoint-anchors`) keep the point a continued turn
    resumed from. Another reply to the same answer, or an edited last message, resumes there. On an
    RTX 4090, such a branch of a 30.7K-token conversation answers in 110 ms instead of 9.5 s, for
    7-8 ms more on the continued turn.
- **Blackwell kernels** (`120a` builds only):
  - MX FP8 MMA with TMA split-K for FP8 A8 projections;
  - FP4 tensor-core QK for an NVFP4 KV cache past 2,048 keys (`--fast-prefill-kernel`);
  - programmatic dependent launches in captured graphs;
  - native FP8 and NVFP4 A16 operands with CUDA 13.2.
- **Qwen3.8-Flash-Next.** ISTA-DASLab's GSQ-RCO GGUF releases of the 125B-parameter MoE (512
  experts, about 6B active) convert without requantization. The experts can live:
  - on the GPU, or on the GPUs of a `--devices` pipeline;
  - in pinned host memory with a GPU cache (`--expert-residency host`);
  - in the file, read into a GPU cache (`--expert-residency disk`, under 1 GB of RAM).

  It serves up to eight requests with prefix reuse, structured output, images and video; no release
  carries an MTP layer, so there is no MTP. On two RTX 5090s the Q2_0 release scores 93.3% on AIME
  2025 and 86.4% on GPQA-Diamond (84.3% within 106,000 output tokens; its card, from llama.cpp:
  96.67 and 89.39). [Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md).
- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported without rounding, with the
  Hadamard rotations fused into the norms and gates. The recipe `bonsai2_27b_ternary` adds
  ProCreations' MTP head and DFlash2 adapter and an exact proposal head.
- **Integer activations for ternary projections.** Decode, verification and prompts up to 192
  tokens use a small-T kernel over s8 activations; longer prompts use the int8-activation GEMM.
- **RTX 3090 tuning.** Four warps per 1024-point rotation, small-T attention splits in whole SM
  waves, the GDN record window in shared memory, the ternary target's DFlash2 adapter in Q4.
- **DFlash2 with Vision in overlay.** An image encode borrows the drafter's memory, so DFlash2,
  Vision and the full 262,144-token window fit on one 24 GB card.
- **Serving fixes.**
  - A forced `tool_choice` opens the named call.
  - A context-cache store that cannot place a request fails only that request (HTTP 429).
  - A Paged KV exhaustion names its page numbers, and three in a row mark the engine unhealthy.
  - Context-cache fixes keep long agent sessions from re-prefilling.
- **Build.** Tests build against CUDA 13's `cudaGraphGetEdges`; compressed device code keeps the
  binaries under 2 GiB.
- **Reference measurements** of Bonsai 2 and Qwen3.8-27B on four cards:
  [September 2026](docs/performance/reference-2026-09.md).

### From other forks

<details>
<summary><b>From TertiumOrganum1's fork</b>: the <code>rk4v4-e8</code> KV cache, the ternary prefill tile, tool-call recovery</summary>

From [TertiumOrganum1/ninfer-3090](https://github.com/TertiumOrganum1/ninfer-3090):

- **`rk4v4-e8` KV cache.** Keys rotated as in `rk8v4` and snapped per octet to the E8 lattice in
  int4 (the E8 codecs first appeared in NInfer-4090, by UDPSendToFailed with Daniel Parker); values
  keep `rk8v4`'s int4 plane. 280 bytes per token and KV head instead of 408. On Ternary Bonsai 2 the
  262,144-token window takes 2.0 GiB less and two lanes get a whole window each; the codes planted
  at 131K and 250K are still found, and quick-corpus perplexity moves from 5.631 to 5.650.
- **A 128x64 int8 tile for ternary prefill.** Activations are quantised per token and 128-column
  group. Bonsai 2 prefill runs 33% faster at 8K, 21% at 32K and 15% at 64K, perplexity unchanged
  (5.631). `NINFER_T2_A8_TILE=off` restores the old kernel.
- **Tool calls.** A malformed tool-call region is recovered as far as it reads instead of leaking
  its markup into the answer.
- **Shared captures.** A capture that releases less than was assessed is abandoned; before, the
  engine failed for good and answered 503 until a restart.
- **Build.** `sm_120a` builds on the `mma.sync` compatibility path.

</details>

<details>
<summary><b>From NInfer-4090</b>: whole-program build, keys past 262,144 with YaRN, the <code>rk2v4-e8</code> KV cache</summary>

From [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) (UDPSendToFailed
unless named), re-implemented here:

- **Whole-program CUDA build** (Matt Anderson). No relocatable device code in the core and ops
  archives: a fifth fewer kernels need a stack frame; the server binary grows by a quarter.
- **Shared-memory scale reads** in the INT8-family attention kernels. With the whole-program build
  the `rk8v4` verify attention takes 9 to 21% less time: on Bonsai 2 an MTP step at 64K is 6.6%
  shorter and prefill 2 to 7% faster from 8K up, with the same answers.
- **`TCP_NODELAY`** on the server socket.
- **Sigmoid, SiLU and softplus on the SFU (opt-in).** `-DNINFER_SFU_SIGMOID_SILU=ON`: Bonsai 2
  prefill 2% faster from 8K up, perplexity 5.6306 → 5.6309, MTP decode unchanged.
  `-DNINFER_SFU_SOFTPLUS=ON` does the GDN decay gate the same way, with a log1p series for slow
  decays (5.6302).
- **Keys past 262,144.** The small-T attention kernels read page indices from the block table past
  the 64 they stage; the visible-key limit is 1,048,576.
- **Four times the native window.** `--max-context` up to 1,048,576 on the 262,144-token models.
  Past the native window positions run plain RoPE, or YaRN with `--rope-yarn` at Qwen's factor
  (`--rope-yarn-factor F` fixes it). On Bonsai 2 with `rk2v4-e8`, plain RoPE found all three codes at
  500,000 tokens; YaRN found two of three at 131,072, 500,000 and 1,000,000, so it stays off unless
  plain RoPE stops answering.
- **`rk2v4-e8` KV cache** (with Daniel Parker, Neroued/ninfer#173). Two bytes per 8-dimension key
  block: 216 bytes per token and KV head, the one format that holds 1,048,576 tokens beside Bonsai 2
  on a 24 GB card. The cost: perplexity 5.631 → 5.820 (`rk4v4-e8`: 5.651), DFlash2 acceptance
  54.4% → 51.8%, decode 4% slower. All three codes are found at 131K, 250K and 500K.
- **D3D12-resident arenas on Windows** (with keylimesoda). `-DNINFER_D3D12_RESIDENCY=ON` adds
  `--wddm-evictable-budget`. Untested: this line has no Windows machine, and the code only passes a
  MinGW syntax check.
- Also: a server default reasoning effort, MTP draft windows up to 15, `/metrics`, `/slots`
  (Sergiusz Michalik) and `/props`, a WebUI compiled in from `NINFER_WEBUI_DIR`, the block
  sampler's candidates in shared memory, an opt-in bf16 residual add
  (`-DNINFER_BF16_RESIDUAL_ADD=ON`), vector stores in the chunked GDN prefill, and bounded split
  compilation with ptxas reports.

</details>

<details>
<summary><b>From other forks and upstream pull requests</b>: disk KV tier, adaptive MTP, structured output, log probabilities, n-gram drafting, unified Linear templates and more</summary>

- **Disk KV tier** ([IMGillusion](https://github.com/IMGillusion/ninfer-disk-kv)). `--disk-kv-path DIR`
  writes evicted conversations' KV pages and state to CRC-checked LRU files that survive restarts;
  `--disk-kv-restore` seeds a request from a stored prefix. On Bonsai 2 with MTP a 17,444-token
  prompt comes back in 1.2 s instead of 9.6 s (1.0 s after a restart). On Windows,
  `-DNINFER_DIRECTSTORAGE=ON` reads restores through DirectStorage (untested).
- **Adaptive MTP** ([Mirko Covizzi](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile)).
  `--adaptive-mtp` verifies 3..K of the K drafts per round, by measured draft survival and round
  cost. On an RTX 3090 with K=5 it did not beat a fixed K=3 (200 against 204 tok/s on short prompts,
  150 against 163 at 8K), and its per-width graphs cost memory.
- **Fast INT8 prompt attention** (Ian Ranson, [Wallawalla47](https://github.com/Wallawalla47/ninfer-custom)).
  Registers hold each warp's rows, scores and output; P·V accumulates in FP16 per 64-key tile. This
  line extends it to `rk8v4` and the packed key codings, and the device profiles turn it on (19 to
  30% less prompt-attention time); `--fast-prefill-kernel` forces it. Perplexity at 64K moves from
  5.2074 to 5.2079. An `nvfp4` KV cache on Blackwell has its own fast kernel with QK on FP4 tensor
  cores: 3.5 to 14.4% faster prefill at 16K-64K on an RTX 5090.
- **Agent-harness tool calls.** `<function name=...>`, `<invoke name=...>`, `<function_calls>` and
  `<param name=...>` forms (upstream PR #300 by Pavel Kochubey, via Wallawalla47).
- **Structured output** through xgrammar, speculation included: `--structured-output` (upstream
  PR #294 by Andrey Shvartsman).
- **Token log probabilities.** `logprobs`/`top_logprobs` in Chat Completions and
  `message.output_text.logprobs` in Responses, up to 20 alternatives, streamed or not (Fedor
  Suchkov's frinfer design).
- **Cache policies.** `--context-cache-policy rolling` rolls one long conversation's frontier
  forward (IMGillusion); `--release-diverged-checkpoints` drops first the checkpoints a conversation
  has moved away from (Ian Ranson, after pkochubey's upstream PR #300).
- **NVFP4 expert banks** (upstream PRs #286-#290 by Mykhailo Dementii). Qwen3.6-35B-A3B NVFP4
  converts with `--recipe qwen3_6_35b_a3b_nvfp4`; an RTX 5090 (native build) prefilled 27,663 tok/s
  at 4K and decoded 397 tok/s. W4A4 prefill exists only on Blackwell, so `sm_8x` builds refuse the
  banks.
- **N-gram copy drafting** (remesis, Ian Ranson). A round may verify up to 15 tokens copied from
  earlier text that matches the last 12; on by default with `--spec`, exact since the target
  verifies every copy (`--ngram-draft-tokens`, `--ngram-min-match`, `--ngram-archive-mib`).
- **Hybrid prefix cache** (Ian Ranson). `--use-alt-prefix-caching`: content-addressed 64-token KV
  blocks plus sparse state snapshots. Around the default catalog: `--recency-eviction`,
  `--kv-lease-growth`, `--host-cache-mib`, `--auto-long-anchors`; on by default, reuse of an aborted
  request's prefill and least-recently-used replacement of automatic shared prefixes.
- **Admission and eviction** (Gideon Zenz, David Oelfke, Ian Ranson): `--thorough-admission-search`,
  `--value-aware-demote`, `--concurrent-prefill`, `--recover-invariant-failures`.
- **Drafting and sampling** (Gideon Zenz). `--mtp-attention-window N` bounds the MTP head's
  attention to its first 64 keys and the newest `N`. Post-thinking sampling switches to its own
  preset once reasoning closes (`--post-thinking*`, or a `post_thinking` object per request).
- **Serving** (Gideon Zenz, Ian Ranson). `GET /stats` (`--stats-port`), a dashboard and wedge
  watchdog in [`tools/monitor`](tools/monitor/README.md), `--request-log-max-mib`,
  `--assistant-prefill`, `--unconstrained-response-format`, `--lenient-assistant-history`,
  `--derive-session-keys`, Anthropic `ping` events, grouped `--help`, `--log-colours`,
  `--log-stats-panel`, the build id in every binary.
- **Vision on CPU and position interpolation** (David Oelfke): `--vision-residency cpu`;
  `--rope-scaling-factor` with `--rope-scaling-original-context`.
- **Kernels and conversion** (Ian Ranson, Duncan Betts). Programmatic dependent launches in decode
  graphs (`-DNINFER_PDL=ON` on compatibility builds), split-KV attention for short prefill steps, a
  BF16 GEMM fallback, mixed-format MTP banks, the fused RMSNorm with NVFP4 attention input,
  converters for ModelOpt NVFP4/FP8 and Quasar NVFP4 checkpoints and a least-squares scale search
  (in `grouped_search`); a native Windows build against a prebuilt vcpkg tree.
- **Unified Linear templates** (Neroued). Upstream's Q4/Q5/Q6/Q8, FP8, NVFP4 and BF16 templates and
  fused projections sit beside this line's routes, and each card takes them only at the widths
  where they measured faster. Q5 runs 1.5 to 1.7 times as fast from about 8 columns; FP8 and NVFP4
  A16 run 1.7 to 7 and 2.5 to 44 times as fast at verify and prefill widths on an RTX 3090.
  `NINFER_LINEAR_ROUTES=legacy|unified` forces one table.
- **Two-stage GDN prefill** (Neroued). Chunks of 16 tokens or more run one preparation pass and one
  FP32-state recurrence. The op is 1.4 to 4.3 times as fast; Engine prefill of Qwen3.8-27B on an
  RTX 3090 moved by about 1%. `NINFER_GDN_TWO_STAGE=0|1` forces either.
- **PackGQA** (Gideon Zenz). `NINFER_PROMPT_PACK_GQA=1` packs each KV head's query heads into the
  prompt kernel's tiles: 2.7% faster on an RTX 3090, 0.5% and 3.9% slower on a 4090 and a 5090, so no
  built-in profile turns it on.
- **Engine and serving fixes.** Worker out-of-memory recovery (David Oelfke, ported by Ian Ranson);
  `--kv-headroom-mib`, `--cuda-graph-allowance-mib`, `--thinking-budget-message` (Ian Ranson);
  `--webui-mcp-proxy`, table-decoded E8 roots and an SM-count RMSNorm cutoff
  ([tmark00](https://github.com/tmark00/ninfer)); MTP graph profiles with topology classes
  (Mykhailo Dementii, upstream PR #221); openable server URLs and CORS preflight echoes (pelebel,
  natpate). Upstream pull requests: GGUF as a conversion source (giveen), a Q6 recipe
  (bingchengcc), sparse-MoE, NVFP4 and attention-epilogue tuning (Mykhailo Dementii, Duncan Betts,
  MOVIBALE), tool-call fixes (Fedor Suchkov, adubkov), Copilot tool shapes (Damian Sromek).

</details>

The [maintainer map](docs/maintainer/consolidated-line.md) lists each change with the files it
touches and the tests that cover it.

## Running

Download an artifact from the [table below](#artifacts) and serve it from the [Docker image](#docker)
or from a [build](#building). The server speaks the OpenAI and Anthropic APIs, and each card picks
up its device profile on its own.

```bash
hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir models

# The image: `serve`, the artifact under /models, the flags. Listens on http://localhost:8080/v1.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5

# A build: the same flags. Listens on 127.0.0.1:8080 unless --host and --port say otherwise.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

The recipes below are the configurations the [reference tables](docs/performance/reference-2026-09.md)
and the model cards use. They are written for a build; in the container, replace
`ninfer-serve models/` with the `docker run ... serve /models/` line above.

<details>
<summary><b>Ternary Bonsai 2 27B</b>: DFlash2, MTP, the largest context, a disk tier</summary>

```bash
# Fastest single stream: DFlash2 with five drafts over the full 262,144-token window.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# MTP drafting through the proposal head.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft

# The largest context a 24 GB card holds: 958,464 tokens of rk4v4.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4 --gdn-state-fp16 --rope-yarn

# Adaptive MTP and a disk tier that keeps evicted conversations.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --lm-head-draft --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```

- **Draft count.** Five drafts are the all-round choice. Seven are faster on short answers; after
  long documents three to seven win ([draft length](docs/performance/reference-2026-09.md#draft-length)).
- **Images.** `--vision --vision-residency overlay --vision-max-merged 12288` adds them. The encode
  borrows the drafter's memory, so the whole window still fits a 24 GB card.
- **Past 958,464 tokens.** The RTX 5090 and the PRO 6000 hold the 1,048,576-token maximum with
  `rk4v4`, DFlash2 or MTP included; the RTX 5090 holds 978,944 with `rk8v4`. Filled to 1,048,576
  tokens, the model misses the code at 90% (about 943K); up to about 880K it found every code on
  every card.

</details>

<details>
<summary><b>Qwen3.8-27B</b>: upstream's artifact and the GSQ-RCO IQ3_S release</summary>

```bash
# Upstream's artifact on a 24 GB card: DFlash2 with five drafts over 245,760 tokens of rk4v4.
ninfer-serve models/qwen3_8_27b.ninfer --model-id qwen3.8-27b \
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# GSQ-RCO IQ3_S: MTP over 176,128 tokens of rk8v4.
ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 176128 --kv-capacity 176128 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3
```

- **Upstream's artifact with `rk8v4`.** The same speculation fits 167,936 tokens on an RTX 4090 and
  176,128 on an RTX 3090. An RTX 5090 takes the full 262,144 with either KV format.
- **IQ3_S.** It drafts with `--spec dflash2 --draft-tokens 5` as well, and `--vision` adds images.

</details>

<details>
<summary><b>Qwen3.8-Flash-Next</b>: experts on the GPUs, in host memory or on disk</summary>

```bash
hf download WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3 \
  Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer --local-dir models
hf download WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3 \
  Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer --local-dir models
M=models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer
T=models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer

# Every expert on one RTX PRO 6000 (96 GB).
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768

# Every expert on two GPUs, one pipeline stage each (Linux): 24 GB cards for Q2_0, 32 GB for IQ3_S.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --devices 0,1

# One 24 GB GPU: the experts in pinned host memory, the most used of them cached on the GPU.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host

# One 24 GB GPU and little RAM: the experts stay in the file and stream into a GPU cache.
ninfer-serve $M --ngram-table $T --model-id qwen3.8-flash-next --max-context 32768 --expert-residency disk

# The container, experts in host memory.
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all serve /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer \
  --ngram-table /models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer \
  --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host
```

- **Other releases.** IQ3_S and the Coder IQ1_M build take the same flags with their own file and
  the same table. The [tables above](#qwen38-flash-next) show the placements measured for each.
- **Memory.** Host experts pin 34 GB of RAM for Q2_0, 50 GB for IQ3_S and 25 GB for the Coder
  build. Disk experts need under 1 GB of RAM; the page cache does the rest. The GPU expert cache
  takes what is free after startup, or `--expert-cache-mib`.
- **The n-gram table.** Its rows are read from the file, 16 per token, unless `--ngram-ram` loads
  all 28.8 GB. A model started without its table is refused; `--no-ngram-table` overrides that, an
  experimental mode with no practical use.
- **Images and video.** `--vision` adds the Vision tower (0.9 GB on the GPU).

</details>

<details>
<summary><b>Qwen3.6-35B-A3B NVFP4</b>: RTX 50 series and RTX PRO 6000 only</summary>

```bash
ninfer-serve models/Qwen3.6-35B-A3B-NVFP4-ninfer-v3.ninfer --model-id qwen3.6-35b-a3b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision
```

The NVFP4 experts prefill through W4A4 on Blackwell's FP4 tensor cores, so this needs a `120a`
build (the image has one); `sm_86` and `sm_89` builds refuse the file. On an RTX 5090 it starts in
17 s and leaves 7.4 GiB of the card free.

</details>

<details>
<summary><b>Qwen3.8-27B fine-tunes</b>: Huihui abliterated, HauhauCS Aggressive, MXFP8-CRACK</summary>

All three have the official Qwen3.8-27B identity and size, so the Qwen3.8-27B recipes above serve
them too; CRACK has no DFlash2 adapter. Their cards' configurations:

```bash
# Huihui abliterated on one RTX 3090: 198,400 tokens with MTP and Vision.
ninfer-serve models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-residency overlay --vision-max-merged 12288

# HauhauCS Aggressive: DFlash2 with seven drafts.
ninfer-serve models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --max-context 32768 --kv-capacity auto \
  --spec dflash2 --draft-tokens 7 --lm-head-draft

# MXFP8-CRACK: MTP.
ninfer-serve models/Qwen3.8-27B-MXFP8-CRACK-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 32768 --kv-capacity auto --spec mtp --draft-tokens 3 --lm-head-draft
```

</details>

<details>
<summary><b>A card without a built-in profile</b></summary>

```bash
ninfer-calibrate --print > my-gpu.json
```

The engine measures it by itself at first start. Running it by hand refreshes the profile after a
driver or clock change. See [device profiles](docs/device-profiles.md).

</details>

## Artifacts

The Bonsai, GSQ-RCO and Flash-Next artifacts use formats that only this line reads.

| model | artifact | size | contents |
|---|---|---:|---|
| Ternary Bonsai 2 27B | [WaveCut/Ternary-Bonsai-2-27B-NInfer-v3](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 GiB | ternary weights, Vision, Bonsai-trained MTP head and DFlash2 adapter, exact proposal head |
| Qwen3.8-27B | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 GiB | upstream's `groupwise-int` (Q4/Q5) with MTP and DFlash2; the reference tables' artifact |
| Qwen3.8-27B GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 GiB | ISTA-DASLab's 3.5-bit GGUF blocks byte for byte, Q6_K MTP head, Vision, DFlash2, proposal head |
| Qwen3.8-27B, abliterated | [WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 GiB | upstream's `qwen3_8_27b` recipe with MTP, DFlash2 and a proposal head |
| Qwen3.8-27B Uncensored, HauhauCS Aggressive | [WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3) | 19.03 GiB | `groupwise-int` with the tune's MTP head, DFlash2 and a proposal head |
| Qwen3.8-27B MXFP8-CRACK | [WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) | 16.96 GiB | `groupwise-int` with MTP and a proposal head; no DFlash2 |
| Qwen3.8-Flash-Next GSQ-RCO Q2_0 | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 35.89 GiB | ISTA-DASLab's 2.4-bit GGUF blocks byte for byte, Vision; reads the n-gram table |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 51.90 GiB | the 3.5-bit release, the same way |
| Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M | [WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 28.42 GiB | the expert-pruned coding build (256 experts per layer), the same way |
| Qwen3.8-Flash-Next n-gram table | [WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.82 GiB | the IQ4_NL n-gram table every Flash-Next artifact reads (`--ngram-table`) |
| Qwen3.6-35B-A3B NVFP4 | [WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 GiB | RedHatAI's NVFP4 experts code for code, Q8 projections, Vision, MTP, proposal head; `sm_120a` GPUs only |

The official NInfer artifacts listed in the original READMEs load here too. Weight conversion shows
how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Docker

`ghcr.io/iamwavecut/ninfer-all:latest` is built from every `master` commit that passes CI, on CUDA
13.4 and Ubuntu 26.04. It carries two builds and starts the one that matches the GPU: `sm_86` for
the RTX 30 and RTX 40 series, `sm_120a` for the RTX 50 series and the RTX PRO 6000 Blackwell.

- **Host.** An NVIDIA driver of the CUDA 13 branch (580 or newer) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Tags.** `latest`, `sha-<commit>` and the `VERSION`; the image is 1.6 GB compressed.
- **Volumes.** `/models` holds artifacts, `/cache` the device profile measured on first start, and
  `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts).
- **Checked** on an RTX 3090 and an RTX 5090 with driver 580.159.03.

The container's command chooses what runs:

| command | runs |
|---|---|
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary with any artifact and flags; `serve` listens on `0.0.0.0:8080` unless given `--host` or `--port` |
| `run <model> [profile]` (the default: `run qwen38-27b`) | the launcher profiles of `scripts/run.sh` for upstream's `qwen38-27b` and `qwen36-35b-a3b`, with its `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models`: `qwen38-27b`, `qwen36-27b` or `qwen36-35b-a3b` |

Upstream's Qwen3.8-27B with the measured `tuned` profile, on http://localhost:8080/v1:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all run qwen38-27b
```

[compose.yaml](compose.yaml) wires the GPU, the port and the volumes for the same:
`docker compose run --rm ninfer download qwen38-27b`, then `docker compose up -d`.
`NINFER_IMAGE_ARCH=sm86|sm120a` overrides the GPU detection, and `docker build -t ninfer .` builds
the image from source (`--build-arg ARCHS=86` for one architecture).

## Building

<details>
<summary>Linux with CUDA 13.1</summary>

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

`CMAKE_CUDA_ARCHITECTURES` is `86` for the RTX 30 series, `89` for the RTX 40 series and `120a`
for the RTX 50 series and the RTX PRO 6000 Blackwell (on the `mma.sync` compatibility path, which
the ternary route needs). A `120a` build needs CUDA 13.1 or newer: CUDA 12.8 and 12.9 miscompile
sm_120a kernels, and configure refuses them. The opt-in build options are listed in the
[Linux build guide](docs/rtx-3090-linux.md#build-options). Windows builds, release packages, tests
and benchmarks work as in the [NInfer-3090 README](https://github.com/ashalliants/ninfer-3090#readme).

</details>

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices. The Qwen3.8-Flash-Next artifacts carry the Qwen
Community License 1.0 of their model.
