# NInfer-all

A C++/CUDA inference engine for the Qwen3.5 model family (Qwen3.6 and Qwen3.8 27B, Qwen3.6-35B-A3B,
their fine-tunes and Ternary Bonsai 2), Qwen3.8-Flash-Next, Gemma 4 26B-A4B and Recursive's Flint,
on NVIDIA GPUs from Volta to consumer Blackwell, from 8 GB of memory. It serves the OpenAI, Anthropic
and llama.cpp APIs from one GPU, or from up to eight as pipeline stages on Linux. The same chat model
answers [typed decisions](docs/serving.md#typed-decisions): choice, yes/no and score questions about
a state, each option with its probability, through TypeSafe's System One and OpenAI's Decisions APIs.

[Quick start](#quick-start) · [Supported GPUs](#supported-gpus) · [Models](#artifacts) ·
[Highlights](#highlights) · [Features](#features) · [Docker](#docker) · [Building](#building) ·
[Documentation](docs/README.md)

NInfer-all continues [NInfer](https://github.com/Neroued/ninfer) through the NInfer-3090 line and
carries the work of several forks; [CONTRIBUTORS.md](CONTRIBUTORS.md) credits each of them.

## Quick start

Download an artifact ([all models](#artifacts)) and serve it from the [Docker image](#docker). The server
speaks the OpenAI and Anthropic APIs on http://localhost:8080/v1, answers typed decisions there, and
each card picks up its device profile on its own.

```bash
hf download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir models

docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai2-27b","messages":[{"role":"user","content":"Hello"}]}'

curl http://localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "My card was charged twice for one order."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments and refunds", "shipping": "deliveries"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"}}}'
```

The second request is a [typed decision](docs/serving.md#typed-decisions): each answer comes with the
probability of every option, read from the model's first-token log probabilities of the labels.

Flags for every model and memory size, from an 8 GB card to the largest context: [launch recipes](docs/recipes.md).

<details>
<summary><b>Without Docker</b> — the same flags on a build</summary>

```bash
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5
```

A build listens on 127.0.0.1:8080 unless `--host` and `--port` say otherwise. [Building](#building) shows
how to make one.

</details>

## Supported GPUs

NVIDIA GPUs from Volta (compute capability 7.0) to consumer Blackwell (12.0), from 8 GB of memory:
Ternary Bonsai 2 27B runs on an 8 GB RTX 3060 Ti, and Qwen3.8-Flash-Next streams its experts from
host memory or disk through 8 GB. The two [Docker images](#docker) each contain all NInfer targets
for their CUDA version in a shared runtime: CUDA 12.9 includes Volta, and CUDA 13.4 includes Blackwell.
The native release pipeline also assembles a portable Windows or Linux package with **both** CUDA
backends: `ninfer-<os>-x64-cuda12.9-13.4-<commit12>`. Its small command executables share one
runtime per CUDA family. The launcher selects CUDA 12 for Volta, CUDA 13 for consumer Blackwell,
and prefers CUDA 13 on overlapping targets when the driver supports it. All selected `--devices`
must fit one backend. Use `--cuda-backend auto|12|13` to select it explicitly. No CUDA Toolkit
installation is needed. Keep `backends/` and `help/` beside the executables. Linux also needs the
recorded Debian system libraries. `--help` and `--version` do not load CUDA.

Every card by name, with what has run on it:
[Supported GPUs](docs/gpus.md).

*Click a section to expand it.*

<details>
<summary><b>Architectures and builds</b> — six architectures, two CUDA variants</summary>

| Architecture | Compute capability | Build | CUDA | Docker tags | Status |
|---|---|---|---|---|---|
| Blackwell: GeForce RTX 50, RTX PRO Blackwell | 12.0 | `120a` | 13.1 or newer | `cuda13.4` | measured; built-in device profiles |
| Ada Lovelace | 8.9 | `89`, or the `86` build | 12.8 or newer | both, native `89` | measured; built-in device profiles |
| Ampere | 8.6 | `86` (the default) | 12.8 or newer | both | measured (the reference card); built-in device profiles |
| Ampere, A100 class | 8.0 | `80` | 12.8 or newer | both | built-in device profiles; not benchmarked |
| Turing | 7.5 | `75` | 12.8 or newer | both | measured; built-in device profiles |
| Volta | 7.0 | `70` | 12.8 or 12.9 | `cuda12.9` | measured; built-in device profiles |
| Hopper; data-center Blackwell; GB10; Pascal and older | | none | | | no build |

</details>

<details>
<summary><b>What fits in your GPU memory</b> — measured, from 8 GB to 48 GB</summary>

| GPU memory | Runs | Measured on |
|---|---|---|
| 8 GB | Ternary Bonsai 2 27B (24,576 tokens with `rk4v4`, no MTP); Qwen3.8-Flash-Next with its experts in host memory or on disk | RTX 3060 Ti |
| 11 GB | Ternary Bonsai 2 27B (200,704 tokens with `rk4v4`; 143,360 with MTP) | RTX 2080 Ti |
| 12 GB | Ternary Bonsai 2 27B (266,240 tokens); Qwen3.8-27B GSQ-RCO IQ3_S (8,192 tokens, no MTP) | RTX 3060 |
| 16 GB | Gemma 4 26B-A4B Q4_0 (61,440 tokens); Qwen3.8-27B IQ3_S (241,664 tokens); the official Qwen3.8-27B with `--embedding-q4 --lm-head-q4` | Tesla V100-SXM2-16GB |
| 24 GB | the official Qwen3.8-27B and its MXFP8 tune; Qwen3.6-35B-A3B `groupwise-int` on an RTX 3090; Flash-Next with every expert on the GPU over two cards | RTX 3090, RTX 4090, TITAN RTX |
| 32 GB | Qwen3.6-35B-A3B NVFP4 (Blackwell only) | RTX 5090 |
| 48 GB | Flash-Next Q2_0 with every expert on one card | L40S, RTX PRO 6000 |

</details>

<details>
<summary><b>Device profiles and limits</b> — how a card gets its kernel routes</summary>

- **Device profiles.** A profile picks the kernel route of every operation and width. The binary
  carries profiles measured on 48 GPU models, from the Tesla V100 and RTX 2060 Super to the RTX PRO
  6000 (the Profile column of [supported GPUs](docs/gpus.md#cards)); a built-in profile applies to
  the same GPU name, compute capability and SM count. Any other card is calibrated once at first
  start, which takes about half a minute to three and a half minutes, and the result is stored for
  later starts; `ninfer-calibrate` measures again after a driver or clock change
  ([device profiles](docs/device-profiles.md)). The `70` build runs its own
  routes, so the V100 profiles hold only the GDN prefill switch and, for the 32 GB card, the Q8 MoE
  prefill frontier.
- **Architecture limits.** NVFP4 expert weights need a `120a` build. The Turing and Volta builds
  replace instructions their GPUs lack and reshape kernels to their shared memory
  ([Turing](docs/build-linux.md#turing-sm_75), [Tesla V100](docs/build-linux.md#tesla-v100-sm_70)).
  Configure accepts no other target, so Pascal and older, Hopper and data-center Blackwell have no
  build.
- **Speed per card.** [Performance](docs/performance.md) lists the measured rates of every card.

</details>

## Artifacts

Each name links to its Hugging Face card; `hf download REPO FILE.ninfer --local-dir models` fetches the
file. The Bonsai, GSQ-RCO, Flash-Next, Gemma 4 and Flint artifacts use formats that only this engine
reads.
✓ means the artifact carries the component, — that it does not.

| Model | Size, GiB | Weights | Vision | MTP | DFlash2 | Proposal head | Notes |
|---|---:|---|:-:|:-:|:-:|:-:|---|
| [Ternary Bonsai 2 27B](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) | 8.87 | ternary, 2.125 bits | ✓ | ✓ | ✓ | ✓ | Bonsai-trained MTP head and DFlash2 adapter |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 19.03 | groupwise-int (Q4/Q5) | ✓ | ✓ | ✓ | ✓ | the official artifact; the reference tables use it |
| [Qwen3.8-27B GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) | 13.99 | GGUF blocks, 3.5 bits | ✓ | ✓ | ✓ | ✓ | ISTA-DASLab's GGUF blocks byte for byte, Q6_K MTP head |
| [Qwen3.8-27B, abliterated](https://huggingface.co/WaveCut/Huihui-Qwen3.8-27B-abliterated-NInfer-v3) | 19.03 | groupwise-int (Q4/Q5) | ✓ | ✓ | ✓ | ✓ | the official recipe |
| [Qwen3.8-27B Uncensored, HauhauCS Aggressive](https://huggingface.co/WaveCut/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-NInfer-v3) | 19.03 | groupwise-int (Q4/Q5) | ✓ | ✓ | ✓ | ✓ | the tune's MTP head |
| [Qwen3.8-27B MXFP8-CRACK](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) | 16.96 | groupwise-int | ✓ | ✓ | — | ✓ |  |
| [Qwen3.6-35B-A3B NVFP4](https://huggingface.co/WaveCut/Qwen3.6-35B-A3B-NVFP4-NInfer-v3) | 20.39 | NVFP4 experts, Q8 projections | ✓ | ✓ | — | ✓ | RedHatAI's NVFP4 experts code for code; `sm_120a` GPUs only |
| [Qwen3.8-Flash-Next GSQ-RCO Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.49 | GGUF blocks, 2.4 bits | ✓ | ✓ | — | — | shared-Q8_0 MTP; reads the n-gram table |
| [Qwen3.8-Flash-Next GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.50 | GGUF blocks, 3.5 bits | ✓ | ✓ | — | — | shared-Q8_0 MTP; reads the n-gram table |
| [Qwen3.8-Flash-Next Coder GSQ-RCO IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 31.02 | GGUF blocks | ✓ | ✓ | — | — | expert-pruned coding build, 256 experts per layer; shared-Q8_0 MTP; reads the n-gram table |
| [Gemma 4 26B-A4B it QAT](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3) | 14.75 | GGUF Q4_0 blocks | ✓ | ✓ | — | — | Google's QAT GGUF byte for byte; the MTP column is Google's QAT assistant |
| [Flint UD-Q4_K_XL](https://huggingface.co/WaveCut/Flint-UD-Q4_K_XL-NInfer-v3) | 23.38 | GGUF blocks, Q8_0 to IQ3_S | — | — | — | — | Unsloth's UD-Q4_K_XL GGUF byte for byte; drafts by context lookup |

The three Flash-Next releases read one shared n-gram table, which the model names by digest: the
[n-gram table](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) (26.92 GiB,
unchanged IQ4_NL rows plus an optional broad hot-row profile), passed with `--ngram-table`.

The official NInfer artifacts (Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B, `groupwise-int` and
`nvfp4`) load too; the [documentation map](docs/README.md#model-artifacts) links them. Weight
conversion shows how the [Bonsai](docs/weight-conversion.md#ternary-bonsai-2-27b) and
[GSQ-RCO](docs/weight-conversion.md#a-mixed-precision-qwen38-27b-gguf) artifacts are built.

## Highlights

Ternary Bonsai 2 27B decodes 397 tok/s on an RTX 5090 with DFlash2 and holds a 978,944-token context with
every planted code found. Qwen3.8-Flash-Next decodes 138 tok/s with every expert on one RTX PRO 6000, and
74 tok/s with its experts in host memory on one RTX 5090. Every table of every card is on the
[performance page](docs/performance.md).

*Click a section to expand it.*

<details>
<summary><b>Ternary Bonsai 2 27B</b> — up to 397 tok/s, a million tokens of context</summary>

Measured in September 2026, one card each, greedy, one request unless the column says otherwise.

| Card | Short chat, DFlash2 7 drafts (tok/s) | After a 261K-token document (tok/s) | 261K-token prompt, to the first token | Largest context, all three codes found | Eight requests at once, MTP 3 (tok/s) |
|---|---:|---:|---:|---:|---:|
| RTX 3090 | 202 | 90 | 215 s | 970,752 | 551 |
| RTX 4090 | 256 | 123 | 102 s | 958,464 | 824 |
| RTX 5090 | 397 | 218 | 82 s | 978,944 | 1,063 |
| RTX PRO 6000 | 381 | 218 | 78 s | 1,048,576, two of three codes | 1,155 |

The RTX PRO 6000 (96 GB) starts at the engine's ceiling of 1,048,576 tokens with every KV format and drafter.
Filled to it, the model misses the code at 90% (about 943K); up to about 880K it found every code on every card.

</details>

<details>
<summary><b>Qwen3.8-27B</b> — up to 237 tok/s, 872,448 tokens of context filled</summary>

The official `groupwise-int` artifact, measured in September 2026, one card each, greedy, one request unless the column says otherwise.

| Card | Short chat, DFlash2 7 drafts (tok/s) | Largest context, all three codes found | Eight requests at once, MTP 3 (tok/s) |
|---|---:|---:|---:|
| RTX 3090 | 118 | 417,792 | 329 |
| RTX 4090 | 149 | 405,504 | 442 |
| RTX 5090 | 236 | 872,448 | 690 |
| RTX PRO 6000 | 237 | 1,048,576, two of three codes | 739 |

- **RTX PRO 6000 against RTX 5090.** Re-measured in October beside an RTX 5090, both at 600 W, the
  PRO 6000 came back within 2% of September wherever the drafts accepted the same share. One
  request's decode is bound by memory bandwidth, and both cards have 1.79 TB/s of GDDR7 (Bonsai 2
  without speculation: 167.7 against 170.4 tok/s). The PRO 6000's 188 SMs against 170 show where
  compute decides: the 261K prompt is 2% faster and eight requests at once 6 to 7% faster
  ([re-check](docs/performance.md#rtx-pro-6000-against-rtx-5090-october-2026)).
- **Draft length.** DFlash2 with seven drafts is fastest on short answers; after long documents the
  best count is three to seven. MTP runs up to fifteen drafts and is fastest at three to five.

</details>

<details>
<summary><b>Qwen3.8-Flash-Next</b> — 138 tok/s with the experts on a GPU, 74 tok/s from host memory</summary>

The Q2_0 release (35.9 GiB model and 26.8 GiB n-gram table), one request, greedy, October 2026. Decode of
a short answer, in tok/s:

| Card | Every expert on the GPUs | Experts in host memory | Experts on disk, in the page cache | Experts on disk, cold |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 138 (one card) | 55 | 76 | 35 |
| RTX 5090 | 136 (two cards) | 74 | 68 | 25 |
| RTX 4090 | 97 (two cards) | 53 | 42 | 18 |
| RTX 3090 | 90 (two RTX 3090 Ti) | 49 | 47 | 17 |

Prefill of a 4,463-token prompt, in tok/s:

| Card | Every expert on the GPUs | Experts in host memory | Experts on disk, in the page cache | Experts on disk, cold |
|---|---:|---:|---:|---:|
| RTX PRO 6000 | 2,939 | 1,297 | 2,024 | 1,000 |
| RTX 5090 | 3,809 | 1,677 | 1,739 | 694 |
| RTX 4090 | 3,608 | 1,138 | 723 | 160 |
| RTX 3090 | 1,504 | 842 | 630 | 195 |

- **The host matters.** Host and disk rows depend on the host as much as on the card. The RTX 5090
  host had PCIe 5.0 x16; the others had PCIe 4.0 x16. The RTX 4090 rows ran pinned to the GPUs' NUMA
  node in a two-socket VM; unpinned, host experts decode there at 40 tok/s and disk experts at 32 to 33.
- **Older host and disk rows.** They were measured before the October 9 decode work. Host experts on
  an RTX 3090 now decode 87-94 tok/s in a different workload, 98-100 with a repeated prompt
  ([host experts on one RTX 3090](docs/qwen3-8-flash-next.md#host-experts-on-one-rtx-3090)).
- **Other releases and cards.** IQ3_S, the Coder IQ1_M build and the 8 and 12 GB cards are in the
  [expert placements](docs/performance.md#expert-placements). Six reasoning requests at once on two
  RTX 5090s produced 193 tok/s of output with Q2_0.

</details>

## Features

*Click a section to expand it.*

<details>
<summary><b>Models and weight formats</b> — Qwen3.5 family, GGUF block formats, Ternary Bonsai 2, Flash-Next, Gemma 4, Flint</summary>

- **Qwen3.5 family.** Dense and MoE (`Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`) with MTP,
  DFlash (35B-A3B) and DFlash2 (Qwen3.8-27B) drafters, Vision for images and video, and
  `groupwise-int`, FP8, NVFP4 and ternary weights. Native MoE routed experts can stay in pinned
  host memory or the artifact's files, brought into whole-layer GPU banks with an optional device
  cache ([expert residency](docs/maintainer/qwen3_5-model.md#routed-expert-residency)). NVFP4
  expert banks prefill through W4A4 and need a `120a` build.
- **GGUF block formats.** Releases that pick a ggml type per tensor, such as ISTA-DASLab's GSQ-RCO,
  convert without requantization and multiply their blocks in place. The 3.5-bit Qwen3.8-27B IQ3_S
  release holds 10.95 GiB of weights instead of 15.9, scores WikiText-2 perplexity 7.071 (the
  official artifact: 7.286) and decodes at 59.9 tok/s against the official artifact's 40.3 on an
  RTX 3090 ([GGUF block formats](docs/gguf.md)).
- **Ternary Bonsai 2 27B.** PrismML's [ternary Qwen3.8-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
  runs from `t2_g128_fp16` weights: 2.125 bits per weight, imported without rounding, with the
  Hadamard rotations fused into the norms and gates and integer activations for its projections.
  The `bonsai2_27b_ternary` recipe adds ProCreations' MTP head and DFlash2 adapter and an exact
  proposal head.
- **Qwen3.8-Flash-Next.** The 125B-parameter MoE (512 experts, about 6B active) from the GSQ-RCO
  GGUF releases, with its experts on the GPUs, in pinned host memory with a GPU cache
  (`--expert-residency host`) or in the artifact's files read into a GPU cache
  (`--expert-residency disk`, under 1 GB of RAM). It serves up to eight requests with prefix reuse,
  structured output, images and video, and MTP drafting. On two RTX 5090s the Q2_0 release scores
  93.3% on AIME 2025 and 86.4% on GPQA-Diamond ([Qwen3.8-Flash-Next](docs/qwen3-8-flash-next.md)).
- **Gemma 4 26B-A4B.** Google's QAT Q4_0 GGUF release, imported block for block (13.4 GiB), or a
  mixed-format GGUF of it: the sliding-window and full-attention blocks (the latter caching values
  only), the dense MLP beside 128 routed experts, every `--kv-dtype` format, up to eight concurrent
  requests with prefix reuse through host memory and shared system prompts, tool calls and
  structured output, images and video through the release's vision tower (on the
  GPU, streamed from pinned memory or on CPU threads), and Google's multi-token-prediction assistant
  for speculative decoding. On an RTX 3090 it prefills a 10,000-token prompt at 6,045 tok/s and decodes at 215
  tok/s, 190 after 10,000 tokens, against llama.cpp's 4,697, 169 and 152 on the same GGUF; with two
  drafts a round chat answers decode at 291 tok/s instead of 209 ([Gemma 4](docs/gemma4.md)).
- **Flint.** Recursive's 39.8B coding MoE (2.5B active), from a llama.cpp GGUF such as Unsloth's
  UD-Q4_K_XL, imported block for block (23.4 GiB): sliding-window and position-free full-attention
  blocks with learned sinks, 256 routed experts beside a shared one, up to eight concurrent
  requests with prefix reuse through host memory, the model's own reasoning, GLM-4.7 tool calls,
  structured output, typed decisions, context-lookup drafting, and experts on the GPU, in pinned
  host memory or in the artifact's files. On an RTX 5090 it prefills a 10,000-token prompt at
  14,656 tok/s and decodes at 348 tok/s, 315 after 10,000 tokens, against llama.cpp's 8,679, 223
  and 209 on the same GGUF; on an RTX 3090, with its experts in host memory, at 5,755, 161 and 140
  against llama.cpp's 2,553, 80 and 77 with four blocks' experts on the CPU
  ([Flint](docs/flint.md)).
- **CPU share of host and disk experts.** With Flash-Next, Gemma 4, Flint or native Qwen3.5/3.6
  MoE experts off the GPU, CPU threads compute a share of each decode call's cache misses from host
  memory while the GPU copies the rest, bit for bit equal to the GPU's products
  (`--expert-dma-share auto`). The pool takes `min(physical cores, usable CPUs / 2)` threads by
  default, where usable CPUs are the affinity mask bounded by a container's CPU quota (15 threads
  on a 30-CPU quota); `--expert-cpu-threads N` sets another count
  ([common options](docs/cli.md#common-options)).

</details>

<details>
<summary><b>Speculative decoding</b> — MTP, DFlash, DFlash2, n-gram copying and context lookup</summary>

- MTP, DFlash and DFlash2 with 1 to 15 drafts; the artifact's proposal head drafts by default when it
  stores one. Under MTP and DFlash a sampled request keeps a draft only while it equals the model's
  own keyed draw, so every emitted token is the model's sample; DFlash2 verifies its proposal
  distribution by rejection sampling.
- Gemma 4 drafts with Google's multi-token-prediction assistant (`--spec mtp`, 1 to 7 drafts), four
  blocks that read the model's own KV cache; its verification rows are bit for bit plain decode rows,
  so the output is the one plain decoding gives.
- Flint, which has no published drafter, drafts by context lookup (`--lookup-ngram N`, 1 to 7
  drafts): what followed the last N tokens' earlier occurrence, verified in one pass of decode rows.
- `--adaptive-mtp` picks each round's width from measured draft survival and round cost.
- N-gram copy drafting beside the neural drafter (on with `--spec`) verifies up to 15 tokens copied
  from earlier prompt, tool-result or output text ([ngram copy proposals](docs/ngram.md)).

</details>

<details>
<summary><b>Context and memory</b> — nine KV formats, up to 1,048,576 tokens, a context cache with a disk tier</summary>

- Nine KV formats, from `bf16` to `rk2v4-e8` at 216 bytes per token and KV head; `rk8v4` is the
  all-round choice and `rk4v4` holds the native 262,144-token window of Qwen3.8-27B on a 24 GB card
  ([context and memory](docs/cli.md#context-and-memory)).
- Up to 1,048,576 tokens of context, four times the native window, with plain RoPE or YaRN
  (`--rope-yarn`, for Qwen3.5, Qwen3.8-Flash-Next and Gemma 4; Flint's full-attention blocks carry
  no rotation and take no scaling).
- A hybrid prefix cache by default, which shares content-addressed KV blocks and sparse state
  snapshots across requests on the device and in pinned host memory; or a checkpoint catalog
  (`--use-original-prefix-caching`) with an optional disk tier that survives restarts
  (`--disk-kv-path`, `--disk-kv-restore`) and endpoint and branch anchors that let an edited or
  regenerated turn resume where it left the conversation.
- Precision trades that buy context: `--embedding-q4`, `--lm-head-q6`, `--gdn-state-fp16`
  ([quality trades](docs/maintainer/quality-trade-experiments.md)).

</details>

<details>
<summary><b>Serving</b> — OpenAI, Anthropic and llama.cpp APIs, typed decisions, model suspend, several models behind one server</summary>

- OpenAI Responses and Chat Completions, Anthropic Messages, llama.cpp's `/completion`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/v1/rerank`, `/metrics`, `/slots` and `/props`,
  streaming, tool calls, structured output through xgrammar (`--structured-output`), token log
  probabilities and prompt grafts ([HTTP serving](docs/serving.md)).
- **Typed decisions.** `/v1/systemone` (TypeSafe's System One API, as llama.cpp, vLLM and Ollama
  serve it) and `/v1/decisions` (OpenAI's Decisions API) answer choice, yes/no and score questions
  about a state with the served chat model: one prompt per question, the answer read from the
  first token's exact label log probabilities, the state prefilled once and every question run
  after it as a branch of one read; up to 200 questions, 255 options, Runoff's order-independent
  choices, a conversation as the state, images and video as data URLs (with `--vision`), and a
  margin by which a client escalates near ties ([typed decisions](docs/serving.md#typed-decisions)).
- **Model suspend.** `--model-suspend` lets an idle server give its device memory back without
  exiting and take it again on the next request, retained conversations included. Ternary Bonsai 2
  27B on an RTX 3090: 7.9 GiB down to 0.3 GiB in 0.33 s, back in 1.1 s
  ([model suspend](docs/serving.md#model-suspend)).
- **Several models behind one server.** With `--models-dir` or a llama.cpp-style
  `--models-preset`, `ninfer-serve` is a router with llama.cpp's model API; with `--model-suspend` a
  model sleeps instead of unloading: two 27B models on one 24 GB RTX 3090 swap in 1.8 s instead of
  an 11.6 s cold load ([several models](docs/serving.md#several-models-router)).
- `GET /stats` and a [monitoring dashboard](tools/monitor/README.md), request logs with rotation,
  and an optional compiled-in WebUI.

</details>

<details>
<summary><b>Kernels and devices</b> — route profiles per card, pipeline stages, FP8 and NVFP4 tensor-core routes</summary>

- **Device route profiles.** Each card's measured profile picks the kernel route for every
  operation and width: built in for the cards in [supported GPUs](#supported-gpus), measured once
  at first start on any other ([device profiles](docs/device-profiles.md)).
- **Pipeline stages.** `--devices 0,1,...` splits the layers over up to eight GPUs on Linux, each
  stage owning its weights, KV and state; the cards may differ where the build runs on each, such as
  RTX 30 with RTX 40 cards on the `86` build ([pipeline stages](docs/maintainer/pipeline-stages.md)).
  `--tensor-parallel` divides every layer over the devices instead: each holds a share of the GDN
  heads and dense FFN rows of a dense GGUF model and reads only those per token
  ([tensor parallelism](docs/maintainer/pipeline-stages.md#tensor-parallelism)).
- FP8 A8 and NVFP4 W4A4 tensor-core routes on every `120a` build, an FP4 tensor-core prompt kernel
  for an NVFP4 KV cache, and programmatic dependent launches on Blackwell.

</details>

## Docker

The release pipeline packages two Debian 13 / Linux x64 images from the native Linux archives.
Each image contains one executable with all NInfer architectures supported by its CUDA variant.
The image build does not compile NInfer again.

| Image tag | Toolkit | Embedded GPU architectures |
|---|---|---|
| `ghcr.io/iamwavecut/ninfer-all:cuda12.9` | CUDA 12.9.2 | `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89` |
| `ghcr.io/iamwavecut/ninfer-all:cuda13.4` | CUDA 13.4.3 | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_120a` |

Choose `cuda12.9` for Volta and `cuda13.4` for Blackwell. Both cover Turing, Ampere and Ada.
Both support a layer pipeline (`--devices`) across different architectures included in that image.
The host needs a compatible NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Windows archives, Linux archives and Docker images use the same two architecture sets:

![GPU architectures included in NInfer's CUDA 12.9.2 and CUDA 13.4.3 builds](docs/assets/cuda-architectures.svg)

Each archive and image contains all device images inside its outline. These are the project's build targets,
not the complete list of architectures supported by NVIDIA's Toolkits.
See the [release process](docs/maintainer/release-process.md) for build and verification status.

<details>
<summary><b>Commands, volumes and tags</b> — what the container runs</summary>

- **Tags.** `cuda12.9` and `cuda13.4` update after both Linux CUDA builds and both Docker jobs succeed for the
  current `master`. Windows builds do not block them.
  Pin a build with `cuda12.9-<commit12>` or `cuda13.4-<commit12>`, where `commit12` is the first
  12 hexadecimal characters of its commit SHA. Both images must build before either floating tag
  moves. The pipeline no longer updates `latest`, `multi` or `VERSION` tags.
- **Volumes.** `/models` holds artifacts, `/cache` the device profile measured on first start, and
  `/grafts/<model>/` optional [prompt grafts](docs/serving.md#prompt-grafts).
- **Verification.** The new image variants still need a complete build and GPU inference checks.
  Earlier image tests on RTX 3090 and RTX 5090 do not establish these variants' runtime behavior.

The container's command chooses what runs:

| command | runs |
|---|---|
| `serve`, `ninfer`, `perplexity`, `calibrate` `[args]` | that binary with any artifact and flags; `serve` listens on `0.0.0.0:8080` unless given `--host` or `--port` |
| `run <model> [profile]` (the default: `run qwen38-27b`) | the [launcher profiles](docs/maintainer/launcher-profiles.md) of `scripts/run.sh` for the official `qwen38-27b` and `qwen36-35b-a3b`, with its `NINFER_*` overrides (`-e NINFER_SPEC=mtp`, `-e NINFER_CONTEXT=131072`, ...) |
| `download <model>` | `scripts/download-model.sh` into `/models`: `qwen38-27b`, `qwen36-27b` or `qwen36-35b-a3b` |

The official Qwen3.8-27B with the measured `tuned` profile, on http://localhost:8080/v1:

```bash
docker run --rm -v "$PWD/models:/models" ghcr.io/iamwavecut/ninfer-all:cuda13.4 download qwen38-27b
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 run qwen38-27b
```

[compose.yaml](compose.yaml) wires the GPU, the port and the volumes for the same:
`docker compose run --rm ninfer download qwen38-27b`, then `docker compose up -d`.
`NINFER_CUDA_TAG=cuda12.9 docker compose up -d` selects the Volta-capable image; Compose defaults
to `cuda13.4`. The published images use their multiarch binary automatically.
`docker build -t ninfer .` builds a separate Ubuntu image from source (`--build-arg ARCHS=86` for one architecture,
`--build-arg 'ARCHS=75;86;120a'` for the multi-architecture build).

</details>

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target ninfer-serve ninfer-calibrate
```

`CMAKE_CUDA_ARCHITECTURES` takes one build from [supported GPUs](#supported-gpus), or a list of them
for a pipeline across cards of different architectures
([several architectures](docs/build-linux.md#cards-of-several-architectures-in-one-pipeline)).

<details>
<summary><b>Toolchains and options</b> — CUDA versions per target, guides</summary>

`120a` runs the `mma.sync` compatibility path, which the ternary route needs. CUDA 12.8 or newer builds
`75`, `80`, `86` and `89`, and `70` needs CUDA 12.8 or 12.9. A `120a` build needs CUDA 13.1 or newer, since
12.8 and 12.9 miscompile sm_120a kernels and configure refuses them. The [Linux](docs/build-linux.md)
and [Windows](docs/build-windows.md) guides cover dependencies, the opt-in build options and the
launcher scripts; [tests](tests/README.md) and [benchmarks](bench/README.md) have their own guides.

</details>

## License

Apache-2.0, as upstream. The Bonsai artifact's weights come from PrismML, ProCreations and Qwen,
all Apache-2.0; its card lists the notices. The Qwen3.8-Flash-Next artifacts carry the Qwen
Community License 1.0 of their model.
