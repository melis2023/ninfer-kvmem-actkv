# TITAN RTX measurements, October 2026

What a `75` build ([Turing](../build-linux.md#turing-sm_75)) does on the TITAN RTX: which models
run on one 24 GB card and on two, the engine's prompt and decode rates per model, drafter and KV
format next to an RTX 3090 with the same artifacts, several requests at once, Flash-Next's expert
placements, perplexity against other cards, and where Turing's time goes. The setup is at the
[end](#how-these-were-measured).

## Models

Every row ran through the `ninfer` CLI on the `examples/cli` fixtures, greedy, and gave the
expected answers: the text check, the chat-history facts, the chart's text and the video's red
circle.

| model | artifact | one TITAN RTX | two, as pipeline stages |
|---|---|---|---|
| Ternary Bonsai 2 27B | [official](https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3) (8.9 GiB) | text, MTP, DFlash2, vision, video, all nine KV formats | |
| Qwen3.8-27B | [GSQ-RCO IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3) (14 GiB) | text, MTP, DFlash2, vision, video | text with MTP |
| Qwen3.8-27B | [official `groupwise-int`](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) (20 GiB) | text, chat history, MTP, vision on the GPU and on the CPU, video | |
| Qwen3.8-27B | [MXFP8](https://huggingface.co/WaveCut/Qwen3.8-27B-MXFP8-CRACK-NInfer-v3) (17 GiB) | text, MTP | |
| Qwen3.6-35B-A3B | official (22 GiB) | | text, MTP, DFlash, vision |
| Qwen3.8-Flash-Next | [GSQ-RCO Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) with its n-gram table | experts in pinned host memory or on disk | experts on the GPUs: text, MTP, vision |
| Gemma 4 26B-A4B | [QAT Q4_0 with MTP](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3) (15 GiB) | text, chat history, MTP, vision on the GPU and on the CPU, video | |

The 35B-A3B weights do not fit one 24 GB card beside the runtime.

## Engine rates

`ninfer_bench` on one TITAN RTX and on an RTX 3090 with the same artifacts, `rk8v4` KV (F16 for
Gemma 4), the mean of three runs after one warm-up. Prompt and decode in tokens per second:

| model | pp512 TITAN | RTX 3090 | ratio | pp2048 TITAN | RTX 3090 | ratio | tg128 TITAN | RTX 3090 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ternary Bonsai 2 27B | 1,304 | 3,026 | 0.43 | 1,375 | 3,108 | 0.44 | 63.4 | 93.9 | 0.68 |
| Qwen3.8-27B IQ3_S | 757 | 1,826 | 0.41 | 772 | 1,877 | 0.41 | 34.8 | 59.2 | 0.59 |
| Qwen3.8-27B official | 775 | 1,807 | 0.43 | 793 | 1,771 | 0.45 | 29.8 | 48.8 | 0.61 |
| Qwen3.8-27B MXFP8 | 810 | | | 832 | | | 30.9 | | |
| Gemma 4 26B-A4B | 2,532 | 5,068 | 0.50 | 3,571 | 6,976 | 0.51 | 158.8 | 222.9 | 0.71 |

The TITAN RTX reads memory at 672 GB/s against the RTX 3090's 936 GB/s (0.72), and one request's
decode runs at 0.59 to 0.71 of the RTX 3090. Prompts run at 0.41 to 0.51: the card draws its 280 W
limit through them (1,545 to 1,665 MHz in the samples taken), and its INT8 prompt GEMMs and GDN
recurrence take longer than the bandwidth and tensor-core rates alone account for
([Where the time goes](#where-the-time-goes)).

### Speculation

The same rows with the drafter, a 512-token prompt and 128 generated tokens; decode tokens per
second, with the drafts accepted and the tokens per verify round in brackets:

| model | drafter | TITAN RTX | RTX 3090 | ratio |
|---|---|---:|---:|---:|
| Ternary Bonsai 2 27B | MTP, 3 drafts | 106.8 (0.47, 2.40) | 174.1 (0.49, 2.46) | 0.61 |
| Ternary Bonsai 2 27B | DFlash2, 5 drafts | 92.1 (0.24, 2.19) | 149.1 (0.23, 2.15) | 0.62 |
| Qwen3.8-27B IQ3_S | MTP, 3 drafts | 41.6 (0.29, 1.87) | 79.1 (0.29, 1.87) | 0.53 |
| Qwen3.8-27B IQ3_S | DFlash2, 3 drafts | 32.5 (0.18, 1.52) | | |
| Qwen3.8-27B official | MTP, 3 drafts | 35.3 (0.34, 2.02) | 78.6 (0.32, 1.95) | 0.45 |
| Gemma 4 26B-A4B | MTP, 2 drafts | 290.7 (0.95, 2.91) | 435.2 (0.95, 2.91) | 0.67 |

A verify round multiplies several columns per weight byte. Where those contractions run on FP16
tensor cores (the T2, Q8 and Gemma paths) a round keeps the decode ratio; the official artifact's
Q4 and Q5 verify kernels do not ([Where the time goes](#where-the-time-goes)).

### KV formats

Ternary Bonsai 2, a 16,384-token prompt, then 64 decoded tokens, one run each:

| KV | bytes per token | prompt tok/s | decode tok/s |
|---|---:|---:|---:|
| `bf16` | 65,536 | 1,065 | 62.8 |
| `int8` | 33,792 | 1,199 | 62.6 |
| `fp8` | 33,024 | 601 | 59.7 |
| `rk8v4` | 26,112 | 1,251 | 62.1 |
| `rk4v4` | 17,920 | 1,234 | 62.0 |
| `rk4v4-e8` | 17,920 | 1,231 | 61.6 |
| `rk2v4-e8` | 13,824 | 1,217 | 61.6 |
| `nvfp4` | 18,432 | 628 | 62.0 |
| `k8v4` | 25,728 | 634 | 59.6 |

The RTX 3090 runs the same prompt at 2,875 tok/s with `rk8v4` and 2,687 with `bf16`, and decodes at
93.9 and 92.0. The FP8, NVFP4 and K8V4 prompts ran at about half the rate of the INT8 family here;
the FP8 prompt kernel has since moved to 16-key tiles ([below](#prompt-attention-tiles)).

### Depth

A 65,536-token prompt, then 64 decoded tokens, one run each:

| model | KV | prompt tok/s | decode tok/s | RTX 3090 |
|---|---|---:|---:|---|
| Ternary Bonsai 2 27B | `rk8v4` | 951 | 61.8 | 2,292 / 93.7 |
| Ternary Bonsai 2 27B | `int8` | 826 | 62.8 | |
| Ternary Bonsai 2 27B | `bf16` | 607 | 62.4 | |
| Qwen3.8-27B IQ3_S | `rk8v4` | 598 | 33.4 | |

At 16,384 tokens IQ3_S prompts at 701 tok/s and decodes at 33.1, Gemma 4 (F16 KV) at 2,660 and
158.5. That Gemma run predates the vendored combine-loop barrier: its full-attention blocks then
took llama.cpp's 8-column tile, and they now take the 32-column one
([Turing tier](../maintainer/build-system.md#turing-tier-75)).

### Prompt attention tiles

Median time of one complete prompt-attention launch at each tile, five interleaved rounds:

| Gemma 4 width-512 attention, query / KV tokens | RTX 2080 Ti, 8 / 32 columns, ms | Quadro RTX 6000, 8 / 32 columns, ms |
|---|---:|---:|
| 512 / 512 | 0.650 / 0.418 | 0.599 / 0.411 |
| 2,048 / 2,048 | 10.18 / 5.54 | 9.55 / 5.19 |
| 2,048 / 16,384 | 94.6 / 42.4 | 85.9 / 39.8 |
| 16,384 / 16,384 | 1,025 / 326 | 815 / 305 |

FP8 prompt attention takes 16-key tiles on Turing: on the RTX 2080 Ti they ran 13-15% faster than
the 32-key tile at every width from 128 to 4,096 queries, at 16 and 24 query heads. Both cards ran
at their default power limits (250 W, 260 W); both tiles passed the FP64 oracles, racecheck and
memcheck.

Whole model on the Quadro RTX 6000, `ninfer_bench -r 3`, the previous tiles and these run one after
the other: resident Gemma 4 26B-A4B with an F16 KV cache prefilled 512, 2,048 and 16,384 tokens at
2,580, 3,802 and 3,251 tok/s instead of 2,465, 3,503 and 2,624 (4.7%, 8.5% and 23.9% faster) and
decoded at 154.1 instead of 151.7. Resident Qwen3.6-35B-A3B with an FP8 KV cache prefilled 512,
4,096 and 16,384 tokens at 570, 717 and 602 tok/s instead of 566, 708 and 582 (0.7%, 1.3% and 3.4%)
and decoded at 131.3 instead of 131.0. In Nsight Systems the width-512 attention of a 2,048- and a
16,384-token Gemma prompt took 1.89 s instead of 3.70 s, and the FP8 prompt attention of a
16,384-token Qwen prompt 10.5 s instead of 12.2 s.

### Several requests at once

`ninfer-serve` with Ternary Bonsai 2, `rk8v4` and eight lanes; 1, 2, 4 and 8 different chat
requests sent at once, each answered greedily with 256 tokens. Generated tokens per second over the
wall time from the first request to the last answer, one run each:

| requests | TITAN RTX | with MTP, 3 drafts | RTX 3090 | with MTP, 3 drafts |
|---:|---:|---:|---:|---:|
| 1 | 54.8 | 94.9 | 73.2 | 133.3 |
| 2 | 119.3 | 161.1 | 172.2 | 255.6 |
| 4 | 211.9 | 247.5 | 308.2 | 404.6 |
| 8 | 327.1 | 308.0 | 498.0 | 521.3 |

The TITAN RTX serves 0.66 to 0.75 of the RTX 3090's rate without a drafter and 0.59 to 0.71 with
MTP. MTP pays on both cards up to four requests at once; at eight it decodes 6% slower than plain
decoding on the TITAN RTX and 5% faster on the RTX 3090.

### Qwen3.8-Flash-Next and the 35B-A3B MoE

`ninfer` CLI, a 144-token prompt (`scenario_code_cuda.json`) and 128 generated tokens, greedy, from a
cold expert cache:

| model | placement | decode tok/s |
|---|---|---:|
| Flash-Next Q2_0 | experts in pinned host memory (`--expert-residency host`), one card, 43.7% expert-cache hits | 53.5 |
| Flash-Next Q2_0 | experts on disk (`--expert-residency disk`), the artifact in the page cache, one card, 83.5% hits | 30.3 |
| Flash-Next Q2_0 | experts on the GPUs, two pipeline stages | 76.0 |
| Qwen3.6-35B-A3B | two pipeline stages | 146.6 |
| Qwen3.6-35B-A3B | two pipeline stages, MTP with 3 drafts (2.70 tokens a round) | 146.2 |

The host's PCIe 3.0 x16 links carry the host and disk experts. Over two stages the 35B-A3B's MTP
rounds gain nothing over plain decoding.

## Device profile

The built-in TITAN RTX entry is a contributor's `ninfer-calibrate` run on Windows (TCC mode, October
11) with the Q8 K-split bands calibrated on this host: the four-K-warp form (`kw4`) for the Q8
head's 248,320 rows (2.08 ms a call against 4.18 at one to eight columns), for the MTP and drafter
contractions of 6,144 to 34,816 rows (6,144 x 5,120 at one column: 100 us against 154), and for the
5,120 x 6,144 contraction from five columns. Two calibrations agreed on ten of the eleven Q8 keys;
the eleventh differed at one column, where the times were within 1%.

A full recalibration with the final kernels (70 keys in 96 s) agreed with a second run on 44 keys,
the others near ties, most of them INT8-family attention tiers. Against the built-in entry, in two
interleaved rounds, it decoded Ternary Bonsai 2 1.6% slower without a drafter and 1.7% faster with
MTP3, and the official artifact within 0.3% with and without MTP3, so the built-in entry stays.

## Where the time goes

Nsight Systems, the busiest kernels of each run on the TITAN RTX against the same work on the
RTX 3090: kernel time per call in decode, and per 2,048-token prompt (64 layers, two chunks) in
prefill:

| run | work | TITAN RTX | RTX 3090 | ratio |
|---|---|---:|---:|---:|
| official, MTP3 decode | Q8 head, 248,320 x 5,120 (Q8 K-split, `kw4` on the TITAN) | 2,155 us | 1,534 us | 1.4 |
| official, MTP3 decode | gate and up, 34,816 x 5,120 Q4, 4 columns (Q4 K-split) | 212 us | 112 us | 1.9 |
| official, MTP3 decode | MLP down, 5,120 x 17,408 Q5, 4 columns | 261 us (SIMT split) | 75 us (Q5 small-T MMA) | 3.5 |
| official, MTP3 decode | attention input, 12,288 x 5,120 Q5, 4 columns | 120 us (SIMT split) | 55 us (Q5 small-T MMA) | 2.2 |
| official, 2,048-token prompt | MLP down, Q5 A8 GEMM | 560 ms | 238 ms | 2.4 |
| official, 2,048-token prompt | gate and up, Q4 A8 SwiGLU GEMM | 755 ms | 427 ms | 1.8 |
| official, 2,048-token prompt | all kernels | 2,644 ms | 1,246 ms | 2.1 |
| Bonsai, 2,048-token prompt | T2 INT8 GEMMs over the 5,120-wide hidden state | 701 ms | 370 ms | 1.9 |
| Bonsai, 2,048-token prompt | GDN two-stage chunk recurrence | 264 ms | 23 ms | 11 |
| Bonsai, 2,048-token prompt | all kernels | 1,516 ms | 707 ms | 2.1 |

- The Q8 head runs at the cards' bandwidth ratio (1.39). The Q4 K-split and Q5 small-T contractions
  run FP16 tensor cores on Turing, on operands decoded to FP16, but still spend their issue slots
  on decoding and conversion: in calibration the Q5 small-T kernel took 272 us at four columns
  against 225 us for the SIMT split kernel, which the profile therefore picks; both stream the
  weights at about a third of the card's bandwidth.
- The prompt chunk is aligned to the attention grid: 1,152 tokens on 72 SMs where 1,024 are
  requested. The groupwise A8 GEMMs of the official and MXFP8 artifacts pad such a chunk to three
  512-token tiles; before, they ran it as nine 128-token tiles, each of which streams the whole
  weight, and the MLP-down GEMM took 808 ms a prompt instead of 560. In two interleaved rounds
  the padding raised the official artifact's prompts from 676 and 667 to 797 and 795 tok/s at
  2,048 tokens and from 643 and 613 to 720 and 712 at 8,192; 512-token prompts, one tile either
  way, stayed within 1.5%, and Ternary Bonsai 2, whose T2 GEMMs already padded, within 1.6%. With
  the requested chunk (`NINFER_PREFILL_ALIGN=0`) nothing is padded: the official artifact then
  prompts at 861 and 809 tok/s, 8 and 13% faster than aligned, while Ternary Bonsai 2 prompts 2 to
  8% faster aligned.
- The GDN recurrence multiplies TF32 operands. Turing runs a TF32 product on FP16 tensor cores
  only for outputs whose operands are all FP16 values, and in FP32 for the others; the 11-fold gap
  suggests that most of the recurrence's tiles take the FP32 product (not counted per tile). The
  two-stage recurrence is still 2.5 times faster than the WY pipeline on this card (8.1 against
  20.4 ms at 8,192 tokens in calibration).

## Quality

Perplexity on one TITAN RTX against the same artifact and protocol on other cards:

| model | protocol | TITAN RTX | reference |
|---|---|---:|---:|
| Ternary Bonsai 2 27B | `ninfer-ppl-1m-v1 --quick`, `rk8v4` | 5.6314 | 5.631 RTX 3090, 5.6305 Tesla V100 |
| Ternary Bonsai 2 27B | the same, `bf16` | 5.6272 | |
| Ternary Bonsai 2 27B | the same, `rk2v4-e8` | 5.8189 | |
| Qwen3.8-Flash-Next Q2_0 | the first 40,000 bytes of the corpus's WikiText stream (`wikitext/00.txt`), INT8 KV | 2.53655, host and disk experts alike | |

A `bf16` cache on Turing multiplies Q, K and V as their FP16 roundings
([Turing tier](../maintainer/build-system.md#turing-tier-75)).

## How these were measured

- **Cards.** Two TITAN RTX (TU102, 72 SMs, 24 GB) at their 280 W limit, PCIe 3.0 x16, in a container
  on a host with a Xeon E5-2673 v3 (24 threads) and 62 GB of RAM, Ubuntu 24.04, driver 570.172.08.
  One card ran every single-card row (`CUDA_VISIBLE_DEVICES=0`); the pipeline-stage rows used both.
  The comparison card is an RTX 3090 at a 380 W limit, PCIe 4.0 x16, AMD EPYC 7702, driver 595.84.
- **Builds.** `-DCMAKE_CUDA_ARCHITECTURES=75` and `86`, CUDA 12.8.93, GCC 13.3, Release, from the
  same source. Each card used its built-in device route profile. The official, MXFP8 and Gemma 4
  rows, the official prompt profile and the serving rows ran on the final source; the other rows
  on the source before the A8 GEMM padding and the width-512 tile change, which touch only those
  GEMMs and Gemma 4's prompt attention.
- **Engine rates.** `ninfer_bench -r 3 --warmup 1 --max-ctx 4096` (`-r 1` and `--max-ctx 17408`
  or `66560` for the long prompts); speculative rows with `--ngram-draft-tokens 0`.
- **Serving.** `ninfer-serve <artifact> --max-concurrency 8 --max-context 8192 --kv-capacity 65536
  --kv-dtype rk8v4` (and `--spec mtp --draft-tokens 3`); the requests are OpenAI chat completions
  with `temperature` 0, `max_tokens` 256 and thinking off.
- **CLI rows.** `ninfer <artifact> --messages examples/cli/messages/<fixture> --no-thinking --greedy`.
- **Profiles.** Nsight Systems 2024.6 (`--trace=cuda --cuda-graph-trace=node`) over one warm-up
  and one repetition of `ninfer_bench`; the prompt rows are half of the traced time.
- **Checks.** The full CTest suite passed over the two cards (294 tests, 28 skipped for want of an
  artifact or a feature; the GGUF flash-attention test again after the width-512 tile change), as
  did the real-artifact tests of every model above and, on the final source, the 44 CLI checks
  behind the Models table. The official artifact's suspend test fails on this host: its resume
  reads 16.4 GiB at about 530 MiB/s, past the 30 s within which a waiting request is admitted
  (IQ3_S resumes in 21 s).
