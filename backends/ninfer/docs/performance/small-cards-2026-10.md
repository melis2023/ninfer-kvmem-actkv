# 8, 11 and 12 GB cards, October 2026

What runs on an RTX 3060 Ti (8 GB), an RTX 2080 Ti (11 GB) and an RTX 3060 (12 GB) from the
published Docker image: which models load, their prompt and decode rates from the CLI, the largest
context the server starts with, and the MoE models whose routed experts stay in host memory or on
disk: Qwen3.8-Flash-Next, Gemma 4 26B-A4B and Qwen3.6-35B-A3B. The setup is at the
[end](#how-these-were-measured).

## Models

| model | RTX 3060 Ti, 8 GB | RTX 2080 Ti, 11 GB | RTX 3060, 12 GB |
|---|---|---|---|
| Ternary Bonsai 2 27B (6.70 GiB of weights; 7.28 GiB with the MTP head) | text | text, MTP | text, MTP |
| Qwen3.8-27B GSQ-RCO IQ3_S (10.9 GiB) | | | text; MTP does not fit |
| Gemma 4 26B-A4B QAT Q4_0 (13.4 GiB; 1.7 GiB on the GPU with streamed experts) | | | experts in host memory or on disk |
| Qwen3.6-35B-A3B groupwise (19.6 GiB of weights; 2.3 GiB on the GPU with streamed experts) | | | experts in host memory or on disk |
| Qwen3.8-Flash-Next GSQ-RCO Q2_0 (2.83 GiB on the GPU) | experts in host memory or on disk | not run | experts in host memory or on disk |

On the RTX 3060 Ti, Bonsai's MTP head leaves the runtime 241 MiB where it needs 693 MiB. Bonsai's
output head is not stored as row-split Q8_G32, so `--lm-head-q4` and `--embedding-q4` do not apply
to it. IQ3_S with MTP stops at the first device allocation of its weights; Gemma 4 with resident
experts does too.

## Ternary Bonsai 2 27B

`ninfer` CLI, a 2,273-token prompt and 256 generated tokens, `--max-context 4096 --kv-dtype int8`,
one run each; MTP with 3 drafts. Prompt and decode in tokens per second:

| card | prompt | decode | with MTP: prompt | decode | tokens a round | peak device memory |
|---|---:|---:|---:|---:|---:|---:|
| RTX 3060 Ti, 8 GB | 1,160 | 50.1 | does not fit | | | 7.3 GiB |
| RTX 2080 Ti, 11 GB | 1,080 | 58.5 | 962 | 95.9 | 2.33 | 7.3 GiB; 7.9 GiB with MTP |
| RTX 3060, 12 GB | 937 | 40.5 | 884 | 69.2 | 2.31 | 7.3 GiB; 7.9 GiB with MTP |

The largest `--max-context` (= `--kv-capacity`, one request) at which the server starts and answers
a request, bisected in 4,096-token steps, with `--gdn-state-fp16`:

| card | `rk4v4` | `int8` | `rk4v4`, MTP 3 drafts |
|---|---:|---:|---:|
| RTX 3060 Ti, 8 GB | 24,576 | | does not fit |
| RTX 2080 Ti, 11 GB | 200,704 | 106,496 | 143,360 |
| RTX 3060, 12 GB | 266,240 | | |

## Qwen3.8-27B GSQ-RCO IQ3_S

On the RTX 3060 (12 GB), the same CLI request: 567 prompt and 25.8 decode tokens per second, 11.6
GiB of device memory at the peak. The largest context with `rk4v4` is 8,192 tokens.

## Qwen3.8-Flash-Next

The Q2_0 release with its n-gram table read from disk, `ninfer` CLI, `--max-context 32768`, BF16
KV, 128 generated tokens, a cold expert cache in every run. The artifact and the table had just been
downloaded and hashed, so the page cache held most or all of them. Prompt and decode in tokens per
second:

| card | experts | expert cache on the GPU | 69-token prompt: prompt / decode | 2,273-token prompt: prompt / decode | peak device memory | peak host memory |
|---|---|---:|---|---|---:|---:|
| RTX 3060 Ti, 8 GB | on disk | 1.32 GiB | 52.8 / 17.6 | 356 / 16.4 | 6.2 GiB | 1.3 GiB |
| RTX 3060 Ti, 8 GB | pinned host memory | 0.91 GiB | 73.3 / 33.0 | 723 / 30.6 | 7.0 GiB | 32.7 GiB |
| RTX 3060, 12 GB | on disk | 5.32 GiB | 37.9 / 23.5 | 215 / 22.9 | 10.2 GiB | 1.2 GiB |
| RTX 3060, 12 GB | pinned host memory | 4.91 GiB | 58.9 / 39.9 | 569 / 38.2 | 11.0 GiB | 32.7 GiB |

## Gemma 4 26B-A4B

The QAT Q4_0 release with its routed experts in pinned host memory or in the artifact's files
(`--expert-residency host|disk`), the automatic device cache (7.5 GiB, 2,413 of the 3,840 experts),
prompt chunks of 2,048 tokens. `ninfer_bench -p 1024,8192 -n 128 -r 3 --warmup 1 --max-ctx 9000`,
mean tokens per second:

| card | experts | 1,024-token prompt | 8,192-token prompt | decode |
|---|---|---:|---:|---:|
| RTX 3060, 12 GB | pinned host memory | 1,948 | 2,347 | 93.7 |
| RTX 3060, 12 GB | artifact files | 489 | 1,027 | 94.6 |

Over each whole run, 97% of the token-expert pairs found their expert in the device cache. Each
prompt chunk still needs most of each layer's experts, so with experts on disk a prompt waits for
the file reads.

## Qwen3.6-35B-A3B

The native groupwise release (Q4 gate/up, Q5 down experts) with the same placements, the automatic
device cache (7.1 GiB) and prompt chunks of 4,096 tokens, the same benchmark; MTP3 adds
`--spec mtp --draft-tokens 3`:

| card | experts | 1,024-token prompt | 8,192-token prompt | decode | MTP3 decode |
|---|---|---:|---:|---:|---:|
| RTX 3060, 12 GB | pinned host memory | 1,039 | 3,058 | 59.8 | 35.8 |
| RTX 3060, 12 GB | artifact files | 339 | 1,385 | 34.9 | |

With 1,024-token chunks (`--prefill-chunk 1024`) the 8,192-token prompt ran at 1,070 tokens per
second: every chunk carries most of the expert bank across the bus, so the default chunk is 4,096
for experts off the GPU. Over the host run 48% of the token-expert pairs found their expert in the
cache. MTP3 decoded slower than plain decode on this card: with the draft layer loaded the cache is
6.1 GiB instead of 7.1 GiB, and each verification needs the experts of four tokens. Which of the two
costs more was not measured.

## How these were measured

- **Cards.** Rented single-GPU hosts, each GPU at its default power limit and without PCIe replay
  errors: an RTX 3060 Ti at 200 W (EPYC 7282, PCIe 4.0 x16, 170 GiB of container memory), an RTX
  2080 Ti at 250 W (PCIe 3.0, x8 by `nvidia-smi`, 85 GiB) and an RTX 3060 at 170 W (EPYC 7452,
  PCIe 4.0 x16, 85 GiB). Drivers 595.71, 595.91 and 580.159.
- **Software.** The `ghcr.io/iamwavecut/ninfer-all` image at `sha-e408e428ae86`, its `sm_86` and
  `sm_75` builds. None of these cards had a built-in device profile at that commit: each calibrated
  its routes at first start, and that start took 82 s (RTX 3060 Ti), 101 s (RTX 3060) and 107 s (RTX 2080 Ti).
- **CLI rows.** One run each, greedy.
- **Largest context.** `ninfer-serve <artifact> --max-context W --kv-capacity W --max-concurrency 1
  --gdn-state-fp16 --kv-dtype KV`, bisected in 4,096-token steps; a step passes when the server
  starts and answers one 16-token chat request.
- **Gemma 4 and Qwen3.6 rows.** A second RTX 3060 at 170 W (Xeon Gold 6154, PCIe 3.0 x16, 60 GiB of container
  memory, driver 595.71), an `sm_86` build from source of the streaming branch, its routes
  calibrated at first start, nothing else running. The disk rows read files that earlier runs had
  read; how much of them the page cache held was not controlled.
- **Concurrent work.** On the RTX 3060 and the RTX 3060 Ti the Bonsai and IQ3_S rows ran while the
  host downloaded and hashed the Flash-Next files (one or two of its 15 to 16 cores). The RTX 2080
  Ti ran nothing else.
