# Qwen3.8-Flash-Next

NInfer runs Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, 125B text parameters, about 6B
active per token) from ISTA-DASLab's GSQ-RCO GGUF releases, converted without requantization:

- [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF):
  Q2_0 (37.6 GB), IQ2_XS (39.2 GB), IQ3_XXS (47.0 GB), IQ3_S (54.8 GB);
- [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF).

Each release is two GGUF shards: the model, and the 28.8 GB n-gram table of the model's per-layer
embedding (PLE), byte for byte the same in every release, the Coder build's included. A NInfer
model artifact describes the table it reads in its `ngram` component (the hash constants, the row
format and the SHA-256 of the rows) and either stores the rows too, as one self-contained file, or
leaves them to a table artifact of their own that every Flash-Next model can share. Nothing reads
the table at load; each token reads the 16 rows it addresses from the file, or `--ngram-ram` loads
the table into RAM. A model stored without its rows takes them from `--ngram-table PATH`, which
must hold the table the model names; without a table the engine refuses to start
([running without it](#without-the-n-gram-table) is an experiment, not a mode).

The published conversions store the models without the table, which is published once:

| Artifact | Size | |
|---|---:|---|
| [n-gram table](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.82 GiB | IQ4_NL rows, read by every model below |
| [Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 35.89 GiB | with the Vision tower |
| [IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 51.90 GiB | with the Vision tower |
| [Coder IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 28.42 GiB | 256 experts per layer, with the Vision tower |

The model runs with up to eight concurrent requests, a context cache of prompt prefixes, structured
output, and images and video through its Vision tower (`--vision`, from an artifact converted with
the tower). MTP drafting is not available yet: no GSQ-RCO release carries the MTP layer, and the engine
does not read the MTP modules published separately. The [plan](maintainer/qwen3-8-flash-next-plan.md)
records what MTP needs here and what other engines measured with it.

## Convert

`--model` needs only the HF checkpoint's configuration and tokenizer files (`config.json`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`) from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

```bash
# The model with its n-gram table, one self-contained file (--components text,ngram, the default).
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0.ninfer

# Or the table once, as an artifact of its own, and each release without it.
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components ngram \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 --out models/flash-next-ngram-table.ninfer
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components text \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0.ninfer
```

Every conversion reads the table shard once more to hash it (a minute or so for 28.8 GB), since
the model records the digest of the table it reads whether or not it stores the rows.

`--components text,vision` (with or without `ngram`) adds the Vision tower from the release's
`mmproj-Qwen3.8-Flash-Next-BF16.gguf` (`--source vision=PATH`): the same tower as Qwen3.5/3.6
(27 blocks of width 1152, merging 2×2 patches onto the text model's 2,560), kept in BF16, 0.9 GB.
`--model` then also needs `preprocessor_config.json` and `video_preprocessor_config.json`.

Every matrix keeps the block type the release chose (see [GGUF block formats](gguf.md)); the expert
banks keep the exporter's expert-major layout, so one expert is one contiguous range of bytes. The
recipe undoes llama.cpp's exporter conventions as the Qwen3.8-27B GGUF recipe does (grouped GDN
value heads, `1 + w` norms, `A_log`, the head-interleaved query and gate). The n-gram table keeps
the release's IQ4_NL rows, and its component carries the hash constants they were written for; the
runtime derives them again from the model's configuration and refuses a table that disagrees, or a
table artifact whose digest or row format differs from the one the model names.

## Run

```bash
# One GPU with room for every expert (48 GB for Q2_0, the 96 GB RTX PRO 6000 for IQ3_S).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --max-context 32768

# Two 24 GB GPUs (two 32 GB for IQ3_S): every expert in device memory, one pipeline stage per GPU (Linux).
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --devices 0,1 --max-context 32768

# One GPU: the experts in pinned host memory, the most used of them cached on the GPU.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --expert-residency host --max-context 32768

# One GPU and little RAM: the experts stay in the artifact's files and stream into a GPU cache.
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --expert-residency disk --max-context 32768
```

A model converted with its table needs no `--ngram-table`. `ninfer` and `ninfer-perplexity` take
the same options, and the Docker image's `serve` command takes them with the files under `/models`
([Running](../README.md#running)).

| Option | Meaning |
|---|---|
| `--expert-residency device\|host\|disk` | expert banks in the stage devices' memory (default); in page-locked host memory that the expert kernels read across the bus; or left in the artifact's files, each layer's routed experts read into a device cache before they run |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts: `auto` (default) takes what each device has free after startup less a margin; `0` disables the host-mode cache (disk mode needs one) |
| `--ngram-table PATH` | the table artifact to read the n-gram rows from; required for a model stored without its table, and it must hold the table the model names (same SHA-256 and row format) |
| `--ngram-ram` | load the 28.8 GB n-gram table into RAM instead of reading the 16 rows each token needs from its file |
| `--no-ngram-table` | run without the n-gram table: see [below](#without-the-n-gram-table) |
| `--devices A,B,...` | one pipeline stage per GPU; layers are split so that every stage holds about the same stored bytes (`--stage-layers` overrides) |

With host experts the GPU holds only the dense weights (3.7 GB for the Q2_0 release), the
per-request state and the expert cache; the host needs the expert banks in page-locked memory
(34 GB for Q2_0). With disk experts the host needs no copy of the banks at all: the expert cache
reads the missing experts from the artifact's files through the OS page cache, eight reads in
flight, into a 256 MB page-locked staging ring, so the page cache keeps whatever the system can
spare and the rest comes from the disk. The n-gram rows are read the same way unless `--ngram-ram`
is given.

### Without the n-gram table

The model was trained with its n-gram embedding, so the engine refuses a model whose table it
cannot find. `--no-ngram-table` starts it anyway, without the PLE injection (exactly what an
all-zero table gives), and warns at startup: this is a non-standard, experimental mode with no
practical use. On the Q2_0 release it nearly doubles WikiText-2 perplexity, 2.66 to 5.01 over the
fourteen windows of the comparison below; short factual answers survive, but nothing measured
improves.

`--kv-dtype` stores the KV cache of the 12 sparse-attention layers in any of the nine formats the
Qwen3.5 family uses. A position costs 24 KiB in `bf16` (the default), 12.4 KiB in `int8`, 12.1 KiB
in `fp8`, 9.6 KiB in `rk8v4`, 9.4 KiB in `k8v4`, 6.8 KiB in `nvfp4`, 6.6 KiB in `rk4v4` and
`rk4v4-e8`, and 5.1 KiB in `rk2v4-e8`; the indexer's pooled keys stay FP32 beside it, another
1.5 KiB a position in every format. The sparse attention decodes each row it reads, keys in the
rotated basis the cache stores them in. What the quantized formats cost in quality has not been
measured on this model yet.

`--max-concurrency N` (one to eight) runs that many requests at once, each on its own sequence with
its own KV and recurrent state, so every sequence costs device memory (the KV of `--max-context`
positions, 24 KiB a position in BF16 and less in a quantized `--kv-dtype`, plus 74 MiB of recurrent
state). Requests are admitted
in arrival order. Prompts prefill one at a time, a chunk at a time; between two chunks every request
that is decoding produces one token, all of them in one batched pass whose experts read their
weights once for the whole batch, so the batch costs little more than one token while the experts
dominate the step.

With the context cache on (the default), a sequence keeps its state when its request ends, and a
snapshot of its recurrent state where the prompt's last user turn closes (or at the prompt's end
when the template marks no turn), 74 MiB on the device: a later prompt that continues what the
sequence holds resumes from its live state, and one that repeats the prompt up to that point (the
next turn of a chat, which renders the previous answer without its reasoning) resumes from the
snapshot.
Either way only the new tokens are prefilled; the response's prompt summary reports how many were
reused. A request goes to the free sequence that holds the longest such prefix of its prompt.
`--no-prefix-reuse` (ninfer-serve) prefills every prompt from scratch.

Structured output (`--structured-output` for the server, `--json`/`--json-schema` for the CLI) works
as for the Qwen3.5 family: the grammar's token mask applies to every sampled token. So do
[token log probabilities](serving.md#token-log-probabilities): a request that asks gathers each
sampled token's top 20 from the same logits before sampling.

`--vision` loads the Vision tower of an artifact converted with it (0.9 GB of BF16 weights on the
first device, beside the token embedding) and takes images and video as the Qwen3.5 family does:
the frontend renders the media tokens, the tower encodes the prompt's media before its first
chunk, and their merged embeddings replace those tokens' embeddings. A media prompt rotates its
positions on the three RoPE axes the frontend computes (text positions on all three, then each
later token at its index plus the prompt's offset); it prefills in a pass of its own and is not
kept for reuse by the context cache.

## Execution

- The residual is a four-stream hyper-connection stack kept in FP32 between layers.
- The 36 Gated DeltaNet layers run the Qwen3.5 GDN kernels with a sigmoid output gate; the 12 sparse
  attention layers run the block indexer and attend only to the blocks it selects (plain dense
  attention below 2,051 positions).
- The PLE layer reads its 16 n-gram rows per token from the table's file (IQ4_NL rows decoded on
  the GPU).
- The 512-expert MoE groups each layer's (token, expert) pairs by expert on the GPU and runs one
  pass over each selected expert's rows for all of its tokens, through device tables of expert
  base pointers: an expert is read wherever the table points, in device memory, a cache slot or the
  pinned host block. Up to eight tokens run vector products; wider calls run ggml's integer
  tensor-core matrix kernel over the routed pairs from device memory. That kernel reads a 640-value
  down row in 256-value steps, so it decodes the bytes after a down matrix as the down's blocks:
  in a bank they are the next expert's down or zeros after the last one, and every cache or stream
  slot keeps zeros after its down, which a smaller down from another layer does not uncover (another
  format's bytes there can hold a non-finite scale, and NaN follows). With host experts, a wide call
  first copies the routed experts the cache does not hold into a device pool (one slot per expert on
  each GPU, 0.7 GB for Q2_0), so each expert crosses the bus once per chunk. Weighted expert outputs
  are summed in fixed point, so the result does not depend on the order experts finish in.
- The expert cache counts the routes each forward pass took (decayed per token) and, between
  passes, copies the experts it needed most into its slots and points the tables at them. With
  disk experts the cache works per layer instead: once a layer has routed its tokens, the experts
  it lacks are read into the slots least recently used (never one the same call needs), and only
  then do the layer's experts run.
- A decode step (one token) replays a CUDA graph per pipeline stage, captured at a sequence's
  second decode step; the token's position reaches the sparse-attention kernels in device memory.
  Disk experts need the host between a layer's routing and its experts, so their steps stay eager.
  `--no-cuda-graph` decodes eagerly everywhere.

## Measurements

Q2_0 release, greedy decoding, CUDA 12.8, 2026-10-05. Decode is measured over the 78 tokens of a
short answer (prompt of 20 tokens) and over the first tokens after a 4,463-token prompt; prefill is
that prompt in 512-token chunks.

| Hardware and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| 2× RTX 3090 Ti (PCIe, no P2P), experts on the GPUs (`--devices 0,1`) | 18.4 + 19.4 GB | 0.7 GB | 90.2 tok/s | 107 tok/s | 1,504 tok/s |
| RTX 3090, experts in pinned host memory, 17.7 GB expert cache | 22.6 GB | 34 GB pinned | 48.9 tok/s | 42.1 tok/s | 842 tok/s |
| RTX 3090, experts on disk, artifact in the page cache | 22.6 GB | 0.9 GB + page cache | 47.0 tok/s | 39.1 tok/s | 630 tok/s |
| RTX 3090, experts on disk, page cache dropped every second (NVMe) | 22.6 GB | 0.9 GB | 17.2 tok/s | 11.3 tok/s | 195 tok/s |

Host memory is the process's peak resident set (the pinned bank for host experts). Decode after
the long prompt covers its first five tokens only. CUDA graphs add 11% to the short-answer decode on
the two GPUs (81.2 tok/s eager) and 7% with host experts (44.3 tok/s); disk experts decode eagerly.
Prefill touches nearly every expert of every layer in each chunk, and with host or disk experts
each of them crosses the bus or comes off the disk once per chunk, so larger `--prefill-chunk`
values serve more tokens per copy: with host experts on an RTX 3090 Ti the long prompt takes 6.05 s
in 512-token chunks and 4.58 s in 2,048 (10.97 s before the experts went through device slots).

The IQ3_S release on one RTX 3090 (310 W power limit, PCIe 4.0 x16) in a host with 62 GB of RAM,
23 cores of an AMD EPYC 7663 and an NVMe drive, 2026-10-05, with the Q2_0 release in the same
sitting; the artifacts are single files with their n-gram tables, and decode counts the 86 tokens of
the short answer:

| Release and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| IQ3_S, experts in pinned host memory, 16.2 GB expert cache | 22.1 GiB | 50.3 GB pinned | 34.4 tok/s | 30.2 tok/s | 533 tok/s |
| IQ3_S, experts on disk, the page cache holding what of the 83.6 GB file fits | 22.1 GiB | page cache | 19.0 tok/s | 21.6 tok/s | 209 tok/s |
| IQ3_S, experts on disk, the file's pages evicted every second | 22.1 GiB | — | 10.7 tok/s | 5.4 tok/s | 47 tok/s |
| Q2_0, experts in pinned host memory | 22.1 GiB | 34.0 GB pinned | 50.2 tok/s | 43.7 tok/s | 847 tok/s |

The Q2_0 release (with the separate table artifact) on RTX 4090s at 450 W (PCIe 4.0 x16, no peer
access) in a cloud VM with two EPYC 7543 sockets of 60 vCPUs each, both GPUs on NUMA node 0, 694 GB
of RAM and its container disk, 2026-10-06. The single-GPU rows ran pinned to node 0's CPUs
(`taskset -c 0-59`; the container refuses a memory policy, so first touch places the pages there);
the ranges are two runs:

| Hardware and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| 2× RTX 4090, experts on the GPUs (`--devices 0,1`) | 18.5 + 19.4 GiB | 0.8 GiB | 92.6-100.7 tok/s | 116-120 tok/s | 3,482-3,734 tok/s |
| RTX 4090, experts in pinned host memory, 15.9 GB expert cache | 20.7 GiB | 34.0 GB pinned | 52.6-52.7 tok/s | 44.3-44.6 tok/s | 1,137-1,139 tok/s |
| RTX 4090, experts on disk, artifact in the page cache | 20.7 GiB | 1.0 GiB + page cache | 42.0 tok/s | 45.6 tok/s | 723 tok/s |
| RTX 4090, experts on disk, the artifact's and table's pages evicted every second | 20.5 GiB | 1.0 GiB | 17.7 tok/s | 12.2 tok/s | 160 tok/s |

Left unpinned on that VM, the host-expert row decodes at 39.5-40.1 tok/s and prefills at 775-827
tok/s, the page-cache row at 31.9-33.0 tok/s and 571-674 tok/s, and the evicted row at 15.7 tok/s
and 144 tok/s: the copies out of host memory then cross the socket link. CUDA graphs add 1% and 10%
to the short-answer decode on the two GPUs in the two runs (91.5-91.7 tok/s eager) and 5 to 6% with
pinned host experts (49.9-50.1 tok/s eager). Host memory is the process's peak resident set, device
memory the most `nvidia-smi` showed in use.

Both releases on Blackwell, 2026-10-06, a `120a` build with CUDA 13.1, the table as the separate
artifact, the single-GPU rows pinned to the CPUs of the GPU's NUMA node, two runs where a range is
given:

- RTX PRO 6000 Blackwell Workstation Edition: 600 W, a PCIe 4.0 x16 host link, 64 vCPUs and
  1.1 TB of RAM.
- RTX 5090s: boards with a 600 W default limit, PCIe 5.0 x16, 512 threads and 1 TB of RAM.

The short answer is 78 tokens for Q2_0, and 87 (PRO 6000) or 81 (RTX 5090) for IQ3_S.

| Hardware and placement | Release | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---|---:|---:|---:|---:|---:|
| RTX PRO 6000, experts on the GPU | Q2_0 | 37.7 GiB | 0.8 GiB | 136.0-139.7 tok/s | 160.7-162.6 tok/s | 2,925-2,952 tok/s |
| RTX PRO 6000, experts on the GPU | IQ3_S | 53.7 GiB | 1.0 GiB | 123.8-125.9 tok/s | 146.8-146.9 tok/s | 2,540-2,541 tok/s |
| RTX PRO 6000, experts in pinned host memory | Q2_0 | 38.5 GiB | 34.0 GB pinned | 54.8-55.0 tok/s | 67.7-70.0 tok/s | 1,296-1,297 tok/s |
| RTX PRO 6000, experts in pinned host memory | IQ3_S | 55.0 GiB | 50.3 GB pinned | 29.3 tok/s | 31.2 tok/s | 803 tok/s |
| RTX PRO 6000, experts on disk, the files in the page cache | Q2_0 | 93.6 GiB | 1.0 GiB + page cache | 75.9 tok/s | 115.8 tok/s | 2,024 tok/s |
| RTX PRO 6000, experts on disk, the files in the page cache | IQ3_S | 93.5 GiB | 1.2 GiB + page cache | 65.6 tok/s | 106.0 tok/s | 1,661 tok/s |
| RTX PRO 6000, experts on disk, the files' pages evicted every second | Q2_0 | 93.5 GiB | 1.0 GiB | 35.1 tok/s | 87.1 tok/s | 1,000 tok/s |
| RTX PRO 6000, experts on disk, the files' pages evicted every second | IQ3_S | 93.5 GiB | 1.2 GiB | 28.7 tok/s | 78.6 tok/s | 773 tok/s |
| 2× RTX 5090, experts on the GPUs (`--devices 0,1`) | Q2_0 | 18.8 + 19.6 GiB | 0.9 GiB | 136.1-136.3 tok/s | 157.0-157.2 tok/s | 3,807-3,811 tok/s |
| 2× RTX 5090, experts on the GPUs (`--devices 0,1`) | IQ3_S | 26.5 + 27.8 GiB | 1.0 GiB | 121.6-121.9 tok/s | 141.8-142.0 tok/s | 3,310-3,341 tok/s |
| RTX 5090, experts in pinned host memory | Q2_0 | 30.0 GiB | 34.0 GB pinned | 74.1-74.2 tok/s | 93.9-94.2 tok/s | 1,676-1,678 tok/s |
| RTX 5090, experts in pinned host memory | IQ3_S | 30.0 GiB | 50.3 GB pinned | 40.7 tok/s | 41.0 tok/s | 1,100 tok/s |
| RTX 5090, experts on disk, the files in the page cache | Q2_0 | 29.9 GiB | 1.0 GiB + page cache | 67.5 tok/s | 76.7 tok/s | 1,739 tok/s |
| RTX 5090, experts on disk, the files in the page cache | IQ3_S | 29.9 GiB | 1.1 GiB + page cache | 55.3 tok/s | 44.3 tok/s | 439 tok/s |
| RTX 5090, experts on disk, the files' pages evicted every second | Q2_0 | 29.9 GiB | 1.0 GiB | 25.3 tok/s | 28.7 tok/s | 694 tok/s |
| RTX 5090, experts on disk, the files' pages evicted every second | IQ3_S | 29.9 GiB | 1.1 GiB | 18.8 tok/s | 14.9 tok/s | 171 tok/s |

**Device expert cache.** With host or disk experts the device expert cache takes the memory the GPU
has free, and the generate test answers its prompts right after the load. The rows therefore
measure a cache that is still filling: 94 GB of it on the PRO 6000, where nearly every expert ends
up resident, and 30 GB on the RTX 5090.

**Host link.** Host and disk experts cross the host link: PCIe 5.0 x16 on the RTX 5090 host,
PCIe 4.0 x16 on the PRO 6000 host. That is the likeliest reason the RTX 5090's host-expert row beats
the PRO 6000's.

**CUDA graphs.** Graphs add 27% to the short-answer decode on the two RTX 5090s (107.0 tok/s eager)
and 23 to 26% on the PRO 6000 (111.0-111.2 tok/s eager).

On the current code every GSQ-RCO release, converted into one file with its table, answers the
generate test's prompts (the facts and the 4,463-token needle) on that card with disk experts and
with host experts, with CUDA graphs and without: Q2_0, IQ2_XS (39.2 GB, 35.5 GB pinned), IQ3_XXS
(47.0 GB, 42.9 GB pinned), IQ3_S, and the Coder build's IQ1_M (29.6 GB, 256 experts, 25.1 GB pinned).
Q2_0 also ran with its experts on two GPUs before the table moved into the artifact. With disk
experts the process peaked at 1.05 to 1.19 GB of RAM for IQ2_XS, IQ3_XXS and IQ1_M, and at 1.10 GB
for IQ3_S (an L40S host).

The published layout, each model without its table and with its Vision tower plus the shared table
artifact, was checked on one NVIDIA L40S (an sm_86 build): the generate test with device, host and
disk experts for Q2_0, host and disk for IQ3_S and the Coder build, three sequences decoded as one
batch against each decoded alone, a restored snapshot, an image question per release, and the
server's concurrency, prefix reuse and JSON Schema checks.

Where a decode step goes, from an Nsight Systems trace of the generate test's graph-replayed steps
(Q2_0, every expert on that L40S): about 1,780 kernels and copies in 10.6 ms, 0.5 ms of it idle
between them. A token reads about 4 GB of weights, and the BF16 hyper-connection projections are
the largest share: 97 down/up pairs of 6.5 MB each, 1.27 GB, more than the ten routed experts of
every layer (0.66 GB). Their GEMVs take 2.2 ms, the routed experts 1.75 ms, the GGUF projections
of the Gated DeltaNet and attention layers with the head about 2.9 ms, and the router, shared
experts, activation quantization, recurrent and sparse-attention kernels the rest. With host
experts on a 24 GB card the expert kernels read what the cache lacks across the bus and take most
of the step instead.

llama.cpp runs the same GGUFs with the experts on the CPU (`--n-cpu-moe 48`). On the second
machine above (23 threads) llama-bench gives 29.6 tok/s decode (tg128) and 312 tok/s prefill
(pp512) for IQ3_S, and 12.7 and 368 tok/s for Q2_0; the first RTX 3090's machine (32 threads) gave
11.2 and 267 tok/s for Q2_0. Its speed follows the host CPU, and its Q2_0 CPU path is the slower one.

Perplexity agrees with llama.cpp: over the first 72 KB of the WikiText sample in
`eval/corpora/perplexity-1m`, 2,560-token windows advancing by 1,024 targets (llama.cpp's
`--ppl-stride 1024 -c 2048`, which widens the window to 2,560), the fourteen windows both evaluate
identically (14,336 targets) give, with a BF16 KV cache in both:

| Release | NInfer | llama.cpp | Window by window |
|---|---:|---:|---|
| Q2_0 | 2.6579 | 2.6502 | -0.014 to +0.016 nats |
| IQ3_S | 2.1317 | 2.1278 | -0.022 to +0.016 nats |

Q2_0 read 2.6558 on the RTX 3090 before wide MoE calls moved to ggml's matrix kernel and wide
hyper-connection reads to BF16 GEMMs. On an L40S the four combinations of the two changes give
2.6489 (both, today's kernels), 2.6494 (the matrix kernel alone), 2.6495 (neither) and 2.6555 (the
GEMMs alone): rounding-order differences within 0.25% that move with the kernel mix and the GPU,
not a loss from either change.

The two agree as closely deep into a long context. On the four PG-19 streams of `perplexity-1m`
joined (261,412 tokens), 65,536-token windows advancing by 32,768 targets (llama.cpp's
`--ppl-stride 32768 -c 49152`) score only the last 32,768 positions of each window, all of them
far past the 2,051 below which the sparse layers attend densely. The five windows both evaluate
identically (163,840 targets), on two RTX 5090s with a BF16 KV cache:

| Window | NInfer | llama.cpp | Difference |
|---|---:|---:|---:|
| 1 | 7.6956 | 7.6874 | +0.0011 nats |
| 2 | 8.9497 | 8.9476 | +0.0002 nats |
| 3 | 9.8699 | 9.8700 | 0.0000 nats |
| 4 | 9.0243 | 9.0153 | +0.0010 nats |
| 5 | 5.6261 | 5.6275 | -0.0002 nats |
| All five | 8.0834 | 8.0802 | +0.0004 nats |

NInfer is the CUDA 12.9 `120a` build that served the evaluation below, llama.cpp `a7fb71f` with
CUDA 12.9 and every layer on the GPUs (2026-10-06). llama.cpp's window values come from its
running perplexity, printed to four decimals, so they are good to about 0.0003 nats. Perplexity
runs prompt passes only; the decode steps that wrote the evaluation's answers are not part of it.

Accuracy on the two reasoning benchmarks of the GSQ-RCO card that EvalScope scores without a code
sandbox, for the Q2_0 release with its table, every expert on two RTX 5090s (`--devices 0,1`), a
BF16 KV cache and six requests at a time: thinking on, temperature 1.0, top-p 0.95, top-k 20 and
one sampled run as in the Qwen3.8-27B campaigns, but at most 106,000 output tokens, what six
sequences hold on the two cards beside the weights
([`eval/configs/qwen3_8_flash_next_reasoning.yaml`](../eval/configs/qwen3_8_flash_next_reasoning.yaml),
2026-10-05). The answers cut at that limit were then continued from where they stopped up to the
27B campaigns' budgets, 122,880 output tokens for AIME and 245,760 for GPQA:

| Benchmark | NInfer, 106,000 tokens | Cut there | NInfer, 27B budgets | Q2_0, model card | BF16, model card |
|---|---:|---:|---:|---:|---:|
| AIME 2025 | 93.33 (28/30) | 1 | 93.33 (28/30) | 96.67 | 100.00 |
| GPQA-Diamond | 84.34 (167/198) | 10 | 86.36 (171/198) | 89.39 | 91.92 |

The card does not state its sampling, output limit or number of runs. A cut answer has given no
answer and counts as wrong; of the answers that finished within 106,000 tokens, 28 of 29 (AIME)
and 167 of 188 (GPQA) are right. One run of GPQA-Diamond's 198 questions has a standard error of
about 2.6 points, so the 106,000-token score sits two standard errors below the card. Nine of the
ten cut GPQA answers were still reasoning coherently at the limit and one was repeating a codon of
its question's DNA sequence; the cut AIME answer was still calculating. Continued by the same server
binary with the same sampling (the request's rendered chat prompt and the reasoning so far, sent as
a raw prompt; EvalScope's own answer extraction scores the result), four of the ten GPQA answers
came out right and four wrong, and two ran out without an answer, one still reasoning at 245,760
tokens and the looping one at the server's 247,000-token context; the AIME answer reached 122,880
still calculating. With those budgets GPQA-Diamond sits 3.0 points below the card, about 1.2
standard errors.

The answers are long: AIME 2025's averaged 26,100 output tokens (median 13,700), GPQA-Diamond's
23,200 (median 9,100), and 27 GPQA answers ran past 65,536. The run took 7 h 44 min (AIME
1 h 12 min, GPQA 6 h 33 min): 5.37 million output tokens at 193 tokens per second across the six
requests. The server was a `120a` build of `3e842a91` with CUDA 12.9, from before `120a` builds
moved to CUDA 13.1; the kernels 12.9 was found to miscompile serve an NVFP4 KV cache, which this
run did not use, and the same binary's prompt passes match llama.cpp over the long windows above.
