# Launch recipes

The `ninfer-serve` commands behind the [reference tables](performance.md), one section per model.
The [quick start](../README.md#quick-start) serves one artifact; these sections add the flags that
fit each model and memory size. `ninfer-serve --help` is the exact source for option spelling and
defaults; [server options](serving.md#server-options) and the [CLI options](cli.md) describe every
flag, and [Supported GPUs](gpus.md) says what fits a given amount of GPU memory.

## Container or build

The recipes are written for a build. In the container, replace `ninfer-serve models/` with the
`docker run ... serve /models/` line of the quick start and keep the flags:

```bash
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 \
  -v "$PWD/models:/models" -v ninfer-cache:/cache \
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve /models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer \
  --model-id bonsai2-27b --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 \
  --gdn-state-fp16 --spec dflash2 --draft-tokens 5
```

A build listens on 127.0.0.1:8080 unless `--host` and `--port` say otherwise; the container listens
on 0.0.0.0:8080.

## Flags in these recipes

| Flag | What it sets |
|---|---|
| `--model-id` | the public model name in the API |
| `--max-context`, `--kv-capacity` | each sequence's context ceiling, and the shared KV capacity (`auto` takes what is free after the weights; omitted means `auto` with the default hybrid prefix cache and `--max-context` with the checkpoint catalog) |
| `--kv-dtype` | KV-cache storage: nine formats from `bf16` to `rk2v4-e8`; `rk8v4` is the all-round choice, `rk4v4` the longest windows ([context and memory](cli.md#context-and-memory)) |
| `--gdn-state-fp16` | keep the Gated DeltaNet recurrent state in FP16 |
| `--rope-yarn` | past the native window, YaRN instead of plain RoPE |
| `--spec`, `--draft-tokens` | the speculative backend (`mtp`, `dflash`, `dflash2`) and the drafts of a round (1 to 15; 1 to 7 for Gemma 4) |
| `--adaptive-mtp` | MTP only: pick each round's width from the drafts' measured survival |
| `--vision`, `--vision-residency`, `--vision-max-merged` | image and video input, where the tower lives (`overlay` borrows device memory per image), and the merged-token budget of one item |
| `--max-concurrency` | requests admitted at once, 1 to 8 |
| `--devices` | one pipeline stage per listed GPU (Linux) |
| `--expert-residency`, `--expert-cache-mib` | native Qwen MoE, Gemma 4 and Qwen3.8-Flash-Next: where the routed experts live and their persistent device cache budget ([expert residency](cli.md#common-options)) |
| `--ngram-table`, `--ngram-residency` | Qwen3.8-Flash-Next: where the n-gram rows live and the table's file |
| `--disk-kv-path`, `--disk-kv-gib`, `--disk-kv-restore` | the disk tier of the context cache |

## Ternary Bonsai 2 27B

The fastest single stream, MTP, the largest context a 24 GB card holds, and a disk tier.

```bash
# Fastest single stream: DFlash2 with five drafts over the full 262,144-token window.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# MTP drafting through the proposal head.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3

# The largest context a 24 GB card holds: 958,464 tokens of rk4v4.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 958464 --kv-capacity 958464 --kv-dtype rk4v4 --gdn-state-fp16 --rope-yarn

# Adaptive MTP and a disk tier that keeps evicted conversations.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 5 --adaptive-mtp \
  --disk-kv-path /var/cache/ninfer --disk-kv-gib 64 --disk-kv-restore
```

- **Draft count.** Five drafts are the all-round choice. Seven are faster on short answers; after
  long documents three to seven win ([draft length](performance/reference-2026-09.md#draft-length)).
- **Images.** `--vision --vision-residency overlay --vision-max-merged 12288` adds them. The encode
  borrows the drafter's memory, so the whole window still fits a 24 GB card.
- **Past 958,464 tokens.** The RTX 5090 and the PRO 6000 hold the 1,048,576-token maximum with
  `rk4v4`, DFlash2 or MTP included; the RTX 5090 holds 978,944 with `rk8v4`. Filled to 1,048,576
  tokens, the model misses the code at 90% (about 943K); up to about 880K it found every code on
  every card.

## Qwen3.8-27B

The official `groupwise-int` artifact and the GSQ-RCO IQ3_S release.

```bash
# The official artifact on a 24 GB card: DFlash2 with five drafts over 245,760 tokens of rk4v4.
ninfer-serve models/qwen3_8_27b.ninfer --model-id qwen3.8-27b \
  --max-context 245760 --kv-capacity 245760 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec dflash2 --draft-tokens 5

# GSQ-RCO IQ3_S: MTP over 176,128 tokens of rk8v4.
ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 176128 --kv-capacity 176128 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3
```

- **The official artifact with `rk8v4`.** The same speculation fits 167,936 tokens on an RTX 4090
  and 176,128 on an RTX 3090. An RTX 5090 takes the full 262,144 with either KV format.
- **IQ3_S.** It drafts with `--spec dflash2 --draft-tokens 5` as well, and `--vision` adds images.

## Qwen3.8-27B fine-tunes

Huihui abliterated, HauhauCS Aggressive and MXFP8-CRACK. All three have the official Qwen3.8-27B identity and size, so the Qwen3.8-27B recipes above serve
them too; CRACK has no DFlash2 adapter. Their cards' configurations:

```bash
# Huihui abliterated on one RTX 3090: 198,400 tokens with MTP and Vision.
ninfer-serve models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 198400 --kv-capacity 198400 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 \
  --vision --vision-residency overlay --vision-max-merged 12288

# HauhauCS Aggressive: DFlash2 with seven drafts.
ninfer-serve models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-DFlash2-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --max-context 32768 --kv-capacity auto \
  --spec dflash2 --draft-tokens 7

# MXFP8-CRACK: MTP.
ninfer-serve models/Qwen3.8-27B-MXFP8-CRACK-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 32768 --kv-capacity auto --spec mtp --draft-tokens 3
```

## Qwen3.8-Flash-Next

Every expert on the GPUs, in pinned host memory, or on disk.

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
  ghcr.io/iamwavecut/ninfer-all:cuda13.4 serve /models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-ninfer-v3.ninfer \
  --ngram-table /models/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer \
  --model-id qwen3.8-flash-next --max-context 32768 --expert-residency host
```

- **Other releases.** IQ3_S and the Coder IQ1_M build take the same flags with their own file and
  the same table. The [performance page](performance.md#qwen38-flash-next) shows the placements
  measured for each.
- **Memory.** Host experts pin 34 GB of RAM for Q2_0, 50 GB for IQ3_S and 25 GB for the Coder
  build. Disk experts need under 1 GB of RAM; the page cache does the rest. The GPU expert cache
  takes what is free after startup, or `--expert-cache-mib`.
- **The n-gram table.** Its rows are read from the file, 16 per token, unless
  `--ngram-residency ram` loads all 28.8 GB or `ram-hot` keeps the rows a profile ranks first in a
  RAM budget ([the n-gram rows](qwen3-8-flash-next.md#the-n-gram-rows)). A model started
  without its table is refused; `--no-ngram-table` overrides that, an experimental mode with no
  practical use.
- **Drafting.** `--spec mtp --draft-tokens 3` drafts with the MTP block the published artifacts
  carry.
- **Images and video.** `--vision` adds the Vision tower (0.9 GB on the GPU).

## Gemma 4 26B-A4B

Converted from Google's QAT GGUF release.

```bash
# The release's GGUF and its Hugging Face text files; one conversion writes the artifact.
mkdir -p models/gemma4/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
         processor_config.json; do
  curl -fsSL -o models/gemma4/hf/$f https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/$f
done
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf gemma-4-26B-it-mmproj.gguf \
  --local-dir models/gemma4
# With the release's QAT assistant (multi-token prediction) as the artifact's `mtp` component and
# its vision tower (the mmproj) as the `vision` component.
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF mtp-gemma-4-26B-A4B-it.gguf --local-dir models/gemma4
python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf --components text,mtp,vision \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf \
  --source mtp=models/gemma4/mtp-gemma-4-26B-A4B-it.gguf \
  --source vision=models/gemma4/gemma-4-26B-it-mmproj.gguf --name Gemma-4-26B-A4B-it-QAT-Q4_0-MTP \
  --out models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --device cpu

# Four concurrent requests over 40K-token contexts, two drafts a round, images and video; thinking
# stays off unless a request asks.
ninfer-serve models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4 --spec mtp --draft-tokens 2 --vision
```

## Qwen3.6-35B-A3B NVFP4

RTX 50 series and RTX PRO 6000 only.

```bash
ninfer-serve models/Qwen3.6-35B-A3B-NVFP4-ninfer-v3.ninfer --model-id qwen3.6-35b-a3b \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk8v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3 --vision
```

The NVFP4 experts prefill through W4A4 on Blackwell's FP4 tensor cores, so this needs a `120a`
build (the image has one); `sm_86` and `sm_89` builds refuse the file. On an RTX 5090 it starts in
17 s and leaves 7.4 GiB of the card free.

## Cards of 8 to 12 GB

The windows the [small cards](performance/small-cards-2026-10.md) page measured: the server started
with `--max-concurrency 1` and answered a request. These three cards carry built-in device profiles;
another card of this size calibrates once at its first start, in one to three minutes.

```bash
# 8 GB (RTX 3060 Ti): Ternary Bonsai 2 with 24,576 tokens of rk4v4 and no drafter. MTP does not fit.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 24576 --kv-capacity 24576 --kv-dtype rk4v4 --gdn-state-fp16

# 11 GB (RTX 2080 Ti): 143,360 tokens of rk4v4 with MTP; without --spec and --draft-tokens, 200,704.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 143360 --kv-capacity 143360 --kv-dtype rk4v4 --gdn-state-fp16 \
  --spec mtp --draft-tokens 3

# 12 GB (RTX 3060): 266,240 tokens of rk4v4.
ninfer-serve models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer --model-id bonsai2-27b \
  --max-context 266240 --kv-capacity 266240 --kv-dtype rk4v4 --gdn-state-fp16

# 12 GB: the GSQ-RCO IQ3_S release with 8,192 tokens and no drafter.
ninfer-serve models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer --model-id qwen3.8-27b \
  --max-context 8192 --kv-capacity 8192 --kv-dtype rk4v4 --gdn-state-fp16
```

- **Qwen3.8-Flash-Next** runs on 8 and 12 GB with `--expert-residency host` or `disk` and the
  recipe above (32,768 tokens of context, no other change). It keeps 2.83 GiB of dense weights on
  the GPU and fills the rest of the card with an expert cache.
- **Gemma 4** needs 16 GB: its 13.4 GiB of weights do not fit 12 GB.
- **Output heads.** `--lm-head-q4` and `--embedding-q4` do not apply to Ternary Bonsai 2, whose
  head is not stored as row-split Q8_G32.

## A card without a built-in profile

```bash
ninfer-calibrate --print > my-gpu.json
```

The engine measures it by itself at first start. Running it by hand refreshes the profile after a
driver or clock change. See [device profiles](device-profiles.md).
