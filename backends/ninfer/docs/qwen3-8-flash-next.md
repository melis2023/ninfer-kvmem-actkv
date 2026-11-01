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
the table at load; each token reads the 16 rows it addresses from the file, unless the table, or the
part of it a profile ranks most used, is loaded into RAM ([the n-gram rows](#the-n-gram-rows)). A
model stored without its rows takes them from `--ngram-table PATH`, which
must hold the table the model names; without a table the engine refuses to start
([running without it](#without-the-n-gram-table) is an experiment, not a mode).

The published conversions store the models without the table, which is published once:

| Artifact | Size | |
|---|---:|---|
| [n-gram table](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3) | 26.92 GiB | IQ4_NL rows and an optional broad hot-row profile |
| [Q2_0](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3) | 38.49 GiB | Vision and shared-Q8_0 MTP |
| [IQ3_S](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3) | 54.50 GiB | Vision and shared-Q8_0 MTP |
| [Coder IQ1_M](https://huggingface.co/WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3) | 31.02 GiB | 256 text experts per layer, Vision and shared-Q8_0 MTP |

The model runs with up to eight concurrent requests, a context cache of prompt prefixes, structured
output, images and video through its Vision tower (`--vision`, from an artifact converted with the
tower), and MTP speculative decoding (`--spec mtp`). The three published NInfer models above now
include Unsloth's shared-Q8_0 MTP block. The upstream GSQ-RCO GGUF sources omit it, so new
conversions must supply it separately; see [MTP](#mtp-speculative-decoding).

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

Every conversion containing PLE reads the table shard once more to hash it (a minute or so for 28.8 GB), since
the model records the digest of the table it reads whether or not it stores the rows.

`--layers A..B` makes a qualification artifact from source layers A through B-1, using zero-based,
half-open bounds. It supports the HF, GGUF and native-Q2 recipes. The embedding, final mixer and
head remain; selected blocks and their activation uses are renumbered from zero, while source
lookups keep the original indices. `provenance.source_layers` records those indices. PLE moves
with its block and keeps the same table constants. If the range contains no PLE, the artifact
omits the `ngram` component and no table source or runtime override is needed. The Engine refuses
unused table options for that slice.

```bash
# Source blocks 3 (QSA) and 4 (GDN), with no PLE or table scan.
python3.11 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gsq_q2 --layers 3..5 --components text \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --device cpu --rows-per-chunk 65536 --out models/flash-next-layers-3-5.ninfer
```

A sliced model runs its shortened stack through the public Engine. Its outputs are for binding,
state and numerical qualification; they do not represent full-model quality or logits at the
same layers of a complete forward, whose input residual also includes the preceding layers.

`qwen3_8_flash_next_gsq_q2` imports the trained Q2_0 routed experts into `q2_g64_fp16` planes
with exact code and signed-scale words. Q2_0/Q8_0 shared projections retain their codes/scales as
native Q2/Q8; other shared block formats decode and round to BF16. Other projections and the MTP
block keep their GGUF representations. Native banks support device residency and host residency at
concurrency one; native disk residency is not implemented. A8 uses grouped integer MMA from 16
columns, decoding the stored native codes in registers without repacking the weights.

The native recipe is an explicit artifact choice; the GGUF artifacts stay the published
representation. On the held-out quick corpus (391,000 targets, context 32,768, BF16 KV, one RTX
A6000) the native-Q2 artifact scored 5.004501 against 4.991555 for the GGUF one (+0.26%, worst
window +4.3%), and scoring took 15% longer. Its expert Op is within a few percent of the GGUF Op at
decode widths with A8 activations, between 28% faster and 34% slower at prompt widths, and slower
everywhere with A16 activations.

`qwen3_8_flash_next` supports HF-source conversion: text routed experts Q4 gate/up and
Q5 down, shared experts and GDN/QSA projections Q8, embedding Q8 and head Q6. HC, router,
indexer, PLE, vision and small GDN gates stay in their direct formats. MTP routed experts use
Q4 and its other projections BF16. No sensitive-layer Q6 mask is selected without quality
measurements. Both text recipes accept the IQ4_NL GGUF table or HF BF16 n-gram shards via
`--source ngram=PATH`. HF rows become `fp8_e4m3fn_row_fp16`: 160 E4M3FN bytes and an FP16
scale per row. Full-model quality and memory requirements of this recipe are unmeasured;
full HF-to-Q4/Q5 artifacts and their model campaign are outside the current delivery scope.
The official FP8 checkpoint's individual expert matrices are supported alongside BF16 fused
banks. E4M3FN codes are multiplied by their stored BF16 or FP32 `weight_scale_inv` values in
128×128 blocks, then rounded to FP32 before the selected artifact quantizer runs. Despite
its name, `weight_scale_inv` is the dequantization multiplier. Source overrides resolve the
selected checkpoint's layout, including MTP. Unscaled FP8 banks are refused.
An FP8 text checkpoint still needs a separate supported table source: IQ4_NL GGUF or HF BF16
shards. Import of the FP8 checkpoint's own n-gram encoding is not implemented.

```bash
python3.11 -m tools.convert --model /path/to/HF-BF16-checkpoint \
  --recipe qwen3_8_flash_next --components text,ngram,mtp \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --device cpu --out models/flash-next-q4-q5.ninfer

# Write just the FP8 table from an already downloaded HF checkpoint.
python3.11 -m tools.convert --model /path/to/HF-BF16-checkpoint \
  --recipe qwen3_8_flash_next --components ngram \
  --source ngram=/path/to/HF-BF16-checkpoint --device cpu \
  --out models/flash-next-ngram-fp8.ninfer

# Fetch only the HF MTP tensors into a new directory, with bounded 8 MiB weight reads.
python3.11 -m tools.reference.fetch_slice --repo Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --mtp --out /path/to/mtp-subset
```

The subset includes a config, safetensors index and SHA-256 manifest for files and individual
tensors. Pass `--source mtp=/path/to/mtp-subset` to the native or GGUF recipe to use it; the HF
MTP block then uses Q4 routed experts and BF16 elsewhere. A server that ignores byte ranges is
refused before a full shard is read. Failed copies retain a `.partial` file, never a published
safetensors file; use a fresh output directory after resolving a failed transfer.

An FP8 table (`fp8_e4m3fn_row_fp16`, 51.8 GB) from the HF BF16 shards is supported but not
published: it improved held-out perplexity by 1.25% (worse on both code domains) for 80% more
table storage than IQ4_NL, which stays the release representation.

`--components text,vision` (with or without `ngram`) adds the Vision tower from the release's
`mmproj-Qwen3.8-Flash-Next-BF16.gguf` (`--source vision=PATH`): the same tower as Qwen3.5/3.6
(27 blocks of width 1152, merging 2×2 patches onto the text model's 2,560), kept in BF16, 0.9 GB.
`--model` then also needs `preprocessor_config.json` and `video_preprocessor_config.json`.

`--components text,mtp` (with or without `ngram` and `vision`) adds the MTP block from one of
Unsloth's MTP GGUFs (`--source mtp=PATH`; [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
folder `MTP/`, a `shared-` file, which borrows the model's token embedding and head):

```bash
python3 -m tools.convert --model /path/to/Qwen3.8-Flash-Next \
  --recipe qwen3_8_flash_next_gguf --components text,ngram,mtp \
  --source gguf=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --source ngram=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --source mtp=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --device cpu --rows-per-chunk 65536 \
  --name qwen3.8-flash-next --out models/flash-next-q2_0-mtp.ninfer
```

The block keeps its matrices in the GGUF's blocks and its 512 experts, also beside the Coder
build's 256; its norms drop their stored `1 + w`, and its hyper-connection matrices keep the
`shared-Q8_0` file's Q8_0 blocks, which their kernels read. The `shared-Q8_0` block adds 2.8 GB to
the model.

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
([launch recipes](recipes.md#qwen38-flash-next)).

| Option | Meaning |
|---|---|
| `--expert-residency device\|host\|disk` | expert banks in the stage devices' memory (default); in page-locked host memory that the expert kernels read across the bus; or left in the artifact's files, each layer's routed experts read into a device cache before they run |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts: `auto` (default) takes what each device has free after startup less a margin, after the warm-up what is still free beyond 640 MiB, and with host experts the prefill buffers between prompts ([the expert cache](#the-expert-cache)); `N` is the cache's size, which the warm-up leaves alone; `0` disables the host-mode cache (disk mode needs one) |
| `--expert-misses staged\|mapped` | GGUF host experts: a decode or verification call copies each routed expert its device cache lacks into device memory by the copy engine while the cached ones run, into a cache slot when one is free or when the expert's routes beat the coldest cached expert the call does not route ([the expert cache](#the-expert-cache)), else into staging (`staged`, the default); or its expert kernels read the missing experts across the bus (`mapped`) |
| `--expert-dma-share F\|auto` | host or disk experts: fraction of a call's missing experts the GPU runs (copied to it), `0..1`; the CPU computes the rest from RAM while the GPU runs the cached ones. `auto` (the default) gives each call of GGUF experts the split whose copies and CPU products end soonest by the times measured on earlier calls, and copies every miss of native experts. Only calls of up to eight tokens use the CPU; GGUF CPU products are the GPU's bit for bit ([the CPU share](#the-cpu-share)) |
| `--expert-tail-skip N` | GGUF host experts with staged misses: a decode or verification call leaves out each routed expert its device cache lacks when every token of the call ranks it past place `N` (`1..9`) of its top ten, instead of copying it; the remaining experts keep their router weights (no renormalization). Lossy, off by default; prefill always runs every routed expert. On an RTX 3090 with Q2_0 and four short chat prompts, `7` decoded 9.0-17.6% and `6` 11.9-24.4% faster, copying 22-43% and 29-52% fewer expert bytes; perplexity over 5,115 tokens scored one token per call went from 4.667 to 4.710 with `7` and 4.688 with `6`, an order opposite to how much each leaves out, so that sample does not resolve the cost |
| `--expert-prefill-buffers 1\|2` | GGUF host experts: device buffers a wide prefill call copies its layer's uncached experts into; `2` alternates two, so the next layer's copies run while this layer's experts do, when half the free memory holds both. Default: `2` with the automatic cache, where both buffers hold cache slots between prompts, else `1`; with an explicit `--expert-cache-mib` the buffers take memory beside the cache. On an RTX 4090 with IQ3_S host experts the second buffer prefilled a 31,865-token prompt at 3,430 instead of 2,683 tokens/s and decoded within 2.4% of the one-buffer speed (one run each); on an RTX 3090 with Q2_0, before the buffers became cache slots, 8.5-10.5% faster at 5,462 tokens and 1.9-2.3% at 31,860 |
| `--expert-cpu-threads N` | host or disk experts with a CPU share: `1..256` CPU threads, the calling one included; default is one per physical core the process may use (its affinity mask), at most half its CPUs (the mask and a container CPU quota); the rest stay with the threads that drive the GPU and serve requests ([the CPU share on one RTX 4090](#the-cpu-share-on-one-rtx-4090)). Each thread is pinned to a physical core and spins while requests decode |
| `--expert-cache-adaptive` | replace cold cached native experts between calls; off by default because it changes the CPU/GPU arithmetic partition |
| `--expert-profile PATH` | fill the expert cache at startup from counts recorded for the same artifact (`--expert-profile-out`); default: the built-in profile of the model family and its bank widths, where the build has one |
| `--expert-profile-out PATH` | record every layer's expert routes after requests, for a later `--expert-profile` |
| `--ngram-table PATH` | the table artifact to read the n-gram rows from; required for a model stored without its table, and it must hold the table the model names (same SHA-256 and row format) |
| `--ngram-residency disk\|ram\|ram-hot` | where the n-gram rows come from: the table's file, 16 rows a token (default); the whole 28.8 GB table in RAM; or the rows a hot-row profile ranks first in RAM and the rest from the file ([the n-gram rows](#the-n-gram-rows)) |
| `--ngram-io buffered\|direct\|mmap` | how rows are read from the file: positioned reads through the OS page cache (default), reads past it (`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`), or copies out of a mapping |
| `--ngram-io-depth N` | row reads in flight (1 to 1024, default 64) |
| `--ngram-hot-profile PATH` | `ram-hot`: override the selected table's embedded profile with a `ninfer-ngram-profile` file |
| `--ngram-ram-mib N` | `disk`: cached rows and their index (default `0`, off); `ram-hot`: resident profile rows and their index (default 4096 MiB) |
| `--ngram-lock` | `ram`, `ram-hot`: lock the resident rows in physical memory (`mlock`, `VirtualLock`; needs the memlock limit or the privilege) |
| `--no-ngram-table` | run without the n-gram table: see [below](#without-the-n-gram-table) |
| `--kv-resident N\|all\|auto` | positions of each sparse-attention layer's KV a sequence keeps in device memory; a longer `--max-context` streams those layers ([streamed KV](#streamed-sparse-attention-kv)). `auto` (default): 32,768 with host or disk experts, all with device experts |
| `--rope-yarn`, `--rope-yarn-factor F` | YaRN past the native 262,144 positions, as for the Qwen3.5 family: at `--max-context` / 262,144 once the window passes it, or at a fixed factor `F` (`1..4`) for every position. It reaches every rotation the model makes, the sparse-attention layers', their indexer's and the MTP block's, and changes every answer, short ones too. Without it positions past 262,144 rotate unscaled. `--rope-scaling-factor` is not available for this model. The startup log names the factor, and the request log's `server_start` record reports the resolved rotation (`engine.rope`) |
| `--devices A,B,...` | one pipeline stage per GPU; layers are split so that every stage holds about the same stored bytes (`--stage-layers` overrides) |

With device experts every expert runs on the GPU. With GGUF host and disk experts, by default the
CPU computes a share of each decode or verification call's missing experts ([the CPU
share](#the-cpu-share)), MTP drafts three tokens a round when the artifact carries the block
([MTP](#mtp-speculative-decoding)), and the sparse-attention KV past 32,768 positions streams from
host memory ([streamed KV](#streamed-sparse-attention-kv)), which leaves the device memory to the
expert cache. Native-format experiments are retained for reference. The existing GGUF artifacts
support both host and disk residency; disk residency uses a CLOCK device cache. Disk residency
relies on the OS page cache for repeated file reads; the engine keeps no RAM cache of experts of its
own, only the device cache and bounded transfer buffers.

### The expert cache

With host experts the device cache's slots follow the expert tier that every MoE family with host
or disk experts shares (`core/expert_tier.h`), one pool of slots per layer. Each decode or
verification call adds one to the count of every (token, expert) pair it routes; every four decode
passes a round swaps up to 96 of the most routed experts the cache lacks into the slots of the
least routed ones it holds, best gain first, each only when its count reaches the victim's plus 1.5
(and at least 2), then multiplies every count by 0.7. Each pass until the next round starts up to
24 of those swaps (a quarter of the limit), best gain first: their victims leave the tables at
once, and their candidates' copies run on a side stream beside the draft, ahead of the next
verification, whose missing experts would otherwise queue on the bus behind a whole round. The
candidates' table entries move to their slots once the copies have landed. A miss the GPU copies
anyway takes a slot at once when one is free, or under the same rule against the coldest cached
expert its call does not route; a miss the CPU computes moves no expert bytes and enters only
through a round. Prompt chunks feed the statistics only. A routing profile (`--expert-profile`,
else the built-in one of the model family) fills the slots at startup and, ranking its counts over
every layer, gives each text layer the slots its most routed experts take; the MTP block keeps an
even share. Strata (MIT) measured this tier and its constants on Qwen3.8-Flash-Next.

With the automatic cache and host experts, the device buffers that prompt chunks copy a layer's
uncached experts into (`--expert-prefill-buffers`) are cache slots between prompts: every layer
gets the same share of them, free until misses and rounds fill them. A prompt pass takes them
back: the most routed of the experts they hold move into the slots of colder ones in the cache's
own memory, the rest leave, and the copies into the buffers wait for the moves. The next decode
or verification pass returns them zeroed and refills them at once with the most routed experts the
cache lacks (those the tier counts at least twice), copied on the side stream as a round's are and
serving once a layer's copies have landed, so the decode after a prompt does not wait for many
rounds to fill them again; buffers' slots left over stay free. On an RTX 4090 with the IQ3_S
release's host experts the 256 tokens after a 31,865-token document decoded 1.052 times as fast as
without the refill (geomean over the nine `--kv-dtype` formats, 0.992-1.118); after short prompts
the ratio stayed within the run-to-run spread (code 0.993, prose 1.011).

With host experts the GPU holds only the dense weights (3.7 GB for the Q2_0 release), the
per-request state and the expert cache; the host needs the expert banks in page-locked memory
(34 GB for Q2_0). With disk experts the host needs no copy of the banks at all: the expert cache
reads the missing experts from the artifact's files through the OS page cache, eight reads in
flight, into a 256 MB page-locked staging ring, so the page cache keeps whatever the system can
spare and the rest comes from the disk. A read of the artifact's files that fails while the Engine
runs (an expert or an n-gram row) fails the requests of that round with `unavailable`; the expert
cache forgets what it held, the Engine counts an `engine_recoveries` and serves the queue on.

Native host banks copy every miss to the GPU by default (`--expert-dma-share 1`). Values below one
let the CPU compute a share of the missing experts; this mode is experimental: on one RTX 3090 with
an 8-core EPYC 7302P it decoded 3.1 tok/s at a share of 0 and 5.3 at 0.5 against 31.4 at 1, and its
logits stay within top-1 but are not numerically qualified (maximum relative L2 0.32-0.60 against
resident experts). A GGUF bank, including a GGUF MTP companion beside native text weights, keeps
its GPU route. Linux registers native weight objects separately; Windows uses pageable weights and
bounded pinned staging. Use `--max-concurrency 1` for native host execution.

### The CPU share

With GGUF host or disk experts, CPU threads compute part of a call's missing experts from their
stored blocks while the GPU copies and runs the rest (`--expert-dma-share auto`, the default, or a
fixed share below `1`); the GPU then adds the CPU's products into the same fixed-point sum as its
own. Only calls of up to eight tokens use the CPU: decode, verification, and prompt steps that
short, such as the tail after a chat prompt's reusable anchor; longer prompt chunks always run on
the GPU. A fixed share gives the CPU the experts with the fewest pairs. `auto` gives each call the
split whose copies and CPU products end soonest by two straight lines fitted to earlier calls' copy
and CPU times (a sample counts at most four times or a quarter of its line's value); a side left
unused for eight calls takes one expert, so both lines follow the machine. An expert the CPU
computed takes no slot and crosses no bus; the expert cache's rounds decide whether it enters the
cache ([the expert cache](#the-expert-cache)). The CPU threads are pinned one to a physical core
and spin while requests decode, so a call's products start without a wake-up.

The CPU computes the GPU's decode products bit for bit, so outputs, log probabilities and context
images do not depend on the share, on `auto`'s choices or on the thread count. The activations are
quantized as the GPU quantizes them (q8_1: per 32 values `d = amax / 127`, codes `round(v / d)`,
`d` rounded to binary16). A 32-value slice gives the exact integer dot of its codes and the
activation's, and its product is `(factor * d) * dot`, each step rounded, with the slice's exact
factor (the block's binary16 multiplier times its integer scale); IQ2_S's is
`d * fma(factor0, dot0, factor1 * dot1)` over its two halves, Q6_K's
`d * (factor0 * dot0 + factor1 * dot1)`, and Q4_K's and Q5_K's `(factor * d) * dot` less the
block's minimum times the slice's activation sum (q8_1's `s`). Slice `s` joins lane `s mod 32` in
slice order, and the 32 lanes meet as in a warp's butterfly (16, 8, 4, 2, 1 apart). SiLU (and
Gemma 4's GELU) is `ops/common/silu.h` on both sides, built from correctly rounded operations only,
and the middle is rounded to BF16 before it is quantized for the down product. No compiler fuses
these steps on either side: the GPU kernels spell them with `_rn` intrinsics and the CPU sources
build without floating-point contraction. Q2_0, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ2_S,
IQ3_XXS, IQ3_S and IQ4_XS have CPU products, in AVX2 (with FMA and F16C) where the CPU has it and
in portable C++ elsewhere, with the same bits.

### The n-gram rows

A pass reads the 16 rows each of its tokens addresses (1,440 bytes a token in IQ4_NL) as soon as it
has hashed them, `--ngram-io-depth` reads at once, and uploads them just before the PLE layer
(block 1): the GPU embeds the tokens and runs block 0 while the rows are read, and waits only if
the read takes longer. A prompt span reads the rows of all its chunks in one batch the same way,
while every chunk passes block 0. `--ngram-io` chooses how the file is read. `buffered` (the default) goes
through the OS page cache, which then keeps a 4 KiB page for each row it read; `direct` bypasses the
OS page cache, so every row-cache miss reads from the drive; `mmap` copies the rows out
of a read-only mapping of the file.

`disk` keeps recently read rows in a four-way CLOCK cache. `--ngram-ram-mib` bounds its
allocation including row tags and CLOCK state, independent of the table's size. The cache is off
by default; for example, `--ngram-ram-mib 4096` enables a 4 GiB budget. Its empty index and row storage use demand-paged
memory; the Linux cache avoids huge pages so one short row does not fault 2 MiB. The budget is
allocated capacity, not current RSS; buffered and
mapped reads can also occupy the OS page cache. Failed batches admit no rows.

Once a prompt step (a chunk, or a span) has read its own rows, it hints the rows of the
Engine's next step, which are then read while the GPU runs this one. MTP hints its anchor
and both highest-logit candidates at each draft step. An auxiliary CUDA stream copies each
step's two tokens and hashes private contexts while later draft steps run. Verification also
hints both candidates after each actual verified prefix; predictions do not change the next
column's context. Linux buffered/mapped readers
issue `POSIX_FADV_WILLNEED`; Windows mapped readers use `PrefetchVirtualMemory`. These are advisory
reads, with duplicate rows and shared pages coalesced. A worker thread issues a batch of more than
256 page ranges and drops a batch that arrives while it still issues the previous one. Direct I/O
and Windows buffered I/O start one bounded speculative batch in separate staging; busy workers
skip extra hints. Only complete successful
batches can supply demand reads, and hints alone admit no rows into the persistent cache.

Buffered/direct row misses use a bounded OS queue: io_uring on Linux and overlapped reads with
IOCP on Windows. One worker drives the queue while device execution continues; direct reads use
at most `--ngram-io-depth` aligned bounce buffers. A failed batch drains its outstanding reads
before returning and admits no rows to the cache. If Linux disables io_uring, the reader uses
positioned-read workers; a buffered batch of at most 1,024 rows then first reads, on the engine
thread, the rows the page cache holds (`RWF_NOWAIT`, or `mincore` over a mapping of the file where
the file system refuses that flag, as overlayfs does), and wakes only as many workers as rows
remain. On an RTX 4090 host in a container without io_uring, the 64 rows of a decode verification
took 3.6 ms per round across 64 woken workers; read on the engine thread, the GPU's idle time
before a verification fell from 2.4 ms to 0.8 ms a round.
Bulk RAM/hot-set loading and mapped reads retain their existing paths.
The Windows reader paths have not been run.

On a serving workload with one RTX 3090, the table's pages evicted before the first request, the
PLE layer waited 0 to 4 microseconds per request for its rows with buffered or direct reads. The
row cache stays off by default: with a 4 GiB cache the first requests took 10-11% longer, and
repeated requests that hit every row generated no faster.

`--ngram-residency ram` reads the whole table into RAM at startup (2 MiB pages where Linux offers
them), and `ram-hot` reads only the rows a hot-row profile ranks most used, as many as
`--ngram-ram-mib` holds beside their index (a bit per table row, 40 MiB), and reads the rest from
the file as `disk` does. A row depends on its token and the two before it and nothing else, so a
profile is counted from text alone, without running the model:

The selected table artifact may carry a profile in `ngram.resources.hot_profile`.
`ram-hot` uses it when `--ngram-hot-profile` is absent; an explicit file overrides it.
An external table supplies its own profile, and a table with no embedded profile still needs
the explicit file. Other residencies leave the resource unread. Conversion embeds a checkpoint's
`ngram.hot` file when present, or the file selected by `--resource ngram.hot=PATH`; the hash,
row bounds and uniqueness must match. This does not change the table digest or its row bytes.

Chat JSONL documents use the artifact's Frontend, including tool definitions, tool calls and the
assistant generation prompt. Set `"enable_thinking": false` or `true` per document when the corpus
uses a fixed thinking mode; without it the chat template chooses its default. Invalid messages
are rejected rather than omitted from the profile.

```bash
# Count the rows a corpus addresses: UTF-8 files, or .jsonl with a "text" or chat "messages" per line.
./build/apps/ninfer-ngram-profile models/flash-next-q2_0.ninfer --out models/flash-next.hot \
  corpus/*.jsonl
# The share of a held-out corpus's row reads the profile's leading rows serve, per RAM budget.
./build/apps/ninfer-ngram-profile models/flash-next-q2_0.ninfer --evaluate models/flash-next.hot \
  heldout/*.jsonl
./build/apps/ninfer-serve models/flash-next-q2_0.ninfer --ngram-table models/flash-next-ngram-table.ninfer \
  --ngram-residency ram-hot --ngram-hot-profile models/flash-next.hot --ngram-ram-mib 4096
```

`--ngram-lock` keeps the resident rows in physical memory. `/stats` reports the rows the passes
read, how many RAM served, the read latency and the time the PLE layer waited
([Stats](serving.md#structured-request-log)), and `ninfer` prints them after its answer.

The published table carries a broad hot-row profile: 26,223,284 ranked rows counted from
3,711,509 tokens sampled with equal per-group ceilings from four Common Corpus shards (prose, web,
press, science) and one github-code-clean shard. On 1,103,918 held-out tokens of the same sources
the leading rows serve this share of row reads per RAM budget:

| RAM budget, MiB | 256 | 512 | 1,024 | 2,048 | 4,096 and more |
|---|---:|---:|---:|---:|---:|
| Held-out row hits | 33.8% | 38.5% | 42.9% | 48.5% | 49.6% |

Across the 239 language and domain groups with at least 2,048 held-out tokens the hit rate at 4 GiB
runs from 20.5% to 95.5% (median 49.4%); Japanese books have the lowest coverage. Hits measure row
reuse, not speed or quality.

### Without the n-gram table

The model was trained with its n-gram embedding, so the engine refuses a model whose table it
cannot find. `--no-ngram-table` starts it anyway, without the PLE injection (exactly what an
all-zero table gives), and warns at startup: this is a non-standard, experimental mode with no
practical use. On the Q2_0 release it nearly doubles WikiText-2 perplexity, 2.66 to 5.01 over the
fourteen windows of the comparison below; short factual answers survive, but nothing measured
improves.

### Streamed sparse-attention KV

Each query of a sparse-attention layer reads 512 blocks of four positions that the indexer selects
and its own incomplete block, so most of a long context's KV sits unread at each step. With
`--kv-resident N` below `--max-context` (by default 32,768 positions with host or disk experts) a
sequence keeps each of the 12 layers' whole KV in pinned host memory, addressed by the GPU, and the
device holds a pool of N / 4 blocks per layer (at least 513 for each token of a verification
window). Before each decode or verification call's attention, a kernel looks up every block the
queries read, claims the missing ones, gives each a pool slot by a CLOCK sweep (a slot the call
uses is never taken), copies them across the bus from the host copy and re-points the table; the
blocks the call wrote are copied again when it reads them, so no slot holds bytes older than the
host copy. A prompt chunk reads its layer's whole history from a device staging copy, refreshed
from the host copy once per span of chunks. The attention reads the same bytes either way: a
streamed and a resident sequence give the same logits. The device memory saved goes to the expert
cache. The residency scheme follows Strata's `kv_stream` (MIT).

### Sequences, KV and the context cache

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
in arrival order. Prompts prefill a step at a time, each step from the prompt with the fewest
tokens left, so a short or cached request is not held behind a long prompt for its whole prefill
(a prompt passed over eight times goes next, and a prompt with media, once begun, finishes
first). A step is one chunk, or with host experts on one GPU a span of up to eight chunks of a text
prompt (see [Execution](#execution)); while other requests decode, a span is two chunks at most,
since they wait for the whole step. Before a step, while requests are decoding, they run
`--decode-rounds-per-prefill` rounds (by default the chunk size over 64, 16 at the default chunk)
for each chunk's worth of prompt the step computes, at least one, so a short prompt gets in after
one round and the streams keep their share during a long one; each round is one batched pass
whose experts read their weights once for the whole batch, so the batch costs little more than one
token while the experts dominate the step.

With the context cache on (the default), a sequence keeps its state when its request ends, and a
snapshot of its recurrent state where the prompt's last user turn closes (or at the prompt's end
when the template marks no turn), 74 MiB on the device: a later prompt that continues what the
sequence holds resumes from its live state, and one that repeats the prompt up to that point (the
next turn of a chat, which renders the previous answer without its reasoning) resumes from the
snapshot.
Either way only the new tokens are prefilled; the response's prompt summary reports how many were
reused. A request goes to the free sequence that holds the longest such prefix of its prompt.
`--no-prefix-reuse` (ninfer-serve) prefills every prompt from scratch.

What a sequence gives up for a new request (its turn-closure snapshot and the state it ended in,
each of 128 tokens or more, the end of a disposable request such as a decision read excepted) is
kept as an image in host memory, and so is the state where the frontend marks a prefix other
requests share (the end of the leading system messages, of the tool declarations, a declared
cache boundary, 128 tokens or more): a prompt step ends there, and requests with the same prefix
resume after it on any sequence, also when they arrive together, since a prompt that has not
started prefilling looks for images again at its first step. An image is the paged KV and pooled keys
of its positions, its recurrent state, and the MTP block's, about 114 MB plus 26 KiB a position in
BF16 KV. The images live in 8 MiB pages of pinned memory that stay pinned for later images: the
pages of two images of a whole context (within a quarter of the budget) are pinned at startup, and
the store pins more only when it holds more. Pinning while a request waits is slow where the
experts and the page cache fill the host's memory: on an RTX 4090 host with IQ3_S host experts,
pinning the two images of a 32K-token prompt held a short prompt after it for 2.3 s, and faulting
in pageable memory for them 1.8 s; into the pages pinned at startup they copy in 15 ms, and the
short prompt's first token came 0.45 s after it arrived (2.9 s before). Pinning the startup
pages (2 GiB with the default 8 GiB budget) adds about 2 s to startup. A prompt that continues a kept prefix further than any free
sequence resumes from its image, copied back into a sequence. `--host-kv-mib` (or `--host-cache-mib`) is the budget, 8 GiB by
default; past it the least recently used images go to the disk tier with `--disk-kv-path DIR`
(`--disk-kv-gib`, 64 by default), one file each under a directory per artifact, KV format, MTP
draft width and n-gram enablement, or are dropped without one. A different verify width or disabling
the n-gram table can change sequence state, so those runs cannot restore another profile's images.
A request that does not draft (a decision read) runs its prompt without the MTP block, so its
sequence's snapshots and images carry no MTP state and serve only requests that do not draft.
With `--disk-kv-restore` a
prompt also resumes from an image on
disk, which a later run finds there too.

Structured output (`--structured-output` for the server, `--json`/`--json-schema` for the CLI) works
as for Gemma 4 ([Structured output](gemma4.md#structured-output)): the grammar's token mask applies
to every sampled token, it is computed on a thread of its own while the step runs, a row whose mask
is late samples once it is ready without holding the others during ordinary decoding. Draft
eligibility does not depend on mask duration; a speculative round waits for all required masks
before acceptance. So do
[token log probabilities](serving.md#token-log-probabilities): a request that asks gathers each
sampled token's top 20 from the same logits before sampling.

`--vision` loads the Vision tower of an artifact converted with it (0.9 GB of BF16 weights on the
first device, beside the token embedding) and takes images and video as the Qwen3.5 family does:
the frontend renders the media tokens, the tower encodes the prompt's media before its first
chunk, and their merged embeddings replace those tokens' embeddings. A media prompt rotates its
positions on the three RoPE axes the frontend computes (text positions on all three, then each
later token at its index plus the prompt's offset); it prefills in a pass of its own and is not
kept for reuse by the context cache. Flash-Next media requests use plain decoding even when
MTP is enabled. A fresh text-only request can use MTP again after the media request finishes.

### MTP speculative decoding

`--spec mtp --draft-tokens N` (1 to 15) decodes with the MTP block of an artifact converted with it
([Convert](#convert)); without the block the engine refuses to start. Without `--spec`, an artifact
with the block and host or disk experts drafts up to three tokens a round with `--draft-min-p 0.5`
(a verification window's routes share the layer's missing experts, which the CPU computes while the
cached ones run); `--spec none` decodes without drafts. Text-only decoding requests then
runs rounds: the MTP block (one more sparse-attention layer with its own 512-expert MoE, its own
final mixer and the model's head) drafts N tokens from the stack the model left at the request's
last position (the head's argmax at each step), the model verifies the last sampled token and the N
drafts in one pass, and the acceptance keeps the drafts while they equal the model's own draw at
each position, then commits the model's draw at the first difference (or after the last draft).
The acceptance samples as the request does (greedy, temperature and top-k/top-p/min-p, presence
and frequency penalties counting the drafts kept before each position, the grammar's masks,
logprobs from each position's own distribution), drawing each position with the request's seed and
that position as the key, the key plain decoding uses there. Every committed token is thus the
model's own draw whatever was drafted. Verification width and batch composition can still change
floating-point reductions, so neither greedy nor sampled answers are promised to match plain
decode byte for byte. The kernels are qualified against independent mathematical oracles; exact
repetition holds the execution configuration fixed. Acceptance is bounded by the remaining output
and thinking-token budgets, including the correction or bonus token.

`--draft-min-p P` optionally ends a round's drafting at the first draft whose absolute
probability over the public vocabulary is at or below P. The round verifies the drafts up to and
including that one, so every round verifies at least one; P=0 disables the calculation and P=1
drafts and verifies one. A batch drafts until each of its requests has reached such a draft (or the
last step), verifies its largest selected prefix, and bounds acceptance separately for each
request. This is distinct from the sampling `--min-p` filter. With a floor each draft step runs
from its own CUDA Graph and the host reads the step's probabilities before it starts the next, so
`drafted_tokens` counts the steps that ran; each verification width has its own CUDA Graph. The
floor is part of the disk context-cache profile. `--spec mtp` keeps the full window unless
`--draft-min-p` is set; the automatic drafting uses 0.5. On an RTX 4090 with the IQ3_S release's
host experts (EPYC 7V13, 32-CPU quota, 64K context; 256 greedy tokens after a code prompt, a prose
prompt and a 31,865-token document, once per `--kv-dtype` format) three drafts with the floor 0.5
decoded 1.040 times as fast as without it (geomean of 27 pairs, 0.953-1.131, slower in 5): a round
drafted 2.48 tokens instead of 3 and committed 2.27 instead of 2.31.

The verification leaves the sequence's state where it was: each Gated DeltaNet layer records its
transitions and the commit replays the kept ones into the state (the Qwen3.5 family's ReplaySSM
fold), the sparse-attention indexer's tail and the PLE convolution history advance over the kept
positions from what the verification recorded, and the n-gram context is hashed again over them.
The MTP block follows every token the model takes in (prompt chunks too), with a KV cache of its own
the size of one sparse-attention layer's (2 KiB a position in BF16), so a request can draft as soon
as its prompt is in, a prefix the context cache restores included. Up to eight requests run their
rounds together, one MTP pass for all of them per draft and one verification pass. A request
decodes without drafts when its prompt has media, near the end of its context, with one token
left in its output or thinking budget, and while a
`--post-thinking` request still reasons. N-gram copy proposals (`--ngram-draft-tokens`) are not
available for this model; `--mtp-attention-window` and `--lm-head-draft` are refused.

With `--lookup-ngram N`, a request whose last `N` tokens (the token it feeds next last) appeared
earlier in its sequence proposes what followed them then, up to the round's width (`--draft-tokens`,
or with `--adaptive-mtp` the width the controller chose), in place of the MTP block's drafts; the MTP block drafts the other requests of the round, and none when every
request has such a proposal. The proposal is verified like any draft, so a wrong one costs speed,
not tokens. Its rounds are reported with the n-gram proposals (`ngram_rounds`).

With `--adaptive-mtp`, `--draft-tokens K` is the most drafts a round makes. The controller the
Qwen3.5 models use (see [Adaptive MTP](serving.md#adaptive-mtp)) picks each round's width from
the requests' measured draft survival and the round times it has measured at each width, but here
it goes down to one draft: the MTP block runs one step per draft, so a narrower round also drafts
less, where the Qwen3.5 models' drafts are ready before the round. Each step count has its own
draft-chain graph and each width its own verification graph, captured at their second use.

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
  tensor-core matrix kernel over the routed pairs from device memory, one CTA per row tile and per
  column tile that holds pairs (the sort that groups the pairs lists those tiles for each tile
  width), rather than one per expert and possible column tile (on an RTX 3090 with IQ3_S host
  experts, a 31,865-token prompt prefilled at 1.71k instead of 1.61k tokens/s). That kernel reads
  a 640-value down row in 256-value steps, so it decodes the bytes after a down matrix as the
  down's blocks: in a bank they are the next expert's down or zeros after the last one, and every cache or stream
  slot keeps zeros after its down, which a smaller down from another layer does not uncover (another
  format's bytes there can hold a non-finite scale, and NaN follows). With host experts, a wide call
  first copies the routed experts the cache does not hold into a device pool on each GPU (0.7 GB for
  Q2_0), a bank per projection laid out as the host's, so experts adjacent on the host cross the bus
  in one copy and each expert crosses it once per chunk; zeros follow each run of copied downs. On
  one GPU a prompt longer than a chunk runs in spans of up to eight chunks layer by layer: each layer
  runs its attention over every chunk of the span, then its experts over every chunk, so the layer's
  uncached experts cross the bus once per span, during the attention phase that precedes them, and
  a span's short last chunk runs from the same pool (a 5,669-token prompt prefilled at 1,334 instead
  of 678 tokens/s on an RTX 3090 when spans came; the bank-shaped pool and the two phases then took a
  9,297-token Q2_0 prompt from 1.12-1.18k to 1.35k tokens/s from a cold expert cache). The expert
  phase runs up to 4,096 tokens of consecutive chunks in one call, each chunk routed on its own,
  so the matrix kernel takes 128-column tiles rather than the 32 of a 1,024-token chunk; a last
  chunk of eight tokens or fewer runs alone (with IQ3_S, a 31,860-token prompt's expert matrix
  products took 4.47 instead of 6.52 s on an RTX 3090). With two prefill buffers (the
  default with the automatic expert cache, `--expert-prefill-buffers`) the pool has two buffers
  that consecutive layers alternate between: each layer's copies start as soon as the layer two
  back is done with its buffer, while the layer before it runs. A thread of its own issues each
  layer's copies, hundreds of operations that would otherwise hold the launching thread while the
  copy queue is full, so the layers in between keep being launched. Weighted
  expert outputs are summed in fixed point, so the result does not depend on the order experts
  finish in.
- Up to eight tokens of GGUF host experts run in two stages. A kernel splits the call's routes into
  the cached experts, which run at once, and the missing ones, which it posts to a host service
  through mapped memory, without a host synchronization; the service copies each missing expert
  into the slot of the layer's least recently used expert of the lowest frequency tier (or the CPU
  computes its share), the
  layer waits for it on the device, and the second stage runs it. The decode form of the expert
  kernels needs no sort or memset: one small kernel lists each slot's pairs, quantizes the tokens'
  inputs once and zeroes the sum; a block of the fused gate/up kernel takes 32 rows of one
  expert and quantizes its piece of the middle for the down kernel itself; one token's down runs
  row by row over all its experts, summing a row in registers. The decoding of Q2_0, the routed
  experts' format, is integer-throughput bound, so its activations come transposed and its codes
  need two operations a word. The shared expert runs on a second stream meanwhile. All of it is
  the separate kernels' arithmetic bit for bit.
- The expert cache counts the routes each forward pass took (decayed per token) and, between
  passes, copies the experts it needed most into its slots and points the tables at them: for
  every layer with mapped misses, and after a prompt chunk with staged ones. With disk experts the
  cache works per layer instead: once a layer has routed its tokens, the experts it lacks are read
  into the slots least recently used (never one the same call needs), and only then do the
  layer's experts run.
- A decode step (one token) replays a CUDA graph per pipeline stage, captured at a sequence's
  second decode step; the token's position reaches the sparse-attention kernels in device memory.
  Disk experts need the host between a layer's routing and its experts, so their steps stay eager.
  `--no-cuda-graph` decodes eagerly everywhere.
- With MTP, a verification of one request (at most eight tokens, so its experts take the vector
  products) replays graphs of its own the same way, and so does the draft chain of one request on
  one GPU, host experts included; several requests' rounds, the MTP catch-up and disk experts run
  eagerly. With staged host misses the MTP block serves its missing experts in draft steps and a
  round's cells (at most eight tokens) as a decode layer does, copied or computed by the CPU, so
  the drafts follow the block's own routing; over longer prompt chunks it runs only its cached
  experts, their weights renormalized per token, since it only steers drafts, which verification
  checks. On an RTX 4090 with the IQ3_S release's host experts (EPYC 7742, 64K context, three
  drafts with `--draft-min-p 0.5`, 256 greedy tokens after a code prompt, a prose
  prompt and a 31,865-token document, once per `--kv-dtype` format) serving those misses instead
  of dropping them committed 5-21% more tokens a round and decoded 1.025 times as fast (geomean of
  27 pairs, 0.959-1.092; slower in 4, the two largest losses after the document).
- A pass whose layers may wait on the device for their missing experts ends synchronized: CUDA
  loads a kernel lazily at its first launch, a load may wait for the device to go idle, and a
  launch during such a wait would deadlock with the service it waits for.

## Measurements

### Expert placements

Greedy decoding, one request. Decode is measured over a short answer (78 tokens for Q2_0) and over
the first tokens after a 4,463-token prompt; prefill is that prompt. Host memory is the process's
peak resident set (the pinned bank for host experts); device memory the most `nvidia-smi` showed in
use.

The Q2_0 release on the RTX 30 series, October 5, CUDA 12.8, the table inside the artifact, 512-token
prefill chunks:

| Hardware and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| 2× RTX 3090 Ti (PCIe, no P2P), experts on the GPUs (`--devices 0,1`) | 18.4 + 19.4 GB | 0.7 GB | 90.2 tok/s | 107 tok/s | 1,504 tok/s |
| RTX 3090, experts in pinned host memory, 17.7 GB expert cache | 22.6 GB | 34 GB pinned | 48.9 tok/s | 42.1 tok/s | 842 tok/s |
| RTX 3090, experts on disk, artifact in the page cache | 22.6 GB | 0.9 GB + page cache | 47.0 tok/s | 39.1 tok/s | 630 tok/s |
| RTX 3090, experts on disk, page cache dropped every second (NVMe) | 22.6 GB | 0.9 GB | 17.2 tok/s | 11.3 tok/s | 195 tok/s |

The IQ3_S release on one RTX 3090 (310 W, PCIe 4.0 x16) in a host with 62 GB of RAM, 23 cores of an
AMD EPYC 7663 and an NVMe drive, October 5, with the Q2_0 release in the same sitting; decode counts
the 86 tokens of the short answer:

| Release and placement | Device memory | Host memory | Decode, short answer | Decode after 4,463 tokens | Prefill |
|---|---:|---:|---:|---:|---:|
| IQ3_S, experts in pinned host memory, 16.2 GB expert cache | 22.1 GiB | 50.3 GB pinned | 34.4 tok/s | 30.2 tok/s | 533 tok/s |
| IQ3_S, experts on disk, the page cache holding what of the 83.6 GB file fits | 22.1 GiB | page cache | 19.0 tok/s | 21.6 tok/s | 209 tok/s |
| IQ3_S, experts on disk, the file's pages evicted every second | 22.1 GiB | — | 10.7 tok/s | 5.4 tok/s | 47 tok/s |
| Q2_0, experts in pinned host memory | 22.1 GiB | 34.0 GB pinned | 50.2 tok/s | 43.7 tok/s | 847 tok/s |

The Q2_0 release (with the separate table artifact) on RTX 4090s at 450 W (PCIe 4.0 x16, no peer
access) in a cloud VM with two EPYC 7543 sockets of 60 vCPUs each, both GPUs on NUMA node 0, 694 GB
of RAM and its container disk, October 6. The single-GPU rows ran pinned to node 0's CPUs
(`taskset -c 0-59`); the ranges are two runs:

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

- **Device expert cache.** With host or disk experts the cache takes the memory the GPU has free
  and fills while the requests run: 94 GB of it on the PRO 6000, where nearly every expert ends up
  resident, and 30 GB on the RTX 5090.
- **Host link.** Host and disk experts cross the host link: PCIe 5.0 x16 on the RTX 5090 host,
  PCIe 4.0 x16 on the PRO 6000 host, the likeliest reason the RTX 5090's host-expert row beats the
  PRO 6000's.
- **CUDA graphs** add 27% to the short-answer decode on the two RTX 5090s (107.0 tok/s eager), 23
  to 26% on the PRO 6000 and 11% on the two RTX 3090 Ti; disk experts decode eagerly.
- **Prefill chunk.** Prefill touches nearly every expert of every layer in each chunk, so with host
  or disk experts a larger `--prefill-chunk` serves more tokens per copy: on an RTX 3090 Ti the long
  prompt took 6.05 s in 512-token chunks and 4.58 s in 2,048.

Every GSQ-RCO release (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S and the Coder IQ1_M) answers the generate
test's prompts with disk and host experts, with and without CUDA graphs; with disk experts the
process peaks at 1.05 to 1.19 GB of RAM. The published layout (models without the table, the shared
table artifact) passed device, host and disk placements, batched against single decoding, a
restored snapshot, an image question per release and the server's concurrency, prefix reuse and
JSON Schema checks on one NVIDIA L40S.

Where a decode step goes, from an Nsight Systems trace of the generate test's graph-replayed steps
(Q2_0, every expert on that L40S): about 1,780 kernels and copies in 10.6 ms, 0.5 ms of it idle
between them. A token read about 4 GB of weights when the hyper-connection projections were BF16,
the largest share (1.27 GB, more than the ten routed experts of every layer, 0.66 GB); they are
Q8_0 now, 0.67 GB.

### Host experts on one RTX 3090

One RTX 3090 (24 GB, PCIe 4.0 x16) with an i9-11900KF and 125 GB RAM, driver 580.82.09 and
CUDA 13.1 ran the Q2 model (GSQ-RCO Q2_0 with the IQ4_NL table) through `ninfer-serve` with host
experts, int8 KV, an 8,192-position context and one request at a time. Three prompts (a Python
LRU cache, the blue sky, the history of the Great Wall in Chinese) ran in that order twice, 256
greedy output tokens each, so a prompt's second run follows the other two. Each binary ran every
configuration once; the baseline is commit `a9155e0`.

| Mode | Baseline decode, tok/s (first / second run) | October 9 build |
|---|---|---|
| plain, code / prose / Chinese | 69.2/78.4, 77.9/80.6, 74.1/79.4 | 87.3/92.3, 93.7/94.3, 93.4/94.3 |
| MTP 2 drafts | 68.6/70.0, 71.9/73.4, 57.2/62.4 | 118.8/118.6, 117.3/114.8, 95.8/101.2 |
| MTP 3 drafts | 66.7/66.9, 67.2/68.3, 53.0/57.5 | 121.2/122.6, 117.7/111.1, 90.0/94.6 |
| MTP 4 drafts | 63.1/62.2, 60.5/60.9, 44.9/48.2 | 111.1/105.9, 101.9/96.2, 74.6/78.5 |
| disk experts, code / prose (first run) | 64.5, 66.8 | 68.1, 71.4 |

The baseline predates the tiered admission of staged misses and Q2_0's transposed decoding. Run
back to back, a repeated prompt decodes at
98.5 (code) and 100.4 (prose) tokens/s plain and 137.6 and 123.7 with three drafts. A cache that
admitted every miss into the least recently used slot let a passing decode evict the experts other
prompts keep routing to; the tiers (below 1.5, below 6, and more routes) restored second-run starts
of 212-305 ms instead of 284-397 ms on a second RTX 3090 for at most 4.4% of first-run decode.

### Host experts and long contexts on one RTX 4090

One RTX 4090 (450 W, PCIe 4.0 x16, driver 590.48.01, CUDA 12.8) in a Vast container with two
EPYC 7742 sockets, a 61.44-CPU quota and 503 GB of RAM ran the IQ3_S release with its IQ4_NL table
through `ninfer-serve`: host experts, one request at a time, 256 greedy output tokens, every server
pinned to the GPU's NUMA node (`taskset -c 0-63,128-191`), 2026-10-20. The baseline is commit
`5ed86dff9`, the build before the expert tier, the CPU share by default, MTP by default and the
streamed sparse-attention KV; its default copies every miss and drafts nothing, and its MTP row
adds `--spec mtp --draft-tokens 3 --expert-dma-share auto`.

At a 65,536-position context, a Python LRU cache prompt and a blue-sky prompt twice and a
31,865-token document once per server, two rounds in rotated server order (eight short answers
per row):

| Build | Expert cache | Decode, short prompts | Decode after 31,865 tokens | Prefill | Bus bytes |
|---|---:|---|---|---|---:|
| baseline | 14,288 MiB | 61.4-75.5 (68.3) tok/s | 68.7 tok/s | 2,833-2,846 tok/s | 660 GB |
| baseline, MTP and CPU share | 14,080 MiB | 66.5-77.1 (71.4) tok/s | 66.5-75.0 tok/s | 2,576-2,714 tok/s | 901 GB |
| defaults | 16,059 MiB | 72.6-82.5 (77.0) tok/s | 66.0-79.0 tok/s | 2,840 tok/s | 223 GB |
| defaults, 59 CPU threads | 16,059 MiB | 70.2-77.8 (73.1) tok/s | 64.4-75.6 tok/s | 2,818-2,831 tok/s | 218 GB |
| defaults, 59 CPU threads, `--kv-resident all` | 15,264 MiB | 68.7-76.9 (71.8) tok/s | 63.2-75.2 tok/s | 2,833-2,859 tok/s | 226 GB |

The bus bytes are every expert byte the server copied to the GPU over its ten requests. The rows
with 59 CPU threads (the pool's size before it took half the CPUs) ran in one sitting with the
baseline; the defaults ran about an hour later on the same host. The defaults copy a quarter of the baseline MTP row's bytes and decode the
short prompts 7.8% faster on its mean, with overlapping ranges; streamed and resident KV decode
alike at this context. Before the prefill buffers became cache slots the 59-thread defaults' cache
held 14,776 MiB and decoded the short prompts at 64.9-72.0 (68.3) tok/s.

An Nsight Systems trace of one 256-token code answer shows where the step goes instead. With the
defaults (then a 59-thread CPU pool) the GPU spent 39% of the answer computing, 38% in the kernels
that wait for the host's expert service (the missing experts' copies and the CPU's products) and
23% idle, and the copies took 23.8 GB; the baseline's MTP row spent 41%, 38% and 21%, with
42.5 GB of copies.

At a 458,752-position context, the same two short prompts and a 136,077-token document once,
one server each: the baseline keeps every layer's KV on the device and has 4,589 MiB left for
experts; the defaults keep the 12 sparse-attention layers' KV in pinned host memory (11,648 MiB)
with 832 MiB of their blocks on the device, and the cache takes 14,643 MiB.

| Build | Expert cache | Decode, code / prose | Decode after 136,077 tokens | Prefill |
|---|---:|---|---|---|
| baseline | 4,589 MiB | 39.2 / 43.0 tok/s | 37.7 tok/s | 2,612 tok/s |
| defaults, 59 CPU threads, `--kv-resident all` | 4,755 MiB | 39.0 / 31.3 tok/s | 34.7 tok/s | 2,548 tok/s |
| defaults, 59 CPU threads | 14,643 MiB | 70.0 / 67.2 tok/s | 62.3 tok/s | 2,700 tok/s |
| defaults | 14,643 MiB | 70.3 / 67.3 tok/s | 67.6 tok/s | 2,708 tok/s |

The baseline's and the defaults' answers part after a few tokens to a few hundred. Since the CPU's
products of K-quant, Q4_0 and Q5_0 blocks were made equal to the GPU's, the GPU's vector kernels
round each step of those slices' products, the one change to the GPU arithmetic of this release's
decode, which keeps 177 tensors in Q4_K, Q5_K and Q6_K; prompt chunks run the matrix kernels, which
did not change. The defaults' configurations (with and without lent buffers, 30 or 59 CPU threads,
streamed or resident KV) give each other's answers token for token with equal log probabilities, and
the 40 KB perplexity slice scores 1.980071 on both builds.

### The CPU share on one RTX 4090

The RTX 4090 host of the previous section, servers pinned to the GPU's NUMA node, the IQ3_S release
at a 65,536-position context with its other defaults, 2026-10-20. Three prompts (a Python LRU
cache, the blue sky, the history of the Great Wall in Chinese) ran twice per server with 256 greedy
output tokens, in two rounds whose server order rotates: 12 answers per row.

| `--expert-dma-share`, `--expert-cpu-threads` | Decode, tok/s (range, mean) | Misses on the CPU | Bus bytes per token |
|---|---|---:|---:|
| 1, the GPU only | 62.7-75.2, 68.4 | — | 172.5 MB |
| `auto`, 59 threads | 64.3-78.6, 71.1 | 37.3% | 85.8 MB |
| `auto`, 16 threads | 65.4-82.8, 74.2 | 29.9% | 102.4 MB |

Every answer with a CPU share has the GPU-only answer's tokens and log probabilities. The bus bytes
are every expert byte the server copied, prompt chunks included, over the tokens it generated. An
expert the CPU computes takes no slot and crosses no bus, so the share halves the bytes; decode
gains less, 4% with 59 threads and 8% with 16: the bus no longer bounds a step on this host. With
59 threads the CPU took a larger share of the misses and still decoded slower than with 16 in 11 of
the 12 pairs of answers. A second sweep (the same three prompts once per server, two rounds) decoded
at a mean of 72.5 tok/s with 8 threads, 74.4 with 16, 74.3 with 24, 73.4 with 32 and 71.3 with 59,
every answer equal; the default pool takes half the process's CPUs (30 here). Spinning on all 61
CPUs of the container's quota, the pool had the process throttled in 208 of 211 scheduling periods
while it decoded.

A second RTX 4090 rental with a 30-CPU quota (the same release and context, 2026-10-20, a later
build) answered the Python and blue-sky prompts twice and a 31,865-token prompt once per server:

| `--expert-cpu-threads` | Decode, tok/s: Python, blue sky, 32K prompt, Python, blue sky | Throttled periods |
|---|---|---:|
| 8 | 81.8, 76.4, 70.4, 82.0, 76.4 | 4 of 291 |
| 15 (the default here) | 84.8, 79.5, 75.2, 84.5, 84.0 | 3 of 278 |
| 22 | 91.3, 82.8, 77.4, 91.4, 87.0 | 4 of 270 |
| 29 | 92.3, 85.2, 76.5, 92.6, 86.7 | 13 of 271 |

Every answer was equal. Here more threads decoded 3-8% faster at 22 than at the default, where the
first host gained nothing past 16; the default stays at half the CPUs, and `--expert-cpu-threads`
tunes a host.

### The expert tier's paced rounds on one RTX 4090

One RTX 4090 with a Ryzen 9 9950X (32 CPUs, the default 15-thread pool), the IQ3_S release at a
65,536-position context with its other defaults, 2026-10-20: one server per `--kv-dtype` format and
build, the Python LRU cache and blue-sky prompts twice and the 31,865-token prompt once, 256 greedy
output tokens. Starting a round's swaps a quarter per pass instead of all after the round's pass
([the expert cache](#the-expert-cache)) cut the GPU's wait for the first layer's misses from 2.6 to
0.9 ms a step and decoded faster with every format:

| `--kv-dtype` | bf16 | int8 | fp8 | k8v4 | nvfp4 | rk8v4 | rk4v4 | rk4v4-e8 | rk2v4-e8 |
|---|---|---|---|---|---|---|---|---|---|
| Decode, whole round / paced, tok/s | 86.7 / 91.8 | 88.9 / 93.0 | 85.3 / 92.4 | 90.8 / 95.3 | 86.8 / 92.8 | 92.7 / 94.6 | 89.0 / 97.2 | 88.1 / 93.6 | 75.6 / 82.9 |
| Paced / whole round, geometric mean | 1.058 | 1.045 | 1.083 | 1.049 | 1.068 | 1.018 | 1.093 | 1.063 | 1.096 |

Over the 45 pairs of answers the paced build decoded 0.970-1.144 times as fast, 1.064 on the
geometric mean; three pairs (two rk8v4 and one int8 blue-sky answer) were slower. Every answer had
the same tokens.

On the same host `auto` had read the CPU's gate and up time as nearly zero in about two of five
calls: the CPU pool's calling thread could return before the last job's thread stamped its end, and
read the previous phase's stamp. With the stamp in place before the pool returns, the same 45 pairs
decoded 0.985-1.128 times as fast, 1.031 on the geometric mean; per format 0.990 (rk4v4, all five
pairs slower by 0.2-1.5%) to 1.068 (k8v4). Every answer had the same tokens.

### MTP

The Q2_0 release with Unsloth's `shared-Q8_0` MTP block, converted into one file with its
table, on two RTX 3090s (350 W, PCIe 4.0 x16, one GPU per socket of an EPYC 7663 host with 629 GB
of RAM, no peer access), 2026-10-07: greedy decoding of a 117-token answer to a 37-token coding
prompt, two runs each. Every speculative output is the plain decode's, token for token.

| Placement | Drafts | Decode | Drafts accepted | Tokens a round |
|---|---:|---:|---:|---:|
| 2× RTX 3090, experts on the GPUs | none | 92.9-93.0 tok/s | | |
| 2× RTX 3090, experts on the GPUs | 1 | 130.1-130.9 tok/s | 96.6% | 1.97 |
| 2× RTX 3090, experts on the GPUs | 2 | 150.3-150.5 tok/s | 91.5% | 2.83 |
| 2× RTX 3090, experts on the GPUs | 3 | 167.8-171.0 tok/s | 95.6% | 3.87 |
| 2× RTX 3090, experts on the GPUs | 4 | 171.9-173.6 tok/s | 91.0% | 4.64 |
| RTX 3090, experts in pinned host memory | none | 39.3-44.1 tok/s | | |
| RTX 3090, experts in pinned host memory | 3 | 47.4-49.4 tok/s | 95.6% | 3.87 |

Coding answers are what MTP drafts best; a prose answer (80 tokens on the water cycle) keeps 38.6%
of three drafts, 2.16 tokens a round, and decodes at 91.6 tok/s, no faster than without drafts.
With host experts a verification's four columns route to up to four times as many experts, each
missing one crossing the bus, so three drafts gained about 12% here; after the October 9
host-expert work three drafts decode 90-123 tok/s against 87-94 plain (table above).

### Hyper-connections in Q8_0

The hyper-connection projections (97 down/up pairs and 96 inject matrices) are stored as ggml Q8_0,
0.67 GB instead of 1.27 GB in BF16. On the rented RTX 3090 of the tier measurements, plain decode
went from 81.3-93.6 to 84.5-97.7 tok/s and decode with three MTP drafts from 84.5-114.4 to 83.0-120.4.
Up to eight tokens the kernels decode the weights to exact FP32 values; wider calls round them to
BF16 for the tensor-core products, which costs prompts and batches of nine or more tokens 5-57% more
for these reads. Perplexity with host experts (4,096-token windows advancing by 2,048, int8 KV):

| Corpus | BF16 | Q8_0 | Change |
|---|---:|---:|---:|
| `perplexity-1m` | 3.93783 | 3.93842 | +0.015% |
| `perplexity-heldout-2026-09` | 4.69573 | 4.69673 | +0.021% |

### Against llama.cpp

llama.cpp runs the same GGUFs with the experts on the CPU (`--n-cpu-moe 48`). On the IQ3_S host
above (23 threads) llama-bench gives 29.6 tok/s decode (tg128) and 312 tok/s prefill (pp512) for
IQ3_S, and 12.7 and 368 tok/s for Q2_0; the Q2_0 RTX 3090 host (32 threads) gave 11.2 and 267
tok/s for Q2_0. Its speed follows the host CPU, and its Q2_0 CPU path is the slower one.

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

### Reasoning benchmarks

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
