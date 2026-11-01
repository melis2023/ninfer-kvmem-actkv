# Flint

NInfer runs Recursive's [Flint](https://huggingface.co/RecursiveSI/flint) (`FlintForCausalLM`), a
39.8B-parameter mixture-of-experts coding model with about 2.5B parameters active per token, from a
llama.cpp GGUF: Unsloth's [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) or any other
quantization of the release. It is its own model family next to Qwen3.5, Flash-Next and Gemma 4:
its own executor (`src/models/flint`) and Engine core (`src/runtime/engine/flint_core`), sharing
the Qwen3.5 frontend for tokenization, chat templates, output parsing and the serving protocols.

## What runs

- **The model.** 49 blocks of hidden width 2048 with grouped-query attention (32 query heads over 4
  key/value heads of width 128), per-head QK RMS norms and a learned sink logit per query head that
  joins each softmax denominator. 35 blocks attend over a sliding window of 513 positions (the
  query's own and the 512 before it) with NeoX RoPE (theta 10,000); 14 full-attention blocks carry
  no positional encoding at all. Blocks 0 and 1 run a dense SwiGLU MLP (6144); the other 47 route
  each token to 6 of 256 experts (SwiGLU, width 512) by sigmoid scores renormalized over the six,
  beside an always-active shared SwiGLU expert (1024). An untied 150,144-token head.
- **The weights stay the GGUF's blocks.** Every matrix keeps its ggml format (UD-Q4_K_XL mixes
  Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS and IQ3_S), imported row for row without requantization and
  multiplied by the GGUF kernels: vector kernels for up to eight tokens, integer matrix kernels for
  prompt chunks, the expert products over the routed experts only. The router, norms and sinks are
  FP32 vectors.
- **Attention.** Prompt chunks run llama.cpp's tensor-core flash attention with the sinks added to
  its softmax; decode runs a kernel of its own: one block per sequence, KV head and key split, whose
  eight warps are the eight query heads of a KV head reading each key tile once, merged with the
  sinks in a second pass. The sliding blocks keep a ring of 768 positions per sequence, the
  full-attention blocks every position, all in F16.
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens (1024 or 512 on a card without room for them), prefix reuse across turns and requests
  with host-memory images ([below](#prefix-reuse)), branched reads for
  [typed decisions](serving.md#typed-decisions) (a choice of at most 26 options: the tokenizer
  groups up to three digits into one token, so a two-digit label cannot be read a digit at a time;
  [how the labels move](#rounding-with-the-pass)), structured output, streaming, causal scoring
  (`ninfer-perplexity`), and the model's GLM-4.7 tool calls
  (`<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>`) returned as
  OpenAI tool calls, each value read by its declared parameter type. A forced `tool_choice` ends
  the generation prompt with `<tool_call>NAME`, so the model continues with the arguments.
- **Reasoning.** The release's chat template has no thinking switch: the model opens `<think>`
  itself when it decides to reason, and the reasoning arrives as `reasoning_content`.
  `enable_thinking: false` (or `reasoning_effort: "none"`) closes an empty reasoning block after
  the assistant marker, the switch GLM's templates use for the same role-marker format, and Flint
  then answers at once; [typed decisions](serving.md#typed-decisions) read their labels there.
  Without it, a one-label question puts about 10^-11 of the first token's probability on the
  labels, the rest on `<think>`. Flint has no early-close control, so thinking budgets do not
  apply and the [reasoning-loop guard](serving.md#openai-chat-completions) stops a looping reply
  under `conclude` as under `stop`.
- **Sampling defaults.** The release's: temperature 1.0, top-p 0.95, top-k 20.
- **Speculative decoding by context lookup.** Recursive publishes no draft model. `--lookup-ngram
  N` proposes what followed the last N tokens' earlier occurrence in the sequence (up to
  `--draft-tokens`, 3 by default, at most 7), and the target verifies the proposal in one pass of
  decode rows; the output is the one plain decoding gives.
- **Expert residency.** The routed experts stay on the GPU, in pinned host memory or in the
  artifact's files, with a device cache and a CPU share of decode misses
  ([below](#expert-residency)).

Not built for this family: the quantized `--kv-dtype` formats (the KV cache is F16), MTP, DFlash
and n-gram copy drafts, rotation scaling (`--rope-yarn`, `--rope-scaling-factor`: the
full-attention blocks have no rotation to scale), strict tool arguments and structured output beside
tools (Flint's tool markup has no grammar here), images and video (Flint reads text only), pipeline
stages, and of the context cache the shared paged KV pool, the disk tier and session slots.

## Converting

The converter reads the weights from the GGUF and the tokenizer, chat template and generation
config from the Hugging Face release:

```bash
mkdir -p models/flint/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
  curl -fsSL -o models/flint/hf/$f https://huggingface.co/RecursiveSI/flint/resolve/main/$f
done
hf download unsloth/flint-GGUF flint-UD-Q4_K_XL.gguf --local-dir models/flint

python -m tools.convert --model models/flint/hf --recipe flint_gguf \
  --source gguf=models/flint/flint-UD-Q4_K_XL.gguf --name Flint-UD-Q4_K_XL \
  --out models/flint-UD-Q4_K_XL-ninfer-v3.ninfer --device cpu
```

The conversion takes about 35 seconds on a 64-CPU container and writes one 25,109,604,096-byte
(23.4 GiB) `.ninfer` file. From UD-Q4_K_XL it holds attention in Q8_0 (Q5_K in 7 of the 49
blocks), the dense MLPs, token table and head in Q8_0, the shared experts in Q8_0, Q5_K or
IQ4_XS, the routed experts' [gate; up] banks in IQ4_XS, Q5_K, Q4_K or IQ3_S and their down banks
in Q5_K, IQ4_XS, Q6_K or Q8_0, block by block as the GGUF chose them.

## Running

```bash
# A chat server for four concurrent requests over 64K-token contexts.
ninfer-serve models/flint-UD-Q4_K_XL-ninfer-v3.ninfer --model-id flint \
  --max-context 65536 --max-concurrency 4

# One prompt from the CLI.
ninfer models/flint-UD-Q4_K_XL-ninfer-v3.ninfer --prompt "Write a binary search in Rust."

# Experts in pinned host memory, for a 16 or 24 GB card.
ninfer-serve models/flint-UD-Q4_K_XL-ninfer-v3.ninfer --model-id flint \
  --max-context 65536 --expert-residency host
```

With its experts on the GPU the model needs 23.4 GiB of weights, 0.7 GiB of workspace and, per
sequence, 28 KiB of F16 keys and values per token of context (its 14 full-attention blocks) beside
53 MiB of sliding-window rings: about 25.5 GiB for one 32K-token sequence, more than a 24 GB card
holds. With `--expert-residency host` the GPU keeps 1.9 GiB of weights, two layer banks of 624 MiB and a
cache of as many experts as fit, and the host pins 21.5 GiB for the routed experts; `disk` reads
them from the artifact through the OS page cache and pins 137 MiB of transfer buffers.

## Prefix reuse

With the context cache on (the default), a prompt resumes from a state a sequence or the host store
keeps instead of prefilling those tokens again:

- **On its sequence.** A sequence keeps its state when its request ends, and a checkpoint of its
  sliding blocks' rings where the prompt's last user turn closes; the next turn of a conversation
  resumes from either.
- **From host memory.** Before another request takes a sequence, what it held goes to an image in
  pinned host memory: the rings (53 MiB) and the full-attention keys and values up to its length
  (28 KiB per token). A prompt on any sequence resumes from the longest image it starts with.
  `--host-kv-mib` (or `--host-cache-mib`) sets the store, 8 GiB by default, pinned at startup in
  8 MiB pages; the least recently used images go first, and a conversation's prefixes shorter than
  1,024 tokens are not kept.
- **Shared prefixes.** A prompt chunk also ends where the frontend marks a prefix other requests
  share (the end of the leading system messages, of the tool declarations, a declared cache
  boundary), and the state there goes to an image too when saving it and copying it back take less
  time than computing its tokens again (about 50 tokens on). A disposable request (a decision read)
  leaves no image of its end.

A sliding-window state cannot be cut back: the rings at a position hold the window before it, and
each block's window reaches further back through the blocks below. A prompt therefore resumes only
where a state was kept, not at an arbitrary common prefix.

## Measurements

`ninfer_bench` against llama.cpp b11804's `llama-bench` on the same UD-Q4_K_XL GGUF (flash
attention on, F16 KV), one sequence, three repetitions each. NInfer's prompts are Flint-tokenized
source code, llama-bench's its own tokens; the experts a prompt reaches depend on its tokens. Decode
after 10,000 tokens follows a 10,000-token prompt (`-pg 10000,128` and `tg128 @ d10000`). The
23.4 GiB of weights fit only the RTX 5090, where both engines keep every expert on the GPU. On the
24 GB cards NInfer keeps the experts in pinned host memory (`--expert-residency host`) with the
cache it sizes itself, 19.3 GiB, and llama.cpp keeps the experts of its first four blocks on the
CPU (`-ncmoe 4`; with fewer, the 10,000-token prompt ran out of memory).

| Card | Engine | Prefill 2,048 | Prefill 10,000 | Decode 128 | Decode after 10,000 |
|---|---|---:|---:|---:|---:|
| RTX 5090 | NInfer | 15,670 | 14,656 | 347.7 | 314.6 |
| RTX 5090 | llama.cpp | 8,702 | 8,679 | 222.7 | 209.2 |
| RTX 4090 | NInfer, host experts | 12,423 | 11,466 | 210.2 | 188.0 |
| RTX 4090 | llama.cpp, `-ncmoe 4` | 4,868 | 4,725 | 165.7 | 149.4 |
| RTX 3090 | NInfer, host experts | 6,225 | 5,755 | 161.4 | 140.2 |
| RTX 3090 | llama.cpp, `-ncmoe 4` | 2,633 | 2,553 | 79.9 | 77.1 |

NInfer against llama.cpp: 1.80, 1.69, 1.56 and 1.50 times on the RTX 5090; 2.55, 2.43, 1.27 and
1.26 on the RTX 4090; 2.36, 2.25, 2.02 and 1.82 on the RTX 3090. NInfer's spreads stay under 1.1%.
llama.cpp's stay under 2.7% except its 5090 prefill of 2,048 tokens (±4.4%) and its 3090 decode
after 10,000 tokens (±7.4%); its 5090 decode after 10,000 tokens comes from a second run of five
repetitions, the first having spread ±22%.

| Card | Board power | Host CPU | CUDA | Build |
|---|---:|---|---:|---|
| RTX 5090 | 575 W | 2 × EPYC 9654 | 13.4 | `120a` |
| RTX 4090 | 500 W | Ryzen 9 7950X | 13.4 | `89` |
| RTX 3090 | 370 W | Threadripper PRO 3955WX | 12.8 | `86` |

- **Context lookup.** `--lookup-ngram 4` through `ninfer-serve` on the RTX 5090, a greedy rewrite
  of a six-function Python module with type hints and docstrings (1,024 tokens, reasoning first):
  335.0 → 377.7 tok/s (1.13x), 164 of 255 proposed tokens accepted. The output equals plain
  decoding's.
- **Typed decisions.** On an NVIDIA A40 (48 GB, `sm_86`, 300 W), `scripts/pods/decision_bench.py`
  (four sequences): about a 340-token state, 1, 4 and 16 questions answer in 118, 135 and 236 ms
  cold and 59, 95 and 203 ms once the state is kept; about an 11,660-token state in 2.20, 2.22 and
  2.39 s cold and 68, 118 and 274 ms warm. A cold read runs the state's last chunk and the first
  questions in one pass; reading them after it took 162, 195 and 299 ms (short state) and 2.24,
  2.27 and 2.44 s (long one).

## Numerical checks

- **Kernels.** `ninfer_flint_kernels_test` qualifies the executor's own kernels against FP64
  oracles at Flint's shapes: decode attention with the sinks, the QK norms with RoPE and their
  cache slots, routing, the norms and expert combination, the prompt and branch masks, and the
  F16-sum guard below.
- **Against an FP64 forward.** `tools/reference/flint.py` runs the model in FP64 over the GGUF's
  exactly decoded blocks. On the first 640 tokens of WikiText (past the 513-position window) NInfer
  scores a total NLL of 1,938.86 against its 1,931.84 (+0.36%), the same to the last digit with
  device, host and disk experts.
- **Against llama.cpp.** Perplexity on 120 KB of WikiText over the five 4,096-token windows both
  score alike: 15.270 against llama.cpp's 15.195 (+0.49%); per window the mean NLL differs by
  -0.012 to +0.022 nats. On 40 KB, device, host and disk experts give the same perplexity,
  19.560887.
- **Branched reads.** `ninfer_flint_branches_real_test` (with `NINFER_FLINT_ARTIFACT`) checks
  that a branch alone reads bit for bit as a plain chunk of its tokens; that beside another
  branch, first or second, it reads the same whatever that branch holds; and that a read joining
  the prompt's last chunk leaves nothing of its branches in the sequence.

### Rounding with the pass

The prompt route's rounding depends on the pass a row runs in: how many rows it has and how its
tokens spread over the experts. The same 51 tokens' last logits move by up to 1.5 (the largest
change over the vocabulary) when other rows share their pass, and by about 1 when their last three
run as vector rows, as a chat's last step does after its user turn closes. A typed decision's
labels move with it. For one choice about a 200-token state (billing, shipping, technical):

| Path | billing | shipping | technical |
|---|---:|---:|---:|
| FP64 forward | 0.819 | 0.017 | 0.164 |
| Chat, last three tokens as vector rows | 0.841 | 0.021 | 0.138 |
| Read alone | 0.787 | 0.025 | 0.188 |
| Read beside another question, first or second | 0.910, 0.901 | 0.011, 0.018 | 0.079, 0.081 |

`decision_bench.py` therefore fails two of its checks on Flint: a label's log probability against
a chat's moved by 0.60 (it allows 0.4), and against the same question read alone, a sure answer
(margin 0.5 or more) moved by 0.082 among 16 questions (it allows 0.03). No answer with a margin
of 0.1 or more changed its choice.

### F16 activation sums

The integer matrix kernels quantize a prompt chunk's activation to q8_1 rows. For Q4_K, Q5_K and
the other types whose blocks carry a minimum, each 32 values' sum is stored in F16 beside the
scale, and a sum past 65,504 becomes infinite. Flint reaches that range: on WikiText a routed
expert's SwiGLU product reaches 81,621 (FP64 oracle) and a 32-value sum 89,234 in block 46, whose
down banks are Q5_K, so the unguarded product turns the whole chunk to NaN. The executor therefore
scales such rows by a power of two before quantizing them:

- **Routed experts.** A SwiGLU row whose largest 32-value sum passes 16,384 is written times
  2^-k (the least k that brings it under) and its route weight is multiplied by 2^k. Both factors
  are exact in BF16 and FP32, so the weighted product equals the unscaled one wherever the
  unscaled one is finite; other rows are unchanged bit for bit.
- **Products added to the residual.** Where such a weight adds to the residual (in UD-Q4_K_XL the
  shared expert's Q5_K down rows of 12 blocks and the Q5_K attention output of 7), the
  activation's columns are scaled the same way together with the residual's columns, and the
  residual's columns are multiplied back after the product: the residual receives the same sum.
- **Decode and verification rows.** The vector kernels' q8_1 rows keep each 32 values' sum in F16
  too, and their Q4_K, Q5_K and IQ1_S products read it (`gguf::vector_activation_f16_sums`). A
  call of at most eight tokens therefore quantizes its routed SwiGLU rows with the same guard
  (`quantize_pairs`), scales the columns of the Q5_K products into the residual as above, and the
  CPU share of host and disk experts applies the same factors, so its products stay the GPU's bit
  for bit. Unguarded, such a sum turned a decode step's logits to NaN and its text to garbage.
- **Not needed elsewhere.** The other activations are RMS-normed rows; their 32-value sums are at
  most sqrt(2048) times the norm weights' largest 32-value L2 norm, which is 1,744 for the
  attention inputs, 7,603 for the FFN inputs and 209 for the head in UD-Q4_K_XL.

## Expert residency

`--expert-residency device` keeps the routed GGUF banks on the GPU. `host` keeps them in pinned host
memory and `disk` in the artifact's files, read through the OS page cache. Both run all expert
arithmetic on the GPU with the resident kernels, through the transport Gemma 4 uses
([Gemma 4's expert residency](gemma4.md#expert-residency) describes the layer banks, the cache, its
rounds and the growing KV):

- **Layer banks.** A prompt chunk's experts missing from the cache land in a whole-layer bank laid
  out like the resident bank; two banks let the next layer arrive while the current one computes.
- **Cache.** `--expert-cache-mib` (default: what is free after startup) holds the experts decode
  routes most, filled at startup from a routing profile (`--expert-profile`, else the built-in
  Flint profile recorded on coding, chat and reasoning requests). `0` keeps the banks only. A
  slot takes the room of its layer's largest expert, from 1.6 to 2.5 MiB with the layer's formats,
  and each layer gets about as many slots: on a 24 GB card the 19.3 GiB cache holds 10,564 of the
  12,032 routed experts.
- **CPU share.** Of a decode or verification call's misses, CPU threads compute a share from the
  stored blocks while the GPU copies and runs the rest (`--expert-dma-share`, `auto` by default;
  `--expert-cpu-threads`). The CPU computes SwiGLU over the [gate; up] rows, the BF16 middle
  quantized as q8_1 and the down rows as the GPU's vector kernels do, into the same 2^-32
  fixed-point sums, so outputs do not depend on the split. Calls of more than eight tokens always
  use the GPU.

On the RTX 3090 of the measurements above, with the cache NInfer sizes itself (`ninfer_bench`, one
sequence, three repetitions; three CPU threads take a share of the decode misses). Disk experts read
through the page cache of a host with 503 GB of RAM, in whatever state the earlier runs left it,
which shows in their spread.

| Experts | Cache | Prefill 2,048 | Prefill 10,000 | Decode 128 | Decode after 10,000 |
|---|---:|---:|---:|---:|---:|
| host | 19.3 GiB | 6,225 | 5,755 | 161.4 | 140.2 |
| disk | 19.4 GiB | 4,495 | 3,879 | 130.0 ± 7.1 | 108.8 |
