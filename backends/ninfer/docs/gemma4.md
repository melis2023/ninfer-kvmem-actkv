# Gemma 4 26B-A4B

NInfer runs Google's Gemma 4 26B-A4B (`Gemma4ForConditionalGeneration`: the text model
`Gemma4ForCausalLM` and its vision tower) from the official quantization-aware-trained GGUF release
[`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf).
It is its own model family next to Qwen3.5 and Flash-Next: its own executor
(`src/models/gemma4`) and Engine core (`src/runtime/engine/gemma4_core`), sharing the Qwen3.5
frontend for tokenization, chat templates, output parsing and the serving protocols.

## What runs

- **The model.** 30 blocks of hidden width 2816: five sliding-window blocks (window 1024, head
  width 256, 16 query and 8 key/value heads) to every full-attention block (head width 512, 16 query
  heads over 2 key/value heads, keys used as values, proportional RoPE over a quarter of the
  width). Each block runs a dense GeGLU MLP (2112) beside 128 routed experts (8 per token, width
  704), with Gemma's sandwich norms and per-layer output scale, a tied 262,144-token head and the
  final logit soft cap.
- **The weights stay the release's blocks.** Every matrix is the GGUF's Q4_0 (the tied token table
  Q6_K), imported row for row without requantization (13.4 GiB) and multiplied by the GGUF kernels:
  vector kernels for up to eight tokens, integer matrix kernels for prompt chunks, the expert
  products over the routed experts only. A mixed-format GGUF of the same model (a Q4_K_M
  quantization with Q6_K values, Q5_0 or Q8_0 expert down projections) converts and runs the same
  way, each matrix in its own format.
- **The full-attention cache holds values only.** Those blocks' keys are their values times the
  key norm's gamma, rotated: the cache keeps V (10 KiB per token for the five blocks) and the
  attention derives each key from it as it reads it, with the same rotations the keys were stored
  with.
- **Every KV-cache format of the engine.** `--kv-dtype` keeps the caches in F16 (`bf16`, the
  default) or in int8, fp8, rk8v4, nvfp4, k8v4, rk4v4, rk4v4-e8 or rk2v4-e8, the engine's own
  codecs; see [below](#kv-cache-formats).
- **Speculative decoding with the release's assistant.** The multi-token-prediction drafter Google
  publishes beside the model (`Gemma4AssistantForCausalLM`, four blocks of width 1024 that read the
  target's own KV cache) drafts up to seven tokens a round (`--spec mtp --draft-tokens N`); the
  target verifies them in one pass and every token it keeps is its own sample, so the output is
  what plain decoding produces. See [below](#speculative-decoding-with-the-assistant).
- **Images and video.** The release's vision tower (27 blocks of width 1152) turns each picture or
  video frame into soft tokens in the prompt, on the GPU, streamed from pinned host memory, or on
  CPU threads (`--vision-residency`); see [below](#images-and-video).
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens (1024 or 512 on a card without room for them, [below](#memory-and-context)), prefix
  reuse across turns and requests with host-memory images ([below](#prefix-reuse)), structured
  output, tool calls in Gemma's
  `<|tool_call>call:NAME{...}<tool_call|>` form returned as OpenAI tool calls (a forced
  `tool_choice` ends the generation prompt with Gemma's own opener, `<|tool_call>call:NAME{`, so
  the model continues with the arguments; see [forced selection](serving.md#function-tools)),
  streaming, reasoning in its own channel when thinking is on, and causal scoring
  (`ninfer-perplexity`).
- **Thinking.** Off unless asked for, as the release's chat template has it: a request turns it on
  with `chat_template_kwargs: {"enable_thinking": true}`, and the reasoning arrives as
  `reasoning_content`. Gemma has no early-close control, so thinking budgets do not apply and the
  [reasoning-loop guard](serving.md#openai-chat-completions) stops a looping reply under `conclude`
  as under `stop`.
- **Sampling defaults.** The release's `generation_config.json` (temperature 1.0, top-p 0.95) with
  top-k held at the sampler's 20 candidates.

Not built for this family: DFlash and ngram copy drafts, pipeline stages over several GPUs, and of
the context cache the shared paged KV pool, the disk tier and session slots.

## Converting

The converter reads the weights from the GGUF and the tokenizer, chat template and generation
config from the Hugging Face release:

```bash
mkdir -p models/gemma4/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json \
         processor_config.json; do
  curl -fsSL -o models/gemma4/hf/$f https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/$f
done
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf --local-dir models/gemma4

python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf --name Gemma-4-26B-A4B-it-QAT \
  --out models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --device cpu
```

The conversion takes under three minutes and writes one 14.5 GB `.ninfer` file.

The assistant joins as the artifact's `mtp` component from a llama.cpp `gemma4-assistant` GGUF.
The QAT model takes the QAT assistant (Google's model card: a QAT target needs a QAT drafter);
Unsloth publishes it quantized to Q4_0, the format its training targets, and the converter imports
its blocks as they are (252 MB):

```bash
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF mtp-gemma-4-26B-A4B-it.gguf --local-dir models/gemma4
python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf --components text,mtp \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf \
  --source mtp=models/gemma4/mtp-gemma-4-26B-A4B-it.gguf --name Gemma-4-26B-A4B-it-QAT \
  --out models/gemma-4-26B-A4B-it-qat-q4_0-mtp.ninfer --device cpu
```

An artifact with the assistant runs without it too unless `--spec mtp` asks for it.

The vision tower joins as the `vision` component from the release's llama.cpp `mmproj` GGUF
(projector `gemma4v`): its BF16 matrices and FP32 norms as stored, the patch projection and the
position table stored as BF16 (their F32 values are BF16 values, which the converter checks), and
the release's `processor_config.json` for the image and video budgets (1.07 GiB more):

```bash
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B-it-mmproj.gguf --local-dir models/gemma4
python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf --components text,mtp,vision \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf \
  --source mtp=models/gemma4/mtp-gemma-4-26B-A4B-it.gguf \
  --source vision=models/gemma4/gemma-4-26B-it-mmproj.gguf --name Gemma-4-26B-A4B-it-QAT-Q4_0-MTP \
  --out models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --device cpu
```

This conversion (14.75 GiB) is published as
[WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3).

## Running

```bash
# A chat server for four concurrent requests over 40K-token contexts.
ninfer-serve models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4

# One prompt from the CLI.
ninfer models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --prompt "Explain RoPE in two sentences."

# With the assistant drafting two tokens a round.
ninfer-serve models/gemma-4-26B-A4B-it-qat-q4_0-mtp.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4 --spec mtp --draft-tokens 2

# Images and video as well.
ninfer-serve models/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4 --spec mtp --draft-tokens 2 --vision
```

Past the native 262,144 positions `--rope-yarn` (or a fixed `--rope-yarn-factor F`, `1..4`) rotates
the five full-attention blocks, the assistant's included, with YaRN at factor `--max-context` /
262,144, as it does for the Qwen families: their proportional RoPE schedule takes YaRN's
frequencies, and their queries the square of its attention factor; the sliding-window blocks never
see a distance past their 1024 positions and keep their rotation. Google publishes no extension of
this model's window, so this is the same mathematics without the vendor's measurement behind it:
the flag is accepted, the startup diagnostic and the `server_start` record's `engine.rope` say it
is on, and a perplexity past the window is the way to judge it on your text
([perplexity](perplexity.md#past-the-native-window)). Linear position interpolation
(`--rope-scaling-factor`) is refused for this model.

On a 24 GB card the weights take 13.4 GiB (the assistant 0.25 GiB more) and the prompt-chunk
workspace about 1.1 GiB. Every sequence holds its whole context from the start: the rings of the 25
sliding blocks (their 1024-token windows; 1280 rows with `--spec mtp`, whose rounds write their
drafts past the window), as much again for the rings' checkpoint while the context cache is on (the
default; `--no-prefix-reuse` drops it), and the values of the five full-attention blocks for every
context token. In F16 the rings take 200 MiB (250 MiB with `--spec mtp`) and the values 10 KiB per
token; the other formats' sizes are [below](#memory-and-context). Shared by all sequences: a 24 MiB
staging ring for prompt chunks, the keys a chunk derives from the values (2 KiB per context token),
with a quantized cache the F16 values a chunk reads back (2 KiB per context token), and the rotation
table (0.5 KiB per context token). In F16 with `--spec mtp`, one sequence of 40,960 tokens takes
1.0 GiB, four of 16,384 tokens 2.6 GiB and eight of 16,384 tokens 5.2 GiB. A configuration that
does not fit is refused at startup with the numbers.

## KV-cache formats

`--kv-dtype` selects the caches' storage. The 16-bit default keeps F16 rows (the prompt kernels'
type). Every other format of the engine stores its codes and scales per 256-wide block with the
engine's D256 codecs (`ops/kv_cache`), each block's codes and scales in two planes:

| `--kv-dtype` | sliding blocks: keys | sliding blocks: values | full-attention plane |
|---|---|---|---|
| `bf16` | F16 | F16 | F16 |
| `int8` | rotated INT8, G64 | INT8, G64 | INT8, G64 |
| `fp8` | rotated E4M3, one scale per 256 | E4M3, one scale per 256 | E4M3, one scale per 256 |
| `rk8v4` | rotated INT8, G64 | INT4, G32 | INT8, G64 |
| `nvfp4` | rotated NVFP4, G16 | rotated NVFP4, G16 | NVFP4, G16 |
| `k8v4` | rotated E4M3, one scale per 256 | rotated NVFP4, G16 | E4M3, one scale per 256 |
| `rk4v4` | rotated Lloyd-Max 4-bit, G64 | INT4, G32 | INT4, G32 |
| `rk4v4-e8` | rotated E8-lattice INT4, G64 | INT4, G32 | INT4, G32 |
| `rk2v4-e8` | rotated E8 roots (2 bytes per 8), G64 | INT4, G32 | INT4, G32 |

Rotated rows hold the normalized Sylvester-Hadamard transform of each block, as the engine's
formats store keys (and NVFP4 values): a decode step turns its queries to meet them, and a turned
value plane's output turns back. The full-attention blocks keep one plane, their values, from which
each key is derived (RoPE(v * k_norm)); the rotation by position rules out a Hadamard-rotated plane,
so it takes an unrotated coding at least as fine as the format's keys, or the packed INT4 value coding
for the 4-bit and 2-bit key formats.

A decode step stores its rows into the planes and its attention decodes them as it loads each tile
into shared memory. A prompt chunk still runs the F16 tensor-core attention: the window before it
comes back from the sequence's ring, its own rows are stored and replaced by the values they
represent, and a full-attention block's values come back from the plane up to the chunk's end, so a
chunk attends to what its decode steps will read. Speculative verification, the assistant's drafts
and prefix checkpoints read and copy the same planes.

### Memory and context

Per sequence, by format: the sliding blocks' rings (twice that with the context cache's checkpoint)
and the full-attention values per context token. The shared buffers above come on top.

| `--kv-dtype` | rings, plain | rings, `--spec mtp` | full-attention values per token |
|---|---:|---:|---:|
| `bf16` | 200.0 MiB | 250.0 MiB | 10,240 B |
| `int8` | 103.1 MiB | 128.9 MiB | 5,280 B |
| `fp8` | 100.8 MiB | 126.0 MiB | 5,160 B |
| `rk8v4` | 79.7 MiB | 99.6 MiB | 5,280 B |
| `k8v4` | 78.5 MiB | 98.1 MiB | 5,160 B |
| `nvfp4` | 56.2 MiB | 70.3 MiB | 2,880 B |
| `rk4v4`, `rk4v4-e8` | 54.7 MiB | 68.4 MiB | 2,880 B |
| `rk2v4-e8` | 42.2 MiB | 52.7 MiB | 2,880 B |

The longest `--max-context` that `ninfer-serve --max-concurrency 8` starts with on the 24 GB RTX
3090, HauhauCS artifact (its mixed-format weights leave 7.50 GiB free with the assistant, 7.72 GiB
without), context cache on, no vision tower; each leaves 0.35 to 0.46 GiB free after startup for
the CUDA graphs captured while serving:

| `--kv-dtype` | 8 sequences, `--spec mtp --draft-tokens 2` | 8 sequences, no drafts |
|---|---:|---:|
| `bf16` | 24,576 | 36,864 |
| `int8` | 82,944 | 96,256 |
| `fp8` | 84,992 | 98,304 |
| `rk8v4` | 92,160 | 103,424 |
| `k8v4` | 94,208 | 105,472 |
| `nvfp4` | 163,840 | 179,200 |
| `rk4v4`, `rk4v4-e8` | 164,864 | 180,224 |
| `rk2v4-e8` | 173,056 | 186,368 |

The startup check counts everything that grows with the context or the sequences (the caches, the
checkpoints, the prompt chunk's derived keys, F16 values and full-attention mask) and keeps 1.5 GiB
for the rest of the workspace and the CUDA graphs. A card without that room (a 16 GB one with the
assistant or the vision tower) prefills in 1,024- or 512-token chunks instead: the check then keeps
512 MiB, and the chunk's planned workspace with 320 MiB beside it must fit what the caches leave.

### Quality and speed

Perplexity of eight disjoint 8,192-token windows of the NInfer source stream (65,528 scored tokens,
`ninfer-perplexity --context 8192 --disjoint`), for Google's QAT conversion and for HauhauCS's
Balanced tune of it:

| `--kv-dtype` | Google QAT | HauhauCS Balanced |
|---|---:|---:|
| `bf16` | 21.061 | 19.592 |
| `int8` | 21.170 | 19.984 |
| `fp8` | 21.070 | 19.697 |
| `rk8v4` | 20.281 | 18.518 |
| `nvfp4` | 22.465 | 21.501 |
| `k8v4` | 24.096 | 21.735 |
| `rk4v4` | 20.419 | 18.758 |
| `rk4v4-e8` | 19.472 | 17.437 |
| `rk2v4-e8` | 21.937 | 20.875 |

The formats whose sliding values are INT4 score below F16 on both checkpoints, and the NVFP4 values
of `nvfp4` and `k8v4` above it. The kernels match their FP64 oracles (see
[Numerical checks](#numerical-checks)); why a coarser value cache lowers this perplexity is not
established, so take the 8-bit formats' agreement with F16 as the measure of their fidelity and
check a 4-bit format on the task it will serve.

llama.cpp (`c83f3058b`) on the same QAT GGUF moves too, and not monotonically:
`llama-perplexity -c 8192 --chunks 8 -fa on` over the same text (it scores the second half of
each window, so its values sit on another scale) gives 18.337 with an F16 cache, 18.780 with q4_0
values, 18.298 with q8_0 values, 18.243 with q8_0 keys and values, 18.851 with q8_0 keys and q4_0
values, and 18.243 with q4_0 keys and values.

Decode tok/s of 128 tokens on a rented RTX 3090 held at 300 W (so below the 350 W numbers in
[Measurements](#measurements)), HauhauCS artifact, one request, no drafts, `ninfer_bench`, three
repetitions; `bf16`, `k8v4` and `rk4v4` are the means of two runs, which differed by at most 0.9%:

| `--kv-dtype` | empty context | after 10,000 | after 40,000 |
|---|---:|---:|---:|
| `bf16` | 199.8 | 176.3 | 150.4 |
| `int8` | 195.0 | 175.2 | 149.2 |
| `rk8v4` | 194.6 | 175.9 | 149.7 |
| `rk4v4` | 193.8 | 175.5 | 150.5 |
| `rk4v4-e8` | 192.6 | 174.8 | 149.7 |
| `rk2v4-e8` | 185.6 | 167.7 | 144.7 |
| `fp8` | 177.9 | 154.7 | 130.5 |
| `k8v4` | 177.0 | 152.6 | 129.1 |
| `nvfp4` | 176.8 | 149.7 | 125.5 |

The INT8 and INT4 formats decode within 4% of F16 (within 1% after 10,000 tokens); `rk2v4-e8`
loses 4-7%; the FP8 and NVFP4 formats lose 11-17%, already on an empty context, where the cache
reads are few: the RTX 3090 has no E4M3 or E2M1 conversion instruction (sm_89 adds them) and those
decodes run in software. Prefill of 2,048, 10,000 and 40,000 tokens stays within 1.5% of F16 in every
format, the spread between runs.

## Prefix reuse

With the context cache on (the default), a prompt resumes from a state a sequence or the host store
keeps instead of prefilling those tokens again:

- **On its sequence.** A sequence keeps its state when its request ends, and a checkpoint of its
  sliding blocks' rings where the prompt's last user turn closes; the next turn of a conversation
  resumes from either.
- **From host memory.** Before another request takes a sequence, what it held (the conversation's
  end and its checkpoint) goes to an image in pinned host memory: the rings and the full-attention
  values up to its length, in the cache's format (150 MiB for 10,000 tokens in `rk8v4` with the
  assistant, 348 MiB in F16). A prompt on any sequence resumes from the longest image it starts
  with. `--host-kv-mib` (or `--host-cache-mib`) sets the store, 8 GiB by default, pinned at
  startup in 8 MiB pages; the least recently used images go first, and a conversation's prefixes
  shorter than 1,024 tokens are not kept. The copies queue on the GPU's stream behind the work
  before them, so no sequence waits for the host.
- **Shared prefixes.** A prompt chunk also ends where the frontend marks a prefix other requests
  share (the end of the leading system messages, of the tool declarations, a declared cache
  boundary), and the state there goes to an image too when saving it and copying it back (at
  12 GB/s each way) take less time than computing its tokens again at the measured prefill rate
  (0.2 ms a token until 4,096 tokens are measured): about 90 tokens on in `rk8v4`, 220 in F16 with
  the assistant. Requests with the same system prompt resume after it on any sequence, also when
  they arrive together, since a prompt that has not started prefilling looks for images again at
  its first chunk. A disposable request (a decision read) leaves no image of its end.

A sliding-window state cannot be cut back: the rings at a position hold the window before it, and
each block's window reaches further back through the blocks below. A prompt therefore resumes only
where a state was kept (a sequence's end, its checkpoint, a stored image), not at an arbitrary
common prefix.

## Measurements

An RTX 3090 (24 GB) at its 350 W power limit, CUDA 12.8; three repetitions each; `ninfer_bench`
over a 65,536-token slice of the perplexity corpus (its first stream, NInfer source text, tokenized
for Gemma) against `llama-bench` (llama.cpp `23b0202`, `-fa 1 -ngl 99`) on the same GGUF. Prefill is
tokens per second of a prompt of that length from an empty context; decode is 128 generated tokens
after a prompt of that length (llama.cpp: `tg128 @ d<N>`).

| Test | NInfer | llama.cpp |
|---|---:|---:|
| prefill 2,048 | 6,566 | 4,925 |
| prefill 10,000 | 6,045 | 4,697 |
| prefill 40,000 | 4,698 | 3,767 |
| decode, empty context | 215.4 | 168.6 |
| decode after 10,000 | 190.4 | 151.7 |
| decode after 40,000 | 162.2 | 135.7 |

Deriving the full-attention keys from the values and the eight-warp attention kernel moved decode
after 40,000 tokens from 148.5 to 162.2 tok/s (after 10,000: 182.6 to 190.4; empty context 216.2 to
215.4, within its spread) and halved those blocks' cache. Perplexity over eleven 8,192-token
windows of a NInfer source stream: 24.8036 with stored keys, 24.7948 with derived ones (the keys
are rounded from the F16 values rather than from the unrounded projection).

Chat requests in parallel streams to `ninfer-serve --max-concurrency N` (eight prompts, 384-token
greedy answers, three requests per stream) finish at 208.4 tok/s together for one stream, 310.1 for
two, 354.7 for three and 397.2 for four.

## Speculative decoding with the assistant

The assistant drafts from the token the next position takes (the anchor) and the target's
final-normed hidden state before it: each step projects [the target's scaled embedding of the last
token; the hidden state] to width 1024, runs its four blocks at the anchor's position with queries
only, over the keys and values the target holds (its sliding-window blocks the target's last
sliding block's window, its full-attention block the target's last full-attention block), and takes
the argmax of its head as the draft; the post-projection of its final state is the next step's
hidden state. The chain runs from one CUDA graph per batch and width.

The target then runs the anchor and the drafts of every speculating request as decode rows of one
pass: a request decoding alone takes the configured drafts, two or three decoding at once take one
draft each, and four or more decode plainly (see the measurements below). The
rows' logits are bit for bit those of decoding the same tokens one at a time, and the acceptance
draws each position as the request samples (greedy or the request's seeded sampler, penalties and
grammars included), keeping drafts while they equal the draw; the first difference, or the token
after the last draft, is the target's own draw. The output is therefore the one plain decoding
gives. `--adaptive-mtp` lets the round width follow each request's draft survival, and
`--lookup-ngram N` proposes what followed the last N tokens' earlier occurrence instead of the
assistant's guess when there is one.

Ten chat requests one at a time to `ninfer-serve` (code in Python and C++, an explanation, a
summary of a 1,000-token source excerpt, a factual question, a translation, a math word problem, a
story, a Russian answer, and a 13,600-token source excerpt to describe), 384 generated tokens each,
on the same RTX 3090. Every configuration generates the same tokens (3,664 greedy, 3,672 sampled):
the drafts change how many tokens a pass commits, not which.

| Decode tok/s | greedy | 13,600-token prompt | sampled (T 1, top-k 64, top-p 0.95) |
|---|---:|---:|---:|
| plain decoding | 209.1 | 186.0 | 207.9 |
| `--draft-tokens 1` (acceptance 0.81) | 280.1 | 240.2 | |
| `--draft-tokens 2` (0.70; 0.63 sampled) | 290.7 | 241.9 | 270.6 |
| `--draft-tokens 3` (0.62; 0.55 sampled) | 293.7 | 229.2 | 268.0 |

llama.cpp `23b0202`'s `llama-server` with the same GGUF and drafter (`--spec-type draft-mtp`,
thinking off) decodes the same greedy requests at 131.4 tok/s plainly, 186.4 with one draft, 201.6
with two and 195.2 with three (acceptance 0.79, 0.70, 0.61).

Two drafts are the best default: within 1% of three on short prompts, ahead after long ones and
when sampling. `--adaptive-mtp` with three drafts kept all three on this set (293.7), and
`--lookup-ngram 8` found little to copy (292.9). On the corpus continuation of `ninfer_bench`
(`-pg 10000,256;40000,256`), where the source text repeats, two drafts decode at 236.1 tok/s after
10,000 tokens and 210.1 after 40,000 (plain: 190.4 and 162.2).

A round of three drafts costs about two decode steps of GPU time (an `nsys` trace at 512 tokens of
context: the verification pass 8.0 ms against 4.7 ms for one token, the three draft steps 1.3 ms).
Most of the verification's extra time is the routed experts: four tokens touch about three times
as many experts as one, and their weights are read once per pass.

With several requests decoding the verification's extra rows route to ever more experts, so the
round drafts shrink as requests join (same streams as above, `--spec mtp`, tok/s together):

| Streams | plain | with the assistant |
|---|---:|---:|
| 1 | 208.4 | 295.5 with three drafts, 291.8 with two |
| 2 | 310.1 | 370.0 with one draft each (332.5 with two, 330.0 with three) |
| 3 | 354.7 | 365.7 with one draft each |
| 4 | 397.2 | 387.6 with one draft each, so four or more decode plainly |

## Structured output

A request with `response_format` or another constraint of
[Structured output](serving.md#structured-output) (with `--structured-output`) samples under its
grammar's token mask; strict tool arguments are refused, since Gemma's tool-call markup has no
grammar here. The mask of its next token is computed on a thread of its own from the grammar's state
after the last accepted token, while the GPU runs the decode step that produces the token's
logits: the step does not wait for the mask, and a row whose mask is late keeps its logits and is
sampled once the mask is ready, while the other requests go on. With `--spec mtp` such a request
also drafts when the usual context, output-budget and assistant-state checks allow it: a speculative
round fills one mask per draft position in parallel while the verification runs. Mask duration
does not disable drafts; slow masks can extend the speculative round's sampling wait.
A single mask over 5 s, or a grammar with no admissible token, fails the request with an
explicit error.

String length bounds (`minLength`, `maxLength`) compile to a counted repetition whose masks do
not depend on the count, in or out of `anyOf`: with Qwen3.5's 248,070-token vocabulary a schema
with strings of `maxLength: 200` and `maxLength: 2800` compiles in 81 ms instead of 24.8 s, and
its masks inside the strings take 0.04 ms instead of 70-96 ms (one Apple M-series core; Gemma's
262,144-token vocabulary was not measured).

Every running request must make progress: one that neither computes a prompt chunk nor accepts a
token for 120 s fails with `inference request made no progress for 120 s` instead of holding its
sequence.

## Images and video

`--vision` loads the artifact's vision tower; the requests then take images and videos as the
[serving protocols](serving.md) and the [CLI](cli.md) accept them. Each picture, and each sampled
frame of a video, is a media unit that becomes soft tokens in the prompt, as the release's
processor lays them out:

- **Preprocessing.** A unit is resized (bicubic with antialiasing, aspect kept) to the largest size
  whose sides are multiples of 48 pixels and whose 16 x 16 patches pool 3 x 3 into at most its
  budget of soft tokens: 70, 140, 280, 560 or 1120. An image takes 280 (`"detail": "low"` 70,
  `"detail": "high"` the `--vision-max-merged` bound, which is rounded down to a budget and defaults
  to 1120); a video gives 32 frames sampled evenly over its length, 70 soft tokens each.
- **Prompt.** An image becomes `<|image>`, its soft tokens and `<image|>`; a video one such run per
  frame (its tokens `<|video|>`), each after the frame's `mm:ss` timestamp.
- **Tower.** The patch projection plus learned x and y position rows; 27 blocks (sandwich RMS norms,
  16 heads of 72 with Q and K norms, a V norm without gamma and 2-D RoPE that turns half of each
  head by the patch's column and half by its row, unscaled attention over the unit's own patches,
  GeGLU 4304); the 3 x 3 average, times sqrt(1152), standardized, RMS-normalized and projected to
  the text model's width 2816. The soft tokens replace their placeholders' embeddings unscaled.
- **Attention in the text model.** In the sliding-window blocks the soft tokens of one unit see
  each other in both directions (the release's `use_bidirectional_attention: vision`); the
  full-attention blocks stay causal. A prompt chunk never splits a unit.
- **Prefix reuse.** A unit is identified by its content and budget: a follow-up turn over the same
  pictures reuses the held prefix, pictures included, and a different picture in the same place
  does not match.

`--vision-residency` places the tower:

| Residency | Tower | Device memory | Encoding |
|---|---|---|---|
| `resident` (default) | on the device | 1.07 GiB | CUDA kernels and cuBLAS |
| `overlay` | blocks in pinned host memory, 1.07 GiB of RAM | a two-block window and the small parts, about 0.14 GiB | the same kernels; each block uploads while the one before computes |
| `cpu` | FP32 copy in host memory, 2.1 GiB of RAM | none | CPU threads (AVX2 and FMA when the CPU has them); `--vision-max-merged` defaults to 256, so a unit takes at most 140 soft tokens |

The device tower multiplies BF16 matrices by BF16 inputs into FP32 results over an FP32 residual;
the streamed tower gives the same soft tokens bit for bit. The CPU tower computes in FP32
throughout. One encode runs at a time per process.

Encoding time on an RTX 3090, the `cpu` residency on the same host's 4-core i3-12100F (AVX2); a
request's `vision` time in the request log, after the server's warm-up:

| Media | Soft tokens | `resident` | `overlay` | `cpu` |
|---|---:|---:|---:|---:|
| a 768 x 512 picture | 260 | 71 ms | 76 ms | 4.5 s (126 tokens) |
| the same, `"detail": "low"` | 60 | 15 ms | 47 ms | 2.1 s |
| the same, `"detail": "high"` | 1,080 | 410 ms | 414 ms | |
| two 640 x 384 pictures | 2 x 252 | 123 ms | 127 ms | 9.5 s (2 x 135) |
| a 640 x 360 video, 32 of its 40 frames | 32 x 66 | 411 ms | 417 ms | 76.4 s |

The overlay's uploads (the blocks' 1.02 GiB per encode) cost little beside a large unit and
dominate a small one. A follow-up turn over the same picture reuses it without encoding (274 of its
496 prompt tokens held).

## Numerical checks

- The GGUF vector and matrix products at Gemma's widths (2816, 2112 and 704, the last two no whole
  number of the matrix kernel's 256-value steps) and the routed-expert GeGLU products match an FP64
  product of the decoded blocks (`ninfer_linear_gguf_test`, `ninfer_gguf_moe_geglu_test`); the
  decode attention (both head widths, stored and derived keys, several rows of one sequence at
  consecutive positions), its q8_1 output and the decode routing match FP64 oracles
  (`ninfer_gemma4_kernels_test`); Gemma's tool-call markup parses as streamed, a forced call
  from its seeded opener included (`ninfer_gemma4_output_test`, `ninfer_qwen3_5_frontend_test`).
- The KV-cache formats (`ninfer_gemma4_kv_test`), each of the eight quantized ones: the codes and
  scales a store writes, decoded on the host by the engine's codec oracles (and turned back in FP64
  for a rotated plane), stay within the coding's error of the rows stored; a ring's load and a prompt
  chunk's stored-and-replaced rows equal that host decode; and the decode attention over the planes
  matches FP64 attention over the host-decoded rows, for a sliding block (rotated keys, turned
  queries, NVFP4 values turned back) and a full-attention block whose keys derive from its plane.
  With an artifact and `NINFER_GEMMA4_KV` set, `ninfer_gemma4_assistant_real_test` also runs the
  assistant, a prefix checkpoint's restore, a host image's round trip through pages, and the
  prompt route against the decode route past the sliding window in that format: prefilled at once,
  the decode route's greedy text gets the same greedy tokens over its last 256 positions (all of
  them in every format but `k8v4`, 255 there). That text repeats itself by then, so this checks
  that a prompt chunk reads the stored window and its own stored rows, not their precision.
- The assistant: [`tools/reference/gemma4.py`](../tools/reference/gemma4.py) `--assistant` runs
  transformers' drafting algorithm in FP64 over the decoded GGUFs. On a 40-token prompt its three
  drafts after the greedy anchor (top-2 margins 15.4, 18.0 and 12.5 logits) are NInfer's
  (`ninfer_gemma4_assistant_real_test`, given an artifact). The same test verifies the anchor and
  the drafts in one pass and decodes them one at a time on another sequence: the logits are equal
  bit for bit.
- End to end, [`tools/reference/gemma4.py`](../tools/reference/gemma4.py), an FP64 NumPy forward
  of the model over the exactly decoded GGUF, is the oracle. NInfer agrees with it within the
  activation-quantization noise llama.cpp shows against the same oracle: over a 147-token WikiText
  window NInfer's total NLL is 1229.1 nats against the oracle's 1220.5, and on the last 64 targets
  of a fresh 256-token window llama.cpp gives 557.9 against the oracle's 555.7. Block by
  block, both engines drift from the oracle alike (about 10% relative error in the last blocks'
  hidden states for a 26-token prompt), the cost of 8-bit activations in a model this deep.
- The vision tower: [`tools/reference/gemma4_vision.py`](../tools/reference/gemma4_vision.py), an
  FP64 NumPy tower over the mmproj GGUF, is the oracle. transformers' own `Gemma4VisionModel` and
  embedder with the BF16 release's weights agree with it block by block within FP64 rounding (fed
  the same input, a relative error below 1e-7 per block). The QAT release's mmproj carries its own
  projection into the text model, up to 0.23 away from the BF16 release's, so the comparison
  projects with the GGUF's. On a 768 x 512 picture (260 soft tokens,
  `ninfer_gemma4_vision_real_test`, given an artifact) the device tower's soft tokens lie within
  2.2% relative RMS error of the oracle's, every token's cosine at least 0.988 (transformers' tower
  in BF16: 6.9% and 0.889); the CPU tower within 0.17% and 0.999998, the error of its BF16 output;
  the streamed tower equals the resident one bit for bit. The tower's kernels and the CPU tower's
  products, softmax and GeGLU match FP64 oracles (`ninfer_gemma4_vision_kernels_test`,
  `ninfer_gemma4_vision_cpu_kernels_test`), and the BF16 projections with BF16 or FP32 results the
  FP64 product (`ninfer_linear_bf16_a16_test`).
- The preprocessing matches transformers' `Gemma4ImageProcessor` on the same picture: the same
  patch grid and soft-token count, pixel values within 2.2 of 255 (mean 0.08), the two bicubic
  resamplers' difference.
- Compare perplexity with llama.cpp on whole fresh windows only: `llama-perplexity
  --ppl-stride` on this model does not reproduce its own fresh-window result for the windows after
  the first (a window scored alone gives 487.6 nats where the strided run reports 401.3), so its
  strided totals are not a reference here.

## Expert residency

`--expert-residency device` keeps the routed GGUF banks on the GPU. `host` keeps them in pinned host
memory and `disk` in the artifact's files, read through the OS page cache. Both run all expert
arithmetic on the GPU with the resident kernels and outputs: top-8 GeGLU, the 704-wide middle, the
dense branch and the sandwich norms. A 12 GB card runs the model this way; the GPU holds 1.7 GiB of
other weights instead of 13.7 GiB.

```sh
ninfer /models/gemma4.ninfer --expert-residency host
ninfer /models/gemma4.ninfer --expert-residency disk --expert-cache-mib 0
```

Experts reach the GPU through the shared transport (`core/expert_delivery`):

- **Layer banks.** A prompt chunk's experts missing from the cache (and decode's, without a cache)
  land in a whole-layer bank laid out like the resident bank, each expert at its index, so
  consecutive experts arrive in one copy. Two banks let the next layer of a prompt chunk arrive
  while the current one computes (one bank where memory is short). Each prompt chunk needs nearly
  every expert of every layer, so the expert bank crosses the bus about once per chunk; prompts take
  the resident chunks of up to 2048 tokens. Between prompts the banks serve as cache slots: once two
  passes have run without a prompt chunk, a decode or verification miss that the cache does not
  admit enters a free slot of a lent bank, else the slot of its least routed expert the call does
  not route, and stays there for later calls; the next prompt chunk reads those slots as misses and
  takes back the bank it fills.
- **Cache.** `--expert-cache-mib` (default: what is free after startup) holds the experts decode and
  verification route most, by the expert tier every MoE family with host or disk experts shares
  ([the expert cache](qwen3-8-flash-next.md#the-expert-cache)). Its slots form a pool for each size
  of expert: the layers whose experts need the same room share one, each layer about as many slots
  as any other, so a model of mixed formats spends no slot bytes on room its experts lack. A miss
  the GPU copies takes a free slot of its pool, or the slot of the coldest expert of that pool the
  call does not route when its routes beat that expert's, and every four decode passes a round
  swaps the most routed missing experts in, copied behind the calls. A prompt chunk evicts nothing;
  it promotes its most routed experts from the bank into free slots by device copies, up to each
  layer's share of its pool, so a long prompt does not flush decode's working set. A routing profile
  (`--expert-profile`, else the built-in one of the model family where the build has one) fills the
  slots at startup. `0` keeps the banks only.
- **CPU share.** Of a decode or verification call's misses that neither the cache nor the layer's
  bank holds, CPU threads compute a share from the stored blocks while the GPU copies and runs the
  rest (`--expert-dma-share`, `auto` by default: the split whose copies and CPU products end
  soonest by earlier calls' times; `--expert-cpu-threads`, one pinned thread per physical core by
  default, at most half the process's CPUs). The CPU computes GeGLU over the [gate; up] rows,
  the BF16 middle quantized as q8_1 and the down rows as the GPU's vector kernels do, into the same
  2^-32 fixed-point sums ([the CPU share](qwen3-8-flash-next.md#the-cpu-share) describes the
  arithmetic), so outputs do not depend on the split. Q2_0, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K,
  IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS blocks have CPU products; other layers copy every miss.
  An expert the CPU computed takes no slot; the cache's rounds decide whether it enters.
- **Calls.** Vector decode and verification run the cached pairs while the rest arrive, then the
  rest; both add to one fixed-point sum, so outputs equal resident execution bit for bit. Prompt
  chunks wait for their experts and run the existing matrix path.
- **Growing KV.** With an automatic cache, the full-attention blocks' value planes and a prompt
  chunk's derived keys, F16 values and mask are backed only as far as their sequences reach (in
  256-position tiles, on devices with CUDA virtual memory management), and the cache takes the rest
  of the device. When the next positions need memory the device lacks beside its 320 MiB graph
  reserve, the cache gives slots back: it keeps its most routed experts, moved by device copies into
  the slots that stay, and the others leave. A reset sequence's rows go back to the device, and the
  cache takes what is free beyond 256 MiB of headroom for the next positions. Startup still checks
  that every sequence's whole context fits beside the layer banks, so a context never runs out of
  memory: the cache can shrink to no slots. An explicit `--expert-cache-mib` and device experts keep
  the whole KV allocated.

On an RTX 4090 (450 W, PCIe 4.0 x16, CUDA 12.8) with the QAT artifact, host experts, four sequences
of 131,072 positions and three drafts a round, 2026-10-20: commit `5ed86dff9`, which reserves every
sequence's KV (7,410 MiB), cached 3,251 of the 3,840 experts; with the growing KV the cache holds
all 3,840 and 5.4 GiB of the device stays free at startup (a 59-thread CPU pool, before the pool
took half the CPUs). Short answers decoded at 295-413 tok/s before and 288-410 after, and after a
107,074-token prompt at 181 and 180 tok/s, since both caches serve nearly every route. The answers
equal those of an explicit 4,096 MiB cache, whose KV stays whole, token for token with equal log
probabilities. Against `5ed86dff9` they part after 45 to 102 tokens or not at all: the GPU's vector
kernels now round each step of Q4_0's slice products as the CPU does.

Lending the banks needs a cache short of the experts decode routes. On an RTX 4090 with one
sequence of 131,072 positions (2026-10-20, every `--kv-dtype`), the cache held all 3,840 experts
and served every route with or without lent banks, which added 254 slots that nothing used: decode
ran at 0.994-1.004 times `c04ea310`'s speed over 45 measurements (two short prompts twice and a
107,074-token document per format), with the same answers and log probabilities. Cards whose cache
holds fewer experts were not measured.

With host or disk experts and an artifact that carries the assistant, the assistant drafts three
tokens a round unless `--spec` says otherwise ([speculative decoding](#speculative-decoding-with-the-assistant)).
The loader keeps the stored GGUF blocks; nothing is repacked. Startup diagnostics report the cache,
the layer banks, metadata and pinned buffers; runtime expert counters start at zero after warm-up.
Disk residency uses two bounded pinned transfer halves and no application RAM cache. Flash-Next's
`--expert-misses`, `--expert-tail-skip`, `--expert-prefill-buffers` and `--expert-cache-adaptive`
do not apply. Gemma model suspend is unsupported, as with resident experts.
