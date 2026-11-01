# Launcher profiles

`scripts/run.sh` and `scripts/run.bat` serve one of the official artifacts that
`scripts/download-model.{sh,bat}` fetches, on one 24 GB card:

```
run.sh <model> [profile]
```

| model | profiles | default profile on Linux | on Windows |
|---|---|---|---|
| `qwen38-27b` | `tuned`, `int8`, `c8` | DFlash2, one lane, 262,144 tokens | DFlash2, one lane, 172,032 tokens |
| `qwen36-35b-a3b` | `tuned` | MTP3, three lanes, 262,144-token pool | MTP3, two lanes, 262,144-token pool |

The Docker image's `run <model> [profile]` command runs `run.sh` inside the container
([README](../../README.md#docker)). Started that way with driver 580.159.03, `run qwen38-27b` took
the full 262,144-token context on an RTX 5090 and 196,608 on an RTX 3090 after two of the
launcher's step-downs.

Lanes share one KV pool: `--kv-capacity` is the pool and `--max-context` the per-request cap, and
the `tuned` profiles set both to the same value. Any one request can use the whole context, but the
lanes' requests together hold at most that many tokens at a time.

## `tuned`

The recommended profile. Every `tuned` profile serves `rk4v4` KV, `--gdn-state-fp16`, Vision in
overlay residency (about 10 MiB of runtime reservation; `NINFER_VISION=off` drops it), and a
context cache sized for several rotating conversations: `--max-private-continuations 8
--max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 --auto-prefix-grid`.

The Qwen3.8-27B profile has three speculative variants, selected with `NINFER_SPEC`; each carries
the context, lanes and prefill chunk that fit it:

| `NINFER_SPEC` | flags beside the common ones | context (Linux / Windows) | lanes |
|---|---|---|---|
| `dflash2` (default) | `--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096 --embedding-q4` | 262,144 / 172,032 | 1 |
| `mtp` | `--spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --prefill-chunk 2048 --embedding-q4 --lm-head-q6` | 262,144 | 2 |
| `none` | the `mtp` set without speculation or `--lm-head-q6` | 262,144 | 2 |

DFlash2 is the fastest at one stream; its draft weights and its refusal of `--lm-head-q6` cost the
room a second lane would use, so the full context with two lanes is the `mtp` variant. The
Qwen3.6-35B-A3B profile runs `--spec mtp --draft-tokens 3 --lm-head-draft --mtp-experts-q4` with a
512-token prefill chunk.

Why these flags (costs from [quality trades](quality-trade-experiments.md)):

- `rk4v4` KV is 31% smaller than `rk8v4` at the same decode speed, for +0.10% perplexity over it.
  On the 27B the runtime reservation grows by 27.07 KiB per token of context with `rk8v4`, which
  is why the 27B's full 262,144-token window needs the 4-bit keys on a 24 GB card; the 35B-A3B
  stores about a third of the 27B's KV per token.
- `--embedding-q4` frees 644 MiB of weights and `--gdn-state-fp16` halves every StateImage
  (147 MiB to 74.5 MiB on the 27B), with no measurable perplexity change.
- `--lm-head-q6` frees another 341 MiB for +0.01% perplexity; DFlash and DFlash2 refuse it, so
  only the `mtp` variant passes it.
- `--prefill-cublas` prefills 43-83% faster for +0.156% perplexity; its workspace (543 MiB
  at chunk 2048, 661 MiB at 4096) is why the `mtp` variant keeps the smaller chunk.
- `--mtp-experts-q4` stores the 35B-A3B MTP layer's routed experts (816 MiB as shipped) in the text
  layers' formats; drafts are verified by the target, so it changes no scored output.
- A StateImage cannot be shared between two frontiers, so the shared-prefix catalog has to hold
  every preamble in play: eight distinct preambles reused 8.4% of their prompts with four shared
  slots and 98.3% with eight, and four agents rotating through one lane reused 12% with the engine
  default of two retained conversations and 76% with room for all four. The host slots pin host
  memory, not device memory.
- `--auto-prefix-grid` lets two callers whose prompts merely start alike share a cached prefix
  without a client hint; a grid point is materialized only once two callers have asked for it.

MTP accepts `NINFER_DRAFT_TOKENS` up to 15. Three suits chat and prose; output that reproduces the
prompt (editing and returning a file) decodes up to 1.85 times as fast with 11 to 15, and newly
written code about 10% faster with seven ([choosing a draft count](../cli.md#choosing-a-draft-count)).
Eight and more drafts add a second CUDA Graph topology class on the 27B, about 64 MiB per lane.

**When the card is busy.** A desktop or another job holding device memory can leave too little for
the default context, and on Windows WDDM charges pinned host memory against the card as well. When
the `tuned` profile is refused at startup for either reason (the engine's `runtime reservation
requires` or `cudaMallocHost failed`), the launcher steps down: an eighth of the context at a time,
at most five times, from the second step with the prefill chunk capped at 2048 and the host state
slots halved. `--kv-capacity auto` does not help here, because the engine still reserves room for
one full `--max-context` sequence. Values the caller sets (`NINFER_CONTEXT`, `NINFER_PREFILL_CHUNK`,
`NINFER_HOST_STATE_SLOTS`, `NINFER_KV_CAPACITY`) are never second-guessed, `NINFER_FALLBACK=off`
disables the step-down, and the `int8` and `c8` profiles never step down.

`--host-kv-mib 8192` pins 8 GiB of host RAM on Linux. On Windows the runtime clamps a pinned host
allocation to (free device memory - 1 GiB) / 2 before the first `cudaMallocHost`, since one failed
pin poisons later attempts in the process; prefix reuse falls back to device pages when it gets
nothing.

## `int8` and `c8`

The Qwen3.8-27B reference profiles, with every serving flag fixed: `int8` is one request at 65,536
tokens of INT8 KV with MTP3, `c8` eight lanes at 8,192 tokens each over a 16,384-token pool with a
context cache sized per lane (two retained conversations and two host StateImages per lane, 2.3 GiB
of pinned host memory).

## Overrides

From the environment, so a launcher never needs editing. Every profile reads `NINFER_MODEL`
(artifact path), `NINFER_MODEL_DIR`, `NINFER_SERVER`, `NINFER_HOST`, `NINFER_PORT`,
`NINFER_CHAT_TEMPLATE`, `NINFER_GRAFT_DIR`, `NINFER_GRAFTS` and `NINFER_DEFAULT_GRAFT` (see
[prompt grafts](../serving.md#prompt-grafts)). `tuned` also reads `NINFER_CONTEXT`,
`NINFER_CONCURRENCY`, `NINFER_KV_CAPACITY`, `NINFER_KV_DTYPE`, `NINFER_SPEC`,
`NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`, `NINFER_VISION`, `NINFER_VISION_RESIDENCY`,
`NINFER_HOST_STATE_SLOTS` and `NINFER_FALLBACK`. The launchers bind `127.0.0.1`: `0.0.0.0`
publishes an unauthenticated endpoint to every network the machine is on, so it is opt-in per run.
