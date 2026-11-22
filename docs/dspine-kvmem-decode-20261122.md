# DSpine on KVMem — the compact-frontier bug, and how to benchmark it (2026-11-22)

Scope: branch `exp/dspine`, backend `ninfer`, target `radixark_daisyrose-27b-iq3_xs-dflash2.ninfer`,
standalone drafters `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` and `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
loaded through `--draft-model`, 16 GB RTX 5060 Ti.

This document records the fix for the acceptance collapse that appeared once a request outgrew the
KVMem device window, the 128k comparison against DFlash2 afterwards, and the measurement trap that
made an earlier reading of this comparison wrong.

## 1. Symptom

DSpine drafted well until the context crossed `--kvmem-budget + --kvmem-gen-reserve`, then the
acceptance fell to 0.3–0.5 % and the decode rate collapsed. The cliff was sharp and landed exactly
on the window boundary, not gradually:

| prompt | acceptance | decode |
|---|---|---|
| 5,900 | 78.6 % | 85.4 tok/s |
| 8,910 | 88.9 % | 94.0 tok/s |
| 10,705 | 0.4 % | 21.6 tok/s |
| 12,649 | 0.0 % | 21.3 tok/s |
| 14,105 | 0.4 % | 21.6 tok/s |
| 19,753 | 0.3 % | 21.0 tok/s |

A control run of DFlash2 on the same target, the same KVMem window and the same 14,105-token cold
prompt accepted 44.3 % at 56.0 tok/s, so the target, the KVMem configuration and the device budget
were all sound. The fault was DSpine's own.

## 2. Root cause

`propose_dspine_batch()` handed the drafter `frame.execution_frontiers` as the block frontier:

```cpp
state.dflash.dspine->propose(
    {.anchors   = frame.anchors.slice(0, 0, batch),
     .frontiers = frame.execution_frontiers.slice(0, 0, batch),   // compact
```

`execution_frontiers` is the **compact** KV address (`frontier - removed_tokens`). The drafter uses
this frontier for two things:

* the RoPE position of every proposal row (`prepare_masked_block`), and
* the head of the cyclic-ring scan, `[frontier - window, frontier)`.

The ring itself is written at absolute positions, and `draft_frontiers` (also absolute) was already
carried in the ingress next to it. So while no reselection had run, compact and absolute coincide
and everything worked; the first Pressure rebased the compact address, and from then on the
proposal block was rotated by `removed_tokens` positions and the scan read a window that ended
`removed_tokens` tokens behind the present. At 119k that is a ~10^5-token offset.

This is the same defect upstream fixed for the in-artifact drafter in `b32b68d4e` ("the proposal
block took its RoPE frontier from `execution_frontiers`"). It came back when DSpine was re-ported as
a standalone drafter module: `propose_dflash2_batch()` kept `frame.draft_frontiers`, the new
`propose_dspine_batch()` did not.

### Fix

```cpp
    {.anchors = frame.anchors.slice(0, 0, batch),
     // The block's RoPE frontier and the ring scan start from the model's logical
     // position: KVMem rebases the compact KV address on every reselection, so an
     // execution frontier here rotates the whole block once the context crosses B/R.
     .frontiers     = frame.draft_frontiers.slice(0, 0, batch),
```

Three consistency fixes ride along, all in the same "the ring lives in absolute positions"
direction:

* `execution/text.cpp`, `execution/text.h` — a new `dspine_draft()` predicate makes the prefill and
  verify feature taps publish **logical** positions instead of the compact cache positions, so the
  ring is written in the same space the proposal scans.
* `execution/draft.cpp` — the acceptance sampling key (`speculative_accept_coupled_drafts`'s
  `frontiers`, keyed as `(seed, length + i + 1, decode)`) also uses `draft_frontiers` for DSpine; a
  compact length draws a different token than the draft was drawn from.
* `program/decode.cpp` — `enqueue_dflash_context_append()` zeroes the ingress and then filled only
  `context_frontiers`/`execution_frontiers`, while the native append reads
  `draft_contexts`/`draft_frontiers`. The catch-up therefore appended nothing (span 0). A probe
  over a full 119k request showed this path is not reached at all in the current scheduler, so the
  change is correctness-only and costs nothing.

## 3. Acceptance after the fix

Cold prompts, `--kvmem-budget 9216 --kvmem-gen-reserve 1024`, bf16 ring:

| prompt | acceptance | decode |
|---|---|---|
| 8,910 | 88.9 % | 94.0 tok/s |
| 14,105 | 76.7 % | 82.3 tok/s |
| 19,753 | 83.3 % | 86.5 tok/s |
| 119,142 | 65.6 % | 67.2 tok/s |

The cliff is gone: acceptance decays smoothly with context instead of collapsing at the window
boundary.

## 4. 128k comparison

Protocol: cold 119,142-token prompt (`p128k.txt`), `max_tokens 128`, `temperature 0`, and the same
KVMem window for both backends (`--kvmem-budget 9216 --kvmem-gen-reserve 1024 --kvmem-host-mib
8192 --kvmem-lrfu --kv-dtype int8 --no-cuda-graph`), one server per row, freshly started.

| backend | `--draft-tokens` | acceptance | cold decode | cached decode |
|---|---|---|---|---|
| **DSpine** | **4** | **65.6 %** (80/122) | **67.2 tok/s** | **75.4–75.6 tok/s** |
| DSpine | 15 | 29.2 % (57/195) | 57.6 tok/s | 67.2 tok/s |
| DFlash2 | 4 | 51.0 % (77/151) | 57.9 tok/s | 63.5 tok/s |

DSpine is now 16 % faster than DFlash2 on a cold 128k request and 19 % faster on a cached one, at a
higher acceptance (65.6 % vs 51.0 %). Prefill is 870–1020 tok/s for all three. `--draft-tokens 4`
stays the right setting; the wider block lowers acceptance faster than it saves rounds.

Note that the two backends were also run at the DFlash2 launcher's own budget
(`16384 + 2048`), which no longer starts on this card — it is short by ~21 MB of engine runtime
reservation. The table therefore compares both at the same `9216 + 1024` window.

## 5. The measurement trap (read this before benchmarking this GPU)

An earlier pass of this document reported 21 tok/s for DSpine at 128k and 57.8 for DFlash2, and
inferred a ~150 ms fixed per-round cost in the DSpine proposal. **That conclusion was wrong**, and
the cause was environmental, not architectural:

* the server occupies ~15.7 GiB of a 16.0 GiB card (`nvidia-smi` reported 15,765 / 16,311 MiB with
  a request in flight);
* when the desktop or another application grows by a few hundred MiB, the driver starts paging our
  allocations into shared memory and decode drops by ~3x;
* three consecutive runs measured 20.7–21.9 tok/s; the identical binary and flags measured 67.2
  cold / 75.5 cached twenty minutes later.

Symptoms of a depressed run: acceptance stays correct while decode sits near 21 tok/s, and the
server's own `throughput` line shows the same number for the whole generation. Always record
`nvidia-smi --query-gpu=memory.used` next to a decode number on this card, and repeat a suspicious
reading before drawing conclusions from it.

For the record, the abbreviated lifespan of the wrong theory: round cost fit `fixed + c·k` only
after mixing runs from different memory states, and the catch-up probe above (added to test it)
fired zero times in 110 decode rounds.

## 6. Reference implementation

`I:\llama\llama-dspine-ref` is `Ajay9o9/llama.cpp@dspine` (single commit, "dflash : add DSpine
draft model support"); DSpine lives in `src/models/dflash.cpp`. Three differences are worth
recording, none of which are needed for correctness here:

* **Candidate ids come for free.** `build_dspine_lattice()` calls `ggml_top_k(res->t_logits,
  top_k)`, reusing the target logits tensor the same graph already computed.
* **The head is gathered, not projected.** `head = ggml_get_rows(output, cand_run)` takes only the
  `top_k` rows of the output weight, then `table = ggml_mul_mat(head, refined)` builds the
  `top_k × top_k` transition table. There is no full-vocabulary matmul per round. Our
  `Drafter::propose()` has a `target_.head_topk` fast path but falls back to a full
  `target_.head(hidden, [vocab, columns])` when the fused head is unavailable; at 248,320 rows that
  fallback is the deviation from the reference, though it does not dominate (it is context
  independent, and the measured round cost is dominated by whatever the memory state allows).
* **One fused attention.** The block reads its window ring through `build_attn(inp_attn_k_iswa,
  ...)` with the SWA mask and `n_swa` from the model, the same kernel the target uses. Our
  equivalent is `ops::sliding_window_attention`, which resolves to the split-KV route with 32
  splits over an 8191-row window, bounded by `window - 1` rather than by the envelope — so the
  128k `max_context` does not widen the scan.

## 7. Operating points

| context | setting | acceptance | decode |
|---|---|---|---|
| under `B+R` (< 10k) | `--draft-tokens 4` | 84–89 % | 82–94 tok/s |
| 128k, cold | `--draft-tokens 4` | 66 % | 67 tok/s |
| 128k, cached prefix | `--draft-tokens 4` | 66 % | 75 tok/s |

## 8. Reproduction

```
ninfer-serve.exe radixark_daisyrose-27b-iq3_xs-dflash2.ninfer ^
  --port 18240 --max-context 131072 ^
  --kvmem-budget 9216 --kvmem-gen-reserve 1024 --kvmem-host-mib 8192 --kvmem-lrfu ^
  --no-cuda-graph --kv-dtype int8 --draft-kv-dtype bf16 ^
  --spec dspine --draft-tokens 4 ^
  --draft-model Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf
```

`workers/ninfer/start-128k-dspine.bat` runs exactly this line.
