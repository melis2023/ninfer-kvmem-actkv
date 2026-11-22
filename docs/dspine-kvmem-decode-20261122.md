# DSpine on KVMem — the compact-frontier bug, and where the remaining time goes (2026-11-22)

Scope: branch `exp/dspine`, backend `ninfer`, target `radixark_daisyrose-27b-iq3_xs-dflash2.ninfer`,
standalone drafters `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf` and `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
loaded through `--draft-model`, 16 GB RTX 5060 Ti.

This document records the fix for the acceptance collapse that appeared once a request outgrew
the KVMem device window, and the separate per-round cost that remains afterwards.

## 1. Symptom

DSpine drafted well until the context crossed `--kvmem-budget + --kvmem-gen-reserve`, then the
acceptance fell to 0.3–0.5 % and the decode rate locked at ~21 tok/s. The cliff was sharp and
landed exactly on the window boundary, not gradually:

| prompt | acceptance | decode |
|---|---|---|
| 5,900 | 78.6 % | 85.4 |
| 8,910 | 88.9 % | 94.0 |
| 10,705 | 0.4 % | 21.6 |
| 12,649 | 0.0 % | 21.3 |
| 14,105 | 0.4 % | 21.6 |
| 19,753 | 0.3 % | 21.0 |

A control run of DFlash2 on the same target, the same KVMem window and the same 14,105-token
cold prompt accepted 44.3 % at 56.0 tok/s, so the target, the KVMem configuration and the device
budget were all sound. The fault was DSpine's own.

## 2. Root cause

`propose_dspine_batch()` handed the drafter `frame.execution_frontiers` as the block frontier:

```cpp
state.dflash.dspine->propose(
    {.anchors   = frame.anchors.slice(0, 0, batch),
     .frontiers = frame.execution_frontiers.slice(0, 0, batch),   // compact
```

`execution_frontiers` is the **compact** KV address (`frontier - removed_tokens`). The drafter
uses this frontier for two things:

* the RoPE position of every proposal row (`prepare_masked_block`), and
* the head of the cyclic-ring scan, `[frontier - window, frontier)`.

The ring itself is written at absolute positions, and `draft_frontiers` (also absolute) was
already carried in the ingress next to it. So while no reselection had run, compact and absolute
coincide and everything worked; the first Pressure rebased the compact address, and from then on
the proposal block was rotated by `removed_tokens` positions and the scan read a window that
ended `removed_tokens` tokens behind the present. At 119k that is a ~10^5-token offset.

This is the same defect upstream fixed for the in-artifact drafter in `b32b68d4e` ("the proposal
block took its RoPE frontier from `execution_frontiers`"). It came back when DSpine was re-ported
as a standalone drafter module: `propose_dflash2_batch()` kept `frame.draft_frontiers`, the new
`propose_dspine_batch()` did not.

### Fix

```cpp
    {.anchors   = frame.anchors.slice(0, 0, batch),
     // The block's RoPE frontier and the ring scan start from the model's logical
     // position: KVMem rebases the compact KV address on every reselection, so an
     // execution frontier here rotates the whole block once the context crosses B/R.
     .frontiers = frame.draft_frontiers.slice(0, 0, batch),
```

Three consistency fixes ride along, all in the same "the ring lives in absolute positions"
direction:

* `execution/text.cpp`, `execution/text.h` — a new `dspine_draft()` predicate makes the prefill
  and verify feature taps publish **logical** positions instead of the compact cache positions,
  so the ring is written in the same space the proposal scans.
* `execution/draft.cpp` — the acceptance sampling key (`speculative_accept_coupled_drafts`'s
  `frontiers`, keyed as `(seed, length + i + 1, decode)`) also uses `draft_frontiers` for
  DSpine; a compact length draws a different token than the draft was drawn from.
* `program/decode.cpp` — `enqueue_dflash_context_append()` zeroes the ingress and then filled
  only `context_frontiers`/`execution_frontiers`, while the native append reads
  `draft_contexts`/`draft_frontiers`. The catch-up therefore appended nothing at all (span 0)
  and each skipped prefix stayed a hole in the ring.

## 3. Acceptance after the fix

Cold prompts, `--kvmem-budget 9216 --kvmem-gen-reserve 1024`, bf16 ring:

| prompt | acceptance | decode |
|---|---|---|
| 8,910 | 88.9 % | 94.0 |
| 14,105 | 76.7 % | 82.3 |
| 19,753 | 83.3 % | 86.5 |
| 119,142 | 65.6 % | 21.0 |

The cliff is gone: acceptance now decays smoothly with context instead of collapsing at the
window boundary.

## 4. 128k comparison, measured at the steady state

An earlier reading of 73.0 tok/s at 119k was a **short-sample artifact**: the first round after
prefill still fits the reserve, so a 58-token generation finished inside the cheap window. Server
throughput lines showed the steady state is entered after a few rounds. Every number below comes
from the same protocol — cold 119,142-token prompt, `max_tokens 256`, `temperature 0`, the same
KVMem window (`9216 + 1024`), `--kv-dtype int8`, `--no-cuda-graph`.

| backend | `--draft-tokens` | acceptance | TTFT | decode |
|---|---|---|---|---|
| DSpine, bf16 ring | 4 | 65.6 % (80/122) | 2m 17.0s | **21.0** |
| DSpine, int8 ring | 4 | 65.6 % (80/122) | 2m 09.8s | **20.7** |
| DSpine, bf16 ring | 15 | 29.2 % (57/195) | 1m 58.8s | **56.5** |
| DFlash2, int8 draft KV | 4 | 51.0 % (77/151) | 1m 59.1s | **57.8** |

Prefill holds 870–1010 tok/s everywhere; the difference is entirely in the decode loop.

Two readings:

* **The ring dtype is not the cost.** bf16 and int8 rings land within noise (21.0 vs 20.7), so
  the scan is not bandwidth-bound on the ring.
* **DSpine pays a large fixed cost per round.** Round cost from the table: 174 ms at `k=4`
  (≈3.6 committed tokens per round) and ≈108 ms at `k=15` (≈5.4 tokens per round). Fitting
  `fixed + c·k` gives ≈150 ms fixed and ≈6 ms per draft column. DFlash2, which shares the same
  target verify and the same per-round KVMem reselection, spends ≈55 ms per round in total — so
  the fixed 150 ms is DSpine's proposal, not the harness.

Because the cost is fixed rather than token-linear, widening the draft window amortises it:
`--draft-tokens 15` reaches 56.5 tok/s at 128k even though acceptance drops to 29.2 %. That is
within 2 % of DFlash2's 57.8 tok/s at the same window.

## 5. Operating points

| context | setting | acceptance | decode |
|---|---|---|---|
| under `B+R` (< 10k) | `--draft-tokens 4` | 84–89 % | 82–94 tok/s |
| 128k | `--draft-tokens 15` | 29 % | 56.5 tok/s |
| 128k | `--draft-tokens 4` | 66 % | 21.0 tok/s |

`--draft-tokens 4` stays the right choice inside the window; past it, `--draft-tokens 15` is worth
roughly 2.7x. Both avoid the 0.3–0.5 % collapse of the pre-fix binary.

## 6. Reproduction

```
ninfer-serve.exe radixark_daisyrose-27b-iq3_xs-dflash2.ninfer ^
  --port 18240 --max-context 131072 ^
  --kvmem-budget 9216 --kvmem-gen-reserve 1024 --kvmem-host-mib 8192 --kvmem-lrfu ^
  --no-cuda-graph --kv-dtype int8 --draft-kv-dtype bf16 ^
  --spec dspine --draft-tokens 15 ^
  --draft-model Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf
```

Note: the DFlash2 launcher's `--kvmem-budget 16384 --kvmem-gen-reserve 2048` no longer starts on
this 16 GB card (short by ~21 MB of engine runtime reservation); the comparison above therefore
uses DFlash2 at the same `9216 + 1024` window as DSpine.

## 7. Open item

The ~150 ms fixed per-round cost of the DSpine proposal is unexplained. It is not the ring read
(dtype-independent), not the target verify or KVMem (DFlash2 shares both), and not the draft
count (it fits `fixed + c·k`). Candidates to measure next: a per-round device/host round-trip in
the proposal path, and the 8192-row scan route, which is outside the registered
`sliding_window_attention` profiles (2048/4096) and may be running the generic kernel.
