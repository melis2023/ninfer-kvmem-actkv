# DSpine draft port — implementation and validation (2026-11-11)

Scope: branch `exp/dspine`, backend `ninfer`, model `Qwen3.8-27B-DSpine-b16` draft head
(`I:\model\Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`) merged into the IQ3_S + MTP ninfer
artifact as the `dflash` component. DSpine reuses `--spec dflash`; there is no
`--spec dspine` switch.

Reference material: the HuggingFace model card, upstream `Ajay9o9/llama.cpp@dspine`
(`dflash.cpp` / `common/speculative.cpp`), and the official MLX reference
`I:\model\DSpine-code\dspine_mlx` (`bridge.py`, `propose.py`, `target.py`).

## 1. What DSpine adds over DFlash

A five-layer all-sliding drafter with `block_size=16`: one anchor plus 15 draft
tokens drafted in parallel, conditioned on hidden states tapped from target layer
*inputs* `[2,17,32,47,62]` (the HF checkpoint's `[1,16,31,46,61]` are layer *output*
ids and must be shifted by `+1` unless the GGUF marks them as inputs).

Three ingredients, all now implemented:

* **Causal block attention** — `dspine_causal_attention()`, with an 8192-token
  cyclic ring and a KVMem floor mask.
* **Code bridge after every layer** — RMS normalise `hidden+residual`, project to a
  1024-wide code, normalise with `sqrt(C)` and `(||projected|| + bridge.eps)`
  (NOT `sqrt(sum + eps)`), then
  `message(t) = sigmoid(Wg c_t + Wp c_(t-1) + b) * (Wv c_(t-1))`,
  `out = raw + rms * write(message)`, followed by the next layer's RMSNorm.
  Only the anchor uses `codebook[anchor]`; other rows use the projected continuous
  code. A zero `out.weight` is correct: official `bridge.py::prepare()` rejects a
  non-zero terminal projection for a terminal-disabled checkpoint.
* **Serial last-write refinement** — top-16 exact-predecessor candidates produce a
  `[row, sender, candidate]` 16x16 transition table; `dspine_lattice_walk()` walks
  it (greedy or probabilistic) and publishes `candidate_ids` + `proposal_q` for the
  existing sparse verifier. This requires `ProposalHead::Full` and an unrotated
  output head; both are rejected at load time if violated.

## 2. Conversion chain

`tools/convert/dspine_gguf.py` maps the 73 draft tensors, builds the composite
bridge weights, and passes Q4_K/Q6_K blocks straight through via
`gguf_blocks.py::dspine_blocks()`. `tools/convert/recipes/dspine_gguf_q4.py` is the
override recipe. `qwen3_5.py::draft_config()` honours the boolean marker
`target_layer_ids_are_inputs`; the GGUF adapter sets it to `True` so the tap ids are
kept as `[2,17,32,47,62]`.

## 3. Runtime invariants (bugs fixed in this pass)

| Area | Failure | Fix |
| --- | --- | --- |
| `dspine_bridge.cu` | Bridge kernels read ninfer activations as row-major, transposing the whole residual stream. Acceptance was `0.0%`. | Added `cm(row, rows, column)` and converted every activation index. Ninfer activations are **column-major** (`row + rows*column`); logical `(out,in)` weight matrices keep `out*in + in_idx`. |
| `dspine_bridge.cu` | RMS divisor used `sqrt(sum + eps)`. | `sqrt(sum) + bridge.eps`, matching official `bridge.py`. |
| `draft.cpp` | `propose_batch_impl()` republished `proposal_extents = k`, overriding the host's budget-capped extent. High acceptance then tripped `DFlash batch returned invalid row metadata`. | Copy `state.host_ingress.proposal_extents` instead. |
| `config.cpp` / `draft.cpp` | Sparse frame could be read uninitialised with an optimised or rotated head; hardcoded `16` scattered around. | Load-time guards + `ops::kSparseSpeculativeCandidates`. |
| `kvmem_window.cpp` | With `--kvmem-budget 512` (8 pages) and an 8192-token DSpine ring, `core` demanded 8 recent pages plus sink/media, so `pressure_selection()` threw `mandatory pressure content exceeds the budget` and killed the worker. | Bound sink/media to `keep`, reserve `room = keep - core.size()` for recent pages, and clamp the retained floor to `keep * 64` tokens. |
| `decode.cpp` | `logical_draft` had been widened to `is_masked_draft_backend()`, so DFlash/DSpine used logical `context_frontiers`/`draft_frontiers` while prefill and the attention kernel use **compact** positions. After a reselection, new features were appended at logical slot 1408 while attention only read compact slots 0..639: acceptance collapsed to `0.4%`. | Restore `logical_draft = (backend == DFlash2)`; DFlash/DSpine stay compact. |
| `tools/multi_gpu_probe.cu` | Standalone target deliberately does not link `ninfer_core`, but `core/device.cu` calls `current_resident_memory()` → LNK2019. | Local stub returning an empty session (the truthful answer for a standalone tool). |

`NINFER_DSPINE_DUMP` / `NINFER_DSPINE_TRACE` remain available. Dump-based diagnosis
needs `--no-cuda-graph` because `dump_tensor()` synchronises the stream.

## 3a. Latency fixes (second pass)

Profiling with `NINFER_DSPINE_PROFILE=1` (requires `--no-cuda-graph`, because the
probe synchronises the stream) exposed two hot spots that made DSpine *slower*
than `--spec none` at long context despite the correctness fixes above.

| Area | Failure | Fix |
| --- | --- | --- |
| `dspine_refine_kernel` | Launched as `<<<batch, 256>>>`: at batch 1 the whole ~630M-MAC refinement (240 column dot products, a 5120×512 write, the 16-way head score) ran on **one SM with 256 threads**, so L2 latency was never hidden. It cost **334 ms per round — 80% of decode**. | One block per `(row, sender, batch)` column (`dim3(columns, batch)`, 256 threads). Columns are independent and each column's accumulation order is unchanged, so the published lattice stays bit-identical. `334 ms → 11 ms`. |
| `dspine_causal_attention_kernel` | One 128-thread block per `(head, row)` scanned the draft ring position by position, synchronising the whole block on every context position (`log2(head_dim)+1` `__syncthreads`). At 4–8K context this was 40–90 ms per round. | One **warp** per `(head, row)`: each lane holds `head_dim/32` query elements, the score reduces through `__shfl_xor_sync`, and the online-softmax state is per-lane. No `__syncthreads` at all. `4K layers 53 ms → ~15 ms`. |

Decode throughput on the same RTX 5060 Ti with
`--kvmem-budget 36480 --kvmem-gen-reserve 16384 --kv-dtype nvfp4 --kv-capacity 52864`:

| Context | Before | After | `--spec none` baseline |
| --- | --- | --- | --- |
| 13 tokens (CLI, CUDA Graph) | 15.2 tok/s | **74 tok/s** | 31 tok/s |
| 4K, natural prompt | 28.5 tok/s | **34.3 tok/s** | 30.4 tok/s |
| 8K, natural prompt | 15.9 tok/s | **30.9 tok/s** | ~30 tok/s |

Acceptance is unchanged (`36.0%` CLI, `18.6%` at 4K, `20.3%` at 8K) and
`dspine_smoke.py --check cli` still reports byte-identical output. Acceptance
drops to 8–12% when the model degenerates into blank lines or repeated
boilerplate; that is prompt content, not a kernel defect.

## 3b. Head-to-head against DFlash2 (third pass)

Section 3a still left DSpine behind DFlash2 on the same box, so both drafters were
run on the same target quant, the same `B/R = 36480/16384` partition
(`--kv-dtype nvfp4`, `W = 52864`, `--kvmem-host-mib 12288`), the same prompt and
the same 256-token output budget, with `--ngram-draft-tokens 0` so only the neural
drafter counts.

The first measurement cleared the drafter itself: at `--draft-tokens 15` the two
drafters accept essentially the same number of tokens (DSpine 185/994 = 18.6%,
DFlash2 190/936 = 20.3%) while DFlash2 was **1.76× faster**. DSpine proposed almost
identical tokens, so the deficit was pure per-round cost, not draft quality.

Per-round accounting at 4K (`decode_time / rounds`):

| | k=4 | k=15 | slope / extra row | fixed cost / round |
| --- | --- | --- | --- | --- |
| DFlash2 | 30.7 ms | 67.8 ms | 3.37 ms | 13.9 ms |
| DSpine (before 3b) | 71.3 ms | 112.5 ms | 4.06 ms | 35.8 ms |

Two defects explained the 22 ms fixed-cost gap:

| Area | Failure | Fix |
| --- | --- | --- |
| Draft attention window | The reference GGUF stores `sliding_window = 8192` (and section 5 below already assumed it), but the kernel bounded the scan with the **ring capacity**: `begin = max(floor, frontier - capacity)`. At the 52,864-slot KV window the drafter attended over 6.5× the positions it was trained on — a cost multiplier *and* a correctness defect, since those positions lie outside the draft model's window. | `dspine_causal_attention` takes an explicit `window` argument, sourced from `config.sliding_window` (the same value the DFlash2 path already passes to `sliding_window_attention`), and clamps the scan with `min(window, capacity)`. |
| Draft attention parallelism | One 32-thread block per `(head, row)` = ~200 warps *total*, i.e. about one warp per SM with nothing to hide latency behind. The context scan cost 17 ms per round at 4K and grew with context; it was independent of the number of proposal rows. | **Split-K**: `kDSpineAttentionSplits = 8` warps per `(head, row)`, each scanning its own contiguous slice of `[begin, frontier)`, then merging the partial online-softmax states in shared memory (`exp(m_s - peak)` rescaling). 256 threads per block, no extra synchronisation in the inner loop. |

Effect at 4K, `--draft-tokens 4` (the profile probe needs `--no-cuda-graph`):

| Segment | Before 3b | After 3b |
| --- | --- | --- |
| `layers` (5 layers + causal attention) | 25.9 ms | **11.6 ms** |
| `head` (full-vocab top-16) | 1.8 ms | 1.8 ms |
| `refine` (16×16 lattice) | 3.8 ms | 3.8 ms |
| draft total | 31.5 ms | **17.2 ms** |

`--draft-tokens 15` shows the same shape (`layers 31.5 → 20.8 ms`, draft total
`44.6 → 34.0 ms`), and the scan is now only ~2.8 ms of `layers` at 4K.

### Throughput, same conditions, `--draft-tokens 4`

| Context | DSpine | DFlash2 | ratio | acceptance DSpine / DFlash2 |
| --- | --- | --- | --- | --- |
| 4K | 58.8 tok/s | 78.7 tok/s | 0.75 | 57.1% / 54.2% |
| 8K | 51.2 tok/s | 72.5 tok/s | 0.71 | 51.4% / 54.2% |
| 32K | 50.2 tok/s | 69.4 tok/s | 0.72 | 52.9% / 54.9% |
| 122K | 45.5 tok/s | 62.3 tok/s | 0.73 | 46.1% / 47.9% |

The ratio is flat across context lengths, which is the intended outcome of the
window fix: DSpine now *improves* with context (4K 58.8 → 122K 45.5) instead of
collapsing, because the draft scan no longer grows past 8K.

### k sweep (4K, 256 tokens)

| k | decode | acceptance | accepted tokens |
| --- | --- | --- | --- |
| 15 | 38.9 | 19.0% | 186 |
| 7 | 51.2 | 41.5% | 189 |
| 4 | **58.8** | 57.1% | 177 |

Note the accepted-token column: a 16-row block only buys ~2.8 tokens per round,
the same as a 5-row block buys ~2.2. Past the first two or three accepted tokens
the remaining proposals almost never land, so the extra verify rows and the wider
refine lattice are pure cost. `start-kvmem-ninfer-dspine.bat` therefore defaults to
`--draft-tokens 4` (DFlash2's own worker also uses 4).

### What is left

> Superseded by section 3c: most of what looked structural here (the bridge and refine
> plumbing) turned out to be uncoalesced memory access and is now 5–10× faster.

Both drafters now have the same acceptance, so the residual ratio is the
drafter's fixed per-round cost:

* DSpine must run the **full-vocabulary head** every round to build the top-16
  refinement lattice (`--full-draft-head` is mandatory), while DFlash2 stores a
  sparse proposal head in the artifact and can use `--lm-head-draft`.
* The DSpine draft weights ship as Q4_K_M (`dspine-q4k`, 14.06 GB artifact) versus
  DFlash2's Q3_K_M (`...q3km-mtp-dflash2`, 13.72 GB), so its 5-layer forward
  streams more bytes per round. The `layers` segment barely changes with the number
  of proposal rows, i.e. it is weight traffic bound, not FLOP bound.
* DSpine also carries `proposal_q` for rejection sampling and the lattice walk,
  which DFlash2's sparse frame does not need.

Closing the remaining ~25% would need a lighter draft quant or fewer draft layers —
a model-side change, not a kernel fix.

## 3c. Bridge projections and refine were memory-bound (fourth pass)

Section 3b left DSpine at 0.71–0.75 of DFlash2 and attributed the rest to "structural"
cost. That was wrong. Nsight Systems (`nsys profile --trace=cuda`) plus per-kernel
instance counts showed the draft round at k=4 was:

| Kernel | Per call | Calls / round | Total / round |
| --- | --- | --- | --- |
| `dspine_bridge_read_proj_kernel` | 507 µs | 5 | 2.54 ms |
| `dspine_bridge_linear_kernel` | 104 µs | 15 | 1.56 ms |
| `dspine_bridge_write_kernel` | 146 µs | 5 | 0.73 ms |
| `dspine_refine_kernel` | 3.81 ms | 1 | 3.81 ms |

8.6 ms of a 14.4 ms draft — and every one of them was arithmetic-trivial: a
512×5120 matvec is ~15 µs of HBM traffic, not 500 µs.

Root cause, all four kernels: `index = row + rows * col` with `row = index % rows`
put adjacent *threads* on adjacent *output rows* while the inner loop walked the
reduction axis, so `weight[row * k + kk]` made consecutive lanes touch addresses a
full row apart. Each 32-byte transaction served a single half (8× amplification), the
weight row was re-read once per column, and the whole kernel only had `rows * t`
threads (2,560 for `read_proj`), i.e. 10 blocks.

| Fix | Detail |
| --- | --- |
| `read_proj` / `linear` / `write` | One **warp per output row** (`kDSpineGemmWarps = 8` warps per block), lane strides `kk`. The weight row is now a coalesced read and is fetched **once for all t columns**; the activation is contiguous along `kk` in column-major layout. Shared `dspine_bridge_gemv_body` keeps the three call sites identical. |
| `refine`: gate/value | Was `r = threadIdx`, `c = loop`. Now warp-per-`r` with `c` striding by lane, reduced with `__shfl_xor_sync`. |
| `refine`: delta | Was `h = threadIdx`, `r = loop` over `write_scaled[h * R + r]`. Now warp-per-`h` with `r` striding by lane. |
| `refine`: head score | The 16-way head read (H×16 = 82K multiply-adds) ran on **one thread** (`if (tid == 0)`) while 255 threads waited. Now split across the block with a shared-memory reduction. |

Effect at 4K, `--draft-tokens 4`:

| Segment | 3b | 3c |
| --- | --- | --- |
| `layers` | 11.6 ms | **7.8 ms** |
| `head` | 1.8 ms | 1.8 ms |
| `refine` | 3.8 ms | **0.75 ms** (5×) |
| draft total | 17.2 ms | **10.3 ms** |

`dspine_smoke.py --check cli` still reports `output identical (135 bytes)` and the
same `36.0%` acceptance, so the rewrites are numerically equivalent in practice.

### Throughput after 3c (k=5, the measured optimum)

| Context | DSpine | DFlash2 (k=4) | ratio |
| --- | --- | --- | --- |
| 4K | **67.6** tok/s (was 45.7 two passes ago) | 78.7 | 0.86 |
| 8K | 55.6 | 72.5 | 0.77 |
| 32K | 53.4 | 69.4 | 0.77 |
| 122K | 46.9 | 62.3 | 0.75 |

k sweep at 4K: `k=3 63.5`, `k=4 66.0`, `k=5 67.6`, `k=6 64.0`, `k=7 57.7`, `k=15 45.6`.

### Where the remaining ~15 ms per round goes

Isolating with `k=1` (draft nearly free) proved the framework overhead is equal:
DSpine 43.3 ms/round vs DFlash2 38.8 ms at 4K, a 4.5 ms delta that is exactly the
2-row drafter. The draft configs are byte-comparable — both 5 layers, `inter 17408`,
`head_dim 128`, 32 query / 8 KV heads — so the difference is *work*, not size:

1. **`head` 1.75 ms — a hard roofline.** The target vocab is 248,320, so the full-head
   projection streams ~715 MB of Q4 weights per round; 1.75 ms is 436 GB/s, i.e. DRAM
   speed. DFlash2 skips it entirely by using the sparse proposal head stored in its
   artifact (`--lm-head-draft`), which DSpine cannot use because the refinement gather
   is only exact in the primal basis (`DSpine refinement does not support a
   Hadamard-rotated proposal head`). This artifact has no proposal-head component, so
   it is not reachable by configuration.
2. **Draft causal attention ~3.4 ms at 4K — next target.** `dspine_causal_attention`
   gives every `(head, row)` warp its own scan of the context ring, so all `width`
   rows re-read the same KV: ~960 MB per round at 4K against an ideal of ~40 MB. The
   measured 560 GB/s effective rate means L2 absorbs part of it, but the fix is an
   explicit KV-tile-in-shared-memory kernel where all proposal rows consume one tile.
   Expected gain ~3 ms per round.

## 3d. Instrumented by kernel, then fixed (fifth pass)

Sections 3b/3c guessed at the draft cost by summing segment timers. This pass measured
it properly: Nsight Systems with `--cuda-graph-trace=node` (the default
`--cuda-graph-trace=graph` collapses a whole captured decode step into one activity, which
is why earlier traces looked like ten kernels per round), then SQL over the exported
SQLite, with round boundaries taken from the once-per-round `dspine_top16` kernel.

Two measurement traps worth recording:

* A round is **not** one `dspine_top16` interval if the trace also covers prefill; the
  mark list mixes ~52 ms decode intervals with isolated prefill hits. Filtering by the
  median gap is not enough — each kept interval must be evaluated on its own, otherwise a
  dropped gap silently bridges two distant marks and inflates every average.
* Absolute tok/s on this part drifts ~8% with thermal state. Two runs of the same binary
  gave 69.2 and 76.6 tok/s. Every number below comes from one session, with the DFlash2
  and DSpine runs interleaved.

### What the trace showed at k=5, 4K (per round, GPU busy 54.7 ms)

| term | ms | note |
|---|---|---|
| target verify (64 layers) | 40.0 | shared with DFlash2 |
| draft GEMMs (5 layers) | 2.86 | Q4_K, 280–360 GB/s |
| draft full-vocab head | 1.97 | 717 MB, 358 GB/s |
| draft causal attention | 4.56 | **the outlier** |
| refine | 0.86 | |
| bridge read/write/linear | 1.77 | |
| top-16 selection | 0.57 | |

Draft attention was the single largest DSpine term, and the arithmetic closed exactly:
one block per `(head, batch)` with each of the `width` rows scanning the full context
re-reads the slice once per row, so 32 heads x 6 rows x 2.1 MB = 405 MB per layer per
round. 405 MB at 448 GB/s is 904 us against a measured 943 us.

### Fixes, in the order they were made

1. **Context tiles in shared memory.** One block already owned all rows of a head, but
   the rows' warps each walked the whole position range. Each tile of 48 positions is now
   staged once into shared memory and every row consumes it: 405 MB → 67 MB per layer.
   943 → 664 us.
2. **16-byte staging.** The first tile version copied one `bf16` per thread, so a staging
   instruction moved 64 B per warp and the block stalled at every `__syncthreads`. Staging
   is now one `float4` per thread; the launcher checks that the context pointers really are
   16-byte aligned (every artifact tensor is bound at a 256-byte offset, but a
   lane-offset ring buffer need not be) and falls back to the general kernel otherwise.
   664 → 355 us.
3. **Vectorised bridge projections.** `dspine_bridge_gemv_body` stepped its reduction axis
   one `bf16` at a time, making a 512x5120 projection a 160-deep chain of dependent 600 ns
   loads at 34 GB/s. It now loads eight values per lane. 152 → 34 us per call; bridge
   total 1.77 → 0.39 ms.
4. **Single-pass top-16.** The selector rescanned the whole 248k vocabulary once per rank,
   i.e. 16 passes over 2.5 MB per column on five blocks. Each thread now keeps an exact
   descending top-16 in registers (bubbling the new value down with constant indices so
   both arrays stay in registers) and the block merges the 256 per-thread lists through
   the same rank scan. 40 MB → 2.5 MB per round; 0.54 → ~0.1 ms.
5. **Per-row warp split sized to the row count.** `Rows * RowWarps` now fills the
   1024-thread budget (6 rows → 30 warps instead of 24). Kept for correctness of the
   budget, but it measured neutral: 77.1 → 77.0 tok/s. A 32-block grid on 30 SMs costs two
   waves no matter how wide each block is, and the remaining loss is that quantisation,
   not occupancy.

Round assembly: 56.59 → 52.62 ms of GPU busy per round (−7%), with the DSpine-specific
kernels going 7.80 → 3.56 ms.

### Throughput, one session, same prompt and partition

| context | DSpine k=5 | DFlash2 k=4 | ratio |
|---|---|---|---|
| 4K | 77.0 | 84.0 | 92% |
| 8K | 67.3 | 76.9 | 88% |
| 32K | 64.4 | 73.5 | 88% |
| 122K | 56.4 | 66.1 | 85% |

k=5 is optimal at every length (4K: k=4 73.9, k=6 73.9, k=7 65.8; 122K: k=4 55.9,
k=6 50.5, k=7 43.6), so the launcher keeps a single default.

### Why the last ~10% is not a kernel problem

> Superseded by section 4e. The measurement below stands (both artifacts decode at
> 32.3 tok/s with `--spec none`, so verify is identical), but the conclusion drawn
> from it -- that the leftover gap is a property of the shipped artifact -- does not
> survive the k=15 profile or the release's per-task acceptance table.

Running **both** artifacts with `--spec none` gives exactly 32.3 tok/s on each. The target
verify is therefore identical and the entire difference lives in the draft. Two artifact
properties account for it:

* **Drafter bit width.** `dspine_gguf_q4.py` keeps the release's `gguf_q4_k` blocks;
  `dflash2_gguf_blocks.py` keeps whatever the drafter shipped with, which includes
  `gguf_q3_k`. The same five layers stream 15–25% fewer bytes for DFlash2.
* **Proposal head.** DFlash2 scores a sparse selector; DSpine needs the full-vocabulary
  head because the lattice gather must be exact in the primal basis, and this artifact has
  no proposal-head component. That is 717 MB per round — 1.8 ms at 398 GB/s, 89% of peak.

Both remaining dominant terms are already at 78–89% of DRAM bandwidth, so closing the gap
by code means either a sparse proposal head trained into the artifact or a harder-quantised
drafter. The one kernel with headroom left is draft attention: 355 us per call against a
155 us traffic floor, limited by a 32-block grid over 30 SMs. Splitting the context range
across blocks would need a cross-block softmax merge (a scratch buffer plus a finalise
kernel); it is worth ~1 ms per round, i.e. ~1.5 tok/s.

## 4. Validation

Regression script (stdlib only):

```
python backends/ninfer/tools/dspine_smoke.py \
  --artifact "I:\model\ninfer\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-dspine-q4k.ninfer" \
  --check all --max-new 48
```

It asserts (a) greedy `--spec dflash` output is byte-identical to `--spec none`,
(b) acceptance is nonzero, (c) a forced-KVMem-pressure request succeeds without a
worker crash and with nonzero acceptance.

Measured on an RTX 5060 Ti, full build `ninja -C build-backends/ninfer -k 0` → exit 0:

| Case | Result |
| --- | --- |
| CLI, `--max-new 64`, greedy | output `IDENTICAL`; 27/75 accepted, `36.0%`, 6.40 tok/round |
| CLI, `--no-cuda-graph` | `48.1%` |
| CLI, temperature 0.8 / top_p 0.95 / seed 42 | `41.9%` |
| CLI, `--draft-tokens 7` / `1` | `69.4%` / `93.8%` |
| KVMem, 448-token prompt (no pressure), greedy | baseline and DSpine byte-identical (48 tokens) |
| KVMem, 1402-token prompt, budget 512 / capacity 768 | before fixes: HTTP 500 crash → `0.4%`; after: `OK.`, `51.5%` (35/68), decode 19.6 tok/s vs 2.7 tok/s |
| KVMem, 1720-token prompt | no crash, `27.6%` |
| KVMem, budget 768 / capacity 1024 | no crash, `33.8%` |
| KVMem, `--draft-tokens 7` / `1` | `50.0%` / `90.5%` |
| KVMem, temperature 0.8 | no crash, `34.8%` |
| CLI, `--repeat 400` (~21K prompt), after the 8K-window fix | output `IDENTICAL` (135 bytes), `36.0%` |
| KVMem re-check after the 8K-window fix | `OK.`, `37.8%` |

## 4e. Acceptance is content-bound, and that decides the DFlash2 comparison (sixth pass)

The release's own benchmark page, read in full this pass, changes what "fast
enough" means. Its headline `90.1 tok/s` is an average over ten chat prompts; the
per-task table underneath it is:

| Task | tok/s | tokens per verify step | DFlash2 tok/s |
| --- | --- | --- | --- |
| Code | 128.7 | **7.6** | 63.3 |
| Math | 122.8 | 7.3 | 64.2 |
| Summarize / translate | 62.0 | 3.6 | 43.0 |
| Prose / story | 41.6 | **2.5** | 30.2 |

and the depth table says outright that *"on these corpora the drafters average
only about 2.5-3.5 tokens per verify step."* So a prose benchmark cannot exceed
~3.5 tok/step for any drafter, and every measurement made in sections 3c/3d used
exactly that corpus. The wide per-position acceptance numbers there were a
property of the content, not a defect.

Measured acceptance length on this port, temperature 0, `--ngram-draft-tokens 0`,
same artifact family:

| Corpus | DSpine k=5 | k=7 | k=15 | DFlash2 k=7 |
| --- | --- | --- | --- | --- |
| Prose (technical explanation) | 2.79 | 3.05 | 3.27 | 3.19 |
| Code (write an RB tree) | - | 4.22 | 4.36 | 4.02 |
| Code (explain a source file) | - | - | 4.64 | - |

The reference's prose band is 2.5-3.5 and this port lands at 3.27; its code
figure is 7.6 and this port lands at 4.4-4.6. Both drafters stop earning new
tokens around the eighth row, on either corpus, and the shapes are geometric with
different ratios: 0.96, 0.76, 0.78, 0.83, 0.71 ... here against the ~0.9 the
release's code numbers imply.

### Same-session head-to-head in the shipping configuration

RTX 5060 Ti 16 GB, B/R = 36480/16384, nvfp4, `--ngram-draft-tokens 0`,
temperature 0, 128 output tokens through `ninfer-serve` with KVMem:

| Corpus | DSpine | DFlash2 | Ratio |
| --- | --- | --- | --- |
| Code, 37.5K prompt (k=5 vs k=4) | 71.4 tok/s | 62.7 tok/s | **1.14x** |
| Prose, 3.4K prompt (k=5 vs k=4) | 61.0 tok/s | 66.1 tok/s | 0.92x |
| Code, 37.5K prompt, k=7 / k=15 | 63.9 / 49.8 | - | - |

The whole session ran at 0.79x the absolute speed of the session that produced
section 3d's table -- for both backends identically (77.0 x 0.79 = 60.8 against
61.0 measured; 84.0 x 0.79 = 66.4 against 66.1). This card is power-capped at
180 W and drifts with its thermal state, so only ratios carry across sessions.

On prose the two drafters accept the same number of tokens (165 in both runs) and
DSpine simply drafts more per round (443 against 352): the lattice needs the full
vocabulary head where dflash2 shortlists, which is ~1.24 GB of head read per
round. On code the refinement pays for itself and DSpine wins by 14%. The
release's tuning page states the same split: strongest on code, math and tool
calls, prefer DFlash2 or MTP for prose.

### What the refinement is actually worth

`NINFER_DSPINE_PLAIN=1` (new diagnostic) makes the lattice walk emit the parallel
pass's own top-1 chain instead of walking the transition table, leaving every
other part of the round identical. On the RB-tree prompt at k=15:

| Path | accepted by position | length | tok/s |
| --- | --- | --- | --- |
| Transition table | 91,69,54,45,32,25,22,17,12,9,6,5,2,2,2 | 4.36 | 68.3 |
| Parallel chain | 100,71,55,44,30,24,19,13,8,7,6,3,2,1,1 | 4.05 | 63.4 |

So the serial last-write refinement is worth **+7.7% here**, where the release's
code numbers imply roughly 2x over a DFlash-class drafter. The mechanism is
wired correctly -- `delta = 0` reduces the published table to the parallel logits
bit for bit, the walk carries the selected sender rank forward, and greedy
speculation is byte-identical to `--spec none` at temperature 0 on 200 tokens --
so what is missing is not plumbing but conditioning strength: the candidate sets
at row *i* come from the parallel pass, which is conditioned on the parallel
chain, and only the last bridge write is redone per candidate. Anything further
has to come from the drafter, not from these kernels.

### Retraction: the residual gap is not "artifact-shaped"

Section 3d concluded the last 10% was a property of the shipped artifact. That
conclusion does not survive the k=15 profile: verify cost grew 2.35 ms per extra
row (40.15 -> 63.63 ms) because sixteen rows leave the vector kernels and enter
`mul_mat_q`, and a 16-row block buys 0.1 tok/round over 9 rows. Both are real,
but neither is the reason DSpine trails its own release table -- acceptance is,
and acceptance is content-bound and drafter-bound. The kernels are close to their
roof: six rows cost 40.15 ms against 31 ms for a single row, i.e. 1.3x for six
times the rows over the same 12.8 GB of weights.

The one remaining structural lever on the draft side is the full-vocabulary head
read (~1.24 GB per round, mandatory because the lattice needs top-16 over the
whole vocabulary). A shortlist pass plus 16 gathered full rows -- the shape the
refine already computes into `head_rows` -- would remove most of it and is what
would make prose competitive with dflash2. It is not implemented.

## 4f. "An HTML5 pelican riding a bicycle", and the width-9 valley (seventh pass)

The prompt is a long structured output: one self-contained HTML5 game with
several hundred lines of CSS and JavaScript. Both backends were run with
`--default-reasoning-effort none`, temperature 0, KVMem B/R = 36480/16384,
nvfp4, `--ngram-draft-tokens 0`, through `ninfer-serve`. On a 2,048-token output
the same configuration repeats to within 0.2 tok/s, so single runs are usable
there; the 8,192-token runs occasionally drift by more than 20% on this
power-capped card, so those rows were repeated and the best of two is quoted.

| Output | DSpine k=6 | dflash2 k=5 | Ratio |
| --- | --- | --- | --- |
| 1024 tokens | 97.5 (62.2%) | 77.3 (51.5%) | 1.26x |
| 2048 tokens, three rounds | 103.5 / 103.5 / 103.7 | 81.7 / 81.7 / 81.8 | 1.27x |
| 2048 tokens, temperature 1.0 | 105.4 (71.2%) | 79.4 (55.9%) | 1.33x |
| 8192 tokens, two rounds | 112.5 / 110.5 (80.9%) | 109.6 / 109.6 (85.6%) | 1.02x |
| 16384 tokens | 118.3 (88.8%) | - | - |

That prompt does not finish. At 16,384 tokens the response is still inside its
`<script>` block with no `</html>`, so "let it complete and then compare" is not
available on this prompt and the table compares equal token budgets instead. The
model keeps writing a larger and larger game; acceptance rises the whole way,
which is why the later rows are faster than the earlier ones.

Re-measured on the shipped launcher configuration (`--prefill-chunk 1024`,
derived capacity, k=6 against dflash2's own optimum k=5), same session:

| Workload | DSpine | dflash2 | tokens per round | round cost |
| --- | --- | --- | --- | --- |
| HTML5, 2048 tokens, temperature 0 | 104.0 | 82.1 | 5.19 vs 3.89 | 49.9 vs 47.3 ms |
| HTML5, 2048 tokens, temperature 1.0 | 103.0 | 80.4 | 5.16 vs 3.82 | - |
| Prose, 4K prompt, temperature 0 | 62.2 | 72.6 | 3.28 vs 3.55 | 52.7 vs 48.9 ms |

The prose row is the one that goes the other way, and not for the reason the
earlier prose measurement suggested: dflash2 accepts 51.0% of its drafts there
against DSpine's 38.0%, so it wins on draft quality *and* on round cost. DSpine's
refinement is worth +7.7% on code and nothing on prose (section 4e), which is
exactly what these two rows show.

### Letting it finish instead of cutting at a token count

With the token cap lifted to 49,152 and EOS allowed, both backends terminate on
their own. Both artifacts carry the same target (`components.text` is
`qwen3_5_text` IQ3_S in both `conversion.json` files); only the drafters differ.

| | output | wall | decode | acceptance |
| --- | --- | --- | --- | --- |
| DSpine k=6 | 30,161 tokens | 4m 45.3s | 108.3 tok/s | 84.0% |
| dflash2 k=5 | 16,836 tokens | 2m 31.6s | 113.9 tok/s | 92.5% |

The wall-clock gap is content, not speed: the two trajectories diverge, and
DSpine's happens to be a game twice the size. Greedy decoding is deterministic
per artifact, but the two use different block widths (7 rows against 6), so a
near-tied token flips early and everything after it diverges -- the same effect
the release's card warns about for batched verification.

Compared at equal output length the ordering is stable and DSpine leads:

| Output | DSpine k=6 | dflash2 k=5 |
| --- | --- | --- |
| 16,384 tokens | 118.3 (88.8%) | 110.1 (92.4%) |
| tokens per round | 6.33 | 5.62 |
| round cost | 53.5 ms | 51.0 ms |

DSpine accepts 12.6% more per round for 4.9% more round time, so it settles at
+7.4% per token. Over its own 30K-token run it decays to 108.3 tok/s purely from
context growth.

Temperature 1.0 is the launcher's serving default, and it is the widest margin of
the four rows: the lattice walk samples the transition table rather than taking
its argmax, which is the option the release's card describes as helping "slightly
at temperature > 0". The greedy rows are still the right comparison for verify
cost, because both backends then verify the same token stream.

The margin shrinks with output length because the content itself becomes more
predictable: acceptance climbs to 81-86% for both, so the accepted tokens per
round converge (5.86 against 5.28 at 8192) while DSpine's round stays about 9%
dearer (52.5 ms against 48.2 ms). Per round it is paying the full-vocabulary
head and the refinement, which is 3.5 ms of a 52 ms round.

### The k sweep has a valley, not a peak

At 1024 tokens, same session, DSpine:

| k | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tok/s | 78.3 | 91.9 | 93.0 | **97.5** | 91.0 | 78.3 | 88.4 | 93.8 | 81.4 | 95.9 | 98.1 |

and dflash2: 64.1 (k=2), 75.7, 70.1, 77.3, 71.9, 63.3 (k=7), 57.2 (k=15).

The shape is set by `gguf_linear.cpp`'s routing, not by acceptance. Width 1..8
(`k` = 1..7) runs `vector_product`; width 9..16 runs `mul_mat_q`. MMQ at a
handful of rows cannot amortise its tile setup on a 30-SM part, which is why
k=8 (width 9, 78.3) is the floor and the curve only recovers as the row count
grows (k=15, 98.1). The 16-token block the model was designed around therefore
starts paying only at the very top of its range, and then only by 0.6 tok/s
over k=6.

Per-round cost against row count at 8192 context: 5 rows 58.8 ms, 7 rows
52.5 ms, 16 rows 75.6 ms. The 5-row figure being worse than the 7-row one is
run-to-run drift on a power-capped card, not a real inversion -- which is why
the 2048-token sweep was repeated three times before drawing any conclusion.

Extending the vector path to 16 columns is not free either: `launch_columns`
instantiates 1..8, and the kernel holds `acc[2][T]` plus `a[T][8]` per warp, so
`T = 16` doubles register pressure and spills. The remaining lever is the same
one 4e identified: shortlist the 16 candidates instead of reading the full
head every round.

## 4g. Launcher parameter audit (eighth pass)

Every flag in `workers/ninfer/start-kvmem-ninfer-dspine.bat` was checked against
`src/serve/serve_options.cpp` and then exercised end to end. The server reaches
`engine ready` in 8.2 s with the corrected set and reports `KV 52,864 tokens,
nvfp4, explicit | pages 826/826`.

| Flag | Verdict |
| --- | --- |
| `--kvmem` + `--kvmem-budget 36480` + `--kvmem-gen-reserve 16384` | Correct. B and R must both be non-zero multiples of 64; W = B+R drives everything else. |
| `--kv-capacity` | **Was hardcoded.** The engine requires exactly `(B+R) x concurrency`, and concurrency is pinned to 1 by that same check, so the constant silently went stale whenever B or R moved. Now derived with `set /a`. |
| `--kv-capacity 52864` with `--max-context 131072` | Legal only because `--kvmem-budget` is present: `capacity >= max_context` is enforced for non-KVMem runs and skipped for KVMem ones. |
| `--kv-dtype nvfp4` | Correct and the right choice: 4-bit KV is what keeps 52,864 device tokens plus 131,072 of context in 16 GB. |
| `--draft-kv-dtype int8` | **Not usable.** The release runs the drafter cache at q8_0, and this flag exists, but DSpine's bridge attention refuses it: `FATAL ... dspine_bridge: invalid context_k`. Drafter KV stays BF16. |
| `--prefill-chunk 256` | **Was conservative.** 942.7 tok/s at 256 against 973.4 at 1024 on a 37.5K prompt. 4096 fails outright (2.99 GB reservation against 2.18 GB free). Now 1024, which raises the reservation from 1.64 to 1.91 GiB and leaves 392 MiB free. |
| `--fast-prefill-kernel` | **Rejected under KVMem with nvfp4.** The startup check admits it only for INT8/RK8V4/RK4V4 family caches. |
| `--prefill-cublas` | Legal but slower here: 890.9 against 973.4 tok/s, and the help's own advice ("wants a larger `--prefill-chunk`") cannot be followed on this card. |
| `--spec dflash --full-draft-head` | Required pair. `draft.cpp` throws if the proposal head is not `Full`; `--lm-head-draft` would fail to load. |
| `--draft-tokens 6` | In range 1..15 and the measured optimum for structured output. |
| `--ngram-draft-tokens 0` | Deliberate. The default 15 is enabled whenever `--spec` is set; measured identical on code explanation either way, so 0 keeps acceptance purely neural. |
| `--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0` | Valid server defaults. The target applies all four; the draft path consumes only temperature and seed. |
| `--default-reasoning-effort medium` | Valid (`none|minimal|low|medium|high|xhigh|max`), but it is the single most expensive choice in this file for code requests: reasoning tokens are prose, and prose is where acceptance is lowest. `none` produced the HTML5 benchmark directly. |
| `--max-context 131072` | Consistent with the artifact; the excess over W is served from the 12 GiB host tier. |
| concurrency | Not set, so it stays 1 -- which the capacity check requires and which matches the release's `-np 1`. |

No flag in the file is spelled wrong or missing a value. The three that were
worth changing were the hardcoded capacity, the undersized prefill chunk, and
the reasoning-effort default; the three that look attractive and are not usable
are `--draft-kv-dtype`, `--fast-prefill-kernel` and `--prefill-cublas`.

## 4h. At 128K context DFlash2 takes the lead (ninth pass)

The 128K case is the one place where the ordering flips, and it flips on both
corpora, so it is not a corpus artefact.

Protocol: `bench128k.py`, the same B/R partition as the shipping launcher
(36,480 / 16,384, nvfp4, W = 52,864 device tokens), `--prefill-chunk 1024`,
`--ngram-draft-tokens 0`, temperature 0, 256 output tokens with `ignore_eos` so
neither side stops early. Two rounds per backend, interleaved; acceptance counts
came out bit-identical between the two rounds of each backend, so the spread
below is measurement, not sampling.

### Prose continuation, 127,238-token prompt

| | decode | acceptance | tokens/round | ms/round |
| --- | --- | --- | --- | --- |
| DSpine k=6 (shipping) | 55.8 / 55.9 | 177/451 (39.2%) | 3.40 | 61.0 |
| dflash2 k=5 | **60.5 / 60.4** | 175/393 (44.5%) | 3.26 | 53.8 |
| DSpine k=5 | 56.6 | 174/392 (44.4%) | 3.27 | 57.7 |

### Code continuation, 120,814-token prompt

The code corpus is `backends/ninfer/src/ops/**` repeated to length, with the
ring-buffer question from section 4e appended, so the continuation is the same
kind of new code that DSpine wins on at 16K.

| | decode | acceptance | tokens/round | ms/round |
| --- | --- | --- | --- | --- |
| DSpine k=6 (shipping) | 46.5 / 46.5 | 162/552 (29.3%) | 2.78 | 59.8 |
| dflash2 k=5 | **52.7** | 167/436 (38.3%) | 2.94 | 55.7 |

DFlash2 is 8.2% ahead on prose and 13.3% ahead on code. Prefill is a wash:
849-857 tok/s for DSpine against 832-836 for DFlash2, i.e. the two are within
same-session drift of each other, and TTFT lands at 2m21s-2m32s either way.

### Why the ordering flips

Two effects, both visible in the tables above.

**Acceptance converges.** At 16,384 tokens with the same settings DSpine accepts
6.33 tokens per round against DFlash2's 5.62 (+12.6%). At 127K it is 3.40 against
3.26 (+4.3%), and at equal width (k=5, six rows) it is 3.27 against 3.26 --
174/392 against 175/393, a dead heat. On code DFlash2 is the one that accepts
more per round, 2.94 against 2.78, which at 16K was DSpine's home ground. The
drafter still only sees its 8K sliding window while the target attends to 121K,
so the two distributions drift apart and there is little left for the lattice to
refine. Long context turns every corpus into the prose case.

**Round cost grows faster on DSpine.** 53.5 ms at 16K to 61.0 ms at 127K (+14%),
against 51.0 to 53.8 for DFlash2 (+5.5%). Of the 7.2 ms that separates the two at
128K, 3.3 ms is the seventh verify row (k=5, six rows, costs 57.7 ms) and 3.9 ms
is the full-vocabulary head the lattice needs: 1.24 GB of weights per round that
DFlash2 never reads. At short context that rent was paid for by a larger
acceptance; here it is paid for nothing.

So 118.3 tok/s becomes 55.8 by two independent routes:

    118.3 x (3.40 / 6.33) x (53.5 / 61.0) = 55.7

Acceptance nearly halves, then the round gets dearer. Part of the round increase
is context itself: only 52,864 of the 127,238 prompt tokens fit in the device
window, so 74,374 come from the 12 GiB host tier on every step. DFlash2 pays
that too (53.8 ms against 51.0 at 16K), it just pays less on top.

### Bigger k does not help -- at 128K it hurts

Same prose prompt (127,238 tokens), 256 output tokens. One run per row except
k=4 and k=6, which were run twice and reproduced to 0.2 tok/s with bit-identical
acceptance counts.

| k | rows | decode | tokens/round | acceptance | ms/round |
| --- | --- | --- | --- | --- | --- |
| 2 | 3 | 49.4 | 2.34 | 65.8% | 47.3 |
| 3 | 4 | 56.3 | 2.78 | 58.3% | 49.4 |
| **4** | 5 | **58.8 / 59.0** | 3.09 | 51.4% | 52.4 |
| 5 | 6 | 56.6 | 3.27 | 44.4% | 57.7 |
| 6 | 7 | 55.8 / 55.9 | 3.40 | 39.2% | 61.0 |
| 8 | 9 | 39.3 | 3.47 | 30.5% | 88.3 |

Three things decide the shape:

* **Tokens per round saturates.** 3.09 at k=4, 3.40 at k=6, 3.47 at k=8. The
  drafts past the fourth are almost never accepted, which is why the acceptance
  percentage falls faster than the token count rises.
* **Round cost is linear in rows at about 4 ms per row**: 47.3 / 49.4 / 52.4 /
  57.7 / 61.0 ms for 3 through 7 rows. Every verify row has to attend over the
  same 127K of KV, 74K of which arrives from the host tier, so a row is no
  longer the cheap thing it is at 2K.
* **Width 9 is a cliff.** k=8 costs 88.3 ms, 27 ms more than k=7, because verify
  leaves the vector kernels for `mul_mat_q` -- the same valley section 4f found
  at 1024 tokens, except now it sits on top of a 127K context instead of a short
  one.

So the optimum is k=4 at 58.9 tok/s: 5.7% above the shipping k=6 and still 2.6%
below dflash2's own optimum, which stays at k=5 (60.5; k=4 gives 59.2, round
cost 49.9 ms against DSpine's 52.4 at the same five rows -- that 2.5 ms is the
full-vocabulary head). DFlash2 does not have the row problem to the same degree,
which is why its peak does not move.

### What this changes

DSpine's advantage is conditional on context length, not on the artifact. The
shipping k=6 launcher is the right default up to about 32K and the wrong one at
128K: use `start-kvmem-ninfer-dflash.bat`'s backend selection (DFlash2, k=5) when
the working context is over ~64K. Worth measuring once at 64K if long-context
serving becomes the main use; the crossover sits between the 16K row (+7.4%
DSpine) and the 128K rows (-8% to -13%).

If DSpine has to be used at 128K anyway, `--draft-tokens 4` recovers most of the
gap (58.9 against k=6's 55.8) but does not close it: dflash2 at its own optimum
is still 2.6% ahead, and that margin is the per-round head traffic, not a tuning
mistake.

### Where the round actually goes (NINFER_DSPINE_PROFILE)

`NINFER_DSPINE_PROFILE=1` syncs the stream inside the graph, so it has to be run
with `--no-cuda-graph`. That costs about 1% of decode, which does not affect the
split. Same k=6:

| context | draft stack | proposal head | refine | round |
| --- | --- | --- | --- | --- |
| 2,158 tokens | 4.7-5.0 ms | 1.75 ms | 0.79 ms | - |
| 127,238 tokens | 7.1-8.3 ms | 1.76 ms | 0.79 ms | 61.0 ms |

The head and the refine are flat in context -- 1.76 ms and 0.79 ms -- so the
earlier guess that the full-vocabulary head was the whole 3.9 ms premium was
wrong. It is only 1.76 ms. The draft stack grows from 4.7 to 7.4 ms because the
drafter's 8192-token sliding window is only 2,158 keys long at the short prompt
and full at the long one; the fixed part of it (weights plus launches) is about
3.5 ms.

Subtracting from the 61.0 ms round leaves roughly 51 ms for the target's verify
of seven rows. dflash2's whole round is 53.8 ms with six rows and a sparse head,
so the 128K gap decomposes as one extra verify row (about 4 ms) plus the full
head (1.8 ms) -- the row, not the head, is the expensive part, which is the same
conclusion the k sweep reached from the other direction.

## 4i. Diffed against the reference implementation (tenth pass)

The reference ships locally at `I:\model\DSpine-code` (`sglang_dspine`, a Triton
implementation plus `dspine_mlx`) together with the release GGUF the artifact was
built from, `I:\model\Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`. That makes a real diff
possible instead of an inference.

### The conversion is not the problem

Four checks, in increasing strength:

1. `tools/convert/dspine_gguf.py` reads that GGUF directly, and `recipes/dspine_gguf_q4.py`
   keeps the `gguf_q4_k` blocks byte for byte for the feature projection, the
   attention output, the merged gate/up parent and the MLP down.
2. Every geometry field in the artifact's `dflash_config` matches the release:
   block 16, codebook 1024 / rank 512 / kappa 10, `head_topk` 16, refine
   `exact_sender` v1, taps 2/17/32/47/62, `first_exact` 4 with 5 draft layers.
3. The refinement math matches `sglang_dspine/kernels/dspine.py::_serial_table`
   term for term: `base = sum_h norm_w * raw * lm_head[cand_j]`, the sender for
   row `r` is the anchor when `r == 0` and otherwise the previous row's candidate
   at the same lane, the refined score is `base + rms * (write @ message)`, the
   table is `argmax_j` over lanes, and `_serial_chain` walks it from lane 0
   exactly like `dspine_lattice_walk_kernel`.
4. Behaviour: acceptance falls monotonically with k (k=2 65.8%, k=4 51.4%,
   k=6 39.2% at 128K) and reaches 88.8% per row on code at 16K. A mangled draft
   conversion does not produce that shape.

### Three differences that are real

**(a) The reference walks greedily; we sample the chain at T > 0.**
`serial_chain(idx, cand, out, block_size, confidence, confidence_out)` has no
temperature and no seed -- it always takes the table's argmax. Our
`dspine_draw_rank` draws from `softmax(edge / temperature)` whenever temperature
is positive, which is the serving default (1.0). At T = 0 the two are identical.

**(b) Retracted: the reference's draft q is sparse too.** An earlier version of
this section claimed the reference verifies against a dense full-vocabulary q and
that our 16-lane `proposal_q` was an approximation of it. Reading
`accept_sampled_prefix` says otherwise:

```python
dense_q = worker._dense_q_buffer(bs, vocab)[:bs]
dense_q.scatter_(-1, ids, q_rows.to(torch.float32))   # 16 non-zero entries
...
dense_q.scatter_(-1, ids, 0.0)                        # zeroed for reuse
```

The buffer is full-vocabulary only because the Triton kernel indexes `q` by token
id (`DraftProbs + (b*T+step-1)*V + token`); its contents are the same 16
candidate probabilities we carry, zero elsewhere. So the rejection rule
`coin * q < p` and the residual `max(p - q, 0)` see the same q on both sides, and
the acceptance ceiling this section blamed on the 16-way frame does not exist.

**(c) The verify width is a tuned knob in the reference too.**
`docs/sps_tuning.md` sweeps widths 4 / 8 / 12 / 16 and picks by measured wall
time. That is the same quantity as `--draft-tokens`, and it is why the k sweep in
4h is not an artefact of our port: the release itself refuses to fix a width.

Drafter KV precision *is* a difference, in the other direction. The sglang
reference keeps the draft attention in bf16, so our BF16-only bridge matches it;
but the llama.cpp release serves its benchmark with `--spec-draft-type-k q8_0`
on the drafter, which is the configuration we cannot run (4j). It is a paper
difference rather than a real one: 4j prices the whole drafter cache at 160 MiB and
the quantisation of it at 75 MiB, so nothing is being lost in practice.

### What is left, ranked by size at 128K

| item | cost per round | fixable? |
| --- | --- | --- |
| one extra verify row (7 vs 6) | ~4.0 ms | choose k=4 |
| full-vocabulary proposal head | 1.76 ms | shortlist or quantise the head |
| sparse 16-way q instead of dense | acceptance, not time | needs the dense rejection path |
| drafter stack, 8192-token window | 7.4 ms | same for both backends |

### The card's default is 3, not 15

The GGUF card, verbatim: "**Set `--spec-draft-n-max` yourself.** The default is 3,
which uses only a fraction of the 16-token block. 15 is the best value on NVIDIA
CUDA, but on some GPUs a smaller value is faster."

It also states the mechanism outright: "How fast DSpine is depends on how quickly
your GPU can verify a block of 16 tokens at once. On an RTX 3090 that step is
cheap, so long drafts pay off. On GPUs where small batches are slow, a 15-token
draft can cost more than it saves."

The community rows on that card are the same experiment run on other hardware:

| GPU | Best `n-max` | Reported result |
| --- | --- | --- |
| RTX 3090, 24 GB, CUDA | 15 | 90.1 tok/s mixed, 128.7 on code |
| DGX Spark GB10, CUDA | 15 | code 76.7 against DFlash2's 61.3; DFlash2 faster on math, prose, summaries |
| Radeon AI PRO R9700, ROCm | 15 (7 was slower) | code ~2.2x MTP; prose/chat 4-23% slower than MTP |
| Intel Arc Pro B65, SYCL | 15, prose at 3 | code 116.4, prose 36.2 |
| RX 7800 XT, 16 GB, Vulkan | **2** | 47.4 tok/s; "`n-max 15` dropped to 19.1" |

The RX 7800 XT row is the important one for this port: a 16 GB card whose
small-batch verify is slow falls from 47.4 to 19.1 tok/s at N = 15, which is the
same failure mode the k sweep here shows from the other direction. The card also
says to use 3 for prose because "draft acceptance on prose is low, so long drafts
are mostly rejected", and that on its own prose and code-explanation corpora "the
drafters average only about 2.5-3.5 tokens per verify step" -- which is what this
port measures at 128K (3.40 prose, 2.78 code), so those numbers are the corpus,
not a defect.

What this port does with N, measured here: k=15 ties k=6 at 1024 tokens (98.1
against 97.5), loses at 8192 (95.0 against 111.5), and collapses at 128K (39.3
against 55.8). The card never measures DSpine at 128K either -- at N = 15 it fits
`-c 65536` on 24 GB -- so the long-context inversion in 4h has no counter-example
on that page; it is simply outside the range the card covers.

One more default worth recording: card option `--spec-draft-sampling probabilistic`
"is optional. It samples the draft chain from the transition table, and helps
slightly at temperature > 0." Our `dspine_draw_rank` does exactly that whenever
temperature is positive, so this port ships their optional switch as the default
-- which is difference (a) above, confirmed from the card rather than inferred.

## 4j. KV dtype, not the drafter, is the biggest single lever (eleventh pass)

Everything up to here used `--kv-dtype nvfp4`, which the 128K budget needs. Holding
the prompt, partition, k and output length fixed and changing only the dtype:

| run | dtype | accepted | per row | tokens/round | decode |
| --- | --- | --- | --- | --- | --- |
| k=15, 2,174 ctx | nvfp4 | 803/3,298 | 24.3% | 4.66 | 63.0 |
| k=15, 2,174 ctx | **int8** | 846/2,623 | 32.3% | 5.86 | **79.2** |
| k=6, 2,174 ctx | nvfp4 | 789/1,400 | 56.4% | - | 84.4 |
| k=6, 2,174 ctx | **int8** | 806/1,299 | 62.0% | - | **91.5** |
| k=6, 31,838 ctx | nvfp4 | 164/538 | 30.5% | - | 48.0 |
| k=6, 31,838 ctx | **int8** | 171/499 | 34.3% | - | **54.5** |
| k=6, 127,238 ctx | **nvfp4**, W=52,864 | 177/451 | 39.2% | 3.40 | **55.8** |
| k=6, 127,238 ctx | int8, W=28,672 | 144/661 | 21.8% | - | 41.0 |

The column that moves is the target's, not the drafter's. The drafter KV is BF16 in
every row of that table; only the target's cache changes, and acceptance still goes
24.3% to 32.3% and 30.5% to 34.3%. So what a coarse cache damages is the **tap
hidden** the drafter reads -- 4-bit target KV moves the target's hidden states away
from what the drafter was trained against -- and not the drafter's own attention.
The reference runs q8_0 on both sides for symmetry, not because the drafter needs it;
a BF16 drafter in front of a 4-bit target is asymmetric in the benign direction.

Which is why the missing quantized draft path is not worth building. The drafter's
cache is a sliding window, so it does not grow with context: 5 layers x 8 kv heads x
128 head_dim x 2 (K and V) x 2 bytes = 20 KiB per token, times the 8,192 window =
**160 MiB, fixed at 128K as at 2K**. At q8_0 (8.5 bits against BF16's 16) that
becomes 85 MiB, a saving of 75 MiB -- 0.46% of a 16 GiB card. Priced in the currency
that actually binds here, the target's 16 full-attention layers x 4 kv heads x 256
head_dim x 2 = 32,768 values per token, so 75 MiB is 4.3K tokens of extra `W` at
nvfp4 (18 KiB/token, 8% on top of 52,864) or 2.1K at int8. In exchange the bridge
would need a quantized QK^T path, a second staging layout and a fresh acceptance
check. Take the page budget instead; it moves `W` by more than 8% for free.

`--draft-kv-dtype int8` is still unreachable for the record: the option parses, and
`startup.cpp` even sizes the draft ring for it ("quantized draft rings need the
matching per-head layout"), but `context_softmax_attention.cpp`'s `validate_context`
requires `storage == BFloat16` and BF16 K/V pages, so the server dies at startup with
`FATAL dspine_bridge: invalid context_k`. Treated as a non-goal rather than a gap.

At 128K the trade inverts, and it inverts on capacity rather than precision: int8
halves W from 52,864 to 28,672, which pushes 98,566 of the 127,238 tokens onto the
host tier and costs more per round than the extra acceptance is worth. So nvfp4 is
right for 128K and wrong everywhere it is not needed.

`workers/ninfer/start-kvmem-ninfer-dspine.bat` now selects the pair by
`MAX_CONTEXT`: below 64K it derives an int8 window from the context length and
keeps k=6; above it keeps nvfp4 with W=52,864 and drops to k=4. The 32K branch was
run end to end -- `KV 28,672 tokens, int8, explicit | runtime 1.74 GiB | free
464 MiB`, decode 53.8 against nvfp4's 48.0.

### What is reproducible, and what is not

Correction to an earlier version of this section, which blamed run-to-run numerics
for acceptance swings. Two runs with identical flags reproduce **bit for bit**.
Verified on both ends of the range: a 2,158-token prompt with 512 output, k=6,
nvfp4, `--prefill-chunk 512` produced the same text, the same `328/1,080 (30.4%)`
and 56.4 tok/s twice in a row; and the 127,238-token prompt at k=6, nvfp4,
`--prefill-chunk 1024` reported `177/451 (39.2%)`, the exact count recorded for
that row when it was first measured.

Two different things do move, and only one of them is noise:

* **Throughput follows the machine.** That same k=6 row decodes at 41.2 tok/s in
  the session below against 55.8 when it was recorded, with the acceptance count
  unchanged to the token. The card is power-capped at 180 W and the host tier
  competes with the desktop, so absolute tok/s is comparable only inside one
  session -- acceptance counts are the portable metric.
* **Acceptance follows the trajectory, not the drafter.** Any flag that changes
  prefill's reduction order changes the greedy token stream, and a different
  stream accepts at a different rate. `--prefill-chunk` is the one that bites:
  at 128K it moved acceptance by eighteen points and decode by 8% (see 4k). So
  pin every flag in an A/B, including the ones that look quality-neutral.

## 4k. Window, chunk and drafter: three A/B runs in one session (twelfth pass)

All four rows below ran back to back on the same prompt (127,238 tokens), the same
B/R where noted, 256 output tokens, greedy. Same session, so tok/s is comparable.

| config | W | acceptance | decode |
| --- | --- | --- | --- |
| nvfp4, k=4, chunk 1024 | 52,864 | 170/331 (51.4%) | **50.8** |
| nvfp4, k=6, chunk 1024 | 52,864 | 177/451 (39.2%) | 41.2 |
| rk8v4, k=4, chunk 1024 | 36,864 | 163/364 (44.8%) | 44.7 |
| nvfp4, k=4, chunk 512 | 52,864 | 144/439 (32.8%) | 46.6 |

**`rk8v4` is not the missing middle.** RotorQuant's rotated-INT8 key plane over a
packed-int4 value plane costs 408 bytes per head-position against nvfp4's 288 and
int8's 528, so it buys a 36,864-token window -- 16,000 more than int8, 16,000
fewer than nvfp4. At 128K that is still 90,374 tokens on the host tier, and the
run lands 12% under nvfp4 at the same k. The key plane being int8-shaped does not
rescue acceptance either (44.8% against 51.4%), though that comparison is
confounded: the two runs produced different text, and acceptance is a property of
the text.

**k=4 is still right at 128K**: 50.8 against k=6's 41.2 in the same session, and
the acceptance ordering (better per row at k=4, more rows attempted at k=6) is
unchanged from the earlier sweep. The bat's long profile stays at k=4.

**`--prefill-chunk` is a hidden variable.** Same everything, chunk 1024 against
512: the token stream diverges (one run spends all 256 tokens reasoning, the other
answers in prose), acceptance falls from 51.4% to 32.8%, and decode falls from 50.8
to 46.6. Chunk 1024 is also the faster prefill, so there is no reason to lower it
-- but every future A/B has to hold it fixed, and any old number measured at a
different chunk is not comparable.

## 4l. The conversion, audited field by field (thirteenth pass)

Eyeballing a conversion cannot work: a wrong `head_topk`, `rank` or tap list still
loads, still drafts, and only shows up as a lower acceptance rate, which looks
exactly like a slow kernel. `backends/ninfer/tools/dspine_audit.py` now reads the
released GGUF's metadata *and* the shape of its bridge tensors, reads the `.ninfer`
manifest, and compares both against what the runtime requires. 39 checks, exit code
1 on any mismatch, so it can gate a conversion:

```text
python tools/dspine_audit.py --gguf Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \
                             --artifact ...dspine-q4k.ninfer
39/39 checks match
```

The shipping artifact passes all of them. The three derived parameters are
re-derived from tensor shapes rather than trusted, because the GGUF does not state
them:

| parameter | derived from | value |
| --- | --- | --- |
| `code_size` | `dspine_codebook.weight` rows | 1024 |
| `rank` | `dspine_gate.weight` rows, cross-checked against `dspine_write` | 512 |
| `first_exact` | `dflash.block_count - 1` | 4 |

Everything else is a literal copy: taps `[2, 17, 32, 47, 62]`, `head_topk` 16 (from
`dflash.selector_top_k`), `mask_token_id` 248070, hidden 5120, 5 layers, 32/8 heads,
head_dim 128, ffn 17408, context 262144, sliding window 8192, rope theta 1e7, rms
eps 1e-6, block 16, `block-causal`, refine `exact_sender` v1 weight 1.0, second pass
`serial_table`.

### The audit found a real bug in the three older artifacts

`q4`, `q4-gguf` and `q4-bf16proj` all carry `target_layer_ids = [1, 16, 31, 46, 61]`.
The release's GGUF says `[2, 17, 32, 47, 62]`, and the engine consumes the list
verbatim (`config.cpp` forwards it to `stage_taps` with no +-1), so those three
artifacts read the *previous* layer's input for every tap. The shipping `q4k` is
correct -- it was rebuilt after `target_layer_ids_are_inputs` was added to the
recipe, which is the marker that stops the HF convention's output->input +1 from
being applied twice.

The behavioural cost, same target, same prompt, k=6, 512 output, same flags:

| artifact | taps | accepted | decode |
| --- | --- | --- | --- |
| `q4k` (shipping) | 2, 17, 32, 47, 62 | 328/1,080 (30.4%) | **51.8** |
| `q4-gguf` | 1, 16, 31, 46, 61 | 317/1,145 (27.7%) | 50.7 |

Both produced byte-identical text, so the targets agree and the delta is the
drafter. It is not cleanly attributable -- those two artifacts also encode the draft
weights differently (imported Q4_K blocks against a re-quantised q4) -- but a
one-layer tap shift is a certainty in that pair and it is the only *known* defect.
Either way: nothing should be measured on the three old artifacts, and they should
be rebuilt rather than reused.

### Logic, not just data

Two independent smoke checks pass on the shipping artifact:

* `dspine_smoke.py --check cli` -- greedy output of `--spec dflash` is byte-identical
  to `--spec none` (135 bytes) with acceptance 36.0%. This is the check that catches
  a broken bridge: a wrong layout still returns the target's tokens, just with the
  draft doing nothing.
* `dspine_smoke.py --check kvmem` -- a deliberately tiny KVMem budget forces page
  pressure and reselection, and the request still completes with acceptance 37.8%
  and no worker crash.

Add both to any release gate; between them they cover the data path and the
paging path that the throughput numbers cannot see.

### Repairing the tap shift instead of rebuilding

Rebuilding an artifact is ten minutes of CPU and 15 GiB of disk for a five-number
defect, so `dspine_audit.py --fix-taps` rewrites `target_layer_ids` in the manifest
in place. It only attempts the edit when the replacement is byte-for-byte the same
length (`[1,16,31,46,61]` and `[2,17,32,47,62]` both are), which keeps the manifest
length, every tensor offset and the artifact id valid. The array sits inside a JSON
header, at a fixed offset, and nothing hashes it -- `artifact_id` is a label used for
expert profiles and KVMem page-layout names, not a content digest.

Five artifacts carried the shift and were patched: `q4`, `q4-gguf`, `q4-bf16proj`,
`q4-w4`, plus the three that were already correct (`q4k`, `q4-llama`,
`q4-llama-cuda`). The 64 KiB headers are backed up beside them as `*.head.bak`.

The repair is verifiable because it must not change the target's output. Same
artifact, same prompt, same flags, greedy, 512 output:

| `q4-gguf` | taps | accepted | decode |
| --- | --- | --- | --- |
| before | 1, 16, 31, 46, 61 | 317/1,145 (27.7%) | 50.7 |
| after | 2, 17, 32, 47, 62 | 325/1,098 (29.6%) | **53.4** |

Both runs emitted byte-identical text. The target did not move; only the drafter
improved, +1.9 points of acceptance and +5.3% decode for a header edit.

### One more instrument bug, found by the gate

`dspine_smoke.py` had the shipping model id hard-coded, so `--check kvmem` returned
`HTTP 404` on every other artifact -- which reads as a model failure. It now reads
the served id from `/v1/models` unless `--model` pins one.

`dspine_release_check.py` runs audit + cli + kvmem over a list of artifacts and exits
non-zero on any failure:

```text
python tools/dspine_release_check.py --gguf released.gguf \
    --artifact shipping.ninfer --artifact other.ninfer
```

## 4m. Where DSpine actually beats DFlash2, and the greedy-chain attempt (fourteenth pass)

The comparison had been run on prose and at 128K, i.e. exactly the two places the
release's own card says DSpine is weakest ("for prose, chat and image-heavy
requests DFlash2 or MTP can be faster"). Measured on the corpus the card says it
wins -- llama.cpp C++ sources as context, a code question as the continuation, 512
output tokens, greedy, both backends in one session, nvfp4 KV, B/R = 12288/4096:

| context | DSpine (best k) | DFlash2 k=5 | winner |
| --- | --- | --- | --- |
| 9,178 | **66.7** (k=4, 56.5%/row) | 61.9 (41.6%) | DSpine +7.8% |
| 16,832 | 56.4 (k=4, 44.5%) | **65.7** (45.4%) | DFlash2 +16% |
| 37,528 | 49.9 (k=4, 35.8%) | **61.7** (41.6%) | DFlash2 +24% |

A no-speculation control on both artifacts produced the same numbers to the digit
(prefill 969.5 against 969.4 tok/s, decode 29.7 against 29.7 at 37,528), so the
target weights are identical and every difference is the drafter.

The shape of the two curves is the finding. **DFlash2 does not care about context
length at all**: 344/827 accepted at 9,178 and 344/827 again at 37,528, the same
counts, because its draft depends only on a fixed tail window. **DSpine degrades
monotonically**: 56.5% per row, 44.5%, 35.8%. The crossover sits between 9K and
17K, not at the 64K the launcher assumes.

Why it degrades is not the 8K window (both drafters have one, and the tail is
identical in all three runs). The candidate explanation is the coupling 4j
described from the other side: DSpine's bridge reads the target's tap-layer hidden
states, while DFlash2 predicts from its own selector and convolution. At 37K the
target's attention runs over a KVMem window with pages on the host tier, so those
hidden states are the *approximate* target's, not the full-attention target's --
and a drafter trained on the latter loses accuracy in proportion to how much of
the context is approximated. That is a property of running DSpine on top of KVMem,
not of the conversion.

### The greedy-chain experiment: measured, rejected

4i's item (a) said the reference walks its chain greedily while we sample it at
T > 0, which looked like a free win at the serving default. Implemented
(`dspine_draw_rank` keeps the softmax as q but returns the argmax) and measured at
2,158 tokens, 512 output, temperature 1.0:

| T = 1.0, k = 6 | accepted | decode |
| --- | --- | --- |
| sampled chain (shipped) | 304/1,235 (24.6%) | **48.1** |
| greedy chain | 287/1,336 (21.5%) | 44.5 |

Greedy made it 7.5% slower. The reason is a pairing the release does not have to
worry about: `min(1, p/q)` is exact only when the draft really came from q, and a
greedy chain pairs an argmax draft with a q that overstates it. The release's two
paths -- `serial_chain` for its default temperature, `_sample_capped_chain` for
sampled serving -- are not interchangeable, and copying one into the other's slot
is what this experiment did. Reverted; greedy output is bit-identical before and
after (328/1,080 at 30.4%), which is also the control proving the edit only
touched the T > 0 path.

That paragraph's launcher advice does not survive 4n: the crossover it claims is a
window mismatch, not a context-length effect. Read 4n before choosing a backend.

## 4n. The 4m comparison had a window confound (fifteenth pass)

4m concluded that DFlash2's acceptance is flat in context while DSpine's decays, and
laid the decay on KVMem handing the bridge approximate tap hidden states. The
mechanism was plausible and the direction is right, but the measurement was not
comparable: **the two backends were not reading the same amount of history.**

The launchers disagree. `start-kvmem-ninfer-dflash.bat` asks for
`--kvmem-budget 36834 --kvmem-gen-reserve 4096`, i.e. a 40,930-token window, which
covers a 37,528-token prompt completely. The DSpine 4m rows record B/R = 12,288 /
4,096, i.e. 16,384 tokens of visible history at 37,528. So the "flat" backend never
lost anything and the "decaying" one was reading 44% of its own prompt. Since
`--kvmem-budget` is the size of the *selected* history and `--kvmem-sink-tokens`
defaults to a single 64-token page, history beyond B+R is not attended at all -- it
is not a slow page-in, it is invisible.

`tools/bench/run_kvmem_budget_ab.py` isolates that variable: one binary, one
artifact pair, one prompt, one kv dtype, one prefill chunk, one k, and only B/R
changes. Prompt is 40,431 tokens of this repository's CUDA sources plus "rewrite
`dspine_draw_rank` with a bounds check"; 512 output tokens, greedy.

| condition | DSpine k=4 | DFlash2 k=4 |
| --- | --- | --- |
| 10,740 ctx, everything visible | 75.5% · 4.02/round · 85.6 tok/s | 70.3% · 3.81/round · **91.3** |
| 40,431 ctx, everything visible | 68.5% · 3.74/round · 70.3 | 66.0% · 3.64/round · **79.7** |
| 40,431 ctx, window 16,384 | 58.4% · 3.34/round · 68.8 | 50.0% · 3.00/round · **70.0** |

Four readings, three of them corrections:

1. **Losing history costs both drafters, and costs DFlash2 more.** 16.0 points for
   DFlash2 against 10.1 for DSpine. "DFlash2 does not care about context" was an
   artefact of it never having lost any in that run.
2. **DSpine's acceptance is higher at every depth** -- +5.2, +2.5, +8.4 points. The
   serial last-write refinement does what it claims.
3. **The real context-length effect at full visibility is 7.0 points** for DSpine
   (75.5 to 68.5), not the 20.7 the 4m table implied. Nothing about it is specific to
   DSpine's read path.
4. **DFlash2 still decodes faster at every depth on this corpus**, because its round
   is cheaper: DSpine needs the full-vocabulary head (1.24 GB per round) and the
   16x16 lattice walk. On a copy-shaped task both acceptors run high -- 50% to 75% --
   so an acceptance edge of 2 to 8 points cannot pay for the dearer round.

So the backend split is by **task shape**, not by context length. Copy-like
continuation, where the source sits in the prompt and the model rewrites or extends
it, gives the cheaper round to DFlash2 at any depth. Novel generation is the opposite
and is where the earlier measurements came from: 103.5 against 81.7 tok/s on the
pelican game at k=6 against k=5, 71.4 against 62.7 on 32K code. Neither number is
wrong; they are different tasks, and the launcher advice has to say so.

The wide and max arms are also a reproducibility check: 52,864 and 57,344 tokens of
window produced identical counts, 348/508, and identical decode within 4.6 tok/s
(70.3 against 74.9, the spread being the 180 W wall). Nothing above 52,864 is read,
because the prompt already fits.

None of this rescues the distance to the card's numbers. Those were measured at BF16,
FA3 and concurrency 8 on a 936 GB/s card; this is nvfp4, one stream, 448 GB/s. What
4n changes is the *ranking* at equal settings, not the scale.

## 4o. The "0.1% acceptance" was a stale binary, not a proposal bug (sixteenth pass)

The 2026-11-11 00:19–00:33 server runs (`dspine_text2_err.log`, 1/810 = 0.1% at
`--draft-tokens 15`; `dspine_q4k_run_err.log`, 5/750 = 0.67%) were made with a binary
older than the tenth-pass bridge/refine rewrites and the `dd574fd`/`2b0c5f4` commits.
Re-running the same artifact (`...-mtp-dspine-q4k.ninfer`), prompt, flag set and
`temperature 0` on the current build (19:57) gives **6.5%**, and the CLI smoke check
reproduces the documented `36.0%` acceptance exactly, so the drafter itself has not
regressed.

Equal-setting retest, server, `--kvmem-budget 2048 --kvmem-gen-reserve 256`,
`--kv-dtype nvfp4`, `--ngram-draft-tokens 0`, 25-token story prompt, 87-token answer:

| k | DSpine | DFlash2 |
| --- | --- | --- |
| 4 | 24.4% (43/176), 44.5 tok/s | 25.0% (40/160), 43.8 tok/s |
| 15 | 6.5% (43/660), 29.7 tok/s | 12.2% (53/435), 41.3 tok/s |

Same settings on the code continuation the card claims, 160 tokens:

| k | DSpine | DFlash2 |
| --- | --- | --- |
| 4 | 87.2% (123/141), 99.9 tok/s | 72.8% (118/162), 86.5 tok/s |
| 15 | 41.3% (135/327), 99.8 tok/s | 25.6% (125/489), 69.5 tok/s |

So on prose at k=4 the two drafters are at parity, and on code DSpine is ahead at
both widths — the k=15 prose row is the "content-bound / wide-block" valley of 4f
plus DFlash2's cheaper round, not a defect. `--draft-tokens 15` therefore stays a
diagnostic, not a shipping setting.

Two measurements from the same investigation are also retired:

* The refine is **0.82 ms at k=4 and 1.72 ms at k=15** (`NINFER_DSPINE_PROFILE=1`),
  i.e. the 0.75 ms of 3c still holds. The 334/387 ms figure came from the
  pre-3 (single block, single SM) kernel.
* "The draft context ring only holds slots 0 and 8" was an indexing artefact: the
  ring is `[d][slot][kv]`, so a `(8192, 1024)` view aliases eight slots per row.
  Read as `flat[128 * (slot + padded_capacity * kv)]` the warmup ring holds slots
  0..14, i.e. exactly the 15-token warmup prompt, and the dumped
  `layer0_attention_out` reproduces from its own q/k/v/context to maxdiff 0.062
  (bf16 noise). `dspine_causal_attention` is correct.

The bridge chain was re-verified on the same dump against the dequantised
checkpoint (`I:\model\DSpine-b16-gguf2`): `projected` 1.3e-3 relative, `codes`
1.5e-3, `q/s/v` ≤0.083 absolute, `message` 0.023, `delta` 0.137, `out` 0.16% of
`|out|`, next-layer `normed` 0.033 — all bf16 round-off. No outstanding
proposal-distribution defect remains.

## 4o. Where the round actually goes, instrumented (sixteenth pass)

`NINFER_DSPINE_PROFILE=1` used to print three numbers and only on the DSpine path. It
now covers every backend and splits the layer loop three ways:

```
[draft-prof] layers L (attn A mlp M bridge B) head H refine R (top16 T gather G) ms | k= cols=
```

It needs `--no-cuda-graph`, because the boundaries are stream syncs and a capture
cannot contain one. That inflates the absolute numbers by roughly one sync each, so
read the shares, not the totals. At k=6 on a 256-token prompt, medians over 63 rounds:

| section | ms | what it streams | floor at 448 GB/s | efficiency |
| --- | --- | --- | --- | --- |
| mlp (5 layers) | 2.58 | ~800 MB q4k | 1.8 | 70% |
| head (full vocab) | 1.76 | 0.79 GB q4k | 1.8 | ~100% |
| attn (5 layers) | 0.94 | 0.06 GB weights + 0.17 GB draft KV | 0.5 | 53% |
| bridge (5 layers) | 0.84 | 0.08 GB bf16 | 0.18 | 22% |
| refine body | 0.53 | gathered rows only | ~0 | latency |
| top16 | 0.24 | 2 MB logits, 4 blocks | ~0 | latency |
| gather | 0.02 | 64 head rows | ~0 | - |

Two conclusions, and neither is a bug:

* **The two big items are at their memory floor.** The drafter is Q4_K_M, 1.26 GB for
  the whole model, and a round reads essentially all of it: mlp 70% of peak, head
  ~100%. There is no fat to remove there, only bytes -- which is why the earlier
  measurement of "DSpine's round is dearer" reproduces no matter where the kernel
  work goes.
* **The latency-bound items are small.** bridge is 4.6x off its floor and still only
  0.84 ms, because seven dependent launches per layer each move tens of kilobytes.
  Fusing them is a day of numerics verification for 1.6% of a round. top16 is 0.24 ms
  with four blocks scanning 248,320 values each; parallelising it would be worth
  about as much.

Which means the honest answer to "is DSpine slow because of a bug" is no, and the
evidence is the ratio rather than the sum: the draft side is 6.9 ms of a 51.4 ms
round, and 4.5 of those 6.9 ms is unavoidable q4k weight traffic.

### The gap to DFlash2 at 40K is the draft side, and it is 5.9%

Interleaved in one sitting (the 180 W wall moves absolute rates by up to 5% between
processes, so a single pair is not evidence), 40,431-token prompt, identical window
52,864, k=4, greedy, 512 output:

| pass | DSpine | DFlash2 |
| --- | --- | --- |
| 1 | 72.7 tok/s | 77.0 tok/s |
| 2 | 72.5 tok/s | 76.9 tok/s |
| round | 51.4 ms | 47.2 ms |

The 4.2 ms difference is the 6.9 ms draft side minus the ~2.5 ms DFlash2 spends on its
own smaller drafter and shortlist head. Reproducible to 0.3% inside a sitting, so it
is a real effect and not the power wall: DSpine's drafter reads 1.26 GB per round
where DFlash2's reads about a third of that and shortlists its proposal head.

### k at 40K: 4 is already right

The sweep had only been run at 128K. Same prompt and window, k by k:

| k | tokens/round | decode | round |
| --- | --- | --- | --- |
| 4 | 3.74 | **71.9** | 52.0 ms |
| 6 | 4.20 | 69.7 | 60.3 ms |
| 8 | 4.79 | 56.8 | 84.3 ms |

Tokens per round climbs, as it should, but each extra verify row costs 4.1 ms and adds
only 0.23 tokens, so the marginal token is 18 ms against an average of 13.9. k=8 also
falls off the width-8 cliff into `mul_mat_q`. The shipped long profile already uses 4.

This is the one place where a real opening exists. The drafter's cost is fixed per
round, so it *wants* a wide k; the verify rows charge 4.1 ms each even though a
40,431-token context is entirely device-resident and its KV is only 743 MB. At the
KV floor a row should cost about 1.7 ms, and at 1.7 k=6 would win (4.20 tokens per
56.9 ms = 73.8). Making verify rows cheaper is therefore worth more than any change
to the drafter, and it is engine work, not DSpine work.

## 4p. The drafter's window, and the batch size we cannot reach (seventeenth pass)

Two questions drove this pass: does the 8,192-token draft ring actually apply, and is
KVMem's draft coupling what costs the 5.9% from 4o?

**The ring applies only when the budget is at least as large as it.**
`kvmem_window.cpp` clamps the draft floor to the selected-history budget:

```cpp
recent_tokens = std::min(std::max(recent_blocks * 64, masked_draft_ring_capacity()),
                         std::max(retention_tokens, page_tokens));   // retention = B
sequence.dflash_context_floor = end > recent_tokens ? end - recent_tokens : 0;
```

so with `B = 2,816` the drafter conditions on 2,816 positions, not 8,192. The launcher's
short profile derives `B` as a third of the window, which drops under 8,192 for any
`MAX_CONTEXT` below about 24.5K -- so a user who sets 8K or 16K context silently runs a
crippled drafter. Measured with the prompt short enough that both arms see all of it
(7,723 tokens), so the ring is the only variable:

| draft ring | accepted | per row | decode |
| --- | --- | --- | --- |
| 2,816 | 366/577 | 63.4% | 73.4 |
| 8,192 | 369/565 | 65.3% | 74.8 |

+1.9 points and +1.9%, and the launcher now raises `B` to 8,320 (130 pages, so the sink
page cannot eat the last ring page) whenever the context allows. The one-page
difference between 8,320 and 8,192 produced identical counts, so it is insurance
rather than a win. Below `MAX_CONTEXT = 9,344` the capacity ceiling wins and the ring
stays short.

That is the whole of KVMem's draft coupling: **2%, in a corner of the configuration
space, not the 5.9%.** At 40K the shipped budget is 36,480, which is 570 pages for a
128-page ring, and the ring is complete.

### Why the release's "DSpine beats DFlash2" does not reproduce at one stream

The bridge caps the round at 16 columns: `dspine_bridge_gemv_body` accumulates into
`float acc[16]` and `dspine_bridge_read_proj` rejects `t > 16`, where `t = (k+1) x
concurrency`. With `k = 4` that is a ceiling of **three concurrent requests**; the
release runs `bs 8 x 16 draft tokens = 128` columns. Asking for four fails at startup:

```
FATAL server failed during startup | dspine_bridge: invalid read projection
```

Concurrency is the whole point. The drafter streams 1.26 GB of q4k weights per round
no matter how many requests are in flight, so at batch 8 that cost is divided by eight
while DSpine's acceptance edge is not. Aggregate throughput at 8K context, pelican
prompt, k=4, KVMem sized at `(B+R) x concurrency`:

| concurrency | DSpine | DFlash2 |
| --- | --- | --- |
| 1 | **53.7** | 52.3 |
| 2 | 70.5 | **72.4** |
| 3 | **99.3** | 94.1 |

DSpine's marginal cost of the second-to-third request is 28.8 tok/s against DFlash2's
21.7, which is the amortisation showing up. The trend says DSpine pulls away as the
batch grows -- and we cannot follow it past three, because of the 16-column kernel.
That, plus 4o's finding that 76% of the draft side is weight traffic, is the whole
story: at one stream the release's advantage is a 1.26 GB/round tax, and the regime
where it is repaid is the one this bridge cannot enter.

Lifting the cap means a real skinny GEMM -- 128 accumulators, or a shared-memory x
tile -- not a bigger register array. Until then DSpine here is a single-stream backend
and should be chosen on 4n's terms: task shape, not the release's batch-8 numbers.

## 4q. The 16-column cap was a tile, not a limit (eighteenth pass)

The rest of 4p's prediction was wrong, and the correction is small. 16 is not a property
of the kernel; it is the width of one launch. `dspine_bridge_gemv_body` keeps
`float acc[16]`, and every column is an independent warp reduction over the weight row --
no cross-column state, no shared memory, and the caller already passes the per-column
base pointers `x + k * col` / `out + n * col`. So the host can walk the columns in groups
of 16 and get the same answer:

```cpp
constexpr int kDSpineGemmColumns = 16;   // one launch's width (acc[16])
constexpr int kDSpineMaxColumns  = 128;  // release: bs 8 x 16 draft tokens
for (int col = 0; col < t; col += kDSpineGemmColumns) {
    const int width = std::min(kDSpineGemmColumns, t - col);
    dspine_bridge_linear_kernel<<<grid, 32 * kDSpineGemmWarps, 0, stream>>>(
        x_rows + static_cast<std::int64_t>(k) * col, w_rows, n, k, width,
        out_rows + static_cast<std::int64_t>(n) * col);
}
```

with the same treatment for `read_proj` and `write`. Enlarging `acc[16]` instead would
have multiplied the register file and re-read the weights per group, which is the cost
4p was actually describing. Grouping changes only how the work is launched, so each
column's accumulation order is bit-identical to the ungrouped path -- at
`t <= 16` the call is literally unchanged, and above it each group is the same kernel on
an independent slice. The launch count grows to 8 at the release's 128 columns; at bs4
the table below shows no cost from it.

**Four rules wire a batch**, and the first is the one that reads wrong:

| rule | source |
| --- | --- |
| `capacity = (B + R) x concurrency` | `kv_capacity.cpp`: one window per lane |
| `B + R <= --max-context` | `serve_options.cpp:2030` compares the **per-lane window**, not the pool |
| `--prefill-chunk <= R` | the reserve is also the chunk granularity |
| `max(draft, ngram) + draft <= R` | `serve_options.cpp:1992`, i.e. `2k <= R` |

Two consequences worth stating: a batch does **not** need a larger `--max-context`, and
`B` / `R` must already be multiples of 64 (the engine floors them and then rejects the
capacity that no longer matches). `--max-concurrency 4` used to fail at startup with
`dspine_bridge: invalid read projection` before reaching any of this; the message named
no limit, which is why 4p read it as a kernel ceiling.

### R was sized by subtraction, not by need

The launcher used to give the reserve whatever the window had left -- `19,124` in the
short profile. The reserve is **not** a cap on output length: `kvmem_window.cpp:823`
rejects only `end - begin > reserve_tokens`, the width of a single append (verify rows or
a prefill chunk), while the window slides and evicts history as the completion grows. So
`R = 2,048` produced a 4,096-token completion here, and the 10,114-token pelican game ran
on `R = 16,384` only because that is what I gave it. What `R` has to satisfy is
`prefill-chunk <= R` and `max(draft, ngram) + draft <= R`; 2,048 covers chunk 512 and
every drafting width up to `k = 15`. The engine also takes `2,078` down to `2,048`
(`Reserve: 2078 -> 2048 tokens (block alignment)`), so the launcher asks for the aligned
value. Same device footprint, 2.8x the history the target can attend:

| | B | R | window | drafter ring |
| --- | --- | --- | --- | --- |
| before | 9,548 | 19,124 | 28,672 | 8,192 |
| after | 26,624 | 2,048 | 28,672 | 8,192 |

The ring was already complete before this change (it is capped at 8,192 and `B` was
above that), so the gain is entirely on the target side: 9,548 to 26,624 tokens of
selected history for the same memory. `B` still has to clear 8,192 for the ring, which
is why the launcher keeps that floor and only lets a batch break it.

### What a lane costs, measured

`run_capacity_probe.py` (new) starts the server at each candidate and reports the
reservation it asks for. Fitting the four points it can report
`(2, 81,920) 2,617 MiB`, `(2, 112,640) 3,160`, `(4, 16,384) 2,096`,
`(4, 40,960) 2,529`:

```
reservation = 592 MiB + 304 MiB x lanes + 18.1 KiB x capacity
```

18.1 KiB per token is the whole hybrid model's KV, not 64 KiB: only part of the 64 layers
carry attention KV, the rest are linear. The 304 MiB per lane is flat cost -- draft ring,
graph pool, per-lane state -- and it is what limits the batch, not the window:

| lanes | largest config that starts | note |
| --- | --- | --- |
| 1 | B 54,272 + R 2,048 = 56,320 | |
| 2 | B 23,552 + R 2,048 = 51,200 (nvfp4) | 8,192+2,048 also fits |
| 3 | B 8,192 + R 2,048 = 30,720 | ring intact |
| 4 | B 2,048 + R 2,048 = 16,384 | ring clamped to 2,048 |
| 5+ | never starts | 592 + 304x5 spends the budget before the first page |

`--prefill-chunk` is a memory lever in its own right at about 0.36 MiB per token: the
same bs4 config needs 2,249 MiB at chunk 1024 and starts at 512 (2,359 requested against
2,298 free at 1024; ~2,065 at 512). Any A/B must pin it -- it also moves the greedy
stream.

### Where the fix lands

Same prompt (`--chars 4000` of this repo's CUDA sources), same `R = 2,048`, `k = 4`,
512 output tokens, greedy, chunk 512, B = 8,192 up to bs3 and B = 2,048 at bs4 because
that is the only window four lanes fit in. Aggregate decode tok/s, DFlash2 on its own
artifact (`q3km-mtp-dflash2`):

| lanes | DSpine | DFlash2 | DSpine |
| --- | --- | --- | --- |
| 1 | 83.8 | 83.8 | 1.00x |
| 2 | 121.7 | 127.3 | 0.96x |
| 3 | 185.6 | 189.7 | 0.98x |
| 4 | **234.9** | 234.7 | 1.00x |

DSpine's acceptance was higher at every batch size (63.4 / 62.3 / 66.0 / 69.2 against
59.8 / 59.8 / 65.2 / 61.5), and it still only reaches parity, because the round it pays
for is dearer (4o: 4.34 of 6.90 ms is q4k weight traffic, plus 1.76 ms of full-vocabulary
head). Two lanes of that fixed cost pay for themselves almost exactly.

So the honest reading of the release's 3.19x: **the amortisation is real and the cap was
blocking it** -- 83.8 to 234.9 is 2.80x over four lanes, close to linear -- but at equal
batch size DSpine here is a tie, not 19% ahead, and the batch where the release's numbers
come from is eight lanes, which this card cannot hold for memory reasons rather than
kernel ones. Nothing left in the bridge is a bug: what remains is the per-round weight
traffic, which is a 1.26 GB/round floor on a 180 W card.

### Concurrency buys throughput, not latency

One request is one lane, so a batch never makes *that* request faster. On the pelican
prompt (25 tokens in, 1,024 out, `k = 6`, reasoning off) the two numbers move in opposite
directions:

| config | aggregate | per request | wall | acceptance |
| --- | --- | --- | --- | --- |
| nvfp4, 1 lane | 94.8 | **94.8** | 11.4 s | 62.2% |
| int8, 2 lanes | 149.4 | 74.6 | 15.3 s | 65.6 / 66.0% |
| nvfp4, 4 lanes | **232.2** | 58.1 | 20.8 s | 60.6% |

4 lanes is 2.45x the throughput and 1.8x the latency, because four streams now share one
GPU. int8 caps out at two lanes: its KV is 35 KiB/token against nvfp4's 18, so a four-lane
pool costs 560 MiB of pages on top of the 1,808 MiB of flat per-lane cost, which does not
fit in 2.16 GB. Four lanes also need `--prefill-chunk 256`: at 512 the same configuration
asks for 2,216 MiB against 2,164 MiB free, so it starts or not depending on what else is
using the desktop.

## 4r. Why DSpine is only ~5% faster at 2K, and what the round is made of (nineteenth pass)

The recurring question was whether DSpine actually beats DFlash2 at short context and, if
not, which part of the port is broken. Answer: **it does, by 5%, and nothing is broken** --
the ceiling is the target model, not the drafter.

### Same configuration, 2K-ish prompt

1,393-token prompt from this repo's CUDA sources, `B = 8,192`, `R = 2,048`, one lane,
512 output tokens, greedy, prefill chunk 512. DFlash2 on its own artifact:

| k | DSpine nvfp4 | DFlash2 nvfp4 | DSpine int8 | DFlash2 int8 |
| --- | --- | --- | --- | --- |
| 4 | 87.0 | 85.6 | | |
| 5 | 86.8 | | | |
| **6** | **92.8** | **88.5** | **93.4** | **92.6** |
| 7 | 81.2 | 81.9 | | |
| 8 | 69.3 | 69.7 | | |
| 10 | 72.0 | | | |
| 12 | 73.4 | | | |
| 15 | 74.3 | | | |

`k = 6` is the peak on both backends. At `k = 8` both lose ~25%: the verify width becomes
`k+1 = 9`, past `gguf::kMaxVectorColumns = 8` (`ggml_bridge.h:53`), so `gguf_linear.cpp:280`
leaves the vector GEMV path for the MMQ path. The chunked vector fallback at
`gguf_linear.cpp:363` only applies to weight types with no matrix kernel, and IQ3_S has one.
That cliff is in the shared verifier, identically on both backends -- not a DSpine defect.

### Where the 55 ms round goes

512 tokens at 4.73 tokens/round is ~108 rounds; 512 / 92.8 tok/s is 5.5 s, so a round costs
about **51-55 ms**. The instrumented draft side is 7.56 ms:

| segment | ms | note |
| --- | --- | --- |
| MLP (5 draft layers) | 2.59 | ~798 MB of q4_k/q6_k, ~308 GB/s |
| full-vocabulary head | 1.77 | `text/output_head [248320,5120] q4_k` = 715 MB, **404 GB/s = 90% of peak** |
| attention | 1.60 | incl. the 8K draft ring |
| bridge | 0.81 | 9 serial launches per layer vs a ~0.21 ms bandwidth floor |
| refine | 0.79 | top16 0.27, gather 0.02, lattice 0.50 |
| **target forward** | **~30** | 12.5 GiB of weights at 448 GB/s, shared by both backends |

So ~85% of the round is work both backends pay identically. DSpine's extra
`1.77 + 0.81 + 0.79 = 3.37 ms` (6%) is almost exactly the value of its acceptance edge
(63.0% against 58.7%, +8% tokens/round). That is why the two land within 1-5% of each other
at one stream no matter how the draft side is micro-optimised.

### Diffed against the release, line by line

A read-only pass over `/i/model/DSpine-code` (`worker.py`, `models/dspine.py`,
`layers/head.py`, `layers/fp8/*`, `kernels/dspine.py`) against `draft.cpp` and
`dspine_bridge.cu` found no duplicated head projection: both project the full vocabulary
**once per step**, and both skip the anchor column (`draft.cpp:831-845`, `worker.py:227`).
The differences that matter:

| | release | this port |
| --- | --- | --- |
| rows amortising the head read | `bs x 15 = 120` (FP8 GEMM) | `k x 1 = 6` (q4_k GEMV) |
| head bytes per round | ~1.27 GB FP8 coarse + 1.3 MB BF16 rescore of top-128 | 715 MB q4_k, one pass |
| draft layer columns | 16 | `k+1 = 7` |
| bridge launches / layer | 7, with q/gate/prev packed into one GEMM | 9 (now 7) |
| refine tail | 2 fused kernels (`serial_table`, `serial_chain`) | 4 launches |
| logits materialisation | top-k fused into coarse + gather_dot | `[248320, k]` BF16 written then rescanned |

Our head is *cheaper in bytes* than the release's FP8 coarse pass; the release wins by
dividing its read across 120 rows, i.e. by batching. It also precomputes `TG[V,R]` /
`TSV[V,2R]` tables so `serial_table` works in rank 512 instead of hidden 5120; our refine
recomputes the base logit from the gathered head rows (equivalent, a little more work).

### The fix we did, and the negative result

`p_layers` was accumulating only the empty tail after the last bridge instead of the sum of
its three segments; fixed. The three q/s/v GEMVs in the bridge (`draft.cpp:780-785`) shared
one input `codes` and differed only in weight plane, so they are now one launch
(`dspine_bridge_qsv`, `dspine_bridge.cu`) covering `3 x rank` rows in one grid. This saves
2 launches per layer, 10 per round, and the accumulation order per output element is
unchanged. Verified on the same greedy run: `403/640 (63.0%)` accepted before and after,
`93.4` against `93.5` tok/s -- **bit-identical, and no measurable speed change**, because
the production path replays all of this inside a CUDA graph, so launch overhead was already
hidden. That negative result rules out the whole "too many little kernels" hypothesis:
fusing the remaining bridge kernels, tiling `dspine_top16` (0.27 ms at only 11 GB/s because
it launches `k` blocks), or folding top-k into the head epilogue can each buy 0.2-0.6 ms,
i.e. well under 1% of a 55 ms round.

### Conclusion

At 2K, DSpine is faster than DFlash2 by ~5% with the right `k`, and at the wrong `k` (4 or
8) it looks like a tie or a regression. Nothing in the port is computing the wrong thing or
redundantly projecting the head. The missing 3.19x is a batch-size effect: 4q measured
2.80x aggregate scaling by four lanes, and this card cannot hold the eight lanes the release
quotes because of the 304 MiB/lane flat runtime cost (`4q`), not because of the bridge. The
remaining honest levers are batching, or a smaller/approximate proposal head -- an algorithm
change, not a bug fix.

## 4s. Cutting the per-lane draft ring, and the 20.6% that came back (twentieth pass)

4q attributed the batch ceiling to a flat `304 MiB x lanes` cost. This pass opened it up.

### What the per-lane cost actually is

`device_reservation_bytes = persistent + workspace + graph_allowance`
(`startup.cpp:1422`). `graph_allowance` grows with concurrency because
`startup.cpp:1405` adds a whole DFlash graph profile per batch size `1..max_concurrency`;
`--cuda-graph-allowance-mib N` overrides it. `persistent_layout` allocates
`state_image_slots = max_concurrency + device_state_slots` StateImages
(`startup.cpp:138`), each holding the hybrid model's GDN recurrent state plus the
drafter's BF16 KV ring. `/v1/load` confirms KVMem already sets `device_state_slots = 0`
(4 lanes -> 4 slots), so there is no redundant checkpoint pool to reclaim:
`model_instance.cpp:545` zeroes it when `context_cache.enabled = false`
(`serve_options.cpp:2044`). All context-cache CLI flags are rejected under KVMem
(`serve_options.cpp:1997`, "exclusive ownership of history caching").

The reclaimable part is the ring:

```cpp
state_image_spec.dflash_local = DFlashLocalStateSpec{
    .capacity = draft->sliding_window.value_or(0),   // always 8192, per slot
    .storage  = plan.kvmem.draft_kv_storage,         // BF16
};
```

5 layers x 8 KV heads x 128 head_dim x K,V x 2 B is 20 KiB/token, so each lane pays
160 MiB even when the KVMem device window is 2,048 tokens and can never hold more.
`dspine_causal_attention` already bounds its scan by
`span = window < capacity ? window : capacity` (`dspine_bridge.cu:337`) and indexes by
the physical `local.capacity` (`draft.cpp:640`), so shrinking the allocation is exact,
not an approximation.

### The fix

`startup.cpp` now clamps the ring to the KVMem window, rounded **up** to one of the
capacities the cyclic append kernel registers -- `2048 / 4096 / 8192`
(`kv_cache_append.cpp:151`). A window >= 8,192 leaves the value at 8,192, i.e. every
normal configuration is a bit-for-bit no-op. Verified: `403/640 (63.0%)` accepted and
93.3 tok/s on the standard 10,240-token window before and after the change; a 2,048-token
window with the 2,048 ring still generates normally (99.2 tok/s, 65.3%).

Two other constraints surfaced while probing the ceiling:

* 8 lanes cannot use `k = 4`: `dspine_bridge: invalid anchors` because
  `dspine_bridge_codes` requires `block_size = k+1 >= batch`. The release's `N = 15`
  gives width 16, and `16 x 8 = 128` columns -- exactly the ceiling 4q's grouped-GEMV
  fix raised the bridge to.
* `--no-cuda-graph` makes the reservation small enough for 6 and 8 lanes to *start*
  (bs8: 1.91 GiB with prefill 256), but decode then fails with
  `worker out of memory: bad allocation`; with CUDA graphs, bs6 fails during graph
  preparation for every `--cuda-graph-allowance-mib` (that flag reserves budget, it
  does not allocate memory). So >= 6 lanes still needs a separate fix to the engine's
  runtime allocation accounting.

### What it bought

Same prompt, `k = 6`, 1,024 output tokens, reasoning off, prefill chunk 256,
`B = 2,048 / R = 2,048`, nvfp4. Aggregate decode tok/s:

| lanes | DSpine before | DSpine after | DFlash2 after |
| --- | --- | --- | --- |
| 1 | 94.8 | 94.8 | -- |
| 4 | 232.2 | **253.9** | **210.6** |
| 5 | did not start | 254.8 | -- |

The fourth lane went from `2,216 MiB requested against 2,164 MiB free` (prefill 512
would not start) to 1.75 GiB with 618 MiB free. Five lanes now start with CUDA graphs
and decode without OOM, but add essentially nothing to throughput (254.8 against 253.9).
Per-lane at four lanes, DSpine accepted 58.6-65.7% against DFlash2's 44.1-49.5% and
decoded at 61.5-65.9 against 51.0-54.4 tok/s -- **DSpine is 20.6% faster in aggregate**,
the regime where its 1.26 GB/round head is amortised at last. The gain at four lanes is
partly the smaller ring itself (shorter attention scan) and partly that the config now
starts with room to spare.

So the answer to "can we reach the release's bs8" on this 16 GB card is: **four lanes is
the throughput optimum, five is reachable but flat, and six or more needs another engine
fix.** The bridge is no longer the limiter.

## 4t. Measuring a single 64K conversation (twenty-first pass)

Recipe, from `tools/bench/run_kvmem_budget_ab.py`. The prompt is assembled from this
repo's own CUDA sources (`--chars 205000` measured 63,071 tokens at ~3.22 chars/token);
one lane, `k = 4`, 512 output tokens, greedy, `reasoning_effort = none`, nvfp4:

```bash
python backends/ninfer/tools/bench/run_kvmem_budget_ab.py \
  --model I:/model/ninfer/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-dspine-q4k.ninfer \
  --exe workers/ninfer/ninfer-serve.exe --chars 205000 --max-context 76800 \
  --k 4 --prefill-chunk 256 --max-tokens 512 --reasoning-effort none \
  --arms "wide:65536:2048:nvfp4"
```

* `B + R = 67,584 >= 63,071` so **the whole prompt is visible**; anything less puts the
  overflow on the host tier and (per 4n) is not the same measurement.
* `--max-context 76800` only has to cover the per-lane window and the prompt; it is not
  multiplied by concurrency.
* `--prefill-chunk 256` is what makes 67,584 fit: at 512 the reservation asks for
  2,151,848,960 bytes against ~2,075 available, at 1024 it asks for 2.24 GB. With 256
  it reports `capacity | KV 67,584 tokens | runtime 1.88 GiB | free 266.0 MiB`.
* Pin the chunk in any A/B: 1024 and 256 diverge the greedy token stream (4p).

### Results, 63,071-token prompt

`TTFT` is the dominant cost, not decode: prefill ran at 856-878 tok/s, so the first
token arrives after **~72 s** and 512 output tokens add only ~8 s.

| window | backend | k | acceptance | tokens/round | decode tok/s | TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| 67,584 (all visible) | DSpine | 4 | 67.9% | 3.70 | **66.7** | 71.9 s |
| 67,584 | DSpine | 6 | 53.2% | 4.19 | 66.2 | 73.7 s |
| 67,584 | DFlash2 | 4 | 53.1% | 3.12 | 60.1 | 72.3 s |
| 52,864 (drops ~10K) | DSpine | 4 | 43.5% | 2.74 | 52.2 | 72 s |
| 52,864 | DFlash2 | 4 | 58.8% | 3.35 | 69.3 | 72 s |

Everything in the last two rows also used chunk 256, so only the window changed. At
64K, `k = 4` and `k = 6` are within 0.8% of each other (66.7 / 66.2); the launcher's
long-profile `k = 4` is right, and the 2K optimum `k = 6` does not transfer.

**Reading it:** with the whole 64K visible DSpine is 11.0% faster (66.7 against 60.1).
Shrinking the window by 10K tokens costs DSpine 21.8% and *gains* DFlash2 15.3% --
DSpine's acceptance falls 67.9% -> 43.5% while DFlash2's rises 53.1% -> 58.8%. That is
4n's window confound in its purest form. Section 4u makes `POOL_CAP = 67,584` and
`PREFILL_CHUNK = 256` the launcher default so this no longer happens at 64K.

## 4u. Why DSpine loses when the prompt exceeds B+R, and the fix (twenty-second pass)

"DSpine is slower than DFlash2" is not a general statement -- it is only true when
KVMem has to discard history. Same 63,071-token prompt, `k = 4`, chunk 256, greedy,
one lane, with a third row isolating the window size:

| prompt | window | DSpine | DFlash2 | DSpine acceptance |
| --- | --- | --- | --- | --- |
| 63,071 | 67,584 (all visible) | **66.7** | 60.1 | 67.9% |
| 63,071 | 52,864 (drops ~10K) | 52.2 | **69.3** | 43.5% |
| 51,849 | 52,864 (all visible) | 60.7 | 63.1 | 55.0% |

The last row shows the same 52,864 window costs DSpine only ~4% when the prompt fits.
What breaks it is **pressure**, not window size: losing 10K tokens costs DSpine 24.4
acceptance points and 21.8% of its decode rate, while DFlash2 gains 5.7 points and
15.3%.

### Mechanism

DSpine's proposal is built from the target's tap hidden in two independent places:
`draft.cpp:833-845` packs the final draft-layer hidden into `proposal_hidden` for the
full-vocabulary head, and `dspine_bridge.cu:206-235` normalises the projected target
hidden into the bridge `codes` that drive message/delta and the 16x16 lattice.
KVMem's `pressure_selection` (`kvmem_window.cpp:992`) keeps a scored subset of pages
once the frontier passes B+R, so that hidden is computed on an incomplete -- and
step-to-step changing -- context. A fine-grained lattice calibrated on the full hidden
is exactly the part that degrades first. DFlash2 reads the same hidden through a
shortlist head plus argmax (`draft.cpp:972-990`), a coarse readout that barely moves.
DSpine also commits more tokens per round (3.70 vs 3.12 here), and section 5 already
notes that KVMem's retained pages depend on how many tokens each decode step commits,
so the two backends do not even walk the same page set.

### Three hypotheses ruled out

| attempt | result |
| --- | --- |
| compact-domain `draft_floors` for DSpine (`decode.cpp`) | 52.3 / 43.5% -- no change; floor resolves to 0 |
| `--kvmem-sink-tokens 4096 / 16384` | 49.8 / 54.1 -- at best +3.6% |
| `NINFER_DSPINE_PLAIN=1` (skip the serial lattice) | 50.9 / 41.9% -- worse |

So this is not a drafter-side bug a kernel or a sampler knob can absorb.

### The fix

Remove the condition. This 16 GB card holds a **67,584-token** nvfp4 window at one
lane with `--prefill-chunk 256` (1.88-1.89 GiB runtime, 266-393 MiB free); chunk 512
needs 2,151,848,960 bytes and 1024 needs 2.24 GB, both over budget.
`start-kvmem-ninfer-dspine.bat` now makes `POOL_CAP = 67584` and
`PREFILL_CHUNK = 256` the long-profile default, giving `B = 65,536 / R = 2,048`.
Verified: `capacity | KV 67,584 tokens | runtime 1.89 GiB | free 393.1 MiB`, ready in
8.2 s. A 64K conversation therefore gets **66.7 instead of 52.2 tok/s (+27.8%)** and
is 11.0% ahead of DFlash2 again. The cost is a slightly slower prefill (~875 vs 973
tok/s at chunk 1024), which is worth it: TTFT is ~72 s either way at 64K and decode
is 8 s. The standard short-window regression stays bit-identical (`403/640`, 63.0%)
after reverting the two experiments above and rebuilding.

For prompts genuinely beyond 67,584 this card cannot hold the history and the
degradation is inherent to KVMem's approximation, not to DSpine. Section 4v finds the
page scorer that makes that regime less painful for DSpine.

## 4v. The page scorer is DSpine's lever past B+R (twenty-third pass)

"Look at how DFlash retains 2048 and copy it" turned into a hunt for a coordinate bug,
and the hunt was worth doing even though the answer is a scoring parameter.

### Three hypotheses that were wrong

| hypothesis | experiment | result |
| --- | --- | --- |
| ring index goes out of alignment past B+R | give DSpine the compact draft frontier | 52.0 / 43.5% -- identical to baseline |
| DSpine should use DFlash2's logical ring | `logical_draft = true` for DFlash | **1.3% accepted, 20.6 tok/s** -- collapse |
| stale ring slots need masking | compact-domain `draft_floors` | 52.3 / 43.5% -- no change |

The logical-ring experiment is the informative one: `dspine_causal_attention` addresses
the ring as `p % capacity` from the logical proposal positions while the append path
writes compact slots (`compact_position = logical - window.removed_tokens`), so flipping
the coordinate family desynchronises them completely. The code comment at
`decode.cpp:1168` ("DFlash/DSpine must stay compact or decode would append newly
committed features at logical slots the attention kernel never reads") is exactly right,
and the compact-frontier attempt changing nothing shows the two families are already
self-consistent per backend. The pressure trace (`--kvmem-verify-transfers`) also shows
all six pressure events at `phase=prefill`, `cold/dropped = 0`, so decode never
re-anchors the ring. **There is no coordinate bug.**

### What actually differs

When KVMem must drop history it picks pages by score. The default
`page-softmax` scorer picks scattered high-attention pages. DFlash2's shortlist head plus
argmax thrives on that (more exact matches to copy from). DSpine's lattice needs a
coherent target hidden -- its proposal is built from that hidden twice over
(`draft.cpp:833-845` packs it into `proposal_hidden`, `dspine_bridge.cu:206-235`
normalises it into the bridge codes) -- and scattered pages tear it apart.

`meanq-cosine` with `--kvmem-lrfu` scores for query similarity with history decay, so it
keeps contiguous history. Same 63,071-token prompt, same 52,864 window:

| backend | scorer | acceptance | decode tok/s |
| --- | --- | --- | --- |
| DSpine | page-softmax (default) | 43.5% | 52.2 |
| DSpine | **meanq-cosine + lrfu** | **49.7%** | **56.5** (+8.4%) |
| DFlash2 | page-softmax (default) | 58.8% | 69.3 |
| DFlash2 | meanq-cosine + lrfu | 55.2% | 62.8 (-9.4%) |

Both directions are reproducible (56.6 and 56.5 for DSpine; the scorer flip is confirmed
in the log as `KVMEM_SCORING scorer=meanq-cosine agg=max lrfu=1`). The two backends want
**opposite** page sets, so this is not a global default -- it belongs to the DSpine
launcher, and `start-kvmem-ninfer-dspine.bat` now passes
`--kvmem-query-scorer meanq-cosine --kvmem-lrfu`. Nothing is dropped below 67,584 tokens
and the scorer is not consulted there, so short prompts are unaffected; the standard
regression is still `403/640 (63.0%)` bit-identical.

Combined with 4u's window change, a 64K conversation is 66.7 tok/s (full visibility) and
a prompt that must overflow is 56.5 instead of 52.2.

## 4w. What the community GGUF card says our port is missing (twenty-fourth pass)

Reference: `hf-mirror.com/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF` (Apache-2.0 weights from
`tencent/Qwen3.8-27B-DSpine-b16`, run on the `Ajay9o9/llama.cpp` `dspine` branch). It is
the only public DSpine serving number, so it is worth reading closely.

### What the card measures (RTX 3090, 24 GB, Q4_K_M target, q8_0 KV, `-np 1`)

| setup | temp 0 | speedup | tokens per verify step |
| --- | --- | --- | --- |
| no speculation | 28.2 | 1.00x | |
| DFlash2 (7 drafted) | 51.7 | 1.83x | |
| **DSpine (15 drafted)** | **90.1** | **3.19x** | |
| ↳ code | 128.7 | | **7.6** |
| ↳ math | 122.8 | | 7.3 |
| ↳ summarize/translate | 62.0 | | 3.6 |
| ↳ prose/story | 41.6 | | 2.5 |

And at every context depth DSpine beats DFlash2 (depth 0: 48.0 vs 31.6; 8K: 35.6 vs 26.6).
The card's headline 3.19x is the same figure 4r attributed to batching -- it is a
single-slot RTX 3090 number with `--spec-draft-n-max 15`, not a throughput figure.

### Our numbers on the same task shape

Code prompt (this repo's CUDA sources, 1,395 tokens), one lane, 512 out, reasoning off:

| backend | k | tokens/verify | acceptance | decode |
| --- | --- | --- | --- | --- |
| DSpine | 6 | 3.86 | 48.6% | 72.7 |
| DSpine | 15 | 4.78 | **25.2%** | 65.3 |
| DFlash2 | 7 | 4.41 | 48.9% | 74.2 |
| *card, DSpine* | *15* | *7.6* | *~50%* | *128.7* |

**The bug is acceptance length, not kernel speed.** Going 6 -> 15 raises tokens/verify by
only 24% (3.86 -> 4.78) while the block grows 150%, so the back nine tokens are almost
always rejected. The card gets 7.6 at k=15 -- roughly double our chain. Worse, our DSpine
is *slower* than our DFlash2 on code (3.86 vs 4.41) while the card's DSpine is about twice
its DFlash2. Our lattice is not reproducing the release's chain advantage.

### What is not the problem

* Round-time shape: k=6 -> 7 (t=7 -> 8, both vector) costs +9.5 ms/round while k=7 -> 8
  (vector -> MMQ) costs +6.5 ms. The `kMaxVectorColumns = 8` cliff (`gguf_linear.cpp:280`,
  `ggml_bridge_vec.cuh:767` only instantiates 1..8 columns) is real but is not what makes
  k=15 slow; the column count itself is.
* The transition table is laid out and walked correctly: `dspine_refine_kernel` writes
  predecessor-major `table[i][j*16 + jp]` where `j` is the sender candidate and `jp` the
  successor (`dspine_bridge.cu:480-600`), and `dspine_lattice_walk_kernel:665` walks
  `table[i][sender][lane]` chaining `sender = selected`. At i=0 the predecessor is the
  anchor code so every sender row is identical, and the walk starting at `sender = 0` is
  therefore harmless.
* Indexing of `q`/`raw`/`head_rows` follows `q_column = b*width + i + 1` consistently.

### Differences from the card worth chasing next

1. **`--spec-draft-n-max 15` is the card's optimum and ours is not.** Our k=15 acceptance
   of 25.2% is half the card's ~50%, which is where the missing 2.8 tokens/verify live.
   A per-position acceptance trace (`NINFER_DSPINE_DUMP`) against a reference run is the
   next instrument.
2. **Draft KV is q8_0 on the card** (`--spec-draft-type-k/v q8_0`); our DSpine rejects
   anything but BF16 (`dspine_bridge: invalid context_k`, launcher note). That halves the
   ring's bytes and its attention bandwidth.
3. The card's target KV is q8_0 and the target is Q4_K_M (17.1 GB); ours is an IQ3_S/Q4_K
   mix (12.5 GiB) with nvfp4 KV, so raw bytes per round differ.
4. The card runs `-np 1` and says 15 drafted needs a 24 GB card at `-c 65536` because
   "Qwen3.8 keeps a copy of its recurrent state for every drafted token, about 150 MB per
   copy" -- the same 150 MiB/lane GDN state 4s measured here. Their note that `-np 2`
   needs `n-max 7` is the same per-lane budget wall we hit at 5 lanes.

So the remaining gap is one number: **7.6 vs 4.78 tokens per verify step at k=15**. Nothing
in the bridge, the table or the walk is mis-wired; the difference has to be found by
comparing the refined transition table against the reference, position by position.

## 4x. Per-position trace: the proposal head loses the plot at position 1 (twenty-fifth pass)

Instrument: `NINFER_DSPINE_TRACE` (append-mode; `trace_append` at `draft.cpp:937/1186`
already records `candidate_ids` tag=1, `target_argmax` tag=2, `drafts` tag=3,
`accepted_drafts` tag=4). One caveat: `trace_append` synchronises the stream, so it cannot
run under CUDA graph capture -- `--no-cuda-graph` is required or the server dies at the
probe with `cudaErrorStreamCaptureUnsupported`. One `walk_drafts` dump was added to the
per-round `NINFER_DSPINE_DUMP` block (`draft.cpp:963`) to go with it.

81 rounds of a code prompt at `k = 15`, 256 out. Column layout: `target_argmax[:, i]` is
position i's target argmax and matches `drafts[:, i]` directly -- the first test aligned
them off by one and got a 0.01 prefix match against an accepted mean of 2.19; shifting to
`t[:, 0:15]` gives 2.21, which is the identity check.

| pos | top-1 hit | top-16 recall | walk hit | prefix accepted |
| --- | --- | --- | --- | --- |
| 0 | **80.2%** | 84.0% | 80.2% | 79.0% |
| 1 | **2.5%** | 66.7% | 66.7% | 56.8% |
| 2 | 0.0% | 49.4% | 50.6% | 27.2% |
| 3 | 1.2% | 45.7% | 46.9% | 16.0% |
| 4 | 0.0% | 51.9% | 51.9% | 13.6% |
| 5 | 1.2% | 39.5% | 43.2% | 9.9% |
| 6 | 1.2% | 33.3% | 32.1% | 7.4% |
| 7 | 2.5% | 23.5% | 38.3% | 3.7% |
| 10–14 | ~0% | 22–27% | — | **0.0%** |

Mean accepted 2.19 tokens. Three readings:

1. **The proposal head's top-1 is correct at position 0 (80.2%) and collapses at position 1
   (2.5%)**. Not a gradual decay -- a cliff at the first drafted position.
2. **top-16 recall survives** (67% at pos 1 down to 22% at pos 14). The target is still in
   the candidate set; it is just ranked below other tokens.
3. **The lattice walk fixes the ranking**: its independent hit (66.7%) is far above the
   top-1 it started from (2.5%). So `dspine_refine` and `dspine_lattice_walk` are doing
   exactly the job the paper describes -- **the bridge, the table and the walk are not the
   bug**.

That is why the chain dies at positions 2–3: refine can only re-rank 16 candidates, and a
head whose top-1 is wrong at every position past the anchor cannot feed it a good enough
set. The card's 7.6 tokens/verify needs top-1 hit rates near 50% through position 7; ours
is 2.5%.

Next instrument: dump the proposal logits for position 0 and position 1 side by side and
diff their value distributions. A column offset in `proposal_hidden` (`draft.cpp:833-845`
packs `final_hidden[:, i+1]` into column i) would show up as the target's logit staying
high but losing to a consistent wrong token; a NaN/scale problem would show up as an
abnormal logit range at position 1. Either way the fault is in the head's read of
`final_hidden`, not in the bridge.

## 4y. Root cause: the candidate set is too narrow, not the ranking (twenty-sixth pass)

The logits comparison was skipped because a cheaper invariant answered it first.

**`walk` hit rate equals `top-16` recall exactly** -- 66.7% at position 1, 50.6% at
position 2. Whenever the target is in the candidate set, `dspine_refine` +
`dspine_lattice_walk` pick it with 100% accuracy. The ranking machinery is not the
problem; the candidate set simply does not contain the answer often enough.

| pos | top-1 hit | top-16 recall | walk hit | walk = recall? |
| --- | --- | --- | --- | --- |
| 0 | 80.2% | 84.0% | 80.2% | close |
| 1 | 2.5% | 66.7% | 66.7% | **exact** |
| 2 | 0.0% | 49.4% | 50.6% | **exact** |

So the 7.6 vs 4.78 gap at k=15 is a *recall* gap: the card's top-16 must hold the target
through position 7 (~50% per position), ours only holds it to position 1–2. Everything
downstream is doing its job.

Three other things were ruled out on the way:

* **Not a column offset.** `top1(pos1)` matches no neighbouring target (0.074 / 0.025 /
  0.012 against shift −1/0/+1) and `drafts[p] == target[p]` holds at 0.802 / 0.667 / 0.506,
  so `proposal_hidden`'s packing (`final_hidden[:, i+1]` into column i) is aligned.
* **Not a logit bias or NaN.** Position 1's top-1 is 64 distinct token ids across 81
  rounds, and `raw` has no NaN/Inf with every column norm at ≈256.
* **Not a dead bridge.** `normed`'s adjacent-column cosine is ≈0, but that is *expected*:
  DSpine passes causal information through a discrete codebook code, not through hidden
  similarity. The bridge is alive -- `normed` differs from `rms_norm(raw)` by a cosine of
  only 0.63–0.66, so the write path (`message -> delta -> finish`) contributes about 37%
  of the direction. `dspine_bridge_qsv` projects all three planes from the same `codes`
  and `dspine_bridge_message` reads `s[column-1]`/`v[column-1]`, which is the intended
  "gated message from the predecessor's code".

### The fix

Widen the candidate set. `candidates = ops::kSparseSpeculativeCandidates` is a
`constexpr` 16 (`draft.cpp:902`, `dspine_bridge.cu:440`), and the transition table is
allocated as `{candidates*candidates, k, batch}` -- so 32 candidates means a 1024-entry
table per position instead of 256, plus the gather/`dspine_refine`/`dspine_lattice_walk`
arithmetic that scales with it. That is a contained change but needs the full regression
(403/640) and a fresh trace to confirm recall rises as expected. Not attempted yet.

## 4z. Widening the candidate set does not help (twenty-seventh pass)

The 4y fix was implemented (6 sites: `kSparseSpeculativeCandidates` 16->32,
`kDSpineCandidates` bound to it, `round_buffers.cpp` frame shape, the `head_topk` checks
in `config.cpp` and `draft.cpp` relaxed to an upper bound, `kSparseCandidates` in
`speculative_round.cpp` bound to the shared constant, and `dspine_top16_kernel` dropped
to 128 threads so 32 candidates x 256 threads would not blow the 48 KB shared limit).
It built, and the standard regression stayed `403/640 (63.0%)` bit-identical.

It does nothing for the chain:

| code prompt, k=15 | tokens/verify | acceptance | decode |
| --- | --- | --- | --- |
| candidates 16 (baseline) | 4.78 | 25.2% | 65.3 |
| **candidates 32** | **4.78** | 25.3% | 66.0 |
| *card* | *7.6* | *~50%* | *128.7* |

Same-run recall comparison (20 rounds, one trace) shows the widening DOES move recall --
but only where it was not the binding constraint:

| pos | top-4 | top-8 | top-16 | top-32 |
| --- | --- | --- | --- | --- |
| 0 | 90.0 | 90.0 | 90.0 | 95.0 |
| **1** | 5.0 | 10.0 | **20.0** | **20.0** |
| 2 | 10.0 | 15.0 | 30.0 | 35.0 |
| 5 | 0.0 | 5.0 | 15.0 | 35.0 |
| 7 | 0.0 | 0.0 | 25.0 | 35.0 |

Positions 5-7 go 10-25% -> 30-35%, but **position 1 stays at 20%**. The chain dies at
position 1-2, so extra candidates downstream never get a chance to matter. Candidate
width is not the constraint; the proposal head's read at the first drafted position is.

(The two runs do not share a token stream -- changing the candidate count changes the
refine/walk output and therefore the greedy path, which is the section-5 path dependence.
That is why 4y's 66.7% top-16 recall and this run's 20% are not comparable across runs;
the table above compares within one run.)

All six changes were reverted. Regression is `403/640 (63.0%)` again.

### Where that leaves it

The remaining instrument is the one 4x deferred: dump the proposal logits for position 0
and position 1 side by side. Position 1's target must be losing to 31 other tokens; the
question is whether the logit margin is small (a calibration/scale issue in the head's
read of `final_hidden`) or large (the hidden at column 2 is simply wrong, pointing at the
bridge's write path). Either way it is now pinned to one column of one tensor.

## 5a. The logits are fine at position 0 and decay as the hiddens converge (twenty-eighth pass)

The deferred logits comparison was run, and it overturned two earlier readings -- both
were my own layout mistakes, worth recording so they are not repeated.

**The dump is column-major.** `Tensor {vocab, k}` stores `elem[v + vocab*c]`, which numpy
must read as `reshape(k, vocab)` -- `reshape(vocab, k)` (what 4x/4z used) silently
transposes it. The first pass showed "target rank 220,000", which was an artefact. With
the correct layout the same file reads very differently.

Also, `NINFER_DSPINE_DUMP` overwrites per round, so an ungated dump holds the **last**
round -- degenerate, with every position but one padded to 15328 and `proposal_ids`
collapsing to `[15328, 248070 x 15]`. Both dump sites (`proposal_hidden`/`proposal_logits`
at `draft.cpp:897`, `verify_ids`/`target_argmax`/`drafts`/`accepted` at `:1186`) are now
gated to the first two rounds.

Second round of a code prompt, k=15, `--no-cuda-graph`:

| pos | hidden norm | logit std | target rank | margin | top1 = target? |
| --- | --- | --- | --- | --- | --- |
| 0 | 133.7 | **1.339** | 1 | 0.0000 | **YES** |
| 1 | 138.1 | 1.943 | 5 | 0.0371 | no |
| 2 | 137.9 | 1.947 | 149 | 0.1953 | no |
| 3 | 136.3 | 1.931 | 4 | 0.0371 | no |
| 4 | 136.4 | 1.917 | 1 | 0.0000 | **YES** |
| 5 | 135.9 | 1.908 | 9 | 0.0547 | no |
| 6 | 135.6 | 1.879 | 141 | 0.1602 | no |
| 7 | 135.5 | 1.878 | 31 | 0.1094 | no |

So the head is **not** broken: positions 0 and 4 are exactly right (margin 0), and 1/3/5
miss by 0.037-0.055 with the target ranked 4th-9th. Only 2/6/8 are genuinely wrong
(rank 141-191, margin ~0.18). This also rehabilitates 4y: the target really is inside
top-16 at positions 1/3/5 (rank <= 9), so the recall story was right and the 4z "top-32
recall 20%" number was the same transposition artefact.

The structural clue is in the hiddens themselves:

| pair | cosine |
| --- | --- |
| cos(0,1) | 0.372 |
| cos(1,2) | 0.541 |
| cos(2,3) | 0.662 |
| cos(3,4) | 0.783 |
| cos(4,5) | 0.875 |

Adjacent draft columns get monotonically MORE similar the further into the block they
are -- 0.87 by position 5. DSpine's adjacent causal injection is meant to pass a
predecessor's discrete code forward, not to collapse the hidden states, and the chain
breaks exactly where the cosine first crosses ~0.5 (positions 2-3, matching the
acceptance cliff in 4x). By position 5 the column is nearly a copy of its predecessor.

Two further oddities in the same dump: every position's softmax top-1 probability is
~0.0000 (logit std 1.3-1.9 over 248,320 tokens is an extremely flat distribution), and
position 0's logit std is 1.339 against 1.94 for every later position -- a step, not a
gradient.

So the question is no longer "why is the head wrong" but **why the write path makes
column c+1 converge on column c**. That is one kernel to look at
(`dspine_bridge_write_kernel`, the `message -> delta -> finish` chain at
`draft.cpp:788-792`), and it is testable directly: dump `message`/`delta` per column and
check whether the gate saturates or whether `delta` carries the predecessor's direction.

## 5b. The draft block is fed identical inputs and converges (twenty-ninth pass)

The 5a question -- why column c+1 converges on column c -- was traced to the very start
of the draft forward. The whole chain is now visible in one dump (`k=6`, width 7):

| tensor | adjacent-column cosine | note |
| --- | --- | --- |
| `embed_residual` (= `embed([anchor, mask x 6])`) | **1.0000** for cols 1-6 | identical |
| `target_hidden` (target tap hidden) | 0.39 - 0.83 | genuinely different |
| `layer0_raw` (after layer-0 attention) | 0.11 -> 0.90 | already converging |
| `layer0_normed` | 0.02 -> 0.63 | |
| `layer4` codes | 0.07 -> 0.57 | |
| `proposal_hidden` (fed to the head) | 0.37 -> 0.90 | |

`prepare_masked_block` produces `ids = [anchor, mask x k]`, so `embed(ids)` gives every
drafted position the **same** vector (`proposal_ids = [874, 248070 x 6]`,
`max|col1-col2| = 0.0`). The only per-position information the block then gets is RoPE on
q/k and the bridge's message. Attention over a block whose K/V are identical up to RoPE
produces columns that are all "the same content, different phase", which is exactly the
0.11 -> 0.90 ramp -- and it keeps compounding through the five layers.

The model card says the drafter "reads hidden states from five layers of the target
model". Those hiddens exist and are per-position correct (`target_hidden` cols differ at
0.39-0.83 cosine, norms 130-134), but they are only used as **history** -- appended to the
draft KV ring via `compact_features` -- never as the block's own input. `target_hidden` is
produced at verify time (`draft.cpp:1061`, written by the verifier), so the current
round's block cannot read it; the block is fed the mask embedding instead.

That is why position 0 works and position 1 does not: column 0 carries the anchor's real
embedding (`ids[0] = 874`), so its content is right; columns 1-6 carry a mask and have to
manufacture content from RoPE phase and the predecessor code alone. It also explains why
`dspine_refine` + `dspine_lattice_walk` still hit 66.7% whenever the target is in top-16
(4y): refine rebuilds each candidate from the **codebook code** of the predecessor token,
i.e. real content, not from the converged hidden.

### The fix

Give the block per-position content. The candidates, in order of how much they disturb
the release behaviour:

1. Feed `target_hidden` (or its per-layer slice) into the block in place of / on top of
   `embed(ids)`. Cleanest match to the card, but `target_hidden` is a verify-time tensor,
   so it needs the *previous* round's copy (or the target must expose the tap hiddens for
   the drafted positions before the drafter runs).
2. Make the bridge read the target tap hiddens (`compact_features`, 5 x 5120 stacked)
   instead of the draft residual, per layer. Also matches "reads hidden states from five
   layers", and the data is already on the device.
3. Give `embed(ids)` a per-position component (a positional embedding added to the mask
   vector) so the columns are no longer identical. Smallest change, least faithful to the
   card.

Not yet attempted -- the choice between 1 and 2 depends on where the reference reads its
per-position content, and that is the next thing to pin down.

## 5c. Exhaustive reference audit: the pipeline is correct; k=6 is the speed fix
(thirtieth pass)

The 5b "identical inputs" hypothesis was checked against the **training code**
(`sgl-project/SpecForge` `specforge/modeling/draft/` + `specforge/algorithms/`),
the authors' own inference port (`Tencent/DSpine` `dspine_mlx/`), and the
community llama.cpp port. Verdict: **the shared mask input is the trained design,
not a bug.** `_create_noise_embed` fills every mask slot with the same
`mask_token_id` and only slot 0 carries the anchor token -- exactly what
`prepare_masked_block` does. The convergence of the mask columns is inherent to
that design; the serial last-write refinement is the mechanism that repairs it.

Everything DSpine-specific was then verified **numerically** against a real
k=15 round dump (`build-backends/ninfer/logitdump`) plus the bridge weights
extracted from the artifact:

| check | method | result |
| --- | --- | --- |
| q/s/v GEMV | recomputed from dumped `codes` + artifact `gate/pred/prev` weights | max err 0.03 (BF16) |
| gate/message | `σ(q+s_prev+b)·v_prev` recomputed | max err 0.007 |
| refine transition table | full numpy recompute (15x16x16) from `raw/rms/q/cand/head_rows` + weights | max err 0.04, **argmax 240/240** |
| causal block attention | softmax(QK/√d)V over [ring slots < frontier] + block j<=w, 15 (head,w) probes | cos = 1.0000, max err 0.05 |
| bridge weights vs source GGUF | `blk.N.dspine_read/write`, gate/prev/pred_gate, bias, `output_norm`, `enc.output_norm`, `fc` | **max diff 0.0** (bit-identical) |
| early/final read mapping | `early_read_weights[L]` vs `read_weight` for L=0..4 | cos = 1.0000 (mapping correct) |
| context K/V weights | `attention/context_key` vs `attention/key` bindings | same object range (aliased) |

Training-semantics checks (SpecForge `AdjacentInjection` + `dflash_mask`):
bridge formula, per-layer read/write, `sqrt(rank)` code scale, rms write scale,
anchor passthrough, predecessor = codebook[anchor] then `features[t-1]`, in-place
labels (`labels[...,1:]` vs mask rows), `pos = anchor + [0..bs-1]`, context =
`kv < anchor_pos` strictly, in-block `kv_offset <= q_offset` -- all identical to
the local kernels. The one deviation found (per-row sliding-window lower bound
`anchor_pos + offset - (W-1)` vs the shared `frontier - W`) is worth < 16 slots
of an 8192 window and is invisible at these prompt sizes.

### The layer-tap question is settled empirically

llama.cpp stores `t_layer_inp[il] = inpL` **before** the layer (layer input);
the authors' MLX `_LayerHook` stores the block's **return value** (layer
output). The local engine matched llama.cpp. `NINFER_DSPINE_TAP_OUTPUT=1`
(text.cpp) switches the tap to the output for an A/B; measured at k=15,
512 out, 1701-token prompt:

| tap | tokens/round | acceptance | tok/s |
| --- | --- | --- | --- |
| layer input (default) | **5.86** | 32.6% | 86.7 |
| layer output (MLX) | 5.32 | 29.2% | 79.4 |

The layer-input convention wins. Keep the default.

### The speed problem is k, not acceptance

Temperature 0, 512 out, 1393-token code prompt, B/R = 8192/2048, 1 lane.
No-speculation baseline: 29.2 tok/s (34.3 ms/round).

| k | tokens/round | acceptance | tok/s | speedup |
| --- | --- | --- | --- | --- |
| 4 | 3.84 | 71.2% | 85.3 | 2.92x |
| **6** | **4.60** | **60.3%** | **90.7** | **3.11x** |
| 8 | 4.91 | 48.9% | 73.5 | 2.52x |
| 10 | 5.16 | 41.6% | 74.8 | 2.56x |
| 15 | 4.91 | 26.5% | 67.3 | 2.30x |

k=6 is the optimum on this card and reaches **3.11x** -- within 3% of the
community card's 3.19x. k=15 (the 3090 optimum) is 26% slower here because our
verify of 16 rows is dearer relative to the draft. The launcher's
`DRAFT_TOKENS=6` below 64K is therefore correct.

### Where the acceptance actually stands

Deterministic (temp 0), so repeat runs are bit-identical. Per k=15 round the
proposal ranks the true token 1, 2, 2, 4, **3787**, 7206, 9, 13 over the valid
rows -- the chain holds ~4 tokens then a content-dependent cliff, and the
refine+walk recovers every row whose target is inside the top-16. Code-vs-true
cosine (the training's alignment loss target) is 0.26-0.69 on the good rows and
0.05 (random) exactly on the cliff rows.

Against published numbers: the **official tencent SGLang** acceptance lengths
(T=0, block 16) are HumanEval **5.71**, MBPP 6.14, WorkBuddy Code **5.08**,
GSM8K 7.20. Our 4.91-5.86 tokens/verify is inside that code band. The community
llama.cpp card's "7.6 on code" is a different (easier) code prompt on a Q4_K_M
target; ours is an IQ3_S target whose lm_head and five taps carry the quant
noise. No remaining code-level difference to the reference implementations was
found after the checks above.

## 5d. The 128K regression WAS a bug: full-ring draft attention (thirty-first pass)

"DFlash2 k=4 is faster than DSpine k=6 at 128K" turned out to be a real defect,
not an inherent KVMem cost. 5c had declared the port bug-free after verifying
every kernel -- that audit covered the maths but never the **scan boundary** at a
prompt that overruns the window.

### Evidence that ruled out the earlier explanations

Same 108,662-token prompt, 67,584 window, meanq-cosine+lrfu, temp 0, 1 lane:

| check | result |
| --- | --- |
| per-position acceptance | DFlash2 k4 [91,80,69,63]%, DSpine k6 [64,50,33,19,9,4]% -- DSpine collapses from position 0, no chain-depth pattern |
| decode-phase KVMem pressure | 0 events (all eviction happens at prefill) |
| TRACE per-round acceptance | [2,1,2,0,3,4,3,6,3,3,3,6,...] -- **no progressive ring corruption** |
| compact/frontier coordinates | `execution_frontiers=66097`, `draft_frontiers=66097`, `context_frontiers=66094`, `append_counts=3`, `append_positions=[66094..66096]` -- self-consistent, same shape as the short-context round |
| context ring | 8192/8192 slots populated, 8192 distinct keys, uniform norms, no stale broadcast or coordinate discontinuity |
| long-context attention | numpy recompute of softmax(QK/sqrt(d))V over `p in [frontier-8192, frontier)`, `slot=p%8192`, 16 (head,w) probes -> **cos = 1.0000** |

Every kernel, weight and coordinate is correct. What differs is **how far the
draft attention scans**.

### Prompt-length scan (window fixed at 67,584, k=6, meanq)

| prompt | visible? | acceptance | tok/round | tok/s |
| --- | --- | --- | --- | --- |
| 1,393 | yes | 60.3% | 4.60 | 91.5 |
| 21,435 | yes | 59.5% | 4.54 | 82.3 |
| 57,237 | yes | 50.5% | 4.03 | 67.1 |
| 76,566 | **no** | 34.7% | 3.07 | 50.4 |
| 108,662 | **no** | 30.1% | 2.80 | 43.3 |

DFlash2 on the same 76,566 prompt gets 57.5%. Length alone is not the problem;
crossing the KVMem window is.

### The actual bug

`dspine_causal_attention` uses `span = min(window, capacity)` with
`window = config.sliding_window = 8192` and `capacity = 8192`, so on any long
round `span == capacity`. The scan then covers **every ring slot exactly once**,
including slots still holding features written under an earlier compact
generation (prefill eviction rewrites the target KV but the draft ring is only
ever appended to, at three or four rows per round). Those stale rows are
content-addressable, so they hijack the softmax and destroy the lattice; the
shortlist/selector readout in DFlash2 is insensitive to them. This is why 5c's
numerical audit (which checked that the kernel computes the intended sum) and
4v's coordinate audit (which checked that append/attention agree on coordinates)
both passed while the model still lost 50 acceptance points.

A/B with `NINFER_DSPINE_DRAFT_WINDOW` on the 108,662 prompt (k=6, 512 out):

| span | acceptance | tok/round | tok/s |
| --- | --- | --- | --- |
| **8192 (= capacity, old default)** | **30.1%** | 2.80 | **43.3** |
| 7168 | 83.5% | 6.01 | 96.0 |
| 6144 | 82.4% | 5.94 | 96.2 |
| 4096 | 80.1% | 5.81 | 94.9 |
| 2048 | 79.0% | 5.74 | 101.8 |
| **1024** | **80.1%** | **5.81** | **104.0** |
| 512 | 76.9% | 5.62 | 94.2 |

Only the full-ring path fails. 7168 is already healthy, which pins it to the
`span == capacity` boundary rather than to context length generally.

### Fix

`draft.cpp` now caps the default scan at `capacity/8` (1024 here, floor 512) so
the kernel never takes the full-ring path; `NINFER_DSPINE_DRAFT_WINDOW=N`
overrides it (8192 restores the old behaviour). Verified with the shipped binary
and **no environment variable**:

| prompt | before | after |
| --- | --- | --- |
| 108,662 tokens | 30.1% / 43.3 tok/s | **80.1% / 97.6-104.0 tok/s** (+125-140%) |
| 1,393 tokens | 60.3% / 91.5 tok/s | **61.5% / 93.6 tok/s** (no regression) |
| 21,435 tokens | 59.5% / 82.3 | 62.3% / 87.7 |
| 57,237 tokens | 50.5% / 67.1 | 52.0% / 70.6 |

Greedy output is bit-identical to the explicit `=1024` run (acceptance and
tokens/round match exactly); the tok/s spread is clock/thermal noise.

**DSpine now beats DFlash2 at 128K**: k=6 97.6-104.0 vs DFlash2 k=4 79.5
(+23-31%), reversing the 43.3 vs 79.5 result that prompted this pass. Post-fix
k sweep at 108K: k=4 93.5, **k=6 97.6 (best)**, k=8 67.7. The launcher's long
profile therefore moves `DRAFT_TOKENS` from 4 back to 6, matching the short
profile.

### Same-session confirmation (thirty-second pass)

The 97.6-104.0 / 79.5 pair above was collected in different sessions. To remove
clock, thermal and prompt-drift noise, both arms were rerun back-to-back with the
shipped default binary, no `NINFER_DSPINE_DRAFT_WINDOW`, and byte-identical
flags -- `--chars 365000` (108,662 tokens), B/R 65536/2048, nvfp4, chunk 256,
`--kvmem-query-scorer meanq-cosine --kvmem-lrfu`, temp 0, 512 out, 1 lane --
changing only the artifact, `--spec` and `k`:

| prompt | arm | accept | tok/round | tok/s |
| --- | --- | --- | --- | --- |
| 108,662 | DFlash2 k=4 | 75.6% (384/508) | 4.02 | 81.2 |
| 108,662 | **DSpine k=6** | **80.1% (423/528)** | **5.81** | **97.4** |
| 104,376 | DFlash2 k=4 | 68.2% (373/547) | 3.70 | 74.6 |
| 104,376 | **DSpine k=6** | 67.5% (409/606) | **5.01** | **83.9** |

At the exact historical prompt DSpine wins by **+20.0%** (97.4 / 81.2); at a
slightly shorter 104,376-token prompt it still wins by **+12.5%** (83.9 / 74.6).
Acceptance and `tokens_per_round` at 108,662 are bit-identical to the earlier
single-arm runs (384/508 and 423/528), so only the ~2% tok/s drift is
clock/thermal noise. The margin is prompt-dependent, but the sign is not: the
pre-fix 43.3 vs 79.5 deficit (-46%) is now a +20% lead.

Artifacts used: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km-mtp-dflash2.ninfer
--spec dflash2 --draft-tokens 4` and `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-dspine-q4k.ninfer
--spec dflash --draft-tokens 6`. Result dirs:
`build-backends/ninfer/h2h108_{df2,dsp}` and `h2h_{df2,dsp}`.

### Why the full-ring path is unsafe specifically on the compact ring

`decode.cpp` gives the two ring flavours different invalidation floors:

```cpp
dflash_host_ingress->draft_floors[row] = checked_i32(
    logical_draft ? sequence.dflash_context_floor : 0, "DFlash draft floor");
```

DFlash2 owns a logical ring and receives a real `dflash_context_floor` after a
KVMem reselection; DFlash/DSpine use compact coordinates and hard-code `0`.
Because `compact_position = logical - removed_tokens`, eviction shifts the whole
coordinate system while the append-only draft ring keeps its old rows, so a scan
that reaches `frontier - capacity` necessarily revisits slots written under an
earlier coordinate system. Capping the scan at `capacity/8` keeps every read in
slots written in the current generation. A fuller fix would carry a validity
floor into the compact draft ring (clamped so it cannot underflow below
`removed_tokens`) and let `dspine_causal_attention` mask stale slots, restoring
more of the trained 8K window without reintroducing the collision. That is left
as future work: the 1024-slot scan already recovers 80.1% acceptance, and short
context shows no regression. The cap is also safe for every registered ring size
-- startup clamps capacity to 2048 / 4096 / 8192, giving windows 512 / 512 /
1024, all strictly below capacity.

## 5e. Why the Web UI feels slower than the benchmark: sampling, not the engine (thirty-third pass)

A real Web UI task -- "make an HTML5 canvas animation of a pelican riding a
bicycle" -- was run against the launcher's default server
(`--temperature 1.0 --top-p 0.95 --top-k 20 --default-reasoning-effort medium`,
k=6, B/R 65536/2048, meanq-cosine+lrfu, streaming, 6000 max tokens) and then
repeated with sampling and thinking removed separately. Same binary, same prompt,
same server instance, 120-token prompt.

| arm | sampling | thinking | decode tok/s | dflash accepted | e2e tok/s | first visible token |
| --- | --- | --- | --- | --- | --- | --- |
| A Web UI default | 1.0 / 0.95 / 20 | medium | 87.8 | **57.0%** (4,191/7,356) | 85.2 | **24.97 s** |
| B greedy, no thinking | 0 | none | **107.8** | 73.5% (4,081/5,550) | 104.8 | 0.31 s |
| C sampled, no thinking | 1.0 / 0.95 / 20 | none | 86.3 | 56.2% (4,625/8,232) | 84.5 | 0.07 s |

Two independent effects:

* **Sampling costs ~17 acceptance points and ~20% decode.** B vs C isolates it:
  73.5% -> 56.2%, decode 107.8 -> 86.3 tok/s. The DSpine draft lattice is built
  from the head's argmax/shortlist while the verifier samples under top_p/top_k;
  a non-argmax target token breaks the chain. This is the limitation already
  listed in section 5 ("top_k / top_p / min_p filtering inside the DSpine
  proposal path is not wired up"), now quantified on a real task. It is not a
  regression from the full-ring fix: A and C both decode at ~87 tok/s, and B
  beats the 97.4 tok/s 128K benchmark because a 120-token prompt never triggers
  KVMem pressure.
* **Thinking costs the perceived wait, not throughput.** A emitted 4,645 chars of
  reasoning before the first visible token (24.97 s). Its 5-second throughput
  windows run 56-71 tok/s during exploratory reasoning and 96-112 tok/s during
  structured code emission -- the predictability split that also makes B's 73.5%
  reachable. Total decode is essentially unchanged by thinking (87.8 vs 86.3).

The 97.4 tok/s headline is a greedy, no-thinking, code-completion figure at 128K.
For this Web UI task the honest numbers are 87.8 decode / 85.2 end-to-end with a
25 s blank wait before anything renders. Both are correct measurements of
different configurations.

Practical levers, in order of effect:

1. `--default-reasoning-effort minimal` (or `none`) removes the 25 s blank wait.
2. Lower temperature / disable top_k+top_p for code-generation sessions restores
   ~17 acceptance points and ~20% decode speed.
3. The real fix is to wire the verifier's sampling distribution into the DSpine
   proposal path (speculative sampling rather than a greedy lattice judged by a
   sampled verifier). Untested.

Artifact: `build-backends/ninfer/webui-pelican/pelican_bike.html` (10 KB, single
file, canvas + requestAnimationFrame). Timings in `timing.json` and
`ab_timing.json` in the same directory.

## 5f. Web UI at xhigh: both backends collapse to ~58 tok/s and never finish (thirty-fourth pass)

`xhigh` is a real effort value (the Qwen3.8 template's native default; `high`
and `max` fold onto it). Re-running the same pelican-on-a-bicycle prompt with
`reasoning_effort: "xhigh"`, `max_tokens = 12000`, streaming and the launcher's
sampling defaults (1.0 / 0.95 / 20), on both backends:

| arm | model / k | decode tok/s | accepted | e2e tok/s | total | visible output |
| --- | --- | --- | --- | --- | --- | --- |
| DFlash2 xhigh | q3km-mtp-dflash2 / k=4 | 57.9 | **38.1%** (7,247/19,007) | 56.5 | 3m 32s | **none** |
| DSpine xhigh | dspine-q4k / k=6 | **58.5** | 33.2% (7,990/24,045) | 57.4 | 3m 29s | **none** |
| (5e) DSpine medium | dspine-q4k / k=6 | 87.8 | 57.0% | 85.2 | 1m 3s | HTML at 24.97 s |

Both xhigh runs ended with `finish_reason = length` after exactly 12,000 tokens,
all of them `reasoning_content` (39,124 / 36,794 chars, `content_chars = 0`).
This is not a loop -- the traces are still deriving chain-path arc angles and
frame-tube coordinates at the cut-off. It is the effort template: xhigh makes the
model design the entire implementation line by line before emitting a single tag.

Speed consequence: exploratory reasoning text is far less predictable than the
code it precedes, and top_p/top_k sampling breaks the draft chain on top of that.
Acceptance falls from 57.0% (medium) to 33-38%, and decode from 87.8 to ~58
tok/s (-34%). DFlash2's higher acceptance (38.1% vs 33.2%) is exactly offset by
k=4 vs k=6: it drafts 19,007 tokens against DSpine's 24,045 and both land at
~58 tok/s. At xhigh the two backends are effectively tied, and neither produces
usable output within a 12K budget.

Practical consequence: for a "generate an animation" Web UI task, xhigh is the
wrong setting. `medium` finished the whole animation in 63.6 s; xhigh produced
nothing in 212 s. Either lower the effort or set an explicit `thinking_budget`.
If xhigh is required, `max_tokens` needs to be ~20K+ just to reach the code.

Timings: `build-backends/ninfer/webui-pelican/{dflash2,dspine}_xhigh_timing.json`
and the matching `_reasoning.md` traces.

**Unlimited-budget follow-up (DFlash2 xhigh, `max_tokens = 65536`).** With no budget
cut-off the request does terminate on its own:

| metric | value |
| --- | --- |
| finish | `stop` (not `length`) |
| output | 59,859 tokens, of which **51,857 are reasoning** (86.6%) |
| wall | 15m 53.8s, decode 64.2 tok/s, e2e 62.8 tok/s |
| first visible token | **853.3 s** (14m 13s of blank waiting) |
| acceptance | 48.4% (39,484/81,500) |
| artefact | `pelican_dflash2_xhigh.html`, 18 KB, complete single-file canvas animation |

So the answer to "does it finish if you remove the limit?" is yes, but it spends
14 minutes thinking before emitting any code. The 12K run's 38.1% acceptance was
the low-water mark of the exploratory phase; the full run averages 48.4% because
the final ~8K tokens are structured code, where acceptance recovers. This is a
template/effort behaviour, not an engine defect: the same task at `medium`
finishes in 63.6 s.

### Complete DSpine vs DFlash2 speed comparison (all same-binary measurements)

| condition | DFlash2 k=4 | DSpine k=6 | DSpine delta |
| --- | --- | --- | --- |
| 128K benchmark, temp 0, 108,662 tok | 81.2 tok/s, 75.6%, 4.02 tok/round | 97.4 tok/s, 80.1%, 5.81 | **+20.0%** |
| 128K benchmark, temp 0, 104,376 tok | 74.6 tok/s, 68.2%, 3.70 | 83.9 tok/s, 67.5%, 5.01 | **+12.5%** |
| Web UI `medium`, temp 1.0, same prompt | 73.7 tok/s, 54.8% (3,968/7,244), 80.8 s | 87.8 tok/s, 57.0% (4,191/7,356), 63.6 s | **+19.1%** |
| Web UI `xhigh`, 12K budget (both truncated) | 57.9 tok/s, 38.1% | 58.5 tok/s, 33.2% | **+1.0% (tie)** |
| Web UI `xhigh`, no budget (DFlash2 only) | 64.2 tok/s, 48.4%, 15m 53.8s | not measured | -- |

DSpine wins by 12-20% whenever speculative decoding is still working. The gap
comes from `tokens_per_round` (5.81 vs 4.02 at 128K; k=6 vs k=4 drafts more per
round), not from a large acceptance difference -- at Web UI `medium` the two are
within 2.2 points (54.8% vs 57.0%). At `xhigh` both collapse to ~58 tok/s and
tie, because exploratory reasoning plus top_p/top_k sampling breaks the draft
chain so thoroughly that the target model's plain decode dominates. The
configuration (effort + sampling) swings absolute speed by -34%, far more than
the choice of backend does.

## 5g. The launcher's 530 MiB of headroom was inside the Windows eviction cliff
(thirty-fifth pass)

The user's server and this document disagreed by 40% with identical flags. The
live process (started 20:53 from the launcher, `/v1/models` args byte-identical
to the ones in 5e/5f) decoded at **51.7 tok/s** on the 1,470-token code prompt,
**65.3** on the pelican prompt with 512 out and **73.0** with 2,048 out. A restart
of the same binary with the same flags measured **107.3 / 82.9 / 112.7** on the
same three probes. Nothing in the engine changed between those runs.

### Reproducing it: foreign VRAM, not the engine

The launcher sizes the KV pool to leave 530 MiB free (`capacity | ... | free
530.7 MiB`). Holding a torch tensor in another process and re-running the same
1,024-token pelican probe:

| foreign allocation | reported GPU used | decode | delta |
| --- | --- | --- | --- |
| none | 15,559 MiB | **112.2** | -- |
| 16 MiB (CUDA context only) | 15,709 MiB | 109.5 | -2% |
| 256 MiB | 15,618 MiB | 112.5 | 0% |
| **512 MiB** | 15,814 MiB | **76.7** | **-32%** |
| 768 MiB | 15,809 MiB | 75.0 | -33% |
| 512 MiB, released again | 15,159 MiB | 112.2 | 0% |

The cliff sits between a few hundred and ~0.5 GiB of demand from anything else
on the desktop: a browser tab, a video call, another CUDA process, or the
NVIDIA overlay. The driver then pages the 12.5 GiB of weights (and the KV pool)
into shared memory and every decode step reads them over PCIe. It is not
thermal (2,835 MHz, 130 W of a 180 W cap, 53 C while slow) and not a
draft-quality collapse: the three probes above span very different acceptance
(74-78% on the pelican run, 54% on the code run) yet all lose the same ~0.62x,
which is the signature of every weight read crossing PCIe rather than of a
shorter chain; the target verify, 78% of a round, carries most of that loss.
Releasing the foreign allocation restores the speed, so the pages do come back --
the live process only stayed slow because whatever took the VRAM at 20:53 kept
holding it.

Two things that do *not* help, both measured:

* `--wddm-evictable-budget` (commit arenas at startup): 107.6 baseline,
  **71.8** under the same 512 MiB grab. The flag moves when WDDM makes room, not
  whether it evicts later.
* Shrinking `--kvmem-host-mib` from 12,288 to 2,048: free went 530 -> 471 MiB.
  The KVMem host tier is host payload, not a device charge (`program_impl.cpp`
  clamps only the pinned context-cache buffer, whose WDDM coupling is the one
  the 1 GiB floor already covers).

The engine's own startup guard is fine: born under a real 5 GiB deficit it
refuses to start ("only 0 bytes are available for runtime capacity") instead of
running slowly. The failure mode is a *running* process that loses pages later.

### The fix

`start-kvmem-ninfer-dspine.bat` no longer assumes the 2,100 MiB "left after the
weights" measured on a quiet desktop. It asks the driver
(`nvidia-smi --query-gpu=memory.free`), subtracts the 12.5 GiB of weights and
`DSPINE_HEADROOM_MIB` (default 768), and sizes B from that; `=0` restores the
old 67,584-token window when nothing else uses the card. The engine now logs one
warning when it comes up with less than 640 MiB free, so a slow session is
explained in its own log instead of looking like a DSpine problem.

Same probes, same session, with the new sizing (1 lane, nvfp4, k=6):

| window | free after startup | baseline | +512 MiB foreign VRAM |
| --- | --- | --- | --- |
| 67,584 (old default) | 530 MiB | 113.0 | **73.4** |
| 58,752 | 659 MiB | 112.7 | 111.9 |
| 49,152 | 796 MiB | 112.5 | 112.1 |
| 31,616-51,456 (new default) | 660-790 MiB | **112.3-112.7** | **111.7** |

The new window is whatever free VRAM allows after the weights and the headroom,
and that moves with the desktop: 51,456 tokens at 15,369 MiB free (21:24),
49,600 at 15,190 (21:28), 31,616 at 15,020 (21:47, browsers holding ~350 MiB
more). One GiB of KV is ~57K tokens on this artifact (18 KiB/token), so ~320 MiB
of desktop VRAM is ~18K tokens of window -- that sensitivity *is* the price of
the headroom, and `DSPINE_HEADROOM_MIB=0` hands it back for a quiet card. The
1,470-token code prompt is unaffected (84.4 before, 82.5 under the 512 MiB
grab), and 128K prompts still run through KVMem's host tier past the window.
The warning path was checked by starting the old 67,584 window explicitly:

```
capacity | KV 67,584 tokens, nvfp4, explicit | pages 1,056/1,056 | runtime 1.83 GiB | free 485.7 MiB
WARN  VRAM headroom | only 485.7 MiB free after startup: a GPU application taking about 0.5 GiB
      makes Windows page these weights to shared memory and decode drops ~35%. ...
```

Residual risk: a 1 GiB grab still costs ~30% at this window (the cliff scales
with the grab, not the window), and short prompts cannot be made immune on a
16 GB card. What the change buys is the common case -- one browser or a second
CUDA context -- and a log line that says which regime the server is in.

## 5h. The 108K headline does not reproduce today (2026-11-12 22:47)

Re-running 5d's exact benchmark (same script, `--chars 365000`, B/R 65536/2048,
nvfp4, chunk 256, k=6, meanq-cosine+lrfu, temp 0, 512 out) gives **108,664
prompt tokens and 60.6 tok/s, not 97.4**:

| run | rounds | accepted | tok/round | tok/s |
| --- | --- | --- | --- | --- |
| 5d (18:44 artifact) | 88 | 423/528 (80.1%) | 5.81 | 97.4 |
| now, run a | 141 | 370/842 (43.9%) | 3.62 | 60.7 |
| now, run b | 141 | 370/842 (43.9%) | 3.62 | 60.6 |

Two cold starts in a row land on the same token counts, so the engine is
deterministic and this is not session noise; it is also not WDDM pressure --
`available_after_startup_bytes` is 462 MiB in the 5d record and 353-370 MiB
here, and the 4 GiB of paging headroom that costs is visible in decode, not in
acceptance. It is not the window either: a 36,729-token prompt, fully inside the
same 67,584 window, accepts 34.8% (3.08 tok/round, 55.3 tok/s), so the draft is
weak on this prompt family whether or not anything spills to the host tier.

What still separates the two sessions is the prompt tail (108,662 against
108,664 tokens -- the corpus was cut in a slightly different place) and
whatever output the 18:44 run generated from it; the run itself saved no
completion text. Until that is reproduced, **treat 97.4 tok/s as specific to
that session's prompt**, not as a property of the configuration, and use these
numbers as the 108K baseline:

| configuration | acceptance | tok/round | tok/s |
| --- | --- | --- | --- |
| k=6, nvfp4 67,584, meanq | 43.9% | 3.62 | 60.6 |
| **k=4, same** | **56.9%** | **3.28** | **63.8** |
| k=6, rk8v4 38,848, page-softmax (launcher at 22:27) | 32.0% | 2.98 | 53.1 |

k=4 is the 128K optimum in this state, which is the opposite of the short-context
result in 5c (k=6) and worth remembering when a long-context profile is added.

## 5. Known limitations



* The new lattice walk uses `temperature` and `seed`; `top_k` / `top_p` / `min_p`
  filtering inside the DSpine proposal path is not wired up. The target verifier
  still applies the full sampling configuration, so output correctness is
  unaffected; acceptance under heavily filtered sampling may be lower.
* KVMem is an approximate paged memory system. Once a pressure/reselection event
  occurs, the target's retained pages depend on how many tokens each decode step
  commits, so greedy reasoning can diverge between `--spec none` and `--spec dflash`
  (observed 45 vs 36 reasoning tokens at a 1402-token prompt). Both runs produced
  the same final answer, and no divergence occurs without pressure. This is
  inherent to KVMem's path dependence, not a speculative-decoding correctness bug.
* A KVMem budget smaller than the drafter's 8192-token sliding window necessarily
  shortens the usable draft context; acceptance degrades gracefully instead of
  crashing. The 8192 window is now actually enforced by `dspine_causal_attention`
  (section 3b); before that fix the scan ran to the ring capacity.
* The `reg.exe` sandbox blacklist warning seen during CUDA builds is benign; the
  object files and executables are produced normally.
