# Device route profiles

Many NInfer operations have several interchangeable schedules: CTA shapes and warp counts of the
small-T attention kernel, the column tiles of the ternary and groupwise projections, whether the
probability-times-value product accumulates in FP16, which prompt-attention kernel prefills. The
compiled tables that choose between them were measured on one card. The best choice depends on the
part: its SM count sets how many CTAs fill a wave, its memory system sets where a kernel stops being
latency-bound, and a schedule that is best on an RTX 5090 can be several times slower on an RTX 3090.

A device route profile records, for one GPU model, the schedule each operation takes per width band
as a measurement found it. The engine installs it before any operation runs; operations without an
entry keep their compiled route.

## Where profiles come from

Up to three layers answer, per route key:

1. **Your profile file.** `$NINFER_DEVICE_PROFILES`, else `$XDG_CACHE_HOME/ninfer/device-profiles.json`,
   else `~/.cache/ninfer/device-profiles.json` (`%LOCALAPPDATA%\ninfer\device-profiles.json` on
   Windows). `--device-profile-path` names another file.
2. **The built-in table.** Profiles measured on 48 GPU models ship inside the binary
   (`src/runtime/engine/device_profiles.json`); the Profile column of the
   [card tables](gpus.md#cards) shows which. Each entry was measured at the card's default power
   limit by the build that serves it: the image's `75`, `86` and `120a` builds (an Ada card runs the
   `86` build), CUDA 12.8 `80` and `70` builds for the A100 and V100, a CUDA 12.8 `86` build for
   the A10, whose host driver (570) predates the image's CUDA 13, and a CUDA 13.1 `120a` build for
   the RTX 5070 Ti, with the image's route catalog. Variants that report different names have
   their own entries, such as the PRO 6000's Workstation (600 W), Max-Q (300 W) and Server
   editions. The Turing entries come from a `75` build, the only build that runs on them; its route
   catalog lacks the keys a Turing build never selects.
3. **The compiled route** of each operation, for a key neither names.

For a key both the file and the built-in table name, the file's bands answer up to their last width
and the built-in bands past it. A key the file does not name keeps its built-in bands, so calibrating
an RTX 3090, 4090 or 5090 replaces the keys calibration measures and keeps every other built-in key
(the `unified/*` switches calibration does not measure, below). A key calibration measured and found
the compiled route best is stored with an empty schedule, which overrides a built-in choice for it.

A GPU with neither a file entry nor a built-in one is calibrated once when the engine starts, before
the weights are loaded. For the built-in table this took 35 to 75 seconds on most Ampere, Ada and
Blackwell cards with 46 or more SMs, 75 to 132 seconds on smaller ones and 94 to 204 seconds on
Turing cards, and the result is written to the profile file for later starts. A `70` build runs its own projection and attention routes, which no profile key
steers, so it calibrates the two-stage GDN switch alone, in about nine seconds on a V100.

An entry applies only to the same hardware class (the GPU name and compute capability) and the same
SM count, so a laptop part that shares a desktop part's name but not its SM count is calibrated on its
own. With several GPUs (`--devices`), every distinct device gets its own profile.

`--device-profile auto` (the default) follows the order above, `--device-profile calibrate` measures
again and stores the result over the file's entry, and `--device-profile off` uses the compiled
tables only.

## Stale and outdated profiles

Each calibrated entry records its provenance in a `calibration` block:

```json
{"hardware_class": "nvidia-geforce-rtx-3090-sm86", "multiprocessors": 82, "origin": "ninfer-calibrate on NVIDIA GeForce RTX 3090",
 "calibration": {"route_catalog": "33f546be871d5bec", "ninfer_build": "v0.12.0-41-gabcdef0",
                 "cuda_driver": 12080, "cuda_runtime": 12080, "date": "2026-10-02T09:30:00Z"},
 "routes": {"...": [[32, "r16c8"]]}}
```

`route_catalog` is a digest of every route key calibration measures and of the candidate schedules
it times for each (`src/calibration/route_catalog.cpp`). When a new build adds, renames or drops a
key or a schedule, its digest changes, and an entry of the profile file measured against another
digest is not applied: the engine logs a warning naming both digests and the build that measured
the entry, and the built-in profile (or, for a GPU without one, a fresh calibration at start)
takes its place. `--device-profile calibrate` or `ninfer-calibrate` measures it again. An entry in
the file without a `calibration` block is ignored the same way. The built-in table is compiled with
the binary and is not checked.

The file's format is `ninfer.device-route-profiles` with `schema_version` 2. A file of another
version is not read: the engine logs that it was written by another build and asks for a
recalibration, which replaces a file of an older version as a whole (its entries are unusable by
this build). A file of a newer version is left untouched.

## Calibrating by hand

```bash
ninfer-calibrate --print > my-gpu.json
```

`ninfer-calibrate` measures every route family on the current GPU (`--device N`) and stores the profile
where the engine looks for it (`--out PATH` elsewhere); `--print` writes the stored entry, with its
`calibration` block, to stdout. `--detail` logs every candidate's time, `--only PREFIX` limits the
run to route keys with that prefix, and `--no-ternary`, `--no-groupwise`, `--no-attention` and
`--no-linear-attention` skip whole families. A limited run replaces only the keys it measured: the
stored entry keeps the others when it was measured against the same route catalog, and is replaced
whole otherwise. Close other GPU work first: the measurement assumes an idle device.

Each candidate is timed through the same dispatch an inference call uses, on synthetic weights and
caches of the registered model shapes, with the L2 cache flushed before every sample, as the median
of eleven runs. A candidate replaces the compiled route only when it is at least 3 % faster, the win
survives a second interleaved measurement, and its output matches the compiled route's to within
5 %.

## What is calibrated

| route key | chooses | measured on |
|---|---|---|
| `t2_i8_route` | small-T kernel or prefill GEMM per width, ternary (Bonsai) projections | the GDN layer's projections at widths 16 to 192 |
| `t2_i8_small/<N>x<K>` | row and column tile of the ternary small-T kernel | every ternary projection shape, widths 1 to 32 |
| `q4_q5_attn_input/...`, `q4_q5_gdn_input/...`, `q5_linear_add/...`, `q4_linear_swiglu/...` | fused groupwise (Qwen3.6/3.8) projection schedules | the model shapes, widths 1 to 64 |
| `attn_i8_small/h24/<kv>/w<W>` | warps, CTAs per SM, key block, split QK (`q`) and early fetch (`e`) of the INT8-family small-T attention; for `int8` also `8x2x32p`, the pipelined kernel that keeps V codes INT8 in shared memory and streams the next 32-key tile through two stages | query widths 1 to 8, KV windows 8K, 64K and 262K, for `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` |
| `attn_pv_f16` | FP16 accumulation of the probability-times-value product per key tile | a 1024-token prompt chunk at 32K and a decode step at 131K |
| `attn_pack_gqa` | the standard INT8 prompt kernel with each KV head's query heads packed into its tiles (PackGQA) | a 1024-token prompt chunk at 32K and 131K |
| `attn_prompt_fast` | the fast prompt-attention kernel (rows kept in registers, FP16 PV per tile) | a wave-aligned prompt chunk at 32K and 131K |
| `attn_prompt_split` | whether the fast INT8-family prompt kernel may split a launch's keys over several CTAs per row block and merge their partial rows (`off`), for calls of more than 64 columns whose row blocks alone would leave SMs idle; the split count comes from a cost model over the SM count | a 256-token continuation at 32K and 131K and a wave-aligned chunk at 131K, where the fast kernel serves prompts |
| `attn_parallel_tiles` | single-row chunked small-T attention over an INT8-family cache as parallel query tiles (one batched append, one split-KV launch over every tile, one reduce) instead of serial fused chunks; widths without an exact 2-8 column tile divisor stay serial; `NINFER_ATTN_PARALLEL_TILES=0\|1` overrides | 16 and 32 verify columns at 32K and 131K |
| `q4_linear_add/5120x<k>` | the small-T Q4 residual projections (`small_t_c8`, `small_t_c16`, `small_t_c32`) for the attention or GDN output (k = 6144) and the MLP down (k = 17408) of a Q4-output artifact | widths 1 to 32 |
| `q6_head/248320x5120` | the Q6 vocabulary head's per-row GEMV (`gemv`, widths 1 and 2) or small-T MMA (`small_t`, widths up to 32) | widths 1 to 32 |
| `q8_ksplit/<rows>x<k>` | the four-K-warp form (`kw4`: at most 128 registers a thread, four CTAs an SM) of the 8-column tile of the Q8 G32 K-split contraction, which MTP heads, DFlash2 drafters and Q8 heads run, instead of the route's own schedule | the eleven contraction shapes of those layers, widths 1 to 8 |
| `unified/q4_q5_attn_input`, `unified/q4_q5_gdn_input`, `unified/q4_linear_swiglu`, `unified/q4_linear_add`, `unified/q5_linear_add/5120x<k>` | upstream's routes for the fused groupwise projections over the unified Linear templates (`unified`) instead of this line's own | the model shapes, widths 1 to 64 (the small-T launches only where those are all the switch covers) |
| `unified/nvfp4_attn_input`, `unified/nvfp4_gdn_input`, `unified/nvfp4_linear_add`, `unified/nvfp4_linear_swiglu` | the switch of the NVFP4 fused projections onto the unified Linear templates (`unified`) instead of this line's own chain. Under the A16 policies both chains are registered at every width a fused launch serves, so a key names a chain rather than a width | the model shapes, widths 1 to 1024. The two chains share the T = 1 decode kernel and, above the own small-T ceiling (32 columns; 16 for SwiGLU), the same wide fused A16 MMA, so what the key really decides is the small-T band: this line's own small-T kernels against the unified SIMT and sliced-K ladder |
| `nvfp4_int8/attn_input`, `nvfp4_int8/gdn_input`, `nvfp4_int8/linear_add`, `nvfp4_int8/linear_swiglu` | the same switch for the NVFP4 fused projections under `--nvfp4-w4a8-int`, where the activation is quantised to s8 and the stored E2M1 codes feed the s8 tensor cores. The band chooses the wide mainloop -- this line's own MMQ against the unified grouped ladder -- while T <= 8 runs the matching row-form kernel (MMVQ, or the unified narrow one at the call's exact width) under either answer. Kept apart from the `unified/nvfp4_*` keys above so calibrating one family never overwrites the other | the model shapes, widths 1 to 1024 at all four keys: this family pads its token tile instead of stopping at a small-T ceiling, so the SwiGLU key covers the whole sweep too |
| `unified/q4_linear_topk`, `unified/q8_*`, `unified/fp8_*`, `unified/bf16_*`, `unified/context_kv_materialize` | the same switch for the Q4 top-k head, the Q8 fused projections (attention and GDN inputs, LinearAdd, pair, SwiGLU, top-k, grouped convolution, context-KV materialization), the FP8 fused projections (attention and GDN inputs with their conv forms, LinearAdd, SwiGLU, the FP8 top-k), and the BF16 attention input and LinearAdd | not calibrated: the built-in RTX 3090, 4090 and 5090 profiles carry the widths where each Op's benchmark ran faster on the unified routes over two rounds per table (at least 3 % over the geometric mean, no case more than 3 % slower), and keep them under a calibrated file entry; elsewhere a hand-edited profile entry or `NINFER_LINEAR_ROUTES` sets them. Those benchmarks ran to width 1024: a wider FP8 call with A16 activations (every FP8 call of a build other than `120a`, which alone carries the FP8 A8 routes) takes the unified LinearAdd and SwiGLU, and the unified pure Linear, where no profile band reaches its width, because the legacy A16 routes run it in 4-token passes that each read the whole weight |
| `w4_cublas/<op>`, `w4_cublas/<op>/a8` | with `--prefill-cublas`, whether a prompt call of each projection (`q4_linear_swiglu`, `linear_add/5120x17408`, `linear_add/5120x6144`, `q4_q5_gdn_input`, `q4_q5_attn_input`) takes the cuBLAS route (`on`, `off`): the plain key at widths the integer A8 mainloop cannot take, `/a8` at the multiples of its 128-column tile. Where no band reaches a width, calls of 512 columns and more take the route; no band sends a call of 128 columns or fewer (a decode or verify round) to it | widths 136 to 1,500, and 256 to 2,048 under `/a8` |
| `gdn_two_stage/h<value heads>` | the two-stage GDN prefill (fused Q/K normalization and control preparation, then one FP32-state recurrence that also writes the output) instead of the WY/state-passing/output pipeline | prompts of 16 to 8192 tokens, 48 and 32 value heads |

`NINFER_SMALLT_PV_F16`, `NINFER_PROMPT_PV_F16`, `NINFER_PROMPT_PACK_GQA`, `NINFER_PROMPT_FAST` and
`NINFER_GDN_TWO_STAGE` (`0` or `1`) override the profile for their route; `NINFER_PROMPT_SPLITS=0`
turns the prompt key splits off and `N` forces N splits wherever a call may split; `NINFER_LINEAR_ROUTES`
(`legacy` or `unified`) takes one table for every Linear shape and every `unified/*` switch. `--fast-prefill-kernel` turns the fast prompt kernel on regardless.

The scoped `int8:legacy|unified` forms affect only NVFP4 W4A8; the `a16:` forms
affect only NVFP4 A16/A4. Plain NVFP4 W4A8 Linear defaults to Unified at every
width, including MTP T=1, because Legacy has no implementation for that policy.
An explicit Legacy override is rejected by both planning and execution.

## Batch composition

A profile picks each width's fastest schedule, and schedules differ in the order in which they sum.
A step that serves four requests runs the projections at width four and splits the attention
differently from a step that serves one, so a request's numbers depend slightly on what shares its
steps, and a greedy answer can change where its two most likely tokens are nearly tied. In a check
on an RTX 5090 (Ternary Bonsai 2, four lanes), one request sent four times at once matched its lone
answer in one copy of four with the built-in profile, and in all four with `--device-profile off`
and `NINFER_PREFILL_ALIGN=0` (the compiled routes and requested chunks) or without the profile's ternary small-T
routes; four different documents at once matched their lone answers in one or two of four with the
profile and in three or four without it. The answers that differ part at such a word and stay
equivalent: four 60,000-token needle documents served at once returned all twelve codes in order,
with and without speculation, on `rk8v4` and `rk4v4`. Where answers should depend on batch
composition as little as possible, run with `--device-profile off` and `NINFER_PREFILL_ALIGN=0`.

## Sizes derived from the device

Some launch sizes follow the device directly and need no profile:

- **Prefill chunk.** `--prefill-chunk` is adjusted to the nearby multiple of 128 whose
  prompt-attention grid leaves the least of its last wave idle on this GPU (see
  `causal_softmax_attention_prompt_aligned_chunk`); `NINFER_PREFILL_ALIGN=0` keeps the requested
  chunk.
- **Wave counts.** The chunked GDN output kernel and the sparse-MoE prefill size their grids from the
  SM count and the kernel's occupancy, and the small-T attention holds its split count to whole waves
  of the device.

## SparseMoe policy keys

SparseMoe, with resident or streamed experts, reads two optional keys outside automatic
calibration:

- `sparse_moe/prefill/<codec>` (`q4-q5`, `q4-q6`, `q8-q8`, `nvfp4`): a band ending at 1 stores the
  token count from which calls take the prefill route instead of Small-T. Accepted values are 2 to
  47 (13 to 47 for NVFP4); anything else keeps the compiled frontier. The workspace query and the
  launch read the same value.
- `sparse_moe/prefill-tiles/q4-q5` and `/q4-q6`, with T as the axis: `narrow` or `wide` forces the
  prefill tile width; an empty schedule keeps the automatic choice.

The built-in profiles use two of them. The V100-SXM2-32GB takes Small-T up to T=32 for Q8 experts
(frontier 33): eight Qwen3.6-35B-A3B requests with three MTP drafts and 512-token prompts produced
2.2% more tokens per second with identical outputs. The RTX 5090 takes the narrow Q4+Q5 tile at
T=1024 (the default prompt chunk): whole generation ran 0.5% faster with identical outputs.
`ninfer_sparse_moe_bench --candidate` and `ninfer_sparse_moe_test --candidates-only` compare and
qualify the candidates a profile may name.
