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
2. **The built-in table.** Profiles measured on the RTX 3090, RTX 4090, RTX 5090 and the RTX PRO 6000
   Blackwell ship inside the binary (`src/runtime/engine/device_profiles.json`). The PRO 6000's
   Workstation (600 W), Max-Q (300 W) and Server editions report different names, so each has its
   own entry.
3. **The compiled route** of each operation, for a key neither names.

For a key both the file and the built-in table name, the file's bands answer up to their last width
and the built-in bands past it. A key the file does not name keeps its built-in bands, so calibrating
an RTX 3090, 4090 or 5090 replaces the keys calibration measures and keeps every other built-in key
(the `unified/*` switches calibration does not measure, below). A key calibration measured and found
the compiled route best is stored with an empty schedule, which overrides a built-in choice for it.

A GPU with neither a file entry nor a built-in one is calibrated once when the engine starts, before
the weights are loaded. This takes 20 to 40 seconds, and the result is written to the profile file
for later starts.

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
| `attn_i8_small/h24/<kv>/w<W>` | warps, CTAs per SM, key block, split QK (`q`) and early fetch (`e`) of the INT8-family small-T attention | query widths 1 to 8, KV windows 8K, 64K and 262K, for `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` |
| `attn_pv_f16` | FP16 accumulation of the probability-times-value product per key tile | a 1024-token prompt chunk at 32K and a decode step at 131K |
| `attn_pack_gqa` | the standard INT8 prompt kernel with each KV head's query heads packed into its tiles (PackGQA) | a 1024-token prompt chunk at 32K and 131K |
| `attn_prompt_fast` | the fast prompt-attention kernel (rows kept in registers, FP16 PV per tile) | a wave-aligned prompt chunk at 32K and 131K |
| `attn_parallel_tiles` | single-row chunked small-T attention over an INT8-family cache as parallel query tiles (one batched append, one split-KV launch over every tile, one reduce) instead of serial fused chunks; widths without an exact 2-8 column tile divisor stay serial; `NINFER_ATTN_PARALLEL_TILES=0\|1` overrides | 16 and 32 verify columns at 32K and 131K |
| `q4_linear_add/5120x<k>` | the small-T Q4 residual projections (`small_t_c8`, `small_t_c16`, `small_t_c32`) for the attention or GDN output (k = 6144) and the MLP down (k = 17408) of a Q4-output artifact | widths 1 to 32 |
| `q6_head/248320x5120` | the Q6 vocabulary head's per-row GEMV (`gemv`, widths 1 and 2) or small-T MMA (`small_t`, widths up to 32) | widths 1 to 32 |
| `unified/q4_q5_attn_input`, `unified/q4_q5_gdn_input`, `unified/q4_linear_swiglu`, `unified/q4_linear_add`, `unified/q5_linear_add/5120x<k>` | upstream's routes for the fused groupwise projections over the unified Linear templates (`unified`) instead of this line's own | the model shapes, widths 1 to 64 (the small-T launches only where those are all the switch covers) |
| `unified/q4_linear_topk`, `unified/q8_*`, `unified/fp8_*`, `unified/nvfp4_*`, `unified/bf16_*`, `unified/context_kv_materialize` | the same switch for the Q4 top-k head, the Q8 fused projections (attention and GDN inputs, LinearAdd, pair, SwiGLU, top-k, grouped convolution, context-KV materialization), the FP8 and NVFP4 fused projections (attention and GDN inputs with their conv forms, LinearAdd, SwiGLU, the FP8 top-k), and the BF16 attention input and LinearAdd | not calibrated: the built-in RTX 3090, 4090 and 5090 profiles carry the widths where each Op's benchmark ran faster on the unified routes over two rounds per table (at least 3 % over the geometric mean, no case more than 3 % slower), and keep them under a calibrated file entry; elsewhere a hand-edited profile entry or `NINFER_LINEAR_ROUTES` sets them |
| `gdn_two_stage/h<value heads>` | the two-stage GDN prefill (fused Q/K normalization and control preparation, then one FP32-state recurrence that also writes the output) instead of the WY/state-passing/output pipeline | prompts of 16 to 8192 tokens, 48 and 32 value heads |

`NINFER_SMALLT_PV_F16`, `NINFER_PROMPT_PV_F16`, `NINFER_PROMPT_PACK_GQA`, `NINFER_PROMPT_FAST` and
`NINFER_GDN_TWO_STAGE` (`0` or `1`) override the profile for their route; `NINFER_LINEAR_ROUTES`
(`legacy` or `unified`) takes one table for every Linear shape and every `unified/*` switch. `--fast-prefill-kernel` turns the fast prompt kernel on regardless.

## Batch composition

A profile picks each width's fastest schedule, and schedules differ in the order in which they sum.
A step that serves four requests runs the projections at width four and splits the attention
differently from a step that serves one, so a request's numbers depend slightly on what shares its
steps, and a greedy answer can change where its two most likely tokens are nearly tied. In a check
on an RTX 5090 (Ternary Bonsai 2, four lanes), one request sent four times at once matched its lone
answer in one copy of four with the built-in profile, and in all four with `--device-profile off`
and `NINFER_PREFILL_ALIGN=0` (as on the previous master) or without the profile's ternary small-T
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
