# Quality trades

The precision options of `ninfer`, `ninfer-serve` and `ninfer-perplexity`: what each stores or
computes differently, what it costs in quality and what it buys. All are off by default except the
integer prefill routes, which `--no-prefill-a8` turns off. Unless a row says otherwise the
measurements are Qwen3.8-27B (`groupwise-int`) on one RTX 3090, perplexity from
`ninfer-perplexity --quick` on `ninfer-ppl-1m-v1` (context 4,096, stride 2,048).

| option | what changes | quality | gain |
|---|---|---|---|
| `--embedding-q4` | token embedding stored as Q4 G64 at load | 4.346413 → 4.343738 (−0.062%, noise) | −644 MiB of weights, about 24K tokens of `rk8v4` context |
| `--embedding-q6` | token embedding stored as Q6 G64 | Qwen3.6-35B-A3B: +0.035% | −341 MiB (27B), −143 MiB (35B-A3B) |
| `--lm-head-q6` | output head stored as Q6 G64 | +0.012% | −341 MiB, about 13K tokens |
| `--lm-head-q4` | output head stored as Q4 G64 | **+0.693%**, greedy text parts within ~50 tokens | −644 MiB; +3% aggregate decode at eight lanes |
| `--mtp-experts-q4` | Qwen3.6-35B-A3B: the MTP layer's routed experts in the text layers' formats | none scored (drafts are verified); acceptance 61.2% → 60.6% | about −350 MiB, two rungs of context |
| `--gdn-state-fp16` | GDN recurrent state stored as FP16 between steps, computed in FP32 | no perplexity change; per-step state drift at most 0.34% | halves every StateImage (27B: 147 → 74.5 MiB); +2% aggregate decode at eight lanes |
| `--prefill-cublas` | wide prefill GEMMs as int8 weights with one scale per row through cuBLAS | +0.156% (+0.072% with `--no-prefill-cublas-projections`) | prefill 43-83% faster; workspace 155.6 → 661.2 MiB at chunk 4096 |
| `--mlp-a8-decode` | MLP gate/up at 16-32 verify columns with int8 activations | oracle bound 0.008-0.037 relative L2; perplexity cannot see it | +1.3% aggregate decode at eight lanes |
| `--no-prefill-a8` | returns full prefill tiles to their A16 routes | the integer routes cost +0.004% | the integer routes prefill 49-67% faster |

`--embedding-q4` and `--embedding-q6`, and `--lm-head-q4` and `--lm-head-q6`, are mutually
exclusive. A Q6 flag is a no-op where the artifact already stores Q6 (the Qwen3.6-27B embedding and
head, the Qwen3.6-35B-A3B head; the head flags are rejected on that model).

## Vocabulary transcoding

`src/artifact/transcode.{h,cpp}` requantizes a row-split `q8_g32_fp16` payload into a narrower
group-64 row-split encoding on the host while the weights load. The binder plans the target
encoding's size, so the arena shrinks, and the materializer writes the target bytes before anything
else (bindings, the overlay eviction mirror) observes the tensor. Each group's FP16 scale is the
best of 25 clipping ratios of `absmax / qmax` in [0.70, 1.18], so a group is never worse than
round-to-nearest. It is multi-threaded and adds a few seconds to load.

- The embedding of the 27B quantizes for free; the 2048-wide embedding of the 35B-A3B does not:
  `--embedding-q4` costs it +0.36%, about what NVFP4 KV costs, so `--embedding-q6` is the option
  there.
- `--lm-head-q4` and `--lm-head-q6` are rejected with DFlash and DFlash2, whose candidate top-k reads
  the stored head directly.
- Measured before the Q6 vocabulary-head small-T kernels existed, `--lm-head-q6` cost 2-5% of
  one-lane decode; the device profiles now route the Q6 head, and that cost has not been measured
  again.
- Overlay Vision borrows its exclusive encode window from the evict-ranked weights (head,
  embedding, draft head, MTP), so they must cover one window. On the 35B-A3B a smaller embedding
  can fall below the 960 MiB window of `--vision-max-merged 16384`; startup then refuses and names
  the ways out (a lower `--vision-max-merged`, `--vision-residency resident`, or dropping a flag).
  The eviction mirror is captured after transcoding, so overlay windows restore transcoded bytes.

Qualification: `ninfer_vocabulary_transcode_test` checks the codec against an independent decoder
and routes transcoded 248320x5120 heads through `ops::linear` at T = 1..32; `ninfer_embedding_test`
qualifies the Q4 gather at [248320,5120] and [248320,2048] against an FP64 oracle, CUDA Graph
replay included.

## FP16 GDN state

The 27B's GDN recurrent state is 48 layers x 48 heads x 128 x 128 FP32, 147 MiB per sequence slot,
read and written every round. With `--gdn-state-fp16` the recurrence still computes in FP32 and only
the stored state between steps is rounded: the four recurrent kernels (direct, batch update,
record, replay fold) are templated on the storage type, and the chunked prefill kernels keep their
FP32 interface with the wrapper staging FP16 through an FP32 workspace.

Perplexity scores through prefill, which rounds the state only at chunk boundaries, so it cannot
clear this option alone. `ninfer_gdn_state_fp16_test` runs 4,096 synthetic decode steps through the
real Op with FP16 against FP32 state: the worst relative error is 0.339% at width 1 and 0.219% at
width 4, flat over the run rather than growing. Greedy output on a real prompt was byte-identical.

## Integer activations

The integer prefill routes (`src/ops/common/rowsplit_a8_mma.cuh`) quantize activations per token
and 64-column group and run `mma.m16n8k32.s8`; they are on by default because their cost is inside
run-to-run noise (+0.004%) and their FP64 oracle bound is 0.010-0.020 relative L2 against the
0.04 allowance A8 activation compute is held to. A per-token scale, as Marlin-class kernels use,
would be about 20% faster but reaches 0.129 relative L2 on outlier-heavy inputs.

`--prefill-cublas` goes further for wide prompts: the weights are materialized as int8 with one
scale per row and the GEMM runs in cuBLAS. cuBLAS reduces over the whole of K, so the per-group
scales are lost; equalizing the activation channels (a per-channel scale folded exactly into the
weights) is what keeps the cost at +0.072% for the MLP and output projections instead of +0.474%.
It wants a large `--prefill-chunk`, since its dequantization pass is weight-sized and its GEMM
token-sized; the trade-off per chunk is in `src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`.

`--mlp-a8-decode` extends the integer gate/up route to the 16-32 columns a batched verification
round reaches (eight lanes of four MTP columns is 32); narrower widths stay on the BF16 small-T
kernel, which is faster there. Perplexity scores the prefill phase and never reaches this
verify-phase route, so its evidence is the oracle bound and output comparison at eight lanes. The
kernel's measurements and rejected variants are in the header of
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`.
