# Multi-GPU: layer pipeline stages

`--devices A,B,...` splits the model's layers into one pipeline stage per GPU. Each stage owns its
layers whole: weights, the KV cache of its attention layers, the recurrent state of its GDN layers,
and the scratch it runs in. The point is memory: a model that does not fit one card, or a context
that does not, spreads across several with every card's memory usable for KV.

## What it is and is not

**It is a memory feature.** The stages run in sequence: while one computes, the others wait.
Decode is weight-bandwidth-bound and each stage reads only its own weights, so a single stream is
about as fast as one GPU, minus the boundary hops. Splitting the batch into micro-batches would make
each stage read its weights once per micro-batch and win nothing.

**It is not tensor parallelism.** No layer is divided across devices, so no collective runs inside a
layer. `--tensor-parallel` divides them instead; see [Tensor parallelism](#tensor-parallelism).

**Linux and Windows.** Repeating one device id (`--devices 0,0`) exercises the whole stage path on
a single card, which is how the path is tested without a second GPU. Under Windows a GeForce card
runs the WDDM driver model, which gives no peer access, so the stages exchange their boundary rows
through pinned host memory (`StageLink`), as Linux does between cards without peer access. WDDM
charges pinned host memory against a card's memory (measured on one card); assuming it charges
every card of the pipeline, the pinned host KV cache is sized for the rank with the least device
memory free. This repository has no Windows host with several GPUs: the Windows path is compiled in
CI and was run by a contributor, not measured here.

**The cards may differ.** Every listed device must run the build's code, which `DeviceContext`
(`core/device.cu`) checks before any weight is uploaded by loading a kernel on each, and the model
family checks each device's compute capability. An `86` build therefore pipelines RTX 30 with RTX
40 cards, and an `80` build, whose sm_80 code also runs on 8.6 and 8.9, an A100 with either. A build
of several architectures (`70;75;80;86;89`, `75;80;86;89;120a`) pipelines cards of different code
tiers, such as a V100 with a TITAN RTX and RTX 30 cards. Each rank runs the code it would run alone:
its tier's kernels, shapes and routes (`ops/common/device_tier.h`), the weight layout and
activation permissions of its tier (`apply_device_layouts`, `apply_rank_policies`), and the route
profile of its own card. The context-cost model of the prefix cache is the first device's.

## How it works

- **`StagePlan`** (`core/stage_plan.h`) says which contiguous layers each stage owns. `--stage-layers
  30,34` sets the counts. Without it `default_stage_layers` (`models/qwen3_5/load.cpp`) sizes each
  layer from the artifact and calls `solve_stage_plan`, an exact memory-balancing solver (maximum
  page groups every stage can hold, then least-full stage), verified against a brute-force
  partition oracle. It works from the bytes free on each device, the stored KV page-group and
  per-slot GDN state bytes of each layer (from the engine's KV storage format and slot count), and
  rough constants for what a stage needs beyond weights (context, workspace, graphs; more on rank
  0 for the head), so it is a good default rather than exact; if it finds nothing fits
  it deals the layers evenly and lets the Program's planning report the device that runs out.
- **Placement.** `bind_text` places every layer's weights on its stage's device
  (`Bindings::place`, recorded for rank 0 as well so an object shared between a rank-0 layer and a
  later stage's layer is refused as a conflicting placement). The embedding, final norm and head stay
  on rank 0.
- **The head stays on rank 0.** The last stage sends the residual back; rank 0 then runs the final
  norm, head and sampling exactly as on one GPU. That costs one extra ~20 us hop per forward pass and
  keeps the round buffers, prefill buffers and every other head-side structure where they are.
- **Per-rank state.** `DeviceKVPagePool` is one page-group allocator over planes bound to several
  ranks' backings, so admission, prefix reuse and the context-cache policy never see how many devices
  hold the cache. Block tables are one host shadow with a device copy per attention-bearing rank
  (`KVExecutionTablePool`); GDN state is one `LinearAttentionStatePool` shard per stage
  (`StateImageDevicePool`); the host image of a state keeps the whole model's layers at their global
  offsets. Data movement takes `RankStreams`, so each plane's copy goes on its own rank's stream, and a
  single stream given to a multi-rank object fails instead of using the wrong device.
- **Capacity.** Each further device has its own affine reservation curve
  (`RankCapacityCurve`); the resolved page-group count is the smallest any device allows, and a
  failure names the device that could not hold the plan. Ranks sharing one physical device split its
  free memory.
- **`StageLink`** (`core/stage_link.h`) moves the residual between stages, and the small control
  tensors (positions, KV rows, slot indices, valid columns) the layers read from rank 0, through a
  ring of pinned-host slots. The copies are ordinary graph nodes. Decode captures the whole pass into
  one multi-device graph, so `StageLink` tracks, per event, which capture made its last record: a wait
  inside a capture may only target an event recorded in that capture, and an eager wait may not target
  one whose last record was in a capture (`cudaErrorInvalidValue`, even after the graph has
  launched).
- **`TextContext::run_staged`** packs the control tensors on rank 0, sends each stage its block up
  front, runs stage 0, and for each later stage rebinds the context's device, stream and workspace
  (`ScopedDeviceRank`, `ScopedArenaRank`), receives the residual and control block, runs the stage's
  layers and passes the residual on.

## What is not covered yet

- **Prefill does not overlap stages.** A prefill chunk runs through the stages in turn and the
  engine synchronizes after each chunk, so at any moment one stage is busy. Overlapping stages needs
  micro-chunks inside a chunk (later stages start on micro-chunk 0 while stage 0 runs micro-chunk 1);
  that changes the chunk shapes the kernels see, and its benefit can only be measured on real cards.
- **Vision works** across stages with resident or CPU residency. The tower, the encode workspace
  and the scatter of the image's embeddings into the prompt stay on rank 0 with the embedding, before
  the layers run; the only thing later stages need is the multimodal RoPE positions, three rows per
  column, which travel in the control block (sized for them: eight int32 per boundary column). The
  default split counts a resident tower on the head stage. Overlay residency stays refused at load,
  before the weights: it borrows its encode window from the evict-ranked weights, and a later stage
  holds whole layers that nothing could lend.
- **MTP works** across stages: the replay records and their fold are per state shard, on the shard's
  own device, and the draft layer and head stay on rank 0. **DFlash and DFlash2 work** too; see
  below. **The context cache works**: its transactions copy each rank's planes and state shards on
  that rank's streams.

### DFlash across stages

A masked drafter reads target features, the outputs of a few layers spread through the model:
Qwen3.8-27B's DFlash2 adapter taps layers 5, 19, 33, 47 and 61 of 64 at width 5,120, the 35B-A3B
DFlash adapter layers 1, 6, 11, 16, 22, 27, 32 and 37 of 40 at width 2,048. The drafter, its context
KV and the feature buffers live on rank 0 with the head, so a tapped layer on a later stage sends its
output there, the way the residual travels.

1. **A link per tapped layer on a later stage** (`StageRuntime::features`), from that stage to rank
   0, built beside `back` in `make_stage_runtime`: a `StageLink` of two slots, each holding the
   widest pass (`stage_boundary_columns`) in BF16, plus a stream of the stage's own and the events
   that hand work to it and back.
2. **Capture on the stage.** As a tapped layer finishes, a stage-local tap copies its output into
   that tap's slab of a staging tensor in the stage's workspace and starts the transfer on the
   feature stream, so it crosses while the stage's later layers run. The stage's stream takes the
   feature stream back after the residual has left, which keeps the staging alive until every
   copy is done.
3. **Re-capture on rank 0.** Rank 0 has nothing to do until the residual returns, so `run_staged`
   queues the receives there first: each tapped output lands in one residual's room on rank 0's
   stream and goes to the sink's own `capture_layer`, so the prefill copy, the batch scatter and the
   check that every tapped layer was published are the ones one device runs. Stage 0's taps go to the
   sink directly.
4. **Graphs and planning.** The links' send and receive halves are capture-safe and the feature
   stream joins the capture through its events, so the DFlash decode graphs capture the extra copies
   with the residual's. The workspace plan reserves the staging slabs on each later stage and one
   receive slab on rank 0, sized for the stage with the most taps, and the default split counts the
   drafter's weights on the head stage, as it counts MTP's layer.

Cost, for a 30/34 split of the 27B, which puts the taps 33, 47 and 61 on the second stage: 30 KiB per
column, so 240 KiB for an eight-column DFlash2 verify and 30 MiB for the default 1,024-token prefill
chunk, sent while the stage's later layers run; only what is still in flight when the stage's last
layer finishes adds to the pass. Each tap's two slots pin twice the widest pass in host memory, 60 MiB
for those three taps at that chunk, beside the residual links' own slots.

## Verification

- `ninfer_multi_gpu_test`, three suites in one executable (ranks share device 0, the staged protocol
  forced): the stage link (byte integrity across sizes and ring reuse, a two-hop chain, payload
  halves through separate graphs, the slot-reuse dependency read off the graph, eager use of a slot
  before and after a capture), the multi-rank KV pool and block-table copies against the single-rank
  pool as the oracle in both plane orders, and the stage solver against a brute-force partition
  enumeration.
- `ninfer_kv_capacity_test`: the per-device curve, including a device without KV and errors that
  name the device.
- `ninfer_qwen3_5_stages_real_test` (set `NINFER_TEST_ARTIFACT`): greedy output with `--devices 0,0`
  and `0,0,0` equals `--device 0` byte for byte, since the layers run the same kernels on the same
  data. Rows: graphs and eager, forced pinned-host transport, an uneven split
  (`NINFER_TEST_STAGE_LAYERS`, 20,44 by default), MTP, DFlash or DFlash2 (whichever drafter the
  artifact carries, each compared with the same backend on one device), Vision plain and with MTP
  (an image prompt spanning two prefill chunks, when the artifact has a tower), and a continuation
  that reuses the context cache. `NINFER_TEST_DEVICE_IDS=0,1` puts the stages on real cards.
- `ninfer_qwen3_5_loading_real_test`: the default split covers the model, gives the head stage
  fewer layers, and gives a device with twice the memory more.

Ranks sharing a device cannot show a wrong-device pointer: a stale pointer into "another rank's"
memory still works there. That class of bug needs a real second card.

## Measured on two real GPUs (2026-09-21)

Rented Linux boxes, CUDA 12.8 build, Qwen3.6-27B groupwise-int (the pinned Hugging Face artifact),
int8 KV. Neither box has peer access, so every boundary transfer goes through pinned host memory.

| | 2x RTX 3090 (PCIe 3.0 x16 each) | 2x RTX A4000 (x8 + x16, cross-NUMA) |
|---|---|---|
| `ninfer_qwen3_5_stages_real_test` | every row byte-identical to one device: 2 stages, 3 stages (devices 0,1,0), graphs and eager, forced staged transport, uneven split, MTP, Vision (plain and MTP, an image over two prefill chunks), prefix reuse | the model does not fit one 16 GB card, so every row is compared with the first: identical across cut points, graphs and eager, staged transport, MTP, prefix reuse |
| CLI, prefill chunk 1024, splits 20/44, 44/20, 32/32 and the default | output identical to one device | output identical across splits, and identical to the 3090 output |
| Decode, plain | 46.9 tok/s on one card, 48.5 split | 24.1 tok/s |
| Decode, MTP3 | 100.2 tok/s on one card, 105.3 split | 52.6 tok/s |
| Prefill | 1.56k tok/s on one card and split | 825 tok/s |
| Weights per card (default split) | 15.3 GiB on one card, 8.35 GiB on the first split card | 7.71 GiB |
| `--kv-capacity auto` at `--max-context 262144` | one card: refused (minimum reservation 9.2 GB, 8.76 GB free after weights); split: 262,144 tokens, 11.2 GiB free afterwards | 262,144 tokens, 3.9 GiB free afterwards |

A `ninfer-serve` check on the 3090 pair (four lanes, concurrent and repeated chat requests, prefix
cache on) gave the same output split as on one card for every request run alone, run in parallel
once, and repeated; a second concurrent round differed on one request, and one card differs from
itself the same way, because lane batching changes the arithmetic.

What this says: the split costs nothing measurable at decode on PCIe 3.0 x16 (one 20 KB residual
crossing per stage boundary per token is about 25 us against a ~21 ms step), and it is what makes the
27B's full 262,144-token context fit at all. It does not speed anything up: prefill is the same and
decode is within noise. The A4000 pair runs the 27B with its full context and its decode rate matches
the plan's estimate for a pipeline on 448 GB/s cards.

On a card with fewer SMs than the tuned route tables assume (the A4000 has 48), a GDN gating
schedule whose cooperative grid would not be resident falls through to the first later route that
fits the device; its speed there has not been tuned.

Later checks on real pairs, none with peer access: Ternary Bonsai 2 27B on two RTX 4090s generates
byte for byte what one GPU does in every row of the stage test, and suspends and resumes with its
prefix reuse intact; Qwen3.8-Flash-Next runs its experts over two RTX 3090 Ti, two RTX 4090 and two
RTX 5090 ([measurements](../qwen3-8-flash-next.md#measurements)).

### DFlash across stages, measured (2026-10-06)

Rented Linux pairs, none with peer access, so every hop goes through pinned host memory; CUDA 12.8
builds for `80`, `86` and `89`, CUDA 13.1 for `120a`; the pinned Qwen3.8-27B (DFlash2 bundle,
`1cbd84e7`) and Qwen3.6-35B-A3B (DFlash bundle, `ee449580`) artifacts; int8 KV.

`ninfer_qwen3_5_stages_real_test` with the stages on two cards (`NINFER_TEST_DEVICE_IDS=0,1`) passed
all 22 rows, 11 per artifact, on 2x RTX 3090 (sm_86), 2x RTX 4090 (sm_89), 2x RTX 5090 (sm_120a) and
2x A100 PCIe 40 GB (sm_80): two and three stages, graphs and eager, forced staged transport, an
uneven split, MTP, and DFlash2 or DFlash in two stages, staged transport, three stages eager and an
uneven split, each over a 1,300-token prompt in three chunks and a continuation served from the
context cache, byte-identical to the same backend on one card.

Speed with the CLI (`--greedy --no-thinking`, seven drafts): a 37-token chat prompt with 256 new
tokens ("short") and a 3,322-token prompt with 64 ("long"), tok/s on one card and then split over the
two by the default split, mean of two or three runs. Every split run generated the one-card text.

| | 2x RTX 3090 | 2x RTX 4090 | 2x RTX 5090 | 2x A100 PCIe |
|---|---|---|---|---|
| 27B DFlash2, short, decode | 206.1 / 223.2 | 264.8 / 264.8 | 416.4 / 418.4 | 242.5 / 233.8 |
| 27B DFlash2, long, prefill | 1,247 / 1,270 | 2,267 / 2,227 | 2,965 / 2,990 | 1,357 / 1,297 |
| 27B DFlash2, long, decode | 106.5 / 114.6 | 135.6 / 135.6 | 213.6 / 214.3 | 125.7 / 130.9 |
| 27B plain, short, decode | 47.9 / 48.9 | 55.8 / 55.5 | 91.7 / 91.0 | 52.3 / 54.7 |
| 35B-A3B DFlash, short, decode | 482.8 / 490.7 | 837.0 / 818.2 | 1,260 / 1,225 | 475.0 / 365.9 |
| 35B-A3B DFlash, long, prefill | 6,320 / 6,233 | 12,033 / 11,667 | 11,300 / 11,750 | 7,327 / 5,173 |
| 35B-A3B DFlash, long, decode | 195.8 / 205.5 | 343.0 / 337.3 | 471.7 / 462.2 | 199.2 / 197.2 |
| 35B-A3B plain, short, decode | 206.1 / 207.9 | 274.0 / 268.7 | not run | 207.6 / 181.9 |

The 3090 column was measured with taps sent together after a stage's last layer; sending each tap as
its layer finishes left decode equal within 1% on the 4090 and A100 pairs and raised the 35B-A3B's
long prefill from 11,433 to 11,667 tok/s (4090) and from 4,777 to 5,173 (A100). The weights per card in the default split were
10.3 + 7.74 GiB for the 27B with DFlash2 (18.0 GiB on one card) and 10.6 + 9.41 GiB for the 35B-A3B
(20.0 GiB); the 27B's workspace grew from 183.9 to 233.9 MiB per card for the boundary and the tap
staging.

What this says: on the GeForce pairs the split costs DFlash and DFlash2 no more than it costs plain
decoding, within about 3% either way (the 3090 pair is faster split, as it was on 2026-09-21), and
the features ride along at no measurable price at decode. The A100 pair is the exception, and not
because of DFlash: the 35B-A3B lost 12% split without speculation and 23% with DFlash. The same split
with both stages on one A100 (`--devices 0,0`, local or forced staged links) cost nothing (466-473
tok/s on one card, 476-481 split), each A100 alone ran at the same speed, and an nsys profile of the
split showed the kernels themselves running longer (the MoE path kernel +21%, the verify GDN layers
+46%) rather than gaps opening between them. That points at the cards' clocks while they wait for each
other (the two sit on different sockets of that host); GPU clock metrics are not permitted in the
container, so the cause is not confirmed.

## Tensor parallelism

`--devices A,B,... --tensor-parallel` divides every layer of a dense Qwen3.5-core model over the
devices instead of dealing layers out. It targets decode: each card reads only its share of the
weights for every token.

- **The shares.** Rank r holds GDN key heads `[r*Hk/N, (r+1)*Hk/N)` with the value heads that read
  them (the value-to-key head map is contiguous), their q/k/v/z projection rows, convolution
  channels, recurrent and convolution state, and the matching input columns of the GDN output
  projection; and rows `[r*I/N, (r+1)*I/N)` of the dense FFN's gate and up projections with those
  input columns of down. The gating projection, `A_log`, `dt_bias`, the gated norm and the layer
  norms are small, so every rank holds all of them: it gates every head and keeps its own.
  Attention, the KV cache, the embedding, the final norm, the head and MTP stay on rank 0.
- **Loading.** A share is a piece of a stored object: a row range and column ranges
  (`artifact/selection.h`), gathered from the file as byte runs at upload. Contiguous matrices and
  ggml blocks cut this way when column bounds fall on whole blocks; the stored objects themselves get
  no device demand. `tensor_parallel_share` (`models/qwen3_5/config.h`) states a rank's geometry.
- **The pass** (`TextContext::run_tensor_parallel`). Rank 0 sends the residual and the control
  tensors to every rank over the stage links; each rank then keeps a replica of the residual. Each
  layer has two exchanges (`core/tensor_parallel.h`): after the mixer (attention: rank 0 alone;
  GDN: every rank) and after the FFN. A contributor writes its output projection as FP32 partials;
  every rank then adds all partials to its replica in rank order and rounds once to BF16, so the
  replicas stay identical. Peer reads where every pair of cards has peer access, mapped pinned host
  memory otherwise; both capture into the decode graphs.
- **Numerics.** Against one device the only difference is the order of the FP32 additions of a
  projection's output: the shares' partial sums are added, then rounded once, where one device
  rounds the full sum. Logits are close, not identical, so greedy text can diverge where two tokens
  tie closely.
- **Coverage.** GGUF weights (the GDN input projection's fused GGUF route), the 27B geometry (16 key
  and 48 value heads, hidden 5120) on two or four ranks, MTP. Refused: MoE models, a masked drafter
  (DFlash, DFlash2), `--stage-layers`, Hadamard-rotated weights, Vision overlay, Flash-Next, Gemma 4
  and Flint. Native Q4/Q5/Q8/T2, NVFP4 and FP8 GDN projections have kernels registered only for the
  whole layer and are refused at preparation. Ranks may share a device (`--devices 0,0`), which is
  how the path is tested on one card.
- **Kernels at the share shapes**, registered next to the whole-layer ones and qualified by the op
  tests: the convolution split (`Rows1024x1024x3072`, `Rows512x512x1536`), the projected
  convolution (5120 and 2560 channels), the replay record and fold (8x24 and 4x12 heads).

Measured on Qwen3.8-27B GSQ-RCO IQ3_S (2026-10-25, `scripts/pods/serve_ab.py`, greedy, four short
prompts of 256 tokens, two passes and two server starts per configuration, plus one 7,014-token
prompt). Every run gave one device's answers.

- **2x RTX 3090** (Vast, PCIe 4.0 x8 links, no peer access, so the mapped host transport). Decode
  72.9-75.2 tok/s against 57.0-57.3 on one card (+27% to +31%) and 58.4-58.5 with two pipeline
  stages; 70.2 against 54.2 after the long prompt (+29%; the second server start, as the first
  overlapped a compile job). With `--spec mtp --draft-tokens 3` 105.5-160.3 against 89.0-134.6
  (+13% to +28%; between server starts the rate moved by up to 9%, and MTP acceptance shifts by a
  few drafts per answer because the hidden states round differently). Prefill pays for it: the
  7,014-token prompt runs at 1,187-1,292 tok/s against 1,495-1,643 (-21%), and a 40-token prompt's
  TTFT is 83-116 ms against 53-57 ms, since every layer of a chunk crosses the bus twice. Four
  ranks on the two cards (`--devices 0,1,0,1`) decode at 71-72 tok/s, and at 80.5-116.3 with MTP,
  below one card's 89-134: a check of the four-rank shares, not a four-card measurement.
- **2x Tesla V100-SXM2-32GB** (Vast, NVLink, so peer reads; the sm_70 routes). Decode 60.2-60.4
  tok/s against 43.9 on one card (+37%); with `NINFER_FORCE_STAGED_LINKS=1` the mapped host
  transport gives 54.6-55.0 (+25%). With MTP 83.4-122.3 against 58.1-87.1 (+29% to +44%). Over
  NVLink prefill gains as well: the 7,014-token prompt runs at 1,300 tok/s against 865-867 (+50%),
  decode after it at 58.3 against 42.9 (+36%), and a 40-token prompt's TTFT is 113-135 ms against
  129-130 ms.

What the transport costs was measured before this was built:

- **Transport** (`tools/tp_probe.cu`, rented boxes, 2026-09-21): neither a 2x RTX 3090 box (one slot
  PCIe 3.0 x4), a second 2x RTX 3090 box with both slots x16, nor a 2x RTX A4000 box had peer access
  or NVLink. The x16 pair measured 15 us (host-mapped kernel) to 24 us (staged, in a graph) for one
  column, 33-38 us for four, 75-78 us for sixteen, 130 us for thirty-two and 3.4 ms for a 20 MB
  prefill chunk, half the x4 box's time at every size. An all-reduce of one decode column
  (20 KB, FP32) costs 17-33 us whether done by a host-mapped kernel, a staged pinned-host copy or
  NCCL; at 640 KB it is 190-440 us and at 20 MB 5-13 ms. The transports are within about 20% of each
  other at every size, which is why the exchange is a fenced copy (or a peer read) and a local
  reduce that captures into graphs, not a spin-wait kernel.
