# Qwen3.5 model reference

This reference describes the Qwen3.5 Dense and MoE mathematics implemented by NInfer, including
Text, MTP, Vision and their state semantics. The Qwen3.6 and Qwen3.8 releases used by the official
artifacts are instances of these architectures. Their release names do not choose another graph.

[Engine architecture](engine-architecture.md) defines ownership and execution;
[artifact container](artifact-container.md) defines stored objects and bindings;
[weight conversion](../weight-conversion.md) explains source mapping and recipes.

## Architecture and configuration

The architecture registry recognizes these Text config pairs:

| `architectures[0]` | `model_type` | Feed-forward block |
|---|---|---|
| `Qwen3_5ForCausalLM` | `qwen3_5_text` | Dense SwiGLU |
| `Qwen3_5MoeForCausalLM` | `qwen3_5_moe_text` | Routed SwiGLU experts and a gated shared expert |

Code fixes the pre-norm residual order, gated GQA, zero-centered Text norms, GDN recurrence,
interleaved partial MRoPE, FFN mathematics and component relationships. Config supplies instance
dimensions and layer choices. It does not describe a graph of arbitrary operators.

[`config.cpp`](../../src/models/qwen3_5/config.cpp) defines the exact fields and validation.
Their responsibilities are:

| Config facts | Meaning |
|---|---|
| `hidden_size`, `vocab_size`, `num_hidden_layers`, `max_position_embeddings` | Mathematical extents |
| `layer_types` | One `full_attention` or `linear_attention` entry per Text block |
| `tie_word_embeddings`, `rms_norm_eps` | Training relationship and normalization constant |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | GQA dimensions; also used by MTP |
| `rope_parameters` | Theta, partial rotary factor and three interleaved MRoPE sections |
| `linear_num_*_heads`, `linear_*_head_dim`, `linear_conv_kernel_dim` | GDN heads, state dimensions and convolution width |
| `intermediate_size` | Dense FFN width |
| `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` | MoE dimensions |

Projection widths, compact attention/GDN layer indices, rotary dimensions and parameter shapes are
derived. The converter expands source shorthand such as a full-attention interval into the explicit
`layer_types` array. Source dtype, quantization metadata, checkpoint name and training-only fields
do not become mathematical config.

Text is required. Optional components target `text` and supply their private data:

- Vision supplies tower dimensions and patch/merge geometry. Its output width comes from Text.
- MTP identifies `Qwen3_5MTP` or `Qwen3_5MoeMTP`. The single-layer predictor inherits Text's
  attention, RoPE and FFN geometry and shares its embedding/output semantics.
- DFlash/DFlash2 supply private draft dimensions, attention pattern, conditioning taps and their
  algorithm parameters. See [DFlash](dflash.md).

Only selected components are semantically bound and materialized. A model's weights and config
remain immutable after loading; request capacity, concurrency, KV codecs and graph profiles are
runtime choices.

### Current official geometries

These are useful workload points, rather than a second config registry:

| Quantity | 27B Dense | 35B-A3B MoE |
|---|---:|---:|
| Hidden width H | 5120 | 2048 |
| Text layers | 64 | 40 |
| Full attention / GDN layers | 16 / 48 | 10 / 30 |
| Q heads / KV heads / head dimension | 24 / 4 / 256 | 16 / 2 / 256 |
| GDN key heads / value heads | 16 / 48 | 16 / 32 |
| GDN key/value head dimension | 128 / 128 | 128 / 128 |
| GDN convolution width | 4 | 4 |
| Dense intermediate width | 17408 | — |
| Routed experts / selected per token | — | 256 / 8 |
| Routed / shared intermediate width | — | 512 / 512 |
| Embedding/output matrix rows | 248320 | 248320 |
| Position capacity | 262144 | 262144 |

Both use `rms_norm_eps=1e-6`, `rope_theta=1e7`, `partial_rotary_factor=0.25` and
`mrope_section=[11,11,10]`. Their full-attention layers occur at indices `3,7,...`; that pattern is
stored in config. Both have untied embedding and output matrices. Qwen3.6-27B and Qwen3.8-27B share
this Dense geometry, while training results, representations and frontend semantics can differ.

Config parsing accepts structural variation. Actual execution remains bounded by implemented
Ops: the current fused input projections have explicit Dense/MoE shapes, GDN uses the implemented
128-dimensional recurrence, and the native sparse-MoE path uses 256 experts with top-8 routing.
The current multimodal RoPE entry requires the implemented interleaved axis mapping. These checks
belong to the relevant consumers; parsing a config is not a promise to execute every size.

## Logical parameters and physical bindings

[`load/`](../../src/models/qwen3_5/load/) expands the selected config into parameter names, shapes
and mathematical input Uses. The converter's
[`qwen3_5.py`](../../tools/convert/qwen3_5.py) maps source tensors into the same roles.

The main namespaces are `text/`, `vision/`, `mtp/`, `dflash/`, `dflash2/` and the optional
`proposal/`. Text block `i` contains `text/layers/i/attention/...` or `.../gdn/...`, plus its Dense
`mlp/...` or `moe/...` parameters. A projection such as `attention/query` has logical shape
`[num_attention_heads * head_dim, H]` and a Use at `text/layers/i/mixer_input`.

Bindings describe the logical values in physical parents. They can use complete objects,
consecutive regions or several ordered parts. The source `q_proj` stores each head's Q rows next
to that head's gate rows; source mapping extracts those rows into separate logical projections.
A simple split at the middle of the source matrix would change the model.

The Dense groupwise attention recipe packs Q/K into Q4 and gate/V into Q5. A supported FP8 or NVFP4
recipe can pack their union into one parent. The same four logical roles then become either the
two-weight or one-weight native fused call. Dense gate/up and MoE expert banks follow their finite
packing and execution rules. [Storage layouts](storage-layouts.md#9-logical-views-and-native-operands)
describes the difference between logical views and native operand restrictions.

`tie_word_embeddings` records a training relationship; bindings choose whether physical data is
shared. MTP and draft components reference Text's embedding/head rather than introduce private
copies by default. Shared weights keep independent Uses and activation permissions. The stored
NVFP4 weight divisor belongs to its parent; an activation divisor belongs to a mathematical input.

## Text block mathematics

In the formulas below, activations are written with feature dimensions first and token columns
last. H is hidden width, D is an attention head dimension, and T is the current token extent.
The two Text norm forms are:

```text
offset_rmsnorm(x,w) = (1 + w) * x / sqrt(mean(x^2) + eps)
plain_rmsnorm(x,w)  =       w  * x / sqrt(mean(x^2) + eps)
```

Text/MTP input, post-attention, final, stem and Q/K norms use the offset form. GDN's internal
output norm uses the plain form. Vision uses LayerNorm with weight and bias.

Every Text block follows:

```text
h = offset_rmsnorm(x, input_norm)
x = x + mixer(h)
h = offset_rmsnorm(x, post_attention_norm)
x = x + ffn(h)
```

The mixer is gated GQA or GDN according to `layer_types`. Dense FFN is
`down(SiLU(gate(h)) * up(h))`; MoE is defined below. Text/MTP projections and GDN convolution are
bias-free. Fusion may combine projection, activation, normalization and residual work while
preserving the declared mathematical and observable precision boundaries.

### Gated full attention

```text
q    = query(h)
gate = gate_projection(h)
k    = key(h)
v    = value(h)

q = offset_rmsnorm(q, query_norm)
k = offset_rmsnorm(k, key_norm)
q, k = partial_interleaved_mrope(q, k)
a = causal_gqa(q, k, v, scale=1/sqrt(D))
y = output(a * sigmoid(gate))
```

Q and gate have one D-dimensional vector per query head. K and V have one per KV head, with each
KV head serving a consecutive group of query heads. The gate is neither normalized nor rotated.
The logical cached values are normalized/rotated K and projected V. Prefill appends a chunk and
uses causal attention; decode appends one column and attends over the visible prefix.

Paging and KV quantization are storage choices. Their contracts are
[`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h),
[`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) and
[paged KV cache](paged-kv-cache.md).

### Gated DeltaNet

Normalized input produces Q, K, V, output gate Z and per-value-head A/B projections. Only Q/K/V
pass through depthwise causal convolution and SiLU. The convolution retains the preceding
`linear_conv_kernel_dim - 1` projected columns. GDN has no RoPE or position-ID input.

For value head j, use Q/K head `j // (value_heads / key_heads)`. Let the recurrent state be
`S[Dv,Dk]`. One token performs:

```text
g     = -exp(A_log) * softplus(a + dt_bias)
alpha = exp(g)
beta  = sigmoid(b)
k     = k / sqrt(sum(k^2) + 1e-6)
q     = q / sqrt(sum(q^2) + 1e-6)
Sbar  = alpha * S
u     = beta * (v - Sbar @ k)
S     = Sbar + u outer k
o     = (S @ q) * (1/sqrt(Dk))
y     = out_projection(plain_rmsnorm(o, gdn_norm) * SiLU(z))
```

The delta error uses the already-decayed state. GDN controls and recurrent state are FP32. The Op
receives raw BF16 q/k; normalization precision and the chunked/recurrent implementation are private
to its qualified route. Large prefill can use parallel state passing, while decode uses recurrence.

Ordinary decode updates the lane's current state. Speculative verification leaves persistent state
unchanged and writes convolution and recurrent records. After output resolution, one Fold applies
the committed prefix of that recorded physical block. The projection/convolution/record fusion
remains in the model's finite execution choices. [ReplaySSM](replayssm-gdn.md) defines the record
and commit semantics, including their numerical boundary.

### Sparse MoE

For token h, with E routed experts and K selected experts:

```text
router_probs = softmax(W_router h)
(values, ids) = topk(router_probs, K)
routing_weight = values / sum(values)

y_e = W_down[e](SiLU(W_gate[e] h) * (W_up[e] h))
y_routed = sum(routing_weight[e] * y_e for e in ids)
shared = W_shared_down(SiLU(W_shared_gate h) * (W_shared_up h))
y = y_routed + sigmoid(W_shared_score h) * shared
```

The shared expert always executes and is outside top-k. Logical expert id e identifies router row e
and the corresponding expert matrices. Physical bank ordering and fused dispatch preserve that
relationship. Router loss coefficients and training-only outputs add no inference term.


#### Routed expert residency

`--expert-residency device` retains the fused resident SparseMoe path. `host` retains the native
routed banks in pinned memory. `disk` leaves them in the artifact and records file extents for
each expert plane. Both offload modes use the common `core/expert_delivery` byte transport.
Shared experts and router weights remain resident and keep the same gated SwiGLU mathematics.
Dense Qwen models reject host/disk expert residency.

A Qwen slot has aliases for gate/up codes, high bits, scales and NVFP4 divisors, followed by the
same down planes. RowSplit Q4/Q5/Q6 use group-64 FP16 scales; Q8 uses group-32 scales. Q5/Q6 high
planes have independent strides. NVFP4 preserves whole 128-row swizzled scale tiles and each
source matrix's divisor. Delivery copies stored bytes without re-encoding. Disk sources can span
multiple objects and part files. Host sources currently require one contiguous span per plane;
a fragmented host plane is rejected rather than assembled into a second RAM bank.

Each Program owns one delivery service per pipeline rank, separate from the immutable Model sources.
Every SparseMoe call makes one delivery request for its layer. The service splits the call's routes
into experts the persistent cache holds and the rest. A host worker copies the rest of a narrow call
into cache slots, and the rest of a wide call into a whole-layer bank laid out like the resident
bank (expert e of each plane at a fixed stride), with one host copy per run of consecutive experts
within one pinned allocation. Two banks let the next layer of a wide call (half the layer's experts
or more: a prompt slice, a wide verification) arrive while the current layer computes; one bank
remains where memory is short.

Calls below the codec's prefill frontier (compiled at 47 columns for Q4/Q5 and Q4/Q6, 20 for Q8 and
13 for NVFP4; a device profile may move it, see [SparseMoe policy
keys](../device-profiles.md#sparsemoe-policy-keys)) route, run the cached pairs and the shared
expert while the rest arrive, then run the delivered pairs; both phases write per-rank FP32 products
that one rank-order sum reduces, adds to the residual and rounds to BF16. This is the resident
Small-T arithmetic; the Op test holds the streamed Q4/Q5, Q4/Q6 and Q8 outputs to the resident bits
at every tested call width. Wider calls take the resident grouped prefill route with per-expert raw
pointer addressing, after one delivery request per bounded native slice (up to 4096 token rows)
covers every selected expert. NVFP4 keeps its admitted W4A4 path. Grouped streaming does not take
the resident adaptive narrow-job shortcut; it takes the same frontier and tile width as resident
calls, so both run the same arithmetic. No routes are dropped or renormalized, and no weights are
re-encoded.

The persistent cache (`--expert-cache-mib`, default: what is free after startup) holds the experts
that decode and verification route most, by the expert tier every MoE family with host or disk
experts shares (`core/expert_tier`; [the expert cache](../qwen3-8-flash-next.md#the-expert-cache)):
a narrow call's miss takes a free slot, or the slot of the coldest expert the call does not route
when its routes beat that expert's, and every four decode passes a round swaps the most routed
missing experts in, copied behind the calls. A routing profile (`--expert-profile`, else the
built-in one of the model family where the build has one) fills the slots at startup. On one
device the CPU computes a share of a narrow call's misses from their host bytes (a disk expert's
read for the call) while the GPU runs the cached experts and the copies arrive
(`ops/sparse_moe/streamed_cpu`, `--expert-dma-share`, by default the split whose copies and CPU
products end soonest by earlier calls' times): each product row is the streamed d3/d4 kernels' warp
arithmetic reproduced lane by lane (fused multiply-adds in lane order, the shfl_down tree,
`ops::portable_silu`), so it is the GPU's bit for bit, and a placement kernel puts the rows into
their pairs' slots before the merge. A wide call evicts nothing; it promotes its
most routed experts from the bank into free slots by device copies, up to each layer's share of the
cache, so a prompt does not flush the decode working set. `--expert-cache-mib 0` keeps the banks
only. Between prompts the layer banks serve as cache slots (`core/expert_delivery`): once two
passes have run without a wide call, a narrow call's miss that the tier does not admit enters a free
slot of a lent bank, else the slot of its least routed expert the call does not route, and stays
there for later calls; a lent slot hits only in narrow calls, and the next wide call reads its
entries as misses and takes back the bank it fills. A bank is lent only when its slots number at
least half a layer's experts, so a narrow call always finds a slot it does not route. With automatic
KV, the cache has priority: KV takes the minimum legal runtime (one request's
full context) and the cache takes the remaining memory, because each cached expert saves bus traffic
on every token that routes to it. Explicit KV keeps its requested shape, and a cache that would
consume that reserved runtime is rejected. Delivery allocations are made before final KV capacity
selection. On one device the cache also grows into free KV pages (`core/kv_expert_loan`): the KV
arena is virtual memory, and after startup and after each commit the free pages no reservation
needs, beyond a margin per pool (8,192 tokens or two prompt chunks, whichever is more, plus a
Vision window), are loaned in units of at least 16,384 tokens from the highest free runs; the arena
releases the physical granules that lie entirely inside loaned pages and the cache takes them. A
reservation the free runs cannot cover recalls the lowest loaned pages, the cache first giving back
the bytes of the granules they need, and a model suspend recalls every loan before it copies the
live pages. Host and disk prompts default to 4096-token chunks: each chunk needs nearly every expert
of every layer, so a wider chunk moves the expert bank across the bus fewer times.

MTP has its own MoE source and gets every selected expert through the same full delivery path.
`--mtp-experts-q4` is an explicit exception: its existing load-time Q4/Q6 transcode keeps the MTP
expert bank resident. It does not create a cache-only draft policy or discard missing draft
experts. This combination retains that one layer's VRAM cost.

Delivery tables, banks and cache outlive all captured Graph users. The next request starts after
every consumer of the previous one. Suspendable Programs use the core VMM allocation mechanism:
suspend drains delivery and releases its physical backing, and restore remaps the same virtual
addresses and invalidates the cache before replay. Immutable pinned sources remain alive.
`RuntimeStats.experts` reports token-expert pairs, hits, admissions and transferred bytes, without
startup Graph warm-up deliveries. Transport diagnostics separately count distinct expert requests,
misses, bank reuses, prefetched layers, promotions, evictions, file reads and service time. The
memory summary reports the cache, the banks and table storage.

## Hadamard-rotated checkpoints

Some checkpoints, such as the ternary Bonsai releases, store a projection as `W * diag(s) * H`
per 1024-column block, where `H` is the normalized Sylvester Walsh-Hadamard matrix and `s` a
fixed sign vector per input width. The rotation spreads outliers so the rotated matrix quantizes
well; the product is unchanged when the activation is rotated the same way:

```text
y = W x = (W diag(s) H) (H diag(s) x) = W_r * hadamard_transform(x, s)
```

The artifact marks every such Use with a `hadamard_signs` auxiliary. The residual stream, the
norms, the KV cache, GDN state, Vision and every draft component stay in the primal basis; only
the inputs of rotated projections are transformed, and the model applies the transform where it
can share it:

- the mixer input once for the four attention or GDN input projections, never for the GDN A/B
  control projections, which read the same normalized input unrotated;
- the gated attention and GDN outputs before their output projections;
- the FFN input before gate/up, and SwiGLU's output through `silu_mul_hadamard` before down;
- the final hidden before every target, MTP, DFlash and proposal head call, on a copy where the
  primal hidden is read again (the DFlash2 candidate selector).

A rotated Use anywhere else, including MoE experts, is refused at load time. A token embedding
stored rotated is converted back to the primal basis by the converter.

## GGUF block checkpoints

An artifact converted with `qwen3_5_dense_gguf` stores its text projections, token table, output head
and MTP head in `gguf_*` formats, one ggml type per tensor, so the parts of a fused projection can
differ in type. The execution layer keeps each GGUF part as its own weight
(`ops::GgufProjectionWeights`): the GDN input projection lands q/k/v (output 0) and z (output 1)
rows, the attention projection query, gate, key and value, and parts that are contiguous rows of one
parent run as one product. FFN gate and up run as one `[gate; up]` product when they are one
parent, otherwise the gate writes an FP32 plane that the up product's epilogue multiplies by
`silu`. The MTP layer's projections stay separate unless they join the same way. The GDN output
projection reads its input through the Use's `input_columns` permutation (llama.cpp's tiled value
heads), applied while the activation is quantized.

## Prefill, decode and MTP

Text prefill gathers embedding columns, replaces media placeholders with Vision outputs, and runs
the configured blocks in chunks. It preserves state and positions across chunks. Final offset
RMSNorm and the full output head produce the target distribution. Ordinary decode repeats this
schedule for one new token. CUDA Graph replay uses the same selected model implementation.

MTP is a one-layer predictor conditioned on a final-normalized target hidden state h_t and the
following token x_(t+1):

```text
e = offset_rmsnorm(composed_embedding(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t, pre_fc_norm_hidden)
u = input_projection(concat(e, h))
u = one_gated_attention_block_with_target_ffn(u)
draft_hidden = offset_rmsnorm(u, mtp_final_norm)
draft_logits = target_output_head(draft_hidden)
```

The stem concatenates embedding first, hidden second. MTP has private stem, block, final norm and
KV state. Its attention and Dense/MoE geometry come from Text. It can recursively propose several
tokens using the previous MTP hidden output.

For prompt tokens x_0 through x_n and the target's first generated token x_(n+1), MTP prefill uses:

```text
token inputs    = [x_1, x_2, ..., x_n, x_(n+1)]
hidden inputs   = [h_0, h_1, ..., h_(n-1), h_n]
position inputs = [p_0, p_1, ..., p_(n-1), p_n]
```

Tokens shift; hidden states and positions do not. The whole aligned sequence populates MTP KV.
Shifted media placeholders use their corresponding composed Vision embedding columns.
[`mtp_alignment.h`](../../src/models/qwen3_5/program/speculative/mtp_alignment.h) and the prefill
implementation own chunk-boundary and continuation alignment.

An optional proposal head supplies an indexed vocabulary subset for draft prediction. Its row map
converts proposal rows to actual token IDs. Full target verification continues to use the full
output head. Backend selection, draft width and proposal-head choice are fixed at startup.

## Vision and multimodal positions

The current native processor uses 16×16 spatial patches, pairs of frames, and 2×2 spatial merge.
It consumes bytes already acquired by the CLI or server, decodes images/video frames, resizes to
merge-aligned dimensions, normalizes RGB and packs BF16 patch rows. Each row contains
`3 * 2 * 16 * 16 = 1536` values in the defined channel/temporal/spatial order. Images repeat their
frame to supply the temporal pair.

The official Vision towers have depth 27, hidden width 1152, intermediate width 4304 and 16 heads
of dimension 72. The learned position table is a 48×48 grid. Patch projection and interpolated
position addition precede the transformer stack. Each block performs:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_rope(q, k, two_dimensional_patch_positions)
a = segmented_attention(q, k, v)
x = x + output_projection(a) + bias
h = LayerNorm(x)
x = x + fc2(GELU_tanh(fc1(h) + bias)) + bias
```

Attention segments separate media/frame grids. The merger normalizes, groups 2×2 patches into
width 4608, and computes `fc2(GELU_exact(fc1(merged) + bias)) + bias`, producing Text-width columns.
Those columns replace matching placeholder embeddings. Vision has no autoregressive state.

Frontend admission uses aggregate media bytes, pixels, patch/token budgets and Engine context
capacity. The aggregate limit is 131,072 raw patches / 32,768 merged tokens; a single tower item
uses at most 16,384 merged tokens, additionally bounded by context. Program executes items
sequentially and reuses their output handoff after scattering the previous item.

The prepared prompt stores axis-major `[3,T]` positions in temporal, height, width order:

- Text tokens advance all three axes together.
- Media placeholders use their merged grid coordinates; following text resumes after the maximum
  multimodal coordinate.
- Video timestamps are rendered into prompt text. Position construction uses the prepared media
  grids and token runs.
- `rope_delta = next_multimodal_position - token_count` supplies the offset for subsequent decode.

For the official Text geometry, 64 of each 256-dimensional Q/K head rotate, using 32 frequency
pairs assigned to interleaved sections `[11,11,10]`. Text-only positions reduce to ordinary partial
RoPE because the three axes are equal.

## Vocabulary and frontend resources

Keep three row domains distinct: config `vocab_size` is the embedding/output matrix extent; the
tokenizer defines addressable token IDs; an optional proposal map defines its own shortlist rows.
The official tokenizer has 248,077 addressable IDs within 248,320 matrix rows. Remaining matrix
rows are not assumed zero and do not become public tokens.

Token spelling, special-token roles and EOS are derived from loaded resources. The frontend checks
the tokenizer's supported BPE/normalization semantics and token-domain agreement. Template bytes
select a compiled Qwen template; arbitrary Jinja execution is not implemented. Mode sampling
presets are architecture-owned, with explicit application/request overrides. Resource loading
does not infer execution identity from tokenizer filenames, release names or sampling values.

## State and numerical boundaries

Each StateImage contains the Text GDN state and continuation hidden. Per GDN layer, recurrent
state has one `[Dv,Dk]` matrix per value head, and convolution history has
`(2 * key_width + value_width) * (kernel_width - 1)` values. Program sizes Device/Host StateImage
capacity under the [context resource contract](resource-scheduling-and-context-cache.md#11-配置语义与有界性).
ReplaySSM records are separate pending-round scratch when speculation is enabled.

Text GQA KV grows with visible context. Selected MTP owns separate KV; DFlash backends own the
state described in their reference. Continuation hidden values and prefix checkpoints belong to
Program. Workspace, storage capacities and graph buffers are derived from actual parameters and
startup options by the planners described in [Engine architecture](engine-architecture.md).

Speculation verifies the candidate sequence against processed target distributions, commits the
accepted prefix and continuation state, then publishes output. Greedy acceptance compares a
proposal token with the target argmax for that verify column. Sampling uses the defined proposal
acceptance and correction distribution. This does not impose token or logits equality between
different quantization, prefill or kernel paths.

Each floating-point Op is qualified against its independent mathematical oracle over represented
public inputs. Packed weights are decoded with their stored scales. Exact transforms use exact
oracles; private accumulation, staging and reduction order use the Op's numerical criteria.
[Op development](op-development.md) owns that qualification contract.
