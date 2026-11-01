# Qwen3.8-Flash-Next: model mathematics

- **Target:** HF `Qwen/Qwen3.8-Flash-Next`, `architectures = ["Qwen4ExpForConditionalGeneration"]`,
  `model_type = "qwen4_exp"` (text `qwen4_exp_text`), checkpoint revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- **Implementation:** `src/models/qwen4_exp/` and the Engine core `src/runtime/engine/qwen4_exp_core.*`;
  the FP64 reference forward of the text path is `tools/reference/qwen4_exp.py`. Commands, options and
  measurements are in [the product guide](../qwen3-8-flash-next.md).
- **Authority order for the mathematics** (highest first):
  1. transformers `models/qwen4_exp` (PR #48337), the modular file and the generated modeling file -
     the only executable reference written by the model's owners' partners;
  2. vLLM `qwen4_exp` (PR #53896, the NVIDIA path) for MTP (transformers has no MTP) and for
     fused-kernel semantics;
  3. llama.cpp PR #27742 (arch `qwen4exp`) for the GGUF names and the GGUF-side hash;
  4. [Strata](https://github.com/Niko1221/Strata) (MIT) for an independently validated CUDA
     transcription and its pitfalls list.
  Where they disagree this document says so and follows (1).
- **Abbreviations:** `H` = 2560 hidden, `HC` = 4 streams, `R` = the residual stack `(HC, H)`, `LR` = 320, `E` = 512
  experts, `K` = 10 routed per token, `I` = 640 expert width, `T` = tokens in a step (1 decode, ≤ 16 verify, chunk
  for prefill).

## Implementation status

Everything the product guide describes runs through the public Engine: the text path with its PLE
n-gram table (stored in the model or in a table artifact of its own), GGUF block banks imported
without requantization, expert residency on the device, in pinned host memory or in the artifact's
files, up to eight concurrent requests, the context cache with host and disk images, structured
output, token log probabilities, Vision, and MTP from Unsloth's shared GGUF block.

The qualification contract: every Op against an independent FP64 oracle; exact repetition with the
execution configuration fixed; exact restore of a context image for the same computation history
and execution profile. Verification width, prefill partition and batch composition change
reductions and can flip a near-tied argmax, so differences between plain and MTP decoding or between
batch compositions are reported, not required to be zero. A split over two RTX 3090s matched one
RTX A6000 through the first end-of-sequence token, and every token at 128K; past that token the
outputs of the two hosts differ, with different drivers and route profiles, for a reason not
established.

Built and kept experimental, not published:

- the native Q2 recipe (`qwen3_8_flash_next_gsq_q2`): +0.26% held-out perplexity against the GGUF
  artifact and slower scoring;
- the HF-source Q4/Q5 recipe (`qwen3_8_flash_next`), whose full-model quality is unmeasured, and the
  FP8 n-gram table, rejected for its 80% larger rows.

Not supported: DFlash or DFlash2 (no drafter exists for this model; PixelML's NVFP4 DFlash drafter
is a research artifact for vLLM), n-gram copy proposals, `--lm-head-draft`,
`--mtp-attention-window`, disk residency of native banks. The Windows paths have not been run.

---

## 1. Forward mathematics

### 1.0 Config fields that matter (text_config) and their derived values

| field | value | used by |
|---|---|---|
| `hidden_size` | 2560 | everything |
| `num_hidden_layers` | 48 | `layer_types` = `[lin, lin, lin, full] x 12` (`full_attention_interval` 4): layers 3,7,...,47 are QSA, the other 36 GDN |
| `vocab_size` | 248320 | embed, head, n-gram hash (multiplier bound) |
| `tie_word_embeddings` | **false** | separate `lm_head.weight` |
| `rms_norm_eps` | 1e-6 | all norms |
| `hc_count`, `hc_lowrank` | 4, 320 | hyper-connections |
| `linear_num_key_heads`, `linear_num_value_heads` | 16, 48 | GDN (GQA ratio 3, `repeat_interleave`) |
| `linear_key_head_dim`, `linear_value_head_dim` | 128, 128 | GDN |
| `linear_conv_kernel_dim` | 4 | GDN causal conv |
| `output_gate_type` | `"sigmoid"` | **GDN gated norm activation** (Qwen3.5 uses SiLU here) |
| `mamba_ssm_dtype` | float32 | GDN recurrent state |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | 24, 2, 256 | QSA main attention |
| `partial_rotary_factor` | 0.25 | rotary dim 64 of 256 |
| `rope_parameters` | theta 1e7, `mrope_section [11,11,10]`, `mrope_interleaved true` | QSA + indexer |
| `indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim` | 4, 1, 128 | QSA indexer |
| `indexer_compress_ratio`, `indexer_budget` | 4, 2048 | block = 4 tokens; `block_topk` = 512 blocks |
| `num_experts`, `num_experts_per_tok`, `norm_topk_prob` | 512, 10, true (class default) | router |
| `moe_intermediate_size`, `shared_expert_intermediate_size` | 640, 640 | experts |
| `ple_layer_ids` | [2] (**one-indexed**) | PLE on `layer_idx = 1` (GGUF `blk.1`) |
| `ple_embed_dim` | 2560 | 16 heads x 160 |
| `ngram_size`, `heads_per_ngram` | 3, 8 | 16 hash heads (8 bigram + 8 trigram) |
| `ngram_vocab_size_base`, `make_ngram_vocab_size_divisible_by` | 20,000,000, 128 | table rows 320,001,536 |
| `split_ngram_parts` | 128 | checkpoint stores the table as 128 row shards of 2,500,012 rows |
| `ple_conv_kernel_size` | 4 (dilation = `ngram_size` = 3) | PLE conv, history 9 tokens |
| `seed` | 1234 (class default; not in config.json) | n-gram multipliers |
| `eos_token_id` | 248044 (endoftext) | **n-gram cut token**. Note: chat turns end with im_end 248046, which does NOT cut |
| `mtp.num_hidden_layers`, `mtp.layer_types`, `mtp.rope_theta` | 1, `["full_attention"]` (= QSA), 1e7 | MTP |
| `mtp_use_dedicated_embeddings` | false | MTP shares `embed_tokens` and `lm_head` |
| `max_position_embeddings` | 262144 | |

All RMSNorm weights of class `Qwen4ExpTextRMSNorm` / `Qwen3_5RMSNorm` / vLLM `GemmaRMSNorm` are **zero-centred**:
`y = x * rsqrt(mean(x^2) + eps) * (1 + w)`, computed in FP32. The GDN gated norm (`linear_attn.norm.weight`) is the
exception: plain `w` (initialised to ones), as in Qwen3.5. A loader that folds `1 + w` once must do it for exactly the
zero-centred class and not for `linear_attn.norm`.

**There is no final `model.norm`.** The index has no `model.language_model.norm.weight`; the final
`hyper_connection_mixer` (with its own `hc_norm`) is the last op before `lm_head`.

### 1.1 Tensor census (from the 131 safetensors headers; all BF16 unless noted)

Prefix `L.` = `model.language_model.`; `{l}` = layer; `{s}` = 0..127.

| tensor | shape | count | role |
|---|---|---|---|
| `L.embed_tokens.weight` | [248320, 2560] | 1 | token embedding |
| `lm_head.weight` | [248320, 2560] | 1 | head (untied) |
| `L.layers.{l}.attn_hyper_connection.hc_norm.weight` | [10240] | 48 | per-stream zero-centred norm gamma |
| `...attn_hyper_connection.input_mix_weight_down.weight` | [320, 10240] | 48 | HC down |
| `...attn_hyper_connection.input_mix_weight_up.weight` | [10240, 320] | 48 | HC up |
| `...attn_hyper_connection.block_inject_weight.weight` | [4, 10240] | 48 | HC inject |
| `...mlp_hyper_connection.{same four}` | same | 48 each | |
| `L.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}` | [10240], [320,10240], [10240,320] | 1 | final mixer, no inject |
| `...linear_attn.in_proj_qkv.weight` | [10240, 2560] | 36 | q 2048 / k 2048 / v 6144 rows |
| `...linear_attn.in_proj_z.weight` | [6144, 2560] | 36 | output gate |
| `...linear_attn.in_proj_a.weight`, `in_proj_b.weight` | [48, 2560] | 36 each | decay / beta |
| `...linear_attn.A_log`, `dt_bias` | [48] (BF16) | 36 each | |
| `...linear_attn.conv1d.weight` | [10240, 1, 4] | 36 | depthwise causal conv |
| `...linear_attn.norm.weight` | [128] | 36 | gated RMSNorm (plain w) |
| `...linear_attn.out_proj.weight` | [2560, 6144] | 36 | |
| `...self_attn.q_proj.weight` | [12288, 2560] | 12 | per head: q (256) then gate (256) |
| `...self_attn.k_proj.weight`, `v_proj.weight` | [512, 2560] | 12 each | 2 KV heads |
| `...self_attn.q_norm.weight`, `k_norm.weight` | [256] | 12 each | zero-centred |
| `...self_attn.o_proj.weight` | [2560, 6144] | 12 | |
| `...self_attn.indexer.index_qk_proj.weight` | [640, 2560] | 12 | rows 0..511 = 4 q heads, 512..639 = 1 key head |
| `...self_attn.indexer.q_layernorm.weight`, `k_layernorm.weight` | [128] | 12 each | zero-centred |
| `...mlp.gate.weight` | [512, 2560] | 48 | router |
| `...mlp.experts.gate_up_proj` | [512, 1280, 2560] | 48 | per expert rows 0..639 gate, 640..1279 up (`chunk(2)`) |
| `...mlp.experts.down_proj` | [512, 2560, 640] | 48 | |
| `...mlp.shared_expert.{gate,up}_proj.weight` | [640, 2560] | 48 each | |
| `...mlp.shared_expert.down_proj.weight` | [2560, 640] | 48 | |
| `...mlp.shared_expert_gate.weight` | [1, 2560] | 48 | scalar sigmoid gate |
| `L.layers.1.ple.key_proj.weight` | [10240, 2560] | 1 | |
| `L.layers.1.ple.value_proj.weight` | [2560, 2560] | 1 | |
| `L.layers.1.ple.norm_{key,query,conv}.weight` | [10240] | 1 each | grouped (per 2560) zero-centred |
| `L.layers.1.ple.conv1d.weight` | [10240, 1, 4] | 1 | depthwise, dilation 3 |
| `L.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight` | [2500012, 160] | 128 | the table, row-sharded on dim 0 |
| `L.layers.1.ple.ple_embedding.layer_multipliers` | I64 [3] | 1 | hash multipliers |
| `L.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes`, `ngram_heads_offsets` | I64 [16] | 1 each | |
| `mtp.pre_fc_norm_embedding.weight` | [2560] | 1 | zero-centred |
| `mtp.pre_fc_norm_hidden.weight` | [10240] | 1 | zero-centred, **one** norm over 10240 (not grouped) |
| `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight` | [2560, 2560] | 1 each | |
| `mtp.layers.0.*` | one QSA decoder layer (attn+indexer, MoE 512x10 + shared, two HC) | | no PLE, no GDN |
| `mtp.hyper_connection_mixer.*` | as the main mixer | 1 | |
| `model.visual.*` | 27-block ViT, hidden 1152, 16 heads, mlp 4304, patch 16, temporal 2, merge 2, pos_embed [2304,1152], merger 4608->2560 | | identical structure to Qwen3.5-MoE vision; `deepstack_visual_indexes = []` |

Checkpoint total 359,999,963,128 B in 131 shards, which my census reproduces exactly (360.0 GB):

| component | params | BF16 bytes |
|---|---:|---:|
| routed experts (48 x 512 x 3 x 640 x 2560) | 120.80 B | 241.6 GB |
| n-gram table (320,001,536 x 160) | 51.20 B | 102.4 GB |
| GDN mixers (36 x 57.96 M) | 2.087 B | 4.17 GB |
| QSA mixers incl. indexer (12 x 51.45 M) | 0.617 B | 1.23 GB |
| hyper-connections (96 halves x 6.60 M + mixer) | 0.641 B | 1.28 GB |
| embed / lm_head | 0.636 B each | 1.27 GB each |
| routers + shared experts | 0.299 B | 0.60 GB |
| PLE projections + norms + conv | 32.8 M | 66 MB |
| MTP (dense 0.091 B + experts 2.517 B) | 2.61 B | 5.2 GB |
| vision | 0.449 B | 0.90 GB |
| **text model without table** | **125.74 B** ("125B") | 251.5 GB |

Active per token: experts 48 x 10 x 4.915 M = 2.36 B; dense path incl. head 4.31 B; total ~6.7 B ("A6B").

### 1.2 Embedding and the residual stack

```
x0[t]   = embed_tokens[token_t]                               (H)      ; image/video rows replaced by vision features
R[t]    = repeat(x0[t], HC)  -> (HC, H), stream c = x0[t]               (modular L1014: hidden.repeat(1,1,hc_count))
```

`R` is carried in FP32 between layers in every serious implementation (vLLM keeps it in the activation dtype, BF16;
Strata and llama.cpp keep FP32). **Decision for NInfer: FP32 R** (40 KB/token, negligible), because the residual is
summed 97 times and the oracle comparison is FP32.

### 1.3 Hyper-connection (`Qwen4ExpTextGatedResidual`, "gated residual")

One module per half (attention half, MLP half) per layer, plus the final mixer (no inject). Weights per module:
`g` = `hc_norm` [HC*H], `Wd` = `input_mix_weight_down` [LR, HC*H], `Wu` = `input_mix_weight_up` [HC*H, LR],
`Wi` = `block_inject_weight` [HC, HC*H].

**Read (mix)**, for one token, `R` flattened to 10240 with stream-major layout `R[c*H + d]`:
```
xn[c,:]  = R[c,:] * rsqrt(mean_d(R[c,d]^2) + eps) * (1 + g[c,:])        grouped RMSNorm, one group per stream
lo       = silu( (Wd @ xn) / HC )                                         (LR)   <- the /HC is INSIDE silu
gate     = sigmoid( Wu @ lo )                                             (HC*H)
mixed[d] = (1/HC) * sum_c gate[c*H+d] * xn[c,d]                           (H)    block input
w_inj[c] = 2 * sigmoid( (Wi @ xn)[c] / HC )                               (HC)   <- centred on 1
```
**Write (combine)** after the block produced `y` (H):
```
R'[c,d] = R[c,d] + y[d] * w_inj[c]          (the UN-normalised R; y broadcast to every stream, per-stream weight)
```
The final mixer is the read without `w_inj`; its `mixed` is the model's last hidden state (input to `lm_head`).

Decoder layer (modular L801-838):
```
if layer has PLE:  R = R + PLE(R, token ids)                      (section 1.4; applied to the stack, before the read)
(a, R0, w1) = attn_HC.read(R);  y1 = mixer(a)  [GDN or QSA];  R = R0 + y1 (x) w1
(m, R1, w2) = mlp_HC.read(R);   y2 = MoE(m);                   R = R1 + y2 (x) w2
```
The executors fold the write of one half into the norm of the next read (`hc_write_read`, as vLLM's
`combine_and_mix`), so `R` is written once per half. The read is 3 small GEMVs over 6.6 M weights per half, 1.27 GB per
token over 96 halves in BF16; the GGUF recipe stores them as Q8_0, 0.67 GB.

### 1.4 PLE: hashed n-gram embedding injected at layer index 1

Applies to `layer_idx = 1` only (`ple_layer_ids` is **one-indexed**: `[2]` -> second layer; GGUF `blk.1.ple_*`). It is
added to the stack **before** that layer's attention hyper-connection read.

**1.4.1 Constants** (all derivable; the checkpoint also stores them as I64 buffers - load and cross-check both):
```
n_heads        = (ngram_size-1) * heads_per_ngram = 16       head h in [0,8): bigram; [8,16): trigram
head_dim_ng    = ple_embed_dim / n_heads = 160
head_vocab[h]  = the (global_h+1)-th prime > base-1  (base = 20,000,000; global_h = ple_layer_index*16 + h, here = h)
               = 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
head_offset[h] = sum_{j<h} head_vocab[j]  = 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
                 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275
rows           = ceil(320,001,446 / 128) * 128 = 320,001,536      (90 trailing pad rows never addressed)
multipliers    = _build_layer_multipliers(vocab=248320, ngram_size=3, ple_layer_index=0, seed=1234):
                   max_long = 2^63-1; half = (max_long // 248320) // 2; base_seed = 1234 + 10007*0
                   m[i] = 2 * (splitmix64((base_seed + 0x9E3779B97F4A7C15*(i+1)) mod 2^64) mod half) + 1
               = [23703573157769, 20109073645365, 8052911324071]  (Strata's transcription of the GGUF metadata)
splitmix64(v): v += 0x9E3779B97F4A7C15; v = (v ^ v>>30) * 0xBF58476D1CE4E5B9; v = (v ^ v>>27) * 0x94D049BB133111EB;
               return v ^ v>>31            (all mod 2^64)
```
The multiplier bound guarantees `token * m < 2^63`, so signed and unsigned 64-bit arithmetic agree; XOR of
non-negatives stays non-negative.

**1.4.2 Context with the EOS cut** (`_shift_right_ignore_eos`, modular L643-657). For token at position p with
predecessors `p-1, p-2`:
```
ctx[0] = tok[p]
for s in 1..2:  ctx[s] = tok[p-s] if p-s >= segment_start(p) else EOS
segment_start(p) = (last position q < p with tok[q] == EOS) + 1, or the sequence start
```
Equivalently (llama.cpp/Strata form): walk predecessors newest first; once a predecessor is EOS or missing, it and every
older one read as EOS. The token's own EOS does not cut its own context. A sequence start is preceded by EOS
(HF initialises the cached context to `[EOS, EOS]`). Padding positions (conv mask) read as EOS. **Image/video
placeholder positions hash with their placeholder ids** (`<|image_pad|>` 248056, `<|video_pad|>` 248057) in HF;
llama.cpp substitutes `ple.image_token_id` for any embedding-batch position, which agrees for images.

**1.4.3 Row ids** (16 per token):
```
mixed2 = (ctx[0]*m[0]) XOR (ctx[1]*m[1])
mixed3 = mixed2 XOR (ctx[2]*m[2])
row[h]      = (mixed2 mod head_vocab[h]) + head_offset[h]          h = 0..7
row[8+h']   = (mixed3 mod head_vocab[8+h']) + head_offset[8+h']    h' = 0..7
```
`mod` is a true modulo by a non-power-of-two prime (not a mask).

**1.4.4 Gather.** `emb[t] = concat_h table[row[h]]` -> 2560, **head-slowest** (head h occupies `[160h, 160h+160)`).

**1.4.5 The PLE block** (`Qwen4ExpTextPLELayer.forward`, modular L763-783), one token, `R` = stack (HC, H):
```
k       = groupnorm_HC( key_proj @ emb , norm_key )          (HC, H)   key_proj [10240, 2560]
v       = value_proj @ emb                                   (H)       value_proj [2560, 2560]
q       = groupnorm_HC( R , norm_query )                     (HC, H)
s[c]    = sum_d k[c,d] * q[c,d] / sqrt(H)
s'[c]   = sign(s[c]) * sqrt(max(|s[c]|, 1e-6))               signed square root
G[c,:]  = sigmoid(s'[c]) * v                                 (HC, H)   "gated value"
N       = groupnorm_HC( G , norm_conv )                      (HC, H)   conv input
conv[c,d] = silu( sum_{k=0..3} W[cH+d, k] * N_{t - 3*(3-k)}[c,d] )      depthwise, dilation 3, taps t-9, t-6, t-3, t
R      <- R + G + conv
```
`groupnorm_HC` = the zero-centred RMSNorm applied per 2560-wide stream. The conv state is the last 9 tokens of `N`
(HF `conv_states[1]`, length `(kernel-1)*dilation = 9`); the n-gram context is the last 2 token ids
(`conv_states[2]`). Both are per-sequence recurrent state and must be checkpointed with the GDN states.
`sign(0) = 0`, so `s = 0` gives gate 0.5.

### 1.5 Gated DeltaNet layers (36 layers)

Identical to Qwen3.5's `Qwen3_5GatedDeltaNet` except the output-gate activation. With `a` = HC-mixed block input (H):
```
qkv   = in_proj_qkv @ a                     (10240)  = q (16x128) | k (16x128) | v (48x128), contiguous runs
qkv   = silu( causal_conv1d_k4(qkv) )       depthwise over 10240 channels, state = last 3 inputs, SiLU on all 10240
z     = in_proj_z @ a                       (48x128)
b     = in_proj_b @ a ;  beta = sigmoid(b)                     (48)
g     = -exp(A_log) * softplus(in_proj_a @ a + dt_bias)        (48)   decay in log space
q,k   = l2norm(q), l2norm(k) per 128-dim head (eps 1e-6);  q *= 1/sqrt(128)
q,k   = repeat_interleave(.., 3) over heads  -> v-head j uses k-head floor(j/3)
S_j   = exp(g_j) * S_j ;  S_j += beta_j * k_j (v_j - S_j^T k_j)^T ;  o_j = S_j^T q_j     (FP32 state 128x128 per v-head)
y_j   = rmsnorm(o_j) * w_norm * SIGMOID(z_j)                   <- Qwen3.5: SiLU(z_j)
out   = out_proj @ y                        (2560)
```
Shapes equal Qwen3.6/3.8-27B's GDN (16 key / 48 value heads, 128 dims), so the existing NInfer GDN kernels apply as
is once the gate activation is a parameter. State per layer: 48 x 128 x 128 FP32 = 3 MiB; conv state 10240 x 3.

### 1.6 Qwen Sparse Attention (QSA) layers (12 layers, and the MTP layer)

**1.6.1 Projections** (block input `a`, position p):
```
qg       = q_proj @ a  -> view (24, 512):  q_h = qg[h, 0:256],  gate_h = qg[h, 256:512]   (per-head halves)
q_h      = rope64( rmsnorm0(q_h, q_norm) )       k = rope64( rmsnorm0(k_proj @ a, k_norm) )   (2, 256)
v        = v_proj @ a                                                                       (2, 256)
iq, ik   = index_qk_proj @ a  -> iq (4, 128) rows 0..511,  ik (128) rows 512..639
iq       = rope64( rmsnorm0(iq, q_layernorm) )        ik is cached RAW (no norm, no rope)
```
`rope64`: rotate-half (NeoX) RoPE on the first 64 dims only (`partial_rotary_factor 0.25`; dims 64..255 / 64..127
pass through), theta 1e7, 32 frequencies. M-RoPE: `mrope_section [11,11,10]` interleaved over the 32 frequency slots
(Qwen3-VL/3.5 interleaved layout: slot i takes T/H/W position by `i mod 3` within the section lengths). For text all
three position ids are equal, so this reduces to ordinary 1-D RoPE; vision positions follow Qwen3.5's 3-D rule
unchanged. The **same 64-dim rotation applies to the 128-dim indexer heads** (config validation requires rotary_dim ≤
indexer_head_dim).

**1.6.2 Indexer and selection** (modular `Qwen4ExpTextQSAIndexer.forward`, L388-474). For query position p, the
visible keys are positions `0..p` (causal). Let `n = p+1`, `nb = floor(n / 4)` complete blocks; block `b` = positions
`4b..4b+3` (blocks are aligned to absolute position 0 of the sequence, not to the chunk).
```
K_b      = rope64_at(pos = 4b)( rmsnorm0( mean_{i<4} ik[4b+i] , k_layernorm ) )     pooled key, rotated at the
                                                                                     block's FIRST position
score_b  = (1/sqrt(128)) * sum_{h<4} relu( iq_h . K_b )                              relu per head, then sum
Sel      = top-min(512, nb) blocks by score   ->  all 4 positions of each selected block
         + the tail positions 4nb..p (0..3 of them), always selected
```
Main attention for query p attends **only** to `Sel` (≤ 2048 + 3 = 2051 positions), causal mask still applied:
```
attn_h = softmax_{j in Sel}( q_h . k_{kv(h)}(j) / 16 ) . v_{kv(h)}(j)      kv(h) = floor(h / 12)
o      = o_proj @ concat_h( attn_h * sigmoid(gate_h) )
```
Consequences:
- Context ≤ 2051 tokens (nb ≤ 512): selection is the identity -> plain dense causal GQA.
- No sliding/local window and no sink beyond the tail rule.
- Pooled keys are deterministic functions of the cached raw keys: an engine caches `K_b` once per completed block
  (FP32 recommended, 512 B per block per layer) plus the raw keys of the incomplete tail (≤ 3 x 128).
- Score scale `1/sqrt(128)` does not change top-k; ties: `torch.topk` order is unspecified. Use "higher score, then
  lower block index" and document it (Strata, llama.cpp do the same).
- **Divergence to avoid:** a cell-level top-2051 over block-broadcast scores (Strata's original `topk_512`) differs from
  HF when the tail is empty (it admits 3 positions of the 513th block). Select **blocks**, then append the tail.
- Prefill: every query row has its own selection (the per-row block-causal mask `4b+3 ≤ p` and its own tail).

**1.6.3 KV and indexer cache per QSA layer:** K, V: 2 heads x 256 x 2 (K,V) = 1024 values/token (2 KiB in BF16,
24 KiB/token over 12 layers); pooled indexer keys 128 FP32 per 4 tokens (128 B/token/layer, 1.5 KiB/token over 12).

### 1.7 MoE (every layer, and the MTP layer)

Qwen3-Next / Qwen3.5-MoE semantics, unchanged (modeling L961-994):
```
p        = softmax(gate.weight @ m)   over 512 (FP32)
(w, ids) = top10(p);  w /= sum(w)                         norm_topk_prob = true
y        = sum_j w_j * down_j( silu(gate_j @ m) * (up_j @ m) )      expert width 640; gate rows first in gate_up_proj
y       += sigmoid(shared_expert_gate @ m) * down_s( silu(gate_s @ m) * (up_s @ m) )
```
The router GEMV is the selection-sensitive step: keep its weight BF16 (or better) and its activation unrounded; Strata
measured that a BF16-rounded activation flips top-10 selections (logit error 8.1e-3).

### 1.8 Final mixer and head

```
h_final = hyper_connection_mixer.read(R)        (no inject, no final RMSNorm)
logits  = lm_head @ h_final                      (248320)
```

### 1.9 MTP (`mtp.*`, one QSA layer, "hybrid": keeps the main model's HC stream count, no PLE)

From vLLM `qwen4_exp_mtp` (Strata reports draft acceptance 0.89/0.86/0.85 at steps 1-3
with this transcription). Cell i of the MTP sequence pairs the main model's **final multi-stream residual `R_i`
(before the final mixer)** with token `x_{i+1}`, at rope position i (vLLM convention), and predicts `x_{i+2}`:
```
e   = fc_embedding @ rmsnorm0(embed_tokens[x_{i+1}], pre_fc_norm_embedding)                 (H)
hn  = rmsnorm0(R_i flattened 10240, pre_fc_norm_hidden)    ONE norm over all 10240 values
h   = per stream c: fc_hidden @ hn[c]                                                         (HC, H)
R   = h + broadcast(e)                                     combine with unit weight (no inject)
R   = MTP decoder layer (QSA attention with its own indexer/KV, MoE 512x10 + shared, two HC modules)
out = mtp.hyper_connection_mixer.read(R)  -> lm_head (shared)  -> draft logits
next draft step: R (pre-mixer stack of this step) replaces R_i; token = this step's draft
```
SGLang agrees with vLLM; llama.cpp (`qwen4exp.cpp:581`) applies `pre_fc_norm_hidden` per stream (gamma reshaped
[2560, 4]) and folds `eh_proj = [fc_embedding | fc_hidden]` - treat it as the outlier.
vLLM selects QSA blocks at draft step 0 and reuses that selection for later steps (`set_skip_topk`; the paper says
the same). Strata attends
densely over a 32,768-cell MTP window instead (exact below 2051 cells). Either only changes draft quality, never output.
`mtp.layers.0.mlp.*` is a full 512-expert MoE (2.52 B params): drafting costs as much expert traffic as one main layer
per draft token.

### 1.10 Vision

`model.visual` is the Qwen3.5-MoE vision tower verbatim (27 blocks, 1152 hidden, 16 heads, MLP 4304 GELU-tanh,
patch 16, temporal patch 2, spatial merge 2, learned 2304-entry position embedding with interpolation, merger
LayerNorm -> 4608 -> GELU -> 2560, no deepstack). Image features replace `<|image_pad|>` rows of `x0`; 3-D M-RoPE
positions as Qwen3.5. Preprocessor: the HF repository's `preprocessor_config.json` and
`video_preprocessor_config.json`. PLE hashes the placeholder ids (1.4.2). NInfer runs it with the Qwen3.5 tower
code, `out_hidden_size` 2560.

### 1.11 Tokenizer, chat template, special tokens

- Qwen2 BPE tokenizer, vocab 248,320 (`tokenizer_config.json`), same special-token block as Qwen3.5/3.6
  (`<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, vision 248053-248057, `<tool_call>` 248058,
  `<think>`/`</think>` 248068/248069, audio/tts 248070-248076); the frontend reuses the Qwen3.5 tokenizer code.
- `generation_config`: eos `[248046, 248044]`, pad 248044, `do_sample`, temperature 1.0, top_k 20, top_p 0.95.
- Chat template (`chat_template.jinja`) is the Qwen3.5 XML tool-call template plus a **`reasoning_effort`**
  variable (`xhigh` default, `medium`, `low`) that injects a system-prompt sentence, `enable_thinking`, and
  `preserve_thinking` (default true: past assistant turns keep their `<think>` blocks). The serving layer passes
  `reasoning_effort` through.

---

## 2. Byte census and state sizes

### 2.1 Stored bytes by representation

| component | params | BF16 | 8-bit (int8 g64 / FP8) | 4.5 bpw (Q4 g64, NVFP4, IQ4_NL) | ~3 bpw (IQ3_XXS class) | ~2.3 bpw (Q2_0 / IQ2_XS class) |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 120.80 B | 241.6 GB | 128.3 GB | 67.9 GB | 46.2 GB | 34.0-34.9 GB |
| one expert (gate+up+down) | 4.915 M | 9.83 MB | 5.22 MB | 2.76 MB | 1.88 MB | 1.38-1.42 MB |
| experts read per token (480) | 2.36 B | 4.72 GB | 2.51 GB | 1.33 GB | 0.90 GB | 0.66-0.68 GB |
| n-gram table | 51.20 B | 102.4 GB | 51.2 GB (+row scales) | 28.8 GB (90 B/row) | - | - |
| GDN + QSA mixers | 2.70 B | 5.41 GB | 2.87 GB | 1.52 GB | | |
| hyper-connections | 0.641 B | 1.28 GB | 0.68 GB | (not advised) | | |
| router (48 x 512 x 2560) | 63 M | 126 MB | keep BF16 | | | |
| shared experts | 236 M | 472 MB | 251 MB | 133 MB | | |
| lm_head | 0.636 B | 1.27 GB | 0.68 GB | 0.36 GB | | |
| embed_tokens (gather only) | 0.636 B | 1.27 GB | host RAM is enough | | | |
| PLE projections | 32.8 M | 66 MB | | | | |
| MTP dense / MTP experts | 0.09 B / 2.52 B | 0.18 / 5.03 GB | 0.10 / 2.67 GB | - / 1.42 GB | | 0.70 GB |
| vision | 0.45 B | 0.90 GB | | | | |

The published GGUFs (ISTA-DASLab GSQ-RCO) put everything except the table into shard 1:
Q2_0 37.6 GB (2.40 bpw average), IQ2_XS 39.2 GB, IQ3_XXS 47.0 GB, IQ3_S 54.8 GB; shard 2 is the table at IQ4_NL,
28.8 GB, identical across variants. `ffn_down_exps` (640 rows) can only be Q2_0 or IQ4_NL because K/I-quants need
256-wide blocks. They ship no MTP; NInfer takes it from Unsloth's MTP GGUFs or fetches the `mtp.*` tensors from the BF16
checkpoint by HTTP range (`tools/reference/fetch_slice.py --mtp`).
Official `Qwen/Qwen3.8-Flash-Next-FP8` and community NVFP4 (`nvidia/Qwen3.8-Flash-Next-NVFP4`) checkpoints also
exist.

### 2.2 What must be touched per token

| item | per decode token | per verify window of T tokens | where it should live |
|---|---|---|---|
| dense mixers + HC + routers + shared experts + head | all (~3.4 GB at W4-dense/BF16-HC, ~5.2 GB at int8-dense) | once (weights shared by T columns) | VRAM |
| routed experts | 480 distinct (~2% of 24,576) | ~U(T) x 480 distinct, U(4) ≈ 2.9 (Strata) | VRAM cache + host RAM |
| n-gram rows | 16 rows (2.5 KB at FP8, 1.4 KB at 4-bit) | 16T rows | disk / page cache / RAM |
| embed row | 1 row | T rows | host RAM (gather + 5 KB H2D) or VRAM |
| QSA KV | ≤ 2051 positions x 12 layers x 2 KiB (BF16) ≈ 48 MiB, bounded | same set + new | VRAM |
| GDN state | 36 x 3 MiB FP32 | read + write | VRAM |

QSA makes attention traffic **flat in context length** past 2051 tokens. The decode cost is dominated by the dense
sweep (once per window) and the expert reads (grow with T).

### 2.3 Per-sequence state and KV sizes

| item | size |
|---|---|
| QSA K+V | 24 KiB/token BF16; ~12.4 KiB int8 g64; 9.6 KiB rk8v4-class (K int8 + V int4) - 12 layers |
| pooled indexer keys | 1.5 KiB/token (FP32, 12 layers) + 3 raw tail keys per layer |
| MTP layer KV + indexer | +1/12 of the above if the MTP keeps full history (or a 32K window as Strata) |
| GDN recurrent | 36 x 48 x 128 x 128 x 4 B = 108 MiB (FP32) |
| GDN conv | 36 x 10240 x 3 x 4 B = 4.2 MiB |
| PLE conv history + n-gram context | 9 x 10240 x 4 B = 360 KiB + 2 token ids |
| context 128K | KV 3.0 GiB BF16 / 1.2 GiB rk8v4; indexer 0.19 GiB |
| context 262K | KV 6.0 GiB BF16 / 2.4 GiB rk8v4; indexer 0.38 GiB |

A checkpoint (StateImage) is 113 MiB + the PLE state; ReplaySSM records are as for 27B.

### 2.4 The n-gram table in the artifact

The table is the `ngram` component: rows in the HF row-index space (head offsets as in 1.4.1), row-contiguous with
no padding between rows, `row * row_bytes` from the payload start; the 90 trailing pad rows are dropped
(320,001,446 addressed rows, the header records 320,001,536). Its descriptor carries the format, `row_bytes`, the
row count, `head_dim_ng` 160, the 16 heads, the three multipliers, `head_vocab` and `head_offset`, the EOS and
image/video placeholder ids, and the SHA-256 of the rows; the runtime derives the constants again from the model's
configuration and refuses a table that disagrees.

| format | bytes per row | table | use |
|---|---:|---:|---|
| `gguf_iq4_nl` | 90 (five 32-value blocks of an FP16 scale and 16 bytes of IQ4_NL codes) | 28.8 GB | the GSQ-RCO releases' shard 2, imported byte for byte; the published table |
| `fp8_e4m3fn_row_fp16` | 162 (160 E4M3FN codes and an FP16 scale) | 51.8 GB | written from the HF BF16 shards; supported, not published |
| `bf16` | 320 | 102.4 GB | reference |

A model either stores the rows itself or names a table artifact, a `.ninfer` with the `ngram` component alone that
every quantization of the model shares (GSQ ships the same IQ4_NL table with every release, the Coder build's
included). Rows are decoded on the GPU, never on the CPU, so the table stays compact in RAM and on the bus. The
optional `ngram.resources.hot_profile` resource ranks rows by frequency for `--ngram-residency ram-hot`; a row
depends only on its token and the two before it, so the profile is counted from text without running the model.
