# NariQ 原生移植（`exp/nariq`）：算法、格式与验证状态

起点：`exp/dspine`（`8cd2c2347`）。本分支把
[nari-labs/nari-llama.cpp](https://github.com/nari-labs/nari-llama.cpp)（HEAD `8e0e016`，分支
`nariq/b12431`）的 **coded-linear（TRL1..TRL6）算法**搬进 ninfer，并以 ninfer 自己的权重、
artifact 和算子体系落地——不是把 `llama.cpp` 再挂成一个后端。

背景分析（为什么不能"打补丁就开"、三条路线、风险）见 `docs/nariq-port-analysis.md`；本文只写
**已经落进代码的部分**、**怎么验证**和**还缺什么**。

---

## 1. 算法本体

一个 coded linear 有 N 个输出、K 个输入（`W` 为 `[K,N]`，ggml 维序）：

```text
y = gather_pout( post ⊙ T_rows^T ( Ŵ · T_cols ( pre ⊙ gather_pin(x) ) ) )
Ŵ[n][k] = s16[k/16] · v( state[n][k] )          # v ∈ [-126, 126] 的 INT8 网格
```

- 权重存 16×16 tile，**输出 tile 主序**：`for J in N/16: for i in K/16: 8*bits 个 uint32`；
  每 16 位窗口查 INT8 网格（`state * 0x83DCD12D`，折四段 6 bit 求和 −126）。
- `T_cols`/`T_rows`：kind 0 恒等、kind 1 分块 Sylvester Hadamard（RHT）、kind 2 整片 Hadamard
  （`m ∈ {1,12,20,68}` Paley 因子，最多 8 个 part 并排）。输出侧用其逆。
- 旋转作用在**激活**上，不在权重上；因此每次前向都要对激活做一次 FWHT / Paley 交叉。

## 2. 落地点

| 文件 | 内容 |
|---|---|
| `include/ninfer/ops/trellis.h` | 公开算子契约：`trellis_linear`（coded 投影）、`trellis_linear_workspace_bytes`、`trellis_embedding`（coded 查表） |
| `src/ops/trellis/trellis_reference.{h,cpp}` | **主机端权威实现**：desc 解析/校验、Paley H12/H20/H68、FWHT、整片 slice、tile 解码、整权重 `dequantize`、`project_host` / `embedding_host`（对拍用） |
| `src/ops/trellis/trellis_kernels.{cuh,cu}` | 设备端：gather+pre、per-column RHT、置换、整片 fwht/cross、tile 乘（16 行 × ≤64 token 一块）、post+gather+BF16、coded 行解码 |
| `src/ops/trellis/trellis_dispatch.cpp` | 校验、**按权重缓存变换计划**（置换 + Paley 上传到显存）、workspace 切分、四条流水线 |
| `src/core/weight.h` | `QType::TRELLIS_1BIT..6BIT`、`QuantLayout::TrellisTile16`、`Weight::trellis_{s16,pre,post,pin,pout,desc,desc_len}` |
| `src/core/weight_view.cpp` | `trellis_tile16_v1` 编码尺寸：`(N/16)·(K/16)·32·bits` 字节，K/N 必须 16 整除 |
| `src/artifact/formats.cpp` | 格式名 `trellis_trl1..6`、布局名 `trellis_tile16_v1` |
| `src/ops/linear/linear.cpp` | `is_coded()` 权重在 `linear()` 里走 `trellis_linear`；workspace 公式同步 |
| `src/ops/CMakeLists.txt` + `src/ops/trellis/sources.cmake` | 编译接入 |
| `tools/artifact/formats.py` / `layouts.py` | Python 侧同名的 `TrellisFormat` / `trellis_tile16_v1` 与尺寸公式 |
| `tools/convert/sources/gguf.py` | GGUF 类型 44..49（`TRL1..6`）的块几何，I32 直读 |
| `tools/convert/trellis_gguf.py` | **NariQ GGUF → `.ninfer` 转换器**：原样拷贝 tile 字，侧张量升为 F32 无损失搬运，desc 进 metadata，banded head 拼成单张 coded 矩阵 + 组合后的 `pout` |
| `tools/trellis/verify_trellis.py` | 独立解码器（两种等价写法互校）+ 合成 GGUF 端到端自测 |

### 执行流水线（`trellis_linear`）

1. `gather_scale`：`xa[k,t] = pre[k] · x[pin[k], t]`（BF16 → FP32）。
2. 列侧变换：kind 1 → 就地分块 RHT；kind 2 → 置换 gather → 每 shard 的 run FWHT → Paley 交叉 → 置换 scatter。
3. `tile_product`：每块 16 行 × ≤64 token，逐 tile 解码到 smem int8，累加 FP32，乘 `s16`。
4. 行侧变换（逆）：kind 1 RHT / kind 2 反序（cross^T → FWHT）→ scatter。
5. `post_and_gather`：乘 `post`、按 `pout` 取行、一次 BF16 舍入。

workspace = `(K + N + 2·max(K,N)) · T · 4` 字节（旋转激活 + FP32 结果 + 两块逻辑序 scratch）。
解码阶段的计划（置换表、Paley 表）按权重缓存一次，之后每个 step 不再做主机侧计算。

### 转换器语义

- 只做**搬运**：tile 字按字节原样写入，绝不重量化（coded 权重没有 per-row `to_float`）。
- `trellis_s16`（F16）升为 F32：fp16→fp32 是精确映射，数学不变。
- 侧张量成为 Use 的 auxiliaries（`s16`/`pre`/`post`/`pin`/`pout`），desc 写入
  `metadata["trellis.descriptors"]`；`trellis.version` 必须是 1。
- banded head：**16 个 band 各自单独成矩阵**（真实文件 band 位宽不同：`output.band00` 是 TRL6、
  尾部 band 是 TRL4/TRL3/TRL2，行数也不同），head 的 `vocab_gather` 单独成对象，band 布局
  （offset/rows/bits）写进 `trellis.head_bands`。`vocab_gather` 的方向是 **vocab → 拼接行**
  （实测：把 `g[v]` 按 band 偏移分桶，每个 band 命中的 vocab 个数恰好等于该 band 的行数），
  所以读法就是 ggml 的 `get_rows(concat(bands), gather)`：`logits[v] = concat[g[v]]`，是 gather 不是 scatter。
  引擎侧已落到 `ops::trellis_banded_head`（见 2.4）。

## 2.1 三处容易踩错的语义（已按 ggml-trellis.cpp 对齐）

1. **`post` 按存储行索引**，不是输出行：`y[i] = post[pout[i]] · Ŵ[pout[i]]`。ggml 是先在存储序
   乘 `post` 再 `memcpy` 取 `pout[i]` 行。设备 kernel 早期按输出行取 `post`，实测与主机参考差
   1.4×，已修。
2. **`pout` 是 gather**（`out[i] = coded[pout[i]]`），不是 scatter、也不是 `argsort`。
3. **`s16` 是每 16 个输入列一个标量**；artifact 只有 FP32/INT32 直接平面，所以写出时把 GGUF 的
   F16 升为 F32（`fp16→fp32` 精确，数值与 ggml 完全一致）。读侧（主机参考、设备 kernel）都按
   F32 取，早期按 half 解会把 5e-4 的 scale 读成垃圾，实测把权重放大 10⁶ 量级。

## 3. 已验证

### 3.1 真实权重转换

```powershell
cd backends\ninfer
python -m tools.convert.trellis_gguf            # 或脚本里 convert(src, dst)
```

`I:\model\Qwen3.8-27B-NariQ-3.0bpw.gguf`（10.10 GB）→ `Qwen3.8-27B-NariQ-3.0bpw.ninfer`
（10,102,881,536 字节，513 个 coded 矩阵：TRL2 82 / TRL3 199 / TRL4 99 / TRL5 132 / TRL6 1，
另加 16 个 head band 与 1 个 `output.vocab_gather`），用时约 40 秒（tile 字节按 64 MB 分块流式拷贝，
不在内存里拼 10 GB）。`trellis.head_bands` / `trellis.head_vocab_gather` / `trellis.descriptors`
均已写进 metadata。

### 3.2 真权重逐字节与解码一致性

`python -m tools.trellis.check_real`：抽样 `blk.0.ffn_up` / `output.band00` / `output.band03` /
`blk.0.attn_qkv` / `token_embd` / `blk.3.attn_q`，artifact 里的 tile 字节与源 GGUF **sha256 相同**，
两套独立解码实现（word-window / bitstream）**逐位一致**，网格落在 `[-126,126]`。

### 3.3 GPU 算子对拍（真权重）

```powershell
python -m tools.trellis.dump_probe I:\model\Qwen3.8-27B-NariQ-3.0bpw.ninfer I:\model\trellis-probe
build-backends\ninfer\tests\ninfer_trellis_real_test.exe I:\model\trellis-probe
```

```text
ok   blk.0.ffn_up.weight: TRL2 17408x5120 relative L2 0.001664 absmax 0.0078
ok   blk.0.attn_qkv.weight: TRL3 10240x5120 relative L2 0.001680 (pout)
ok   output.band00.weight: TRL6 15488x5120 relative L2 0.001664
trellis real-weight gate ok
```

相对 L2 只有 BF16 输出舍入（1/256 = 0.3%）的一半，说明设备时序与主机定义在同一 FP32 算术上重合；
解码后的 `rms W` 为 0.010 / 0.016 / 0.014，与 Qwen3.8-27B 同类矩阵的量级一致（早期 `s16` 按 half
读时这里是 6320，是发现该 bug 的入口）。

**注意**：主机参考是本分支对 `ggml-trellis.cpp` 的移植，因此这项门禁证明"设备 == 本移植定义"；
与 `ggml` 原实现的逐元素对拍仍需编译 `libggml`（`decode_reference.py` 的 ctypes 路线）后补。

### 3.4 banded head 门禁（真权重）

```powershell
build-backends\ninfer\tests\ninfer_trellis_head_test.exe I:\model\trellis-probe
```

```text
bands 17408x5120 TRL2 + 15488x5120 TRL6
ok   banded head: 32896 rows, 2 tokens, relative L2 0.00166186 absmax 0.0107117
trellis banded-head gate ok
```

取真实文件里位宽与行数都不同的两个 shard 拼成 head，`vocab_gather` 用旋转模拟非恒等映射，
逐行与主机参考比较：证明拼接偏移、每 band 的 K 一致性与最终行读都没错位。

## 4. 还缺什么（按顺序）

| # | 内容 | 位置 |
|---|---|---|
| 1 | ✅ loader 落 `Weight::trellis_*` 与 desc：`Bindings::trellis_side` 按绑定去重、`weight_input.cpp::single()` 校验几何并禁止 coded fused、`Model::attach_trellis` 上传平面（desc 走 host） | `src/models/qwen3_5/load/prepare.cpp`、`src/ops/weight_input.cpp`、`model.cpp`（`8198ff5b3`） |
| 2 | ✅ 引擎命名：`I:\model\gguf转ninfer工具 dspine 1.8\convert_nariq.py` 生成 `text/layers/N/<角色>`、`text/output_head/bandNN`、`text/token_embedding` 与 `components.text.config`（866 参数、`missing: []`） | 该工具目录 |
| 3 | ✅ 算子：`ops::trellis_input_proj`（一次 `trellis_linear` + 行映射分发，`0aa1f3b12`）与门禁；剩引擎把 coded 父矩阵当"一条绑定 + 多路 Part"接进 attention/GDN | `src/ops/trellis`、`load/prepare.cpp` |
| 4 | ✅ 算子：`ops::trellis_banded_head`（band 拼接 + vocab 行读）与门禁（`419084c18`）；剩 head 绑定与 executor | `src/ops/trellis`、`execution/text.cpp` |
| 5 | coded embedding：`ops::trellis_embedding` 已就绪，等执行层的查表接过来 | 同上 |
| 6 | KVMem / 白名单：`serve_options` 对 coded 权重的 target 放行 `--spec dflash2`；`--spec dspine` 仍要等 DSpine 自己的 KVMem 打通（`907ed3bb3` 之后仍被拒） | `src/serve/serve_options.cpp` |
| 7 | 性能：当前 `tile_product` 是 smem int8 + FP32 SIMT，正确优先；decode 走 sm_80+ INT8 tensor core、prefill 复用 A8 主循环是后续 1.5–2 周的量 | `src/ops/trellis/trellis_kernels.cu` |
| 8 | 视觉：NariQ 发布只有 BF16 mmproj，与"视觉必须 Q4"的约定冲突，需单独决策 | 转换路径 |

## 5. 引擎侧两个结构性缺口（实测，2026-11-24）

跑 `ninfer.exe I:\model\Qwen3.8-27B-NariQ-3.0bpw.engine.ninfer --prompt hi --max-context 256` 的结果：
引擎起来了（DLL 需 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64` 在 PATH 里，
`cublas64_13.dll` 只在那层），第一个失败点是 config：

```text
error: Qwen3.5 config: text config: missing max_position_embeddings
```

也就是说候选名的 config 必须按引擎的键重写，而不是 HF 原样：

| 键 | 本模型取值 | 来源 |
| --- | --- | --- |
| `architectures` / `model_type` | `Qwen3_5ForCausalLM` / `qwen3_5_text` | 必须是 `resolve_architecture` 认的配对 |
| `max_position_embeddings` | 262144 | `qwen35.context_length` |
| `rms_norm_eps` | 1e-6 | `qwen35.attention.layer_norm_rms_epsilon` |
| `rope_parameters` | `{rope_theta: 1e7, partial_rotary_factor: 0.25, mrope_section: [11,11,10]}` | `qwen35.rope.*`（`dimension_sections` 前 3 段和为 32 = rotary_dim/2） |
| `linear_num_key_heads` / `linear_key_head_dim` | 16 / 128 | `ssm.group_count` / `ssm.state_size`（**不是** attention 头数） |
| `linear_num_value_heads` / `linear_value_head_dim` | 48 / 128 | `ssm.time_step_rank` / `ssm.state_size`（48 % 16 == 0 才过引擎校验） |
| `linear_conv_kernel_dim` | 4 | `ssm.conv_kernel` |

由此推出的 GDN 几何：key_dim = 2048，value_dim = 6144，`attn_qkv` 10240 = `[q 2048 | k 2048 | v 6144]`；
全注意力层 `attn_q` 12288 = query 6144 与 output-gate 6144 **按 256 行交错**（`dense_geometry.py::attention_rows`：
`(r // 256) * 512 + (256 if gate else 0) + r % 256`），`attn_k` / `attn_v` 各 1024 独立。

两个必须在引擎里改掉的结构：

1. **head 在引擎里是单个 `text/output_head` `[vocab,hidden]` 对象**（`load/text.cpp:307`），而 NariQ 的 head
   是 16 个位宽/行数各不相同的 band + 一张 vocab 映射，拼不成一张矩阵。要么给 `Bindings` 加
   `trellis_head_bands(name)` 让 head 变成"多 band 绑定"，要么在 head executor 里直接走
   `ops::trellis_banded_head`（算子已就绪）。
2. **coded 父矩阵不能像稠密那样拆成 `attention/query`/`gate`/`key`/`value` 四个对象**：`bind_attention`
   现在给四路各绑一个 `Parameter`，`prepare_attn_input_proj_weights` 再按"同一父矩阵的连续行段"融合。
   coded 需要新路径：一条 `Parameter` 承载 12288 行的 coded 父矩阵，`AttentionWeights` 改为
   "dense 四路 / coded 单父 + 行映射" 的两态，投影走 `ops::trellis_input_proj`（算子已就绪）。

引擎参数清单（`load/text.cpp` 权威）：`text/token_embedding{vocab,h}`、`text/output_head{vocab,h}`、
`text/final_norm{h}`、每层 `text/layers/N/{input_norm,post_attention_norm}`，注意力层
`attention/{query{q,h},key{k,h},gate{q,h},value{k,h},query_norm{256},key_norm{256},output{h,q}}`，
GDN 层 `gdn/{query{2048,h},key{2048,h},value{6144,h},z{6144,h},a_projection{48,h},b_projection{48,h},
a_log{48},dt_bias{48},convolution{4,10240},norm{128},output{h,6144}}`，FFN `mlp/{gate,up,down}`。

## 6. 已知风险

- 官方 kernel 只在 RTX PRO 6000 上调过；本机（5060 Ti / sm_120）不要假设 88.9 tok/s 可复现。
- 16 GB 显存：3.0bpw + BF16 视觉 + DFlash2 drafter ≈ 11.5 GB，KV 只剩约 4 GB；没有 KVMem 时
  128K 上下文基本不可能。
- coded 权重无法 `llama-quantize` 重量化、不接受 LoRA（与 nari-llama 的限制一致）。
- `pout`-only 的 banded head 表达要求 band 间 `pre/s16/desc` 一致；发布文件满足，但未来若不一致，
  只能改成"每 band 一次 Op"。

## 7. GDN head 映射（2026-11-25 修复）：PPL 从 1236 降到对齐参考

### 症状与定位

对齐语料（`corpus64.txt` = ppl.txt 前 64 token，c=64 / 1 window）下引擎 PPL **1236.73**，
参考 `llama-perplexity` 在同一窗口的评分为 6.7199。逐级对拍后，GDN 的所有中间量与参考
**逐列精确一致**（cos 1.000000）：

| 量 | 参考 `refF/` | 引擎 dump | 结论 |
| --- | --- | --- | --- |
| input_layernorm 输出 | `ref-attn_norm-0` | `gdn-h` | 一致 |
| q/k/v（conv + silu 后） | `ref-conv_output_silu-0` | `gdn-q`/`gdn-k`/`gdn-v` | 一致 |
| gate 投影（conv 前） | `ref-z-0` | `gdn-z0` | 一致 |
| `beta` / `g` | `ref-beta_sigmoid-0` / `ref-a_softplus-0` | `gdn-beta` / `gdn-g` | 一致（`g = -exp(A_log)·softplus`） |
| q/k 的 per-head L2 | `ref-q_conv_predelta-0` | `gdn-q` 归一化 | 一致 |

只有 `linear_attn_out` 差 5.8%（cos 0.9971）。用主机端完整管线复算（`decode_matrix` +
`pre`/`post`/`pin` + desc 的块 Hadamard + gated RMSNorm + 与 `dequantize`/`project_host` 等价的
numpy 实现，dense 权重 rms 与 `test_trellis_real` 的输出完全一致），证明投影链本身无误；
把 delta-rule 换成参考的映射后与 `ref-linear_attn_out` **cos 1.000000、rms 0.19793 vs 0.19794**。

### 根因

`gated_delta_net` 把 v-head `h` 映射到 q/k-head `h / (H_v/H_qk)`（分组），
而参考实现（`ggml_compute_forward_gated_delta_net_one_chunk`，`iq1 = iv1 % neq1`）是 `h % H_qk`
（循环重复，对应 HF 侧 `q.repeat(1,1,n_v/n_k,1)`）。48 个 v-head / 16 个 k-head 下两者完全不同：
每个 v-head 都配错了 q/k，delta-rule 的每个 head 都有小幅偏差，逐层累积。

修复点（四处，descriptor 与几何不变）：

- `src/ops/linear_attention/gated_delta_net/common.cuh`：`head_map::qk_head` → `h_v % H_qk`
  （去掉 fastdiv 的 `group_magic`）。
- `.../two_stage/prepare.cu`：`qh = head % args.qk_heads`；Q/K 的发布者由 `head % group == 0`
  改为 `head < args.qk_heads`。
- `.../two_stage/recurrence.cu`：`qh = head % args.qk_heads`。
- `.../recurrent.cuh`：`store_key` 的单写者条件由 `value_head % group_size() == 0` 改为
  `value_head < heads.H_qk`。

### 修复后门禁（`Qwen3.8-27B-NariQ-3.0bpw.engine10.ninfer`，bf16 KV，单序列）

- 逐层隐藏状态对 `ref-layerNN`（col 62）：L0 cos 0.999999、L8 0.999996、L32 0.999983、
  L62 0.999959；rms 逐层吻合（L0 0.21354 / 0.21396，L62 5.07071 / 5.06732）。
- layer0 的 GDN 输出 `resid - x` 对 `ref-linear_attn_out`：**全部 63 列 cos 1.000000**。
- 逐位置 CE 与参考 logits 对齐后一致（`ref-logits-0` 覆盖窗口后半段，`ref col r ↔ 引擎 col r+32`，
  logits cos 0.99997+、argmax 相同）。
- 生成（`apps/ninfer --no-cuda-graph`，最小模板、贪心）：
  `"The capital of France is"` → ` Paris.` / `"…over the lazy"` → ` dog`；
  默认 chat 模板下思考链落在 stderr，内容连贯。

### 评分口径（后续对拍必须对齐）

参考 `tools/perplexity/perplexity.cpp` 用 `const int first = n_ctx/2; need_logits = pos >= first;`
——**只评分每个窗口的后半段**，使每个被评 token 都有完整上下文。引擎的 `--disjoint` 评整窗
（`context-1` 个位置，含冷启动位置），两边 PPL 不能直接比：

| 口径 | 引擎 | 参考 |
| --- | --- | --- |
| 整窗（63 位置，含前 32 个冷启动位置） | 9.94589 | —（参考不提供该口径） |
| 窗口后半段（位置 32..62） | **6.7385** | **6.7199** |

对齐口径后相差 0.3%。512 以上上下文同理。另外 `-c 64` 时参考要求语料 ≥ 128 token，
所以 6.7199 是用 ppl.txt（130 token）跑出来的首窗口，`corpus64.txt` 本身无法直接喂参考。

### 附带修掉：trace dump 的 dtype

`execution/text.cpp::trace_dump_matrix` 原来无条件按 BF16 读（每元素 2 字节、每列步进
`ne[0]·2`）。对 FP32 张量（`g`/`beta`）会只读到前一半列、并把每个 float 拆成“低 16 位垃圾 +
高 16 位近似值”，表现为 rms 1e35、偶发 nan 的“GDN 硬 bug”。现已按 `t.dtype` 分派 FP32/BF16。
据此作废的结论：`gdn-g` 的 1e35、`gdn-beta` 的 nan 都是打点假象，不是数值错误。

### CUDA graph decode 崩溃（已修）：capture 时烘进图的指针被后续扩容释放

现象：第二次 decode 步（其实是首次图重放）在 `core/decode_graph.cpp:155 cudaGraphLaunch` 之后报
`cudaErrorIllegalAddress`；`--no-cuda-graph` 完全正常，且**同样的 profile envelope 走直接执行也
正常**（后者用 `NINFER_ENVELOPE_ONLY` 验过），所以不是注意力窗口读越界，而是图里烘死的指针失效。

根因：三处 `trellis` 暂存缓冲是**进程级、按需扩容**的 `DeviceBuffer`。图在启动时 capture，把当时的
指针烘进节点；随后的 prefill 需要更大的缓冲，`scratch = DeviceBuffer(bytes)` 释放旧块并换地址，
图重放就写进已释放显存。提示词长度会改变扩容后的地址是否恰好复用，所以崩溃表现为与长度相关但非单调。

修法：

- `ops/trellis/trellis_dispatch.cpp`：coded parent 平面改为从**本轮 workspace** 分配
  （`trellis_input_proj_workspace_bytes` 增加 `widest × max_tokens × 2 + 256`，
  `trellis_input_proj` 用 `workspace.alloc`）。
- 同文件：banded head 的 band 拼接平面、coded embedding 的暂存保持进程级，但扩容时把旧块
  **退休（retire）而不是释放**，让已捕获图的指针继续有效。

### 提速：与主线格式的实测差距、已修项与剩余项

同架构（Qwen3.8-27B）、同命令、361 token 提示、`--greedy`、同一台机器交替测量：

| 模型 | 体积 | prefill | decode |
| --- | --- | --- | --- |
| `Qwen3.8-27B-GSQ-RCO-IQ3_S-q4km-mtp-dflash2.ninfer`（主线） | 12.9 GB | 774 tok/s | 23.7 tok/s |
| NariQ 3.0bpw 修前 | 9.4 GB | 153 tok/s | 5.3 tok/s |
| NariQ 3.0bpw 修后 | 9.4 GB | **201 tok/s** | **17.9 tok/s** |

已修（全部在 `src/ops/trellis`，格式与数值不变）：

1. 窗口提取从每 lane 每 K-tile 16 次全局 load 降到 4 次（8 个窗口在环形流里连续）。
2. `grid_value` 用 `__dp4a((x >> 2) & 0x3F3F3F3F, 0x01010101, -126)`：8 条指令 → 3 条。
3. 码值用小整数直接 `__int2half_rn` 打包，去掉每权重一次 int→float。
4. `tokens == 1` 走 split-K（16 段 + 归约）：1 token 只有 `N/16` 个 warp，纯延迟受限。
5. 一个 warp 覆盖 2 个 16-token tile（`kTokenTiles = 2`），prefill +10%。

剩余差距的性质（实测数据，不是猜测）：

- **prefill 199 vs 774（3.9x）**：主线 IQ3_S 走 INT8 tensor core，按 774 tok/s 折算约 38.6 TFLOP/s，
  已超过本卡 bf16 峰值（约 24 TFLOPS），只有 int8 通路能解释。coded 路径必须先把码本解成 fp16
  再走 bf16 MMA：每个权重约 10 条 ALU（提取 4 + 网格 3 + 打包 3），按 kernel 内 ALU/张量比估算，
  ALU 与 MMA 各占约一半，因此 fp16 通路的上限约 2× 当前值。实测把每 warp 的 token 数从 16 提到 64
  （权重复用 4 倍、解码摊销 4 倍）后 **prefill 反而下降**（191 vs 199），kTokenTiles = 8 更是 164，这也证明 prefill 不是带宽
  受限而是 ALU 受限。
- **decode 17.8 vs 23.7（1.33x）**：主线 GEMV 每权重约 2~3 条指令（4-bit nibble + 每 32 权重一个
  scale），coded 路径约 10 条，因此有效带宽 168 GB/s 对 303 GB/s。

要真正追平，需要给 coded 格式补一条 **INT8 通路**（这正是该格式的设计用法，nari-llama /
exllamav3 就是这么跑的）：`W_hat` 本来就在 INT8 网格上，激活按 16 列 tile 用现成的 `s16` 量化成
int8，用 `mma.sync.m16n8k32.s8` 或 `__dp4a` 直接算，再乘 `s16 × 激活 tile 尺度`。这样
prefill 可吃到 2× 的张量速率、decode 的每权重指令数也能降到 3~4 条。代价是激活多一次 int8 量化
（参考实现本身就这么做，当前 fp16 通路其实比参考更精确）。

### 采纳 libtrellis 的 A-fragment 构造（第 1 步，已完成）

参考侧不是 ggml 自己写的 kernel，而是官方 `libtrellis`（Apache-2.0，源码在
`ggml/src/ggml-cuda/libtrellis/`）：prefill 走 `tq_gemm`，decode 走 `tq_gemv`。它们的解码核心在
`src/common.cuh`，与 ninfer 原实现的差别：

| 项 | libtrellis | ninfer 原实现 |
| --- | --- | --- |
| 码值 → fp16 | `h_grid`：`__dp4a((st*mult) & 0xFCFCFCFC, 0x01010101, 0x6400)`，dp4a 的**累加器直接给 fp16 指数位**，一步得到 `1024+4v` 的位型；再一条 `__hsub2` 减 1528 得精确 `4v` | 先算整数 `v`（含 `-126`），再 `cvt.rn.f16.s32` 转换 |
| 窗口提取 | `Plan`/`anchors`：2 次 funnel shift 得到 2 个 32 位锚，覆盖 8 个窗口，之后每窗口一次 `low16` | 每窗口 1 次 funnel shift + 1 次移位 |
| 系数 4 | 折进激活标度（`4v × s16/4`），prefill 无需任何求和修正 | — |

已移植（`src/ops/trellis/trellis_kernels.cu` 的 `h_grid`/`half2_sub_1528`，两个 MMA kernel 都用）：
decode **17.9 → 19.5 tok/s**；prefill 198 不变；PPL 9.94589 → **9.90995**（`4v × s16/4` 换了舍入
路径，门禁基线改为 9.91）；`test_trellis_real` rel L2 0.001651/0.001623/0.001685。

### 第 2/3 步未完成：prefill/decode 需要换 kernel 结构

libtrellis 的 `tq_gemm`/`tq_gemv` 的差距不在解码指令，而在结构：

- `tq_gemm`：一个 CTA = **128 输出（8 warp 各一个 tile）× TT = 64/128 行**；每级 4~8 个 k-tile 用
  `cp.async` 把**码字和激活一起**预取进 shared，S 级流水；激活按 B-fragment 顺序**在 prologue 里
  准备一次**（`u = x*pre; H128; u *= s16`，每行一个 2^e 标度）；解码提前一级发出；收尾走 shared
  分 32 行做 epilogue；小 grid 用 split-K。
- `tq_gemv`：`W4 = 8` warp/CTA、`C4 = 8` k-tile/chunk、`MAX_ROWS = 64`，同样 `cp.async` S 级预取，
  A fragment 保留 `4v+1528`、用 `1528·Σx` 事后修正。

ninfer 侧对应要改的：把 `tile_product` 拆成 prefill GEMM（CTA tiling + cp.async + 预置激活）与
decode GEMV（预置激活 + 8 段 split-K + 64 行/组）两个 kernel，并把激活 fragment 的准备从"每个输出
块各做一次"（10240 行的权重会重复 80 次）提到每次投影只做一次。

已用实验排除"只加大 token 复用"这条路：`kTokenTiles` 16/32/64/128 token 在移植后仍为
198/185（2/4 tile），说明瓶颈是激活准备与预取的重复，而不是解码摊销。

### 第 2 步（已完成）：shared 预置激活 + 128 token/块

上一轮加大 token tile 反而变慢，原因不是寄存器，而是**激活读是散的**：`xa` 里相邻 token 相隔
`K×4` 字节（5120 输入时 20 KB），每个 tile 的 `a0` 一条 LDG 要打 8 个 sector，而每个 tile 每
k-tile 要 8 条这样的 LDG。参考实现（`libtrellis/src/gemm.cu` 的 prologue + `cp.async`）正是把这
部分搬进 shared memory 解决的。

按此改 `tile_product_mma_kernel`：

- 块内 cooperative 一次性把当前 k-tile 的整块激活（`16 k × 16·kTokenTiles token`）搬进 shared，
  双缓冲 + 每 k-tile 一次 `__syncthreads()`；各 warp 的 fragment 从 shared 读（`LDS`）。
- warp 早退改成 `mine` 标志（`if (J*16 >= N)` 的 warp 也必须参加 barrier，否则部分块会死锁）。
- `kTokenTiles` 扫描（每块 token 数）：2 → 242、4 → 211、**8 → 300**、12 → 235、16 → 181。
  8（128 token/块）是甜点，与参考的 TT = 64/128 一致。

| 指标 | 修前 | 修后 |
| --- | --- | --- |
| prefill | 198.7 | **297.7** |
| decode | 19.5 | 19.3（不受影响） |
| PPL(c=64) | 9.90995 | 9.90995（完全一致） |
| `test_trellis_real` rel L2 | 0.0016515 | 0.0016515（完全一致） |

累计（相对本移植最初）：prefill 153 → 297（1.94x）、decode 5.3 → 19.4（3.7x）；同架构主线
IQ3_S 为 prefill 747 / decode 23.5，差距收窄到 2.5x / 1.2x。

### 第 3 步（未完成）：decode 的 GEMV 结构

decode 现在 19.4 vs 主线 23.5。参考的 `tq_gemv`（`src/gemv.cu`）还有三处未移植：`W4 = 8` warp/CTA
共享一次解码（我们每 warp 一个 tile）、`C4 = 8` k-tile 一个 chunk 的 `cp.async` 预取、以及
`4v + 1528` 保留到 epilogue 再用 `1528·Σx` 修正（省掉每权重的 `__hsub2`）。预计还能拿 20~30%。

### 权威对照：参考实现（libtrellis CUDA）在本机的实测

`I:\model\nariq-ref\bin` 里有参考侧的完整 CUDA 构建（`llama-bench.exe` + `ggml-cuda.dll`，内含
libtrellis）。同一张卡、同一个 GGUF、不开 MTP：

```
llama-bench.exe -m Qwen3.8-27B-NariQ-3.0bpw.gguf -p 512 -n 128 -r 2 -ngl 99
  pp512   675.61 ± 5.48 t/s
  tg128    33.59 ± 0.10 t/s
```

ninfer 现在（481 token 提示 / 128 new）：prefill **291.2**、decode **19.5**。即还差 **2.3x / 1.7x**。

两者折算后都落在约 36 TFLOP/s（prefill）与约 316 GB/s（decode），说明这张卡的有效上限就在那里，
coded 格式并不需要 int8 张量核——先前"必须走 int8"的判断据此修正。

已实测排除的两项：把码字也搬进 shared（287→280，码字本就命中 cache，还多了一次每 k-tile 的
barrier）、单纯加大 token tile（12/16 tile 反而下降）。剩余差距在 libtrellis 的**软件流水**：
`cp.async` 按 S 级预取码字与激活、**当前 k-tile 的 MMA 与下一个 k-tile 的解码重叠**（gemm.cu 头注释
"the next k-tile's decode is issued before this one's products"）、以及 `__launch_bounds__(256, CTAS)`
的占用率调优。ninfer 目前每个 k-tile 是"读码字 → 解码 → MMA"严格串行。

### 按官方结构重写 prefill：cp.async 多级流水（已完成这一步）

按 libtrellis `gemm.cu` 的做法，把激活取数从"每 k-tile 同步标量读 + barrier"改成
**`cp.async` 多级预取 + 与解码/MMA 重叠**：

- 每个 k-tile 一个 cp.async group，`kActStages = 3` 级缓冲（`s_act[3][128*16]`，24 KB shared）；
- 循环体：`__pipeline_wait_prior(kStages-1)` → 计算本级 → `__syncthreads()` → 用本级的缓冲
  发起 `i+kStages` 的预取并 `__pipeline_commit()`；
- `__pipeline_memcpy_async` 单次只支持 4/8/16 字节，一个 token 的 16 个 float 拆成 4 次 16 字节拷贝
  （传 64 字节会 `cudaErrorLaunchFailure`，这是踩过的坑）；
- 越界 token 用 `min(t0+idx, T-1)` 夹住（结果被 store 的 `r0 < T` 丢弃）。

级数扫描：2 → 319.3、**3 → 322.6**、4 → 319.5。

| 指标 | 上一版 | 现在 |
| --- | --- | --- |
| prefill（481 token） | 291.2 | **322.6** |
| decode | 19.5 | 19.5 |
| PPL(c=64) | 9.90995 | 9.90995 |
| `test_trellis_real` | ok | ok |

对照：llama + libtrellis 在同一台机器上是 prefill 675.6 / decode 33.59；GSQ 模型（ggml
`mul_mat_q`）是 747 / 23.5。ninfer 现在 322.6 / 19.5，即 prefill 还差 2.1x、decode 1.7x。

### 码字加入同一套 cp.async 流水（本次）

把 CTA 的 8 个码字 tile 放进和激活相同的 cp.async group、共用 `kActStages = 3` 的级数
（`s_codes[3][8*nw]`，每 tile `nw` 个字 = 16 字节对齐，所以 `nw/4` 次 16 字节拷贝即可）：

- `stage_activation` 里追加码字拷贝，`Jw` 用 `min(base_J + w, N/16 - 1)` 夹住越界 tile；
- 计算时 `tile = s_codes[i % kStages] + warp * nw`，`quad` 从 shared 取；
- 这是本次唯一有效的改动：prefill **322.6 → 361.0**。

顺带做了两个零收益的 A/B（都保留中性版本）：`__launch_bounds__(kThreads, 2)`（355.5 vs 356.6，
去掉；说明占用率不是瓶颈）、`kDecodeSplits` 16/24/32（19.5 / 19.3 / 18.9，维持 16）。

| 指标 | 本移植最初 | 现在 |
| --- | --- | --- |
| prefill（481 token） | 153 | **361.0** |
| decode | 5.3 | **19.7** |
| PPL(c=64) | — | 9.90995 |
| `test_trellis_real` rel L2 | 0.0016 | 0.0016515 / 0.0016227 / 0.0016851 |

对照（同机、同 GGUF、不开 MTP）：llama + libtrellis = 675.6 / 33.59；GSQ（ggml `mul_mat_q`）= 747 / 23.5。
NariQ + ninfer 现为 361.0 / 19.7，即 prefill 1.87x、decode 1.70x 的差距。

剩余（未做，但已定位）：
1. decode 的 GEMV 结构：`W4=8` warp/CTA 共享一次解码、`C4=8` 个 k-tile 一个 chunk 的 cp.async、
   `4v+1528` 保留到 epilogue 用 `1528·Σx` 修正。decode 现在还在用"每 warp 4 条 LDG + 每 k-tile
   串行"的老结构（prefill 已流水化，decode 没有）。
2. prefill 的解码提前一级（把 `i+1` 的 8 个窗口排在 `i` 的 MMA 之前），需要把 compute 区拆成两级
   缓冲的 fragment，和 barrier 一起重排。

## 2026-11-25 复测与 NSYS kernel 归因（最新代码 HEAD ea49aec03）

测速口径：engine10.ninfer，388 prompt（fox 句 ×3）/ 128 new，greedy，passthrough 模板，`--max-context 1024`，默认 eager（CUDA graph used 0 B）。
实测：**prefill 290.1 tok/s、decode 18.9 tok/s**（文档旧值 361.0/19.7 是 481 prompt 口径；本批短 prompt 下 prefill 被固定开销拉低，decode 一致）。

### NSYS kernel 汇总（prof-cur3.nsys-rep，388 prompt / 16 new，总 GPU 1407 ms）

| kernel | 占比 | 总 ms | 次数 | avg us | 备注 |
| --- | --- | --- | --- | --- | --- |
| `tile_product_mma_kernel`（prefill GEMM） | **81.5%** | 1147.3 | 496 | 2313 | 首要瓶颈 |
| `tile_product_mma_split_kernel`（decode GEMV） | 6.4% | 89.8 | 1040 | 86.4 | 次要 |
| `rht_kernel` | 4.7% | 65.7 | 2787 | 23.6 | 单次极小、启动次数多 |
| `gather_rows_into_kernel` | 2.3% | 32.6 | 1056 | 30.9 | |
| `post_gather_kernel` | 1.9% | 26.6 | 1536 | 17.3 | |
| `gather_scale_kernel` | 1.5% | 21.1 | 1536 | 13.8 | |
| 其余 GDN/attention/silu/norm/sampling | <2% | — | — | — | 非瓶颈 |

prefill MMA 按形状（grid.x=N/16/8，grid.y=4）：

| 位宽 | grid.x | N | avg ms |
| --- | --- | --- | --- |
| b3 | 136 | 17408 | 4.07 |
| b3 | 40 | 5120(×17408) | 3.09 |
| b2 | 136 | 17408 | 3.79 |
| b4 | 40 | 6144 | 2.30 |
| b4 | 136 | 17408 | 3.79 |
| b3 | 80 | 10240 | 2.50 |
| b3 | 48 | 6144 | 1.65 |
| b5 | 1 | 48 | 0.395 |

### 归因判断

- 最大 prefill 形状 n17408/k5120/b3：权重 33.4 MB，4.07 ms → 8.2 GB/s；11.4 GFLOP → 2.8 TFLOPS（5060 Ti FP16 张量核利用率个位数百分比）。既不是带宽 bound 也不是张量核满载，是 **MMA 内核结构效率低**。
- 当前 `tile_product_mma_kernel` 每个 warp 在每个 k-tile 内为 8 个 token tile 重复打包激活 B-fragment（每 tile 4 次 fmul + 4 次 fp16 cvt），激活随 k-tile 从 shared 重读；参考 libtrellis 在 prologue 里一次性把激活生成 fp16 B-fragment（带单行 2^e 缩放），GEMM 内层只做权重解码 + MMA。
- 占用率：reg 127/线程、smem 26.9 KB，约 2 CTA/SM；文档已验 `__launch_bounds__` 不是瓶颈，问题在指令结构与 fragment 复用。
- decode split 内核无 cp.async/shared 流水、无激活 fragment 预生成、16 K-split 后另起 reduce kernel；但当前只占 6.4% GPU，优先级低于 prefill。

### 下一步（按收益排序）

1. **prefill GEMM 对齐 `libtrellis gemm.cu`**：prologue 预生成 fp16 B-fragment（单行 2^e）；CTA 按 TT=64/128 行分块 + 三档 tiling 选择；cp.async 多级流水；权重解码提前一级；split-K + reduce_k。这是 prefill 291→675 的主路径。
2. decode GEMV 对齐 `gemv.cu`：prepare 分块 fragment、W4=8 warp/CTA、C4=8 k-tile chunk cp.async、1528 偏移延后修正、atomic-counter 末 CTA 归约融合 post。
3. 主机侧：`trellis_linear` 每次调用重复 `validate_weight`/`parse_desc`/workspace 字节重算，env 查询未静态缓存；先把 plan/desc/bytes 按 Weight 缓存。

## 2026-11-25 decode 瓶颈归因（cur6/cur7 NSYS）

- trace_step 分步：submit 0.1ms / bookkeeping 0.0ms / **wait 54ms**（每 token）。主机 submit/记账不是瓶颈。
- eager `--no-cuda-graph` 采样 cur7：split GEMV 1083ms/24tok = **45ms/tok，占设备 87.4%**；rht 5.2%、reduce 1.5%、post 1.3%、gather_scale 1.1%。graph 回放时 CUPTI 漏记内核耗时，cur6 里设备仅 3.14ms/tok 是假象。
- split 内核：reg 40–60、smem 0、256 线程，occupancy 充足（≥4 CTA/SM），不是寄存器/占用受限。
- 形状耗时（eager）：b3 N17408 173.6µs/次；b3 N5120×17408 130.2µs；b2 N17408 127.0µs；b4 N5120×17408 71.2µs；b3 N10240 107.4µs。
- 带宽反推：N17408/b3 权重 33.4MB，单次 split 173.6µs ≈ 192GB/s；整模型 9.41GB/token，18.9 tok/s ≈ 178GB/s（参考 33.59 ≈ 316GB/s，5060Ti HBM ~448GB/s）。
- 已试（均 env 默认关）：
  - `NINFER_TRELLIS_PFDECODE=1`：decode 走新 staged prefill GEMM（无 K 切分）→ 8.3 tok/s，occupancy/延迟更差。
  - `NINFER_TRELLIS_OLDMMA=1` + PFDECODE：旧大批量 staged MMA 跑单 token → 2.1 tok/s。
  - `NINFER_TRELLIS_STAGEDEC=1`：split 内核加 3 级 cp.async 代码分片流水 → 16.8 tok/s，略低于旧 18.9；**代码流水/合并不是瓶颈**。
- 待 ncu 定位 split 内核真实 stall（long scoreboard / MIO / 张量核 / 带宽 sector）。
- 待移植参考 gemv.cu：prepare 分块 fp16 fragment（2^-e + 块和）、W4=8 warp/CTA、C4=8 chunk cp.async、-1528 块和延后修正、atomic-counter 末 CTA 融合归约、自适应 splits（目标 2×SM CTA）。

### splits 扫描（2026-11-25 追加，默认仍 16）

env `NINFER_TRELLIS_SPLITS=<n>` 覆盖 K 切分数（数学等价，reduce 按实际 splits 求和）；workspace 固定 16 平面，n 上限 16。

| splits | decode tok/s | prefill tok/s |
| --- | --- | --- |
| 1 | 10.2 | 9.7 |
| 3 | 16.0 | 13.8 |
| 4 | 16.7 | 14.0 |
| 8 | 18.3 | 14.3 |
| 16 | 19.0 | 14.9 |
| 24 | 18.8 | 14.8 |
| 32 | 18.5 | 14.4 |

- 提速随 splits 上升、16 到顶，24/32 回落：**并发/占用在 16 split 已饱和**，不是单纯带宽 bound；继续加 split 只会增加 partial 写入与 reduce 开销。
- 32 临时验证过（workspace 曾放到 32 平面），确认无收益后已回退 kDecodeSplits=16。
- ncu 不可用：本机非提权、消费卡计数器锁定，`ERR_NVGPUCTRPERM`；stall 类别（long scoreboard/MIO/张量核/带宽 sector）无法用 Nsight Compute 采集，只能靠代码对照推断。需管理员 + 计数器权限才能跑 `--section SpeedOfLight/WarpStateStats`。

### 参考 gemv 与现有 split 的结构差异（重构清单，task #9）

1. **权重置 A、激活置 B**：参考 `gemv_body` 解码权重成 A fragment（`af[4]`），激活预生成 B fragment；每 k-tile 只 1 个 mma16816。现有 split 反过来（激活 A、权重 B），每 tile 发 2 个 mma（n8 两半），且 A 的 8..15 行补零。
2. **prepare 预生成 fp16 B-fragment**：参考 `prepare_k/prep_block` 每 128 块定 2^e、存 fp16 fragment + 块和；body 内层只从 smem 读半字。现有 split 在热循环内逐 tile 重读 xa(F32)+s16，做 fmul/f16 cvt/pack。
3. **chunk-of-8 cp.async 流水**：参考 C4=8 k-tile/chunk，S=2 级，每 chunk 一次 `__syncthreads`；代码 16B 协作拷 smem。现有 STAGEDEC 变体按 tile 拷 + 每 tile 两次 syncthreads，流水被 barrier 抵消（16.8 略差）。
4. **-1528 块和延后修正**：参考 body `acc += (g - 1528·sum)·2^-e`；现有 split 在 fragment 内 `hsub2` 减 1528。
5. **自适应 splits**：`split_kb` 目标 2×SM CTA、kb 为 8 的倍数、≤128；单 row max 16。参考 K=5120(kt320) 实际 splits=3，而非固定 16。
6. **atomic-counter 末 CTA 融合归约**：末位 CTA 在 smem 求和 + 1/4 + had/post 直接出栈，省掉独立 `reduce_split_kernel` 与 partial 平面往返。
7. **decode 前置 gate**：现有 gate 只覆盖 T=4 prefill，decode T=1 无数值对拍；重构前应先补一个 T=1 的 probe，对参考 dump 比 cos/L2，避免盲改只靠文本连贯度掩盖漂移。

### 2026-11-25 深夜：T=1 decode 门控补齐 + workspace 对齐修复

- `test_trellis_real.cpp` 抽出 `check_tokens(probe, tokens, ...)`，同一 probe 跑 T=4（prefill GEMM）与 T=1（split-K decode GEMV），对主机 `project_host` 算相对 L2（容差 2/256）。参考值主机现算，无需新 dump。
- 门控结果（全绿）：

| 权重 | bits | T=4 L2 | T=1 L2 |
| --- | --- | --- | --- |
| ffn_up 17408×5120 | 2 | 0.00167733 | 0.00166957 |
| attn_qkv 10240×5120 | 3 | 0.00169177 | 0.00168206 |
| band00 15488×5120 | 6 | 0.00167628 | 0.00167918 |

- **修复 workspace 欠分配**：`trellis_linear_workspace_bytes` 原先按打包原始字节算，未给每个 256 对齐平面留填充；T=1 时末位 `partials` 分配 `end > cap_` 抛 `std::bad_alloc`（arena.cu:325）。改为逐平面（xa/yr/tmp×2/partials/xs/xf）`round256` 后求和。引擎长 prompt 路径此前因平面大、填充占比小未暴露。
- 复测正式基线（cur9，390 prompt / 125 new）：prefill 514.9 tok/s、decode 19.5 tok/s，无回归。
- 前置条件已满足：decode split 内核现在有数值对拍，可安全按 task #9 清单做参考 gemv 重构，不必再只靠文本连贯度判断。

## gemv_single 实现蓝图（task #9，下一次直接照此落地）

目标：decode T=1 专用 GEMV，取向与流水对齐参考 gemv.cu，复用已门控的 pf_mma 已验证片段。

### 可直接复用的现成件
- prologue：`trellis_prologue_f32_kernel`（369 行）已按 token 定 2^e、写 fp16 B-fragment 平面 xf + xs[0]=2^-e。decode 新路径先调它（dispatch 已为 tokens=1 分配 xf_span/xs）。
- 权重 A 碎片解码：`decode_tile_proven<BITS>`（309 行）输出 exact 4v A fragment af[4]，pf_mma 已门控验证。
- cp.async 原语：`cp16s/cp_commit/cp_wait`、`cp16s_l2_256`（common.cuh 181-189；本文件等价封装见 pf 476-492）。
- MMA：`mma_f16(c[4], a[4], b0, b1)`，A=权重 af，B=激活 uint2（pf 526 行调用顺序为准）。

### 新内核 `gemv_split_kernel<BITS>`（仿 pf_mma_kernel，GT=1）
- 线程 256（8 warp/CTA），grid.x = nt/8 列块，grid.z = splits；每 warp 一个 16 输出 tile J=blockIdx.x*8+warp。
- CK=8 k-tile/chunk，S=2 级 smem ring；smem 代码区 8*CK*WORDS 词，激活区只放 row0。
- 代码 staging 直接照搬 pf 456-463 协作 16B 拷贝；激活 staging 把 pf NA 映射改成单行：CK*16  halves/行，线程 0..31 拷 16B。
- 内层每 chunk：cp_wait<S-2> + __syncthreads（每 chunk 一次，不要每 tile 同步）；逐 tile 解码权重 af[4]，从 smem 激活区读 B uint2，**每 tile 1 个 mma**（pf 519-534 循环，去掉 GT 维）。
- 归约：acc[4]；partials[split*N + 输出] 写 `acc * 0.25 * xs[0]`（线性缩放可在 partial 侧乘，reduce 直接求和）。沿用现有 `reduce_split_kernel`。
- 启动器：env `NINFER_TRELLIS_GEMV=1` 选新内核；默认仍走旧 split。dispatch decode 分支在新路径下先跑 prologue 再调 gemv_split。

### 验证顺序
1. 编译 → T=1 gate（test_trellis_real，3 条 probe L2 ≤2/256）必须全绿；T=4 gate 不回归。
2. 短 prompt eager 测速；再正式 bench（bench.ps1 -New 128）对比 19.5 tok/s 基线。
3. 若 L2 漂移：优先核对 xf B 碎片寻址（prologue 395-408 与 pf 515/525 的 uint2 索引），其次权重 A fragment 的 row/col 顺序。

### 仍待移植的参考能力（确认新内核跑通后再做）
- 块和延后修正（-1528·sum）替换 fragment 内 hsub2；自适应 splits（split_kb 目标 2×SM CTA）；atomic-counter 末 CTA 融合归约，消独立 reduce 内核。

## 2026-11-25 默认 27B 转换/读取链路 vs NariQ coded 链路（回答“为什么转换后反而更慢、能不能复刻默认速度”）

### 结论先行

慢**不在转换器**。NariQ 转换器对 tile 码字逐字节拷贝（`tools/convert/trellis_gguf.py`），不上量化、不解码回 FP32，设备上码字与官方 nari-llama 读到的完全一致；加载期也是直传、不重打包。差距全部在**执行内核族的成熟度**：默认 27B 每个量化格式都有一套按形状、按 token 宽度调好的专用 GEMV/MMA 实例；trellis 目前只有一个通用 split GEMV 内核。能否追上：追官方 libtrellis 参考（同为 coded 表示）可行，按下面的 gemv 清单重构即可；追 NVFP4-A4 / FP8-A8 张量核吞吐则受 coded 无 A8/A4 路线这一数学约束，属另一条边界。

### 默认 native 27B：转换 → 落盘 → 读取 → 选路

1. **转换（Python，`tools/convert/`）**
   - recipe（`official_recipes.py`，如 `qwen3_8_27b`）为每个逻辑投影指定 `format`/`method`/`activation_policy`；`grouped_absmax`/`grouped_search` 把 FP32 源压成码字 + 组尺度（q4/q5/q6 组-64 FP16 尺度，q8 组-32）；`fp8_row_maxabs` 出逐行 FP8，NVFP4 走 `import_encoded` 原样导入。
   - `recipe.group` 把 Q/K 并成一个 Q4 parent、gate/V 并成一个 Q5 parent、`[gate;up]` 并成一个 parent；打包规则在 `tools/artifact/formats` 与 storage layouts。
2. **加载（C++，`src/models/qwen3_5/load/`）**：配置展开成逻辑名/形状/Use；loader 把存储表示**原样上传，不做运行期重打包**（`weight-conversion.md` “The loader uploads the stored representation, without repacking”）。`execution/parameters.cpp` 按 `is_coded(format)` 分流成 coded view 或 `native_weight(view)`。
3. **分发（`src/ops/linear/linear.cpp::dispatch_linear`）**：`is_coded` → `trellis_linear`；否则 sm70 路由；否则 gguf 桥；否则 `switch(w.qtype)` → `q4/q5/q6/q8/fp8/nvfp4/bf16/t2/ptq1` 各自 dispatch。
4. **内核族（以 q4 为代表，`src/ops/linear/q4/`）**：
   - 形状表 `kShapes` 按 `(n,k)` 逐条登记（1024/5120、6144/5120、17408/5120、5120/17408、34816/5120、248320/5120 …），每个形状的 `select(n,k,t)` 再按 token 宽度 T 选实例。
   - decode（T=1）：`GemvR4W1` / `GemvR1W8K5120` 等专用 GEMV；prefill：`SlicedR16T8/T16/T32`、R32T8 等 sliced-K / MMA 实例；另有 ksplit 路线。
   - GEMV schedule 是一张十余个模板参数的表：`RowsPerCta / WarpsPerRow / GroupsPerWarpTile / PipelineStages(1..8 cp.async) / ActivationAccess(direct|cta-shared full-K) / LaneMapping / DecodeMode / CodeTransfer(sync|async vector16) / ScaleAccess / CodeCache / StaticGroupsPerRow / MinBlocksPerSm`（`q4_rowsplit_gemv.cuh` 40-57）。即每个 (形状, 宽度) 都有调好的占用/流水/拷贝/尺度访问组合。

### NariQ coded 链路现状

1. **转换（`tools/convert/trellis_gguf.py`）**：`GGML_TYPE_TRL1..6` tile 码字按 `N*(K/16)*8*bits` 逐字节写进 `trellis_tile16_v1`；s16 由 F16 精确升 F32；`pre/post/pin/pout/desc` 各自成对象；banded head 按 band 顺序拼接 + `vocab_gather` 散列。**不二次量化、不解码**。
2. **加载**：coded `Weight` 带 `trellis_desc` 与 side plane 辅助；`is_coded` 命中后只走 trellis plan，`LinearPolicy` 被忽略，**没有 A8/A4 备选路线**（`linear.cpp` 102-110 注释）。
3. **内核**：prefill `pf_mma_kernel`（已按 libtrellis gemm 移植过门控）；decode 只有一个通用 `tile_product_mma_split_kernel`：固定 256 线程、`kDecodeSplits=16`、独立 `reduce_split_kernel` + partial 平面往返；无按形状/宽度的实例表；decode 热循环内逐 tile 重读 xa(F32)+s16 做 fmul/cvt/pack；激活置 A、权重置 B，每 tile 两个 mma 且 A 8..15 行补零；按 tile 两次 syncthreads；splits 固定。

### 实测差距与带宽归因（cur7/cur9）

| 路径 | prefill tok/s | decode tok/s | decode 带宽反推 |
| --- | ---: | ---: | ---: |
| 官方 libtrellis 参考 | 675.6 | 33.59 | ≈316 GB/s |
| ninfer native GSQ 主线 | 747 | 23.5 | — |
| 本机 trellis 当前（cur9） | 514.9 | 19.5 | ≈178 GB/s |

- decode split GEMV 45.1ms/tok、占设备 87.4%；16 split 并发已饱和（24/32 回落），不是寄存器/占用受限，是内核访存/流水结构未对齐参考。
- prefill 缺口较小（515 vs 676）：side/gather/post/gather_scale 偏重，缺窄 N split-K 与 `cp16s_l2_256` 预取。

### 能不能复刻、从哪一侧复刻

| 方面 | 转换器侧能否改善 | 内核侧对应动作 |
| --- | --- | --- |
| decode 带宽/流水 | 几乎无空间：码字已逐字节落盘，设备布局即参考布局 | 按 gemv 蓝图：权重置 A fragment、激活预生成 fp16 B fragment、C4=8 chunk 两级 cp.async、每 chunk 一次同步、-1528 块和延后修正、自适应 splits、atomic 末 CTA 融合归约 |
| 形状专用化 | 不涉及 | 借鉴 native：建 `(n,k)` 启动器表，T=1 走专用 GEMV、宽 T 走 sliced-K/MMA，而不是一个通用内核吃所有形状 |
| 激活精度路线 | coded 数学上无 A8/A4（FP32 旋转），不可照搬 NVFP4-A4/FP8-A8 张量核路线 | 保持 A16/fp16 mma；目标先对齐 libtrellis A16 参考的 33.6 decode / 676 prefill |
| 主机侧 | 转换期无推理开销；运行期 `plans_for` 互斥 + 字符串 key、每调用重复 validate/parse 是次要清理项（trace submit 0.1ms，非 decode 主瓶颈） | 计划结果可启动期一次固化 |

**一句话**：默认快，是因为每种量化都配了按形状、按宽度调好的一整套专用 GEMV/MMA 实例和 schedule；NariQ 慢，是 coded 这套刚移植、decode 还停在通用 split 内核。追参考速度要在内核族上补齐专用 GEMV 与流水，改转换器帮不上忙。

## decode GEMV 移植落地（cur10）

### 内核结构

`gemv_split_kernel<BITS>`（`backends/ninfer/src/ops/trellis/trellis_kernels.cu`，蓝图第 1 条）：

1. **方向反转**：权重当 A 操作数，用 `decode_tile_proven` 解出 exact 4v fragment；激活用 prologue 预生成的 fp16 B fragment。每 k-tile 一个 `m16n8k16`（旧 split 要两个，且 A 的 8..15 行补零）。
2. **CK=8 chunk（正好一个 128 输入块）+ 2 级 cp.async 环**：chunk c+1 的拷贝在本 chunk 乘加之前发出，每 chunk 一次 `__syncthreads`，decode 不再等本轮发出的全局读（旧内核是逐 tile 的 LDG → decode 依赖链）。
3. **激活 staging** 拷该 128 块的整块行组 slab（512 词 = 128 个 16B 单元），fragment 寻址沿用 pf 的 `uint2 d*32+q`；mma 八列装的是同一个 token，只存 `q==0` 的第 0 列（输出 row `lane/4` 与 `+8`）。
4. **拆 K 按 chunk 对齐**：`kb = ceil(kt/splits/CK)*CK`，`active = ceil(kt/kb)`；partials 仍由 `reduce_split_kernel` 求和，`0.25 * xs[0]`（码字因子 4 与行 2^-e）在 partial 上乘，线性不损精度。
5. `NINFER_TRELLIS_GEMV` 默认开，`=0` 回旧 split；`K % 128 != 0` 或 `kt < 8` 自动回落。

### 门控（`ninfer_trellis_real_test` + `I:\model\trellis-probe`）

| 权重 | T=1 相对 L2 | T=4 相对 L2 |
| --- | ---: | ---: |
| blk.0.ffn_up TRL2 17408x5120 | 0.00166077 | 0.00167733 |
| blk.0.attn_qkv TRL3 10240x5120 | 0.00168215 | 0.00169177 |
| output.band00 TRL6 15488x5120 | 0.001677 | 0.00167628 |

T=4 路径完全没动（同一份 pf 内核），T=1 与旧 split 同量级（差在 partial 求和顺序），三条全绿。

### 测速（128 new，同一 artifact，`--max-context 1024 --greedy`）

| 路径 | prefill tok/s | decode tok/s |
| --- | ---: | ---: |
| 旧 split（`GEMV=0`） | 495.6 / 497.6 | 18.8 / 18.8 |
| 新 GEMV（默认） | 494.3 / 496.5 / 503.8 | 20.4 / 20.4 / 20.5 |

prefill 不变（同一路径，符合预期），decode 18.8 → 20.4（+8.5%），对 cur9 基线 19.5 是 +4.6%。两次交替复测完全一致，不是噪声。

### 单变量 A/B（96 new）

- **splits**：16 → 20.4；12 → 20.2；8 → 20.2；4 → 19.3；3 → 19.2；2 → 18.2。并发越多越好，参考在 5080 上用的 3 split 在这张卡不成立。
- **级数**：S=2 → 20.4；S=3 → 20.3（smem 30KB 把占用压到 3 CTA/SM）。
- **CTA 形状**：8 warp → 20.4；4 warp（CTAs 翻倍、环减半）→ 19.9。
- **`cp.async` 的 `L2::256B` 预取**：开/关都 20.4，中性，没进内核。

### 结论与下一步

- 结构对齐参考后，每 k-tile 少一个 mma、少 4 次 F32 全局读 + 4 次 fmul + 2 次 pack，decode 提了 8.5%。但 A/B 说明**并发（splits）比单 CTA 流水更值钱**：瓶颈仍在每 CTA 的码字读入延迟，不在 mma 数量，所以离参考 33.6 的 1.65 倍缺口还没补上。
- 注：这台机器上 nsys/CUPTI 的 kernel 时长不可信（decode 走 graph 回放时漏记，同一内核能差近一个数量级），口径一律用 engine 的 e2e 数字。
- 下一批按收益排序：
  1. `-1528` 块和延后修正：热循环每 tile 4 个 `hsub2` 换成每 chunk 4 个 fma（需要 prologue 或内核多算一份 128 块激活和）。
  2. 末 CTA `atomic` 融合归约：省掉 `reduce_split_kernel` 与 partial 平面往返（每 token 约 70MB 写+读）。
  3. 码字读入加长：按 warp 的 2.3KB 流仍偏短，试一次发两个 chunk（CK=16，激活区扩到两块）或 `ld.global.nc` 手动双缓冲。

## 2026-11-25 三种产物共同支持：GSQ 回归的两个根因与修复

### 现象

在 `exp/nariq` 工作区跑历史 GSQ 产物（`Qwen3.8-27B-GSQ-RCO-IQ3_S-q4km-mtp-dflash2.ninfer`）：

| 路径 | dspine 分支 | nariq 工作区（修复前） |
| --- | --- | --- |
| 短提示 `The capital of France is` | ` Paris.` | ` a country in Europe.` |
| dflash2 draft 5 | 54.8% / 84.7 tok/s | 0% / 23.6 tok/s，输出乱码 |
| MTP draft 3 | 60.6% / 63.5 tok/s | 0% / 23.5 tok/s，输出乱码 |

`--spec none` 还会在 `Parameters` 构造后崩溃（0xC0000005）。

### 根因 1：norm 的 `unit_offset` 被全局改成 plain

nariq 工作区把所有 `ops::rmsnorm` 的 `true`（`gain = 1+w`）改成 `false`（`gain = w`），
但 GSQ 产物是**逐权重混合**：

| 权重 | GSQ 存储 | NariQ 存储 |
| --- | --- | --- |
| block `input_norm` / `post_attention_norm` | `w-1`（offset） | `w`（plain） |
| attention `q_norm` / `k_norm` | `w-1` | `w` |
| `final_norm` | `w-1` | `w` |
| GDN 内部 `gdn/norm` | `w` | `w` |
| draft norm | `w` | `w` |

判定规则：权重最小值 `< -0.08` → offset，否则 plain。同一转换产物内部一致，所以按**组件级**
（首层 `input_norm` 的结果）统一 text/MTP 的所有 norm；draft 独立判定。

fused D256 q/k norm 原本是编译期 `RmsEpilogue::Plain`，改成运行时
`gain = w + (offset ? 1 : 0)`；融合 int8/nvfp4 的 input-norm 路径只在 offset 时启用。

### 根因 2：GDN value-head 布局两种历史约定

`v` 的第 h 个 value head 读哪个 q/k head，两种 HF 布局都出现过：

- `repeat`（HF `q.repeat(1,1,n_v/n_k,1)`）：`h % H_qk`，NariQ 产物；
- `repeat_interleave`（旧 ninfer-v3 转换器）：`h / (H_v/H_qk)`，GSQ 产物。

修复：`head_map` 增加 `interleave` 字段与 `publishes_key()`，`two_stage::Arguments` 增加
`head_interleave`；加载时按 artifact 判定：

- `metadata` 含 `trellis.version` → `repeat`（NariQ coded）；
- 否则 → `repeat_interleave`（ninfer-v3 历史产物）；
- artifact 写了 `gdn.head_layout` 时以它为准；`NINFER_GDN_HEAD_LAYOUT=repeat|repeat_interleave`
  可覆盖（A/B 用）。

### 验证（同机、同产物、greedy）

| 门禁 | 结果 |
| --- | --- |
| GSQ 短提示 | ` Paris. / Berlin / Italy` |
| GSQ + dflash2 draft 5 | **54.8% / 86.2 tok/s** |
| GSQ + MTP draft 3 | **60.6% / 66.2 tok/s** |
| NariQ 短提示 | ` Paris. / Berlin / Italy` |
| NariQ PPL(c=64, corpus64) | **9.95056**（旧基线 9.90995，+0.41%） |
| NariQ prefill / decode（481/128） | **588.7 / 21.2**（修复前 585.7 / 21.3） |
| `test_trellis_real` | ok（rel L2 0.00162–0.00168） |

### 调试经验

- `git stash` 做基线对比后，HEAD 编译留下的 object 与恢复后的 `parameters.h` 布局不一致，
  表现为 `Parameters` 构造/首轮前向的 0xC0000005 或 heap corruption。恢复改动后必须清理
  `build-backends/ninfer/src/models`、`src/runtime`、`apps` 并重编，否则 Ninja 的增量依赖
  不会覆盖头文件布局变化。
- `NINFER_NORM_TRACE=1` 打印每个 norm 的 `min` 与判定，是定位 norm 模式最快的入口。

## decode 调优的下一步：本轮实测与结论

### splits 自适应方向已证伪

NariQ decode（481 提示 / 128 new）按 `NINFER_TRELLIS_SPLITS` 扫描：

| splits | 4 | 8 | 12 | 16（默认） |
| --- | ---: | ---: | ---: | ---: |
| decode tok/s | 19.4 | 20.5 | 20.3 | **21.2** |

profile 显示小 n 投影的固定开销很大（`n=48` 的 GDN gating 35.7 µs/call、`n=1024` 44.4 µs），
但它们不是靠减少 split 能救的：全局降到 4/8/12 都会伤到大矩阵。按 n 自适应 split 的方向据此
放弃，除非把 gather_scale / prologue / reduce 融合成一次 launch。

### 剩余两项（未做）

1. **1528 块和延后修正**：`decode_tile_proven` 目前每个 k-tile 做 4 条 `hsub2`；参考实现把
   `4v+1528` 直接喂 MMA，在 epilogue 用 `(c - 1528·Σx)·2^-e` 修正。预期 2–5%。
2. **最后 CTA 融合归约**：每个 coded linear 现在多一个 `reduce_split_kernel` 和 partials
   平面往返；用 atomic counter 让最后一个 split 的 CTA 直接做 reduce + post。预期 2–3%。

两项都不改变数学，但都要改热循环/启动器，建议单独一轮并用 `test_trellis_real` 与 PPL 双门禁。

### decode 21.2 vs 参考 33.6 的差距在哪

profile 的 T1 条目里，`lin ... T1` 的 product 阶段（gemv）只占约四分之一；其余是注意力、
GDN 的小矩阵、gather/prologue/reduce/post 与 KV 读取。参考的 33.6 建立在其整套 decode 调度
（多实例、形状专用、融合 prologue/epilogue）上，不是一个 GEMV 内核能补齐的。下一轮应先按
形状列出 T1 的 per-call 预算，再决定是否值得继续往 coded 内核里投。

## NariQ 与 DFlash2 / MTP 的关系（2026-11-25 检查）

### coded 量化不是挂载失败的原因

DFlash2 只依赖 target 的三样东西：指定层的 hidden taps、token embedding、输出 head。NariQ 三样都支持
（coded embedding 走 `trellis_embedding`，banded head 走 banded 投影 + `candidate_topk`）。真正的
阻塞是 **replay-record GDN**：验证轮要求 GDN 只算不写状态。

### 已修复：CUDA Graph 与 coded record

`--no-cuda-graph` 下 NariQ + dflash2 的 verify 首轮输出正确（` Paris.`）；带 graph 时同一轮乱码。
根因是 coded record 路径在 capture 期间做 device→host 读 state slot。修复：coded 权重 + 任意
speculative 后端时关闭 CUDA Graph（单步 decode 的 graph 保留）。

### 剩余：verify 后续轮次的状态折叠

`--no-cuda-graph`、draft-tokens 2：前两个 token 正确（` Paris is`），第 3 个起乱，acceptance 0%。
说明 `gdn_projection_record` 的首轮 q/k/v 与 replay 输出可用，但轮次间的状态更新（replay records /
conv state）不正确。下一步应对比 `records.conv`（应为卷积输入 qkv）与 fold 后 `conv_layer0` 的
`[C,3]` 窗口，以及 `key/value/gate` records 在 fold 中的消费。

### MTP GGUF（`Qwen3.8-27B-NariQ-3.0bpw-MTP.gguf`）

第 65 层是 `blk.64.nextn.*`，16 个参数与 GSQ 的 `mtp/...` 一一对应：

| GGUF | engine |
| --- | --- |
| `blk.64.nextn.eh_proj` | `mtp/input_projection` |
| `blk.64.nextn.enorm` / `hnorm` / `shared_head_norm` | `mtp/embedding_norm` / `hidden_norm` / `final_norm` |
| `blk.64.attn_norm` / `post_attention_norm` | `mtp/layers/0/input_norm` / `post_attention_norm` |
| `blk.64.attn_q` (12288x5120) | `mtp/layers/0/attention/packed` |
| `blk.64.attn_k` / `attn_v` / `attn_output` | key / value / output |
| `blk.64.attn_q_norm` / `attn_k_norm` | query_norm / key_norm |
| `blk.64.ffn_gate` / `up` / `down` | mlp gate / up / down |
| `blk.64.nextn.embed_tokens` / `shared_head_head` | 跳过（用 target 的） |

当前 `convert_nariq.py` / `nariq_model.py` 还不能正确转：

1. `text_config` 写 `num_hidden_layers = 65`，`blk.64` 被当成普通 text 层；
2. `plan()` 没有 MTP 命名映射；
3. MTP 权重是 Q4_K/Q6_K，`read_dense` 只支持 F32/F16/BF16/I32，需要 GGUF 块直传（engine 的
   `bind_mtp` 支持 `is_gguf`）或解码后重量化。

MTP 的验证同样走 `target_verify_batch` 的 `RecordForReplay`，所以即使转换器补齐，NariQ + MTP 仍受
上面同一个 record 状态问题限制。
