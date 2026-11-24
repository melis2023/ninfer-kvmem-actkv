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
