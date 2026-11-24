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
  （offset/rows/bits）写进 `trellis.head_bands`。引擎按 ggml 的语义执行：逐 band 出 logits，
  再按 `vocab_gather` 落到词表位置（scatter），等价于 `concat(bands)` + `get_rows(·, gather)`。

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

## 4. 还缺什么（按顺序）

| # | 内容 | 位置 |
|---|---|---|
| 1 | loader 把 Use auxiliaries 落进 `Weight::trellis_*`（s16/pre/post/pin/pout + desc）：binder 目前只认 `input_columns` 这一个 auxiliary | `src/artifact/binder.cpp`、`src/core/weight_view.cpp` |
| 2 | 参数命名：本转换器沿用 GGUF 名（`blk.0.attn_qkv.weight`…），`qwen3_5` 引擎要的是 `text/layers/0/...` —— 需要一张映射表，且 **coded 投影是"整矩阵"**，不能像 `gguf_blocks.py` 那样把 `attn_qkv` 拆成 query/key/value、把 `attn_q` 拆成 query+gate 的交错行（拆分会破坏 16 行 tile 组） | `tools/convert/*`、`src/models/qwen3_5/load/prepare.cpp` |
| 3 | 因此模型侧要能"一条 coded 投影 → 多路逻辑投影"：`attn_qkv`(10240) → q/k/v、`attn_q`(12288) → query/gate 交错、DeltaNet 的 value 头；这是把 ggml 图里 `split/reshape` 搬到 ninfer 的关键 | `src/models/qwen3_5/execution/text.cpp` |
| 4 | banded head：16 个 band 各一次 coded linear，再按 `trellis.head_bands` + `vocab_gather` **scatter** 到 logits（需要一个新的落行步骤，或把 band 循环放进 head 的 executor） | 同上 + `src/ops/trellis` |
| 5 | coded embedding（`token_embd`，TRL2、24.8 万行、CPU 查表是 nari-llama 已知瓶颈）接 `trellis_embedding` | 同上 |
| 6 | KVMem / 白名单：`serve_options` 对 coded 权重的 target 放行 `--spec dflash2`；`--spec dspine` 仍要等 DSpine 自己的 KVMem 打通（`907ed3bb3` 之后仍被拒） | `src/serve/serve_options.cpp` |
| 7 | 性能：当前 `tile_product` 是 smem int8 + FP32 SIMT，正确优先；decode 走 sm_80+ INT8 tensor core、prefill 复用 A8 主循环是后续 1.5–2 周的量 | `src/ops/trellis/trellis_kernels.cu` |
| 8 | 视觉：NariQ 发布只有 BF16 mmproj，与"视觉必须 Q4"的约定冲突，需单独决策 | 转换路径 |

## 5. 已知风险

- 官方 kernel 只在 RTX PRO 6000 上调过；本机（5060 Ti / sm_120）不要假设 88.9 tok/s 可复现。
- 16 GB 显存：3.0bpw + BF16 视觉 + DFlash2 drafter ≈ 11.5 GB，KV 只剩约 4 GB；没有 KVMem 时
  128K 上下文基本不可能。
- coded 权重无法 `llama-quantize` 重量化、不接受 LoRA（与 nari-llama 的限制一致）。
- `pout`-only 的 banded head 表达要求 band 间 `pre/s16/desc` 一致；发布文件满足，但未来若不一致，
  只能改成"每 band 一次 Op"。
