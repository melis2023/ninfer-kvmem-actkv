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
- banded head：16 个 band 拼成一张矩阵（band 顺序），`pout = argsort(vocab_gather)`；
  要求 band 之间 `pre`/`s16`/desc 完全一致，否则拒绝（一个 coded linear 只能带一套）。

## 3. 已验证

```powershell
cd backends\ninfer
python -m tools.trellis.verify_trellis selftest     # 合成 NariQ GGUF → .ninfer → 读回 → 解码
```

当前输出：

```text
converted 2 coded matrices -> 7808 bytes
  blk.0.attn_q.weight: (32, 64) trellis_trl3 grid=[-111,105] w_std=0.7662
  output.weight: (32, 64) trellis_trl3 grid=[-110,114] w_std=0.9570
selftest ok
```

覆盖到的：GGUF 类型 44..49 的读取、desc/侧张量校验、banded head 拼接与 `pout` 组合、
`.ninfer` 目录/对象/编码尺寸校验（`Artifact.open` 会拒绝任何尺寸不符）、
**两种独立 tile 解码写法逐位一致**、解码网格落在 `[-126,126]`。

未覆盖（下一步的门槛）：

1. **主机参考 ↔ 设备实现逐元素一致**：`trellis_reference.cpp` 是权威，但目前没有编译好的
   `trellis_reference_test`；需要在能出 `sm_120a` 的机器上核对 tile 乘、RHT、Paley 交叉。
2. **真实权重**：本地没有 `Qwen3.8-27B-NariQ-*.gguf`，所有数字都来自合成 tile。
3. **模型接线**：见下。

## 4. 还缺什么（按顺序）

| # | 内容 | 位置 |
|---|---|---|
| 1 | loader 把 Use auxiliaries 落进 `Weight::trellis_*`（s16/pre/post/pin/pout + desc）：binder 目前只认 `input_columns` 这一个 auxiliary | `src/artifact/binder.cpp`、`src/core/weight_view.cpp` |
| 2 | `text/token_embedding`、`text/output_head` 接受 coded 权重；`embed_tokens` 加 coded 分支（`ops::trellis_embedding` 已就绪）；banded head 在 `project` 处改成"16 band + `pout`"（转换器已输出该语义的单张矩阵，引擎只需一次 Op） | `src/models/qwen3_5/load/text.cpp`、`execution/text.cpp` |
| 3 | KVMem / 白名单：`serve_options` 对 coded 权重的 target 放行 `--spec dflash2`；`--spec dspine` 要等 DSpine 自己的 KVMem 打通（`907ed3bb3` 之后仍被拒） | `src/serve/serve_options.cpp` |
| 4 | 性能：当前 `tile_product` 是 smem int8 + FP32 SIMT，正确优先；decode 走 sm_80+ INT8 tensor core、prefill 复用 A8 主循环是后续 1.5–2 周的量 | `src/ops/trellis/trellis_kernels.cu` |
| 5 | 视觉：NariQ 发布只有 BF16 mmproj，与"视觉必须 Q4"的约定冲突，需单独决策 | 转换路径 |

## 5. 已知风险

- 官方 kernel 只在 RTX PRO 6000 上调过；本机（5060 Ti / sm_120）不要假设 88.9 tok/s 可复现。
- 16 GB 显存：3.0bpw + BF16 视觉 + DFlash2 drafter ≈ 11.5 GB，KV 只剩约 4 GB；没有 KVMem 时
  128K 上下文基本不可能。
- coded 权重无法 `llama-quantize` 重量化、不接受 LoRA（与 nari-llama 的限制一致）。
- `pout`-only 的 banded head 表达要求 band 间 `pre/s16/desc` 一致；发布文件满足，但未来若不一致，
  只能改成"每 band 一次 Op"。
