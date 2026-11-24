# NariQ / nari-llama.cpp 分析：能不能移植到 `exp/dspine`

分析对象：<https://github.com/nari-labs/nari-llama.cpp>（HEAD `8e0e016`，分支 `nariq/b12431`）
本地副本：`%TEMP%\opencode\nari-llama`（shallow clone）
结论文件：本页。写于 2026-11-24。

## 0. 结论摘要

- **NariQ 不是一种"新量化格式"，而是一套"编码权重 + 激活侧旋转 + 分组解码"的算子方案**：
  GGML 里新增 6 个类型（`TRL1..TRL6`）、2 个算子（`GGML_OP_TRELLIS`、`GGML_OP_TRELLIS_GET_ROWS`）、
  约 4.5k 行新代码（含 121 KB CUDA 库）。
- **它已经是一个完整可用的 llama.cpp fork**，自带 `llama-server`、`llama-cli`、mtmd 视觉、DFlash/DFlash2 投机解码
  （上游 b12431 本身就有 `src/models/dflash2.cpp`）。
- **对 `exp/dspine`（ninfer 后端）没有"打补丁就能开"的路径**：ninfer 不是 ggml，没有类型表/算子图，
  权重格式是封闭白名单。要吃到 NariQ，只有两条现实路线：
  1. **把 nari-llama 当第三个后端接进来**（`backends/nariq`，约 1 周）；可选把已有的
     `llamacpp-kvmem.patch` 重放上去，就同时有 KVMem。
  2. **在 ninfer 里新写 coded-linear 算子**（约 3–5 周，主要是 prefill GEMM），才能和 DSpine / KVMem /
     KV 量化共用同一套栈。
- **实测发布的 NariQ 文件没有 MTP 头**（64 层、`nextn_predict_layers` 缺失、无 `blk.64`），
  所以 `--spec mtp` 不可能用；KVMem 的白名单只剩 **单卡 DFlash2** 一条路。
- 视觉只有 **BF16 mmproj**（0.87 GB），和"视觉必须 Q4"的约定冲突，需要单独决策。

---

## 1. 这个项目是什么

| 项 | 内容 |
|---|---|
| 基础 | `ggml-org/llama.cpp` tag `b12431`（commit `a946eb5`）的 fork，单分支 `nariq/b12431` |
| 规模 | 相对上游 `+10,374 / -10,809` 行，其中 trellis 相关约 4.5k 行新增 |
| 许可 | 仓库 MIT；`ggml/src/ggml-cuda/libtrellis/` 单独 Apache-2.0（与 ninfer 的 Apache-2.0 兼容） |
| 稳定性 | README 明确"研究性质"，Kernel 只针对 RTX PRO 6000 调优，**其他卡上比同尺寸普通 GGUF 慢** |
| 依赖 | CUDA compute capability ≥ 8.0（sm_80+）才走 GPU，否则自动退回 CPU（慢到不可用） |
| 生态 | 上游 llama.cpp / Ollama / LM Studio 都读不了这些文件（自定义 type id 44..49） |

发布权重：`nari-labs/Qwen3.8-27B-NariQ-GGUF`，2.0 / 2.5 / 3.0 bpw 三个文件（6.29 / 7.86 / 9.43 GB）
+ `mmproj-BF16.gguf`，上下文声明 262144。

官方质量/速度（README，RTX PRO 6000，单会话）：

| Qwen3.8-27B | Size | DeepSWE solved | DeepSWE | TB 2.1 | AIME | 4K/64K decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 53.8 GB | 44 | 83.0 | 62.6 | 97.7 | — |
| NariQ 3.0bpw | 10.1 GB | **47** | **84.9** | 63.8 | 98.3 | 88.9 / 70.7 |
| NariQ 2.5bpw | 8.4 GB | 46 | 84.5 | 60.2 | 96.5 | 99.2 / 77.5 |
| NariQ 2.0bpw | 6.8 GB | 33 | 75.7 | 52.4 | 86.0 | 111.7 / 85.0 |

3.0bpw 在 agent 基准上**超过 BF16**（47 vs 44）——这是"量化当正则"的说法，样本 113 题，
只能当趋势看，但至少说明 3.0bpw 不是"能跑就行"的档位。

---

## 2. NariQ 格式机制（v1，`nariq/FORMAT.md`）

### 2.1 数学定义

一个 coded linear 有 N 个输出、K 个输入（`W` 是 `[K, N]`，ggml 的维度顺序）：

```text
y = gather_pout( post ⊙ T_rows^T ( Ŵ · T_cols ( pre ⊙ gather_pin(x) ) ) )
Ŵ[n][k] = s16[k/16] · v( state[n][k] )          # v ∈ [-126, 126] 的 INT8 网格
```

- `T_cols`：输入侧变换（kind 1 = 分块 Sylvester Hadamard；kind 2 = 整片 Hadamard + Paley `H12/H20/H68`）。
- `T_rows`：输出侧变换，**用逆变换**；输出侧还允许把最多 8 个"切片"并排融合（`parts`/`shards`）。
- `pre` / `post`：F32 `[K]` / `[N]` 的符号×缩放，等价于随机符号对角阵。
- `s16`：F16 `[K/16]`，每 16 个输入一个缩放。
- `pin` / `pout`：可选 I32 排列，把 DeltaNet 的 V-head 顺序、转换器里的 `perm_rows/perm_cols` 吸收进来。

**旋转作用在激活上，不在权重上**——这是和 ninfer 现有 `hadamard_transform`（把符号烘进权重）最关键的差别：
每次前向都要对激活做 FWHT，decode 时靠 `tq_gemv_rht` 把这个成本融进 GEMV 的读取循环。

### 2.2 存储布局

- `GGML_TYPE_TRL1..TRL6` = id **44..49**，`blk_size = 16`，`type_size = 2*bits` 字节 → 1..6 bit/权重。
- **输出 tile 主序**：`for J in N/16: for i in K/16: 16*bits 个 int16 码字`（即 8*bits 个 uint32）。
- 每 16×16 的 tile 是 256 个权重，码字按"环"写，解码时取 16 位窗口查 INT8 网格
  （`state * 0x83DCD12D` 再折 4 段 6 bit 求和 −126），再按 exllamav3 的 tensor-core 顺序映射到 (row, col)。
- **没有 per-row `to_float`**：一行权重的字节不能独立解码，`llama-quantize` 无法重量化，
  所有"取某几行"的操作必须按 16 行一组。

### 2.3 侧张量与元数据

| 名字（`<w>` = 去掉 `.weight` 的前缀） | 类型/形状 |
|---|---|
| `<w>.trellis_s16` | F16 `[K/16]` |
| `<w>.trellis_pre` | F32 `[K]` |
| `<w>.trellis_post` | F32 `[N]` |
| `<w>.trellis_pin` | I32 `[K]`，可选 |
| `<w>.trellis_pout` | I32 `[N]`，可选 |
| `trellis.desc.<w>` | INT32 `[rows_kind, rows_block, rows_shards, cols_kind, cols_block, cols_shards, nparts, part0..partN]` |

`trellis.version` 必须是 UINT32 `1`。加载时 `check_trellis_gathers` 会验证 `pin/pout/vocab_gather/embed_gather`
是**排列**（否则后端越界读），`check_trellis_uses` 拒绝：编码权重被 Hadamard 折叠、被 tie 成 embedding、
被 MTP 块当普通 `get_rows`/matmul 使用。

### 2.4 banded head / coded embedding

- lm_head 拆成 `output.bandNN.weight`（16 个 band，按词频排序），加 `output.vocab_gather`（I32 词表排列）
  和 `trellis.lm_head_bands`；**没有** `output.weight`。graph 里 16 个 coded linear 依次 concat，
  再用一次 `get_rows` 还原成词表顺序。
- 实测 band 大小不齐：`output.band00.weight` 是 `[5120, 15488]`，16 个 band 合计 247,808 行，
  而词表 248,320——排列后按 `vocab_gather` 取行。
- `token_embd.weight` 也可以是 coded（本文件是 `TRL2`，`[5120, 248320]`），解码**在 CPU 上**逐行查表；
  带 band 的 coded embedding 用 `ggml_trellis_get_rows_band` 链式原地写入。

### 2.5 实测文件画像（3.0bpw）

用 GGUF 头解析得到（`%TEMP%\opencode\nariq_head.bin`，前 80 MB）：

- `general.architecture = qwen35`，`block_count = 64`，`embedding_length = 5120`，词表 248,320。
- 2,710 个张量，其中 **513 个 coded**（64 层 × 8 + embedding）：`TRL2` 82 个、`TRL3` 199、`TRL4` 99、
  `TRL5` 132、`TRL6` 1。
- 层内 8 个 coded 投影：`attn_qkv/attn_gate/attn_k/attn_q/attn_v/attn_output/ssm_alpha/ssm_beta/
  ssm_out/ffn_gate/ffn_up/ffn_down` 按层型交替（full-attn 层 4 个、DeltaNet 层 8 个）。
- **无 `nextn`/`eh_proj`/`shared_head`/`blk.64`** → 没有 MTP 头。
- 视觉是独立 `mmproj-BF16.gguf`，不在主文件里。

---

## 3. 为什么 `dspine` 分支不能"打补丁就开"

`backends/ninfer` 是一个**自研 CUDA 推理引擎**（Apache-2.0），不是 ggml：

- 没有 `GGML_TYPE_*` 类型表、没有 `GGML_OP_*` 计算图，算子按目录显式分派。
- 权重是 `core/weight.h` 的 `Weight{payload, qtype, shape, quant layout...}`，`QType` 是**封闭枚举**（32 个），
  artifact 格式也是白名单（`tools/artifact/formats.py` + `src/artifact/formats.cpp`）。
- 激活是 BF16，`ops::linear` 的输出也是 BF16；NariQ 的参考实现要求 F32 激活和 F32 输出。

所以"移植"= 在 ninfer 里**从零实现一个 coded-linear 算子族**，落地点是固定的五处：

| # | 位置 | 要做什么 |
|---|---|---|
| 1 | `src/core/weight.h`、`src/core/dtype.h` | 新增 `QType::TRELLIS_K`（或 6 个）+ `QuantLayout::Tile16`；`Weight` 之外挂 `s16/pre/post/pin/pout/desc`，生命周期跟随权重（`evictable_weight_pool` / `direct_storage_reader`） |
| 2 | `src/artifact/formats.cpp`、`tools/artifact/formats.py` | 新 `TrellisFormat`：**`GgufFormat` 的"行子集 = 字节拷贝"假设在这里不成立**（tile 主序、16 行一组），必须单独的物化/切片路径 |
| 3 | `tools/convert/gguf_blocks.py`、`tools/convert/sources/gguf.py` | 复制 TRL 块（不重量化）+ 搬侧张量与 `trellis.desc.*` + 16 个 band 与 `vocab_gather`；`trellis.version` 校验 |
| 4 | `src/ops/linear/`（新目录 `trellis/`）、`route_table`、`linear_workspace_capacity_bytes` | decode GEMV（≤64 列，sm_80+ 走 INT8 tensor core）+ prefill GEMM；激活侧旋转融合 |
| 5 | `src/models/qwen3_5/execution/text.cpp`、`load/` | lm_head 换成 16 band + gather；embedding 走 coded 查表；`project()` 改成 coded 分支 |

额外两个"看起来能复用、其实不行"的地方：

- `include/ninfer/ops/hadamard_transform.h` 只做 **1024 点** FWHT + 符号（K ∈ {5120,6144,17408}）。
  NariQ 用 128 / 2048 / 4096 点块 + 整片 `H12/H20/H68` Paley 变换（`kind 2` + `parts/shards`），
  需要把这个 Op 扩到任意 2 的幂块和 Paley 交叉矩阵，而不是"直接调用"。
- ninfer 的 INT8 mma 主循环（`ops/linear` 的 A8 路线）假设权重是"预量化好的 int8 + group scale"，
  NariQ 需要在 smem 里**现场解码**成 int8 再进 mma。这部分可以复用调度，但不能复用加载。

参考实现规模（nari-llama 侧要读的代码）：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `ggml/src/ggml-trellis.cpp` | 340 | 主机端参考：desc 校验、tile 解码、整权重量化还原（**移植时唯一可直接照抄的算法文档**） |
| `ggml/src/ggml-cpu/trellis.cpp` | 464 | CPU 参考执行（正确性对拍用） |
| `ggml/src/ggml-cuda/trellis.cu` | 596 | 算子外壳：gather → 输入变换 → gemv/gemm → 输出侧 → gather |
| `ggml/src/ggml-cuda/libtrellis/src/{gemv,gemm,rotate,decode}.cu` | 2,000 | 真正的 Kernel（GEMV 42 KB、GEMM 25 KB、旋转 26 KB） |
| `ggml/src/ggml-metal/kernels/trellis.metal` | 298 | Metal 版，可忽略 |

---

## 4. 三条路线

### 路线 A：不移植，直接用 nari-llama 二进制验证（1–2 天）

`nari-llama.cpp` 自带 OpenAI 兼容 `llama-server`，上游 b12431 已有 `dflash2`，所以可以：

```text
llama-server -m Qwen3.8-27B-NariQ-3.0bpw.gguf --mmproj ...-mmproj-BF16.gguf
             -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ctk q8_0 -ctv q8_0
```

先回答两个问题：**16 GB 卡上 128K agent 场景能不能跑**、**工具调用质量比现在的 IQ3_S + DFlash2 好多少**。
风险：无 KVMem（长上下文全量 KV），262144 官方只在 24 GB 卡上验到 ~135K；视觉只有 BF16。

### 路线 B：把 nari-llama 作为 `backends/nariq` 接进框架（约 1 周，推荐）

仓库本来就是 `backends/{llamacpp,ninfer}` + `versions.json` + patch + launcher 的结构，
再挂一个后端是既有路径，不需要写 Kernel：

1. submodule 锁 `8e0e016`（或 release tag `b12431-nariv0`），`backends/versions.json` 记 `base_commit`。
2. 把 `llamacpp-kvmem.patch` 重放到 nari-llama（两边都改 `ggml.c`、`ggml-cuda.cu`、`getrows.cu`、
   `llama-graph.cpp`、`qwen35.cpp`，冲突集中但**不重叠**：KVMem 加窗口选择，trellis 加算子）；
   `backends/llamacpp` 已经有了"把 KVMem 打到另一个 llama.cpp 代码库"的先例。
3. `scripts/windows/start-backend.ps1` 加一个 `-Backend nariq` 分支；worker bat 与现有
   `start-kvmem-ninfer.bat` 同构。
4. 收益：NariQ 权重 + DFlash2 投机 + （可选）KVMem + 视觉，全在 llama.cpp 生态里；
   代价：再维护一份 patch（三份 llama.cpp 树了）。

### 路线 C：ninfer 原生 coded-linear（3–5 周，只有需要"同一套栈"时才做）

只有当你坚持"NariQ 必须跑在 ninfer 的 KVMem / DSpine / KV 量化 / 显存分层里"才值得。
建议分阶段，每阶段都有可回退的验证点：

| 阶段 | 内容 | 验收 | 估计 |
|---|---|---|---|
| C0 | 主机端移植 `ggml_trellis_dequantize` + `desc` 校验（纯 C++，无 CUDA），把 NariQ 权重解成 FP32 | 与 nari-llama CPU 参考逐元素一致（≤1e-6） | 2–3 天 |
| C1 | decode GEMV：1–6 bit 现场解码 + 输入 RHT 融合 + `s16` + 输出侧变换 + `pin/pout` | 单层输出与 C0 一致；小模型（27B 单层/几层）greedy 逐字节对齐 nari-llama | 1 周 |
| C2 | 扩 `hadamard_transform`：任意 2 的幂块 + `H12/H20/H68` + `parts/shards` | 与 `ggml_trellis_hadfull_slice` 逐元素一致 | 3–4 天 |
| C3 | prefill GEMM：寄存器解码 → smem int8 → 复用现有 A8 mma 主循环 | prefill 正确性 + 在 16 GB 卡上不低于 IQ3_S 的 prefill 速度 | 1.5–2 周 |
| C4 | banded head（16 band + gather）+ coded embedding（查表落 GPU 或 CPU 暂存） | logits 与 nari-llama 一致；top-k 采样在词表序上工作 | 3–4 天 |
| C5 | 接入 artifact/转换器、`--spec dflash2` 验证、KVMem 白名单 | 128K 长上下文 + DFlash2 + IQ3_S 同级吞吐 | 1 周 |

**不建议**做的中间态：把 NariQ 解成 FP32 再压成 Q5_K/Q6_K 存成 ninfer artifact。
数学上无损（编码是可精确还原的），但 27B 压到 4.5 bpw ≈ 14.3 GiB，比现在的 IQ3_S 还大，
16 GB 卡放不下 target + drafter + KV，收益为负，只适合当 C0 的对拍工具。

---

## 5. 与现有加速栈的交互

| 组合 | 可行性 | 说明 |
|---|---|---|
| NariQ target + `--spec dflash2` | ✅ | 上游 b12431 已有 `dflash2.cpp`；投机解码只要求 target logits 精确 |
| NariQ target + `--spec dspine` | ❌（nari-llama）/ ✅（ninfer 移植后） | DSpine 是 ninfer 专属实现（ring 按绝对位置读写） |
| NariQ target + `--spec mtp` | ❌ | 发布文件没有 MTP 头；`check_trellis_uses` 也拒绝 banded head + MTP |
| NariQ + KVMem | 仅 DFlash2 单卡 | KVMem 白名单是 `ordinary/fixed/adaptive MTP1..15` 或单卡 DFlash2；当前 ninfer 构建对 `--spec dspine` 直接拒绝 |
| NariQ + 视觉 | ⚠️ | 只有 BF16 mmproj（0.87 GB），与"视觉必须 Q4"冲突 |
| KV 量化（`rk8v4` / `int8`） | ✅ | 与权重格式无关 |
| LoRA | ❌ | nari-llama 拒绝在 NariQ 权重上挂 adapter |
| `llama-quantize` 重量化 | ❌ | 无 per-row `to_float` |

---

## 6. 风险清单

1. **性能只在 RTX PRO 6000 上调过**（sm_100/120 的 tensor core 排布、128 点融合旋转）。
   5060 Ti 是 sm_120、显存带宽低很多，README 说"其他卡比同尺寸普通 GGUF 慢"——不要假设 88.9 tok/s 能复现。
2. **16 GB 显存**：3.0bpw 10.1 GB + 视觉 0.87 GB + DFlash2 drafter ≈ 11.5 GB，KV 只剩 ~4 GB。
   没有 KVMem 时 128K 上下文基本不可能；`-ctk/-ctv q8_0` 也只是缓解。
3. **banded head 每次前向都要算完 16 个 band**（`build_trellis_bands` 是全量 concat 再 gather），
   不是"按需算 band"，对 decode 是 248,320 × 5120 的完整 GEMV——和 IQ3_S 的全词表 head 同量级，
   但**行序被打散**，ninfer 若实现成 16 次小 GEMV 会额外损失带宽效率。
4. **coded embedding 在 CPU 上查表**：prefill 128K token 时每 token 一次 5120 维 FWHT 逆变换，
   在 nari-llama 里是已知瓶颈；移植时要落 GPU 才有意义。
5. 维护面：第三个 llama.cpp 树（llamacpp / nariq 各自带 patch）会让 `versions.json` 与 CI 变复杂。

---

## 7. 建议的下一步

1. **先走路线 A**（不改本仓库代码）：在 5060 Ti 上跑 NariQ 3.0bpw + 我们的 DFlash2 drafter，
   用现有的 192K / 鹈鹕 HTML / 工具调用三个基准取数。**这一步只回答"值不值"**，不承诺任何移植。
2. 若 decode 速度 ≥ 现在的 DFlash2（87.9 tok/s）且工具调用不再坏，再评估路线 B（1 周，收益明确、可回退）。
3. 只有在"必须共用 ninfer 的 KVMem/DSpine/显存分层"成为硬需求时，才启动路线 C，并按 C0→C1→C3 的顺序，
   每一阶段都以"与 nari-llama CPU 参考逐元素一致"为唯一门槛。

**当前不建议**直接在 `exp/dspine` 上开 C 的代码：`907ed3bb3` 之后 `KVMem + --spec dspine` 还被
`serve_options.cpp` 的白名单拒绝，DSpine 的 KVMem 打通本身还没做，再叠一个 crate 级的新算子会让
"谁把输出改坏了"无法二分定位。
