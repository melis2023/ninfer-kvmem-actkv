# KVMem + ActKV · ninfer 后端

基于 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 与 **ActKV**（[arXiv:2609.31395](https://arxiv.org/pdf/2609.31395)，中国科学技术大学提出）技术开发：

- **KVMem**：把长上下文稀疏化成固定窗口（`budget` 选取历史块 B + `gen_reserve` 追加工作区 R），Host/磁盘归档历史 KV，用一个固定大小的注意力窗口换来 200K/262K 的可用上下文。
- **ActKV**：面向行动的检索保持策略（LRFU 历史排序等），提升长会话中关键块的召回率。

**QQ 交流群：1040777853**

本项目仅供研究学习，稳定使用可前往kvmem原作者。

## 2026-10-26 更新：不再需要 `--no-cuda-graph`，推荐草稿换成 q3km

### 1. CUDA Graph 探针校准（+ 读数复测）

旧代码按拓扑盲预留约 **1 GiB** 给 CUDA Graph，实际只占 **90–145 MiB**，导致 KV 少给、
长上下文启动失败，只能 `--no-cuda-graph`。现在启动时用最小 KV 探针实测整套图，按
`max(实测, 256 MiB)` 规划。

WDDM 下探针读数会抖动（同一配置连续启动实测过 0.0 / 83.3 / 90.0 / 117.3 / 135.6 /
288.6 MiB），288.6 那次会把规划值抬高到容量之外。现在首次读数高于 256 MiB 时会
**复测一次取较小值**。

效果：

- **不需要再写 `--no-cuda-graph`**，开图即可；
- 256K 窗口下 `nvfp4` / `q4km` / `q3km` 等草稿都能加载，不再出现启动即爆显存；
- 开图 decode 比关图快约 6%，且 KV 容量比盲预留多约 700–800 MiB。

### 2. 推荐草稿：q3km（直通草稿，统一量化精度）

**当前引擎已经支持直通草稿**：草稿权重按源工件的 GGUF 块格式直接加载，不再统一重化成
Q8/NVFP4。q3km 工件里的 DFlash2 草稿实际精度：

| 张量 | 精度 |
|---|---|
| `attention/output`、`mlp/down`（各 5 个） | `Q4_K` |
| `feature_projection`、`candidate_selector/hidden_projection` | `Q3_K` |
| conv `kernel_projection` / `base_kernel`、norm、codebook | BF16（prepare 路径没有量化内核） |

**为什么这样做：降低显存占用。** 直通保留源块字节，不再为第二种精度另存一份；草稿按
Q3/Q4 块存储后，256K 窗口下省出来的显存留给 KV 和 CUDA Graph，开图也不需要
`--no-cuda-graph`。两个启动器启动后的 free 分别是 **863.8 MiB**（DFlash2 + nvfp4 KV）
和 **1,001.8 MiB**（MTP + int8 KV）。

#### 256K 速度对比（两个启动器，prefill 取第一遍）

同一 950K 字符 prompt（253,031 token）、输出 256 token、temperature 0、thinking off，
RTX 5060 Ti 16 GB：

| 启动器 | 草稿 | KV | B/R | prefill（第一遍） | decode | 接受率 | wall |
|---|---|---|---:|---:|---:|---:|---:|
| `start-kvmem-ninfer-dflash.bat` | dflash2 draft 5 | nvfp4 | 36,480/16,384 | **872.5 tok/s**（290 s） | 34.8 tok/s | 20.2%（88/435） | **584.0 s** |
| `start-kvmem-ninfer.bat` | MTP draft 3 adaptive | int8 | 36,864/16,384 | 803.3 tok/s（315 s） | **47.5 tok/s** | **40.7%**（140/344） | 630.9 s |

- **DFlash2 prefill 快 8.6%、总 wall 少 7.4%**（长 prompt 下 prefill 占绝对主导）；
- **MTP decode 快 36.5%**，接受率高 20.5 个点；
- 256K 长 prompt + 短输出（报告/问答）选 DFlash2；长输出、多轮对话选 MTP；
- 两个启动器都不需要 `--no-cuda-graph`；prefill 只算第一遍，第二遍是 hidden replay
  （不计 token，但计入 wall）。

同模型不同量化的历史对照（B/R=77,824、nvfp4 KV、253K prompt，prefill 为两遍口径）：

| 草稿 | prefill | decode | wall |
|---|---:|---:|---:|
| q3blocks | 404 tok/s | 46.6 tok/s | 627 s |
| nvfp4full | 334 tok/s | 46.4 tok/s | 759 s |

q3blocks 比 nvfp4full prefill 快约 21%；q3km 与 q3blocks 同属 Q3 家族，作为默认推荐。

#### 128K 严谨对比（同条件、无 ngram）

126,668 token prompt，同一目标模型 q3km，`--kv-dtype nvfp4`，B/R=36,480/16,384，
`--ngram-draft-tokens 0`，只换草稿：

| 草稿 | 第一遍 prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| dflash2 + **q3km** | 904.8 tok/s（140 s） | 42.8 tok/s | 23.4%（137/585） | 282.8 s |
| dflash2 + **q4km** | 904.8 tok/s（140 s） | 43.6 tok/s | 24.2%（139/575） | 282.4 s |
| **MTP + q3km**（draft 3） | 904.8 tok/s（140 s） | **47.7 tok/s** | **40.8%**（140/343） | 280.9 s |

- **q3 与 q4 草稿几乎无差别**：decode ±1.9%，接受率 ±0.8 个点，q4 工件还多占约 0.18 GB
  → 推荐 q3km；
- **MTP decode 快 11.5%**、接受率高 17.4 个点；prefill 三组完全相同，说明 prefill
  由目标模型决定，换草稿不影响；
- 三组 wall 都在 281–283 s，其中约一半是 query replay；decode 只占 6 s，
  所以长 prompt 场景优先看 prefill。

> **两遍 prefill 的来源**：prompt 超过 KVMem 窗口、且 query 覆盖整段用户消息时，
> 引擎会在 prefill 末尾回卷到 `query_begin` 重跑一遍（`memory_query_probe`），用来算
> mean-Q 检索打分。单条巨型消息是最坏情况（第二遍≈整段）；多轮“长历史 + 短问题”
> 只重跑最后一轮。prefill 速度按第一遍计，wall 含两遍。

启动参数：

```bat
--max-context 262144 --prefill-chunk 256 --full-draft-head ^
--kvmem-budget 55296 --kvmem-gen-reserve 13312 --kvmem-host-mib 12288 --kvmem-lrfu ^
--kv-dtype nvfp4 --spec dflash2 --draft-tokens 5 --ngram-draft-tokens 31
```

（`--kvmem-budget`/`--kvmem-gen-reserve` 按显卡调整；B/R 不再要求小于 4。）

### 3. KVMem 驱逐回到满窗口 100%（issue #6）

80% 水位（`W - W/5`）在 `B > 4R` 时会让每个 256-token chunk 都空转重选一次。现在任何
B/R 都只在窗口满 `W = B + R` 时重选：

```cpp
const bool reselecting =
    force_selection || compact_position(sequence, end) > kvmem_window_tokens;
```

实测（B/R=4.15，61K prompt）：旧版 55 次 reselect / 82.2 s → 新版 **0 次 / 75.8 s**；
253K NIAH 仍 3/3。代价是不再有“提前交换与 decode 重叠”。

### 4. Jinja 模板 cache boundary

`trim_blocks` / `lstrip_blocks` / `{%- -%}` 折叠源码后，落在折叠区的 cache marker 无法
解析，prefix cache 命中下降。现在折叠边界跟随 `string_part` 传递，marker 偏移可唯一
解析。

详细数据：[docs/integration-20261025.md](docs/integration-20261025.md)、
[docs/kvmem-reselect-high-water-20261019.md](docs/kvmem-reselect-high-water-20261019.md)、
[docs/cuda-graph-calibration.md](docs/cuda-graph-calibration.md)。

## 2026-10-23 更新：DFlash2 NVFP4 草稿支持

ninfer 后端现在可以直接加载 **NVFP4 编码的 DFlash2 草稿**，`attention/output` 与
`mlp/down` 以 NVFP4 权重跑在动态卷积的 finish 路径上。

**DFlash2 草稿推荐地址**（ModelOpt NVFP4，原样导入、无损）：

<https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal/tree/main>

### 引擎改动

| commit | 内容 |
|---|---|
| `f4d18713` | NVFP4 线性表注册 DFlash2 的五个几何（Feature/QKV/AttnOut/ConvProj/Selector），own 与 unified 两条调度同时生效 |
| `18e3c7c8` | 动态卷积的 add 路径接受 NVFP4 父权重；补上 `dflash2_import_nvfp4.py` 与 `dflash2_nvfp4.py` 两个转换配方 |

约束（NVFP4 导入路径）：`kernel_projection` 仍必须是 BF16（prepare 路径没有 NVFP4
内核），QKV 仍是 Q8（融合 attention 输入只对 `4096x5120` 放行 NVFP4，本模型是
`6144x5120`）。如果走 **GGUF 直通**（`dflash2_gguf_blocks.py`），`attention/output`
与 `mlp/down` 保持源块类型（Q2_K/Q3_K/Q4_K），草稿不再混用 Q8/NVFP4，详见
[docs/dflash2-nvfp4-20261023.md](docs/dflash2-nvfp4-20261023.md) 的“直通草稿模式”。

### 256K 实测（RTX 5060 Ti 16 GB，nvfp4 KV）

条件：`--max-context 262144 --kvmem-budget 36864 --kvmem-gen-reserve 16384
--kv-dtype nvfp4`，prompt 257,089 token，输出 256，temperature 0。

| 后端 | draft | CUDA graphs | decode | 接受率 | prefill |
|---|---|---|---:|---:|---:|
| dflash2 | 4 | 关（当时） | **55.1 tok/s** | 44.8% | 419 tok/s |
| mtp | 3 | 开 | 50.2 tok/s | 48.7% | 432 tok/s |
| mtp | 3 | 关 | 47.7 tok/s | 48.7% | 431 tok/s |

DFlash2 权重比 MTP 多约 1.1 GiB。**2026-10-26 起不再需要 `--no-cuda-graph`**：探针校准
后开图也不会 OOM，见本文开头的更新。接受率约 44.8%，参数层面已基本榨干。DFlash2 比 MTP
开图快约 9.8%，比 MTP 关图快约 15.5%。

推荐参数（旧配置，保留作历史；新配置见 2026-10-26 节）：

```
--max-context 262144 --kvmem-budget 36864 --kvmem-gen-reserve 16384 \
--kv-dtype nvfp4 --spec dflash2 --draft-tokens 5 --full-draft-head
```

详细说明（转换用法、`.safetensors` 限制、参数规律）见
[docs/dflash2-nvfp4-20261023.md](docs/dflash2-nvfp4-20261023.md)。
可下载Releases内tool进行转换。

## 2026-10-22 更新：sm70 / sm75 源码支持

ninfer 后端新增 **sm75（RTX 20 系、TITAN RTX、T4）** 与 **sm70（Tesla V100）** 的构建支持。召回与驱逐保持 ActKV 不变。

### 改动范围

最小移植，不升级上游 210 个提交：

| 项 | 内容 |
|---|---|
| 新增文件 | 31 个（sm70 整套 GEMM、sm75 MMA、`pre_sm80.cuh` / `sm70_emulation.cuh` 兼容层） |
| 接入修改 | 8 处（CMake 白名单、`mma.cuh`、`memory.cuh`、sources、`QType`、2 个 codec） |
| GPU profile | 9 个（V100×2、TITAN RTX、2060S、2070、2070S、2080Ti、Quadro RTX 6000、T4） |
| ActKV | **未改动**：LRFU 公式、mean-Q 打分、sink/recent 逻辑与 10-21 版一致 |

### 构建

```bash
# RTX 20 系（sm75，CUDA 13.x 支持 compute_75）
cmake -S backends/ninfer -B build-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75 --target ninfer-serve

# V100（sm70，必须 CUDA 12.x；CUDA 13 已移除 Volta）
cmake -S backends/ninfer -B build-sm70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
cmake --build build-sm70 --target ninfer-serve
```

启动参数与现有配置一致，不需要 `--kvmem-query-scorer`（本地版本默认即 mean-Q cosine）。

### 验证

| 项 | 结果 |
|---|---|
| sm120a 编译 | 0 错误 |
| 64K | prefill 427.1 tok/s / decode 45.5 tok/s / 接受率 48.1% |
| 256K NIAH | 33/33、14/14、80 工具调用 |
| sm75 / sm70 运行 | **未验证**（本机只有 sm120a） |

### 块大小不可调

尝试把页大小从 64 改到 32，编译期被断言挡住：

```cpp
static_assert(kCausalPromptI8FastBc == kPagedKVPageSize);
```

页大小同时是 fast prompt 内核的 K 方向 tile，改它需要重写 MMA 迭代与共享内存预算。48 更不可行：页表索引是 `position >> 6`，要求 2 的幂。

### 文档

- [sm70/sm75 移植与块大小结论（2026-10-22）](docs/kvmem-sm75-sm70-port-20261022.md)
- [子模块 patch 恢复说明](docs/patches/README.md)

## 2026-10-21 更新：ActKV LRFU 与增量 restore，新版需自行git编译。

### 打分与驱逐

召回打分保持不变：64-token 块的 mean key 与 query 均值的余弦，prefill 结束后一次性计算。

驱逐改用 ActKV Algorithm 1（[arXiv 2609.31395](https://arxiv.org/pdf/2609.31395)）：

```
hit : KS_t = λ·KS_{t-1} + score
miss: KS_t = λ·KS_{t-1}
```

新块从 0 开始，首次命中时被当前分数点亮。与之前 EMA（`λ·KS + (1-λ)·score`）的差别在于：EMA 是加权平均，ActKV 是衰减累积和，后者让「偶尔命中但单次分高」的块活得更久。

### 增量 restore

`start_restore` 不再 `truncate(0)` 后全量重映射：

- 选中页按 id 稳定排序，同一逻辑页在重选后保持物理槽位；
- 只对变化的槽位做 H2D，也只读变化的 host 记录；
- 变化超过一半时自动回退全量，避免差异比全量还贵。

| 64K 场景 | 全量 | 增量 |
|---|---:|---:|
| restore 页数 | 576 | **124–181** |
| read | 69 ms | **14–21 ms** |
| sync | 94 ms | **20–29 ms** |
| 总耗时 | 163 ms | **33–51 ms** |

256K 时 `changed` 中位数 495/576，自动回退全量，行为与之前一致。

### 256K 召回（NIAH 33 针）

| 指标 | 结果 |
|---|---:|
| 同轮 / 终轮严格 / 并集 | 33/33 |
| 跨工具（send_alert） | 14/14 |
| 工具调用 | 80 |
| 端到端（到 final sweep） | 595.8 s |

### LongMemEval 上下文扫描

同一组实例（idx 0,1,2）、20 题、`What exactly did the user say` probe，对比 ActKV 与 EMA：

| 上下文 | ActKV | EMA |
|---|---:|---:|
| 128K | **20/20** | 19/20 |
| 192K | **18/20** | 16/20 |
| 240K | **16/20** | 15/20 |
| 合计 | **54/60** | 50/60 |

- **最佳工作区间：128K–192K**；
- 192K 开始掉分（ActKV 100% → 90%），240K 明显劣化（80%）；
- 240K 的 4 道失败里 3 道是模型拒答，不是 KV 淘汰。

### 旧更新日志

- 2026-10-18 DFlash2 × KVMem 修复：[docs/dflash2-kvmem-fixes-20261018.md](docs/dflash2-kvmem-fixes-20261018.md)
- 2026-10-19 召回传输优化（DMA 4.0 → 14.1 GB/s）：[docs/kvmem-recall-opt-20261019.md](docs/kvmem-recall-opt-20261019.md)
- 2026-10-19 重选水位提前到 80%：[docs/kvmem-reselect-high-water-20261019.md](docs/kvmem-reselect-high-water-20261019.md)

## GGUF → .ninfer 模型转换

转换工具可在Releases中下载（Python 3.11 + PyTorch/NumPy）。
`qwen3_8_27b_gguf` 配方把 Qwen3.8-27B GGUF **按自身块格式直接转换**（不做反量化），
可选加入 MTP 头、DFlash2 草稿与视觉投影器。在 `backends/ninfer` 目录下运行：

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --source dflash2=Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --source vision=mmproj-Qwen3.8-27B-Q8_0.gguf \
  --components text,vision,mtp,dflash2 \
  --name qwen3.8-27b \
  --out model.ninfer
```

- `--model` 只需要基础检查点的配置与分词器（准备转换输入时生成的 `metadata` 目录）。
- `dflash2` 源支持三种 safetensors 布局和 GGUF：ModelOpt NVFP4（推荐，原样导入）、
  BF16（重新量化）、per-row FP8（重新量化）。
- 不需要的组件可省略（例如纯文本去掉 `vision`）；`mtp` 组件要求 `-mtp` GGUF
  （`blk.64.nextn.*`）。
- CPU 转换追加 `--device cpu`。

### DFlash2 NVFP4 草稿

推荐使用 ModelOpt NVFP4 目录，并加 `--override` 走无损导入：

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --override tools/convert/recipes/dflash2_import_nvfp4.py \
  --components text,mtp,dflash2 \
  --device cpu \
  --name qwen3.8-27b-dflash2-nvfp4 \
  --out model.ninfer
```

| 源格式 | 特征 | 处理 |
|---|---|---|
| ModelOpt NVFP4 | `quant_algo=NVFP4`；`weight`(U8) + `weight_scale`(E4M3) + `weight_scale_2`(F32) | 原样导入 |
| BF16 | 普通 BF16 权重 | 重新量化为 NVFP4 |
| per-row FP8 | `quant_method=fp8`，每行一个 `weight_scale` | 重新量化为 NVFP4 |
| **128×128 block FP8** | `weight_block_size=[128,128]`，scale 名 `weight_scale_inv` | **不支持**，报 `missing ... weight_scale`，需先反量化成 BF16 |
| **GGUF NVFP4** | ggml type 40（64 元素/36 字节 super-block） | **不支持直通导入** |

> 推荐草稿地址：<https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal/tree/main>
>
> 详细说明见 [docs/dflash2-nvfp4-20261023.md](docs/dflash2-nvfp4-20261023.md)。

## 快速开始

1. 下载 `.ninfer` 模型（或按上文自行转换）。
2. 双击 `workers/ninfer/start-kvmem-ninfer-dflash.bat`（DFlash2）或
   `start-kvmem-ninfer.bat`（MTP）。
   - 模型路径、启动目录与参数都在 bat 文件里，直接编辑即可。
   - 显存 16 GiB 推荐：`--max-context 262144 --kvmem-budget 36864
     --kvmem-gen-reserve 16384 --kvmem-host-mib 12288`。
3. 服务地址 `http://127.0.0.1:18200`，OpenAI 兼容 API。


## 文档

- [DFlash2 NVFP4 草稿支持（2026-10-23）](docs/dflash2-nvfp4-20261023.md)
- [DFlash2 × KVMem 修复记录（2026-10-18）](docs/dflash2-kvmem-fixes-20261018.md)
- [召回传输优化：带宽基准、256K A/B 与逐次对比（2026-10-19）](docs/kvmem-recall-opt-20261019.md)
- [后端能力与启动说明](docs/multi-backend.md)
- [仓库结构与构建](docs/multi-backend-repository.md)
- [后端差异](docs/multi-backend-differences.md)
- [sm70/sm75 移植与块大小结论（2026-10-22）](docs/kvmem-sm75-sm70-port-20261022.md)
- [子模块 patch 恢复说明](docs/patches/README.md)

