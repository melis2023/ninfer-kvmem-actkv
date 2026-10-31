# KVMem + ActKV · ninfer 后端

基于 [kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) 与 **ActKV**
（[arXiv:2609.31395](https://arxiv.org/pdf/2609.31395)，中国科学技术大学提出）：

- **KVMem**：把长上下文稀疏成固定窗口（`budget` 历史块 B + `gen_reserve` 工作区 R），
  Host/磁盘归档其余 KV，用固定大小的注意力窗口换取 200K/262K 可用上下文。
- **ActKV**：面向行动的检索保持策略（LRFU 历史排序等），提升长会话关键块召回。

QQ 交流群：**1040777853**。本项目仅供研究学习，稳定使用请前往 kvmem 原作者。

**所有修补、测速与召回数据汇总在 [docs/CHANGELOG.md](docs/CHANGELOG.md)。**

## 当前推荐（2026-10-26）

| 项 | 推荐 |
|---|---|
| 工件 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km.ninfer` |
| 草稿 | **q3km 直通**（Q4_K×10 + Q3_K×2，conv/norm BF16） |
| 草稿后端 | **DFlash2**，`draft 4`，`ngram 0`（agent 可用 15） |
| KV | `nvfp4`，B/R = 36,480 / 16,384（W=52,864） |
| CUDA Graph | **开**；探针校准 256 MiB，不需要 `--no-cuda-graph` |
| 驱逐 | 满窗口 100% |

```bat
--max-context 262144 --prefill-chunk 256 ^
--kvmem-budget 36480 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 --kvmem-lrfu ^
--kv-dtype nvfp4 --spec dflash2 --draft-tokens 4 --ngram-draft-tokens 0 ^
--vision --vision-residency cpu --vision-max-merged 256 ^
--default-reasoning-effort medium --default-thinking-budget 16384
```

16 GB 卡启动 free 约 780–955 MiB；B/R 按显卡调整，不要再加 `--no-cuda-graph`。

## 性能

### 128K 代码（148,774 token，补函数）

| 后端 | prefill（第一遍） | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km，draft 4 | 884.6 | 71.5 tok/s | 82.1% | 340.8 s |
| **MTP + q3km，draft 3** | 887.0 | **76.3 tok/s** | **86.9%** | 340.8 s |

自然停止，输出 120 token。decode 只有约 1.7 s，样本短；用 `ignore_eos` 强制 512 token
复测：DFlash2 62.5 / 47.1%，MTP 64.7 / 70.0%，方向一致。**代码任务上两者接近，MTP 略快
3–7%**；旧表里的 DFlash2 87.1 来自更早的二进制，当前版本不再复现。

### 128K 总结（126,668 token）

| 后端 | prefill | decode | 接受率 | wall |
|---|---:|---:|---:|---:|
| DFlash2 + q3km，draft 4 | 904.8 | 45.8 | 28.3% | 288.5 s |
| **MTP + q3km，draft 3** | 904.8 | **47.7** | **40.8%** | 280.9 s |

### 256K（253,099 token，两个启动器）

| 启动器 | 草稿 | KV | prefill（第一遍） | decode | 接受率 | wall |
|---|---|---|---:|---:|---:|---:|
| dflash | **draft 4** | nvfp4 | **854.8** | **50.5** | **37.8%** | 597.4 s |
| mtp | draft 3 adaptive | int8 | 803.3 | 47.5 | 40.7% | 630.9 s |

DFlash2 行为 2026-10-26 复测（q3km、`ngram 0`、当前二进制）；MTP 行仍是旧启动器数据。
253K prompt、输出 256 token；prefill 报告值 427.9 tok/s 含隐藏 replay，第一遍按两倍折算。

### 鹈鹕 HTML draft 扫描（72 token prompt，ngram 0）

| draft | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| decode | 68.8 | **71.4** | 70.6 | 67.8 | 64.0 |
| 接受率 | **65.7%** | 54.9% | 48.4% | 42.4% | 38.1% |

### 短 prompt / agent

| 场景 | decode | 接受率 |
|---|---:|---:|
| WebUI 鹈鹕（2 工具，thinking 209） | **96.0 tok/s** | 76.7% |
| harness（38 工具，thinking 4,826） | 68.2 tok/s | 51.9% |
| agent 9.8K（ngram 0） | 87.8 tok/s | 70.8% |
| agent 9.8K（ngram 31 仲裁） | 89.2 tok/s | 68.1% |

### KV 精度

| | `nvfp4` | `rk8v4` |
|---|---:|---:|
| 52,864 窗口 runtime | **1.62 GiB** | 1.99 GiB |
| 启动 free | **835.8 MiB** | 645.9 MiB |
| 鹈鹕 decode（draft 4） | **73.2 tok/s** | 68.6 tok/s |
| 接受率 | **58.6%** | 51.3% |

`nvfp4` 是 16 GB 卡默认；`rk8v4` 的 key 保持 INT8，理论长上下文检索更稳，代价是多约
0.37 GiB、decode 慢约 6%。

### 召回

| 测试 | 结果 |
|---|---|
| 256K NIAH，33 针 | **33/33**（同轮 / 终轮严格 / 并集） |
| 256K 跨工具 `send_alert` | **14/14** |
| 256K 三针（5% / 50% / 95%） | **3/3** |
| 128K NIAH（5% / 50% / 95%） | **3/3** |
| LongMemEval 54 题 | **54/60**（ActKV） |
| Agent 220K | **30/30** 逐轮工具召回 |

## 更新日志（2026-10-31）

### 本轮改动

| 领域 | 改动 | 验证 |
|---|---|---|
| DFlash2 同步 | 记录 `pending_features` 的捕获基准；append 起点不符即拒绝，所有 frontier 改写点重置标记 | 192K 262 轮零违约 |
| 思考等级 | `tool` 消息可携带 `reasoning_effort`，顶层未指定时生效；pi-ai UA 省略 effort 解析为 off | none=0 / low=805 / xhigh=554 reasoning tokens |
| keep 动态 | 开 keep 后 sink 取 `[0, first_user)` 实际页（cap 10% B），recent 取最后 8 条消息实际页（cap 15% B）；recent 仅在带 tools 时启用 | 带 tools `sink=38 recent=5`；纯文本回退固定 recent |
| 图传播 | `graph_alpha` 默认 0.2；graph 未覆盖当前 frontier 时跳过传播 | 192K 20/20 |
| 日志 | `req# done` 行增加每位置接受率 `pos` | — |
| 投机内核 | coupled drafts accept；invariant attention 可选（默认关） | 数学等价，速度一致 |
| GPU | NVML 时钟下限 `[floor, max]` | 负载 2790 MHz < 3090 上限 |
| KVMem | 移植 0.19 checkpoint / speculative store | — |
| 构建 | FFMPEG 路径转义；tool schema 顺序稳定 | — |

### 192K LongMemEval 召回（20 题）

| 配置 | 召回 | write | prefill | recall decode |
|---|---|---:|---:|---:|
| rk8v4 B=41600，无 keep | 20/20 | 365 s | 530–750 | 91.3 |
| rk8v4 B=41600，keep 旧（固定 cap） | 20/20 | 344 s | — | 96.3 |
| k8v4 B=38400，keep 全动态 | 19/20 | 454 s | 434 | 84.7 |
| rk8v4 B=41600，keep 全动态 | 19/20 | 373 s | 538 | 89.6 |
| **rk8v4 B=41600，keep 动态（gated）** | **20/20** | **356 s** | **579** | **95.3** |

结论：动态 recent 只服务于带工具的多任务对话；纯文本长对话保持固定 recent，避免最近 8 条大消息挤占长召回所需的检索页。动态 sink 不受影响，始终按实际开头结构取页。

## 关键修补

完整原因、提交和验证数据见 [docs/CHANGELOG.md](docs/CHANGELOG.md)。

| 类别 | 结论 |
|---|---|
| KVMem 驱逐 | 满窗口 100%；61K 从 55 次重选降到 **0 次**，wall 82.2 → 75.8 s |
| KVMem 召回 | ActKV LRFU；LongMemEval 54/60 vs EMA 50/60 |
| KVMem 传输 | 差异 restore；64K 163 ms → **33–51 ms** |
| DFlash2 × KVMem | envelope/floor/slot 修复；越窗接受率 1.5% → **42%** |
| DFlash2 bypass | 按期望收益判定；误关草稿 **0 次** |
| ngram 仲裁 | copy 只在收益 > 1.3× 草稿时触发；鹈鹕 74.7 vs 72.1 tok/s |
| CUDA Graph | 探针校准；省 **700–800 MiB**，开图 +6% |
| NVFP4 | Blackwell 保留 scale `0x7F` → 0（仅编译验证） |
| 模板缓存 | Jinja trim 后 cache boundary 不再丢失 |
| 平台 | sm70/sm75 源码支持（未真机验证） |

## 模型转换

转换工具从 Releases 下载（Python 3.11 + PyTorch/NumPy）。在 `backends/ninfer` 下运行：

```bash
python3 -m tools.convert \
  --model /path/to/conversion-input/metadata \
  --recipe qwen3_8_27b_gguf \
  --source gguf=Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP.gguf \
  --source dflash2=Qwen3.8-27B-DFlash2-Q3_K_M.gguf \
  --source vision=mmproj-Qwen3.8-27B-Q8_0.gguf \
  --components text,vision,mtp,dflash2 \
  --name qwen3.8-27b-q3km \
  --out model.ninfer
```

- `dflash2` 源支持 GGUF、ModelOpt NVFP4、BF16、per-row FP8；
- 直通草稿用 `dflash2_gguf_blocks.py`（默认配方）；
- NVFP4 草稿走 `--override tools/convert/recipes/dflash2_import_nvfp4.py`，
  详见 [docs/dflash2-nvfp4-20261023.md](docs/dflash2-nvfp4-20261023.md)；
- `kernel_projection` 与 conv 必须保持 BF16（prepare 路径没有量化内核）。

## 快速开始

1. 下载 `.ninfer` 工件，或按上文自行转换。
2. 双击 `workers/ninfer/start-kvmem-ninfer-dflash.bat`（DFlash2）或
   `start-kvmem-ninfer.bat`（MTP）。模型路径和参数都在 bat 里。
3. 服务地址 `http://127.0.0.1:18200`，OpenAI 兼容 API。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | **所有修补、测速、召回总览** |
| [docs/dflash2-passthrough-20261026.md](docs/dflash2-passthrough-20261026.md) | q3km 直通、draft 扫描、bypass/floor 细节 |
| [docs/dflash2-kvmem-fixes-20261018.md](docs/dflash2-kvmem-fixes-20261018.md) | DFlash2 × KVMem 首次修复 |
| [docs/kvmem-reselect-high-water-20261019.md](docs/kvmem-reselect-high-water-20261019.md) | 驱逐水位与 issue #6 |
| [docs/kvmem-actkv-lrfu-20261020.md](docs/kvmem-actkv-lrfu-20261020.md) | ActKV LRFU 与 LongMemEval |
| [docs/cuda-graph-calibration.md](docs/cuda-graph-calibration.md) | Graph 探针与启动数据 |
| [docs/multi-backend.md](docs/multi-backend.md) | 后端能力与启动说明 |
| [docs/multi-backend-repository.md](docs/multi-backend-repository.md) | 仓库结构与构建 |
